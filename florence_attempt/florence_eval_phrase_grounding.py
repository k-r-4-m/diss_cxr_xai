import os
import torch
# need transformers version>=4.53.1
from transformers import get_scheduler, AutoModelForCausalLM, AutoProcessor, AutoConfig  
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torchvision.transforms.functional import to_pil_image
# from IPython.display import display
from pydicom.pixel_data_handlers.util import apply_voi_lut
from difflib import get_close_matches
from typing import List, Dict, Any, Tuple, Generator
from PIL import Image
from tqdm import tqdm
# from IPython.core.display import HTML
from datetime import datetime
from pathvalidate import sanitize_filename
from collections import defaultdict
from  supervision.detection.utils import box_iou_batch
from florence_tools import *
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torchvision.transforms as T
import supervision as sv
import yaml
import pydicom
import re
import json
import html
import base64
import itertools

# loads the config file for epochs, revision, pathnames, etc.
config_path = "./config.yaml"
config = load_config(config_path)

EPOCHS = config.get('epochs')
REVISION = config.get('revision')
DICOM_DIR = config.get('dicom_dir')
ANNOTATIONS_CSV = config.get('annotations_csv')
OUTPUT_DIR = config.get('output_dir')
BATCH_SIZE = config.get('batch_size')
NUM_WORKERS = config.get('num_workers')
print("config loaded")

# collates samples to form a batch of tensors
# needed for dataloader
def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

# builds the dataloader for the validation set
val_dataset = DetectionDataset(
    # jsonl_file_path = f"{OUTPUT_DIR}/valid/annotations.jsonl",
    jsonl_file_path = f"{OUTPUT_DIR}/valid/annotations_caption_to_phrase.jsonl",
    image_directory_path = f"{OUTPUT_DIR}/valid/"
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)


## Fine-tuned model evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = f"./model_checkpoints/epoch_{EPOCHS}"  # gets the last checkpoint of the model training

config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)
config.vision_config.model_type = 'davit'
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, config=config, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

# gets a list of all the classes in the dataset
df = pd.read_csv(ANNOTATIONS_CSV)
CLASSES = df['class_name'].unique().tolist()
CLASSES.remove("No finding")  # removes no finding from the classes list

targets = []
predictions = []

# helper function for Florence-2 inference
def run_florence_task(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(DEVICE, torch.float32)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
        output_scores=True,
        return_dict_in_generate=True
    )

    generated_text = processor.batch_decode(
        generated_ids.sequences,
        skip_special_tokens=False
    )[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=image.size
    )
    return parsed_answer

def clean_suffix(suffix_with_expl):
    # Split by '.' assuming explanations end with period, then keep only first part with loc tags
    # This can be fragile if explanations contain dots; better to split by regex for loc tags

    # Regex to extract all class + loc tag groups ignoring explanations:
    # Pattern: one or more word characters/spaces + four <loc_###> tags
    matches = re.findall(r'([A-Za-z\s]+(?:<loc_\d+>){4})', suffix_with_expl)
    # Join with space, this forms a clean suffix for Florence
    clean = " ".join(m.strip() for m in matches)
    return clean


targets = []
predictions = []

# evaluates model on the validation split
# for i in range(len(val_dataset.dataset)):
for i in range(10):
    image, data = val_dataset.dataset[i]
    prefix = data['prefix']
    suffix = data['suffix']  # suffix is the ground truth annotations

    # STEP 1: Get detailed caption
    # detailed_caption_result = run_florence_task("<DETAILED_CAPTION>", image)
    # detailed_caption_text = detailed_caption_result["<DETAILED_CAPTION>"]
    detailed_caption_result = run_florence_task("<MORE_DETAILED_CAPTION>", image)
    detailed_caption_text = detailed_caption_result["<MORE_DETAILED_CAPTION>"]

    # STEP 2: Grounding based on caption
    # grounding_result = run_florence_task("<CAPTION_TO_PHRASE_GROUNDING>", image, detailed_caption_text)
    grounding_result = run_florence_task("<OD>", image)

    # Convert grounding output to Detections object
    prediction = sv.Detections.from_lmm(
        sv.LMM.FLORENCE_2,
        grounding_result,
        resolution_wh=image.size
    )

    print(f"Pred class name: {prediction['class_name']}")

    # Fuzzy match predicted classes to known classes
    fuzzy_class_names = []
    for pred_class in prediction['class_name']:
        match = get_close_matches(pred_class, CLASSES, n=1, cutoff=0.5)
        if match:
            fuzzy_class_names.append(match[0])
        else:
            fuzzy_class_names.append(None)

    # Keep only matched predictions
    valid_indices = [idx for idx, name in enumerate(fuzzy_class_names) if name is not None]
    prediction = prediction[valid_indices]
    fuzzy_class_names = [fuzzy_class_names[idx] for idx in valid_indices]

    # Assign corrected class IDs and confidence
    prediction.data['class_name'] = np.array(fuzzy_class_names)
    prediction.class_id = np.array([CLASSES.index(name) for name in fuzzy_class_names])
    prediction.confidence = np.ones(len(prediction))

    # Ground truth conversion
    cleaned_suffix = clean_suffix(suffix)
    target = processor.post_process_generation(
        cleaned_suffix,
        task="<OD>",
        image_size=image.size)
    target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
    target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])

    # Store predictions and targets
    targets.append(target)
    predictions.append(prediction)

    print(f"\nImage {i+1}/{len(val_dataset.dataset)}")
    print(f"Detailed caption: {detailed_caption_text}")
    print(f"Target: {target}")
    print(f"Prediction: {prediction}")
    print(f"Original caption: {suffix}")