import os
import torch
# need transformers version 4.53.1
from transformers import get_scheduler, AutoModelForCausalLM, AutoProcessor, AutoConfig  
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


EPOCHS = 50
REVISION = 'refs/pr/24' 
DICOM_DIR = "E:/vinbigdata_xrays/vinbigdata/train/dicom"
ANNOTATIONS_CSV = "E:/vinbigdata_xrays/vinbigdata/train_original.csv"
OUTPUT_DIR = "E:/vinbigdata_xrays/output_dataset/"
BATCH_SIZE = 6  # batch size for training
NUM_WORKERS = 0  # number of workers for data loading

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

# makes a dataset given a jsonl file
class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")


# forms a dataloader for a given dataset
class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data['prefix']  # prefix is the task
        suffix = data['suffix']  # suffix is the annotations
        return prefix, suffix, image


# builds the dataloader for the validation set
val_dataset = DetectionDataset(
    jsonl_file_path = f"{OUTPUT_DIR}/valid/annotations.jsonl",
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

# # regex pattern to grab only the class names from the suffixes of annotations
# # e.g. turn "Cardiomegaly<loc_286><loc_445><loc_725><loc_789>" into "Cardiomegaly"
# PATTERN = r'(.*?)<loc_\d+>'

# # extracts the class names from the annotations jsonl file
# def extract_classes(dataset: DetectionDataset):
#     class_set = set()
#     for i in range(len(dataset.dataset)):
#         image, data = dataset.dataset[i]
#         suffix = data["suffix"]
#         classes = re.findall(PATTERN, suffix)
#         class_set.update(classes)
#     return sorted(class_set)

# CLASSES = extract_classes(train_dataset)

# gets a list of all the classes in the dataset
df = pd.read_csv(ANNOTATIONS_CSV)
CLASSES = df['class_name'].unique().tolist()
CLASSES.remove("No finding")  # removes no finding from the classes list

targets = []
predictions = []

# evaluates model on the validation split
for i in range(len(val_dataset.dataset)):
    image, data = val_dataset.dataset[i]
    prefix = data['prefix']  # prefix is the task
    suffix = data['suffix']  # suffix is the annotations

    # gets the output from florence-2
    inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # builds a Detections object for this prediction
    prediction = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
    prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)

    # prediction = prediction[np.isin(prediction['class_name'], CLASSES)]
    # prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction['class_name']])
    # prediction.confidence = np.ones(len(prediction))

    # uses get_close_matches to fuzzy match predicted class names to known classes
    # this is because florence-2 often misspells class names
    fuzzy_class_names = []
    for pred_class in prediction['class_name']:
        match = get_close_matches(pred_class, CLASSES, n=1, cutoff=0.75)
        if match:
            fuzzy_class_names.append(match[0])
        else:
            fuzzy_class_names.append(None) 

    # filters out unmatched predictions
    valid_indices = [i for i, name in enumerate(fuzzy_class_names) if name is not None]
    prediction = prediction[valid_indices]
    fuzzy_class_names = [fuzzy_class_names[i] for i in valid_indices]

    # assigns corrected class ids
    prediction.data['class_name'] = np.array(fuzzy_class_names)
    prediction.class_id = np.array([CLASSES.index(name) for name in fuzzy_class_names])
    prediction.confidence = np.ones(len(prediction))

    target = processor.post_process_generation(suffix, task='<OD>', image_size=image.size)
    target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
    target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])

    targets.append(target)
    predictions.append(prediction)

    print(f"Target {target}")
    print(f"pred {prediction}")


# calculates mAP scores
mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets,
)

print(f"map50_95: {mean_average_precision.map50_95:.2f}")
print(f"map50: {mean_average_precision.map50:.2f}")
print(f"map75: {mean_average_precision.map75:.2f}")

# calculates and outputs the confusion matrix
confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=CLASSES
)

# saves the confusion matrix as an image
fig = confusion_matrix.plot()
fig.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
