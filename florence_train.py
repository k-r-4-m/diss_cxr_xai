"""
    Trains Florence-2 using the VinDr-CXR dataset

    Hyperparameters are set in the config.yaml file

    Requires:
        VinDr-CXR to have been downloaded and preprocessed (vindr_preprocessing_aug.py)
"""


import os
import torch
# need transformers version 4.53.2
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

print("RUNNING: FLORENCE TRAINING")

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
USE_PEFT = config.get('use_peft')
print("config loaded")

# collates samples to form a batch of tensors
# needed for dataloader
def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

CHECKPOINT = "microsoft/Florence-2-base-ft"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads the model and processor for florence-2
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

# builds the dataloader for the training set
train_dataset = DetectionDataset(
    jsonl_file_path = f"{OUTPUT_DIR}/train/annotations.jsonl",
    image_directory_path = f"{OUTPUT_DIR}/train/"
)

# builds the dataloader for the validation set
val_dataset = DetectionDataset(
    jsonl_file_path = f"{OUTPUT_DIR}/valid/annotations.jsonl",
    image_directory_path = f"{OUTPUT_DIR}/valid/"
)

# # builds the dataloader for the training set
# train_dataset = DetectionDataset(
#     jsonl_file_path = f"{OUTPUT_DIR}/train/annotations_caption_to_phrase.jsonl",
#     image_directory_path = f"{OUTPUT_DIR}/train/"
# )

# # builds the dataloader for the validation set
# val_dataset = DetectionDataset(
#     jsonl_file_path = f"{OUTPUT_DIR}/valid/annotations_caption_to_phrase.jsonl",
#     image_directory_path = f"{OUTPUT_DIR}/valid/"
# )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# if using peft to reduce number of trainable parameters
if USE_PEFT:
    print("using peft")
    # specifies the configuration for LoRa
    config = LoraConfig(
        r=config.get('lora_r'),
        lora_alpha=config.get('lora_alpha'),
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
        revision=REVISION
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

torch.cuda.empty_cache()

# saves the inference results as a jpg to track how the model is doing over time
def save_inference_results(model, dataset: DetectionDataset, count: int, save_dir: str, epoch="none"):
    os.makedirs(save_dir, exist_ok=True)
    count = min(count, len(dataset))

    for i in range(count):
        image, data = dataset.dataset[i]
        prefix = data['prefix']
        suffix = data['suffix']

        inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        answer = processor.post_process_generation(generated_text, task='<CAPTION_TO_PHRASE_GROUNDING>', image_size=image.size)

        try:
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, answer, resolution_wh=image.size)
            image_annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image.copy(), detections)
            image_annotated = sv.LabelAnnotator(text_scale=1, text_thickness=2, color_lookup=sv.ColorLookup.INDEX).annotate(image_annotated, detections)
            
            # saves example images of annotations
            filename = f"epoch{epoch}_{i:03d}.jpg".replace(" ", "_")
            filename = sanitize_filename(filename)   # cleans filename to make sure it gets saved
            filepath = os.path.join(save_dir, filename)
            image_annotated.save(filepath)
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Failed to annotate image {i}: {e}")

save_inference_results(model, val_dataset, 4, save_dir="./before_training")


## Fine-tune Florence-2 on custom object detection dataset

# defines the train loop for the model
def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # save_inference_results(model, val_loader.dataset, 6, save_dir="./inference_training")

    # runs model for given number of epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # for each input and answer in the train dataloader, get the ouputs and calculate the loss
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(DEVICE)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward(), optimizer.step(), lr_scheduler.step(), optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0

        # don't update the gradients when updating the weights 
        # would otherwise affect back propagation
        with torch.no_grad():
            # for each input and answer in the validation dataloader, get the ouputs and calculate the loss
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(DEVICE)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

            save_inference_results(model, val_loader.dataset, 6, save_dir="./training_images_annotated", epoch=str(epoch))

        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)



LR = 5e-6  # learning rate

# runs the training loop to fine tune the model
train_model(train_loader, val_loader, model, processor, epochs=EPOCHS, lr=LR)