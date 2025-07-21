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

DICOM_DIR = config.get('dicom_dir')
ANNOTATIONS_CSV = config.get('annotations_csv')
OUTPUT_DIR = config.get('output_dir')
print("config loaded")

IMAGE_EXT = ".png"
TARGET_LONG_SIDE = 1500

# gets the chest x-ray image from a DICOM file
def load_dicom_image(path, voi_lut=True, fix_monochrome=True):
    try:
        dicom = pydicom.dcmread(path)

        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

    except:
        raise ValueError(f"File at {path} is not a valid DICOM file.")

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    # DICOM files are greyscale, need to convert to RGB
    return Image.fromarray(data).convert("RGB")

# resizes an image but keeps the aspect ratio
def resize_image_keep_aspect(image, target_long_side=500):
    w, h = image.size
    if w >= h:
        scale = target_long_side / w
        new_size = (target_long_side, int(h * scale))
    else:
        scale = target_long_side / h
        new_size = (int(w * scale), target_long_side)
    return image.resize(new_size), scale

# normalises bounding boxes so that they are values between 0 and 1000 (florence-2 requirement)
def normalise_bbox(x, y, w, h, image_w, image_h):
    return [
        int((x / image_w) * 1000),
        int((y / image_h) * 1000),
        int(((x + w) / image_w) * 1000),
        int(((y + h) / image_h) * 1000)
    ]

# adds <loc> tags to the bounding boxes (florence-2 requirement)
def encode_suffix(class_name, bbox):
    x_min, y_min, x_max, y_max = bbox
    return f"{class_name}<loc_{x_min}><loc_{y_min}><loc_{x_max}><loc_{y_max}>"


df = pd.read_csv(ANNOTATIONS_CSV)
df = df.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])  # drop all rows with missing bounding boxes (i.e. no finding)
df['bbox'] = df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
df = df.drop(columns=['x_min', 'y_min', 'x_max', 'y_max'])

# group by image
grouped = df.groupby('image_id')

# train/val split
image_ids = df['image_id'].unique()
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
splits = {'train': train_ids, 'valid': val_ids}

# splits images into seperate folders and puts annotations into seperate jsonl files
for split_name, split_ids in splits.items():
    os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, split_name, "annotations.jsonl")

    with open(jsonl_path, "w") as f_out:
        for image_id in tqdm(split_ids, desc=f"Processing {split_name}"):
            image_path = os.path.join(DICOM_DIR, f"{image_id}.dicom")
            if not os.path.exists(image_path):
                continue

            image = load_dicom_image(image_path)
            original_w, original_h = image.size

            resized_image, scale = resize_image_keep_aspect(image, TARGET_LONG_SIDE)
            resized_w, resized_h = resized_image.size

            resized_path = os.path.join(OUTPUT_DIR, split_name, f"{image_id}{IMAGE_EXT}")
            resized_image.save(resized_path)

            try:
                entries = grouped.get_group(image_id)
            except KeyError:
                continue  # Image has no annotations

            # suffix_parts = []
            # for _, row in entries.iterrows():
            #     x_min, y_min, x_max, y_max = row['bbox']
            #     x_min *= scale
            #     y_min *= scale
            #     x_max *= scale
            #     y_max *= scale
            #     box_norm = normalise_bbox(
            #         x=x_min,
            #         y=y_min,
            #         w=x_max - x_min,
            #         h=y_max - y_min,
            #         image_w=resized_w,
            #         image_h=resized_h
            #     )
            #     suffix_parts.append(encode_suffix(row['class_name'], box_norm))

            ## trying no duplicate annotations
            suffix_parts = []
            seen_classes = set()  # uses a set to ensure no duplicate annotations
            for _, row in entries.iterrows():
                class_name = row['class_name']
                if class_name in seen_classes:
                    continue  # skip duplicate class for this image
                seen_classes.add(class_name)

                x_min, y_min, x_max, y_max = row['bbox']
                x_min *= scale
                y_min *= scale
                x_max *= scale
                y_max *= scale
                box_norm = normalise_bbox(
                    x=x_min,
                    y=y_min,
                    w=x_max - x_min,
                    h=y_max - y_min,
                    image_w=resized_w,
                    image_h=resized_h
                )
                suffix_parts.append(encode_suffix(class_name, box_norm))

            json_entry = {
                "image": f"{image_id}{IMAGE_EXT}",
                "prefix": "<OD>",
                "suffix": " ".join(suffix_parts)
            }
            f_out.write(json.dumps(json_entry) + "\n")
