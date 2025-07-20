import os
import torch
from transformers import get_scheduler, AutoModelForCausalLM, AutoProcessor, AutoConfig  
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_pil_image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from difflib import get_close_matches
from typing import List, Dict, Any, Tuple, Generator
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
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
import random
from collections import Counter
import cv2

# loads the config file for epochs, revision, pathnames, etc.
config_path = "./config.yaml"
config = load_config(config_path)

DICOM_DIR = config.get('dicom_dir')
ANNOTATIONS_CSV = config.get('annotations_csv')
OUTPUT_DIR = config.get('output_dir')
print("config loaded")

IMAGE_EXT = ".png"
TARGET_LONG_SIDE = 1500

# config for augmentation
AUGMENTATION_CONFIG = {
    'brightness_range': (0.7, 1.3),
    'contrast_range': (0.7, 1.3),
    'zoom_range': (0.5, 1.5),
}

# Target samples per class (set this based on your needs)
TARGET_SAMPLES_PER_CLASS = 7162

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

# applies a zoom transformation to a given set of bounding boxes 
def apply_zoom_to_bbox(bbox, zoom_factor, image_center):
    x_min, y_min, x_max, y_max = bbox
    
    # convert to center format (from min/max)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    box_w = (x_max - x_min) * zoom_factor
    box_h = (y_max - y_min) * zoom_factor
    
    # scale relative to image center
    cx_rel = (cx - image_center[0]) * zoom_factor + image_center[0]
    cy_rel = (cy - image_center[1]) * zoom_factor + image_center[1]
    
    # convert back to min/max format
    new_x_min = cx_rel - box_w/2
    new_y_min = cy_rel - box_h/2
    new_x_max = cx_rel + box_w/2
    new_y_max = cy_rel + box_h/2
    
    return [new_x_min, new_y_min, new_x_max, new_y_max]

def augment_image_and_bboxes(image, bboxes, config=AUGMENTATION_CONFIG):
    """
    Apply augmentations to image and corresponding bounding boxes
    Returns augmented image and transformed bounding boxes
    """
    augmented_image = image.copy()
    augmented_bboxes = bboxes.copy()
    
    w, h = image.size
    image_center = (w/2, h/2)
    
    # brightness
    if config['brightness_range']:
        brightness_factor = random.uniform(*config['brightness_range'])
        enhancer = ImageEnhance.Brightness(augmented_image)
        augmented_image = enhancer.enhance(brightness_factor)
    
    # contrast
    if config['contrast_range']:
        contrast_factor = random.uniform(*config['contrast_range'])
        enhancer = ImageEnhance.Contrast(augmented_image)
        augmented_image = enhancer.enhance(contrast_factor)
    
    # zoom
    if config['zoom_range']:
        zoom_factor = random.uniform(*config['zoom_range'])
        if zoom_factor != 1.0:
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            augmented_image = augmented_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # crop or pad to the original size
            if zoom_factor > 1.0:  # crop
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                augmented_image = augmented_image.crop((left, top, left + w, top + h))
            else:  # padding
                pad_w = (w - new_w) // 2
                pad_h = (h - new_h) // 2
                new_image = Image.new('RGB', (w, h), (0, 0, 0))
                new_image.paste(augmented_image, (pad_w, pad_h))
                augmented_image = new_image
            
            # transforms bounding boxes
            augmented_bboxes = [apply_zoom_to_bbox(bbox, zoom_factor, image_center) 
                              for bbox in augmented_bboxes]
    
    return augmented_image, augmented_bboxes

# calculates the distribution of classes in the dataset
def calculate_class_distribution(df):
    class_counts = df['class_name'].value_counts()
    print("Current class distribution: ")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    return class_counts

# calculates how many augmented samples that are needed for each class
def calculate_augmentation_needs(class_counts, target_samples=TARGET_SAMPLES_PER_CLASS):
    augmentation_needs = {}
    for class_name, current_count in class_counts.items():
        if current_count < target_samples:
            augmentation_needs[class_name] = target_samples - current_count
    return augmentation_needs


### run the code, pre-processing images with augmentation ###
# loads and prepares data
df = pd.read_csv(ANNOTATIONS_CSV)
df = df.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])  # drop all rows with missing bounding boxes (i.e. no finding)
df['bbox'] = df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
df = df.drop(columns=['x_min', 'y_min', 'x_max', 'y_max'])

# analyses class distribution
class_counts = calculate_class_distribution(df)
augmentation_needs = calculate_augmentation_needs(class_counts)

print("\nAugmentation plan:")
for class_name, needed in augmentation_needs.items():
    print(f"  {class_name}: +{needed} samples")

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
    
    # track augmentation counts for this split
    split_class_counts = {}
    split_augmentation_tracker = {class_name: 0 for class_name in augmentation_needs.keys()}
    
    with open(jsonl_path, "w") as f_out:
        # makes a first pass and processes original images
        print(f"\nProcessing original {split_name} images...")
        for image_id in tqdm(split_ids, desc=f"Original {split_name}"):
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
                continue
            
            # track class counts in this split
            for _, row in entries.iterrows():
                class_name = row['class_name']
                split_class_counts[class_name] = split_class_counts.get(class_name, 0) + 1
            
            # process bounding boxes
            suffix_parts = []
            bboxes_for_aug = []
            class_names_for_aug = []
            
            for _, row in entries.iterrows():
                x_min, y_min, x_max, y_max = row['bbox']
                x_min *= scale
                y_min *= scale
                x_max *= scale
                y_max *= scale
                
                bbox_scaled = [x_min, y_min, x_max, y_max]
                bboxes_for_aug.append(bbox_scaled)
                class_names_for_aug.append(row['class_name'])
                
                box_norm = normalise_bbox(
                    x=x_min,
                    y=y_min,
                    w=x_max - x_min,
                    h=y_max - y_min,
                    image_w=resized_w,
                    image_h=resized_h
                )
                suffix_parts.append(encode_suffix(row['class_name'], box_norm))
            
            json_entry = {
                "image": f"{image_id}{IMAGE_EXT}",
                "prefix": "<OD>",
                "suffix": " ".join(suffix_parts)
            }
            f_out.write(json.dumps(json_entry) + "\n")
        
        # makes a second pass, generating augmented images
        # done only for training split
        if split_name == 'train':
            print(f"\nGenerating augmented {split_name} images...")
            
            # create list of images that contain underrepresented classes
            augmentation_candidates = []
            for image_id in split_ids:
                try:
                    entries = grouped.get_group(image_id)
                    for _, row in entries.iterrows():
                        if row['class_name'] in augmentation_needs:
                            augmentation_candidates.append(image_id)
                            break
                except KeyError:
                    continue
            
            augmentation_candidates = list(set(augmentation_candidates))  # remove duplicates
            
            aug_counter = 0
            while any(split_augmentation_tracker[cls] < augmentation_needs[cls] 
                        for cls in augmentation_needs.keys()):
                
                for image_id in augmentation_candidates:
                    image_path = os.path.join(DICOM_DIR, f"{image_id}.dicom")
                    if not os.path.exists(image_path):
                        continue
                    
                    try:
                        entries = grouped.get_group(image_id)
                    except KeyError:
                        continue
                    
                    # check if this image contains classes that still need augmentation
                    needs_aug = False
                    for _, row in entries.iterrows():
                        class_name = row['class_name']
                        if (class_name in augmentation_needs and 
                            split_augmentation_tracker[class_name] < augmentation_needs[class_name]):
                            needs_aug = True
                            break
                    
                    if not needs_aug:
                        continue
                    
                    # load and process image
                    image = load_dicom_image(image_path)
                    resized_image, scale = resize_image_keep_aspect(image, TARGET_LONG_SIDE)
                    resized_w, resized_h = resized_image.size
                    
                    # prepare bounding boxes for augmentation
                    bboxes_for_aug = []
                    class_names_for_aug = []
                    for _, row in entries.iterrows():
                        x_min, y_min, x_max, y_max = row['bbox']
                        x_min *= scale
                        y_min *= scale
                        x_max *= scale
                        y_max *= scale
                        bboxes_for_aug.append([x_min, y_min, x_max, y_max])
                        class_names_for_aug.append(row['class_name'])
                    
                    # apply augmentation
                    aug_image, aug_bboxes = augment_image_and_bboxes(
                        resized_image, bboxes_for_aug, AUGMENTATION_CONFIG
                    )
                    
                    # save augmented image
                    aug_counter += 1
                    aug_image_name = f"{image_id}_aug_{aug_counter:04d}{IMAGE_EXT}"
                    aug_image_path = os.path.join(OUTPUT_DIR, split_name, aug_image_name)
                    aug_image.save(aug_image_path)
                    
                    # process augmented bounding boxes
                    suffix_parts = []
                    for i, (bbox, class_name) in enumerate(zip(aug_bboxes, class_names_for_aug)):
                        x_min, y_min, x_max, y_max = bbox
                        
                        # clamp bounding boxes to image bounds
                        x_min = max(0, min(x_min, resized_w))
                        y_min = max(0, min(y_min, resized_h))
                        x_max = max(x_min, min(x_max, resized_w))
                        y_max = max(y_min, min(y_max, resized_h))
                        
                        # skip invalid bounding boxes
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        
                        box_norm = normalise_bbox(
                            x=x_min,
                            y=y_min,
                            w=x_max - x_min,
                            h=y_max - y_min,
                            image_w=resized_w,
                            image_h=resized_h
                        )
                        suffix_parts.append(encode_suffix(class_name, box_norm))
                        
                        # update augmentation tracker
                        if class_name in split_augmentation_tracker:
                            split_augmentation_tracker[class_name] += 1
                    
                    # write augmented annotation
                    json_entry = {
                        "image": aug_image_name,
                        "prefix": "<OD>",
                        "suffix": " ".join(suffix_parts)
                    }
                    f_out.write(json.dumps(json_entry) + "\n")
                    
                    # break if we've generated enough augmented samples
                    if all(split_augmentation_tracker[cls] >= augmentation_needs[cls] 
                            for cls in augmentation_needs.keys()):
                        break
                
                # safety break to prevent infinite loop
                if aug_counter > len(split_ids) * 10:
                    print(f"Warning: Reached maximum augmentation attempts for {split_name}")
                    break
            
            print(f"Generated {aug_counter} augmented samples for {split_name}")
            print("Final augmentation counts:")
            for class_name, count in split_augmentation_tracker.items():
                print(f"  {class_name}: +{count}")
