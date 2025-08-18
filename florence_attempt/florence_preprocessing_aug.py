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
    'zoom_range': (1.1, 1.5),
}

# target number of samples for each class
# set to the current majority class (aortic enlargement with 3057 samples)
TARGET_SAMPLES_PER_CLASS = 3057

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

# applies augmentations to an image and the corresponding bounding boxes
def augment_image_and_bboxes(image, bboxes, config=AUGMENTATION_CONFIG):
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

# calculates the IoU of two bounding boxes
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # calc intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # calc union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# applies majority voting for dupe/overlapping annotations
# consoldiates the annotations to be the mean bounding box
def apply_majority_voting(annotations, iou_threshold=0.5):
    if not annotations:
        return []
    
    # group annotations by class name
    class_groups = {}
    for ann in annotations:
        class_name = ann['class_name']
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(ann)
    
    consolidated = []
    
    for class_name, class_annotations in class_groups.items():
        if len(class_annotations) == 1:
            # only one annotation for this class, keep it
            consolidated.append(class_annotations[0])
        else:
            # multiple annotations for same class
            # apply majority voting
            clusters = []
            
            for ann in class_annotations:
                bbox = ann['bbox']
                placed = False
                
                # try to place in existing cluster
                for cluster in clusters:
                    # check if the annotation overlaps enough with a cluster given the IoU threshold
                    cluster_ious = [calculate_iou(bbox, existing_ann['bbox']) 
                                  for existing_ann in cluster]
                    if any(iou >= iou_threshold for iou in cluster_ious):
                        cluster.append(ann)
                        placed = True
                        break
                
                if not placed:
                    # creates a new cluster
                    clusters.append([ann])
            
            # for each cluster, create a consoldiated annotation using the mean bounding boxes
            for cluster in clusters:
                if len(cluster) == 1:
                    consolidated.append(cluster[0])
                else:
                    # calc mean bounding box
                    mean_bbox = [
                        np.mean([ann['bbox'][0] for ann in cluster]),  # x_min
                        np.mean([ann['bbox'][1] for ann in cluster]),  # y_min
                        np.mean([ann['bbox'][2] for ann in cluster]),  # x_max
                        np.mean([ann['bbox'][3] for ann in cluster])   # y_max
                    ]
                    
                    consolidated_ann = {
                        'class_name': class_name,
                        'bbox': mean_bbox
                    }
                    consolidated.append(consolidated_ann)
    
    return consolidated

# processes all annotations for a single image using the majority voting
def process_image_annotations(entries):
    # convert dataframe rows to annotation dicts
    annotations = []
    for _, row in entries.iterrows():
        annotations.append({
            'class_name': row['class_name'],
            'bbox': row['bbox']
        })
    
    # apply majority voting
    consolidated_annotations = apply_majority_voting(annotations)
    
    return consolidated_annotations

# calculates the distribution of classes in the dataset
def calculate_class_distribution(df):
    # group by image and apply majority voting to get actual distribution
    grouped = df.groupby('image_id')
    class_counts = Counter()
    
    for image_id, entries in grouped:
        consolidated = process_image_annotations(entries)
        for ann in consolidated:
            class_counts[ann['class_name']] += 1
    
    print("Current class distribution (after majority voting):")
    for class_name, count in class_counts.most_common():
        print(f"  {class_name}: {count}")
    return dict(class_counts)

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

# analyses class distribution before train/val split
print("Global class distribution: ")
class_counts = calculate_class_distribution(df)

# uses the majority class
TARGET_SAMPLES_PER_CLASS = max(class_counts.values())  # should be 4046
print(f"Target samples per class: {TARGET_SAMPLES_PER_CLASS}")

grouped = df.groupby('image_id')

# train/val split
image_ids = df['image_id'].unique()
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
splits = {'train': train_ids, 'valid': val_ids}

# calculates class distribution per split
split_class_counts = {'train': {}, 'valid': {}}
for split_name, split_ids in splits.items():
    print(f"\n {split_name} class distribution")
    for image_id in split_ids:
        try:
            entries = grouped.get_group(image_id)
            consolidated = process_image_annotations(entries)
            for ann in consolidated:
                class_name = ann['class_name']
                split_class_counts[split_name][class_name] = split_class_counts[split_name].get(class_name, 0) + 1
        except KeyError:
            continue
    
    print(f"{split_name.upper()} class counts:")
    for class_name, count in sorted(split_class_counts[split_name].items()):
        print(f"  {class_name}: {count}")

# calculates augmentation needs based on training set
train_augmentation_needs = {}
for class_name, current_count in split_class_counts['train'].items():
    target_for_train = int(TARGET_SAMPLES_PER_CLASS * 0.8)  # 80% of target goes to train
    if current_count < target_for_train:
        train_augmentation_needs[class_name] = target_for_train - current_count

print(f"\n Aug plan for training set (target: {int(TARGET_SAMPLES_PER_CLASS * 0.8)}):")
for class_name, needed in train_augmentation_needs.items():
    print(f"  {class_name}: +{needed} samples")

# track final class counts across all splits
final_class_counts = {'train': {}, 'valid': {}}

for split_name, split_ids in splits.items():
    os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)
    jsonl_path = os.path.join(OUTPUT_DIR, split_name, "annotations.jsonl")
    
    # initialize split class counts
    final_class_counts[split_name] = split_class_counts[split_name].copy()
    
    with open(jsonl_path, "w") as f_out:
        # makes a first pass and processes the original images
        print(f"\nGoing through original {split_name} images")
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
            
            # process annotations with majority voting
            consolidated_annotations = process_image_annotations(entries)
            
            if not consolidated_annotations:
                continue
            
            # process consolidated bounding boxes
            suffix_parts = []
            
            for ann in consolidated_annotations:
                class_name = ann['class_name']
                x_min, y_min, x_max, y_max = ann['bbox']
                
                # apply scaling
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
        
        # makes a second pass, generating augmented images
        if split_name == 'train' and train_augmentation_needs:
            print(f"\nGenerating augmented {split_name} images")
            
            # creates weighted list of images for augmentation based on class needs
            augmentation_candidates = []
            candidate_weights = {}
            
            for image_id in split_ids:
                try:
                    entries = grouped.get_group(image_id)
                    consolidated = process_image_annotations(entries)
                    
                    # calculate weight based on how many underrepresented classes this image contains
                    weight = 0
                    classes_in_image = []
                    for ann in consolidated:
                        class_name = ann['class_name']
                        if class_name in train_augmentation_needs:
                            weight += train_augmentation_needs[class_name]
                            classes_in_image.append(class_name)
                    
                    if weight > 0:
                        augmentation_candidates.append(image_id)
                        candidate_weights[image_id] = (weight, classes_in_image)
                        
                except KeyError:
                    continue
            
            # sort candidates by weight (prioritize images with most needed classes)
            augmentation_candidates.sort(key=lambda x: candidate_weights[x][0], reverse=True)
            
            print(f"Found {len(augmentation_candidates)} candidate images for augmentation")
            
            # track how many samples we still need per class
            remaining_needs = train_augmentation_needs.copy()
            aug_counter = 0
            max_attempts = len(augmentation_candidates) * 20  # prevents infinite loops
            attempt = 0
            
            while any(remaining_needs.values()) and attempt < max_attempts:
                for image_id in augmentation_candidates:
                    if not any(remaining_needs.values()):
                        break
                    
                    attempt += 1
                    
                    # check if this image still has classes we need
                    _, classes_in_image = candidate_weights[image_id]
                    if not any(remaining_needs.get(cls, 0) > 0 for cls in classes_in_image):
                        continue
                    
                    image_path = os.path.join(DICOM_DIR, f"{image_id}.dicom")
                    if not os.path.exists(image_path):
                        continue
                    
                    try:
                        entries = grouped.get_group(image_id)
                        consolidated_annotations = process_image_annotations(entries)
                    except KeyError:
                        continue
                    
                    # load and process image
                    image = load_dicom_image(image_path)
                    resized_image, scale = resize_image_keep_aspect(image, TARGET_LONG_SIDE)
                    resized_w, resized_h = resized_image.size
                    
                    # prepares bounding boxes for augmentation
                    bboxes_for_aug = []
                    class_names_for_aug = []
                    
                    for ann in consolidated_annotations:
                        x_min, y_min, x_max, y_max = ann['bbox']
                        x_min *= scale
                        y_min *= scale
                        x_max *= scale
                        y_max *= scale
                        bboxes_for_aug.append([x_min, y_min, x_max, y_max])
                        class_names_for_aug.append(ann['class_name'])

                    # applies augmentation
                    aug_image, aug_bboxes = augment_image_and_bboxes(
                        resized_image, bboxes_for_aug, AUGMENTATION_CONFIG
                    )
                    
                    # saves augmented image
                    aug_counter += 1
                    aug_image_name = f"{image_id}_aug_{aug_counter:04d}{IMAGE_EXT}"
                    aug_image_path = os.path.join(OUTPUT_DIR, split_name, aug_image_name)
                    aug_image.save(aug_image_path)
                    
                    # processes augmented bounding boxes and update counts
                    suffix_parts = []
                    classes_added_this_aug = []
                    
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
                        classes_added_this_aug.append(class_name)
                    
                    # write augmented annotation
                    if suffix_parts:
                        json_entry = {
                            "image": aug_image_name,
                            "prefix": "<OD>",
                            "suffix": " ".join(suffix_parts)
                        }
                        f_out.write(json.dumps(json_entry) + "\n")
                        
                        # update remaining needs and final counts
                        for class_name in classes_added_this_aug:
                            if class_name in remaining_needs and remaining_needs[class_name] > 0:
                                remaining_needs[class_name] -= 1
                                final_class_counts[split_name][class_name] = final_class_counts[split_name].get(class_name, 0) + 1
                
                if attempt >= max_attempts:
                    print(f"Warning!!! Reached maximum augmentation attempts ({max_attempts})")
                    break
            
            print(f"Generated {aug_counter} augmented samples for {split_name}")
            print("Final augmentation results:")
            for class_name, needed in train_augmentation_needs.items():
                achieved = needed - remaining_needs.get(class_name, 0)
                print(f"  {class_name}: {achieved}/{needed} samples added")


print("\n" + "="*50)
print("FINAL CLASS DISTRIBUTION SUMMARY")
print("="*50)

for split_name in ['train', 'valid']:
    print(f"\n{split_name.upper()} SET:")
    split_counts = final_class_counts[split_name]
    if split_counts:
        total_samples = sum(split_counts.values())
        print(f"Total images with annotations: {total_samples}")
        for class_name, count in sorted(split_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    else:
        print("  No annotations found")

# Combined statistics
all_classes = set()
for split_counts in final_class_counts.values():
    all_classes.update(split_counts.keys())

print(f"\nCOMBINED STATS:")
combined_counts = {}
for class_name in all_classes:
    total = sum(split_counts.get(class_name, 0) for split_counts in final_class_counts.values())
    combined_counts[class_name] = total

total_combined = sum(combined_counts.values())
print(f"Total images with annotations: {total_combined}")
for class_name, count in sorted(combined_counts.items()):
    percentage = (count / total_combined) * 100
    print(f"  {class_name}: {count} ({percentage:.1f}%)")

print(f"\nTarget was {TARGET_SAMPLES_PER_CLASS} per class")
print("Deviation from target:")
for class_name, count in sorted(combined_counts.items()):
    deviation = count - TARGET_SAMPLES_PER_CLASS
    print(f"  {class_name}: {deviation:+d} ({count})")