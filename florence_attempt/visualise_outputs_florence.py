# need transformers version>=4.53.1
from transformers import get_scheduler, AutoModelForCausalLM, AutoProcessor, AutoConfig  
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torchvision.transforms.functional import to_pil_image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from difflib import get_close_matches
from typing import List, Dict, Any, Tuple, Generator
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from pathvalidate import sanitize_filename
from collections import defaultdict
from supervision.detection.utils import box_iou_batch
from florence_tools import *
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import os
import torch
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
import sys
import colorsys

print("RUNNING FLORENCE VISUALISATIONS")

# loads the config file for epochs, revision, pathnames, etc.
config_path = "./config.yaml"
config = load_config(config_path)

REVISION = config.get('revision')
EPOCHS = config.get('epochs')
DICOM_DIR = config.get('dicom_dir')
ANNOTATIONS_CSV = config.get('annotations_csv')
OUTPUT_DIR = config.get('output_dir')
print("config loaded")

ANNOTATIONS_JSONL = f"{OUTPUT_DIR}/valid/annotations.jsonl"
IMAGE_PATH = f"{OUTPUT_DIR}/valid"

### loads model
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
CHECKPOINT = f"./model_checkpoints/epoch_{EPOCHS}"  # gets the last checkpoint of the model training

config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)
config.vision_config.model_type = 'davit'
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, config=config, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

# gets a list of all the classes in the dataset
df = pd.read_csv(ANNOTATIONS_CSV)
CLASSES = df['class_name'].unique().tolist()
CLASSES.remove("No finding")  # removes no finding from the classes list

try:
    input_image = sys.argv[1]
except IndexError:
    print("Please provide an image!")
    sys.exit()

prefix = '<OD>'
suffix = None

# scrubs the jsonl file to find the annotations
with open(ANNOTATIONS_JSONL, "r") as f:
    for line in f:
        row = json.loads(line)
        if row["image"] == input_image:
            suffix = row["suffix"]
            break  # stops after finding the match

if suffix == None:
    print("Error finding annotation! Are you sure you provided a valid image file?")
    sys.exit()
else:
    image = Image.open(f"{IMAGE_PATH}/{input_image}").convert('RGB')

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
    prediction.confidence = np.ones(len(prediction))  # florence-2 doesn't output confidence

    target = processor.post_process_generation(suffix, task='<OD>', image_size=image.size)
    target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
    target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])

    print(f"Target {target}")
    print(f"pred {prediction}")

    image_np = np.array(image)


# -------- colour helpers --------
def class_base_colour(name: str) -> tuple:
    """
    Deterministic base colour per class name (RGB in 0-1).
    Uses HSV with class-hash-based hue so it's stable across runs.
    """
    h = (hash(name) % 360) / 360.0
    s, v = 0.65, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

def blend(c, target, t: float):
    """Linear blend between colours c and target with weight t in [0,1]."""
    return tuple((1 - t) * c[i] + t * target[i] for i in range(3))

def gt_shade(base):       # lighter shade (towards white)
    return blend(base, (1, 1, 1), 0.35)

def pred_shade(base):     # darker shade (towards black)
    return blend(base, (0, 0, 0), 0.35)

# -------- drawing --------
def draw_overlay(image_pil, target_dets, pred_dets):
    img = np.asarray(image_pil)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    # Ground truth (lighter, solid)
    for xyxy, cname in zip(target_dets.xyxy, target_dets.data['class_name']):
        base = class_base_colour(cname)
        col = gt_shade(base)
        x1, y1, x2, y2 = xyxy
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2.5, edgecolor=col, facecolor='none', linestyle='-')
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 4, 0), f"{cname} (GT)", fontsize=10, color=col,
                bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.7))

    # Prediction (darker, dashed)
    for xyxy, cname in zip(pred_dets.xyxy, pred_dets.data['class_name']):
        base = class_base_colour(cname)
        col = pred_shade(base)
        x1, y1, x2, y2 = xyxy
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=col, facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 4, 0), f"{cname} (Pred)", fontsize=9, color=col,
                bbox=dict(boxstyle="round,pad=0.2", fc='black', ec='none', alpha=0.35))

    # Legend explaining styles (colour varies by class)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='black', lw=2.5, linestyle='-', label='Ground truth'),
        Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='Prediction'),
    ]
    ax.legend(handles=legend_elems, loc='lower right')
    ax.axis('off')

    plt.savefig("florence_output_visualised", dpi=200, bbox_inches="tight")
    plt.close()


# call it
draw_overlay(image, target, prediction)