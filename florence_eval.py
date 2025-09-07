"""
    Evaluates the classification and localisation performance of Florence-2 on the VinDr-CXR dataset

    Evaluates the following:
        Classification performance (precision, recall, F1)
        Localisation performance (mean average precision)

    Requires:
        Florence to have been pretrained with a model checkpoint stored in model_checkpoints/epoch_n
        VinDr-CXR to have been downloaded and preprocessed
"""

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
import os
import torch
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

print("RUNNING: FLORENCE EVALUATION")

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
    jsonl_file_path = f"{OUTPUT_DIR}/valid/annotations.jsonl",
    image_directory_path = f"{OUTPUT_DIR}/valid/"
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)


### loads model
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

### eval loop
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
    prediction.confidence = np.ones(len(prediction))  # florence-2 doesn't output confidence

    target = processor.post_process_generation(suffix, task='<OD>', image_size=image.size)
    target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
    target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])

    targets.append(target)
    predictions.append(prediction)

    print(f"Target {target}")
    print(f"Pred {prediction}")


### calculates mAP scores
mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets)

print(f"map50: {mean_average_precision.map50:.2f}")
print(f"map75: {mean_average_precision.map75:.2f}")
print(f"map50_95: {mean_average_precision.map50_95:.2f}")

def compute_map_per_class(predictions, targets, class_id, iou_thresholds=[0.5, 0.75]):
    # filter preds and targets for this class
    class_preds = []
    class_gts = []
    for pred, gt in zip(predictions, targets):
        pred_cls = pred[pred.class_id == class_id]
        gt_cls = gt[gt.class_id == class_id]
        class_preds.append(pred_cls)
        class_gts.append(gt_cls)

    # compute per-class AP
    # uses the mAP object, but is just doing over 1 class, so it's just AP
    map_metrics = sv.MeanAveragePrecision.from_detections(
        predictions=class_preds,
        targets=class_gts,
    )
    return map_metrics.map50, map_metrics.map75, map_metrics.map50_95

# compute for all classes
print("\nPer-class AP values:")
for class_id, class_name in enumerate(CLASSES):
    ap50, ap75, ap50_95 = compute_map_per_class(predictions, targets, class_id)
    print(f"{class_name}: AP@50={ap50:.3f}, AP@75={ap75:.3f}, AP@50-95={ap50_95:.3f}")


### classification-style metrics without considering IoU
y_true_multi, y_pred_multi = [], []

for pred, gt in zip(predictions, targets):
    gt_ids = set(gt.class_id.tolist())
    pred_ids = set(pred.class_id.tolist())

    # create binary vector per image for each class
    y_true_multi.append([1 if i in gt_ids else 0 for i in range(len(CLASSES))])
    y_pred_multi.append([1 if i in pred_ids else 0 for i in range(len(CLASSES))])

y_true_multi = np.array(y_true_multi)
y_pred_multi = np.array(y_pred_multi)

# macro averages across classes
precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
    y_true_multi, y_pred_multi, average="macro", zero_division=0
)

print("\nClassification-style precision, recall, F1 (ignoring IoU):")
print(f"precision = {precision_cls:.3f}")
print(f"recall = {recall_cls:.3f}")
print(f"f1 = {f1_cls:.3f}")

# creates output directory for confusion matrices
os.makedirs("confusion_matrices_florence", exist_ok=True)

def get_binary_confusion_matrix_for_class(class_id: int, class_name: str, predictions, targets):
    y_true = []
    y_pred = []

    for pred, gt in zip(predictions, targets):
        gt_ids = set(gt.class_id.tolist())
        pred_ids = set(pred.class_id.tolist())

        y_true.append(int(class_id in gt_ids))
        y_pred.append(int(class_id in pred_ids))

    # computes 2x2 conf matrix
    # [[TN, FP], [FN, TP]]
    cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    reordered_cm = np.array([[tp, fn],
                            [fp, tn]])

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        reordered_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Positive", "Negative"],
        yticklabels=["Positive", "Negative"]
    )
    plt.title(f"{class_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    filename = f"confusion_matrices_florence/conf_matrix_{sanitize_filename(class_name)}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    # calculates per-class precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return cm, precision, recall, f1 

# loops through each class and generate/save its confusion matrix
print("\nGenerating per-class binary confusion matrices:")
for class_id, class_name in enumerate(CLASSES):
    cm, precision, recall, f1 = get_binary_confusion_matrix_for_class(class_id, class_name, predictions, targets)
    print(f"{class_name}:\n{cm}")
    print(f" Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")