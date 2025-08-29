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
from supervision.detection.utils import box_iou_batch
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
    prediction.confidence = np.ones(len(prediction))  # florence-2 doesn't output confidence

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
conf_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=CLASSES
)

# saves the confusion matrix as an image
fig = conf_matrix.plot()
fig.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')


# calculates the per-class IoU values
def compute_per_class_iou(preds, gts, num_classes):
    # dictionary to hold IoU values for each class
    iou_per_class = defaultdict(list)

    # for each prediction and ground truth
    for pred, gt in zip(preds, gts):
        # for each class, calcualte the IoU
        for cls in range(num_classes):
            pred_cls = pred[pred.class_id == cls]
            gt_cls = gt[gt.class_id == cls]

            # no predicted or ground truth for this class
            if len(gt_cls) == 0 and len(pred_cls) == 0:
                continue

            iou_matrix = box_iou_batch(gt_cls.xyxy, pred_cls.xyxy)

            if iou_matrix.size == 0:
                continue  # nothing to match, i.e. no prediction

            matched_pred = set()
            matched_gt = set()

            for gt_idx, ious in enumerate(iou_matrix):
                if ious.size == 0:
                    continue

                best_pred_idx = np.argmax(ious)
                iou = ious[best_pred_idx]
                if iou >= 0.5 and best_pred_idx not in matched_pred:
                    iou_per_class[cls].append(iou)
                    matched_pred.add(best_pred_idx)
                    matched_gt.add(gt_idx)

    # returns the average IoU per class
    avg_iou_per_class = {CLASSES[cls]: np.mean(iou_list) if iou_list else 0.0 for cls, iou_list in iou_per_class.items()}

    # ensure all classes are included, even if empty
    for cls in range(num_classes):
        class_name = CLASSES[cls]
        if class_name not in avg_iou_per_class:
            avg_iou_per_class[class_name] = 0.0

    return avg_iou_per_class

# calculates precision and recall
conf_mat = conf_matrix.matrix  # gets the actual matrix from the confusion matrix
precision_per_class = {}
recall_per_class = {}
f1_per_class = {}
epsilon = 1e-8  # used to add small constant for f1 score, prevents divison by zero

# for each class, calculate precision and recall
for i, class_name in enumerate(CLASSES):
    TP = conf_mat[i, i]  # true positives
    FP = conf_mat[:, i].sum() - TP  # false positives
    FN = conf_mat[i, :].sum() - TP  # false negatives

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    precision_per_class[class_name] = precision
    recall_per_class[class_name] = recall

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f1_per_class[class_name] = f1

total_TP = np.trace(conf_mat)  # total true positives
total_FP = conf_mat.sum(axis=0).sum() - total_TP  # total false positives
total_FN = conf_mat.sum(axis=1).sum() - total_TP  # total false negatives

overall_precision = np.mean(list(precision_per_class.values()))
overall_recall = np.mean(list(recall_per_class.values()))
overall_f1 = np.mean(list(f1_per_class.values()))

print("\n per class IoU:")
iou_scores = compute_per_class_iou(predictions, targets, num_classes=len(CLASSES))
for cls, iou in iou_scores.items():
    print(f"{cls}: IoU = {iou:.3f}")

print("\n per class precision, recall, and f1")
for cls in CLASSES:
    print(f"{cls}: precision = {precision_per_class[cls]:.3f}, "
          f"recall = {recall_per_class[cls]:.3f}, "
          f"f1 = {f1_per_class[cls]:.3f}")

print("\n overall precision, recall, and f1")
print(f"precision = {overall_precision:.3f}")
print(f"recall = {overall_recall:.3f}")
print(f"f1 = {overall_f1:.3f}")

# classification-style metrics without considering IoU
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
from sklearn.metrics import precision_recall_fscore_support

precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
    y_true_multi, y_pred_multi, average="macro", zero_division=0
)

print("\nClassification-style precision, recall, F1 (ignoring IoU):")
print(f"precision = {precision_cls:.3f}")
print(f"recall = {recall_cls:.3f}")
print(f"f1 = {f1_cls:.3f}")

# create output directory for confusion matrices
os.makedirs("confusion_matrices_florence", exist_ok=True)

def get_binary_confusion_matrix_for_class(class_id: int, class_name: str, predictions, targets):
    y_true = []
    y_pred = []

    for pred, gt in zip(predictions, targets):
        gt_ids = set(gt.class_id.tolist())
        pred_ids = set(pred.class_id.tolist())

        y_true.append(int(class_id in gt_ids))
        y_pred.append(int(class_id in pred_ids))

    # Compute 2x2 confusion matrix: [[TN, FP], [FN, TP]]
    cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    reordered_cm = np.array([[tp, fn],
                            [fp, tn]])

    # Plot
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

    # Save to folder
    filename = f"confusion_matrices_florence/conf_matrix_{sanitize_filename(class_name)}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    # calculate per-class precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return cm, precision, recall, f1 

# Loop through each class and generate/save its confusion matrix
print("\nGenerating per-class binary confusion matrices:")
for class_id, class_name in enumerate(CLASSES):
    cm, precision, recall, f1 = get_binary_confusion_matrix_for_class(class_id, class_name, predictions, targets)
    print(f"{class_name}:\n{cm}")
    print(f" Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")