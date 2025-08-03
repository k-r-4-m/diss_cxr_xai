import os
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.densenet import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import seaborn as sns
import re

# Configuration
DATA_DIR = 'output_dataset'
VALID_DIR = os.path.join(DATA_DIR, 'valid')
VALID_JSONL = os.path.join(VALID_DIR, 'annotations.jsonl')
CHECKPOINT_DIR = 'densenet_checkpoints'
INPUT_SIZE = 1500
BATCH_SIZE = 18
THRESHOLD = 0.3

CLASSES = [
    'Cardiomegaly', 'Aortic enlargement', 'Pleural thickening', 'ILD', 'Nodule/Mass',
    'Pulmonary fibrosis', 'Lung Opacity', 'Atelectasis', 'Other lesion', 'Infiltration',
    'Pleural effusion', 'Calcification', 'Consolidation', 'Pneumothorax'
]


def load_annotations(jsonl_path):
    annotations = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            image_name = data['image']
            suffix = data['suffix']
            labels = []
            parts = suffix.split('<loc_')
            for i, part in enumerate(parts):
                if i == 0:
                    if part.strip():
                        labels.append(part.strip())
                else:
                    if '>' in part:
                        next_label = part.split('>', 1)[1].strip()
                        if next_label and not next_label.startswith('<'):
                            labels.append(next_label.split('<')[0].strip())
            clean_labels = [label.strip() for label in labels if label.strip()]
            annotations[image_name] = clean_labels
    return annotations


def pad_image_to_square(image, target_size=INPUT_SIZE):
    h, w = image.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return cv.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT, value=[0, 0, 0])


def labels_to_one_hot(labels_list, label_map):
    y = np.zeros(len(label_map))
    for label in labels_list:
        if label in label_map:
            y[label_map[label]] = 1
    return y


def load_validation_data():
    x_val, y_val = [], []
    for img_name in valid_annotations:
        img_path = os.path.join(VALID_DIR, img_name)
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_path = img_path.rsplit('.', 1)[0] + ext
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        if os.path.exists(img_path):
            img = cv.imread(img_path)
            if img is not None:
                img = pad_image_to_square(img)
                img = preprocess_input(img.astype(np.float32))
                x_val.append(img)
                y_val.append(labels_to_one_hot(valid_annotations[img_name], label_2_idx))
    return np.array(x_val), np.array(y_val)

def get_best_model_by_val_loss(checkpoint_dir):
    best_loss = float('inf')
    best_model = None
    pattern = re.compile(r'\.(?:\d+)-(\d+\.\d+)\.keras$')

    for filename in os.listdir(checkpoint_dir):
        match = pattern.search(filename)
        if match:
            loss = float(match.group(1))
            if loss < best_loss:
                best_loss = loss
                best_model = os.path.join(checkpoint_dir, filename)
    return best_model

print("Loading validation annotations...")
valid_annotations = load_annotations(VALID_JSONL)
label_2_idx = {label: idx for idx, label in enumerate(CLASSES)}
idx_2_label = {idx: label for label, idx in label_2_idx.items()}
N_CLASSES = len(CLASSES)

# model = load_model(os.path.join(CHECKPOINT_DIR, 'chest_xray_densenet_final.keras'))
best_model_path = get_best_model_by_val_loss(CHECKPOINT_DIR)
print(f"Loading best model from: {best_model_path}")
model = load_model(best_model_path)

print("Loading validation images...")
x_val, y_val = load_validation_data()

print("Running predictions...")
y_pred_prob = model.predict(x_val, batch_size=BATCH_SIZE)
y_pred = (y_pred_prob > THRESHOLD).astype(int)

# Metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

print("\nEvaluation Results on Validation Set:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("Generating per-class confusion matrices...")

mcm = multilabel_confusion_matrix(y_val, y_pred)

matrix = np.zeros((N_CLASSES + 1, N_CLASSES + 1), dtype=int)

for i, conf in enumerate(mcm):
    tn, fp, fn, tp = conf.ravel()
    matrix[i, i] = tp  # True Positive
    matrix[i, -1] = fn  # False Negative
    matrix[-1, i] = fp  # False Positive
    # Off-diagonal cross-class predictions are skipped as they don't apply

xticks = CLASSES + ['FN']
yticks = CLASSES + ['FP']

plt.figure(figsize=(20, 20))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=xticks, yticklabels=yticks, cbar=True)
plt.title("Multi-label Confusion Matrix (Diagonal: TP, Last Col: FN, Last Row: FP)", fontsize=18)
plt.ylabel("True Label", fontsize=14)
plt.xlabel("Predicted Label", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix_densenet.png", dpi=300)
print("Confusion matrix saved as confusion_matrix_densenet.png")

for i, label in enumerate(CLASSES):
    prec = precision_score(y_val[:, i], y_pred[:, i], zero_division=0)
    rec = recall_score(y_val[:, i], y_pred[:, i], zero_division=0)
    f1c = f1_score(y_val[:, i], y_pred[:, i], zero_division=0)
    print(f"{label:20s}  Precision: {prec:.2f}  Recall: {rec:.2f}  F1: {f1c:.2f}")
