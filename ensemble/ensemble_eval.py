"""
    Evaluates the classification performance of the ensemble model on the VinDr-CXR dataset

    Outputs:
        Precision, recall, and F1 per-class and overall
        Youden's J analysis to analyse what the best thresholds are for classification for each class    

    Requires:
        The ensemble model to have been trained and for a model checkpoint to be in ensemble_checkpoints/epoch_n

"""


import os
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.densenet import preprocess_input as densenet_preprocess
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re
import tensorflow as tf

print("RUNNING: ENSEMBLE EVALUATION")

DATA_DIR = 'output_dataset'
VALID_DIR = os.path.join(DATA_DIR, 'valid')
VALID_JSONL = os.path.join(VALID_DIR, 'annotations.jsonl')
# ENSEMBLE_CHECKPOINT_DIR = 'ensemble_checkpoints_no_focal'
ENSEMBLE_CHECKPOINT_DIR = 'ensemble_checkpoints'
INPUT_SIZE = 1500
BATCH_SIZE = 9
THRESHOLD = 0.1

CLASSES = [
    'Cardiomegaly', 'Aortic enlargement', 'Pleural thickening', 'ILD', 'Nodule/Mass',
    'Pulmonary fibrosis', 'Lung Opacity', 'Atelectasis', 'Other lesion', 'Infiltration',
    'Pleural effusion', 'Calcification', 'Consolidation', 'Pneumothorax'
]


# loads annotations from the jsonl file and extract the labels
def load_annotations(jsonl_path):
    annotations = {}
    all_labels = set()
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            image_name = data['image']
            suffix = data['suffix']
            
            # extracts the labels by getting rid of <loc_...> tags
            labels = []
            parts = suffix.split('<loc_')
            for i, part in enumerate(parts):
                if i == 0:
                    # first part contains the first label
                    if part.strip():
                        labels.append(part.strip())
                else:
                    # finds the next label based on the loc tag
                    if '>' in part:
                        next_label = part.split('>', 1)[1].strip()
                        if next_label and not next_label.startswith('<'):
                            labels.append(next_label.split('<')[0].strip())
            
            # clean and filter the labels
            clean_labels = [label.strip() for label in labels if label.strip()]
            annotations[image_name] = clean_labels
            all_labels.update(clean_labels)
    
    return annotations, sorted(list(all_labels))


# pads an image to a target size
# maintains aspect ratio
def pad_image_to_square(image, target_size=INPUT_SIZE):
    h, w = image.shape[:2]
    
    # calc the padding needed for this image
    pad_h = target_size - h
    pad_w = target_size - w
    
    # pads equally on both sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # padding with black pixels
    # (0, 0, 0 is black)
    padded_image = cv.copyMakeBorder(
        image, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    
    return padded_image


# different preprocessing needed for each model
def preprocess_for_model(image, model_type):
    if model_type == 'densenet':
        return densenet_preprocess(image.astype(np.float32))
    elif model_type == 'efficientnet':
        return efficientnet_preprocess(image.astype(np.float32))
    else:
        return image.astype(np.float32) / 255.0


# converts a list of labels into one-hot encoding
def labels_to_one_hot(labels_list, label_map):
    y = np.zeros(len(label_map))
    for label in labels_list:
        if label in label_map:
            y[label_map[label]] = 1
    return y


# loads the validation data for the model
def load_validation_data_for_ensemble():
    x_val_densenet, x_val_efficientnet, y_val = [], [], []
    
    for img_name in valid_annotations:
        img_path = os.path.join(VALID_DIR, img_name)
        
        # just in case the images were saved as non png for whatever reason
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_path = img_path.rsplit('.', 1)[0] + ext
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        if os.path.exists(img_path):
            img = cv.imread(img_path)
            if img is not None:
                # same preprocessing as in training
                img_padded = pad_image_to_square(img)
                
                # preprocess for each model branch
                img_densenet = preprocess_for_model(img_padded.copy(), 'densenet')
                img_efficientnet = preprocess_for_model(img_padded.copy(), 'efficientnet')
                
                x_val_densenet.append(img_densenet)
                x_val_efficientnet.append(img_efficientnet)
                
                # converts labels to one-hot
                y_val.append(labels_to_one_hot(valid_annotations[img_name], label_2_idx))
    
    return np.array(x_val_densenet), np.array(x_val_efficientnet), np.array(y_val)


# gets the best model checkpoint with the lowest loss
def get_best_model_by_loss(checkpoint_dir):
    best_loss = float('inf')
    best_model = None
    pattern = re.compile(r'\.(\d+)-(\d+\.\d+)\.keras$')

    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('ensemble_') and filename.endswith('.keras'):
            match = pattern.search(filename)
            if match:
                loss = float(match.group(2))
                if loss < best_loss:
                    best_loss = loss
                    best_model = os.path.join(checkpoint_dir, filename)
    return best_model


# gets the checkpoint from the last epoch
# prolly not gonna use, just curious
def get_last_epoch_model(checkpoint_dir):
    last_epoch = -1
    last_model = None
    pattern = re.compile(r'\.(\d+)-(\d+\.\d+)\.keras$')

    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('ensemble_') and filename.endswith('.keras'):
            match = pattern.search(filename)
            if match:
                epoch = int(match.group(1))
                if epoch > last_epoch:
                    last_epoch = epoch
                    last_model = os.path.join(checkpoint_dir, filename)
    return last_model


print("Loading validation annotations")
valid_annotations, all_labels = load_annotations(VALID_JSONL)

label_2_idx = {label: idx for idx, label in enumerate(CLASSES)}
idx_2_label = {idx: label for label, idx in label_2_idx.items()}
N_CLASSES = len(CLASSES)

print(f"{len(valid_annotations)} validation samples")
print(f"{N_CLASSES} classes: {CLASSES}")

# loads the model checkpoint
print("Loading model")
# model_path = get_last_epoch_model(ENSEMBLE_CHECKPOINT_DIR)
model_path = get_best_model_by_loss(ENSEMBLE_CHECKPOINT_DIR)

if model_path is None:
    raise ValueError(f"No ensemble model found in {ENSEMBLE_CHECKPOINT_DIR}!!!")

print(f"Loading ensemble model with path: {model_path}")
ensemble_model = load_model(model_path)

# loads the validation data
print("Loading validation items")
x_val_densenet, x_val_efficientnet, y_val = load_validation_data_for_ensemble()

print(f"Loaded {len(x_val_densenet)} validation images")
print(f"DenseNet input shape: {x_val_densenet.shape}")
print(f"EfficientNet input shape: {x_val_efficientnet.shape}")
print(f"Labels shape: {y_val.shape}")

print("Running model predictions")
y_pred_prob = ensemble_model.predict(
    [x_val_densenet, x_val_efficientnet], 
    batch_size=BATCH_SIZE, 
    verbose=1
)

# youden's J analysis suggests these thresholds per-class
# these values were arrived at after running the code once
class_thresholds =  {
    'Cardiomegaly': 0.64,
    'Aortic enlargement': 0.10,
    'Pleural thickening': 0.31,
    'ILD': 0.96,
    'Nodule/Mass': 0.11,
    'Pulmonary fibrosis': 0.10,
    'Lung Opacity': 0.20,
    'Atelectasis': 0.43,
    'Other lesion': 0.31,
    'Infiltration': 0.27,
    'Pleural effusion': 0.30,
    'Calcification': 0.30,
    'Consolidation': 0.06,
    'Pneumothorax': 0.47
}

# # converts the probabilities to binary predictions using the threshold
# y_pred = (y_pred_prob > THRESHOLD).astype(int)

# initialise empty predictions
y_pred = np.zeros_like(y_pred_prob, dtype=int)

# apply per-class thresholds
for label, t in class_thresholds.items():
    idx = label_2_idx[label]
    y_pred[:, idx] = (y_pred_prob[:, idx] > t).astype(int)


# calculate precision, recall, f1
precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

print(f"\nModel eval results: ")
print(f"Threshold: {THRESHOLD}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# generate confusion matrices
print("Generating per-class confusion matrices")
os.makedirs("confusion_matrices_joint_ensemble", exist_ok=True)

mcm = multilabel_confusion_matrix(y_val, y_pred)

for i, label in enumerate(CLASSES):
    cm = mcm[i]
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
    plt.title(f"{label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    filename = f"confusion_matrices_joint_ensemble/conf_matrix_{label.replace(' ', '_').replace('/', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

print(f"Confusion matrices saved in: confusion_matrices_joint_ensemble/")

# per-class precision, recall, and F1
print(f"\nPer-class Results:")
print(f"{'Class':<20} {'Precision':<10} {'Recall':<8} {'F1':<8}")
print("-" * 60)

for i, label in enumerate(CLASSES):
    prec = precision_score(y_val[:, i], y_pred[:, i], zero_division=0)
    rec = recall_score(y_val[:, i], y_pred[:, i], zero_division=0)
    f1c = f1_score(y_val[:, i], y_pred[:, i], zero_division=0)
    
    print(f"{label:<20} {prec:<10.3f} {rec:<8.3f} {f1c:<8.3f}")

### testing different thresholds (don't remove)
# print(f"\nThreshold analysis: ")
# thresholds = np.arange(0.1, 0.9, 0.1)
# threshold_results = []

# for thresh in thresholds:
#     temp_pred = (y_pred_prob > thresh).astype(int)
#     temp_f1 = f1_score(y_val, temp_pred, average='macro', zero_division=0)
#     temp_precision = precision_score(y_val, temp_pred, average='macro', zero_division=0)
#     temp_recall = recall_score(y_val, temp_pred, average='macro', zero_division=0)
#     threshold_results.append((thresh, temp_f1, temp_precision, temp_recall))
#     print(f"Threshold {thresh:.1f}: F1={temp_f1:.4f}, Prec={temp_precision:.4f}, Rec={temp_recall:.4f}")

# # find the optimal threshold
# best_threshold_idx = np.argmax([result[1] for result in threshold_results])
# best_threshold = threshold_results[best_threshold_idx][0]
# print(f"\nOptimal threshold for F1: {best_threshold:.1f} (F1={threshold_results[best_threshold_idx][1]:.4f})")


### youden's j statistic calculation
print("\nYouden's J analysis (per class):")
youden_results = []

thresholds = np.arange(0.0, 1.0, 0.01)  # sweeping through with values of 0.01
best_thresholds = {}

for i, label in enumerate(CLASSES):
    y_true = y_val[:, i]
    best_j, best_t = -1, None
    
    for t in thresholds:
        y_pred_t = (y_pred_prob[:, i] > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0,1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sensitivity + specificity - 1
        
        if j > best_j:
            best_j = j
            best_t = t
    
    best_thresholds[label] = best_t
    youden_results.append((label, best_t, best_j))
    print(f"{label:<20} Best Thresh={best_t:.2f}, J={best_j:.3f}")
