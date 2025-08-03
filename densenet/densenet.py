import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.optimizers import AdamW
from keras.models import Model, load_model
from keras.layers import *
from keras.callbacks import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, ConfusionMatrixDisplay

# Dataset configuration
DATA_DIR = 'output_dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TRAIN_JSONL = os.path.join(TRAIN_DIR, 'annotations.jsonl')
VALID_JSONL = os.path.join(VALID_DIR, 'annotations.jsonl')

# Training configuration
BATCH_SIZE = 18
INPUT_SIZE = 1500  # Target size - all images have at least one side of 1500px
EPOCHS = 50
LEARNING_RATE = 5e-6

# creates dir for checkpoints
CHECKPOINT_DIR = 'densenet_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_annotations(jsonl_path):
    """Load annotations from JSONL file and extract labels"""
    annotations = {}
    all_labels = set()
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            image_name = data['image']  # Adjust key name if different
            suffix = data['suffix']     # Adjust key name if different
            
            # Extract labels by removing <loc_xxx> tags
            labels = []
            parts = suffix.split('<loc_')
            for i, part in enumerate(parts):
                if i == 0:
                    # First part contains the first label
                    if part.strip():
                        labels.append(part.strip())
                else:
                    # Find the next label after the location tag
                    if '>' in part:
                        next_label = part.split('>', 1)[1].strip()
                        if next_label and not next_label.startswith('<'):
                            labels.append(next_label.split('<')[0].strip())
            
            # Clean and filter labels
            clean_labels = [label.strip() for label in labels if label.strip()]
            annotations[image_name] = clean_labels
            all_labels.update(clean_labels)
    
    return annotations, sorted(list(all_labels))

# Load training and validation annotations
print("Loading annotations...")
train_annotations, train_labels = load_annotations(TRAIN_JSONL)
valid_annotations, valid_labels = load_annotations(VALID_JSONL)

# Create unified label vocabulary
all_unique_labels = sorted(list(set(train_labels + valid_labels)))
N_CLASSES = len(all_unique_labels)

print(f'Total unique labels: {N_CLASSES}')
print(f'Training samples: {len(train_annotations)}')
print(f'Validation samples: {len(valid_annotations)}')

# Create label mappings
label_2_idx = {label: idx for idx, label in enumerate(all_unique_labels)}
idx_2_label = {idx: label for label, idx in label_2_idx.items()}

print(f"Labels: {all_unique_labels}")

def pad_image_to_square(image, target_size=1500):
    """
    Pad image to 1500x1500 square while maintaining aspect ratio.
    All images already have at least one side that is 1500px.
    """
    h, w = image.shape[:2]
    
    # Since all images already have at least one side = 1500px, no resizing needed
    # Just pad the shorter dimension to make it square
    
    # Calculate padding needed
    pad_h = target_size - h
    pad_w = target_size - w
    
    # Pad equally on both sides (center the image)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding with black pixels (0 values)
    padded_image = cv.copyMakeBorder(
        image, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    
    return padded_image

def labels_to_one_hot(labels_list, n_classes=N_CLASSES, lookup_dict=label_2_idx):
    """Convert list of labels to one-hot encoding"""
    y = np.zeros(n_classes)
    for label in labels_list:
        if label in lookup_dict:
            idx = lookup_dict[label]
            y[idx] = 1
    return y

def ImageDataGen(annotations_dict, img_dir, n_classes=N_CLASSES, 
                 input_size=INPUT_SIZE, bs=BATCH_SIZE, returnIds=False):
    """Data generator for training/validation with aspect-ratio preserving padding"""
    image_names = list(annotations_dict.keys())
    
    while True:
        # Shuffle the data each epoch
        np.random.shuffle(image_names)
        
        for start in range(0, len(image_names), bs):
            x_batch = []
            y_batch = []
            ids_batch = []
            end = min(start + bs, len(image_names))
            batch_names = image_names[start:end]
            
            for img_name in batch_names:
                img_path = os.path.join(img_dir, img_name)
                
                # Try different extensions if needed
                if not os.path.exists(img_path):
                    for ext in ['.jpg', '.png', '.jpeg']:
                        img_path_ext = os.path.join(img_dir, img_name.replace('.jpg', ext).replace('.png', ext).replace('.jpeg', ext))
                        if os.path.exists(img_path_ext):
                            img_path = img_path_ext
                            break
                
                if os.path.exists(img_path):
                    img = cv.imread(img_path)
                    if img is not None:
                        # Pad image to 1500x1500 square (no resizing needed)
                        img = pad_image_to_square(img)
                        img = preprocess_input(img.astype(np.float32))
                        x_batch.append(img)
                        
                        labels = annotations_dict[img_name]
                        y = labels_to_one_hot(labels, n_classes=n_classes)
                        y_batch.append(y)
                        ids_batch.append(img_name)
            
            if len(x_batch) > 0:
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.float32)
                
                if returnIds:
                    yield x_batch, y_batch, ids_batch
                else:
                    yield x_batch, y_batch

# creates the densenet model
def ClsModel(n_classes=N_CLASSES, input_shape=(1500, 1500, 3)):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # freeze base model initially
    base_model.trainable = True
    
    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = Dense(1024, activation='relu', name='dense_post_pool')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='dense_intermediate')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Create model
print("Creating model...")
model = ClsModel(N_CLASSES)
model.summary()

# Compile model with AdamW optimizer
model.compile(
    optimizer=AdamW(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Set up callbacks
model_checkpoint = ModelCheckpoint(
    os.path.join(CHECKPOINT_DIR, 'chest_xray_densenet.{epoch:02d}-{val_loss:.4f}.keras'),
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-8
)

# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=7,
#     verbose=1,
#     restore_best_weights=True
# )

# callbacks = [model_checkpoint, reduce_learning_rate, early_stopping]
callbacks = [model_checkpoint, reduce_learning_rate]

# Create data generators
print("Creating data generators...")
train_gen = ImageDataGen(train_annotations, TRAIN_DIR)
valid_gen = ImageDataGen(valid_annotations, VALID_DIR)

# Calculate steps per epoch
train_steps = np.ceil(len(train_annotations) / BATCH_SIZE).astype(int)
valid_steps = np.ceil(len(valid_annotations) / BATCH_SIZE).astype(int)

print(f"Training steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {valid_steps}")

# Train the model
print("Starting training...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    callbacks=callbacks,
    validation_data=valid_gen,
    validation_steps=valid_steps,
    verbose=1
)

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
if 'precision' in history.history:
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

plt.tight_layout()
plt.savefig('densenet_training_history.png', dpi=300) 
plt.close()