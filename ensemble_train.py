"""
    Trains the ensemble model (DenseNet-121 and EfficientNet-B5) on the VinDr-CXR dataset

    Requires:
        The VinDr-CXR dataset to have been downloaded and preprocessed (vindr_preprocessing_aug.py)

"""


import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict
from keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
from keras.applications.efficientnet import EfficientNetB5, preprocess_input as efficientnet_preprocess
from keras.optimizers import AdamW
from keras.models import Model, load_model
from keras.layers import *
from keras.callbacks import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.metrics import AUC
from tensorflow.keras import mixed_precision
import tensorflow as tf

print("RUNNING: ENSEMBLE TRAINING")

DATA_DIR = 'output_dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TRAIN_JSONL = os.path.join(TRAIN_DIR, 'annotations.jsonl')
VALID_JSONL = os.path.join(VALID_DIR, 'annotations.jsonl')

BATCH_SIZE = 10
INPUT_SIZE = 1500  # all images have at least one side of 1500px
EPOCHS = 50
LEARNING_RATE = 5e-6 

CHECKPOINT_DIR = 'ensemble_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# loads the annotations from the jsonl file and extract the labels
def load_annotations(jsonl_path):
    annotations = {}
    all_labels = set()
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            image_name = data['image']
            suffix = data['suffix']
            
            # extract labels by removing <loc_xxx> tags
            labels = []
            parts = suffix.split('<loc_')
            for i, part in enumerate(parts):
                if i == 0:
                    # first part contains the first label
                    if part.strip():
                        labels.append(part.strip())
                else:
                    # find the next label after the loc tag
                    if '>' in part:
                        next_label = part.split('>', 1)[1].strip()
                        if next_label and not next_label.startswith('<'):
                            labels.append(next_label.split('<')[0].strip())
            
            # cleans and filters labels
            clean_labels = [label.strip() for label in labels if label.strip()]
            annotations[image_name] = clean_labels
            all_labels.update(clean_labels)
    
    return annotations, sorted(list(all_labels))

train_annotations, train_labels = load_annotations(TRAIN_JSONL)
valid_annotations, valid_labels = load_annotations(VALID_JSONL)

all_unique_labels = sorted(list(set(train_labels + valid_labels)))
N_CLASSES = len(all_unique_labels)

print(f'Total unique labels: {N_CLASSES}')
print(f'Training samples: {len(train_annotations)}')
print(f'Validation samples: {len(valid_annotations)}')

# makes a label mapping
label_2_idx = {label: idx for idx, label in enumerate(all_unique_labels)}
idx_2_label = {idx: label for label, idx in label_2_idx.items()}

print(f"Labels: {all_unique_labels}")

# pads images to a square of dimensions 1500x1500
# maintains aspect ratio
# all images have at least one side that is 1500px
def pad_image_to_square(image, target_size=1500):
    h, w = image.shape[:2]
    
    # calc padding needed
    pad_h = target_size - h
    pad_w = target_size - w
    
    # pad equally on both sides (centers the image)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # apply padding with black pixels (0 values)
    padded_image = cv.copyMakeBorder(
        image, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    
    return padded_image

# converts a list of labels to one-hot encoding
def labels_to_one_hot(labels_list, n_classes=N_CLASSES, lookup_dict=label_2_idx):
    y = np.zeros(n_classes)
    for label in labels_list:
        if label in lookup_dict:
            idx = lookup_dict[label]
            y[idx] = 1
    return y

# apply preprocessing for each model
def preprocess_for_model(image, model_type):
    if model_type == 'densenet':
        return densenet_preprocess(image.astype(np.float32))
    elif model_type == 'efficientnet':
        return efficientnet_preprocess(image.astype(np.float32))
    else:
        return image.astype(np.float32) / 255.0

# a generator that yields an sample one at a time
def ImageDataGen(annotations_dict, img_dir, n_classes=N_CLASSES, input_size=INPUT_SIZE, returnIds=False):
    image_names = list(annotations_dict.keys())
    
    while True:
        np.random.shuffle(image_names)
        
        for img_name in image_names:
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                for ext in ['.jpg', '.png', '.jpeg']:
                    img_path_ext = os.path.join(img_dir, img_name.replace('.jpg', ext).replace('.png', ext).replace('.jpeg', ext))
                    if os.path.exists(img_path_ext):
                        img_path = img_path_ext
                        break
            
            if os.path.exists(img_path):
                img = cv.imread(img_path)
                if img is not None:
                    img_padded = pad_image_to_square(img)  # pads image to square input_size x input_size
                    
                    # preprocess for each model
                    img_densenet = preprocess_for_model(img_padded.copy(), 'densenet')
                    img_efficientnet = preprocess_for_model(img_padded.copy(), 'efficientnet')
                    
                    labels = annotations_dict[img_name]
                    y = labels_to_one_hot(labels)
                    
                    if returnIds:
                        yield ((img_densenet, img_efficientnet), y, img_name)
                    else:
                        yield ((img_densenet, img_efficientnet), y)


## creates individual model branches
# DenseNet branch
def create_densenet_branch(input_shape=(1500, 1500, 3)):
    input_layer = Input(shape=input_shape, name='densenet_input')
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=input_layer)
    
    x = GlobalAveragePooling2D(name='densenet_gap')(base_model.output)
    x = Dense(1024, activation='relu', name='densenet_dense1')(x)
    x = BatchNormalization(name='densenet_bn1')(x)
    x = Dropout(0.3, name='densenet_dropout1')(x)
    x = Dense(512, activation='relu', name='densenet_dense2')(x)
    x = BatchNormalization(name='densenet_bn2')(x)
    x = Dropout(0.2, name='densenet_dropout2')(x)
    
    return Model(inputs=input_layer, outputs=x, name='densenet_branch')

# EfficientNet branch
def create_efficientnet_branch(input_shape=(1500, 1500, 3)):
    input_layer = Input(shape=input_shape, name='efficientnet_input')
    base_model = EfficientNetB5(weights='imagenet', include_top=False, input_tensor=input_layer)
    
    x = GlobalAveragePooling2D(name='efficientnet_gap')(base_model.output)
    x = Dense(1024, activation='relu', name='efficientnet_dense1')(x)
    x = BatchNormalization(name='efficientnet_bn1')(x)
    x = Dropout(0.3, name='efficientnet_dropout1')(x)
    x = Dense(512, activation='relu', name='efficientnet_dense2')(x)
    x = BatchNormalization(name='efficientnet_bn2')(x)
    x = Dropout(0.2, name='efficientnet_dropout2')(x)
    
    return Model(inputs=input_layer, outputs=x, name='efficientnet_branch')

# creates the ensemble model by combining densenet and efficientnet
def create_ensemble_model(n_classes=N_CLASSES, input_shape=(1500, 1500, 3)):
    # create the individual branches
    densenet_branch = create_densenet_branch(input_shape)
    efficientnet_branch = create_efficientnet_branch(input_shape)

    # combine the outputs
    combined = concatenate([
        densenet_branch.output,
        efficientnet_branch.output
    ], name='ensemble_concat')
    
    # adds final classification layers
    x = Dense(1024, activation='relu', name='ensemble_dense1')(combined)
    x = BatchNormalization(name='ensemble_bn1')(x)
    x = Dropout(0.4, name='ensemble_dropout1')(x)
    
    x = Dense(512, activation='relu', name='ensemble_dense2')(x)
    x = BatchNormalization(name='ensemble_bn2')(x)
    x = Dropout(0.3, name='ensemble_dropout2')(x)
    
    # final output layer
    output = Dense(n_classes, activation='sigmoid', name='predictions')(x)

    model = Model(
        inputs=[densenet_branch.input, efficientnet_branch.input],
        outputs=output,
        name='ensemble_model'
    )
    
    return model

model = create_ensemble_model(N_CLASSES)

# gets total number of parameters
total_params = model.count_params()
trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])

print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Non-trainable params: {non_trainable_params:,}")


# # print model summary
# # this summary is VERY long
# print("\nEnsemble Model Summary:")
# model.summary()

# count total parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# compile model with AdamW optimizer
print("Compiling model...")
model.compile(
    optimizer=AdamW(learning_rate=LEARNING_RATE),
    # loss='binary_crossentropy',
    loss='binary_focal_crossentropy',
    metrics=['accuracy', 'precision', 'recall', AUC(multi_label=True)]
)

# set up callbacks
model_checkpoint = ModelCheckpoint(
    os.path.join(CHECKPOINT_DIR, 'ensemble_effnetb5_dnet121.{epoch:02d}-{val_loss:.4f}.keras'),
    monitor='val_loss', 
    verbose=1,
    save_best_only=False, 
    save_weights_only=False
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    verbose=1,
    min_lr=1e-8
)

# early stopping (uncomment if needed)
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     verbose=1,
#     restore_best_weights=True
# )

callbacks = [model_checkpoint, reduce_learning_rate]

# wraps generators in tf.data.Dataset.from_generator()
# need to explicitly define output_signature for efficiennet
train_gen = tf.data.Dataset.from_generator(
    lambda: ImageDataGen(train_annotations, TRAIN_DIR),
    output_signature=(
        (
            tf.TensorSpec(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(N_CLASSES,), dtype=tf.float32),
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


valid_gen = tf.data.Dataset.from_generator(
    lambda: ImageDataGen(valid_annotations, VALID_DIR),
    output_signature=(
        (
            tf.TensorSpec(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(N_CLASSES,), dtype=tf.float32),
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# calculate steps per epoch
train_steps = np.ceil(len(train_annotations) / BATCH_SIZE).astype(int)
valid_steps = np.ceil(len(valid_annotations) / BATCH_SIZE).astype(int)

print(f"Training steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {valid_steps}")

# enabling mixed precision if available
# is available on NVIDIA A100, so should work
try:
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except:
    print("Mixed precision not available, using default precision")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    callbacks=callbacks,
    validation_data=valid_gen,
    validation_steps=valid_steps,
    verbose=1
)

# plot training history
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Ensemble Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Ensemble Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 4, 3)
if 'precision' in history.history:
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Ensemble Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

plt.subplot(1, 4, 4)
if 'auc' in history.history:
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Ensemble Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

plt.tight_layout()
plt.savefig('ensemble_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Training completed!")
print(f"Model checkpoints saved in: {CHECKPOINT_DIR}")
print("Training history plot saved as: ensemble_training_history.png")