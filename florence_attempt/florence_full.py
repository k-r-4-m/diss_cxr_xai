import os
import torch
# need transformers version 4.53.1
from transformers import get_scheduler, AutoModelForCausalLM, AutoProcessor, AutoConfig  
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_pil_image
from IPython.display import display
from pydicom.pixel_data_handlers.util import apply_voi_lut
from difflib import get_close_matches
from typing import List, Dict, Any, Tuple, Generator
from PIL import Image
from tqdm import tqdm
from IPython.core.display import HTML
from datetime import datetime
from pathvalidate import sanitize_filename
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import albumentations as A
import torchvision.transforms as T
import supervision as sv
import yaml
import pydicom
import tensorflow as tf
import os
import re
import json
import html
import base64
import itertools



CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/24'  # revision of florence-2 that fixes the GenerationMixin import error
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads the model and processor for florence-2
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

DICOM_DIR = "./dicom/"
ANNOTATIONS_CSV = "./train_original.csv"
OUTPUT_DIR = "./output_dataset/"


###-----prepares the data----###
## not needed to run if the data is already preprocessed and ready for training

# IMAGE_EXT = ".png"
# TARGET_LONG_SIDE = 500

# # gets the chest x-ray image from a DICOM file
# def load_dicom_image(path, voi_lut=True, fix_monochrome=True):
#     try:
#         dicom = pydicom.dcmread(path)

#         if voi_lut:
#             data = apply_voi_lut(dicom.pixel_array, dicom)
#         else:
#             data = dicom.pixel_array

#         if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
#             data = np.amax(data) - data

#     except:
#         raise ValueError(f"File at {path} is not a valid DICOM file.")

#     data = data - np.min(data)
#     data = data / np.max(data)
#     data = (data * 255).astype(np.uint8)

#     # DICOM files are greyscale, need to convert to RGB
#     return Image.fromarray(data).convert("RGB")

# # resizes an image but keeps the aspect ratio
# def resize_image_keep_aspect(image, target_long_side=500):
#     w, h = image.size
#     if w >= h:
#         scale = target_long_side / w
#         new_size = (target_long_side, int(h * scale))
#     else:
#         scale = target_long_side / h
#         new_size = (int(w * scale), target_long_side)
#     return image.resize(new_size), scale

# # normalises bounding boxes so that they are values between 0 and 1000 (florence-2 requirement)
# def normalise_bbox(x, y, w, h, image_w, image_h):
#     return [
#         int((x / image_w) * 1000),
#         int((y / image_h) * 1000),
#         int(((x + w) / image_w) * 1000),
#         int(((y + h) / image_h) * 1000)
#     ]

# # adds <loc> tags to the bounding boxes (florence-2 requirement)
# def encode_suffix(class_name, bbox):
#     x_min, y_min, x_max, y_max = bbox
#     return f"{class_name}<loc_{x_min}><loc_{y_min}><loc_{x_max}><loc_{y_max}>"


# df = pd.read_csv(ANNOTATIONS_CSV)
# df = df.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])  # drop all rows with missing bounding boxes (i.e. no finding)
# df['bbox'] = df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
# df = df.drop(columns=['x_min', 'y_min', 'x_max', 'y_max'])

# # group by image
# grouped = df.groupby('image_id')

# # train/val split
# image_ids = df['image_id'].unique()
# train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
# splits = {'train': train_ids, 'valid': val_ids}

# # splits images into seperate folders and puts annotations into seperate jsonl files
# for split_name, split_ids in splits.items():
#     os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)
#     jsonl_path = os.path.join(OUTPUT_DIR, split_name, "annotations.jsonl")

#     with open(jsonl_path, "w") as f_out:
#         for image_id in tqdm(split_ids, desc=f"Processing {split_name}"):
#             image_path = os.path.join(DICOM_DIR, f"{image_id}.dicom")
#             if not os.path.exists(image_path):
#                 continue

#             image = load_dicom_image(image_path)
#             original_w, original_h = image.size

#             resized_image, scale = resize_image_keep_aspect(image, TARGET_LONG_SIDE)
#             resized_w, resized_h = resized_image.size

#             resized_path = os.path.join(OUTPUT_DIR, split_name, f"{image_id}{IMAGE_EXT}")
#             resized_image.save(resized_path)

#             try:
#                 entries = grouped.get_group(image_id)
#             except KeyError:
#                 continue  # Image has no annotations

#             suffix_parts = []
#             for _, row in entries.iterrows():
#                 x_min, y_min, x_max, y_max = row['bbox']
#                 x_min *= scale
#                 y_min *= scale
#                 x_max *= scale
#                 y_max *= scale
#                 box_norm = normalise_bbox(
#                     x=x_min,
#                     y=y_min,
#                     w=x_max - x_min,
#                     h=y_max - y_min,
#                     image_w=resized_w,
#                     image_h=resized_h
#                 )
#                 suffix_parts.append(encode_suffix(row['class_name'], box_norm))

#             json_entry = {
#                 "image": f"{image_id}{IMAGE_EXT}",
#                 "prefix": "<OD>",
#                 "suffix": " ".join(suffix_parts)
#             }
#             f_out.write(json.dumps(json_entry) + "\n")

# from PIL import Image
# import os

# # Set your image directory
# image_dir = 'E:/vinbigdata_xrays/output_dataset/train'

# min_size = (10000, 10000)
# min_path = None
# max_size = (0, 0)
# max_path = None

# # Loop through all files in the directory
# for filename in os.listdir(image_dir):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
#         image_path = os.path.join(image_dir, filename)
#         try:
#             with Image.open(image_path) as img:
#                 if img.size < min_size:
#                     min_size = img.size
#                     min_path = image_path

#                 if img.size > max_size:
#                     max_size = img.size
#                     max_path = image_path

#         except Exception as e:
#             print(f"Error opening {filename}: {e}")

# print(min_size)
# print(min_path)
# print(max_size)
# print(max_path)
# ## calculates the intersection over union of different radiologist annotations when they annotate the same label

# import pandas as pd
# import itertools
# import numpy as np

# def compute_iou(box1, box2):
#     """
#     box format: (x_min, y_min, x_max, y_max)
#     """
#     xA = max(box1[0], box2[0])
#     yA = max(box1[1], box2[1])
#     xB = min(box1[2], box2[2])
#     yB = min(box1[3], box2[3])

#     inter_area = max(0, xB - xA) * max(0, yB - yA)
#     if inter_area == 0:
#         return 0.0

#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area

# def analyse_duplicate_ious(csv_path):
#     df = pd.read_csv(csv_path)

#     # filter out no finding
#     df = df[df["class_name"].str.lower() != "no finding"]

#     # group by image and class_name
#     grouped = df.groupby(["image_id", "class_name"])

#     results = []

#     for (image_id, class_name), group in grouped:
#         if len(group) < 2:
#             continue  # no dupes

#         # get all boxes for the group
#         boxes = group[["x_min", "y_min", "x_max", "y_max"]].values

#         ious = []
#         for box1, box2 in itertools.combinations(boxes, 2):
#             iou = compute_iou(box1, box2)
#             ious.append(iou)

#         if ious:
#             results.append({
#                 "image_id": image_id,
#                 "class_name": class_name,
#                 "num_annotations": len(boxes),
#                 "min_iou": np.min(ious),
#                 "max_iou": np.max(ious),
#                 "avg_iou": np.mean(ious)
#             })

#     return pd.DataFrame(results)


# results_df = analyse_duplicate_ious(ANNOTATIONS_CSV)
# print(results_df["min_iou"].mean())
# print(results_df["max_iou"].mean())
# print(results_df["avg_iou"].mean())


# makes a dataset given a jsonl file
class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")


# forms a dataloader for a given dataset
class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data['prefix']  # prefix is the task
        suffix = data['suffix']  # suffix is the annotations
        return prefix, suffix, image


BATCH_SIZE = 6  # batch size for training
NUM_WORKERS = 0  # number of workers for data loading

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

# builds the dataloader for the training set
train_dataset = DetectionDataset(
    jsonl_file_path = f"{OUTPUT_DIR}/train/annotations.jsonl",
    image_directory_path = f"{OUTPUT_DIR}/train/"
)

# builds the dataloader for the validation set
val_dataset = DetectionDataset(
    jsonl_file_path = f"{OUTPUT_DIR}/valid/annotations.jsonl",
    image_directory_path = f"{OUTPUT_DIR}/valid/"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# specifies the configuration for LoRa
config = LoraConfig(
    r=32,  # previously 8
    lora_alpha=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
    revision=REVISION
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

torch.cuda.empty_cache()

# @title Run inference with pre-trained Florence-2 model on validation dataset


# def render_inline(image: Image.Image, resize=(128, 128)):
#     """Convert image into inline html."""
#     image.resize(resize)
#     with io.BytesIO() as buffer:
#         image.save(buffer, format='jpeg')
#         image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
#         return f"data:image/jpeg;base64,{image_b64}"


# def render_example(image: Image.Image, response):
#     try:
#         detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
#         image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image.copy(), detections)
#         image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image, detections)
#     except:
#         print('failed to render model response')
#     return f"""
# <div style="display: inline-flex; align-items: center; justify-content: center;">
#     <img style="width:256px; height:256px;" src="{render_inline(image, resize=(128, 128))}" />
#     <p style="width:512px; margin:10px; font-size:small;">{html.escape(json.dumps(response))}</p>
# </div>
# """


# def render_inference_results(model, dataset: DetectionDataset, count: int):
#     html_out = ""
#     count = min(count, len(dataset))
#     for i in range(count):
#         image, data = dataset.dataset[i]
#         prefix = data['prefix']
#         suffix = data['suffix']
#         inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
#         generated_ids = model.generate(
#             input_ids=inputs["input_ids"],
#             pixel_values=inputs["pixel_values"],
#             max_new_tokens=1024,
#             num_beams=1
#         )
#         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
#         answer = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
#         html_out += render_example(image, answer)

#     display(HTML(html_out))

# render_inference_results(peft_model, val_dataset, 4)

# saves the inference results as a jpg to track how the model is doing over time
def save_inference_results(model, dataset: DetectionDataset, count: int, save_dir: str, epoch="none"):
    os.makedirs(save_dir, exist_ok=True)
    count = min(count, len(dataset))

    for i in range(count):
        image, data = dataset.dataset[i]
        prefix = data['prefix']
        suffix = data['suffix']

        inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        answer = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)

        try:
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, answer, resolution_wh=image.size)
            image_annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image.copy(), detections)
            image_annotated = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image_annotated, detections)
            
            # saves example images of annotations
            filename = f"epoch{epoch}_{i:03d}_{detections.data['class_name']}.jpg".replace(" ", "_")
            filename = sanitize_filename(filename, platform="windows")   # cleans filename to make sure it gets saved
            filepath = os.path.join(save_dir, filename)
            image_annotated.save(filepath)
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Failed to annotate image {i}: {e}")

save_inference_results(peft_model, val_dataset, 4, save_dir="./before_training")


## Fine-tune Florence-2 on custom object detection dataset

# defines the train loop for the model
def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # save_inference_results(peft_model, val_loader.dataset, 6, save_dir="./inference_training")

    # runs model for given number of epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # for each input and answer in the train dataloader, get the ouputs and calculate the loss
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(DEVICE)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward(), optimizer.step(), lr_scheduler.step(), optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0

        # don't update the gradients when updating the weights 
        # would otherwise affect back propagation
        with torch.no_grad():
            # for each input and answer in the validation dataloader, get the ouputs and calculate the loss
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(DEVICE)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

            save_inference_results(peft_model, val_loader.dataset, 6, save_dir="./training_images_annotated", epoch=str(epoch))

        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)



EPOCHS = 10
LR = 5e-6  # learning rate

# runs the training loop to fine tune the model
train_model(train_loader, val_loader, peft_model, processor, epochs=EPOCHS, lr=LR)

## Fine-tuned model evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = f"./model_checkpoints/epoch_{EPOCHS}"  # gets the last checkpoint of the model training

model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

# # regex pattern to grab only the class names from the suffixes of annotations
# # e.g. turn "Cardiomegaly<loc_286><loc_445><loc_725><loc_789>" into "Cardiomegaly"
# PATTERN = r'(.*?)<loc_\d+>'

# # extracts the class names from the annotations jsonl file
# def extract_classes(dataset: DetectionDataset):
#     class_set = set()
#     for i in range(len(dataset.dataset)):
#         image, data = dataset.dataset[i]
#         suffix = data["suffix"]
#         classes = re.findall(PATTERN, suffix)
#         class_set.update(classes)
#     return sorted(class_set)

# CLASSES = extract_classes(train_dataset)

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
    prediction.confidence = np.ones(len(prediction))

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
confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=CLASSES
)

# saves the confusion matrix as an image
fig = confusion_matrix.plot()
fig.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')