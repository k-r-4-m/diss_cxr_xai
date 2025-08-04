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
from  supervision.detection.utils import box_iou_batch
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

## Fine-tuned model evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = f"./model_checkpoints/epoch_{EPOCHS}"  # gets the last checkpoint of the model training

config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)
config.vision_config.model_type = 'davit'
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, config=config, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

def run_example(task_prompt, text_input=None, image=None):
    if image is None:
        print("No image passed")
        return

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    print(f"Prompt {prompt}")
    inputs = processor(text=prompt,
                       images=image,
                       return_tensors="pt").to('cuda', torch.float32)

    print(f"Tokenized input text: {processor.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)}")

    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
      output_scores=True,
      return_dict_in_generate=True
    )
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=False)[0]
    print(f"Generated text: {generated_text}")
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

image = Image.open("000d68e42b71d3eac10ccc077aba07c1.png")
print(image)

task = '<CAPTION>'
answer = run_example(task, image=image)
print(answer)

task = '<DETAILED_CAPTION>'
answer = run_example(task, image=image)
print(answer)

task = '<MORE_DETAILED_CAPTION>'
answer = run_example(task, image=image)
print(answer)
