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
import matplotlib.patches as patches


def plot_bbox(image, data):
   # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # # Show the plot
    # plt.show()

    filename = "example.png"
    plt.savefig(filename, dpi=300)

# # loads the config file for epochs, revision, pathnames, etc.
# config_path = "./config.yaml"
# config = load_config(config_path)

# EPOCHS = config.get('epochs')
# REVISION = config.get('revision')
# print("config loaded")

EPOCHS = 50
REVISION = 'refs/pr/24'  # revision of florence-2 that fixes the GenerationMixin import error

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

image = Image.open("4007175543191290349892200982275727462_qwczh2.png").convert('RGB')
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

# print("\nCombo: ")
# task_prompt = '<DETAILED_CAPTION>'
# results = run_example(task_prompt, image=image)
# text_input = results[task_prompt]
# task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
# results = run_example(task_prompt, text_input, image)
# results['<DETAILED_CAPTION>'] = text_input
# print(results)

# plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])

print("\nOD: ")
task_prompt = '<OD>'
results = run_example(task_prompt, image=image)
print(results)
plot_bbox(image, results['<OD>'])