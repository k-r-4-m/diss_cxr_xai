"""
    Evaluates MAIRA-2's textual output on the PadChest-GR dataset. 

    Outputs the following:
        BLEU-1 (1-gram precision)
        BLEU-4 (4-gram precision)
        ROUGE-L (longest common subsequence)

    REQUIRES TRANSFORMERS==4.51.3 !!!
"""

import os
import re
import json
import yaml
import string
from collections import defaultdict
from difflib import get_close_matches
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from huggingface_hub import login
import supervision as sv
from supervision.detection.utils import box_iou_batch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathvalidate import sanitize_filename
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

print("RUNNING: MAIRA TEXTUAL EVALUATION")

OUTPUT_DIR = './padchest/'
ANNOTATIONS_JSON = f'{OUTPUT_DIR}/annotations.jsonl'
IMAGE_DIR = f'{OUTPUT_DIR}/images'

# logs into huggingface
# token is stored in a seperate yaml file called token_file.yaml
# please place your own token in token_file.yaml
with open("./token_file.yaml", 'r') as f:
    token_file = yaml.safe_load(f)
    hf_token = token_file.get('hf_token')
login(hf_token)

df = pd.read_csv(ANNOTATIONS_CSV)
CLASSES = df['class_name'].unique().tolist()
if "No finding" in CLASSES:
    CLASSES.remove("No finding")

### loads model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "microsoft/maira-2"
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

model = model.eval()
model = model.to(DEVICE)

# save the output to a jsonl file
results_jsonl = os.path.join(OUTPUT_DIR, "maira_caption_eval.jsonl")
results_file = open(results_jsonl, "w")

# smoothing function for BLEU
smooth_fn = SmoothingFunction().method1

# initialises the rouge scoring function
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# normalises the text for BLEU
def tokenise(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # removes punctuation
    return text.split()

targets = []
predictions = []

with open(ANNOTATIONS_JSON, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)

        # gets image and converts to RGB
        img_path = os.path.join(IMAGE_DIR, item["image"])
        image = Image.open(img_path).convert("RGB")

        # removes <loc_xxx> tags if present
        reference_caption = re.sub(r"<loc_\d+>", "", item.get("suffix", "")).strip()
        if not reference_caption:
            continue

        # model forward
        # there's only have frontal x-rays with padchest
        inputs = processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            indication=None,
            technique=None,
            comparison=None,
            prior_report=None,
            return_tensors="pt",
            get_grounding=False,
        ).to(DEVICE)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                use_cache=True,
                early_stopping=True,
                num_beams=3
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_text = processor.decode(out_ids[0][prompt_len:], skip_special_tokens=True).strip()

        predictions.append(generated_text)
        targets.append(reference_caption)

        # tokenises the text
        ref_tokens = tokenise(reference_caption)
        pred_tokens = tokenise(generated_text)

        # calculates BLEU scores
        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
        
        # calculates ROUGE-L score
        rougeL = scorer.score(reference_caption, generated_text)["rougeL"].fmeasure

        # writes everything to jsonl
        results_file.write(json.dumps({
            "image": item["image"],
            "reference": reference_caption,
            "prediction": generated_text,
            "bleu1": bleu1,
            "bleu4": bleu4,
            "rougeL": rougeL
        }) + "\n")

results_file.close()

### metrics
bleu1_scores = [
    sentence_bleu([tokenise(ref)], tokenise(pred), weights=(1,0,0,0), smoothing_function=smooth_fn)
    for ref, pred in zip(targets, predictions)
]
bleu4_scores = [
    sentence_bleu([tokenise(ref)], tokenise(pred), weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth_fn)
    for ref, pred in zip(targets, predictions)
]

avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores)

rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(targets, predictions)]
avg_rougeL = sum(rouge_scores) / len(rouge_scores)

print(f"\nEvaluation finished at {datetime.now()}")
print(f"BLEU-1 score: {avg_bleu1:.4f}")
print(f"BLEU-4 score: {avg_bleu4:.4f}")
print(f"ROUGE-L score: {avg_rougeL:.4f}")
print(f"Results written to {results_jsonl}")