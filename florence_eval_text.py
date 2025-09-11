"""
    Evaluates the performance of Florence on the PadChest-GR dataset for textual radiology report generation

    Outputs the following:
        BLEU-1 (1-gram precision)
        BLEU-4 (4-gram precision)
        ROUGE-L (longest common subsequence)

    Requires:
        Florence to have been pretrained with a model checkpoint stored in florence_checkpoints/epoch_n
        PadChest to have been downloaded and preprocessed
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import json
from florence_tools import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from florence_tools import load_config
import re

print("RUNNING: FLORENCE CAPTIONING EVALUATION")

config_path = "./config.yaml"
config = load_config(config_path)

EPOCHS = config.get('epochs')
REVISION = config.get('revision')
BATCH_SIZE = config.get('batch_size')
NUM_WORKERS = config.get('num_workers')

OUTPUT_DIR = "./padchest/"

print("config loaded")

### loads model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = f"./florence_checkpoints/epoch_{EPOCHS}"

config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)
config.vision_config.model_type = 'davit'
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, config=config, trust_remote_code=True, revision=REVISION
).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)

val_dataset = DetectionDataset(
    jsonl_file_path=f"{OUTPUT_DIR}/annotations.jsonl",
    image_directory_path=f"{OUTPUT_DIR}/images/"
)

# save the output to jsonl file
results_jsonl = os.path.join(OUTPUT_DIR, "florence_caption_eval.jsonl")
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

### eval loop
for i in tqdm(range(len(val_dataset.dataset))):
# for i in range(10):
    image, data = val_dataset.dataset[i]

    # florence needs RGB images, images in padchest are greyscale
    if image.mode != "RGB":
        image = image.convert("RGB")  

    prefix = "<MORE_DETAILED_CAPTION>"  # florence task
    reference_caption = data["suffix"]  # reference caption

    # model forward
    inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
        "image": data["image"],
        "reference": reference_caption,
        "prediction": generated_text,
        "bleu1": bleu1,
        "bleu4": bleu4,
        "rougeL": rougeL
    }) + "\n")

results_file.close()

### metrics
bleu1_scores = [
    sentence_bleu([ref.split()], pred.split(), weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
    for ref, pred in zip(targets, predictions)
]
bleu4_scores = [
    sentence_bleu([ref.split()], pred.split(), weights=(1./4., 1./4., 1./4., 1./4.), smoothing_function=smooth_fn)
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
