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

print("RUNNING: FLORENCE CAPTIONING EVALUATION")

# === CONFIG ===
config_path = "./config.yaml"
config = load_config(config_path)

EPOCHS = config.get('epochs')
REVISION = config.get('revision')
BATCH_SIZE = config.get('batch_size')
NUM_WORKERS = config.get('num_workers')

OUTPUT_DIR = "./padchest/"

print("config loaded")

## loads model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = f"./model_checkpoints/epoch_{EPOCHS}"

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

# saves output to jsonl file
results_jsonl = os.path.join(OUTPUT_DIR, "caption_eval.jsonl")
results_file = open(results_jsonl, "w")

targets = []
predictions = []

### eval loop
for i in tqdm(range(len(val_dataset.dataset))):
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

    # writes to jsonl file 
    results_file.write(json.dumps({
        "image": data["image"],
        "reference": reference_caption,
        "prediction": generated_text
    }) + "\n")

results_file.close()

### metrics
smooth_fn = SmoothingFunction().method1
bleu_scores = [
    sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)
    for ref, pred in zip(targets, predictions)
]
avg_bleu = sum(bleu_scores) / len(bleu_scores)

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(targets, predictions)]
avg_rougeL = sum(rouge_scores) / len(rouge_scores)

print(f"\nEvaluation finished at {datetime.now()}")
print(f"BLEU score: {avg_bleu:.4f}")
print(f"ROUGE-L score: {avg_rougeL:.4f}")
print(f"Results written to {results_jsonl}")
