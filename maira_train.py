"""
    Trains MAIRA-2 using the VinDr-CXR dataset

    Hyperparameters have been set manually in the file since they differ from Florence quite a bit

    Requires:
        VinDr-CXR to have been downloaded and preprocessed (vindr_preprocessing_aug.py)
"""

import os
import json
import re
import torch
import yaml
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler, AutoConfig
from peft import LoraConfig, get_peft_model
from PIL import Image

def load_config(path="./config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config("./config.yaml")
OUTPUT_DIR = cfg.get("output_dir")
ANNOTATIONS_JSON_TRAIN = f"{OUTPUT_DIR}/train/annotations.jsonl"
ANNOTATIONS_JSON_VAL = f"{OUTPUT_DIR}/valid/annotations.jsonl"
IMAGE_DIR_TRAIN = f"{OUTPUT_DIR}/train/"
IMAGE_DIR_VAL = f"{OUTPUT_DIR}/valid/"
BATCH_SIZE = 1
NUM_WORKERS = 0
EPOCHS = 3
LR = 5e-6
USE_PEFT = True
# REVISION = cfg.get("revision", None)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# converts florence suffixes to maira grounded sequences
LOC_RE = re.compile(r"<loc_(\d+)>", flags=re.I)  # finds florence loc tags
def suffix_to_maira_grounded(suffix: str) -> str:
    if not suffix or not suffix.strip():
        return ""  # empty target (no findings)
    parts = suffix.strip().split()  # florence uses spaces between findings
    out_objs = []
    i = 0
    while i < len(parts):
        token = parts[i]
        if "<loc_" in token and not re.match(r".*<loc_\d+>", token):
            i += 1
            continue
        locs = LOC_RE.findall(token)
        label = LOC_RE.sub("", token).strip()
        if label == "":
            i += 1
            continue
        j = i + 1
        while len(locs) < 4 and j < min(len(parts), i + 6):
            more = LOC_RE.findall(parts[j])
            if more:
                locs.extend(more)
                j += 1
            else:
                break
        i = j
        if len(locs) >= 4:
            x1, y1, x2, y2 = locs[:4]
            out_objs.append(f"<obj> {label}<box><x{x1}><y{y1}><x{x2}><y{y2}></box></obj>")
        else:
            out_objs.append(f"<obj> {label}</obj>")
    return "".join(out_objs)

# creates a dataset for maira
# returns an image and a prompted target_text
# does not tokenise here
class MairaReportingDataset(Dataset):
    def __init__(self, annotations_jsonl_path, image_dir):
        self.items = []
        with open(annotations_jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                self.items.append(obj)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")

        # builds maira-style grounded target
        grounded = suffix_to_maira_grounded(item.get("suffix", ""))

        # dont need florence-2 prefix (<OD>) for maira
        target_text = grounded.strip()

        return {"image": image, "target_text": target_text, "image_name": item.get("image")}

# collates the inputs properly
def make_collate_fn(processor, max_length=2048):
    def collate_fn(batch):
        all_pixel_values = []
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for ex in batch:
            image = ex["image"]
            target_text = ex["target_text"] or ""

            # returns batched tensors
            base = processor.format_and_preprocess_reporting_input(
                current_frontal=image,
                current_lateral=None,
                prior_frontal=None,
                indication=None,
                technique=None,
                comparison=None,
                prior_report=None,
                return_tensors="pt",
                get_grounding=True,
            )

            # squeeze batch dim
            base = {k: v.squeeze(0) for k, v in base.items()}

            # base should contain 'input_ids' that include image token placeholders 
            base_input_ids = base["input_ids"]
            base_attention_mask = base.get("attention_mask", torch.ones_like(base_input_ids))

            # tokenise target text and add EOS token
            tgt_tokenized = processor.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
            )
            tgt_ids = tgt_tokenized.input_ids.squeeze(0) if tgt_tokenized.input_ids.numel() else torch.tensor([], dtype=torch.long)
            
            # adds EOS token to target
            if processor.tokenizer.eos_token_id is not None:
                eos_token = torch.tensor([processor.tokenizer.eos_token_id], dtype=torch.long)
                tgt_ids = torch.cat([tgt_ids, eos_token], dim=0)

            # concatenate base prompt and target
            full_input_ids = torch.cat([base_input_ids, tgt_ids], dim=0)
            full_attention_mask = torch.cat([base_attention_mask, torch.ones_like(tgt_ids)], dim=0)

            # set base prompt tokens to -100 (ignore in loss)
            labels = full_input_ids.clone()
            labels[:len(base_input_ids)] = -100  # mask base prompt
            # keep target text tokens for training

            pixel_values = base["pixel_values"]

            all_pixel_values.append(pixel_values)
            all_input_ids.append(full_input_ids)
            all_attention_masks.append(full_attention_mask)
            all_labels.append(labels)

        # pad sequences to max length in batch but cap at max_length 
        batch_max_len = min(max(len(x) for x in all_input_ids), max_length)

        padded_input_ids = torch.full((len(all_input_ids), batch_max_len), 
                                    fill_value=processor.tokenizer.pad_token_id, dtype=torch.long)
        padded_attention_mask = torch.zeros((len(all_attention_masks), batch_max_len), dtype=torch.long)
        padded_labels = torch.full((len(all_labels), batch_max_len), fill_value=-100, dtype=torch.long)

        for i, (ids, att, lbl) in enumerate(zip(all_input_ids, all_attention_masks, all_labels)):
            L = min(ids.size(0), batch_max_len)
            padded_input_ids[i, :L] = ids[:L]
            padded_attention_mask[i, :L] = att[:L]
            padded_labels[i, :L] = lbl[:L]

        # labels: copy of input_ids but masked where attention_mask == 0 (pad) -> -100 
        padded_labels[padded_attention_mask == 0] = -100

        pixel_values_batch = torch.stack(all_pixel_values, dim=0)

        # move to the device (cuda, cpu, etc.)
        pixel_values_batch = pixel_values_batch.to(DEVICE)
        padded_input_ids = padded_input_ids.to(DEVICE)
        padded_attention_mask = padded_attention_mask.to(DEVICE)
        padded_labels = padded_labels.to(DEVICE)

        return {
            "pixel_values": pixel_values_batch,
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,  # Now properly masked
            "image_names": [b.get("image_name") for b in batch],
        }
    return collate_fn


### loads maira
CHECKPOINT = "microsoft/maira-2"
print("Loading MAIRA and processor...")

config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
config.vision_config.model_type = 'davit'
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

# freeze image encoder
for name, param in model.named_parameters():
    if any(k in name.lower() for k in ("vision", "image", "encoder", "vit", "rad", "dino")):
        param.requires_grad = False

# applies lora to reduce number of trainable params
if USE_PEFT:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

### forms dataset and dataloaders
train_ds = MairaReportingDataset(ANNOTATIONS_JSON_TRAIN, IMAGE_DIR_TRAIN)
val_ds = MairaReportingDataset(ANNOTATIONS_JSON_VAL, IMAGE_DIR_VAL)

collate_fn = make_collate_fn(processor, max_length=2048)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)


optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

### training loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - train"):
            optimizer.zero_grad()
            outputs = model(**{k: v for k, v in batch.items() if k not in ("image_names",)})
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
        avg_train = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} train loss: {avg_train:.4f}")

        # quick validation loss (no grad)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - val"):
                outputs = model(**{k: v for k, v in batch.items() if k not in ("image_names",)})
                val_loss += outputs.loss.item()
            avg_val = val_loss / max(1, len(val_loader))
            print(f"Epoch {epoch+1} val loss: {avg_val:.4f}")
        model.train()

        # checkpoint
        out_dir = f"./maira_checkpoints/epoch_{epoch+1}"
        os.makedirs(out_dir, exist_ok=True)
        # save model and peft adapters
        model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)
        print(f"Saved checkpoint to {out_dir}")

if __name__ == "__main__":
    train()
