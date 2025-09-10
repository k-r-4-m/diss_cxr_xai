"""
    Visualises the outputs of MAIRA-2 alongside the ground truth using bounding boxes

    Requires:
        A chest X-Ray file name *that is present in the validation dataset* given as an argument when running the file
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
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from pathvalidate import sanitize_filename
import sys
import colorsys

print("RUNNING MAIRA VISUALISATIONS")

# loads the config.yaml file for model configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# loads the config file for epochs, revision, pathnames, etc.
config_path = "./config.yaml"
config = load_config(config_path)

REVISION = config.get('revision')
EPOCHS = config.get('epochs')
DICOM_DIR = config.get('dicom_dir')
ANNOTATIONS_CSV = config.get('annotations_csv')
OUTPUT_DIR = config.get('output_dir')
print("config loaded")

ANNOTATIONS_JSONL = f"{OUTPUT_DIR}/valid/annotations.jsonl"
IMAGE_PATH = f"{OUTPUT_DIR}/valid"

### loads model
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "microsoft/maira-2"
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

model = model.eval()
model = model.to(DEVICE)

# gets a list of all the classes in the dataset
df = pd.read_csv(ANNOTATIONS_CSV)
CLASSES = df['class_name'].unique().tolist()
CLASSES.remove("No finding")  # removes no finding from the classes list

CLASS_COLOURS = {
    "aortic enlargement": "#ff0000",
    "atelectasis": "#1e77b4",
    "calcification": "#ff7f0e",
    "cardiomegaly": "#008000",
    "consolidation": "#9366bd",
    "ild": "#8c564b",
    "infiltration": "#efb3dd",
    "lung opacity": "#a52a2a",
    "nodule/mass": "#aec7e8",
    "other lesion": "#FFDA0A",
    "pleural effusion": "#808000",
    "pleural thickening": "#06fafa",
    "pneumothorax": "#ffba78",
    "pulmonary fibrosis": "#c5b0d4"
}


# a dictionary to map synonyms for classes
# this is because MAIRA-2 will refer to classes by different names/terms
# not all classes will have synonyms
CLASS_SYNONYMS = {
    "aortic enlargement": ["tortuous aorta", "enlarged aorta", "widened aortic contour", "aortic elongation", "aorta is tortuous", "aorta is markedly tortuous"],
    "atelectasis": [],
    "calcification": ["calcified"],
    "cardiomegaly": ["enlarged heart", "big heart", "increased cardiac silhouette", "heart size is enlarged", "heart size is slightly enlarged", "cardiac silhouette is mildly enlarged"],
    "consolidation": ["lung consolidation", "consolidative opacity"],
    "ild": ["interstitial lung disease", "interstitial prominence", "interstitial lung"],
    "infiltration": ["infiltrates", "infiltrate"],
    "lung opacity": ["increased density in the"],
    "nodule/mass": ["mass", "nodule", "calcified granuloma", "nodular density", "granulomas"],
    "other lesion": ["subcutaneous emphysema"],
    "pleural effusion": ["pleural fluid", "effusion", "effusions", "fluid in pleural space"],
    "pleural thickening": ["thickening"],
    "pneumothorax": ["collapsed lung", "air in pleural space"],
    "pulmonary fibrosis": ["interstitial fibrotic"],
}

# makes both CLASSES and CLASS_SYNONYMS lowercase
CLASS_CANON = {c.lower(): c for c in CLASSES}
CLASS_SYNONYMS_LOWER = {k.lower(): [s.lower() for s in v] for k, v in CLASS_SYNONYMS.items()}

# regex pattern to capture the label and the loc tags
GT_PATTERN = re.compile(
    r"([A-Za-z0-9 _\-/]+?)\s*<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
)

# clamps bounding boxes
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# parses the ground truth suffixess
def parse_suffix_to_detections(suffix_text: str, classes: List[str], img_size: Tuple[int, int]) -> sv.Detections:
    if not suffix_text or not suffix_text.strip():
        return sv.Detections.empty()

    W, H = img_size
    xyxy = []
    class_ids = []
    confs = []

    for m in GT_PATTERN.finditer(suffix_text):
        raw_label = m.group(1).strip()
        try:
            x1n, y1n, x2n, y2n = map(int, m.group(2, 3, 4, 5))
        except ValueError:
            continue

        # converts label to class_id
        if raw_label in classes:
            cid = classes.index(raw_label)
        else:
            # skips unknown label
            continue

        # scale from 0..1000 to pixel coords
        x1 = clamp((x1n / 1000.0) * W, 0, W)
        y1 = clamp((y1n / 1000.0) * H, 0, H)
        x2 = clamp((x2n / 1000.0) * W, 0, W)
        y2 = clamp((y2n / 1000.0) * H, 0, H)
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        xyxy.append([x_min, y_min, x_max, y_max])
        class_ids.append(cid)
        confs.append(1.0)

    if not xyxy:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=float),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confs, dtype=float),
    )

# handles cases where maira-2 repeats findings cyclically
def clean_repetitive_generation(decoded_text):
    max_reps = 3  # max number of times a finding can be repeated before it's repetitive

    # checks for repetitive <obj>...</obj> patterns
    obj_pattern = re.compile(r'<obj>(.*?)<box>(.*?)</box></obj>')
    matches = obj_pattern.findall(decoded_text)

    print(f"matches: {matches}")
    
    if not matches:
        return decoded_text
    
    # count occurrences of each finding text
    finding_counts = {}
    for finding_text, box_coords in matches:
        print(f"Finding text: {finding_text}")
        finding_text = finding_text.strip()
        finding_counts[finding_text] = finding_counts.get(finding_text, 0) + 1
    
    # if any finding appears more than max_reps times, it's likely repetitive
    repetitive_findings = [finding for finding, count in finding_counts.items() if count > max_reps]
    
    if repetitive_findings:
        print(f"detected repetitive generation for: {repetitive_findings}")
        
        # keep only the first few occurrences of each repetitive finding
        seen_count = {}
        cleaned_parts = []
        
        for match in obj_pattern.finditer(decoded_text):
            finding_text = match.group(1).strip()
            if finding_text in repetitive_findings:
                seen_count[finding_text] = seen_count.get(finding_text, 0) + 1
                if seen_count[finding_text] <= max_reps:
                    cleaned_parts.append(match.group(0))
            else:
                cleaned_parts.append(match.group(0))
        
        return ''.join(cleaned_parts)
    
    return decoded_text

# normalise the maira processer outputs to a list
def _coerce_maira_items(maira_output):
    items = []
    if maira_output is None:
        return items

    for it in maira_output:
        if isinstance(it, tuple) and len(it) == 2:
            items.append(it)
        elif isinstance(it, str):
            items.append((it, None))
        elif isinstance(it, dict):
            # seen the shapes: {"text": "...", "boxes": [[...], ...]} or similar
            txt = it.get("text") or it.get("finding") or it.get("label") or ""
            boxes = it.get("boxes") or it.get("bboxes") or it.get("bbox") or None
            # makes sure boxes is list of 4-lists if present
            if boxes is not None and isinstance(boxes, (list, tuple)):
                if len(boxes) == 4 and all(isinstance(x, (int, float)) for x in boxes):
                    boxes = [boxes]
            items.append((txt, boxes))
        # skip unknown shapes
    return items

# patterns for when a finding is mentioned but the model says it isnt present
NEGATION_PATTERNS = [
    r"\bno\b", r"\bwithout\b", r"\bnot seen\b", r"\babsent\b",
    r"\bfree of\b", r"\bclear\b", r"\bunremarkable\b", r"\bnormal\b",
    r"\bno evidence of\b", r"\bno signs of\b", r"\bnegative for\b",
]

# returns true if a finding contains a phrase in NEGATION_PATTERNS
def is_negated(text: str) -> bool:
    text_lower = text.lower().strip()
    return any(re.search(pat, text_lower) for pat in NEGATION_PATTERNS)

# maps the synonyms/variations of findings found in CLASS_SYNONYMS back to the real class names
def normalise_finding(text: str) -> str:

    if not text or not isinstance(text, str):
        return None

    text_lc = text.lower().strip()  # make the text lowercase and strip whitespace
    # remove punctuation but keep slashes/hyphens because some class names use them (nodule/mass)
    text_lc = text_lc.translate(str.maketrans('', '', string.punctuation.replace('/', '').replace('-', '')))

    # first checks for exact class name matches
    for c_lc, canonical in CLASS_CANON.items():
        if c_lc in text_lc:
            return canonical
    
    # otherwise check if a full synonym phrase appears
    for real_lc, syns in CLASS_SYNONYMS_LOWER.items():
        for syn in syns:
            if syn in text_lc:
                return CLASS_CANON.get(real_lc, None)
            
    # final check to check each word for synonyms
    text_words = set(re.findall(r'\b\w+\b', text_lc))
    for real_lc, syns in CLASS_SYNONYMS_LOWER.items():
        for syn in syns:
            for w in re.findall(r'\b\w+\b', syn):
                if len(w) > 3 and w in text_words:
                    if w == "tortuous":  # special case for "aortic"
                        # only map to aortic enlargement if aortic context present
                        if any(ctx in text_words for ctx in ["aorta", "aortic", "thoracic"]):
                            return CLASS_CANON.get(real_lc, None)
                        continue
                    return CLASS_CANON.get(real_lc, None)

    return None

# extracts individual findings from maira output
def extract_findings_from_maira_text(text: str):
    # regex pattern to mwatch <obj> tags
    obj_pattern = r'<obj>(.*?)</obj>'
    findings = re.findall(obj_pattern, text, re.IGNORECASE | re.DOTALL)
    
    # cleans up the findings
    cleaned_findings = []
    for finding in findings:
        # remove any more tags and clean whitespace
        clean_finding = re.sub(r'<[^>]+>', '', finding).strip()
        if clean_finding:
            cleaned_findings.append(clean_finding)
    
    return cleaned_findings

# makes sure values are pixels and not normalised values
def _to_pixel_box(b, W, H):
    x1, y1, x2, y2 = [float(v) for v in b]

    mx = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if mx <= 1.0 + 1e-6:
        # 0..1
        x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
    elif mx <= 1000.0 + 1e-6:
        # 0..1000
        x1, y1, x2, y2 = (x1 / 1000.0) * W, (y1 / 1000.0) * H, (x2 / 1000.0) * W, (y2 / 1000.0) * H
    # else: already pixels

    x_min, x_max = sorted([clamp(x1, 0, W), clamp(x2, 0, W)])
    y_min, y_max = sorted([clamp(y1, 0, H), clamp(y2, 0, H)])
    if x_max - x_min < 1: x_max = x_min + 1
    if y_max - y_min < 1: y_max = y_min + 1
    return [x_min, y_min, x_max, y_max]

# converts the MAIRA-2 output into a Detections object
# MAIRA-2 output is quite different than Florence-2, explicit class names aren't given
# need to extract any class names mentioned in each part of the report
# def parse_maira_prediction_to_detections(decoded_text: str, img_size: Tuple[int, int], processor) -> sv.Detections:
#     if not decoded_text or not decoded_text.strip():
#         return sv.Detections.empty()

#     W, H = img_size
#     xyxy, class_ids, confs = [], [], []
#     unlocalised = []

#     # for each <obj> block
#     for obj in re.findall(r"<obj>(.*?)</obj>", decoded_text, flags=re.DOTALL | re.IGNORECASE):
#         text_block = obj.strip()

#         # skip negated findings
#         if is_negated(text_block):
#             print(f"Skipping negated finding: {text_block}")
#             continue

#         # looks for <box> tags
#         box_match = re.search(r"<box><x(\d+)><y(\d+)><x(\d+)><y(\d+)>", text_block)
#         clean_text = re.sub(r"<box>.*?</box>", "", text_block).strip()
#         mapped = normalise_finding(clean_text)

#         # skip findings not known
#         if mapped is None:
#             continue

#         cid = CLASSES.index(mapped)

#         if box_match:
#             # extract the box, normalised in florence-style (i.e. 0-1000)
#             x1, y1, x2, y2 = map(int, box_match.groups())
#             # converts cords to 0-1 for maira
#             norm_box = [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
#             # adjust the box back to the original image using the processor
#             adjusted = processor.adjust_box_for_original_image_size(norm_box, width=W, height=H)
#             # convert it to pixel cords
#             px_box = [adjusted[0] * W, adjusted[1] * H, adjusted[2] * W, adjusted[3] * H]

#             xyxy.append(px_box)
#             class_ids.append(cid)
#             confs.append(1.0)
#         else:
#             # prediction is unlocalised
#             unlocalised.append(mapped)

#     if not xyxy:
#         dets = sv.Detections.empty()
#     else:
#         dets = sv.Detections(
#             xyxy=np.array(xyxy, dtype=float),
#             class_id=np.array(class_ids, dtype=int),
#             confidence=np.array(confs, dtype=float),
#         )
#         dets.data['class_name'] = [CLASSES[cid] for cid in dets.class_id]

#     dets.data['unlocalised'] = unlocalised
#     return dets


def parse_maira_prediction_to_detections(maira_output_or_text, img_size: Tuple[int, int], processor=None) -> sv.Detections:
    """
    Parse MAIRA-2 output (either raw decoded text or processor-converted list)
    and return sv.Detections aligned to the original image.

    - Supports either a string (decoded_text) or the structured list returned by
      processor.convert_output_to_plaintext_or_grounded_sequence(...).
    - Correctly handles MAIRA bin coordinates (num_box_coord_bins, default 100).
    - Maps square-normalised coords back to the original rectangular image by
      accounting for the padding used to form the square (side = max(W,H)).
    """
    if maira_output_or_text is None:
        return sv.Detections.empty()

    W, H = img_size  # note: image.size = (width, height)

    # Try to get number of bins (default 100)
    num_bins = None
    try:
        num_bins = int(getattr(processor, "num_box_coord_bins", 100))
    except Exception:
        num_bins = 100

    # If user passed decoded text (string), convert it to grounded sequence if possible
    if isinstance(maira_output_or_text, str):
        if processor is not None:
            maira_items = processor.convert_output_to_plaintext_or_grounded_sequence(maira_output_or_text)
        else:
            # fallback: we will parse <obj>...<box><x..>... style directly below
            maira_items = None
    else:
        maira_items = maira_output_or_text

    xyxy = []
    class_ids = []
    confs = []
    unlocalised = []

    side = max(W, H)
    pad_x = (side - W) / 2.0
    pad_y = (side - H) / 2.0

    def _norm_from_raw_value(v):
        """Convert a single raw coordinate to a normalized value relative to MAIRA square (0..1)."""
        v = float(v)
        if -1e-9 <= v <= 1.0 + 1e-9:
            # already normalized
            return max(0.0, min(1.0, v))
        # If value looks like a bin index (e.g. 55 for 0.555)
        if 0 <= v <= num_bins + 0.5:
            return (v + 0.5) / float(num_bins)
        # If value looks like Florence 0..1000 encoding
        if v <= 1000.0:
            # here we interpret v as 0..1000 coordinate relative to original image:
            # convert to original pixel then to square-normalized
            # for x coordinates scale by W, for y by H  but we don't know which coord this is here.
            # The caller will handle mapping x vs y separately using W/H.
            return v / 1000.0
        # Fallback: treat as pixel coordinate on original image -> convert to square-normalized
        # This will be handled by caller because we need to know axis size (W or H).
        return v  # leave raw; caller will handle if >1

    # If we have structured items, they will be like: [("text", [(x,y,x2,y2), ...]), ...]
    if maira_items is not None:
        for entry in maira_items:
            # entry can be tuple (text, boxes) or a plain string
            if isinstance(entry, tuple) and len(entry) == 2:
                finding_text, boxes = entry
            elif isinstance(entry, str):
                finding_text, boxes = entry, None
            elif isinstance(entry, dict):
                # handle shapes like {"text": "...", "boxes": [[...], ...]}
                finding_text = entry.get("text") or entry.get("finding") or entry.get("label") or ""
                boxes = entry.get("boxes") or entry.get("bboxes") or entry.get("bbox") or None
            else:
                continue

            if not isinstance(finding_text, str) or not finding_text.strip():
                continue
            if is_negated(finding_text):
                continue

            mapped = normalise_finding(finding_text)
            if mapped is None:
                # unlocalised mention (or unknown), keep as unlocalised
                if finding_text.strip():
                    unlocalised.append(finding_text.strip())
                continue

            cid = CLASSES.index(mapped)

            if boxes is None:
                unlocalised.append(mapped)
                continue

            for b in boxes:
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    continue

                # each b might be floats already normalized (0..1),
                # or might be bin integers (e.g. 55),
                # or might be 0..1000 integers.
                bx = list(b)
                # detect and convert:
                mx = max(abs(v) for v in bx)
                if mx <= 1.0 + 1e-9:
                    norm_x1, norm_y1, norm_x2, norm_y2 = bx
                elif mx <= num_bins + 1e-6:
                    # bin indices: convert to normalized value using num_bins
                    norm_x1 = (bx[0] + 0.5) / float(num_bins)
                    norm_y1 = (bx[1] + 0.5) / float(num_bins)
                    norm_x2 = (bx[2] + 0.5) / float(num_bins)
                    norm_y2 = (bx[3] + 0.5) / float(num_bins)
                elif mx <= 1000.0 + 1e-6:
                    # Florence-style 0..1000 relative to original image
                    # convert to pixel on original image, then to square normalized:
                    px_x1 = (bx[0] / 1000.0) * W
                    px_y1 = (bx[1] / 1000.0) * H
                    px_x2 = (bx[2] / 1000.0) * W
                    px_y2 = (bx[3] / 1000.0) * H
                    # pixel on square = pixel_orig + pad
                    sx1 = px_x1 + pad_x
                    sy1 = px_y1 + pad_y
                    sx2 = px_x2 + pad_x
                    sy2 = px_y2 + pad_y
                    norm_x1, norm_y1, norm_x2, norm_y2 = sx1 / side, sy1 / side, sx2 / side, sy2 / side
                else:
                    # assume these are pixel coords on original image already
                    px_x1, px_y1, px_x2, px_y2 = bx
                    sx1 = px_x1 + pad_x
                    sy1 = px_y1 + pad_y
                    sx2 = px_x2 + pad_x
                    sy2 = px_y2 + pad_y
                    norm_x1, norm_y1, norm_x2, norm_y2 = sx1 / side, sy1 / side, sx2 / side, sy2 / side

                # Map square-normalized to pixel on square to pixel on original
                sq_x1 = norm_x1 * side
                sq_y1 = norm_y1 * side
                sq_x2 = norm_x2 * side
                sq_y2 = norm_y2 * side

                px_x1 = sq_x1 - pad_x
                px_y1 = sq_y1 - pad_y
                px_x2 = sq_x2 - pad_x
                px_y2 = sq_y2 - pad_y

                # Clip coordinates to image bounds (float)
                px_x1 = max(0.0, min(W, px_x1))
                px_y1 = max(0.0, min(H, px_y1))
                px_x2 = max(0.0, min(W, px_x2))
                px_y2 = max(0.0, min(H, px_y2))

                xyxy.append([px_x1, px_y1, px_x2, px_y2])
                class_ids.append(cid)
                confs.append(1.0)

    else:
        # fallback: parse raw decoded_text with <obj> and <box><x..> tags (if processor is not given)
        decoded_text = maira_output_or_text
        for obj in re.findall(r"<obj>(.*?)</obj>", decoded_text, flags=re.DOTALL | re.IGNORECASE):
            text_block = obj.strip()
            if is_negated(text_block):
                continue
            # find box tags with digits inside
            box_match = re.search(r"<box><x(\d+)><y(\d+)><x(\d+)><y(\d+)>", text_block)
            clean_text = re.sub(r"<box>.*?</box>", "", text_block).strip()
            mapped = normalise_finding(clean_text)
            if mapped is None:
                continue
            cid = CLASSES.index(mapped)
            if box_match:
                bvals = list(map(int, box_match.groups()))
                # Interpret as bins (MAIRA uses bin tags)
                norm_x1 = (bvals[0] + 0.5) / float(num_bins)
                norm_y1 = (bvals[1] + 0.5) / float(num_bins)
                norm_x2 = (bvals[2] + 0.5) / float(num_bins)
                norm_y2 = (bvals[3] + 0.5) / float(num_bins)

                # Map to pixels via square
                sq_x1, sq_y1 = norm_x1 * side, norm_y1 * side
                sq_x2, sq_y2 = norm_x2 * side, norm_y2 * side

                px_x1 = max(0.0, min(W, sq_x1 - pad_x))
                px_y1 = max(0.0, min(H, sq_y1 - pad_y))
                px_x2 = max(0.0, min(W, sq_x2 - pad_x))
                px_y2 = max(0.0, min(H, sq_y2 - pad_y))

                xyxy.append([px_x1, px_y1, px_x2, px_y2])
                class_ids.append(cid)
                confs.append(1.0)
            else:
                unlocalised.append(mapped)

    if not xyxy:
        dets = sv.Detections.empty()
    else:
        dets = sv.Detections(
            xyxy=np.array(xyxy, dtype=float),
            class_id=np.array(class_ids, dtype=int),
            confidence=np.array(confs, dtype=float),
        )
        dets.data['class_name'] = [CLASSES[cid] for cid in dets.class_id]

    dets.data['unlocalised'] = unlocalised
    return dets



try:
    input_image = sys.argv[1]
except IndexError:
    print("Please provide an image!")
    sys.exit()

prefix = '<OD>'
suffix = None

# scrubs the jsonl file to find the annotations
with open(ANNOTATIONS_JSONL, "r") as f:
    for line in f:
        row = json.loads(line)
        if row["image"] == input_image:
            suffix = row["suffix"]
            break  # stops after finding the match

if suffix == None:
    print("Error finding annotation! Are you sure you provided a valid image file?")
    sys.exit()
else:
    image = Image.open(f"{IMAGE_PATH}/{input_image}").convert('RGB')
    W, H = image.size

    gt = parse_suffix_to_detections(suffix, CLASSES, (W, H))
    target = sv.Detections(
        xyxy=gt.xyxy,
        class_id=gt.class_id,
        confidence=gt.confidence,
        data={'class_name': [CLASSES[cid] for cid in gt.class_id]}
    )

    # model forward
    # there's only have frontal x-rays with vindr
    inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=image,
        current_lateral=None,
        prior_frontal=None,
        indication=None,
        technique=None,
        comparison=None,
        prior_report=None,
        return_tensors="pt",
        get_grounding=True,
    ).to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,
            early_stopping=True,
            num_beams=3,
            # min_new_tokens=50,
            # do_sample=True,
            # pad_token_id=processor.tokenizer.eos_token_id,  # ensures proper padding
            # eos_token_id=processor.tokenizer.eos_token_id,  # ensures proper termination
        )

    prompt_len = inputs["input_ids"].shape[-1]
    decoded = processor.decode(out_ids[0][prompt_len:], skip_special_tokens=True).lstrip()

    print(f"raw decoded: {decoded}")

    # language generation sometimes goes into a loop
    # if output is larger than 1000 characters, something is most likely wrong
    if len(decoded) > 1000:
        print("long decoded item")
        # cleans output from repetitive generations
        decoded = clean_repetitive_generation(decoded)
        
    maira_output = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
    # expected: list of (finding_text, [(x1,y1,x2,y2)] or None)

    print(f"maira out: {maira_output}")

    prediction = parse_maira_prediction_to_detections(decoded, (W, H), processor)
    prediction.data['class_name'] = [CLASSES[cid] for cid in prediction.class_id]

    image_np = np.array(image)


def class_base_colour(name: str) -> tuple:
    """Return consistent base colour (RGB tuple in 0-1) for a class."""
    if name in CLASS_COLOURS:
        return CLASS_COLOURS[name][:3]  # drop alpha
    # fallback if unseen class
    return (0.5, 0.5, 0.5)

def blend(c, target, t: float):
    """Linear blend between colours c and target with weight t in [0,1]."""
    return tuple((1 - t) * c[i] + t * target[i] for i in range(3))

def gt_shade(base):       # lighter shade (towards white)
    return blend(base, (1, 1, 1), 0.35)

def pred_shade(base):     # darker shade (towards black)
    return blend(base, (0, 0, 0), 0.35)

# brighten or darken a hex colour
def adjust_lightness(color: str, amount: float = 1.0) -> str:
    rgb = mcolors.to_rgb(color)  # (r,g,b) in [0,1]
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0, min(1, l * amount))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex([r, g, b])

def draw_overlay(image_pil, target_dets, pred_dets, save_path="maira_output_visualised.png"):
    img = np.asarray(image_pil)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    # ground truth boxes
    if target_dets is not None and len(target_dets) > 0:
        for xyxy, cname in zip(target_dets.xyxy, target_dets.data['class_name']):
            base = CLASS_COLOURS.get(cname.lower(), "#808080")  # default grey if missing
            col = adjust_lightness(base, 1.0)  # keep normal/light
            x1, y1, x2, y2 = xyxy
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2.5, edgecolor=col, facecolor='none', linestyle='-')
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 4, 0), f"{cname} (GT)", fontsize=10, color=col,
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.7))

    # prediction boxes
    if pred_dets is not None and len(pred_dets) > 0:
        for xyxy, cname in zip(pred_dets.xyxy, pred_dets.data['class_name']):
            base = CLASS_COLOURS.get(cname.lower(), "#808080")  # default grey if missing
            col = adjust_lightness(base, 0.7)  # darker for prediction
            x1, y1, x2, y2 = xyxy
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=col, facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 4, 0), f"{cname} (Pred)", fontsize=9, color=col,
                    bbox=dict(boxstyle="round,pad=0.2", fc='black', ec='none', alpha=0.35))

    legend_elems = [
        Line2D([0], [0], color='black', lw=2.5, linestyle='-', label='Ground truth'),
        Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='Prediction'),
    ]
    ax.legend(handles=legend_elems, loc='lower right')
    ax.axis('off')

    if pred_dets is not None and 'unlocalised' in pred_dets.data and pred_dets.data['unlocalised']:
        unloc_text = "Predicted (no localisation): " + ", ".join(set(pred_dets.data['unlocalised']))
        ax.text(0.5, -0.05, unloc_text, fontsize=11, ha='center', va='top',
                transform=ax.transAxes, color='red')
        
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

###
# assume pred and gt are sv.Detections for the current image
mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=[prediction], 
    targets=[target]
)

print(f"map50: {mean_average_precision.map50:.3f}")
print(f"map75: {mean_average_precision.map75:.3f}")
print(f"map50-95: {mean_average_precision.map50_95:.3f}")

# draws the boxes
draw_overlay(image, target, prediction)