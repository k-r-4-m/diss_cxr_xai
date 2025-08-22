"""
    Evaluates MAIRA-2 on the validation set of VinDr. 

    Evaluates the following:
        Classification performance (precision, recall, F1)
        Localisation performance (mean average precision)
        Text generation performance (BLEU, ROUGE-L)

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

# loads the config.yaml file for model configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config_path = "./config.yaml"
config = load_config(config_path)
OUTPUT_DIR = config.get('output_dir')
ANNOTATIONS_CSV = config.get('annotations_csv')
ANNOTATIONS_JSON = f"{OUTPUT_DIR}/valid/annotations.jsonl"
IMAGE_DIR = f"{OUTPUT_DIR}/valid/"

# logs into huggingface
# token is stored in a seperate yaml file called token_file.yaml
# please place your own token here
with open("./token_file.yaml", 'r') as f:
    token_file = yaml.safe_load(f)
    hf_token = token_file.get('hf_token')
login(hf_token)

df = pd.read_csv(ANNOTATIONS_CSV)
CLASSES = df['class_name'].unique().tolist()
if "No finding" in CLASSES:
    CLASSES.remove("No finding")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "microsoft/maira-2"
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

model = model.eval()
model = model.to(DEVICE)

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
    "lung opacity": ["increased density in the right", "increased density in the left"],
    "nodule/mass": ["mass", "nodule", "calcified granuloma", "nodular density", "granulomas"],
    "other lesion": ["subcutaneous emphysema", ],
    "pleural effusion": ["pleural fluid", "effusion", "effusions", "fluid in pleural space"],
    "pleural thickening": ["thickening"],
    "pneumothorax": ["collapsed lung", "air in pleural space"],
    "pulmonary fibrosis": ["interstitial fibrotic"],
}

# patterns for when a finding is mentioned but the model says it isnt present
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnot seen\b",
    r"\babsent\b",
    r"\bfree of\b",
    r"\bclear\b",
    r"\bunremarkable\b",
    r"\bnormal\b"
]

# returns true if a finding contains a phrase in NEGATION_PATTERNS
def is_negated(text: str) -> bool:
    text_lower = text.lower().strip()
    return any(re.search(pat, text_lower) for pat in NEGATION_PATTERNS)

# maps the synonyms/variations of findings found in CLASS_SYNONYMS back to the real class names
def normalise_finding(text: str) -> str:
    # makes lowercase, strips of whitespace, removes punctuation
    text = text.lower().strip().translate(str.maketrans('', '', string.punctuation))

    # first checks for exact class name matches
    for class_name in CLASSES:
        if class_name.lower() in text:
            return class_name
    
    # otherwise check if a full synonym phrase appears
    for real_class, synonyms in CLASS_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower() in text:
                return real_class
            
    # final check to check each word for synonyms
    text_words = re.findall(r'\b\w+\b', text)
    
    for class_name, synonyms in CLASS_SYNONYMS.items():
        for synonym in synonyms:
            synonym_words = re.findall(r'\b\w+\b', synonym.lower())
            for word in synonym_words:
                if len(word) > 3 and word in text_words:  # only get actual words
                    print(f"found word match: '{word}' from synonym '{synonym}' -> {class_name}")
                    # for other words, return immediately
                    return class_name
    
    return None

    # # list of key trigger words for each class
    # key_word_mappings = {
    #     "aortic enlargement": ["tortuous"],
    #     "atelectasis": ["atelectasis", "collapse"],
    #     "calcification": ["calcified", "calcification"],
    #     "cardiomegaly": ["cardiac"],
    #     "consolidation": ["consolidation", "consolidative"],
    #     "ild": ["interstitial"],
    #     "infiltration": ["infiltrate", "infiltrates", "infiltration"],
    #     "lung opacity": ["opacity", "opacification", "density"],
    #     "nodule/mass": ["nodule", "nodular", "mass", "granuloma", "granulomas"],
    #     "other lesion": ["emphysema", "lesion"],
    #     "pleural effusion": ["effusion", "effusions"],
    #     "pleural thickening": ["thickening"],
    #     "pneumothorax": ["pneumothorax"],
    #     "pulmonary fibrosis": ["fibrotic", "fibrosis"]
    # }
    
    # # Check for key word matches

    # # check for any matches of key words
    # for class_name, key_words in key_word_mappings.items():
    #     for key_word in key_words:
    #         if key_word in text_words:
    #             # tortuous only maps when other words are used (i.e. aorta, aortic, thoracic)
    #             if key_word == "tortuous":
    #                 if any(word in text_words for word in ["aorta", "aortic", "thoracic"]):
    #                     return class_name
    #             # cardic only maps when enlargement is mentioned
    #             elif key_word == "cardiac":
    #                 if any(word in text_words for word in ["enlarged", "enlargement", "silhouette", "size"]):
    #                     return class_name
    #             # opacity or opacification only maps to lung opacity
    #             elif key_word in ["opacity", "opacification"]:
    #                 # need to also check if its not matched to something else more specific
    #                 if not any(specific_word in text_words for specific_word in 
    #                           ["consolidation", "consolidative", "infiltrate", "effusion"]):
    #                     return class_name
    #             else:
    #                 return class_name
    
    # # fallback mechanism to check if any word from the main class name appears
    # for real_class in CLASSES:
    #     class_words = re.findall(r'\b\w+\b', real_class.lower())
    #     if any(word in text_words and len(word) > 3 for word in class_words):
    #         return real_class
    
    # return None

# defines colors for each class
# used to draw bounding boxes
CLASS_COLORS = plt.cm.Set3(np.linspace(0, 1, len(CLASSES)))
CLASS_COLOR_MAP = {cls: tuple(int(c*255) for c in CLASS_COLORS[i][:3]) for i, cls in enumerate(CLASSES)} 

# clamps bounding boxes
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# regex pattern to capture the label and the loc tags
GT_PATTERN = re.compile(
    r"([A-Za-z0-9 _\-/]+?)\s*<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
)

# function to draw bounding boxes on image
def draw_bboxes_on_image(image, detections, title, class_names):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    try:
    # Try to use a default font, fallback to default if not available
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    if len(detections) > 0:
        for i, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            x1, y1, x2, y2 = bbox
            class_name = class_names[class_id]
            color = CLASS_COLOR_MAP.get(class_name, (255, 0, 0))  # Default to red if class not found

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background
            text_bbox = draw.textbbox((x1, y1-25), class_name, font=font)
            draw.rectangle(text_bbox, fill=color)

            # Draw label text
            draw.text((x1, y1-25), class_name, fill=(255, 255, 255), font=font)

    return img_with_boxes

# plots the two bounding boxed images side by side
def create_side_by_side_visualization(image, gt_detections, pred_detections, class_names, image_name, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    # Draw bboxes on images
    gt_image = draw_bboxes_on_image(image, gt_detections, "Ground Truth", class_names)
    pred_image = draw_bboxes_on_image(image, pred_detections, "Predictions", class_names)

    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Ground truth subplot
    axes[0].imshow(gt_image)
    axes[0].set_title(f"Ground Truth - {image_name}", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    # Add ground truth statistics
    gt_counts = {}

    if len(gt_detections) > 0:
        for class_id in gt_detections.class_id:
            class_name = class_names[class_id]
            gt_counts[class_name] = gt_counts.get(class_name, 0) + 1

    gt_text = "GT Classes:\n" + "\n".join([f"{cls}: {count}" for cls, count in gt_counts.items()]) if gt_counts else "GT Classes: None"
    axes[0].text(0.02, 0.98, gt_text, transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Predictions subplot
    axes[1].imshow(pred_image)
    axes[1].set_title(f"Predictions - {image_name}", fontsize=14, fontweight='bold')
    axes[1].axis('off')
	
    # Add prediction statistics
    pred_counts = {}
    if len(pred_detections) > 0:
        for class_id in pred_detections.class_id:
            class_name = class_names[class_id]
            pred_counts[class_name] = pred_counts.get(class_name, 0) + 1

    pred_text = "Pred Classes:\n" + "\n".join([f"{cls}: {count}" for cls, count in pred_counts.items()]) if pred_counts else "Pred Classes: None"
    axes[1].text(0.02, 0.98, pred_text, transform=axes[1].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add color legend
    legend_elements = []
    for class_name, color in CLASS_COLOR_MAP.items():
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=[c/255 for c in color], label=class_name))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=min(len(CLASSES), 6), fontsize=10)

    plt.tight_layout()

    # Save the visualization
    save_path = os.path.join(save_dir, f"{sanitize_filename(image_name)}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path 

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

# converts the MAIRA-2 output into a Detections object
# MAIRA-2 output is quite different than Florence-2, explicit class names aren't given
# need to extract any class names mentioned in each part of the report
def parse_maira_prediction_to_detections(maira_list: List[Tuple[str, List[Tuple[float, float, float, float]]]], img_size: Tuple[int, int],) -> sv.Detections:
    if not maira_list:
        return sv.Detections.empty()

    W, H = img_size
    xyxy = []
    class_ids = []
    confs = []

    print(f"maira list: {maira_list}")

    for item in maira_list:
        # outputs dont always have bounding boxes
        # need to handle different possible formats
        if isinstance(item, tuple) and len(item) == 2:
            finding_text, bboxes = item
        elif isinstance(item, str):
            finding_text = item
            bboxes = None
        else:
            continue

        if not isinstance(finding_text, str) or not finding_text.strip():
            continue

        print(f"Processing finding: '{finding_text}'")

        # skips negated findings
        if is_negated(finding_text):
            print(f"Skipping negated finding: '{finding_text}'")
            continue

        # tries to normalise the finding to a known class
        normalised_class = normalise_finding(finding_text)
        
        if normalised_class is None:
            # Try extracting individual findings if text contains <obj> tags
            sub_findings = extract_findings_from_maira_text(finding_text)
            for sub_finding in sub_findings:
                if not is_negated(sub_finding):
                    sub_mapped = normalise_finding(sub_finding)
                    if sub_mapped:
                        class_id = CLASSES.index(sub_mapped)
                        # no localisaton given, add 1x1 bounding box in top left
                        xyxy.append([1.0, 1.0, 2.0, 2.0])
                        class_ids.append(class_id)
                        confs.append(1.0)
                        print(f"  Mapped sub-finding '{sub_finding}' -> {sub_mapped}")
            continue
        
        if normalised_class not in CLASSES:
            print(f"Mapped class '{normalised_class}' not in classes list!")
            continue
            
        class_id = CLASSES.index(normalised_class)
        print(f"  Final mapping: '{finding_text}' -> {normalised_class} (ID: {class_id})")
        
        # handles bounding boxes
        if bboxes and len(bboxes) > 0:
            for bbox in bboxes:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # converts normalised coords to pixels
                    x1_px = max(0, min(W, x1 * W))
                    y1_px = max(0, min(H, y1 * H))
                    x2_px = max(0, min(W, x2 * W))
                    y2_px = max(0, min(H, y2 * H))
                    
                    # ensures valid box
                    x_min, x_max = sorted([x1_px, x2_px])
                    y_min, y_max = sorted([y1_px, y2_px])
                    
                    if x_max - x_min < 1:
                        x_max = x_min + 1
                    if y_max - y_min < 1:
                        y_max = y_min + 1
                    
                    xyxy.append([x_min, y_min, x_max, y_max])
                    class_ids.append(class_id)
                    confs.append(1.0)
        else:
            # keep class mention even without grounding
            xyxy.append([1.0, 1.0, 2.0, 2.0])  # just a 1x1 bounding box in the top left
            class_ids.append(class_id)
            confs.append(1.0)
    
    print(f"\nFinal result: {len(xyxy)} detections")
    print(f"Classes: {[CLASSES[cid] for cid in class_ids]}")
    
    if not xyxy:
        return sv.Detections.empty()
    
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=float),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confs, dtype=float),
    )

    # for finding_text, bboxes in maira_list:
    #     if not isinstance(finding_text, str):
    #         continue

    #     # skip negated findings
    #     if is_negated(finding_text):
    #         continue

    #     norm_finding = normalise_finding(finding_text)
    #     if norm_finding is None or norm_finding not in CLASSES:
    #         print("None finding")
    #         continue

    #     cid = CLASSES.index(norm_finding)

    #     if bboxes is not None and len(bboxes) > 0:  # maira sometimes responds with bboxes as None
    #         for (x1n, y1n, x2n, y2n) in bboxes:
    #             x1 = clamp(x1n * W, 0, W)
    #             y1 = clamp(y1n * H, 0, H)
    #             x2 = clamp(x2n * W, 0, W)
    #             y2 = clamp(y2n * H, 0, H)
    #             x_min, x_max = sorted([x1, x2])
    #             y_min, y_max = sorted([y1, y2])
    #             xyxy.append([x_min, y_min, x_max, y_max])
    #             class_ids.append(cid)
    #             confs.append(1.0)
    #     else:
    #         # keep class mention even without grounding
    #         xyxy.append([1.0, 1.0, 2.0, 2.0])  # just a 1x1 bounding box in the top left
    #         class_ids.append(cid)
    #         confs.append(1.0)

    #     # finding_lc = finding_text.lower()

    #     # for cid, cname in enumerate(CLASSES):
    #     #     if cname.lower() in finding_lc:
    #     #         if bboxes is not None and len(bboxes) > 0:  # maira sometimes responds with bboxes as None
    #     #             for (x1n, y1n, x2n, y2n) in bboxes:
    #     #                 x1 = clamp(x1n * W, 0, W)
    #     #                 y1 = clamp(y1n * H, 0, H)
    #     #                 x2 = clamp(x2n * W, 0, W)
    #     #                 y2 = clamp(y2n * H, 0, H)
    #     #                 x_min, x_max = sorted([x1, x2])
    #     #                 y_min, y_max = sorted([y1, y2])
    #     #                 xyxy.append([x_min, y_min, x_max, y_max])
    #     #                 class_ids.append(cid)
    #     #                 confs.append(1.0)
    #     #         else:  # keep class mention even without grounding
    #     #             xyxy.append([1.0, 1.0, 2.0, 2.0])  # just a 1x1 bounding box in the top left
    #     #             class_ids.append(cid)
    #     #             confs.append(1.0)

    # if not xyxy:
    #     return sv.Detections.empty()

    # return sv.Detections(
    #     xyxy=np.array(xyxy, dtype=float),
    #     class_id=np.array(class_ids, dtype=int),
    #     confidence=np.array(confs, dtype=float),
    # )

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

### runs eval loop
predictions = []
targets = []

with open(ANNOTATIONS_JSON, "r") as f:
    for line in tqdm(f, desc="Evaluating MAIRA-2"):
        item = json.loads(line)

        print(item)

        # image and size
        img_path = os.path.join(IMAGE_DIR, item["image"])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # ground truth (from suffix)
        gt = parse_suffix_to_detections(item.get("suffix", ""), CLASSES, (W, H))
        targets.append(gt)

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
                max_new_tokens=1024,  # 450 recommended for report generation from maira docs
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

        pred = parse_maira_prediction_to_detections(maira_output, (W, H))
        predictions.append(pred)

        # # Create side-by-side visualization for this image
        # image_name = os.path.splitext(item["image"])[0]
        # viz_path = create_side_by_side_visualization(image, gt, pred, CLASSES, image_name)
        # print(f"Saved visualization: {viz_path}")


# mean average precision
mean_average_precision = sv.MeanAveragePrecision.from_detections(predictions=predictions, targets=targets)
print(f"map50_95: {mean_average_precision.map50_95:.2f}")
print(f"map50: {mean_average_precision.map50:.2f}")
print(f"map75: {mean_average_precision.map75:.2f}")

# confusion matrix
conf_matrix = sv.ConfusionMatrix.from_detections(predictions=predictions, targets=targets, classes=CLASSES)
fig = conf_matrix.plot()
fig.savefig("confusion_matrix_maira2.png", dpi=300, bbox_inches='tight')

# per class IoU
def compute_per_class_iou(preds, gts, num_classes):
    iou_per_class = defaultdict(list)
    for pred, gt in zip(preds, gts):
        for cls in range(num_classes):
            pred_cls = pred[pred.class_id == cls]
            gt_cls = gt[gt.class_id == cls]
            if len(gt_cls) == 0 and len(pred_cls) == 0:
                continue
            iou_matrix = box_iou_batch(gt_cls.xyxy, pred_cls.xyxy)
            if iou_matrix.size == 0:
                continue
            matched_pred, matched_gt = set(), set()
            for gt_idx, ious in enumerate(iou_matrix):
                if ious.size == 0:
                    continue
                best_pred_idx = np.argmax(ious)
                iou = ious[best_pred_idx]
                if iou >= 0.5 and best_pred_idx not in matched_pred:
                    iou_per_class[cls].append(iou)
                    matched_pred.add(best_pred_idx)
                    matched_gt.add(gt_idx)
    avg_iou_per_class = {CLASSES[cls]: np.mean(lst) if lst else 0.0 for cls, lst in iou_per_class.items()}
    for cls in range(num_classes):
        if CLASSES[cls] not in avg_iou_per_class:
            avg_iou_per_class[CLASSES[cls]] = 0.0
    return avg_iou_per_class

iou_scores = compute_per_class_iou(predictions, targets, len(CLASSES))
print("\nPer-class IoU:")
for cls, val in iou_scores.items():
    print(f"{cls}: IoU = {val:.3f}")

# precision, recall, F1
conf_mat = conf_matrix.matrix
precision_per_class, recall_per_class, f1_per_class = {}, {}, {}
for i, class_name in enumerate(CLASSES):
    TP = conf_mat[i, i]
    FP = conf_mat[:, i].sum() - TP
    FN = conf_mat[i, :].sum() - TP
    # FP = conf_mat[i, :].sum() - TP   # row = predictions
    # FN = conf_mat[:, i].sum() - TP   # col = ground truth
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    precision_per_class[class_name] = precision
    recall_per_class[class_name] = recall
    f1_per_class[class_name] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

overall_precision = np.mean(list(precision_per_class.values()))
overall_recall = np.mean(list(recall_per_class.values()))
overall_f1 = np.mean(list(f1_per_class.values()))

print("\nPer-class precision, recall, F1:")
for cls in CLASSES:
    print(f"{cls}: precision = {precision_per_class[cls]:.3f}, "
          f"recall = {recall_per_class[cls]:.3f}, "
          f"f1 = {f1_per_class[cls]:.3f}")

print("\nOverall precision, recall, F1:")
print(f"precision = {overall_precision:.3f}")
print(f"recall = {overall_recall:.3f}")
print(f"f1 = {overall_f1:.3f}")


# classification without iou
y_true_multi, y_pred_multi = [], []
for pred, gt in zip(predictions, targets):
    gt_ids = set(gt.class_id.tolist())
    pred_ids = set(pred.class_id.tolist())
    y_true_multi.append([1 if i in gt_ids else 0 for i in range(len(CLASSES))])
    y_pred_multi.append([1 if i in pred_ids else 0 for i in range(len(CLASSES))])

y_true_multi = np.array(y_true_multi)
y_pred_multi = np.array(y_pred_multi)

# macro averages
precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
    y_true_multi, y_pred_multi, average="macro", zero_division=0
)

print("\nClassification-style metrics ")
print(f"precision = {precision_cls:.3f}")
print(f"recall = {recall_cls:.3f}")
print(f"f1 = {f1_cls:.3f}")

# per class binary conf matrices
os.makedirs("confusion_matrices_maira2", exist_ok=True)

# generates and saves confusion matrices per class
def get_binary_confusion_matrix_for_class(class_id: int, class_name: str, predictions, targets):
    y_true, y_pred = [], []
    for pred, gt in zip(predictions, targets):
        gt_ids = set(gt.class_id.tolist())
        pred_ids = set(pred.class_id.tolist())
        y_true.append(int(class_id in gt_ids))
        y_pred.append(int(class_id in pred_ids))
    cm = sk_confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    reordered_cm = np.array([[tp, fn],
                             [fp, tn]])
    plt.figure(figsize=(4,3))
    sns.heatmap(reordered_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Positive","Negative"], yticklabels=["Positive","Negative"])
    plt.title(f"{class_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    filename = f"confusion_matrices_maira2/conf_matrix_{sanitize_filename(class_name)}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    # calculate per-class precision, recall, and F1 classification-style
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return cm, precision, recall, f1

print("\nGenerating per-class binary confusion matrices:")
for class_id, class_name in enumerate(CLASSES):
    cm, precision, recall, f1 = get_binary_confusion_matrix_for_class(class_id, class_name, predictions, targets)
    print(f"{class_name}:\n{cm}")

    print(f" Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")