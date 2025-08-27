import os
import pandas as pd
import json
from tqdm import tqdm
from florence_tools import load_config

# load config
config_path = "./config.yaml"
config = load_config(config_path)

OUTPUT_DIR = "./padchest/"
ANNOTATIONS_CSV = f"./{OUTPUT_DIR}/padchest_annotations.csv"
IMAGE_DIR = f"./{OUTPUT_DIR}/images"
print("config loaded")

# only including labels that are also present in vindr
INCLUDE_LABELS = {
    "apical pleural thickening",  # pleural thickening
    "mediastinal mass",  # nodule/mass
    "atelectasis",  # atelectasis
    "infiltrates",  # infiltration
    "cardiomegaly",  # cardiomegaly
    "calcified granuloma",  # other lesion
    "aortic elongation",  # aortic enlargement
    "pleural effusion",  # pleural effusion
    "nodule",  # nodule/mass
    "segmental atelectasis",  # atelectasis
    "lobar atelectasis",  # atelectasis
    "interstitial pattern",  # ILD
    "pulmonary mass",  # nodule/mass
    "descendent aortic elongation",  # aortic enlargement
    "consolidation",  # consolidation
    "reticulonodular interstitial pattern",  # ILD
    "supra aortic elongation",  # aortic enlargement
    "aortic button enlargement",  # aortic enlargement
    "reticular interstitial pattern",  # ILD
    "pneumothorax",  # pneumothorax
    "granuloma",  # other lesion
    "multiple nodules",  # nodule/mass
    "pleural thickening",  # plueral thickening
    "mass",  # nodule/mass
    "ascendent aortic elongation",  # aortic enlargement
    "calcified pleural thickening",  # pleural thickening
    "total atelectasis",  # atelectasis
    "lytic bone lesion",  # other lesion
    "pleural mass",  # nodule/mass
    "atelectasis basal",  # atelectasis
    "calcified pleural plaques",  # calcification
    "calcified densities",  # calcification
    "calcified adenopathy",  # calcification
    "calcified fibroadenoma",  # calcification
    "heart valve calcified",  # calcification
    "ground glass pattern",  # lung opacity
    "aveolar pattern",  # lung opacity
    "air bronchogram",  # lung opacity
    "miliary opacities",  # lung opacity
    "kerley lines",  # lung opacity
    "fibrotic band"  # pulmonary fibrosis
}

# load CSV
df = pd.read_csv(ANNOTATIONS_CSV)

# filter to only include rows with desired labels
df = df[df['label'].isin(INCLUDE_LABELS)]

print(f"Unique labels in new dataset: {df['label'].nunique()}")

# group by image and concatenate findings
grouped = df.groupby("ImageID")["sentence_en"].apply(lambda x: " ".join(x)).reset_index()

os.makedirs(OUTPUT_DIR, exist_ok=True)
jsonl_path = os.path.join(OUTPUT_DIR, "annotations.jsonl")

with open(jsonl_path, "w") as f_out:
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Writing JSONL"):
        image_filename = row["ImageID"]
        report_text = row["sentence_en"]

        # Florence JSONL format
        json_entry = {
            "image": image_filename,
            "prefix": "<MORE_DETAILED_CAPTION>",
            "suffix": report_text
        }
        f_out.write(json.dumps(json_entry) + "\n")

print(f"Saved {len(grouped)} entries to {jsonl_path}")
