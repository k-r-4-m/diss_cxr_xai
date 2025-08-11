import json
import re
from pathlib import Path

# base directories for train and valid splits
train_dir = Path("./output_dataset/train")
valid_dir = Path("./output_dataset/valid")

input_filename = "annotations.jsonl"
output_filename = "annotations_caption_to_phrase.jsonl"

# Dictionary of disease definitions (expand as needed)
disease_definitions = {
    'Aortic enlargement': "an enlarged artery.",
    'Atelectasis': "darkened or reduced areas of the lung.",
    'Calcification': "bright white spots.",
    'Cardiomegaly': "an unusually thick or streched heart.",
    'Consolidation': "dense solid areas.",
    'ILD': "reticular opacities in the lungs.",
    'Infiltration': "increased density in the lung tissue.",
    'Lung Opacity': "white or greyish patches in the lungs.",
    'Nodule/Mass': "well or poorly-defined shape in the lungs.", 
    'Other lesion': "localised structures with irregular borders or density.",
    'Pleural effusion': "shadows around the lungs.",
    'Pleural thickening': "a dense layer around the lungs.",
    'Pneumothorax': "a gap or absence of lung tissue.",
    'Pulmonary fibrosis': "a dense or fibrous region."
}


# regex pattern for a label and then its loc tags
pattern = re.compile(r'([A-Za-z0-9 _\-/]+?)(<loc_\d+><loc_\d+><loc_\d+><loc_\d+>)')

# converts the OD suffixes to CAPTION_TO_PHRASE_GROUNDING
# puts the loc tags after the label
# then puts definitions after
def convert_suffix(od_suffix: str) -> str:
    matches = pattern.findall(od_suffix)

    if not matches:
        return od_suffix  # fallback if no matches found

    # First part: all disease names + loc tags, including duplicates
    part1 = "".join(f"{disease.strip()}{loc}" for disease, loc in matches)

    # Second part: unique disease names in order of first appearance
    seen = set()
    unique_diseases = []
    for disease, _ in matches:
        disease = disease.strip()
        if disease not in seen:
            seen.add(disease)
            unique_diseases.append(disease)

    part2 = " ".join(
        f"{disease} can be seen as {disease_definitions.get(disease, 'an unspecified abnormality')}"
        for disease in unique_diseases
    )

    return f"{part1} {part2}"



def process_folder(folder: Path):
    input_path = folder / input_filename
    output_path = folder / output_filename

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)

            # new prefix, switches from OD
            data["prefix"] = "<CAPTION_TO_PHRASE_GROUNDING>"

            # changes suffix format to include defs
            data["suffix"] = convert_suffix(data["suffix"])

            outfile.write(json.dumps(data) + "\n")

    print(f"converted {input_path} to {output_path}")

# runs on both training and validation annotations jsonl files
for folder in [train_dir, valid_dir]:
    process_folder(folder)
