"""
    Plots the distribution of classes before and after majority voting in the dataset using a bar plot

    Requires:
        annotations_train.jsonl: The jsonl file for the training annotations
        annotations_valid.jsonl: The jsonl file for the validation annotations
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import re
from collections import Counter
import numpy as np

# original csv file with annotations
df_before = pd.read_csv("train_original.csv")
df_before = df_before[df_before["class_name"] != "No finding"]  # remove no finding
before_counts = df_before["class_name"].value_counts()

# func to extract class names from "suffix" of jsonl files
def extract_classes_from_suffix(suffix):
    raw_classes = re.findall(r'([A-Za-z/ ]+)<loc_', suffix)  # removes loc tags
    return [cls.strip() for cls in raw_classes]  # remove extra spaces too

after_counts = Counter()

# annotations stored across two jsonl files, for train and validation sets
for file in ["annotations_train.jsonl", "annotations_valid.jsonl"]:
    with open(file, "r") as f:
        for line in f:
            entry = json.loads(line)

            # skip augmented images
            # jsonl files were generated after augmenting
            # augmented image filenames end in _aug_XXXXX.png, where XXXXX is a 4 or more digit number
            if re.search(r'_aug_\d{4,}\.png$', entry["image"]):
                continue

            class_names = extract_classes_from_suffix(entry["suffix"])
            after_counts.update(class_names)

after_counts = pd.Series(after_counts)

# combine all classes
sorted_before = before_counts.sort_values(ascending=False)
all_classes = list(sorted_before.index)
new_after = [cls for cls in after_counts.index if cls not in all_classes]
all_classes.extend(sorted(new_after))

before_vals = [before_counts.get(cls, 0) for cls in all_classes]
after_vals =  [after_counts.get(cls, 0) for cls in all_classes]

x = np.arange(len(all_classes))
width = 0.4

# different colours for each class
cmap = plt.get_cmap("tab20")  # colour map
colours = [cmap(i % 20) for i in range(len(all_classes))]

# plots the comparison chart
plt.figure(figsize=(8, 6))
for i, cls in enumerate(all_classes):
    plt.bar(x[i] - width/2, before_vals[i], width=width, color=colours[i])
    plt.bar(x[i] + width/2, after_vals[i], width=width, color=colours[i])

plt.xticks(x, all_classes, rotation=90, ha='right')
plt.ylabel("Count")
# plt.title("Class Distribution: Before vs After")
plt.tight_layout()
plt.show()

# plots ONLY the original annotations csv file
plt.figure(figsize=(6, 6))
plt.bar(before_counts.index, before_counts.values, color=colours[:len(before_counts)])
plt.xticks(rotation=90, ha='right')
plt.ylabel("Count")
# plt.title("Class Distribution in Original CSV")
plt.tight_layout()
plt.show()


# plotting after augmentation
after_counts_incl_aug = Counter()

for file in ["annotations_train.jsonl", "annotations_valid.jsonl"]:
    with open(file, "r") as f:
        for line in f:
            entry = json.loads(line)
            class_names = extract_classes_from_suffix(entry["suffix"])
            after_counts_incl_aug.update(class_names)

after_counts_incl_aug = pd.Series(after_counts_incl_aug)

# sorts by count, descending
after_sorted = after_counts_incl_aug.sort_values(ascending=False)

# plots figure
plt.figure(figsize=(6, 6))
plt.bar(after_sorted.index, after_sorted.values, color=[cmap(i % 20) for i in range(len(after_sorted))])
plt.xticks(rotation=90, ha='right')
plt.ylabel("Count")
plt.tight_layout()
plt.show()



# print before and after majority voting counts
print("\nClass Distribution (Before vs After, excluding augmented):")
print(f"{'Class':<30} {'Before':>8} {'After':>8} {'Percentage Change':>8}")
for cls, b, a in zip(all_classes, before_vals, after_vals):
    print(f"{cls:<30} {b:>8} {a:>8} {1-round(a/b, 2):>8.2f}")
    
# print after augmenting counts
print("\nClass Distribution in JSONL (including augmented images):")
print(f"{'Class':<30} {'Count':>8}")
for cls, count in after_sorted.items():
    print(f"{cls:<30} {count:>8}")
