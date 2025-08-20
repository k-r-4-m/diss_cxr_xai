"""
    Computes the average width and height of items in the VinDr-CXR dataset
    Requires the train_original.csv with the image annotations
"""

import pandas as pd

# Load CSV (adjust the path)
df = pd.read_csv("train_original.csv")

# Remove rows with "No finding" or missing coordinates
df = df[(df['class_name'] != 'No finding') & df[['x_min','y_min','x_max','y_max']].notnull().all(axis=1)]

# Compute width and height
df['width'] = df['x_max'] - df['x_min']
df['height'] = df['y_max'] - df['y_min']

# Compute average width and height per class
avg_bbox_per_class = df.groupby('class_name')[['width','height']].mean()

avg_bbox_per_class['size'] = avg_bbox_per_class['width'] * avg_bbox_per_class['height']

avg_bbox_per_class.sort_values(by=['size'], inplace=True)

# Print results
print(avg_bbox_per_class)
