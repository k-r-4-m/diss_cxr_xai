"""
    Computes the average width and height of items in the VinDr-CXR dataset
    Requires train_original.csv from VinDr-CXR that contains the image annotations
"""

import pandas as pd

# loads csv
df = pd.read_csv("train_original.csv")

# remove rows with no finding or missing coordinates
df = df[(df['class_name'] != 'No finding') & df[['x_min','y_min','x_max','y_max']].notnull().all(axis=1)]

# compute width and height
df['width'] = df['x_max'] - df['x_min']
df['height'] = df['y_max'] - df['y_min']

# take average width and height per class
avg_bbox_per_class = df.groupby('class_name')[['width','height']].mean()

avg_bbox_per_class['size'] = avg_bbox_per_class['width'] * avg_bbox_per_class['height']

avg_bbox_per_class.sort_values(by=['size'], inplace=True)

print(avg_bbox_per_class)
