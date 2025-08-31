"""
    Visualises training images before and after data augmentation

    Requires:
        jsonl_path: A path to the annotations json file formatted in Florence-style
        ground_truth_csv_path: A path to the original csv file included in Vindr's dataset
        image_dir: A path to the directory where training images are held
        dicom_dir: A path to the directory where the original dicom files are held
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import re
import pydicom


class XRayBBoxVisualizer:
    def __init__(self, jsonl_path, ground_truth_csv_path, image_dir, dicom_dir=None):
        self.jsonl_path = jsonl_path
        self.ground_truth_csv_path = ground_truth_csv_path
        self.image_dir = image_dir
        self.dicom_dir = dicom_dir
        self.original_image_sizes = {}

    # parses florence output using regex to remove the <loc_xxxx> tags
    def parse_florence_output(self, suffix_text):
        pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        matches = re.findall(pattern, suffix_text)  # remove the loc tags
        return [{'label': m[0].strip(), 'bbox': list(map(int, m[1:]))} for m in matches]  # remove extra spaces too

    # loads predictions fron the jsonl file
    def load_predictions(self):
        predictions = {}
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                predictions[data['image']] = self.parse_florence_output(data['suffix'])
        return predictions

    # loads the ground truth from the annotations csv
    def load_ground_truth_from_csv(self):
        df = pd.read_csv(self.ground_truth_csv_path)
        ground_truth = {}
        # annotations are spread out across multiple rows for single images
        for image_id, group in df.groupby('image_id'):
            detections = []
            for _, row in group.iterrows():
                if row['class_name'] == 'No finding' or pd.isna(row['x_min']):
                    continue
                detections.append({
                    'label': row['class_name'],
                    'bbox': [int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])]
                })
            image_name = image_id if image_id.endswith('.png') else f"{image_id}.png"
            ground_truth[image_name] = detections
        return ground_truth

    # takes the normalised florence coordinates and transformers them into the actual pixel locations
    def normalise_florence_coordinates(self, bbox, img_width, img_height):
        x1, y1, x2, y2 = bbox
        return [
            (x1 / 1000.0) * img_width,
            (y1 / 1000.0) * img_height,
            (x2 / 1000.0) * img_width,
            (y2 / 1000.0) * img_height
        ]

    # scales the ground truth coordinates
    # images have been resized from the original ground truth
    def scale_ground_truth_coordinates(self, bbox, original_width, original_height, current_width, current_height):
        x1, y1, x2, y2 = bbox
        scale_x = current_width / original_width
        scale_y = current_height / original_height
        return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

    # gets image size from DICOM files
    def get_dicom_image_size(self, image_id):
        if self.dicom_dir is None:
            raise ValueError("DICOM directory not set!")
        dicom_path = os.path.join(self.dicom_dir, f"{image_id}.dicom")
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
        ds = pydicom.dcmread(dicom_path)
        return ds.Columns, ds.Rows  # width, height

    # plots the bounding boxes on to the image
    def plot_bboxes(self, ax, image, detections, img_width, img_height, title, color_map, 
                    is_florence=False, original_size=None):
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        for detection in detections:
            label = detection['label']
            bbox = detection['bbox']
            if is_florence:  # only florence predictions needs to transform the normalised coordinates
                x1, y1, x2, y2 = self.normalise_florence_coordinates(bbox, img_width, img_height)
            elif original_size:
                x1, y1, x2, y2 = self.scale_ground_truth_coordinates(
                    bbox, original_size[0], original_size[1], img_width, img_height
                )
            else:
                x1, y1, x2, y2 = bbox

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            width, height = x2 - x1, y2 - y1

            if width <= 0 or height <= 0:
                continue
            
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                     edgecolor=color_map.get(label, 'red'), facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, label, fontsize=8, color=color_map.get(label, 'red'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    def create_color_map(self, all_labels):
        colours = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        return {label: colours[i % len(colours)] for i, label in enumerate(sorted(set(all_labels)))}

    # plots the images next to each other so they can be compared
    def visualise_comparisons(self, num_images=10, figsize=(15, 8), save_dir="comparison_outputs", save_images=True):
        if save_images:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving comparison images to: {save_dir}")

        print("Loading predictions")
        predictions = self.load_predictions()
        print("Loading ground truth")
        ground_truth = self.load_ground_truth_from_csv()

        common_images = list(set(predictions.keys()) & set(ground_truth.keys()))
        if not common_images:
            print("No matching images between predictions and ground truth")
            return

        images_to_plot = common_images[:num_images]
        all_labels = [det['label'] for img in images_to_plot for det in predictions[img] + ground_truth[img]]
        color_map = self.create_color_map(all_labels)

        for i, img_name in enumerate(images_to_plot):
            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            try:
                image = Image.open(img_path)
                img_width, img_height = image.size
            except Exception as e:
                print(f"Failed to load image {img_name}: {e}")
                continue

            image_id = img_name.replace('.png', '')
            try:
                original_size = self.get_dicom_image_size(image_id)
            except Exception as e:
                print(f"Could not get DICOM size for {img_name}: {e}")
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            self.plot_bboxes(ax1, image, ground_truth[img_name], img_width, img_height,
                 f"Before Majority Voting\n{len(ground_truth[img_name])} boxes",
                 color_map, is_florence=False, original_size=original_size)
            self.plot_bboxes(ax2, image, predictions[img_name], img_width, img_height,
                 f"After Majority Voting\n{len(predictions[img_name])} boxes",
                 color_map, is_florence=True)

            plt.suptitle(f"{img_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_images:
                save_path = os.path.join(save_dir, f"comparison_{i+1:02d}_{img_name}")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")
                plt.close()
            else:
                plt.show()


if __name__ == "__main__":
    visualizer = XRayBBoxVisualizer(
        jsonl_path="./output_dataset/train/annotations.jsonl",
        ground_truth_csv_path="train_original.csv",
        image_dir="./output_dataset/train/",
        dicom_dir="./dicom/"
    )
    visualizer.visualise_comparisons(num_images=10, save_images=True)
