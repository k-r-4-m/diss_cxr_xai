from PIL import Image
import os

input_folder = "C:/Users/Mark/Downloads/confusion_matrices_no_defs"  # folder containing confusion matrices
output_file = "all_conf_matrix.png"
images_per_row = 3
padding = 350
scale_factor = 2.0

# sorts images alphabetically
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# load + resize images
images = []
for f in image_files:
    img = Image.open(os.path.join(input_folder, f))
    # scales the images
    img = img.resize((int(img.width * scale_factor), int(img.height * scale_factor)), Image.LANCZOS)
    images.append(img)

# max width and height for each grid cell
max_width = max(img.width for img in images)
max_height = max(img.height for img in images)

# grid size
num_images = len(images)
rows = (num_images + images_per_row - 1) // images_per_row 
cols = min(images_per_row, num_images)
total_width = cols * max_width + (cols - 1) * padding
total_height = rows * max_height + (rows - 1) * padding
combined_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

# pastes the images onto the grid
for index, img in enumerate(images):
    row = index // images_per_row
    col = index % images_per_row
    x = col * (max_width + padding)
    y = row * (max_height + padding)
    combined_image.paste(img, (x, y))

combined_image.save(output_file)
print(f"combined image saved as {output_file}")
