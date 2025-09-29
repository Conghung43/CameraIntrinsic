import os
import shutil
import random

# Define paths
dataset_dir = "/Users/nguyenconghung/Documents/barcode_data/mosaic_0327_2_yoloformat"  # Change this to your dataset folder
train_dir = "/Users/nguyenconghung/Downloads/yolov8/Barcode1/train"
valid_dir = "/Users/nguyenconghung/Downloads/yolov8/Barcode1/valid"

# Create train and valid folders if they don't exist
for folder in [train_dir, valid_dir]:
    os.makedirs(folder, exist_ok=True)

# Get all image files (supports .jpg, .png, .jpeg)
image_extensions = (".jpg", ".png", ".jpeg",".JPG")
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(image_extensions)]

# Shuffle dataset for randomness
random.shuffle(image_files)

# Split data (80% train, 20% valid)
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

train_files = image_files[:split_index]
valid_files = image_files[split_index:]

# Update train and valid directories to include subfolders for images and labels
train_images_dir = os.path.join(train_dir, "images")
train_labels_dir = os.path.join(train_dir, "labels")
valid_images_dir = os.path.join(valid_dir, "images")
valid_labels_dir = os.path.join(valid_dir, "labels")

# Create subfolders for images and labels
for folder in [train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

# Function to move image and corresponding .txt file
def move_files(files, images_destination, labels_destination):
    for img_file in files:
        img_path = os.path.join(dataset_dir, img_file)
        txt_path = os.path.join(dataset_dir, os.path.splitext(img_file)[0] + ".txt")  # Get corresponding txt file

        # Move image file
        shutil.move(img_path, os.path.join(images_destination, img_file))
        
        # Move annotation file if it exists
        if os.path.exists(txt_path):
            shutil.move(txt_path, os.path.join(labels_destination, os.path.basename(txt_path)))

# Move files to respective folders
move_files(train_files, train_images_dir, train_labels_dir)
move_files(valid_files, valid_images_dir, valid_labels_dir)

print(f"Moved {len(train_files)} images to {train_images_dir} and labels to {train_labels_dir}")
print(f"Moved {len(valid_files)} images to {valid_images_dir} and labels to {valid_labels_dir}")
print("Dataset split complete!")
