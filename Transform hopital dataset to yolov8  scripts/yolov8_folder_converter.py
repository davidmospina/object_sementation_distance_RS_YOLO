import os
import shutil
import random

# Define dataset paths
DATASET_DIR = "Hospital_Scene_Data-main"
IMAGE_SRC_DIR = os.path.join(DATASET_DIR, "image")
LABEL_SRC_DIR = os.path.join(DATASET_DIR, "labels_yolo")  # Use the converted YOLO labels

# Define new YOLO directories
NEW_IMAGE_DIR = os.path.join(DATASET_DIR, "images")
NEW_LABEL_DIR = os.path.join(DATASET_DIR, "labels")

for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(NEW_IMAGE_DIR, folder), exist_ok=True)
    os.makedirs(os.path.join(NEW_LABEL_DIR, folder), exist_ok=True)

# Collect all images and labels
all_images = []
for image_folder in sorted(os.listdir(IMAGE_SRC_DIR)):
    folder_path = os.path.join(IMAGE_SRC_DIR, image_folder)
    if os.path.isdir(folder_path):
        all_images.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))])

# Shuffle and split data (80% train, 10% val, 10% test)
random.shuffle(all_images)
train_split = int(0.8 * len(all_images))
val_split = int(0.9 * len(all_images))

train_images = all_images[:train_split]
val_images = all_images[train_split:val_split]
test_images = all_images[val_split:]

# Function to move files
def move_files(images, split_name):
    for img_path in images:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"

        # Determine source and destination paths
        src_label_path = os.path.join(LABEL_SRC_DIR, "label" + img_path.split("/")[-2][-1], label_name)  # Adjust label path
        dest_img_path = os.path.join(NEW_IMAGE_DIR, split_name, img_name)
        dest_label_path = os.path.join(NEW_LABEL_DIR, split_name, label_name)

        # Move image
        shutil.copy(img_path, dest_img_path)

        # Move corresponding label (if exists)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dest_label_path)

# Move images and labels into correct folders
move_files(train_images, "train")
move_files(val_images, "val")
move_files(test_images, "test")

print("✅ Dataset restructuring complete!")
