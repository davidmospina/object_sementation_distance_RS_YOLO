import os
import shutil
import random
from PIL import Image

# ------------------ CONFIGURATION ------------------
source_folder = "./resized_images_640x640"  # Folder containing IMG_####.JPEG and TXT
output_base = "./payload_dataset"  # Root dataset folder
start_index = 4428  # Starting image index
image_size = (640, 640)  # Resize target
split_ratio = [0.8, 0.1, 0.1]  # Train, Val, Test

# Destination folders
image_dest = {
    "train": os.path.join(output_base, "images/train"),
    "val": os.path.join(output_base, "images/val"),
    "test": os.path.join(output_base, "images/test"),
}
label_dest = {
    "train": os.path.join(output_base, "labels/train"),
    "val": os.path.join(output_base, "labels/val"),
    "test": os.path.join(output_base, "labels/test"),
}

# ----------------------------------------------------

# Create output directories
for folder in list(image_dest.values()) + list(label_dest.values()):
    os.makedirs(folder, exist_ok=True)

# Get all labeled images
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpeg")]
random.shuffle(image_files)  # Shuffle for split

# Split into train/val/test
n = len(image_files)
train_split = int(n * split_ratio[0])
val_split = int(n * (split_ratio[0] + split_ratio[1]))

splits = {
    "train": image_files[:train_split],
    "val": image_files[train_split:val_split],
    "test": image_files[val_split:]
}

# Rename, resize, and copy files
current_index = start_index

for split_name, files in splits.items():
    for img_file in files:
        label_file = os.path.splitext(img_file)[0] + ".txt"

        img_path = os.path.join(source_folder, img_file)
        label_path = os.path.join(source_folder, label_file)

        # Rename
        new_basename = f"image_{current_index}"
        new_img_name = new_basename + ".jpg"
        new_label_name = new_basename + ".txt"

        # Resize and save image
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(image_size, Image.LANCZOS)
        img_resized.save(os.path.join(image_dest[split_name], new_img_name))

        # Copy label file
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(label_dest[split_name], new_label_name))
        else:
            print(f"⚠️ Missing label for {img_file}")

        current_index += 1

print("✅ Dataset updated and merged with proper formatting and splitting!")
