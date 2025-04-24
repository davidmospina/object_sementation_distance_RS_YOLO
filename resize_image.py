import os
from PIL import Image

# Input and output paths
input_folder = "./hugo_raw_pictures"
output_folder = "./resized_images_640x640"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
image_extensions = [".jpg", ".jpeg", ".png"]

# Resize loop
for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")  # Ensure RGB

        # Resize to 640x640
        resized_img = img.resize((640, 640), Image.LANCZOS)

        # Save to output folder
        save_path = os.path.join(output_folder, filename)
        resized_img.save(save_path)

print(f"✅ All images resized to 640x640 and saved in '{output_folder}'")
