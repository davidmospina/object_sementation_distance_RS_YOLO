import os
import xml.etree.ElementTree as ET

# Define dataset paths
DATASET_DIR = "Hospital_Scene_Data-main"
IMAGES_DIR = os.path.join(DATASET_DIR, "image")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_LABELS_DIR = os.path.join(DATASET_DIR, "labels_yolo")

# Create output directory for YOLO labels
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# Define class names (adjust as needed)
CLASS_NAMES = [
    "sofa",
    "chair",
    "foot board",
    "overbed table",
    "hospital bed",
    "staff",
    "door handle",
    "table",
    "bedside monitor",
    "iv pole",
    "surgical light",
    "breathing tube",
    "wheel chair",
    "patient",
    "drawer",
    "mouse",
    "computer",
    "bedrail",
    "curtain",
    "keyboard",
    "infusion pump",
    "ventilator",
    "utility cart",
    "panda baby warmer",
    "visitor",
    "dispenser",
    "medical drawer",
    "handle",
    "countertop",
    "cabinet",
    "waste_bin",
    "faucet",
    "TV",
    "telephone",
    "syringe pump",
    "light switch",
    "elevator panel",
    "counter",
    "medical waste container",
    "push latch",
    "operating bed",
    "electrosurgical unit",
    "sink",
    "restroom assist bar",
    "incubator",
    "exam table",
    "bedside table",
    "hallway assist bar",
    "sequential compression",
    "toilet",
    "toilet handle",
    "person",
    "press to open",
    "surgical instrument",
    "xray machine",
    "xray bed"
]

def convert_bbox(size, box):
    """Convert Pascal VOC bbox to YOLO format (normalized x_center, y_center, width, height)."""
    width, height = size

    # Convert values to integers (fixing float issue)
    xmin, ymin, xmax, ymax = map(lambda x: int(float(x)), box)

    # Normalize values
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return (x_center, y_center, box_width, box_height)


def process_xml_file(xml_file, output_txt_file):
    """Parse XML file and convert it to YOLO format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image size
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    with open(output_txt_file, "w") as txt_file:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in CLASS_NAMES:
                continue  # Skip unknown classes

            class_id = CLASS_NAMES.index(class_name)  # Convert class name to ID
            bndbox = obj.find("bndbox")
            bbox = [
                bndbox.find("xmin").text,
                bndbox.find("ymin").text,
                bndbox.find("xmax").text,
                bndbox.find("ymax").text,
            ]
            x_center, y_center, box_width, box_height = convert_bbox((img_width, img_height), bbox)
            
            # Write to file in YOLO format
            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

def convert_dataset():
    """Iterate over label directories and convert XML files."""
    for label_folder in sorted(os.listdir(LABELS_DIR)):
        label_path = os.path.join(LABELS_DIR, label_folder)

        # Ensure it's a directory
        if not os.path.isdir(label_path):
            continue

        # Create corresponding output directory
        output_path = os.path.join(OUTPUT_LABELS_DIR, label_folder)
        os.makedirs(output_path, exist_ok=True)

        for xml_file in sorted(os.listdir(label_path)):
            if not xml_file.endswith(".xml"):
                continue
            
            xml_path = os.path.join(label_path, xml_file)
            txt_filename = os.path.splitext(xml_file)[0] + ".txt"
            output_txt_path = os.path.join(output_path, txt_filename)
            
            process_xml_file(xml_path, output_txt_path)

    print(f"✅ Conversion complete! YOLO annotations saved in: {OUTPUT_LABELS_DIR}")

# Run conversion
convert_dataset()
