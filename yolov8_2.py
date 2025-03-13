from ultralytics import YOLO
import random
import cv2
import numpy as np

# Load YOLOv8 segmentation model
model = YOLO("yolov8m-seg.pt")

# Read input image
img = cv2.imread("./image_1.jpeg")

# Get YOLO class names
yolo_classes = list(model.names.values())

# Get class IDs for all YOLO classes
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

# Confidence threshold
conf = 0.5

# Perform YOLOv8 segmentation
results = model.predict(img, conf=conf)

# Generate random colors for each class
colors = [random.choices(range(256), k=3) for _ in classes_ids]

# Process each detected object
for result in results:
    # masks = result.masks  # Get segmentation masks
    # boxes = result.boxes  # Get bounding boxes
    # names = result.names  # Get class names

    print("Detected Classes:", names)  # Print detected class names

    # Iterate over each mask and bounding box
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])  # Convert to integer points

        # Get the class index
        class_id = int(box.cls[0])  # Convert to int
        class_name = names[class_id]  # Get the class label
        color = colors[classes_ids.index(class_id)]  # Assign color

        # Draw the segmentation mask
        cv2.fillPoly(img, points, color)

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Display the class name above the bounding box
        label = f"{class_name} {box.conf[0]:.2f}"  # Label with confidence score
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Show the final segmented image with bounding boxes and labels
cv2.imshow("Segmented Image", img)
cv2.waitKey(0)

# Save the output image
cv2.imwrite("segmented_output.jpg", img)
