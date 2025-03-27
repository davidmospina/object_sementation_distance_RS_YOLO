from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("best.pt")

# Run inference on an image
results = model.predict(source="image_2.jpg", conf=0.5)

# Loop through results and display them
for result in results:
    result.show()  # This works for individual result objects
