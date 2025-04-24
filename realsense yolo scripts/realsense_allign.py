import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
from ultralytics import YOLO

# Load custom YOLOv8 model (detection only, no segmentation)
# model = YOLO("../best.pt")
model = YOLO("../payload_hugo_model.pt")

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device}")

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline
profile = pipeline.start(config)

# Get depth sensor scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is:", depth_scale)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Get camera intrinsics
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy
print("Camera Intrinsics:", fx, fy, cx, cy)

# Generate random colors for each class
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(256), k=3) for _ in classes_ids]

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color frame to BGR (YOLOv8 expects this)
        img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Resize to (640, 640) for YOLOv8 compatibility
        img_resized = cv2.resize(img, (640, 640))

        # Convert to tensor, reorder dimensions, normalize
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img_tensor /= 255.0  # Normalize pixel values

        # Perform YOLOv8 detection on GPU
        results = model.predict(img_tensor, conf=0.5)

        # Scale factors to map YOLO output (640x640) back to RealSense frame (640x480)
        scale_x = 640 / 640  # Width remains the same
        scale_y = 480 / 640  # Height must be scaled down

        # Process detected objects
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            names = result.names

            for box in boxes:
                if box is None:
                    continue

                # Rescale bounding box coordinates to match 640x480 RealSense frame
                x1, y1, x2, y2 = map(int, [
                    box.xyxy[0][0] * scale_x,
                    box.xyxy[0][1] * scale_y,
                    box.xyxy[0][2] * scale_x,
                    box.xyxy[0][3] * scale_y
                ])

                # Compute object center pixel coordinates
                u, v = (x1 + x2) // 2, (y1 + y2) // 2

                # Define small region around the center to compute depth (avoid outliers)
                region_size = 5
                roi = depth_image[max(0, v - region_size):min(480, v + region_size),
                                  max(0, u - region_size):min(640, u + region_size)]
                
                # Flatten and remove invalid depths
                valid_depths = roi[roi > 0].flatten()

                if len(valid_depths) > 0:
                    # Remove outliers using percentiles
                    lower = np.percentile(valid_depths, 10)
                    upper = np.percentile(valid_depths, 90)
                    inlier_depths = valid_depths[(valid_depths >= lower) & (valid_depths <= upper)]
                    Z = np.mean(inlier_depths) * depth_scale if len(inlier_depths) > 0 else 0
                else:
                    Z = 0

                # Compute real-world 3D coordinates
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                # Get class information
                class_id = int(box.cls[0])
                class_name = names[class_id]
                color = colors[classes_ids.index(class_id)]

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Display class name, confidence, and 3D position
                label = f"{class_name} {box.conf[0]:.2f} | X:{X:.2f}m Y:{Y:.2f}m Z:{Z:.2f}m"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 2)

        # Show results
        cv2.imshow('YOLOv8 + RealSense', img)

        # Press 'q' to quit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream stopped by user.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
