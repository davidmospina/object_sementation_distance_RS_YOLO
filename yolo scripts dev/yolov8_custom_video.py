import cv2
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO("best.pt")  # Change this if needed

# Open video file (change 'video.mp4' to your video file)
video_path = "video_1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save output
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process video frame-by-frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends

    # Run YOLOv8 inference
    results = model.predict(frame, conf=0.5)

    # Draw bounding boxes on the frame
    for result in results:
        frame = result.plot()  # Get image with detections

    # Show the frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Ensure the window updates properly
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # Press 'q' to exit early

    # Write the frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Processed video saved as: {output_video_path}")
