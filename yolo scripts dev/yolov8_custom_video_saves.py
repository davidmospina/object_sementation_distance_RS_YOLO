import cv2
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO("best.pt")  # Change this if needed

# Open video file
video_path = "video_1.mp4"  # Change this to your actual video file
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save output
output_video_path = "output_video_with_annotations.mp4"
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

    # Show the frame in real-time
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Write the annotated frame to the output video file
    out.write(frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Processed video saved as: {output_video_path}")
