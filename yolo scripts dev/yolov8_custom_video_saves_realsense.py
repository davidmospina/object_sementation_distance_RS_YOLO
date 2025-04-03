import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
# Load trained YOLOv8 model
model = YOLO("best.pt")  # Change this if needed

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Set color stream resolution & FPS

# Start streaming
pipeline.start(config)

# Get video properties
frame_width = 640
frame_height = 480
fps = 30  # Ensure it matches your camera's frame rate

# Define the codec and create VideoWriter object
output_video_path = "output_realsense.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("🎥 Starting RealSense camera... Press 'q' to stop.")

try:
    while True:
        # Get frame from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert frame to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 inference
        results = model.predict(frame, conf=0.5)

        # Draw bounding boxes on the frame
        for result in results:
            frame = result.plot()  # Get image with detections

        # Show the frame in real-time
        cv2.imshow("YOLOv8 RealSense Detection", frame)

        # Write the annotated frame to the output video file
        out.write(frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    # Release resources
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed video saved as: {output_video_path}")
