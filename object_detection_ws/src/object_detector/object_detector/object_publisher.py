import rclpy
from rclpy.node import Node
from object_msgs.msg import Object, ObjectArray
import torch
import pyrealsense2 as rs
from ultralytics import YOLO
import random
import numpy as np
import cv2
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge






class ObjectPublisher(Node):

    def __init__(self):
        super().__init__('object_publisher')
        

        # Load YOLOv8 model
        MODEL_PATH = str(Path(__file__).parent / "best.pt")
        self.model = YOLO(MODEL_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.get_logger().info(f"Using device: {self.device}")

        # Setup RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

        # Depth alignment
        self.align = rs.align(rs.stream.color)

        # Depth scale and intrinsics
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intrinsics.fx, intrinsics.fy
        self.cx, self.cy = intrinsics.ppx, intrinsics.ppy

        
        #Fetch classes from the model and assign boundinbox colors
        self.yolo_classes = list(self.model.names.values())
        self.get_logger().info(f"Classes: {self.yolo_classes}")
        self.classes_ids = [self.yolo_classes.index(clas) for clas in self.yolo_classes]
        self.colors = [random.choices(range(256), k=3) for _ in self.classes_ids]


        self.publisher_ = self.create_publisher(ObjectArray, 'detected_objects', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()




    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #Preparing image to be publish in a topic
        ros_image = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')

        img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img, (640, 640))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img_tensor /= 255.0

        results = self.model.predict(img_tensor, conf=0.5)
        scale_x = 640 / 640
        scale_y = 480 / 640

        msg = ObjectArray()
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            names = result.names

            for box in boxes:
                x1, y1, x2, y2 = map(int, [
                    box.xyxy[0][0] * scale_x,
                    box.xyxy[0][1] * scale_y,
                    box.xyxy[0][2] * scale_x,
                    box.xyxy[0][3] * scale_y
                ])

                u, v = (x1 + x2) // 2, (y1 + y2) // 2
                region_size = 5
                roi = depth_image[max(0, v - region_size):min(480, v + region_size),
                                  max(0, u - region_size):min(640, u + region_size)]
                valid_depths = roi[roi > 0].flatten()

                if len(valid_depths) > 0:
                    lower = np.percentile(valid_depths, 10)
                    upper = np.percentile(valid_depths, 90)
                    inlier_depths = valid_depths[(valid_depths >= lower) & (valid_depths <= upper)]
                    Z = np.mean(inlier_depths) * self.depth_scale if len(inlier_depths) > 0 else 0
                else:
                    Z = 0

                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy

                class_id = int(box.cls[0])
                obj = Object()
                obj.class_name = names[class_id]
                obj.confidence = float(box.conf[0])
                obj.x = float(X)
                obj.y = float(Y)
                obj.z = float(Z)
                msg.objects.append(obj)
                
        self.image_pub.publish(ros_image)
        self.publisher_.publish(msg)
        
    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    object_publisher = ObjectPublisher()

    try:

        rclpy.spin(object_publisher)

    except KeyboardInterrupt:
            pass
    

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()