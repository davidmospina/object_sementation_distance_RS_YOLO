import rclpy
from rclpy.node import Node
from object_msgs.msg import ObjectArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ObjectSubscriber(Node):
    def __init__(self):
        super().__init__('object_subscriber')

        self.bridge = CvBridge()
        self.image = None
        self.objects = []

        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.objects_sub = self.create_subscription(
            ObjectArray,
            'detected_objects',
            self.objects_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.visualize_callback)

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def objects_callback(self, msg):
        self.objects = msg.objects

    def visualize_callback(self):
        if self.image is None:
            return

        img = self.image.copy()

        for obj in self.objects:

            x, y, z = obj.x, obj.y, obj.z

            # fx = 378.39862060546875
            # fy = 378.03509521484375
            # cx = 321.93994140625
            # cy = 243.1790313720703


            #small camera:
            fx = 616.7725830078125
            fy = 616.8902587890625
            cx = 319.1792297363281
            cy = 251.24098205566406
            

            u = int((x * fx / z) + cx)
            v = int((y * fy / z) + cy)

           

            cv2.circle(img, (u, v), 5, (0, 255, 0), -1)
            label = f"{obj.class_name} {obj.confidence:.2f} ({x:.2f},{y:.2f},{z:.2f})"
            cv2.putText(img, label, (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow("Detected Objects", img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
