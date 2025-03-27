# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from object_msgs.msg import Object, ObjectArray
from std_msgs.msg import String


class ObjectSubscriber(Node):

    def __init__(self):
        super().__init__('object_subscriber')
        self.subscription = self.create_subscription(
            ObjectArray,
            'detected_objects',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        for obj in msg.objects:
            self.get_logger().info(
                f"Detected {obj.class_name} | Confidence: {obj.confidence:.2f} | X:{obj.x:.2f} Y:{obj.y:.2f} Z:{obj.z:.2f}"
            )



def main(args=None):
    rclpy.init(args=args)

    object_subscriber = ObjectSubscriber()

    rclpy.spin(object_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
