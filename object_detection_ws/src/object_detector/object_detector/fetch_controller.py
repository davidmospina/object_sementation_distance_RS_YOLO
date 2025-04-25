import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import GoalStatus
import tf2_ros
from std_srvs.srv import SetBool
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

class FetchController(Node):

    def __init__(self):
        super().__init__('fetch_controller')

        self.subscription = self.create_subscription(
            PoseStamped,
            '/payload_pose',
            self.pose_callback,
            10
        )

        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        self.srv = self.create_service(SetBool, 'toggle_fetch_mode', self.toggle_fetch_mode_callback)


        self.fetch_mode = False  # You can toggle this later

    def toggle_fetch_mode_callback(self, request, response):
        self.fetch_mode = request.data
        response.success = True
        response.message = f"Fetch mode set to {self.fetch_mode}"
        self.get_logger().info(f"Fetch mode is now {'ON' if self.fetch_mode else 'OFF'}")
        return response


    def pose_callback(self, msg: PoseStamped):
        if not self.fetch_mode:
            self.get_logger().info("Fetch mode off. Ignoring payload.")
            return

        try:
            # Transform pose to map frame
            transform = self.tf_buffer.lookup_transform(
                'map',  # Target frame
                msg.header.frame_id,  # Source frame (camera_link or base_link)
                rclpy.time.Time()
            )
            transformed_pose = do_transform_pose(msg, transform)
            transformed_pose.header.frame_id = 'map'
            self.get_logger().info(
                f"Transformed pose: ({transformed_pose.pose.position.x:.2f}, {transformed_pose.pose.position.y:.2f})"
            )
            self.send_goal_to_nav2(transformed_pose)

        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")

    def send_goal_to_nav2(self, pose: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info("Waiting for Nav2 action server...")
        self.action_client.wait_for_server()
        self.get_logger().info("Sending goal...")

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2')
            return

        self.get_logger().info('Goal accepted! Navigating...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("✅ Navigation SUCCESSFUL: Robot reached the payload.")
            # Optional: toggle fetch mode off after success
            self.fetch_mode = False
            self.get_logger().info("Fetch mode is now OFF after success.")
        else:
            self.get_logger().warn(f"❌ Navigation FAILED with status code: {status}")
            # Optionally retry here or handle failure

def main(args=None):
    rclpy.init(args=args)
    node = FetchController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()



#To test next day: 

# ros2 service call /toggle_fetch_mode std_srvs/srv/SetBool "{data: true}"
# ros2 service call /toggle_fetch_mode std_srvs/srv/SetBool "{data: false}"
