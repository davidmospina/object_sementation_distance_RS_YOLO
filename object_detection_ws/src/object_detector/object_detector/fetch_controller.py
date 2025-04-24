import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

class FetchController(Node):

    def __init__(self):
        super().__init__('fetch_controller')

        # Subscribe to the transformed payload pose
        self.subscription = self.create_subscription(
            PoseStamped,
            '/payload_pose',
            self.pose_callback,
            10
        )

        # Nav2 goal action client
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initially, fetch mode is OFF
        self.fetch_mode = True  # You can start as False and toggle later

    def pose_callback(self, msg: PoseStamped):
        if not self.fetch_mode:
            self.get_logger().info("Fetch mode off. Ignoring payload.")
            return

        self.get_logger().info(f"Payload detected at ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.send_goal_to_nav2(msg)

    def send_goal_to_nav2(self, pose: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        # Wait for the action server
        self.action_client.wait_for_server()

        # Send goal asynchronously
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
        self.get_logger().info(f"Navigation completed with status: {result}")
        # Optional: toggle fetch_mode off here

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
