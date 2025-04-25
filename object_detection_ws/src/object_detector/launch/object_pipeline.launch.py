from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_detector',
            executable='object_publisher',
            name='object_publisher',
            output='screen'
        ),
        # Node(
        #     package='object_detector',
        #     executable='object_subscriber',
        #     name='object_subscriber',
        #     output='screen'
        # ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_static_tf',
            arguments=['0.1', '0.0', '0.2', '0', '0', '0', '1', 'base_link', 'camera_link']
        ),
        Node(
            package='object_detector', 
            executable='fetch_controller',
            name='fetch_controller',
            output='screen'
        )
    ])
