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
        Node(
            package='object_detector',
            executable='object_subscriber',
            name='object_subscriber',
            output='screen'
        )
    ])
