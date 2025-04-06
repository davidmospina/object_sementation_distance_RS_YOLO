from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    rviz_config_path = os.path.join( os.getenv('HOME'), 'yolov8', 'object_detection_ws', 'rviz', 'object_pipeline.rviz')
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
        ),
        Node(
            package='object_detector',
            executable='static_tf_broadcaster',
            name='static_tf_broadcaster',
            output='screen'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        ),
    ])
