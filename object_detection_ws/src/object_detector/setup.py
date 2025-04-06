from setuptools import find_packages, setup

package_name = 'object_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={'object_detector': ['best.pt']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/object_pipeline.launch.py']),
    ],
    install_requires=['setuptools', 'opencv-python'],
    zip_safe=True,
    maintainer='david',
    maintainer_email='davidmospinae@gmail.com',
    description='object detection with location based on yolov8, custom dataset and realsense camera',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_publisher = object_detector.object_publisher:main',
            'object_subscriber = object_detector.object_subscriber:main',
            'static_tf_broadcaster = object_detector.static_tf_broadcaster:main',
        ],
    },
)
