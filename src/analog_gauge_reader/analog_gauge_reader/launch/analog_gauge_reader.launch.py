from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='analog_gauge_reader',
            executable='gauge_reader_node',  # Replace with your executable name
            name='gauge_reader',
            output='screen',
            parameters=[
                {"detection_model_path": "models/gauge_detection_model.pt"},
                {"key_point_model_path": "models/key_point_model.pt"},
                {"segmentation_model_path": "models/segmentation_model.pt"},
                {"image_topic": "/camera/image_raw"},
                {"round_decimals": -1},
                {"latch": False},
                {"continuous": True}
            ],
        ),
    ])