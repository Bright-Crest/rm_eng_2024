import os

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    default_video_url = '/home/pan/rm.cv.eng/TEST/2.mp4'

    # get parameters from <Upper launch.py> or <CLI>
    declare_video_url_cmd = DeclareLaunchArgument(name = 'video_url',
                          default_value = default_video_url)

    declare_is_serial_used = DeclareLaunchArgument(name = 'is_serial_used',
                          default_value = 'false')

    container = ComposableNodeContainer(
        name = 'Vision_Component_Container',
        namespace = '',
        package = 'rclcpp_components',
        executable = 'component_container',
        composable_node_descriptions = [
            ComposableNode(
                package = 'hik_camera',
                plugin = 'hik_camera::HikCameraNode',
                name = 'hik_camera_node',
            ),
            ComposableNode(
                package = 'hik_camera',
                plugin = 'image_process::ImageProcessNode',
                name = 'image_process_node',
                parameters =[{'is_serial_used':LaunchConfiguration('is_serial_used')}]
            )
        ],
        output = 'screen',
    )

    return LaunchDescription([
        declare_video_url_cmd,
        declare_is_serial_used,
        container
        ])
