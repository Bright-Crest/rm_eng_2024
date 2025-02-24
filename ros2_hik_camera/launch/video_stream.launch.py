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

    params_file = os.path.join(
        get_package_share_directory('hik_camera'), 'config', 'camera_params.yaml')
    camera_info_url = 'package://hik_camera/config/camera_info.yaml'

    default_video_url = '/home/pan/TEST_IMG/2.jpg'
    #default_video_url = '/home/pan/rm.cv.eng/TEST/3.mp4'

    # get parameters from <Upper launch.py> or <CLI>
    declare_is_gpu = DeclareLaunchArgument(name = 'is_gpu',
                          default_value = 'false')
    declare_video_url_cmd = DeclareLaunchArgument(name = 'video_url',
                          default_value = default_video_url)

    declare_is_serial_used = DeclareLaunchArgument(name = 'is_serial_used',
                          default_value = 'false')
    declare_params_file =  DeclareLaunchArgument(name='params_file',
                              default_value=params_file)
    declare_camera_info =  DeclareLaunchArgument(name='camera_info_url',
                              default_value=camera_info_url)

    container = ComposableNodeContainer(
        name = 'Vision_Component_Container',
        namespace = '',
        package = 'rclcpp_components',
        executable = 'component_container',
        composable_node_descriptions = [
            ComposableNode(
                package = 'hik_camera',
                plugin = 'video_capturer::VideoCapturerNode',
                name = 'video_capturer_node',
                parameters =[{'video_url':LaunchConfiguration('video_url')},
                             {'camera_info_url':LaunchConfiguration('camera_info_url')}]
            ),
            ComposableNode(
                package = 'hik_camera',
                plugin = 'image_process::ImageProcessNode',
                name = 'image_process_node',
                parameters =[{'is_serial_used':LaunchConfiguration('is_serial_used')},
                             {'is_gpu':LaunchConfiguration('is_gpu')}]
            )
        ],
        output = 'screen',
    )

    return LaunchDescription([
        declare_params_file,
        declare_camera_info,
        declare_is_serial_used,
        declare_video_url_cmd,
        declare_is_gpu,
        container
        ])
