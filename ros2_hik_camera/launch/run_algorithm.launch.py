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
    # get parameters from <Upper launch.py> or <CLI>
    declare_is_serial_used = DeclareLaunchArgument(name = 'is_serial_used',
                          default_value = 'false')
    declare_is_gpu = DeclareLaunchArgument(name = 'is_gpu',
                          default_value = 'true')
    declare_is_debug = DeclareLaunchArgument(name = 'is_debug',
                          default_value = 'true')
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
                plugin = 'hik_camera::HikCameraNode',
                name = 'hik_camera_node',
                parameters =[{'camera_info_url':LaunchConfiguration('camera_info_url')}],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package = 'hik_camera',
                plugin = 'image_process::ImageProcessNode',
                name = 'image_process_node',
                parameters =[{'is_serial_used':LaunchConfiguration('is_serial_used')},
                             {'is_gpu':LaunchConfiguration('is_gpu')},
                             {'is_debug':LaunchConfiguration('is_debug')}
                             ],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        ],
        output = 'screen',
    )

    return LaunchDescription([
        declare_params_file,
        declare_camera_info,
        declare_is_serial_used,
        declare_is_gpu,
        declare_is_debug,
        container
        ])
