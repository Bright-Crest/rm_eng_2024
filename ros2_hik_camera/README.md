# ros2_hik_camera

A ROS2 packge for Hikvision USB3.0 industrial camera

## Usage

```
ros2 launch hik_camera hik_camera.launch.py
```
You can run videostreaming to publish image_raw in the form of node by using the command below:
```
ros2 run hik_camera video_capturer_node --ros-args -p video_url:=${YOUR_VIDEO_URL}
```
And also, videostreaming could run in the form of `Composition` to achieve IPC.
```
ros2 run rclcpp_components component_container
ros2 component load /ComponentManager hik_camera video_capturer::VideoCapturerNode -p video_url:=${YOUR_VIDEO_URL}
```

## Params

- exposure_time
- gain
