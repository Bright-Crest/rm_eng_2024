#pragma once
// used for 2 objects and 12 key points whose order is specified
// When the model is changed, the following functions must be carefully checked and changed:

// system
#include <chrono>
#include <queue>

/// ros2
#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
/// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <cv_bridge/cv_bridge.h>

namespace video_recorder
{
    class VideoRecorderNode : public rclcpp::Node
    {
    public:
        explicit VideoRecorderNode(const rclcpp::NodeOptions &options);
        ~VideoRecorderNode();

    private:
        /// @brief the Callback function called by the subscriptor
        void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info);

    private:
        cv::VideoWriter out_{};
        image_transport::CameraSubscriber image_sub_;
        std::queue<cv_bridge::CvImageConstPtr> frame_queue_;
        std::thread process_thread_;
        };

} // namespace video_recorder