#include "video_recorder_node.hpp"
#include <ctime>
#include <iomanip>

namespace video_recorder
{
    VideoRecorderNode::VideoRecorderNode(const rclcpp::NodeOptions &options) : Node("video_recorder", options)
    {
        bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
        auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;

        std::string package_share_directory = ament_index_cpp::get_package_share_directory("hik_camera");
        image_sub_ = image_transport::create_camera_subscription(this, "image_raw",
                                                                 std::bind(&VideoRecorderNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2),
                                                                 "raw", qos);
        // 获取当前时间作为文件名
        std::time_t current_time = std::time(nullptr);
        std::tm *local_time = std::localtime(&current_time);
        std::stringstream filename_stream;
        filename_stream << package_share_directory+"/../../../../" << std::put_time(local_time, "%Y-%m-%d_%H-%M-%S") << ".avi";
        std::string output_file = filename_stream.str();

        // 设置视频编解码器和输出文件名
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        cv::Size frame_size(1280, 768);
        out_ = cv::VideoWriter(output_file, fourcc, 20.0, frame_size);
        process_thread_ = std::thread{[this]() -> void
                                      {
                                          while (rclcpp::ok())
                                          {
                                          }
                                      }};

        RCLCPP_INFO(this->get_logger(), "Starting Image Process!");
    }

    VideoRecorderNode::~VideoRecorderNode()
    {
        image_sub_.shutdown();
        out_.release();
        if (process_thread_.joinable())
            process_thread_.join();
        RCLCPP_INFO(this->get_logger(), "VideoRecorderNode destroyed!");
    }

    void VideoRecorderNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info)
    {
        try
        {
            cv_bridge::CvImageConstPtr frame = cv_bridge::toCvShare(msg, "rgb8");
            cv::Mat image = frame->image;
            cv::Mat bgrFrame;
            cv::cvtColor(image, bgrFrame, cv::COLOR_RGB2BGR);
            out_.write(bgrFrame);
        }
        catch (const cv_bridge::Exception &e)
        {
            auto logger = rclcpp::get_logger("VideoRecorder Error");
            RCLCPP_ERROR(logger, "Could not convert from sensor_msgs::msg::Image of encoding '%s' to cv::Mat of encoding 'bgr8'.", msg->encoding.c_str());
        }
    }
} // namespace video_recorder

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(video_recorder::VideoRecorderNode)
