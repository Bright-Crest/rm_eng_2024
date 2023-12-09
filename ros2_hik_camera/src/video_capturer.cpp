#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/image.hpp>

/// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

namespace video_capturer
{
  class VideoCapturerNode : public rclcpp::Node
  {
  public:
    explicit VideoCapturerNode(const rclcpp::NodeOptions &options) : Node("video_capturer", options)
    {
      /// The prefix is our package share directory
      /// ${your_colcon_ws{/install/${Package_name}/share/
      std::string video_url =
          this->declare_parameter("video_url", "");
      if (video_url.empty())
      {
        RCLCPP_ERROR(this->get_logger(), "Video URL Error");
        return;
      }
      cv_cap_ = cv::VideoCapture(video_url);
      if (!cv_cap_.isOpened())
      {
        RCLCPP_ERROR(this->get_logger(), "Video Opening Error");
        return;
      }
      frame_cnt_ = cv_cap_.get(cv::CAP_PROP_FRAME_COUNT);

      RCLCPP_INFO(this->get_logger(), "Starting VideoCapturerNode!");
      bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
      auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
      video_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);

      capture_thread_ = std::thread{[this]() -> void
                                    {
                                      cv::Mat frame;
                                      RCLCPP_INFO(this->get_logger(), "Publishing video!");
                                      std_msgs::msg::Header hdr;

                                      while (rclcpp::ok())
                                      {
                                        cv_cap_.read(frame);
                                        if (!frame.empty())
                                        {
                                          image_msg_ptr_ = cv_bridge::CvImage(hdr, "bgr8", frame).toImageMsg();
                                          video_pub_.publish(*image_msg_ptr_, camera_info_msg_);
                                          cv::waitKey(10);
                                        }
                                        else
                                        {
                                          cv_cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                                          RCLCPP_INFO(this->get_logger(), "Recycling Video!");
                                        }
                                      }
                                    }};
    }

    ~VideoCapturerNode()
    {
      if (capture_thread_.joinable())
        capture_thread_.join();

      RCLCPP_INFO(this->get_logger(), "VideoCapturerNode destroyed!");
    }

  private:
    cv::VideoCapture cv_cap_;
    int frame_cnt_;

    sensor_msgs::msg::Image::SharedPtr image_msg_ptr_;
    sensor_msgs::msg::CameraInfo camera_info_msg_;
    image_transport::CameraPublisher video_pub_;

    std::thread capture_thread_;
  };
} // namespace video_capturer

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(video_capturer::VideoCapturerNode)
