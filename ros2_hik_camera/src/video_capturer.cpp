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
      std::string video_url = this->declare_parameter("video_url", "");

      if (video_url.empty())
      {
        RCLCPP_ERROR(this->get_logger(), "Video URL Error");
        return;
      }

      // get suffix name
      bool is_video = true;
      cv::Mat frame_temp;
      std::string suffix = video_url.substr(video_url.find_last_of('.') + 1);
      if (suffix != "mp4")
        is_video = false;

      if (is_video)
      {
        cv_cap_ = cv::VideoCapture(video_url);
        if (!cv_cap_.isOpened())
        {
          RCLCPP_ERROR(this->get_logger(), "Video Opening Error");
          return;
        }
        frame_cnt_ = cv_cap_.get(cv::CAP_PROP_FRAME_COUNT);
        RCLCPP_ERROR(this->get_logger(), "Video Opening Success");
      }
      else
      {
        try
        {
          frame_temp = cv::imread(video_url);
        }
        catch (...)
        {
          RCLCPP_ERROR(this->get_logger(), "Incorrect photo file suffix");
        }
      }

      RCLCPP_INFO(this->get_logger(), "Starting VideoCapturerNode!");
      bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
      auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
      video_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);

      // Magic Code: if you delete this, then it is impossible to open video_capturer node
      // before image process node.
      {
        cv::Mat magic_mat(cv::Size(20, 20), CV_8UC1, 255);
        cv::Rect magic_rec(10, 10, 5, 5);
        cv::rectangle(magic_mat, magic_rec, cv::Scalar(255, 255, 255), 2);
      }

      capture_thread_ = std::thread{[this, is_video, frame_temp]() -> void
                                    {
                                      RCLCPP_INFO(this->get_logger(), "Publishing video!");
                                      std_msgs::msg::Header hdr;
                                      cv::Mat frame = frame_temp;

                                      while (rclcpp::ok())
                                      {
                                        if (is_video)
                                          cv_cap_.read(frame);
                                        if (!frame.empty())
                                        {
                                          image_msg_ptr_ = cv_bridge::CvImage(hdr, "bgr8", frame).toImageMsg();
                                          video_pub_.publish(*image_msg_ptr_, camera_info_msg_);
                                          cv::waitKey(30);
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
