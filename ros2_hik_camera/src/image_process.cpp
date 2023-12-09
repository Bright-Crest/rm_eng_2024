#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/image.hpp>

/// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <cv_bridge/cv_bridge.h>

namespace image_process
{
  class ImageProcessNode : public rclcpp::Node
  {
  public:
    explicit ImageProcessNode(const rclcpp::NodeOptions &options) : Node("image_process", options)
    {
      bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
      auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
      image_sub_ = image_transport::create_camera_subscription(this, "image_raw",
                                                               std::bind(&ImageProcessNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2),
                                                               "raw", qos);
      process_thread_ = std::thread{[this]() -> void
                                    {
                                      while (rclcpp::ok())
                                      {
                                        if (!frame_.empty())
                                        {
                                          std::lock_guard<std::mutex> lock(g_mutex);
                                          /*
                                            add your process code here
                                          */
                                          RCLCPP_INFO(this->get_logger(), "Processing!");
                                        }
                                      }
                                    }};
      RCLCPP_INFO(this->get_logger(), "Starting Image Process!");
    }

    ~ImageProcessNode()
    {
      image_sub_.shutdown();
      if (process_thread_.joinable())
        process_thread_.join();
      RCLCPP_INFO(this->get_logger(), "ImageProcessNode deestroyed!");
    }

  private:
    image_transport::CameraSubscriber image_sub_;
    cv::Mat frame_;
    std::thread process_thread_;
    std::mutex g_mutex;

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg,
                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camara_info)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      try
      {
        frame_ = cv_bridge::toCvShare(msg, "bgr8")->image;
      }
      catch (const cv_bridge::Exception &e)
      {
        auto logger = rclcpp::get_logger("ImageProcess Error");
        RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
      }
    }
  };
} // namespace image_process

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(image_process::ImageProcessNode)
