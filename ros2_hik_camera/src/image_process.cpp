#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/image.hpp>

/// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <cv_bridge/cv_bridge.h>

/// yolov8
#include "yolov8_inference/inference.h"

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
        // ? toCvShare return a immutable image?
        // frame_ = cv_bridge::toCvShare(msg, "bgr8")->image;
        frame_ = cv_bridge::toCvCopy(msg, "bgr8:CV_8UC3")->image;
      }
      catch (const cv_bridge::Exception &e)
      {
        auto logger = rclcpp::get_logger("ImageProcess Error");
        RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
      }
    }
  };

  class ImageProcessor
  {
  private:
    static const double length_{126.5};
    static const cv::Matx33d object_points_;

    yolov8::Inference model_;
    std::vector<yolov8::Detection> predict_result_;

    static cv::Matx33d camera_matrix_;
    static cv::Vec<double, 5>  distortion_matrix_;
    
    cv::Vec3d rvec_;
    cv::Vec3d tvec_;

    bool Determine4PointsOrder(std::vector<cv::Point2f> &image_points, std::vector<std::string> &classes, cv::Point2f *center_p = nullptr);
    
  public:
    ImageProcessor(const std::string &model_path, const cv::Size &model_shape = {640, 480},
                   const float &model_score_threshold = 0.45, const float &model_nms_threshold = 0.50);
    ~ImageProcessor() = default;

    inline void ModelPredict(const cv::Mat &image) { model_.runInference(image, predict_result_); }
    bool SolvePnP();
    inline std::pair<cv::Vec3d, cv::Vec3d> OutputRvecTvec() { return std::make_pair(rvec_, tvec_); }

    cv::Point2f Object2Picture(const cv::Point3f &point);
  };

  const cv::Matx33d _tmp{1.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 1.0, 0.0};
  const cv::Matx33d ImageProcessor::object_points_{ImageProcessor::length_ * _tmp};

  bool ImageProcessor::Determine4PointsOrder(std::vector<cv::Point2f> &image_points, std::vector<std::string> &classes, cv::Point2f *center_p)
  {
      return false;
  }

  ImageProcessor::ImageProcessor(const std::string &model_path, const cv::Size &model_shape,
                                 const float &model_score_threshold, const float &model_nms_threshold)
  {
    model_.init(model_path, model_shape, model_score_threshold, model_nms_threshold);
  }

  bool ImageProcessor::SolvePnP()
  {
      return false;
  }

  cv::Point2f ImageProcessor::Object2Picture(const cv::Point3f &point)
  {
      return cv::Point2f();
  }

} // namespace image_process

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(image_process::ImageProcessNode)
