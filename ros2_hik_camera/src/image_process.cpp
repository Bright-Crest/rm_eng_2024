// ABANDONDED
// used for 2 objects and 12 key points whose order is specified
// When the model is changed, the following functions must be carefully checked and changed:
// ImageProcessor::SolvePnP()

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
/// yolov8
#include "yolov8_inference/inference.h"
/// Serial
#include "serial/serial.h"
#include "serial/serial_format.h"

#define LEN_A 137.5f
#define LEN_B 87.5f

namespace image_process
{
  // imgmsg => cv Mat => yolov8 model => solvePnP
  class ImageProcessor
  {
  private:
    // for SolvePnP()
    static const std::vector<cv::Point3f> object_points_;

    // for SolvePnP() and AddImagePoints()
    std::vector<cv::Point2f> image_points_;
    // yolov8 model result
    std::vector<yolov8::Detection> predict_result_;

    // the default parameter from the Infantry MV-CS016-10UC
    cv::Matx33f camera_matrix_{1622.412925, 0.000000, 601.366953, 0.000000, 1619.246860, 414.637285, 0.000000, 0.000000, 1.000000};
    cv::Vec<float, 5> distortion_coefficients_{-0.144865, 0.216839, -0.002233, -0.003314, 0.000000};

    // sovlePnP result
    cv::Vec3f rvec_;
    // sovlePnP result
    cv::Vec3f tvec_;

    /// @brief called in SolvePnP(); determine the order of the four target points in the picture
    /// @param points 4 key points generated from the model
    /// @param classes the classes of the 4 key points
    /// @param center output; the central point of the 4 key points
    /// @param order output; the expected order of the 4 key points
    /// @return is success
    bool Determine4PointsOrder(const std::vector<cv::Point2f> &points, const std::vector<std::string> &classes, cv::Point2f &center, std::vector<int> &order);

  public:
    // for object_points_
    yolov8::Inference model_;

    // whether the node has already got the camera_matrix
    bool hasCalibrationCoefficient = false;

    ImageProcessor() = default;
    ImageProcessor(const std::string &model_path, const cv::Size &model_shape = {640, 640},
                   const float &model_score_threshold = 0.45f, const float &model_nms_threshold = 0.50f);
    ~ImageProcessor() = default;

    // unused
    void GetCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info);

    inline void ModelPredict(const cv::Mat &image) { model_.runInference(image, predict_result_); }
    // call after calling ModelPrdeict()
    // modify image_points_
    bool SolvePnP(int point_num);
    // call after calling SolvePnP()
    inline std::pair<cv::Vec3f, cv::Vec3f> OutputRvecTvec() { return std::make_pair(rvec_, tvec_); }

    /// @brief bring a 3d point from the object coordinate system to the picture(2d point)
    cv::Point2f Object2Picture(const cv::Point3f &point);

    /// @brief for testing yolov8 model; draw boxes and keypoints
    void AddPredictResult(cv::Mat img, bool add_boxes = true, bool add_keypoints = true);

    /// @brief for testing SolvePnP(), call only when SolvePnP() returns true; draw image_points_ and their order
    void AddImagePoints(cv::Mat img, bool add_order = true, bool add_points = true);

    /// @brief get tvec_
    cv::Vec3f getTvec();

    /// @brief get rvec_
    cv::Vec3f getRvec();
  };

  class ImageProcessNode : public rclcpp::Node
  {
  public:
    explicit ImageProcessNode(const rclcpp::NodeOptions &options) : Node("image_process", options)
    {
      bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
      auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;

      is_serial_used_ = this->declare_parameter("is_serial_used", false);
      if (is_serial_used_)
      {
        transport_serial_.setPort("/dev/ttyUSB0");
        transport_serial_.setBaudrate(115200);
        serial::Timeout _time = serial::Timeout::simpleTimeout(2000);
        transport_serial_.setTimeout(_time);
        transport_serial_.setStopbits(serial::stopbits_one);

        transport_serial_.open();
      }

      if (is_serial_used_)
      {
        if (!transport_serial_.isOpen())
        {
          RCLCPP_ERROR(this->get_logger(), "Serial Port is not open!");
          return;
        }
        else
          RCLCPP_INFO(this->get_logger(), "Seiral Port is open.");
      }
      else
        RCLCPP_INFO(this->get_logger(), "Not using Serial Port.");

      // may throw ament_index_cpp::PackageNotFoundError exception
      std::string package_share_directory = ament_index_cpp::get_package_share_directory("hik_camera");
      const std::string model_path = package_share_directory + "/model/best.onnx";

      img_processor_.model_.init(model_path, cv::Size(640, 640));

      image_sub_ = image_transport::create_camera_subscription(this, "image_raw",
                                                               std::bind(&ImageProcessNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2),
                                                               "raw", qos);

      processed_image_pub_ = image_transport::create_publisher(this, "processed_image", qos);

      process_thread_ = std::thread{[this]() -> void
                                    {
                                      uint8_t send_data_buffer[17];
                                      uint8_t pose_data[12];

                                      while (rclcpp::ok())
                                      {
                                        cv_bridge::CvImageConstPtr frame{};
                                        if (!frame_queue_.empty())
                                        {
                                          std::lock_guard<std::mutex> lock(g_mutex);
                                          // only use the newest frame
                                          frame = std::move(frame_queue_.back());
                                          frame_queue_.pop();

                                          // Warning : do not unlock mutex earlier, because the speed of in-queue is faster
                                          // largely than out-queue operation.
                                          // RCLCPP_INFO(this->get_logger(), "%d!", frame_queue_.size());

                                          img_processor_.ModelPredict(frame->image);
                                          img_processor_.AddPredictResult(frame->image, true, false);

                                          if (img_processor_.SolvePnP(12))
                                          {
                                            AddSolvePnPResult(frame->image);
                                            img_processor_.AddImagePoints(frame->image);
                                          }

                                          if (is_serial_used_)
                                          {
                                            // Debug
                                            send_data_buffer[0] = 0x23;
                                            send_data_buffer[1] = MsgStream::PC2BOARD | MsgType::AUTO_EXCHANGE;
                                            send_data_buffer[2] = 12;
                                            for (int i = 0; i < 3; ++i)
                                            {
                                              uint16_t send_temp = img_processor_.getTvec()[i] * 10.f;
                                              pose_data[2 * i] = send_temp;
                                              pose_data[2 * i + 1] = send_temp >> 8;
                                              send_temp = img_processor_.getRvec()[i] * 10000.f;
                                              pose_data[2 * i + 6] = send_temp;
                                              pose_data[2 * i + 1 + 6] = send_temp >> 8;
                                            }
                                            uint16_t crc_result = CRC16_Calc(pose_data, 12);
                                            send_data_buffer[15] = crc_result;
                                            send_data_buffer[16] = crc_result >> 8;
                                            uint8_t pack_offset = 3;
                                            for (int i = 0; i < 12; ++i)
                                            {
                                              send_data_buffer[i + pack_offset] = pose_data[i];
                                            }
                                            /*
                                            cv::Vec3f drawback;
                                            for (int i = 0; i < 3; ++i)
                                            {
                                              drawback[i] = (send_data_buffer[2 * i] | (send_data_buffer[2 * i + 1] << 8)) / 10.f;
                                            }
                                            for (int i = 0; i < 3; ++i)
                                            {
                                              drawback[i] = static_cast<int16_t>(send_data_buffer[2 * i + 6] | (send_data_buffer[2 * i + 1 + 6] << 8)) / 10000.f;
                                            }
                                            std::cout << "target: " << drawback << "\n";
                                            */

                                            transport_serial_.write(send_data_buffer, sizeof(send_data_buffer));
                                          }
                                          processed_image_pub_.publish(frame->toImageMsg());
                                          // flush the frame_queue_, not efficient
                                          frame_queue_ = {};
                                        }
                                      }
                                    }};

      RCLCPP_INFO(this->get_logger(), "Starting Image Process!");
    }

    ~ImageProcessNode()
    {
      image_sub_.shutdown();
      processed_image_pub_.shutdown();
      if (process_thread_.joinable())
        process_thread_.join();
      RCLCPP_INFO(this->get_logger(), "ImageProcessNode destroyed!");
    }

  private:
    image_transport::CameraSubscriber image_sub_;
    image_transport::Publisher processed_image_pub_;

    std::queue<cv_bridge::CvImageConstPtr> frame_queue_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;

    std::thread process_thread_;
    std::mutex g_mutex;

    ImageProcessor img_processor_;
    bool is_serial_used_;
    serial::Serial transport_serial_;

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      if (!img_processor_.hasCalibrationCoefficient)
      {
        // img_processor_.GetCameraInfo(camera_info);
        img_processor_.hasCalibrationCoefficient = true;
      }
      try
      {
        frame_queue_.push(cv_bridge::toCvShare(msg, "bgr8"));
      }
      catch (const cv_bridge::Exception &e)
      {
        auto logger = rclcpp::get_logger("ImageProcess Error");
        RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
      }
    }

    /// @brief for testing SolvePnP(); draw xyz coordinate system
    void AddSolvePnPResult(cv::Mat img)
    {
      cv::Point3f real_center{};
      std::vector<cv::Point3f> xyz_points;
      for (unsigned int i = 0; i < 3; ++i)
      {
        cv::Vec3f tmp{};
        tmp(i) = LEN_A;
        xyz_points.emplace_back(std::move(tmp));
      }

      cv::Point2i real_center_pic = img_processor_.Object2Picture(real_center);
      for (auto &point : xyz_points)
      {
        cv::Point2i point_pic = img_processor_.Object2Picture(point);
        cv::arrowedLine(img, real_center_pic, point_pic, cv::Scalar(255, 0, 255), 2);
      }
    }
  };

  const std::vector<cv::Point3f> ImageProcessor::object_points_{
      {LEN_A, LEN_B, 0.0f}, {LEN_A, LEN_A, 0.0f}, {LEN_B, LEN_A, 0.0f}, {-LEN_B, LEN_A, 0.0f}, {-LEN_A, LEN_A, 0.0f}, {-LEN_A, LEN_B, 0.0f}, {-LEN_A, -LEN_B, 0.0f}, {-LEN_A, -LEN_A, 0.0f}, {-LEN_B, -LEN_A, 0.0f}, {LEN_B, -LEN_A, 0.0f}, {LEN_A, -LEN_A, 0.0f}, {LEN_A, -LEN_B, 0.0f}};

  /// @details check classes and then take the special point(with 2 gaps) as the first and push others counterclockwise
  bool ImageProcessor::Determine4PointsOrder(const std::vector<cv::Point2f> &points, const std::vector<std::string> &classes, cv::Point2f &center, std::vector<int> &order)
  {
    // preprocess
    if (points.size() != 4)
      return false;
    if (classes.size() != 4)
      return false;

    // check classes; expected one special point and 3 other points of the same class

    // all class types
    std::set<std::string> classes_set{};
    for (const std::string &cls : classes)
      classes_set.emplace(cls);

    if (classes_set.size() != 2)
    {
      std::cerr << "Warning: ImageProcessor::Determine4PointsOrder(): more or less than 2 classes\n";
      return false;
    }

    int special_idx = 0;
    std::string special_class{};

    // get the special index
    int tmp_count = std::count(classes.begin(), classes.end(), *classes_set.begin());
    switch (tmp_count)
    {
    case 1:
      special_class = *classes_set.begin();
      break;
    case 3:
      special_class = *classes_set.rbegin();
      break;
    default:
      std::cout << "Warning: ImageProcessor::Determine4PointsOrder(): more or less than one key point is special class\n";
      return false;
    }
    for (unsigned int i = 0; i < classes.size(); i++)
    {
      if (classes[i] == special_class)
      {
        special_idx = i;
        break;
      }
    }

    // compute center
    center.x = 0.0f;
    center.y = 0.0f;
    for (auto &point : points)
    {
      center.x += point.x;
      center.y += point.y;
    }
    center.x /= points.size();
    center.y /= points.size();

    // determine order of the points

    // determine order by angles first
    order.clear();
    std::vector<std::pair<float, int>> angles_and_indices{};
    for (unsigned int i = 0; i < points.size(); ++i)
    {
      float angle = std::atan2(points[i].y - center.y, points[i].x - center.x);
      angles_and_indices.emplace_back(std::make_pair(angle, i));
    }

    // counterclockwise
    std::sort(angles_and_indices.begin(), angles_and_indices.end(),
              [](const std::pair<float, int> &a, const std::pair<float, int> &b) -> bool
              { return a.first >= b.first; });

    for (auto &i : angles_and_indices)
      order.emplace_back(i.second);

    // determine final order by the special index
    auto tmp_it = std::find(order.begin(), order.end(), special_idx);
    std::rotate(order.begin(), tmp_it, order.end());

    return true;
  }

  ImageProcessor::ImageProcessor(const std::string &model_path, const cv::Size &model_shape,
                                 const float &model_score_threshold, const float &model_nms_threshold)
  {
    model_.init(model_path, model_shape, model_score_threshold, model_nms_threshold);
  }

  void ImageProcessor::GetCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info)
  {
    for (unsigned int i = 0; i < camera_info->k.size(); ++i)
    {
      camera_matrix_ << camera_info->k[i];
    }

    if (camera_info->d.size() != 5)
    {
      std::cerr << "Error: ImageProcessor::GetCameraInfo(): the distortion matrix size is not 5\n";
      throw std::exception();
    }

    for (unsigned int i = 0; i < camera_info->d.size(); ++i)
    {
      distortion_coefficients_ << camera_info->d[i];
    }
  }

  cv::Vec3f ImageProcessor::getTvec()
  {
    return tvec_;
  }

  cv::Vec3f ImageProcessor::getRvec()
  {
    return rvec_;
  }

  bool ImageProcessor::SolvePnP(int point_num)
  {
    image_points_.clear();

    if (predict_result_.size() != 4)
      return false;

    std::vector<cv::Point2f> four_image_points{};
    std::vector<std::string> classes{};
    cv::Point2f center{};
    std::vector<int> order{};

    for (auto &detection : predict_result_)
    {
      // 1: outermost point
      four_image_points.emplace_back(detection.keypoints[1]);
      classes.emplace_back(detection.class_name);
    }

    if (!Determine4PointsOrder(four_image_points, classes, center, order))
      return false;

    // apply order to points
    for (auto &i : order)
    {
      for (auto &key_point : predict_result_[i].keypoints)
        image_points_.emplace_back(key_point);
    }

    if (point_num == 12)
    {
      if (!cv::solvePnP(object_points_, image_points_, camera_matrix_, distortion_coefficients_, rvec_, tvec_))
        return false;
    }
    else if (point_num == 4)
    {
      std::vector<cv::Point3f> tmp_object_points{};
      std::vector<cv::Point2f> tmp_img_points{};

      for (unsigned int i = 1; i < object_points_.size(); i += 3)
      {
        tmp_object_points.emplace_back(object_points_[i]);
        tmp_img_points.emplace_back(image_points_[i]);
      }
      if (!cv::solvePnP(tmp_object_points, tmp_img_points, camera_matrix_, distortion_coefficients_, rvec_, tvec_))
        return false;
    }

    return true;
  }

  cv::Point2f ImageProcessor::Object2Picture(const cv::Point3f &point)
  {
    cv::Matx33f rotation_matrix;
    cv::Rodrigues(rvec_, rotation_matrix);

    // cv::Mat identity_matrix = cv::Mat::eye(3, 3, CV_64F);
    // rotation_matrix = identity_matrix;
    // object coordinate system => camera coordinate system
    cv::Vec3f camera_point_3d = rotation_matrix * point + cv::Point3f(tvec_);

    if (camera_point_3d[2] == 0)
      return cv::Point2f{};

    // normalized x and y
    float x = camera_point_3d[0] / camera_point_3d[2];
    float y = camera_point_3d[1] / camera_point_3d[2];
    // r^2
    float r_2 = x * x + y * y;

    const float &k1 = distortion_coefficients_[0];
    const float &k2 = distortion_coefficients_[1];
    const float &k3 = distortion_coefficients_[4];
    const float &p1 = distortion_coefficients_[2];
    const float &p2 = distortion_coefficients_[3];
    // distorted x and y
    float distorted_x = x * (1 + k1 * r_2 + k2 * r_2 * r_2 + k3 * std::pow(r_2, 3)) + 2 * p1 * x * y + p2 * (r_2 + 2 * x * x);
    float distorted_y = y * (1 + k1 * r_2 + k2 * r_2 * r_2 + k3 * std::pow(r_2, 3)) + 2 * p2 * x * y + p1 * (r_2 + 2 * y * y);
    cv::Point3f distorted_point_3d{std::move(distorted_x), std::move(distorted_y), 1};
    // x and y in the picture
    cv::Point3f tmp = camera_matrix_ * distorted_point_3d;

    return cv::Point2f{std::move(tmp.x), std::move(tmp.y)};
  }

  void ImageProcessor::AddPredictResult(cv::Mat img, bool add_boxes, bool add_keypoints)
  {
    if (add_boxes)
    {
      for (auto &detection : predict_result_)
      {
        cv::rectangle(img, detection.box, cv::Scalar(255, 255, 255), 2);
        std::string classString = detection.class_name + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(detection.box.x, detection.box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(img, textBox, cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img, classString, cv::Point(detection.box.x + 5, detection.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
      }
    }
    if (add_keypoints)
    {
      for (auto &detection : predict_result_)
      {
        for (unsigned int i = 0; i < detection.keypoints.size(); ++i)
        {
          cv::circle(img, detection.keypoints[i], 3, cv::Scalar(255, 0, 255), 2);
          // cv::putText(frame, std::to_string(i), detection.keypoints[i], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2);
        }
      }
    }
  }

  void ImageProcessor::AddImagePoints(cv::Mat img, bool add_order, bool add_points)
  {
    if (add_order)
    {
      for (unsigned int i = 0; i < image_points_.size(); ++i)
      {
        cv::putText(img, std::to_string(i), image_points_[i], cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 255), 2);
      }
    }
    if (add_points)
    {
      for (auto &point : image_points_)
      {
        cv::circle(img, point, 5, cv::Scalar(255, 0, 255), 3);
      }
    }
  }

} // namespace image_process

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(image_process::ImageProcessNode)
