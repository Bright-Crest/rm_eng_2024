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

    cv::Matx33f camera_matrix_{2.8782842692538566e+02, 0., 2.8926852144861482e+02, 0.,
                               2.8780772641312882e+02, 2.4934996600204786e+02, 0., 0., 1.};
    cv::Vec<float, 5> distortion_matrix_{-6.8732455025061909e-02, 1.7584447291315711e-01,
                                         1.0621261625698190e-03, -2.1403059368057149e-03,
                                         -1.3665333157303680e-01};

    // sovlePnP result
    cv::Vec3f rvec_;
    // sovlePnP result
    cv::Vec3f tvec_;

    /// @brief called in SolvePnP(); determine the order of the four target points in the picture
    /// @param points input and output; 4 key points generated from the model
    /// @param classes the classes of the 4 key points
    /// @param center output; the central point of the 4 key points
    /// @return is success
    bool Determine4PointsOrder(std::vector<cv::Point2f> &points, const std::vector<std::string> &classes, cv::Point2f &center);

  public:
    // for object_points_
    static const float length_;
    yolov8::Inference model_;

    ImageProcessor() = default;
    ImageProcessor(const std::string &model_path, const cv::Size &model_shape = {640, 640},
                   const float &model_score_threshold = 0.45f, const float &model_nms_threshold = 0.50f);
    ~ImageProcessor() = default;

    void GetCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info);
    inline void ModelPredict(cv::Mat image) { model_.runInference(image, predict_result_); }
    // call after calling ModelPrdeict()
    bool SolvePnP();
    // call after calling SolvePnP()
    inline std::pair<cv::Vec3f, cv::Vec3f> OutputRvecTvec() { return std::make_pair(rvec_, tvec_); }

    /// @brief bring a 3d point from the object coordinate system to the picture(2d point)
    cv::Point2f Object2Picture(const cv::Point3f &point);
    /// @brief for testing yolov8 model; draw boxes and keypoints
    void AddPredictResult(cv::Mat img, bool add_boxes = true, bool add_keypoints = true);
    /// @brief for testing SolvePnP(), call only when SolvePnP() returns true; draw image_points_ and their order
    void AddImagePoints(cv::Mat img, bool add_order = true, bool add_points = true);
  };

  class ImageProcessNode : public rclcpp::Node
  {
  public:
    explicit ImageProcessNode(const rclcpp::NodeOptions &options) : Node("image_process", options)
    {
      bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
      auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;

      // TODO use "model_path" as parameter in cmd line
      const std::string model_path = "/home/rmv/rm_eng_ws/src/best.onnx";
      img_processor_.model_.init(model_path, cv::Size(640, 640));

      image_sub_ = image_transport::create_camera_subscription(this, "image_raw",
                                                               std::bind(&ImageProcessNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2),
                                                               "raw", qos);

      processed_image_pub_ = image_transport::create_publisher(this, "processed_image", qos);

      process_thread_ = std::thread{[this]() -> void
                                    {
                                      while (rclcpp::ok())
                                      {
                                        if (!frame_queue_.empty())
                                        {
                                          std::lock_guard<std::mutex> lock(g_mutex);

                                          cv_bridge::CvImageConstPtr frame = frame_queue_.front();
                                          frame_queue_.pop();
                                          // TODO ? unlock mutex? delay?

                                          img_processor_.ModelPredict(frame->image);
                                          img_processor_.AddPredictResult(frame->image, true, false);

                                          if (img_processor_.SolvePnP())
                                          {
                                            AddSolvePnPResult(frame->image);
                                            img_processor_.AddImagePoints(frame->image);
                                          }

                                          processed_image_pub_.publish(frame->toImageMsg());

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
    image_transport::Publisher processed_image_pub_;

    std::queue<cv_bridge::CvImageConstPtr> frame_queue_;
    // std::queue<cv_bridge::CvImagePtr> frame_queue_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;

    std::thread process_thread_;
    std::mutex g_mutex;

    ImageProcessor img_processor_;

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg,
                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camara_info)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      try
      {
        // TODO ? toCvShare return a immutable image?
        frame_queue_.push(cv_bridge::toCvShare(msg, "bgr8"));
        // frame_queue_.push(cv_bridge::toCvCopy(msg, "bgr8")->image);
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
        tmp(i) = img_processor_.length_;
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

  const float ImageProcessor::length_ = 126.5;
  const std::vector<cv::Point3f> _tmp{{1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}};
  const std::vector<cv::Point3f> ImageProcessor::object_points_{_tmp[0] * ImageProcessor::length_, _tmp[1] * ImageProcessor::length_, _tmp[2] * ImageProcessor::length_, _tmp[3] * ImageProcessor::length_};

  /// @details check classes and then take the special point(with 2 gaps) as the first and push others clockwise
  bool ImageProcessor::Determine4PointsOrder(std::vector<cv::Point2f> &points, const std::vector<std::string> &classes, cv::Point2f &center)
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
    center.x = 0.0;
    center.y = 0.0;
    for (auto &point : points)
    {
      center.x += point.x;
      center.y += point.y;
    }
    center.x /= points.size();
    center.y /= points.size();

    // determine order of the points
    std::vector<int> order{};
    std::vector<std::pair<float, int>> angles_and_indices{};
    for (unsigned int i = 0; i < points.size(); ++i)
    {
      float angle = std::atan2(points[i].y - center.y, points[i].x - center.x);
      angles_and_indices.emplace_back(std::make_pair(angle, i));
    }

    std::sort(angles_and_indices.begin(), angles_and_indices.end());
    for (auto &i : angles_and_indices)
      order.emplace_back(i.second);

    auto tmp_it = std::find(order.begin(), order.end(), special_idx);
    std::rotate(order.begin(), tmp_it, order.end());

    // apply order to points
    std::vector<cv::Point2f> ordered_points{};
    ordered_points.resize(points.size());
    for (unsigned int i = 0; i < points.size(); ++i)
    {
      ordered_points[i] = std::move(points[order[i]]);
    }
    points = std::move(ordered_points);

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
      distortion_matrix_ << camera_info->d[i];
    }
  }

  bool ImageProcessor::SolvePnP()
  {
    if (predict_result_.size() != 4)
      return false;

    std::vector<std::string> classes{};
    cv::Point2f center{};

    for (auto &detection : predict_result_)
    {
      // for developers: check the index of the innermost point whenever changing a model
      // 1: innermost point
      image_points_.emplace_back(detection.keypoints[1]);
      classes.emplace_back(detection.class_name);
    }

    if (!Determine4PointsOrder(image_points_, classes, center))
      return false;

    if (!cv::solvePnP(object_points_, image_points_, camera_matrix_, distortion_matrix_, rvec_, tvec_))
      return false;

    return true;
  }

  cv::Point2f ImageProcessor::Object2Picture(const cv::Point3f &point)
  {
    cv::Matx33f rotation_matrix;
    cv::Rodrigues(rvec_, rotation_matrix);

    // object coordinate system => camera coordinate system
    cv::Vec3f camera_point_3d = rotation_matrix * point + cv::Point3f(tvec_);

    if (camera_point_3d[2] == 0)
      return cv::Point2f{};

    // normalized x and y
    float x = camera_point_3d[0] / camera_point_3d[2];
    float y = camera_point_3d[1] / camera_point_3d[2];
    // r^2
    float r_2 = x * x + y * y;

    const float &k1 = distortion_matrix_[0];
    const float &k2 = distortion_matrix_[1];
    const float &k3 = distortion_matrix_[4];
    const float &p1 = distortion_matrix_[2];
    const float &p2 = distortion_matrix_[3];
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
