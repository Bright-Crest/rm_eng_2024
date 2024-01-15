// calibration.cpp (ABANDONDED)
// c++ common
#include <queue>

// ros2
// ros2 rclcpp
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
// ros2 transport image
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
// ros2 Image from and to cv::Mat
#include <cv_bridge/cv_bridge.h>
// ros2 imu
#include <sensor_msgs/msg/imu.hpp>

// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

// Eigen
#include <Eigen/Dense>

namespace calibration
{
  class CalibrationNode : public rclcpp::Node
  {
  public:
    explicit CalibrationNode(const rclcpp::NodeOptions &options) : Node("calibration", options)
    {
      bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
      auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;

      imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("imu", 10, std::bind(&CalibrationNode::ImuCallback, this, std::placeholders::_1));

      image_sub_ = image_transport::create_camera_subscription(this, "image_raw",
                                                               std::bind(&CalibrationNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2),
                                                               "raw", qos);

      calibration_image_pub_ = image_transport::create_publisher(this, "calibration_image", qos);

      calibration_thread_ = std::thread{[this]
                                        {
                                          while (rclcpp::ok())
                                          {
                                            // TODO confirm frame_vector_ and imu_vector_ is finished getting new data
                                            std::this_thread::sleep_for(std::chrono::seconds(2));
                                          }

                                          std::lock_guard<std::mutex> frame_lock(frame_mutex_);
                                          std::lock_guard<std::mutex> imu_lock(imu_mutex_);

                                          RunCalibration(frame_vector_);
                                        }};

      RCLCPP_INFO(this->get_logger(), "Starting CalibrationNode");
    }

    ~CalibrationNode()
    {
      image_sub_.shutdown();

      if (calibration_thread_.joinable())
        calibration_thread_.join();

      RCLCPP_INFO(this->get_logger(), "CalibrationNode destroyed!");
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    image_transport::CameraSubscriber image_sub_;
    image_transport::Publisher calibration_image_pub_;

    std::vector<cv_bridge::CvImageConstPtr> frame_vector_;
    std::vector<sensor_msgs::msg::Imu::SharedPtr> imu_vector_;

    bool use_imu_ = false;

    std::thread calibration_thread_;
    std::mutex frame_mutex_;
    std::mutex imu_mutex_;

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg,
                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info)
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      try
      {
        frame_vector_.emplace_back(cv_bridge::toCvShare(msg, "bgr8"));
      }
      catch (const cv_bridge::Exception &e)
      {
        auto logger = rclcpp::get_logger("Calibration Error");
        RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
      }
    }

    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr imu)
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      imu_vector_.emplace_back(imu);
    }

    void RunCalibration(const std::vector<cv_bridge::CvImageConstPtr> &images, bool is_saving_chessboard_corners = false, std::vector<Eigen::Quaternionf> q_vector = std::vector<Eigen::Quaternionf>{})
    {
      if (use_imu_)
      {
        if (images.size() != q_vector.size())
          return;
      }
      else
      {
        q_vector.resize(images.size());
      }

      // 图像size
      cv::Size size;
      // 棋盘宽高 (the number of blocks)
      const int w = 11;
      const int h = 8;
      // 棋盘间隔距离(cm)
      const double d = 45;
      // 世界坐标点
      std::vector<cv::Point3f> obj(w * h);
      for (int j = 0; j < h; j++)
      {
        for (int i = 0; i < w; i++)
        {
          obj[j * w + i] = cv::Point3f(d * i, d * j, 0);
        }
      }

      // 世界坐标点序列
      std::vector<std::vector<cv::Point3f>> obj_seq;
      // 棋盘角点序列
      std::vector<std::vector<cv::Point2f>> corners_seq;
      // 图像序列
      std::vector<cv::Mat> image_seq;
      // 陀螺仪姿态序列
      std::vector<Eigen::Matrix3f> imu_seq;

      for (int i = 0; i < images.size(); ++i)
      {
        cv::Mat img = images[i]->image;
        Eigen::Quaternionf q = q_vector[i];
        size = {img.cols, img.rows};

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners_curr;
        // 棋盘角点检测
        bool found = cv::findChessboardCorners(gray, {w, h}, corners_curr);
        if (!found || corners_curr.size() != w * h)
        {
          std::cout << "Fail to find chess board corners" << std::endl;
          continue;
        }

        // 指定亚像素计算迭代标注
        cv::TermCriteria criteria = cv::TermCriteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 40, 0.01);
        // 亚像素计算
        cv::cornerSubPix(gray, corners_curr, {5, 5}, {-1, -1}, criteria);

        if (is_saving_chessboard_corners)
        {
          cv::drawChessboardCorners(img, {w, h}, corners_curr, found);
          cv::imwrite("./images/" + std::to_string(i) + ".jpg", img);
        }

        // 保存数据
        corners_seq.emplace_back(corners_curr);
        image_seq.emplace_back(gray);
        imu_seq.emplace_back(q.matrix());
        obj_seq.emplace_back(obj);
      }

      std::cout << obj_seq.size() << " images collected" << std::endl;

      // 进行标定计算
      // 先计算相机内外参
      cv::Mat mtx, coff;
      std::vector<cv::Mat> rvecs, tvecs;
      double err = cv::calibrateCamera(obj_seq, corners_seq, size, mtx, coff, rvecs, tvecs);
      std::cout << "camera project error: " << err << std::endl;

      // TODO imu

      // TODO path and data format(not opencv)
      cv::FileStorage fout("/home/rmv/rm_eng_ws/src/ros2_hik_camera/config/camera_info_new.yaml", cv::FileStorage::WRITE);
      fout << "F" << mtx;
      fout << "C" << coff;
    }
  };
} // namespace calibration
