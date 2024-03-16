#pragma once
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
#include "../src/yolov8_inference/inference.h"
/// Serial
#include "serial/serial.h"
#include "serial/serial_format.h"

#define LEN_A 137.5f
#define LEN_B 87.5f
#define SIDE_LEN_A 100.0f
#define SIDE_LEN_B 45.5f

namespace ExchangeInfo
{
    static const std::string Special_Corner_Tag = "0";
    static const std::string Normal_Corner_Tag = "1";
} // namespace ExchangeInfo

namespace image_process
{
    // imgmsg => cv Mat => yolov8 model => solvePnP
    class ImageProcessor
    {
    private:
        // for SolvePnP()
        static const std::vector<cv::Point3f> object_points_;
        static const std::vector<cv::Point3f> side_plate_object_points_;

        // for SolvePnP() and AddImagePoints()
        std::vector<cv::Point2f>
            image_points_;
        // yolov8 model result
        std::vector<yolov8::Detection> predict_result_;

        // the default parameter from the Infantry MV-CS016-10UC
        cv::Matx33f camera_matrix_{1622.412925, 0.000000, 601.366953, 0.000000, 1619.246860, 414.637285, 0.000000, 0.000000, 1.000000};
        cv::Vec<float, 5> distortion_coefficients_{-0.144865, 0.216839, -0.002233, -0.003314, 0.000000};

        // sovlePnP result
        cv::Vec3f rvec_;
        cv::Vec3f last_rvec_;
        // sovlePnP result
        cv::Vec3f tvec_;
        cv::Vec3f last_tvec_;

        /// @brief called in SolvePnP(); determine the order of the arbitrary target points in the picture
        /// @param whole_points all key points of the front of the exchange station which is generated from the model
        /// @param classes the classes of the key points
        /// @param order output; the expected order of the key points
        /// @return is success
        bool DetermineAbitraryPointsOrder(const std::vector<cv::Point2f> &whole_points, const std::vector<std::string> &classes, std::vector<int> &order);
        bool TraditionalDetect(const std::vector<cv::Point2f> &whole_points, std::vector<int> &order);

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
        bool SolvePnP(const cv::Mat &frame, int point_num);
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
            const std::string model_path = package_share_directory + "/model/n_pose_4boxes_12points.onnx";

            img_processor_.model_.init(model_path, cv::Size(640, 640));

            image_sub_ = image_transport::create_camera_subscription(this, "image_raw",
                                                                     std::bind(&ImageProcessNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2),
                                                                     "raw", qos);

            processed_image_pub_ = image_transport::create_publisher(this, "processed_image", qos);

            process_thread_ = std::thread{[this]() -> void
                                          {
                                              uint8_t send_data_buffer[17];

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

                                                      if (img_processor_.SolvePnP(frame->image, 12))
                                                      {
                                                          AddSolvePnPResult(frame->image);
                                                          img_processor_.AddImagePoints(frame->image);

                                                          if (is_serial_used_)
                                                          {
                                                              // Debug
                                                              uint8_t pack_offset = 3;
                                                              send_data_buffer[0] = 0x23;
                                                              send_data_buffer[1] = MsgStream::PC2BOARD | MsgType::AUTO_EXCHANGE;
                                                              send_data_buffer[2] = 12;
                                                              for (int i = 0; i < 3; ++i)
                                                              {
                                                                  uint16_t send_temp = img_processor_.getTvec()[i] * 10.f;
                                                                  send_data_buffer[2 * i + pack_offset] = send_temp;
                                                                  send_data_buffer[2 * i + 1 + pack_offset] = send_temp >> 8;
                                                                  send_temp = img_processor_.getRvec()[i] * 10000.f;
                                                                  send_data_buffer[2 * i + 6 + pack_offset] = send_temp;
                                                                  send_data_buffer[2 * i + 1 + 6 + pack_offset] = send_temp >> 8;
                                                              }
                                                              uint16_t crc_result = CRC16_Calc(send_data_buffer, 15);
                                                              send_data_buffer[15] = crc_result;
                                                              send_data_buffer[16] = crc_result >> 8;
                                                              transport_serial_.write(send_data_buffer, sizeof(send_data_buffer));
                                                          }
                                                          /*
                                                          for (int i = 0; i < 12; ++i)
                                                              std::cout << std::hex << static_cast<int>(send_data_buffer[3 + i]) << " ";
                                                          std::cout << " " << std::endl;
                                                           */
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

        /// @brief the Callback function called by the subscriptor
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
    // the right plate
    const std::vector<cv::Point3f> ImageProcessor::side_plate_object_points_{
        {144.0f, SIDE_LEN_A, -SIDE_LEN_A - SIDE_LEN_B}, {144.0f, 0, SIDE_LEN_B}, {144.0f, -SIDE_LEN_A, -SIDE_LEN_A - SIDE_LEN_B}};
} // namespace image_process
