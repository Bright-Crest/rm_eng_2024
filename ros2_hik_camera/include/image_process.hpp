#pragma once
// used for 2 objects and 12 key points whose order is specified
// When the model is changed, the following functions must be carefully checked and changed:

// system
#include <chrono>

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
#include "../include/yolov8_inference/inference.h"
/// Serial
#include "serial/serial.h"
#include "serial/serial_format.h"

#define LEN_A 137.5f
#define LEN_B 87.5f
#define SIDE_LEN_A 100.0f
#define SIDE_LEN_B 45.5f

// For model inference
#define MODEL_INPUT_SHAPE cv::Size(640, 640)
#define MODEL_SCORE_THRESHOLD 0.45f
#define MODEL_NMS_THRESHOLD 0.50f

// only for GPU inference
// Number of points per object
#define POINT_NUM               3
// TODO: specify the directory to generate or find the engine file
#define ENGINE_DIR              "/home/nvidia/rm_eng_2024_ws/build"
// The precision to be used for inference
#define PRECISION               Precision::FP16

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
        // the default parameter from the Infantry MV-CS016-10UC
        cv::Matx33f camera_matrix_{863.060425, 0.000000, 626.518063, 0.000000, 863.348503, 357.765589, 0.000000, 0.000000, 1.000000};
        cv::Vec<float, 5> distortion_coefficients_{-0.082726, 0.069328, 0.000180, -0.002923, 0.000000};

        // for SolvePnP()
        static const std::vector<cv::Point3f> object_points_;
        static const std::vector<cv::Point3f> side_plate_object_points_;
        std::vector<cv::Point2f> image_points_;
        // yolov8 model result
        std::vector<yolov8::Detection> predict_result_;

        // for fps calculation
        double last_system_tick_;
        // sovlePnP result
        cv::Vec3f rvec_;
        cv::Vec3f last_rvec_;
        cv::Vec3f tvec_;
        cv::Vec3f last_tvec_;

        /// @brief called in SolvePnP(); determine the order of the arbitrary target points in the picture
        /// @param whole_points all key points of the front of the exchange station which is generated from the model
        /// @param classes the classes of the key points
        /// @param order output; the expected order of the key points
        /// @return is success
        bool DetermineAbitraryPointsOrder(const std::vector<cv::Point2f> &whole_points, const std::vector<std::string> &classes, std::vector<int> &order);

    public:
        ImageProcessor() = default;
        ImageProcessor(const std::string &model_path, const cv::Size &model_shape = {640, 640},
                       const float &model_score_threshold = 0.45f, const float &model_nms_threshold = 0.50f, bool is_gpu = false);
        ~ImageProcessor() = default;

        /// @brief get camera_matrix and distortion_coefficients from the first frame
        void GetCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info);

        inline void ModelPredict(const cv::Mat &image) { model_.runInference(image, predict_result_); }
        // call after calling ModelPrdeict()
        // modify image_points_
        bool SolvePnP(const cv::Mat &frame);
        // call after calling SolvePnP()
        inline std::pair<cv::Vec3f, cv::Vec3f> OutputRvecTvec() { return std::make_pair(rvec_, tvec_); }

        /// @brief for testing yolov8 model; draw boxes and keypoints
        void AddPredictResult(cv::Mat img, bool add_boxes = true, bool add_keypoints = true, bool add_fps = true);
        /// @brief for testing SolvePnP(), call only when SolvePnP() returns true; draw image_points_ and their order
        void AddImagePoints(cv::Mat img, bool add_order = true, bool add_points = true);
        /// @brief bring a 3d point from the object coordinate system to the picture(2d point)
        cv::Point2f Object2Picture(const cv::Point3f &point);

        /// @brief get tvec_
        cv::Vec3f getTvec();
        /// @brief get rvec_
        cv::Vec3f getRvec();

        void initFPStick();

        // for object_points_
        yolov8::Inference model_;

        // whether the node has already got the camera_matrix
        bool hasCalibrationCoefficient = false;
    };

    class ImageProcessNode : public rclcpp::Node
    {
    public:
        explicit ImageProcessNode(const rclcpp::NodeOptions &options);
        ~ImageProcessNode();

    private:
        /// @brief the Callback function called by the subscriptor
        void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info);
        /// @brief send the 6-D information data through serial
        void sendSerialData(const cv::Vec3f &tvec, const cv::Vec3f &rvec);
        /// @brief for testing SolvePnP(); draw xyz coordinate system
        void AddSolvePnPResult(cv::Mat img);

    private:
        bool is_gpu_;
        bool is_serial_used_;
        bool is_debug_;

        image_transport::CameraSubscriber image_sub_;
        image_transport::Publisher processed_image_pub_;

        std::queue<cv_bridge::CvImageConstPtr> frame_queue_;
        sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;

        std::thread process_thread_;
        std::mutex g_mutex;

        ImageProcessor img_processor_;
        serial::Serial transport_serial_;
    };

} // namespace image_process
