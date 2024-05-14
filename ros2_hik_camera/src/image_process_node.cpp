#include "image_process.hpp"

namespace image_process
{
    ImageProcessNode::ImageProcessNode(const rclcpp::NodeOptions &options) : Node("image_process", options)
    {
        bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
        is_debug_ = this->declare_parameter("is_debug", true);
        is_gpu_ = this->declare_parameter("is_gpu", true);

        auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;

        is_serial_used_ = this->declare_parameter("is_serial_used", false);
        if (is_serial_used_)
        {
            transport_serial_.setPort("/dev/ttyUSB0");
            transport_serial_.setBaudrate(9600);
            serial::Timeout _time = serial::Timeout::simpleTimeout(2000);
            transport_serial_.setTimeout(_time);
            transport_serial_.setStopbits(serial::stopbits_one);

            transport_serial_.open();
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

        yolov8::YoloV8Config config{};
        config.is_gpu = is_gpu_;
        config.model_input_shape = MODEL_INPUT_SHAPE;
        config.model_score_threshold = MODEL_SCORE_THRESHOLD;
        config.model_nms_threshold = MODEL_NMS_THRESHOLD;
        config.is_letterbox_for_square = true;
        config.classes.push_back(ExchangeInfo::Special_Corner_Tag);
        config.classes.push_back(ExchangeInfo::Normal_Corner_Tag);
        config.point_num = POINT_NUM;
        // TODO: specify the directory to generate or find the engine file
        config.engineDirectory = this->declare_parameter("engine_dir", package_share_directory);
#ifdef ENABLE_GPU
        config.precision = PRECISION;
#endif 
        img_processor_.init(PARALLEL_THRESHOLD, EXPECTED_OBJECT_NUM, SPECIAL_CORNER_NUM);
        img_processor_.model_.init(model_path, config);

        image_sub_ = image_transport::create_camera_subscription(this, "image_raw",
                                                                 std::bind(&ImageProcessNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2),
                                                                 "raw", qos);

        processed_image_pub_ = image_transport::create_publisher(this, "processed_image", qos);

        process_thread_ = std::thread{[this]() -> void
                                      {
                                          uint8_t send_data_buffer[17];

                                          while (rclcpp::ok())
                                          {
                                              if (frame_queue_.empty())
                                                  continue;
                                              std::lock_guard<std::mutex> lock(g_mutex);
                                              // only use the newest frame
                                              cv_bridge::CvImageConstPtr frame = std::move(frame_queue_.back());
                                              // Warning : do not unlock mutex earlier, because the speed of in-queue is faster
                                              // largely than out-queue operation.
                                              // RCLCPP_INFO(this->get_logger(), "%d!", frame_queue_.size());

                                              img_processor_.ModelPredict(frame->image);
                                              img_processor_.AddPredictResult(frame->image, is_debug_, is_debug_, is_debug_);
                                              if (img_processor_.SolvePnP())
                                              {
                                                  img_processor_.AddSolvePnPResult(frame->image);
                                                  img_processor_.AddImagePoints(frame->image, is_debug_, is_debug_);
                                                  if (is_serial_used_)
                                                  {
                                                      // READY FOR TEST
                                                      // sendSerialData(img_processor_.getTvec(), img_processor_.getRvec());
                                                      uint8_t pack_offset = 3;
                                                      send_data_buffer[0] = 0x23;
                                                      send_data_buffer[1] = MsgStream::PC2BOARD | MsgType::AUTO_EXCHANGE;
                                                      send_data_buffer[2] = 12;
                                                      for (int i = 0; i < 3; ++i)
                                                      {
                                                          int16_t send_temp = img_processor_.getTvec()[i] * 10.f;
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
                                              }

                                              if (this->is_debug_)
                                                  processed_image_pub_.publish(frame->toImageMsg());
                                              std::queue<cv_bridge::CvImageConstPtr> emptyQueue;
                                              std::swap(frame_queue_, emptyQueue);
                                          }
                                      }};

        RCLCPP_INFO(this->get_logger(), "Starting Image Process!");
    }

    ImageProcessNode::~ImageProcessNode()
    {
        image_sub_.shutdown();
        processed_image_pub_.shutdown();
        if (process_thread_.joinable())
            process_thread_.join();
        RCLCPP_INFO(this->get_logger(), "ImageProcessNode destroyed!");
    }

    void ImageProcessNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg, const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!img_processor_.hasCalibrationCoefficient)
        {
            img_processor_.GetCameraInfo(camera_info);
            img_processor_.hasCalibrationCoefficient = true;
            img_processor_.initFPStick();
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

    void ImageProcessNode::sendSerialData(const cv::Vec3f &tvec, const cv::Vec3f &rvec)
    {
        uint8_t pack_offset = 3;
        uint8_t send_data_buffer[17];
        send_data_buffer[0] = 0x23;
        send_data_buffer[1] = MsgStream::PC2BOARD | MsgType::AUTO_EXCHANGE;
        send_data_buffer[2] = 12;
        for (int i = 0; i < 3; ++i)
        {
            uint16_t send_temp = tvec[i] * 10.f;
            send_data_buffer[2 * i + pack_offset] = send_temp;
            send_data_buffer[2 * i + 1 + pack_offset] = send_temp >> 8;
            send_temp = rvec[i] * 10000.f;
            send_data_buffer[2 * i + 6 + pack_offset] = send_temp;
            send_data_buffer[2 * i + 1 + 6 + pack_offset] = send_temp >> 8;
        }
        uint16_t crc_result = CRC16_Calc(send_data_buffer, 15);
        send_data_buffer[15] = crc_result;
        send_data_buffer[16] = crc_result >> 8;
        transport_serial_.write(send_data_buffer, sizeof(send_data_buffer));
    }
} // namespace image_process

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(image_process::ImageProcessNode)
