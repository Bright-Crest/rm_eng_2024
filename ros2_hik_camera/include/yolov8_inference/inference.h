// inference.h
// Used for yolov8 pose models
// Currently used for 2 object and 3 points per object

#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// TensorRT GPU Inference
#ifdef ENABLE_GPU
#include "engine.h"
#include "gpu_inference.h"
#endif

// A [x,y,w,h] box takes 4 dimensions (should never be changed)
#define BOX_NUM 4

namespace yolov8
{
    struct Detection
    {
        std::string class_name{};
        float confidence{0.0f};
        cv::Rect box{};
        std::vector<cv::Point2f> keypoints{};

        Detection() = default;
        Detection(const Detection &) = default;
        Detection(Detection &&source)
        {
            class_name = std::move(source.class_name);
            confidence = std::move(source.confidence);
            box = std::move(source.box);
            keypoints = std::move(source.keypoints);
        }

        Detection &operator=(const Detection &) = default;
        Detection &operator=(Detection &&source)
        {
            if (&source == this)
                return *this;

            class_name = std::move(source.class_name);
            confidence = std::move(source.confidence);
            box = std::move(source.box);
            keypoints = std::move(source.keypoints);

            return *this;
        }
    };

    struct YoloV8Config
    {
        bool is_gpu = false;

        // For both CPU and GPU inference
        // Must be provided
        cv::Size model_input_shape;
        float model_score_threshold = 0.45f;
        float model_nms_threshold = 0.50f;
        bool is_letterbox_for_square = true;
        // The following three variables are used to determine the class names.
        // Provide at least one variable. They are checked by order.
        std::string classes_txt_file;
        std::vector<std::string> classes;
        // Expand to {"0", "1", ...}
        int class_num = 0;

        // Only for GPU inference; used only when is_gpu is true
        // Must be provided.
        // Number of points per object
        int point_num;
        // Recommended to provide if using ROS, or maybe the engine file cannot
        // be found by the program
        std::string engineDirectory = "";
        // TODO: what about CPU precision
#ifdef ENABLE_GPU
        // The precision to be used for inference
        Precision precision = Precision::FP16;
#endif
        // Calibration data directory. Must be specified only when using INT8 precision.
        std::string calibrationDataDirectory;
    };

    class Inference
    {
    public:
        // cv::Size: (width, height)
        Inference() = default;
        Inference(const std::string &onnxModelPath, const YoloV8Config &config);
        void init(const std::string &onnxModelPath, const YoloV8Config &config);
        // input: image to be inference;
        // detections: output results
        void runInference(const cv::Mat &input, std::vector<Detection> &detections);

    private:
        void runCpuInference(const cv::Mat &input, std::vector<Detection> &detections);
        void runGpuInference(const cv::Mat &input, std::vector<Detection> &detections);

        // For both CPU and GPU
        void preprocess(cv::Mat &model_input, const cv::Mat &input);
        // For CPU
        void forward(std::vector<cv::Mat> &model_outputs, const cv::Mat &model_input);
        // For both CPU and GPU.
        // Do not worry about the number of key points because it is derived automatically
        void postprocess(std::vector<Detection> &detections, std::vector<cv::Mat> &model_outputs);

        // load into CPU
        void loadOnnxNetwork();
        cv::Mat formatToSquare(const cv::Mat &source);
        void loadClassesFromFile(const std::string &classes_path);

        bool is_gpu_{};
        std::string model_path_{};
        std::vector<std::string> classes_{};
        cv::Size2f model_shape_{};
        cv::Size2f image_shape_{};
        bool is_letterbox_for_square_{true};

        float model_score_threshold_{0.45f};
        // nms: non_max_suppression
        float model_nms_threshold_{0.50f};


        cv::dnn::Net net_;

        // Only for GPU
#ifdef ENABLE_GPU
        yolov8_gpu::GpuInference gpu_inference_{};
#endif
        // Number of points per object
        int point_num_;
    };
}

#endif // INFERENCE_H
