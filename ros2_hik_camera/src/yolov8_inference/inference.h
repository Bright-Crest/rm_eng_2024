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

// a [x,y,w,h] box takes 4 dimensions
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

    class Inference
    {
    public:
        // do not worry about the number of key points because it is derived automatically
        // cv::Size: (width, height)
        Inference() = default;
        Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640},
                  const float &modelScoreThreshold = 0.45f, const float &modelNMSThreshold = 0.50f, const std::string &classesTxtFile = "");
        void init(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640},
                  const float &modelScoreThreshold = 0.45f, const float &modelNMSThreshold = 0.50f, const std::string &classesTxtFile = "");
        // input: image to be inference;
        // detections: output results
        void runInference(const cv::Mat &input, std::vector<Detection> &detections);

    private:
        void loadClassesFromFile();
        void loadOnnxNetwork();
        cv::Mat formatToSquare(const cv::Mat &source);

        std::string model_path_{};
        std::string classes_path_{};

        std::vector<std::string> classes_{};

        cv::Size2f model_shape_{};

        float model_score_threshold_{0.45f};
        // nms: non_max_suppression
        float model_nms_threshold_{0.50f};

        bool is_letterbox_for_square_{true};

        cv::dnn::Net net_;
    };
}

#endif // INFERENCE_H
