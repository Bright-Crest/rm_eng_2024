// inference.h
// used for 2 object and 12 points

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
#include "gpu_inference.h"

// A [x,y,w,h] box takes 4 dimensions
#define BOX_NUM 4

// TODO: Test these two functions
// Reference: https://blog.csdn.net/guyuealian/article/details/80253066
template<typename _Tp>
cv::Mat Vector2Mat(vector<_Tp> v, int channels, int rows)
{
	cv::Mat mat = cv::Mat(v);                          //将vector变成单列的mat; shallow copy
	cv::Mat dest = mat.reshape(channels, rows).clone();//PS：必须clone()一份，否则返回出错
	return dest;
}

template<typename _Tp>
cv::Mat Vector2Mat(vector<vector<_Tp>> v2d, int channels, int rows)
{
    cv::Mat mat;
    for (auto vp = vp2d.begin(); vp != vp2d.end(); vp++)
    {
        if (vp == v2d.begin())
        {
            mat = Vector2Mat(*vp, 1, 1);
        }
        else
        {
            cv::vconcat(mat, Vector2Mat(*vp, 1, 1));
        }
    }
}

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
                  const float &modelScoreThreshold = 0.45f, const float &modelNMSThreshold = 0.50f, const std::string &classesTxtFile = "", bool is_gpu = false);
        void init(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640},
                  const float &modelScoreThreshold = 0.45f, const float &modelNMSThreshold = 0.50f, const std::string &classesTxtFile = "", bool is_gpu = false);
        // input: image to be inference;
        // detections: output results
        void runInference(const cv::Mat &input, std::vector<Detection> &detections);

    private:
        void runCpuInference(const cv::Mat &input, std::vector<Detection> &detections);
        void runGpuInference(const cv::Mat &input, std::vector<Detection> &detections);

        // for CPU
        void preprocess(cv::Mat& model_input, const cv::Mat &input);
        void forward(std::vector<cv::Mat>& model_outputs, const cv::Mat& model_input);
        void postprocess(std::vector<Detection>& detections, std::vector<cv::Mat>& model_outputs);

        void loadClassesFromFile();
        // load into CPU
        void loadOnnxNetwork();
        cv::Mat formatToSquare(const cv::Mat &source);

        bool is_gpu_{};
        yolov8_gpu::GpuInference gpu_inference_{};

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
