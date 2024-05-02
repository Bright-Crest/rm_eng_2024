// Yolov8 GPU inference API
// Reference: https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP
// modified

#pragma once
#include "engine.h"
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

// Utility method for checking if a file exists on disk
inline bool doesFileExist(const std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

namespace yolov8_gpu
{
    struct Object
    {
        // The object class.
        int label{};
        // The detection's confidence probability.
        float probability{};
        // The object bounding box rectangle.
        cv::Rect_<float> rect;
        // Semantic segmentation mask
        cv::Mat boxMask;
        // Pose estimation key points
        std::vector<float> kps{};
    };

    // Config the behavior of the GpuInference detector.
    // Can pass these arguments as command line parameters.
    struct YoloV8Config
    {
        // The precision to be used for inference
        Precision precision = Precision::FP16;
        // Calibration data directory. Must be specified when using INT8 precision.
        std::string calibrationDataDirectory;
        // Probability threshold used to filter detected objects
        float probabilityThreshold = 0.25f;
        // Non-maximum suppression threshold
        float nmsThreshold = 0.65f;
        // Max number of detected objects to return
        int topK = 100;
        // Segmentation config options
        int segChannels = 32;
        int segH = 160;
        int segW = 160;
        float segmentationThreshold = 0.5f;
        // Pose estimation options
        int numKPS = 17;
        float kpsThreshold = 0.5f;
        // Class thresholds
        std::vector<std::string> classNames = {"0", "1"};
    };

    class GpuInference
    {
    public:
        GpuInference() = default;
        // Builds the onnx model into a TensorRT engine, and loads the engine into memory
        GpuInference(const std::string &onnxModelPath, const YoloV8Config &config);
        void init(const std::string &onnxModelPath, const YoloV8Config &config);

        // Detect the objects in the image
        std::vector<Object> detectObjects(const cv::Mat &inputImageBGR);
        std::vector<Object> detectObjects(const cv::cuda::GpuMat &inputImageBGR);

        // Draw the object bounding boxes and labels on the image
        void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale = 2);

        // Used as API (only forward the model on GPU, no postprocess)
        void forward(std::vector<std::vector<std::vector<float>>> &outputs, const cv::Mat &inputImage);
        void forward(std::vector<std::vector<std::vector<float>>> &outputs, const cv::cuda::GpuMat &inputImage);

    private:
        // Preprocess the input
        std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg);

        // Postprocess the output
        std::vector<Object> postprocessDetect(std::vector<float> &featureVector);

        // Postprocess the output for pose model
        std::vector<Object> postprocessPose(std::vector<float> &featureVector);

        // Postprocess the output for segmentation model
        std::vector<Object> postProcessSegmentation(std::vector<std::vector<float>> &featureVectors);

        std::unique_ptr<Engine<float>> m_trtEngine = nullptr;

        // Used for image preprocessing
        // YoloV8 model expects values between [0.f, 1.f] so we use the following params
        const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
        const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
        const bool NORMALIZE = true;

        float m_ratio = 1;
        float m_imgWidth = 0;
        float m_imgHeight = 0;

        // Filter thresholds
        float PROBABILITY_THRESHOLD;
        float NMS_THRESHOLD;
        int TOP_K;

        // Segmentation constants
        int SEG_CHANNELS;
        int SEG_H;
        int SEG_W;
        float SEGMENTATION_THRESHOLD;

        // Object classes as strings
        std::vector<std::string> CLASS_NAMES;

        // Pose estimation constant
        int NUM_KPS;
        float KPS_THRESHOLD;

        // Color list for drawing objects
        const std::vector<std::vector<float>> COLOR_LIST = {{1, 1, 1},
                                                            {0.098, 0.325, 0.850},
                                                            {0.125, 0.694, 0.929},
                                                            {0.556, 0.184, 0.494},
                                                            {0.188, 0.674, 0.466},
                                                            {0.933, 0.745, 0.301},
                                                            {0.184, 0.078, 0.635},
                                                            {0.300, 0.300, 0.300},
                                                            {0.600, 0.600, 0.600},
                                                            {0.000, 0.000, 1.000},
                                                            {0.000, 0.500, 1.000},
                                                            {0.000, 0.749, 0.749},
                                                            {0.000, 1.000, 0.000},
                                                            {1.000, 0.000, 0.000},
                                                            {1.000, 0.000, 0.667},
                                                            {0.000, 0.333, 0.333},
                                                            {0.000, 0.667, 0.333},
                                                            {0.000, 1.000, 0.333},
                                                            {0.000, 0.333, 0.667},
                                                            {0.000, 0.667, 0.667},
                                                            {0.000, 1.000, 0.667},
                                                            {0.000, 0.333, 1.000},
                                                            {0.000, 0.667, 1.000},
                                                            {0.000, 1.000, 1.000},
                                                            {0.500, 0.333, 0.000},
                                                            {0.500, 0.667, 0.000},
                                                            {0.500, 1.000, 0.000},
                                                            {0.500, 0.000, 0.333},
                                                            {0.500, 0.333, 0.333},
                                                            {0.500, 0.667, 0.333},
                                                            {0.500, 1.000, 0.333},
                                                            {0.500, 0.000, 0.667},
                                                            {0.500, 0.333, 0.667},
                                                            {0.500, 0.667, 0.667},
                                                            {0.500, 1.000, 0.667},
                                                            {0.500, 0.000, 1.000},
                                                            {0.500, 0.333, 1.000},
                                                            {0.500, 0.667, 1.000},
                                                            {0.500, 1.000, 1.000},
                                                            {1.000, 0.333, 0.000},
                                                            {1.000, 0.667, 0.000},
                                                            {1.000, 1.000, 0.000},
                                                            {1.000, 0.000, 0.333},
                                                            {1.000, 0.333, 0.333},
                                                            {1.000, 0.667, 0.333},
                                                            {1.000, 1.000, 0.333},
                                                            {1.000, 0.000, 0.667},
                                                            {1.000, 0.333, 0.667},
                                                            {1.000, 0.667, 0.667},
                                                            {1.000, 1.000, 0.667},
                                                            {1.000, 0.000, 1.000},
                                                            {1.000, 0.333, 1.000},
                                                            {1.000, 0.667, 1.000},
                                                            {0.000, 0.000, 0.333},
                                                            {0.000, 0.000, 0.500},
                                                            {0.000, 0.000, 0.667},
                                                            {0.000, 0.000, 0.833},
                                                            {0.000, 0.000, 1.000},
                                                            {0.000, 0.167, 0.000},
                                                            {0.000, 0.333, 0.000},
                                                            {0.000, 0.500, 0.000},
                                                            {0.000, 0.667, 0.000},
                                                            {0.000, 0.833, 0.000},
                                                            {0.000, 1.000, 0.000},
                                                            {0.167, 0.000, 0.000},
                                                            {0.333, 0.000, 0.000},
                                                            {0.500, 0.000, 0.000},
                                                            {0.667, 0.000, 0.000},
                                                            {0.833, 0.000, 0.000},
                                                            {1.000, 0.000, 0.000},
                                                            {0.000, 0.000, 0.000},
                                                            {0.143, 0.143, 0.143},
                                                            {0.286, 0.286, 0.286},
                                                            {0.429, 0.429, 0.429},
                                                            {0.571, 0.571, 0.571},
                                                            {0.714, 0.714, 0.714},
                                                            {0.857, 0.857, 0.857},
                                                            {0.741, 0.447, 0.000},
                                                            {0.741, 0.717, 0.314},
                                                            {0.000, 0.500, 0.500}};

        const std::vector<std::vector<unsigned int>> KPS_COLORS = {
            {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

        const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

        const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
            {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}};
    };
} // namespace yolov8_gpu
