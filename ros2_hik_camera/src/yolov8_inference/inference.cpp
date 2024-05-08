// inference.cpp
// Used for yolov8 pose models
// Currently used for 2 object and 3 points per object

#include "yolov8_inference/inference.h"

yolov8::Inference::Inference(const std::string &onnxModelPath, const YoloV8Config &config)
{
    init(onnxModelPath, config);
}

void yolov8::Inference::init(const std::string &onnxModelPath, const YoloV8Config &config)
{
    model_path_ = onnxModelPath;
    is_gpu_ = config.is_gpu;
    model_shape_ = config.model_input_shape;
    model_score_threshold_ = config.model_score_threshold;
    model_nms_threshold_ = config.model_nms_threshold;
    is_letterbox_for_square_ = config.is_letterbox_for_square;

    // determine classes
    if (!config.classes_txt_file.empty())
        loadClassesFromFile(config.classes_txt_file);
    else if (!config.classes.empty())
        classes_ = config.classes;
    else if (config.class_num > 0)
        for (int i = 0; i < config.class_num; i++)
            classes_.push_back(std::to_string(i));
    else
        throw("Error: classes of the model must be provided");

    if (!is_gpu_)
    {
        loadOnnxNetwork();
    }
    else
    {
        point_num_ = config.point_num;
#ifdef ENABLE_GPU
        yolov8_gpu::YoloV8GpuConfig gpu_config{};
        gpu_config.precision = config.precision;
        gpu_config.calibrationDataDirectory = config.calibrationDataDirectory;
        gpu_config.probabilityThreshold = config.model_score_threshold;
        gpu_config.engineDirectory = config.engineDirectory;

        gpu_inference_.init(onnxModelPath, gpu_config);
#endif
    }
}

void yolov8::Inference::runInference(const cv::Mat &input, std::vector<Detection> &detections)
{
    image_shape_ = input.size();

    if (!is_gpu_)
        runCpuInference(input, detections);
    else
        runGpuInference(input, detections);
}

void yolov8::Inference::runCpuInference(const cv::Mat &input, std::vector<Detection> &detections)
{
    cv::Mat model_input;
    std::vector<cv::Mat> model_outputs;

    preprocess(model_input, input);
    forward(model_outputs, model_input);
    postprocess(detections, model_outputs);
}

void yolov8::Inference::runGpuInference(const cv::Mat &input, std::vector<Detection> &detections)
{
    cv::Mat model_input;
    std::vector<std::vector<std::vector<float>>> tmp_model_outputs;

    preprocess(model_input, input);
#ifdef ENABLE_GPU
    gpu_inference_.forward(tmp_model_outputs, model_input);
#endif
    if (tmp_model_outputs.size() != 1 || tmp_model_outputs[0].size() != 1)
        throw("Error: the size of model outputs of gpu inference is wrong.");

    // Hard-coded: vector (1, 1, 100800) to cv::Mat (1, 12, 8400)
    static const int channels = 1;
    static const int dimensions = 3;
    int new_size[dimensions];
    new_size[0] = 1;
    // go to function postprocess for explanation
    new_size[1] = BOX_NUM + classes_.size() + point_num_ * 2;
    if (tmp_model_outputs[0][0].size() % new_size[0] != 0 || tmp_model_outputs[0][0].size() % new_size[1] != 0)
        throw("Error: the capacity of model outputs is not divisible");
    new_size[2] = tmp_model_outputs[0][0].size() / new_size[0] / new_size[1];

    cv::Mat tmp_mat{};
    std::vector<cv::Mat> model_outputs{};

    for (auto &v : tmp_model_outputs[0])
    {
        tmp_mat.push_back<float>(v);
    }
    model_outputs.push_back(tmp_mat.reshape(channels, dimensions, new_size));

    postprocess(detections, model_outputs);
}

void yolov8::Inference::preprocess(cv::Mat &model_input, const cv::Mat &input)
{
    cv::Mat tmp_mat = input;
    if (is_letterbox_for_square_ && model_shape_.width == model_shape_.height)
        tmp_mat = formatToSquare(input);
    image_shape_ = tmp_mat.size();

    // TODO: why does this fail for GPU inference?
    // cv::dnn::blobFromImage(tmp_mat, model_input, 1.0 / 255.0, model_shape_, cv::Scalar(), true, false);
    model_input = tmp_mat;
}

void yolov8::Inference::forward(std::vector<cv::Mat> &model_outputs, const cv::Mat &model_input)
{
    cv::Mat blob = model_input;
    cv::dnn::blobFromImage(model_input, blob, 1.0 / 255.0, model_shape_, cv::Scalar(), true, false);
    
    net_.setInput(blob);
    net_.forward(model_outputs, net_.getUnconnectedOutLayersNames());
}

void yolov8::Inference::postprocess(std::vector<Detection> &detections, std::vector<cv::Mat> &model_outputs)
{
    // yolov8 has an output of shape (batchSize, dimensions, rows)
    // the batch size is usually one; rows do not matter
    // dimensions = BOX_NUM + classes_.size() + keypoints.size() * 2
    // e.g. in our case, dimensions = 4 + 2 + 3 * 2 = 12
    // (1, 12, 8400)
    int dimensions = model_outputs[0].size[1];
    int rows = model_outputs[0].size[2];
    // 1: channel; dimensions: rows
    model_outputs[0] = model_outputs[0].reshape(1, dimensions); // (18, 6300)
    cv::transpose(model_outputs[0], model_outputs[0]);          // (6300, 18)

    float *data = (float *)model_outputs[0].data;

    // used to convert the model shape to the original image shape
    float x_factor = image_shape_.width / model_shape_.width;
    float y_factor = image_shape_.height / model_shape_.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    // record the original index(row) for getting key points below
    std::vector<int> indices;

    // loop: get the max confidence, its class_id, its box
    for (int i = 0; i < rows; ++i)
    {
        float *classes_scores = data + BOX_NUM;

        // 1: dimension;
        cv::Mat scores(1, classes_.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        // get min and max values and locations
        // class_id.y is useless because scores.rows is 1
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > model_score_threshold_)
        {
            indices.emplace_back(i);
            confidences.emplace_back(maxClassScore);
            class_ids.emplace_back(class_id.x);

            // (x, y) is the center of the object as well as the box
            float &x = data[0];
            float &y = data[1];
            float &w = data[2];
            float &h = data[3];

            int left = std::round((x - 0.5 * w) * x_factor);
            int top = std::round((y - 0.5 * h) * y_factor);
            int width = std::round(w * x_factor);
            int height = std::round(h * y_factor);

            boxes.emplace_back(cv::Rect(left, top, width, height));
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, model_score_threshold_, model_nms_threshold_, nms_result);

    // for each Inference compute once
    static const int kKeyPointsNum = (dimensions - BOX_NUM - classes_.size()) / 2;

    // Note: must clear
    detections.clear();
    for (unsigned int i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        int original_idx = indices[idx];

        Detection result;
        result.confidence = confidences[idx];

        result.class_name = classes_[class_ids[idx]];
        result.box = std::move(boxes[idx]);

        float *data_ptr = (float *)model_outputs[0].data + original_idx * dimensions;
        float *kp_ptr = data_ptr + BOX_NUM + classes_.size();
        for (int i = 0; i < kKeyPointsNum; ++i)
        {
            float x = kp_ptr[i * 2] * x_factor;
            float y = kp_ptr[i * 2 + 1] * y_factor;
            result.keypoints.emplace_back(cv::Point2f{x, y});
        }

        detections.emplace_back(std::move(result));
    }
}

void yolov8::Inference::loadClassesFromFile(const std::string &classes_path)
{
    std::ifstream inputFile(classes_path);
    if (inputFile.is_open())
    {
        std::string classLine;
        while (std::getline(inputFile, classLine))
            classes_.push_back(classLine);
        inputFile.close();
    }
}

void yolov8::Inference::loadOnnxNetwork()
{
    net_ = cv::dnn::readNetFromONNX(model_path_);
    // run on CPU
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

cv::Mat yolov8::Inference::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);

    // cv::Mat operater(): Extract a submatrix specified as a rectangle
    // copy source to the left-top of the result
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
