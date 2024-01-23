#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "depthai/depthai.hpp"

#include <ctime>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>

// 定义一个结构体，代表 linux里目录分层符号 ‘/’，再后面std：：find_if 查找符号的函数中会用到
struct pathseperator
{
    pathseperator()
    {
    }

    bool operator()(char ch) const
    {
        return ch == '/';
    }
};

// 检查路径是否已创建
int checkFolderExist(std::string const &name)
{
    struct stat fileStatus;
    if (stat(name.c_str(), &fileStatus) == -1)
        return -1;
    return 0;
}

bool createDirectory(std::string target)
{
    std::string sep = "/";

    // 通过查找 / 的方式将每层路径拆开
    std::vector<std::string> container;
    std::string::const_iterator const end = target.end();
    std::string::const_iterator it = target.begin();
    // char psc = '/';
    pathseperator pathsep = pathseperator();
    while (it != end)
    {
        std::string::const_iterator sep = std::find_if(it, end, pathsep);
        container.push_back(std::string(it, sep));
        it = sep;
        if (it != end)
            ++it;
    }

    // 将拆解的路径重新一层一层的拼接，并调用系统函数mkdir创建
    std::string path;
    for (int i = 1; i < container.size(); i++)
    {
        path += sep;
        path += container[i];
        if (checkFolderExist(path) == 0)
        {
            continue;
        }
        int res = mkdir(path.c_str(), 0777);
        std::cout << "the result of making directory " << path << std::endl;
    }
    return true;
}

static std::atomic<bool> downscaleColor{true};
static constexpr int fps = 30;
// The disparity is computed at this resolution, then upscaled to RGB resolution
static constexpr auto monoRes = dai::MonoCameraProperties::SensorResolution::THE_720_P;

static float rgbWeight = 0.4f;
static float depthWeight = 0.6f;

static void updateBlendWeights(int percentRgb, void *ctx)
{
    rgbWeight = float(percentRgb) / 100.f;
    depthWeight = 1.f - rgbWeight;
}

class OakCaptureNode : public rclcpp::Node
{
public:
    explicit OakCaptureNode(std::string name) : Node(name)
    {
        using namespace std;

        // Create pipeline
        dai::Pipeline pipeline;
        dai::Device device;
        std::vector<std::string> queueNames;

        // Define sources and outputs
        auto camRgb = pipeline.create<dai::node::ColorCamera>();
        auto left = pipeline.create<dai::node::MonoCamera>();
        auto right = pipeline.create<dai::node::MonoCamera>();
        auto stereo = pipeline.create<dai::node::StereoDepth>();

        auto rgbOut = pipeline.create<dai::node::XLinkOut>();
        auto depthOut = pipeline.create<dai::node::XLinkOut>();

        rgbOut->setStreamName("rgb");
        queueNames.push_back("rgb");
        depthOut->setStreamName("depth");
        queueNames.push_back("depth");

        // Properties
        camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
        camRgb->setBoardSocket(dai::CameraBoardSocket::RGB);
        camRgb->setFps(fps);
        if (downscaleColor)
            camRgb->setIspScale(2, 3);
        // For now, RGB needs fixed focus to properly align with depth.
        // This value was used during calibration
        try
        {
            auto calibData = device.readCalibration2();
            auto lensPosition = calibData.getLensPosition(dai::CameraBoardSocket::RGB);
            if (lensPosition)
            {
                camRgb->initialControl.setManualFocus(lensPosition);
            }
        }
        catch (const std::exception &ex)
        {
            std::cout << ex.what() << std::endl;
            return;
        }

        left->setResolution(monoRes);
        left->setBoardSocket(dai::CameraBoardSocket::LEFT);
        left->setFps(fps);
        right->setResolution(monoRes);
        right->setBoardSocket(dai::CameraBoardSocket::RIGHT);
        right->setFps(fps);

        stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
        // LR-check is required for depth alignment
        stereo->setLeftRightCheck(true);
        stereo->setDepthAlign(dai::CameraBoardSocket::RGB);

        // Linking
        camRgb->isp.link(rgbOut->input);
        left->out.link(stereo->left);
        right->out.link(stereo->right);
        stereo->disparity.link(depthOut->input);

        // Connect to device and start pipeline
        device.startPipeline(pipeline);

        // define the RGB and depth image save path
        // default : ~/${workspace}/save
        size_t size = 80;
        char buf[size];
        getcwd(buf, size);
        std::string prefix = std::string(buf) + "/save/";
        std::string default_rgb_save_path = prefix + "rgb/";
        std::string default_depth_save_path = prefix + "depth/";
        createDirectory(default_depth_save_path);
        createDirectory(default_rgb_save_path);
        std::cout << "The default save path: " + prefix << std::endl;

        // Sets queues size and behavior
        for (const auto &name : queueNames)
        {
            device.getOutputQueue(name, 4, false);
        }

        std::unordered_map<std::string, cv::Mat> frame;

        std::string rgbWindowName = "rgb";
        std::string depthWindowName = "depth";
        std::string blendedWindowName = "rgb-depth";
        cv::namedWindow(rgbWindowName);
        cv::namedWindow(depthWindowName);
        cv::namedWindow(blendedWindowName);
        int defaultValue = (int)(rgbWeight * 100);
        cv::createTrackbar("RGB Weight %", blendedWindowName, &defaultValue, 100, updateBlendWeights);

        char time_buffer[80];
        while (rclcpp::ok())
        {
            std::unordered_map<std::string, std::shared_ptr<dai::ImgFrame>> latestPacket;

            // the name of the available XLinkOut stream
            auto queueEvents = device.getQueueEvents(queueNames);
            for (const std::string &name : queueEvents)
            {
                auto packets = device.getOutputQueue(name)->tryGetAll<dai::ImgFrame>();
                auto count = packets.size();
                // only get the newest frame
                if (count > 0)
                    latestPacket[name] = packets[count - 1];
            }

            for (const auto &name : queueNames)
            {
                if (latestPacket.find(name) != latestPacket.end())
                {
                    if (name == depthWindowName)
                    {
                        frame[name] = latestPacket[name]->getFrame();
                        auto maxDisparity = stereo->initialConfig.getMaxDisparity();
                        // Optional, extend range 0..95 -> 0..255, for a better visualisation
                        if (1)
                            frame[name].convertTo(frame[name], CV_8UC1, 255. / maxDisparity);
                        // Optional, apply false colorization
                        if (1)
                            cv::applyColorMap(frame[name], frame[name], cv::COLORMAP_HOT);
                    }
                    else
                    {
                        frame[name] = latestPacket[name]->getCvFrame();
                    }

                    cv::imshow(name, frame[name]);
                }
            }

            // Blend when both received
            if (frame.find(rgbWindowName) != frame.end() && frame.find(depthWindowName) != frame.end())
            {
                // Need to have both frames in BGR format before blending
                if (frame[depthWindowName].channels() < 3)
                {
                    cv::cvtColor(frame[depthWindowName], frame[depthWindowName], cv::COLOR_GRAY2BGR);
                }
                cv::Mat blended;
                cv::addWeighted(frame[rgbWindowName], rgbWeight, frame[depthWindowName], depthWeight, 0, blended);
                cv::imshow(blendedWindowName, blended);

                int key = cv::waitKey(1);
                if (key == 'o' || key == 'O')
                {
                    time_t now = time(0);
                    strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", localtime(&now));
                    std::cout << "Recorded" << std::endl;
                    cv::imwrite(default_rgb_save_path + time_buffer + ".jpg", frame[rgbWindowName]);
                    cv::imwrite(default_depth_save_path + +time_buffer + ".jpg", frame[depthWindowName]);
                }
                frame.clear();
            }
        }
    }
    void startStream()
    {
    }

    ~OakCaptureNode()
    {
        cv::destroyAllWindows();
    }

private:
    dai::Pipeline pipeline_{};
    std::shared_ptr<dai::Device> device_ = nullptr;
    std::shared_ptr<dai::DataOutputQueue> RGB_q_ = nullptr;
    std::shared_ptr<dai::node::ColorCamera> camRGB_ = nullptr;
    std::shared_ptr<dai::node::XLinkOut> xoutRGB_ = nullptr;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OakCaptureNode>("OakNode");
    node->startStream();

    rclcpp::shutdown();
    return 0;
}