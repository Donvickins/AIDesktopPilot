#include "yolo.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define LOG(...) std::cout << "[INFO] " << __VA_ARGS__ << std::endl
#define LOG_ERR(...) std::cerr << "[ERROR] " << __VA_ARGS__ << std::endl;

int main()
{

    LOG("Starting Webcam Feed...");
    LOG("Press CTRL + C to exit");

    const int targetFps = 30;
    int frameDelayMs = 1000 / targetFps;
    long long frameCount = 0;
    bool quit = false;

    cv::dnn::Net yolo_net;
    std::vector<std::string> class_names_vector;

    const std::string YOLO_MODEL_PATH = (std::filesystem::current_path() / "models/yolo/yolo11l.onnx").generic_string();
    const std::string CLASS_NAMES_PATH = (std::filesystem::current_path() / "models/yolo/coco.names.txt").generic_string();

    try
    {
        yolo_net = cv::dnn::readNetFromONNX(YOLO_MODEL_PATH);

        if (yolo_net.empty())
        {
            std::cerr << "Error: Failed to load YOLO model." << std::endl;
            return -1;
        }
        // Check if there is Cuda enable device
        if (cv::cuda::getCudaEnabledDeviceCount() > 0)
        {
            LOG("Cuda is Supported: Using Cuda");
            yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        else
        {
            LOG("Cuda is not Supported: Using CPU");
            yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        LOG("YOLO11l model loaded successfully.");
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "Failed to load YOLO model: " << e.what() << std::endl;
        return -1;
    }

    static const std::string windowName = "Webcam Live Feed";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(windowName, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(windowName, 1280, 720);

    cv::VideoCapture webcam(0);

    if (!webcam.isOpened())
    {
        LOG_ERR("Cannot open webcam");
        return -1;
    }

    LOG("Webcam Initialized successfully");

    while (!quit)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        cv::Mat frame_bgr;
        webcam >> frame_bgr;

        if (!frame_bgr.empty())
        {
            frameCount++;

            processFrameWithYOLO(frame_bgr, yolo_net, class_names_vector);
            if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) >= 1)
            {
                cv::imshow(windowName, frame_bgr);
            }

            int key = cv::waitKey(1);
            // Check for ESC key
            if (key == 27)
            {
                quit = true;
            }
            // Check if window was closed
            if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
            {
                quit = true;
            }

            // Maintain target frame rate
            auto endTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            if (elapsedTime < frameDelayMs)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(frameDelayMs - elapsedTime));
            }
        }
        else
        {
            LOG_ERR("Webcam Disconnected or Failed to get frames");
            quit = true;
        }
    }

    webcam.release();
    cv::destroyAllWindows();
    LOG("Webcam Feed Ended");
    return 0;
}