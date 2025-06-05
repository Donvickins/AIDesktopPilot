#include "dxdiag.hpp"
#include "yolo.hpp"

#define LOG(...) std::cout << "[INFO] " << __VA_ARGS__ << std::endl

// YOLOv8 parameters -

cv::dnn::Net yolo_net;
std::vector<std::string> class_names_vec;

int main()
{
    LOG("Starting continuous screen capture with Desktop Duplication API...");
    LOG("Press Ctrl+C or ESC in the window to stop.");

    const int targetFps = 60;
    const int frameDelayMs = 1000 / targetFps;
    long long frameCount = 0;
    bool quit = false;

    const std::string YOLO_MODEL_PATH = (std::filesystem::current_path() / "models/yolo/yolo11l.onnx").generic_string();
    const std::string CLASS_NAMES_PATH = (std::filesystem::current_path() / "models/yolo/coco.names.txt").generic_string();

    // Create a resizable OpenCV window before the main loop
    std::string windowName = "Live Feed DXGI";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(windowName, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(windowName, 1280, 720);

    std::string input;

    LOG("Loading YOLO11l model from: " << YOLO_MODEL_PATH);
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

    LOG("Loading class names from: " << CLASS_NAMES_PATH);
    if (!loadClassNames(CLASS_NAMES_PATH, class_names_vec) || class_names_vec.empty())
    {
        std::cerr << "Error: Failed to load class names or class names file is empty." << std::endl;
        return -1;
    }

    LOG("Class names loaded: " << class_names_vec.size() << " classes.");

    while (!quit)
    {
        DXGIContext ctx;
        if (!InitializeDXGI(ctx))
        {
            std::cerr << "Initialization failed. Retrying in 2 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        int width = 0, height = 0;
        std::vector<BYTE> pixelBuffer;
        bool duplication_active = true;
        int consecutive_failures = 0;
        const int MAX_CONSECUTIVE_FAILURES = 5;

        while (duplication_active && !quit)
        {
            auto startTime = std::chrono::high_resolution_clock::now();
            if (!GetScreenPixelsDXGI(ctx.pDesktopDupl, ctx.pDevice, ctx.pImmediateContext, width, height, pixelBuffer))
            {
                DXGI_OUTDUPL_FRAME_INFO frameInfoCheck;
                IDXGIResource *resourceCheck = nullptr;
                HRESULT checkHr = ctx.pDesktopDupl->AcquireNextFrame(0, &frameInfoCheck, &resourceCheck);
                SafeRelease(&resourceCheck);

                if (checkHr == DXGI_ERROR_ACCESS_LOST)
                {
                    std::cerr << "Desktop Duplication access lost. Re-initializing..." << std::endl;
                    duplication_active = false;
                    break;
                }

                consecutive_failures++;
                if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES)
                {
                    std::cerr << "Too many consecutive failures. Re-initializing..." << std::endl;
                    duplication_active = false;
                    break;
                }

                // Add a small delay before retrying
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            consecutive_failures = 0; // Reset failure counter on success

            if (!pixelBuffer.empty())
            {
                frameCount++;
                cv::Mat frame(height, width, CV_8UC4, pixelBuffer.data());
                cv::Mat frame_bgr;
                cv::cvtColor(frame, frame_bgr, cv::COLOR_BGRA2BGR);

                // Process frame with YOLOv11
                processFrameWithYOLO(frame_bgr, yolo_net, class_names_vec);

                if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) >= 1)
                {
                    cv::imshow(windowName, frame_bgr);
                }

                int key = cv::waitKey(1);
                // Check for ESC key
                if (key == 27)
                {
                    duplication_active = false;
                    quit = true;
                }
                // Check if window was closed
                if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
                {
                    duplication_active = false;
                    quit = true;
                }

                // Maintain target frame rate
                auto endTime = std::chrono::high_resolution_clock::now();
                auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
                if (elapsedTime < frameDelayMs)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(frameDelayMs - elapsedTime));
                }

                if (frameCount % 100 == 0)
                {
                    LOG("Processed " << frameCount << " frames via DXGI.");
                }
            }
        }
        CleanupDXGI(ctx);
        if (!quit)
        {
            std::cerr << "Attempting to re-initialize DXGI in 2 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    LOG("Screen capture stopped.");
    return 0;
}