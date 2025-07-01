#include "dxdiag.hpp"
#include "yolo.hpp"
#include "utils.hpp"

int main()
{
    if (!setUpEnv())
    {
        LOG_ERR("Failed to set up environment. Exiting.");
        return -1;
    }

    cv::ocl::setUseOpenCL(true);
    DXGIContext ctx;
    HARDWARE_INFO hw_info;
    cv::Mat screenshot;
    detectSystemArch(hw_info);
    // Initialize COM for WIC (Windows Imaging Component)
    // CoInitializeEx is needed for COM operations like CoCreateInstance.
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr))
    {
        std::cerr << "Failed to initialize COM. HRESULT: 0x" << std::hex << hr << std::dec << std::endl;
        return 1;
    }

    if (FAILED(InitDesktopDuplication(ctx)))
    {
        Cleanup(ctx);
        return 1;
    }

    std::cout << "Starting screenshot capture loop. Press Ctrl+C to stop." << std::endl;

    const std::chrono::seconds interval(5); // Capture every 5 seconds

    // Now initialize YOLO
    cv::dnn::Net yolo_net;
    std::vector<std::string> class_names_vec;

    const std::string YOLO_MODEL_PATH = (std::filesystem::current_path() / "models/yolo/yolo11l.onnx").generic_string();
    const std::string CLASS_NAMES_PATH = (std::filesystem::current_path() / "models/yolo/coco.names.txt").generic_string();

    LOG("Initializing YOLO network...");
    if (!setupYoloNetwork(yolo_net, YOLO_MODEL_PATH, CLASS_NAMES_PATH, class_names_vec, hw_info))
    {
        LOG_ERR("Failed to setup YOLO network");
        return -1;
    }
    // Main loop for capturing screenshots
    while (true)
    {
        try
        {
            std::string imagePath = "screenshots";
            bool captured = false;
            hr = CaptureScreenshot(ctx, imagePath, captured, screenshot);

            if (FAILED(hr))
            {
                // Specific error handling for capture errors
                if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_HUNG)
                {
                    std::cerr << "DirectX device lost or reset. Attempting to reinitialize..." << std::endl;
                    Cleanup(ctx);
                    hr = InitDesktopDuplication(ctx);
                    if (FAILED(hr))
                    {
                        std::cerr << "Failed to reinitialize DirectX. Exiting." << std::endl;
                        break;
                    }
                }
                else if (hr == DXGI_ERROR_ACCESS_LOST)
                {
                    std::cerr << "Desktop duplication access lost (e.g., session switch, UAC). Releasing and reacquiring..." << std::endl;
                    if (ctx.pDesktopDupl)
                    {
                        ctx.pDesktopDupl->Release();
                        ctx.pDesktopDupl = nullptr;
                    }
                    hr = InitDesktopDuplication(ctx);
                    if (FAILED(hr))
                    {
                        std::cerr << "Failed to re-establish desktop duplication. Exiting." << std::endl;
                        break;
                    }
                }
                else
                {
                    std::cerr << "Unhandled capture error, continuing. HRESULT: 0x" << std::hex << hr << std::dec << std::endl;
                }
            }

            // Implement yoloProcessing here
            // try
            // {
            //     processFrameWithYOLO(screenshot, yolo_net, class_names_vec);
            // }
            // catch (const cv::Exception &e)
            // {
            //     LOG_ERR("OpenCV error processing YOLO: " << e.what());
            //     continue;
            // }
            // catch (const std::exception &e)
            // {
            //     LOG_ERR("Processing frame YOLO: " << e.what());
            //     break;
            // }

            // cv::imshow("Screenshot", screenshot);
            // cv::waitKey(1);
            // if (cv::getWindowProperty("Screenshot", cv::WND_PROP_VISIBLE) < 1)
            //     break;

            std::this_thread::sleep_for(interval);
        }
        catch (const std::exception &e)
        {
            std::cerr << "An unexpected C++ exception occurred: " << e.what() << std::endl;
            break;
        }
        catch (...)
        {
            std::cerr << "An unknown exception occurred." << std::endl;
            break;
        }
    }

    Cleanup(ctx); // Perform final cleanup
    return 0;
}
