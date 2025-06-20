#include "dxdiag.hpp"

int main()
{
    DXGIContext ctx;
    cv::Mat screenshot;
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
                    Cleanup(ctx);                     // Clean up existing components
                    hr = InitDesktopDuplication(ctx); // Try to reinitialize
                    if (FAILED(hr))
                    {
                        std::cerr << "Failed to reinitialize DirectX. Exiting." << std::endl;
                        break; // Exit loop on critical failure
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
                    hr = InitDesktopDuplication(ctx); // Try to reinitialize (will create new g_DeskDupl)
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

            // Wait for the specified interval before the next capture
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
