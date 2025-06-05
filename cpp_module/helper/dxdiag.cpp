#include "dxdiag.hpp"

bool GetScreenPixelsDXGI(
    IDXGIOutputDuplication *pDuplication,
    ID3D11Device *pDevice,
    ID3D11DeviceContext *pImmediateContext,
    int &width,
    int &height,
    std::vector<BYTE> &pixel_data_out)
{
    static const int MAX_RETRIES = 3;
    static const int RETRY_DELAY_MS = 10;

    for (int retry = 0; retry < MAX_RETRIES; retry++)
    {
        IDXGIResource *pDesktopResource = nullptr;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;

        // Try to acquire a new frame with a small timeout
        HRESULT hr = pDuplication->AcquireNextFrame(100, &frameInfo, &pDesktopResource);

        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
        {
            // No new frame available yet, this is normal
            SafeRelease(&pDesktopResource);
            if (retry < MAX_RETRIES - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                continue;
            }
            return false;
        }

        if (FAILED(hr) || !pDesktopResource)
        {
            std::cerr << "Failed to acquire next frame. HR: " << std::hex << hr << std::endl;
            if (hr == DXGI_ERROR_ACCESS_LOST)
            {
                std::cerr << "Access to desktop duplication was lost (e.g. mode change, fullscreen app). Re-initialization needed." << std::endl;
            }
            SafeRelease(&pDesktopResource);
            if (retry < MAX_RETRIES - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                continue;
            }
            return false;
        }

        // Query the texture interface from the resource
        ID3D11Texture2D *pAcquiredDesktopImage = nullptr;
        hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&pAcquiredDesktopImage));
        SafeRelease(&pDesktopResource); // Release the IDXGIResource, we have the texture now or failed

        if (FAILED(hr) || !pAcquiredDesktopImage)
        {
            std::cerr << "Failed to query ID3D11Texture2D from IDXGIResource. HR: " << std::hex << hr << std::endl;
            if (retry < MAX_RETRIES - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                continue;
            }
            return false;
        }

        D3D11_TEXTURE2D_DESC desc;
        pAcquiredDesktopImage->GetDesc(&desc);

        // Create a staging texture to copy the desktop image to
        ID3D11Texture2D *pStagingTexture = nullptr;
        D3D11_TEXTURE2D_DESC stagingDesc = desc;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.BindFlags = 0;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stagingDesc.MiscFlags = 0;

        hr = pDevice->CreateTexture2D(&stagingDesc, nullptr, &pStagingTexture);
        if (FAILED(hr) || !pStagingTexture)
        {
            std::cerr << "Failed to create staging texture. HR: " << std::hex << hr << std::endl;
            SafeRelease(&pAcquiredDesktopImage);
            if (retry < MAX_RETRIES - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                continue;
            }
            return false;
        }

        // Copy the desktop image to the staging texture
        pImmediateContext->CopyResource(pStagingTexture, pAcquiredDesktopImage);
        SafeRelease(&pAcquiredDesktopImage);

        // Map the staging texture to access the pixel data
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        hr = pImmediateContext->Map(pStagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
        if (FAILED(hr))
        {
            std::cerr << "Failed to map staging texture. HR: " << std::hex << hr << std::endl;
            SafeRelease(&pStagingTexture);
            if (retry < MAX_RETRIES - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                continue;
            }
            return false;
        }

        // Update the output dimensions
        width = desc.Width;
        height = desc.Height;

        // Calculate the total size needed for the pixel data
        size_t totalSize = width * height * 4; // 4 bytes per pixel (BGRA)
        pixel_data_out.resize(totalSize);

        // Copy the pixel data row by row
        BYTE *pSrcData = static_cast<BYTE *>(mappedResource.pData);
        BYTE *pDstData = pixel_data_out.data();
        for (int row = 0; row < height; row++)
        {
            memcpy(pDstData + row * width * 4,
                   pSrcData + row * mappedResource.RowPitch,
                   width * 4);
        }

        // Unmap the staging texture
        pImmediateContext->Unmap(pStagingTexture, 0);
        SafeRelease(&pStagingTexture);

        // Release the frame
        hr = pDuplication->ReleaseFrame();
        if (FAILED(hr))
        {
            std::cerr << "Failed to release frame. HR: " << std::hex << hr << std::endl;
            if (retry < MAX_RETRIES - 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                continue;
            }
            return false;
        }

        return true;
    }

    return false;
}

void CleanupDXGI(DXGIContext &ctx)
{
    SafeRelease(&ctx.pDesktopDupl);
    SafeRelease(&ctx.pOutput1);
    SafeRelease(&ctx.pImmediateContext);
    SafeRelease(&ctx.pDevice);
    SafeRelease(&ctx.pAdapter);
    SafeRelease(&ctx.pFactory);
}

// Initialize DXGI/DirectX and duplication objects. Returns true on success.
bool InitializeDXGI(DXGIContext &ctx)
{
    HRESULT hr;
    // Create DXGI Factory
    hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&ctx.pFactory));
    if (FAILED(hr))
    {
        std::cerr << "Failed to create DXGI Factory. HR: " << std::hex << hr << std::endl;
        return false;
    }
    // Enumerate adapters (graphics cards)
    hr = ctx.pFactory->EnumAdapters1(0, &ctx.pAdapter);
    if (FAILED(hr))
    {
        std::cerr << "Failed to enumerate adapters. HR: " << std::hex << hr << std::endl;
        SafeRelease(&ctx.pFactory);
        return false;
    }
    // Create D3D11 Device and Context
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_1,
    };
    D3D_FEATURE_LEVEL featureLevel;
    hr = D3D11CreateDevice(
        ctx.pAdapter,
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        0,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &ctx.pDevice,
        &featureLevel,
        &ctx.pImmediateContext);

    if (FAILED(hr))
    {
        std::cerr << "Failed to create D3D11 device. HR: " << std::hex << hr << std::endl;
        SafeRelease(&ctx.pAdapter);
        SafeRelease(&ctx.pFactory);
        return false;
    }
    // Enumerate outputs (monitors) on the adapter
    IDXGIOutput *pOutput = nullptr;
    hr = ctx.pAdapter->EnumOutputs(0, &pOutput);
    if (FAILED(hr))
    {
        std::cerr << "Failed to enumerate outputs. HR: " << std::hex << hr << std::endl;
        SafeRelease(&ctx.pImmediateContext);
        SafeRelease(&ctx.pDevice);
        SafeRelease(&ctx.pAdapter);
        SafeRelease(&ctx.pFactory);
        return false;
    }
    // Query for IDXGIOutput1 interface
    hr = pOutput->QueryInterface(__uuidof(IDXGIOutput1), reinterpret_cast<void **>(&ctx.pOutput1));
    SafeRelease(&pOutput);
    if (FAILED(hr))
    {
        std::cerr << "Failed to query IDXGIOutput1. HR: " << std::hex << hr << std::endl;
        SafeRelease(&ctx.pImmediateContext);
        SafeRelease(&ctx.pDevice);
        SafeRelease(&ctx.pAdapter);
        SafeRelease(&ctx.pFactory);
        return false;
    }
    // Create Desktop Duplication
    hr = ctx.pOutput1->DuplicateOutput(ctx.pDevice, &ctx.pDesktopDupl);
    if (FAILED(hr))
    {
        std::cerr << "Failed to create duplicate output. HR: " << std::hex << hr << std::endl;
        if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE)
        {
            std::cerr << "Desktop Duplication is not available. Max number of applications using it already reached?" << std::endl;
        }
        else if (hr == E_ACCESSDENIED)
        {
            std::cerr << "Access denied. Possibly due to protected content or system settings." << std::endl;
        }
        SafeRelease(&ctx.pOutput1);
        SafeRelease(&ctx.pImmediateContext);
        SafeRelease(&ctx.pDevice);
        SafeRelease(&ctx.pAdapter);
        SafeRelease(&ctx.pFactory);
        return false;
    }
    return true;
}