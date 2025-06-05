#pragma once

#include <vector>
#include <chrono>
#include <thread>
#include <iostream>

#include <dxgi1_2.h>
#include <d3d11.h>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")

template <class T>
void SafeRelease(T **ppT)
{
    if (*ppT)
    {
        (*ppT)->Release();
        *ppT = nullptr;
    }
}

bool GetScreenPixelsDXGI(
    IDXGIOutputDuplication *pDuplication,
    ID3D11Device *pDevice,
    ID3D11DeviceContext *pImmediateContext,
    int &width,
    int &height,
    std::vector<BYTE> &pixel_data_out);

// Helper struct to hold DXGI/DirectX objects
struct DXGIContext
{
    ID3D11Device *pDevice = nullptr;
    ID3D11DeviceContext *pImmediateContext = nullptr;
    IDXGIFactory1 *pFactory = nullptr;
    IDXGIAdapter1 *pAdapter = nullptr;
    IDXGIOutput1 *pOutput1 = nullptr;
    IDXGIOutputDuplication *pDesktopDupl = nullptr;
};

void CleanupDXGI(DXGIContext &ctx);
bool InitializeDXGI(DXGIContext &ctx);