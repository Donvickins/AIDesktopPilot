#include "utils.hpp"

bool setUpEnv()
{
    std::filesystem::path opencv_kernel = std::filesystem::current_path() / "kernel_cache";

    if (!std::filesystem::exists(opencv_kernel))
    {
        if (!std::filesystem::create_directory(opencv_kernel))
        {
            LOG_ERR("Creating Kernel Cache Directory");
            return false;
        }
    }

    if (_putenv_s("OPENCV_OCL4DNN_CONFIG_PATH", opencv_kernel.generic_string().c_str()) != 0)
    {
        LOG_ERR("SET Kernel Cache ENV Failed");
        return false;
    }

    return true;
}

void detectSystemArch(HARDWARE_INFO &hw_info)
{
    // Check CUDA (NVIDIA)
    try
    {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0)
        {
            hw_info.has_cuda = true;
            hw_info.has_nvidia = true;
            cv::cuda::printCudaDeviceInfo(0);
        }
    }
    catch (const cv::Exception &e)
    {
        LOG_ERR("CUDA check failed: " << e.what());
    }

    // Check OpenCL (AMD, Intel, NVIDIA)
    if (cv::ocl::haveOpenCL())
    {
        cv::ocl::Context context;
        if (context.create(cv::ocl::Device::TYPE_ALL))
        {
            hw_info.has_opencl = true;
            cv::ocl::Device device = context.device(0);
            hw_info.gpu_name = device.name();
            hw_info.gpu_vendor = device.vendorName();

            // Detect vendor
            if (hw_info.gpu_vendor.find("AMD") != std::string::npos)
            {
                hw_info.has_amd = true;
            }
            else if (hw_info.gpu_vendor.find("Intel") != std::string::npos)
            {
                hw_info.has_intel = true;
            }
            else if (hw_info.gpu_vendor.find("NVIDIA") != std::string::npos)
            {
                hw_info.has_nvidia = true;
            }
        }
    }
}