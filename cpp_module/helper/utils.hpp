#pragma once

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#define LOG(...) std::cout << "[INFO] " << __VA_ARGS__ << std::endl;
#define LOG_ERR(...) std::cout << "[ERROR] " << __VA_ARGS__ << std::endl;

struct HARDWARE_INFO
{
    bool has_cuda = false;
    bool has_opencl = false;
    bool has_amd = false;
    bool has_intel = false;
    bool has_nvidia = false;
    std::string gpu_name;
    std::string gpu_vendor;
};

bool setUpEnv();
void detectSystemArch(HARDWARE_INFO &hw_info);