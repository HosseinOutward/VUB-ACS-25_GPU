#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <windows.h>

inline cl::Context makeContextAndQueue(cl::CommandQueue* outQueue, cl::Device* outDevice) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
        throw std::runtime_error("No OpenCL platforms found");

    cl::Device device;
    char dev_name[128];
    for (const auto& plat : platforms)
    {
        std::vector<cl::Device> devices;
        plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        plat.getInfo(CL_PLATFORM_NAME, &dev_name);
        if (!devices.empty() && std::string(dev_name).find("NVIDIA") != std::string::npos) {
            device = devices[0];
            break;
        }
    }
    if (!device())
        throw std::runtime_error("No GPU devices found");
    
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context({device});
    // Enable profiling for the command queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    if (outQueue)  *outQueue  = queue;
    if (outDevice) *outDevice = device;

    return context;
}

std::string loadKernelSource() {
    // Get the path of the current executable
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    
    // Extract the filename without extension
    std::string fullPath(buffer);
    size_t lastSlash = fullPath.find_last_of("/\\");
    size_t lastDot = fullPath.find_last_of(".");
    
    // Get only the executable name without extension
    std::string exec_file_name;
    if (lastSlash != std::string::npos && lastDot != std::string::npos) {
        exec_file_name = fullPath.substr(lastSlash + 1, lastDot - lastSlash - 1);
    } else {
        exec_file_name = "unknown";
    }

    // if has _HTP, remove it
    if (exec_file_name.find("_analysis") != std::string::npos) {
        exec_file_name.erase(exec_file_name.find("_analysis"), 10);
    }

    // Construct the full path to the kernel file
    std::string path = "src/" + exec_file_name + "/kernel.ocl";

    std::ifstream file(path);
    if (!file){
        throw std::runtime_error("Cannot open kernel file: " + path);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}