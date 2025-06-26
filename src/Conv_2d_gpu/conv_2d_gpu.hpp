#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <windows.h>
#include <cl2.hpp>

std::vector<int> get_nvidia_gpu_indices() {
    std::vector<int> nvidia_gpu_indices;
    
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        int gpu_index = 0;
        for (const auto& platform : platforms) {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            
            std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
            if (platform_name.find("NVIDIA") != std::string::npos) {
                for (size_t i = 0; i < devices.size(); ++i) {
                    nvidia_gpu_indices.push_back(gpu_index + i);
                }
            }
            gpu_index += devices.size();
        }
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL Error while detecting GPUs: " << e.what() << std::endl;
    }
    
    return nvidia_gpu_indices;
}

inline cl::Context makeContextAndQueue(cl::CommandQueue* outQueue, cl::Device* outDevice, int gpu_idx) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
        throw std::runtime_error("No OpenCL platforms found");

    // Find NVIDIA platform and get the device directly at gpu_idx
    for (const auto& plat : platforms) {
        std::string platform_name = plat.getInfo<CL_PLATFORM_NAME>();
        if (platform_name.find("NVIDIA") != std::string::npos) {
            std::vector<cl::Device> devices;
            plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            
            if (gpu_idx >= 0 && gpu_idx < static_cast<int>(devices.size())) {
                cl::Device device = devices[gpu_idx];
                std::cout << "Using GPU index " << gpu_idx << " - device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

                cl::Context context({device});
                cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

                if (outQueue)  *outQueue  = queue;
                if (outDevice) *outDevice = device;

                return context;
            } else {
                throw std::runtime_error("GPU index " + std::to_string(gpu_idx) + " out of range. Available NVIDIA GPUs: " + std::to_string(devices.size()));
            }
        }
    }
    
    throw std::runtime_error("No NVIDIA platform found");
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

// Helper function for GPU initialization
void initializeGPU(
    int gpu_idx,
    cl::Context& context, 
    cl::CommandQueue& queue, 
    cl::Device& device, 
    cl::Program& program, 
    cl::Kernel& kernel,
    int uses_local_mem,
    int stride, int input_width, int input_height, int channel_count,
    int filter_width, int filter_height, int filter_channels,
    int output_width, int output_height
) {
    context = makeContextAndQueue(&queue, &device, gpu_idx);
    const std::string kernel_source_code = loadKernelSource();
    program = cl::Program(context, kernel_source_code);
    program.build({device});
    if (uses_local_mem)
        kernel = cl::Kernel(program, "kernel_app_local");
    else
        kernel = cl::Kernel(program, "kernel_app");

    kernel.setArg(3, stride);
    kernel.setArg(4, input_width);
    kernel.setArg(5, input_height);
    kernel.setArg(6, channel_count);
    kernel.setArg(7, filter_width);
    kernel.setArg(8, filter_height);
    kernel.setArg(9, filter_channels);
    kernel.setArg(10, output_width);
    kernel.setArg(11, output_height);
}

// Helper function for local memory and worker configuration
int getOptimalWGSize(
    const cl::Device& device, int stride, 
    int filter_width, int filter_height, int filter_channels, int input_channels
) {
    // _______________________________ calculate optimal local mem size _______________________________
    size_t max_local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() 
                - filter_width*filter_height*filter_channels*input_channels*sizeof(float);
    
    int optimal_wg_size = -1;
    std::vector<float> quads = {1.0f*stride*stride, 1.0f*stride*(filter_width+filter_height),
        1.0f*(filter_width+1)/2*(filter_height+1)/2 - (1.0f*max_local_mem_size)/sizeof(unsigned char)/input_channels};

    double discriminant = static_cast<double>(quads[1])*quads[1] - 4.0*quads[0]*quads[2];
    if (discriminant >= 0)
        optimal_wg_size = 1 + static_cast<int>((-quads[1] + std::sqrt(discriminant)) / (2.0f * quads[0]));
    else
        std::cerr << "Warning: Negative discriminant." << std::endl;

    // consider hardware limits
    std::vector<size_t> max_item_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    if (!max_item_sizes.empty())
        optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(max_item_sizes[0]));

    size_t max_total_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(
            std::sqrt(static_cast<double>(max_total_wg_size))));
    
    // consider depth
    optimal_wg_size = optimal_wg_size / sqrtf(filter_channels);

    return optimal_wg_size;
}

// Helper function for local memory and worker configuration
void configureLocalMemoryAndWorkers(
    const cl::Device& device, cl::Kernel& kernel, int stride, 
    int filter_width, int filter_height, int filter_channels,
    int output_width, int output_height, int input_channels,
    size_t& local_wg_dim_x, size_t& local_wg_dim_y,
    size_t& global_work_size_x, size_t& global_work_size_y
) {
    // consider depth
    int optimal_wg_size = getOptimalWGSize(device, stride,
         filter_width, filter_height, filter_channels, input_channels);

    std::cout << "WI per WG: " << optimal_wg_size*optimal_wg_size*filter_channels << std::endl;

    // _______________________________ allocate _______________________________
    local_wg_dim_x = optimal_wg_size;
    local_wg_dim_y = optimal_wg_size;
    
    global_work_size_x = ((output_width + local_wg_dim_x - 1) / local_wg_dim_x) * local_wg_dim_x;
    global_work_size_y = ((output_height + local_wg_dim_y - 1) / local_wg_dim_y) * local_wg_dim_y;
    
    size_t tile_w_host = local_wg_dim_x * stride + (filter_height+1)/2;
    size_t tile_h_host = local_wg_dim_y * stride + (filter_height+1)/2;
    size_t local_memory_size_bytes = tile_w_host * tile_h_host * input_channels * sizeof(unsigned char);
    size_t local_filter_memory_size_bytes = static_cast<size_t>(filter_width) * filter_height 
                                    * input_channels * filter_channels * sizeof(float);

    kernel.setArg(12, static_cast<int>(tile_w_host));
    kernel.setArg(13, static_cast<int>(tile_h_host));
    kernel.setArg(14, cl::Local(local_memory_size_bytes));
    kernel.setArg(15, cl::Local(local_filter_memory_size_bytes));
}
