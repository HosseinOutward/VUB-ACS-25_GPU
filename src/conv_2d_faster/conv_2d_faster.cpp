#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <ocl_utilities.hpp>
#include "..\conv_2d_sequential\function.cpp"
#include "..\conv_2d_sequential\utilities.cpp"
#include <cmath> // Required for std::sqrt

int main()
{
    try { // Add try block for OpenCL errors
    /*------------------ INPUT ------------------*/
    // Load paths from config file
    FilePaths paths = loadPathsFromConfig("data/in/config_conv.txt");

    int input_width, input_height, channel_count;
    std::vector<float> input_img = readPngImage(paths.inputImagePath, input_width, input_height, channel_count);

    // Filter
    int filter_width, filter_height;
    std::vector<float> filter = loadConvKernelFromFile(paths.kernelFilePath, filter_height, filter_width);

    int stride = 1;
    const int output_width = (input_width - filter_width) / stride + 1;
    const int output_height = (input_height - filter_height) / stride + 1;
    std::vector<float> output(static_cast<size_t>(output_width) * output_height * channel_count);

    // ------------------ gpu init ------------------
    cl::CommandQueue queue;
    cl::Device device;
    cl::Context context = makeContextAndQueue(&queue, &device);
    const std::string kernel_source_code = loadKernelSource();
    cl::Program program(context, kernel_source_code);
    program.build({device});
    cl::Kernel kernel(program, "kernel_app");

    cl::Buffer d_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * input_img.size(), input_img.data());
    cl::Buffer d_filter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filter.size(), filter.data());
    cl::Buffer d_output(context, CL_MEM_READ_WRITE,
                        sizeof(float) * output.size(), nullptr);
    kernel.setArg(0, d_input);
    kernel.setArg(1, d_filter);
    kernel.setArg(2, d_output);
    kernel.setArg(3, stride);
    kernel.setArg(4, input_width);
    kernel.setArg(5, input_height);
    kernel.setArg(6, filter_width);
    kernel.setArg(7, filter_height);
    kernel.setArg(8, channel_count);
    kernel.setArg(11, output_width);
    kernel.setArg(12, output_height);

    // ------------------ query device limits ------------------
    size_t max_total_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::vector<size_t> max_item_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    size_t max_local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    // std::cout << "Device Max Work Group Size: " << max_total_wg_size << std::endl;
    // std::cout << "Max local memory size: " << max_local_mem_size / 1024 << " KB" << std::endl;
    // if (max_item_sizes.size() >= 1) std::cout << "Device Max Work Item Sizes[0]: " << max_item_sizes[0] << std::endl;
    // if (max_item_sizes.size() >= 2) std::cout << "Device Max Work Item Sizes[1]: " << max_item_sizes[1] << std::endl;
    // if (max_item_sizes.size() >= 3) std::cout << "Device Max Work Item Sizes[2]: " << max_item_sizes[2] << std::endl;

    // ------------------ local memory and worker config ------------------
    // ------------ calculate optimal workgroup size
    std::vector<float> quads = {1.0f, 1.0f*stride*(filter_width+filter_height), 
        1.0f*(filter_width*filter_height) - (1.0f*max_local_mem_size)/sizeof(float)};
    double discriminant = static_cast<double>(quads[1])*quads[1] - 4.0*quads[0]*quads[2];
    int optimal_wg_size = 1 + static_cast<int>(
        (-quads[1] + std::sqrt(discriminant)) / (2.0f * quads[0]));

    std::cout << "Calculated optimal_wg_size (complete coverage local mem): " << optimal_wg_size << std::endl;
    optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(max_item_sizes[0]));
    std::cout << "Calculated optimal_wg_size (considering max #worker): " << optimal_wg_size << std::endl;
    optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(
            std::sqrt(static_cast<double>(max_total_wg_size))));
    std::cout << "Calculated optimal_wg_size (considering max #wg): " << optimal_wg_size << std::endl;

    const size_t local_wg_dim_x = optimal_wg_size;
    const size_t local_wg_dim_y = optimal_wg_size;

    // ------------ per group tile dimensions & local memory size
    size_t tile_w_host = (local_wg_dim_x - 1) * stride + filter_width;
    size_t tile_h_host = (local_wg_dim_y - 1) * stride + filter_height;
    size_t local_memory_size_bytes = tile_w_host * tile_h_host * sizeof(float);
    kernel.setArg(9, static_cast<int>(tile_w_host));
    kernel.setArg(10, static_cast<int>(tile_h_host));
    kernel.setArg(13, cl::Local(local_memory_size_bytes));
    
    // ------------ global and local work size
    cl::NDRange local_dims(local_wg_dim_x, local_wg_dim_y, 1); // worker groups

    size_t global_work_size_x = ((output_width + local_wg_dim_x - 1) / local_wg_dim_x) * local_wg_dim_x;
    size_t global_work_size_y = ((output_height + local_wg_dim_y - 1) / local_wg_dim_y) * local_wg_dim_y;
    cl::NDRange global_dims(global_work_size_x, global_work_size_y, channel_count);

    // ------------------ run gpu convolution ------------------
    std::cout << "\nbeginning conv2d fast\n";

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims);
    std::cout << "finished conv2d fast\n";
    
    queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data());
    std::cout << "finished reading output off vram\n";

    // ------------------ result -------------------------
    writePngImage(output, output_width, output_height, channel_count, paths.outputImagePath);

    // ------------------ try catch -------------------------
    } catch (cl::Error const &e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        // You can get more specific error codes using e.err()
        // e.g. CL_INVALID_WORK_GROUP_SIZE, CL_OUT_OF_RESOURCES, etc.
        // Consult OpenCL documentation for clEnqueueNDRangeKernel error codes.
        return 1; // Indicate an error
    } catch (std::exception const &e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0; // Indicate success
}
