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

// Define local work-group size (e.g., 16x16)
const size_t local_wg_dim_x = 16;
const size_t local_wg_dim_y = 16;

int main()
{
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

    // Calculate local memory size needed by the kernel for the input tile.
    // The tile dimensions depend on local work-group size, stride, and filter dimensions.
    size_t tile_w = (local_wg_dim_x > 0 ? (local_wg_dim_x - 1) : 0) * stride + filter_width;
    size_t tile_h = (local_wg_dim_y > 0 ? (local_wg_dim_y - 1) : 0) * stride + filter_height;
    size_t local_memory_size_bytes = tile_w * tile_h * sizeof(float);

    cl::Buffer d_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * input_img.size(), input_img.data());
    cl::Buffer d_filter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filter.size(), filter.data());
    cl::Buffer d_output(context, CL_MEM_READ_WRITE,
                        sizeof(float) * output.size(), nullptr);
    kernel.setArg(0, d_input);
    kernel.setArg(1, d_filter);
    kernel.setArg(2, d_output);

    std::vector<int> other_ints = {stride, input_width, input_height, filter_width, filter_height, channel_count}; // Added channel_count
    cl::Buffer d_other_ints(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(int) * other_ints.size(), other_ints.data());
    kernel.setArg(3, d_other_ints); // d_other_ints is now argument 3 (0-indexed)
    kernel.setArg(4, cl::Local(local_memory_size_bytes)); // New local memory argument is 4

    // ------------------ run gpu convolution ------------------
    std::cout << "\nbeginning conv2d\n";
    // Update NDRange to include channels and specify local work size
    // Global work size needs to be a multiple of local work size for simplicity, or handle edge cases in kernel.
    // For now, let's ensure global size is rounded up to be a multiple of local size.
    size_t global_work_size_x = ((output_width + local_wg_dim_x - 1) / local_wg_dim_x) * local_wg_dim_x;
    size_t global_work_size_y = ((output_height + local_wg_dim_y - 1) / local_wg_dim_y) * local_wg_dim_y;

    cl::NDRange global_dims(global_work_size_x, global_work_size_y, channel_count);
    cl::NDRange local_dims(local_wg_dim_x, local_wg_dim_y, 1); // Local work group for 2D patch, channel processed by 3rd global dim

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims);
    std::cout << "finished conv2d\n";
    queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data());
    std::cout << "finished reading output off vram\n";

    // ------------------ result -------------------------
    writePngImage(output, output_width, output_height, channel_count, paths.outputImagePath); // Write the actual output data
}
