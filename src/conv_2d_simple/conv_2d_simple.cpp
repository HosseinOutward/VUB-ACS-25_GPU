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

int main()
{
    /*------------------ INPUT ------------------*/
    int input_width, input_height, channel_count;
    // std::vector<float> input_img = readPngImage("data/in/input_5x5.png", input_width, input_height, channel_count);
    std::vector<float> input_img = readPngImage("data/in/input_lena.png", input_width, input_height, channel_count);

    // Filter
    int filter_width, filter_height;
    // std::vector<float> filter = loadConvKernelFromFile("data/in/kernel_same.txt", filter_height, filter_width);
    std::vector<float> filter = loadConvKernelFromFile("data/in/kernel_edge.txt", filter_height, filter_width);

    int stride = 1; // Stride for convolution
    const int output_width = (input_width - filter_width) / stride + 1;
    const int output_height = (input_height - filter_height) / stride + 1;
    std::vector<float> output(static_cast<size_t>(output_width) * output_height * channel_count); // Correctly sized for multi-channel output

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

    std::vector<int> other_ints = {stride, input_width, input_height, filter_width, filter_height, channel_count}; // Added channel_count
    cl::Buffer d_other_ints(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(int) * other_ints.size(), other_ints.data());
    kernel.setArg(3, d_other_ints);

    // ------------------ run gpu convolution ------------------
    std::cout << "\nbeginning conv2d_cpu_threaded\n";
    // Update NDRange to include channels
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output_width, output_height, channel_count));
    std::cout << "finished conv2d_cpu_threaded\n";
    queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data());
    std::cout << "finished reading output off vram\n";

    // ------------------ result -------------------------
    writePngImage(output, output_width, output_height, channel_count, "data/out/output.png"); // Write the actual output data
}
