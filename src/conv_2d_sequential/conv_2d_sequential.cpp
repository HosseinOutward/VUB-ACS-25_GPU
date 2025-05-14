#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "function.cpp"
#include "utilities.cpp"

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

    // ------------------ choose thread count ------------------
    int threads = 0; // Changed from 12 to 0 to use hardware_concurrency()
    std::cout << "Using " << (threads == 0 ? "all available hardware" : std::to_string(threads)) << " thread"
              << (threads == 1 ? "" : "s") << "\n\n";
              
    // ------------------ run CPU convolution ------------------
    std::cout << "\nbeginning conv2d_cpu_threaded\n";
    conv2d_cpu_threaded(input_img.data(), input_width, input_height, channel_count,
               filter.data(), filter_width, filter_height,
               output.data(), threads, stride);
    std::cout << "finished conv2d_cpu_threaded\n";

    // ------------------ result -------------------------
    writePngImage(output, output_width, output_height, channel_count, "data/out/output.png"); // Write the actual output data
}
