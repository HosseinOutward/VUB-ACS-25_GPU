#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <ocl_utilities.hpp>
#include <cmath>
#include <direct.h>
#include "../Conv_2d_gpu/conv_2d_gpu.hpp"
#include "function.cpp"

void run(char* argv[], 
        std::vector<unsigned char>& input_batch, int& input_width, int& input_height, int& channel_count,
        std::vector<float>& filter, int& filter_width, int& filter_height, int& filter_channels, int& stride,
        std::vector<float>& output, int& output_width, int& output_height)
{
    /*------------------ run ------------------*/
    int batch_repeats = std::stoi(argv[4]);
    int threads = std::stoi(argv[3]);
    std::cout << "Running " << batch_repeats << " times\n\n";
    std::cout << "Input size: " << input_width << "x" << input_height << "x" << channel_count << "\n";
    std::cout << "Using " << (threads == 0 ? "all available hardware" : std::to_string(threads))
            << " thread" << (threads == 1 ? "" : "s") << "\n\n";

auto time_overall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < batch_repeats; ++i)
        conv2d_cpu_threaded(input_batch, input_width, input_height, channel_count,
                filter, filter_width, filter_height, filter_channels,
                output, threads, stride, output_width, output_height);

auto time_overall_end = std::chrono::high_resolution_clock::now();
auto time_overall = std::chrono::duration_cast<std::chrono::nanoseconds>(time_overall_end - time_overall_start);

    /*------------------ timming results ------------------*/
std::cout << "Overall_time: " << time_overall.count()/1000000.0 << std::endl;
}


int main(int argc, char* argv[]) {
    char** default_argv = new char*[8];
    default_argv[0] = argv[0];
    default_argv[1] = (char*)"input_lena.png"; // input file (w h img_c)
    default_argv[2] = (char*)
        "kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt"; // kernel file (fw fh img_c fc)
        // "kernel_same_1.txt,kernel_same_2.txt,kernel_same_3.txt";
    default_argv[3] = (char*)"0"; // cpu: threads --- gpu: local mem
    default_argv[4] = (char*)"2"; // batch counts (w h c batch_size)
    default_argv[5] = (char*)"output_seq.png"; // output file
    default_argv[6] = (char*)"2"; // stride
    default_argv[7] = (char*)"10"; // simulated single batch size 
    return replace_argv(argc, argv, default_argv, run);
}