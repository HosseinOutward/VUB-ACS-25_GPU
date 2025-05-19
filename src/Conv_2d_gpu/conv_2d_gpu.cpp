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
#include "conv_2d_gpu.hpp"


void meow(char* argv[])
{
    /*------------------ INPUT ------------------*/
    std::vector<unsigned char> input_batch;
    std::vector<float> filter, output; 
    int input_width, input_height, channel_count, 
        filter_width, filter_height, filter_channels;

    // Load batch
    int reps = 1;
    if (isDimensionString(argv[1])) {
        std::vector<int> d = parseDimensions(argv[1]);
        input_width = d[0];
        input_height = d[1];
        channel_count = d[2];
        input_batch = load_input_batch(d[0], d[1], d[2]);
    } else {
        reps = 10;
        input_batch = load_input_batch(argv[1], reps, input_width, input_height, channel_count);
        channel_count *= reps;
    }

    // Load filter
    if (isDimensionString(argv[2])) {
        std::vector<int> d = parseDimensions(argv[2]);
        filter_width = d[0];
        filter_height = d[1];
        filter_channels = d[2];
        filter = load_filter(d[0], d[1], d[2]);
    } else
        filter = load_filter(argv[2], filter_width, filter_height, filter_channels);

    /*------------------ output dim ------------------*/
    int stride = 2;
    int output_width = (input_width - filter_width) / stride + 1;
    int output_height = (input_height - filter_height) / stride + 1;
    output.resize(static_cast<size_t>(output_width) * output_height
                                     * channel_count * filter_channels);

    /*------------------ gpu init ------------------*/
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
    cl::Program program;
    cl::Kernel kernel;

    initializeGPU(context, queue, device, program, kernel, (strcmp(argv[3], "true") == 0),
        stride, input_width, input_height, channel_count, filter_width, filter_height, 
        filter_channels, output_width, output_height);

    /*------------------ local memory info ------------------*/
    size_t global_work_size_x, global_work_size_y;
    global_work_size_x = output_width;
    global_work_size_y = output_height;

    cl::NDRange local_dims = cl::NullRange;
    
    if (strcmp(argv[3], "true") == 0) {
        size_t local_wg_dim_x, local_wg_dim_y;
        configureLocalMemoryAndWorkers(
            device, kernel, stride, 
            filter_width, filter_height, output_width, output_height, 
            local_wg_dim_x, local_wg_dim_y, global_work_size_x, global_work_size_y
        );
        
        local_dims = cl::NDRange(local_wg_dim_x, local_wg_dim_y, 1);
    }

    cl::NDRange global_dims(global_work_size_x, global_work_size_y, channel_count);

    /*------------------ set input/output vector ------------------*/
    cl::Buffer d_filter = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filter.size(), const_cast<float*>(filter.data()));
    kernel.setArg(2, d_filter);

    int batch_repeats = std::stoi(argv[4]);
    std::vector<cl::Buffer> d_input(batch_repeats), d_output_buf(batch_repeats);

    for (int i = 0; i < batch_repeats; ++i) {
        d_input[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(unsigned char) * input_batch.size(), 
                        const_cast<unsigned char*>(input_batch.data()));

        d_output_buf[i] = cl::Buffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * output.size(), nullptr);

        kernel.setArg(0, d_input[i]);
        kernel.setArg(1, d_output_buf[i]);

        // enqueue the kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims);
    }

    for (int i = 0; i < batch_repeats; ++i) {
        queue.enqueueReadBuffer(d_output_buf[i], CL_FALSE, 0, 
            sizeof(float) * output.size(), output.data());
    }

    // ------------------ result -------------------------
    if (strcmp(argv[5], "null") == 0)
        return;
    else{
        // only give the first 3 channels
        output.resize(static_cast<size_t>(output_width) * output_height * 3);
        writePngImage(output, output_width, output_height, 3, argv[5]);
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 1) {
            char** new_argv = new char*[5];
            new_argv[0] = argv[0];
            new_argv[1] = (char*)"input_lena.png";
            new_argv[2] = (char*)"kernel_edge_3c.txt";
            new_argv[3] = (char*)"true";
            new_argv[4] = (char*)"1";
            new_argv[5] = (char*)"output.png";
            meow(new_argv);
            delete[] new_argv;
        } else
            meow(argv);
    // ------------------ try catch -------------------------
    } catch (cl::Error const &e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1; // Indicate an error
    } catch (std::exception const &e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0; // Indicate success
}