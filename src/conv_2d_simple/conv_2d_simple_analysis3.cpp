#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono> // Added for timing
#include <ocl_utilities.hpp>
#include "..\conv_2d_sequential\function.cpp"
#include "..\conv_2d_sequential\utilities.cpp"
#include "conv_2d_faster.hpp"
#include <cmath>

int main(int argc, char* argv[])
{
    /*------------------ INPUT ------------------*/
    FilePaths paths;
    std::vector<unsigned char> input_img;
    std::vector<float> filter_original_loaded, filter, single_output_template;
    int input_width, input_height, channel_count,
        filter_width, filter_height, stride, output_width, output_height;

    // Call loadInputs - it will also calculate output dimensions and initialize a template output vector
    loadInputs(paths, input_img, input_width, input_height,
                channel_count, filter, filter_width, filter_height,
                stride, output_width, output_height, single_output_template);

    // ------------------ gpu init ------------------
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
    cl::Program program;
    cl::Kernel kernel;

    initializeGPU(context, queue, device, program, kernel,
        stride, input_width, input_height, filter_width, filter_height,
        channel_count, output_width, output_height);
    
    // ------------------ local memory and worker config ------------------
    size_t global_work_size_x, global_work_size_y;
    global_work_size_x = static_cast<size_t>(output_width);
    global_work_size_y = static_cast<size_t>(output_height); 

    cl::NDRange global_dims(global_work_size_x, global_work_size_y, channel_count);

    // ------------------ high throughput batch pipeline ------------------
    int batch_count = 50;
    if (argc > 1) batch_count = std::stoi(argv[1]);

    // Buffer for the filter
    cl::Buffer d_filter(context, CL_MEM_READ_ONLY, sizeof(float) * filter.size(), nullptr);
    
    // Vectors for input/output buffers
    std::vector<cl::Buffer> d_inputs(batch_count);
    std::vector<cl::Buffer> d_outputs(batch_count);
    
    std::vector<std::vector<float>> outputs(batch_count);

    std::cout << "\nBeginning sequential conv2d with " << batch_count << " batches\n";

    auto overall_start_time = std::chrono::high_resolution_clock::now(); // Start timing before VRAM transfers

    // Enqueue write for the filter buffer and wait for it to complete
    cl::Event filter_write_event;
    queue.enqueueWriteBuffer(d_filter, CL_TRUE, 0, sizeof(float) * filter.size(), filter.data(), nullptr, &filter_write_event);
    filter_write_event.wait(); // Ensure filter is written before any batch processing

    // Prepare output vectors and OpenCL output buffers (can still be done upfront)
    for (int i = 0; i < batch_count; ++i) {
        outputs[i].resize(single_output_template.size());
        d_inputs[i] = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * input_img.size(), nullptr);
        d_outputs[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputs[i].size(), nullptr);
    }

    // Main sequential loop: Write, Kernel, Read for each batch, waiting for completion of each step.
    std::cout << "Processing tasks for " << batch_count << " batches sequentially..." << std::endl;

    for (int i = 0; i < batch_count; ++i) {
        // 1. Enqueue Input Write & Wait
        cl::Event input_write_event;
        queue.enqueueWriteBuffer(d_inputs[i], CL_TRUE, 0, sizeof(unsigned char) * input_img.size(),
                                 (input_img.data()), nullptr, &input_write_event);
        input_write_event.wait();

        // 2. Enqueue Kernel & Wait
        kernel.setArg(0, d_inputs[i]);
        kernel.setArg(1, d_filter);
        kernel.setArg(2, d_outputs[i]);
        
        cl::Event kernel_event;
        // No explicit event dependencies needed as previous operations are blocking (CL_TRUE or explicit wait)
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, cl::NullRange, nullptr, &kernel_event);
        kernel_event.wait();

        // 3. Enqueue Output Read & Wait
        cl::Event read_event;
        queue.enqueueReadBuffer(d_outputs[i], CL_TRUE, 0, sizeof(float) * outputs[i].size(),
                                outputs[i].data(), nullptr, &read_event);
        read_event.wait();
    }
    std::cout << "All batches processed sequentially." << std::endl;

    auto overall_end_time = std::chrono::high_resolution_clock::now();
    auto overall_duration_ms = std::chrono::duration<double, std::milli>(
                overall_end_time - overall_start_time).count();

    // Print only the overall timing information
    std::cout << "OverallBatchProcessingTime_ms:" << overall_duration_ms << std::endl;
    std::cout << "ActualBatchCount:" << batch_count << std::endl; 

    return 0;
}
