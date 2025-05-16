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
#include <string> // Required for std::stoi, std::stof
#include <iomanip> // Required for std::fixed, std::setprecision

int main(int argc, char* argv[])
{
    /*------------------ INPUT ------------------*/
    // Default kernel dimension if not provided
    int kernel_dim_arg = 3; // Default to 3x3 if not specified

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <batch_count> [kernel_dimension]" << std::endl;
        // Outputting in a way Python script can understand failure
        std::cout << "OverallBatchProcessingTime_ms:0.0" << std::endl;
        std::cout << "ActualBatchCount:0" << std::endl;
        std::cout << "KernelSize:" << kernel_dim_arg << std::endl;
        std::cout << "Error:MissingArguments" << std::endl;
        return 1;
    }
    int batch_count = std::stoi(argv[1]);
    if (argc > 2) {
        kernel_dim_arg = std::stoi(argv[2]);
    }
    if (kernel_dim_arg <= 0) {
        std::cerr << "Error: Kernel dimension must be positive." << std::endl;
        std::cout << "OverallBatchProcessingTime_ms:0.0" << std::endl;
        std::cout << "ActualBatchCount:" << batch_count << std::endl;
        std::cout << "KernelSize:" << kernel_dim_arg << std::endl;
        std::cout << "Error:InvalidKernelDimension" << std::endl;
        return 1;
    }


    FilePaths paths;
    std::vector<unsigned char> input_img;
    std::vector<float> filter_original_loaded, single_output_template_original_loaded;
    int input_width, input_height, channel_count,
        filter_width_original_loaded, filter_height_original_loaded, stride,
        output_width_original_loaded, output_height_original_loaded;

    // Call loadInputs - it will also calculate output dimensions and initialize a template output vector
    loadInputs(paths, input_img, input_width, input_height,
                channel_count, filter_original_loaded, filter_width_original_loaded, filter_height_original_loaded,
                stride, output_width_original_loaded, output_height_original_loaded, single_output_template_original_loaded);

    // ---- Override filter and related dimensions based on kernel_dim_arg ----
    int actual_filter_width = kernel_dim_arg;
    int actual_filter_height = kernel_dim_arg;
    std::vector<float> actual_filter(actual_filter_width * actual_filter_height);
    
    if (actual_filter_width * actual_filter_height > 0) {
        float filter_val = 1.0f / (float)(actual_filter_width * actual_filter_height);
        for(size_t i = 0; i < actual_filter.size(); ++i) {
            actual_filter[i] = filter_val;
        }
    }


    // Recalculate output dimensions
    if (input_width < actual_filter_width || input_height < actual_filter_height) {
        std::cerr << "Error: Kernel dimensions (" << actual_filter_width << "x" << actual_filter_height
                  << ") are larger than input dimensions (" << input_width << "x" << input_height << ")." << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "OverallBatchProcessingTime_ms:0.0" << std::endl;
        std::cout << "ActualBatchCount:" << batch_count << std::endl;
        std::cout << "KernelSize:" << kernel_dim_arg << std::endl;
        std::cout << "Error:KernelTooLargeForInput" << std::endl;
        return 0; // Graceful exit for Python script
    }

    int actual_output_width = (input_width - actual_filter_width) / stride + 1;
    int actual_output_height = (input_height - actual_filter_height) / stride + 1;

    if (actual_output_width <= 0 || actual_output_height <= 0) {
        std::cerr << "Error: Calculated output dimensions are non-positive ("
                  << actual_output_width << "x" << actual_output_height
                  << "). Check input/filter dimensions and stride." << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "OverallBatchProcessingTime_ms:0.0" << std::endl;
        std::cout << "ActualBatchCount:" << batch_count << std::endl;
        std::cout << "KernelSize:" << kernel_dim_arg << std::endl;
        std::cout << "Error:InvalidOutputDimensions" << std::endl;
        return 0; // Graceful exit for Python script
    }

    std::vector<float> actual_single_output_template( (size_t)actual_output_width * actual_output_height * channel_count, 0.0f);
    // ---- End of Override ----


    // ------------------ gpu init ------------------
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
    cl::Program program;
    cl::Kernel kernel;

    initializeGPU(context, queue, device, program, kernel,
        stride, input_width, input_height, // input dimensions remain the same
        actual_filter_width, actual_filter_height, // use actual filter dimensions
        channel_count,
        actual_output_width, actual_output_height); // use actual output dimensions
    
    // ------------------ local memory and worker config ------------------
    size_t global_work_size_x, global_work_size_y;
    global_work_size_x = static_cast<size_t>(actual_output_width);
    global_work_size_y = static_cast<size_t>(actual_output_height); 

    cl::NDRange global_dims(global_work_size_x, global_work_size_y, channel_count);

    // ------------------ high throughput batch pipeline ------------------
    // batch_count is already parsed from argv[1]
    // if (argc > 1) batch_count = std::stoi(argv[1]); // This line is now redundant

    // Buffer for the filter
    cl::Buffer d_filter(context, CL_MEM_READ_ONLY, sizeof(float) * actual_filter.size(), nullptr); // Use actual_filter.size()
    
    // Vectors for input/output buffers
    std::vector<cl::Buffer> d_inputs(batch_count);
    std::vector<cl::Buffer> d_outputs(batch_count);
    
    std::vector<std::vector<float>> outputs(batch_count);

    std::cout << "\nBeginning sequential conv2d with " << batch_count << " batches and " 
              << actual_filter_width << "x" << actual_filter_height << " kernel\n"; // Updated message

    auto overall_start_time = std::chrono::high_resolution_clock::now(); // Start timing before VRAM transfers

    // Enqueue write for the filter buffer and wait for it to complete
    cl::Event filter_write_event;
    // Use actual_filter.data() and actual_filter.size()
    queue.enqueueWriteBuffer(d_filter, CL_TRUE, 0, sizeof(float) * actual_filter.size(), actual_filter.data(), nullptr, &filter_write_event);
    filter_write_event.wait(); // Ensure filter is written before any batch processing

    // Prepare output vectors and OpenCL output buffers (can still be done upfront)
    for (int i = 0; i < batch_count; ++i) {
        outputs[i].resize(actual_single_output_template.size()); // Use actual_single_output_template.size()
        d_inputs[i] = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(u_char) * input_img.size(), nullptr); // input_img size is constant, type is unsigned char
        d_outputs[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputs[i].size(), nullptr); // outputs[i] size is now based on actual output
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
    std::cout << std::fixed << std::setprecision(3); // Ensure consistent float output
    std::cout << "OverallBatchProcessingTime_ms:" << overall_duration_ms << std::endl;
    std::cout << "ActualBatchCount:" << batch_count << std::endl; 
    std::cout << "KernelSize:" << kernel_dim_arg << std::endl; // Output the kernel size used

    return 0;
}
