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
#include <cmath>
#include "conv_2d_faster.hpp"

int main()
{
    try {
    /*------------------ INPUT ------------------*/
    FilePaths paths;
    std::vector<cl_half> input_img, filter, output;
    int input_width, input_height, channel_count, 
        filter_width, filter_height, stride, output_width, output_height;

    loadInputs(paths, input_img, input_width, input_height, 
                channel_count, filter, filter_width, filter_height,
                stride, output_width, output_height, output);

    /*------------------ gpu init ------------------*/
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
    cl::Program program;
    cl::Kernel kernel;

    initializeGPU(context, queue, device, program, kernel, 
        stride, input_width, input_height, filter_width, filter_height, 
        channel_count, output_width, output_height);

    /*------------------ local memory info ------------------*/
    size_t local_wg_dim_x, local_wg_dim_y,
           global_work_size_x, global_work_size_y;
    configureLocalMemoryAndWorkers(device, kernel, stride, filter_width, filter_height, 
                output_width, output_height, local_wg_dim_x, local_wg_dim_y, 
                global_work_size_x, global_work_size_y);

    /*------------------ set input/output vector ------------------*/
    cl::Buffer d_input, d_filter, d_output_buf;
    d_input = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(cl_half) * input_img.size(), const_cast<cl_half*>(input_img.data()));
    d_filter = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(cl_half) * filter.size(), const_cast<cl_half*>(filter.data()));
    d_output_buf = cl::Buffer(context, CL_MEM_READ_WRITE,
                        sizeof(cl_half) * output.size(), nullptr);
    kernel.setArg(0, d_input);
    kernel.setArg(1, d_filter);
    kernel.setArg(2, d_output_buf);
    
    // ------------ global and local work size
    cl::NDRange local_dims(local_wg_dim_x, local_wg_dim_y, 1); // worker groups
    cl::NDRange global_dims(global_work_size_x, global_work_size_y, channel_count);

    // ------------------ run gpu convolution ------------------
    std::cout << "\nbeginning conv2d fast\n";

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims);
    
    std::cout << "finished conv2d fast\n";
    
    queue.enqueueReadBuffer(d_output_buf, CL_TRUE, 0, sizeof(cl_half) * output.size(), output.data());
    std::cout << "finished reading output off vram\n";

    // ------------------ result -------------------------
    // Before writing to PNG, convert cl_half output back to float if writePngImage expects float
    std::vector<float> float_output(output.size());
    std::transform(output.begin(), output.end(), float_output.begin(), [](cl_half val) { return static_cast<float>(val); });
    writePngImage(float_output, output_width, output_height, channel_count, paths.outputImagePath);

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
