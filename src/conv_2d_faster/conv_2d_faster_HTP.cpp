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
#include "conv_2d_faster.hpp"
#include <cmath>

int main()
{
    try {
    /*------------------ INPUT ------------------*/
    FilePaths paths;
    std::vector<float> input_img, filter, single_output_template;
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
    size_t local_wg_dim_x, local_wg_dim_y,
           global_work_size_x, global_work_size_y;

    configureLocalMemoryAndWorkers(device, kernel, stride, filter_width, filter_height,
                                   output_width, output_height, local_wg_dim_x, local_wg_dim_y,
                                   global_work_size_x, global_work_size_y);

    cl::NDRange local_dims(local_wg_dim_x, local_wg_dim_y, 1);
    cl::NDRange global_dims(global_work_size_x, global_work_size_y, channel_count);

    // ------------------ high throughput batch pipeline ------------------
    const int batch_count = 4;

    std::vector<cl::Buffer> d_inputs(batch_count);
    std::vector<cl::Buffer> d_outputs(batch_count);
    std::vector<cl::Event> kernel_events(batch_count);
    std::vector<cl::Event> read_events(batch_count);
    std::vector<std::vector<float>> outputs(batch_count);
    cl::Buffer d_filter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filter.size(), filter.data());

    for (int i = 0; i < batch_count; ++i) {
        outputs[i].resize(single_output_template.size());

        d_inputs[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * input_img.size(), const_cast<float*>(input_img.data()));

        d_outputs[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof(float) * outputs[i].size(), nullptr);
    }

    std::cout << "\nBeginning high-throughput conv2d with " << batch_count << " batches\n";

    // ------------------ start kernel execution ------------------
    for (int i = 0; i < batch_count; ++i) {
        kernel.setArg(0, d_inputs[i]);
        kernel.setArg(1, d_filter);
        kernel.setArg(2, d_outputs[i]);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims, nullptr, &kernel_events[i]);
    }

    // Read back all results (overlap with kernel execution)
    for (int i = 0; i < batch_count; ++i) {
        kernel_events[i].wait();
        queue.enqueueReadBuffer(d_outputs[i], CL_FALSE, 0, 
            sizeof(float) * outputs[i].size(), outputs[i].data(), nullptr, &read_events[i]);
    }

    // Wait for all reads to finish
    for (int i = 0; i < batch_count; ++i)
        read_events[i].wait();
    std::cout << "Finished all batches\n";

    // ------------------ start writing res to disk ------------------
    // Write all outputs
    for (int i = 0; i < batch_count; ++i) {
        std::string outPath = paths.outputImagePath;
        if (batch_count > 1) {
            // Add batch index to output filename
            size_t dot = outPath.find_last_of('.');
            if (dot != std::string::npos) {
                outPath.insert(dot, "_batch" + std::to_string(i));
            } else {
                outPath += "_batch" + std::to_string(i);
            }
        }
        writePngImage(outputs[i], output_width, output_height, channel_count, outPath);
        std::cout << "Wrote output: " << outPath << std::endl;
    }

    } catch (cl::Error const &e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    } catch (std::exception const &e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
