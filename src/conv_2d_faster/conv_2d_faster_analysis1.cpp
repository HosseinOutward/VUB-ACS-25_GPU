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
    std::vector<cl_half> input_img, filter, single_output_template;
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
    int batch_count = 4;
    if (argc > 1) batch_count = std::stoi(argv[1]);
    else batch_count = 4;

    // Buffer for the filter and its write event
    cl::Buffer d_filter(context, CL_MEM_READ_ONLY, sizeof(cl_half) * filter.size(), nullptr);
    cl::Event filter_write_event;
    cl_ulong filter_write_queue_duration_ns;
    cl_ulong filter_write_exec_duration_ns;

    // Vectors for input/output buffers and events
    std::vector<cl::Buffer> d_inputs(batch_count);
    std::vector<cl::Buffer> d_outputs(batch_count);

    std::vector<cl::Event> input_write_events(batch_count);
    std::vector<cl::Event> kernel_events(batch_count);
    std::vector<cl::Event> read_events(batch_count);
    
    std::vector<std::vector<cl_half>> outputs(batch_count);

    // Vectors to store durations
    std::vector<cl_ulong> input_write_queue_durations_ns(batch_count);
    std::vector<cl_ulong> input_write_exec_durations_ns(batch_count);
    std::vector<cl_ulong> kernel_queue_durations_ns(batch_count);
    std::vector<cl_ulong> kernel_exec_durations_ns(batch_count);
    std::vector<cl_ulong> read_queue_durations_ns(batch_count);
    std::vector<cl_ulong> read_exec_durations_ns(batch_count);

    std::cout << "\nBeginning high-throughput conv2d with " << batch_count << " batches\n";

    // IMPORTANT: Ensure your cl::CommandQueue (queue) was created with CL_QUEUE_PROFILING_ENABLE for event timing.
    auto overall_start_time = std::chrono::high_resolution_clock::now(); // Start timing before VRAM transfers

    // Enqueue write for the filter buffer
    queue.enqueueWriteBuffer(d_filter, CL_FALSE, 0, sizeof(cl_half) * filter.size(), filter.data(), nullptr, &filter_write_event);

    // Create buffers and enqueue writes for inputs
    for (int i = 0; i < batch_count; ++i)
        outputs[i].resize(single_output_template.size());

    for (int i = 0; i < batch_count; ++i) {
        d_inputs[i] = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_half) * input_img.size(), nullptr);
        queue.enqueueWriteBuffer(d_inputs[i], CL_FALSE, 0, sizeof(cl_half) * input_img.size(), 
                const_cast<cl_half*>(input_img.data()), nullptr, &input_write_events[i]);
        d_outputs[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_half) * outputs[i].size(), nullptr);
    }

    // ------------------ start kernel execution ------------------
    for (int i = 0; i < batch_count; ++i) {
        kernel.setArg(0, d_inputs[i]);
        kernel.setArg(1, d_filter);
        kernel.setArg(2, d_outputs[i]);
        std::vector<cl::Event> kernel_dependencies;
        kernel_dependencies.push_back(input_write_events[i]);
        kernel_dependencies.push_back(filter_write_event); // All kernels depend on the filter being written
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims, &kernel_dependencies, &kernel_events[i]);
    }

    // Wait for filter write to complete and profile it
    filter_write_event.wait();
    cl_ulong queued_fw, start_fw, end_fw;
    filter_write_event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued_fw);
    filter_write_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_fw);
    filter_write_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_fw);
    filter_write_queue_duration_ns = start_fw - queued_fw;
    filter_write_exec_duration_ns = end_fw - start_fw;

    // Wait for input writes to complete and profile them
    for (int i = 0; i < batch_count; ++i) {
        input_write_events[i].wait();
        cl_ulong queued_iw, start_iw, end_iw;
        input_write_events[i].getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued_iw);
        input_write_events[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &start_iw);
        input_write_events[i].getProfilingInfo(CL_PROFILING_COMMAND_END, &end_iw);
        input_write_queue_durations_ns[i] = start_iw - queued_iw;
        input_write_exec_durations_ns[i] = end_iw - start_iw;
    }

    // Read back all results (overlap with kernel execution)
    for (int i = 0; i < batch_count; ++i) {
        kernel_events[i].wait(); // Wait for kernel i to complete
        // Profile kernel event
        cl_ulong queued_k, start_k, end_k;
        kernel_events[i].getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued_k);
        kernel_events[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &start_k);
        kernel_events[i].getProfilingInfo(CL_PROFILING_COMMAND_END, &end_k);
        kernel_queue_durations_ns[i] = start_k - queued_k;
        kernel_exec_durations_ns[i] = end_k - start_k;
        
        queue.enqueueReadBuffer(d_outputs[i], CL_FALSE, 0, 
            sizeof(cl_half) * outputs[i].size(), outputs[i].data(), nullptr, &read_events[i]);
    }

    // Wait for all reads to finish
    for (int i = 0; i < batch_count; ++i) {
        read_events[i].wait(); // Wait for read i to complete
        // Profile read event
        cl_ulong queued_r, start_r, end_r;
        read_events[i].getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued_r);
        read_events[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &start_r);
        read_events[i].getProfilingInfo(CL_PROFILING_COMMAND_END, &end_r);
        read_queue_durations_ns[i] = start_r - queued_r;
        read_exec_durations_ns[i] = end_r - start_r;
    }
    
    auto overall_end_time = std::chrono::high_resolution_clock::now();
    auto overall_duration_ms = std::chrono::duration<double, std::milli>(
                overall_end_time - overall_start_time).count();

    // Print all timing information
    // DeviceProfilingTimerResolution_ns is printed earlier, after initializeGPU()
    std::cout << "FilterWriteQueueTime_ns:" << filter_write_queue_duration_ns << std::endl;
    std::cout << "FilterWriteExecTime_ns:" << filter_write_exec_duration_ns << std::endl;
    for (int i = 0; i < batch_count; ++i) {
        std::cout << "Batch_" << i << "_InputWriteQueueTime_ns:" << input_write_queue_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_InputWriteExecTime_ns:" << input_write_exec_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_KernelQueueTime_ns:" << kernel_queue_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_KernelExecTime_ns:" << kernel_exec_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_ReadQueueTime_ns:" << read_queue_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_ReadExecTime_ns:" << read_exec_durations_ns[i] << std::endl;
    }
    std::cout << "OverallBatchProcessingTime_ms:" << overall_duration_ms << std::endl;
    std::cout << "ActualBatchCount:" << batch_count << std::endl; // C++ knows its actual batch count
    // ------------------ done -------------------------

    return 0;
}
