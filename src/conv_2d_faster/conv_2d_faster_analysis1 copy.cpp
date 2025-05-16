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
    int batch_count = 50;
    if (argc > 1) batch_count = std::stoi(argv[1]);

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

    // New vectors for overall batch times (enqueue to wait completion)
    std::vector<std::chrono::high_resolution_clock::time_point> batch_input_write_overall_starts(batch_count);
    std::vector<double> batch_input_write_overall_times_ms(batch_count);
    std::vector<std::chrono::high_resolution_clock::time_point> batch_kernel_overall_starts(batch_count);
    std::vector<double> batch_kernel_overall_times_ms(batch_count);
    std::vector<std::chrono::high_resolution_clock::time_point> batch_read_overall_starts(batch_count);
    std::vector<double> batch_read_overall_times_ms(batch_count);

    // New vectors for chrono end times and recording flags
    std::vector<std::chrono::high_resolution_clock::time_point> batch_input_write_overall_ends(batch_count);
    std::vector<bool> batch_input_write_chrono_recorded(batch_count, false);
    std::vector<std::chrono::high_resolution_clock::time_point> batch_kernel_overall_ends(batch_count);
    std::vector<bool> batch_kernel_chrono_recorded(batch_count, false);
    std::vector<std::chrono::high_resolution_clock::time_point> batch_read_overall_ends(batch_count);
    std::vector<bool> batch_read_chrono_recorded(batch_count, false);

    std::cout << "\nBeginning high-throughput conv2d with " << batch_count << " batches\n";

    // IMPORTANT: Ensure your cl::CommandQueue (queue) was created with CL_QUEUE_PROFILING_ENABLE for event timing.
    auto overall_start_time = std::chrono::high_resolution_clock::now(); // Start timing before VRAM transfers

    // Enqueue write for the filter buffer
    queue.enqueueWriteBuffer(d_filter, CL_FALSE, 0, sizeof(cl_half) * filter.size(), filter.data(), nullptr, &filter_write_event);

    // Prepare output vectors and OpenCL output buffers
    for (int i = 0; i < batch_count; ++i) {
        outputs[i].resize(single_output_template.size());
        // Create input and output buffers for each batch upfront
        d_inputs[i] = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_half) * input_img.size(), nullptr);
        d_outputs[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_half) * outputs[i].size(), nullptr);
    }

    // Main enqueue loop: Write, Kernel, Read for each batch
    std::cout << "Enqueueing tasks for " << batch_count << " batches..." << std::endl;

    auto get_event_status_string = [](cl::Event& ev) -> std::string {
        cl_int status_code;
        cl_int err = ev.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status_code);
        if (err != CL_SUCCESS) {
            return "Error(" + std::to_string(err) + ")";
        }
        switch (status_code) {
            case CL_QUEUED: return "Queued";
            case CL_SUBMITTED: return "Submitted";
            case CL_RUNNING: return "Running";
            case CL_COMPLETE: return "Complete";
            default: return "Unknown(" + std::to_string(status_code) + ")";
        }
    };

    for (int i = 0; i < batch_count; ++i) {
        // 1. Enqueue Input Write
        batch_input_write_overall_starts[i] = std::chrono::high_resolution_clock::now();
        queue.enqueueWriteBuffer(d_inputs[i], CL_FALSE, 0, sizeof(cl_half) * input_img.size(),
                                 const_cast<cl_half*>(input_img.data()), nullptr, &input_write_events[i]);

        // 2. Enqueue Kernel
        kernel.setArg(0, d_inputs[i]);
        kernel.setArg(1, d_filter);
        kernel.setArg(2, d_outputs[i]);
        std::vector<cl::Event> kernel_dependencies;
        kernel_dependencies.push_back(input_write_events[i]);
        kernel_dependencies.push_back(filter_write_event); // All kernels depend on the filter being written
        batch_kernel_overall_starts[i] = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims, &kernel_dependencies, &kernel_events[i]);

        // 3. Enqueue Output Read
        std::vector<cl::Event> read_dependencies;
        read_dependencies.push_back(kernel_events[i]); // Read depends on its corresponding kernel
        batch_read_overall_starts[i] = std::chrono::high_resolution_clock::now();
        queue.enqueueReadBuffer(d_outputs[i], CL_FALSE, 0, sizeof(cl_half) * outputs[i].size(),
                                outputs[i].data(), &read_dependencies, &read_events[i]);

        // Periodic status check (non-blocking)
        // Checks status every 5 batches, and also after the last batch is enqueued.
        if ((i + 1) % 5 == 0 || (i + 1) == batch_count) {
            std::cout << "\n--- Status Check after enqueuing Batch " << i << " ---" << std::endl;
            for (int k = 0; k <= i; k += std::max(1, batch_count / 10)) { // Check roughly 10 spread out samples up to current
                cl_int status_code_val; // Renamed to avoid conflict with the function's status_code
                std::string iw_s, k_s, r_s;

                // Input Write Event
                input_write_events[k].getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status_code_val);
                if (!batch_input_write_chrono_recorded[k] && status_code_val == CL_COMPLETE) {
                    batch_input_write_overall_ends[k] = std::chrono::high_resolution_clock::now();
                    batch_input_write_overall_times_ms[k] = std::chrono::duration<double, std::milli>(
                        batch_input_write_overall_ends[k] - batch_input_write_overall_starts[k]).count();
                    batch_input_write_chrono_recorded[k] = true;
                }
                iw_s = get_event_status_string(input_write_events[k]);
                if (batch_input_write_chrono_recorded[k] && status_code_val == CL_COMPLETE) iw_s += " (Chrono End Recorded)";

                // Kernel Event
                kernel_events[k].getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status_code_val);
                if (!batch_kernel_chrono_recorded[k] && status_code_val == CL_COMPLETE) {
                    batch_kernel_overall_ends[k] = std::chrono::high_resolution_clock::now();
                    batch_kernel_overall_times_ms[k] = std::chrono::duration<double, std::milli>(
                        batch_kernel_overall_ends[k] - batch_kernel_overall_starts[k]).count();
                    batch_kernel_chrono_recorded[k] = true;
                }
                k_s = get_event_status_string(kernel_events[k]);
                if (batch_kernel_chrono_recorded[k] && status_code_val == CL_COMPLETE) k_s += " (Chrono End Recorded)";
                
                // Read Event
                read_events[k].getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status_code_val);
                if (!batch_read_chrono_recorded[k] && status_code_val == CL_COMPLETE) {
                    batch_read_overall_ends[k] = std::chrono::high_resolution_clock::now();
                    batch_read_overall_times_ms[k] = std::chrono::duration<double, std::milli>(
                        batch_read_overall_ends[k] - batch_read_overall_starts[k]).count();
                    batch_read_chrono_recorded[k] = true;
                }
                r_s = get_event_status_string(read_events[k]);
                if (batch_read_chrono_recorded[k] && status_code_val == CL_COMPLETE) r_s += " (Chrono End Recorded)";

                std::cout << "  Batch " << k << ": Write: " << iw_s
                          << ", Kernel: " << k_s
                          << ", Read: " << r_s << std::endl;
            }
            std::cout << "--- End Status Check ---" << std::endl;
        }
    }
    std::cout << "All tasks enqueued." << std::endl;

    // Wait for filter write to complete and profile it
    filter_write_event.wait();
    cl_ulong queued_fw, start_fw, end_fw;
    filter_write_event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued_fw);
    filter_write_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_fw);
    filter_write_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_fw);
    filter_write_queue_duration_ns = start_fw - queued_fw;
    filter_write_exec_duration_ns = end_fw - start_fw;

    // Wait for input writes to complete and profile them
    std::cout << "Profiling input writes..." << std::endl;
    for (int i = 0; i < batch_count; ++i) {
        input_write_events[i].wait();
        if (!batch_input_write_chrono_recorded[i]) { // If not recorded by periodic check
            batch_input_write_overall_ends[i] = std::chrono::high_resolution_clock::now();
            batch_input_write_overall_times_ms[i] = std::chrono::duration<double, std::milli>(
                batch_input_write_overall_ends[i] - batch_input_write_overall_starts[i]).count();
            // batch_input_write_chrono_recorded[i] = true; // Mark as recorded
        }
        // auto batch_input_write_overall_end = std::chrono::high_resolution_clock::now(); // Stop input write overall timer // OLD
        // batch_input_write_overall_times_ms[i] = std::chrono::duration<double, std::milli>( // OLD
        //     batch_input_write_overall_end - batch_input_write_overall_starts[i]).count(); // OLD

        cl_ulong queued_iw, start_iw, end_iw;
        input_write_events[i].getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued_iw);
        input_write_events[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &start_iw);
        input_write_events[i].getProfilingInfo(CL_PROFILING_COMMAND_END, &end_iw);
        input_write_queue_durations_ns[i] = start_iw - queued_iw;
        input_write_exec_durations_ns[i] = end_iw - start_iw;
    }

    // Wait for kernels to complete and profile them
    std::cout << "Profiling kernel executions..." << std::endl;
    for (int i = 0; i < batch_count; ++i) {
        kernel_events[i].wait(); // Wait for kernel i to complete
        if (!batch_kernel_chrono_recorded[i]) {
            batch_kernel_overall_ends[i] = std::chrono::high_resolution_clock::now();
            batch_kernel_overall_times_ms[i] = std::chrono::duration<double, std::milli>(
                batch_kernel_overall_ends[i] - batch_kernel_overall_starts[i]).count();
        }
        // auto batch_kernel_overall_end = std::chrono::high_resolution_clock::now(); // Stop kernel overall timer // OLD
        // batch_kernel_overall_times_ms[i] = std::chrono::duration<double, std::milli>( // OLD
        //     batch_kernel_overall_end - batch_kernel_overall_starts[i]).count(); // OLD
        
        // Profile kernel event
        cl_ulong queued_k, start_k, end_k;
        kernel_events[i].getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued_k);
        kernel_events[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &start_k);
        kernel_events[i].getProfilingInfo(CL_PROFILING_COMMAND_END, &end_k);
        kernel_queue_durations_ns[i] = start_k - queued_k;
        kernel_exec_durations_ns[i] = end_k - start_k;
    }
    
    // Wait for reads to complete and profile them
    std::cout << "Profiling output reads..." << std::endl;
    for (int i = 0; i < batch_count; ++i) {
        read_events[i].wait(); // Wait for read i to complete
        if (!batch_read_chrono_recorded[i]) {
            batch_read_overall_ends[i] = std::chrono::high_resolution_clock::now();
            batch_read_overall_times_ms[i] = std::chrono::duration<double, std::milli>(
                batch_read_overall_ends[i] - batch_read_overall_starts[i]).count();
        }
        // auto batch_read_overall_end = std::chrono::high_resolution_clock::now(); // Stop read overall timer // OLD
        // batch_read_overall_times_ms[i] = std::chrono::duration<double, std::milli>( // OLD
        //     batch_read_overall_end - batch_read_overall_starts[i]).count(); // OLD

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
        std::cout << "Batch_" << i << "_InputWriteOverallTime_ms:" << batch_input_write_overall_times_ms[i] << std::endl; // New
        std::cout << "Batch_" << i << "_KernelQueueTime_ns:" << kernel_queue_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_KernelExecTime_ns:" << kernel_exec_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_KernelOverallTime_ms:" << batch_kernel_overall_times_ms[i] << std::endl; // New
        std::cout << "Batch_" << i << "_ReadQueueTime_ns:" << read_queue_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_ReadExecTime_ns:" << read_exec_durations_ns[i] << std::endl;
        std::cout << "Batch_" << i << "_ReadOverallTime_ms:" << batch_read_overall_times_ms[i] << std::endl; // New
    }
    std::cout << "OverallBatchProcessingTime_ms:" << overall_duration_ms << std::endl;
    std::cout << "ActualBatchCount:" << batch_count << std::endl; // C++ knows its actual batch count
    // ------------------ done -------------------------

    return 0;
}
