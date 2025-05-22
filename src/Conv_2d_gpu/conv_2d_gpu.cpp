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
#include <direct.h>



void check_exec_and_queue_write(
    cl::CommandQueue& queue, cl::Kernel& kernel, cl::Buffer& d_output_buf,
    int& batch_repeats, std::vector<cl::Event>& exec_events,
    std::vector<cl::Event>& write_events, std::vector<float>& output,
    std::vector<std::chrono::high_resolution_clock::time_point> time_start_write,
    std::vector<std::chrono::high_resolution_clock::time_point> time_end_exec,
    int& last_i_exec, int& last_i_write
) {
    std::chrono::high_resolution_clock::time_point temp_time = 
        std::chrono::high_resolution_clock::now();

    cl_int execution_status;

    for (int j = last_i_exec; j < batch_repeats; ++j) {
        if (!exec_events[j]())
            break;

        exec_events[j].getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &execution_status);
        if (execution_status == CL_COMPLETE) {
            #ifdef DETAILED_TIMINGS
                time_start_write[j] = std::chrono::high_resolution_clock::now();
                time_end_exec[j] = time_start_write[j];
            #endif

            queue.enqueueReadBuffer(d_output_buf[j], CL_FALSE, 0, 
                sizeof(float) * output.size(), output.data(), nullptr, &write_events[j]);
        
            last_i_exec = j + 1;
        }
    }
    
    // Check for write events
    for (int j = last_i_write; j < batch_repeats; ++j) {
        if (!write_events[j]())
            break;

        write_events[j].getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &execution_status);
        if (execution_status == CL_COMPLETE) {
            #ifdef DETAILED_TIMINGS
                time_end_write[j] = std::chrono::high_resolution_clock::now();
            #endif

            queue.enqueueReadBuffer(d_output_buf[j], CL_FALSE, 0, 
                sizeof(float) * output.size(), output.data(), nullptr, &write_events[j]);
        
            last_i_write = j + 1;
        }
    }

    printf("time_start_write: %f\n", 
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - temp_time).count()/1'000'000.0);
}


void run(char* argv[], 
        std::vector<unsigned char>& input_batch, int& input_width, int& input_height, int& channel_count,
        std::vector<float>& filter, int& filter_width, int& filter_height, int& filter_channels, int& stride,
        std::vector<float>& output, int& output_width, int& output_height)
{
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
    size_t global_work_size_x, global_work_size_y, global_work_size_z;
    global_work_size_x = output_width;
    global_work_size_y = output_height;
    global_work_size_z = filter_channels;

    cl::NDRange local_dims = cl::NullRange;
    
    if (strcmp(argv[3], "true") == 0) {
        std::cout << "Using local memory" << std::endl;
        size_t local_wg_dim_x, local_wg_dim_y;
        configureLocalMemoryAndWorkers(
            device, kernel, stride, 
            filter_width, filter_height, filter_channels,
            output_width, output_height, channel_count,
            local_wg_dim_x, local_wg_dim_y, 
            global_work_size_x, global_work_size_y, global_work_size_z
        );
        
        local_dims = cl::NDRange(local_wg_dim_x, local_wg_dim_y);
    } else
        std::cout << "no local memory" << std::endl;

    cl::NDRange global_dims(global_work_size_x, global_work_size_y, global_work_size_z);

    /*------------------ set input/output vector ------------------*/
    int batch_repeats = std::stoi(argv[4]);
    std::vector<cl::Buffer> d_input(batch_repeats), d_output_buf(batch_repeats);
    std::vector<cl::Event> exec_events(batch_repeats), write_events(batch_repeats);

    #define DETAILED_TIMINGS
    #ifdef DETAILED_TIMINGS
        std::vector<std::chrono::high_resolution_clock::time_point> time_start_read(batch_repeats); // also batch start
        std::vector<std::chrono::high_resolution_clock::time_point> time_start_exec(batch_repeats);
        std::vector<std::chrono::high_resolution_clock::time_point> time_start_write(batch_repeats);

        std::vector<std::chrono::high_resolution_clock::time_point> time_end_read(batch_repeats);
        std::vector<std::chrono::high_resolution_clock::time_point> time_end_exec(batch_repeats);
        std::vector<std::chrono::high_resolution_clock::time_point> time_end_write(batch_repeats);
    #endif

    std::chrono::high_resolution_clock::time_point time_overall_start_base = 
        std::chrono::high_resolution_clock::now();

    // ------------------ load filter -------------------------
    cl::Buffer d_filter = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filter.size(), const_cast<float*>(filter.data()));
    kernel.setArg(2, d_filter);

    // ------------------ load and process batches -------------------------
    for (int i = 0; i < batch_repeats; ++i) {
        #ifdef DETAILED_TIMINGS
            time_start_read[i] = std::chrono::high_resolution_clock::now();
        #endif
        d_input[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(unsigned char) * input_batch.size(), 
                        const_cast<unsigned char*>(input_batch.data()));

        #ifdef DETAILED_TIMINGS
            time_end_read[i] = std::chrono::high_resolution_clock::now();
        #endif
        d_output_buf[i] = cl::Buffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * output.size(), nullptr);

        kernel.setArg(0, d_input[i]);
        kernel.setArg(1, d_output_buf[i]);

        #ifdef DETAILED_TIMINGS
            time_start_exec[i] = std::chrono::high_resolution_clock::now();
        #endif
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, 
            global_dims, local_dims, nullptr, &exec_events[i]);

        check_exec_and_queue_write(queue, kernel, d_output_buf[i], batch_repeats, 
            exec_events, write_events, output, time_start_write, time_end_exec, i, i);
    }

    for (int i = 0; i < batch_repeats; ++i) {
        if (!write_events[i]()) {
            #ifdef DETAILED_TIMINGS
                time_start_write[i] = std::chrono::high_resolution_clock::now();
            #endif

            queue.enqueueReadBuffer(d_output_buf[i], CL_FALSE, 0, 
                sizeof(float) * output.size(), output.data(), nullptr, &write_events[j]);
            }
    }

    for (int i = 0; i < batch_repeats; ++i) {
        write_events[i].wait();
        #ifdef DETAILED_TIMINGS
            time_end_write[i] = std::chrono::high_resolution_clock::now();
        #endif
    }

    queue.finish();

    // ------------------ result -------------------------
    // Print timing results for python
    auto ms_cast = [](auto duration) { 
        return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()/1'000'000.0; };
    auto time_diff = ms_cast(std::chrono::high_resolution_clock::now() - time_overall_start_base);
    std::cout << "Overall_time: " << time_diff << std::endl;

    #ifdef DETAILED_TIMINGS
        for (int i = 0; i < batch_repeats; ++i) {
            time_diff = ms_cast(time_start_read[i] - time_overall_start_base);
            std::cout << "Start_read_" << i << ": " << time_diff << std::endl;
            time_diff = ms_cast(time_start_exec[i] - time_overall_start_base);
            std::cout << "Start_exec_" << i << ": " << time_diff << std::endl;
            time_diff = ms_cast(time_start_write[i] - time_overall_start_base);
            std::cout << "Start_write_" << i << ": " << time_diff << std::endl;

            time_diff = ms_cast(time_end_read[i] - time_start_read[i]);
            std::cout << "Read_time_" << i << ": " << time_diff << std::endl;
            time_diff = ms_cast(time_end_exec[i] - time_start_exec[i]);
            std::cout << "Exec_time_" << i << ": " << time_diff << std::endl;
            time_diff = ms_cast(time_end_write[i] - time_start_write[i]);
            std::cout << "Write_time_" << i << ": " << time_diff << std::endl;
        }
    #endif
}

int main(int argc, char* argv[]) {
    char** default_argv = new char*[8];
    default_argv[0] = argv[0];
    default_argv[1] = (char*)"input_lena.png"; // input file (w h img_c)
    default_argv[2] = (char*)
        "kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt"; // kernel file (fw fh img_c fc)
        // "kernel_same_1.txt,kernel_same_2.txt,kernel_same_3.txt";
    default_argv[3] = (char*)"true"; // cpu: threads --- gpu: local mem
    default_argv[4] = (char*)"2"; // batch counts (w h c batch_size)
    default_argv[5] = (char*)"output_gpu.png"; // output file
    default_argv[6] = (char*)"2"; // stride
    default_argv[7] = (char*)"10"; // simulated single batch size 
    return replace_argv(argc, argv, default_argv, run);
}