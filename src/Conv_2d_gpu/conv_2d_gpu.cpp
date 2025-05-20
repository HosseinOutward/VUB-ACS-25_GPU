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

std::vector<std::chrono::microseconds> time_per_batch(batch_repeats);
std::vector<std::chrono::microseconds> time_per_read(batch_repeats);
std::vector<std::chrono::microseconds> time_per_exec(batch_repeats);
std::vector<std::chrono::microseconds> time_per_write(batch_repeats);

std::vector<std::chrono::microseconds> time_start_read(batch_repeats);
std::vector<std::chrono::microseconds> time_start_exec(batch_repeats);
std::vector<std::chrono::microseconds> time_start_write(batch_repeats);

auto time_overall_start = std::chrono::high_resolution_clock::now();

    cl::Buffer d_filter = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filter.size(), const_cast<float*>(filter.data()));
    kernel.setArg(2, d_filter);

    // time for each each run

    for (int i = 0; i < batch_repeats; ++i) {
auto run_start = std::chrono::high_resolution_clock::now();
auto read_start = std::chrono::high_resolution_clock::now();
time_start_read[i] = std::chrono::duration_cast<std::chrono::microseconds>(read_start-time_overall_start);

        d_input[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(unsigned char) * input_batch.size(), 
                        const_cast<unsigned char*>(input_batch.data()));

auto read_end = std::chrono::high_resolution_clock::now();
time_per_read[i] = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start);

        d_output_buf[i] = cl::Buffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * output.size(), nullptr);

        kernel.setArg(0, d_input[i]);
        kernel.setArg(1, d_output_buf[i]);

        // enqueue the kernel
auto exec_start = std::chrono::high_resolution_clock::now();
time_start_exec[i] = std::chrono::duration_cast<std::chrono::microseconds>(exec_start - time_overall_start);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims);

auto exec_end = std::chrono::high_resolution_clock::now();
time_per_exec[i] = std::chrono::duration_cast<std::chrono::microseconds>(exec_end - exec_start);

auto run_end = std::chrono::high_resolution_clock::now();
time_per_batch[i] = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start);
    }

    queue.finish();

    for (int i = 0; i < batch_repeats; ++i) {
auto write_start = std::chrono::high_resolution_clock::now();
time_start_write[i] = std::chrono::duration_cast<std::chrono::microseconds>(write_start-time_overall_start);

        queue.enqueueReadBuffer(d_output_buf[i], CL_FALSE, 0, 
            sizeof(float) * output.size(), output.data());

auto write_end = std::chrono::high_resolution_clock::now();
time_per_write[i] = std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start);
    }

    queue.finish();

auto time_overall_end = std::chrono::high_resolution_clock::now();
auto time_overall = std::chrono::duration_cast<std::chrono::microseconds>(time_overall_end - time_overall_start);

    // ------------------ result -------------------------
    // Print timing results for python
std::cout << "Overall_time: " << time_overall.count() << std::endl;
for (int i = 0; i < batch_repeats; ++i) {
std::cout << "Batch_time_" << i << ": " << time_per_batch[i].count() << std::endl;
std::cout << "Read_time_" << i << ": " << time_per_read[i].count() << std::endl;
std::cout << "Exec_time_" << i << ": " << time_per_exec[i].count() << std::endl;
std::cout << "Write_time_" << i << ": " << time_per_write[i].count() << std::endl;

std::cout << "Start_read_" << i << ": " << time_start_read[i].count() << std::endl;
std::cout << "Start_exec_" << i << ": " << time_start_exec[i].count() << std::endl;
std::cout << "Start_write_" << i << ": " << time_start_exec[i].count() << std::endl;
}

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