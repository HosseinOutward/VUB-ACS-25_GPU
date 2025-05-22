#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#define DETAILED_TIMINGS

#include <cl2.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <chrono>

/* ---- project headers (unchanged) ---- */
#include <ocl_utilities.hpp>
#include "conv_2d_gpu.hpp"
#include <direct.h>

/* --------------------------------------------------------------------- */
static inline double ns_to_ms(long long ns)        { return ns / 1'000'000.0; }
static inline double ns_to_ms(cl_ulong  ns)        { return ns / 1'000'000.0; }

/* --------------------------------------------------------------------- */
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

    /* Replace the queue with an **out-of-order, profiling** queue */
    cl_command_queue_properties qprops =
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    queue = cl::CommandQueue(context, device, qprops);

    /* -------------------- work-group sizes ---------------------- */
    std::size_t gws_x = output_width,
                gws_y = output_height,
                gws_z = filter_channels;

    cl::NDRange local_dims = cl::NullRange;

    if (std::strcmp(argv[3], "true") == 0) {
        std::cout << "Using local memory\n";
        std::size_t ldx, ldy;
        configureLocalMemoryAndWorkers(device, kernel, stride,
            filter_width,  filter_height,  filter_channels, 
            output_width,  output_height,  channel_count,
            ldx, ldy, gws_x, gws_y, gws_z);
        local_dims = cl::NDRange(ldx, ldy);
    } else
        std::cout << "no local memory" << std::endl;

    cl::NDRange global_dims(gws_x, gws_y, gws_z);

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

    /* ------------------ human-readable output ------------------- */
    for (int i = 0; i < batch_repeats; ++i)
    {
        // Convert hardware start times to relative times from hardware reference
        auto hw_rel = [&](double hw_start_ms){
            return hw_start_ms - ns_to_ms(hw_ref_ns);
        };

        std::cout << "Start_upload_" << i << ": " << hw_rel(hw_start_upload_ms[i]) << std::endl;
        std::cout << "Start_kernel_" << i << ": " << hw_rel(hw_start_kernel_ms[i]) << std::endl;
        std::cout << "Start_download_" << i << ": " << hw_rel(hw_start_download_ms[i]) << std::endl;

        std::cout << "Upload_time_" << i << ": " << upload_ms[i] << std::endl;
        std::cout << "Kernel_time_" << i << ": " << kernel_ms[i] << std::endl;
        std::cout << "Download_time_" << i << ": " << download_ms[i] << std::endl;
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