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

/* ---- project headers ---- */
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
        stride, input_width, input_height, channel_count, 
        filter_width, filter_height, filter_channels, output_width, output_height);

    /* IMPORTANT PART - Create separate queues for maximum parallelism */
    cl_command_queue_properties qprops =
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    
    // Separate queues for different operations to enable true parallelism
    cl::CommandQueue compute_queue(context, device, qprops);  // For kernel execution
    cl::CommandQueue transfer_queue(context, device, qprops); // For data transfers (DMA)

    /* -------------------- work-group sizes ---------------------- */
    std::size_t gws_x = output_width,
                gws_y = output_height;

    cl::NDRange local_dims = cl::NullRange;

    if (std::strcmp(argv[3], "true") == 0) {
        std::cout << "Using local memory\n";
        std::size_t ldx, ldy;
        configureLocalMemoryAndWorkers(device, kernel, stride,
            filter_width,  filter_height,  filter_channels, 
            output_width,  output_height,  channel_count,
            ldx, ldy, gws_x, gws_y);
        local_dims = cl::NDRange(ldx, ldy, filter_channels);
    } else
        std::cout << "no local memory" << std::endl;

    int b_size = std::stoi(argv[7]);
    cl::NDRange global_dims(gws_x, gws_y, filter_channels*b_size);

    /* ------------------ constants / filter upload --------------- */
    cl::Buffer d_filter(context,
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filter.size(),
                        const_cast<float*>(filter.data()));
    kernel.setArg(2, d_filter);

    /* ------------------ batch buffers & events ------------------ */
    const int batch_repeats = std::stoi(argv[4]);

    std::vector<cl::Buffer> d_input(batch_repeats);
    std::vector<cl::Buffer> d_output(batch_repeats);

    std::vector<cl::Event> ev_upload(batch_repeats);
    std::vector<cl::Event> ev_kernel(batch_repeats);
    std::vector<cl::Event> ev_download(batch_repeats);

#ifdef DETAILED_TIMINGS
    std::vector<double> hw_start_upload_ms(batch_repeats);
    std::vector<double> hw_start_kernel_ms(batch_repeats);
    std::vector<double> hw_start_download_ms(batch_repeats);

    std::vector<double> upload_ms(batch_repeats);
    std::vector<double> kernel_ms(batch_repeats);
    std::vector<double> download_ms(batch_repeats);
#endif

    /* ------------------ main loop: enqueue ---------------------- */
    const std::size_t in_bytes = sizeof(unsigned char) * input_batch.size();
    const std::size_t out_bytes = sizeof(float) * output.size();

    const auto overall_start = std::chrono::high_resolution_clock::now();

    // Pre-allocate all buffers to avoid allocation overhead
    for (int i = 0; i < batch_repeats; ++i) {
        d_input[i] = cl::Buffer(context, CL_MEM_READ_ONLY,  in_bytes);
        d_output[i]= cl::Buffer(context, CL_MEM_READ_WRITE, out_bytes);
    }

    for (int i = 0; i < batch_repeats; ++i) {
        // 1) Upload data using transfer queue (async)
        transfer_queue.enqueueWriteBuffer(d_input[i], CL_FALSE, 0, in_bytes,
                                         input_batch.data(),
                                         nullptr, &ev_upload[i]);
        
        // 2) Execute kernel using compute queue (waits for upload)
        kernel.setArg(0, d_input[i]);
        kernel.setArg(1, d_output[i]);
        
        std::vector<cl::Event> upload_deps = {ev_upload[i]};
        compute_queue.enqueueNDRangeKernel(kernel,
                                          cl::NullRange,
                                          global_dims, local_dims,
                                          &upload_deps, &ev_kernel[i]);
        
        // 3) Download previous result using transfer queue
        if (i > 0) {
            std::vector<cl::Event> kernel_deps = {ev_kernel[i-1]};
            transfer_queue.enqueueReadBuffer(d_output[i-1], CL_FALSE, 0, out_bytes,
                                            output.data(),
                                            &kernel_deps, &ev_download[i-1]);
        }
    }
    
    // Download the last batch result
    if (batch_repeats > 0) {
        std::vector<cl::Event> kernel_deps = {ev_kernel[batch_repeats-1]};
        transfer_queue.enqueueReadBuffer(d_output[batch_repeats-1], CL_FALSE, 0, out_bytes,
                                        output.data(),
                                        &kernel_deps, &ev_download[batch_repeats-1]);
    }    /* ---------- drain both pipelines ---------- */
    cl::Event::waitForEvents(ev_download);
    transfer_queue.finish();
    compute_queue.finish();

    /* ------------------ collect timings ---------------- */
    const auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Overall_time: "
              << ns_to_ms(std::chrono::duration_cast<
                    std::chrono::nanoseconds>(t_end - overall_start).count()) << std::endl;

#ifdef DETAILED_TIMINGS
    for (int i = 0; i < batch_repeats; ++i) {
        cl_ulong ns0, ns1;

        /* upload (write buffer) */
        ev_upload[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &ns0);
        ev_upload[i].getProfilingInfo(CL_PROFILING_COMMAND_END,   &ns1);
        hw_start_upload_ms[i] = ns_to_ms(ns0);
        upload_ms[i]  = ns_to_ms(ns1 - ns0);

        /* kernel */
        ev_kernel[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &ns0);
        ev_kernel[i].getProfilingInfo(CL_PROFILING_COMMAND_END,   &ns1);
        hw_start_kernel_ms[i] = ns_to_ms(ns0);
        kernel_ms[i]  = ns_to_ms(ns1 - ns0);

        /* download (read buffer) */
        ev_download[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &ns0);
        ev_download[i].getProfilingInfo(CL_PROFILING_COMMAND_END,   &ns1);
        hw_start_download_ms[i] = ns_to_ms(ns0);
        download_ms[i]= ns_to_ms(ns1 - ns0);
    }

    /* ------------------ human-readable output ------------------- */
    // calculate relative times
    cl_ulong t0;
    ev_upload[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &t0);
    auto hw_rel = [&](double hw_start_ms){ return hw_start_ms - ns_to_ms(t0); };

    for (int i = 0; i < batch_repeats; ++i) {
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
    default_argv[1] = (char*)
        "30,30,3"; // kernel file input file (w h img_c)
        // "input_lena.png";
    default_argv[2] = (char*)
        "3,3,4"; // kernel file (fw fh <not given -> img_c> fc)
        // "kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt";
        // "kernel_same_1.txt,kernel_same_2.txt,kernel_same_3.txt";
    default_argv[3] = (char*)"true"; // cpu: threads --- gpu: local mem
    default_argv[4] = (char*)"2"; // batch counts (w h c batch_size)
    default_argv[5] = (char*)"output_gpu.png"; // output file
    default_argv[6] = (char*)"2"; // stride
    default_argv[7] = (char*)"5"; // simulated single batch size 
    
    try {
        if (argc == 1) 
            prep_and_run(default_argv, run);
        else
            prep_and_run(argv, run);
    } 
    // ------------------ try catch -------------------------
    catch (cl::Error const &e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1; // Indicate an error
    } catch (std::exception const &e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}