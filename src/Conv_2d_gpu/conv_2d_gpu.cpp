#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#define DETAILED_TIMINGS
#define PINNED_MEMORY
#define FULL_DUPLEX

#include <cl2.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <direct.h>

/* ---- project headers ---- */
#include "utilities.hpp"
#include "conv_2d_gpu.hpp"

/* --------------------------------------------------------------------- */
static inline double ns_to_ms(cl_ulong  ns)        { return ns / 1'000'000.0; }

/* --------------------------------------------------------------------- */
void run(char* argv[],
        std::vector<unsigned char>& input_batch, int& input_width, int& input_height, int& channel_count,
        std::vector<float>& filter, int& filter_width, int& filter_height, int& filter_channels, int& stride,
        std::vector<unsigned char>& output, int& output_width, int& output_height)
{
    /*------------------ gpu init ------------------*/
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
    cl::Program program;
    cl::Kernel kernel;
    
    /* IMPORTANT PART - Create separate queues for maximum parallelism */
    initializeGPU(context, queue, device, program, kernel, (strcmp(argv[3], "true") == 0),
        stride, input_width, input_height, channel_count, 
        filter_width, filter_height, filter_channels, output_width, output_height);
    
    // Check if device supports out-of-order execution for full-duplex
    cl_command_queue_properties qprops = CL_QUEUE_PROFILING_ENABLE;
    qprops |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    
    // Three separate queues for true full duplex operation
    cl::CommandQueue compute_queue(context, device, qprops);   // For kernel execution
    cl::CommandQueue upload_queue(context, device, qprops);    // For data uploads (DMA in)
    cl::CommandQueue download_queue(context, device, qprops);  // For data downloads (DMA out)

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

    const std::size_t in_bytes = sizeof(unsigned char) * input_batch.size();
    const std::size_t out_bytes = sizeof(unsigned char) * output.size();

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

int out_buffer_count;
#ifdef PINNED_MEMORY
    out_buffer_count = 2;
#else
    out_buffer_count = batch_repeats;
#endif

std::vector<unsigned char*> mapped_pinned_input_pointers(batch_repeats);
std::vector<unsigned char*> mapped_pinned_output_pointers(out_buffer_count);

#ifdef PINNED_MEMORY
    /* ------------------ pinned memory buffers ------------------ */
    std::vector<cl::Buffer> pinned_input_host_buffers(batch_repeats);
    std::vector<cl::Buffer> pinned_output_host_buffers(out_buffer_count);
    // simulate as if pinned memory was used when loading input data
    for (int i = 0; i < batch_repeats; ++i) {
        pinned_input_host_buffers[i] = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY, in_bytes);
        mapped_pinned_input_pointers[i] = (unsigned char*)queue.enqueueMapBuffer(
            pinned_input_host_buffers[i], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, in_bytes);

        std::copy(input_batch.data(), input_batch.data() + input_batch.size(), mapped_pinned_input_pointers[i]);
    }
#else
    for (int i = 0; i < batch_repeats; ++i)
        mapped_pinned_input_pointers[i] = input_batch.data();
    for (int i = 0; i < out_buffer_count; ++i)
        mapped_pinned_output_pointers[i] = output.data();
#endif

    /* ------------------ main loop: enqueue ---------------------- */
    // Get device timestamp at start
    // const auto overall_start = std::chrono::high_resolution_clock::now();
    cl::UserEvent overall_start_event(context);
    overall_start_event.setStatus(CL_COMPLETE);
    queue.enqueueMarkerWithWaitList(nullptr, &overall_start_event);
    overall_start_event.wait();
    const auto overall_start = overall_start_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    std::vector<cl::Event> last_ev_download(out_buffer_count);
    for (int i = 0; i < out_buffer_count; ++i) {
        cl::UserEvent ue(context);
        ue.setStatus(CL_COMPLETE);
        last_ev_download[i] = ue;
    }

#ifdef PINNED_MEMORY
    // double buffering for output
    for (int i = 0; i < out_buffer_count; ++i) {
        pinned_output_host_buffers[i] = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_READ_ONLY, out_bytes);        mapped_pinned_output_pointers[i] = (unsigned char*)queue.enqueueMapBuffer(
        pinned_output_host_buffers[i], CL_TRUE, CL_MAP_READ, 0, out_bytes);
    }
    queue.finish();
#endif

    // warmup the queues
    d_input[0] = cl::Buffer(context, CL_MEM_READ_ONLY,  in_bytes);
    upload_queue.enqueueWriteBuffer(d_input[0], CL_FALSE, 0, in_bytes,
                                   mapped_pinned_input_pointers[0],
                                   nullptr, &ev_upload[0]);

    // main loop - (user's new order: kernel, download, upload)
    for (int i = 0; i <= batch_repeats; ++i) {
        if (i < batch_repeats) {
            // run kernel on current input
            d_output[i]= cl::Buffer(context, CL_MEM_READ_WRITE, out_bytes);
            kernel.setArg(0, d_input[i]);
            kernel.setArg(1, d_output[i]);
            std::vector<cl::Event> upload_deps_loop = {ev_upload[i]};
            compute_queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                            global_dims, local_dims,
                                            &upload_deps_loop, &ev_kernel[i]);
#ifndef FULL_DUPLEX
                ev_kernel[i].wait();
#endif
        }
        
        // Download the previous output into pinned memory and copy to host
        // move the previous output to the host
        if (i-1 > -1) {
            std::vector<cl::Event> kernel_deps_loop = {
                ev_kernel[i-1], last_ev_download[(i-1)%out_buffer_count]};
            kernel_deps_loop[1].wait(); // ensure previous download is complete
#ifdef PINNED_MEMORY
                std::copy(mapped_pinned_output_pointers[(i-1)%out_buffer_count],
                        mapped_pinned_output_pointers[(i-1)%out_buffer_count] + output.size(),
                        output.data());
#endif
            download_queue.enqueueReadBuffer(d_output[i-1], CL_FALSE, 0, out_bytes,
                                mapped_pinned_output_pointers[(i-1)%out_buffer_count],
                                &kernel_deps_loop, &ev_download[i-1]);
#ifdef PINNED_MEMORY
                last_ev_download[(i-1)%out_buffer_count] = ev_download[i-1];
#endif
#ifndef FULL_DUPLEX
                ev_download[i-1].wait();
#endif
        }

        // upload the next input
        if (i+1 < batch_repeats) {
            d_input[i+1] = cl::Buffer(context, CL_MEM_READ_ONLY,  in_bytes);
            upload_queue.enqueueWriteBuffer(d_input[i+1], CL_FALSE, 0, in_bytes,
                                            mapped_pinned_input_pointers[i+1],
                                            nullptr, &ev_upload[i+1]);
#ifndef FULL_DUPLEX
            ev_upload[i+1].wait();
#endif
        }
    }

      /* ---------- drain all three pipelines ---------- */
    upload_queue.finish();
    compute_queue.finish();
    download_queue.finish();

    /* ------------------ collect timings ---------------- */
    // const auto overall_end = std::chrono::high_resolution_clock::now();
    cl::UserEvent overall_end_event(context);
    overall_end_event.setStatus(CL_COMPLETE);
    queue.enqueueMarkerWithWaitList(nullptr, &overall_end_event);
    overall_end_event.wait();
    const auto overall_end = overall_end_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Overall_time: "
              << ns_to_ms(overall_end - overall_start)<< std::endl;

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
    cl_ulong t0 = overall_start_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
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

#ifdef PINNED_MEMORY
    // Cleanup: Unmap pinned buffers
    for (int i = 0; i < batch_repeats; ++i)
        queue.enqueueUnmapMemObject(pinned_input_host_buffers[i], 
            mapped_pinned_input_pointers[i], nullptr, nullptr);
    for (int i = 0; i < out_buffer_count; ++i)
        queue.enqueueUnmapMemObject(pinned_output_host_buffers[i], 
            mapped_pinned_output_pointers[i], nullptr, nullptr);
    queue.finish(); // Ensure unmaps complete
#endif
}

int main(int argc, char* argv[]) {
    char** default_argv = new char*[8];
    default_argv[0] = argv[0];
    default_argv[1] = (char*)
        // "30,30,3"; // kernel file input file (w h img_c)
        "input_lena.png";
    default_argv[2] = (char*)
        // "3,3,4"; // kernel file (fw fh <not given -> img_c> fc)
        "kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt";
        // "kernel_same_1.txt,kernel_same_2.txt,kernel_same_3.txt";
    default_argv[3] = (char*)"true"; // cpu: threads --- gpu: local mem
    default_argv[4] = (char*)"4"; // batch counts (w h c batch_size)
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