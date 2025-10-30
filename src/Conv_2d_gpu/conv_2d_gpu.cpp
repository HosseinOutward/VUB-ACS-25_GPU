#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <cl2.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <cstring>
#include <thread>
#include <future>
#include <algorithm>

/* ---- project headers ---- */
#include "utilities.hpp"
#include "conv_2d_gpu.hpp"

/* --------------------------------------------------------------------- */
// Forward declarations
struct GPUContext;
struct ConvParams;

void initial_setup(GPUContext& gpu_ctx, const ConvParams& params, int gpu_idx);
void main_loop(GPUContext& gpu_ctx, const ConvParams& params);
void wrap_up(GPUContext& gpu_ctx);
void reporting_times(GPUContext& gpu_ctx, const ConvParams& params);
std::vector<int> get_nvidia_gpu_indices();

/* --------------------------------------------------------------------- */
static inline double ns_to_ms(cl_ulong  ns)        { return ns / 1'000'000.0; }

// Structure to hold per-GPU data
struct GPUContext {
    cl::Context context;
    cl::CommandQueue queue;
    cl::CommandQueue compute_queue;
    cl::CommandQueue upload_queue;
    cl::CommandQueue download_queue;
    cl::Device device;
    cl::Program program;
    cl::Kernel kernel;
    cl::Buffer d_filter;
    
    std::vector<cl::Buffer> d_input;
    std::vector<cl::Buffer> d_output;
    std::vector<cl::Buffer> pinned_input_host_buffers;
    std::vector<cl::Buffer> pinned_output_host_buffers;
    
    std::vector<cl::Event> ev_upload;
    std::vector<cl::Event> ev_kernel;
    std::vector<cl::Event> ev_download;
    std::vector<cl::Event> last_ev_download;
    
    std::vector<unsigned char*> mapped_pinned_input_pointers;
    std::vector<unsigned char*> mapped_pinned_output_pointers;
    
    std::vector<double> hw_start_upload_ms, hw_start_kernel_ms, 
        hw_start_download_ms, upload_ms, kernel_ms, download_ms;
    
    cl::NDRange local_dims;
    cl::NDRange global_dims;
    
    cl::UserEvent overall_start_event;
    cl::UserEvent overall_end_event;
    cl_ulong overall_start;
    cl_ulong overall_end;
    
    int out_buffer_count;
    int ctx_idx;
};

// Structure to hold shared parameters
struct ConvParams {
    char** argv;
    std::vector<unsigned char>& input_batch;
    int input_width, input_height, channel_count;
    std::vector<float>& filter;
    int filter_width, filter_height, filter_channels, stride;
    std::vector<unsigned char>& output;
    int output_width, output_height;
    int batch_repeats;
    int b_size;
    bool detailed_timings;
    bool pinned_memory;
    bool full_duplex;
    size_t in_bytes;
    size_t out_bytes;
    
    // GPU-specific batch distribution
    int batch_start;
    int batch_end;
    int gpu_batch_count;
    
    ConvParams(char* argv[], 
               std::vector<unsigned char>& input_batch, int& input_width, int& input_height, int& channel_count,
               std::vector<float>& filter, int& filter_width, int& filter_height, int& filter_channels, int& stride,
               std::vector<unsigned char>& output, int& output_width, int& output_height)
        : argv(argv), input_batch(input_batch), input_width(input_width), input_height(input_height), 
          channel_count(channel_count), filter(filter), filter_width(filter_width), 
          filter_height(filter_height), filter_channels(filter_channels), stride(stride),
          output(output), output_width(output_width), output_height(output_height) {
        
        detailed_timings = (std::strcmp(argv[8], "true") == 0);
        pinned_memory = (std::strcmp(argv[9], "true") == 0);
        full_duplex = (std::strcmp(argv[10], "true") == 0);
        batch_repeats = std::stoi(argv[3]);
        b_size = std::stoi(argv[6]);
        in_bytes = sizeof(unsigned char) * input_batch.size();
        out_bytes = sizeof(unsigned char) * output.size();
        
        // Initialize batch distribution (will be set per GPU)
        batch_start = 0;
        batch_end = batch_repeats;
        gpu_batch_count = batch_repeats;
    }
    
    // Copy constructor for GPU-specific params with different output buffer
    ConvParams(const ConvParams& other, std::vector<unsigned char>& new_output)
        : argv(other.argv), input_batch(other.input_batch), 
          input_width(other.input_width), input_height(other.input_height), 
          channel_count(other.channel_count), filter(other.filter),
          filter_width(other.filter_width), filter_height(other.filter_height),
          filter_channels(other.filter_channels), stride(other.stride),
          output(new_output), output_width(other.output_width), output_height(other.output_height),
          batch_repeats(other.batch_repeats), b_size(other.b_size),
          detailed_timings(other.detailed_timings), pinned_memory(other.pinned_memory),
          full_duplex(other.full_duplex), in_bytes(other.in_bytes),
          batch_start(other.batch_start), batch_end(other.batch_end),
          gpu_batch_count(other.gpu_batch_count) {
        out_bytes = sizeof(unsigned char) * new_output.size();
    }
    
    // Set batch range for a specific GPU
    void setBatchRange(int start, int end) {
        batch_start = start;
        batch_end = end;
        gpu_batch_count = end - start;
    }
};

/* --------------------------------------------------------------------- */
void run(char* argv[],
        std::vector<unsigned char>& input_batch, int& input_width, int& input_height, int& channel_count,
        std::vector<float>& filter, int& filter_width, int& filter_height, int& filter_channels, int& stride,
        std::vector<unsigned char>& output, int& output_width, int& output_height)
{
    std::vector<int> nvidia_gpu_indices = get_nvidia_gpu_indices();
    if (nvidia_gpu_indices.empty()) {
        std::cerr << "No NVIDIA GPU found. Exiting." << std::endl;
        return;
    }
    
    // Create shared parameters
    ConvParams base_params(argv, input_batch, input_width, input_height, channel_count,
                          filter, filter_width, filter_height, filter_channels, stride,
                          output, output_width, output_height);
    
    int total_batches = base_params.batch_repeats;
    size_t num_gpus = nvidia_gpu_indices.size();

    // limit gpu count to number of batches
    num_gpus = std::min(num_gpus, static_cast<size_t>(total_batches));
    
    std::cout << "Distributing " << total_batches << " batches across " << num_gpus << " GPUs" << std::endl;
    
    // Calculate batch distribution across GPUs
    std::vector<ConvParams> gpu_params;
    std::vector<std::vector<unsigned char>> gpu_outputs(num_gpus);
    
    for (size_t i = 0; i < num_gpus; ++i) {
        // Calculate batch range for this GPU
        int batch_start = (total_batches * i) / num_gpus;
        int batch_end = (total_batches * (i + 1)) / num_gpus;
        int gpu_batch_count = batch_end - batch_start;
        
        std::cout << "GPU " << nvidia_gpu_indices[i] << " will handle batches " 
                  << batch_start << " to " << (batch_end - 1) << " (" << gpu_batch_count << " batches)" << std::endl;
        
        // Create separate output buffer for this GPU
        // Each GPU's output size is proportional to its batch count
        size_t single_batch_output_size = output.size() / total_batches;
        gpu_outputs[i].resize(single_batch_output_size * gpu_batch_count);
        
        // Create a copy of params for this GPU with its own output buffer
        gpu_params.emplace_back(base_params, gpu_outputs[i]);
        gpu_params[i].setBatchRange(batch_start, batch_end);
    }
    
    // Create GPU contexts for each GPU
    std::vector<GPUContext> gpu_contexts(num_gpus);
    
    // Do initial setup for each GPU
    for (size_t i = 0; i < num_gpus; ++i) {
        std::cout << "Setting up GPU " << nvidia_gpu_indices[i] << std::endl;
        initial_setup(gpu_contexts[i], gpu_params[i], nvidia_gpu_indices[i]);
        gpu_contexts[i].ctx_idx = i;
    }
    
    // Run all GPUs in parallel using futures
    std::vector<std::future<void>> gpu_futures;
    
    std::cout << "Starting parallel execution on " << num_gpus << " GPUs..." << std::endl;

    // overall time timer start
    auto overall_start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_gpus; ++i) {
        gpu_futures.push_back(std::async(std::launch::async, [&, i]() {
            std::cout << "GPU " << nvidia_gpu_indices[i] << " starting main loop..." << std::endl;
            main_loop(gpu_contexts[i], gpu_params[i]);
            std::cout << "GPU " << nvidia_gpu_indices[i] << " completed main loop" << std::endl;
        }));
    }

    // Wait for all GPUs to complete
    for (auto& future : gpu_futures)
        future.wait();
    
    // Wrap up for each GPU (collect timing data)
    for (size_t i = 0; i < num_gpus; ++i) {
        std::cout << "Wrapping up GPU " << nvidia_gpu_indices[i] << std::endl;
        wrap_up(gpu_contexts[i]);
    }

    // overall time timer end
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(total_end_time - overall_start_time);
    double total_time_ms = total_duration.count() / 1'000'000.0;
    
    // Collect and merge results from all GPUs
    std::cout << "Collecting results from all GPUs..." << std::endl;

    // reporting the overall time
    std::cout << "Overall_time: " << total_time_ms << std::endl;
    
    for (size_t i = 0; i < num_gpus; ++i) {
        // std::cout << "Wrapping up GPU " << nvidia_gpu_indices[i] << std::endl;
        reporting_times(gpu_contexts[i], gpu_params[i]);
    }
    
    // Calculate output dimensions per batch
    size_t single_batch_output_size = output.size() / total_batches;
    
    // Merge results from all GPUs back into the main output buffer
    for (size_t gpu_idx = 0; gpu_idx < num_gpus; ++gpu_idx) {
        int batch_start = gpu_params[gpu_idx].batch_start;
        int batch_count = gpu_params[gpu_idx].gpu_batch_count;
        
        // std::cout << "Merging " << batch_count << " batches from GPU " << nvidia_gpu_indices[gpu_idx] 
        //           << " (batches " << batch_start << " to " << (batch_start + batch_count - 1) << ")" << std::endl;
        
        // Copy each batch from GPU output to the correct position in main output
        for (int batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
            size_t global_batch_idx = batch_start + batch_idx;
            size_t src_offset = batch_idx * single_batch_output_size;
            size_t dst_offset = global_batch_idx * single_batch_output_size;
            
            // Copy this batch's data from GPU output to main output
            std::copy(gpu_outputs[gpu_idx].begin() + src_offset,
                     gpu_outputs[gpu_idx].begin() + src_offset + single_batch_output_size,
                     output.begin() + dst_offset);
        }
    }
}

/* --------------------------------------------------------------------- */
void initial_setup(GPUContext& gpu_ctx, const ConvParams& params, int gpu_idx) {
    /*------------------ gpu init ------------------*/
    /* IMPORTANT PART - Create separate queues for maximum parallelism */
    initializeGPU(gpu_idx, gpu_ctx.context, gpu_ctx.queue, gpu_ctx.device, gpu_ctx.program, gpu_ctx.kernel, 
                  (strcmp(params.argv[7], "true") == 0),
                  params.stride, params.input_width, params.input_height, params.channel_count, 
                  params.filter_width, params.filter_height, params.filter_channels, 
                  params.output_width, params.output_height);
    
    // Check if device supports out-of-order execution for full-duplex
    cl_command_queue_properties qprops = CL_QUEUE_PROFILING_ENABLE;
    qprops |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    
    // Three separate queues for true full duplex operation
    gpu_ctx.compute_queue = cl::CommandQueue(gpu_ctx.context, gpu_ctx.device, qprops);   // For kernel execution
    gpu_ctx.upload_queue = cl::CommandQueue(gpu_ctx.context, gpu_ctx.device, qprops);    // For data uploads (DMA in)
    gpu_ctx.download_queue = cl::CommandQueue(gpu_ctx.context, gpu_ctx.device, qprops);  // For data downloads (DMA out)

    /* -------------------- work-group sizes ---------------------- */
    std::size_t gws_x = params.output_width,
                gws_y = params.output_height;

    gpu_ctx.local_dims = cl::NullRange;

    if (std::strcmp(params.argv[7], "true") == 0) {
        std::cout << "Using local memory\n";
        std::size_t ldx, ldy;
        configureLocalMemoryAndWorkers(gpu_ctx.device, gpu_ctx.kernel, params.stride,
            params.filter_width,  params.filter_height,  params.filter_channels, 
            params.output_width,  params.output_height,  params.channel_count,
            ldx, ldy, gws_x, gws_y);
        gpu_ctx.local_dims = cl::NDRange(ldx, ldy, params.filter_channels);
    } else
        std::cout << "no local memory" << std::endl;

    gpu_ctx.global_dims = cl::NDRange(gws_x, gws_y, params.filter_channels * params.b_size);

    /* ------------------ constants / filter upload --------------- */
    gpu_ctx.d_filter = cl::Buffer(gpu_ctx.context,
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * params.filter.size(),
                        const_cast<float*>(params.filter.data()));
    gpu_ctx.kernel.setArg(2, gpu_ctx.d_filter);
    
    /* ------------------ batch buffers & events ------------------ */
    gpu_ctx.d_input.resize(params.gpu_batch_count);
    gpu_ctx.d_output.resize(params.gpu_batch_count);

    gpu_ctx.ev_upload.resize(params.gpu_batch_count);
    gpu_ctx.ev_kernel.resize(params.gpu_batch_count);
    gpu_ctx.ev_download.resize(params.gpu_batch_count);

    if(params.detailed_timings) {
        gpu_ctx.hw_start_upload_ms.resize(params.gpu_batch_count);
        gpu_ctx.hw_start_kernel_ms.resize(params.gpu_batch_count);
        gpu_ctx.hw_start_download_ms.resize(params.gpu_batch_count);
        gpu_ctx.upload_ms.resize(params.gpu_batch_count);
        gpu_ctx.kernel_ms.resize(params.gpu_batch_count);
        gpu_ctx.download_ms.resize(params.gpu_batch_count);
    }

    if(params.pinned_memory)
        gpu_ctx.out_buffer_count = 2;
    else
        gpu_ctx.out_buffer_count = params.gpu_batch_count;

    gpu_ctx.mapped_pinned_input_pointers.resize(params.gpu_batch_count);
    gpu_ctx.mapped_pinned_output_pointers.resize(gpu_ctx.out_buffer_count);

    if(params.pinned_memory) {
        /* ------------------ pinned memory buffers ------------------ */
        gpu_ctx.pinned_input_host_buffers.resize(params.gpu_batch_count);
        gpu_ctx.pinned_output_host_buffers.resize(gpu_ctx.out_buffer_count);
        // simulate as if pinned memory was used when loading input data
        for (int i = 0; i < params.gpu_batch_count; ++i) {
            gpu_ctx.pinned_input_host_buffers[i] = cl::Buffer(gpu_ctx.context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY, params.in_bytes);
            gpu_ctx.mapped_pinned_input_pointers[i] = (unsigned char*)gpu_ctx.queue.enqueueMapBuffer(
                gpu_ctx.pinned_input_host_buffers[i], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, params.in_bytes);

            std::copy(params.input_batch.data(), params.input_batch.data() + params.input_batch.size(), gpu_ctx.mapped_pinned_input_pointers[i]);
        }
    } else {
        for (int i = 0; i < params.gpu_batch_count; ++i)
            gpu_ctx.mapped_pinned_input_pointers[i] = const_cast<unsigned char*>(params.input_batch.data());
        for (int i = 0; i < gpu_ctx.out_buffer_count; ++i)
            gpu_ctx.mapped_pinned_output_pointers[i] = params.output.data();
    }
}

/* --------------------------------------------------------------------- */
void main_loop(GPUContext& gpu_ctx, const ConvParams& params) {
    /* ------------------ main loop: enqueue ---------------------- */
    // Get device timestamp at start
    gpu_ctx.overall_start_event = cl::UserEvent(gpu_ctx.context);
    gpu_ctx.overall_start_event.setStatus(CL_COMPLETE);
    gpu_ctx.queue.enqueueMarkerWithWaitList(nullptr, &gpu_ctx.overall_start_event);
    gpu_ctx.overall_start_event.wait();
    gpu_ctx.overall_start = gpu_ctx.overall_start_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    gpu_ctx.last_ev_download.resize(gpu_ctx.out_buffer_count);
    for (int i = 0; i < gpu_ctx.out_buffer_count; ++i) {
        cl::UserEvent ue(gpu_ctx.context);
        ue.setStatus(CL_COMPLETE);
        gpu_ctx.last_ev_download[i] = ue;
    }

    if(params.pinned_memory) {
        // double buffering for output
        for (int i = 0; i < gpu_ctx.out_buffer_count; ++i) {
            gpu_ctx.pinned_output_host_buffers[i] = cl::Buffer(gpu_ctx.context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_READ_ONLY, params.out_bytes);
            gpu_ctx.mapped_pinned_output_pointers[i] = (unsigned char*)gpu_ctx.queue.enqueueMapBuffer(
                gpu_ctx.pinned_output_host_buffers[i], CL_TRUE, CL_MAP_READ, 0, params.out_bytes);
        }
        gpu_ctx.queue.finish();
    }

    // warmup the queues
    gpu_ctx.d_input[0] = cl::Buffer(gpu_ctx.context, CL_MEM_READ_ONLY,  params.in_bytes);
    gpu_ctx.upload_queue.enqueueWriteBuffer(gpu_ctx.d_input[0], CL_FALSE, 0, params.in_bytes,
                                   gpu_ctx.mapped_pinned_input_pointers[0],
                                   nullptr, &gpu_ctx.ev_upload[0]);

    // main loop - (user's new order: kernel, download, upload)
    for (int i = 0; i <= params.gpu_batch_count; ++i) {
        if (i < params.gpu_batch_count) {
            // run kernel on current input
            gpu_ctx.d_output[i]= cl::Buffer(gpu_ctx.context, CL_MEM_READ_WRITE, params.out_bytes);
            gpu_ctx.kernel.setArg(0, gpu_ctx.d_input[i]);
            gpu_ctx.kernel.setArg(1, gpu_ctx.d_output[i]);
            std::vector<cl::Event> upload_deps_loop = {gpu_ctx.ev_upload[i]};
            gpu_ctx.compute_queue.enqueueNDRangeKernel(gpu_ctx.kernel, cl::NullRange,
                                            gpu_ctx.global_dims, gpu_ctx.local_dims,
                                            &upload_deps_loop, &gpu_ctx.ev_kernel[i]);

            if(!params.full_duplex)
                gpu_ctx.ev_kernel[i].wait();
        }
        
        // Download the previous output into pinned memory and copy to host
        // move the previous output to the host
        if (i-1 > -1) {
            std::vector<cl::Event> kernel_deps_loop = {
                gpu_ctx.ev_kernel[i-1], gpu_ctx.last_ev_download[(i-1)%gpu_ctx.out_buffer_count]};
            kernel_deps_loop[1].wait(); // ensure previous download is complete
            if(params.pinned_memory) {
                std::copy(gpu_ctx.mapped_pinned_output_pointers[(i-1)%gpu_ctx.out_buffer_count],
                        gpu_ctx.mapped_pinned_output_pointers[(i-1)%gpu_ctx.out_buffer_count] + params.output.size(),
                        const_cast<unsigned char*>(params.output.data()));
            }
            gpu_ctx.download_queue.enqueueReadBuffer(gpu_ctx.d_output[i-1], CL_FALSE, 0, params.out_bytes,
                                gpu_ctx.mapped_pinned_output_pointers[(i-1)%gpu_ctx.out_buffer_count],
                                &kernel_deps_loop, &gpu_ctx.ev_download[i-1]);
            if(params.pinned_memory)
                gpu_ctx.last_ev_download[(i-1)%gpu_ctx.out_buffer_count] = gpu_ctx.ev_download[i-1];
                
            if(!params.full_duplex)
                gpu_ctx.ev_download[i-1].wait();
        }

        // upload the next input
        if (i+1 < params.gpu_batch_count) {
            gpu_ctx.d_input[i+1] = cl::Buffer(gpu_ctx.context, CL_MEM_READ_ONLY,  params.in_bytes);
            gpu_ctx.upload_queue.enqueueWriteBuffer(gpu_ctx.d_input[i+1], CL_FALSE, 0, params.in_bytes,
                                            gpu_ctx.mapped_pinned_input_pointers[i+1],
                                            nullptr, &gpu_ctx.ev_upload[i+1]);
            
            if(!params.full_duplex)
                gpu_ctx.ev_upload[i+1].wait();
        }
    }
}

/* --------------------------------------------------------------------- */
void wrap_up(GPUContext& gpu_ctx) {
      /* ---------- drain all three pipelines ---------- */
    gpu_ctx.upload_queue.finish();
    gpu_ctx.compute_queue.finish();
    gpu_ctx.download_queue.finish();
}

/* --------------------------------------------------------------------- */
void reporting_times(GPUContext& gpu_ctx, const ConvParams& params) {
    /* ------------------ collect timings ---------------- */
    gpu_ctx.overall_end_event = cl::UserEvent(gpu_ctx.context);
    gpu_ctx.overall_end_event.setStatus(CL_COMPLETE);
    gpu_ctx.queue.enqueueMarkerWithWaitList(nullptr, &gpu_ctx.overall_end_event);
    gpu_ctx.overall_end_event.wait();
    gpu_ctx.overall_end = gpu_ctx.overall_end_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    std::cout << "gpu_overall_time_" << gpu_ctx.ctx_idx << ": "
              << ns_to_ms(gpu_ctx.overall_end - gpu_ctx.overall_start)<< std::endl;


    if(params.detailed_timings) {
        for (int i = 0; i < params.gpu_batch_count; ++i) {
            cl_ulong ns0, ns1;

            /* upload (write buffer) */
            gpu_ctx.ev_upload[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &ns0);
            gpu_ctx.ev_upload[i].getProfilingInfo(CL_PROFILING_COMMAND_END,   &ns1);
            gpu_ctx.hw_start_upload_ms[i] = ns_to_ms(ns0);
            gpu_ctx.upload_ms[i]  = ns_to_ms(ns1 - ns0);

            /* kernel */
            gpu_ctx.ev_kernel[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &ns0);
            gpu_ctx.ev_kernel[i].getProfilingInfo(CL_PROFILING_COMMAND_END,   &ns1);
            gpu_ctx.hw_start_kernel_ms[i] = ns_to_ms(ns0);
            gpu_ctx.kernel_ms[i]  = ns_to_ms(ns1 - ns0);

            /* download (read buffer) */
            gpu_ctx.ev_download[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &ns0);
            gpu_ctx.ev_download[i].getProfilingInfo(CL_PROFILING_COMMAND_END,   &ns1);
            gpu_ctx.hw_start_download_ms[i] = ns_to_ms(ns0);
            gpu_ctx.download_ms[i]= ns_to_ms(ns1 - ns0);
        }

        /* ------------------ human-readable output ------------------- */
        // calculate relative times
        cl_ulong t0 = gpu_ctx.overall_start_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        auto hw_rel = [&](double hw_start_ms){ return hw_start_ms - ns_to_ms(t0); };

        for (int i = 0; i < params.gpu_batch_count; ++i) {
            std::cout << "Start_upload_" << (params.batch_start + i) << ": " << hw_rel(gpu_ctx.hw_start_upload_ms[i]) << std::endl;
            std::cout << "Start_kernel_" << (params.batch_start + i) << ": " << hw_rel(gpu_ctx.hw_start_kernel_ms[i]) << std::endl;
            std::cout << "Start_download_" << (params.batch_start + i) << ": " << hw_rel(gpu_ctx.hw_start_download_ms[i]) << std::endl;

            std::cout << "Upload_time_" << (params.batch_start + i) << ": " << gpu_ctx.upload_ms[i] << std::endl;
            std::cout << "Kernel_time_" << (params.batch_start + i) << ": " << gpu_ctx.kernel_ms[i] << std::endl;
            std::cout << "Download_time_" << (params.batch_start + i) << ": " << gpu_ctx.download_ms[i] << std::endl;
        }
    }

    if(params.pinned_memory) {
        // Cleanup: Unmap pinned buffers
        for (int i = 0; i < params.gpu_batch_count; ++i)
            gpu_ctx.queue.enqueueUnmapMemObject(gpu_ctx.pinned_input_host_buffers[i], 
                gpu_ctx.mapped_pinned_input_pointers[i], nullptr, nullptr);
        for (int i = 0; i < gpu_ctx.out_buffer_count; ++i)
            gpu_ctx.queue.enqueueUnmapMemObject(gpu_ctx.pinned_output_host_buffers[i], 
                gpu_ctx.mapped_pinned_output_pointers[i], nullptr, nullptr);
        gpu_ctx.queue.finish(); // Ensure unmaps complete
    }
}

/* --------------------------------------------------------------------- */
int main(int argc, char* argv[]) {
    char** default_argv = new char*[11];
    default_argv[0] = argv[0];
    default_argv[1] = (char*)
        // "30,30,3"; // kernel file input file (w h img_c)
        "input_lena.png";
    default_argv[2] = (char*)
        // "3,3,4"; // kernel file (fw fh <not given -> img_c> fc)
        "kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt";
        // "kernel_same_1.txt,kernel_same_2.txt,kernel_same_3.txt";
    default_argv[3] = (char*)"4"; // batch counts (w h c batch_size)
    default_argv[4] = (char*)"output_gpu.png"; // output file
    default_argv[5] = (char*)"2"; // stride
    default_argv[6] = (char*)"5"; // simulated single batch size 
    default_argv[7] = (char*)"False"; // cpu: threads --- gpu: local mem
    default_argv[8] = (char*)"true"; // detailed timings
    default_argv[9] = (char*)"true"; // pinned memory
    default_argv[10] = (char*)"true"; // full duplex
    
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