#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

// Include existing utility functions
#include "../Conv_2d_gpu/conv_2d_gpu.hpp"

// Generate random array
std::vector<float> randomArray(size_t size) {
    std::vector<float> arr(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);
    for (size_t i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }
    return arr;
}

int main(int argc, char* argv[]) {
    // Default values
    std::string kernel_type = "sum";
    size_t array_size = 1000000;
    int repetitions = 100;
    int float_type = 32;
    int elements_per_thread = 1;  // New parameter

    // Parse arguments with defaults
    if (argc >= 2) kernel_type = argv[1];
    if (argc >= 3) array_size = std::stoull(argv[2]);
    if (argc >= 4) repetitions = std::stoi(argv[3]);
    if (argc >= 5) float_type = std::stoi(argv[4]);
    if (argc >= 6) elements_per_thread = std::stoi(argv[5]);

    // Validate arguments
    if (kernel_type != "sum" && kernel_type != "fma" && kernel_type != "sum_vec4" &&
        kernel_type != "sum_overlap" && kernel_type != "sum_strided" &&
        kernel_type != "sum_reverse" && kernel_type != "sum_redundant" &&
        kernel_type != "fma_overlap" && kernel_type != "fma_strided" &&
        kernel_type != "fma_reverse" && kernel_type != "fma_redundant") {
        std::cout << "Error: Invalid kernel_type, using default 'sum'\n";
        std::cout << "Valid options: sum, fma, sum_vec4, sum_overlap, sum_strided, sum_reverse, sum_redundant,\n";
        std::cout << "               fma_overlap, fma_strided, fma_reverse, fma_redundant\n";
        kernel_type = "sum";
    }
    
    if (float_type != 16 && float_type != 32) {
        std::cout << "Error: float_type must be 16 or 32, using default 32\n";
        float_type = 32;
    }
    
    if (elements_per_thread < 1) {
        std::cout << "Error: elements_per_thread must be >= 1, using default 1\n";
        elements_per_thread = 1;
    }

    if (argc == 1 || argc > 6) {
        std::cout << "Usage: " << argv[0] << " [kernel_type] [array_size] [repetitions] [float_type] [elements_per_thread]\n";
        std::cout << "Defaults: kernel_type=sum, array_size=1000000, repetitions=100, float_type=32, elements_per_thread=1\n";
        std::cout << "Efficient kernels: sum, fma, sum_vec4\n";
        std::cout << "Inefficient kernels: sum_overlap, sum_strided, sum_reverse, sum_redundant,\n";
        std::cout << "                     fma_overlap, fma_strided, fma_reverse, fma_redundant\n";
        std::cout << "Running with defaults...\n\n";
    }
    
    std::cout << "Running " << kernel_type << " with " << array_size 
              << " elements, " << repetitions << " repetitions, float" << float_type
              << ", " << elements_per_thread << " elements/thread\n";

    // Generate data
    std::vector<float> input1 = randomArray(array_size);
    std::vector<float> input2 = randomArray(array_size);
    std::vector<float> output(array_size);

    // Setup GPU using existing functions
    cl::CommandQueue queue;
    cl::Device device;
    cl::Context context = makeContextAndQueue(&queue, &device, 0);
    
    const std::string kernel_source_code = loadKernelSource();
    cl::Program program(context, kernel_source_code);
    
    // Add compiler options for half precision if needed
    std::string build_options = "";
    if (float_type == 16) {
        build_options = "-cl-std=CL2.0 -DUSE_HALF";
    }
    
    try {
        program.build({device}, build_options.c_str());
    } catch (const cl::Error& e) {
        std::cout << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        throw;
    }
    
    // Select appropriate kernel
    std::string kernel_name;
    bool use_multi_kernel = elements_per_thread > 1;
    bool is_inefficient = false;

    if (kernel_type == "sum_vec4") {
        kernel_name = "kernel_app_multi_vec4";
        use_multi_kernel = true;
    } else if (kernel_type == "sum_overlap") {
        kernel_name = "kernel_app_multi_overlap";
        use_multi_kernel = true;
        is_inefficient = true;
    } else if (kernel_type == "sum_strided") {
        kernel_name = "kernel_app_multi_strided";
        use_multi_kernel = true;
        is_inefficient = true;
    } else if (kernel_type == "sum_reverse") {
        kernel_name = "kernel_app_multi_reverse";
        use_multi_kernel = true;
        is_inefficient = true;
    } else if (kernel_type == "sum_redundant") {
        kernel_name = "kernel_app_multi_redundant";
        use_multi_kernel = true;
        is_inefficient = true;
    } else if (kernel_type == "fma_overlap") {
        kernel_name = "kernel_app_FMA_multi_overlap";
        use_multi_kernel = true;
        is_inefficient = true;
    } else if (kernel_type == "fma_strided") {
        kernel_name = "kernel_app_FMA_multi_strided";
        use_multi_kernel = true;
        is_inefficient = true;
    } else if (kernel_type == "fma_reverse") {
        kernel_name = "kernel_app_FMA_multi_reverse";
        use_multi_kernel = true;
        is_inefficient = true;
    } else if (kernel_type == "fma_redundant") {
        kernel_name = "kernel_app_FMA_multi_redundant";
        use_multi_kernel = true;
        is_inefficient = true;
    } else if (use_multi_kernel) {
        kernel_name = (kernel_type == "sum") ? "kernel_app_multi" : "kernel_app_FMA_multi";
    } else {
        kernel_name = (kernel_type == "sum") ? "kernel_app" : "kernel_app_FMA";
    }

    cl::Kernel kernel(program, kernel_name);

    // GPU memory - handle both float16 and float32
    cl::Buffer d_input1, d_input2, d_output;

    auto start_upload = std::chrono::high_resolution_clock::now();
    if (float_type == 16) {
        // Convert float32 to float16 for GPU
        std::vector<cl_half> input1_half(array_size), input2_half(array_size), output_half(array_size);
        
        // Convert to half precision (simple conversion)
        for (size_t i = 0; i < array_size; ++i) {
            input1_half[i] = (cl_half)input1[i];
            input2_half[i] = (cl_half)input2[i];
        }
        start_upload = std::chrono::high_resolution_clock::now();
        d_input1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(cl_half) * input1_half.size(), input1_half.data());
        d_input2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(cl_half) * input2_half.size(), input2_half.data());
        d_output = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_half) * output_half.size());
    } else {
        // Use float32
        start_upload = std::chrono::high_resolution_clock::now();
        d_input1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * input1.size(), input1.data());
        d_input2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * input2.size(), input2.data());
        d_output = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());
    }


    // Set kernel arguments
    kernel.setArg(0, d_input1);
    kernel.setArg(1, d_input2);
    kernel.setArg(2, d_output);
    kernel.setArg(3, repetitions);

    // Calculate number of threads needed
    size_t num_threads;
    if (use_multi_kernel) {
        kernel.setArg(4, elements_per_thread);
        kernel.setArg(5, (int)array_size);

        if (is_inefficient && (kernel_type.find("overlap") != std::string::npos)) {
            // For overlap kernels, we need more threads to cover all elements
            // due to the inefficient overlapping assignment
            num_threads = array_size; // One thread per element (massive overlap)
        } else {
            num_threads = (array_size + elements_per_thread - 1) / elements_per_thread; // Ceiling division
        }
    } else {
        num_threads = array_size;
    }

    std::cout << "Launching " << num_threads << " threads, " << elements_per_thread << " elements per thread\n";
    if (is_inefficient) {
        std::cout << "⚠️  Using INEFFICIENT kernel variant - expect poor performance!\n";
    }

    // Run kernel and time it
    cl::Event event;
    auto start = std::chrono::high_resolution_clock::now();
    
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num_threads), cl::NullRange, nullptr, &event);
    queue.finish();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto host_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Get GPU time
    cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double gpu_time = (end_time - start_time) / 1e6; // Convert to ms

    // Read results - handle both float types
    if (float_type == 16) {
        std::vector<cl_half> output_half(array_size);
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(cl_half) * output_half.size(), output_half.data());
        
        // Convert back to float32 for display
        for (size_t i = 0; i < array_size; ++i) {
            output[i] = (float)output_half[i];
        }
    } else {
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data());
    }
    
    auto end_upload = std::chrono::high_resolution_clock::now();
    auto upload_time = std::chrono::duration<double, std::milli>(end_upload - start_upload).count();

    // Print results with efficiency warnings
    std::cout << "\nTiming Results:\n";
    std::cout << "Host time: " << upload_time-host_time << " ms\n";
    std::cout << "GPU time:  " << host_time << " ms\n";
    std::cout << "Threads launched: " << num_threads << "\n";
    std::cout << "Elements per thread: " << elements_per_thread << "\n";

    if (is_inefficient) {
        std::cout << "⚠️  INEFFICIENT KERNEL WARNINGS:\n";
        if (kernel_type.find("overlap") != std::string::npos) {
            std::cout << "   - Overlapping work: Multiple threads process same elements\n";
            std::cout << "   - Race conditions: Last writer wins, wasting previous work\n";
            std::cout << "   - Work efficiency: ~" << (double)array_size / (num_threads * elements_per_thread) * 100 << "%\n";
        } else if (kernel_type.find("strided") != std::string::npos) {
            std::cout << "   - Poor cache locality: Large stride memory access pattern\n";
            std::cout << "   - Memory bandwidth waste: Non-coalesced global memory access\n";
        } else if (kernel_type.find("reverse") != std::string::npos) {
            std::cout << "   - Branch divergence: Different threads take different execution paths\n";
            std::cout << "   - Work imbalance: Unequal work distribution between threads\n";
        } else if (kernel_type.find("redundant") != std::string::npos) {
            std::cout << "   - Redundant memory reads: Same data read multiple times\n";
            std::cout << "   - Memory bandwidth waste: Unnecessary global memory traffic\n";
        }
    } else {
        std::cout << "Thread efficiency: " << (double)array_size / (num_threads * elements_per_thread) * 100 << "%\n";
    }

    double ops = (double)array_size * repetitions;
    double gops = ops / (gpu_time / 1000.0) / 1e9;
    std::cout << "Performance: " << gops << " GOPS\n";
    
    // Show sample
    std::cout << "\nSample results:\n";
    for (int i = 0; i < 3 && i < (int)array_size; ++i) {
        std::cout << input1[i] << " " << ((kernel_type == "sum") ? "+" : "fma") 
                  << " " << input2[i] << " = " << output[i] << "\n";
    }
    
    return 0;
}
