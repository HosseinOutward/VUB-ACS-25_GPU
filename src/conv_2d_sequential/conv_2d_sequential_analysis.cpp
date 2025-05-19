#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
// #include <cl2.hpp> // Not strictly needed for CPU sequential version

#include <vector>
#include <iostream>
#include <stdexcept>
#include <string> // Required for std::string, std::stoi
#include <chrono> // Required for timing
#include <iomanip> // Required for std::fixed, std::setprecision
// #include <thread> // For std::thread::hardware_concurrency() if you want to display it, but function.cpp handles actual threading choice

#include "function.cpp"
#include "utilities.cpp"

// Function to generate a kernel of a specific size (width x height)
std::vector<float> generate_dummy_kernel(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Kernel dimensions must be positive.");
    }
    std::vector<float> kernel(static_cast<size_t>(width) * height);
    // Example: Normalized box filter
    float val = 1.0f / (static_cast<float>(width) * height);
    for (size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = val;
    }
    return kernel;
}

int main(int argc, char* argv[]) {
    // Default values
    int batch_size = 1;
    int kernel_width_arg = 3; // Default kernel width if not provided
    int kernel_height_arg = 3; // Default kernel height if not provided

    // Parse command-line arguments
    // argv[0] is the program name
    // argv[1] is batch_size
    // argv[2] is kernel_width
    // argv[3] is kernel_height
    if (argc > 1) {
        try {
            batch_size = std::stoi(argv[1]);
            if (batch_size <= 0) {
                std::cerr << "Warning: Batch size must be positive. Using default: 1" << std::endl;
                batch_size = 1;
            }
        } catch (const std::exception& e) {
            std::cout << "Error: Invalid batch_size argument '" << argv[1] << "'. Details: " << e.what() << std::endl;
            // std::cout << "Error: Invalid batch_size argument" << std::endl; // For Python script
            return 1; // Indicate error to Python script
        }
    }

    if (argc > 2) {
        try {
            kernel_width_arg = std::stoi(argv[2]);
            if (kernel_width_arg <= 0) {
                std::cerr << "Warning: Kernel width must be positive. Using default: 3" << std::endl;
                kernel_width_arg = 3;
            }
        } catch (const std::exception& e) {
            std::cout << "Error: Invalid kernel_width argument '" << argv[2] << "'. Details: " << e.what() << std::endl;
            // std::cout << "Error: Invalid kernel_width argument" << std::endl; // For Python script
            return 1; // Indicate error
        }
    }

    if (argc > 3) {
        try {
            kernel_height_arg = std::stoi(argv[3]);
            if (kernel_height_arg <= 0) {
                std::cerr << "Warning: Kernel height must be positive. Using default: 3" << std::endl;
                kernel_height_arg = 3;
            }
        } catch (const std::exception& e) {
            std::cout << "Error: Invalid kernel_height argument '" << argv[3] << "'. Details: " << e.what() << std::endl;
            // std::cout << "Error: Invalid kernel_height argument" << std::endl; // For Python script
            return 1; // Indicate error
        }
    }


    std::vector<float> filter;

    try {
        filter = generate_dummy_kernel(kernel_width_arg, kernel_height_arg);
        // std::cout << "GeneratedKernelSize: " << kernel_width_arg << "x" << kernel_height_arg << std::endl; // Optional debug output
    } catch (const std::runtime_error& e) {
        std::cout << "Error: Failed to generate kernel: " << e.what() << std::endl;
        return 1; // Critical error, cannot proceed
    }

    /*------------------ INPUT ------------------*/
    FilePaths paths = loadPathsFromConfig("data/in/config_conv.txt"); // Relative to execution CWD

    int input_width, input_height, channel_count;
    std::vector<float> input_img;
    try {
        input_img = readPngImage(paths.inputImagePath, input_width, input_height, channel_count);
    } catch (const std::runtime_error& e) {
        std::cout << "Error: Failed to load input image '" << paths.inputImagePath << "': " << e.what() << std::endl;
        return 1; // Critical error
    }

    int stride = 1;
    const int output_width = (input_width - kernel_width_arg) / stride + 1;
    const int output_height = (input_height - kernel_height_arg) / stride + 1;

    if (output_width <= 0 || output_height <= 0) {
        std::cout << "Error: Invalid output dimensions. Input (" << input_width << "x" << input_height
                  << ", channels: " << channel_count << ") with filter (" << kernel_width_arg << "x" << kernel_height_arg
                  << ") results in non-positive output dimension (" << output_width << "x" << output_height << ")." << std::endl;
        return 1;
    }
    std::vector<float> output(static_cast<size_t>(output_width) * output_height * channel_count);

    int threads = 0; // Use all available hardware threads (handled by conv2d_cpu_threaded)

    // ------------------ run CPU convolution (timed) ------------------
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < batch_size; ++i) {
        conv2d_cpu_threaded(input_img.data(), input_width, input_height, channel_count,
                           filter.data(), kernel_width_arg, kernel_height_arg, // Use kernel_width_arg, kernel_height_arg
                           output.data(), threads, stride);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double duration_ms = static_cast<double>(duration_ns.count()) / 1e6;

    // ------------------ Output for Python script -------------------------
    // Ensure this output is clean and easily parsable.
    std::cout << "OverallBatchProcessingTime_ms: " << std::fixed << std::setprecision(4) << duration_ms << std::endl;
    std::cout << "ActualBatchCount: " << batch_size << std::endl;
    std::cout << "KernelWidth: " << kernel_width_arg << std::endl;
    std::cout << "KernelHeight: " << kernel_height_arg << std::endl;
    // std::cout << "UsedKernelFile: " << kernel_file_path << std::endl; // Path used or "generated" // Removed
    // std::cout << "InputImage: " << paths.inputImagePath << std::endl; // Optional for debugging
    // std::cout << "ThreadsUsed: " << (threads == 0 ? std::thread::hardware_concurrency() : threads) << std::endl; // Optional

    // ------------------ result (optional, typically commented out for analysis) ------------------
    // writePngImage(output, output_width, output_height, channel_count, paths.outputImagePath);
    // std::cout << "Output image saved to: " << paths.outputImagePath << std::endl;

    return 0;
}
