#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <ocl_utilities.hpp>
#include "..\conv_2d_sequential\function.cpp"
#include "..\conv_2d_sequential\utilities.cpp"
#include <cmath> // Required for std::sqrt

int main()
{
    try {
    /*------------------ INPUT ------------------*/
    // Load paths from config file
    FilePaths paths = loadPathsFromConfig("data/in/config_conv.txt");

    int input_width, input_height, channel_count;
    std::vector<float> input_img = readPngImage(paths.inputImagePath, input_width, input_height, channel_count);

    // Filter
    int filter_width, filter_height;
    std::vector<float> filter = loadConvKernelFromFile(paths.kernelFilePath, filter_height, filter_width);

    int stride = 1;
    const int output_width = (input_width - filter_width) / stride + 1;
    const int output_height = (input_height - filter_height) / stride + 1;
    const int batch_count = 4; // Number of batches to process for high throughput
    std::vector<std::vector<float>> outputs(batch_count, std::vector<float>(static_cast<size_t>(output_width) * output_height * channel_count));

    // ------------------ gpu init ------------------
    cl::CommandQueue queue;
    cl::Device device;
    cl::Context context = makeContextAndQueue(&queue, &device);
    const std::string kernel_source_code = loadKernelSource();
    cl::Program program(context, kernel_source_code);
    program.build({device});
    cl::Kernel kernel(program, "kernel_app");

    // ------------------ query device limits ------------------
    size_t max_total_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::vector<size_t> max_item_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    size_t max_local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    // ------------------ local memory and worker config ------------------
    std::vector<float> quads = {1.0f, 1.0f*stride*(filter_width+filter_height), 
        1.0f*(filter_width*filter_height) - (1.0f*max_local_mem_size)/sizeof(float)};
    double discriminant = static_cast<double>(quads[1])*quads[1] - 4.0*quads[0]*quads[2];
    int optimal_wg_size = 1 + static_cast<int>(
        (-quads[1] + std::sqrt(discriminant)) / (2.0f * quads[0]));

    optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(max_item_sizes[0]));
    optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(
            std::sqrt(static_cast<double>(max_total_wg_size))));
    std::cout << "WI per WG: " << optimal_wg_size*optimal_wg_size << std::endl;

    const size_t local_wg_dim_x = optimal_wg_size;
    const size_t local_wg_dim_y = optimal_wg_size;

    // ------------ per group tile dimensions & local memory size
    size_t tile_w_host = (local_wg_dim_x - 1) * stride + filter_width;
    size_t tile_h_host = (local_wg_dim_y - 1) * stride + filter_height;
    size_t local_memory_size_bytes = tile_w_host * tile_h_host * sizeof(float);

    cl::NDRange local_dims(local_wg_dim_x, local_wg_dim_y, 1); // worker groups
    size_t global_work_size_x = ((output_width + local_wg_dim_x - 1) / local_wg_dim_x) * local_wg_dim_x;
    size_t global_work_size_y = ((output_height + local_wg_dim_y - 1) / local_wg_dim_y) * local_wg_dim_y;
    cl::NDRange global_dims(global_work_size_x, global_work_size_y, channel_count);

    // ------------------ high throughput batch pipeline ------------------
    std::vector<cl::Buffer> d_inputs(batch_count);
    std::vector<cl::Buffer> d_outputs(batch_count);
    std::vector<cl::Event> kernel_events(batch_count);
    std::vector<cl::Event> read_events(batch_count);
    cl::Buffer d_filter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filter.size(), filter.data());

    // Pre-create device buffers for all batches
    for (int i = 0; i < batch_count; ++i) {
        d_inputs[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * input_img.size(), input_img.data());
        d_outputs[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof(float) * outputs[i].size(), nullptr);
    }

    std::cout << "\nBeginning high-throughput conv2d with " << batch_count << " batches\n";

    // Launch all kernels (overlap compute and transfer)
    for (int i = 0; i < batch_count; ++i) {
        kernel.setArg(0, d_inputs[i]);
        kernel.setArg(1, d_filter);
        kernel.setArg(2, d_outputs[i]);
        kernel.setArg(3, stride);
        kernel.setArg(4, input_width);
        kernel.setArg(5, input_height);
        kernel.setArg(6, filter_width);
        kernel.setArg(7, filter_height);
        kernel.setArg(8, channel_count);
        kernel.setArg(9, static_cast<int>(tile_w_host));
        kernel.setArg(10, static_cast<int>(tile_h_host));
        kernel.setArg(11, output_width);
        kernel.setArg(12, output_height);
        kernel.setArg(13, cl::Local(local_memory_size_bytes));
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_dims, local_dims, nullptr, &kernel_events[i]);
    }

    // Read back all results (overlap with kernel execution)
    for (int i = 0; i < batch_count; ++i) {
        // Wait for kernel to finish before reading
        kernel_events[i].wait();
        queue.enqueueReadBuffer(d_outputs[i], CL_FALSE, 0, sizeof(float) * outputs[i].size(), outputs[i].data(), nullptr, &read_events[i]);
    }
    // Wait for all reads to finish
    for (int i = 0; i < batch_count; ++i) {
        read_events[i].wait();
    }
    std::cout << "Finished all batches\n";

    // Write all outputs
    for (int i = 0; i < batch_count; ++i) {
        std::string outPath = paths.outputImagePath;
        if (batch_count > 1) {
            // Add batch index to output filename
            size_t dot = outPath.find_last_of('.');
            if (dot != std::string::npos) {
                outPath.insert(dot, "_batch" + std::to_string(i));
            } else {
                outPath += "_batch" + std::to_string(i);
            }
        }
        writePngImage(outputs[i], output_width, output_height, channel_count, outPath);
        std::cout << "Wrote output: " << outPath << std::endl;
    }

    } catch (cl::Error const &e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    } catch (std::exception const &e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
