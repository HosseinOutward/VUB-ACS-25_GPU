#include <algorithm> // Required for std::min/max if not already included, or for a custom clamp

// Helper function to clamp values before casting to unsigned char
inline unsigned char float_to_uchar_clamped(float val) {
    val = val * 255.0f;
    if (val < 0.0f) val = 0.0f;
    if (val > 255.0f) val = 255.0f;
    return static_cast<unsigned char>(val);
}

void loadInputs(
    FilePaths& paths, std::vector<unsigned char>& input_img, int& input_width, 
    int& input_height, int& channel_count, std::vector<float>& filter, // Changed to std::vector<float>
    int& filter_width, int& filter_height,
    int& stride, int& output_width, int& output_height, std::vector<float>& output // Output is now std::vector<float>
) {
    paths = loadPathsFromConfig("data/in/config_conv.txt");
    
    std::vector<float> float_input_img = readPngImage(paths.inputImagePath, input_width, input_height, channel_count);
    // filter is now float, so we can use float_filter directly or assign
    filter = loadConvKernelFromFile(paths.kernelFilePath, filter_height, filter_width);

    // Convert float input_img to unsigned char vector
    input_img.resize(float_input_img.size());
    std::transform(float_input_img.begin(), float_input_img.end(), input_img.begin(), [](float val) { 
        return float_to_uchar_clamped(val);
    });

    // No conversion needed for filter as it's now float
    // filter.resize(float_filter.size());
    // std::transform(float_filter.begin(), float_filter.end(), filter.begin(), [](float val) { 
    //     return float_to_uchar_clamped(val); // This was for uchar filter
    // });

    stride = 1; 
    output_width = (input_width - filter_width) / stride + 1;
    output_height = (input_height - filter_height) / stride + 1;
    // Initialize float output vector with 0.0f
    output.assign(static_cast<size_t>(output_width) * output_height * channel_count, 0.0f); 
}

// Helper function for GPU initialization
void initializeGPU(
    cl::Context& context, 
    cl::CommandQueue& queue, 
    cl::Device& device, 
    cl::Program& program, 
    cl::Kernel& kernel,
    int stride, int input_width, int input_height, 
    int filter_width, int filter_height, int channel_count,
    int output_width, int output_height
) {
    context = makeContextAndQueue(&queue, &device);
    const std::string kernel_source_code = loadKernelSource();
    program = cl::Program(context, kernel_source_code);
    program.build({device});
    kernel = cl::Kernel(program, "kernel_app");

    kernel.setArg(3, stride);
    kernel.setArg(4, input_width);
    kernel.setArg(5, input_height);
    kernel.setArg(6, filter_width);
    kernel.setArg(7, filter_height);
    kernel.setArg(8, channel_count);
    kernel.setArg(11, output_width);
    kernel.setArg(12, output_height);
}

// Helper function for local memory and worker configuration
void configureLocalMemoryAndWorkers(
    const cl::Device& device, cl::Kernel& kernel, int stride, 
    int filter_width, int filter_height, 
    int output_width, int output_height,
    size_t& local_wg_dim_x, size_t& local_wg_dim_y,
    size_t& global_work_size_x, size_t& global_work_size_y
) {
    size_t max_local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    // Adjust quads for cl_half if necessary, though optimal_wg_size calculation might be complex
    // For simplicity, keeping the existing logic, but be mindful of sizeof(cl_half)
    std::vector<float> quads = {1.0f, 1.0f*stride*(filter_width+filter_height),
        1.0f*(filter_width*filter_height) - (1.0f*max_local_mem_size)/sizeof(unsigned char)}; // Changed sizeof(float) to sizeof(cl_half)
    double discriminant = static_cast<double>(quads[1])*quads[1] - 4.0*quads[0]*quads[2];
    int optimal_wg_size = 1;
    if (discriminant >= 0) { // Check if discriminant is non-negative before taking sqrt
        optimal_wg_size = 1 + static_cast<int>(
            (-quads[1] + std::sqrt(discriminant)) / (2.0f * quads[0]));
    } else {
        // Handle the case where discriminant is negative, e.g., set a default optimal_wg_size
        // Or log an error, or adjust parameters. For now, keeping it 1.
        std::cerr << "Warning: Negative discriminant in worker configuration. Optimal WG size may not be ideal." << std::endl;
    }

    std::vector<size_t> max_item_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    if (!max_item_sizes.empty()) { // Ensure max_item_sizes is not empty
        optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(max_item_sizes[0]));
    }

    size_t max_total_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(
            std::sqrt(static_cast<double>(max_total_wg_size))));
    
    // Ensure optimal_wg_size is at least 1 to prevent issues with local_wg_dim being 0
    if (optimal_wg_size < 1) {
        optimal_wg_size = 1;
    }

    std::cout << "WI per WG: " << optimal_wg_size*optimal_wg_size << std::endl;

    local_wg_dim_x = optimal_wg_size;
    local_wg_dim_y = optimal_wg_size;
    
    size_t tile_w_host = (local_wg_dim_x - 1) * stride + filter_width;
    size_t tile_h_host = (local_wg_dim_y - 1) * stride + filter_height;
    size_t local_memory_size_bytes = tile_w_host * tile_h_host * sizeof(unsigned char); // Changed sizeof(float) to sizeof(cl_half)

    global_work_size_x = ((output_width + local_wg_dim_x - 1) / local_wg_dim_x) * local_wg_dim_x;
    global_work_size_y = ((output_height + local_wg_dim_y - 1) / local_wg_dim_y) * local_wg_dim_y;
    kernel.setArg(9, static_cast<int>(tile_w_host));
    kernel.setArg(10, static_cast<int>(tile_h_host));
    kernel.setArg(13, cl::Local(local_memory_size_bytes));
}
