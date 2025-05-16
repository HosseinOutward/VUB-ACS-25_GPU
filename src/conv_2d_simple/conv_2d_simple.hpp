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
    kernel.setArg(9, output_width);
    kernel.setArg(10, output_height);
}
