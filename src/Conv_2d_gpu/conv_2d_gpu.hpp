#include <algorithm>
#include <optional>
#include <direct.h>
#include "utilities.cpp"

std::vector<float> loadFilterFromFile(const std::string& filename, 
    int& kernel_height, int& kernel_width, int& filter_channel) {
    std::ifstream kernelFile(filename);
    if (!kernelFile) {
        throw std::runtime_error("Failed to open kernel file: " + filename);
    }

    kernelFile >> kernel_height >> kernel_width >> filter_channel;
    if (kernelFile.fail() || kernel_height <= 0 || kernel_width <= 0 || filter_channel <= 0) {
        throw std::runtime_error("Invalid kernel dimensions in file or read error: " + filename + 
                                ". Height: " + std::to_string(kernel_height) + 
                                ", Width: " + std::to_string(kernel_width) + 
                                ", Channels: " + std::to_string(filter_channel));
    }

    // Allocate for all channels
    std::vector<float> kernel(kernel_width * kernel_height * filter_channel);
    
    // Read values for each channel
    for (int c = 0; c < filter_channel; ++c) {
        for (int h = 0; h < kernel_height; ++h) {
            for (int w = 0; w < kernel_width; ++w) {
                size_t idx = c * (kernel_height * kernel_width) + h * kernel_width + w;
                if (!(kernelFile >> kernel[idx])) {
                    if (kernelFile.eof()) {
                        throw std::runtime_error("Insufficient kernel values in file: " + filename + 
                                              ". Expected " + std::to_string(kernel.size()) + 
                                              ", got " + std::to_string(idx));
                    }
                    throw std::runtime_error("Error reading kernel data (channel " + std::to_string(c) + 
                                          ", row " + std::to_string(h) + 
                                          ", col " + std::to_string(w) + ") from file: " + filename);
                }
            }
        }
    }
    
    float dummy; // Check for extra numerical data
    if (kernelFile >> dummy) {
        std::cout << "Warning: Extra numerical data found in kernel file " << filename << " after expected values." << std::endl;
    }

    std::cout << "Kernel loaded from " << filename << " (" << kernel_width << "x" << kernel_height << 
             ", " << filter_channel << " channels)" << std::endl;
    return kernel;
}

bool isDimensionString(const std::string& str) {
    return str.find_first_not_of("0123456789,") == std::string::npos && 
           str.find(',') != std::string::npos;
}
inline unsigned char float_to_uchar_clamped(float val) {
    val = val * 255.0f;
    if (val < 0.0f) val = 0.0f;
    if (val > 255.0f) val = 255.0f;
    return static_cast<unsigned char>(val);
}

std::vector<int> parseDimensions(const std::string& str) {
    std::vector<int> dimensions;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            dimensions.push_back(std::stoi(token));
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("Invalid dimension value: " + token);
        }
    }
    return dimensions;
}

// Function to load input image data from a file
std::vector<unsigned char> load_input_img(int& input_width, int& input_height, int& channel_count) {
    std::vector<unsigned char> input_img((size_t)input_width * input_height * channel_count);
    srand(static_cast<unsigned int>(time(nullptr)));
    for(size_t i = 0; i < input_img.size(); ++i)
        input_img[i] = static_cast<unsigned char>(rand() % 256);
    return input_img;
}

std::vector<unsigned char> load_input_img(const std::string& filename,
        int& input_width, int& input_height, int& channel_count) {
    std::vector<unsigned char> input_img;
    std::vector<float> temp;
    temp=readPngImage("data/in/"+filename, input_width, input_height, channel_count);
    input_img.resize(temp.size());
    std::transform(temp.begin(), temp.end(), input_img.begin(), [](float val) { 
        return float_to_uchar_clamped(val);
    });

    return input_img;
}

std::vector<unsigned char> repeat_to_make_batch(std::vector<unsigned char> input_img,
        int reps, int& input_width, int& input_height, int& channel_count) {

    std::vector<unsigned char> reps_input_img(
        static_cast<size_t>(input_width) * input_height * channel_count * reps);
    for (int i = 0; i < reps; ++i)
        std::copy(
            input_img.begin(), 
            input_img.end(), 
            reps_input_img.begin() + i * input_width * input_height * channel_count);
    return reps_input_img;
}

// Function to load a convolution kernel from a file
std::vector<float> load_filter(int input_width, int input_height, int channel_count, int filter_channel) {
    std::vector<float> kernel_data((size_t)input_width * input_height * filter_channel * channel_count);
    srand(static_cast<unsigned int>(time(nullptr)));
    for(size_t i = 0; i < kernel_data.size(); ++i)
        kernel_data[i] = static_cast<unsigned char>(rand() % 256);
    return kernel_data;
}

std::vector<float> load_filter(const std::string& filenames, 
        int& input_width, int& input_height, int& filter_channel, int channel_count) {
    // Parse comma-separated filenames
    std::vector<std::string> filenames_array;
    std::stringstream ss(filenames);
    std::string filename;
    while (std::getline(ss, filename, ','))
        filenames_array.push_back(filename);

    filter_channel = filenames_array.size();
    
    std::vector<float> kernel_data;
    for (int i = 0; i < filter_channel; ++i) {
        std::string filename = filenames_array[i];

        int temp_channel;
        std::vector<float> temp = loadFilterFromFile("data/in/kernels/"+filename,
             input_height, input_width, temp_channel);

        if (temp_channel != channel_count)
            throw std::runtime_error("Mismatch in channel count for filter " + filename);
        
        if (i == 0)
            kernel_data.resize(temp.size() * filter_channel);

        // Copy the kernel data into the appropriate position in kernel_data
        std::copy(temp.begin(), temp.end(), kernel_data.begin() + i * temp.size());
    }
    return kernel_data;
}

void loadInputs(
    FilePaths& paths, std::vector<unsigned char>& input_img, int& input_width, 
    int& input_height, int& channel_count, std::vector<float>& filter, // Changed to std::vector<float>
    int& filter_width, int& filter_height, int& filter_channels,
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

    stride = 1; 
    output_width = (input_width - filter_width) / stride + 1;
    output_height = (input_height - filter_height) / stride + 1;
    output.assign(static_cast<size_t>(output_width) * output_height * filter_channels, 0.0f); // (w h fc)
}

// Helper function for GPU initialization
void initializeGPU(
    cl::Context& context, 
    cl::CommandQueue& queue, 
    cl::Device& device, 
    cl::Program& program, 
    cl::Kernel& kernel,
    int uses_local_mem,
    int stride, int input_width, int input_height, int channel_count, 
    int filter_width, int filter_height, int filter_channels,
    int output_width, int output_height
) {
    context = makeContextAndQueue(&queue, &device);
    const std::string kernel_source_code = loadKernelSource();
    program = cl::Program(context, kernel_source_code);
    program.build({device});
    if (uses_local_mem)
        kernel = cl::Kernel(program, "kernel_app_local");
    else
        kernel = cl::Kernel(program, "kernel_app");

    kernel.setArg(3, stride);
    kernel.setArg(4, input_width);
    kernel.setArg(5, input_height);
    kernel.setArg(6, channel_count);
    kernel.setArg(7, filter_width);
    kernel.setArg(8, filter_height);
    kernel.setArg(9, filter_channels);
    kernel.setArg(10, output_width);
    kernel.setArg(11, output_height);
}


// Helper function for local memory and worker configuration
void configureLocalMemoryAndWorkers(
    const cl::Device& device, cl::Kernel& kernel, int stride, 
    int filter_width, int filter_height, int filter_channels,
    int output_width, int output_height, int input_channels,
    size_t& local_wg_dim_x, size_t& local_wg_dim_y,
    size_t& global_work_size_x, size_t& global_work_size_y, size_t& global_work_size_z
) {
    // _______________________________ calculate optimal local mem size _______________________________
    size_t max_local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() 
                - filter_width*filter_height*filter_channels*input_channels*sizeof(float);
    
    int optimal_wg_size;
    std::vector<float> quads = {1.0f*stride*stride, 1.0f*stride*(filter_width+filter_height),
        1.0f*(filter_width+1)/2*(filter_height+1)/2 - (1.0f*max_local_mem_size)/sizeof(unsigned char)/input_channels};

    double discriminant = static_cast<double>(quads[1])*quads[1] - 4.0*quads[0]*quads[2];
    if (discriminant >= 0)
        optimal_wg_size = 1 + static_cast<int>((-quads[1] + std::sqrt(discriminant)) / (2.0f * quads[0]));
    else
        std::cerr << "Warning: Negative discriminant." << std::endl;

    std::vector<size_t> max_item_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    if (!max_item_sizes.empty())
        optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(max_item_sizes[0]));

    size_t max_total_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    optimal_wg_size = std::min(optimal_wg_size, static_cast<int>(
            std::sqrt(static_cast<double>(max_total_wg_size))));
    
    // Ensure optimal_wg_size is at least 1 to prevent issues with local_wg_dim being 0
    if (optimal_wg_size < 1)
        optimal_wg_size = 1;

    std::cout << "WI per WG: " << optimal_wg_size*optimal_wg_size << std::endl;

    // _______________________________ allocate _______________________________
    local_wg_dim_x = optimal_wg_size;
    local_wg_dim_y = optimal_wg_size;
    
    global_work_size_x = ((output_width + local_wg_dim_x - 1) / local_wg_dim_x) * local_wg_dim_x;
    global_work_size_y = ((output_height + local_wg_dim_y - 1) / local_wg_dim_y) * local_wg_dim_y;
    global_work_size_z = filter_channels;
    
    size_t tile_w_host = local_wg_dim_x * stride + (filter_height+1)/2;
    size_t tile_h_host = local_wg_dim_y * stride + (filter_height+1)/2;
    size_t local_memory_size_bytes = tile_w_host * tile_h_host * input_channels * sizeof(unsigned char);
    size_t local_filter_memory_size_bytes = static_cast<size_t>(filter_width) * 
            filter_height * input_channels * filter_channels * sizeof(float);

    kernel.setArg(12, static_cast<int>(tile_w_host));
    kernel.setArg(13, static_cast<int>(tile_h_host));
    kernel.setArg(14, cl::Local(local_memory_size_bytes));
    kernel.setArg(15, cl::Local(local_filter_memory_size_bytes));
}

void prep_and_run(char* argv[], 
    void (*run_f)(char* argv[], std::vector<unsigned char>& input_batch, 
        int& input_width, int& input_height, int& channel_count,
        std::vector<float>& filter, int& filter_width, int& filter_height, int& filter_count, 
        int& stride, std::vector<float>& output, int& output_width, int& output_height)
    )
{
    /*------------------ INPUT ------------------*/
    std::vector<unsigned char> input_img, input_batch;
    std::vector<float> filter;
    int input_width, input_height, channel_count,
        filter_width, filter_height, filter_count;

    // Load batch
    if (isDimensionString(argv[1])) {
        std::vector<int> d = parseDimensions(argv[1]);
        input_width = d[0];
        input_height = d[1];
        channel_count = d[2];
        input_img = load_input_img(d[0], d[1], d[2]);
    } else
        input_img = load_input_img(argv[1], input_width, input_height, channel_count); // (width, height, channels, reps)

    int img_c_reps = std::stoi(argv[7]);
    input_batch = repeat_to_make_batch(input_img, img_c_reps, input_width, input_height, channel_count);

    // Load filter
    if (isDimensionString(argv[2])) {
        std::vector<int> d = parseDimensions(argv[2]);
        filter_width = d[0];
        filter_height = d[1];
        filter_count = d[2];
        filter = load_filter(d[0], d[1], channel_count, d[2]);
    } else
        filter = load_filter(argv[2], 
            filter_width, filter_height, filter_count, channel_count); // (fw, fh, img_channels, fc)
    
    // asset 

    /*------------------ output dim ------------------*/
    int stride = std::stoi(argv[6]);
    int output_width = 1.0f*(input_width - filter_width) / stride + 1;
    int output_height = 1.0f*(input_height - filter_height) / stride + 1;
    std::vector<float> output(
        static_cast<size_t>(output_width) * output_height * filter_count);

    /*------------------ run and out ------------------*/
    run_f(argv, input_batch, input_width, input_height, channel_count,
            filter, filter_width, filter_height, filter_count, stride,
            output, output_width, output_height);
    
    // ------------------ result -------------------------
    if (strcmp(argv[5], "null") == 0)
        return;
    else{
        output.resize(static_cast<size_t>(output_width) * output_height * 3);

        std::transform(output.begin(), output.end(), output.begin(), 
            [](float val) {return std::abs(val);});
        float max_val = *std::max_element(output.begin(), output.end());
        float min_val = *std::min_element(output.begin(), output.end());
        std::transform(output.begin(), output.end(), output.begin(), 
            [min_val, max_val](float val) {return (val - min_val) / (max_val - min_val);});

        writePngImage(output, output_width, output_height, 3, std::string("data/out/")+argv[5]);
    }
}

int replace_argv(int argc, char* argv[], char* default_argv[],
    void (*prep_and_run_func)(char* argv[], 
        std::vector<unsigned char>& input_batch, int& input_width, int& input_height, int& channel_count,
        std::vector<float>& filter, int& filter_width, int& filter_height, int& filter_channels, int& stride,
        std::vector<float>& output, int& output_width, int& output_height)
    )
{
    // print working directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working dir: " << cwd << std::endl;
    } else {
        perror("getcwd() error");
        return 1;
    }
    // print executable name
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string fullPath(buffer);

    try {
        if (argc == 1) {
            prep_and_run(default_argv, prep_and_run_func);
        } else
            prep_and_run(argv, prep_and_run_func);
    // ------------------ try catch -------------------------
    } catch (cl::Error const &e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1; // Indicate an error
    } catch (std::exception const &e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0; // Indicate success
}
