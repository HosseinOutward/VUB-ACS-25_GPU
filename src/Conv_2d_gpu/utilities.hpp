#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stb_image.h>
#include <stb_image_write.h>

// Function to read a png image
// Normalizes pixel values to floats between 0.0 and 1.0
// Returns a std::vector<float> representing the image data in channel-major format.
// width (w_int), height (h_int), and channels (c_int_out) are set to the image dimensions.
std::vector<float> readPngImage(const std::string& filename, int& w_int, int& h_int, int& c_int_out) {
    unsigned char* raw_data = stbi_load(filename.c_str(), &w_int, &h_int, &c_int_out, 0);
    if (!raw_data)
        throw std::runtime_error("Failed to load image: " + filename + " (" + stbi_failure_reason() + ")");

    int processed_channels = c_int_out;
    if (c_int_out == 4)
        processed_channels = 3;

    std::vector<float> float_vector(processed_channels * h_int * w_int);
    for (int c = 0; c < processed_channels; ++c)
        for (int h = 0; h < h_int; ++h)
            for (int w = 0; w < w_int; ++w) {
                unsigned char* pixel_start_in_raw = raw_data + (h * w_int + w) * c_int_out;
                float val = static_cast<float>(pixel_start_in_raw[c]) / 255.0f;
                float_vector[c * (h_int * w_int) + h * w_int + w] = val;
            }

    stbi_image_free(raw_data); // Free the memory allocated by stbi_load

    std::cout << "Image loaded from " << filename << " (" << w_int <<
            "x" << h_int << ", " << processed_channels <<
            " processed channels from " << c_int_out << " original channels)" << std::endl;

    c_int_out = processed_channels;
    return float_vector;
}

// Function to write a png image
void writePngImage(const std::vector<float>& src, // (fc, h, w) and 0-1
            int w, int h, int c, const std::string& file)  {
    std::vector<unsigned char> dst(src.size());
    std::size_t k = 0;

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int ch = 0; ch < c; ++ch)
                dst[k++] = static_cast<unsigned char>(
                    std::min(1.f, std::max(0.f, src[ch * h * w + y * w + x])) * 255.f + 0.5f);
    stbi_write_png(file.c_str(), w, h, c, dst.data(), w * c);
    std::cout << "Image saved to " << file << " (" << w << "x" << h << ", " << c << " channels)" << std::endl;
}

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

// Function to generate a random input image data
std::vector<unsigned char> generate_input_data(int& input_width, int& input_height, int& channel_count) {
    std::vector<unsigned char> input_img((size_t)input_width * input_height * channel_count);
    srand(42);
    for(size_t i = 0; i < input_img.size(); ++i)
        input_img[i] = static_cast<unsigned char>(rand() % 256);
    return input_img;
}

// Function to load input image data from a file
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
        std::copy(input_img.begin(), input_img.end(), 
            reps_input_img.begin() + i * input_width * input_height * channel_count);

    // # define WATERMARK_PATCH_INPUT
    #ifdef WATERMARK_PATCH_INPUT
    // add patch for the first i*m pixels for each reps for batch size watermarks
    for (int i = 1; i < reps; ++i) {
        int b_offset = i * (input_width * input_height * channel_count);

        for (int j = 0; j < 150*i; ++j)
            for (int k = 0; k < 150*i; ++k)
                for (int c = 0; c < 3; ++c) {
                    int c_offset = c * input_width * input_height;
                    int index = b_offset + c_offset + j*input_width + k;
                    reps_input_img[index] = static_cast<unsigned char>(rand() % 256);
                }
    }
    #endif
    
    return reps_input_img;
}

// generate a convolution kernel
std::vector<float> generate_filter(int input_width, int input_height, int channel_count, int filter_channel) {
    std::vector<float> kernel_data((size_t)input_width * input_height * filter_channel * channel_count);
    srand(42);
    for(size_t i = 0; i < kernel_data.size(); ++i)
        kernel_data[i] = static_cast<float>(rand() % 10);
    return kernel_data;
}

// Function to load a convolution kernel from a file
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

void prep_and_run(char* argv[], 
    void (*run_f)(char* argv[], std::vector<unsigned char>& input_batch, 
        int& input_width, int& input_height, int& channel_count,
        std::vector<float>& filter, int& filter_width, int& filter_height, int& filter_count, 
        int& stride, std::vector<unsigned char>& output, int& output_width, int& output_height)
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
        input_img = generate_input_data(d[0], d[1], d[2]);
    } else
        input_img = load_input_img(argv[1], input_width, input_height, channel_count); // (width, height, channels, reps)

    int b_size = std::stoi(argv[7]);
    input_batch = repeat_to_make_batch(input_img, b_size, input_width, input_height, channel_count);

    // Load filter
    if (isDimensionString(argv[2])) {
        std::vector<int> d = parseDimensions(argv[2]);
        filter_width = d[0];
        filter_height = d[1];
        filter_count = d[2];
        filter = generate_filter(d[0], d[1], channel_count, d[2]);
    } else
        filter = load_filter(argv[2], 
            filter_width, filter_height, filter_count, channel_count); // (fw, fh, img_channels, fc)
    
    // asset 

    /*------------------ output dim ------------------*/
    int stride = std::stoi(argv[6]);    int output_width = 1.0f*(input_width - filter_width) / stride + 1;
    int output_height = 1.0f*(input_height - filter_height) / stride + 1;
    std::vector<unsigned char> output(
        static_cast<size_t>(output_width) * output_height * filter_count * b_size);

    /*------------------ run and out ------------------*/
    run_f(argv, input_batch, input_width, input_height, channel_count,
            filter, filter_width, filter_height, filter_count, stride,
            output, output_width, output_height);
    
    // ------------------ result -------------------------
    if (strcmp(argv[5], "null") == 0)
        return;
    else{
        std::vector<float> float_output(output.size());
        float_output.resize(static_cast<size_t>(output_width) * output_height * 3);
        std::transform(output.begin(), output.end(), float_output.begin(), 
            [](unsigned char val) {return static_cast<float>(val) / 255.0f;});

        // take the second 3 channels
        int single_img_size = output_width*output_height*filter_count;
        for (int i = 0; i < single_img_size; ++i)
            float_output[i] = float_output[i + single_img_size*2];
        float_output.resize(static_cast<size_t>(output_width) * output_height * 3);

        std::transform(float_output.begin(), float_output.end(), float_output.begin(), 
            [](float val) {return std::abs(val);});
        float max_val = *std::max_element(float_output.begin(), float_output.end());
        float min_val = *std::min_element(float_output.begin(), float_output.end());
        std::transform(float_output.begin(), float_output.end(), float_output.begin(), 
            [min_val, max_val](float val) {return (val - min_val) / (max_val - min_val);});

        writePngImage(float_output, output_width, output_height, 3, std::string("data/out/")+argv[5]);
    }
}

