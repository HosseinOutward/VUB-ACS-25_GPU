#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <limits>
#include <algorithm>
#include <stb_image.h>
#include <stb_image_write.h>


// Function to load a kernel from a text file
// First line: kernel_height kernel_width
// Subsequent lines: kernel values (row major)
std::vector<float> loadConvKernelFromFile(const std::string& filename, int& kernel_height, int& kernel_width) {
    std::ifstream kernelFile(filename);
    if (!kernelFile) {
        throw std::runtime_error("Failed to open kernel file: " + filename);
    }

    kernelFile >> kernel_height >> kernel_width;
    if (kernelFile.fail() || kernel_height == 0 || kernel_width == 0) {
        throw std::runtime_error("Invalid kernel dimensions in file or read error: " + filename + ". Height: " + std::to_string(kernel_height) + ", Width: " + std::to_string(kernel_width));
    }

    std::vector<float> kernel(kernel_width * kernel_height);
    for (size_t i = 0; i < kernel.size(); ++i) {
        if (!(kernelFile >> kernel[i])) {
            if (kernelFile.eof()) {
                 throw std::runtime_error("Insufficient kernel values in file: " + filename + ". Expected " + std::to_string(kernel.size()) + ", got " + std::to_string(i));
            }
            throw std::runtime_error("Error reading kernel data (value index " + std::to_string(i) + ") from file: " + filename);
        }
    }
    
    float dummy; // Check for extra numerical data
    if (kernelFile >> dummy) {
        std::cout << "Warning: Extra numerical data found in kernel file " << filename << " after expected values." << std::endl;
    }

    std::cout << "Kernel loaded from " << filename << " (" << kernel_width << "x" << kernel_height << ")" << std::endl;
    return kernel;
}


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


void writePngImage(const std::vector<float>& data, uint32_t width, uint32_t height, int channels, const std::string& filename)  {
    if (channels < 1 || channels > 4) {
        throw std::runtime_error("Unsupported number of channels for PNG: " + std::to_string(channels) + 
                                 ". stbi_write_png supports 1, 2, 3, or 4 channels.");
    }
    if (data.empty() || width == 0 || height == 0) {
        throw std::runtime_error("Invalid image data or dimensions for writing PNG.");
    }
    
    size_t expected_data_size = static_cast<size_t>(width) * height * channels;
    if (data.size() != expected_data_size) {
        throw std::runtime_error("Data size (" + std::to_string(data.size()) + 
                                 ") does not match expected size based on dimensions and channels (" + 
                                 std::to_string(expected_data_size) + ").");
    }

    std::vector<unsigned char> image_data_char(expected_data_size);

    size_t plane_size = static_cast<size_t>(height) * width; // Total pixels in one channel plane

    for (uint32_t h = 0; h < height; ++h) {
        for (uint32_t w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                // Index for input channel-major float data
                size_t float_idx = static_cast<size_t>(c) * plane_size + static_cast<size_t>(h) * width + w;
                
                // Index for output interleaved unsigned char data
                size_t uchar_idx = (static_cast<size_t>(h) * width + w) * channels + c;
                
                float pixel_float_val = data[float_idx];
                
                // Clamp values to [0.0, 1.0] before scaling
                pixel_float_val = std::max(0.0f, std::min(1.0f, pixel_float_val));
                
                // Scale to 0-255 and add 0.5f for rounding before casting to unsigned char
                image_data_char[uchar_idx] = static_cast<unsigned char>(pixel_float_val * 255.0f + 0.5f);
            }
        }
    }

    int success = stbi_write_png(filename.c_str(), 
                                 static_cast<int>(width), 
                                 static_cast<int>(height), 
                                 channels, 
                                 image_data_char.data(), 
                                 static_cast<int>(width) * channels);
    
    if (!success)
        throw std::runtime_error("Failed to write PNG image: " + filename);

    std::cout << "Image successfully written to " << filename << " (" << width <<
            "x" << height << ", " << channels << " channels)" << std::endl;
}