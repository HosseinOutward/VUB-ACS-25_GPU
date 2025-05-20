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


struct FilePaths {
    std::string inputImagePath;
    std::string kernelFilePath;
    std::string outputImagePath;
};

FilePaths loadPathsFromConfig(const std::string& configFilePath) {
    std::ifstream configFile(configFilePath);
    if (!configFile) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    FilePaths paths;
    if (!(std::getline(configFile, paths.inputImagePath))) {
        throw std::runtime_error("Failed to read input image path from config file: " + configFilePath);
    }
    if (!(std::getline(configFile, paths.kernelFilePath))) {
        throw std::runtime_error("Failed to read kernel file path from config file: " + configFilePath);
    }
    if (!(std::getline(configFile, paths.outputImagePath))) {
        throw std::runtime_error("Failed to read output image path from config file: " + configFilePath);
    }

    // Basic validation: check if paths are empty
    if (paths.inputImagePath.empty() || paths.kernelFilePath.empty() || paths.outputImagePath.empty()) {
        throw std::runtime_error("One or more paths in the config file are empty: " + configFilePath);
    }
    
    // std::cout << "Loaded Input Image Path: " << paths.inputImagePath << std::endl;
    // std::cout << "Loaded Kernel File Path: " << paths.kernelFilePath << std::endl;
    // std::cout << "Loaded Output Image Path: " << paths.outputImagePath << std::endl;

    configFile.close();
    return paths;
}
