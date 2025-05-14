#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Utility: read a whole text file into a std::string
std::string loadKernelSource()
{
    // Extract folder name dynamically from current file path
    std::string filePath = __FILE__;
    std::replace(filePath.begin(), filePath.end(), '\\', '/');
            
    // Extract the folder name from the path
    size_t lastSep = filePath.rfind('/');
    size_t prevSep = filePath.rfind('/', lastSep - 1);
    std::string folder_name = (prevSep != std::string::npos) ? 
        filePath.substr(prevSep + 1, lastSep - prevSep - 1) : "";

    // Construct the full path to the kernel file
    std::string path = "src/" + folder_name + "/kernel.ocl";

    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open kernel file: " + path);

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

void writePgmImage(const std::vector<float>& data, int width, int height, const std::string& filename) {
    std::ofstream imageFile(filename, std::ios::binary);
    if (!imageFile) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    // Find min and max values for normalization
    float minVal = data[0];
    float maxVal = data[0];
    for (size_t i = 1; i < data.size(); ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
    }

    // Write PGM header
    imageFile << "P5\n" // binary PGM
                << width << " " << height << "\n"
                << "255\n"; // Max value

    // Write the pixel data (normalize to 0-255)
    std::vector<unsigned char> imageData(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        // Normalize to 0-255 range
        if (maxVal > minVal) {
            imageData[i] = static_cast<unsigned char>(255.0f * (data[i] - minVal) / (maxVal - minVal));
        } else {
            imageData[i] = 0;
        }
    }
    imageFile.write(reinterpret_cast<char*>(imageData.data()), imageData.size());

    std::cout << "Output written to " << filename << std::endl;
}

// Create OpenCL context and queue
inline cl::Context makeContextAndQueue(cl::CommandQueue* outQueue, cl::Device* outDevice)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
        throw std::runtime_error("No OpenCL platforms found");

    cl::Device device;
    for (const auto& plat : platforms)
    {
        std::vector<cl::Device> devices;
        plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (!devices.empty())
        {
            device = devices[0];
            break;
        }
    }
    if (!device())
        throw std::runtime_error("No GPU devices found");

    cl::Context context({device});
    cl::CommandQueue queue(context, device);

    if (outQueue)  *outQueue  = queue;
    if (outDevice) *outDevice = device;

    return context;
}


int main()
{
    // -----------------------------------------------------------------
    try
    {
        cl::CommandQueue queue;
        cl::Device device;
        cl::Context context = makeContextAndQueue(&queue, &device);

        // Load the kernel source code
        const std::string kernel_source_code = loadKernelSource();

        cl::Program program(context, kernel_source_code);
        program.build({device});

        // Create the kernel
        cl::Kernel kernel(program, "kernel_app");

        /*___________________________INPUT____________________________*/
        // todo: replace with dynamic inputs
        // Host-side data
        const uint32_t input_width = 2048;
        const uint32_t input_height = 2048;
        const uint32_t filter_width = 3;
        const uint32_t filter_height = 3;


        // Input image (5x5)
        std::vector<float> input(input_width * input_height);
        for (uint32_t i = 0; i < input_width * input_height; i++)
            input[i] = 0.5f*(i % input_width);

        // Filter (3x3)
        std::vector<float> filter = {
            0.5f, 0.8f, 1.0f,
            0.5f, 0.8f, 1.0f,
            0.5f, 0.8f, 1.0f
        };
        /*___________________________End INPUT_________________________*/

        const int output_width = input_width - filter_width + 1;
        const int output_height = input_height - filter_height + 1;

        // Output image (after convolution max size input_width * input_height)
        std::vector<float> output(output_width * output_height, 0.0f);

        // Device buffers
        cl::Buffer d_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * input.size(), input.data());
        cl::Buffer d_filter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * filter.size(), filter.data());
        cl::Buffer d_output(context, CL_MEM_READ_WRITE,
                            sizeof(float) * output.size(), nullptr);

        // Set kernel arguments
        kernel.setArg(0, d_input);
        kernel.setArg(1, d_filter);
        kernel.setArg(2, d_output);
        kernel.setArg(3, input_width);
        kernel.setArg(4, input_height);
        kernel.setArg(5, filter_width);
        kernel.setArg(6, filter_height);

        // Execute the kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output_width, output_height));
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data());

        /*___________________________outPUT_________________________*/
        writePgmImage(output, output_width, output_height, "output.pgm");

    // -----------------------------------------------------------------
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what()
                  << " (code " << e.err() << ")\n";
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
