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
#include "..\conv_2d_gpu\utilities.cpp"

int main()
{
    /*------------------ INPUT ------------------*/
    FilePaths paths = loadPathsFromConfig("data/in/config_lab_proj.txt");

    int input_width, input_height, channel_count;
    std::vector<float> input_img1 = readPngImage(paths.inputImagePath, input_width, input_height, channel_count);
    std::vector<float> input_img2 = readPngImage(paths.kernelFilePath, input_width, input_height, channel_count);
    std::vector<float> output(static_cast<size_t>(input_width)*input_height * channel_count);

    // ------------------ gpu init ------------------
    cl::CommandQueue queue;
    cl::Device device;
    cl::Context context = makeContextAndQueue(&queue, &device);
    const std::string kernel_source_code = loadKernelSource();
    cl::Program program(context, kernel_source_code);
    program.build({device});
    cl::Kernel kernel(program, "kernel_app");

    cl::Buffer d_input1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * input_img1.size(), input_img1.data());
    cl::Buffer d_input2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * input_img2.size(), input_img2.data());
    cl::Buffer d_output(context, CL_MEM_READ_WRITE,
                        sizeof(float) * output.size(), nullptr);
    kernel.setArg(0, d_input1);
    kernel.setArg(1, d_input2);
    kernel.setArg(2, d_output);

    // ------------------ run gpu convolution ------------------
    std::cout << "\nbeginning lab test\n";
    // Update NDRange to include channels
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_width*input_height*channel_count));
    std::cout << "finished lab test\n";
    queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data());
    std::cout << "finished reading output off vram\n";

    // ------------------ result -------------------------
    writePngImage(output, input_width, input_height, channel_count, paths.outputImagePath);
}
