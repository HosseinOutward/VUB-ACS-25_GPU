#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cl2.hpp>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

// ─────────────────────────────────────────────────────────────
// Utility: read a whole text file into a std::string
// ─────────────────────────────────────────────────────────────
std::string loadKernelSource(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open kernel file: " + path);

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// -----------------------------------------------------------------------------
// makeContextAndQueue
//   • Picks the first platform that contains at least one GPU
//   • Returns a cl::Context
//   • Also fills in *outQueue and *outDevice (if the pointers are non-null)
// -----------------------------------------------------------------------------
inline cl::Context makeContextAndQueue(cl::CommandQueue* outQueue,
                                       cl::Device*       outDevice)
{
    // 1) Enumerate all OpenCL platforms on the system
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
        throw std::runtime_error("No OpenCL platforms found");

    // 2) Find the first GPU device we can use
    cl::Device device;                // default-constructed => invalid
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
    if (!device())                    // operator() returns underlying cl_device_id
        throw std::runtime_error("No GPU devices found on any platform");

    // 3) Create a context and a command queue for that device
    cl::Context      context({device});
    cl::CommandQueue queue(context, device);

    // 4) Pass them back to the caller
    if (outQueue)  *outQueue  = queue;
    if (outDevice) *outDevice = device;

    return context;                   // RAII: context, queue, device auto-manage resources
}


// ─────────────────────────────────────────────────────────────
int main()
{
    try
    {
        //----------------------------------------------------------------------
        // 1. Context, command-queue and device via your helper
        //----------------------------------------------------------------------
        cl::CommandQueue queue;
        cl::Device       device;
        cl::Context      context = makeContextAndQueue(&queue, &device);

        //----------------------------------------------------------------------
        // 2. Compile the kernel
        //----------------------------------------------------------------------
        const std::string kernel_source_code =
            loadKernelSource("src/test/kernel.ocl");

        cl::Program program(context, kernel_source_code);
        program.build({device});                     // build just for this device
        cl::Kernel kernel(program, "kernel_app");         // entry-point name

        //----------------------------------------------------------------------
        // 3. Host-side data
        //----------------------------------------------------------------------
        const size_t N = 1 << 20;                    // 2^20 = 1M
        std::vector<float> x(N, 2.0f);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = static_cast<float>(i);
        std::vector<float> y(N, 1.0f);
        for (size_t i = 0; i < y.size(); ++i)
            y[i] = .05f*static_cast<float>(i);

        //----------------------------------------------------------------------
        // 4. Device buffers (uploaded immediately via CL_MEM_COPY_HOST_PTR)
        //----------------------------------------------------------------------
        cl::Buffer dX(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * N, x.data());
        cl::Buffer dY(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * N, y.data());

        //----------------------------------------------------------------------
        // 5. Launch kernel
        //----------------------------------------------------------------------
        kernel.setArg(0, .1f);
        kernel.setArg(1, dX);
        kernel.setArg(2, dY);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
        queue.enqueueReadBuffer(dY, CL_TRUE, 0, sizeof(float) * N, y.data());

        std::cout << "y[0] = " << y.front()
                  << "   y[N-1] = " << y.back() << '\n';
    }
    
    // -------------------------------------------------------------------------
    // 6. Error handling
    // -------------------------------------------------------------------------
    catch (const cl::Error& e)
    {
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
