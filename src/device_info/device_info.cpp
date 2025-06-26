// g++ -std=c++17 query_local_limits.cpp -lOpenCL && ./a.out
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    cl_uint pcount;
    clGetPlatformIDs(0, nullptr, &pcount);
    std::vector<cl_platform_id> plats(pcount);
    clGetPlatformIDs(pcount, plats.data(), nullptr);

    // pick first GPU device we find
    cl_device_id dev = nullptr;
    for (auto p : plats) {
        std::cout << "Platform: ";
        size_t sz; clGetPlatformInfo(p, CL_PLATFORM_NAME, 0, nullptr, &sz);
        std::vector<char> buf(sz);
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sz, buf.data(), nullptr);
        std::cout << buf.data() << '\n';
        
        cl_uint dcount;
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &dcount);
        if (buf.data() == std::string("NVIDIA CUDA") && dcount) 
            { clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr); break; }
    }
    if (!dev) { std::cerr << "No GPU device found\n"; return 1; }

    auto gi = [&](cl_device_info param){
        size_t sz; clGetDeviceInfo(dev, param, 0, nullptr, &sz);
        std::vector<char> buf(sz);
        clGetDeviceInfo(dev, param, sz, buf.data(), nullptr);
        return buf;
    };

    cl_uint computeUnits = *reinterpret_cast<cl_uint*>(gi(CL_DEVICE_MAX_COMPUTE_UNITS).data());

    std::string name(gi(CL_DEVICE_NAME).data());
    cl_device_type dtype = *reinterpret_cast<cl_device_type*>(gi(CL_DEVICE_TYPE).data());
    std::cout << "Device name: " << name << '\n';
    std::cout << "Device type: "
            << ((dtype & CL_DEVICE_TYPE_GPU) ? "GPU" :
                (dtype & CL_DEVICE_TYPE_CPU) ? "CPU" : "Other")
            << '\n';
    std::cout << "Compute units (C): " << computeUnits << "\n\n";


    size_t   maxWGSize   = *reinterpret_cast<size_t*>(gi(CL_DEVICE_MAX_WORK_GROUP_SIZE).data());
    size_t   maxWIDims[3];
    std::memcpy(maxWIDims, gi(CL_DEVICE_MAX_WORK_ITEM_SIZES).data(), sizeof(maxWIDims));
    cl_ulong localMem    = *reinterpret_cast<cl_ulong*>(gi(CL_DEVICE_LOCAL_MEM_SIZE).data());

    std::cout << "=== Device-level ===\n";
    std::cout << "Compute units (C):           " << computeUnits        << '\n';
    std::cout << "Local mem per CU (L_dev):    " << localMem/1024 << " KiB\n";
    std::cout << "Max WG size (WG_max):        " << maxWGSize          << '\n';
    std::cout << "Max WI dims (x,y,z):         "
              << maxWIDims[0] << ' ' << maxWIDims[1] << ' ' << maxWIDims[2] << '\n';

    /* --- kernel-specific part ------------------------------------------------ */
    // build a tiny sample kernel that uses dynamic local memory
    const char* src = R"(
        __kernel void sample(__local float* scratch) {
            const uint lid = get_local_id(0);
            scratch[lid] = lid;
        })";

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    cl_program prg = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    clBuildProgram(prg, 1, &dev, "", nullptr, nullptr);
    cl_kernel k = clCreateKernel(prg, "sample", &err);

    // request dynamic local mem of N * sizeof(float)
    const size_t WU   = 128;                      // pretend each WG has 128 work-items
    const size_t Ldyn_perWI = sizeof(float);      // bytes per work-item we need
    const size_t Ldyn_perWG = WU * Ldyn_perWI;
    clSetKernelArg(k, 0, Ldyn_perWG, nullptr);

    /* query kernel-specific resource usage */
    size_t   WG_kernel;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(WG_kernel), &WG_kernel, nullptr);

    cl_ulong L_static;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE,
                             sizeof(L_static), &L_static, nullptr);

    std::cout << "\n=== Kernel-specific ===\n";
    std::cout << "Compiler-max WG size (WG_kernel):   " << WG_kernel      << '\n';
    std::cout << "Static local mem (L_static):        " << L_static       << " B\n";
    std::cout << "Dynamic local mem (L_dyn per WG):   " << Ldyn_perWG     << " B\n";

    /* derived limits */
    size_t WI_max_local = (localMem - L_static) / Ldyn_perWI;
    size_t WG_max      = maxWGSize;
    size_t WG_size_cap  = std::min({WG_max, WG_kernel, WI_max_local});

    size_t WG_perCU     = (localMem) / (L_static + Ldyn_perWG);
    size_t WG_device    = WG_perCU * computeUnits;

    std::cout << "\n=== Derived ===\n";
    std::cout << "Max WIs per WG limited by local mem: " << WI_max_local   << '\n';
    std::cout << "Usable WG size with all caps applied: " << WG_size_cap    << '\n';
    std::cout << "Resident WGs per CU (local-mem-only): " << WG_perCU       << '\n';
    std::cout << "Theoretical max WGs on device:        " << WG_device      << '\n';


    // core count and speed information
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    cl_device_id device = nullptr;
    for (auto p : platforms) {
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) == CL_SUCCESS && deviceCount > 0) {
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            break;
        }
    }
    if (!device) {
        std::cerr << "No GPU device found\n";
        return 1;
    }

    // Example for a Pascal GPU with 128 cores per SM:
    unsigned int estimatedCores = computeUnits * 128;

    std::cout << "Compute units: " << computeUnits << "\n";
    std::cout << "Estimated total cores: " << estimatedCores << "\n";

    size_t maxFreq;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxFreq), &maxFreq, nullptr);
    std::cout << "Max clock frequency: " << maxFreq << " MHz\n";

    std::cout << "Estimated max speed: "
        << (static_cast<double>(maxFreq) * estimatedCores * 2.0 / 1e6)
        << " TFlops\n";

}
