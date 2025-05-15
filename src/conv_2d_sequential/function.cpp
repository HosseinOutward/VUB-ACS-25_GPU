#include <thread>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>

void conv2d_cpu_threaded(const float* in, int in_w, int in_h, int channel_count,
                         const float* filt, int filt_w, int filt_h,
                         float* out, int threads, int stride)
{
    if (stride <= 0) {
        throw std::invalid_argument("Stride must be positive.");
    }
    const int out_w = (in_w - filt_w) / stride + 1;
    const int out_h = (in_h - filt_h) / stride + 1;
    const int in_channel_size = in_w * in_h;
    const int out_channel_size = out_w * out_h;

    if (out_w <= 0 || out_h <= 0) {
        std::cerr << "Warning: Output dimensions are non-positive (out_w: " << out_w << ", out_h: " << out_h 
                  << "). No output will be generated." << std::endl;
        return; 
    }

    // ----------- decide thread count ----------------------------------------
    if (threads <= 0)
        threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads < 1)   // safeguard for very odd systems
        threads = 1;

    // ----------- worker that owns a contiguous row-block --------------------
    auto worker = [&](int y_begin, int y_end, int channel) {
        const float* current_in_channel = in + channel * in_channel_size;
        float* current_out_channel = out + channel * out_channel_size;

        for (int y = y_begin; y < y_end; ++y)
            for (int x = 0; x < out_w; ++x) {
                float sum = 0.f;
                for (int fy = 0; fy < filt_h; ++fy) {
                    const int iy = y * stride + fy; 
                    for (int fx = 0; fx < filt_w; ++fx) {
                        const int ix = x * stride + fx;
                        if (!(iy < in_h && ix < in_w))
                            throw std::out_of_range("Input coordinates out of bounds: (" + std::to_string(iy) + ", " + std::to_string(ix) + ")");

                        sum += current_in_channel[iy * in_w + ix] * filt[fy * filt_w + fx];
                    }
                }
                current_out_channel[y * out_w + x] = sum;
            }
    };

    // ----------- launch threads (or run inline) -----------------------------
    // Process each channel seperately but rows in parallel
    for (int c = 0; c < channel_count; ++c) {
        if (threads == 1) {
            worker(0, out_h, c);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(threads);

            const int rows_per_thread = (out_h + threads - 1) / threads;
            int y0 = 0;
            for (int t = 0; t < threads && y0 < out_h; ++t) {
                int y1 = std::min(out_h, y0 + rows_per_thread);
                pool.emplace_back(worker, y0, y1, c);
                y0 = y1;
            }

            for (auto& th : pool)
                th.join();
        }
    }
}
