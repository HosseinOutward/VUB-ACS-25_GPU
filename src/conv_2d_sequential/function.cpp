#include <thread>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>

inline int index3d(int z, int y, int x, int height, int width) {
    return z * (height) * (width) + y * (width) + x;
}
inline int index4d(int k, int z, int y, int x, int depth, int height, int width) {
    return k * (depth) * (height) * (width) + index3d(z, y, x, height, width);
}

void conv2d_cpu_threaded(
    std::vector<unsigned char>& in, // (w h img_c)
    int in_w, int in_h, int channel_count,
    std::vector<float>& filt, // (fw fh img_c fc)
    int filt_w, int filt_h, int filt_c,
    std::vector<float>& out, // (ow oh fc)
    int threads, int stride, int out_w, int out_h)
{
    // ----------- decide thread count ----------------------------------------
    if (threads <= 0)
        threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads < 1)
        threads = 1;

    // ----------- worker that owns a contiguous row-block --------------------
    auto worker = [&](int out_y_begin, int out_y_end) {
        for (int out_y = out_y_begin; out_y < out_y_end; ++out_y)
            for (int out_x = 0; out_x < out_w; ++out_x)

                for (int fc = 0; fc < filt_c; ++fc) {
                    float sum = 0.0f;

                    for (int in_c = 0; in_c < channel_count; ++in_c)
                        for (int filt_y = 0; filt_y < filt_h; ++filt_y)
                            for (int filt_x = 0; filt_x < filt_w; ++filt_x) {
                                int in_x = out_x * stride + filt_x;
                                int in_y = out_y * stride + filt_y;
                                sum += (float)in[index3d(in_c, in_y, in_x, in_h, in_w)] / 255.f *
                                       filt[index4d(fc, in_c, filt_y, filt_x, channel_count, filt_h, filt_w)];
                            }

                    out[index3d(fc, out_y, out_x, out_h, out_w)] = sum;
                }
    };

    // ----------- launch threads (or run inline) -----------------------------
    // Process each channel seperately but rows in parallel
    if (threads == 1)
        worker(0, out_h);

    else {
        std::vector<std::thread> pool;
        pool.reserve(threads);

        const int rows_per_thread = (out_h + threads-1) / threads;
        for (int y0 = 0; y0 < out_h; y0+=rows_per_thread)
            pool.emplace_back(worker, y0, std::min(out_h, y0+rows_per_thread));

        for (auto& th : pool)
            th.join();
    }
}
