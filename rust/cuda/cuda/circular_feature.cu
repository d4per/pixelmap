#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>    // for sqrtf, atan2f, etc.
#include <stdint.h> // for int32_t, int64_t

// Example descriptor (mirrors your Rust struct layout)
struct CircularFeatureDescriptor {
    uint16_t center_x;
    uint16_t center_y;
    float total_angle;
    float total_radius;
    int32_t sum_red;
    int32_t sum_green;
    int32_t sum_blue;
    float aligned_red_x;
    float aligned_red_y;
    float aligned_green_x;
    float aligned_green_y;
    float aligned_blue_x;
    float aligned_blue_y;
    int64_t feature_vector[6];
};

__global__ void computeCircularFeatureDescriptorKernel(
    const unsigned char* d_in,
    int width,
    int height,
    int radius,
    CircularFeatureDescriptor* d_out
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    long sum_red = 0, sum_green = 0, sum_blue = 0;
    long sum_weighted_red_x = 0, sum_weighted_red_y = 0;
    long sum_weighted_green_x = 0, sum_weighted_green_y = 0;
    long sum_weighted_blue_x = 0, sum_weighted_blue_y = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        float row_f = sqrtf((float)(radius*radius - dy*dy));
        int row_radius = (int)roundf(row_f);

        for (int dx = -row_radius; dx <= row_radius; dx++) {
            int wrapped_x = (x + dx + width) % width;
            int wrapped_y = (y + dy + height) % height;

            size_t pix_idx = 4UL * (wrapped_x + (size_t)wrapped_y * width);
            unsigned char r = d_in[pix_idx + 0];
            unsigned char g = d_in[pix_idx + 1];
            unsigned char b = d_in[pix_idx + 2];
            // alpha = d_in[pix_idx + 3];

            sum_red += r; sum_green += g; sum_blue += b;
            sum_weighted_red_x   += (long)dx * r;
            sum_weighted_red_y   += (long)dy * r;
            sum_weighted_green_x += (long)dx * g;
            sum_weighted_green_y += (long)dy * g;
            sum_weighted_blue_x  += (long)dx * b;
            sum_weighted_blue_y  += (long)dy * b;
        }
    }

    long sum_all = sum_red + sum_green + sum_blue;

    float f_sum_red = (float)sum_red;
    float f_sum_green = (float)sum_green;
    float f_sum_blue = (float)sum_blue;
    float f_sum_all = (float)sum_all;

    float red_cm_x = (f_sum_red == 0.f) ? 0.f : (sum_weighted_red_x / f_sum_red);
    float red_cm_y = (f_sum_red == 0.f) ? 0.f : (sum_weighted_red_y / f_sum_red);
    float red_angle = (f_sum_red == 0.f) ? 0.f : atan2f(red_cm_y, red_cm_x);
    float red_radius = sqrtf(red_cm_x*red_cm_x + red_cm_y*red_cm_y);

    float green_cm_x = (f_sum_green == 0.f) ? 0.f : (sum_weighted_green_x / f_sum_green);
    float green_cm_y = (f_sum_green == 0.f) ? 0.f : (sum_weighted_green_y / f_sum_green);
    float green_angle = (f_sum_green == 0.f) ? 0.f : atan2f(green_cm_y, green_cm_x);
    float green_radius = sqrtf(green_cm_x*green_cm_x + green_cm_y*green_cm_y);

    float blue_cm_x = (f_sum_blue == 0.f) ? 0.f : (sum_weighted_blue_x / f_sum_blue);
    float blue_cm_y = (f_sum_blue == 0.f) ? 0.f : (sum_weighted_blue_y / f_sum_blue);
    float blue_angle = (f_sum_blue == 0.f) ? 0.f : atan2f(blue_cm_y, blue_cm_x);
    float blue_radius = sqrtf(blue_cm_x*blue_cm_x + blue_cm_y*blue_cm_y);

    float total_cm_x = (f_sum_all == 0.f) ? 0.f
        : (float)(sum_weighted_red_x + sum_weighted_green_x + sum_weighted_blue_x) / f_sum_all;
    float total_cm_y = (f_sum_all == 0.f) ? 0.f
        : (float)(sum_weighted_red_y + sum_weighted_green_y + sum_weighted_blue_y) / f_sum_all;

    float total_angle = (f_sum_all == 0.f) ? 0.f : atan2f(total_cm_y, total_cm_x);
    float total_radius = sqrtf(total_cm_x*total_cm_x + total_cm_y*total_cm_y);

    float red_diff_angle = red_angle - total_angle;
    float green_diff_angle = green_angle - total_angle;
    float blue_diff_angle = blue_angle - total_angle;

    float aligned_red_x   = cosf(red_diff_angle)   * red_radius;
    float aligned_red_y   = sinf(red_diff_angle)   * red_radius;
    float aligned_green_x = cosf(green_diff_angle) * green_radius;
    float aligned_green_y = sinf(green_diff_angle) * green_radius;
    float aligned_blue_x  = cosf(blue_diff_angle)  * blue_radius;
    float aligned_blue_y  = sinf(blue_diff_angle)  * blue_radius;

    size_t out_idx = (size_t)y * width + (size_t)x;
    CircularFeatureDescriptor &desc = d_out[out_idx];

    desc.center_x = (uint16_t)x;
    desc.center_y = (uint16_t)y;
    desc.total_angle = total_angle;
    desc.total_radius = total_radius;
    desc.sum_red = (int32_t)sum_red;
    desc.sum_green = (int32_t)sum_green;
    desc.sum_blue = (int32_t)sum_blue;
    desc.aligned_red_x = aligned_red_x;
    desc.aligned_red_y = aligned_red_y;
    desc.aligned_green_x = aligned_green_x;
    desc.aligned_green_y = aligned_green_y;
    desc.aligned_blue_x = aligned_blue_x;
    desc.aligned_blue_y = aligned_blue_y;

    desc.feature_vector[0] = (int64_t)llrintf(aligned_red_x   * 100.f);
    desc.feature_vector[1] = (int64_t)llrintf(aligned_red_y   * 100.f);
    desc.feature_vector[2] = (int64_t)llrintf(aligned_green_x * 100.f);
    desc.feature_vector[3] = (int64_t)llrintf(aligned_green_y * 100.f);
    desc.feature_vector[4] = (int64_t)llrintf(aligned_blue_x  * 100.f);
    desc.feature_vector[5] = (int64_t)llrintf(aligned_blue_y  * 100.f);
}

extern "C" {

    char* alloc_cuda_memory(long bytes) {
        char* x;
        cudaMallocManaged(&x, bytes);
        return x;
    }

    void free_cuda_memory(char* x) {
        cudaFree(x);
    }

    void sync_cuda_device() {
        cudaDeviceSynchronize();
    }

    void computeDescriptorsOnGPU(
        const unsigned char* d_in,  // device pointer
        int width,
        int height,
        int radius,
        CircularFeatureDescriptor* d_out // device pointer
    ) {
        dim3 blockDim(8, 8);
        dim3 gridDim( (width + blockDim.x - 1) / blockDim.x,
                      (height + blockDim.y - 1) / blockDim.y );

        computeCircularFeatureDescriptorKernel<<<gridDim, blockDim>>>(d_in, width, height, radius, d_out);
        cudaDeviceSynchronize();
    }
}
