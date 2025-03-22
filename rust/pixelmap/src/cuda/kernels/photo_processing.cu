#include <device_launch_parameters.h>

// Bicubic interpolation helper function
__device__ float bicubic(float x) {
    const float a = -0.5f;
    x = fabsf(x);
    
    if (x < 1.0f) {
        return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
    } else {
        return 0.0f;
    }
}

// Kernel to scale a photo using bicubic interpolation
extern "C" __global__ void scale_bicubic(
    const unsigned char* input,
    unsigned char* output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) {
        return;
    }
    
    // Calculate scaling factors
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    
    // Calculate source position (center of pixel)
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    // Compute the base coordinates for interpolation
    int x0 = floorf(src_x);
    int y0 = floorf(src_y);
    
    // Calculate fractional parts
    float dx = src_x - x0;
    float dy = src_y - y0;
    
    // Initialize accumulators for each channel
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
    float weight_sum = 0.0f;
    
    // Compute the bicubic interpolation
    for (int j = -1; j <= 2; j++) {
        float y_weight = bicubic(j - dy);
        
        for (int i = -1; i <= 2; i++) {
            float x_weight = bicubic(i - dx);
            float weight = x_weight * y_weight;
            
            // Compute source position with clamping to edge
            int sx = max(0, min(src_width - 1, x0 + i));
            int sy = max(0, min(src_height - 1, y0 + j));
            
            // Get the source pixel
            int src_idx = (sy * src_width + sx) * 4;
            
            // Accumulate weighted contributions
            r += input[src_idx] * weight;
            g += input[src_idx + 1] * weight;
            b += input[src_idx + 2] * weight;
            a += input[src_idx + 3] * weight;
            
            weight_sum += weight;
        }
    }
    
    // Normalize and clamp the results
    if (weight_sum > 0.0f) {
        r = max(0.0f, min(255.0f, r / weight_sum));
        g = max(0.0f, min(255.0f, g / weight_sum));
        b = max(0.0f, min(255.0f, b / weight_sum));
        a = max(0.0f, min(255.0f, a / weight_sum));
    }
    
    // Write the output pixel
    int dst_idx = (y * dst_width + x) * 4;
    output[dst_idx] = (unsigned char)r;
    output[dst_idx + 1] = (unsigned char)g;
    output[dst_idx + 2] = (unsigned char)b;
    output[dst_idx + 3] = (unsigned char)a;
}

// Kernel to apply Gaussian blur
extern "C" __global__ void gaussian_blur(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate kernel radius based on sigma (3*sigma covers ~99.7%)
    int radius = (int)(3.0f * sigma + 0.5f);
    
    // Initialize accumulators for each channel
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
    float weight_sum = 0.0f;
    
    // Compute the Gaussian weighted sum
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            // Calculate Gaussian weight
            float dist2 = (float)(i * i + j * j);
            float weight = expf(-dist2 / (2.0f * sigma * sigma));
            
            // Compute source position with clamping to edge
            int sx = max(0, min(width - 1, x + i));
            int sy = max(0, min(height - 1, y + j));
            
            // Get the source pixel
            int src_idx = (sy * width + sx) * 4;
            
            // Accumulate weighted contributions
            r += input[src_idx] * weight;
            g += input[src_idx + 1] * weight;
            b += input[src_idx + 2] * weight;
            a += input[src_idx + 3] * weight;
            
            weight_sum += weight;
        }
    }
    
    // Normalize the results
    if (weight_sum > 0.0f) {
        r /= weight_sum;
        g /= weight_sum;
        b /= weight_sum;
        a /= weight_sum;
    }
    
    // Write the output pixel
    int dst_idx = (y * width + x) * 4;
    output[dst_idx] = (unsigned char)r;
    output[dst_idx + 1] = (unsigned char)g;
    output[dst_idx + 2] = (unsigned char)b;
    output[dst_idx + 3] = (unsigned char)a;
}

// REVIEW SUGGESTIONS:
// • Add comments describing the shared memory layout and how "bicubic" is implemented.
// • Verify that boundary clamping uses device-safe functions (e.g. __saturatef if available).
