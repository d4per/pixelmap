#include <device_launch_parameters.h>

// Kernel to interpolate grid points from one size to another
extern "C" __global__ void interpolate_grid(
    const float* src_data,   // Source data in packed format [x|y|used]
    float* dst_data,         // Destination data in packed format
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= dst_width || y >= dst_height) {
        return;
    }
    
    // Calculate scale factors
    float scale_x = (float)(src_width - 1) / (dst_width - 1);
    float scale_y = (float)(src_height - 1) / (dst_height - 1);
    
    // Calculate source coordinates
    float src_x = x * scale_x;
    float src_y = y * scale_y;
    
    // Calculate integer source coordinates and fractional parts
    int x0 = floorf(src_x);
    int y0 = floorf(src_y);
    float fx = src_x - x0;
    float fy = src_y - y0;
    
    // Source cell counts
    int src_cell_count = src_width * src_height;
    
    // Output index
    int dst_idx = y * dst_width + x;
    int dst_cell_count = dst_width * dst_height;
    
    // Clamp to valid source grid coordinates
    x0 = max(0, min(src_width - 2, x0));
    y0 = max(0, min(src_height - 2, y0));
    
    // Get the four corners for bilinear interpolation
    int idx00 = y0 * src_width + x0;
    int idx10 = idx00 + 1;
    int idx01 = idx00 + src_width;
    int idx11 = idx01 + 1;
    
    // Get the used flags
    float used00 = src_data[idx00 + 2 * src_cell_count];
    float used10 = src_data[idx10 + 2 * src_cell_count];
    float used01 = src_data[idx01 + 2 * src_cell_count];
    float used11 = src_data[idx11 + 2 * src_cell_count];
    
    // Only interpolate if all four corners are valid
    if (used00 > 0.5f && used10 > 0.5f && used01 > 0.5f && used11 > 0.5f) {
        // Get X values and interpolate
        float x00 = src_data[idx00];
        float x10 = src_data[idx10];
        float x01 = src_data[idx01];
        float x11 = src_data[idx11];
        
        float x0_interp = x00 * (1.0f - fx) + x10 * fx;
        float x1_interp = x01 * (1.0f - fx) + x11 * fx;
        float x_interp = x0_interp * (1.0f - fy) + x1_interp * fy;
        
        // Get Y values and interpolate
        float y00 = src_data[idx00 + src_cell_count];
        float y10 = src_data[idx10 + src_cell_count];
        float y01 = src_data[idx01 + src_cell_count];
        float y11 = src_data[idx11 + src_cell_count];
        
        float y0_interp = y00 * (1.0f - fx) + y10 * fx;
        float y1_interp = y01 * (1.0f - fx) + y11 * fx;
        float y_interp = y0_interp * (1.0f - fy) + y1_interp * fy;
        
        // Write interpolated values
        dst_data[dst_idx] = x_interp;
        dst_data[dst_idx + dst_cell_count] = y_interp;
        dst_data[dst_idx + 2 * dst_cell_count] = 1.0f;  // Mark as used
    } else {
        // Invalid source data, mark as unused
        dst_data[dst_idx] = 0.0f;
        dst_data[dst_idx + dst_cell_count] = 0.0f;
        dst_data[dst_idx + 2 * dst_cell_count] = 0.0f;  // Mark as unused
    }
}

// REVIEW SUGGESTIONS:
// • Clarify comments on bilinear interpolation and mention assumptions on data layout.
// • Consider refactoring repeated clamping logic into an inline __device__ function.
