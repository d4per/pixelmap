/**
 * CUDA kernel implementations for correspondence mapping operations
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Structure for affine transform
struct AffineTransform {
    float m00, m01, m02;
    float m10, m11, m12;
};

// Calculate color difference between two pixels
__device__ float calculateColorDifference(
    const unsigned char* photo1,
    const unsigned char* photo2,
    int x1, int y1,
    int x2, int y2,
    int width1, int height1,
    int width2, int height2,
    int patch_size
) {
    float total_diff = 0.0f;
    int valid_pixels = 0;
    int half_patch = patch_size / 2;
    
    for (int dy = -half_patch; dy <= half_patch; dy++) {
        for (int dx = -half_patch; dx <= half_patch; dx++) {
            // Calculate pixel coordinates in both photos
            int px1 = x1 + dx;
            int py1 = y1 + dy;
            int px2 = x2 + dx;
            int py2 = y2 + dy;
            
            // Check bounds
            if (px1 >= 0 && px1 < width1 && py1 >= 0 && py1 < height1 &&
                px2 >= 0 && px2 < width2 && py2 >= 0 && py2 < height2) {
                
                // Calculate pixel indices
                int idx1 = (py1 * width1 + px1) * 4; // RGBA
                int idx2 = (py2 * width2 + px2) * 4;
                
                // Calculate color difference for RGB channels
                for (int c = 0; c < 3; c++) {
                    float diff = fabsf((float)photo1[idx1 + c] - (float)photo2[idx2 + c]);
                    total_diff += diff;
                }
                
                valid_pixels++;
            }
        }
    }
    
    // Normalize by number of valid pixels and channels
    if (valid_pixels > 0) {
        return total_diff / (valid_pixels * 3.0f);
    } else {
        return 255.0f; // Maximum difference if no valid pixels
    }
}

// Kernel to compute correspondence scores for multiple transforms
extern "C" __global__ void computeCorrespondenceScores(
    const unsigned char* source_photo,
    const unsigned char* target_photo,
    int source_width, int source_height,
    int target_width, int target_height,
    const AffineTransform* transforms,
    const float2* points,
    int transform_count,
    float* scores,
    int patch_size
) {
    int transform_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (transform_idx >= transform_count) {
        return;
    }
    
    // Get transform
    AffineTransform t = transforms[transform_idx];
    
    // Initialize score
    float total_diff = 0.0f;
    int valid_samples = 0;
    
    // Sample each point
    const int NUM_SAMPLE_POINTS = 500;
    for (int i = 0; i < NUM_SAMPLE_POINTS && i < transform_count; i++) {
        // Get sample point
        float2 point = points[i];
        
        // Apply transform
        float tx = t.m00 * point.x + t.m01 * point.y + t.m02;
        float ty = t.m10 * point.x + t.m11 * point.y + t.m12;
        
        // Round to nearest pixel
        int source_x = __float2int_rn(point.x);
        int source_y = __float2int_rn(point.y);
        int target_x = __float2int_rn(tx);
        int target_y = __float2int_rn(ty);
        
        // Check bounds
        if (source_x >= 0 && source_x < source_width &&
            source_y >= 0 && source_y < source_height &&
            target_x >= 0 && target_x < target_width &&
            target_y >= 0 && target_y < target_height) {
            
            // Calculate color difference for this point
            float diff = calculateColorDifference(
                source_photo, target_photo,
                source_x, source_y,
                target_x, target_y,
                source_width, source_height,
                target_width, target_height,
                patch_size
            );
            
            // Accumulate difference
            total_diff += diff;
            valid_samples++;
        }
    }
    
    // Compute final score
    if (valid_samples > 0) {
        scores[transform_idx] = total_diff / valid_samples;
    } else {
        scores[transform_idx] = 100000.0f; // Very high score for invalid transforms
    }
}

// Kernel to remove outliers in a dense photo map
extern "C" __global__ void removeOutliers(
    float* map_data,         // Packed map data for the current map [x|y|used]
    const float* other_map_data, // Packed map data for the other map
    int width, int height,
    float max_dist
) {
    // Calculate cell coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate data indices
    int idx = y * width + x;
    int cell_count = width * height;
    
    // Check if cell is used
    if (map_data[idx + 2 * cell_count] < 0.5f) {
        return; // Skip unused cells
    }
    
    // Get mapped coordinates
    float mapped_x = map_data[idx];
    float mapped_y = map_data[idx + cell_count];
    
    // Round to nearest integer
    int mx = __float2int_rd(mapped_x);
    int my = __float2int_rd(mapped_y);
    
    // Check bounds
    if (mx < 0 || mx >= width || my < 0 || my >= height) {
        // Out of bounds, mark as unused
        map_data[idx + 2 * cell_count] = 0.0f;
        return;
    }
    
    // Get reverse mapping
    int other_idx = my * width + mx;
    float other_x = other_map_data[other_idx];
    float other_y = other_map_data[other_idx + cell_count];
    float other_used = other_map_data[other_idx + 2 * cell_count];
    
    // Check if other cell is used
    if (other_used < 0.5f) {
        // Other cell not used, mark as unused
        map_data[idx + 2 * cell_count] = 0.0f;
        return;
    }
    
    // Calculate distance to verify consistency
    float dx = other_x - (float)x;
    float dy = other_y - (float)y;
    float dist = sqrtf(dx * dx + dy * dy);
    
    // Check if distance exceeds threshold
    if (dist > max_dist) {
        // Mark as outlier (unused)
        map_data[idx + 2 * cell_count] = 0.0f;
    }
}

// Kernel to smooth grid points
extern "C" __global__ void smoothGridPoints(
    const float* in_x,
    const float* in_y,
    const float* in_used,
    float* out_x,
    float* out_y,
    float* out_used,
    int width, int height
) {
    // Calculate cell coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate data index
    int idx = y * width + x;
    
    // Copy used flag directly
    out_used[idx] = in_used[idx];
    
    // Skip if cell is not used or on the boundary
    if (in_used[idx] < 0.5f || x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        out_x[idx] = in_x[idx];
        out_y[idx] = in_y[idx];
        return;
    }
    
    // Get neighbor indices
    int left = idx - 1;
    int right = idx + 1;
    int up = idx - width;
    int down = idx + width;
    
    // Check horizontal neighbors
    bool horizontal_valid = in_used[left] > 0.5f && in_used[right] > 0.5f;
    
    // Check vertical neighbors
    bool vertical_valid = in_used[up] > 0.5f && in_used[down] > 0.5f;
    
    // Smooth based on valid neighbors
    if (horizontal_valid && vertical_valid) {
        // Average all four neighbors
        out_x[idx] = (in_x[left] + in_x[right] + in_x[up] + in_x[down]) / 4.0f;
        out_y[idx] = (in_y[left] + in_y[right] + in_y[up] + in_y[down]) / 4.0f;
    } else if (horizontal_valid) {
        // Average horizontal neighbors
        out_x[idx] = (in_x[left] + in_x[right]) / 2.0f;
        out_y[idx] = (in_y[left] + in_y[right]) / 2.0f;
    } else if (vertical_valid) {
        // Average vertical neighbors
        out_x[idx] = (in_x[up] + in_x[down]) / 2.0f;
        out_y[idx] = (in_y[up] + in_y[down]) / 2.0f;
    } else {
        // No valid neighbors, keep original values
        out_x[idx] = in_x[idx];
        out_y[idx] = in_y[idx];
    }
}

// REVIEW SUGGESTIONS:
// • Document each kernel’s purpose and parameter ranges.
// • Improve error message propagation from kernel launch failures.
