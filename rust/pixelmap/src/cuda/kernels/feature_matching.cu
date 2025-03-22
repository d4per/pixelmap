/**
 * CUDA kernel implementations for feature matching operations
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Packed descriptor format (7 floats per descriptor)
struct PackedDescriptor {
    float center_x;
    float center_y;
    float total_angle;
    float r_weight;
    float g_weight;
    float b_weight;
    float intensity;
};

// Calculate squared Euclidean distance between two descriptors
__device__ float calculate_descriptor_distance(
    const PackedDescriptor& desc1,
    const PackedDescriptor& desc2
) {
    // Weighted distance calculation
    const float position_weight = 0.2f;
    const float angle_weight = 0.4f;
    const float color_weight = 0.3f;
    const float intensity_weight = 0.1f;
    
    // Position distance (normalized by expected proximity)
    float dx = (desc1.center_x - desc2.center_x) / 100.0f;
    float dy = (desc1.center_y - desc2.center_y) / 100.0f;
    float position_dist = dx*dx + dy*dy;
    
    // Angle distance (handles circular nature of angles)
    float angle_diff = desc1.total_angle - desc2.total_angle;
    // Normalize to [-PI, PI]
    while (angle_diff > 3.14159f) angle_diff -= 6.28318f;
    while (angle_diff < -3.14159f) angle_diff += 6.28318f;
    float angle_dist = angle_diff * angle_diff;
    
    // Color distance
    float dr = desc1.r_weight - desc2.r_weight;
    float dg = desc1.g_weight - desc2.g_weight;
    float db = desc1.b_weight - desc2.b_weight;
    float color_dist = dr*dr + dg*dg + db*db;
    
    // Intensity distance
    float di = desc1.intensity - desc2.intensity;
    float intensity_dist = di*di;
    
    // Weighted sum
    return position_weight * position_dist +
           angle_weight * angle_dist +
           color_weight * color_dist +
           intensity_weight * intensity_dist;
}

/**
 * Kernel to compute feature descriptor distance between two sets of descriptors
 */
extern "C" __global__ void compute_descriptor_distances(
    const float* descriptors1, 
    const float* descriptors2,
    float* distances,
    int descriptor_size,
    int num_descriptors1,
    int num_descriptors2
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx1 < num_descriptors1 && idx2 < num_descriptors2) {
        float sum = 0.0f;
        
        for (int i = 0; i < descriptor_size; i++) {
            float diff = descriptors1[idx1 * descriptor_size + i] - 
                         descriptors2[idx2 * descriptor_size + i];
            sum += diff * diff;
        }
        
        distances[idx1 * num_descriptors2 + idx2] = sqrtf(sum);
    }
}

// Kernel to find nearest descriptors using brute force approach
extern "C" __global__ void find_feature_matches(
    const float* descriptors1,    // Packed descriptor data for first image
    const float* descriptors2,    // Packed descriptor data for second image
    int count1,                  // Number of descriptors in first set
    int count2,                  // Number of descriptors in second set
    int* match_indices,          // Output match indices (-1 if no match)
    float* match_distances,      // Output match distances
    float max_distance           // Maximum acceptable distance
) {
    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (idx >= count1) {
        return;
    }
    
    // Create descriptor for first feature
    PackedDescriptor desc1;
    desc1.center_x = descriptors1[idx];
    desc1.center_y = descriptors1[idx + count1];
    desc1.total_angle = descriptors1[idx + 2 * count1];
    desc1.r_weight = descriptors1[idx + 3 * count1];
    desc1.g_weight = descriptors1[idx + 4 * count1];
    desc1.b_weight = descriptors1[idx + 5 * count1];
    desc1.intensity = descriptors1[idx + 6 * count1];
    
    // Find best match
    float best_distance = max_distance;
    int best_match = -1;
    
    for (int i = 0; i < count2; i++) {
        // Create descriptor for second feature
        PackedDescriptor desc2;
        desc2.center_x = descriptors2[i];
        desc2.center_y = descriptors2[i + count2];
        desc2.total_angle = descriptors2[i + 2 * count2];
        desc2.r_weight = descriptors2[i + 3 * count2];
        desc2.g_weight = descriptors2[i + 4 * count2];
        desc2.b_weight = descriptors2[i + 5 * count2];
        desc2.intensity = descriptors2[i + 6 * count2];
        
        // Calculate distance
        float dist = calculate_descriptor_distance(desc1, desc2);
        
        // Update best match if better
        if (dist < best_distance) {
            best_distance = dist;
            best_match = i;
        }
    }
    
    // Store results
    match_indices[idx] = best_match;
    match_distances[idx] = best_distance;
}

// Optimized kernel using shared memory tiling
extern "C" __global__ void find_feature_matches_tiled(
    const float* descriptors1,    // Packed descriptor data for first image
    const float* descriptors2,    // Packed descriptor data for second image
    int count1,                  // Number of descriptors in first set
    int count2,                  // Number of descriptors in second set
    int* match_indices,          // Output match indices (-1 if no match)
    float* match_distances,      // Output match distances
    float max_distance           // Maximum acceptable distance
) {
    extern __shared__ float shared_descriptors[];
    
    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (idx >= count1) {
        return;
    }
    
    // Create descriptor for first feature
    PackedDescriptor desc1;
    desc1.center_x = descriptors1[idx];
    desc1.center_y = descriptors1[idx + count1];
    desc1.total_angle = descriptors1[idx + 2 * count1];
    desc1.r_weight = descriptors1[idx + 3 * count1];
    desc1.g_weight = descriptors1[idx + 4 * count1];
    desc1.b_weight = descriptors1[idx + 5 * count1];
    desc1.intensity = descriptors1[idx + 6 * count1];
    
    // Find best match
    float best_distance = max_distance;
    int best_match = -1;
    
    // Process descriptors2 in tiles
    const int TILE_SIZE = blockDim.x;
    const int DESCRIPTOR_SIZE = 7;
    
    for (int tile = 0; tile < (count2 + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load a tile of descriptors into shared memory
        int tile_start = tile * TILE_SIZE;
        int tile_end = min(tile_start + TILE_SIZE, count2);
        int tile_offset = threadIdx.x;
        
        if (tile_offset < tile_end - tile_start) {
            int idx2 = tile_start + tile_offset;
            for (int i = 0; i < DESCRIPTOR_SIZE; i++) {
                shared_descriptors[tile_offset + i * TILE_SIZE] = 
                    descriptors2[idx2 + i * count2];
            }
        }
        
        __syncthreads();
        
        // Compare with all descriptors in the tile
        for (int i = 0; i < tile_end - tile_start; i++) {
            PackedDescriptor desc2;
            desc2.center_x = shared_descriptors[i];
            desc2.center_y = shared_descriptors[i + TILE_SIZE];
            desc2.total_angle = shared_descriptors[i + 2 * TILE_SIZE];
            desc2.r_weight = shared_descriptors[i + 3 * TILE_SIZE];
            desc2.g_weight = shared_descriptors[i + 4 * TILE_SIZE];
            desc2.b_weight = shared_descriptors[i + 5 * TILE_SIZE];
            desc2.intensity = shared_descriptors[i + 6 * TILE_SIZE];
            
            // Calculate distance
            float dist = calculate_descriptor_distance(desc1, desc2);
            
            // Update best match if better
            if (dist < best_distance) {
                best_distance = dist;
                best_match = tile_start + i;
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    match_indices[idx] = best_match;
    match_distances[idx] = best_distance;
}
