/**
 * CUDA kernel implementation for circular feature descriptor matching
 */
#include <cuda_runtime.h>

/**
 * Kernel to find the nearest descriptor from set 1 for a single descriptor from set 2
 */
extern "C" __global__ void find_nearest_descriptor(
    const int64_t* descriptors1,   // All descriptors from image 1
    const int64_t* descriptors2,   // All descriptors from image 2
    float* distances,              // Output distances
    int* match_indices,            // Output match indices
    int num_desc1,                 // Number of descriptors in set 1
    int desc2_idx,                 // Current descriptor index from set 2
    int dim                        // Dimension of feature vectors
) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one descriptor from image 1
    if (idx1 < num_desc1) {
        float distance_sum = 0.0f;
        
        // Calculate squared Euclidean distance
        for (int d = 0; d < dim; d++) {
            int64_t diff = descriptors1[idx1 * dim + d] - descriptors2[desc2_idx * dim + d];
            distance_sum += diff * diff;
        }
        
        // Store distance in shared memory for reduction
        __shared__ float shared_distances[256];
        __shared__ int shared_indices[256];
        
        int tid = threadIdx.x;
        shared_distances[tid] = distance_sum;
        shared_indices[tid] = idx1;
        
        __syncthreads();
        
        // Parallel reduction to find minimum distance
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (shared_distances[tid] > shared_distances[tid + stride]) {
                    shared_distances[tid] = shared_distances[tid + stride];
                    shared_indices[tid] = shared_indices[tid + stride];
                }
            }
            __syncthreads();
        }
        
        // Thread 0 writes block result
        if (tid == 0) {
            int block_idx = blockIdx.x;
            distances[block_idx] = shared_distances[0];
            match_indices[block_idx] = shared_indices[0];
        }
    }
}

/**
 * Kernel to find best matches using tiling for memory coherence
 * 
 * This kernel uses shared memory to load tiles of descriptors, enabling
 * efficient memory access patterns and better performance.
 */
extern "C" __global__ void match_descriptors_tiled(
    const int64_t* descriptors1,   // All descriptors from image 1
    const int64_t* descriptors2,   // All descriptors from image 2
    int* match_indices,            // Output match indices
    int num_desc1,                 // Number of descriptors in set 1
    int num_desc2,                 // Number of descriptors in set 2
    int dim                        // Dimension of feature vectors
) {
    // Shared memory for descriptor tiles
    __shared__ int64_t tile1[32][6]; // Assuming 32 descriptors per tile, 6 dimensions
    __shared__ int64_t tile2[32][6];
    
    // Each thread block processes one tile of the descriptor comparison matrix
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int desc1_idx = bx * blockDim.x + tx;
    int desc2_idx = by * blockDim.y + ty;
    
    // Variables to track best match
    float min_distance = FLT_MAX;
    int best_match_idx = -1;
    
    // Iterate through tiles along diagonal of comparison matrix
    for (int tile = 0; tile < (num_desc1 + blockDim.x - 1) / blockDim.x; tile++) {
        int tile_desc1_idx = tile * blockDim.x + tx;
        
        // Load tile from descriptors1 into shared memory
        if (tile_desc1_idx < num_desc1) {
            for (int d = 0; d < dim; d++) {
                tile1[tx][d] = descriptors1[tile_desc1_idx * dim + d];
            }
        }
        
        // Load tile from descriptors2 into shared memory
        if (desc2_idx < num_desc2) {
            for (int d = 0; d < dim; d++) {
                tile2[ty][d] = descriptors2[desc2_idx * dim + d];
            }
        }
        
        __syncthreads();
        
        // Compare descriptors within the tile
        if (desc2_idx < num_desc2) {
            for (int i = 0; i < blockDim.x; i++) {
                int curr_desc1_idx = tile * blockDim.x + i;
                if (curr_desc1_idx < num_desc1) {
                    // Calculate squared distance
                    float distance_sum = 0.0f;
                    for (int d = 0; d < dim; d++) {
                        int64_t diff = tile1[i][d] - tile2[ty][d];
                        distance_sum += diff * diff;
                    }
                    
                    // Update best match
                    if (distance_sum < min_distance) {
                        min_distance = distance_sum;
                        best_match_idx = curr_desc1_idx;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write out best match for this descriptor2
    if (desc2_idx < num_desc2 && tx == 0) {
        match_indices[desc2_idx] = best_match_idx;
    }
}

/**
 * Alternative tiled kernel with better memory coalescing
 */
extern "C" __global__ void match_descriptors_optimized(
    const int64_t* descriptors1,
    const int64_t* descriptors2,
    int* match_indices,
    float* match_distances,
    int num_desc1,
    int num_desc2,
    int dim
) {
    // Shared memory size optimized for the GPU architecture
    extern __shared__ int64_t shared_mem[];
    
    // Split shared memory between the two descriptor sets
    int64_t* tile1 = &shared_mem[0];
    int64_t* tile2 = &shared_mem[blockDim.x * dim];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int desc2_idx = blockIdx.y * blockDim.y + ty;
    
    float min_distance = FLT_MAX;
    int best_match = -1;
    
    // Process tiles of descriptors1
    for (int tile_start = 0; tile_start < num_desc1; tile_start += blockDim.x) {
        // Collaborative loading of descriptors1 tile into shared memory
        int desc1_idx = tile_start + tx;
        if (desc1_idx < num_desc1) {
            for (int d = 0; d < dim; d++) {
                tile1[tx * dim + d] = descriptors1[desc1_idx * dim + d];
            }
        }
        
        // Collaborative loading of descriptors2 tile into shared memory
        if (desc2_idx < num_desc2) {
            for (int d = 0; d < dim; d++) {
                tile2[ty * dim + d] = descriptors2[desc2_idx * dim + d];
            }
        }
        
        __syncthreads();
        
        // Compare current descriptor2 with all descriptors1 in this tile
        if (desc2_idx < num_desc2) {
            for (int i = 0; i < blockDim.x && (tile_start + i) < num_desc1; i++) {
                float distance = 0.0f;
                
                for (int d = 0; d < dim; d++) {
                    int64_t diff = tile1[i * dim + d] - tile2[ty * dim + d];
                    distance += diff * diff;
                }
                
                if (distance < min_distance) {
                    min_distance = distance;
                    best_match = tile_start + i;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write out the best match
    if (desc2_idx < num_desc2 && tx == 0) {
        match_indices[desc2_idx] = best_match;
        match_distances[desc2_idx] = min_distance;
    }
}
