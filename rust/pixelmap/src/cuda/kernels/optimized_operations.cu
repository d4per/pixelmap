#include <device_launch_parameters.h>

// Structure for packed map data (structure of arrays format)
// [x values | y values | used flags]

// Optimized version of removeOutliers that works with packed structure-of-arrays data
extern "C" __global__ void remove_outliers_optimized(
    float* mapData,        // Packed data [x values | y values | used flags]
    const float* otherMapData, // Packed data [x values | y values | used flags]
    int width, int height,
    float maxDist
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
    int cellCount = width * height;
    
    // Load data from current map (structure of arrays layout)
    float mapX = mapData[idx];
    float mapY = mapData[idx + cellCount];
    float mapUsed = mapData[idx + 2 * cellCount];
    
    // Skip if not used
    if (mapUsed < 0.5f) {
        return;
    }
    
    // Get target position (integer coordinates)
    int tx = __float2int_rd(mapX);
    int ty = __float2int_rd(mapY);
    
    // Check bounds
    if (tx < 0 || tx >= width || ty < 0 || ty >= height) {
        mapData[idx + 2 * cellCount] = 0.0f; // Mark as not used
        return;
    }
    
    // Get other map index
    int tidx = ty * width + tx;
    
    // Check if other cell is used
    float otherUsed = otherMapData[tidx + 2 * cellCount];
    if (otherUsed < 0.5f) {
        mapData[idx + 2 * cellCount] = 0.0f; // Mark as not used
        return;
    }
    
    // Get reverse mapping
    float rx = otherMapData[tidx];
    float ry = otherMapData[tidx + cellCount];
    
    // Calculate distance
    float dx = rx - (float)x;
    float dy = ry - (float)y;
    float dist = sqrtf(dx * dx + dy * dy);
    
    // Mark as outlier if distance is too large
    if (dist > maxDist) {
        mapData[idx + 2 * cellCount] = 0.0f;
    }
}

// Optimized version of smooth grid points that works with packed structure-of-arrays data
// Uses shared memory for faster access to neighboring cells
extern "C" __global__ void smooth_grid_points_optimized(
    const float* inData,   // Packed data [x values | y values | used flags]
    float* outData,        // Output packed data
    int width, int height
) {
    // Shared memory for caching (includes halo cells)
    extern __shared__ float sharedData[];
    
    // Tile dimensions (without halo)
    const int TILE_WIDTH = blockDim.x;
    const int TILE_HEIGHT = blockDim.y;
    
    // Padded tile dimensions (with halo)
    const int PADDED_TILE_WIDTH = TILE_WIDTH + 2;
    const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2;
    
    // Shared memory layout: [x values | y values | used flags]
    float* sharedX = sharedData;
    float* sharedY = &sharedData[PADDED_TILE_WIDTH * PADDED_TILE_HEIGHT];
    float* sharedUsed = &sharedData[2 * PADDED_TILE_WIDTH * PADDED_TILE_HEIGHT];
    
    // Block starting position
    int blockStartX = blockIdx.x * TILE_WIDTH - 1;
    int blockStartY = blockIdx.y * TILE_HEIGHT - 1;
    
    // Local thread position
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    
    // Global data size
    int cellCount = width * height;
    
    // Collaboratively load data into shared memory (with halo)
    for (int y = ly; y < PADDED_TILE_HEIGHT; y += TILE_HEIGHT) {
        for (int x = lx; x < PADDED_TILE_WIDTH; x += TILE_WIDTH) {
            // Calculate global coordinates
            int gx = blockStartX + x;
            int gy = blockStartY + y;
            
            // Check bounds
            bool inBounds = (gx >= 0 && gx < width && gy >= 0 && gy < height);
            
            // Calculate indices
            int sharedIdx = y * PADDED_TILE_WIDTH + x;
            int globalIdx = inBounds ? (gy * width + gx) : 0;
            
            // Load data
            if (inBounds) {
                sharedX[sharedIdx] = inData[globalIdx];
                sharedY[sharedIdx] = inData[globalIdx + cellCount];
                sharedUsed[sharedIdx] = inData[globalIdx + 2 * cellCount];
            } else {
                // Default values for out-of-bounds cells
                sharedX[sharedIdx] = 0.0f;
                sharedY[sharedIdx] = 0.0f;
                sharedUsed[sharedIdx] = 0.0f;
            }
        }
    }
    
    // Ensure all threads have loaded data
    __syncthreads();
    
    // Calculate global coordinates
    int gx = blockIdx.x * TILE_WIDTH + lx;
    int gy = blockIdx.y * TILE_HEIGHT + ly;
    
    // Check bounds for output
    if (gx >= width || gy >= height) {
        return;
    }
    
    // Calculate indices
    int globalIdx = gy * width + gx;
    int sharedIdx = (ly + 1) * PADDED_TILE_WIDTH + (lx + 1); // +1 for halo
    
    // Copy used flag directly to output
    outData[globalIdx + 2 * cellCount] = sharedUsed[sharedIdx];
    
    // Skip smoothing if not used
    if (sharedUsed[sharedIdx] < 0.5f) {
        outData[globalIdx] = inData[globalIdx];
        outData[globalIdx + cellCount] = inData[globalIdx + cellCount];
        return;
    }
    
    // Get neighbor indices in shared memory
    int left = sharedIdx - 1;
    int right = sharedIdx + 1;
    int up = sharedIdx - PADDED_TILE_WIDTH;
    int down = sharedIdx + PADDED_TILE_WIDTH;
    
    // Accumulate values from neighbors
    float sumX = sharedX[sharedIdx];
    float sumY = sharedY[sharedIdx];
    int count = 1;
    
    // Check horizontal neighbors
    bool horizontalValid = false;
    if (sharedUsed[left] > 0.5f && sharedUsed[right] > 0.5f) {
        horizontalValid = true;
        sumX += sharedX[left] + sharedX[right];
        sumY += sharedY[left] + sharedY[right];
        count += 2;
    }
    
    // Check vertical neighbors
    bool verticalValid = false;
    if (sharedUsed[up] > 0.5f && sharedUsed[down] > 0.5f) {
        verticalValid = true;
        sumX += sharedX[up] + sharedX[down];
        sumY += sharedY[up] + sharedY[down];
        count += 2;
    }
    
    // Only update if we have valid neighbors
    if (horizontalValid || verticalValid) {
        outData[globalIdx] = sumX / count;
        outData[globalIdx + cellCount] = sumY / count;
    } else {
        // No valid neighbors, copy input to output
        outData[globalIdx] = inData[globalIdx];
        outData[globalIdx + cellCount] = inData[globalIdx + cellCount];
    }
}

// Optimized version of feature matching using tiled memory access patterns
extern "C" __global__ void match_features_tiled(
    const float* descriptors1,    // Packed descriptor data for first image
    const float* descriptors2,    // Packed descriptor data for second image
    int* matchIndices,           // Output match indices
    float* matchDistances,       // Output match distances
    int count1,                  // Number of descriptors in first set
    int count2,                  // Number of descriptors in second set
    int descriptor_size          // Size of each descriptor (typically 7)
) {
    // Shared memory for tile of descriptors from second set
    extern __shared__ float sharedDescriptors[];
    
    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (idx >= count1) {
        return;
    }
    
    // Load descriptor from first set
    float desc1[7]; // Assumes descriptor_size = 7
    for (int i = 0; i < descriptor_size; i++) {
        desc1[i] = descriptors1[idx + i * count1]; // Strided access for AoS->SoA conversion
    }
    
    // Initialize best match
    float bestDistance = 1e9f;
    int bestIdx = -1;
    
    // Process descriptors from second set in tiles
    const int TILE_SIZE = blockDim.x;
    
    for (int tile_start = 0; tile_start < count2; tile_start += TILE_SIZE) {
        // Load tile of descriptors from second set into shared memory
        if (tile_start + threadIdx.x < count2) {
            for (int i = 0; i < descriptor_size; i++) {
                sharedDescriptors[threadIdx.x + i * TILE_SIZE] = 
                    descriptors2[tile_start + threadIdx.x + i * count2];
            }
        }
        
        // Make sure all threads loaded their part of the tile
        __syncthreads();
        
        // Compare descriptor from first set with all descriptors in tile
        int tile_end = min(tile_start + TILE_SIZE, count2);
        for (int j = 0; j < tile_end - tile_start; j++) {
            // Calculate squared distance between descriptors
            float dist = 0.0f;
            
            // Center coordinates
            float dx = desc1[0] - sharedDescriptors[j];
            float dy = desc1[1] - sharedDescriptors[j + TILE_SIZE];
            dist += dx * dx + dy * dy;
            
            // Feature values
            for (int k = 2; k < descriptor_size; k++) {
                float diff = desc1[k] - sharedDescriptors[j + k * TILE_SIZE];
                dist += diff * diff;
            }
            
            // Check if this is a better match
            if (dist < bestDistance) {
                bestDistance = dist;
                bestIdx = tile_start + j;
            }
        }
        
        // Sync before loading next tile
        __syncthreads();
    }
    
    // Store results
    matchIndices[idx] = bestIdx;
    matchDistances[idx] = bestDistance;
}

// Kernel to warp a photo using a dense map with nearest neighbor sampling
extern "C" __global__ void warp_photo(
    const unsigned char* inputPhoto,  // Input photo data (RGBA)
    unsigned char* outputPhoto,       // Output photo data (RGBA)
    const float* mapData,            // Packed map data [x | y | used]
    int width, int height            // Photo dimensions
) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate indices
    int pixelIdx = y * width + x;
    int outputIdx = pixelIdx * 4; // RGBA
    int cellCount = width * height;
    
    // Get mapping data
    float mapX = mapData[pixelIdx];
    float mapY = mapData[pixelIdx + cellCount];
    float mapUsed = mapData[pixelIdx + 2 * cellCount];
    
    // Check if mapping exists
    if (mapUsed < 0.5f) {
        // No mapping, use red color
        outputPhoto[outputIdx] = 255;     // R
        outputPhoto[outputIdx + 1] = 0;   // G
        outputPhoto[outputIdx + 2] = 0;   // B
        outputPhoto[outputIdx + 3] = 255; // A
        return;
    }
    
    // Round to nearest pixel
    int sourceX = __float2int_rn(mapX);
    int sourceY = __float2int_rn(mapY);
    
    // Check bounds
    if (sourceX < 0 || sourceX >= width || sourceY < 0 || sourceY >= height) {
        // Out of bounds, use transparent
        outputPhoto[outputIdx] = 0;     // R
        outputPhoto[outputIdx + 1] = 0; // G
        outputPhoto[outputIdx + 2] = 0; // B
        outputPhoto[outputIdx + 3] = 0; // A
        return;
    }
    
    // Get source pixel
    int sourceIdx = (sourceY * width + sourceX) * 4;
    
    // Copy pixel data
    outputPhoto[outputIdx] = inputPhoto[sourceIdx];         // R
    outputPhoto[outputIdx + 1] = inputPhoto[sourceIdx + 1]; // G
    outputPhoto[outputIdx + 2] = inputPhoto[sourceIdx + 2]; // B
    outputPhoto[outputIdx + 3] = inputPhoto[sourceIdx + 3]; // A
}

// Kernel to warp a photo using a dense map with bilinear interpolation
extern "C" __global__ void warp_photo_bilinear(
    const unsigned char* inputPhoto,  // Input photo data (RGBA)
    unsigned char* outputPhoto,       // Output photo data (RGBA)
    const float* mapData,            // Packed map data [x | y | used]
    int width, int height            // Photo dimensions
) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate indices
    int pixelIdx = y * width + x;
    int outputIdx = pixelIdx * 4; // RGBA
    int cellCount = width * height;
    
    // Get mapping data
    float mapX = mapData[pixelIdx];
    float mapY = mapData[pixelIdx + cellCount];
    float mapUsed = mapData[pixelIdx + 2 * cellCount];
    
    // Check if mapping exists
    if (mapUsed < 0.5f) {
        // No mapping, use red color
        outputPhoto[outputIdx] = 255;     // R
        outputPhoto[outputIdx + 1] = 0;   // G
        outputPhoto[outputIdx + 2] = 0;   // B
        outputPhoto[outputIdx + 3] = 255; // A
        return;
    }
    
    // Calculate base coordinates and fractional parts for bilinear interpolation
    int x0 = __float2int_rd(mapX);
    int y0 = __float2int_rd(mapY);
    float xFrac = mapX - x0;
    float yFrac = mapY - y0;
    
    // Check bounds
    if (x0 < 0 || x0 + 1 >= width || y0 < 0 || y0 + 1 >= height) {
        // Out of bounds, use transparent
        outputPhoto[outputIdx] = 0;     // R
        outputPhoto[outputIdx + 1] = 0; // G
        outputPhoto[outputIdx + 2] = 0; // B
        outputPhoto[outputIdx + 3] = 0; // A
        return;
    }
    
    // Get source pixels
    int idx00 = (y0 * width + x0) * 4;
    int idx10 = idx00 + 4;
    int idx01 = idx00 + width * 4;
    int idx11 = idx01 + 4;
    
    // Perform bilinear interpolation for each channel
    for (int channel = 0; channel < 4; channel++) {
        float p00 = inputPhoto[idx00 + channel];
        float p10 = inputPhoto[idx10 + channel];
        float p01 = inputPhoto[idx01 + channel];
        float p11 = inputPhoto[idx11 + channel];
        
        // Interpolate
        float p0 = p00 * (1.0f - xFrac) + p10 * xFrac;
        float p1 = p01 * (1.0f - xFrac) + p11 * xFrac;
        float p = p0 * (1.0f - yFrac) + p1 * yFrac;
        
        // Store result
        outputPhoto[outputIdx + channel] = (unsigned char)p;
    }
}

// Kernel for batch transform applications using coalesced memory access
extern "C" __global__ void batch_transform_coalesced(
    const float* transformData,    // Packed transform data [m00|m01|m02|m10|m11|m12] for all transforms
    const float* points,           // Packed point data [x|y] for all points
    float* results,                // Output transformed points
    int transformCount,           // Number of transforms
    int pointCount                // Number of points
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (idx >= transformCount * pointCount) {
        return;
    }
    
    // Calculate transform and point indices
    int transformIdx = idx / pointCount;
    int pointIdx = idx % pointCount;
    
    // Load transform (6 floats per transform)
    float m00 = transformData[transformIdx * 6];
    float m01 = transformData[transformIdx * 6 + 1];
    float m02 = transformData[transformIdx * 6 + 2];
    float m10 = transformData[transformIdx * 6 + 3];
    float m11 = transformData[transformIdx * 6 + 4];
    float m12 = transformData[transformIdx * 6 + 5];
    
    // Load point (2 floats per point)
    float x = points[pointIdx * 2];
    float y = points[pointIdx * 2 + 1];
    
    // Apply transform
    float resultX = m00 * x + m01 * y + m02;
    float resultY = m10 * x + m11 * y + m12;
    
    // Store result
    results[idx * 2] = resultX;
    results[idx * 2 + 1] = resultY;
}

// Add a new optimized kernel that uses shared memory
extern "C" __global__ void gaussian_blur_optimized(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    float sigma
) {
    extern __shared__ unsigned char shared_input[];
    
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local thread position
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    
    // Tile dimensions
    const int TILE_WIDTH = blockDim.x;
    const int TILE_HEIGHT = blockDim.y;
    
    // Calculate kernel radius based on sigma (3*sigma covers ~99.7%)
    int radius = (int)(3.0f * sigma + 0.5f);
    
    // Padded tile dimensions
    const int PADDED_TILE_WIDTH = TILE_WIDTH + 2 * radius;
    const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2 * radius;
    
    // Block starting position (including halo)
    int blockStartX = blockIdx.x * TILE_WIDTH - radius;
    int blockStartY = blockIdx.y * TILE_HEIGHT - radius;
    
    // Collaboratively load data into shared memory (with halo)
    for (int ty = ly; ty < PADDED_TILE_HEIGHT; ty += TILE_HEIGHT) {
        for (int tx = lx; tx < PADDED_TILE_WIDTH; tx += TILE_WIDTH) {
            // Calculate global coordinates
            int gx = blockStartX + tx;
            int gy = blockStartY + ty;
            
            // Clamp to image boundaries
            gx = max(0, min(width - 1, gx));
            gy = max(0, min(height - 1, gy));
            
            // Load input pixel data to shared memory
            int sharedIdx = ty * PADDED_TILE_WIDTH + tx;
            int globalIdx = (gy * width + gx) * 4;
            
            // Load RGBA components
            for (int c = 0; c < 4; c++) {
                shared_input[sharedIdx * 4 + c] = input[globalIdx + c];
            }
        }
    }
    
    // Ensure all data is loaded
    __syncthreads();
    
    // Check output bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate local position in shared memory (with halo offset)
    int localX = lx + radius;
    int localY = ly + radius;
    
    // Initialize accumulators for each channel
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f, a_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Precompute 1/(2*sigma^2) for Gaussian weight calculation
    float inv_2sigma_squared = 1.0f / (2.0f * sigma * sigma);
    
    // Apply Gaussian blur using shared memory
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            // Calculate squared distance for Gaussian weight
            float dist_squared = dx * dx + dy * dy;
            
            // Calculate Gaussian weight: exp(-dist^2/(2*sigma^2))
            float weight = expf(-dist_squared * inv_2sigma_squared);
            
            // Calculate the shared memory index
            int sIdx = (localY + dy) * PADDED_TILE_WIDTH + (localX + dx);
            
            // Accumulate weighted pixel values
            r_sum += shared_input[sIdx * 4] * weight;
            g_sum += shared_input[sIdx * 4 + 1] * weight;
            b_sum += shared_input[sIdx * 4 + 2] * weight;
            a_sum += shared_input[sIdx * 4 + 3] * weight;
            
            weight_sum += weight;
        }
    }
    
    // Normalize and write output
    int outputIdx = (y * width + x) * 4;
    output[outputIdx]     = (unsigned char)(r_sum / weight_sum);
    output[outputIdx + 1] = (unsigned char)(g_sum / weight_sum);
    output[outputIdx + 2] = (unsigned char)(b_sum / weight_sum);
    output[outputIdx + 3] = (unsigned char)(a_sum / weight_sum);
}

// Kernel to warp a photo using a dense map with neighborhood optimization
extern "C" __global__ void warp_photo_optimized(
    const unsigned char* inputPhoto,  // Input photo data (RGBA)
    unsigned char* outputPhoto,       // Output photo data (RGBA)
    const float* mapData,            // Packed map data [x | y | used]
    int width, int height            // Photo dimensions
) {
    // Shared memory for caching input photo tile
    extern __shared__ unsigned char sharedPhoto[];
    
    // Thread position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local thread position
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    
    // Tile dimensions
    const int TILE_WIDTH = blockDim.x;
    const int TILE_HEIGHT = blockDim.y;
    
    // Load a tile of the source photo into shared memory
    // This is speculative loading - we don't yet know which pixels we'll need
    // but by loading a tile, we increase the chance of cache hits
    int tile_x = blockIdx.x * TILE_WIDTH;
    int tile_y = blockIdx.y * TILE_HEIGHT;
    
    // Collaboratively load the tile
    for (int j = ly; j < TILE_HEIGHT; j += blockDim.y) {
        for (int i = lx; i < TILE_WIDTH; i += blockDim.x) {
            int sx = tile_x + i;
            int sy = tile_y + j;
            
            if (sx < width && sy < height) {
                int sharedIdx = (j * TILE_WIDTH + i) * 4; // RGBA
                int globalIdx = (sy * width + sx) * 4;
                
                sharedPhoto[sharedIdx] = inputPhoto[globalIdx];
                sharedPhoto[sharedIdx + 1] = inputPhoto[globalIdx + 1];
                sharedPhoto[sharedIdx + 2] = inputPhoto[globalIdx + 2];
                sharedPhoto[sharedIdx + 3] = inputPhoto[globalIdx + 3];
            }
        }
    }
    
    __syncthreads(); // Ensure all threads loaded their part
    
    // Skip if out of bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate indices
    int pixelIdx = y * width + x;
    int outputIdx = pixelIdx * 4; // RGBA
    int cellCount = width * height;
    
    // Get mapping data
    float mapX = mapData[pixelIdx];
    float mapY = mapData[pixelIdx + cellCount];
    float mapUsed = mapData[pixelIdx + 2 * cellCount];
    
    // Check if mapping exists
    if (mapUsed < 0.5f) {
        // No mapping, use red color
        outputPhoto[outputIdx] = 255;     // R
        outputPhoto[outputIdx + 1] = 0;   // G
        outputPhoto[outputIdx + 2] = 0;   // B
        outputPhoto[outputIdx + 3] = 255; // A
        return;
    }
    
    // Round to nearest pixel
    int sourceX = __float2int_rn(mapX);
    int sourceY = __float2int_rn(mapY);
    
    // Check bounds
    if (sourceX < 0 || sourceX >= width || sourceY < 0 || sourceY >= height) {
        // Out of bounds, use transparent
        outputPhoto[outputIdx] = 0;     // R
        outputPhoto[outputIdx + 1] = 0; // G
        outputPhoto[outputIdx + 2] = 0; // B
        outputPhoto[outputIdx + 3] = 0; // A
        return;
    }
    
    // Check if source pixel is in the current tile
    if (sourceX >= tile_x && sourceX < tile_x + TILE_WIDTH &&
        sourceY >= tile_y && sourceY < tile_y + TILE_HEIGHT) {
        // Get source pixel from shared memory
        int sharedIdx = ((sourceY - tile_y) * TILE_WIDTH + (sourceX - tile_x)) * 4;
        
        // Copy pixel data
        outputPhoto[outputIdx] = sharedPhoto[sharedIdx];         // R
        outputPhoto[outputIdx + 1] = sharedPhoto[sharedIdx + 1]; // G
        outputPhoto[outputIdx + 2] = sharedPhoto[sharedIdx + 2]; // B
        outputPhoto[outputIdx + 3] = sharedPhoto[sharedIdx + 3]; // A
    } else {
        // Source pixel not in tile, get from global memory
        int sourceIdx = (sourceY * width + sourceX) * 4;
        
        // Copy pixel data
        outputPhoto[outputIdx] = inputPhoto[sourceIdx];         // R
        outputPhoto[outputIdx + 1] = inputPhoto[sourceIdx + 1]; // G
        outputPhoto[outputIdx + 2] = inputPhoto[sourceIdx + 2]; // B
        outputPhoto[outputIdx + 3] = inputPhoto[sourceIdx + 3]; // A
    }
}
