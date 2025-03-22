/**
 * CUDA kernel implementations for affine transform operations
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Structure representing an affine transform
// [m00, m01, m02]
// [m10, m11, m12]
// [ 0,   0,   1 ]
struct AffineTransform {
    float m00, m01, m02;
    float m10, m11, m12;
};

// Kernel to apply a batch of transforms to a batch of points
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

// Generate random float in range [-1, 1]
__device__ float random_float(unsigned int seed) {
    // Simple xorshift random number generator
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    
    // Convert to float in range [-1, 1]
    return (float)(seed & 0x7FFFFFFF) / (float)(0x7FFFFFFF) * 2.0f - 1.0f;
}

// Kernel to generate transform candidates with random variations
extern "C" __global__ void generate_transform_candidates(
    const AffineTransform* baseTransform,
    AffineTransform* candidates,
    const unsigned int* seeds,
    int count,
    float variationScale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= count) {
        return;
    }
    
    // Get seed for this thread
    unsigned int seed = seeds[idx];
    
    // Load base transform
    AffineTransform base = baseTransform[0];
    
    // Generate variations
    // Scale - small random changes to m00 and m11
    float scaleVar = 1.0f + random_float(seed) * 0.05f * variationScale;
    
    // Rotation - small random change to rotation component
    float rotationVar = random_float(seed ^ 0x12345678) * 0.05f * variationScale;
    
    // Translation - small random changes to m02 and m12
    float transXVar = random_float(seed ^ 0x87654321) * 10.0f * variationScale;
    float transYVar = random_float(seed ^ 0x55555555) * 10.0f * variationScale;
    
    // Apply variations to create candidate transform
    float cosTheta = __cosf(rotationVar);
    float sinTheta = __sinf(rotationVar);
    
    // Create rotation matrix
    float r00 = cosTheta;
    float r01 = -sinTheta;
    float r10 = sinTheta;
    float r11 = cosTheta;
    
    // Compose with base transform
    candidates[idx].m00 = (base.m00 * r00 + base.m01 * r10) * scaleVar;
    candidates[idx].m01 = (base.m00 * r01 + base.m01 * r11) * scaleVar;
    candidates[idx].m02 = base.m02 + transXVar;
    candidates[idx].m10 = (base.m10 * r00 + base.m11 * r10) * scaleVar;
    candidates[idx].m11 = (base.m10 * r01 + base.m11 * r11) * scaleVar;
    candidates[idx].m12 = base.m12 + transYVar;
}

// Kernel to compose two transforms
extern "C" __global__ void compose_transforms(
    const AffineTransform* transforms1,
    const AffineTransform* transforms2,
    AffineTransform* result_transforms,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= count) {
        return;
    }
    
    // Load transforms
    AffineTransform t1 = transforms1[idx];
    AffineTransform t2 = transforms2[idx];
    
    // Compose transforms: result = t1 * t2
    result_transforms[idx].m00 = t1.m00 * t2.m00 + t1.m01 * t2.m10;
    result_transforms[idx].m01 = t1.m00 * t2.m01 + t1.m01 * t2.m11;
    result_transforms[idx].m02 = t1.m00 * t2.m02 + t1.m01 * t2.m12 + t1.m02;
    
    result_transforms[idx].m10 = t1.m10 * t2.m00 + t1.m11 * t2.m10;
    result_transforms[idx].m11 = t1.m10 * t2.m01 + t1.m11 * t2.m11;
    result_transforms[idx].m12 = t1.m10 * t2.m02 + t1.m11 * t2.m12 + t1.m12;
}

// Kernel to invert transforms
extern "C" __global__ void invert_transforms(
    const AffineTransform* transforms,
    AffineTransform* inverse_transforms,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= count) {
        return;
    }
    
    // Load transform
    AffineTransform t = transforms[idx];
    
    // Calculate determinant
    float det = t.m00 * t.m11 - t.m01 * t.m10;
    
    // Check if determinant is close to zero
    if (fabsf(det) < 1e-6f) {
        // Singular matrix - return identity
        inverse_transforms[idx].m00 = 1.0f;
        inverse_transforms[idx].m01 = 0.0f;
        inverse_transforms[idx].m02 = 0.0f;
        inverse_transforms[idx].m10 = 0.0f;
        inverse_transforms[idx].m11 = 1.0f;
        inverse_transforms[idx].m12 = 0.0f;
        return;
    }
    
    // Calculate inverse
    float inv_det = 1.0f / det;
    
    inverse_transforms[idx].m00 = t.m11 * inv_det;
    inverse_transforms[idx].m01 = -t.m01 * inv_det;
    inverse_transforms[idx].m02 = (t.m01 * t.m12 - t.m02 * t.m11) * inv_det;
    
    inverse_transforms[idx].m10 = -t.m10 * inv_det;
    inverse_transforms[idx].m11 = t.m00 * inv_det;
    inverse_transforms[idx].m12 = (t.m02 * t.m10 - t.m00 * t.m12) * inv_det;
}

// Helper function to calculate L1 color difference
__device__ float colorDifference(
    const unsigned char* photo1,
    const unsigned char* photo2,
    int x1, int y1,
    int x2, int y2,
    int width1, int height1,
    int width2, int height2
) {
    // Check bounds
    if (x1 < 0 || x1 >= width1 || y1 < 0 || y1 >= height1 ||
        x2 < 0 || x2 >= width2 || y2 < 0 || y2 >= height2) {
        return 255.0f * 3.0f; // Max difference
    }
    
    // Get pixel indices
    int idx1 = (y1 * width1 + x1) * 4;
    int idx2 = (y2 * width2 + x2) * 4;
    
    // Calculate L1 difference
    float diff = 0.0f;
    diff += fabsf((float)photo1[idx1] - (float)photo2[idx2]);         // R
    diff += fabsf((float)photo1[idx1 + 1] - (float)photo2[idx2 + 1]); // G
    diff += fabsf((float)photo1[idx1 + 2] - (float)photo2[idx2 + 2]); // B
    
    return diff;
}

// Kernel to find the best transform by comparing image regions
extern "C" __global__ void find_best_transform(
    const unsigned char* sourcePhoto,
    const unsigned char* targetPhoto,
    const AffineTransform* transforms,
    int transformCount,
    float* scores,
    int* bestTransformIdx,
    int width, int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= transformCount) {
        return;
    }
    
    // Load transform
    AffineTransform t = transforms[idx];
    
    // Sample a grid of points and compute color difference
    const int GRID_SIZE = 32; // Sample an evenly spaced grid
    float totalDiff = 0.0f;
    int validPoints = 0;
    
    for (int y = 0; y < height; y += height / GRID_SIZE) {
        for (int x = 0; x < width; x += width / GRID_SIZE) {
            // Transform point from source to target
            float tx = t.m00 * x + t.m01 * y + t.m02;
            float ty = t.m10 * x + t.m11 * y + t.m12;
            
            // Round to nearest pixel
            int tx_int = __float2int_rn(tx);
            int ty_int = __float2int_rn(ty);
            
            // Add color difference
            float diff = colorDifference(
                sourcePhoto, targetPhoto,
                x, y, tx_int, ty_int,
                width, height, width, height
            );
            
            // Only count points that are in bounds
            if (tx_int >= 0 && tx_int < width && ty_int >= 0 && ty_int < height) {
                totalDiff += diff;
                validPoints++;
            }
        }
    }
    
    // Calculate score (lower is better)
    float score = (validPoints > 0) ? (totalDiff / validPoints) : FLT_MAX;
    
    // Store score
    scores[idx] = score;
    
    // Find best transform (using atomicMin to find minimum score)
    if (idx == 0) {
        // Initialize with first transform
        *bestTransformIdx = 0;
    }
    
    // Wait for initialization
    __syncthreads();
    
    // Use atomic operations to find minimum score
    if (validPoints > 0) {
        atomicMin((int*)bestTransformIdx, idx);
    }
}
