/**
 * CUDA kernel implementations for 3D model operations
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Vertex structure with position and texture coordinates
struct Vertex {
    float x, y, z;        // Position
    float nx, ny, nz;     // Normal
    float u, v;           // Texture coordinates
};

// Triangle structure with vertex indices
struct Triangle {
    int v1, v2, v3;
};

// Kernel to generate vertices from dense photo map data
extern "C" __global__ void generate_vertices(
    const float* map_x,          // X mapping data
    const float* map_y,          // Y mapping data
    const float* map_used,       // Usage flags
    Vertex* vertices,           // Output vertices
    int width, int height,
    float z_scale               // Z scaling factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate grid index
    int idx = y * width + x;
    
    // Get mapping data
    float mx = map_x[idx];
    float my = map_y[idx];
    float used = map_used[idx];
    
    // Check if cell is used
    if (used > 0.5f) {
        // Calculate z value based on mapping distance
        float dx = mx - (float)x;
        float dy = my - (float)y;
        float z = sqrtf(dx * dx + dy * dy) * z_scale;
        
        // Initialize vertex
        vertices[idx].x = (float)x;
        vertices[idx].y = (float)y;
        vertices[idx].z = z;
        
        // Initialize other fields
        vertices[idx].nx = 0.0f;
        vertices[idx].ny = 0.0f;
        vertices[idx].nz = 1.0f;  // Default normal pointing up
        
        // Texture coordinates (normalized to [0, 1])
        vertices[idx].u = (float)x / (float)(width - 1);
        vertices[idx].v = (float)y / (float)(height - 1);
    } else {
        // Initialize unused vertex with sentinel value
        vertices[idx].x = 0.0f;
        vertices[idx].y = 0.0f;
        vertices[idx].z = -999999.0f;  // Sentinel value
        
        // Initialize other fields
        vertices[idx].nx = 0.0f;
        vertices[idx].ny = 0.0f;
        vertices[idx].nz = 0.0f;
        
        // Default texture coordinates
        vertices[idx].u = 0.0f;
        vertices[idx].v = 0.0f;
    }
}

// Helper function to calculate the normal of a triangle
__device__ void calculateNormal(
    const Vertex* vertices,
    int v1, int v2, int v3,
    float* nx, float* ny, float* nz
) {
    // Get vertex positions
    float x1 = vertices[v1].x;
    float y1 = vertices[v1].y;
    float z1 = vertices[v1].z;
    
    float x2 = vertices[v2].x;
    float y2 = vertices[v2].y;
    float z2 = vertices[v2].z;
    
    float x3 = vertices[v3].x;
    float y3 = vertices[v3].y;
    float z3 = vertices[v3].z;
    
    // Calculate vectors for the two edges
    float ux = x2 - x1;
    float uy = y2 - y1;
    float uz = z2 - z1;
    
    float vx = x3 - x1;
    float vy = y3 - y1;
    float vz = z3 - z1;
    
    // Cross product
    *nx = uy * vz - uz * vy;
    *ny = uz * vx - ux * vz;
    *nz = ux * vy - uy * vx;
    
    // Normalize
    float length = sqrtf((*nx) * (*nx) + (*ny) * (*ny) + (*nz) * (*nz));
    
    if (length > 1e-6f) {
        *nx /= length;
        *ny /= length;
        *nz /= length;
    } else {
        *nx = 0.0f;
        *ny = 0.0f;
        *nz = 1.0f;  // Default to pointing up
    }
}

// Kernel to generate triangle normals
extern "C" __global__ void generate_normals(
    const Vertex* vertices,
    const Triangle* triangles,
    float* normals,  // Output normals (nx, ny, nz for each triangle)
    int vertex_count,
    int triangle_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= triangle_count) {
        return;
    }
    
    // Get triangle
    Triangle tri = triangles[idx];
    
    // Check if any vertex is unused (has sentinel value)
    if (vertices[tri.v1].z < -999998.0f ||
        vertices[tri.v2].z < -999998.0f ||
        vertices[tri.v3].z < -999998.0f) {
        // Invalid triangle - set normal to (0,0,0)
        normals[idx * 3] = 0.0f;
        normals[idx * 3 + 1] = 0.0f;
        normals[idx * 3 + 2] = 0.0f;
        return;
    }
    
    // Calculate normal
    float nx, ny, nz;
    calculateNormal(vertices, tri.v1, tri.v2, tri.v3, &nx, &ny, &nz);
    
    // Store normal
    normals[idx * 3] = nx;
    normals[idx * 3 + 1] = ny;
    normals[idx * 3 + 2] = nz;
}

// Kernel to apply texture coordinates from map data
extern "C" __global__ void apply_texture_coordinates(
    Vertex* vertices,
    const float* map_data,  // Packed data [x | y | used]
    int vertex_count,
    int width, int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= vertex_count) {
        return;
    }
    
    // Get vertex
    Vertex vertex = vertices[idx];
    
    // Check if unused
    if (vertex.z < -999998.0f) {
        return;
    }
    
    // Get map data for this vertex
    int cell_count = width * height;
    int x = (int)vertex.x;
    int y = (int)vertex.y;
    
    if (x >= 0 && x < width && y >= 0 && y < height) {
        int map_idx = y * width + x;
        float mapped_x = map_data[map_idx];
        float mapped_y = map_data[map_idx + cell_count];
        float used = map_data[map_idx + 2 * cell_count];
        
        if (used > 0.5f) {
            // Normalize texture coordinates to [0,1]
            float u = mapped_x / (float)(width - 1);
            float v = mapped_y / (float)(height - 1);
            
            // Update vertex texture coordinates
            vertices[idx].u = u;
            vertices[idx].v = v;
        }
    }
}
