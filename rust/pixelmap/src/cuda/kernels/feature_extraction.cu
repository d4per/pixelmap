#include <device_launch_parameters.h>
#include <math_constants.h>

// Raw descriptor structure matching the host-side layout
struct CircularFeatureDescriptorRaw {
    float total_angle;
    float r_weight;
    float g_weight;
    float b_weight;
    float intensity;
};

// Calculate if point (x,y) is inside the circle with given center and radius
__device__ bool is_inside_circle(int x, int y, int center_x, int center_y, int radius) {
    int dx = x - center_x;
    int dy = y - center_y;
    return (dx * dx + dy * dy) <= (radius * radius);
}

// Calculate the angle in radians between two points
__device__ float calculate_angle(float x1, float y1, float x2, float y2) {
    return atan2f(y2 - y1, x2 - x1);
}

// Extract circular feature descriptors
extern "C" __global__ void extract_circular_features(
    const unsigned char* photo,
    CircularFeatureDescriptorRaw* descriptors,
    const unsigned int width,
    const unsigned int height,
    const unsigned int circle_radius,
    const unsigned int max_color_value
) {
    // Calculate thread position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Initialize accumulation variables
    float total_angle = 0.0f;
    float r_sum = 0.0f;
    float g_sum = 0.0f;
    float b_sum = 0.0f;
    float intensity = 0.0f;
    int pixel_count = 0;
    
    // Process all pixels in the circular neighborhood
    for (int dy = -circle_radius; dy <= (int)circle_radius; dy++) {
        for (int dx = -circle_radius; dx <= (int)circle_radius; dx++) {
            int px = x + dx;
            int py = y + dy;
            
            // Skip if out of bounds or not in circle
            if (px < 0 || px >= width || py < 0 || py >= height ||
                !is_inside_circle(px, py, x, y, circle_radius)) {
                continue;
            }
            
            // Get pixel colors
            int idx = (py * width + px) * 4;
            float r = photo[idx];     // R
            float g = photo[idx + 1]; // G
            float b = photo[idx + 2]; // B
            
            // Skip transparent pixels
            if (photo[idx + 3] < 128) { // Alpha
                continue;
            }
            
            // Accumulate color values
            r_sum += r;
            g_sum += g;
            b_sum += b;
            
            // Calculate angle contribution from this pixel
            if (dx != 0 || dy != 0) {
                float pixel_angle = calculate_angle(0, 0, dx, dy);
                
                // Use standard perceptual luminance formula (BT.709)
                // This matches the Rust implementation for computing intensity
                float pixel_intensity = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                
                total_angle += pixel_angle * pixel_intensity;
                intensity += pixel_intensity;
            }
            
            pixel_count++;
        }
    }
    
    // Store the results
    int desc_idx = y * width + x;
    
    if (pixel_count > 0) {
        float total_color = r_sum + g_sum + b_sum;
        
        descriptors[desc_idx].total_angle = total_angle;
        descriptors[desc_idx].r_weight = (total_color > 0) ? r_sum / total_color : 0.0f;
        descriptors[desc_idx].g_weight = (total_color > 0) ? g_sum / total_color : 0.0f;
        descriptors[desc_idx].b_weight = (total_color > 0) ? b_sum / total_color : 0.0f;
        descriptors[desc_idx].intensity = intensity / (float)max_color_value;
    } else {
        // No valid pixels in circle
        descriptors[desc_idx].total_angle = 0.0f;
        descriptors[desc_idx].r_weight = 0.0f;
        descriptors[desc_idx].g_weight = 0.0f;
        descriptors[desc_idx].b_weight = 0.0f;
        descriptors[desc_idx].intensity = 0.0f;
    }
}

// Extract circular feature descriptors from an image
extern "C" __global__ void extract_features(
    const unsigned char* photo,  // Input photo data (RGBA)
    float* descriptors,          // Output descriptors (packed format)
    int width, int height,
    int circle_radius,
    int max_color_value
) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate descriptor index
    int desc_idx = y * width + x;
    
    // Accumulators for total R/G/B values
    float sum_red = 0.0f;
    float sum_green = 0.0f;
    float sum_blue = 0.0f;
    
    // Accumulators for weighted sums
    float sum_weighted_red_x = 0.0f;
    float sum_weighted_red_y = 0.0f;
    
    float sum_weighted_green_x = 0.0f;
    float sum_weighted_green_y = 0.0f;
    
    float sum_weighted_blue_x = 0.0f;
    float sum_weighted_blue_y = 0.0f;
    
    // Count processed pixels
    int pixel_count = 0;
    
    // Loop over circular area
    for (int dy = -circle_radius; dy <= circle_radius; dy++) {
        // Calculate row radius using circle equation
        int row_radius = sqrtf((circle_radius * circle_radius) - (dy * dy));
        
        for (int dx = -row_radius; dx <= row_radius; dx++) {
            // Use toroidal coordinates (wrap around edges)
            int wrapped_x = (x + dx + width) % width;
            int wrapped_y = (y + dy + height) % height;
            
            // Index into image data (RGBA format)
            int pixel_idx = (wrapped_y * width + wrapped_x) * 4;
            
            // Get RGB values
            float red = photo[pixel_idx];
            float green = photo[pixel_idx + 1];
            float blue = photo[pixel_idx + 2];
            
            // Accumulate sums
            sum_red += red;
            sum_green += green;
            sum_blue += blue;
            
            // Accumulate weighted sums for center of mass calculation
            sum_weighted_red_x += dx * red;
            sum_weighted_red_y += dy * red;
            
            sum_weighted_green_x += dx * green;
            sum_weighted_green_y += dy * green;
            
            sum_weighted_blue_x += dx * blue;
            sum_weighted_blue_y += dy * blue;
            
            pixel_count++;
        }
    }
    
    // Calculate center of mass for each channel
    float red_cm_x = (sum_red > 0.0f) ? (sum_weighted_red_x / sum_red) : 0.0f;
    float red_cm_y = (sum_red > 0.0f) ? (sum_weighted_red_y / sum_red) : 0.0f;
    float red_angle = (sum_red > 0.0f) ? atan2f(red_cm_y, red_cm_x) : 0.0f;
    float red_radius = sqrtf(red_cm_x * red_cm_x + red_cm_y * red_cm_y);
    
    float green_cm_x = (sum_green > 0.0f) ? (sum_weighted_green_x / sum_green) : 0.0f;
    float green_cm_y = (sum_green > 0.0f) ? (sum_weighted_green_y / sum_green) : 0.0f;
    float green_angle = (sum_green > 0.0f) ? atan2f(green_cm_y, green_cm_x) : 0.0f;
    float green_radius = sqrtf(green_cm_x * green_cm_x + green_cm_y * green_cm_y);
    
    float blue_cm_x = (sum_blue > 0.0f) ? (sum_weighted_blue_x / sum_blue) : 0.0f;
    float blue_cm_y = (sum_blue > 0.0f) ? (sum_weighted_blue_y / sum_blue) : 0.0f;
    float blue_angle = (sum_blue > 0.0f) ? atan2f(blue_cm_y, blue_cm_x) : 0.0f;
    float blue_radius = sqrtf(blue_cm_x * blue_cm_x + blue_cm_y * blue_cm_y);
    
    // Calculate total center of mass
    float sum_all = sum_red + sum_green + sum_blue;
    float total_cm_x = (sum_all > 0.0f) ? 
        ((sum_weighted_red_x + sum_weighted_green_x + sum_weighted_blue_x) / sum_all) : 0.0f;
    float total_cm_y = (sum_all > 0.0f) ? 
        ((sum_weighted_red_y + sum_weighted_green_y + sum_weighted_blue_y) / sum_all) : 0.0f;
    float total_angle = (sum_all > 0.0f) ? atan2f(total_cm_y, total_cm_x) : 0.0f;
    
    // Calculate RGB weights (normalized by total)
    float r_weight = (sum_all > 0.0f) ? (sum_red / sum_all) : 0.0f;
    float g_weight = (sum_all > 0.0f) ? (sum_green / sum_all) : 0.0f;
    float b_weight = (sum_all > 0.0f) ? (sum_blue / sum_all) : 0.0f;
    
    // Calculate intensity (normalized by max possible value)
    float intensity = (max_color_value > 0) ? 
        ((sum_red + sum_green + sum_blue) / (3.0f * max_color_value)) : 0.0f;
    
    // Store descriptor data in packed format (structure of arrays)
    descriptors[desc_idx] = total_angle;
    descriptors[desc_idx + width * height] = r_weight;
    descriptors[desc_idx + 2 * width * height] = g_weight;
    descriptors[desc_idx + 3 * width * height] = b_weight;
    descriptors[desc_idx + 4 * width * height] = intensity;
}
