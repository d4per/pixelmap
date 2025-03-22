use crate::model_3d::{Model3D, Vertex3D, Triangle};
use crate::dense_photo_map::DensePhotoMap;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaError};
use crate::cuda::kernels::KernelCollection;

/// CUDA-accelerated 3D model generation
pub struct CudaModel3DGenerator {
    context: CudaContext,
    stream: CudaStream,
}

impl CudaModel3DGenerator {
    /// Create a new CUDA 3D model generator
    pub fn new() -> Result<Self, String> {
        let context = get_cuda_context()?;
        let stream = context.create_stream()
            .map_err(|e| format!("Failed to create CUDA stream: {:?}", e))?;
            
        Ok(Self {
            context,
            stream,
        })
    }
    
    /// Generate a 3D model from a dense photo map
    pub fn generate_model(&self, map: &DensePhotoMap, z_scale: f32) -> Result<Model3D, String> {
        let kernels = get_kernel_collection()?;
        
        // Extract map data
        let width = map.width;
        let height = map.height;
        let map_x = map.get_x_values();
        let map_y = map.get_y_values();
        let map_used = map.get_usage_flags();
        
        // Upload map data to GPU
        let d_map_x = self.upload_float_array(&map_x)?;
        let d_map_y = self.upload_float_array(&map_y)?;
        let d_map_used = self.upload_bool_array(&map_used)?;
        
        // Calculate vertex count (one vertex per grid cell)
        let vertex_count = width * height;
        
        // Allocate memory for vertices
        let mut d_vertices = CudaBuffer::new::<Vertex3D>(vertex_count)
            .map_err(|e| format!("Failed to allocate memory for vertices: {:?}", e))?;
            
        // Launch kernel to generate vertices
        unsafe {
            kernels.launch_generate_vertices(
                d_map_x.as_device_ptr(),
                d_map_y.as_device_ptr(),
                d_map_used.as_device_ptr(),
                d_vertices.as_device_ptr_mut(),
                width as i32,
                height as i32,
                z_scale,
                self.stream,
            ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
        }
        
        // Download vertices
        let mut vertices = vec![Vertex3D::default(); vertex_count];
        d_vertices.copy_to_host(&mut vertices)
            .map_err(|e| format!("Failed to download vertices: {:?}", e))?;
        
        // Generate triangles on CPU (not as computationally intensive as vertex generation)
        let triangles = self.generate_triangles(width, height, &map_used);
        
        // Create model
        Ok(Model3D {
            vertices,
            triangles,
        })
    }
    
    // Helper methods 
    // ...existing code...
    
    fn generate_triangles(&self, width: usize, height: usize, used: &[bool]) -> Vec<Triangle> {
        // Generate triangles for the mesh
        let mut triangles = Vec::new();
        
        for y in 0..(height - 1) {
            for x in 0..(width - 1) {
                let idx00 = y * width + x;
                let idx10 = y * width + (x + 1);
                let idx01 = (y + 1) * width + x;
                let idx11 = (y + 1) * width + (x + 1);
                
                // Only create triangles where all vertices are used
                if used[idx00] && used[idx10] && used[idx01] && used[idx11] {
                    // Two triangles for each quad
                    triangles.push(Triangle {
                        v1: idx00 as u32,
                        v2: idx10 as u32, 
                        v3: idx01 as u32,
                    });
                    
                    triangles.push(Triangle {
                        v1: idx10 as u32,
                        v2: idx11 as u32,
                        v3: idx01 as u32,
                    });
                }
            }
        }
        
        triangles
    }
}