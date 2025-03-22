//! CUDA-accelerated affine transform operations

use crate::affine_transform::AffineTransform;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaError, CudaStream};
use super::common::{get_cuda_context, get_kernel_collection};

/// CUDA-accelerated transform operations
pub struct CudaTransformProcessor {
    context: CudaContext,
    stream: CudaStream,
}

impl CudaTransformProcessor {
    /// Create a new CUDA transform processor
    pub fn new() -> Result<Self, String> {
        let context = get_cuda_context()?;
        let stream = context.create_stream()
            .map_err(|e| format!("Failed to create CUDA stream: {:?}", e))?;
            
        Ok(Self {
            context,
            stream,
        })
    }
    
    /// Apply a batch of transforms to a batch of points
    pub fn batch_transform(
        &self,
        transforms: &[AffineTransform],
        points: &[(f32, f32)]
    ) -> Result<Vec<(f32, f32)>, String> {
        let kernels = get_kernel_collection()?;
        
        // Upload transforms to GPU
        let d_transforms = self.upload_transforms(transforms)?;
        
        // Upload points to GPU
        let d_points = self.upload_points(points)?;
        
        // Allocate memory for results (one result per transform per point)
        let result_count = transforms.len() * points.len();
        let mut d_results = CudaBuffer::new(result_count)
            .map_err(|e| format!("Failed to allocate memory for results: {:?}", e))?;
        
        // Configure kernel launch
        let block_size = 256; // or another power of 2 that fits your GPU
        let grid_size = (result_count + block_size - 1) / block_size;
        
        // Launch kernel - fix to use the proper clone() method to avoid move
        unsafe {
            kernels.launch_batch_transform(
                d_transforms.as_device_ptr(),
                d_points.as_device_ptr(),
                transforms.len() as i32,
                points.len() as i32,
                d_results.as_device_ptr_mut(),
                self.stream.clone(), // Fix: use clone() instead of moving
            ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
        }
        
        // Download results
        let mut results = vec![(0.0f32, 0.0f32); result_count];
        d_results.copy_to_host(&mut results)
            .map_err(|e| format!("Failed to download results: {:?}", e))?;
            
        Ok(results)
    }
    
    /// Generate multiple candidate transforms with variations
    pub fn generate_transform_candidates(
        &self,
        base_transform: &AffineTransform,
        count: usize,
        variation_scale: f32
    ) -> Result<Vec<AffineTransform>, String> {
        let kernels = get_kernel_collection()?;
        
        // Upload base transform to GPU
        let mut d_base_transform = CudaBuffer::new(1)
            .map_err(|e| format!("Failed to allocate memory for base transform: {:?}", e))?;
            
        d_base_transform.copy_from_host(&[*base_transform])
            .map_err(|e| format!("Failed to upload base transform: {:?}", e))?;
            
        // Allocate memory for candidate transforms
        let mut d_candidates = CudaBuffer::new(count)
            .map_err(|e| format!("Failed to allocate memory for candidates: {:?}", e))?;
            
        // Allocate memory for random seeds
        let mut d_seeds = CudaBuffer::new(count)
            .map_err(|e| format!("Failed to allocate memory for seeds: {:?}", e))?;
            
        // Generate random seeds on CPU and upload
        let seeds: Vec<u32> = (0..count).map(|i| i as u32).collect();
        d_seeds.copy_from_host(&seeds)
            .map_err(|e| format!("Failed to upload seeds: {:?}", e))?;
            
        // Launch kernel
        unsafe {
            kernels.launch_generate_transform_candidates(
                d_base_transform.as_device_ptr(),
                d_candidates.as_device_ptr_mut(),
                d_seeds.as_device_ptr(),
                count as i32,
                variation_scale,
                self.stream.clone(),
            ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
        }
        
        // Download results
        let mut candidates = vec![AffineTransform::identity(); count];
        d_candidates.copy_to_host(&mut candidates)
            .map_err(|e| format!("Failed to download candidates: {:?}", e))?;
            
        Ok(candidates)
    }
    
    // Helper methods
    fn upload_transforms(&self, transforms: &[AffineTransform]) -> Result<CudaBuffer<AffineTransform>, String> {
        let mut buffer = CudaBuffer::new(transforms.len())
            .map_err(|e| format!("Failed to allocate memory for transforms: {:?}", e))?;
            
        buffer.copy_from_host(transforms)
            .map_err(|e| format!("Failed to upload transforms: {:?}", e))?;
            
        Ok(buffer)
    }
    
    fn upload_points(&self, points: &[(f32, f32)]) -> Result<CudaBuffer<(f32, f32)>, String> {
        let mut buffer = CudaBuffer::new(points.len())
            .map_err(|e| format!("Failed to allocate memory for points: {:?}", e))?;
            
        buffer.copy_from_host(points)
            .map_err(|e| format!("Failed to upload points: {:?}", e))?;
            
        Ok(buffer)
    }
}

impl Drop for CudaTransformProcessor {
    fn drop(&mut self) {
        // Clean up CUDA resources
    }
}

impl AffineTransform {
    pub fn identity() -> Self {
        AffineTransform {
            a11: 1.0,
            a12: 0.0,
            a21: 0.0,
            a22: 1.0,
            translate_x: 0.0,
            translate_y: 0.0,
            origin_x: 0,
            origin_y: 0,
        }
    }
}
