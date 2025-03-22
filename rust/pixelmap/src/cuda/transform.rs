use crate::affine_transform::AffineTransform;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaError};
use crate::cuda::kernels::KernelCollection;

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
        points: &[(f32, f32)],
    ) -> Result<Vec<(f32, f32)>, String> {
        let kernels = get_kernel_collection()?;
        
        // Upload transforms and points to GPU
        let d_transforms = self.upload_transforms(transforms)?;
        let d_points = self.upload_points(points)?;
        
        // Allocate memory for results
        let result_count = transforms.len() * points.len();
        let mut d_results = CudaBuffer::new::<(f32, f32)>(result_count)
            .map_err(|e| format!("Failed to allocate memory for results: {:?}", e))?;
            
        // Launch kernel
        unsafe {
            kernels.launch_batch_transform(
                d_transforms.as_device_ptr(),
                d_points.as_device_ptr(),
                transforms.len() as i32,
                points.len() as i32,
                d_results.as_device_ptr_mut(),
                self.stream,
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
        let mut d_base_transform = CudaBuffer::new::<AffineTransform>(1)
            .map_err(|e| format!("Failed to allocate memory for base transform: {:?}", e))?;
            
        d_base_transform.copy_from_host(&[*base_transform])
            .map_err(|e| format!("Failed to upload base transform: {:?}", e))?;
            
        // Allocate memory for candidate transforms
        let mut d_candidates = CudaBuffer::new::<AffineTransform>(count)
            .map_err(|e| format!("Failed to allocate memory for candidates: {:?}", e))?;
            
        // Allocate memory for random seeds
        let mut d_seeds = CudaBuffer::new::<u32>(count)
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
                self.stream,
            ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
        }
        
        // Download results
        let mut candidates = vec![AffineTransform::identity(); count];
        d_candidates.copy_to_host(&mut candidates)
            .map_err(|e| format!("Failed to download candidates: {:?}", e))?;
            
        Ok(candidates)
    }
    
    // Helper methods
    // ...existing code...
}