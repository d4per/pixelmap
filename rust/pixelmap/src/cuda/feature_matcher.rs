//! CUDA-accelerated implementation of CircularFeatureDescriptorMatcher
//!
//! This module provides a GPU-accelerated matcher for CircularFeatureDescriptors
//! using a highly parallel brute-force approach optimized with tiling and
//! efficient memory access patterns.

use crate::circular_feature_descriptor::CircularFeatureDescriptor;
use crate::circular_feature_grid::CircularFeatureGrid;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaError};
use crate::cuda::kernels::KernelCollection;
use rustacuda::memory::DeviceBox;
use rustacuda::launch;
use rustacuda::stream::Stream;
use std::sync::Arc;
use crate::circular_feature_grid::FeaturePair;

/// A CUDA-accelerated matcher for circular feature descriptors
pub struct CudaCircularFeatureDescriptorMatcher {
    /// CUDA context for GPU operations
    context: Arc<CudaContext>,
    /// CUDA kernels
    kernels: Arc<KernelCollection>,
    /// CUDA stream for asynchronous operations
    stream: Stream,
}

impl CudaCircularFeatureDescriptorMatcher {
    /// Creates a new CUDA-accelerated matcher
    pub fn new(
        context: Arc<CudaContext>,
        kernels: Arc<KernelCollection>
    ) -> Result<Self, CudaError> {
        let stream = Stream::new()?;
        
        Ok(Self {
            context,
            kernels,
            stream,
        })
    }
    
    /// Matches feature descriptors between two images using brute force on GPU
    ///
    /// This version uses tiling for optimal memory coherence and performance
    pub fn match_areas(
        &self,
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
    ) -> Result<Vec<(CircularFeatureDescriptor, CircularFeatureDescriptor)>, CudaError> {
        let descriptors1 = img1.get_infos();
        let descriptors2 = img2.get_infos();
        
        let num_desc1 = descriptors1.len();
        let num_desc2 = descriptors2.len();
        
        // 1. Transfer descriptor data to GPU
        let mut device_desc1 = self.upload_descriptors(descriptors1)?;
        let mut device_desc2 = self.upload_descriptors(descriptors2)?;
        
        // 2. Allocate space for distance matrix and nearest match indices
        let mut device_distances = CudaBuffer::<f32>::new(
            &self.context, 
            num_desc2 // We only need to store distances for current tile
        )?;
        
        let mut device_match_indices = CudaBuffer::<i32>::new(
            &self.context,
            num_desc2
        )?;
        
        // 3. Setup kernel parameters
        const TILE_SIZE: usize = 256; // Adjusted based on GPU capabilities
        const BLOCK_SIZE: usize = 256; // Threads per block
        
        // 4. Launch kernel with tiling for efficient memory access
        let mut matches = Vec::with_capacity(num_desc2);
        
        // Process descriptors in tiles for better memory coherence
        for i in 0..num_desc2 {
            // Launch kernel to find closest descriptor from img1 for this descriptor from img2
            unsafe {
                if let Some(kernel) = &self.kernels.find_nearest_descriptor {
                    let params = (
                        device_desc1.as_device_ptr(),
                        device_desc2.as_device_ptr(),
                        device_distances.as_device_ptr(),
                        device_match_indices.as_device_ptr(),
                        num_desc1 as i32,
                        i as i32, // Current descriptor index from img2
                        6 as i32, // Feature vector dimension (6 in CircularFeatureDescriptor)
                    );
                    
                    launch!(
                        kernel<<<(num_desc1 as u32 + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32, BLOCK_SIZE as u32, 0, self.stream>>>(
                            params
                        )
                    )?;
                } else {
                    return Err(CudaError::KernelNotFound("find_nearest_descriptor".to_string()));
                }
            }
        }
        
        // 5. Download match results
        let mut host_match_indices = vec![0i32; num_desc2];
        device_match_indices.copy_to_host(&mut host_match_indices)?;
        
        // 6. Construct result pairs
        for (idx2, idx1) in host_match_indices.iter().enumerate() {
            if *idx1 >= 0 && (*idx1 as usize) < num_desc1 {
                matches.push((descriptors1[*idx1 as usize], descriptors2[idx2]));
            }
        }
        
        Ok(matches)
    }
    
    /// Alternative implementation with aggressive tiling for even better memory coherence
    pub fn match_areas_tiled(
        &self,
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
    ) -> Result<Vec<(CircularFeatureDescriptor, CircularFeatureDescriptor)>, CudaError> {
        let descriptors1 = img1.get_infos();
        let descriptors2 = img2.get_infos();
        
        let num_desc1 = descriptors1.len();
        let num_desc2 = descriptors2.len();
        
        // Upload descriptors in compact format optimized for CUDA
        let device_desc1 = self.upload_compact_descriptors(descriptors1)?;
        let device_desc2 = self.upload_compact_descriptors(descriptors2)?;
        
        // Allocate distance matrix and match indices
        let mut device_match_indices = CudaBuffer::<i32>::new(&self.context, num_desc2)?;
        
        // Define tile sizes for better memory coherence
        const TILE_WIDTH: usize = 32;
        const TILE_HEIGHT: usize = 32;
        
        // Calculate grid size based on tile dimensions
        let grid_dim_x = (num_desc1 + TILE_WIDTH - 1) / TILE_WIDTH;
        let grid_dim_y = (num_desc2 + TILE_HEIGHT - 1) / TILE_HEIGHT;
        
        unsafe {
            if let Some(kernel) = &self.kernels.match_descriptors_tiled {
                let params = (
                    device_desc1.as_device_ptr(),
                    device_desc2.as_device_ptr(),
                    device_match_indices.as_device_ptr(),
                    num_desc1 as i32,
                    num_desc2 as i32,
                    6 as i32, // Feature vector dimension
                );
                
                launch!(
                    kernel<<<dim3(grid_dim_x as u32, grid_dim_y as u32, 1), dim3(TILE_WIDTH as u32, TILE_HEIGHT as u32, 1), 0, self.stream>>>(
                        params
                    )
                )?;
            } else {
                return Err(CudaError::KernelNotFound("match_descriptors_tiled".to_string()));
            }
        }
        
        // Download match results
        let mut host_match_indices = vec![0i32; num_desc2];
        device_match_indices.copy_to_host(&mut host_match_indices)?;
        
        // Construct result pairs
        let mut matches = Vec::with_capacity(num_desc2);
        for (idx2, idx1) in host_match_indices.iter().enumerate() {
            if *idx1 >= 0 && (*idx1 as usize) < num_desc1 {
                matches.push((descriptors1[*idx1 as usize], descriptors2[idx2]));
            }
        }
        
        Ok(matches)
    }
    
    /// Match features using brute force matching on GPU
    /// This is much faster than KD-tree for large feature sets when using GPU
    pub fn match_features_brute_force<T1, T2>(&self, 
        grid1: &T1, 
        grid2: &T2,
        max_features: usize
    ) -> Result<Vec<FeaturePair>, String> 
    where
        T1: FeatureGrid,
        T2: FeatureGrid,
    {
        let kernels = get_kernel_collection()?;
        
        // Get feature descriptors
        let descriptors1 = grid1.get_feature_descriptors();
        let descriptors2 = grid2.get_feature_descriptors();
        
        // Select top features by intensity
        let top_descriptors1 = self.select_top_features(descriptors1, max_features)?;
        let top_descriptors2 = self.select_top_features(descriptors2, max_features)?;
        
        // Upload descriptors to GPU
        let d_descriptors1 = self.upload_descriptors(&top_descriptors1)?;
        let d_descriptors2 = self.upload_descriptors(&top_descriptors2)?;
        
        // Allocate device memory for match indices and distances
        let mut d_match_indices = CudaBuffer::new::<i32>(top_descriptors1.len())
            .map_err(|e| format!("Failed to allocate memory for match indices: {:?}", e))?;
        
        let mut d_match_distances = CudaBuffer::new::<f32>(top_descriptors1.len())
            .map_err(|e| format!("Failed to allocate memory for match distances: {:?}", e))?;
        
        // Launch kernel to find nearest neighbors for each descriptor
        unsafe {
            kernels.launch_find_nearest_neighbors(
                d_descriptors1.as_device_ptr(),
                d_descriptors2.as_device_ptr(),
                top_descriptors1.len() as i32,
                top_descriptors2.len() as i32,
                d_match_indices.as_device_ptr_mut(),
                d_match_distances.as_device_ptr_mut(),
                self.stream,
            ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
        }
        
        // Download match results
        let mut match_indices = vec![0i32; top_descriptors1.len()];
        let mut match_distances = vec![0.0f32; top_descriptors1.len()];
        
        d_match_indices.copy_to_host(&mut match_indices)
            .map_err(|e| format!("Failed to download match indices: {:?}", e))?;
        
        d_match_distances.copy_to_host(&mut match_distances)
            .map_err(|e| format!("Failed to download match distances: {:?}", e))?;
        
        // Create feature pairs from matches
        let mut feature_pairs = Vec::new();
        
        for (i, &match_idx) in match_indices.iter().enumerate() {
            if match_idx >= 0 && match_distances[i] < 0.1 { // Threshold for good match
                feature_pairs.push(FeaturePair(
                    top_descriptors1[i].clone(),
                    top_descriptors2[match_idx as usize].clone()
                ));
            }
        }
        
        Ok(feature_pairs)
    }
    
    // Helper methods
    fn upload_descriptors(
        &self, 
        descriptors: &[CircularFeatureDescriptor]
    ) -> Result<CudaBuffer<f32>, CudaError> {
        // Upload feature vectors to GPU in a format optimized for CUDA
        todo!("Implement descriptor upload")
    }
    
    fn upload_compact_descriptors(
        &self,
        descriptors: &[CircularFeatureDescriptor]
    ) -> Result<CudaBuffer<i64>, CudaError> {
        // Extract just the feature vector components in a memory-coherent layout
        let mut compact_data = Vec::with_capacity(descriptors.len() * 6);
        
        for desc in descriptors {
            for i in 0..6 {
                compact_data.push(desc.feature_vector[i]);
            }
        }
        
        let mut buffer = CudaBuffer::<i64>::new(&self.context, compact_data.len())?;
        buffer.copy_from_host(&compact_data)?;
        
        Ok(buffer)
    }
}

impl CudaError {
    fn KernelNotFound(name: String) -> Self {
        CudaError::KernelLoadFailure(format!("Kernel '{}' not found", name))
    }
}
