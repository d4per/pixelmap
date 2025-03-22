//! CUDA-accelerated implementation of the CorrespondenceMappingAlgorithm
//!
//! This module provides a GPU-accelerated version of the correspondence mapping
//! algorithm that uses tiling for memory coherence and efficient parallelization.

use crate::affine_transform::AffineTransform;
use crate::affine_transform_cell::AffineTransformCell;
use crate::ac_grid::AcGrid;
use crate::photo::Photo;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaError};
use crate::cuda::kernels::KernelCollection;
use rustacuda::launch;
use rustacuda::stream::Stream;
use rustacuda::memory::DeviceBox;
use std::sync::Arc;

/// Configuration for the CUDA correspondence mapping algorithm
pub struct CudaCorrespondenceMappingConfig {
    /// Number of iterations to perform
    pub iterations: usize,
    /// Number of candidates to evaluate per cell
    pub candidates_per_cell: usize,
    /// Size of the grid (number of cells in x and y dimensions)
    pub grid_size: (usize, usize),
    /// Size of each tile for memory coherence (in pixels)
    pub tile_size: (usize, usize),
    /// Convergence threshold for early stopping
    pub convergence_threshold: f32,
}

impl Default for CudaCorrespondenceMappingConfig {
    fn default() -> Self {
        Self {
            iterations: 5,
            candidates_per_cell: 16,
            grid_size: (16, 16),
            tile_size: (32, 32),
            convergence_threshold: 0.01,
        }
    }
}

/// CUDA-accelerated correspondence mapping algorithm
pub struct CudaCorrespondenceMapping {
    /// CUDA context for GPU operations
    context: Arc<CudaContext>,
    /// CUDA kernels
    kernels: Arc<KernelCollection>,
    /// CUDA stream for asynchronous operations
    stream: Stream,
    /// Configuration settings
    config: CudaCorrespondenceMappingConfig,
}

impl CudaCorrespondenceMapping {
    /// Creates a new CUDA-accelerated correspondence mapping algorithm
    pub fn new(
        context: Arc<CudaContext>,
        kernels: Arc<KernelCollection>,
        config: CudaCorrespondenceMappingConfig,
    ) -> Result<Self, CudaError> {
        let stream = Stream::new()?;
        
        Ok(Self {
            context,
            kernels,
            stream,
            config,
        })
    }
    
    /// Runs the correspondence mapping algorithm on the GPU
    pub fn run(
        &self,
        photo1: &Photo,
        photo2: &Photo,
        initial_grid: Option<&AcGrid>,
    ) -> Result<AcGrid, CudaError> {
        // Setup grid dimensions
        let (grid_width, grid_height) = self.config.grid_size;
        let photo1_width = photo1.width;
        let photo1_height = photo1.height;
        let photo2_width = photo2.width;
        let photo2_height = photo2.height;
        
        // Upload photos to GPU memory
        let device_photo1 = self.upload_photo(photo1)?;
        let device_photo2 = self.upload_photo(photo2)?;
        
        // Create initial grid or use provided one
        let mut grid = match initial_grid {
            Some(grid) => grid.clone(),
            None => AcGrid::new(grid_width, grid_height, photo1_width, photo1_height),
        };
        
        // Upload grid to GPU memory, in a tiled layout for memory coherence
        let mut device_grid = self.upload_grid_tiled(&grid)?;
        
        // Create buffers for candidate transforms and scores
        let candidates_per_cell = self.config.candidates_per_cell;
        let transform_size = grid_width * grid_height * candidates_per_cell;
        let mut device_candidates = CudaBuffer::<AffineTransform>::new(&self.context, transform_size)?;
        let mut device_scores = CudaBuffer::<f32>::new(&self.context, transform_size)?;
        
        // Configure CUDA kernel launch parameters
        let tile_size = self.config.tile_size;
        let block_size = (32, 4); // Optimal block size (can be tuned)
        let grid_dim = (
            (grid_width * tile_size.0 + block_size.0 - 1) / block_size.0,
            (grid_height * tile_size.1 + block_size.1 - 1) / block_size.1
        );

        // For each iteration
        for iteration in 0..self.config.iterations {
            // Launch kernel to generate and score candidate transforms
            unsafe {
                if let Some(kernel) = &self.kernels.generate_score_candidates {
                    let params = (
                        device_photo1.as_device_ptr(),
                        device_photo2.as_device_ptr(),
                        device_grid.as_device_ptr(),
                        device_candidates.as_device_ptr(),
                        device_scores.as_device_ptr(),
                        photo1_width as i32,
                        photo1_height as i32,
                        photo2_width as i32,
                        photo2_height as i32,
                        grid_width as i32,
                        grid_height as i32,
                        tile_size.0 as i32,
                        tile_size.1 as i32,
                        candidates_per_cell as i32,
                        iteration as i32
                    );
                    
                    let shared_mem_size = 4 * tile_size.0 * tile_size.1 * std::mem::size_of::<u8>() as u32;
                    
                    launch!(
                        kernel<<<dim3(grid_dim.0 as u32, grid_dim.1 as u32), dim3(block_size.0 as u32, block_size.1 as u32), shared_mem_size, self.stream>>>(
                            params
                        )
                    )?;
                } else {
                    return Err(CudaError::KernelNotFound("generate_score_candidates".to_string()));
                }
            }
            
            // Launch kernel to select the best candidate for each cell
            unsafe {
                if let Some(kernel) = &self.kernels.select_best_candidates {
                    let params = (
                        device_grid.as_device_ptr(),
                        device_candidates.as_device_ptr(),
                        device_scores.as_device_ptr(),
                        grid_width as i32,
                        grid_height as i32,
                        candidates_per_cell as i32
                    );
                    
                    launch!(
                        kernel<<<dim3(grid_width as u32, grid_height as u32), 256, 0, self.stream>>>(
                            params
                        )
                    )?;
                } else {
                    return Err(CudaError::KernelNotFound("select_best_candidates".to_string()));
                }
            }
            
            // Optionally: Perform spatial coherence adjustment
            if iteration < self.config.iterations - 1 {
                unsafe {
                    if let Some(kernel) = &self.kernels.spatial_coherence {
                        let params = (
                            device_grid.as_device_ptr(),
                            grid_width as i32,
                            grid_height as i32
                        );
                        
                        launch!(
                            kernel<<<dim3(grid_width as u32, grid_height as u32), dim3(8, 8), 0, self.stream>>>(
                                params
                            )
                        )?;
                    } else {
                        return Err(CudaError::KernelNotFound("spatial_coherence".to_string()));
                    }
                }
            }
        }
        
        // Download final grid from GPU
        self.download_grid_tiled(&mut grid, &device_grid)?;
        
        Ok(grid)
    }
    
    // Helper method to upload a photo to GPU memory
    fn upload_photo(&self, photo: &Photo) -> Result<CudaBuffer<u8>, CudaError> {
        let mut buffer = CudaBuffer::<u8>::new(&self.context, photo.img_data.len())?;
        buffer.copy_from_host(&photo.img_data)?;
        Ok(buffer)
    }
    
    // Helper method to upload grid in a tiled layout for memory coherence
    fn upload_grid_tiled(&self, grid: &AcGrid) -> Result<CudaBuffer<AffineTransformCell>, CudaError> {
        let width = grid.width;
        let height = grid.height;
        let size = width * height;
        
        // Create a reorganized tiled layout
        let mut tiled_data = Vec::with_capacity(size);
        
        // Tile size for memory layout (can be different from processing tile size)
        let tile_width = 8; // Adjust based on GPU architecture
        let tile_height = 8;
        
        // Number of tiles in each dimension
        let num_tiles_x = (width + tile_width - 1) / tile_width;
        let num_tiles_y = (height + tile_height - 1) / tile_height;
        
        // Organize cells in tiles for better memory coherence
        for ty in 0..num_tiles_y {
            for tx in 0..num_tiles_x {
                for local_y in 0..tile_height {
                    for local_x in 0..tile_width {
                        let x = tx * tile_width + local_x;
                        let y = ty * tile_height + local_y;
                        
                        if x < width && y < height {
                            tiled_data.push(grid.cells[y * width + x].clone());
                        }
                    }
                }
            }
        }
        
        // Upload tiled data to GPU
        let mut buffer = CudaBuffer::<AffineTransformCell>::new(&self.context, tiled_data.len())?;
        buffer.copy_from_host(&tiled_data)?;
        
        Ok(buffer)
    }
    
    // Helper method to download grid from GPU memory, undoing the tiling
    fn download_grid_tiled(
        &self,
        grid: &mut AcGrid,
        device_grid: &CudaBuffer<AffineTransformCell>
    ) -> Result<(), CudaError> {
        let width = grid.width;
        let height = grid.height;
        let size = width * height;
        
        // Download tiled data from GPU
        let mut tiled_data = vec![AffineTransformCell::default(); size];
        device_grid.copy_to_host(&mut tiled_data)?;
        
        // Tile size for memory layout
        let tile_width = 8;
        let tile_height = 8;
        
        // Number of tiles in each dimension
        let num_tiles_x = (width + tile_width - 1) / tile_width;
        let num_tiles_y = (height + tile_height - 1) / tile_height;
        
        // Reorganize from tiled layout back to grid layout
        let mut linear_index = 0;
        
        for ty in 0..num_tiles_y {
            for tx in 0..num_tiles_x {
                for local_y in 0..tile_height {
                    for local_x in 0..tile_width {
                        let x = tx * tile_width + local_x;
                        let y = ty * tile_height + local_y;
                        
                        if x < width && y < height {
                            grid.cells[y * width + x] = tiled_data[linear_index].clone();
                            linear_index += 1;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}

impl CudaError {
    fn KernelNotFound(name: String) -> Self {
        CudaError::KernelLoadFailure(format!("Kernel '{}' not found", name))
    }
}
