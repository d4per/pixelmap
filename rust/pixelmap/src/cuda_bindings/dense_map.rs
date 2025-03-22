//! CUDA-accelerated dense map operations

use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;
use crate::affine_transform::AffineTransform;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaStream};
// Comment out missing imports
// use crate::cuda::kernels::KernelCollection;
use super::common::{get_cuda_context, get_kernel_collection};
// Import the KernelCollection from common
use super::common::KernelCollection;
use std::ffi::CString;

/// CUDA-accelerated implementation of DensePhotoMap operations
pub struct CudaDenseMapProcessor {
    context: CudaContext,
    stream: CudaStream,
}

impl CudaDenseMapProcessor {
    /// Create a new CudaDenseMapProcessor
    pub fn new() -> Result<Self, String> {
        let context = get_cuda_context()?;
        let stream = context.create_stream()
            .map_err(|e| format!("Failed to create CUDA stream: {:?}", e))?;
            
        Ok(Self {
            context,
            stream,
        })
    }
    
    /// Accelerated interpolation of the mapping
    pub fn interpolate_mapping(&self, map: &DensePhotoMap, target_width: usize, target_height: usize) -> Result<DensePhotoMap, String> {
        let kernels = get_kernel_collection()?;
        // Get source dimensions
        let src_width = map.grid_width;
        let src_height = map.grid_height;
        // Get packed map data
        let src_data = map.get_packed_data();
        // Create pinned memory for inputs and outputs
        let mut d_src_data = CudaBuffer::new(src_data.len())
            .map_err(|e| format!("Failed to allocate memory for source data: {:?}", e))?;
        d_src_data.copy_from_host(&src_data)
            .map_err(|e| format!("Failed to upload source data: {:?}", e))?;
        let dst_data_len = target_width * target_height * 3; // x, y, used
        let mut d_dst_data = CudaBuffer::new(dst_data_len)
            .map_err(|e| format!("Failed to allocate memory for target data: {:?}", e))?;
        // Configure kernel launch
        let block_dim = 16;
        let grid_dim_x = (target_width + block_dim - 1) / block_dim;
        let grid_dim_y = (target_height + block_dim - 1) / block_dim;
        unsafe {
            // Check if interpolate_grid kernel exists
            let kernel_name = std::ffi::CString::new("interpolate_grid").unwrap();
            if let Ok(_) = kernels.get_function(&kernel_name) {
                kernels.launch_interpolate_grid(
                    d_src_data.as_device_ptr(),
                    d_dst_data.as_device_ptr_mut(),
                    src_width as i32,
                    src_height as i32,
                    target_width as i32,
                    target_height as i32,
                    self.stream.clone(),  // Use clone instead of move
                ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
                // Download result
                let mut dst_data = vec![0.0f32; dst_data_len];
                d_dst_data.copy_to_host(&mut dst_data)
                    .map_err(|e| format!("Failed to download result: {:?}", e))?;
                // Create new map
                return Ok(DensePhotoMap::from_packed_data(
                    map.photo1.clone(),
                    map.photo2.clone(),
                    target_width,
                    target_height,
                    &dst_data,
                ));
            }
        }
        // Fall back to CPU implementation if kernel not found
        Ok(map.interpolate(target_width, target_height))
    }

    /// Accelerated warp of a photo using a mapping
    pub fn warp_photo(&self, photo: &Photo, map: &DensePhotoMap) -> Result<Photo, String> {
        let kernels = get_kernel_collection()?;
        // Upload photo data
        let d_photo = self.upload_photo(photo)?;
        // Upload map data in packed format
        let map_data = map.get_packed_data();
        let mut d_map_data = CudaBuffer::new(map_data.len())
            .map_err(|e| format!("Failed to allocate memory for map data: {:?}", e))?;
        d_map_data.copy_from_host(&map_data)
            .map_err(|e| format!("Failed to upload map data: {:?}", e))?;
        // Allocate output photo buffer
        let mut d_output = CudaBuffer::new(photo.width * photo.height * 4)
            .map_err(|e| format!("Failed to allocate memory for output: {:?}", e))?;
        // Configure and launch kernel
        let block_size = 16;
        let grid_dim_x = (photo.width + block_size - 1) / block_size;
        let grid_dim_y = (photo.height + block_size - 1) / block_size;
        unsafe {
            let kernel_name = std::ffi::CString::new("warp_photo").unwrap();
            if let Ok(_) = kernels.get_function(&kernel_name) {
                kernels.launch_warp_photo(
                    d_photo.as_device_ptr(),
                    d_output.as_device_ptr_mut(),
                    d_map_data.as_device_ptr(),
                    photo.width as i32,
                    photo.height as i32,
                    self.stream.clone(),  // Use clone instead of move
                ).map_err(|e| format!("Failed to launch warp kernel: {:?}", e))?;
                // Download result
                let mut output_buffer = vec![0u8; photo.width * photo.height * 4];
                d_output.copy_to_host(&mut output_buffer)
                    .map_err(|e| format!("Failed to download warped photo: {:?}", e))?;
                return Ok(Photo {
                    img_data: output_buffer, // Changed from buffer to img_data
                    width: photo.width,
                    height: photo.height,
                });
            }
        }
        // Fallback to CPU implementation
        Ok(map.warp(photo))
    }

    /// Accelerated warp of a photo using a mapping with filtering
    pub fn warp_photo_with_filter(&self, 
        photo: &Photo, 
        map: &DensePhotoMap, 
        filter_type: FilterType
    ) -> Result<Photo, String> {
        let kernels = get_kernel_collection()?;
        // Upload photo data
        let d_photo = self.upload_photo(photo)?;
        // Upload map data in packed format
        let map_data = map.get_packed_data();
        let mut d_map_data = CudaBuffer::new(map_data.len())
            .map_err(|e| format!("Failed to allocate memory for map data: {:?}", e))?;
        d_map_data.copy_from_host(&map_data)
            .map_err(|e| format!("Failed to upload map data: {:?}", e))?;
        // Allocate output photo buffer
        let mut d_output = CudaBuffer::new(photo.width * photo.height * 4)
            .map_err(|e| format!("Failed to allocate memory for output: {:?}", e))?;
        // Configure and launch kernel based on filter type
        let block_size = 16;
        let grid_dim_x = (photo.width + block_size - 1) / block_size;
        let grid_dim_y = (photo.height + block_size - 1) / block_size;
        let success = unsafe {
            match filter_type {
                FilterType::Nearest => {
                    let kernel_name = std::ffi::CString::new("warp_photo").unwrap();
                    if let Ok(_) = kernels.get_function(&kernel_name) {
                        kernels.launch_warp_photo(
                            d_photo.as_device_ptr(),
                            d_output.as_device_ptr_mut(),
                            d_map_data.as_device_ptr(),
                            photo.width as i32,
                            photo.height as i32,
                            self.stream.clone(),  // Use clone instead of move
                        ).map_err(|e| format!("Failed to launch warp kernel: {:?}", e))?;
                        true
                    } else {
                        false
                    }
                },
                FilterType::Bilinear => {
                    let kernel_name = std::ffi::CString::new("warp_photo_bilinear").unwrap();
                    if let Ok(_) = kernels.get_function(&kernel_name) {
                        kernels.launch_warp_photo_bilinear(
                            d_photo.as_device_ptr(),
                            d_output.as_device_ptr_mut(),
                            d_map_data.as_device_ptr(),
                            photo.width as i32,
                            photo.height as i32,
                            self.stream.clone(),  // Use clone instead of move
                        ).map_err(|e| format!("Failed to launch warp bilinear kernel: {:?}", e))?;
                        true
                    } else {
                        false
                    }
                },
                FilterType::Bicubic => {
                    // Currently no CUDA implementation for bicubic
                    false
                }
            }
        };
        if success {
            // Download result
            let mut output_buffer = vec![0u8; photo.width * photo.height * 4];
            d_output.copy_to_host(&mut output_buffer)
                .map_err(|e| format!("Failed to download warped photo: {:?}", e))?;
            return Ok(Photo {
                img_data: output_buffer, // Changed from buffer to img_data
                width: photo.width,
                height: photo.height,
            });
        }
        // Fall back to CPU implementation
        match filter_type {
            FilterType::Nearest => Ok(map.warp(photo)),
            FilterType::Bilinear => Ok(map.warp_bilinear(photo)),
            FilterType::Bicubic => Ok(map.warp_bicubic(photo)),
        }
    }

    /// Accelerated blending of two photos using a mapping
    pub fn blend_photos(&self, 
        photo1: &Photo, 
        photo2: &Photo, 
        map: &DensePhotoMap,
        blend_factor: f32
    ) -> Result<Photo, String> {
        // TODO: Implement CUDA kernel for photo blending
        // For now, fallback to CPU implementation
        Ok(map.blend(photo1, photo2, blend_factor))
    }

    /// Upload a photo to GPU memory
    fn upload_photo(&self, photo: &Photo) -> Result<CudaBuffer<u8>, String> {
        let mut buffer = CudaBuffer::new(photo.width * photo.height * 4)
            .map_err(|e| format!("Failed to allocate memory for photo: {:?}", e))?;
            
        buffer.copy_from_host(&photo.img_data) // Changed from buffer to img_data
            .map_err(|e| format!("Failed to upload photo: {:?}", e))?;
            
        Ok(buffer)
    }
}

/// Filter types for photo warping
pub enum FilterType {
    Nearest,
    Bilinear,
    Bicubic,
}

