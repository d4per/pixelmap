//! Photo processing kernel launchers

use std::ffi::CString;
use std::os::raw::c_void;
use crate::cuda::memory::CudaStream;
use crate::cuda::memory::CudaError;
use super::core::{KernelCollection, launch_kernel};

impl KernelCollection {
    /// Launch Gaussian blur kernel
    pub unsafe fn launch_gaussian_blur(
        &self,
        input: *const c_void,
        output: *mut c_void,
        width: i32,
        height: i32,
        sigma: f32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("gaussian_blur").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &input as *const _ as *const c_void,
            &output as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &sigma as *const _ as *const c_void,
        ];
        
        let block_dim = 16;
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        
        launch_kernel(
            func,
            (grid_dim_x as u32, grid_dim_y as u32, 1),
            (block_dim as u32, block_dim as u32, 1),
            0, // no shared memory
            stream,
            &args,
        )
    }
    
    /// Launch optimized Gaussian blur kernel with shared memory
    pub unsafe fn launch_gaussian_blur_optimized(
        &self,
        input: *const c_void,
        output: *mut c_void,
        width: i32,
        height: i32,
        sigma: f32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("gaussian_blur_optimized").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &input as *const _ as *const c_void,
            &output as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &sigma as *const _ as *const c_void,
        ];
        
        let block_dim = 16;
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        
        // Calculate radius based on sigma
        let radius = (3.0 * sigma + 0.5) as i32;
        
        // Calculate shared memory size
        let padded_tile_width = block_dim + 2 * radius;
        let padded_tile_height = block_dim + 2 * radius;
        let shared_mem_size = padded_tile_width * padded_tile_height * 4; // RGBA
        
        launch_kernel(
            func,
            (grid_dim_x as u32, grid_dim_y as u32, 1),
            (block_dim as u32, block_dim as u32, 1),
            shared_mem_size as u32,
            stream,
            &args,
        )
    }
    
    /// Launch scale bicubic kernel
    pub unsafe fn launch_scale_bicubic(
        &self,
        input: *const c_void,
        output: *mut c_void,
        src_width: i32,
        src_height: i32,
        dst_width: i32,
        dst_height: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("scale_bicubic").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &input as *const _ as *const c_void,
            &output as *const _ as *const c_void,
            &src_width as *const _ as *const c_void,
            &src_height as *const _ as *const c_void,
            &dst_width as *const _ as *const c_void,
            &dst_height as *const _ as *const c_void,
        ];
        
        let block_dim = 16;
        let grid_dim_x = (dst_width + block_dim - 1) / block_dim;
        let grid_dim_y = (dst_height + block_dim - 1) / block_dim;
        
        launch_kernel(
            func,
            (grid_dim_x as u32, grid_dim_y as u32, 1),
            (block_dim as u32, block_dim as u32, 1),
            0, // no shared memory
            stream,
            &args,
        )
    }
}
