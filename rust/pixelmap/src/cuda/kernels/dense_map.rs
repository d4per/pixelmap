//! Dense map kernel launchers

use std::ffi::CString;
use std::os::raw::c_void;
use crate::cuda::memory::CudaStream;
use crate::cuda::memory::CudaError;
use super::core::{KernelCollection, launch_kernel};

impl KernelCollection {
    /// Launch interpolate grid kernel
    pub unsafe fn launch_interpolate_grid(
        &self,
        src_data: *const c_void,
        dst_data: *mut c_void,
        src_width: i32,
        src_height: i32,
        dst_width: i32,
        dst_height: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("interpolate_grid").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &src_data as *const _ as *const c_void,
            &dst_data as *const _ as *const c_void,
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
    
    /// Launch warp photo kernel
    pub unsafe fn launch_warp_photo(
        &self,
        input_photo: *const c_void,
        output_photo: *mut c_void,
        map_data: *const c_void,
        width: i32,
        height: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("warp_photo").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &input_photo as *const _ as *const c_void,
            &output_photo as *const _ as *const c_void,
            &map_data as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
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
    
    /// Launch warp photo bilinear kernel
    pub unsafe fn launch_warp_photo_bilinear(
        &self,
        input_photo: *const c_void,
        output_photo: *mut c_void,
        map_data: *const c_void,
        width: i32,
        height: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("warp_photo_bilinear").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &input_photo as *const _ as *const c_void,
            &output_photo as *const _ as *const c_void,
            &map_data as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
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
    
    /// Launch warp photo optimized kernel
    pub unsafe fn launch_warp_photo_optimized(
        &self,
        input_photo: *const c_void,
        output_photo: *mut c_void,
        map_data: *const c_void,
        width: i32,
        height: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("warp_photo_optimized").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &input_photo as *const _ as *const c_void,
            &output_photo as *const _ as *const c_void,
            &map_data as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
        ];
        
        let block_dim = 16;
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        
        // Calculate shared memory size for tile caching
        let shared_mem_size = block_dim * block_dim * 4; // RGBA
        
        launch_kernel(
            func,
            (grid_dim_x as u32, grid_dim_y as u32, 1),
            (block_dim as u32, block_dim as u32, 1),
            shared_mem_size as u32,
            stream,
            &args,
        )
    }
}
