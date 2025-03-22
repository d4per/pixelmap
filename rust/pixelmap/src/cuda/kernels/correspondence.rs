//! Correspondence mapping kernel launchers

use std::ffi::CString;
use std::os::raw::c_void;
use crate::cuda::memory::CudaStream;
use crate::cuda::memory::CudaError;
use super::core::{KernelCollection, launch_kernel};

impl KernelCollection {
    /// Launch compute correspondence scores kernel
    pub unsafe fn launch_compute_correspondence_scores(
        &self,
        source_photo: *const c_void,
        target_photo: *const c_void,
        source_width: i32,
        source_height: i32,
        target_width: i32,
        target_height: i32,
        transforms: *const c_void,
        points: *const c_void,
        transform_count: i32,
        scores: *mut c_void,
        patch_size: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("computeCorrespondenceScores").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &source_photo as *const _ as *const c_void,
            &target_photo as *const _ as *const c_void,
            &source_width as *const _ as *const c_void,
            &source_height as *const _ as *const c_void,
            &target_width as *const _ as *const c_void,
            &target_height as *const _ as *const c_void,
            &transforms as *const _ as *const c_void,
            &points as *const _ as *const c_void,
            &transform_count as *const _ as *const c_void,
            &scores as *const _ as *const c_void,
            &patch_size as *const _ as *const c_void,
        ];
        
        let block_size = 256; // Threads per block
        let grid_size = (transform_count + block_size - 1) / block_size;
        
        launch_kernel(
            func,
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            0, // no shared memory
            stream,
            &args,
        )
    }
    
    /// Launch remove outliers kernel
    pub unsafe fn launch_remove_outliers(
        &self,
        map_data: *mut c_void,
        other_map_data: *const c_void,
        width: i32,
        height: i32,
        max_dist: f32,
        stream: CudaStream,
        shared_mem_size: u32,
    ) -> Result<(), CudaError> {
        // Try optimized version first
        if let Ok(result) = self.launch_remove_outliers_optimized(
            map_data,
            other_map_data,
            width,
            height,
            max_dist,
            stream,
            shared_mem_size,
        ) {
            return Ok(result);
        }
        
        // Fall back to standard version
        let kernel_name = CString::new("removeOutliers").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &map_data as *const _ as *const c_void,
            &other_map_data as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &max_dist as *const _ as *const c_void,
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
    
    /// Launch optimized remove outliers kernel
    pub unsafe fn launch_remove_outliers_optimized(
        &self,
        map_data: *mut c_void,
        other_map_data: *const c_void,
        width: i32,
        height: i32,
        max_dist: f32,
        stream: CudaStream,
        shared_mem_size: u32,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("remove_outliers_optimized").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &map_data as *const _ as *const c_void,
            &other_map_data as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &max_dist as *const _ as *const c_void,
        ];
        
        let block_dim = 16;
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        
        launch_kernel(
            func,
            (grid_dim_x as u32, grid_dim_y as u32, 1),
            (block_dim as u32, block_dim as u32, 1),
            shared_mem_size,
            stream,
            &args,
        )
    }
    
    /// Launch smooth grid points kernel
    pub unsafe fn launch_smooth_grid_points(
        &self,
        in_x: *const c_void,
        in_y: *const c_void,
        in_used: *const c_void,
        out_x: *mut c_void,
        out_y: *mut c_void,
        out_used: *mut c_void,
        width: i32,
        height: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("smoothGridPoints").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &in_x as *const _ as *const c_void,
            &in_y as *const _ as *const c_void,
            &in_used as *const _ as *const c_void,
            &out_x as *const _ as *const c_void,
            &out_y as *const _ as *const c_void,
            &out_used as *const _ as *const c_void,
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
    
    /// Launch optimized smooth grid points kernel
    pub unsafe fn launch_smooth_grid_points_optimized(
        &self,
        in_data: *const c_void,
        out_data: *mut c_void,
        width: i32,
        height: i32,
        stream: CudaStream,
        shared_mem_size: u32,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("smooth_grid_points_optimized").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &in_data as *const _ as *const c_void,
            &out_data as *const _ as *const c_void,
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
            shared_mem_size,
            stream,
            &args,
        )
    }
}
