//! Feature extraction kernel launchers

use std::ffi::CString;
use std::os::raw::c_void;
use crate::cuda::memory::CudaStream;
use crate::cuda::memory::CudaError;
use super::core::{KernelCollection, launch_kernel};

impl KernelCollection {
    /// Launch the feature extraction kernel
    pub unsafe fn launch_extract_features(
        &self,
        photo: *const c_void,
        descriptors: *mut c_void,
        width: i32,
        height: i32,
        circle_radius: i32,
        max_color_value: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("extract_features").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &photo as *const _ as *const c_void,
            &descriptors as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &circle_radius as *const _ as *const c_void,
            &max_color_value as *const _ as *const c_void,
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
    
    /// Launch kernel to find feature matches
    pub unsafe fn launch_find_matches(
        &self,
        descriptors1: *const c_void,
        descriptors2: *const c_void,
        count1: i32,
        count2: i32,
        match_indices: *mut c_void,
        match_distances: *mut c_void,
        max_distance: f32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("find_feature_matches").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &descriptors1 as *const _ as *const c_void,
            &descriptors2 as *const _ as *const c_void,
            &count1 as *const _ as *const c_void,
            &count2 as *const _ as *const c_void,
            &match_indices as *const _ as *const c_void,
            &match_distances as *const _ as *const c_void,
            &max_distance as *const _ as *const c_void,
        ];
        
        let block_dim = 256;
        let grid_dim = (count1 + block_dim - 1) / block_dim;
        
        launch_kernel(
            func,
            (grid_dim as u32, 1, 1),
            (block_dim as u32, 1, 1),
            0, // no shared memory
            stream,
            &args,
        )
    }
    
    /// Launch optimized tiled feature matching kernel 
    pub unsafe fn launch_find_matches_tiled(
        &self,
        descriptors1: *const c_void,
        descriptors2: *const c_void,
        count1: i32,
        count2: i32,
        match_indices: *mut c_void,
        match_distances: *mut c_void,
        max_distance: f32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("find_feature_matches_tiled").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &descriptors1 as *const _ as *const c_void,
            &descriptors2 as *const _ as *const c_void,
            &count1 as *const _ as *const c_void,
            &count2 as *const _ as *const c_void,
            &match_indices as *const _ as *const c_void,
            &match_distances as *const _ as *const c_void,
            &max_distance as *const _ as *const c_void,
        ];
        
        let block_dim = 256;
        let grid_dim = (count1 + block_dim - 1) / block_dim;
        
        // Calculate shared memory size (7 floats per descriptor)
        let shared_mem_size = block_dim as u32 * 7 * std::mem::size_of::<f32>() as u32;
        
        launch_kernel(
            func,
            (grid_dim as u32, 1, 1),
            (block_dim as u32, 1, 1),
            shared_mem_size,
            stream,
            &args,
        )
    }
}
