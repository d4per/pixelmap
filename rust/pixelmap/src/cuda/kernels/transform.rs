//! Transform kernel launchers

use std::ffi::CString;
use std::os::raw::c_void;
use crate::cuda::memory::CudaStream;
use crate::cuda::memory::CudaError;
use super::core::{KernelCollection, launch_kernel};

impl KernelCollection {
    /// Launch batch transform kernel
    pub unsafe fn launch_batch_transform(
        &self,
        transforms: *const c_void,
        points: *const c_void,
        transform_count: i32,
        point_count: i32,
        results: *mut c_void,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("batch_transform_coalesced").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &transforms as *const _ as *const c_void,
            &points as *const _ as *const c_void,
            &results as *const _ as *const c_void,
            &transform_count as *const _ as *const c_void,
            &point_count as *const _ as *const c_void,
        ];
        
        // Calculate total number of threads needed
        let total_work = transform_count * point_count;
        let block_size = 256;
        let grid_size = (total_work + block_size - 1) / block_size;
        
        launch_kernel(
            func,
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            0, // No shared memory needed
            stream,
            &args,
        )
    }
    
    /// Launch kernel to generate transform candidates
    pub unsafe fn launch_generate_transform_candidates(
        &self,
        base_transform: *const c_void,
        candidates: *mut c_void,
        seeds: *const c_void,
        count: i32,
        variation_scale: f32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("generate_transform_candidates").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &base_transform as *const _ as *const c_void,
            &candidates as *const _ as *const c_void,
            &seeds as *const _ as *const c_void,
            &count as *const _ as *const c_void,
            &variation_scale as *const _ as *const c_void,
        ];
        
        let block_size = 256;
        let grid_size = (count + block_size - 1) / block_size;
        
        launch_kernel(
            func,
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            0, // No shared memory needed
            stream,
            &args,
        )
    }
    
    /// Launch kernel for finding the best transform
    pub unsafe fn launch_find_best_transform(
        &self,
        source_photo: *const c_void,
        target_photo: *const c_void,
        transforms: *const c_void,
        transform_count: i32,
        scores: *mut c_void,
        best_transform_idx: *mut c_void,
        width: i32,
        height: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("find_best_transform").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &source_photo as *const _ as *const c_void,
            &target_photo as *const _ as *const c_void,
            &transforms as *const _ as *const c_void,
            &transform_count as *const _ as *const c_void,
            &scores as *const _ as *const c_void,
            &best_transform_idx as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
        ];
        
        let block_size = 256;
        let grid_size = (transform_count + block_size - 1) / block_size;
        
        launch_kernel(
            func,
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            0, // No shared memory needed
            stream,
            &args,
        )
    }
    
    /// Launch transform composition kernel
    pub unsafe fn launch_compose_transforms(
        &self,
        transforms1: *const c_void,
        transforms2: *const c_void,
        result_transforms: *mut c_void,
        count: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("compose_transforms").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &transforms1 as *const _ as *const c_void,
            &transforms2 as *const _ as *const c_void,
            &result_transforms as *const _ as *const c_void,
            &count as *const _ as *const c_void,
        ];
        
        let block_size = 256;
        let grid_size = (count + block_size - 1) / block_size;
        
        launch_kernel(
            func,
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            0, // No shared memory needed
            stream,
            &args,
        )
    }
    
    /// Launch transform inverse kernel
    pub unsafe fn launch_invert_transforms(
        &self,
        transforms: *const c_void,
        inverse_transforms: *mut c_void,
        count: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("invert_transforms").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &transforms as *const _ as *const c_void,
            &inverse_transforms as *const _ as *const c_void,
            &count as *const _ as *const c_void,
        ];
        
        let block_size = 256;
        let grid_size = (count + block_size - 1) / block_size;
        
        launch_kernel(
            func,
            (grid_size as u32, 1, 1),
            (block_size as u32, 1, 1),
            0, // No shared memory needed
            stream,
            &args,
        )
    }
}
