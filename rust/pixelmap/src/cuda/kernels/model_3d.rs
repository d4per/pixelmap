//! 3D model generation kernel launchers

use std::ffi::CString;
use std::os::raw::c_void;
use crate::cuda::memory::CudaStream;
use crate::cuda::memory::CudaError;
use super::core::{KernelCollection, launch_kernel};

impl KernelCollection {
    /// Launch kernel to generate vertices from DensePhotoMap
    pub unsafe fn launch_generate_vertices(
        &self,
        map_x: *const c_void,
        map_y: *const c_void,
        map_used: *const c_void,
        vertices: *mut c_void,
        width: i32,
        height: i32,
        z_scale: f32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("generate_vertices").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &map_x as *const _ as *const c_void,
            &map_y as *const _ as *const c_void,
            &map_used as *const _ as *const c_void,
            &vertices as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &z_scale as *const _ as *const c_void,
        ];
        
        let block_dim = 16;
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        
        launch_kernel(
            func,
            (grid_dim_x as u32, grid_dim_y as u32, 1),
            (block_dim as u32, block_dim as u32, 1),
            0, // No shared memory needed
            stream,
            &args,
        )
    }
    
    /// Launch kernel to generate triangle normals
    pub unsafe fn launch_generate_normals(
        &self,
        vertices: *const c_void,
        triangles: *const c_void,
        normals: *mut c_void,
        vertex_count: i32,
        triangle_count: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("generate_normals").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &vertices as *const _ as *const c_void,
            &triangles as *const _ as *const c_void,
            &normals as *const _ as *const c_void,
            &vertex_count as *const _ as *const c_void,
            &triangle_count as *const _ as *const c_void,
        ];
        
        let block_dim = 256;
        let grid_dim = (triangle_count + block_dim - 1) / block_dim;
        
        launch_kernel(
            func,
            (grid_dim as u32, 1, 1),
            (block_dim as u32, 1, 1),
            0, // No shared memory needed
            stream,
            &args,
        )
    }
    
    /// Launch kernel to apply texture coordinates
    pub unsafe fn launch_apply_texture_coordinates(
        &self,
        vertices: *mut c_void,
        map_data: *const c_void,
        vertex_count: i32,
        width: i32,
        height: i32,
        stream: CudaStream,
    ) -> Result<(), CudaError> {
        let kernel_name = CString::new("apply_texture_coordinates").unwrap();
        let func = self.get_function(&kernel_name)?;
        
        let args = [
            &vertices as *const _ as *const c_void,
            &map_data as *const _ as *const c_void,
            &vertex_count as *const _ as *const c_void,
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
        ];
        
        let block_dim = 256;
        let grid_dim = (vertex_count + block_dim - 1) / block_dim;
        
        launch_kernel(
            func,
            (grid_dim as u32, 1, 1),
            (block_dim as u32, 1, 1),
            0, // No shared memory needed
            stream,
            &args,
        )
    }
}
