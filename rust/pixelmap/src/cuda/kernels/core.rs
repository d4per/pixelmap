//! Core kernel management functionality

use std::ffi::CString;
use std::os::raw::c_void;
use crate::cuda::memory::{CudaError, CudaStream};
use crate::cuda::bindings;

/// Collection of CUDA kernels
pub struct KernelCollection {
    /// CUDA module containing compiled PTX code
    pub module: *mut c_void,
}

impl KernelCollection {
    pub fn new() -> Result<Self, CudaError> {
        use std::env;
        use std::path::Path;
        use std::fs;
        
        // Load the PTX files from the build directory
        let ptx_dir = env::var("CUDA_PTX_DIR")
            .map_err(|_| CudaError::FileError("CUDA_PTX_DIR environment variable not set".to_string()))?;
        
        let ptx_path = Path::new(&ptx_dir);
        if !ptx_path.exists() {
            return Err(CudaError::FileError(format!("PTX directory not found: {}", ptx_dir)));
        }
        
        // Load a combined module from all PTX files
        let mut ptx_data = String::new();
        let mut found_files = false;
        
        for entry in fs::read_dir(ptx_path)
            .map_err(|e| CudaError::FileError(format!("Failed to read PTX directory: {}", e)))? {
            
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "ptx") {
                    let content = fs::read_to_string(&path)
                        .map_err(|e| CudaError::FileError(format!("Failed to read PTX file {}: {}", 
                            path.display(), e)))?;
                    
                    ptx_data.push_str(&content);
                    ptx_data.push('\n');
                    found_files = true;
                }
            }
        }
        
        if !found_files {
            return Err(CudaError::FileError("No PTX files found".to_string()));
        }
        
        // Load the module
        let module = bindings::load_module_data(&ptx_data)
            .map_err(|e| CudaError::KernelLoadFailure(e))?;
        
        Ok(Self { module })
    }
    
    /// Get a function from the module
    pub fn get_function(&self, name: &CString) -> Result<*mut c_void, CudaError> {
        bindings::get_function(self.module, name.to_str().unwrap())
            .map_err(|e| CudaError::KernelNotFound(e))
    }
}

impl Drop for KernelCollection {
    fn drop(&mut self) {
        unsafe {
            if !self.module.is_null() {
                // If module dropping fails, just log the error - can't propagate from Drop
                if let Err(e) = bindings::unload_module(self.module) {
                    eprintln!("Failed to unload CUDA module: {}", e);
                }
            }
        }
    }
}

/// Launch a kernel with the given parameters
pub unsafe fn launch_kernel(
    func: *mut c_void,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_size: u32,
    stream: CudaStream,
    args: &[*const c_void],
) -> Result<(), CudaError> {
    bindings::launch_kernel(func, grid_dim, block_dim, shared_mem_size, *stream, args)
        .map_err(|e| CudaError::Other(e))
}
