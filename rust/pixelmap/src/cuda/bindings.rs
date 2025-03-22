//! Raw FFI bindings to CUDA driver API
//!
//! This module provides low-level bindings to the CUDA driver API
//! for memory management, context creation, and kernel launches.

use std::ffi::{c_void, CString};
use std::sync::Once;

// Ensure we initialize CUDA driver API once
static INIT_DRIVER_API: Once = Once::new();

// Initialize CUDA driver API
pub fn init_cuda_driver() -> Result<(), String> {
    let mut result = Ok(());
    
    INIT_DRIVER_API.call_once(|| {
        unsafe {
            let init_result = cuda_driver_sys::cuInit(0);
            if init_result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
                let mut err_str: *mut ::std::os::raw::c_char = std::ptr::null_mut();
                cuda_driver_sys::cuGetErrorString(init_result, &mut err_str);
                let error_msg = if err_str.is_null() {
                    format!("Failed to initialize CUDA: error code {:?}", init_result)
                } else {
                    let c_str = std::ffi::CStr::from_ptr(err_str);
                    format!("Failed to initialize CUDA: {}", c_str.to_string_lossy())
                };
                result = Err(error_msg);
            }
        }
    });
    
    result
}

// Get CUDA device count
pub fn get_device_count() -> Result<i32, String> {
    init_cuda_driver()?;
    
    let mut count = 0;
    unsafe {
        let result = cuda_driver_sys::cuDeviceGetCount(&mut count);
        if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            let mut err_str: *mut ::std::os::raw::c_char = std::ptr::null_mut();
            cuda_driver_sys::cuGetErrorString(result, &mut err_str);
            let error_msg = if err_str.is_null() {
                format!("Failed to get device count: error code {:?}", result)
            } else {
                let c_str = std::ffi::CStr::from_ptr(err_str);
                format!("Failed to get device count: {}", c_str.to_string_lossy())
            };
            return Err(error_msg);
        }
    }
    
    Ok(count)
}

// Get CUDA device by index
pub fn get_device(index: i32) -> Result<cuda_driver_sys::CUdevice, String> {
    init_cuda_driver()?;
    
    let mut device = 0;
    unsafe {
        let result = cuda_driver_sys::cuDeviceGet(&mut device, index);
        if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            let mut err_str: *mut ::std::os::raw::c_char = std::ptr::null_mut();
            cuda_driver_sys::cuGetErrorString(result, &mut err_str);
            let error_msg = if err_str.is_null() {
                format!("Failed to get device {}: error code {:?}", index, result)
            } else {
                let c_str = std::ffi::CStr::from_ptr(err_str);
                format!("Failed to get device {}: {}", index, c_str.to_string_lossy())
            };
            return Err(error_msg);
        }
    }
    
    Ok(device)
}

// Create CUDA context
pub fn create_context(device: cuda_driver_sys::CUdevice) -> Result<*mut c_void, String> {
    let mut context = std::ptr::null_mut();
    
    unsafe {
        let result = cuda_driver_sys::cuCtxCreate_v2(
            &mut context,
            cuda_driver_sys::CU_CTX_SCHED_AUTO as u32,
            device
        );
        
        if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            let mut err_str: *mut ::std::os::raw::c_char = std::ptr::null_mut();
            cuda_driver_sys::cuGetErrorString(result, &mut err_str);
            let error_msg = if err_str.is_null() {
                format!("Failed to create context: error code {:?}", result)
            } else {
                let c_str = std::ffi::CStr::from_ptr(err_str);
                format!("Failed to create context: {}", c_str.to_string_lossy())
            };
            return Err(error_msg);
        }
    }
    
    Ok(context)
}

// Get kernel function from a module
pub fn get_function(module: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let name_c = CString::new(name).map_err(|e| {
        format!("Invalid kernel name: {}", e)
    })?;
    
    let mut func = std::ptr::null_mut();
    
    unsafe {
        let result = cuda_driver_sys::cuModuleGetFunction(
            &mut func,
            module as cuda_driver_sys::CUmodule,
            name_c.as_ptr()
        );
        
        if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            let mut err_str: *mut ::std::os::raw::c_char = std::ptr::null_mut();
            cuda_driver_sys::cuGetErrorString(result, &mut err_str);
            let error_msg = if err_str.is_null() {
                format!("Failed to get function '{}': error code {:?}", name, result)
            } else {
                let c_str = std::ffi::CStr::from_ptr(err_str);
                format!("Failed to get function '{}': {}", name, c_str.to_string_lossy())
            };
            return Err(error_msg);
        }
    }
    
    Ok(func)
}

// Launch a kernel
pub unsafe fn launch_kernel(
    func: *mut c_void,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: *mut c_void,
    args: &[*mut c_void]
) -> Result<(), String> {
    let result = cuda_driver_sys::cuLaunchKernel(
        func as cuda_driver_sys::CUfunction,
        grid_dim.0, grid_dim.1, grid_dim.2,
        block_dim.0, block_dim.1, block_dim.2,
        shared_mem_bytes,
        stream as cuda_driver_sys::CUstream,
        args.as_ptr() as *mut *mut c_void,
        std::ptr::null_mut()
    );
    
    if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
        let mut err_str: *mut ::std::os::raw::c_char = std::ptr::null_mut();
        cuda_driver_sys::cuGetErrorString(result, &mut err_str);
        let error_msg = if err_str.is_null() {
            format!("Failed to launch kernel: error code {:?}", result)
        } else {
            let c_str = std::ffi::CStr::from_ptr(err_str);
            format!("Failed to launch kernel: {}", c_str.to_string_lossy())
        };
        return Err(error_msg);
    }
    
    Ok(())
}

/// Load a CUDA module from PTX data
pub fn load_module_data(ptx_data: &str) -> Result<*mut c_void, String> {
    let ptx_c_str = CString::new(ptx_data)
        .map_err(|e| format!("Failed to convert PTX data to C string: {}", e))?;
    
    let mut module = std::ptr::null_mut();
    
    unsafe {
        let result = cuda_driver_sys::cuModuleLoadData(
            &mut module,
            ptx_c_str.as_ptr() as *const c_void
        );
        
        if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            let mut err_str: *mut ::std::os::raw::c_char = std::ptr::null_mut();
            cuda_driver_sys::cuGetErrorString(result, &mut err_str);
            let error_msg = if err_str.is_null() {
                format!("Failed to load module: error code {:?}", result)
            } else {
                let c_str = std::ffi::CStr::from_ptr(err_str);
                format!("Failed to load module: {}", c_str.to_string_lossy())
            };
            return Err(error_msg);
        }
    }
    
    Ok(module)
}
