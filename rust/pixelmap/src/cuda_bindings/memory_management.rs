//! Memory management utilities for CUDA operations

use crate::cuda::memory::{CudaBuffer, CudaContext, CudaError, CudaStream};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, Once};
use std::ffi::{c_void, CStr};

// Remove the explicit extern crate declaration and directly use the crate
use cuda_driver_sys;

// Memory pool for reusing allocations
pub struct MemoryPool {
    buffers: HashMap<usize, Vec<CudaBuffer<u8>>>,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }
    
    pub fn get_buffer(&mut self, size: usize) -> Result<CudaBuffer<u8>, CudaError> {
        if let Some(buffers) = self.buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer);
            }
        }
        
        // No existing buffer, allocate new one
        CudaBuffer::new(size)
    }
    
    pub fn return_buffer(&mut self, buffer: CudaBuffer<u8>) {
        let size = buffer.capacity();
        self.buffers.entry(size).or_insert_with(Vec::new).push(buffer);
    }
}

static mut MEMORY_POOL_INSTANCE: Option<MemoryPool> = None;
static INIT: Once = Once::new();

// Get a reference to the memory pool, initialize it if needed
pub fn get_memory_pool() -> &'static mut MemoryPool {
    unsafe {
        INIT.call_once(|| {
            MEMORY_POOL_INSTANCE = Some(MemoryPool::new());
        });
        MEMORY_POOL_INSTANCE.as_mut().unwrap()
    }
}

/// Helper trait for working with pinned memory
pub trait PinnedMemory<T: Copy> {
    /// Create pinned memory for faster host-device transfers
    fn create_pinned_memory(count: usize) -> Result<Vec<T>, String>;
    
    /// Upload data from pinned memory to device with optimal alignment
    fn upload_from_pinned(data: &[T]) -> Result<CudaBuffer<T>, String>;
}

/// Implementation for pinned memory management
pub struct CudaPinnedMemory;

// Fix the corrupted implementation of PinnedMemory trait
impl<T: Copy> PinnedMemory<T> for CudaPinnedMemory {
    fn create_pinned_memory(count: usize) -> Result<Vec<T>, String> {
        let size = count * std::mem::size_of::<T>();
        let mut host_ptr = std::ptr::null_mut();
        
        unsafe {
            let result = cuda_driver_sys::cuMemHostAlloc(
                &mut host_ptr as *mut _ as *mut *mut c_void,
                size,
                cuda_driver_sys::CU_MEMHOSTALLOC_PORTABLE as u32
            );
            
            if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
                let mut err_str: *const i8 = std::ptr::null();
                cuda_driver_sys::cuGetErrorString(result, &mut err_str as *mut *const i8);
                let error_msg = if err_str.is_null() {
                    format!("Failed to allocate pinned memory: error code {:?}", result)
                } else {
                    let c_str = std::ffi::CStr::from_ptr(err_str);
                    format!("Failed to allocate pinned memory: {}", c_str.to_string_lossy())
                };
                return Err(error_msg);
            }
            
            // Create a Vec that will free the pinned memory when dropped
            let mut vec = Vec::from_raw_parts(
                host_ptr as *mut T,
                count,
                count
            );
            
            // Initialize to zero
            for item in &mut vec {
                *item = std::mem::zeroed();
            }
            
            Ok(vec)
        }
    }
    
    fn upload_from_pinned(data: &[T]) -> Result<CudaBuffer<T>, String> {
        let mut buffer = CudaBuffer::new(data.len())
            .map_err(|e| format!("Failed to allocate device memory: {:?}", e))?;
            
        buffer.copy_from_host(data)
            .map_err(|e| format!("Failed to upload data: {:?}", e))?;
            
        Ok(buffer)
    }
}

pub struct MemoryManager {
    context: CudaContext,
    stream: CudaStream,
}

impl MemoryManager {
    pub fn new() -> Result<Self, String> {
        let context = CudaContext::new().map_err(|e| format!("Failed to create CUDA context: {:?}", e))?;
        let stream = context.create_stream().map_err(|e| format!("Failed to create CUDA stream: {:?}", e))?;
        Ok(Self { context, stream })
    }

    pub fn allocate_host_memory<T>(&self, size: usize) -> Result<*mut T, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            let result = cuda_driver_sys::cuMemHostAlloc(
                &mut ptr,
                size,
                cuda_driver_sys::CU_MEMHOSTALLOC_PORTABLE as u32,
            );

            if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
                let mut err_str: *const i8 = std::ptr::null();
                cuda_driver_sys::cuGetErrorString(result, &mut err_str as *mut *const i8);
                let error_msg = if err_str.is_null() {
                    format!("CUDA error code: {:?}", result)
                } else {
                    let c_str = std::ffi::CStr::from_ptr(err_str);
                    c_str.to_string_lossy().to_string()
                };
                return Err(format!("Failed to allocate host memory: {}", error_msg));
            }
        }
        Ok(ptr as *mut T)
    }

    pub fn allocate_and_copy_buffer<T: Copy>(&self, data: &[T]) -> Result<CudaBuffer<T>, String> {
        let mut buffer = CudaBuffer::new(data.len())
            .map_err(|e| format!("Failed to allocate memory: {:?}", e))?;
        buffer.copy_from_host(data)
            .map_err(|e| format!("Failed to copy data to device: {:?}", e))?;
        Ok(buffer)
    }
}
