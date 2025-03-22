//! CUDA acceleration bindings for pixelmap
//! 
//! This module provides CUDA-accelerated implementations of various
//! computationally intensive operations in the pixelmap library.

mod correspondence;
mod feature_matcher;
mod photo_processor;
mod memory_management;
mod common;
mod transform;
mod dense_map;

// Re-export the public API
pub use correspondence::CudaCorrespondenceMapping;
pub use correspondence::CudaCorrespondenceMappingResult;
pub use feature_matcher::CudaCircularFeatureMatcher;
pub use feature_matcher::FeaturePair;
pub use photo_processor::CudaPhotoProcessor;
pub use photo_processor::ImageStatistics;
pub use transform::CudaTransformProcessor;
pub use dense_map::CudaDenseMapProcessor;
pub use dense_map::FilterType;

// Re-export initialization functions
pub use common::{init_cuda, is_cuda_available, get_cuda_device_info, CudaDeviceInfo};

// Re-export memory management utilities
pub use memory_management::{MemoryPool, get_memory_pool};

// Import CudaStream type from cuda::memory to make it available
use crate::cuda::memory::CudaStream;

// MEMORY_POOL is a global memory pool used for efficient memory management in CUDA operations.
// Buffers are recycled to avoid frequent allocations and deallocations, improving performance.
// The global state is thread-safe, protected by Mutex and Once.

/// Custom error type for CUDA operations
#[derive(Debug)]
pub enum CudaError {
    InitializationError(String),
    MemoryAllocationError(String),
    DeviceError(String),
    Other(String),
}
