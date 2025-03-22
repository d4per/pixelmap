//! CUDA kernel launching functionality
//!
//! This module provides Rust wrappers for launching CUDA kernels.

mod core;
mod feature_extraction;
mod photo_processing;
mod dense_map;
mod correspondence;
mod transform;

// Re-export main KernelCollection struct
pub use self::core::KernelCollection;
pub use self::core::launch_kernel;

// Re-export error types
pub use crate::cuda::memory::CudaError;
