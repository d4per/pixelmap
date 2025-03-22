//! CUDA support module - stub implementation when CUDA is not available

/// Initialize CUDA system
pub fn init() -> Result<(), String> {
    Err("CUDA support not compiled in this build".to_string())
}

/// Check if CUDA is available on this system
pub fn is_available() -> bool {
    false
}

/// Information about a CUDA device
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: String,
    pub memory: usize,
}

/// Get info about available CUDA devices
pub fn get_device_info() -> Vec<DeviceInfo> {
    Vec::new()
}

// Empty modules to satisfy imports
pub mod memory {
    pub struct CudaBuffer<T> {
        _phantom: std::marker::PhantomData<T>,
    }

    pub struct CudaContext;

    pub struct CudaStream;

    #[derive(Debug)]
    pub enum CudaError {
        NotImplemented,
        OutOfMemory,
        InvalidArgument,
        DeviceError,
    }

    impl CudaContext {
        pub fn new() -> Result<Self, CudaError> {
            Err(CudaError::NotImplemented)
        }

        pub fn create_stream(&self) -> Result<CudaStream, CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn synchronize(&self) -> Result<(), CudaError> {
            Err(CudaError::NotImplemented)
        }
    }
    
    impl<T> CudaBuffer<T> {
        pub fn new(_context: &CudaContext, _size: usize) -> Result<Self, CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn len(&self) -> usize {
            0
        }
        
        pub fn is_empty(&self) -> bool {
            true
        }
        
        pub fn upload(&mut self, _data: &[T], _stream: &CudaStream) -> Result<(), CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn download(&self, _dest: &mut [T], _stream: &CudaStream) -> Result<(), CudaError> {
            Err(CudaError::NotImplemented)
        }
    }
    
    impl CudaStream {
        pub fn synchronize(&self) -> Result<(), CudaError> {
            Err(CudaError::NotImplemented)
        }
    }
}

pub mod feature_grid {
    use crate::cuda2::memory::{CudaBuffer, CudaContext, CudaError, CudaStream};
    
    pub struct CudaCircularFeatureGrid;
    
    impl CudaCircularFeatureGrid {
        pub fn new(_context: &CudaContext, _radius: f32, _feature_dim: usize) -> Result<Self, CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn get_features(&self) -> Result<&CudaBuffer<f32>, CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn get_mutable_features(&mut self) -> Result<&mut CudaBuffer<f32>, CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn feature_dim(&self) -> usize {
            0
        }
        
        pub fn radius(&self) -> f32 {
            0.0
        }
    }
}

pub mod kernels {
    use crate::cuda2::memory::{CudaBuffer, CudaContext, CudaError, CudaStream};
    use crate::cuda2::feature_grid::CudaCircularFeatureGrid;
    
    pub struct KernelCollection;
    
    impl KernelCollection {
        pub fn new(_context: &crate::cuda2::memory::CudaContext) -> Result<Self, CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn sample_feature_grid(
            &self,
            _grid: &CudaCircularFeatureGrid,
            _points: &CudaBuffer<f32>,
            _output: &mut CudaBuffer<f32>,
            _stream: &CudaStream
        ) -> Result<(), CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn sample_feature_grid_backward(
            &self,
            _grid: &mut CudaCircularFeatureGrid,
            _points: &CudaBuffer<f32>,
            _grad_output: &CudaBuffer<f32>,
            _stream: &CudaStream
        ) -> Result<(), CudaError> {
            Err(CudaError::NotImplemented)
        }
        
        pub fn clear_buffer(
            &self,
            _buffer: &mut CudaBuffer<f32>,
            _stream: &CudaStream
        ) -> Result<(), CudaError> {
            Err(CudaError::NotImplemented)
        }
    }
}
