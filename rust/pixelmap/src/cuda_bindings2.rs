use crate::correspondence_mapping_algorithm::CorrespondenceMappingAlgorithm;
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;
use crate::circular_feature_grid::FeaturePair;
use crate::affine_transform::AffineTransform;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaError, CudaStream};
use crate::cuda::feature_grid::CudaCircularFeatureGrid;
use crate::cuda::kernels::KernelCollection;
use std::sync::{Arc, Mutex, Once};
use std::cell::RefCell;
use std::collections::HashMap;

// Global CUDA state management
thread_local! {
    static CUDA_CONTEXT: RefCell<Option<CudaContext>> = RefCell::new(None);
    static KERNEL_COLLECTION: RefCell<Option<KernelCollection>> = RefCell::new(None);
}

static CUDA_INIT: Once = Once::new();
static INIT_RESULT: Mutex<Result<(), String>> = Mutex::new(Ok(()));

// Memory pool for reusing allocations
struct MemoryPool {
    buffers: HashMap<usize, Vec<CudaBuffer<u8>>>,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }
    
    fn get_buffer(&mut self, size: usize) -> Result<CudaBuffer<u8>, CudaError> {
        if let Some(buffers) = self.buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer);
            }
        }
        
        // No existing buffer, allocate new one
        CudaBuffer::new(size)
    }
    
    fn return_buffer(&mut self, buffer: CudaBuffer<u8>) {
        let size = buffer.capacity();
        self.buffers.entry(size).or_insert_with(Vec::new).push(buffer);
    }
}

lazy_static::lazy_static! {
    static ref MEMORY_POOL: Mutex<MemoryPool> = Mutex::new(MemoryPool::new());
}

// Result type for CUDA operations
pub struct CudaCorrespondenceMappingResult {
    pub manager: CorrespondenceMappingAlgorithm,
    pub total_comparisons: usize,
}

/// Initialize CUDA for the current thread
fn ensure_cuda_initialized() -> Result<(), String> {
    CUDA_INIT.call_once(|| {
        let result = match crate::cuda::init() {
            Ok(_) => {
                match CudaContext::new() {
                    Ok(context) => {
                        CUDA_CONTEXT.with(|ctx| {
                            *ctx.borrow_mut() = Some(context.clone());
                        });
                        
                        match KernelCollection::new(&context) {
                            Ok(kernels) => {
                                KERNEL_COLLECTION.with(|k| {
                                    *k.borrow_mut() = Some(kernels);
                                });
                                Ok(())
                            },
                            Err(e) => Err(format!("Failed to initialize CUDA kernels: {:?}", e))
                        }
                    },
                    Err(e) => Err(format!("Failed to create CUDA context: {:?}", e))
                }
            },
            Err(e) => Err(e)
        };
        
        *INIT_RESULT.lock().unwrap() = result;
    });
    
    INIT_RESULT.lock().unwrap().clone()
}

/// Get the current CUDA context or initialize it
fn get_cuda_context() -> Result<CudaContext, String> {
    ensure_cuda_initialized()?;
    
    CUDA_CONTEXT.with(|ctx| {
        ctx.borrow().clone().ok_or_else(|| "CUDA context not initialized".to_string())
    })
}

/// Get the kernel collection or initialize it
fn get_kernel_collection() -> Result<KernelCollection, String> {
    ensure_cuda_initialized()?;
    
    KERNEL_COLLECTION.with(|kernels| {
        kernels.borrow().clone().ok_or_else(|| "CUDA kernels not initialized".to_string())
    })
}

/// Interface for CUDA-accelerated correspondence mapping
pub struct CudaCorrespondenceMapping {
    // FFI context or handle for the CUDA module
    context: CudaContext,
    stream: CudaStream,
}

impl CudaCorrespondenceMapping {
    /// Create a new CUDA correspondence mapping handler
    pub fn new() -> Result<Self, String> {
        let context = get_cuda_context()?;
        let stream = context.create_stream()
            .map_err(|e| format!("Failed to create CUDA stream: {:?}", e))?;
        
        Ok(Self {
            context,
            stream,
        })
    }
    
    /// Run correspondence mapping with CUDA acceleration
    pub fn run_correspondence_mapping(
        &self,
        manager: &mut CorrespondenceMappingAlgorithm,
        source_photo: &Photo,
        target_photo: &Photo,
    ) -> Result<CudaCorrespondenceMappingResult, String> {
        let kernels = get_kernel_collection()?;
        
        // Extract transforms and points from manager
        let transforms = manager.get_transforms();
        let points = manager.get_sample_points();
        
        // Create pinned memory for faster host-device transfers
        let mut pinned_transforms = self.create_pinned_memory::<AffineTransform>(transforms.len())?;
        pinned_transforms.copy_from_slice(&transforms);
        
        let mut pinned_points = self.create_pinned_memory::<(f32, f32)>(points.len())?;
        pinned_points.copy_from_slice(&points);
        
        // Upload photos with pitch alignment for coalesced memory access
        let d_source_photo = self.upload_aligned_photo(source_photo)?;
        let d_target_photo = self.upload_aligned_photo(target_photo)?;
        
        // Upload transforms and points from pinned memory
        let d_transforms = self.upload_from_pinned::<AffineTransform>(&pinned_transforms)?;
        let d_points = self.upload_from_pinned::<(f32, f32)>(&pinned_points)?;
        
        // Allocate memory for scores with proper alignment
        let scores_count = transforms.len();
        let d_scores = self.allocate_aligned_buffer::<f32>(scores_count)?;
        
        // Process in batches for better cache utilization
        const BATCH_SIZE: usize = 256; // Adjust based on your GPU
        let num_batches = (transforms.len() + BATCH_SIZE - 1) / BATCH_SIZE;
        let block_size = 256;
        
        // Create a stream for asynchronous processing
        let compute_stream = self.context.create_stream()
            .map_err(|e| format!("Failed to create compute stream: {:?}", e))?;
            
        // Process each batch
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * BATCH_SIZE;
            let end_idx = std::cmp::min(start_idx + BATCH_SIZE, transforms.len());
            let batch_size = end_idx - start_idx;
            
            if batch_size == 0 {
                continue;
            }
            
            // Configure kernel parameters for this batch
            let grid_dim = (batch_size + block_size - 1) / block_size;
            
            unsafe {
                // Launch score computation kernel for this batch
                kernels.launch_compute_correspondence_scores_batch(
                    d_source_photo.as_device_ptr(),
                    d_target_photo.as_device_ptr(),
                    source_photo.width as i32,
                    source_photo.height as i32,
                    target_photo.width as i32,
                    target_photo.height as i32,
                    d_transforms.as_device_ptr().add(start_idx),
                    d_points.as_device_ptr(),
                    batch_size as i32,
                    d_scores.as_device_ptr_mut().add(start_idx),
                    11, // Patch size
                    compute_stream, // Use separate stream for compute
                ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
            }
        }
        
        // Synchronize compute stream to ensure all batches are processed
        compute_stream.synchronize()
            .map_err(|e| format!("Failed to synchronize compute stream: {:?}", e))?;
        
        // Download scores to pinned memory for faster transfer
        let mut pinned_scores = self.create_pinned_memory::<f32>(scores_count)?;
        d_scores.copy_to_host(&mut pinned_scores)
            .map_err(|e| format!("Failed to download scores: {:?}", e))?;
        
        // Update manager with scores
        manager.update_scores(&pinned_scores);
        
        // Update transforms in the manager
        manager.update_transforms_from_scores();
        
        Ok(CudaCorrespondenceMappingResult {
            manager: manager.clone(),
            total_comparisons: manager.get_total_comparisons(),
        })
    }
    
    /// Remove outliers using CUDA acceleration with optimized memory access
    pub fn remove_outliers(
        &self,
        map: &mut DensePhotoMap, 
        other_map: &DensePhotoMap, 
        max_dist: f32
    ) -> Result<(), String> {
        let kernels = get_kernel_collection()?;
        let width = map.width;
        let height = map.height;
        
        // Use structure-of-arrays layout for better memory coalescing
        let map_data = map.get_packed_data();
        let other_map_data = other_map.get_packed_data();
        
        // Create pinned memory buffers for faster transfers
        let mut pinned_map_data = self.create_pinned_memory::<f32>(width * height * 3)?; // x, y, used
        pinned_map_data.copy_from_slice(&map_data);
        
        let mut pinned_other_map_data = self.create_pinned_memory::<f32>(width * height * 3)?;
        pinned_other_map_data.copy_from_slice(&other_map_data);
        
        // Upload map data using page-locked memory for faster transfer
        let mut d_map_data = self.upload_from_pinned::<f32>(&pinned_map_data)?;
        let d_other_map_data = self.upload_from_pinned::<f32>(&pinned_other_map_data)?;
        
        // Configure kernel for optimal execution
        let block_dim = 16; // 16x16 threads per block = 256 threads
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        
        // Use shared memory to cache map data for better locality
        let shared_mem_size = 3 * block_dim * block_dim * std::mem::size_of::<f32>();
        
        unsafe {
            // Launch optimized kernel for outlier removal
            kernels.launch_remove_outliers_optimized(
                d_map_data.as_device_ptr_mut(),
                d_other_map_data.as_device_ptr(),
                width as i32,
                height as i32,
                max_dist,
                self.stream,
                shared_mem_size as u32,
            ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
        }
        
        // Download results to pinned memory
        d_map_data.copy_to_host(&mut pinned_map_data)
            .map_err(|e| format!("Failed to download map data: {:?}", e))?;
        
        // Update the map with the new data
        map.update_from_packed_data(&pinned_map_data);
        
        Ok(())
    }
    
    /// Smooth grid points using CUDA acceleration with optimized memory access
    pub fn smooth_grid_points(
        &self, 
        map: &DensePhotoMap, 
        iterations: usize
    ) -> Result<DensePhotoMap, String> {
        if iterations == 0 {
            return Ok(map.clone());
        }
        
        let kernels = get_kernel_collection()?;
        let width = map.width;
        let height = map.height;
        
        // Get packed map data for efficient transfers (x, y, used flags interleaved)
        let map_data = map.get_packed_data();
        
        // Create pinned memory for faster transfers
        let mut pinned_input = self.create_pinned_memory::<f32>(map_data.len())?;
        pinned_input.copy_from_slice(&map_data);
        
        // Upload input data to GPU
        let d_input = self.upload_from_pinned::<f32>(&pinned_input)?;
        
        // Allocate buffers for ping-pong computation
        let mut d_output = CudaBuffer::new::<f32>(map_data.len())
            .map_err(|e| format!("Failed to allocate output buffer: {:?}", e))?;
        
        // Configure kernel launch parameters
        let block_dim = 16; // 16x16 threads per block
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        let shared_mem_size = 3 * (block_dim + 2) * (block_dim + 2) * std::mem::size_of::<f32>();
        
        let mut input_ptr = d_input.as_device_ptr();
        let mut output_ptr = d_output.as_device_ptr_mut();
        
        // Process iterations with ping-pong buffers
        for i in 0..iterations {
            unsafe {
                kernels.launch_smooth_grid_points_optimized(
                    input_ptr,
                    output_ptr,
                    width as i32,
                    height as i32,
                    self.stream,
                    shared_mem_size as u32,
                ).map_err(|e| format!("Failed to launch smooth kernel: {:?}", e))?;
            }
            
            // Swap buffers for next iteration
            std::mem::swap(&mut input_ptr, &mut output_ptr);
        }
        
        // Ensure the final result is in d_input if iterations is odd
        let final_buffer = if iterations % 2 == 0 { &d_input } else { &d_output };
        
        // Download result to pinned memory
        let mut pinned_output = self.create_pinned_memory::<f32>(map_data.len())?;
        final_buffer.copy_to_host(&mut pinned_output)
            .map_err(|e| format!("Failed to download smoothed data: {:?}", e))?;
        
        // Create new map from packed data
        let result_map = DensePhotoMap::from_packed_data(width, height, &pinned_output);
        
        Ok(result_map)
    }
    
    // Helper methods for optimized memory management
    
    /// Create pinned memory for faster host-device transfers
    fn create_pinned_memory<T: Copy>(&self, count: usize) -> Result<Vec<T>, String> {
        // In a real implementation, this would use CUDA page-locked memory
        // For now, just return a normal Vec
        Ok(vec![unsafe { std::mem::zeroed() }; count])
    }
    
    /// Upload data from pinned memory to device with optimal alignment
    fn upload_from_pinned<T: Copy>(&self, data: &[T]) -> Result<CudaBuffer<T>, String> {
        let mut buffer = CudaBuffer::new::<T>(data.len())
            .map_err(|e| format!("Failed to allocate device memory: {:?}", e))?;
            
        buffer.copy_from_host(data)
            .map_err(|e| format!("Failed to upload data: {:?}", e))?;
            
        Ok(buffer)
    }
    
    /// Upload photo with pitch alignment for coalesced memory access
    fn upload_aligned_photo(&self, photo: &Photo) -> Result<CudaBuffer<u8>, String> {
        // In a real implementation, this would use cuMemAllocPitch for aligned memory
        // For now, just use regular buffer
        self.upload_photo(photo)
    }
    
    /// Allocate buffer with proper alignment for coalesced memory access
    fn allocate_aligned_buffer<T>(&self, count: usize) -> Result<CudaBuffer<T>, String> {
        CudaBuffer::new::<T>(count)
            .map_err(|e| format!("Failed to allocate aligned memory: {:?}", e))
    }
    
    /// Run correspondence mapping with CUDA acceleration
    pub fn run_correspondence_mapping(
        &self,
        manager: &mut CorrespondenceMappingAlgorithm,
        source_photo: &Photo,
        target_photo: &Photo,
    ) -> Result<CudaCorrespondenceMappingResult, String> {
        let kernels = get_kernel_collection()?;
        
        // Get transforms and points from the manager
        let transforms = manager.get_transforms();
        let points = manager.get_sample_points();
        
        // Upload photos and transforms to GPU
        let d_source_photo = self.upload_photo(source_photo)?;
        let d_target_photo = self.upload_photo(target_photo)?;
        let d_transforms = self.allocate_and_upload_transforms(&transforms)?;
        let d_points = self.allocate_and_upload_points(&points)?;
        
        // Allocate memory for scores
        let mut d_scores = CudaBuffer::new::<f32>(transforms.len())
            .map_err(|e| format!("Failed to allocate memory for scores: {:?}", e))?;
        
        // Configure kernel parameters
        let patch_size = 11; // Should match CPU implementation
        
        unsafe {
            // Launch the CUDA kernel for correspondence scoring
            kernels.launch_compute_correspondence_scores(
                d_source_photo.as_device_ptr(),
                d_target_photo.as_device_ptr(),
                source_photo.width as i32,
                source_photo.height as i32,
                target_photo.width as i32,
                target_photo.height as i32,
                d_transforms.as_device_ptr(),
                d_points.as_device_ptr(),
                points.len() as i32,
                d_scores.as_device_ptr_mut(),
                patch_size,
                self.stream,
            ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
        }
        
        // Download scores from GPU
        let mut scores = vec![0.0f32; transforms.len()];
        d_scores.copy_to_host(&mut scores)
            .map_err(|e| format!("Failed to download scores: {:?}", e))?;
        
        // Update the manager with the scores
        manager.update_scores(&scores);
        
        // Update transforms in the manager
        manager.update_transforms_from_scores();
        
        Ok(CudaCorrespondenceMappingResult {
            manager: manager.clone(),
            total_comparisons: manager.get_total_comparisons(),
        })
    }
    
    /// Remove outliers using CUDA acceleration
    pub fn remove_outliers(
        &self,
        map: &mut DensePhotoMap, 
        other_map: &DensePhotoMap, 
        max_dist: f32
    ) -> Result<(), String> {
        let kernels = get_kernel_collection()?;
        let width = map.width;
        let height = map.height;
        
        // Extract the map data
        let map_x = map.get_x_values();
        let map_y = map.get_y_values();
        let map_used = map.get_usage_flags();
        
        let other_map_x = other_map.get_x_values();
        let other_map_y = other_map.get_y_values();
        let other_map_used = other_map.get_usage_flags();
        
        // Upload map data to GPU
        let mut d_map_x = CudaBuffer::new::<f32>(width * height)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;
        let mut d_map_y = CudaBuffer::new::<f32>(width * height)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;
        let mut d_map_used = CudaBuffer::new::<bool>(width * height)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;
        
        d_map_x.copy_from_host(&map_x)
            .map_err(|e| format!("Failed to upload map data: {:?}", e))?;
        d_map_y.copy_from_host(&map_y)
            .map_err(|e| format!("Failed to upload map data: {:?}", e))?;
        d_map_used.copy_from_host(&map_used)
            .map_err(|e| format!("Failed to upload map data: {:?}", e))?;
        
        // Upload other map data to GPU
        let mut d_other_map_x = CudaBuffer::new::<f32>(width * height)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;
        let mut d_other_map_y = CudaBuffer::new::<f32>(width * height)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;
        let mut d_other_map_used = CudaBuffer::new::<bool>(width * height)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;
        
        d_other_map_x.copy_from_host(&other_map_x)
            .map_err(|e| format!("Failed to upload map data: {:?}", e))?;
        d_other_map_y.copy_from_host(&other_map_y)
            .map_err(|e| format!("Failed to upload map data: {:?}", e))?;
        d_other_map_used.copy_from_host(&other_map_used)
            .map_err(|e| format!("Failed to upload map data: {:?}", e))?;
        
        // Configure kernel parameters
        let block_dim = 16;
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        
        unsafe {
            // Launch the kernel for outlier removal
            kernels.launch_remove_outliers(
                d_map_x.as_device_ptr_mut(),
                d_map_y.as_device_ptr_mut(),
                d_map_used.as_device_ptr_mut(),
                d_other_map_x.as_device_ptr(),
                d_other_map_y.as_device_ptr(),
                d_other_map_used.as_device_ptr(),
                width as i32,
                height as i32,
                max_dist,
                self.stream,
            ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
        }
        
        // Download the updated usage flags
        let mut new_map_used = vec![false; width * height];
        d_map_used.copy_to_host(&mut new_map_used)
            .map_err(|e| format!("Failed to download data: {:?}", e))?;
        
        // Update the map with the new usage flags
        map.update_usage_flags(&new_map_used);
        
        Ok(())
    }
    
    /// Smooth grid points using CUDA acceleration
    pub fn smooth_grid_points(
        &self, 
        map: &DensePhotoMap, 
        iterations: usize
    ) -> Result<DensePhotoMap, String> {
        // Get kernel collection
        let kernels = get_kernel_collection()?;
        
        // Get map dimensions
        let (width, height) = (map.width, map.height);
        
        // Upload map data to GPU
        let mut d_map_x = CudaBuffer::new::<f32>(width * height)
            .map_err(|e| format!("Failed to allocate memory for map x: {:?}", e))?;
        let mut d_map_y = CudaBuffer::new::<f32>(width * height)
            .map_err(|e| format!("Failed to allocate memory for map y: {:?}", e))?;
        let mut d_map_used = CudaBuffer::new::<bool>(width * height)
            .map_err(|e| format!("Failed to allocate memory for map used: {:?}", e))?;
            
        // Allocate memory for temporary results during smoothing
        let mut d_temp_x = CudaBuffer::new::<f32>(width * height)
            .map_err(|e| format!("Failed to allocate memory for temp x: {:?}", e))?;
        let mut d_temp_y = CudaBuffer::new::<f32>(width * height)
            .map_err(|e| format!("Failed to allocate memory for temp y: {:?}", e))?;
        
        // For now, still using CPU implementation
        let result = map.smooth_grid_points_n_times(iterations);
        
        // TODO: Implement CUDA kernel for smoothing
        // Would be much faster than CPU implementation for large maps and many iterations
        
        Ok(result)
    }

    /// Calculate matched area using CUDA
    pub fn calculate_matched_area(
        &self,
        map1: &DensePhotoMap,
        map2: &DensePhotoMap,
        threshold: f32
    ) -> Result<f32, String> {
        // Clone maps to avoid modifying originals
        let mut map1_clone = map1.clone();
        
        // Use CUDA for outlier removal
        self.remove_outliers(&mut map1_clone, map2, threshold)?;
        
        // Get kernel collection
        let kernels = get_kernel_collection()?;
        
        // Upload map data to GPU
        let (width, height) = (map1_clone.width, map1_clone.height);
        let mut d_map_used = CudaBuffer::new::<bool>(width * height)
            .map_err(|e| format!("Failed to allocate memory for map used: {:?}", e))?;
            
        // TODO: Implement CUDA kernel for counting used cells
        // For now, use CPU implementation
        Ok(map1_clone.calculate_used_area())
    }
    
    /// Batch process multiple transformations in parallel using CUDA
    pub fn batch_transform(
        &self,
        transforms: &[AffineTransform],
        source_points: &[(f32, f32)],
    ) -> Result<Vec<(f32, f32)>, String> {
        // Get kernel collection
        let kernels = get_kernel_collection()?;
        
        // Allocate memory for transforms and points
        let d_transforms = self.allocate_and_upload_transforms(transforms)?;
        let d_points = self.allocate_and_upload_points(source_points)?;
        
        // Allocate memory for results
        let result_count = transforms.len() * source_points.len();
        let mut d_results = CudaBuffer::new::<(f32, f32)>(result_count)
            .map_err(|e| format!("Failed to allocate memory for results: {:?}", e))?;
            
        // Configure kernel launch parameters
        let block_size = 256;
        let num_blocks = (result_count + block_size - 1) / block_size;
        
        // TODO: Launch CUDA kernel for batch transforms
        // For now, using sequential CPU implementation
        
        let mut results = Vec::with_capacity(result_count);
        
        for transform in transforms {
            for &(x, y) in source_points {
                let (tx, ty) = transform.transform(x, y);
                results.push((tx, ty));
            }
        }
        
        Ok(results)
    }
    
    /// Compute correspondence scores in parallel using CUDA
    pub fn compute_correspondence_scores(
        &self,
        photo1: &Photo,
        photo2: &Photo,
        transforms: &[AffineTransform],
        points: &[(f32, f32)],
        patch_size: usize
    ) -> Result<Vec<f32>, String> {
        // Get kernel collection
        let kernels = get_kernel_collection()?;
        
        // Upload photos to GPU
        let d_photo1 = self.upload_photo(photo1)?;
        let d_photo2 = self.upload_photo(photo2)?;
        
        // Upload transforms and points
        let d_transforms = self.allocate_and_upload_transforms(transforms)?;
        let d_points = self.allocate_and_upload_points(points)?;
        
        // Allocate memory for scores
        let mut d_scores = CudaBuffer::new::<f32>(transforms.len())
            .map_err(|e| format!("Failed to allocate memory for scores: {:?}", e))?;
            
        // Configure kernel launch parameters
        let block_size = 256;
        let num_blocks = (transforms.len() + block_size - 1) / block_size;
        
        // TODO: Launch CUDA kernel for computing scores
        // For now, return placeholder scores
        Ok(vec![1.0; transforms.len()])
    }

    /// Initialize correspondence mapping algorithm with CUDA acceleration
    pub fn create_correspondence_mapping_algorithm(
        &self,
        photo_width: usize,
        photo1: &Photo,
        photo2: &Photo,
        grid_cell_size: usize,
        neighborhood_radius: usize
    ) -> Result<CorrespondenceMappingAlgorithm, String> {
        // Currently just creates a CPU-based algorithm
        // In the future, could pre-allocate GPU memory and prepare for CUDA acceleration
        let algorithm = CorrespondenceMappingAlgorithm::new(
            photo_width,
            photo1,
            photo2,
            grid_cell_size,
            neighborhood_radius
        );
        
        Ok(algorithm)
    }
    
    // Helper methods for memory management
    
    fn upload_photo(&self, photo: &Photo) -> Result<CudaBuffer<u8>, String> {
        let size = photo.width * photo.height * 4;
        let mut buffer = CudaBuffer::new::<u8>(size)
            .map_err(|e| format!("Failed to allocate memory for photo: {:?}", e))?;
            
        buffer.copy_from_host(&photo.buffer)
            .map_err(|e| format!("Failed to upload photo: {:?}", e))?;
            
        Ok(buffer)
    }
    
    fn allocate_and_upload_transforms(&self, transforms: &[AffineTransform]) -> Result<CudaBuffer<AffineTransform>, String> {
        let mut buffer = CudaBuffer::new::<AffineTransform>(transforms.len())
            .map_err(|e| format!("Failed to allocate memory for transforms: {:?}", e))?;
            
        buffer.copy_from_host(transforms)
            .map_err(|e| format!("Failed to upload transforms: {:?}", e))?;
            
        Ok(buffer)
    }
    
    fn allocate_and_upload_points(&self, points: &[(f32, f32)]) -> Result<CudaBuffer<(f32, f32)>, String> {
        let mut buffer = CudaBuffer::new::<(f32, f32)>(points.len())
            .map_err(|e| format!("Failed to allocate memory for points: {:?}", e))?;
            
        buffer.copy_from_host(points)
            .map_err(|e| format!("Failed to upload points: {:?}", e))?;
            
        Ok(buffer)
    }
}

impl Drop for CudaCorrespondenceMapping {
    fn drop(&mut self) {
        // Clean up CUDA resources when the struct is dropped
        // This would free any GPU memory or contexts
    }
}

/// Interface for CUDA-accelerated feature matching
pub struct CudaCircularFeatureMatcher {
    // FFI context or handle for the CUDA module
    context: *mut std::ffi::c_void,
}

impl CudaCircularFeatureMatcher {
    /// Create a new CUDA feature matcher
    pub fn new() -> Result<Self, String> {
        // Initialize CUDA context through FFI
        Ok(Self {
            context: std::ptr::null_mut(),
        })
    }
    
    /// Match features between two photos using CUDA
    pub fn match_features(
        &self,
        photo1: &Photo,
        photo2: &Photo,
        feature_radius: usize
    ) -> Result<Vec<FeaturePair>, String> {
        // This would extract features and match them using CUDA
        // For now, we'll use a placeholder that would be replaced with actual CUDA FFI calls
        
        // This code simulates what would happen in CUDA, but it's using the CPU implementation
        use crate::circular_feature_grid;
        
        let image1 = circular_feature_grid::CircularFeatureGrid::new(
            photo1,
            photo1.width,
            photo1.height,
            feature_radius
        );
        let image2 = circular_feature_grid::CircularFeatureGrid::new(
            photo2,
            photo2.width,
            photo2.height,
            feature_radius
        );

        use crate::circular_feature_descriptor_matcher::CircularFeatureDescriptorMatcher;
        let matcher = CircularFeatureDescriptorMatcher::new();
        let pairs = matcher.match_areas(&image1, &image2);
        
        Ok(pairs)
    }
}

impl Drop for CudaCircularFeatureMatcher {
    fn drop(&mut self) {
        // Clean up CUDA resources when the struct is dropped
        // This would free any GPU memory or contexts
    }
}

/// Additional CUDA-accelerated photo operations
pub struct CudaPhotoProcessor {
    context: *mut std::ffi::c_void,
}

impl CudaPhotoProcessor {
    /// Create a new CUDA photo processor
    pub fn new() -> Result<Self, String> {
        // Initialize CUDA context
        Ok(Self {
            context: std::ptr::null_mut(),
        })
    }
    
    /// Scale photo using CUDA
    pub fn scale_photo(&self, photo: &Photo, new_width: usize) -> Result<Photo, String> {
        // Would use CUDA for efficient image scaling
        // For now, use CPU implementation
        Ok(photo.get_scaled_proportional(new_width))
    }
    
    /// Apply image filters using CUDA (e.g., blur, sharpen)
    pub fn apply_filter(&self, photo: &Photo, filter_type: &str) -> Result<Photo, String> {
        // Would apply various filters using CUDA
        match filter_type {
            "blur" => Ok(photo.clone()), // Placeholder
            "sharpen" => Ok(photo.clone()), // Placeholder
            _ => Err(format!("Unknown filter type: {}", filter_type))
        }
    }
    
    /// Compute image statistics in parallel using CUDA
    pub fn compute_statistics(&self, photo: &Photo) -> Result<ImageStatistics, String> {
        // Would compute histogram, mean, variance, etc. in parallel on GPU
        Ok(ImageStatistics {
            mean_r: 0.0,
            mean_g: 0.0,
            mean_b: 0.0,
            variance: 0.0,
        })
    }
}

impl Drop for CudaPhotoProcessor {
    fn drop(&mut self) {
        // Clean up CUDA resources
    }
}

/// Statistics about an image
pub struct ImageStatistics {
    pub mean_r: f32,
    pub mean_g: f32,
    pub mean_b: f32,
    pub variance: f32,
}

// Re-export initialization functions
pub use common::{init_cuda, is_cuda_available, get_cuda_device_info, CudaDeviceInfo};

// Re-export memory management utilities
pub use memory_management::{MemoryPool, MEMORY_POOL};

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
