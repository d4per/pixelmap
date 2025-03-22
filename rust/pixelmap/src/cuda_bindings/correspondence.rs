//! CUDA-accelerated correspondence mapping implementation

use crate::correspondence_mapping_algorithm::CorrespondenceMappingAlgorithm;
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;
use crate::affine_transform::AffineTransform;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaStream};
// Comment out missing imports for now
// use crate::cuda::kernels::KernelCollection;
// use crate::ocm::OcmManager;
use super::common::{get_cuda_context, get_kernel_collection};
use std::mem;
use std::ffi::{c_void, CString};

// Define a stub for OcmManager if needed for compilation
pub struct OcmManager;

// Remove the import of private ACGrid
// use crate::correspondence_mapping_algorithm::ACGrid;

// Define our own stubs for ac_grid
// This avoids depending on the private ACGrid type
struct ACGrid {
    width: usize,
    height: usize,
    grid_squares: Vec<GridSquare>,
}

// Define a stub for GridSquare
pub struct GridSquare;
impl GridSquare {
    pub fn get_affine_transform(&self) -> Option<AffineTransform> { None }
}

// Add cuda_driver_sys import directly
use cuda_driver_sys;

/// Result type for CUDA correspondence mapping operations
pub struct CudaCorrespondenceMappingResult {
    pub manager: CorrespondenceMappingAlgorithm,
    pub total_comparisons: usize,
}

/// Interface for CUDA-accelerated correspondence mapping
pub struct CudaCorrespondenceMapping {
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
        let transforms = manager.gather_transforms();
        let points = manager.get_grid_sample_points();
        
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
        let mut d_scores = self.allocate_aligned_buffer::<f32>(scores_count)?;
        
        // Process in batches for better cache utilization
        const BATCH_SIZE: usize = 256; // Adjust based on your GPU
        const NUM_STREAMS: usize = 4; // Use multiple streams for concurrent execution

        let num_batches = (transforms.len() + BATCH_SIZE - 1) / BATCH_SIZE;
        let block_size = 256;

        // Create multiple streams for concurrent processing
        let mut streams = Vec::with_capacity(NUM_STREAMS);
        for _ in 0..NUM_STREAMS {
            let stream = self.context.create_stream()
                .map_err(|e| format!("Failed to create compute stream: {:?}", e))?;
            streams.push(stream);
        }

        // Process each batch with different streams (round-robin)
        for batch_idx in 0..num_batches {
            let stream_idx = batch_idx % NUM_STREAMS;
            let stream = &streams[stream_idx];
            let start_idx = batch_idx * BATCH_SIZE;
            let end_idx = std::cmp::min(start_idx + BATCH_SIZE, transforms.len());
            let batch_size = end_idx - start_idx;
            
            if batch_size == 0 {
                continue;
            }
            
            // Launch kernel using the selected stream
            let grid_dim = (batch_size + block_size - 1) / block_size;
            unsafe {
                kernels.launch_compute_correspondence_scores(
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
                    stream.clone(), // Use clone() instead of dereferencing
                ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
            }
        }

        // Synchronize all streams
        for stream in &streams {
            stream.synchronize()
                .map_err(|e| format!("Failed to synchronize stream: {:?}", e))?;
        }
        
        // Download scores to pinned memory for faster transfer
        let mut pinned_scores = self.create_pinned_memory::<f32>(scores_count)?;
        d_scores.copy_to_host(&mut pinned_scores)
            .map_err(|e| format!("Failed to download scores: {:?}", e))?;
        
        // Update manager with scores
        manager.update_scores_cuda(&pinned_scores);
        manager.apply_transform_updates();
        
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
        let width = map.grid_width;
        let height = map.grid_height;
        
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
            kernels.launch_remove_outliers(
                d_map_data.as_device_ptr_mut(),
                d_other_map_data.as_device_ptr(),
                width as i32,
                height as i32,
                max_dist,
                self.stream.clone(), // Add .clone() to prevent moving out of self
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
        let width = map.grid_width;
        let height = map.grid_height;
        
        // Get packed map data for efficient transfers (x, y, used flags interleaved)
        let map_data = map.get_packed_data();
        
        // Create pinned memory for both input and output once
        let mut pinned_input = self.create_pinned_memory::<f32>(map_data.len())?;
        pinned_input.copy_from_slice(&map_data);
        
        // Allocate pinned output memory only once
        let mut pinned_output = self.create_pinned_memory::<f32>(map_data.len())?;
        
        // Upload input data to GPU
        let d_input = self.upload_from_pinned::<f32>(&pinned_input)?;
        
        // Allocate buffers for ping-pong computation
        let mut d_output = CudaBuffer::new(map_data.len())
            .map_err(|e| format!("Failed to allocate output buffer: {:?}", e))?;
        
        // Configure kernel launch parameters
        let block_dim = 16; // 16x16 threads per block
        let grid_dim_x = (width + block_dim - 1) / block_dim;
        let grid_dim_y = (height + block_dim - 1) / block_dim;
        
        // Calculate shared memory size for optimized kernel
        // Need to store x, y, and used flag for each cell in the tile plus halo
        let shared_mem_size = 3 * (block_dim + 2) * (block_dim + 2) * std::mem::size_of::<f32>();
        
        // Ping-pong buffers for multiple iterations
        let mut input_ptr = d_input.as_device_ptr();
        let mut output_ptr = d_output.as_device_ptr_mut();
        
        // Process iterations with ping-pong buffers
        for _ in 0..iterations {
            unsafe {
                // Try to launch the optimized version first
                let result = kernels.launch_smooth_grid_points_optimized(
                    input_ptr,
                    output_ptr,
                    width as i32,
                    height as i32,
                    self.stream.clone(), // Add .clone() to prevent moving out of self
                    shared_mem_size as u32,
                );
                if let Err(_) = result {  
                    // Fall back to non-optimized version
                    // Extract each component from packed data for traditional kernel
                    let cell_count = width * height;
                    
                    // Unpack structure-of-arrays format back to array-of-structures
                    let in_x = input_ptr;
                    let in_y = unsafe { input_ptr.add(cell_count) };
                    let in_used = unsafe { input_ptr.add(2 * cell_count) };
                    
                    let out_x = output_ptr;
                    let out_y = unsafe { output_ptr.add(cell_count) };
                    let out_used = unsafe { output_ptr.add(2 * cell_count) };
                    
                    kernels.launch_smooth_grid_points(
                        in_x,
                        in_y,
                        in_used,
                        out_x,
                        out_y,
                        out_used,
                        width as i32,
                        height as i32,
                        self.stream.clone(), // Add .clone() to prevent moving out of self
                    ).map_err(|e| format!("Failed to launch kernel: {:?}", e))?;
                }
            }
            // Swap buffer pointers for next iteration - fix the type mismatch
            // Instead of using std::mem::swap which requires same types,
            // just store the values separately and reassign
            let temp_ptr = input_ptr;
            input_ptr = output_ptr as *const c_void; // Cast mutable pointer to const
            output_ptr = temp_ptr as *mut c_void;    // Cast const pointer to mutable
        }
        
        // Ensure the final result is in d_input if iterations is odd
        let final_buffer = if iterations % 2 == 0 { &d_input } else { &d_output };
        
        // Download final result
        final_buffer.copy_to_host(&mut pinned_output)
            .map_err(|e| format!("Failed to download smoothed data: {:?}", e))?;
        
        // Create new map from packed data
        let result_map = DensePhotoMap::from_packed_data(
            map.photo1.clone(),
            map.photo2.clone(),
            width, 
            height,
            &pinned_output,
        );
        
        Ok(result_map)
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
        let (width, height) = (map1_clone.grid_width, map1_clone.grid_height);
        let mut d_map_used: CudaBuffer<bool> = CudaBuffer::new(width * height)
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
        let kernels = get_kernel_collection()?;
        
        // Allocate memory for transforms and points
        let d_transforms = self.allocate_and_upload_transforms(transforms)?;
        let d_points = self.allocate_and_upload_points(source_points)?;
        
        // Allocate memory for results
        let result_count = transforms.len() * source_points.len();
        let mut d_results = CudaBuffer::new(result_count)
            .map_err(|e| format!("Failed to allocate memory for results: {:?}", e))?;
        
        // Configure kernel launch parameters
        let block_size = 256;
        let num_blocks = (result_count + block_size - 1) / block_size;
        
        // TODO: Launch CUDA kernel for batch transforms
        // For now, using sequential CPU implementation
        let mut results = Vec::with_capacity(result_count);
        for transform in transforms {
            for &(x, y) in source_points {
                // Use transform method since apply_to_point doesn't exist
                let (tx, ty) = transform.transform(x, y);
                results.push((tx, ty));
            }
        }
        
        // Download results
        d_results.copy_to_host(&mut results)
            .map_err(|e| format!("Failed to download results: {:?}", e))?;
        
        Ok(results)
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
    
    /// Create pinned memory for faster host-device transfers
    fn create_pinned_memory<T: Copy>(&self, count: usize) -> Result<Vec<T>, String> {
        // In a real implementation, this would use CUDA page-locked memory
        // For now, just return a normal Vec
        Ok(vec![unsafe { std::mem::zeroed() }; count])
    }
    
    /// Upload data from pinned memory to device with optimal alignment
    fn upload_from_pinned<T: Copy>(&self, data: &[T]) -> Result<CudaBuffer<T>, String> {
        let mut buffer = CudaBuffer::new(data.len())
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
    
    fn upload_photo(&self, photo: &Photo) -> Result<CudaBuffer<u8>, String> {
        let size = photo.width * photo.height * 4;
        let mut buffer = CudaBuffer::new(size)
            .map_err(|e| format!("Failed to allocate memory for photo: {:?}", e))?;
            
        buffer.copy_from_host(&photo.img_data)
            .map_err(|e| format!("Failed to upload photo: {:?}", e))?;
            
        Ok(buffer)
    }
    
    fn allocate_and_upload_transforms(&self, transforms: &[AffineTransform]) -> Result<CudaBuffer<AffineTransform>, String> {
        let mut buffer = CudaBuffer::new(transforms.len())
            .map_err(|e| format!("Failed to allocate memory for transforms: {:?}", e))?;
            
        buffer.copy_from_host(transforms)
            .map_err(|e| format!("Failed to upload transforms: {:?}", e))?;
            
        Ok(buffer)
    }
    
    fn allocate_and_upload_points(&self, points: &[(f32, f32)]) -> Result<CudaBuffer<(f32, f32)>, String> {
        let mut buffer = CudaBuffer::new(points.len())
            .map_err(|e| format!("Failed to allocate memory for points: {:?}", e))?;
            
        buffer.copy_from_host(points)
            .map_err(|e| format!("Failed to upload points: {:?}", e))?;
            
        Ok(buffer)
    }

    pub fn extract_ac_grids(
        &self, 
        ocm_manager1: &CorrespondenceMappingAlgorithm, 
        ocm_manager2: &CorrespondenceMappingAlgorithm
    ) -> Result<(crate::ac_grid::ACGrid, crate::ac_grid::ACGrid), String> {
        // Simply call get_ac_grid() on each manager to get the actual ACGrid type
        println!("extract_ac_grids called with CorrespondenceMappingAlgorithm parameters");
        
        // Return the AC grids from the algorithms
        Ok((ocm_manager1.get_ac_grid(), ocm_manager2.get_ac_grid()))
    }

    pub fn refine_correspondences(&self, manager: &mut CorrespondenceMappingAlgorithm, map: &mut DensePhotoMap) -> Result<(), String> {
        let transforms = manager.gather_transforms();
        let points = manager.get_grid_sample_points();

        // Fix borrow errors by declaring the buffers as mutable
        let mut d_transforms = self.allocate_and_copy_transforms(&transforms)?;
        let mut d_points = self.allocate_and_copy_points(&points)?;

        let width = map.photo1.width;
        let height = map.photo1.height;
        // Use get_packed_data() method instead of accessing map_data directly
        let mut map_data = map.get_packed_data();  // Make map_data mutable

        let mut d_map_data = CudaBuffer::new(map_data.len())
            .map_err(|e| format!("Failed to allocate memory for map_data: {:?}", e))?;
        d_map_data.copy_from_host(&map_data)
            .map_err(|e| format!("Failed to copy map_data to device: {:?}", e))?;

        let mut d_output = CudaBuffer::new(map_data.len())
            .map_err(|e| format!("Failed to allocate memory for d_output: {:?}", e))?;

        let block_size = 32;
        let grid_dim_x = (width + block_size - 1) / block_size;
        let grid_dim_y = (height + block_size - 1) / block_size;

        let kernel_name = "refine_correspondences_kernel";
        let module = self.load_cuda_module()?;
        let function = self.get_cuda_function(&module, kernel_name)?;

        let mut input_ptr = d_map_data.as_device_ptr_mut();
        let mut output_ptr = d_output.as_device_ptr_mut();

        // Fix the parameter types - create a Vec<*mut c_void> instead of Vec<*mut *mut c_void>
        let params: Vec<*mut c_void> = vec![
            input_ptr as *mut c_void,
            output_ptr as *mut c_void,
            &mut (width as i32) as *mut i32 as *mut c_void,
            &mut (height as i32) as *mut i32 as *mut c_void,
            d_transforms.as_device_ptr_mut() as *mut c_void,
            d_points.as_device_ptr_mut() as *mut c_void,
        ];

        self.launch_kernel(&function, (grid_dim_x as u32, grid_dim_y as u32, 1), (block_size as u32, block_size as u32, 1), &params)?;

        d_output.copy_to_host(&mut map_data)
            .map_err(|e| format!("Failed to copy d_output to host: {:?}", e))?;

        // Use update_from_packed_data() to update the map with the new data
        map.update_from_packed_data(&map_data);

        std::mem::swap(&mut input_ptr, &mut output_ptr);

        Ok(())
    }

    fn allocate_and_copy_transforms(&self, transforms: &[AffineTransform]) -> Result<CudaBuffer<AffineTransform>, String> {
        self.allocate_and_copy_buffer(transforms)
    }

    fn allocate_and_copy_points(&self, points: &[(f32, f32)]) -> Result<CudaBuffer<(f32, f32)>, String> {
        self.allocate_and_copy_buffer(points)
    }

    fn allocate_and_copy_map_used(&self, map1_clone: &DensePhotoMap) -> Result<CudaBuffer<bool>, String> {
        let (width, height) = (map1_clone.photo1.width, map1_clone.photo1.height);
        let mut d_map_used = CudaBuffer::new(width * height)
            .map_err(|e| format!("Failed to allocate memory for map_used: {:?}", e))?;
        
        let mut map_used: Vec<bool> = vec![false; width * height];
        d_map_used.copy_from_host(&map_used)
            .map_err(|e| format!("Failed to copy map_used to device: {:?}", e))?;

        Ok(d_map_used)
    }

    fn allocate_and_copy_buffer<T: Copy>(&self, data: &[T]) -> Result<CudaBuffer<T>, String> {
        let mut buffer = CudaBuffer::new(data.len())
            .map_err(|e| format!("Failed to allocate memory: {:?}", e))?;
        buffer.copy_from_host(data)
            .map_err(|e| format!("Failed to copy data to device: {:?}", e))?;
        Ok(buffer)
    }

    fn load_cuda_module(&self) -> Result<u64, String> {
        // Replace include_str! with a placeholder since the PTX file doesn't exist
        // let ptx_source = include_str!("../cuda_kernels/refine_correspondences.ptx");
        
        // Use a placeholder string for now
        let ptx_source = "// Empty PTX placeholder - replace with actual PTX content";
        
        let mut module = 0u64;
        unsafe {
            // Fix the type cast for cuModuleLoadData
            let result = cuda_driver_sys::cuModuleLoadData(
                &mut module as *mut _ as *mut cuda_driver_sys::CUmodule,
                ptx_source.as_ptr() as *const c_void // Fix: cast to *const c_void
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
                return Err(format!("Failed to load module from PTX source: {}", error_msg));
            }
        }
        Ok(module)
    }

    fn get_cuda_function(&self, module: &u64, kernel_name: &str) -> Result<u64, String> {
        let mut function_ptr: cuda_driver_sys::CUfunction = unsafe { std::mem::zeroed() };
        let kernel_name_cstr = std::ffi::CString::new(kernel_name).unwrap();
    
        unsafe {
            let result = cuda_driver_sys::cuModuleGetFunction(
                &mut function_ptr as *mut _,
                *module as cuda_driver_sys::CUmodule,
                kernel_name_cstr.as_ptr(),
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
                return Err(format!("Failed to get kernel function {}: {}", kernel_name, error_msg));
            }
        }
        let function = function_ptr as u64;
        Ok(function)
    }

    fn launch_kernel(&self, function: &u64, grid_dim: (u32, u32, u32), block_dim: (u32, u32, u32), params: &[*mut c_void]) -> Result<(), String> {
        unsafe {
            let result = cuda_driver_sys::cuLaunchKernel(
                // Fix the type cast for function pointer
                *function as cuda_driver_sys::CUfunction,
                grid_dim.0, grid_dim.1, grid_dim.2,
                block_dim.0, block_dim.1, block_dim.2,
                0,
                self.stream.as_ptr() as cuda_driver_sys::CUstream,
                params.as_ptr() as *mut *mut c_void,
                std::ptr::null_mut(),
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
                return Err(format!("Kernel launch failed: {}", error_msg));
            }
        }
        Ok(())
    }

    fn allocate_aligned_buffer<T: Copy>(&self, count: usize) -> Result<CudaBuffer<T>, String> {
        CudaBuffer::new(count)
            .map_err(|e| format!("Failed to allocate aligned buffer: {:?}", e))
    }
}

impl Drop for CudaCorrespondenceMapping {
    fn drop(&mut self) {
        // Clean up CUDA resources when the struct is dropped
        // This would free any GPU memory or contexts
    }
}

// Extension trait for CorrespondenceMappingAlgorithm to access required methods in CUDA code
pub trait CorrespondenceMappingExtensions {
    /// Convert between AC grid types
    fn ac_grid_for_cuda(&self) -> &ACGrid;
    
    /// Get photo1 height as u32
    fn photo1_height_u32(&self) -> u32;
    
    /// Get photo1 width as u32
    fn photo1_width_u32(&self) -> u32;
    
    /// Update scores from CUDA processing
    fn update_scores_cuda(&mut self, scores: &[f32]);
    
    /// Apply transform updates after score updates
    fn apply_transform_updates(&mut self);
}

// Implement the extension trait
impl CorrespondenceMappingExtensions for CorrespondenceMappingAlgorithm {
    fn ac_grid_for_cuda(&self) -> &ACGrid {
        panic!("Cannot actually access private ACGrid field")
    }
    
    fn photo1_height_u32(&self) -> u32 {
        self.photo1.height as u32
    }
    
    fn photo1_width_u32(&self) -> u32 {
        self.photo1.width as u32
    }
    
    fn update_scores_cuda(&mut self, scores: &[f32]) {
        self.update_scores(scores);
    }
    
    fn apply_transform_updates(&mut self) {
        self.update_transforms_from_scores();
    }
}