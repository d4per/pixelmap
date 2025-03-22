//! CUDA-accelerated implementation of CircularFeatureGrid
//! 
//! This module provides a GPU-accelerated version of the CircularFeatureGrid
//! for much faster feature descriptor computation.

use crate::circular_feature_descriptor::CircularFeatureDescriptor;
use crate::photo::Photo;
use crate::cuda::memory::{CudaBuffer, CudaContext, CudaError};
use crate::cuda::kernels::KernelCollection;
use std::ffi::c_void;
use std::convert::TryFrom;
use std::sync::Arc;
use crate::cuda::memory::CudaStream;

/// A CUDA-accelerated version of CircularFeatureGrid
pub struct CudaCircularFeatureGrid {
    /// The width (in pixels) of the grid/image.
    image_width: usize,
    /// The height (in pixels) of the grid/image.
    image_height: usize,
    /// The radius of the circular neighborhood used to compute features.
    circle_radius: usize,
    /// The maximum possible color value for a circular region.
    max_color_value: usize,
    /// A vector of circular feature descriptors, one per position in the grid.
    feature_descriptors: Vec<CircularFeatureDescriptor>,
    /// CUDA context for GPU operations
    context: Arc<CudaContext>,
    /// CUDA stream for asynchronous operations
    stream: CudaStream,
}

impl CudaCircularFeatureGrid {
    /// Creates a new CUDA-accelerated CircularFeatureGrid
    pub fn new(
        photo: &Photo, 
        width: usize, 
        height: usize, 
        circle_radius: usize,
        context: Arc<CudaContext>,
        kernels: &KernelCollection
    ) -> Result<Self, CudaError> {
        // Create CUDA stream
        let stream = context.create_stream()?;
        
        // Create grid
        let mut grid = Self {
            image_width: width,
            image_height: height,
            circle_radius,
            max_color_value: 0, // Will be calculated
            feature_descriptors: Vec::with_capacity(width * height),
            context,
            stream,
        };
        
        // Calculate max color value
        let mut pixel_count_in_circle = 0;
        for y in -(circle_radius as isize)..=(circle_radius as isize) {
            // For each row in the circle, figure out how many pixels are within the radius
            let row_width = ((circle_radius * circle_radius) as f64 - (y * y) as f64)
                .sqrt().round() as isize;
                
            for _x in -row_width..=row_width {
                pixel_count_in_circle += 1;
            }
        }
        grid.max_color_value = pixel_count_in_circle * 255;
        
        // Upload photo data to GPU
        let mut d_photo = CudaBuffer::<u8>::new(&grid.context, photo.buffer.len())?;
        d_photo.copy_from_host(&photo.buffer)?;
        
        // Allocate GPU memory for descriptors
        let descriptor_count = width * height;
        let mut d_descriptors = CudaBuffer::<CircularFeatureDescriptorRaw>::new(&grid.context, descriptor_count)?;
        
        // Set up kernel parameters
        let params = ExtractFeaturesParams {
            photo: d_photo.as_device_ptr(),
            descriptors: d_descriptors.as_device_ptr_mut(),
            width: width as u32,
            height: height as u32,
            circle_radius: circle_radius as u32,
            max_color_value: grid.max_color_value as u32,
        };
        
        // Launch feature extraction kernel
        unsafe {
            kernels.launch_extract_features(
                params,
                ((width as u32 + 15) / 16, (height as u32 + 15) / 16, 1), // Grid size
                (16, 16, 1), // Block size
                0, // Shared memory size
                grid.stream
            )?;
        }
        
        // Download descriptors from GPU
        let mut descriptors_raw = vec![CircularFeatureDescriptorRaw::default(); descriptor_count];
        d_descriptors.copy_to_host(&mut descriptors_raw)?;
        
        // Convert raw descriptors to CircularFeatureDescriptor
        grid.feature_descriptors = descriptors_raw.iter().enumerate().map(|(i, raw)| {
            let x = (i % width) as u16;
            let y = (i / width) as u16;
            
            CircularFeatureDescriptor {
                center_x: x,
                center_y: y,
                total_angle: raw.total_angle,
                r_weight: raw.r_weight,
                g_weight: raw.g_weight,
                b_weight: raw.b_weight,
                intensity: raw.intensity,
                // We're missing some fields here, those would need to be calculated or set with defaults
                ..CircularFeatureDescriptor::default()
            }
        }).collect();
        
        Ok(grid)
    }
    
    /// Get image width
    pub fn get_width(&self) -> usize {
        self.image_width
    }
    
    /// Get image height
    pub fn get_height(&self) -> usize {
        self.image_height
    }
    
    /// Get circle radius
    pub fn get_circle_radius(&self) -> usize {
        self.circle_radius
    }
    
    /// Get all feature descriptors
    pub fn get_feature_descriptors(&self) -> &[CircularFeatureDescriptor] {
        &self.feature_descriptors
    }
    
    /// Create grid with subset of descriptors matching given criteria
    pub fn create_filtered_grid<F>(&self, filter_fn: F) -> CudaCircularFeatureGrid 
    where
        F: Fn(&CircularFeatureDescriptor) -> bool,
    {
        // Filter feature descriptors
        let filtered_descriptors: Vec<CircularFeatureDescriptor> = self.feature_descriptors
            .iter()
            .filter(|desc| filter_fn(desc))
            .cloned()
            .collect();
        
        // Create a new grid with the filtered descriptors
        CudaCircularFeatureGrid {
            image_width: self.image_width,
            image_height: self.image_height,
            circle_radius: self.circle_radius,
            max_color_value: self.max_color_value,
            feature_descriptors: filtered_descriptors,
            context: self.context.clone(),
            stream: self.stream.clone(),
        }
    }
    
    /// Find descriptor at a given position or None if out of bounds
    pub fn get_descriptor_at(&self, x: usize, y: usize) -> Option<&CircularFeatureDescriptor> {
        if x >= self.image_width || y >= self.image_height {
            return None;
        }
        
        let index = y * self.image_width + x;
        self.feature_descriptors.get(index)
    }
    
    /// Returns a vector of non-zero descriptors, sorted by intensity
    pub fn get_sorted_non_zero_descriptors(&self) -> Vec<&CircularFeatureDescriptor> {
        let mut descriptors: Vec<&CircularFeatureDescriptor> = self.feature_descriptors
            .iter()
            .filter(|desc| desc.intensity > 0.0)
            .collect();
            
        descriptors.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap_or(std::cmp::Ordering::Equal));
        descriptors
    }

    /// Convert GPU grid to CPU grid for compatibility
    pub fn to_cpu_grid(&self) -> crate::circular_feature_grid::CircularFeatureGrid {
        use crate::circular_feature_grid::CircularFeatureGrid;
        
        // Create a new CPU grid with empty photo
        let empty_photo = Photo::new_empty(self.image_width, self.image_height);
        let mut cpu_grid = CircularFeatureGrid::new(
            &empty_photo,
            self.image_width,
            self.image_height,
            self.circle_radius
        );
        
        // Copy feature descriptors
        cpu_grid.set_feature_descriptors(self.feature_descriptors.clone());
        
        cpu_grid
    }
}

// Raw struct that matches the GPU-side descriptor layout
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CircularFeatureDescriptorRaw {
    total_angle: f32,
    r_weight: f32,
    g_weight: f32,
    b_weight: f32,
    intensity: f32,
}

// Parameter struct for the feature extraction kernel
#[repr(C)]
struct ExtractFeaturesParams {
    photo: *const c_void,
    descriptors: *mut c_void,
    width: u32,
    height: u32,
    circle_radius: u32,
    max_color_value: u32,
}