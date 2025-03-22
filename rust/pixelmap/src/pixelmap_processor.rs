use crate::ac_grid::ACGrid;
use crate::circular_feature_descriptor_matcher::CircularFeatureDescriptorMatcher;
use crate::circular_feature_grid;
use crate::correspondence_mapping_algorithm::CorrespondenceMappingAlgorithm;
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;

// Consolidated CUDA imports
#[cfg(feature = "cuda")]
use crate::cuda2::{self, memory::CudaContext};
#[cfg(feature = "cuda")]
use crate::cuda_bindings::{CudaCorrespondenceMapping, CudaCircularFeatureMatcher, CudaPhotoProcessor};

// Error type for CUDA operations
#[cfg(feature = "cuda")]
type CudaResult<T> = Result<T, String>;

/// Manages a pipeline for finding and refining a mapping between two images (`photo1` and `photo2`).
///
/// The process typically involves:
/// 1. **Circular feature extraction and matching** on scaled versions of the photos.
/// 2. Initializing a [CorrespondenceMappingAlgorithm] with matched points.
/// 3. Iterative refinement steps that remove outliers, smooth the mappings, and further optimize.
/// 4. Producing final [DensePhotoMap]s describing forward (`photo1` → `photo2`) and backward (`photo2` → `photo1`) transformations.
pub struct PixelMapProcessor {
    /// The first image to be matched/registered.
    photo1: Photo,

    /// The second image to be matched/registered.
    photo2: Photo,

    /// An algorithm that manages local transformations and outlier filtering
    /// from `photo1` to `photo2`.
    ocm_manager1: CorrespondenceMappingAlgorithm,

    /// An algorithm that manages local transformations and outlier filtering
    /// from `photo2` to `photo1`.
    ocm_manager2: CorrespondenceMappingAlgorithm,

    /// A running total of all comparisons made by both correspondence managers.
    total_comparisons: usize,

    /// The initial width used for scaling the photos, if needed, during setup.
    initial_photo_width: usize,
}

impl PixelMapProcessor {
    /// Constructs a new `PixelMapProcessor` with the given `photo1`, `photo2`, and an `initial_photo_width`.
    ///
    /// The width is used to scale images down (if needed) before feature extraction. This
    /// method also sets up two correspondence managers (`ocm_manager1`, `ocm_manager2`),
    /// initially with dummy images, then re-initializes them with the real images to ensure
    /// references are valid.
    ///
    /// # Parameters
    /// - `photo1`: The first [Photo].
    /// - `photo2`: The second [Photo].
    /// - `photo_width`: The width to which images may be scaled, ensuring
    ///   consistent dimensions during feature matching.
    ///
    /// # Returns
    /// A new `PixelMapProcessor` ready to be initialized.
    pub fn new(photo1: Photo, photo2: Photo, photo_width: usize) -> Self {
        // Temporary dummy Photo, used only so that CorrespondenceMappingAlgorithm can be constructed.
        let dummy_photo = Photo::default();

        let mut result = PixelMapProcessor {
            photo1,
            photo2,
            ocm_manager1: CorrespondenceMappingAlgorithm::new(photo_width, &dummy_photo, &dummy_photo, 5, 5),
            ocm_manager2: CorrespondenceMappingAlgorithm::new(photo_width, &dummy_photo, &dummy_photo, 5, 5),
            total_comparisons: 0,
            initial_photo_width: photo_width,
        };

        // Re-initialize with the actual photos.
        result.ocm_manager1 = CorrespondenceMappingAlgorithm::new(
            photo_width,
            &result.photo1,
            &result.photo2,
            5,
            5,
        );
        result.ocm_manager2 = CorrespondenceMappingAlgorithm::new(
            photo_width,
            &result.photo2,
            &result.photo1,
            5,
            5,
        );

        result
    }

    /// Constructs a new CUDA-accelerated `PixelMapProcessor`
    /// 
    /// This is a deprecated method. Use `new_with_cuda` instead.
    #[deprecated(since = "1.1.0", note = "Use new_with_cuda instead")]
    #[cfg(feature = "cuda")]
    pub fn new_cuda(photo1: Photo, photo2: Photo, photo_width: usize) -> Result<Self, String> {
        // Initialize CUDA context first to ensure GPU is available
        cuda2::init()?;
        
        // Check if CUDA is available
        if !cuda2::is_available() {
            return Err("CUDA is not available on this system".to_string());
        }
        
        // Use CPU implementation for initialization, will be replaced with CUDA operations
        let processor = Self::new(photo1, photo2, photo_width);
        
        Ok(processor)
    }

    /// Constructs a new `PixelMapProcessor` optimized for CUDA with proper initialization and error handling.
    ///
    /// This constructor:
    /// 1. Initializes CUDA and checks for GPU availability
    /// 2. Creates CUDA contexts and streams 
    /// 3. Prepares kernel collections
    /// 4. Sets up the correspondence mapping algorithms
    ///
    /// # Parameters
    /// - `photo1`: The first [Photo].
    /// - `photo2`: The second [Photo].
    /// - `photo_width`: The width to which images may be scaled, ensuring
    ///   consistent dimensions during feature matching.
    ///
    /// # Returns
    /// Either a new CUDA-optimized `PixelMapProcessor` or an error message.
    #[cfg(feature = "cuda")]
    pub fn new_with_cuda(photo1: Photo, photo2: Photo, photo_width: usize) -> CudaResult<Self> {
        // Initialize CUDA
        if let Err(e) = cuda2::init() {
            // Log the error but continue with CPU fallback
            eprintln!("CUDA initialization failed: {}. Falling back to CPU implementation.", e);
            return Ok(Self::new(photo1, photo2, photo_width));
        }
        
        if !cuda2::is_available() {
            eprintln!("CUDA is not available. Falling back to CPU implementation.");
            return Ok(Self::new(photo1, photo2, photo_width));
        }
        
        // First create the processor with CPU implementation
        let mut processor = Self::new(photo1, photo2, photo_width);
        
        // Pre-process the photos with CUDA for optimal performance
        if let Err(e) = processor.preprocess_photos_cuda() {
            eprintln!("CUDA photo preprocessing failed: {}. Using original photos.", e);
        }
        
        // Initialize CUDA manager for correspondence mapping
        let cuda_mapper = match CudaCorrespondenceMapping::new() {
            Ok(mapper) => mapper,
            Err(e) => {
                eprintln!("Failed to create CUDA mapper: {}. Falling back to CPU implementation.", e);
                return Ok(processor);
            }
        };
        
        // Try to recreate the correspondence managers with CUDA optimization
        let ocm_manager1 = match cuda_mapper.create_correspondence_mapping_algorithm(
            photo_width,
            &processor.photo1,
            &processor.photo2,
            5,
            5
        ) {
            Ok(manager) => manager,
            Err(e) => {
                eprintln!("Failed to create CUDA-optimized correspondence manager: {}", e);
                processor.ocm_manager1.clone()
            }
        };
        
        let ocm_manager2 = match cuda_mapper.create_correspondence_mapping_algorithm(
            photo_width,
            &processor.photo2,
            &processor.photo1,
            5,
            5
        ) {
            Ok(manager) => manager,
            Err(e) => {
                eprintln!("Failed to create CUDA-optimized correspondence manager: {}", e);
                processor.ocm_manager2.clone()
            }
        };
        
        // Update the processors with CUDA-optimized managers
        processor.ocm_manager1 = ocm_manager1;
        processor.ocm_manager2 = ocm_manager2;
        
        // Print CUDA device info for debugging
        let devices = cuda2::get_device_info();
        if !devices.is_empty() {
            let device = &devices[0];
            println!("Using CUDA device: {} (Compute capability: {}, Memory: {} MB)",
                device.name, device.compute_capability, device.memory / (1024 * 1024));
        }
        
        Ok(processor)
    }

    /// Performs the initial matching step:
    /// 1. Scales both `photo1` and `photo2` to `initial_photo_width` (if needed).
    /// 2. Uses [circular_feature_grid::CircularFeatureGrid] to extract circular feature descriptors.
    /// 3. Matches these descriptors with [CircularFeatureDescriptorMatcher].
    /// 4. Initializes new [CorrespondenceMappingAlgorithm] instances with the matched points.
    /// 5. Runs both correspondence managers until completion.
    ///
    /// Upon completion, `total_comparisons` is updated with the sum of both managers' comparisons.
    pub fn init(&mut self) {
        // Scale down images if needed.
        let width = usize::min(self.initial_photo_width, self.photo1.width);
        let photo1scaled = self.photo1.get_scaled_proportional(width);
        let photo2scaled = self.photo2.get_scaled_proportional(width);

        // Create circular feature grids.
        let image1 = circular_feature_grid::CircularFeatureGrid::new(
            &photo1scaled,
            photo1scaled.width,
            photo1scaled.height,
            10
        );
        let image2 = circular_feature_grid::CircularFeatureGrid::new(
            &photo2scaled,
            photo2scaled.width,
            photo2scaled.height,
            10
        );

        // Match features across the two scaled images.
        let circle_area_info_matcher = CircularFeatureDescriptorMatcher::new();
        let pairs = circle_area_info_matcher.match_areas(&image1, &image2);

        // Create new managers for the scaled images.
        let mut ocm_manager1 = CorrespondenceMappingAlgorithm::new(
            photo1scaled.width,
            &self.photo1,
            &self.photo2,
            5,
            5
        );
        let mut ocm_manager2 = CorrespondenceMappingAlgorithm::new(
            photo1scaled.width,
            &self.photo2,
            &self.photo1,
            5,
            5
        );

        // Add the initial matched points.
        for pair in pairs.iter() {
            let (p1, p2) = (pair.0, pair.1);  // Directly access the tuple struct fields
            let s = 1.0;
            let v = p1.total_angle - p2.total_angle;
            let vv = -v;

            // Add forward mapping: (photo1 → photo2)
            ocm_manager1.add_init_point(
                p1.center_x as f32 * s,
                p1.center_y as f32 * s,
                p2.center_x as f32 * s,
                p2.center_y as f32 * s,
                vv,
            );

            // Add reverse mapping: (photo2 → photo1)
            ocm_manager2.add_init_point(
                p2.center_x as f32 * s,
                p2.center_y as f32 * s,
                p1.center_x as f32 * s,
                p1.center_y as f32 * s,
                -vv,
            );
        }
        println!("init points added");

        // Run both managers to completion.
        ocm_manager1.run_until_done();
        ocm_manager2.run_until_done();

        // Update total comparisons, store the managers.
        self.total_comparisons = ocm_manager1.get_total_comparisons() + self.ocm_manager2.get_total_comparisons();
        self.ocm_manager1 = ocm_manager1;
        self.ocm_manager2 = ocm_manager2;
    }

    /// Performs the initial matching step using CUDA acceleration:
    /// 1. Scales both `photo1` and `photo2` to `initial_photo_width` (if needed).
    /// 2. Uses CUDA-accelerated feature extraction for circular feature descriptors.
    /// 3. Matches these descriptors with CUDA-accelerated feature matching.
    /// 4. Initializes new [CorrespondenceMappingAlgorithm] instances with the matched points.
    /// 5. Runs both correspondence managers until completion.
    ///
    /// Upon completion, `total_comparisons` is updated with the sum of both managers' comparisons.
    #[cfg(feature = "cuda")]
    pub fn init_cuda(&mut self) -> CudaResult<()> {
        // Scale down images if needed.
        let width = usize::min(self.initial_photo_width, self.photo1.width);
        let photo1scaled = self.photo1.get_scaled_proportional(width);
        let photo2scaled = self.photo2.get_scaled_proportional(width);

        // Use CUDA-accelerated feature matching
        let cuda_matcher = CudaCircularFeatureMatcher::new()?;
        let pairs = cuda_matcher.match_features(&photo1scaled, &photo2scaled, 10)?;

        // Create new managers for the scaled images.
        let mut ocm_manager1 = CorrespondenceMappingAlgorithm::new(
            photo1scaled.width,
            &self.photo1,
            &self.photo2,
            5,
            5
        );
        let mut ocm_manager2 = CorrespondenceMappingAlgorithm::new(
            photo1scaled.width,
            &self.photo2,
            &self.photo1,
            5,
            5
        );

        // Add the initial matched points.
        for pair in pairs.iter() {
            let (p1, p2) = (pair.0, pair.1);  // Directly access the tuple struct fields
            let s = 1.0;
            let v = p1.total_angle - p2.total_angle;
            let vv = -v;

            // Add forward mapping: (photo1 → photo2)
            ocm_manager1.add_init_point(
                p1.center_x as f32 * s,
                p1.center_y as f32 * s,
                p2.center_x as f32 * s,
                p2.center_y as f32 * s,
                vv,
            );

            // Add reverse mapping: (photo2 → photo1)
            ocm_manager2.add_init_point(
                p2.center_x as f32 * s,
                p2.center_y as f32 * s,
                p1.center_x as f32 * s,
                p1.center_y as f32 * s,
                -vv,
            );
        }
        println!("CUDA: init points added");

        // Use CUDA-accelerated correspondence mapping
        let cuda_mapper = CudaCorrespondenceMapping::new()?;
        
        // Run both managers with CUDA acceleration
        let ocm_manager1_result = cuda_mapper.run_correspondence_mapping(
            &mut ocm_manager1,
            &self.photo1,
            &self.photo2
        )?;
        
        let ocm_manager2_result = cuda_mapper.run_correspondence_mapping(
            &mut ocm_manager2,
            &self.photo2,
            &self.photo1
        )?;

        // Update total comparisons, store the managers.
        self.total_comparisons = ocm_manager1_result.total_comparisons + ocm_manager2_result.total_comparisons;
        self.ocm_manager1 = ocm_manager1_result.manager;
        self.ocm_manager2 = ocm_manager2_result.manager;
        
        Ok(())
    }

    /// Uses CUDA acceleration to scale photos before processing
    #[cfg(feature = "cuda")]
    pub fn preprocess_photos_cuda(&mut self) -> CudaResult<()> {
        let cuda_processor = CudaPhotoProcessor::new()?;
        
        // Scale down images if needed
        let width = usize::min(self.initial_photo_width, self.photo1.width);
        self.photo1 = cuda_processor.scale_photo(&self.photo1, width)?;
        self.photo2 = cuda_processor.scale_photo(&self.photo2, width)?;
        
        Ok(())
    }

    /// Returns the total number of comparisons made so far by the two correspondence managers.
    pub fn get_total_comparisons(&self) -> usize {
        self.total_comparisons
    }

    /// Retrieves a pair of [ACGrid]s from the two correspondence managers
    /// (forward and backward mappings).
    ///
    /// # Returns
    /// A tuple `(ACGrid, ACGrid)`, where the first corresponds to `ocm_manager1`
    /// and the second to `ocm_manager2`.
    pub fn get_ac_grids(&self) -> (ACGrid, ACGrid) {
        (self.ocm_manager1.get_ac_grid(), self.ocm_manager2.get_ac_grid())
    }

    /// Gets ACGrids using CUDA acceleration
    #[cfg(feature = "cuda")]
    pub fn get_ac_grids_cuda(&self) -> CudaResult<(ACGrid, ACGrid)> {
        let cuda_mapper = match CudaCorrespondenceMapping::new() {
            Ok(mapper) => mapper,
            Err(e) => {
                eprintln!("Failed to create CUDA mapper: {}. Falling back to CPU implementation.", e);
                return Ok(self.get_ac_grids());
            }
        };
        
        // Call extract_ac_grids with the correct parameter types
        match cuda_mapper.extract_ac_grids(&self.ocm_manager1, &self.ocm_manager2) {
            Ok(ac_grids) => Ok(ac_grids),
            Err(e) => {
                eprintln!("CUDA ACGrid extraction failed: {}. Falling back to CPU implementation.", e);
                Ok(self.get_ac_grids())
            }
        }
    }

    /// Performs an **iteration** of the mapping refinement process:
    ///
    /// 1. Extracts the current dense photo maps from each manager.
    /// 2. Removes outliers in each map by checking consistency with the other map.
    /// 3. Smooths (averages) each map over several iterations to reduce noise.
    /// 4. Re-initializes the `ocm_manager1` and `ocm_manager2` with the smoothed maps.
    /// 5. Runs both managers until done again.
    /// 6. Accumulates the total comparison count.
    ///
    /// # Parameters
    /// - `photo_width`: Used to control internal scaling for the re-init step.
    /// - `grid_cell_size`: A parameter for how large each grid cell is in the new managers.
    /// - `neighborhood_radius`: Another parameter controlling how far each manager looks for matches.
    /// - `smooth_iterations`: How many times to smooth (average) the grid maps.
    /// - `clean_max_dist`: Maximum distance threshold used in outlier removal.
    pub fn iterate(
        &mut self,
        photo_width: usize,
        grid_cell_size: usize,
        neighborhood_radius: usize,
        smooth_iterations: usize,
        clean_max_dist: f32
    ) {
        println!("{photo_width}");
        let mut pm1 = self.ocm_manager1.get_photo_mapping();
        let mut pm2 = self.ocm_manager2.get_photo_mapping();

        // Remove outliers by forward-backward consistency check.
        pm1.remove_outliers(&pm2, clean_max_dist);
        pm2.remove_outliers(&pm1, clean_max_dist);

        // Smooth the remaining mapping.
        let pm1_smooth = pm1.smooth_grid_points_n_times(smooth_iterations);
        let pm2_smooth = pm2.smooth_grid_points_n_times(smooth_iterations);

        // Re-initialize managers with the smoothed maps.
        let mut ocm_manager1 = CorrespondenceMappingAlgorithm::new(
            photo_width,
            &self.photo1,
            &self.photo2,
            grid_cell_size,
            neighborhood_radius
        );
        let mut ocm_manager2 = CorrespondenceMappingAlgorithm::new(
            photo_width,
            &self.photo2,
            &self.photo1,
            grid_cell_size,
            neighborhood_radius
        );
        ocm_manager1.init_from_photomapping(&pm1_smooth);
        ocm_manager2.init_from_photomapping(&pm2_smooth);

        // Run again with the updated maps.
        ocm_manager1.run_until_done();
        ocm_manager2.run_until_done();

        // Accumulate total comparisons.
        self.total_comparisons +=
            ocm_manager1.get_total_comparisons() + ocm_manager2.get_total_comparisons();

        // Store the new managers.
        self.ocm_manager1 = ocm_manager1;
        self.ocm_manager2 = ocm_manager2;
    }

    /// CUDA-accelerated version of iterate
    #[cfg(feature = "cuda")]
    pub fn iterate_cuda(
        &mut self,
        photo_width: usize,
        grid_cell_size: usize,
        neighborhood_radius: usize,
        smooth_iterations: usize,
        clean_max_dist: f32
    ) -> CudaResult<()> {
        println!("{photo_width}");
        let mut pm1 = self.ocm_manager1.get_photo_mapping();
        let mut pm2 = self.ocm_manager2.get_photo_mapping();

        // Use CUDA acceleration for outlier removal and smoothing
        let cuda_mapper = CudaCorrespondenceMapping::new()?;
        
        // Remove outliers by forward-backward consistency check using CUDA
        cuda_mapper.remove_outliers(&mut pm1, &pm2, clean_max_dist)?;
        cuda_mapper.remove_outliers(&mut pm2, &pm1, clean_max_dist)?;

        // Smooth the remaining mapping using CUDA
        let pm1_smooth = cuda_mapper.smooth_grid_points(&pm1, smooth_iterations)?;
        let pm2_smooth = cuda_mapper.smooth_grid_points(&pm2, smooth_iterations)?;

        // Re-initialize managers with the smoothed maps.
        let mut ocm_manager1 = CorrespondenceMappingAlgorithm::new(
            photo_width,
            &self.photo1,
            &self.photo2,
            grid_cell_size,
            neighborhood_radius
        );
        let mut ocm_manager2 = CorrespondenceMappingAlgorithm::new(
            photo_width,
            &self.photo2,
            &self.photo1,
            grid_cell_size,
            neighborhood_radius
        );
        ocm_manager1.init_from_photomapping(&pm1_smooth);
        ocm_manager2.init_from_photomapping(&pm2_smooth);

        // Run with CUDA acceleration
        let ocm_manager1_result = cuda_mapper.run_correspondence_mapping(
            &mut ocm_manager1,
            &self.photo1,
            &self.photo2
        )?;
        
        let ocm_manager2_result = cuda_mapper.run_correspondence_mapping(
            &mut ocm_manager2,
            &self.photo2,
            &self.photo1
        )?;

        // Accumulate total comparisons.
        self.total_comparisons +=
            ocm_manager1_result.total_comparisons + ocm_manager2_result.total_comparisons;

        // Store the new managers.
        self.ocm_manager1 = ocm_manager1_result.manager;
        self.ocm_manager2 = ocm_manager2_result.manager;
        
        Ok(())
    }

    /// Retrieves the final forward and backward mappings, removing any remaining outliers
    /// with a given distance threshold.
    ///
    /// # Parameters
    /// - `clean_max_dist`: The maximum distance for outlier checking.
    ///   Points whose forward-backward mapping is too large are discarded.
    ///
    /// # Returns
    /// A tuple of two [DensePhotoMap]s:
    /// - First: from `photo1` to `photo2`.
    /// - Second: from `photo2` to `photo1`.
    ///
    /// Additionally, prints the total number of comparisons made in the process.
    pub fn get_result(&mut self, clean_max_dist: f32) -> (DensePhotoMap, DensePhotoMap) {
        let mut pm1 = self.ocm_manager1.get_photo_mapping();
        let mut pm2 = self.ocm_manager2.get_photo_mapping();

        // Remove outliers in both directions.
        pm1.remove_outliers(&pm2, clean_max_dist);
        pm2.remove_outliers(&pm1, clean_max_dist);

        println!("tot comparisons : {}", self.total_comparisons);

        (pm1, pm2)
    }

    /// Get result using CUDA acceleration
    #[cfg(feature = "cuda")]
    pub fn get_result_cuda(&mut self, clean_max_dist: f32) -> CudaResult<(DensePhotoMap, DensePhotoMap)> {
        let mut pm1 = self.ocm_manager1.get_photo_mapping();
        let mut pm2 = self.ocm_manager2.get_photo_mapping();

        // Use CUDA for outlier removal
        let cuda_mapper = CudaCorrespondenceMapping::new()?;
        
        // Remove outliers in both directions.
        cuda_mapper.remove_outliers(&mut pm1, &pm2, clean_max_dist)?;
        cuda_mapper.remove_outliers(&mut pm2, &pm1, clean_max_dist)?;

        println!("CUDA: tot comparisons : {}", self.total_comparisons);

        Ok((pm1, pm2))
    }

    /// Computes how much of the mapping is valid (non-outlier) in `ocm_manager1`,
    /// by performing a small outlier check with threshold `2.0`.
    ///
    /// # Returns
    /// The fraction of valid cells in the forward map, a value between 0.0 and 1.0.
    pub fn get_matched_area(& self) -> f32 {
        let pm1 = self.ocm_manager1.get_photo_mapping();
        let pm2 = self.ocm_manager2.get_photo_mapping();

        // Clone to avoid mutating the originals during outlier removal.
        let mut pm1 = pm1.clone();
        let mut pm2 = pm2.clone();

        pm1.remove_outliers(&pm2, 2.0);
        pm2.remove_outliers(&pm1, 2.0);

        pm1.calculate_used_area()
    }

    /// CUDA-accelerated version to compute how much of the mapping is valid
    #[cfg(feature = "cuda")]
    pub fn get_matched_area_cuda(&self) -> CudaResult<f32> {
        let pm1 = self.ocm_manager1.get_photo_mapping();
        let pm2 = self.ocm_manager2.get_photo_mapping();

        // Use CUDA for efficient area calculation
        match CudaCorrespondenceMapping::new() {
            Ok(cuda_mapper) => {
                match cuda_mapper.calculate_matched_area(&pm1, &pm2, 2.0) {
                    Ok(area) => Ok(area),
                    Err(e) => {
                        eprintln!("CUDA area calculation failed: {}. Falling back to CPU implementation.", e);
                        Ok(self.get_matched_area())
                    }
                }
            },
            Err(e) => {
                eprintln!("Failed to create CUDA mapper: {}. Falling back to CPU implementation.", e);
                Ok(self.get_matched_area())
            }
        }
    }

    /// Performs complete CUDA-accelerated processing pipeline in one call
    #[cfg(feature = "cuda")]
    pub fn process_with_cuda(
        &mut self, 
        iterations: usize, 
        grid_cell_size: usize,
        neighborhood_radius: usize,
        smooth_iterations: usize,
        clean_max_dist: f32
    ) -> CudaResult<(DensePhotoMap, DensePhotoMap)> {
        // Initialize with CUDA
        self.init_cuda()?;
        
        // Perform iterations
        for i in 0..iterations {
            println!("CUDA iteration {}/{}", i+1, iterations);
            
            // Width for this iteration - can use a decreasing width strategy
            // to progressively improve precision
            let photo_width = self.initial_photo_width / (1 + i.min(2));
            
            self.iterate_cuda(
                photo_width, 
                grid_cell_size, 
                neighborhood_radius,
                smooth_iterations,
                clean_max_dist
            )?;
        }
        
        // Get final result
        self.get_result_cuda(clean_max_dist)
    }
    
    /// Generate a CUDA-accelerated photo mapping directly from source photos
    /// 
    /// This method performs the end-to-end pipeline with CUDA acceleration,
    /// optimized for speed and accuracy:
    /// 
    /// 1. Scales photos using CUDA
    /// 2. Extracts and matches features with CUDA
    /// 3. Iteratively refines the mapping entirely on the GPU
    /// 4. Returns the final CUDA-processed dense maps
    #[cfg(feature = "cuda")]
    pub fn generate_mapping_cuda(
        &mut self,
        iterations: usize,
        clean_max_dist: f32
    ) -> CudaResult<(DensePhotoMap, DensePhotoMap)> {
        // Ensure CUDA is available
        if !cuda2::is_available() {
            return Err("CUDA is not available. Cannot generate mapping with CUDA.".to_string());
        }
        
        // Get CUDA context
        let cuda_context = CudaContext::new()
            .map_err(|e| format!("Failed to create CUDA context: {:?}", e))?;
        
        // Create CUDA processors
        let cuda_mapper = CudaCorrespondenceMapping::new()?;
        let cuda_photo_processor = CudaPhotoProcessor::new()?;
        
        // Scale photos
        let width = usize::min(self.initial_photo_width, self.photo1.width);
        let photo1_scaled = cuda_photo_processor.scale_photo(&self.photo1, width)?;
        let photo2_scaled = cuda_photo_processor.scale_photo(&self.photo2, width)?;
        
        // Replace original photos with scaled versions to save memory
        self.photo1 = photo1_scaled;
        self.photo2 = photo2_scaled;
        
        // Extract and match features
        println!("CUDA: Extracting and matching features...");
        self.init_cuda()?;
        
        // Process iterations
        println!("CUDA: Running {} iterations...", iterations);
        
        for i in 0..iterations {
            let grid_cell_size = 5;
            let neighborhood_radius = 5;
            let smooth_iterations = 3;
            
            self.iterate_cuda(
                width,
                grid_cell_size,
                neighborhood_radius,
                smooth_iterations,
                clean_max_dist
            )?;
            
            println!("CUDA: Iteration {}/{} complete", i+1, iterations);
        }
        
        // Get final result
        println!("CUDA: Getting final result...");
        let result = self.get_result_cuda(clean_max_dist)?;
        
        println!("CUDA: Mapping generation complete.");
        Ok(result)
    }
    
    // Add a CPU fallback version of the CUDA methods for when CUDA is not enabled
    #[cfg(not(feature = "cuda"))]
    pub fn process_with_cuda(
        &mut self, 
        iterations: usize, 
        grid_cell_size: usize,
        neighborhood_radius: usize,
        smooth_iterations: usize,
        clean_max_dist: f32
    ) -> Result<(DensePhotoMap, DensePhotoMap), String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn generate_mapping_cuda(
        &mut self,
        _iterations: usize,
        _clean_max_dist: f32
    ) -> Result<(DensePhotoMap, DensePhotoMap), String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn init_cuda(&mut self) -> Result<(), String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn preprocess_photos_cuda(&mut self) -> Result<(), String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn get_ac_grids_cuda(&self) -> Result<(ACGrid, ACGrid), String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn iterate_cuda(
        &mut self,
        _photo_width: usize,
        _grid_cell_size: usize,
        _neighborhood_radius: usize,
        _smooth_iterations: usize,
        _clean_max_dist: f32
    ) -> Result<(), String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn get_result_cuda(&mut self, _clean_max_dist: f32) -> Result<(DensePhotoMap, DensePhotoMap), String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn get_matched_area_cuda(&self) -> Result<f32, String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn new_with_cuda(_photo1: Photo, _photo2: Photo, _photo_width: usize) -> Result<Self, String> {
        Err("CUDA support is not enabled. Recompile with --features cuda".to_string())
    }

    /// Complete processing pipeline with configurable steps
    /// 
    /// This method provides a unified interface to the entire pixelmap process:
    /// 1. Initialization with feature extraction and matching
    /// 2. Multiple iterations of refinement with configurable parameters
    /// 3. Final cleanup and result generation
    /// 
    /// Uses the CPU implementation but tries to optimize performance.
    pub fn process_complete(
        &mut self,
        iterations: usize,
        grid_cell_size: usize,
        neighborhood_radius: usize,
        smooth_iterations: usize,
        clean_max_dist: f32
    ) -> (DensePhotoMap, DensePhotoMap) {
        // Initialize the processor
        self.init();
        
        // Perform iterations
        for i in 0..iterations {
            println!("CPU iteration {}/{}", i+1, iterations);
            
            // Width for this iteration - can use a decreasing width strategy
            // to progressively improve precision
            let photo_width = self.initial_photo_width / (1 + i.min(2));
            
            self.iterate(
                photo_width,
                grid_cell_size,
                neighborhood_radius,
                smooth_iterations,
                clean_max_dist
            );
        }
        
        // Get final result
        self.get_result(clean_max_dist)
    }
}
