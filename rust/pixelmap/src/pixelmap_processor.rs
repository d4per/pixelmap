use crate::ac_grid::ACGrid;
use crate::circular_feature_descriptor_matcher::CircularFeatureDescriptorMatcher;
use crate::circular_feature_grid;
use crate::correspondence_mapping_algorithm::CorrespondenceMappingAlgorithm;
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;

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
        for (p1, p2) in pairs {
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
}
