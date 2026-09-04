//! The pipeline driver behind [`crate::Correspondence`].

use crate::ac_grid::ACGrid;
use crate::circular_feature_descriptor_matcher::CircularFeatureDescriptorMatcher;
use crate::circular_feature_grid;
use crate::correspondence_mapping_algorithm::CorrespondenceMappingAlgorithm;
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;
use std::collections::HashMap;
use std::sync::Arc;

/// The seed [`PixelMapProcessor::new`] uses when the caller does not pick one.
///
/// A fixed value rather than OS entropy: a library that returns a different answer each
/// time it is called is hard to test against and hard to build on, so reproducibility is
/// the default and [`PixelMapProcessor::with_seed`] is the way to vary it.
pub const DEFAULT_SEED: u64 = 0x5049_5845_4C4D_4150; // "PIXELMAP"

/// Manages a pipeline for finding and refining a mapping between two images (`photo1` and `photo2`).
///
/// The process typically involves:
/// 1. **Circular feature extraction and matching** on scaled versions of the photos.
/// 2. Initializing a `CorrespondenceMappingAlgorithm` with matched points.
/// 3. Iterative refinement steps that remove outliers, smooth the mappings, and further optimize.
/// 4. Producing final [DensePhotoMap]s describing forward (`photo1` → `photo2`) and backward (`photo2` → `photo1`) transformations.
pub struct PixelMapProcessor {
    /// The first image to be matched/registered.
    photo1: Arc<Photo>,

    /// The second image to be matched/registered.
    photo2: Arc<Photo>,

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

    /// Base seed for the two solvers' queue shuffling. See [`PixelMapProcessor::with_seed`].
    seed: u64,

    /// Counts the solvers built by *this* processor, so each gets a distinct RNG stream
    /// without reaching for process-global state. Purely local, so the mapping depends
    /// only on the inputs and `seed` — not on what the rest of the process has done.
    algorithms_built: u64,

    /// Scaled copies of `photo1`/`photo2`, keyed by `(photo index, target width)`.
    ///
    /// Every `iterate()` builds two managers and each used to rescale both originals
    /// from full resolution, even though the schedule only ever asks for three distinct
    /// widths and the two managers of one iteration need the very same pair of images
    /// with their roles swapped. At `high` that was 52 resamples of a 4096x2304 source.
    scaled: HashMap<(usize, usize), Arc<Photo>>,
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
    pub fn new(photo1: Arc<Photo>, photo2: Arc<Photo>, photo_width: usize) -> Self {
        Self::with_seed(photo1, photo2, photo_width, DEFAULT_SEED)
    }

    /// Same as [`Self::new`], but pins the seed that drives the solvers' queue shuffling.
    ///
    /// The order in which the queue is drained decides which local optimum the relaxation
    /// settles into, so the seed is what makes a run reproducible: the same photos, the
    /// same schedule and the same seed give the same mapping. [`Self::new`] uses
    /// [`DEFAULT_SEED`], so reproducibility is the default rather than something the
    /// caller has to opt into.
    pub fn with_seed(
        photo1: Arc<Photo>,
        photo2: Arc<Photo>,
        photo_width: usize,
        seed: u64,
    ) -> Self {
        // Temporary dummy Photo, used only so that CorrespondenceMappingAlgorithm can be constructed.
        let dummy_photo = Photo::default();

        PixelMapProcessor {
            photo1,
            photo2,
            ocm_manager1: CorrespondenceMappingAlgorithm::new(photo_width, &dummy_photo, &dummy_photo, 5, 5, seed),
            ocm_manager2: CorrespondenceMappingAlgorithm::new(photo_width, &dummy_photo, &dummy_photo, 5, 5, seed),
            total_comparisons: 0,
            initial_photo_width: photo_width,
            seed,
            algorithms_built: 0,
            scaled: HashMap::new(),
        }
    }

    /// Hands out the next solver seed for this processor.
    ///
    /// Mixing a local counter in keeps the forward and backward passes on separate
    /// streams (they would otherwise shuffle identically) while staying a pure function
    /// of `seed` and the number of solvers built so far.
    fn next_seed(&mut self) -> u64 {
        let n = self.algorithms_built;
        self.algorithms_built += 1;
        self.seed ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Returns `photo1` (`which == 0`) or `photo2` (`which == 1`) scaled to `width`,
    /// computing it at most once per `(photo, width)` pair.
    fn scaled(&mut self, which: usize, width: usize) -> Arc<Photo> {
        if let Some(p) = self.scaled.get(&(which, width)) {
            return p.clone();
        }
        let src = if which == 0 { &self.photo1 } else { &self.photo2 };
        let p = Arc::new(src.get_scaled_proportional(width));
        self.scaled.insert((which, width), p.clone());
        p
    }

    /// Performs the initial matching step:
    /// 1. Scales both `photo1` and `photo2` to `initial_photo_width` (if needed).
    /// 2. Uses `CircularFeatureGrid` to extract circular feature descriptors.
    /// 3. Matches these descriptors with `CircularFeatureDescriptorMatcher`.
    /// 4. Initializes new `CorrespondenceMappingAlgorithm` instances with the matched points.
    /// 5. Runs both correspondence managers until completion.
    ///
    /// Upon completion, `total_comparisons` is updated with the sum of both managers' comparisons.
    pub fn init(&mut self) {
        // Scale down images if needed.
        let width = usize::min(self.initial_photo_width, self.photo1.width);
        let photo1scaled = self.scaled(0, width);
        let photo2scaled = self.scaled(1, width);

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
        let w = photo1scaled.width;
        let (s1, s2) = (self.scaled(0, w), self.scaled(1, w));
        let (seed1, seed2) = (self.next_seed(), self.next_seed());
        let mut ocm_manager1 = CorrespondenceMappingAlgorithm::with_scaled(s1.clone(), s2.clone(), 5, 5, seed1);
        let mut ocm_manager2 = CorrespondenceMappingAlgorithm::with_scaled(s2, s1, 5, 5, seed2);

        // Add the initial matched points.
        for m in pairs {
            let vv = -m.angle_delta;

            // Add forward mapping: (photo1 → photo2)
            ocm_manager1.add_init_point(
                m.x1 as f32, m.y1 as f32, m.x2 as f32, m.y2 as f32, vv,
            );

            // Add reverse mapping: (photo2 → photo1)
            ocm_manager2.add_init_point(
                m.x2 as f32, m.y2 as f32, m.x1 as f32, m.y1 as f32, -vv,
            );
        }

        // Run both managers to completion.
        ocm_manager1.run_until_done();
        ocm_manager2.run_until_done();

        // Update total comparisons, store the managers.
        self.total_comparisons = ocm_manager1.get_total_comparisons() + ocm_manager2.get_total_comparisons();
        self.ocm_manager1 = ocm_manager1;
        self.ocm_manager2 = ocm_manager2;
    }

    /// Returns the total number of comparisons made so far by the two correspondence managers.
    pub fn get_total_comparisons(&self) -> usize {
        self.total_comparisons
    }

    /// Retrieves a pair of `ACGrid`s from the two correspondence managers
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
        let mut pm1 = self.ocm_manager1.get_photo_mapping();
        let mut pm2 = self.ocm_manager2.get_photo_mapping();

        // Remove outliers by forward-backward consistency check.
        
        pm1.remove_outliers(&pm2, clean_max_dist);
        pm2.remove_outliers(&pm1, clean_max_dist);

        // Smooth the remaining mapping.
        let pm1_smooth = pm1.smooth_grid_points_n_times(smooth_iterations);
        let pm2_smooth = pm2.smooth_grid_points_n_times(smooth_iterations);

        // Re-initialize managers with the smoothed maps.
        let (s1, s2) = (self.scaled(0, photo_width), self.scaled(1, photo_width));
        let (seed1, seed2) = (self.next_seed(), self.next_seed());
        let mut ocm_manager1 = CorrespondenceMappingAlgorithm::with_scaled(
                s1.clone(), s2.clone(), grid_cell_size, neighborhood_radius, seed1);
        let mut ocm_manager2 = CorrespondenceMappingAlgorithm::with_scaled(
                s2, s1, grid_cell_size, neighborhood_radius, seed2);
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
    pub fn get_result(&mut self, clean_max_dist: f32) -> (DensePhotoMap, DensePhotoMap) {
        let mut pm1 = self.ocm_manager1.get_photo_mapping();
        let mut pm2 = self.ocm_manager2.get_photo_mapping();

        // Remove outliers in both directions.
        pm1.remove_outliers(&pm2, clean_max_dist);
        pm2.remove_outliers(&pm1, clean_max_dist);

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
