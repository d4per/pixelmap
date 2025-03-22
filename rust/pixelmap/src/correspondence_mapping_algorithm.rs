use crate::ac_grid::ACGrid;
use crate::affine_transform::AffineTransform;
use crate::correspondence_scoring::CorrespondenceScoring;
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;
use rand::seq::SliceRandom;
use rand::Rng;
use std::rc::Rc;

/// Manages an iterative process for matching two images (`photo1` and `photo2`) by
/// assigning an [AffineTransform] to each cell in a grid. The algorithm refines these
/// transforms based on a scoring function (see [CorrespondenceScoring]) until no
/// further improvements are found.
///
/// # How It Works
/// 1. **Initialization**: The images are scaled, a grid is created, and initial transforms
///    are pushed into a queue.
/// 2. **Scoring**: Each transform is evaluated (lower score is better).
/// 3. **Refinement**: Transforms are slightly adjusted, and any improvements are propagated
///    to neighboring cells, allowing the algorithm to converge over multiple iterations.
/// 4. **Final Mapping**: Once no more improvements occur, you can retrieve a [DensePhotoMap]
///    describing the best transforms found for each cell.
pub struct CorrespondenceMappingAlgorithm {
    /// Reference-counted handle to the first (scaled) photo.
    pub photo1: Rc<Photo>,
    /// Reference-counted handle to the second (scaled) photo.
    pub photo2: Rc<Photo>,
    /// Size of each grid cell in pixels.
    grid_cell_size: usize,
    /// A queue of candidate [AffineTransform] objects to evaluate and refine.
    queue: Vec<AffineTransform>,
    /// Scores transforms by comparing corresponding regions of `photo1` and `photo2`.
    scorer: CorrespondenceScoring,
    /// A 2D grid that stores the best-known transform for each cell (and its score).
    ac_grid: ACGrid,
}

impl CorrespondenceMappingAlgorithm {
    /// Creates a new mapping algorithm for the given images and parameters.
    ///
    /// # Parameters
    /// - `photo_width`: The width to which both photos are scaled (preserves aspect ratio).
    /// - `photo1`, `photo2`: References to the original images.
    /// - `grid_cell_size`: Size of each cell in the grid (in pixels).
    /// - `neighborhood_radius`: Radius (in pixels) for the circular neighborhood scoring.
    ///
    /// # Returns
    /// A new `CorrespondenceMappingAlgorithm` with scaled images, an empty queue, and
    /// an initialized scoring mechanism and grid.
    pub fn new(
        photo_width: usize,
        photo1: &Photo,
        photo2: &Photo,
        grid_cell_size: usize,
        neighborhood_radius: usize
    ) -> Self {
        // Scale the original photos to the specified width.
        let photo1a = Rc::new(photo1.get_scaled_proportional(photo_width));
        let photo2a = Rc::new(photo2.get_scaled_proportional(photo_width));

        // Determine how many cells fit in the scaled images.
        let grid_width = photo1a.width / grid_cell_size + 1;
        let grid_height = photo1a.height / grid_cell_size + 1;

        CorrespondenceMappingAlgorithm {
            photo1: photo1a.clone(),
            photo2: photo2a.clone(),
            grid_cell_size,
            queue: vec![],
            scorer: CorrespondenceScoring::new(
                photo1a.clone(),
                photo2a.clone(),
                neighborhood_radius as isize
            ),
            ac_grid: ACGrid::new(grid_width, grid_height),
        }
    }

    /// Retrieves a clone of the internal [ACGrid], which stores the best-known transforms
    /// for each cell.
    pub fn get_ac_grid(&self) -> ACGrid {
        self.ac_grid.clone()
    }

    /// Returns the total number of scoring function invocations
    /// performed so far (for diagnostic or debugging purposes).
    pub fn get_total_comparisons(&self) -> usize {
        self.scorer.get_num_comparisons()
    }

    /// Repeatedly processes (and shuffles) the queue of transforms until no more
    /// improvements can be made (i.e., the queue is empty at the end of a cycle).
    pub fn run_until_done(&mut self) {
        loop {
            // Shuffle transforms to avoid bias.
            self.queue.shuffle(&mut rand::thread_rng());
            let is_done = self.run_queue();
            if is_done {
                break;
            }
        }
    }

    /// Adds a new transform to the queue, deriving rotation from the given `angle`.
    /// The transform is then "snapped" to a corresponding grid coordinate.
    ///
    /// # Parameters
    /// - `x1, y1`: Origin coordinates in `photo1`.
    /// - `x2, y2`: Target coordinates in `photo2`.
    /// - `angle`: Rotation angle (in radians) around `(x1, y1)`.
    pub fn add_init_point(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, angle: f32) {
        let s = f32::sin(angle);
        let c = f32::cos(angle);

        // Create an affine transform from x1,y1 to x2,y2 with rotation.
        let cm = AffineTransform {
            origin_x: x1 as u16,
            origin_y: y1 as u16,
            translate_x: x2,
            translate_y: y2,
            a11: c,
            a12: -s,
            a21: s,
            a22: c,
        };
        // Snap transform to the nearest grid cell.
        let snap_to_grid_cm = cm.extrapolate_mapping(
            (x1 as usize / self.grid_cell_size * self.grid_cell_size) as u16,
            (y1 as usize / self.grid_cell_size * self.grid_cell_size) as u16
        );
        self.queue.push(snap_to_grid_cm);
    }

    /// Initializes the queue with transforms derived from an existing [DensePhotoMap].
    /// This allows the algorithm to continue refining a previously computed mapping.
    ///
    /// # Parameters
    /// - `pm`: A [`DensePhotoMap`] containing approximate mappings from `photo1` to `photo2`.
    pub fn init_from_photomapping(&mut self, pm: &DensePhotoMap) {
        let pm_grid_cell_size = pm.get_grid_cell_size();

        // For each cell in the DensePhotoMap, create an AffineTransform
        // and push it into the queue for refinement.
        for y in 0 .. pm.grid_height {
            for x in 0 .. pm.grid_width {
                let (x2a, y2a) = pm.get_grid_coordinates(x, y);
                if x2a.is_nan() {
                    continue;
                }
                // Compute the origin in the scaled photo1.
                let x1 = f32::round(
                    ((x * pm_grid_cell_size * self.photo1.width) as f32)
                        / pm.photo1.width as f32
                ) as usize;
                let y1 = f32::round(
                    ((y * pm_grid_cell_size * self.photo1.width) as f32)
                        / pm.photo1.width as f32
                ) as usize;

                // Compute the mapped position in scaled photo2.
                let x2 = x2a * self.photo1.width as f32 / pm.photo1.width as f32;
                let y2 = y2a * self.photo1.width as f32 / pm.photo1.width as f32;

                let mut cm = AffineTransform {
                    origin_x: x1 as u16,
                    origin_y: y1 as u16,
                    translate_x: x2,
                    translate_y: y2,
                    a11: 1.0,
                    a12: 0.0,
                    a21: 0.0,
                    a22: 1.0,
                };

                // Approximate local scaling from neighbors (left/up).
                if x > 0 {
                    let (left_x, left_y) = pm.get_grid_coordinates(x - 1, y);
                    if !left_x.is_nan() {
                        cm.a11 = (x2a - left_x) / pm_grid_cell_size as f32;
                        cm.a21 = (y2a - left_y) / pm_grid_cell_size as f32;
                    }
                }
                if y > 0 {
                    let (up_x, up_y) = pm.get_grid_coordinates(x, y - 1);
                    if !up_x.is_nan() {
                        cm.a22 = (y2a - up_y) / pm_grid_cell_size as f32;
                        cm.a12 = (x2a - up_x) / pm_grid_cell_size as f32;
                    }
                }

                // Snap the transform to the nearest grid coordinates.
                let snap_to_grid_x = f32::round(x1 as f32 / self.grid_cell_size as f32) as usize * self.grid_cell_size;
                let snap_to_grid_y = f32::round(y1 as f32 / self.grid_cell_size as f32) as usize * self.grid_cell_size;
                let snap_to_grid_cm = cm.extrapolate_mapping(
                    snap_to_grid_x as u16,
                    snap_to_grid_y as u16
                );
                self.queue.push(snap_to_grid_cm);
            }
        }
    }

    /// Initializes the queue with the "identity" transform for each cell. This means
    /// each `(x, y)` in `photo1` initially maps to the same `(x, y)` in `photo2`,
    /// with no rotation or scaling.
    pub fn init_identity(&mut self) {
        for y in (0 .. self.photo1.height).step_by(self.grid_cell_size) {
            for x in (0 .. self.photo1.width).step_by(self.grid_cell_size) {
                let cm = AffineTransform {
                    origin_x: x as u16,
                    origin_y: y as u16,
                    translate_x: x as f32,
                    translate_y: y as f32,
                    a11: 1.0,
                    a12: 0.0,
                    a21: 0.0,
                    a22: 1.0,
                };
                self.queue.push(cm);
            }
        }
    }

    /// Processes the current queue of transforms. For each transform:
    /// 1. Validates its scale and position (no out-of-bounds).
    /// 2. Uses `optimize_position` to adjust it.
    /// 3. Checks whether it improves upon the transform stored in the grid cell.
    /// 4. If improved, updates that cell and extrapolates to neighboring cells,
    ///    pushing new transforms back into a temporary queue.
    ///
    /// Returns `true` if the queue is empty afterward (no improvements), or
    /// `false` if there are still transforms to process in the next iteration.
    fn run_queue(&mut self) -> bool {
        let ac_grid = &self.ac_grid;
        let mut out_queue: Vec<AffineTransform> = vec![];

        loop {
            let cm_opt = self.queue.pop();
            if cm_opt.is_none() {
                break;
            }
            let cm = cm_opt.unwrap();

            // Skip if the transform is out of scale or invalid in terms of translation.
            if !cm.is_scale_valid(4.0) || !self.is_valid(&cm) {
                continue;
            }

            // Attempt a small local optimization on (cm).
            let (score, cm_out) = self.optimize_position(&cm);

            // Determine which grid cell (cm_out) belongs to.
            let grid_x = cm_out.origin_x as usize / self.grid_cell_size;
            let grid_y = cm_out.origin_y as usize / self.grid_cell_size;

            // Check bounds in the ACGrid.
            if grid_x >= ac_grid.get_grid_width() || grid_y >= ac_grid.get_grid_height() {
                println!(
                    "grid_x {} grid_y {} out of range: {} {}",
                    grid_x, grid_y,
                    ac_grid.get_grid_width(), ac_grid.get_grid_height()
                );
                continue;
            }

            // Retrieve the current best cell transform/score.
            let grid_square = ac_grid.get_grid_square(grid_x, grid_y);
            if grid_square.get_score() > score {
                // If we've found an improvement, update and extrapolate to neighbors.
                grid_square.set(cm_out, score);

                // Generate child transforms for neighboring cells and push them to out_queue.
                self.update_neighbors(grid_x, grid_y, &cm, &mut out_queue);
            }
        }

        // Replace the main queue with out_queue for the next iteration.
        self.queue = out_queue;
        // If it's empty, the algorithm is done (no further improvements).
        self.queue.is_empty()
    }

    /// Updates neighboring cells by generating child transforms and pushing them to the out_queue.
    fn update_neighbors(&self, grid_x: usize, grid_y: usize, cm: &AffineTransform, out_queue: &mut Vec<AffineTransform>) {
        let ac_grid = &self.ac_grid;

        if grid_x > 0 {
            out_queue.push(
                cm.extrapolate_mapping(((grid_x - 1) * self.grid_cell_size) as u16,
                                       (grid_y * self.grid_cell_size) as u16)
            );
        }
        if grid_x < ac_grid.get_grid_width() - 1 {
            out_queue.push(
                cm.extrapolate_mapping(((grid_x + 1) * self.grid_cell_size) as u16,
                                       (grid_y * self.grid_cell_size) as u16)
            );
        }
        if grid_y > 0 {
            out_queue.push(
                cm.extrapolate_mapping((grid_x * self.grid_cell_size) as u16,
                                       ((grid_y - 1) * self.grid_cell_size) as u16)
            );
        }
        if grid_y < ac_grid.get_grid_height() - 1 {
            out_queue.push(
                cm.extrapolate_mapping((grid_x * self.grid_cell_size) as u16,
                                       ((grid_y + 1) * self.grid_cell_size) as u16)
            );
        }
    }

    /// Returns the current length of the queue (for debugging or monitoring).
    pub fn queue_length(&self) -> usize {
        self.queue.len()
    }

    /// Performs a simple local search by checking a few neighboring translations
    /// (±1 pixel in x or y) to see if they improve the score.
    ///
    /// # Returns
    /// A tuple `(best_score, best_cm)`, where `best_cm` is the transform
    /// (among the few tested) with the lowest score.
    pub fn optimize_position(&self, cm: &AffineTransform) -> (f32, AffineTransform) {
        let mut best_cm = cm;
        let score0 = self.scorer.calculate_similarity_score(cm);
        let mut best_score = score0;

        // Try shifting translate_x by ±1 and see if score is better.
        let test_cmx1 = AffineTransform {
            translate_x: cm.translate_x - 1.0,
            ..*cm
        };
        let score_x1 = self.scorer.calculate_similarity_score(&test_cmx1);

        let test_cmx2 = AffineTransform {
            translate_x: cm.translate_x + 1.0,
            ..*cm
        };
        let score_x2 = self.scorer.calculate_similarity_score(&test_cmx2);

        // Compare x-shifted results.
        if score_x1 < best_score && score_x1 < score_x2 {
            best_cm = &test_cmx1;
            best_score = score_x1;
        } else if score_x2 < best_score {
            best_cm = &test_cmx2;
            best_score = score_x2;
        }
        // Evaluate shifting translate_y by ±1 on the chosen "best" so far.
        let test_cmy1 = AffineTransform {
            translate_y: best_cm.translate_y - 1.0,
            ..*best_cm
        };
        let score_y1 = self.scorer.calculate_similarity_score(&test_cmy1);

        let test_cmy2 = AffineTransform {
            translate_y: best_cm.translate_y + 1.0,
            ..*best_cm
        };
        let score_y2 = self.scorer.calculate_similarity_score(&test_cmy2);

        // Compare y-shifted results.
        if score_y1 < best_score && score_y1 < score_y2 {
            best_cm = &test_cmy1;
            best_score = score_y1;
        } else if score_y2 < best_score {
            best_cm = &test_cmy2;
            best_score = score_y2;
        }

        (best_score, *best_cm)
    }

    /// Checks if a given [AffineTransform] has valid translation coordinates (within image bounds).
    fn is_valid(&self, cm: &AffineTransform) -> bool {
        cm.translate_x >= 0.0
            && cm.translate_y >= 0.0
            && cm.translate_x <= self.photo1.width as f32
            && cm.translate_y <= self.photo1.height as f32
    }

    /// Builds a [DensePhotoMap] from the best transforms currently stored in the [ACGrid].
    /// Each grid cell is translated into a single mapping `(x, y) -> (translate_x, translate_y)`.
    ///
    /// # Returns
    /// A `DensePhotoMap` describing how each cell in `photo1` maps to coordinates in `photo2`.
    pub fn get_photo_mapping(&self) -> DensePhotoMap {
        let ac_grid = &self.ac_grid;
        let mut pm = DensePhotoMap::new(
            self.photo1.clone(),
            self.photo2.clone(),
            ac_grid.get_grid_width(),
            ac_grid.get_grid_height()
        );

        for y in 0 .. ac_grid.get_grid_height() {
            for x in 0 .. ac_grid.get_grid_width() {
                let grid = self.ac_grid.get_grid_square(x, y);
                // If the cell has a transform, set it in the DensePhotoMap.
                grid.get_affine_transform().iter().for_each(|cmm| {
                    pm.set_grid_coordinates(
                        cmm.origin_x as usize / self.grid_cell_size,
                        cmm.origin_y as usize / self.grid_cell_size,
                        cmm.translate_x,
                        cmm.translate_y
                    );
                });
            }
        }
        pm
    }

    /// Get photo1 height as u32
    pub fn photo1_height(&self) -> u32 {
        self.photo1.height as u32
    }

    /// Get photo1 width as u32
    pub fn photo1_width(&self) -> u32 {
        self.photo1.width as u32
    }
    
    /// Get grid cell size
    pub fn grid_cell_size(&self) -> usize {
        self.grid_cell_size
    }

    /// Gather all transforms from the ac_grid
    pub fn gather_transforms(&self) -> Vec<AffineTransform> {
        // Instead of accessing a non-existent field, collect transforms from AC grid
        let mut transforms = Vec::new();
        
        // Get transforms from each cell in the grid
        for y in 0..self.ac_grid.get_grid_height() {
            for x in 0..self.ac_grid.get_grid_width() {
                if let Some(transform) = self.ac_grid.get_grid_square(x, y).get_affine_transform() {
                    transforms.push(transform);
                }
            }
        }
        
        transforms
    }

    /// Get grid sample points based on grid_cell_size
    pub fn get_grid_sample_points(&self) -> Vec<(f32, f32)> {
        // Instead of accessing a non-existent field, generate sample points
        let mut points = Vec::new();
        let step = self.grid_cell_size as f32 / 2.0;
        
        // Generate sample points based on photo1 dimensions and grid cell size
        let width = self.photo1.width;
        let height = self.photo1.height;
        
        for y in (0..height).step_by(self.grid_cell_size) {
            for x in (0..width).step_by(self.grid_cell_size) {
                points.push((x as f32 + step, y as f32 + step));
            }
        }
        
        points
    }

    /// Update scores from external systems
    pub fn update_scores(&mut self, scores: &[f32]) {
        // Implementation for updating scores
        // This would update the AC grid with scores
        println!("Updating {} scores", scores.len());
        // TODO: Implement proper score updating
    }

    /// Apply transform updates after score updates
    pub fn update_transforms_from_scores(&mut self) {
        // Implementation for updating transforms from scores
        // This would apply any pending score updates to the transforms
        println!("Updating transforms from scores");
        // TODO: Implement proper transform updating
    }
}

impl Clone for CorrespondenceMappingAlgorithm {
    fn clone(&self) -> Self {
        CorrespondenceMappingAlgorithm {
            photo1: self.photo1.clone(),
            photo2: self.photo2.clone(),
            grid_cell_size: self.grid_cell_size,
            queue: self.queue.clone(),
            scorer: self.scorer.clone(),
            ac_grid: self.ac_grid.clone(),
        }
    }
}
