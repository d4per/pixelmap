use crate::ac_grid::ACGrid;
use crate::affine_transform::AffineTransform;
use crate::correspondence_scoring::CorrespondenceScoring;
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;
use rand::seq::SliceRandom;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Process-wide base seed for the queue shuffling.
///
/// The order in which the queue is drained decides which local optimum the
/// relaxation settles into, so an unseeded run is not reproducible — the final
/// mapping differs measurably between two runs of the same binary. Setting
/// `PIXELMAP_SEED` pins it, which is what makes before/after comparisons of a
/// change to the algorithm meaningful. Unset, the behaviour is as before: seeded
/// from OS entropy.
fn base_seed() -> u64 {
    static BASE: OnceLock<u64> = OnceLock::new();
    *BASE.get_or_init(|| {
        std::env::var("PIXELMAP_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or_else(rand::random)
    })
}

/// Distinguishes the RNGs of the managers created during one run, so the forward
/// and backward passes do not share a stream.
static INSTANCE: AtomicU64 = AtomicU64::new(0);

/// Edge length, in pixels, of the region the queue is grouped by between passes.
///
/// Scoring one transform touches an 11x11 patch of *both* photos. Grouping the queue
/// so that consecutive transforms share most of that patch is what makes the pass
/// cache-resident; at 1600 px the two photos are 5.8 MB each, so an ungrouped pass
/// streams from DRAM.
/// Measured on a 4096x2304 pair at `high`: anything from 6 to 16 performs the same,
/// 4 is worse, and no grouping at all costs about 15%.
const QUEUE_TILE: usize = 10;

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
    photo1: Rc<Photo>,
    /// Reference-counted handle to the second (scaled) photo.
    photo2: Rc<Photo>,
    /// Size of each grid cell in pixels.
    grid_cell_size: usize,
    /// A queue of candidate [AffineTransform] objects to evaluate and refine.
    queue: Vec<AffineTransform>,
    /// Scores transforms by comparing corresponding regions of `photo1` and `photo2`.
    scorer: CorrespondenceScoring,
    /// A 2D grid that stores the best-known transform for each cell (and its score).
    ac_grid: ACGrid,
    /// Shuffles the queue between passes. See [`base_seed`].
    rng: SmallRng,
    /// Reused buffer for the tile-ordering counting sort.
    sort_scratch: Vec<AffineTransform>,
    /// Reused bucket boundaries for the tile-ordering counting sort.
    tile_offsets: Vec<u32>,
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
        Self::with_scaled(photo1a, photo2a, grid_cell_size, neighborhood_radius)
    }

    /// Same as [`Self::new`], but takes photos that have already been scaled to the
    /// working width. Callers that build several algorithms over the same pair of
    /// images can then scale once instead of once per algorithm.
    pub fn with_scaled(
        photo1a: Rc<Photo>,
        photo2a: Rc<Photo>,
        grid_cell_size: usize,
        neighborhood_radius: usize,
    ) -> Self {
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
            rng: SmallRng::seed_from_u64(
                base_seed() ^ INSTANCE
                    .fetch_add(1, Ordering::Relaxed)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15),
            ),
            sort_scratch: Vec::new(),
            tile_offsets: Vec::new(),
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
            // Group transforms by image region, shuffling within each region.
            self.order_queue_by_tile();
            let is_done = self.run_queue();
            if is_done {
                break;
            }
        }
    }

    /// Reorders the queue so that transforms whose origins fall in the same tile of
    /// `photo1` are processed together, and shuffles within each tile.
    ///
    /// The original code shuffled the whole queue before every pass, so consecutive
    /// scorings landed on unrelated parts of two multi-megabyte images. Grouping by
    /// tile keeps a pass working inside a region small enough to stay cached; the
    /// `photo2` side follows because the mapping being refined is smooth. The
    /// within-tile shuffle preserves the reason the original shuffled at all — the
    /// bias it avoids is between transforms competing for nearby cells, which is
    /// local. Tiles are visited in serpentine order so that stepping from one tile to
    /// the next is always a step to an adjacent region.
    ///
    /// This is a counting sort: two linear passes plus a scatter, reusing its buffers
    /// across passes. That is cheaper than the full-width Fisher-Yates it replaces.
    fn order_queue_by_tile(&mut self) {
        let n = self.queue.len();
        if n < 2 {
            return;
        }
        let tile = QUEUE_TILE;
        let tw = self.photo1.width / tile + 1;
        let th = self.photo1.height / tile + 1;
        let n_tiles = tw * th;

        let queue = std::mem::take(&mut self.queue);
        let mut sorted = std::mem::take(&mut self.sort_scratch);
        let mut offsets = std::mem::take(&mut self.tile_offsets);

        let tile_of = |cm: &AffineTransform| -> usize {
            let tx = (cm.origin_x as usize / tile).min(tw - 1);
            let ty = (cm.origin_y as usize / tile).min(th - 1);
            // Serpentine: reverse the x order on odd rows so the last tile of one row
            // is adjacent to the first tile of the next.
            let tx = if ty & 1 == 1 { tw - 1 - tx } else { tx };
            ty * tw + tx
        };

        offsets.clear();
        offsets.resize(n_tiles + 1, 0u32);
        for cm in &queue {
            offsets[tile_of(cm) + 1] += 1;
        }
        for i in 1..=n_tiles {
            offsets[i] += offsets[i - 1];
        }

        sorted.clear();
        sorted.resize(n, queue[0]);
        {
            let mut cursor = offsets.clone();
            for cm in &queue {
                let t = tile_of(cm);
                sorted[cursor[t] as usize] = *cm;
                cursor[t] += 1;
            }
        }

        for w in offsets.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            if b - a > 1 {
                sorted[a..b].shuffle(&mut self.rng);
            }
        }

        self.queue = sorted;
        self.sort_scratch = queue;
        self.tile_offsets = offsets;
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
        let mut out_queue: Vec<AffineTransform> = Vec::with_capacity(self.queue.len());

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
                debug_assert!(false, "grid coordinate {grid_x},{grid_y} out of range");
                continue;
            }

            // Retrieve the current best cell transform/score.
            let grid_square = ac_grid.get_grid_square(grid_x, grid_y);
            if grid_square.get_score() > score {
                // If we've found an improvement, update and extrapolate to neighbors.
                grid_square.set(cm_out, score);

                // Generate child transforms for neighboring cells and push them to
                // out_queue.
                //
                // These extrapolate `cm_out`, the transform that actually won the cell,
                // not the `cm` that was popped: propagating the pre-optimization
                // transform threw away the +-1 pixel correction `optimize_position` had
                // just found, so every neighbour had to rediscover it. Measured over
                // seven seeds on a 1170x893 pair at `medium`, propagating the optimized
                // transform raises the matched area from 0.384 to 0.408 for about 14%
                // more time, the extra time being spent converging to the larger result.
                if grid_x > 0 {
                    out_queue.push(
                        cm_out.extrapolate_mapping(((grid_x - 1) * self.grid_cell_size) as u16,
                                                   (grid_y * self.grid_cell_size) as u16)
                    );
                }
                if grid_x < ac_grid.get_grid_width() - 1 {
                    out_queue.push(
                        cm_out.extrapolate_mapping(((grid_x + 1) * self.grid_cell_size) as u16,
                                                   (grid_y * self.grid_cell_size) as u16)
                    );
                }
                if grid_y > 0 {
                    out_queue.push(
                        cm_out.extrapolate_mapping((grid_x * self.grid_cell_size) as u16,
                                                   ((grid_y - 1) * self.grid_cell_size) as u16)
                    );
                }
                if grid_y < ac_grid.get_grid_height() - 1 {
                    out_queue.push(
                        cm_out.extrapolate_mapping((grid_x * self.grid_cell_size) as u16,
                                                   ((grid_y + 1) * self.grid_cell_size) as u16)
                    );
                }
            }
        }

        // Replace the main queue with out_queue for the next iteration.
        self.queue = out_queue;
        // If it's empty, the algorithm is done (no further improvements).
        self.queue.is_empty()
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
    ///
    /// All five candidates share one circular neighbourhood and differ only by a
    /// constant offset into `photo2`, so the scorer resolves that neighbourhood once
    /// and evaluates the five offsets over it — see
    /// [`CorrespondenceScoring::optimize_translation`].
    pub fn optimize_position(&self, cm: &AffineTransform) -> (f32, AffineTransform) {
        self.scorer.optimize_translation(cm)
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
                        (cmm.origin_x as usize / self.grid_cell_size),
                        (cmm.origin_y as usize / self.grid_cell_size),
                        cmm.translate_x,
                        cmm.translate_y
                    );
                });
            }
        }
        pm
    }
}
