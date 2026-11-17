use crate::ac_grid::ACGrid;
use crate::affine_transform::AffineTransform;
use crate::correspondence_scoring::{CorrespondenceScoring, FixedTransform, PackedPhoto};
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;
use crate::rng::Rng;
use std::sync::Arc;

/// How far a cell's score has to drop over one pass before the cell passes its transform
/// on to its neighbours, as a fraction of the score it began the pass with.
///
/// A cell that was empty when the pass began always passes its transform on; this only
/// governs cells that already held one. Replacing a transform by one that scores a
/// fraction of a percent lower is still worth doing for the cell itself, but handing that
/// near-identical transform to four neighbours to score again is where the relaxation's
/// long tail came from.
///
/// Measured on two sets of three real photos (3 to 8 seeds each, judged by how well the
/// three pairwise mappings close the loop A -> B -> C -> A) and on rendered scenes with
/// an exact ground truth: at 0.05 the result stays inside the spread the solver shows
/// from one seed to the next, for about ten percentage points fewer evaluations. At 0.10
/// the loop error on the harder set is visibly worse. On the hardest pairs at
/// `Quality::Low` 0.05 already sits at the weak end of that spread, so this is not a
/// number to raise casually.
const REPRODUCE_GAIN: f32 = 0.05;

/// Whether a cell that began a pass at `score_at_start` and ended it at `score_now`
/// passes its transform on. Only called for cells whose transform changed in the pass.
fn reproduces(score_at_start: f32, score_now: f32) -> bool {
    // `f32::MAX` is the score of an empty cell: whatever filled it is news to the
    // neighbours however well or badly it fits.
    score_at_start == f32::MAX || score_now < score_at_start * (1.0 - REPRODUCE_GAIN)
}

/// The candidates a cell has already evaluated that are worth remembering.
///
/// Scoring is a pure function of a candidate's [`FixedTransform`], a cell's best score
/// only ever falls, and a candidate has to score strictly lower to be accepted. A
/// candidate equal to one the cell has evaluated before would therefore score what that
/// one scored and be rejected, so it can be dropped unscored without changing anything.
///
/// Two are enough to catch the common case. When a cell improves and hands its transform
/// to a neighbour that adopts it unshifted, the neighbour hands the very same transform
/// straight back.
#[derive(Clone, Copy, Default)]
struct Seen {
    /// The candidate whose evaluation produced the cell's current transform.
    accepted: Option<FixedTransform>,
    /// The candidate the cell rejected most recently.
    rejected: Option<FixedTransform>,
}

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
/// 3. **Refinement**: Transforms are slightly adjusted, and a cell that ends a pass with a
///    clearly better transform than it began with propagates it to its neighbours,
///    allowing the algorithm to converge over multiple passes.
/// 4. **Final Mapping**: Once no more improvements occur, you can retrieve a [DensePhotoMap]
///    describing the best transforms found for each cell.
pub struct CorrespondenceMappingAlgorithm {
    /// Reference-counted handle to the first (scaled) photo.
    photo1: Arc<Photo>,
    /// Reference-counted handle to the second (scaled) photo.
    photo2: Arc<Photo>,
    /// Size of each grid cell in pixels.
    grid_cell_size: usize,
    /// A queue of candidate [AffineTransform] objects to evaluate and refine.
    queue: Vec<AffineTransform>,
    /// Scores transforms by comparing corresponding regions of `photo1` and `photo2`.
    scorer: CorrespondenceScoring,
    /// A 2D grid that stores the best-known transform for each cell (and its score).
    ac_grid: ACGrid,
    /// Shuffles the queue between passes.
    ///
    /// Within a pass a cell ends up with the best-scoring candidate it was offered
    /// whatever the order, so the order only decides between candidates that score
    /// exactly the same. That happens often — many candidates resolve to the very same
    /// pixels — and which of them a cell keeps decides what its neighbours are handed
    /// next, so the stream has to be reproducible from the caller's seed for two runs of
    /// the same input to agree.
    rng: Rng,
    /// Reused buffer for the tile-ordering counting sort.
    sort_scratch: Vec<AffineTransform>,
    /// Reused bucket boundaries for the tile-ordering counting sort.
    tile_offsets: Vec<u32>,
    /// Per cell, the candidates it has already evaluated; see [`Seen`].
    seen: Vec<Seen>,
    /// The cells whose transform changed during the pass in progress, in the order they
    /// first did, each with the score it held when the pass began.
    improved: Vec<(u32, f32)>,
    /// Whether a cell is already in `improved`.
    is_improved: Vec<bool>,
    /// Whether candidates a cell has already evaluated are dropped unscored. Always on;
    /// the tests turn it off to check that it changes the work and nothing else.
    skip_seen: bool,
}

impl CorrespondenceMappingAlgorithm {
    /// Creates a new mapping algorithm for the given images and parameters.
    ///
    /// # Parameters
    /// - `photo_width`: The width to which both photos are scaled (preserves aspect ratio).
    /// - `photo1`, `photo2`: References to the original images.
    /// - `grid_cell_size`: Size of each cell in the grid (in pixels).
    /// - `neighborhood_radius`: Radius (in pixels) for the circular neighborhood scoring.
    /// - `seed`: Seeds the queue shuffling. The same seed on the same input reproduces
    ///   the same mapping.
    ///
    /// # Returns
    /// A new `CorrespondenceMappingAlgorithm` with scaled images, an empty queue, and
    /// an initialized scoring mechanism and grid.
    pub fn new(
        photo_width: usize,
        photo1: &Photo,
        photo2: &Photo,
        grid_cell_size: usize,
        neighborhood_radius: usize,
        seed: u64,
    ) -> Self {
        // Scale the original photos to the specified width.
        let photo1a = PackedPhoto::new(Arc::new(photo1.scaled_to_width(photo_width)));
        let photo2a = PackedPhoto::new(Arc::new(photo2.scaled_to_width(photo_width)));
        Self::with_scaled(photo1a, photo2a, grid_cell_size, neighborhood_radius, seed)
    }

    /// Same as [`Self::new`], but takes photos that have already been scaled to the
    /// working width. Callers that build several algorithms over the same pair of
    /// images can then scale and pack once instead of once per algorithm.
    pub(crate) fn with_scaled(
        packed1: PackedPhoto,
        packed2: PackedPhoto,
        grid_cell_size: usize,
        neighborhood_radius: usize,
        seed: u64,
    ) -> Self {
        // Determine how many cells fit in the scaled images. The last cell's origin is
        // `(grid_width - 1) * grid_cell_size`, so the count is derived from the last
        // *pixel* index: `width / grid_cell_size + 1` puts that origin at `width`, one
        // column past the image, where `resolve_disc` finds no pixels to score and the
        // cell can never be filled. That dead column and row then took their neighbours
        // with them through `remove_outliers`, and counted against `coverage()` besides.
        let (photo1a, photo2a) = (packed1.photo.clone(), packed2.photo.clone());
        let grid_width = photo1a.width.saturating_sub(1) / grid_cell_size + 1;
        let grid_height = photo1a.height.saturating_sub(1) / grid_cell_size + 1;

        CorrespondenceMappingAlgorithm {
            photo1: photo1a,
            photo2: photo2a,
            grid_cell_size,
            queue: vec![],
            scorer: CorrespondenceScoring::with_packed(
                packed1,
                packed2,
                neighborhood_radius as isize,
            ),
            ac_grid: ACGrid::new(grid_width, grid_height),
            rng: Rng::seed_from_u64(seed),
            sort_scratch: Vec::new(),
            tile_offsets: Vec::new(),
            seen: vec![Seen::default(); grid_width * grid_height],
            improved: Vec::new(),
            is_improved: vec![false; grid_width * grid_height],
            skip_seen: true,
        }
    }

    /// Returns the total number of scoring function invocations
    /// performed so far (for diagnostic or debugging purposes).
    pub fn total_comparisons(&self) -> usize {
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
                self.rng.shuffle(&mut sorted[a..b]);
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
            (y1 as usize / self.grid_cell_size * self.grid_cell_size) as u16,
        );
        self.queue.push(snap_to_grid_cm);
    }

    /// Initializes the queue with transforms derived from an existing [DensePhotoMap].
    /// This allows the algorithm to continue refining a previously computed mapping.
    ///
    /// # Parameters
    /// - `pm`: A [`DensePhotoMap`] containing approximate mappings from `photo1` to `photo2`.
    pub fn init_from_photomapping(&mut self, pm: &DensePhotoMap) {
        let pm_grid_cell_size = pm.grid_cell_size();

        // For each cell in the DensePhotoMap, create an AffineTransform
        // and push it into the queue for refinement.
        for y in 0..pm.grid_height {
            for x in 0..pm.grid_width {
                let (x2a, y2a) = pm.grid_coordinates(x, y);
                if x2a.is_nan() {
                    continue;
                }
                // Compute the origin in the scaled photo1.
                let x1 = f32::round(
                    ((x * pm_grid_cell_size * self.photo1.width) as f32) / pm.photo1.width as f32,
                ) as usize;
                let y1 = f32::round(
                    ((y * pm_grid_cell_size * self.photo1.width) as f32) / pm.photo1.width as f32,
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
                    let (left_x, left_y) = pm.grid_coordinates(x - 1, y);
                    if !left_x.is_nan() {
                        cm.a11 = (x2a - left_x) / pm_grid_cell_size as f32;
                        cm.a21 = (y2a - left_y) / pm_grid_cell_size as f32;
                    }
                }
                if y > 0 {
                    let (up_x, up_y) = pm.grid_coordinates(x, y - 1);
                    if !up_x.is_nan() {
                        cm.a22 = (y2a - up_y) / pm_grid_cell_size as f32;
                        cm.a12 = (x2a - up_x) / pm_grid_cell_size as f32;
                    }
                }

                // Snap the transform to the nearest grid coordinates.
                let snap_to_grid_x = f32::round(x1 as f32 / self.grid_cell_size as f32) as usize
                    * self.grid_cell_size;
                let snap_to_grid_y = f32::round(y1 as f32 / self.grid_cell_size as f32) as usize
                    * self.grid_cell_size;
                let snap_to_grid_cm =
                    cm.extrapolate_mapping(snap_to_grid_x as u16, snap_to_grid_y as u16);
                self.queue.push(snap_to_grid_cm);
            }
        }
    }

    /// Runs one pass: scores every queued transform against its grid cell, then lets the
    /// cells that came out of the pass clearly better hand their transform to their
    /// neighbours, which is the queue of the next pass.
    ///
    /// For each transform:
    /// 1. Validates its scale and position (no out-of-bounds).
    /// 2. Drops it unscored if its cell has evaluated the same transform before.
    /// 3. Uses `optimize_position` to adjust it.
    /// 4. Stores it in its grid cell if it scores lower than what the cell holds.
    ///
    /// # Who reproduces
    ///
    /// A cell hands on a transform **once per pass, after the pass, from whatever it
    /// holds then** — not at every improvement along the way. It used to queue four
    /// children the moment a transform was accepted, so a transform displaced a moment
    /// later by a better candidate for the same cell had already reproduced, and its
    /// children went on to be scored, sometimes accepted, and to reproduce in turn.
    /// Measured on a 4128x3096 pair at `high`, about half of all evaluations were
    /// children of a transform that left no descendant in the final grid, and a quarter
    /// of all accepted transforms were replaced within the pass that accepted them.
    ///
    /// On top of that, a cell that already held a transform when the pass began hands
    /// its new one on only if the score dropped by more than [`REPRODUCE_GAIN`].
    ///
    /// Together with dropping transforms a cell has already evaluated (see [`Seen`]),
    /// this takes about 40% of the evaluations away. Both the numbers and the checks
    /// that the result did not suffer are described at [`REPRODUCE_GAIN`].
    ///
    /// What was deliberately *not* cut is the spreading of poorly fitting transforms
    /// into cells that have nothing better. That looks like the larger waste — on a
    /// hard pair two thirds of all evaluations land in cells the round-trip check then
    /// discards — but it is how the solver finds the regions it does keep. Limiting how
    /// far a transform may spread through unmapped cells, letting a transform spread
    /// only into worse-fitting cells, and refusing to reproduce from a poorly fitting
    /// entry into an empty cell were all tried, and each cost coverage or accuracy on
    /// the harder of the two photo sets.
    ///
    /// Returns `true` if the queue is empty afterward (no improvements), or
    /// `false` if there are still transforms to process in the next pass.
    fn run_queue(&mut self) -> bool {
        let grid_width = self.ac_grid.get_grid_width();
        let grid_height = self.ac_grid.get_grid_height();

        while let Some(cm) = self.queue.pop() {
            // Skip if the transform is out of scale or invalid in terms of translation.
            if !cm.is_scale_valid(4.0) || !self.is_valid(&cm) {
                continue;
            }

            // Determine which grid cell the transform belongs to. Optimizing it moves
            // where it lands in `photo2`, never its origin, so this can be decided up
            // front.
            //
            // A transform can sit one cell past the grid: `init_from_photomapping`
            // rounds to the nearest grid node, and when a step works at a lower width
            // than the map it starts from, the map's last column or row rounds up to a
            // node at the photo's far edge. There is no cell there to compete for.
            let grid_x = cm.origin_x as usize / self.grid_cell_size;
            let grid_y = cm.origin_y as usize / self.grid_cell_size;
            if grid_x >= grid_width || grid_y >= grid_height {
                continue;
            }
            let cell = grid_x + grid_y * grid_width;

            // A transform this cell has already evaluated would score the same again and
            // lose; see `Seen`.
            let key = FixedTransform::new(&cm);
            let seen = self.seen[cell];
            if self.skip_seen && (seen.accepted == Some(key) || seen.rejected == Some(key)) {
                continue;
            }

            // Attempt a small local optimization on (cm).
            let (score, cm_out) = self.optimize_position(&cm);

            // Compare with the current best cell transform/score.
            let grid_square = self.ac_grid.get_grid_square(grid_x, grid_y);
            let score_before = grid_square.get_score();
            if score_before > score {
                grid_square.set(cm_out, score);
                self.seen[cell].accepted = Some(key);
                // Remember that the cell changed, and what it had to beat when the pass
                // began. It reproduces after the pass, from whatever it holds by then.
                if !self.is_improved[cell] {
                    self.is_improved[cell] = true;
                    self.improved.push((cell as u32, score_before));
                }
            } else {
                self.seen[cell].rejected = Some(key);
            }
        }

        // Generate child transforms for the neighbours of every cell that came out of
        // the pass clearly better than it went in.
        //
        // These extrapolate the transform the cell holds, which is the *optimized* one,
        // not the candidate that was popped: propagating the pre-optimization transform
        // threw away the +-1 pixel correction `optimize_position` had just found, so
        // every neighbour had to rediscover it. Measured over seven seeds on a 1170x893
        // pair at `medium`, propagating the optimized transform raises the matched area
        // from 0.384 to 0.408 for about 14% more time, the extra time being spent
        // converging to the larger result.
        let mut improved = std::mem::take(&mut self.improved);
        let mut out_queue: Vec<AffineTransform> = Vec::with_capacity(4 * improved.len());
        let cell_size = self.grid_cell_size;
        for &(cell, score_at_start) in &improved {
            let cell = cell as usize;
            self.is_improved[cell] = false;
            let (grid_x, grid_y) = (cell % grid_width, cell / grid_width);
            let grid_square = self.ac_grid.get_grid_square(grid_x, grid_y);
            if !reproduces(score_at_start, grid_square.get_score()) {
                continue;
            }
            let Some(cm_out) = grid_square.get_affine_transform() else {
                continue;
            };
            let mut child = |x: usize, y: usize| {
                out_queue.push(
                    cm_out.extrapolate_mapping((x * cell_size) as u16, (y * cell_size) as u16),
                );
            };
            if grid_x > 0 {
                child(grid_x - 1, grid_y);
            }
            if grid_x < grid_width - 1 {
                child(grid_x + 1, grid_y);
            }
            if grid_y > 0 {
                child(grid_x, grid_y - 1);
            }
            if grid_y < grid_height - 1 {
                child(grid_x, grid_y + 1);
            }
        }
        improved.clear();
        self.improved = improved;

        // Replace the main queue with out_queue for the next pass.
        self.queue = out_queue;
        // If it's empty, the algorithm is done (no further improvements).
        self.queue.is_empty()
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
        // The solver laid this grid out, so it passes the spacing rather than letting
        // `DensePhotoMap::new` divide it back out of the photo width — which only
        // recovers it when the width is an exact multiple of the cell size.
        let mut pm = DensePhotoMap::with_cell_size(
            self.photo1.clone(),
            self.photo2.clone(),
            ac_grid.get_grid_width(),
            ac_grid.get_grid_height(),
            self.grid_cell_size,
        );

        for y in 0..ac_grid.get_grid_height() {
            for x in 0..ac_grid.get_grid_width() {
                let grid = self.ac_grid.get_grid_square(x, y);
                // If the cell has a transform, set it in the DensePhotoMap.
                grid.get_affine_transform().iter().for_each(|cmm| {
                    pm.set_grid_coordinates(
                        cmm.origin_x as usize / self.grid_cell_size,
                        cmm.origin_y as usize / self.grid_cell_size,
                        cmm.translate_x,
                        cmm.translate_y,
                    );
                });
            }
        }
        pm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 120;
    const HEIGHT: usize = 90;
    const SHIFT: usize = 4;

    /// Two windows onto one noisy texture, the second `SHIFT` pixels further along both
    /// axes, so what the first shows at `(x, y)` sits at `(x - SHIFT, y - SHIFT)` in the
    /// second.
    ///
    /// The second window gets a little noise of its own, as two real photos would have,
    /// so that even the true mapping does not score a perfect zero.
    fn shifted_pair() -> (Photo, Photo) {
        let (bw, bh) = (WIDTH + SHIFT, HEIGHT + SHIFT);
        let mut rng = Rng::seed_from_u64(0x7E57_0FF5_9121);
        let mut base = Vec::with_capacity(bw * bh * 4);
        for y in 0..bh {
            for x in 0..bw {
                let noise = (rng.next_u64() % 70) as f32 - 35.0;
                let wave = 127.0 + 100.0 * (x as f32 / 9.0).sin() * (y as f32 / 11.0).cos() + noise;
                let v = wave.clamp(0.0, 255.0) as u8;
                base.extend_from_slice(&[v, (v / 2).wrapping_add(40), 255 - v, 255]);
            }
        }
        let window = |ox: usize, oy: usize| {
            let mut img_data = Vec::with_capacity(WIDTH * HEIGHT * 4);
            for y in 0..HEIGHT {
                let row = (y + oy) * bw + ox;
                img_data.extend_from_slice(&base[row * 4..(row + WIDTH) * 4]);
            }
            Photo {
                img_data,
                width: WIDTH,
                height: HEIGHT,
            }
        };
        let (first, mut second) = (window(0, 0), window(SHIFT, SHIFT));
        for pixel in second.img_data.chunks_exact_mut(4) {
            for channel in &mut pixel[..3] {
                let jitter = (rng.next_u64() % 7) as i32 - 3;
                *channel = (*channel as i32 + jitter).clamp(0, 255) as u8;
            }
        }
        (first, second)
    }

    fn solver(seed: u64) -> CorrespondenceMappingAlgorithm {
        let (a, b) = shifted_pair();
        CorrespondenceMappingAlgorithm::new(WIDTH, &a, &b, 5, 5, seed)
    }

    /// A solver seeded on a coarse lattice with the true shift and, next to each, a
    /// wrong guess: something to spread, something to compete with it, and something
    /// to lose.
    fn seeded_solver(seed: u64) -> CorrespondenceMappingAlgorithm {
        let mut solver = solver(seed);
        let shift = SHIFT as f32;
        for y in (10..HEIGHT - 10).step_by(15) {
            for x in (10..WIDTH - 10).step_by(15) {
                let (x, y) = (x as f32, y as f32);
                solver.add_init_point(x, y, x - shift, y - shift, 0.0);
                solver.add_init_point(x, y, x + 7.0, y - 9.0, 0.3);
            }
        }
        solver
    }

    /// Every cell's transform and score, for comparing two solvers exactly.
    fn grid_state(solver: &CorrespondenceMappingAlgorithm) -> Vec<(Option<AffineTransform>, u32)> {
        let grid = &solver.ac_grid;
        (0..grid.get_grid_height())
            .flat_map(|y| (0..grid.get_grid_width()).map(move |x| (x, y)))
            .map(|(x, y)| {
                let square = grid.get_grid_square(x, y);
                (square.get_affine_transform(), square.get_score().to_bits())
            })
            .collect()
    }

    /// Dropping a candidate a cell has already evaluated is justified by an argument, not
    /// by a measurement: it would have been rejected. So switching the drop off has to
    /// leave every cell with exactly the same transform and score, and only cost more.
    #[test]
    fn dropping_seen_candidates_changes_the_work_and_nothing_else() {
        let mut dropping = seeded_solver(11);
        let mut scoring_everything = seeded_solver(11);
        scoring_everything.skip_seen = false;

        dropping.run_until_done();
        scoring_everything.run_until_done();

        assert!(
            grid_state(&dropping) == grid_state(&scoring_everything),
            "dropping seen candidates changed the grid"
        );
        assert!(
            dropping.total_comparisons() < scoring_everything.total_comparisons(),
            "nothing was dropped: {} comparisons either way",
            dropping.total_comparisons()
        );
        let coverage = dropping.get_photo_mapping().calculate_used_area();
        assert!(
            coverage > 0.9,
            "the solve only filled {coverage:.2} of the grid"
        );
    }

    /// Three candidates for one cell, popped worst first, so the cell is improved more
    /// than once within the pass. Each improvement used to queue four children of its
    /// own; now the cell reproduces once, from the transform it ends the pass with.
    #[test]
    fn a_cell_reproduces_once_per_pass_from_what_it_ends_up_holding() {
        let mut solver = solver(3);
        let (at, landing) = (40.0, 40.0 - SHIFT as f32);
        // The queue is popped from the back, so the true shift goes in first.
        for (dx, dy) in [(0.0, 0.0), (3.0, 0.0), (5.0, -6.0)] {
            solver.add_init_point(at, at, landing + dx, landing + dy, 0.0);
        }

        let done = solver.run_queue();

        assert!(!done, "an improved cell has neighbours to tell");
        assert_eq!(solver.total_comparisons(), 3 * 5, "all three were scored");
        let best = solver
            .ac_grid
            .get_grid_square(8, 8)
            .get_affine_transform()
            .expect("the cell was filled");
        assert_eq!(
            (best.translate_x, best.translate_y),
            (landing, landing),
            "the true shift, scored last, should have displaced the other two"
        );
        assert_eq!(
            solver.queue.len(),
            4,
            "one cell changed, so exactly its four neighbours are queued"
        );
        for child in &solver.queue {
            assert_eq!(
                *child,
                best.extrapolate_mapping(child.origin_x, child.origin_y),
                "a child that does not descend from the cell's final transform"
            );
        }
    }

    #[test]
    fn only_a_clear_improvement_is_passed_on() {
        assert!(
            reproduces(f32::MAX, 5000.0),
            "whatever fills an empty cell is news to its neighbours"
        );
        assert!(reproduces(100.0, 94.0));
        assert!(!reproduces(100.0, 96.0));
        assert!(!reproduces(100.0, 100.0));
    }

    /// The cause of the straight unmapped bands. A transform that points a cell at the very
    /// edge of the other photo with a compressed matrix keeps a single sample of its disc.
    /// The variance of one sample is zero, the best score there is, so it used to beat the
    /// genuine transform of any cell it was offered to — and since moving along the edge
    /// changes nothing about it, it took the cell above and below as well, and so on down
    /// the whole column. The round-trip check then deleted that column.
    ///
    /// The numbers are the ones found in a real run: every cell of one column of an
    /// 800 px map held `a11 = 0.328, a21 = 0.569`, a target 0.2 px short of the right
    /// edge, a score of exactly 0.0, and 1 sample of 81.
    #[test]
    fn a_transform_that_barely_touches_the_other_photo_cannot_take_over_a_column() {
        let mut solver = seeded_solver(21);
        solver.run_until_done();
        let before = grid_state(&solver);
        let coverage = solver.get_photo_mapping().calculate_used_area();
        assert!(
            coverage > 0.9,
            "the solve only filled {coverage:.2} of the grid"
        );

        let (grid_x, grid_y) = (12usize, 9usize);
        let genuine = solver.ac_grid.get_grid_square(grid_x, grid_y).get_score();
        assert!(
            genuine > 0.0,
            "the true mapping should not score a perfect zero"
        );
        solver.queue.push(AffineTransform {
            origin_x: (grid_x * 5) as u16,
            origin_y: (grid_y * 5) as u16,
            translate_x: WIDTH as f32 - 0.15,
            translate_y: (grid_y * 5) as f32,
            a11: 0.328,
            a12: 0.0,
            a21: 0.569,
            a22: 1.0,
        });
        solver.run_until_done();

        let after = grid_state(&solver);
        let overwritten = before.iter().zip(&after).filter(|(b, a)| b != a).count();
        assert_eq!(
            overwritten, 0,
            "a transform comparing almost nothing overwrote {overwritten} cells"
        );
    }

    /// The same candidate offered to a cell twice is scored once.
    #[test]
    fn a_repeated_candidate_is_not_scored_again() {
        let mut solver = solver(5);
        let (at, landing) = (40.0, 40.0 - SHIFT as f32);
        for _ in 0..3 {
            solver.add_init_point(at, at, landing + 3.0, landing, 0.0);
        }
        solver.run_queue();
        assert_eq!(solver.total_comparisons(), 5);
    }
}
