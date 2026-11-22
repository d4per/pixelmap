//! The pipeline driver behind [`crate::Correspondence`].

use crate::local_consistency::LocalConsistency;
use crate::circular_feature_descriptor_matcher::{
    CircularFeatureDescriptorMatcher, DEFAULT_MATCH_STRIDE,
};
use crate::correspondence_mapping_algorithm::CorrespondenceMappingAlgorithm;
use crate::dense_photo_map::DensePhotoMap;
use crate::photo::Photo;
use crate::processing_mode::{IterationParams, MAX_NEIGHBORHOOD_RADIUS};
use std::collections::{HashMap, VecDeque};
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
    /// The two full-resolution images to be matched, until [`Self::prepare`] has
    /// scaled everything the run will need from them and let them go.
    sources: Option<(Arc<Photo>, Arc<Photo>)>,

    /// Width of the full-resolution images, which outlives them.
    source_width: usize,

    /// The raw forward (`photo1` → `photo2`) and backward maps the last solve ended
    /// with, before any outlier removal. `None` until [`Self::init`] has run.
    ///
    /// Only the maps are kept, not the solvers that produced them. A solver holds per-cell
    /// state and queue buffers worth about 60 MB at 3200 px, and keeping the previous
    /// step's pair alive while the next one ran made two pairs coexist: that was the peak
    /// of a `Quality::Ultra` run.
    maps: Option<(DensePhotoMap, DensePhotoMap)>,

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

    /// The working widths still to come, in order, the init solve's first: set by
    /// [`Self::prepare`] from the schedule, and `None` for a caller that runs without
    /// one. Each step drops the entries of `scaled` that no later step will ask for.
    widths_ahead: Option<VecDeque<usize>>,

    /// The disc radius of the init solve, the one seeded from the feature matches; see
    /// [`Self::set_init_neighborhood_radius`].
    init_radius: usize,

    /// How [`Self::finish`] fills the cells the final map left empty; see
    /// [`FillMode`].
    fill_mode: FillMode,

    /// The raw forward and backward maps of the last step at each earlier working width,
    /// most recent last. Only collected when `fill_mode` is not [`FillMode::Off`].
    history: Vec<(DensePhotoMap, DensePhotoMap)>,

    /// The filter [`Self::finish`] applies last, if any; see [`LocalConsistency`].
    local_consistency: Option<LocalConsistency>,

    /// Whether [`Self::iterate`] has run, so that the init solve's map, which is far
    /// rougher than anything after it, is not kept in `history`.
    iterated: bool,
}

/// How [`PixelMapProcessor::finish`] treats the cells the final map left empty.
///
/// Every schedule step starts a fresh solver from the previous step's cleaned map, so a
/// region that loses the round-trip check at a fine working width is gone even if a
/// coarser step had mapped it consistently. Filling takes such cells from the last map
/// of each earlier width instead, finest first.
///
/// The filled cells were mapped consistently at a coarser width but not confirmed at the
/// final one, so they are less precise than the rest: filling trades end-point accuracy
/// for coverage. [`crate::Quality::Low`] leaves it off; the higher presets use
/// [`FillMode::Plain`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum FillMode {
    /// Leave empty cells empty.
    #[default]
    Off,
    /// Take an empty cell from the most recent earlier map that answers there.
    Plain,
    /// As `Plain`, then run the round trip again on the filled cells against the filled
    /// maps of the other direction, with the threshold the earlier map was cleaned with.
    Checked,
}

impl PixelMapProcessor {
    /// Constructs a new `PixelMapProcessor` with the given `photo1`, `photo2`, and an `initial_photo_width`.
    ///
    /// The width is used to scale images down (if needed) before feature extraction. Until
    /// [`Self::init`] runs, the processor holds empty maps.
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
        PixelMapProcessor {
            source_width: photo1.width,
            sources: Some((photo1, photo2)),
            maps: None,
            total_comparisons: 0,
            initial_photo_width: photo_width,
            seed,
            algorithms_built: 0,
            scaled: HashMap::new(),
            widths_ahead: None,
            init_radius: 5,
            fill_mode: FillMode::Off,
            history: Vec::new(),
            local_consistency: None,
            iterated: false,
        }
    }

    /// Sets how [`Self::finish`] fills empty cells. Defaults to [`FillMode::Off`]; the
    /// presets' choice is [`crate::Quality::fill_mode`].
    ///
    /// Call it before the first [`Self::iterate`]: the maps it fills from are collected
    /// as the iterations run.
    pub fn set_fill_mode(&mut self, mode: FillMode) {
        self.fill_mode = mode;
    }

    /// Sets the filter [`Self::finish`] applies to both maps once they are cleaned and
    /// filled, or `None` (the default) for none.
    pub fn set_local_consistency(&mut self, filter: Option<LocalConsistency>) {
        self.local_consistency = filter;
    }

    /// Sets the disc radius of the solve [`Self::init`] runs from the feature matches.
    /// Defaults to 5. Each [`Self::iterate`] takes its own radius.
    ///
    /// Call it before [`Self::init`]. See
    /// [`Builder::init_neighborhood_radius`](crate::Builder::init_neighborhood_radius).
    ///
    /// # Panics
    /// If `radius` is 0 or above [`MAX_NEIGHBORHOOD_RADIUS`].
    pub fn set_init_neighborhood_radius(&mut self, radius: usize) {
        assert!(
            (1..=MAX_NEIGHBORHOOD_RADIUS).contains(&radius),
            "neighborhood radius must be in 1..={MAX_NEIGHBORHOOD_RADIUS}, not {radius}"
        );
        self.init_radius = radius;
    }

    /// The width of the full-resolution photos, which [`crate::Quality::steps_for`]
    /// takes to fit a schedule to them.
    pub fn source_width(&self) -> usize {
        self.source_width
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
    ///
    /// # Panics
    /// If the width was not scaled before [`Self::prepare`] released the sources. Only
    /// [`crate::Builder`] calls `prepare`, with the whole schedule it is about to run.
    fn scaled(&mut self, which: usize, width: usize) -> Arc<Photo> {
        if let Some(p) = self.scaled.get(&(which, width)) {
            return p.clone();
        }
        let (photo1, photo2) = self
            .sources
            .as_ref()
            .expect("every working width is scaled before the sources are released");
        let src = if which == 0 { photo1 } else { photo2 };
        let p = Arc::new(src.scaled_to_width(width));
        self.scaled.insert((which, width), p.clone());
        p
    }

    /// The width [`Self::init`] works at.
    fn init_width(&self) -> usize {
        usize::min(self.initial_photo_width, self.source_width)
    }

    /// Scales the photos to every width [`Self::init`] and `steps` will ask for, then
    /// drops the full-resolution originals.
    ///
    /// Those are by far the largest buffers of a run — 75 MB for a 4096x2304 pair — and
    /// nothing reads them once each working width has been made, so holding them for the
    /// whole run only raised the peak. They are freed only if this processor held the
    /// last reference to them.
    pub(crate) fn prepare(&mut self, steps: &[IterationParams]) {
        let widths: VecDeque<usize> = std::iter::once(self.init_width())
            .chain(steps.iter().map(|s| s.photo_width))
            .collect();
        // One photo at a time, each let go as soon as its own widths are made, so the
        // second source is the only one alive while the second photo's widths are scaled.
        if let Some((photo1, photo2)) = self.sources.take() {
            for (which, source) in [(0, photo1), (1, photo2)] {
                for &width in &widths {
                    self.scaled
                        .entry((which, width))
                        .or_insert_with(|| Arc::new(source.scaled_to_width(width)));
                }
            }
        }
        self.widths_ahead = Some(widths);
    }

    /// Called once a step's solvers hold their photos: forgets the scaled photos that no
    /// later step will ask for. The solvers keep what they hold alive until they finish.
    fn step_started(&mut self) {
        let Some(ahead) = self.widths_ahead.as_mut() else {
            return;
        };
        ahead.pop_front();
        self.scaled.retain(|&(_, width), _| ahead.contains(&width));
    }

    /// Performs the initial matching step:
    /// 1. Scales both `photo1` and `photo2` to `initial_photo_width` (if needed).
    /// 2. Extracts circular feature descriptors and matches them with
    ///    `CircularFeatureDescriptorMatcher`.
    /// 3. Initializes new `CorrespondenceMappingAlgorithm` instances with the matched points.
    /// 4. Runs both correspondence managers until completion.
    ///
    /// Upon completion, `total_comparisons` is updated with the sum of both managers' comparisons.
    pub fn init(&mut self) {
        // Scale down images if needed.
        let width = self.init_width();
        let photo1scaled = self.scaled(0, width);
        let photo2scaled = self.scaled(1, width);

        // Extract circular feature descriptors and match them across the two images.
        let pairs = CircularFeatureDescriptorMatcher::match_photos(
            &photo1scaled,
            &photo2scaled,
            10,
            DEFAULT_MATCH_STRIDE,
        );

        // Create new managers for the scaled images.
        let w = photo1scaled.width;
        let (s1, s2) = (self.scaled(0, w), self.scaled(1, w));
        let (seed1, seed2) = (self.next_seed(), self.next_seed());
        let r = self.init_radius;
        let mut ocm_manager1 =
            CorrespondenceMappingAlgorithm::with_scaled(s1.clone(), s2.clone(), 5, r, seed1);
        let mut ocm_manager2 = CorrespondenceMappingAlgorithm::with_scaled(s2, s1, 5, r, seed2);

        // Add the initial matched points.
        for m in pairs {
            let vv = -m.angle_delta;

            // Add forward mapping: (photo1 → photo2)
            ocm_manager1.add_init_point(m.x1 as f32, m.y1 as f32, m.x2 as f32, m.y2 as f32, vv);

            // Add reverse mapping: (photo2 → photo1)
            ocm_manager2.add_init_point(m.x2 as f32, m.y2 as f32, m.x1 as f32, m.y1 as f32, -vv);
        }

        self.step_started();
        self.total_comparisons = 0;
        self.solve(ocm_manager1, ocm_manager2);
    }

    /// Returns the total number of comparisons made so far by the two correspondence managers.
    pub fn total_comparisons(&self) -> usize {
        self.total_comparisons
    }

    /// Performs an **iteration** of the mapping refinement process:
    ///
    /// 1. Takes the maps the previous solve ended with.
    /// 2. Removes outliers in each map by checking consistency with the other map.
    /// 3. Smooths (averages) each map over several iterations to reduce noise.
    /// 4. Builds a new pair of solvers from the smoothed maps.
    /// 5. Runs both to completion and keeps their maps.
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
        clean_max_dist: f32,
    ) {
        let (mut pm1, mut pm2) = self.last_maps();

        // Keep the last map of each earlier width for `finish` to fill holes from.
        if self.fill_mode != FillMode::Off && self.iterated && pm1.dimensions().0 != photo_width {
            self.history.push((pm1.clone(), pm2.clone()));
        }
        self.iterated = true;

        // Remove outliers by forward-backward consistency check.

        pm1.remove_outliers(&pm2, clean_max_dist);
        pm2.remove_outliers(&pm1, clean_max_dist);

        // Smooth the remaining mapping.
        let (pm1_smooth, pm2_smooth) = join(
            || pm1.smooth_grid_points_n_times(smooth_iterations),
            || pm2.smooth_grid_points_n_times(smooth_iterations),
        );

        // Re-initialize managers with the smoothed maps.
        let (s1, s2) = (self.scaled(0, photo_width), self.scaled(1, photo_width));
        let (seed1, seed2) = (self.next_seed(), self.next_seed());
        let mut ocm_manager1 = CorrespondenceMappingAlgorithm::with_scaled(
            s1.clone(),
            s2.clone(),
            grid_cell_size,
            neighborhood_radius,
            seed1,
        );
        let mut ocm_manager2 = CorrespondenceMappingAlgorithm::with_scaled(
            s2,
            s1,
            grid_cell_size,
            neighborhood_radius,
            seed2,
        );
        ocm_manager1.init_from_photomapping(&pm1_smooth);
        ocm_manager2.init_from_photomapping(&pm2_smooth);

        drop((pm1, pm2, pm1_smooth, pm2_smooth));
        self.step_started();
        self.solve(ocm_manager1, ocm_manager2);
    }

    /// Copies of [`Self::maps`].
    ///
    /// # Panics
    /// If [`Self::init`] has not run yet.
    fn last_maps(&self) -> (DensePhotoMap, DensePhotoMap) {
        self.maps
            .clone()
            .expect("PixelMapProcessor::init must run before the maps are used")
    }

    /// Runs both solvers to completion, counts their comparisons, and keeps their maps.
    /// The solvers themselves are dropped here; see [`Self::maps`].
    fn solve(
        &mut self,
        mut forward: CorrespondenceMappingAlgorithm,
        mut backward: CorrespondenceMappingAlgorithm,
    ) {
        run_both(&mut forward, &mut backward);
        self.total_comparisons += forward.total_comparisons() + backward.total_comparisons();
        self.maps = Some((forward.get_photo_mapping(), backward.get_photo_mapping()));
    }

    /// Retrieves the final forward and backward mappings, removing any remaining outliers
    /// with a given distance threshold.
    ///
    /// It leaves the processor as it was: it can be called after every [`Self::iterate`]
    /// to look at the mapping so far, and a later call still fills holes from every
    /// earlier working width.
    ///
    /// # Parameters
    /// - `clean_max_dist`: The maximum distance for outlier checking.
    ///   Points whose forward-backward mapping is too large are discarded.
    ///
    /// # Returns
    /// A tuple of two [DensePhotoMap]s:
    /// - First: from `photo1` to `photo2`.
    /// - Second: from `photo2` to `photo1`.
    pub fn finish(&mut self, clean_max_dist: f32) -> (DensePhotoMap, DensePhotoMap) {
        let (mut pm1, mut pm2) = self.last_maps();

        // Remove outliers in both directions.
        pm1.remove_outliers(&pm2, clean_max_dist);
        pm2.remove_outliers(&pm1, clean_max_dist);

        if self.fill_mode != FillMode::Off && !self.history.is_empty() {
            fill_from_history(
                &mut pm1,
                &mut pm2,
                &self.history,
                clean_max_dist,
                self.fill_mode == FillMode::Checked,
            );
        }

        // Last, so that every answer handed out passed it, filled ones included.
        if let Some(filter) = &self.local_consistency {
            filter.apply(&mut pm1);
            filter.apply(&mut pm2);
        }

        (pm1, pm2)
    }

    /// Computes how much of the forward mapping is valid (non-outlier),
    /// by performing a small outlier check with threshold `2.0`.
    ///
    /// # Returns
    /// The fraction of valid cells in the forward map, a value between 0.0 and 1.0.
    pub fn matched_area(&self) -> f32 {
        // Cleaning copies leaves the stored maps alone.
        let (mut pm1, mut pm2) = self.last_maps();

        pm1.remove_outliers(&pm2, 2.0);
        pm2.remove_outliers(&pm1, 2.0);

        pm1.calculate_used_area()
    }
}

/// Fills the empty cells of the final maps `fwd`/`bwd` from `history`, the raw maps of
/// earlier working widths (oldest first); see [`FillMode`].
///
/// Each earlier pair is cleaned with `max_dist` in *its own* grid first, the check it
/// would have had to pass had the schedule ended there. An empty cell then takes its
/// value from the most recent pair that answers at its position. With `checked`, the
/// filled cells must also survive a round trip through the filled map of the other
/// direction, with `max_dist` scaled to the grid of the map the fill came from.
fn fill_from_history(
    fwd: &mut DensePhotoMap,
    bwd: &mut DensePhotoMap,
    history: &[(DensePhotoMap, DensePhotoMap)],
    max_dist: f32,
    checked: bool,
) {
    let width = fwd.dimensions().0 as f32;
    let earlier: Vec<(DensePhotoMap, DensePhotoMap, f32)> = history
        .iter()
        .rev()
        .map(|(a, b)| {
            let (mut a, mut b) = (a.clone(), b.clone());
            a.remove_outliers(&b, max_dist);
            b.remove_outliers(&a, max_dist);
            let scale = a.dimensions().0 as f32 / width;
            (a, b, scale)
        })
        .collect();

    // For each cell, the index into `earlier` its value was filled from.
    let fill = |map: &mut DensePhotoMap,
                pick: fn(&(DensePhotoMap, DensePhotoMap, f32)) -> &DensePhotoMap| {
        let (gw, gh) = map.grid_dimensions();
        let cs = map.grid_cell_size() as f32;
        let mut source = vec![None; gw * gh];
        for y in 0..gh {
            for x in 0..gw {
                if !map.grid_coordinates(x, y).0.is_nan() {
                    continue;
                }
                let (px, py) = (x as f32 * cs, y as f32 * cs);
                for (i, entry) in earlier.iter().enumerate() {
                    let scale = entry.2;
                    if let Some((qx, qy)) = pick(entry).lookup(px * scale, py * scale) {
                        map.set_grid_coordinates(x, y, qx / scale, qy / scale);
                        source[x + y * gw] = Some(i);
                        break;
                    }
                }
            }
        }
        source
    };
    let fwd_source = fill(fwd, |e| &e.0);
    let bwd_source = fill(bwd, |e| &e.1);

    if checked {
        recheck(fwd, bwd, &fwd_source, &earlier, max_dist);
        recheck(bwd, fwd, &bwd_source, &earlier, max_dist);
    }
}

/// The round trip of [`DensePhotoMap::remove_outliers`], on the filled cells of `map`
/// only, each with `max_dist` scaled from the grid it was filled from to this one.
fn recheck(
    map: &mut DensePhotoMap,
    other: &DensePhotoMap,
    source: &[Option<usize>],
    earlier: &[(DensePhotoMap, DensePhotoMap, f32)],
    max_dist: f32,
) {
    let (gw, gh) = map.grid_dimensions();
    let cs = map.grid_cell_size() as f32;
    for y in 0..gh {
        for x in 0..gw {
            let Some(i) = source[x + y * gw] else {
                continue;
            };
            let limit = max_dist / earlier[i].2;
            let (mx, my) = map.grid_coordinates(x, y);
            let (bx, by) = other.map_photo_pixel(mx, my);
            let (dx, dy) = (x as f32 - bx / cs, y as f32 - by / cs);
            if dx.is_nan() || dy.is_nan() || dx * dx + dy * dy > limit * limit {
                map.set_grid_coordinates(x, y, f32::NAN, f32::NAN);
            }
        }
    }
}

/// Runs the forward and the backward solver to completion, side by side when the
/// `parallel` feature is on.
///
/// The two share nothing mutable — each owns its grid, its queue and its RNG stream, and
/// the photos they read are immutable — so running them at once changes only when the
/// work happens, not what it computes. The mapping is bit-identical to the serial order.
fn run_both(a: &mut CorrespondenceMappingAlgorithm, b: &mut CorrespondenceMappingAlgorithm) {
    join(|| a.run_until_done(), || b.run_until_done());
}

/// `rayon::join` under the `parallel` feature, and the two closures one after the other
/// without it — the configuration a `wasm32-unknown-unknown` build uses.
#[cfg(feature = "parallel")]
fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    rayon::join(a, b)
}

#[cfg(not(feature = "parallel"))]
fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB,
{
    (a(), b())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank(width: usize, height: usize) -> Arc<Photo> {
        Arc::new(Photo {
            img_data: vec![0; width * height * 4],
            width,
            height,
        })
    }

    /// A map over a `width`-wide photo that sends `(x, y)` to `(x + shift, y)`.
    fn shifted_map(width: usize, height: usize, shift: f32) -> DensePhotoMap {
        let (gw, gh) = (width / 5, height / 5);
        let mut map =
            DensePhotoMap::with_cell_size(blank(width, height), blank(width, height), gw, gh, 5);
        for y in 0..gh {
            for x in 0..gw {
                map.set_grid_coordinates(x, y, (x * 5) as f32 + shift, (y * 5) as f32);
            }
        }
        map
    }

    #[test]
    fn filling_only_touches_empty_cells_and_scales_the_earlier_map() {
        let mut fwd = shifted_map(100, 60, 0.0);
        let mut bwd = shifted_map(100, 60, 0.0);
        fwd.set_grid_coordinates(3, 3, 99.0, 99.0); // a value filling must not overwrite
        fwd.set_grid_coordinates(10, 6, f32::NAN, f32::NAN);
        let history = vec![(shifted_map(50, 30, 0.0), shifted_map(50, 30, 0.0))];

        fill_from_history(&mut fwd, &mut bwd, &history, 2.0, false);

        assert_eq!(fwd.grid_coordinates(3, 3), (99.0, 99.0));
        assert_eq!(fwd.grid_coordinates(10, 6), (50.0, 30.0));
    }

    #[test]
    fn a_checked_fill_must_survive_the_other_directions_final_map() {
        let hole = |map: &mut DensePhotoMap| map.set_grid_coordinates(10, 6, f32::NAN, f32::NAN);
        // The earlier pair agrees with itself on a 15 px (at width 50) shift; the final
        // backward map says the photo does not move at all.
        let history = || vec![(shifted_map(50, 30, 15.0), shifted_map(50, 30, -15.0))];

        let (mut fwd, mut bwd) = (shifted_map(100, 60, 0.0), shifted_map(100, 60, 0.0));
        hole(&mut fwd);
        fill_from_history(&mut fwd, &mut bwd, &history(), 2.0, false);
        assert_eq!(
            fwd.grid_coordinates(10, 6),
            (80.0, 30.0),
            "a plain fill keeps it"
        );

        let (mut fwd, mut bwd) = (shifted_map(100, 60, 0.0), shifted_map(100, 60, 0.0));
        hole(&mut fwd);
        fill_from_history(&mut fwd, &mut bwd, &history(), 2.0, true);
        assert!(
            fwd.grid_coordinates(10, 6).0.is_nan(),
            "the round trip is 6 cells off"
        );
        assert_eq!(
            fwd.grid_coordinates(9, 6),
            (45.0, 30.0),
            "unfilled cells stay"
        );
    }
}
