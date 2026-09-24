use crate::circular_feature_descriptor::CircularFeatureDescriptor;
#[cfg(any(test, feature = "bench"))]
use crate::circular_feature_grid::CircularFeatureGrid;
use crate::circular_feature_grid::DescriptorRows;
use crate::kdtree::{KdTree, Point};
use crate::photo::Photo;
#[cfg(any(test, feature = "bench"))]
use std::time::Duration;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// How many of image 2's descriptors are queried, by default: every `n`-th one.
///
/// Each match becomes one init point that is snapped into a `grid_cell_size` cell
/// (5x5 by default) and pushed on the correspondence queue, so a stride of 4 still
/// leaves roughly six candidates per cell. Skipping the rest costs no measurable
/// mapping quality and is close to a linear time saving in both the search and the
/// queue that consumes it.
pub const DEFAULT_MATCH_STRIDE: usize = 4;

/// Which nearest-neighbour strategy [`CircularFeatureDescriptorMatcher`] should use.
///
/// [`MatcherBackend::BruteForce`] is the ground truth the others are measured against;
/// see `crate::matcher_bench`.
#[cfg(any(test, feature = "bench"))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(not(feature = "bench"), allow(dead_code))]
pub(crate) enum MatcherBackend {
    /// Exhaustive scan: for every queried descriptor, compare against every descriptor of
    /// image 1. Quadratic and far too slow for real use, which is exactly why it makes a
    /// good baseline — it cannot be wrong, so any disagreement is the other backend's.
    BruteForce,
    /// The kd-tree, querying every `stride`-th descriptor of image 2. `parallel` is
    /// ignored unless the `parallel` feature is enabled, so a wasm build silently and
    /// correctly falls back to a single thread.
    KdTree { stride: usize, parallel: bool },
}

/// A clock that only runs under the `bench` feature.
///
/// `Instant::now()` compiles for `wasm32-unknown-unknown` and then *panics* — std maps
/// that target's time module to `unsupported/time.rs`, whose body is
/// `panic!("time not implemented on this platform")`. Timing the match on the main path
/// therefore took the whole pipeline down in a browser. The measurements only ever feed
/// `matcher_bench`, so outside that they are compiled away to nothing.
#[cfg(feature = "bench")]
#[derive(Clone, Copy)]
struct Stopwatch(std::time::Instant);

#[cfg(feature = "bench")]
impl Stopwatch {
    fn start() -> Self {
        Stopwatch(std::time::Instant::now())
    }
    fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

#[cfg(all(test, not(feature = "bench")))]
#[derive(Clone, Copy)]
struct Stopwatch;

#[cfg(all(test, not(feature = "bench")))]
impl Stopwatch {
    fn start() -> Self {
        Stopwatch
    }
    fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

/// How long a backend spent building its index versus querying it.
///
/// The index is built once per `match_areas` call and thrown away, so build time is
/// fully on the critical path and has to be reported alongside query time. Zero unless
/// the `bench` feature is on; see [`Stopwatch`].
#[cfg(any(test, feature = "bench"))]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(feature = "bench"), allow(dead_code))]
pub(crate) struct MatchTiming {
    pub build: Duration,
    pub query: Duration,
}

/// True when this build can actually use more than one thread.
///
/// Parallelism is a compile-time property, not a caller's choice: on
/// `wasm32-unknown-unknown` rayon has no threads to hand out, so the `parallel` feature
/// is switched off for that target and every request to run in parallel degrades to a
/// serial run rather than failing.
#[cfg(any(test, feature = "bench"))]
pub const fn parallelism_available() -> bool {
    cfg!(feature = "parallel")
}

/// Provides a way to match circular feature descriptors between two images.
///
/// For each queried descriptor of image 2 this finds the nearest descriptor of image 1
/// in the 6-D quantized feature space, so that similar neighbourhoods across the two
/// images can be paired up.
pub struct CircularFeatureDescriptorMatcher;

/// The materialized path: both descriptor grids built in full, then searched. The
/// pipeline runs [`Self::match_photos`] instead; this is kept as its reference, and for
/// `matcher_bench`, which compares backends over the same grids.
#[cfg(any(test, feature = "bench"))]
#[allow(dead_code)] // The tests and `matcher_bench` each use a different subset.
impl CircularFeatureDescriptorMatcher {
    pub fn new() -> Self {
        CircularFeatureDescriptorMatcher {}
    }

    /// Matches `img2`'s descriptors against `img1`'s, using [`DEFAULT_MATCH_STRIDE`].
    pub fn match_areas(
        &self,
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
    ) -> Vec<FeatureMatch> {
        self.match_areas_with_stride(img1, img2, DEFAULT_MATCH_STRIDE)
    }

    /// [`Self::match_areas`] with an explicit stride: only every `stride`-th descriptor
    /// of `img2` is queried. `1` matches every descriptor, as the original did.
    ///
    /// Only the two centres and the angle between the descriptors are ever used by the
    /// caller, so the result is a compact [`FeatureMatch`] (12 bytes) rather than a copy
    /// of both 96-byte descriptors.
    /// Overrides the stride from `PIXELMAP_MATCH_STRIDE`, when set. Lets the whole
    /// pipeline be run end to end at a given stride without threading a parameter
    /// through every caller; unset, it is [`DEFAULT_MATCH_STRIDE`].
    #[cfg(feature = "bench")]
    fn stride_override() -> Option<usize> {
        std::env::var("PIXELMAP_MATCH_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
    }

    pub fn match_areas_with_stride(
        &self,
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
        stride: usize,
    ) -> Vec<FeatureMatch> {
        #[cfg(feature = "bench")]
        let stride = Self::stride_override().unwrap_or(stride);
        let backend = MatcherBackend::KdTree {
            stride: stride.max(1),
            parallel: true,
        };
        self.match_areas_timed(img1, img2, backend).0
    }

    /// [`Self::match_areas_with_stride`], also reporting how the time split between
    /// building the index and querying it. Used by `matcher_bench`.
    pub(crate) fn match_areas_timed(
        &self,
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
        backend: MatcherBackend,
    ) -> (Vec<FeatureMatch>, MatchTiming) {
        match backend {
            MatcherBackend::BruteForce => Self::brute_force(img1, img2),
            MatcherBackend::KdTree { stride, parallel } => {
                Self::kdtree_search(img1, img2, stride.max(1), parallel)
            }
        }
    }

    /// Ground truth for `matcher_bench`: every query against every candidate.
    ///
    /// Ties break by lowest image-1 index, matching [`crate::kdtree`], so the two agree
    /// exactly rather than merely equivalently.
    fn brute_force(
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
    ) -> (Vec<FeatureMatch>, MatchTiming) {
        let (infos1, infos2) = (img1.get_infos(), img2.get_infos());

        let query_at = |i: usize| -> Option<FeatureMatch> {
            let cai2 = &infos2[i];
            let mut best: Option<(u64, usize)> = None;
            for (j, cai1) in infos1.iter().enumerate() {
                let mut sum = 0i64;
                for k in 0..6 {
                    let d = cai1.feature_vector[k] as i64 - cai2.feature_vector[k] as i64;
                    sum += d * d;
                }
                let d2 = sum as u64;
                if best.map_or(true, |(bd, _)| d2 < bd) {
                    best = Some((d2, j));
                }
            }
            best.map(|(_, j)| FeatureMatch::new(&infos1[j], cai2))
        };

        let t = Stopwatch::start();
        let ans = par_query(infos2.len(), 1, &query_at);
        (
            ans,
            MatchTiming {
                build: Duration::ZERO,
                query: t.elapsed(),
            },
        )
    }

    /// Indexes image 1 into a [`crate::kdtree`] and queries every `stride`-th descriptor
    /// of image 2 against it.
    ///
    /// No key-range guard and no widening pass: the tree accumulates in `i64`, so the
    /// descriptor's `i16` key is indexed exactly as it is stored.
    fn kdtree_search(
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
        stride: usize,
        parallel: bool,
    ) -> (Vec<FeatureMatch>, MatchTiming) {
        let (infos1, infos2) = (img1.get_infos(), img2.get_infos());
        let parallel = parallel && parallelism_available();

        let t0 = Stopwatch::start();
        let points = infos1
            .iter()
            .enumerate()
            .map(|(i, d)| Point {
                v: d.feature_vector,
                id: i as u32,
            })
            .collect();
        let tree = if parallel {
            KdTree::build(points)
        } else {
            KdTree::build_serial(points)
        };
        let build = t0.elapsed();

        // One query, shared by the serial and parallel drivers so they cannot drift.
        let query_at = |i: usize| -> Option<FeatureMatch> {
            let cai2 = &infos2[i];
            tree.nearest(&cai2.feature_vector)
                .map(|found| FeatureMatch::new(&infos1[found.id as usize], cai2))
        };

        let t1 = Stopwatch::start();
        let ans = if parallel {
            par_query(infos2.len(), stride, &query_at)
        } else {
            (0..infos2.len())
                .step_by(stride)
                .filter_map(query_at)
                .collect()
        };
        (
            ans,
            MatchTiming {
                build,
                query: t1.elapsed(),
            },
        )
    }
}

impl CircularFeatureDescriptorMatcher {
    /// What the pipeline runs: [`Self::match_areas_with_stride`] over two photos'
    /// descriptors at `circle_radius`, without ever holding either descriptor grid.
    ///
    /// The two grids used to be built in full first — 20 bytes per pixel each, ~58 MB for
    /// the pair at a 1600 px working width — although the search needs far less:
    ///
    /// - Image 1 goes into the tree, which keeps only a [`Point`] per pixel. The rest of a
    ///   descriptor is its centre, which is recoverable from the point's id, and its angle,
    ///   which is kept in a side table of 4 bytes per pixel.
    /// - Image 2 is only ever queried, and only at every `stride`-th pixel. Its rows are
    ///   computed, queried and dropped one at a time.
    ///
    /// The result is identical to the materialized path's, element for element and in the
    /// same order: the same flat indices of image 2 are queried against the same set of
    /// keys and ids. A test pins that.
    pub(crate) fn match_photos(
        photo1: &Photo,
        photo2: &Photo,
        circle_radius: usize,
        stride: usize,
    ) -> Vec<FeatureMatch> {
        #[cfg(feature = "bench")]
        let stride = Self::stride_override().unwrap_or(stride);
        let stride = stride.max(1);

        let rows1 = DescriptorRows::new(photo1, circle_radius);
        let w1 = rows1.width();
        let n1 = w1 * rows1.height();
        let mut points = vec![Point { v: [0; 6], id: 0 }; n1];
        let mut angles = vec![0f32; n1];
        let index_row = |y: usize, points: &mut [Point], angles: &mut [f32], buf: &mut Vec<_>| {
            rows1.fill_row(y, buf);
            for (x, d) in buf.iter().enumerate() {
                points[x] = Point {
                    v: d.feature_vector,
                    id: (y * w1 + x) as u32,
                };
                angles[x] = d.total_angle;
            }
        };
        let row_buf = |w: usize| vec![CircularFeatureDescriptor::default(); w];
        if w1 > 0 {
            #[cfg(feature = "parallel")]
            points
                .par_chunks_mut(w1)
                .zip(angles.par_chunks_mut(w1))
                .enumerate()
                .for_each_init(|| row_buf(w1), |buf, (y, (p, a))| index_row(y, p, a, buf));
            #[cfg(not(feature = "parallel"))]
            {
                let mut buf = row_buf(w1);
                for (y, (p, a)) in points
                    .chunks_exact_mut(w1)
                    .zip(angles.chunks_exact_mut(w1))
                    .enumerate()
                {
                    index_row(y, p, a, &mut buf);
                }
            }
        }
        let tree = KdTree::build(points);

        let rows2 = DescriptorRows::new(photo2, circle_radius);
        let w2 = rows2.width();
        // The materialized path queried flat indices `0, stride, 2*stride, ...`; this
        // visits the same ones row by row, in the same order.
        let query_row = |y: usize, buf: &mut Vec<CircularFeatureDescriptor>| -> Vec<FeatureMatch> {
            let row_start = y * w2;
            let first = (stride - row_start % stride) % stride;
            if first >= w2 {
                return Vec::new();
            }
            rows2.fill_row(y, buf);
            (first..w2)
                .step_by(stride)
                .filter_map(|x| {
                    let cai2 = &buf[x];
                    tree.nearest(&cai2.feature_vector).map(|found| {
                        let id = found.id as usize;
                        FeatureMatch {
                            x1: (id % w1) as u16,
                            y1: (id / w1) as u16,
                            x2: cai2.center_x,
                            y2: cai2.center_y,
                            angle_delta: angles[id] - cai2.total_angle,
                        }
                    })
                })
                .collect()
        };
        if w2 == 0 {
            return Vec::new();
        }
        #[cfg(feature = "parallel")]
        let per_row: Vec<Vec<FeatureMatch>> = (0..rows2.height())
            .into_par_iter()
            .map_init(|| row_buf(w2), |buf, y| query_row(y, buf))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let per_row: Vec<Vec<FeatureMatch>> = {
            let mut buf = row_buf(w2);
            (0..rows2.height())
                .map(|y| query_row(y, &mut buf))
                .collect()
        };
        per_row.concat()
    }
}

/// `par_iter().step_by().filter_map().collect()` preserves source order, so a parallel
/// run returns exactly the same vector as a serial one rather than merely an equivalent
/// set. That is what lets the wasm (serial) and native (parallel) builds agree.
#[cfg(all(any(test, feature = "bench"), feature = "parallel"))]
fn par_query<F>(len: usize, stride: usize, query_at: &F) -> Vec<FeatureMatch>
where
    F: Fn(usize) -> Option<FeatureMatch> + Sync,
{
    (0..len)
        .into_par_iter()
        .step_by(stride)
        .filter_map(query_at)
        .collect()
}

#[cfg(all(any(test, feature = "bench"), not(feature = "parallel")))]
fn par_query<F>(len: usize, stride: usize, query_at: &F) -> Vec<FeatureMatch>
where
    F: Fn(usize) -> Option<FeatureMatch>,
{
    (0..len).step_by(stride).filter_map(query_at).collect()
}

/// One matched pair of circular features: a centre in each image, and the rotation
/// between the two descriptors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureMatch {
    pub x1: u16,
    pub y1: u16,
    pub x2: u16,
    pub y2: u16,
    pub angle_delta: f32,
}

#[cfg(any(test, feature = "bench"))]
impl FeatureMatch {
    #[inline]
    fn new(cai1: &CircularFeatureDescriptor, cai2: &CircularFeatureDescriptor) -> Self {
        FeatureMatch {
            x1: cai1.center_x,
            y1: cai1.center_y,
            x2: cai2.center_x,
            y2: cai2.center_y,
            angle_delta: cai1.total_angle - cai2.total_angle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Noise over a smooth gradient, with a flat block in it. The flat block yields one
    /// key over and over, which is the case the tree's de-duplication exists for.
    fn photo(width: usize, height: usize, seed: u64) -> Photo {
        let mut rng = Rng::seed_from_u64(seed);
        let mut img_data = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                let flat = x < width / 3 && y < height / 2;
                let n = rng.next_u64();
                let c = |shift: u32, base: usize| {
                    if flat {
                        90
                    } else {
                        ((base + (n >> shift) as usize % 64) % 256) as u8
                    }
                };
                img_data.extend_from_slice(&[c(0, 3 * x), c(8, 2 * y), c(16, x + y), 255]);
            }
        }
        Photo {
            img_data,
            width,
            height,
        }
    }

    /// The streamed matcher has to give exactly what matching the two materialized grids
    /// gives: same pairs, same order. Widths that are and are not multiples of the stride
    /// cover the per-row offset of the first queried pixel.
    #[test]
    fn streamed_matches_equal_materialized_matches() {
        for &(w, h) in &[(37usize, 29usize), (64, 48), (50, 33)] {
            let (p1, p2) = (
                photo(w, h, 0x5EED ^ w as u64),
                photo(w, h, 0xFACE ^ h as u64),
            );
            let g1 = CircularFeatureGrid::new(&p1, w, h, 10);
            let g2 = CircularFeatureGrid::new(&p2, w, h, 10);
            for stride in [1usize, 3, 4, 7] {
                let expected = CircularFeatureDescriptorMatcher::new()
                    .match_areas_with_stride(&g1, &g2, stride);
                let got = CircularFeatureDescriptorMatcher::match_photos(&p1, &p2, 10, stride);
                assert!(!got.is_empty());
                assert_eq!(got, expected, "{w}x{h} at stride {stride}");
            }
        }
    }
}
