use crate::circular_feature_descriptor::CircularFeatureDescriptor;
use crate::circular_feature_grid::CircularFeatureGrid;
use kd_tree::{KdPoint, KdTree};
use std::time::Duration;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Retained so that [`MatcherBackend::KdTreeLegacy`] can index the descriptor directly,
/// exactly as the original implementation did. Every other backend uses [`FeaturePoint`].
///
/// The keys are `i16` in the descriptor; widening them to `i64` here keeps this backend
/// bit-for-bit the baseline it was, whatever the storage type underneath.
impl KdPoint for CircularFeatureDescriptor {
    type Scalar = i64;
    type Dim = typenum::U6;
    fn at(&self, k: usize) -> i64 {
        self.feature_vector[k] as i64
    }
}

/// The 6-D search key of a descriptor, packed next to the index of the descriptor
/// it came from.
///
/// This was introduced when [`CircularFeatureDescriptor`] was 96 bytes and carried a
/// pile of intermediate values the tree never looked at: indexing 28 bytes instead cut
/// the bytes the build's median partition has to move by about 3.4x, worth roughly 1.4x
/// on its own. The descriptor has since been trimmed to 20 bytes — everything the
/// matcher reads and nothing else — so `FeaturePoint` is now the *larger* of the two and
/// no longer pays for itself on size. What it still provides is the `i32` scalar the
/// tree needs (the keys themselves are `i16`, whose squared distance would overflow),
/// and a second point type for `matcher_bench` to measure the backends against.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FeaturePoint {
    key: [i32; 6],
    idx: u32,
}

impl KdPoint for FeaturePoint {
    type Scalar = i32;
    type Dim = typenum::U6;
    fn at(&self, k: usize) -> i32 {
        self.key[k]
    }
}

/// Largest coordinate magnitude for which a 6-axis squared distance still fits in `i32`.
///
/// `6 * (2 * KEY_LIMIT)^2 < i32::MAX` holds for `KEY_LIMIT = 9000`. The real values are
/// far smaller: an aligned centre of mass cannot exceed the disc radius (10), and the
/// feature vector scales it by 100, so |key| stays around 1000.
const KEY_LIMIT: i64 = 9_000;

/// How many of image 2's descriptors are queried, by default: every `n`-th one.
///
/// Each match becomes one init point that is snapped into a `grid_cell_size` cell
/// (5x5 by default) and pushed on the correspondence queue, so a stride of 4 still
/// leaves roughly six candidates per cell. Skipping the rest costs no measurable
/// mapping quality and is close to a linear time saving in both the search and the
/// queue that consumes it.
pub const DEFAULT_MATCH_STRIDE: usize = 4;

impl FeaturePoint {
    #[inline]
    fn new(idx: usize, descriptor: &CircularFeatureDescriptor) -> Self {
        let mut key = [0i32; 6];
        for (k, slot) in key.iter_mut().enumerate() {
            *slot = descriptor.feature_vector[k] as i32;
        }
        FeaturePoint { key, idx: idx as u32 }
    }

    /// Checks the whole descriptor set against [`KEY_LIMIT`] once, in release too.
    ///
    /// An out-of-range key would not fail loudly — it would silently truncate to `i32`
    /// and overflow the squared distance, returning wrong neighbours. One O(n) pass of
    /// comparisons is far too cheap to skip next to the O(n log n) tree build it guards.
    fn assert_key_range(infos: &[CircularFeatureDescriptor]) {
        let worst = infos
            .iter()
            .flat_map(|d| d.feature_vector.iter())
            .fold(0i64, |acc, &v| acc.max((v as i64).abs()));
        assert!(
            worst <= KEY_LIMIT,
            "feature vector magnitude {worst} exceeds the i32 squared-distance budget of {KEY_LIMIT}"
        );
    }

    fn build_all(infos: &[CircularFeatureDescriptor]) -> Vec<FeaturePoint> {
        infos.iter().enumerate().map(|(i, d)| FeaturePoint::new(i, d)).collect()
    }
}

/// Which nearest-neighbour strategy [`CircularFeatureDescriptorMatcher`] should use.
///
/// [`MatcherBackend::KdTreeLegacy`] is kept verbatim as the correctness and performance
/// baseline the others are measured against; see `crate::matcher_bench`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MatcherBackend {
    /// Exact, serial, indexing the descriptor itself with `i64` keys.
    KdTreeLegacy,
    /// Indexes the compact [`FeaturePoint`], querying every `stride`-th descriptor of
    /// image 2. `parallel` is ignored unless the `parallel` feature is enabled, so a
    /// wasm build silently and correctly falls back to a single thread.
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

#[cfg(not(feature = "bench"))]
#[derive(Clone, Copy)]
struct Stopwatch;

#[cfg(not(feature = "bench"))]
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
#[derive(Clone, Copy, Debug, Default)]
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
pub const fn parallelism_available() -> bool {
    cfg!(feature = "parallel")
}

/// Provides a way to match circular feature descriptors between two images.
///
/// For each queried descriptor of image 2 this finds the nearest descriptor of image 1
/// in the 6-D quantized feature space, so that similar neighbourhoods across the two
/// images can be paired up.
pub struct CircularFeatureDescriptorMatcher;

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
        std::env::var("PIXELMAP_MATCH_STRIDE").ok().and_then(|v| v.parse().ok())
    }

    pub fn match_areas_with_stride(
        &self,
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
        stride: usize,
    ) -> Vec<FeatureMatch> {
        #[cfg(feature = "bench")]
        let stride = Self::stride_override().unwrap_or(stride);
        let backend = MatcherBackend::KdTree { stride: stride.max(1), parallel: true };
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
            MatcherBackend::KdTreeLegacy => Self::kdtree_legacy(img1, img2),
            MatcherBackend::KdTree { stride, parallel } => {
                // The legacy backend keeps the original i64 keys and needs no check.
                FeaturePoint::assert_key_range(img1.get_infos());
                FeaturePoint::assert_key_range(img2.get_infos());
                Self::kdtree_search(img1, img2, stride.max(1), parallel)
            }
        }
    }

    /// The original implementation: clone every descriptor into an owned kd-tree and
    /// query it serially. Retained as the baseline the other backends are compared to.
    fn kdtree_legacy(
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
    ) -> (Vec<FeatureMatch>, MatchTiming) {
        let t0 = Stopwatch::start();
        let kdtree = KdTree::build(img1.get_infos().clone());
        let build = t0.elapsed();

        let t1 = Stopwatch::start();
        let infos2 = img2.get_infos();
        let mut ans: Vec<FeatureMatch> = Vec::with_capacity(infos2.len());
        for cai2 in infos2 {
            if let Some(found) = kdtree.nearest(cai2) {
                ans.push(FeatureMatch::new(found.item, cai2));
            }
        }
        (ans, MatchTiming { build, query: t1.elapsed() })
    }

    fn kdtree_search(
        img1: &CircularFeatureGrid,
        img2: &CircularFeatureGrid,
        stride: usize,
        parallel: bool,
    ) -> (Vec<FeatureMatch>, MatchTiming) {
        let (infos1, infos2) = (img1.get_infos(), img2.get_infos());
        let parallel = parallel && parallelism_available();

        let t0 = Stopwatch::start();
        let points = FeaturePoint::build_all(infos1);
        let kdtree = if parallel { par_build(points) } else { KdTree::build(points) };
        let build = t0.elapsed();

        // One query, shared by the serial and parallel drivers so they cannot drift.
        let query_at = |i: usize| -> Option<FeatureMatch> {
            let cai2 = &infos2[i];
            let query = FeaturePoint::new(i, cai2);
            kdtree
                .nearest(&query)
                .map(|found| FeatureMatch::new(&infos1[found.item.idx as usize], cai2))
        };

        let t1 = Stopwatch::start();
        let ans = if parallel {
            par_query(infos2.len(), stride, &query_at)
        } else {
            (0..infos2.len()).step_by(stride).filter_map(query_at).collect()
        };
        (ans, MatchTiming { build, query: t1.elapsed() })
    }
}

#[cfg(feature = "parallel")]
fn par_build(points: Vec<FeaturePoint>) -> KdTree<FeaturePoint> {
    KdTree::par_build(points)
}

#[cfg(not(feature = "parallel"))]
fn par_build(points: Vec<FeaturePoint>) -> KdTree<FeaturePoint> {
    KdTree::build(points)
}

/// `par_iter().step_by().filter_map().collect()` preserves source order, so a parallel
/// run returns exactly the same vector as a serial one rather than merely an equivalent
/// set. That is what lets the wasm (serial) and native (parallel) builds agree.
#[cfg(feature = "parallel")]
fn par_query<F>(len: usize, stride: usize, query_at: &F) -> Vec<FeatureMatch>
where
    F: Fn(usize) -> Option<FeatureMatch> + Sync,
{
    (0..len).into_par_iter().step_by(stride).filter_map(query_at).collect()
}

#[cfg(not(feature = "parallel"))]
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
