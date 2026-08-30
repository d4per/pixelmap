//! Head-to-head harness for the nearest-neighbour backends behind
//! [`crate::circular_feature_descriptor_matcher`].
//!
//! This exists to settle two questions with measurements rather than argument: whether
//! the `kd_tree` index is the right structure for this workload (it is — an HNSW vector
//! index was measured and lost badly at every scale, because 6 dimensions is the regime
//! kd-trees are best at), and how far the match stride can be pushed before mapping
//! quality suffers.
//!
//! The workload is unusual for a vector index: `CircularFeatureGrid` emits one descriptor
//! **per pixel**, so at the `high` working width of 1600 there are ~1.4M points *and*
//! ~1.4M queries — but in only **6 dimensions**, over integer-quantized keys. Both halves
//! matter, so every backend is reported with its build time and its query time separated,
//! and with how far its answers drift from the exact baseline.
//!
//! Serial rows are measured on equal terms with the parallel ones, because a
//! `wasm32-unknown-unknown` build has no threads and runs the serial path.
//!
//! Compiled only under the `bench` feature.

use std::time::Duration;

use crate::circular_feature_descriptor::CircularFeatureDescriptor;
use crate::circular_feature_descriptor_matcher::{
    parallelism_available, CircularFeatureDescriptorMatcher, FeatureMatch, MatcherBackend,
};
use crate::circular_feature_grid::CircularFeatureGrid;
use crate::photo::Photo;

/// The disc radius the pipeline uses for the initial match (see `PixelMapProcessor::init`).
const CIRCLE_RADIUS: usize = 10;

/// What one backend cost, and how much its answers differ from the exact baseline.
#[derive(Clone, Debug)]
pub struct BackendReport {
    pub name: String,
    pub build: Duration,
    pub query: Duration,
    /// Resident-set growth across the build, in bytes. A coarse but honest figure: it is
    /// the whole process, so it also catches allocator behaviour the index causes.
    pub build_rss_bytes: i64,
    pub matches: usize,
    /// Fraction of queries that returned the exact same image-1 centre as the baseline.
    pub same_point: f64,
    /// Fraction of queries whose match is *as good as* the baseline's — the same squared
    /// distance, reached via a different but equally near descriptor. This, not
    /// `same_point`, is the quality number that matters: ties are common in a quantized
    /// 6-D space and picking a different tie is not an error.
    pub same_distance: f64,
    /// Mean of `sqrt(found_d2 / exact_d2)` over queries where the baseline distance was
    /// non-zero. 1.0 means no quality loss at all.
    pub mean_ratio: f64,
    pub p99_ratio: f64,
    pub max_ratio: f64,
    /// Mean absolute distance (key units) of the baseline's match and of this backend's.
    /// The ratio alone can look alarming when the exact distance is tiny; these say how
    /// far off the answers are in absolute terms.
    pub mean_exact_dist: f64,
    pub mean_found_dist: f64,
}

impl BackendReport {
    pub fn total(&self) -> Duration {
        self.build + self.query
    }
}

/// Threads available to the search, which is 1 in a build without the `parallel`
/// feature — the configuration a wasm build runs.
fn thread_count() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Current resident set size of this process, in bytes.
fn rss_bytes() -> i64 {
    // /proc/self/statm reports sizes in pages; field 1 is the resident set.
    let Ok(s) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let pages: i64 = s.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    pages * 4096
}

/// Squared distance between two descriptors in the 6-D quantized key space — the same
/// metric the kd-tree and the HNSW index both minimise.
fn key_distance_sq(a: &CircularFeatureDescriptor, b: &CircularFeatureDescriptor) -> i64 {
    (0..6)
        .map(|k| {
            let d = a.feature_vector[k] as i64 - b.feature_vector[k] as i64;
            d * d
        })
        .sum()
}

/// Scores `candidate` against the exact `baseline`.
///
/// `stride` says which query each candidate entry answers: entry `k` of a strided run
/// corresponds to entry `k * stride` of the baseline.
fn score(
    baseline: &[FeatureMatch],
    candidate: &[FeatureMatch],
    stride: usize,
    infos1: &[CircularFeatureDescriptor],
    infos2: &[CircularFeatureDescriptor],
    width: usize,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    // The grid is filled row-major (`out[x + y * w]`), so a centre maps straight back to
    // the descriptor that produced it.
    let descriptor_at = |x: u16, y: u16| -> &CircularFeatureDescriptor {
        &infos1[x as usize + y as usize * width]
    };

    let mut same_point = 0usize;
    let mut same_distance = 0usize;
    let mut ratios: Vec<f64> = Vec::with_capacity(candidate.len());
    let mut sum_exact = 0f64;
    let mut sum_found = 0f64;
    let mut compared = 0usize;

    for (k, cand) in candidate.iter().enumerate() {
        let bi = k * stride;
        if bi >= baseline.len() {
            break;
        }
        let base = &baseline[bi];
        compared += 1;

        if (base.x1, base.y1) == (cand.x1, cand.y1) {
            same_point += 1;
        }

        let query = &infos2[bi];
        let exact_d2 = key_distance_sq(descriptor_at(base.x1, base.y1), query);
        let found_d2 = key_distance_sq(descriptor_at(cand.x1, cand.y1), query);

        if found_d2 == exact_d2 {
            same_distance += 1;
        }
        sum_exact += (exact_d2 as f64).sqrt();
        sum_found += (found_d2 as f64).sqrt();
        if exact_d2 > 0 {
            ratios.push((found_d2 as f64 / exact_d2 as f64).sqrt());
        }
    }

    if compared == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let n = compared as f64;
    let mean = if ratios.is_empty() {
        1.0
    } else {
        ratios.iter().sum::<f64>() / ratios.len() as f64
    };
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = ratios.get((ratios.len() as f64 * 0.99) as usize).copied().unwrap_or(1.0);
    let max = ratios.last().copied().unwrap_or(1.0);

    (
        same_point as f64 / n,
        same_distance as f64 / n,
        mean,
        p99,
        max,
        sum_exact / n,
        sum_found / n,
    )
}

/// Runs every available backend on `photo1`/`photo2` scaled to `width`, and returns one
/// report per backend. The first entry is always the exact baseline.
pub fn run(photo1: &Photo, photo2: &Photo, width: usize) -> Vec<BackendReport> {
    let width = usize::min(width, photo1.width);
    let p1 = photo1.get_scaled_proportional(width);
    let p2 = photo2.get_scaled_proportional(width);

    println!(
        "grid {}x{} = {} descriptors per image ({} threads)",
        p1.width,
        p1.height,
        p1.width * p1.height,
        thread_count()
    );

    let grid1 = CircularFeatureGrid::new(&p1, p1.width, p1.height, CIRCLE_RADIUS);
    let grid2 = CircularFeatureGrid::new(&p2, p2.width, p2.height, CIRCLE_RADIUS);
    let (infos1, infos2) = (grid1.get_infos(), grid2.get_infos());

    print_key_space_stats(infos1, "image1 key space");
    print_key_space_stats(infos2, "image2 key space");

    let matcher = CircularFeatureDescriptorMatcher::new();

    let mut backends: Vec<(String, MatcherBackend, usize)> = vec![
        ("kdtree-legacy (baseline)".into(), MatcherBackend::KdTreeLegacy, 1),
    ];
    // Serial is the configuration a wasm build actually runs, so it is measured first
    // and on equal terms; the parallel rows only exist where threads are available.
    for stride in [1usize, 2, 4, 8, 16] {
        backends.push((
            format!("kdtree serial stride{stride}"),
            MatcherBackend::KdTree { stride, parallel: false },
            stride,
        ));
    }
    if parallelism_available() {
        for stride in [1usize, 2, 4, 8, 16] {
            backends.push((
                format!("kdtree parallel stride{stride}"),
                MatcherBackend::KdTree { stride, parallel: true },
                stride,
            ));
        }
    }

    let mut baseline: Option<Vec<FeatureMatch>> = None;
    let mut reports = Vec::new();

    for (name, backend, stride) in backends {
        println!("running {name} ...");
        let before = rss_bytes();
        let (result, timing) = matcher.match_areas_timed(&grid1, &grid2, backend);
        let after = rss_bytes();

        let (
            same_point,
            same_distance,
            mean_ratio,
            p99_ratio,
            max_ratio,
            mean_exact_dist,
            mean_found_dist,
        ) = match &baseline {
            None => {
                let (_, _, _, _, _, e, f) = score(&result, &result, 1, infos1, infos2, p1.width);
                (1.0, 1.0, 1.0, 1.0, 1.0, e, f)
            }
            Some(base) => score(base, &result, stride, infos1, infos2, p1.width),
        };

        reports.push(BackendReport {
            name,
            build: timing.build,
            query: timing.query,
            build_rss_bytes: after - before,
            matches: result.len(),
            same_point,
            same_distance,
            mean_ratio,
            p99_ratio,
            max_ratio,
            mean_exact_dist,
            mean_found_dist,
        });

        if baseline.is_none() {
            baseline = Some(result);
        }
    }

    reports
}

/// Reports how degenerate the 6-D key space actually is.
///
/// This matters for interpreting the results: a graph index navigates by distance, so a
/// key space with massive duplicate clusters is pathological for HNSW in a way it is not
/// for a kd-tree, whose median splits handle duplicates without getting stuck.
pub fn print_key_space_stats(infos: &[CircularFeatureDescriptor], label: &str) {
    use std::collections::HashMap;
    let mut counts: HashMap<[i16; 6], usize> = HashMap::new();
    for d in infos {
        *counts.entry(d.feature_vector).or_insert(0) += 1;
    }
    let mut multiplicities: Vec<usize> = counts.values().copied().collect();
    multiplicities.sort_unstable_by(|a, b| b.cmp(a));
    let distinct = multiplicities.len();
    let top: usize = multiplicities.iter().take(10).sum();
    println!(
        "{label}: {} descriptors, {} distinct keys ({:.1}% unique); \
         largest duplicate cluster {}, top-10 clusters cover {:.1}% of all points",
        infos.len(),
        distinct,
        100.0 * distinct as f64 / infos.len() as f64,
        multiplicities.first().copied().unwrap_or(0),
        100.0 * top as f64 / infos.len() as f64,
    );
}

/// Prints `reports` as a table, relative to the first (baseline) entry.
pub fn print_reports(reports: &[BackendReport]) {
    let Some(base) = reports.first() else { return };
    let base_total = base.total().as_secs_f64();

    println!();
    println!(
        "{:<34} {:>9} {:>9} {:>9} {:>7} {:>10} {:>9} {:>9} {:>8} {:>8}",
        "backend",
        "build s",
        "query s",
        "total s",
        "speedup",
        "matches",
        "same pt",
        "same dist",
        "mean r",
        "mean d"
    );
    println!("{}", "-".repeat(126));
    for r in reports {
        let total = r.total().as_secs_f64();
        println!(
            "{:<34} {:>9.3} {:>9.3} {:>9.3} {:>6.2}x {:>10} {:>8.2}% {:>8.2}% {:>8.4} {:>8.3}",
            r.name,
            r.build.as_secs_f64(),
            r.query.as_secs_f64(),
            total,
            if total > 0.0 { base_total / total } else { 0.0 },
            r.matches,
            r.same_point * 100.0,
            r.same_distance * 100.0,
            r.mean_ratio,
            r.mean_found_dist,
        );
    }
    println!();
    println!(
        "same pt   = returned the identical image-1 centre as the exact baseline\n\
         same dist = returned an equally near descriptor (ties count as correct)\n\
         mean r    = mean sqrt(found_dist^2 / exact_dist^2); 1.0 means no quality loss\n\
         mean d    = mean absolute distance of the returned match, in key units"
    );
}
