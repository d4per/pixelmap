//! A 6-dimensional kd-tree over `i16` keys, specialised for the feature matcher.
//!
//! The matcher's only search need is "given this descriptor's 6-D key, which descriptor of
//! the other image is nearest", over a few hundred thousand to a few million points. That
//! is narrow enough to be worth owning outright rather than depending on a general tree:
//!
//! - **No dependency.** The `kd-tree` crate pulled in `ordered-float`, `num-traits`,
//!   `paste` and `typenum`, and its `ordered-float` requirement alone set the whole
//!   crate's minimum supported Rust version. With this module the library has no
//!   dependencies at all under `--no-default-features`, which is the configuration a
//!   `wasm32-unknown-unknown` build uses.
//! - **The key is stored as-is.** A general tree needs a scalar wide enough that squared
//!   distances cannot overflow, which meant widening the descriptor's `[i16; 6]` into an
//!   `[i32; 6]` for every point — a 28-byte index entry for a 20-byte descriptor, plus a
//!   conversion pass over both images. [`Point`] is 16 bytes and indexes the key
//!   directly. At the `High` schedule's 1600 px working width that is roughly 40 MB of
//!   index rather than 70 MB, which matters most on wasm, where the heap is the binding
//!   constraint.
//! - **Ties break deterministically.** Quantized 6-D keys collide often, and a tree that
//!   resolves ties by traversal order makes the whole pipeline's output depend on the
//!   order points were inserted. Here the lowest `id` wins, which is a total order.
//!
//! The tree is exact: it returns a true nearest neighbour, not an approximation.

/// Dimensionality of the search key, matching
/// [`CircularFeatureDescriptor::feature_vector`](crate::circular_feature_descriptor::CircularFeatureDescriptor::feature_vector).
const D: usize = 6;

/// Subtrees this size or smaller are stored unordered and scanned linearly.
///
/// Below some size the branch and the bounds check cost more than simply looking at every
/// point, and a linear scan over contiguous 16-byte entries is close to ideal for a
/// prefetcher. Tuned with `matcher_bench`; 16 and 32 are the neighbouring values worth
/// re-checking if the point layout ever changes.
const LEAF_SIZE: usize = 24;

/// Subtrees smaller than this are built on the current thread.
///
/// `rayon::join` is cheap but not free, and the work below this threshold does not repay
/// the split. Only the top few levels of a large build are worth handing to the pool.
#[cfg(feature = "parallel")]
const PARALLEL_BUILD_CUTOFF: usize = 32 * 1024;

/// One indexed point: a search key and the caller's index for it.
///
/// `#[repr(C)]` to guarantee rather than merely expect the 16-byte layout — 12 bytes of
/// key followed by a `u32` that is already 4-aligned at offset 12, with no tail padding.
/// The whole point of this type is its size, so it should not be at the compiler's
/// discretion.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Point {
    /// The 6-D search key.
    pub v: [i16; D],
    /// Index of the descriptor this key came from.
    pub id: u32,
}

/// The result of a [`KdTree::nearest`] query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Nearest {
    /// [`Point::id`] of the nearest point.
    pub id: u32,
    /// Squared Euclidean distance to it. Squared, so the comparison stays in integers.
    pub distance_squared: u64,
}

/// A balanced kd-tree over 6-D `i16` keys.
pub(crate) struct KdTree {
    /// The points, permuted in place into tree order: within any subtree, the median on
    /// that node's split axis sits at the midpoint, with smaller keys before it and
    /// larger after. No node structs and no child pointers — the structure is the
    /// ordering.
    points: Vec<Point>,
    /// The split axis of the node whose pivot is `points[i]`, at `axes[i]` (an index into
    /// [`AXES`]). Entries at leaf positions are unused.
    axes: Vec<u8>,
}

impl KdTree {
    /// Builds the tree, permuting `points` in place, using the thread pool when the
    /// `parallel` feature is on.
    pub(crate) fn build(points: Vec<Point>) -> Self {
        Self::build_with::<true>(points)
    }

    /// [`Self::build`], pinned to one thread.
    ///
    /// The finished tree is identical to the parallel one — the split is over *when* the
    /// work happens, never over the partition — so this exists to let `matcher_bench`
    /// time the two against each other, and for callers that are already inside a pool.
    #[allow(dead_code)] // Used by `matcher_bench` and by tests.
    pub(crate) fn build_serial(points: Vec<Point>) -> Self {
        Self::build_with::<false>(points)
    }

    fn build_with<const PARALLEL: bool>(mut points: Vec<Point>) -> Self {
        dedup_keys::<PARALLEL>(&mut points);
        let mut axes = vec![0u8; points.len()];
        build_recursive::<PARALLEL>(&mut points, &mut axes);
        KdTree { points, axes }
    }

    /// The point nearest to `query`, or `None` if the tree is empty.
    ///
    /// Exact. Among points at equal distance the lowest [`Point::id`] wins, so the answer
    /// does not depend on the order the points were built in.
    pub(crate) fn nearest(&self, query: &[i16; D]) -> Option<Nearest> {
        if self.points.is_empty() {
            return None;
        }
        let mut best = Nearest {
            id: u32::MAX,
            distance_squared: u64::MAX,
        };
        let mut projected = [0i64; D];
        for (axis, p) in projected.iter_mut().enumerate() {
            *p = project(axis, query);
        }
        let mut search = Search {
            query,
            projected,
            offsets: [0; D],
            best,
        };
        search.descend(&self.points, &self.axes, 0);
        best = search.best;
        Some(best)
    }

    /// How many points the tree holds: one per distinct key, see [`dedup_keys`].
    #[allow(dead_code)] // Used by tests and by `matcher_bench`.
    pub(crate) fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the tree holds no points at all.
    #[allow(dead_code)] // Paired with `len` so the two cannot disagree.
    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Keeps only the lowest-id point of every distinct key.
///
/// The others can never be an answer: any query sees them at exactly the same distance as
/// their lowest-id twin, and [`consider`] breaks that tie in the twin's favour. Dropping
/// them therefore changes no result — but it changes the cost a great deal. The search has
/// to explore exact ties (the `<=` prune in [`Search::descend`]), so a query landing on a
/// key shared by `m` points used to visit all `m`. Flat image regions produce one key
/// hundreds of thousands of times over, which made the match quadratic: on a 4096x2304 pair
/// at `high` it took 99 s, against 0.3 s with the duplicates dropped.
///
/// Sorting by `(key, id)` puts each run of duplicates together with its lowest id first.
/// The ids are distinct, so the unstable sort is still a total order and the result is
/// the same in the serial and parallel builds.
fn dedup_keys<const PARALLEL: bool>(points: &mut Vec<Point>) {
    let order = |a: &Point, b: &Point| a.v.cmp(&b.v).then(a.id.cmp(&b.id));
    #[cfg(feature = "parallel")]
    if PARALLEL {
        use rayon::slice::ParallelSliceMut;
        points.par_sort_unstable_by(order);
    } else {
        points.sort_unstable_by(order);
    }
    #[cfg(not(feature = "parallel"))]
    points.sort_unstable_by(order);
    points.dedup_by(|later, first| later.v == first.v);
}

/// The directions the tree splits along, as rows: orthogonal integer vectors over the
/// key, so that [`project`] is exact. [`project`] spells them out; this table is what the
/// tests check it, the orthogonality and [`WEIGHT`] against.
///
/// The key is the centre of mass of the red, green and blue channel of a disc, each
/// rotated so that the centre of mass of all three together lies along the first
/// coordinate: `[red_x, red_y, green_x, green_y, blue_x, blue_y]`. The colour channels of
/// a photo move together, so on a real 1600 px photo the three `_x` coordinates
/// correlate at 0.96–0.98 and each spreads about ten times further than any `_y`. A tree
/// splitting the six coordinates in turn spent half its levels on the narrow ones, and
/// its axis-aligned boxes cut diagonally across the one direction the keys really vary
/// along. So the first three directions span the `_x` coordinates along their common
/// direction and across it, and the last three are the `_y` coordinates as they are.
///
/// Splitting along these, at the axis of widest spread, more than halved the work of a
/// query (77 nodes and 355 leaf points instead of 142 and 1042). The answers cannot
/// change: distances are still measured on the keys themselves, and a direction only
/// decides how the points are partitioned and how far a box can be from a query.
#[cfg(test)]
const AXES: [[i64; D]; D] = [
    [1, 0, 1, 0, 1, 0],
    [1, 0, -1, 0, 0, 0],
    [1, 0, 1, 0, -2, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
];

/// `6 / |AXES[axis]|^2` (see [`project`]). An offset `t` along an axis is `t / |AXES[axis]|` in key units,
/// so `WEIGHT[axis] * t^2` is six times its square: integers throughout, compared against
/// six times a squared distance.
const WEIGHT: [u64; D] = [2, 3, 1, 6, 6, 6];

/// How many points [`split_axis`] looks at to judge a subtree's spread. A sample is
/// enough: the axis only has to be good, not optimal, and any choice keeps the search
/// exact.
const SPREAD_SAMPLE: usize = 256;

/// `key` along split direction `axis`, unnormalized: the dot product with row `axis` of
/// `AXES`.
#[inline(always)]
fn project(axis: usize, key: &[i16; D]) -> i64 {
    let v = |k: usize| key[k] as i64;
    match axis {
        0 => v(0) + v(2) + v(4),
        1 => v(0) - v(2),
        2 => v(0) + v(2) - 2 * v(4),
        3 => v(1),
        4 => v(3),
        _ => v(5),
    }
}

/// The axis along which `points` spread widest, judged on an evenly strided sample and
/// in key units. Ties go to the lower axis, so the choice is a function of the points'
/// order alone, which [`dedup_keys`] fixes.
fn split_axis(points: &[Point]) -> usize {
    let step = (points.len() / SPREAD_SAMPLE).max(1);
    let mut best = (0usize, 0u128);
    for (axis, &weight) in WEIGHT.iter().enumerate() {
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for p in points.iter().step_by(step) {
            let x = project(axis, &p.v);
            lo = lo.min(x);
            hi = hi.max(x);
        }
        let spread = (hi - lo) as u128;
        let weighted = spread * spread * weight as u128;
        if axis == 0 || weighted > best.1 {
            best = (axis, weighted);
        }
    }
    best.0
}

/// Arranges `points` so that the median along the subtree's widest axis is at the
/// midpoint, records that axis, and recurses into both halves.
fn build_recursive<const PARALLEL: bool>(points: &mut [Point], axes: &mut [u8]) {
    if points.len() <= LEAF_SIZE {
        return;
    }

    let axis = split_axis(points);
    let middle = points.len() / 2;

    // Partition-based selection: linear on average, and it only orders the slice enough
    // to put the median in place, which is all the search needs.
    points.select_nth_unstable_by_key(middle, |p| project(axis, &p.v));
    axes[middle] = axis as u8;

    let (left, rest) = points.split_at_mut(middle);
    let (left_axes, rest_axes) = axes.split_at_mut(middle);
    build_children::<PARALLEL>(left, left_axes, &mut rest[1..], &mut rest_axes[1..]);
}

/// Recurses into both halves, in parallel when the subtree is large enough to pay for it.
///
/// The halves are disjoint `&mut` slices, so this is a straightforward `rayon::join`. It
/// changes only *when* the work happens, never the partition, so the finished tree — and
/// therefore every query answer — is identical either way.
#[cfg(feature = "parallel")]
fn build_children<const PARALLEL: bool>(
    left: &mut [Point],
    left_axes: &mut [u8],
    right: &mut [Point],
    right_axes: &mut [u8],
) {
    if !PARALLEL || left.len() + right.len() < PARALLEL_BUILD_CUTOFF {
        build_recursive::<PARALLEL>(left, left_axes);
        build_recursive::<PARALLEL>(right, right_axes);
    } else {
        rayon::join(
            || build_recursive::<PARALLEL>(left, left_axes),
            || build_recursive::<PARALLEL>(right, right_axes),
        );
    }
}

/// Serial build, used when the `parallel` feature is off — notably on
/// `wasm32-unknown-unknown`, which has no threads to hand out.
#[cfg(not(feature = "parallel"))]
fn build_children<const PARALLEL: bool>(
    left: &mut [Point],
    left_axes: &mut [u8],
    right: &mut [Point],
    right_axes: &mut [u8],
) {
    build_recursive::<PARALLEL>(left, left_axes);
    build_recursive::<PARALLEL>(right, right_axes);
}

/// Records `candidate` as the new best if it beats `best`, breaking ties by lowest id.
#[inline(always)]
fn consider(best: &mut Nearest, candidate: &Point, distance_squared: u64) {
    if distance_squared < best.distance_squared
        || (distance_squared == best.distance_squared && candidate.id < best.id)
    {
        best.distance_squared = distance_squared;
        best.id = candidate.id;
    }
}

/// The state of one [`KdTree::nearest`] query.
struct Search<'q> {
    query: &'q [i16; D],
    /// `project(axis, query)` for every axis.
    projected: [i64; D],
    /// How far the query lies outside the current subtree's box along each axis, in
    /// projected units; zero on the side it is inside.
    offsets: [i64; D],
    best: Nearest,
}

impl Search<'_> {
    /// Descends a subtree whose box is `box_distance / 6` (squared, key units) or more
    /// from the query.
    ///
    /// The bound is the whole box's, not just the last split's: the far child of a node
    /// is pruned when the squared offsets along *every* axis so far add up past the best
    /// distance (Arya and Mount's incremental distance), where a classic kd-tree looks at
    /// the one axis it just split. The axes are orthogonal, so the weighted sum of
    /// squared offsets is a true lower bound on the distance to any point in the box.
    fn descend(&mut self, points: &[Point], axes: &[u8], box_distance: u64) {
        if points.len() <= LEAF_SIZE {
            for p in points {
                consider(&mut self.best, p, distance_squared(self.query, &p.v));
            }
            return;
        }

        let middle = points.len() / 2;
        let pivot = &points[middle];
        consider(
            &mut self.best,
            pivot,
            distance_squared(self.query, &pivot.v),
        );

        let axis = axes[middle] as usize;
        let delta = self.projected[axis] - project(axis, &pivot.v);

        // Descend the side the query actually falls on first: it tightens `best` before
        // the other side is considered at all, which is what makes the prune below worth
        // anything.
        let (near, near_axes, far, far_axes) = if delta <= 0 {
            (
                &points[..middle],
                &axes[..middle],
                &points[middle + 1..],
                &axes[middle + 1..],
            )
        } else {
            (
                &points[middle + 1..],
                &axes[middle + 1..],
                &points[..middle],
                &axes[..middle],
            )
        };
        self.descend(near, near_axes, box_distance);

        // Every point on the far side is at least `|delta|` along this axis from the
        // query, which replaces the box's old offset along it. The comparison is `<=`
        // rather than `<` on purpose: an exact tie has to be explored, or a point
        // equidistant with the current best could be missed — and with it the lowest-id
        // tie-break that makes this tree order-independent.
        let old = self.offsets[axis];
        let w = WEIGHT[axis];
        let far_distance = box_distance - w * (old * old) as u64 + w * (delta * delta) as u64;
        if far_distance <= self.best.distance_squared.saturating_mul(6) {
            self.offsets[axis] = delta;
            self.descend(far, far_axes, far_distance);
            self.offsets[axis] = old;
        }
    }
}

/// Squared Euclidean distance between two keys.
///
/// In `i64` because the intermediate does not fit anything narrower: two `i16`s can differ
/// by 65535, and `65535² = 4_294_836_225` overflows `i32`. Summing six of those needs
/// 64 bits regardless.
///
/// Working in `i64` rather than a tighter type is also what lets the matcher drop the
/// range check it used to run over every descriptor: there is no overflow budget left to
/// police. The real keys stay near ±1000, so this is head-room, not a working range.
#[inline(always)]
fn distance_squared(a: &[i16; D], b: &[i16; D]) -> u64 {
    let mut sum = 0i64;
    for k in 0..D {
        let d = a[k] as i64 - b[k] as i64;
        sum += d * d;
    }
    sum as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// The answer the tree has to reproduce: look at everything, same tie-break.
    fn brute_force(points: &[Point], query: &[i16; D]) -> Option<Nearest> {
        let mut best = Nearest {
            id: u32::MAX,
            distance_squared: u64::MAX,
        };
        for p in points {
            consider(&mut best, p, distance_squared(query, &p.v));
        }
        (!points.is_empty()).then_some(best)
    }

    fn random_points(rng: &mut Rng, count: usize, spread: i16) -> Vec<Point> {
        (0..count)
            .map(|i| {
                let mut v = [0i16; D];
                for slot in v.iter_mut() {
                    *slot = (rng.next_u64() % (spread as u64 * 2 + 1)) as i16 - spread;
                }
                Point { v, id: i as u32 }
            })
            .collect()
    }

    fn random_key(rng: &mut Rng, spread: i16) -> [i16; D] {
        let mut v = [0i16; D];
        for slot in v.iter_mut() {
            *slot = (rng.next_u64() % (spread as u64 * 2 + 1)) as i16 - spread;
        }
        v
    }

    /// The headline property. Sizes straddle `LEAF_SIZE` so both the linear-scan path and
    /// the recursive path are covered, including the boundary itself.
    #[test]
    fn nearest_agrees_with_brute_force() {
        let mut rng = Rng::seed_from_u64(0xA11CE);
        for &count in &[1usize, 2, 23, 24, 25, 49, 500, 2000] {
            let points = random_points(&mut rng, count, 1000);
            let tree = KdTree::build(points.clone());
            assert_eq!(tree.len(), count);

            for _ in 0..200 {
                let query = random_key(&mut rng, 1200);
                assert_eq!(
                    tree.nearest(&query),
                    brute_force(&points, &query),
                    "disagreed on a {count}-point tree at query {query:?}"
                );
            }
        }
    }

    /// Ties are the interesting case: a narrow key space makes exact-distance collisions
    /// the rule rather than the exception, which is what the `<=` prune and the id
    /// tie-break exist for.
    #[test]
    fn nearest_agrees_with_brute_force_when_ties_are_everywhere() {
        let mut rng = Rng::seed_from_u64(0xB0B);
        for &count in &[30usize, 200, 1500] {
            // A spread of 2 over 6 axes gives 5^6 distinct keys for up to 1500 points, so
            // duplicates and equidistant candidates are guaranteed.
            let points = random_points(&mut rng, count, 2);
            let tree = KdTree::build(points.clone());

            for _ in 0..500 {
                let query = random_key(&mut rng, 3);
                assert_eq!(
                    tree.nearest(&query),
                    brute_force(&points, &query),
                    "disagreed on a {count}-point tree with heavy ties"
                );
            }
        }
    }

    /// Keys shaped like the matcher's: the three `_x` coordinates move together and
    /// spread wide, the `_y` ones stay narrow. This is the distribution [`AXES`] is
    /// chosen for, so it is where the rotated splits and the box bound do the most work.
    #[test]
    fn nearest_agrees_with_brute_force_on_correlated_keys() {
        let mut rng = Rng::seed_from_u64(0xC0DE);
        let key = |rng: &mut Rng| {
            let mut n = |spread: u64| (rng.next_u64() % (2 * spread + 1)) as i16 - spread as i16;
            let common = n(120);
            [
                common + n(6),
                n(5),
                common + n(6),
                n(4),
                common + n(8),
                n(5),
            ]
        };
        let points: Vec<Point> = (0..20_000)
            .map(|i| Point {
                v: key(&mut rng),
                id: i,
            })
            .collect();
        let tree = KdTree::build(points.clone());
        for _ in 0..2000 {
            let query = key(&mut rng);
            assert_eq!(tree.nearest(&query), brute_force(&points, &query));
        }
    }

    /// Exact duplicates: every point is equidistant from every query, so the answer is
    /// decided entirely by the tie-break.
    #[test]
    fn identical_points_resolve_to_the_lowest_id() {
        let points: Vec<Point> = (0..100).map(|i| Point { v: [7; D], id: i }).collect();
        let tree = KdTree::build(points);
        let found = tree.nearest(&[9; D]).expect("tree is not empty");
        assert_eq!(found.id, 0);
        assert_eq!(found.distance_squared, 4 * D as u64);
    }

    /// The property the tie-break buys: the answer does not depend on the order the tree
    /// was built in. Without it the mapping would silently depend on descriptor ordering.
    #[test]
    fn the_answer_does_not_depend_on_build_order() {
        let mut rng = Rng::seed_from_u64(0xC0FFEE);
        let points = random_points(&mut rng, 800, 6);
        let queries: Vec<[i16; D]> = (0..300).map(|_| random_key(&mut rng, 8)).collect();

        let reference = KdTree::build(points.clone());
        let expected: Vec<_> = queries.iter().map(|q| reference.nearest(q)).collect();

        for _ in 0..8 {
            let mut shuffled = points.clone();
            rng.shuffle(&mut shuffled);
            let tree = KdTree::build(shuffled);
            let got: Vec<_> = queries.iter().map(|q| tree.nearest(q)).collect();
            assert_eq!(
                got, expected,
                "a reordered build produced different answers"
            );
        }
    }

    /// The parallel build must be a pure scheduling change.
    #[test]
    fn the_parallel_build_matches_the_serial_one() {
        let mut rng = Rng::seed_from_u64(0xD00D);
        let points = random_points(&mut rng, 5000, 400);
        let parallel = KdTree::build(points.clone());
        let serial = KdTree::build_serial(points);
        assert_eq!(
            parallel.points, serial.points,
            "the two builds produced different trees"
        );
        assert_eq!(
            parallel.axes, serial.axes,
            "the two builds split differently"
        );
    }

    #[test]
    fn degenerate_trees_behave() {
        let empty = KdTree::build(Vec::new());
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.nearest(&[0; D]), None);

        let single = KdTree::build(vec![Point {
            v: [1, 2, 3, 4, 5, 6],
            id: 42,
        }]);
        assert!(!single.is_empty());
        let found = single
            .nearest(&[1, 2, 3, 4, 5, 6])
            .expect("tree is not empty");
        assert_eq!(
            found,
            Nearest {
                id: 42,
                distance_squared: 0
            }
        );
    }

    /// The widest keys the type allows, where an `i32` accumulator would overflow.
    #[test]
    fn extreme_keys_do_not_overflow() {
        let points = vec![
            Point {
                v: [i16::MIN; D],
                id: 0,
            },
            Point {
                v: [i16::MAX; D],
                id: 1,
            },
        ];
        let tree = KdTree::build(points.clone());

        let query = [i16::MAX; D];
        let found = tree.nearest(&query).expect("tree is not empty");
        assert_eq!(
            found,
            Nearest {
                id: 1,
                distance_squared: 0
            }
        );

        // 6 * 65535^2 = 25_769_017_350, which needs more than 32 bits.
        let far = distance_squared(&[i16::MIN; D], &[i16::MAX; D]);
        assert_eq!(far, 6 * 65_535u64 * 65_535);
        assert_eq!(brute_force(&points, &query), Some(found));
    }

    /// The search bound is only a lower bound if the split directions are orthogonal and
    /// `WEIGHT` really is `6 / |direction|^2`; and `project` has to be the dot product it
    /// claims to be.
    #[test]
    fn the_split_directions_are_orthogonal_and_weighted_right() {
        for (a, row_a) in AXES.iter().enumerate() {
            for (b, row_b) in AXES.iter().enumerate() {
                let dot: i64 = row_a.iter().zip(row_b).map(|(x, y)| x * y).sum();
                if a == b {
                    assert_eq!(WEIGHT[a] as i64 * dot, 6, "weight of axis {a}");
                } else {
                    assert_eq!(dot, 0, "axes {a} and {b} are not orthogonal");
                }
            }
        }
        let mut rng = Rng::seed_from_u64(0xA7E5);
        for _ in 0..1000 {
            let key = random_key(&mut rng, 16_000);
            for (axis, row) in AXES.iter().enumerate() {
                let dot: i64 = (0..D).map(|k| row[k] * key[k] as i64).sum();
                assert_eq!(project(axis, &key), dot);
            }
        }
    }

    /// The index layout is the reason this type exists, so it is worth asserting.
    #[test]
    fn a_point_is_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Point>(), 16);
    }
}
