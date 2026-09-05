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
    /// that level's axis sits at the midpoint, with smaller keys before it and larger
    /// after. No node structs and no child pointers — the structure is the ordering, so
    /// the whole tree is one allocation.
    points: Vec<Point>,
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
        build_recursive::<PARALLEL>(&mut points, 0);
        KdTree { points }
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
        search_recursive(&self.points, query, 0, &mut best);
        Some(best)
    }

    /// How many points the tree holds.
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

/// Arranges `points` so that the median on this level's axis is at the midpoint.
///
/// The axis simply cycles (`depth % D`). Choosing the widest-spread axis instead gives a
/// slightly better-shaped tree but has to measure every dimension at every node, which
/// costs more at build time than it returns at query time for keys as uniform as these.
fn build_recursive<const PARALLEL: bool>(points: &mut [Point], depth: usize) {
    if points.len() <= LEAF_SIZE {
        return;
    }

    let axis = depth % D;
    let middle = points.len() / 2;

    // Partition-based selection: linear on average, and it only orders the slice enough
    // to put the median in place, which is all the search needs.
    let (left, _pivot, right) =
        points.select_nth_unstable_by(middle, |a, b| a.v[axis].cmp(&b.v[axis]));

    build_children::<PARALLEL>(left, right, depth);
}

/// Recurses into both halves, in parallel when the subtree is large enough to pay for it.
///
/// The halves are disjoint `&mut` slices, so this is a straightforward `rayon::join`. It
/// changes only *when* the work happens, never the partition, so the finished tree — and
/// therefore every query answer — is identical either way.
#[cfg(feature = "parallel")]
fn build_children<const PARALLEL: bool>(left: &mut [Point], right: &mut [Point], depth: usize) {
    if !PARALLEL || left.len() + right.len() < PARALLEL_BUILD_CUTOFF {
        build_recursive::<PARALLEL>(left, depth + 1);
        build_recursive::<PARALLEL>(right, depth + 1);
    } else {
        rayon::join(
            || build_recursive::<PARALLEL>(left, depth + 1),
            || build_recursive::<PARALLEL>(right, depth + 1),
        );
    }
}

/// Serial build, used when the `parallel` feature is off — notably on
/// `wasm32-unknown-unknown`, which has no threads to hand out.
#[cfg(not(feature = "parallel"))]
fn build_children<const PARALLEL: bool>(left: &mut [Point], right: &mut [Point], depth: usize) {
    build_recursive::<PARALLEL>(left, depth + 1);
    build_recursive::<PARALLEL>(right, depth + 1);
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

/// Descends the tree, narrowing `best`.
///
/// Not `#[inline(always)]`: it is recursive, so the hint cannot be honoured anyway.
fn search_recursive(points: &[Point], query: &[i16; D], depth: usize, best: &mut Nearest) {
    if points.is_empty() {
        return;
    }

    if points.len() <= LEAF_SIZE {
        for p in points {
            consider(best, p, distance_squared(query, &p.v));
        }
        return;
    }

    let middle = points.len() / 2;
    let pivot = &points[middle];
    consider(best, pivot, distance_squared(query, &pivot.v));

    let axis = depth % D;
    let delta = query[axis] as i64 - pivot.v[axis] as i64;

    // Descend the side the query actually falls on first: it tightens `best` before the
    // other side is considered at all, which is what makes the prune below worth anything.
    let (near, far) = if delta <= 0 {
        (&points[..middle], &points[middle + 1..])
    } else {
        (&points[middle + 1..], &points[..middle])
    };
    search_recursive(near, query, depth + 1, best);

    // Every point on the far side is at least `|delta|` away along this axis, so it can
    // only hold something better if the sphere around the current best crosses the
    // splitting plane. The comparison is `<=` rather than `<` on purpose: an exact tie has
    // to be explored, or a point equidistant with the current best could be missed — and
    // with it the lowest-id tie-break that makes this tree order-independent.
    if (delta * delta) as u64 <= best.distance_squared {
        search_recursive(far, query, depth + 1, best);
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

    /// The index layout is the reason this type exists, so it is worth asserting.
    #[test]
    fn a_point_is_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<Point>(), 16);
    }
}
