//! Two-view perspective geometry: the fundamental matrix, the relative pose it implies
//! under an assumed focal length, and triangulation.
//!
//! Everything here is textbook (Hartley & Zisserman, chapters 9–12), chosen for being
//! robust on a *dense* correspondence field, which differs from a sparse feature set in
//! two ways that matter: there are tens of thousands of matches rather than hundreds, so
//! hypotheses are scored on a sample of them; and neighbouring matches are strongly
//! correlated, so a wrong region of the map shows up as a coherent block of outliers
//! rather than scattered noise, which is exactly what RANSAC is good at ignoring.

use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen, Vector3};

use crate::rng::Rng;
use crate::{quantile, Settings};

type Matrix9 = SMatrix<f64, 9, 9>;
type Vector9 = SVector<f64, 9>;

/// Probability that RANSAC draws at least one all-inlier sample before it stops.
const CONFIDENCE: f64 = 0.999;
/// Hard cap on RANSAC iterations, reached only when the inlier ratio is very low.
const MAX_ITERATIONS: usize = 2000;
/// A floor on iterations, so a lucky first draw on messy data is not the final word.
const MIN_ITERATIONS: usize = 32;
/// Hypotheses are scored against at most this many correspondences. A dense map has
/// tens of thousands, and ranking hypotheses does not need all of them; the winner is
/// re-scored and refined against the full set.
const SCORING_POINTS: usize = 4000;
/// Rounds of refitting on the inlier set after RANSAC has picked a hypothesis.
const REFINEMENT_ROUNDS: usize = 4;
/// Inlier threshold for the epipolar distance, as a fraction of the larger image
/// dimension: 1.6 px at an 800 px working width.
const INLIER_THRESHOLD: f64 = 0.002;
/// Never ask for better than one pixel: the solver refines translations in whole pixels.
const MIN_THRESHOLD_PX: f64 = 1.0;
/// A cell is triangulated if its epipolar distance is within this multiple of the
/// RANSAC threshold. Looser than the fit itself, which wants only clean inliers, because
/// a cell slightly off the epipolar line still triangulates to a sensible depth.
const KEEP_THRESHOLD_FACTOR: f64 = 2.0;
/// Fewer correspondences than this are not worth fitting perspective geometry to.
const MIN_CORRESPONDENCES: usize = 50;
/// Below this inlier fraction the epipolar geometry does not explain the map.
const MIN_INLIER_FRACTION: f64 = 0.3;
/// When a single homography explains this share of what the fundamental matrix does,
/// the scene is a plane or the camera only rotated, and neither constrains depth.
const MAX_HOMOGRAPHY_RATIO: f64 = 0.9;
/// Triangulation angles below this leave depth dominated by matching noise.
const MIN_PARALLAX_DEGREES: f64 = 0.5;
/// Correspondences used to decide which of the four pose candidates is the real one.
const POSE_POINTS: usize = 2000;

/// One correspondence, in working-resolution pixels of the two photos.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PointPair {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// The perspective relationship between the two photos, as estimated from a mapping.
///
/// Coordinates are those of the correspondence map's working resolution. Camera 1 sits at
/// the origin looking down `+z` with `y` pointing down the image; a point `X` in its frame
/// is at `rotation * X + translation` in camera 2's frame.
#[derive(Clone, Debug, PartialEq)]
pub struct TwoViewGeometry {
    /// The fundamental matrix `F`, with `p2ᵀ F p1 = 0` for homogeneous pixel coordinates,
    /// scaled to unit Frobenius norm.
    pub fundamental: nalgebra::Matrix3<f64>,
    /// The essential matrix implied by `F` and the assumed intrinsics, projected onto the
    /// essential manifold (singular values `1, 1, 0`).
    pub essential: nalgebra::Matrix3<f64>,
    /// Rotation from camera 1's frame to camera 2's.
    pub rotation: nalgebra::Matrix3<f64>,
    /// Translation from camera 1's frame to camera 2's, of unit length: two views fix the
    /// direction of the baseline but not its length.
    pub translation: nalgebra::Vector3<f64>,
    /// The focal lengths assumed for the two cameras, in pixels.
    pub focal_lengths: [f64; 2],
    /// The principal point assumed for both cameras: the image centre, in pixels.
    pub principal_point: [f64; 2],
    /// How many correspondences the estimate was made from.
    pub correspondences: usize,
    /// The share of correspondences consistent with `fundamental`.
    pub inlier_fraction: f64,
    /// The share of correspondences the best single homography explains. Close to
    /// `inlier_fraction` means the views carry no usable depth.
    pub homography_inlier_fraction: f64,
    /// The median angle, in degrees, between the two viewing rays of a triangulated point.
    pub median_parallax_degrees: f64,
}

impl TwoViewGeometry {
    /// Whether this estimate is trustworthy enough to reconstruct from: enough of the map
    /// agrees with it, a homography does not explain the map just as well, and the rays
    /// meet at a wide enough angle for depth to rise above the matching noise.
    pub fn is_well_conditioned(&self) -> bool {
        self.inlier_fraction >= MIN_INLIER_FRACTION
            && self.homography_inlier_fraction < MAX_HOMOGRAPHY_RATIO * self.inlier_fraction
            && self.median_parallax_degrees >= MIN_PARALLAX_DEGREES
    }

    /// Triangulates one correspondence into camera 1's frame (`z` is depth along its
    /// optical axis), or `None` if the rays are parallel or the point would lie behind
    /// either camera.
    ///
    /// The depths along the two rays are the least-squares solution of
    /// `λ2·b = R·(λ1·a) + t`: the midpoint method, which needs no iteration and degrades
    /// gracefully for rays that almost but not quite meet.
    pub fn triangulate(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> Option<[f64; 3]> {
        let a = self.ray(0, x1, y1);
        let b = self.ray(1, x2, y2);
        let (l1, l2) = solve_depths(&self.rotation, &self.translation, &a, &b)?;
        if l1 > 0.0 && l2 > 0.0 {
            let p = a * l1;
            Some([p.x, p.y, p.z])
        } else {
            None
        }
    }

    /// The viewing ray of pixel `(x, y)` in camera `which`, with unit `z`.
    fn ray(&self, which: usize, x: f64, y: f64) -> Vector3<f64> {
        let f = self.focal_lengths[which];
        let [cx, cy] = self.principal_point;
        Vector3::new((x - cx) / f, (y - cy) / f, 1.0)
    }
}

/// The result of [`estimate`]: the geometry, and which of the input correspondences are
/// consistent with it and lie in front of both cameras.
pub(crate) struct Estimate {
    pub geometry: TwoViewGeometry,
    pub keep: Vec<bool>,
}

/// Estimates the two-view geometry of a set of correspondences between photos of
/// `width` x `height` pixels, or `None` if there is too little to fit or the fit fails.
///
/// Returns an estimate even when it is not [well
/// conditioned](TwoViewGeometry::is_well_conditioned); deciding what to do about that is
/// the caller's business.
pub(crate) fn estimate(
    pairs: &[PointPair],
    width: usize,
    height: usize,
    settings: &Settings,
) -> Option<Estimate> {
    if pairs.len() < MIN_CORRESPONDENCES {
        return None;
    }
    let larger = width.max(height) as f64;
    let threshold = (INLIER_THRESHOLD * larger).max(MIN_THRESHOLD_PX);
    let threshold2 = threshold * threshold;
    let norm = Normalized::new(pairs);
    let mut rng = Rng::new(settings.seed);

    let fundamental = ransac(
        pairs.len(),
        8,
        threshold2,
        &mut rng,
        |sample| fit_fundamental(&norm, sample, None),
        |f, i| sampson(f, &pairs[i]),
        |f, inliers| {
            // Weight each equation by its Sampson denominator, so the refit minimises
            // (a first-order approximation of) geometric distance rather than the
            // algebraic residual, which over-weights points far from the epipoles.
            let mut weights: Vec<f64> = inliers
                .iter()
                .map(|&i| 1.0 / sampson_denominator(f, &pairs[i]).max(1e-300))
                .collect();
            let mean = weights.iter().sum::<f64>() / weights.len() as f64;
            weights.iter_mut().for_each(|w| *w /= mean);
            fit_fundamental(&norm, inliers, Some(&weights))
        },
    )?;
    let homography = ransac(
        pairs.len(),
        4,
        threshold2,
        &mut rng,
        |sample| fit_homography(&norm, sample),
        |h, i| transfer_error(h, &pairs[i]),
        |_, inliers| fit_homography(&norm, inliers),
    );
    let homography_count = homography.map_or(0, |h| h.count);

    // Intrinsics: square pixels, no skew, principal point at the centre of the frame.
    let focal_lengths = settings.focal_lengths.map(|ratio| ratio * larger);
    let principal_point = [(width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0];
    let k = |f: f64| {
        Matrix3::new(
            f,
            0.0,
            principal_point[0],
            0.0,
            f,
            principal_point[1],
            0.0,
            0.0,
            1.0,
        )
    };
    let (k1, k2) = (k(focal_lengths[0]), k(focal_lengths[1]));

    let (u, _, vt) = svd3(&(k2.transpose() * fundamental.model * k1))?;
    let essential = u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0)) * vt;

    let mut geometry = TwoViewGeometry {
        fundamental: fundamental.model,
        essential,
        rotation: Matrix3::identity(),
        translation: Vector3::zeros(),
        focal_lengths,
        principal_point,
        correspondences: pairs.len(),
        inlier_fraction: fundamental.count as f64 / pairs.len() as f64,
        homography_inlier_fraction: homography_count as f64 / pairs.len() as f64,
        median_parallax_degrees: 0.0,
    };

    // The essential matrix factors four ways; only one puts the scene in front of both
    // cameras. Decide by counting, over a spread of inliers.
    let inliers: Vec<usize> = (0..pairs.len())
        .filter(|&i| fundamental.inliers[i])
        .collect();
    let step = (inliers.len() / POSE_POINTS).max(1);
    let sample: Vec<[Vector3<f64>; 2]> = inliers
        .iter()
        .step_by(step)
        .map(|&i| {
            let p = &pairs[i];
            [geometry.ray(0, p.x1, p.y1), geometry.ray(1, p.x2, p.y2)]
        })
        .collect();
    let (rotation, translation) = pose_candidates(&u, &vt).into_iter().max_by_key(|(r, t)| {
        sample
            .iter()
            .filter(|ab| {
                solve_depths(r, t, &ab[0], &ab[1]).is_some_and(|(l1, l2)| l1 > 0.0 && l2 > 0.0)
            })
            .count()
    })?;
    geometry.rotation = rotation;
    geometry.translation = translation;

    let keep_threshold2 = threshold2 * KEEP_THRESHOLD_FACTOR * KEEP_THRESHOLD_FACTOR;
    let mut parallax = Vec::new();
    let keep = pairs
        .iter()
        .map(|p| {
            if sampson(&geometry.fundamental, p) > keep_threshold2 {
                return false;
            }
            let a = geometry.ray(0, p.x1, p.y1);
            let b = geometry.ray(1, p.x2, p.y2);
            match solve_depths(&rotation, &translation, &a, &b) {
                Some((l1, l2)) if l1 > 0.0 && l2 > 0.0 => {
                    parallax.push(angle_between(&(rotation * a), &b));
                    true
                }
                _ => false,
            }
        })
        .collect();
    if !parallax.is_empty() {
        geometry.median_parallax_degrees = quantile(&mut parallax, 0.5).to_degrees();
    }

    Some(Estimate { geometry, keep })
}

/// The four `(R, t)` factorizations of an essential matrix with SVD `U diag(1,1,0) Vᵀ`.
fn pose_candidates(u: &Matrix3<f64>, vt: &Matrix3<f64>) -> [(Matrix3<f64>, Vector3<f64>); 4] {
    // Negating U or Vᵀ only flips the sign of E, which is defined up to scale anyway, and
    // makes both proper rotations so that the R built from them is one too.
    let u = if u.determinant() < 0.0 { -u } else { *u };
    let vt = if vt.determinant() < 0.0 { -vt } else { *vt };
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let ra = u * w * vt;
    let rb = u * w.transpose() * vt;
    let t: Vector3<f64> = u.column(2).into_owned();
    [(ra, t), (ra, -t), (rb, t), (rb, -t)]
}

/// Depths `(λ1, λ2)` minimising `|R·(λ1·a) + t − λ2·b|`, or `None` for parallel rays.
fn solve_depths(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    a: &Vector3<f64>,
    b: &Vector3<f64>,
) -> Option<(f64, f64)> {
    let ra = r * a;
    let (aa, ab, bb) = (ra.dot(&ra), ra.dot(b), b.dot(b));
    let det = aa * bb - ab * ab;
    if det <= 1e-12 * aa * bb {
        return None;
    }
    let (c1, c2) = (-ra.dot(t), b.dot(t));
    Some(((bb * c1 + ab * c2) / det, (ab * c1 + aa * c2) / det))
}

fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    (a.dot(b) / (a.norm() * b.norm())).clamp(-1.0, 1.0).acos()
}

/// The correspondences in Hartley-normalized coordinates (centroid at the origin, mean
/// distance `√2`), which is what keeps the linear fits below well conditioned.
struct Normalized {
    t1: Matrix3<f64>,
    t2: Matrix3<f64>,
    p1: Vec<[f64; 2]>,
    p2: Vec<[f64; 2]>,
}

impl Normalized {
    fn new(pairs: &[PointPair]) -> Self {
        let t1 = similarity(pairs.iter().map(|p| (p.x1, p.y1)));
        let t2 = similarity(pairs.iter().map(|p| (p.x2, p.y2)));
        let apply = |t: &Matrix3<f64>, x: f64, y: f64| {
            [t[(0, 0)] * x + t[(0, 2)], t[(1, 1)] * y + t[(1, 2)]]
        };
        Normalized {
            p1: pairs.iter().map(|p| apply(&t1, p.x1, p.y1)).collect(),
            p2: pairs.iter().map(|p| apply(&t2, p.x2, p.y2)).collect(),
            t1,
            t2,
        }
    }
}

fn similarity(points: impl Iterator<Item = (f64, f64)> + Clone) -> Matrix3<f64> {
    let n = points.clone().count().max(1) as f64;
    let (sx, sy) = points
        .clone()
        .fold((0.0, 0.0), |(ax, ay), (x, y)| (ax + x, ay + y));
    let (mx, my) = (sx / n, sy / n);
    let mean_distance = points.map(|(x, y)| (x - mx).hypot(y - my)).sum::<f64>() / n;
    let s = if mean_distance > 0.0 {
        std::f64::consts::SQRT_2 / mean_distance
    } else {
        1.0
    };
    Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0)
}

/// The unit vector minimising `vᵀ M v` for a symmetric positive semi-definite `M`.
fn smallest_eigenvector(m: Matrix9) -> Vector9 {
    let eigen = SymmetricEigen::new(m);
    eigen
        .eigenvectors
        .column(eigen.eigenvalues.imin())
        .into_owned()
}

/// The (weighted) normalized eight-point algorithm over the correspondences `indices`,
/// with the rank-2 constraint enforced and the result scaled to unit norm.
fn fit_fundamental(
    norm: &Normalized,
    indices: &[usize],
    weights: Option<&[f64]>,
) -> Option<Matrix3<f64>> {
    if indices.len() < 8 {
        return None;
    }
    let mut ata = Matrix9::zeros();
    for (k, &i) in indices.iter().enumerate() {
        let [x, y] = norm.p1[i];
        let [u, v] = norm.p2[i];
        let row = Vector9::from_column_slice(&[u * x, u * y, u, v * x, v * y, v, x, y, 1.0]);
        ata += row * row.transpose() * weights.map_or(1.0, |w| w[k]);
    }
    let f = Matrix3::from_row_slice(smallest_eigenvector(ata).as_slice());
    let (u, mut s, vt) = svd3(&f)?;
    s[2] = 0.0;
    let f = norm.t2.transpose() * (u * Matrix3::from_diagonal(&s) * vt) * norm.t1;
    unit_norm(f)
}

/// The normalized direct linear transform for a homography over `indices`.
fn fit_homography(norm: &Normalized, indices: &[usize]) -> Option<Matrix3<f64>> {
    if indices.len() < 4 {
        return None;
    }
    let mut ata = Matrix9::zeros();
    for &i in indices {
        let [x, y] = norm.p1[i];
        let [u, v] = norm.p2[i];
        let r1 = Vector9::from_column_slice(&[0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]);
        let r2 = Vector9::from_column_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
        ata += r1 * r1.transpose() + r2 * r2.transpose();
    }
    let h = Matrix3::from_row_slice(smallest_eigenvector(ata).as_slice());
    unit_norm(norm.t2.try_inverse()? * h * norm.t1)
}

fn unit_norm(m: Matrix3<f64>) -> Option<Matrix3<f64>> {
    let n = m.norm();
    (n.is_finite() && n > 0.0).then(|| m / n)
}

/// `F·p1` and `Fᵀ·p2`, the epipolar lines of a correspondence in the other image.
fn epipolar_lines(f: &Matrix3<f64>, p: &PointPair) -> (Vector3<f64>, Vector3<f64>, f64) {
    let p1 = Vector3::new(p.x1, p.y1, 1.0);
    let p2 = Vector3::new(p.x2, p.y2, 1.0);
    let fp1 = f * p1;
    let ftp2 = f.transpose() * p2;
    (fp1, ftp2, p2.dot(&fp1))
}

fn sampson_denominator(f: &Matrix3<f64>, p: &PointPair) -> f64 {
    let (fp1, ftp2, _) = epipolar_lines(f, p);
    fp1.x * fp1.x + fp1.y * fp1.y + ftp2.x * ftp2.x + ftp2.y * ftp2.y
}

/// The squared Sampson distance of a correspondence from `F`, in pixels².
fn sampson(f: &Matrix3<f64>, p: &PointPair) -> f64 {
    let (fp1, ftp2, e) = epipolar_lines(f, p);
    let d = fp1.x * fp1.x + fp1.y * fp1.y + ftp2.x * ftp2.x + ftp2.y * ftp2.y;
    if d > 0.0 {
        e * e / d
    } else {
        f64::INFINITY
    }
}

/// The squared distance, in pixels², between `p2` and where `H` sends `p1`.
fn transfer_error(h: &Matrix3<f64>, p: &PointPair) -> f64 {
    let q = h * Vector3::new(p.x1, p.y1, 1.0);
    if q.z.abs() < 1e-12 {
        return f64::INFINITY;
    }
    (q.x / q.z - p.x2).powi(2) + (q.y / q.z - p.y2).powi(2)
}

/// SVD of a 3x3 matrix with the singular values in descending order, which nalgebra does
/// not promise.
fn svd3(m: &Matrix3<f64>) -> Option<(Matrix3<f64>, Vector3<f64>, Matrix3<f64>)> {
    let svd = m.svd(true, true);
    let (u, vt, s) = (svd.u?, svd.v_t?, svd.singular_values);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    let u = Matrix3::from_columns(&order.map(|i| u.column(i).into_owned()));
    let vt = Matrix3::from_rows(&order.map(|i| vt.row(i).into_owned()));
    Some((u, Vector3::new(s[order[0]], s[order[1]], s[order[2]]), vt))
}

struct Fit<M> {
    model: M,
    inliers: Vec<bool>,
    count: usize,
}

/// Adaptive RANSAC followed by iterative refitting on the inlier set.
///
/// `fit` builds a model from a minimal sample, `error` is the squared residual of one
/// correspondence, and `refit` re-estimates a model from a set of inliers, given the
/// current model for any weighting it wants.
fn ransac<M>(
    n: usize,
    sample_size: usize,
    threshold2: f64,
    rng: &mut Rng,
    fit: impl Fn(&[usize]) -> Option<M>,
    error: impl Fn(&M, usize) -> f64,
    refit: impl Fn(&M, &[usize]) -> Option<M>,
) -> Option<Fit<M>> {
    if n < sample_size {
        return None;
    }
    let scoring: Vec<usize> = if n <= SCORING_POINTS {
        (0..n).collect()
    } else {
        (0..SCORING_POINTS).map(|_| rng.below(n)).collect()
    };

    let mut best: Option<(M, usize)> = None;
    let mut needed = MAX_ITERATIONS;
    let mut sample = Vec::with_capacity(sample_size);
    let mut iteration = 0;
    while iteration < needed.clamp(MIN_ITERATIONS, MAX_ITERATIONS) {
        iteration += 1;
        rng.sample_distinct(n, sample_size, &mut sample);
        let Some(model) = fit(&sample) else {
            continue;
        };
        let count = scoring
            .iter()
            .filter(|&&i| error(&model, i) <= threshold2)
            .count();
        if best.as_ref().is_none_or(|(_, c)| count > *c) {
            needed = iterations_needed(count as f64 / scoring.len() as f64, sample_size);
            best = Some((model, count));
        }
    }

    let (mut model, _) = best?;
    let classify = |m: &M| -> (Vec<bool>, usize) {
        let inliers: Vec<bool> = (0..n).map(|i| error(m, i) <= threshold2).collect();
        let count = inliers.iter().filter(|&&b| b).count();
        (inliers, count)
    };
    let (mut inliers, mut count) = classify(&model);
    for _ in 0..REFINEMENT_ROUNDS {
        let indices: Vec<usize> = (0..n).filter(|&i| inliers[i]).collect();
        let Some(candidate) = refit(&model, &indices) else {
            break;
        };
        let (candidate_inliers, candidate_count) = classify(&candidate);
        if candidate_count < count {
            break;
        }
        let grew = candidate_count > count;
        (model, inliers, count) = (candidate, candidate_inliers, candidate_count);
        if !grew {
            break;
        }
    }
    Some(Fit {
        model,
        inliers,
        count,
    })
}

/// Iterations after which an all-inlier sample has been drawn with probability
/// [`CONFIDENCE`], given inlier ratio `w`.
fn iterations_needed(w: f64, sample_size: usize) -> usize {
    let p = w.powi(sample_size as i32);
    if p >= 1.0 - 1e-12 {
        return 0;
    }
    if p <= 0.0 {
        return MAX_ITERATIONS;
    }
    let n = (1.0 - CONFIDENCE).ln() / (1.0 - p).ln();
    if n.is_finite() {
        (n.ceil() as usize).min(MAX_ITERATIONS)
    } else {
        MAX_ITERATIONS
    }
}
