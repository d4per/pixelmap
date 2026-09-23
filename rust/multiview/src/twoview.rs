//! Stage 2: the relative pose of each pair, and triage of pairs too weak to use.
//!
//! Dense correspondences are sampled on a grid and fed to an 8-point essential-matrix
//! estimator inside RANSAC. RANSAC is still needed even though pixelmap has already
//! filtered its output. A relaxation-based matcher can be *smoothly* wrong across an
//! occlusion boundary or on repetitive texture, and such errors agree with each other, so
//! they get past a forward/backward check.
//!
//! The 8-point method needs the scene to have depth: a single plane is a degenerate
//! configuration for it. A 5-point solver would handle that, and can replace
//! [`eight_point`] later without changing anything else here.

use std::fmt;

use nalgebra::{Matrix3, Rotation3, SMatrix, SVector, Vector3};

use crate::calib::Intrinsics;
use crate::lookup::PairLookup;
use crate::pose::Pose;
use crate::rng::Rng;
use crate::triangulate;
use crate::types::{Norm, PairId, PhotoPx};

/// One correspondence: the same scene point seen in both photos of a pair.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Match {
    /// In the pair's first photo.
    pub a: PhotoPx,
    /// In the pair's second photo.
    pub b: PhotoPx,
}

/// Tuning for [`estimate`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Sampling stride in photo pixels. `None` uses the lookup's native stride.
    pub stride: Option<f32>,
    /// Inlier threshold, as a multiple of the lookup's precision.
    pub inlier_threshold: f32,
    /// The most RANSAC iterations to run, however low the inlier ratio.
    pub max_iterations: usize,
    /// The probability RANSAC should have of drawing at least one all-inlier sample.
    pub confidence: f64,
    /// Fewer sampled matches than this and the pair is not estimated at all.
    pub min_matches: usize,
    /// Below this inlier ratio the pair is degenerate.
    pub min_inlier_ratio: f64,
    /// Below this median triangulation angle, in degrees, the pair is degenerate.
    pub min_median_angle_deg: f64,
    /// How far the matches must stray from what a pure rotation explains, as a multiple
    /// of the lookup's precision, for the pair to count as seeing any depth at all.
    pub min_structure: f64,
    /// The inlier threshold for the flat-scene check's homography, as a multiple of the
    /// essential matrix's.
    pub homography_threshold: f64,
    /// Above this share of the inliers explained by one homography, the scene counts as
    /// flat.
    pub max_homography_ratio: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            stride: None,
            inlier_threshold: 2.0,
            max_iterations: 2000,
            confidence: 0.999,
            min_matches: 100,
            min_inlier_ratio: 0.5,
            min_median_angle_deg: 2.0,
            min_structure: 2.0,
            homography_threshold: 1.5,
            max_homography_ratio: 0.95,
        }
    }
}

/// The estimated geometry of one pair.
#[derive(Clone, Debug)]
pub struct RelativePose {
    /// The pair.
    pub pair: PairId,
    /// The fraction of the first photo the pair's mapping covers.
    pub coverage: f32,
    /// The second camera in the first camera's frame, with `‖t‖ = 1`. The pair's own scale
    /// is unknown; registration resolves it.
    pub pose: Pose,
    /// The essential matrix, with `b̂ᵀ E â = 0` for normalized homogeneous points.
    pub essential: Matrix3<f64>,
    /// How many matches were sampled.
    pub matches: usize,
    /// How many of them fit `essential`.
    pub inliers: usize,
    /// `inliers / matches`.
    pub inlier_ratio: f64,
    /// Median triangulation angle over the inliers, in degrees.
    pub median_angle_deg: f64,
    /// Median angle, in degrees, by which the inliers disagree with the best pure
    /// rotation. Near zero when the camera only turned and nothing about depth can be
    /// learnt.
    pub structure_deg: f64,
    /// The share of the inliers a single homography explains. Near 1 for a flat scene.
    pub homography_ratio: f64,
    /// The fraction of an 8 × 8 grid over the first photo that holds at least one inlier.
    pub spread: f64,
    /// The inlier threshold used, in photo pixels.
    pub threshold_px: f64,
    /// RANSAC iterations run.
    pub iterations: usize,
    /// Whether the pair is fit to use for registration.
    pub verdict: Verdict,
}

impl RelativePose {
    /// Whether registration can use this pair.
    pub fn is_usable(&self) -> bool {
        self.verdict == Verdict::Usable
    }
}

/// The outcome of triage.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Verdict {
    /// Fit to seed or anchor a registration.
    Usable,
    /// Not fit for registration. The pair's correspondences can still contribute to
    /// dense depth, where more than two views condition the result.
    Degenerate(Degeneracy),
}

/// Why a pair cannot anchor a registration.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Degeneracy {
    /// Too few correspondences were mapped to estimate anything.
    TooFewMatches {
        /// How many were sampled.
        found: usize,
        /// How many are needed.
        required: usize,
    },
    /// No sample produced a valid essential matrix.
    NoModel,
    /// Too few matches agree on one geometry.
    LowInlierRatio {
        /// The ratio found.
        ratio: f64,
        /// The ratio needed.
        required: f64,
    },
    /// A rotation alone explains the matches: the camera turned without moving.
    NoParallax {
        /// How far the matches stray from a pure rotation, in degrees.
        structure_deg: f64,
        /// How far they need to.
        required_deg: f64,
    },
    /// One plane explains nearly every match: the scene is flat, or the camera barely
    /// moved. Either way, the 8-point method cannot recover camera motion.
    Planar {
        /// The share of the inliers one homography explains.
        homography_ratio: f64,
        /// The largest share allowed.
        max_ratio: f64,
    },
    /// The cameras moved, but too little for depth to be measured precisely.
    SmallBaseline {
        /// The median triangulation angle, in degrees.
        median_angle_deg: f64,
        /// The angle needed.
        required_deg: f64,
    },
}

impl fmt::Display for Degeneracy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Degeneracy::TooFewMatches { found, required } => write!(
                f,
                "only {found} matches (need {required}); the photos share too little of the scene"
            ),
            Degeneracy::NoModel => f.write_str("no consistent camera motion fits the matches"),
            Degeneracy::LowInlierRatio { ratio, required } => write!(
                f,
                "only {:.0}% of matches agree on one camera motion (need {:.0}%)",
                ratio * 100.0,
                required * 100.0
            ),
            Degeneracy::NoParallax { structure_deg, required_deg } => write!(
                f,
                "the camera turned but barely moved ({structure_deg:.2}° of parallax, need {required_deg:.2}°); step sideways between shots"
            ),
            Degeneracy::Planar {
                homography_ratio,
                max_ratio,
            } => write!(
                f,
                "one plane explains {:.0}% of the matches (at most {:.0}% allowed): the scene is too flat, \
                 or the camera barely moved; step sideways between shots and include things at different distances",
                homography_ratio * 100.0,
                max_ratio * 100.0
            ),
            Degeneracy::SmallBaseline { median_angle_deg, required_deg } => write!(
                f,
                "the viewpoints are too close together ({median_angle_deg:.1}° between rays, need {required_deg:.1}°)"
            ),
        }
    }
}

/// Samples `lookup` on a grid with spacing `stride`, keeping every point that maps inside
/// a `width` × `height` photo.
pub fn sample_matches<L: PairLookup + ?Sized>(
    lookup: &L,
    (width, height): (usize, usize),
    stride: f32,
) -> Vec<Match> {
    let stride = stride.max(1.0);
    let (max_x, max_y) = (width as f32 - 1.0, height as f32 - 1.0);
    let inside = |p: PhotoPx| (0.0..=max_x).contains(&p.x()) && (0.0..=max_y).contains(&p.y());

    let mut matches = Vec::new();
    let mut y = stride / 2.0;
    while y <= max_y {
        let mut x = stride / 2.0;
        while x <= max_x {
            let a = PhotoPx::new(x, y);
            if let Some(b) = lookup.a_to_b(a).filter(|&b| inside(b)) {
                matches.push(Match { a, b });
            }
            x += stride;
        }
        y += stride;
    }
    matches
}

/// Estimates the relative pose of `pair` from its mapping and triages it.
///
/// # Errors
/// [`Degeneracy::TooFewMatches`] or [`Degeneracy::NoModel`] when there is no pose to
/// report at all. Every other degeneracy comes back as an `Ok` with a
/// [`Verdict::Degenerate`], since its metrics are still worth showing.
pub fn estimate<L: PairLookup + ?Sized>(
    pair: PairId,
    lookup: &L,
    intrinsics: &Intrinsics,
    size: (usize, usize),
    params: &Params,
    rng: &mut Rng,
) -> Result<RelativePose, Degeneracy> {
    let stride = params.stride.unwrap_or_else(|| lookup.native_stride());
    let matches = sample_matches(lookup, size, stride);
    let required = params.min_matches.max(8);
    if matches.len() < required {
        return Err(Degeneracy::TooFewMatches {
            found: matches.len(),
            required,
        });
    }

    let homogeneous = |n: Norm| Vector3::new(n.x(), n.y(), 1.0);
    let x1: Vec<_> = matches
        .iter()
        .map(|m| homogeneous(intrinsics.normalize(m.a)))
        .collect();
    let x2: Vec<_> = matches
        .iter()
        .map(|m| homogeneous(intrinsics.normalize(m.b)))
        .collect();

    let focal = (intrinsics.fx + intrinsics.fy) / 2.0;
    let precision_px = lookup.precision_px() as f64;
    let threshold_px = params.inlier_threshold as f64 * precision_px;
    let threshold_sq = (threshold_px / focal).powi(2);

    // RANSAC over minimal samples, stopping as soon as the best inlier ratio so far makes
    // an all-inlier sample likely enough.
    let n = matches.len();
    let mut best: Option<(Matrix3<f64>, usize)> = None;
    let mut sample = [0usize; 8];
    let mut needed = params.max_iterations;
    let mut iterations = 0;
    while iterations < needed {
        iterations += 1;
        rng.sample_distinct(n, &mut sample);
        let Some(e) = eight_point(&x1, &x2, &sample) else {
            continue;
        };
        let count = count_inliers(&e, &x1, &x2, threshold_sq);
        if best.map_or(true, |(_, c)| count > c) {
            best = Some((e, count));
            let ratio = count as f64 / n as f64;
            needed = ransac_iterations(ratio, 8, params.confidence).min(params.max_iterations);
        }
    }
    let (mut essential, _) = best.ok_or(Degeneracy::NoModel)?;

    // Refit on all inliers, which averages out the noise a minimal sample carries.
    let mut inliers = inlier_indices(&essential, &x1, &x2, threshold_sq);
    for _ in 0..3 {
        let Some(refit) = eight_point(&x1, &x2, &inliers) else {
            break;
        };
        let refit_inliers = inlier_indices(&refit, &x1, &x2, threshold_sq);
        if refit_inliers.len() < inliers.len() {
            break;
        }
        essential = refit;
        inliers = refit_inliers;
    }

    let pose = choose_pose(&essential, &x1, &x2, &inliers);
    let median_angle_deg = median_triangulation_angle(&pose, &x1, &x2, &inliers);
    let structure_deg = rotation_residual(&x1, &x2, &inliers);
    let inlier_ratio = inliers.len() as f64 / n as f64;
    let spread = spread(&matches, &inliers, size);
    let homography_ratio = homography_ratio(
        &x1,
        &x2,
        &inliers,
        params.homography_threshold * threshold_px / focal,
        params.confidence,
        rng,
    );

    let required_structure_deg = (params.min_structure * precision_px / focal).to_degrees();
    let verdict = if inlier_ratio < params.min_inlier_ratio {
        Verdict::Degenerate(Degeneracy::LowInlierRatio {
            ratio: inlier_ratio,
            required: params.min_inlier_ratio,
        })
    } else if structure_deg < required_structure_deg {
        Verdict::Degenerate(Degeneracy::NoParallax {
            structure_deg,
            required_deg: required_structure_deg,
        })
    } else if homography_ratio > params.max_homography_ratio {
        // One plane explains nearly every match. From two views, a flat scene and a camera
        // that barely moved look alike. The 8-point pose is unreliable either way, and so
        // is any angle computed from it, so this comes before the baseline check.
        Verdict::Degenerate(Degeneracy::Planar {
            homography_ratio,
            max_ratio: params.max_homography_ratio,
        })
    } else if median_angle_deg < params.min_median_angle_deg {
        Verdict::Degenerate(Degeneracy::SmallBaseline {
            median_angle_deg,
            required_deg: params.min_median_angle_deg,
        })
    } else {
        Verdict::Usable
    };

    Ok(RelativePose {
        pair,
        coverage: lookup.coverage(),
        pose,
        essential,
        matches: n,
        inliers: inliers.len(),
        inlier_ratio,
        median_angle_deg,
        structure_deg,
        homography_ratio,
        spread,
        threshold_px,
        iterations,
        verdict,
    })
}

/// How many RANSAC iterations give `confidence` of drawing one all-inlier sample of
/// `sample_size` when a fraction `inlier_ratio` of the data are inliers.
pub(crate) fn ransac_iterations(inlier_ratio: f64, sample_size: i32, confidence: f64) -> usize {
    let p = inlier_ratio.powi(sample_size);
    if p <= f64::EPSILON {
        usize::MAX
    } else if p >= 1.0 - f64::EPSILON {
        1
    } else {
        ((1.0 - confidence).ln() / (1.0 - p).ln()).ceil() as usize
    }
}

/// The normalized 8-point algorithm over the matches at `indices`, projected onto the
/// essential manifold. Least squares when given more than eight.
///
/// Points are homogeneous normalized camera coordinates. `None` if the points are
/// degenerate, for example all at the same position.
pub fn eight_point(
    x1: &[Vector3<f64>],
    x2: &[Vector3<f64>],
    indices: &[usize],
) -> Option<Matrix3<f64>> {
    if indices.len() < 8 {
        return None;
    }
    let t1 = hartley(indices.iter().map(|&i| &x1[i]))?;
    let t2 = hartley(indices.iter().map(|&i| &x2[i]))?;

    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for &i in indices {
        let p = t1 * x1[i];
        let q = t2 * x2[i];
        let a = SVector::<f64, 9>::from_column_slice(&[
            q.x * p.x,
            q.x * p.y,
            q.x,
            q.y * p.x,
            q.y * p.y,
            q.y,
            p.x,
            p.y,
            1.0,
        ]);
        ata += a * a.transpose();
    }

    let eigen = ata.symmetric_eigen();
    let v = eigen.eigenvectors.column(eigen.eigenvalues.imin());
    let entries: [f64; 9] = std::array::from_fn(|k| v[k]);
    let e = t2.transpose() * Matrix3::from_row_slice(&entries) * t1;
    project_essential(&e)
}

/// Hartley's normalizing similarity: centroid to the origin, mean distance √2.
fn hartley<'a>(points: impl Iterator<Item = &'a Vector3<f64>> + Clone) -> Option<Matrix3<f64>> {
    let count = points.clone().count() as f64;
    let (sx, sy) = points
        .clone()
        .fold((0.0, 0.0), |(sx, sy), p| (sx + p.x, sy + p.y));
    let (cx, cy) = (sx / count, sy / count);
    let mean = points
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / count;
    if mean.is_nan() || mean <= 1e-12 {
        return None;
    }
    let s = std::f64::consts::SQRT_2 / mean;
    Some(Matrix3::new(
        s,
        0.0,
        -s * cx, //
        0.0,
        s,
        -s * cy, //
        0.0,
        0.0,
        1.0,
    ))
}

/// The nearest essential matrix: two equal singular values and one zero.
fn project_essential(e: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = e.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let sv = svd.singular_values;
    let smallest = sv.imin();
    let s = (sv.sum() - sv[smallest]) / 2.0;
    let mut d = Vector3::repeat(s);
    d[smallest] = 0.0;
    let projected = u * Matrix3::from_diagonal(&d) * v_t;
    let norm = projected.norm();
    (norm > 0.0 && norm.is_finite()).then(|| projected / norm)
}

/// Squared Sampson distance of one match to the epipolar geometry, in normalized units.
fn sampson_sq(e: &Matrix3<f64>, e_t: &Matrix3<f64>, x1: &Vector3<f64>, x2: &Vector3<f64>) -> f64 {
    let ex1 = e * x1;
    let etx2 = e_t * x2;
    let c = x2.dot(&ex1);
    let d = ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y;
    if d > 0.0 {
        c * c / d
    } else {
        f64::INFINITY
    }
}

fn count_inliers(
    e: &Matrix3<f64>,
    x1: &[Vector3<f64>],
    x2: &[Vector3<f64>],
    threshold_sq: f64,
) -> usize {
    let e_t = e.transpose();
    x1.iter()
        .zip(x2)
        .filter(|(a, b)| sampson_sq(e, &e_t, a, b) < threshold_sq)
        .count()
}

fn inlier_indices(
    e: &Matrix3<f64>,
    x1: &[Vector3<f64>],
    x2: &[Vector3<f64>],
    threshold_sq: f64,
) -> Vec<usize> {
    let e_t = e.transpose();
    (0..x1.len())
        .filter(|&i| sampson_sq(e, &e_t, &x1[i], &x2[i]) < threshold_sq)
        .collect()
}

/// The four `(R, t)` an essential matrix decomposes into.
fn decompose(e: &Matrix3<f64>) -> Vec<Pose> {
    let svd = e.svd(true, true);
    let (Some(mut u), Some(mut v_t)) = (svd.u, svd.v_t) else {
        return Vec::new();
    };
    // Put the null direction last, whatever order the SVD returned.
    let smallest = svd.singular_values.imin();
    if smallest != 2 {
        u.swap_columns(smallest, 2);
        v_t.swap_rows(smallest, 2);
    }
    if u.determinant() < 0.0 {
        u = -u;
    }
    if v_t.determinant() < 0.0 {
        v_t = -v_t;
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let t = u.column(2).into_owned();
    let rotations = [u * w * v_t, u * w.transpose() * v_t];
    rotations
        .iter()
        .flat_map(|r| {
            // Products of rotations drift from orthonormal in the last bits.
            let mut rotation = Rotation3::from_matrix_unchecked(*r);
            rotation.renormalize();
            [t, -t].map(|translation| Pose {
                rotation,
                translation,
            })
        })
        .collect()
}

/// Of the four decompositions, the one that puts the most inliers in front of both
/// cameras.
fn choose_pose(
    e: &Matrix3<f64>,
    x1: &[Vector3<f64>],
    x2: &[Vector3<f64>],
    inliers: &[usize],
) -> Pose {
    let step = (inliers.len() / 500).max(1);
    decompose(e)
        .into_iter()
        .max_by_key(|pose| {
            inliers
                .iter()
                .step_by(step)
                .filter(|&&i| {
                    let observations = observations(pose, &x1[i], &x2[i]);
                    triangulate::dlt(&observations)
                        .is_some_and(|x| triangulate::in_front(&observations, &x))
                })
                .count()
        })
        .unwrap_or_else(Pose::identity)
}

fn observations(pose: &Pose, a: &Vector3<f64>, b: &Vector3<f64>) -> [(Pose, Norm); 2] {
    [
        (Pose::identity(), Norm::new(a.x, a.y)),
        (*pose, Norm::new(b.x, b.y)),
    ]
}

fn median_triangulation_angle(
    pose: &Pose,
    x1: &[Vector3<f64>],
    x2: &[Vector3<f64>],
    inliers: &[usize],
) -> f64 {
    let (centre_a, centre_b) = (Pose::identity().centre(), pose.centre());
    let step = (inliers.len() / 2000).max(1);
    let angles = inliers
        .iter()
        .step_by(step)
        .filter_map(|&i| {
            let observations = observations(pose, &x1[i], &x2[i]);
            let x = triangulate::dlt(&observations)?;
            triangulate::in_front(&observations, &x)
                .then(|| triangulate::triangulation_angle(&centre_a, &centre_b, &x).to_degrees())
        })
        .collect();
    median(angles)
}

/// The median angle, in degrees, between each inlier's second ray and its first ray
/// turned by the rotation that best aligns the two sets (Wahba's problem).
fn rotation_residual(x1: &[Vector3<f64>], x2: &[Vector3<f64>], inliers: &[usize]) -> f64 {
    let mut b = Matrix3::zeros();
    for &i in inliers {
        b += x2[i].normalize() * x1[i].normalize().transpose();
    }
    let svd = b.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return 0.0;
    };
    let mut d = Vector3::repeat(1.0);
    d[svd.singular_values.imin()] = (u * v_t).determinant().signum();
    let rotation = u * Matrix3::from_diagonal(&d) * v_t;

    let residuals = inliers
        .iter()
        .map(|&i| {
            let turned = rotation * x1[i].normalize();
            turned.angle(&x2[i].normalize()).to_degrees()
        })
        .collect();
    median(residuals)
}

/// The share of `inliers` a single homography explains, found by RANSAC over 4-point
/// samples. Close to 1 when the scene is flat, and also when the camera only turned.
fn homography_ratio(
    x1: &[Vector3<f64>],
    x2: &[Vector3<f64>],
    inliers: &[usize],
    threshold: f64,
    confidence: f64,
    rng: &mut Rng,
) -> f64 {
    const MAX_ITERATIONS: usize = 500;
    let step = (inliers.len() / 2000).max(1);
    let subset: Vec<usize> = inliers.iter().step_by(step).copied().collect();
    if subset.len() < 8 {
        return 0.0;
    }
    let threshold_sq = threshold * threshold;
    let mut best = 0;
    let mut sample = [0usize; 4];
    let mut needed = MAX_ITERATIONS;
    let mut iterations = 0;
    while iterations < needed {
        iterations += 1;
        rng.sample_distinct(subset.len(), &mut sample);
        let chosen = sample.map(|k| subset[k]);
        let Some(h) = homography(x1, x2, &chosen) else {
            continue;
        };
        let count = subset
            .iter()
            .filter(|&&i| transfer_sq(&h, &x1[i], &x2[i]) < threshold_sq)
            .count();
        if count > best {
            best = count;
            let ratio = count as f64 / subset.len() as f64;
            needed = ransac_iterations(ratio, 4, confidence).min(MAX_ITERATIONS);
        }
    }
    best as f64 / subset.len() as f64
}

/// The normalized DLT homography taking the points of `x1` at `indices` onto `x2`.
fn homography(x1: &[Vector3<f64>], x2: &[Vector3<f64>], indices: &[usize]) -> Option<Matrix3<f64>> {
    let t1 = hartley(indices.iter().map(|&i| &x1[i]))?;
    let t2 = hartley(indices.iter().map(|&i| &x2[i]))?;
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for &i in indices {
        let p = t1 * x1[i];
        let q = t2 * x2[i];
        let r1 = SVector::<f64, 9>::from_column_slice(&[
            0.0,
            0.0,
            0.0,
            -p.x,
            -p.y,
            -1.0,
            q.y * p.x,
            q.y * p.y,
            q.y,
        ]);
        let r2 = SVector::<f64, 9>::from_column_slice(&[
            p.x,
            p.y,
            1.0,
            0.0,
            0.0,
            0.0,
            -q.x * p.x,
            -q.x * p.y,
            -q.x,
        ]);
        ata += r1 * r1.transpose() + r2 * r2.transpose();
    }
    let eigen = ata.symmetric_eigen();
    let v = eigen.eigenvectors.column(eigen.eigenvalues.imin());
    let entries: [f64; 9] = std::array::from_fn(|k| v[k]);
    let h = t2.try_inverse()? * Matrix3::from_row_slice(&entries) * t1;
    h.iter().all(|x| x.is_finite()).then_some(h)
}

/// Squared distance between `H · a` and `b`, in normalized units.
fn transfer_sq(h: &Matrix3<f64>, a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let y = h * a;
    if y.z.abs() < 1e-12 {
        return f64::INFINITY;
    }
    (y.x / y.z - b.x).powi(2) + (y.y / y.z - b.y).powi(2)
}

fn spread(matches: &[Match], inliers: &[usize], (width, height): (usize, usize)) -> f64 {
    const CELLS: usize = 8;
    let mut occupied = [false; CELLS * CELLS];
    for &i in inliers {
        let a = matches[i].a;
        let cx = ((a.x() as f64 / width as f64) * CELLS as f64) as usize;
        let cy = ((a.y() as f64 / height as f64) * CELLS as f64) as usize;
        occupied[cy.min(CELLS - 1) * CELLS + cx.min(CELLS - 1)] = true;
    }
    occupied.iter().filter(|&&o| o).count() as f64 / (CELLS * CELLS) as f64
}

pub(crate) fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let (_, m, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ransac_iterations_track_the_inlier_ratio() {
        assert_eq!(ransac_iterations(1.0, 8, 0.999), 1);
        assert!(ransac_iterations(0.9, 8, 0.999) < ransac_iterations(0.6, 8, 0.999));
        assert_eq!(ransac_iterations(0.0, 8, 0.999), usize::MAX);
    }

    #[test]
    fn projected_essential_matrix_has_the_right_singular_values() {
        let e = Matrix3::new(0.3, -1.2, 0.5, 2.0, 0.1, -0.7, 0.4, 0.9, 1.1);
        let p = project_essential(&e).unwrap();
        let mut sv: Vec<f64> = p
            .svd(false, false)
            .singular_values
            .iter()
            .copied()
            .collect();
        sv.sort_by(f64::total_cmp);
        assert!(sv[0].abs() < 1e-12);
        assert!((sv[1] - sv[2]).abs() < 1e-12);
    }
}
