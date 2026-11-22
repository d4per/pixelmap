//! A fundamental matrix by normalized 8-point inside RANSAC, and the Sampson distance
//! to it. Pixel coordinates in, pixel distances out.

use nalgebra::{Matrix3, SMatrix, Vector3};

type Pt = [f64; 2];

/// Hartley normalization: centre on the centroid, scale the mean distance to √2.
fn normalizer(pts: &[Pt], idx: &[usize]) -> Matrix3<f64> {
    let n = idx.len() as f64;
    let (mut cx, mut cy) = (0.0, 0.0);
    for &i in idx {
        cx += pts[i][0];
        cy += pts[i][1];
    }
    cx /= n;
    cy /= n;
    let mean = idx.iter().map(|&i| (pts[i][0] - cx).hypot(pts[i][1] - cy)).sum::<f64>() / n;
    let s = if mean > 1e-12 { std::f64::consts::SQRT_2 / mean } else { 1.0 };
    Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0)
}

/// The rank-2 F with `x2ᵀ F x1 = 0` that best fits the given matches (≥ 8).
fn fit(a: &[Pt], b: &[Pt], idx: &[usize]) -> Option<Matrix3<f64>> {
    let ta = normalizer(a, idx);
    let tb = normalizer(b, idx);
    let rows = idx.len().max(9);
    let mut m = nalgebra::DMatrix::<f64>::zeros(rows, 9);
    for (r, &i) in idx.iter().enumerate() {
        let p = ta * Vector3::new(a[i][0], a[i][1], 1.0);
        let q = tb * Vector3::new(b[i][0], b[i][1], 1.0);
        let row = [
            q[0] * p[0], q[0] * p[1], q[0],
            q[1] * p[0], q[1] * p[1], q[1],
            p[0], p[1], 1.0,
        ];
        for (k, v) in row.iter().enumerate() {
            m[(r, k)] = *v;
        }
    }
    let svd = m.svd(false, true);
    let vt = svd.v_t?;
    // The right singular vector of the smallest singular value.
    let (k, _) = svd
        .singular_values
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.total_cmp(y.1))?;
    let f = vt.row(k);
    let f = Matrix3::new(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);
    let svd = f.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut s = svd.singular_values;
    // nalgebra sorts singular values in descending order for SVD of a static matrix.
    let min = (0..3).min_by(|&x, &y| s[x].total_cmp(&s[y]))?;
    s[min] = 0.0;
    let f = u * SMatrix::<f64, 3, 3>::from_diagonal(&s) * vt;
    Some(tb.transpose() * f * ta)
}

/// Sampson distance of the match `a → b` to `f`, in pixels.
pub fn sampson(f: &Matrix3<f64>, a: Pt, b: Pt) -> f64 {
    let p = Vector3::new(a[0], a[1], 1.0);
    let q = Vector3::new(b[0], b[1], 1.0);
    let fp = f * p;
    let ftq = f.transpose() * q;
    let e = q.dot(&fp);
    let d = fp[0] * fp[0] + fp[1] * fp[1] + ftq[0] * ftq[0] + ftq[1] * ftq[1];
    if d > 0.0 {
        (e * e / d).sqrt()
    } else {
        f64::INFINITY
    }
}

/// RANSAC over at most 4000 of the matches (evenly subsampled), then a refit on every
/// inlier of the best sample. `None` with fewer than 16 matches.
pub fn ransac(a: &[Pt], b: &[Pt], threshold: f64, iterations: usize, seed: u64) -> Option<Matrix3<f64>> {
    if a.len() < 16 {
        return None;
    }
    let stride = a.len().div_ceil(4000);
    let pool: Vec<usize> = (0..a.len()).step_by(stride).collect();
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut best: Option<(usize, Matrix3<f64>)> = None;
    for _ in 0..iterations {
        let mut sample = Vec::with_capacity(8);
        while sample.len() < 8 {
            let i = pool[(next() % pool.len() as u64) as usize];
            if !sample.contains(&i) {
                sample.push(i);
            }
        }
        let Some(f) = fit(a, b, &sample) else { continue };
        let count = pool.iter().filter(|&&i| sampson(&f, a[i], b[i]) < threshold).count();
        if best.as_ref().is_none_or(|(c, _)| count > *c) {
            best = Some((count, f));
        }
    }
    let (_, f) = best?;
    let inliers: Vec<usize> = pool.iter().copied().filter(|&i| sampson(&f, a[i], b[i]) < threshold).collect();
    if inliers.len() >= 8 {
        fit(a, b, &inliers).or(Some(f))
    } else {
        Some(f)
    }
}
