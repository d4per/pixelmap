//! The local-consistency filter: drops the answers of a finished map that disagree with
//! the answers around them; see [`LocalConsistency`].

use crate::dense_photo_map::DensePhotoMap;

/// Rejects mapped cells whose answer is out of line with their neighbours', after the
/// round-trip check. Off unless [`crate::Builder::local_consistency`] turns it on.
///
/// The round trip catches a cell whose two directions disagree. It cannot catch a
/// region that is *consistently* wrong in both directions, which is what the relaxation
/// produces at occlusion boundaries, on repetitive texture and where the mapping leaves
/// the frame. Such regions still give themselves away locally: their flow jumps away from
/// that of the cells around them, or the local affine transform the map implies, its
/// Jacobian, is stretched far along one axis or bent away from its neighbours'.
///
/// A cell is dropped if **any** of the three tests below fails. Each one compares a cell
/// with its own neighbourhood, never with an absolute expectation, because a legitimate
/// mapping can scale, shear or move a long way: on strong viewpoint changes an absolute
/// limit on the Jacobian rejected a fifth or more of the correct answers. A test that
/// cannot be evaluated at a cell (no mapped neighbour to differentiate against) does not
/// count against it.
///
/// The defaults were chosen on the training splits of KITTI 2015, MPI-Sintel, ETH3D and
/// HPatches (`High`, fill off), as one setting for all four: there they drop 24–37% of the
/// grid cells whose answer is more than 3 px wrong, and 0.7–1.8% of the rest.
///
/// Counted in pixels the cost is higher, because a pixel is answered only when all four
/// corners of its cell are: on the same benchmarks the share of pixels answered fell by
/// 4–10 points (KITTI 77% → 69%, ETH3D 85% → 76%), while the share of the answers within
/// 3 px of the truth rose by 2–5 points (KITTI 88% → 93%). Lowering
/// [`crate::Builder::max_round_trip_error`] trades coverage for precision at about the
/// same rate, so try that first. The two catch partly different errors and stack: with
/// a round trip of `0.7` the filter still raised that share by 0.5–1.4 points, for
/// 1–2 points of coverage.
///
/// All three are measured at the final working resolution; see
/// [`crate::Correspondence::working_scale`].
#[derive(Copy, Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LocalConsistency {
    /// The most a cell's flow may differ from the median flow of its 3×3 neighbourhood,
    /// in working pixels. Default `3.2`.
    pub max_flow_deviation: f32,
    /// The most a cell's Jacobian may differ from the element-wise median Jacobian of its
    /// 5×5 neighbourhood, as a Frobenius norm. Default `1.2`.
    pub max_jacobian_deviation: f32,
    /// The most the Jacobian may stretch along one axis relative to the other: the ratio
    /// of its larger singular value to its smaller one. Default `5.0`.
    pub max_anisotropy: f32,
}

impl Default for LocalConsistency {
    fn default() -> Self {
        LocalConsistency {
            max_flow_deviation: 3.2,
            max_jacobian_deviation: 1.2,
            max_anisotropy: 5.0,
        }
    }
}

impl LocalConsistency {
    /// The defaults; see [`LocalConsistency`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets [`Self::max_flow_deviation`].
    pub fn max_flow_deviation(mut self, px: f32) -> Self {
        self.max_flow_deviation = px;
        self
    }

    /// Sets [`Self::max_jacobian_deviation`].
    pub fn max_jacobian_deviation(mut self, deviation: f32) -> Self {
        self.max_jacobian_deviation = deviation;
        self
    }

    /// Sets [`Self::max_anisotropy`].
    pub fn max_anisotropy(mut self, ratio: f32) -> Self {
        self.max_anisotropy = ratio;
        self
    }

    /// Empties the cells of `map` that fail a test. Every cell is judged against the map
    /// as it was before any was removed, so the order of the cells does not matter.
    pub(crate) fn apply(&self, map: &mut DensePhotoMap) {
        let rejected = self.rejected(map);
        let (gw, _) = map.grid_dimensions();
        for (i, _) in rejected.iter().enumerate().filter(|(_, r)| **r) {
            map.set_grid_coordinates(i % gw, i / gw, f32::NAN, f32::NAN);
        }
    }

    /// Which cells of `map` fail a test, row-major over the grid.
    fn rejected(&self, map: &DensePhotoMap) -> Vec<bool> {
        let (gw, gh) = map.grid_dimensions();
        let c = map.grid_cell_size() as f32;
        let flow = |x: usize, y: usize| -> Option<(f32, f32)> {
            let (tx, ty) = map.grid_coordinates(x, y);
            (!tx.is_nan() && !ty.is_nan()).then_some((tx - x as f32 * c, ty - y as f32 * c))
        };
        let jac = map.jacobians();
        // ln of the ratio, as the lab measured it, so a degenerate Jacobian (σ_min = 0)
        // compares as a very large ratio rather than as a division by zero.
        let max_log_aniso = self.max_anisotropy.max(1.0).ln();

        let mut out = vec![false; gw * gh];
        let mut fx = Vec::with_capacity(9);
        let mut fy = Vec::with_capacity(9);
        let mut jbuf = Vec::with_capacity(25);
        for y in 0..gh {
            for x in 0..gw {
                let Some(f0) = flow(x, y) else { continue };
                let i = y * gw + x;

                fx.clear();
                fy.clear();
                for yy in y.saturating_sub(1)..(y + 2).min(gh) {
                    for xx in x.saturating_sub(1)..(x + 2).min(gw) {
                        if let Some(n) = flow(xx, yy) {
                            fx.push(n.0);
                            fy.push(n.1);
                        }
                    }
                }
                let flow_dev = (f0.0 - median(&mut fx)).hypot(f0.1 - median(&mut fy));
                if flow_dev > self.max_flow_deviation {
                    out[i] = true;
                    continue;
                }

                let Some(j) = jac[i] else { continue };
                if log_anisotropy(&j) > max_log_aniso {
                    out[i] = true;
                    continue;
                }
                let mut dev2 = 0.0f32;
                for k in 0..4 {
                    jbuf.clear();
                    for yy in y.saturating_sub(2)..(y + 3).min(gh) {
                        for xx in x.saturating_sub(2)..(x + 3).min(gw) {
                            if let Some(n) = jac[yy * gw + xx] {
                                jbuf.push(n[k]);
                            }
                        }
                    }
                    dev2 += (j[k] - median(&mut jbuf)).powi(2);
                }
                if dev2.sqrt() > self.max_jacobian_deviation {
                    out[i] = true;
                }
            }
        }
        out
    }
}

/// `ln(σ_max / σ_min)` of a 2×2 matrix `[a11, a12, a21, a22]`, with both singular values
/// floored at `1e-6`.
pub(crate) fn log_anisotropy(j: &[f32; 4]) -> f32 {
    let [a, b, c, d] = *j;
    let s = a * a + b * b + c * c + d * d;
    let det = a * d - b * c;
    let disc = (s * s - 4.0 * det * det).max(0.0).sqrt();
    let smax = ((s + disc) / 2.0).max(0.0).sqrt();
    let smin = if smax > 0.0 { det.abs() / smax } else { 0.0 };
    (smax.max(1e-6) / smin.max(1e-6)).ln()
}

/// The median of a non-empty slice, the mean of the middle two for an even length.
fn median(v: &mut [f32]) -> f32 {
    v.sort_unstable_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photo::Photo;
    use std::sync::Arc;

    const GW: usize = 20;
    const GH: usize = 16;
    const C: usize = 5;

    /// A map over a 100×80 photo pair whose every cell moves by `(3, -2)`.
    fn shifted() -> DensePhotoMap {
        let photo = Arc::new(Photo::from_rgba(GW * C, GH * C, vec![128; GW * C * GH * C * 4]).unwrap());
        let mut map = DensePhotoMap::with_cell_size(photo.clone(), photo, GW, GH, C);
        for y in 0..GH {
            for x in 0..GW {
                map.set_grid_coordinates(x, y, (x * C) as f32 + 3.0, (y * C) as f32 - 2.0);
            }
        }
        map
    }

    fn mapped(map: &DensePhotoMap) -> usize {
        (0..GH)
            .flat_map(|y| (0..GW).map(move |x| (x, y)))
            .filter(|&(x, y)| !map.grid_coordinates(x, y).0.is_nan())
            .count()
    }

    #[test]
    fn a_smooth_map_keeps_every_cell() {
        let mut map = shifted();
        LocalConsistency::default().apply(&mut map);
        assert_eq!(mapped(&map), GW * GH);
    }

    #[test]
    fn a_smooth_zoom_keeps_every_cell() {
        // Strong but uniform scaling: an absolute limit on the Jacobian would reject it.
        let mut map = shifted();
        for y in 0..GH {
            for x in 0..GW {
                map.set_grid_coordinates(x, y, (x * C) as f32 * 0.3, (y * C) as f32 * 0.3);
            }
        }
        LocalConsistency::default().apply(&mut map);
        assert_eq!(mapped(&map), GW * GH);
    }

    #[test]
    fn a_cell_that_jumps_away_is_dropped() {
        let mut map = shifted();
        let (x, y) = (10, 8);
        map.set_grid_coordinates(x, y, (x * C) as f32 + 3.0 + 20.0, (y * C) as f32 - 2.0);
        LocalConsistency::default().apply(&mut map);
        assert!(map.grid_coordinates(x, y).0.is_nan(), "the outlier itself goes");
        assert!(!map.grid_coordinates(0, 0).0.is_nan(), "far away nothing changes");
    }

    #[test]
    fn a_column_squashed_against_an_edge_is_dropped() {
        // The last columns all map onto nearly the same x: the pile-up at a border.
        let mut map = shifted();
        for y in 0..GH {
            for x in GW - 4..GW {
                let t = (GW - 4) as f32 * C as f32 + 3.0 + 0.05 * (x - (GW - 4)) as f32;
                map.set_grid_coordinates(x, y, t, (y * C) as f32 - 2.0);
            }
        }
        LocalConsistency::default().apply(&mut map);
        let dropped = (0..GH).filter(|&y| map.grid_coordinates(GW - 2, y).0.is_nan()).count();
        assert_eq!(dropped, GH, "the squashed column is gone");
        assert!(!map.grid_coordinates(2, 8).0.is_nan());
    }

    #[test]
    fn the_result_does_not_depend_on_the_order_cells_are_removed_in() {
        // Two neighbouring outliers: had the first been removed before the second was
        // judged, the second would see a different neighbourhood.
        let mut map = shifted();
        map.set_grid_coordinates(5, 5, 25.0 + 30.0, 23.0);
        map.set_grid_coordinates(6, 5, 30.0 + 30.0, 23.0);
        let filter = LocalConsistency::default();
        let expected = filter.rejected(&map);
        filter.apply(&mut map);
        for (i, r) in expected.iter().enumerate() {
            assert_eq!(map.grid_coordinates(i % GW, i / GW).0.is_nan(), *r);
        }
    }

    #[test]
    fn anisotropy_of_a_rotation_is_zero() {
        let (s, c) = 0.7f32.sin_cos();
        assert!(log_anisotropy(&[c, -s, s, c]).abs() < 1e-5);
        assert!((log_anisotropy(&[4.0, 0.0, 0.0, 1.0]) - 4.0f32.ln()).abs() < 1e-5);
    }
}
