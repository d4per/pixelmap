//! Per-cell diagnostics of a finished map, for experiments on telling the wrong answers
//! of a mapping from the right ones. The experiments themselves live in the
//! `filter_lab` crate; this module only reaches the parts of the scorer they cannot.
//!
//! Compiled only under the `bench` feature.

use crate::affine_transform::AffineTransform;
use crate::correspondence_scoring::{CorrespondenceScoring, Samples};
use crate::dense_photo_map::DensePhotoMap;

/// The local Jacobian of a map at one grid node, as `[dX/dx, dX/dy, dY/dx, dY/dy]`:
/// the same layout as an [`AffineTransform`]'s `a11, a12, a21, a22`.
pub type Jacobian = [f32; 4];

/// What the scorer sees at one cell of a finished map.
#[derive(Clone, Copy, Debug)]
pub struct CellProbe {
    /// The score at the mapped position: the summed per-channel variance of the colour
    /// differences over the disc. Lower is better.
    pub score: f32,
    /// The scores one pixel to the `[-x, +x, -y, +y]` of the mapped position.
    pub taps: [f32; 4],
    /// The summed per-channel variance of `photo1`'s colours over the same samples, in the
    /// units of `score`. A textureless disc has a small one.
    pub texture: f32,
    /// The share of the disc `photo1` offered that landed inside `photo2`.
    pub compared: f32,
}

/// The Jacobian of `map` at every grid node, by central differences where both
/// neighbours along an axis are mapped and one-sided ones where only one is. `None`
/// where the node is unmapped, or has no mapped neighbour along an axis.
pub fn jacobians(map: &DensePhotoMap) -> Vec<Option<Jacobian>> {
    map.jacobians()
}

/// Scores every mapped node of `map` against the map's own (working-resolution) photos
/// over a disc of `radius`, warped by the node's Jacobian (identity where it has none).
/// `None` for unmapped nodes and for nodes whose disc mostly falls outside `photo2`.
pub fn probe_cells(map: &DensePhotoMap, radius: usize) -> Vec<Option<CellProbe>> {
    let scorer = CorrespondenceScoring::new(map.photo1().clone(), map.photo2().clone(), radius as isize);
    let mut samples = Samples::new(radius);
    let (gw, gh) = map.grid_dimensions();
    let c = map.grid_cell_size();
    let jac = map.jacobians();
    let mut out = Vec::with_capacity(gw * gh);
    for y in 0..gh {
        for x in 0..gw {
            let (tx, ty) = map.grid_coordinates(x, y);
            if tx.is_nan() || ty.is_nan() {
                out.push(None);
                continue;
            }
            let [a11, a12, a21, a22] = jac[y * gw + x].unwrap_or([1.0, 0.0, 0.0, 1.0]);
            let cm = AffineTransform {
                origin_x: (x * c) as u16,
                origin_y: (y * c) as u16,
                translate_x: tx,
                translate_y: ty,
                a11,
                a12,
                a21,
                a22,
            };
            out.push(scorer.probe(&cm, &mut samples).map(|(s, texture, compared)| CellProbe {
                score: s[0],
                taps: [s[1], s[2], s[3], s[4]],
                texture,
                compared,
            }));
        }
    }
    out
}
