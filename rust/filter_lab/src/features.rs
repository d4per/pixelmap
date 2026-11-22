//! Candidate signals that a grid node's answer is wrong, one value per node of the final
//! forward map. Every feature is oriented so that **higher means more suspect**. `NaN`
//! means the feature could not be computed there (no neighbours to differentiate, no
//! coarser answer to compare with, ...); the evaluation ranks those as most suspect.

use pixelmap::filter_bench::{self, Jacobian};
use pixelmap::DensePhotoMap;

use crate::fundamental;

pub const NAMES: [&str; 16] = [
    "round_trip", // distance a round trip through both maps lands from the start, px
    "fold",       // -det J: positive where the map mirrors
    "compress",   // -ln σ_min(J)
    "aniso",      // ln(σ_max / σ_min)
    "j_dev",      // |J - median J of the 5x5 neighbourhood|_F
    "j_round",    // |J_bwd(f(x)) J_fwd(x) - I|_F
    "border",     // -(distance of the target from photo2's edge), px
    "island",     // -ln(size of the flow-connected component)
    "flow_dev",   // |flow - median flow of the 3x3 neighbourhood|, px
    "prev_width", // |answer - answer at the previous working width|, px
    "sampson",    // Sampson distance to a RANSAC fundamental matrix, px
    "score",      // the solver's residual at the answer
    "score_rel",  // residual / texture of the disc
    "flat",       // -(least curvature of the score) / texture
    "not_min",    // residual - best of the four ±1 px taps (> 0: not at a minimum)
    "cut",        // share of the disc that fell outside photo2
];
pub const N: usize = NAMES.len();

fn sv(j: &Jacobian) -> (f32, f32) {
    let [a, b, c, d] = *j;
    let s = a * a + b * b + c * c + d * d;
    let det = a * d - b * c;
    let disc = (s * s - 4.0 * det * det).max(0.0).sqrt();
    let smax = ((s + disc) / 2.0).max(0.0).sqrt();
    let smin = if smax > 0.0 { det.abs() / smax } else { 0.0 };
    (smax, smin)
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Features of every node of `fwd` (NaN rows for unmapped nodes).
///
/// `prev` is the cleaned forward map of the last step at the previous working width, if
/// the schedule had one; `radius` is the disc radius of the final step.
pub fn compute(fwd: &DensePhotoMap, bwd: &DensePhotoMap, prev: Option<&DensePhotoMap>, radius: usize) -> Vec<[f32; N]> {
    let (gw, gh) = fwd.grid_dimensions();
    let c = fwd.grid_cell_size() as f32;
    let (w2, h2) = (fwd.photo2().width() as f32, fwd.photo2().height() as f32);
    let at = |x: usize, y: usize| -> Option<(f32, f32)> {
        let p = fwd.grid_coordinates(x, y);
        (!p.0.is_nan() && !p.1.is_nan()).then_some(p)
    };
    let flow = |x: usize, y: usize| at(x, y).map(|(tx, ty)| (tx - x as f32 * c, ty - y as f32 * c));

    let jf = filter_bench::jacobians(fwd);
    let jb = filter_bench::jacobians(bwd);
    let (bgw, bgh) = bwd.grid_dimensions();
    let bc = bwd.grid_cell_size() as f32;
    let probes = filter_bench::probe_cells(fwd, radius);

    // Flow-connected components over 4-neighbours: two mapped nodes are joined when
    // their flows differ by at most one cell.
    let mut comp = vec![usize::MAX; gw * gh];
    let mut sizes = Vec::new();
    for start in 0..gw * gh {
        if comp[start] != usize::MAX || flow(start % gw, start / gw).is_none() {
            continue;
        }
        let id = sizes.len();
        let mut stack = vec![start];
        comp[start] = id;
        let mut size = 0usize;
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = (i % gw, i / gw);
            let f0 = flow(x, y).unwrap();
            let mut visit = |nx: usize, ny: usize| {
                let j = ny * gw + nx;
                if comp[j] == usize::MAX {
                    if let Some(f1) = flow(nx, ny) {
                        if (f1.0 - f0.0).hypot(f1.1 - f0.1) <= c {
                            comp[j] = id;
                            stack.push(j);
                        }
                    }
                }
            };
            if x > 0 { visit(x - 1, y); }
            if x + 1 < gw { visit(x + 1, y); }
            if y > 0 { visit(x, y - 1); }
            if y + 1 < gh { visit(x, y + 1); }
        }
        sizes.push(size);
    }

    // Fundamental matrix over the mapped nodes, in working pixels.
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for y in 0..gh {
        for x in 0..gw {
            if let Some(t) = at(x, y) {
                src.push([x as f64 * c as f64, y as f64 * c as f64]);
                dst.push([t.0 as f64, t.1 as f64]);
            }
        }
    }
    let f = fundamental::ransac(&src, &dst, 1.0, 1000, 0x5eed);

    let prev_scale = prev.map(|p| p.dimensions().0 as f32 / fwd.dimensions().0 as f32);

    let mut out = vec![[f32::NAN; N]; gw * gh];
    let mut jbuf = Vec::with_capacity(25);
    let mut fx = Vec::with_capacity(9);
    let mut fy = Vec::with_capacity(9);
    for y in 0..gh {
        for x in 0..gw {
            let Some((tx, ty)) = at(x, y) else { continue };
            let i = y * gw + x;
            let row = &mut out[i];
            let (sx, sy) = (x as f32 * c, y as f32 * c);

            let back = bwd.map_photo_pixel(tx, ty);
            row[0] = (back.0 - sx).hypot(back.1 - sy);

            if let Some(j) = jf[i] {
                let (smax, smin) = sv(&j);
                row[1] = -(j[0] * j[3] - j[1] * j[2]);
                row[2] = -smin.max(1e-6).ln();
                row[3] = (smax.max(1e-6) / smin.max(1e-6)).ln();

                for k in 0..4 {
                    jbuf.clear();
                    for yy in y.saturating_sub(2)..(y + 3).min(gh) {
                        for xx in x.saturating_sub(2)..(x + 3).min(gw) {
                            if let Some(n) = jf[yy * gw + xx] {
                                jbuf.push(n[k]);
                            }
                        }
                    }
                    let m = median(&mut jbuf);
                    row[4] = if k == 0 { 0.0 } else { row[4] } + (j[k] - m).powi(2);
                }
                row[4] = row[4].sqrt();

                let (bx, by) = ((tx / bc).round(), (ty / bc).round());
                if bx >= 0.0 && by >= 0.0 && (bx as usize) < bgw && (by as usize) < bgh {
                    if let Some(b) = jb[by as usize * bgw + bx as usize] {
                        // b · j - I, with both in [a11, a12, a21, a22] layout.
                        let m = [
                            b[0] * j[0] + b[1] * j[2] - 1.0,
                            b[0] * j[1] + b[1] * j[3],
                            b[2] * j[0] + b[3] * j[2],
                            b[2] * j[1] + b[3] * j[3] - 1.0,
                        ];
                        row[5] = m.iter().map(|v| v * v).sum::<f32>().sqrt();
                    }
                }
            }

            row[6] = -(tx.min(ty).min(w2 - 1.0 - tx).min(h2 - 1.0 - ty));
            row[7] = -(sizes[comp[i]] as f32).ln();

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
            let f0 = flow(x, y).unwrap();
            row[8] = (f0.0 - median(&mut fx)).hypot(f0.1 - median(&mut fy));

            if let (Some(p), Some(s)) = (prev, prev_scale) {
                if let Some((px, py)) = p.lookup(sx * s, sy * s) {
                    row[9] = (px / s - tx).hypot(py / s - ty);
                }
            }

            if let Some(f) = &f {
                row[10] = fundamental::sampson(f, [sx as f64, sy as f64], [tx as f64, ty as f64]) as f32;
            }

            if let Some(p) = probes[i] {
                let t = p.texture + 1.0;
                let cx = 0.5 * (p.taps[0] + p.taps[1]) - p.score;
                let cy = 0.5 * (p.taps[2] + p.taps[3]) - p.score;
                row[11] = p.score;
                row[12] = p.score / t;
                row[13] = -cx.min(cy) / t;
                row[14] = p.score - p.taps.iter().copied().fold(f32::MAX, f32::min);
                row[15] = 1.0 - p.compared;
            }
        }
    }
    out
}
