//! Stage 7: depth maps fused into one surface.
//!
//! **Truncated signed distance fusion** into a sparse voxel grid, followed by **surface
//! nets** to extract the zero level.
//!
//! TSDF fusion merges several overlapping partial surfaces natively, and it averages
//! conflicting observations instead of layering them. Only blocks of 8³ voxels near an
//! observed surface are allocated. The voxel size follows the spacing of the depth
//! samples, and it grows automatically if the grid would exceed a voxel budget.
//!
//! Each voxel is projected into every depth map, and its signed distance to the surface that
//! map sees is truncated to a band of a few voxels. Positive is in front of the surface, on
//! the camera's side; far behind a surface is left unobserved. The distance is measured along
//! the surface normal, estimated from the depth map, not along the camera ray: along the ray,
//! a surface seen obliquely, such as a floor or a step tread, gets a band too thin to hold a
//! surface and comes out full of holes. Each observation is weighted by how precise its
//! depth is.
//!
//! Surface nets place one vertex inside every cell whose corners change sign, at the mean
//! of the edge crossings, and join the cells around each crossing edge with a quad. This
//! gives the same kind of mesh as marching cubes, somewhat smoother and with fewer
//! triangles, without a case table. Crossings between two saturated values, a jump from
//! one side of the band to the other, are artefacts of a depth discontinuity and do not
//! count.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

use nalgebra::{Point3, Vector3};

use crate::calib::Intrinsics;
use crate::depth::DepthMap;
use crate::error::Error;
use crate::event::{report, silent, Event, Flow, Stage};
use crate::mesh::Mesh;
use crate::pose::Pose;
use crate::twoview::median;
use crate::types::Norm;

/// The smallest share of the triangles the largest connected piece must hold.
pub const MIN_LARGEST_COMPONENT: f64 = 0.3;

/// Tuning for [`fuse`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Voxel size as a multiple of the median spacing between depth samples on the
    /// surface.
    pub voxel_factor: f64,
    /// Half-width of the truncation band, in voxels.
    pub truncation: f64,
    /// The most voxels to allocate. The voxel size grows until the grid fits.
    pub max_voxels: usize,
    /// The total observation weight a voxel needs to take part in the surface. Each depth
    /// adds up to 1, less the less precise it is.
    pub min_weight: f32,
    /// Connected pieces with less than this fraction of the triangles are dropped.
    pub min_component_fraction: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            voxel_factor: 2.0,
            truncation: 4.0,
            max_voxels: 16_000_000,
            min_weight: 0.1,
            min_component_fraction: 0.02,
        }
    }
}

/// The fused surface.
#[derive(Clone, Debug)]
pub struct Fused {
    /// The mesh, with normals.
    pub mesh: Mesh,
    /// The voxel edge length used, in reconstruction units.
    pub voxel_size: f64,
    /// Allocated blocks of 8³ voxels.
    pub blocks: usize,
    /// The share of triangles in the largest connected piece, before small pieces were
    /// dropped.
    pub largest_component: f64,
    /// Triangles dropped with the small pieces.
    pub dropped_triangles: usize,
}

const BLOCK: i64 = 8;
const VOXELS_PER_BLOCK: usize = 512;

#[derive(Copy, Clone, Debug, Default)]
struct Voxel {
    distance: f32,
    weight: f32,
}

/// A fixed hasher, so that nothing about the run depends on a random seed.
type Deterministic = BuildHasherDefault<DefaultHasher>;
type Blocks = HashMap<[i64; 3], Box<[Voxel; VOXELS_PER_BLOCK]>, Deterministic>;

/// Fuses `maps`, whose views are posed in `cameras`, into a mesh.
///
/// # Errors
/// [`Error::EmptyMesh`] if there is nothing to fuse or no surface comes out;
/// [`Error::FragmentedMesh`] if the largest connected piece is too small a share of the
/// whole.
pub fn fuse(
    maps: &[DepthMap],
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    params: &Params,
) -> Result<Fused, Error> {
    fuse_with_progress(maps, cameras, intrinsics, params, &mut silent)
}

/// [`fuse`], reporting each phase through `on_event`.
///
/// # Errors
/// As [`fuse`], plus [`Error::Cancelled`] if `on_event` asks the run to stop.
pub fn fuse_with_progress(
    maps: &[DepthMap],
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    params: &Params,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<Fused, Error> {
    let focal = (intrinsics.fx + intrinsics.fy) / 2.0;
    let spacings: Vec<f64> = maps
        .iter()
        .flat_map(|m| {
            m.depth
                .iter()
                .filter(|d| d.is_finite())
                .step_by(7)
                .map(move |&d| d as f64 * m.stride as f64 / focal)
        })
        .collect();
    if spacings.is_empty() {
        return Err(Error::EmptyMesh);
    }

    let uncertainties: Vec<f64> = maps
        .iter()
        .flat_map(|m| {
            m.depth
                .iter()
                .zip(&m.uncertainty)
                .filter(|(d, _)| d.is_finite())
                .step_by(7)
                .map(|(_, &u)| u as f64)
        })
        .collect();
    let typical_uncertainty = median(uncertainties).max(1e-9);

    let cosines: Vec<Vec<f32>> = maps.iter().map(|m| incidence(m, intrinsics)).collect();

    let mut voxel_size = median(spacings) * params.voxel_factor;
    let keys = loop {
        report(
            on_event,
            Stage::Fusion,
            0.1,
            format!("finding the occupied volume at voxel size {voxel_size:.4}"),
        )?;
        let keys = allocate(
            maps,
            cameras,
            intrinsics,
            voxel_size,
            params.truncation,
            &cosines,
        );
        if keys.len() * VOXELS_PER_BLOCK <= params.max_voxels {
            break keys;
        }
        // Too fine to fit in the voxel budget; coarsen and measure again.
        voxel_size *= 2f64.cbrt();
    };

    let mut blocks: Blocks = keys
        .iter()
        .map(|&key| (key, Box::new([Voxel::default(); VOXELS_PER_BLOCK])))
        .collect();
    integrate(
        &mut blocks,
        maps,
        cameras,
        intrinsics,
        voxel_size,
        params.truncation * voxel_size,
        typical_uncertainty,
        &cosines,
        on_event,
    )?;

    report(
        on_event,
        Stage::Fusion,
        0.8,
        format!("extracting the surface from {} blocks", keys.len()),
    )?;
    let mut mesh = surface_nets(&blocks, &keys, voxel_size, params.min_weight);
    report(
        on_event,
        Stage::Fusion,
        0.95,
        format!("{} triangles before pruning", mesh.triangles.len()),
    )?;
    if mesh.triangles.is_empty() {
        return Err(Error::EmptyMesh);
    }
    let before = mesh.triangles.len();
    let largest_component = mesh.keep_large_components(params.min_component_fraction);
    if largest_component < MIN_LARGEST_COMPONENT {
        return Err(Error::FragmentedMesh {
            largest_component,
            required: MIN_LARGEST_COMPONENT,
        });
    }
    mesh.compute_normals();

    Ok(Fused {
        dropped_triangles: before - mesh.triangles.len(),
        mesh,
        voxel_size,
        blocks: keys.len(),
        largest_component,
    })
}

/// The camera looking at each depth map, with its inverse.
fn posed<'a>(
    maps: &'a [DepthMap],
    cameras: &[Option<Pose>],
) -> impl Iterator<Item = (&'a DepthMap, Pose)> {
    let poses: Vec<Pose> = maps
        .iter()
        .map(|m| cameras[m.view.index()].expect("depth maps are of registered views"))
        .collect();
    maps.iter().zip(poses)
}

/// The blocks within the truncation band of any depth sample, sorted.
fn allocate(
    maps: &[DepthMap],
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    voxel_size: f64,
    truncation: f64,
    cosines: &[Vec<f32>],
) -> Vec<[i64; 3]> {
    let block_size = voxel_size * BLOCK as f64;
    let band = truncation * voxel_size;
    let mut keys: HashSet<[i64; 3], Deterministic> = HashSet::default();
    for ((map, pose), cosines) in posed(maps, cameras).zip(cosines) {
        let to_world = pose.inverse();
        let centre = pose.centre();
        for row in 0..map.rows {
            for column in 0..map.columns {
                let Some(d) = map.get(column, row) else {
                    continue;
                };
                let d = d as f64;
                let n = intrinsics.normalize(map.pixel(column, row));
                let point = to_world.to_camera(&Point3::new(n.x() * d, n.y() * d, d));
                let direction = (point - centre).normalize();
                // The band is measured along the surface normal, so along a ray that meets
                // the surface obliquely it reaches further.
                let cosine =
                    (cosines[row * map.columns + column] as f64).max(MIN_ALLOCATION_COSINE);
                let reach = band / cosine;
                let steps = ((2.0 * reach) / (block_size / 2.0)).ceil().max(1.0) as usize;
                for step in 0..=steps {
                    let t = -reach + 2.0 * reach * step as f64 / steps as f64;
                    let q = point + direction * t;
                    keys.insert(q.coords.map(|c| (c / block_size).floor() as i64).into());
                }
            }
        }
    }
    let mut keys: Vec<[i64; 3]> = keys.into_iter().collect();
    keys.sort_unstable();
    keys
}

/// The smallest cosine between a ray and the surface normal that distances are scaled by.
/// Surfaces seen more obliquely than this are treated as if seen at this angle.
const MIN_COSINE: f64 = 0.1;

/// The smallest cosine allocation stretches the band for, which bounds how many blocks a
/// single sample can allocate.
const MIN_ALLOCATION_COSINE: f64 = 0.25;

/// For each depth sample, the cosine between its viewing ray and the surface normal that
/// the neighbouring samples imply. 1 where the neighbours do not give a normal.
fn incidence(map: &DepthMap, intrinsics: &Intrinsics) -> Vec<f32> {
    let point = |column: usize, row: usize| -> Option<Vector3<f64>> {
        let d = map.get(column, row)? as f64;
        let n = intrinsics.normalize(map.pixel(column, row));
        Some(Vector3::new(n.x() * d, n.y() * d, d))
    };
    let mut cosines = vec![1.0f32; map.depth.len()];
    for row in 0..map.rows {
        for column in 0..map.columns {
            let Some(centre) = point(column, row) else {
                continue;
            };
            let tangent =
                |before: Option<Vector3<f64>>, after: Option<Vector3<f64>>| match (before, after) {
                    (Some(a), Some(b)) => Some(b - a),
                    (Some(a), None) => Some(centre - a),
                    (None, Some(b)) => Some(b - centre),
                    (None, None) => None,
                };
            let across = tangent(
                column.checked_sub(1).and_then(|c| point(c, row)),
                (column + 1 < map.columns)
                    .then(|| point(column + 1, row))
                    .flatten(),
            );
            let down = tangent(
                row.checked_sub(1).and_then(|r| point(column, r)),
                (row + 1 < map.rows)
                    .then(|| point(column, row + 1))
                    .flatten(),
            );
            let (Some(across), Some(down)) = (across, down) else {
                continue;
            };
            // Neighbours across a depth edge belong to another surface.
            let limit = 0.2 * centre.z;
            if across.norm() > limit || down.norm() > limit {
                continue;
            }
            let Some(normal) = across.cross(&down).try_normalize(1e-12) else {
                continue;
            };
            let cosine = normal.dot(&centre.normalize()).abs().max(MIN_COSINE);
            cosines[row * map.columns + column] = cosine as f32;
        }
    }
    cosines
}

fn voxel_centre(global: [i64; 3], voxel_size: f64) -> Point3<f64> {
    Point3::new(
        (global[0] as f64 + 0.5) * voxel_size,
        (global[1] as f64 + 0.5) * voxel_size,
        (global[2] as f64 + 0.5) * voxel_size,
    )
}

fn local_index(local: [i64; 3]) -> usize {
    ((local[2] * BLOCK + local[1]) * BLOCK + local[0]) as usize
}

#[allow(clippy::too_many_arguments)]
fn integrate(
    blocks: &mut Blocks,
    maps: &[DepthMap],
    cameras: &[Option<Pose>],
    intrinsics: &Intrinsics,
    voxel_size: f64,
    truncation: f64,
    typical_uncertainty: f64,
    cosines: &[Vec<f32>],
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<(), Error> {
    let views: Vec<((&DepthMap, Pose), &Vec<f32>)> = posed(maps, cameras).zip(cosines).collect();
    let total = blocks.len();
    for (index, (key, block)) in blocks.iter_mut().enumerate() {
        // Every voxel of every block is projected into every view, so this is the longest
        // loop of the stage. Reporting in batches keeps the callback off the hot path.
        if index % 64 == 0 {
            report(
                on_event,
                Stage::Fusion,
                0.2 + 0.6 * index as f32 / total.max(1) as f32,
                format!("merging depth into block {index} of {total}"),
            )?;
        }
        for z in 0..BLOCK {
            for y in 0..BLOCK {
                for x in 0..BLOCK {
                    let global = [key[0] * BLOCK + x, key[1] * BLOCK + y, key[2] * BLOCK + z];
                    let centre = voxel_centre(global, voxel_size);
                    let voxel = &mut block[local_index([x, y, z])];
                    for ((map, pose), cosines) in &views {
                        let c = pose.to_camera(&centre);
                        if c.z <= 0.0 {
                            continue;
                        }
                        let p = intrinsics.denormalize(Norm::new(c.x / c.z, c.y / c.z));
                        let Some(i) = map.index_of(p).filter(|&i| map.depth[i].is_finite()) else {
                            continue;
                        };
                        let (d, uncertainty) = (map.depth[i], map.uncertainty[i]);
                        // Depth difference along the optical axis, turned into distance along
                        // the ray and then along the surface normal. Measured along the ray
                        // alone, a surface seen obliquely gets a band too thin to hold a
                        // zero crossing, and comes out full of holes.
                        let distance = (d as f64 - c.z) * c.coords.norm() / c.z * cosines[i] as f64;
                        if distance < -truncation {
                            continue;
                        }
                        let value = (distance / truncation).min(1.0) as f32;
                        // Less precise depths count for less, so that where precise and
                        // imprecise observations overlap, the precise ones decide.
                        let relative = uncertainty as f64 / typical_uncertainty;
                        let weight = (1.0 / (1.0 + relative * relative)) as f32;
                        voxel.distance = (voxel.distance * voxel.weight + value * weight)
                            / (voxel.weight + weight);
                        voxel.weight += weight;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Offsets of a cell's eight corners, indexed by bits: x = 1, y = 2, z = 4.
const CORNERS: [[i64; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// A crossing between two saturated values is a jump across the band, not a surface.
const MAX_CROSSING_JUMP: f32 = 1.0;

fn surface_nets(blocks: &Blocks, keys: &[[i64; 3]], voxel_size: f64, min_weight: f32) -> Mesh {
    let lookup = |global: [i64; 3]| -> Option<f32> {
        let key = global.map(|g| g.div_euclid(BLOCK));
        let local = global.map(|g| g.rem_euclid(BLOCK));
        let voxel = blocks.get(&key)?[local_index(local)];
        (voxel.weight >= min_weight).then_some(voxel.distance)
    };
    let crosses = |a: f32, b: f32| (a < 0.0) != (b < 0.0) && (a - b).abs() <= MAX_CROSSING_JUMP;

    let mut mesh = Mesh::default();
    let mut vertex_of: HashMap<[i64; 3], u32, Deterministic> = HashMap::default();

    // A vertex in every cell with a crossing edge.
    for key in keys {
        for local in 0..VOXELS_PER_BLOCK as i64 {
            let origin = [
                key[0] * BLOCK + local % BLOCK,
                key[1] * BLOCK + (local / BLOCK) % BLOCK,
                key[2] * BLOCK + local / (BLOCK * BLOCK),
            ];
            let mut values = [0.0f32; 8];
            let mut observed = true;
            for (value, offset) in values.iter_mut().zip(CORNERS) {
                match lookup([
                    origin[0] + offset[0],
                    origin[1] + offset[1],
                    origin[2] + offset[2],
                ]) {
                    Some(v) => *value = v,
                    None => {
                        observed = false;
                        break;
                    }
                }
            }
            if !observed {
                continue;
            }

            let mut sum = Vector3::zeros();
            let mut crossings = 0;
            for a in 0..8 {
                for bit in [1, 2, 4] {
                    let b = a | bit;
                    if b == a || !crosses(values[a], values[b]) {
                        continue;
                    }
                    let t = values[a] / (values[a] - values[b]);
                    let pa = Vector3::from(CORNERS[a].map(|c| c as f64));
                    let pb = Vector3::from(CORNERS[b].map(|c| c as f64));
                    sum += pa + (pb - pa) * t as f64;
                    crossings += 1;
                }
            }
            if crossings == 0 {
                continue;
            }
            let offset = sum / crossings as f64;
            let position = voxel_centre(origin, voxel_size) + offset * voxel_size;
            vertex_of.insert(origin, mesh.positions.len() as u32);
            mesh.positions.push(position);
        }
    }

    // A quad around every crossing voxel edge, joining the four cells that share it.
    for key in keys {
        for local in 0..VOXELS_PER_BLOCK as i64 {
            let v = [
                key[0] * BLOCK + local % BLOCK,
                key[1] * BLOCK + (local / BLOCK) % BLOCK,
                key[2] * BLOCK + local / (BLOCK * BLOCK),
            ];
            let Some(dv) = lookup(v) else { continue };
            for axis in 0..3 {
                let mut w = v;
                w[axis] += 1;
                let Some(dw) = lookup(w) else { continue };
                if !crosses(dv, dw) {
                    continue;
                }
                let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
                let cell = |i: i64, j: i64| {
                    let mut origin = v;
                    origin[b] -= i;
                    origin[c] -= j;
                    vertex_of.get(&origin).copied()
                };
                let (Some(c00), Some(c10), Some(c11), Some(c01)) =
                    (cell(0, 0), cell(1, 0), cell(1, 1), cell(0, 1))
                else {
                    continue;
                };
                // (c00, c10, c11) winds counter-clockwise around +axis. The normal must
                // point from the negative side to the positive one.
                if dv < 0.0 {
                    mesh.triangles.push([c00, c10, c11]);
                    mesh.triangles.push([c00, c11, c01]);
                } else {
                    mesh.triangles.push([c00, c11, c10]);
                    mesh.triangles.push([c00, c01, c11]);
                }
            }
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid filled directly with the truncated distance to the plane `z = 1.03`,
    /// positive towards negative z.
    fn plane_blocks(voxel_size: f64) -> (Blocks, Vec<[i64; 3]>) {
        let mut blocks = Blocks::default();
        let mut keys = Vec::new();
        for bz in -1..3 {
            for by in -1..1 {
                for bx in -1..1 {
                    let key = [bx, by, bz];
                    let mut block = Box::new([Voxel::default(); VOXELS_PER_BLOCK]);
                    for z in 0..BLOCK {
                        for y in 0..BLOCK {
                            for x in 0..BLOCK {
                                let global = [bx * BLOCK + x, by * BLOCK + y, bz * BLOCK + z];
                                let centre = voxel_centre(global, voxel_size);
                                let distance =
                                    ((1.03 - centre.z) / (4.0 * voxel_size)).clamp(-1.0, 1.0);
                                block[local_index([x, y, z])] = Voxel {
                                    distance: distance as f32,
                                    weight: 1.0,
                                };
                            }
                        }
                    }
                    blocks.insert(key, block);
                    keys.push(key);
                }
            }
        }
        keys.sort_unstable();
        (blocks, keys)
    }

    #[test]
    fn extracts_a_plane_facing_the_positive_side() {
        let voxel_size = 0.05;
        let (blocks, keys) = plane_blocks(voxel_size);
        let mut mesh = surface_nets(&blocks, &keys, voxel_size, 1.0);
        assert!(mesh.triangles.len() > 100);
        for p in &mesh.positions {
            assert!(
                (p.z - 1.03).abs() < 1e-6,
                "vertex off the plane at z = {}",
                p.z
            );
        }
        mesh.compute_normals();
        for n in &mesh.normals {
            assert!(
                (n - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-6,
                "normal {n:?}"
            );
        }
        let (_, sizes) = mesh.components();
        assert_eq!(sizes.len(), 1);
    }
}
