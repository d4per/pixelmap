//! Stage 1: dense correspondence between every pair of photos.
//!
//! This is the only module that sees pixelmap's working resolution. Everything downstream
//! gets [`PhotoPx`] through the [`PairLookup`] implementation below.

use std::sync::Arc;

use pixelmap::{Correspondence, DensePhotoMap, Photo, Quality};

use crate::error::Error;
use crate::event::{emit, Event, Flow, PairMap, Stage};
use crate::input::MIN_VIEWS;
use crate::lookup::{Directed, PairLookup};
use crate::types::{PairId, PhotoPx, ViewId};

/// The coverage below which a pair is not trusted to connect its two views.
///
/// A pair whose [`PairMap::coverage`] falls below this does not count as a link between
/// its photos, and a photo with no such link is left out. It is public so that a caller
/// showing pairs can judge them by the same number the reconstruction does, rather than
/// keeping a copy that drifts. The name is stable; the value may be retuned in any release.
pub const MIN_PAIR_COVERAGE: f32 = 0.25;

/// One grid of a pixelmap mapping, without the photos it was computed from.
#[derive(Clone, Debug)]
struct Grid {
    columns: usize,
    rows: usize,
    /// `(x, y)` per cell, row by row, in working-resolution pixels. NaN where unmapped.
    points: Vec<f32>,
}

impl Grid {
    fn copy(map: &DensePhotoMap) -> Grid {
        let (columns, rows) = map.grid_dimensions();
        let mut points = Vec::with_capacity(columns * rows * 2);
        for row in 0..rows {
            for column in 0..columns {
                let (x, y) = map.grid_coordinates(column, row);
                points.push(x);
                points.push(y);
            }
        }
        Grid {
            columns,
            rows,
            points,
        }
    }

    fn at(&self, column: usize, row: usize) -> (f32, f32) {
        if column >= self.columns || row >= self.rows {
            return (f32::NAN, f32::NAN);
        }
        let i = (row * self.columns + column) * 2;
        (self.points[i], self.points[i + 1])
    }

    /// The mapping at fractional grid coordinates.
    ///
    /// Mirrors `DensePhotoMap::interpolated_point` in pixelmap exactly, so that lookups
    /// through a [`Mapping`] match `Correspondence::lookup` bit for bit: only corners with
    /// a non-zero bilinear weight must be mapped, and a quad whose contributing corners
    /// spread more than three cells from their centroid is too distorted to interpolate.
    fn interpolate(&self, gx: f32, gy: f32, cell_size: usize) -> Option<(f32, f32)> {
        if gx.is_nan() || gy.is_nan() || gx < 0.0 || gy < 0.0 {
            return None;
        }
        let (column, row) = (gx as usize, gy as usize);
        let (xr, yr) = (gx - column as f32, gy - row as f32);
        let corners = [
            (self.at(column, row), (1.0 - xr) * (1.0 - yr)),
            (self.at(column + 1, row), xr * (1.0 - yr)),
            (self.at(column + 1, row + 1), xr * yr),
            (self.at(column, row + 1), (1.0 - xr) * yr),
        ];

        let (mut xt, mut yt, mut sum_x, mut sum_y, mut contributing) = (0.0, 0.0, 0.0, 0.0, 0.0f32);
        for ((cx, cy), weight) in corners {
            if weight == 0.0 {
                continue;
            }
            if cx.is_nan() || cy.is_nan() {
                return None;
            }
            xt += cx * weight;
            yt += cy * weight;
            sum_x += cx;
            sum_y += cy;
            contributing += 1.0;
        }

        let max_dist_sq = (cell_size as f32 * 3.0).powi(2);
        let (centre_x, centre_y) = (sum_x / contributing, sum_y / contributing);
        for ((cx, cy), weight) in corners {
            if weight != 0.0 && (centre_x - cx).powi(2) + (centre_y - cy).powi(2) > max_dist_sq {
                return None;
            }
        }
        // `interpolated_point` returns NaN for a missing mapping and `lookup` turns any
        // NaN component into `None`; do the same.
        (!xt.is_nan() && !yt.is_nan()).then_some((xt, yt))
    }
}

/// A finished pixelmap mapping between two photos, reduced to what reconstruction reads.
///
/// A [`Correspondence`] keeps the two photos it was computed from alive, scaled to the
/// solver's final working width: at [`Quality::High`] that is two 1600 px RGBA images,
/// some 30 MB a pair. A run keeps every pair until the end, so with many photos those
/// copies, not the mappings, filled memory. This holds only the grids, about 1 MB a pair.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Mapping {
    forward: Grid,
    backward: Grid,
    cell_size: usize,
    /// Working pixels per photo pixel.
    working_scale: f32,
    coverage: f32,
}

impl From<Correspondence> for Mapping {
    fn from(mapping: Correspondence) -> Mapping {
        Mapping {
            forward: Grid::copy(mapping.forward()),
            backward: Grid::copy(mapping.backward()),
            cell_size: mapping.forward().grid_cell_size(),
            working_scale: mapping.working_scale(),
            coverage: mapping.coverage(),
        }
    }
}

impl Mapping {
    /// The forward grid's columns and rows.
    pub fn grid_dimensions(&self) -> (usize, usize) {
        (self.forward.columns, self.forward.rows)
    }

    /// `p` through `grid`, converted into and out of the working resolution as
    /// `Correspondence::lookup` does.
    fn look(&self, grid: &Grid, p: PhotoPx) -> Option<PhotoPx> {
        let scale = self.working_scale;
        let cell = self.cell_size as f32;
        let (x, y) =
            grid.interpolate(p.x() * scale / cell, p.y() * scale / cell, self.cell_size)?;
        Some(PhotoPx::new(x / scale, y / scale))
    }
}

impl PairLookup for Mapping {
    fn a_to_b(&self, p: PhotoPx) -> Option<PhotoPx> {
        self.look(&self.forward, p)
    }

    fn b_to_a(&self, p: PhotoPx) -> Option<PhotoPx> {
        self.look(&self.backward, p)
    }

    fn coverage(&self) -> f32 {
        self.coverage
    }

    fn native_stride(&self) -> f32 {
        self.cell_size as f32 / self.working_scale
    }

    fn precision_px(&self) -> f32 {
        // The solver's mapping is good to about a pixel at the resolution it works at.
        1.0 / self.working_scale
    }
}

/// One mapping per pair of views.
///
/// [`run`](crate::run) builds one of these from pixelmap's correspondences. To reconstruct
/// from correspondences of your own, build one with [`Self::from_fn`] and hand it to
/// [`reconstruct`](crate::reconstruct).
#[derive(Clone, Debug)]
pub struct PairGraph<L> {
    views: usize,
    maps: Vec<L>,
}

impl<L: PairLookup> PairGraph<L> {
    /// A graph over `views` views, with `make` called once per pair in
    /// [`PairId::all`] order.
    pub fn from_fn(views: usize, make: impl FnMut(PairId) -> L) -> Self {
        PairGraph {
            views,
            maps: PairId::all(views as u32).map(make).collect(),
        }
    }

    /// How many views the graph spans.
    pub fn views(&self) -> usize {
        self.views
    }

    /// The mapping of `pair`, from its first view to its second. `None` if either view is
    /// outside the graph.
    pub fn get(&self, pair: PairId) -> Option<&L> {
        (pair.b().index() < self.views).then(|| &self.maps[index(self.views, pair)])
    }

    /// The mapping from `from` to `to`, whichever order it is stored in. `None` if the
    /// views are equal or outside the graph.
    // Hidden: `Directed` belongs to the stages, which are hidden for the reasons given in
    // lib.rs.
    #[doc(hidden)]
    pub fn directed(&self, from: ViewId, to: ViewId) -> Option<Directed<'_, L>> {
        let pair = PairId::new(from, to)?;
        Some(Directed::new(self.get(pair)?, from > to))
    }

    /// The largest typical localization error among the mappings, in photo pixels.
    pub fn precision_px(&self) -> f32 {
        self.maps
            .iter()
            .map(|m| m.precision_px())
            .fold(0.0, f32::max)
    }

    /// Every pair with its mapping.
    pub fn pairs(&self) -> impl Iterator<Item = (PairId, &L)> {
        PairId::all(self.views as u32).zip(&self.maps)
    }
}

/// The position of `pair` in [`PairId::all`] order.
fn index(views: usize, pair: PairId) -> usize {
    let (a, b) = (pair.a().index(), pair.b().index());
    a * (2 * views - a - 1) / 2 + (b - a - 1)
}

/// The finished mapping of `pair` as a [`PairMap`], converted out of the solver's working
/// resolution into the coordinates of the photos the caller handed in.
fn pair_map(pair: PairId, mapping: &Mapping) -> PairMap {
    let forward = &mapping.forward;
    // `working_scale` is working pixels per source pixel, so dividing takes a grid entry
    // back to the photo the caller passed in. Unmapped cells hold NaN and stay NaN.
    let scale = mapping.working_scale;
    PairMap {
        pair,
        columns: forward.columns,
        rows: forward.rows,
        cell_size: mapping.native_stride(),
        points: forward.points.iter().map(|v| v / scale).collect(),
        coverage: mapping.coverage,
    }
}

/// Runs pixelmap over every pair of `photos`, one pair at a time.
///
/// Reports progress through `on_event`. A `Break` stops the run at the next pixelmap
/// schedule step, so the longest it waits is one step of the pair being mapped — the last
/// steps of [`Quality::High`] work at 1600 px and are the slowest.
///
/// The event that finishes each pair carries that pair's [`PairMap`], so a caller can show
/// the correspondence as it is found.
///
/// # Errors
/// [`Error::Correspondence`] if pixelmap rejects a pair; [`Error::Cancelled`].
pub fn compute(
    photos: &[Arc<Photo>],
    quality: Quality,
    seed: u64,
    on_event: &mut dyn FnMut(Event) -> Flow,
) -> Result<PairGraph<Mapping>, Error> {
    let views = photos.len();
    let total = PairId::count(views);
    let mut maps = Vec::with_capacity(total);

    for (index, pair) in PairId::all(views as u32).enumerate() {
        let mapping = Correspondence::builder()
            .quality(quality)
            .seed(seed)
            .run_with_control(
                photos[pair.a().index()].clone(),
                photos[pair.b().index()].clone(),
                |p| {
                    on_event(Event::PairProgress {
                        pair,
                        index,
                        of: total,
                        step: p.step as u32,
                        steps: p.total as u32,
                    })
                },
            )
            .map_err(|source| Error::Correspondence { pair, source })?;

        // pixelmap gives nothing back when the callback asked it to stop.
        let Some(mapping) = mapping else {
            return Err(Error::Cancelled {
                stage: Stage::Pairs,
            });
        };

        // Reduced at once, so the working photos the solver scaled are freed with it.
        let mapping = Mapping::from(mapping);
        let map = pair_map(pair, &mapping);
        maps.push(mapping);
        emit(
            on_event,
            Event::PairMapped {
                pair,
                index,
                of: total,
                map: Box::new(map),
            },
        )?;
    }

    Ok(PairGraph { views, maps })
}

/// The largest set of views linked to each other through pairs with at least
/// `min_coverage` coverage, in ascending order. When two sets tie, the one containing
/// the lowest view id wins.
pub fn connected_views<L: PairLookup>(graph: &PairGraph<L>, min_coverage: f32) -> Vec<ViewId> {
    let n = graph.views();
    let mut seen = vec![false; n];
    let mut best: Vec<usize> = Vec::new();

    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut members = vec![start];
        let mut next = 0;
        while next < members.len() {
            let v = members[next];
            next += 1;
            for (w, seen_w) in seen.iter_mut().enumerate() {
                if *seen_w {
                    continue;
                }
                let pair = PairId::new(ViewId(v as u32), ViewId(w as u32))
                    .expect("v is seen and w is not, so they differ");
                if graph
                    .get(pair)
                    .is_some_and(|m| m.coverage() >= min_coverage)
                {
                    *seen_w = true;
                    members.push(w);
                }
            }
        }
        if members.len() > best.len() {
            best = members;
        }
    }

    best.sort_unstable();
    best.into_iter().map(|v| ViewId(v as u32)).collect()
}

/// [`connected_views`], failing when fewer than [`MIN_VIEWS`] remain.
///
/// # Errors
/// [`Error::DisconnectedViews`].
pub fn require_connected<L: PairLookup>(
    graph: &PairGraph<L>,
    min_coverage: f32,
) -> Result<Vec<ViewId>, Error> {
    let connected = connected_views(graph, min_coverage);
    if connected.len() < MIN_VIEWS {
        return Err(Error::DisconnectedViews {
            connected,
            minimum: MIN_VIEWS,
            min_coverage,
        });
    }
    Ok(connected)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(f32);

    impl PairLookup for Fixed {
        fn a_to_b(&self, p: PhotoPx) -> Option<PhotoPx> {
            Some(PhotoPx::new(p.x() + self.0, p.y()))
        }
        fn b_to_a(&self, p: PhotoPx) -> Option<PhotoPx> {
            Some(PhotoPx::new(p.x() - self.0, p.y()))
        }
        fn coverage(&self) -> f32 {
            self.0
        }
        fn native_stride(&self) -> f32 {
            1.0
        }
        fn precision_px(&self) -> f32 {
            1.0
        }
    }

    fn graph(views: usize, coverage: impl Fn(PairId) -> f32) -> PairGraph<Fixed> {
        PairGraph::from_fn(views, |pair| Fixed(coverage(pair)))
    }

    #[test]
    fn indexes_pairs_in_enumeration_order() {
        for views in 2..8 {
            for (expected, pair) in PairId::all(views as u32).enumerate() {
                assert_eq!(index(views, pair), expected);
            }
        }
    }

    #[test]
    fn reads_either_direction() {
        let g = graph(3, |pair| pair.b().0 as f32);
        let forward = g.directed(ViewId(0), ViewId(2)).unwrap();
        let reverse = g.directed(ViewId(2), ViewId(0)).unwrap();
        let p = PhotoPx::new(10.0, 5.0);
        assert_eq!(forward.map(p), Some(PhotoPx::new(12.0, 5.0)));
        assert_eq!(reverse.map(p), Some(PhotoPx::new(8.0, 5.0)));
        assert_eq!(reverse.map_back(p), forward.map(p));
        assert!(g.directed(ViewId(1), ViewId(1)).is_none());
        assert!(g.directed(ViewId(1), ViewId(3)).is_none());
    }

    #[test]
    fn keeps_the_largest_connected_set() {
        // Views 0–1–2 linked, 3 and 4 linked only to each other.
        let links = |pair: PairId| match (pair.a().0, pair.b().0) {
            (0, 1) | (1, 2) | (3, 4) => 0.6,
            _ => 0.1,
        };
        let g = graph(5, links);
        assert_eq!(
            connected_views(&g, MIN_PAIR_COVERAGE),
            [ViewId(0), ViewId(1), ViewId(2)]
        );
        assert!(require_connected(&g, MIN_PAIR_COVERAGE).is_ok());
        assert!(matches!(
            require_connected(&g, 0.7),
            Err(Error::DisconnectedViews { .. })
        ));
    }
}
