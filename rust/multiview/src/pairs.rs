//! Stage 1: dense correspondence between every pair of photos.
//!
//! This is the only module that sees pixelmap's working resolution. Everything downstream
//! gets [`PhotoPx`] through the [`PairLookup`] implementation below.

use std::sync::Arc;

use pixelmap::{Correspondence, Photo, Quality};

use crate::error::Error;
use crate::event::{emit, Event, Flow, PairMap, Stage};
use crate::input::MIN_VIEWS;
use crate::lookup::{Directed, PairLookup};
use crate::types::{PairId, PhotoPx, ViewId};

/// The coverage below which a pair is not trusted to connect its two views.
pub const MIN_PAIR_COVERAGE: f32 = 0.25;

impl PairLookup for Correspondence {
    fn a_to_b(&self, p: PhotoPx) -> Option<PhotoPx> {
        self.lookup(p.x(), p.y()).map(|(x, y)| PhotoPx::new(x, y))
    }

    fn b_to_a(&self, p: PhotoPx) -> Option<PhotoPx> {
        self.lookup_back(p.x(), p.y())
            .map(|(x, y)| PhotoPx::new(x, y))
    }

    fn coverage(&self) -> f32 {
        Correspondence::coverage(self)
    }

    fn native_stride(&self) -> f32 {
        Correspondence::forward(self).grid_cell_size() as f32 / self.working_scale()
    }

    fn precision_px(&self) -> f32 {
        // The solver's mapping is good to about a pixel at the resolution it works at.
        1.0 / self.working_scale()
    }
}

/// One mapping per pair of views.
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

    /// The mapping of `pair`, from its first view to its second.
    ///
    /// # Panics
    /// If either view is outside the graph.
    pub fn get(&self, pair: PairId) -> &L {
        assert!(
            pair.b().index() < self.views,
            "{pair} is outside a graph of {} views",
            self.views
        );
        &self.maps[index(self.views, pair)]
    }

    /// The mapping from `from` to `to`, whichever order it is stored in. `None` if the
    /// views are equal or outside the graph.
    pub fn directed(&self, from: ViewId, to: ViewId) -> Option<Directed<'_, L>> {
        let pair = PairId::new(from, to)?;
        (pair.b().index() < self.views).then(|| Directed::new(self.get(pair), from > to))
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
fn pair_map(pair: PairId, mapping: &Correspondence) -> PairMap {
    let forward = mapping.forward();
    let (columns, rows) = forward.grid_dimensions();
    // `working_scale` is working pixels per source pixel, so dividing takes a grid entry
    // back to the photo the caller passed in. Unmapped cells hold NaN and stay NaN.
    let scale = mapping.working_scale();
    let mut points = Vec::with_capacity(columns * rows * 2);
    for row in 0..rows {
        for column in 0..columns {
            let (x, y) = forward.grid_coordinates(column, row);
            points.push(x / scale);
            points.push(y / scale);
        }
    }
    PairMap {
        pair,
        columns,
        rows,
        cell_size: mapping.native_stride(),
        points,
        coverage: mapping.coverage(),
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
) -> Result<PairGraph<Correspondence>, Error> {
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
                    on_event(Event::PairStarted {
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
                if graph.get(pair).coverage() >= min_coverage {
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
