//! Stage 3: tracks, each one scene point followed across every view that sees it.
//!
//! Anchors are laid on a grid in one view and pushed through the pair mappings into
//! every other view. The next view then anchors only the parts of itself no earlier track
//! reached, and so on.
//!
//! A track is kept only if its observations agree with each other. Every observation is
//! mapped directly into every other observing view and must land on that view's
//! observation. This covers every edge of every triangle of views, plus each pair's own
//! round trip. It is a far stronger outlier test than anything a single pair can apply,
//! and it is the reason every pair is mapped rather than only the pairs touching one
//! view. It costs nothing but lookups.

use crate::error::Error;
use crate::lookup::PairLookup;
use crate::pairs::PairGraph;
use crate::types::{PhotoPx, ViewId};

/// The fewest tracks seen in three or more views that registration will work from.
pub const MIN_TRACKS: usize = 500;

/// Tuning for [`build`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Anchor spacing, in photo pixels. Also the size of the cells used to avoid tracking
    /// the same point twice.
    pub stride: f32,
    /// The largest disagreement allowed between an observation and where another view's
    /// mapping puts it, as a multiple of the mappings' precision.
    pub max_residual: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            stride: 8.0,
            max_residual: 4.0,
        }
    }
}

/// One scene point, observed in two or more views.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Track {
    /// The view the track was started from. Its observation is a grid point rather than
    /// a measurement.
    pub anchor: ViewId,
    /// Where each observing view sees the point, in ascending view order.
    pub observations: Vec<(ViewId, PhotoPx)>,
    /// The largest disagreement between the observations, in photo pixels.
    pub residual: f32,
}

impl Track {
    /// How many views observe the point.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether no view observes the point. Never true for a track [`build`] returns.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Where `view` sees the point, if it does.
    pub fn observation(&self, view: ViewId) -> Option<PhotoPx> {
        self.observations
            .binary_search_by_key(&view, |&(v, _)| v)
            .ok()
            .map(|i| self.observations[i].1)
    }
}

/// Builds every track `graph` supports over photos of `size`.
pub fn build<L: PairLookup>(
    graph: &PairGraph<L>,
    (width, height): (usize, usize),
    params: &Params,
) -> Vec<Track> {
    let views = graph.views();
    let max_residual = params.max_residual * graph.precision_px();
    let stride = params.stride.max(1.0);
    let columns = (width as f32 / stride).ceil() as usize;
    let rows = (height as f32 / stride).ceil() as usize;
    let (max_x, max_y) = (width as f32 - 1.0, height as f32 - 1.0);
    let inside = |p: PhotoPx| (0.0..=max_x).contains(&p.x()) && (0.0..=max_y).contains(&p.y());
    let cell = |p: PhotoPx| {
        let cx = ((p.x() / stride) as usize).min(columns - 1);
        let cy = ((p.y() / stride) as usize).min(rows - 1);
        cy * columns + cx
    };

    let mut claimed = vec![vec![false; columns * rows]; views];
    let mut tracks = Vec::new();
    let mut observations = Vec::with_capacity(views);

    for anchor in (0..views as u32).map(ViewId) {
        for index in 0..columns * rows {
            if claimed[anchor.index()][index] {
                continue;
            }
            let p = PhotoPx::new(
                ((index % columns) as f32 + 0.5) * stride,
                ((index / columns) as f32 + 0.5) * stride,
            );
            if !inside(p) {
                continue;
            }

            observations.clear();
            let mut duplicate = false;
            for view in (0..views as u32).map(ViewId) {
                if view == anchor {
                    observations.push((view, p));
                    continue;
                }
                let Some(q) = graph
                    .directed(anchor, view)
                    .and_then(|mapping| mapping.map(p))
                    .filter(|&q| inside(q))
                else {
                    continue;
                };
                // Landing where an earlier track already is means this is that point
                // again, seen from another anchor.
                duplicate |= claimed[view.index()][cell(q)];
                observations.push((view, q));
            }
            if duplicate || observations.len() < 2 {
                continue;
            }

            let Some(residual) = consistency(graph, anchor, &observations) else {
                continue;
            };
            if residual > max_residual {
                continue;
            }
            for &(view, q) in &observations {
                claimed[view.index()][cell(q)] = true;
            }
            tracks.push(Track {
                anchor,
                observations: observations.clone(),
                residual,
            });
        }
    }
    tracks
}

/// The largest distance between an observation and where another observing view's
/// mapping puts it. `None` if some observation could not be checked at all.
///
/// Mappings out of the anchor are skipped: they are where the other observations came
/// from, so they would agree by construction.
fn consistency<L: PairLookup>(
    graph: &PairGraph<L>,
    anchor: ViewId,
    observations: &[(ViewId, PhotoPx)],
) -> Option<f32> {
    let mut worst = 0.0f32;
    let mut verified = vec![false; observations.len()];
    for (i, &(from, p)) in observations.iter().enumerate() {
        if from == anchor {
            continue;
        }
        for (j, &(to, q)) in observations.iter().enumerate() {
            if i == j {
                continue;
            }
            let Some(mapped) = graph.directed(from, to).and_then(|m| m.map(p)) else {
                continue;
            };
            worst = worst.max((mapped.0 - q.0).norm());
            verified[i] = true;
            verified[j] = true;
        }
    }
    verified.iter().all(|&v| v).then_some(worst)
}

/// Checks that enough tracks reach three or more views to build a rigid reconstruction
/// on. Returns how many do.
///
/// # Errors
/// [`Error::TooFewTracks`].
pub fn require_enough(tracks: &[Track]) -> Result<usize, Error> {
    let found = tracks.iter().filter(|t| t.len() >= 3).count();
    if found < MIN_TRACKS {
        return Err(Error::TooFewTracks {
            found,
            required: MIN_TRACKS,
        });
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every view is the same picture shifted `offset × view` pixels to the right, except
    /// that the mapping from view 0 into view 2 is `lie` pixels off.
    struct Shift {
        offset: f32,
        lie: f32,
    }

    impl PairLookup for Shift {
        fn a_to_b(&self, p: PhotoPx) -> Option<PhotoPx> {
            Some(PhotoPx::new(p.x() + self.offset + self.lie, p.y()))
        }
        fn b_to_a(&self, p: PhotoPx) -> Option<PhotoPx> {
            Some(PhotoPx::new(p.x() - self.offset, p.y()))
        }
        fn coverage(&self) -> f32 {
            1.0
        }
        fn native_stride(&self) -> f32 {
            8.0
        }
        fn precision_px(&self) -> f32 {
            1.0
        }
    }

    fn graph(lie: f32) -> PairGraph<Shift> {
        PairGraph::from_fn(3, |pair| Shift {
            offset: 2.0 * (pair.b().0 - pair.a().0) as f32,
            lie: if (pair.a().0, pair.b().0) == (0, 2) {
                lie
            } else {
                0.0
            },
        })
    }

    #[test]
    fn follows_points_through_every_view_once() {
        let tracks = build(&graph(0.0), (80, 40), &Params::default());
        assert!(!tracks.is_empty());
        assert!(tracks.iter().all(|t| t.residual < 1e-4));

        // Anchored in view 0 wherever all three views see the point, and not again from
        // views 1 and 2 at the same place.
        let full: Vec<_> = tracks.iter().filter(|t| t.len() == 3).collect();
        assert!(!full.is_empty());
        for track in &full {
            let p0 = track.observation(ViewId(0)).unwrap();
            let p2 = track.observation(ViewId(2)).unwrap();
            assert_eq!(p2.x() - p0.x(), 4.0);
        }
        let mut cells: Vec<_> = tracks
            .iter()
            .filter_map(|t| t.observation(ViewId(0)))
            .map(|p| ((p.x() / 8.0) as i32, (p.y() / 8.0) as i32))
            .collect();
        let before = cells.len();
        cells.sort_unstable();
        cells.dedup();
        assert_eq!(cells.len(), before, "a point was tracked twice");
    }

    #[test]
    fn rejects_tracks_whose_views_disagree() {
        // The lie only shows up through the third view: 0→1→2 and 0→2 disagree.
        let honest = build(&graph(0.0), (80, 40), &Params::default());
        let lied = build(&graph(10.0), (80, 40), &Params::default());
        assert!(lied.iter().all(|t| t.len() < 3 || t.anchor != ViewId(0)));
        assert!(honest.iter().filter(|t| t.len() == 3).count() > 0);
        assert!(require_enough(&lied).is_err());
    }
}
