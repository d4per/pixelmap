# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- **The mapping for a given seed is no longer the one 0.3.0 produced.** It is still fully
  determined by the photos, the schedule and the seed; it is a different, equally good
  mapping, reached with about 40% fewer region comparisons. On a 4128x3096 pair `High`
  went from 5.9 s to 4.3 s and `Medium` from 1.7 s to 1.3 s.

  The solver used to let a grid cell hand its transform to its four neighbours every time
  the cell improved. Measured on real photos, about half of all comparisons were spent on
  the offspring of transforms that left no trace in the result: a quarter of all accepted
  transforms were replaced within the very pass that accepted them, after they had already
  reproduced. Three things changed:

  - A cell now reproduces once per pass, after the pass, from the transform it ends up
    holding.
  - A cell that already held a transform reproduces only if its score dropped by more than
    5% over the pass. Smaller gains are still kept by the cell itself.
  - A transform a cell has already evaluated is not scored a second time. This one changes
    the work and nothing else: such a transform could only have been rejected.

  The result was checked against an exact ground truth on rendered scenes, where the median
  error is unchanged or slightly lower, and on two sets of three real photos by how well
  the three pairwise mappings close the loop from the first photo to the second, the third
  and back. On both it stays inside the spread the solver shows from one seed to the next.
  On the hardest pairs at `Low` it sits at the weak end of that spread.
- `Correspondence::comparisons()` reports correspondingly fewer comparisons.

### Fixed

- Straight unmapped bands, one grid cell wide, could run down a column or along a row of
  the mapping, usually near where it leaves the edge of the second photo. A cell's score
  is the variance of the colour differences over the part of its disc that lands inside
  the second photo, and nothing required that part to be large. A transform aimed at the
  very edge kept a single sample, whose variance is exactly zero, the best possible
  score. It then won every cell it was offered along that column or row, and the
  round-trip check deleted them all afterwards. A transform is now scored only if at
  least half of the disc pixels the first photo offers land inside the second photo.
  Cells on the border of the first photo, whose discs are smaller to begin with, are
  unaffected. On the pair that showed it, the bands appeared with 2 of 6 seeds at
  `Medium` and `High` and are now gone at every quality and seed tried; cycle
  consistency on two sets of three photos stays within the seed-to-seed spread.
- `Medium` and `High` panicked in any build with debug assertions on photos 800 px wide or
  more. Their first steps work at a lower width than the map they start from, which can
  place a candidate one cell past the grid; the solver has always skipped those, but also
  carried a `debug_assert!` claiming it could not happen. Release builds were unaffected.

## 0.3.0 - 2026-09-24

### Added

- `Builder::run_with_control`, which runs the pipeline with a progress callback that can
  stop it. Returning `ControlFlow::Break(())` abandons the run and yields `Ok(None)`.
  Cancellation is deliberately not an `Error` variant: every one of those describes a
  problem with the input, and a caller asking a run to stop is not one. A run is abandoned
  only between schedule steps, so a `Break` waits at most one step.
  `run_with_progress` now delegates to it with a callback that never breaks.

### Changed

- Faster and leaner, with output unchanged: for the same input and seed the dense map is
  bit-identical to 0.2.0 at every `Quality`. The forward and backward solvers now run
  concurrently, descriptor rows are streamed into the matcher instead of being held all at
  once, the packed photo words are shared between the passes that read them, `prepare()`
  drops the original photos once they are no longer needed, and `scaled_to_width` runs in
  parallel. The kd-tree keeps only one point per distinct descriptor key: flat image
  regions produce the same key hundreds of thousands of times, and every query landing on
  it had to visit all of them, which made matching quadratic. On one pair of test photos
  `High` went from 9.2 s to 4.8 s and from 318 MB to 215 MB peak memory; on a pair with
  large flat regions it went from 93 s to 2.8 s.

## 0.2.0 - 2026-09-07

Fixes the unmapped bands that could open up in the correspondence grid, and the dead
column and row along its right and bottom edges.

### Fixed

- The grid was laid out one cell wider and taller than the image it describes: the cell
  count came from `width / cell_size + 1`, which puts the last cell's origin at `width`,
  one column past the last pixel. That column and row could never be filled, took their
  neighbours with them through `remove_outliers`, and counted against `coverage()`.
- `interpolated_point` required all four surrounding corners to be set even when only one
  or two of them carried a non-zero bilinear weight. Run through `remove_outliers`, which
  queries exact grid nodes, that discarded a good cell whenever its right or lower
  neighbour was missing — so each pass ate one more column, and the last column, having no
  right-hand neighbour at all, went every time. These were the dark bands.
- `average_grid_points` computed one centre averaged over all four neighbours and used it
  for both the horizontal and the vertical test, so a single `NaN` neighbour poisoned that
  centre and failed both. The documented "average only the valid ones" could never
  happen and a gap wider than one cell never closed. Each axis is now judged on its own
  pair of neighbours.
- An unmapped sample reaching the write in `interpolate_photo` landed on the top left
  pixel, because blending with `NaN` gives `NaN` and `f32::round(NaN) as usize` is 0.
- `interpolated_point` read a negative or `NaN` coordinate as cell zero with a nonsensical
  fraction, since `as usize` saturates. It now reports no mapping.
- `get_photo_mapping` let `DensePhotoMap::new` divide the cell size back out of the photo
  width, which only recovers the original spacing when the width is an exact multiple of
  it.
- `remove_outliers` compared a squared distance against an unsquared threshold.

### Added

- `DensePhotoMap::with_cell_size`, for a producer that laid the grid out itself and knows
  the spacing exactly, rather than having `DensePhotoMap::new` infer it from the width.
- An integration test suite covering the public API.

### Changed

- **`DensePhotoMap::remove_outliers` reinterprets `max_dist`.** It is now squared
  internally, so the parameter means a tolerance in whole grid cells; previously it was
  compared against the squared distance directly, so it meant `sqrt(max_dist)` cells. A
  caller that passed `2.0` and wanted the old behaviour must now pass `1.414`.
- **`DensePhotoMap::interpolate_photo` no longer paints unmapped pixels red.** A source
  pixel with no mapping contributes nothing, and an output pixel nothing lands on is left
  opaque black. Use `Correspondence::lookup` to distinguish a region with no
  correspondence from one the warp stretched past `detail_level`.
- `DensePhotoMap::new` now panics when the cell size it infers comes out as zero — a grid
  finer than the photo it describes, which every later lookup would have divided by zero.
- Corrected the README's Performance section and the `.cargo/config.toml` comment. The
  scoring inner loop walks `photo2` in 16.16 fixed point and vectorizes to a 32-bit
  integer multiply, which SSE2 lacks (`pmulld`); it does not use `f32::round`.

## 0.1.0 - 2026-09-05

Initial release: the reference implementation of the PIXELMAP framework for dense image
correspondence.

- `Correspondence` and its `Builder` as the high-level entry point, with `correspond` as
  the one-call shorthand.
- `Quality` presets (`Low`, `Medium`, `High`) over an explicit `IterationParams` schedule,
  and `PixelMapProcessor` for callers driving the solver step by step.
- `DensePhotoMap` for the raw forward and backward correspondence grids, including a
  versioned serialization format that validates on decode.
- `Photo` as a plain RGBA buffer, with `From` conversions for the `image` crate's types
  behind the `image` feature.
- Deterministic results from a seed, identical with and without the `parallel` feature.
