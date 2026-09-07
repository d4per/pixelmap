# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
