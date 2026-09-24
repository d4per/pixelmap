# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - Unreleased

First release: 3D reconstruction from three or more photos of one scene, on top of
pixelmap 0.3.

### Added

- `run`, which goes from photos to a textured mesh in one call. It works through
  pairwise correspondence, two-view geometry, tracks, registration, bundle adjustment,
  dense depth, fusion and texturing. It returns a `Model` with the mesh, texture,
  intrinsics, camera poses and a report on every pair.
- `reconstruct`, which does the same from correspondences the caller supplies through
  `PairLookup` and `PairGraph`.
- `Options`, which sets the quality, focal length, seed, focal refinement and the
  largest texture size.
- `Event`, typed progress reports that a callback can answer with `Flow::Break` to stop
  the run, and `Status`, which folds them into one snapshot.
- `Job` and `Cancel`, behind the default `threads` feature, which run a reconstruction
  on a worker thread.
- `MIN_PAIR_COVERAGE`, the coverage a pair needs to link its two photos, so a UI can
  judge pairs by the same threshold.
- `export`, with writers for PLY, OBJ with MTL, and X3D.
- `Error`, which names the stage that failed and what to change about the photos.
