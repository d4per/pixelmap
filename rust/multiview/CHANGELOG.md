# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.2 - 2026-11-17

### Changed

- Requires pixelmap 0.3.1. Its solver is faster: each pair is mapped with about 40% fewer
  region comparisons, which on 4128x3096 photos takes `High` from 5.9 s to 4.3 s per pair.
  It also no longer leaves straight unmapped bands, one grid cell wide, through the
  correspondence of some pairs; those bands showed up here as gaps in the depth maps and
  the mesh. For the same photos and seed the reconstruction therefore differs from
  0.1.1's. See pixelmap's CHANGELOG for details.
- The README's example is now a complete program: it decodes the photos, runs the
  reconstruction, and writes the textured mesh as GLB and as a web page.

## 0.1.1 - 2026-10-04

### Added

- `export::write_textured_glb`, which writes the textured mesh as binary glTF 2.0 (GLB):
  one file with the atlas embedded, for viewers and engines that read glTF but not X3D.
  The CLI writes it to `mesh.glb`.
- `export::write_textured_html`, which writes one self-contained HTML page that shows the
  textured mesh in 3D: the GLB is embedded as a `data:` URI and displayed with Google's
  `<model-viewer>`, loaded from a pinned CDN URL. The CLI writes it to `mesh.html`.

### Changed

- The CLI always writes the textured mesh, to the current directory when no `--dump-dir`
  is given; before, it wrote nothing without one. `--dump-dir` still adds the photos,
  sparse model, depth and coverage maps.
- The README and `Options` examples now take the focal length from EXIF and turn on
  `refine_focal`, where they used to hardcode 28 mm, a value a copied example would
  keep for any camera.

## 0.1.0 - 2026-09-26

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
- `DropReason`, why a photo was left out, carried by `Event::ViewDropped`,
  `Error::RegistrationFailed` and `RegistrationWarning::Unregistered`.

### Fixed

- Memory during pairwise correspondence. Each pair's mapping kept alive the two photos
  pixelmap had scaled to its working width, about 30 MB per pair at `Quality::High`.
  With 23 photos (253 pairs) that exhausted memory partway through. Now only the grids
  are kept, about 1 MB per pair, and lookups return the same values bit for bit.
