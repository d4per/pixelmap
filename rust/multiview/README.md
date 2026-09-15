# pixelmap_multiview

3D reconstruction from three or more photos of the same scene.

`pixelmap_model_3d` lifts a single mapping between two photos into a surface. This crate
goes further: it maps every pair of N ≥ 3 photos with [`pixelmap`](../pixelmap), recovers
where each photo was taken from, and fuses all of the views into one mesh in a single
frame.

**Status: under construction.** Implemented so far:

1. **Pairwise correspondence** — pixelmap over every pair, and a check that the
   well-mapped pairs link enough views together.
2. **Two-view geometry** — the relative pose of each pair from an 8-point essential
   matrix in RANSAC, plus triage that flags cameras which only rotated or moved too
   little to measure depth.

Next, in order: tracks with a cycle-consistency filter, incremental registration, bundle
adjustment, dense depth, TSDF fusion and texturing.

`synthetic` renders scenes with known geometry, and answers correspondence queries
exactly, so each stage can be tested against ground truth instead of against the
matcher's limitations.

Like the library it builds on, it does no I/O and has no platform dependencies, so it
builds for `wasm32-unknown-unknown` with `--no-default-features`. Decoding, EXIF and
resizing are the caller's job; the `pixelmap-multiview` binary in
[`../multiview_cli`](../multiview_cli) does them for photos on disc:

    cargo run --release -p pixelmap_multiview_cli -- a.jpg b.jpg c.jpg d.jpg
    cargo run --release -p pixelmap_multiview_cli -- --synthetic sphere --views 4

It is not published to crates.io.

## Cost

Pairwise correspondence dominates: N photos need N(N − 1)/2 pixelmap runs. One run on
a 4:3 photo pair, native build, measured with `pixelmap-multiview --time-pair`:

| Quality | 800 px | 1200 px | 1800 px |
|---------|--------|---------|---------|
| low     | 0.60 s | 0.56 s  | 0.50 s  |
| medium  | 2.25 s | 2.30 s  | 2.38 s  |

The input size barely matters, because pixelmap scales every photo to a fixed working
width (400 px for low, 800 px for medium) before it starts. What the input resolution buys is
the precision of the geometry that follows, not matching time.
