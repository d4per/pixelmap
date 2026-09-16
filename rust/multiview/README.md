# pixelmap_multiview

3D reconstruction from three or more photos of the same scene.

`pixelmap_model_3d` lifts a single mapping between two photos into a surface. This crate
goes further: it maps every pair of N ≥ 3 photos with [`pixelmap`](../pixelmap), recovers
where each photo was taken from, and fuses all of the views into one mesh in a single
frame.

The stages, in order:

1. **Pairwise correspondence** — pixelmap over every pair, and a check that the
   well-mapped pairs link enough views together.
2. **Two-view geometry** — the relative pose of each pair from an 8-point essential
   matrix in RANSAC, plus triage that flags cameras which only rotated or moved too
   little to measure depth.
3. **Tracks** — points followed across every view that sees them, kept only where each
   observation agrees with every other view's mapping.
4. **Registration** — the best pair defines the frame, the other views are added by
   PnP, and every track is triangulated over all the views that see it.
5. **Bundle adjustment** — every camera and point refined together by Levenberg–Marquardt
   with a Schur complement and a Huber loss, optionally including the focal length.
6. **Dense depth** — a depth map per view from the full correspondences, solved along each
   pixel's ray and kept only where the other views' depth maps agree. Each depth carries its
   expected error, so a depth only two photos support is judged by what two photos can
   deliver rather than by the standard of three.
7. **Fusion** — the depth maps merged in a sparse truncated signed distance grid, and the
   surface extracted with surface nets into a mesh.
8. **Texturing** — a colour per vertex, and a texture atlas in which each part of the surface
   comes from the photo that sees it best.

`pipeline::run` runs every stage in one call and returns every intermediate result as well
as the textured mesh. Each way a capture can fail, such as a camera that only turned, a
flat scene, or photos with too little in common, comes back as an error that names the
stage and says what to do differently.

## Feedback while it runs

Every stage that takes noticeable time reports through the progress callback, which is a
`FnMut(Event) -> Flow`: pairwise correspondence per solver step, two-view geometry per
pair, registration per view placed, bundle adjustment per round, dense depth per view
solved, cross-checked and cleaned, fusion per batch of volume blocks, and texturing per
batch of vertices. `Event::fraction` weighs the stages against each other and gives one
number for the whole run. Returning `Flow::Break(())` stops the run at the next report and
comes back as `Error::Cancelled`, naming the stage it stopped in.

The event that finishes each pair also carries that pair's `PairMap`: the dense
correspondence itself, as the raw grid, in the coordinates of the photos handed in rather
than the lower resolution the solver works at. It is plain data — no borrows, no generics —
so a frontend can draw the match for every pair as it is found. How to draw it is the
caller's decision; the crate renders nothing.

Each stage can also be called on its own with a callback: `depth::estimate_with_progress`,
`fusion::fuse_with_progress`, `texture::build_with_progress`, `ba::adjust_with_progress`
and `sfm::reconstruct_with_progress` sit beside the plain forms, which stay unchanged.

`synthetic` renders scenes with known geometry, and answers correspondence queries
exactly, so each stage can be tested against ground truth instead of against the
matcher's limitations.

Like the library it builds on, it does no I/O and has no platform dependencies, so it
builds for `wasm32-unknown-unknown` with `--no-default-features`. Decoding, EXIF and
resizing are the caller's job; the `pixelmap-multiview` binary in
[`../multiview_cli`](../multiview_cli) does them for photos on disc:

    cargo run --release -p pixelmap_multiview_cli -- a.jpg b.jpg c.jpg d.jpg --dump-dir out
    cargo run --release -p pixelmap_multiview_cli -- --synthetic sphere --views 4

With `--dump-dir`, the sparse model is written to `out/sparse.ply`, with coloured points and a
red pyramid per camera; the textured surface to `out/mesh.obj`, with `mesh.mtl` and
`mesh_texture.png`, and as X3D to `out/mesh.x3d`; the surface with a colour per vertex to `out/mesh_colours.ply`; and each
depth map to `out/depth_N.png`. They open in MeshLab, Blender or CloudCompare.

The CLI also prints, per photo, how much of it was matched to no other photo and where the
remaining depth samples were dropped.

Known limitation: pixelmap matches little within about 25–50 px (at 1200 px) of a photo's
edge, so a strip along each edge that only two photos share usually stays empty. Photos that
overlap generously avoid it; so does anything seen by a third photo away from its edge.

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
