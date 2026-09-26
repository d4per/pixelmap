# pixelmap_multiview

Turn a set of ordinary photos of a scene into a 3D model.

[`pixelmap_model_3d`](https://github.com/d4per/pixelmap/tree/main/rust/model_3d) lifts a single mapping between two photos into a surface. This crate
goes further: it maps every pair of N ≥ 3 photos with [`pixelmap`](https://crates.io/crates/pixelmap), recovers
where each photo was taken from, and fuses all of the views into one mesh in a single
frame.

## From photos to a 3D model

These 23 photos were taken with a phone while walking once around a statue. Nothing else
went in: no calibration, no markers, no measurements.

![Twenty-three photos of a bronze statue of a woman carrying a child, taken from all sides while walking around it](https://raw.githubusercontent.com/d4per/pixelmap/main/images/docs/landala.webp)

`pixelmap_multiview` worked out where each photo was taken from and fused all 23 views into
this mesh, shown here untextured in MeshLab from three sides:

![The reconstructed 3D mesh of the statue, seen from three different angles](https://raw.githubusercontent.com/d4per/pixelmap/main/images/docs/landala3D.webp)

## Using it

Hand over the photos, say how much effort to spend, and follow along:

```rust,ignore
use pixelmap::Quality;
use pixelmap_multiview::{Flow, Focal, Options};

// `photos: Vec<Arc<pixelmap::Photo>>`, decoded by the caller, all the same size.
let options = Options::new()
    .quality(Quality::Medium)
    .focal(Focal::Equivalent35mm(28.0));

let model = pixelmap_multiview::run(&photos, &options, &mut |event| {
    println!("{}: {}", event.stage(), event.message());
    Flow::Continue(())
})?;

println!("{} triangles", model.mesh.triangles.len());
```

`Options` is the whole of what a caller chooses: the quality, what is known about the
camera's focal length, the seed, whether to refine the focal length, and how large a
texture atlas to produce. The thresholds each stage judges by — about fifty of them — are
derived from those and are deliberately not part of the API: they are meaningful only
together, and publishing them would turn tuning decisions into this crate's contract.

`run` gives back a `Model`: the mesh, its texture, the camera intrinsics, where each photo
was taken from, and which photos made it. Everything is in one frame: the first camera of
the best pair sits at the origin, and the distance between that pair's cameras is the unit
of length, so nothing is metric. Everything computed along the way — tracks, both sparse
models, the depth maps and their per-sample statistics — is behind `Model::diagnostics()`,
for working out why a reconstruction came out as it did. Most callers never look, and
`Model::take_diagnostics()` frees that memory.

To reconstruct from correspondences that did not come from pixelmap, implement
`PairLookup` for them, build a `PairGraph` with `PairGraph::from_fn`, and call
`reconstruct`.

Positions and poses are `nalgebra` types, and photos are `pixelmap` types. Both crates
are re-exported as `pixelmap_multiview::nalgebra` and `pixelmap_multiview::pixelmap`, and a
new major version of either is a breaking change for this crate.

Each way a capture can fail — a camera that only turned, a flat scene, photos with too
little in common — comes back as an `Error` that names the stage and says what to do
differently.

## The stages

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

## Feedback while it runs

A run is minutes of work, so the caller hears about it throughout. `Event` carries data
rather than prose: which pair is being mapped and how far through it, why a pair was
rejected, which photo was left out and why. Showing progress never means parsing a
sentence, and the wording of a message is never part of the contract.

| Variant | Carries |
|---|---|
| `Started` | how many views and pairs the run is |
| `Stage` | the stage, how far through it, and a line for a log |
| `PairProgress` | the pair, its index of the total, and the solver step within it |
| `PairMapped` | the pair and its finished `PairMap` |
| `PairRejected` | the pair and a typed `Degeneracy` |
| `ViewDropped` | the photo left out, the stage, and why |
| `Log` | a level and a message |

`Event::progress()` weighs the stages against each other and gives one number for the whole
run. It is an `Option`: a log line or a dropped view can happen anywhere within a stage, and
reporting the stage's start for those would drive a progress bar backwards. `Event::message()`
renders any event as one line, for callers that want a log.

`PairMapped` carries that pair's `PairMap`: the dense correspondence itself, as the raw
grid, in the coordinates of the photos handed in rather than the lower resolution the solver
works at. It is plain data — no borrows, no generics — so a frontend can draw the match for
every pair as it is found. How to draw it is the caller's decision; the crate renders
nothing.

## Stopping a run

Returning `Flow::Break(())` from the callback stops the run, which comes back as
`Error::Cancelled` naming the stage it stopped in. A stop takes effect at the next
checkpoint. Within pairwise correspondence that is one pixelmap schedule step, which is the
longest a stop ever waits; every other stage reports often enough to be prompt.

Measured on three photos of stone steps at 1200 px, asking a run to stop at various points:

| Quality | One pair | Typical wait | Worst seen |
|---------|----------|--------------|------------|
| medium  | 2.3 s    | 0.04 s       | 0.04 s     |
| high    | 5.8 s    | 0.12–0.18 s  | 0.53 s     |

So a run stops well inside a second even at the slowest setting, without interrupting a
pixelmap solve partway. `pixelmap-multiview --cancel-after SECONDS` is what measures this.

With the `threads` feature (on by default), `Job` runs a reconstruction on a worker thread
and hands back a channel of events, a `Status` to poll, and `Job::cancel()`. Turn the
feature off for `wasm32-unknown-unknown`, which has no threads to hand out; `run` works
there and is what `Job` drives underneath, so nothing is lost but the convenience.

## The command-line tool

Like the library it builds on, this crate does no I/O and has no platform dependencies, so
it builds for `wasm32-unknown-unknown` with `--no-default-features`. Decoding, EXIF and
resizing are the caller's job; the `pixelmap-multiview` binary in
[`rust/multiview_cli`](https://github.com/d4per/pixelmap/tree/main/rust/multiview_cli) of
the repository does them for photos on disc:

    cargo run --release -p pixelmap_multiview_cli -- a.jpg b.jpg c.jpg d.jpg --dump-dir out

With `--dump-dir`, the sparse model is written to `out/sparse.ply`, with coloured points and a
red pyramid per camera; the textured surface to `out/mesh.obj`, with `mesh.mtl` and
`mesh_texture.png`, and as X3D to `out/mesh.x3d`; the surface with a colour per vertex to `out/mesh_colours.ply`; and each
depth map to `out/depth_N.png`. They open in MeshLab, Blender or CloudCompare.

The CLI also prints, per photo, how much of it was matched to no other photo and where the
remaining depth samples were dropped.

Known limitation: pixelmap matches little within about 25–50 px (at 1200 px) of a photo's
edge, so a strip along each edge that only two photos share usually stays empty. Photos that
overlap generously avoid it; so does anything seen by a third photo away from its edge.

The command-line tool is not published to crates.io; build it from the repository.

## Cost

Pairwise correspondence dominates: N photos need N(N − 1)/2 pixelmap runs. One run on
a 4:3 photo pair, native build, measured with `pixelmap-multiview --time-pair`:

| Quality | 800 px | 1200 px | 1800 px |
|---------|--------|---------|---------|
| low     | 0.60 s | 0.56 s  | 0.50 s  |
| medium  | 2.25 s | 2.30 s  | 2.38 s  |
| high    |        | 5.81 s  |         |

The input size barely matters, because pixelmap scales every photo to a fixed working
width (400 px for low, 800 px for medium, 1600 px for high) before it starts. What the input
resolution buys is the precision of the geometry that follows, not matching time.

End to end, those three photos of stone steps at 1200 px and low quality reconstruct in 1.9 s into a
mesh of 23,314 vertices and 45,708 triangles, textured from 1,528 charts in a 2048 × 2048
atlas.
