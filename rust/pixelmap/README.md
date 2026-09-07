# pixelmap

[![crates.io](https://img.shields.io/crates/v/pixelmap.svg)](https://crates.io/crates/pixelmap)
[![docs.rs](https://docs.rs/pixelmap/badge.svg)](https://docs.rs/pixelmap)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Dense image correspondence: given two photographs of the same scene, work out where
each pixel of the first one went in the second.

This is the reference implementation of the **PIXELMAP** framework
([white paper](https://doi.org/10.36227/techrxiv.173749998.89779329/v1)). Every cell of
an *affine correspondence grid* acts as an autonomous agent holding its own local affine
transform; agents refine their transform against the image data and propagate what they
find to their neighbours, and a forward/backward consistency check culls the ones that
disagree. Repeating that coarse-to-fine yields a dense, geometrically consistent mapping.

Useful for optical flow, image registration and stitching, stereo matching, morphing, and
as the front half of a 3D reconstruction — the `pixelmap_model_3d` crate in the
[repository](https://github.com/d4per/pixelmap) lifts a finished mapping into a textured
3D mesh.

![Two photographs of a monkey statue taken from different positions, above three renderings of the correspondence recovered between them](https://raw.githubusercontent.com/d4per/pixelmap/v0.2.0/images/docs/apa.webp)

*Two views of the same statue (top) and the correspondence recovered between them,
rendered at three settings (bottom). Regions left blank are those with no accepted
match.*

## Example

```rust,no_run
use pixelmap::{Correspondence, Photo, Quality};

let a = Photo::from_rgba(width, height, rgba_bytes)?;
let b = Photo::from_rgba(width, height, other_bytes)?;   // same dimensions

let mapping = Correspondence::builder()
    .quality(Quality::Low)
    .run(a, b)?;

// Where did the pixel at (120, 84) go?
match mapping.lookup(120.0, 84.0) {
    Some((x, y)) => println!("({x:.1}, {y:.1})"),
    None => println!("not mapped here"),
}

println!("mapped {:.0}% of the image", mapping.coverage() * 100.0);
```

With the `image` feature, `Photo::from(image::open("a.jpg")?)` replaces the decoding step.

Regions the algorithm could not map — occlusions, featureless sky, anything the
forward/backward consistency check rejected — come back as `None` rather than as a
sentinel value, so "no mapping here" cannot be mistaken for a coordinate.

`lookup` answers in the coordinates of the photos you passed in. The solver works at a
reduced resolution internally; `forward()` and `backward()` expose those raw grids, and
`working_scale()` relates the two.

## Quality

`Quality` picks the iteration schedule, trading time for accuracy:

| Mode | Working widths | Iterations |
|---|---|---|
| `Low` | 400 | 4 |
| `Medium` | 400, 800 | 10 |
| `High` | 400, 800, 1600 | 13 |

`Builder::schedule` takes an explicit list of `IterationParams` for callers tuning the
algorithm itself.

## Reproducibility

The order in which the solver drains its queue decides which local optimum the relaxation
settles into, so it is seeded, and the seed defaults to `DEFAULT_SEED`. The same photos,
schedule and seed give the same mapping — run to run, thread to thread, and machine to
machine. Use `Builder::seed` to vary it. Enabling or disabling `parallel` does not change
the result.

## Threading

Everything a caller holds is `Send + Sync`, so a mapping can be computed on a worker
thread and shared afterwards, and independent pairs can be mapped concurrently without
influencing each other.

## Errors

Bad input is reported, never panicked on: `Error::SizeMismatch` when the photos disagree
on dimensions, `Error::PhotoTooSmall`, `Error::EmptyPhoto`, and `Error::BufferLength` when
a pixel buffer does not match the dimensions it claims.

A finished mapping can be written out with `DensePhotoMap::serialize` and read back with
`DensePhotoMap::deserialize`. The encoding carries a magic number and a version, and the
decoder validates every field before using it, so bytes off a disc or a network come back
as `Error::Decode` rather than as a panic.

## Features

| Feature | Default | Effect |
|---|:---:|---|
| `parallel` | ✅ | Multi-threaded feature matching via rayon. Turn it off for `wasm32-unknown-unknown`, which has no threads to hand out; the matcher falls back to a serial search that produces the same result. |
| `image` | | `From<image::RgbaImage>` and `From<image::DynamicImage>` for `Photo`, for callers that already decode with the [`image`](https://crates.io/crates/image) crate. |
| `bench` | | Compiles the matcher benchmark harness. Not part of the pipeline. |

The only default dependency is `rayon`, and with `--no-default-features` there are **no
dependencies at all**. The crate does no file or
network I/O and never writes to stdout — a `Photo` is a plain RGBA byte buffer, decoding
images is the caller's business, and progress is reported through
`Builder::run_with_progress` rather than printed.

## Performance

The scoring inner loop walks `photo2` in 16.16 fixed point, and the per-pixel step it
vectorizes to is a 32-bit integer multiply — an instruction (`pmulld`) that SSE2, the
baseline `x86_64` targets by default, does not have. Building with at least

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=x86-64-v2"]
```

lets that loop vectorize properly. `target-cpu=native` doubles the lane count again on an
AVX2 host, at the cost of a binary that will not run on older hardware.

## Minimum supported Rust version

1.80, set by `rayon`, and checked by CI. Treated as a breaking change if raised.

## License

MIT — see [LICENSE](LICENSE).
