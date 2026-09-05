# Rust implementation of the PIXELMAP framework

The workspace contains four crates:

- `pixelmap` — the library implementing the framework. This is the part published to
  [crates.io](https://crates.io/crates/pixelmap); see [pixelmap/README.md](pixelmap/README.md)
  for the API.
- `model_3d` — `pixelmap_model_3d`, which lifts a finished mapping into a textured 3D
  mesh and writes it as X3D. Kept out of the published library: it consumes the
  algorithm rather than being part of it, and it is the only crate here that needs a
  linear-algebra dependency.
- `command_line_tool` — the `pixelmap` binary, which writes interpolated images to disc.
- `viewer` — the `pixelmap-viewer` binary, which animates the interpolation in a window.

## Using the library

```rust
use pixelmap::{Correspondence, Photo, Quality};

let mapping = Correspondence::builder()
    .quality(Quality::Low)
    .run(photo1, photo2)?;

let (x, y) = mapping.lookup(120.0, 84.0).expect("mapped here");
```

## Clone and build the project

    # Clone the repository
    git clone git@github.com:d4per/pixelmap.git
    cd rust
    # Build the project
    cargo build --release
    # Run the command line tool
    cargo run --release --bin pixelmap -- photo1.jpg photo2.jpg

Replace photo1 and photo2 with your own images. Use --help for more options. Note that the two photos must have the same dimensions.

The order in which the solver relaxes its queue decides which local optimum the relaxation
settles into, so it is seeded. The library seeds from a fixed default, which makes runs
reproducible out of the box; the command line tool reads `PIXELMAP_SEED` to override it —
for example to check that a change to the algorithm is the only thing that moved:

    PIXELMAP_SEED=1 cargo run --release --bin pixelmap -- photo1.jpg photo2.jpg

(The library itself never reads the environment. `Correspondence::builder().seed(n)` is the
programmatic equivalent.)

## Animating the result

`pixelmap-viewer` opens a window and plays the interpolation as an animation. It either
runs the algorithm itself:

    cargo run --release --bin pixelmap-viewer -- photo1.jpg photo2.jpg --processing-mode medium

The window opens immediately and shows a progress bar while the mapping is computed; each
frame starts animating as soon as it has been rendered.

Or it replays images the command line tool has already written to disc:

    cargo run --release --bin pixelmap-viewer -- --frames interpolation_*.png

### Controls

| Key           | Action                    |
|---------------|---------------------------|
| `space`       | play / pause              |
| `left`/`right`| step one frame            |
| `up`/`down`   | faster / slower           |
| `p`           | toggle ping-pong / loop   |
| `r`           | reverse direction         |
| `esc` / `q`   | quit                      |

### Options

| Option              | Default | Meaning                                                         |
|---------------------|---------|-----------------------------------------------------------------|
| `--processing-mode` | `low`   | `low`, `medium` or `high`, as in the command line tool           |
| `--num-frames`      | `12`    | frames to render, including both end photos                      |
| `--detail-level`    | `4`     | supersampling per frame; higher is slower but leaves fewer holes |
| `--fps`             | `12`    | animation speed                                                  |
| `--hold`            | `0.4`   | seconds to linger on the first and last frame                    |
| `--no-ping-pong`    | off     | loop forwards instead of playing forwards and backwards          |
| `--max-width`       | `1200`  | scale photos down before processing (`0` keeps them as they are) |

Use `--help` for the full list.
