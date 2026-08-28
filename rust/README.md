# Rust implementation of the PIXELMAP framework

The workspace contains three crates:

- `pixelmap` — the library implementing the framework.
- `command_line_tool` — the `pixelmap` binary, which writes interpolated images to disc.
- `viewer` — the `pixelmap-viewer` binary, which animates the interpolation in a window.

## Clone and build the project

    # Clone the repository
    git clone git@github.com:d4per/pixelmap.git
    cd rust
    # Build the project
    cargo build --release
    # Run the command line tool
    cargo run --release --bin pixelmap -- photo1.jpg photo2.jpg

Replace photo1 and photo2 with your own images. Use --help for more options. Note that the two photos must have the same dimensions.

The order in which the solver relaxes its queue is randomised, so two runs on the same
input do not produce exactly the same mapping. Set `PIXELMAP_SEED` to pin it when you need
a reproducible result — for example to compare two builds:

    PIXELMAP_SEED=1 cargo run --release --bin pixelmap -- photo1.jpg photo2.jpg

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
