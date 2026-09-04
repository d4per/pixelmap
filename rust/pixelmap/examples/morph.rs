//! Morph one photo into another and write the in-between frames.
//!
//! ```sh
//! cargo run --release --example morph --features image -- a.jpg b.jpg 8
//! ```
//!
//! The two photos must have the same dimensions.

use std::process::ExitCode;

use pixelmap::{Correspondence, Photo, Quality};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [first, second, rest @ ..] = args.as_slice() else {
        eprintln!("usage: morph <photo1> <photo2> [frames]");
        return ExitCode::FAILURE;
    };
    let frames: usize = rest.first().and_then(|n| n.parse().ok()).unwrap_or(8);

    let (photo1, photo2) = match (load(first), load(second)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let mapping = match Correspondence::builder()
        .quality(Quality::Low)
        .run_with_progress(photo1, photo2, |p| {
            eprintln!("step {}/{}", p.step, p.total);
        }) {
        Ok(mapping) => mapping,
        Err(error) => {
            eprintln!("Error: {error}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("mapped {:.1}% of the image", mapping.coverage() * 100.0);

    let forward = mapping.forward();
    for frame in 0..frames {
        let alpha = frame as f32 / (frames.max(2) - 1) as f32;
        let photo = forward.interpolate_photo(alpha, 4);
        let name = format!("morph_{frame:02}.png");
        let (w, h) = (photo.width() as u32, photo.height() as u32);
        let buffer = image::RgbaImage::from_raw(w, h, photo.into_rgba())
            .expect("buffer came from a Photo of exactly these dimensions");
        if let Err(error) = buffer.save(&name) {
            eprintln!("could not write {name}: {error}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {name}");
    }

    ExitCode::SUCCESS
}

fn load(path: &str) -> Result<Photo, String> {
    // `Photo: From<DynamicImage>` comes from the `image` feature.
    image::open(path)
        .map(Photo::from)
        .map_err(|e| format!("could not read {path}: {e}"))
}
