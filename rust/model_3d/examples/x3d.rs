//! Reconstruct a 3D surface from two photos and write it as an X3D mesh.
//!
//! ```sh
//! cargo run --release -p pixelmap_model_3d --example x3d -- a.jpg b.jpg
//! ```
//!
//! The two photos must have the same dimensions, and should be two views of the same
//! static scene taken from slightly different positions. Writes `model.x3d` next to a
//! `model_texture.png` it references, so the pair can be dropped into any X3D viewer.

use std::process::ExitCode;

use pixelmap::{Correspondence, Photo, Quality};
use pixelmap_model_3d::Model3D;

const MESH: &str = "model.x3d";
const TEXTURE: &str = "model_texture.png";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [first, second] = args.as_slice() else {
        eprintln!("usage: x3d <photo1> <photo2>");
        return ExitCode::FAILURE;
    };

    let (photo1, photo2) = match (load(first), load(second)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let mapping = match Correspondence::builder()
        .quality(Quality::Medium)
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

    let model = Model3D::new(mapping.forward());

    // The mesh's texture coordinates index the first photo at the solver's working
    // resolution, which is what `Model3D` keeps a handle to — so write that one out
    // rather than the original the caller passed in.
    let texture = &model.photo;
    let (w, h) = (texture.width() as u32, texture.height() as u32);
    let buffer = image::RgbaImage::from_raw(w, h, texture.as_rgba().to_vec())
        .expect("buffer came from a Photo of exactly these dimensions");
    if let Err(error) = buffer.save(TEXTURE) {
        eprintln!("could not write {TEXTURE}: {error}");
        return ExitCode::FAILURE;
    }

    let x3d = model.to_x3d().replace("[photo_placeholder]", TEXTURE);
    if let Err(error) = std::fs::write(MESH, x3d) {
        eprintln!("could not write {MESH}: {error}");
        return ExitCode::FAILURE;
    }

    eprintln!("wrote {MESH} and {TEXTURE}");
    ExitCode::SUCCESS
}

fn load(path: &str) -> Result<Photo, String> {
    // `Photo: From<DynamicImage>` comes from pixelmap's `image` feature.
    image::open(path)
        .map(Photo::from)
        .map_err(|e| format!("could not read {path}: {e}"))
}
