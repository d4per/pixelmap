//! Reconstruct a 3D surface from two photos and write it as an X3D mesh.
//!
//! ```sh
//! cargo run --release -p pixelmap_model_3d --example x3d -- a.jpg b.jpg [f1 f2]
//! ```
//!
//! The two photos must have the same dimensions, and should be two views of the same
//! static scene taken from slightly different positions. `f1` and `f2` are the 35 mm
//! equivalent focal lengths of the two shots, as most cameras record in EXIF; without
//! them a typical phone lens is assumed. `--projection perspective` or `--projection
//! affine` overrides the automatic choice of camera model. Writes `model.x3d` next to a
//! `model_texture.png` it references, so the pair can be dropped into any X3D viewer.

use std::process::ExitCode;

use pixelmap::{Correspondence, Photo, Quality};
use pixelmap_model_3d::{Model3D, Projection, Settings};

const MESH: &str = "model.x3d";
const TEXTURE: &str = "model_texture.png";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut projection = Projection::Auto;
    if let Some(at) = args.iter().position(|a| a == "--projection") {
        projection = match args.get(at + 1).map(String::as_str) {
            Some("auto") => Projection::Auto,
            Some("perspective") => Projection::Perspective,
            Some("affine") => Projection::Affine,
            _ => {
                eprintln!("--projection takes auto, perspective or affine");
                return ExitCode::FAILURE;
            }
        };
        args.drain(at..at + 2);
    }
    let (first, second, focal) = match args.as_slice() {
        [a, b] => (a, b, None),
        [a, b, f1, f2] => match (f1.parse::<f64>(), f2.parse::<f64>()) {
            (Ok(f1), Ok(f2)) => (a, b, Some([f1, f2])),
            _ => {
                eprintln!("focal lengths must be numbers, in millimetres");
                return ExitCode::FAILURE;
            }
        },
        _ => {
            eprintln!(
                "usage: x3d <photo1> <photo2> [focal1_35mm focal2_35mm] \
                 [--projection auto|perspective|affine]"
            );
            return ExitCode::FAILURE;
        }
    };

    let (photo1, photo2) = match (load(first), load(second)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let mut settings = Settings {
        projection,
        ..Settings::default()
    };
    if let Some(millimetres) = focal {
        let (w, h) = (photo1.width(), photo1.height());
        settings.focal_lengths = millimetres.map(|mm| Settings::focal_length_from_35mm(mm, w, h));
    }

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

    let model = Model3D::with_settings(mapping.forward(), &settings);

    eprintln!("projection: {:?}", model.projection());
    if let Some(g) = model.two_view_geometry() {
        eprintln!(
            "epipolar inliers {:.1}%, homography inliers {:.1}%, median parallax {:.2}°, {}",
            g.inlier_fraction * 100.0,
            g.homography_inlier_fraction * 100.0,
            g.median_parallax_degrees,
            if g.is_well_conditioned() {
                "well conditioned"
            } else {
                "not well conditioned"
            }
        );
        let angle = ((g.rotation.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        let t = g.translation;
        eprintln!(
            "camera 2 turned {:.2}°, baseline direction ({:.2}, {:.2}, {:.2})",
            angle.to_degrees(),
            t.x,
            t.y,
            t.z
        );
    }

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
