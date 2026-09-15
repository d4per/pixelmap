//! `pixelmap-multiview`: reconstruct a scene from three or more photos on disc.
//!
//! The library takes RGBA buffers and intrinsics. Everything that touches files happens
//! here: decoding, EXIF orientation and focal length, and resizing every photo to one
//! common size.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use image::imageops::FilterType;
use image::DynamicImage;
use nalgebra::Rotation3;
use pixelmap::{Correspondence, Photo, ProcessingMode, DEFAULT_SEED};
use pixelmap_multiview::pairs::{self, MIN_PAIR_COVERAGE};
use pixelmap_multiview::rng::Rng;
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::twoview::{self, Params, Verdict};
use pixelmap_multiview::{input, Flow, FocalSource, Intrinsics, PairId, ViewId};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Reconstruct a 3D model from three or more photos of the same scene."
)]
struct Args {
    /// Photos of the scene, all from the same camera at the same zoom. At least three.
    #[arg(required_unless_present = "synthetic", num_args = 3..)]
    photos: Vec<PathBuf>,

    /// Render views of a synthetic scene instead of reading photos: plane, sphere or
    /// corner. The results are then checked against the known geometry.
    #[arg(long, value_name = "SCENE", conflicts_with = "photos")]
    synthetic: Option<String>,

    /// How many views of the synthetic scene to render.
    #[arg(long, default_value_t = 4)]
    views: usize,

    /// The angle, in degrees, the synthetic cameras spread across.
    #[arg(long, default_value_t = 30.0)]
    spread: f64,

    /// Scale photos down so their long edge is at most this many pixels. 0 keeps them as
    /// they are. For a synthetic scene, the width to render at.
    #[arg(long, default_value_t = 1200)]
    long_edge: u32,

    /// Focal length in pixels of the original photos. Overrides EXIF.
    #[arg(long)]
    focal_px: Option<f64>,

    /// Processing mode for pairwise correspondence: low, medium, or high.
    #[arg(long, default_value = "low")]
    processing_mode: ProcessingMode,

    /// Write intermediate results (the normalized photos, for now) into this directory.
    #[arg(long)]
    dump_dir: Option<PathBuf>,

    /// Time one correspondence run between the first two photos, then exit. Every
    /// estimate of how long a reconstruction takes derives from this number.
    #[arg(long)]
    time_pair: bool,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Photos ready for the library, and the ground truth when they were rendered.
struct Input {
    photos: Vec<Arc<Photo>>,
    intrinsics: Intrinsics,
    truth: Option<SyntheticSet>,
}

fn run(args: Args) -> Result<(), String> {
    let Input {
        photos,
        intrinsics,
        truth,
    } = match &args.synthetic {
        Some(name) => synthetic_input(&args, name)?,
        None => photo_input(&args)?,
    };

    let (width, height) = input::validate(&photos, &intrinsics).map_err(|e| e.to_string())?;
    eprintln!(
        "{} photos at {width}×{height}, {} pairs",
        photos.len(),
        PairId::count(photos.len())
    );
    eprintln!(
        "focal length {:.1} px ({})",
        intrinsics.fx,
        match intrinsics.source {
            FocalSource::Provided => "provided",
            FocalSource::Exif35mm => "from EXIF",
            _ => "ESTIMATED: no EXIF focal length, the model may be skewed",
        }
    );

    if let Some(dir) = &args.dump_dir {
        dump_photos(dir, &photos)?;
    }

    let seed = seed();
    if args.time_pair {
        return time_pair(&args, &photos, seed);
    }

    let start = Instant::now();
    let graph = pairs::compute(&photos, args.processing_mode, seed, &mut |event| {
        eprint!("\r{:5.1}%  {:<60}", event.fraction() * 100.0, event.message);
        Flow::Continue(())
    })
    .map_err(|e| e.to_string())?;
    eprintln!(
        "\rmapped every pair in {:.1} s{:<60}",
        start.elapsed().as_secs_f64(),
        ""
    );

    let connected =
        pairs::require_connected(&graph, MIN_PAIR_COVERAGE).map_err(|e| e.to_string())?;
    if connected.len() < photos.len() {
        let dropped: Vec<String> = (0..photos.len() as u32)
            .map(ViewId)
            .filter(|v| !connected.contains(v))
            .map(|v| v.to_string())
            .collect();
        eprintln!(
            "warning: dropping {}: no pair with at least {:.0}% coverage links it to the rest",
            dropped.join(", "),
            MIN_PAIR_COVERAGE * 100.0
        );
    }

    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8} {:>9}  verdict",
        "pair", "coverage", "matches", "inliers", "angle", "parallax"
    );
    let root = Rng::new(seed);
    for (index, (pair, mapping)) in graph.pairs().enumerate() {
        let estimate = twoview::estimate(
            pair,
            mapping,
            &intrinsics,
            (width, height),
            &Params::default(),
            &mut root.derive(index as u64),
        );
        match estimate {
            Ok(estimate) => {
                let verdict = match &estimate.verdict {
                    Verdict::Usable => "usable".to_string(),
                    Verdict::Degenerate(reason) => reason.to_string(),
                };
                println!(
                    "{:<10} {:>7.1}% {:>8} {:>7.1}% {:>7.1}° {:>8.2}°  {verdict}",
                    pair.to_string(),
                    mapping.coverage() * 100.0,
                    estimate.matches,
                    estimate.inlier_ratio * 100.0,
                    estimate.median_angle_deg,
                    estimate.structure_deg,
                );
                if let Some(truth) = &truth {
                    let expected = truth.relative_pose(pair);
                    println!(
                        "{:<10} against the true pose: rotation off by {:.2}°, translation direction off by {:.2}°",
                        "",
                        angle_between(&estimate.pose.rotation, &expected.rotation),
                        estimate
                            .pose
                            .translation
                            .angle(&expected.translation)
                            .to_degrees(),
                    );
                }
            }
            Err(reason) => println!(
                "{:<10} {:>7.1}% {:>8} {:>8} {:>8} {:>9}  {reason}",
                pair.to_string(),
                mapping.coverage() * 100.0,
                "-",
                "-",
                "-",
                "-"
            ),
        }
    }

    eprintln!("registration and the stages after it are not implemented yet");
    Ok(())
}

/// The angle, in degrees, of the rotation taking `b` to `a`.
fn angle_between(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    // `Rotation3::angle` does not clamp, and a near-perfect estimate lands just above 1.
    let cos = ((a * b.inverse()).matrix().trace() - 1.0) / 2.0;
    cos.clamp(-1.0, 1.0).acos().to_degrees()
}

fn seed() -> u64 {
    std::env::var("PIXELMAP_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SEED)
}

fn synthetic_input(args: &Args, name: &str) -> Result<Input, String> {
    let scene = Scene::named(name).ok_or_else(|| {
        format!(
            "unknown scene {name:?}; choose one of {}",
            Scene::NAMES.join(", ")
        )
    })?;
    let width = if args.long_edge > 0 {
        args.long_edge as usize
    } else {
        800
    };
    let height = width * 3 / 4;
    let set = SyntheticSet::orbit(scene, args.views, args.spread, width, height);
    eprintln!(
        "rendering {} views of the {name} scene across {}°",
        args.views, args.spread
    );
    let photos = (0..set.views())
        .map(|v| Arc::new(set.render(ViewId(v as u32))))
        .collect();
    Ok(Input {
        photos,
        intrinsics: set.intrinsics,
        truth: Some(set),
    })
}

fn photo_input(args: &Args) -> Result<Input, String> {
    let loaded = args
        .photos
        .iter()
        .map(|path| load(path))
        .collect::<Result<Vec<_>, _>>()?;

    let original = (loaded[0].image.width(), loaded[0].image.height());
    for photo in &loaded[1..] {
        let dimensions = (photo.image.width(), photo.image.height());
        if dimensions != original {
            return Err(format!(
                "{} is {}×{} but {} is {}×{}; all photos must come from one camera at one resolution",
                photo.path.display(),
                dimensions.0,
                dimensions.1,
                loaded[0].path.display(),
                original.0,
                original.1,
            ));
        }
    }

    let long = original.0.max(original.1);
    let scale = if args.long_edge > 0 && long > args.long_edge {
        args.long_edge as f64 / long as f64
    } else {
        1.0
    };
    let size = (
        (original.0 as f64 * scale).round() as u32,
        (original.1 as f64 * scale).round() as u32,
    );
    let (width, height) = (size.0 as usize, size.1 as usize);

    let intrinsics = intrinsics(args, &loaded, original, size)?;

    let photos = loaded
        .into_iter()
        .map(|photo| {
            let image = if scale < 1.0 {
                photo
                    .image
                    .resize_exact(size.0, size.1, FilterType::CatmullRom)
            } else {
                photo.image
            };
            let rgba = image.into_rgba8().into_raw();
            Photo::from_rgba(width, height, rgba)
                .map(Arc::new)
                .map_err(|e| format!("{}: {e}", photo.path.display()))
        })
        .collect::<Result<_, _>>()?;

    Ok(Input {
        photos,
        intrinsics,
        truth: None,
    })
}

/// A decoded photo, already turned upright.
struct Loaded {
    path: PathBuf,
    image: DynamicImage,
    focal_35mm: Option<f64>,
}

fn load(path: &Path) -> Result<Loaded, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let exif = exif::Reader::new()
        .read_from_container(&mut std::io::Cursor::new(&bytes))
        .ok();
    let tag = |tag| {
        exif.as_ref()?
            .get_field(tag, exif::In::PRIMARY)?
            .value
            .get_uint(0)
    };
    let orientation = tag(exif::Tag::Orientation).unwrap_or(1);
    let focal_35mm = tag(exif::Tag::FocalLengthIn35mmFilm)
        .filter(|&f| f > 0)
        .map(f64::from);

    let image = image::load_from_memory(&bytes)
        .map_err(|e| format!("could not decode {}: {e}", path.display()))?;
    Ok(Loaded {
        path: path.to_path_buf(),
        image: upright(image, orientation),
        focal_35mm,
    })
}

/// Applies an EXIF orientation, so that every photo is stored the way it is displayed.
fn upright(image: DynamicImage, orientation: u32) -> DynamicImage {
    match orientation {
        2 => image.fliph(),
        3 => image.rotate180(),
        4 => image.flipv(),
        5 => image.rotate90().fliph(),
        6 => image.rotate90(),
        7 => image.rotate270().fliph(),
        8 => image.rotate270(),
        _ => image,
    }
}

/// The camera for the resized photos: `--focal-px` first, then EXIF, then a guess.
fn intrinsics(
    args: &Args,
    loaded: &[Loaded],
    original: (u32, u32),
    size: (u32, u32),
) -> Result<Intrinsics, String> {
    let (width, height) = (size.0 as usize, size.1 as usize);
    if let Some(focal) = args.focal_px {
        let factor = size.0 as f64 / original.0 as f64;
        return Ok(Intrinsics::from_focal(
            focal * factor,
            width,
            height,
            FocalSource::Provided,
        ));
    }

    let focal = loaded[0].focal_35mm;
    if let Some(other) = loaded.iter().find(|photo| photo.focal_35mm != focal) {
        return Err(format!(
            "{} and {} report different focal lengths ({} and {}); the photos must be taken at one fixed zoom. \
             Pass --focal-px to override",
            loaded[0].path.display(),
            other.path.display(),
            describe_focal(focal),
            describe_focal(other.focal_35mm),
        ));
    }

    Ok(match focal {
        Some(focal) => Intrinsics::from_35mm(focal, width, height),
        None => Intrinsics::estimated(width, height),
    })
}

fn describe_focal(focal: Option<f64>) -> String {
    focal.map_or_else(|| "none".to_string(), |f| format!("{f} mm equivalent"))
}

fn dump_photos(dir: &Path, photos: &[Arc<Photo>]) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    for (index, photo) in photos.iter().enumerate() {
        let path = dir.join(format!("view_{index}.png"));
        let buffer = image::RgbaImage::from_raw(
            photo.width() as u32,
            photo.height() as u32,
            photo.as_rgba().to_vec(),
        )
        .expect("buffer came from a Photo of exactly these dimensions");
        buffer
            .save(&path)
            .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    }
    eprintln!("wrote {} photos to {}", photos.len(), dir.display());
    Ok(())
}

fn time_pair(args: &Args, photos: &[Arc<Photo>], seed: u64) -> Result<(), String> {
    let start = Instant::now();
    let mapping = Correspondence::builder()
        .quality(args.processing_mode)
        .seed(seed)
        .run(photos[0].clone(), photos[1].clone())
        .map_err(|e| e.to_string())?;
    let elapsed = start.elapsed();

    let pairs = PairId::count(photos.len());
    println!(
        "{}×{} {}: {:.2} s for one pair, coverage {:.1}%; about {:.0} s for all {pairs} pairs",
        photos[0].width(),
        photos[0].height(),
        args.processing_mode,
        elapsed.as_secs_f64(),
        mapping.coverage() * 100.0,
        elapsed.as_secs_f64() * pairs as f64,
    );
    Ok(())
}
