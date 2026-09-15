//! `pixelmap-multiview`: reconstruct a scene from three or more photos on disc.
//!
//! The library takes RGBA buffers and intrinsics. Everything that touches files happens
//! here: decoding, EXIF orientation and focal length, resizing every photo to one common
//! size, and writing the results.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use image::imageops::FilterType;
use image::DynamicImage;
use nalgebra::Rotation3;
use pixelmap::{Correspondence, Photo, ProcessingMode, DEFAULT_SEED};
use pixelmap_multiview::depth::DepthMap;
use pixelmap_multiview::pipeline::{self, Reconstruction};
use pixelmap_multiview::sfm::SparseModel;
use pixelmap_multiview::synthetic::{Scene, SyntheticSet};
use pixelmap_multiview::tracks::Track;
use pixelmap_multiview::{
    align, export, input, Event, Flow, FocalSource, Intrinsics, PairId, Pose, Stage, ViewId, World,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Reconstruct a textured 3D model from three or more photos of the same scene."
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

    /// Write the photos as used, the sparse model, the depth maps and the textured mesh (as
    /// OBJ and X3D) into this directory.
    #[arg(long)]
    dump_dir: Option<PathBuf>,

    /// Time one correspondence run between the first two photos, then exit. Every
    /// estimate of how long a reconstruction takes derives from this number.
    #[arg(long)]
    time_pair: bool,

    /// Refine the focal length during bundle adjustment. Always on when the photos carry
    /// no EXIF focal length.
    #[arg(long)]
    refine_focal: bool,
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
            _ => "ESTIMATED: no EXIF focal length, so it will be refined",
        }
    );

    if let Some(dir) = &args.dump_dir {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
        for (index, photo) in photos.iter().enumerate() {
            save_photo(&dir.join(format!("view_{index}.png")), photo)?;
        }
    }

    let seed = seed();
    if args.time_pair {
        return time_pair(&args, &photos, seed);
    }

    let mut params = pipeline::Params {
        quality: args.processing_mode,
        seed,
        ..pipeline::Params::default()
    };
    params.ba.refine_focal = args.refine_focal;

    let start = Instant::now();
    let mut mapping_line = false;
    let result = pipeline::run(&photos, &intrinsics, &params, &mut |event| {
        print_event(&event, &mut mapping_line);
        Flow::Continue(())
    });
    if mapping_line {
        eprintln!();
    }
    let reconstruction = result.map_err(|e| e.to_string())?;
    println!("reconstructed in {:.1} s", start.elapsed().as_secs_f64());

    if let Some(truth) = &truth {
        report_truth(truth, &reconstruction);
    }
    if let Some(dir) = &args.dump_dir {
        write_outputs(dir, &photos, &reconstruction)?;
    }
    Ok(())
}

/// Prints a progress event. The many steps of mapping share one line that rewrites
/// itself; every stage summary gets a line of its own.
fn print_event(event: &Event, mapping_line: &mut bool) {
    if event.stage == Stage::Pairs && event.message.starts_with("mapping ") {
        eprint!("\r{:5.1}%  {:<72}", event.fraction() * 100.0, event.message);
        *mapping_line = true;
        return;
    }
    if *mapping_line {
        eprint!("\r{:<80}\r", "");
        *mapping_line = false;
    }
    println!(
        "{:5.1}%  {}: {}",
        event.fraction() * 100.0,
        event.stage,
        event.message
    );
}

/// Compares every stage of a reconstruction of a synthetic scene with the ground truth.
fn report_truth(truth: &SyntheticSet, r: &Reconstruction) {
    println!();
    println!("against the ground truth:");
    for report in &r.pairs {
        if let Ok(estimate) = &report.estimate {
            let expected = truth.relative_pose(report.pair);
            println!(
                "  {}: rotation off by {:.2}°, translation direction off by {:.2}°",
                report.pair,
                angle_between(&estimate.pose.rotation, &expected.rotation),
                estimate
                    .pose
                    .translation
                    .angle(&expected.translation)
                    .to_degrees()
            );
        }
    }
    if let Some((_, cameras, points)) = align_with_truth(truth, &r.tracks, &r.registered) {
        println!(
            "  registered: cameras off by at most {cameras:.2}%, points by a median {points:.2}% of the camera spread"
        );
    }
    let Some((similarity, cameras, points)) = align_with_truth(truth, &r.tracks, &r.adjusted)
    else {
        eprintln!("could not align the reconstruction with the ground truth");
        return;
    };
    println!(
        "  bundle adjusted: cameras off by at most {cameras:.2}%, points by a median {points:.2}% of the camera spread"
    );

    let mut depth_errors: Vec<f64> = r
        .depth
        .iter()
        .flat_map(|map| {
            (0..map.rows)
                .flat_map(move |row| (0..map.columns).map(move |column| (map, column, row)))
        })
        .filter_map(|(map, column, row)| {
            let d = map.get(column, row)? as f64 * similarity.scale;
            let t = truth.depth(map.view, map.pixel(column, row))?;
            Some((d - t).abs() / t * 100.0)
        })
        .collect();
    if !depth_errors.is_empty() {
        println!(
            "  depth: median error {:.2}%, 95th percentile {:.2}%",
            percentile(&mut depth_errors, 0.5),
            percentile(&mut depth_errors, 0.95)
        );
    }

    let extent = camera_spread(truth);
    let mesh = &r.fused.mesh;
    let mut distances: Vec<f64> = mesh
        .positions
        .iter()
        .map(|p| truth.scene.distance(&similarity.apply(p)) / extent * 100.0)
        .collect();
    let mut colour_errors: Vec<f64> = mesh
        .positions
        .iter()
        .zip(&r.texture.vertex_colours)
        .map(|(p, colour)| {
            let expected = truth.scene.colour(&similarity.apply(p));
            (0..3)
                .map(|k| (colour[k] as f64 - expected[k] as f64).abs())
                .sum::<f64>()
                / 3.0
        })
        .collect();
    if !distances.is_empty() {
        println!(
            "  mesh: median distance {:.2}%, 90th percentile {:.2}% of the camera spread",
            percentile(&mut distances, 0.5),
            percentile(&mut distances, 0.9)
        );
        println!(
            "  vertex colours: off by a median {:.1}, 90th percentile {:.1} levels of 255",
            percentile(&mut colour_errors, 0.5),
            percentile(&mut colour_errors, 0.9)
        );
    }
}

/// Aligns `model` to the truth on its cameras and points together. Returns the
/// similarity, the largest camera error and the median point error, both as percentages
/// of the true cameras' spread.
fn align_with_truth(
    truth: &SyntheticSet,
    tracks: &[Track],
    model: &SparseModel,
) -> Option<(align::Similarity, f64, f64)> {
    let mut estimated = Vec::new();
    let mut expected = Vec::new();
    for view in model.registered() {
        estimated.push(model.cameras[view.index()]?.centre());
        expected.push(truth.poses[view.index()].centre());
    }
    let cameras = estimated.len();
    for point in &model.points {
        let track = &tracks[point.track];
        let surface = track
            .observation(track.anchor)
            .and_then(|p| truth.surface_point(track.anchor, p));
        if let Some(surface) = surface {
            estimated.push(point.position.0);
            expected.push(surface.0);
        }
    }
    let similarity = align::umeyama(&estimated, &expected)?;
    let extent = camera_spread(truth);
    let mut errors: Vec<f64> = estimated
        .iter()
        .zip(&expected)
        .map(|(e, x)| (similarity.apply(e) - x).norm() / extent * 100.0)
        .collect();
    let camera_error = errors[..cameras].iter().copied().fold(0.0, f64::max);
    let point_error = percentile(&mut errors[cameras..], 0.5);
    Some((similarity, camera_error, point_error))
}

fn camera_spread(truth: &SyntheticSet) -> f64 {
    truth
        .poses
        .iter()
        .flat_map(|a| {
            truth
                .poses
                .iter()
                .map(move |b| (a.centre() - b.centre()).norm())
        })
        .fold(0.0, f64::max)
}

/// The angle, in degrees, of the rotation taking `b` to `a`.
fn angle_between(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    // `Rotation3::angle` does not clamp, and a near-perfect estimate lands just above 1.
    let cos = ((a * b.inverse()).matrix().trace() - 1.0) / 2.0;
    cos.clamp(-1.0, 1.0).acos().to_degrees()
}

fn percentile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q) as usize]
}

/// Writes the sparse model, depth maps and textured mesh into `dir`.
fn write_outputs(dir: &Path, photos: &[Arc<Photo>], r: &Reconstruction) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;

    let points: Vec<(World, [u8; 3])> = r
        .adjusted
        .points
        .iter()
        .map(|p| {
            let track = &r.tracks[p.track];
            let anchor = track
                .observation(track.anchor)
                .expect("a track observes its anchor");
            let colour = export::sample_colour(&photos[track.anchor.index()], anchor);
            (p.position, colour)
        })
        .collect();
    let cameras: Vec<Pose> = r.adjusted.cameras.iter().flatten().copied().collect();
    write_file(&dir.join("sparse.ply"), |out| {
        export::write_ply(out, &points, &cameras, &r.intrinsics, r.size, 0.3)
    })?;

    for map in &r.depth {
        write_depth_png(&dir.join(format!("depth_{}.png", map.view.0)), map)?;
    }

    let mesh = &r.fused.mesh;
    write_file(&dir.join("mesh.obj"), |out| {
        export::write_textured_obj(out, mesh, &r.texture, "mesh.mtl")
    })?;
    write_file(&dir.join("mesh.mtl"), |out| {
        export::write_mtl(out, "mesh_texture.png")
    })?;
    save_photo(&dir.join("mesh_texture.png"), &r.texture.atlas)?;
    write_file(&dir.join("mesh.x3d"), |out| {
        export::write_textured_x3d(out, mesh, &r.texture, "mesh_texture.png")
    })?;
    write_file(&dir.join("mesh_colours.ply"), |out| {
        export::write_ply_mesh(out, mesh, &r.texture.vertex_colours)
    })?;

    eprintln!(
        "wrote view_N.png, sparse.ply, depth_N.png, mesh.obj, mesh.mtl, mesh.x3d, mesh_texture.png and mesh_colours.ply to {}",
        dir.display()
    );
    Ok(())
}

fn write_file(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
) -> Result<(), String> {
    File::create(path)
        .and_then(|file| {
            let mut out = BufWriter::new(file);
            write(&mut out)?;
            out.flush()
        })
        .map_err(|e| format!("could not write {}: {e}", path.display()))
}

fn save_photo(path: &Path, photo: &Photo) -> Result<(), String> {
    image::RgbaImage::from_raw(
        photo.width() as u32,
        photo.height() as u32,
        photo.as_rgba().to_vec(),
    )
    .expect("buffer came from a Photo of exactly these dimensions")
    .save(path)
    .map_err(|e| format!("could not write {}: {e}", path.display()))
}

/// A depth map as a greyscale image: near is bright, far is dark, unknown is black.
fn write_depth_png(path: &Path, map: &DepthMap) -> Result<(), String> {
    let mut known: Vec<f32> = map
        .depth
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    known.sort_by(f32::total_cmp);
    let (near, far) = match known.len() {
        0 => (0.0, 1.0),
        n => (known[n / 50], known[n - 1 - n / 50]),
    };
    let range = (far - near).max(f32::EPSILON);
    let pixels = map
        .depth
        .iter()
        .map(|&d| {
            if d.is_finite() {
                (255.0 - ((d - near) / range).clamp(0.0, 1.0) * 215.0) as u8
            } else {
                0
            }
        })
        .collect();
    image::GrayImage::from_raw(map.columns as u32, map.rows as u32, pixels)
        .expect("one byte per sample")
        .save(path)
        .map_err(|e| format!("could not write {}: {e}", path.display()))
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
