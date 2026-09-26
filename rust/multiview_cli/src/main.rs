//! `pixelmap-multiview`: reconstruct a scene from three or more photos on disc.
//!
//! The library takes RGBA buffers and options. Everything that touches files happens
//! here: decoding, EXIF orientation and focal length, resizing every photo to one common
//! size, and writing the results.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use image::imageops::FilterType;
use image::DynamicImage;
use pixelmap::{Correspondence, Photo, ProcessingMode, DEFAULT_SEED};
use pixelmap_multiview::depth::{self, DepthMap};
use pixelmap_multiview::{
    export, input, Error, Event, Flow, Focal, FocalSource, Job, Model, Options, PairId, Pose, World,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Reconstruct a textured 3D model from three or more photos of the same scene."
)]
struct Args {
    /// Photos of the scene, all from the same camera at the same zoom. At least three.
    #[arg(required = true, num_args = 3..)]
    photos: Vec<PathBuf>,

    /// Scale photos down so their long edge is at most this many pixels. 0 keeps them as
    /// they are.
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

    /// Ask the run to stop after this many seconds, and report how long it actually took
    /// to stop. How promptly a run can be cancelled is a claim worth measuring rather than
    /// assuming: the wait is bounded by one pixelmap schedule step, which grows with
    /// `--processing-mode`.
    #[arg(long, value_name = "SECONDS")]
    cancel_after: Option<f64>,
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

/// Photos ready for the library.
struct Input {
    photos: Vec<Arc<Photo>>,
    focal: Focal,
}

fn run(args: Args) -> Result<(), String> {
    let Input { photos, focal } = photo_input(&args)?;

    // The library works this out for itself from the options; asking the same question
    // here gives something to show before the run starts.
    let first = photos.first().ok_or("no photos")?;
    let intrinsics = focal.intrinsics(first.width(), first.height());
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

    let options = Options::new()
        .quality(args.processing_mode)
        .focal(focal)
        .seed(seed)
        .refine_focal(args.refine_focal);

    let start = Instant::now();
    let result = match args.cancel_after {
        Some(seconds) => cancel_after(photos.clone(), options, seconds),
        None => {
            let mut mapping_line = false;
            let result = pixelmap_multiview::run(&photos, &options, &mut |event| {
                print_event(&event, &mut mapping_line);
                Flow::Continue(())
            });
            if mapping_line {
                eprintln!();
            }
            result
        }
    };
    let model = result.map_err(|e| e.to_string())?;
    println!("reconstructed in {:.1} s", start.elapsed().as_secs_f64());
    print_depth_stats(&model);

    if let Some(dir) = &args.dump_dir {
        write_outputs(dir, &photos, &model)?;
    }
    Ok(())
}

/// Runs on a worker thread and stops it after `seconds`, reporting how long the stop took
/// to take effect.
fn cancel_after(photos: Vec<Arc<Photo>>, options: Options, seconds: f64) -> Result<Model, Error> {
    let job = Job::start(photos, options);
    let canceller = job.canceller();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs_f64(seconds));
        canceller.cancel();
    });

    let asked_at = Instant::now();
    let mut mapping_line = false;
    for event in job.events() {
        print_event(&event, &mut mapping_line);
    }
    if mapping_line {
        eprintln!();
    }
    let elapsed = asked_at.elapsed().as_secs_f64();
    println!(
        "asked to stop after {seconds:.1} s; the run ended {:.2} s later, at {elapsed:.2} s",
        (elapsed - seconds).max(0.0)
    );
    job.join()
}

/// Prints an event. The many steps of mapping share one line that rewrites itself; every
/// other event gets a line of its own.
fn print_event(event: &Event, mapping_line: &mut bool) {
    if let Event::PairProgress { .. } = event {
        let percent = event.progress().unwrap_or(0.0) * 100.0;
        eprint!("\r{percent:5.1}%  {:<72}", event.message());
        *mapping_line = true;
        return;
    }
    if *mapping_line {
        eprint!("\r{:<80}\r", "");
        *mapping_line = false;
    }
    match event.progress() {
        Some(progress) => println!(
            "{:5.1}%  {}: {}",
            progress * 100.0,
            event.stage(),
            event.message()
        ),
        // A log line or a dropped view says nothing about how far along the run is.
        None => println!("        {}: {}", event.stage(), event.message()),
    }
}

/// How dense depth treated each photo's samples, by how many other photos they map into.
fn print_depth_stats(model: &Model) {
    let percent = |part: usize, whole: usize| 100.0 * part as f64 / whole.max(1) as f64;
    let describe = |counts: &depth::Counts| {
        format!(
            "{} samples, {:.0}% kept; dropped {:.0}% by fit, {:.0}% by narrow ray angle, {:.0}% by consistency, {:.0}% as speckle",
            counts.samples,
            percent(counts.kept, counts.samples),
            percent(counts.rejected_by_fit, counts.samples),
            percent(counts.rejected_by_angle, counts.samples),
            percent(counts.rejected_by_consistency, counts.samples),
            percent(counts.removed_as_speckle, counts.samples),
        )
    };
    println!("dense depth per photo:");
    let Some(d) = model.diagnostics() else {
        return;
    };
    for stats in &d.depth_stats {
        let total = stats.unmapped + stats.single.samples + stats.multiple.samples;
        println!(
            "  {}: {:.0}% matched to no other photo",
            stats.view,
            percent(stats.unmapped, total)
        );
        if percent(stats.unmapped, total) >= 10.0 {
            println!(
                "    those parts cannot be reconstructed: no other photo shows them, or the matcher \
                 could not follow them there. Another photo overlapping them would help"
            );
        }
        println!(
            "    matched to one other photo: {}",
            describe(&stats.single)
        );
        println!(
            "    matched to two or more:     {}",
            describe(&stats.multiple)
        );
    }
}

/// Writes the sparse model, depth maps and textured mesh into `dir`.
fn write_outputs(dir: &Path, photos: &[Arc<Photo>], model: &Model) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let d = model
        .diagnostics()
        .ok_or("the model's diagnostics were dropped")?;

    let points: Vec<(World, [u8; 3])> = d
        .adjusted
        .points
        .iter()
        .map(|p| {
            let track = &d.tracks[p.track];
            let anchor = track
                .observation(track.anchor)
                .expect("a track observes its anchor");
            let colour = export::sample_colour(&photos[track.anchor.index()], anchor);
            (p.position, colour)
        })
        .collect();
    let cameras: Vec<Pose> = d.adjusted.cameras.iter().flatten().copied().collect();
    write_file(&dir.join("sparse.ply"), |out| {
        export::write_ply(out, &points, &cameras, &model.intrinsics, d.size, 0.3)
    })?;

    for map in &d.depth {
        write_depth_png(&dir.join(format!("depth_{}.png", map.view.0)), map)?;
    }
    for (map, stats) in d.depth.iter().zip(&d.depth_stats) {
        write_coverage_png(
            &dir.join(format!("coverage_{}.png", map.view.0)),
            &photos[map.view.index()],
            map,
            stats,
        )?;
    }

    let mesh = &model.mesh;
    write_file(&dir.join("mesh.obj"), |out| {
        export::write_textured_obj(out, mesh, &model.texture, "mesh.mtl")
    })?;
    write_file(&dir.join("mesh.mtl"), |out| {
        export::write_mtl(out, "mesh_texture.png")
    })?;
    save_photo(&dir.join("mesh_texture.png"), &model.texture.atlas)?;
    write_file(&dir.join("mesh.x3d"), |out| {
        export::write_textured_x3d(out, mesh, &model.texture, "mesh_texture.png")
    })?;
    write_file(&dir.join("mesh_colours.ply"), |out| {
        export::write_ply_mesh(out, mesh, &model.texture.vertex_colours)
    })?;

    eprintln!(
        "wrote view_N.png, sparse.ply, depth_N.png, coverage_N.png, mesh.obj, mesh.mtl, mesh.x3d, mesh_texture.png and mesh_colours.ply to {}",
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

/// A photo tinted by what became of its depth samples: red where no other photo was matched
/// to it, yellow where a match was found but the depth rejected, blue where only one other
/// photo supports the depth, and untinted where two or more do.
fn write_coverage_png(
    path: &Path,
    photo: &Photo,
    map: &DepthMap,
    stats: &depth::DepthStats,
) -> Result<(), String> {
    let (width, height) = (photo.width(), photo.height());
    let mut rgba = photo.as_rgba().to_vec();
    for y in 0..height {
        let row = ((y as f32 / map.stride as f32).round() as usize).min(map.rows - 1);
        for x in 0..width {
            let column = ((x as f32 / map.stride as f32).round() as usize).min(map.columns - 1);
            let i = row * map.columns + column;
            let tint = match stats.fates[i] {
                depth::Fate::Unmapped => Some([220, 30, 30]),
                depth::Fate::Kept if stats.matched[i] >= 2 => None,
                depth::Fate::Kept => Some([40, 90, 230]),
                _ => Some([240, 200, 0]),
            };
            if let Some(tint) = tint {
                let p = (y * width + x) * 4;
                for (channel, t) in rgba[p..p + 3].iter_mut().zip(tint) {
                    *channel = ((*channel as u16 + t as u16) / 2) as u8;
                }
            }
        }
    }
    image::RgbaImage::from_raw(width as u32, height as u32, rgba)
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

    let focal = focal_of(args, &loaded, original, size)?;

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

    Ok(Input { photos, focal })
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

/// What is known about the camera for the resized photos: `--focal-px` first, then EXIF,
/// then nothing, which the library treats as a guess worth refining.
fn focal_of(
    args: &Args,
    loaded: &[Loaded],
    original: (u32, u32),
    size: (u32, u32),
) -> Result<Focal, String> {
    if let Some(focal) = args.focal_px {
        // Given for the originals, so it scales with them.
        let factor = size.0 as f64 / original.0 as f64;
        return Ok(Focal::Pixels(focal * factor));
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

    // The 35 mm equivalent does not depend on pixel count, so resizing leaves it alone.
    Ok(match focal {
        Some(focal) => Focal::Equivalent35mm(focal),
        None => Focal::Unknown,
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
