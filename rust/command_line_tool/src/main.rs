use clap::Parser;
use pixelmap::{Correspondence, DensePhotoMap, ProcessingMode, DEFAULT_SEED};
use std::sync::Arc;

use std::fs::File;
use std::io::Write;
use image::open;
use pixelmap::Photo;

/// Command line arguments structure.
#[derive(Parser, Debug)]
#[command(author, version, about = "CLI for pixelmap-based photo matching and interpolation.")]
struct Args {
    /// First photo filename
    #[arg()]
    photo1: String,

    /// Second photo filename
    #[arg()]
    photo2: String,

    /// Optionally output DensePhotoMap as a text file
    #[arg(long)]
    output_dense_map: Option<String>,

    /// Processing mode: low, medium, or high.
    /// - low: Fast, but may be less accurate.
    /// - medium: Slower, but more accurate.
    /// - high: Slowest, but likely the best result.
    #[arg(long, default_value = "low")]
    processing_mode: ProcessingMode,

    /// How many interpolations to produce (default is 10)
    #[arg(long, default_value_t = 10)]
    num_interpolations: usize,

    /// Base filename for interpolated outputs (e.g. "interp" -> "interp_1.jpg", ...)
    #[arg(long, default_value = "interpolation")]
    base_filename: String,

    /// Benchmark the feature matcher's nearest-neighbour backends against each other
    /// and exit, without running the mapping pipeline.
    #[cfg(feature = "bench")]
    #[arg(long, hide = true)]
    bench_matchers: bool,

    /// Working width for --bench-matchers. Defaults to the processing mode's width.
    #[cfg(feature = "bench")]
    #[arg(long, hide = true)]
    bench_width: Option<usize>,
}

fn main() {
    let args = Args::parse();

    // Load the two input photos (adjust these calls to match your actual I/O).
    let photo1 = Arc::new(read_photo(&args.photo1));
    let photo2 = Arc::new(read_photo(&args.photo2));

    #[cfg(feature = "bench")]
    if args.bench_matchers {
        let width = args.bench_width.unwrap_or_else(|| args.processing_mode.photo_width());
        let reports = pixelmap::matcher_bench::run(&photo1, &photo2, width);
        pixelmap::matcher_bench::print_reports(&reports);
        return;
    }

    // Run the pipeline, reporting each schedule step as it completes. The library itself
    // prints nothing; progress is something the application decides to show.
    let mapping = Correspondence::builder()
        .quality(args.processing_mode)
        .seed(seed())
        .run_with_progress(photo1, photo2, |p| {
            eprintln!("step {}/{} ({:.0}%)", p.step, p.total, p.fraction() * 100.0);
        });

    let mapping = match mapping {
        Ok(mapping) => mapping,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    println!("matched area: {:.4}", mapping.coverage());

    let final_map: DensePhotoMap = mapping.into_parts().0;

    // If the user requested JSON output for the DensePhotoMap, write it out.
    if let Some(dense_map_path) = args.output_dense_map {
        let data_text = to_dense_map(&final_map);
        let mut file = File::create(&dense_map_path)
            .expect("Could not create data output file");
        file.write_all(data_text.as_bytes())
            .expect("Failed to write data to file");
        println!("DensePhotoMap written to {dense_map_path}");
    }

    // Produce interpolated photos if requested.
    if args.num_interpolations > 0 {
        let base_filename = &args.base_filename;

        // Generate equally spaced interpolation increments from 0.0 to 1.0.
        for i in 1..=args.num_interpolations {
            let alpha = i as f32 / (args.num_interpolations + 1) as f32;
            // detail_level is an example parameter you can adjust as needed:
            let detail_level = 4;
            let interpolated_photo = final_map.interpolate_photo(alpha, detail_level);

            // Construct an output filename like "interp_1.png", etc.
            let filename = format!("{}_{}.png", base_filename, i);
            save_photo(interpolated_photo, &filename);
        }
    }

    println!("Done.");
}

/// The seed to run the solver with.
///
/// The library defaults to [`pixelmap::pixelmap_processor::DEFAULT_SEED`] and takes the
/// seed as an argument; reading the environment is the *application's* job, so this lives
/// here rather than in the library, where an ambient process-wide setting would be
/// invisible to a caller and impossible for two callers to disagree on.
fn seed() -> u64 {
    match std::env::var("PIXELMAP_SEED") {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(value) => value,
            Err(_) => {
                eprintln!("Ignoring PIXELMAP_SEED={raw:?}: not an unsigned integer.");
                DEFAULT_SEED
            }
        },
        Err(_) => DEFAULT_SEED,
    }
}

fn to_dense_map(dense_photo_map: &DensePhotoMap) -> String {
    let mut out = String::with_capacity(1000000);
    let (width, height) = dense_photo_map.dimensions();
    for y in 0..height {
        for x in 0..width {
            let (x2, y2) = dense_photo_map.map_photo_pixel(x as f32, y as f32);
            if x2.is_nan() {
                continue;
            }
            out.push_str(&format!("{} {} {:.3} {:.3}\n", x, y, x2, y2));
        }
    }
    out
}

pub fn save_photo(photo: Photo, filename: &str) {
    println!("Writing image {filename}");
    let (width, height) = (photo.width() as u32, photo.height() as u32);
    let img = image::RgbaImage::from_raw(width, height, photo.into_rgba())
        .expect("the buffer came from a Photo of exactly these dimensions");
    img.save(filename).unwrap();
}

pub fn read_photo(filename: &str) -> Photo {
    println!("Reading image file: {filename}");
    // `Photo: From<DynamicImage>` comes from the library's `image` feature.
    Photo::from(open(filename).expect("Could not load image"))
}