//! A window that animates the pixelmap interpolation.
//!
//! Either runs the algorithm on two photos and animates the morph as the frames
//! are rendered:
//!
//! ```text
//! pixelmap-viewer photo1.jpg photo2.jpg --processing-mode medium
//! ```
//!
//! or replays images that were already written to disk by the `pixelmap`
//! command line tool:
//!
//! ```text
//! pixelmap-viewer --frames interpolation_*.png
//! ```

mod frame;
mod player;
mod source;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use pixelmap::processing_mode::ProcessingMode;

use player::PlaybackOptions;
use source::MorphRequest;

/// Command line arguments structure.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Animates pixelmap interpolations in a window.",
    after_help = "Controls: space = play/pause, left/right = step, up/down = speed, \
                  p = ping-pong, r = reverse, esc/q = quit."
)]
struct Args {
    /// First photo filename (omit when using --frames)
    #[arg()]
    photo1: Option<PathBuf>,

    /// Second photo filename (omit when using --frames)
    #[arg()]
    photo2: Option<PathBuf>,

    /// Animate existing images instead of running the algorithm,
    /// e.g. --frames interpolation_*.png
    #[arg(long, num_args = 1.., value_name = "FILE")]
    frames: Vec<PathBuf>,

    /// Processing mode: low, medium, or high.
    /// - low: Fast, but may be less accurate.
    /// - medium: Slower, but more accurate.
    /// - high: Slowest, but likely the best result.
    #[arg(long, default_value = "low")]
    processing_mode: ProcessingMode,

    /// How many frames to render, including both end photos
    #[arg(long, default_value_t = 12)]
    num_frames: usize,

    /// Supersampling used when warping each frame (higher is slower, fewer holes)
    #[arg(long, default_value_t = 4)]
    detail_level: usize,

    /// Animation speed in frames per second
    #[arg(long, default_value_t = 12.0)]
    fps: f32,

    /// Loop forwards instead of playing forwards and backwards
    #[arg(long)]
    no_ping_pong: bool,

    /// Seconds to linger on the first and last frame
    #[arg(long, default_value_t = 0.4)]
    hold: f32,

    /// Scale the photos down to at most this width before processing (0 keeps them as they are)
    #[arg(long, default_value_t = 1200)]
    max_width: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();

    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("Err: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    let (receiver, title) = if !args.frames.is_empty() {
        let mut paths = args.frames.clone();
        sort_numerically(&mut paths);
        let title = format!("pixelmap — {} frames", paths.len());
        (source::spawn_files(paths), title)
    } else {
        let (photo1, photo2) = match (args.photo1.clone(), args.photo2.clone()) {
            (Some(photo1), Some(photo2)) => (photo1, photo2),
            _ => {
                return Err(
                    "Give two photos to morph, or --frames with images to replay. \
                     See --help."
                        .to_string(),
                )
            }
        };

        if args.num_frames < 2 {
            return Err("--num-frames must be at least 2".to_string());
        }
        if args.detail_level < 1 {
            return Err("--detail-level must be at least 1".to_string());
        }

        let title = format!(
            "pixelmap — {} → {} ({})",
            file_label(&photo1),
            file_label(&photo2),
            args.processing_mode
        );
        let request = MorphRequest {
            photo1,
            photo2,
            mode: args.processing_mode,
            frame_count: args.num_frames,
            detail_level: args.detail_level,
            max_width: (args.max_width > 0).then_some(args.max_width),
        };
        (source::spawn_morph(request), title)
    };

    player::run(
        receiver,
        PlaybackOptions {
            fps: args.fps.max(0.1),
            ping_pong: !args.no_ping_pong,
            hold: args.hold.max(0.0),
            title,
        },
    )
}

/// Sorts frame files by the last number in their name, so `frame_2.png` comes
/// before `frame_10.png` even though the shell hands them over the other way.
fn sort_numerically(paths: &mut [PathBuf]) {
    paths.sort_by(|a, b| {
        trailing_number(a)
            .cmp(&trailing_number(b))
            .then_with(|| a.cmp(b))
    });
}

fn trailing_number(path: &PathBuf) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    let digits: String = stem
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.chars().rev().collect::<String>().parse().ok()
}

fn file_label(path: &PathBuf) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_ordered_by_their_number_not_their_name() {
        let mut paths: Vec<PathBuf> = ["interp_10.png", "interp_2.png", "interp_1.png"]
            .iter()
            .map(PathBuf::from)
            .collect();
        sort_numerically(&mut paths);

        let names: Vec<_> = paths.iter().map(|p| p.to_string_lossy()).collect();
        assert_eq!(names, ["interp_1.png", "interp_2.png", "interp_10.png"]);
    }

    #[test]
    fn files_without_a_number_keep_a_stable_order() {
        let mut paths: Vec<PathBuf> = ["b.png", "a.png", "frame_1.png"]
            .iter()
            .map(PathBuf::from)
            .collect();
        sort_numerically(&mut paths);

        let names: Vec<_> = paths.iter().map(|p| p.to_string_lossy()).collect();
        assert_eq!(names, ["a.png", "b.png", "frame_1.png"]);
    }
}
