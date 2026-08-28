//! Background producers of animation frames.
//!
//! Both sources run on their own thread and stream [`ViewerMsg`] values back to
//! the window, so the viewer stays responsive while the (potentially slow)
//! correspondence mapping is still running. Frames are sent in playback order
//! and can be animated while later ones are still being produced.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use image::open;
use pixelmap::photo::Photo;
use pixelmap::pixelmap_processor::PixelMapProcessor;
use pixelmap::processing_mode::ProcessingMode;

use crate::frame::Frame;

/// A message from a frame source to the window.
pub enum ViewerMsg {
    /// Sent once, before any frame, so the window can pick a matching size.
    Init {
        width: usize,
        height: usize,
        frame_count: usize,
    },
    /// What the source is doing right now, and how far along it is (0.0..=1.0).
    Progress { label: String, fraction: f32 },
    /// The next frame of the animation.
    Frame(Frame),
    /// All frames have been produced.
    Done,
    /// The source gave up; the message is shown to the user.
    Failed(String),
}

/// What to morph, and how.
pub struct MorphRequest {
    /// The photo the animation starts from.
    pub photo1: PathBuf,
    /// The photo the animation ends at.
    pub photo2: PathBuf,
    /// How much work the correspondence mapping should do.
    pub mode: ProcessingMode,
    /// Number of frames to render, including both endpoints.
    pub frame_count: usize,
    /// Supersampling factor used when warping each frame.
    pub detail_level: usize,
    /// Downscale the photos to at most this width before processing.
    pub max_width: Option<usize>,
}

/// Reason a source stopped early.
enum Abort {
    /// Something went wrong; report it to the user.
    Failed(String),
    /// The window is gone, so nobody is listening any more.
    Disconnected,
}

impl<T> From<std::sync::mpsc::SendError<T>> for Abort {
    fn from(_: std::sync::mpsc::SendError<T>) -> Abort {
        Abort::Disconnected
    }
}

/// Runs the pixelmap algorithm on a photo pair and streams the morph frames.
pub fn spawn_morph(request: MorphRequest) -> Receiver<ViewerMsg> {
    let (tx, rx) = channel();
    thread::spawn(move || {
        if let Err(Abort::Failed(message)) = produce_morph(&request, &tx) {
            let _ = tx.send(ViewerMsg::Failed(message));
        }
    });
    rx
}

/// Streams already rendered images (for instance the output of the `pixelmap`
/// command line tool) without running the algorithm.
pub fn spawn_files(paths: Vec<PathBuf>) -> Receiver<ViewerMsg> {
    let (tx, rx) = channel();
    thread::spawn(move || {
        if let Err(Abort::Failed(message)) = produce_files(&paths, &tx) {
            let _ = tx.send(ViewerMsg::Failed(message));
        }
    });
    rx
}

fn produce_morph(request: &MorphRequest, tx: &Sender<ViewerMsg>) -> Result<(), Abort> {
    let mut photo1 = read_photo(&request.photo1)?;
    let mut photo2 = read_photo(&request.photo2)?;

    if photo1.width != photo2.width || photo1.height != photo2.height {
        return Err(Abort::Failed(format!(
            "Photos must have the same dimensions ({}x{} vs {}x{})",
            photo1.width, photo1.height, photo2.width, photo2.height
        )));
    }

    // Warping happens at the resolution of the input photos, which is wasteful
    // (and memory hungry) far above the width the algorithm works at internally.
    if let Some(max_width) = request.max_width {
        if photo1.width > max_width {
            println!("Scaling photos down to {max_width} px wide");
            photo1 = photo1.get_scaled_proportional(max_width);
            photo2 = photo2.get_scaled_proportional(max_width);
        }
    }

    tx.send(ViewerMsg::Init {
        width: photo1.width,
        height: photo1.height,
        frame_count: request.frame_count,
    })?;

    let photo1 = Rc::new(photo1);
    let photo2 = Rc::new(photo2);

    // Show the first photo right away, so there is something to look at while
    // the mapping is computed.
    tx.send(ViewerMsg::Progress {
        label: "matching features".to_string(),
        fraction: 0.0,
    })?;

    let mut processor = PixelMapProcessor::new(photo1, photo2, request.mode.photo_width());
    processor.init();

    // The schedule is the bulk of the work, and each step reports back.
    let mut progress = Ok(());
    request.mode.run_with_progress(&mut processor, |done, total| {
        if progress.is_ok() {
            progress = tx.send(ViewerMsg::Progress {
                label: format!("refining mapping (step {done}/{total})"),
                fraction: done as f32 / total as f32,
            });
        }
    });
    progress?;

    let (final_map, _reverse_map) = processor.get_result(2.0);

    for i in 0..request.frame_count {
        let alpha = if request.frame_count > 1 {
            i as f32 / (request.frame_count - 1) as f32
        } else {
            0.0
        };
        let photo = final_map.interpolate_photo(alpha, request.detail_level);
        tx.send(ViewerMsg::Progress {
            label: format!("rendering frame {}/{}", i + 1, request.frame_count),
            fraction: (i + 1) as f32 / request.frame_count as f32,
        })?;
        tx.send(ViewerMsg::Frame(Frame::from_photo(&photo)))?;
    }

    tx.send(ViewerMsg::Done)?;
    Ok(())
}

fn produce_files(paths: &[PathBuf], tx: &Sender<ViewerMsg>) -> Result<(), Abort> {
    let mut initialized = false;

    for (i, path) in paths.iter().enumerate() {
        let photo = read_photo(path)?;
        if !initialized {
            tx.send(ViewerMsg::Init {
                width: photo.width,
                height: photo.height,
                frame_count: paths.len(),
            })?;
            initialized = true;
        }
        tx.send(ViewerMsg::Progress {
            label: format!("loading frame {}/{}", i + 1, paths.len()),
            fraction: (i + 1) as f32 / paths.len() as f32,
        })?;
        tx.send(ViewerMsg::Frame(Frame::from_photo(&photo)))?;
    }

    tx.send(ViewerMsg::Done)?;
    Ok(())
}

fn read_photo(path: &Path) -> Result<Photo, Abort> {
    println!("Reading image file: {}", path.display());
    let img = open(path)
        .map_err(|e| Abort::Failed(format!("Could not load {}: {e}", path.display())))?;
    Ok(Photo {
        width: img.width() as usize,
        height: img.height() as usize,
        img_data: img.to_rgba8().into_raw(),
    })
}
