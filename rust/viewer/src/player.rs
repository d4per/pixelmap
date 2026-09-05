//! The window: shows progress while frames are produced, then animates them.

use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use minifb::{Key, KeyRepeat, Window, WindowOptions};

use crate::frame::Frame;
use crate::source::ViewerMsg;

/// Colors of the window chrome, as `0x00RRGGBB`.
const BACKGROUND: u32 = 0x0011_1417;
const PROGRESS_TRACK: u32 = 0x0026_2b30;
const PROGRESS_FILL: u32 = 0x0064_c8ff;

/// Largest window the viewer opens by itself; bigger photos are scaled to fit.
const MAX_INITIAL_WIDTH: usize = 1280;
const MAX_INITIAL_HEIGHT: usize = 800;

/// How often the window itself is refreshed, independent of the animation rate.
const REFRESH_FPS: usize = 60;

/// How the frames should be played back.
pub struct PlaybackOptions {
    /// Animation frames per second.
    pub fps: f32,
    /// Play forwards and then backwards, instead of looping forwards.
    pub ping_pong: bool,
    /// Extra time to linger on the first and last frame, in seconds.
    pub hold: f32,
    /// Window title prefix.
    pub title: String,
}

/// Opens the window and animates the frames as they arrive from `rx`.
///
/// Returns once the user closes the window, or with an error if the frame
/// source failed before producing anything.
pub fn run(rx: Receiver<ViewerMsg>, options: PlaybackOptions) -> Result<(), String> {
    let (photo_width, photo_height) = wait_for_init(&rx)?;
    let (window_width, window_height) = initial_window_size(photo_width, photo_height);

    let mut window = Window::new(
        &options.title,
        window_width,
        window_height,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .map_err(|e| format!("Could not open a window: {e}"))?;
    window.set_target_fps(REFRESH_FPS);

    print_controls();

    let mut state = State::new(options);
    let mut canvas: Vec<u32> = Vec::new();
    let mut shown_title = String::new();

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        state.receive(&rx);
        state.handle_keys(&window);
        state.advance(Instant::now());

        let (width, height) = window.get_size();
        canvas.clear();
        canvas.resize(width * height, BACKGROUND);

        if let Some(frame) = state.current_frame() {
            frame.blit_fitted(&mut canvas, width, height);
        }
        if let Some((_, fraction)) = &state.status {
            draw_progress_bar(&mut canvas, width, height, *fraction);
        }

        let title = state.title();
        if title != shown_title {
            window.set_title(&title);
            shown_title = title;
        }

        window
            .update_with_buffer(&canvas, width, height)
            .map_err(|e| format!("Could not draw to the window: {e}"))?;

        if let Some(error) = state.error.take() {
            return Err(error);
        }
    }

    Ok(())
}

/// Blocks until the source reports the frame size, so the window can be opened
/// with the right aspect ratio.
fn wait_for_init(rx: &Receiver<ViewerMsg>) -> Result<(usize, usize), String> {
    loop {
        match rx.recv() {
            Ok(ViewerMsg::Init { width, height, .. }) => return Ok((width, height)),
            Ok(ViewerMsg::Failed(message)) => return Err(message),
            Ok(_) => continue,
            Err(_) => return Err("The frame source stopped before producing anything".to_string()),
        }
    }
}

fn initial_window_size(photo_width: usize, photo_height: usize) -> (usize, usize) {
    let scale = (MAX_INITIAL_WIDTH as f32 / photo_width.max(1) as f32)
        .min(MAX_INITIAL_HEIGHT as f32 / photo_height.max(1) as f32)
        .min(1.0);
    let width = ((photo_width as f32 * scale) as usize).max(320);
    let height = ((photo_height as f32 * scale) as usize).max(240);
    (width, height)
}

fn print_controls() {
    println!(
        "\nControls:\n  \
         space        play / pause\n  \
         left, right  step one frame\n  \
         up, down     faster / slower\n  \
         p            toggle ping-pong / loop\n  \
         r            reverse direction\n  \
         esc, q       quit\n"
    );
}

/// Everything the window loop needs to keep between redraws.
struct State {
    options: PlaybackOptions,
    frames: Vec<Frame>,
    /// Total number of frames the source promised, used for the progress text.
    expected_frames: usize,
    index: usize,
    /// `1` while playing forwards, `-1` while playing backwards.
    direction: i32,
    playing: bool,
    next_step: Instant,
    /// Current source activity and how far along it is, cleared when done.
    status: Option<(String, f32)>,
    error: Option<String>,
}

impl State {
    fn new(options: PlaybackOptions) -> State {
        State {
            options,
            frames: Vec::new(),
            expected_frames: 0,
            index: 0,
            direction: 1,
            playing: true,
            next_step: Instant::now(),
            status: None,
            error: None,
        }
    }

    /// Picks up whatever the frame source has produced since the last redraw.
    fn receive(&mut self, rx: &Receiver<ViewerMsg>) {
        loop {
            match rx.try_recv() {
                Ok(ViewerMsg::Init { frame_count, .. }) => self.expected_frames = frame_count,
                Ok(ViewerMsg::Progress { label, fraction }) => {
                    self.status = Some((label, fraction))
                }
                Ok(ViewerMsg::Frame(frame)) => self.frames.push(frame),
                Ok(ViewerMsg::Done) => self.status = None,
                Ok(ViewerMsg::Failed(message)) => {
                    // A failure after the first frame is worth showing rather
                    // than aborting: the frames already produced still play.
                    if self.frames.is_empty() {
                        self.error = Some(message);
                    } else {
                        self.status = Some((format!("failed: {message}"), 1.0));
                    }
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return,
            }
        }
    }

    fn handle_keys(&mut self, window: &Window) {
        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Space => {
                    self.playing = !self.playing;
                    self.next_step = Instant::now();
                }
                Key::Right => self.step_manually(1),
                Key::Left => self.step_manually(-1),
                Key::Up => self.options.fps = (self.options.fps + 1.0).min(120.0),
                Key::Down => self.options.fps = (self.options.fps - 1.0).max(1.0),
                Key::P => self.options.ping_pong = !self.options.ping_pong,
                Key::R => self.direction = -self.direction,
                _ => {}
            }
        }
    }

    fn step_manually(&mut self, direction: i32) {
        self.playing = false;
        self.direction = direction;
        self.step();
    }

    /// Moves to the next frame if enough time has passed.
    fn advance(&mut self, now: Instant) {
        if !self.playing || self.frames.len() < 2 {
            return;
        }
        if now < self.next_step {
            return;
        }

        let at_end = self.index == 0 || self.index + 1 >= self.frames.len();
        let mut delay = Duration::from_secs_f32(1.0 / self.options.fps.max(0.1));
        if at_end && self.options.hold > 0.0 {
            delay += Duration::from_secs_f32(self.options.hold);
        }

        self.step();

        // Catch up on a schedule we can still keep; otherwise restart from now
        // (for instance after the window was dragged or the process was busy).
        self.next_step += delay;
        if self.next_step < now {
            self.next_step = now + delay;
        }
    }

    /// Moves one frame in the current direction, turning around or wrapping
    /// around at the ends.
    fn step(&mut self) {
        let last = match self.frames.len().checked_sub(1) {
            Some(last) if last > 0 => last,
            _ => return,
        };

        if self.options.ping_pong {
            if self.direction > 0 && self.index >= last {
                self.direction = -1;
            } else if self.direction < 0 && self.index == 0 {
                self.direction = 1;
            }
        }

        self.index = if self.direction > 0 {
            if self.index >= last {
                0
            } else {
                self.index + 1
            }
        } else if self.index == 0 {
            last
        } else {
            self.index - 1
        };
    }

    fn current_frame(&self) -> Option<&Frame> {
        self.frames
            .get(self.index.min(self.frames.len().saturating_sub(1)))
    }

    fn title(&self) -> String {
        let mut title = self.options.title.clone();
        if let Some((label, fraction)) = &self.status {
            title.push_str(&format!(" — {label} ({:.0}%)", fraction * 100.0));
        }
        if !self.frames.is_empty() {
            let total = self.expected_frames.max(self.frames.len());
            title.push_str(&format!(
                " — frame {}/{} · {:.0} fps · {} · {}",
                self.index + 1,
                total,
                self.options.fps,
                if self.options.ping_pong {
                    "ping-pong"
                } else {
                    "loop"
                },
                if self.playing { "playing" } else { "paused" },
            ));
        }
        title
    }
}

/// Draws a progress bar along the bottom of the window.
fn draw_progress_bar(canvas: &mut [u32], width: usize, height: usize, fraction: f32) {
    let margin = 16.min(width / 8);
    let bar_height = 6.min(height / 4).max(1);
    let bottom = height.saturating_sub(margin.max(bar_height + 2));
    if width <= margin * 2 || bottom < bar_height {
        return;
    }

    let bar_width = width - margin * 2;
    let filled = (bar_width as f32 * fraction.clamp(0.0, 1.0)) as usize;

    for y in (bottom - bar_height)..bottom {
        let row = y * width;
        for x in 0..bar_width {
            canvas[row + margin + x] = if x < filled {
                PROGRESS_FILL
            } else {
                PROGRESS_TRACK
            };
        }
    }
}
