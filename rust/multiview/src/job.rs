//! Running a reconstruction on a worker thread.
//!
//! [`run`](crate::run) blocks for as long as the whole reconstruction takes, which is
//! minutes. A [`Job`] runs it on a thread instead and hands back a channel of [`Event`]s,
//! a [`Status`] to poll, and a way to stop it.
//!
//! This is the ergonomic wrapper, not the mechanism: it drives [`run`](crate::run) like any
//! other caller. That matters because `wasm32-unknown-unknown` has no threads to hand out,
//! so this module is behind the `threads` feature and the blocking form is what works
//! everywhere. A wasm caller runs the blocking form inside a web worker and gets the same
//! events through the same callback.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use pixelmap::Photo;

use crate::error::Error;
use crate::event::{Event, Flow, Status};
use crate::options::Options;
use crate::pipeline::{self, Model};

/// Asks a running reconstruction to stop.
///
/// Cloneable and shareable, so the thread that decides to stop need not be the one holding
/// the [`Job`]. Setting it is all a caller does; the run notices at its next checkpoint.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A token that has not been set.
    pub fn new() -> Self {
        Cancel::default()
    }

    /// Asks the run to stop. Returns at once; the run itself ends a moment later, with
    /// [`Error::Cancelled`].
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether a stop has been asked for.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// A reconstruction running on a worker thread.
///
/// ```no_run
/// use std::sync::Arc;
/// use pixelmap::Photo;
/// use pixelmap_multiview::{Job, Options};
///
/// # fn photos() -> Vec<Arc<Photo>> { Vec::new() }
/// let job = Job::start(photos(), Options::new());
/// for event in job.events() {
///     println!("{:.0}% {}", job.status().progress * 100.0, event.message());
/// }
/// let model = job.join()?;
/// # Ok::<(), pixelmap_multiview::Error>(())
/// ```
///
/// Dropping a job without joining it asks the run to stop and leaves it to wind down on
/// its own, rather than leaving a thread grinding through a reconstruction nobody is
/// waiting for.
pub struct Job {
    events: Receiver<Event>,
    status: Arc<Mutex<Status>>,
    cancel: Cancel,
    worker: Option<JoinHandle<Result<Model, Error>>>,
}

impl Job {
    /// Starts reconstructing `photos` on a worker thread.
    ///
    /// Nothing is validated here: the photos go straight to [`run`](crate::run) on the
    /// worker, and anything wrong with them comes back from [`Self::join`] as the same
    /// [`Error`] the blocking form would have returned.
    pub fn start(photos: Vec<Arc<Photo>>, options: Options) -> Job {
        let (sender, events) = mpsc::channel();
        let status = Arc::new(Mutex::new(Status::new()));
        let cancel = Cancel::new();

        let worker_status = Arc::clone(&status);
        let worker_cancel = cancel.clone();
        let worker = thread::spawn(move || {
            pipeline::run(&photos, &options, &mut |event| {
                if let Ok(mut status) = worker_status.lock() {
                    status.update(&event);
                }
                // A caller who has stopped reading events has not stopped the run; only
                // the cancel token does that.
                let _ = sender.send(event);
                if worker_cancel.is_cancelled() {
                    Flow::Break(())
                } else {
                    Flow::Continue(())
                }
            })
        });

        Job {
            events,
            status,
            cancel,
            worker: Some(worker),
        }
    }

    /// Every event the run reports, in order.
    ///
    /// Iterating this blocks until the next event and ends when the run does, so it is the
    /// simplest way to follow a job to completion. [`Receiver::try_recv`] drains it without
    /// blocking, for a caller with its own event loop to keep turning.
    pub fn events(&self) -> &Receiver<Event> {
        &self.events
    }

    /// Where the run has got to.
    pub fn status(&self) -> Status {
        self.status
            .lock()
            .map(|status| status.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
    }

    /// Asks the run to stop. Returns at once.
    ///
    /// The run ends at its next checkpoint and [`Self::join`] then gives
    /// [`Error::Cancelled`]. Within pairwise correspondence a checkpoint is one pixelmap
    /// schedule step; elsewhere it is prompt.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// A token that stops this run, for handing to whatever decides to.
    pub fn canceller(&self) -> Cancel {
        self.cancel.clone()
    }

    /// Waits for the run to finish and gives its result.
    ///
    /// # Errors
    /// Any [`Error`] the reconstruction produced, including [`Error::Cancelled`] when the
    /// run was stopped.
    ///
    /// # Panics
    /// If the worker thread panicked, which resumes that panic here.
    pub fn join(mut self) -> Result<Model, Error> {
        self.worker
            .take()
            .expect("a job holds its worker until it is joined")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }
}

impl std::fmt::Debug for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Job")
            .field("status", &self.status())
            .field("cancelled", &self.cancel.is_cancelled())
            .field("finished", &self.worker.as_ref().map_or(true, |w| w.is_finished()))
            .finish_non_exhaustive()
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // Joined already, so there is nothing running to stop.
        if self.worker.is_none() {
            return;
        }
        // Not joined: nobody is waiting for this any more, so ask it to stop rather than
        // leave a thread working through a reconstruction for minutes. Detached rather
        // than joined, because blocking in `drop` would surprise the caller.
        self.cancel.cancel();
    }
}
