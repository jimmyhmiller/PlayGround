//! The data layer — the *only* thing the game reads from.
//!
//! The game never talks to the filesystem, an API, or Claude directly. It asks a
//! [`WorldSource`] for a [`WorldSnapshot`], and renders whatever it finds. To feed
//! the game from somewhere new (a socket, an HTTP endpoint, a different agent
//! runtime, a database) you implement ONE trait method, [`WorldSource::poll`], and
//! hand it to [`crate::data::SourceRunner`]. Nothing in the game/render layers
//! changes.
//!
//! ```ignore
//! struct MySource { /* ... */ }
//! impl WorldSource for MySource {
//!     fn name(&self) -> &str { "my-source" }
//!     fn poll(&mut self) -> WorldSnapshot { /* build cities/sessions */ }
//! }
//! let runner = SourceRunner::spawn(Box::new(MySource::new()), 3.0);
//! ```

pub mod claude;
pub mod mock;

use crate::util::now_unix;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::Duration;

/// A point-in-time view of every "civilization" we know about.
#[derive(Debug, Clone, Default)]
pub struct WorldSnapshot {
    /// One entry per project. Order is up to the source; the game keys off `id`.
    pub cities: Vec<CityInfo>,
    /// When this snapshot was captured (fractional unix seconds).
    pub captured_at: f64,
}

/// One project == one city.
#[derive(Debug, Clone)]
pub struct CityInfo {
    /// Stable key used to match a city across polls (e.g. the project directory).
    pub id: String,
    /// Human-friendly display name (e.g. `datalog-db`).
    pub name: String,
    /// Filesystem path / cwd, if known.
    pub path: Option<String>,
    /// The sessions that have happened in this project.
    pub sessions: Vec<SessionInfo>,
}

/// One agent session == one building + the villagers it employs.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Stable key (session id / uuid).
    pub id: String,
    /// AI-generated title for the session, if any.
    pub title: Option<String>,
    /// Model that did the work, e.g. `claude-opus-4-8`.
    pub model: Option<String>,
    pub user_messages: u32,
    pub assistant_messages: u32,
    /// Number of tool calls the assistant made (work performed).
    pub tool_uses: u32,
    /// First activity (unix secs), if known.
    pub first_active: Option<f64>,
    /// Last activity (unix secs) — usually the session file's mtime.
    pub last_active: Option<f64>,
    pub git_branch: Option<String>,
}

impl SessionInfo {
    pub fn total_messages(&self) -> u32 {
        self.user_messages + self.assistant_messages
    }
    /// Seconds since this session was last active (relative to `now`).
    pub fn idle_secs(&self, now: f64) -> f64 {
        self.last_active.map(|t| (now - t).max(0.0)).unwrap_or(f64::INFINITY)
    }
    /// "Live" == touched within the last `window` seconds.
    pub fn is_live(&self, now: f64, window: f64) -> bool {
        self.idle_secs(now) <= window
    }
}

impl CityInfo {
    pub fn total_messages(&self) -> u32 {
        self.sessions.iter().map(|s| s.total_messages()).sum()
    }
    pub fn total_tool_uses(&self) -> u32 {
        self.sessions.iter().map(|s| s.tool_uses).sum()
    }
    pub fn live_sessions(&self, now: f64, window: f64) -> usize {
        self.sessions.iter().filter(|s| s.is_live(now, window)).count()
    }
    /// Most-recent activity across all sessions (unix secs).
    pub fn last_active(&self) -> Option<f64> {
        self.sessions.iter().filter_map(|s| s.last_active).fold(None, |acc, t| {
            Some(acc.map_or(t, |a: f64| a.max(t)))
        })
    }
}

/// Implement this to feed the game from anywhere.
pub trait WorldSource: Send {
    /// Short label shown in the HUD (e.g. `claude-projects`, `mock`).
    fn name(&self) -> &str;
    /// Produce the latest snapshot. Called on a timer from a background thread,
    /// so it may block on I/O; it should set `captured_at` to "now".
    fn poll(&mut self) -> WorldSnapshot;
}

/// Runs a [`WorldSource`] on its own thread and delivers fresh snapshots over a
/// channel, so a slow `poll()` (reading many MB of logs) never stalls the 60 fps
/// render loop. The game just calls [`SourceRunner::latest`] each frame.
pub struct SourceRunner {
    rx: Receiver<WorldSnapshot>,
    last: WorldSnapshot,
    source_name: String,
    /// True until the very first snapshot arrives.
    loading: bool,
}

impl SourceRunner {
    /// Spawn the polling thread. `interval_secs` is the gap between polls.
    pub fn spawn(mut source: Box<dyn WorldSource>, interval_secs: f64) -> SourceRunner {
        let source_name = source.name().to_string();
        let (tx, rx) = mpsc::channel();
        thread::Builder::new()
            .name("world-source".into())
            .spawn(move || loop {
                let snap = source.poll();
                if tx.send(snap).is_err() {
                    break; // game dropped the receiver; stop polling.
                }
                thread::sleep(Duration::from_secs_f64(interval_secs.max(0.25)));
            })
            .expect("spawn world-source thread");
        SourceRunner {
            rx,
            last: WorldSnapshot { captured_at: now_unix(), ..Default::default() },
            source_name,
            loading: true,
        }
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    pub fn is_loading(&self) -> bool {
        self.loading
    }

    /// Drain the channel and return the freshest snapshot. Returns `true` if a new
    /// snapshot arrived this frame (so the caller can re-sync game entities).
    pub fn poll_latest(&mut self) -> bool {
        let mut updated = false;
        loop {
            match self.rx.try_recv() {
                Ok(snap) => {
                    self.last = snap;
                    self.loading = false;
                    updated = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        updated
    }

    pub fn latest(&self) -> &WorldSnapshot {
        &self.last
    }
}
