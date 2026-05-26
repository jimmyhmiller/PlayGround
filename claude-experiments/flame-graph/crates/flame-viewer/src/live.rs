//! Live profiling session: reader thread pumps a `LiveAggregator`; builder
//! thread periodically snapshots it into an `Arc<Profile>`; main thread polls
//! for new snapshots between frames.
//!
//! The reader and builder are intentionally decoupled — sample ingestion
//! shouldn't block waiting for a previous snapshot build to finish, and we
//! don't want a snapshot per sample. The builder coalesces by checking the
//! aggregator's monotonic `version` counter every `BUILD_INTERVAL`.

use std::io::BufReader;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use flame_core::Profile;
use flame_live::{read_binary_stream, read_ndjson_stream, LiveAggregator, SymbolStore};

const CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(100);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// How often the builder thread checks for fresh events and rebuilds the
/// snapshot. The renderer can apply each new snapshot in ~50µs, the build
/// itself is sub-millisecond for typical sample counts, so 16ms (≈60Hz)
/// gives a continuously-updating flame graph at perceived real-time.
/// Tradeoff at very high sample counts (~1M+): the build cost will grow,
/// and you may want to bump this back up to 33ms / 50ms / 100ms.
const BUILD_INTERVAL: Duration = Duration::from_millis(16);

#[derive(Clone, Copy, Debug)]
pub enum LiveFormat {
    Binary,
    Ndjson,
}

pub struct LiveSession {
    pub label: String,
    /// Wall-clock moment the session was created. Used to compute "live for
    /// X.Y s" in the status bar.
    pub started_at: Instant,
    /// Held so the reader's Arc-clone keeps the aggregator alive for the
    /// life of the session.
    aggregator: Arc<Mutex<LiveAggregator>>,
    snapshot: Arc<Mutex<SnapshotSlot>>,
    /// We never `join` these — the reader exits on EOF / disconnect, the
    /// builder on `stop`. They're held to keep the threads alive for the
    /// session's lifetime.
    _reader: JoinHandle<()>,
    _builder: JoinHandle<()>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    /// In `--record` mode: the spawned samply process. Shared with the
    /// child-watcher thread (which calls `try_wait` to reap the child as
    /// soon as it exits). Killed in `Drop` so closing the viewer also stops
    /// sampling.
    child: Option<Arc<Mutex<Child>>>,
    /// Tmp socket path to remove on drop in record mode.
    tmp_socket: Option<PathBuf>,
    /// Path samply was told to write the final profile to. In `--record`
    /// mode samply only flushes this when the target process exits cleanly.
    /// Used by the viewer to auto-promote to a fully-symbolicated profile
    /// once recording finishes.
    output_profile: Option<PathBuf>,
    /// Set to `true` by the reader thread on stream EOF or disconnect. The
    /// viewer polls it to know when to look for a finished `output_profile`.
    stream_done: Arc<std::sync::atomic::AtomicBool>,
    /// Set to `true` by a watcher thread once the spawned samply child has
    /// exited. The auto-promote path waits for this before reading the
    /// output file, because samply writes the JSON only as it exits —
    /// loading mid-write yields a truncated parse failure.
    child_exited: Arc<std::sync::atomic::AtomicBool>,
}

struct SnapshotSlot {
    /// Latest fully-built profile. Replaced wholesale by the builder thread.
    profile: Option<Arc<Profile>>,
    /// Aggregator version baked into `profile`. Used by the builder thread to
    /// skip rebuilds when nothing has changed.
    built_version: u64,
    /// Bumped each time `profile` is replaced. Reader-side uses this to know
    /// when there's something new to apply.
    seq: u64,
    /// Wall-clock instant of the freshest event included in `profile`.
    /// The main thread subtracts this from `Instant::now()` after the swap
    /// to log end-to-end ingest→display latency.
    freshest_event_at: Option<Instant>,
}

impl LiveSession {
    /// Connect to a unix socket samply opened with `--live-socket`. Retries
    /// for up to `CONNECT_TIMEOUT` so the viewer can be launched before
    /// samply.
    pub fn connect_socket(path: PathBuf, format: LiveFormat) -> std::io::Result<Self> {
        let label = format!("socket {}", path.display());
        let stream = connect_with_retry(&path)?;
        log::info!("flame-live: connected to {}", path.display());
        Self::spawn(label, Box::new(stream), format)
    }

    /// Open a file or fifo written by samply with `--live-file`. For a fifo
    /// the open blocks until samply has its end open.
    pub fn open_file(path: PathBuf, format: LiveFormat) -> std::io::Result<Self> {
        let label = format!("file {}", path.display());
        let file = std::fs::File::open(&path)?;
        log::info!("flame-live: opened {}", path.display());
        Self::spawn(label, Box::new(file), format)
    }

    fn spawn(
        label: String,
        source: Box<dyn std::io::Read + Send>,
        format: LiveFormat,
    ) -> std::io::Result<Self> {
        // Spin up the symbol store + loader thread alongside the aggregator
        // so frames get resolved to real names as their libs' symbol tables
        // finish parsing on the background thread.
        let symbols = SymbolStore::new();
        let aggregator = Arc::new(Mutex::new(LiveAggregator::with_symbols(symbols)));
        let snapshot = Arc::new(Mutex::new(SnapshotSlot {
            profile: None,
            built_version: u64::MAX, // sentinel: ensure first build fires
            seq: 0,
            freshest_event_at: None,
        }));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stream_done = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let reader = {
            let agg = aggregator.clone();
            let stop = stop.clone();
            let done = stream_done.clone();
            thread::Builder::new()
                .name("flame-live-reader".into())
                .spawn(move || {
                    let r = BufReader::new(source);
                    let result = match format {
                        LiveFormat::Binary => read_binary_stream(r, |ev| {
                            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                                return false;
                            }
                            if let Ok(mut a) = agg.lock() {
                                a.apply(ev);
                            }
                            true
                        }),
                        LiveFormat::Ndjson => read_ndjson_stream(r, |ev| {
                            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                                return false;
                            }
                            if let Ok(mut a) = agg.lock() {
                                a.apply(ev);
                            }
                            true
                        }),
                    };
                    match result {
                        Ok(()) => log::info!("flame-live: stream ended"),
                        Err(e) => log::error!("flame-live: reader error: {e}"),
                    }
                    done.store(true, std::sync::atomic::Ordering::Release);
                })?
        };

        let builder = {
            let agg = aggregator.clone();
            let slot = snapshot.clone();
            let stop = stop.clone();
            thread::Builder::new()
                .name("flame-live-builder".into())
                .spawn(move || loop {
                    if stop.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    // Snapshot the aggregator (clones data inside the lock,
                    // releases it before the heavy build work).
                    let (version, profile, freshest_event_at, ingest_to_build_lag) = {
                        let a = match agg.lock() {
                            Ok(a) => a,
                            Err(_) => return,
                        };
                        // combined_version includes background symbol loads
                        // so the snapshot rebuilds when new function names
                        // land, not only when new samples arrive.
                        let want_version = a.combined_version();
                        let last_built = slot.lock().map(|s| s.built_version).unwrap_or(0);
                        if want_version == last_built {
                            drop(a);
                            thread::sleep(BUILD_INTERVAL);
                            continue;
                        }
                        let freshest = a.last_apply_instant;
                        let ingest_lag = freshest.map(|t| t.elapsed());
                        let t0 = Instant::now();
                        let p = a.snapshot();
                        log::debug!(
                            "flame-live: snapshot built in {:?} ({} samples, {} threads); \
                             ingest→build lag = {:?}",
                            t0.elapsed(),
                            a.sample_count(),
                            a.thread_count(),
                            ingest_lag,
                        );
                        (want_version, p, freshest, ingest_lag)
                    };
                    let _ = ingest_to_build_lag; // logged above, not propagated
                    let (threads, samples) = {
                        let a = agg.lock().ok();
                        a.map(|a| (a.thread_count(), a.sample_count())).unwrap_or((0, 0))
                    };
                    let seq = if let Ok(mut s) = slot.lock() {
                        s.profile = Some(Arc::new(profile));
                        s.built_version = version;
                        s.seq = s.seq.wrapping_add(1);
                        s.freshest_event_at = freshest_event_at;
                        s.seq
                    } else {
                        0
                    };
                    log::debug!(
                        "flame-live: snapshot seq={seq} version={version} \
                         threads={threads} samples={samples}",
                    );
                    thread::sleep(BUILD_INTERVAL);
                })?
        };

        Ok(Self {
            label,
            started_at: Instant::now(),
            aggregator,
            snapshot,
            _reader: reader,
            _builder: builder,
            stop,
            child: None,
            tmp_socket: None,
            output_profile: None,
            stream_done,
            child_exited: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Spawn `samply record --live-socket <tmp> --no-open --save-only -o <tmp.json>
    /// -- <cmd...>`, wait for the socket to appear, then connect. The samply
    /// child is killed on drop so the viewer's window-close also stops the
    /// recorder.
    ///
    /// `samply_bin` is the path to the samply executable (caller resolves
    /// PATH lookup, or accepts `--samply-bin`).
    pub fn spawn_samply(
        samply_bin: PathBuf,
        target_cmd: Vec<String>,
        output_profile: PathBuf,
        samply_port: u16,
    ) -> std::io::Result<Self> {
        if target_cmd.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "no target command supplied to --record",
            ));
        }
        let tmp_socket = std::env::temp_dir().join(format!(
            "samply-live-{}.sock",
            std::process::id()
        ));
        // Best-effort cleanup of any stale socket from a previous run.
        let _ = std::fs::remove_file(&tmp_socket);

        let mut cmd = Command::new(&samply_bin);
        cmd.arg("record")
            .arg("--live-socket")
            .arg(&tmp_socket)
            .arg("--no-open")
            .arg("--save-only")
            .arg("-P")
            .arg(samply_port.to_string())
            .arg("-o")
            .arg(&output_profile)
            .arg("--")
            .args(&target_cmd);
        // Forward samply's stdout/stderr so the user sees its messages.
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        log::info!(
            "flame-live: spawning {} record --live-socket {} -o {} -- {}",
            samply_bin.display(),
            tmp_socket.display(),
            output_profile.display(),
            target_cmd.join(" "),
        );
        let child = cmd.spawn()?;

        // connect_with_retry will spin until samply binds the socket.
        let stream = connect_with_retry(&tmp_socket)?;
        log::info!("flame-live: connected to spawned samply at {}", tmp_socket.display());

        let label = format!("samply record -- {}", target_cmd.join(" "));
        let mut session = Self::spawn(label, Box::new(stream), LiveFormat::Binary)?;
        let child_arc = Arc::new(Mutex::new(child));
        // Watcher thread: poll `try_wait` so we both reap the child (avoid
        // a zombie) AND notice the exit promptly. The mutex is contended
        // only on shutdown, when Drop also wants to access the child.
        let exited = session.child_exited.clone();
        let watcher_child = child_arc.clone();
        thread::Builder::new()
            .name("flame-live-child-watch".into())
            .spawn(move || loop {
                {
                    let mut guard = match watcher_child.lock() {
                        Ok(g) => g,
                        Err(_) => return,
                    };
                    match guard.try_wait() {
                        Ok(Some(_)) => {
                            exited.store(true, std::sync::atomic::Ordering::Release);
                            return;
                        }
                        Ok(None) => {}
                        Err(_) => return,
                    }
                }
                thread::sleep(Duration::from_millis(50));
            })?;
        session.child = Some(child_arc);
        session.tmp_socket = Some(tmp_socket);
        session.output_profile = Some(output_profile);
        Ok(session)
    }

    /// True once the spawned samply child has exited (record mode only).
    /// In connect/file modes this is always `false`; the auto-promote path
    /// in the viewer checks `stream_done()` instead.
    pub fn child_exited(&self) -> bool {
        self.child_exited
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// True once the reader thread has hit EOF or an error. Used by the
    /// viewer to decide it's safe to load a final, fully-symbolicated
    /// profile from disk.
    pub fn stream_done(&self) -> bool {
        self.stream_done.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The path samply was told to write its final profile to (record
    /// mode only). Only meaningful once `stream_done()` is true AND the
    /// target process exited cleanly — samply skips the final write if it
    /// gets SIGKILL'd before finishing.
    pub fn output_profile(&self) -> Option<&Path> {
        self.output_profile.as_deref()
    }

    /// Return `(seq, profile, freshest_event_at)` of the most-recent
    /// snapshot, or `None` if no snapshot has been built yet. Caller
    /// compares `seq` against the last one it applied. `freshest_event_at`
    /// is the wall-clock instant of the newest event included in the
    /// profile; subtract it from `Instant::now()` after applying to log
    /// total ingest→display latency.
    pub fn latest_snapshot(&self) -> Option<(u64, Arc<Profile>, Option<Instant>)> {
        let s = self.snapshot.lock().ok()?;
        s.profile
            .as_ref()
            .map(|p| (s.seq, p.clone(), s.freshest_event_at))
    }

    pub fn stats(&self) -> Option<(usize, usize)> {
        let a = self.aggregator.lock().ok()?;
        Some((a.thread_count(), a.sample_count()))
    }

    /// Wall-clock seconds since this session started. The status bar uses
    /// this together with `stats()` to compute an instantaneous sample rate.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(child_arc) = self.child.take() {
            // samply ignores SIGINT and SIGTERM while its target is alive
            // (it explicitly suppresses them so Ctrl+C reaches the target
            // instead — see samply/src/mac/profiler.rs). To get a saved
            // profile out, we must kill the *target* (samply's child) and
            // let samply observe the exit, finalize, and flush profile.json.
            let samply_pid = child_arc.lock().map(|g| g.id()).unwrap_or(0);
            if samply_pid != 0 {
                log::info!(
                    "flame-live: terminating samply target (pkill -TERM -P {samply_pid}), \
                     waiting for samply to flush profile..."
                );
                let _ = Command::new("pkill")
                    .arg("-TERM")
                    .arg("-P")
                    .arg(samply_pid.to_string())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            // Wait up to 5s for samply to finalize. The watcher thread is
            // calling try_wait in a tight loop, so we mostly busy-wait on
            // the exited flag here rather than fighting it for the lock.
            let deadline = Instant::now() + Duration::from_secs(5);
            while !self.child_exited.load(std::sync::atomic::Ordering::Acquire) {
                if Instant::now() >= deadline {
                    log::warn!("flame-live: samply did not exit after 5s, sending SIGKILL");
                    if let Ok(mut guard) = child_arc.lock() {
                        let _ = guard.kill();
                        let _ = guard.wait();
                    }
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
        if let Some(path) = self.tmp_socket.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn connect_with_retry(path: &Path) -> std::io::Result<UnixStream> {
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    let mut last_err: Option<std::io::Error> = None;
    while Instant::now() < deadline {
        match UnixStream::connect(path) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last_err = Some(e);
                thread::sleep(CONNECT_RETRY_INTERVAL);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::TimedOut, "live socket connect timeout")
    }))
}
