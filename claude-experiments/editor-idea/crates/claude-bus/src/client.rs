//! In-process subscriber for the bus.
//!
//! Spawns a worker thread that owns the unix socket, decodes
//! `BusFrame::Event` frames, and ships them through an `mpsc::Receiver`
//! the caller polls at its own cadence. The worker reconnects forever
//! whenever the socket drops — so a launchd-driven bus restart looks
//! to subscribers like a small hiccup, not a fatal error.
//!
//! Resume semantics: the worker tracks the highest seq it has
//! delivered, and on reconnect asks the bus to replay `since_seq =
//! last + 1`. If the bus's ring no longer holds that seq the worker
//! receives a `Lagged` marker and emits a `BusEvent::Gap` so the
//! caller can decide whether to fall back to JSONL.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;

use crate::proto::{decode, encode, BusFrame, ClientFrame, Role};

/// One delivered event, mirroring `BusFrame::Event`.
#[derive(Debug, Clone)]
pub struct BusEvent {
    pub seq: u64,
    pub kind: String,
    pub ts: u64,
    pub terminal_session_id: String,
    pub claude_pid: u32,
    /// Payload as a JSON string. Caller parses on demand — most
    /// consumers only care about a few specific kinds.
    pub payload_json: String,
}

/// Out-of-band signals the subscriber may emit alongside events. Kept
/// in a single enum so callers have one channel to drain.
#[derive(Debug, Clone)]
pub enum BusItem {
    Event(BusEvent),
    /// Worker had to reconnect. `last_delivered_seq` is the last seq
    /// the caller saw before the gap; `replay_from` is the oldest seq
    /// the bus still holds. Caller can read the JSONL between them if
    /// they need that range.
    Gap {
        last_delivered_seq: Option<u64>,
        replay_from: u64,
    },
    /// Worker lost the socket and is retrying. Mostly informational —
    /// emit so a UI can show "bus offline".
    Disconnected,
    /// Worker successfully reattached after a `Disconnected`.
    Reconnected,
}

/// Handle to a running subscriber. Drop it to ask the worker to stop;
/// the thread joins on the next poll cycle.
pub struct Subscriber {
    rx: Receiver<BusItem>,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl Subscriber {
    /// Spawn the worker. `socket` is the bus path (typically
    /// `claude_bus::socket_path().unwrap()`). `since_seq = None` means
    /// "live only"; `Some(n)` requests replay from `n` if the bus
    /// still has it.
    pub fn spawn(socket: PathBuf, since_seq: Option<u64>) -> Self {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_w = stop.clone();
        let join = thread::Builder::new()
            .name("claude-bus-subscriber".into())
            .spawn(move || worker_loop(socket, since_seq, tx, stop_w))
            .expect("spawn subscriber thread");
        Self {
            rx,
            stop,
            join: Some(join),
        }
    }

    /// Non-blocking. Returns `None` when no item is queued right now.
    pub fn try_recv(&self) -> Option<BusItem> {
        self.rx.try_recv().ok()
    }

    /// Drain everything currently queued without blocking. Useful for
    /// Bevy systems that want to flush per-frame.
    pub fn drain(&self) -> Vec<BusItem> {
        let mut out = Vec::new();
        while let Ok(item) = self.rx.try_recv() {
            out.push(item);
        }
        out
    }

    /// Blocking variant with a deadline.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<BusItem> {
        self.rx.recv_timeout(timeout).ok()
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(j) = self.join.take() {
            // We don't wait — the worker may be parked in a blocking
            // socket read. Detaching is fine since the OS will reap
            // it once the bus drops the conn on our process exit.
            std::mem::drop(j);
        }
    }
}

/// Forever-running worker. Tries to keep a connection open; on every
/// disconnect, waits a bit and retries with a resumed `since_seq`.
fn worker_loop(
    socket: PathBuf,
    initial_since: Option<u64>,
    tx: Sender<BusItem>,
    stop: Arc<AtomicBool>,
) {
    let mut last_delivered: Option<u64> = None;
    let mut requested_since = initial_since;
    let mut backoff = Duration::from_millis(100);
    let mut announced_disconnect = false;

    while !stop.load(Ordering::Relaxed) {
        let resume_seq = last_delivered.map(|s| s + 1).or(requested_since);
        match run_session(&socket, resume_seq, &tx, &stop, &mut last_delivered) {
            Ok(()) => {
                // Session ended cleanly (stop flag).
                return;
            }
            Err(e) => {
                // Any error → drop and retry. Don't spam Disconnected
                // on the first attempt before we ever connected.
                if !announced_disconnect && last_delivered.is_some() {
                    let _ = tx.send(BusItem::Disconnected);
                    announced_disconnect = true;
                }
                eprintln!("[claude-bus-client] session ended: {} — retrying in {:?}", e, backoff);
                // Cap backoff at a few seconds so a long bus outage
                // still gets noticed quickly when it ends.
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(5));
                requested_since = None; // initial since only applies on first attempt
            }
        }
        if announced_disconnect && !stop.load(Ordering::Relaxed) {
            // We're about to try again; announce reconnect on success.
        }
    }
}

/// One connect → Hello → recv loop. Returns Ok only when `stop` flips
/// to true; any I/O error returns Err so the outer loop retries.
fn run_session(
    socket: &std::path::Path,
    since_seq: Option<u64>,
    tx: &Sender<BusItem>,
    stop: &Arc<AtomicBool>,
    last_delivered: &mut Option<u64>,
) -> std::io::Result<()> {
    let mut s = UnixStream::connect(socket)?;
    // Short read timeout lets us check the stop flag responsively
    // without spinning.
    s.set_read_timeout(Some(Duration::from_millis(250)))?;

    let hello = encode(&ClientFrame::Hello {
        role: Role::Subscriber { since_seq },
    })
    .map_err(|e| std::io::Error::other(format!("encode hello: {}", e)))?;
    s.write_all(&hello)?;

    // If we were Disconnected, tell the caller we're back.
    if last_delivered.is_some() {
        let _ = tx.send(BusItem::Reconnected);
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 16 * 1024];

    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        match s.read(&mut tmp) {
            Ok(0) => {
                return Err(std::io::Error::other("bus closed"));
            }
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(e) => return Err(e),
        }

        loop {
            match decode::<BusFrame>(&buf) {
                Ok(Some((f, consumed))) => {
                    buf.drain(0..consumed);
                    match f {
                        BusFrame::Event {
                            seq,
                            kind,
                            ts,
                            terminal_session_id,
                            claude_pid,
                            payload_json,
                        } => {
                            *last_delivered = Some(seq);
                            let ev = BusEvent {
                                seq,
                                kind,
                                ts,
                                terminal_session_id,
                                claude_pid,
                                payload_json,
                            };
                            if tx.send(BusItem::Event(ev)).is_err() {
                                // Subscriber dropped — stop cleanly.
                                stop.store(true, Ordering::SeqCst);
                                return Ok(());
                            }
                        }
                        BusFrame::Lagged {
                            requested,
                            replay_from,
                        } => {
                            let _ = tx.send(BusItem::Gap {
                                last_delivered_seq: Some(requested.saturating_sub(1)),
                                replay_from,
                            });
                        }
                        BusFrame::ReplayEnd => {
                            // Informational; nothing to forward.
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(std::io::Error::other(format!("decode: {}", e)));
                }
            }
        }
    }
}
