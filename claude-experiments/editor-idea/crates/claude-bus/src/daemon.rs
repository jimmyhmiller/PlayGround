//! The bus daemon.
//!
//! Single-threaded `poll(2)` loop. One listener socket; each accepted
//! connection is either a publisher (writes frames, never reads) or a
//! subscriber (after Hello, receives a stream of `BusFrame::Event`).
//!
//! Publishes are durable: every accepted event is appended to
//! `~/.claude/events.jsonl` (same envelope shape the standalone logger
//! used to write) before the in-memory ring is updated. If the daemon
//! crashes between append and broadcast, subscribers will simply see
//! the events on next attach via `since_seq` + JSONL fallback.

#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::proto::{decode, encode, BusFrame, ClientFrame, Role};

/// Cap on retained events for replay. ~4 KB per event × 4096 ≈ 16 MB
/// worst case, well under the JSONL file we'd be replaying from anyway.
const RING_CAPACITY: usize = 4096;

/// Cap on a single subscriber's outbound buffer before we kick them.
/// Subscribers that can't keep up shouldn't be allowed to balloon the
/// daemon's memory — they'll see a `Lagged` on reattach.
const MAX_SEND_BUF: usize = 4 * 1024 * 1024;

/// Cap on a publisher's accumulated parse buffer. One hook payload is a
/// few KB; anything bigger is malformed.
const MAX_RECV_BUF: usize = 1 * 1024 * 1024;

/// Envelope written to the JSONL log. Kept in sync with the format the
/// standalone `claude-event-logger` used to write so existing readers
/// (drainer, ad-hoc `tail`) keep working.
#[derive(Serialize)]
struct JsonlEnvelope<'a> {
    kind: &'a str,
    ts: u64,
    terminal_session_id: &'a str,
    claude_pid: u32,
    /// Verbatim JSON payload object — already-encoded so we don't
    /// re-parse + re-serialize on the hot path.
    payload: &'a serde_json::Value,
}

/// One stored event, ready to ship to any subscriber.
struct Stored {
    seq: u64,
    /// Pre-encoded `BusFrame::Event` bytes (length-prefixed). Encoding
    /// once on publish saves N-subscribers' worth of work per event.
    frame: Vec<u8>,
}

enum ConnState {
    /// Hello not yet received.
    Anonymous { recv_buf: Vec<u8> },
    Publisher { recv_buf: Vec<u8> },
    Subscriber {
        send_buf: Vec<u8>,
        /// Index into the ring while replaying; `None` once caught up
        /// to live.
        replay_cursor: Option<usize>,
    },
}

struct Conn {
    stream: UnixStream,
    state: ConnState,
}

impl Conn {
    fn fd(&self) -> i32 {
        self.stream.as_raw_fd()
    }
    fn wants_pollout(&self) -> bool {
        matches!(&self.state, ConnState::Subscriber { send_buf, .. } if !send_buf.is_empty())
    }
}

pub struct Daemon {
    listener: UnixListener,
    conns: Vec<Conn>,
    /// Monotonically increasing event id. Survives only as long as the
    /// process — restarts begin at 0. Subscribers that care about
    /// continuity across restarts read JSONL.
    next_seq: u64,
    ring: VecDeque<Stored>,
    /// Append-only JSONL durable log. Held open for the daemon's life;
    /// `O_APPEND` keeps writes atomic w.r.t. any fallback writer.
    jsonl: std::fs::File,
}

impl Daemon {
    pub fn new(socket: &Path, jsonl: &Path) -> std::io::Result<Self> {
        if let Some(parent) = socket.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Stale socket from a previous (now-dead) daemon — clean up so
        // bind() succeeds. A live predecessor would still hold the
        // listener, in which case bind() will fail with EADDRINUSE and
        // we bail out below.
        if socket.exists() {
            let _ = std::fs::remove_file(socket);
        }
        let listener = UnixListener::bind(socket)?;
        listener.set_nonblocking(true)?;

        if let Some(parent) = jsonl.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let jsonl = OpenOptions::new()
            .append(true)
            .create(true)
            .mode(0o600)
            .open(jsonl)?;

        Ok(Self {
            listener,
            conns: Vec::new(),
            next_seq: 0,
            ring: VecDeque::with_capacity(RING_CAPACITY),
            jsonl,
        })
    }

    pub fn run(mut self) -> std::io::Result<()> {
        let listener_fd = self.listener.as_raw_fd();
        loop {
            let mut pollfds: Vec<libc::pollfd> = Vec::with_capacity(1 + self.conns.len());
            pollfds.push(libc::pollfd {
                fd: listener_fd,
                events: libc::POLLIN,
                revents: 0,
            });
            for c in &self.conns {
                let mut events = libc::POLLIN;
                if c.wants_pollout() {
                    events |= libc::POLLOUT;
                }
                pollfds.push(libc::pollfd {
                    fd: c.fd(),
                    events,
                    revents: 0,
                });
            }

            let ret = unsafe {
                libc::poll(pollfds.as_mut_ptr(), pollfds.len() as libc::nfds_t, 1000)
            };
            if ret < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() != std::io::ErrorKind::Interrupted {
                    std::thread::sleep(Duration::from_millis(10));
                }
                continue;
            }

            // Accept new connections.
            if pollfds[0].revents & libc::POLLIN != 0 {
                loop {
                    match self.listener.accept() {
                        Ok((s, _)) => {
                            let _ = s.set_nonblocking(true);
                            self.conns.push(Conn {
                                stream: s,
                                state: ConnState::Anonymous { recv_buf: Vec::new() },
                            });
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(_) => break,
                    }
                }
            }

            // Process each existing connection.
            let mut to_drop: Vec<usize> = Vec::new();
            for i in 0..self.conns.len() {
                // pollfds is [listener, conn0, conn1, ...]; conn i is at
                // pollfds[i+1]. New conns added this tick have no
                // corresponding entry — they'll be processed next loop.
                let pf_idx = i + 1;
                if pf_idx >= pollfds.len() {
                    continue;
                }
                let revents = pollfds[pf_idx].revents;
                if revents & libc::POLLOUT != 0 {
                    if self.flush_subscriber(i).is_err() {
                        to_drop.push(i);
                        continue;
                    }
                }
                if revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0 {
                    if self.handle_readable(i).is_err() {
                        to_drop.push(i);
                        continue;
                    }
                }
                self.pump_replay(i);
            }

            if !to_drop.is_empty() {
                to_drop.sort_unstable();
                to_drop.dedup();
                for i in to_drop.into_iter().rev() {
                    self.conns.remove(i);
                }
            }
        }
    }

    /// Read whatever is available on `conns[i]` and parse out frames.
    /// `Err(())` means "drop this connection".
    fn handle_readable(&mut self, i: usize) -> Result<(), ()> {
        let mut tmp = [0u8; 8192];
        // Distinguish EOF from a fatal read error so we can still flush
        // any frames that arrived alongside the disconnect (the common
        // case for fire-and-forget publishers).
        let mut peer_closed = false;
        loop {
            match self.conns[i].stream.read(&mut tmp) {
                Ok(0) => {
                    peer_closed = true;
                    break;
                }
                Ok(n) => match &mut self.conns[i].state {
                    ConnState::Anonymous { recv_buf } | ConnState::Publisher { recv_buf } => {
                        if recv_buf.len() + n > MAX_RECV_BUF {
                            return Err(());
                        }
                        recv_buf.extend_from_slice(&tmp[..n]);
                    }
                    // Subscribers shouldn't send post-Hello. Be lenient
                    // and drop the bytes rather than killing them.
                    ConnState::Subscriber { .. } => {}
                },
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => return Err(()),
            }
        }

        // Parse as many complete frames as recv_buf holds.
        loop {
            let (consumed, frame) = match &self.conns[i].state {
                ConnState::Anonymous { recv_buf } | ConnState::Publisher { recv_buf } => {
                    match decode::<ClientFrame>(recv_buf) {
                        Ok(Some((f, n))) => (n, f),
                        Ok(None) => break,
                        Err(_) => return Err(()),
                    }
                }
                ConnState::Subscriber { .. } => break,
            };
            match &mut self.conns[i].state {
                ConnState::Anonymous { recv_buf } | ConnState::Publisher { recv_buf } => {
                    recv_buf.drain(0..consumed);
                }
                ConnState::Subscriber { .. } => unreachable!(),
            }
            self.dispatch_frame(i, frame)?;
        }
        if peer_closed { Err(()) } else { Ok(()) }
    }

    fn dispatch_frame(&mut self, i: usize, frame: ClientFrame) -> Result<(), ()> {
        match frame {
            ClientFrame::Hello { role } => {
                if !matches!(self.conns[i].state, ConnState::Anonymous { .. }) {
                    return Err(()); // double Hello
                }
                match role {
                    Role::Publisher => {
                        // Preserve any bytes already buffered past the
                        // Hello — they're the first Publish frame(s).
                        let recv_buf = match std::mem::replace(
                            &mut self.conns[i].state,
                            ConnState::Anonymous { recv_buf: Vec::new() },
                        ) {
                            ConnState::Anonymous { recv_buf } => recv_buf,
                            _ => Vec::new(),
                        };
                        self.conns[i].state = ConnState::Publisher { recv_buf };
                    }
                    Role::Subscriber { since_seq } => {
                        let mut send_buf: Vec<u8> = Vec::new();
                        let replay_cursor = self.seed_replay(&mut send_buf, since_seq);
                        self.conns[i].state = ConnState::Subscriber {
                            send_buf,
                            replay_cursor,
                        };
                    }
                }
            }
            ClientFrame::Publish {
                kind,
                ts,
                terminal_session_id,
                claude_pid,
                payload_json,
            } => {
                if !matches!(self.conns[i].state, ConnState::Publisher { .. }) {
                    return Err(());
                }
                self.accept_publish(kind, ts, terminal_session_id, claude_pid, payload_json);
            }
        }
        Ok(())
    }

    /// Seed a fresh subscriber's send_buf based on its `since_seq`.
    /// Returns the index into `self.ring` the replay pump should
    /// continue from, or `None` if the subscriber starts at live.
    fn seed_replay(&self, send_buf: &mut Vec<u8>, since_seq: Option<u64>) -> Option<usize> {
        let Some(since) = since_seq else {
            return None;
        };
        if self.ring.is_empty() {
            return None;
        }
        let oldest = self.ring.front().map(|s| s.seq).unwrap_or(0);
        let newest = self.ring.back().map(|s| s.seq).unwrap_or(0);
        if since > newest {
            return None;
        }
        if since < oldest {
            if let Ok(b) = encode(&BusFrame::Lagged {
                requested: since,
                replay_from: oldest,
            }) {
                send_buf.extend_from_slice(&b);
            }
            return Some(0);
        }
        for (idx, s) in self.ring.iter().enumerate() {
            if s.seq >= since {
                return Some(idx);
            }
        }
        None
    }

    /// Top up a subscriber's send_buf with more replay frames. Emits a
    /// `ReplayEnd` once the cursor catches the ring tail.
    fn pump_replay(&mut self, i: usize) {
        let ConnState::Subscriber {
            send_buf,
            replay_cursor,
        } = &mut self.conns[i].state
        else {
            return;
        };
        let Some(mut cursor) = *replay_cursor else {
            return;
        };
        while cursor < self.ring.len() && send_buf.len() < MAX_SEND_BUF {
            send_buf.extend_from_slice(&self.ring[cursor].frame);
            cursor += 1;
        }
        if cursor >= self.ring.len() {
            if let Ok(b) = encode(&BusFrame::ReplayEnd) {
                send_buf.extend_from_slice(&b);
            }
            *replay_cursor = None;
        } else {
            *replay_cursor = Some(cursor);
        }
    }

    fn flush_subscriber(&mut self, i: usize) -> Result<(), ()> {
        let conn = &mut self.conns[i];
        let ConnState::Subscriber { send_buf, .. } = &mut conn.state else {
            return Ok(());
        };
        while !send_buf.is_empty() {
            match conn.stream.write(send_buf) {
                Ok(0) => break,
                Ok(n) => {
                    send_buf.drain(0..n);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => return Err(()),
            }
        }
        if send_buf.len() > MAX_SEND_BUF {
            return Err(());
        }
        Ok(())
    }

    fn accept_publish(
        &mut self,
        kind: String,
        ts: u64,
        terminal_session_id: String,
        claude_pid: u32,
        payload_json: String,
    ) {
        let seq = self.next_seq;
        self.next_seq += 1;

        let payload: serde_json::Value = serde_json::from_str(&payload_json)
            .unwrap_or_else(|_| serde_json::Value::String(payload_json.clone()));

        let env = JsonlEnvelope {
            kind: &kind,
            ts,
            terminal_session_id: &terminal_session_id,
            claude_pid,
            payload: &payload,
        };
        if let Ok(mut line) = serde_json::to_string(&env) {
            line.push('\n');
            let _ = self.jsonl.write_all(line.as_bytes());
        }

        let frame = match encode(&BusFrame::Event {
            seq,
            kind,
            ts,
            terminal_session_id,
            claude_pid,
            payload_json,
        }) {
            Ok(b) => b,
            Err(_) => return,
        };

        if self.ring.len() == RING_CAPACITY {
            self.ring.pop_front();
        }
        self.ring.push_back(Stored {
            seq,
            frame: frame.clone(),
        });

        // Broadcast to live subscribers (those past their replay). The
        // ones still replaying will pick this up naturally — we just
        // pushed it into the ring they're walking.
        for c in &mut self.conns {
            if let ConnState::Subscriber {
                send_buf,
                replay_cursor,
            } = &mut c.state
            {
                if replay_cursor.is_none() && send_buf.len() < MAX_SEND_BUF {
                    send_buf.extend_from_slice(&frame);
                }
            }
        }
    }
}

/// Daemonize using the same double-fork pattern as terminal-daemon.
/// Skip when `CLAUDE_BUS_FOREGROUND=1` (tests / debugging).
pub fn daemonize_if_requested() {
    if std::env::var_os("CLAUDE_BUS_FOREGROUND").is_some() {
        return;
    }
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return;
    }
    if pid > 0 {
        std::process::exit(0);
    }
    unsafe {
        libc::setsid();
    }
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return;
    }
    if pid > 0 {
        std::process::exit(0);
    }
    if let Ok(devnull) = std::fs::OpenOptions::new().read(true).write(true).open("/dev/null") {
        let fd = devnull.as_raw_fd();
        unsafe {
            libc::dup2(fd, 0);
            libc::dup2(fd, 1);
            libc::dup2(fd, 2);
        }
    }
    if let Some(path) = std::env::var_os("CLAUDE_BUS_LOG") {
        if let Ok(f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let fd = f.as_raw_fd();
            unsafe {
                libc::dup2(fd, 2);
            }
            std::mem::forget(f);
        }
    }
}

pub fn write_pid_file(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, std::process::id().to_string().as_bytes())
}

pub fn run(socket: PathBuf, jsonl: PathBuf, pid_path: PathBuf) -> std::io::Result<()> {
    let d = Daemon::new(&socket, &jsonl)?;
    write_pid_file(&pid_path)?;
    eprintln!(
        "[claude-bus] listening on {} (pid={})",
        socket.display(),
        std::process::id()
    );
    d.run()
}
