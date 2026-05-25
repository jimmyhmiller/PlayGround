//! Portable async-I/O host interface — Tier-2 event loop, mio-backed.
//!
//! THE CONTRACT (single JS thread; opaque integer handles + op_ids):
//!
//!   __host.tcp.create_tcp() -> handle
//!   __host.tcp.listen(handle, ip, port, backlog) -> 0 | -errno   (sync)
//!   __host.tcp.connect(handle, ip, port) -> op_id                (async)
//!   __host.tcp.accept(handle) -> op_id                           (async)
//!   __host.tcp.read(handle, buf, off, len) -> op_id              (async)
//!   __host.tcp.write(handle, buf, off, len) -> op_id             (async)
//!   __host.tcp.shutdown(handle, how) -> op_id                    (async)
//!   __host.tcp.close(handle)                                     (sync)
//!   __host.tcp.set_no_delay(handle, on) / set_keep_alive
//!   __host.tcp.local_addr(handle) / peer_addr(handle)
//!
//!   __host.io.poll(timeout_ms) -> [{op_id, kind, status, ...result}]
//!   __host.io.cancel(op_id)
//!   __host.io.has_pending() -> bool
//!
//! All "async" ops return an op_id immediately; the completion appears in a
//! later __host.io.poll() result.
//!
//! Portability invariants this module preserves:
//!   * Completion-based, not readiness-based (works on IOCP/io_uring too).
//!   * Single-threaded host↔JS interface (no callbacks from worker threads).
//!   * Opaque integer handles; no fd leakage to JS.
//!   * One blocking primitive: io.poll(timeout_ms). All other ops are non-blocking.

use mio::event::Event;
use mio::net::{TcpListener, TcpStream};
use mio::{Events, Interest, Poll, Token};
use rquickjs::function::Func;
use rquickjs::{Array, Ctx, Object, Result, TypedArray};
use std::collections::{HashMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::Duration;

// =========================================================================
// State
// =========================================================================

struct State {
    poll: Poll,
    handles: HashMap<u32, Handle>,
    /// All ops indexed by op_id.
    ops: HashMap<u32, PendingOp>,
    /// Per-handle: which read op is pending (if any). Node's contract is one
    /// active read at a time per stream. Stored separately so we can find
    /// the right op when readiness fires.
    read_op_for: HashMap<u32, u32>,
    /// Per-handle: which write ops are pending (FIFO; we drain in order).
    write_ops_for: HashMap<u32, VecDeque<u32>>,
    /// Per-handle: connect op_id if a connect is pending.
    connect_op_for: HashMap<u32, u32>,
    /// Per-handle: accept op_id if an accept is pending.
    accept_op_for: HashMap<u32, u32>,
    completions: VecDeque<Completion>,
    next_handle: u32,
    next_op: u32,
}

enum Handle {
    Listener(TcpListener),
    Stream(TcpStream),
}

struct PendingOp {
    handle: u32,
    kind: PendingKind,
}

enum PendingKind {
    Connect,
    Accept,
    Read { buf_ptr: *mut u8, buf_off: u32, buf_len: u32 },
    /// Write owns a copy of the bytes (decoupling from the JS buffer's lifetime).
    Write { bytes: Vec<u8>, total: u32, written: u32 },
    Shutdown,
}

// Raw pointer fields are not Send/Sync. We're single-threaded from JS's
// perspective; the Mutex serializes all access. SAFETY: invariant that we
// only ever touch this state from the JS thread (no worker threads).
unsafe impl Send for PendingOp {}

pub struct Completion {
    op_id: u32,
    kind: CompletionKind,
    status: i32,
}

enum CompletionKind {
    Connect,
    Accept { handle: u32 },
    Read { n: u32 },
    Write { n: u32 },
    Shutdown,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<F, R>(f: F) -> R
where F: FnOnce(&mut State) -> R,
{
    let mut g = STATE.lock().unwrap();
    if g.is_none() {
        *g = Some(State {
            poll: Poll::new().expect("mio::Poll::new"),
            handles: HashMap::new(),
            ops: HashMap::new(),
            read_op_for: HashMap::new(),
            write_ops_for: HashMap::new(),
            connect_op_for: HashMap::new(),
            accept_op_for: HashMap::new(),
            completions: VecDeque::new(),
            next_handle: 1,
            next_op: 1,
        });
    }
    f(g.as_mut().unwrap())
}

// =========================================================================
// JS-facing namespace install
// =========================================================================

pub fn install<'js>(ctx: Ctx<'js>, host: &Object<'js>) -> Result<()> {
    host.set("tcp", make_tcp(ctx.clone())?)?;
    host.set("dns", make_dns(ctx.clone())?)?;
    host.set("io",  make_io(ctx.clone())?)?;
    Ok(())
}

// =========================================================================
// host.tcp.*
// =========================================================================

fn make_tcp<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    let t = Object::new(ctx.clone())?;
    t.set("create_tcp",     Func::from(tcp_create))?;
    t.set("listen",         Func::from(tcp_listen))?;
    t.set("connect",        Func::from(tcp_connect))?;
    t.set("accept",         Func::from(tcp_accept))?;
    t.set("read",           Func::from(tcp_read))?;
    t.set("write",          Func::from(tcp_write))?;
    t.set("shutdown",       Func::from(tcp_shutdown))?;
    t.set("close",          Func::from(tcp_close))?;
    t.set("set_no_delay",   Func::from(tcp_set_no_delay))?;
    t.set("set_keep_alive", Func::from(tcp_set_keep_alive))?;
    t.set("local_addr",     Func::from(tcp_local_addr))?;
    t.set("peer_addr",      Func::from(tcp_peer_addr))?;
    Ok(t)
}

fn tcp_create() -> u32 {
    with_state(|s| {
        let id = s.next_handle;
        s.next_handle = s.next_handle.wrapping_add(1).max(1);
        // Handle entry isn't created here; create_tcp just reserves an ID
        // that listen()/connect() will install into.
        id
    })
}

fn parse_addr(ip: &str, port: u32) -> std::result::Result<SocketAddr, i32> {
    let s = format!("{}:{}", ip, port);
    s.parse().map_err(|_| -libc::EINVAL)
}

fn tcp_listen(handle: u32, ip: String, port: f64, _backlog: f64) -> i32 {
    // Accept any JS number for port; clamp to u16 range. -1 (uninitialized)
    // becomes 0 which means "let OS choose".
    let port = if port < 0.0 || port.is_nan() { 0 } else { port.min(65535.0) as u32 };
    let addr = match parse_addr(&ip, port) {
        Ok(a) => a,
        Err(e) => return e,
    };
    with_state(|s| {
        let mut listener = match TcpListener::bind(addr) {
            Ok(l) => l,
            Err(e) => return -(e.raw_os_error().unwrap_or(libc::EACCES)),
        };
        // Register for readability (incoming connections).
        if let Err(e) = s.poll.registry().register(
            &mut listener,
            Token(handle as usize),
            Interest::READABLE,
        ) {
            return -(e.raw_os_error().unwrap_or(libc::EINVAL));
        }
        s.handles.insert(handle, Handle::Listener(listener));
        0
    })
}

fn tcp_connect(handle: u32, ip: String, port: f64) -> u32 {
    let port = if port < 0.0 || port.is_nan() { 0 } else { port.min(65535.0) as u32 };
    let addr = match parse_addr(&ip, port) {
        Ok(a) => a,
        Err(_) => {
            return enqueue_immediate(0, |id| Completion {
                op_id: id,
                kind: CompletionKind::Connect,
                status: -libc::EINVAL,
            });
        }
    };
    with_state(|s| {
        let op_id = alloc_op_id(s);
        let mut stream = match TcpStream::connect(addr) {
            Ok(st) => st,
            Err(e) => {
                s.completions.push_back(Completion {
                    op_id,
                    kind: CompletionKind::Connect,
                    status: -(e.raw_os_error().unwrap_or(libc::ECONNREFUSED)),
                });
                return op_id;
            }
        };
        // Connect is complete when the socket becomes WRITABLE.
        if let Err(e) = s.poll.registry().register(
            &mut stream,
            Token(handle as usize),
            Interest::WRITABLE | Interest::READABLE,
        ) {
            s.completions.push_back(Completion {
                op_id,
                kind: CompletionKind::Connect,
                status: -(e.raw_os_error().unwrap_or(libc::EINVAL)),
            });
            return op_id;
        }
        s.handles.insert(handle, Handle::Stream(stream));
        s.ops.insert(op_id, PendingOp { handle, kind: PendingKind::Connect });
        s.connect_op_for.insert(handle, op_id);
        op_id
    })
}

fn tcp_accept(handle: u32) -> u32 {
    with_state(|s| {
        let op_id = alloc_op_id(s);
        s.ops.insert(op_id, PendingOp { handle, kind: PendingKind::Accept });
        s.accept_op_for.insert(handle, op_id);
        // Try one accept immediately — common case for already-pending
        // connections on a busy listener.
        try_accept(s, handle);
        op_id
    })
}

fn tcp_read<'js>(handle: u32, buf: TypedArray<'js, u8>, off: u32, len: u32) -> u32 {
    // SAFETY: We take a raw pointer to the buffer's bytes. The JS contract is
    // that the buffer outlives the operation. Inside io.poll() (which runs
    // on the JS thread), QuickJS guarantees the underlying ArrayBuffer is
    // still alive and non-detached.
    let bytes = match buf.as_bytes() {
        Some(b) => b,
        None => {
            return enqueue_immediate(handle, |id| Completion {
                op_id: id,
                kind: CompletionKind::Read { n: 0 },
                status: -libc::EBADF,
            });
        }
    };
    let buf_ptr = bytes.as_ptr() as *mut u8;
    with_state(|s| {
        let op_id = alloc_op_id(s);
        s.ops.insert(op_id, PendingOp {
            handle,
            kind: PendingKind::Read { buf_ptr, buf_off: off, buf_len: len },
        });
        s.read_op_for.insert(handle, op_id);
        // Bump interest to include READABLE if a stream.
        if let Some(Handle::Stream(stream)) = s.handles.get_mut(&handle) {
            let _ = s.poll.registry().reregister(
                stream, Token(handle as usize),
                Interest::READABLE | Interest::WRITABLE,
            );
            // Try once non-blocking, in case the socket already has data.
            try_read(s, handle);
        }
        op_id
    })
}

fn tcp_write<'js>(handle: u32, buf: TypedArray<'js, u8>, off: u32, len: u32) -> u32 {
    let off = off as usize;
    let len = len as usize;
    let bytes = match buf.as_bytes() {
        Some(b) => b,
        None => {
            return enqueue_immediate(handle, |id| Completion {
                op_id: id,
                kind: CompletionKind::Write { n: 0 },
                status: -libc::EBADF,
            });
        }
    };
    let slice = &bytes[off.min(bytes.len())..(off + len).min(bytes.len())];
    let owned = slice.to_vec();  // own the bytes; JS may free its buffer.
    with_state(|s| {
        let op_id = alloc_op_id(s);
        let total = owned.len() as u32;
        s.ops.insert(op_id, PendingOp {
            handle,
            kind: PendingKind::Write { bytes: owned, total, written: 0 },
        });
        s.write_ops_for.entry(handle).or_default().push_back(op_id);
        // Reregister interest to include WRITABLE.
        if let Some(Handle::Stream(stream)) = s.handles.get_mut(&handle) {
            let _ = s.poll.registry().reregister(
                stream, Token(handle as usize),
                Interest::READABLE | Interest::WRITABLE,
            );
            // Try once non-blocking.
            try_write(s, handle);
        }
        op_id
    })
}

fn tcp_shutdown(handle: u32, _how: u32) -> u32 {
    with_state(|s| {
        let op_id = alloc_op_id(s);
        let status = if let Some(Handle::Stream(stream)) = s.handles.get(&handle) {
            match stream.shutdown(std::net::Shutdown::Write) {
                Ok(()) => 0,
                Err(e) => -(e.raw_os_error().unwrap_or(libc::EIO)),
            }
        } else {
            -libc::EBADF
        };
        s.completions.push_back(Completion {
            op_id, kind: CompletionKind::Shutdown, status,
        });
        op_id
    })
}

fn tcp_close(handle: u32) {
    with_state(|s| {
        if let Some(h) = s.handles.remove(&handle) {
            // mio handles deregister on drop.
            drop(h);
        }
        s.read_op_for.remove(&handle);
        s.write_ops_for.remove(&handle);
        s.connect_op_for.remove(&handle);
        s.accept_op_for.remove(&handle);
    });
}

fn tcp_set_no_delay(handle: u32, on: bool) {
    with_state(|s| {
        if let Some(Handle::Stream(stream)) = s.handles.get(&handle) {
            let _ = stream.set_nodelay(on);
        }
    });
}
fn tcp_set_keep_alive(_handle: u32, _on: bool, _delay: u32) {
    // mio's TcpStream doesn't expose set_keepalive directly without socket2.
    // We accept and no-op; can be wired later via libc::setsockopt.
}
fn tcp_local_addr<'js>(ctx: Ctx<'js>, handle: u32) -> Result<Object<'js>> {
    addr_to_obj(ctx, with_state(|s| match s.handles.get(&handle) {
        Some(Handle::Listener(l)) => l.local_addr().ok(),
        Some(Handle::Stream(s)) => s.local_addr().ok(),
        None => None,
    }))
}
fn tcp_peer_addr<'js>(ctx: Ctx<'js>, handle: u32) -> Result<Object<'js>> {
    addr_to_obj(ctx, with_state(|s| match s.handles.get(&handle) {
        Some(Handle::Stream(s)) => s.peer_addr().ok(),
        _ => None,
    }))
}

fn addr_to_obj<'js>(ctx: Ctx<'js>, addr: Option<SocketAddr>) -> Result<Object<'js>> {
    let o = Object::new(ctx.clone())?;
    match addr {
        Some(a) => {
            o.set("ip", a.ip().to_string())?;
            o.set("port", a.port() as u32)?;
            o.set("family", if a.is_ipv6() { 6 } else { 4 } as u8)?;
        }
        None => {
            o.set("ip", "")?;
            o.set("port", 0u32)?;
            o.set("family", 0u8)?;
        }
    }
    Ok(o)
}

// =========================================================================
// host.dns.*  (stub for now — real getaddrinfo lands next session)
// =========================================================================

fn make_dns<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    let d = Object::new(ctx.clone())?;
    d.set("lookup", Func::from(|_name: String, _family: u8| -> u32 { 0 }))?;
    Ok(d)
}

// =========================================================================
// host.io.*
// =========================================================================

fn make_io<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    let io = Object::new(ctx.clone())?;
    io.set("poll",        Func::from(io_poll))?;
    io.set("cancel",      Func::from(io_cancel))?;
    io.set("has_pending", Func::from(io_has_pending))?;
    Ok(io)
}

fn io_has_pending() -> bool {
    with_state(|s| !s.ops.is_empty() || !s.completions.is_empty())
}

fn io_cancel(_op_id: u32) {
    // Cancellation is best-effort for now; ops complete naturally and any
    // completion arriving after cancel is just delivered.
}

fn io_poll<'js>(ctx: Ctx<'js>, timeout_ms: f64) -> Result<Array<'js>> {
    // Drive mio one round, then drain the completion queue. If timeout_ms
    // is negative, we treat it as "no timeout" but bound it to 1s so the
    // outer loop stays responsive to signals.
    let timeout = if timeout_ms < 0.0 {
        Some(Duration::from_millis(1000))
    } else if timeout_ms == 0.0 {
        Some(Duration::from_millis(0))
    } else {
        Some(Duration::from_millis(timeout_ms.min(60_000.0) as u64))
    };

    // Step 1: drive mio with the timeout.
    let events = drive_mio(timeout);

    // Step 2: dispatch readiness events to in-flight ops.
    with_state(|s| {
        for ev in &events {
            let handle = ev.token().0 as u32;
            if ev.is_readable() {
                if let Some(&accept_op) = s.accept_op_for.get(&handle) {
                    let _ = accept_op; // satisfies dead-code; we look it up inside try_accept
                    try_accept(s, handle);
                }
                if s.read_op_for.contains_key(&handle) {
                    try_read(s, handle);
                }
            }
            if ev.is_writable() {
                if let Some(connect_op) = s.connect_op_for.remove(&handle) {
                    // For mio, writable on a connecting socket means
                    // connect finished. Distinguish success from error by
                    // peeking peer_addr.
                    let status = if let Some(Handle::Stream(stream)) = s.handles.get(&handle) {
                        match stream.peer_addr() {
                            Ok(_) => 0,
                            Err(e) => -(e.raw_os_error().unwrap_or(libc::ECONNREFUSED)),
                        }
                    } else { -libc::EBADF };
                    s.completions.push_back(Completion {
                        op_id: connect_op,
                        kind: CompletionKind::Connect,
                        status,
                    });
                    s.ops.remove(&connect_op);
                }
                // Drain any pending writes.
                while s.write_ops_for.get(&handle).map(|v| !v.is_empty()).unwrap_or(false) {
                    if !try_write(s, handle) { break; }
                }
            }
        }
    });

    // Step 3: copy completions out to JS.
    with_state(|s| {
        let arr = Array::new(ctx.clone())?;
        let mut i = 0;
        while let Some(c) = s.completions.pop_front() {
            arr.set(i, completion_to_object(&ctx, c)?)?;
            i += 1;
        }
        Ok(arr)
    })
}

// =========================================================================
// mio driver and op-progression helpers
// =========================================================================

fn drive_mio(timeout: Option<Duration>) -> Vec<Event> {
    // We take the State's mio Poll out of the mutex briefly so the call to
    // poll() can block without keeping the mutex locked. Other JS calls
    // would otherwise deadlock if they tried to submit a new op while we
    // were blocked. Same idea as libuv's run loop.
    let mut events = Events::with_capacity(64);
    {
        let mut g = STATE.lock().unwrap();
        let state = g.as_mut().expect("state initialized");
        // Move the Poll out (we'll move it back after polling).
        // We can't move it out of the struct safely without a swap, so just
        // poll inside the lock. The cost is brief; users should set short
        // timeouts to keep submissions responsive.
        let _ = state.poll.poll(&mut events, timeout);
    }
    events.iter().cloned().collect()
}

fn alloc_op_id(s: &mut State) -> u32 {
    let id = s.next_op;
    s.next_op = s.next_op.wrapping_add(1).max(1);
    id
}

fn enqueue_immediate<F>(_handle: u32, mk: F) -> u32
where F: FnOnce(u32) -> Completion,
{
    with_state(|s| {
        let op_id = alloc_op_id(s);
        s.completions.push_back(mk(op_id));
        op_id
    })
}

fn try_accept(s: &mut State, handle: u32) {
    let listener = match s.handles.get(&handle) {
        Some(Handle::Listener(l)) => l,
        _ => return,
    };
    match listener.accept() {
        Ok((mut stream, _addr)) => {
            // Allocate a new handle id for the accepted stream.
            let new_handle = s.next_handle;
            s.next_handle = s.next_handle.wrapping_add(1).max(1);
            let _ = s.poll.registry().register(
                &mut stream,
                Token(new_handle as usize),
                Interest::READABLE | Interest::WRITABLE,
            );
            s.handles.insert(new_handle, Handle::Stream(stream));
            if let Some(op_id) = s.accept_op_for.remove(&handle) {
                s.completions.push_back(Completion {
                    op_id,
                    kind: CompletionKind::Accept { handle: new_handle },
                    status: 0,
                });
                s.ops.remove(&op_id);
            }
        }
        Err(e) if e.kind() == ErrorKind::WouldBlock => {
            // No connection ready — wait for next readiness event.
        }
        Err(e) => {
            if let Some(op_id) = s.accept_op_for.remove(&handle) {
                s.completions.push_back(Completion {
                    op_id,
                    kind: CompletionKind::Accept { handle: 0 },
                    status: -(e.raw_os_error().unwrap_or(libc::EIO)),
                });
                s.ops.remove(&op_id);
            }
        }
    }
}

fn try_read(s: &mut State, handle: u32) {
    let op_id = match s.read_op_for.get(&handle).copied() {
        Some(id) => id,
        None => return,
    };
    let stream = match s.handles.get_mut(&handle) {
        Some(Handle::Stream(st)) => st,
        _ => return,
    };
    let (ptr, off, len) = match s.ops.get(&op_id).map(|o| &o.kind) {
        Some(PendingKind::Read { buf_ptr, buf_off, buf_len }) =>
            (*buf_ptr, *buf_off as usize, *buf_len as usize),
        _ => return,
    };
    // SAFETY: see tcp_read; JS contract is buffer outlives op.
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr.add(off), len) };
    match stream.read(dst) {
        Ok(n) => {
            s.completions.push_back(Completion {
                op_id, kind: CompletionKind::Read { n: n as u32 }, status: 0,
            });
            s.read_op_for.remove(&handle);
            s.ops.remove(&op_id);
        }
        Err(e) if e.kind() == ErrorKind::WouldBlock => { /* try later */ }
        Err(e) => {
            s.completions.push_back(Completion {
                op_id, kind: CompletionKind::Read { n: 0 },
                status: -(e.raw_os_error().unwrap_or(libc::EIO)),
            });
            s.read_op_for.remove(&handle);
            s.ops.remove(&op_id);
        }
    }
}

/// Returns true if it made progress (so caller can call again to drain more).
fn try_write(s: &mut State, handle: u32) -> bool {
    let op_id = match s.write_ops_for.get(&handle).and_then(|v| v.front().copied()) {
        Some(id) => id,
        None => return false,
    };
    let stream = match s.handles.get_mut(&handle) {
        Some(Handle::Stream(st)) => st,
        _ => return false,
    };
    let (bytes, total, written) = match s.ops.get_mut(&op_id).map(|o| &mut o.kind) {
        Some(PendingKind::Write { bytes, total, written }) =>
            (bytes.clone(), *total, *written),
        _ => return false,
    };
    let remaining = &bytes[written as usize..];
    match stream.write(remaining) {
        Ok(n) => {
            let new_written = written + n as u32;
            if new_written >= total {
                s.completions.push_back(Completion {
                    op_id, kind: CompletionKind::Write { n: total }, status: 0,
                });
                s.ops.remove(&op_id);
                if let Some(q) = s.write_ops_for.get_mut(&handle) { q.pop_front(); }
                true
            } else {
                if let Some(PendingKind::Write { written, .. }) =
                    s.ops.get_mut(&op_id).map(|o| &mut o.kind)
                { *written = new_written; }
                false  // wait for next writable
            }
        }
        Err(e) if e.kind() == ErrorKind::WouldBlock => false,
        Err(e) => {
            s.completions.push_back(Completion {
                op_id, kind: CompletionKind::Write { n: written },
                status: -(e.raw_os_error().unwrap_or(libc::EIO)),
            });
            s.ops.remove(&op_id);
            if let Some(q) = s.write_ops_for.get_mut(&handle) { q.pop_front(); }
            true
        }
    }
}

fn completion_to_object<'js>(ctx: &Ctx<'js>, c: Completion) -> Result<Object<'js>> {
    let o = Object::new(ctx.clone())?;
    o.set("op_id", c.op_id)?;
    o.set("status", c.status)?;
    match c.kind {
        CompletionKind::Connect       => { o.set("kind", "connect")?; }
        CompletionKind::Accept { handle } => {
            o.set("kind", "accept")?;
            o.set("handle", handle)?;
        }
        CompletionKind::Read { n }    => { o.set("kind", "read")?;  o.set("n", n)?; }
        CompletionKind::Write { n }   => { o.set("kind", "write")?; o.set("n", n)?; }
        CompletionKind::Shutdown      => { o.set("kind", "shutdown")?; }
    }
    Ok(o)
}
