//! TCP transport for the `at` protocol.
//!
//! Two message kinds in v1:
//!
//! - `Call(closure_bytes)` — the client ships a serialized zero-arg
//!   closure. The server decodes it, invokes the JIT'd lambda the
//!   closure points to (looked up by content hash in the server's
//!   code table), and returns the result.
//! - `Result(value_bytes)` — the server's reply.
//!
//! Both sides must have compiled the same source so the server has the
//! lambda's JIT'd entry point in its code table. The future
//! `NeedCode`/`Code` round-trip relaxes this: the server can pull
//! missing canonical defs from the client mid-decode.
//!
//! ## Frame format
//!
//! Every message is a length-prefixed payload: `u32 BE len` followed by
//! `len` bytes. The first byte of the payload is the message kind.
//!
//! ```text
//! frame := len:u32_BE  body:[u8; len]
//! body  := kind:u8  payload:[u8]
//!
//! kind = 0 → Call    : payload is encoded closure bytes (per `wire`)
//! kind = 1 → Result  : payload is encoded value bytes (per `wire`)
//! ```
//!
//! ## v1 restrictions
//!
//! - **Zero-arg closures only.** A general `at(node, |x| …)` would need
//!   to ship arguments too; we can extend the protocol later.
//! - **Int return only.** The protocol can carry any wire value, but
//!   the demo invocation expects the closure to return Int.
//! - **No timeouts / retries / failure model.** A broken connection
//!   surfaces as `io::Error`; future work adds `Failure::Unreachable`
//!   etc. to match the proposal.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;

use crate::codegen::{IncrementalJit, ShapeMeta};
use crate::gc::{Full, TypeInfo};
use crate::hash::Hash;
use crate::knowledge::KnowledgeBase;
use crate::resolve::AtBinding as ResolverAtBinding;
use crate::runtime::{Runtime, Thread, ai_gc_alloc_closure};
use crate::wire::{WireError, WireValue, decode_value, encode_value};

// =============================================================================
// Frame protocol
// =============================================================================

pub const KIND_CALL: u8 = 0;
pub const KIND_RESULT: u8 = 1;
pub const KIND_NEED_CODE: u8 = 2;
pub const KIND_CODE: u8 = 3;

// Located-state (distributed atom) request kinds. Each request body is
// `[kind][cell_key: 32 bytes][payload...]` where `cell_key` is
// `blake3(name)` identifying the cell on the owning node.
//
// GET   payload: none.            reply: RESULT(value) | ABSENT
// SET   payload: encoded value.   reply: RESULT(value)
// INIT  payload: encoded init.    reply: RESULT(current)  (init-if-absent)
// SWAP  payload: encoded closure. reply: RESULT(new) | ABSENT
//
// The server stores the **raw `encode_value` bytes** as a cell's value;
// GET/SET/INIT are pure byte operations that never touch the heap or
// JIT. Only SWAP decodes (the closure, and the current value) and may
// run the NeedCode/Code handshake to pull the closure's code.
pub const KIND_ATOM_GET: u8 = 4;
pub const KIND_ATOM_SET: u8 = 5;
pub const KIND_ATOM_INIT: u8 = 6;
pub const KIND_ATOM_SWAP: u8 = 7;
/// Reply indicating GET/SWAP referenced a cell that does not exist.
pub const KIND_ATOM_ABSENT: u8 = 8;

/// Length of a cell key (a Blake3 hash) in an atom request frame.
const CELL_KEY_LEN: usize = 32;

/// Tag identifying what kind of canonical AST item a [`Code`] entry carries.
///
/// `Def` items are `encode_def`'d top-level definitions (Fn / Struct /
/// Enum). `Lambda` items are `encode_expr`'d `Expr::Lambda { ... }` nodes
/// — lifted lambdas have their own content hash distinct from any
/// enclosing def, and the receiver may need just the lambda's bytes
/// (not the whole def) to materialize a closure shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ItemKind {
    Def = 0,
    Lambda = 1,
    /// A C/host extern requirement. The bytes are an
    /// [`crate::codec::encode_extern`] blob (name + signature + library).
    /// Externs aren't content-addressed; this carries the "to run this
    /// code you must be able to resolve symbol X from library Y"
    /// requirement to the receiving node, which resolves it (or fails
    /// clearly if the library/symbol isn't available there).
    Extern = 2,
}

impl ItemKind {
    pub fn from_u8(b: u8) -> Result<Self, NetError> {
        match b {
            0 => Ok(ItemKind::Def),
            1 => Ok(ItemKind::Lambda),
            2 => Ok(ItemKind::Extern),
            other => Err(NetError::ProtocolViolation_owned(format!(
                "unknown ItemKind byte {}",
                other
            ))),
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Encode a `NeedCode` payload (without the outer length prefix and
/// without the kind byte; caller prepends those).
///
/// Layout: u32 BE count + (32 bytes Hash) × count.
pub fn encode_need_code(hashes: &[Hash]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + hashes.len() * 32);
    out.push(KIND_NEED_CODE);
    let n: u32 = hashes.len().try_into().expect("too many hashes");
    out.extend_from_slice(&n.to_be_bytes());
    for h in hashes {
        out.extend_from_slice(h.as_bytes());
    }
    out
}

/// Decode a `NeedCode` payload. Input is the frame body WITH the leading
/// kind byte (which must equal `KIND_NEED_CODE`).
pub fn decode_need_code(body: &[u8]) -> Result<Vec<Hash>, NetError> {
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty NeedCode body"));
    }
    if body[0] != KIND_NEED_CODE {
        return Err(NetError::BadKind(body[0]));
    }
    if body.len() < 5 {
        return Err(NetError::ProtocolViolation(
            "NeedCode body truncated before count",
        ));
    }
    let count = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
    let need = 5 + count * 32;
    if body.len() < need {
        return Err(NetError::ProtocolViolation(
            "NeedCode body truncated before all hashes",
        ));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = 5 + i * 32;
        let mut buf = [0u8; 32];
        buf.copy_from_slice(&body[start..start + 32]);
        out.push(Hash(buf));
    }
    if body.len() != need {
        return Err(NetError::ProtocolViolation(
            "NeedCode body has trailing bytes",
        ));
    }
    Ok(out)
}

/// Encode a `Code` payload. Each entry is `(item_kind, hash, canonical_bytes)`.
///
/// Layout: u32 BE count + (u8 kind + 32 byte Hash + u32 BE len + len bytes) × count.
pub fn encode_code(items: &[(ItemKind, Hash, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + items.len() * (1 + 32 + 4 + 32));
    out.push(KIND_CODE);
    let n: u32 = items.len().try_into().expect("too many items");
    out.extend_from_slice(&n.to_be_bytes());
    for (k, h, bytes) in items {
        out.push(k.as_u8());
        out.extend_from_slice(h.as_bytes());
        let blen: u32 = bytes.len().try_into().expect("item too large");
        out.extend_from_slice(&blen.to_be_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Decode a `Code` payload. Returns the same `(kind, hash, bytes)`
/// triples that `encode_code` produced.
pub fn decode_code(body: &[u8]) -> Result<Vec<(ItemKind, Hash, Vec<u8>)>, NetError> {
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty Code body"));
    }
    if body[0] != KIND_CODE {
        return Err(NetError::BadKind(body[0]));
    }
    if body.len() < 5 {
        return Err(NetError::ProtocolViolation(
            "Code body truncated before count",
        ));
    }
    let count = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
    let mut out = Vec::with_capacity(count);
    let mut pos = 5usize;
    for _ in 0..count {
        if pos + 1 + 32 + 4 > body.len() {
            return Err(NetError::ProtocolViolation(
                "Code body truncated mid-entry header",
            ));
        }
        let kind = ItemKind::from_u8(body[pos])?;
        pos += 1;
        let mut hash_buf = [0u8; 32];
        hash_buf.copy_from_slice(&body[pos..pos + 32]);
        pos += 32;
        let blen = u32::from_be_bytes([
            body[pos],
            body[pos + 1],
            body[pos + 2],
            body[pos + 3],
        ]) as usize;
        pos += 4;
        if pos + blen > body.len() {
            return Err(NetError::ProtocolViolation(
                "Code body truncated before item payload",
            ));
        }
        let bytes = body[pos..pos + blen].to_vec();
        pos += blen;
        out.push((kind, Hash(hash_buf), bytes));
    }
    if pos != body.len() {
        return Err(NetError::ProtocolViolation("Code body has trailing bytes"));
    }
    Ok(out)
}

/// Maximum frame body size — guards against pathological inputs from a
/// hostile peer. 16 MiB is plenty for any value we'd encode in v1.
const MAX_FRAME_BYTES: u32 = 16 * 1024 * 1024;

#[derive(Debug)]
pub enum NetError {
    Io(io::Error),
    Wire(WireError),
    FrameTooLarge(u32),
    BadKind(u8),
    ProtocolViolation(&'static str),
    /// Same as `ProtocolViolation` but with a dynamically-built message
    /// (so we can include the bad byte / count in the text).
    ProtocolViolationOwned(String),
    /// Server invoked a closure whose code hash isn't in its JIT code
    /// table. The future `NeedCode` round-trip removes this error.
    UnknownCode(Hash),
    /// Server's `NeedCode` exchange exceeded the depth limit — likely
    /// a bug (a cycle in dependency resolution, or a client that
    /// doesn't hold the requested hash).
    CodeFetchDepthExceeded,
    /// The client got a `NeedCode` request asking for a hash it doesn't
    /// hold in its `KnowledgeBase`.
    MissingFromKnowledgeBase(Hash),
    /// Installing fetched code into the running JIT failed.
    InstallFailed(String),
    /// The peer closed the connection cleanly between frames — no
    /// in-progress message was interrupted. Servers should treat
    /// this as a normal shutdown signal and exit the serve loop
    /// without logging. Clients that hit this on a cached
    /// connection should re-open transparently.
    ConnectionClosed,
    /// An atom GET or SWAP referenced a cell that does not exist on the
    /// owning node. Maps to a `Failure` value at the language boundary.
    AtomAbsent,
}

impl NetError {
    #[allow(non_snake_case)]
    pub fn ProtocolViolation_owned(msg: String) -> Self {
        NetError::ProtocolViolationOwned(msg)
    }
}

impl core::fmt::Display for NetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NetError::Io(e) => write!(f, "io error: {}", e),
            NetError::Wire(e) => write!(f, "wire error: {}", e),
            NetError::FrameTooLarge(n) => write!(f, "frame too large: {} bytes", n),
            NetError::BadKind(b) => write!(f, "bad message kind: {}", b),
            NetError::ProtocolViolation(msg) => write!(f, "protocol violation: {}", msg),
            NetError::ProtocolViolationOwned(msg) => write!(f, "protocol violation: {}", msg),
            NetError::UnknownCode(h) => write!(f, "closure code hash {} not JIT'd locally", h),
            NetError::CodeFetchDepthExceeded => {
                write!(f, "code-fetch depth exceeded (likely client/server mismatch or cycle)")
            }
            NetError::MissingFromKnowledgeBase(h) => {
                write!(f, "code hash {} not present in local knowledge base", h)
            }
            NetError::InstallFailed(msg) => write!(f, "install of fetched code failed: {}", msg),
            NetError::ConnectionClosed => write!(f, "peer closed connection"),
            NetError::AtomAbsent => write!(f, "atom cell does not exist on the owning node"),
        }
    }
}

impl std::error::Error for NetError {}

impl From<io::Error> for NetError {
    fn from(e: io::Error) -> Self {
        NetError::Io(e)
    }
}

impl From<WireError> for NetError {
    fn from(e: WireError) -> Self {
        NetError::Wire(e)
    }
}

/// Read one length-prefixed frame from `stream`. Returns the frame's
/// body (without the length prefix).
///
/// Distinguishes between a clean disconnect (peer closed the
/// connection between frames — returns `NetError::ConnectionClosed`)
/// and a torn connection mid-frame (returns the underlying io error).
/// Servers should treat `ConnectionClosed` as a normal shutdown
/// signal, not as something to log.
pub fn read_frame<R: Read>(stream: &mut R) -> Result<Vec<u8>, NetError> {
    let mut len_buf = [0u8; 4];
    let mut got = 0usize;
    while got < 4 {
        match stream.read(&mut len_buf[got..]) {
            Ok(0) => {
                if got == 0 {
                    return Err(NetError::ConnectionClosed);
                }
                return Err(NetError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "peer disconnected mid-frame-length",
                )));
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(NetError::Io(e)),
        }
    }
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(NetError::FrameTooLarge(len));
    }
    let mut body = vec![0u8; len as usize];
    stream.read_exact(&mut body)?;
    Ok(body)
}

/// Write a length-prefixed frame to `stream`.
pub fn write_frame<W: Write>(stream: &mut W, body: &[u8]) -> Result<(), NetError> {
    let len: u32 = body
        .len()
        .try_into()
        .map_err(|_| NetError::FrameTooLarge(u32::MAX))?;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

// =============================================================================
// Channel — transport-agnostic frame I/O
// =============================================================================

/// A bidirectional pipe that reads and writes length-prefixed frames.
///
/// `Channel` is the abstraction that lets the `at()` protocol run over
/// any transport — TCP today, in-process pipes for tests, future
/// Unix-socket / QUIC / TLS implementations. The protocol layer
/// (`serve_one`, `at_remote_on_channel`, the NeedCode/Code handshake)
/// only talks to a `&mut dyn Channel`; the concrete transport is
/// chosen at the call site.
///
/// One `Channel` represents one end of a connected pair. Calling
/// `read_frame` blocks until the peer writes; `write_frame` ships the
/// body (along with its length prefix) to the peer.
pub trait Channel: Send {
    fn read_frame(&mut self) -> Result<Vec<u8>, NetError>;
    fn write_frame(&mut self, body: &[u8]) -> Result<(), NetError>;
}

impl Channel for TcpStream {
    fn read_frame(&mut self) -> Result<Vec<u8>, NetError> {
        read_frame(self)
    }
    fn write_frame(&mut self, body: &[u8]) -> Result<(), NetError> {
        write_frame(self, body)
    }
}

/// In-process Channel backed by a pair of `mpsc` queues.
///
/// `InProcessChannel::pair()` returns `(client, server)` where each
/// side's `write_frame` is the other side's `read_frame`. Drop either
/// side and the other side's reads return `BrokenPipe`.
///
/// Frames are passed by value (no shared memory, no length prefix —
/// the `Vec<u8>` body is the unit). The format on the wire is
/// irrelevant for in-process; only the byte content matters.
pub struct InProcessChannel {
    /// Sends frame bodies to the peer.
    tx: std::sync::mpsc::Sender<Vec<u8>>,
    /// Receives frame bodies from the peer.
    rx: std::sync::mpsc::Receiver<Vec<u8>>,
}

impl InProcessChannel {
    /// Create a pair of connected channels. Whatever one side writes
    /// the other reads. Both sides are `Send`, so each can move to a
    /// different thread.
    pub fn pair() -> (InProcessChannel, InProcessChannel) {
        let (tx_a, rx_a) = std::sync::mpsc::channel::<Vec<u8>>();
        let (tx_b, rx_b) = std::sync::mpsc::channel::<Vec<u8>>();
        let client = InProcessChannel { tx: tx_a, rx: rx_b };
        let server = InProcessChannel { tx: tx_b, rx: rx_a };
        (client, server)
    }
}

impl Channel for InProcessChannel {
    fn read_frame(&mut self) -> Result<Vec<u8>, NetError> {
        self.rx.recv().map_err(|_| {
            NetError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "in-process channel closed",
            ))
        })
    }
    fn write_frame(&mut self, body: &[u8]) -> Result<(), NetError> {
        self.tx.send(body.to_vec()).map_err(|_| {
            NetError::Io(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "in-process channel closed",
            ))
        })
    }
}

// =============================================================================
// Server
// =============================================================================

/// Bind a TCP listener on `addr`. Returns the listener (so callers can
/// learn the OS-assigned port via `local_addr` if they bound to `:0`).
pub fn bind(addr: impl ToSocketAddrs) -> Result<TcpListener, NetError> {
    let listener = TcpListener::bind(addr)?;
    Ok(listener)
}

/// Serve one connection: read a frame, dispatch (only `Call` is
/// supported in v1), write the reply. Closes the connection after one
/// exchange — a real server would loop.
///
/// `runtime` must be the server-side runtime whose code table contains
/// the lambda's JIT'd entry point. The lambda is invoked via the same
/// indirect-call path JIT'd code uses (`ai_gc_lookup_code` → indirect
/// call), so the runtime's thread must be in a valid state.
///
/// # Safety
/// Invokes JIT'd code via raw function-pointer transmute. The lambda's
/// LLVM signature must be `unsafe extern "C" fn(*mut Thread, *const u8) -> i64`
/// — true for any zero-arg closure produced by our codegen.
pub unsafe fn serve_one(
    runtime: &Runtime,
    channel: &mut dyn Channel,
) -> Result<(), NetError> {
    let body = channel.read_frame()?;
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty frame body"));
    }
    let kind = body[0];
    match kind {
        KIND_CALL => {
            let payload = &body[1..];
            // Cache key: blake3 of the encoded Call payload (closure
            // code hash + all captures). Identical bytes → identical
            // hash → identical computation, by the content-addressed
            // property. If we've handled this exact call before, the
            // cached reply frame is the answer.
            let cache_key = Hash::of_bytes(payload);
            if let Some(cached_reply) = runtime.try_cached_result(&cache_key) {
                channel.write_frame(&cached_reply)?;
                return Ok(());
            }
            let (value, _) = unsafe { decode_value(runtime, payload)? };
            let closure_ptr = match value {
                WireValue::Heap(p) => p,
                WireValue::Int(_) => {
                    return Err(NetError::ProtocolViolation(
                        "Call payload must be a heap closure",
                    ));
                }
            };
            let result_ptr = unsafe { invoke_zero_arg_closure(runtime, closure_ptr)? };
            let mut reply = Vec::with_capacity(64);
            reply.push(KIND_RESULT);
            unsafe { encode_value(runtime, WireValue::Heap(result_ptr), &mut reply)? };
            runtime.store_cached_result(cache_key, reply.clone());
            channel.write_frame(&reply)?;
            Ok(())
        }
        other => Err(NetError::BadKind(other)),
    }
}

/// Convenience: serve one connection from `listener`.
pub fn accept_one(listener: &TcpListener) -> Result<TcpStream, NetError> {
    let (stream, _) = listener.accept()?;
    Ok(stream)
}

/// Maximum NeedCode/Code rounds the server will tolerate per Call.
const MAX_SERVER_INSTALL_ROUNDS: usize = 5;

/// Serve one connection with code-fetch handshake support.
///
/// The flow:
/// 1. Read the Call frame.
/// 2. Attempt `decode_value(rt, payload)`.
///    - If it succeeds, invoke the closure and ship a Result frame.
///    - If it fails with `MissingShape(h)`, send `NeedCode([h])`,
///      read the response, install via `IncrementalJit::install`,
///      and retry decode. Loops up to `MAX_SERVER_INSTALL_ROUNDS`
///      before giving up.
///
/// # Safety
/// Like `serve_one`, this invokes JIT'd code via raw transmute. The
/// runtime and JIT must outlive any spawned threads. Caller is also
/// responsible for ensuring no concurrent mutation of the runtime
/// while this is running (single-threaded v1 is fine).
pub unsafe fn serve_with_install<'ctx>(
    rt: &mut Runtime,
    jit: &mut IncrementalJit<'ctx>,
    channel: &mut dyn Channel,
) -> Result<(), NetError> {
    let body = channel.read_frame()?;
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty frame body"));
    }
    if body[0] != KIND_CALL {
        return Err(NetError::BadKind(body[0]));
    }
    let mut payload: Vec<u8> = body[1..].to_vec();

    // Retry-on-MissingShape loop.
    for round in 0..=MAX_SERVER_INSTALL_ROUNDS {
        let result = unsafe { decode_value(rt, &payload) };
        match result {
            Ok((value, _)) => {
                let closure_ptr = match value {
                    WireValue::Heap(p) => p,
                    WireValue::Int(_) => {
                        return Err(NetError::ProtocolViolation(
                            "Call payload must be a heap closure",
                        ));
                    }
                };
                let result_ptr = unsafe { invoke_zero_arg_closure(rt, closure_ptr)? };
                let mut reply = Vec::with_capacity(64);
                reply.push(KIND_RESULT);
                unsafe { encode_value(rt, WireValue::Heap(result_ptr), &mut reply)? };
                channel.write_frame(&reply)?;
                return Ok(());
            }
            Err(WireError::MissingShape(missing)) => {
                if round == MAX_SERVER_INSTALL_ROUNDS {
                    return Err(NetError::CodeFetchDepthExceeded);
                }
                // Ask client for the missing code.
                let req = encode_need_code(&[missing]);
                channel.write_frame(&req)?;
                // Wait for Code response.
                let resp = channel.read_frame()?;
                if resp.is_empty() {
                    return Err(NetError::ProtocolViolation(
                        "empty response to NeedCode",
                    ));
                }
                if resp[0] != KIND_CODE {
                    return Err(NetError::BadKind(resp[0]));
                }
                let items = decode_code(&resp)?;
                jit.install(rt, items)
                    .map_err(|e| NetError::InstallFailed(format!("{}", e)))?;
                // Re-attempt decode in next iteration.
                let _ = &mut payload; // unchanged; the same bytes try again
            }
            Err(other) => return Err(NetError::Wire(other)),
        }
    }
    Err(NetError::CodeFetchDepthExceeded)
}

/// Spawn a server thread that handles one connection then exits. Used
/// by tests; production code would `accept` in a loop.
///
/// # Safety
/// The runtime must outlive the spawned thread. Caller is responsible
/// for joining the handle.
pub unsafe fn serve_one_in_thread(
    runtime: Arc<RuntimeHandle>,
    listener: TcpListener,
) -> std::thread::JoinHandle<Result<(), NetError>> {
    std::thread::spawn(move || -> Result<(), NetError> {
        let mut stream = accept_one(&listener)?;
        // SAFETY: caller's contract.
        unsafe { serve_one(&runtime.0, &mut stream) }
    })
}

/// Spawn a server that handles exactly `n` connections (one
/// closure-invocation each) and then exits cleanly. Tests that issue a
/// known number of `at()` calls can join on the returned handle and
/// see any per-connection error.
///
/// # Safety
/// The runtime must outlive the spawned thread. Caller joins the handle.
pub unsafe fn serve_n_in_thread(
    runtime: Arc<RuntimeHandle>,
    listener: TcpListener,
    n: usize,
) -> std::thread::JoinHandle<Result<(), NetError>> {
    std::thread::spawn(move || -> Result<(), NetError> {
        for _ in 0..n {
            let mut stream = accept_one(&listener)?;
            // SAFETY: caller's contract.
            unsafe { serve_one(&runtime.0, &mut stream)? };
        }
        Ok(())
    })
}

/// Spawn a server that loops `accept` + `serve_one` forever. The
/// thread exits when the listener errors out (e.g., process shutdown)
/// or when a `serve_one` call fails — failures are logged-and-continue
/// so a single bad connection doesn't kill the server. Returned handle
/// is detached-friendly; tests typically don't join it because the
/// process exits while it's still in `accept`.
///
/// # Safety
/// The runtime must outlive the spawned thread.
pub unsafe fn serve_forever_in_thread(
    runtime: Arc<RuntimeHandle>,
    listener: TcpListener,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        loop {
            let mut stream = match accept_one(&listener) {
                Ok(s) => s,
                Err(_) => break,
            };
            // SAFETY: caller's contract. Log + continue on errors.
            let _ = unsafe { serve_one(&runtime.0, &mut stream) };
        }
    })
}

/// Persistent-connection server: each accepted stream is reused for
/// many sequential Call/Result cycles until the client disconnects.
/// Pair with [`at_remote_on_stream`] on the client side (and a
/// connection cache keyed by addr) to amortise the TCP-handshake cost
/// across many `at()` calls to the same node.
///
/// # Safety
/// Like [`serve_forever_in_thread`]: runtime must outlive the spawned
/// thread, and JIT'd code is invoked via raw fn-ptr transmute.
pub unsafe fn serve_persistent_in_thread(
    runtime: Arc<RuntimeHandle>,
    listener: TcpListener,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        loop {
            let mut stream = match accept_one(&listener) {
                Ok(s) => s,
                Err(_) => break,
            };
            // Keep handling requests on THIS stream until the client
            // disconnects (read returns EOF / connection-reset / etc.)
            // or the protocol breaks. After that the outer loop
            // accepts the next client.
            loop {
                match unsafe { serve_one(&runtime.0, &mut stream) } {
                    Ok(()) => continue,
                    Err(_) => break,
                }
            }
        }
    })
}

/// Thread-safe wrapper around `Runtime` so it can be moved across
/// thread boundaries for the server. The runtime's internal state
/// (Heap, ThreadState, CodeTable) is already Sync; we just box the
/// thread-struct pointer location.
pub struct RuntimeHandle(pub Runtime);

unsafe impl Send for RuntimeHandle {}
unsafe impl Sync for RuntimeHandle {}

// =============================================================================
// Client
// =============================================================================

/// Maximum number of `NeedCode` rounds we'll tolerate before giving up.
/// In a healthy session the server asks once (for the top-level closure
/// hash); we receive a bundle containing every transitive dep and we're
/// done. A second round is plausible if the server's dependency
/// inference is incomplete; beyond that, we're almost certainly in a
/// pathological state.
const MAX_CODE_FETCH_ROUNDS: usize = 5;

/// Connect to `addr`, ship the closure pointed to by `closure_ptr`,
/// handle any `NeedCode` round-trips from the server, wait for the
/// final `Result`, return a pointer to the decoded result value in
/// `runtime`'s heap.
///
/// The lambda's return type is unconstrained: under the uniform
/// closure ABI every lifted lambda returns a pointer, so the wire
/// always carries a `WireValue::Heap`. For `fn() -> Int` thunks that
/// pointer lands as a fresh `BoxedInt` on the receiver's heap; for
/// struct / enum returns it lands as the corresponding heap object.
///
/// `knowledge_base` provides the canonical bytes for any def / lambda
/// the server requests via `NeedCode`. If a request mentions a hash
/// that isn't in `knowledge_base`, returns
/// `NetError::MissingFromKnowledgeBase`.
///
/// # Safety
/// `closure_ptr` must be a live closure in `runtime`'s heap.
pub unsafe fn at_remote(
    runtime: &Runtime,
    knowledge_base: &KnowledgeBase,
    addr: impl ToSocketAddrs,
    closure_ptr: *const u8,
) -> Result<*const u8, NetError> {
    let mut stream = TcpStream::connect(addr)?;
    unsafe { at_remote_on_channel(runtime, knowledge_base, &mut stream, closure_ptr) }
}

thread_local! {
    /// Per-thread cache of open client connections keyed by destination
    /// address. Populated lazily on the first `at()` call to a given
    /// address; the entry is evicted if its request errors out (so a
    /// torn connection doesn't poison subsequent calls).
    static AT_CONN_CACHE: RefCell<HashMap<String, TcpStream>> =
        RefCell::new(HashMap::new());
}

/// Clear the thread-local at() connection cache. Useful for tests
/// that want a clean slate, or for shutdown.
pub fn clear_at_conn_cache() {
    AT_CONN_CACHE.with(|c| c.borrow_mut().clear());
}

/// `at_remote` variant that reuses (or opens) a cached TCP stream for
/// the given address. Pairs with [`serve_persistent_in_thread`] on the
/// server side. On error, evicts the cached entry so the next call
/// reopens cleanly.
///
/// # Safety
/// Same as [`at_remote`].
pub unsafe fn at_remote_cached(
    runtime: &Runtime,
    knowledge_base: &KnowledgeBase,
    addr: &str,
    closure_ptr: *const u8,
) -> Result<*const u8, NetError> {
    // Two-phase to keep the RefCell borrow short and to allow eviction
    // on error without re-borrowing inside the call path.
    let result: Result<*const u8, NetError> = AT_CONN_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let stream = match cache.entry(addr.to_owned()) {
            std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let s = TcpStream::connect(addr)?;
                // Disable Nagle to keep per-cell latency low; the
                // protocol is request/response and we never batch.
                s.set_nodelay(true).ok();
                v.insert(s)
            }
        };
        unsafe { at_remote_on_channel(runtime, knowledge_base, stream, closure_ptr) }
    });
    if result.is_err() {
        AT_CONN_CACHE.with(|c| {
            c.borrow_mut().remove(addr);
        });
    }
    result
}

/// Same as [`at_remote`] but uses an already-connected stream. Lets
/// callers cache TCP connections across many `at()` calls — the
/// connection stays open between the response and the next request,
/// the server's persistent-serve loop reads the next frame, etc.
///
/// # Safety
/// `closure_ptr` must be a live closure in `runtime`'s heap. `stream`
/// must be paired with a server in "persistent serve" mode (i.e.,
/// [`serve_persistent_in_thread`]) — a server that exits after one
/// request will close the stream after replying and the next request
/// on this stream will fail.
pub unsafe fn at_remote_on_channel(
    runtime: &Runtime,
    knowledge_base: &KnowledgeBase,
    channel: &mut dyn Channel,
    closure_ptr: *const u8,
) -> Result<*const u8, NetError> {
    // Build and send the Call frame.
    let mut body = Vec::with_capacity(1024);
    body.push(KIND_CALL);
    unsafe { encode_value(runtime, WireValue::Heap(closure_ptr), &mut body)? };
    channel.write_frame(&body)?;

    // Track which hashes we've already shipped this session so we don't
    // re-ship the same bytes if the server asks twice (defensive — a
    // good server shouldn't, but cheap to guard).
    let mut already_shipped: std::collections::HashSet<Hash> =
        std::collections::HashSet::new();

    for round in 0..=MAX_CODE_FETCH_ROUNDS {
        let reply = channel.read_frame()?;
        if reply.is_empty() {
            return Err(NetError::ProtocolViolation("empty reply body"));
        }
        let kind = reply[0];
        match kind {
            KIND_RESULT => {
                let payload = &reply[1..];
                let (value, _) = unsafe { decode_value(runtime, payload)? };
                return match value {
                    WireValue::Heap(p) => Ok(p),
                    WireValue::Int(_) => Err(NetError::ProtocolViolation(
                        "Result payload must be a heap value (lambdas return pointers \
                         under the uniform ABI; Int returns ship as BoxedInt)",
                    )),
                };
            }
            KIND_NEED_CODE => {
                if round == MAX_CODE_FETCH_ROUNDS {
                    return Err(NetError::CodeFetchDepthExceeded);
                }
                let requested = decode_need_code(&reply)?;
                // Look up each requested hash, collect transitive deps,
                // ship the lot in dependency order.
                let deps = knowledge_base
                    .collect_transitive_deps(&requested)
                    .map_err(|e| match e {
                        crate::knowledge::KbError::MissingHash(h) => {
                            NetError::MissingFromKnowledgeBase(h)
                        }
                        other => NetError::InstallFailed(format!("{}", other)),
                    })?;
                let mut items: Vec<(ItemKind, Hash, Vec<u8>)> =
                    Vec::with_capacity(deps.len());
                for h in &deps {
                    if already_shipped.contains(h) {
                        continue;
                    }
                    let (kind, bytes) = knowledge_base
                        .lookup(h)
                        .ok_or(NetError::MissingFromKnowledgeBase(*h))?;
                    items.push((*kind, *h, bytes.clone()));
                    already_shipped.insert(*h);
                }
                // Attach the C/host extern requirements the shipped code
                // calls, so the receiver can resolve them (or fail clearly
                // if the library/symbol is unavailable on that node). Keyed
                // by a synthetic hash of the name for shipment dedup.
                for (name, sig) in knowledge_base.extern_requirements(&deps) {
                    let ext_hash = Hash::of_bytes(name.as_bytes());
                    if already_shipped.contains(&ext_hash) {
                        continue;
                    }
                    let bytes = crate::codec::encode_extern(
                        &name,
                        &sig.params,
                        &sig.ret,
                        sig.library.as_deref(),
                        sig.variadic,
                    );
                    items.push((ItemKind::Extern, ext_hash, bytes));
                    already_shipped.insert(ext_hash);
                }
                let payload = encode_code(&items);
                channel.write_frame(&payload)?;
                // Loop continues — wait for next reply (another
                // NeedCode round, or the final Result).
            }
            other => return Err(NetError::BadKind(other)),
        }
    }
    Err(NetError::CodeFetchDepthExceeded)
}

// =============================================================================
// AtBinding — runtime-side metadata for constructing the Result enum
// =============================================================================

/// Runtime layout for one variant the `at(...)` builtin constructs.
///
/// Mirrors the codegen-side `VariantInfo` but holds raw addresses
/// (`*const TypeInfo`) instead of LLVM globals. Built once from the
/// resolved module + the runtime's shape metadata.
#[derive(Copy, Clone, Debug)]
pub struct AtVariantLayout {
    /// `TypeInfo` pointer to pass to `ai_gc_alloc_closure`.
    pub ti: *const TypeInfo,
    /// Byte offset of the variant-tag word (relative to object start).
    pub tag_offset: u32,
    /// The tag value (= variant index) to store.
    pub tag_value: u32,
    /// Byte offset of the payload slot, if any.
    pub payload_offset: Option<u32>,
    /// `true` if the payload is a pointer field (stored in
    /// `value_field[0]`); `false` if it's a raw Int.
    pub payload_is_pointer: bool,
}

// Raw `*const TypeInfo` is process-local; safe to share read-only.
unsafe impl Send for AtVariantLayout {}
unsafe impl Sync for AtVariantLayout {}

/// All variant layouts the `at(...)` runtime needs to build its
/// `Result<Int, Failure>` return value.
#[derive(Clone, Debug)]
pub struct AtRuntimeBinding {
    pub ok: AtVariantLayout,
    pub err: AtVariantLayout,
    pub unreachable: AtVariantLayout,
    pub crashed: AtVariantLayout,
    pub code_missing: AtVariantLayout,
    pub cancelled: AtVariantLayout,
}

unsafe impl Send for AtRuntimeBinding {}
unsafe impl Sync for AtRuntimeBinding {}

/// Build an [`AtRuntimeBinding`] from the resolver's `AtBinding` and
/// the runtime's shape-metadata maps. Returns `None` if any variant
/// hash is missing from the runtime (e.g. the runtime was constructed
/// from a different module).
pub fn build_at_runtime_binding(
    rt: &Runtime,
    rb: &ResolverAtBinding,
) -> Option<AtRuntimeBinding> {
    fn lookup(
        rt: &Runtime,
        enum_hash: Hash,
        variant_index: u32,
    ) -> Option<AtVariantLayout> {
        // Find the variant hash via shape_meta. Each `ShapeMeta::EnumVariant`
        // entry is keyed by the per-variant hash, not the enum hash;
        // scan for the matching (enum_ref, variant_index) pair.
        for (variant_hash, meta) in &rt.shape_meta {
            if let ShapeMeta::EnumVariant {
                enum_ref,
                variant_index: idx,
                tag_offset,
                payload,
            } = meta
            {
                if *enum_ref == enum_hash && *idx == variant_index {
                    let ti = rt.type_info_for(variant_hash)?;
                    return Some(AtVariantLayout {
                        ti,
                        tag_offset: *tag_offset,
                        tag_value: variant_index,
                        payload_offset: payload.map(|f| f.offset),
                        payload_is_pointer: payload.map(|f| f.is_pointer).unwrap_or(false),
                    });
                }
            }
        }
        None
    }

    Some(AtRuntimeBinding {
        ok: lookup(rt, rb.result_hash, rb.ok_variant_index)?,
        err: lookup(rt, rb.result_hash, rb.err_variant_index)?,
        unreachable: lookup(rt, rb.failure_hash, rb.unreachable_variant_index)?,
        crashed: lookup(rt, rb.failure_hash, rb.crashed_variant_index)?,
        code_missing: lookup(rt, rb.failure_hash, rb.code_missing_variant_index)?,
        cancelled: lookup(rt, rb.failure_hash, rb.cancelled_variant_index)?,
    })
}

// =============================================================================
// Thread-local Runtime pointer for the JIT-callable `at` runtime fn.
// =============================================================================

thread_local! {
    static CURRENT_RUNTIME: Cell<*const Runtime> = const { Cell::new(std::ptr::null()) };
    static CURRENT_KB: Cell<*const KnowledgeBase> = const { Cell::new(std::ptr::null()) };
    static CURRENT_AT_BINDING: Cell<*const AtRuntimeBinding> =
        const { Cell::new(std::ptr::null()) };
}

/// Install the per-thread `AtRuntimeBinding` that `ai_net_at` uses to
/// construct its `Result` return value. Must be called before any
/// JIT'd code that reaches `at(...)`.
pub fn install_current_at_binding(b: &AtRuntimeBinding) {
    CURRENT_AT_BINDING.with(|c| c.set(b as *const AtRuntimeBinding));
}

pub fn clear_current_at_binding() {
    CURRENT_AT_BINDING.with(|c| c.set(std::ptr::null()));
}

fn current_at_binding() -> Option<&'static AtRuntimeBinding> {
    let ptr = CURRENT_AT_BINDING.with(|c| c.get());
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &*ptr })
    }
}

/// Install the current Runtime so JIT-emitted calls to `core/net.at`
/// (`ai_net_at` at runtime) can find shape metadata and thread state.
///
/// Caller is responsible for ensuring the `Runtime` outlives any JIT'd
/// code that runs after this call. In practice `examples/distributed_lang.rs`
/// installs once at startup; the Runtime stays in scope for the whole
/// program.
pub fn install_current_runtime(rt: &Runtime) {
    CURRENT_RUNTIME.with(|c| c.set(rt as *const Runtime));
}

/// Clear the thread-local. Useful in tests for hygiene.
pub fn clear_current_runtime() {
    CURRENT_RUNTIME.with(|c| c.set(std::ptr::null()));
}

/// Install the current `KnowledgeBase` for JIT-driven `at` calls. The
/// KB supplies canonical bytes for any hash a remote server requests
/// via `NeedCode`. Must outlive any JIT'd `at` call.
pub fn install_current_knowledge_base(kb: &KnowledgeBase) {
    CURRENT_KB.with(|c| c.set(kb as *const KnowledgeBase));
}

pub fn clear_current_knowledge_base() {
    CURRENT_KB.with(|c| c.set(std::ptr::null()));
}

fn current_runtime() -> Option<&'static Runtime> {
    let ptr = CURRENT_RUNTIME.with(|c| c.get());
    if ptr.is_null() {
        None
    } else {
        // SAFETY: we promised in `install_current_runtime` that the
        // Runtime outlives any JIT call that follows. The reference is
        // synthesized with `'static` here but is really `'rt`; only used
        // for the duration of the runtime fn call.
        Some(unsafe { &*ptr })
    }
}

fn current_knowledge_base() -> Option<&'static KnowledgeBase> {
    let ptr = CURRENT_KB.with(|c| c.get());
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &*ptr })
    }
}

// =============================================================================
// `at` runtime function (language-level call from JIT'd code)
// =============================================================================

/// Read a 64-bit field at `offset` from the start of an ai-lang struct
/// object. Assumes the field is an `Int` (raw bytes after the header).
unsafe fn read_int_field(obj_ptr: *const u8, offset: usize) -> i64 {
    unsafe { *(obj_ptr.add(offset) as *const i64) }
}

/// The runtime implementation of the language-level `at(node, thunk)`
/// primitive. JIT'd code calls this through inkwell's
/// `add_global_mapping`.
///
/// `node_ptr` must point to a struct laid out as five consecutive Int
/// fields starting at offset `Full::SIZE` (16): `a, b, c, d, port`.
/// These five Ints encode an IPv4 address + TCP port.
///
/// `closure_ptr` is a closure (zero-arg, returns Int) — the thunk to ship.
///
/// Returns a heap pointer to the user-defined `Result` enum value:
///   `Ok(Int)` on success, `Err(Failure)` wrapping `node_ptr` on a
/// network error.
///
/// # Safety
/// All three pointers must point to live heap objects of the right
/// shape. `install_current_runtime`, `install_current_knowledge_base`,
/// and `install_current_at_binding` must all have been called on this
/// thread before any JIT'd code reaches this fn.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_net_at(
    thread: *mut Thread,
    node_ptr: *const u8,
    closure_ptr: *const u8,
) -> *const u8 {
    let rt = current_runtime()
        .expect("ai_net_at: install_current_runtime must be called before any at() in JIT");
    let kb = current_knowledge_base().expect(
        "ai_net_at: install_current_knowledge_base must be called before any at() in JIT",
    );
    let binding = current_at_binding().expect(
        "ai_net_at: install_current_at_binding must be called before any at() in JIT",
    );

    let header_size = <Full as crate::gc::ObjHeader>::SIZE;
    let (a, b, c, d, port) = unsafe {
        (
            read_int_field(node_ptr, header_size),
            read_int_field(node_ptr, header_size + 8),
            read_int_field(node_ptr, header_size + 16),
            read_int_field(node_ptr, header_size + 24),
            read_int_field(node_ptr, header_size + 32),
        )
    };

    let addr = format!("{}.{}.{}.{}:{}", a, b, c, d, port);

    match unsafe { at_remote_cached(rt, kb, &addr, closure_ptr) } {
        Ok(n) => unsafe { build_ok(thread, binding, n) },
        Err(e) => {
            eprintln!("[ai_net_at] at_remote failed: {}", e);
            let failure_variant = match &e {
                NetError::Io(_) | NetError::ConnectionClosed => &binding.unreachable,
                NetError::UnknownCode(_)
                | NetError::MissingFromKnowledgeBase(_)
                | NetError::CodeFetchDepthExceeded => &binding.code_missing,
                NetError::Wire(_)
                | NetError::FrameTooLarge(_)
                | NetError::BadKind(_)
                | NetError::ProtocolViolation(_)
                | NetError::ProtocolViolationOwned(_)
                | NetError::AtomAbsent
                | NetError::InstallFailed(_) => &binding.crashed,
            };
            unsafe { build_err(thread, binding, failure_variant, node_ptr) }
        }
    }
}

/// Allocate the variant heap object, write the tag, and (if any)
/// store the payload at the right offset.
unsafe fn alloc_variant(
    thread: *mut Thread,
    v: &AtVariantLayout,
    payload_int: Option<i64>,
    payload_ptr: Option<*const u8>,
) -> *const u8 {
    let obj = unsafe { ai_gc_alloc_closure(thread, v.ti) };
    // Tag write (u32).
    let tag_slot = unsafe { obj.add(v.tag_offset as usize) as *mut u32 };
    unsafe { *tag_slot = v.tag_value };
    // Payload write.
    if let Some(off) = v.payload_offset {
        if v.payload_is_pointer {
            let p = payload_ptr.expect("pointer payload required for this variant");
            let slot = unsafe { obj.add(off as usize) as *mut *const u8 };
            unsafe { *slot = p };
        } else {
            let n = payload_int.expect("int payload required for this variant");
            let slot = unsafe { obj.add(off as usize) as *mut i64 };
            unsafe { *slot = n };
        }
    }
    obj as *const u8
}

/// Build `Result::Ok(payload)` where `payload` is a heap pointer.
///
/// The generic `Result<T, E>` ABI says the Ok variant's payload slot
/// is always a pointer (uniform representation). For Int-returning
/// thunks the payload is a `BoxedInt`; for struct/enum returns it
/// points at the decoded heap object directly. Either way, `build_ok`
/// just drops it into the variant's payload slot.
unsafe fn build_ok(
    thread: *mut Thread,
    b: &AtRuntimeBinding,
    payload: *const u8,
) -> *const u8 {
    debug_assert!(
        b.ok.payload_is_pointer,
        "Result<T, E> must have a pointer payload on Ok — \
         monomorphic Result<Int, ...> is no longer supported by at()"
    );
    unsafe { alloc_variant(thread, &b.ok, None, Some(payload)) }
}

unsafe fn build_err(
    thread: *mut Thread,
    b: &AtRuntimeBinding,
    failure_variant: &AtVariantLayout,
    node_ptr: *const u8,
) -> *const u8 {
    // Build Failure::<variant>(node_ptr).
    let failure = unsafe { alloc_variant(thread, failure_variant, None, Some(node_ptr)) };
    // Then wrap in Result::Err(failure).
    unsafe { alloc_variant(thread, &b.err, None, Some(failure)) }
}

/// Map a transport error to the `Failure` variant the language sees.
/// Shared by `ai_net_at` and the atom ops.
fn failure_variant_for<'a>(b: &'a AtRuntimeBinding, e: &NetError) -> &'a AtVariantLayout {
    match e {
        NetError::Io(_) | NetError::ConnectionClosed => &b.unreachable,
        NetError::UnknownCode(_)
        | NetError::MissingFromKnowledgeBase(_)
        | NetError::CodeFetchDepthExceeded
        | NetError::AtomAbsent => &b.code_missing,
        NetError::Wire(_)
        | NetError::FrameTooLarge(_)
        | NetError::BadKind(_)
        | NetError::ProtocolViolation(_)
        | NetError::ProtocolViolationOwned(_)
        | NetError::InstallFailed(_) => &b.crashed,
    }
}

// =============================================================================
// Located state (distributed atom) — JIT-callable runtime fns
// =============================================================================

/// Byte offsets of the `node` and `name` pointer fields within an
/// `Atom { node: Node, name: String }` heap object. Both are pointer
/// fields, so they occupy the first two `value_field` slots right after
/// the GC header — an ABI contract with stdlib's `Atom` layout, the
/// same kind `ai_net_at` relies on for `Node`'s five Int fields.
const ATOM_NODE_OFFSET: usize = <Full as crate::gc::ObjHeader>::SIZE;
const ATOM_NAME_OFFSET: usize = <Full as crate::gc::ObjHeader>::SIZE + 8;

/// Read an atom handle: its owning node's `addr:port`, the cell key
/// (`blake3(name)`), and the raw `node` pointer (for building a
/// `Failure` on error).
///
/// # Safety
/// `atom_ptr` must be a live `Atom` heap object laid out per stdlib.
unsafe fn read_atom_handle(atom_ptr: *const u8) -> (String, Hash, *const u8) {
    let node_ptr = unsafe { *(atom_ptr.add(ATOM_NODE_OFFSET) as *const *const u8) };
    let name_ptr = unsafe { *(atom_ptr.add(ATOM_NAME_OFFSET) as *const *const u8) };
    let hs = <Full as crate::gc::ObjHeader>::SIZE;
    let (a, b, c, d, port) = unsafe {
        (
            read_int_field(node_ptr, hs),
            read_int_field(node_ptr, hs + 8),
            read_int_field(node_ptr, hs + 16),
            read_int_field(node_ptr, hs + 24),
            read_int_field(node_ptr, hs + 32),
        )
    };
    let addr = format!("{}.{}.{}.{}:{}", a, b, c, d, port);
    let key = Hash::of_bytes(unsafe { crate::ffi::heap_str_bytes(name_ptr) });
    (addr, key, node_ptr)
}

/// `atom_request_on_channel` over a cached connection keyed by address
/// (reuses the same per-thread cache as `at()`).
///
/// # Safety
/// Any closure referenced by `frame` must be live in `runtime`'s heap.
unsafe fn atom_request_cached(
    runtime: &Runtime,
    knowledge_base: &KnowledgeBase,
    addr: &str,
    frame: &[u8],
) -> Result<*const u8, NetError> {
    let result = AT_CONN_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let stream = match cache.entry(addr.to_owned()) {
            std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let s = TcpStream::connect(addr)?;
                s.set_nodelay(true).ok();
                v.insert(s)
            }
        };
        unsafe { atom_request_on_channel(runtime, knowledge_base, stream, frame) }
    });
    if result.is_err() {
        AT_CONN_CACHE.with(|c| {
            c.borrow_mut().remove(addr);
        });
    }
    result
}

/// Shared body for the four atom ops: read the handle, build the
/// request frame's payload, ship it, wrap the reply as `Result`.
///
/// # Safety
/// `atom_ptr` (and any payload pointer it encodes) must be live heap
/// objects; the three thread-locals (`runtime`, `kb`, `at_binding`)
/// must be installed, as for `ai_net_at`.
unsafe fn atom_op(
    thread: *mut Thread,
    atom_ptr: *const u8,
    kind: u8,
    payload: &[u8],
) -> *const u8 {
    let rt = current_runtime()
        .expect("ai_atom_*: install_current_runtime must be called before any atom op");
    let kb = current_knowledge_base()
        .expect("ai_atom_*: install_current_knowledge_base must be called before any atom op");
    let binding = current_at_binding()
        .expect("ai_atom_*: install_current_at_binding must be called before any atom op");
    let (addr, key, node_ptr) = unsafe { read_atom_handle(atom_ptr) };
    let frame = build_atom_frame(kind, &key, payload);
    match unsafe { atom_request_cached(rt, kb, &addr, &frame) } {
        Ok(p) => unsafe { build_ok(thread, binding, p) },
        Err(e) => {
            eprintln!("[ai_atom_op] request failed: {}", e);
            let fv = failure_variant_for(binding, &e);
            unsafe { build_err(thread, binding, fv, node_ptr) }
        }
    }
}

/// Encode a heap value into an atom request payload, or return the
/// `Crashed` failure if encoding fails (a value whose shape the
/// encoder can't walk — shouldn't happen for wire-checked types).
///
/// # Safety
/// `value_ptr` must be a live heap object in `rt`.
unsafe fn encode_atom_payload(rt: &Runtime, value_ptr: *const u8) -> Result<Vec<u8>, NetError> {
    let mut payload = Vec::with_capacity(64);
    unsafe { encode_value(rt, WireValue::Heap(value_ptr), &mut payload) }
        .map_err(NetError::Wire)?;
    Ok(payload)
}

/// `atom_deref(a)` — read the cell's current value. `Result<T, Failure>`.
///
/// # Safety
/// See [`atom_op`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atom_deref(thread: *mut Thread, atom_ptr: *const u8) -> *const u8 {
    unsafe { atom_op(thread, atom_ptr, KIND_ATOM_GET, &[]) }
}

/// `atom_set(a, v)` — overwrite the cell. Returns `v`. `Result<T, Failure>`.
///
/// # Safety
/// See [`atom_op`]; `value_ptr` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atom_set(
    thread: *mut Thread,
    atom_ptr: *const u8,
    value_ptr: *const u8,
) -> *const u8 {
    let rt = current_runtime().expect("ai_atom_set: runtime not installed");
    match unsafe { encode_atom_payload(rt, value_ptr) } {
        Ok(payload) => unsafe { atom_op(thread, atom_ptr, KIND_ATOM_SET, &payload) },
        Err(e) => unsafe { atom_err(thread, atom_ptr, &e) },
    }
}

/// `atom_init(a, v)` — set the cell only if absent. Returns the current
/// value (the init, or the pre-existing one). `Result<T, Failure>`.
///
/// # Safety
/// See [`atom_op`]; `value_ptr` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atom_init(
    thread: *mut Thread,
    atom_ptr: *const u8,
    value_ptr: *const u8,
) -> *const u8 {
    let rt = current_runtime().expect("ai_atom_init: runtime not installed");
    match unsafe { encode_atom_payload(rt, value_ptr) } {
        Ok(payload) => unsafe { atom_op(thread, atom_ptr, KIND_ATOM_INIT, &payload) },
        Err(e) => unsafe { atom_err(thread, atom_ptr, &e) },
    }
}

/// `atom_swap(a, f)` — ship the updater `f: fn(T) -> T` to the cell's
/// node, apply it under the cell lock, return the new value.
/// `Result<T, Failure>`.
///
/// # Safety
/// See [`atom_op`]; `closure_ptr` must be a live closure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atom_swap(
    thread: *mut Thread,
    atom_ptr: *const u8,
    closure_ptr: *const u8,
) -> *const u8 {
    let rt = current_runtime().expect("ai_atom_swap: runtime not installed");
    match unsafe { encode_atom_payload(rt, closure_ptr) } {
        Ok(payload) => unsafe { atom_op(thread, atom_ptr, KIND_ATOM_SWAP, &payload) },
        Err(e) => unsafe { atom_err(thread, atom_ptr, &e) },
    }
}

/// Build a `Result::Err(Failure)` for an atom op that failed before it
/// could ship (e.g. payload encoding). `Unreachable`-shaped via the
/// usual mapping.
///
/// # Safety
/// `atom_ptr` must be a live `Atom` handle; thread-locals installed.
unsafe fn atom_err(thread: *mut Thread, atom_ptr: *const u8, e: &NetError) -> *const u8 {
    let binding = current_at_binding().expect("ai_atom_*: at binding not installed");
    let node_ptr = unsafe { *(atom_ptr.add(ATOM_NODE_OFFSET) as *const *const u8) };
    let fv = failure_variant_for(binding, e);
    unsafe { build_err(thread, binding, fv, node_ptr) }
}

// =============================================================================
// Closure invocation
// =============================================================================

/// Invoke a zero-arg closure: load its `code_hash`, look up the JIT'd
/// entry point in the runtime's code table, indirect-call.
///
/// Returns the raw pointer the lambda returned. Per the uniform
/// closure ABI every lifted lambda returns a pointer — for
/// `fn() -> Int` thunks that pointer is a `BoxedInt`; for thunks
/// returning structs/enums it's a pointer to that heap object. The
/// wire encoder handles all of these via shape lookup, so the server
/// never has to know which case it's in.
///
/// # Safety
/// `closure_ptr` must be a valid heap closure. The looked-up function
/// pointer must match the `unsafe extern "C" fn(*mut Thread, *const u8) -> *mut u8`
/// signature (true for any zero-arg lambda our codegen emits).
unsafe fn invoke_zero_arg_closure(
    runtime: &Runtime,
    closure_ptr: *const u8,
) -> Result<*const u8, NetError> {
    // Look up the closure's code_hash via its header type_id rather
    // than by reading bytes at a fixed offset — when the closure has
    // pointer captures they live in `value_field` slots ahead of the
    // raw-bytes header, which shifts code_hash's absolute offset.
    // type_id → shape_hash mapping is the source of truth.
    let type_id_off = <crate::gc::Full as crate::gc::ObjHeader>::TYPE_ID_OFFSET;
    let type_id = unsafe { *(closure_ptr.add(type_id_off) as *const u16) };
    let code_hash = runtime
        .shape_by_type_id
        .get(type_id as usize)
        .copied()
        .flatten()
        .ok_or_else(|| NetError::UnknownCode(Hash([0u8; 32])))?;
    let fn_ptr = runtime
        .code_table
        .lookup(&code_hash)
        .ok_or(NetError::UnknownCode(code_hash))?;
    let lambda: unsafe extern "C" fn(*mut Thread, *const u8) -> *mut u8 =
        unsafe { core::mem::transmute(fn_ptr) };
    let ret_ptr = unsafe { lambda(runtime.thread_ptr(), closure_ptr) };
    Ok(ret_ptr as *const u8)
}

/// Invoke a one-argument closure (e.g. an `atom_swap` updater
/// `fn(T) -> T`). Under the uniform closure ABI every lifted lambda is
/// `unsafe extern "C" fn(*mut Thread, *const Closure, *const u8) -> *mut u8`
/// for a single parameter — the closure-env pointer first, then the
/// argument. `arg_ptr` is the cell's current value (a heap pointer; for
/// an `Int` cell it is a `BoxedInt`, which the lambda body unboxes).
///
/// # Safety
/// `closure_ptr` and `arg_ptr` must be live heap objects in `runtime`'s
/// heap, and the closure's LLVM signature must match the transmute.
unsafe fn invoke_one_arg_closure(
    runtime: &Runtime,
    closure_ptr: *const u8,
    arg_ptr: *const u8,
) -> Result<*const u8, NetError> {
    let type_id_off = <crate::gc::Full as crate::gc::ObjHeader>::TYPE_ID_OFFSET;
    let type_id = unsafe { *(closure_ptr.add(type_id_off) as *const u16) };
    let code_hash = runtime
        .shape_by_type_id
        .get(type_id as usize)
        .copied()
        .flatten()
        .ok_or_else(|| NetError::UnknownCode(Hash([0u8; 32])))?;
    let fn_ptr = runtime
        .code_table
        .lookup(&code_hash)
        .ok_or(NetError::UnknownCode(code_hash))?;
    let lambda: unsafe extern "C" fn(*mut Thread, *const u8, *const u8) -> *mut u8 =
        unsafe { core::mem::transmute(fn_ptr) };
    let ret_ptr = unsafe { lambda(runtime.thread_ptr(), closure_ptr, arg_ptr) };
    Ok(ret_ptr as *const u8)
}

// =============================================================================
// Located state (distributed atom) — server side
// =============================================================================

/// Pull the 32-byte cell key out of an atom request frame body
/// (`body[1..33]`), validating the length.
fn cell_key_from_frame(body: &[u8]) -> Result<Hash, NetError> {
    if body.len() < 1 + CELL_KEY_LEN {
        return Err(NetError::ProtocolViolation(
            "atom frame too short for cell key",
        ));
    }
    let mut key = [0u8; CELL_KEY_LEN];
    key.copy_from_slice(&body[1..1 + CELL_KEY_LEN]);
    Ok(Hash(key))
}

/// Apply a swap to cell `key`: decode `closure_payload`, apply it to
/// the cell's current value, store and return the new value's wire
/// bytes. Returns `Ok(None)` if the cell is absent. The decode of the
/// closure must succeed against the runtime's current shapes (no
/// install handshake here); callers needing code-fetch use
/// [`serve_atom_with_install`].
///
/// # Safety
/// Invokes JIT'd code via raw transmute; see [`invoke_one_arg_closure`].
/// Upper bound on CAS retries before we give up. A correct, lightly
/// contended cell converges in one or two tries; hitting this many means
/// pathological contention (or a bug), and erroring beats spinning
/// forever holding a connection.
const MAX_SWAP_RETRIES: usize = 10_000;

unsafe fn apply_swap_locally(
    rt: &Runtime,
    key: &Hash,
    closure_payload: &[u8],
) -> Result<Option<Vec<u8>>, NetError> {
    // Lock-free CAS loop, the faithful `swap!`: snapshot (generation,
    // value), run the shipped updater with NO lock held, then commit
    // only if the generation is unchanged. A concurrent writer that
    // moved the cell forces a retry — the updater is pure by contract,
    // so re-running it is sound. The cell lock is taken only for the
    // compare-and-set inside `cell_compare_and_set`, never across the
    // (arbitrary, possibly slow or re-entrant) updater.
    for _ in 0..MAX_SWAP_RETRIES {
        let (generation, current_bytes) = match rt.cell_get_versioned(key) {
            Some(snap) => snap,
            None => return Ok(None),
        };
        // Decode the updater and the current value fresh each attempt
        // (cheap, and avoids holding wire-decoded heap pointers across a
        // retry where a GC could move them).
        let (value, _) = unsafe { decode_value(rt, closure_payload)? };
        let closure_ptr = match value {
            WireValue::Heap(p) => p,
            WireValue::Int(_) => {
                return Err(NetError::ProtocolViolation(
                    "atom swap payload must be a heap closure",
                ));
            }
        };
        let (cur_val, _) = unsafe { decode_value(rt, &current_bytes)? };
        let cur_ptr = match cur_val {
            WireValue::Heap(p) => p,
            WireValue::Int(_) => {
                return Err(NetError::ProtocolViolation(
                    "atom cell value must be a heap value",
                ));
            }
        };
        let new_ptr = unsafe { invoke_one_arg_closure(rt, closure_ptr, cur_ptr)? };
        let mut new_bytes = Vec::with_capacity(64);
        unsafe { encode_value(rt, WireValue::Heap(new_ptr), &mut new_bytes)? };
        if rt.cell_compare_and_set(key, generation, new_bytes.clone()) {
            return Ok(Some(new_bytes));
        }
        // Lost the race; another swap committed first. Re-snapshot.
    }
    Err(NetError::ProtocolViolation(
        "atom swap exceeded retry limit under contention",
    ))
}

/// Serve one located-state request, requiring any swap closure's code
/// to already be present in the runtime (no code-fetch handshake). The
/// `&Runtime` analogue of [`serve_atom_with_install`], suitable when
/// client and server share a code table.
///
/// # Safety
/// SWAP invokes JIT'd code via raw transmute. The runtime must outlive
/// the call.
pub unsafe fn serve_atom_one(
    rt: &Runtime,
    channel: &mut dyn Channel,
) -> Result<(), NetError> {
    let body = channel.read_frame()?;
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty atom frame body"));
    }
    let kind = body[0];
    let key = cell_key_from_frame(&body)?;
    let payload_start = 1 + CELL_KEY_LEN;

    let value_bytes: Option<Vec<u8>> = match kind {
        KIND_ATOM_GET => rt.cell_get(&key),
        KIND_ATOM_SET => {
            let v = body[payload_start..].to_vec();
            rt.cell_set(key, v.clone());
            Some(v)
        }
        KIND_ATOM_INIT => {
            let init = body[payload_start..].to_vec();
            Some(rt.cell_init(key, init))
        }
        KIND_ATOM_SWAP => unsafe { apply_swap_locally(rt, &key, &body[payload_start..])? },
        other => return Err(NetError::BadKind(other)),
    };

    match value_bytes {
        Some(bytes) => {
            let mut reply = Vec::with_capacity(1 + bytes.len());
            reply.push(KIND_RESULT);
            reply.extend_from_slice(&bytes);
            channel.write_frame(&reply)?;
        }
        None => channel.write_frame(&[KIND_ATOM_ABSENT])?,
    }
    Ok(())
}

/// Serve exactly one located-state request on `channel`.
///
/// GET / SET / INIT are pure byte operations on the runtime's cell
/// store and never touch the heap or JIT. SWAP decodes the shipped
/// updater closure (running the NeedCode/Code handshake if the server
/// lacks its code), applies it to the cell's current value under the
/// cell-store lock, and stores the result — the read-apply-write is the
/// atom's compare-and-set point.
///
/// # Safety
/// Like [`serve_with_install`], SWAP invokes JIT'd code via raw
/// transmute and may mutate `jit`. The runtime and JIT must outlive the
/// call; no other thread may mutate them concurrently.
pub unsafe fn serve_atom_with_install<'ctx>(
    rt: &mut Runtime,
    jit: &mut IncrementalJit<'ctx>,
    channel: &mut dyn Channel,
) -> Result<(), NetError> {
    let body = channel.read_frame()?;
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty atom frame body"));
    }
    let kind = body[0];
    let key = cell_key_from_frame(&body)?;
    let payload_start = 1 + CELL_KEY_LEN;

    match kind {
        KIND_ATOM_GET => {
            match rt.cell_get(&key) {
                Some(value_bytes) => {
                    let mut reply = Vec::with_capacity(1 + value_bytes.len());
                    reply.push(KIND_RESULT);
                    reply.extend_from_slice(&value_bytes);
                    channel.write_frame(&reply)?;
                }
                None => channel.write_frame(&[KIND_ATOM_ABSENT])?,
            }
            Ok(())
        }
        KIND_ATOM_SET => {
            let value_bytes = body[payload_start..].to_vec();
            rt.cell_set(key, value_bytes.clone());
            let mut reply = Vec::with_capacity(1 + value_bytes.len());
            reply.push(KIND_RESULT);
            reply.extend_from_slice(&value_bytes);
            channel.write_frame(&reply)?;
            Ok(())
        }
        KIND_ATOM_INIT => {
            let init_bytes = body[payload_start..].to_vec();
            let current = rt.cell_init(key, init_bytes);
            let mut reply = Vec::with_capacity(1 + current.len());
            reply.push(KIND_RESULT);
            reply.extend_from_slice(&current);
            channel.write_frame(&reply)?;
            Ok(())
        }
        KIND_ATOM_SWAP => {
            if rt.cell_get(&key).is_none() {
                channel.write_frame(&[KIND_ATOM_ABSENT])?;
                return Ok(());
            }
            let closure_payload: Vec<u8> = body[payload_start..].to_vec();

            // First, ensure the updater's code is present, running the
            // NeedCode/Code handshake on MissingShape. This is purely
            // about installing code; the actual mutation is the CAS loop
            // in `apply_swap_locally` below (so the install path gets the
            // same lock-free read-apply-commit semantics as the no-install
            // path — no inline racy read-modify-write).
            for round in 0..=MAX_SERVER_INSTALL_ROUNDS {
                match unsafe { decode_value(rt, &closure_payload) } {
                    Ok(_) => break,
                    Err(WireError::MissingShape(missing)) => {
                        if round == MAX_SERVER_INSTALL_ROUNDS {
                            return Err(NetError::CodeFetchDepthExceeded);
                        }
                        let req = encode_need_code(&[missing]);
                        channel.write_frame(&req)?;
                        let resp = channel.read_frame()?;
                        if resp.is_empty() || resp[0] != KIND_CODE {
                            return Err(NetError::ProtocolViolation(
                                "expected Code in response to atom-swap NeedCode",
                            ));
                        }
                        let items = decode_code(&resp)?;
                        jit.install(rt, items)
                            .map_err(|e| NetError::InstallFailed(format!("{}", e)))?;
                    }
                    Err(other) => return Err(NetError::Wire(other)),
                }
            }

            // Code is present — perform the atomic CAS swap.
            match unsafe { apply_swap_locally(rt, &key, &closure_payload)? } {
                Some(new_bytes) => {
                    let mut reply = Vec::with_capacity(1 + new_bytes.len());
                    reply.push(KIND_RESULT);
                    reply.extend_from_slice(&new_bytes);
                    channel.write_frame(&reply)?;
                }
                None => channel.write_frame(&[KIND_ATOM_ABSENT])?,
            }
            Ok(())
        }
        other => Err(NetError::BadKind(other)),
    }
}

// =============================================================================
// Located state (distributed atom) — client side
// =============================================================================

/// Send a pre-built atom request frame on `channel`, answer any
/// NeedCode round-trips (shipping closure code from `knowledge_base`,
/// exactly as [`at_remote_on_channel`] does), and return a pointer to
/// the decoded reply value in `runtime`'s heap. An `ABSENT` reply
/// becomes [`NetError::AtomAbsent`].
///
/// # Safety
/// Any closure referenced by `frame` must be live in `runtime`'s heap,
/// and `knowledge_base` must hold the canonical bytes for its code.
pub unsafe fn atom_request_on_channel(
    runtime: &Runtime,
    knowledge_base: &KnowledgeBase,
    channel: &mut dyn Channel,
    frame: &[u8],
) -> Result<*const u8, NetError> {
    channel.write_frame(frame)?;

    let mut already_shipped: std::collections::HashSet<Hash> =
        std::collections::HashSet::new();

    for round in 0..=MAX_CODE_FETCH_ROUNDS {
        let reply = channel.read_frame()?;
        if reply.is_empty() {
            return Err(NetError::ProtocolViolation("empty atom reply body"));
        }
        match reply[0] {
            KIND_RESULT => {
                let (value, _) = unsafe { decode_value(runtime, &reply[1..])? };
                return match value {
                    WireValue::Heap(p) => Ok(p),
                    WireValue::Int(_) => Err(NetError::ProtocolViolation(
                        "atom reply payload must be a heap value",
                    )),
                };
            }
            KIND_ATOM_ABSENT => return Err(NetError::AtomAbsent),
            KIND_NEED_CODE => {
                if round == MAX_CODE_FETCH_ROUNDS {
                    return Err(NetError::CodeFetchDepthExceeded);
                }
                let requested = decode_need_code(&reply)?;
                let deps = knowledge_base
                    .collect_transitive_deps(&requested)
                    .map_err(|e| match e {
                        crate::knowledge::KbError::MissingHash(h) => {
                            NetError::MissingFromKnowledgeBase(h)
                        }
                        other => NetError::InstallFailed(format!("{}", other)),
                    })?;
                let mut items: Vec<(ItemKind, Hash, Vec<u8>)> = Vec::with_capacity(deps.len());
                for h in &deps {
                    if already_shipped.contains(h) {
                        continue;
                    }
                    let (k, bytes) = knowledge_base
                        .lookup(h)
                        .ok_or(NetError::MissingFromKnowledgeBase(*h))?;
                    items.push((*k, *h, bytes.clone()));
                    already_shipped.insert(*h);
                }
                for (name, sig) in knowledge_base.extern_requirements(&deps) {
                    let ext_hash = Hash::of_bytes(name.as_bytes());
                    if already_shipped.contains(&ext_hash) {
                        continue;
                    }
                    let bytes = crate::codec::encode_extern(
                        &name,
                        &sig.params,
                        &sig.ret,
                        sig.library.as_deref(),
                        sig.variadic,
                    );
                    items.push((ItemKind::Extern, ext_hash, bytes));
                    already_shipped.insert(ext_hash);
                }
                channel.write_frame(&encode_code(&items))?;
            }
            other => return Err(NetError::BadKind(other)),
        }
    }
    Err(NetError::CodeFetchDepthExceeded)
}

/// Build an atom request frame: `[kind][cell_key][payload]`.
pub fn build_atom_frame(kind: u8, key: &Hash, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(1 + CELL_KEY_LEN + payload.len());
    frame.push(kind);
    frame.extend_from_slice(key.as_bytes());
    frame.extend_from_slice(payload);
    frame
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CompiledModule, Jit, init_native_target};
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use inkwell::context::Context;
    use std::sync::Once;

    static INIT: Once = Once::new();
    fn init() {
        INIT.call_once(|| {
            init_native_target().expect("init native target");
        });
    }

    #[test]
    fn frame_roundtrip() {
        let mut buf: Vec<u8> = Vec::new();
        let body = b"hello world".to_vec();
        write_frame(&mut buf, &body).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let read = read_frame(&mut cursor).unwrap();
        assert_eq!(read, body);
    }

    #[test]
    fn frame_too_large_errors() {
        // Inject a 4-byte length that exceeds MAX_FRAME_BYTES.
        let mut buf = Vec::new();
        buf.extend_from_slice(&(MAX_FRAME_BYTES + 1).to_be_bytes());
        let mut cursor = std::io::Cursor::new(buf);
        match read_frame(&mut cursor) {
            Err(NetError::FrameTooLarge(n)) => assert_eq!(n, MAX_FRAME_BYTES + 1),
            other => panic!("expected FrameTooLarge, got {:?}", other),
        }
    }

    /// Build a Runtime + Jit from `src`, used by both sides of the
    /// end-to-end TCP test. Both must come from the *same source* so
    /// their code tables agree on lambda code_hashes.
    fn make_compiled_runtime<'ctx>(
        ctx: &'ctx Context,
        src: &str,
    ) -> (Runtime, Jit<'ctx>, std::collections::HashMap<String, Hash>) {
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let names: std::collections::HashMap<String, Hash> =
            r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
        let cm = CompiledModule::build(ctx, &r).unwrap();
        let rt = Runtime::new_with_metadata(
            cm.closure_type_infos.clone(),
            cm.shape_registry.clone(),
            cm.shape_meta.clone(),
            cm.shape_by_type_id.clone(),
        );
        let jit = Jit::new(cm, &rt).unwrap();
        (rt, jit, names)
    }

    /// The headline test: client constructs a closure capturing `n`,
    /// ships it via TCP to a server thread, server invokes it and
    /// returns the result.
    ///
    /// Source defines `make_thunk(n) -> fn() -> Int = || n + 100`.
    /// Client builds `thunk = make_thunk(42)`. Server runs it. Expect 142.
    #[test]
    fn tcp_roundtrip_closure_invocation() {
        init();
        let src = "
            def make_thunk(n: Int) -> fn() -> Int = || n + 100
        ";

        // Server side: own Context, Runtime, Jit. Stays on a thread.
        // Use Arc<RuntimeHandle> so we can hand it to the spawned thread
        // while keeping the JIT (which holds module refs) on the main
        // thread.
        //
        // Subtle: Jit's lifetime is tied to its Context. The server
        // thread only needs Runtime + access to the JIT'd function
        // pointers (already published into the Runtime's code_table
        // at Jit::new time). So we can let the server thread hold the
        // Runtime alone; the Jit and Context live on the main thread
        // for the duration of the test.
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit; // keep alive until end of test (drops EE).

        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();

        // Move the Runtime into the server thread via an Arc handle.
        let server_handle = Arc::new(RuntimeHandle(server_rt));
        let handle = unsafe { serve_one_in_thread(server_handle, listener) };

        // Client side: separate Context, Runtime, Jit — same source.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);

        // Build `make_thunk(42)` on the client.
        let make_thunk = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["make_thunk"]),
                )
                .unwrap()
        };
        let closure = unsafe { make_thunk.call(client_rt.thread_ptr(), 42) };

        // Ship to server, get the result back. An empty KB suffices here:
        // server and client compile the same source, so the server never
        // needs to request code.
        let kb = KnowledgeBase::new();
        let result_ptr = unsafe {
            at_remote(&client_rt, &kb, server_addr, closure as *const u8).unwrap()
        };
        let result = unsafe { crate::runtime::ai_gc_unbox_int(result_ptr) };
        assert_eq!(result, 142);

        // Make sure the server thread finished cleanly.
        let server_result = handle.join().unwrap();
        server_result.expect("server thread errored");
    }

    /// Server returns 0 from a no-capture closure shipped by the client.
    #[test]
    fn tcp_roundtrip_zero_capture_closure() {
        init();
        let src = "
            def make_zero() -> fn() -> Int = || 0
        ";
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;

        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let handle = unsafe {
            serve_one_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);

        let make_zero = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["make_zero"]),
                )
                .unwrap()
        };
        let closure = unsafe { make_zero.call(client_rt.thread_ptr()) };
        let kb = KnowledgeBase::new();
        let result_ptr = unsafe {
            at_remote(&client_rt, &kb, server_addr, closure as *const u8).unwrap()
        };
        let result = unsafe { crate::runtime::ai_gc_unbox_int(result_ptr) };
        assert_eq!(result, 0);
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn need_code_frame_roundtrip() {
        let hashes = vec![Hash([1; 32]), Hash([2; 32]), Hash([3; 32])];
        let buf = encode_need_code(&hashes);
        let decoded = decode_need_code(&buf).unwrap();
        assert_eq!(decoded, hashes);
    }

    #[test]
    fn code_frame_roundtrip() {
        let items = vec![
            (ItemKind::Def, Hash([0xaa; 32]), b"hello".to_vec()),
            (ItemKind::Lambda, Hash([0xbb; 32]), b"world!".to_vec()),
        ];
        let buf = encode_code(&items);
        let decoded = decode_code(&buf).unwrap();
        assert_eq!(decoded, items);
    }

    /// `at()` round-trip over `InProcessChannel` — no TCP, no port
    /// binding, no listener accept. Proves the Channel abstraction:
    /// the same protocol works over any pipe that supports framed
    /// I/O. The runtime/JIT setup is identical to the TCP test
    /// (`tcp_roundtrip_closure_invocation`); only the transport
    /// changes.
    #[test]
    fn in_process_at_roundtrip() {
        init();
        let src = "
            def work(x: Int) -> Int = x * x + 7
            def make_thunk(x: Int) -> fn() -> Int = || work(x)
        ";

        // Server side: full source so it can run the closure.
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;

        // Make the in-process channel pair BEFORE moving the runtime
        // into the server thread. `RuntimeHandle` provides the
        // `Send` shim required to ship the runtime across threads.
        let (mut client_chan, mut server_chan) = InProcessChannel::pair();
        let server_runtime = Arc::new(RuntimeHandle(server_rt));

        let server_runtime_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || -> Result<(), NetError> {
            unsafe { serve_one(&server_runtime_for_thread.0, &mut server_chan) }
        });

        // Client side: build a closure, ship it over the in-process
        // channel, recover the Int return.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);

        let make_thunk = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["make_thunk"]),
                )
                .unwrap()
        };
        let closure = unsafe { make_thunk.call(client_rt.thread_ptr(), 9) };
        let kb = KnowledgeBase::new();
        let result_ptr = unsafe {
            at_remote_on_channel(&client_rt, &kb, &mut client_chan, closure as *const u8)
                .unwrap()
        };
        let result = unsafe { crate::runtime::ai_gc_unbox_int(result_ptr) };
        assert_eq!(result, 9 * 9 + 7);

        server_handle.join().unwrap().expect("server ok");
    }

    /// Distributed atom round-trip over an in-process channel: a counter
    /// cell lives on the server; the client `set`s it to 0, ships an
    /// increment closure via `swap`, then `get`s the result. Proves the
    /// located-state core: a `Runtime` holds mutable cell bytes across
    /// calls, and `swap` ships behavior that runs against state on the
    /// owning node (server-side scripting, no second language).
    #[test]
    fn in_process_atom_set_swap_get() {
        init();
        // The increment updater is shipped from the client and run on
        // the server. Both compile the same source so the server already
        // holds the closure's code (no install handshake needed here).
        let src = "def mk_inc() -> fn(Int) -> Int = |x: Int| x + 1";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let (mut client_chan, mut server_chan) = InProcessChannel::pair();
        let server_runtime = Arc::new(RuntimeHandle(server_rt));

        // Server handles exactly three sequential requests: set, swap, get.
        let server_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || -> Result<(), NetError> {
            for _ in 0..3 {
                unsafe { serve_atom_one(&server_for_thread.0, &mut server_chan)? };
            }
            Ok(())
        });

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        let kb = KnowledgeBase::new();
        let key = Hash::of_bytes(b"counter");

        // set counter = 0 (boxed Int on the wire).
        let zero = unsafe { crate::runtime::ai_gc_box_int(client_rt.thread_ptr(), 0) };
        let mut zero_enc = Vec::new();
        unsafe {
            encode_value(&client_rt, WireValue::Heap(zero as *const u8), &mut zero_enc).unwrap()
        };
        let set_frame = build_atom_frame(KIND_ATOM_SET, &key, &zero_enc);
        let after_set = unsafe {
            atom_request_on_channel(&client_rt, &kb, &mut client_chan, &set_frame).unwrap()
        };
        assert_eq!(unsafe { crate::runtime::ai_gc_unbox_int(after_set) }, 0);

        // swap with the increment closure.
        let mk_inc = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["mk_inc"]),
                )
                .unwrap()
        };
        let inc_closure = unsafe { mk_inc.call(client_rt.thread_ptr()) };
        let mut closure_enc = Vec::new();
        unsafe {
            encode_value(
                &client_rt,
                WireValue::Heap(inc_closure as *const u8),
                &mut closure_enc,
            )
            .unwrap()
        };
        let swap_frame = build_atom_frame(KIND_ATOM_SWAP, &key, &closure_enc);
        let after_swap = unsafe {
            atom_request_on_channel(&client_rt, &kb, &mut client_chan, &swap_frame).unwrap()
        };
        assert_eq!(unsafe { crate::runtime::ai_gc_unbox_int(after_swap) }, 1);

        // get confirms the mutation persisted on the server.
        let get_frame = build_atom_frame(KIND_ATOM_GET, &key, &[]);
        let got = unsafe {
            atom_request_on_channel(&client_rt, &kb, &mut client_chan, &get_frame).unwrap()
        };
        assert_eq!(unsafe { crate::runtime::ai_gc_unbox_int(got) }, 1);

        server_handle.join().unwrap().expect("server ok");
    }

    /// A distributed atom whose owning node has a durable cell store
    /// survives a server "restart": the value written before the
    /// restart is readable after a fresh Runtime reloads the same
    /// directory. Disk + atoms + distribution compose. Run at the
    /// Rust-API level (in-process channel) so the two server lifetimes
    /// are explicit.
    #[test]
    fn durable_atom_survives_server_restart() {
        init();
        let src = "def mk_inc() -> fn(Int) -> Int = |x: Int| x + 1";
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ai-lang-durable-atom-{}", nanos));
        let key = Hash::of_bytes(b"counter");

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        let kb = KnowledgeBase::new();

        // Encode helpers bound to the client runtime.
        let encode = |rt: &Runtime, ptr: *const u8| -> Vec<u8> {
            let mut b = Vec::new();
            unsafe { encode_value(rt, WireValue::Heap(ptr), &mut b).unwrap() };
            b
        };
        let mk_inc = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["mk_inc"]),
                )
                .unwrap()
        };

        // --- First server lifetime: set 0, swap twice -> 2. ---
        {
            let server_ctx = Context::create();
            let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
            let _ = server_jit;
            assert_eq!(server_rt.enable_cell_persistence(&dir).unwrap(), 0);
            let (mut client_chan, mut server_chan) = InProcessChannel::pair();
            let server_runtime = Arc::new(RuntimeHandle(server_rt));
            let server_for_thread = server_runtime.clone();
            let handle = std::thread::spawn(move || -> Result<(), NetError> {
                for _ in 0..3 {
                    unsafe { serve_atom_one(&server_for_thread.0, &mut server_chan)? };
                }
                Ok(())
            });

            let zero = unsafe { crate::runtime::ai_gc_box_int(client_rt.thread_ptr(), 0) };
            let set = build_atom_frame(KIND_ATOM_SET, &key, &encode(&client_rt, zero as *const u8));
            unsafe {
                atom_request_on_channel(&client_rt, &kb, &mut client_chan, &set).unwrap()
            };
            for _ in 0..2 {
                let c = unsafe { mk_inc.call(client_rt.thread_ptr()) };
                let swap =
                    build_atom_frame(KIND_ATOM_SWAP, &key, &encode(&client_rt, c as *const u8));
                unsafe {
                    atom_request_on_channel(&client_rt, &kb, &mut client_chan, &swap).unwrap()
                };
            }
            handle.join().unwrap().expect("server ok");
            // server_runtime dropped here — the node "goes down".
        }

        // --- Second server lifetime: fresh Runtime, same dir. ---
        {
            let server_ctx = Context::create();
            let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
            let _ = server_jit;
            // Reloads the one persisted cell.
            assert_eq!(server_rt.enable_cell_persistence(&dir).unwrap(), 1);
            let (mut client_chan, mut server_chan) = InProcessChannel::pair();
            let server_runtime = Arc::new(RuntimeHandle(server_rt));
            let server_for_thread = server_runtime.clone();
            let handle = std::thread::spawn(move || -> Result<(), NetError> {
                unsafe { serve_atom_one(&server_for_thread.0, &mut server_chan) }
            });

            let get = build_atom_frame(KIND_ATOM_GET, &key, &[]);
            let got = unsafe {
                atom_request_on_channel(&client_rt, &kb, &mut client_chan, &get).unwrap()
            };
            assert_eq!(
                unsafe { crate::runtime::ai_gc_unbox_int(got) },
                2,
                "counter value should survive the server restart"
            );
            handle.join().unwrap().expect("server ok");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// GET on a never-initialized cell reports absence rather than
    /// returning a bogus value.
    #[test]
    fn in_process_atom_get_absent() {
        init();
        let src = "def noop() -> Int = 0";
        let server_ctx = Context::create();
        let (server_rt, _jit, _) = make_compiled_runtime(&server_ctx, src);
        let (mut client_chan, mut server_chan) = InProcessChannel::pair();
        let server_runtime = Arc::new(RuntimeHandle(server_rt));
        let server_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || -> Result<(), NetError> {
            unsafe { serve_atom_one(&server_for_thread.0, &mut server_chan) }
        });

        let client_ctx = Context::create();
        let (client_rt, _client_jit, _names) = make_compiled_runtime(&client_ctx, src);
        let kb = KnowledgeBase::new();
        let key = Hash::of_bytes(b"missing");
        let get_frame = build_atom_frame(KIND_ATOM_GET, &key, &[]);
        let result =
            unsafe { atom_request_on_channel(&client_rt, &kb, &mut client_chan, &get_frame) };
        assert!(matches!(result, Err(NetError::AtomAbsent)));
        server_handle.join().unwrap().expect("server ok");
    }

    /// End-to-end distributed counter written in ai-lang: the language
    /// surface (`atom_init` / `atom_swap` / `atom_deref`) lowers to the
    /// runtime fns, ships over TCP to a cell living on the server, and
    /// the increment closure runs server-side. This is the headline
    /// language-level proof of located state.
    #[test]
    fn lang_distributed_counter() {
        init();
        let src = "
            struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
            enum Failure {
                Unreachable(Node),
                Crashed(Node),
                CodeMissing(Node),
                Cancelled(Node),
            }
            enum Result<T, E> { Ok(T), Err(E) }
            struct Atom<T> { node: Node, name: String }

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            def make_counter(node: Node) -> Atom<Int> =
                Atom { node: node, name: \"counter\" }

            def do_init(a: Atom<Int>) -> Int =
                match atom_init(a, 0) { Ok(v) => v, Err(_) => 0 - 1 }

            def do_inc(a: Atom<Int>) -> Int =
                match atom_swap(a, |x: Int| x + 1) { Ok(v) => v, Err(_) => 0 - 1 }

            def do_get(a: Atom<Int>) -> Int =
                match atom_deref(a) { Ok(v) => v, Err(_) => 0 - 1 }
        ";

        // Server: shares source so it already holds the increment
        // closure's code. Persistent atom-serving loop on one connection.
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));
        let server_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || {
            if let Ok(mut stream) = accept_one(&listener) {
                loop {
                    match unsafe { serve_atom_one(&server_for_thread.0, &mut stream) } {
                        Ok(()) => continue,
                        Err(_) => break,
                    }
                }
            }
        });

        // Client: install the three thread-locals, build the handle,
        // run init → inc → inc → get.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        clear_at_conn_cache();
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("atom ops should build an at_binding");
        let rt_binding = build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let make_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["make_node"]))
                .unwrap()
        };
        let make_counter = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["make_counter"]),
                )
                .unwrap()
        };
        let getter = |n: &str| unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names[n]),
                )
                .unwrap()
        };
        let do_init = getter("do_init");
        let do_inc = getter("do_inc");
        let do_get = getter("do_get");

        let node = unsafe { make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port) };
        let counter = unsafe { make_counter.call(client_rt.thread_ptr(), node) };

        assert_eq!(unsafe { do_init.call(client_rt.thread_ptr(), counter) }, 0);
        assert_eq!(unsafe { do_inc.call(client_rt.thread_ptr(), counter) }, 1);
        assert_eq!(unsafe { do_inc.call(client_rt.thread_ptr(), counter) }, 2);
        assert_eq!(unsafe { do_get.call(client_rt.thread_ptr(), counter) }, 2);

        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        server_handle.join().unwrap();
    }

    /// "Redis as a library": a distributed key-value store is just a few
    /// lines of ORDINARY ai-lang over atoms — no privileged core
    /// support. Keys are cell names on a node; `kv_put`/`kv_get`/`kv_bump`
    /// are user-space wrappers. Two independent keys live on one node and
    /// behave as a store; `kv_bump` ships an increment closure (atomic
    /// read-modify-write, server-side, no Lua).
    #[test]
    fn lang_kv_store_over_atoms() {
        init();
        let src = "
            struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
            enum Failure {
                Unreachable(Node), Crashed(Node), CodeMissing(Node), Cancelled(Node),
            }
            enum Result<T, E> { Ok(T), Err(E) }
            struct Atom<T> { node: Node, name: String }

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            // --- the entire KV 'library', in user space ---
            def kv_put(node: Node, key: String, v: Int) -> Int = {
                let cell = Atom { node: node, name: key };
                match atom_set(cell, v) { Ok(x) => x, Err(_) => 0 - 1 }
            }
            def read_cell(cell: Atom<Int>) -> Int =
                match atom_deref(cell) { Ok(x) => x, Err(_) => 0 - 1 }
            def kv_get(node: Node, key: String) -> Int = {
                let cell = Atom { node: node, name: key };
                read_cell(cell)
            }
            def kv_bump(node: Node, key: String) -> Int = {
                let cell = Atom { node: node, name: key };
                match atom_swap(cell, |x: Int| x + 1) { Ok(x) => x, Err(_) => 0 - 1 }
            }
        ";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));
        let server_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || {
            if let Ok(mut stream) = accept_one(&listener) {
                while let Ok(()) =
                    unsafe { serve_atom_one(&server_for_thread.0, &mut stream) }
                {}
            }
        });

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        clear_at_conn_cache();
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("atom ops build an at_binding");
        let rt_binding = build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let t = client_rt.thread_ptr();
        let make_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread, i64, i64, i64, i64, i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["make_node"]))
                .unwrap()
        };
        let node = unsafe { make_node.call(t, 127, 0, 0, 1, server_port) };
        // String-keyed ops: (Node, String, Int) -> Int and (Node, String) -> Int.
        let put = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["kv_put"]),
                )
                .unwrap()
        };
        let get = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["kv_get"]),
                )
                .unwrap()
        };
        let bump = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["kv_bump"]),
                )
                .unwrap()
        };
        let key_x = unsafe { crate::ffi::owned_str_to_heap(t, "x") };
        let key_y = unsafe { crate::ffi::owned_str_to_heap(t, "y") };

        assert_eq!(unsafe { put.call(t, node, key_x, 10) }, 10);
        assert_eq!(unsafe { put.call(t, node, key_y, 20) }, 20);
        assert_eq!(unsafe { bump.call(t, node, key_x) }, 11);
        assert_eq!(unsafe { get.call(t, node, key_x) }, 11);
        assert_eq!(unsafe { get.call(t, node, key_y) }, 20, "keys are independent");

        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        server_handle.join().unwrap();
    }

    /// THE model Jimmy wants: a whole key-value store is ONE atom
    /// holding an immutable persistent map (PMap). `kv_put` ships a
    /// closure `|m| pmap_assoc(m, key, val)` to the owning node, which
    /// CAS-swaps the cell's root. This is a true compare-and-set over an
    /// immutable value, with cross-key consistent snapshots — unlike the
    /// N-scalar-cells `lang_kv_store_over_atoms`. It also exercises the
    /// new wire support for Array (the HAMT's bitmap kids) and String
    /// (the HAMT's keys, and the swap closure's captured key).
    #[test]
    fn lang_kv_store_as_one_pmap_atom() {
        init();
        let driver = "
            struct Atom<T> { node: Node, name: String }
            def store(node: Node) -> Atom<PMap<Int>> = Atom { node: node, name: \"kv\" }
            def kv_init(node: Node) -> Int = {
                let s = store(node);
                match atom_init(s, pmap_empty()) { Ok(_) => 0, Err(_) => 0 - 9 }
            }
            def kv_put(node: Node, key: String, val: Int) -> Int = {
                let s = store(node);
                match atom_swap(s, |m: PMap<Int>| pmap_assoc(m, key, val)) {
                    Ok(_) => val, Err(_) => 0 - 9
                }
            }
            def kv_get(node: Node, key: String) -> Int = {
                let s = store(node);
                match atom_deref(s) {
                    Ok(m) => match pmap_get(m, key) { Some(v) => v, None => 0 - 1 },
                    Err(_) => 0 - 2
                }
            }
        ";
        let src = format!("{}\n{}", crate::stdlib::SOURCE, driver);

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));
        let server_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || {
            if let Ok(mut stream) = accept_one(&listener) {
                while let Ok(()) =
                    unsafe { serve_atom_one(&server_for_thread.0, &mut stream) }
                {}
            }
        });

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        clear_at_conn_cache();
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("atom ops build an at_binding");
        let rt_binding = build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let t = client_rt.thread_ptr();
        let tcp_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread, i64, i64, i64, i64, i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["tcp_node"]))
                .unwrap()
        };
        let node = unsafe { tcp_node.call(t, 127, 0, 0, 1, server_port) };
        let init = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["kv_init"]),
                )
                .unwrap()
        };
        let put = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["kv_put"]),
                )
                .unwrap()
        };
        let get = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["kv_get"]),
                )
                .unwrap()
        };
        let key_x = unsafe { crate::ffi::owned_str_to_heap(t, "x") };
        let key_y = unsafe { crate::ffi::owned_str_to_heap(t, "y") };
        let key_z = unsafe { crate::ffi::owned_str_to_heap(t, "z") };

        assert_eq!(unsafe { init.call(t, node) }, 0, "init the store with an empty PMap");
        assert_eq!(unsafe { put.call(t, node, key_x, 10) }, 10);
        assert_eq!(unsafe { put.call(t, node, key_y, 20) }, 20);
        assert_eq!(unsafe { put.call(t, node, key_z, 30) }, 30);
        // All three keys live in ONE atom holding ONE immutable map.
        assert_eq!(unsafe { get.call(t, node, key_x) }, 10);
        assert_eq!(unsafe { get.call(t, node, key_y) }, 20);
        assert_eq!(unsafe { get.call(t, node, key_z) }, 30);
        // Overwrite an existing key (swap produces a new map sharing the
        // untouched subtrees; old value is replaced, others intact).
        assert_eq!(unsafe { put.call(t, node, key_x, 99) }, 99);
        assert_eq!(unsafe { get.call(t, node, key_x) }, 99);
        assert_eq!(unsafe { get.call(t, node, key_y) }, 20, "other keys survive");

        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        server_handle.join().unwrap();
    }

    /// Generic payoff over the wire: a distributed atom holding a
    /// `PMap<String>` (string -> string). The swap closure captures a
    /// String key AND a String value, and the whole immutable string-
    /// valued map ships back on deref — exercising generics + the Array/
    /// String wire support together. Returns string LENGTHS as i64 since
    /// the Rust harness reads i64.
    #[test]
    fn lang_kv_store_pmap_string_values_distributed() {
        init();
        let driver = "
            struct Atom<T> { node: Node, name: String }
            def store(node: Node) -> Atom<PMap<String>> = Atom { node: node, name: \"kvs\" }
            def kvs_init(node: Node) -> Int = {
                let s = store(node);
                match atom_init(s, pmap_empty()) { Ok(_) => 0, Err(_) => 0 - 9 }
            }
            def kvs_put(node: Node, key: String, val: String) -> Int = {
                let s = store(node);
                match atom_swap(s, |m: PMap<String>| pmap_assoc(m, key, val)) {
                    Ok(_) => string_len(val), Err(_) => 0 - 9
                }
            }
            def kvs_get_len(node: Node, key: String) -> Int = {
                let s = store(node);
                match atom_deref(s) {
                    Ok(m) => match pmap_get(m, key) { Some(v) => string_len(v), None => 0 - 1 },
                    Err(_) => 0 - 2
                }
            }
        ";
        let src = format!("{}\n{}", crate::stdlib::SOURCE, driver);

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));
        let server_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || {
            if let Ok(mut stream) = accept_one(&listener) {
                while let Ok(()) =
                    unsafe { serve_atom_one(&server_for_thread.0, &mut stream) }
                {}
            }
        });

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        clear_at_conn_cache();
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("atom ops build an at_binding");
        let rt_binding = build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let t = client_rt.thread_ptr();
        let tcp_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread, i64, i64, i64, i64, i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["tcp_node"]))
                .unwrap()
        };
        let node = unsafe { tcp_node.call(t, 127, 0, 0, 1, server_port) };
        let init = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["kvs_init"]),
                )
                .unwrap()
        };
        let put = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, *mut u8, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["kvs_put"]),
                )
                .unwrap()
        };
        let get_len = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["kvs_get_len"]),
                )
                .unwrap()
        };
        let s = |txt: &str| unsafe { crate::ffi::owned_str_to_heap(t, txt) };

        assert_eq!(unsafe { init.call(t, node) }, 0);
        // "ai-lang" len 7, "ada" len 3
        assert_eq!(unsafe { put.call(t, node, s("lang"), s("ai-lang")) }, 7);
        assert_eq!(unsafe { put.call(t, node, s("name"), s("ada")) }, 3);
        assert_eq!(unsafe { get_len.call(t, node, s("lang")) }, 7, "string value survived round-trip");
        assert_eq!(unsafe { get_len.call(t, node, s("name")) }, 3);
        assert_eq!(unsafe { get_len.call(t, node, s("missing")) }, -1);

        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        server_handle.join().unwrap();
    }

    /// The receiver-side memoization cache: shipping the same closure
    /// twice runs the body once. Asserts via the cache_hits /
    /// cache_misses counters on the server `Runtime`.
    ///
    /// Persistent in-process channel so two Call frames flow over
    /// one server context (the cache lives on the Runtime, not on
    /// the connection — so it'd persist across connections too).
    #[test]
    fn at_cache_short_circuits_repeated_calls() {
        init();
        let src = "
            def work(x: Int) -> Int = x * x + 7
            def make_thunk(x: Int) -> fn() -> Int = || work(x)
        ";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));
        let hits_handle = server_runtime.0.cache_hits.clone();
        let misses_handle = server_runtime.0.cache_misses.clone();

        let (mut client_chan, mut server_chan) = InProcessChannel::pair();

        // Server loops handling Call frames until the client closes.
        let runtime_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || {
            loop {
                match unsafe { serve_one(&runtime_for_thread.0, &mut server_chan) } {
                    Ok(()) => continue,
                    Err(_) => break,
                }
            }
        });

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        let make_thunk = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["make_thunk"]),
                )
                .unwrap()
        };
        let kb = KnowledgeBase::new();

        // Two identical calls. The closure captures the SAME value
        // (x = 5), so the encoded Call payloads are byte-identical
        // and the cache key matches.
        let closure1 = unsafe { make_thunk.call(client_rt.thread_ptr(), 5) };
        let r1 = unsafe {
            at_remote_on_channel(&client_rt, &kb, &mut client_chan, closure1 as *const u8)
                .unwrap()
        };
        let v1 = unsafe { crate::runtime::ai_gc_unbox_int(r1) };

        let closure2 = unsafe { make_thunk.call(client_rt.thread_ptr(), 5) };
        let r2 = unsafe {
            at_remote_on_channel(&client_rt, &kb, &mut client_chan, closure2 as *const u8)
                .unwrap()
        };
        let v2 = unsafe { crate::runtime::ai_gc_unbox_int(r2) };

        assert_eq!(v1, 32);
        assert_eq!(v2, 32);

        // Close the client side so the server loop exits.
        drop(client_chan);
        server_handle.join().unwrap();

        let hits = hits_handle.load(std::sync::atomic::Ordering::Relaxed);
        let misses = misses_handle.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(misses, 1, "first call should miss");
        assert_eq!(hits, 1, "second identical call should hit the cache");
    }

    /// End-to-end code-fetch: server starts with empty program, client
    /// ships a closure; server fetches code, installs it, runs it.
    ///
    /// Threading note: neither the `IncrementalJit` (holds an LLVM
    /// `Context`) nor the `Runtime` (holds raw pointers into the heap)
    /// is `Send`, so we can't move them into a spawned thread. We use
    /// the inverse setup: client connects from a thread, server runs
    /// inline on the test thread.
    #[test]
    fn tcp_code_fetch_install_and_run() {
        init();
        let client_src = "
            def work(x: Int) -> Int = x * x + 7
            def make_thunk(x: Int) -> fn() -> Int = || work(x)
        ";

        // --- SERVER side (this thread): empty program, IncrementalJit. ---
        let server_ctx = Context::create();
        let empty_m = parse_module("").unwrap();
        let empty_r = resolve_module(&empty_m).unwrap();
        let empty_cm = crate::codegen::CompiledModule::build(&server_ctx, &empty_r).unwrap();
        let mut server_rt = Runtime::new_with_metadata(
            empty_cm.closure_type_infos.clone(),
            empty_cm.shape_registry.clone(),
            empty_cm.shape_meta.clone(),
            empty_cm.shape_by_type_id.clone(),
        );
        let mut server_jit =
            crate::codegen::IncrementalJit::new(empty_cm, &server_rt).unwrap();
        // The empty module has no user-defined shapes; the runtime
        // reserves the three trailing slots for BoxedInt + String + Array.
        assert_eq!(server_rt.shape_by_type_id.len(), 3);

        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();

        // --- CLIENT side (spawned thread): full program. Unbox the
        // BoxedInt the server returns *inside* the thread (the
        // client Runtime is local to it).
        let client_handle = std::thread::spawn(move || -> i64 {
            let client_ctx = Context::create();
            let client_m = parse_module(client_src).unwrap();
            let client_r = resolve_module(&client_m).unwrap();
            let names: std::collections::HashMap<String, Hash> = client_r
                .defs
                .iter()
                .map(|d| (d.name.clone(), d.hash))
                .collect();
            let client_cm =
                crate::codegen::CompiledModule::build(&client_ctx, &client_r).unwrap();
            let client_rt = Runtime::new_with_metadata(
                client_cm.closure_type_infos.clone(),
                client_cm.shape_registry.clone(),
                client_cm.shape_meta.clone(),
                client_cm.shape_by_type_id.clone(),
            );
            let client_jit = Jit::new(client_cm, &client_rt).unwrap();
            let kb = KnowledgeBase::build(&client_r);

            let make_thunk = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                        &crate::codegen::def_symbol(&names["make_thunk"]),
                    )
                    .unwrap()
            };
            let closure_ptr = unsafe { make_thunk.call(client_rt.thread_ptr(), 6) };
            let result_ptr = unsafe {
                at_remote(&client_rt, &kb, server_addr, closure_ptr as *const u8).unwrap()
            };
            unsafe { crate::runtime::ai_gc_unbox_int(result_ptr) }
        });

        let mut stream = accept_one(&listener).unwrap();
        unsafe { serve_with_install(&mut server_rt, &mut server_jit, &mut stream).unwrap() };

        let result = client_handle.join().unwrap();
        assert_eq!(result, 6 * 6 + 7);
        assert!(
            server_rt.shape_by_type_id.len() >= 1,
            "server should have installed at least the lambda's shape"
        );
    }

    /// The crown jewel: ship a `swap` updater the server has NEVER
    /// seen, mutating located state with code that arrives over the
    /// wire. Server starts empty; client `set`s a cell, then `swap`s
    /// with an increment closure; the server fetches+installs the
    /// closure's code via the NeedCode handshake, applies it to the
    /// cell, and stores the result. Server-side scripting with no
    /// second language — the code IS the language.
    #[test]
    fn tcp_atom_swap_fetches_updater_code() {
        init();
        let client_src = "def mk_inc() -> fn(Int) -> Int = |x: Int| x + 1";

        // --- SERVER (this thread): empty program + IncrementalJit. ---
        let server_ctx = Context::create();
        let empty_m = parse_module("").unwrap();
        let empty_r = resolve_module(&empty_m).unwrap();
        let empty_cm = crate::codegen::CompiledModule::build(&server_ctx, &empty_r).unwrap();
        let mut server_rt = Runtime::new_with_metadata(
            empty_cm.closure_type_infos.clone(),
            empty_cm.shape_registry.clone(),
            empty_cm.shape_meta.clone(),
            empty_cm.shape_by_type_id.clone(),
        );
        let mut server_jit =
            crate::codegen::IncrementalJit::new(empty_cm, &server_rt).unwrap();
        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let key = Hash::of_bytes(b"c");

        // --- CLIENT (spawned thread): full program; ships set then swap. ---
        let client_handle = std::thread::spawn(move || -> i64 {
            let client_ctx = Context::create();
            let client_m = parse_module(client_src).unwrap();
            let client_r = resolve_module(&client_m).unwrap();
            let names: std::collections::HashMap<String, Hash> = client_r
                .defs
                .iter()
                .map(|d| (d.name.clone(), d.hash))
                .collect();
            let client_cm =
                crate::codegen::CompiledModule::build(&client_ctx, &client_r).unwrap();
            let client_rt = Runtime::new_with_metadata(
                client_cm.closure_type_infos.clone(),
                client_cm.shape_registry.clone(),
                client_cm.shape_meta.clone(),
                client_cm.shape_by_type_id.clone(),
            );
            let client_jit = Jit::new(client_cm, &client_rt).unwrap();
            let kb = KnowledgeBase::build(&client_r);
            let mut stream = TcpStream::connect(server_addr).unwrap();

            // set cell = 10 (boxed Int; server stores the bytes raw).
            let ten = unsafe { crate::runtime::ai_gc_box_int(client_rt.thread_ptr(), 10) };
            let mut enc = Vec::new();
            unsafe {
                encode_value(&client_rt, WireValue::Heap(ten as *const u8), &mut enc).unwrap()
            };
            let set_frame = build_atom_frame(KIND_ATOM_SET, &key, &enc);
            let _ = unsafe {
                atom_request_on_channel(&client_rt, &kb, &mut stream, &set_frame).unwrap()
            };

            // swap with the increment closure — server must fetch its code.
            let mk_inc = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> *mut u8>(
                        &crate::codegen::def_symbol(&names["mk_inc"]),
                    )
                    .unwrap()
            };
            let closure = unsafe { mk_inc.call(client_rt.thread_ptr()) };
            let mut cenc = Vec::new();
            unsafe {
                encode_value(&client_rt, WireValue::Heap(closure as *const u8), &mut cenc)
                    .unwrap()
            };
            let swap_frame = build_atom_frame(KIND_ATOM_SWAP, &key, &cenc);
            let new_ptr = unsafe {
                atom_request_on_channel(&client_rt, &kb, &mut stream, &swap_frame).unwrap()
            };
            unsafe { crate::runtime::ai_gc_unbox_int(new_ptr) }
        });

        // Server: serve the SET (no code needed), then the SWAP (fetches
        // + installs the updater's code mid-handshake).
        let mut stream = accept_one(&listener).unwrap();
        unsafe {
            serve_atom_with_install(&mut server_rt, &mut server_jit, &mut stream).unwrap()
        };
        unsafe {
            serve_atom_with_install(&mut server_rt, &mut server_jit, &mut stream).unwrap()
        };

        let result = client_handle.join().unwrap();
        assert_eq!(result, 11, "swap shipped to a code-less server should yield 10 + 1");
    }

    /// Shipping code that calls a C extern: the extern requirement must
    /// travel with the code so the (initially empty) server can resolve
    /// the real symbol and run it. `work` calls libc `abs` through the C
    /// FFI; the server starts with no knowledge of it and fetches the
    /// def + lambda + the `abs` extern requirement, resolves `abs` from
    /// libc, and runs the thunk.
    #[test]
    fn tcp_code_fetch_ships_c_extern_requirement() {
        init();
        let client_src = "
            extern \"C\" lib \"c\" { fn abs(n: Int) -> Int }
            def work(x: Int) -> Int = abs(0 - x) + 1
            def make_thunk(x: Int) -> fn() -> Int = || work(x)
        ";

        // SERVER: empty program, IncrementalJit (knows nothing of `abs`).
        let server_ctx = Context::create();
        let empty_m = parse_module("").unwrap();
        let empty_r = resolve_module(&empty_m).unwrap();
        let empty_cm = crate::codegen::CompiledModule::build(&server_ctx, &empty_r).unwrap();
        let mut server_rt = Runtime::new_with_metadata(
            empty_cm.closure_type_infos.clone(),
            empty_cm.shape_registry.clone(),
            empty_cm.shape_meta.clone(),
            empty_cm.shape_by_type_id.clone(),
        );
        let mut server_jit =
            crate::codegen::IncrementalJit::new(empty_cm, &server_rt).unwrap();

        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();

        let client_handle = std::thread::spawn(move || -> i64 {
            let client_ctx = Context::create();
            let client_m = parse_module(client_src).unwrap();
            let client_r = resolve_module(&client_m).unwrap();
            let names: std::collections::HashMap<String, Hash> = client_r
                .defs
                .iter()
                .map(|d| (d.name.clone(), d.hash))
                .collect();
            let client_cm =
                crate::codegen::CompiledModule::build(&client_ctx, &client_r).unwrap();
            let client_rt = Runtime::new_with_metadata(
                client_cm.closure_type_infos.clone(),
                client_cm.shape_registry.clone(),
                client_cm.shape_meta.clone(),
                client_cm.shape_by_type_id.clone(),
            );
            let client_jit = Jit::new(client_cm, &client_rt).unwrap();
            // KB carries the `abs` extern signature alongside the code.
            let kb = KnowledgeBase::build(&client_r);

            let make_thunk = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                        &crate::codegen::def_symbol(&names["make_thunk"]),
                    )
                    .unwrap()
            };
            let closure_ptr = unsafe { make_thunk.call(client_rt.thread_ptr(), 6) };
            let result_ptr = unsafe {
                at_remote(&client_rt, &kb, server_addr, closure_ptr as *const u8).unwrap()
            };
            unsafe { crate::runtime::ai_gc_unbox_int(result_ptr) }
        });

        let mut stream = accept_one(&listener).unwrap();
        unsafe { serve_with_install(&mut server_rt, &mut server_jit, &mut stream).unwrap() };

        let result = client_handle.join().unwrap();
        // work(6) = abs(-6) + 1 = 7, computed on the server via the
        // shipped + resolved libc `abs`.
        assert_eq!(result, 7);
    }

    /// End-to-end: the lang program itself calls `at(...)` and the
    /// runtime constructs a `Result::Ok(n)` heap value. The program's
    /// match extracts `n`; we assert the integer flows through.
    ///
    /// The runtime boxes the i64 return value into a `BoxedInt`; the
    /// user's match-arm extraction unboxes it.
    #[test]
    fn at_returns_ok_with_generic_result() {
        init();
        let src = "
            struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
            enum Failure {
                Unreachable(Node),
                Crashed(Node),
                CodeMissing(Node),
                Cancelled(Node),
            }
            enum Result<T, E> { Ok(T), Err(E) }

            def work(x: Int) -> Int = x * 10 + 3

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            def run(node: Node, x: Int) -> Int =
                match at(node, || work(x)) {
                    Ok(n) => n,
                    Err(_) => 0 - 1,
                }
        ";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let server_port = server_addr.port() as i64;
        let server_handle = unsafe {
            serve_one_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let make_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["make_node"]))
                .unwrap()
        };
        let node = unsafe {
            make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port)
        };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let result = unsafe { run.call(client_rt.thread_ptr(), node, 4) };
        assert_eq!(result, 4 * 10 + 3, "Ok(43) round-trip should yield 43");

        server_handle.join().unwrap().expect("server ok");
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// `at(...)` to an address with no listener should produce
    /// `Err(Unreachable(node))` — verified by the lang-side match
    /// returning -1 (the Err arm).
    #[test]
    fn at_returns_err_on_unreachable() {
        init();
        let src = "
            struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
            enum Failure {
                Unreachable(Node),
                Crashed(Node),
                CodeMissing(Node),
                Cancelled(Node),
            }
            enum Result<T, E> { Ok(T), Err(E) }

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            def run(node: Node) -> Int =
                match at(node, || 42) {
                    Ok(n) => n,
                    Err(_) => 0 - 1,
                }
        ";

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let resolver_binding = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, resolver_binding).expect("rt binding");
        install_current_at_binding(&rt_binding);

        // Bind a listener, immediately drop it so the port goes back to
        // the kernel. Connecting to that port should yield ECONNREFUSED
        // → NetError::Io → Failure::Unreachable.
        let dead_port = {
            let l = bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port() as i64
        };

        let make_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["make_node"]))
                .unwrap()
        };
        let node = unsafe {
            make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, dead_port)
        };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let result = unsafe { run.call(client_rt.thread_ptr(), node) };
        assert_eq!(result, -1, "expected Err arm to fire (returns -1)");

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// End-to-end "real program" test. Exercises:
    /// - The full stdlib (concatenated ahead of user source).
    /// - Recursive types (`List<T>` + `ListCell<T>`).
    /// - Generic constructor inference (`Cons(ListCell { head: ..., tail: ... })`).
    /// - Fully generic higher-order fns (`list_map`, `list_foldl`, `list_filter`).
    /// - Closure capture + uniform closure ABI.
    /// - `at()` wire roundtrip with a non-trivial server-side computation.
    /// - Typechecker (run before codegen).
    ///
    /// The program: client ships a closure to the server. Closure
    /// builds the range `[1..=n]`, doubles each element, filters out
    /// multiples of 3, and sums the result. Returns the sum wrapped
    /// in `Result::Ok(...)`.
    #[test]
    fn at_runs_full_stack_computation() {
        init();
        // Node / Failure / Result / tcp_node come from the stdlib.
        let user_src = "
            // Heavy lifting on the server side. Exercises:
            //   - `int_list_range` → builds `List<Int>` via tail-rec.
            //   - `list_map`       → generic HOF doubling each element.
            //   - `list_filter`    → generic HOF keeping non-multiples-of-3.
            //   - `list_foldl`     → generic HOF summing the result.
            //   - Closures with captured `n` flow through generic boundaries
            //     via the uniform closure ABI.
            def compute(n: Int) -> Int =
                list_foldl(
                    list_filter(
                        list_map(int_list_range(1, n + 1), |x: Int| x * 2),
                        |x: Int| if x - (x / 3) * 3 == 0 { 0 } else { 1 }
                    ),
                    0,
                    |acc: Int, x: Int| acc + x
                )

            def run(node: Node, n: Int) -> Int =
                match at(node, || compute(n)) {
                    Ok(v) => v,
                    Err(_) => 0 - 1,
                }
        ";

        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        // Pre-check: full pipeline must typecheck before we even
        // build either runtime. If this fails, the assertion below
        // does too — but the typecheck error message is much clearer.
        {
            let m = parse_module(&src).expect("parse");
            let r = resolve_module(&m).expect("resolve");
            let mut cache = crate::typecheck::TypeCache::new();
            crate::typecheck::typecheck_module(&r, &mut cache)
                .expect("typecheck stdlib + user");
        }

        // Server side: same source so it can run the shipped closure.
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let server_port = server_addr.port() as i64;
        let server_handle = unsafe {
            serve_one_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        // Client side.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);

        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let resolver_binding = r.at_binding.as_ref().expect("at_binding populated");
        let rt_binding = build_at_runtime_binding(&client_rt, resolver_binding)
            .expect("runtime binding built");
        install_current_at_binding(&rt_binding);

        let tcp_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["tcp_node"]))
                .unwrap()
        };
        let node = unsafe {
            tcp_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port)
        };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let n = 10i64;
        let result = unsafe { run.call(client_rt.thread_ptr(), node, n) };

        // Expected: [1..=10] -> doubled = [2,4,6,8,10,12,14,16,18,20]
        // filter !div_by_3 -> [2,4,8,10,14,16,20] (drop 6, 12, 18)
        // sum = 74.
        let expected = {
            let mut s = 0i64;
            for x in 1..=n {
                let d = x * 2;
                if d % 3 != 0 {
                    s += d;
                }
            }
            s
        };
        assert_eq!(result, expected);

        server_handle.join().unwrap().expect("server ok");

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// `at()` round-trips a user-defined struct: the closure returns
    /// a `Pair { a, b }` and the client reads back both fields. This
    /// Phase 3 preflight: `at(node, b.f)` where `b.f` is a stored
    /// fn-typed struct field, not an inline lambda literal. This is
    /// the load-bearing assumption for the lazy `Value<T>` design —
    /// if `at()` only accepts inline lambdas, composition via
    /// closure-in-struct doesn't work.
    #[test]
    fn at_accepts_stored_fn_field() {
        init();
        let user_src = "
            struct Box { f: fn() -> Int }

            def make_box(x: Int) -> Box = Box { f: || x + 100 }

            def run(b: Box, node: Node) -> Int =
                match at(node, b.f) {
                    Ok(v) => v,
                    Err(_) => 0 - 1,
                }
        ";
        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        // Pre-typecheck so a failure surfaces cleanly.
        {
            let m = parse_module(&src).expect("parse");
            let r = resolve_module(&m).expect("resolve");
            let mut cache = crate::typecheck::TypeCache::new();
            crate::typecheck::typecheck_module(&r, &mut cache)
                .expect("typecheck");
        }

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));

        let (mut client_chan, mut server_chan) = InProcessChannel::pair();
        let rt_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || {
            // Single Call expected.
            let _ = unsafe { serve_one(&rt_for_thread.0, &mut server_chan) };
        });

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding = build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        // Build a Box on the client.
        let make_box = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["make_box"]),
                )
                .unwrap()
        };
        let boxv = unsafe { make_box.call(client_rt.thread_ptr(), 42) };

        // Build a dummy "node" — for in-process we just need *something*
        // shaped like a Node; the channel ignores the address.
        let tcp_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["tcp_node"]))
                .unwrap()
        };
        let node = unsafe { tcp_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, 0) };

        // Invoke run(b, node) using at_remote_on_channel directly,
        // since the in-process channel can't take a TCP node address.
        // We bypass `run` and ship b.f manually — equivalent payload.
        //
        // Actually, just call `run` via the JIT — but run() calls at()
        // which dispatches to ai_net_at which goes through TCP. To
        // exercise in-process we need a different path.
        //
        // Workaround: peek the closure pointer out of the box field
        // directly, ship it via at_remote_on_channel.
        let _ = node;
        // Box's `f` field is the first (only) field. Box has 0 raw
        // fields and 1 pointer field, so f sits at offset
        // Full::SIZE = 16 from the box start.
        let f_slot = unsafe { (boxv as *const u8).add(<crate::gc::Full as crate::gc::ObjHeader>::SIZE) };
        let f_ptr = unsafe { *(f_slot as *const *const u8) };

        let result_ptr = unsafe {
            at_remote_on_channel(&client_rt, &kb, &mut client_chan, f_ptr).unwrap()
        };
        let result = unsafe { crate::runtime::ai_gc_unbox_int(result_ptr) };
        assert_eq!(result, 142);

        // Make sure the server thread joined cleanly.
        drop(client_chan);
        server_handle.join().unwrap();

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// The lazy-Value laziness proof: stacking two `value_map`s
    /// client-side and forcing once should ship ONE Call frame.
    /// Composition is just closure-in-closure — the worker
    /// evaluates the whole chain locally and returns one T.
    #[test]
    fn value_map_chains_lazily() {
        init();
        let user_src = "
            // Pipeline: force(map(map(pure(5), *2), +100)) = (5*2)+100 = 110
            def build(node: Node) -> Value<Int> =
                value_map(
                    value_map(value_pure(node, 5), |x: Int| x * 2),
                    |x: Int| x + 100
                )
        ";
        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        // Pre-typecheck for clear failure if Value<Int> typechecking
        // has any gaps.
        {
            let m = parse_module(&src).expect("parse");
            let r = resolve_module(&m).expect("resolve");
            let mut cache = crate::typecheck::TypeCache::new();
            crate::typecheck::typecheck_module(&r, &mut cache)
                .expect("typecheck");
        }

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));
        // Tracking: count Call frames seen by the server.
        let calls_seen_counter = server_runtime.0.cache_misses.clone();

        let (mut client_chan, mut server_chan) = InProcessChannel::pair();
        let rt_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || {
            loop {
                match unsafe { serve_one(&rt_for_thread.0, &mut server_chan) } {
                    Ok(()) => continue,
                    Err(_) => break,
                }
            }
        });

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let tcp_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["tcp_node"]))
                .unwrap()
        };
        let node = unsafe { tcp_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, 0) };

        // Build the lazy chain — should not have shipped anything yet.
        let build = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["build"]),
                )
                .unwrap()
        };
        let value = unsafe { build.call(client_rt.thread_ptr(), node) };
        assert_eq!(
            calls_seen_counter.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "value_map should be lazy — no frames before force"
        );

        // Force by pulling out value.recipe and shipping. Bypass the
        // ai_net_at path (which uses TCP) and use the in-process
        // channel directly. Value layout: 1 pointer field (node) +
        // 1 pointer field (recipe), both in value_field slots after
        // the header.
        let header = <crate::gc::Full as crate::gc::ObjHeader>::SIZE;
        // Value<T> has 0 raw fields, 2 pointer fields. node at +16, recipe at +24.
        let recipe_slot = unsafe { (value as *const u8).add(header + 8) };
        let recipe_ptr = unsafe { *(recipe_slot as *const *const u8) };

        let result_ptr = unsafe {
            at_remote_on_channel(&client_rt, &kb, &mut client_chan, recipe_ptr)
                .unwrap()
        };
        let result = unsafe { crate::runtime::ai_gc_unbox_int(result_ptr) };
        assert_eq!(result, 5 * 2 + 100, "chain should compute (5*2)+100 = 110");

        // ONE Call frame seen by the server (recorded as a cache MISS
        // since this exact closure hasn't been seen before).
        assert_eq!(
            calls_seen_counter.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the whole chain ships as ONE Call — intermediates never leave the worker"
        );

        drop(client_chan);
        server_handle.join().unwrap();

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// Phase 3 pipeline E2E: split a list into chunks across 3 TCP
    /// workers, build a lazy map+filter+reduce pipeline, force.
    /// Each chunk's whole chain (map, filter, foldl) runs in one
    /// Call per chunk; client combines partials.
    #[test]
    fn dataset_pipeline_e2e() {
        init();
        let user_src = "
            // Three chunks of four → [1..=12]. Pipeline output for
            // double-then-keep-not-div-by-3-then-sum:
            // [1,2,3,4]    *2 → [2,4,6,8]    filter → [2,4,8]     sum 14
            // [5,6,7,8]    *2 → [10,12,14,16] filter → [10,14,16]  sum 40
            // [9,10,11,12] *2 → [18,20,22,24] filter → [20,22]     sum 42
            // grand sum = 96.
            def chunks() -> List<List<Int>> =
                Cons(ListCell { head: int_list_range(1, 5), tail:
                Cons(ListCell { head: int_list_range(5, 9), tail:
                Cons(ListCell { head: int_list_range(9, 13), tail: Nil })
                })
                })

            // Run the pipeline against three (possibly equal) nodes.
            def pipeline(nodes: List<Node>) -> Int = {
                let d = dataset_from_chunks(chunks(), nodes);
                let mapped = dataset_map(d, |x: Int| x * 2);
                let filtered = dataset_filter(mapped, |x: Int|
                    if x - (x / 3) * 3 == 0 { 0 } else { 1 });
                dataset_reduce(filtered, 0, |a: Int, b: Int| a + b)
            }

            // Build a List<Node> of three identical nodes pointing
            // at the same worker (port supplied by the test driver).
            def make_pool(port: Int) -> List<Node> =
                Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail:
                Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail:
                Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail: Nil })
                })
                })

            def run(port: Int) -> Int = pipeline(make_pool(port))
        ";
        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        // Typecheck up front.
        {
            let m = parse_module(&src).expect("parse");
            let r = resolve_module(&m).expect("resolve");
            let mut cache = crate::typecheck::TypeCache::new();
            crate::typecheck::typecheck_module(&r, &mut cache)
                .expect("typecheck");
        }

        // Server (persistent — handles multiple Calls).
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let _server_handle = unsafe {
            serve_persistent_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        // Client.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let result = unsafe { run.call(client_rt.thread_ptr(), server_port) };
        assert_eq!(result, 96);

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        clear_at_conn_cache();
    }

    /// RemotePtr end-to-end: allocate memory ON a remote node, write a
    /// value to it, read it back, and free it — four separate `at(...)`
    /// RPCs to the same persistent worker. The raw `Ptr` never crosses
    /// the wire (the guard forbids that); only the address travels, as a
    /// plain Int inside a wire-portable `RemotePtr`, and each access
    /// reconstructs the pointer ON the owning node via `int_to_ptr`.
    #[test]
    fn remote_ptr_alloc_write_read_free() {
        init();
        let user_src = "
            def run_free(rp: RemotePtr, v: Int) -> Int =
                match remote_free(rp) {
                    Ok(f) => v,
                    Err(e) => 0 - 3
                }
            def run_read(rp: RemotePtr) -> Int =
                match remote_read_i64(rp) {
                    Ok(v) => run_free(rp, v),
                    Err(e) => 0 - 2
                }
            def run_write(rp: RemotePtr) -> Int =
                match remote_write_i64(rp, 12345) {
                    Ok(w) => run_read(rp),
                    Err(e) => 0 - 1
                }
            def run(port: Int) -> Int = {
                let node = tcp_node(127, 0, 0, 1, port);
                match remote_alloc(node, 8) {
                    Ok(rp) => run_write(rp),
                    Err(e) => 0 - 4
                }
            }
        ";
        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        // Typecheck up front (the at() Ptr-guard must accept the remote_*
        // thunks: they capture only Ints and return Ints).
        {
            let m = parse_module(&src).expect("parse");
            let r = resolve_module(&m).expect("resolve");
            let mut cache = crate::typecheck::TypeCache::new();
            crate::typecheck::typecheck_module(&r, &mut cache).expect("typecheck");
        }

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let _server_handle = unsafe {
            serve_persistent_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding = build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let result = unsafe { run.call(client_rt.thread_ptr(), server_port) };
        // Wrote 12345 to remote memory, read it back through the handle.
        assert_eq!(result, 12345);

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        clear_at_conn_cache();
    }

    /// Stress test for the gol-style chunk pattern: 8 single-element
    /// chunks computed via a recursive index-builder, each going to
    /// the same worker, mapped with `|i| f(captured, i)`, summed.
    /// Trip-wire for any per-chunk-Int-capture bug.
    #[test]
    fn dataset_map_with_recursive_chunks() {
        init();
        let user_src = "
            def compute(captured: Int, i: Int) -> Int = captured * i + i

            def indices(n: Int) -> List<List<Int>> =
                list_reverse(indices_acc(n, 0, Nil))

            def indices_acc(n: Int, i: Int, acc: List<List<Int>>) -> List<List<Int>> =
                if i >= n { acc } else {
                    indices_acc(n, i + 1, Cons(ListCell {
                        head: Cons(ListCell { head: i, tail: Nil }),
                        tail: acc,
                    }))
                }

            def make_pool(port: Int, n: Int) -> List<Node> =
                make_pool_acc(port, n, Nil)

            def make_pool_acc(port: Int, n: Int, acc: List<Node>) -> List<Node> =
                if n <= 0 { acc } else {
                    make_pool_acc(port, n - 1, Cons(ListCell {
                        head: tcp_node(127, 0, 0, 1, port), tail: acc,
                    }))
                }

            def run(port: Int, captured: Int, n: Int) -> Int = {
                let pool = make_pool(port, n);
                let d = dataset_from_chunks(indices(n), pool);
                let mapped = dataset_map(d, |i: Int| compute(captured, i));
                let bits = dataset_collect(mapped);
                list_sum(bits)
            }
        ";
        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let _server_handle = unsafe {
            serve_persistent_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let captured: i64 = 115202;
        let n: i64 = 49;
        let result = unsafe { run.call(client_rt.thread_ptr(), server_port, captured, n) };
        // Sum_{i=0..n-1} (captured*i + i) = (captured+1) * (n-1)*n/2
        let expected: i64 = (0..n).map(|i| captured * i + i).sum();
        assert_eq!(result, expected, "got {}, expected {}", result, expected);

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        clear_at_conn_cache();
    }

    /// dataset_map with a lambda that captures an Int from outer
    /// scope. This is the gol.ail pattern (capture `packed`, apply
    /// per-cell function). Tests that the captured Int survives the
    /// closure-in-closure shipping path.
    #[test]
    fn dataset_map_with_int_capture() {
        init();
        let user_src = "
            def add_offset(offset: Int, x: Int) -> Int = offset + x

            // Single-element chunks: [[10], [20], [30]] on one node.
            def chunks() -> List<List<Int>> =
                Cons(ListCell { head: Cons(ListCell { head: 10, tail: Nil }), tail:
                Cons(ListCell { head: Cons(ListCell { head: 20, tail: Nil }), tail:
                Cons(ListCell { head: Cons(ListCell { head: 30, tail: Nil }), tail: Nil })
                })
                })

            def make_pool(port: Int) -> List<Node> =
                Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail:
                Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail:
                Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail: Nil })
                })
                })

            def run(port: Int, offset: Int) -> Int = {
                let pool = make_pool(port);
                let d = dataset_from_chunks(chunks(), pool);
                let mapped = dataset_map(d, |x: Int| add_offset(offset, x));
                let bits = dataset_collect(mapped);
                list_sum(bits)
            }
        ";
        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        // Typecheck up front.
        {
            let m = parse_module(&src).expect("parse");
            let r = resolve_module(&m).expect("resolve");
            let mut cache = crate::typecheck::TypeCache::new();
            crate::typecheck::typecheck_module(&r, &mut cache)
                .expect("typecheck");
        }

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let _server_handle = unsafe {
            serve_persistent_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let offset: i64 = 100;
        let result = unsafe { run.call(client_rt.thread_ptr(), server_port, offset) };
        // Each element gets offset added: [110, 120, 130]. Sum = 360.
        assert_eq!(result, 110 + 120 + 130);

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        clear_at_conn_cache();
    }

    /// is the headline behaviour of phase 1A — before, `at()` was
    /// restricted to Int returns.
    #[test]
    fn at_returns_struct() {
        init();
        let src = "
            struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
            struct Pair { a: Int, b: Int }
            enum Failure {
                Unreachable(Node),
                Crashed(Node),
                CodeMissing(Node),
                Cancelled(Node),
            }
            enum Result<T, E> { Ok(T), Err(E) }

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            def make_pair(x: Int, y: Int) -> Pair = Pair { a: x, b: y }

            def run(node: Node, x: Int, y: Int) -> Int =
                match at(node, || make_pair(x, y)) {
                    Ok(p) => p.a + p.b,
                    Err(_) => 0 - 1,
                }
        ";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let server_port = server_addr.port() as i64;
        let server_handle = unsafe {
            serve_one_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let make_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["make_node"]))
                .unwrap()
        };
        let node = unsafe {
            make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port)
        };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let result = unsafe { run.call(client_rt.thread_ptr(), node, 13, 29) };
        assert_eq!(result, 13 + 29, "Pair with a=13, b=29 round-tripped intact");

        server_handle.join().unwrap().expect("server ok");
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// `at()` round-trips a generic enum: the closure builds a
    /// `List<Int>` server-side and the client folds it. Exercises
    /// recursive enum encoding/decoding through the wire and the
    /// uniform-ABI unboxing on the receiving side.
    #[test]
    fn at_returns_list() {
        init();
        let user_src = "
            // Server builds [1, 2, 3, 4, 5] tail-rec.
            def build(n: Int) -> List<Int> = build_acc(n, Nil)
            def build_acc(n: Int, acc: List<Int>) -> List<Int> =
                if n <= 0 { acc } else {
                    build_acc(n - 1, Cons(ListCell { head: n, tail: acc }))
                }

            def run(node: Node, n: Int) -> Int =
                match at(node, || build(n)) {
                    Ok(xs) => list_foldl(xs, 0, |acc: Int, x: Int| acc + x),
                    Err(_) => 0 - 1,
                }
        ";

        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let server_port = server_addr.port() as i64;
        let server_handle = unsafe {
            serve_one_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let tcp_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["tcp_node"]))
                .unwrap()
        };
        let node = unsafe {
            tcp_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port)
        };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let n: i64 = 5;
        let result = unsafe { run.call(client_rt.thread_ptr(), node, n) };
        // Sum 1..=n = n*(n+1)/2.
        assert_eq!(result, n * (n + 1) / 2);

        server_handle.join().unwrap().expect("server ok");
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// `at()` ships a closure that captures a **heap value** — a
    /// `List<Int>` built on the client. The server sums the list
    /// and returns the result. Phase 1B's headline behaviour:
    /// captures aren't restricted to Int anymore.
    #[test]
    fn at_with_heap_capture() {
        init();
        let user_src = "
            def build(n: Int) -> List<Int> = build_acc(n, Nil)
            def build_acc(n: Int, acc: List<Int>) -> List<Int> =
                if n <= 0 { acc } else {
                    build_acc(n - 1, Cons(ListCell { head: n, tail: acc }))
                }

            // sum_list lives on both sides; the closure captures
            // `xs` (a heap List<Int>) and the server calls sum_list
            // on it.
            def sum_list(xs: List<Int>) -> Int =
                list_foldl(xs, 0, |acc: Int, x: Int| acc + x)

            def run(node: Node, n: Int) -> Int = {
                let xs = build(n);
                match at(node, || sum_list(xs)) {
                    Ok(v) => v,
                    Err(_) => 0 - 1,
                }
            }
        ";

        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let server_port = server_addr.port() as i64;
        let server_handle = unsafe {
            serve_one_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let tcp_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["tcp_node"]))
                .unwrap()
        };
        let node = unsafe {
            tcp_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port)
        };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let n: i64 = 10;
        let result = unsafe { run.call(client_rt.thread_ptr(), node, n) };
        // Expected: 1+2+…+n = n*(n+1)/2 = 55.
        assert_eq!(result, n * (n + 1) / 2);

        server_handle.join().unwrap().expect("server ok");
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// Conway's Game of Life where **each cell's next state is its own
    /// distributed computation**. For a 5x5 grid over G generations,
    /// 25*G separate `at(node, || ...)` calls happen — every single
    /// cell ships a closure to the server, server runs it, returns
    /// 0 or 1.
    ///
    /// The grid is packed into the bottom 25 bits of an `Int`. Each
    /// closure captures the packed grid + the cell's `(x, y)` and
    /// returns the next state by counting neighbours and applying
    /// the rules. The client weaves the per-cell results back into a
    /// new packed grid, then advances to the next generation.
    ///
    /// Test pattern: a horizontal blinker. The blinker is a period-2
    /// oscillator (alternates horizontal / vertical), so live-cell
    /// count stays 3 across every generation. We assert this at every
    /// step.
    #[test]
    fn game_of_life_each_cell_is_remote() {
        init();

        // The grid: 5x5 cells, packed into bits 0..25 of an Int.
        // Bit y*5 + x = cell (x, y). Out-of-bounds reads return 0.
        //
        // `compute_cell(packed, x, y)` is the kernel: given a packed
        // snapshot and a cell coord, return that cell's next-state
        // (0 or 1). This is what each `at()` runs remotely.
        let user_src = "
            // ---- Bit-packed grid ops ----
            // The grid is 5x5 = 25 cells, each cell 1 bit. `pow(2, i)`
            // is computed via the stdlib's tail-rec `pow`.

            def bit_at(packed: Int, i: Int) -> Int = {
                let p = pow(2, i);
                let shifted = packed / p;
                shifted - (shifted / 2) * 2
            }

            def set_bit_one(packed: Int, i: Int) -> Int = {
                let cur = bit_at(packed, i);
                if cur == 1 { packed } else { packed + pow(2, i) }
            }

            def cell_at(packed: Int, x: Int, y: Int) -> Int =
                if x < 0 { 0 } else { if x >= 5 { 0 } else {
                    if y < 0 { 0 } else { if y >= 5 { 0 } else {
                        bit_at(packed, y * 5 + x)
                    }}
                }}

            def count_neighbours(packed: Int, x: Int, y: Int) -> Int =
                cell_at(packed, x - 1, y - 1) + cell_at(packed, x, y - 1) +
                cell_at(packed, x + 1, y - 1) +
                cell_at(packed, x - 1, y) + cell_at(packed, x + 1, y) +
                cell_at(packed, x - 1, y + 1) + cell_at(packed, x, y + 1) +
                cell_at(packed, x + 1, y + 1)

            // The KERNEL: this is what each remote computation runs.
            // Returns 0 or 1 — the next state of cell (x, y).
            def compute_cell(packed: Int, x: Int, y: Int) -> Int = {
                let alive = cell_at(packed, x, y);
                let n = count_neighbours(packed, x, y);
                if alive == 1 {
                    if n == 2 { 1 } else { if n == 3 { 1 } else { 0 } }
                } else {
                    if n == 3 { 1 } else { 0 }
                }
            }

            // Ship a single cell's kernel to the server. The closure
            // captures `packed`, `x`, `y` (three Int captures). The
            // server's JIT re-runs the same `compute_cell` def.
            def remote_cell(node: Node, packed: Int, x: Int, y: Int) -> Int =
                match at(node, || compute_cell(packed, x, y)) {
                    Ok(v) => v,
                    Err(_) => 0,
                }

            // Compute one generation by issuing 25 independent `at()`
            // calls, one per cell. Build up the next packed Int by OR'ing
            // in the bit for each live cell.
            def distributed_step(node: Node, packed: Int) -> Int =
                distributed_step_acc(node, packed, 0, 0)

            def distributed_step_acc(node: Node, src: Int, i: Int, acc: Int) -> Int =
                if i >= 25 { acc } else {
                    let x = i - (i / 5) * 5;
                    let y = i / 5;
                    let bit = remote_cell(node, src, x, y);
                    let next = if bit == 1 { set_bit_one(acc, i) } else { acc };
                    distributed_step_acc(node, src, i + 1, next)
                }

            // Population count across the whole grid.
            def live_count(packed: Int) -> Int = live_count_acc(packed, 0, 0)

            def live_count_acc(packed: Int, i: Int, acc: Int) -> Int =
                if i >= 25 { acc } else {
                    live_count_acc(packed, i + 1, acc + bit_at(packed, i))
                }

            // Public entry points — each generation is one round of 25
            // remote calls. Returns the new packed grid.
            def step_one_gen(node: Node, packed: Int) -> Int =
                distributed_step(node, packed)

            // Horizontal blinker at row 2 (y=2), columns 1..=3 (x=1,2,3).
            // Bits: y*5+x → 11, 12, 13. Packed = 2^11 + 2^12 + 2^13.
            def initial_blinker() -> Int = pow(2, 11) + pow(2, 12) + pow(2, 13)
        ";

        let src = format!("{}\n{}", crate::stdlib::SOURCE, user_src);

        // Typecheck full program before either side compiles.
        {
            let m = parse_module(&src).expect("parse");
            let r = resolve_module(&m).expect("resolve");
            let mut cache = crate::typecheck::TypeCache::new();
            crate::typecheck::typecheck_module(&r, &mut cache)
                .expect("typecheck stdlib + user");
        }

        // 4 generations × 25 cells = 100 `at()` calls.
        let generations: usize = 4;

        // Server. Use persistent serve so the client can reuse a
        // single cached TCP connection across all 100 at() calls
        // (the default behavior of `ai_net_at` now).
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, &src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let server_port = server_addr.port() as i64;
        let _server_handle = unsafe {
            serve_persistent_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        // Client.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&src).unwrap();
        let r = resolve_module(&m).unwrap();
        let resolver_binding = r.at_binding.as_ref().expect("at_binding populated");
        let rt_binding = build_at_runtime_binding(&client_rt, resolver_binding)
            .expect("runtime binding built");
        install_current_at_binding(&rt_binding);

        let tcp_node = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(
                    *mut Thread,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> *mut u8>(&crate::codegen::def_symbol(&names["tcp_node"]))
                .unwrap()
        };
        let node = unsafe {
            tcp_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port)
        };

        let initial_blinker = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                    &crate::codegen::def_symbol(&names["initial_blinker"]),
                )
                .unwrap()
        };
        let live_count = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["live_count"]),
                )
                .unwrap()
        };
        let step_one_gen = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["step_one_gen"]),
                )
                .unwrap()
        };

        // Start state: blinker, 3 live cells.
        let mut grid = unsafe { initial_blinker.call(client_rt.thread_ptr()) };
        assert_eq!(unsafe { live_count.call(client_rt.thread_ptr(), grid) }, 3);

        // Run G generations. The live count is invariant for a blinker
        // (period-2 oscillator) — 3 every step. Each step is 25
        // distributed at() calls.
        let mut counts = vec![3i64];
        for _ in 0..generations {
            grid = unsafe { step_one_gen.call(client_rt.thread_ptr(), node, grid) };
            let c = unsafe { live_count.call(client_rt.thread_ptr(), grid) };
            counts.push(c);
        }

        // Blinker invariant.
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(*c, 3, "gen {} live count expected 3, got {}", i, c);
        }

        // Also: after an even number of generations, the grid bits are
        // identical to the start (blinker has period 2).
        let restart = unsafe { initial_blinker.call(client_rt.thread_ptr()) };
        // Need to re-fetch grid hash equivalent — but `grid` is the bit
        // pattern (an Int) returned through unbox. Compare directly.
        assert_eq!(grid, restart, "after {} gens blinker should match initial", generations);

        // Persistent server runs forever; detach by simply not joining.

        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    // ---- C-FFI / wire safety: `Ptr` must not cross the at() boundary ----

    fn typecheck_src(user_src: &str) -> Result<(), crate::typecheck::TypeError> {
        let m = parse_module(user_src).expect("parse");
        let r = resolve_module(&m).expect("resolve");
        let mut cache = crate::typecheck::TypeCache::new();
        crate::typecheck::typecheck_module(&r, &mut cache).map(|_| ())
    }

    const AT_PRELUDE: &str = "
        struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
        enum Failure { Unreachable(Node), Crashed(Node), CodeMissing(Node), Cancelled(Node) }
        enum Result<T, E> { Ok(T), Err(E) }
    ";

    /// A thunk shipped via `at(...)` may not RETURN a `Ptr`: it is a raw
    /// local address, garbage on another node.
    #[test]
    fn at_thunk_returning_ptr_is_rejected() {
        init();
        let src = format!(
            "{}
             def run(node: Node) -> Int =
                match at(node, || ptr_null()) {{
                    Ok(p) => 0,
                    Err(e) => 0 - 1
                }}",
            AT_PRELUDE
        );
        let err = typecheck_src(&src).expect_err("Ptr-returning thunk must be rejected");
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("Ptr") && msg.to_lowercase().contains("return"),
            "error should explain the Ptr return ban, got: {}",
            msg
        );
    }

    /// A thunk shipped via `at(...)` may not CAPTURE a `Ptr`: it would
    /// travel as a plain Int and deref to garbage on the remote node.
    #[test]
    fn at_thunk_capturing_ptr_is_rejected() {
        init();
        let src = format!(
            "{}
             def run(node: Node) -> Int = {{
                let p = ptr_null();
                match at(node, || ptr_is_null(p)) {{
                    Ok(n) => n,
                    Err(e) => 0 - 1
                }}
             }}",
            AT_PRELUDE
        );
        let err = typecheck_src(&src).expect_err("Ptr-capturing thunk must be rejected");
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("Ptr") && msg.to_lowercase().contains("captur"),
            "error should explain the Ptr capture ban, got: {}",
            msg
        );
    }

    /// Sanity: an ordinary `Int`-returning thunk with no `Ptr` anywhere
    /// still typechecks (the guard is not over-broad).
    #[test]
    fn at_thunk_without_ptr_is_accepted() {
        init();
        let src = format!(
            "{}
             def work(x: Int) -> Int = x + 1
             def run(node: Node, x: Int) -> Int =
                match at(node, || work(x)) {{
                    Ok(n) => n,
                    Err(e) => 0 - 1
                }}",
            AT_PRELUDE
        );
        typecheck_src(&src).expect("a Ptr-free thunk must still typecheck");
    }
}
