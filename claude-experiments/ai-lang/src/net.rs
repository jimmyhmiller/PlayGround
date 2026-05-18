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
use crate::runtime::{Runtime, Thread, ai_gc_alloc_closure, ai_gc_box_int};
use crate::wire::{WireError, WireValue, decode_value, encode_value};

// =============================================================================
// Frame protocol
// =============================================================================

pub const KIND_CALL: u8 = 0;
pub const KIND_RESULT: u8 = 1;
pub const KIND_NEED_CODE: u8 = 2;
pub const KIND_CODE: u8 = 3;

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
}

impl ItemKind {
    pub fn from_u8(b: u8) -> Result<Self, NetError> {
        match b {
            0 => Ok(ItemKind::Def),
            1 => Ok(ItemKind::Lambda),
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
pub fn read_frame<R: Read>(stream: &mut R) -> Result<Vec<u8>, NetError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
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
    stream: &mut TcpStream,
) -> Result<(), NetError> {
    let body = read_frame(stream)?;
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty frame body"));
    }
    let kind = body[0];
    match kind {
        KIND_CALL => {
            let payload = &body[1..];
            let (value, _) = unsafe { decode_value(runtime, payload)? };
            let closure_ptr = match value {
                WireValue::Heap(p) => p,
                WireValue::Int(_) => {
                    return Err(NetError::ProtocolViolation(
                        "Call payload must be a heap closure",
                    ));
                }
            };
            let result = unsafe { invoke_zero_arg_closure(runtime, closure_ptr)? };
            let mut reply = Vec::with_capacity(9);
            reply.push(KIND_RESULT);
            unsafe { encode_value(runtime, WireValue::Int(result), &mut reply)? };
            write_frame(stream, &reply)?;
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
    stream: &mut TcpStream,
) -> Result<(), NetError> {
    let body = read_frame(stream)?;
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
                let result = unsafe { invoke_zero_arg_closure(rt, closure_ptr)? };
                let mut reply = Vec::with_capacity(9);
                reply.push(KIND_RESULT);
                unsafe { encode_value(rt, WireValue::Int(result), &mut reply)? };
                write_frame(stream, &reply)?;
                return Ok(());
            }
            Err(WireError::MissingShape(missing)) => {
                if round == MAX_SERVER_INSTALL_ROUNDS {
                    return Err(NetError::CodeFetchDepthExceeded);
                }
                // Ask client for the missing code.
                let req = encode_need_code(&[missing]);
                write_frame(stream, &req)?;
                // Wait for Code response.
                let resp = read_frame(stream)?;
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
/// final `Result`, return the integer value.
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
) -> Result<i64, NetError> {
    let mut stream = TcpStream::connect(addr)?;
    unsafe { at_remote_on_stream(runtime, knowledge_base, &mut stream, closure_ptr) }
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
) -> Result<i64, NetError> {
    // Two-phase to keep the RefCell borrow short and to allow eviction
    // on error without re-borrowing inside the call path.
    let result: Result<i64, NetError> = AT_CONN_CACHE.with(|c| {
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
        unsafe { at_remote_on_stream(runtime, knowledge_base, stream, closure_ptr) }
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
pub unsafe fn at_remote_on_stream(
    runtime: &Runtime,
    knowledge_base: &KnowledgeBase,
    stream: &mut TcpStream,
    closure_ptr: *const u8,
) -> Result<i64, NetError> {
    // Build and send the Call frame.
    let mut body = Vec::with_capacity(1024);
    body.push(KIND_CALL);
    unsafe { encode_value(runtime, WireValue::Heap(closure_ptr), &mut body)? };
    write_frame(stream, &body)?;

    // Track which hashes we've already shipped this session so we don't
    // re-ship the same bytes if the server asks twice (defensive — a
    // good server shouldn't, but cheap to guard).
    let mut already_shipped: std::collections::HashSet<Hash> =
        std::collections::HashSet::new();

    for round in 0..=MAX_CODE_FETCH_ROUNDS {
        let reply = read_frame(stream)?;
        if reply.is_empty() {
            return Err(NetError::ProtocolViolation("empty reply body"));
        }
        let kind = reply[0];
        match kind {
            KIND_RESULT => {
                let payload = &reply[1..];
                let (value, _) = unsafe { decode_value(runtime, payload)? };
                return match value {
                    WireValue::Int(n) => Ok(n),
                    _ => Err(NetError::ProtocolViolation(
                        "v1 Result must be an Int (zero-arg closure return type)",
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
                for h in deps {
                    if already_shipped.contains(&h) {
                        continue;
                    }
                    let (kind, bytes) = knowledge_base
                        .lookup(&h)
                        .ok_or(NetError::MissingFromKnowledgeBase(h))?;
                    items.push((*kind, h, bytes.clone()));
                    already_shipped.insert(h);
                }
                let payload = encode_code(&items);
                write_frame(stream, &payload)?;
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
                NetError::Io(_) => &binding.unreachable,
                NetError::UnknownCode(_)
                | NetError::MissingFromKnowledgeBase(_)
                | NetError::CodeFetchDepthExceeded => &binding.code_missing,
                NetError::Wire(_)
                | NetError::FrameTooLarge(_)
                | NetError::BadKind(_)
                | NetError::ProtocolViolation(_)
                | NetError::ProtocolViolationOwned(_)
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

unsafe fn build_ok(
    thread: *mut Thread,
    b: &AtRuntimeBinding,
    value: i64,
) -> *const u8 {
    if b.ok.payload_is_pointer {
        // Generic `Result<T, E>`: the Ok variant's payload slot is a
        // pointer (uniform representation). Box the i64 into a
        // `BoxedInt` so the heap holds a real pointer, and the
        // caller's match arm can unbox it back to an Int.
        let boxed = unsafe { ai_gc_box_int(thread, value) };
        unsafe { alloc_variant(thread, &b.ok, None, Some(boxed as *const u8)) }
    } else {
        // Monomorphic `Result`: payload is a raw Int in the heap obj.
        unsafe { alloc_variant(thread, &b.ok, Some(value), None) }
    }
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

// =============================================================================
// Closure invocation
// =============================================================================

/// Invoke a zero-arg closure: load its `code_hash`, look up the JIT'd
/// entry point in the runtime's code table, indirect-call.
///
/// # Safety
/// `closure_ptr` must be a valid heap closure. The looked-up function
/// pointer must match the `unsafe extern "C" fn(*mut Thread, *const u8) -> i64`
/// signature (true for any zero-arg lambda our codegen emits).
unsafe fn invoke_zero_arg_closure(
    runtime: &Runtime,
    closure_ptr: *const u8,
) -> Result<i64, NetError> {
    // code_hash sits at offset Full::SIZE within the closure object.
    let mut hash_bytes = [0u8; 32];
    unsafe {
        core::ptr::copy_nonoverlapping(
            closure_ptr.add(<Full as crate::gc::ObjHeader>::SIZE),
            hash_bytes.as_mut_ptr(),
            32,
        );
    }
    let code_hash = Hash(hash_bytes);
    let fn_ptr = runtime
        .code_table
        .lookup(&code_hash)
        .ok_or(NetError::UnknownCode(code_hash))?;
    // Uniform closure ABI: lifted lambdas always return a pointer
    // (BoxedInt for Int-returning thunks). The at() thunks are
    // `fn() -> Int`, so unbox the returned BoxedInt before returning
    // an i64 to the caller.
    let lambda: unsafe extern "C" fn(*mut Thread, *const u8) -> *mut u8 =
        unsafe { core::mem::transmute(fn_ptr) };
    let ret_ptr = unsafe { lambda(runtime.thread_ptr(), closure_ptr) };
    let result = unsafe { crate::runtime::ai_gc_unbox_int(ret_ptr) };
    Ok(result)
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
        let result = unsafe {
            at_remote(&client_rt, &kb, server_addr, closure as *const u8).unwrap()
        };
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
        let result = unsafe {
            at_remote(&client_rt, &kb, server_addr, closure as *const u8).unwrap()
        };
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
        assert_eq!(server_rt.shape_by_type_id.len(), 0);

        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();

        // --- CLIENT side (spawned thread): full program. ---
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
            unsafe {
                at_remote(&client_rt, &kb, server_addr, closure_ptr as *const u8).unwrap()
            }
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

    /// End-to-end: the lang program itself calls `at(...)` and the
    /// runtime constructs a `Result::Ok(n)` heap value. The program's
    /// match extracts `n`; we assert the integer flows through.
    #[test]
    fn at_returns_ok_on_success() {
        init();
        // Shared source for both sides. Result/Failure/Node are
        // user-defined; `at(...)` lowers to a builtin that returns
        // `Result`. The lang code matches it to recover the Int.
        let src = "
            struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
            enum Failure {
                Unreachable(Node),
                Crashed(Node),
                CodeMissing(Node),
                Cancelled(Node),
            }
            enum Result { Ok(Int), Err(Failure) }

            def work(x: Int) -> Int = x * x + 7

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            def run(node: Node, x: Int) -> Int =
                match at(node, || work(x)) {
                    Ok(n) => n,
                    Err(_) => 0 - 1,
                }
        ";

        // Server side: same source so it can run the closure.
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_addr = listener.local_addr().unwrap();
        let server_port = server_addr.port() as i64;
        let server_handle = unsafe {
            serve_one_in_thread(Arc::new(RuntimeHandle(server_rt)), listener)
        };

        // Client side.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);

        // Install the per-thread runtime + KB + at-binding.
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let resolver_binding = r.at_binding.as_ref().expect("at_binding populated");
        let rt_binding = build_at_runtime_binding(&client_rt, resolver_binding)
            .expect("runtime binding built");
        install_current_at_binding(&rt_binding);

        // Build the Node and call `run`.
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
        let result = unsafe { run.call(client_rt.thread_ptr(), node, 6) };
        assert_eq!(result, 6 * 6 + 7);

        // Server thread completed cleanly.
        server_handle.join().unwrap().expect("server ok");

        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// End-to-end with a **generic** `Result<T, E>` user-defined
    /// enum. The runtime boxes the i64 return value into a
    /// `BoxedInt`; the user's match-arm extraction unboxes it.
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
            enum Result { Ok(Int), Err(Failure) }

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
        let user_src = "
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

}
