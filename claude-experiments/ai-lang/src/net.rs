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
//! ## Failure model — errors are values
//!
//! Every way a remote call can fail surfaces as a `Failure` enum value
//! in the caller's `Result<T, Failure>` — never a hang, never a dead
//! process:
//!
//! - **Unreachable** — connect refused/reset, torn connection.
//! - **TimedOut** (optional variant; else `Unreachable`) — the connect
//!   timeout or whole-call deadline expired (see [`at_call_deadline`]).
//! - **Crashed** — the thunk failed ON the node. Most importantly a
//!   panic: it propagates out of JIT'd code as a pending-panic value,
//!   is taken at the serve boundary, and is reported in a `KIND_ERR`
//!   frame — the node keeps serving; one bad request fails one request.
//! - **CodeMissing** — the code-fetch handshake couldn't materialize
//!   the shipped closure on the node.
//!
//! Pure thunks (per effects inference) are transparently retried on
//! transient failures ([`at_pure_retries`]); stateful thunks are never
//! re-sent (at-most-once for effects). `at_async(node, thunk)` runs the
//! same exchange from a background thread, returning a
//! `ThreadHandle<Result<T, Failure>>` that `join` awaits.
//!
//! ## v1 restrictions
//!
//! - **Zero-arg closures only.** A general `at(node, |x| …)` would need
//!   to ship arguments too; we can extend the protocol later.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;

use crate::codegen::{IncrementalJit, ShapeMeta};
use crate::gc::{Full, Heap, ThreadState, TypeInfo};
use crate::hash::Hash;
use crate::knowledge::KnowledgeBase;
use crate::resolve::AtBinding as ResolverAtBinding;
use crate::runtime::{Runtime, Thread, ThreadContext, ai_gc_alloc_closure};
use crate::wire::{WireError, WireValue, decode_value, encode_value};

// =============================================================================
// Frame protocol
// =============================================================================

pub const KIND_CALL: u8 = 0;
pub const KIND_RESULT: u8 = 1;
pub const KIND_NEED_CODE: u8 = 2;
pub const KIND_CODE: u8 = 3;
/// A deploy request: install a new version, verify it (interface +
/// state-shape continuity), flip bindings. See [`crate::deploy`].
pub const KIND_DEPLOY: u8 = 4;
/// Successful deploy/rollback reply carrying the report.
pub const KIND_DEPLOY_OK: u8 = 5;
/// Refused deploy/rollback/invoke reply carrying the error message.
/// A refusal is a *reply*, not a connection error: the node is
/// untouched and the connection stays usable.
pub const KIND_DEPLOY_ERR: u8 = 6;
/// Flip a binding back to its previous version.
pub const KIND_ROLLBACK: u8 = 7;
/// Invoke a deployed binding by name with Int args (v1). The reply is
/// `KIND_INVOKE_RESULT` + 8-byte BE i64.
pub const KIND_INVOKE: u8 = 8;
pub const KIND_INVOKE_RESULT: u8 = 9;
/// Connection auth: first frame on a token-protected node must be
/// `KIND_AUTH` + token bytes; the node echoes `KIND_AUTH` on success
/// and closes the connection on mismatch.
pub const KIND_AUTH: u8 = 10;
/// Error reply: the node failed to run the shipped thunk and is
/// REPORTING it instead of dropping the connection. Body:
/// `[KIND_ERR][code:u8][message:utf8...]`. The connection stays usable
/// — the node wrote a complete reply and is ready for the next Call.
/// The client maps `code` onto the user's `Failure` enum, so a remote
/// crash arrives as a VALUE (`Err(Crashed(node))`), exactly like every
/// other failure in the language.
pub const KIND_ERR: u8 = 11;

/// `KIND_ERR` code: the thunk (or its decode/invoke/encode) failed on
/// the node — most importantly a thunk PANIC, which propagates out of
/// JIT'd code as a pending-panic value, is taken at the serve boundary,
/// and reported here. The node keeps serving.
pub const ERR_CRASHED: u8 = 1;
/// `KIND_ERR` code: the node couldn't obtain the code for the shipped
/// closure (unknown hash, code-fetch depth exceeded, install failure).
pub const ERR_CODE_MISSING: u8 = 2;
/// `KIND_ERR` code: the node refused or abandoned the call before
/// running it (reserved; maps to `Failure::Cancelled`).
pub const ERR_CANCELLED: u8 = 3;
/// `KIND_ERR` code: the node gave up on the call after its own
/// deadline (reserved; maps to `Failure::TimedOut` when declared).
pub const ERR_TIMED_OUT: u8 = 4;

/// Build a `KIND_ERR` frame body.
pub fn encode_err_frame(code: u8, msg: &str) -> Vec<u8> {
    let mut body = Vec::with_capacity(2 + msg.len());
    body.push(KIND_ERR);
    body.push(code);
    body.extend_from_slice(msg.as_bytes());
    body
}

// =============================================================================
// at() configuration — timeouts and retries
// =============================================================================
//
// Slow and dead must be distinguishable: without deadlines, a hung node
// (or a shipped thunk stuck in a loop) blocks the caller forever and no
// `Failure` value ever materializes. Defaults are finite and process-
// global; override via the setters or environment variables at startup:
//
//   AI_LANG_AT_CONNECT_MS   TCP connect timeout      (default 5000, 0 = none)
//   AI_LANG_AT_DEADLINE_MS  whole-call deadline      (default 30000, 0 = none)
//   AI_LANG_AT_RETRIES      max auto-retries of PURE thunks on transient
//                           failures (default 2, 0 = never retry)

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

static AT_CONNECT_TIMEOUT_MS: AtomicU64 = AtomicU64::new(u64::MAX);
static AT_CALL_DEADLINE_MS: AtomicU64 = AtomicU64::new(u64::MAX);
static AT_PURE_RETRIES: AtomicU64 = AtomicU64::new(u64::MAX);

fn env_ms(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

/// Lazily resolve a config cell: `u64::MAX` means "unset — read the env
/// var (or default) and cache it".
fn config_cell(cell: &AtomicU64, env: &str, default: u64) -> u64 {
    let v = cell.load(AtomicOrdering::Relaxed);
    if v != u64::MAX {
        return v;
    }
    let resolved = env_ms(env, default);
    cell.store(resolved, AtomicOrdering::Relaxed);
    resolved
}

/// TCP connect timeout for `at()` calls. `None` = wait forever (0).
pub fn at_connect_timeout() -> Option<Duration> {
    match config_cell(&AT_CONNECT_TIMEOUT_MS, "AI_LANG_AT_CONNECT_MS", 5_000) {
        0 => None,
        ms => Some(Duration::from_millis(ms)),
    }
}

/// Whole-call deadline for one `at()` exchange (send + NeedCode rounds +
/// result). `None` = wait forever (0).
///
/// Under `AI_LANG_GC_STRESS` (collect on every allocation) the default
/// is NO deadline: stress mode slows execution by orders of magnitude,
/// and a wall-clock deadline would convert GC-correctness runs into
/// timing-flake runs. An explicit `AI_LANG_AT_DEADLINE_MS` still wins.
pub fn at_call_deadline() -> Option<Duration> {
    let default = if std::env::var_os("AI_LANG_GC_STRESS").is_some() {
        0
    } else {
        30_000
    };
    match config_cell(&AT_CALL_DEADLINE_MS, "AI_LANG_AT_DEADLINE_MS", default) {
        0 => None,
        ms => Some(Duration::from_millis(ms)),
    }
}

/// How many times `at()` may transparently re-send a PURE thunk after a
/// transient failure (unreachable / timed out). Stateful thunks are
/// never auto-retried (at-most-once is preserved for effects).
pub fn at_pure_retries() -> u64 {
    config_cell(&AT_PURE_RETRIES, "AI_LANG_AT_RETRIES", 2)
}

/// Set the connect timeout (ms). 0 disables the timeout.
pub fn set_at_connect_timeout_ms(ms: u64) {
    AT_CONNECT_TIMEOUT_MS.store(ms.min(u64::MAX - 1), AtomicOrdering::Relaxed);
}

/// Set the whole-call deadline (ms). 0 disables the deadline.
pub fn set_at_call_deadline_ms(ms: u64) {
    AT_CALL_DEADLINE_MS.store(ms.min(u64::MAX - 1), AtomicOrdering::Relaxed);
}

/// Set the pure-thunk auto-retry budget. 0 disables retries.
pub fn set_at_pure_retries(n: u64) {
    AT_PURE_RETRIES.store(n.min(u64::MAX - 1), AtomicOrdering::Relaxed);
}

/// Forget any programmatic at() config overrides: the next reads
/// re-resolve from the environment (or the built-in defaults). Tests
/// use this to restore the ambient config rather than guessing it.
pub fn reset_at_config() {
    AT_CONNECT_TIMEOUT_MS.store(u64::MAX, AtomicOrdering::Relaxed);
    AT_CALL_DEADLINE_MS.store(u64::MAX, AtomicOrdering::Relaxed);
    AT_PURE_RETRIES.store(u64::MAX, AtomicOrdering::Relaxed);
}


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

/// Append an item list (`u32 BE count + (u8 kind + 32-byte Hash +
/// u32 BE len + len bytes) × count`) to `out`. Shared by `Code` frames
/// and deploy frames.
pub fn encode_items(items: &[(ItemKind, Hash, Vec<u8>)], out: &mut Vec<u8>) {
    let n: u32 = items.len().try_into().expect("too many items");
    out.extend_from_slice(&n.to_be_bytes());
    for (k, h, bytes) in items {
        out.push(k.as_u8());
        out.extend_from_slice(h.as_bytes());
        let blen: u32 = bytes.len().try_into().expect("item too large");
        out.extend_from_slice(&blen.to_be_bytes());
        out.extend_from_slice(bytes);
    }
}

/// Decode an item list written by [`encode_items`], starting at `*pos`;
/// advances `*pos` past it.
pub fn decode_items(
    body: &[u8],
    pos: &mut usize,
) -> Result<Vec<(ItemKind, Hash, Vec<u8>)>, NetError> {
    if *pos + 4 > body.len() {
        return Err(NetError::ProtocolViolation(
            "item list truncated before count",
        ));
    }
    let count = u32::from_be_bytes([body[*pos], body[*pos + 1], body[*pos + 2], body[*pos + 3]])
        as usize;
    *pos += 4;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if *pos + 1 + 32 + 4 > body.len() {
            return Err(NetError::ProtocolViolation(
                "item list truncated mid-entry header",
            ));
        }
        let kind = ItemKind::from_u8(body[*pos])?;
        *pos += 1;
        let mut hash_buf = [0u8; 32];
        hash_buf.copy_from_slice(&body[*pos..*pos + 32]);
        *pos += 32;
        let blen = u32::from_be_bytes([
            body[*pos],
            body[*pos + 1],
            body[*pos + 2],
            body[*pos + 3],
        ]) as usize;
        *pos += 4;
        if *pos + blen > body.len() {
            return Err(NetError::ProtocolViolation(
                "item list truncated before item payload",
            ));
        }
        let bytes = body[*pos..*pos + blen].to_vec();
        *pos += blen;
        out.push((kind, Hash(hash_buf), bytes));
    }
    Ok(out)
}

/// Encode a `Code` payload. Each entry is `(item_kind, hash, canonical_bytes)`.
///
/// Layout: u32 BE count + (u8 kind + 32 byte Hash + u32 BE len + len bytes) × count.
pub fn encode_code(items: &[(ItemKind, Hash, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + items.len() * (1 + 32 + 4 + 32));
    out.push(KIND_CODE);
    encode_items(items, &mut out);
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
    let mut pos = 1usize;
    let out = decode_items(body, &mut pos)?;
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
    /// The node refused a deploy / rollback / invoke. The node is
    /// untouched and the connection stays usable; the message is the
    /// node-side [`crate::deploy::DeployError`] rendered to text.
    DeployRefused(String),
    /// The shipped thunk PANICKED on this side (the serve boundary took
    /// the pending-panic value after invoking it). The server converts
    /// this into a `KIND_ERR`/`ERR_CRASHED` reply; it never escapes to
    /// a client as this variant.
    HandlerPanicked(String),
    /// The node reported a failure via a `KIND_ERR` frame: the call
    /// reached the node, the node could not complete it, and the
    /// connection is still healthy. `code` is one of the `ERR_*`
    /// constants; `msg` is the node's human-readable detail (e.g. the
    /// remote panic message).
    Remote { code: u8, msg: String },
    /// The call's deadline expired (connect or any read/write leg of
    /// the exchange). Slow and dead are indistinguishable from here:
    /// the request may or may not have executed on the node, so a
    /// stateful thunk must NOT be blindly re-sent. Maps to the user's
    /// `Failure::TimedOut(node)` when declared, else `Unreachable`.
    TimedOut(String),
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
            NetError::DeployRefused(msg) => write!(f, "node refused: {}", msg),
            NetError::HandlerPanicked(msg) => write!(f, "handler panicked: {}", msg),
            NetError::TimedOut(phase) => write!(f, "timed out: {}", phase),
            NetError::Remote { code, msg } => {
                let kind = match *code {
                    ERR_CRASHED => "crashed",
                    ERR_CODE_MISSING => "code missing",
                    ERR_CANCELLED => "cancelled",
                    ERR_TIMED_OUT => "timed out",
                    _ => "unknown failure",
                };
                write!(f, "node reported {}: {}", kind, msg)
            }
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
    /// Bound the next read/write legs to `d` (None = no bound). A leg
    /// exceeding the bound fails with an `Io` error whose kind is
    /// `WouldBlock`/`TimedOut`; the at() loop maps that to
    /// [`NetError::TimedOut`]. Default: no-op (in-process channels and
    /// tests don't time out).
    fn set_io_deadline(&mut self, _d: Option<std::time::Duration>) {}
}

impl Channel for TcpStream {
    fn read_frame(&mut self) -> Result<Vec<u8>, NetError> {
        read_frame(self)
    }
    fn write_frame(&mut self, body: &[u8]) -> Result<(), NetError> {
        write_frame(self, body)
    }
    fn set_io_deadline(&mut self, d: Option<std::time::Duration>) {
        // A zero Duration would mean "no timeout" to setsockopt — clamp
        // to 1ms so an already-expired deadline still errors promptly.
        let d = d.map(|d| d.max(std::time::Duration::from_millis(1)));
        let _ = self.set_read_timeout(d);
        let _ = self.set_write_timeout(d);
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
    park_home_on_foreign_serve(runtime);
    // Wait for the request in a BLOCKED region so a STW collection on
    // another thread scans us in place rather than hanging.
    let body = {
        let _blocked = BlockedRegion::enter(runtime);
        channel.read_frame()
    }?;
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty frame body"));
    }
    let kind = body[0];
    match kind {
        KIND_CALL => {
            match unsafe { serve_call_payload(runtime, &body[1..]) } {
                Ok(reply) => {
                    channel.write_frame(&reply)?;
                    Ok(())
                }
                Err(e) => match err_reply_code(&e) {
                    // Errors are values end-to-end: report the failure
                    // in a complete KIND_ERR reply and KEEP SERVING —
                    // one bad request fails one request, not the node.
                    Some(code) => {
                        channel.write_frame(&encode_err_frame(code, &format!("{}", e)))?;
                        Ok(())
                    }
                    // Transport-level: the stream state is unknown;
                    // surface the error so the caller drops the
                    // connection.
                    None => Err(e),
                },
            }
        }
        other => Err(NetError::BadKind(other)),
    }
}

/// The body of a `Call`: decode the shipped thunk, invoke it, encode the
/// reply frame. Pure with respect to the channel — callers write the
/// returned frame (or a `KIND_ERR` report if this fails).
///
/// # Safety
/// Same contract as [`serve_one`].
unsafe fn serve_call_payload(
    runtime: &Runtime,
    payload: &[u8],
) -> Result<Vec<u8>, NetError> {
    // The Call payload is an encoded closure: `[1 (Closure kind)]
    // [code_hash 32B] [n_captures] [captures...]`. Read the lambda
    // code hash from that prefix (no decode needed) to decide
    // cacheability.
    //
    // Cache key: blake3 of the whole payload (code hash + all
    // captures). Identical bytes → identical computation, by the
    // content-addressed property — SO LONG AS the thunk is pure. A
    // thunk that transitively touches node `state` is NOT pure:
    // caching it would skip its mutation on a repeated identical
    // call. Such thunks bypass the cache entirely.
    let cacheable = if payload.len() >= 33 && payload[0] == 1 {
        let mut h = [0u8; 32];
        h.copy_from_slice(&payload[1..33]);
        !runtime.is_stateful(&Hash(h))
    } else {
        // Non-closure payload: leave it to decode_value to reject;
        // don't cache something we can't classify.
        false
    };
    let cache_key = Hash::of_bytes(payload);
    if cacheable {
        if let Some(cached_reply) = runtime.try_cached_result(&cache_key) {
            return Ok(cached_reply);
        }
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
    let thread = serve_thread_ptr(runtime);
    let result_ptr = unsafe { invoke_zero_arg_closure(runtime, thread, closure_ptr)? };
    let mut reply = Vec::with_capacity(64);
    reply.push(KIND_RESULT);
    unsafe { encode_value(runtime, WireValue::Heap(result_ptr), &mut reply)? };
    if cacheable {
        runtime.store_cached_result(cache_key, reply.clone());
    }
    Ok(reply)
}

/// Map a handler-stage failure onto a `KIND_ERR` code, or `None` when
/// the failure is transport-level (stream state unknown — the server
/// should drop the connection instead of replying).
///
/// Reportable failures all share one property: the request frame was
/// fully read and nothing has been written, so a complete error reply
/// keeps the connection in a clean request/reply state.
fn err_reply_code(e: &NetError) -> Option<u8> {
    match e {
        // A panicking thunk — THE reason errors must be values: the
        // caller gets `Err(Crashed(node))`, the node keeps serving.
        NetError::HandlerPanicked(_) => Some(ERR_CRASHED),
        // The node can't materialize the shipped code.
        NetError::UnknownCode(_)
        | NetError::CodeFetchDepthExceeded
        | NetError::MissingFromKnowledgeBase(_)
        | NetError::InstallFailed(_) => Some(ERR_CODE_MISSING),
        // The call's payload was unusable (malformed closure bytes,
        // non-closure payload) or its result couldn't be encoded. The
        // frame boundary is intact, so report it.
        NetError::Wire(_)
        | NetError::ProtocolViolation(_)
        | NetError::ProtocolViolationOwned(_) => Some(ERR_CRASHED),
        // Transport / framing: don't guess, drop.
        NetError::Io(_)
        | NetError::ConnectionClosed
        | NetError::FrameTooLarge(_)
        | NetError::BadKind(_)
        | NetError::DeployRefused(_)
        | NetError::TimedOut(_)
        | NetError::Remote { .. } => None,
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
    park_home_on_foreign_serve(rt);
    let body = {
        let _blocked = BlockedRegion::enter(rt);
        channel.read_frame()
    }?;
    if body.is_empty() {
        return Err(NetError::ProtocolViolation("empty frame body"));
    }
    if body[0] != KIND_CALL {
        return Err(NetError::BadKind(body[0]));
    }
    unsafe { handle_call_frame(rt, jit, channel, &body) }
}

/// Handle an already-read `Call` frame (`body[0] == KIND_CALL`),
/// including the NeedCode/Code fetch handshake. Shared between
/// [`serve_with_install`] and the deploy node's dispatcher
/// ([`crate::deploy::serve_deploy_turn`]).
///
/// # Safety
/// Same contract as [`serve_with_install`].
pub unsafe fn handle_call_frame<'ctx>(
    rt: &mut Runtime,
    jit: &mut IncrementalJit<'ctx>,
    channel: &mut dyn Channel,
    body: &[u8],
) -> Result<(), NetError> {
    park_home_on_foreign_serve(rt);
    let payload: Vec<u8> = body[1..].to_vec();

    // Run the decode/fetch/invoke pipeline; any reportable failure
    // becomes a complete `KIND_ERR` reply (errors are values — the
    // caller gets a `Failure`, the node keeps serving). Transport
    // failures surface to the caller, which drops the connection.
    match unsafe { handle_call_payload(rt, jit, channel, &payload) } {
        Ok(reply) => {
            channel.write_frame(&reply)?;
            Ok(())
        }
        Err(e) => match err_reply_code(&e) {
            Some(code) => {
                channel.write_frame(&encode_err_frame(code, &format!("{}", e)))?;
                Ok(())
            }
            None => Err(e),
        },
    }
}

/// Decode (fetching missing code via the NeedCode handshake as needed),
/// invoke, and encode the reply frame for one Call payload. The channel
/// is used ONLY for the NeedCode/Code exchange; the final reply (or
/// error report) is written by the caller.
///
/// # Safety
/// Same contract as [`serve_with_install`].
unsafe fn handle_call_payload<'ctx>(
    rt: &mut Runtime,
    jit: &mut IncrementalJit<'ctx>,
    channel: &mut dyn Channel,
    payload: &[u8],
) -> Result<Vec<u8>, NetError> {
    // Retry-on-MissingShape loop.
    for round in 0..=MAX_SERVER_INSTALL_ROUNDS {
        let result = unsafe { decode_value(rt, payload) };
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
                let thread = serve_thread_ptr(rt);
                let result_ptr = unsafe { invoke_zero_arg_closure(rt, thread, closure_ptr)? };
                let mut reply = Vec::with_capacity(64);
                reply.push(KIND_RESULT);
                unsafe { encode_value(rt, WireValue::Heap(result_ptr), &mut reply)? };
                return Ok(reply);
            }
            Err(WireError::MissingShape(missing)) => {
                if round == MAX_SERVER_INSTALL_ROUNDS {
                    return Err(NetError::CodeFetchDepthExceeded);
                }
                // Ask client for the missing code.
                let req = encode_need_code(&[missing]);
                channel.write_frame(&req)?;
                // Wait for Code response (blocking region: client may be slow).
                let resp = {
                    let _blocked = BlockedRegion::enter(rt);
                    channel.read_frame()
                }?;
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
        let mut stream = blocking_accept(&runtime.0, &listener)?;
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
            let mut stream = blocking_accept(&runtime.0, &listener)?;
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
            let mut stream = match blocking_accept(&runtime.0, &listener) {
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
            let mut stream = match blocking_accept(&runtime.0, &listener) {
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
    let mut stream = connect_with_at_timeout(addr)?;
    unsafe { at_remote_on_channel(runtime, knowledge_base, &mut stream, closure_ptr) }
}

/// `TcpStream::connect` bounded by [`at_connect_timeout`]. Tries each
/// resolved address in turn; a timeout surfaces as
/// [`NetError::TimedOut`] (→ `Failure::TimedOut`/`Unreachable`), other
/// connect failures as `Io` (→ `Failure::Unreachable`).
fn connect_with_at_timeout(addr: impl ToSocketAddrs) -> Result<TcpStream, NetError> {
    let Some(d) = at_connect_timeout() else {
        return Ok(TcpStream::connect(addr)?);
    };
    let mut last: Option<io::Error> = None;
    for sa in addr.to_socket_addrs()? {
        match TcpStream::connect_timeout(&sa, d) {
            Ok(s) => return Ok(s),
            Err(e) => last = Some(e),
        }
    }
    match last {
        Some(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {
            Err(NetError::TimedOut("connecting".to_owned()))
        }
        Some(e) => Err(NetError::Io(e)),
        None => Err(NetError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            "address resolved to nothing",
        ))),
    }
}

thread_local! {
    /// Per-thread cache of open client connections keyed by destination
    /// address. Populated lazily on the first `at()` call to a given
    /// address; the entry is evicted if its request errors out (so a
    /// torn connection doesn't poison subsequent calls).
    static AT_CONN_CACHE: RefCell<HashMap<String, TcpStream>> =
        RefCell::new(HashMap::new());

    /// Per-OS-thread JIT execution context, created lazily the first
    /// time *this* thread invokes a shipped closure (see
    /// [`serve_thread_ptr`]). Every server thread that runs JIT'd code
    /// needs its own `Thread` (its own shadow-stack head) so concurrent
    /// connection handlers don't corrupt each other's GC roots. Dropped
    /// — deregistering the GC `ThreadState` — when the OS thread exits.
    static SERVE_CTX: RefCell<Option<ThreadContext>> = const { RefCell::new(None) };
}

/// Resolve the `Thread*` to drive a closure invocation on the *current*
/// OS thread.
///
/// On the runtime's home thread we reuse the existing `Thread` (so the
/// single-threaded path is unchanged). On any other OS thread we lazily
/// build — and cache for the lifetime of that thread — a dedicated
/// [`ThreadContext`], so concurrent handlers never share one shadow
/// stack.
pub(crate) fn serve_thread_ptr(runtime: &Runtime) -> *mut Thread {
    if std::thread::current().id() == runtime.home_thread {
        assert!(
            !runtime.dyna_thread.is_blocked(),
            "this Runtime was handed off to a server thread (its home \
             ThreadState is parked); the constructing thread must not \
             invoke JIT'd code through it anymore"
        );
        return runtime.thread_ptr();
    }
    SERVE_CTX.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(runtime.new_thread_context());
        }
        slot.as_ref().unwrap().thread_ptr()
    })
}

/// The current OS thread's GC `ThreadState` — the home thread's, or this
/// thread's own context (created if needed, same as [`serve_thread_ptr`]).
/// Returned as a cloned `Arc` to avoid borrowing the thread-local.
///
/// Used to mark blocking syscalls (`accept`, frame reads) as
/// [`ThreadState::enter_blocked`] regions so a stop-the-world collection
/// on another thread scans this thread's roots in place instead of
/// busy-waiting for it to poll — which it never would while parked in a
/// syscall.
fn serve_thread_state(runtime: &Runtime) -> Arc<ThreadState> {
    if std::thread::current().id() == runtime.home_thread {
        assert!(
            !runtime.dyna_thread.is_blocked(),
            "this Runtime was handed off to a server thread (its home \
             ThreadState is parked); the constructing thread must not \
             enter blocked regions / invoke JIT'd code through it anymore"
        );
        return runtime.dyna_thread.clone();
    }
    SERVE_CTX.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(runtime.new_thread_context());
        }
        slot.as_ref().unwrap().dyna_thread().clone()
    })
}

/// RAII guard that puts the current thread into `STATE_BLOCKED` for the
/// duration of a blocking call, then transitions it back to running
/// (waiting for any in-flight GC to finish first) on drop.
///
/// Precondition (caller's responsibility): no live ail heap pointer is
/// held un-rooted across the blocked region. The server blocking points
/// (`accept`, the initial frame read) hold none — the previous request's
/// values are already done and the next request hasn't been decoded yet.
pub(crate) struct BlockedRegion<'a> {
    ts: Arc<ThreadState>,
    heap: &'a Heap,
}

impl<'a> BlockedRegion<'a> {
    pub(crate) fn enter(runtime: &'a Runtime) -> Self {
        let ts = serve_thread_state(runtime);
        // Expose this thread's JIT frame chain to root scans for the
        // duration of the block. A blocked thread is scanned IN PLACE by
        // a concurrent STW collection; without this, live pointers in the
        // caller's JIT frame (e.g. locals held across a blocking at()
        // reply read or join) are invisible — the GC relocates their
        // objects without updating the spill slots, and the thread
        // resumes on stale from-space addresses. Null when the blocking
        // point holds no frames (server accept/read) — the walker skips
        // null.
        let thread = serve_thread_ptr(runtime);
        ts.set_parked_jit_fp(unsafe { (*thread).top_frame } as *const u8);
        ts.enter_blocked();
        BlockedRegion {
            ts,
            heap: &runtime.heap,
        }
    }
}

impl Drop for BlockedRegion<'_> {
    fn drop(&mut self) {
        // exit_blocked waits out any in-flight collection; only then is
        // it safe to stop exposing the frame (we are RUNNING again, so
        // any later GC will wait for us to park through normal channels).
        self.ts.exit_blocked(self.heap);
        self.ts.clear_parked_jit_fp();
    }
}

/// `accept_one` wrapped in a [`BlockedRegion`] so a server thread parked
/// waiting for the next connection doesn't stall a stop-the-world GC.
pub fn blocking_accept(runtime: &Runtime, listener: &TcpListener) -> Result<TcpStream, NetError> {
    park_home_on_foreign_serve(runtime);
    let _blocked = BlockedRegion::enter(runtime);
    accept_one(listener)
}

/// Handoff detection: the first time a runtime is served from an OS
/// thread other than the one that constructed it, permanently park its
/// home [`ThreadState`].
///
/// Handing a `Runtime` to a server thread (the `RuntimeHandle` /
/// `serve_*_in_thread` pattern) leaves the home `ThreadState` —
/// registered by `Runtime::new_with_metadata` on the constructing
/// thread — in `STATE_RUNNING` forever: its OS thread goes off to run
/// unrelated code (typically the client side of the test) and never
/// polls this heap's safepoints. Any stop-the-world collection on a
/// server thread then spins forever in `is_safely_at_safepoint`
/// waiting for it. Parking it as `STATE_BLOCKED` (frame chain is empty
/// at handoff) lets the GC scan it in place, like a thread blocked in
/// a syscall.
///
/// Contract this encodes: once a runtime is served from a foreign
/// thread, its constructing thread must never invoke JIT'd code
/// through it again. [`serve_thread_ptr`] / [`serve_thread_state`]
/// enforce this with a hard panic.
fn park_home_on_foreign_serve(runtime: &Runtime) {
    if std::thread::current().id() != runtime.home_thread {
        runtime.dyna_thread.park_handed_off();
    }
}

/// Client side of connection auth: send `KIND_AUTH` + token, expect the
/// node to echo a bare `KIND_AUTH` ack.
pub fn client_authenticate(channel: &mut dyn Channel, token: &str) -> Result<(), NetError> {
    let mut frame = Vec::with_capacity(1 + token.len());
    frame.push(KIND_AUTH);
    frame.extend_from_slice(token.as_bytes());
    channel.write_frame(&frame)?;
    let reply = channel.read_frame()?;
    if reply.as_slice() == [KIND_AUTH] {
        Ok(())
    } else {
        Err(NetError::ProtocolViolation("node rejected auth token"))
    }
}

/// Server side of connection auth: the first frame on a token-protected
/// connection must be `KIND_AUTH` + the exact token; ack with a bare
/// `KIND_AUTH`. On mismatch returns an error — the caller drops the
/// connection without replying (don't tell a prober why).
///
/// v1 shared-secret check; the comparison is not constant-time, which is
/// acceptable for the current trust model (the token gates accidental
/// access, the deploy trust model already assumes honest peers — see
/// the hash-by-fiat note in `IncrementalJit::install`).
pub fn server_expect_auth(channel: &mut dyn Channel, token: &str) -> Result<(), NetError> {
    let frame = channel.read_frame()?;
    if frame.first() == Some(&KIND_AUTH) && &frame[1..] == token.as_bytes() {
        channel.write_frame(&[KIND_AUTH])?;
        Ok(())
    } else {
        Err(NetError::ProtocolViolation("bad or missing auth token"))
    }
}

/// Clear the thread-local at() connection cache. Useful for tests
/// that want a clean slate, or for shutdown.
pub fn clear_at_conn_cache() {
    AT_CONN_CACHE.with(|c| c.borrow_mut().clear());
}

/// `at_remote` variant that reuses (or opens) a cached TCP stream for
/// the given address, amortising the TCP handshake across many `at()`
/// calls. Works with EITHER a persistent server ([`serve_persistent_in_thread`],
/// where the cache is a pure win) OR a one-shot server (the stdlib `serve`
/// loop, which closes the connection after each reply): if a *reused* cached
/// connection turns out to be dead — the common case against a one-shot
/// server — we evict it and retry ONCE on a guaranteed-fresh connection.
/// Without that retry, the second call to a one-shot node would fail (and a
/// counting server would then deadlock waiting for a connection the client
/// never opens).
///
/// # Safety
/// Same as [`at_remote`].
pub unsafe fn at_remote_cached(
    runtime: &Runtime,
    knowledge_base: &KnowledgeBase,
    addr: &str,
    closure_ptr: *const u8,
) -> Result<*const u8, NetError> {
    // Did we already have an open connection to this addr? If so, the first
    // attempt reuses it and a failure is most likely a stale (server-closed)
    // socket worth retrying; if not, the first attempt is already fresh and a
    // failure is a genuine error (no retry).
    let reused = AT_CONN_CACHE.with(|c| c.borrow().contains_key(addr));

    // First attempt: reuse the cached connection, or open one if absent.
    let attempt = |runtime: &Runtime, kb: &KnowledgeBase| -> Result<*const u8, NetError> {
        AT_CONN_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            let stream = match cache.entry(addr.to_owned()) {
                std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
                std::collections::hash_map::Entry::Vacant(v) => {
                    let s = connect_with_at_timeout(addr)?;
                    // Disable Nagle to keep per-cell latency low; the
                    // protocol is request/response and we never batch.
                    s.set_nodelay(true).ok();
                    v.insert(s)
                }
            };
            unsafe { at_remote_on_channel(runtime, kb, stream, closure_ptr) }
        })
    };

    let first = attempt(runtime, knowledge_base);
    let first_err = match first {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    // A `Remote` failure is a COMPLETE reply on a healthy connection: the
    // call reached the node and failed there. Keep the connection cached
    // and do NOT retry — re-sending might re-execute a stateful thunk.
    if matches!(first_err, NetError::Remote { .. }) {
        return Err(first_err);
    }
    // Evict the (now-suspect) connection.
    AT_CONN_CACHE.with(|c| {
        c.borrow_mut().remove(addr);
    });
    // Retry once ONLY when the failed attempt reused a cached socket AND
    // the failure is a transport error consistent with a stale (server-
    // closed) connection. A fresh-socket failure would just repeat; a
    // timeout is not retried here (the request may be mid-execution on
    // the node — re-sending could double-execute a stateful thunk).
    let stale_transport = matches!(
        first_err,
        NetError::Io(_) | NetError::ConnectionClosed
    );
    if !reused || !stale_transport {
        return Err(first_err);
    }
    let second = attempt(runtime, knowledge_base);
    if second.is_err() {
        AT_CONN_CACHE.with(|c| {
            c.borrow_mut().remove(addr);
        });
    }
    second
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
    // One deadline bounds the WHOLE exchange (send + NeedCode rounds +
    // result) so a hung node yields `Err(TimedOut)` instead of blocking
    // the caller forever.
    let deadline: Option<Instant> = at_call_deadline().map(|d| Instant::now() + d);
    let arm = |channel: &mut dyn Channel,
               phase: &'static str|
     -> Result<(), NetError> {
        match deadline {
            None => {
                channel.set_io_deadline(None);
                Ok(())
            }
            Some(t) => {
                let now = Instant::now();
                if now >= t {
                    return Err(NetError::TimedOut(phase.to_owned()));
                }
                channel.set_io_deadline(Some(t - now));
                Ok(())
            }
        }
    };
    let map_timeout = |e: NetError, phase: &'static str| -> NetError {
        match e {
            NetError::Io(ref io)
                if matches!(
                    io.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                NetError::TimedOut(phase.to_owned())
            }
            other => other,
        }
    };

    // Build and send the Call frame.
    let mut body = Vec::with_capacity(1024);
    body.push(KIND_CALL);
    unsafe { encode_value(runtime, WireValue::Heap(closure_ptr), &mut body)? };
    arm(channel, "sending call")?;
    channel
        .write_frame(&body)
        .map_err(|e| map_timeout(e, "sending call"))?;

    // Track which hashes we've already shipped this session so we don't
    // re-ship the same bytes if the server asks twice (defensive — a
    // good server shouldn't, but cheap to guard).
    let mut already_shipped: std::collections::HashSet<Hash> =
        std::collections::HashSet::new();

    for round in 0..=MAX_CODE_FETCH_ROUNDS {
        // Block on the reply in a BLOCKED region: the closure we shipped
        // is still rooted by our JIT caller's frame (so a concurrent GC
        // can relocate it safely); we hold no un-rooted heap pointer here.
        arm(channel, "waiting for reply")?;
        let reply = {
            let _blocked = BlockedRegion::enter(runtime);
            channel.read_frame()
        }
        .map_err(|e| map_timeout(e, "waiting for reply"))?;
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
            KIND_ERR => {
                // The node reported a failure instead of a result. The
                // connection is healthy (complete reply); the failure
                // becomes a `Failure` VALUE for the caller.
                let code = reply.get(1).copied().unwrap_or(ERR_CRASHED);
                let msg = String::from_utf8_lossy(reply.get(2..).unwrap_or(&[])).into_owned();
                return Err(NetError::Remote { code, msg });
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
                arm(channel, "shipping code")?;
                channel
                    .write_frame(&payload)
                    .map_err(|e| map_timeout(e, "shipping code"))?;
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
    /// Present only when the module declares `Failure::TimedOut(Node)`;
    /// deadline expirations fall back to `unreachable` otherwise.
    pub timed_out: Option<AtVariantLayout>,
    /// `DecodeError::TypeMismatch` / `DecodeError::Malformed` (nullary).
    /// Present only when the module declares `enum DecodeError`.
    pub decode_type_mismatch: Option<AtVariantLayout>,
    pub decode_malformed: Option<AtVariantLayout>,
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

    // DecodeError is optional — only present when the module declared it.
    let (decode_type_mismatch, decode_malformed) = match rb.decode_error_hash {
        Some(deh) => (
            lookup(rt, deh, rb.decode_type_mismatch_index),
            lookup(rt, deh, rb.decode_malformed_index),
        ),
        None => (None, None),
    };
    Some(AtRuntimeBinding {
        ok: lookup(rt, rb.result_hash, rb.ok_variant_index)?,
        err: lookup(rt, rb.result_hash, rb.err_variant_index)?,
        unreachable: lookup(rt, rb.failure_hash, rb.unreachable_variant_index)?,
        crashed: lookup(rt, rb.failure_hash, rb.crashed_variant_index)?,
        code_missing: lookup(rt, rb.failure_hash, rb.code_missing_variant_index)?,
        cancelled: lookup(rt, rb.failure_hash, rb.cancelled_variant_index)?,
        timed_out: rb
            .timed_out_variant_index
            .and_then(|idx| lookup(rt, rb.failure_hash, idx)),
        decode_type_mismatch,
        decode_malformed,
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

    unsafe { at_call_and_build(rt, kb, binding, thread, node_ptr, closure_ptr) }
}

/// The full `at()` exchange plus `Result` construction: resolve the
/// address from the Node struct, ship the thunk (with effects-driven
/// retry for pure thunks), and build the user's `Result<T, Failure>` on
/// `thread`'s context. Shared by the synchronous `at()` entry point and
/// the `at_async` worker thread.
///
/// # Safety
/// `thread` must be the CURRENT OS thread's `Thread*`; `node_ptr` and
/// `closure_ptr` must be live heap objects of the right shape.
unsafe fn at_call_and_build(
    rt: &Runtime,
    kb: &KnowledgeBase,
    binding: &AtRuntimeBinding,
    thread: *mut Thread,
    node_ptr: *const u8,
    closure_ptr: *const u8,
) -> *const u8 {
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
    if std::env::var_os("AI_LANG_AT_TRACE").is_some() {
        eprintln!("[at-trace] node={node_ptr:p} addr={addr}");
    }

    // Root `node_ptr` and `closure_ptr` across the exchange: decoding
    // the reply allocates (and a retry backoff parks us in a blocked
    // region), either of which can trigger a collection that relocates
    // them out from under these Rust locals. The scratch scope keeps
    // them visible to the GC and hands back the relocated addresses.
    let dyna = unsafe { &*(*thread).dyna_thread };
    let scope = dyna.scratch_scope();
    let node_slot = scope.push(node_ptr as *mut u8);
    let closure_slot = scope.push(closure_ptr as *mut u8);

    // Effects-driven retry policy: a PURE thunk (per the same knowledge
    // the at-cache uses) is safe to re-send after a transient failure —
    // re-executing it is observationally identical. A stateful thunk is
    // never auto-retried: the failed attempt may have executed on the
    // node, and re-sending would double-apply its effects.
    let is_pure = unsafe {
        let type_id_off = <crate::gc::Full as crate::gc::ObjHeader>::TYPE_ID_OFFSET;
        let type_id = *(closure_ptr.add(type_id_off) as *const u16);
        rt.shape_by_type_id
            .get(type_id as usize)
            .copied()
            .flatten()
            .map(|h| !rt.is_stateful(&h))
            .unwrap_or(false)
    };
    let budget = if is_pure { at_pure_retries() } else { 0 };

    let mut attempt: u64 = 0;
    let result = loop {
        let closure_now = scope.get(closure_slot) as *const u8;
        match unsafe { at_remote_cached(rt, kb, &addr, closure_now) } {
            Ok(n) => break Ok(n),
            Err(e) => {
                let transient = matches!(
                    e,
                    NetError::Io(_) | NetError::ConnectionClosed | NetError::TimedOut(_)
                );
                if transient && attempt < budget {
                    attempt += 1;
                    eprintln!(
                        "[ai_net_at] transient failure ({}); retrying pure thunk \
                         (attempt {}/{})",
                        e, attempt, budget
                    );
                    // Linear backoff, parked as BLOCKED so a concurrent
                    // STW collection isn't held up by the sleep.
                    let _blocked = BlockedRegion::enter(rt);
                    std::thread::sleep(Duration::from_millis(50 * attempt));
                    continue;
                }
                break Err(e);
            }
        }
    };

    match result {
        Ok(n) => unsafe { build_ok(thread, binding, n) },
        Err(e) => {
            eprintln!("[ai_net_at] at_remote failed: {}", e);
            let failure_variant = failure_variant_for(binding, &e);
            let node_now = scope.get(node_slot) as *const u8;
            unsafe { build_err(thread, binding, failure_variant, node_now) }
        }
    }
}

/// The runtime implementation of `at_async(node, thunk)`: ship the
/// thunk from a BACKGROUND OS thread and return immediately with a
/// `ThreadHandle` (a BoxedInt registry id). The existing `join` awaits
/// it and yields the same `Result<T, Failure>` value `at()` would have
/// returned — the failure model is identical, only the waiting moves.
///
/// The node and closure stay GC-rooted in the registry slot until the
/// worker reads them; the built Result stays rooted in the slot until
/// `join` hands it out. The worker runs on its own lazily-created
/// per-thread context (same mechanism as a serve thread).
///
/// # Safety
/// Same contract as [`ai_net_at`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_net_at_async(
    thread: *mut Thread,
    node_ptr: *const u8,
    closure_ptr: *const u8,
) -> *mut u8 {
    let rt = current_runtime().expect(
        "ai_net_at_async: install_current_runtime must be called before any at_async() in JIT",
    );
    let kb = current_knowledge_base().expect(
        "ai_net_at_async: install_current_knowledge_base must be called before any at_async()",
    );
    let binding = current_at_binding().expect(
        "ai_net_at_async: install_current_at_binding must be called before any at_async()",
    );

    // Root the inputs in the registry (scanned by every runtime's
    // collector) BEFORE spawning, so a collection between spawn and the
    // worker's first read relocates them in place.
    let (id, slot) = crate::runtime::async_task_register(closure_ptr, node_ptr);

    // The installed runtime/KB/binding outlive any JIT call by the
    // install contract; hand the worker raw pointers under that same
    // contract. (They are thread-locals on THIS thread — the worker
    // cannot re-read them itself.)
    struct Env(*const Runtime, *const KnowledgeBase, *const AtRuntimeBinding);
    unsafe impl Send for Env {}
    let env = Env(rt as *const _, kb as *const _, binding as *const _);

    let worker_slot = slot.clone();
    let jh = std::thread::spawn(move || {
        // Capture `env` as a whole (disjoint field capture would try to
        // move the raw pointers, which are not Send — the wrapper is).
        let env = env;
        let Env(rt_p, kb_p, binding_p) = env;
        // SAFETY: install contract (see above).
        let rt = unsafe { &*rt_p };
        let kb = unsafe { &*kb_p };
        let binding = unsafe { &*binding_p };
        // This worker's own execution context (own shadow-stack head +
        // GC ThreadState), created lazily exactly like a serve thread's.
        let wthread = serve_thread_ptr(rt);
        // Re-read the (possibly relocated) rooted inputs.
        let (closure_now, node_now) = crate::runtime::async_task_roots(&worker_slot);
        let result = unsafe {
            at_call_and_build(rt, kb, binding, wthread, node_now, closure_now)
        };
        // At-call failures are already VALUES (an Err Result); a pending
        // panic here would be an internal invariant break, but forward it
        // as a value anyway rather than losing it.
        let panic = unsafe { crate::runtime::take_pending_panic(wthread) };
        crate::runtime::async_task_finish(&worker_slot, result, panic);
    });
    crate::runtime::async_task_attach_handle(&slot, jh);

    unsafe { crate::runtime::ai_gc_box_int(thread, id) }
}

/// Map a client-side `NetError` onto the user's `Failure` variant.
fn failure_variant_for<'b>(
    binding: &'b AtRuntimeBinding,
    e: &NetError,
) -> &'b AtVariantLayout {
    match e {
        NetError::Io(_) | NetError::ConnectionClosed => &binding.unreachable,
        // Deadline expiry: distinct variant when the user declared
        // `TimedOut(Node)`, otherwise the closest honest mapping.
        NetError::TimedOut(_) => binding
            .timed_out
            .as_ref()
            .unwrap_or(&binding.unreachable),
        NetError::UnknownCode(_)
        | NetError::MissingFromKnowledgeBase(_)
        | NetError::CodeFetchDepthExceeded => &binding.code_missing,
        // The node reported the failure explicitly — honor its code.
        NetError::Remote { code, .. } => match *code {
            ERR_CODE_MISSING => &binding.code_missing,
            ERR_CANCELLED => &binding.cancelled,
            ERR_TIMED_OUT => binding
                .timed_out
                .as_ref()
                .unwrap_or(&binding.unreachable),
            _ => &binding.crashed,
        },
        NetError::Wire(_)
        | NetError::FrameTooLarge(_)
        | NetError::BadKind(_)
        | NetError::ProtocolViolation(_)
        | NetError::ProtocolViolationOwned(_)
        | NetError::InstallFailed(_)
        | NetError::HandlerPanicked(_)
        | NetError::DeployRefused(_) => &binding.crashed,
    }
}

/// Allocate the variant heap object, write the tag, and (if any)
/// store the payload at the right offset.
///
/// A pointer payload is rooted in the thread's scratch scope across the
/// allocation (which can collect and relocate it) and re-read before
/// the store — writing the pre-GC address would resurrect a stale
/// pointer into a live object.
unsafe fn alloc_variant(
    thread: *mut Thread,
    v: &AtVariantLayout,
    payload_int: Option<i64>,
    payload_ptr: Option<*const u8>,
) -> *const u8 {
    let dyna = unsafe { &*(*thread).dyna_thread };
    let scope = dyna.scratch_scope();
    let payload_slot = payload_ptr.map(|p| scope.push(p as *mut u8));
    let obj = unsafe { ai_gc_alloc_closure(thread, v.ti) };
    // Tag write (u32).
    let tag_slot = unsafe { obj.add(v.tag_offset as usize) as *mut u32 };
    unsafe { *tag_slot = v.tag_value };
    // Payload write.
    if let Some(off) = v.payload_offset {
        if v.payload_is_pointer {
            let slot_idx = payload_slot.expect("pointer payload required for this variant");
            let p = scope.get(slot_idx) as *const u8;
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
    thread: *mut Thread,
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
    let ret_ptr = unsafe { lambda(thread, closure_ptr) };
    // A thunk panic propagates out of JIT'd code as a pending-panic
    // value; this serve boundary takes it and converts it into an error
    // the protocol can report (`KIND_ERR`/`ERR_CRASHED`). The node keeps
    // running — one bad request fails one request.
    if let Some(msg) = unsafe { crate::runtime::take_pending_panic(thread) } {
        return Err(NetError::HandlerPanicked(msg));
    }
    Ok(ret_ptr as *const u8)
}

// =============================================================================
// Value boundary: encode / decode / invoke, exposed to ai-lang
// =============================================================================
//
// These three runtime fns expose the wire codec (and closure invocation)
// to ai-lang code so that a *node* — the accept/recv/dispatch/respond
// loop — can be written in the language itself, instead of the hardcoded
// Rust `serve_with_install`. They all require `install_current_runtime`
// on the calling thread (same contract as `ai_net_at`), because the codec
// reads the runtime's shape metadata and the invoker reads its code table.
//
// v1 deliberately does NOT do the open-world `Any` + reflection layer: the
// node is assumed to already have the shipped code (both ends compiled the
// same source), so decode never needs to fetch a missing shape. Malformed
// input is a LOUD panic, not a silent wrong value — proper value-level
// `Result<_, DecodeError>` is the next increment.

/// Encode any heap value to a fresh ai-lang `Bytes` (same heap layout as
/// `String`). Reuses `encode_value`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_wire_encode(
    thread: *mut Thread,
    value_ptr: *const u8,
) -> *const u8 {
    let rt = current_runtime()
        .expect("ai_wire_encode: install_current_runtime must be called first");
    let mut buf = Vec::with_capacity(256);
    if let Err(e) = unsafe { encode_value(rt, WireValue::Heap(value_ptr), &mut buf) } {
        unsafe {
            crate::runtime::raise_pending_panic(
                thread,
                format!("wire_encode: encode failed: {:?}", e),
            )
        };
        return core::ptr::null();
    }
    unsafe { crate::runtime::ai_str_new(thread, buf.as_ptr(), buf.len() as i64) as *const u8 }
}

/// Copy a heap `Bytes`/`String` payload into a stable Rust buffer BEFORE
/// decoding it. `decode_value` allocates, an allocation can collect, and
/// a collection relocates the source object — a borrowed `&[u8]` into
/// the heap would keep reading the rest of the frame out of from-space
/// garbage. (Bit only when the wire value reads bytes AFTER its first
/// allocation — e.g. a closure with captures; zero-capture closures
/// decode entirely before allocating, which is why this lay hidden.)
unsafe fn stable_wire_bytes(bytes_ptr: *const u8) -> Vec<u8> {
    unsafe { crate::ffi::heap_str_bytes(bytes_ptr) }.to_vec()
}

/// Decode a `Bytes` produced by `ai_wire_encode` whose value is an `Int`,
/// returning the raw i64. (v1 Int-only decode; the generic/open-world
/// decode comes later.) Panics loudly on malformed input or a non-Int.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_wire_decode_int(
    thread: *mut Thread,
    bytes_ptr: *const u8,
) -> i64 {
    let rt = current_runtime()
        .expect("ai_wire_decode_int: install_current_runtime must be called first");
    let bytes = unsafe { stable_wire_bytes(bytes_ptr) };
    match unsafe { decode_value(rt, &bytes) } {
        Ok((WireValue::Int(n), _)) => n,
        Ok((WireValue::Heap(p), _)) => unsafe { crate::runtime::ai_gc_unbox_int(p) },
        Err(e) => {
            unsafe {
                crate::runtime::raise_pending_panic(
                    thread,
                    format!("wire_decode_int: decode failed: {:?}", e),
                )
            };
            0
        }
    }
}

/// Decode a `Bytes` whose value is a heap object (e.g. a closure) and
/// return the reconstructed heap pointer. Used to decode a shipped
/// updater closure into a callable value on the node, so the node can
/// `swap` its own atom with it. Panics loudly on malformed input or an
/// unexpected Int.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_wire_decode_ptr(
    thread: *mut Thread,
    bytes_ptr: *const u8,
) -> *const u8 {
    let rt = current_runtime()
        .expect("ai_wire_decode_ptr: install_current_runtime must be called first");
    let bytes = unsafe { stable_wire_bytes(bytes_ptr) };
    let raise = |msg: String| -> *const u8 {
        unsafe { crate::runtime::raise_pending_panic(thread, msg) };
        core::ptr::null()
    };
    match unsafe { decode_value(rt, &bytes) } {
        Ok((WireValue::Heap(p), _)) => p,
        Ok((WireValue::Int(_), _)) => {
            raise("wire_decode_ptr: expected a heap value (closure), got Int".to_owned())
        }
        Err(e) => raise(format!("wire_decode_ptr: decode failed: {:?}", e)),
    }
}

/// The heart of an ai-lang node: take the wire bytes of a shipped zero-arg
/// closure, decode it onto the local heap, invoke it, and encode the
/// result back to `Bytes`. This is exactly `serve_one`'s body with the
/// socket removed — so the socket/accept/loop can live in ai-lang.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_wire_invoke(
    thread: *mut Thread,
    closure_bytes_ptr: *const u8,
) -> *const u8 {
    let rt = current_runtime()
        .expect("ai_wire_invoke: install_current_runtime must be called first");
    let bytes = unsafe { stable_wire_bytes(closure_bytes_ptr) };
    let raise = |msg: String| -> *const u8 {
        unsafe { crate::runtime::raise_pending_panic(thread, msg) };
        core::ptr::null()
    };
    let (value, _) = match unsafe { decode_value(rt, &bytes) } {
        Ok(v) => v,
        Err(e) => return raise(format!("wire_invoke: decode failed: {:?}", e)),
    };
    let closure_ptr = match value {
        WireValue::Heap(p) => p,
        WireValue::Int(_) => {
            return raise("wire_invoke: payload must be a closure, got Int".to_owned());
        }
    };
    // Use the `Thread` our JIT caller passed: it is already this OS
    // thread's own context (own shadow-stack head), so the invoked
    // closure builds its frame chain on the right thread rather than on
    // the runtime's shared home `Thread`.
    //
    // `invoke_zero_arg_closure` converts an invoked-thunk panic into
    // `HandlerPanicked`; re-raise it here so the (language-level) node
    // loop's own error handling sees it as a value.
    let result_ptr = match unsafe { invoke_zero_arg_closure(rt, thread, closure_ptr) } {
        Ok(p) => p,
        Err(NetError::HandlerPanicked(msg)) => {
            return raise(format!("wire_invoke: shipped thunk panicked: {}", msg));
        }
        Err(e) => return raise(format!("wire_invoke: closure invocation failed: {}", e)),
    };
    let mut buf = Vec::with_capacity(256);
    if let Err(e) = unsafe { encode_value(rt, WireValue::Heap(result_ptr), &mut buf) } {
        return raise(format!("wire_invoke: encode of result failed: {:?}", e));
    }
    unsafe { crate::runtime::ai_str_new(thread, buf.as_ptr(), buf.len() as i64) as *const u8 }
}

/// The runtime *identity hash* of a decoded heap value, matching what
/// `resolve::decode_expected_hash` computes for a type: an enum value
/// reports its ENUM hash (not the per-variant hash); everything else
/// reports its shape hash (a struct's content hash, the BoxedInt shape,
/// the String/Array canonical shape).
unsafe fn identity_hash_of(rt: &Runtime, p: *const u8) -> Hash {
    let type_id_off = <crate::gc::Full as crate::gc::ObjHeader>::TYPE_ID_OFFSET;
    let type_id = unsafe { *(p.add(type_id_off) as *const u16) };
    let shape_hash = rt
        .shape_by_type_id
        .get(type_id as usize)
        .copied()
        .flatten()
        .unwrap_or(Hash([0u8; 32]));
    if let Some(ShapeMeta::EnumVariant { enum_ref, .. }) = rt.shape_meta.get(&shape_hash) {
        *enum_ref
    } else {
        shape_hash
    }
}

/// Build `Result::Err(DecodeError::<variant>)` for a nullary DecodeError
/// variant. Reuses the `at()` Result `Err` layout. Panics with a clear
/// message if the module didn't declare `enum DecodeError` (the resolver
/// already rejects `decode::<T>` in that case, so this is unreachable in
/// practice).
unsafe fn build_decode_err(
    thread: *mut Thread,
    b: &AtRuntimeBinding,
    variant: &Option<AtVariantLayout>,
) -> *const u8 {
    let v = variant.as_ref().expect(
        "ai_wire_decode_checked: DecodeError variant layout missing \
         (module must declare `enum DecodeError { TypeMismatch, Malformed }`)",
    );
    let de = unsafe { alloc_variant(thread, v, None, None) };
    unsafe { alloc_variant(thread, &b.err, None, Some(de)) }
}

/// Generic CHECKED decode, exposed to ai-lang as `decode::<T>(bytes)`.
/// The 32-byte expected identity hash for `T` arrives as four i64 words
/// (`h0..h3`, little-endian), baked in by the resolver. Decodes the
/// bytes, verifies the decoded value's identity matches `T`, and returns
/// the user's `Result<T, Int>`: `Ok(value)` on match, `Err(1)` on type
/// mismatch, `Err(2)` on malformed input. Requires `install_current_runtime`
/// and `install_current_at_binding` (same contract as `at()`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_wire_decode_checked(
    thread: *mut Thread,
    bytes_ptr: *const u8,
    h0: i64,
    h1: i64,
    h2: i64,
    h3: i64,
) -> *const u8 {
    let rt = current_runtime()
        .expect("ai_wire_decode_checked: install_current_runtime must be called first");
    let binding = current_at_binding()
        .expect("ai_wire_decode_checked: install_current_at_binding must be called first");

    let mut hb = [0u8; 32];
    hb[0..8].copy_from_slice(&(h0 as u64).to_le_bytes());
    hb[8..16].copy_from_slice(&(h1 as u64).to_le_bytes());
    hb[16..24].copy_from_slice(&(h2 as u64).to_le_bytes());
    hb[24..32].copy_from_slice(&(h3 as u64).to_le_bytes());
    let expected = Hash(hb);

    let bytes = unsafe { stable_wire_bytes(bytes_ptr) };
    let decoded = match unsafe { decode_value(rt, &bytes) } {
        Ok((v, _)) => v,
        Err(_) => return unsafe { build_decode_err(thread, binding, &binding.decode_malformed) },
    };
    let (value_ptr, identity) = match decoded {
        WireValue::Int(n) => {
            let boxed = unsafe { crate::runtime::ai_gc_box_int(thread, n) };
            (boxed as *const u8, crate::runtime::boxed_int_shape_hash())
        }
        WireValue::Heap(p) => (p, unsafe { identity_hash_of(rt, p) }),
    };
    if identity != expected {
        return unsafe { build_decode_err(thread, binding, &binding.decode_type_mismatch) };
    }
    unsafe { build_ok(thread, binding, value_ptr) }
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

    /// Build a Runtime + Jit from the full stdlib SOURCE plus `extra`
    /// (so stdlib defs — including the ai-lang socket/framing layer — are
    /// in scope). Returns the names table too.
    fn make_stdlib_runtime<'ctx>(
        ctx: &'ctx Context,
        extra: &str,
    ) -> (Runtime, Jit<'ctx>, std::collections::HashMap<String, Hash>) {
        let full = format!("{}\n{}", crate::stdlib::SOURCE, extra);
        let m = parse_module(&full).unwrap();
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

    /// Pick a free TCP port by binding :0 and dropping the listener.
    fn free_port() -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    }

    /// The full function-driven KV store source (examples/kvstore.ail).
    const KVSTORE_SRC: &str = include_str!("../examples/kvstore.ail");

    /// The explicit-TCP KV store, driven purely by calling functions
    /// (`kv_set_tcp` / `kv_get_tcp` / `kv_del_tcp`) — the client never builds a
    /// command enum or touches a frame. set 41 -> get 41 -> del 1 -> get
    /// missing (-404). Proves the function-driven TCP path end to end over real
    /// loopback, mutating the node's own `db` state.
    #[test]
    fn kvstore_tcp_function_driven() {
        init();
        let port = free_port();
        // Nested enum patterns: unwrap Result/Option/Val in one match, and
        // multiple arms share the outer `Ok` variant.
        let drivers = r#"
            def kv_listen(p: Int) -> Int =
                match tcp_listen(p) { Result::Ok(fd) => fd, Result::Err(_e) => 0 - 1 }
            def kv_run(fd: Int) -> Int = serve_turns(fd, 4)

            def t_set(p: Int, d: Int) -> Int =
                match kv_set_tcp(tcp_node(127, 0, 0, 1, p), "x", Val::VInt(d)) {
                    Result::Ok(Val::VInt(n)) => n,
                    Result::Ok(Val::VStr(_s)) => 0 - 2,
                    Result::Err(_e) => 0 - 1,
                }
            def t_get(p: Int) -> Int =
                match kv_get_tcp(tcp_node(127, 0, 0, 1, p), "x") {
                    Result::Ok(Option::Some(Val::VInt(n))) => n,
                    Result::Ok(Option::Some(Val::VStr(_s))) => 0 - 2,
                    Result::Ok(Option::None) => 0 - 404,
                    Result::Err(_e) => 0 - 1,
                }
            def t_del(p: Int) -> Int =
                match kv_del_tcp(tcp_node(127, 0, 0, 1, p), "x") {
                    Result::Ok(n) => n,
                    Result::Err(_e) => 0 - 1,
                }
        "#;
        let src = format!("{}\n{}", KVSTORE_SRC, drivers);

        let server_ctx = Context::create();
        let (server_rt, server_jit, names) = make_stdlib_runtime(&server_ctx, &src);
        install_current_runtime(&server_rt);
        let listen = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["kv_listen"]),
                )
                .unwrap()
        };
        let fd = unsafe { listen.call(server_rt.thread_ptr(), port as i64) };
        assert!(fd >= 0, "listen failed: {}", fd);

        let src_for_client = src.clone();
        let client = std::thread::spawn(move || -> Vec<i64> {
            let client_ctx = Context::create();
            let (client_rt, client_jit, names) =
                make_stdlib_runtime(&client_ctx, &src_for_client);
            install_current_runtime(&client_rt);
            // `decode::<T>` reuses the Result/DecodeError binding, so the
            // client needs the at() runtime binding installed even though this
            // path drives the socket explicitly.
            let full = format!("{}\n{}", crate::stdlib::SOURCE, src_for_client);
            let m = parse_module(&full).unwrap();
            let r = resolve_module(&m).unwrap();
            let rb = r.at_binding.as_ref().expect("at_binding");
            let rt_binding = build_at_runtime_binding(&client_rt, rb).expect("rt binding");
            install_current_at_binding(&rt_binding);
            let set = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64) -> i64>(
                        &crate::codegen::def_symbol(&names["t_set"]),
                    )
                    .unwrap()
            };
            let getf = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                        &crate::codegen::def_symbol(&names["t_get"]),
                    )
                    .unwrap()
            };
            let del = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                        &crate::codegen::def_symbol(&names["t_del"]),
                    )
                    .unwrap()
            };
            let p = port as i64;
            let r1 = unsafe { set.call(client_rt.thread_ptr(), p, 41) };
            let r2 = unsafe { getf.call(client_rt.thread_ptr(), p) };
            let r3 = unsafe { del.call(client_rt.thread_ptr(), p) };
            let r4 = unsafe { getf.call(client_rt.thread_ptr(), p) };
            clear_current_at_binding();
            clear_current_runtime();
            vec![r1, r2, r3, r4]
        });

        let _ = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["kv_run"]),
                )
                .unwrap()
                .call(server_rt.thread_ptr(), fd)
        };

        let results = client.join().unwrap();
        clear_current_runtime();
        assert_eq!(
            results,
            vec![41, 41, 1, 0 - 404],
            "function-driven TCP KV: set/get/del/get-missing"
        );
    }

    /// The same KV store via `at()` — the client calls `kv_set_at` then
    /// `kv_get_at` and the runtime handles connect/encode/serve/decode. set 7
    /// -> get 7, mutating and then reading the node's own `db` across TWO
    /// at() calls. The server is a one-shot counting loop (`serve_n`), which
    /// closes each connection after replying; the cached at() client must
    /// reconnect for the second call rather than hang on the stale socket.
    #[test]
    fn kvstore_at_function_driven() {
        init();
        clear_at_conn_cache();
        let drivers = r#"
            def a_set(p: Int, d: Int) -> Int =
                match kv_set_at(tcp_node(127, 0, 0, 1, p), "y", Val::VInt(d)) {
                    Result::Ok(Val::VInt(n)) => n,
                    Result::Ok(Val::VStr(_s)) => 0 - 2,
                    Result::Err(_e) => 0 - 1,
                }
            def a_get(p: Int) -> Int =
                match kv_get_at(tcp_node(127, 0, 0, 1, p), "y") {
                    Result::Ok(Option::Some(Val::VInt(n))) => n,
                    Result::Ok(Option::Some(Val::VStr(_s))) => 0 - 2,
                    Result::Ok(Option::None) => 0 - 404,
                    Result::Err(_e) => 0 - 1,
                }
        "#;
        let src = format!("{}\n{}", KVSTORE_SRC, drivers);
        let full = format!("{}\n{}", crate::stdlib::SOURCE, src);

        // Server: a one-shot loop that serves exactly two at() invocations
        // (set, get) on two separate connections, then exits.
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_stdlib_runtime(&server_ctx, &src);
        let _keep_server_jit = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        let server_handle =
            unsafe { serve_n_in_thread(Arc::new(RuntimeHandle(server_rt)), listener, 2) };

        // Client: install the at() runtime binding, then just call functions.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_stdlib_runtime(&client_ctx, &src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(&full).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        let rt_binding = build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        let set = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["a_set"]),
                )
                .unwrap()
        };
        let getf = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["a_get"]),
                )
                .unwrap()
        };
        let r1 = unsafe { set.call(client_rt.thread_ptr(), server_port, 7) };
        let r2 = unsafe { getf.call(client_rt.thread_ptr(), server_port) };

        server_handle.join().unwrap().expect("server ok");
        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
        assert_eq!(
            (r1, r2),
            (7, 7),
            "function-driven at(): set then get across two one-shot connections"
        );
    }

    /// The ai-lang transport stack (sockets + framing) works end to end
    /// over real loopback TCP, with NO Rust networking code in the path:
    /// the server (listen/accept/recv/echo) and client (connect/send/recv)
    /// are both ordinary ai-lang functions from the stdlib.
    #[test]
    fn ail_socket_frame_roundtrip() {
        init();
        let port = free_port();
        // Entries are () -> Int so they use the known main()-style ABI.
        // The port is baked in as a literal.
        let src = format!(
            r#"
            // Probe the (Int)->Int / ()->Int top-level ABI up front.
            def probe_abi() -> Int = 41 + 1

            def make_msg() -> Bytes = {{
                let b = bytes_new(5);
                let _0 = bytes_set(b, 0, 10);
                let _1 = bytes_set(b, 1, 20);
                let _2 = bytes_set(b, 2, 30);
                let _3 = bytes_set(b, 3, 40);
                let _4 = bytes_set(b, 4, 50);
                b
            }}

            def bytes_sum(b: Bytes, i: Int, n: Int, acc: Int) -> Int =
                if i >= n {{ acc }} else {{ bytes_sum(b, i + 1, n, acc + bytes_get(b, i)) }}

            // Bind+listen; return the listener fd (or negative on error).
            def server_listen() -> Int =
                match tcp_listen({port}) {{
                    Result::Ok(fd) => fd,
                    Result::Err(_e) => 0 - 1,
                }}

            // Accept one conn, echo one frame back, return bytes received.
            def server_accept_echo(listener: Int) -> Int =
                match tcp_accept(listener) {{
                    Result::Ok(conn) => match recv_frame(conn) {{
                        Result::Ok(b) => {{
                            let n = bytes_len(b);
                            let _s = send_frame(conn, b);
                            let _c = conn_close(conn);
                            n
                        }},
                        Result::Err(_e) => 0 - 2,
                    }},
                    Result::Err(_e) => 0 - 3,
                }}

            // Connect, send a frame, read the echo, return the byte sum.
            def client_roundtrip() -> Int =
                match tcp_connect(127, 0, 0, 1, {port}) {{
                    Result::Ok(conn) => match send_frame(conn, make_msg()) {{
                        Result::Ok(_w) => match recv_frame(conn) {{
                            Result::Ok(echo) => {{
                                let s = bytes_sum(echo, 0, bytes_len(echo), 0);
                                let _c = conn_close(conn);
                                s
                            }},
                            Result::Err(_e) => 0 - 5,
                        }},
                        Result::Err(_e) => 0 - 6,
                    }},
                    Result::Err(_e) => 0 - 4,
                }}
            "#,
            port = port
        );

        // SERVER side runs inline on the test thread (it blocks in accept);
        // CLIENT runs in a spawned thread (builds its own Runtime to avoid
        // moving the non-Send Runtime across threads).
        let server_ctx = Context::create();
        let (server_rt, server_jit, names) = make_stdlib_runtime(&server_ctx, &src);

        // ABI probe: a top-level () -> Int returns a raw i64.
        let probe = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                    &crate::codegen::def_symbol(&names["probe_abi"]),
                )
                .unwrap()
        };
        assert_eq!(unsafe { probe.call(server_rt.thread_ptr()) }, 42, "()->Int ABI");

        let listen_fn = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                    &crate::codegen::def_symbol(&names["server_listen"]),
                )
                .unwrap()
        };
        let fd = unsafe { listen_fn.call(server_rt.thread_ptr()) };
        assert!(fd >= 0, "server_listen failed: {}", fd);

        // Client connects from a separate thread / Runtime, same source.
        let src_for_client = src.clone();
        let client_handle = std::thread::spawn(move || -> i64 {
            let client_ctx = Context::create();
            let (client_rt, client_jit, names) =
                make_stdlib_runtime(&client_ctx, &src_for_client);
            let f = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                        &crate::codegen::def_symbol(&names["client_roundtrip"]),
                    )
                    .unwrap()
            };
            unsafe { f.call(client_rt.thread_ptr()) }
        });

        // Server accepts + echoes inline.
        let accept_fn = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["server_accept_echo"]),
                )
                .unwrap()
        };
        let recvd = unsafe { accept_fn.call(server_rt.thread_ptr(), fd) };

        let client_sum = client_handle.join().unwrap();
        assert_eq!(recvd, 5, "server should receive 5 bytes");
        assert_eq!(client_sum, 150, "client should get echo back (10+20+30+40+50)");
    }

    /// The headline: a *node loop written in ai-lang* (accept / recv /
    /// invoke / respond) runs a closure shipped by an ai-lang client over
    /// real TCP. The only Rust in the value path is the `wire_invoke`
    /// primitive (decode+invoke+encode); the loop, framing, and transport
    /// are all ai-lang. Replaces the hardcoded Rust `serve_with_install`.
    #[test]
    fn ail_node_loop_runs_shipped_closure() {
        init();
        let port = free_port();
        let src = format!(
            r#"
            def make_thunk(n: Int) -> fn() -> Int = || n + 100

            def server_listen() -> Int =
                match tcp_listen({port}) {{ Result::Ok(fd) => fd, Result::Err(_e) => 0 - 1 }}

            // One turn of an ai-lang node: accept, decode+invoke the
            // shipped closure via wire_invoke, ship the encoded result.
            def server_node_once(listener: Int) -> Int =
                match tcp_accept(listener) {{
                    Result::Ok(conn) => match recv_frame(conn) {{
                        Result::Ok(req) => {{
                            let resp = wire_invoke(req);
                            let _s = send_frame(conn, resp);
                            let _c = conn_close(conn);
                            0
                        }},
                        Result::Err(_e) => 0 - 2,
                    }},
                    Result::Err(_e) => 0 - 3,
                }}

            // Client: ship make_thunk(42), decode the Int reply.
            def client_call() -> Int =
                match tcp_connect(127, 0, 0, 1, {port}) {{
                    Result::Ok(conn) => match send_frame(conn, wire_encode(make_thunk(42))) {{
                        Result::Ok(_w) => match recv_frame(conn) {{
                            Result::Ok(resp) => {{
                                let v = wire_decode_int(resp);
                                let _c = conn_close(conn);
                                v
                            }},
                            Result::Err(_e) => 0 - 5,
                        }},
                        Result::Err(_e) => 0 - 6,
                    }},
                    Result::Err(_e) => 0 - 4,
                }}
            "#,
            port = port
        );

        let server_ctx = Context::create();
        let (server_rt, server_jit, names) = make_stdlib_runtime(&server_ctx, &src);
        // wire_invoke reads shape metadata + the code table from the
        // installed current runtime (same contract as at()).
        install_current_runtime(&server_rt);

        let fd = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                    &crate::codegen::def_symbol(&names["server_listen"]),
                )
                .unwrap()
                .call(server_rt.thread_ptr())
        };
        assert!(fd >= 0, "server_listen failed: {}", fd);

        let src_for_client = src.clone();
        let client_handle = std::thread::spawn(move || -> i64 {
            let client_ctx = Context::create();
            let (client_rt, client_jit, names) =
                make_stdlib_runtime(&client_ctx, &src_for_client);
            install_current_runtime(&client_rt);
            let v = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                        &crate::codegen::def_symbol(&names["client_call"]),
                    )
                    .unwrap()
                    .call(client_rt.thread_ptr())
            };
            clear_current_runtime();
            v
        });

        let server_rc = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["server_node_once"]),
                )
                .unwrap()
                .call(server_rt.thread_ptr(), fd)
        };

        let client_result = client_handle.join().unwrap();
        clear_current_runtime();
        assert_eq!(server_rc, 0, "server node turn should succeed");
        assert_eq!(client_result, 142, "shipped || 42+100 should evaluate to 142");
    }

    /// A REMOTE ATOM, built entirely as ai-lang over the node loop: the
    /// server holds a local `Atom<Int>` and, each turn, decodes a shipped
    /// `fn(Int)->Int` updater and `swap`s its own atom with it (the real
    /// Clojure-atom move, now distributed). State persists across separate
    /// client connections. No new wire kinds, no Rust atom protocol — the
    /// semantics are visible ai-lang.
    #[test]
    fn ail_remote_atom_counter() {
        init();
        let port = free_port();
        let src = format!(
            r#"
            def make_inc(d: Int) -> fn(Int) -> Int = |n: Int| n + d

            def server_listen() -> Int =
                match tcp_listen({port}) {{ Result::Ok(fd) => fd, Result::Err(_e) => 0 - 1 }}

            // One turn: accept, decode the shipped updater, swap the
            // server's atom, ship back the new value.
            def server_atom_turn(listener: Int, a: Atom<Int>) -> Int =
                match tcp_accept(listener) {{
                    Result::Ok(conn) => match recv_frame(conn) {{
                        Result::Ok(req) => {{
                            let updater = wire_decode_fn1(req);
                            let now = swap(a, updater);
                            let _s = send_frame(conn, wire_encode(now));
                            let _c = conn_close(conn);
                            now
                        }},
                        Result::Err(_e) => 0 - 2,
                    }},
                    Result::Err(_e) => 0 - 3,
                }}

            def server_atom_loop(listener: Int, a: Atom<Int>, turns: Int) -> Int =
                if turns <= 0 {{ deref(a) }} else {{
                    let _t = server_atom_turn(listener, a);
                    server_atom_loop(listener, a, turns - 1)
                }}

            // The node owns the atom; the loop threads it. Runs 3 turns.
            def server_main(listener: Int) -> Int = {{
                let a = atom_new(0);
                server_atom_loop(listener, a, 3)
            }}

            // Client: ship `|n| n + d`, get the new server-side value.
            def client_inc(d: Int) -> Int =
                match tcp_connect(127, 0, 0, 1, {port}) {{
                    Result::Ok(conn) => match send_frame(conn, wire_encode(make_inc(d))) {{
                        Result::Ok(_w) => match recv_frame(conn) {{
                            Result::Ok(resp) => {{
                                let v = wire_decode_int(resp);
                                let _c = conn_close(conn);
                                v
                            }},
                            Result::Err(_e) => 0 - 5,
                        }},
                        Result::Err(_e) => 0 - 6,
                    }},
                    Result::Err(_e) => 0 - 4,
                }}
            "#,
            port = port
        );

        let server_ctx = Context::create();
        let (server_rt, server_jit, names) = make_stdlib_runtime(&server_ctx, &src);
        install_current_runtime(&server_rt);

        let fd = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                    &crate::codegen::def_symbol(&names["server_listen"]),
                )
                .unwrap()
                .call(server_rt.thread_ptr())
        };
        assert!(fd >= 0, "server_listen failed: {}", fd);

        let src_for_client = src.clone();
        let client_handle = std::thread::spawn(move || -> Vec<i64> {
            let client_ctx = Context::create();
            let (client_rt, client_jit, names) =
                make_stdlib_runtime(&client_ctx, &src_for_client);
            install_current_runtime(&client_rt);
            let inc = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                        &crate::codegen::def_symbol(&names["client_inc"]),
                    )
                    .unwrap()
            };
            // Three separate connections: +1, +10, +100.
            let r1 = unsafe { inc.call(client_rt.thread_ptr(), 1) };
            let r2 = unsafe { inc.call(client_rt.thread_ptr(), 10) };
            let r3 = unsafe { inc.call(client_rt.thread_ptr(), 100) };
            clear_current_runtime();
            vec![r1, r2, r3]
        });

        let final_val = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["server_main"]),
                )
                .unwrap()
                .call(server_rt.thread_ptr(), fd)
        };

        let client_results = client_handle.join().unwrap();
        clear_current_runtime();
        // The atom accumulates across connections: 0 -> 1 -> 11 -> 111.
        assert_eq!(client_results, vec![1, 11, 111], "client sees accumulating state");
        assert_eq!(final_val, 111, "server atom holds the final accumulated value");
    }

    /// The node-`state` remote-handler model over REAL loopback TCP, with
    /// the node loop written entirely in ail via the generic stdlib
    /// `serve_turns`. The server `serve`s; remote participants ship
    /// `|| handle(msg)` closures that run on the node and mutate its OWN
    /// `state` cell. Proves the model end-to-end on the real transport
    /// (not just the in-process channel) and exercises the stdlib `serve`.
    #[test]
    fn ail_tcp_node_state_remote_handlers() {
        init();
        let port = free_port();
        let src = format!(
            r#"
            enum Cmd {{ Bump(Int), Get }}
            state tcp_counter: Atom<Int> = atom(0)
            def tcp_handle(c: Cmd) -> Int =
                match c {{
                    Cmd::Bump(d) => swap(tcp_counter, |n: Int| n + d),
                    Cmd::Get => deref(tcp_counter),
                }}

            // Node: bind + serve 3 requests via the GENERIC stdlib loop.
            def tcp_listen_node() -> Int =
                match tcp_listen({port}) {{ Result::Ok(fd) => fd, Result::Err(_e) => 0 - 1 }}
            def tcp_node_main(fd: Int) -> Int = serve_turns(fd, 3)

            // Client: ship `|| tcp_handle(msg)` and read the Int reply.
            def tcp_bump_thunk(d: Int) -> fn() -> Int = || tcp_handle(Cmd::Bump(d))
            def tcp_get_thunk() -> fn() -> Int = || tcp_handle(Cmd::Get)
            def tcp_send(payload: Bytes) -> Int =
                match tcp_connect(127, 0, 0, 1, {port}) {{
                    Result::Ok(conn) => match send_frame(conn, payload) {{
                        Result::Ok(_w) => match recv_frame(conn) {{
                            Result::Ok(resp) => {{
                                let v = wire_decode_int(resp);
                                let _c = conn_close(conn);
                                v
                            }},
                            Result::Err(_e) => 0 - 5,
                        }},
                        Result::Err(_e) => 0 - 6,
                    }},
                    Result::Err(_e) => 0 - 4,
                }}
            def tcp_client_bump(d: Int) -> Int = tcp_send(wire_encode(tcp_bump_thunk(d)))
            def tcp_client_get() -> Int = tcp_send(wire_encode(tcp_get_thunk()))
            "#,
            port = port
        );

        let server_ctx = Context::create();
        let (server_rt, server_jit, names) = make_stdlib_runtime(&server_ctx, &src);
        install_current_runtime(&server_rt);
        let fd = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                    &crate::codegen::def_symbol(&names["tcp_listen_node"]),
                )
                .unwrap()
                .call(server_rt.thread_ptr())
        };
        assert!(fd >= 0, "listen failed: {}", fd);

        let src_for_client = src.clone();
        let client = std::thread::spawn(move || -> Vec<i64> {
            let client_ctx = Context::create();
            let (client_rt, client_jit, names) = make_stdlib_runtime(&client_ctx, &src_for_client);
            install_current_runtime(&client_rt);
            let bump = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                        &crate::codegen::def_symbol(&names["tcp_client_bump"]),
                    )
                    .unwrap()
            };
            let get = unsafe {
                client_jit
                    .engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                        &crate::codegen::def_symbol(&names["tcp_client_get"]),
                    )
                    .unwrap()
            };
            let r1 = unsafe { bump.call(client_rt.thread_ptr(), 5) };
            let r2 = unsafe { bump.call(client_rt.thread_ptr(), 10) };
            let r3 = unsafe { get.call(client_rt.thread_ptr()) };
            clear_current_runtime();
            vec![r1, r2, r3]
        });

        let _ = unsafe {
            server_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["tcp_node_main"]),
                )
                .unwrap()
                .call(server_rt.thread_ptr(), fd)
        };

        let results = client.join().unwrap();
        clear_current_runtime();
        assert_eq!(
            results,
            vec![5, 15, 15],
            "remote handlers mutate the node's own state over TCP"
        );
    }

    /// Generic, CHECKED, open-world decode: one `decode::<T>(bytes)`
    /// works for any data type the caller names, returns a `Result`, and
    /// rejects a type mismatch as a value (not a crash). Round-trips an
    /// Int and a struct, and catches decoding struct-bytes as the wrong
    /// type.
    #[test]
    fn ail_generic_checked_decode() {
        init();
        let src = "
            struct Pair { a: Int, b: Int }

            def enc_int() -> Bytes = wire_encode(7)
            def dec_int(b: Bytes) -> Int =
                match decode::<Int>(b) { Result::Ok(v) => v, Result::Err(_e) => 0 - 1 }

            def enc_pair() -> Bytes = wire_encode(Pair { a: 3, b: 4 })
            def dec_pair_sum(b: Bytes) -> Int =
                match decode::<Pair>(b) { Result::Ok(p) => p.a + p.b, Result::Err(_e) => 0 - 1 }

            // Decoding Pair bytes as Int must be a typed DecodeError, not a
            // crash. Match on the actual enum variants.
            def dec_mismatch(b: Bytes) -> Int =
                match decode::<Int>(b) {
                    Result::Ok(_v) => 0 - 99,
                    Result::Err(e) => match e {
                        DecodeError::TypeMismatch => 1,
                        DecodeError::Malformed => 2,
                    },
                }
        ";

        let ctx = Context::create();
        let full = format!("{}\n{}", crate::stdlib::SOURCE, src);
        let m = parse_module(&full).unwrap();
        let r = resolve_module(&m).unwrap();
        let names: std::collections::HashMap<String, Hash> =
            r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let rt = Runtime::new_with_metadata(
            cm.closure_type_infos.clone(),
            cm.shape_registry.clone(),
            cm.shape_meta.clone(),
            cm.shape_by_type_id.clone(),
        );
        let jit = Jit::new(cm, &rt).unwrap();

        // decode builds the user Result, so it needs the at() runtime
        // binding (Result/Failure layouts) and the current runtime.
        install_current_runtime(&rt);
        let rb = r.at_binding.as_ref().expect("stdlib uses at(), so binding exists");
        let binding = build_at_runtime_binding(&rt, rb).expect("build at runtime binding");
        install_current_at_binding(&binding);

        let fn0 = |n: &str| unsafe {
            jit.engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> *mut u8>(
                    &crate::codegen::def_symbol(&names[n]),
                )
                .unwrap()
        };
        let fn1i = |n: &str| unsafe {
            jit.engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names[n]),
                )
                .unwrap()
        };

        let int_bytes = unsafe { fn0("enc_int").call(rt.thread_ptr()) };
        assert_eq!(unsafe { fn1i("dec_int").call(rt.thread_ptr(), int_bytes) }, 7);

        let pair_bytes = unsafe { fn0("enc_pair").call(rt.thread_ptr()) };
        assert_eq!(unsafe { fn1i("dec_pair_sum").call(rt.thread_ptr(), pair_bytes) }, 7);

        // Same pair bytes, decoded as Int -> type-mismatch error code 1.
        let pair_bytes2 = unsafe { fn0("enc_pair").call(rt.thread_ptr()) };
        assert_eq!(unsafe { fn1i("dec_mismatch").call(rt.thread_ptr(), pair_bytes2) }, 1);

        clear_current_runtime();
        clear_current_at_binding();
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

    /// The node-`state` remote-handler model: a node owns a `state`
    /// (`counter`) and handler defs that close over it; remote participants
    /// ship `|| handle(msg)` closures, which run ON the node and mutate the
    /// node's OWN cell. Distinct participants see each other's writes.
    ///
    /// The atom never crosses the wire: only the closure travels; `counter`
    /// resolves by identity on the server via `ai_state_get`. The shipped
    /// lambda captures only `d: Int` and references `handle` (TopRef) +
    /// `counter` (StateRef, not a capture), so the bare-atom guardrail does
    /// not fire.
    #[test]
    fn node_state_remote_handler_shared() {
        init();
        let extra = "
            enum Cmd { Bump(Int), Get }
            state rh_counter: Atom<Int> = atom(0)
            def rh_handle(c: Cmd) -> Int =
                match c {
                    Cmd::Bump(d) => swap(rh_counter, |n: Int| n + d),
                    Cmd::Get => deref(rh_counter),
                }
            // Client-side thunk builders: ship `|| rh_handle(msg)`.
            def rh_call_bump(d: Int) -> fn() -> Int = || rh_handle(Cmd::Bump(d))
            def rh_call_get() -> fn() -> Int = || rh_handle(Cmd::Get)
        ";

        // Server: full source, so it has `rh_handle` + an installed
        // `rh_counter` cell (the Jit::new install pass ran).
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_stdlib_runtime(&server_ctx, extra);
        let _ = server_jit;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));

        let (mut client_chan, mut server_chan) = InProcessChannel::pair();
        let runtime_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || {
            loop {
                match unsafe { serve_one(&runtime_for_thread.0, &mut server_chan) } {
                    Ok(()) => continue,
                    Err(_) => break,
                }
            }
        });

        // Client: builds closures and ships them. Same source => the
        // shipped lambda's code hash + `rh_handle`'s hash already exist on
        // the server, so nothing is fetched; the server resolves both
        // locally and the swap hits the server's own counter.
        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_stdlib_runtime(&client_ctx, extra);
        let kb = KnowledgeBase::new();

        let bump = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["rh_call_bump"]),
                )
                .unwrap()
        };
        let get = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["rh_call_get"]),
                )
                .unwrap()
        };

        let ship = |closure: *mut u8, chan: &mut InProcessChannel| -> i64 {
            let r = unsafe {
                at_remote_on_channel(&client_rt, &kb, chan, closure as *const u8).unwrap()
            };
            unsafe { crate::runtime::ai_gc_unbox_int(r) }
        };

        // Two participants bump (+5, +10); a third reads the shared total.
        let c1 = unsafe { bump.call(client_rt.thread_ptr(), 5) };
        assert_eq!(ship(c1, &mut client_chan), 5, "first remote bump");
        let c2 = unsafe { bump.call(client_rt.thread_ptr(), 10) };
        assert_eq!(ship(c2, &mut client_chan), 15, "second bump sees the first");
        let c3 = unsafe { get.call(client_rt.thread_ptr()) };
        assert_eq!(ship(c3, &mut client_chan), 15, "remote read sees shared node state");

        drop(client_chan);
        server_handle.join().unwrap();
    }

    /// Deployment as a LIBRARY over at(): the stdlib `svc_*` slot holds
    /// a stack of handler versions in node state. `at(node, ||
    /// install(...))` ships the new handler's code via the ordinary
    /// code-fetch path and pushes it; `at(node, || handle(x))` runs the
    /// node's CURRENT version (callee-version semantics — the caller
    /// doesn't name a handler hash); rollback pops to the previous
    /// version, whose code is still JIT-resident. The slot type pins
    /// the interface, so an ill-typed handler can't even typecheck.
    #[test]
    fn at_driven_install_upgrade_rollback() {
        init();
        let extra = "
            state svc: Atom<List<fn(Int) -> Int>> = atom_new(svc_slot())
            def sv_install(h: fn(Int) -> Int) -> Int = svc_install(svc, h)
            def sv_rollback() -> Int = svc_rollback(svc)
            def sv_handle(x: Int) -> Int = {
                let f = svc_current(svc);
                f(x)
            }
            // Handler versions are named defs — content-addressed, like
            // any deployable code.
            def handler_v1(x: Int) -> Int = x + 1
            def handler_v2(x: Int) -> Int = x * 10
            // Client-side thunk builders: the at() payloads.
            def c_install_v1() -> fn() -> Int = || sv_install(handler_v1)
            def c_install_v2() -> fn() -> Int = || sv_install(handler_v2)
            def c_handle(x: Int) -> fn() -> Int = || sv_handle(x)
            def c_rollback() -> fn() -> Int = || sv_rollback()
        ";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_stdlib_runtime(&server_ctx, extra);
        let _ = server_jit;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));

        let (mut client_chan, mut server_chan) = InProcessChannel::pair();
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
        let (client_rt, client_jit, names) = make_stdlib_runtime(&client_ctx, extra);
        let kb = KnowledgeBase::new();

        let mk0 = |name: &str| unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> *mut u8>(
                    &crate::codegen::def_symbol(&names[name]),
                )
                .unwrap()
        };
        let install_v1 = mk0("c_install_v1");
        let install_v2 = mk0("c_install_v2");
        let rollback = mk0("c_rollback");
        let handle = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["c_handle"]),
                )
                .unwrap()
        };

        let ship = |closure: *mut u8, chan: &mut InProcessChannel| -> i64 {
            let r = unsafe {
                at_remote_on_channel(&client_rt, &kb, chan, closure as *const u8).unwrap()
            };
            unsafe { crate::runtime::ai_gc_unbox_int(r) }
        };

        // Deploy v1 (x + 1): one version live.
        let c = unsafe { install_v1.call(client_rt.thread_ptr()) };
        assert_eq!(ship(c, &mut client_chan), 1, "v1 installed");
        let c = unsafe { handle.call(client_rt.thread_ptr(), 5) };
        assert_eq!(ship(c, &mut client_chan), 6, "v1 handles");

        // Upgrade to v2 (x * 10): same call, new behavior, history kept.
        let c = unsafe { install_v2.call(client_rt.thread_ptr()) };
        assert_eq!(ship(c, &mut client_chan), 2, "v2 installed on top");
        let c = unsafe { handle.call(client_rt.thread_ptr(), 5) };
        assert_eq!(ship(c, &mut client_chan), 50, "v2 handles the same calls");
        let c = unsafe { handle.call(client_rt.thread_ptr(), 7) };
        assert_eq!(ship(c, &mut client_chan), 70);

        // Rollback: instant flip to v1, still JIT-resident.
        let c = unsafe { rollback.call(client_rt.thread_ptr()) };
        assert_eq!(ship(c, &mut client_chan), 1, "back to one version");
        let c = unsafe { handle.call(client_rt.thread_ptr(), 5) };
        assert_eq!(ship(c, &mut client_chan), 6, "v1 behavior again");

        drop(client_chan);
        server_handle.join().unwrap();
    }

    /// The cache fix: two BYTE-IDENTICAL stateful Calls must each run the
    /// mutation. `|| rh_handle(Bump(5))` shipped twice has an identical
    /// payload (same lambda hash + same capture 5), so the pure-thunk cache
    /// would short-circuit the second call and skip its swap. Because the
    /// thunk transitively touches `rh_counter`, it's marked stateful and
    /// bypasses the cache: counter goes 5 then 10.
    #[test]
    fn node_state_stateful_calls_bypass_cache() {
        init();
        let extra = "
            enum Cmd { Bump(Int), Get }
            state cc_counter: Atom<Int> = atom(0)
            def cc_handle(c: Cmd) -> Int =
                match c {
                    Cmd::Bump(d) => swap(cc_counter, |n: Int| n + d),
                    Cmd::Get => deref(cc_counter),
                }
            def cc_call_bump(d: Int) -> fn() -> Int = || cc_handle(Cmd::Bump(d))
        ";
        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_stdlib_runtime(&server_ctx, extra);
        let _ = server_jit;
        let server_runtime = Arc::new(RuntimeHandle(server_rt));
        let misses_handle = server_runtime.0.cache_misses.clone();

        let (mut client_chan, mut server_chan) = InProcessChannel::pair();
        let runtime_for_thread = server_runtime.clone();
        let server_handle = std::thread::spawn(move || loop {
            match unsafe { serve_one(&runtime_for_thread.0, &mut server_chan) } {
                Ok(()) => continue,
                Err(_) => break,
            }
        });

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_stdlib_runtime(&client_ctx, extra);
        let kb = KnowledgeBase::new();
        let bump = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["cc_call_bump"]),
                )
                .unwrap()
        };
        let ship = |c: *mut u8, ch: &mut InProcessChannel| -> i64 {
            let r =
                unsafe { at_remote_on_channel(&client_rt, &kb, ch, c as *const u8).unwrap() };
            unsafe { crate::runtime::ai_gc_unbox_int(r) }
        };

        // Two identical Bump(5) calls. Distinct closures, identical bytes.
        let a = unsafe { bump.call(client_rt.thread_ptr(), 5) };
        assert_eq!(ship(a, &mut client_chan), 5, "first stateful bump");
        let b = unsafe { bump.call(client_rt.thread_ptr(), 5) };
        assert_eq!(
            ship(b, &mut client_chan),
            10,
            "identical stateful call must re-run the swap, not hit the cache"
        );

        drop(client_chan);
        server_handle.join().unwrap();

        // Both calls missed (the cache was bypassed, not consulted-and-hit).
        let misses = misses_handle.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(misses, 0, "stateful calls never consult the cache");
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
        // reserves the trailing slots for BoxedInt + String + Array +
        // Atom + Bytes.
        assert_eq!(server_rt.shape_by_type_id.len(), 5);

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
                    Result::Ok(n) => n,
                    Result::Err(_) => 0 - 1,
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
                    Result::Ok(n) => n,
                    Result::Err(_) => 0 - 1,
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

    /// A shipped thunk that PANICS on the node:
    /// - the panic propagates out of JIT'd code as a value (no abort),
    /// - the serve boundary reports it in a `KIND_ERR` frame,
    /// - the client maps it onto `Failure::Crashed(node)` — an ordinary
    ///   enum value the program matches on,
    /// - and the NODE KEEPS SERVING: the next call on the same node
    ///   succeeds. One bad request fails one request.
    #[test]
    fn at_returns_err_crashed_on_remote_panic_and_node_survives() {
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

            def boom(x: Int) -> Int =
                if x < 0 { panic(\"remote boom\") } else { x + 1 }

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            def run(node: Node, x: Int) -> Int =
                match at(node, || boom(x)) {
                    Result::Ok(n) => n,
                    Result::Err(f) => match f {
                        Failure::Crashed(_) => 0 - 2,
                        Failure::Unreachable(_) => 0 - 3,
                        Failure::CodeMissing(_) => 0 - 4,
                        Failure::Cancelled(_) => 0 - 5,
                    },
                }
        ";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let server_port = listener.local_addr().unwrap().port() as i64;
        // TWO requests on one server: the panicking one, then a healthy
        // one — proving the node survives the first.
        let server_handle = unsafe {
            serve_n_in_thread(Arc::new(RuntimeHandle(server_rt)), listener, 2)
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

        // First call: the thunk panics on the node → Failure::Crashed.
        let crashed = unsafe { run.call(client_rt.thread_ptr(), node, -7) };
        assert_eq!(crashed, -2, "remote panic should arrive as Failure::Crashed");
        // The client-side runtime is clean (the panic happened on the
        // node and arrived as a value, not as a local pending panic).
        assert!(unsafe { crate::runtime::take_pending_panic(client_rt.thread_ptr()) }.is_none());

        // Second call on the SAME node: it survived and still works.
        let node2 = unsafe {
            make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port)
        };
        let ok = unsafe { run.call(client_rt.thread_ptr(), node2, 7) };
        assert_eq!(ok, 8, "node should keep serving after a handler panic");

        server_handle.join().unwrap().expect("server stayed healthy");
        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// Serializes the tests that mutate the process-global at() config
    /// (deadline / retry knobs) so they don't race each other.
    static AT_CONFIG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A node that accepts the connection and then never replies: the
    /// call deadline fires and surfaces as the user's declared
    /// `Failure::TimedOut(node)` — slow and dead are distinguishable,
    /// and the caller is never blocked forever.
    #[test]
    fn at_times_out_on_silent_node() {
        init();
        let _config = AT_CONFIG_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let src = "
            struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
            enum Failure {
                Unreachable(Node),
                Crashed(Node),
                CodeMissing(Node),
                Cancelled(Node),
                TimedOut(Node),
            }
            enum Result<T, E> { Ok(T), Err(E) }

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            def run(node: Node) -> Int =
                match at(node, || 42) {
                    Result::Ok(n) => n,
                    Result::Err(f) => match f {
                        Failure::TimedOut(_) => 0 - 9,
                        Failure::Unreachable(_) => 0 - 3,
                        Failure::Crashed(_) => 0 - 2,
                        Failure::CodeMissing(_) => 0 - 4,
                        Failure::Cancelled(_) => 0 - 5,
                    },
                }
        ";

        let client_ctx = Context::create();
        let (client_rt, client_jit, names) = make_compiled_runtime(&client_ctx, src);
        install_current_runtime(&client_rt);
        let kb = KnowledgeBase::new();
        install_current_knowledge_base(&kb);
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let rb = r.at_binding.as_ref().expect("at_binding");
        assert!(
            rb.timed_out_variant_index.is_some(),
            "resolver should discover the optional TimedOut variant"
        );
        let rt_binding =
            build_at_runtime_binding(&client_rt, rb).expect("rt binding");
        install_current_at_binding(&rt_binding);

        // A listener that accepts and then sits silent (no reply).
        let listener = bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port() as i64;
        let silent = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            // Hold the connection open well past the client's deadline.
            std::thread::sleep(std::time::Duration::from_millis(4_000));
            drop(stream);
        });

        set_at_call_deadline_ms(500);
        set_at_pure_retries(0); // isolate the deadline from the retry logic

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
        let node = unsafe { make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, port) };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let started = std::time::Instant::now();
        let result = unsafe { run.call(client_rt.thread_ptr(), node) };
        let elapsed = started.elapsed();

        // Restore the ambient config before asserting so a failure
        // doesn't leak the short deadline into other tests.
        reset_at_config();

        assert_eq!(result, -9, "expected the TimedOut arm");
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "deadline should fire promptly, took {:?}",
            elapsed
        );

        silent.join().unwrap();
        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// Effects-driven auto-retry: a PURE thunk whose first attempt dies
    /// on a dropped connection is transparently re-sent and succeeds.
    /// (A stateful thunk would never be auto-retried — re-sending could
    /// double-apply its effects; purity comes from the same effects
    /// knowledge the at-cache uses.)
    #[test]
    fn at_retries_pure_thunk_after_dropped_connection() {
        init();
        let _config = AT_CONFIG_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let src = "
            struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
            enum Failure {
                Unreachable(Node),
                Crashed(Node),
                CodeMissing(Node),
                Cancelled(Node),
            }
            enum Result<T, E> { Ok(T), Err(E) }

            def work(x: Int) -> Int = x * 7

            def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
                Node { a: a, b: b, c: c, d: d, port: port }

            def run(node: Node, x: Int) -> Int =
                match at(node, || work(x)) {
                    Result::Ok(n) => n,
                    Result::Err(_) => 0 - 1,
                }
        ";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let server_rt = Arc::new(RuntimeHandle(server_rt));
        let listener = bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port() as i64;
        let flaky = {
            let rt = server_rt.clone();
            std::thread::spawn(move || {
                // First connection: drop it without reading or replying —
                // the client sees a transport failure.
                let (first, _) = listener.accept().unwrap();
                drop(first);
                // Second connection (the retry): serve it properly.
                let mut second = accept_one(&listener).unwrap();
                unsafe { serve_one(&rt.0, &mut second) }
            })
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

        set_at_pure_retries(2);
        // No deadline for this test: under AI_LANG_GC_STRESS the serve
        // path can legitimately exceed the default 30s, and a timeout
        // here would burn the retry budget on timing noise rather than
        // exercising the dropped-connection retry.
        set_at_call_deadline_ms(0);

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
        let node = unsafe { make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, port) };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let result = unsafe { run.call(client_rt.thread_ptr(), node, 6) };
        reset_at_config();
        assert_eq!(result, 42, "pure thunk should be retried to success");

        flaky.join().unwrap().expect("server side ok");
        clear_at_conn_cache();
        clear_current_runtime();
        clear_current_knowledge_base();
        clear_current_at_binding();
    }

    /// `at_async(node, thunk)` ships the thunk from a background thread
    /// and returns a `ThreadHandle<Result<T, Failure>>`; the existing
    /// `join` awaits it. Two calls overlap in flight; both Results come
    /// back as ordinary values with the same failure model as `at()`.
    #[test]
    fn at_async_join_returns_result_values() {
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

            def run(node: Node, x: Int) -> Int = {
                let h1 = at_async(node, || work(x));
                let h2 = at_async(node, || work(x + 1));
                let a = match thread_join(h1) {
                    Result::Ok(n) => n,
                    Result::Err(_) => 0 - 1,
                };
                let b = match thread_join(h2) {
                    Result::Ok(n) => n,
                    Result::Err(_) => 0 - 1,
                };
                a * 1000 + b
            }

            def run_dead(node: Node) -> Int = {
                let h = at_async(node, || 42);
                match thread_join(h) {
                    Result::Ok(n) => n,
                    Result::Err(f) => match f {
                        Failure::Unreachable(_) => 0 - 3,
                        Failure::Crashed(_) => 0 - 2,
                        Failure::CodeMissing(_) => 0 - 4,
                        Failure::Cancelled(_) => 0 - 5,
                    },
                }
            }
        ";

        let server_ctx = Context::create();
        let (server_rt, server_jit, _) = make_compiled_runtime(&server_ctx, src);
        let _ = server_jit;
        let listener = bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port() as i64;
        let server_handle = unsafe {
            serve_n_in_thread(Arc::new(RuntimeHandle(server_rt)), listener, 2)
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
        let node = unsafe { make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, port) };
        let run = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
        };
        let result = unsafe { run.call(client_rt.thread_ptr(), node, 4) };
        // work(4) = 43, work(5) = 53 → 43 * 1000 + 53.
        assert_eq!(result, 43_053, "both async results should round-trip");

        server_handle.join().unwrap().expect("server ok");

        // And against a dead node: the failure arrives as a VALUE through
        // the same handle/join path (no hang — the connect timeout and
        // refused connection surface as Err(Unreachable)).
        let dead_port = {
            let l = bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port() as i64
        };
        let dead_node =
            unsafe { make_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, dead_port) };
        let run_dead = unsafe {
            client_jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> i64>(
                    &crate::codegen::def_symbol(&names["run_dead"]),
                )
                .unwrap()
        };
        let dead = unsafe { run_dead.call(client_rt.thread_ptr(), dead_node) };
        assert_eq!(dead, -3, "dead node should yield Err(Unreachable) via join");

        clear_at_conn_cache();
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
                    Result::Ok(v) => v,
                    Result::Err(_) => 0 - 1,
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
                    Result::Ok(v) => v,
                    Result::Err(_) => 0 - 1,
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
                List::Cons(ListCell { head: int_list_range(1, 5), tail:
                List::Cons(ListCell { head: int_list_range(5, 9), tail:
                List::Cons(ListCell { head: int_list_range(9, 13), tail: List::Nil })
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
                List::Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail:
                List::Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail:
                List::Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail: List::Nil })
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
                    Result::Ok(f) => v,
                    Result::Err(e) => 0 - 3
                }
            def run_read(rp: RemotePtr) -> Int =
                match remote_read_i64(rp) {
                    Result::Ok(v) => run_free(rp, v),
                    Result::Err(e) => 0 - 2
                }
            def run_write(rp: RemotePtr) -> Int =
                match remote_write_i64(rp, 12345) {
                    Result::Ok(w) => run_read(rp),
                    Result::Err(e) => 0 - 1
                }
            def run(port: Int) -> Int = {
                let node = tcp_node(127, 0, 0, 1, port);
                match remote_alloc(node, 8) {
                    Result::Ok(rp) => run_write(rp),
                    Result::Err(e) => 0 - 4
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
                list_reverse(indices_acc(n, 0, List::Nil))

            def indices_acc(n: Int, i: Int, acc: List<List<Int>>) -> List<List<Int>> =
                if i >= n { acc } else {
                    indices_acc(n, i + 1, List::Cons(ListCell {
                        head: List::Cons(ListCell { head: i, tail: List::Nil }),
                        tail: acc,
                    }))
                }

            def make_pool(port: Int, n: Int) -> List<Node> =
                make_pool_acc(port, n, List::Nil)

            def make_pool_acc(port: Int, n: Int, acc: List<Node>) -> List<Node> =
                if n <= 0 { acc } else {
                    make_pool_acc(port, n - 1, List::Cons(ListCell {
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
                List::Cons(ListCell { head: List::Cons(ListCell { head: 10, tail: List::Nil }), tail:
                List::Cons(ListCell { head: List::Cons(ListCell { head: 20, tail: List::Nil }), tail:
                List::Cons(ListCell { head: List::Cons(ListCell { head: 30, tail: List::Nil }), tail: List::Nil })
                })
                })

            def make_pool(port: Int) -> List<Node> =
                List::Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail:
                List::Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail:
                List::Cons(ListCell { head: tcp_node(127, 0, 0, 1, port), tail: List::Nil })
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
                    Result::Ok(p) => p.a + p.b,
                    Result::Err(_) => 0 - 1,
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
            def build(n: Int) -> List<Int> = build_acc(n, List::Nil)
            def build_acc(n: Int, acc: List<Int>) -> List<Int> =
                if n <= 0 { acc } else {
                    build_acc(n - 1, List::Cons(ListCell { head: n, tail: acc }))
                }

            def run(node: Node, n: Int) -> Int =
                match at(node, || build(n)) {
                    Result::Ok(xs) => list_foldl(xs, 0, |acc: Int, x: Int| acc + x),
                    Result::Err(_) => 0 - 1,
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
            def build(n: Int) -> List<Int> = build_acc(n, List::Nil)
            def build_acc(n: Int, acc: List<Int>) -> List<Int> =
                if n <= 0 { acc } else {
                    build_acc(n - 1, List::Cons(ListCell { head: n, tail: acc }))
                }

            // sum_list lives on both sides; the closure captures
            // `xs` (a heap List<Int>) and the server calls sum_list
            // on it.
            def sum_list(xs: List<Int>) -> Int =
                list_foldl(xs, 0, |acc: Int, x: Int| acc + x)

            def run(node: Node, n: Int) -> Int = {
                let xs = build(n);
                match at(node, || sum_list(xs)) {
                    Result::Ok(v) => v,
                    Result::Err(_) => 0 - 1,
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
                    Result::Ok(v) => v,
                    Result::Err(_) => 0,
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
        //
        // The node is re-created per generation: `node` here is a raw
        // heap pointer held in a Rust local, which is NOT a GC root.
        // Each step_one_gen call collects (every alloc, under
        // AI_LANG_GC_STRESS) and relocates the node; a stale host-held
        // pointer would read garbage on the next generation.
        let _ = node;
        let mut counts = vec![3i64];
        for _ in 0..generations {
            let node = unsafe {
                tcp_node.call(client_rt.thread_ptr(), 127, 0, 0, 1, server_port)
            };
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
                    Result::Ok(p) => 0,
                    Result::Err(e) => 0 - 1
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
                    Result::Ok(n) => n,
                    Result::Err(e) => 0 - 1
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

    /// A thunk may not CAPTURE a bare `Atom`: a node-resident mutable cell
    /// shipped by value would fork. Shared node state must be a `state`
    /// binding reached by reference on the owning node.
    #[test]
    fn at_thunk_capturing_atom_is_rejected() {
        init();
        let src = format!(
            "{}
             def run(node: Node) -> Int = {{
                let a = atom_new(0);
                match at(node, || atom_load(a)) {{
                    Result::Ok(n) => n,
                    Result::Err(e) => 0 - 1
                }}
             }}",
            AT_PRELUDE
        );
        let err = typecheck_src(&src).expect_err("Atom-capturing thunk must be rejected");
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("Atom") && msg.to_lowercase().contains("captur"),
            "error should explain the Atom capture ban, got: {}",
            msg
        );
    }

    /// A thunk may not RETURN a bare `Atom` either.
    #[test]
    fn at_thunk_returning_atom_is_rejected() {
        init();
        let src = format!(
            "{}
             def run(node: Node) -> Int =
                match at(node, || atom_new(0)) {{
                    Result::Ok(a) => 0,
                    Result::Err(e) => 0 - 1
                }}",
            AT_PRELUDE
        );
        let err = typecheck_src(&src).expect_err("Atom-returning thunk must be rejected");
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("Atom") && msg.to_lowercase().contains("return"),
            "error should explain the Atom return ban, got: {}",
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
                    Result::Ok(n) => n,
                    Result::Err(e) => 0 - 1
                }}",
            AT_PRELUDE
        );
        typecheck_src(&src).expect("a Ptr-free thunk must still typecheck");
    }
}
