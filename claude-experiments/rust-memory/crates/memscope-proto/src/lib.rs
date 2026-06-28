//! Shared wire vocabulary between the in-process agent (`memscope-core`) and any
//! consumer (the CLI, the eventual UI). Two layers live here:
//!
//! * **Hot-path POD** ([`RawEvent`], [`EventKind`]) — `Copy`, allocation-free,
//!   cheap to push into a ring buffer from inside the global allocator.
//! * **Resolved / serializable** ([`Snapshot`], [`SiteInfo`], [`TypeInfo`], …) —
//!   produced off the hot path once DWARF symbolication has run. These carry the
//!   human-facing detail (stack frames, concrete type names) and serialize for
//!   transport or on-disk heap dumps.
//!
//! IDs are small integers interned by the agent. They are stable for the life of
//! a process; a consumer resolves them through the tables shipped in a
//! [`Snapshot`] (or via incremental table updates over the stream).

#![forbid(unsafe_code)]

pub mod recfmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Interned allocation site (a captured stack trace). `u32::MAX` is reserved as
/// "no site" (e.g. backtrace capture was disabled for this allocation).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SiteId(pub u32);

impl SiteId {
    pub const NONE: SiteId = SiteId(u32::MAX);
    #[inline]
    pub fn is_some(self) -> bool {
        self != SiteId::NONE
    }
}

/// Interned resolved type. `u32::MAX` is reserved as "type not yet resolved".
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TypeId(pub u32);

impl TypeId {
    pub const UNKNOWN: TypeId = TypeId(u32::MAX);
    #[inline]
    pub fn is_known(self) -> bool {
        self != TypeId::UNKNOWN
    }
}

impl Default for TypeId {
    fn default() -> Self {
        TypeId::UNKNOWN
    }
}

impl Default for SiteId {
    fn default() -> Self {
        SiteId::NONE
    }
}

/// What a [`RawEvent`] represents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum EventKind {
    Alloc = 0,
    Dealloc = 1,
    /// A realloc is emitted as a Dealloc(old) immediately followed by an
    /// Alloc(new); this variant marks the Alloc half so consumers can stitch
    /// growth history rather than treating it as a fresh allocation.
    ReallocGrow = 2,
    /// Enter a metadata scope. The scope's interned context id rides in the
    /// `site` field; the key/value pairs are in the recording's `TAG_META` table.
    MetaEnter = 3,
    /// Leave the innermost metadata scope (LIFO). `site` carries the same context
    /// id for robustness.
    MetaExit = 4,
    /// A named checkpoint in the stream (`memscope::mark("label")`). The interned
    /// label id rides in the `site` field; the label string is in the recording's
    /// `TAG_MARK` table. Carries no allocation — it's a timestamped fencepost the
    /// reader uses to reconstruct the live set at a semantic moment.
    Mark = 5,
}

/// A metadata value attached to a scope via `meta!`. Kept to a few flat,
/// cheap-to-serialize cases; anything structured is stringified at the call site.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MetaValue {
    Str(String),
    Int(i64),
    Uint(u64),
    F64(f64),
    Bool(bool),
}

impl MetaValue {
    /// Render for grouping/filtering/display.
    pub fn to_display(&self) -> String {
        match self {
            MetaValue::Str(s) => s.clone(),
            MetaValue::Int(i) => i.to_string(),
            MetaValue::Uint(u) => u.to_string(),
            MetaValue::F64(f) => f.to_string(),
            MetaValue::Bool(b) => b.to_string(),
        }
    }
}

impl From<&str> for MetaValue {
    fn from(s: &str) -> Self {
        MetaValue::Str(s.to_string())
    }
}
impl From<String> for MetaValue {
    fn from(s: String) -> Self {
        MetaValue::Str(s)
    }
}
impl From<&String> for MetaValue {
    fn from(s: &String) -> Self {
        MetaValue::Str(s.clone())
    }
}
impl From<bool> for MetaValue {
    fn from(b: bool) -> Self {
        MetaValue::Bool(b)
    }
}
impl From<f64> for MetaValue {
    fn from(f: f64) -> Self {
        MetaValue::F64(f)
    }
}
macro_rules! meta_from_int {
    ($($t:ty),*) => { $( impl From<$t> for MetaValue { fn from(v: $t) -> Self { MetaValue::Int(v as i64) } } )* };
}
macro_rules! meta_from_uint {
    ($($t:ty),*) => { $( impl From<$t> for MetaValue { fn from(v: $t) -> Self { MetaValue::Uint(v as u64) } } )* };
}
meta_from_int!(i8, i16, i32, i64, isize);
meta_from_uint!(u8, u16, u32, u64, usize);

/// The allocation-free event pushed onto the hot-path ring buffer. `#[repr(C)]`
/// + `Copy` so it can also be memcpy'd across a shared-memory transport.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct RawEvent {
    pub kind: EventKind,
    /// Monotonic per-process sequence number. Total order across threads.
    pub seq: u64,
    /// Nanoseconds since the agent's start instant.
    pub ts_nanos: u64,
    /// Allocation address (the pointer the program sees).
    pub addr: u64,
    /// Bytes. For Dealloc this is the freed size (from the live table).
    pub size: u64,
    pub align: u32,
    pub site: SiteId,
    /// OS thread id, truncated to 32 bits.
    pub thread: u32,
}

/// One frame of a captured stack trace, after symbolication.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Frame {
    /// Instruction pointer (return address) for this frame.
    pub ip: u64,
    /// Demangled function name, if symbolicated.
    pub function: Option<String>,
    /// Source file, if line info is present.
    pub file: Option<String>,
    pub line: Option<u32>,
    /// True if this frame was inlined into its caller (from DWARF inline info).
    pub inlined: bool,
}

/// A resolved allocation site: the stack trace plus the recovered concrete type
/// of whatever was allocated there.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SiteInfo {
    pub id: u32,
    pub frames: Vec<Frame>,
    /// The recovered allocated type, if DWARF type recovery succeeded.
    pub ty: TypeId,
    /// The allocation "shape" we recognized (Box, Vec, Rc, …), if any.
    pub shape: Option<AllocShape>,
}

/// Recognized allocation container shape, recovered from the frame that pins the
/// element/payload type. Tells the consumer how to interpret `SiteInfo::ty`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AllocShape {
    /// `Box<T>` — `ty` is `T`.
    Boxed,
    /// `Vec<T>` / `RawVec<T>` backing store — `ty` is the element `T`.
    Vec,
    /// `Rc<T>` inner allocation — `ty` is `T`.
    Rc,
    /// `Arc<T>` inner allocation — `ty` is `T`.
    Arc,
    /// `HashMap`/`HashSet` table — `ty` is the entry type.
    HashTable,
    /// `String` backing store — `ty` is `u8`.
    StringBuf,
    /// Recognized as a heap container but element type not pinned.
    Other,
}

/// A resolved concrete type.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TypeInfo {
    pub id: u32,
    /// Fully-qualified name, e.g. `alloc::vec::Vec<u64, alloc::alloc::Global>`.
    pub name: String,
    /// `size_of::<T>()` if known from DWARF.
    pub size: Option<u64>,
}

/// A single live allocation in a heap snapshot.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LiveAlloc {
    pub addr: u64,
    pub size: u64,
    pub align: u32,
    pub site: SiteId,
    pub ts_nanos: u64,
    pub thread: u32,
}

/// A full heap dump: every live allocation plus the tables needed to interpret
/// their site/type ids. Self-contained, so it can be written to disk and
/// explored posthoc with no live process.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Snapshot {
    /// Wall-clock-ish: nanoseconds since agent start when the dump was taken.
    pub taken_at_nanos: u64,
    /// If sampling was active, each tracked allocation statistically represents
    /// this many real allocations; consumers scale aggregates by it.
    pub sample_scale: f64,
    pub live: Vec<LiveAlloc>,
    pub sites: Vec<SiteInfo>,
    pub types: Vec<TypeInfo>,
    /// Live bytes currently tracked (sum of `live[*].size`).
    pub total_live_bytes: u64,
    /// Allocations that overflowed the event ring and were dropped from the
    /// stream (the live table is still exact in Full mode).
    pub dropped_events: u64,
}

// --- heap reference graph ----------------------------------------------------

/// A node in the heap reference graph: one live allocation.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphNode {
    pub addr: u64,
    pub size: u64,
    pub ty: Option<String>,
    pub shape: Option<AllocShape>,
    /// Bytes that would be freed if this node became unreachable — i.e. the
    /// total size of everything it dominates (itself included).
    pub retained_size: u64,
    /// Index of this node's immediate dominator, or -1 for a root / the virtual
    /// super-root's children.
    pub idom: i64,
    pub in_degree: u32,
    pub out_degree: u32,
}

/// A directed reference edge: `from` holds a pointer (at byte `offset` within it)
/// into `to`.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphEdge {
    pub from: u32,
    pub to: u32,
    pub offset: u64,
}

/// The reconstructed heap reference graph with retained sizes + dominators.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HeapGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    /// Indices of nodes referenced by no other tracked allocation (approximate
    /// GC roots — real roots on the stack/statics aren't tracked).
    pub roots: Vec<u32>,
    /// Sum of all node sizes (the live bytes the graph covers).
    pub total_bytes: u64,
    /// Allocations whose type/layout we couldn't walk (enums-in-variant skipped,
    /// HashMap interiors, unknown types) — edges out of these are incomplete.
    pub opaque_nodes: u32,
}

// --- transport messages ------------------------------------------------------

/// Serializable view of the agent's aggregate counters.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StatsView {
    pub live_bytes: u64,
    pub total_allocs: u64,
    pub total_alloc_bytes: u64,
    pub dropped_events: u64,
    /// 0 = Off, 1 = Full, 2 = Sampled.
    pub mode: u8,
    pub sample_rate: u32,
}

/// A request from a consumer (CLI / UI) to the in-process agent. One JSON
/// object per line.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClientMsg {
    /// Switch mode (0=Off, 1=Full, 2=Sampled).
    SetMode(u8),
    SetSampleRate(u32),
    SetCaptureSites(bool),
    /// Request current aggregate counters.
    GetStats,
    /// Request a full, type-resolved heap dump.
    GetSnapshot,
    /// Drain up to `max` queued raw events (live stream).
    PollEvents { max: usize },
    /// Resolve the given site ids to typed labels/frames (incremental table for
    /// the live view).
    ResolveSites { ids: Vec<u32> },
    /// Reconstruct the heap reference graph (edges, retained sizes, dominators).
    GetGraph,
}

/// A reply from the agent. One JSON object per line.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ServerMsg {
    /// Sent once on connect.
    Hello { pid: u32, agent_version: String },
    Stats(StatsView),
    Snapshot(Box<Snapshot>),
    Events(Vec<RawEvent>),
    /// Resolved site table (answer to `ResolveSites`, or pushed with a snapshot).
    Sites(Vec<SiteInfo>),
    /// Type table referenced by resolved sites.
    Types(Vec<TypeInfo>),
    /// The reconstructed heap reference graph.
    Graph(Box<HeapGraph>),
    Error(String),
}
