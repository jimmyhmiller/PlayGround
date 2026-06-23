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
}

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
