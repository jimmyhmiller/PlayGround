//! Canonical in-memory trace data model. Every format crate normalizes into a
//! [`Profile`] via [`ProfileBuilder`]. Slices are stored struct-of-arrays, sorted
//! by `(track, depth, start_ns)`, with a per-row index for fast viewport culling.

pub mod builder;
pub mod profile;
pub mod stacks;
pub mod strings;
pub mod trace_source;

pub use builder::ProfileBuilder;
pub use profile::{
    AttrTable, Category, CategoryId, Process, ProcessId, Profile, Sample, SliceTable, Thread,
    ThreadId, Track, TrackId, TrackKind,
};
pub use stacks::{Frame, FrameId, StackId, StackNode, StackTable};
pub use strings::{StringId, StringInterner};
pub use trace_source::{LoadError, LoadResult, TraceSource};
