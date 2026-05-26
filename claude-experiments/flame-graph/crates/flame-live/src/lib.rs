//! Consume samply's live profile stream and snapshot it into a flame-core
//! `Profile`.
//!
//! Wire spec lives in `samply/docs/live-streaming.md`. We mirror the
//! `LiveEvent` schema verbatim so postcard / serde_json decode the same bytes
//! samply writes.
//!
//! Usage shape:
//!
//! 1. `LiveAggregator::default()` — owns all accumulated state.
//! 2. A reader thread pumps `LiveEvent`s into it via `apply`.
//! 3. The render thread periodically calls `snapshot()` to get a
//!    fully-finalized `flame_core::Profile`. The snapshot is built from
//!    scratch each time but reuses pre-computed sample aggregation so the
//!    cost is O(stack-tree + aggregated nodes), not O(samples).

pub mod event;
pub mod reader;
pub mod aggregator;
pub mod symbols;

pub use aggregator::LiveAggregator;
pub use event::{LiveEvent, LiveFrame, LiveFrameKind};
pub use reader::{read_binary_stream, read_ndjson_stream, ReadError};
pub use symbols::SymbolStore;
