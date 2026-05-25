//! v0.1 reference runtime for the constrained-language model.
//!
//! Single-threaded scheduler, in-memory state, append-only event log,
//! native (Rust closure) bodies registered by URI. WASM components and
//! parallelism are deferred to later versions.

pub mod body;
pub mod effect;
pub mod generator;
pub mod log;
pub mod replay;
pub mod runtime;
pub mod scheduler;
pub mod state;
pub mod value;
pub mod wasm;

pub use body::{BodyCtx, BodyError, BodyFn, NativeBodyRegistry};
pub use effect::{Adapter, AdapterRegistry, AdapterResult, MockAdapter, ScriptedAdapter};
pub use generator::{Generator, GeneratorRegistry};
pub use log::{EffectOutcome, EmitRecord, EventLog, LogEntry, LogEntryKind, LogicalClock, WriteRecord};
pub use replay::{apply_writes_from_log, ReplayAdapter};
pub use runtime::{Runtime, RuntimeError};
pub use scheduler::{EventQueue, InboundEvent};
pub use state::StateStore;
pub use value::Value;
