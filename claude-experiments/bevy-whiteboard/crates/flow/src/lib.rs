//! Flow — a small substrate for simulating distributed-system-like dynamics.
//!
//! The formalism: typed slots hold scalar state; rules pattern-match on
//! inbound packets + slot values, guarded by expressions, and fire
//! atomically. Time lives on edges as latency expressions, not inside
//! rules. Stochasticity comes from a seeded RNG exposed via
//! distribution-sampling expressions. The event log is the source of
//! truth for scrubbing, plotting, and (eventually) rewind.

pub mod value;
pub mod expr;
pub mod samples;
pub mod rule;
pub mod sim;
pub mod engine;
pub mod event;
pub mod template;
pub mod scenario;
pub mod history;
pub mod render;
pub mod repl;
pub mod dsl;

pub use value::{Pattern, Value, Bindings, match_pattern};
pub use expr::{Expr, BinOp, EvalCtx};
pub use samples::Samples;
pub use rule::{Rule, When, Effect, EmitTo, MetaOp, ReturnPathOp};
pub use sim::{Sim, Node, Edge, NodeId, EdgeId, Packet, PacketId, Time};
pub use event::{Event, EventLog};
pub use template::{Template, EdgeSpec, EdgeEnd};
pub use scenario::{Scenario, Action};
pub use history::{Snapshot, SnapshotRing};
