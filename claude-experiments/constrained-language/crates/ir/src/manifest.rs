use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::access::AccessPath;
use crate::schema::{SchemaDef, SchemaRef};

/// The full program manifest. This is the single artifact the runtime,
/// validator, and WIT generator all consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: String,

    #[serde(default)]
    pub schemas: IndexMap<String, SchemaDef>,

    #[serde(default)]
    pub events: IndexMap<String, EventDef>,

    #[serde(default)]
    pub state: IndexMap<String, StateDecl>,

    #[serde(default)]
    pub effects: IndexMap<String, EffectDef>,

    #[serde(default)]
    pub handlers: Vec<Handler>,

    /// Typed sources of events. Each generator is a WASM component that
    /// produces values of an event's payload type; the runtime wraps those
    /// values into `InboundEvent`s and pushes them onto the queue.
    ///
    /// Two component shapes are accepted; the loader detects which from the
    /// component's WIT:
    ///   - **pull** — exports `call: func() -> option<payload>`. Runtime
    ///     calls it on a dedicated worker thread in a loop; stops on `none`.
    ///   - **push** — imports `emit: func(v: payload)`, exports
    ///     `start: func()`. Runtime starts it once on its own thread; the
    ///     component calls `emit` whenever (HTTP callbacks, timers, etc.).
    #[serde(default)]
    pub generators: IndexMap<String, GeneratorDecl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventDef {
    pub payload: SchemaRef,
}

/// A named piece of program state.
///
/// `Atom<T>` is a single typed slot — one slice for the scheduler.
/// `Map<K, V>` is keyed — one slice per key.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StateDecl {
    Atom { schema: SchemaRef },
    Map { key: SchemaRef, value: SchemaRef },
}

/// An effect type. `response` is the success variant only; the universal
/// `Failed { reason }` variant is supplied by the runtime and not declared
/// in the manifest.
///
/// If `response_event` is set, the runtime automatically enqueues an event of
/// that kind after the effect is fulfilled. The synthesized payload has the
/// shape `{ emit_id: u64, outcome: { tag: "ok" | "failed", ... } }`; declared
/// events with this name should match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectDef {
    pub request: SchemaRef,
    pub response: SchemaRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_event: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handler {
    pub name: String,
    /// Name of the event this handler responds to.
    pub on: String,
    #[serde(default)]
    pub read: Vec<AccessPath>,
    #[serde(default)]
    pub write: Vec<AccessPath>,
    /// Names of effect types this handler may emit.
    #[serde(default)]
    pub emit: Vec<String>,
    pub body: ComponentRef,
}

/// Content-addressed pointer to a WASM component blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentRef {
    /// `sha256:<64 hex>` of the component bytes.
    pub hash: String,
    /// Filesystem path, URL, or registry reference. Opaque to the IR.
    pub uri: String,
}

/// A typed source of events. v0.1 wires the generator role to a Rust trait
/// impl via a `kind` string, matching how adapters are wired today; both
/// will move to WIT-component implementations in a later pass without
/// changing the manifest's conceptual shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorDecl {
    /// Name of the event this generator's outputs become. Multiple
    /// generators may target the same event.
    pub event: String,
    /// Name of the generator implementation kind. cl-run resolves this to
    /// a built-in (or eventually a WIT-component binding).
    pub kind: String,
    /// Payload template. String values may contain `$input` placeholders
    /// that the generator substitutes per produced item. Non-string values
    /// pass through unchanged.
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}
