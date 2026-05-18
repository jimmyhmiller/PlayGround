//! Body context and native body registry.
//!
//! A body is a Rust closure that mutates a [`BodyCtx`]. The context records
//! reads (for snapshotting in the log), accumulates writes and emits, and
//! enforces the declared footprint on every operation.
//!
//! v0.2 will replace native bodies with WASM components; the contract surface
//! exposed to the body will be the same.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexMap;
use thiserror::Error;

use ir::access::{AccessSegment, KeyBinding};
use ir::manifest::{Handler, Manifest};

use crate::log::{EmitRecord, WriteRecord};
use crate::state::StateStore;
use crate::value::Value;

#[derive(Debug, Error, Clone)]
pub enum BodyError {
    #[error("undeclared read of state cell `{0}`")]
    UndeclaredRead(String),
    #[error("undeclared read of map entry `{cell}[{key}]` (no matching declared path)")]
    UndeclaredMapKey { cell: String, key: String },
    #[error("undeclared write of state cell `{0}`")]
    UndeclaredWrite(String),
    #[error("undeclared emit of effect `{0}`")]
    UndeclaredEmit(String),
    #[error("body error: {0}")]
    Other(String),
}

pub type BodyFn = Box<dyn FnMut(&mut BodyCtx<'_>) -> Result<(), BodyError> + 'static>;

#[derive(Default)]
pub struct NativeBodyRegistry {
    bodies: HashMap<String, BodyFn>,
}

impl NativeBodyRegistry {
    pub fn register<F>(&mut self, uri: impl Into<String>, body: F)
    where
        F: FnMut(&mut BodyCtx<'_>) -> Result<(), BodyError> + 'static,
    {
        self.bodies.insert(uri.into(), Box::new(body));
    }

    pub fn get_mut(&mut self, uri: &str) -> Option<&mut BodyFn> {
        self.bodies.get_mut(uri)
    }

    pub fn contains(&self, uri: &str) -> bool {
        self.bodies.contains_key(uri)
    }

    /// Move the body out of the registry. Used by the runtime to invoke it
    /// without holding a mutable borrow on the registry while other fields
    /// are also borrowed. Must be paired with [`bodies_put`].
    pub(crate) fn bodies_take(&mut self, uri: &str) -> Option<BodyFn> {
        self.bodies.remove(uri)
    }

    pub(crate) fn bodies_put(&mut self, uri: String, body: BodyFn) {
        self.bodies.insert(uri, body);
    }
}

/// Shared emit-id allocator. Lives in the runtime; cloned into every
/// `BodyCtx` and into the WASM loader's host-side state so both sides
/// allocate from the same sequence — that's what makes the id returned by
/// `emit` synchronously to a WASM body match the id the runtime then uses
/// for the auto-routed response event.
pub type EmitIdCounter = Arc<AtomicU64>;

pub fn fresh_emit_id_counter() -> EmitIdCounter {
    Arc::new(AtomicU64::new(0))
}

/// What a body sees and produces during one invocation.
pub struct BodyCtx<'a> {
    pub event_name: &'a str,
    pub event: &'a Value,
    pub event_id: u64,
    pub arrival_ts: u64,

    handler: &'a Handler,
    _manifest: &'a Manifest,
    state: &'a StateStore,

    emit_id_counter: EmitIdCounter,

    pub(crate) writes: Vec<WriteRecord>,
    pub(crate) emits: Vec<EmitRecord>,
    pub(crate) reads: IndexMap<String, Value>,
}

impl<'a> BodyCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        event_name: &'a str,
        event: &'a Value,
        event_id: u64,
        arrival_ts: u64,
        handler: &'a Handler,
        manifest: &'a Manifest,
        state: &'a StateStore,
        emit_id_counter: EmitIdCounter,
    ) -> Self {
        Self {
            event_name,
            event,
            event_id,
            arrival_ts,
            handler,
            _manifest: manifest,
            state,
            emit_id_counter,
            writes: Vec::new(),
            emits: Vec::new(),
            reads: IndexMap::new(),
        }
    }

    /// Allocate a fresh emit id from the shared counter without recording an
    /// `EmitRecord`. WASM bodies use this to return the real id to the guest
    /// synchronously, then call [`record_emit`] on the replay path with the
    /// same id.
    pub fn allocate_emit_id(&self) -> u64 {
        self.emit_id_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Like [`emit`], but the caller supplies the id (presumably allocated
    /// via [`allocate_emit_id`] earlier in the same invocation). Used by the
    /// WASM loader to push the body's pre-allocated emits without double-
    /// counting against the shared counter.
    pub fn record_emit(
        &mut self,
        effect: &str,
        request: Value,
        emit_id: u64,
    ) -> Result<(), BodyError> {
        if !self.handler.emit.iter().any(|e| e == effect) {
            return Err(BodyError::UndeclaredEmit(effect.to_string()));
        }
        self.emits.push(EmitRecord {
            emit_id,
            effect: effect.to_string(),
            request,
        });
        Ok(())
    }

    /// Borrow the underlying shared counter. The WASM loader clones this to
    /// hand id allocation to its host functions.
    pub fn emit_id_counter(&self) -> EmitIdCounter {
        self.emit_id_counter.clone()
    }

    /// Deterministic id derived from this event and a body-supplied salt.
    /// Same (event_id, emit-count-so-far, salt) ⇒ same id, so replay is stable.
    pub fn derive_id(&self, salt: &str) -> u64 {
        let mut h = DefaultHasher::new();
        self.event_id.hash(&mut h);
        self.emits.len().hash(&mut h);
        salt.hash(&mut h);
        h.finish()
    }

    pub fn read_atom(&mut self, cell: &str) -> Result<Value, BodyError> {
        self.check_can_read(cell)?;
        let v = self.state.get_atom(cell).cloned().unwrap_or(Value::Null);
        self.reads.insert(cell.to_string(), v.clone());
        Ok(v)
    }

    pub fn read_map_entry(&mut self, cell: &str, key: &Value) -> Result<Option<Value>, BodyError> {
        self.check_map_key_in_footprint(cell, key)?;
        let v = self.state.get_map_entry(cell, key).cloned();
        let log_key = format!("{cell}[{}]", value_short(key));
        self.reads
            .insert(log_key, serde_json::json!({ "key": key, "value": v }));
        Ok(v)
    }

    pub fn list_map(&mut self, cell: &str) -> Result<Vec<(Value, Value)>, BodyError> {
        self.check_can_read_wildcard(cell)?;
        let all = self.state.list_map(cell);
        self.reads.insert(
            format!("{cell}[*]"),
            serde_json::to_value(&all).unwrap_or(Value::Null),
        );
        Ok(all)
    }

    pub fn set_atom(&mut self, cell: &str, value: Value) -> Result<(), BodyError> {
        self.check_can_write(cell)?;
        self.writes.push(WriteRecord::SetAtom {
            cell: cell.to_string(),
            value,
        });
        Ok(())
    }

    pub fn put_map(&mut self, cell: &str, key: Value, value: Value) -> Result<(), BodyError> {
        self.check_can_write(cell)?;
        self.writes.push(WriteRecord::PutMap {
            cell: cell.to_string(),
            key,
            value,
        });
        Ok(())
    }

    pub fn delete_map(&mut self, cell: &str, key: Value) -> Result<(), BodyError> {
        self.check_can_write(cell)?;
        self.writes.push(WriteRecord::DeleteMap {
            cell: cell.to_string(),
            key,
        });
        Ok(())
    }

    pub fn emit(&mut self, effect: &str, request: Value) -> Result<u64, BodyError> {
        let id = self.allocate_emit_id();
        self.record_emit(effect, request, id)?;
        Ok(id)
    }

    fn check_can_read(&self, cell: &str) -> Result<(), BodyError> {
        if self.handler.read.iter().any(|p| p.cell == cell) {
            Ok(())
        } else {
            Err(BodyError::UndeclaredRead(cell.to_string()))
        }
    }

    fn check_can_read_wildcard(&self, cell: &str) -> Result<(), BodyError> {
        let ok = self.handler.read.iter().any(|p| {
            p.cell == cell
                && matches!(
                    p.segments.first(),
                    Some(AccessSegment::Wildcard) | None
                )
        });
        if ok {
            Ok(())
        } else {
            Err(BodyError::UndeclaredRead(format!("{cell}[*]")))
        }
    }

    fn check_map_key_in_footprint(&self, cell: &str, key: &Value) -> Result<(), BodyError> {
        let matches = self.handler.read.iter().any(|p| {
            if p.cell != cell {
                return false;
            }
            match p.segments.first() {
                Some(AccessSegment::Wildcard) | None => true,
                Some(AccessSegment::Key(KeyBinding::Event(field_path))) => {
                    navigate_event(self.event, field_path).as_ref() == Some(key)
                }
                Some(AccessSegment::Field(_)) => false,
            }
        });
        if matches {
            Ok(())
        } else {
            Err(BodyError::UndeclaredMapKey {
                cell: cell.to_string(),
                key: value_short(key),
            })
        }
    }

    fn check_can_write(&self, cell: &str) -> Result<(), BodyError> {
        if self.handler.write.iter().any(|p| p.cell == cell) {
            Ok(())
        } else {
            Err(BodyError::UndeclaredWrite(cell.to_string()))
        }
    }
}

pub(crate) fn navigate_event(event: &Value, path: &[String]) -> Option<Value> {
    let mut cursor = event;
    for f in path {
        cursor = cursor.get(f)?;
    }
    Some(cursor.clone())
}

fn value_short(v: &Value) -> String {
    let s = serde_json::to_string(v).unwrap_or_else(|_| "?".into());
    if s.len() > 40 {
        format!("{}…", &s[..40])
    } else {
        s
    }
}
