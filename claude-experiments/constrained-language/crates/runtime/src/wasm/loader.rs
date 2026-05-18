//! Generic WASM-component handler loader.
//!
//! Given a manifest and a handler name (which references a `.wasm` component
//! by path/URI), produce a `BodyFn` the runtime can register. The body, when
//! invoked, will:
//!
//! 1. Pre-snapshot every state slice declared in the handler's `read`
//!    footprint, populating an in-memory cache the host functions serve from.
//!    The runtime's semantics already say writes apply *after* the body
//!    returns, so this snapshot is observationally equivalent to a live read.
//! 2. Build a wasmtime `Linker` containing exactly the imports declared by
//!    the footprint, with names matching what `ir::wit::generate_world` would
//!    produce (`get-<cell>`, `list-<cell>`, `set-<cell>`, `put-<cell>`,
//!    `delete-<cell>`, `emit-<fx>`).
//! 3. Convert the inbound event payload to a `Val` matching the event
//!    schema, instantiate the component, and call its `handle` export.
//! 4. Drain the accumulated writes and emits back through `BodyCtx`, which
//!    applies the runtime's footprint check a second time (defense in depth)
//!    and produces the records the scheduler logs and applies.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;

use indexmap::IndexMap;
use serde_json::Value as Json;
use thiserror::Error;
use wasmtime::component::{Component, Linker, Val};
use wasmtime::{Engine, Store};

use ir::access::{AccessSegment, KeyBinding};
use ir::manifest::{Handler, Manifest, StateDecl};
use ir::schema::SchemaRef;
use ir::wit::kebab;

use crate::body::{BodyCtx, BodyError, BodyFn, EmitIdCounter};

use super::values::{json_to_val, val_to_json, ConvertError};

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("handler `{0}` not found in manifest")]
    UnknownHandler(String),
    #[error("event `{0}` referenced by handler not found in manifest")]
    UnknownEvent(String),
    #[error("wasmtime: {0}")]
    Wasmtime(String),
}

/// Load a component file and return a `BodyFn` for the named handler.
pub fn load_handler_body(
    manifest: &Manifest,
    handler_name: &str,
    component_path: impl AsRef<Path>,
) -> Result<BodyFn, LoadError> {
    let handler = manifest
        .handlers
        .iter()
        .find(|h| h.name == handler_name)
        .ok_or_else(|| LoadError::UnknownHandler(handler_name.to_string()))?
        .clone();

    let event_schema = manifest
        .events
        .get(&handler.on)
        .map(|e| e.payload.clone())
        .ok_or_else(|| LoadError::UnknownEvent(handler.on.clone()))?;

    let engine = Engine::default();
    let component = Component::from_file(&engine, component_path.as_ref())
        .map_err(|e| LoadError::Wasmtime(e.to_string()))?;

    let manifest_cloned = manifest.clone();

    Ok(Box::new(move |ctx: &mut BodyCtx<'_>| -> Result<(), BodyError> {
        invoke(&engine, &component, &handler, &event_schema, &manifest_cloned, ctx)
    }))
}

/// Per-invocation host-side state. The wasmtime `Store` carries this; host
/// functions read snapshots and accumulate outputs here.
#[derive(Default)]
struct HostCtx {
    /// Atom snapshots, keyed by cell name.
    atoms: IndexMap<String, Json>,
    /// Map snapshots, keyed by cell name. The inner vec is the full
    /// `[(key, value)]` list — we serve both `get-cell(key)` and
    /// `list-cell()` from it.
    maps: IndexMap<String, Vec<(Json, Json)>>,

    /// Accumulated writes to replay through BodyCtx after the body returns.
    writes: Vec<PendingWrite>,
    /// Accumulated emits, each with the real emit id already allocated from
    /// the shared counter so the id returned to the WASM body matches the
    /// id the runtime then uses for the auto-routed response event.
    emits: Vec<PendingEmit>,
    /// Shared counter; cloned from BodyCtx at invoke time. Host functions
    /// fetch_add against this to allocate real emit ids.
    emit_id_counter: Option<EmitIdCounter>,
}

enum PendingWrite {
    SetAtom { cell: String, value: Json },
    PutMap { cell: String, key: Json, value: Json },
    DeleteMap { cell: String, key: Json },
}

struct PendingEmit {
    emit_id: u64,
    effect: String,
    request: Json,
}

fn invoke(
    engine: &Engine,
    component: &Component,
    handler: &Handler,
    event_schema: &SchemaRef,
    manifest: &Manifest,
    ctx: &mut BodyCtx<'_>,
) -> Result<(), BodyError> {
    // ---- 1. Pre-snapshot every declared read slice through BodyCtx so the
    // footprint check fires up front (an undeclared read here means a bug
    // in the manifest/codegen, not the body). ----
    let mut host = HostCtx {
        atoms: IndexMap::new(),
        maps: IndexMap::new(),
        writes: Vec::new(),
        emits: Vec::new(),
        emit_id_counter: Some(ctx.emit_id_counter()),
    };
    let event_json = ctx.event.clone();
    snapshot_reads(ctx, manifest, handler, &event_json, &mut host)?;

    // ---- 2. Wire up the linker. ----
    let mut linker: Linker<Arc<Mutex<HostCtx>>> = Linker::new(engine);
    register_read_imports(&mut linker, handler, manifest)?;
    register_write_imports(&mut linker, handler, manifest)?;
    register_emit_imports(&mut linker, handler, manifest)?;

    // ---- 3. Convert event payload, instantiate, call handle. ----
    let event_val = json_to_val(&event_json, event_schema, manifest)
        .map_err(|e| BodyError::Other(format!("event payload: {e}")))?;

    let shared = Arc::new(Mutex::new(host));
    let mut store = Store::new(engine, shared.clone());
    let instance = linker
        .instantiate(&mut store, component)
        .map_err(|e| BodyError::Other(format!("instantiate: {e}")))?;
    let handle = instance
        .get_func(&mut store, "handle")
        .ok_or_else(|| BodyError::Other("component does not export `handle`".into()))?;

    handle
        .call(&mut store, &[event_val], &mut [])
        .map_err(|e| BodyError::Other(format!("call handle: {e}")))?;

    // ---- 4. Drain outputs and replay through BodyCtx. ----
    // wasmtime's Store retains a clone of the data, so Arc::try_unwrap
    // won't work; pull the contents out via mem::take instead.
    let host: HostCtx = {
        let mut guard = shared.lock().map_err(|e| {
            BodyError::Other(format!("mutex poisoned: {e}"))
        })?;
        std::mem::take(&mut *guard)
    };
    drop(store);
    drop(shared);

    for w in host.writes {
        match w {
            PendingWrite::SetAtom { cell, value } => ctx.set_atom(&cell, value)?,
            PendingWrite::PutMap { cell, key, value } => ctx.put_map(&cell, key, value)?,
            PendingWrite::DeleteMap { cell, key } => ctx.delete_map(&cell, key)?,
        }
    }
    for e in host.emits {
        ctx.record_emit(&e.effect, e.request, e.emit_id)?;
    }

    Ok(())
}

fn snapshot_reads(
    ctx: &mut BodyCtx<'_>,
    manifest: &Manifest,
    handler: &Handler,
    event: &Json,
    host: &mut HostCtx,
) -> Result<(), BodyError> {
    for path in &handler.read {
        let Some(decl) = manifest.state.get(&path.cell) else {
            // Validator should have caught this; treat as undeclared read.
            return Err(BodyError::UndeclaredRead(path.cell.clone()));
        };
        match decl {
            StateDecl::Atom { .. } => {
                if host.atoms.contains_key(&path.cell) {
                    continue;
                }
                let v = ctx.read_atom(&path.cell)?;
                host.atoms.insert(path.cell.clone(), v);
            }
            StateDecl::Map { .. } => {
                match path.segments.first() {
                    Some(AccessSegment::Wildcard) | None => {
                        if host.maps.contains_key(&path.cell) {
                            continue;
                        }
                        let entries = ctx.list_map(&path.cell)?;
                        host.maps.insert(path.cell.clone(), entries);
                    }
                    Some(AccessSegment::Key(KeyBinding::Event(field_path))) => {
                        // We have to materialize the bound key from the event.
                        let key = match navigate_event(event, field_path) {
                            Some(k) => k,
                            None => continue, // event missing field → treat as no read
                        };
                        let cur = ctx.read_map_entry(&path.cell, &key)?;
                        let bucket = host.maps.entry(path.cell.clone()).or_default();
                        // Replace any prior pair with this key, else push.
                        if let Some(slot) = bucket.iter_mut().find(|(k, _)| k == &key) {
                            slot.1 = cur.unwrap_or(Json::Null);
                        } else if let Some(v) = cur {
                            bucket.push((key, v));
                        }
                    }
                    Some(AccessSegment::Field(_)) => {
                        // Validator should have caught this.
                        return Err(BodyError::UndeclaredRead(path.cell.clone()));
                    }
                }
            }
        }
    }
    Ok(())
}

fn navigate_event(event: &Json, path: &[String]) -> Option<Json> {
    let mut cursor = event;
    for f in path {
        cursor = cursor.get(f)?;
    }
    Some(cursor.clone())
}

// ----------------------------------------------------------------------------
// Read imports: get-<cell>, list-<cell>
// ----------------------------------------------------------------------------

fn register_read_imports(
    linker: &mut Linker<Arc<Mutex<HostCtx>>>,
    handler: &Handler,
    manifest: &Manifest,
) -> Result<(), BodyError> {
    let mut seen: std::collections::HashSet<(String, ReadKind)> =
        std::collections::HashSet::new();
    for path in &handler.read {
        let Some(decl) = manifest.state.get(&path.cell) else { continue };
        match decl {
            StateDecl::Atom { schema } => {
                let kind = (path.cell.clone(), ReadKind::AtomGet);
                if !seen.insert(kind) {
                    continue;
                }
                let fn_name = format!("get-{}", kebab(&path.cell));
                let cell = path.cell.clone();
                let schema = schema.clone();
                let manifest_c = manifest.clone();
                linker
                    .root()
                    .func_new(
                        &fn_name,
                        move |store: wasmtime::StoreContextMut<'_, Arc<Mutex<HostCtx>>>,
                              _func,
                              _params: &[Val],
                              results: &mut [Val]|
                              -> wasmtime::Result<()> {
                            let host = store.data().lock().unwrap();
                            let v = host.atoms.get(&cell).cloned().unwrap_or(Json::Null);
                            let val = json_to_val(&v, &schema, &manifest_c)
                                .map_err(into_anyhow)?;
                            results[0] = val;
                            Ok(())
                        },
                    )
                    .map_err(|e| BodyError::Other(format!("link {fn_name}: {e}")))?;
            }
            StateDecl::Map { key, value } => match path.segments.first() {
                Some(AccessSegment::Key(_)) => {
                    let kind = (path.cell.clone(), ReadKind::MapGet);
                    if !seen.insert(kind) {
                        continue;
                    }
                    let fn_name = format!("get-{}", kebab(&path.cell));
                    let cell = path.cell.clone();
                    let key_schema = key.clone();
                    let val_schema = value.clone();
                    let manifest_c = manifest.clone();
                    linker
                        .root()
                        .func_new(
                            &fn_name,
                            move |store, _func, params: &[Val], results: &mut [Val]| -> wasmtime::Result<()> {
                                let key_val = &params[0];
                                let key_json = val_to_json(key_val, &key_schema, &manifest_c)
                                    .map_err(into_anyhow)?;
                                let host = store.data().lock().unwrap();
                                let entries = host.maps.get(&cell);
                                let found = entries
                                    .and_then(|e| e.iter().find(|(k, _)| k == &key_json))
                                    .map(|(_, v)| v.clone());
                                let inner = match found {
                                    Some(v) => Some(Box::new(
                                        json_to_val(&v, &val_schema, &manifest_c)
                                            .map_err(into_anyhow)?,
                                    )),
                                    None => None,
                                };
                                results[0] = Val::Option(inner);
                                Ok(())
                            },
                        )
                        .map_err(|e| BodyError::Other(format!("link {fn_name}: {e}")))?;
                }
                Some(AccessSegment::Wildcard) | None => {
                    let kind = (path.cell.clone(), ReadKind::MapList);
                    if !seen.insert(kind) {
                        continue;
                    }
                    let fn_name = format!("list-{}", kebab(&path.cell));
                    let cell = path.cell.clone();
                    let key_schema = key.clone();
                    let val_schema = value.clone();
                    let manifest_c = manifest.clone();
                    linker
                        .root()
                        .func_new(
                            &fn_name,
                            move |store, _func, _params: &[Val], results: &mut [Val]| -> wasmtime::Result<()> {
                                let host = store.data().lock().unwrap();
                                let entries = host
                                    .maps
                                    .get(&cell)
                                    .cloned()
                                    .unwrap_or_default();
                                let mut items = Vec::with_capacity(entries.len());
                                for (k, v) in entries {
                                    let kv = json_to_val(&k, &key_schema, &manifest_c)
                                        .map_err(into_anyhow)?;
                                    let vv = json_to_val(&v, &val_schema, &manifest_c)
                                        .map_err(into_anyhow)?;
                                    items.push(Val::Tuple(vec![kv, vv]));
                                }
                                results[0] = Val::List(items);
                                Ok(())
                            },
                        )
                        .map_err(|e| BodyError::Other(format!("link {fn_name}: {e}")))?;
                }
                Some(AccessSegment::Field(_)) => {
                    // Validator catches this; don't link a function.
                }
            },
        }
    }
    Ok(())
}

#[derive(Hash, PartialEq, Eq)]
enum ReadKind {
    AtomGet,
    MapGet,
    MapList,
}

// ----------------------------------------------------------------------------
// Write imports: set-<cell>, put-<cell>, delete-<cell>
// ----------------------------------------------------------------------------

fn register_write_imports(
    linker: &mut Linker<Arc<Mutex<HostCtx>>>,
    handler: &Handler,
    manifest: &Manifest,
) -> Result<(), BodyError> {
    let mut seen: std::collections::HashSet<(String, WriteKind)> =
        std::collections::HashSet::new();
    for path in &handler.write {
        let Some(decl) = manifest.state.get(&path.cell) else { continue };
        match decl {
            StateDecl::Atom { schema } => {
                let kind = (path.cell.clone(), WriteKind::AtomSet);
                if !seen.insert(kind) {
                    continue;
                }
                let fn_name = format!("set-{}", kebab(&path.cell));
                let cell = path.cell.clone();
                let schema = schema.clone();
                let manifest_c = manifest.clone();
                linker
                    .root()
                    .func_new(
                        &fn_name,
                        move |store, _func, params: &[Val], _results: &mut [Val]| -> wasmtime::Result<()> {
                            let v = val_to_json(&params[0], &schema, &manifest_c)
                                .map_err(into_anyhow)?;
                            let mut host = store.data().lock().unwrap();
                            host.writes.push(PendingWrite::SetAtom {
                                cell: cell.clone(),
                                value: v,
                            });
                            Ok(())
                        },
                    )
                    .map_err(|e| BodyError::Other(format!("link {fn_name}: {e}")))?;
            }
            StateDecl::Map { key, value } => {
                // put-<cell>
                let put_kind = (path.cell.clone(), WriteKind::MapPut);
                if seen.insert(put_kind) {
                    let put_name = format!("put-{}", kebab(&path.cell));
                    let cell = path.cell.clone();
                    let key_schema = key.clone();
                    let val_schema = value.clone();
                    let manifest_c = manifest.clone();
                    linker
                        .root()
                        .func_new(
                            &put_name,
                            move |store, _func, params: &[Val], _results: &mut [Val]| -> wasmtime::Result<()> {
                                let k = val_to_json(&params[0], &key_schema, &manifest_c)
                                    .map_err(into_anyhow)?;
                                let v = val_to_json(&params[1], &val_schema, &manifest_c)
                                    .map_err(into_anyhow)?;
                                let mut host = store.data().lock().unwrap();
                                host.writes.push(PendingWrite::PutMap {
                                    cell: cell.clone(),
                                    key: k,
                                    value: v,
                                });
                                Ok(())
                            },
                        )
                        .map_err(|e| BodyError::Other(format!("link {put_name}: {e}")))?;
                }

                // delete-<cell>
                let del_kind = (path.cell.clone(), WriteKind::MapDelete);
                if seen.insert(del_kind) {
                    let del_name = format!("delete-{}", kebab(&path.cell));
                    let cell = path.cell.clone();
                    let key_schema = key.clone();
                    let manifest_c = manifest.clone();
                    linker
                        .root()
                        .func_new(
                            &del_name,
                            move |store, _func, params: &[Val], _results: &mut [Val]| -> wasmtime::Result<()> {
                                let k = val_to_json(&params[0], &key_schema, &manifest_c)
                                    .map_err(into_anyhow)?;
                                let mut host = store.data().lock().unwrap();
                                host.writes.push(PendingWrite::DeleteMap {
                                    cell: cell.clone(),
                                    key: k,
                                });
                                Ok(())
                            },
                        )
                        .map_err(|e| BodyError::Other(format!("link {del_name}: {e}")))?;
                }
            }
        }
    }
    Ok(())
}

#[derive(Hash, PartialEq, Eq)]
enum WriteKind {
    AtomSet,
    MapPut,
    MapDelete,
}

// ----------------------------------------------------------------------------
// Emit imports: emit-<effect>
// ----------------------------------------------------------------------------

fn register_emit_imports(
    linker: &mut Linker<Arc<Mutex<HostCtx>>>,
    handler: &Handler,
    manifest: &Manifest,
) -> Result<(), BodyError> {
    for effect_name in &handler.emit {
        let Some(def) = manifest.effects.get(effect_name) else { continue };
        let fn_name = format!("emit-{}", kebab(effect_name));
        let effect = effect_name.clone();
        let req_schema = def.request.clone();
        let manifest_c = manifest.clone();
        linker
            .root()
            .func_new(
                &fn_name,
                move |store, _func, params: &[Val], results: &mut [Val]| -> wasmtime::Result<()> {
                    let req_json = val_to_json(&params[0], &req_schema, &manifest_c)
                        .map_err(into_anyhow)?;
                    let mut host = store.data().lock().unwrap();
                    let id = host
                        .emit_id_counter
                        .as_ref()
                        .expect("emit_id_counter set at invoke time")
                        .fetch_add(1, Ordering::SeqCst);
                    host.emits.push(PendingEmit {
                        emit_id: id,
                        effect: effect.clone(),
                        request: req_json,
                    });
                    results[0] = Val::U64(id);
                    Ok(())
                },
            )
            .map_err(|e| BodyError::Other(format!("link {fn_name}: {e}")))?;
    }
    Ok(())
}

fn into_anyhow(e: ConvertError) -> wasmtime::Error {
    wasmtime::Error::msg(e.to_string())
}
