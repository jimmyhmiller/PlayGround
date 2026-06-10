//! Serialization of VM state to disk (spec §6.2 "serializable for
//! save/restore" + §7 atom externalization).
//!
//! Shared mutable identities (Atoms, captured-`let mut` Cells) are encoded
//! through side tables so aliasing — and even cycles — survive a round trip.
//! `Native` host values cannot be serialized: capturing state that contains
//! one fails loudly (never silently drops, per project rule).
//!
//! Native *functions* are fine: they serialize as names and are re-linked on
//! restore, so the host must register the same natives (by name) before
//! calling `restore_state`. Missing natives fail loudly.

use crate::bytecode::FnProto;
use crate::value::{AtomCell, Closure, Value, Variant, VariantPayload};
use crate::vm::{Fault, Frame, Funct, FunctError, Status, VmState};
use serde::{Deserialize, Serialize};
use crate::value::shared::{Lock, Sh};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SValue {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<SValue>),
    Tuple(Vec<SValue>),
    Record(Vec<(String, SValue)>),
    VariantUnit(String),
    VariantPos(String, Vec<SValue>),
    VariantNamed(String, Vec<(String, SValue)>),
    Closure { fn_id: u32, upvals: Vec<SValue> },
    /// index into `native_names`
    NativeFn(u32),
    AtomRef(usize),
    CellRef(usize),
    Range(i64, i64, bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAtom {
    pub id: u64,
    pub value: SValue,
    pub watchers: Vec<(String, SValue)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SFrame {
    pub fn_id: u32,
    pub ip: u32,
    pub locals: Vec<SValue>,
    pub upvals: Vec<SValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SStatus {
    Running,
    Done(SValue),
    Faulted { msg: String, at: Option<String> },
}

/// A complete, self-contained execution snapshot: code + globals + atoms +
/// the paused VM state. Restorable in a fresh process (natives re-registered
/// by the host first).
#[derive(Serialize, Deserialize)]
pub struct StateSnapshot {
    pub version: u32,
    pub fns: Vec<FnProto>,
    pub fn_ids: Vec<(String, u32)>,
    pub fn_count: u32,
    pub global_names: Vec<String>,
    pub globals: Vec<Option<SValue>>,
    pub native_names: Vec<String>,
    pub atom_table: Vec<SAtom>,
    pub cell_table: Vec<SValue>,
    pub frames: Vec<SFrame>,
    pub stack: Vec<SValue>,
    pub status: SStatus,
    pub atom_counter: u64,
    /// global ids visible inside modules without an import
    #[serde(default)]
    pub shared_globals: Vec<u32>,
    /// loaded module registry (module code itself lives in fns/globals)
    #[serde(default)]
    pub modules: Vec<(String, SModuleExports)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SModuleExports {
    File(Vec<(String, u32)>),
    Host(u32),
}

/// Just the atoms (spec §7 `capture_atoms`/`restore_atoms`).
#[derive(Serialize, Deserialize)]
pub struct AtomSnapshot {
    pub version: u32,
    pub native_names: Vec<String>,
    pub atom_table: Vec<SAtom>,
    pub cell_table: Vec<SValue>,
    pub atom_counter: u64,
}

pub const SNAPSHOT_VERSION: u32 = 1;

// ---------- encoding ----------

#[derive(Default)]
struct Encoder {
    atom_ids: HashMap<usize, usize>,
    atoms: Vec<Option<SAtom>>,
    cell_ids: HashMap<usize, usize>,
    cells: Vec<Option<SValue>>,
}

impl Encoder {
    fn enc(&mut self, v: &Value) -> Result<SValue, Fault> {
        Ok(match v {
            Value::Unit => SValue::Unit,
            Value::Bool(b) => SValue::Bool(*b),
            Value::Int(i) => SValue::Int(*i),
            Value::Float(f) => SValue::Float(*f),
            Value::Str(s) => SValue::Str(s.to_string()),
            Value::List(items) => SValue::List(self.enc_all(items)?),
            Value::Tuple(items) => SValue::Tuple(self.enc_all(items)?),
            Value::Record(r) => SValue::Record(
                r.iter()
                    .map(|(k, v)| Ok((k.clone(), self.enc(v)?)))
                    .collect::<Result<_, Fault>>()?,
            ),
            Value::Variant(var) => match &var.payload {
                VariantPayload::Unit => SValue::VariantUnit(var.tag.clone()),
                VariantPayload::Positional(items) => {
                    SValue::VariantPos(var.tag.clone(), self.enc_all(items)?)
                }
                VariantPayload::Named(fields) => SValue::VariantNamed(
                    var.tag.clone(),
                    fields
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.enc(v)?)))
                        .collect::<Result<_, Fault>>()?,
                ),
            },
            Value::Closure(c) => SValue::Closure {
                fn_id: c.fn_id,
                upvals: self.enc_all(&c.upvals)?,
            },
            Value::NativeFn(id) => SValue::NativeFn(*id),
            Value::Atom(a) => {
                let key = Sh::as_ptr(a) as usize;
                if let Some(&i) = self.atom_ids.get(&key) {
                    SValue::AtomRef(i)
                } else {
                    let i = self.atoms.len();
                    self.atoms.push(None); // reserve (handles cycles)
                    self.atom_ids.insert(key, i);
                    let value = {
                        let inner = a.value.read().clone();
                        self.enc(&inner)?
                    };
                    let watchers = {
                        let ws = a.watchers.read().clone();
                        ws.iter()
                            .map(|(k, f)| Ok((k.clone(), self.enc(f)?)))
                            .collect::<Result<Vec<_>, Fault>>()?
                    };
                    self.atoms[i] = Some(SAtom { id: a.id, value, watchers });
                    SValue::AtomRef(i)
                }
            }
            Value::Cell(c) => {
                let key = Sh::as_ptr(c) as usize;
                if let Some(&i) = self.cell_ids.get(&key) {
                    SValue::CellRef(i)
                } else {
                    let i = self.cells.len();
                    self.cells.push(None);
                    self.cell_ids.insert(key, i);
                    let inner = c.read().clone();
                    let sv = self.enc(&inner)?;
                    self.cells[i] = Some(sv);
                    SValue::CellRef(i)
                }
            }
            Value::Native(n) => {
                return Err(Fault::new(format!(
                    "cannot serialize native host value of type `{}`; \
                     remove it from reachable state or keep it host-side",
                    n.type_name
                )))
            }
            Value::Range(a, b, inc) => SValue::Range(*a, *b, *inc),
        })
    }

    fn enc_all(&mut self, vs: &[Value]) -> Result<Vec<SValue>, Fault> {
        vs.iter().map(|v| self.enc(v)).collect()
    }

    fn finish(self) -> (Vec<SAtom>, Vec<SValue>) {
        (
            self.atoms.into_iter().map(|a| a.expect("atom slot filled")).collect(),
            self.cells.into_iter().map(|c| c.expect("cell slot filled")).collect(),
        )
    }
}

// ---------- decoding ----------

struct Decoder {
    atoms: Vec<Sh<AtomCell>>,
    cells: Vec<Sh<Lock<Value>>>,
    /// old native id -> new native id
    native_map: Vec<u32>,
}

impl Decoder {
    fn dec(&self, sv: &SValue) -> Result<Value, Fault> {
        Ok(match sv {
            SValue::Unit => Value::Unit,
            SValue::Bool(b) => Value::Bool(*b),
            SValue::Int(i) => Value::Int(*i),
            SValue::Float(f) => Value::Float(*f),
            SValue::Str(s) => Value::str(s.clone()),
            SValue::List(items) => Value::List(Sh::new(self.dec_all(items)?)),
            SValue::Tuple(items) => Value::Tuple(Sh::new(self.dec_all(items)?)),
            SValue::Record(fields) => {
                let mut map = BTreeMap::new();
                for (k, v) in fields {
                    map.insert(k.clone(), self.dec(v)?);
                }
                Value::Record(Sh::new(map))
            }
            SValue::VariantUnit(tag) => Value::variant(tag, VariantPayload::Unit),
            SValue::VariantPos(tag, items) => {
                Value::variant(tag, VariantPayload::Positional(self.dec_all(items)?))
            }
            SValue::VariantNamed(tag, fields) => {
                let mut map = BTreeMap::new();
                for (k, v) in fields {
                    map.insert(k.clone(), self.dec(v)?);
                }
                Value::variant(tag, VariantPayload::Named(map))
            }
            SValue::Closure { fn_id, upvals } => {
                Value::Closure(Sh::new(Closure { fn_id: *fn_id, upvals: self.dec_all(upvals)? }))
            }
            SValue::NativeFn(old_id) => {
                let new_id = self
                    .native_map
                    .get(*old_id as usize)
                    .copied()
                    .ok_or_else(|| Fault::new(format!("snapshot references unknown native id {}", old_id)))?;
                Value::NativeFn(new_id)
            }
            SValue::AtomRef(i) => Value::Atom(
                self.atoms
                    .get(*i)
                    .cloned()
                    .ok_or_else(|| Fault::new(format!("snapshot atom ref {} out of range", i)))?,
            ),
            SValue::CellRef(i) => Value::Cell(
                self.cells
                    .get(*i)
                    .cloned()
                    .ok_or_else(|| Fault::new(format!("snapshot cell ref {} out of range", i)))?,
            ),
            SValue::Range(a, b, inc) => Value::Range(*a, *b, *inc),
        })
    }

    fn dec_all(&self, svs: &[SValue]) -> Result<Vec<Value>, Fault> {
        svs.iter().map(|v| self.dec(v)).collect()
    }
}

/// Map saved native names to the current engine's native ids. Loud error if
/// a needed native was not re-registered.
fn build_native_map(vm: &Funct, native_names: &[String]) -> Result<Vec<u32>, Fault> {
    native_names
        .iter()
        .map(|name| {
            vm.native_ids.get(name).copied().ok_or_else(|| {
                Fault::new(format!(
                    "snapshot needs native fn `{}` which is not registered; \
                     register it (same name) before restoring",
                    name
                ))
            })
        })
        .collect()
}

fn dec_status(d: &Decoder, s: &SStatus) -> Result<Status, Fault> {
    Ok(match s {
        SStatus::Running => Status::Running,
        SStatus::Done(v) => Status::Done(d.dec(v)?),
        SStatus::Faulted { msg, at } => {
            Status::Faulted(Fault { msg: msg.clone(), at: at.clone() })
        }
    })
}

impl Funct {
    /// Serialize the engine (code, globals, atoms) plus a paused `VmState`
    /// to JSON. Fails loudly if any reachable value is a Native host object.
    pub fn save_state(&mut self, st: &VmState) -> Result<String, Fault> {
        let mut enc = Encoder::default();
        let mut globals = Vec::with_capacity(self.globals.len());
        let globals_src = self.globals.clone();
        for g in &globals_src {
            globals.push(match g {
                Some(v) => Some(enc.enc(v)?),
                None => None,
            });
        }
        let mut frames = Vec::with_capacity(st.frames.len());
        for f in &st.frames {
            frames.push(SFrame {
                fn_id: f.fn_id,
                ip: f.ip,
                locals: enc.enc_all(&f.locals)?,
                upvals: enc.enc_all(&f.upvals)?,
            });
        }
        let stack = enc.enc_all(&st.stack)?;
        let status = match &st.status {
            Status::Running => SStatus::Running,
            Status::Done(v) => SStatus::Done(enc.enc(v)?),
            Status::Faulted(f) => SStatus::Faulted { msg: f.msg.clone(), at: f.at.clone() },
        };
        // include host-held registry atoms not reachable from the state
        for a in self.live_atoms() {
            enc.enc(&Value::Atom(a))?;
        }
        let (atom_table, cell_table) = enc.finish();
        let snap = StateSnapshot {
            version: SNAPSHOT_VERSION,
            fns: self.fns.iter().map(|p| (**p).clone()).collect(),
            fn_ids: self.ctx.fn_ids.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            fn_count: self.ctx.fn_count,
            global_names: self.ctx.global_names.clone(),
            globals,
            native_names: self.natives.iter().map(|n| n.name.clone()).collect(),
            atom_table,
            cell_table,
            frames,
            stack,
            status,
            atom_counter: self.atom_counter,
            shared_globals: self.ctx.shared.iter().copied().collect(),
            modules: self
                .modules
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        match v {
                            crate::vm::ModuleExports::File(list) => SModuleExports::File(list.clone()),
                            crate::vm::ModuleExports::Host(g) => SModuleExports::Host(*g),
                        },
                    )
                })
                .collect(),
        };
        serde_json::to_string(&snap).map_err(|e| Fault::new(format!("snapshot encode failed: {}", e)))
    }

    /// Restore a snapshot produced by `save_state`. Replaces this engine's
    /// code table, globals and atoms. All natives the snapshot references
    /// must already be registered (same names) — e.g. by using `Funct::new()`
    /// plus the same `register*` calls as the saving process.
    pub fn restore_state(&mut self, json: &str) -> Result<VmState, FunctError> {
        let snap: StateSnapshot = serde_json::from_str(json)
            .map_err(|e| FunctError::Fault(Fault::new(format!("snapshot decode failed: {}", e))))?;
        if snap.version != SNAPSHOT_VERSION {
            return Err(FunctError::Fault(Fault::new(format!(
                "snapshot version {} not supported (expected {})",
                snap.version, SNAPSHOT_VERSION
            ))));
        }
        let native_map = build_native_map(self, &snap.native_names).map_err(FunctError::Fault)?;

        // install code table + name maps
        self.fns = snap.fns.into_iter().map(Sh::new).collect();
        self.ctx.fn_ids = snap.fn_ids.into_iter().collect();
        self.ctx.fn_count = snap.fn_count;
        self.ctx.global_names = snap.global_names;
        self.ctx.global_ids = self
            .ctx
            .global_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as u32))
            .collect();
        self.ctx.shared = snap.shared_globals.iter().copied().collect();
        self.modules = snap
            .modules
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    match v {
                        SModuleExports::File(list) => crate::vm::ModuleExports::File(list.clone()),
                        SModuleExports::Host(g) => crate::vm::ModuleExports::Host(*g),
                    },
                )
            })
            .collect();

        // shell atoms/cells first (so references — even cyclic — resolve)
        let atoms: Vec<Sh<AtomCell>> = snap
            .atom_table
            .iter()
            .map(|sa| {
                Sh::new(AtomCell {
                    id: sa.id,
                    value: Lock::new(Value::Unit),
                    watchers: Lock::new(Vec::new()),
                })
            })
            .collect();
        let cells: Vec<Sh<Lock<Value>>> =
            snap.cell_table.iter().map(|_| Sh::new(Lock::new(Value::Unit))).collect();
        let dec = Decoder { atoms, cells, native_map };

        for (i, sa) in snap.atom_table.iter().enumerate() {
            *dec.atoms[i].value.write() = dec.dec(&sa.value).map_err(FunctError::Fault)?;
            let mut ws = Vec::new();
            for (k, f) in &sa.watchers {
                ws.push((k.clone(), dec.dec(f).map_err(FunctError::Fault)?));
            }
            *dec.atoms[i].watchers.write() = ws;
        }
        for (i, sc) in snap.cell_table.iter().enumerate() {
            *dec.cells[i].write() = dec.dec(sc).map_err(FunctError::Fault)?;
        }

        self.globals = snap
            .globals
            .iter()
            .map(|g| g.as_ref().map(|v| dec.dec(v)).transpose())
            .collect::<Result<_, Fault>>()
            .map_err(FunctError::Fault)?;
        self.sync_globals();

        let mut frames = Vec::with_capacity(snap.frames.len());
        for sf in &snap.frames {
            let proto = self
                .fns
                .get(sf.fn_id as usize)
                .cloned()
                .ok_or_else(|| FunctError::Fault(Fault::new(format!("snapshot frame references unknown fn id {}", sf.fn_id))))?;
            frames.push(Frame {
                fn_id: sf.fn_id,
                proto,
                ip: sf.ip,
                locals: dec.dec_all(&sf.locals).map_err(FunctError::Fault)?,
                upvals: dec.dec_all(&sf.upvals).map_err(FunctError::Fault)?,
            });
        }
        let stack = dec.dec_all(&snap.stack).map_err(FunctError::Fault)?;
        let status = dec_status(&dec, &snap.status).map_err(FunctError::Fault)?;

        self.atom_counter = snap.atom_counter;
        self.atoms = dec.atoms.iter().map(Sh::downgrade).collect();

        Ok(VmState { frames, stack, status })
    }

    /// Serialize every live atom (spec §7). The program's complete mutable
    /// root set.
    pub fn capture_atoms(&mut self) -> Result<String, Fault> {
        let mut enc = Encoder::default();
        for a in self.live_atoms() {
            enc.enc(&Value::Atom(a))?;
        }
        let (atom_table, cell_table) = enc.finish();
        let snap = AtomSnapshot {
            version: SNAPSHOT_VERSION,
            native_names: self.natives.iter().map(|n| n.name.clone()).collect(),
            atom_table,
            cell_table,
            atom_counter: self.atom_counter,
        };
        serde_json::to_string(&snap).map_err(|e| Fault::new(format!("atom snapshot encode failed: {}", e)))
    }

    /// Rehydrate atom values into the *current* program's live atoms,
    /// matched by stable AtomId. Typical flow: restart, re-eval the program
    /// (which re-creates its atoms deterministically), then restore. Fails
    /// loudly on id mismatches.
    pub fn restore_atoms(&mut self, json: &str) -> Result<(), Fault> {
        let snap: AtomSnapshot = serde_json::from_str(json)
            .map_err(|e| Fault::new(format!("atom snapshot decode failed: {}", e)))?;
        if snap.version != SNAPSHOT_VERSION {
            return Err(Fault::new(format!(
                "atom snapshot version {} not supported (expected {})",
                snap.version, SNAPSHOT_VERSION
            )));
        }
        let native_map = build_native_map(self, &snap.native_names)?;
        let live = self.live_atoms();
        let by_id: HashMap<u64, Sh<AtomCell>> = live.iter().map(|a| (a.id, a.clone())).collect();

        // shells: prefer live atoms with matching id; otherwise error loudly
        let mut atoms: Vec<Sh<AtomCell>> = Vec::with_capacity(snap.atom_table.len());
        for sa in &snap.atom_table {
            match by_id.get(&sa.id) {
                Some(a) => atoms.push(a.clone()),
                None => {
                    return Err(Fault::new(format!(
                        "restore_atoms: no live atom with id {} (program shape changed? \
                         re-eval the program before restoring)",
                        sa.id
                    )))
                }
            }
        }
        let cells: Vec<Sh<Lock<Value>>> =
            snap.cell_table.iter().map(|_| Sh::new(Lock::new(Value::Unit))).collect();
        let dec = Decoder { atoms, cells, native_map };
        for (i, sc) in snap.cell_table.iter().enumerate() {
            *dec.cells[i].write() = dec.dec(sc)?;
        }
        for (i, sa) in snap.atom_table.iter().enumerate() {
            *dec.atoms[i].value.write() = dec.dec(&sa.value)?;
            let mut ws = Vec::new();
            for (k, f) in &sa.watchers {
                ws.push((k.clone(), dec.dec(f)?));
            }
            *dec.atoms[i].watchers.write() = ws;
        }
        if snap.atom_counter > self.atom_counter {
            self.atom_counter = snap.atom_counter;
        }
        Ok(())
    }
}

// Keep Variant import used (encoder builds them via Value::variant).
#[allow(unused_imports)]
use Variant as _VariantImportKeeper;
