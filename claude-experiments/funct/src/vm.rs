//! The engine + reified VM.
//!
//! Script frames never use the host call stack: `VmState` is plain owned data
//! and `step()` executes exactly one instruction, so execution can pause
//! between any two instructions, be snapshotted (Clone), serialized, and
//! resumed. Native (Rust) calls are the one atomic unit: a native call —
//! including any script closures it invokes reentrantly — completes within a
//! single step.

use crate::ast::{ImportDef, ImportKind, Item};
use crate::bytecode::{CaptureSrc, Const, FnProto, Instr, Pat};
use crate::compiler::{compile_program, module_global_name, ProgramCtx};
use crate::parser::parse;
use crate::value::shared::{HostBound, Lock, Sh, ShWeak};
use crate::value::{AtomCell, Closure, Value, Variant, VariantPayload};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct Fault {
    pub msg: String,
    /// "fn_name:line" where the fault was raised, when known
    pub at: Option<String>,
}

impl Fault {
    pub fn new(msg: impl Into<String>) -> Fault {
        Fault { msg: msg.into(), at: None }
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.at {
            Some(at) => write!(f, "fault at {}: {}", at, self.msg),
            None => write!(f, "fault: {}", self.msg),
        }
    }
}

#[derive(Debug)]
pub enum FunctError {
    Parse(String),
    Compile(String),
    Fault(Fault),
}

impl fmt::Display for FunctError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctError::Parse(m) => write!(f, "parse error: {}", m),
            FunctError::Compile(m) => write!(f, "compile error: {}", m),
            FunctError::Fault(fa) => write!(f, "{}", fa),
        }
    }
}

impl std::error::Error for FunctError {}

#[derive(Clone)]
pub struct Frame {
    pub fn_id: u32,
    /// Snapshot of the proto at call time: hot reload swaps the table, but
    /// in-flight frames finish on the code they started with (spec §8).
    pub proto: Sh<FnProto>,
    pub ip: u32,
    pub locals: Vec<Value>,
    pub upvals: Vec<Value>,
}

#[derive(Clone)]
pub enum Status {
    Running,
    Done(Value),
    Faulted(Fault),
}

/// All execution state. Plain data: Clone = snapshot (time travel).
#[derive(Clone)]
pub struct VmState {
    pub frames: Vec<Frame>,
    pub stack: Vec<Value>,
    pub status: Status,
}

impl VmState {
    pub fn current_line(&self) -> Option<u32> {
        let f = self.frames.last()?;
        Some(f.proto.line_at(f.ip as usize))
    }

    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, Status::Running)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StepResult {
    Running,
    Done(Value),
    Faulted(Fault),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunResult {
    Done(Value),
    Faulted(Fault),
    Paused(Cause),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Cause {
    FuelExhausted,
    Breakpoint(u32),
    NextLine(u32),
}

#[derive(Debug, Clone)]
pub enum StopWhen {
    Never,
    Fuel(u64),
    Breakpoints(HashSet<u32>),
    NextLine,
}

pub(crate) type NativeImpl = Sh<dyn Fn(&mut Funct, Vec<Value>) -> Result<Value, Fault> + Send + Sync>;

/// Bound for host callbacks registered into the engine (see `register_raw`).
pub trait HostFn: Fn(&mut Funct, Vec<Value>) -> Result<Value, Fault> + HostBound {}
impl<F: Fn(&mut Funct, Vec<Value>) -> Result<Value, Fault> + HostBound> HostFn for F {}

pub(crate) struct NativeEntry {
    pub name: String,
    pub f: NativeImpl,
}

pub(crate) type Getter = Sh<dyn Fn(&mut Funct, &Value) -> Result<Value, Fault> + Send + Sync>;

struct LoadResult {
    main: Option<u32>,
    exports: Vec<(String, u32)>,
}

/// What a loaded module exposes to importers.
#[derive(Debug, Clone)]
pub(crate) enum ModuleExports {
    /// file module: exported (plain name, global slot) pairs
    File(Vec<(String, u32)>),
    /// host module registered via `register_module`: global slot of its record
    Host(u32),
}

/// The engine: code table, globals, natives, modules, atom registry.
pub struct Funct {
    pub(crate) fns: Vec<Sh<FnProto>>,
    pub(crate) ctx: ProgramCtx,
    pub(crate) globals: Vec<Option<Value>>,
    pub(crate) natives: Vec<NativeEntry>,
    pub(crate) native_ids: HashMap<String, u32>,
    /// (type_name, field) -> getter, for Native host values
    pub(crate) getters: HashMap<(String, String), Getter>,
    pub(crate) atom_counter: u64,
    pub(crate) atoms: Vec<ShWeak<AtomCell>>,
    /// loaded modules by path; loaded once, cached
    pub(crate) modules: HashMap<String, ModuleExports>,
    /// directory module paths resolve against (`<root>/<path>.ft`)
    pub(crate) module_root: PathBuf,
    /// module-load stack for cycle detection
    loading: Vec<String>,
    /// `#[test]` functions seen in top-level (non-module) loads
    pub(crate) test_fns: Vec<String>,
}

impl Funct {
    /// Engine with the prelude (natives + funct-source stdlib) installed.
    pub fn new() -> Funct {
        let mut vm = Funct::bare();
        crate::prelude::install(&mut vm);
        // everything that exists at this point (natives + prelude stdlib) is
        // visible inside modules without an import
        let n = vm.ctx.global_names.len() as u32;
        vm.ctx.shared.extend(0..n);
        vm
    }

    /// Engine with no prelude at all (used by tests/snapshot restore).
    pub fn bare() -> Funct {
        Funct {
            fns: Vec::new(),
            ctx: ProgramCtx {
                fn_ids: HashMap::new(),
                fn_count: 0,
                global_ids: HashMap::new(),
                global_names: Vec::new(),
                shared: HashSet::new(),
            },
            globals: Vec::new(),
            natives: Vec::new(),
            native_ids: HashMap::new(),
            getters: HashMap::new(),
            atom_counter: 0,
            atoms: Vec::new(),
            modules: HashMap::new(),
            module_root: PathBuf::from("."),
            loading: Vec::new(),
            test_fns: Vec::new(),
        }
    }

    /// Set the directory module import paths resolve against.
    pub fn set_module_root(&mut self, root: impl Into<PathBuf>) {
        self.module_root = root.into();
    }

    // ---------- compiling & running ----------

    /// Compile and immediately run source. Re-defining an existing `fn` name
    /// hot-swaps it (same FnId, new code).
    pub fn eval(&mut self, src: &str) -> Result<Value, FunctError> {
        let main = self.load(src)?;
        match main {
            Some(fn_id) => {
                let mut st = self.state_for(fn_id, vec![])?;
                match self.run(&mut st, StopWhen::Never) {
                    RunResult::Done(v) => Ok(v),
                    RunResult::Faulted(f) => Err(FunctError::Fault(f)),
                    RunResult::Paused(_) => unreachable!("StopWhen::Never paused"),
                }
            }
            None => Ok(Value::Unit),
        }
    }

    /// Compile source and install its items, but run nothing. Returns the
    /// fn id of the top-level code, if any.
    pub fn load(&mut self, src: &str) -> Result<Option<u32>, FunctError> {
        self.load_with_prefix(src, None).map(|r| r.main)
    }

    fn load_with_prefix(&mut self, src: &str, prefix: Option<&str>) -> Result<LoadResult, FunctError> {
        let prog = parse(src).map_err(FunctError::Parse)?;
        // imports and extern declarations first, so the bindings exist when
        // the rest compiles
        for item in &prog.items {
            match item {
                Item::Import(imp) => self.process_import(imp, prefix)?,
                Item::Extern { name, .. } => self.process_extern(name),
                Item::ExternLet { name, .. } => {
                    // declare the slot (shared, like all host-provided
                    // globals); reading it unset faults loudly already
                    let g = self.ctx.ensure_global(name);
                    self.ctx.shared.insert(g);
                    self.sync_globals();
                }
                _ => {}
            }
        }
        let compiled =
            compile_program(&mut self.ctx, &prog, prefix).map_err(FunctError::Compile)?;
        for (id, proto) in compiled.protos {
            let id = id as usize;
            if self.fns.len() <= id {
                self.fns.resize_with(id + 1, || {
                    Sh::new(FnProto {
                        name: "<hole>".into(),
                        arity: 0,
                        num_locals: 0,
                        num_upvals: 0,
                        code: vec![],
                        consts: vec![],
                        pats: vec![],
                        lines: vec![],
                        closure_captures: vec![],
                    })
                });
            }
            self.fns[id] = Sh::new(proto);
        }
        self.sync_globals();
        for (gslot, fn_id) in compiled.fn_globals {
            self.globals[gslot as usize] =
                Some(Value::Closure(Sh::new(Closure { fn_id, upvals: vec![] })));
        }
        // tests register for the runner only from top-level files, not from
        // imported modules (run a module file directly to run its tests)
        if prefix.is_none() {
            for t in &compiled.tests {
                if !self.test_fns.contains(t) {
                    self.test_fns.push(t.clone());
                }
            }
        }
        Ok(LoadResult { main: compiled.main, exports: compiled.exports })
    }

    /// `extern fn name(...)`: if the host registered a native with this name
    /// it is already bound — nothing to do. Otherwise install a placeholder
    /// native that faults loudly when CALLED, so files declaring a host
    /// interface still load (e.g. for `funct test` over pure logic).
    fn process_extern(&mut self, name: &str) {
        if let Some(&g) = self.ctx.global_ids.get(name) {
            if self.globals.get(g as usize).map(|v| v.is_some()).unwrap_or(false) {
                return; // host already provided it
            }
        }
        let fn_name = name.to_string();
        self.register_raw(name, move |_vm, _args| {
            Err(Fault::new(format!(
                "host function `{}` was declared `extern` but this host does not provide it",
                fn_name
            )))
        });
    }

    /// Names of `#[test]` functions loaded so far (declaration order).
    pub fn test_names(&self) -> Vec<String> {
        self.test_fns.clone()
    }

    // ---------- modules ----------

    /// Load a module (if not already cached) and return its export names.
    pub fn load_module(&mut self, path: &str) -> Result<Vec<String>, FunctError> {
        self.ensure_module(path)?;
        Ok(self.module_export_names(path))
    }

    /// Re-read and re-evaluate a file module: functions hot-swap (importers
    /// see new code immediately); module top-level `let`s re-run, so module
    /// atoms are recreated. Previously imported `let` *values* keep the copy
    /// they were bound to.
    pub fn reload_module(&mut self, path: &str) -> Result<Vec<String>, FunctError> {
        if let Some(ModuleExports::Host(_)) = self.modules.get(path) {
            return Err(FunctError::Fault(Fault::new(format!(
                "`{}` is a host module registered from Rust; re-register it instead",
                path
            ))));
        }
        self.modules.remove(path);
        self.load_module(path)
    }

    /// Make a host record (already stored at global `gid`) importable.
    pub(crate) fn register_host_module(&mut self, name: &str, gid: u32) {
        self.modules.insert(name.to_string(), ModuleExports::Host(gid));
    }

    fn module_export_names(&self, path: &str) -> Vec<String> {
        match self.modules.get(path) {
            Some(ModuleExports::File(list)) => list.iter().map(|(n, _)| n.clone()).collect(),
            Some(ModuleExports::Host(gid)) => match self.globals.get(*gid as usize) {
                Some(Some(Value::Record(r))) => r.keys().cloned().collect(),
                _ => vec![],
            },
            None => vec![],
        }
    }

    fn ensure_module(&mut self, path: &str) -> Result<(), FunctError> {
        if self.modules.contains_key(path) {
            return Ok(());
        }
        if path.split('/').any(|seg| seg.is_empty() || seg == "." || seg == "..") {
            return Err(FunctError::Fault(Fault::new(format!(
                "invalid module path `{}`: segments must be plain names (no `..`, `.` or empty)",
                path
            ))));
        }
        if let Some(i) = self.loading.iter().position(|p| p == path) {
            let mut chain: Vec<&str> = self.loading[i..].iter().map(|s| s.as_str()).collect();
            chain.push(path);
            return Err(FunctError::Fault(Fault::new(format!(
                "circular module imports: {}",
                chain.join(" -> ")
            ))));
        }
        let file = self.module_root.join(format!("{}.ft", path));
        let src = std::fs::read_to_string(&file).map_err(|e| {
            FunctError::Fault(Fault::new(format!(
                "cannot load module `{}`: {} ({})",
                path,
                file.display(),
                e
            )))
        })?;
        self.loading.push(path.to_string());
        let result = (|| {
            let loaded = self.load_with_prefix(&src, Some(path))?;
            // run the module's top-level code now (its lets/side effects)
            if let Some(fn_id) = loaded.main {
                let mut st = self.state_for(fn_id, vec![])?;
                match self.run(&mut st, StopWhen::Never) {
                    RunResult::Done(_) => {}
                    RunResult::Faulted(f) => return Err(FunctError::Fault(f)),
                    RunResult::Paused(_) => unreachable!(),
                }
            }
            Ok(loaded.exports)
        })();
        self.loading.pop();
        let exports = result?;
        self.modules.insert(path.to_string(), ModuleExports::File(exports));
        Ok(())
    }

    /// Resolve one exported name from a loaded module to its current value.
    fn export_value(&self, path: &str, name: &str) -> Result<Value, FunctError> {
        let loud_missing = |available: Vec<String>| {
            FunctError::Fault(Fault::new(format!(
                "module `{}` has no export `{}` (exports: {})",
                path,
                name,
                if available.is_empty() { "none".to_string() } else { available.join(", ") }
            )))
        };
        match self.modules.get(path) {
            Some(ModuleExports::File(list)) => match list.iter().find(|(n, _)| n == name) {
                Some((_, gid)) => self
                    .globals
                    .get(*gid as usize)
                    .cloned()
                    .flatten()
                    .ok_or_else(|| {
                        FunctError::Fault(Fault::new(format!(
                            "export `{}` of module `{}` is uninitialized",
                            name, path
                        )))
                    }),
                None => Err(loud_missing(list.iter().map(|(n, _)| n.clone()).collect())),
            },
            Some(ModuleExports::Host(gid)) => match self.globals.get(*gid as usize) {
                Some(Some(Value::Record(r))) => r
                    .get(name)
                    .cloned()
                    .ok_or_else(|| loud_missing(r.keys().cloned().collect())),
                _ => Err(FunctError::Fault(Fault::new(format!(
                    "host module `{}` record is missing",
                    path
                )))),
            },
            None => Err(FunctError::Fault(Fault::new(format!("module `{}` not loaded", path)))),
        }
    }

    fn process_import(&mut self, imp: &ImportDef, prefix: Option<&str>) -> Result<(), FunctError> {
        self.ensure_module(&imp.path)?;
        match &imp.kind {
            ImportKind::Named(names) => {
                for (name, alias) in names {
                    let v = self.export_value(&imp.path, name)?;
                    let bind = alias.as_deref().unwrap_or(name);
                    let full = module_global_name(prefix, bind);
                    let g = self.ctx.ensure_global(&full);
                    self.sync_globals();
                    self.globals[g as usize] = Some(v);
                }
            }
            ImportKind::Qualified(alias) => {
                let bind = match alias {
                    Some(a) => a.clone(),
                    None => {
                        let last = imp.path.rsplit('/').next().unwrap_or(&imp.path);
                        let valid = !last.is_empty()
                            && last.chars().next().map(|c| c.is_ascii_lowercase() || c == '_').unwrap_or(false)
                            && last.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                        if !valid {
                            return Err(FunctError::Fault(Fault::new(format!(
                                "`{}` is not a usable module alias; add one: import \"{}\" as <name>",
                                last, imp.path
                            ))));
                        }
                        last.to_string()
                    }
                };
                let v = match self.modules.get(&imp.path).cloned() {
                    Some(ModuleExports::Host(gid)) => self
                        .globals
                        .get(gid as usize)
                        .cloned()
                        .flatten()
                        .ok_or_else(|| {
                            FunctError::Fault(Fault::new(format!(
                                "host module `{}` record is missing",
                                imp.path
                            )))
                        })?,
                    Some(ModuleExports::File(list)) => {
                        // Skip exports with no value yet: an `extern let` the
                        // host hasn't provided is uninitialized, and a bare
                        // `import "host"` must still load (the externs' real
                        // job is declaring the shared globals, a side effect of
                        // ensure_module above). extern fns always have a
                        // placeholder value, so they're always included.
                        let mut map = BTreeMap::new();
                        for (name, gid) in &list {
                            if let Some(Some(v)) = self.globals.get(*gid as usize) {
                                map.insert(name.clone(), v.clone());
                            }
                        }
                        Value::record(map)
                    }
                    None => unreachable!("ensure_module just loaded it"),
                };
                let full = module_global_name(prefix, &bind);
                let g = self.ctx.ensure_global(&full);
                self.sync_globals();
                self.globals[g as usize] = Some(v);
            }
        }
        Ok(())
    }

    /// Compile source like `eval`, but return a resumable VmState for its
    /// top-level code instead of running it.
    pub fn eval_resumable(&mut self, src: &str) -> Result<VmState, FunctError> {
        let main = self.load(src)?;
        match main {
            Some(fn_id) => self.state_for(fn_id, vec![]),
            None => Ok(VmState {
                frames: vec![],
                stack: vec![],
                status: Status::Done(Value::Unit),
            }),
        }
    }

    pub(crate) fn sync_globals(&mut self) {
        if self.globals.len() < self.ctx.global_names.len() {
            self.globals.resize(self.ctx.global_names.len(), None);
        }
    }

    fn state_for(&mut self, fn_id: u32, args: Vec<Value>) -> Result<VmState, FunctError> {
        let proto = self
            .fns
            .get(fn_id as usize)
            .cloned()
            .ok_or_else(|| FunctError::Fault(Fault::new(format!("unknown fn id {}", fn_id))))?;
        let frame = make_frame(fn_id, proto, args).map_err(FunctError::Fault)?;
        Ok(VmState { frames: vec![frame], stack: vec![], status: Status::Running })
    }

    pub fn global(&self, name: &str) -> Option<Value> {
        let id = *self.ctx.global_ids.get(name)?;
        self.globals.get(id as usize)?.clone()
    }

    /// Set (or create) a global binding from the host — e.g. injecting
    /// `canvas_w` before calling a handler. Host-provided globals are
    /// `shared`: visible inside modules without an import, like natives.
    pub fn set_global(&mut self, name: &str, v: Value) {
        let g = self.ctx.ensure_global(name);
        self.ctx.shared.insert(g);
        self.sync_globals();
        self.globals[g as usize] = Some(v);
    }

    /// Create a resumable state that calls global `name` with `args`.
    pub fn start(&mut self, name: &str, args: Vec<Value>) -> Result<VmState, FunctError> {
        let f = self
            .global(name)
            .ok_or_else(|| FunctError::Fault(Fault::new(format!("unknown function `{}`", name))))?;
        self.start_value(&f, args)
    }

    pub fn start_value(&mut self, f: &Value, args: Vec<Value>) -> Result<VmState, FunctError> {
        match f {
            Value::Closure(c) => {
                let proto = self.fns[c.fn_id as usize].clone();
                let mut frame = make_frame(c.fn_id, proto, args).map_err(FunctError::Fault)?;
                frame.upvals = c.upvals.clone();
                Ok(VmState { frames: vec![frame], stack: vec![], status: Status::Running })
            }
            Value::NativeFn(id) => {
                // no script frames; run it now and produce a Done state
                let f = self.natives[*id as usize].f.clone();
                match f(self, args) {
                    Ok(v) => Ok(VmState { frames: vec![], stack: vec![], status: Status::Done(v) }),
                    Err(e) => Err(FunctError::Fault(e)),
                }
            }
            other => Err(FunctError::Fault(Fault::new(format!(
                "value of type {} is not callable",
                other.type_name()
            )))),
        }
    }

    /// Call a script function to completion (used by host code and natives).
    pub fn call_value(&mut self, f: &Value, args: Vec<Value>) -> Result<Value, Fault> {
        let mut st = self.start_value(f, args).map_err(|e| match e {
            FunctError::Fault(f) => f,
            other => Fault::new(other.to_string()),
        })?;
        match self.run(&mut st, StopWhen::Never) {
            RunResult::Done(v) => Ok(v),
            RunResult::Faulted(f) => Err(f),
            RunResult::Paused(_) => unreachable!(),
        }
    }

    /// Call a global by name to completion.
    pub fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value, FunctError> {
        let f = self
            .global(name)
            .ok_or_else(|| FunctError::Fault(Fault::new(format!("unknown function `{}`", name))))?;
        self.call_value(&f, args).map_err(FunctError::Fault)
    }

    pub fn run(&mut self, st: &mut VmState, stop: StopWhen) -> RunResult {
        // Fast path: no breakpoints / single-stepping → a tight dispatch loop
        // with none of the per-instruction line tracking the debug modes need.
        // This is what `eval`, host calls, and ordinary execution use.
        match &stop {
            StopWhen::Never => return self.run_fast(st, None),
            StopWhen::Fuel(n) => return self.run_fast(st, Some(*n)),
            _ => {}
        }
        let mut fuel: Option<u64> = None;
        // for breakpoints: only trigger when we *enter* the line, so resuming
        // from a breakpoint doesn't immediately re-trigger it
        let mut last_line = st.current_line().unwrap_or(0);
        let start_line = st.current_line().unwrap_or(0);
        let start_depth = st.depth();
        loop {
            match &st.status {
                Status::Done(v) => return RunResult::Done(v.clone()),
                Status::Faulted(f) => return RunResult::Faulted(f.clone()),
                Status::Running => {}
            }
            if let Some(0) = fuel {
                return RunResult::Paused(Cause::FuelExhausted);
            }
            if let Some(line) = st.current_line() {
                match &stop {
                    StopWhen::Breakpoints(bps) => {
                        if line != last_line && bps.contains(&line) {
                            return RunResult::Paused(Cause::Breakpoint(line));
                        }
                    }
                    StopWhen::NextLine => {
                        if line != start_line && st.depth() <= start_depth {
                            return RunResult::Paused(Cause::NextLine(line));
                        }
                    }
                    _ => {}
                }
                last_line = line;
            }
            match self.step(st) {
                StepResult::Running => {}
                StepResult::Done(v) => return RunResult::Done(v),
                StepResult::Faulted(f) => return RunResult::Faulted(f),
            }
            if let Some(n) = fuel.as_mut() {
                *n -= 1;
            }
        }
    }

    /// Tight execution loop for the common case (no breakpoints / stepping):
    /// calls `step_inner` directly — no per-instruction line lookup, no extra
    /// `step()` dispatch layer. `fuel` = Some(budget) caps instruction count.
    fn run_fast(&mut self, st: &mut VmState, mut fuel: Option<u64>) -> RunResult {
        loop {
            // cheap tag check: a native tail-call on the last frame finishes by
            // setting status (rather than returning a value), so catch it here
            match &st.status {
                Status::Done(v) => return RunResult::Done(v.clone()),
                Status::Faulted(f) => return RunResult::Faulted(f.clone()),
                Status::Running => {}
            }
            if let Some(0) = fuel {
                return RunResult::Paused(Cause::FuelExhausted);
            }
            match self.step_inner(st) {
                Ok(None) => {}
                Ok(Some(v)) => {
                    st.status = Status::Done(v.clone());
                    return RunResult::Done(v);
                }
                Err(mut fault) => {
                    if fault.at.is_none() {
                        if let Some(f) = st.frames.last() {
                            let ip = (f.ip as usize).saturating_sub(1);
                            fault.at = Some(format!("{}:{}", f.proto.name, f.proto.line_at(ip)));
                        }
                    }
                    st.status = Status::Faulted(fault.clone());
                    return RunResult::Faulted(fault);
                }
            }
            if let Some(n) = fuel.as_mut() {
                *n -= 1;
            }
        }
    }

    // ---------- the step loop ----------

    /// Execute exactly one instruction.
    pub fn step(&mut self, st: &mut VmState) -> StepResult {
        match &st.status {
            Status::Done(v) => return StepResult::Done(v.clone()),
            Status::Faulted(f) => return StepResult::Faulted(f.clone()),
            Status::Running => {}
        }
        match self.step_inner(st) {
            Ok(Some(v)) => {
                st.status = Status::Done(v.clone());
                StepResult::Done(v)
            }
            Ok(None) => StepResult::Running,
            Err(mut fault) => {
                if fault.at.is_none() {
                    if let Some(f) = st.frames.last() {
                        let ip = (f.ip as usize).saturating_sub(1);
                        fault.at = Some(format!("{}:{}", f.proto.name, f.proto.line_at(ip)));
                    }
                }
                st.status = Status::Faulted(fault.clone());
                StepResult::Faulted(fault)
            }
        }
    }

    fn step_inner(&mut self, st: &mut VmState) -> Result<Option<Value>, Fault> {
        let frame = st
            .frames
            .last_mut()
            .ok_or_else(|| Fault::new("vm has no frames (already finished?)"))?;
        let ip = frame.ip as usize;
        // `Instr` is `Copy`, so this is a trivial register/stack copy, not a
        // heap-touching clone — the hottest line in the interpreter.
        let instr = *frame
            .proto
            .code
            .get(ip)
            .ok_or_else(|| Fault::new(format!("ip {} out of bounds in {}", ip, frame.proto.name)))?;
        frame.ip += 1;

        macro_rules! pop {
            () => {
                st.stack.pop().ok_or_else(|| Fault::new("operand stack underflow"))?
            };
        }

        match instr {
            Instr::Nop => {}
            Instr::Const(k) => {
                let frame = st.frames.last().unwrap();
                let v = match &frame.proto.consts[k as usize] {
                    Const::Int(i) => Value::Int(*i),
                    Const::Float(f) => Value::Float(*f),
                    Const::Str(s) => Value::str(s.clone()),
                    c => return Err(Fault::new(format!("cannot load const {:?}", c))),
                };
                st.stack.push(v);
            }
            Instr::Unit => st.stack.push(Value::Unit),
            Instr::True => st.stack.push(Value::Bool(true)),
            Instr::False => st.stack.push(Value::Bool(false)),
            Instr::LoadLocal(n) => {
                let v = st.frames.last().unwrap().locals[n as usize].clone();
                st.stack.push(v);
            }
            Instr::StoreLocal(n) => {
                let v = pop!();
                st.frames.last_mut().unwrap().locals[n as usize] = v;
            }
            Instr::LoadUpval(n) => {
                let v = st.frames.last().unwrap().upvals[n as usize].clone();
                st.stack.push(v);
            }
            Instr::LoadGlobal(g) => {
                let v = self.globals.get(g as usize).cloned().flatten().ok_or_else(|| {
                    let name = self
                        .ctx
                        .global_names
                        .get(g as usize)
                        .cloned()
                        .unwrap_or_else(|| format!("#{}", g));
                    Fault::new(format!("global `{}` used before it was initialized", name))
                })?;
                st.stack.push(v);
            }
            Instr::StoreGlobal(g) => {
                let v = pop!();
                self.sync_globals();
                if (g as usize) >= self.globals.len() {
                    self.globals.resize(g as usize + 1, None);
                }
                self.globals[g as usize] = Some(v);
            }
            Instr::NewCell => {
                let v = pop!();
                st.stack.push(Value::Cell(Sh::new(Lock::new(v))));
            }
            Instr::CellGet => {
                let c = pop!();
                match c {
                    Value::Cell(c) => st.stack.push(c.read().clone()),
                    other => return Err(Fault::new(format!("CellGet on {}", other.type_name()))),
                }
            }
            Instr::CellSet => {
                let v = pop!();
                let c = pop!();
                match c {
                    Value::Cell(c) => *c.write() = v,
                    other => return Err(Fault::new(format!("CellSet on {}", other.type_name()))),
                }
            }
            Instr::MakeList(n) => {
                let items = pop_n(&mut st.stack, n as usize)?;
                st.stack.push(Value::list(items));
            }
            Instr::MakeTuple(n) => {
                let items = pop_n(&mut st.stack, n as usize)?;
                st.stack.push(Value::tuple(items));
            }
            Instr::MakeRecord(names_k) => {
                let names = self.names_const(st, names_k)?;
                let vals = pop_n(&mut st.stack, names.len())?;
                let mut map = BTreeMap::new();
                for (n, v) in names.into_iter().zip(vals) {
                    map.insert(n, v);
                }
                st.stack.push(Value::record(map));
            }
            Instr::RecordUpdate(names_k) => {
                let names = self.names_const(st, names_k)?;
                let vals = pop_n(&mut st.stack, names.len())?;
                let base = pop!();
                let mut map = match base {
                    Value::Record(r) => r.clone(),
                    other => {
                        return Err(Fault::new(format!(
                            "record update `{{ ..base }}` needs a record, got {}",
                            other.type_name()
                        )))
                    }
                };
                for (n, v) in names.into_iter().zip(vals) {
                    map.insert(n, v);
                }
                st.stack.push(Value::Record(map));
            }
            Instr::GetField(name_k) => {
                let name = self.name_const(st, name_k)?;
                let recv = pop!();
                let v = self.get_field(&recv, &name)?;
                st.stack.push(v);
            }
            Instr::Index => {
                let idx = pop!();
                let recv = pop!();
                st.stack.push(index_value(&recv, &idx)?);
            }
            Instr::MakeVariantUnit { tag } => {
                let tag = self.name_const(st, tag)?;
                st.stack.push(Value::Variant(Sh::new(Variant { tag, payload: VariantPayload::Unit })));
            }
            Instr::MakeVariantPos { tag, count } => {
                let tag = self.name_const(st, tag)?;
                let items = pop_n(&mut st.stack, count as usize)?;
                st.stack
                    .push(Value::Variant(Sh::new(Variant { tag, payload: VariantPayload::Positional(items) })));
            }
            Instr::MakeVariantNamed { tag, names } => {
                let tag = self.name_const(st, tag)?;
                let names = self.names_const(st, names)?;
                let vals = pop_n(&mut st.stack, names.len())?;
                let mut map = BTreeMap::new();
                for (n, v) in names.into_iter().zip(vals) {
                    map.insert(n, v);
                }
                st.stack
                    .push(Value::Variant(Sh::new(Variant { tag, payload: VariantPayload::Named(map) })));
            }
            Instr::Add => bin_op(&mut st.stack, BinKind::Add)?,
            Instr::Sub => bin_op(&mut st.stack, BinKind::Sub)?,
            Instr::Mul => bin_op(&mut st.stack, BinKind::Mul)?,
            Instr::Div => bin_op(&mut st.stack, BinKind::Div)?,
            Instr::Mod => bin_op(&mut st.stack, BinKind::Mod)?,
            Instr::Pow => bin_op(&mut st.stack, BinKind::Pow)?,
            Instr::Eq => {
                let b = pop!();
                let a = pop!();
                st.stack.push(Value::Bool(a == b));
            }
            Instr::Ne => {
                let b = pop!();
                let a = pop!();
                st.stack.push(Value::Bool(a != b));
            }
            Instr::Lt => cmp_op(&mut st.stack, |o| o == std::cmp::Ordering::Less)?,
            Instr::Le => cmp_op(&mut st.stack, |o| o != std::cmp::Ordering::Greater)?,
            Instr::Gt => cmp_op(&mut st.stack, |o| o == std::cmp::Ordering::Greater)?,
            Instr::Ge => cmp_op(&mut st.stack, |o| o != std::cmp::Ordering::Less)?,
            Instr::Neg => {
                let v = pop!();
                st.stack.push(match v {
                    Value::Int(i) => Value::Int(
                        i.checked_neg().ok_or_else(|| Fault::new("integer overflow in negation"))?,
                    ),
                    Value::Float(f) => Value::Float(-f),
                    other => return Err(Fault::new(format!("cannot negate {}", other.type_name()))),
                });
            }
            Instr::Not => {
                let v = pop!();
                match v {
                    Value::Bool(b) => st.stack.push(Value::Bool(!b)),
                    other => return Err(Fault::new(format!("`not` needs a Bool, got {}", other.type_name()))),
                }
            }
            Instr::MakeRange { inclusive } => {
                let hi = pop!();
                let lo = pop!();
                match (lo, hi) {
                    (Value::Int(a), Value::Int(b)) => st.stack.push(Value::Range(a, b, inclusive)),
                    (a, b) => {
                        return Err(Fault::new(format!(
                            "range bounds must be Int, got {} and {}",
                            a.type_name(),
                            b.type_name()
                        )))
                    }
                }
            }
            Instr::Jump(t) => st.frames.last_mut().unwrap().ip = t,
            Instr::JumpIfFalse(t) => {
                let v = pop!();
                match v {
                    Value::Bool(false) => st.frames.last_mut().unwrap().ip = t,
                    Value::Bool(true) => {}
                    other => {
                        return Err(Fault::new(format!(
                            "condition must be a Bool, got {}",
                            other.type_name()
                        )))
                    }
                }
            }
            Instr::JumpIfFalsePeek(t) => {
                match st.stack.last() {
                    Some(Value::Bool(false)) => st.frames.last_mut().unwrap().ip = t,
                    Some(Value::Bool(true)) => {}
                    Some(other) => {
                        return Err(Fault::new(format!(
                            "`and`/`or` operands must be Bool, got {}",
                            other.type_name()
                        )))
                    }
                    None => return Err(Fault::new("operand stack underflow")),
                }
            }
            Instr::JumpIfTruePeek(t) => {
                match st.stack.last() {
                    Some(Value::Bool(true)) => st.frames.last_mut().unwrap().ip = t,
                    Some(Value::Bool(false)) => {}
                    Some(other) => {
                        return Err(Fault::new(format!(
                            "`and`/`or` operands must be Bool, got {}",
                            other.type_name()
                        )))
                    }
                    None => return Err(Fault::new("operand stack underflow")),
                }
            }
            Instr::MatchPat { pat, fail } => {
                let subject = st
                    .stack
                    .last()
                    .ok_or_else(|| Fault::new("operand stack underflow"))?
                    .clone();
                let frame = st.frames.last_mut().unwrap();
                let pat = frame.proto.pats[pat as usize].clone();
                if !match_pat(&pat, &subject, &mut frame.locals) {
                    frame.ip = fail;
                }
            }
            Instr::MakeClosure { fn_id, captures } => {
                let frame = st.frames.last().unwrap();
                let caps = &frame.proto.closure_captures[captures as usize];
                let upvals: Vec<Value> = caps
                    .iter()
                    .map(|c| match c {
                        CaptureSrc::Local(s) => frame.locals[*s as usize].clone(),
                        CaptureSrc::Upval(i) => frame.upvals[*i as usize].clone(),
                    })
                    .collect();
                st.stack.push(Value::Closure(Sh::new(Closure { fn_id, upvals })));
            }
            Instr::Call(argc) => {
                let args = pop_n(&mut st.stack, argc as usize)?;
                let callee = pop!();
                self.do_call(st, callee, args, false)?;
            }
            Instr::TailCall(argc) => {
                let args = pop_n(&mut st.stack, argc as usize)?;
                let callee = pop!();
                self.do_call(st, callee, args, true)?;
            }
            Instr::Invoke { name, global, argc } => {
                let name = self.name_const(st, name)?;
                let args = pop_n(&mut st.stack, argc as usize)?;
                let recv = pop!();
                // UFCS: a record field named `name` wins (spec §4.1)
                let field_callable = match &recv {
                    Value::Record(r) => r.get(&name).cloned(),
                    _ => None,
                };
                match field_callable {
                    Some(f @ (Value::Closure(_) | Value::NativeFn(_))) => {
                        self.do_call(st, f, args, false)?;
                    }
                    Some(other) => {
                        return Err(Fault::new(format!(
                            "field `{}` is not callable (it is {})",
                            name,
                            other.type_name()
                        )))
                    }
                    None => {
                        // compile-time resolved slot (module-aware), then a
                        // plain runtime lookup for late-defined globals
                        let f = global
                            .and_then(|g| self.globals.get(g as usize).cloned().flatten())
                            .or_else(|| self.global(&name))
                            .ok_or_else(|| {
                                Fault::new(format!(
                                    "no function `{}` for method call on {}",
                                    name,
                                    recv.type_name()
                                ))
                            })?;
                        let mut full_args = Vec::with_capacity(args.len() + 1);
                        full_args.push(recv);
                        full_args.extend(args);
                        self.do_call(st, f, full_args, false)?;
                    }
                }
            }
            Instr::Ret => {
                let ret = pop!();
                st.frames.pop();
                if st.frames.is_empty() {
                    return Ok(Some(ret));
                }
                st.stack.push(ret);
            }
            Instr::Deref => {
                let v = pop!();
                match v {
                    Value::Atom(a) => st.stack.push(a.value.read().clone()),
                    other => {
                        return Err(Fault::new(format!(
                            "`@` deref needs an Atom, got {}",
                            other.type_name()
                        )))
                    }
                }
            }
            Instr::Try => {
                let v = pop!();
                match &v {
                    Value::Variant(var) => match (var.tag.as_str(), &var.payload) {
                        ("Ok", VariantPayload::Positional(p)) | ("Some", VariantPayload::Positional(p))
                            if p.len() == 1 =>
                        {
                            st.stack.push(p[0].clone());
                        }
                        ("Err", _) | ("None", _) => {
                            // return the carrier from the enclosing function
                            st.frames.pop();
                            if st.frames.is_empty() {
                                return Ok(Some(v));
                            }
                            st.stack.push(v);
                        }
                        _ => {
                            return Err(Fault::new(format!(
                                "`?` needs Ok/Err/Some/None, got {}",
                                v.type_name()
                            )))
                        }
                    },
                    other => {
                        return Err(Fault::new(format!(
                            "`?` needs a Result/Option, got {}",
                            other.type_name()
                        )))
                    }
                }
            }
            Instr::Fault(msg_k) => {
                let msg = self.name_const(st, msg_k)?;
                return Err(Fault::new(msg));
            }
            Instr::Pop => {
                pop!();
            }
            Instr::Dup => {
                let v = st.stack.last().cloned().ok_or_else(|| Fault::new("stack underflow"))?;
                st.stack.push(v);
            }
            Instr::IterNext { iter, idx, end } => {
                let frame = st.frames.last_mut().unwrap();
                let i = match &frame.locals[idx as usize] {
                    Value::Int(i) => *i,
                    other => return Err(Fault::new(format!("loop index corrupted: {}", other.type_name()))),
                };
                let item: Option<Value> = match &frame.locals[iter as usize] {
                    Value::List(items) | Value::Tuple(items) => items.get(i as usize).cloned(),
                    Value::Range(a, b, inc) => {
                        let v = a + i;
                        let ok = if *inc { v <= *b } else { v < *b };
                        if ok {
                            Some(Value::Int(v))
                        } else {
                            None
                        }
                    }
                    Value::Str(s) => s.chars().nth(i as usize).map(|c| Value::str(c.to_string())),
                    other => {
                        return Err(Fault::new(format!(
                            "cannot iterate over {}",
                            other.type_name()
                        )))
                    }
                };
                match item {
                    Some(v) => {
                        frame.locals[idx as usize] = Value::Int(i + 1);
                        st.stack.push(v);
                    }
                    None => frame.ip = end,
                }
            }
        }
        Ok(None)
    }

    fn do_call(&mut self, st: &mut VmState, callee: Value, args: Vec<Value>, tail: bool) -> Result<(), Fault> {
        match callee {
            Value::Closure(c) => {
                let proto = self
                    .fns
                    .get(c.fn_id as usize)
                    .cloned()
                    .ok_or_else(|| Fault::new(format!("unknown fn id {}", c.fn_id)))?;
                let mut frame = make_frame(c.fn_id, proto, args)?;
                frame.upvals = c.upvals.clone();
                if tail {
                    st.frames.pop();
                }
                st.frames.push(frame);
                Ok(())
            }
            Value::NativeFn(id) => {
                let f = self
                    .natives
                    .get(id as usize)
                    .ok_or_else(|| Fault::new(format!("unknown native fn id {}", id)))?
                    .f
                    .clone();
                let result = f(self, args)?;
                if tail {
                    // behave like `return result`
                    st.frames.pop();
                    if st.frames.is_empty() {
                        st.status = Status::Done(result);
                        return Ok(());
                    }
                }
                st.stack.push(result);
                Ok(())
            }
            other => Err(Fault::new(format!("value of type {} is not callable", other.type_name()))),
        }
    }

    pub(crate) fn get_field(&mut self, recv: &Value, name: &str) -> Result<Value, Fault> {
        match recv {
            Value::Record(r) => r
                .get(name)
                .cloned()
                .ok_or_else(|| Fault::new(format!("record has no field `{}`", name))),
            Value::Variant(v) => match &v.payload {
                VariantPayload::Named(fields) => fields
                    .get(name)
                    .cloned()
                    .ok_or_else(|| Fault::new(format!("variant {} has no field `{}`", v.tag, name))),
                _ => Err(Fault::new(format!("variant {} has no named fields", v.tag))),
            },
            Value::Atom(a) if name == "value" => Ok(a.value.read().clone()),
            Value::Native(n) => {
                let key = (n.type_name.to_string(), name.to_string());
                match self.getters.get(&key).cloned() {
                    Some(g) => g(self, recv),
                    None => Err(Fault::new(format!(
                        "native type {} has no registered field `{}`",
                        n.type_name, name
                    ))),
                }
            }
            other => Err(Fault::new(format!("{} has no fields", other.type_name()))),
        }
    }

    fn name_const(&self, st: &VmState, k: u32) -> Result<String, Fault> {
        let frame = st.frames.last().ok_or_else(|| Fault::new("no frame"))?;
        match &frame.proto.consts[k as usize] {
            Const::Name(s) | Const::Str(s) => Ok(s.clone()),
            c => Err(Fault::new(format!("expected name const, got {:?}", c))),
        }
    }

    fn names_const(&self, st: &VmState, k: u32) -> Result<Vec<String>, Fault> {
        let frame = st.frames.last().ok_or_else(|| Fault::new("no frame"))?;
        match &frame.proto.consts[k as usize] {
            Const::Names(s) => Ok(s.clone()),
            c => Err(Fault::new(format!("expected names const, got {:?}", c))),
        }
    }

    // ---------- atoms ----------

    pub fn make_atom(&mut self, v: Value) -> Value {
        let id = self.atom_counter;
        self.atom_counter += 1;
        let cell = Sh::new(AtomCell {
            id,
            value: Lock::new(v),
            watchers: Lock::new(Vec::new()),
        });
        self.atoms.push(Sh::downgrade(&cell));
        Value::Atom(cell)
    }

    pub(crate) fn fire_watchers(&mut self, atom: &Sh<AtomCell>, old: Value, new: Value) -> Result<(), Fault> {
        let watchers: Vec<Value> = atom.watchers.read().iter().map(|(_, f)| f.clone()).collect();
        for w in watchers {
            self.call_value(&w, vec![old.clone(), new.clone()])?;
        }
        Ok(())
    }

    pub fn live_atoms(&mut self) -> Vec<Sh<AtomCell>> {
        self.atoms.retain(|w| w.strong_count() > 0);
        self.atoms.iter().filter_map(|w| w.upgrade()).collect()
    }
}

impl Default for Funct {
    fn default() -> Self {
        Funct::new()
    }
}

fn make_frame(fn_id: u32, proto: Sh<FnProto>, args: Vec<Value>) -> Result<Frame, Fault> {
    if args.len() != proto.arity as usize {
        return Err(Fault::new(format!(
            "{} expects {} argument(s), got {}",
            proto.name,
            proto.arity,
            args.len()
        )));
    }
    let mut locals = args;
    locals.resize(proto.num_locals as usize, Value::Unit);
    Ok(Frame { fn_id, proto, ip: 0, locals, upvals: vec![] })
}

fn pop_n(stack: &mut Vec<Value>, n: usize) -> Result<Vec<Value>, Fault> {
    if stack.len() < n {
        return Err(Fault::new("operand stack underflow"));
    }
    Ok(stack.split_off(stack.len() - n))
}

enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

fn bin_op(stack: &mut Vec<Value>, kind: BinKind) -> Result<(), Fault> {
    let b = stack.pop().ok_or_else(|| Fault::new("operand stack underflow"))?;
    let a = stack.pop().ok_or_else(|| Fault::new("operand stack underflow"))?;
    use Value::*;
    let v = match (&kind, &a, &b) {
        // string concat
        (BinKind::Add, Str(x), Str(y)) => Value::str(format!("{}{}", x, y)),
        // list concat
        (BinKind::Add, List(x), List(y)) => {
            let mut items = (**x).clone();
            items.extend(y.iter().cloned());
            Value::list_v(items)
        }
        (_, Int(x), Int(y)) => match kind {
            BinKind::Add => Int(x.checked_add(*y).ok_or_else(|| Fault::new("integer overflow in +"))?),
            BinKind::Sub => Int(x.checked_sub(*y).ok_or_else(|| Fault::new("integer overflow in -"))?),
            BinKind::Mul => Int(x.checked_mul(*y).ok_or_else(|| Fault::new("integer overflow in *"))?),
            BinKind::Div => {
                if *y == 0 {
                    return Err(Fault::new("division by zero"));
                }
                Int(x / y)
            }
            BinKind::Mod => {
                if *y == 0 {
                    return Err(Fault::new("modulo by zero"));
                }
                Int(x % y)
            }
            BinKind::Pow => {
                if *y >= 0 {
                    let e: u32 = (*y)
                        .try_into()
                        .map_err(|_| Fault::new("exponent too large"))?;
                    Int(x.checked_pow(e).ok_or_else(|| Fault::new("integer overflow in **"))?)
                } else {
                    Float((*x as f64).powi(*y as i32))
                }
            }
        },
        (_, a, b) => {
            let (x, y) = match (as_f64(a), as_f64(b)) {
                (Some(x), Some(y)) => (x, y),
                _ => {
                    let op = match kind {
                        BinKind::Add => "+",
                        BinKind::Sub => "-",
                        BinKind::Mul => "*",
                        BinKind::Div => "/",
                        BinKind::Mod => "%",
                        BinKind::Pow => "**",
                    };
                    return Err(Fault::new(format!(
                        "cannot apply `{}` to {} and {}",
                        op,
                        a.type_name(),
                        b.type_name()
                    )));
                }
            };
            match kind {
                BinKind::Add => Float(x + y),
                BinKind::Sub => Float(x - y),
                BinKind::Mul => Float(x * y),
                BinKind::Div => {
                    if y == 0.0 {
                        return Err(Fault::new("division by zero"));
                    }
                    Float(x / y)
                }
                BinKind::Mod => {
                    if y == 0.0 {
                        return Err(Fault::new("modulo by zero"));
                    }
                    Float(x % y)
                }
                BinKind::Pow => Float(x.powf(y)),
            }
        }
    };
    stack.push(v);
    Ok(())
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    }
}

fn cmp_op(stack: &mut Vec<Value>, test: fn(std::cmp::Ordering) -> bool) -> Result<(), Fault> {
    let b = stack.pop().ok_or_else(|| Fault::new("operand stack underflow"))?;
    let a = stack.pop().ok_or_else(|| Fault::new("operand stack underflow"))?;
    let ord = match (&a, &b) {
        (Value::Str(x), Value::Str(y)) => x.cmp(y),
        (x, y) => match (as_f64(x), as_f64(y)) {
            (Some(x), Some(y)) => x
                .partial_cmp(&y)
                .ok_or_else(|| Fault::new("cannot compare NaN"))?,
            _ => {
                return Err(Fault::new(format!(
                    "cannot compare {} and {}",
                    a.type_name(),
                    b.type_name()
                )))
            }
        },
    };
    stack.push(Value::Bool(test(ord)));
    Ok(())
}

fn index_value(recv: &Value, idx: &Value) -> Result<Value, Fault> {
    match (recv, idx) {
        (Value::List(items), Value::Int(i)) | (Value::Tuple(items), Value::Int(i)) => {
            let i = *i;
            if i < 0 || i as usize >= items.len() {
                return Err(Fault::new(format!(
                    "index {} out of bounds (length {})",
                    i,
                    items.len()
                )));
            }
            Ok(items[i as usize].clone())
        }
        (Value::Record(r), Value::Str(k)) => r
            .get(&**k)
            .cloned()
            .ok_or_else(|| Fault::new(format!("record has no field `{}`", k))),
        (Value::Str(s), Value::Int(i)) => {
            let c = s
                .chars()
                .nth(*i as usize)
                .ok_or_else(|| Fault::new(format!("string index {} out of bounds", i)))?;
            Ok(Value::str(c.to_string()))
        }
        (r, i) => Err(Fault::new(format!(
            "cannot index {} with {}",
            r.type_name(),
            i.type_name()
        ))),
    }
}

/// Try to match `pat` against `v`, writing bindings into `locals`.
pub fn match_pat(pat: &Pat, v: &Value, locals: &mut Vec<Value>) -> bool {
    match pat {
        Pat::Wildcard => true,
        Pat::Bind(slot) => {
            locals[*slot as usize] = v.clone();
            true
        }
        Pat::LitInt(i) => v == &Value::Int(*i),
        Pat::LitFloat(f) => v == &Value::Float(*f),
        Pat::LitStr(s) => matches!(v, Value::Str(x) if &**x == s.as_str()),
        Pat::LitBool(b) => v == &Value::Bool(*b),
        Pat::LitUnit => matches!(v, Value::Unit),
        Pat::VariantPos { tag, items } => match v {
            Value::Variant(var) if var.tag == *tag => match &var.payload {
                VariantPayload::Unit => items.is_empty(),
                VariantPayload::Positional(vals) => {
                    vals.len() == items.len()
                        && items.iter().zip(vals).all(|(p, x)| match_pat(p, x, locals))
                }
                VariantPayload::Named(_) => false,
            },
            _ => false,
        },
        Pat::VariantNamed { tag, fields, rest } => match v {
            Value::Variant(var) if var.tag == *tag => match &var.payload {
                VariantPayload::Named(vals) => {
                    match_fields(fields, *rest, vals, locals)
                }
                _ => false,
            },
            _ => false,
        },
        Pat::Record { fields, rest } => match v {
            Value::Record(vals) => match_fields(fields, *rest, vals, locals),
            _ => false,
        },
        Pat::Tuple(items) => match v {
            Value::Tuple(vals) => {
                vals.len() == items.len()
                    && items.iter().zip(vals.iter()).all(|(p, x)| match_pat(p, x, locals))
            }
            _ => false,
        },
        Pat::List { items, rest } => match v {
            Value::List(vals) => {
                match rest {
                    None => {
                        vals.len() == items.len()
                            && items.iter().zip(vals.iter()).all(|(p, x)| match_pat(p, x, locals))
                    }
                    Some(rest_bind) => {
                        if vals.len() < items.len() {
                            return false;
                        }
                        if !items.iter().zip(vals.iter()).all(|(p, x)| match_pat(p, x, locals)) {
                            return false;
                        }
                        if let Some(slot) = rest_bind {
                            let rest_items: Vec<Value> =
                                vals.iter().skip(items.len()).cloned().collect();
                            locals[*slot as usize] = Value::list(rest_items);
                        }
                        true
                    }
                }
            }
            _ => false,
        },
        Pat::Range { lo, hi, inclusive } => match v {
            Value::Int(i) => {
                if *inclusive {
                    i >= lo && i <= hi
                } else {
                    i >= lo && i < hi
                }
            }
            _ => false,
        },
        Pat::Or(alts) => alts.iter().any(|p| match_pat(p, v, locals)),
        Pat::As(inner, slot) => {
            if match_pat(inner, v, locals) {
                locals[*slot as usize] = v.clone();
                true
            } else {
                false
            }
        }
    }
}

/// Field lookup shared by record patterns (`FMap`) and named-variant patterns
/// (still a plain `BTreeMap`), so the matcher works on both.
trait FieldMap {
    fn flen(&self) -> usize;
    fn fget(&self, k: &str) -> Option<&Value>;
}
impl FieldMap for BTreeMap<String, Value> {
    fn flen(&self) -> usize {
        self.len()
    }
    fn fget(&self, k: &str) -> Option<&Value> {
        self.get(k)
    }
}
impl FieldMap for crate::value::FMap {
    fn flen(&self) -> usize {
        self.len()
    }
    fn fget(&self, k: &str) -> Option<&Value> {
        self.get(k)
    }
}

fn match_fields(
    fields: &[(String, Pat)],
    rest: bool,
    vals: &impl FieldMap,
    locals: &mut Vec<Value>,
) -> bool {
    if !rest && vals.flen() != fields.len() {
        return false;
    }
    for (name, p) in fields {
        match vals.fget(name) {
            Some(x) => {
                if !match_pat(p, x, locals) {
                    return false;
                }
            }
            None => return false,
        }
    }
    true
}
