//! Clojure on the dynamic-toolkit.
//!
//! See `docs/microlisp-plan.md` for the underlying incremental-JIT
//! infrastructure. This crate adapts the microlisp pattern with
//! Clojure-specific data shapes (namespaces, vars, vectors, maps).

pub mod collections;
pub mod compile;
pub mod expand;
pub mod externs;
pub mod host;
pub mod namespace;
pub mod printer;
pub mod reader;
pub mod symbols;
pub mod types;
pub mod value;

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use dynexec::NanBoxConfig;
use dynir::builder::ModuleBuilder;
use dynir::ir::FuncRef;
use dynir::types::{Signature, Type};
use dynlang::gc::DynGcRuntime;
use dynlang::{DynModule, GcConfig, GcPolicy, NanBoxTags};
use dynlower::{CallMode, JitModule};
use dynobj::roots::{FrameChain, RootSet};
use dynruntime::active_jit_safepoint_handler;

use crate::host::Host;
use crate::symbols::SymbolTable;
use crate::types::declare_types;

const JIT_CALL_TABLE_CAPACITY: usize = 64 * 1024;
const JIT_LITERAL_POOL_CAPACITY: usize = 64 * 1024;
/// Default heap size: 4 MiB. Each semispace is half this.
const HEAP_SIZE: usize = 4 * 1024 * 1024;

pub struct Engine {
    /// dynlang module — used to look up obj_types, externs, slow paths.
    /// Kept alive because the GC's type table aliases its `obj_types`.
    pub dm: DynModule,
    /// Compile-time mutable state. Locked during compile (a brief
    /// section before extend); released so `call_compiled` running
    /// concurrently isn't blocked on it.
    compile: Mutex<CompileState>,
    /// JIT module behind a `RwLock`. **All** access is via the read
    /// guard — including `extend`, which since the toolkit migration
    /// to growable per-function metadata takes `&self`. The
    /// JitModule's internal `extend_lock` serializes concurrent
    /// extends; multiple `run_jit` callers and concurrent extends all
    /// hold read guards and proceed in parallel.
    jit: RwLock<Box<JitModule>>,
    /// GC runtime owning the heap. Boxed for stable address.
    gc: Box<DynGcRuntime>,
    /// Per-thread shadow stack for rooting raw NanBox handles across
    /// allocations. Boxed for a stable address.
    chain: Box<FrameChain>,
    /// Long-lived heap roots: slot 0 = Registry singleton,
    /// slot 1 = `clojure.core` namespace.
    globals: Box<RootSet>,
    /// Host context (symbol table, type IDs, GC pointer).
    pub host: Host,
    /// GC policy applied at every JIT call.
    jit_gc_policy: GcPolicy,
}

/// Compile-side mutable state, locked during the (brief) compile
/// pass. Doesn't include the JitModule itself — that lives in its
/// own `RwLock` so concurrent runs aren't blocked.
struct CompileState {
    mb: ModuleBuilder,
    func_refs: HashMap<String, FuncRef>,
    /// Externs in declaration order. Mutated only at startup.
    externs: Vec<*const u8>,
    anon_counter: u32,
}

const GLOBAL_SLOT_REGISTRY: usize = 0;
const GLOBAL_SLOT_CORE_NS: usize = 1;

// Engine is Send + Sync so a single instance can be shared by
// multiple threads via `Arc<Engine>`. The unsafe pieces:
//   - `host.gc: *const DynGcRuntime` — points into our own
//     `Box<DynGcRuntime>`, which stays alive as long as the Engine.
//   - The `chain: Box<FrameChain>` is shared, but only mutated under
//     the implicit serialization of `eval` (which still requires
//     `&mut self`). The new concurrent path `call_compiled` doesn't
//     install or mutate the chain.
//   - The DynGcRuntime contains a `RefCell<extra_root_sources>`. We
//     only `borrow_mut` it from `eval`, which is `&mut self`. The
//     concurrent `call_compiled` path only triggers `borrow()` (read)
//     access via `gc.run_jit`, which is safe to call from many
//     threads.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    pub fn new() -> Self {
        // Set up the dynlang module FIRST and declare all heap types
        // on it. The GC's type table aliases dm.obj_types.
        let gc_config = GcConfig::generational(HEAP_SIZE);
        let tags = NanBoxTags::default();
        let mut dm = DynModule::new(gc_config.clone(), tags.clone());
        let types = declare_types(&mut dm);

        // GC runtime: needs obj_types to scan heap objects correctly.
        let gc = Box::new(DynGcRuntime::new(&gc_config, &tags, &dm.obj_types));
        let gc_ptr: *const DynGcRuntime = &*gc;

        // JitModule. ControlAware so we can later add macros without
        // re-architecting; safepoint_handler is the standard one.
        let call_mode = CallMode::ControlAware {
            safepoint_handler: active_jit_safepoint_handler as u64,
        };
        let jit = Box::new(JitModule::new_empty::<
            NanBoxConfig,
            dynlower::Arm64Backend,
            dynlower::regalloc::LinearScanAllocator,
        >(
            JIT_CALL_TABLE_CAPACITY,
            JIT_LITERAL_POOL_CAPACITY,
            call_mode,
        ));

        let chain = Box::new(FrameChain::new());

        // Reserve two slots for the Registry singleton and the
        // `clojure.core` namespace. We allocate them later (once the
        // GC + host are wired) and write them into these slots.
        let mut globals = Box::new(RootSet::new());
        globals.add(value::NIL);
        globals.add(value::NIL);

        // Declare every primitive on the ModuleBuilder. Each gets a
        // FuncRef that the compiler looks up by source name when it
        // sees `(name args...)`. The extern pointers go into `externs`
        // in the same order as their FuncRef's table-index — the
        // JitModule reads this slice when filling call-table slots.
        let mut mb = ModuleBuilder::new();
        let mut func_refs: HashMap<String, FuncRef> = HashMap::new();
        let mut externs: Vec<*const u8> = Vec::new();
        for prim in externs::all_prims() {
            let sig = Signature {
                params: vec![Type::I64; prim.arity],
                ret: Some(Type::I64),
            };
            let fref = mb.declare_extern(prim.name, sig);
            func_refs.insert(prim.name.to_string(), fref);
            externs.push(prim.ptr);
        }

        let mut engine = Engine {
            dm,
            compile: Mutex::new(CompileState {
                mb,
                func_refs,
                externs,
                anon_counter: 0,
            }),
            jit: RwLock::new(jit),
            gc,
            chain,
            globals,
            host: Host {
                sym: std::sync::Mutex::new(SymbolTable::new()),
                gc: gc_ptr,
                types,
            },
            jit_gc_policy: GcPolicy::OnPressure { threshold: 0.75 },
        };

        // Register the JIT literal pool, the FrameChain, and the
        // long-lived `globals` slot array as GC root sources. All
        // three must have stable addresses for the lifetime of the
        // engine, which the surrounding `Box`es guarantee. The
        // literal-pool pointer comes from inside the RwLock; the
        // pointer is stable across read/write transitions because it
        // points into the inner `Box`'s heap allocation.
        let pool_ptr: *const dyn dynobj::RootSource = {
            let g = engine.jit.read().unwrap();
            g.literal_pool() as *const dyn dynobj::RootSource
        };
        unsafe { engine.gc.register_extra_root_source(pool_ptr) };
        let chain_ptr: *const dyn dynobj::RootSource = &*engine.chain;
        unsafe { engine.gc.register_extra_root_source(chain_ptr) };
        let globals_ptr: *const dyn dynobj::RootSource = &*engine.globals;
        unsafe { engine.gc.register_extra_root_source(globals_ptr) };

        // Allocate the Registry singleton and the `clojure.core`
        // namespace, writing both into `globals` so they survive
        // future collections.
        engine.bootstrap_namespaces();

        engine
    }

    /// Allocate the singleton Registry and the `clojure.core`
    /// Namespace. Both are stored in `globals` and remain alive for
    /// the lifetime of the engine.
    fn bootstrap_namespaces(&mut self) {
        let _gc_thread = self.gc.install_thread();
        let _host_g = host::install(&self.host);
        let _chain_g = dynobj::roots::install_chain(&self.chain);
        let core_name = {
            let mut sym = self.host.sym.lock().unwrap();
            sym.intern("clojure.core")
        };
        dynobj::roots::with_scope(64, |scope| {
            let registry = namespace::alloc_registry(scope);
            let core_ns = namespace::registry_create_ns(
                scope,
                registry.get(),
                value::encode_sym_id(core_name),
            );
            self.globals.set(GLOBAL_SLOT_REGISTRY, registry.get());
            self.globals.set(GLOBAL_SLOT_CORE_NS, core_ns.get());
        });
    }

    /// Tagged pointer to the `clojure.core` namespace. Re-read on
    /// every access in case a GC moved it.
    pub fn core_ns(&self) -> u64 {
        self.globals.get(GLOBAL_SLOT_CORE_NS)
    }

    /// Tagged pointer to the singleton Registry.
    pub fn registry(&self) -> u64 {
        self.globals.get(GLOBAL_SLOT_REGISTRY)
    }

    /// Call an already-compiled function from the `clojure.core`
    /// namespace by name. Takes `&self`, so multiple threads can
    /// hold an `Arc<Engine>` and call concurrently — the JIT is
    /// behind a `RwLock` whose read guard is shared between callers.
    ///
    /// Used to demonstrate per-form parallelism: while one thread is
    /// computing inside `gc.run_jit`, another thread can also be in
    /// `gc.run_jit` against a different (or the same!) FuncRef.
    ///
    /// Returns NIL if the function isn't found. For panics during
    /// execution (undefined var, type mismatch, etc.) the JIT itself
    /// panics or the extern asserts.
    pub fn call_compiled(&self, name: &str, args: &[u64]) -> u64 {
        // Resolve the symbol → Var → Fn → FuncRef. Brief sym lock.
        let sym_id = self.host.sym.lock().unwrap().intern(name);
        let var = namespace::ns_lookup(self.core_ns(), value::encode_sym_id(sym_id));
        if !value::is_ptr(var) {
            return value::NIL;
        }
        let fn_obj = namespace::var_root(var);
        if !value::is_ptr(fn_obj) {
            return value::NIL;
        }
        let fref = FuncRef::from_u32(namespace::fn_func_ref(fn_obj));

        // Install thread-locals for this call. NOTE: we deliberately
        // do NOT install the engine's shared `chain` because (a) it
        // is not Sync, and (b) pure-runtime calls into already-
        // compiled functions don't need a host-side rooting chain
        // unless the called code triggers a host-side allocation
        // (e.g. via the `cons` extern). Any function that just does
        // arithmetic and recursion (`fib`, `square`, `fact`) is safe
        // without the chain.
        let _gc_thread = self.gc.install_thread();
        let _host_g = host::install(&self.host);

        // Read lock on the JIT — many threads can hold simultaneously.
        let jit_r = self.jit.read().expect("jit RwLock poisoned");
        match self
            .gc
            .run_jit(&jit_r, fref, args, self.jit_gc_policy)
        {
            dynlower::JitOutcome::Value(v) => v,
            dynlower::JitOutcome::Void => value::NIL,
            other => panic!("unexpected JIT outcome: {other:?}"),
        }
    }

    /// Read source, compile each form, JIT it, and return the value
    /// of the last evaluated expression.
    ///
    /// Takes `&self`: multiple threads may call `eval` concurrently.
    /// The `compile` Mutex serializes the brief compile section;
    /// the JIT's `RwLock<read>` is held for both extend (which uses
    /// `&self` per the toolkit migration) and run.
    pub fn eval(&self, src: &str) -> u64 {
        // Thread-local guards.
        let _gc_thread = self.gc.install_thread();
        let _host_g = host::install(&self.host);

        // PER-CALL chain. The engine's `chain` field is reserved for
        // the bootstrap path (single-threaded engine startup). For
        // multi-threaded `eval`, each call constructs its own
        // FrameChain on the stack and registers it as a GC root for
        // the duration of the call. dynlang's `extra_root_sources`
        // is a `Mutex` (toolkit migration) so concurrent register/
        // deregister is safe.
        let local_chain = dynobj::roots::FrameChain::new();
        let chain_root: &dyn dynobj::RootSource = &local_chain;
        let _chain_root_guard =
            unsafe { self.gc.push_extra_root_source(chain_root as *const _) };
        let _chain_g = dynobj::roots::install_chain(&local_chain);

        let jit_g = self.jit.read().expect("jit RwLock poisoned");
        let jit: &JitModule = &*jit_g;

        // Read all forms upfront.
        let mut forms: Vec<u64> = Vec::new();
        {
            let mut sym = self.host.sym.lock().unwrap();
            let mut r = reader::Reader::new(src, &mut sym);
            while !r.at_eof() {
                match r.read() {
                    Ok(v) => forms.push(v),
                    Err(e) => panic!("read error: {}", e),
                }
            }
        }

        let mut pending = dynobj::roots::RootSet::new();
        for f in &forms {
            pending.add(*f);
        }
        drop(forms);
        let pending_src: &dyn dynobj::RootSource = &pending;
        let _root_guard =
            unsafe { self.gc.push_extra_root_source(pending_src as *const _) };

        let mut last = value::NIL;
        for i in 0..pending.len() {
            let form = pending.get(i);

            // Macroexpansion uses gc.run_jit for any macro Var. No
            // compile lock needed here — macros are read-only on
            // the JIT, but they may need the symbol table.
            let expanded = {
                let mut sym = self.host.sym.lock().unwrap();
                let mut ctx = expand::ExpandCtx {
                    core_ns: self.globals.get(GLOBAL_SLOT_CORE_NS),
                    sym: &mut sym,
                    gc: &self.gc,
                    jit,
                    jit_gc_policy: self.jit_gc_policy,
                    max_iters: 256,
                };
                ctx.expand_all(form)
            };

            // Compile + extend, holding the compile Mutex.
            let result = {
                let mut compile = self.compile.lock().expect("compile poisoned");
                let CompileState {
                    mb,
                    func_refs,
                    externs,
                    anon_counter,
                } = &mut *compile;

                let result = {
                    let mut sym = self.host.sym.lock().unwrap();
                    let mut compiler = compile::Compiler {
                        mb,
                        func_refs,
                        sym: &mut sym,
                        anon_counter,
                        literal_pool: jit.literal_pool(),
                    };
                    compiler.compile_top(expanded)
                };
                let snap = mb.snapshot();
                jit.extend::<
                    NanBoxConfig,
                    dynlower::Arm64Backend,
                    dynlower::regalloc::LinearScanAllocator,
                >(&snap, externs);
                result
            };

            last = match result {
                compile::TopResult::Expr(fref) => {
                    match self.gc.run_jit(jit, fref, &[], self.jit_gc_policy) {
                        dynlower::JitOutcome::Value(v) => v,
                        dynlower::JitOutcome::Void => value::NIL,
                        other => panic!("unexpected JIT outcome: {other:?}"),
                    }
                }
                compile::TopResult::Define {
                    name,
                    fref,
                    arity,
                    is_macro,
                } => {
                    let sym_id = self.host.sym.lock().unwrap().intern(&name);
                    dynobj::roots::with_scope(64, |scope| {
                        let fn_obj = namespace::alloc_fn(scope, fref.as_u32(), arity);
                        let var = namespace::ns_intern(
                            scope,
                            self.globals.get(GLOBAL_SLOT_CORE_NS),
                            value::encode_sym_id(sym_id),
                            fn_obj.get(),
                        );
                        if is_macro {
                            namespace::var_set_flag(var.get(), namespace::FLAG_MACRO);
                        }
                    });
                    value::NIL
                }
                compile::TopResult::None => value::NIL,
            };
        }
        last
    }

    pub fn print(&self, v: u64) -> String {
        // Type-id checks live behind `host::with_host`, so install the
        // engine's host before printing. (The shared FrameChain is fine
        // here — `print` doesn't allocate.)
        let _host_g = host::install(&self.host);
        let sym = self.host.sym.lock().unwrap();
        printer::print(v, &sym)
    }
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new()
    }
}
