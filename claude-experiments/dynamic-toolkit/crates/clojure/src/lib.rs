//! Clojure on the dynamic-toolkit.
//!
//! See `docs/microlisp-plan.md` for the underlying incremental-JIT
//! infrastructure. This crate adapts the microlisp pattern with
//! Clojure-specific data shapes (namespaces, vars, vectors, maps).

pub mod collections;
pub mod compile;
pub mod expand;
pub mod externs;
pub mod freevars;
pub mod host;
pub mod namespace;
pub mod printer;
pub mod protocol;
pub mod quasiquote;
pub mod reader;
pub mod symbols;
pub mod types;
pub mod value;

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use dynexec::NanBoxConfig;
use dynir::ir::FuncRef;
use dynir::types::{Signature, Type};
use dynlang::closure::{ArgsListReaders, CallConv, CaptureShape, ClosureConfig, ClosureKit};
use dynlang::gc::DynGcRuntime;
use dynlang::{DynModule, GcConfig, GcPolicy, NanBoxTags};
use dynlower::{CallMode, JitModule};
use dynobj::roots::{FrameChain, RootSet};
use dynruntime::active_jit_safepoint_handler;

use crate::host::Host;
use crate::symbols::SymbolTable;
use crate::types::{Layouts, declare_types};

const JIT_CALL_TABLE_CAPACITY: usize = 64 * 1024;
const JIT_LITERAL_POOL_CAPACITY: usize = 64 * 1024;
/// Default heap size: 64 MiB. Each semispace is half this.
const HEAP_SIZE: usize = 64 * 1024 * 1024;

pub struct Engine {
    /// First-class function-value primitive. Built atop `types.fn_obj`
    /// (the existing `Fn` heap shape) using `DynModule::closures_for`.
    /// Used by every closure-related IR site to emit the allocation,
    /// body prologue, and indirect-call sequences.
    pub closures: ClosureKit,
    /// Pre-resolved FuncRefs for every runtime extern the lowering
    /// pipeline calls by name. Stable for the engine's lifetime.
    pub externs: compile::Externs,
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
    /// The `DynModule` owns the toolkit-side state (obj types, NanBox
    /// tags, GC config) and the inner `ModuleBuilder` that every
    /// compile pass mutates. Lives behind the same `Mutex` as the
    /// other compile-pass mutables; post-init reads of obj_types from
    /// outside the lock would require an accessor (today, nothing
    /// reads them post-init — the GC copies what it needs at
    /// startup).
    dyn_module: DynModule,
    func_refs: HashMap<String, compile::FnEntry>,
    /// Externs in declaration order. Mutated only at startup.
    /// Raw C function pointers for every extern declared on
    /// `dyn_module.module_builder`, in declaration order. Passed to
    /// `JitModule::extend` so the JIT can fill its call-table slots
    /// with the right pointers. Distinct from `Engine.externs`, which
    /// is the typed FuncRef registry compile-side code looks up by
    /// field name.
    extern_ptrs: Vec<*const u8>,
    anon_counter: u32,
    /// Compile-time mirror of the runtime `deftype_fields` map:
    /// `type-name sym → field-name syms in declaration order`.
    /// Populated by `(deftype* …)` and read by `(extend-type T …)`
    /// so each method body can be wrapped with implicit field
    /// bindings — `[this] (foo a)` resolves `a` to `(.-a this)`
    /// the same way Clojure does it.
    deftype_fields: HashMap<u32, Vec<u32>>,
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
        // on it. The GC's type table aliases dyn_module.obj_types.
        let gc_config = GcConfig::generational(HEAP_SIZE);
        let tags = NanBoxTags::default();
        let mut dyn_module = DynModule::new(gc_config.clone(), tags.clone());
        let types = declare_types(&mut dyn_module);
        // Resolve every field offset from the dynlang ObjType
        // registry. Single source of truth for the heap field layout —
        // see `Layouts` for the rationale.
        let layouts = Layouts::from_module(&dyn_module, &types);

        // GC runtime: needs obj_types to scan heap objects correctly.
        let gc = Box::new(DynGcRuntime::new(&gc_config, &tags, &dyn_module.obj_types));
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

        // Declare every primitive on `dyn_module`'s underlying
        // `ModuleBuilder` — the same one the toolkit auto-registered
        // `__gc_alloc__` into when the first obj_type was declared
        // above. Single FuncRef table; no parallel-builder shenanigans.
        //
        // The JIT's `extend` walks the snapshot's func_table in order,
        // increments an extern counter per `FuncDef::Extern` slot, and
        // indexes into `externs[]` with that counter. So `externs` must
        // mirror the declaration order of externs on
        // `dyn_module.module_builder`:
        //
        //   FuncRef 0 → __gc_alloc__ (auto-registered by `declare_types`
        //                via the first `obj_type` call) → gc_alloc_thunk
        //   FuncRef 1.. → Clojure prims in `all_prims()` order
        //                 → prim.ptr in matching order
        //
        // If anyone ever adds another auto-registered extern to
        // DynModule that lands between obj_type declarations and the
        // prim loop, this invariant breaks and JIT calls will jump to
        // the wrong function pointer. Guard with the assertion at the
        // end of this block.
        let mut func_refs: HashMap<String, compile::FnEntry> = HashMap::new();
        let mut extern_ptrs: Vec<*const u8> = vec![dynlang::gc::gc_alloc_thunk as *const u8];
        for prim in externs::all_prims() {
            let sig = Signature {
                params: vec![Type::I64; prim.arity],
                ret: Some(Type::I64),
            };
            let fref = dyn_module.module_builder.declare_extern(prim.name, sig);
            func_refs.insert(
                prim.name.to_string(),
                compile::FnEntry::Extern {
                    fref,
                    arity: prim.arity,
                },
            );
            extern_ptrs.push(prim.ptr);
        }

        // Invariant check: extern table on `dyn_module.module_builder`
        // matches `externs[]` in length (auto-registered __gc_alloc__ +
        // every prim). If this fails, something declared an extern
        // somewhere this code path doesn't see and the externs vec
        // ordering will be wrong at JIT time.
        debug_assert_eq!(
            extern_ptrs.len(),
            dyn_module.module_builder.func_count(),
            "extern_ptrs vec out of sync with dyn_module's FuncRef table"
        );

        // Resolve the typed extern registry once all prims are declared.
        // Used by every lowering site that calls a runtime extern.
        let externs = compile::Externs::resolve(&func_refs);

        // Build the ClosureKit on top of the already-declared `Fn`
        // type. The kit uses `dyn_module`'s auto-detected
        // `__gc_alloc__` FuncRef — no override needed now that we
        // share a single builder.
        let closures = {
            let lookup = |name: &str| match func_refs.get(name) {
                Some(compile::FnEntry::Extern { fref, .. }) => *fref,
                Some(compile::FnEntry::DefFn { fref }) => *fref,
                None => panic!("ClosureKit setup: missing extern {name:?}"),
            };
            let readers = ArgsListReaders {
                first: lookup("__reader_list_first"),
                rest: lookup("__reader_list_rest"),
                count: lookup("__reader_list_count"),
                encode_arity: |n| value::encode_int(n as i64),
                check_arity: lookup("__check_args_list"),
                no_matching_arity: lookup("__make_no_matching_arity_exception"),
                raise: lookup("__raise_exception"),
                // The kit emits `raise` via call_via_func_ref through
                // the JIT call table so the Exception outcome
                // propagates — see ArgsListReaders::raise doc. The
                // JitModule's call-table base is stable for the
                // engine's lifetime.
                call_table_base: jit.call_table_base_addr(),
                success_sentinel: value::NIL,
            };
            let config = ClosureConfig {
                captures: CaptureShape::Inline,
                call_conv: CallConv::ArgsList { readers },
                extra_fields: Vec::new(),
            };
            dyn_module.closures_for(types.fn_obj, config, "func_ref", "arity")
        };

        let mut engine = Engine {
            closures,
            externs,
            compile: Mutex::new(CompileState {
                dyn_module,
                func_refs,
                extern_ptrs,
                anon_counter: 0,
                deftype_fields: HashMap::new(),
            }),
            jit: RwLock::new(jit),
            gc,
            chain,
            globals,
            host: Host {
                sym: SymbolTable::new(),
                gc: gc_ptr,
                types,
                layouts,
                // Filled in below once we have a stable address for
                // the JitModule. Engine fields are constructed before
                // the host because the host's `jit` pointer must live
                // as long as the JitModule itself.
                jit: std::ptr::null(),
                kw_index: std::sync::Mutex::new(std::collections::HashMap::new()),
                kw_roots: Box::new(dynobj::roots::AtomicRootSet::new()),
                deftype_fields: std::sync::Mutex::new(std::collections::HashMap::new()),
                method_table: std::sync::Mutex::new(std::collections::HashMap::new()),
                method_roots: Box::new(dynobj::roots::AtomicRootSet::new()),
                // Filled in below — needs the engine's `globals`
                // address, which only becomes stable once the
                // engine value is in its final location.
                globals_ptr: std::ptr::null(),
                core_ns_slot: GLOBAL_SLOT_CORE_NS,
                // Filled in below: needs the symbol table to be
                // initialized first so we can intern the user-facing
                // names for the built-in types.
                builtin_type_names: std::sync::Mutex::new(std::collections::HashMap::new()),
                protocol_membership: std::sync::Mutex::new(std::collections::HashSet::new()),
                // Filled in by bootstrap_builtin_type_names; the
                // initial 0 placeholders are never read because that
                // bootstrap runs before the first eval call.
                seq_method_sym: 0,
                first_method_sym: 0,
                next_method_sym: 0,
                rest_method_sym: 0,
                count_method_sym: 0,
                iseq_sym: 0,
                ivector_sym: 0,
                imap_sym: 0,
                ilist_sym: 0,
            },
            jit_gc_policy: GcPolicy::OnPressure { threshold: 0.75 },
        };

        // Stash a raw pointer to the JitModule in the host so that
        // higher-order `clj_invoke_N` externs can resolve a runtime
        // FuncRef into the code pointer to call. The JitModule lives
        // inside `Box<JitModule>` whose address is stable for the
        // lifetime of the engine.
        let jit_ptr: *const JitModule = {
            let g = engine.jit.read().unwrap();
            &**g as *const JitModule
        };
        engine.host.jit = jit_ptr;

        // Register the JIT literal pool, the FrameChain, and the
        // long-lived `globals` slot array as GC root sources.
        let pool_ptr: *const dyn dynobj::RootSource = {
            let g = engine.jit.read().unwrap();
            g.literal_pool() as *const dyn dynobj::RootSource
        };
        unsafe { engine.gc.register_extra_root_source(pool_ptr) };
        let chain_ptr: *const dyn dynobj::RootSource = &*engine.chain;
        unsafe { engine.gc.register_extra_root_source(chain_ptr) };
        let globals_ptr: *const dyn dynobj::RootSource = &*engine.globals;
        unsafe { engine.gc.register_extra_root_source(globals_ptr) };
        // The keyword intern table holds GC-traced Keyword pointers.
        // It must be a root source so a moving collector rewrites the
        // interned pointers in place.
        let kw_ptr: *const dyn dynobj::RootSource = &*engine.host.kw_roots;
        unsafe { engine.gc.register_extra_root_source(kw_ptr) };
        let method_ptr: *const dyn dynobj::RootSource = &*engine.host.method_roots;
        unsafe { engine.gc.register_extra_root_source(method_ptr) };

        // Now that the Engine has settled into its final location,
        // wire host.globals_ptr to the engine's globals RootSet so
        // externs can read the current `clojure.core` namespace
        // pointer from a stable, GC-traced source.
        engine.host.globals_ptr = &*engine.globals;

        // Allocate the Registry singleton and the `clojure.core`
        // namespace, writing both into `globals` so they survive
        // future collections.
        engine.bootstrap_namespaces();

        // Map every built-in heap type to its core.clj-facing name
        // (`__ReaderList`, `__ReaderVector`, …). Without this,
        // `(extend-type __ReaderList …)` would only register methods
        // for a type that the runtime never recognizes — built-ins
        // have no per-instance type-name field, so we resolve via
        // this table at dispatch time.
        engine.bootstrap_builtin_type_names();

        engine
    }

    fn bootstrap_builtin_type_names(&mut self) {
        let sym = &self.host.sym;
        let mut tbl = self.host.builtin_type_names.lock().unwrap();
        // Names mirror the `(extend-type X …)` heads core.clj uses
        // for the corresponding heap shapes.
        let pairs: &[(usize, &str)] = &[
            (self.host.types.list.0, "__ReaderList"),
            (self.host.types.vector.0, "__ReaderVector"),
            (self.host.types.map.0, "__ReaderMap"),
            (self.host.types.set.0, "__ReaderSet"),
        ];
        for &(type_id, name) in pairs {
            let id = sym.intern(name);
            tbl.insert(type_id, id);
        }
        // Pre-intern protocol-method and protocol names so
        // `protocol::invoke_method_0` and the lock-free helpers in
        // collections / printer don't need to grab `host.sym`. The
        // expander and compiler each hold that lock when they call
        // into the seq/print machinery.
        let seq_m = sym.intern("-seq");
        let first_m = sym.intern("-first");
        let next_m = sym.intern("-next");
        let rest_m = sym.intern("-rest");
        let count_m = sym.intern("-count");
        let iseq_p = sym.intern("ISeq");
        let ivector_p = sym.intern("IVector");
        let imap_p = sym.intern("IMap");
        let ilist_p = sym.intern("IList");
        drop(tbl);
        drop(sym);
        self.host.seq_method_sym = seq_m;
        self.host.first_method_sym = first_m;
        self.host.next_method_sym = next_m;
        self.host.rest_method_sym = rest_m;
        self.host.count_method_sym = count_m;
        self.host.iseq_sym = iseq_p;
        self.host.ivector_sym = ivector_p;
        self.host.imap_sym = imap_p;
        self.host.ilist_sym = ilist_p;
    }

    /// Allocate the singleton Registry and the `clojure.core`
    /// Namespace. Both are stored in `globals` and remain alive for
    /// the lifetime of the engine.
    fn bootstrap_namespaces(&mut self) {
        let _gc_thread = self.gc.install_thread();
        let _host_g = host::install(&self.host);
        let _chain_g = dynobj::roots::install_chain(&self.chain);
        let core_name = {
            let sym = &self.host.sym;
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
        let sym_id = self.host.sym.intern(name);
        let var = namespace::ns_lookup(self.core_ns(), value::encode_sym_id(sym_id));
        if !value::is_ptr(var) {
            return value::NIL;
        }
        let fn_obj = namespace::var_root(var);
        if !value::is_ptr(fn_obj) {
            return value::NIL;
        }
        let fref = FuncRef::from_u32(namespace::fn_func_ref(fn_obj));

        // Install thread-locals for this call. Each thread needs:
        //   - the GC's per-thread MutatorThread (for allocs)
        //   - the host (so externs can reach the symbol table etc.)
        //   - a FrameChain (needed by `with_scope` which the cons
        //     extern uses to root list cells during alloc)
        let _gc_thread = self.gc.install_thread();
        let _host_g = host::install(&self.host);
        let local_chain = dynobj::roots::FrameChain::new();
        let chain_src: *const dyn dynobj::RootSource = &local_chain;
        let _chain_root_guard = unsafe { self.gc.push_extra_root_source(chain_src) };
        let _chain_g = dynobj::roots::install_chain(&local_chain);

        // User-defined fns use the unified `(self_fn, args_list)`
        // ABI. Pack args into a list; pass `fn_obj` as self_fn so
        // closures can read their captures (def-fns ignore it but
        // still accept it).
        let args_list = if args.is_empty() {
            value::NIL
        } else {
            dynobj::roots::with_scope(args.len() + 4, |scope| {
                let acc = scope.root::<value::NanBoxTag>(value::NIL);
                for &x in args.iter().rev() {
                    let new_bits = dynobj::roots::with_scope(3, |inner| {
                        value::alloc_list_cell_from_raw(inner, x, acc.get()).get()
                    });
                    acc.set(new_bits);
                }
                acc.get()
            })
        };

        // Read lock on the JIT — many threads can hold simultaneously.
        let jit_r = self.jit.read().expect("jit RwLock poisoned");
        match self
            .gc
            .run_jit(&jit_r, fref, &[fn_obj, args_list], self.jit_gc_policy)
        {
            dynlower::JitOutcome::Value(v) => v,
            dynlower::JitOutcome::Void => value::NIL,
            dynlower::JitOutcome::Exception(exc) => {
                let printed = printer::print(exc, &self.host.sym);
                panic!("Exception: {}", printed);
            }
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
        let _chain_root_guard = unsafe { self.gc.push_extra_root_source(chain_root as *const _) };
        let _chain_g = dynobj::roots::install_chain(&local_chain);

        let jit_g = self.jit.read().expect("jit RwLock poisoned");
        let jit: &JitModule = &*jit_g;

        // Read all forms upfront — but each parsed form must be
        // immediately rooted before the next read runs, otherwise an
        // intermediate allocation could relocate earlier forms and
        // leave us with stale pointers. We register `pending` as a
        // GC root source UP FRONT and `add()` each value as it comes
        // back from the reader.
        let mut pending = dynobj::roots::RootSet::new();
        let pending_src: *const dyn dynobj::RootSource = &pending;
        let _root_guard = unsafe { self.gc.push_extra_root_source(pending_src) };
        {
            let mut r = reader::Reader::new(src, &self.host.sym);
            while !r.at_eof() {
                match r.read() {
                    Ok(v) => {
                        pending.add(v);
                    }
                    Err(e) => panic!("read error: {}", e),
                }
            }
        }

        let mut last = value::NIL;
        for i in 0..pending.len() {
            let form = pending.get(i);

            // Macroexpansion uses gc.run_jit for any macro Var. We
            // pass `&Mutex<SymbolTable>` rather than holding the
            // lock for the duration: when expand JIT-calls a macro
            // body, that body may invoke externs (gensym, list_like_*,
            // …) that themselves lock `host.sym`. Holding it across
            // the JIT call deadlocks. Each lookup site locks briefly.
            let expanded = {
                let mut ctx = expand::ExpandCtx {
                    core_ns: self.globals.get(GLOBAL_SLOT_CORE_NS),
                    sym: &self.host.sym,
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
                    dyn_module,
                    func_refs,
                    extern_ptrs,
                    anon_counter,
                    deftype_fields,
                } = &mut *compile;

                let result = {
                    // Same pattern as expand: pass `&Mutex<…>`,
                    // lock briefly per access. compile_top can JIT
                    // (e.g. via macros invoked from compile-time
                    // helpers) and those JIT bodies may need the sym
                    // table; holding the lock across them deadlocks.
                    let mut compiler = compile::Compiler {
                        dyn_module,
                        func_refs,
                        externs: &self.externs,
                        sym: &self.host.sym,
                        anon_counter,
                        literal_pool: jit.literal_pool(),
                        call_table_base: jit.call_table_base_addr(),
                        loop_targets: Vec::new(),
                        last_expr_non_returning: false,
                        core_ns: self.globals.get(GLOBAL_SLOT_CORE_NS),
                        deftype_fields,
                        closures: &self.closures,
                    };
                    compiler.compile_top(expanded)
                };
                let snap = dyn_module.snapshot();
                jit.extend::<
                    NanBoxConfig,
                    dynlower::Arm64Backend,
                    dynlower::regalloc::LinearScanAllocator,
                >(&snap, extern_ptrs);
                result
            };

            last = match result {
                compile::TopResult::Expr(fref) => {
                    match self.gc.run_jit(jit, fref, &[], self.jit_gc_policy) {
                        dynlower::JitOutcome::Value(v) => v,
                        dynlower::JitOutcome::Void => value::NIL,
                        dynlower::JitOutcome::Exception(exc) => {
                            // Uncaught throw bubbled to the top.
                            // Print and panic — same surface
                            // behavior as the legacy `clj_throw`
                            // path.
                            let printed = printer::print(exc, &self.host.sym);
                            panic!("Exception: {}", printed);
                        }
                        other => panic!("unexpected JIT outcome: {other:?}"),
                    }
                }
                compile::TopResult::Define {
                    name,
                    fref,
                    arity_word,
                    is_macro,
                } => {
                    let sym_id = self.host.sym.intern(&name);
                    dynobj::roots::with_scope(64, |scope| {
                        let fn_obj = namespace::alloc_fn(scope, fref.as_u32(), arity_word as usize);
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
                compile::TopResult::DefineValue { name, value_thunk } => {
                    // Run the thunk to compute the value, then
                    // intern it as a Var in clojure.core.
                    let value = match self.gc.run_jit(jit, value_thunk, &[], self.jit_gc_policy) {
                        dynlower::JitOutcome::Value(v) => v,
                        dynlower::JitOutcome::Void => value::NIL,
                        dynlower::JitOutcome::Exception(exc) => {
                            let printed = printer::print(exc, &self.host.sym);
                            panic!("Exception: {}", printed);
                        }
                        other => panic!("unexpected JIT outcome: {other:?}"),
                    };
                    let sym_id = self.host.sym.intern(&name);
                    dynobj::roots::with_scope(16, |scope| {
                        let _ = namespace::ns_intern(
                            scope,
                            self.globals.get(GLOBAL_SLOT_CORE_NS),
                            value::encode_sym_id(sym_id),
                            value,
                        );
                    });
                    value::NIL
                }
                compile::TopResult::None => value::NIL,
            };
        }
        last
    }

    pub fn print(&self, v: u64) -> String {
        // Install the same per-thread context the eval driver does:
        // GC thread, host, FrameChain. The printer now JIT-calls
        // `-seq`/`-first`/`-next` to walk records that satisfy ISeq,
        // which allocates list cells via `dynobj::roots::with_scope`
        // and runs JIT code (so the safepoint session needs gc.run_jit
        // to set it up).
        let _gc_thread = self.gc.install_thread();
        let _host_g = host::install(&self.host);
        let _chain_g = dynobj::roots::install_chain(&self.chain);
        let sym = &self.host.sym;
        printer::print(v, &sym)
    }
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new()
    }
}
