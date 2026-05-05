//! Microlisp: a tiny Lisp built on the dynamic-toolkit incremental JIT.
//!
//! Demonstrates same-image macros — macro bodies are JIT-compiled functions
//! living in the same `JitModule` as the runtime code that calls them.

pub mod compile;
pub mod expand;
pub mod host;
pub mod printer;
pub mod prims;
pub mod reader;
pub mod symbols;
pub mod value;

use std::collections::HashMap;

use dynexec::NanBoxConfig;
use dynir::builder::ModuleBuilder;
use dynir::ir::FuncRef;
use dynlang::gc::DynGcRuntime;
use dynlang::{GcConfig, GcPolicy, NanBoxTags, ObjType};
use dynobj::roots::{FrameChain, RootScope, RootSet};
use dynlower::{CallMode, JitModule};
use dynobj::{Compact, ObjHeader, TypeInfo, VarLenKind};
use dynruntime::active_jit_safepoint_handler;

use crate::host::{Host, HostGuard};
use crate::symbols::SymbolTable;

pub struct Engine {
    mb: ModuleBuilder,
    func_refs: HashMap<String, FuncRef>,
    /// Boxed for a stable address so the GC can hold a pointer to
    /// `jit.literal_pool` as a root source across moves of `Engine`.
    jit: Box<JitModule>,
    /// Externs in declaration order. Grown by `declare_primitives` once at startup.
    externs: Vec<*const u8>,
    /// GC runtime owning the heap that cons cells live on.
    /// Boxed so the address is stable — the `Host` carries a raw pointer to it.
    gc: Box<DynGcRuntime>,
    /// Per-thread shadow stack. Every Rust function that holds a NanBox
    /// cons handle across a GC point must push a `RootScope` onto this
    /// chain. Boxed for a stable address (the GC holds a pointer to it
    /// via `register_extra_root_source`).
    chain: Box<FrameChain>,
    pub host: Host,
    anon_counter: u32,
    /// When set, `run_source` triggers a moving collection between every
    /// pair of top-level forms. The literal pool gets walked and slots
    /// rewritten as objects move; the next form's compilation reads (and
    /// the JIT-emitted code loads) the relocated values transparently.
    gc_stress: bool,
    /// GC policy passed to `gc.run_jit` for every JIT invocation
    /// (top-level form evaluation and macro expansion). Default is
    /// `OnPressure { threshold: 0.75 }`. Set to `EveryPoint` to force a
    /// moving collection at every JIT safepoint — the strongest stress
    /// test of root coverage; if any safepoint's stack-map omits a live
    /// value, this mode will corrupt or panic.
    jit_gc_policy: GcPolicy,
}

const JIT_CALL_TABLE_CAPACITY: usize = 64 * 1024;
const JIT_LITERAL_POOL_CAPACITY: usize = 64 * 1024;
/// Default heap size for microlisp's SemiSpace allocator (4 MiB). Each
/// space is half this; collection runs when the from-space fills.
const HEAP_SIZE: usize = 4 * 1024 * 1024;

/// Build the `cons` ObjType. Two GC-traced value fields (car, cdr) behind a
/// `Compact` header. Returned with `type_id = 0` so it matches
/// `value::CONS_TYPE_ID`.
fn cons_obj_type() -> ObjType {
    let info = TypeInfo::for_header(Compact::SIZE)
        .with_fields(2)
        .with_type_id(0);
    let info_static: &'static TypeInfo = Box::leak(Box::new(info));
    ObjType {
        name: "cons".into(),
        type_info: info_static,
        field_offsets: HashMap::new(),
        varlen: VarLenKind::None,
    }
}

impl Engine {
    pub fn new() -> Self {
        let mut mb = ModuleBuilder::new();
        let mut func_refs: HashMap<String, FuncRef> = HashMap::new();
        let externs = compile::declare_primitives(&mut mb, &mut func_refs);

        // GC: Generational moving collector. Cons cells live here.
        // tags.ptr = 2 (TAG_CONS) — the only pointer-tagged value class.
        // Generational is the only safepoint-aware backend; pairing it with
        // `CallMode::ControlAware { safepoint_handler }` and emitting
        // `Inst::Safepoint` at every allocator call is the supported
        // configuration. `Module::validate_safepoints` enforces this at
        // every `JitModule::extend`.
        let obj_types = vec![cons_obj_type()];
        let gc_config = GcConfig::generational(HEAP_SIZE);
        let tags = NanBoxTags { nil: 0, bool_tag: 0, ptr: 2 };
        let gc = Box::new(DynGcRuntime::new(&gc_config, &tags, &obj_types));
        let gc_ptr: *const DynGcRuntime = &*gc;

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

        let engine = Engine {
            mb,
            func_refs,
            jit,
            externs,
            gc,
            chain,
            host: Host {
                sym: std::cell::RefCell::new(SymbolTable::new()),
                macro_env: std::cell::RefCell::new(HashMap::new()),
                gc: gc_ptr,
            },
            anon_counter: 0,
            gc_stress: false,
            jit_gc_policy: GcPolicy::OnPressure { threshold: 0.75 },
        };

        // Register the JitModule's literal pool AND the host-side
        // FrameChain with the GC, AFTER moving them into Engine. Both
        // need stable addresses for the lifetime of the engine —
        // recording before the move would leave a dangling pointer.
        let pool_ptr: *const dyn dynobj::RootSource = engine.jit.literal_pool();
        unsafe { engine.gc.register_extra_root_source(pool_ptr); }
        let chain_ptr: *const dyn dynobj::RootSource = &*engine.chain;
        unsafe { engine.gc.register_extra_root_source(chain_ptr); }

        engine
    }

    /// Read every form from `src` and process them in order, returning the
    /// last form's result (or `nil` if the source had no expression forms).
    ///
    /// If `gc_stress` is set, a moving collection runs between every pair
    /// of top-level forms. The final form's result is **not** followed by
    /// a collection — that would invalidate any heap-pointer it returns.
    ///
    /// The full list of pending forms is registered as an extra GC root
    /// for the duration of this call. Without that, a between-form
    /// collection would invalidate the cons-tree pointers of forms that
    /// haven't been processed yet — every quoted list and every
    /// macroexpansion result lives on the GC heap.
    pub fn run_source(&mut self, src: &str) -> u64 {
        // Split-borrow `self` so the immutable borrows held by the thread
        // guards don't block the `&mut` we need on the other fields.
        let Engine {
            mb,
            func_refs,
            jit,
            externs,
            gc,
            chain,
            host,
            anon_counter,
            gc_stress,
            jit_gc_policy,
        } = self;

        // Three thread-local guards must all be in scope:
        //  - dynlang `RUNTIME` (so `NanBoxPolicy::try_decode_ptr` and
        //    `ml_cons` can find the GC).
        //  - microlisp `Host` (so externs see the symbol table).
        //  - dynobj `ACTIVE_CHAIN` (so `with_scope` finds our shadow
        //    stack — the GC scans this chain at every safepoint).
        let _gc_thread = gc.install_thread();
        let _host_g = host::install(host);
        let _chain_g = dynobj::roots::install_chain(chain);

        // Read all forms upfront. Each form may be a heap-allocated cons
        // tree; the slice's u64 bits are GC pointers we MUST keep visible
        // to the collector for the rest of this call.
        let forms = read_forms_in(src, host);
        let mut pending = RootSet::new();
        for f in &forms {
            pending.add(*f);
        }
        // Drop the temporary `forms` Vec — its values are now mirrored in
        // `pending`'s `Cell<u64>` slots, which the GC will trace and
        // rewrite in place.
        drop(forms);

        // Scope the root registration: the guard pops from the GC's
        // extra-root stack when this function returns.
        let pending_src: &dyn dynobj::RootSource = &pending;
        let _root_guard = unsafe { gc.push_extra_root_source(pending_src as *const _) };

        let n_forms = pending.len();
        let mut last = value::NIL;
        for i in 0..n_forms {
            // Re-read each iteration: a previous between-form collection
            // may have rewritten this slot to a relocated address.
            let form = pending.get(i);
            last = process_form_inner(form, mb, func_refs, jit, externs, gc, chain, host, anon_counter, *jit_gc_policy);
            if *gc_stress && i + 1 < n_forms {
                gc.collect();
            }
        }
        last
    }

    /// Enable / disable stress-GC mode. When enabled, `run_source` runs a
    /// full moving collection between every pair of top-level forms.
    pub fn set_gc_stress(&mut self, stress: bool) {
        self.gc_stress = stress;
    }

    /// Whether stress-GC mode is currently active.
    pub fn gc_stress(&self) -> bool {
        self.gc_stress
    }

    /// Override the GC policy used when the JIT runs (top-level form
    /// evaluation and macro-body invocation). The default is
    /// `OnPressure { threshold: 0.75 }`. Set to `EveryPoint` to validate
    /// JIT root coverage by collecting at every safepoint.
    pub fn set_jit_gc_policy(&mut self, policy: GcPolicy) {
        self.jit_gc_policy = policy;
    }

    /// Force a garbage collection. The active backend copies all
    /// reachable objects to to-space; the LiteralPool's slots and the
    /// host-side FrameChain's slots get rewritten in place to point at
    /// the new addresses.
    pub fn collect(&self) {
        let _gc_thread = self.gc.install_thread();
        let _g = host::install(&self.host);
        let _chain_g = dynobj::roots::install_chain(&self.chain);
        self.gc.collect();
    }

    /// Direct access to the GC runtime.
    pub fn gc(&self) -> &DynGcRuntime {
        &self.gc
    }

    pub fn read_forms(&self, src: &str) -> Vec<u64> {
        read_forms_in(src, &self.host)
    }

    /// Print a value using the live symbol table.
    pub fn print(&self, v: u64) -> String {
        let sym = self.host.sym.borrow();
        printer::print(v, &sym)
    }

    /// Test-only: access the JitModule's literal pool. Used by tests that
    /// want to simulate a moving GC by rewriting pool slots and observe that
    /// emitted code reads the new value.
    pub fn literal_pool(&self) -> &dynlower::LiteralPool {
        self.jit.literal_pool()
    }

    /// Convenience: install host for the duration of a closure (e.g., for
    /// tests that want to call externs via raw FFI without going through
    /// `run_source`).
    pub fn with_host_installed<R>(&self, f: impl FnOnce(&Engine) -> R) -> R {
        let _g = host::install(&self.host);
        f(self)
    }

    /// Install the full thread-local state needed to allocate or run
    /// JIT code through this engine: the dynlang `RUNTIME`, microlisp's
    /// `Host`, and dynobj's active `FrameChain`. Used by tests that want
    /// to allocate cons cells (via the reader, e.g.) without going
    /// through `run_source`.
    pub fn with_thread_state<R>(&self, f: impl FnOnce(&Host) -> R) -> R {
        let _gc_thread = self.gc.install_thread();
        let _g = host::install(&self.host);
        let _chain_g = dynobj::roots::install_chain(&self.chain);
        f(&self.host)
    }
}

impl Default for Engine {
    fn default() -> Self { Self::new() }
}

// Convenience: install a HostGuard while running a function on `self`.
fn _silence_unused_guard(_: HostGuard) {}

/// Process one already-read form. Same semantics as the old method, but
/// takes explicit field references so callers can split-borrow `Engine`.
fn process_form_inner(
    form: u64,
    mb: &mut ModuleBuilder,
    func_refs: &mut HashMap<String, FuncRef>,
    jit: &mut JitModule,
    externs: &[*const u8],
    gc: &DynGcRuntime,
    chain: &FrameChain,
    host: &Host,
    anon_counter: &mut u32,
    jit_gc_policy: GcPolicy,
) -> u64 {
    // Top-level rooting scope. Holds the in-flight `form` and `expanded`
    // values across expansion (which fires JIT calls and may collect)
    // and across compile / extend (which may also allocate cons cells
    // for the literal pool, intermediate trees, etc.).
    let scope = RootScope::new(chain, 4);
    let form_root = scope.root::<crate::value::NanBoxTag>(form);

    // Expand macros.
    let expanded_root = {
        let macro_env = host.macro_env.borrow().clone();
        let mut ctx = expand::ExpandCtx {
            host,
            macro_env,
            jit,
            gc,
            max_iters: 256,
            jit_gc_policy,
        };
        let result_bits = ctx.expand_all(form_root.get());
        scope.root::<crate::value::NanBoxTag>(result_bits)
    };

    // Compile.
    let result = {
        let mut sym = host.sym.borrow_mut();
        let lit_pool = jit.literal_pool();
        let mut compiler = compile::Compiler {
            mb,
            func_refs,
            sym: &mut sym,
            anon_counter,
            literal_pool: lit_pool,
            live_stack: Vec::new(),
        };
        compiler.compile_top(expanded_root.get())
    };

    // Snapshot + extend the JIT, with safepoint validation.
    let snap = mb.snapshot();
    let allocator_frefs = compile::allocator_frefs(func_refs);
    snap.validate_safepoints(&allocator_frefs);
    jit.extend::<
        NanBoxConfig,
        dynlower::Arm64Backend,
        dynlower::regalloc::LinearScanAllocator,
    >(&snap, externs);

    match result {
        compile::TopResult::Define { .. } => value::NIL,
        compile::TopResult::Defmacro { name, fref } => {
            let mut sym = host.sym.borrow_mut();
            let id = sym.intern(&name);
            drop(sym);
            host.macro_env.borrow_mut().insert(id, fref);
            value::NIL
        }
        compile::TopResult::Expr(fref) => {
            match gc.run_jit(jit, fref, &[], jit_gc_policy) {
                dynlower::JitOutcome::Value(v) => v,
                dynlower::JitOutcome::Void => value::NIL,
                other => panic!("unexpected outcome from top-level form: {other:?}"),
            }
        }
        compile::TopResult::None => value::NIL,
    }
}

/// Read every form from `src` against the given Host's symbol table.
/// Free function so callers can use it without holding `&mut Engine`.
fn read_forms_in(src: &str, host: &Host) -> Vec<u64> {
    let mut sym = host.sym.borrow_mut();
    let mut r = reader::Reader::new(src, &mut sym);
    let mut forms = Vec::new();
    while !r.at_eof() {
        match r.read() {
            Ok(v) => forms.push(v),
            Err(e) => panic!("read error: {}", e),
        }
    }
    forms
}
