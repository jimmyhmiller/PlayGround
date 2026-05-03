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
use dynlang::{GcConfig, NanBoxTags, ObjType};
use dynobj::roots::RootSet;
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
    pub host: Host,
    anon_counter: u32,
    /// When set, `run_source` triggers a moving collection between every
    /// pair of top-level forms. The literal pool gets walked and slots
    /// rewritten as objects move; the next form's compilation reads (and
    /// the JIT-emitted code loads) the relocated values transparently.
    ///
    /// True "GC on every cons allocation" requires safepoints inside the
    /// JIT (so the mutator can be paused mid-execution with stack maps);
    /// that's deliberately out of scope for v0. Between-form GC is the
    /// largest dose of GC pressure we can apply safely.
    gc_stress: bool,
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

        let mut engine = Engine {
            mb,
            func_refs,
            jit,
            externs,
            gc,
            host: Host {
                sym: std::cell::RefCell::new(SymbolTable::new()),
                macro_env: std::cell::RefCell::new(HashMap::new()),
                gc: gc_ptr,
            },
            anon_counter: 0,
            gc_stress: false,
        };

        // Register the JitModule's literal pool with the GC AFTER moving it
        // into Engine. The pool's address must be stable for the lifetime of
        // the engine — recording it before the move would leave a dangling
        // pointer once `jit` relocates into `Engine`.
        let pool_ptr: *const dyn dynobj::RootSource = engine.jit.literal_pool();
        unsafe { engine.gc.register_extra_root_source(pool_ptr); }

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
            host,
            anon_counter,
            gc_stress,
        } = self;

        // Both guards must be in scope: the dynlang `RUNTIME` thread-local
        // (so `NanBoxPolicy::try_decode_ptr` and ml_cons can find the GC),
        // and microlisp's `Host` (so externs see the symbol table).
        let _gc_thread = gc.install_thread();
        let _host_g = host::install(host);

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
            last = process_form_inner(form, mb, func_refs, jit, externs, gc, host, anon_counter);
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

    /// Force a garbage collection. The active SemiSpace backend copies all
    /// reachable objects to to-space; the LiteralPool's slots get rewritten
    /// in place to point at the new addresses.
    pub fn collect(&self) {
        let _gc_thread = self.gc.install_thread();
        let _g = host::install(&self.host);
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

    /// Install both the dynlang `RUNTIME` and microlisp `Host` thread-locals
    /// for the closure. Used by tests that want to allocate cons cells (via
    /// the reader, e.g.) without going through `run_source`.
    pub fn with_thread_state<R>(&self, f: impl FnOnce(&Host) -> R) -> R {
        let _gc_thread = self.gc.install_thread();
        let _g = host::install(&self.host);
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
    host: &Host,
    anon_counter: &mut u32,
) -> u64 {
    // Expand macros.
    let expanded = {
        let macro_env = host.macro_env.borrow().clone();
        let mut ctx = expand::ExpandCtx {
            host,
            macro_env,
            jit,
            gc,
            max_iters: 256,
        };
        ctx.expand_all(form)
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
        };
        compiler.compile_top(expanded)
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
        compile::TopResult::Expr(fref) => match gc.run_jit(jit, fref, &[]) {
            dynlower::JitOutcome::Value(v) => v,
            dynlower::JitOutcome::Void => value::NIL,
            other => panic!("unexpected outcome from top-level form: {other:?}"),
        },
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
