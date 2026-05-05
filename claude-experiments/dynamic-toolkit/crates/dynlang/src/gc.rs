//! The single standard facade for GC integration.
//!
//! [`DynGcRuntime`] owns the allocator and is the only supported way to
//! execute a `DynModule`. It automatically binds `__gc_alloc__`, installs
//! the correct `PtrPolicy`, and (for the generational backend) manages
//! the JIT safepoint session.
//!
//! Language implementors should never construct `Heap`, `SemiSpace`,
//! `BumpAllocator`, `JitSafepointSession`, or a `PtrPolicy` themselves —
//! those are implementation details of this facade.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use dynalloc::{BumpAllocator, Heap, PtrPolicy, alloc_obj};
use dynir::interp::ExternCallResult;
use dynir::{FuncDef, FuncRef, Module};
use dynexec::{CodegenConfig, FrameStrategy};
use dynlower::{JitModule, JitOutcome, backend::LoweringBackend, regalloc::RegisterAllocator};
use dynobj::roots::{Rooted, RootScope, RootSource};
use dynobj::{Compact, TypeInfo};
use dynruntime::{GcPolicy, JitSafepointSession, StackMapJitTransport, active_jit_safepoint_handler};

use crate::{GcConfig, NanBoxTags, ObjType};

// ── NanBox PtrPolicy ──────────────────────────────────────────────
//
// `PtrPolicy` is a trait with only associated functions, so it has no
// per-instance data. We stash the ptr tag in a thread-local for the
// duration of a run. `DynGcRuntime::install_thread` sets it; all callers
// go through `DynGcRuntime::run_jit` or the interp helpers, which call
// `install_thread` themselves.

const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
const TAG_FIELD_MASK: u64 = 0x0003_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

thread_local! {
    static PTR_TAG: Cell<u32> = const { Cell::new(u32::MAX) };
    static RUNTIME: Cell<*const DynGcRuntime> = const { Cell::new(std::ptr::null()) };
}

/// `PtrPolicy` that reads the current thread's ptr tag from a thread-local
/// installed by [`DynGcRuntime`]. User code never names this type.
pub struct NanBoxPolicy;

impl PtrPolicy for NanBoxPolicy {
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        let tag = PTR_TAG.with(|c| c.get());
        debug_assert_ne!(tag, u32::MAX, "NanBoxPolicy used without DynGcRuntime::install_thread");
        let expected = TAG_PATTERN | ((tag as u64) << 48);
        if (bits & (FULL_MASK | TAG_FIELD_MASK)) != expected {
            return None;
        }
        let payload = bits & PAYLOAD_MASK;
        if payload == 0 { None } else { Some(payload as *mut u8) }
    }

    fn encode_ptr(ptr: *mut u8) -> u64 {
        let tag = PTR_TAG.with(|c| c.get());
        debug_assert_ne!(tag, u32::MAX, "NanBoxPolicy used without DynGcRuntime::install_thread");
        TAG_PATTERN | ((tag as u64) << 48) | ((ptr as u64) & PAYLOAD_MASK)
    }
}

// ── Root Set (for interp use) ─────────────────────────────────────

/// Simple root set used by the interpreter-side entry points. The JIT
/// path uses stack-map roots instead and doesn't touch this.
pub struct RootSet {
    slots: RefCell<Vec<Cell<u64>>>,
}

impl RootSet {
    pub fn new() -> Self { RootSet { slots: RefCell::new(Vec::new()) } }
    pub fn add(&self, val: u64) -> usize {
        let mut s = self.slots.borrow_mut();
        let idx = s.len();
        s.push(Cell::new(val));
        idx
    }
    pub fn set(&self, idx: usize, val: u64) { self.slots.borrow()[idx].set(val); }
    pub fn get(&self, idx: usize) -> u64 { self.slots.borrow()[idx].get() }
    pub fn clear(&self) { self.slots.borrow_mut().clear(); }
}

impl Default for RootSet { fn default() -> Self { Self::new() } }

impl RootSource for RootSet {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for cell in self.slots.borrow().iter() {
            visitor(cell.as_ptr());
        }
    }
}

// ── Backend ───────────────────────────────────────────────────────

enum Backend {
    /// Bump allocator, never collects. Zero overhead.
    Leak(BumpAllocator),
    /// Generational, thread-aware, concurrent-capable `Heap`.
    Generational(Heap),
}

// ── DynGcRuntime ──────────────────────────────────────────────────

/// The single standard GC facade. Created from a built module; owns the
/// allocator and provides the only supported execution entry points.
pub struct DynGcRuntime {
    backend: Backend,
    type_infos: Vec<TypeInfo>,
    tags: NanBoxTags,
    roots: RootSet,
    /// Externs to auto-bind at JIT time, in addition to the built-in
    /// `__gc_alloc__` and `__dynlang_prop_slow__`. Populated via
    /// [`set_auto_externs`](Self::set_auto_externs); typically the
    /// `DynModule::auto_externs` map verbatim.
    auto_externs: HashMap<String, *const u8>,
    /// Additional root sources walked at every collection (whether triggered
    /// by `collect()` or by a JIT safepoint). Each entry is a raw pointer; the
    /// caller is responsible for keeping the pointee alive at least as long
    /// as the runtime lives. Typical use: register a `JitModule`'s literal
    /// pool so quote-literals are traced by the moving GC.
    extra_root_sources: RefCell<Vec<*const dyn RootSource>>,
}

/// The canonical name of dynlang's GC-allocation extern. `compile_jit`
/// recognizes it and routes it to the internal thunk automatically.
pub const GC_ALLOC_EXTERN: &str = "__gc_alloc__";

impl DynGcRuntime {
    /// Create a runtime for a built module.
    pub fn new(
        config: &GcConfig,
        tags: &NanBoxTags,
        obj_types: &[ObjType],
    ) -> Self {
        let type_infos: Vec<TypeInfo> = obj_types.iter().map(|t| *t.type_info).collect();

        let backend = match config {
            GcConfig::Leak => Backend::Leak(BumpAllocator::new::<Compact>(64 * 1024 * 1024)),
            GcConfig::Generational { heap_size, nursery_size } => {
                let heap = match nursery_size {
                    Some(nsize) => Heap::new_generational::<Compact>(*nsize, *heap_size, type_infos.clone()),
                    None => Heap::new::<Compact>(*heap_size, type_infos.clone()),
                };
                Backend::Generational(heap)
            }
        };

        DynGcRuntime {
            backend,
            type_infos,
            tags: tags.clone(),
            roots: RootSet::new(),
            auto_externs: HashMap::new(),
            extra_root_sources: RefCell::new(Vec::new()),
        }
    }

    /// Register an additional `RootSource` to be scanned at every GC.
    ///
    /// Permanent registration — the source stays in the root list until
    /// the runtime is dropped. For scoped (LIFO) registration, use
    /// [`push_extra_root_source`](Self::push_extra_root_source).
    ///
    /// # Safety
    /// `source` must remain valid for the lifetime of the runtime. The
    /// runtime stores it as a raw pointer; the caller is responsible for
    /// keeping the pointee alive (e.g., behind a `Box` or other stable
    /// allocation).
    pub unsafe fn register_extra_root_source(&self, source: *const dyn RootSource) {
        self.extra_root_sources.borrow_mut().push(source);
    }

    /// Push a temporary `RootSource` and return a guard that pops it on
    /// drop. Use this for stack-scoped roots — for example, a list of
    /// pending top-level forms that the compiler hasn't reached yet.
    ///
    /// Pops in LIFO order: each guard's drop pops the most-recently-pushed
    /// extra root. Drop a guard out of order at your peril.
    ///
    /// # Safety
    /// `source` must remain valid for the lifetime of the returned guard.
    pub unsafe fn push_extra_root_source<'a>(
        &'a self,
        source: *const (dyn RootSource + 'a),
    ) -> ExtraRootGuard<'a> {
        // `'a` is the lifetime of the borrow; the cast erases it because
        // `extra_root_sources` stores `'static` raw pointers internally.
        // Soundness is the caller's responsibility (see Safety).
        let erased: *const dyn RootSource = unsafe {
            core::mem::transmute::<*const (dyn RootSource + 'a), *const dyn RootSource>(source)
        };
        self.extra_root_sources.borrow_mut().push(erased);
        ExtraRootGuard { gc: self }
    }

    /// Install the auto-bound extern map (typically
    /// `DynModule::auto_externs`). At JIT time, `compile_jit` consults
    /// this after the reserved names (`__gc_alloc__`,
    /// `__dynlang_prop_slow__`) and before the embedder-supplied
    /// resolver. Replaces any previous map.
    pub fn set_auto_externs(&mut self, map: HashMap<String, *const u8>) {
        self.auto_externs = map;
    }

    pub fn tags(&self) -> &NanBoxTags { &self.tags }
    pub fn roots(&self) -> &RootSet { &self.roots }
    pub fn type_info(&self, type_id: usize) -> &TypeInfo { &self.type_infos[type_id] }
    pub fn type_count(&self) -> usize { self.type_infos.len() }

    /// Install this runtime as the current thread's active runtime. All
    /// `__gc_alloc__` thunks and `NanBoxPolicy` lookups read this. The
    /// returned guard restores the previous state on drop.
    pub fn install_thread(&self) -> ThreadGuard<'_> {
        let prev_rt = RUNTIME.with(|c| c.replace(self as *const _));
        let prev_tag = PTR_TAG.with(|c| c.replace(self.tags.ptr));
        ThreadGuard { prev_rt, prev_tag, _phantom: std::marker::PhantomData }
    }

    /// Allocate an object. Returns a raw, untagged heap pointer.
    ///
    /// **Prefer [`alloc_in`](Self::alloc_in)** — it returns a `Rooted` handle
    /// scoped to a `RootScope`, which is required for safety any time the
    /// caller might invoke another allocation or JIT call before letting the
    /// pointer go. This raw form is provided for the IR `gc_alloc` thunk
    /// (called from JIT code, where stack-map roots cover the caller's
    /// frame) and for low-level toolkit code that knows what it's doing.
    ///
    /// # Safety / contract
    /// The returned pointer is a GC root only for as long as a caller's
    /// stack map / `Rooted` keeps it alive. Crossing any allocation,
    /// safepoint, or JIT call without that protection corrupts under a
    /// moving GC.
    pub fn alloc(&self, type_id: usize, varlen_len: usize) -> *mut u8 {
        let info = &self.type_infos[type_id];
        match &self.backend {
            Backend::Leak(bump) => unsafe { alloc_obj::<Compact>(bump, info, varlen_len) },
            Backend::Generational(heap) => heap.alloc_obj::<Compact>(info, varlen_len),
        }
    }

    /// Allocate and immediately root in `scope`.
    ///
    /// This is the safe public allocator. The returned `Rooted` borrows a
    /// slot from `scope`, so the GC tracks the freshly-allocated pointer
    /// until the scope drops — even across additional allocations or JIT
    /// calls inside the same scope.
    ///
    /// `T` is a phantom type tag (typically a frontend's value type, like
    /// `NanBox`). It plays no runtime role today; future `Rootable` work
    /// can give it `from_bits` / `to_bits` semantics.
    pub fn alloc_in<'a, T>(
        &self,
        scope: &'a RootScope<'_>,
        type_id: usize,
        varlen_len: usize,
    ) -> Rooted<'a, T> {
        let raw = self.alloc(type_id, varlen_len);
        assert!(
            !raw.is_null(),
            "DynGcRuntime::alloc_in: allocator returned null \
             (heap exhausted? — collection should have been triggered \
             by the JIT-side safepoint preceding this call)"
        );
        let tagged = self.tag_ptr(raw);
        scope.root::<T>(tagged)
    }

    /// Encode a raw pointer as a NanBox ptr-tagged value.
    ///
    /// Unsafe to use directly under a moving GC: a tagged value held in an
    /// unrooted `u64` is invalidated on the next collection. Prefer
    /// [`alloc_in`](Self::alloc_in), which returns a `Rooted`.
    pub fn tag_ptr(&self, ptr: *mut u8) -> u64 {
        if ptr.is_null() { return 0; }
        TAG_PATTERN | ((self.tags.ptr as u64) << 48) | ((ptr as u64) & PAYLOAD_MASK)
    }

    /// Number of collections performed by the underlying heap so far.
    /// Always 0 for `Leak`. Useful for tests and diagnostics.
    pub fn collection_count(&self) -> usize {
        match &self.backend {
            Backend::Leak(_) => 0,
            Backend::Generational(heap) => heap.collections(),
        }
    }

    /// Force a collection (no-op for Leak).
    pub fn collect(&self) {
        match &self.backend {
            Backend::Leak(_) => {}
            Backend::Generational(heap) => {
                let _guard = self.install_thread();
                let extras = self.extra_root_sources.borrow();
                let mut sources: Vec<&dyn RootSource> = Vec::with_capacity(1 + extras.len());
                sources.push(&self.roots);
                for &p in extras.iter() {
                    sources.push(unsafe { &*p });
                }
                unsafe { heap.collect::<NanBoxPolicy>(&sources); }
            }
        }
    }

    // ── JIT path ────────────────────────────────────────────────

    /// Compile a module with JIT, auto-binding `__gc_alloc__` and the
    /// safepoint handler. The caller supplies externs for non-GC
    /// externs only.
    ///
    /// Panics if the module declares an extern that isn't `__gc_alloc__`
    /// and isn't resolved by `user_extern_for`.
    pub fn compile_jit<Cfg, B, R>(
        &self,
        module: &Module,
        user_extern_for: impl Fn(&str) -> Option<*const u8>,
    ) -> JitModule
    where
        Cfg: CodegenConfig,
        B: LoweringBackend,
        R: RegisterAllocator,
        Cfg::Frames: FrameStrategy<Cfg::Layout, Cfg::Roots, Cfg::CallingConvention>,
    {
        let externs = self.build_extern_table(module, user_extern_for);
        let safepoint_handler = match &self.backend {
            Backend::Generational(_) => Some(active_jit_safepoint_handler as u64),
            _ => None,
        };
        JitModule::compile_with_regalloc::<Cfg, B, R>(module, &externs, safepoint_handler)
    }

    /// Run a JIT function with the correct safepoint session and thread
    /// state installed. This is the only supported way to execute a
    /// `JitModule` produced from a `DynModule` — call sites must not
    /// construct `JitSafepointSession` themselves.
    ///
    /// `policy` controls when the safepoint handler runs a collection.
    /// Pass [`GcPolicy::OnPressure { threshold: 0.75 }`] for typical
    /// production behavior; [`GcPolicy::EveryPoint`] for GC-stress
    /// testing; [`GcPolicy::NeverAuto`] when the frontend manages
    /// collection cadence itself (e.g. between top-level forms).
    ///
    /// The module's `literal_pool` is registered as an extra root source so
    /// quote-style GC literals embedded as `Inst::GcLiteral(idx)` are traced
    /// (and updated in place if the GC moves the referenced object).
    pub fn run_jit(
        &self,
        jit: &JitModule,
        entry: FuncRef,
        args: &[u64],
        policy: GcPolicy,
    ) -> JitOutcome {
        let _thread = self.install_thread();
        match &self.backend {
            Backend::Generational(heap) => {
                let safepoints = jit.all_safepoints();
                let session = JitSafepointSession::<NanBoxPolicy, _>::new(
                    heap, StackMapJitTransport, &safepoints,
                )
                .with_gc_policy(policy);
                // Safety: `jit` outlives this call, so `&jit.literal_pool()`
                // is valid for the session's lifetime.
                let pool_ptr: *const dyn RootSource = jit.literal_pool();
                unsafe { session.register_extra_root(pool_ptr); }
                // Mirror the runtime's permanent extra root sources onto
                // the session — those are typically per-thread frame
                // chains and similar host-side root sets that must also
                // be scanned at every JIT-time safepoint, not just at
                // explicit `gc.collect()` calls. Without this, host code
                // that holds GC handles in `Rooted` slots would silently
                // see them collected when the JIT triggers a GC.
                let extras = self.extra_root_sources.borrow();
                for &ptr in extras.iter() {
                    unsafe { session.register_extra_root(ptr); }
                }
                drop(extras);
                session.with_installed(|| jit.call_outcome(entry, args))
            }
            _ => jit.call_outcome(entry, args),
        }
    }

    // ── Interpreter extern binder ───────────────────────────────

    /// Returns a closure suitable for `ModuleInterpreter::bind_by_name("__gc_alloc__", ...)`.
    /// The interpreter path currently requires the user to bind it because
    /// the interpreter owns the extern vec; future work can hide this.
    pub fn interp_gc_alloc(&self) -> impl Fn(&[u64]) -> ExternCallResult + '_ {
        move |args: &[u64]| {
            let type_id = args[0] as usize;
            let varlen_len = args[1] as usize;
            let ptr = self.alloc(type_id, varlen_len);
            assert!(!ptr.is_null(), "dynlang: gc_alloc returned null");
            ExternCallResult::Value(Some(ptr as u64))
        }
    }

    fn build_extern_table(
        &self,
        module: &Module,
        user_extern_for: impl Fn(&str) -> Option<*const u8>,
    ) -> Vec<*const u8> {
        module
            .func_table
            .iter()
            .filter_map(|def| match def {
                FuncDef::Extern(ef) => {
                    if ef.name == GC_ALLOC_EXTERN {
                        Some(gc_alloc_thunk as *const u8)
                    } else if ef.name == crate::ic::PROP_SLOW_EXTERN {
                        Some(crate::ic::prop_slow_thunk as *const u8)
                    } else if let Some(ptr) = self.auto_externs.get(&ef.name) {
                        Some(*ptr)
                    } else {
                        Some(user_extern_for(&ef.name).unwrap_or_else(|| {
                            panic!(
                                "dynlang: unresolved extern `{}` — DynGcRuntime::compile_jit's \
                                 user_extern_for returned None and it isn't in auto_externs",
                                ef.name
                            )
                        }))
                    }
                }
                FuncDef::Internal(_) => None,
            })
            .collect()
    }
}

/// Guard returned by `install_thread`. On drop, restores the previous
/// RAII guard that pops the most-recently-pushed extra root source on drop.
/// Returned by [`DynGcRuntime::push_extra_root_source`].
pub struct ExtraRootGuard<'a> {
    gc: &'a DynGcRuntime,
}

impl Drop for ExtraRootGuard<'_> {
    fn drop(&mut self) {
        let popped = self.gc.extra_root_sources.borrow_mut().pop();
        debug_assert!(popped.is_some(), "ExtraRootGuard: pop on empty stack");
    }
}

/// thread-local runtime + ptr tag.
pub struct ThreadGuard<'a> {
    prev_rt: *const DynGcRuntime,
    prev_tag: u32,
    _phantom: std::marker::PhantomData<&'a DynGcRuntime>,
}

impl Drop for ThreadGuard<'_> {
    fn drop(&mut self) {
        RUNTIME.with(|c| c.set(self.prev_rt));
        PTR_TAG.with(|c| c.set(self.prev_tag));
    }
}

// ── Internal thunk called by JIT code ─────────────────────────────

/// The extern function the JIT calls for `__gc_alloc__`. Reads the
/// installed runtime from thread-local storage.
extern "C" fn gc_alloc_thunk(type_id: u64, varlen_len: u64) -> u64 {
    let rt_ptr = RUNTIME.with(|c| c.get());
    assert!(!rt_ptr.is_null(), "dynlang: __gc_alloc__ called without DynGcRuntime installed");
    let rt = unsafe { &*rt_ptr };
    let ptr = rt.alloc(type_id as usize, varlen_len as usize);
    assert!(!ptr.is_null(), "dynlang: gc_alloc returned null (OOM after GC)");
    ptr as u64
}
