//! Property-access inline cache scaffolding.
//!
//! Built on `dynsym`'s `SymbolTable` / `DispatchTable` / `InlineCacheArray`.
//! Hides the bits every frontend would otherwise rebuild:
//!
//! - The class-key encoding (`u16 type_id + 1` so `0` stays free as
//!   `InlineCacheEntry::EMPTY`'s sentinel).
//! - A stable IC base address that survives "we don't know how many sites
//!   exist until lowering finishes". An indirection cell whose *address*
//!   is baked into IR; the array pointer it holds is filled at finalize.
//! - The slow-path extern declaration, the JIT thunk, and the
//!   forwarding-pointer chase that every embedder writes themselves.
//!
//! ## Usage
//!
//! ```ignore
//! let mut ic = PropertyIc::new(&mut dyn_module);
//!
//! // Register dispatch tables for each struct/object type:
//! for ty in &dyn_module.obj_types {
//!     ic.register_type(ty);
//! }
//!
//! // During lowering, one call per `obj.field` site:
//! let v = ic.emit_load(&mut f, obj_val, "field_name");
//!
//! // After all lowering, allocate the cache array and freeze:
//! let ic_runtime = ic.finalize();
//!
//! // Before running JIT code, install the runtime on the calling thread:
//! let _guard = ic_runtime.install_thread();
//! gc.run_jit(&jit, main, &args);
//! ```

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, RwLock};

use dynalloc::follow_forwarding;
use dynir::{CmpOp, FuncRef, Type, Value};
use dynobj::roots::RootSource;
use dynsym::{DispatchTable, InlineCacheArray, InlineCacheEntry, Symbol, SymbolTable};

use crate::{DynFunc, DynModule, NanBoxTags, ObjType, Signature};

/// Canonical name of the property-IC slow-path extern. `DynGcRuntime::compile_jit`
/// recognizes this and routes it to the toolkit thunk automatically; embedders
/// never declare or bind it themselves.
pub const PROP_SLOW_EXTERN: &str = "__dynlang_prop_slow__";

thread_local! {
    static ACTIVE_IC: Cell<*const PropertyIcRuntime> =
        const { Cell::new(std::ptr::null()) };
}

/// Heap-allocated cell whose address is stable for the program's lifetime.
/// IR loads `*cell` to find the live IC array; the value at the cell is
/// written exactly once, at `PropertyIc::finalize`.
struct IcArrayCell {
    array_ptr: AtomicPtr<InlineCacheEntry>,
}

/// Builder for the property-access inline cache. Mints cache slot ids and
/// emits the guard / fast-load / slow-call / merge IR shape at every
/// `emit_load` call. Convert to `PropertyIcRuntime` via `finalize`.
pub struct PropertyIc {
    symbols: SymbolTable,
    /// class_key (`u16 type_id` + 1, as u64) → field-offset table.
    per_type: HashMap<u64, DispatchTable>,
    /// Address of `*cell` is baked into IR; `cell.array_ptr` is filled at
    /// finalize. Boxing pins the cell at a stable heap address.
    cell: Box<IcArrayCell>,
    next_site_id: u32,
    slow_ref: FuncRef,
    tags: NanBoxTags,
}

/// Live IC runtime. Owns the cache array (entries mutated in place by the
/// slow-path thunk) and the symbol/dispatch tables it consults. Must
/// outlive the JIT — IR holds the address of `cell`.
pub struct PropertyIcRuntime {
    symbols: SymbolTable,
    per_type: HashMap<u64, DispatchTable>,
    /// Owned. Address baked into the cell at finalize. Never reallocated.
    array: InlineCacheArray,
    /// Owned. Its address is in IR — must outlive the JIT.
    cell: Box<IcArrayCell>,
    tags: NanBoxTags,
}

impl PropertyIc {
    /// Create a new IC builder. Declares the slow-path extern on the
    /// module so call sites can target it; the toolkit binds the thunk
    /// automatically at JIT time.
    pub fn new(dyn_module: &mut DynModule) -> Self {
        let slow_ref = dyn_module.declare_extern(
            PROP_SLOW_EXTERN,
            Signature {
                params: vec![Type::I64, Type::I64, Type::I64],
                ret: Some(Type::I64),
            },
        );
        let tags = dyn_module.tags().clone();
        PropertyIc {
            symbols: SymbolTable::new(),
            per_type: HashMap::new(),
            cell: Box::new(IcArrayCell {
                array_ptr: AtomicPtr::new(std::ptr::null_mut()),
            }),
            next_site_id: 0,
            slow_ref,
            tags,
        }
    }

    /// FuncRef of the IC slow-path extern (`__dynlang_prop_slow__`).
    /// The slow path may allocate (boxing methods into bound closures),
    /// so frontends running `Module::validate_safepoints` should include
    /// this in their allocator list.
    pub fn slow_ref(&self) -> FuncRef {
        self.slow_ref
    }

    /// Register an object type's field-offset table. Hides the
    /// `class_key = type_id + 1` encoding the slow path uses.
    pub fn register_type(&mut self, ty: &ObjType) {
        let class_key = (ty.type_info.type_id as u64) + 1;
        let mut table = DispatchTable::new();
        for (fname, (off, _kind)) in &ty.field_offsets {
            let sym = self.symbols.intern(fname);
            table.set(sym, *off as u64);
        }
        self.per_type.insert(class_key, table);
    }

    /// Emit IR for `obj.field` using a fresh IC slot. Returns the loaded
    /// NanBox value. Mints + tracks cache_id internally.
    pub fn emit_load(&mut self, f: &mut DynFunc, obj: Value, field: &str) -> Value {
        let sym = self.symbols.intern(field);
        let cache_id = self.next_site_id;
        self.next_site_id += 1;

        // Class key from object header. dynobj::Compact stores u16 type_id
        // at offset 0 with zeroed padding, so a full I64 load gives
        // `type_id as u64`. Adding 1 keeps `0` reserved as the IC empty
        // sentinel (matches InlineCacheEntry::EMPTY.cached_class_id).
        let raw = f.obj_unwrap(obj);
        let type_id = f.fb.load(Type::I64, raw, 0);
        let one = f.fb.iconst(Type::I64, 1);
        let class_key = f.fb.add(type_id, one);

        // IC entry address. The cell's address is stable (heap-Box'd);
        // its value (the array's data pointer) is written at finalize.
        let cell_addr = self.cell.as_ref() as *const IcArrayCell as i64;
        let cell_const = f.fb.iconst(Type::I64, cell_addr);
        let array_base = f.fb.load(Type::I64, cell_const, 0);
        let entry_size = std::mem::size_of::<InlineCacheEntry>() as i64;
        let off = f.fb.iconst(Type::I64, (cache_id as i64) * entry_size);
        let entry_addr = f.fb.add(array_base, off);

        let cached_class = f.fb.load(Type::I64, entry_addr, 0);
        let hit = f.fb.icmp(CmpOp::Eq, cached_class, class_key);

        let hit_bb = f.fb.create_block(&[]);
        let miss_bb = f.fb.create_block(&[]);
        let merge_bb = f.fb.create_block(&[Type::I64]);
        f.fb.br_if(hit, hit_bb, &[], miss_bb, &[]);

        // Fast path: load(raw + cached_offset).
        f.fb.switch_to_block(hit_bb);
        let cached_off = f.fb.load(Type::I64, entry_addr, 8); // cached_value field
        let addr = f.fb.add(raw, cached_off);
        let fast = f.fb.load(Type::I64, addr, 0);
        f.fb.jump(merge_bb, &[fast]);

        // Slow path: extern call fills the entry and returns the value.
        // The slow path may allocate (e.g. boxing a method into a bound
        // closure), so emit an explicit safepoint immediately before
        // the call. `Module::validate_safepoints` enforces this.
        //
        // `obj` must be listed as a live root at the safepoint: it's a
        // heap-pointer regalloc Value with no `is_gc_root` stack slot,
        // and the call uses it after the safepoint. Without rooting,
        // a moving GC firing at this safepoint would forward the object
        // but leave `obj`'s spill slot pointing at the from-space copy;
        // the slow_ref call would then read a stale pointer.
        f.fb.switch_to_block(miss_bb);
        let sym_v = f.fb.iconst(Type::I64, sym.as_u32() as i64);
        let cid_v = f.fb.iconst(Type::I64, cache_id as i64);
        f.fb.safepoint(&[obj]);
        let slow = f.fb.call(self.slow_ref, &[obj, sym_v, cid_v]).unwrap();
        f.fb.jump(merge_bb, &[slow]);

        f.fb.switch_to_block(merge_bb);
        f.fb.block_param(merge_bb, 0)
    }

    /// Number of IC sites emitted so far. Mostly useful for diagnostics.
    pub fn site_count(&self) -> u32 {
        self.next_site_id
    }

    /// Allocate the cache array, write its address into the indirection
    /// cell, and freeze. The returned runtime must outlive the JIT.
    pub fn finalize(self) -> PropertyIcRuntime {
        let array = InlineCacheArray::new(self.next_site_id as usize);
        let array_ptr = array.as_ptr() as *mut InlineCacheEntry;
        self.cell.array_ptr.store(array_ptr, Ordering::Release);
        PropertyIcRuntime {
            symbols: self.symbols,
            per_type: self.per_type,
            array,
            cell: self.cell,
            tags: self.tags,
        }
    }
}

impl PropertyIcRuntime {
    /// Look up a symbol's name. Useful for embedder-side error messages.
    pub fn try_name(&self, sym: Symbol) -> Option<&str> {
        self.symbols.try_name(sym)
    }

    /// Number of cache slots.
    pub fn site_count(&self) -> usize {
        self.array.len()
    }

    /// Install this runtime as the current thread's active property IC.
    /// JIT code emitted by the matching `PropertyIc` will route slow-path
    /// calls to this runtime. The returned guard restores the previous
    /// state on drop.
    pub fn install_thread(&self) -> IcThreadGuard<'_> {
        let prev = ACTIVE_IC.with(|c| c.replace(self as *const _));
        IcThreadGuard {
            prev,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// GC root scanning. PropertyIc caches field *offsets* (GC-invariant
/// integers) in `cached_value`, so this impl visits no slots —
/// `InlineCacheArray::cached_value_slots` returns empty when the array
/// wasn't constructed with `new_with_pointer_values`. The trait is still
/// implemented so embedders can uniformly register every IC runtime as a
/// root source without policy branching on the IC's kind.
impl RootSource for PropertyIcRuntime {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for slot in self.array.cached_value_slots() {
            // The scanner is responsible for inspecting the slot's NanBox
            // tag and ignoring non-pointer values (immediates, sentinels)
            // in Direct-mode ICs that mix the two. PropertyIc never holds
            // pointers here at all.
            visitor(slot);
        }
    }
}

/// Guard returned by `PropertyIcRuntime::install_thread`. On drop,
/// restores whatever IC runtime was previously installed (if any).
pub struct IcThreadGuard<'a> {
    prev: *const PropertyIcRuntime,
    _phantom: std::marker::PhantomData<&'a PropertyIcRuntime>,
}

impl Drop for IcThreadGuard<'_> {
    fn drop(&mut self) {
        ACTIVE_IC.with(|c| c.set(self.prev));
    }
}

// ── Slow-path thunk ────────────────────────────────────────────────

/// JIT-bound thunk for `__dynlang_prop_slow__`. Reads the active IC
/// runtime from TLS, walks any forwarding pointer, fills the cache entry,
/// returns the loaded field value.
///
/// `pub(crate)` so `gc.rs` can register it; not part of the public API.
pub(crate) extern "C" fn prop_slow_thunk(obj_bits: u64, sym_id: u64, cache_id: u64) -> u64 {
    let ic_ptr = ACTIVE_IC.with(|c| c.get());
    assert!(
        !ic_ptr.is_null(),
        "dynlang: __dynlang_prop_slow__ called without PropertyIcRuntime installed \
         (call PropertyIcRuntime::install_thread before run_jit)",
    );
    let ic = unsafe { &*ic_ptr };

    // Decode NanBox to a raw object pointer using the configured ptr tag.
    // Layout: high 14 bits + 2-bit tag = TAG_PATTERN | (tag << 48); low
    // 48 bits = payload.
    const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
    const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
    const TAG_FIELD_MASK: u64 = 0xFFFF_0000_0000_0000;
    let expected = TAG_PATTERN | ((ic.tags.ptr as u64) << 48);
    if (obj_bits & TAG_FIELD_MASK) != expected {
        panic!(
            "dynlang: prop slow called on non-object NanBox 0x{:x} (expected ptr tag {})",
            obj_bits, ic.tags.ptr,
        );
    }
    let raw_from_bits = (obj_bits & PAYLOAD_MASK) as usize as *const u8;
    // Follow forwarding via dynalloc (single hop; debug-asserts no chain).
    let raw_ptr = unsafe { follow_forwarding(raw_from_bits) };
    let header = unsafe { *(raw_ptr as *const u64) };

    let class_key = header + 1;
    let sym = Symbol::from_raw(sym_id as u32);
    let table = ic.per_type.get(&class_key).unwrap_or_else(|| {
        panic!(
            "dynlang: no dispatch table for class_key {} (sym `{}`)",
            class_key,
            ic.symbols.try_name(sym).unwrap_or("<unknown>"),
        )
    });
    let offset = table.get(sym).unwrap_or_else(|| {
        panic!(
            "dynlang: class_key {} has no field `{}`",
            class_key,
            ic.symbols.try_name(sym).unwrap_or("<unknown>"),
        )
    });

    // Update the IC entry. The array isn't mutably reachable through `&ic`
    // (we hold a shared reference), but the entries are POD and the slow
    // path is single-threaded per JIT thread — write through the cell's
    // raw pointer.
    let array_ptr = ic.cell.array_ptr.load(Ordering::Acquire);
    debug_assert!(
        !array_ptr.is_null(),
        "dynlang: prop slow ran before PropertyIc::finalize stored the array address",
    );
    let entry_ptr = unsafe { array_ptr.add(cache_id as usize) };
    unsafe {
        (*entry_ptr).cached_class_id = class_key;
        (*entry_ptr).cached_value = offset;
    }

    unsafe { *(raw_ptr.add(offset as usize) as *const u64) }
}

// ── TypeDispatchIc ─────────────────────────────────────────────────
//
// Generalized inline cache for "given a receiver and a key, find a
// per-type value." Strictly more general than `PropertyIc`:
//
//   - Configurable class-key strategy (default: type_id from header
//     + 1; frontends can supply their own — e.g. Clojure switches on
//     `is_record` and reads the record's type_name field).
//
//   - Two cache policies. `LoadOffset` matches PropertyIc semantics:
//     the cached value is an offset, fast path does `load(recv + cv)`.
//     `Direct` returns `cv` itself — for method dispatch (cached
//     value is a fn_obj NanBox), protocol-satisfaction sentinels, and
//     any future "value-as-cached" pattern.
//
//   - Live `per_type` table: `extend-type` / `set` mutate the runtime
//     after finalize. PropertyIc froze its table at finalize because
//     field layouts are static; method dispatch isn't.
//
//   - Runtime threading via IR-baked cell pointer instead of TLS. Each
//     `TypeDispatchIc` instance owns a `Box<DispatchIcCell>` whose
//     address is constant-folded into every call site; the slow path
//     reads everything it needs (runtime, policy, array) from the cell.
//     No `install_thread` step.
//
// PropertyIc still uses its TLS-based mechanism; this is parallel,
// not a replacement. The two can coexist in the same module.

/// Canonical name of the TypeDispatchIc slow-path extern.
/// `DynGcRuntime::compile_jit` auto-binds it (see `gc.rs`).
pub const DISPATCH_SLOW_EXTERN: &str = "__dynlang_dispatch_slow__";

/// Policy for what `cached_value` means at the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    /// `cached_value` is a byte offset; fast-path = `load(recv_raw + cached_value)`.
    /// GC-invariant — slots are never scanned. Matches `PropertyIc`.
    LoadOffset,
    /// `cached_value` is the result. Fast-path returns it directly.
    /// Slots may hold NanBox heap pointers; the cache array is registered
    /// as a `RootSource` so the moving GC can update them.
    Direct,
}

/// Closure that emits IR producing the class-key `i64` from a receiver
/// `Value`. The class-key is what the IC compares against `cached_class_id`.
///
/// Default behavior (when `class_key_strategy` is omitted): read the type_id
/// at offset 0 of the receiver's raw pointer, add 1 (keeps 0 reserved as the
/// `InlineCacheEntry::EMPTY` sentinel).
///
/// Custom strategies can switch on receiver shape — e.g. Clojure's records
/// all share one `type_id`, so the strategy reads the instance-side
/// `type_name` symbol instead.
pub type ClassKeyStrategy = Box<dyn Fn(&mut crate::DynFunc, Value) -> Value + Send + Sync>;

/// Cell baked into IR call sites. Address is heap-stable (Box'd) so it can
/// be reused across module finalize / JIT runs. Holds the array data pointer
/// and a back-pointer to the runtime that owns the cell, letting the slow
/// path find its dispatch tables without TLS.
pub(crate) struct DispatchIcCell {
    pub(crate) array_ptr: AtomicPtr<InlineCacheEntry>,
    pub(crate) runtime_ptr: AtomicPtr<TypeDispatchIcRuntime>,
}

/// Builder for a TypeDispatchIc. Mints cache-slot ids, emits per-call-site
/// IR, accumulates per-type dispatch tables. `finalize` returns the runtime.
pub struct TypeDispatchIc {
    name: String,
    symbols: SymbolTable,
    policy: CachePolicy,
    /// `class_key → DispatchTable`. Shared with `TypeDispatchIcRuntime`
    /// so post-finalize `set()` calls (e.g. `extend-type`) are visible
    /// to slow-path lookups without re-finalizing.
    per_type: Arc<RwLock<HashMap<u64, DispatchTable>>>,
    cell: Box<DispatchIcCell>,
    next_site_id: u32,
    slow_ref: FuncRef,
    tags: NanBoxTags,
    class_key_strategy: ClassKeyStrategy,
}

/// Live runtime. Holds the cache array (mutated in place by the slow path),
/// the per-type dispatch tables (mutated in place by `set`), and the cell
/// whose address is baked into IR. Must outlive the JIT it's paired with.
pub struct TypeDispatchIcRuntime {
    name: String,
    symbols: SymbolTable,
    policy: CachePolicy,
    per_type: Arc<RwLock<HashMap<u64, DispatchTable>>>,
    array: InlineCacheArray,
    cell: Box<DispatchIcCell>,
    tags: NanBoxTags,
}

impl TypeDispatchIc {
    /// Create a new dispatch IC builder. `name` is used for the slow-path
    /// extern declaration (so multiple ICs in the same module each get their
    /// own FuncRef) and in diagnostics.
    ///
    /// The default class-key strategy reads type_id from the receiver
    /// header (offset 0) and adds 1. Override via [`with_class_key_strategy`].
    pub fn new(dyn_module: &mut DynModule, name: &str, policy: CachePolicy) -> Self {
        let slow_ref = dyn_module.declare_extern(
            DISPATCH_SLOW_EXTERN,
            Signature {
                params: vec![Type::I64, Type::I64, Type::I64, Type::I64, Type::I64],
                ret: Some(Type::I64),
            },
        );
        let tags = dyn_module.tags().clone();
        let default_strategy: ClassKeyStrategy = Box::new(|f, obj| {
            // Same encoding PropertyIc uses.
            let raw = f.obj_unwrap(obj);
            let type_id = f.fb.load(Type::I64, raw, 0);
            let one = f.fb.iconst(Type::I64, 1);
            f.fb.add(type_id, one)
        });
        TypeDispatchIc {
            name: name.to_string(),
            symbols: SymbolTable::new(),
            policy,
            per_type: Arc::new(RwLock::new(HashMap::new())),
            cell: Box::new(DispatchIcCell {
                array_ptr: AtomicPtr::new(std::ptr::null_mut()),
                runtime_ptr: AtomicPtr::new(std::ptr::null_mut()),
            }),
            next_site_id: 0,
            slow_ref,
            tags,
            class_key_strategy: default_strategy,
        }
    }

    /// Override the IR-emitting closure that derives the class-key from a
    /// receiver. The closure must produce an `I64` Value whose result will
    /// be compared against `cached_class_id`. The same encoding must be
    /// used when calling [`set`] (the class_key key passed there matches
    /// what the strategy emits).
    pub fn with_class_key_strategy(mut self, strategy: ClassKeyStrategy) -> Self {
        self.class_key_strategy = strategy;
        self
    }

    /// FuncRef of the slow-path extern. Frontends running
    /// `Module::validate_safepoints` should include this in their
    /// allocator list since the slow path may allocate.
    pub fn slow_ref(&self) -> FuncRef {
        self.slow_ref
    }

    /// Intern a symbol on this IC's symbol table. Returns the Symbol
    /// the IR will pass to the slow path for this name. Use this when
    /// you need the Symbol but aren't emitting a lookup right now
    /// (e.g. to pre-populate `set` calls before emit_lookup).
    pub fn intern(&mut self, name: &str) -> Symbol {
        self.symbols.intern(name)
    }

    /// Look up a symbol without interning. Returns `None` if not seen.
    pub fn lookup_symbol(&self, name: &str) -> Option<Symbol> {
        self.symbols.lookup(name)
    }

    /// Set the value for `(class_key, key)`. Both the *builder* and the
    /// *runtime* see this — writes are visible to slow-path lookups
    /// after `finalize`.
    ///
    /// For `LoadOffset` policy, `value` is the byte offset of the field
    /// in the receiver. For `Direct`, `value` is the NanBox bits (or
    /// any u64) the fast-path will return on a hit.
    ///
    /// Writes also invalidate any cached entry on this class_key (cheap:
    /// O(sites)). Cache entries for other class_keys are unaffected.
    pub fn set(&mut self, class_key: u64, key: &str, value: u64) {
        let sym = self.symbols.intern(key);
        let mut tables = self.per_type.write().unwrap();
        tables
            .entry(class_key)
            .or_insert_with(DispatchTable::new)
            .set(sym, value);
        drop(tables);
        self.invalidate_for_class(class_key);
    }

    /// Pre-register a type so the slow path has somewhere to insert into
    /// even before the first `set`. Optional — `set` auto-creates the
    /// table — but useful when the caller wants to walk a `dyn_module`'s
    /// `ObjType` list and register them all up front.
    pub fn register_class_key(&mut self, class_key: u64) {
        let mut tables = self.per_type.write().unwrap();
        tables.entry(class_key).or_insert_with(DispatchTable::new);
    }

    /// Convenience: register an object type using the default class-key
    /// encoding (`type_id + 1`). Use only when the IC was built with the
    /// default `class_key_strategy`.
    pub fn register_type(&mut self, ty: &ObjType) {
        let class_key = (ty.type_info.type_id as u64) + 1;
        self.register_class_key(class_key);
    }

    fn invalidate_for_class(&self, class_key: u64) {
        let array_ptr = self.cell.array_ptr.load(Ordering::Acquire);
        if array_ptr.is_null() {
            return; // not finalized yet — no cache entries exist
        }
        // Linear scan; cache count is typically modest (dozens to hundreds).
        // If this becomes a bottleneck, add a class_key → cache_id index.
        for i in 0..self.next_site_id as usize {
            let entry = unsafe { &mut *array_ptr.add(i) };
            if entry.cached_class_id == class_key {
                *entry = InlineCacheEntry::EMPTY;
            }
        }
    }

    /// Emit IR for `lookup(receiver, key)`. Returns the loaded value
    /// (LoadOffset) or the cached value (Direct). Mints a fresh cache slot.
    pub fn emit_lookup(&mut self, f: &mut crate::DynFunc, receiver: Value, key: &str) -> Value {
        let sym = self.symbols.intern(key);
        let cache_id = self.next_site_id;
        self.next_site_id += 1;

        let class_key = (self.class_key_strategy)(f, receiver);

        // For LoadOffset we need the raw pointer in the fast path; for
        // Direct we don't. Compute it once if needed.
        let raw = match self.policy {
            CachePolicy::LoadOffset => Some(f.obj_unwrap(receiver)),
            CachePolicy::Direct => None,
        };

        let cell_addr = self.cell.as_ref() as *const DispatchIcCell as i64;
        let cell_const = f.fb.iconst(Type::I64, cell_addr);
        let array_base = f.fb.load(Type::I64, cell_const, 0);
        let entry_size = std::mem::size_of::<InlineCacheEntry>() as i64;
        let off = f.fb.iconst(Type::I64, (cache_id as i64) * entry_size);
        let entry_addr = f.fb.add(array_base, off);

        let cached_class = f.fb.load(Type::I64, entry_addr, 0);
        let hit = f.fb.icmp(CmpOp::Eq, cached_class, class_key);

        let hit_bb = f.fb.create_block(&[]);
        let miss_bb = f.fb.create_block(&[]);
        let merge_bb = f.fb.create_block(&[Type::I64]);
        f.fb.br_if(hit, hit_bb, &[], miss_bb, &[]);

        // Fast path.
        f.fb.switch_to_block(hit_bb);
        let cached_value = f.fb.load(Type::I64, entry_addr, 8);
        let fast = match self.policy {
            CachePolicy::LoadOffset => {
                let raw_v = raw.expect("LoadOffset emit_lookup precomputes raw");
                let addr = f.fb.add(raw_v, cached_value);
                f.fb.load(Type::I64, addr, 0)
            }
            CachePolicy::Direct => cached_value,
        };
        f.fb.jump(merge_bb, &[fast]);

        // Slow path. Safepoint covers the receiver (live across the call).
        f.fb.switch_to_block(miss_bb);
        let sym_v = f.fb.iconst(Type::I64, sym.as_u32() as i64);
        let cid_v = f.fb.iconst(Type::I64, cache_id as i64);
        // Recompute class_key in the miss block — it was originally
        // emitted in the entry block, but the IR safepoint+call lives in
        // miss_bb. Cross-block uses of SSA values typically work, but the
        // interpreter materializes block-param values per-block; we keep
        // semantics simple by recomputing on the slow path (which runs
        // at most once per miss anyway).
        let class_key_in_miss = (self.class_key_strategy)(f, receiver);
        f.fb.safepoint(&[receiver]);
        let slow =
            f.fb.call(
                self.slow_ref,
                &[cell_const, receiver, class_key_in_miss, sym_v, cid_v],
            )
            .expect("dispatch slow path returns a value");
        f.fb.jump(merge_bb, &[slow]);

        f.fb.switch_to_block(merge_bb);
        f.fb.block_param(merge_bb, 0)
    }

    /// Number of IC sites emitted so far.
    pub fn site_count(&self) -> u32 {
        self.next_site_id
    }

    /// Finalize: allocate the cache array, store its address in the cell,
    /// and consume the builder into a runtime. The runtime is `Box`'d so
    /// the `runtime_ptr` baked into IR remains valid even if the caller
    /// moves the handle around.
    pub fn finalize(self) -> Box<TypeDispatchIcRuntime> {
        let array = match self.policy {
            CachePolicy::LoadOffset => InlineCacheArray::new(self.next_site_id as usize),
            CachePolicy::Direct => {
                InlineCacheArray::new_with_pointer_values(self.next_site_id as usize)
            }
        };
        let array_ptr = array.as_ptr() as *mut InlineCacheEntry;
        self.cell.array_ptr.store(array_ptr, Ordering::Release);

        let runtime = Box::new(TypeDispatchIcRuntime {
            name: self.name,
            symbols: self.symbols,
            policy: self.policy,
            per_type: self.per_type,
            array,
            cell: self.cell,
            tags: self.tags,
        });
        // Now that the runtime has a stable Box'd address, publish it
        // into the cell so the slow path can find it. The cell is owned
        // by the runtime; the Box won't move once we return it because
        // its address is what every IR call site uses.
        let rt_ptr = &*runtime as *const TypeDispatchIcRuntime as *mut TypeDispatchIcRuntime;
        runtime.cell.runtime_ptr.store(rt_ptr, Ordering::Release);
        runtime
    }
}

impl TypeDispatchIcRuntime {
    /// Set (or update) a value for `(class_key, key)`. Visible to slow-path
    /// lookups immediately. Invalidates any cache slot currently pinned to
    /// this `class_key` so the next call refills.
    pub fn set(&self, class_key: u64, key: &str, value: u64) {
        let sym = {
            // Symbols are append-only; we need write access only when
            // interning a fresh name. Re-borrow under a write lock once
            // we know we're racing.
            //
            // SymbolTable doesn't currently support concurrent intern, so
            // we hold an outer mutex via the per_type write lock as a
            // proxy: callers must serialize symbol interning either by
            // pre-interning before JIT, or by ensuring all `set` calls
            // happen single-threadedly (today's frontends do).
            //
            // We avoid `&mut self` so the runtime can stay behind a `Box`
            // (its address is baked into IR; mutating through &mut would
            // require Sync/&self anyway).
            //
            // SAFETY: see invariant above.
            let symtab_ptr = &self.symbols as *const SymbolTable as *mut SymbolTable;
            unsafe { (*symtab_ptr).intern(key) }
        };
        let mut tables = self.per_type.write().unwrap();
        tables
            .entry(class_key)
            .or_insert_with(DispatchTable::new)
            .set(sym, value);
        drop(tables);
        self.invalidate_for_class(class_key);
    }

    /// Look up `(class_key, key)`. Returns `None` if the key is unknown to
    /// the IC (never interned) or the table has no entry for it.
    /// Cheap — does not invoke the JIT or fill any cache.
    pub fn lookup(&self, class_key: u64, key: &str) -> Option<u64> {
        let sym = self.symbols.lookup(key)?;
        let tables = self.per_type.read().unwrap();
        tables.get(&class_key).and_then(|t| t.get(sym))
    }

    /// Look up by Symbol when the caller already holds one (avoids the
    /// string lookup). Mirrors `DispatchTable::get`.
    pub fn lookup_sym(&self, class_key: u64, sym: Symbol) -> Option<u64> {
        let tables = self.per_type.read().unwrap();
        tables.get(&class_key).and_then(|t| t.get(sym))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn policy(&self) -> CachePolicy {
        self.policy
    }

    pub fn site_count(&self) -> usize {
        self.array.len()
    }

    pub fn tags(&self) -> &NanBoxTags {
        &self.tags
    }

    fn invalidate_for_class(&self, class_key: u64) {
        let array_ptr = self.cell.array_ptr.load(Ordering::Acquire);
        if array_ptr.is_null() {
            return;
        }
        for i in 0..self.array.len() {
            let entry = unsafe { &mut *array_ptr.add(i) };
            if entry.cached_class_id == class_key {
                *entry = InlineCacheEntry::EMPTY;
            }
        }
    }
}

/// GC root scanning for Direct-policy runtimes whose `cached_value`
/// slots may hold heap pointers. LoadOffset runtimes' array reports
/// `value_is_ptr() == false`, so the iterator is empty and this is a
/// no-op. Frontends should register the runtime as an extra root source
/// via `DynGcRuntime::register_extra_root_source` after `finalize`.
impl RootSource for TypeDispatchIcRuntime {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for slot in self.array.cached_value_slots() {
            visitor(slot);
        }
    }
}

// ── Dispatch slow-path thunk ───────────────────────────────────────

/// JIT-bound thunk for `__dynlang_dispatch_slow__`. Unlike PropertyIc's
/// slow path, the IC runtime is reached via the IR-baked cell pointer
/// (no TLS).
///
/// Args: `(cell_ptr, receiver, class_key, sym_id, cache_id)`.
/// Returns the resolved value per the IC's `CachePolicy`.
///
/// `pub(crate)` so `gc.rs` can register it; not part of the public API.
pub(crate) extern "C" fn dispatch_slow_thunk(
    cell_ptr: u64,
    receiver: u64,
    class_key: u64,
    sym_id: u64,
    cache_id: u64,
) -> u64 {
    let cell = unsafe { &*(cell_ptr as usize as *const DispatchIcCell) };
    let runtime_ptr = cell.runtime_ptr.load(Ordering::Acquire);
    assert!(
        !runtime_ptr.is_null(),
        "dynlang: dispatch slow thunk ran before TypeDispatchIc::finalize"
    );
    let runtime = unsafe { &*runtime_ptr };

    eprintln!(
        "[dispatch_slow_thunk] cell_ptr={:#x} receiver={} class_key={} sym_id={} cache_id={}",
        cell_ptr, receiver, class_key, sym_id, cache_id,
    );
    let sym = Symbol::from_raw(sym_id as u32);
    let value = {
        let tables = runtime.per_type.read().unwrap();
        let table = tables.get(&class_key).unwrap_or_else(|| {
            let keys: Vec<u64> = tables.keys().copied().collect();
            panic!(
                "dynlang ({}): no dispatch table for class_key {} (key `{}`); have keys {:?}",
                runtime.name,
                class_key,
                runtime.symbols.try_name(sym).unwrap_or("<unknown>"),
                keys,
            )
        });
        table.get(sym).unwrap_or_else(|| {
            panic!(
                "dynlang ({}): class_key {} has no entry for key `{}`",
                runtime.name,
                class_key,
                runtime.symbols.try_name(sym).unwrap_or("<unknown>"),
            )
        })
    };

    // Update the cache entry. `cached_value` semantics depend on policy
    // (offset vs result), but storage is identical.
    let array_ptr = cell.array_ptr.load(Ordering::Acquire);
    debug_assert!(!array_ptr.is_null());
    let entry_ptr = unsafe { array_ptr.add(cache_id as usize) };
    unsafe {
        (*entry_ptr).cached_class_id = class_key;
        (*entry_ptr).cached_value = value;
    }

    // For LoadOffset, mirror PropertyIc's behavior: walk the receiver's
    // forwarding pointer and return `load(raw + offset)`. For Direct,
    // return `value` verbatim.
    match runtime.policy {
        CachePolicy::Direct => value,
        CachePolicy::LoadOffset => {
            const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
            const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
            const TAG_FIELD_MASK: u64 = 0xFFFF_0000_0000_0000;
            let expected = TAG_PATTERN | ((runtime.tags.ptr as u64) << 48);
            if (receiver & TAG_FIELD_MASK) != expected {
                panic!(
                    "dynlang ({}): dispatch LoadOffset slow called on non-object NanBox 0x{:x}",
                    runtime.name, receiver,
                );
            }
            let raw_from_bits = (receiver & PAYLOAD_MASK) as usize as *const u8;
            let raw_ptr = unsafe { follow_forwarding(raw_from_bits) };
            unsafe { *(raw_ptr.add(value as usize) as *const u64) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FieldKind, GcConfig};

    #[test]
    fn build_and_finalize_empty() {
        let mut dyn_module =
            DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let ic = PropertyIc::new(&mut dyn_module);
        let rt = ic.finalize();
        assert_eq!(rt.site_count(), 0);
    }

    #[test]
    fn register_type_populates_table() {
        let mut dyn_module =
            DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let id = dyn_module
            .obj_type("Point")
            .field("x", FieldKind::Value)
            .field("y", FieldKind::Value)
            .build();
        let mut ic = PropertyIc::new(&mut dyn_module);
        let ty = dyn_module.get_obj_type(id);
        ic.register_type(ty);
        // class_key = type_id + 1
        let class_key = (ty.type_info.type_id as u64) + 1;
        assert!(ic.per_type.contains_key(&class_key));
    }

    #[test]
    fn install_guard_restores() {
        let mut dyn_module =
            DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());
        let ic = PropertyIc::new(&mut dyn_module);
        let rt = ic.finalize();
        ACTIVE_IC.with(|c| assert!(c.get().is_null()));
        {
            let _g = rt.install_thread();
            ACTIVE_IC.with(|c| assert!(!c.get().is_null()));
        }
        ACTIVE_IC.with(|c| assert!(c.get().is_null()));
    }

    /// End-to-end Direct-mode test for `TypeDispatchIc`. Uses a custom
    /// class_key_strategy that treats the receiver bits *as* the class_key,
    /// so the test doesn't need real heap objects — it isolates the IC's
    /// guard/fast-path/slow-path/cache-fill/invalidation logic.
    ///
    /// Exercises:
    ///   1. First lookup misses, slow path fills cache, returns value.
    ///   2. Second lookup with the same class_key hits the fast path.
    ///   3. `runtime.set` updates the table AND invalidates the slot, so
    ///      the next lookup re-misses with the new value.
    ///   4. A different class_key uses a different table; cache entries
    ///      for one class_key don't pollute another.
    #[test]
    fn direct_mode_emit_lookup_round_trip() {
        use dynalloc::LowBitPtrPolicy;
        use dynir::gc_runtime::GcInterpCtx;
        use dynir::interp::{ExternCallResult, InterpResult, ModuleInterpreter};
        use dynobj::Compact;
        use dynvalue::NanBox;
        type TestRoots = GcInterpCtx<Compact, LowBitPtrPolicy<3>>;

        let mut dyn_module =
            DynModule::new(GcConfig::generational(64 * 1024), NanBoxTags::default());

        // First, a sanity-check function: take one arg and return it.
        // If the interpreter can't echo our arg, the test setup is wrong.
        let echo_fn = dyn_module.declare_func("echo", 1);
        {
            let mut f = dyn_module.start_func(echo_fn);
            let entry = f.fb.entry_block();
            let arg = f.fb.block_param(entry, 0);
            f.fb.ret(arg);
            dyn_module.finish_func(f);
        }

        // Custom class-key strategy: identity on receiver bits. The
        // class_key passed to `set` will equal the receiver value the
        // test invokes the function with.
        let identity_strategy: ClassKeyStrategy = Box::new(|_f, recv| recv);
        let mut ic = TypeDispatchIc::new(&mut dyn_module, "test_methods", CachePolicy::Direct)
            .with_class_key_strategy(identity_strategy);

        // Declare `lookup(receiver) -> u64`.
        let lookup_fn = dyn_module.declare_func("lookup", 1);
        let mut f = dyn_module.start_func(lookup_fn);
        let receiver = f.fb.block_param(f.fb.entry_block(), 0);
        let result = ic.emit_lookup(&mut f, receiver, "the_method");
        f.fb.ret(result);
        dyn_module.finish_func(f);

        let runtime = ic.finalize();
        let built = dyn_module.build();

        // Sanity: echo(100) must return 100. If this fails, the test
        // harness itself is wrong; if it passes, the slow-path issue
        // is downstream.
        {
            let roots: TestRoots = GcInterpCtx::new_unallocating();
            let interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);
            match interp.run(echo_fn, &[100]) {
                Ok(InterpResult::Value(v)) => assert_eq!(v, 100, "echo sanity check"),
                other => panic!("echo returned: {:?}", other),
            }
        }

        // Dump the lookup function's IR for debugging.
        for func in &built.module.functions {
            if func.name == "lookup" {
                eprintln!("--- lookup IR ---\n{}\n---", func);
            }
        }

        // ── 1. First lookup misses → slow path fills cache.
        runtime.set(100, "the_method", 42);
        let result = run_with_dispatch(&built.module, lookup_fn, &[100], &runtime);
        assert_eq!(result, 42, "first lookup must return what we set");

        // ── 2. Second lookup with same class_key hits fast path.
        // (We can't easily observe "fast path hit" from outside, but it
        // must still return the cached value.)
        let result = run_with_dispatch(&built.module, lookup_fn, &[100], &runtime);
        assert_eq!(result, 42);

        // ── 3. Update via `set` — must invalidate the cache slot.
        runtime.set(100, "the_method", 99);
        let result = run_with_dispatch(&built.module, lookup_fn, &[100], &runtime);
        assert_eq!(result, 99, "set must invalidate cached entry");

        // ── 4. Different class_key → independent table.
        runtime.set(200, "the_method", 7);
        let result = run_with_dispatch(&built.module, lookup_fn, &[200], &runtime);
        assert_eq!(result, 7);
        // class_key 100 still returns 99.
        let result = run_with_dispatch(&built.module, lookup_fn, &[100], &runtime);
        assert_eq!(result, 99);

        // Helper kept inside this test mod so it doesn't leak.
        fn run_with_dispatch(
            module: &dynir::ir::Module,
            entry: FuncRef,
            args: &[u64],
            _runtime: &TypeDispatchIcRuntime,
        ) -> u64 {
            let roots: GcInterpCtx<Compact, LowBitPtrPolicy<3>> = GcInterpCtx::new_unallocating();
            let mut interp = ModuleInterpreter::<NanBox, _>::new(module, &roots);
            interp.bind_by_name(DISPATCH_SLOW_EXTERN, |args| {
                eprintln!("[binding closure] received args: {:?}", args);
                let v = dispatch_slow_thunk(args[0], args[1], args[2], args[3], args[4]);
                ExternCallResult::Value(Some(v))
            });
            match interp.run(entry, args) {
                Ok(InterpResult::Value(v)) => v,
                Ok(InterpResult::Void) => 0,
                other => panic!("unexpected: {:?}", other),
            }
        }
    }

    /// A NanBox stored in a pointer-mode IC slot must follow forwarding
    /// when the moving GC relocates the underlying object — otherwise
    /// the next fast-path hit reads a stale pointer.
    ///
    /// This test exercises the Phase 1.1 plumbing end-to-end on a
    /// generational backend:
    ///   1. Build an `InlineCacheArray::new_with_pointer_values(1)`.
    ///   2. Allocate a heap object and write a magic value into it.
    ///   3. Store the object's NanBox bits into the cache slot.
    ///   4. Register a `RootSource` wrapper that exposes the slot
    ///      (this is what the slot keeps alive — there is no other
    ///      reference).
    ///   5. Force `gc.collect()`.
    ///   6. Read the slot again; verify the NanBox payload shifted to
    ///      the new heap location AND the magic value survived.
    ///
    /// Models how Direct-mode method ICs will be wired in Phase 2/3.
    #[test]
    fn pointer_mode_cache_slot_survives_moving_gc() {
        use crate::NanBoxTags;
        use dynsym::InlineCacheArray;
        use std::cell::UnsafeCell;

        // Test-only wrapper: a `RootSource` over an `InlineCacheArray`
        // whose `cached_value` slots are scanned. Future `TypeDispatchIc`
        // will provide this as a public type; here we inline it so the
        // GC-integration test doesn't depend on the not-yet-built
        // generalization.
        struct PtrIcRootSource {
            array: UnsafeCell<InlineCacheArray>,
        }
        impl RootSource for PtrIcRootSource {
            fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
                // SAFETY: GC walks roots under STW; no concurrent mutator
                // access to the array. UnsafeCell is the right vehicle
                // for interior mutability without needing &mut.
                let array = unsafe { &*self.array.get() };
                for slot in array.cached_value_slots() {
                    visitor(slot);
                }
            }
        }

        let mut dyn_module = DynModule::new(GcConfig::generational(8192), NanBoxTags::default());
        let pair_ty = dyn_module
            .obj_type("Pair")
            .field("value", FieldKind::Value)
            .build();

        let gc = crate::gc::DynGcRuntime::new(
            &GcConfig::generational(8192),
            dyn_module.tags(),
            &dyn_module.obj_types,
        );
        let _thread = gc.install_thread();

        // Allocate a heap object and stamp a magic value into its
        // `value` field so we can verify content survives across the
        // copy.
        const MAGIC: u64 = 0xDECAFC0FFEE_u64;
        let raw = gc.alloc(pair_ty.0, 0);
        assert!(!raw.is_null());
        let value_offset: usize = dyn_module.obj_types[pair_ty.0]
            .field_offsets
            .iter()
            .find(|(name, _)| *name == "value")
            .map(|(_, (off, _))| *off as usize)
            .expect("Pair has 'value' field");
        unsafe {
            (raw.add(value_offset) as *mut u64).write(MAGIC);
        }
        let nanbox_before = gc.tag_ptr(raw);

        // Build the IC array and write the NanBox into slot 0.
        let mut array = InlineCacheArray::new_with_pointer_values(1);
        array.get_mut(0).cached_class_id = (pair_ty.0 as u64) + 1;
        array.get_mut(0).cached_value = nanbox_before;
        let root_src = PtrIcRootSource {
            array: UnsafeCell::new(array),
        };

        // Register the wrapper. Safety: `root_src` outlives the
        // `gc.collect()` call below — both live in this stack frame.
        unsafe {
            gc.register_extra_root_source(&root_src as *const _ as *const dyn RootSource);
        }

        let before_collections = gc.collection_count();
        gc.collect();
        assert!(
            gc.collection_count() > before_collections,
            "expected a collection"
        );

        // After collection: the NanBox in the slot should have been
        // rewritten to point to the new heap location, AND the object
        // must still hold MAGIC at its `value` field.
        let nanbox_after = unsafe { (*root_src.array.get()).get(0).cached_value };

        // The slot must remain a heap-ptr NanBox (tag preserved).
        const TAG_FIELD_MASK: u64 = 0xFFFF_0000_0000_0000;
        const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
        assert_eq!(
            nanbox_after & TAG_FIELD_MASK,
            nanbox_before & TAG_FIELD_MASK,
            "NanBox tag bits should survive collection unchanged"
        );

        // Read the (possibly relocated) object and check MAGIC.
        let new_raw = (nanbox_after & PAYLOAD_MASK) as usize as *const u8;
        let got = unsafe { (new_raw.add(value_offset) as *const u64).read() };
        assert_eq!(got, MAGIC, "object content must survive the move");
    }
}
