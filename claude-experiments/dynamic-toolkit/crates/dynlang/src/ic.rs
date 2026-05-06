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
//! let mut ic = PropertyIc::new(&mut dm);
//!
//! // Register dispatch tables for each struct/object type:
//! for ty in &dm.obj_types {
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

use dynalloc::follow_forwarding;
use dynir::{CmpOp, FuncRef, Type, Value};
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
    pub fn new(dm: &mut DynModule) -> Self {
        let slow_ref = dm.declare_extern(
            PROP_SLOW_EXTERN,
            Signature {
                params: vec![Type::I64, Type::I64, Type::I64],
                ret: Some(Type::I64),
            },
        );
        let tags = dm.tags().clone();
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
pub(crate) extern "C" fn prop_slow_thunk(
    obj_bits: u64,
    sym_id: u64,
    cache_id: u64,
) -> u64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FieldKind, GcConfig};

    #[test]
    fn build_and_finalize_empty() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let ic = PropertyIc::new(&mut dm);
        let rt = ic.finalize();
        assert_eq!(rt.site_count(), 0);
    }

    #[test]
    fn register_type_populates_table() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let id = dm
            .obj_type("Point")
            .field("x", FieldKind::Value)
            .field("y", FieldKind::Value)
            .build();
        let mut ic = PropertyIc::new(&mut dm);
        let ty = dm.get_obj_type(id);
        ic.register_type(ty);
        // class_key = type_id + 1
        let class_key = (ty.type_info.type_id as u64) + 1;
        assert!(ic.per_type.contains_key(&class_key));
    }

    #[test]
    fn install_guard_restores() {
        let mut dm = DynModule::new(GcConfig::leak(), NanBoxTags::default());
        let ic = PropertyIc::new(&mut dm);
        let rt = ic.finalize();
        ACTIVE_IC.with(|c| assert!(c.get().is_null()));
        {
            let _g = rt.install_thread();
            ACTIVE_IC.with(|c| assert!(!c.get().is_null()));
        }
        ACTIVE_IC.with(|c| assert!(c.get().is_null()));
    }
}
