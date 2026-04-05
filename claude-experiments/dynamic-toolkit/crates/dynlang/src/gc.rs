//! GC runtime for dynlang — provides the allocator, root tracking,
//! and PtrPolicy auto-derived from NanBoxTags.

use std::cell::Cell;

use dynalloc::{BumpAllocator, PtrPolicy, SemiSpace, alloc_obj};
use dynobj::roots::RootSource;
use dynobj::{Compact, ObjHeader, TypeInfo};

use crate::{GcConfig, NanBoxTags, ObjType};

// ── NanBox PtrPolicy ──────────────────────────────────────────────

/// PtrPolicy derived from NanBoxTags. Identifies which u64 values
/// are heap pointers based on the NanBox tag scheme.
pub struct NanBoxPtrPolicy {
    ptr_tag: u32,
}

/// Full mask + tag pattern for NanBox
const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

fn nanbox_has_tag(bits: u64, tag: u32) -> bool {
    let expected = TAG_PATTERN | ((tag as u64) << 48);
    let mask = FULL_MASK | (0x0003u64 << 48);
    (bits & mask) == expected
}

fn nanbox_payload(bits: u64) -> u64 {
    bits & PAYLOAD_MASK
}

fn nanbox_encode(tag: u32, payload: u64) -> u64 {
    TAG_PATTERN | ((tag as u64) << 48) | (payload & PAYLOAD_MASK)
}

/// Static PtrPolicy implementation for use with SemiSpace::collect.
/// The ptr_tag is baked in at creation time.
pub struct NanBoxPolicy;

// We use a thread-local to pass the tag to the static PtrPolicy methods.
thread_local! {
    static PTR_TAG: Cell<u32> = const { Cell::new(0) };
}

impl PtrPolicy for NanBoxPolicy {
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        let tag = PTR_TAG.with(|c| c.get());
        if nanbox_has_tag(bits, tag) {
            let payload = nanbox_payload(bits);
            if payload == 0 {
                return None; // null object
            }
            Some(payload as *mut u8)
        } else {
            None
        }
    }

    fn encode_ptr(ptr: *mut u8) -> u64 {
        let tag = PTR_TAG.with(|c| c.get());
        nanbox_encode(tag, ptr as u64)
    }
}

// ── Root Set ──────────────────────────────────────────────────────

/// A simple root set: a Vec of cells that the GC can scan and update.
pub struct RootSet {
    slots: Vec<Cell<u64>>,
}

impl RootSet {
    pub fn new() -> Self {
        RootSet { slots: Vec::new() }
    }

    /// Add a value as a root. Returns the slot index.
    pub fn add(&mut self, val: u64) -> usize {
        let idx = self.slots.len();
        self.slots.push(Cell::new(val));
        idx
    }

    /// Update a root slot.
    pub fn set(&self, idx: usize, val: u64) {
        self.slots[idx].set(val);
    }

    /// Read a root slot (may have been updated by GC).
    pub fn get(&self, idx: usize) -> u64 {
        self.slots[idx].get()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

impl RootSource for RootSet {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for cell in &self.slots {
            visitor(cell.as_ptr());
        }
    }
}

// ── GC Runtime ────────────────────────────────────────────────────

/// Allocator backend (either leak or semi-space).
enum Allocator {
    Leak(BumpAllocator),
    SemiSpace(SemiSpace),
}

/// Runtime GC support. Created from a GcConfig after module compilation.
/// Holds the allocator, type registry, and root set.
pub struct DynGcRuntime {
    allocator: Allocator,
    /// Type registry — indexed by ObjTypeId.
    type_infos: Vec<&'static TypeInfo>,
    /// The NanBox ptr tag for PtrPolicy.
    ptr_tag: u32,
    /// Root set for values held by the language runtime (globals, stack, etc.)
    pub roots: RootSet,
    /// Number of allocations since last GC.
    alloc_count: usize,
    /// Trigger GC every N allocations (0 = never for Leak).
    gc_threshold: usize,
}

impl DynGcRuntime {
    /// Create a new GC runtime from config and the module's object types.
    pub fn new(config: &GcConfig, tags: &NanBoxTags, obj_types: &[ObjType]) -> Self {
        let type_infos: Vec<&'static TypeInfo> = obj_types.iter()
            .map(|t| t.type_info)
            .collect();

        let (allocator, gc_threshold) = match config {
            GcConfig::Leak => {
                // 64 MB bump allocator — should be enough for most programs
                let bump = BumpAllocator::new::<Compact>(64 * 1024 * 1024);
                (Allocator::Leak(bump), 0)
            }
            GcConfig::SemiSpace { heap_size } => {
                let ss = SemiSpace::new::<Compact>(*heap_size);
                (Allocator::SemiSpace(ss), 1024) // collect every 1024 allocs
            }
        };

        DynGcRuntime {
            allocator,
            type_infos,
            ptr_tag: tags.ptr,
            roots: RootSet::new(),
            alloc_count: 0,
            gc_threshold,
        }
    }

    /// Allocate an object. Returns a raw pointer (NOT tagged — caller
    /// must encode with NanBox ptr tag). Returns null on OOM.
    pub fn alloc(&mut self, type_id: usize, varlen_len: usize) -> *mut u8 {
        if type_id >= self.type_infos.len() {
            panic!("gc_alloc: unknown type_id {}", type_id);
        }
        let info = self.type_infos[type_id];

        // Try to collect if semi-space and threshold reached
        if self.gc_threshold > 0 {
            self.alloc_count += 1;
            if self.alloc_count >= self.gc_threshold {
                self.collect();
                self.alloc_count = 0;
            }
        }

        let ptr = match &self.allocator {
            Allocator::Leak(bump) => {
                unsafe { alloc_obj::<Compact>(bump, info, varlen_len) }
            }
            Allocator::SemiSpace(ss) => {
                ss.alloc_obj::<Compact>(info, varlen_len)
            }
        };

        if ptr.is_null() {
            // OOM — try collecting and retry
            self.collect();
            match &self.allocator {
                Allocator::Leak(bump) => {
                    unsafe { alloc_obj::<Compact>(bump, info, varlen_len) }
                }
                Allocator::SemiSpace(ss) => {
                    ss.alloc_obj::<Compact>(info, varlen_len)
                }
            }
        } else {
            ptr
        }
    }

    /// Run garbage collection. For Leak, this is a no-op.
    pub fn collect(&mut self) {
        match &mut self.allocator {
            Allocator::Leak(_) => {} // no-op
            Allocator::SemiSpace(ss) => {
                PTR_TAG.with(|c| c.set(self.ptr_tag));
                unsafe {
                    ss.collect::<NanBoxPolicy>(&mut [&self.roots]);
                }
            }
        }
    }

    /// Encode a raw pointer as a NanBox ptr-tagged value.
    pub fn tag_ptr(&self, ptr: *mut u8) -> u64 {
        if ptr.is_null() {
            return 0;
        }
        nanbox_encode(self.ptr_tag, ptr as u64)
    }

    /// Decode a NanBox ptr-tagged value to a raw pointer.
    pub fn untag_ptr(&self, val: u64) -> *mut u8 {
        if !nanbox_has_tag(val, self.ptr_tag) {
            return std::ptr::null_mut();
        }
        nanbox_payload(val) as *mut u8
    }

    /// Get the TypeInfo for an object type.
    pub fn type_info(&self, type_id: usize) -> &'static TypeInfo {
        self.type_infos[type_id]
    }

    /// Number of registered object types.
    pub fn type_count(&self) -> usize {
        self.type_infos.len()
    }
}
