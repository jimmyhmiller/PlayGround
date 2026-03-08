use dynobj::{
    read_type_info, read_varlen_count, scan_object,
    ObjHeader, RootSource, TypeInfo, VarLenKind,
};

use crate::alloc::{Alloc, BumpAllocator};

// Forwarding pointers are stored in the type_info slot with bit 0 set.
// This requires TypeInfo to have alignment ≥ 2 so valid pointers always
// have bit 0 = 0. Assert this at compile time.
const _: () = assert!(
    core::mem::align_of::<TypeInfo>() >= 2,
    "TypeInfo must be at least 2-byte aligned for forwarding pointer tagging"
);

// ─── PtrPolicy ──────────────────────────────────────────────────────

/// Policy for how the GC identifies and rewrites heap pointers in value slots.
///
/// Each value slot in a root or object field holds a `u64`. The GC is
/// tag-scheme-agnostic — it delegates to this trait to determine which
/// values are heap pointers and how to re-encode them after relocation.
pub trait PtrPolicy {
    /// If `bits` encodes a heap pointer, return the raw pointer.
    /// Return `None` for immediates (fixnums, booleans, nil, etc.).
    fn try_decode_ptr(bits: u64) -> Option<*mut u8>;

    /// Encode a raw heap pointer back into a tagged `u64` value.
    /// Must be the inverse of `try_decode_ptr`.
    fn encode_ptr(ptr: *mut u8) -> u64;
}

// ─── SemiSpace ──────────────────────────────────────────────────────

/// Cheney semi-space copying collector.
///
/// Owns two equal-sized bump regions (from-space and to-space).
/// Allocation goes into from-space. On collection, live objects are
/// copied to to-space using Cheney's breadth-first scan, then the
/// spaces are swapped.
///
/// Generic over:
/// - `PtrPolicy` (how to identify/rewrite heap pointers in value slots)
/// - `ObjHeader` (object header layout — determines where TypeInfo lives)
///
/// The header type is fixed at construction and stored as `type_info_offset`.
pub struct SemiSpace {
    from: BumpAllocator,
    to: BumpAllocator,
    type_info_offset: usize,
    collections: usize,
}

impl SemiSpace {
    /// Create a new semi-space collector with two regions of `space_size` bytes each.
    pub fn new<H: ObjHeader>(space_size: usize) -> Self {
        SemiSpace {
            from: BumpAllocator::new::<H>(space_size),
            to: BumpAllocator::new::<H>(space_size),
            type_info_offset: H::TYPE_INFO_OFFSET,
            collections: 0,
        }
    }

    /// Allocate an object in from-space (raw, no header init).
    ///
    /// Returns null if from-space is exhausted. Call `collect` first
    /// to free space, or check with `from_remaining`.
    pub fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        self.from.alloc(info, varlen_len)
    }

    /// Allocate an object in from-space, initialize header and varlen count.
    ///
    /// Returns null if from-space is exhausted.
    pub fn alloc_obj<H: ObjHeader>(
        &self,
        info: &'static TypeInfo,
        varlen_len: usize,
    ) -> *mut u8 {
        unsafe { crate::alloc::alloc_obj::<H>(&self.from, info, varlen_len) }
    }

    /// Bytes used in from-space.
    pub fn from_used(&self) -> usize {
        self.from.used()
    }

    /// Bytes remaining in from-space.
    pub fn from_remaining(&self) -> usize {
        self.from.remaining()
    }

    /// Total size of each space.
    pub fn space_size(&self) -> usize {
        self.from.size()
    }

    /// Number of collections performed so far.
    pub fn collections(&self) -> usize {
        self.collections
    }

    /// Check if a pointer is in from-space.
    pub fn contains(&self, ptr: *const u8) -> bool {
        self.from.contains(ptr)
    }

    /// Access from-space for heap walking.
    pub fn from_space(&self) -> &BumpAllocator {
        &self.from
    }

    // ─── Collection ─────────────────────────────────────────────

    /// Run a Cheney collection.
    ///
    /// Copies all objects reachable from `roots` into to-space, updates
    /// all root slots and inter-object pointers, then swaps spaces.
    ///
    /// # Safety
    /// - All objects in from-space must have valid headers and varlen counts.
    /// - `roots` must enumerate all live root slots (mutator roots, globals, etc.).
    /// - No references into from-space may be held across this call.
    pub unsafe fn collect<P: PtrPolicy>(&mut self, roots: &mut [&dyn RootSource]) {
        // Phase 1: scan roots, copy/forward root targets
        for source in roots.iter() {
            source.scan_roots(&mut |slot| {
                unsafe { self.process_slot::<P>(slot) };
            });
        }

        // Phase 2: Cheney scan — walk to-space linearly
        let mut scan_offset = 0usize;
        // We re-read to.used() each iteration because copying objects
        // during scan_object grows to-space.
        while scan_offset < self.to.used() {
            let obj = unsafe { self.to.base().add(scan_offset) };
            let info = unsafe { &*read_type_info(obj, self.type_info_offset) };

            unsafe {
                scan_object(obj, info, |slot| {
                    self.process_slot::<P>(slot);
                });
            }

            // Advance past this object
            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(obj, info) },
            };
            let obj_size = info.allocation_size(varlen_len);
            let align = 1usize << info.align_log2;
            scan_offset = (scan_offset + obj_size + align - 1) & !(align - 1);
        }

        // Phase 3: swap spaces, reset old from-space (now to-space)
        core::mem::swap(&mut self.from, &mut self.to);
        self.to.reset();
        self.collections += 1;
    }

    /// Process a single value slot: if it points into from-space,
    /// copy or forward the target and update the slot.
    unsafe fn process_slot<P: PtrPolicy>(&mut self, slot: *mut u64) {
        let bits = unsafe { *slot };
        if let Some(ptr) = P::try_decode_ptr(bits) {
            if self.from.contains(ptr) {
                let new_ptr = unsafe { self.copy_or_forward(ptr) };
                unsafe { *slot = P::encode_ptr(new_ptr) };
            }
        }
    }

    /// Read the type_info slot of an object in from-space.
    /// Returns `None` if the slot contains a forwarding pointer (bit 0 set).
    /// Returns `Some(forwarding_addr)` if forwarded.
    unsafe fn check_forwarded(&self, old: *mut u8) -> Option<*mut u8> {
        let slot = unsafe { old.add(self.type_info_offset) as *const u64 };
        let word = unsafe { *slot };
        if word & 1 == 1 {
            // Bit 0 set → forwarding pointer. Clear bit 0 to recover address.
            Some((word & !1) as *mut u8)
        } else {
            None
        }
    }

    /// Write a forwarding pointer into the type_info slot of the old object.
    ///
    /// Sets bit 0 to distinguish from a valid TypeInfo pointer.
    /// This is safe because:
    /// - TypeInfo has alignment ≥ 2 (contains u16 fields), so real
    ///   TypeInfo pointers always have bit 0 = 0
    /// - Heap pointers are 8-byte aligned, so `ptr | 1` doesn't
    ///   corrupt the address (recovered by `ptr & !1`)
    unsafe fn install_forwarding(&self, old: *mut u8, new: *mut u8) {
        let slot = unsafe { old.add(self.type_info_offset) as *mut u64 };
        unsafe { *slot = (new as u64) | 1 };
    }

    /// If the object at `old` has already been forwarded, return the
    /// forwarding address. Otherwise copy it to to-space and install
    /// a forwarding pointer.
    unsafe fn copy_or_forward(&mut self, old: *mut u8) -> *mut u8 {
        // Check for forwarding pointer in the type_info slot
        if let Some(forwarded) = unsafe { self.check_forwarded(old) } {
            return forwarded;
        }

        // Read type info to determine object size (type_info slot is valid here)
        let info = unsafe { &*read_type_info(old, self.type_info_offset) };
        let varlen_len = match info.varlen {
            VarLenKind::None => 0,
            _ => unsafe { read_varlen_count(old, info) },
        };
        let size = info.allocation_size(varlen_len);

        // Allocate in to-space
        let new = self.to.alloc(info, varlen_len);
        assert!(!new.is_null(), "to-space exhausted during collection");

        // Copy the object intact
        unsafe {
            core::ptr::copy_nonoverlapping(old, new, size);
        }

        // Install forwarding pointer in old object's type_info slot
        unsafe { self.install_forwarding(old, new) };

        new
    }
}
