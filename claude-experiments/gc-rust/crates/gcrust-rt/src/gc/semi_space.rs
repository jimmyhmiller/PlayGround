use crate::gc::field::{read_type_id, read_varlen_count};
use crate::gc::header::ObjHeader;
use crate::gc::roots::RootSource;
use crate::gc::scan::scan_object;
use crate::gc::type_info::{TypeInfo, VarLenKind};

use crate::gc::alloc::{Alloc, BumpAllocator};

impl Alloc for SemiSpace {
    fn alloc(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        SemiSpace::alloc(self, info, varlen_len)
    }
}

/// High bit of an object header word indicates the object has been moved
/// by the collector; the low 63 bits hold the to-space address.
///
/// Valid heap pointers use at most 48 bits on current hardware, and the
/// original header word (u16 type_id + zero padding) never has bit 63
/// set, so this is a safe sentinel. (Earlier revisions used bit 0, but
/// that broke when headers switched from `*const TypeInfo` — always
/// aligned — to a u16 type_id, which can be odd.)
pub const FORWARDING_BIT: u64 = 1 << 63;

/// If `ptr`'s header word is a forwarding entry, return the to-space
/// pointer; otherwise return `ptr` unchanged. Reads exactly one header
/// word.
///
/// In debug builds, asserts that the followed pointer's header is itself
/// not a forwarding entry — chains are a collector bug.
///
/// # Safety
/// `ptr` must point to a live or forwarded GC object header. Calling on
/// arbitrary memory is UB. Single-threaded read; for concurrent paths
/// see [`crate::gc::read_barrier_atomic`].
#[inline]
pub unsafe fn follow_forwarding(ptr: *const u8) -> *const u8 {
    let header = unsafe { *(ptr as *const u64) };
    if header & FORWARDING_BIT == 0 {
        return ptr;
    }
    let to = (header & !FORWARDING_BIT) as usize as *const u8;
    debug_assert_eq!(
        unsafe { *(to as *const u64) } & FORWARDING_BIT,
        0,
        "dynalloc: forwarding pointer chain (to-space {:p} also forwarded)",
        to,
    );
    to
}

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
/// The header type is fixed at construction and stored as `type_id_offset`.
pub struct SemiSpace {
    from: BumpAllocator,
    to: BumpAllocator,
    type_id_offset: usize,
    collections: usize,
    /// Temporarily set during collect() so copy_or_forward can look up TypeInfo.
    /// Only valid during a collection cycle.
    type_table_ptr: *const TypeInfo,
    type_table_len: usize,
}

impl SemiSpace {
    /// Create a new semi-space collector with two regions of `space_size` bytes each.
    pub fn new<H: ObjHeader>(space_size: usize) -> Self {
        SemiSpace {
            from: BumpAllocator::new::<H>(space_size),
            to: BumpAllocator::new::<H>(space_size),
            type_id_offset: H::TYPE_ID_OFFSET,
            collections: 0,
            type_table_ptr: core::ptr::null(),
            type_table_len: 0,
        }
    }

    /// Allocate an object in from-space (raw, no header init).
    ///
    /// Returns null if from-space is exhausted. Call `collect` first
    /// to free space, or check with `from_remaining`.
    pub fn alloc(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        self.from.alloc(info, varlen_len)
    }

    /// Allocate an object in from-space, initialize header and varlen count.
    ///
    /// Returns null if from-space is exhausted.
    pub fn alloc_obj<H: ObjHeader>(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        unsafe { crate::gc::alloc::alloc_obj::<H>(&self.from, info, varlen_len) }
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
    /// Access the type table set during collect().
    fn type_table(&self) -> &[TypeInfo] {
        unsafe { core::slice::from_raw_parts(self.type_table_ptr, self.type_table_len) }
    }

    pub unsafe fn collect<P: PtrPolicy>(&mut self, type_table: &[TypeInfo], roots: &mut [&dyn RootSource]) {
        // Store type_table reference for use by copy_or_forward
        self.type_table_ptr = type_table.as_ptr();
        self.type_table_len = type_table.len();

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
            let type_id = unsafe { read_type_id(obj, self.type_id_offset) };
            let info = &type_table[type_id as usize];

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

        // Clear type_table reference (no longer valid after collect returns)
        self.type_table_ptr = core::ptr::null();
        self.type_table_len = 0;
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

    /// Check if an object in from-space has been forwarded.
    /// Returns `Some(forwarding_addr)` if bit 63 is set (forwarding marker).
    /// Returns `None` if the header word is a normal type_id (not forwarded).
    unsafe fn check_forwarded(&self, old: *mut u8) -> Option<*mut u8> {
        let slot = unsafe { old.add(self.type_id_offset) as *const u64 };
        let word = unsafe { *slot };
        if word & FORWARDING_BIT != 0 {
            Some((word & !FORWARDING_BIT) as *mut u8)
        } else {
            None
        }
    }

    /// Write a forwarding pointer into the header slot of the old object.
    /// Sets bit 63 to mark it as forwarded.
    unsafe fn install_forwarding(&self, old: *mut u8, new: *mut u8) {
        let slot = unsafe { old.add(self.type_id_offset) as *mut u64 };
        unsafe { *slot = (new as u64) | FORWARDING_BIT };
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
        let type_id = unsafe { read_type_id(old, self.type_id_offset) };
        // PRECISE-LAYOUT DETECTOR (see heap::gc_verify_armed): a traced slot
        // pointing into from-space always targets a real object. An out-of-range
        // type_id means a scalar leaked into a traced slot — panic loudly rather
        // than index past the table. Never a silent conservative skip.
        if crate::gc::heap::gc_verify_armed() {
            assert!(
                (type_id as usize) < self.type_table().len(),
                "GC precise-layout violation (semi-space): traced slot points at \
                 {old:p} whose header type_id={type_id} is out of range \
                 (type_table len {}). A non-pointer reached a traced slot.",
                self.type_table().len(),
            );
        }
        let info = &self.type_table()[type_id as usize];
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

#[cfg(test)]
mod follow_forwarding_tests {
    use super::*;

    /// Header word with no forwarding bit set should pass through.
    #[test]
    fn passthrough_when_no_forwarding_bit() {
        let header: u64 = 0x0000_0000_0000_0042; // arbitrary type_id
        let header_ptr = &header as *const u64 as *const u8;
        let result = unsafe { follow_forwarding(header_ptr) };
        assert_eq!(result, header_ptr);
    }

    /// Header word with FORWARDING_BIT set returns the to-space pointer
    /// stored in the low 63 bits.
    #[test]
    fn follows_when_forwarding_bit_set() {
        // Build a "to-space" object with a normal header.
        let to_header: u64 = 0x0000_0000_0000_0007;
        let to_ptr = &to_header as *const u64 as *const u8;

        // Build a "from-space" header that points at it.
        let from_header: u64 = (to_ptr as u64) | FORWARDING_BIT;
        let from_ptr = &from_header as *const u64 as *const u8;

        let result = unsafe { follow_forwarding(from_ptr) };
        assert_eq!(result, to_ptr);

        // The header at the followed location is itself not forwarded.
        let followed_header = unsafe { *(result as *const u64) };
        assert_eq!(followed_header & FORWARDING_BIT, 0);
        assert_eq!(followed_header, 0x0000_0000_0000_0007);
    }

    /// Single-hop only: even if to-space happens to itself look forwarded,
    /// the function only does one hop. The debug assertion guards against
    /// chains in debug builds; release builds silently follow once.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "forwarding pointer chain"))]
    fn single_hop_only() {
        let leaf_header: u64 = 0x0000_0000_0000_0099;
        let leaf_ptr = &leaf_header as *const u64 as *const u8;

        let middle_header: u64 = (leaf_ptr as u64) | FORWARDING_BIT;
        let middle_ptr = &middle_header as *const u64 as *const u8;

        let root_header: u64 = (middle_ptr as u64) | FORWARDING_BIT;
        let root_ptr = &root_header as *const u64 as *const u8;

        // Debug build: the assertion fires because middle is forwarded.
        // Release: silently returns middle_ptr (one hop).
        let _ = unsafe { follow_forwarding(root_ptr) };
    }

    /// FORWARDING_BIT exposes the same constant `pub(crate)` callers
    /// inside the crate already use. Sanity check the public re-export.
    #[test]
    fn forwarding_bit_is_high_bit() {
        assert_eq!(FORWARDING_BIT, 1 << 63);
        assert_eq!(crate::gc::FORWARDING_BIT, FORWARDING_BIT);
    }
}
