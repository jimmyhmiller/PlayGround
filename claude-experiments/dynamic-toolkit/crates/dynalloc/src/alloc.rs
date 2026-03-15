use core::cell::Cell;

use dynobj::{
    Compact, ObjHeader, TypeInfo, VarLenKind, init_header, read_type_info, read_varlen_count,
    write_varlen_count,
};

// ─── Traits ──────────────────────────────────────────────────────────

/// Allocator trait for heap objects.
///
/// Uses `&self` (not `&mut self`) so that the GC can hold a reference
/// during scanning while allocation is still possible from the same
/// reference. Interior mutability via `Cell` (single-threaded, no atomics).
///
/// Returns `*mut u8`, null on failure. The allocator does NOT write the
/// header — use [`alloc_obj`] for alloc + header init + varlen count.
pub trait Alloc {
    /// Allocate space for an object described by `info` with `varlen_len`
    /// variable-length elements. Returns a zeroed pointer, or null if
    /// the allocation cannot be satisfied.
    fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8;
}

/// Heap walking trait for GC.
///
/// Separate from `Alloc` because not all allocators support iteration,
/// and heap walking is a GC-time concern.
pub trait HeapWalker {
    /// Walk all live objects in the heap, calling `visitor` for each one.
    ///
    /// # Safety
    /// All objects in the heap region must be valid (headers initialized,
    /// varlen counts written). Typically called during GC with mutators stopped.
    unsafe fn walk(&self, visitor: &mut dyn FnMut(*mut u8, &'static TypeInfo));
}

// ─── alloc_obj helper ────────────────────────────────────────────────

/// Allocate an object, initialize its header, and write the varlen count.
///
/// Returns null if the underlying allocator returns null.
///
/// # Safety
/// - `allocator` must return properly aligned, zeroed memory.
/// - `info` must accurately describe the object layout.
pub unsafe fn alloc_obj<H: ObjHeader>(
    allocator: &dyn Alloc,
    info: &'static TypeInfo,
    varlen_len: usize,
) -> *mut u8 {
    let ptr = allocator.alloc(info, varlen_len);
    if ptr.is_null() {
        return ptr;
    }
    unsafe {
        init_header::<H>(ptr, info as *const TypeInfo);
        if info.varlen != VarLenKind::None {
            write_varlen_count(ptr, info, varlen_len);
        }
    }
    ptr
}

// ─── BumpAllocator ───────────────────────────────────────────────────

/// A bump (linear) allocator for heap objects.
///
/// Allocates by bumping a cursor forward. No per-object deallocation —
/// the entire region is freed at once via `reset()` (after GC evacuation)
/// or on `Drop`.
///
/// Uses `Cell<usize>` for the cursor so `alloc(&self)` works without
/// `&mut self`. Single-threaded only (`!Sync`).
pub struct BumpAllocator {
    base: *mut u8,
    cursor: Cell<usize>,
    size: usize,
    type_info_offset: usize,
    owned: bool,
}

// Safety: BumpAllocator can be moved between threads (Send),
// but cannot be shared (&self across threads) because of Cell (!Sync).
unsafe impl Send for BumpAllocator {}

impl BumpAllocator {
    /// Create a new bump allocator that owns a region of `size` bytes.
    ///
    /// The header type `H` determines the `type_info_offset` used by
    /// the heap walker.
    pub fn new<H: ObjHeader>(size: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null(), "BumpAllocator: allocation failed");
        Self {
            base,
            cursor: Cell::new(0),
            size,
            type_info_offset: H::TYPE_INFO_OFFSET,
            owned: true,
        }
    }

    /// Wrap an externally-owned memory region as a bump allocator.
    ///
    /// # Safety
    /// - `base` must point to a valid, zeroed region of at least `size` bytes.
    /// - The region must remain valid for the lifetime of this allocator.
    /// - The caller is responsible for freeing the region.
    pub unsafe fn from_region<H: ObjHeader>(base: *mut u8, size: usize) -> Self {
        Self {
            base,
            cursor: Cell::new(0),
            size,
            type_info_offset: H::TYPE_INFO_OFFSET,
            owned: false,
        }
    }

    /// Reset the cursor to 0. After this, the entire region can be reused.
    ///
    /// Typically called after GC evacuation has copied all live objects out.
    pub fn reset(&self) {
        self.cursor.set(0);
    }

    /// Number of bytes currently used.
    pub fn used(&self) -> usize {
        self.cursor.get()
    }

    /// Number of bytes remaining.
    pub fn remaining(&self) -> usize {
        self.size - self.cursor.get()
    }

    /// Base pointer of the region.
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Total size of the region in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if a pointer falls within this allocator's region.
    pub fn contains(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let base_addr = self.base as usize;
        addr >= base_addr && addr < base_addr + self.size
    }

    /// The type_info_offset stored at construction.
    pub fn type_info_offset(&self) -> usize {
        self.type_info_offset
    }
}

impl Alloc for BumpAllocator {
    fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        let obj_size = info.allocation_size(varlen_len);
        let align = 1usize << info.align_log2;

        let cur = self.cursor.get();
        // Align cursor up
        let aligned = (cur + align - 1) & !(align - 1);
        let new_cursor = aligned + obj_size;

        if new_cursor > self.size {
            return core::ptr::null_mut();
        }

        // Zero the allocation (the region may have been reset but contain
        // stale data from previous use).
        let ptr = unsafe { self.base.add(aligned) };
        unsafe {
            core::ptr::write_bytes(ptr, 0, obj_size);
        }

        self.cursor.set(new_cursor);
        ptr
    }
}

impl HeapWalker for BumpAllocator {
    unsafe fn walk(&self, visitor: &mut dyn FnMut(*mut u8, &'static TypeInfo)) {
        let mut offset = 0usize;
        let used = self.cursor.get();

        while offset < used {
            let ptr = unsafe { self.base.add(offset) };
            let info = unsafe { read_type_info(ptr, self.type_info_offset) };

            // Compute actual object size. For varlen objects we need to read
            // the element count from the object.
            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(ptr, info) },
            };
            let obj_size = info.allocation_size(varlen_len);

            visitor(ptr, info);

            // Advance past this object, respecting alignment of the next one.
            // Since we don't know the next object's alignment here, we advance
            // by at least obj_size and align to 8 (minimum alignment).
            let align = 1usize << info.align_log2;
            offset = ((offset + obj_size) + align - 1) & !(align - 1);
        }
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        if self.owned {
            let layout = std::alloc::Layout::from_size_align(self.size, 8).unwrap();
            unsafe {
                std::alloc::dealloc(self.base, layout);
            }
        }
    }
}

// ─── AtomicBumpAllocator ────────────────────────────────────────────

use core::sync::atomic::{AtomicUsize, Ordering};

/// Thread-safe bump allocator using an atomic cursor.
///
/// Unlike `BumpAllocator` (which uses `Cell` and is `!Sync`), this
/// allocator uses `AtomicUsize` for the cursor and can be shared
/// across threads. Used for:
/// - Shared from-space allocation (multiple mutator threads)
/// - To-space allocation during parallel GC copying
///
/// Uses `fetch_add` with a CAS retry loop for alignment.
pub struct AtomicBumpAllocator {
    base: *mut u8,
    cursor: AtomicUsize,
    size: usize,
    type_info_offset: usize,
    owned: bool,
}

// Safety: AtomicBumpAllocator is designed for cross-thread use.
// The base pointer is immutable after construction, and the cursor
// uses atomic operations.
unsafe impl Send for AtomicBumpAllocator {}
unsafe impl Sync for AtomicBumpAllocator {}

impl AtomicBumpAllocator {
    /// Create a new atomic bump allocator that owns a region of `size` bytes.
    pub fn new<H: ObjHeader>(size: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null(), "AtomicBumpAllocator: allocation failed");
        Self {
            base,
            cursor: AtomicUsize::new(0),
            size,
            type_info_offset: H::TYPE_INFO_OFFSET,
            owned: true,
        }
    }

    /// Reset the cursor to 0, making the entire region available for reuse.
    ///
    /// Note: this does NOT zero the memory. Newly allocated objects will
    /// see stale data in uninitialized fields. Callers must initialize
    /// all GC-traceable fields before the next collection.
    ///
    /// # Safety
    /// Must only be called when no other thread is allocating.
    pub fn reset(&self) {
        self.cursor.store(0, Ordering::Release);
    }

    /// Number of bytes currently used.
    pub fn used(&self) -> usize {
        self.cursor.load(Ordering::Acquire)
    }

    /// Number of bytes remaining.
    pub fn remaining(&self) -> usize {
        self.size - self.cursor.load(Ordering::Acquire)
    }

    /// Base pointer of the region.
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Total size of the region in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if a pointer falls within this allocator's region.
    pub fn contains(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let base_addr = self.base as usize;
        addr >= base_addr && addr < base_addr + self.size
    }

    /// The type_info_offset stored at construction.
    pub fn type_info_offset(&self) -> usize {
        self.type_info_offset
    }
}

impl Alloc for AtomicBumpAllocator {
    fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        let obj_size = info.allocation_size(varlen_len);
        let align = 1usize << info.align_log2;

        // CAS loop to atomically bump the cursor with alignment
        loop {
            let cur = self.cursor.load(Ordering::Relaxed);
            let aligned = (cur + align - 1) & !(align - 1);
            let new_cursor = aligned + obj_size;

            if new_cursor > self.size {
                return core::ptr::null_mut();
            }

            match self.cursor.compare_exchange_weak(
                cur,
                new_cursor,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let ptr = unsafe { self.base.add(aligned) };
                    unsafe {
                        core::ptr::write_bytes(ptr, 0, obj_size);
                    }
                    return ptr;
                }
                Err(_) => continue, // Another thread won, retry
            }
        }
    }
}

impl HeapWalker for AtomicBumpAllocator {
    unsafe fn walk(&self, visitor: &mut dyn FnMut(*mut u8, &'static TypeInfo)) {
        let mut offset = 0usize;
        let used = self.cursor.load(Ordering::Acquire);

        while offset < used {
            let ptr = unsafe { self.base.add(offset) };
            let info = unsafe { read_type_info(ptr, self.type_info_offset) };

            let varlen_len = match info.varlen {
                VarLenKind::None => 0,
                _ => unsafe { read_varlen_count(ptr, info) },
            };
            let obj_size = info.allocation_size(varlen_len);

            visitor(ptr, info);

            let align = 1usize << info.align_log2;
            offset = ((offset + obj_size) + align - 1) & !(align - 1);
        }
    }
}

impl Drop for AtomicBumpAllocator {
    fn drop(&mut self) {
        if self.owned {
            let layout = std::alloc::Layout::from_size_align(self.size, 8).unwrap();
            unsafe {
                std::alloc::dealloc(self.base, layout);
            }
        }
    }
}

// ─── FFI ─────────────────────────────────────────────────────────────

/// FFI: allocate from a bump allocator (raw, no header init).
///
/// # Safety
/// - `allocator` must point to a valid `BumpAllocator`.
/// - `info` must point to a valid `TypeInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bump_alloc(
    allocator: *const BumpAllocator,
    info: *const TypeInfo,
    varlen_len: usize,
) -> *mut u8 {
    let allocator = unsafe { &*allocator };
    let info = unsafe { &*info };
    // We need a 'static reference. TypeInfo descriptors are always static
    // constants in practice, so this is safe at FFI boundaries.
    let info: &'static TypeInfo = unsafe { &*(info as *const TypeInfo) };
    allocator.alloc(info, varlen_len)
}

/// FFI: allocate from a bump allocator and initialize a Compact header.
///
/// # Safety
/// - `allocator` must point to a valid `BumpAllocator`.
/// - `info` must point to a valid, static `TypeInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bump_alloc_init_compact(
    allocator: *const BumpAllocator,
    info: *const TypeInfo,
    varlen_len: usize,
) -> *mut u8 {
    let allocator = unsafe { &*allocator };
    let info: &'static TypeInfo = unsafe { &*(info as *const TypeInfo) };
    unsafe { alloc_obj::<Compact>(allocator, info, varlen_len) }
}

/// FFI: reset a bump allocator's cursor to 0.
///
/// # Safety
/// `allocator` must point to a valid `BumpAllocator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bump_reset(allocator: *const BumpAllocator) {
    let allocator = unsafe { &*allocator };
    allocator.reset();
}
