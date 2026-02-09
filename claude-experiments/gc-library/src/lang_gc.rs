//! Layout-agnostic GC with callback-based tracing.
//!
//! This module provides a C API for garbage collection that makes no
//! assumptions about object layout, pointer tagging, or header format.
//! All object-model-specific operations are delegated to caller-provided
//! callbacks.
//!
//! # Design
//!
//! The GC is a pure memory manager + collector. The caller controls everything
//! about the object model:
//!
//! - **Marking**: `is_marked`/`set_mark` callbacks — the caller owns mark bits.
//! - **Sizing**: `get_size` callback — the caller knows object sizes from their headers.
//! - **Tracing**: `tracer` callback — the caller knows which fields are pointers.
//! - **Root enumeration**: `root_enumerator` callback — the caller knows where roots live.
//!
//! The visitor callback simply adds pointers to the mark worklist. It does no
//! filtering — if the caller calls `visit` for a slot, the GC trusts that `*slot`
//! is a valid GC-managed pointer (or null, which is skipped).
//!
//! # Contract
//!
//! - `get_size(obj)` must return the byte count originally passed to `gc_lib_custom_allocate`.
//! - All objects must be initialized (headers written) before GC can run.
//!   GC only runs inside `gc_lib_custom_allocate` (before the new allocation) or
//!   during explicit `gc_lib_custom_collect` calls, so all existing objects are initialized.
//! - The `tracer` and `root_enumerator` must only call `visit` for slots containing
//!   valid GC-managed pointers (or null).

use std::ffi::c_void;

use crate::gc::{LibcMemoryProvider, MemoryProvider};

// =============================================================================
// Callback Types
// =============================================================================

/// Visit callback: called for each pointer slot during root enumeration or tracing.
///
/// `slot` is the address where the pointer is stored (`void**`).
/// The GC reads `*slot` to find the pointed-to object, and may update `*slot`
/// if the object moves (copying/compacting GC).
pub type GcCustomVisitor = extern "C" fn(slot: *mut *mut c_void, gc_ctx: *mut c_void);

/// Root enumerator: called during collection to discover all GC roots.
///
/// `user_ctx` is the value passed to `gc_lib_custom_allocate`/`gc_lib_custom_collect`.
/// Must call `visit` for every root slot containing a GC-managed pointer.
pub type GcCustomRootEnumerator =
    extern "C" fn(user_ctx: *mut c_void, visit: GcCustomVisitor, gc_ctx: *mut c_void);

/// Object tracer: called during marking for each reachable object.
///
/// Must call `visit` for every pointer field in the object.
pub type GcCustomTracer =
    extern "C" fn(object: *mut c_void, visit: GcCustomVisitor, gc_ctx: *mut c_void);

/// Check if an object is marked.
pub type GcCustomIsMarked = extern "C" fn(object: *mut c_void) -> i32;

/// Set or clear the mark on an object.
pub type GcCustomSetMark = extern "C" fn(object: *mut c_void, marked: i32);

/// Get the total byte size of an object.
///
/// Must return the same `size` value that was passed to `gc_lib_custom_allocate`
/// when this object was allocated.
pub type GcCustomGetSize = extern "C" fn(object: *mut c_void, size_out: *mut usize);

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for creating a callback-based GC instance.
#[repr(C)]
pub struct GcCustomConfig {
    pub root_enumerator: GcCustomRootEnumerator,
    pub tracer: GcCustomTracer,
    pub is_marked: GcCustomIsMarked,
    pub set_mark: GcCustomSetMark,
    pub get_size: GcCustomGetSize,
    pub initial_heap: usize,
}

// =============================================================================
// Internal Free List
// =============================================================================

#[derive(Copy, Clone)]
struct FreeEntry {
    offset: usize,
    size: usize,
}

impl FreeEntry {
    fn end(&self) -> usize {
        self.offset + self.size
    }

    fn contains(&self, offset: usize) -> bool {
        offset >= self.offset && offset < self.end()
    }
}

struct FreeList {
    entries: Vec<FreeEntry>,
}

impl FreeList {
    fn new(initial: FreeEntry) -> Self {
        FreeList {
            entries: vec![initial],
        }
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if entry.size >= size {
                let offset = entry.offset;
                debug_assert!(offset % 8 == 0, "unaligned free list entry");
                entry.offset += size;
                entry.size -= size;
                if entry.size == 0 {
                    self.entries.remove(i);
                }
                return Some(offset);
            }
        }
        None
    }

    fn insert(&mut self, new_entry: FreeEntry) {
        let i = match self
            .entries
            .binary_search_by_key(&new_entry.offset, |e| e.offset)
        {
            Ok(i) | Err(i) => i,
        };

        // Try coalesce with previous
        if i > 0 && self.entries[i - 1].end() == new_entry.offset {
            let prev = i - 1;
            self.entries[prev].size += new_entry.size;
            // Try coalesce with next
            if prev + 1 < self.entries.len()
                && self.entries[prev].end() == self.entries[prev + 1].offset
            {
                self.entries[prev].size += self.entries[prev + 1].size;
                self.entries.remove(prev + 1);
            }
        } else {
            self.entries.insert(i, new_entry);
            // Try coalesce with next
            if i + 1 < self.entries.len()
                && self.entries[i].end() == self.entries[i + 1].offset
            {
                self.entries[i].size += self.entries[i + 1].size;
                self.entries.remove(i + 1);
            }
        }
    }

    fn find_containing(&self, offset: usize) -> Option<&FreeEntry> {
        self.entries.iter().find(|e| e.contains(offset))
    }
}

// =============================================================================
// GC Handle
// =============================================================================

/// Opaque handle to a callback-based GC instance.
///
/// Uses mark-and-sweep collection. The caller provides all object-model-specific
/// operations through the [`GcCustomConfig`] callbacks.
pub struct GcCustomHandle {
    config: GcCustomConfig,
    heap_base: *mut u8,
    heap_committed: usize,
    heap_reserved: usize,
    free_list: FreeList,
    highmark: usize,
    total_allocated: usize,
    gc_threshold: usize,
    memory: LibcMemoryProvider,
}

unsafe impl Send for GcCustomHandle {}

/// Context passed through the visitor callback during the mark phase.
struct MarkContext {
    worklist: Vec<*mut c_void>,
}

impl GcCustomHandle {
    fn new(config: GcCustomConfig) -> Option<Self> {
        let mut memory = LibcMemoryProvider::new();
        let page_size = memory.page_size();

        // Reserve a large virtual address range (up to ~4TB on 4KB pages)
        let max_pages = 1_000_000;
        let max_heap = max_pages * page_size;
        let heap_base = memory.allocate_region(max_heap)?;

        // Commit initial pages
        let initial = if config.initial_heap > 0 {
            // Round up to page size
            ((config.initial_heap + page_size - 1) / page_size) * page_size
        } else {
            page_size * 1024 // ~4MB default
        };

        if !memory.commit(heap_base, initial) {
            return None;
        }

        let gc_threshold = initial / 2;

        Some(GcCustomHandle {
            config,
            heap_base,
            heap_committed: initial,
            heap_reserved: max_heap,
            free_list: FreeList::new(FreeEntry {
                offset: 0,
                size: initial,
            }),
            highmark: 0,
            total_allocated: 0,
            gc_threshold,
            memory,
        })
    }

    fn allocate(&mut self, size: usize, user_ctx: *mut c_void) -> *mut c_void {
        // Align to 8 bytes
        let alloc_size = (size + 7) & !7;

        // Check if we should collect before allocating
        if self.total_allocated + alloc_size > self.gc_threshold {
            self.collect_inner(user_ctx);
        }

        // Try to allocate
        if let Some(ptr) = self.try_alloc(alloc_size) {
            return ptr;
        }

        // GC and retry
        self.collect_inner(user_ctx);
        if let Some(ptr) = self.try_alloc(alloc_size) {
            return ptr;
        }

        // Grow and retry
        self.grow();
        if let Some(ptr) = self.try_alloc(alloc_size) {
            return ptr;
        }

        // Grow again and final retry
        self.grow();
        self.try_alloc(alloc_size).unwrap_or(std::ptr::null_mut())
    }

    fn try_alloc(&mut self, alloc_size: usize) -> Option<*mut c_void> {
        let offset = self.free_list.allocate(alloc_size)?;

        if offset + alloc_size > self.highmark {
            self.highmark = offset + alloc_size;
        }

        self.total_allocated += alloc_size;

        let ptr = unsafe { self.heap_base.add(offset) };

        // Zero the allocation
        unsafe {
            std::ptr::write_bytes(ptr, 0, alloc_size);
        }

        Some(ptr as *mut c_void)
    }

    fn collect_inner(&mut self, user_ctx: *mut c_void) {
        self.mark(user_ctx);
        self.sweep();
        // Adjust GC threshold: collect when live data doubles, but at least 1/4 of heap
        self.gc_threshold = (self.total_allocated * 2).max(self.heap_committed / 4);
    }

    fn mark(&self, user_ctx: *mut c_void) {
        let root_enumerator = self.config.root_enumerator;
        let tracer = self.config.tracer;
        let is_marked = self.config.is_marked;
        let set_mark = self.config.set_mark;

        let mut ctx = MarkContext {
            worklist: Vec::with_capacity(256),
        };

        /// Visitor function passed to root_enumerator and tracer callbacks.
        ///
        /// Reads `*slot` and adds the pointer to the mark worklist.
        /// The caller is responsible for only calling visit on slots that
        /// contain valid GC-managed pointers. Null pointers are skipped.
        extern "C" fn visit_slot(slot: *mut *mut c_void, gc_ctx: *mut c_void) {
            let ctx = unsafe { &mut *(gc_ctx as *mut MarkContext) };
            let ptr = unsafe { *slot };
            if !ptr.is_null() {
                ctx.worklist.push(ptr);
            }
        }

        // Phase 1: Enumerate roots
        let gc_ctx = &mut ctx as *mut MarkContext as *mut c_void;
        (root_enumerator)(user_ctx, visit_slot, gc_ctx);

        // Phase 2: Trace from roots (DFS via explicit worklist)
        let heap_start = self.heap_base as usize;
        let heap_end = heap_start + self.heap_committed;
        while let Some(obj) = ctx.worklist.pop() {
            // Skip pointers outside the GC heap (e.g., raw C strings on the stack)
            let addr = obj as usize;
            if addr < heap_start || addr >= heap_end {
                continue;
            }
            if (is_marked)(obj) != 0 {
                continue;
            }
            (set_mark)(obj, 1);

            // Trace this object's pointer fields
            let gc_ctx = &mut ctx as *mut MarkContext as *mut c_void;
            (tracer)(obj, visit_slot, gc_ctx);
        }
    }

    fn sweep(&mut self) {
        let mut offset: usize = 0;
        let mut new_allocated: usize = 0;
        let is_marked = self.config.is_marked;
        let set_mark = self.config.set_mark;
        let get_size = self.config.get_size;

        while offset < self.highmark {
            // Skip free list entries
            if let Some(entry) = self.free_list.find_containing(offset) {
                offset = entry.end();
                continue;
            }

            let obj_ptr = unsafe { self.heap_base.add(offset) as *mut c_void };

            // Ask the caller for the object's size
            let mut obj_size: usize = 0;
            (get_size)(obj_ptr, &mut obj_size);

            if obj_size == 0 {
                panic!(
                    "gc_lib_custom: get_size returned 0 at heap offset {}",
                    offset
                );
            }

            // Align to match allocation alignment
            let alloc_size = (obj_size + 7) & !7;

            if (is_marked)(obj_ptr) != 0 {
                // Live object — clear mark for next cycle
                (set_mark)(obj_ptr, 0);
                new_allocated += alloc_size;
            } else {
                // Dead object — reclaim
                self.free_list.insert(FreeEntry {
                    offset,
                    size: alloc_size,
                });
            }

            offset += alloc_size;
        }

        self.total_allocated = new_allocated;
    }

    fn grow(&mut self) {
        let new_committed = (self.heap_committed * 2).min(self.heap_reserved);
        if new_committed > self.heap_committed {
            if self.memory.commit(self.heap_base, new_committed) {
                let old = self.heap_committed;
                self.heap_committed = new_committed;
                self.free_list.insert(FreeEntry {
                    offset: old,
                    size: new_committed - old,
                });
                self.gc_threshold =
                    (self.total_allocated * 2).max(new_committed / 4);
            }
        }
    }

    fn should_collect(&self) -> bool {
        self.total_allocated >= self.gc_threshold
    }
}

// =============================================================================
// C API
// =============================================================================

/// Create a callback-based GC instance.
///
/// Returns NULL on failure (e.g., mmap failed).
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_create(config: GcCustomConfig) -> *mut GcCustomHandle {
    match GcCustomHandle::new(config) {
        Some(handle) => Box::into_raw(Box::new(handle)),
        None => std::ptr::null_mut(),
    }
}

/// Destroy a callback-based GC instance and free its resources.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_destroy(gc: *mut GcCustomHandle) {
    if !gc.is_null() {
        unsafe {
            drop(Box::from_raw(gc));
        }
    }
}

/// Allocate `size` bytes of zeroed memory, managed by the GC.
///
/// May trigger collection (calling `root_enumerator` + `tracer`).
/// `user_ctx` is passed to `root_enumerator`.
/// Returns NULL on out-of-memory.
///
/// The caller must initialize the object (write headers) before the next
/// allocation or collection, so that `get_size` returns the correct value.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_allocate(
    gc: *mut GcCustomHandle,
    size: usize,
    user_ctx: *mut c_void,
) -> *mut c_void {
    if gc.is_null() {
        return std::ptr::null_mut();
    }
    let gc = unsafe { &mut *gc };
    gc.allocate(size, user_ctx)
}

/// Run an explicit garbage collection cycle.
///
/// `user_ctx` is passed to `root_enumerator`.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_collect(gc: *mut GcCustomHandle, user_ctx: *mut c_void) {
    if gc.is_null() {
        return;
    }
    let gc = unsafe { &mut *gc };
    gc.collect_inner(user_ctx);
}

/// Write barrier (for future generational support).
///
/// Currently a no-op for the mark-and-sweep collector.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_write_barrier(
    _gc: *mut GcCustomHandle,
    _object: *mut c_void,
    _new_value: *mut c_void,
) {
    // No-op for mark-and-sweep
}

/// Check if collection is recommended (for safepoints).
///
/// Returns 1 if the GC recommends running a collection, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_should_collect(gc: *mut GcCustomHandle) -> i32 {
    if gc.is_null() {
        return 0;
    }
    let gc = unsafe { &*gc };
    if gc.should_collect() { 1 } else { 0 }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test object model
    //
    // Layout:
    //   offset 0: u32 mark_flag (0=unmarked, 1=marked)
    //   offset 4: u32 num_fields
    //   offset 8+: pointer fields (void*, 8 bytes each)
    //
    // Total size = 8 + 8 * num_fields
    // -------------------------------------------------------------------------

    extern "C" fn test_is_marked(obj: *mut c_void) -> i32 {
        unsafe { *(obj as *const u32) as i32 }
    }

    extern "C" fn test_set_mark(obj: *mut c_void, marked: i32) {
        unsafe {
            *(obj as *mut u32) = marked as u32;
        }
    }

    extern "C" fn test_get_size(obj: *mut c_void, size_out: *mut usize) {
        let num_fields = unsafe { *((obj as *const u32).add(1)) } as usize;
        unsafe {
            *size_out = 8 + 8 * num_fields;
        }
    }

    extern "C" fn test_tracer(obj: *mut c_void, visit: GcCustomVisitor, gc_ctx: *mut c_void) {
        let num_fields = unsafe { *((obj as *const u32).add(1)) } as usize;
        let fields = unsafe { (obj as *mut u8).add(8) as *mut *mut c_void };
        for i in 0..num_fields {
            let slot = unsafe { fields.add(i) };
            visit(slot, gc_ctx);
        }
    }

    // Root set for tests: fixed-size array of pointer slots
    #[repr(C)]
    struct TestRoots {
        slots: [*mut c_void; 16],
        count: usize,
    }

    impl TestRoots {
        fn new() -> Self {
            TestRoots {
                slots: [std::ptr::null_mut(); 16],
                count: 0,
            }
        }

        fn add(&mut self, ptr: *mut c_void) {
            assert!(self.count < 16);
            self.slots[self.count] = ptr;
            self.count += 1;
        }
    }

    extern "C" fn test_root_enumerator(
        user_ctx: *mut c_void,
        visit: GcCustomVisitor,
        gc_ctx: *mut c_void,
    ) {
        let roots = unsafe { &mut *(user_ctx as *mut TestRoots) };
        for i in 0..roots.count {
            visit(&mut roots.slots[i] as *mut *mut c_void, gc_ctx);
        }
    }

    extern "C" fn empty_root_enumerator(
        _user_ctx: *mut c_void,
        _visit: GcCustomVisitor,
        _gc_ctx: *mut c_void,
    ) {
        // No roots
    }

    fn test_config() -> GcCustomConfig {
        GcCustomConfig {
            root_enumerator: empty_root_enumerator,
            tracer: test_tracer,
            is_marked: test_is_marked,
            set_mark: test_set_mark,
            get_size: test_get_size,
            initial_heap: 64 * 1024, // 64KB for tests
        }
    }

    fn init_test_object(ptr: *mut c_void, num_fields: u32) {
        unsafe {
            // mark_flag at offset 0 — already zeroed
            // num_fields at offset 4
            *((ptr as *mut u32).add(1)) = num_fields;
        }
    }

    fn set_field(obj: *mut c_void, index: usize, value: *mut c_void) {
        unsafe {
            let fields = (obj as *mut u8).add(8) as *mut *mut c_void;
            *fields.add(index) = value;
        }
    }

    fn get_field(obj: *mut c_void, index: usize) -> *mut c_void {
        unsafe {
            let fields = (obj as *const u8).add(8) as *const *mut c_void;
            *fields.add(index)
        }
    }

    // ---- Tests ----

    #[test]
    fn test_create_destroy() {
        let gc = gc_lib_custom_create(test_config());
        assert!(!gc.is_null());
        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_allocate_returns_zeroed_memory() {
        let gc = gc_lib_custom_create(test_config());

        let ptr = gc_lib_custom_allocate(gc, 32, std::ptr::null_mut());
        assert!(!ptr.is_null());

        // Check memory is zeroed
        let bytes = ptr as *const u8;
        for i in 0..32 {
            assert_eq!(unsafe { *bytes.add(i) }, 0, "byte {} not zero", i);
        }

        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_allocate_multiple_no_overlap() {
        let gc = gc_lib_custom_create(test_config());

        // Allocate objects and immediately init so get_size works
        let a = gc_lib_custom_allocate(gc, 24, std::ptr::null_mut()); // 8 + 8*2
        init_test_object(a, 2);
        let b = gc_lib_custom_allocate(gc, 24, std::ptr::null_mut());
        init_test_object(b, 2);
        let c = gc_lib_custom_allocate(gc, 24, std::ptr::null_mut());
        init_test_object(c, 2);

        assert!(!a.is_null());
        assert!(!b.is_null());
        assert!(!c.is_null());

        let addrs = [a as usize, b as usize, c as usize];
        for i in 0..3 {
            for j in (i + 1)..3 {
                let diff = addrs[i].abs_diff(addrs[j]);
                assert!(diff >= 24, "objects overlap: diff={}", diff);
            }
        }

        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_gc_collects_unreachable() {
        let config = GcCustomConfig {
            root_enumerator: test_root_enumerator,
            ..test_config()
        };
        let gc = gc_lib_custom_create(config);
        let mut roots = TestRoots::new();

        let obj_size = 8 + 8 * 2; // header(8) + 2 pointer fields
        let a = gc_lib_custom_allocate(gc, obj_size, &mut roots as *mut TestRoots as *mut c_void);
        init_test_object(a, 2);
        let b = gc_lib_custom_allocate(gc, obj_size, &mut roots as *mut TestRoots as *mut c_void);
        init_test_object(b, 2);

        // Only root 'a'
        roots.add(a);

        // Collect — 'b' should be reclaimed
        gc_lib_custom_collect(gc, &mut roots as *mut TestRoots as *mut c_void);

        // 'a' should still be valid (its mark was cleared by sweep)
        assert_eq!(test_is_marked(a), 0);

        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_gc_traces_object_graph() {
        let config = GcCustomConfig {
            root_enumerator: test_root_enumerator,
            ..test_config()
        };
        let gc = gc_lib_custom_create(config);
        let mut roots = TestRoots::new();

        let obj_size = 8 + 8 * 2; // header(8) + 2 fields

        // Allocate a chain: root -> a -> b -> c
        let a = gc_lib_custom_allocate(gc, obj_size, std::ptr::null_mut());
        init_test_object(a, 2);
        let b = gc_lib_custom_allocate(gc, obj_size, std::ptr::null_mut());
        init_test_object(b, 2);
        let c = gc_lib_custom_allocate(gc, obj_size, std::ptr::null_mut());
        init_test_object(c, 2);

        // Wire up: a.field[0] = b, b.field[0] = c
        set_field(a, 0, b);
        set_field(b, 0, c);

        // Allocate an unreachable object
        let unreachable = gc_lib_custom_allocate(gc, obj_size, std::ptr::null_mut());
        init_test_object(unreachable, 2);

        // Root only 'a'
        roots.add(a);

        gc_lib_custom_collect(gc, &mut roots as *mut TestRoots as *mut c_void);

        // a, b, c should survive — verify the object graph is intact
        assert_eq!(get_field(a, 0), b);
        assert_eq!(get_field(b, 0), c);

        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_gc_reclaims_and_reuses_memory() {
        let config = GcCustomConfig {
            root_enumerator: test_root_enumerator,
            initial_heap: 4096, // Small heap to force reuse
            ..test_config()
        };
        let gc = gc_lib_custom_create(config);
        let mut roots = TestRoots::new();

        let obj_size = 8 + 8 * 2;

        // Fill up heap with unreachable objects
        for _ in 0..20 {
            let obj = gc_lib_custom_allocate(
                gc,
                obj_size,
                &mut roots as *mut TestRoots as *mut c_void,
            );
            assert!(!obj.is_null());
            init_test_object(obj, 2);
        }

        // No roots — everything is garbage
        roots.count = 0;
        gc_lib_custom_collect(gc, &mut roots as *mut TestRoots as *mut c_void);

        // Should be able to allocate again in reclaimed space
        for _ in 0..20 {
            let obj = gc_lib_custom_allocate(
                gc,
                obj_size,
                &mut roots as *mut TestRoots as *mut c_void,
            );
            assert!(!obj.is_null());
            init_test_object(obj, 2);
        }

        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_should_collect() {
        let config = GcCustomConfig {
            initial_heap: 4096,
            ..test_config()
        };
        let gc = gc_lib_custom_create(config);

        // Initially should not need collection
        assert_eq!(gc_lib_custom_should_collect(gc), 0);

        // Allocate until threshold is exceeded
        let gc_ref = unsafe { &mut *gc };
        let threshold = gc_ref.gc_threshold;
        let obj_size = 64;
        let mut allocated = 0;
        while allocated < threshold {
            let ptr = gc_ref.try_alloc(obj_size);
            assert!(ptr.is_some());
            // Init each allocation so get_size works
            init_test_object(ptr.unwrap(), (obj_size as u32 - 8) / 8);
            allocated += obj_size;
        }

        assert_eq!(gc_lib_custom_should_collect(gc), 1);

        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_write_barrier_noop() {
        let gc = gc_lib_custom_create(test_config());
        // Should not crash
        gc_lib_custom_write_barrier(gc, std::ptr::null_mut(), std::ptr::null_mut());
        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_null_gc_handle() {
        let null: *mut GcCustomHandle = std::ptr::null_mut();
        assert!(gc_lib_custom_allocate(null, 32, std::ptr::null_mut()).is_null());
        gc_lib_custom_collect(null, std::ptr::null_mut());
        gc_lib_custom_write_barrier(null, std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(gc_lib_custom_should_collect(null), 0);
        gc_lib_custom_destroy(null); // should not crash
    }

    #[test]
    fn test_various_allocation_sizes() {
        let gc = gc_lib_custom_create(test_config());

        // Test various sizes — all get initialized so get_size works
        let sizes: &[(usize, u32)] = &[
            (8, 0),   // header only, 0 fields
            (16, 1),  // 1 field
            (24, 2),  // 2 fields
            (32, 3),  // 3 fields
            (64, 7),  // 7 fields
            (128, 15), // 15 fields
            (256, 31), // 31 fields
            (1024, 127), // 127 fields
        ];
        for &(size, nfields) in sizes {
            let ptr = gc_lib_custom_allocate(gc, size, std::ptr::null_mut());
            assert!(!ptr.is_null(), "allocation of {} bytes failed", size);
            init_test_object(ptr, nfields);

            // Verify zeroed (init only writes at offsets 0 and 4)
            let bytes = ptr as *const u8;
            for i in 8..size {
                assert_eq!(
                    unsafe { *bytes.add(i) },
                    0,
                    "byte {} not zero for size {}",
                    i,
                    size
                );
            }
        }

        gc_lib_custom_destroy(gc);
    }

    #[test]
    fn test_allocation_pointer_is_direct() {
        // Verify there's no hidden prefix — the returned pointer IS the object
        let gc = gc_lib_custom_create(test_config());

        let a = gc_lib_custom_allocate(gc, 24, std::ptr::null_mut());
        init_test_object(a, 2);
        let b = gc_lib_custom_allocate(gc, 24, std::ptr::null_mut());
        init_test_object(b, 2);

        // Objects should be exactly 24 bytes apart (no prefix overhead)
        let diff = (b as usize) - (a as usize);
        assert_eq!(diff, 24, "expected 24-byte spacing, got {}", diff);

        gc_lib_custom_destroy(gc);
    }
}
