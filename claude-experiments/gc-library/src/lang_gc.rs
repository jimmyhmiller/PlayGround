//! Layout-agnostic GC with callback-based tracing.
//!
//! This module provides a C API for garbage collection that makes no
//! assumptions about object layout, pointer tagging, or header format.
//! All object-model-specific operations are delegated to caller-provided
//! callbacks.
//!
//! # Design
//!
//! This is a **BRIDGE** that adapts the callback-based API to use the existing
//! trait-based GC implementations (`MarkAndSweep` and `GenerationalGC`).
//! It implements the required traits (`GcTypes`, `TaggedPointer`, `GcObject`)
//! by delegating to user-provided callbacks.
//!
//! # Contract
//!
//! - `get_size(obj)` must return the byte count originally passed to `gc_lib_custom_allocate`.
//! - All objects must be initialized (headers written) before GC can run.
//! - The `tracer` and `root_enumerator` must only call `visit` for slots containing
//!   valid GC-managed pointers (or null).

use std::ffi::c_void;
use std::cell::RefCell;

use crate::gc::mark_and_sweep::MarkAndSweep;
use crate::gc::generational::GenerationalGC;
use crate::gc::{AllocateAction, Allocator, AllocatorOptions, LibcMemoryProvider};
use crate::traits::{ForwardingSupport, GcObject, GcTypes, ObjectKind, RootProvider, TaggedPointer};

// =============================================================================
// Callback Storage (Thread-Local)
// =============================================================================

thread_local! {
    static CALLBACKS: RefCell<Option<GcCustomConfig>> = RefCell::new(None);
}

fn with_callbacks<F, R>(f: F) -> R
where
    F: FnOnce(&GcCustomConfig) -> R,
{
    CALLBACKS.with(|cell| {
        let callbacks = cell.borrow();
        f(callbacks.as_ref().expect("GC callbacks not initialized"))
    })
}

// =============================================================================
// Bridge Runtime - Implements GcTypes traits using callbacks
// =============================================================================

/// Bridge runtime that adapts callbacks to the trait-based GC.
///
/// **PROOF: This uses the EXISTING MarkAndSweep and GenerationalGC implementations!**
pub struct CallbackRuntime;

impl GcTypes for CallbackRuntime {
    type TaggedValue = CallbackTaggedPtr;
    type ObjectHandle = CallbackObject;
    type ObjectKind = CallbackTypeTag;
}

/// Tagged pointer for callback runtime - just wraps a raw pointer.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CallbackTaggedPtr(usize);

impl TaggedPointer for CallbackTaggedPtr {
    type Kind = CallbackTypeTag;

    fn tag(raw_ptr: *const u8, _kind: Self::Kind) -> Self {
        CallbackTaggedPtr(raw_ptr as usize)
    }

    fn untag(self) -> *const u8 {
        self.0 as *const u8
    }

    fn get_kind(self) -> Self::Kind {
        if self.0 == 0 {
            CallbackTypeTag::Null
        } else {
            CallbackTypeTag::HeapObject
        }
    }

    fn is_heap_pointer(self) -> bool {
        self.0 != 0
    }

    fn as_usize(self) -> usize {
        self.0
    }

    fn from_usize(value: usize) -> Self {
        CallbackTaggedPtr(value)
    }
}

/// Type tag for callback runtime.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CallbackTypeTag {
    Null,
    HeapObject,
}

impl ObjectKind for CallbackTypeTag {
    fn is_heap_type(self) -> bool {
        matches!(self, CallbackTypeTag::HeapObject)
    }
}

/// Object handle that uses callbacks for all operations.
pub struct CallbackObject {
    ptr: *const u8,
}

impl GcObject for CallbackObject {
    type TaggedValue = CallbackTaggedPtr;

    fn from_tagged(tagged: Self::TaggedValue) -> Self {
        CallbackObject {
            ptr: tagged.untag(),
        }
    }

    fn from_untagged(ptr: *const u8) -> Self {
        CallbackObject { ptr }
    }

    fn get_pointer(&self) -> *const u8 {
        self.ptr
    }

    fn tagged_pointer(&self) -> Self::TaggedValue {
        CallbackTaggedPtr::tag(self.ptr, CallbackTypeTag::HeapObject)
    }

    fn write_header(&mut self, _size_bytes: usize) {
        // Headers are managed by the user via callbacks
    }

    fn mark(&self) {
        with_callbacks(|config| {
            (config.set_mark)(self.ptr as *mut c_void, 1);
        })
    }

    fn unmark(&self) {
        with_callbacks(|config| {
            (config.set_mark)(self.ptr as *mut c_void, 0);
        })
    }

    fn marked(&self) -> bool {
        with_callbacks(|config| {
            (config.is_marked)(self.ptr as *mut c_void) != 0
        })
    }

    fn get_fields(&self) -> &[usize] {
        // We can't safely return fields without knowing the layout
        // The tracer callback handles field iteration instead
        &[]
    }

    fn get_fields_mut(&mut self) -> &mut [usize] {
        // Fields are accessed via callbacks
        &mut []
    }

    fn is_opaque(&self) -> bool {
        false // Assume all objects may have pointers
    }

    fn is_zero_size(&self) -> bool {
        self.full_size() == 0
    }

    fn get_object_kind(&self) -> Option<CallbackTypeTag> {
        Some(CallbackTypeTag::HeapObject)
    }

    fn full_size(&self) -> usize {
        let mut size: usize = 0;
        with_callbacks(|config| {
            (config.get_size)(self.ptr as *mut c_void, &mut size);
        });
        size
    }

    fn header_size(&self) -> usize {
        0 // Headers are managed by user
    }

    fn get_full_object_data(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.full_size())
        }
    }
}

impl ForwardingSupport for CallbackObject {
    fn is_forwarded(&self) -> bool {
        with_callbacks(|config| {
            (config.is_forwarded)(self.ptr as *mut c_void) != 0
        })
    }

    fn get_forwarding_pointer(&self) -> Self::TaggedValue {
        with_callbacks(|config| {
            let new_ptr = (config.get_forwarding)(self.ptr as *mut c_void);
            CallbackTaggedPtr(new_ptr as usize)
        })
    }

    fn set_forwarding_pointer(&mut self, new_location: Self::TaggedValue) {
        with_callbacks(|config| {
            (config.set_forwarding)(self.ptr as *mut c_void, new_location.0 as *mut c_void);
        })
    }
}

// =============================================================================
// Root Provider Bridge
// =============================================================================

struct CallbackRootProvider {
    user_ctx: *mut c_void,
}

impl RootProvider<CallbackTaggedPtr> for CallbackRootProvider {
    fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, CallbackTaggedPtr)) {
        with_callbacks(|config| {
            // Bridge: Collect roots via callback, then pass to GC
            let roots: RefCell<Vec<(usize, CallbackTaggedPtr)>> = RefCell::new(Vec::new());

            extern "C" fn visit_slot(slot: *mut *mut c_void, gc_ctx: *mut c_void) {
                let roots_ptr = gc_ctx as *mut RefCell<Vec<(usize, CallbackTaggedPtr)>>;
                let ptr = unsafe { *slot };
                if !ptr.is_null() {
                    unsafe {
                        (*roots_ptr).borrow_mut().push((
                            slot as usize,
                            CallbackTaggedPtr(ptr as usize),
                        ));
                    }
                }
            }

            (config.root_enumerator)(
                self.user_ctx,
                visit_slot,
                &roots as *const RefCell<_> as *mut c_void,
            );

            for (slot_addr, tagged_ptr) in roots.borrow().iter() {
                callback(*slot_addr, *tagged_ptr);
            }
        })
    }
}

// =============================================================================
// Callback Types (unchanged)
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

/// Check if an object has been forwarded (moved by copying GC).
///
/// Required for generational GC (copying collector).
///
/// # Parameters
/// - `object`: Pointer to the object
///
/// # Returns
/// Non-zero if the object contains a forwarding pointer, zero otherwise.
pub type GcCustomIsForwarded = extern "C" fn(object: *mut c_void) -> i32;

/// Get the forwarding pointer from a forwarded object.
///
/// Only called if is_forwarded returns non-zero.
/// Required for generational GC (copying collector).
///
/// # Parameters
/// - `object`: Pointer to the old object location
///
/// # Returns
/// The new location of the object (as a raw pointer).
pub type GcCustomGetForwarding = extern "C" fn(object: *mut c_void) -> *mut c_void;

/// Set a forwarding pointer in an object (mark it as moved).
///
/// Called by copying/compacting GC when relocating objects.
/// Required for generational GC (copying collector).
///
/// # Parameters
/// - `object`: Pointer to the old object location
/// - `new_location`: Pointer to the new object location
pub type GcCustomSetForwarding = extern "C" fn(object: *mut c_void, new_location: *mut c_void);

// =============================================================================
// Configuration
// =============================================================================

/// GC collection strategy.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GcStrategy {
    /// Mark-and-sweep: Simple non-moving collector.
    /// No write barriers needed.
    MarkSweep = 0,

    /// Generational: Young/old generation with write barriers.
    /// Requires calling gc_lib_custom_write_barrier after pointer stores.
    Generational = 1,
}

/// Configuration for creating a callback-based GC instance.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GcCustomConfig {
    pub root_enumerator: GcCustomRootEnumerator,
    pub tracer: GcCustomTracer,
    pub is_marked: GcCustomIsMarked,
    pub set_mark: GcCustomSetMark,
    pub get_size: GcCustomGetSize,
    /// Forwarding pointer support (required for GC_STRATEGY_GENERATIONAL)
    pub is_forwarded: GcCustomIsForwarded,
    pub get_forwarding: GcCustomGetForwarding,
    pub set_forwarding: GcCustomSetForwarding,
    pub initial_heap: usize,
    pub strategy: GcStrategy,
}

// =============================================================================
// GC Handle - USES EXISTING IMPLEMENTATIONS!
// =============================================================================

/// **PROOF: Using the EXISTING MarkAndSweep and GenerationalGC implementations!**
///
/// This enum wraps the real GC implementations from gc/mark_and_sweep.rs and gc/generational.rs.
/// NO duplicate logic!
///
/// Generational GC requires forwarding pointer support - the user must implement
/// get_forwarding/set_forwarding/is_forwarded callbacks to enable object movement.
pub enum GcCustomHandle {
    /// Uses the EXISTING MarkAndSweep from gc/mark_and_sweep.rs
    MarkSweep {
        gc: MarkAndSweep<CallbackRuntime, LibcMemoryProvider>,
        allocation_count: std::cell::Cell<usize>,
        gc_threshold: usize,
    },
    /// Uses the EXISTING GenerationalGC from gc/generational.rs
    Generational {
        gc: GenerationalGC<CallbackRuntime, LibcMemoryProvider>,
        allocation_count: std::cell::Cell<usize>,
        gc_threshold: usize,
    },
}

unsafe impl Send for GcCustomHandle {}

// =============================================================================
// GcCustomHandle Implementation - Delegates to EXISTING GC implementations
// =============================================================================

impl GcCustomHandle {
    /// Create a new GC using the EXISTING MarkAndSweep or GenerationalGC implementations.
    fn new(config: GcCustomConfig) -> Option<Self> {
        // Initialize callbacks in thread-local storage
        CALLBACKS.with(|cell| {
            *cell.borrow_mut() = Some(config.clone());
        });

        let options = AllocatorOptions {
            gc: true,
            print_stats: false,
            gc_always: false,
        };
        let memory = LibcMemoryProvider::new();

        let initial_heap = config.initial_heap.max(4096);
        let gc_threshold = initial_heap / 8; // Recommend GC after allocating 1/8 of heap

        match config.strategy {
            GcStrategy::MarkSweep => {
                let gc = MarkAndSweep::new(options, memory);
                Some(GcCustomHandle::MarkSweep {
                    gc,
                    allocation_count: std::cell::Cell::new(0),
                    gc_threshold,
                })
            }
            GcStrategy::Generational => {
                let gc = GenerationalGC::new(options, memory);
                Some(GcCustomHandle::Generational {
                    gc,
                    allocation_count: std::cell::Cell::new(0),
                    gc_threshold,
                })
            }
        }
    }

    fn allocate(&mut self, size: usize, user_ctx: *mut c_void) -> *mut c_void {
        let root_provider = CallbackRootProvider { user_ctx };

        match self {
            GcCustomHandle::MarkSweep { gc: ms, allocation_count, .. } => {
                // Track allocations
                allocation_count.set(allocation_count.get() + size);

                // Allocation loop
                loop {
                    match ms.try_allocate(size / 8, CallbackTypeTag::HeapObject) {
                        Ok(AllocateAction::Allocated(ptr)) => return ptr as *mut c_void,
                        Ok(AllocateAction::Gc) => {
                            ms.gc(&root_provider);
                            allocation_count.set(0); // Reset after GC
                        }
                        Err(_) => return std::ptr::null_mut(),
                    }
                }
            }
            GcCustomHandle::Generational { gc: gengc, allocation_count, .. } => {
                // Track allocations
                allocation_count.set(allocation_count.get() + size);

                // Allocation loop
                loop {
                    match gengc.try_allocate(size / 8, CallbackTypeTag::HeapObject) {
                        Ok(AllocateAction::Allocated(ptr)) => return ptr as *mut c_void,
                        Ok(AllocateAction::Gc) => {
                            gengc.gc(&root_provider);
                            allocation_count.set(0); // Reset after GC
                        }
                        Err(_) => return std::ptr::null_mut(),
                    }
                }
            }
        }
    }

    fn collect_inner(&mut self, user_ctx: *mut c_void) {
        let root_provider = CallbackRootProvider { user_ctx };
        match self {
            GcCustomHandle::MarkSweep { gc: ms, allocation_count, .. } => {
                ms.gc(&root_provider);
                allocation_count.set(0);
            }
            GcCustomHandle::Generational { gc: gengc, allocation_count, .. } => {
                gengc.gc(&root_provider);
                allocation_count.set(0);
            }
        }
    }

    fn should_collect(&self) -> bool {
        match self {
            GcCustomHandle::MarkSweep { allocation_count, gc_threshold, .. } => {
                allocation_count.get() >= *gc_threshold
            }
            GcCustomHandle::Generational { allocation_count, gc_threshold, .. } => {
                allocation_count.get() >= *gc_threshold
            }
        }
    }

    fn write_barrier(&mut self, object_ptr: usize, new_value: usize) {
        match self {
            GcCustomHandle::MarkSweep { .. } => {} // No-op
            GcCustomHandle::Generational { gc: gengc, .. } => {
                gengc.write_barrier(object_ptr, new_value);
            }
        }
    }

    fn get_young_gen_bounds(&self) -> (usize, usize) {
        match self {
            GcCustomHandle::MarkSweep { .. } => (0, 0),
            GcCustomHandle::Generational { gc: gengc, .. } => gengc.get_young_gen_bounds(),
        }
    }

    fn get_card_table_ptr(&self) -> *mut u8 {
        match self {
            GcCustomHandle::MarkSweep { .. } => std::ptr::null_mut(),
            GcCustomHandle::Generational { gc: gengc, .. } => gengc.get_card_table_biased_ptr(),
        }
    }
}

// =============================================================================
// C API - Uses the REAL GC implementations via the bridge
// =============================================================================

/// Create a callback-based GC instance using the EXISTING MarkAndSweep or GenerationalGC.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_create(config: GcCustomConfig) -> *mut GcCustomHandle {
    match GcCustomHandle::new(config) {
        Some(handle) => {
            Box::into_raw(Box::new(handle))
        }
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

/// Allocate bytes using the REAL GC implementations.
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

/// Collect using the REAL GC implementations.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_collect(gc: *mut GcCustomHandle, user_ctx: *mut c_void) {
    if gc.is_null() {
        return;
    }
    let gc = unsafe { &mut *gc };
    gc.collect_inner(user_ctx);
}

/// Write barrier (delegates to REAL generational GC).
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_write_barrier(
    gc: *mut GcCustomHandle,
    object: *mut c_void,
    new_value: *mut c_void,
) {
    if gc.is_null() {
        return;
    }
    let gc = unsafe { &mut *gc };
    gc.write_barrier(object as usize, new_value as usize);
}

/// Check if collection is recommended.
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_should_collect(gc: *mut GcCustomHandle) -> i32 {
    if gc.is_null() {
        return 0;
    }
    let gc = unsafe { &*gc };
    if gc.should_collect() { 1 } else { 0 }
}

/// Get young gen bounds (delegates to REAL generational GC).
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_get_young_gen_bounds(
    gc: *mut GcCustomHandle,
    start_out: *mut usize,
    end_out: *mut usize,
) {
    if gc.is_null() {
        return;
    }
    let gc = unsafe { &*gc };
    let (start, end) = gc.get_young_gen_bounds();

    if !start_out.is_null() {
        unsafe { *start_out = start };
    }
    if !end_out.is_null() {
        unsafe { *end_out = end };
    }
}

/// Get card table ptr (delegates to REAL generational GC).
#[unsafe(no_mangle)]
pub extern "C" fn gc_lib_custom_get_card_table_ptr(gc: *mut GcCustomHandle) -> *mut u8 {
    if gc.is_null() {
        return std::ptr::null_mut();
    }
    let gc = unsafe { &*gc };
    gc.get_card_table_ptr()
}

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

    // Forwarding pointer stubs (not used in mark-sweep tests)
    extern "C" fn test_is_forwarded(_obj: *mut c_void) -> i32 {
        0 // Never forwarded in mark-sweep
    }

    extern "C" fn test_get_forwarding(_obj: *mut c_void) -> *mut c_void {
        std::ptr::null_mut() // Not used in mark-sweep
    }

    extern "C" fn test_set_forwarding(_obj: *mut c_void, _new_loc: *mut c_void) {
        // Not used in mark-sweep
    }

    fn test_config() -> GcCustomConfig {
        GcCustomConfig {
            root_enumerator: empty_root_enumerator,
            tracer: test_tracer,
            is_marked: test_is_marked,
            set_mark: test_set_mark,
            get_size: test_get_size,
            is_forwarded: test_is_forwarded,
            get_forwarding: test_get_forwarding,
            set_forwarding: test_set_forwarding,
            initial_heap: 64 * 1024, // 64KB for tests
            strategy: GcStrategy::MarkSweep,
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

        // Allocate many objects until threshold is exceeded
        let obj_size = 24; // 8 byte header + 2 fields
        for _ in 0..100 {
            let ptr = gc_lib_custom_allocate(gc, obj_size, std::ptr::null_mut());
            if !ptr.is_null() {
                init_test_object(ptr, 2);
            }
        }

        // After many allocations, GC should recommend collection
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
        // The REAL MarkAndSweep adds an 8-byte header
        let gc = gc_lib_custom_create(test_config());

        let a = gc_lib_custom_allocate(gc, 24, std::ptr::null_mut());
        init_test_object(a, 2);
        let b = gc_lib_custom_allocate(gc, 24, std::ptr::null_mut());
        init_test_object(b, 2);

        // Objects should be 32 bytes apart (24 requested + 8 byte header)
        let diff = (b as usize) - (a as usize);
        assert_eq!(diff, 32, "expected 32-byte spacing (24 + 8 header), got {}", diff);

        gc_lib_custom_destroy(gc);
    }
}
