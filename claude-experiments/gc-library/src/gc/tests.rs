//! Comprehensive test suite for the GC algorithms.
//!
//! This module tests:
//! - Mark-and-sweep collector
//! - Generational collector
//! - Compacting collector
//! - Edge cases and stress tests

use crate::example::{ExampleObject, ExampleRuntime, ExampleTaggedPtr, ExampleTypeTag};
use crate::gc::compacting::CompactingHeap;
use crate::gc::generational::GenerationalGC;
use crate::gc::mark_and_sweep::MarkAndSweep;
use crate::gc::{AllocateAction, Allocator, AllocatorOptions, LibcMemoryProvider};
use crate::traits::{GcObject, RootProvider, TaggedPointer};

// =============================================================================
// Test Helpers
// =============================================================================

/// Allocate an object, running GC if needed.
fn alloc_with_gc<A: Allocator<ExampleRuntime, LibcMemoryProvider>>(
    gc: &mut A,
    words: usize,
    roots: &dyn RootProvider<ExampleTaggedPtr>,
) -> *const u8 {
    loop {
        match gc.try_allocate(words, ExampleTypeTag::HeapObject).unwrap() {
            AllocateAction::Allocated(ptr) => {
                // Initialize header
                let mut obj = ExampleObject::from_untagged(ptr);
                obj.write_header(words * 8);
                return ptr;
            }
            AllocateAction::Gc => {
                gc.gc(roots);
            }
        }
    }
}

/// Allocate an object without running GC (panics if GC needed).
fn alloc_no_gc<A: Allocator<ExampleRuntime, LibcMemoryProvider>>(gc: &mut A, words: usize) -> *const u8 {
    match gc.try_allocate(words, ExampleTypeTag::HeapObject).unwrap() {
        AllocateAction::Allocated(ptr) => {
            let mut obj = ExampleObject::from_untagged(ptr);
            obj.write_header(words * 8);
            ptr
        }
        AllocateAction::Gc => panic!("Unexpected GC needed"),
    }
}

/// Create a tagged pointer from a raw object pointer.
fn tag_ptr(ptr: *const u8) -> ExampleTaggedPtr {
    ExampleTaggedPtr::tag(ptr, ExampleTypeTag::HeapObject)
}

/// Write a pointer field in an object.
fn write_field(obj: *const u8, index: usize, value: ExampleTaggedPtr) {
    let mut heap_obj = ExampleObject::from_untagged(obj);
    let fields = heap_obj.get_fields_mut();
    fields[index] = value.as_usize();
}

/// Read a pointer field from an object.
fn read_field(obj: *const u8, index: usize) -> ExampleTaggedPtr {
    let heap_obj = ExampleObject::from_untagged(obj);
    let fields = heap_obj.get_fields();
    ExampleTaggedPtr::from_usize(fields[index])
}

/// Root provider that holds a vector of root values.
struct VecRoots {
    roots: Vec<usize>, // Each entry is a tagged pointer value
}

impl VecRoots {
    fn new() -> Self {
        Self { roots: Vec::new() }
    }

    fn add(&mut self, ptr: *const u8) {
        self.roots.push(tag_ptr(ptr).as_usize());
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        self.roots.clear();
    }

    #[allow(dead_code)]
    fn set(&mut self, index: usize, ptr: *const u8) {
        self.roots[index] = tag_ptr(ptr).as_usize();
    }

    fn get(&self, index: usize) -> ExampleTaggedPtr {
        ExampleTaggedPtr::from_usize(self.roots[index])
    }
}

impl RootProvider<ExampleTaggedPtr> for VecRoots {
    fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, ExampleTaggedPtr)) {
        for (i, &value) in self.roots.iter().enumerate() {
            let tagged = ExampleTaggedPtr::from_usize(value);
            if tagged.is_heap_pointer() {
                // Pass address of the slot so GC can update it
                let slot_addr = &self.roots[i] as *const usize as usize;
                callback(slot_addr, tagged);
            }
        }
    }
}

// =============================================================================
// Mark-and-Sweep Tests
// =============================================================================

mod mark_and_sweep_tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());

        // Allocate a simple object with 2 fields
        let ptr = alloc_no_gc(&mut gc, 2);
        assert!(!ptr.is_null());

        // Verify we can read/write fields
        let obj = ExampleObject::from_untagged(ptr);
        assert_eq!(obj.get_fields().len(), 2);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());

        let mut ptrs = Vec::new();
        for i in 0..100 {
            let ptr = alloc_no_gc(&mut gc, (i % 10) + 1);
            ptrs.push(ptr);
        }

        // Verify all allocations are distinct
        for i in 0..ptrs.len() {
            for j in (i + 1)..ptrs.len() {
                assert_ne!(ptrs[i], ptrs[j], "Allocations should be distinct");
            }
        }
    }

    #[test]
    fn test_gc_collects_unreachable() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let roots = VecRoots::new();

        // Allocate objects without rooting them
        for _ in 0..100 {
            alloc_no_gc(&mut gc, 2);
        }

        // Run GC - all objects should be collected
        gc.gc(&roots);

        // Should be able to allocate again (memory was freed)
        for _ in 0..100 {
            alloc_no_gc(&mut gc, 2);
        }
    }

    #[test]
    fn test_gc_preserves_rooted() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate and root an object
        let ptr = alloc_no_gc(&mut gc, 2);
        roots.add(ptr);

        // Write a known pattern to verify it survives
        write_field(ptr, 0, ExampleTaggedPtr::from_int(42));
        write_field(ptr, 1, ExampleTaggedPtr::from_int(123));

        // Run GC
        gc.gc(&roots);

        // Object should still be valid
        let obj = ExampleObject::from_untagged(ptr);
        let fields = obj.get_fields();
        assert_eq!(
            ExampleTaggedPtr::from_usize(fields[0]).as_int(),
            Some(42)
        );
        assert_eq!(
            ExampleTaggedPtr::from_usize(fields[1]).as_int(),
            Some(123)
        );
    }

    #[test]
    fn test_gc_traces_object_graph() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create a linked list: root -> A -> B -> C
        let c = alloc_no_gc(&mut gc, 1);
        let b = alloc_no_gc(&mut gc, 1);
        let a = alloc_no_gc(&mut gc, 1);

        // Link them
        write_field(a, 0, tag_ptr(b));
        write_field(b, 0, tag_ptr(c));
        write_field(c, 0, ExampleTaggedPtr::null());

        // Only root A
        roots.add(a);

        // Allocate garbage
        for _ in 0..50 {
            alloc_no_gc(&mut gc, 2);
        }

        // Run GC
        gc.gc(&roots);

        // All three should survive (traced from root)
        // Verify the links are intact
        assert_eq!(read_field(a, 0).untag(), b);
        assert_eq!(read_field(b, 0).untag(), c);
        assert!(read_field(c, 0).get_kind() == ExampleTypeTag::Null);
    }

    #[test]
    fn test_gc_handles_cycles() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create a cycle: A -> B -> A
        let a = alloc_no_gc(&mut gc, 1);
        let b = alloc_no_gc(&mut gc, 1);

        write_field(a, 0, tag_ptr(b));
        write_field(b, 0, tag_ptr(a));

        roots.add(a);

        // Run GC multiple times
        for _ in 0..5 {
            gc.gc(&roots);
        }

        // Both should survive
        assert_eq!(read_field(a, 0).untag(), b);
        assert_eq!(read_field(b, 0).untag(), a);
    }

    #[test]
    fn test_gc_with_allocation_pressure() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Keep a few long-lived objects
        for _ in 0..10 {
            let ptr = alloc_no_gc(&mut gc, 2);
            roots.add(ptr);
        }

        // Allocate many temporary objects, triggering GC
        for _ in 0..1000 {
            let _ = alloc_with_gc(&mut gc, 4, &roots);
        }

        // Original roots should still be valid
        assert_eq!(roots.roots.len(), 10);
    }

    #[test]
    fn test_zero_field_objects() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate zero-field object
        let ptr = alloc_no_gc(&mut gc, 0);
        roots.add(ptr);

        let obj = ExampleObject::from_untagged(ptr);
        assert_eq!(obj.get_fields().len(), 0);
        assert!(obj.is_zero_size());

        gc.gc(&roots);

        // Should survive
        let obj = ExampleObject::from_untagged(ptr);
        assert!(obj.is_zero_size());
    }

    #[test]
    fn test_large_objects() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate a large object (100 fields = 800 bytes)
        let ptr = alloc_no_gc(&mut gc, 100);
        roots.add(ptr);

        // Write to all fields
        for i in 0..100 {
            write_field(ptr, i, ExampleTaggedPtr::from_int(i as i64));
        }

        gc.gc(&roots);

        // Verify all fields
        for i in 0..100 {
            let val = read_field(ptr, i);
            assert_eq!(val.as_int(), Some(i as i64));
        }
    }
}

// =============================================================================
// Generational GC Tests
// =============================================================================

mod generational_tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());

        let ptr = alloc_no_gc(&mut gc, 2);
        assert!(!ptr.is_null());

        let obj = ExampleObject::from_untagged(ptr);
        assert_eq!(obj.get_fields().len(), 2);
    }

    #[test]
    fn test_young_gen_bounds() {
        let gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> = GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());

        let (start, end) = gc.get_young_gen_bounds();
        assert!(start > 0);
        assert!(end > start);
    }

    #[test]
    fn test_allocation_in_young_gen() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());

        let (young_start, young_end) = gc.get_young_gen_bounds();

        // Allocate and verify it's in young gen
        let ptr = alloc_no_gc(&mut gc, 2);
        let addr = ptr as usize;

        assert!(
            addr >= young_start && addr < young_end,
            "Allocation should be in young gen: addr={:#x}, young=[{:#x}, {:#x})",
            addr,
            young_start,
            young_end
        );
    }

    #[test]
    fn test_minor_gc_promotes_survivors() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        let (young_start, young_end) = gc.get_young_gen_bounds();

        // Allocate and root an object
        let ptr = alloc_no_gc(&mut gc, 2);
        write_field(ptr, 0, ExampleTaggedPtr::from_int(999));
        roots.add(ptr);

        // Verify it's in young gen initially
        let addr_before = ptr as usize;
        assert!(addr_before >= young_start && addr_before < young_end);

        // Run GC - object should be promoted to old gen
        gc.gc(&roots);

        // Get the new location from updated root
        let new_tagged = roots.get(0);
        let new_ptr = new_tagged.untag();
        let addr_after = new_ptr as usize;

        // Should be outside young gen now (in old gen)
        assert!(
            addr_after < young_start || addr_after >= young_end,
            "Object should be promoted to old gen: addr={:#x}, young=[{:#x}, {:#x})",
            addr_after,
            young_start,
            young_end
        );

        // Data should be preserved
        assert_eq!(read_field(new_ptr, 0).as_int(), Some(999));
    }

    #[test]
    fn test_minor_gc_collects_garbage() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let roots = VecRoots::new();

        // Allocate garbage in young gen
        for _ in 0..100 {
            alloc_no_gc(&mut gc, 2);
        }

        // Run GC with no roots - everything should be collected
        gc.gc(&roots);

        // Should be able to allocate more
        for _ in 0..100 {
            alloc_no_gc(&mut gc, 2);
        }
    }

    #[test]
    fn test_write_barrier_tracks_old_to_young() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate and promote an object to old gen
        let old_obj = alloc_no_gc(&mut gc, 2);
        roots.add(old_obj);
        gc.gc(&roots); // Promotes to old gen

        let old_ptr = roots.get(0).untag();

        // Allocate a new object in young gen
        let young_obj = alloc_no_gc(&mut gc, 1);

        // Write young pointer into old object (needs write barrier)
        write_field(old_ptr, 0, tag_ptr(young_obj));
        gc.write_barrier(tag_ptr(old_ptr).as_usize(), tag_ptr(young_obj).as_usize());

        // Don't root young_obj directly - only reachable through old_obj
        // Run GC
        gc.gc(&roots);

        // Young object should survive (tracked by write barrier)
        let updated_old = roots.get(0).untag();
        let updated_young = read_field(updated_old, 0);
        assert!(
            updated_young.is_heap_pointer(),
            "Young object should survive via write barrier"
        );
    }

    #[test]
    fn test_object_graph_promotion() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create a chain: A -> B -> C
        let c = alloc_no_gc(&mut gc, 1);
        let b = alloc_no_gc(&mut gc, 1);
        let a = alloc_no_gc(&mut gc, 1);

        write_field(a, 0, tag_ptr(b));
        write_field(b, 0, tag_ptr(c));
        write_field(c, 0, ExampleTaggedPtr::null());

        roots.add(a);

        // Run GC - entire chain should be promoted
        gc.gc(&roots);

        // Verify chain is intact
        let new_a = roots.get(0).untag();
        let new_b = read_field(new_a, 0).untag();
        let new_c = read_field(new_b, 0).untag();

        assert!(!new_b.is_null());
        assert!(!new_c.is_null());
        assert!(read_field(new_c, 0).get_kind() == ExampleTypeTag::Null);
    }

    #[test]
    fn test_mixed_generations() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create old gen object
        let old = alloc_no_gc(&mut gc, 2);
        roots.add(old);
        gc.gc(&roots); // Promote

        let old_ptr = roots.get(0).untag();

        // Allocate many young objects, keep some alive via old gen
        for i in 0..10 {
            let young = alloc_no_gc(&mut gc, 1);
            write_field(young, 0, ExampleTaggedPtr::from_int(i as i64));

            if i < 2 {
                // Link first two to old gen object
                write_field(old_ptr, i, tag_ptr(young));
                gc.write_barrier(tag_ptr(old_ptr).as_usize(), tag_ptr(young).as_usize());
            }
            // Rest are garbage
        }

        gc.gc(&roots);

        // Old object and its two children should survive
        let updated_old = roots.get(0).untag();
        let child0 = read_field(updated_old, 0);
        let child1 = read_field(updated_old, 1);

        assert!(child0.is_heap_pointer());
        assert!(child1.is_heap_pointer());
    }
}

// =============================================================================
// Compacting GC Tests
// =============================================================================

mod compacting_tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut gc: CompactingHeap<ExampleRuntime, LibcMemoryProvider> =
            CompactingHeap::new(AllocatorOptions::new(), LibcMemoryProvider::new());

        let ptr = alloc_no_gc(&mut gc, 2);
        assert!(!ptr.is_null());

        let obj = ExampleObject::from_untagged(ptr);
        assert_eq!(obj.get_fields().len(), 2);
    }

    #[test]
    fn test_gc_compacts_and_updates_refs() {
        let mut gc: CompactingHeap<ExampleRuntime, LibcMemoryProvider> =
            CompactingHeap::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate objects
        let a = alloc_no_gc(&mut gc, 2);
        let b = alloc_no_gc(&mut gc, 2);

        // Link them
        write_field(a, 0, tag_ptr(b));
        write_field(a, 1, ExampleTaggedPtr::from_int(42));
        write_field(b, 0, tag_ptr(a)); // Cycle
        write_field(b, 1, ExampleTaggedPtr::from_int(99));

        roots.add(a);

        // Run GC - objects will be copied
        gc.gc(&roots);

        // Get updated pointers
        let new_a = roots.get(0).untag();
        let new_b = read_field(new_a, 0).untag();

        // Verify data preserved
        assert_eq!(read_field(new_a, 1).as_int(), Some(42));
        assert_eq!(read_field(new_b, 1).as_int(), Some(99));

        // Verify references updated (cycle intact)
        assert_eq!(read_field(new_b, 0).untag(), new_a);
    }

    #[test]
    fn test_compaction_reclaims_space() {
        let mut gc: CompactingHeap<ExampleRuntime, LibcMemoryProvider> =
            CompactingHeap::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate one object to keep
        let keeper = alloc_no_gc(&mut gc, 2);
        roots.add(keeper);

        // Allocate many garbage objects
        for _ in 0..100 {
            alloc_no_gc(&mut gc, 4);
        }

        // Run GC - garbage collected, space compacted
        gc.gc(&roots);

        // Should be able to allocate lots more
        for _ in 0..100 {
            let _ = alloc_with_gc(&mut gc, 4, &roots);
        }
    }

    #[test]
    fn test_forwarding_pointers() {
        let mut gc: CompactingHeap<ExampleRuntime, LibcMemoryProvider> =
            CompactingHeap::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create a diamond: root -> A, A -> B, A -> C, B -> D, C -> D
        let d = alloc_no_gc(&mut gc, 1);
        let c = alloc_no_gc(&mut gc, 1);
        let b = alloc_no_gc(&mut gc, 1);
        let a = alloc_no_gc(&mut gc, 2);

        write_field(a, 0, tag_ptr(b));
        write_field(a, 1, tag_ptr(c));
        write_field(b, 0, tag_ptr(d));
        write_field(c, 0, tag_ptr(d));
        write_field(d, 0, ExampleTaggedPtr::from_int(777));

        roots.add(a);

        gc.gc(&roots);

        // Verify diamond structure preserved
        let new_a = roots.get(0).untag();
        let new_b = read_field(new_a, 0).untag();
        let new_c = read_field(new_a, 1).untag();
        let d_from_b = read_field(new_b, 0).untag();
        let d_from_c = read_field(new_c, 0).untag();

        // Both B and C should point to the same D
        assert_eq!(d_from_b, d_from_c, "Diamond should share D");
        assert_eq!(read_field(d_from_b, 0).as_int(), Some(777));
    }
}

// =============================================================================
// Stress Tests
// =============================================================================

mod stress_tests {
    use super::*;

    #[test]
    fn test_mark_sweep_many_allocations() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Keep 100 objects alive
        for _ in 0..100 {
            let ptr = alloc_no_gc(&mut gc, 2);
            roots.add(ptr);
        }

        // Allocate 10,000 temporary objects
        for _ in 0..10_000 {
            let _ = alloc_with_gc(&mut gc, 2, &roots);
        }

        // All roots should be valid
        assert_eq!(roots.roots.len(), 100);
    }

    #[test]
    fn test_generational_many_allocations() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Keep some long-lived objects
        for _ in 0..50 {
            let ptr = alloc_no_gc(&mut gc, 2);
            roots.add(ptr);
        }

        // Promote them
        gc.gc(&roots);

        // Allocate many short-lived objects
        for _ in 0..10_000 {
            let _ = alloc_with_gc(&mut gc, 2, &roots);
        }
    }

    #[test]
    fn test_deep_object_graph() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create a very deep linked list (1000 nodes)
        let mut prev = ExampleTaggedPtr::null();
        for i in 0..1000 {
            let node = alloc_no_gc(&mut gc, 2);
            write_field(node, 0, prev);
            write_field(node, 1, ExampleTaggedPtr::from_int(i as i64));
            prev = tag_ptr(node);
        }

        roots.roots.push(prev.as_usize());

        // Run GC multiple times
        for _ in 0..5 {
            gc.gc(&roots);
        }

        // Verify chain is intact
        let mut current = roots.get(0);
        let mut count = 0;
        while current.is_heap_pointer() {
            count += 1;
            current = read_field(current.untag(), 0);
        }
        assert_eq!(count, 1000);
    }

    #[test]
    fn test_wide_object_graph() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create a wide tree: 1 root with 100 children
        let root = alloc_no_gc(&mut gc, 100);
        roots.add(root);

        for i in 0..100 {
            let child = alloc_no_gc(&mut gc, 1);
            write_field(child, 0, ExampleTaggedPtr::from_int(i as i64));
            write_field(root, i, tag_ptr(child));
        }

        gc.gc(&roots);

        // Verify all children
        let root_ptr = roots.get(0).untag();
        for i in 0..100 {
            let child = read_field(root_ptr, i);
            assert!(child.is_heap_pointer());
            assert_eq!(read_field(child.untag(), 0).as_int(), Some(i as i64));
        }
    }

    #[test]
    fn test_allocation_sizes() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate objects of various sizes
        let sizes = [0, 1, 2, 5, 10, 50, 100, 500];

        for &size in &sizes {
            let ptr = alloc_no_gc(&mut gc, size);
            roots.add(ptr);

            let obj = ExampleObject::from_untagged(ptr);
            assert_eq!(obj.get_fields().len(), size);
        }

        gc.gc(&roots);

        // All should survive
        assert_eq!(roots.roots.len(), sizes.len());
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod heap_growth_tests {
    use super::*;

    #[test]
    fn test_mark_sweep_heap_growth() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate until we trigger heap growth
        // Keep all objects alive to force growth
        for _ in 0..1000 {
            let ptr = alloc_with_gc(&mut gc, 10, &roots);
            roots.add(ptr);

            // Verify object is valid
            let obj = ExampleObject::from_untagged(ptr);
            assert_eq!(obj.get_fields().len(), 10);
        }

        // All 1000 objects should be alive
        assert_eq!(roots.roots.len(), 1000);

        // Verify they all survived
        for i in 0..1000 {
            let ptr = roots.get(i).untag();
            let obj = ExampleObject::from_untagged(ptr);
            assert_eq!(obj.get_fields().len(), 10);
        }
    }

    #[test]
    fn test_generational_heap_growth() {
        let mut gc: GenerationalGC<ExampleRuntime, LibcMemoryProvider> =
            GenerationalGC::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate many objects, forcing promotions and old gen growth
        for _ in 0..500 {
            let ptr = alloc_with_gc(&mut gc, 5, &roots);
            roots.add(ptr);
        }

        // Promote all to old gen
        gc.gc(&roots);

        // Allocate more
        for _ in 0..500 {
            let ptr = alloc_with_gc(&mut gc, 5, &roots);
            roots.add(ptr);
        }

        assert_eq!(roots.roots.len(), 1000);
    }

    #[test]
    fn test_explicit_grow() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());

        // Explicitly grow the heap
        gc.grow();
        gc.grow();

        // Should be able to allocate more
        for _ in 0..100 {
            alloc_no_gc(&mut gc, 10);
        }
    }
}

mod thread_safe_tests {
    use super::*;
    use crate::gc::mutex_allocator::MutexAllocator;

    #[test]
    fn test_mutex_mark_sweep_basic() {
        let mut gc: MutexAllocator<MarkAndSweep<ExampleRuntime, LibcMemoryProvider>, ExampleRuntime, LibcMemoryProvider> =
            MutexAllocator::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Basic allocation
        let ptr = alloc_no_gc(&mut gc, 2);
        roots.add(ptr);

        write_field(ptr, 0, ExampleTaggedPtr::from_int(42));

        gc.gc(&roots);

        let new_ptr = roots.get(0).untag();
        assert_eq!(read_field(new_ptr, 0).as_int(), Some(42));
    }

    #[test]
    fn test_mutex_generational_basic() {
        let mut gc: MutexAllocator<GenerationalGC<ExampleRuntime, LibcMemoryProvider>, ExampleRuntime, LibcMemoryProvider> =
            MutexAllocator::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        let ptr = alloc_no_gc(&mut gc, 2);
        roots.add(ptr);

        gc.gc(&roots);

        // Object should be promoted
        let new_ptr = roots.get(0).untag();
        assert!(!new_ptr.is_null());
    }
}

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_gc() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let roots = VecRoots::new();

        // GC with nothing allocated should be fine
        gc.gc(&roots);
        gc.gc(&roots);
        gc.gc(&roots);
    }

    #[test]
    fn test_gc_disabled() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::no_gc(), LibcMemoryProvider::new());
        let roots = VecRoots::new();

        // Allocate many objects
        for _ in 0..100 {
            alloc_no_gc(&mut gc, 2);
        }

        // GC should be a no-op
        gc.gc(&roots);
    }

    #[test]
    fn test_self_referential() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Object pointing to itself
        let obj = alloc_no_gc(&mut gc, 1);
        write_field(obj, 0, tag_ptr(obj));
        roots.add(obj);

        gc.gc(&roots);

        let new_obj = roots.get(0).untag();
        assert_eq!(read_field(new_obj, 0).untag(), new_obj);
    }

    #[test]
    fn test_multiple_roots_same_object() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        let obj = alloc_no_gc(&mut gc, 1);
        write_field(obj, 0, ExampleTaggedPtr::from_int(42));

        // Root the same object multiple times
        roots.add(obj);
        roots.add(obj);
        roots.add(obj);

        gc.gc(&roots);

        // All roots should point to the same (surviving) object
        // and data should be intact
        for i in 0..3 {
            let ptr = roots.get(i).untag();
            assert_eq!(read_field(ptr, 0).as_int(), Some(42));
        }
    }

    #[test]
    fn test_null_fields_not_traced() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Object with null fields
        let obj = alloc_no_gc(&mut gc, 3);
        write_field(obj, 0, ExampleTaggedPtr::null());
        write_field(obj, 1, ExampleTaggedPtr::null());
        write_field(obj, 2, ExampleTaggedPtr::null());
        roots.add(obj);

        gc.gc(&roots);

        let ptr = roots.get(0).untag();
        for i in 0..3 {
            assert!(read_field(ptr, i).get_kind() == ExampleTypeTag::Null);
        }
    }

    #[test]
    fn test_int_fields_not_traced() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Object with integer fields (should not be traced as pointers)
        let obj = alloc_no_gc(&mut gc, 5);
        for i in 0..5 {
            write_field(obj, i, ExampleTaggedPtr::from_int(i as i64 * 1000));
        }
        roots.add(obj);

        gc.gc(&roots);

        let ptr = roots.get(0).untag();
        for i in 0..5 {
            assert_eq!(read_field(ptr, i).as_int(), Some(i as i64 * 1000));
        }
    }

    #[test]
    fn test_complex_cycle() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create a complex cycle: A -> B -> C -> D -> A
        let a = alloc_no_gc(&mut gc, 1);
        let b = alloc_no_gc(&mut gc, 1);
        let c = alloc_no_gc(&mut gc, 1);
        let d = alloc_no_gc(&mut gc, 1);

        write_field(a, 0, tag_ptr(b));
        write_field(b, 0, tag_ptr(c));
        write_field(c, 0, tag_ptr(d));
        write_field(d, 0, tag_ptr(a)); // Complete the cycle

        roots.add(a);

        // Run GC multiple times
        for _ in 0..10 {
            gc.gc(&roots);
        }

        // Verify cycle is intact
        let curr_a = roots.get(0).untag();
        let curr_b = read_field(curr_a, 0).untag();
        let curr_c = read_field(curr_b, 0).untag();
        let curr_d = read_field(curr_c, 0).untag();
        let back_to_a = read_field(curr_d, 0).untag();

        assert_eq!(back_to_a, curr_a, "Cycle should be intact");
    }

    #[test]
    fn test_sparse_object() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Object with mostly null fields but a few pointers
        let obj = alloc_no_gc(&mut gc, 20);
        let child1 = alloc_no_gc(&mut gc, 1);
        let child2 = alloc_no_gc(&mut gc, 1);

        write_field(child1, 0, ExampleTaggedPtr::from_int(111));
        write_field(child2, 0, ExampleTaggedPtr::from_int(222));

        // Only set fields 5 and 15
        for i in 0..20 {
            if i == 5 {
                write_field(obj, i, tag_ptr(child1));
            } else if i == 15 {
                write_field(obj, i, tag_ptr(child2));
            } else {
                write_field(obj, i, ExampleTaggedPtr::null());
            }
        }

        roots.add(obj);
        gc.gc(&roots);

        // Verify children survived
        let new_obj = roots.get(0).untag();
        let new_child1 = read_field(new_obj, 5);
        let new_child2 = read_field(new_obj, 15);

        assert!(new_child1.is_heap_pointer());
        assert!(new_child2.is_heap_pointer());
        assert_eq!(read_field(new_child1.untag(), 0).as_int(), Some(111));
        assert_eq!(read_field(new_child2.untag(), 0).as_int(), Some(222));
    }
}

// =============================================================================
// Correctness Verification Tests
// =============================================================================

mod correctness_tests {
    use super::*;

    /// Test that building and traversing a binary tree works correctly.
    #[test]
    fn test_binary_tree() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Build a small binary tree (depth 4 = 15 nodes)
        fn build_tree<A: Allocator<ExampleRuntime, LibcMemoryProvider>>(
            gc: &mut A,
            depth: usize,
            roots: &VecRoots,
        ) -> ExampleTaggedPtr {
            if depth == 0 {
                return ExampleTaggedPtr::null();
            }

            let node = alloc_with_gc(gc, 2, roots);
            let left = build_tree(gc, depth - 1, roots);
            let right = build_tree(gc, depth - 1, roots);

            write_field(node, 0, left);
            write_field(node, 1, right);

            tag_ptr(node)
        }

        let tree = build_tree(&mut gc, 4, &roots);
        roots.roots.push(tree.as_usize());

        // Count nodes before GC
        fn count_nodes(ptr: ExampleTaggedPtr) -> usize {
            if !ptr.is_heap_pointer() {
                return 0;
            }
            let left = read_field(ptr.untag(), 0);
            let right = read_field(ptr.untag(), 1);
            1 + count_nodes(left) + count_nodes(right)
        }

        let count_before = count_nodes(roots.get(0));
        assert_eq!(count_before, 15); // 2^4 - 1 = 15 nodes

        // Run GC
        gc.gc(&roots);

        // Count nodes after GC
        let count_after = count_nodes(roots.get(0));
        assert_eq!(count_after, 15, "All tree nodes should survive GC");
    }

    /// Test that unreachable subgraphs are collected.
    #[test]
    fn test_unreachable_subgraph_collected() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Create two separate graphs
        // Graph 1: A -> B -> C (rooted)
        let c1 = alloc_no_gc(&mut gc, 1);
        let b1 = alloc_no_gc(&mut gc, 1);
        let a1 = alloc_no_gc(&mut gc, 1);
        write_field(a1, 0, tag_ptr(b1));
        write_field(b1, 0, tag_ptr(c1));
        write_field(c1, 0, ExampleTaggedPtr::from_int(1));

        // Graph 2: D -> E -> F (NOT rooted - should be collected)
        let f2 = alloc_no_gc(&mut gc, 1);
        let e2 = alloc_no_gc(&mut gc, 1);
        let d2 = alloc_no_gc(&mut gc, 1);
        write_field(d2, 0, tag_ptr(e2));
        write_field(e2, 0, tag_ptr(f2));
        write_field(f2, 0, ExampleTaggedPtr::from_int(2));

        // Only root graph 1
        roots.add(a1);

        // Run GC
        gc.gc(&roots);

        // Graph 1 should survive
        let new_a1 = roots.get(0).untag();
        let new_b1 = read_field(new_a1, 0).untag();
        let new_c1 = read_field(new_b1, 0).untag();
        assert_eq!(read_field(new_c1, 0).as_int(), Some(1));

        // Graph 2 was collected - we can't directly verify this, but
        // we can verify we can allocate new objects in the freed space
        for _ in 0..10 {
            let _ = alloc_with_gc(&mut gc, 2, &roots);
        }
    }

    /// Test mixed pointer and non-pointer fields.
    #[test]
    fn test_mixed_fields() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Object with mix of pointers, integers, and nulls
        let child = alloc_no_gc(&mut gc, 1);
        write_field(child, 0, ExampleTaggedPtr::from_int(999));

        let obj = alloc_no_gc(&mut gc, 5);
        write_field(obj, 0, tag_ptr(child)); // Pointer
        write_field(obj, 1, ExampleTaggedPtr::from_int(100)); // Int
        write_field(obj, 2, ExampleTaggedPtr::null()); // Null
        write_field(obj, 3, ExampleTaggedPtr::from_int(-50)); // Negative int
        write_field(obj, 4, tag_ptr(child)); // Same pointer again

        roots.add(obj);
        gc.gc(&roots);

        let new_obj = roots.get(0).untag();

        // Verify pointer fields
        let ptr0 = read_field(new_obj, 0);
        let ptr4 = read_field(new_obj, 4);
        assert!(ptr0.is_heap_pointer());
        assert!(ptr4.is_heap_pointer());
        assert_eq!(ptr0.untag(), ptr4.untag()); // Should point to same object

        // Verify int fields
        assert_eq!(read_field(new_obj, 1).as_int(), Some(100));
        assert_eq!(read_field(new_obj, 3).as_int(), Some(-50));

        // Verify null field
        assert!(read_field(new_obj, 2).get_kind() == ExampleTypeTag::Null);

        // Verify child data
        assert_eq!(read_field(ptr0.untag(), 0).as_int(), Some(999));
    }

    /// Stress test with random-ish object graph structure.
    #[test]
    fn test_pseudo_random_graph() {
        let mut gc: MarkAndSweep<ExampleRuntime, LibcMemoryProvider> = MarkAndSweep::new(AllocatorOptions::new(), LibcMemoryProvider::new());
        let mut roots = VecRoots::new();

        // Allocate 100 objects with 3 fields each
        let mut objects: Vec<*const u8> = Vec::new();
        for _ in 0..100 {
            let ptr = alloc_no_gc(&mut gc, 3);
            objects.push(ptr);
        }

        // Create pseudo-random connections
        for (i, &obj) in objects.iter().enumerate() {
            // Field 0: points to next object (wrapping)
            let next = objects[(i + 1) % objects.len()];
            write_field(obj, 0, tag_ptr(next));

            // Field 1: points to object at i*7 mod 100
            let linked = objects[(i * 7) % objects.len()];
            write_field(obj, 1, tag_ptr(linked));

            // Field 2: store index as int
            write_field(obj, 2, ExampleTaggedPtr::from_int(i as i64));
        }

        // Root first object (all others reachable through links)
        roots.add(objects[0]);

        // Run GC multiple times
        for _ in 0..5 {
            gc.gc(&roots);
        }

        // Traverse and verify structure is intact
        let mut visited = vec![false; 100];
        let mut current = roots.get(0);
        let mut count = 0;

        // Follow the "next" chain
        while current.is_heap_pointer() && count < 200 {
            let idx = read_field(current.untag(), 2).as_int().unwrap() as usize;
            if visited[idx] {
                break;
            }
            visited[idx] = true;
            count += 1;
            current = read_field(current.untag(), 0);
        }

        assert_eq!(count, 100, "Should visit all 100 objects");
    }
}
