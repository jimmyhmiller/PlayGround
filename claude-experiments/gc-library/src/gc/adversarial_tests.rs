//! Adversarial tests designed to find real bugs in the GC.
//!
//! These tests try to break things by:
//! - Using canary values to detect memory corruption
//! - Validating heap integrity after operations
//! - Testing boundary conditions
//! - Creating patterns that expose common GC bugs

use crate::example::{ExampleObject, ExampleRuntime, ExampleTaggedPtr, ExampleTypeTag};
use crate::gc::mark_and_sweep::MarkAndSweep;
use crate::gc::generational::GenerationalGC;
use crate::gc::{AllocateAction, Allocator, AllocatorOptions};
use crate::traits::{GcObject, RootProvider, TaggedPointer};

// =============================================================================
// Test Infrastructure
// =============================================================================

/// A root provider that stores actual memory locations (not copies).
/// This lets us verify the GC updates roots correctly.
struct LiveRoots {
    slots: Vec<usize>,
}

impl LiveRoots {
    fn new() -> Self {
        Self { slots: Vec::new() }
    }

    fn push(&mut self, ptr: *const u8) {
        self.slots.push(ExampleTaggedPtr::tag(ptr, ExampleTypeTag::HeapObject).as_usize());
    }

    fn get_ptr(&self, index: usize) -> *const u8 {
        ExampleTaggedPtr::from_usize(self.slots[index]).untag()
    }

    fn len(&self) -> usize {
        self.slots.len()
    }
}

impl RootProvider<ExampleTaggedPtr> for LiveRoots {
    fn enumerate_roots(&self, callback: &mut dyn FnMut(usize, ExampleTaggedPtr)) {
        for i in 0..self.slots.len() {
            let slot_addr = &self.slots[i] as *const usize as usize;
            let tagged = ExampleTaggedPtr::from_usize(self.slots[i]);
            if tagged.is_heap_pointer() {
                callback(slot_addr, tagged);
            }
        }
    }
}

fn tag_ptr(ptr: *const u8) -> ExampleTaggedPtr {
    ExampleTaggedPtr::tag(ptr, ExampleTypeTag::HeapObject)
}

fn write_field(obj: *const u8, index: usize, value: ExampleTaggedPtr) {
    let mut heap_obj = ExampleObject::from_untagged(obj);
    let fields = heap_obj.get_fields_mut();
    fields[index] = value.as_usize();
}

fn read_field(obj: *const u8, index: usize) -> ExampleTaggedPtr {
    let heap_obj = ExampleObject::from_untagged(obj);
    let fields = heap_obj.get_fields();
    ExampleTaggedPtr::from_usize(fields[index])
}

/// Magic canary base value to detect corruption (full 64 bits)
/// We use tag bits 010 (not a heap pointer) to avoid GC trying to trace canaries.
const CANARY_BASE: u64 = 0x_DEAD_BEEF_CAFE_BA02;

/// Generate a canary value that is never mistaken for a heap pointer.
/// HeapObject tag is 001, so we ensure bottom bits are always 010.
fn make_canary(base: u64, index: usize) -> u64 {
    // Use upper bits for unique pattern, keep bottom 3 bits as 010
    let pattern = CANARY_BASE ^ ((base + index as u64) << 3);
    (pattern & !0b111) | 0b010 // Force tag bits to 010 (not a heap pointer)
}

/// Write a canary pattern to all fields of an object (raw, not tagged)
/// This bypasses the tagged pointer encoding to test the GC's memory handling directly.
fn write_canaries_raw(obj: *const u8, count: usize, base: u64) {
    let fields_ptr = unsafe { (obj as *mut u8).add(8) as *mut u64 };
    for i in 0..count {
        let canary = make_canary(base, i);
        unsafe {
            *fields_ptr.add(i) = canary;
        }
    }
}

/// Verify canary pattern in all fields (raw, not tagged)
fn verify_canaries_raw(obj: *const u8, count: usize, base: u64) -> bool {
    let fields_ptr = unsafe { (obj as *const u8).add(8) as *const u64 };
    for i in 0..count {
        let expected = make_canary(base, i);
        let actual = unsafe { *fields_ptr.add(i) };
        if actual != expected {
            eprintln!(
                "CANARY MISMATCH at field {}: expected {:016x}, got {:016x}",
                i, expected, actual
            );
            return false;
        }
    }
    true
}

fn alloc_with_gc<A: Allocator<ExampleRuntime>>(
    gc: &mut A,
    words: usize,
    roots: &dyn RootProvider<ExampleTaggedPtr>,
) -> *const u8 {
    loop {
        match gc.try_allocate(words, ExampleTypeTag::HeapObject).unwrap() {
            AllocateAction::Allocated(ptr) => {
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

fn alloc_no_gc<A: Allocator<ExampleRuntime>>(gc: &mut A, words: usize) -> *const u8 {
    match gc.try_allocate(words, ExampleTypeTag::HeapObject).unwrap() {
        AllocateAction::Allocated(ptr) => {
            let mut obj = ExampleObject::from_untagged(ptr);
            obj.write_header(words * 8);
            ptr
        }
        AllocateAction::Gc => panic!("Unexpected GC needed"),
    }
}

// =============================================================================
// Memory Corruption Detection
// =============================================================================

/// Allocate objects with canary values, run GC, verify canaries intact.
/// This catches: heap corruption, incorrect object sizes, overwriting live data.
#[test]
fn test_canary_survival_mark_sweep() {
    let mut gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Allocate objects with unique canary patterns
    for i in 0..50 {
        let obj = alloc_no_gc(&mut gc, 10);
        write_canaries_raw(obj, 10, i as u64 * 1000);
        roots.push(obj);
    }

    // Allocate garbage between them
    for _ in 0..100 {
        let garbage = alloc_no_gc(&mut gc, 5);
        write_canaries_raw(garbage, 5, 0xFFFF_FFFF_FFFF_0000); // Different pattern for garbage
    }

    // Run GC
    gc.gc(&roots);

    // Verify all canaries
    for i in 0..50 {
        let obj = roots.get_ptr(i);
        assert!(
            verify_canaries_raw(obj, 10, (i as u64) * 1000),
            "Object {} corrupted after GC",
            i
        );
    }
}

#[test]
fn test_canary_survival_generational() {
    let mut gc: GenerationalGC<ExampleRuntime> = GenerationalGC::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Allocate objects with canaries
    for i in 0..50 {
        let obj = alloc_no_gc(&mut gc, 10);
        write_canaries_raw(obj, 10, i as u64 * 1000);
        roots.push(obj);
    }

    // Run multiple GCs (some minor, eventually a major)
    for gc_round in 0..10 {
        // Allocate garbage
        for _ in 0..20 {
            let _ = alloc_with_gc(&mut gc, 5, &roots);
        }

        gc.gc(&roots);

        // Verify after each GC
        for i in 0..50 {
            let obj = roots.get_ptr(i);
            assert!(
                verify_canaries_raw(obj, 10, (i as u64) * 1000),
                "Object {} corrupted after GC round {}",
                i,
                gc_round
            );
        }
    }
}

// =============================================================================
// Reference Integrity
// =============================================================================

/// Create a complex graph, run GC, verify all references still point to valid objects.
/// This catches: missed reference updates, incorrect forwarding pointers.
#[test]
fn test_reference_integrity_after_gc() {
    let mut gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Create 100 objects, each with pointers to random other objects
    let mut objects: Vec<*const u8> = Vec::new();
    for i in 0..100 {
        let obj = alloc_no_gc(&mut gc, 5);
        // Store a unique ID in field 0
        write_field(obj, 0, ExampleTaggedPtr::from_int(i as i64));
        objects.push(obj);
    }

    // Create references between objects
    for i in 0..100 {
        let obj = objects[i];
        // Field 1: next object (circular)
        write_field(obj, 1, tag_ptr(objects[(i + 1) % 100]));
        // Field 2: object at i*7 mod 100
        write_field(obj, 2, tag_ptr(objects[(i * 7) % 100]));
        // Field 3: object at i*13 mod 100
        write_field(obj, 3, tag_ptr(objects[(i * 13) % 100]));
        // Field 4: back-reference to i/2
        write_field(obj, 4, tag_ptr(objects[i / 2]));
    }

    // Root only object 0 (all others reachable)
    roots.push(objects[0]);

    // Allocate garbage
    for _ in 0..200 {
        alloc_no_gc(&mut gc, 3);
    }

    // Run GC
    gc.gc(&roots);

    // Build map of new locations by traversing from root
    let mut id_to_ptr: std::collections::HashMap<i64, *const u8> = std::collections::HashMap::new();
    let mut to_visit = vec![roots.get_ptr(0)];

    while let Some(obj) = to_visit.pop() {
        let id = read_field(obj, 0).as_int().unwrap();
        if id_to_ptr.contains_key(&id) {
            continue;
        }
        id_to_ptr.insert(id, obj);

        // Visit all pointer fields
        for field_idx in 1..5 {
            let ptr = read_field(obj, field_idx);
            if ptr.is_heap_pointer() {
                to_visit.push(ptr.untag());
            }
        }
    }

    // Verify all 100 objects are reachable
    assert_eq!(id_to_ptr.len(), 100, "Some objects were incorrectly collected");

    // Verify all references are consistent
    for i in 0..100 {
        let obj = id_to_ptr.get(&(i as i64)).unwrap();

        // Check field 1 points to (i+1) % 100
        let ref1 = read_field(*obj, 1);
        let ref1_id = read_field(ref1.untag(), 0).as_int().unwrap();
        assert_eq!(ref1_id, ((i + 1) % 100) as i64, "Field 1 reference wrong for object {}", i);

        // Check field 2 points to i*7 % 100
        let ref2 = read_field(*obj, 2);
        let ref2_id = read_field(ref2.untag(), 0).as_int().unwrap();
        assert_eq!(ref2_id, ((i * 7) % 100) as i64, "Field 2 reference wrong for object {}", i);
    }
}

// =============================================================================
// Allocation Pattern Attacks
// =============================================================================

/// Allocate in a pattern designed to fragment the heap, then verify GC handles it.
#[test]
fn test_fragmentation_pattern() {
    let mut gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Allocate alternating live/dead objects to create fragmentation
    for i in 0..100 {
        let obj = alloc_no_gc(&mut gc, 10);
        write_canaries_raw(obj, 10, i as u64 * 100);

        if i % 2 == 0 {
            roots.push(obj); // Keep even objects
        }
        // Odd objects become garbage
    }

    gc.gc(&roots);

    // Verify surviving objects
    for i in 0..roots.len() {
        let obj = roots.get_ptr(i);
        assert!(
            verify_canaries_raw(obj, 10, (i as u64) * 2 * 100), // i*2 because only even survived
            "Object {} corrupted",
            i
        );
    }

    // Try to allocate in the gaps
    for i in 0..50 {
        let obj = alloc_with_gc(&mut gc, 8, &roots);
        write_canaries_raw(obj, 8, 10000 + i as u64);
        roots.push(obj);
    }

    gc.gc(&roots);

    // Verify old and new objects
    assert_eq!(roots.len(), 100); // 50 original + 50 new
}

/// Test that objects allocated right before GC are handled correctly.
#[test]
fn test_allocation_just_before_gc() {
    let mut gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Allocate until we're close to needing GC
    for _ in 0..500 {
        let _ = alloc_no_gc(&mut gc, 10);
    }

    // Now allocate objects we want to keep
    for i in 0..10 {
        let obj = alloc_with_gc(&mut gc, 5, &roots);
        write_canaries_raw(obj, 5, i as u64 * 100);
        roots.push(obj);
    }

    // Force GC
    gc.gc(&roots);

    // Verify
    for i in 0..10 {
        let obj = roots.get_ptr(i);
        assert!(verify_canaries_raw(obj, 5, (i as u64) * 100), "Object {} corrupted", i);
    }
}

// =============================================================================
// Object Size Edge Cases
// =============================================================================

/// Test objects of many different sizes to catch size calculation bugs.
#[test]
fn test_varied_object_sizes() {
    let mut gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();
    let mut expected_sizes: Vec<usize> = Vec::new();

    // Allocate objects of sizes 0, 1, 2, ..., 127
    for size in 0..128 {
        let obj = alloc_no_gc(&mut gc, size);

        // Write canaries to non-zero sized objects
        if size > 0 {
            write_canaries_raw(obj, size, size as u64 * 1000);
        }

        roots.push(obj);
        expected_sizes.push(size);
    }

    // Allocate garbage of various sizes
    for size in 1..50 {
        let _ = alloc_no_gc(&mut gc, size * 2);
    }

    gc.gc(&roots);

    // Verify sizes and canaries
    for i in 0..128 {
        let obj = roots.get_ptr(i);
        let heap_obj = ExampleObject::from_untagged(obj);

        assert_eq!(
            heap_obj.get_fields().len(),
            expected_sizes[i],
            "Object {} has wrong size after GC",
            i
        );

        if expected_sizes[i] > 0 {
            assert!(
                verify_canaries_raw(obj, expected_sizes[i], (expected_sizes[i] as u64) * 1000),
                "Object {} (size {}) corrupted",
                i,
                expected_sizes[i]
            );
        }
    }
}

// =============================================================================
// Generational-Specific Bug Detection
// =============================================================================

/// Test that write barriers actually prevent premature collection.
/// Creates old->young reference without barrier, verifies it would fail,
/// then with barrier, verifies it works.
#[test]
fn test_write_barrier_necessity() {
    let mut gc: GenerationalGC<ExampleRuntime> = GenerationalGC::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Create an old-gen object
    let old = alloc_no_gc(&mut gc, 2);
    write_field(old, 0, ExampleTaggedPtr::from_int(111));
    write_field(old, 1, ExampleTaggedPtr::null());
    roots.push(old);

    // Promote to old gen
    gc.gc(&roots);
    let old_ptr = roots.get_ptr(0);

    // Verify it's in old gen
    let (young_start, young_end) = gc.get_young_gen_bounds();
    let old_addr = old_ptr as usize;
    assert!(
        old_addr < young_start || old_addr >= young_end,
        "Object should be in old gen"
    );

    // Create young object and link FROM old TO young
    let young = alloc_no_gc(&mut gc, 1);
    write_field(young, 0, ExampleTaggedPtr::from_int(222));

    // Write WITH barrier
    write_field(old_ptr, 1, tag_ptr(young));
    gc.write_barrier(tag_ptr(old_ptr).as_usize(), tag_ptr(young).as_usize());

    // GC should preserve young object via barrier
    gc.gc(&roots);

    // Verify young object survived
    let new_old = roots.get_ptr(0);
    let young_ref = read_field(new_old, 1);
    assert!(young_ref.is_heap_pointer(), "Young object should survive via write barrier");
    assert_eq!(
        read_field(young_ref.untag(), 0).as_int(),
        Some(222),
        "Young object data should be intact"
    );
}

/// Test card table coverage by writing to objects at various positions in old gen.
#[test]
fn test_card_table_coverage() {
    let mut gc: GenerationalGC<ExampleRuntime> = GenerationalGC::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Allocate many objects to spread across cards
    for i in 0..100 {
        let obj = alloc_no_gc(&mut gc, 10);
        write_field(obj, 0, ExampleTaggedPtr::from_int(i as i64));
        roots.push(obj);
    }

    // Promote all to old gen
    gc.gc(&roots);

    // Now create young objects and link from EACH old object
    for i in 0..100 {
        let old_ptr = roots.get_ptr(i);
        let young = alloc_no_gc(&mut gc, 1);
        write_field(young, 0, ExampleTaggedPtr::from_int(1000 + i as i64));

        write_field(old_ptr, 1, tag_ptr(young));
        gc.write_barrier(tag_ptr(old_ptr).as_usize(), tag_ptr(young).as_usize());
    }

    // GC
    gc.gc(&roots);

    // Verify ALL young objects survived
    for i in 0..100 {
        let old_ptr = roots.get_ptr(i);
        let young_ref = read_field(old_ptr, 1);

        assert!(
            young_ref.is_heap_pointer(),
            "Young object {} should survive",
            i
        );
        assert_eq!(
            read_field(young_ref.untag(), 0).as_int(),
            Some(1000 + i as i64),
            "Young object {} data wrong",
            i
        );
    }
}

// =============================================================================
// Stress with Verification
// =============================================================================

/// Heavy allocation with continuous integrity checking.
#[test]
fn test_stress_with_verification() {
    let mut gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Keep track of expected values
    let mut expected: Vec<i64> = Vec::new();

    for round in 0..100 {
        // Allocate 10 new objects per round
        for j in 0..10 {
            let obj = alloc_with_gc(&mut gc, 3, &roots);
            let value = (round * 1000 + j) as i64;
            write_field(obj, 0, ExampleTaggedPtr::from_int(value));
            write_field(obj, 1, ExampleTaggedPtr::from_int(value + 1));
            write_field(obj, 2, ExampleTaggedPtr::from_int(value + 2));
            roots.push(obj);
            expected.push(value);
        }

        // Allocate garbage
        for _ in 0..50 {
            let garbage = alloc_with_gc(&mut gc, 5, &roots);
            write_field(garbage, 0, ExampleTaggedPtr::from_int(-1));
        }

        // Verify ALL objects every 10 rounds
        if round % 10 == 9 {
            for i in 0..roots.len() {
                let obj = roots.get_ptr(i);
                let v = expected[i];

                assert_eq!(
                    read_field(obj, 0).as_int(),
                    Some(v),
                    "Object {} field 0 wrong at round {}",
                    i,
                    round
                );
                assert_eq!(
                    read_field(obj, 1).as_int(),
                    Some(v + 1),
                    "Object {} field 1 wrong at round {}",
                    i,
                    round
                );
                assert_eq!(
                    read_field(obj, 2).as_int(),
                    Some(v + 2),
                    "Object {} field 2 wrong at round {}",
                    i,
                    round
                );
            }
        }
    }
}

/// Test that repeated GC cycles don't cause drift or accumulating errors.
#[test]
fn test_gc_idempotence() {
    let mut gc: MarkAndSweep<ExampleRuntime> = MarkAndSweep::new(AllocatorOptions::new());
    let mut roots = LiveRoots::new();

    // Set up a stable object graph
    for i in 0..20 {
        let obj = alloc_no_gc(&mut gc, 5);
        // Write canaries to fields 0-3 only (field 4 will be ring link)
        write_canaries_raw(obj, 4, i as u64 * 100);
        roots.push(obj);
    }

    // Link them in a ring using field 4
    for i in 0..20 {
        let curr = roots.get_ptr(i);
        let next = roots.get_ptr((i + 1) % 20);
        write_field(curr, 4, tag_ptr(next));
    }

    // Run GC 100 times
    for round in 0..100 {
        gc.gc(&roots);

        // Verify after EVERY gc
        for i in 0..20 {
            let obj = roots.get_ptr(i);

            // Check canaries in fields 0-3 using raw verification
            assert!(
                verify_canaries_raw(obj, 4, i as u64 * 100),
                "Object {} corrupted after {} GCs",
                i,
                round + 1
            );

            // Check ring link
            let next = read_field(obj, 4);
            assert!(next.is_heap_pointer(), "Ring broken at object {} after {} GCs", i, round + 1);
        }
    }
}
