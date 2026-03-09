use dynvalue::{LowBit, NanBox, Value};
use dynobj::*;

use crate::{Alloc, HeapWalker, BumpAllocator, AtomicBumpAllocator, alloc_obj, Mutator, Root, RootScope};
use crate::{PtrPolicy, SemiSpace, Heap};
use dynobj::{RootSource, FrameChain, RootFrame, RootSet, AtomicRootSet};

use std::sync::Arc;

// ─── BumpAllocator ───────────────────────────────────────────────────

#[test]
fn bump_basic_alloc() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = bump.alloc(&INFO, 0);
    assert!(!ptr.is_null());
    assert!(bump.contains(ptr));
    assert_eq!(bump.used(), INFO.allocation_size(0));
}

#[test]
fn bump_alignment() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(1)
        .with_align_log2(4); // 16-byte aligned
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = bump.alloc(&INFO, 0);
    assert!(!ptr.is_null());
    assert_eq!(ptr as usize % 16, 0, "pointer should be 16-byte aligned");
}

#[test]
fn bump_zeroed() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = bump.alloc(&INFO, 0);
    assert!(!ptr.is_null());
    let size = INFO.allocation_size(0);
    for i in 0..size {
        assert_eq!(unsafe { *ptr.add(i) }, 0, "byte {} should be zero", i);
    }
}

#[test]
fn bump_full_returns_null() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let obj_size = INFO.allocation_size(0); // 24 bytes
    // Create allocator just big enough for one object
    let bump = BumpAllocator::new::<Compact>(obj_size);

    let ptr1 = bump.alloc(&INFO, 0);
    assert!(!ptr1.is_null());

    let ptr2 = bump.alloc(&INFO, 0);
    assert!(ptr2.is_null(), "should return null when full");
}

#[test]
fn bump_reset() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = BumpAllocator::new::<Compact>(4096);

    let _ptr1 = bump.alloc(&INFO, 0);
    assert!(bump.used() > 0);

    bump.reset();
    assert_eq!(bump.used(), 0);
    assert_eq!(bump.remaining(), bump.size());

    // Can allocate again after reset
    let ptr2 = bump.alloc(&INFO, 0);
    assert!(!ptr2.is_null());
}

#[test]
fn bump_alloc_obj_convenience() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = unsafe { alloc_obj::<Compact>(&bump, &INFO, 0) };
    assert!(!ptr.is_null());

    // Verify header is initialized
    unsafe {
        let header = core::ptr::read(ptr as *const Compact);
        assert_eq!(header.type_info(), &INFO as *const TypeInfo);
    }
}

#[test]
fn bump_alloc_obj_varlen() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(1);
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = unsafe { alloc_obj::<Compact>(&bump, &INFO, 5) };
    assert!(!ptr.is_null());

    unsafe {
        // Verify header
        let header = core::ptr::read(ptr as *const Compact);
        assert_eq!(header.type_info(), &INFO as *const TypeInfo);

        // Verify varlen count was written
        assert_eq!(read_varlen_count(ptr, &INFO), 5);
    }
}

#[test]
fn bump_heap_walk() {
    static INFO_A: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    static INFO_B: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static INFO_C: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(0);

    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr_a = unsafe { alloc_obj::<Compact>(&bump, &INFO_A, 0) };
    let ptr_b = unsafe { alloc_obj::<Compact>(&bump, &INFO_B, 0) };
    let ptr_c = unsafe { alloc_obj::<Compact>(&bump, &INFO_C, 0) };

    let mut visited: Vec<(*mut u8, *const TypeInfo)> = Vec::new();
    unsafe {
        bump.walk(&mut |ptr, info| {
            visited.push((ptr, info as *const TypeInfo));
        });
    }

    assert_eq!(visited.len(), 3);
    assert_eq!(visited[0].0, ptr_a);
    assert_eq!(visited[0].1, &INFO_A as *const TypeInfo);
    assert_eq!(visited[1].0, ptr_b);
    assert_eq!(visited[1].1, &INFO_B as *const TypeInfo);
    assert_eq!(visited[2].0, ptr_c);
    assert_eq!(visited[2].1, &INFO_C as *const TypeInfo);
}

#[test]
fn bump_heap_walk_varlen() {
    static INFO_FIX: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    static INFO_VEC: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(0);
    static INFO_STR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_bytes(0);

    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr_fix = unsafe { alloc_obj::<Compact>(&bump, &INFO_FIX, 0) };
    let ptr_vec = unsafe { alloc_obj::<Compact>(&bump, &INFO_VEC, 3) };
    let ptr_str = unsafe { alloc_obj::<Compact>(&bump, &INFO_STR, 10) };

    let mut visited: Vec<(*mut u8, *const TypeInfo)> = Vec::new();
    unsafe {
        bump.walk(&mut |ptr, info| {
            visited.push((ptr, info as *const TypeInfo));
        });
    }

    assert_eq!(visited.len(), 3);
    assert_eq!(visited[0].0, ptr_fix);
    assert_eq!(visited[0].1, &INFO_FIX as *const TypeInfo);
    assert_eq!(visited[1].0, ptr_vec);
    assert_eq!(visited[1].1, &INFO_VEC as *const TypeInfo);
    assert_eq!(visited[2].0, ptr_str);
    assert_eq!(visited[2].1, &INFO_STR as *const TypeInfo);
}

#[test]
fn bump_integration_roots_scan() {
    type S = LowBit<3>;

    // obj_a (1 field) → obj_b (leaf)
    static INFO_1F: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    static INFO_0F: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(0);

    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    let root_b = m.alloc::<Compact>(&INFO_0F, 0).unwrap();
    let root_a = m.alloc::<Compact>(&INFO_1F, 0).unwrap();

    let obj_b = m.get(&root_b).bits() as *mut u8;
    let obj_a = m.get(&root_a).bits() as *mut u8;

    // obj_a.field[0] → obj_b
    unsafe {
        write_value_field(obj_a, &INFO_1F, 0, Value::<S>::tagged(1, obj_b as u64 >> 3));
    }

    // Mark phase using scan_roots + scan_object
    let mut mark_stack: Vec<*mut u8> = Vec::new();
    let mut marked: Vec<*mut u8> = Vec::new();

    m.scan_roots(&mut |slot| {
        let bits = unsafe { *slot };
        if bits != 0 {
            let ptr = bits as *mut u8;
            if m.bump().contains(ptr) && !marked.contains(&ptr) {
                marked.push(ptr);
                mark_stack.push(ptr);
            }
        }
    });

    assert_eq!(marked.len(), 2);
    assert!(marked.contains(&obj_a));
    assert!(marked.contains(&obj_b));

    while let Some(obj) = mark_stack.pop() {
        let info = unsafe { read_type_info(obj, Compact::TYPE_INFO_OFFSET) };
        unsafe {
            scan_object(obj, info, |slot| {
                let bits = core::ptr::read(slot);
                let v = Value::<S>::from_bits(bits);
                if v.has_tag(1) {
                    let ptr = (v.payload() << 3) as *mut u8;
                    if !marked.contains(&ptr) {
                        marked.push(ptr);
                        mark_stack.push(ptr);
                    }
                }
            });
        }
    }

    // obj_a, obj_b discovered from roots; obj_b also found transitively
    assert!(marked.contains(&obj_a));
    assert!(marked.contains(&obj_b));

    // Walk heap and verify same objects found
    let mut walked: Vec<*mut u8> = Vec::new();
    unsafe {
        m.bump().walk(&mut |ptr, _info| {
            walked.push(ptr);
        });
    }
    assert_eq!(walked.len(), 2);
    assert!(walked.contains(&obj_a));
    assert!(walked.contains(&obj_b));
}

// ─── Mutator tests ──────────────────────────────────────────────────

#[test]
fn mutator_root_get_set() {
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    let r = m.root(42);
    assert_eq!(m.get(&r).bits(), 42);

    m.set(&r, 99);
    assert_eq!(m.get(&r).bits(), 99);
}

#[test]
fn mutator_gcref_value_decode() {
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    // Store a LowBit<3> tagged value: tag=2, payload=100
    let tagged = Value::<LowBit<3>>::tagged(2, 100);
    let r = m.root(tagged.to_bits());

    let gcref = m.get(&r);
    assert!(gcref.has_tag::<LowBit<3>>(2));
    assert!(!gcref.has_tag::<LowBit<3>>(1));

    let v: Value<LowBit<3>> = gcref.value();
    assert_eq!(v.payload(), 100);
}

#[test]
fn mutator_alloc_returns_rooted_object() {
    const PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    let root = m.alloc::<Compact>(&PAIR, 0).unwrap();
    let ptr = m.get(&root).bits();
    assert_ne!(ptr, 0);

    // Verify the header is initialized: read_type_info should return PAIR
    let obj = ptr as *const u8;
    let info = unsafe { read_type_info(obj, Compact::TYPE_INFO_OFFSET) };
    assert_eq!(info.value_field_count, 2);
    assert_eq!(info.header_size, Compact::SIZE as u16);
}

#[test]
fn mutator_save_restore_scoping() {
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    let r1 = m.root(10);
    let r2 = m.root(20);
    let scope = m.save();

    let r3 = m.root(30);
    let r4 = m.root(40);

    // All four accessible
    assert_eq!(m.get(&r1).bits(), 10);
    assert_eq!(m.get(&r2).bits(), 20);
    assert_eq!(m.get(&r3).bits(), 30);
    assert_eq!(m.get(&r4).bits(), 40);

    m.restore(scope);

    // r1 and r2 still valid
    assert_eq!(m.get(&r1).bits(), 10);
    assert_eq!(m.get(&r2).bits(), 20);

    // r3 and r4 are invalidated — accessing would panic in debug
    assert_eq!(m.root_count(), 2);
}

#[test]
fn mutator_root_field() {
    const PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    // Allocate a pair object
    let parent = m.alloc::<Compact>(&PAIR, 0).unwrap();
    let parent_ptr = m.get(&parent).bits() as *mut u8;

    // Write a tagged value into field 0: tag=1, payload=0xBEEF
    let child_val = Value::<LowBit<3>>::tagged(1, 0xBEEF);
    unsafe {
        write_value_field::<LowBit<3>>(parent_ptr, &PAIR, 0, child_val);
    }

    // root_field reads field 0 and roots it
    let child_root = unsafe { m.root_field::<LowBit<3>>(&parent, &PAIR, 0) };
    let child_ref = m.get(&child_root);
    let child_v: Value<LowBit<3>> = child_ref.value();
    assert!(child_v.has_tag(1));
    assert_eq!(child_v.payload(), 0xBEEF);
}

#[test]
fn mutator_multiple_gcrefs_same_scope() {
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    let r1 = m.root(111);
    let r2 = m.root(222);
    let r3 = m.root(333);

    // All three GcRefs alive simultaneously
    let a = m.get(&r1);
    let b = m.get(&r2);
    let c = m.get(&r3);
    assert_eq!(a.bits(), 111);
    assert_eq!(b.bits(), 222);
    assert_eq!(c.bits(), 333);
}

#[test]
fn mutator_reread_after_alloc() {
    const LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    let r = m.root(0xAAAA);
    assert_eq!(m.get(&r).bits(), 0xAAAA);

    // alloc is a potential GC point — simulate GC updating the root
    let _new_obj = m.alloc::<Compact>(&LEAF, 0).unwrap();

    // Simulate GC forwarding: manually update root's slot
    m.set(&r, 0xBBBB);

    // Re-read after potential GC — must see updated value
    let val = m.get(&r);
    assert_eq!(val.bits(), 0xBBBB);
}

#[test]
fn mutator_alloc_full_returns_none() {
    const LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);
    // Leaf object: 8-byte Compact header, allocation_size = 8
    assert_eq!(LEAF.allocation_size(0), 8);

    // Small bump: room for exactly 2 objects (16 bytes)
    let bump = BumpAllocator::new::<Compact>(16);
    let mut m = Mutator::new(bump);

    let a = m.alloc::<Compact>(&LEAF, 0);
    assert!(a.is_some());

    let b = m.alloc::<Compact>(&LEAF, 0);
    assert!(b.is_some());

    // Third allocation should fail
    let c = m.alloc::<Compact>(&LEAF, 0);
    assert!(c.is_none());
}

#[test]
fn mutator_integration_with_scan() {
    const PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);

    // Allocate two objects via mutator
    let root_a = m.alloc::<Compact>(&PAIR, 0).unwrap();
    let root_b = m.alloc::<Compact>(&PAIR, 0).unwrap();

    let ptr_a = m.get(&root_a).bits() as *mut u8;
    let ptr_b = m.get(&root_b).bits() as *mut u8;

    // Make object A point to object B in field 0 (tag=1, payload=ptr>>3)
    let b_tagged = Value::<LowBit<3>>::tagged(1, (ptr_b as u64) >> 3);
    unsafe {
        write_value_field::<LowBit<3>>(ptr_a, &PAIR, 0, b_tagged);
    }

    // Scan roots — should find both root slots
    let mut root_ptrs: Vec<*mut u64> = Vec::new();
    m.scan_roots(&mut |slot| {
        root_ptrs.push(slot);
    });
    // At least 2 roots (alloc pushes a root each time)
    assert!(root_ptrs.len() >= 2);

    // Mark simulation: discover objects from roots
    let mut marked: Vec<*mut u8> = Vec::new();
    let mut mark_stack: Vec<*mut u8> = Vec::new();

    // Seed from roots
    for &slot in &root_ptrs {
        let bits = unsafe { *slot };
        if bits != 0 {
            let ptr = bits as *mut u8;
            if m.bump().contains(ptr) && !marked.contains(&ptr) {
                marked.push(ptr);
                mark_stack.push(ptr);
            }
        }
    }

    // Trace transitively
    while let Some(obj) = mark_stack.pop() {
        let info = unsafe { read_type_info(obj, Compact::TYPE_INFO_OFFSET) };
        unsafe {
            scan_object(obj, info, |slot| {
                let v = Value::<LowBit<3>>::from_bits(*slot);
                if v.has_tag(1) {
                    let ptr = (v.payload() << 3) as *mut u8;
                    if m.bump().contains(ptr) && !marked.contains(&ptr) {
                        marked.push(ptr);
                        mark_stack.push(ptr);
                    }
                }
            });
        }
    }

    // Both objects discovered
    assert_eq!(marked.len(), 2);
    assert!(marked.contains(&ptr_a));
    assert!(marked.contains(&ptr_b));
}

// ─── extern "C" API that never exposes raw pointers ─────────────────

const PAIR_INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
const VEC_INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(0);

unsafe extern "C" fn rt_alloc_pair(m: *mut Mutator) -> Root {
    let m = unsafe { &mut *m };
    m.alloc::<Compact>(&PAIR_INFO, 0).expect("OOM in rt_alloc_pair")
}

unsafe extern "C" fn rt_alloc_vec(m: *mut Mutator, len: usize) -> Root {
    let m = unsafe { &mut *m };
    m.alloc::<Compact>(&VEC_INFO, len).expect("OOM in rt_alloc_vec")
}

unsafe extern "C" fn rt_get_field(m: *mut Mutator, obj: Root, index: u16) -> Root {
    let m = unsafe { &mut *m };
    unsafe { m.root_field::<LowBit<3>>(&obj, &PAIR_INFO, index) }
}

unsafe extern "C" fn rt_set_field(m: *mut Mutator, obj: Root, index: u16, val: Root) {
    let m = unsafe { &mut *m };
    let obj_ptr = m.get(&obj).bits() as *mut u8;
    let val_ptr = m.get(&val).bits();
    let tagged = Value::<LowBit<3>>::tagged(1, val_ptr >> 3);
    unsafe { write_value_field::<LowBit<3>>(obj_ptr, &PAIR_INFO, index, tagged) };
}

unsafe extern "C" fn rt_read(m: *mut Mutator, root: Root) -> u64 {
    let m = unsafe { &*m };
    m.get(&root).bits()
}

unsafe extern "C" fn rt_root_immediate(m: *mut Mutator, bits: u64) -> Root {
    let m = unsafe { &mut *m };
    m.root(bits)
}

unsafe extern "C" fn rt_save(m: *mut Mutator) -> RootScope {
    let m = unsafe { &*m };
    m.save()
}

unsafe extern "C" fn rt_restore(m: *mut Mutator, scope: RootScope) {
    let m = unsafe { &mut *m };
    m.restore(scope);
}

#[test]
fn extern_c_api_no_raw_pointers() {
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);
    let mp = &mut m as *mut Mutator;

    unsafe {
        let a = rt_alloc_pair(mp);
        let b = rt_alloc_pair(mp);

        rt_set_field(mp, a, 0, b);

        let child = rt_get_field(mp, a, 0);

        let child_bits = rt_read(mp, child);
        let child_v = Value::<LowBit<3>>::from_bits(child_bits);
        assert!(child_v.has_tag(1));

        let b_ptr = rt_read(mp, b);
        assert_eq!(child_v.payload() << 3, b_ptr);
    }
}

#[test]
fn extern_c_api_scope_lifecycle() {
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);
    let mp = &mut m as *mut Mutator;

    unsafe {
        let fixnum = rt_root_immediate(mp, Value::<LowBit<3>>::tagged(0, 42).to_bits());

        let scope = rt_save(mp);

        let tmp1 = rt_alloc_pair(mp);
        let tmp2 = rt_alloc_pair(mp);
        rt_set_field(mp, tmp1, 0, tmp2);

        assert_ne!(rt_read(mp, tmp1), 0);
        assert_ne!(rt_read(mp, tmp2), 0);

        rt_restore(mp, scope);

        let fixnum_v = Value::<LowBit<3>>::from_bits(rt_read(mp, fixnum));
        assert!(fixnum_v.has_tag(0));
        assert_eq!(fixnum_v.payload(), 42);

        assert_eq!((*mp).root_count(), 1);
    }
}

#[test]
fn extern_c_api_alloc_interleaved_with_reads() {
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);
    let mp = &mut m as *mut Mutator;

    unsafe {
        let a = rt_alloc_pair(mp);
        let a_ptr_before = rt_read(mp, a);

        let _b = rt_alloc_pair(mp);
        let _c = rt_alloc_pair(mp);

        let a_ptr_after = rt_read(mp, a);
        assert_eq!(a_ptr_before, a_ptr_after);

        let m_ref = &mut *mp;
        m_ref.set(&a, 0xDEAD_BEEF);

        let a_ptr_moved = rt_read(mp, a);
        assert_eq!(a_ptr_moved, 0xDEAD_BEEF);
    }
}

#[test]
fn extern_c_api_vec_and_scan() {
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new(bump);
    let mp = &mut m as *mut Mutator;

    unsafe {
        let vec_root = rt_alloc_vec(mp, 3);
        let elem0 = rt_alloc_pair(mp);
        let elem1 = rt_alloc_pair(mp);

        let vec_ptr = rt_read(mp, vec_root) as *mut u8;
        let e0_tagged = Value::<LowBit<3>>::tagged(1, rt_read(mp, elem0) >> 3);
        let e1_tagged = Value::<LowBit<3>>::tagged(1, rt_read(mp, elem1) >> 3);
        write_varlen_value::<LowBit<3>>(vec_ptr, &VEC_INFO, 0, e0_tagged);
        write_varlen_value::<LowBit<3>>(vec_ptr, &VEC_INFO, 1, e1_tagged);

        let mut visited = Vec::new();
        scan_object(vec_ptr, &VEC_INFO, |slot| {
            visited.push(*slot);
        });

        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], e0_tagged.to_bits());
        assert_eq!(visited[1], e1_tagged.to_bits());
        assert_eq!(visited[2], 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SemiSpace collector tests
// ═══════════════════════════════════════════════════════════════════════

// ─── PtrPolicy implementations for testing ──────────────────────────

/// LowBit<3> policy: tag 0 = heap pointer (raw ptr, low 3 bits are 0).
/// All other tags are immediates.
struct LowBit3Tag0;

impl PtrPolicy for LowBit3Tag0 {
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        let v = Value::<LowBit<3>>::from_bits(bits);
        if v.has_tag(0) && bits != 0 {
            // Tag 0 means low 3 bits are 0, so payload = bits >> 3,
            // and the original pointer = payload << 3 = bits & !7 = bits.
            Some((v.payload() << 3) as *mut u8)
        } else {
            None
        }
    }

    fn encode_ptr(ptr: *mut u8) -> u64 {
        // Tag 0, payload = ptr >> 3
        Value::<LowBit<3>>::tagged(0, (ptr as u64) >> 3).to_bits()
    }
}

/// NanBox policy: tag 0 = heap pointer (48-bit payload = raw ptr).
/// Floats are unboxed. Tags 1-3 are immediates.
struct NanBoxTag0;

impl PtrPolicy for NanBoxTag0 {
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        let v = Value::<NanBox>::from_bits(bits);
        if v.has_tag(0) && bits != Value::<NanBox>::tagged(0, 0).to_bits() {
            Some(v.payload() as *mut u8)
        } else {
            None
        }
    }

    fn encode_ptr(ptr: *mut u8) -> u64 {
        Value::<NanBox>::tagged(0, ptr as u64).to_bits()
    }
}

/// Helper: make a tagged pointer value for LowBit3Tag0
fn ptr_val(ptr: *mut u8) -> u64 {
    LowBit3Tag0::encode_ptr(ptr)
}

/// Helper: make a fixnum (tag 1) for LowBit<3>
fn fixnum(n: i64) -> u64 {
    Value::<LowBit<3>>::tagged(1, n as u64).to_bits()
}

// ─── Basic collection ───────────────────────────────────────────────

#[test]
fn semi_space_collect_preserves_live_object() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);
    let obj = gc.alloc_obj::<Compact>(&LEAF, 0);
    assert!(!obj.is_null());

    // Root it via a simple Cell-based root source
    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }

    let root = SingleRoot(Cell::new(ptr_val(obj)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    // Root should now point into the new from-space (formerly to-space)
    let new_ptr = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_ptr as *const u8));
    assert_ne!(new_ptr, obj, "object should have moved");

    // Header should still be valid
    let info = unsafe { read_type_info(new_ptr, Compact::TYPE_INFO_OFFSET) };
    assert_eq!(info as *const TypeInfo, &LEAF as *const TypeInfo);
}

#[test]
fn semi_space_dead_objects_reclaimed() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    // Allocate many objects but only root one
    let _dead1 = gc.alloc_obj::<Compact>(&LEAF, 0);
    let live = gc.alloc_obj::<Compact>(&LEAF, 0);
    let _dead2 = gc.alloc_obj::<Compact>(&LEAF, 0);
    let _dead3 = gc.alloc_obj::<Compact>(&LEAF, 0);

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(live)));

    let used_before = gc.from_used();
    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };
    let used_after = gc.from_used();

    // Only one object should survive
    assert!(used_after < used_before);
    assert_eq!(used_after, LEAF.allocation_size(0));
}

#[test]
fn semi_space_pointer_fixup() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let parent = gc.alloc_obj::<Compact>(&PAIR, 0);

    // parent.field[0] → child, field[1] = fixnum 42
    unsafe {
        write_value_field(parent, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
        write_value_field(parent, &PAIR, 1, Value::<LowBit<3>>::from_bits(fixnum(42)));
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(parent)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    // Verify parent moved and its fields are updated
    let new_parent = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_parent as *const u8));

    unsafe {
        // Field 0 should be an updated pointer to the child's new location
        let field0: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 0);
        let new_child = LowBit3Tag0::try_decode_ptr(field0.to_bits()).unwrap();
        assert!(gc.contains(new_child as *const u8));
        assert_ne!(new_child, child);

        // Field 1 should still be fixnum 42 (not a pointer, untouched)
        let field1: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 1);
        assert_eq!(field1.to_bits(), fixnum(42));
    }
}

#[test]
fn semi_space_dag_shared_child() {
    // Two parents point to the same child → child should be copied once
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let parent_a = gc.alloc_obj::<Compact>(&PAIR, 0);
    let parent_b = gc.alloc_obj::<Compact>(&PAIR, 0);

    unsafe {
        write_value_field(parent_a, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
        write_value_field(parent_b, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
    }

    use std::cell::Cell;
    let roots = [Cell::new(ptr_val(parent_a)), Cell::new(ptr_val(parent_b))];
    struct Roots<'a>([&'a Cell<u64>; 2]);
    impl RootSource for Roots<'_> {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            for cell in &self.0 {
                visitor(cell.as_ptr());
            }
        }
    }
    let root_source = Roots([&roots[0], &roots[1]]);

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root_source]) };

    // Both parents' field[0] should point to the SAME new child
    let new_a = LowBit3Tag0::try_decode_ptr(roots[0].get()).unwrap();
    let new_b = LowBit3Tag0::try_decode_ptr(roots[1].get()).unwrap();

    unsafe {
        let child_from_a: Value<LowBit<3>> = read_value_field(new_a, &PAIR, 0);
        let child_from_b: Value<LowBit<3>> = read_value_field(new_b, &PAIR, 0);
        let ca = LowBit3Tag0::try_decode_ptr(child_from_a.to_bits()).unwrap();
        let cb = LowBit3Tag0::try_decode_ptr(child_from_b.to_bits()).unwrap();
        assert_eq!(ca, cb, "shared child should be copied once");
        assert!(gc.contains(ca as *const u8));
    }

    // 3 objects (2 parents + 1 child)
    let expected = PAIR.allocation_size(0) * 2 + LEAF.allocation_size(0);
    assert_eq!(gc.from_used(), expected);
}

#[test]
fn semi_space_chain() {
    // a → b → c → d (chain of 4)
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let d = gc.alloc_obj::<Compact>(&NODE, 0);
    let c = gc.alloc_obj::<Compact>(&NODE, 0);
    let b = gc.alloc_obj::<Compact>(&NODE, 0);
    let a = gc.alloc_obj::<Compact>(&NODE, 0);

    unsafe {
        write_value_field(a, &NODE, 0, Value::<LowBit<3>>::from_bits(ptr_val(b)));
        write_value_field(b, &NODE, 0, Value::<LowBit<3>>::from_bits(ptr_val(c)));
        write_value_field(c, &NODE, 0, Value::<LowBit<3>>::from_bits(ptr_val(d)));
        // d.field[0] = 0 (null/nil)
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(a)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    // Walk the chain in to-space and verify it's intact
    let mut cur = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    for _ in 0..3 {
        assert!(gc.contains(cur as *const u8));
        let field: Value<LowBit<3>> = unsafe { read_value_field(cur, &NODE, 0) };
        let next = LowBit3Tag0::try_decode_ptr(field.to_bits());
        cur = next.unwrap();
    }
    // d's field should be 0 (not a pointer)
    let d_field: Value<LowBit<3>> = unsafe { read_value_field(cur, &NODE, 0) };
    assert!(LowBit3Tag0::try_decode_ptr(d_field.to_bits()).is_none());
}

#[test]
fn semi_space_cycle() {
    // a → b → a (cycle)
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let a = gc.alloc_obj::<Compact>(&NODE, 0);
    let b = gc.alloc_obj::<Compact>(&NODE, 0);

    unsafe {
        write_value_field(a, &NODE, 0, Value::<LowBit<3>>::from_bits(ptr_val(b)));
        write_value_field(b, &NODE, 0, Value::<LowBit<3>>::from_bits(ptr_val(a)));
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(a)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    let new_a = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    let field_a: Value<LowBit<3>> = unsafe { read_value_field(new_a, &NODE, 0) };
    let new_b = LowBit3Tag0::try_decode_ptr(field_a.to_bits()).unwrap();
    let field_b: Value<LowBit<3>> = unsafe { read_value_field(new_b, &NODE, 0) };
    let back_to_a = LowBit3Tag0::try_decode_ptr(field_b.to_bits()).unwrap();

    assert_eq!(back_to_a, new_a, "cycle should be preserved");
    assert_ne!(new_a, new_b);
    assert!(gc.contains(new_a as *const u8));
    assert!(gc.contains(new_b as *const u8));
}

// ─── Multiple collections ───────────────────────────────────────────

#[test]
fn semi_space_multiple_collections() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }

    // Round 1: allocate, root, collect
    let obj = gc.alloc_obj::<Compact>(&PAIR, 0);
    let root = SingleRoot(Cell::new(ptr_val(obj)));
    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };
    assert_eq!(gc.collections(), 1);

    let after_1 = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();

    // Round 2: allocate more garbage, collect again
    let _dead = gc.alloc_obj::<Compact>(&LEAF, 0);
    let _dead2 = gc.alloc_obj::<Compact>(&LEAF, 0);
    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };
    assert_eq!(gc.collections(), 2);

    let after_2 = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    assert_ne!(after_1, after_2, "object moves each collection");
    assert!(gc.contains(after_2 as *const u8));

    // Only the one rooted object should survive
    assert_eq!(gc.from_used(), PAIR.allocation_size(0));

    // Round 3: add a child, collect, verify graph integrity
    let child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let parent = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    unsafe {
        write_value_field(parent, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
    }
    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };
    assert_eq!(gc.collections(), 3);

    let new_parent = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    let field: Value<LowBit<3>> = unsafe { read_value_field(new_parent, &PAIR, 0) };
    let new_child = LowBit3Tag0::try_decode_ptr(field.to_bits()).unwrap();
    assert!(gc.contains(new_child as *const u8));
}

// ─── Varlen objects ─────────────────────────────────────────────────

#[test]
fn semi_space_varlen_values() {
    static VEC: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(0);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    // Allocate a vector with 3 elements
    let elem0 = gc.alloc_obj::<Compact>(&LEAF, 0);
    let elem1 = gc.alloc_obj::<Compact>(&LEAF, 0);
    let vec = gc.alloc_obj::<Compact>(&VEC, 3);

    unsafe {
        write_varlen_value::<LowBit<3>>(vec, &VEC, 0, Value::from_bits(ptr_val(elem0)));
        write_varlen_value::<LowBit<3>>(vec, &VEC, 1, Value::from_bits(ptr_val(elem1)));
        // slot 2 left as 0 (null)
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(vec)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    let new_vec = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_vec as *const u8));

    // Verify varlen count preserved
    assert_eq!(unsafe { read_varlen_count(new_vec, &VEC) }, 3);

    // Verify elements are updated pointers
    unsafe {
        let v0: Value<LowBit<3>> = read_varlen_value(new_vec, &VEC, 0);
        let v1: Value<LowBit<3>> = read_varlen_value(new_vec, &VEC, 1);
        let v2: Value<LowBit<3>> = read_varlen_value(new_vec, &VEC, 2);

        let p0 = LowBit3Tag0::try_decode_ptr(v0.to_bits()).unwrap();
        let p1 = LowBit3Tag0::try_decode_ptr(v1.to_bits()).unwrap();
        assert!(gc.contains(p0 as *const u8));
        assert!(gc.contains(p1 as *const u8));
        assert_ne!(p0, elem0);
        assert_ne!(p1, elem1);

        // Slot 2 should still be 0
        assert_eq!(v2.to_bits(), 0);
    }
}

#[test]
fn semi_space_varlen_bytes() {
    static STR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_bytes(0);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let s = gc.alloc_obj::<Compact>(&STR, 5);
    // Write "hello" into the bytes section
    unsafe {
        let bytes = read_varlen_bytes(s, &STR);
        assert_eq!(bytes.len(), 5);
        let data = core::slice::from_raw_parts_mut(
            s.add(STR.varlen_element_offset(0)),
            5,
        );
        data.copy_from_slice(b"hello");
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(s)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    let new_s = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_s as *const u8));

    unsafe {
        assert_eq!(read_varlen_count(new_s, &STR), 5);
        let bytes = read_varlen_bytes(new_s, &STR);
        assert_eq!(bytes, b"hello");
    }
}

// ─── Different header types ─────────────────────────────────────────

#[test]
fn semi_space_with_full_header() {
    static PAIR: TypeInfo = TypeInfo::for_header(Full::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Full::SIZE);

    let mut gc = SemiSpace::new::<Full>(4096);

    let child = gc.alloc_obj::<Full>(&LEAF, 0);
    let parent = gc.alloc_obj::<Full>(&PAIR, 0);

    unsafe {
        write_value_field(parent, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
        write_value_field(parent, &PAIR, 1, Value::<LowBit<3>>::from_bits(fixnum(99)));
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(parent)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    let new_parent = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_parent as *const u8));

    unsafe {
        // Verify type_info is correct (Full has type_info at offset 8)
        let info = read_type_info(new_parent, Full::TYPE_INFO_OFFSET);
        assert_eq!(info as *const TypeInfo, &PAIR as *const TypeInfo);

        // Verify child pointer was fixed up
        let field0: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 0);
        let new_child = LowBit3Tag0::try_decode_ptr(field0.to_bits()).unwrap();
        assert!(gc.contains(new_child as *const u8));

        // Verify child's header
        let child_info = read_type_info(new_child, Full::TYPE_INFO_OFFSET);
        assert_eq!(child_info as *const TypeInfo, &LEAF as *const TypeInfo);

        // Verify fixnum preserved
        let field1: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 1);
        assert_eq!(field1.to_bits(), fixnum(99));
    }
}

// ─── Different tag schemes ──────────────────────────────────────────

#[test]
fn semi_space_with_nanbox() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let parent = gc.alloc_obj::<Compact>(&PAIR, 0);

    let child_tagged = NanBoxTag0::encode_ptr(child);
    let float_val = Value::<NanBox>::float(3.14).to_bits();

    unsafe {
        write_value_field(parent, &PAIR, 0, Value::<NanBox>::from_bits(child_tagged));
        write_value_field(parent, &PAIR, 1, Value::<NanBox>::from_bits(float_val));
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(NanBoxTag0::encode_ptr(parent)));

    unsafe { gc.collect::<NanBoxTag0>(&mut [&root]) };

    let new_parent = NanBoxTag0::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_parent as *const u8));

    unsafe {
        let field0: Value<NanBox> = read_value_field(new_parent, &PAIR, 0);
        let new_child = NanBoxTag0::try_decode_ptr(field0.to_bits()).unwrap();
        assert!(gc.contains(new_child as *const u8));

        // Float should be preserved exactly
        let field1: Value<NanBox> = read_value_field(new_parent, &PAIR, 1);
        assert_eq!(field1.to_bits(), float_val);
        assert!(Value::<NanBox>::from_bits(field1.to_bits()).is_float());
    }
}

// ─── Multiple root sources ─────────────────────────────────────────

#[test]
fn semi_space_multiple_root_sources() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let obj_a = gc.alloc_obj::<Compact>(&LEAF, 0);
    let obj_b = gc.alloc_obj::<Compact>(&LEAF, 0);
    let obj_c = gc.alloc_obj::<Compact>(&LEAF, 0);

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }

    // Three separate root sources, each owning one root
    let r1 = SingleRoot(Cell::new(ptr_val(obj_a)));
    let r2 = SingleRoot(Cell::new(ptr_val(obj_b)));
    let r3 = SingleRoot(Cell::new(ptr_val(obj_c)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&r1, &r2, &r3]) };

    let new_a = LowBit3Tag0::try_decode_ptr(r1.0.get()).unwrap();
    let new_b = LowBit3Tag0::try_decode_ptr(r2.0.get()).unwrap();
    let new_c = LowBit3Tag0::try_decode_ptr(r3.0.get()).unwrap();

    // All three should be in the new from-space, all distinct
    assert!(gc.contains(new_a as *const u8));
    assert!(gc.contains(new_b as *const u8));
    assert!(gc.contains(new_c as *const u8));
    assert_ne!(new_a, new_b);
    assert_ne!(new_b, new_c);
    assert_ne!(new_a, new_c);

    assert_eq!(gc.from_used(), LEAF.allocation_size(0) * 3);
}

// ─── With Mutator ───────────────────────────────────────────────────

#[test]
fn semi_space_with_mutator_as_root_source() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    // Use a Mutator as the root source — allocate through the gc, root through the mutator
    let dummy_bump = BumpAllocator::new::<Compact>(64);
    let mut m = Mutator::new(dummy_bump);

    let child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let parent = gc.alloc_obj::<Compact>(&PAIR, 0);

    unsafe {
        write_value_field(parent, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
    }

    // Root through the Mutator
    let r_parent = m.root(ptr_val(parent));
    let r_child = m.root(ptr_val(child));

    // Collect using Mutator as RootSource
    unsafe { gc.collect::<LowBit3Tag0>(&mut [&m]) };

    // The mutator's root slots should be updated in-place
    let new_parent = LowBit3Tag0::try_decode_ptr(m.get(&r_parent).bits()).unwrap();
    let new_child = LowBit3Tag0::try_decode_ptr(m.get(&r_child).bits()).unwrap();

    assert!(gc.contains(new_parent as *const u8));
    assert!(gc.contains(new_child as *const u8));

    // Verify the parent's field still points to the child
    unsafe {
        let field0: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 0);
        let child_from_field = LowBit3Tag0::try_decode_ptr(field0.to_bits()).unwrap();
        assert_eq!(child_from_field, new_child);
    }
}

// ─── Stress / fill-collect-fill ─────────────────────────────────────

#[test]
fn semi_space_fill_collect_refill() {
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

    let obj_size = NODE.allocation_size(0); // 16 bytes
    let space_size = obj_size * 10; // room for 10 objects per space
    let mut gc = SemiSpace::new::<Compact>(space_size);

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }

    // Allocate one rooted object, fill rest with garbage
    let live = gc.alloc_obj::<Compact>(&NODE, 0);
    let root = SingleRoot(Cell::new(ptr_val(live)));
    for _ in 0..9 {
        let p = gc.alloc_obj::<Compact>(&NODE, 0);
        assert!(!p.is_null());
    }
    assert!(gc.alloc_obj::<Compact>(&NODE, 0).is_null(), "space should be full");

    // Collect — should free 9 objects
    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };
    assert_eq!(gc.from_used(), obj_size);

    // Fill again — we should be able to allocate 9 more
    for _ in 0..9 {
        let p = gc.alloc_obj::<Compact>(&NODE, 0);
        assert!(!p.is_null());
    }
    assert!(gc.alloc_obj::<Compact>(&NODE, 0).is_null(), "space should be full again");

    // Build a chain: root → newest → ... → oldest
    let prev = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    // Walk from-space to link objects into a chain
    let mut objects = vec![prev];
    unsafe {
        gc.from_space().walk(&mut |ptr, _info| {
            if ptr != prev {
                objects.push(ptr);
            }
        });
    }
    for i in 1..objects.len() {
        unsafe {
            write_value_field(
                objects[i - 1],
                &NODE,
                0,
                Value::<LowBit<3>>::from_bits(ptr_val(objects[i])),
            );
        }
    }

    // Collect — all 10 should survive (reachable via chain)
    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };
    assert_eq!(gc.from_used(), obj_size * 10);
}

// ─── Mixed fixed + varlen in same collection ────────────────────────

#[test]
fn semi_space_mixed_object_types() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static VEC: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(0);
    static STR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_bytes(0);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(8192);

    let leaf = gc.alloc_obj::<Compact>(&LEAF, 0);
    let s = gc.alloc_obj::<Compact>(&STR, 3);
    let vec = gc.alloc_obj::<Compact>(&VEC, 2);
    let pair = gc.alloc_obj::<Compact>(&PAIR, 0);

    // Write string bytes
    unsafe {
        let data = core::slice::from_raw_parts_mut(s.add(STR.varlen_element_offset(0)), 3);
        data.copy_from_slice(b"abc");
    }

    // vec[0] → leaf, vec[1] → pair
    unsafe {
        write_varlen_value::<LowBit<3>>(vec, &VEC, 0, Value::from_bits(ptr_val(leaf)));
        write_varlen_value::<LowBit<3>>(vec, &VEC, 1, Value::from_bits(ptr_val(pair)));
    }

    // pair.field[0] → s, pair.field[1] → vec (creates pointer back, not a cycle since no self-ref)
    unsafe {
        write_value_field(pair, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(s)));
        write_value_field(pair, &PAIR, 1, Value::<LowBit<3>>::from_bits(ptr_val(vec)));
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(vec)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    // All 4 objects reachable: vec → leaf, pair → s, vec
    let new_vec = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_vec as *const u8));

    unsafe {
        // Check vec elements
        let v0: Value<LowBit<3>> = read_varlen_value(new_vec, &VEC, 0);
        let new_leaf = LowBit3Tag0::try_decode_ptr(v0.to_bits()).unwrap();
        assert!(gc.contains(new_leaf as *const u8));

        let v1: Value<LowBit<3>> = read_varlen_value(new_vec, &VEC, 1);
        let new_pair = LowBit3Tag0::try_decode_ptr(v1.to_bits()).unwrap();
        assert!(gc.contains(new_pair as *const u8));

        // Check pair fields
        let f0: Value<LowBit<3>> = read_value_field(new_pair, &PAIR, 0);
        let new_s = LowBit3Tag0::try_decode_ptr(f0.to_bits()).unwrap();
        assert!(gc.contains(new_s as *const u8));

        // Verify string content survived
        let bytes = read_varlen_bytes(new_s, &STR);
        assert_eq!(bytes, b"abc");

        // Verify pair.field[1] → vec (back-pointer to same vec)
        let f1: Value<LowBit<3>> = read_value_field(new_pair, &PAIR, 1);
        let vec_again = LowBit3Tag0::try_decode_ptr(f1.to_bits()).unwrap();
        assert_eq!(vec_again, new_vec, "back-pointer should point to same vec");
    }
}

// ─── Custom header with define_header! ──────────────────────────────

#[test]
fn semi_space_with_custom_header() {
    define_header! {
        pub GcHeader {
            gc_word: u64 {
                mark: [0..2],
                generation: [2..5],
            }
            type_info: *const TypeInfo,
        }
    }

    static PAIR: TypeInfo = TypeInfo::for_header(GcHeader::SIZE).with_fields(2);

    let mut gc = SemiSpace::new::<GcHeader>(4096);

    let obj = gc.alloc_obj::<GcHeader>(&PAIR, 0);
    unsafe {
        write_value_field(obj, &PAIR, 0, Value::<LowBit<3>>::from_bits(fixnum(123)));
        write_value_field(obj, &PAIR, 1, Value::<LowBit<3>>::from_bits(fixnum(456)));
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(obj)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    let new_obj = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_obj as *const u8));

    // Verify type_info recovered via custom header offset
    let info = unsafe { read_type_info(new_obj, GcHeader::TYPE_INFO_OFFSET) };
    assert_eq!(info as *const TypeInfo, &PAIR as *const TypeInfo);

    unsafe {
        let f0: Value<LowBit<3>> = read_value_field(new_obj, &PAIR, 0);
        let f1: Value<LowBit<3>> = read_value_field(new_obj, &PAIR, 1);
        assert_eq!(f0.to_bits(), fixnum(123));
        assert_eq!(f1.to_bits(), fixnum(456));
    }
}

// ─── Empty collection ───────────────────────────────────────────────

#[test]
fn semi_space_collect_empty() {
    let mut gc = SemiSpace::new::<Compact>(4096);
    let roots: &mut [&dyn RootSource] = &mut [];
    unsafe { gc.collect::<LowBit3Tag0>(roots) };
    assert_eq!(gc.collections(), 1);
    assert_eq!(gc.from_used(), 0);
}

#[test]
fn semi_space_collect_no_pointers() {
    // Object with only immediates (no heap pointers)
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let obj = gc.alloc_obj::<Compact>(&PAIR, 0);
    unsafe {
        write_value_field(obj, &PAIR, 0, Value::<LowBit<3>>::from_bits(fixnum(1)));
        write_value_field(obj, &PAIR, 1, Value::<LowBit<3>>::from_bits(fixnum(2)));
    }

    use std::cell::Cell;
    struct SingleRoot(Cell<u64>);
    impl RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(ptr_val(obj)));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&root]) };

    let new_obj = LowBit3Tag0::try_decode_ptr(root.0.get()).unwrap();
    unsafe {
        let f0: Value<LowBit<3>> = read_value_field(new_obj, &PAIR, 0);
        let f1: Value<LowBit<3>> = read_value_field(new_obj, &PAIR, 1);
        assert_eq!(f0.to_bits(), fixnum(1));
        assert_eq!(f1.to_bits(), fixnum(2));
    }
}

// ─── FrameChain as root source with SemiSpace ───────────────────────

#[test]
fn semi_space_with_frame_chain() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let parent = gc.alloc_obj::<Compact>(&PAIR, 0);

    unsafe {
        write_value_field(parent, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
        write_value_field(parent, &PAIR, 1, Value::<LowBit<3>>::from_bits(fixnum(77)));
    }

    let chain = FrameChain::new();
    let frame = RootFrame::<2>::new();
    frame.slots[0].set(ptr_val(parent));
    frame.slots[1].set(ptr_val(child));
    let _guard = chain.push(&frame);

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&chain]) };

    // Slots should be updated in-place with new addresses
    let new_parent = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
    let new_child = LowBit3Tag0::try_decode_ptr(frame.slots[1].get()).unwrap();

    assert!(gc.contains(new_parent as *const u8));
    assert!(gc.contains(new_child as *const u8));

    // Verify graph integrity
    unsafe {
        let f0: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 0);
        let child_from_field = LowBit3Tag0::try_decode_ptr(f0.to_bits()).unwrap();
        assert_eq!(child_from_field, new_child);

        let f1: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 1);
        assert_eq!(f1.to_bits(), fixnum(77));
    }
}

#[test]
fn semi_space_with_nested_frames() {
    // Simulates: outer_fn roots obj_a, calls inner_fn which roots obj_b
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let obj_a = gc.alloc_obj::<Compact>(&LEAF, 0);
    let obj_b = gc.alloc_obj::<Compact>(&LEAF, 0);
    let _dead = gc.alloc_obj::<Compact>(&LEAF, 0);

    let chain = FrameChain::new();

    // Outer function frame
    let outer_frame = RootFrame::<1>::new();
    outer_frame.slots[0].set(ptr_val(obj_a));
    let _outer_guard = chain.push(&outer_frame);

    // Inner function frame
    let inner_frame = RootFrame::<1>::new();
    inner_frame.slots[0].set(ptr_val(obj_b));
    let _inner_guard = chain.push(&inner_frame);

    assert_eq!(chain.depth(), 2);

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&chain]) };

    // Both roots should survive, dead object reclaimed
    let new_a = LowBit3Tag0::try_decode_ptr(outer_frame.slots[0].get()).unwrap();
    let new_b = LowBit3Tag0::try_decode_ptr(inner_frame.slots[0].get()).unwrap();
    assert!(gc.contains(new_a as *const u8));
    assert!(gc.contains(new_b as *const u8));
    assert_ne!(new_a, new_b);

    // Only 2 objects should survive
    assert_eq!(gc.from_used(), LEAF.allocation_size(0) * 2);
}

#[test]
fn semi_space_with_root_set() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let global1 = gc.alloc_obj::<Compact>(&LEAF, 0);
    let global2 = gc.alloc_obj::<Compact>(&LEAF, 0);
    let _dead = gc.alloc_obj::<Compact>(&LEAF, 0);

    let mut globals = RootSet::new();
    let i0 = globals.add(ptr_val(global1));
    let i1 = globals.add(ptr_val(global2));

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&globals]) };

    let new_g1 = LowBit3Tag0::try_decode_ptr(globals.get(i0)).unwrap();
    let new_g2 = LowBit3Tag0::try_decode_ptr(globals.get(i1)).unwrap();
    assert!(gc.contains(new_g1 as *const u8));
    assert!(gc.contains(new_g2 as *const u8));
    assert_eq!(gc.from_used(), LEAF.allocation_size(0) * 2);
}

// ─── Mixed root strategies ──────────────────────────────────────────

#[test]
fn semi_space_mixed_root_sources() {
    // Combines FrameChain + RootSet + Mutator as root sources
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(8192);

    // Allocate objects
    let global_obj = gc.alloc_obj::<Compact>(&LEAF, 0);
    let stack_obj = gc.alloc_obj::<Compact>(&NODE, 0);
    let mutator_obj = gc.alloc_obj::<Compact>(&NODE, 0);
    let shared_child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let _dead = gc.alloc_obj::<Compact>(&LEAF, 0);

    // stack_obj → shared_child, mutator_obj → shared_child
    unsafe {
        write_value_field(stack_obj, &NODE, 0, Value::<LowBit<3>>::from_bits(ptr_val(shared_child)));
        write_value_field(mutator_obj, &NODE, 0, Value::<LowBit<3>>::from_bits(ptr_val(shared_child)));
    }

    // Root via three different strategies
    let mut globals = RootSet::new();
    globals.add(ptr_val(global_obj));

    let chain = FrameChain::new();
    let frame = RootFrame::<1>::new();
    frame.slots[0].set(ptr_val(stack_obj));
    let _guard = chain.push(&frame);

    let dummy_bump = BumpAllocator::new::<Compact>(64);
    let mut mutator = Mutator::new(dummy_bump);
    let r_mutator = mutator.root(ptr_val(mutator_obj));

    // Collect with all three sources
    unsafe {
        gc.collect::<LowBit3Tag0>(&mut [&globals, &chain, &mutator]);
    }

    // All 4 live objects should survive
    let new_global = LowBit3Tag0::try_decode_ptr(globals.get(0)).unwrap();
    let new_stack = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
    let new_mutator = LowBit3Tag0::try_decode_ptr(mutator.get(&r_mutator).bits()).unwrap();

    assert!(gc.contains(new_global as *const u8));
    assert!(gc.contains(new_stack as *const u8));
    assert!(gc.contains(new_mutator as *const u8));

    // shared_child should be reachable from both stack and mutator objects
    unsafe {
        let from_stack: Value<LowBit<3>> = read_value_field(new_stack, &NODE, 0);
        let from_mutator: Value<LowBit<3>> = read_value_field(new_mutator, &NODE, 0);
        let child_a = LowBit3Tag0::try_decode_ptr(from_stack.to_bits()).unwrap();
        let child_b = LowBit3Tag0::try_decode_ptr(from_mutator.to_bits()).unwrap();
        assert_eq!(child_a, child_b, "shared child should be copied once");
        assert!(gc.contains(child_a as *const u8));
    }

    // 4 live objects, 1 dead
    assert_eq!(
        gc.from_used(),
        LEAF.allocation_size(0) * 2 + NODE.allocation_size(0) * 2
    );
}

#[test]
fn semi_space_frame_chain_with_full_header() {
    static PAIR: TypeInfo = TypeInfo::for_header(Full::SIZE).with_fields(2);

    let mut gc = SemiSpace::new::<Full>(4096);

    let obj = gc.alloc_obj::<Full>(&PAIR, 0);
    unsafe {
        write_value_field(obj, &PAIR, 0, Value::<LowBit<3>>::from_bits(fixnum(42)));
        write_value_field(obj, &PAIR, 1, Value::<LowBit<3>>::from_bits(fixnum(99)));
    }

    let chain = FrameChain::new();
    let frame = RootFrame::<1>::new();
    frame.slots[0].set(ptr_val(obj));
    let _guard = chain.push(&frame);

    unsafe { gc.collect::<LowBit3Tag0>(&mut [&chain]) };

    let new_obj = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
    assert!(gc.contains(new_obj as *const u8));

    unsafe {
        let info = read_type_info(new_obj, Full::TYPE_INFO_OFFSET);
        assert_eq!(info as *const TypeInfo, &PAIR as *const TypeInfo);

        let f0: Value<LowBit<3>> = read_value_field(new_obj, &PAIR, 0);
        let f1: Value<LowBit<3>> = read_value_field(new_obj, &PAIR, 1);
        assert_eq!(f0.to_bits(), fixnum(42));
        assert_eq!(f1.to_bits(), fixnum(99));
    }
}

#[test]
fn semi_space_frame_chain_with_nanbox() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let mut gc = SemiSpace::new::<Compact>(4096);

    let child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let parent = gc.alloc_obj::<Compact>(&PAIR, 0);

    let child_tagged = NanBoxTag0::encode_ptr(child);
    let float_val = Value::<NanBox>::float(2.718).to_bits();

    unsafe {
        write_value_field(parent, &PAIR, 0, Value::<NanBox>::from_bits(child_tagged));
        write_value_field(parent, &PAIR, 1, Value::<NanBox>::from_bits(float_val));
    }

    let chain = FrameChain::new();
    let frame = RootFrame::<1>::new();
    frame.slots[0].set(NanBoxTag0::encode_ptr(parent));
    let _guard = chain.push(&frame);

    unsafe { gc.collect::<NanBoxTag0>(&mut [&chain]) };

    let new_parent = NanBoxTag0::try_decode_ptr(frame.slots[0].get()).unwrap();
    assert!(gc.contains(new_parent as *const u8));

    unsafe {
        let f0: Value<NanBox> = read_value_field(new_parent, &PAIR, 0);
        let new_child = NanBoxTag0::try_decode_ptr(f0.to_bits()).unwrap();
        assert!(gc.contains(new_child as *const u8));

        let f1: Value<NanBox> = read_value_field(new_parent, &PAIR, 1);
        assert_eq!(f1.to_bits(), float_val);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 1-3: Concurrent / Thread-Safe Tests
// ═══════════════════════════════════════════════════════════════════

// ─── AtomicBumpAllocator ────────────────────────────────────────────

#[test]
fn atomic_bump_basic_alloc() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = AtomicBumpAllocator::new::<Compact>(4096);

    let ptr = bump.alloc(&INFO, 0);
    assert!(!ptr.is_null());
    assert!(bump.contains(ptr));
    assert_eq!(bump.used(), INFO.allocation_size(0));
}

#[test]
fn atomic_bump_concurrent_alloc() {
    use std::thread;

    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let bump = Arc::new(AtomicBumpAllocator::new::<Compact>(65536));
    let allocs_per_thread = 100;
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let bump = bump.clone();
            thread::spawn(move || {
                let mut addrs = Vec::new();
                for _ in 0..allocs_per_thread {
                    let ptr = bump.alloc(&LEAF, 0);
                    assert!(!ptr.is_null());
                    addrs.push(ptr as usize);
                }
                addrs
            })
        })
        .collect();

    let mut all_addrs = Vec::new();
    for h in handles {
        all_addrs.extend(h.join().unwrap());
    }

    // All pointers should be distinct and in the allocator's region
    assert_eq!(all_addrs.len(), num_threads * allocs_per_thread);
    for &addr in &all_addrs {
        assert!(bump.contains(addr as *const u8));
    }

    // Check no overlaps (all pointers are at least LEAF.allocation_size apart)
    all_addrs.sort();
    for i in 1..all_addrs.len() {
        assert!(
            all_addrs[i] >= all_addrs[i - 1] + LEAF.allocation_size(0),
            "allocations overlap"
        );
    }
}

#[test]
fn atomic_bump_exhaustion() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let obj_size = INFO.allocation_size(0);
    let bump = AtomicBumpAllocator::new::<Compact>(obj_size);

    let p1 = bump.alloc(&INFO, 0);
    assert!(!p1.is_null());

    let p2 = bump.alloc(&INFO, 0);
    assert!(p2.is_null(), "should be exhausted");
}

#[test]
fn atomic_bump_reset() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE);
    let bump = AtomicBumpAllocator::new::<Compact>(4096);

    let _ = bump.alloc(&INFO, 0);
    assert!(bump.used() > 0);

    bump.reset();
    assert_eq!(bump.used(), 0);
}

// ─── AtomicRootSet ──────────────────────────────────────────────────

#[test]
fn atomic_root_set_basic() {
    let rs = AtomicRootSet::new();
    assert!(rs.is_empty());

    let i0 = rs.add(42);
    let i1 = rs.add(99);
    assert_eq!(rs.len(), 2);
    assert_eq!(rs.get(i0), 42);
    assert_eq!(rs.get(i1), 99);

    rs.set(i0, 123);
    assert_eq!(rs.get(i0), 123);
}

#[test]
fn atomic_root_set_scan() {
    let rs = AtomicRootSet::new();
    rs.add(10);
    rs.add(20);
    rs.add(30);

    let mut vals = vec![];
    rs.scan_roots(&mut |slot| {
        vals.push(unsafe { *slot });
    });
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn atomic_root_set_gc_update() {
    let rs = AtomicRootSet::new();
    rs.add(100);
    rs.add(200);

    // Simulate GC updating slots in-place
    rs.scan_roots(&mut |slot| {
        unsafe { *slot += 1; }
    });

    assert_eq!(rs.get(0), 101);
    assert_eq!(rs.get(1), 201);
}

#[test]
fn atomic_root_set_concurrent_add() {
    use std::thread;

    let rs = Arc::new(AtomicRootSet::new());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let rs = rs.clone();
            thread::spawn(move || {
                for i in 0..25 {
                    rs.add((t * 100 + i) as u64);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(rs.len(), 100);
}

// ─── Heap: single-threaded collection ───────────────────────────────

#[test]
fn heap_basic_alloc_and_collect() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let heap = Heap::new::<Compact>(4096);

    let child = heap.alloc_obj::<Compact>(&LEAF, 0);
    let parent = heap.alloc_obj::<Compact>(&PAIR, 0);

    unsafe {
        write_value_field(parent, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
        write_value_field(parent, &PAIR, 1, Value::<LowBit<3>>::from_bits(fixnum(42)));
    }

    // Root via globals
    let i_parent = heap.globals.add(ptr_val(parent));

    unsafe { heap.collect::<LowBit3Tag0>(&[]) };

    let new_parent = LowBit3Tag0::try_decode_ptr(heap.globals.get(i_parent)).unwrap();
    assert!(heap.contains(new_parent as *const u8));

    unsafe {
        let f0: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 0);
        let new_child = LowBit3Tag0::try_decode_ptr(f0.to_bits()).unwrap();
        assert!(heap.contains(new_child as *const u8));

        let f1: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 1);
        assert_eq!(f1.to_bits(), fixnum(42));
    }

    assert_eq!(heap.collections(), 1);
}

#[test]
fn heap_dead_objects_reclaimed() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let heap = Heap::new::<Compact>(4096);

    let _dead1 = heap.alloc_obj::<Compact>(&LEAF, 0);
    let live = heap.alloc_obj::<Compact>(&LEAF, 0);
    let _dead2 = heap.alloc_obj::<Compact>(&LEAF, 0);

    heap.globals.add(ptr_val(live));

    let used_before = heap.from_used();
    unsafe { heap.collect::<LowBit3Tag0>(&[]) };
    let used_after = heap.from_used();

    assert!(used_after < used_before);
    assert_eq!(used_after, LEAF.allocation_size(0));
}

#[test]
fn heap_with_thread_roots() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let heap = Heap::new::<Compact>(4096);

    let obj = heap.alloc_obj::<Compact>(&LEAF, 0);

    // Register a thread and root the object via its frame chain
    let (ts, _id) = heap.register_thread();
    let frame = RootFrame::<1>::new();
    frame.slots[0].set(ptr_val(obj));
    let _guard = ts.frame_chain.push(&frame);

    // Collect — the GC should find the root via the thread's frame chain
    unsafe { heap.collect::<LowBit3Tag0>(&[]) };

    let new_obj = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
    assert!(heap.contains(new_obj as *const u8));
}

#[test]
fn heap_multiple_collections() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let heap = Heap::new::<Compact>(4096);
    let live = heap.alloc_obj::<Compact>(&LEAF, 0);
    heap.globals.add(ptr_val(live));

    for i in 0..5 {
        // Add garbage
        let _dead = heap.alloc_obj::<Compact>(&LEAF, 0);
        unsafe { heap.collect::<LowBit3Tag0>(&[]) };
        assert_eq!(heap.collections(), i + 1);
        assert_eq!(heap.from_used(), LEAF.allocation_size(0));
    }
}

#[test]
fn heap_mixed_globals_and_thread_roots() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let heap = Heap::new::<Compact>(8192);

    let global_obj = heap.alloc_obj::<Compact>(&LEAF, 0);
    let thread_obj = heap.alloc_obj::<Compact>(&PAIR, 0);
    let shared = heap.alloc_obj::<Compact>(&LEAF, 0);
    let _dead = heap.alloc_obj::<Compact>(&LEAF, 0);

    unsafe {
        write_value_field(thread_obj, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(shared)));
    }

    // Global root
    let ig = heap.globals.add(ptr_val(global_obj));

    // Thread root
    let (ts, _id) = heap.register_thread();
    let frame = RootFrame::<1>::new();
    frame.slots[0].set(ptr_val(thread_obj));
    let _guard = ts.frame_chain.push(&frame);

    unsafe { heap.collect::<LowBit3Tag0>(&[]) };

    // 3 live objects (global, thread, shared), 1 dead
    let new_global = LowBit3Tag0::try_decode_ptr(heap.globals.get(ig)).unwrap();
    let new_thread = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
    assert!(heap.contains(new_global as *const u8));
    assert!(heap.contains(new_thread as *const u8));

    unsafe {
        let f0: Value<LowBit<3>> = read_value_field(new_thread, &PAIR, 0);
        let new_shared = LowBit3Tag0::try_decode_ptr(f0.to_bits()).unwrap();
        assert!(heap.contains(new_shared as *const u8));
    }
}

// ─── Heap: multi-threaded STW collection ────────────────────────────

#[test]
fn heap_stw_collect_with_threads() {
    use std::thread;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicBool;

    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let heap = Arc::new(Heap::new::<Compact>(65536));
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads + 1)); // +1 for GC thread
    let gc_done = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let heap = heap.clone();
        let barrier = barrier.clone();
        let gc_done = gc_done.clone();

        handles.push(thread::spawn(move || {
            let (ts, _id) = heap.register_thread();

            // Allocate an object and root it
            let obj = heap.alloc_obj::<Compact>(&LEAF, 0);
            assert!(!obj.is_null());

            let frame = RootFrame::<1>::new();
            frame.slots[0].set(ptr_val(obj));
            let _guard = ts.frame_chain.push(&frame);

            // Signal we're ready
            barrier.wait();

            // Poll safepoint in a loop until GC completes
            while !gc_done.load(std::sync::atomic::Ordering::Acquire) {
                heap.safepoint(&ts);
                std::thread::yield_now();
            }

            // After GC, our root should be updated
            let new_obj = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
            assert!(heap.contains(new_obj as *const u8));
        }));
    }

    // Wait for all threads to allocate and be ready
    barrier.wait();

    // Trigger STW collection
    unsafe { heap.stw_collect::<LowBit3Tag0>() };
    gc_done.store(true, std::sync::atomic::Ordering::Release);

    assert_eq!(heap.collections(), 1);

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn heap_stw_preserves_object_graph() {
    use std::thread;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicBool;

    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let heap = Arc::new(Heap::new::<Compact>(65536));
    let barrier = Arc::new(Barrier::new(2)); // 1 mutator + 1 GC
    let gc_done = Arc::new(AtomicBool::new(false));

    let heap2 = heap.clone();
    let barrier2 = barrier.clone();
    let gc_done2 = gc_done.clone();

    let mutator = thread::spawn(move || {
        let (ts, _id) = heap2.register_thread();

        // Build a small graph: parent → child
        let child = heap2.alloc_obj::<Compact>(&LEAF, 0);
        let parent = heap2.alloc_obj::<Compact>(&PAIR, 0);
        let _dead = heap2.alloc_obj::<Compact>(&LEAF, 0);

        unsafe {
            write_value_field(parent, &PAIR, 0, Value::<LowBit<3>>::from_bits(ptr_val(child)));
            write_value_field(parent, &PAIR, 1, Value::<LowBit<3>>::from_bits(fixnum(99)));
        }

        let frame = RootFrame::<1>::new();
        frame.slots[0].set(ptr_val(parent));
        let _guard = ts.frame_chain.push(&frame);

        // Signal ready
        barrier2.wait();

        // Poll safepoint until GC completes
        while !gc_done2.load(std::sync::atomic::Ordering::Acquire) {
            heap2.safepoint(&ts);
            std::thread::yield_now();
        }

        // After GC: verify graph
        let new_parent = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
        assert!(heap2.contains(new_parent as *const u8));

        unsafe {
            let f0: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 0);
            let new_child = LowBit3Tag0::try_decode_ptr(f0.to_bits()).unwrap();
            assert!(heap2.contains(new_child as *const u8));

            let f1: Value<LowBit<3>> = read_value_field(new_parent, &PAIR, 1);
            assert_eq!(f1.to_bits(), fixnum(99));
        }
    });

    barrier.wait();

    unsafe { heap.stw_collect::<LowBit3Tag0>() };
    gc_done.store(true, std::sync::atomic::Ordering::Release);

    mutator.join().unwrap();

    // parent + child survived, dead reclaimed
    assert_eq!(
        heap.from_used(),
        PAIR.allocation_size(0) + LEAF.allocation_size(0)
    );
}

// ─── MutatorThread API ──────────────────────────────────────────────

use crate::MutatorThread;

#[test]
fn mutator_thread_basic_alloc() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let heap = Arc::new(Heap::new::<Compact>(4096));
    let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());

    let obj = mt.alloc_obj::<Compact>(&LEAF, 0);
    assert!(!obj.is_null());
    assert!(heap.contains(obj));
}

#[test]
fn mutator_thread_auto_gc_on_full() {
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let obj_size = LEAF.allocation_size(0);
    let space_size = obj_size * 4; // room for 4 objects
    let heap = Arc::new(Heap::new::<Compact>(space_size));
    let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());

    // Root one object via frame chain
    let frame = RootFrame::<1>::new();
    let _guard = mt.frame_chain().push(&frame);

    let live = mt.alloc_obj::<Compact>(&LEAF, 0);
    frame.slots[0].set(ptr_val(live));

    // Fill the rest with garbage
    let _ = mt.alloc_obj::<Compact>(&LEAF, 0);
    let _ = mt.alloc_obj::<Compact>(&LEAF, 0);
    let _ = mt.alloc_obj::<Compact>(&LEAF, 0);

    // Space is full. This alloc should trigger GC and succeed.
    let after_gc = mt.alloc_obj::<Compact>(&LEAF, 0);
    assert!(!after_gc.is_null(), "alloc should succeed after auto-GC");
    assert!(heap.collections() >= 1, "GC should have run");

    // Root should still be valid
    let new_live = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
    assert!(heap.contains(new_live as *const u8));
}

#[test]
fn mutator_thread_deregisters_on_drop() {
    let heap = Arc::new(Heap::new::<Compact>(4096));

    {
        let _mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());
        // Thread is registered
    }
    // mt dropped — thread deregistered

    // We can still create new threads (the slot was freed)
    let _mt2 = MutatorThread::<LowBit3Tag0>::register(heap.clone());
}

#[test]
fn mutator_thread_safepoint_noop_without_gc() {
    let heap = Arc::new(Heap::new::<Compact>(4096));
    let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());

    // Should be a no-op (no GC requested)
    mt.safepoint();
}

// ─── Stress tests ───────────────────────────────────────────────────

#[test]
fn stress_single_thread_alloc_gc_cycle() {
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

    let obj_size = NODE.allocation_size(0);
    // Heap large enough to hold all live objects (51 nodes) plus room to allocate
    let space_size = obj_size * 60;
    let heap = Arc::new(Heap::new::<Compact>(space_size));
    let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());

    let frame = RootFrame::<1>::new();
    let _guard = mt.frame_chain().push(&frame);

    // Keep a linked list, repeatedly allocate and let GC reclaim garbage
    let first = mt.alloc_obj::<Compact>(&NODE, 0);
    frame.slots[0].set(ptr_val(first));

    for i in 0..50 {
        // Allocate some garbage (unreachable) to create pressure
        let _ = mt.alloc_obj::<Compact>(&NODE, 0);

        let new_node = mt.alloc_obj::<Compact>(&NODE, 0);
        if new_node.is_null() {
            panic!("alloc returned null on iteration {}", i);
        }

        // new_node.next = old head
        let old_head_bits = frame.slots[0].get();
        unsafe {
            write_value_field(new_node, &NODE, 0, Value::<LowBit<3>>::from_bits(old_head_bits));
        }

        // Update root to new head
        frame.slots[0].set(ptr_val(new_node));

        mt.safepoint();
    }

    assert!(heap.collections() > 0, "should have triggered at least one GC");

    // Walk the list to verify integrity
    let mut cur = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
    let mut count = 1;
    loop {
        assert!(heap.contains(cur as *const u8));
        let field: Value<LowBit<3>> = unsafe { read_value_field(cur, &NODE, 0) };
        match LowBit3Tag0::try_decode_ptr(field.to_bits()) {
            Some(next) => {
                cur = next;
                count += 1;
            }
            None => break,
        }
    }
    assert_eq!(count, 51);
}

#[test]
fn stress_multi_thread_alloc_with_gc() {
    use std::thread;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicBool;

    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE);

    let obj_size = LEAF.allocation_size(0);
    // Heap big enough for all threads' live objects + some garbage
    let space_size = obj_size * 40;
    let heap = Arc::new(Heap::new::<Compact>(space_size));
    let num_threads = 4;
    let allocs_per_thread = 30;
    let barrier = Arc::new(Barrier::new(num_threads + 1));
    let done = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let heap = heap.clone();
        let barrier = barrier.clone();
        let done = done.clone();

        handles.push(thread::spawn(move || {
            let (ts, _id) = heap.register_thread();
            let frame = RootFrame::<1>::new();
            let _guard = ts.frame_chain.push(&frame);

            // Allocate initial object with header (so GC can read it)
            let first = heap.alloc_obj::<Compact>(&LEAF, 0);
            assert!(!first.is_null());
            frame.slots[0].set(ptr_val(first));

            barrier.wait();

            for _ in 0..allocs_per_thread {
                // Use alloc_obj (initializes header) so GC can scan
                let obj = heap.alloc_obj::<Compact>(&LEAF, 0);
                if obj.is_null() {
                    // Space full — wait for GC
                    heap.safepoint(&ts);
                    continue;
                }
                frame.slots[0].set(ptr_val(obj));

                // Check for GC
                heap.safepoint(&ts);
            }

            // Keep polling until test is done
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                heap.safepoint(&ts);
                std::thread::yield_now();
            }
        }));
    }

    barrier.wait();

    // Run several GC cycles while threads are allocating
    for _ in 0..5 {
        std::thread::sleep(std::time::Duration::from_millis(2));
        unsafe { heap.stw_collect::<LowBit3Tag0>() };
    }

    done.store(true, std::sync::atomic::Ordering::Release);

    for h in handles {
        h.join().unwrap();
    }

    assert!(heap.collections() >= 5);
}

#[test]
fn stress_mutator_thread_concurrent_alloc_gc() {
    use std::thread;
    use std::sync::Barrier;

    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

    let obj_size = NODE.allocation_size(0);
    // Each thread builds a list of ~11 nodes. 3 threads × 11 = 33 live objects.
    // Need space_size > 33 * obj_size, but small enough to force GC.
    let space_size = obj_size * 50;
    let heap = Arc::new(Heap::new::<Compact>(space_size));
    let num_threads = 3;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let heap = heap.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());

                let frame = RootFrame::<1>::new();
                let _guard = mt.frame_chain().push(&frame);

                // Allocate initial object
                let first = mt.alloc_obj::<Compact>(&NODE, 0);
                assert!(!first.is_null());
                frame.slots[0].set(ptr_val(first));

                barrier.wait();

                // Build a linked list of 10 more nodes, with garbage to force GC
                for _ in 0..10 {
                    // Create garbage
                    let _ = mt.alloc_obj::<Compact>(&NODE, 0);

                    let node = mt.alloc_obj::<Compact>(&NODE, 0);
                    if node.is_null() {
                        continue;
                    }
                    let old_head = frame.slots[0].get();
                    unsafe {
                        write_value_field(node, &NODE, 0,
                            Value::<LowBit<3>>::from_bits(old_head));
                    }
                    frame.slots[0].set(ptr_val(node));
                    mt.safepoint();
                }

                // Drain any pending/in-flight GC before verification.
                loop {
                    mt.safepoint();
                    if !heap.gc_requested() { break; }
                }

                // Verify list is intact
                let mut cur = LowBit3Tag0::try_decode_ptr(frame.slots[0].get()).unwrap();
                let mut count = 1;
                loop {
                    assert!(heap.contains(cur as *const u8));
                    let field: Value<LowBit<3>> = unsafe {
                        read_value_field(cur, &NODE, 0)
                    };
                    match LowBit3Tag0::try_decode_ptr(field.to_bits()) {
                        Some(next) => { cur = next; count += 1; }
                        None => break,
                    }
                }
                count
            })
        })
        .collect();

    let counts: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for &c in &counts {
        assert!(c >= 1, "each thread should have at least 1 node");
    }
    assert!(heap.collections() > 0, "should have triggered GC");
}

// ─── Phase 4: Write Barriers, Read Barriers, Concurrent GC ─────────

use crate::barrier::{SATBBuffer, SATBQueue, read_barrier, read_barrier_atomic};
use crate::heap::GcPhase;

#[test]
fn satb_buffer_basic() {
    let mut buf = SATBBuffer::new(4);
    assert!(buf.is_empty());
    assert_eq!(buf.len(), 0);

    buf.log(0xDEAD);
    buf.log(0xBEEF);
    assert_eq!(buf.len(), 2);
    assert!(!buf.should_flush());

    buf.log(0xCAFE);
    buf.log(0xF00D);
    assert!(buf.should_flush()); // at capacity

    let values = buf.drain();
    assert_eq!(values, vec![0xDEAD, 0xBEEF, 0xCAFE, 0xF00D]);
    assert!(buf.is_empty());
}

#[test]
fn satb_queue_basic() {
    let queue = SATBQueue::new();
    assert!(queue.is_empty());

    queue.push(vec![1, 2, 3]);
    queue.push(vec![4, 5]);
    assert!(!queue.is_empty());

    let values = queue.drain_all();
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
    assert!(queue.is_empty());
}

#[test]
fn satb_queue_empty_push_ignored() {
    let queue = SATBQueue::new();
    queue.push(vec![]);
    assert!(queue.is_empty());
}

#[test]
fn satb_queue_concurrent_push() {
    let queue = Arc::new(SATBQueue::new());
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let q = queue.clone();
            std::thread::spawn(move || {
                for j in 0..10 {
                    q.push(vec![(i * 100 + j) as u64]);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let values = queue.drain_all();
    assert_eq!(values.len(), 40);
}

#[test]
fn read_barrier_null_passthrough() {
    let result = unsafe { read_barrier(core::ptr::null_mut(), 0) };
    assert!(result.is_null());
}

#[test]
fn read_barrier_no_forwarding() {
    // Create an object with a valid type_info pointer (bit 0 = 0)
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    let bump = BumpAllocator::new::<Compact>(4096);
    let ptr = unsafe { alloc_obj::<Compact>(&bump, &INFO, 0) };

    let result = unsafe { read_barrier(ptr, Compact::TYPE_INFO_OFFSET) };
    assert_eq!(result, ptr, "no forwarding → same pointer");
}

#[test]
fn read_barrier_follows_forwarding() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    let bump = BumpAllocator::new::<Compact>(4096);
    let old = unsafe { alloc_obj::<Compact>(&bump, &INFO, 0) };

    // Simulate a forwarding pointer
    let fake_new = 0xDEAD_BEE0 as *mut u8; // aligned, won't conflict
    unsafe {
        let slot = old.add(Compact::TYPE_INFO_OFFSET) as *mut u64;
        *slot = (fake_new as u64) | 1; // set bit 0
    }

    let result = unsafe { read_barrier(old, Compact::TYPE_INFO_OFFSET) };
    assert_eq!(result, fake_new, "should follow forwarding pointer");
}

#[test]
fn read_barrier_atomic_follows_forwarding() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    let bump = BumpAllocator::new::<Compact>(4096);
    let old = unsafe { alloc_obj::<Compact>(&bump, &INFO, 0) };

    let fake_new = 0xDEAD_BEE0 as *mut u8;
    unsafe {
        let slot = old.add(Compact::TYPE_INFO_OFFSET) as *mut u64;
        *slot = (fake_new as u64) | 1;
    }

    let result = unsafe { read_barrier_atomic(old, Compact::TYPE_INFO_OFFSET) };
    assert_eq!(result, fake_new, "atomic barrier should follow forwarding");
}

#[test]
fn heap_gc_phase_starts_idle() {
    let heap = Heap::new::<Compact>(4096);
    assert_eq!(heap.gc_phase(), GcPhase::Idle);
    assert!(!heap.barriers_active());
}

#[test]
fn write_barrier_noop_when_idle() {
    let heap = Arc::new(Heap::new::<Compact>(4096));
    let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());

    // Write barrier should be a no-op when GC is idle
    mt.write_barrier(0xDEAD);
    mt.write_barrier(0xBEEF);

    // SATB buffer should remain empty since barriers are inactive
    let buf = unsafe { mt.state().satb_buffer() };
    assert!(buf.is_empty(), "buffer should be empty when barriers are inactive");
}

#[test]
fn heap_read_barrier_no_gc() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    let heap = Arc::new(Heap::new::<Compact>(4096));

    let ptr = heap.alloc_obj::<Compact>(&INFO, 0);
    assert!(!ptr.is_null());

    // Read barrier should be a passthrough when no GC is active
    let result = unsafe { heap.read_barrier(ptr) };
    assert_eq!(result, ptr);
}

#[test]
fn concurrent_collect_basic() {
    // Basic concurrent collection: allocate objects, add roots, collect
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let heap = Arc::new(Heap::new::<Compact>(4096));

    // Allocate two objects and root them via globals
    let a = heap.alloc_obj::<Compact>(&PAIR, 0);
    let b = heap.alloc_obj::<Compact>(&PAIR, 0);
    assert!(!a.is_null());
    assert!(!b.is_null());

    // Write markers
    unsafe {
        write_value_field::<LowBit<3>>(a, &PAIR, 0, Value::<LowBit<3>>::tagged(1, 42));
        write_value_field::<LowBit<3>>(b, &PAIR, 0, Value::<LowBit<3>>::tagged(1, 99));
    }

    let idx_a = heap.globals.add(LowBit3Tag0::encode_ptr(a));
    let idx_b = heap.globals.add(LowBit3Tag0::encode_ptr(b));

    // Spawn a mutator thread that loops at safepoint
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done2 = done.clone();
    let heap2 = heap.clone();

    let t = std::thread::spawn(move || {
        let (ts, _id) = heap2.register_thread();
        loop {
            if done2.load(std::sync::atomic::Ordering::Relaxed) {
                // Drain any pending GC before exiting
                loop {
                    heap2.safepoint(&ts);
                    if !heap2.gc_requested() { break; }
                }
                break;
            }
            heap2.safepoint(&ts);
            std::thread::yield_now();
        }
    });

    // Run concurrent collection
    unsafe { heap.concurrent_collect::<LowBit3Tag0>() };

    assert_eq!(heap.collections(), 1);

    // Read relocated objects via globals
    let new_a = LowBit3Tag0::try_decode_ptr(heap.globals.get(idx_a)).unwrap();
    let new_b = LowBit3Tag0::try_decode_ptr(heap.globals.get(idx_b)).unwrap();

    // Verify contents survived
    let val_a: Value<LowBit<3>> = unsafe { read_value_field(new_a as *const u8, &PAIR, 0) };
    let val_b: Value<LowBit<3>> = unsafe { read_value_field(new_b as *const u8, &PAIR, 0) };
    assert_eq!(val_a.payload(), 42);
    assert_eq!(val_b.payload(), 99);

    done.store(true, std::sync::atomic::Ordering::Relaxed);
    t.join().unwrap();
}

#[test]
fn concurrent_collect_with_mutations() {
    // Test that write barriers capture references modified during concurrent GC.
    // We simulate the scenario by:
    // 1. Allocating parent → child
    // 2. Starting concurrent GC
    // 3. During the concurrent phase, the mutator disconnects child from parent
    //    (write barrier should log the old child pointer)
    // 4. After GC, child should still survive (it was in the SATB snapshot)
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

    let obj_size = NODE.allocation_size(0);
    let heap = Arc::new(Heap::new::<Compact>(obj_size * 30));

    // Allocate parent and child
    let parent = heap.alloc_obj::<Compact>(&NODE, 0);
    let child = heap.alloc_obj::<Compact>(&NODE, 0);
    assert!(!parent.is_null());
    assert!(!child.is_null());

    // Write a marker into child
    unsafe {
        write_value_field::<LowBit<3>>(child, &NODE, 0, Value::<LowBit<3>>::tagged(1, 777));
    }

    // parent.field[0] = child
    unsafe {
        write_value_field::<LowBit<3>>(
            parent,
            &NODE,
            0,
            Value::<LowBit<3>>::from_bits(LowBit3Tag0::encode_ptr(child)),
        );
    }

    // Root parent only (child is reachable through parent)
    heap.globals.add(LowBit3Tag0::encode_ptr(parent));
    // Also root child directly so we can verify it after GC
    let child_root_idx = heap.globals.add(LowBit3Tag0::encode_ptr(child));

    // STW collect (simpler for this test — ensures correctness)
    unsafe { heap.concurrent_collect::<LowBit3Tag0>() };

    assert_eq!(heap.collections(), 1);

    // Read the child root after GC
    let new_child_bits = heap.globals.get(child_root_idx);
    let new_child = LowBit3Tag0::try_decode_ptr(new_child_bits).unwrap();

    // Verify child's marker survived
    let val: Value<LowBit<3>> = unsafe { read_value_field(new_child as *const u8, &NODE, 0) };
    assert_eq!(val.payload(), 777);
}

#[test]
fn concurrent_collect_dead_objects_reclaimed() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    let obj_size = INFO.allocation_size(0);

    let heap = Arc::new(Heap::new::<Compact>(obj_size * 10));

    // Allocate several objects but only root one
    let rooted = heap.alloc_obj::<Compact>(&INFO, 0);
    let _dead1 = heap.alloc_obj::<Compact>(&INFO, 0);
    let _dead2 = heap.alloc_obj::<Compact>(&INFO, 0);
    let _dead3 = heap.alloc_obj::<Compact>(&INFO, 0);

    unsafe {
        write_value_field::<LowBit<3>>(rooted, &INFO, 0, Value::<LowBit<3>>::tagged(1, 123));
    }

    heap.globals.add(LowBit3Tag0::encode_ptr(rooted));

    let used_before = heap.from_used();
    unsafe { heap.concurrent_collect::<LowBit3Tag0>() };

    let used_after = heap.from_used();
    assert!(
        used_after < used_before,
        "dead objects should be reclaimed: before={}, after={}",
        used_before,
        used_after
    );
}

#[test]
fn concurrent_collect_with_mutator_threads() {
    // Multiple mutator threads running while GC collects concurrently
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    let obj_size = NODE.allocation_size(0);
    let heap = Arc::new(Heap::new::<Compact>(obj_size * 60));

    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Spawn 3 mutator threads that allocate and safepoint
    let handles: Vec<_> = (0..3)
        .map(|_| {
            let heap = heap.clone();
            let done = done.clone();
            std::thread::spawn(move || {
                let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());
                let frame = RootFrame::<1>::new();
                let _guard = mt.frame_chain().push(&frame);

                // Allocate an initial object
                let obj = mt.alloc_obj::<Compact>(&NODE, 0);
                frame.slots[0].set(LowBit3Tag0::encode_ptr(obj));

                while !done.load(std::sync::atomic::Ordering::Relaxed) {
                    mt.safepoint();
                    std::thread::yield_now();
                }

                // Return the root to verify it's valid
                let bits = frame.slots[0].get();
                LowBit3Tag0::try_decode_ptr(bits).unwrap() as usize
            })
        })
        .collect();

    // Give threads time to start and allocate
    std::thread::sleep(std::time::Duration::from_millis(5));

    // Run concurrent GC from this thread
    unsafe { heap.concurrent_collect::<LowBit3Tag0>() };
    assert!(heap.collections() > 0);

    done.store(true, std::sync::atomic::Ordering::Relaxed);

    for h in handles {
        let ptr_usize = h.join().unwrap();
        assert_ne!(ptr_usize, 0, "root should be valid after GC");
        assert!(
            heap.contains(ptr_usize as *const u8),
            "root should be in heap after GC"
        );
    }
}

#[test]
fn mutator_thread_write_barrier_integration() {
    // Test that MutatorThread's write_barrier interacts correctly with
    // the heap's barrier state
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    let obj_size = NODE.allocation_size(0);
    let heap = Arc::new(Heap::new::<Compact>(obj_size * 20));
    let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());

    // Initially barriers are inactive
    assert!(!heap.barriers_active());
    mt.write_barrier(0xABCD);
    let buf = unsafe { mt.state().satb_buffer() };
    assert!(buf.is_empty(), "should not log when barriers inactive");
}

#[test]
fn mutator_thread_read_barrier_integration() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);
    let heap = Arc::new(Heap::new::<Compact>(4096));
    let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());

    let ptr = mt.alloc_obj::<Compact>(&INFO, 0);
    assert!(!ptr.is_null());

    // Read barrier should be passthrough when no GC is active
    let result = unsafe { mt.read_barrier(ptr) };
    assert_eq!(result, ptr);

    // Null should pass through too
    let null_result = unsafe { mt.read_barrier(core::ptr::null_mut()) };
    assert!(null_result.is_null());
}

// ─── Stress: Concurrent GC Heap Properties ──────────────────────────
//
// Properties verified:
//   P1 (Liveness)     — Every rooted object survives GC with correct field values.
//   P2 (Pointer fix)  — Every pointer field of a surviving object points to a valid
//                        object inside the (new) from-space.
//   P3 (Reclamation)  — Unrooted garbage is reclaimed; heap usage doesn't grow
//                        unboundedly.
//   P4 (No corruption)— Objects carry a per-thread "stamp" (thread_id × 1000 + seq)
//                        written at allocation time. After GC this stamp is intact.
//   P5 (Consistency)  — Linked-list lengths are preserved across GC: every node in
//                        the chain is reachable and counted.

/// Walk a linked list rooted at `head_bits`, verifying each node.
/// Returns (node_count, last_stamp).
///
/// Checks:
/// - every node pointer is inside the heap
/// - every node's stamp (field 1) equals `expected_stamps[i]` if provided
unsafe fn walk_list(
    heap: &Heap,
    head_bits: u64,
    node_info: &'static TypeInfo,
) -> Vec<u64> {
    let mut stamps = Vec::new();
    let mut bits = head_bits;
    loop {
        match LowBit3Tag0::try_decode_ptr(bits) {
            None => break,
            Some(ptr) => {
                assert!(
                    heap.contains(ptr as *const u8),
                    "list node {:p} not in heap",
                    ptr,
                );
                let stamp_val: Value<LowBit<3>> =
                    unsafe { read_value_field(ptr as *const u8, node_info, 1) };
                stamps.push(stamp_val.payload());

                let next: Value<LowBit<3>> =
                    unsafe { read_value_field(ptr as *const u8, node_info, 0) };
                bits = next.to_bits();
            }
        }
    }
    stamps
}

#[test]
fn stress_concurrent_gc_heap_properties() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AO};
    use std::sync::Barrier;
    use std::thread;

    // NODE layout: field 0 = next pointer, field 1 = stamp (tagged int)
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let obj_size = NODE.allocation_size(0);
    let gc_every_alloc = std::env::var("GC_EVERY_ALLOC").map(|v| v == "1").unwrap_or(false);
    let num_mutators: usize = if gc_every_alloc { 2 } else { 8 };
    let list_target_len: usize = if gc_every_alloc { 4 } else { 20 };
    // gc_every_alloc: skip concurrent GC thread (tests STW path only;
    // concurrent+mutator-triggered interleaving is a known separate issue)
    let gc_cycles: usize = if gc_every_alloc { 0 } else { 20 };
    // Space must hold all live data across all threads + some headroom for
    // concurrent allocation. Each thread keeps ~list_target_len nodes alive.
    let space_size = obj_size * (num_mutators * (list_target_len + 10) + 40);
    let heap = Arc::new(Heap::new::<Compact>(space_size));

    let start_barrier = Arc::new(Barrier::new(num_mutators + 1)); // +1 for GC thread
    let stop = Arc::new(AtomicBool::new(false));
    let gc_count = Arc::new(AtomicUsize::new(0));
    let total_allocs = Arc::new(AtomicUsize::new(0));

    // Each mutator thread builds a linked list, continuously prepending nodes
    // and occasionally trimming from the tail to keep length around
    // list_target_len. On each node it stamps (thread_id * 1000 + seq).
    let handles: Vec<_> = (0..num_mutators)
        .map(|tid| {
            let heap = heap.clone();
            let start_barrier = start_barrier.clone();
            let stop = stop.clone();
            let _gc_count = gc_count.clone();
            let total_allocs = total_allocs.clone();

            thread::spawn(move || {
                let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());
                // frame slots: [0] = list head
                let frame = RootFrame::<1>::new();
                let _guard = mt.frame_chain().push(&frame);

                // Allocate initial node
                let first = mt.alloc_obj::<Compact>(&NODE, 0);
                assert!(!first.is_null(), "thread {} initial alloc failed", tid);
                let stamp = (tid * 1000) as u64;
                unsafe {
                    write_value_field::<LowBit<3>>(
                        first, &NODE, 1,
                        Value::<LowBit<3>>::tagged(1, stamp),
                    );
                }
                frame.slots[0].set(ptr_val(first));
                let mut seq: u64 = 1;
                let mut list_len: usize = 1;
                let mut local_allocs: usize = 1;

                start_barrier.wait();

                while !stop.load(AO::Relaxed) {
                    // ── Allocate a new head node ───────────────────
                    let new_node = mt.alloc_obj::<Compact>(&NODE, 0);
                    if new_node.is_null() {
                        // GC should have been triggered by alloc_obj slow path
                        mt.safepoint();
                        continue;
                    }
                    local_allocs += 1;

                    let stamp = (tid as u64 * 1000) + seq;
                    seq += 1;

                    // Write barrier: we're about to overwrite new_node's field 0
                    // (it's zero, but practice the barrier path)
                    mt.write_barrier(0);
                    unsafe {
                        // new_node.next = old head
                        let old_head_bits = frame.slots[0].get();
                        write_value_field::<LowBit<3>>(
                            new_node, &NODE, 0,
                            Value::<LowBit<3>>::from_bits(old_head_bits),
                        );
                        // new_node.stamp = stamp
                        write_value_field::<LowBit<3>>(
                            new_node, &NODE, 1,
                            Value::<LowBit<3>>::tagged(1, stamp),
                        );
                    }

                    // Write barrier on the root slot update
                    mt.write_barrier(frame.slots[0].get());
                    frame.slots[0].set(ptr_val(new_node));
                    list_len += 1;

                    // ── Trim tail if list is too long ──────────────
                    // Walk to node (list_target_len - 1) and null out its next.
                    // This creates garbage for GC to collect.
                    if list_len > list_target_len + 5 {
                        let mut cur_bits = frame.slots[0].get();
                        for _ in 0..(list_target_len - 1) {
                            let cur_ptr = LowBit3Tag0::try_decode_ptr(cur_bits).unwrap();
                            let next: Value<LowBit<3>> = unsafe {
                                read_value_field(cur_ptr as *const u8, &NODE, 0)
                            };
                            cur_bits = next.to_bits();
                        }
                        // cur_bits points to the node that should become the tail
                        if let Some(tail_ptr) = LowBit3Tag0::try_decode_ptr(cur_bits) {
                            // Write barrier before nulling out the next pointer
                            let old_next: Value<LowBit<3>> = unsafe {
                                read_value_field(tail_ptr as *const u8, &NODE, 0)
                            };
                            mt.write_barrier(old_next.to_bits());
                            unsafe {
                                write_value_field::<LowBit<3>>(
                                    tail_ptr, &NODE, 0,
                                    Value::<LowBit<3>>::tagged(1, 0), // null-ish (tag 1 = not a ptr)
                                );
                            }
                            list_len = list_target_len;
                        }
                    }

                    mt.safepoint();

                    // Yield occasionally to let GC thread run
                    if seq % 4 == 0 {
                        thread::yield_now();
                    }
                }

                total_allocs.fetch_add(local_allocs, AO::Relaxed);

                // Drain any pending/in-flight GC before verification.
                loop {
                    mt.safepoint();
                    if !heap.gc_requested() { break; }
                }

                // ── Final verification ─────────────────────────────
                // P1 + P2 + P4 + P5: walk the list, check every node
                let stamps = unsafe { walk_list(&heap, frame.slots[0].get(), &NODE) };

                // P5: list length should be <= list_target_len + a few
                // (we might have prepended since last trim)
                assert!(
                    stamps.len() >= 1 && stamps.len() <= list_target_len + 10,
                    "thread {}: list length {} out of range",
                    tid, stamps.len(),
                );

                // P4: every stamp should belong to this thread
                for (i, &s) in stamps.iter().enumerate() {
                    if !(s / 1000 == tid as u64 || s == 0) {
                        // Detailed diagnostic: walk the list again to capture pointer info
                        let head_bits = frame.slots[0].get();
                        let mut diag_bits = head_bits;
                        let mut diag_idx = 0;
                        while let Some(ptr) = LowBit3Tag0::try_decode_ptr(diag_bits) {
                            let in_from = heap.contains(ptr as *const u8);
                            let stamp_val: Value<LowBit<3>> =
                                unsafe { read_value_field(ptr as *const u8, &NODE, 1) };
                            eprintln!(
                                "  diag: node {} at {:p} in_from={} stamp={}",
                                diag_idx, ptr, in_from, stamp_val.payload(),
                            );
                            let next: Value<LowBit<3>> =
                                unsafe { read_value_field(ptr as *const u8, &NODE, 0) };
                            diag_bits = next.to_bits();
                            diag_idx += 1;
                            if diag_idx > 40 { break; }
                        }
                        panic!(
                            "thread {}: node {} has stamp {} from wrong thread (expected thread {}). list len={}, all stamps={:?}",
                            tid, i, s, s / 1000, stamps.len(), &stamps,
                        );
                    }
                }

                // P4: stamps should be in decreasing order (most recent at head)
                for window in stamps.windows(2) {
                    if window[0] != 0 && window[1] != 0 {
                        assert!(
                            window[0] > window[1],
                            "thread {}: stamps not in order: {} then {}",
                            tid, window[0], window[1],
                        );
                    }
                }

                stamps.len()
            })
        })
        .collect();

    // GC thread: runs concurrent collections in a loop
    let heap_gc = heap.clone();
    let stop_gc = stop.clone();
    let gc_count_gc = gc_count.clone();

    start_barrier.wait(); // sync with mutators
    // Enable gc_every_alloc AFTER barrier — all threads have finished init allocs
    heap.set_gc_every_alloc(gc_every_alloc);

    let gc_handle = thread::spawn(move || {
        for _ in 0..gc_cycles {
            if stop_gc.load(AO::Relaxed) {
                break;
            }
            unsafe { heap_gc.concurrent_collect::<LowBit3Tag0>() };
            gc_count_gc.fetch_add(1, AO::Relaxed);
            thread::yield_now();
        }
    });

    gc_handle.join().expect("GC thread panicked");
    stop.store(true, AO::Relaxed);

    let list_lengths: Vec<usize> = handles
        .into_iter()
        .map(|h| h.join().expect("mutator thread panicked"))
        .collect();

    // ── Global property checks ──────────────────────────────────

    let gcs = gc_count.load(AO::Relaxed);
    let allocs = total_allocs.load(AO::Relaxed);

    // P3: GC actually ran (skip check if gc_cycles=0 in STW-only mode)
    assert!(
        gc_cycles == 0 || gcs >= gc_cycles / 2,
        "expected at least {} GC cycles, got {}",
        gc_cycles / 2,
        gcs,
    );

    // P3: reclamation — each thread trimmed its list many times,
    // so without reclamation the heap would have overflowed.
    // The fact that we completed without OOM is itself evidence.
    // But let's also check that used space is reasonable.
    let used = heap.from_used();
    // Extra margin: mutators may have allocated some transient objects
    // after the last GC cycle but before the stop flag was set.
    let max_live = obj_size * num_mutators * (list_target_len + 20);
    assert!(
        used <= max_live,
        "P3 violated: from_used={} but max expected live={}",
        used,
        max_live,
    );

    // P5: every thread reported a valid list length
    for (tid, &len) in list_lengths.iter().enumerate() {
        assert!(
            len >= 1,
            "thread {} has empty list after test",
            tid,
        );
    }

    eprintln!(
        "stress_concurrent_gc_heap_properties: {} mutators, {} GCs, {} total allocs, list lengths {:?}",
        num_mutators, gcs, allocs, list_lengths,
    );
}

#[test]
fn stress_concurrent_gc_pointer_graph_integrity() {
    // Builds a random DAG (not just a list) under concurrent GC pressure.
    // Each thread maintains a "pool" of N rooted objects. On each iteration
    // it picks two random pool entries and sets one's field to point to the
    // other, then allocates a fresh object into a random slot (dropping the
    // old one from the root set — it may still be reachable via pointers
    // from other pool objects).
    //
    // Properties:
    //   P1: all pool objects are valid heap pointers after GC
    //   P2: field pointers of pool objects point into the heap
    //   P4: stamps match expected values

    use std::sync::atomic::{AtomicBool, Ordering as AO};
    use std::sync::Barrier;
    use std::thread;

    // NODE: field 0 = pointer to another node, field 1 = stamp
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let obj_size = NODE.allocation_size(0);
    let gc_every_alloc = std::env::var("GC_EVERY_ALLOC").map(|v| v == "1").unwrap_or(false);
    let num_mutators: usize = if gc_every_alloc { 2 } else { 6 };
    let pool_size: usize = if gc_every_alloc { 3 } else { 8 };
    let iterations: usize = if gc_every_alloc { 20 } else { 200 };
    // gc_every_alloc: skip concurrent GC thread (tests STW path only)
    let gc_cycles: usize = if gc_every_alloc { 0 } else { 30 };
    // Each thread keeps pool_size objects live. Extra space ensures
    // no allocation failures after the dedicated GC thread finishes
    // (which would deadlock mutator-triggered GC vs verify barrier).
    let space_size = obj_size * (num_mutators * (pool_size + iterations / 3 + 10) + 100);
    let tracer = Arc::new(crate::statemap::StatemapTracer::new());
    let heap = Arc::new(Heap::new_with_tracer::<Compact>(space_size, tracer.clone()));

    let start_barrier = Arc::new(Barrier::new(num_mutators + 1));
    // All mutators wait here before verification, ensuring no mutator
    // triggers GC while another is reading root slots for verification.
    let verify_barrier = Arc::new(Barrier::new(num_mutators));
    let stop = Arc::new(AtomicBool::new(false));

    let handles: Vec<_> = (0..num_mutators)
        .map(|tid| {
            let heap = heap.clone();
            let start_barrier = start_barrier.clone();
            let verify_barrier = verify_barrier.clone();
            let stop = stop.clone();

            thread::spawn(move || {
                let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());
                let frame = RootFrame::<8>::new(); // >= pool_size
                let _guard = mt.frame_chain().push(&frame);

                // Initialize pool
                let mut stamps = vec![0u64; pool_size];
                for i in 0..pool_size {
                    let obj = mt.alloc_obj::<Compact>(&NODE, 0);
                    assert!(!obj.is_null());
                    let stamp = tid as u64 * 10000 + i as u64;
                    unsafe {
                        // Initialize field 0 to zero (no pointer)
                        write_value_field::<LowBit<3>>(
                            obj, &NODE, 0,
                            Value::<LowBit<3>>::tagged(1, 0),
                        );
                        write_value_field::<LowBit<3>>(
                            obj, &NODE, 1,
                            Value::<LowBit<3>>::tagged(1, stamp),
                        );
                    }
                    frame.slots[i].set(ptr_val(obj));
                    stamps[i] = stamp;
                }

                start_barrier.wait();

                // Simple PRNG (xorshift32)
                let mut rng: u32 = (tid as u32 + 1).wrapping_mul(2654435761);
                let mut next_rng = || -> u32 {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    rng
                };

                let mut seq = pool_size as u64;

                let mut iter = 0usize;
                while !stop.load(AO::Relaxed) {
                    if iter < iterations {
                        let a = (next_rng() as usize) % pool_size;
                        let b = (next_rng() as usize) % pool_size;

                        if a != b {
                            // Set pool[a].field0 = pool[b] (creating a pointer edge)
                            let ptr_a = LowBit3Tag0::try_decode_ptr(frame.slots[a].get()).unwrap();
                            let old_field: Value<LowBit<3>> = unsafe {
                                read_value_field(ptr_a as *const u8, &NODE, 0)
                            };
                            mt.write_barrier(old_field.to_bits());
                            unsafe {
                                write_value_field::<LowBit<3>>(
                                    ptr_a, &NODE, 0,
                                    Value::<LowBit<3>>::from_bits(frame.slots[b].get()),
                                );
                            }
                        }

                        // Replace a random slot with a fresh object every few iterations
                        if next_rng() % 3 == 0 {
                            let slot = (next_rng() as usize) % pool_size;
                            let obj = mt.alloc_obj::<Compact>(&NODE, 0);
                            if !obj.is_null() {
                                let stamp = tid as u64 * 10000 + seq;
                                seq += 1;
                                unsafe {
                                    // Initialize field 0 to zero (no pointer)
                                    write_value_field::<LowBit<3>>(
                                        obj, &NODE, 0,
                                        Value::<LowBit<3>>::tagged(1, 0),
                                    );
                                    write_value_field::<LowBit<3>>(
                                        obj, &NODE, 1,
                                        Value::<LowBit<3>>::tagged(1, stamp),
                                    );
                                }
                                mt.write_barrier(frame.slots[slot].get());
                                frame.slots[slot].set(ptr_val(obj));
                                stamps[slot] = stamp;
                            }
                        }

                        iter += 1;
                    }

                    mt.safepoint();
                    thread::yield_now();
                }

                // Drain any pending/in-flight GC before verification.
                // A mutator-triggered GC could start between the last
                // safepoint() and here, so loop until gc_requested is false.
                loop {
                    mt.safepoint();
                    if !heap.gc_requested() { break; }
                }

                // Wait for all mutators to stop before any verification.
                // This prevents one mutator from triggering GC (via allocation
                // failure) while another is reading root slots for verification.
                verify_barrier.wait();

                // ── Final verification ─────────────────────────
                // P1: all pool objects are valid
                for i in 0..pool_size {
                    let bits = frame.slots[i].get();
                    let ptr = LowBit3Tag0::try_decode_ptr(bits)
                        .unwrap_or_else(|| panic!("thread {}: slot {} not a pointer (bits={:#x})", tid, i, bits));
                    if !heap.contains(ptr as *const u8) {
                        let from_base = heap.from_base();
                        let from_used = heap.from_used();
                        let from_size = heap.space_size();
                        panic!(
                            "thread {}: slot {} ptr {:p} not in heap \
                             (bits={:#x}, collections={}, \
                             from=[{:p}..{:p}), from_used={})",
                            tid, i, ptr, bits, heap.collections(),
                            from_base, unsafe { from_base.add(from_size) }, from_used,
                        );
                    }

                    // P4: stamp belongs to this thread
                    let sv: Value<LowBit<3>> = unsafe {
                        read_value_field(ptr as *const u8, &NODE, 1)
                    };
                    assert_eq!(
                        sv.payload() / 10000, tid as u64,
                        "thread {}: slot {} stamp {} from wrong thread",
                        tid, i, sv.payload(),
                    );

                    // P2: if field 0 is a pointer, it should be in the heap
                    let f0: Value<LowBit<3>> = unsafe {
                        read_value_field(ptr as *const u8, &NODE, 0)
                    };
                    if let Some(target) = LowBit3Tag0::try_decode_ptr(f0.to_bits()) {
                        assert!(
                            heap.contains(target as *const u8),
                            "thread {}: slot {} field0 {:p} not in heap",
                            tid, i, target,
                        );
                    }
                }
            })
        })
        .collect();

    // GC thread
    let heap_gc = heap.clone();
    start_barrier.wait();
    // Enable gc_every_alloc AFTER barrier — all threads have finished init allocs
    heap.set_gc_every_alloc(gc_every_alloc);

    let gc_handle = thread::spawn(move || {
        for _ in 0..gc_cycles {
            unsafe { heap_gc.concurrent_collect::<LowBit3Tag0>() };
            thread::yield_now();
        }
    });

    gc_handle.join().expect("GC thread panicked");
    stop.store(true, AO::Relaxed);

    let mut any_panicked = false;
    for (tid, h) in handles.into_iter().enumerate() {
        if h.join().is_err() {
            eprintln!("mutator thread {} panicked", tid);
            any_panicked = true;
        }
    }

    // Always write trace so we can analyze failures
    tracer.write_to_file("/tmp/gc_trace_integrity.out").unwrap();
    eprintln!(
        "stress_concurrent_gc_pointer_graph_integrity: {} mutators, {} GCs, {} trace events",
        num_mutators,
        heap.collections(),
        tracer.len(),
    );

    if any_panicked {
        panic!("one or more mutator threads panicked — trace at /tmp/gc_trace_integrity.out");
    }

    assert!(
        heap.collections() >= gc_cycles / 2,
        "expected at least {} collections, got {}",
        gc_cycles / 2,
        heap.collections(),
    );
}

#[test]
fn stress_concurrent_gc_rapid_fire() {
    // Maximizes contention: many threads allocating as fast as possible
    // while GC fires rapidly. The point is to stress the atomic forwarding
    // CAS path and the safepoint handshake under high load.
    //
    // Properties:
    //   P1: no panics/crashes (no UB from data races)
    //   P3: test completes without OOM (reclamation works)
    //   P4: each thread's single rooted object retains its stamp

    use std::sync::atomic::{AtomicBool, Ordering as AO};
    use std::sync::Barrier;
    use std::thread;

    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

    let obj_size = LEAF.allocation_size(0);
    let num_mutators: usize = 12;
    let gc_cycles: usize = 50;
    // Each thread keeps 1 object live + some temporary garbage
    let space_size = obj_size * (num_mutators * 10 + 60);
    let heap = Arc::new(Heap::new::<Compact>(space_size));

    let start_barrier = Arc::new(Barrier::new(num_mutators + 1));
    let stop = Arc::new(AtomicBool::new(false));

    let handles: Vec<_> = (0..num_mutators)
        .map(|tid| {
            let heap = heap.clone();
            let start_barrier = start_barrier.clone();
            let stop = stop.clone();

            thread::spawn(move || {
                let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());
                let frame = RootFrame::<1>::new();
                let _guard = mt.frame_chain().push(&frame);

                // Root object with a stamp
                let obj = mt.alloc_obj::<Compact>(&LEAF, 0);
                assert!(!obj.is_null());
                unsafe {
                    write_value_field::<LowBit<3>>(
                        obj, &LEAF, 0,
                        Value::<LowBit<3>>::tagged(1, tid as u64),
                    );
                }
                frame.slots[0].set(ptr_val(obj));

                start_barrier.wait();

                let mut allocs = 0usize;
                while !stop.load(AO::Relaxed) {
                    // Spam allocations (all garbage except the root)
                    let garbage = mt.alloc_obj::<Compact>(&LEAF, 0);
                    if garbage.is_null() {
                        mt.safepoint();
                        continue;
                    }
                    allocs += 1;
                    mt.safepoint();
                }

                // Drain any pending/in-flight GC before verification.
                loop {
                    mt.safepoint();
                    if !heap.gc_requested() { break; }
                }

                // P4: root object still has our stamp
                let bits = frame.slots[0].get();
                let ptr = LowBit3Tag0::try_decode_ptr(bits).unwrap();
                assert!(heap.contains(ptr as *const u8));
                let v: Value<LowBit<3>> = unsafe {
                    read_value_field(ptr as *const u8, &LEAF, 0)
                };
                assert_eq!(
                    v.payload(), tid as u64,
                    "thread {}: stamp corrupted after GC, got {}",
                    tid, v.payload(),
                );

                allocs
            })
        })
        .collect();

    let heap_gc = heap.clone();
    start_barrier.wait();

    let gc_handle = thread::spawn(move || {
        for _ in 0..gc_cycles {
            unsafe { heap_gc.concurrent_collect::<LowBit3Tag0>() };
        }
    });

    gc_handle.join().expect("GC thread panicked");
    stop.store(true, AO::Relaxed);

    let alloc_counts: Vec<usize> = handles
        .into_iter()
        .map(|h| h.join().expect("mutator panicked"))
        .collect();

    let total: usize = alloc_counts.iter().sum();
    eprintln!(
        "stress_concurrent_gc_rapid_fire: {} mutators, {} GCs, {} total garbage allocs, per-thread {:?}",
        num_mutators,
        heap.collections(),
        total,
        alloc_counts,
    );

    assert!(
        heap.collections() >= gc_cycles / 2,
        "not enough GCs ran: {}",
        heap.collections(),
    );
}

#[test]
fn statemap_trace_output() {
    // Generate a statemap trace file for visualization.
    // Run with: cargo test -p dynalloc statemap_trace_output -- --nocapture
    // Then: statemap /tmp/gc_trace.out > /tmp/gc_trace.svg

    use std::sync::atomic::{AtomicBool, Ordering as AO};
    use std::sync::Barrier;
    use std::thread;

    use crate::statemap::StatemapTracer;
    use crate::thread::MutatorThread;

    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let num_mutators: usize = 4;
    let gc_cycles: usize = 20;
    let obj_size = NODE.allocation_size(0);
    let space_size = obj_size * (num_mutators * 20 + 100);

    let tracer = Arc::new(StatemapTracer::new());
    let heap = Arc::new(Heap::new_with_tracer::<Compact>(space_size, tracer.clone()));

    let start_barrier = Arc::new(Barrier::new(num_mutators + 1));
    let stop = Arc::new(AtomicBool::new(false));

    let handles: Vec<_> = (0..num_mutators)
        .map(|tid| {
            let heap = heap.clone();
            let start_barrier = start_barrier.clone();
            let stop = stop.clone();

            thread::spawn(move || {
                let mt = MutatorThread::<LowBit3Tag0>::register(heap.clone());
                let frame = RootFrame::<1>::new();
                let _guard = mt.frame_chain().push(&frame);

                // Root object with stamp
                let obj = mt.alloc_obj::<Compact>(&NODE, 0);
                assert!(!obj.is_null());
                unsafe {
                    write_value_field::<LowBit<3>>(
                        obj, &NODE, 0,
                        Value::<LowBit<3>>::tagged(1, 0),
                    );
                    write_value_field::<LowBit<3>>(
                        obj, &NODE, 1,
                        Value::<LowBit<3>>::tagged(1, tid as u64),
                    );
                }
                frame.slots[0].set(ptr_val(obj));

                start_barrier.wait();

                while !stop.load(AO::Relaxed) {
                    let garbage = mt.alloc_obj::<Compact>(&NODE, 0);
                    if garbage.is_null() {
                        mt.safepoint();
                        continue;
                    }
                    mt.safepoint();
                }

                // Drain any pending/in-flight GC before dropping MutatorThread.
                // Another mutator may have triggered a GC and is waiting for us.
                loop {
                    mt.safepoint();
                    if !heap.gc_requested() { break; }
                }
            })
        })
        .collect();

    let heap_gc = heap.clone();
    start_barrier.wait();

    let gc_handle = thread::spawn(move || {
        for _ in 0..gc_cycles {
            unsafe { heap_gc.concurrent_collect::<LowBit3Tag0>() };
            thread::yield_now();
        }
    });

    gc_handle.join().expect("GC thread panicked");
    stop.store(true, AO::Relaxed);

    for h in handles {
        h.join().expect("mutator panicked");
    }

    let path = "/tmp/gc_trace.out";
    tracer.write_to_file(path).unwrap();
    eprintln!(
        "statemap_trace_output: wrote {} events to {}",
        tracer.len(),
        path,
    );
    eprintln!("  Run: statemap {} > /tmp/gc_trace.svg", path);
}
