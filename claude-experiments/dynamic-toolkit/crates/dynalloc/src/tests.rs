use dynvalue::{LowBit, Value};
use dynobj::*;

use crate::{Alloc, HeapWalker, BumpAllocator, alloc_obj, Mutator, Root, RootScope};

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

    let obj_b = unsafe { alloc_obj::<Compact>(&bump, &INFO_0F, 0) };
    let obj_a = unsafe { alloc_obj::<Compact>(&bump, &INFO_1F, 0) };

    // obj_a.field[0] → obj_b
    unsafe {
        write_value_field(obj_a, &INFO_1F, 0, Value::<S>::tagged(1, obj_b as u64 >> 3));
    }

    // Set up roots
    let mut ss = ShadowStack::new(32);
    ss.push_frame(1);
    ss.set(0, Value::<S>::tagged(1, obj_a as u64 >> 3).to_bits());

    // Mark phase using read_type_info + scan_object
    let mut mark_stack: Vec<*mut u8> = Vec::new();
    let mut marked: Vec<*mut u8> = Vec::new();

    ss.scan_roots(&mut |slot| {
        let bits = unsafe { *slot };
        let v = Value::<S>::from_bits(bits);
        if v.has_tag(1) {
            let ptr = (v.payload() << 3) as *mut u8;
            if !marked.contains(&ptr) {
                marked.push(ptr);
                mark_stack.push(ptr);
            }
        }
    });

    assert_eq!(marked.len(), 1);
    assert!(marked.contains(&obj_a));

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

    assert_eq!(marked.len(), 2);
    assert!(marked.contains(&obj_a));
    assert!(marked.contains(&obj_b));

    // Verify both objects are in the bump allocator
    assert!(bump.contains(obj_a));
    assert!(bump.contains(obj_b));

    // Walk heap and verify same objects found
    let mut walked: Vec<*mut u8> = Vec::new();
    unsafe {
        bump.walk(&mut |ptr, _info| {
            walked.push(ptr);
        });
    }
    assert_eq!(walked.len(), 2);
    assert!(walked.contains(&obj_a));
    assert!(walked.contains(&obj_b));

    ss.pop_frame();
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
    assert_eq!(m.roots().len(), 2);
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
    m.roots().scan_roots(&mut |slot| {
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

        assert_eq!((*mp).roots().len(), 1);
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
