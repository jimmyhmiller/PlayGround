use super::*;
use std::cell::Cell;

// ────────────────────────────────────────────────────────────────────
// TypeInfo offset calculations
// ────────────────────────────────────────────────────────────────────

#[test]
fn compact_cons_cell_layout() {
    const CONS: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    assert_eq!(CONS.header_size, 8);
    assert_eq!(CONS.value_field_count, 2);
    assert_eq!(CONS.value_field_offset(0), 8);
    assert_eq!(CONS.value_field_offset(1), 16);
    assert_eq!(CONS.allocation_size(0), 24);
}

#[test]
fn full_cons_cell_layout() {
    const CONS: TypeInfo = TypeInfo::for_header(Full::SIZE).with_fields(2);

    assert_eq!(CONS.header_size, 16);
    assert_eq!(CONS.value_field_offset(0), 16);
    assert_eq!(CONS.value_field_offset(1), 24);
    assert_eq!(CONS.allocation_size(0), 32);
}

#[test]
fn string_layout_compact() {
    const STR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_bytes(0);

    assert_eq!(STR.varlen_count_offset(), 8);
    assert_eq!(STR.varlen_element_offset(0), 16);
    assert_eq!(STR.varlen_element_offset(4), 20);

    assert_eq!(STR.allocation_size(0), 16);
    assert_eq!(STR.allocation_size(5), 24);
    assert_eq!(STR.allocation_size(8), 24);
}

#[test]
fn vector_layout_full() {
    const VEC: TypeInfo = TypeInfo::for_header(Full::SIZE).with_varlen_values(0);

    assert_eq!(VEC.varlen_count_offset(), 16);
    assert_eq!(VEC.varlen_element_offset(0), 24);
    assert_eq!(VEC.varlen_element_offset(1), 32);

    assert_eq!(VEC.allocation_size(0), 24);
    assert_eq!(VEC.allocation_size(3), 48);
}

#[test]
fn mixed_fields_and_raw_bytes() {
    const OBJ: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(1)
        .with_raw_bytes(4);

    assert_eq!(OBJ.value_field_offset(0), 8);
    assert_eq!(OBJ.raw_data_offset(), 16);
    assert_eq!(OBJ.allocation_size(0), 24);
}

#[test]
fn mixed_fields_raw_bytes_and_varlen() {
    const OBJ: TypeInfo = TypeInfo::for_header(Full::SIZE)
        .with_fields(2)
        .with_raw_bytes(3)
        .with_varlen_values(2);

    assert_eq!(OBJ.value_field_count, 2);
    assert_eq!(OBJ.raw_data_offset(), 32);
    assert_eq!(OBJ.varlen_count_offset(), 40);
    assert_eq!(OBJ.varlen_element_offset(0), 48);

    assert_eq!(OBJ.allocation_size(2), 64);
}

// ────────────────────────────────────────────────────────────────────
// ObjHeader implementations
// ────────────────────────────────────────────────────────────────────

#[test]
fn compact_header_size() {
    assert_eq!(Compact::SIZE, 8);
    assert_eq!(core::mem::size_of::<Compact>(), 8);
}

#[test]
fn full_header_size() {
    assert_eq!(Full::SIZE, 16);
    assert_eq!(core::mem::size_of::<Full>(), 16);
}

#[test]
fn compact_header_stores_type_info() {
    let _info = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let header = Compact::new(7);
    assert_eq!(header.type_id(), 7);
}

#[test]
fn full_header_stores_type_info_and_gc_word() {
    let _info = TypeInfo::for_header(Full::SIZE).with_fields(1);
    let mut header = Full::new(7);
    assert_eq!(header.type_id(), 7);
    assert_eq!(header.gc_word(), 0);

    header.set_gc_word(0xDEAD_BEEF);
    assert_eq!(header.gc_word(), 0xDEAD_BEEF);
}

// ────────────────────────────────────────────────────────────────────
// Raw memory helpers (no dynvalue dep)
// ────────────────────────────────────────────────────────────────────

fn alloc_raw(info: &TypeInfo, varlen_len: usize) -> *mut u8 {
    let size = info.allocation_size(varlen_len);
    let layout = std::alloc::Layout::from_size_align(size, 1 << info.align_log2).unwrap();
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout);
        assert!(!ptr.is_null());
        ptr
    }
}

unsafe fn dealloc_raw(ptr: *mut u8, info: &TypeInfo, varlen_len: usize) {
    let size = info.allocation_size(varlen_len);
    let layout = std::alloc::Layout::from_size_align(size, 1 << info.align_log2).unwrap();
    unsafe { std::alloc::dealloc(ptr, layout) }
}

/// Write a raw u64 into value field `index` (no tag scheme).
unsafe fn write_u64_field(obj: *mut u8, info: &TypeInfo, index: u16, val: u64) {
    let offset = info.value_field_offset(index);
    unsafe {
        core::ptr::write(obj.add(offset) as *mut u64, val);
    }
}

/// Read a raw u64 from value field `index`.
unsafe fn read_u64_field(obj: *const u8, info: &TypeInfo, index: u16) -> u64 {
    let offset = info.value_field_offset(index);
    unsafe { core::ptr::read(obj.add(offset) as *const u64) }
}

unsafe fn write_u64_varlen(obj: *mut u8, info: &TypeInfo, index: usize, val: u64) {
    let offset = info.varlen_element_offset(index);
    unsafe {
        core::ptr::write(obj.add(offset) as *mut u64, val);
    }
}

unsafe fn read_u64_varlen(obj: *const u8, info: &TypeInfo, index: usize) -> u64 {
    let offset = info.varlen_element_offset(index);
    unsafe { core::ptr::read(obj.add(offset) as *const u64) }
}

#[test]
fn raw_bytes_read_write() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(0)
        .with_raw_bytes(4);

    let obj = alloc_raw(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, 0);

        let data = raw_data_mut(obj, &INFO);
        data.copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let read = read_raw_bytes(obj, &INFO);
        assert_eq!(read, &[0xDE, 0xAD, 0xBE, 0xEF]);

        dealloc_raw(obj, &INFO, 0);
    }
}

#[test]
fn varlen_bytes_read_write() {
    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_varlen_bytes(0);

    let data = b"hello, world!";
    let n = data.len();
    let obj = alloc_raw(&INFO, n);
    unsafe {
        init_header::<Full>(obj, 0);
        write_varlen_count(obj, &INFO, n);

        let base = INFO.varlen_count_offset() + 8;
        core::ptr::copy_nonoverlapping(data.as_ptr(), obj.add(base), n);

        let bytes = read_varlen_bytes(obj, &INFO);
        assert_eq!(bytes, data);

        dealloc_raw(obj, &INFO, n);
    }
}

// ────────────────────────────────────────────────────────────────────
// scan_object (raw u64 slots)
// ────────────────────────────────────────────────────────────────────

#[test]
fn scan_fixed_fields() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(3);

    let obj = alloc_raw(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, 0);

        for i in 0..3u16 {
            write_u64_field(obj, &INFO, i, (i as u64 + 1) * 100);
        }

        let mut slots: Vec<u64> = Vec::new();
        scan_object(obj, &INFO, |slot| {
            slots.push(core::ptr::read(slot));
        });

        assert_eq!(slots, vec![100, 200, 300]);

        dealloc_raw(obj, &INFO, 0);
    }
}

#[test]
fn scan_varlen_values() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(1);

    let varlen_n = 4usize;
    let obj = alloc_raw(&INFO, varlen_n);
    unsafe {
        init_header::<Compact>(obj, 0);

        write_u64_field(obj, &INFO, 0, 999);

        write_varlen_count(obj, &INFO, varlen_n);
        for i in 0..varlen_n {
            write_u64_varlen(obj, &INFO, i, i as u64);
        }

        let mut slots: Vec<u64> = Vec::new();
        scan_object(obj, &INFO, |slot| {
            slots.push(core::ptr::read(slot));
        });

        assert_eq!(slots.len(), 5);
        assert_eq!(slots[0], 999);
        for i in 0..varlen_n {
            assert_eq!(slots[1 + i], i as u64);
        }

        dealloc_raw(obj, &INFO, varlen_n);
    }
}

#[test]
fn scan_bytes_object_skips_bytes() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_bytes(1);

    let varlen_n = 10usize;
    let obj = alloc_raw(&INFO, varlen_n);
    unsafe {
        init_header::<Compact>(obj, 0);

        write_u64_field(obj, &INFO, 0, 42);
        write_varlen_count(obj, &INFO, varlen_n);

        let mut count = 0usize;
        scan_object(obj, &INFO, |_slot| {
            count += 1;
        });

        assert_eq!(count, 1);

        dealloc_raw(obj, &INFO, varlen_n);
    }
}

#[test]
fn scan_no_fields() {
    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE)
        .with_fields(0)
        .with_raw_bytes(16);

    let obj = alloc_raw(&INFO, 0);
    unsafe {
        init_header::<Full>(obj, 0);

        let mut count = 0usize;
        scan_object(obj, &INFO, |_slot| {
            count += 1;
        });

        assert_eq!(count, 0);

        dealloc_raw(obj, &INFO, 0);
    }
}

#[test]
fn scan_can_update_slots() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let obj = alloc_raw(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, 0);
        write_u64_field(obj, &INFO, 0, 100);
        write_u64_field(obj, &INFO, 1, 200);

        scan_object(obj, &INFO, |slot| {
            let bits = core::ptr::read(slot);
            core::ptr::write(slot, bits * 2);
        });

        assert_eq!(read_u64_field(obj, &INFO, 0), 200);
        assert_eq!(read_u64_field(obj, &INFO, 1), 400);

        dealloc_raw(obj, &INFO, 0);
    }
}

// ────────────────────────────────────────────────────────────────────
// Shadow Frame Chain
// ────────────────────────────────────────────────────────────────────

#[test]
fn frame_chain_empty() {
    let chain = FrameChain::new();
    assert_eq!(chain.depth(), 0);

    let mut slots = vec![];
    chain.scan_roots(&mut |slot| {
        slots.push(unsafe { *slot });
    });
    assert!(slots.is_empty());
}

#[test]
fn frame_chain_push_pop() {
    let chain = FrameChain::new();
    let frame = RootFrame::<3>::new();
    frame.slots[0].set(0xAABB);
    frame.slots[1].set(0xCCDD);
    frame.slots[2].set(0xEEFF);

    {
        let _guard = chain.push(&frame);
        assert_eq!(chain.depth(), 1);

        let mut slots = vec![];
        chain.scan_roots(&mut |slot| {
            slots.push(unsafe { *slot });
        });
        assert_eq!(slots, vec![0xAABB, 0xCCDD, 0xEEFF]);
    }
    assert_eq!(chain.depth(), 0);
}

#[test]
fn frame_chain_nested_frames() {
    let chain = FrameChain::new();
    let outer = RootFrame::<2>::new();
    let inner = RootFrame::<1>::new();

    outer.slots[0].set(100);
    outer.slots[1].set(200);

    let _guard_outer = chain.push(&outer);
    assert_eq!(chain.depth(), 1);

    inner.slots[0].set(300);
    {
        let _guard_inner = chain.push(&inner);
        assert_eq!(chain.depth(), 2);

        let mut slots = vec![];
        chain.scan_roots(&mut |slot| {
            slots.push(unsafe { *slot });
        });
        assert_eq!(slots, vec![300, 100, 200]);
    }
    assert_eq!(chain.depth(), 1);

    let mut slots = vec![];
    chain.scan_roots(&mut |slot| {
        slots.push(unsafe { *slot });
    });
    assert_eq!(slots, vec![100, 200]);
}

#[test]
fn frame_chain_gc_updates_slots() {
    let chain = FrameChain::new();
    let frame = RootFrame::<2>::new();
    frame.slots[0].set(0x1000);
    frame.slots[1].set(0x2000);

    let _guard = chain.push(&frame);

    chain.scan_roots(&mut |slot| unsafe {
        let old = *slot;
        *slot = old * 2;
    });

    assert_eq!(frame.slots[0].get(), 0x2000);
    assert_eq!(frame.slots[1].get(), 0x4000);
}

// ────────────────────────────────────────────────────────────────────
// RootSet
// ────────────────────────────────────────────────────────────────────

#[test]
fn root_set_basic() {
    let mut rs = RootSet::new();
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
fn root_set_scan_roots() {
    let mut rs = RootSet::new();
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
fn root_set_gc_update() {
    let mut rs = RootSet::new();
    rs.add(100);
    rs.add(200);

    rs.scan_roots(&mut |slot| unsafe {
        *slot += 1;
    });

    assert_eq!(rs.get(0), 101);
    assert_eq!(rs.get(1), 201);
}

// ────────────────────────────────────────────────────────────────────
// Alignment
// ────────────────────────────────────────────────────────────────────

#[test]
fn custom_alignment() {
    const INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(1)
        .with_align_log2(4);

    assert_eq!(INFO.allocation_size(0), 16);
}

#[test]
fn allocation_size_respects_alignment() {
    const INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(0)
        .with_raw_bytes(3)
        .with_align_log2(4);

    assert_eq!(INFO.allocation_size(0), 16);
}

// ────────────────────────────────────────────────────────────────────
// TYPE_ID_OFFSET
// ────────────────────────────────────────────────────────────────────

#[test]
fn compact_type_id_offset() {
    assert_eq!(Compact::TYPE_ID_OFFSET, 0);
}

#[test]
fn full_type_id_offset() {
    assert_eq!(Full::TYPE_ID_OFFSET, 8);
}

#[test]
fn read_type_id_compact() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let obj = alloc_raw(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, 42);
        let recovered = read_type_id(obj, Compact::TYPE_ID_OFFSET);
        assert_eq!(recovered, 42);
        dealloc_raw(obj, &INFO, 0);
    }
}

#[test]
fn read_type_id_full() {
    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_fields(1);
    let obj = alloc_raw(&INFO, 0);
    unsafe {
        init_header::<Full>(obj, 42);
        let recovered = read_type_id(obj, Full::TYPE_ID_OFFSET);
        assert_eq!(recovered, 42);
        dealloc_raw(obj, &INFO, 0);
    }
}

// ────────────────────────────────────────────────────────────────────
// BumpAllocator
// ────────────────────────────────────────────────────────────────────

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
        .with_align_log2(4);
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = bump.alloc(&INFO, 0);
    assert!(!ptr.is_null());
    assert_eq!(ptr as usize % 16, 0);
}

#[test]
fn bump_zeroed() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = bump.alloc(&INFO, 0);
    assert!(!ptr.is_null());
    let size = INFO.allocation_size(0);
    for i in 0..size {
        assert_eq!(unsafe { *ptr.add(i) }, 0);
    }
}

#[test]
fn bump_full_returns_null() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let obj_size = INFO.allocation_size(0);
    let bump = BumpAllocator::new::<Compact>(obj_size);

    let ptr1 = bump.alloc(&INFO, 0);
    assert!(!ptr1.is_null());

    let ptr2 = bump.alloc(&INFO, 0);
    assert!(ptr2.is_null());
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

    let ptr2 = bump.alloc(&INFO, 0);
    assert!(!ptr2.is_null());
}

#[test]
fn bump_alloc_obj_convenience() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(2);
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = unsafe { alloc_obj::<Compact>(&bump, &INFO, 0) };
    assert!(!ptr.is_null());

    unsafe {
        let header = core::ptr::read(ptr as *const Compact);
        assert_eq!(header.type_id(), INFO.type_id);
    }
}

#[test]
fn bump_alloc_obj_varlen() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_varlen_values(1);
    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr = unsafe { alloc_obj::<Compact>(&bump, &INFO, 5) };
    assert!(!ptr.is_null());

    unsafe {
        let header = core::ptr::read(ptr as *const Compact);
        assert_eq!(header.type_id(), INFO.type_id);
        assert_eq!(read_varlen_count(ptr, &INFO), 5);
    }
}

#[test]
fn bump_heap_walk() {
    static INFO_A: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(1);
    static INFO_B: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(1)
        .with_fields(2);
    static INFO_C: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(2)
        .with_fields(0);

    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr_a = unsafe { alloc_obj::<Compact>(&bump, &INFO_A, 0) };
    let ptr_b = unsafe { alloc_obj::<Compact>(&bump, &INFO_B, 0) };
    let ptr_c = unsafe { alloc_obj::<Compact>(&bump, &INFO_C, 0) };

    let mut visited: Vec<(*mut u8, u16)> = Vec::new();
    unsafe {
        bump.walk(&[INFO_A, INFO_B, INFO_C], &mut |ptr, info| {
            visited.push((ptr, info.type_id));
        });
    }

    assert_eq!(visited.len(), 3);
    assert_eq!(visited[0], (ptr_a, INFO_A.type_id));
    assert_eq!(visited[1], (ptr_b, INFO_B.type_id));
    assert_eq!(visited[2], (ptr_c, INFO_C.type_id));
}

#[test]
fn bump_heap_walk_varlen() {
    static INFO_FIX: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(1);
    static INFO_VEC: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(1)
        .with_varlen_values(0);
    static INFO_STR: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(2)
        .with_varlen_bytes(0);

    let bump = BumpAllocator::new::<Compact>(4096);

    let ptr_fix = unsafe { alloc_obj::<Compact>(&bump, &INFO_FIX, 0) };
    let ptr_vec = unsafe { alloc_obj::<Compact>(&bump, &INFO_VEC, 3) };
    let ptr_str = unsafe { alloc_obj::<Compact>(&bump, &INFO_STR, 10) };

    let mut visited: Vec<(*mut u8, u16)> = Vec::new();
    unsafe {
        bump.walk(&[INFO_FIX, INFO_VEC, INFO_STR], &mut |ptr, info| {
            visited.push((ptr, info.type_id));
        });
    }

    assert_eq!(visited.len(), 3);
    assert_eq!(visited[0], (ptr_fix, INFO_FIX.type_id));
    assert_eq!(visited[1], (ptr_vec, INFO_VEC.type_id));
    assert_eq!(visited[2], (ptr_str, INFO_STR.type_id));
}

// ────────────────────────────────────────────────────────────────────
// AtomicBumpAllocator
// ────────────────────────────────────────────────────────────────────

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
    use std::sync::Arc;

    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = Arc::new(AtomicBumpAllocator::new::<Compact>(1024 * 1024));

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let bump = Arc::clone(&bump);
            std::thread::spawn(move || {
                let mut addrs = Vec::new();
                for _ in 0..100 {
                    let p = bump.alloc(&INFO, 0);
                    assert!(!p.is_null());
                    addrs.push(p as usize);
                }
                addrs
            })
        })
        .collect();

    let mut all_addrs = Vec::new();
    for t in threads {
        all_addrs.extend(t.join().unwrap());
    }

    all_addrs.sort();
    let len_before = all_addrs.len();
    all_addrs.dedup();
    assert_eq!(all_addrs.len(), len_before);
}

#[test]
fn atomic_bump_exhaustion() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let size = INFO.allocation_size(0);
    let bump = AtomicBumpAllocator::new::<Compact>(size);

    assert!(!bump.alloc(&INFO, 0).is_null());
    assert!(bump.alloc(&INFO, 0).is_null());
}

#[test]
fn atomic_bump_reset() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = AtomicBumpAllocator::new::<Compact>(4096);

    let _ = bump.alloc(&INFO, 0);
    assert!(bump.used() > 0);

    bump.reset();
    assert_eq!(bump.used(), 0);

    assert!(!bump.alloc(&INFO, 0).is_null());
}

// ────────────────────────────────────────────────────────────────────
// SemiSpace GC (IdentityPtrPolicy — raw pointers as values)
// ────────────────────────────────────────────────────────────────────

struct SingleRoot(Cell<u64>);

impl RootSource for SingleRoot {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        visitor(self.0.as_ptr());
    }
}

/// Wraps a Vec of cells.
#[allow(dead_code)]
struct VecRoots(Vec<Cell<u64>>);

impl RootSource for VecRoots {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for cell in &self.0 {
            visitor(cell.as_ptr());
        }
    }
}

#[test]
fn semi_space_collect_preserves_live_object() {
    static INFO_LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(0);

    let type_table = vec![INFO_LEAF];

    let mut gc = SemiSpace::new::<Compact>(4096);
    let obj = gc.alloc_obj::<Compact>(&INFO_LEAF, 0);
    assert!(!obj.is_null());

    let root = SingleRoot(Cell::new(obj as u64));

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };

    let new_ptr = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(new_ptr));
    assert_ne!(new_ptr, obj, "object should have been moved");
}

#[test]
fn semi_space_dead_objects_reclaimed() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(0);

    let type_table = vec![INFO];

    let mut gc = SemiSpace::new::<Compact>(4096);

    // Allocate 5 objects, root only the first one
    let kept = gc.alloc_obj::<Compact>(&INFO, 0);
    let _dead1 = gc.alloc_obj::<Compact>(&INFO, 0);
    let _dead2 = gc.alloc_obj::<Compact>(&INFO, 0);
    let _dead3 = gc.alloc_obj::<Compact>(&INFO, 0);
    let _dead4 = gc.alloc_obj::<Compact>(&INFO, 0);

    let before = gc.from_used();
    let root = SingleRoot(Cell::new(kept as u64));

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };

    let after = gc.from_used();
    assert!(after < before, "garbage should have been reclaimed");
    // Exactly one object survives
    assert_eq!(after, INFO.allocation_size(0));
}

#[test]
fn semi_space_pointer_fixup() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(2);

    let type_table = vec![PAIR];

    let mut gc = SemiSpace::new::<Compact>(4096);
    let child = gc.alloc_obj::<Compact>(&PAIR, 0);
    let parent = gc.alloc_obj::<Compact>(&PAIR, 0);
    unsafe {
        write_u64_field(parent, &PAIR, 0, child as u64);
        write_u64_field(parent, &PAIR, 1, 0);
    }

    let root = SingleRoot(Cell::new(parent as u64));

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };

    let new_parent = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    let field0 = unsafe { read_u64_field(new_parent, &PAIR, 0) };
    let new_child = IdentityPtrPolicy::try_decode_ptr(field0).unwrap();
    assert!(gc.contains(new_child));
    assert_ne!(new_child, child, "child should have moved too");
}

/// ADVERSARIAL precise-layout proof for the int/pointer-discrimination P0.
///
/// Models a monomorphized mixed enum like `Either { L(ref), R(i64) }`: one
/// TRACED pointer slot (`with_fields(1)`) followed by an UNTRACED raw i64
/// (`with_raw_bytes(8)` — the `R(i64)` payload region). Into that raw region we
/// store the *bit pattern of a real, live, in-from-space heap address* — the
/// maximally adversarial integer: it IS a valid 8-aligned from-space pointer, so
/// the ONLY thing stopping the collector from relocating it is that the raw
/// region is never scanned. (A small int like 42 proves nothing — it is not in
/// from-space, so `from.contains` is false and it could never trip the bug even
/// if it leaked into a traced slot.)
///
/// If the layout is genuinely precise, this adversarial int is UNTOUCHED across
/// a collection (it keeps the stale pre-move address) even though the object it
/// numerically points at moves. If a scalar could ever reach a traced slot,
/// THIS is the value that would be silently mis-moved — the documented bug.
#[test]
fn semi_space_adversarial_int_payload_in_raw_region_not_relocated() {
    // slot 0: traced (the `L` ref); raw i64: the `R(i64)` payload — UNTRACED.
    static MIXED: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(1)
        .with_raw_bytes(8);
    let type_table = vec![MIXED];

    let mut gc = SemiSpace::new::<Compact>(4096);

    // `referee` is a real object kept live via the traced slot (so it survives
    // and is relocated). `mixed` carries the adversarial int in its raw region.
    let referee = gc.alloc_obj::<Compact>(&MIXED, 0);
    let mixed = gc.alloc_obj::<Compact>(&MIXED, 0);

    unsafe { write_u64_field(mixed, &MIXED, 0, referee as u64) }; // traced -> referee
    let raw_off = MIXED.raw_data_offset();
    let adversarial = referee as u64; // a REAL 8-aligned from-space address
    assert_eq!(adversarial & 0b111, 0, "from-space objects are 8-aligned");
    assert!(gc.contains(adversarial as *const u8), "addr is in from-space");
    unsafe { core::ptr::write(mixed.add(raw_off) as *mut u64, adversarial) };

    let root = SingleRoot(Cell::new(mixed as u64));
    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };

    // After GC both `mixed` and `referee` have moved.
    let new_mixed = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    let new_referee = unsafe { read_u64_field(new_mixed, &MIXED, 0) };
    assert!(gc.contains(new_referee as *const u8));
    assert_ne!(new_referee, referee as u64, "traced slot must follow the move");

    // THE PROOF: the raw-region int is UNCHANGED. The collector never scanned
    // it, so even an int whose bits are a valid from-space address is never
    // relocated. (It is now a stale/dangling address — correct: a scalar is
    // opaque to the GC.) If this fires, a scalar reached a traced slot.
    let raw_after = unsafe { core::ptr::read(new_mixed.add(raw_off) as *const u64) };
    assert_eq!(
        raw_after, adversarial,
        "raw-region scalar was relocated — a non-pointer leaked into a traced \
         slot (precise-layout violation)"
    );
}

/// NEGATIVE CONTROL for the proof above: the collector itself does NOT and
/// cannot distinguish an integer from a pointer — a real from-space address in
/// a TRACED slot is unconditionally relocated. This is the counterfactual: if
/// the front end ever placed a scalar in a traced slot, this is the silent
/// mis-move that would result. Soundness rests ENTIRELY on precise layout
/// (scalars in the raw region), never on the collector guessing — which is why
/// the conservative `type_id`-range silent-skip was removed in favour of a
/// layout invariant + a panicking detector.
#[test]
fn semi_space_traced_slot_relocated_regardless_of_value_negative_control() {
    static TWO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(2);
    let type_table = vec![TWO];
    let mut gc = SemiSpace::new::<Compact>(4096);
    let referee = gc.alloc_obj::<Compact>(&TWO, 0);
    let holder = gc.alloc_obj::<Compact>(&TWO, 0);
    unsafe {
        write_u64_field(holder, &TWO, 0, referee as u64);
        write_u64_field(holder, &TWO, 1, referee as u64); // same bits in a 2nd traced slot
    }
    let root = SingleRoot(Cell::new(holder as u64));
    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };
    let new_holder = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    let s0 = unsafe { read_u64_field(new_holder, &TWO, 0) };
    let s1 = unsafe { read_u64_field(new_holder, &TWO, 1) };
    assert_ne!(s0, referee as u64, "traced slot 0 must be relocated");
    assert_eq!(s0, s1, "both traced slots followed the same object to its new home");
}

/// PROVE THE DETECTOR TRIPS. A detector that never demonstrably fires is
/// indistinguishable from a no-op. Here we deliberately simulate the failure the
/// precise-layout invariant forbids — a traced slot pointing at bits whose
/// header `type_id` is out of range (what a leaked scalar, or a corrupted
/// reference, looks like) — and assert the collector PANICS loudly rather than
/// following garbage or silently skipping. Armed unconditionally in debug; the
/// same panic occurs in release under `GCR_GC_VERIFY=1` (gated off here because
/// release-default would instead fault with a generic bounds-check message).
#[test]
#[should_panic(expected = "precise-layout violation")]
#[cfg_attr(not(debug_assertions), ignore = "detector armed via GCR_GC_VERIFY in release")]
fn semi_space_detector_panics_on_bad_header_in_traced_slot() {
    static ONE: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(1);
    let type_table = vec![ONE]; // table len 1 → only type_id 0 is valid
    let mut gc = SemiSpace::new::<Compact>(4096);

    // A from-space object whose header we corrupt to an out-of-range type_id —
    // standing in for a non-pointer (or corrupted ref) reached via a traced slot.
    let bad_target = gc.alloc_obj::<Compact>(&ONE, 0);
    unsafe { core::ptr::write(bad_target as *mut u64, 9999u64) }; // type_id 9999 ≫ len 1

    let holder = gc.alloc_obj::<Compact>(&ONE, 0);
    unsafe { write_u64_field(holder, &ONE, 0, bad_target as u64) }; // traced slot → bad bits

    let root = SingleRoot(Cell::new(holder as u64));
    // Following the traced slot, copy_or_forward reads the out-of-range type_id
    // and the armed detector panics — NOT a silent `return old`.
    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };
}

#[test]
fn semi_space_chain() {
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(1);

    let type_table = vec![NODE];

    let mut gc = SemiSpace::new::<Compact>(4096);
    let d = gc.alloc_obj::<Compact>(&NODE, 0);
    let c = gc.alloc_obj::<Compact>(&NODE, 0);
    let b = gc.alloc_obj::<Compact>(&NODE, 0);
    let a = gc.alloc_obj::<Compact>(&NODE, 0);

    unsafe {
        write_u64_field(a, &NODE, 0, b as u64);
        write_u64_field(b, &NODE, 0, c as u64);
        write_u64_field(c, &NODE, 0, d as u64);
        write_u64_field(d, &NODE, 0, 0);
    }

    let root = SingleRoot(Cell::new(a as u64));

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };

    // Walk the chain — every link should be valid in new from-space
    let mut cur = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    let mut count = 1;
    while let Some(next) =
        unsafe { IdentityPtrPolicy::try_decode_ptr(read_u64_field(cur, &NODE, 0)) }
    {
        assert!(gc.contains(next));
        cur = next;
        count += 1;
    }
    assert_eq!(count, 4);
}

#[test]
fn semi_space_cycle() {
    static NODE: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(1);

    let type_table = vec![NODE];

    let mut gc = SemiSpace::new::<Compact>(4096);
    let b = gc.alloc_obj::<Compact>(&NODE, 0);
    let a = gc.alloc_obj::<Compact>(&NODE, 0);

    unsafe {
        write_u64_field(a, &NODE, 0, b as u64);
        write_u64_field(b, &NODE, 0, a as u64);
    }

    let root = SingleRoot(Cell::new(a as u64));

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };

    let new_a = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    let field_a = unsafe { read_u64_field(new_a, &NODE, 0) };
    let new_b = IdentityPtrPolicy::try_decode_ptr(field_a).unwrap();
    let field_b = unsafe { read_u64_field(new_b, &NODE, 0) };
    let back_to_a = IdentityPtrPolicy::try_decode_ptr(field_b).unwrap();
    assert_eq!(back_to_a, new_a);
}

#[test]
fn semi_space_multiple_collections() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(0);

    let type_table = vec![INFO];

    let mut gc = SemiSpace::new::<Compact>(4096);
    let obj = gc.alloc_obj::<Compact>(&INFO, 0);
    let root = SingleRoot(Cell::new(obj as u64));

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };
    let after_1 = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(after_1));

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };
    let after_2 = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    assert!(gc.contains(after_2));

    assert_eq!(gc.collections(), 2);
}

#[test]
fn semi_space_varlen_values() {
    static VEC: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_varlen_values(0);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(1)
        .with_fields(0);

    let type_table = vec![VEC, LEAF];

    let mut gc = SemiSpace::new::<Compact>(4096);

    let elem0 = gc.alloc_obj::<Compact>(&LEAF, 0);
    let elem1 = gc.alloc_obj::<Compact>(&LEAF, 0);
    let vec = gc.alloc_obj::<Compact>(&VEC, 3);

    unsafe {
        write_u64_varlen(vec, &VEC, 0, elem0 as u64);
        write_u64_varlen(vec, &VEC, 1, elem1 as u64);
        write_u64_varlen(vec, &VEC, 2, 0);
    }

    let root = SingleRoot(Cell::new(vec as u64));
    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&root]) };

    let new_vec = IdentityPtrPolicy::try_decode_ptr(root.0.get()).unwrap();
    unsafe {
        assert_eq!(read_varlen_count(new_vec, &VEC), 3);
        let v0 = read_u64_varlen(new_vec, &VEC, 0);
        let v1 = read_u64_varlen(new_vec, &VEC, 1);
        let v2 = read_u64_varlen(new_vec, &VEC, 2);
        assert!(IdentityPtrPolicy::try_decode_ptr(v0).is_some());
        assert!(IdentityPtrPolicy::try_decode_ptr(v1).is_some());
        assert!(IdentityPtrPolicy::try_decode_ptr(v2).is_none());
    }
}

#[test]
fn semi_space_collect_empty() {
    let type_table: Vec<TypeInfo> = vec![];
    let mut gc = SemiSpace::new::<Compact>(4096);
    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut []) };
    assert_eq!(gc.from_used(), 0);
    assert_eq!(gc.collections(), 1);
}

#[test]
fn semi_space_with_frame_chain() {
    static PAIR: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(2);

    let type_table = vec![PAIR];

    let mut gc = SemiSpace::new::<Compact>(4096);
    let child = gc.alloc_obj::<Compact>(&PAIR, 0);
    let parent = gc.alloc_obj::<Compact>(&PAIR, 0);
    unsafe {
        write_u64_field(parent, &PAIR, 0, child as u64);
        write_u64_field(parent, &PAIR, 1, 77);
    }

    let chain = FrameChain::new();
    let frame = RootFrame::<2>::new();
    frame.slots[0].set(parent as u64);
    frame.slots[1].set(child as u64);

    let _guard = chain.push(&frame);

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&chain]) };

    let new_parent = IdentityPtrPolicy::try_decode_ptr(frame.slots[0].get()).unwrap();
    let new_child = IdentityPtrPolicy::try_decode_ptr(frame.slots[1].get()).unwrap();
    assert!(gc.contains(new_parent));
    assert!(gc.contains(new_child));

    let f0 = unsafe { read_u64_field(new_parent, &PAIR, 0) };
    let child_from_field = IdentityPtrPolicy::try_decode_ptr(f0).unwrap();
    assert_eq!(child_from_field, new_child);

    let f1 = unsafe { read_u64_field(new_parent, &PAIR, 1) };
    assert_eq!(f1, 77);
}

#[test]
fn semi_space_with_root_set() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(0);

    let type_table = vec![INFO];

    let mut gc = SemiSpace::new::<Compact>(4096);
    let g1 = gc.alloc_obj::<Compact>(&INFO, 0);
    let g2 = gc.alloc_obj::<Compact>(&INFO, 0);

    let mut globals = RootSet::new();
    let i0 = globals.add(g1 as u64);
    let i1 = globals.add(g2 as u64);

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&globals]) };

    let new_g1 = IdentityPtrPolicy::try_decode_ptr(globals.get(i0)).unwrap();
    let new_g2 = IdentityPtrPolicy::try_decode_ptr(globals.get(i1)).unwrap();
    assert!(gc.contains(new_g1));
    assert!(gc.contains(new_g2));
}

#[test]
fn semi_space_traces_and_relocates_interior_pointer() {
    // HOLDER: a 16-byte raw region holding a scalar at raw offset 0 and a GC ref
    // at raw offset 8 (absolute offset header+8) — an *interior* pointer (a ref
    // embedded in a flattened value field), not a leading value slot. It is only
    // reachable to the collector via `interior_ptrs`.
    static INTERIOR: [u16; 1] = [Compact::SIZE as u16 + 8];
    static HOLDER: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_raw_bytes(16)
        .with_interior_ptrs(&INTERIOR);
    static LEAF: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_type_id(1);

    let type_table = vec![HOLDER, LEAF];
    let mut gc = SemiSpace::new::<Compact>(4096);

    let child = gc.alloc_obj::<Compact>(&LEAF, 0);
    let holder = gc.alloc_obj::<Compact>(&HOLDER, 0);
    let scalar_off = Compact::SIZE; // raw offset 0 — NOT a pointer
    let ref_off = Compact::SIZE + 8; // raw offset 8 — the interior pointer
    unsafe {
        (holder.add(scalar_off) as *mut u64).write(42);
        (holder.add(ref_off) as *mut u64).write(child as u64);
    }

    // Root ONLY the holder. The child survives solely via the interior pointer —
    // if `scan_object` doesn't trace `interior_ptrs`, it is collected (dangling).
    let chain = FrameChain::new();
    let frame = RootFrame::<1>::new();
    frame.slots[0].set(holder as u64);
    let _guard = chain.push(&frame);

    unsafe { gc.collect::<IdentityPtrPolicy>(&type_table, &mut [&chain]) };

    let new_holder = IdentityPtrPolicy::try_decode_ptr(frame.slots[0].get()).unwrap();
    assert!(gc.contains(new_holder));
    // The raw scalar at offset 0 is untraced and preserved verbatim.
    let scalar = unsafe { (new_holder.add(scalar_off) as *const u64).read() };
    assert_eq!(scalar, 42);
    // The interior ref was traced, the child relocated, and the slot fixed in place.
    let raw = unsafe { (new_holder.add(ref_off) as *const u64).read() };
    let new_child = IdentityPtrPolicy::try_decode_ptr(raw).unwrap();
    assert!(
        gc.contains(new_child),
        "interior pointer not traced → child collected (dangling)"
    );
}

// ────────────────────────────────────────────────────────────────────
// Mutator
// ────────────────────────────────────────────────────────────────────

#[test]
fn mutator_root_get_set() {
    let mut m = Mutator::new();
    let r = m.root(42);
    assert_eq!(m.get(&r).bits(), 42);
    m.set(&r, 99);
    assert_eq!(m.get(&r).bits(), 99);
}

#[test]
fn mutator_alloc_returns_rooted_object() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let bump = BumpAllocator::new::<Compact>(4096);
    let mut m = Mutator::new();

    let r = m.alloc::<Compact>(&bump, &INFO, 0).unwrap();
    let ptr = m.get(&r).bits() as *const u8;
    assert!(!ptr.is_null());
    assert!(bump.contains(ptr));
}

#[test]
fn mutator_save_restore_scoping() {
    let mut m = Mutator::new();
    let r1 = m.root(1);
    let scope = m.save();

    let _r2 = m.root(2);
    let _r3 = m.root(3);
    assert_eq!(m.root_count(), 3);

    m.restore(scope);
    assert_eq!(m.root_count(), 1);
    assert_eq!(m.get(&r1).bits(), 1);
}

#[test]
fn mutator_alloc_full_returns_none() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let size = INFO.allocation_size(0);
    let bump = BumpAllocator::new::<Compact>(size);
    let mut m = Mutator::new();

    let _r1 = m.alloc::<Compact>(&bump, &INFO, 0).unwrap();
    assert!(m.alloc::<Compact>(&bump, &INFO, 0).is_none());
}

// ────────────────────────────────────────────────────────────────────
// Heap
// ────────────────────────────────────────────────────────────────────

#[test]
fn heap_basic_alloc_and_collect() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(0);

    let heap = Heap::new::<Compact>(4096, vec![INFO]);

    let obj = heap.alloc_obj::<Compact>(&INFO, 0);
    assert!(!obj.is_null());

    let root = SingleRoot(Cell::new(obj as u64));

    unsafe { heap.collect::<IdentityPtrPolicy>(&[&root]) };

    assert!(heap.from_used() > 0);
    assert!(IdentityPtrPolicy::try_decode_ptr(root.0.get()).is_some());
}

#[test]
fn heap_dead_objects_reclaimed() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(0);

    let heap = Heap::new::<Compact>(4096, vec![INFO]);

    let kept = heap.alloc_obj::<Compact>(&INFO, 0);
    for _ in 0..5 {
        heap.alloc_obj::<Compact>(&INFO, 0);
    }

    let before = heap.from_used();
    let root = SingleRoot(Cell::new(kept as u64));

    unsafe { heap.collect::<IdentityPtrPolicy>(&[&root]) };

    assert!(heap.from_used() < before);
}

// ────────────────────────────────────────────────────────────────────
// MutatorThread (single-thread sanity)
// ────────────────────────────────────────────────────────────────────

#[test]
fn mutator_thread_basic_alloc() {
    use std::sync::Arc;
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_type_id(0)
        .with_fields(0);

    let heap = Arc::new(Heap::new::<Compact>(4096, vec![INFO]));
    let mt: MutatorThread<IdentityPtrPolicy> = MutatorThread::register(heap.clone());

    let obj = mt.alloc_obj::<Compact>(&INFO, 0);
    assert!(!obj.is_null());
}

#[test]
fn alloc_site_profile_merges_threads_and_salvages_retired() {
    use crate::gc::reflect::{AllocSite, TypeKind, TypeMeta};

    static INFO0: TypeInfo = TypeInfo::for_header(Full::SIZE).with_type_id(0).with_fields(1);
    static INFO1: TypeInfo = TypeInfo::for_header(Full::SIZE).with_type_id(1).with_fields(2);

    let heap = Heap::new::<Full>(64 * 1024, vec![INFO0, INFO1]);
    heap.set_type_meta(vec![
        TypeMeta { type_id: 0, name: "Point".into(), kind: TypeKind::Opaque },
        TypeMeta { type_id: 1, name: "Pair".into(), kind: TypeKind::Opaque },
    ]);
    // Three distinct sites: same type (Point) in two different functions is two
    // sites — the honest function+type granularity.
    heap.set_alloc_sites(vec![
        AllocSite { function: "main".into(), type_id: 0 },   // site 0
        AllocSite { function: "worker".into(), type_id: 1 }, // site 1
        AllocSite { function: "worker".into(), type_id: 0 }, // site 2
    ]);

    let (main_ts, _) = heap.register_thread();
    let (worker_ts, _) = heap.register_thread();

    // Non-atomic per-thread counters, written by the owning thread.
    unsafe {
        for _ in 0..3 {
            main_ts.record_alloc(0, 100);
        }
        for _ in 0..2 {
            worker_ts.record_alloc(1, 40);
        }
        for _ in 0..5 {
            worker_ts.record_alloc(2, 24);
        }
    }

    let prof = heap.alloc_site_profile();
    assert_eq!(prof.len(), 3);
    // Sorted by bytes desc: site0=300, site2=120, site1=80.
    assert_eq!(prof[0].site_id, 0);
    assert_eq!(prof[0].function, "main");
    assert_eq!(prof[0].type_name, "Point");
    assert_eq!(prof[0].count, 3);
    assert_eq!(prof[0].bytes, 300);

    assert_eq!(prof[1].site_id, 2);
    assert_eq!(prof[1].function, "worker");
    assert_eq!(prof[1].type_name, "Point");
    assert_eq!(prof[1].count, 5);
    assert_eq!(prof[1].bytes, 120);

    assert_eq!(prof[2].site_id, 1);
    assert_eq!(prof[2].type_name, "Pair");
    assert_eq!(prof[2].count, 2);
    assert_eq!(prof[2].bytes, 80);

    // Deregister the worker: its allocations must survive via the retired
    // accumulator (no silent loss when a thread joins before the profile).
    heap.deregister_thread(&worker_ts);
    drop(worker_ts);

    let prof2 = heap.alloc_site_profile();
    assert_eq!(prof2.len(), 3, "deregistered worker's sites must persist");
    let site1 = prof2.iter().find(|s| s.site_id == 1).unwrap();
    assert_eq!((site1.count, site1.bytes), (2, 80));
    let site2 = prof2.iter().find(|s| s.site_id == 2).unwrap();
    assert_eq!((site2.count, site2.bytes), (5, 120));
    // main's live counters still summed in too.
    let site0 = prof2.iter().find(|s| s.site_id == 0).unwrap();
    assert_eq!((site0.count, site0.bytes), (3, 300));
}

#[test]
fn alloc_site_profile_real_threads_join_then_profile() {
    use crate::gc::reflect::AllocSite;
    use std::sync::Arc;

    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_type_id(0).with_fields(0);
    let heap = Arc::new(Heap::new::<Full>(64 * 1024, vec![INFO]));
    heap.set_alloc_sites(vec![AllocSite { function: "worker".into(), type_id: 0 }]);

    let n_threads = 4u64;
    let per_thread = 1000u64;
    let bytes_each = 16u64;

    // Real OS threads (distinct thread ids), each records into its OWN non-atomic
    // counter, then drops its MutatorThread → deregisters → folds into the retired
    // accumulator. After join, the profile must sum them with no loss and no
    // double-count.
    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let h = heap.clone();
            std::thread::spawn(move || {
                let mt: MutatorThread<IdentityPtrPolicy> = MutatorThread::register(h);
                for _ in 0..per_thread {
                    // Safety: each thread writes only its own counter.
                    unsafe { mt.state().record_alloc(0, bytes_each) };
                }
                // mt drops here → deregister → fold into retired.
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    let prof = heap.alloc_site_profile();
    assert_eq!(prof.len(), 1);
    assert_eq!(prof[0].function, "worker");
    assert_eq!(prof[0].count, n_threads * per_thread);
    assert_eq!(prof[0].bytes, n_threads * per_thread * bytes_each);
}

#[test]
fn alloc_site_profile_dump_concurrent_with_live_worker_no_hang() {
    // Proves the dump-race fix: the main thread hammers alloc_site_profile (which
    // STW-pauses every other mutator via pause_world) WHILE a worker is a live,
    // registered, actively-allocating mutator. The pause must (a) never hang —
    // the worker polls a safepoint each iteration so it parks promptly, same
    // discipline as a collection — and (b) never observe a torn/garbage count
    // (monotonic, bounded by the total), because the worker is parked during the
    // read. Finally, after the worker joins, its counts survive via the retired
    // fold and the total is exact.
    use crate::gc::reflect::AllocSite;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_type_id(0).with_fields(0);
    let heap = Arc::new(Heap::new::<Full>(64 * 1024, vec![INFO]));
    heap.set_alloc_sites(vec![AllocSite { function: "worker".into(), type_id: 0 }]);

    let n_allocs = 5000u64;
    let bytes_each = 16u64;
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicU64::new(0));

    let worker = {
        let h = heap.clone();
        let stop = stop.clone();
        let done = done.clone();
        std::thread::spawn(move || {
            let mt: MutatorThread<IdentityPtrPolicy> = MutatorThread::register(h);
            for _ in 0..n_allocs {
                unsafe { mt.state().record_alloc(0, bytes_each) };
                done.fetch_add(1, Ordering::Relaxed);
                mt.safepoint(); // park here if a dump is pausing the world
            }
            // Stay a live, safepoint-responsive mutator (no more allocs) so the
            // main thread can dump while we're still registered.
            while !stop.load(Ordering::Relaxed) {
                mt.safepoint();
                std::thread::yield_now();
            }
            // mt drops -> deregister -> fold counts into retired.
        })
    };

    // Hammer the dump while the worker is live + allocating.
    let mut last = 0u64;
    for _ in 0..200 {
        let prof = heap.alloc_site_profile();
        let c = prof.iter().find(|s| s.site_id == 0).map(|s| s.count).unwrap_or(0);
        assert!(c <= n_allocs, "torn/garbage read: count {c} > total {n_allocs}");
        assert!(c >= last, "count went backwards {last} -> {c}");
        last = c;
    }

    // Let the worker finish, then stop + join (no pause active during join).
    while done.load(Ordering::Relaxed) < n_allocs {
        std::thread::yield_now();
    }
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap();

    let prof = heap.alloc_site_profile();
    assert_eq!(prof.len(), 1);
    assert_eq!(prof[0].count, n_allocs);
    assert_eq!(prof[0].bytes, n_allocs * bytes_each);
}

#[test]
fn alloc_site_profile_dump_concurrent_resize_stress_asan() {
    // ASan-oriented variant: the worker records at STRICTLY INCREASING site ids,
    // so its per-thread counter Vec keeps GROWING (Vec::resize → realloc → frees
    // the old buffer) while the main thread repeatedly dumps (alloc_site_profile
    // clones that Vec). This is EXACTLY the heap-use-after-free the pre-fix
    // reviewer reproduced under AddressSanitizer: a read of a buffer the writer
    // just realloc'd away. With the STW fix the worker is parked during every
    // read, so there is no concurrent realloc and ASan stays clean; WITHOUT the
    // fix this surfaces as an ASan UAF. Run in CI under
    // `RUSTFLAGS=-Zsanitizer=address cargo +nightly test --target <host>`.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_type_id(0).with_fields(0);
    let heap = Arc::new(Heap::new::<Full>(64 * 1024, vec![INFO]));

    let n_allocs = 2000u64;
    let bytes_each = 16u64;
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicU64::new(0));

    let worker = {
        let h = heap.clone();
        let stop = stop.clone();
        let done = done.clone();
        std::thread::spawn(move || {
            let mt: MutatorThread<IdentityPtrPolicy> = MutatorThread::register(h);
            for i in 0..n_allocs {
                // Increasing site id → the counter Vec grows (reallocs).
                unsafe { mt.state().record_alloc(i as u32, bytes_each) };
                done.fetch_add(1, Ordering::Relaxed);
                mt.safepoint();
            }
            while !stop.load(Ordering::Relaxed) {
                mt.safepoint();
                std::thread::yield_now();
            }
        })
    };

    // Dump hard while the worker is growing its Vec. A residual race = ASan UAF.
    for _ in 0..200 {
        let prof = heap.alloc_site_profile();
        let total: u64 = prof.iter().map(|s| s.count).sum();
        assert!(total <= n_allocs, "torn/garbage read: total {total} > {n_allocs}");
    }

    while done.load(Ordering::Relaxed) < n_allocs {
        std::thread::yield_now();
    }
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap();

    let prof = heap.alloc_site_profile();
    let total_count: u64 = prof.iter().map(|s| s.count).sum();
    let total_bytes: u64 = prof.iter().map(|s| s.bytes).sum();
    assert_eq!(total_count, n_allocs);
    assert_eq!(total_bytes, n_allocs * bytes_each);
}

#[test]
fn alloc_site_profile_unlabelled_sites_not_dropped() {
    // Counters with no matching site-table entry (e.g. table not installed) are
    // surfaced under a synthetic label, never silently dropped.
    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_type_id(0).with_fields(0);
    let heap = Heap::new::<Full>(4096, vec![INFO]);
    let (ts, _) = heap.register_thread();
    unsafe {
        ts.record_alloc(7, 16);
    }
    let prof = heap.alloc_site_profile();
    assert_eq!(prof.len(), 1);
    assert_eq!(prof[0].site_id, 7);
    assert_eq!(prof[0].count, 1);
    assert_eq!(prof[0].bytes, 16);
    assert!(prof[0].function.contains("site 7"));
    assert_eq!(prof[0].type_id, None);
}

#[test]
fn mutator_thread_deregisters_on_drop() {
    use std::sync::Arc;
    let heap = Arc::new(Heap::new::<Compact>(4096, vec![]));
    {
        let _mt: MutatorThread<IdentityPtrPolicy> = MutatorThread::register(heap.clone());
    }
    // Dropped — heap should be usable for a new registration.
    let _mt2: MutatorThread<IdentityPtrPolicy> = MutatorThread::register(heap.clone());
}

#[test]
fn mutator_thread_safepoint_noop_without_gc() {
    use std::sync::Arc;
    let heap = Arc::new(Heap::new::<Compact>(4096, vec![]));
    let mt: MutatorThread<IdentityPtrPolicy> = MutatorThread::register(heap.clone());
    // Should not block when no GC is requested.
    mt.safepoint();
}

// ────────────────────────────────────────────────────────────────────
// SATB + read barriers
// ────────────────────────────────────────────────────────────────────

#[test]
fn satb_buffer_basic() {
    let mut buf = SATBBuffer::new(4);
    assert!(buf.is_empty());

    buf.log(0xAAA);
    buf.log(0xBBB);
    assert_eq!(buf.len(), 2);
    assert!(!buf.should_flush());

    buf.log(0xCCC);
    buf.log(0xDDD);
    assert!(buf.should_flush());

    let drained = buf.drain();
    assert_eq!(drained, vec![0xAAA, 0xBBB, 0xCCC, 0xDDD]);
    assert!(buf.is_empty());
}

#[test]
fn satb_queue_basic() {
    let q = SATBQueue::new();
    assert!(q.is_empty());
    q.push(vec![1, 2, 3]);
    q.push(vec![4, 5]);
    assert!(!q.is_empty());
    let drained = q.drain_all();
    assert_eq!(drained, vec![1, 2, 3, 4, 5]);
    assert!(q.is_empty());
}

#[test]
fn satb_queue_empty_push_ignored() {
    let q = SATBQueue::new();
    q.push(vec![]);
    assert!(q.is_empty());
}

#[test]
fn read_barrier_null_passthrough() {
    let p = std::ptr::null_mut();
    let r = unsafe { read_barrier(p, 0) };
    assert!(r.is_null());
}

#[test]
fn read_barrier_no_forwarding() {
    let mut word: u64 = 0x0000_0000_0000_0042;
    let p = &mut word as *mut u64 as *mut u8;
    let r = unsafe { read_barrier(p, 0) };
    assert_eq!(r, p);
}

#[test]
fn read_barrier_follows_forwarding() {
    // Target word; not itself forwarded.
    let mut to_word: u64 = 0x0000_0000_0000_0007;
    let to_ptr = &mut to_word as *mut u64 as *mut u8;

    // Source whose header word is forwarding (bit 0 set).
    let mut from_word: u64 = (to_ptr as u64) | 1;
    let from_ptr = &mut from_word as *mut u64 as *mut u8;

    let r = unsafe { read_barrier(from_ptr, 0) };
    assert_eq!(r, to_ptr);
}
