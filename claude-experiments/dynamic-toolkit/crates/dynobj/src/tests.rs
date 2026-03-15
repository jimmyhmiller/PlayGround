use dynvalue::{LowBit, NanBox, Value};

use crate::*;

// ─── TypeInfo offset calculations ───────────────────────────────────

#[test]
fn compact_cons_cell_layout() {
    // Cons cell: Compact header (8) + 2 value fields
    const CONS: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    assert_eq!(CONS.header_size, 8);
    assert_eq!(CONS.value_field_count, 2);
    assert_eq!(CONS.value_field_offset(0), 8);
    assert_eq!(CONS.value_field_offset(1), 16);
    assert_eq!(CONS.allocation_size(0), 24);
}

#[test]
fn full_cons_cell_layout() {
    // Cons cell: Full header (16) + 2 value fields
    const CONS: TypeInfo = TypeInfo::for_header(Full::SIZE).with_fields(2);

    assert_eq!(CONS.header_size, 16);
    assert_eq!(CONS.value_field_offset(0), 16);
    assert_eq!(CONS.value_field_offset(1), 24);
    assert_eq!(CONS.allocation_size(0), 32);
}

#[test]
fn string_layout_compact() {
    // String: Compact header (8) + varlen bytes
    const STR: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_bytes(0);

    assert_eq!(STR.varlen_count_offset(), 8); // right after header
    assert_eq!(STR.varlen_element_offset(0), 16); // after count word
    assert_eq!(STR.varlen_element_offset(4), 20);

    // Empty string: header(8) + count(8) = 16, aligned to 8
    assert_eq!(STR.allocation_size(0), 16);
    // 5-byte string: header(8) + count(8) + 5 = 21, aligned to 24
    assert_eq!(STR.allocation_size(5), 24);
    // 8-byte string: header(8) + count(8) + 8 = 24, aligned to 24
    assert_eq!(STR.allocation_size(8), 24);
}

#[test]
fn vector_layout_full() {
    // Vector: Full header (16) + varlen values
    const VEC: TypeInfo = TypeInfo::for_header(Full::SIZE).with_varlen_values(0);

    assert_eq!(VEC.varlen_count_offset(), 16); // right after header
    assert_eq!(VEC.varlen_element_offset(0), 24); // after count word
    assert_eq!(VEC.varlen_element_offset(1), 32);

    // Empty vec: header(16) + count(8) = 24
    assert_eq!(VEC.allocation_size(0), 24);
    // 3-element vec: header(16) + count(8) + 3×8 = 48
    assert_eq!(VEC.allocation_size(3), 48);
}

#[test]
fn mixed_fields_and_raw_bytes() {
    // Object with 1 value field + 4 raw bytes: Compact header
    const OBJ: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(1)
        .with_raw_bytes(4);

    assert_eq!(OBJ.value_field_offset(0), 8);
    assert_eq!(OBJ.raw_data_offset(), 16);
    // header(8) + field(8) + raw(4) = 20, aligned to 24
    assert_eq!(OBJ.allocation_size(0), 24);
}

#[test]
fn mixed_fields_raw_bytes_and_varlen() {
    // 2 value fields + 3 raw bytes + varlen values
    const OBJ: TypeInfo = TypeInfo::for_header(Full::SIZE)
        .with_fields(2)
        .with_raw_bytes(3)
        .with_varlen_values(2); // note: with_varlen_values sets value_field_count

    // with_varlen_values overrides value_field_count, so:
    assert_eq!(OBJ.value_field_count, 2);
    assert_eq!(OBJ.raw_data_offset(), 32); // 16 + 2*8
    // raw section: 3 bytes, padded to 8
    assert_eq!(OBJ.varlen_count_offset(), 40); // align8(32 + 3) = 40
    assert_eq!(OBJ.varlen_element_offset(0), 48);

    // 2 varlen elements: header(16) + fields(16) + pad(8) + count(8) + 2×8 = 64
    assert_eq!(OBJ.allocation_size(2), 64);
}

// ─── ObjHeader implementations ─────────────────────────────────────

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
    let info = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let header = Compact::new(&info as *const TypeInfo);
    assert_eq!(header.type_info(), &info as *const TypeInfo);
}

#[test]
fn full_header_stores_type_info_and_gc_word() {
    let info = TypeInfo::for_header(Full::SIZE).with_fields(1);
    let mut header = Full::new(&info as *const TypeInfo);
    assert_eq!(header.type_info(), &info as *const TypeInfo);
    assert_eq!(header.gc_word(), 0);

    header.set_gc_word(0xDEAD_BEEF);
    assert_eq!(header.gc_word(), 0xDEAD_BEEF);
}

// ─── Field read/write with actual memory ────────────────────────────

fn alloc_obj(info: &TypeInfo, varlen_len: usize) -> *mut u8 {
    let size = info.allocation_size(varlen_len);
    let layout = std::alloc::Layout::from_size_align(size, 1 << info.align_log2).unwrap();
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout);
        assert!(!ptr.is_null());
        ptr
    }
}

unsafe fn dealloc_obj(ptr: *mut u8, info: &TypeInfo, varlen_len: usize) {
    let size = info.allocation_size(varlen_len);
    let layout = std::alloc::Layout::from_size_align(size, 1 << info.align_log2).unwrap();
    unsafe { std::alloc::dealloc(ptr, layout) }
}

#[test]
fn write_read_value_fields_lowbit() {
    type S = LowBit<3>;
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let obj = alloc_obj(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, &INFO as *const TypeInfo);

        let v0 = Value::<S>::tagged(1, 42);
        let v1 = Value::<S>::tagged(2, 99);
        write_value_field(obj, &INFO, 0, v0);
        write_value_field(obj, &INFO, 1, v1);

        let r0: Value<S> = read_value_field(obj, &INFO, 0);
        let r1: Value<S> = read_value_field(obj, &INFO, 1);
        assert_eq!(r0.to_bits(), v0.to_bits());
        assert_eq!(r1.to_bits(), v1.to_bits());

        // Verify header
        let header = core::ptr::read(obj as *const Compact);
        assert_eq!(header.type_info(), &INFO as *const TypeInfo);

        dealloc_obj(obj, &INFO, 0);
    }
}

#[test]
fn write_read_value_fields_nanbox() {
    type S = NanBox;
    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_fields(3);

    let obj = alloc_obj(&INFO, 0);
    unsafe {
        init_header::<Full>(obj, &INFO as *const TypeInfo);

        let v0 = Value::<S>::tagged(0, 0xCAFE);
        let v1 = Value::<S>::tagged(1, 123);
        let v2 = Value::<S>::float(3.14);
        write_value_field(obj, &INFO, 0, v0);
        write_value_field(obj, &INFO, 1, v1);
        write_value_field(obj, &INFO, 2, v2);

        let r0: Value<S> = read_value_field(obj, &INFO, 0);
        let r1: Value<S> = read_value_field(obj, &INFO, 1);
        let r2: Value<S> = read_value_field(obj, &INFO, 2);
        assert_eq!(r0.to_bits(), v0.to_bits());
        assert_eq!(r1.to_bits(), v1.to_bits());
        assert_eq!(r2.to_bits(), v2.to_bits());

        dealloc_obj(obj, &INFO, 0);
    }
}

#[test]
fn raw_bytes_read_write() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(0)
        .with_raw_bytes(4);

    let obj = alloc_obj(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, &INFO as *const TypeInfo);

        let data = raw_data_mut(obj, &INFO);
        data.copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let read = read_raw_bytes(obj, &INFO);
        assert_eq!(read, &[0xDE, 0xAD, 0xBE, 0xEF]);

        dealloc_obj(obj, &INFO, 0);
    }
}

#[test]
fn varlen_values_read_write() {
    type S = LowBit<3>;
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(0);

    let n = 5usize;
    let obj = alloc_obj(&INFO, n);
    unsafe {
        init_header::<Compact>(obj, &INFO as *const TypeInfo);
        write_varlen_count(obj, &INFO, n);

        for i in 0..n {
            let v = Value::<S>::tagged(1, i as u64);
            write_varlen_value(obj, &INFO, i, v);
        }

        assert_eq!(read_varlen_count(obj, &INFO), n);
        for i in 0..n {
            let v: Value<S> = read_varlen_value(obj, &INFO, i);
            assert!(v.has_tag(1));
            assert_eq!(v.payload(), i as u64);
        }

        dealloc_obj(obj, &INFO, n);
    }
}

#[test]
fn varlen_bytes_read_write() {
    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_varlen_bytes(0);

    let data = b"hello, world!";
    let n = data.len();
    let obj = alloc_obj(&INFO, n);
    unsafe {
        init_header::<Full>(obj, &INFO as *const TypeInfo);
        write_varlen_count(obj, &INFO, n);

        // Write bytes manually
        let base = INFO.varlen_count_offset() + 8;
        core::ptr::copy_nonoverlapping(data.as_ptr(), obj.add(base), n);

        let bytes = read_varlen_bytes(obj, &INFO);
        assert_eq!(bytes, data);

        dealloc_obj(obj, &INFO, n);
    }
}

// ─── GC scanning ────────────────────────────────────────────────────

#[test]
fn scan_fixed_fields() {
    type S = LowBit<3>;
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(3);

    let obj = alloc_obj(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, &INFO as *const TypeInfo);

        for i in 0..3u16 {
            let v = Value::<S>::tagged(0, (i as u64 + 1) * 100);
            write_value_field(obj, &INFO, i, v);
        }

        let mut slots = Vec::new();
        scan_object(obj, &INFO, |slot| {
            slots.push(core::ptr::read(slot));
        });

        assert_eq!(slots.len(), 3);
        for (i, &bits) in slots.iter().enumerate() {
            let v = Value::<S>::from_bits(bits);
            assert!(v.has_tag(0));
            assert_eq!(v.payload(), (i as u64 + 1) * 100);
        }

        dealloc_obj(obj, &INFO, 0);
    }
}

#[test]
fn scan_varlen_values() {
    type S = LowBit<3>;
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(1);

    let varlen_n = 4usize;
    let obj = alloc_obj(&INFO, varlen_n);
    unsafe {
        init_header::<Compact>(obj, &INFO as *const TypeInfo);

        // Write the fixed field
        let fixed_val = Value::<S>::tagged(2, 999);
        write_value_field(obj, &INFO, 0, fixed_val);

        // Write varlen
        write_varlen_count(obj, &INFO, varlen_n);
        for i in 0..varlen_n {
            let v = Value::<S>::tagged(1, i as u64);
            write_varlen_value(obj, &INFO, i, v);
        }

        let mut slots = Vec::new();
        scan_object(obj, &INFO, |slot| {
            slots.push(core::ptr::read(slot));
        });

        // 1 fixed + 4 varlen = 5 total slots
        assert_eq!(slots.len(), 5);

        // First slot is the fixed field
        let v = Value::<S>::from_bits(slots[0]);
        assert!(v.has_tag(2));
        assert_eq!(v.payload(), 999);

        // Remaining are varlen
        for i in 0..varlen_n {
            let v = Value::<S>::from_bits(slots[1 + i]);
            assert!(v.has_tag(1));
            assert_eq!(v.payload(), i as u64);
        }

        dealloc_obj(obj, &INFO, varlen_n);
    }
}

#[test]
fn scan_bytes_object_skips_bytes() {
    // A string-like object: fixed field + varlen bytes
    // Scan should only visit the fixed field, NOT the bytes
    type S = LowBit<3>;
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_bytes(1);

    let varlen_n = 10usize;
    let obj = alloc_obj(&INFO, varlen_n);
    unsafe {
        init_header::<Compact>(obj, &INFO as *const TypeInfo);

        let fixed_val = Value::<S>::tagged(3, 42);
        write_value_field(obj, &INFO, 0, fixed_val);
        write_varlen_count(obj, &INFO, varlen_n);

        let mut count = 0usize;
        scan_object(obj, &INFO, |_slot| {
            count += 1;
        });

        // Only the 1 fixed value field, bytes are not GC-traced
        assert_eq!(count, 1);

        dealloc_obj(obj, &INFO, varlen_n);
    }
}

#[test]
fn scan_no_fields() {
    // Object with no traceable fields at all (pure raw bytes)
    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE)
        .with_fields(0)
        .with_raw_bytes(16);

    let obj = alloc_obj(&INFO, 0);
    unsafe {
        init_header::<Full>(obj, &INFO as *const TypeInfo);

        let mut count = 0usize;
        scan_object(obj, &INFO, |_slot| {
            count += 1;
        });

        assert_eq!(count, 0);

        dealloc_obj(obj, &INFO, 0);
    }
}

// ─── Scan can mutate slots (for GC forwarding) ─────────────────────

#[test]
fn scan_can_update_slots() {
    type S = LowBit<3>;
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

    let obj = alloc_obj(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, &INFO as *const TypeInfo);
        write_value_field(obj, &INFO, 0, Value::<S>::tagged(0, 100));
        write_value_field(obj, &INFO, 1, Value::<S>::tagged(0, 200));

        // Simulate GC forwarding: double every payload
        scan_object(obj, &INFO, |slot| {
            let bits = core::ptr::read(slot);
            let v = Value::<S>::from_bits(bits);
            let new_v = Value::<S>::tagged(0, v.payload() * 2);
            core::ptr::write(slot, new_v.to_bits());
        });

        let r0: Value<S> = read_value_field(obj, &INFO, 0);
        let r1: Value<S> = read_value_field(obj, &INFO, 1);
        assert_eq!(r0.payload(), 200);
        assert_eq!(r1.payload(), 400);

        dealloc_obj(obj, &INFO, 0);
    }
}

// ─── Shadow Frame Chain ─────────────────────────────────────────────

#[test]
fn frame_chain_empty() {
    let chain = crate::roots::FrameChain::new();
    assert_eq!(chain.depth(), 0);

    let mut slots = vec![];
    chain.scan_roots(&mut |slot| {
        slots.push(unsafe { *slot });
    });
    assert!(slots.is_empty());
}

#[test]
fn frame_chain_push_pop() {
    let chain = crate::roots::FrameChain::new();
    let frame = crate::roots::RootFrame::<3>::new();
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
    // guard dropped → frame popped
    assert_eq!(chain.depth(), 0);
}

#[test]
fn frame_chain_nested_frames() {
    let chain = crate::roots::FrameChain::new();
    let outer = crate::roots::RootFrame::<2>::new();
    let inner = crate::roots::RootFrame::<1>::new();

    outer.slots[0].set(100);
    outer.slots[1].set(200);

    let _guard_outer = chain.push(&outer);
    assert_eq!(chain.depth(), 1);

    inner.slots[0].set(300);
    {
        let _guard_inner = chain.push(&inner);
        assert_eq!(chain.depth(), 2);

        // scan should visit inner first (top of chain), then outer
        let mut slots = vec![];
        chain.scan_roots(&mut |slot| {
            slots.push(unsafe { *slot });
        });
        assert_eq!(slots, vec![300, 100, 200]);
    }
    // inner popped
    assert_eq!(chain.depth(), 1);

    let mut slots = vec![];
    chain.scan_roots(&mut |slot| {
        slots.push(unsafe { *slot });
    });
    assert_eq!(slots, vec![100, 200]);
}

#[test]
fn frame_chain_gc_updates_slots() {
    // Simulate GC updating root slots in-place
    let chain = crate::roots::FrameChain::new();
    let frame = crate::roots::RootFrame::<2>::new();
    frame.slots[0].set(0x1000);
    frame.slots[1].set(0x2000);

    let _guard = chain.push(&frame);

    // "GC" doubles all values
    chain.scan_roots(&mut |slot| unsafe {
        let old = *slot;
        *slot = old * 2;
    });

    assert_eq!(frame.slots[0].get(), 0x2000);
    assert_eq!(frame.slots[1].get(), 0x4000);
}

// ─── RootSet ────────────────────────────────────────────────────────

#[test]
fn root_set_basic() {
    let mut rs = crate::roots::RootSet::new();
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
    let mut rs = crate::roots::RootSet::new();
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
    let mut rs = crate::roots::RootSet::new();
    rs.add(100);
    rs.add(200);

    // "GC" adds 1 to all values
    rs.scan_roots(&mut |slot| unsafe {
        *slot += 1;
    });

    assert_eq!(rs.get(0), 101);
    assert_eq!(rs.get(1), 201);
}

// ─── Alignment ──────────────────────────────────────────────────────

#[test]
fn custom_alignment() {
    const INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(1)
        .with_align_log2(4); // 16-byte aligned

    // header(8) + field(8) = 16, already 16-byte aligned
    assert_eq!(INFO.allocation_size(0), 16);
}

#[test]
fn allocation_size_respects_alignment() {
    // 3 bytes raw, 16-byte alignment
    const INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE)
        .with_fields(0)
        .with_raw_bytes(3)
        .with_align_log2(4);

    // header(8) + raw(3) = 11, align8 = 16, align to 16 = 16
    assert_eq!(INFO.allocation_size(0), 16);
}

// ─── Bitfield header (define_header! macro) ─────────────────────────

define_header! {
    pub GcHeader {
        type_info: *const TypeInfo,
        gc_bits: u64 {
            mark:       [0..2],
            pinned:     [2..3],
            generation: [3..6],
            forwarding: [6..64],
        }
    }
}

#[test]
fn gc_header_size() {
    assert_eq!(GcHeader::SIZE, 16);
    assert_eq!(core::mem::size_of::<GcHeader>(), 16);
}

#[test]
fn gc_header_type_info() {
    let info = TypeInfo::for_header(GcHeader::SIZE).with_fields(1);
    let header = GcHeader::new(&info as *const TypeInfo);
    assert_eq!(header.type_info(), &info as *const TypeInfo);
    assert_eq!(header.gc_bits(), 0);
}

#[test]
fn gc_header_bitfield_set_get() {
    let info = TypeInfo::for_header(GcHeader::SIZE);
    let mut header = GcHeader::new(&info as *const TypeInfo);

    // mark: bits [0..2] — 2 bits, max value 3
    header.set_mark(3);
    assert_eq!(header.mark(), 3);

    // pinned: bits [2..3] — 1 bit, max value 1
    header.set_pinned(1);
    assert_eq!(header.pinned(), 1);

    // generation: bits [3..6] — 3 bits, max value 7
    header.set_generation(5);
    assert_eq!(header.generation(), 5);

    // forwarding: bits [6..64] — 58 bits
    header.set_forwarding(42);
    assert_eq!(header.forwarding(), 42);
}

#[test]
fn gc_header_bitfield_independence() {
    let info = TypeInfo::for_header(GcHeader::SIZE);
    let mut header = GcHeader::new(&info as *const TypeInfo);

    // Set all sub-fields
    header.set_mark(2);
    header.set_pinned(1);
    header.set_generation(7);
    header.set_forwarding(42);

    // Verify all values
    assert_eq!(header.mark(), 2);
    assert_eq!(header.pinned(), 1);
    assert_eq!(header.generation(), 7);
    assert_eq!(header.forwarding(), 42);

    // Change one field, verify others unchanged
    header.set_mark(1);
    assert_eq!(header.mark(), 1);
    assert_eq!(header.pinned(), 1);
    assert_eq!(header.generation(), 7);
    assert_eq!(header.forwarding(), 42);

    // Change another
    header.set_forwarding(999);
    assert_eq!(header.mark(), 1);
    assert_eq!(header.pinned(), 1);
    assert_eq!(header.generation(), 7);
    assert_eq!(header.forwarding(), 999);
}

#[test]
fn gc_header_whole_word_accessor() {
    let info = TypeInfo::for_header(GcHeader::SIZE);
    let mut header = GcHeader::new(&info as *const TypeInfo);

    header.set_gc_bits(0xDEAD_BEEF_CAFE_BABE);
    assert_eq!(header.gc_bits(), 0xDEAD_BEEF_CAFE_BABE);

    // Sub-fields should reflect the raw bits
    // 0xBABE = 1011_1010_1011_1110
    // mark: bits [0..2] = 10 = 2
    assert_eq!(header.mark(), 2);
    // pinned: bit [2] = 1
    assert_eq!(header.pinned(), 1);
}

#[test]
fn gc_header_with_plain_field() {
    // Header with type_info + bitfield + plain field
    define_header! {
        pub RichHeader {
            type_info: *const TypeInfo,
            gc_bits: u64 {
                mark:   [0..2],
                gennum: [2..5],
            }
            identity_hash: u32,
        }
    }

    assert_eq!(RichHeader::SIZE, 24); // ptr(8) + u64(8) + u32(4) + padding(4)

    let info = TypeInfo::for_header(RichHeader::SIZE);
    let mut header = RichHeader::new(&info as *const TypeInfo);
    assert_eq!(header.type_info(), &info as *const TypeInfo);
    assert_eq!(header.gc_bits(), 0);
    assert_eq!(header.identity_hash(), 0);

    header.set_mark(3);
    header.set_gennum(6);
    header.set_identity_hash(0xCAFE);

    assert_eq!(header.mark(), 3);
    assert_eq!(header.gennum(), 6);
    assert_eq!(header.identity_hash(), 0xCAFE);
}

// ─── TYPE_INFO_OFFSET ────────────────────────────────────────────────

#[test]
fn compact_type_info_offset() {
    assert_eq!(Compact::TYPE_INFO_OFFSET, 0);
}

#[test]
fn full_type_info_offset() {
    // Full has gc_word (u64) first, then type_info
    assert_eq!(Full::TYPE_INFO_OFFSET, 8);
}

#[test]
fn read_type_info_compact() {
    static INFO: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);
    let obj = alloc_obj(&INFO, 0);
    unsafe {
        init_header::<Compact>(obj, &INFO as *const TypeInfo);
        let recovered = read_type_info(obj, Compact::TYPE_INFO_OFFSET);
        assert_eq!(recovered as *const TypeInfo, &INFO as *const TypeInfo);
        dealloc_obj(obj, &INFO, 0);
    }
}

#[test]
fn read_type_info_full() {
    static INFO: TypeInfo = TypeInfo::for_header(Full::SIZE).with_fields(1);
    let obj = alloc_obj(&INFO, 0);
    unsafe {
        init_header::<Full>(obj, &INFO as *const TypeInfo);
        let recovered = read_type_info(obj, Full::TYPE_INFO_OFFSET);
        assert_eq!(recovered as *const TypeInfo, &INFO as *const TypeInfo);
        dealloc_obj(obj, &INFO, 0);
    }
}
