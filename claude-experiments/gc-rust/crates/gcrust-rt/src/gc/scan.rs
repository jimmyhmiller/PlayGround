use crate::gc::type_info::{TypeInfo, VarLenKind};

/// Scan a heap object, calling `visitor` for each GC-traceable slot.
///
/// The visitor receives a `*mut u64` pointing to each slot that contains
/// a tagged Value. This is scheme-agnostic — the GC works with raw `u64`
/// values and uses the tag scheme to determine which ones are heap pointers.
///
/// Visits:
/// 1. All fixed Value fields (indices 0..value_field_count)
/// 2. All interior pointers — GC refs embedded in flattened `#[value]` fields in
///    the raw region (`info.interior_ptrs`, absolute byte offsets)
/// 3. All varlen Value elements (if varlen == VarLenKind::Values)
///
/// # Safety
/// - `obj` must point to a valid heap object described by `info`.
/// - The varlen count must be initialized if `info.varlen != None`.
/// - The visitor must not invalidate the object pointer.
pub unsafe fn scan_object(obj: *mut u8, info: &TypeInfo, mut visitor: impl FnMut(*mut u64)) {
    // Visit fixed value fields
    let field_base = info.header_size as usize;
    for i in 0..info.value_field_count as usize {
        let slot = unsafe { obj.add(field_base + i * 8) as *mut u64 };
        visitor(slot);
    }

    // Visit interior pointers (refs embedded in flattened value fields).
    for &off in info.interior_ptrs {
        let slot = unsafe { obj.add(off as usize) as *mut u64 };
        visitor(slot);
    }

    // Visit varlen value elements
    if info.varlen == VarLenKind::Values {
        let count =
            unsafe { core::ptr::read(obj.add(info.varlen_count_offset()) as *const u64) as usize };
        let element_base = info.varlen_count_offset() + 8;
        for i in 0..count {
            let slot = unsafe { obj.add(element_base + i * 8) as *mut u64 };
            visitor(slot);
        }
    }
}
