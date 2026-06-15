use crate::gc::header::ObjHeader;
use crate::gc::type_info::{TypeInfo, VarLenKind};

/// Read the type_id from a heap object header, given the byte offset
/// of the `type_id` field within the header.
///
/// # Safety
/// - `obj` must point to a valid heap object.
/// - `type_id_offset` must be the correct offset of the `type_id` field
///   (i.e. `H::TYPE_ID_OFFSET` for the header type used).
#[inline(always)]
pub unsafe fn read_type_id(obj: *const u8, type_id_offset: usize) -> u16 {
    unsafe { core::ptr::read(obj.add(type_id_offset) as *const u16) }
}

/// Look up TypeInfo from a type_id using the runtime's type table.
/// Kept for GC scanning which still needs layout info.
#[inline(always)]
pub fn lookup_type_info<'a>(type_id: u16, type_table: &'a [&'static TypeInfo]) -> &'a TypeInfo {
    type_table[type_id as usize]
}

/// Write the object header into a freshly allocated object.
///
/// # Safety
/// `obj` must point to a valid allocation of at least `H::SIZE` bytes.
/// The memory must not be concurrently accessed.
#[inline(always)]
pub unsafe fn init_header<H: ObjHeader>(obj: *mut u8, type_id: u16) {
    let header = H::new(type_id);
    unsafe {
        core::ptr::write(obj as *mut H, header);
    }
}

/// Read the raw (untraced) byte section as a slice.
///
/// # Safety
/// `obj` must point to a valid heap object described by `info`.
#[inline(always)]
pub unsafe fn read_raw_bytes<'a>(obj: *const u8, info: &TypeInfo) -> &'a [u8] {
    let offset = info.raw_data_offset();
    let len = info.raw_byte_count as usize;
    unsafe { core::slice::from_raw_parts(obj.add(offset), len) }
}

/// Get a mutable pointer to the raw (untraced) byte section.
///
/// # Safety
/// `obj` must point to a valid heap object described by `info`.
/// The caller must ensure no aliasing violations.
#[inline(always)]
pub unsafe fn raw_data_mut<'a>(obj: *mut u8, info: &TypeInfo) -> &'a mut [u8] {
    let offset = info.raw_data_offset();
    let len = info.raw_byte_count as usize;
    unsafe { core::slice::from_raw_parts_mut(obj.add(offset), len) }
}

/// Read the varlen element count.
///
/// # Safety
/// `obj` must point to a valid heap object with `info.varlen != VarLenKind::None`.
#[inline(always)]
pub unsafe fn read_varlen_count(obj: *const u8, info: &TypeInfo) -> usize {
    debug_assert!(info.varlen != VarLenKind::None);
    let offset = info.varlen_count_offset();
    unsafe { core::ptr::read(obj.add(offset) as *const u64) as usize }
}

/// Write the varlen element count.
///
/// # Safety
/// `obj` must point to a valid heap object with `info.varlen != VarLenKind::None`.
#[inline(always)]
pub unsafe fn write_varlen_count(obj: *mut u8, info: &TypeInfo, count: usize) {
    debug_assert!(info.varlen != VarLenKind::None);
    let offset = info.varlen_count_offset();
    unsafe {
        core::ptr::write(obj.add(offset) as *mut u64, count as u64);
    }
}

/// Read the varlen byte section as a slice.
///
/// # Safety
/// - `obj` must point to a valid heap object with `info.varlen == VarLenKind::Bytes`.
/// - The varlen count must have been initialized.
#[inline(always)]
pub unsafe fn read_varlen_bytes<'a>(obj: *const u8, info: &TypeInfo) -> &'a [u8] {
    debug_assert!(info.varlen == VarLenKind::Bytes);
    unsafe {
        let count = read_varlen_count(obj, info);
        let base = info.varlen_count_offset() + 8;
        core::slice::from_raw_parts(obj.add(base), count)
    }
}
