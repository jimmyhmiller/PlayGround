use std::sync::atomic::{AtomicU64, Ordering};

use dynvalue::{TagScheme, Value};

use crate::header::ObjHeader;
use crate::type_info::{TypeInfo, VarLenKind};

/// Read the TypeInfo pointer from a heap object, given the byte offset
/// of the `type_info` field within the header.
///
/// This is used by the heap walker to recover TypeInfo from objects
/// without knowing the concrete header type at compile time.
///
/// # Safety
/// - `obj` must point to a valid heap object.
/// - `type_info_offset` must be the correct offset of the `type_info` field
///   (i.e. `H::TYPE_INFO_OFFSET` for the header type used).
#[inline(always)]
pub unsafe fn read_type_info(obj: *const u8, type_info_offset: usize) -> &'static TypeInfo {
    unsafe {
        let ptr = core::ptr::read(obj.add(type_info_offset) as *const *const TypeInfo);
        &*ptr
    }
}

/// Write the object header into a freshly allocated object.
///
/// # Safety
/// `obj` must point to a valid allocation of at least `H::SIZE` bytes.
/// The memory must not be concurrently accessed.
#[inline(always)]
pub unsafe fn init_header<H: ObjHeader>(obj: *mut u8, type_info: *const TypeInfo) {
    let header = H::new(type_info);
    unsafe {
        core::ptr::write(obj as *mut H, header);
    }
}

/// Read a GC-traced Value field by index.
///
/// Uses a relaxed atomic load to avoid data races with the concurrent GC,
/// which may read fields via `atomic_copy_words` or `process_slot_concurrent`.
/// On x86-64 and ARM64, relaxed atomic loads of aligned u64 compile to
/// plain load instructions — zero overhead.
///
/// # Safety
/// - `obj` must point to a valid heap object described by `info`.
/// - `index` must be less than `info.value_field_count`.
#[inline(always)]
pub unsafe fn read_value_field<S: TagScheme>(
    obj: *const u8,
    info: &TypeInfo,
    index: u16,
) -> Value<S> {
    debug_assert!(index < info.value_field_count);
    let offset = info.value_field_offset(index);
    unsafe {
        let atomic = &*(obj.add(offset) as *const AtomicU64);
        Value::from_bits(atomic.load(Ordering::Relaxed))
    }
}

/// Write a GC-traced Value field by index.
///
/// Uses a relaxed atomic store to avoid data races with the concurrent GC,
/// which may read fields via `atomic_copy_words` or `process_slot_concurrent`.
/// On x86-64 and ARM64, relaxed atomic stores of aligned u64 compile to
/// plain store instructions — zero overhead.
///
/// # Safety
/// - `obj` must point to a valid heap object described by `info`.
/// - `index` must be less than `info.value_field_count`.
#[inline(always)]
pub unsafe fn write_value_field<S: TagScheme>(
    obj: *mut u8,
    info: &TypeInfo,
    index: u16,
    value: Value<S>,
) {
    debug_assert!(index < info.value_field_count);
    let offset = info.value_field_offset(index);
    unsafe {
        let atomic = &*(obj.add(offset) as *const AtomicU64);
        atomic.store(value.to_bits(), Ordering::Relaxed);
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

/// Read a varlen Value element by index.
///
/// Uses a relaxed atomic load for concurrent GC safety (see `read_value_field`).
///
/// # Safety
/// - `obj` must point to a valid heap object with `info.varlen == VarLenKind::Values`.
/// - `index` must be less than the varlen count.
#[inline(always)]
pub unsafe fn read_varlen_value<S: TagScheme>(
    obj: *const u8,
    info: &TypeInfo,
    index: usize,
) -> Value<S> {
    debug_assert!(info.varlen == VarLenKind::Values);
    let offset = info.varlen_element_offset(index);
    unsafe {
        let atomic = &*(obj.add(offset) as *const AtomicU64);
        Value::from_bits(atomic.load(Ordering::Relaxed))
    }
}

/// Write a varlen Value element by index.
///
/// Uses a relaxed atomic store for concurrent GC safety (see `write_value_field`).
///
/// # Safety
/// - `obj` must point to a valid heap object with `info.varlen == VarLenKind::Values`.
/// - `index` must be less than the varlen count.
#[inline(always)]
pub unsafe fn write_varlen_value<S: TagScheme>(
    obj: *mut u8,
    info: &TypeInfo,
    index: usize,
    value: Value<S>,
) {
    debug_assert!(info.varlen == VarLenKind::Values);
    let offset = info.varlen_element_offset(index);
    unsafe {
        let atomic = &*(obj.add(offset) as *const AtomicU64);
        atomic.store(value.to_bits(), Ordering::Relaxed);
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
