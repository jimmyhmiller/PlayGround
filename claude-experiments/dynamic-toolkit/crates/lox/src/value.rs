//! NanBox value helpers for the Lox runtime.
//!
//! Tag layout (matching dynlang NanBoxTags):
//!   Tag 0 = object (payload = raw GC pointer)
//!   Tag 1 = nil
//!   Tag 2 = bool (payload 0=false, 1=true)
//!   Untagged = IEEE 754 float (number)

use dynvalue::{NanBox, TagScheme};

pub const TAG_OBJ: u32 = 0;
pub const TAG_NIL: u32 = 1;
pub const TAG_BOOL: u32 = 2;

const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
pub const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

// ── NanBox encoding ───────────────────────────────────────────────

pub fn nil_val() -> u64 {
    NanBox::encode_tagged(TAG_NIL, 0)
}

pub fn bool_val(b: bool) -> u64 {
    NanBox::encode_tagged(TAG_BOOL, b as u64)
}

pub fn number_val(n: f64) -> u64 {
    NanBox::encode_float(n)
}

/// Encode a raw GC pointer as a NanBox object value.
pub fn obj_val(ptr: *mut u8) -> u64 {
    NanBox::encode_tagged(TAG_OBJ, ptr as u64)
}

/// Extract the raw GC pointer from a NanBox object value.
pub fn obj_ptr(v: u64) -> *mut u8 {
    (v & PAYLOAD_MASK) as *mut u8
}

pub fn is_number(v: u64) -> bool {
    (v & FULL_MASK) != TAG_PATTERN
}

pub fn is_nil(v: u64) -> bool {
    NanBox::has_tag(v, TAG_NIL)
}

pub fn is_bool(v: u64) -> bool {
    NanBox::has_tag(v, TAG_BOOL)
}

pub fn is_obj(v: u64) -> bool {
    NanBox::has_tag(v, TAG_OBJ)
}

pub fn as_number(v: u64) -> f64 {
    f64::from_bits(v)
}

pub fn as_bool(v: u64) -> bool {
    NanBox::extract_payload(v) != 0
}

pub fn is_falsey(v: u64) -> bool {
    is_nil(v) || (is_bool(v) && !as_bool(v))
}

pub fn values_equal(a: u64, b: u64) -> bool {
    if is_number(a) && is_number(b) {
        return as_number(a) == as_number(b);
    }
    a == b
}

pub fn format_number(n: f64) -> String {
    if n == 0.0 && n.is_sign_negative() {
        "-0".to_string()
    } else if n == n.trunc() && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

// ── GC object field access helpers ───────────────────────────────

/// Read a u64 field from a GC object at the given byte offset.
/// `ptr` is the raw object pointer (NOT NanBox-tagged).
#[inline]
pub unsafe fn gc_read_field(ptr: *mut u8, offset: i32) -> u64 {
    unsafe {
        let field_ptr = ptr.offset(offset as isize) as *const u64;
        field_ptr.read()
    }
}

/// Write a u64 field to a GC object at the given byte offset.
#[inline]
pub unsafe fn gc_write_field(ptr: *mut u8, offset: i32, val: u64) {
    unsafe {
        let field_ptr = ptr.offset(offset as isize) as *mut u64;
        field_ptr.write(val);
    }
}

/// Read a varlen element at the given index from a GC object.
/// `base_offset` is the byte offset of the first varlen element.
#[inline]
pub unsafe fn gc_read_elem(ptr: *mut u8, base_offset: i32, index: usize) -> u64 {
    unsafe {
        let elem_ptr = ptr.offset(base_offset as isize + (index * 8) as isize) as *const u64;
        elem_ptr.read()
    }
}

/// Write a varlen element at the given index to a GC object.
#[inline]
pub unsafe fn gc_write_elem(ptr: *mut u8, base_offset: i32, index: usize, val: u64) {
    unsafe {
        let elem_ptr = ptr.offset(base_offset as isize + (index * 8) as isize) as *mut u64;
        elem_ptr.write(val);
    }
}

/// Read the varlen byte data from a GC string object.
/// Returns a slice of bytes.
#[inline]
pub unsafe fn gc_read_bytes(ptr: *mut u8, base_offset: i32, len: usize) -> &'static [u8] {
    unsafe {
        let data_ptr = ptr.offset(base_offset as isize);
        std::slice::from_raw_parts(data_ptr, len)
    }
}

/// Write bytes to the varlen section of a GC string object.
#[inline]
pub unsafe fn gc_write_bytes(ptr: *mut u8, base_offset: i32, data: &[u8]) {
    unsafe {
        let dest = ptr.offset(base_offset as isize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dest, data.len());
    }
}
