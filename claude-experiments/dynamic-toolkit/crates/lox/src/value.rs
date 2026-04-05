use dynvalue::{NanBox, TagScheme};

use crate::object::{Obj, ObjType};

const TAG_OBJ: u32 = 0;
const TAG_NIL: u32 = 1;
const TAG_BOOL: u32 = 2;

pub type Value = u64;

pub const NIL_VAL: Value = 0x7FFC_0000_0000_0000 | (1u64 << 48); // TAG_NIL, payload 0
pub const TRUE_VAL: Value = 0x7FFC_0000_0000_0000 | (2u64 << 48) | 1; // TAG_BOOL, payload 1
pub const FALSE_VAL: Value = 0x7FFC_0000_0000_0000 | (2u64 << 48); // TAG_BOOL, payload 0

#[inline(always)]
pub fn nil_val() -> Value {
    NIL_VAL
}

#[inline(always)]
pub fn bool_val(b: bool) -> Value {
    if b { TRUE_VAL } else { FALSE_VAL }
}

#[inline(always)]
pub fn number_val(n: f64) -> Value {
    NanBox::encode_float(n)
}

#[inline(always)]
pub fn obj_val(ptr: *mut Obj) -> Value {
    NanBox::encode_tagged(TAG_OBJ, ptr as u64 & 0x0000_FFFF_FFFF_FFFF)
}

#[inline(always)]
pub fn is_nil(v: Value) -> bool {
    NanBox::has_tag(v, TAG_NIL)
}

#[inline(always)]
pub fn is_bool(v: Value) -> bool {
    NanBox::has_tag(v, TAG_BOOL)
}

#[inline(always)]
pub fn is_number(v: Value) -> bool {
    NanBox::is_float(v)
}

#[inline(always)]
pub fn is_obj(v: Value) -> bool {
    NanBox::has_tag(v, TAG_OBJ)
}

#[inline(always)]
pub fn as_bool(v: Value) -> bool {
    NanBox::extract_payload(v) != 0
}

#[inline(always)]
pub fn as_number(v: Value) -> f64 {
    f64::from_bits(v)
}

#[inline(always)]
pub fn as_obj(v: Value) -> *mut Obj {
    NanBox::extract_payload(v) as *mut Obj
}

#[inline(always)]
pub fn is_string(v: Value) -> bool {
    is_obj_type(v, ObjType::String)
}

#[inline(always)]
pub fn is_function(v: Value) -> bool {
    is_obj_type(v, ObjType::Function)
}

#[inline(always)]
pub fn is_closure(v: Value) -> bool {
    is_obj_type(v, ObjType::Closure)
}

#[inline(always)]
pub fn is_class(v: Value) -> bool {
    is_obj_type(v, ObjType::Class)
}

#[inline(always)]
pub fn is_instance(v: Value) -> bool {
    is_obj_type(v, ObjType::Instance)
}

#[inline(always)]
fn is_obj_type(v: Value, t: ObjType) -> bool {
    is_obj(v) && unsafe { (*as_obj(v)).obj_type == t }
}

#[inline(always)]
pub fn is_falsey(v: Value) -> bool {
    is_nil(v) || (is_bool(v) && !as_bool(v))
}

pub fn values_equal(a: Value, b: Value) -> bool {
    if is_number(a) && is_number(b) {
        return as_number(a) == as_number(b);
    }
    a == b
}

pub fn print_value(v: Value) {
    if is_nil(v) {
        print!("nil");
    } else if is_bool(v) {
        print!("{}", if as_bool(v) { "true" } else { "false" });
    } else if is_number(v) {
        let n = as_number(v);
        // Match clox formatting
        print!("{}", format_number(n));
    } else if is_obj(v) {
        print_object(v);
    }
}

pub fn format_number(n: f64) -> String {
    // Match C's %g format: up to 6 significant digits, no trailing zeros
    let s = format!("{:.6e}", n);
    // Parse the mantissa and exponent
    let parts: Vec<&str> = s.split('e').collect();
    let mantissa: f64 = parts[0].parse().unwrap();
    let exp: i32 = parts[1].parse().unwrap();

    if exp >= -4 && exp < 6 {
        // Use fixed-point notation
        let decimals = if exp >= 0 { (5 - exp).max(0) as usize } else { (6 + (-exp - 1).min(5)) as usize };
        let formatted = format!("{:.*}", decimals, n);
        // Trim trailing zeros after decimal point
        if formatted.contains('.') {
            formatted.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            formatted
        }
    } else {
        // Use scientific notation like %g
        let formatted = format!("{:.*e}", 5, mantissa);
        // Trim trailing zeros in mantissa
        if formatted.contains('.') {
            let parts: Vec<&str> = formatted.split('e').collect();
            let m = parts[0].trim_end_matches('0').trim_end_matches('.');
            format!("{}e+{:02}", m, exp)
        } else {
            formatted
        }
    }
}

pub fn value_to_string(v: Value) -> String {
    if is_nil(v) {
        "nil".to_string()
    } else if is_bool(v) {
        if as_bool(v) { "true".to_string() } else { "false".to_string() }
    } else if is_number(v) {
        format_number(as_number(v))
    } else if is_obj(v) {
        object_to_string(v)
    } else {
        "unknown".to_string()
    }
}

fn print_object(v: Value) {
    print!("{}", object_to_string(v));
}

fn object_to_string(v: Value) -> String {
    use crate::object::*;
    let obj = as_obj(v);
    unsafe {
        match (*obj).obj_type {
            ObjType::String => {
                let s = &*(obj as *mut ObjString);
                s.chars.clone()
            }
            ObjType::Native => "<native fn>".to_string(),
            ObjType::Class => {
                let c = &*(obj as *mut ObjClass);
                (*c.name).chars.clone()
            }
            ObjType::Instance => {
                let i = &*(obj as *mut ObjInstance);
                format!("{} instance", (*(*i.class).name).chars)
            }
            _ => format!("<obj {:?}>", (*obj).obj_type),
        }
    }
}
