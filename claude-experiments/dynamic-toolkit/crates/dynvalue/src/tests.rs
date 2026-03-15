use crate::*;

// ─── Low-bit tagging ────────────────────────────────────────────────

type Val3 = Value<LowBit<3>>;

#[test]
fn low_bit_roundtrip() {
    for tag in 0..8u32 {
        let v = Val3::tagged(tag, 12345);
        assert_eq!(
            v.decode(),
            Decoded::Tagged {
                tag,
                payload: 12345
            }
        );
        assert!(v.has_tag(tag));
        assert_eq!(v.payload(), 12345);
    }
}

#[test]
fn low_bit_zero_payload() {
    let v = Val3::tagged(5, 0);
    assert!(v.has_tag(5));
    assert_eq!(v.payload(), 0);
}

#[test]
fn low_bit_max_payload() {
    // 61 payload bits
    let max = (1u64 << 61) - 1;
    let v = Val3::tagged(0, max);
    assert_eq!(v.payload(), max);
}

#[test]
fn low_bit_pointer_tag_zero() {
    // Common pattern: tag 0 = pointer, so aligned pointers need no masking
    let ptr = 0xDEAD_BEE0u64; // 16-byte aligned (low 4 bits = 0)
    // With 3 tag bits, pointer must be 8-byte aligned (low 3 bits = 0)
    let aligned_ptr = ptr & !0x7;
    let v = Val3::tagged(0, aligned_ptr >> 3);
    assert!(v.has_tag(0));
    // Reconstruct the pointer
    assert_eq!(v.payload() << 3, aligned_ptr);
}

#[test]
fn low_bit_is_never_float() {
    let v = Val3::tagged(1, 42);
    assert!(!v.is_float());
}

#[test]
fn low_bit_different_widths() {
    // 1-bit tag: 2 tags, 63-bit payload
    type Val1 = Value<LowBit<1>>;
    let v = Val1::tagged(0, 100);
    assert!(v.has_tag(0));
    assert_eq!(v.payload(), 100);
    let v = Val1::tagged(1, 100);
    assert!(v.has_tag(1));

    // 4-bit tag: 16 tags, 60-bit payload
    type Val4 = Value<LowBit<4>>;
    for tag in 0..16u32 {
        let v = Val4::tagged(tag, 999);
        assert!(v.has_tag(tag));
        assert_eq!(v.payload(), 999);
    }
}

// ─── NaN-boxing ─────────────────────────────────────────────────────

type NVal = Value<NanBox>;

#[test]
fn nan_box_tagged_roundtrip() {
    for tag in 0..4u32 {
        let v = NVal::tagged(tag, 0xBEEF);
        assert_eq!(
            v.decode(),
            Decoded::Tagged {
                tag,
                payload: 0xBEEF
            }
        );
        assert!(v.has_tag(tag));
        assert_eq!(v.payload(), 0xBEEF);
        assert!(!v.is_float());
    }
}

#[test]
fn nan_box_float_roundtrip() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.14159,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for &f in &values {
        let v = NVal::float(f);
        assert!(v.is_float());
        assert_eq!(v.as_f64().to_bits(), f.to_bits());
    }
}

#[test]
fn nan_box_nan_roundtrip() {
    // Standard NaN should survive encoding
    let v = NVal::float(f64::NAN);
    assert!(v.is_float());
    assert!(v.as_f64().is_nan());
}

#[test]
fn nan_box_float_not_tagged() {
    let v = NVal::float(42.0);
    assert!(v.is_float());
    for tag in 0..4 {
        assert!(!v.has_tag(tag));
    }
}

#[test]
fn nan_box_tagged_not_float() {
    let v = NVal::tagged(0, 42);
    assert!(!v.is_float());
}

#[test]
fn nan_box_pointer_in_payload() {
    // 48-bit pointers fit in the payload
    let ptr: u64 = 0x0000_7FFF_DEAD_BEE0;
    let v = NVal::tagged(0, ptr);
    assert_eq!(v.payload(), ptr);
    assert_eq!(v.as_ptr::<u8>(), ptr as *const u8);
}

#[test]
fn nan_box_max_payload() {
    let max = 0x0000_FFFF_FFFF_FFFF_u64; // 48 bits
    let v = NVal::tagged(0, max);
    assert_eq!(v.payload(), max);
}

#[test]
fn nan_box_tag_discrimination() {
    let v0 = NVal::tagged(0, 100);
    let v1 = NVal::tagged(1, 100);
    let v2 = NVal::tagged(2, 100);
    let v3 = NVal::tagged(3, 100);

    assert!(v0.has_tag(0) && !v0.has_tag(1));
    assert!(v1.has_tag(1) && !v1.has_tag(0));
    assert!(v2.has_tag(2) && !v2.has_tag(3));
    assert!(v3.has_tag(3) && !v3.has_tag(2));
}

// ─── Generic code over TagScheme ────────────────────────────────────

fn count_tagged_with_tag<S: TagScheme>(values: &[Value<S>], tag: u32) -> usize {
    values.iter().filter(|v| v.has_tag(tag)).count()
}

#[test]
fn generic_over_scheme() {
    // Same function works with both schemes
    let low_vals = vec![
        Val3::tagged(0, 10),
        Val3::tagged(1, 20),
        Val3::tagged(0, 30),
    ];
    assert_eq!(count_tagged_with_tag(&low_vals, 0), 2);

    let nan_vals = vec![NVal::tagged(0, 10), NVal::tagged(1, 20), NVal::float(3.14)];
    assert_eq!(count_tagged_with_tag(&nan_vals, 0), 1);
    assert_eq!(count_tagged_with_tag(&nan_vals, 1), 1);
}

// ─── Value is just u64 ─────────────────────────────────────────────

#[test]
fn value_is_u64_sized() {
    assert_eq!(std::mem::size_of::<Val3>(), 8);
    assert_eq!(std::mem::size_of::<NVal>(), 8);
    assert_eq!(std::mem::align_of::<Val3>(), 8);
    assert_eq!(std::mem::align_of::<NVal>(), 8);
}

#[test]
fn from_bits_roundtrip() {
    let bits = 0xDEAD_BEEF_CAFE_BABEu64;
    let v = Val3::from_bits(bits);
    assert_eq!(v.to_bits(), bits);
}

// ─── define_value! macro ────────────────────────────────────────────

define_value! {
    pub SchemeVal: LowBit<3> {
        #[heap] Ptr(0): *mut u8,
        Fixnum(1): i64,
        Bool(2): bool,
        Nil(3),
        Symbol(4): u32,
    }
}

#[test]
fn macro_constructors() {
    let n = SchemeVal::fixnum(42);
    assert!(n.is_fixnum());
    assert_eq!(n.as_fixnum(), 42);

    let b = SchemeVal::bool(true);
    assert!(b.is_bool());
    assert_eq!(b.as_bool(), true);

    let nil = SchemeVal::nil();
    assert!(nil.is_nil());

    let sym = SchemeVal::symbol(99);
    assert!(sym.is_symbol());
    assert_eq!(sym.as_symbol(), 99);
}

#[test]
fn macro_pointer() {
    let mut x: u64 = 0xCAFE;
    let p = &mut x as *mut u64 as *mut u8;
    let v = SchemeVal::ptr(p);
    assert!(v.is_ptr());
    assert_eq!(v.as_ptr(), p);
}

#[test]
fn macro_kind_matching() {
    let v = SchemeVal::fixnum(-7);
    match v.kind() {
        SchemeValKind::Fixnum(n) => assert_eq!(n, -7),
        other => panic!("expected Fixnum, got {:?}", other),
    }

    let v = SchemeVal::nil();
    match v.kind() {
        SchemeValKind::Nil => {}
        other => panic!("expected Nil, got {:?}", other),
    }
}

#[test]
fn macro_negative_fixnum() {
    // Sign extension should work
    let v = SchemeVal::fixnum(-1);
    assert_eq!(v.as_fixnum(), -1);

    let v = SchemeVal::fixnum(-12345);
    assert_eq!(v.as_fixnum(), -12345);

    let v = SchemeVal::fixnum(i64::MIN >> 3); // fits in 61 bits
    assert_eq!(v.as_fixnum(), i64::MIN >> 3);
}

#[test]
fn macro_heap_ptr_tracking() {
    use crate::TaggedValue;

    let v = SchemeVal::ptr(std::ptr::null_mut());
    assert!(v.is_heap_ptr());

    let v = SchemeVal::fixnum(42);
    assert!(!v.is_heap_ptr());

    let v = SchemeVal::nil();
    assert!(!v.is_heap_ptr());
}

#[test]
fn macro_debug_format() {
    let v = SchemeVal::fixnum(42);
    let s = format!("{:?}", v);
    assert!(s.contains("Fixnum"), "debug was: {s}");

    let v = SchemeVal::nil();
    let s = format!("{:?}", v);
    assert!(s.contains("Nil"), "debug was: {s}");
}

#[test]
fn macro_value_roundtrip() {
    let v = SchemeVal::fixnum(42);
    let raw = v.value();
    let v2 = SchemeVal::from_value(raw);
    assert_eq!(v.to_bits(), v2.to_bits());
}

#[test]
fn macro_is_u64_sized() {
    assert_eq!(std::mem::size_of::<SchemeVal>(), 8);
}

// ─── define_value! with NanBox ──────────────────────────────────────

define_value! {
    pub JSVal: NanBox {
        #[heap] Ptr(0): *mut u8,
        Int(1): i32,
        Bool(2): bool,
        Null(3),
    }
}

#[test]
fn nanbox_macro_constructors() {
    let v = JSVal::int(42);
    assert!(v.is_int());
    assert_eq!(v.as_int(), 42);

    let v = JSVal::bool(false);
    assert!(v.is_bool());
    assert_eq!(v.as_bool(), false);

    let v = JSVal::null();
    assert!(v.is_null());
}

#[test]
fn nanbox_macro_kind() {
    let v = JSVal::int(-5);
    match v.kind() {
        JSValKind::Int(n) => assert_eq!(n, -5),
        other => panic!("expected Int, got {:?}", other),
    }

    let v = JSVal::null();
    match v.kind() {
        JSValKind::Null => {}
        other => panic!("expected Null, got {:?}", other),
    }
}

#[test]
fn nanbox_macro_heap_tracking() {
    use crate::TaggedValue;

    let v = JSVal::ptr(std::ptr::null_mut());
    assert!(v.is_heap_ptr());

    let v = JSVal::int(1);
    assert!(!v.is_heap_ptr());
}
