//! Low-bit tagged value representation for clojure-jvm.
//!
//! Replaces NaN-boxing (where doubles were native and everything else was a
//! NaN payload — and integers were silently coerced to f64). Low-bit tagging
//! fits Clojure far better: integers are the common case and get a real
//! 61-bit inline fixnum; doubles (rarer) are heap-boxed.
//!
//! Heap pointers are 8-byte aligned, so tag `000` = pointer with NO decode
//! (matches the GC's `LowBitPtrPolicy<3>`, which already exists). The other
//! seven tags are immediates.
//!
//! ```text
//!   bit:  63 ........................................ 3 2 1 0
//!   ptr     |            aligned heap address            |0 0 0|
//!   fixnum  |            signed 61-bit value             |0 0 1|   (arith >>3)
//!   nil      0 .............................. 0 1 0   (singleton 0b010)
//!   bool    |.............................. b|0 1 1|   (b at bit 3)
//!   char    |...... 21-bit codepoint .......|1 0 0|
//! ```
//! Doubles, strings, collections, symbols, keywords, closures, and a boxed
//! `Long` (for values outside 61-bit range) are all tag-000 heap pointers,
//! distinguished by their heap `type_id` (as today).

pub const TAG_BITS: u64 = 3;
pub const TAG_MASK: u64 = 0b111;

pub const TAG_PTR: u64 = 0b000;
pub const TAG_FIXNUM: u64 = 0b001;
pub const TAG_NIL: u64 = 0b010;
pub const TAG_BOOL: u64 = 0b011;
pub const TAG_CHAR: u64 = 0b100;

/// The `nil` singleton.
pub const NIL: u64 = TAG_NIL;
pub const FALSE: u64 = TAG_BOOL; // bit 3 clear
pub const TRUE: u64 = TAG_BOOL | (1 << 3);

/// Range of inline fixnums: signed 61-bit. Values outside box to a `Long`.
pub const FIXNUM_MAX: i64 = (1 << 60) - 1;
pub const FIXNUM_MIN: i64 = -(1 << 60);

#[inline]
pub fn is_ptr(bits: u64) -> bool { bits & TAG_MASK == TAG_PTR && bits != 0 }

#[inline]
pub fn fits_fixnum(n: i64) -> bool { (FIXNUM_MIN..=FIXNUM_MAX).contains(&n) }

#[inline]
pub fn encode_fixnum(n: i64) -> u64 { ((n << TAG_BITS) as u64) | TAG_FIXNUM }

#[inline]
pub fn is_fixnum(bits: u64) -> bool { bits & TAG_MASK == TAG_FIXNUM }

/// Arithmetic shift recovers the sign. Precondition: `is_fixnum(bits)`.
#[inline]
pub fn decode_fixnum(bits: u64) -> i64 { (bits as i64) >> TAG_BITS }

#[inline]
pub fn is_nil(bits: u64) -> bool { bits == NIL }

#[inline]
pub fn encode_bool(b: bool) -> u64 { if b { TRUE } else { FALSE } }

#[inline]
pub fn is_bool(bits: u64) -> bool { bits & TAG_MASK == TAG_BOOL }

#[inline]
pub fn decode_bool(bits: u64) -> bool { (bits >> TAG_BITS) & 1 != 0 }

#[inline]
pub fn encode_char(cp: u32) -> u64 { ((cp as u64) << TAG_BITS) | TAG_CHAR }

#[inline]
pub fn is_char(bits: u64) -> bool { bits & TAG_MASK == TAG_CHAR }

#[inline]
pub fn decode_char(bits: u64) -> u32 { (bits >> TAG_BITS) as u32 }

/// A heap pointer is already tag-000 (8-aligned); encoding is identity.
#[inline]
pub fn encode_ptr(addr: u64) -> u64 {
    debug_assert_eq!(addr & TAG_MASK, 0, "heap pointer {addr:#x} not 8-aligned");
    addr
}

// ── Numeric tower ──────────────────────────────────────────────────
//
// Clojure's long/double promotion rules (pinned against the 1.11 oracle):
//   long  op long  → long      (overflow: Clojure's `+`/`*` THROW; we defer
//                               to a checked op and report overflow so the
//                               caller can throw — corpus never overflows)
//   long  op double → double
//   double op _     → double
// `=` is type-aware (1 and 1.0 are NOT `=`); `==` is numeric (1 == 1.0).
// Division: long/long that divides exactly → long, else Clojure yields a
// Ratio — we DON'T have rationals yet, so non-exact long/long falls to
// double (documented deviation, e.g. `(/ 7 2)` → 3.5 not 7/2).

/// A decoded number: either a 64-bit long or an IEEE double.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
    Long(i64),
    Double(f64),
}

impl Num {
    #[inline]
    pub fn as_f64(self) -> f64 {
        match self {
            Num::Long(n) => n as f64,
            Num::Double(d) => d,
        }
    }
    #[inline]
    fn both_long(a: Num, b: Num) -> Option<(i64, i64)> {
        match (a, b) {
            (Num::Long(x), Num::Long(y)) => Some((x, y)),
            _ => None,
        }
    }
}

/// Result of a checked long arithmetic op: `Long` on success, `Overflow`
/// when a long+long result doesn't fit (Clojure throws ArithmeticException).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumResult {
    Num(Num),
    Overflow,
}

pub fn num_add(a: Num, b: Num) -> NumResult {
    if let Some((x, y)) = Num::both_long(a, b) {
        match x.checked_add(y) {
            Some(v) => NumResult::Num(Num::Long(v)),
            None => NumResult::Overflow,
        }
    } else {
        NumResult::Num(Num::Double(a.as_f64() + b.as_f64()))
    }
}

pub fn num_sub(a: Num, b: Num) -> NumResult {
    if let Some((x, y)) = Num::both_long(a, b) {
        match x.checked_sub(y) {
            Some(v) => NumResult::Num(Num::Long(v)),
            None => NumResult::Overflow,
        }
    } else {
        NumResult::Num(Num::Double(a.as_f64() - b.as_f64()))
    }
}

pub fn num_mul(a: Num, b: Num) -> NumResult {
    if let Some((x, y)) = Num::both_long(a, b) {
        match x.checked_mul(y) {
            Some(v) => NumResult::Num(Num::Long(v)),
            None => NumResult::Overflow,
        }
    } else {
        NumResult::Num(Num::Double(a.as_f64() * b.as_f64()))
    }
}

/// `/` — long/long exact → long; non-exact long/long → double (no Ratio
/// yet, documented deviation); any double → double.
pub fn num_div(a: Num, b: Num) -> Num {
    if let Some((x, y)) = Num::both_long(a, b) {
        if y != 0 && x % y == 0 {
            return Num::Long(x / y);
        }
    }
    Num::Double(a.as_f64() / b.as_f64())
}

/// `quot`/`rem` — long results for long args; otherwise truncating-double.
pub fn num_quot(a: Num, b: Num) -> Num {
    match Num::both_long(a, b) {
        Some((x, y)) => Num::Long(x / y),
        None => Num::Double((a.as_f64() / b.as_f64()).trunc()),
    }
}
pub fn num_rem(a: Num, b: Num) -> Num {
    match Num::both_long(a, b) {
        Some((x, y)) => Num::Long(x % y),
        None => Num::Double(a.as_f64() % b.as_f64()),
    }
}

pub fn num_inc(a: Num) -> NumResult { num_add(a, Num::Long(1)) }
pub fn num_dec(a: Num) -> NumResult { num_sub(a, Num::Long(1)) }

/// Numeric ordering, type-agnostic (used by `<` `>` `<=` `>=` and `==`).
fn num_cmp(a: Num, b: Num) -> std::cmp::Ordering {
    match Num::both_long(a, b) {
        Some((x, y)) => x.cmp(&y),
        None => a.as_f64().partial_cmp(&b.as_f64()).unwrap_or(std::cmp::Ordering::Less),
    }
}
pub fn num_lt(a: Num, b: Num) -> bool { num_cmp(a, b) == std::cmp::Ordering::Less }
pub fn num_gt(a: Num, b: Num) -> bool { num_cmp(a, b) == std::cmp::Ordering::Greater }
pub fn num_le(a: Num, b: Num) -> bool { num_cmp(a, b) != std::cmp::Ordering::Greater }
pub fn num_ge(a: Num, b: Num) -> bool { num_cmp(a, b) != std::cmp::Ordering::Less }

/// `==` — numeric equality across types (`(== 1 1.0)` → true).
pub fn num_eq(a: Num, b: Num) -> bool { num_cmp(a, b) == std::cmp::Ordering::Equal }

/// `=` on numbers — type-aware: a long and a double are never `=`, even if
/// numerically equal (`(= 1 1.0)` → false). Two longs / two doubles compare
/// by value.
pub fn val_num_eq(a: Num, b: Num) -> bool {
    match (a, b) {
        (Num::Long(x), Num::Long(y)) => x == y,
        (Num::Double(x), Num::Double(y)) => x == y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long(n: i64) -> Num { Num::Long(n) }
    fn dbl(d: f64) -> Num { Num::Double(d) }

    #[test]
    fn numeric_tower_matches_clojure() {
        // From the 1.11 oracle (/tmp/numtower.clj):
        assert_eq!(num_add(long(1), long(2)), NumResult::Num(Num::Long(3)));      // (+ 1 2) => 3 Long
        assert_eq!(num_add(long(1), dbl(2.0)), NumResult::Num(Num::Double(3.0))); // (+ 1 2.0) => 3.0 Double
        assert_eq!(num_add(dbl(1.0), dbl(2.0)), NumResult::Num(Num::Double(3.0)));
        assert_eq!(num_mul(long(3), long(4)), NumResult::Num(Num::Long(12)));     // (* 3 4) => 12 Long
        assert_eq!(num_sub(long(7), dbl(2.0)), NumResult::Num(Num::Double(5.0))); // (- 7 2.0) => 5.0
        assert_eq!(num_quot(long(17), long(5)), Num::Long(3));                    // (quot 17 5) => 3
        assert_eq!(num_rem(long(17), long(5)), Num::Long(2));                     // (rem 17 5) => 2
        assert!(num_lt(long(1), long(2)));
        assert!(num_lt(long(1), dbl(2.0)));
        assert!(num_lt(dbl(1.5), long(2)));                                       // (< 1.5 2) => true
        assert_eq!(num_div(long(6), long(2)), Num::Long(3));                      // (/ 6 2) => 3 Long
        assert_eq!(num_div(long(7), long(2)), Num::Double(3.5));                  // (/ 7 2) => 3.5 (deviation: not 7/2)
        assert_eq!(num_div(dbl(7.0), long(2)), Num::Double(3.5));                 // (/ 7.0 2) => 3.5
        assert_eq!(num_inc(long(5)), NumResult::Num(Num::Long(6)));              // (inc 5) => 6
        assert_eq!(num_inc(dbl(5.0)), NumResult::Num(Num::Double(6.0)));         // (inc 5.0) => 6.0
    }

    #[test]
    fn equality_semantics() {
        assert!(val_num_eq(long(1), long(1)));   // (= 1 1) => true
        assert!(!val_num_eq(long(1), dbl(1.0))); // (= 1 1.0) => false (type-aware)
        assert!(num_eq(long(1), dbl(1.0)));      // (== 1 1.0) => true (numeric)
        assert!(!val_num_eq(long(1), long(2)));
    }

    #[test]
    fn overflow_is_flagged() {
        assert_eq!(num_add(long(i64::MAX), long(1)), NumResult::Overflow);
        assert_eq!(num_mul(long(i64::MAX), long(2)), NumResult::Overflow);
    }

    #[test]
    fn fixnum_roundtrip() {
        for n in [0i64, 1, -1, 42, -42, 1_000_000, -1_000_000, FIXNUM_MAX, FIXNUM_MIN] {
            assert!(fits_fixnum(n));
            let b = encode_fixnum(n);
            assert!(is_fixnum(b), "{n} not tagged fixnum: {b:#x}");
            assert!(!is_ptr(b) && !is_nil(b) && !is_bool(b));
            assert_eq!(decode_fixnum(b), n);
        }
    }

    #[test]
    fn fixnum_range() {
        assert!(fits_fixnum(FIXNUM_MAX));
        assert!(fits_fixnum(FIXNUM_MIN));
        assert!(!fits_fixnum(FIXNUM_MAX + 1));
        assert!(!fits_fixnum(FIXNUM_MIN - 1));
    }

    #[test]
    fn immediates_distinct() {
        assert!(is_nil(NIL));
        assert!(is_bool(TRUE) && decode_bool(TRUE));
        assert!(is_bool(FALSE) && !decode_bool(FALSE));
        // immediates are not pointers and not each other's tags
        for v in [NIL, TRUE, FALSE, encode_char('A' as u32), encode_fixnum(7)] {
            assert!(!is_ptr(v), "{v:#x} mistaken for ptr");
        }
        assert!(is_char(encode_char('λ' as u32)));
        assert_eq!(decode_char(encode_char('λ' as u32)), 'λ' as u32);
    }

    #[test]
    fn ptr_is_tag_zero() {
        let addr = 0x1_0000_0008u64; // 8-aligned
        let b = encode_ptr(addr);
        assert!(is_ptr(b));
        assert_eq!(b, addr);
        assert!(!is_fixnum(b) && !is_nil(b) && !is_bool(b) && !is_char(b));
    }
}
