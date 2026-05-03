use crate::scheme::{Decoded, HasUnboxedFloat, TagScheme};

/// NaN-boxing: doubles stored directly, everything else in the NaN payload.
///
/// IEEE 754 doubles use a specific bit pattern for NaN (Not a Number).
/// Since there are many possible NaN bit patterns but hardware only produces
/// a few, we can hijack unused NaN patterns to store tagged values.
///
/// ## Layout
///
/// **Floats** — stored as raw IEEE 754 bits, no transformation:
/// ```text
/// 63 62       52 51                                  0
/// ┌──┬──────────┬─────────────────────────────────────┐
/// │S │ exponent │             mantissa                │
/// └──┴──────────┴─────────────────────────────────────┘
/// ```
///
/// **Tagged values** — encoded as a quiet NaN with extra marker bit:
/// ```text
/// 63 62    52 51 50 49  48 47                        0
/// ┌──┬────────┬──┬──┬──────┬──────────────────────────┐
/// │0 │11...11 │1 │1 │ tag  │         payload          │
/// └──┴────────┴──┴──┴──────┴──────────────────────────┘
///      exp=NaN  Q  M   2b          48 bits
/// ```
///
/// - **Q** (bit 51): quiet NaN bit (required by IEEE 754)
/// - **M** (bit 50): our marker bit — distinguishes tagged values from real NaN
/// - **tag** (bits 49-48): 2 bits = 4 tag values
/// - **payload** (bits 47-0): 48 bits (enough for pointers on current hardware)
///
/// ## Detecting tagged vs float
///
/// A value is tagged iff `(bits & 0x7FFC_0000_0000_0000) == 0x7FFC_0000_0000_0000`.
/// This checks: positive, exponent all 1s, quiet NaN, and marker bit set.
/// The sign bit is included in the mask to avoid matching negative NaN.
///
/// ## Tag allocation suggestions
/// - Tag 0: heap pointer (most frequent check — fastest)
/// - Tag 1: fixnum (small integer in 48 bits, or 47 bits signed)
/// - Tag 2: symbol / interned string
/// - Tag 3: immediate constants (nil, true, false, undefined — sub-tag in payload)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NanBox;

/// Mask to detect our tagged-value NaN pattern.
/// Bits: sign=0, exponent=all 1s, quiet=1, marker=1
/// We check with `(bits & FULL_MASK) == TAG_PATTERN` where FULL_MASK
/// includes the sign bit to reject negative NaN.
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;

/// Full mask: sign + exponent + quiet + marker bits.
const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;

/// Mask for the 2-bit tag field (bits 49-48).
const TAG_FIELD_MASK: u64 = 0x0003_0000_0000_0000;

/// Mask for the 48-bit payload (bits 47-0).
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

impl TagScheme for NanBox {
    const TAG_COUNT: u32 = 4; // 2 bits
    const PAYLOAD_BITS: u32 = 48;
    const HAS_UNBOXED_FLOAT: bool = true;

    #[inline(always)]
    fn encode_tagged(tag: u32, payload: u64) -> u64 {
        debug_assert!(
            tag < Self::TAG_COUNT,
            "tag {tag} >= TAG_COUNT {}",
            Self::TAG_COUNT
        );
        debug_assert!(
            payload <= PAYLOAD_MASK,
            "payload {payload:#x} exceeds 48 bits"
        );
        TAG_PATTERN | ((tag as u64) << 48) | payload
    }

    #[inline(always)]
    fn encode_float(f: f64) -> u64 {
        let bits = f.to_bits();
        // If the float happens to be a NaN that collides with our tag pattern,
        // canonicalize it. In practice this essentially never happens — hardware
        // produces 0x7FF8_0000_0000_0000 as the canonical NaN.
        if (bits & FULL_MASK) == TAG_PATTERN {
            // Canonical quiet NaN (without our marker bit)
            0x7FF8_0000_0000_0000
        } else {
            bits
        }
    }

    #[inline(always)]
    fn decode(bits: u64) -> Decoded {
        if (bits & FULL_MASK) == TAG_PATTERN {
            Decoded::Tagged {
                tag: ((bits & TAG_FIELD_MASK) >> 48) as u32,
                payload: bits & PAYLOAD_MASK,
            }
        } else {
            Decoded::Float(f64::from_bits(bits))
        }
    }

    #[inline(always)]
    fn is_float(bits: u64) -> bool {
        (bits & FULL_MASK) != TAG_PATTERN
    }

    #[inline(always)]
    fn has_tag(bits: u64, tag: u32) -> bool {
        // Check tagged marker AND specific tag in one comparison.
        // We mask out the payload and compare against the expected pattern.
        (bits & (FULL_MASK | TAG_FIELD_MASK)) == (TAG_PATTERN | ((tag as u64) << 48))
    }

    #[inline(always)]
    fn extract_payload(bits: u64) -> u64 {
        bits & PAYLOAD_MASK
    }
}

impl HasUnboxedFloat for NanBox {
    #[inline(always)]
    fn decode_float(bits: u64) -> f64 {
        f64::from_bits(bits)
    }
}

// ── Embedder helpers ─────────────────────────────────────────────────
//
// Conveniences for host code that needs to construct NanBox values
// without going through the JIT or interpreter.

impl NanBox {
    /// Canonical nil bit pattern: tag 0 with payload 0. Assumes the
    /// embedder uses `NanBoxTags::default()` (nil = tag 0). Custom tag
    /// schemes should call `encode_tagged(custom_nil_tag, 0)` instead.
    pub const NIL: u64 = TAG_PATTERN;

    /// Encode an integer as a NanBox float. Lossless for `|n| < 2^53`;
    /// larger magnitudes lose low-order bits to f64 rounding.
    #[inline(always)]
    pub fn from_int(n: i64) -> u64 {
        (n as f64).to_bits()
    }

    /// Encode an `f64`. Equivalent to `<NanBox as TagScheme>::encode_float`,
    /// surfaced as an inherent method for discoverability. Canonicalizes
    /// NaN bit patterns that would collide with the tag pattern.
    #[inline(always)]
    pub fn from_f64(f: f64) -> u64 {
        <NanBox as TagScheme>::encode_float(f)
    }
}

#[cfg(test)]
mod inherent_tests {
    use super::*;
    use crate::scheme::Decoded;

    #[test]
    fn nil_decodes_to_tag_zero() {
        match <NanBox as TagScheme>::decode(NanBox::NIL) {
            Decoded::Tagged { tag: 0, payload: 0 } => {}
            other => panic!("NIL decoded as {:?}", other),
        }
    }

    #[test]
    fn nil_is_canonical_tag_pattern() {
        // tag 0, payload 0 → just the bare tag pattern.
        assert_eq!(NanBox::NIL, TAG_PATTERN);
    }

    #[test]
    fn from_int_round_trips_for_small_ints() {
        for n in [-1000i64, -1, 0, 1, 42, 524272] {
            let bits = NanBox::from_int(n);
            match <NanBox as TagScheme>::decode(bits) {
                Decoded::Float(f) => assert_eq!(f as i64, n, "lost precision on {}", n),
                other => panic!("from_int({}) decoded as {:?}, not a float", n, other),
            }
        }
    }

    #[test]
    fn from_int_lossless_below_2_pow_53() {
        // 2^53 - 1 fits in f64 mantissa exactly.
        let n = (1i64 << 53) - 1;
        let bits = NanBox::from_int(n);
        match <NanBox as TagScheme>::decode(bits) {
            Decoded::Float(f) => assert_eq!(f as i64, n),
            other => panic!("decoded as {:?}", other),
        }
    }

    #[test]
    fn from_f64_passthrough_for_finite() {
        for f in [0.0f64, -1.5, 3.14159, 1e100, -1e-100] {
            let bits = NanBox::from_f64(f);
            match <NanBox as TagScheme>::decode(bits) {
                Decoded::Float(decoded) => {
                    if f.is_nan() {
                        assert!(decoded.is_nan());
                    } else {
                        assert_eq!(decoded, f);
                    }
                }
                other => panic!("from_f64({}) decoded as {:?}", f, other),
            }
        }
    }

    #[test]
    fn from_f64_canonicalizes_colliding_nan() {
        // The tag pattern itself is a NaN bit pattern that collides with
        // tagged values; encode_float canonicalizes it.
        let colliding = f64::from_bits(TAG_PATTERN);
        assert!(colliding.is_nan());
        let bits = NanBox::from_f64(colliding);
        // Result must NOT be a tagged value — it should still decode as
        // a Float.
        match <NanBox as TagScheme>::decode(bits) {
            Decoded::Float(f) => assert!(f.is_nan()),
            other => panic!("canonicalized NaN decoded as {:?}, not Float", other),
        }
    }
}
