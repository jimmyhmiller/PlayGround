use crate::scheme::{Decoded, TagScheme};

/// Low-bit tagging with `N` tag bits.
///
/// Uses the lowest N bits of a `u64` as the type tag. The remaining
/// `64 - N` bits store the payload (shifted left by N).
///
/// ## Layout
/// ```text
/// 63                    N  N-1   0
/// ┌──────────────────────┬───────┐
/// │      payload         │  tag  │
/// └──────────────────────┴───────┘
/// ```
///
/// ## Common configurations
/// - `LowBit<1>`: 2 tags, 63-bit payload (e.g. fixnum + pointer)
/// - `LowBit<2>`: 4 tags, 62-bit payload
/// - `LowBit<3>`: 8 tags, 61-bit payload (common choice)
/// - `LowBit<4>`: 16 tags, 60-bit payload
///
/// ## Pointer alignment
/// If one of your tags is a heap pointer, the pointer must be aligned
/// to `2^N` bytes. Tag 0 is a convenient choice for pointers since
/// aligned pointers naturally have zero low bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowBit<const N: u32>;

impl<const N: u32> TagScheme for LowBit<N> {
    const TAG_COUNT: u32 = 1 << N;
    const PAYLOAD_BITS: u32 = 64 - N;
    const HAS_UNBOXED_FLOAT: bool = false;

    #[inline(always)]
    fn encode_tagged(tag: u32, payload: u64) -> u64 {
        debug_assert!(
            tag < Self::TAG_COUNT,
            "tag {tag} >= TAG_COUNT {}",
            Self::TAG_COUNT
        );
        debug_assert!(
            payload < (1u64 << Self::PAYLOAD_BITS),
            "payload {payload:#x} exceeds {} bits",
            Self::PAYLOAD_BITS
        );
        (payload << N) | (tag as u64)
    }

    #[inline(always)]
    fn encode_float(_f: f64) -> u64 {
        panic!("LowBit tagging does not support unboxed floats — box them behind a pointer")
    }

    #[inline(always)]
    fn decode(bits: u64) -> Decoded {
        Decoded::Tagged {
            tag: (bits & ((1u64 << N) - 1)) as u32,
            payload: bits >> N,
        }
    }

    #[inline(always)]
    fn is_float(_bits: u64) -> bool {
        false
    }

    #[inline(always)]
    fn has_tag(bits: u64, tag: u32) -> bool {
        (bits & ((1u64 << N) - 1)) as u32 == tag
    }

    #[inline(always)]
    fn extract_payload(bits: u64) -> u64 {
        bits >> N
    }
}
