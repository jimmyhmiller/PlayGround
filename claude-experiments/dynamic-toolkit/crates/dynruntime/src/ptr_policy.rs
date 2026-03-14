use dynalloc::PtrPolicy;
use dynvalue::{LowBit, NanBox, TagScheme};

/// PtrPolicy for `LowBit<N>` tagging where tag 0 = heap pointer.
///
/// With low-bit tagging, aligned pointers naturally have zero low bits,
/// which corresponds to tag 0. So the raw bits ARE the pointer — no
/// encoding/decoding of the payload is needed.
///
/// Requirement: all heap pointers must be aligned to `2^N` bytes.
pub struct LowBitPtrPolicy<const N: u32>;

impl<const N: u32> PtrPolicy for LowBitPtrPolicy<N> {
    #[inline(always)]
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        // Tag 0 = heap pointer. Aligned pointers have zero low bits.
        // Null (0) is not a valid heap pointer.
        if LowBit::<N>::has_tag(bits, 0) && bits != 0 {
            Some(bits as *mut u8)
        } else {
            None
        }
    }

    #[inline(always)]
    fn encode_ptr(ptr: *mut u8) -> u64 {
        // Aligned pointer already has tag 0 (zero low bits).
        debug_assert!(
            (ptr as u64) & ((1u64 << N) - 1) == 0,
            "pointer {ptr:p} not aligned to 2^{N}"
        );
        ptr as u64
    }
}

/// PtrPolicy for NaN-boxing where tag 0 = heap pointer.
///
/// In NaN-boxing, tagged values are encoded in the NaN payload space.
/// Tag 0 is used for heap pointers, with the 48-bit payload holding
/// the pointer address (sufficient for current hardware).
pub struct NanBoxPtrPolicy;

impl PtrPolicy for NanBoxPtrPolicy {
    #[inline(always)]
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        if NanBox::has_tag(bits, 0) {
            let payload = NanBox::extract_payload(bits);
            if payload != 0 {
                Some(payload as *mut u8)
            } else {
                None
            }
        } else {
            None
        }
    }

    #[inline(always)]
    fn encode_ptr(ptr: *mut u8) -> u64 {
        NanBox::encode_tagged(0, ptr as u64)
    }
}
