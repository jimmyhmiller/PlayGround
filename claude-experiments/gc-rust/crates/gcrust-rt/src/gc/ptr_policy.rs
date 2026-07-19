//! Identity `PtrPolicy`: slot bits are interpreted as the raw pointer.
//!
//! Zero is treated as the null/non-pointer sentinel. Any non-zero `u64`
//! is taken to be a heap pointer. Frontends that want tagged values
//! (NaN-boxing, low-bit tagging, etc.) should provide their own
//! `PtrPolicy` impl in their own crate.

use crate::gc::semi_space::PtrPolicy;

/// `PtrPolicy` that treats slot bits as a raw pointer. `0` is non-pointer.
pub struct IdentityPtrPolicy;

impl PtrPolicy for IdentityPtrPolicy {
    #[inline(always)]
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        // `0` is the null/non-pointer sentinel. Additionally, every heap object
        // is at least 8-byte aligned (TypeInfo.align_log2 >= 3), so a real
        // pointer always has its low 3 bits clear. In the i64-uniform value
        // model this rejects most integers that merely look like heap addresses
        // (e.g. an enum int-payload) before the GC would follow them — a precise
        // filter complementing the type_id validity check at the copy/promote
        // sites. (A frontend using tagged pointers must supply its own policy.)
        if bits == 0 || (bits & 0b111) != 0 {
            None
        } else {
            Some(bits as *mut u8)
        }
    }

    #[inline(always)]
    fn encode_ptr(ptr: *mut u8) -> u64 {
        ptr as u64
    }
}
