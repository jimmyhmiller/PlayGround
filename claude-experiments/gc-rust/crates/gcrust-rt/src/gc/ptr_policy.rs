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
        if bits == 0 {
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
