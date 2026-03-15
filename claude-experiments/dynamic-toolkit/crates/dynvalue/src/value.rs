use std::fmt;
use std::marker::PhantomData;

use crate::scheme::{Decoded, HasUnboxedFloat, TagScheme};

/// A tagged value parameterized by a [`TagScheme`].
///
/// This is a zero-cost wrapper around `u64`. The scheme determines
/// how tags and payloads are packed into those 64 bits.
///
/// `Value<LowBit<3>>` and `Value<NanBox>` are both just `u64` at runtime,
/// but their methods encode/decode according to their scheme's rules.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Value<S: TagScheme> {
    bits: u64,
    _scheme: PhantomData<S>,
}

impl<S: TagScheme> Value<S> {
    /// Create a tagged value from a tag and payload.
    #[inline(always)]
    pub fn tagged(tag: u32, payload: u64) -> Self {
        Self {
            bits: S::encode_tagged(tag, payload),
            _scheme: PhantomData,
        }
    }

    /// Create a value from raw bits. No validation.
    #[inline(always)]
    pub const fn from_bits(bits: u64) -> Self {
        Self {
            bits,
            _scheme: PhantomData,
        }
    }

    /// Get the raw bits.
    #[inline(always)]
    pub const fn to_bits(self) -> u64 {
        self.bits
    }

    /// Decode this value into its components.
    #[inline(always)]
    pub fn decode(self) -> Decoded {
        S::decode(self.bits)
    }

    /// Check if this value has a specific tag.
    /// Returns `false` for unboxed floats in NaN-boxing schemes.
    #[inline(always)]
    pub fn has_tag(self, tag: u32) -> bool {
        S::has_tag(self.bits, tag)
    }

    /// Check if this value is an unboxed float.
    /// Always `false` for low-bit tagging schemes.
    #[inline(always)]
    pub fn is_float(self) -> bool {
        S::is_float(self.bits)
    }

    /// Extract the payload without checking the tag.
    /// Caller must ensure this is a tagged value, not an unboxed float.
    #[inline(always)]
    pub fn payload(self) -> u64 {
        S::extract_payload(self.bits)
    }

    /// Extract the payload as a pointer.
    /// Caller must ensure the payload actually holds a valid pointer.
    #[inline(always)]
    pub fn as_ptr<T>(self) -> *const T {
        S::extract_payload(self.bits) as *const T
    }

    /// Extract the payload as a mutable pointer.
    #[inline(always)]
    pub fn as_mut_ptr<T>(self) -> *mut T {
        S::extract_payload(self.bits) as *mut T
    }
}

impl<S: HasUnboxedFloat> Value<S> {
    /// Create an unboxed float value. Only available on schemes
    /// that support unboxed floats (e.g., NaN-boxing).
    #[inline(always)]
    pub fn float(f: f64) -> Self {
        Self {
            bits: S::encode_float(f),
            _scheme: PhantomData,
        }
    }

    /// Extract the float value. Caller must ensure `is_float()` is true.
    #[inline(always)]
    pub fn as_f64(self) -> f64 {
        S::decode_float(self.bits)
    }
}

impl<S: TagScheme> fmt::Debug for Value<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match S::decode(self.bits) {
            Decoded::Tagged { tag, payload } => f
                .debug_struct("Value")
                .field("tag", &tag)
                .field("payload", &format_args!("{payload:#x}"))
                .finish(),
            Decoded::Float(v) => f.debug_struct("Value").field("float", &v).finish(),
        }
    }
}
