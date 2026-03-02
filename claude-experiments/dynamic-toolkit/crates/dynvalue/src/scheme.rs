/// What a decoded value looks like: either a tagged payload or an unboxed float.
///
/// Low-bit tagging always produces `Tagged` — floats live behind a pointer.
/// NaN-boxing produces `Float` for doubles and `Tagged` for everything else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decoded {
    Tagged { tag: u32, payload: u64 },
    Float(f64),
}

impl Decoded {
    /// Get the tag, if this is a tagged value. Panics for unboxed floats.
    pub fn tag(self) -> u32 {
        match self {
            Decoded::Tagged { tag, .. } => tag,
            Decoded::Float(_) => panic!("cannot get tag of unboxed float"),
        }
    }
}

/// A tagging scheme defines how to pack type tags and payloads into a `u64`.
///
/// Two families of scheme exist:
///
/// - **Low-bit tagging** (`LowBit<N>`): Uses the bottom N bits as the tag.
///   Every value has a tag. Pointers must be aligned to `2^N`.
///   Floats are not stored inline — they go behind a heap pointer.
///
/// - **NaN-boxing** (`NanBox`): Stores IEEE 754 doubles directly.
///   Non-float values are encoded in the NaN payload space.
///   Gives you unboxed floats at the cost of fewer tag bits and
///   48-bit pointer width.
///
/// # For compiler authors
///
/// The constants and methods on this trait give you the information needed
/// to emit code that tags/untags values. The `encode`/`decode` methods
/// are also usable at runtime (in your GC, builtins, etc).
pub trait TagScheme: Copy + 'static {
    /// How many distinct tag values this scheme supports (for the `Tagged` variant).
    /// NaN-boxing also supports unboxed floats outside this count.
    const TAG_COUNT: u32;

    /// How many bits of payload each tagged value can carry.
    const PAYLOAD_BITS: u32;

    /// Whether this scheme can store `f64` without a tag.
    const HAS_UNBOXED_FLOAT: bool;

    /// Pack a tag and payload into raw bits.
    ///
    /// # Panics
    /// Panics (debug) if `tag >= TAG_COUNT` or `payload` exceeds `PAYLOAD_BITS`.
    fn encode_tagged(tag: u32, payload: u64) -> u64;

    /// Pack an `f64` into raw bits. Only meaningful when `HAS_UNBOXED_FLOAT` is true.
    /// For schemes without unboxed floats, this panics.
    fn encode_float(f: f64) -> u64;

    /// Decode raw bits into either a tagged value or a float.
    fn decode(bits: u64) -> Decoded;

    /// Fast check: do these raw bits represent an unboxed float?
    /// Always returns `false` for schemes without unboxed float support.
    fn is_float(bits: u64) -> bool;

    /// Fast check: do these raw bits have the given tag?
    /// Returns `false` for unboxed floats in NaN-boxing.
    fn has_tag(bits: u64, tag: u32) -> bool;

    /// Extract the payload without checking the tag. Caller must ensure
    /// this is a tagged value (not an unboxed float).
    fn extract_payload(bits: u64) -> u64;
}

/// Marker trait for schemes that support storing `f64` without a tag.
/// Implemented by `NanBox` but not by `LowBit`.
pub trait HasUnboxedFloat: TagScheme {
    /// Decode raw bits as an `f64`. Caller must ensure `is_float()` is true.
    fn decode_float(bits: u64) -> f64;
}
