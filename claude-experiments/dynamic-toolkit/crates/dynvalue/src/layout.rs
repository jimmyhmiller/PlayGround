use crate::scheme::TagScheme;
use crate::value::Value;

/// Trait for types that serve as typed wrappers around [`Value`].
///
/// Provides GC integration — the garbage collector needs to know
/// which values are heap pointers so it can trace them.
pub trait TaggedValue: Copy {
    type Scheme: TagScheme;

    fn from_value(v: Value<Self::Scheme>) -> Self;
    fn to_value(self) -> Value<Self::Scheme>;

    /// Does this value contain a heap pointer that the GC must trace?
    fn is_heap_ptr(self) -> bool;
}

/// Trait for types that can be stored in a tagged value's payload.
///
/// The `payload_bits` parameter tells you how wide the payload is
/// (scheme-dependent), so you can do sign extension, range checks, etc.
pub trait Payload: Copy {
    fn encode(self, payload_bits: u32) -> u64;
    fn decode(payload: u64, payload_bits: u32) -> Self;
}

impl Payload for u64 {
    #[inline(always)]
    fn encode(self, _payload_bits: u32) -> u64 {
        self
    }
    #[inline(always)]
    fn decode(payload: u64, _payload_bits: u32) -> Self {
        payload
    }
}

impl Payload for u32 {
    #[inline(always)]
    fn encode(self, _payload_bits: u32) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn decode(payload: u64, _payload_bits: u32) -> Self {
        payload as u32
    }
}

impl Payload for i64 {
    #[inline(always)]
    fn encode(self, payload_bits: u32) -> u64 {
        // Mask to payload width so negative values don't overflow
        let mask = if payload_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << payload_bits) - 1
        };
        (self as u64) & mask
    }
    #[inline(always)]
    fn decode(payload: u64, payload_bits: u32) -> Self {
        // Sign-extend from payload_bits to 64 bits
        let shift = 64 - payload_bits;
        ((payload as i64) << shift) >> shift
    }
}

impl Payload for i32 {
    #[inline(always)]
    fn encode(self, payload_bits: u32) -> u64 {
        let mask = if payload_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << payload_bits) - 1
        };
        (self as u64) & mask
    }
    #[inline(always)]
    fn decode(payload: u64, _payload_bits: u32) -> Self {
        payload as i32
    }
}

impl Payload for bool {
    #[inline(always)]
    fn encode(self, _payload_bits: u32) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn decode(payload: u64, _payload_bits: u32) -> Self {
        payload != 0
    }
}

impl Payload for usize {
    #[inline(always)]
    fn encode(self, _payload_bits: u32) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn decode(payload: u64, _payload_bits: u32) -> Self {
        payload as usize
    }
}

impl<T> Payload for *mut T {
    #[inline(always)]
    fn encode(self, _payload_bits: u32) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn decode(payload: u64, _payload_bits: u32) -> Self {
        payload as *mut T
    }
}

impl<T> Payload for *const T {
    #[inline(always)]
    fn encode(self, _payload_bits: u32) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn decode(payload: u64, _payload_bits: u32) -> Self {
        payload as *const T
    }
}

/// Define a typed value wrapper with named tags, constructors, and accessors.
///
/// # Syntax
///
/// ```rust
/// # use dynvalue::define_value;
/// # use dynvalue::LowBit;
/// define_value! {
///     pub MyVal: LowBit<3> {
///         #[heap] Ptr(0): *mut u8,
///         Fixnum(1): i64,
///         Bool(2): bool,
///         Nil(3),
///         Symbol(4): u32,
///     }
/// }
/// ```
///
/// ## What this generates
///
/// - `pub struct MyVal(Value<LowBit<3>>)` — a newtype wrapper
/// - `pub enum MyValKind { Ptr(*mut u8), Fixnum(i64), ... Nil, ... }` — for matching
/// - Constructors: `MyVal::ptr(p)`, `MyVal::fixnum(n)`, `MyVal::nil()`, etc.
/// - Checkers: `val.is_ptr()`, `val.is_fixnum()`, `val.is_nil()`, etc.
/// - Accessors: `val.as_ptr()`, `val.as_fixnum()`, etc. (not for unit variants)
/// - `val.kind() -> MyValKind` — for exhaustive matching
/// - `impl TaggedValue` with `is_heap_ptr()` based on `#[heap]` annotations
///
/// ## Variants
///
/// - `Name(tag): Type` — a tagged variant with a typed payload
/// - `Name(tag)` — a unit variant (no payload, like nil/undefined)
/// - `#[heap]` before a variant marks it as a GC-traceable heap pointer
#[macro_export]
macro_rules! define_value {
    // ── Entry point ──────────────────────────────────────────────
    (
        $vis:vis $name:ident : $scheme:ty {
            $($body:tt)*
        }
    ) => {
        $crate::define_value!(@munch
            $vis $name ($scheme)
            typed=[] unit=[] heap=[]
            $($body)*
        );
    };

    // ── TT muncher: typed variant with #[heap] ──────────────────
    (@munch
        $vis:vis $name:ident ($scheme:ty)
        typed=[$($t:tt)*] unit=[$($u:tt)*] heap=[$($h:tt)*]
        #[heap] $V:ident($tag:expr): $pay:ty, $($rest:tt)*
    ) => {
        $crate::define_value!(@munch
            $vis $name ($scheme)
            typed=[$($t)* ($V, $tag, $pay)]
            unit=[$($u)*]
            heap=[$($h)* $tag]
            $($rest)*
        );
    };

    // ── TT muncher: typed variant ───────────────────────────────
    (@munch
        $vis:vis $name:ident ($scheme:ty)
        typed=[$($t:tt)*] unit=[$($u:tt)*] heap=[$($h:tt)*]
        $V:ident($tag:expr): $pay:ty, $($rest:tt)*
    ) => {
        $crate::define_value!(@munch
            $vis $name ($scheme)
            typed=[$($t)* ($V, $tag, $pay)]
            unit=[$($u)*]
            heap=[$($h)*]
            $($rest)*
        );
    };

    // ── TT muncher: unit variant with #[heap] ───────────────────
    (@munch
        $vis:vis $name:ident ($scheme:ty)
        typed=[$($t:tt)*] unit=[$($u:tt)*] heap=[$($h:tt)*]
        #[heap] $V:ident($tag:expr), $($rest:tt)*
    ) => {
        $crate::define_value!(@munch
            $vis $name ($scheme)
            typed=[$($t)*]
            unit=[$($u)* ($V, $tag)]
            heap=[$($h)* $tag]
            $($rest)*
        );
    };

    // ── TT muncher: unit variant ────────────────────────────────
    (@munch
        $vis:vis $name:ident ($scheme:ty)
        typed=[$($t:tt)*] unit=[$($u:tt)*] heap=[$($h:tt)*]
        $V:ident($tag:expr), $($rest:tt)*
    ) => {
        $crate::define_value!(@munch
            $vis $name ($scheme)
            typed=[$($t)*]
            unit=[$($u)* ($V, $tag)]
            heap=[$($h)*]
            $($rest)*
        );
    };

    // ── TT muncher: done ────────────────────────────────────────
    (@munch
        $vis:vis $name:ident ($scheme:ty)
        typed=[$($t:tt)*] unit=[$($u:tt)*] heap=[$($h:tt)*]
    ) => {
        $crate::define_value!(@generate
            $vis $name ($scheme)
            typed=[$($t)*]
            unit=[$($u)*]
            heap=[$($h)*]
        );
    };

    // ── Code generation ─────────────────────────────────────────
    (@generate
        $vis:vis $name:ident ($scheme:ty)
        typed=[$(($TV:ident, $TTag:expr, $TPay:ty))*]
        unit=[$(($UV:ident, $UTag:expr))*]
        heap=[$($HTag:expr)*]
    ) => {
        ::paste::paste! {
            #[repr(transparent)]
            #[derive(Clone, Copy, PartialEq, Eq)]
            $vis struct $name($crate::Value<$scheme>);

            #[derive(Debug, Clone, Copy, PartialEq)]
            #[allow(dead_code)]
            $vis enum [<$name Kind>] {
                $($TV($TPay),)*
                $($UV,)*
            }

            #[allow(dead_code)]
            impl $name {
                // ── Typed variant constructors ───────────────
                $(
                    #[inline(always)]
                    $vis fn [<$TV:lower>](val: $TPay) -> Self {
                        Self($crate::Value::tagged(
                            $TTag,
                            <$TPay as $crate::Payload>::encode(
                                val,
                                <$scheme as $crate::TagScheme>::PAYLOAD_BITS,
                            ),
                        ))
                    }
                )*

                // ── Unit variant constructors ────────────────
                $(
                    #[inline(always)]
                    $vis fn [<$UV:lower>]() -> Self {
                        Self($crate::Value::tagged($UTag, 0))
                    }
                )*

                // ── Typed variant checkers ───────────────────
                $(
                    #[inline(always)]
                    $vis fn [<is_ $TV:lower>](self) -> bool {
                        self.0.has_tag($TTag)
                    }
                )*

                // ── Unit variant checkers ────────────────────
                $(
                    #[inline(always)]
                    $vis fn [<is_ $UV:lower>](self) -> bool {
                        self.0.has_tag($UTag)
                    }
                )*

                // ── Typed variant accessors ──────────────────
                $(
                    #[inline(always)]
                    $vis fn [<as_ $TV:lower>](self) -> $TPay {
                        <$TPay as $crate::Payload>::decode(
                            self.0.payload(),
                            <$scheme as $crate::TagScheme>::PAYLOAD_BITS,
                        )
                    }
                )*

                // ── Kind (for matching) ──────────────────────
                $vis fn kind(self) -> [<$name Kind>] {
                    $(
                        if self.0.has_tag($TTag) {
                            return [<$name Kind>]::$TV(
                                <$TPay as $crate::Payload>::decode(
                                    self.0.payload(),
                                    <$scheme as $crate::TagScheme>::PAYLOAD_BITS,
                                )
                            );
                        }
                    )*
                    $(
                        if self.0.has_tag($UTag) {
                            return [<$name Kind>]::$UV;
                        }
                    )*
                    panic!("unknown tag")
                }

                // ── Underlying value access ──────────────────
                #[inline(always)]
                $vis fn value(self) -> $crate::Value<$scheme> {
                    self.0
                }

                #[inline(always)]
                $vis fn from_value(v: $crate::Value<$scheme>) -> Self {
                    Self(v)
                }

                #[inline(always)]
                $vis fn to_bits(self) -> u64 {
                    self.0.to_bits()
                }

                #[inline(always)]
                $vis fn from_bits(bits: u64) -> Self {
                    Self($crate::Value::from_bits(bits))
                }
            }

            impl ::core::fmt::Debug for $name {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    ::core::fmt::Debug::fmt(&self.kind(), f)
                }
            }

            impl $crate::TaggedValue for $name {
                type Scheme = $scheme;

                fn from_value(v: $crate::Value<$scheme>) -> Self {
                    Self(v)
                }

                fn to_value(self) -> $crate::Value<$scheme> {
                    self.0
                }

                fn is_heap_ptr(self) -> bool {
                    false $(|| self.0.has_tag($HTag))*
                }
            }
        }
    };
}
