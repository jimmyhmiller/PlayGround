use crate::type_info::TypeInfo;

/// Defines the header prepended to every heap object.
///
/// Choose compact (8 bytes), full (16 bytes), or define your own.
/// The header lives at the start of every heap allocation and lets
/// the runtime (GC, debugger, etc.) identify the object's shape.
///
/// # Safety
///
/// Implementations must be `#[repr(C)]` and have a size equal to `Self::SIZE`.
pub trait ObjHeader: Copy + 'static {
    /// Size of this header in bytes.
    const SIZE: usize;

    /// Byte offset of the `type_id` field within this header.
    /// Used by the heap walker to recover the type from any object.
    const TYPE_ID_OFFSET: usize;

    /// Initialize a header for a newly allocated object.
    fn new(type_id: u16) -> Self;

    /// Get the type ID (index into the runtime's TypeInfo table).
    fn type_id(&self) -> u16;
}

/// Declarative macro for defining header structs with optional bitfield sub-fields.
///
/// # Field kinds
///
/// - `type_info: *const TypeInfo,` — special field, wires up `ObjHeader` trait impl
/// - `name: Type,` — plain field, gets getter/setter
/// - `name: Type { sub: [lo..hi], ... }` — bitfield word with named sub-field accessors
///
/// # Example
///
/// ```ignore
/// define_header! {
///     pub MyHeader {
///         type_info: *const TypeInfo,
///         gc_bits: u64 {
///             mark:       [0..2],
///             pinned:     [2..3],
///             generation: [3..6],
///             forwarding: [6..64],
///         }
///         identity_hash: u32,
///     }
/// }
/// ```
#[macro_export]
macro_rules! define_header {
    // ─── Entry point ────────────────────────────────────────────────
    (pub $name:ident { $($body:tt)* }) => {
        $crate::define_header! {
            @munch
            name = $name,
            struct_fields = [],
            has_type_info = no,
            accessors = [],
            rest = [$($body)*]
        }
    };

    // ─── Munch: type_id special field ─────────────────────────────
    (@munch
        name = $name:ident,
        struct_fields = [$($sf:tt)*],
        has_type_info = $_old:ident,
        accessors = [$($acc:tt)*],
        rest = [type_info : * const TypeInfo , $($rest:tt)*]
    ) => {
        $crate::define_header! {
            @munch
            name = $name,
            struct_fields = [$($sf)* type_id: u16,],
            has_type_info = yes,
            accessors = [$($acc)*],
            rest = [$($rest)*]
        }
    };

    // ─── Munch: bitfield word ───────────────────────────────────────
    // Uses $ty:ident (not $ty:ty) so `u64 { ... }` isn't consumed as a type.
    (@munch
        name = $name:ident,
        struct_fields = [$($sf:tt)*],
        has_type_info = $hti:ident,
        accessors = [$($acc:tt)*],
        rest = [$field:ident : $ty:ident { $($bits:tt)* } $($rest:tt)*]
    ) => {
        $crate::define_header! {
            @munch_bits
            parent_name = $name,
            struct_fields = [$($sf)* $field: $ty,],
            has_type_info = $hti,
            accessors = [$($acc)*
                #[inline(always)]
                pub fn $field(&self) -> $ty {
                    self.$field
                }
                #[inline(always)]
                pub fn [<set_ $field>](&mut self, val: $ty) {
                    self.$field = val;
                }
            ],
            field = $field,
            field_ty = $ty,
            bits_rest = [$($bits)*],
            munch_rest = [$($rest)*]
        }
    };

    // ─── Munch: plain field ─────────────────────────────────────────
    (@munch
        name = $name:ident,
        struct_fields = [$($sf:tt)*],
        has_type_info = $hti:ident,
        accessors = [$($acc:tt)*],
        rest = [$field:ident : $ty:ty , $($rest:tt)*]
    ) => {
        $crate::define_header! {
            @munch
            name = $name,
            struct_fields = [$($sf)* $field: $ty,],
            has_type_info = $hti,
            accessors = [$($acc)*
                #[inline(always)]
                pub fn $field(&self) -> $ty {
                    self.$field
                }
                #[inline(always)]
                pub fn [<set_ $field>](&mut self, val: $ty) {
                    self.$field = val;
                }
            ],
            rest = [$($rest)*]
        }
    };

    // ─── Munch: done ────────────────────────────────────────────────
    (@munch
        name = $name:ident,
        struct_fields = [$($sf:tt)*],
        has_type_info = $hti:ident,
        accessors = [$($acc:tt)*],
        rest = []
    ) => {
        $crate::define_header! {
            @generate
            name = $name,
            struct_fields = [$($sf)*],
            has_type_info = $hti,
            accessors = [$($acc)*]
        }
    };

    // ─── Munch bits: sub-field ──────────────────────────────────────
    (@munch_bits
        parent_name = $name:ident,
        struct_fields = [$($sf:tt)*],
        has_type_info = $hti:ident,
        accessors = [$($acc:tt)*],
        field = $field:ident,
        field_ty = $ty:ident,
        bits_rest = [$sub:ident : [$lo:literal .. $hi:literal] , $($more:tt)*],
        munch_rest = [$($mrest:tt)*]
    ) => {
        $crate::define_header! {
            @munch_bits
            parent_name = $name,
            struct_fields = [$($sf)*],
            has_type_info = $hti,
            accessors = [$($acc)*
                #[inline(always)]
                pub fn $sub(&self) -> u64 {
                    (self.$field as u64 >> $lo) & ((1u64 << ($hi - $lo)) - 1)
                }
                #[inline(always)]
                pub fn [<set_ $sub>](&mut self, val: u64) {
                    let mask: u64 = ((1u64 << ($hi - $lo)) - 1) << $lo;
                    self.$field = ((self.$field as u64 & !mask) | ((val << $lo) & mask)) as $ty;
                }
            ],
            field = $field,
            field_ty = $ty,
            bits_rest = [$($more)*],
            munch_rest = [$($mrest)*]
        }
    };

    // ─── Munch bits: done ───────────────────────────────────────────
    (@munch_bits
        parent_name = $name:ident,
        struct_fields = [$($sf:tt)*],
        has_type_info = $hti:ident,
        accessors = [$($acc:tt)*],
        field = $_field:ident,
        field_ty = $_ty:ident,
        bits_rest = [],
        munch_rest = [$($mrest:tt)*]
    ) => {
        $crate::define_header! {
            @munch
            name = $name,
            struct_fields = [$($sf)*],
            has_type_info = $hti,
            accessors = [$($acc)*],
            rest = [$($mrest)*]
        }
    };

    // ─── Generate: with type_info (implements ObjHeader) ────────────
    (@generate
        name = $name:ident,
        struct_fields = [$($sf:tt)*],
        has_type_info = yes,
        accessors = [$($acc:tt)*]
    ) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name {
            $($sf)*
        }

        impl $crate::ObjHeader for $name {
            const SIZE: usize = core::mem::size_of::<$name>();
            const TYPE_ID_OFFSET: usize = core::mem::offset_of!($name, type_id);

            #[inline(always)]
            fn new(type_id: u16) -> Self {
                let mut h: Self = unsafe { core::mem::zeroed() };
                h.type_id = type_id;
                h
            }

            #[inline(always)]
            fn type_id(&self) -> u16 {
                self.type_id
            }
        }

        $crate::__paste! {
            impl $name {
                $($acc)*
            }
        }
    };

    // ─── Generate: without type_info ────────────────────────────────
    (@generate
        name = $name:ident,
        struct_fields = [$($sf:tt)*],
        has_type_info = no,
        accessors = [$($acc:tt)*]
    ) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name {
            $($sf)*
        }

        $crate::__paste! {
            impl $name {
                $($acc)*
            }
        }
    };
}

// Compact header: type_id + padding (8 bytes total).
// The remaining 6 bytes after type_id are available for future use
// (e.g., GC mark bits, hash code, etc.)
define_header! {
    pub Compact {
        type_info: *const TypeInfo,
        _pad: u16,
        _pad2: u32,
    }
}

// Full header: GC word + type_id + padding (16 bytes total).
define_header! {
    pub Full {
        gc_word: u64,
        type_info: *const TypeInfo,
        _pad: u16,
        _pad2: u32,
    }
}
