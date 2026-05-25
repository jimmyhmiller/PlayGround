//! Phantom-typed wrappers around raw heap pointers.
//!
//! `TypedPtr<T>` is a `#[repr(transparent)]` newtype over `*const u8`.
//! The phantom `T` is a frontend-defined marker that ties the wrapper
//! to a particular `ObjType`. Two `TypedPtr<ListMarker>` and
//! `TypedPtr<MapMarker>` are *different* Rust types — passing one
//! where the other is expected is a compile error, even though both
//! are pointers under the hood.
//!
//! ## What this protects against
//!
//! Without these wrappers, a function like
//! ```ignore
//! fn read_first(p: *const u8) -> u64 { unsafe { *(p.add(8) as *const u64) } }
//! ```
//! is happy to read offset 8 from any heap pointer — so passing in a
//! `Map` (where offset 8 isn't `first` but the varlen count) silently
//! returns garbage. With a typed wrapper:
//! ```ignore
//! fn read_first(p: TypedPtr<ListMarker>) -> u64 { /* ... */ }
//! ```
//! the caller cannot pass a `Map` pointer at all without an explicit
//! `try_cast` / `cast_unchecked`.
//!
//! ## How to use
//!
//! Define a marker in your frontend:
//! ```ignore
//! pub struct ListMarker;
//! pub type ListPtr = dynobj::TypedPtr<ListMarker>;
//! ```
//!
//! Construct from a raw pointer with a runtime type-id check:
//! ```ignore
//! let p: ListPtr = TypedPtr::try_cast(raw_ptr, list_type_id)?;
//! ```
//!
//! Or construct without checking, when the caller has already
//! verified the shape (e.g. inside a hot inner loop):
//! ```ignore
//! let p: ListPtr = unsafe { TypedPtr::cast_unchecked(raw_ptr) };
//! ```
//!
//! ## What this does NOT do
//!
//! - **No NanBox awareness.** `TypedPtr` holds a raw `*const u8` —
//!   the post-unwrap heap pointer. NanBox-tagged frontends should
//!   wrap their tag-decode in a function that returns `TypedPtr<T>`.
//! - **No field accessors.** Methods like `read_first` belong in the
//!   frontend, since field offsets come from the frontend's
//!   `Layouts` struct, not from `TypedPtr`.
//! - **No GC root tracking.** A bare `TypedPtr<T>` does not survive
//!   GC. For values that must survive an allocation, root the
//!   underlying tagged value via `dynobj::roots::Rooted` and convert
//!   on each use.

use core::marker::PhantomData;

use crate::field::read_type_id;

/// Phantom-typed wrapper around a raw heap pointer.
#[repr(transparent)]
#[derive(Debug)]
#[allow(dead_code)] // Public API — consumers live in downstream crates.
pub struct TypedPtr<T> {
    ptr: *const u8,
    _marker: PhantomData<fn() -> T>,
}

// Manual impls — derive would require T: Clone/Copy.
impl<T> Clone for TypedPtr<T> {
    fn clone(&self) -> Self { *self }
}
impl<T> Copy for TypedPtr<T> {}

#[allow(dead_code)] // Public API — consumers live in downstream crates.
impl<T> TypedPtr<T> {
    /// Construct a `TypedPtr<T>` from a raw pointer if the heap
    /// object's `type_id` matches `expected_type_id`. Returns `None`
    /// for null pointers or type-id mismatches.
    ///
    /// `expected_type_id` is normally the `type_id` of the `ObjType`
    /// that `T` is the marker for — typically read once at startup
    /// from your frontend's `Types` registry.
    #[inline]
    pub fn try_cast(ptr: *const u8, expected_type_id: u16) -> Option<Self> {
        if ptr.is_null() {
            return None;
        }
        let actual = unsafe { read_type_id(ptr, 0) };
        if actual == expected_type_id {
            Some(Self { ptr, _marker: PhantomData })
        } else {
            None
        }
    }

    /// Construct a `TypedPtr<T>` from a raw pointer without a type
    /// check. Use this only when the caller has already verified the
    /// shape (e.g. inside an iterator after the first cell was
    /// type-checked, where every `rest` link is known to be the same
    /// shape).
    ///
    /// # Safety
    /// `ptr` must point to a heap object whose layout matches `T`.
    /// Reading fields from a `TypedPtr<T>` constructed with the wrong
    /// `T` is undefined behavior.
    #[inline]
    pub unsafe fn cast_unchecked(ptr: *const u8) -> Self {
        Self { ptr, _marker: PhantomData }
    }

    /// The underlying raw pointer.
    #[inline]
    pub fn as_ptr(self) -> *const u8 {
        self.ptr
    }

    /// True iff the underlying pointer is null. Constructed
    /// `TypedPtr`s from `try_cast` are never null (it returns `None`),
    /// but `cast_unchecked` does not enforce non-null.
    #[inline]
    pub fn is_null(self) -> bool {
        self.ptr.is_null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub struct FooMarker;
    #[allow(dead_code)]
    pub struct BarMarker;

    #[test]
    fn try_cast_null_returns_none() {
        let p = TypedPtr::<FooMarker>::try_cast(core::ptr::null(), 0);
        assert!(p.is_none());
    }

    #[test]
    fn distinct_markers_are_distinct_types() {
        // Compile-time check: this must not compile if uncommented.
        // let foo: TypedPtr<FooMarker> = unsafe { TypedPtr::cast_unchecked(core::ptr::null()) };
        // let _bar: TypedPtr<BarMarker> = foo;
    }
}
