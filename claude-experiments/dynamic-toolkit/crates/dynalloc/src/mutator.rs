use std::marker::PhantomData;

use dynvalue::{TagScheme, Value};

use crate::alloc::{alloc_obj, BumpAllocator};
use dynobj::{read_value_field, ObjHeader, RootStack, TypeInfo};

// ─── Root ───────────────────────────────────────────────────────────

/// Opaque handle to a GC root slot.
///
/// Just an index into the `RootStack`. `Copy` because it has no ownership
/// semantics — multiple copies refer to the same slot. The underlying slot
/// is updated in-place by the GC (forwarding pointers).
///
/// Invalidated by [`Mutator::restore`] — debug-mode bounds checks catch
/// use-after-restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Root {
    index: usize,
}

// ─── GcRef ──────────────────────────────────────────────────────────

/// A lifetime-guarded snapshot of a rooted value.
///
/// Borrows `&'a Mutator`, preventing any `&mut Mutator` operations
/// (allocation, safepoint, restore) while this reference exists.
/// After any GC point, you must call [`Mutator::get`] again to re-read.
///
/// `Copy` so multiple reads in the same scope are fine — all invalidated
/// together when the borrow ends.
#[derive(Clone, Copy)]
pub struct GcRef<'a> {
    bits: u64,
    _borrow: PhantomData<&'a Mutator>,
}

impl<'a> GcRef<'a> {
    /// Get the raw `u64` bits.
    ///
    /// **Escape hatch**: if this encodes a heap pointer, the caller must
    /// root it before the next GC point.
    #[inline(always)]
    pub fn bits(self) -> u64 {
        self.bits
    }

    /// Decode as a `Value<S>` for tag checking and payload extraction.
    #[inline(always)]
    pub fn value<S: TagScheme>(self) -> Value<S> {
        Value::from_bits(self.bits)
    }

    /// Convenience: check if the value has a given tag under scheme `S`.
    #[inline(always)]
    pub fn has_tag<S: TagScheme>(self, tag: u32) -> bool {
        Value::<S>::from_bits(self.bits).has_tag(tag)
    }
}

// ─── RootScope ──────────────────────────────────────────────────────

/// Watermark token for scoped root cleanup.
///
/// Created by [`Mutator::save`], consumed by [`Mutator::restore`].
/// Newtype prevents accidentally using an arbitrary `usize` as a watermark.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct RootScope {
    watermark: usize,
}

// ─── Mutator ────────────────────────────────────────────────────────

/// Safe wrapper over [`RootStack`] + [`BumpAllocator`].
///
/// Uses Rust's borrow checker to enforce GC safety:
/// - **`&self`** methods: read roots (`get`), update values (`set`), save scope
/// - **`&mut self`** methods: allocate, push new roots, restore scope, safepoint
///
/// Since `&self` and `&mut self` conflict, all [`GcRef`]s must be dropped
/// before any GC-triggering operation.
pub struct Mutator {
    roots: RootStack,
    bump: BumpAllocator,
}

impl Mutator {
    /// Create a new mutator with the given bump allocator.
    pub fn new(bump: BumpAllocator) -> Self {
        Mutator {
            roots: RootStack::new(),
            bump,
        }
    }

    // ─── &mut self (structural changes / GC points) ─────────────

    /// Register a value as a GC root, returning a [`Root`] handle.
    ///
    /// Takes `&mut self` because `Vec::push` may reallocate the root
    /// stack's backing storage, which would invalidate any outstanding
    /// `scan_roots` slot pointers.
    pub fn root(&mut self, bits: u64) -> Root {
        let index = self.roots.push(bits);
        Root { index }
    }

    /// Allocate a heap object, initialize its header, and auto-root it.
    ///
    /// Returns `Some(Root)` on success, `None` if the bump allocator is full.
    /// The root slot stores the raw pointer (`ptr as u64`), not a tagged value.
    /// Use [`Mutator::set`] to replace it with a tagged encoding if needed.
    ///
    /// Takes `&mut self` because it pushes a new root.
    ///
    /// # Safety
    /// `info` must accurately describe the object layout for header type `H`.
    pub fn alloc<H: ObjHeader>(
        &mut self,
        info: &'static TypeInfo,
        varlen_len: usize,
    ) -> Option<Root> {
        let ptr = unsafe { alloc_obj::<H>(&self.bump, info, varlen_len) };
        if ptr.is_null() {
            return None;
        }
        let root = self.root(ptr as u64);
        Some(root)
    }

    /// Explicit GC trigger point (no-op for now).
    ///
    /// Takes `&mut self` to force all `GcRef`s to be dropped first,
    /// ensuring no stale pointers survive across a potential collection.
    pub fn safepoint(&mut self) {
        // Future: trigger GC here
    }

    /// Pop all roots pushed after the given scope watermark.
    ///
    /// Takes `&mut self` to prevent use while `GcRef`s are outstanding.
    pub fn restore(&mut self, scope: RootScope) {
        self.roots.truncate(scope.watermark);
    }

    // ─── &self (reads / interior-mutable writes) ────────────────

    /// Read a root's current value, returning a [`GcRef`] that borrows `&self`.
    ///
    /// The `GcRef` prevents `&mut self` operations until dropped,
    /// forcing a re-read after any GC point.
    pub fn get(&self, root: &Root) -> GcRef<'_> {
        GcRef {
            bits: self.roots.get(root.index),
            _borrow: PhantomData,
        }
    }

    /// Update a root's value.
    ///
    /// Uses interior mutability (`Cell`), so this takes `&self`.
    /// No GC, no reallocation — safe to call while other roots are read.
    pub fn set(&self, root: &Root, bits: u64) {
        self.roots.set(root.index, bits);
    }

    /// Snapshot the current root stack length for later [`restore`](Self::restore).
    pub fn save(&self) -> RootScope {
        RootScope {
            watermark: self.roots.watermark(),
        }
    }

    /// Access the underlying [`RootStack`] for GC implementation.
    pub fn roots(&self) -> &RootStack {
        &self.roots
    }

    /// Access the underlying [`BumpAllocator`] for GC implementation.
    pub fn bump(&self) -> &BumpAllocator {
        &self.bump
    }

    // ─── Convenience (&mut self, no GC) ─────────────────────────

    /// Read a value field from a rooted heap object and root it.
    ///
    /// Reads field `index` from the object pointed to by `root`,
    /// then pushes the field value as a new root.
    ///
    /// # Safety
    /// - The root must contain a valid heap pointer (raw `ptr as u64`
    ///   or a tagged pointer under scheme `S`).
    /// - `info` must describe the object's layout.
    /// - `index` must be less than `info.value_field_count`.
    pub unsafe fn root_field<S: TagScheme>(
        &mut self,
        root: &Root,
        info: &'static TypeInfo,
        index: u16,
    ) -> Root {
        let obj_bits = self.roots.get(root.index);
        let obj_ptr = obj_bits as *const u8;
        let field_val: Value<S> = unsafe { read_value_field(obj_ptr, info, index) };
        self.root(field_val.to_bits())
    }
}
