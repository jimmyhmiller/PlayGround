//! Borrow-checked GC handles for **Rust-side** allocation code.
//!
//! ## The bug this makes uncompilable
//!
//! ai-lang's GC is a moving (semi-space copying) collector: any allocation
//! can relocate every live object. Rust code that allocates while holding a
//! raw heap pointer in a local (an arg builder, the wire codec, a runtime
//! helper) reads a stale, pre-move pointer after the next allocation — a
//! crash that only manifests under precise GC timing (and reliably under
//! `AI_LANG_GC_STRESS=1`). We were fixing these by hand with a per-thread
//! scratch stack, which is easy to get wrong (forget to re-read; leak a
//! slot on an early return).
//!
//! This module is the principled fix, modelled on V8's `Local`/`HandleScope`
//! and on dynamic-toolkit's `dynobj::roots::{Heap, Raw, Rooted}`:
//!
//! - [`Alloc`] is the allocation token. Allocating borrows it **mutably**
//!   (`&mut Alloc`).
//! - [`Raw`] is an unrooted heap reference that borrows the token
//!   **immutably** (`&'h Alloc`) for as long as it lives.
//!
//! Because `&Alloc` (held by a live `Raw`) and `&mut Alloc` (needed to
//! allocate) cannot coexist, **holding a `Raw` across an allocation is a
//! borrow-check error**. To survive an allocation you must [`root`](Raw::root)
//! the reference into a [`RootScope`], producing a [`Rooted`] whose
//! [`get`](Rooted::get) re-reads the (possibly relocated) slot.
//!
//! `Alloc`/`RootScope` are independent borrows (the scope roots into the
//! thread's scratch stack, not into the token), so an open scope does not
//! block allocation — exactly the shape that lets a loop allocate while its
//! accumulator stays rooted.
//!
//! This layer is for Rust callers only. JIT-emitted code roots its live
//! pointers through the shadow-stack frame slots its codegen reserves
//! (`compile_operands_rooted` & friends); the borrow checker can't constrain
//! emitted IR.

use std::marker::PhantomData;

use crate::gc::{ThreadState, TypeInfo};
use crate::runtime::{
    Thread, ai_array_new, ai_array_set, ai_bytes_new, ai_gc_alloc_closure, ai_gc_box_int,
    ai_str_new,
};

/// Allocation token for the current thread's heap.
///
/// Borrow `&mut Alloc` to allocate; every live [`Raw`] borrows `&Alloc`, so
/// the borrow checker forbids holding an unrooted reference across an
/// allocation. Mint once at a Rust→runtime boundary with [`Alloc::enter`].
///
/// `!Send`/`!Sync`: GC roots are per-thread.
///
/// # The bug is a compile error
///
/// Holding an unrooted [`Raw`] across an allocation does not type-check —
/// the live `&Alloc` borrow conflicts with the `&mut Alloc` the allocation
/// needs:
///
/// ```compile_fail
/// # use ai_lang::handle::Alloc;
/// # fn demo(a: &mut Alloc) {
/// let first = a.box_int(1);   // Raw<'_> borrows `a`
/// let second = a.box_int(2);  // ERROR: cannot borrow `*a` as mutable —
///                             // `first` still holds the shared borrow
/// let _ = (first, second);
/// # }
/// ```
///
/// Root each reference as it's produced — a `Rooted` borrows the *scope*,
/// not the token, so it leaves `a` free to allocate again. Re-reading via
/// [`get`](Rooted::get) observes the post-relocation pointer:
///
/// ```no_run
/// # use ai_lang::handle::Alloc;
/// # fn demo(a: &mut Alloc, scope: &ai_lang::handle::RootScope) {
/// let first = a.box_int(1).root(scope);  // pinned; `a` is free again
/// let second = a.box_int(2).root(scope); // OK: allocate while `first` stays live
/// // Both `get(a)` borrows are shared, so they coexist:
/// let _ = (first.get(a), second.get(a));
/// # }
/// ```
pub struct Alloc {
    thread: *mut Thread,
    _not_send: PhantomData<*const ()>,
}

/// A bare, **unrooted** heap reference, valid only until the next allocation.
/// It borrows `&'h Alloc`, so it cannot be held across a `&mut Alloc`
/// allocation — that conflict is the stale-pointer bug made uncompilable.
///
/// To survive a collection, [`root`](Raw::root) it into a [`RootScope`].
#[derive(Clone, Copy)]
#[must_use = "a Raw is invalidated by the next allocation; root it or use its pointer now"]
pub struct Raw<'h> {
    ptr: *mut u8,
    _alloc: PhantomData<&'h Alloc>,
}

impl<'h> Raw<'h> {
    /// Extract the raw pointer. This severs the token borrow, so the result
    /// is a plain pointer the borrow checker no longer protects: use it
    /// immediately (store it into a freshly-rooted object, return it). Prefer
    /// passing `Raw`/`Rooted` directly where a helper accepts them.
    #[inline]
    pub fn ptr(self) -> *mut u8 {
        self.ptr
    }

    /// True if this is the null reference.
    #[inline]
    pub fn is_null(self) -> bool {
        self.ptr.is_null()
    }

    /// Anchor this reference into `scope`, producing a [`Rooted`] the GC will
    /// keep up to date across allocations.
    #[inline]
    pub fn root<'s>(self, scope: &'s RootScope) -> Rooted<'s> {
        scope.root(self)
    }
}

impl Alloc {
    /// Run `f` at a Rust→runtime boundary with a fresh allocation token and a
    /// root scope. The scope's roots live on the thread's GC scratch stack
    /// (scanned by every collection) and are popped when `f` returns.
    ///
    /// `thread` must be a valid `*mut Thread` whose `dyna_thread` is live.
    ///
    /// ```ignore
    /// Alloc::enter(thread, |a, scope| {
    ///     let arr = a.array_new(n).root(scope);   // survives the loop's allocs
    ///     for item in items {
    ///         let v = a.box_int(build(item));       // &mut a — arr stays rooted
    ///         a.array_set(arr.get(a), i, v);        // re-read relocated arr
    ///     }
    ///     arr.get(a).ptr()
    /// })
    /// ```
    ///
    /// # Safety
    /// `thread` must be the current thread's live `Thread` for the duration
    /// of `f`, and `f` must not retain any `Raw`/`Rooted` past its return.
    pub unsafe fn enter<R>(
        thread: *mut Thread,
        f: impl FnOnce(&mut Alloc, &RootScope<'_>) -> R,
    ) -> R {
        let dyna = unsafe { (*thread).dyna_thread };
        let scope = RootScope {
            dyna,
            mark: unsafe { (*dyna).scratch_mark() },
            _life: PhantomData,
        };
        let mut alloc = Alloc {
            thread,
            _not_send: PhantomData,
        };
        f(&mut alloc, &scope)
    }

    /// Open a nested root scope. Independent of the `&mut self` allocation
    /// borrow (it roots into the thread's scratch stack), so allocation still
    /// works while the scope is open. Popped (scratch reset) on drop — keep
    /// scopes strictly nested (LIFO), like lexical blocks.
    #[inline]
    pub fn scope<'s>(&self) -> RootScope<'s> {
        // The returned scope's lifetime `'s` is independent of this `&self`
        // borrow (it captures the thread's scratch stack by raw pointer), so
        // it does NOT keep `*self` borrowed — allocation through `&mut Alloc`
        // still works while the scope is open. (A `Rooted` minted from it
        // still can't outlive the scope binding: it borrows `&RootScope`.)
        let dyna = unsafe { (*self.thread).dyna_thread };
        RootScope {
            dyna,
            mark: unsafe { (*dyna).scratch_mark() },
            _life: PhantomData,
        }
    }

    /// The underlying `*mut Thread`, for the few interop points that still
    /// need it (passing to JIT'd functions, FFI marshalers).
    #[inline]
    pub fn thread(&self) -> *mut Thread {
        self.thread
    }

    // ─── Allocating ops (take `&mut self`) ──────────────────────────────

    /// Box an `i64` into a `BoxedInt` heap object.
    #[inline]
    pub fn box_int(&mut self, v: i64) -> Raw<'_> {
        self.wrap(unsafe { ai_gc_box_int(self.thread, v) })
    }

    /// Allocate an `n`-slot (null-initialized) `Array`.
    #[inline]
    pub fn array_new(&mut self, n: i64) -> Raw<'_> {
        self.wrap(unsafe { ai_array_new(self.thread, n) })
    }

    /// Allocate a fresh object of the given runtime shape (closure, struct,
    /// or enum variant — `ai_gc_alloc_closure` is generic over heap shapes).
    #[inline]
    pub fn alloc_shape(&mut self, ti: *const TypeInfo) -> Raw<'_> {
        self.wrap(unsafe { ai_gc_alloc_closure(self.thread, ti) })
    }

    /// Allocate a heap `String` holding `bytes`.
    #[inline]
    pub fn str_new(&mut self, bytes: &[u8]) -> Raw<'_> {
        self.wrap(unsafe { ai_str_new(self.thread, bytes.as_ptr(), bytes.len() as i64) })
    }

    /// Allocate a zero-filled `Bytes` of `len` bytes.
    #[inline]
    pub fn bytes_new(&mut self, len: i64) -> Raw<'_> {
        self.wrap(unsafe { ai_bytes_new(self.thread, len) })
    }

    // ─── Non-allocating writes (take `&self`) ───────────────────────────
    //
    // These only store into an already-live object, so they don't collect
    // and can borrow the token immutably — coexisting with the `Raw`s they
    // take as operands.

    /// Store pointer `val` into `arr[i]`.
    #[inline]
    pub fn array_set(&self, arr: Raw<'_>, i: i64, val: Raw<'_>) {
        unsafe { ai_array_set(self.thread, arr.ptr, i, val.ptr) };
    }

    /// Store pointer `val` into `obj` at byte offset `off`.
    #[inline]
    pub fn store_ptr(&self, obj: Raw<'_>, off: usize, val: Raw<'_>) {
        unsafe { *(obj.ptr.add(off) as *mut *mut u8) = val.ptr };
    }

    /// Store the raw 8-byte scalar `v` into `obj` at byte offset `off`.
    #[inline]
    pub fn store_i64(&self, obj: Raw<'_>, off: usize, v: i64) {
        unsafe { *(obj.ptr.add(off) as *mut i64) = v };
    }

    /// Store a 32-bit tag (e.g. an enum variant index) into `obj` at `off`.
    #[inline]
    pub fn store_u32(&self, obj: Raw<'_>, off: usize, v: u32) {
        unsafe { *(obj.ptr.add(off) as *mut u32) = v };
    }

    /// Adopt a pointer produced by an out-of-band allocator (e.g. a
    /// marshaler that already called a runtime constructor) as a `Raw` tied
    /// to this token. The caller asserts `ptr` is a current (not stale) heap
    /// pointer with no allocation since it was produced.
    #[inline]
    pub fn adopt(&self, ptr: *mut u8) -> Raw<'_> {
        self.wrap(ptr)
    }

    #[inline]
    fn wrap(&self, ptr: *mut u8) -> Raw<'_> {
        Raw {
            ptr,
            _alloc: PhantomData,
        }
    }
}

/// A heap reference pinned for the GC. Holding a `Rooted` keeps its scratch
/// slot scannable: a moving collection rewrites the slot in place, and
/// [`get`](Rooted::get) always observes the current (relocated) value.
///
/// Bound to its [`RootScope`] by lifetime — it cannot outlive the scope. The
/// only way to mint one is [`RootScope::root`] / [`Raw::root`]; there is no
/// constructor from a bare pointer, so "I cached the raw pointer" is not
/// expressible.
#[must_use = "a Rooted must be held in a binding to keep the slot rooted"]
#[derive(Clone, Copy)]
pub struct Rooted<'s> {
    dyna: *const ThreadState,
    idx: usize,
    _scope: PhantomData<&'s RootScope<'s>>,
}

impl<'s> Rooted<'s> {
    /// Re-read the current (possibly relocated) reference as a fresh [`Raw`]
    /// re-borrowing `alloc`. Because the result borrows `&Alloc`, the borrow
    /// checker forbids holding it across a `&mut Alloc` allocation — a moving
    /// GC that rewrote the slot is always observed here.
    #[inline]
    pub fn get<'h>(&self, _alloc: &'h Alloc) -> Raw<'h> {
        let ptr = unsafe { (*self.dyna).scratch_at(self.idx) };
        Raw {
            ptr,
            _alloc: PhantomData,
        }
    }

    /// Read the current pointer directly (no token in hand). Use only where
    /// no further allocation can occur before the pointer is consumed —
    /// prefer [`get`](Rooted::get), which is borrow-checked.
    #[inline]
    pub fn ptr_now(&self) -> *mut u8 {
        unsafe { (*self.dyna).scratch_at(self.idx) }
    }
}

/// A LIFO batch of GC roots backed by the thread's scratch stack. Roots
/// pushed here are scanned by every collection; the batch is popped (scratch
/// reset to its opening mark) when the scope drops — including on `?`
/// early-returns, so error paths can't leak roots.
pub struct RootScope<'life> {
    dyna: *const ThreadState,
    mark: usize,
    _life: PhantomData<&'life ()>,
}

impl<'life> RootScope<'life> {
    /// Pin `value` into this scope, returning a [`Rooted`] handle.
    #[inline]
    pub fn root(&self, value: Raw<'_>) -> Rooted<'_> {
        let idx = unsafe { (*self.dyna).push_scratch(value.ptr as *const u8) };
        Rooted {
            dyna: self.dyna,
            idx,
            _scope: PhantomData,
        }
    }
}

impl Drop for RootScope<'_> {
    fn drop(&mut self) {
        unsafe { (*self.dyna).scratch_reset(self.mark) };
    }
}
