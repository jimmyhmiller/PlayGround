//! Thread-installed host context for JIT extern thunks.
//!
//! JIT-bound externs are bare `extern "C" fn` — they have no `&self`,
//! so any state they need (interned strings, type metadata, monotonic
//! clock origins, etc.) has to live somewhere a free function can find.
//! Without help from the toolkit, every embedder builds its own grab-bag
//! of `thread_local!`s and re-implements the install ceremony.
//!
//! This module provides one TLS slot and a typed accessor on top of it,
//! so a single `BeagleHost`-shaped struct replaces all of those:
//!
//! ```ignore
//! struct BeagleHost {
//!     strings: StringPool,
//!     array_seq: IndexedSeq,
//!     time_origin: std::cell::OnceCell<std::time::Instant>,
//! }
//!
//! let host = BeagleHost::new(...);
//! let _guard = dynlang::host::install_thread(&host);
//!
//! // Inside an extern thunk:
//! extern "C" fn ext_print(val: u64) {
//!     let host = dynlang::host::host::<BeagleHost>();
//!     // ... use host.strings, etc.
//! }
//! ```
//!
//! The slot is single-typed: one host per thread. Embedders can install
//! a different host on each thread, but a thunk must agree with the
//! installer about the type. (In practice, that means using a single
//! `*Host` struct per program.)
//!
//! ## Lifetime caveats
//!
//! [`host`] returns `&'static H`. The lifetime is a convenience — the
//! pointer is valid only while a [`HostGuard`] is in scope. Calling
//! `host` outside that scope is a panic. Storing the returned reference
//! past the guard's lifetime is UB.

use std::cell::Cell;
use std::marker::PhantomData;
use std::ptr;

thread_local! {
    /// Type-erased pointer to the active host. The reader (`host`)
    /// reinterprets it as `&H`; type safety is the embedder's
    /// responsibility (all installers + readers in a program agree on
    /// `H`).
    static HOST_PTR: Cell<*const ()> = const { Cell::new(ptr::null()) };
}

/// Install `host` as the active host on the current thread. The
/// returned guard restores whatever host was previously installed (if
/// any) on drop. Panics if attempting to install the null pointer (a
/// thunk reading the slot would interpret it as "no host").
pub fn install_thread<H>(host: &H) -> HostGuard<'_, H> {
    let p = host as *const H as *const ();
    let prev = HOST_PTR.with(|c| c.replace(p));
    HostGuard {
        prev,
        _phantom: PhantomData,
    }
}

/// Guard returned by [`install_thread`]. On drop, restores whatever
/// host was previously installed (or null if none).
pub struct HostGuard<'a, H> {
    prev: *const (),
    _phantom: PhantomData<&'a H>,
}

impl<H> Drop for HostGuard<'_, H> {
    fn drop(&mut self) {
        HOST_PTR.with(|c| c.set(self.prev));
    }
}

/// Fetch the host installed on the current thread. Panics if no host
/// is installed.
///
/// The returned reference is `&'static H` for ergonomics — it can be
/// used inside extern thunks without lifetime annotations. The
/// lifetime is honored only while a [`HostGuard`] is alive on the
/// thread; storing the reference past the guard's drop is undefined
/// behavior.
///
/// # Safety
/// This function is safe to call. The unsafety hides inside: the type
/// `H` you ask for must match the type the installer used. If they
/// don't, the cast produces an invalid reference and any field access
/// is UB. In practice the embedder uses one `*Host` struct per
/// program; the type parameter is constant.
#[inline]
pub fn host<H>() -> &'static H {
    let p = HOST_PTR.with(|c| c.get());
    assert!(
        !p.is_null(),
        "dynlang::host::host called with no host installed for this thread \
         (call install_thread before transferring control to JIT/extern code)",
    );
    // SAFETY: The pointer was set by `install_thread` from `&H`, and
    // remains valid until the matching `HostGuard` drops. Caller
    // contract: don't call this outside an install scope.
    unsafe { &*(p as *const H) }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHost {
        n: u32,
    }

    #[test]
    fn install_and_read() {
        let h = TestHost { n: 42 };
        let _g = install_thread(&h);
        assert_eq!(host::<TestHost>().n, 42);
    }

    #[test]
    fn guard_restores_previous() {
        let outer = TestHost { n: 1 };
        let _outer_g = install_thread(&outer);
        assert_eq!(host::<TestHost>().n, 1);

        {
            let inner = TestHost { n: 2 };
            let _inner_g = install_thread(&inner);
            assert_eq!(host::<TestHost>().n, 2);
        }

        // _inner_g dropped — outer host is back.
        assert_eq!(host::<TestHost>().n, 1);
    }

    #[test]
    fn no_host_after_guard_drops() {
        {
            let h = TestHost { n: 7 };
            let _g = install_thread(&h);
            assert_eq!(host::<TestHost>().n, 7);
        }
        // Guard dropped; HOST_PTR is back to null.
        HOST_PTR.with(|c| assert!(c.get().is_null()));
    }

    #[test]
    #[should_panic(expected = "no host installed")]
    fn panics_when_uninstalled() {
        // Make sure no other test left a host installed.
        HOST_PTR.with(|c| c.set(ptr::null()));
        let _: &TestHost = host::<TestHost>();
    }
}
