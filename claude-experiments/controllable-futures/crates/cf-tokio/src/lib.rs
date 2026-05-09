//! `cf-tokio` — the controllable-futures tokio compatibility shim.
//!
//! Offers a subset of `tokio`'s public API, just enough to get mini-redis
//! running on `cf-runtime`. Naming, signatures, and module layout mirror
//! `tokio`'s, so application code migrated by replacing `tokio::` with
//! `cf_tokio::` should compile with minimal further changes.
//!
//! Intentionally NOT here: tokio-test utilities, the full `tokio::fs` API,
//! UDP, and select! arms that mini-redis doesn't exercise. If something is
//! missing it should produce a clear compile error telling the reader what
//! to add — not a runtime stub returning a wrong answer.

pub mod io;
pub mod net;
pub mod runtime;
pub mod signal;
pub mod sync;
pub mod task;
pub mod time;

pub use cf_tokio_macros::main;
pub use task::spawn;
// `task_local!` is a macro_rules in `task.rs`; #[macro_export] makes it
// reachable as `cf_tokio::task_local!` automatically.

#[doc(hidden)]
pub mod macros;

/// Re-exports used by the `select!` macro expansion. Hidden — not part of the
/// stable surface, but the macro needs a path it can name from any user
/// crate, even one that doesn't depend on `futures_lite` directly.
#[doc(hidden)]
pub mod __private {
    pub use futures_lite::future::poll_fn;
}

/// Internal helper: stamp the current task's wait_reason just before a
/// shim primitive returns Pending. No-op if not in a task context.
pub(crate) fn note_wait(reason: cf_runtime::WaitReason) {
    if let Some(h) = cf_runtime::try_current() {
        h.set_wait_reason(reason);
    }
}

pin_project_lite::pin_project! {
    /// Wraps any future and tags the current task's `wait_reason` whenever
    /// the inner future returns `Pending`. The closure is re-invoked on
    /// every Pending so it can re-read live state (queue depth, etc.).
    pub(crate) struct WaitTagged<F, R> {
        #[pin]
        inner: F,
        reason_fn: R,
    }
}

impl<F: std::future::Future, R: FnMut() -> cf_runtime::WaitReason> std::future::Future
    for WaitTagged<F, R>
{
    type Output = F::Output;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let me = self.project();
        match me.inner.poll(cx) {
            std::task::Poll::Pending => {
                note_wait((me.reason_fn)());
                std::task::Poll::Pending
            }
            r => r,
        }
    }
}

pub(crate) fn tag_wait<F, R>(inner: F, reason_fn: R) -> WaitTagged<F, R>
where
    F: std::future::Future,
    R: FnMut() -> cf_runtime::WaitReason,
{
    WaitTagged { inner, reason_fn }
}
