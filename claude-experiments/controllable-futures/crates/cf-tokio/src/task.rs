//! Mirrors `tokio::task` — spawn, JoinHandle, task-local storage,
//! `block_in_place`, `spawn_blocking`.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub use cf_runtime::JoinHandle;

/// Tokio-shaped task-local key. Tokio implements task-locals via per-task
/// storage, save/restored at every poll boundary by a `Scope` future.
/// Our implementation is the same shape: a thread-local slot holds the
/// "currently installed" value; `scope().await` is a future that
/// installs/restores it across each poll.
///
/// Tokio's `LocalKey` is the type produced by `task_local!`; we ship a
/// matching macro_rules form below.
pub struct LocalKey<T: 'static> {
    /// Thread-local slot. We store `*mut T` instead of `T` so the value
    /// can be borrowed by the user from inside `try_with` — they get an
    /// `&T` whose lifetime we tie to the closure scope.
    pub __slot: &'static std::thread::LocalKey<RefCell<Option<*mut T>>>,
}

#[derive(Debug)]
pub struct AccessError;

impl std::fmt::Display for AccessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("task-local accessed outside its scope")
    }
}

impl std::error::Error for AccessError {}

impl<T: 'static> LocalKey<T> {
    pub fn scope<F: Future>(&'static self, value: T, fut: F) -> TaskLocalFuture<F, T> {
        TaskLocalFuture {
            key: self,
            value: Some(value),
            fut,
        }
    }

    pub async fn scope_async<F: Future>(&'static self, value: T, fut: F) -> F::Output {
        self.scope(value, fut).await
    }

    /// Synchronous scope: install `value` for the duration of the closure
    /// call. Used by tokio code that needs task-locals without an
    /// `await` (e.g. callbacks called from poll bodies).
    pub fn sync_scope<R, F: FnOnce() -> R>(&'static self, mut value: T, f: F) -> R {
        let prev = self
            .__slot
            .with(|slot| slot.borrow_mut().replace(&mut value as *mut T));
        let r = f();
        self.__slot.with(|slot| {
            *slot.borrow_mut() = prev;
        });
        r
    }

    pub fn try_with<R, F: FnOnce(&T) -> R>(&'static self, f: F) -> Result<R, AccessError> {
        self.__slot.with(|slot| {
            let b = slot.borrow();
            match *b {
                Some(p) => {
                    // SAFETY: the pointer is valid for as long as the scope
                    // future is alive on this thread; we always pop on
                    // poll-exit, so the pointer can't outlive its source.
                    let r = unsafe { &*p };
                    Ok(f(r))
                }
                None => Err(AccessError),
            }
        })
    }

    pub fn with<R, F: FnOnce(&T) -> R>(&'static self, f: F) -> R {
        self.try_with(f).expect("task-local accessed outside its scope")
    }

    pub fn get(&'static self) -> T
    where
        T: Copy,
    {
        self.with(|v| *v)
    }
}

pin_project_lite::pin_project! {
    /// Future produced by `LocalKey::scope`. On every poll it installs the
    /// owned value into the task-local slot, polls the inner future, then
    /// pulls it back out — so the slot is only set during the inner poll.
    pub struct TaskLocalFuture<F, T: 'static> {
        key: &'static LocalKey<T>,
        // Owned during waiting; temporarily lent to the slot during poll.
        value: Option<T>,
        #[pin]
        fut: F,
    }
}

impl<F: Future, T: 'static> Future for TaskLocalFuture<F, T> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let me = self.project();
        let value = me.value.as_mut().expect("polled after completion");
        let key = *me.key;
        let prev = key
            .__slot
            .with(|slot| slot.borrow_mut().replace(value as *mut T));
        let result = me.fut.poll(cx);
        // Restore previous slot value (or clear).
        key.__slot.with(|slot| {
            *slot.borrow_mut() = prev;
        });
        if result.is_ready() {
            *me.value = None;
        }
        result
    }
}

/// `tokio::task_local!`. Defines one or more `LocalKey<T>` statics. Each
/// static gets its own thread-local slot.
#[macro_export]
macro_rules! task_local {
    () => {};
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty; $($rest:tt)*) => {
        $crate::task_local!{ @one $(#[$attr])* $vis $name $t }
        $crate::task_local!{ $($rest)* }
    };
    (@one $(#[$attr:meta])* $vis:vis $name:ident $t:ty) => {
        $(#[$attr])*
        $vis static $name: $crate::task::LocalKey<$t> = {
            ::std::thread_local! {
                static SLOT: ::std::cell::RefCell<::std::option::Option<*mut $t>> =
                    ::std::cell::RefCell::new(::std::option::Option::None);
            }
            $crate::task::LocalKey { __slot: &SLOT }
        };
    };
}

/// `tokio::task::spawn_blocking` analog. We don't have a separate blocking
/// pool; we just run the closure on a fresh OS thread and return a
/// JoinHandle that resolves with its result. Simpler than tokio's pool but
/// equivalent for non-pathological workloads.
pub fn spawn_blocking<F, R>(f: F) -> JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = async_channel::bounded(1);
    std::thread::spawn(move || {
        let r = f();
        let _ = tx.try_send(r);
    });
    let h = cf_runtime::current();
    h.spawn("spawn_blocking", async move {
        rx.recv().await.expect("spawn_blocking thread panicked")
    })
}

/// `tokio::task::block_in_place` analog. In tokio's multi-thread runtime,
/// this hands off the worker's queue to a replacement thread so the
/// blocking call doesn't starve other tasks. We don't do that here — we
/// just run the closure synchronously. For workloads that block briefly,
/// this is fine. For sustained CPU-bound work, the worker is held
/// hostage; users should prefer `spawn_blocking`.
pub fn block_in_place<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

/// `tokio::task::yield_now`. Yields control once.
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

pub struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// `tokio::spawn` analog. Spawns onto the current cf-runtime; panics if no
/// runtime is set on the calling thread.
#[track_caller]
pub fn spawn<F>(fut: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let h = cf_runtime::current();
    let loc = std::panic::Location::caller();
    h.spawn(format!("spawn@{}:{}", loc.file(), loc.line()), fut)
}

// (yield_now defined above)
#[allow(dead_code)]
fn __unused_yield_marker() {
}
