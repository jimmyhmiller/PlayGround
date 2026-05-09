//! Mirrors `tokio::time`.

pub use std::time::{Duration, Instant};

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub use cf_runtime::time::Sleep;

pub fn sleep(d: Duration) -> Sleep {
    cf_runtime::time::sleep(d)
}

pub fn sleep_until(deadline: Instant) -> Sleep {
    cf_runtime::time::sleep_until(deadline)
}

/// `tokio::time::timeout` analog. Resolves to `Err(Elapsed)` if the given
/// duration passes before the inner future completes.
pub fn timeout<F: Future>(dur: Duration, fut: F) -> Timeout<F> {
    Timeout {
        sleep: cf_runtime::time::sleep(dur),
        fut,
    }
}

pin_project_lite::pin_project! {
    pub struct Timeout<F> {
        #[pin]
        sleep: Sleep,
        #[pin]
        fut: F,
    }
}

#[derive(Debug)]
pub struct Elapsed;

impl std::fmt::Display for Elapsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("deadline has elapsed")
    }
}

impl std::error::Error for Elapsed {}

impl<F: Future> Future for Timeout<F> {
    type Output = Result<F::Output, Elapsed>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.project();
        if let Poll::Ready(v) = me.fut.poll(cx) {
            return Poll::Ready(Ok(v));
        }
        if let Poll::Ready(()) = me.sleep.poll(cx) {
            return Poll::Ready(Err(Elapsed));
        }
        Poll::Pending
    }
}
