//! Timer primitive built on `async-io`'s reactor. We delegate the heavy
//! lifting (epoll/kqueue, timer wheel) and just expose a cf-runtime-shaped
//! `sleep`. Intercepted by the cf-tokio shim in turn.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

pub struct Sleep {
    inner: async_io::Timer,
    deadline: Instant,
}

impl Sleep {
    pub fn new(dur: Duration) -> Self {
        Self {
            inner: async_io::Timer::after(dur),
            deadline: Instant::now() + dur,
        }
    }

    pub fn until(deadline: Instant) -> Self {
        Self {
            inner: async_io::Timer::at(deadline),
            deadline,
        }
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }
}

impl Future for Sleep {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let deadline = self.deadline;
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.inner) };
        match inner.poll(cx) {
            Poll::Ready(_instant) => Poll::Ready(()),
            Poll::Pending => {
                if let Some(h) = crate::try_current() {
                    let now = Instant::now();
                    let remaining_ms = if deadline > now {
                        (deadline - now).as_millis() as u64
                    } else {
                        0
                    };
                    h.set_wait_reason(crate::WaitReason::Sleep { remaining_ms });
                }
                Poll::Pending
            }
        }
    }
}

pub fn sleep(dur: Duration) -> Sleep {
    Sleep::new(dur)
}

pub fn sleep_until(deadline: Instant) -> Sleep {
    Sleep::until(deadline)
}
