//! Mirrors `tokio::signal`. Only `ctrl_c` is implemented, and it currently
//! never resolves — mini-redis uses it as a "block forever" sentinel and the
//! host tears the runtime down by other means. Real signal handling can be
//! added by integrating signal-hook if needed.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub fn ctrl_c() -> CtrlC {
    CtrlC
}

pub struct CtrlC;

impl Future for CtrlC {
    type Output = std::io::Result<()>;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Never resolves. mini-redis treats this as the shutdown signal; we
        // expect the host to shut down the runtime (worker join exits) when
        // the UI window closes, which terminates the process before this
        // future ever needs to resolve.
        Poll::Pending
    }
}
