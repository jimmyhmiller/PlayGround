//! `tokio::select!` analog. Two-arm only; arms must be separated by commas
//! (we patch mini-redis source where it omits them, since macro_rules can't
//! follow an `expr` fragment with a `pat` fragment).
//!
//! The bodies are evaluated in the surrounding async context — not inside
//! the polling closure — so `?`, `return`, and other control-flow inside
//! arm bodies behave the way users expect from `tokio::select!`.

#[doc(hidden)]
pub enum SelectOutcome<A, B> {
    A(A),
    B(B),
}

#[macro_export]
macro_rules! select {
    // Two-arm, both bodies are blocks. Block bodies don't require commas
    // between arms — macro_rules can disambiguate `block` from `pat`.
    (
        $bind1:pat = $fut1:expr => $body1:block
        $bind2:pat = $fut2:expr => $body2:block $(,)?
    ) => {
        $crate::select! {
            $bind1 = $fut1 => $body1,
            $bind2 = $fut2 => $body2,
        }
    };
    // Two-arm with explicit commas; bodies can be any expression.
    (
        $bind1:pat = $fut1:expr => $body1:expr ,
        $bind2:pat = $fut2:expr => $body2:expr $(,)?
    ) => {{
        use ::std::future::Future;
        use ::std::pin::Pin;
        use ::std::task::{Context, Poll};
        let mut __cf_a = $fut1;
        let mut __cf_b = $fut2;
        let mut __cf_a = unsafe { Pin::new_unchecked(&mut __cf_a) };
        let mut __cf_b = unsafe { Pin::new_unchecked(&mut __cf_b) };
        let __cf_outcome = $crate::__private::poll_fn(move |cx: &mut Context<'_>| {
            if let Poll::Ready(__cf_v) = __cf_a.as_mut().poll(cx) {
                return Poll::Ready($crate::macros::SelectOutcome::A(__cf_v));
            }
            if let Poll::Ready(__cf_v) = __cf_b.as_mut().poll(cx) {
                return Poll::Ready($crate::macros::SelectOutcome::B(__cf_v));
            }
            Poll::Pending
        }).await;
        match __cf_outcome {
            $crate::macros::SelectOutcome::A(__cf_v) => {
                let $bind1 = __cf_v;
                $body1
            }
            $crate::macros::SelectOutcome::B(__cf_v) => {
                let $bind2 = __cf_v;
                $body2
            }
        }
    }};
}
