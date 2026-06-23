//! Stack capture, abstracted so the platform/strategy can be swapped (the
//! design calls for portable unwinding). The hot path only ever collects raw
//! return addresses into a fixed buffer — no symbolication, no allocation of
//! the result itself. Symbolication happens later, off the hot path, in
//! `memscope-symbols`.

/// Captures return addresses for the current call stack.
///
/// Implementations must be cheap and must not record allocations of their own
/// (callers invoke this inside the recorder reentrancy guard, so any incidental
/// allocation by the unwinder is bypassed rather than re-entering the table).
pub trait Unwind: Send + Sync + 'static {
    /// Fill `out` with return addresses, innermost frame first, skipping the
    /// first `skip` frames (used to drop our own allocator/hook frames).
    /// Returns the number of frames written.
    fn capture(&self, out: &mut [usize], skip: usize) -> usize;
}

/// Default unwinder backed by the `backtrace` crate (libunwind / framepointer
/// depending on platform). Resolves nothing — just walks IPs.
pub struct DefaultUnwind;

impl Unwind for DefaultUnwind {
    #[inline]
    fn capture(&self, out: &mut [usize], skip: usize) -> usize {
        let mut n = 0usize;
        let mut skipped = 0usize;
        backtrace::trace(|frame| {
            if skipped < skip {
                skipped += 1;
                return true;
            }
            if n >= out.len() {
                return false;
            }
            out[n] = frame.ip() as usize;
            n += 1;
            n < out.len()
        });
        n
    }
}
