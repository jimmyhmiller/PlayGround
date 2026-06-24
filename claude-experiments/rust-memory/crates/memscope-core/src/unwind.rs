//! Stack capture, abstracted so the platform/strategy can be swapped (the
//! design calls for portable unwinding). The hot path only ever collects raw
//! return addresses into a fixed buffer — no symbolication, no allocation of
//! the result itself. Symbolication happens later, off the hot path, in
//! `memscope-symbols`.
//!
//! Two strategies:
//! * [`FramePointerUnwind`] — walks the frame-pointer chain directly (two memory
//!   reads per frame, no CFI). ~10–20× cheaper than libunwind. Correct only when
//!   frame pointers are present, which they always are on aarch64-apple-darwin
//!   (the Apple ABI mandates them) and on x86-64 when built with frame pointers.
//! * [`DefaultUnwind`] — the `backtrace` crate (libunwind). Always correct;
//!   the fallback for platforms / builds without frame pointers.

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

/// Whether frame-pointer unwinding is available for the target architecture.
pub const FRAME_POINTER_SUPPORTED: bool =
    cfg!(any(target_arch = "aarch64", target_arch = "x86_64"));

/// How far above the starting frame pointer we trust the stack to extend. Reads
/// are confined to `[start_fp, start_fp + STACK_WINDOW)` so a corrupt link can't
/// send us dereferencing arbitrary memory.
const STACK_WINDOW: usize = 16 * 1024 * 1024;

/// Frame-pointer unwinder. Reads the current frame pointer and walks the saved
/// `(frame_pointer, return_address)` records up the stack.
///
/// The standard frame record on both aarch64 and x86-64 is two words at the
/// frame pointer: `[fp] = caller_fp`, `[fp + 8] = return_address`. We stop when
/// the chain is null, non-ascending, misaligned, or leaves the trusted window.
pub struct FramePointerUnwind;

impl Unwind for FramePointerUnwind {
    #[inline(never)]
    fn capture(&self, out: &mut [usize], skip: usize) -> usize {
        if out.is_empty() {
            return 0;
        }

        // Read the current frame pointer (x29 on aarch64, rbp on x86-64).
        let mut fp: usize;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("mov {}, x29", out(reg) fp, options(nomem, nostack, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!("mov {}, rbp", out(reg) fp, options(nomem, nostack, preserves_flags));
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            // No frame-pointer support: defer to the backtrace path.
            return DefaultUnwind.capture(out, skip);
        }

        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            let start = fp;
            let hi = start.saturating_add(STACK_WINDOW);
            let mut n = 0usize;
            let mut skipped = 0usize;
            // Bound iterations independently of `out.len()` so a `skip` that
            // exceeds the stack depth still terminates.
            for _ in 0..(out.len() + skip + 8) {
                // Validate the frame pointer before dereferencing it.
                if fp == 0 || fp & (core::mem::size_of::<usize>() - 1) != 0 {
                    break;
                }
                if fp < start || fp >= hi || fp.saturating_add(16) > hi {
                    break;
                }
                // SAFETY: `fp` is word-aligned and inside the trusted stack
                // window; the frame record is two readable words.
                let caller_fp = unsafe { *(fp as *const usize) };
                let ret = unsafe { *((fp + core::mem::size_of::<usize>()) as *const usize) };

                if skipped < skip {
                    skipped += 1;
                } else {
                    out[n] = ret;
                    n += 1;
                    if n >= out.len() {
                        break;
                    }
                }

                // The chain must strictly ascend (the stack grows down, so each
                // caller frame sits at a higher address). Anything else is the
                // top of the chain or corruption — stop.
                if caller_fp <= fp {
                    break;
                }
                fp = caller_fp;
            }
            n
        }
    }
}
