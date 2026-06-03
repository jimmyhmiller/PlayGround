//! Shared continuation operations.
//!
//! Defines the trait boundary between backend-specific frame
//! manipulation (interpreter vs JIT) and the shared heap layer.
//!
//! Two traits:
//!   - `FrameCapture`: serialize live frames → `CapturedStackBuilder`
//!   - `FrameRestorable`: deserialize `ContinuationView` → live frames
//!
//! Three shared functions:
//!   - `do_capture`: extract + heap-allocate
//!   - `do_resume`: heap-read + splice
//!   - `resolve_prompt_boundary`: decide trampoline vs handler
//!
//! The interpreter implements these traits on `InterpStackRuntime`.
//! The JIT implements them on its own frame state types.

use crate::FrameResume;
use crate::cont_heap::{CapturedStackBuilder, ContinuationContext, ContinuationView};

// ─── Traits ────────────────────────────────────────────────────────

/// Backend-specific: how to get frames OUT of the live execution state.
///
/// The interpreter implements this by walking its `Vec<InterpFrame>`.
/// The JIT implements this by reading from `FrameReifyRecord` +
/// suspended frame chain.
pub trait FrameCapture {
    /// Serialize the stack slice from the prompt-owning frame to the
    /// top into a `CapturedStackBuilder`. The builder is
    /// backend-agnostic and feeds directly into `capture_continuation`.
    fn extract_capture(&self, prompt_id: u32, resume_dest: usize) -> CapturedStackBuilder;
}

/// Backend-specific: how to put frames INTO the live execution state.
///
/// The interpreter implements this by pushing `InterpFrame`s onto
/// its stack. The JIT implements this by setting up native re-entry
/// state from the captured frame data.
pub trait FrameRestorable {
    /// Splice captured frames (from a `ContinuationView`) on top of
    /// the current execution state. The bottom frame gets
    /// `bottom_resume` as its caller linkage so that when the
    /// captured computation completes, control returns to the resumer.
    fn splice_restore(
        &mut self,
        view: &ContinuationView<'_>,
        args: &[u64],
        bottom_resume: FrameResume,
    );
}

// ─── Shared functions ──────────────────────────────────────────────

/// Capture a continuation: extract frames from the backend, allocate
/// on the heap. Returns the tagged handle.
///
/// One line of backend-specific code (`extract_capture`), one line
/// of shared heap code (`ctx.capture`).
pub fn do_capture(
    backend: &impl FrameCapture,
    ctx: &dyn ContinuationContext,
    prompt_id: u32,
    resume_dest: usize,
) -> u64 {
    let builder = backend.extract_capture(prompt_id, resume_dest);
    ctx.capture(&builder).expect("OOM capturing continuation")
}

/// Resume a continuation: read from the heap, splice into the backend.
///
/// One line of shared heap code (`ctx.read`), one line of
/// backend-specific code (`splice_restore`).
pub fn do_resume(
    backend: &mut impl FrameRestorable,
    ctx: &dyn ContinuationContext,
    handle: u64,
    args: &[u64],
    return_block: usize,
    return_param_dest: Option<usize>,
) {
    let view = ctx.read(handle).expect("invalid continuation handle");
    backend.splice_restore(
        &view,
        args,
        FrameResume::FromResume {
            return_block,
            return_param_dest,
        },
    );
}

/// Clone a continuation handle. Since the heap representation is
/// immutable, cloning is just returning the same pointer.
pub fn do_clone(handle: u64) -> u64 {
    handle
}

// ─── Prompt boundary resolution ────────────────────────────────────

/// What to do when an abort reaches the prompt-owning frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptBoundaryAction {
    /// The prompt-owning frame is a normal live frame. Run the
    /// handler block with the aborted value.
    RunHandler { value: Option<u64> },
    /// The prompt-owning frame is a spliced captured copy (its resume
    /// is `FromResume`). Trampoline back to the resumer — pop the
    /// captured frame and deliver the value to the resumer's
    /// continuation block.
    TrampolineToResumer {
        return_block: usize,
        return_param_dest: Option<usize>,
        value: Option<u64>,
    },
}

/// Given the prompt-owning frame's `FrameResume`, decide whether to
/// run the handler or trampoline back to the resumer.
///
/// Both the interpreter's `AbortToPrompt` handler and the JIT's
/// abort handler call this to make the shared decision; each then
/// executes the action in its own backend-specific way.
pub fn resolve_prompt_boundary(resume: &FrameResume, value: Option<u64>) -> PromptBoundaryAction {
    match resume {
        FrameResume::FromResume {
            return_block,
            return_param_dest,
        } => PromptBoundaryAction::TrampolineToResumer {
            return_block: *return_block,
            return_param_dest: *return_param_dest,
            value,
        },
        _ => PromptBoundaryAction::RunHandler { value },
    }
}
