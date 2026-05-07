use std::cell::RefCell;
use std::marker::PhantomData;

use dynalloc::{Heap, PtrPolicy};
use dynexec::{
    BuilderFrame, FrameResume, CapturedStackBuilder,
    ContinuationContext, FrameSliceError,
};
use dynir::interp::{ConfiguredModuleInterpreter, InterpError, InterpResult, InterpRootManager};
use dynlower::{
    FrameReifyKind, FrameReifyRecord, JitFunction, JitModule, JitOutcome,
    SafepointHandlerPayloadKind, SafepointRecord, SuspendedJitFrame, take_suspended_frames,
};
use dynobj::{RootSet, RootSource};

/// Continuation store for the JIT path. Delegates to a
/// `ContinuationContext` (heap-backed) so captured continuations are
/// real GC objects: traced, forwarded during collection, and
/// reclaimed when unreferenced.
///
/// Handles returned from `capture` are heap-tagged pointers (u64),
/// not Vec indices. `read` returns a `ContinuationView` — a zero-copy
/// view into the heap object (one linear scan of the packed metadata).
pub struct JitFrameSliceRuntime<'a> {
    ctx: &'a dyn ContinuationContext,
}

impl<'a> JitFrameSliceRuntime<'a> {
    pub fn new(ctx: &'a dyn ContinuationContext) -> Self {
        JitFrameSliceRuntime { ctx }
    }

    /// Capture a continuation on the heap from a `CapturedStackBuilder`.
    pub fn capture_from_builder(
        &self,
        builder: &CapturedStackBuilder,
    ) -> Result<u64, FrameSliceError> {
        self.ctx
            .capture(builder)
            .ok_or(FrameSliceError::MissingSlice)
    }

    /// Read a previously-captured continuation as a zero-copy view
    /// into the heap object. The view's lifetime is tied to `&self`.
    pub fn read<'s>(&'s self, handle: u64) -> Result<dynexec::ContinuationView<'s>, FrameSliceError> {
        self.ctx
            .read(handle)
            .ok_or(FrameSliceError::MissingSlice)
    }

    /// Clone a continuation handle. Since the heap representation is
    /// immutable, cloning is just returning the same handle.
    pub fn clone_handle(&self, handle: u64) -> u64 {
        handle
    }
}

// JitFrameSliceRuntime no longer needs to be a RootSource — the
// heap objects are traced by scan_object during collection. This
// empty impl satisfies any leftover RootSource bounds.
impl<'a> RootSource for JitFrameSliceRuntime<'a> {
    fn scan_roots(&self, _visitor: &mut dyn FnMut(*mut u64)) {
        // No-op: heap objects are traced by the GC directly.
    }
}

pub trait JitRootTransportRuntime {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind;

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    );
}

pub struct StackMapJitTransport;
pub struct ShadowStackJitTransport;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitFrameControl {
    CaptureSlice {
        record: FrameReifyRecord,
        values: Vec<u64>,
    },
    CloneSlice {
        record: FrameReifyRecord,
        values: Vec<u64>,
    },
    ResumeSlice {
        record: FrameReifyRecord,
        values: Vec<u64>,
    },
    AbortToPrompt {
        record: FrameReifyRecord,
        values: Vec<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitFrameControlError {
    MissingRecord(usize),
    KindMismatch {
        expected: FrameReifyKind,
        actual: FrameReifyKind,
    },
    FrameSlice(FrameSliceError),
    UnsupportedOutcome,
}

#[derive(Debug)]
pub enum ResumeWithInterpreterError {
    FrameSlice(FrameSliceError),
    Interp(InterpError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitExecutionResult {
    Value(u64),
    Void,
    Exception(u64),
    Deopt {
        deopt_id: dynir::DeoptId,
        resume_point: u64,
        live_values: Vec<u64>,
    },
    CaptureSlice {
        handle: u64,
        record: FrameReifyRecord,
    },
    CloneSlice {
        handle: u64,
        record: FrameReifyRecord,
    },
    ResumeSlice {
        handle: u64,
        args: Vec<u64>,
        record: FrameReifyRecord,
    },
    AbortToPrompt {
        prompt: dynir::PromptId,
        values: Vec<u64>,
        record: FrameReifyRecord,
    },
}

pub fn decode_frame_control_outcome(
    outcome: JitOutcome,
    records: &[FrameReifyRecord],
) -> Result<Option<JitFrameControl>, JitFrameControlError> {
    match outcome {
        JitOutcome::CaptureSlice { record_idx, values, .. } => {
            let record = records
                .get(record_idx)
                .cloned()
                .ok_or(JitFrameControlError::MissingRecord(record_idx))?;
            if record.kind != FrameReifyKind::CaptureSlice {
                return Err(JitFrameControlError::KindMismatch {
                    expected: FrameReifyKind::CaptureSlice,
                    actual: record.kind,
                });
            }
            Ok(Some(JitFrameControl::CaptureSlice { record, values }))
        }
        JitOutcome::CloneSlice { record_idx, values, .. } => {
            let record = records
                .get(record_idx)
                .cloned()
                .ok_or(JitFrameControlError::MissingRecord(record_idx))?;
            if record.kind != FrameReifyKind::CloneSlice {
                return Err(JitFrameControlError::KindMismatch {
                    expected: FrameReifyKind::CloneSlice,
                    actual: record.kind,
                });
            }
            Ok(Some(JitFrameControl::CloneSlice { record, values }))
        }
        JitOutcome::ResumeSlice { record_idx, values, .. } => {
            let record = records
                .get(record_idx)
                .cloned()
                .ok_or(JitFrameControlError::MissingRecord(record_idx))?;
            if record.kind != FrameReifyKind::ResumeSlice {
                return Err(JitFrameControlError::KindMismatch {
                    expected: FrameReifyKind::ResumeSlice,
                    actual: record.kind,
                });
            }
            Ok(Some(JitFrameControl::ResumeSlice { record, values }))
        }
        JitOutcome::AbortToPrompt { record_idx, values, .. } => {
            let record = records
                .get(record_idx)
                .cloned()
                .ok_or(JitFrameControlError::MissingRecord(record_idx))?;
            if record.kind != FrameReifyKind::AbortToPrompt {
                return Err(JitFrameControlError::KindMismatch {
                    expected: FrameReifyKind::AbortToPrompt,
                    actual: record.kind,
                });
            }
            Ok(Some(JitFrameControl::AbortToPrompt { record, values }))
        }
        JitOutcome::Value(_)
        | JitOutcome::Void
        | JitOutcome::Exception(_)
        | JitOutcome::Deopt { .. } => Ok(None),
    }
}

/// Build a `CapturedStackBuilder` directly from JIT frame state.
/// This is the JIT's `FrameCapture` — equivalent to the interpreter's
/// `build_captured_stack_impl` but reads from `FrameReifyRecord` +
/// suspended frame chain instead of `InterpFrame`.
fn build_capture_builder(
    record: &FrameReifyRecord,
    values: &[u64],
    mut suspended: Vec<SuspendedJitFrame>,
) -> Result<CapturedStackBuilder, JitFrameControlError> {
    let prompt = record
        .prompt
        .ok_or(JitFrameControlError::UnsupportedOutcome)?;

    // Build the top frame from the reify record.
    let mut frame_values = vec![0u64; record.frame_value_count];
    for (&value_idx, &bits) in record.value_indices.iter().zip(values.iter()) {
        if value_idx >= frame_values.len() {
            return Err(JitFrameControlError::UnsupportedOutcome);
        }
        frame_values[value_idx] = bits;
    }
    let root_indices: Vec<u16> = record
        .root_payload_indices
        .iter()
        .filter_map(|&payload_idx| {
            record.value_indices.get(payload_idx).map(|&vi| vi as u16)
        })
        .collect();
    let resume_arg_slot = record.return_dest.map(|d| d as u32);

    let top_caller_resume = if suspended.is_empty() {
        FrameResume::TopLevel
    } else {
        suspended.last().unwrap().callee_caller_resume.clone()
    };

    let top_builder_frame = BuilderFrame {
        func_idx: record.resume.func_idx as u32,
        block_idx: record.resume.block_idx as u32,
        inst_idx: record.resume.inst_idx as u32,
        values: frame_values,
        active_prompts: record.active_prompts.iter().map(|p| p.index() as u32).collect(),
        root_indices,
        resume_arg_slot,
        caller_resume: top_caller_resume,
    };

    let builder_frames = if suspended.is_empty() {
        vec![top_builder_frame]
    } else {
        let mut frames_rev = vec![top_builder_frame];
        while let Some(mut entry) = suspended.pop() {
            entry.frame.caller_resume = if suspended.is_empty() {
                FrameResume::TopLevel
            } else {
                suspended.last().unwrap().callee_caller_resume.clone()
            };
            frames_rev.push(entry.frame);
        }
        frames_rev.reverse();
        frames_rev
    };

    Ok(CapturedStackBuilder {
        prompt_id: prompt.index() as u32,
        frames: builder_frames,
    })
}

pub fn materialize_capture_slice(
    store: &JitFrameSliceRuntime,
    record: &FrameReifyRecord,
    values: &[u64],
) -> Result<u64, JitFrameControlError> {
    let builder = build_capture_builder(record, values, Vec::new())?;
    store
        .capture_from_builder(&builder)
        .map_err(|_| JitFrameControlError::UnsupportedOutcome)
}

pub fn resume_stored_slice_with_interpreter<'a, Cfg, R>(
    interpreter: &ConfiguredModuleInterpreter<'a, Cfg, R>,
    store: &JitFrameSliceRuntime,
    handle: u64,
    args: &[u64],
) -> Result<InterpResult, ResumeWithInterpreterError>
where
    Cfg: dynexec::CodegenConfig,
    Cfg::Layout: dynexec::LayoutConfigDefaults,
    R: InterpRootManager<Cfg::Layout, Cfg::Roots, Cfg::RootTransport>,
{
    let view = store
        .read(handle)
        .map_err(ResumeWithInterpreterError::FrameSlice)?;
    interpreter
        .resume_view(&view, args)
        .map_err(ResumeWithInterpreterError::Interp)
}

pub fn resume_stored_slice_with_jit(
    jit: &JitFunction,
    _record: &FrameReifyRecord,
    store: &JitFrameSliceRuntime,
    handle: u64,
    args: &[u64],
) -> Result<JitOutcome, ResumeWithInterpreterError> {
    let view = store
        .read(handle)
        .map_err(ResumeWithInterpreterError::FrameSlice)?;
    if view.frame_count() == 0 {
        return Err(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice));
    }
    let frame = view.frame(view.frame_count() - 1);
    let resume_point = dynexec::FrameResumePoint {
        func_idx: frame.func_idx as usize,
        block_idx: frame.block_idx as usize,
        inst_idx: frame.inst_idx as usize,
    };
    let capture_record = jit
        .frame_reify_records()
        .iter()
        .find(|record| {
            record.kind == FrameReifyKind::CaptureSlice
                && record.native_resume_offset.is_some()
                && record.resume == resume_point
        })
        .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
    Ok(jit.call_resume_outcome(capture_record, frame.values.as_ptr(), args))
}

pub fn resume_stored_slice_with_jit_module(
    jit: &JitModule,
    _record: &FrameReifyRecord,
    store: &JitFrameSliceRuntime,
    handle: u64,
    args: &[u64],
) -> Result<JitOutcome, ResumeWithInterpreterError> {
    let view = store
        .read(handle)
        .map_err(ResumeWithInterpreterError::FrameSlice)?;
    let mut frame_idx = view
        .frame_count()
        .checked_sub(1)
        .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;

    let frame_resume_point = |i: usize| {
        let f = view.frame(i);
        dynexec::FrameResumePoint {
            func_idx: f.func_idx as usize,
            block_idx: f.block_idx as usize,
            inst_idx: f.inst_idx as usize,
        }
    };

    let call_resume = |i: usize, args: &[u64]| -> Option<JitOutcome> {
        let f = view.frame(i);
        jit.call_view_resume_outcome(&frame_resume_point(i), f.values, args)
    };
    let call_invoke = |i: usize, is_exc: bool, args: &[u64]| -> Option<JitOutcome> {
        let f = view.frame(i);
        jit.call_view_invoke_resume_outcome(&frame_resume_point(i), f.values, is_exc, args)
    };

    let missing = || ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice);

    let mut outcome = call_resume(frame_idx, args).ok_or_else(missing)?;

    loop {
        let caller = view.frame(frame_idx).caller_resume.clone();
        match (&caller, outcome.clone()) {
            (FrameResume::TopLevel, final_outcome) => return Ok(final_outcome),
            (FrameResume::FromCall { .. }, JitOutcome::Value(v)) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Value(v));
                }
                frame_idx -= 1;
                outcome = call_resume(frame_idx, &[v]).ok_or_else(missing)?;
            }
            (FrameResume::FromCall { .. }, JitOutcome::Void) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Void);
                }
                frame_idx -= 1;
                outcome = call_resume(frame_idx, &[]).ok_or_else(missing)?;
            }
            (FrameResume::FromCall { .. }, JitOutcome::Exception(exc)) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Exception(exc));
                }
                frame_idx -= 1;
                continue;
            }
            (FrameResume::FromInvoke { .. }, JitOutcome::Value(v)) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Value(v));
                }
                frame_idx -= 1;
                outcome = call_invoke(frame_idx, false, &[v]).ok_or_else(missing)?;
            }
            (FrameResume::FromInvoke { .. }, JitOutcome::Void) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Void);
                }
                frame_idx -= 1;
                outcome = call_invoke(frame_idx, false, &[]).ok_or_else(missing)?;
            }
            (FrameResume::FromInvoke { .. }, JitOutcome::Exception(exc)) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Exception(exc));
                }
                frame_idx -= 1;
                outcome = call_invoke(frame_idx, true, &[exc]).ok_or_else(missing)?;
            }
            (_, other) => return Ok(other),
        }
    }
}

fn continue_outcome_with_function(
    jit: &JitFunction,
    outcome: JitOutcome,
    store: &JitFrameSliceRuntime,
) -> Result<JitExecutionResult, JitFrameControlError> {
    if let Some(control) = decode_frame_control_outcome(outcome.clone(), jit.frame_reify_records())? {
        return match control {
            JitFrameControl::CaptureSlice { record, values } => {
                let builder = build_capture_builder(&record, &values, take_suspended_frames())?;
                let handle = store
                    .capture_from_builder(&builder)
                    .map_err(|_| JitFrameControlError::UnsupportedOutcome)?;
                Ok(JitExecutionResult::CaptureSlice { handle, record })
            }
            JitFrameControl::CloneSlice { record, values } => {
                let source_idx = *record
                    .control_value_indices
                    .first()
                    .ok_or(JitFrameControlError::UnsupportedOutcome)?;
                let Some(source_bits) = values.get(source_idx).copied() else {
                    return Err(JitFrameControlError::UnsupportedOutcome);
                };
                let handle = source_bits;
                let cloned = store.clone_handle(handle);
                if record.native_resume_offset.is_some() {
                    return continue_outcome_with_function(
                        jit,
                        jit.call_resume_outcome(&record, values.as_ptr(), &[cloned]),
                        store,
                    );
                }
                Ok(JitExecutionResult::CloneSlice {
                    handle: cloned,
                    record,
                })
            }
            JitFrameControl::ResumeSlice { record, values } => {
                let (slice_bits, args) = values
                    .split_first()
                    .ok_or(JitFrameControlError::UnsupportedOutcome)?;
                let handle = *slice_bits;
                return continue_outcome_with_function(
                    jit,
                    resume_stored_slice_with_jit(jit, &record, store, handle, args)
                        .map_err(|err| match err {
                            ResumeWithInterpreterError::FrameSlice(err) => {
                                JitFrameControlError::FrameSlice(err)
                            }
                            ResumeWithInterpreterError::Interp(_) => {
                                JitFrameControlError::UnsupportedOutcome
                            }
                        })?,
                    store,
                );
            }
            JitFrameControl::AbortToPrompt { record, values } => Ok(
                {
                    let _ = take_suspended_frames();
                    JitExecutionResult::AbortToPrompt {
                        prompt: record.prompt.expect("abort_to_prompt record missing prompt"),
                        values,
                        record,
                    }
                },
            ),
        };
    }

    let _ = take_suspended_frames();
    Ok(match outcome {
        JitOutcome::Value(v) => JitExecutionResult::Value(v),
        JitOutcome::Void => JitExecutionResult::Void,
        JitOutcome::Exception(v) => JitExecutionResult::Exception(v),
        JitOutcome::Deopt {
            deopt_id,
            resume_point,
            live_values,
        } => JitExecutionResult::Deopt {
            deopt_id,
            resume_point,
            live_values,
        },
        JitOutcome::CaptureSlice { .. }
        | JitOutcome::CloneSlice { .. }
        | JitOutcome::ResumeSlice { .. }
        | JitOutcome::AbortToPrompt { .. } => unreachable!(),
    })
}

pub fn execute_jit_function(
    jit: &JitFunction,
    args: &[u64],
    store: &JitFrameSliceRuntime,
) -> Result<JitExecutionResult, JitFrameControlError> {
    execute_outcome(jit.call_outcome(args), jit.frame_reify_records(), store)
}

pub fn execute_jit_function_to_terminal(
    jit: &JitFunction,
    args: &[u64],
    store: &JitFrameSliceRuntime,
) -> Result<JitExecutionResult, JitFrameControlError> {
    continue_outcome_with_function(jit, jit.call_outcome(args), store)
}

pub fn execute_jit_module_function(
    jit: &JitModule,
    func_ref: dynir::FuncRef,
    args: &[u64],
    store: &JitFrameSliceRuntime,
) -> Result<JitExecutionResult, JitFrameControlError> {
    continue_outcome_with_module(
        jit,
        jit.frame_reify_records_for_function(func_ref.index()),
        jit.call_outcome(func_ref, args),
        store,
    )
}

fn continue_outcome_with_module(
    jit: &JitModule,
    records: &[FrameReifyRecord],
    outcome: JitOutcome,
    store: &JitFrameSliceRuntime,
) -> Result<JitExecutionResult, JitFrameControlError> {
    let records = match &outcome {
        JitOutcome::CaptureSlice { func_idx, .. }
        | JitOutcome::CloneSlice { func_idx, .. }
        | JitOutcome::ResumeSlice { func_idx, .. }
        | JitOutcome::AbortToPrompt { func_idx, .. } => {
            jit.frame_reify_records_for_function(*func_idx)
        }
        _ => records,
    };
    if let Some(control) = decode_frame_control_outcome(outcome.clone(), records)? {
        return match control {
            JitFrameControl::CaptureSlice { record, values } => {
                let builder = build_capture_builder(&record, &values, take_suspended_frames())?;
                let handle = store
                    .capture_from_builder(&builder)
                    .map_err(|_| JitFrameControlError::UnsupportedOutcome)?;
                Ok(JitExecutionResult::CaptureSlice { handle, record })
            }
            JitFrameControl::CloneSlice { record, values } => {
                let source_idx = *record
                    .control_value_indices
                    .first()
                    .ok_or(JitFrameControlError::UnsupportedOutcome)?;
                let Some(source_bits) = values.get(source_idx).copied() else {
                    return Err(JitFrameControlError::UnsupportedOutcome);
                };
                let handle = source_bits;
                let cloned = store.clone_handle(handle);
                if record.native_resume_offset.is_some() {
                    let next = jit.call_resume_outcome(
                        record.resume.func_idx,
                        &record,
                        values.as_ptr(),
                        &[cloned],
                    );
                    return continue_outcome_with_module(
                        jit,
                        jit.frame_reify_records_for_function(record.resume.func_idx),
                        next,
                        store,
                    );
                }
                Ok(JitExecutionResult::CloneSlice {
                    handle: cloned,
                    record,
                })
            }
            JitFrameControl::ResumeSlice { record, values } => {
                let (slice_bits, args) = values
                    .split_first()
                    .ok_or(JitFrameControlError::UnsupportedOutcome)?;
                let handle = *slice_bits;
                let next = resume_stored_slice_with_jit_module(jit, &record, store, handle, args)
                    .map_err(|err| match err {
                        ResumeWithInterpreterError::FrameSlice(err) => {
                            JitFrameControlError::FrameSlice(err)
                        }
                        ResumeWithInterpreterError::Interp(_) => {
                            JitFrameControlError::UnsupportedOutcome
                        }
                    })?;
                continue_outcome_with_module(
                    jit,
                    jit.frame_reify_records_for_function(record.resume.func_idx),
                    next,
                    store,
                )
            }
            JitFrameControl::AbortToPrompt { record, values } => Ok(
                JitExecutionResult::AbortToPrompt {
                    prompt: record.prompt.expect("abort_to_prompt record missing prompt"),
                    values,
                    record,
                },
            ),
        };
    }

    Ok(match outcome {
        JitOutcome::Value(v) => JitExecutionResult::Value(v),
        JitOutcome::Void => JitExecutionResult::Void,
        JitOutcome::Exception(v) => JitExecutionResult::Exception(v),
        JitOutcome::Deopt {
            deopt_id,
            resume_point,
            live_values,
        } => JitExecutionResult::Deopt {
            deopt_id,
            resume_point,
            live_values,
        },
        JitOutcome::CaptureSlice { .. }
        | JitOutcome::CloneSlice { .. }
        | JitOutcome::ResumeSlice { .. }
        | JitOutcome::AbortToPrompt { .. } => unreachable!(),
    })
}

pub fn execute_jit_module_function_to_terminal(
    jit: &JitModule,
    func_ref: dynir::FuncRef,
    args: &[u64],
    store: &JitFrameSliceRuntime,
) -> Result<JitExecutionResult, JitFrameControlError> {
    continue_outcome_with_module(
        jit,
        jit.frame_reify_records_for_function(func_ref.index()),
        jit.call_outcome(func_ref, args),
        store,
    )
}

fn execute_outcome(
    outcome: JitOutcome,
    records: &[FrameReifyRecord],
    store: &JitFrameSliceRuntime,
) -> Result<JitExecutionResult, JitFrameControlError> {
    if let Some(control) = decode_frame_control_outcome(outcome.clone(), records)? {
        return match control {
            JitFrameControl::CaptureSlice { record, values } => {
                let handle = materialize_capture_slice(store, &record, &values)?;
                Ok(JitExecutionResult::CaptureSlice { handle, record })
            }
            JitFrameControl::CloneSlice { record, values } => {
                let handle = *values.first().ok_or(
                    JitFrameControlError::UnsupportedOutcome,
                )?;
                let cloned = store.clone_handle(handle);
                Ok(JitExecutionResult::CloneSlice {
                    handle: cloned,
                    record,
                })
            }
            JitFrameControl::ResumeSlice { record, values } => {
                let (slice_bits, args) = values
                    .split_first()
                    .ok_or(JitFrameControlError::UnsupportedOutcome)?;
                let handle = *slice_bits;
                Ok(JitExecutionResult::ResumeSlice {
                    handle,
                    args: args.to_vec(),
                    record,
                })
            }
            JitFrameControl::AbortToPrompt { record, values } => Ok(
                JitExecutionResult::AbortToPrompt {
                    prompt: record.prompt.expect("abort_to_prompt record missing prompt"),
                    values,
                    record,
                },
            ),
        };
    }

    Ok(match outcome {
        JitOutcome::Value(v) => JitExecutionResult::Value(v),
        JitOutcome::Void => JitExecutionResult::Void,
        JitOutcome::Exception(v) => JitExecutionResult::Exception(v),
        JitOutcome::Deopt {
            deopt_id,
            resume_point,
            live_values,
        } => JitExecutionResult::Deopt {
            deopt_id,
            resume_point,
            live_values,
        },
        JitOutcome::CaptureSlice { .. }
        | JitOutcome::CloneSlice { .. }
        | JitOutcome::ResumeSlice { .. }
        | JitOutcome::AbortToPrompt { .. } => unreachable!(),
    })
}

impl JitRootTransportRuntime for StackMapJitTransport {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        SafepointHandlerPayloadKind::SafepointIndex
    }

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    ) {
        let record = safepoints
            .get(payload)
            .unwrap_or_else(|| panic!("missing stack-map safepoint record {payload}"));
        let root_source = SlotOffsetRootSource {
            frame_ptr,
            slot_offsets: &record.root_slots,
        };
        root_source.scan_roots(visitor);
    }
}

impl JitRootTransportRuntime for ShadowStackJitTransport {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        SafepointHandlerPayloadKind::SafepointIndex
    }

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    ) {
        let record = safepoints
            .get(payload)
            .unwrap_or_else(|| panic!("missing shadow-stack safepoint record {payload}"));
        let root_source = SlotOffsetRootSource {
            frame_ptr,
            slot_offsets: &record.root_slots,
        };
        root_source.scan_roots(visitor);
    }
}

struct SlotOffsetRootSource<'a> {
    frame_ptr: *mut u8,
    slot_offsets: &'a [i32],
}

impl RootSource for SlotOffsetRootSource<'_> {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for &offset in self.slot_offsets {
            let slot = unsafe { self.frame_ptr.add(offset as usize).cast::<u64>() };
            visitor(slot);
        }
    }
}

struct RuntimeRootSource<'a, T: JitRootTransportRuntime> {
    transport: &'a T,
    frame_ptr: *mut u8,
    payload: usize,
    safepoints: &'a [SafepointRecord],
}

impl<T: JitRootTransportRuntime> RootSource for RuntimeRootSource<'_, T> {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        unsafe {
            self.transport
                .scan_roots(self.frame_ptr, self.payload, self.safepoints, visitor);
        }
    }
}

/// When the safepoint handler should run a collection.
///
/// The handler runs at every JIT safepoint (entry to allocation, loop
/// backedge, etc.) once a session is installed; this enum controls
/// whether each invocation actually triggers a collection cycle.
///
/// Replaces the older `gc_threshold: f64` API where `0.0` meant "every
/// point" and a positive fraction meant "only past that usage" — that
/// encoding made the stress contract (every safepoint) the default,
/// which surprised frontends.
#[derive(Debug, Clone, Copy)]
pub enum GcPolicy {
    /// Don't auto-collect at safepoints. The frontend triggers
    /// collection itself (e.g. `gc.collect()` between top-level forms,
    /// or never — the heap is large enough for the workload).
    NeverAuto,
    /// Collect when from-space usage reaches `threshold` (0.0..=1.0).
    /// `0.75` is a reasonable default for a copying collector.
    OnPressure { threshold: f64 },
    /// Collect at every safepoint. The GC-stress contract: useful for
    /// testing root coverage but several orders of magnitude slower
    /// than `OnPressure`. Don't ship.
    EveryPoint,
}

impl GcPolicy {
    /// Decide whether to collect now given current heap usage.
    pub(crate) fn should_collect(&self, used: usize, space_size: usize) -> bool {
        match self {
            GcPolicy::NeverAuto => false,
            GcPolicy::EveryPoint => true,
            GcPolicy::OnPressure { threshold } => {
                let usage = used as f64 / space_size as f64;
                usage >= *threshold
            }
        }
    }
}

pub struct JitSafepointSession<'a, P: PtrPolicy, T: JitRootTransportRuntime> {
    heap: &'a Heap,
    transport: T,
    safepoints: &'a [SafepointRecord],
    gc_policy: GcPolicy,
    extra_roots: RefCell<Vec<*const dyn RootSource>>,
    _policy: PhantomData<P>,
}

#[derive(Clone, Copy)]
struct ActiveJitSafepointSession {
    ptr: *const (),
    handle: unsafe fn(*const (), *mut u8, usize),
    push_root_source: unsafe fn(*const (), *const dyn RootSource),
    pop_root_source: unsafe fn(*const ()),
}

impl<'a, P: PtrPolicy, T: JitRootTransportRuntime> JitSafepointSession<'a, P, T> {
    /// Create a session that never auto-collects at safepoints. The
    /// frontend is expected to either call `gc.collect()` at its own
    /// cadence, or use `with_gc_policy` to opt into pressure-based or
    /// stress-mode collection.
    pub fn new(heap: &'a Heap, transport: T, safepoints: &'a [SafepointRecord]) -> Self {
        Self {
            heap,
            transport,
            safepoints,
            gc_policy: GcPolicy::NeverAuto,
            extra_roots: RefCell::new(Vec::new()),
            _policy: PhantomData,
        }
    }

    /// Set the collection policy for this session. See [`GcPolicy`].
    pub fn with_gc_policy(mut self, policy: GcPolicy) -> Self {
        self.gc_policy = policy;
        self
    }

    /// Register an additional `RootSource` to be scanned at every collection
    /// during this session. Use this for module-scoped roots that aren't
    /// frame-resident (literal pools, global root sets, interned constants).
    ///
    /// # Safety
    ///
    /// `source` must remain valid for the lifetime of the session. The
    /// session does not take ownership; the caller is responsible for
    /// keeping the pointee alive (e.g., via `&'a` shared borrow).
    pub unsafe fn register_extra_root(&self, source: *const dyn RootSource) {
        self.extra_roots.borrow_mut().push(source);
    }

    pub fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        self.transport.payload_kind()
    }

    pub fn with_installed<R>(&self, f: impl FnOnce() -> R) -> R {
        ACTIVE_JIT_SAFEPOINT_SESSION.with(|cell| {
            let previous = {
                let mut slot = cell.borrow_mut();
                slot.replace(ActiveJitSafepointSession {
                    ptr: self as *const Self as *const (),
                    handle: Self::handle_erased,
                    push_root_source: Self::push_root_source_erased,
                    pop_root_source: Self::pop_root_source_erased,
                })
            };
            let result = f();
            let mut slot = cell.borrow_mut();
            *slot = previous;
            result
        })
    }

    unsafe fn handle_erased(ptr: *const (), frame_ptr: *mut u8, payload: usize) {
        let session = unsafe { &*(ptr as *const Self) };
        unsafe {
            session.handle(frame_ptr, payload);
        }
    }

    unsafe fn push_root_source_erased(ptr: *const (), source: *const dyn RootSource) {
        let session = unsafe { &*(ptr as *const Self) };
        session.extra_roots.borrow_mut().push(source);
    }

    unsafe fn pop_root_source_erased(ptr: *const ()) {
        let session = unsafe { &*(ptr as *const Self) };
        session.extra_roots.borrow_mut().pop();
    }

    unsafe fn handle(&self, frame_ptr: *mut u8, payload: usize) {
        if !self
            .gc_policy
            .should_collect(self.heap.from_used(), self.heap.space_size())
        {
            return;
        }
        let root_source = RuntimeRootSource {
            transport: &self.transport,
            frame_ptr,
            payload,
            safepoints: self.safepoints,
        };
        // Walk ancestor JIT frames via the native FP chain. The current
        // frame is already scanned by root_source above; the FP walker
        // scans all callers whose return addresses fall in registered
        // JIT code ranges.
        let ancestor_roots = dynlower::JitFrameRoots { jit_fp: frame_ptr };
        let extra_roots = self.extra_roots.borrow();
        let mut sources: Vec<&dyn RootSource> = vec![&root_source, &ancestor_roots];
        for &ptr in extra_roots.iter() {
            sources.push(unsafe { &*ptr });
        }
        unsafe {
            self.heap.collect::<P>(&sources);
        }
    }
}

thread_local! {
    static ACTIVE_JIT_SAFEPOINT_SESSION: RefCell<Option<ActiveJitSafepointSession>> =
        RefCell::new(None);
}

pub extern "C" fn active_jit_safepoint_handler(frame_ptr: *mut u8, payload: usize) {
    ACTIVE_JIT_SAFEPOINT_SESSION.with(|cell| {
        let session = (*cell.borrow()).expect("no active JIT safepoint session installed");
        unsafe {
            (session.handle)(session.ptr, frame_ptr, payload);
        }
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopedJitRoot(usize);

/// A small dynamic root set for native/JIT helpers that need to keep boxed
/// values alive across allocations inside a moving-GC safepoint session.
pub struct ScopedJitRoots {
    roots: RootSet,
}

impl ScopedJitRoots {
    pub fn new() -> Self {
        Self { roots: RootSet::new() }
    }

    pub fn push(&mut self, bits: u64) -> ScopedJitRoot {
        ScopedJitRoot(self.roots.add(bits))
    }

    pub fn get(&self, root: ScopedJitRoot) -> u64 {
        self.roots.get(root.0)
    }

    pub fn with_active<R>(&self, f: impl FnOnce() -> R) -> R {
        with_registered_active_jit_roots(&self.roots, f)
    }
}

struct ActiveRootSourceGuard {
    session: ActiveJitSafepointSession,
}

impl Drop for ActiveRootSourceGuard {
    fn drop(&mut self) {
        unsafe {
            (self.session.pop_root_source)(self.session.ptr);
        }
    }
}

pub fn with_registered_active_jit_roots<R>(
    source: &dyn RootSource,
    f: impl FnOnce() -> R,
) -> R {
    ACTIVE_JIT_SAFEPOINT_SESSION.with(|cell| {
        let maybe_session = *cell.borrow();
        let Some(session) = maybe_session else {
            return f();
        };
        // SAFETY: the root source reference remains live for the full dynamic
        // extent of this function call, and the guard below unregisters it
        // before we return.
        let source_ptr: *const dyn RootSource = unsafe {
            std::mem::transmute::<*const dyn RootSource, *const dyn RootSource>(
                source as *const dyn RootSource,
            )
        };
        unsafe {
            (session.push_root_source)(session.ptr, source_ptr);
        }
        let _guard = ActiveRootSourceGuard { session };
        f()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use dynalloc::Heap;
    use dynexec::FrameResumePoint;
    use dynir::builder::{FunctionBuilder, ModuleBuilder};
    use dynir::interp::ModuleInterpreter;
    use dynir::Module;
    use dynir::types::Type;
    use dynlower::FrameReifyKind;
    use dynobj::Compact;
    use dynvalue::NanBox;
    // Tests use JitFrameSliceRuntime (imported via super::*)

    /// Build a GcInterpCtx suitable for continuation tests.
    fn make_cont_ctx() -> dynir::gc_runtime::GcInterpCtx<dynobj::Compact, TestPol> {
        use dynalloc::SemiSpace;
        use dynexec::ContinuationTypes;

        let mut type_table = Vec::new();
        let cont_types = ContinuationTypes::register_into::<dynobj::Compact>(&mut type_table);
        let heap = SemiSpace::new::<dynobj::Compact>(64 * 1024);
        dynir::gc_runtime::GcInterpCtx::<dynobj::Compact, TestPol>::new(heap, type_table, cont_types)
    }

    struct TestPol;
    impl dynalloc::PtrPolicy for TestPol {
        fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
            if bits == 0 { return None; }
            if bits & 0b111 == 0 { Some(bits as *mut u8) } else { None }
        }
        fn encode_ptr(ptr: *mut u8) -> u64 { ptr as u64 }
    }

    #[cfg(target_arch = "aarch64")]
    core::arch::global_asm!(
        ".globl _dynruntime_test_throw_exception_stub",
        "_dynruntime_test_throw_exception_stub:",
        "mov x1, #2",
        "mov x2, #1234",
        "ret",
    );

    #[cfg(target_arch = "aarch64")]
    unsafe extern "C" {
        fn dynruntime_test_throw_exception_stub();
    }

    #[test]
    fn metadata_transport_scans_only_recorded_slots() {
        let mut frame = [0u64; 6];
        frame[2] = 33;
        frame[4] = 55;
        let safepoints = [SafepointRecord {
            code_offset: 0,
            return_offset: 0,
            root_slots: vec![16, 32],
        }];

        let mut seen = Vec::new();
        unsafe {
            StackMapJitTransport.scan_roots(
                frame.as_mut_ptr().cast::<u8>(),
                0,
                &safepoints,
                &mut |slot| seen.push(*slot),
            );
        }
        assert_eq!(seen, vec![33, 55]);
    }

    #[test]
    fn decode_capture_frame_control_outcome_uses_record() {
        let record = FrameReifyRecord {
            kind: FrameReifyKind::CaptureSlice,
            prompt: Some(dynir::PromptId::from_index(3)),
            active_prompts: vec![dynir::PromptId::from_index(1), dynir::PromptId::from_index(3)],
            resume: FrameResumePoint {
                func_idx: 4,
                block_idx: 5,
                inst_idx: 6,
            },
            native_resume_offset: None,
            frame_value_count: 8,
            value_indices: vec![0, 2],
            control_value_indices: vec![],
            value_types: vec![Type::I64, Type::GcPtr],
            root_payload_indices: vec![1],
            return_dest: Some(7),
        };

        let decoded = decode_frame_control_outcome(
            JitOutcome::CaptureSlice {
                func_idx: 0,
                record_idx: 0,
                values: vec![11, 22],
            },
            &[record.clone()],
        )
        .unwrap();

        assert_eq!(
            decoded,
            Some(JitFrameControl::CaptureSlice {
                record,
                values: vec![11, 22],
            })
        );
    }

    #[test]
    fn materialize_capture_slice_inserts_snapshot() {
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);
        let record = FrameReifyRecord {
            kind: FrameReifyKind::CaptureSlice,
            prompt: Some(dynir::PromptId::from_index(1)),
            active_prompts: vec![dynir::PromptId::from_index(0), dynir::PromptId::from_index(1)],
            resume: FrameResumePoint {
                func_idx: 2,
                block_idx: 3,
                inst_idx: 4,
            },
            native_resume_offset: None,
            frame_value_count: 32,
            value_indices: vec![10, 20],
            control_value_indices: vec![],
            value_types: vec![Type::I64, Type::GcPtr],
            root_payload_indices: vec![1],
            return_dest: Some(30),
        };

        let handle = materialize_capture_slice(&store, &record, &[88, 99]).unwrap();
        let view = store.read(handle).unwrap();
        assert_eq!(view.prompt_id(), 1);
        assert_eq!(view.frame_count(), 1);
        let f = view.frame(0);
        assert_eq!(f.func_idx, record.resume.func_idx as u32);
        assert_eq!(f.block_idx, record.resume.block_idx as u32);
        assert_eq!(f.inst_idx, record.resume.inst_idx as u32);
        assert_eq!(f.values.len(), 32);
        assert_eq!(f.values[10], 88);
        assert_eq!(f.values[20], 99);
        assert_eq!(f.root_indices, &[20u16]);
        assert_eq!(f.resume_arg_slot, Some(30));
        assert_eq!(f.active_prompts, &[0u32, 1]);
    }

    #[test]
    fn execute_jit_function_materializes_capture_slice() {
        let mut b = FunctionBuilder::new("capture", &[Type::I64], Some(Type::FrameSlice));
        let entry = b.entry_block();
        let arg = b.block_param(entry, 0);
        let handler_bb = b.create_block(&[Type::FrameSlice]);
        let prompt = b.create_prompt();
        b.push_prompt(prompt, handler_bb);
        let slice = b.capture_slice(prompt, &[arg]);
        b.pop_prompt(prompt);
        b.jump(handler_bb, &[slice]);
        b.switch_to_block(handler_bb);
        let _handler_param = b.block_param(handler_bb, 0);
        b.unreachable();
        let func = b.build();

        let jit = JitFunction::compile::<NanBox>(&func, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);
        let result = execute_jit_function(&jit, &[123], &store).unwrap();

        match result {
            JitExecutionResult::CaptureSlice { handle, record } => {
                assert_eq!(record.kind, FrameReifyKind::CaptureSlice);
                let view = store.read(handle).unwrap();
                assert_eq!(view.frame_count(), 1);
                let f = view.frame(0);
                assert_eq!(f.values[arg.index()], 123);
                assert_eq!(f.resume_arg_slot, Some(slice.index() as u32));
            }
            other => panic!("expected capture execution result, got {other:?}"),
        }
    }

    #[test]
    fn execute_jit_module_function_surfaces_abort_to_prompt() {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));
        let mut fb = mb.define_func(f_main);
        let entry = fb.entry_block();
        let arg = fb.block_param(entry, 0);
        let prompt = fb.create_prompt();
        fb.abort_to_prompt(prompt, &[arg]);
        mb.finish_func(f_main, fb);
        let module = mb.build();

        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);
        let result = execute_jit_module_function(&jit, f_main, &[77], &store).unwrap();

        match result {
            JitExecutionResult::AbortToPrompt { prompt, values, record } => {
                assert_eq!(Some(prompt), record.prompt);
                assert_eq!(values, vec![77]);
            }
            other => panic!("expected abort execution result, got {other:?}"),
        }
    }

    #[test]
    fn execute_jit_module_function_materializes_multi_frame_capture() {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[], Some(Type::FrameSlice));
        let f_inner = mb.declare_func("inner", &[], Some(Type::FrameSlice));

        let mut inner = mb.define_func(f_inner);
        let forwarded_prompt = inner.create_prompt();
        let slice = inner.capture_slice(forwarded_prompt, &[]);
        inner.ret(slice);
        mb.finish_func(f_inner, inner);

        let mut main = mb.define_func(f_main);
        let handler_bb = main.create_block(&[Type::FrameSlice]);
        let prompt = main.create_prompt();
        main.push_prompt(prompt, handler_bb);
        let slice = main.call(f_inner, &[]).expect("call should produce a slice");
        main.pop_prompt(prompt);
        main.jump(handler_bb, &[slice]);
        main.switch_to_block(handler_bb);
        let popped = main.block_param(handler_bb, 0);
        main.ret(popped);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);
        let result = execute_jit_module_function(&jit, f_main, &[], &store).unwrap();

        match result {
            JitExecutionResult::CaptureSlice { handle, record } => {
                assert_eq!(record.kind, FrameReifyKind::CaptureSlice);
                let view = store.read(handle).unwrap();
                assert_eq!(view.frame_count(), 2);
                assert_eq!(view.frame(0).func_idx, 0);
                assert_eq!(view.frame(1).func_idx, 1);
                assert_eq!(
                    view.frame(1).caller_resume,
                    FrameResume::FromCall {
                        return_dest: Some(slice.index())
                    }
                );
                assert_eq!(view.frame(0).active_prompts, &[prompt.index() as u32]);
            }
            other => panic!("expected capture execution result, got {other:?}"),
        }
    }

    #[test]
    fn decode_clone_and_resume_frame_control_outcomes() {
        let clone_record = FrameReifyRecord {
            kind: FrameReifyKind::CloneSlice,
            prompt: None,
            active_prompts: vec![],
            resume: FrameResumePoint {
                func_idx: 0,
                block_idx: 1,
                inst_idx: 2,
            },
            native_resume_offset: None,
            frame_value_count: 5,
            value_indices: vec![3],
            control_value_indices: vec![3],
            value_types: vec![Type::FrameSlice],
            root_payload_indices: vec![],
            return_dest: Some(4),
        };
        let resume_record = FrameReifyRecord {
            kind: FrameReifyKind::ResumeSlice,
            prompt: None,
            active_prompts: vec![],
            resume: FrameResumePoint {
                func_idx: 5,
                block_idx: 6,
                inst_idx: 7,
            },
            native_resume_offset: None,
            frame_value_count: 10,
            value_indices: vec![8, 9],
            control_value_indices: vec![8, 9],
            value_types: vec![Type::FrameSlice, Type::I64],
            root_payload_indices: vec![],
            return_dest: None,
        };

        assert!(matches!(
            decode_frame_control_outcome(
                JitOutcome::CloneSlice {
                    func_idx: 0,
                    record_idx: 0,
                    values: vec![11],
                },
                &[clone_record.clone()]
            )
            .unwrap(),
            Some(JitFrameControl::CloneSlice { record, values })
            if record == clone_record && values == vec![11]
        ));

        assert!(matches!(
            decode_frame_control_outcome(
                JitOutcome::ResumeSlice {
                    func_idx: 0,
                    record_idx: 0,
                    values: vec![22, 33],
                },
                &[resume_record.clone()]
            )
            .unwrap(),
            Some(JitFrameControl::ResumeSlice { record, values })
            if record == resume_record && values == vec![22, 33]
        ));
    }

    #[test]
    fn execute_jit_function_bridges_clone_and_resume_requests() {
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);
        let builder = CapturedStackBuilder {
            prompt_id: 0,
            frames: vec![BuilderFrame {
                func_idx: 1,
                block_idx: 2,
                inst_idx: 3,
                values: vec![44],
                root_indices: vec![],
                resume_arg_slot: None,
                active_prompts: vec![],
                caller_resume: FrameResume::TopLevel,
            }],
        };
        let handle = store.capture_from_builder(&builder).unwrap();

        let clone_result = execute_outcome(
            JitOutcome::CloneSlice {
                func_idx: 0,
                record_idx: 0,
                values: vec![handle],
            },
            &[FrameReifyRecord {
                kind: FrameReifyKind::CloneSlice,
                prompt: None,
                active_prompts: vec![],
                resume: FrameResumePoint {
                    func_idx: 0,
                    block_idx: 0,
                    inst_idx: 0,
                },
                native_resume_offset: None,
                frame_value_count: 2,
                value_indices: vec![0],
                control_value_indices: vec![0],
                value_types: vec![Type::FrameSlice],
                root_payload_indices: vec![],
                return_dest: Some(1),
            }],
            &store,
        )
        .unwrap();

        let cloned_handle = match clone_result {
            JitExecutionResult::CloneSlice { handle, .. } => handle,
            other => panic!("expected clone result, got {other:?}"),
        };
        assert_eq!(handle, cloned_handle);

        let resume_result = execute_outcome(
            JitOutcome::ResumeSlice {
                func_idx: 0,
                record_idx: 0,
                values: vec![cloned_handle, 99],
            },
            &[FrameReifyRecord {
                kind: FrameReifyKind::ResumeSlice,
                prompt: None,
                active_prompts: vec![],
                resume: FrameResumePoint {
                    func_idx: 9,
                    block_idx: 8,
                    inst_idx: 7,
                },
                native_resume_offset: None,
                frame_value_count: 2,
                value_indices: vec![0, 1],
                control_value_indices: vec![0, 1],
                value_types: vec![Type::FrameSlice, Type::I64],
                root_payload_indices: vec![],
                return_dest: None,
            }],
            &store,
        )
        .unwrap();

        match resume_result {
            JitExecutionResult::ResumeSlice { handle, args, .. } => {
                assert_eq!(
                    handle,
                    cloned_handle
                );
                assert_eq!(args, vec![99]);

            }
            other => panic!("expected resume result, got {other:?}"),
        }
    }

    #[test]
    fn jit_capture_snapshot_can_resume_through_interpreter() {
        let mut b = FunctionBuilder::new("capture", &[], Some(Type::I64));
        let handler_bb = b.create_block(&[Type::I64]);
        let prompt = b.create_prompt();
        b.push_prompt(prompt, handler_bb);
        let _slice = b.capture_slice(prompt, &[]);
        let resumed = b.iconst(Type::I64, 77);
        b.pop_prompt(prompt);
        b.jump(handler_bb, &[resumed]);
        b.switch_to_block(handler_bb);
        let popped = b.block_param(handler_bb, 0);
        b.ret(popped);
        let func = b.build();

        let jit = JitFunction::compile::<NanBox>(&func, &[]);
        let (module, _entry) = Module::from_function(func);

        let ctx = make_cont_ctx();

        let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &ctx);
        interp.set_cont_ctx(&ctx);
        let store = JitFrameSliceRuntime::new(&ctx);

        let handle = match execute_jit_function(&jit, &[], &store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, .. } => handle,
            other => panic!("expected capture result, got {other:?}"),
        };

        match resume_stored_slice_with_interpreter(&interp, &store, handle, &[]).unwrap() {
            InterpResult::Value(v) => assert_eq!(v, 77),
            other => panic!("expected resumed interpreter value, got {other:?}"),
        }
    }

    #[test]
    fn jit_capture_snapshot_can_resume_natively() {
        let mut b = FunctionBuilder::new("capture", &[], Some(Type::I64));
        let handler_bb = b.create_block(&[Type::I64]);
        let prompt = b.create_prompt();
        b.push_prompt(prompt, handler_bb);
        let _slice = b.capture_slice(prompt, &[]);
        let resumed = b.iconst(Type::I64, 77);
        b.pop_prompt(prompt);
        b.jump(handler_bb, &[resumed]);
        b.switch_to_block(handler_bb);
        let popped = b.block_param(handler_bb, 0);
        b.ret(popped);
        let func = b.build();

        let jit = JitFunction::compile::<NanBox>(&func, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);

        let (handle, record) = match execute_jit_function(&jit, &[], &store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
            other => panic!("expected capture result, got {other:?}"),
        };

        match resume_stored_slice_with_jit(&jit, &record, &store, handle, &[]).unwrap() {
            JitOutcome::Value(v) => assert_eq!(v, 77),
            other => panic!("expected native resumed value, got {other:?}"),
        }
    }

    #[test]
    fn jit_module_clone_and_resume_runs_to_terminal_natively() {
        let mut mb = ModuleBuilder::new();
        let f_capture = mb.declare_func("capture", &[], Some(Type::I64));
        let f_clone_resume =
            mb.declare_func("clone_resume", &[Type::FrameSlice], Some(Type::I64));

        let mut capture = mb.define_func(f_capture);
        let handler_bb = capture.create_block(&[Type::I64]);
        let prompt = capture.create_prompt();
        capture.push_prompt(prompt, handler_bb);
        let _slice = capture.capture_slice(prompt, &[]);
        let resumed = capture.iconst(Type::I64, 77);
        capture.pop_prompt(prompt);
        capture.jump(handler_bb, &[resumed]);
        capture.switch_to_block(handler_bb);
        let popped = capture.block_param(handler_bb, 0);
        capture.ret(popped);
        mb.finish_func(f_capture, capture);

        let mut clone_resume = mb.define_func(f_clone_resume);
        let entry = clone_resume.entry_block();
        let slice = clone_resume.block_param(entry, 0);
        let cloned = clone_resume.clone_slice(slice);
        let return_block = clone_resume.create_block(&[Type::I64]);
        clone_resume.resume_slice(cloned, &[], return_block, &[]);
        clone_resume.switch_to_block(return_block);
        let ret_val = clone_resume.block_param(return_block, 0);
        clone_resume.ret(ret_val);
        mb.finish_func(f_clone_resume, clone_resume);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);

        let handle = match execute_jit_module_function(&jit, f_capture, &[], &store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, .. } => handle,
            other => panic!("expected capture result, got {other:?}"),
        };

        match execute_jit_module_function_to_terminal(
            &jit,
            f_clone_resume,
            &[handle],
            &store,
        )
        .unwrap()
        {
            JitExecutionResult::Value(v) => assert_eq!(v, 77),
            other => panic!("expected terminal native value, got {other:?}"),
        }
    }

    #[test]
    fn jit_multi_frame_capture_can_resume_natively() {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[], Some(Type::I64));
        let f_inner = mb.declare_func("inner", &[], Some(Type::I64));

        let mut inner = mb.define_func(f_inner);
        let forwarded_prompt = inner.create_prompt();
        let _slice = inner.capture_slice(forwarded_prompt, &[]);
        let resumed = inner.iconst(Type::I64, 91);
        inner.ret(resumed);
        mb.finish_func(f_inner, inner);

        let mut main = mb.define_func(f_main);
        let handler_bb = main.create_block(&[Type::I64]);
        let prompt = main.create_prompt();
        main.push_prompt(prompt, handler_bb);
        let ret = main.call(f_inner, &[]).expect("inner call should return i64");
        main.pop_prompt(prompt);
        main.jump(handler_bb, &[ret]);
        main.switch_to_block(handler_bb);
        let popped = main.block_param(handler_bb, 0);
        main.ret(popped);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);

        let (handle, record) = match execute_jit_module_function(&jit, f_main, &[], &store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
            other => panic!("expected capture result, got {other:?}"),
        };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, handle, &[]).unwrap() {
            JitOutcome::Value(v) => assert_eq!(v, 91),
            other => panic!("expected native resumed multi-frame value, got {other:?}"),
        }
    }

    #[test]
    fn jit_multi_frame_invoke_capture_can_resume_natively() {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[], Some(Type::I64));
        let f_inner = mb.declare_func("inner", &[], Some(Type::I64));

        let mut inner = mb.define_func(f_inner);
        let forwarded_prompt = inner.create_prompt();
        let _slice = inner.capture_slice(forwarded_prompt, &[]);
        let resumed = inner.iconst(Type::I64, 111);
        inner.ret(resumed);
        mb.finish_func(f_inner, inner);

        let mut main = mb.define_func(f_main);
        let handler_bb = main.create_block(&[Type::I64]);
        let prompt = main.create_prompt();
        let _entry = main.entry_block();
        let normal = main.create_block(&[Type::I64]);
        let exception = main.create_block(&[]);
        main.push_prompt(prompt, handler_bb);
        main.invoke(f_inner, &[], normal, &[], exception, &[]);
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.jump(handler_bb, &[ret]);
        main.switch_to_block(exception);
        let zero = main.iconst(Type::I64, 0);
        main.ret(zero);
        main.switch_to_block(handler_bb);
        let popped = main.block_param(handler_bb, 0);
        main.ret(popped);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);

        let (handle, record) = match execute_jit_module_function(&jit, f_main, &[], &store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
            other => panic!("expected capture result, got {other:?}"),
        };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, handle, &[]).unwrap() {
            JitOutcome::Value(v) => assert_eq!(v, 111),
            other => panic!("expected native resumed invoke-chain value, got {other:?}"),
        }
    }

    #[test]
    fn jit_multi_frame_invoke_capture_preserves_resume_args() {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[], Some(Type::I64));
        let f_inner = mb.declare_func("inner", &[], Some(Type::I64));

        let mut inner = mb.define_func(f_inner);
        let forwarded_prompt = inner.create_prompt();
        let _slice = inner.capture_slice(forwarded_prompt, &[]);
        let resumed = inner.iconst(Type::I64, 333);
        inner.ret(resumed);
        mb.finish_func(f_inner, inner);

        let mut main = mb.define_func(f_main);
        let handler_bb = main.create_block(&[Type::I64]);
        let prompt = main.create_prompt();
        let normal = main.create_block(&[Type::I64, Type::I64]);
        let exception = main.create_block(&[Type::I64]);
        let normal_extra = main.iconst(Type::I64, 44);
        let exception_extra = main.iconst(Type::I64, 99);
        main.push_prompt(prompt, handler_bb);
        main.invoke(
            f_inner,
            &[],
            normal,
            &[normal_extra],
            exception,
            &[exception_extra],
        );
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.jump(handler_bb, &[ret]);
        main.switch_to_block(exception);
        let exc = main.block_param(exception, 0);
        main.ret(exc);
        main.switch_to_block(handler_bb);
        let popped = main.block_param(handler_bb, 0);
        main.ret(popped);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);

        let handle = match execute_jit_module_function(&jit, f_main, &[], &store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, .. } => handle,
            other => panic!("expected capture result, got {other:?}"),
        };
        let view = store.read(handle).unwrap();
        assert_eq!(view.frame_count(), 2);
        match &view.frame(1).caller_resume {
            FrameResume::FromInvoke {
                normal_args_vals,
                exception_args_vals,
                ..
            } => {
                assert_eq!(normal_args_vals, &[44u64]);
                assert_eq!(exception_args_vals, &[99u64]);
            }
            other => panic!("expected invoke caller resume, got {other:?}"),
        }
    }

    #[test]
    fn jit_multi_frame_invoke_indirect_capture_can_resume_natively() {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[Type::Ptr], Some(Type::I64));
        let f_inner = mb.declare_func("inner", &[], Some(Type::I64));

        let mut inner = mb.define_func(f_inner);
        let forwarded_prompt = inner.create_prompt();
        let _slice = inner.capture_slice(forwarded_prompt, &[]);
        let resumed = inner.iconst(Type::I64, 515);
        inner.ret(resumed);
        mb.finish_func(f_inner, inner);

        let mut main = mb.define_func(f_main);
        let handler_bb = main.create_block(&[Type::I64]);
        let prompt = main.create_prompt();
        let entry = main.entry_block();
        let callee = main.block_param(entry, 0);
        let normal = main.create_block(&[Type::I64]);
        let exception = main.create_block(&[]);
        main.push_prompt(prompt, handler_bb);
        main.invoke_indirect(callee, &[], Some(Type::I64), normal, &[], exception, &[]);
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.jump(handler_bb, &[ret]);
        main.switch_to_block(exception);
        let zero = main.iconst(Type::I64, 0);
        main.ret(zero);
        main.switch_to_block(handler_bb);
        let popped = main.block_param(handler_bb, 0);
        main.ret(popped);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);
        let inner_ptr = jit.function_ptr(f_inner) as u64;

        let (handle, record) =
            match execute_jit_module_function(&jit, f_main, &[inner_ptr], &store).unwrap() {
                JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
                other => panic!("expected capture result, got {other:?}"),
            };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, handle, &[]).unwrap() {
            JitOutcome::Value(v) => assert_eq!(v, 515),
            other => panic!("expected native resumed invoke-indirect value, got {other:?}"),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn jit_multi_frame_invoke_exception_capture_can_resume_natively() {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[Type::I64], Some(Type::I64));
        let f_inner = mb.declare_func("inner", &[Type::I64], Some(Type::I64));

        let mut inner = mb.define_func(f_inner);
        let callee = inner.block_param(inner.entry_block(), 0);
        let forwarded_prompt = inner.create_prompt();
        let _slice = inner.capture_slice(forwarded_prompt, &[]);
        let ret = inner
            .call_indirect(callee, &[], Some(Type::I64))
            .expect("indirect call should return i64");
        inner.ret(ret);
        mb.finish_func(f_inner, inner);

        let mut main = mb.define_func(f_main);
        let handler_bb = main.create_block(&[Type::I64]);
        let prompt = main.create_prompt();
        let entry = main.entry_block();
        let callee = main.block_param(entry, 0);
        let normal = main.create_block(&[Type::I64]);
        let exception = main.create_block(&[]);
        main.push_prompt(prompt, handler_bb);
        main.invoke(f_inner, &[callee], normal, &[], exception, &[]);
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.jump(handler_bb, &[ret]);
        main.switch_to_block(exception);
        let ex = main.iconst(Type::I64, 222);
        main.ret(ex);
        main.switch_to_block(handler_bb);
        let popped = main.block_param(handler_bb, 0);
        main.ret(popped);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);
        let throw_ptr = dynruntime_test_throw_exception_stub as usize as u64;

        let (handle, record) =
            match execute_jit_module_function(&jit, f_main, &[throw_ptr], &store).unwrap() {
                JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
                other => panic!("expected capture result, got {other:?}"),
            };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, handle, &[]).unwrap() {
            JitOutcome::Value(v) => assert_eq!(v, 222),
            other => panic!("expected native resumed invoke-exception value, got {other:?}"),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn jit_multi_frame_invoke_indirect_exception_capture_can_resume_natively() {
        let mut mb = ModuleBuilder::new();
        let f_main = mb.declare_func("main", &[Type::Ptr, Type::Ptr], Some(Type::I64));
        let f_inner = mb.declare_func("inner", &[Type::Ptr], Some(Type::I64));

        let mut inner = mb.define_func(f_inner);
        let entry = inner.entry_block();
        let throw_ptr = inner.block_param(entry, 0);
        let forwarded_prompt = inner.create_prompt();
        let _slice = inner.capture_slice(forwarded_prompt, &[]);
        let ret = inner
            .call_indirect(throw_ptr, &[], Some(Type::I64))
            .expect("indirect call should return i64");
        inner.ret(ret);
        mb.finish_func(f_inner, inner);

        let mut main = mb.define_func(f_main);
        let handler_bb = main.create_block(&[Type::I64]);
        let prompt = main.create_prompt();
        let entry = main.entry_block();
        let callee = main.block_param(entry, 0);
        let throw_ptr = main.block_param(entry, 1);
        let normal = main.create_block(&[Type::I64]);
        let exception = main.create_block(&[]);
        main.push_prompt(prompt, handler_bb);
        main.invoke_indirect(callee, &[throw_ptr], Some(Type::I64), normal, &[], exception, &[]);
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.jump(handler_bb, &[ret]);
        main.switch_to_block(exception);
        let ex = main.iconst(Type::I64, 616);
        main.ret(ex);
        main.switch_to_block(handler_bb);
        let popped = main.block_param(handler_bb, 0);
        main.ret(popped);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let ctx = make_cont_ctx();
        let store = JitFrameSliceRuntime::new(&ctx);
        let inner_ptr = jit.function_ptr(f_inner) as u64;
        let throw_stub_ptr = dynruntime_test_throw_exception_stub as usize as u64;

        let (handle, record) =
            match execute_jit_module_function(&jit, f_main, &[inner_ptr, throw_stub_ptr], &store).unwrap() {
                JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
                other => panic!("expected capture result, got {other:?}"),
            };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, handle, &[]).unwrap() {
            JitOutcome::Value(v) => assert_eq!(v, 616),
            other => panic!(
                "expected native resumed invoke-indirect exception value, got {other:?}"
            ),
        }
    }

    struct CountingTransport<'a> {
        last_payload: &'a Cell<usize>,
    }

    impl JitRootTransportRuntime for CountingTransport<'_> {
        fn payload_kind(&self) -> SafepointHandlerPayloadKind {
            SafepointHandlerPayloadKind::SafepointIndex
        }

        unsafe fn scan_roots(
            &self,
            _frame_ptr: *mut u8,
            payload: usize,
            _safepoints: &[SafepointRecord],
            _visitor: &mut dyn FnMut(*mut u64),
        ) {
            self.last_payload.set(payload);
        }
    }

    #[test]
    fn installed_session_routes_handler_calls_through_transport() {
        use dynobj::{ObjHeader, TypeInfo};
        let dummy_type = TypeInfo::for_header(Compact::SIZE);
        let heap = Heap::new::<Compact>(256, vec![dummy_type]);
        let last_payload = Cell::new(usize::MAX);
        let transport = CountingTransport {
            last_payload: &last_payload,
        };
        let session = JitSafepointSession::<
            crate::ptr_policy::LowBitPtrPolicy<3>,
            _,
        >::new(&heap, transport, &[])
        .with_gc_policy(GcPolicy::EveryPoint);
        let mut frame = [0u64; 2];

        session.with_installed(|| {
            active_jit_safepoint_handler(frame.as_mut_ptr().cast::<u8>(), 7);
        });

        assert_eq!(last_payload.get(), 7);
    }
}
