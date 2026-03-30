use std::cell::RefCell;
use std::marker::PhantomData;

use dynalloc::{Heap, PtrPolicy};
use dynexec::{
    CapturedCallerResume, CapturedFrame, FrameSliceError, FrameSliceMode, FrameSliceSnapshot,
    FrameSliceStore,
};
use dynir::interp::{ConfiguredModuleInterpreter, InterpError, InterpResult, InterpRootManager};
use dynlower::{
    FrameReifyKind, FrameReifyRecord, JitFunction, JitModule, JitOutcome,
    SafepointHandlerPayloadKind, SafepointRecord, SuspendedJitFrame, take_suspended_frames,
};
use dynobj::RootSource;

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

pub struct FrameScanJitTransport;
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
pub enum JitExecutionResult<H> {
    Value(u64),
    Void,
    Exception(u64),
    Deopt {
        deopt_id: dynir::DeoptId,
        resume_point: u64,
        live_values: Vec<u64>,
    },
    CaptureSlice {
        handle: H,
        record: FrameReifyRecord,
    },
    CloneSlice {
        handle: H,
        record: FrameReifyRecord,
    },
    ResumeSlice {
        handle: H,
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

fn build_capture_snapshot(
    record: &FrameReifyRecord,
    values: &[u64],
    mut suspended: Vec<SuspendedJitFrame>,
) -> Result<FrameSliceSnapshot, JitFrameControlError> {
    let prompt = record
        .prompt
        .ok_or(JitFrameControlError::UnsupportedOutcome)?;
    let mut frame_values = vec![0u64; record.frame_value_count];
    for (&value_idx, &bits) in record.value_indices.iter().zip(values.iter()) {
        if value_idx >= frame_values.len() {
            return Err(JitFrameControlError::UnsupportedOutcome);
        }
        frame_values[value_idx] = bits;
    }
    let mut root_value_indices = Vec::with_capacity(record.root_payload_indices.len());
    for &payload_idx in &record.root_payload_indices {
        let Some(&value_idx) = record.value_indices.get(payload_idx) else {
            return Err(JitFrameControlError::UnsupportedOutcome);
        };
        root_value_indices.push(value_idx);
    }
    let resume_arg_value_indices = record.return_dest.into_iter().collect();
    let frame = CapturedFrame {
        resume: record.resume,
        values: frame_values,
        root_value_indices,
        resume_arg_value_indices,
        active_prompts: record.active_prompts.iter().map(|p| p.index() as u32).collect(),
        caller_resume: CapturedCallerResume::TopLevel,
    };
    let mut frames = if suspended.is_empty() {
        vec![frame]
    } else {
        let mut current = frame;
        current.caller_resume = suspended
            .last()
            .map(|entry| entry.callee_caller_resume.clone())
            .unwrap_or(CapturedCallerResume::TopLevel);
        let mut frames_rev = vec![current];
        while let Some(entry) = suspended.pop() {
            let mut caller = entry.frame;
            caller.caller_resume = suspended
                .last()
                .map(|outer| outer.callee_caller_resume.clone())
                .unwrap_or(CapturedCallerResume::TopLevel);
            frames_rev.push(caller);
        }
        frames_rev.reverse();
        frames_rev
    };
    Ok(FrameSliceSnapshot {
        prompt_id: prompt.index() as u32,
        mode: FrameSliceMode::OneShot,
        frames: std::mem::take(&mut frames),
        consumed: false,
    })
}

pub fn materialize_capture_slice<S: FrameSliceStore>(
    store: &mut S,
    record: &FrameReifyRecord,
    values: &[u64],
) -> Result<S::Handle, JitFrameControlError> {
    let snapshot = build_capture_snapshot(record, values, Vec::new())?;
    store
        .insert_slice(snapshot)
        .map_err(|_| JitFrameControlError::UnsupportedOutcome)
}

fn decode_store_handle<S: FrameSliceStore>(bits: u64) -> Result<S::Handle, JitFrameControlError> {
    S::decode_handle(bits).map_err(|_| JitFrameControlError::UnsupportedOutcome)
}

pub fn resume_stored_slice_with_interpreter<'a, Cfg, R, IS, S>(
    interpreter: &ConfiguredModuleInterpreter<'a, Cfg, R, IS>,
    store: &S,
    handle: &S::Handle,
    args: &[u64],
) -> Result<InterpResult, ResumeWithInterpreterError>
where
    Cfg: dynexec::ExecutionConfig,
    Cfg::Layout: dynexec::LayoutConfigDefaults,
    R: InterpRootManager<Cfg::Layout, Cfg::Roots, Cfg::RootTransport>,
    IS: FrameSliceStore,
    S: FrameSliceStore,
{
    let snapshot = store
        .slice(handle)
        .map_err(ResumeWithInterpreterError::FrameSlice)?
        .clone();
    interpreter
        .resume_snapshot(snapshot, args)
        .map_err(ResumeWithInterpreterError::Interp)
}

pub fn resume_stored_slice_with_jit<S: FrameSliceStore>(
    jit: &JitFunction,
    _record: &FrameReifyRecord,
    store: &S,
    handle: &S::Handle,
    args: &[u64],
) -> Result<JitOutcome, ResumeWithInterpreterError> {
    let snapshot = store
        .slice(handle)
        .map_err(ResumeWithInterpreterError::FrameSlice)?;
    let frame = snapshot
        .frames
        .last()
        .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
    let capture_record = jit
        .frame_reify_records()
        .iter()
        .find(|record| {
            record.kind == FrameReifyKind::CaptureSlice
                && record.native_resume_offset.is_some()
                && record.resume == frame.resume
        })
        .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
    Ok(jit.call_resume_outcome(capture_record, frame.values.as_ptr(), args))
}

pub fn resume_stored_slice_with_jit_module<S: FrameSliceStore>(
    jit: &JitModule,
    _record: &FrameReifyRecord,
    store: &S,
    handle: &S::Handle,
    args: &[u64],
) -> Result<JitOutcome, ResumeWithInterpreterError> {
    let snapshot = store
        .slice(handle)
        .map_err(ResumeWithInterpreterError::FrameSlice)?;
    let mut frame_idx = snapshot
        .frames
        .len()
        .checked_sub(1)
        .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
    let mut outcome = jit
        .call_captured_frame_resume_outcome(&snapshot.frames[frame_idx], args)
        .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;

    loop {
        let frame = &snapshot.frames[frame_idx];
        match (&frame.caller_resume, outcome.clone()) {
            (CapturedCallerResume::TopLevel, final_outcome) => return Ok(final_outcome),
            (CapturedCallerResume::FromCall { .. }, JitOutcome::Value(v)) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Value(v));
                }
                frame_idx -= 1;
                outcome = jit
                    .call_captured_frame_resume_outcome(&snapshot.frames[frame_idx], &[v])
                    .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
            }
            (CapturedCallerResume::FromCall { .. }, JitOutcome::Void) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Void);
                }
                frame_idx -= 1;
                outcome = jit
                    .call_captured_frame_resume_outcome(&snapshot.frames[frame_idx], &[])
                    .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
            }
            (CapturedCallerResume::FromCall { .. }, JitOutcome::Exception(exc)) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Exception(exc));
                }
                frame_idx -= 1;
                continue;
            }
            (
                CapturedCallerResume::FromInvoke { .. },
                JitOutcome::Value(v),
            ) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Value(v));
                }
                frame_idx -= 1;
                outcome = jit
                    .call_invoke_frame_resume_outcome(&snapshot.frames[frame_idx], false, &[v])
                    .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
            }
            (
                CapturedCallerResume::FromInvoke { .. },
                JitOutcome::Void,
            ) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Void);
                }
                frame_idx -= 1;
                outcome = jit
                    .call_invoke_frame_resume_outcome(&snapshot.frames[frame_idx], false, &[])
                    .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
            }
            (
                CapturedCallerResume::FromInvoke { .. },
                JitOutcome::Exception(exc),
            ) => {
                if frame_idx == 0 {
                    return Ok(JitOutcome::Exception(exc));
                }
                frame_idx -= 1;
                outcome = jit
                    .call_invoke_frame_resume_outcome(&snapshot.frames[frame_idx], true, &[exc])
                    .ok_or(ResumeWithInterpreterError::FrameSlice(FrameSliceError::MissingSlice))?;
            }
            (_, other) => return Ok(other),
        }
    }
}

fn continue_outcome_with_function<S: FrameSliceStore>(
    jit: &JitFunction,
    outcome: JitOutcome,
    store: &mut S,
) -> Result<JitExecutionResult<S::Handle>, JitFrameControlError> {
    if let Some(control) = decode_frame_control_outcome(outcome.clone(), jit.frame_reify_records())? {
        return match control {
            JitFrameControl::CaptureSlice { record, values } => {
                let snapshot = build_capture_snapshot(&record, &values, take_suspended_frames())?;
                let handle = store
                    .insert_slice(snapshot)
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
                let handle = decode_store_handle::<S>(source_bits)?;
                let cloned = store
                    .clone_slice(&handle)
                    .map_err(JitFrameControlError::FrameSlice)?;
                let cloned_bits = S::encode_handle(&cloned);
                if record.native_resume_offset.is_some() {
                    return continue_outcome_with_function(
                        jit,
                        jit.call_resume_outcome(&record, values.as_ptr(), &[cloned_bits]),
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
                let handle = decode_store_handle::<S>(*slice_bits)?;
                if matches!(
                    store
                        .slice(&handle)
                        .map_err(JitFrameControlError::FrameSlice)?
                        .mode,
                    FrameSliceMode::OneShot
                ) {
                    store
                        .mark_consumed(&handle)
                        .map_err(JitFrameControlError::FrameSlice)?;
                }
                return continue_outcome_with_function(
                    jit,
                    resume_stored_slice_with_jit(jit, &record, store, &handle, args)
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

pub fn execute_jit_function<S: FrameSliceStore>(
    jit: &JitFunction,
    args: &[u64],
    store: &mut S,
) -> Result<JitExecutionResult<S::Handle>, JitFrameControlError> {
    execute_outcome(jit.call_outcome(args), jit.frame_reify_records(), store)
}

pub fn execute_jit_function_to_terminal<S: FrameSliceStore>(
    jit: &JitFunction,
    args: &[u64],
    store: &mut S,
) -> Result<JitExecutionResult<S::Handle>, JitFrameControlError> {
    continue_outcome_with_function(jit, jit.call_outcome(args), store)
}

pub fn execute_jit_module_function<S: FrameSliceStore>(
    jit: &JitModule,
    func_ref: dynir::FuncRef,
    args: &[u64],
    store: &mut S,
) -> Result<JitExecutionResult<S::Handle>, JitFrameControlError> {
    continue_outcome_with_module(
        jit,
        jit.frame_reify_records_for_function(func_ref.index()),
        jit.call_outcome(func_ref, args),
        store,
    )
}

fn continue_outcome_with_module<S: FrameSliceStore>(
    jit: &JitModule,
    records: &[FrameReifyRecord],
    outcome: JitOutcome,
    store: &mut S,
) -> Result<JitExecutionResult<S::Handle>, JitFrameControlError> {
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
                let snapshot = build_capture_snapshot(&record, &values, take_suspended_frames())?;
                let handle = store
                    .insert_slice(snapshot)
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
                let handle = decode_store_handle::<S>(source_bits)?;
                let cloned = store
                    .clone_slice(&handle)
                    .map_err(JitFrameControlError::FrameSlice)?;
                let cloned_bits = S::encode_handle(&cloned);
                if record.native_resume_offset.is_some() {
                    let next = jit.call_resume_outcome(
                        record.resume.func_idx,
                        &record,
                        values.as_ptr(),
                        &[cloned_bits],
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
                let handle = decode_store_handle::<S>(*slice_bits)?;
                if matches!(
                    store
                        .slice(&handle)
                        .map_err(JitFrameControlError::FrameSlice)?
                        .mode,
                    FrameSliceMode::OneShot
                ) {
                    store
                        .mark_consumed(&handle)
                        .map_err(JitFrameControlError::FrameSlice)?;
                }
                let next = resume_stored_slice_with_jit_module(jit, &record, store, &handle, args)
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

pub fn execute_jit_module_function_to_terminal<S: FrameSliceStore>(
    jit: &JitModule,
    func_ref: dynir::FuncRef,
    args: &[u64],
    store: &mut S,
) -> Result<JitExecutionResult<S::Handle>, JitFrameControlError> {
    continue_outcome_with_module(
        jit,
        jit.frame_reify_records_for_function(func_ref.index()),
        jit.call_outcome(func_ref, args),
        store,
    )
}

fn execute_outcome<S: FrameSliceStore>(
    outcome: JitOutcome,
    records: &[FrameReifyRecord],
    store: &mut S,
) -> Result<JitExecutionResult<S::Handle>, JitFrameControlError> {
    if let Some(control) = decode_frame_control_outcome(outcome.clone(), records)? {
        return match control {
            JitFrameControl::CaptureSlice { record, values } => {
                let handle = materialize_capture_slice(store, &record, &values)?;
                Ok(JitExecutionResult::CaptureSlice { handle, record })
            }
            JitFrameControl::CloneSlice { record, values } => {
                let handle = decode_store_handle::<S>(*values.first().ok_or(
                    JitFrameControlError::UnsupportedOutcome,
                )?)?;
                let cloned = store
                    .clone_slice(&handle)
                    .map_err(|_| JitFrameControlError::UnsupportedOutcome)?;
                Ok(JitExecutionResult::CloneSlice {
                    handle: cloned,
                    record,
                })
            }
            JitFrameControl::ResumeSlice { record, values } => {
                let (slice_bits, args) = values
                    .split_first()
                    .ok_or(JitFrameControlError::UnsupportedOutcome)?;
                let handle = decode_store_handle::<S>(*slice_bits)?;
                if matches!(
                    store
                        .slice(&handle)
                        .map_err(JitFrameControlError::FrameSlice)?
                        .mode,
                    FrameSliceMode::OneShot
                ) {
                    store
                        .mark_consumed(&handle)
                        .map_err(JitFrameControlError::FrameSlice)?;
                }
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

impl JitRootTransportRuntime for FrameScanJitTransport {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        SafepointHandlerPayloadKind::FrameSize
    }

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        _safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    ) {
        let root_source = FrameWordRootSource {
            frame_ptr,
            frame_size: payload,
        };
        root_source.scan_roots(visitor);
    }
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

struct FrameWordRootSource {
    frame_ptr: *mut u8,
    frame_size: usize,
}

impl RootSource for FrameWordRootSource {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let word_count = self.frame_size / 8;
        for idx in 0..word_count {
            let slot = unsafe { self.frame_ptr.add(idx * 8).cast::<u64>() };
            visitor(slot);
        }
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

pub struct JitSafepointSession<'a, P: PtrPolicy, T: JitRootTransportRuntime> {
    heap: &'a Heap,
    transport: T,
    safepoints: &'a [SafepointRecord],
    gc_threshold: f64,
    _policy: PhantomData<P>,
}

#[derive(Clone, Copy)]
struct ActiveJitSafepointSession {
    ptr: *const (),
    handle: unsafe fn(*const (), *mut u8, usize),
}

impl<'a, P: PtrPolicy, T: JitRootTransportRuntime> JitSafepointSession<'a, P, T> {
    pub fn new(heap: &'a Heap, transport: T, safepoints: &'a [SafepointRecord]) -> Self {
        Self {
            heap,
            transport,
            safepoints,
            gc_threshold: 0.0,
            _policy: PhantomData,
        }
    }

    /// Set the GC threshold: only collect when from-space usage exceeds
    /// this fraction (0.0–1.0). Default 0.0 means collect at every safepoint.
    /// Mirrors `MutatorRootManager::with_gc_threshold`.
    pub fn with_gc_threshold(mut self, threshold: f64) -> Self {
        self.gc_threshold = threshold;
        self
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

    unsafe fn handle(&self, frame_ptr: *mut u8, payload: usize) {
        if self.gc_threshold > 0.0 {
            let usage = self.heap.from_used() as f64 / self.heap.space_size() as f64;
            if usage < self.gc_threshold {
                return;
            }
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
        unsafe {
            self.heap.collect::<P>(&[&root_source, &ancestor_roots]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use dynalloc::Heap;
    use dynexec::FrameResumePoint;
    use dynir::builder::{FunctionBuilder, ModuleBuilder};
    use dynir::interp::{ModuleInterpreter, NoGcRoots};
    use dynir::Module;
    use dynir::types::Type;
    use dynlower::FrameReifyKind;
    use dynobj::Compact;
    use dynvalue::NanBox;
    use crate::OwnedFrameSliceStore;

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
    fn frame_scan_transport_scans_full_frame_words() {
        let mut frame = [0u64; 4];
        frame[1] = 11;
        frame[2] = 22;

        let mut seen = Vec::new();
        unsafe {
            FrameScanJitTransport.scan_roots(
                frame.as_mut_ptr().cast::<u8>(),
                frame.len() * 8,
                &[],
                &mut |slot| seen.push(*slot),
            );
        }
        assert_eq!(seen, vec![0, 11, 22, 0]);
    }

    #[test]
    fn metadata_transport_scans_only_recorded_slots() {
        let mut frame = [0u64; 6];
        frame[2] = 33;
        frame[4] = 55;
        let safepoints = [SafepointRecord {
            code_offset: 0,
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
        let mut store = OwnedFrameSliceStore::new();
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

        let handle = materialize_capture_slice(&mut store, &record, &[88, 99]).unwrap();
        let snapshot = store.slice(&handle).unwrap();
        assert_eq!(snapshot.prompt_id, 1);
        assert_eq!(snapshot.frames.len(), 1);
        assert_eq!(snapshot.frames[0].resume, record.resume);
        assert_eq!(snapshot.frames[0].values.len(), 32);
        assert_eq!(snapshot.frames[0].values[10], 88);
        assert_eq!(snapshot.frames[0].values[20], 99);
        assert_eq!(snapshot.frames[0].root_value_indices, vec![20]);
        assert_eq!(snapshot.frames[0].resume_arg_value_indices, vec![30]);
        assert_eq!(snapshot.frames[0].active_prompts, vec![0, 1]);
    }

    #[test]
    fn execute_jit_function_materializes_capture_slice() {
        let mut b = FunctionBuilder::new("capture", &[Type::I64], Some(Type::FrameSlice));
        let entry = b.entry_block();
        let arg = b.block_param(entry, 0);
        let prompt = b.create_prompt();
        b.push_prompt(prompt);
        let slice = b.capture_slice(prompt, &[arg]);
        b.pop_prompt(prompt);
        b.unreachable();
        let func = b.build();

        let jit = JitFunction::compile::<NanBox>(&func, &[]);
        let mut store = OwnedFrameSliceStore::new();
        let result = execute_jit_function(&jit, &[123], &mut store).unwrap();

        match result {
            JitExecutionResult::CaptureSlice { handle, record } => {
                assert_eq!(record.kind, FrameReifyKind::CaptureSlice);
                let snapshot = store.slice(&handle).unwrap();
                assert_eq!(snapshot.frames.len(), 1);
                assert_eq!(snapshot.frames[0].values[arg.index()], 123);
                assert_eq!(snapshot.frames[0].resume_arg_value_indices, vec![slice.index()]);
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
        let mut store = OwnedFrameSliceStore::new();
        let result = execute_jit_module_function(&jit, f_main, &[77], &mut store).unwrap();

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
        let prompt = main.create_prompt();
        main.push_prompt(prompt);
        let slice = main.call(f_inner, &[]).expect("call should produce a slice");
        main.pop_prompt(prompt);
        main.ret(slice);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let mut store = OwnedFrameSliceStore::new();
        let result = execute_jit_module_function(&jit, f_main, &[], &mut store).unwrap();

        match result {
            JitExecutionResult::CaptureSlice { handle, record } => {
                assert_eq!(record.kind, FrameReifyKind::CaptureSlice);
                let snapshot = store.slice(&handle).unwrap();
                assert_eq!(snapshot.frames.len(), 2);
                assert_eq!(snapshot.frames[0].resume.func_idx, 0);
                assert_eq!(snapshot.frames[1].resume.func_idx, 1);
                assert_eq!(
                    snapshot.frames[1].caller_resume,
                    CapturedCallerResume::FromCall {
                        return_dest: Some(slice.index())
                    }
                );
                assert_eq!(snapshot.frames[0].active_prompts, vec![prompt.index() as u32]);
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
        let mut store = OwnedFrameSliceStore::new();
        let handle = store
            .insert_slice(FrameSliceSnapshot {
                prompt_id: 0,
                mode: FrameSliceMode::OneShot,
                frames: vec![CapturedFrame {
                    resume: FrameResumePoint {
                        func_idx: 1,
                        block_idx: 2,
                        inst_idx: 3,
                    },
                    values: vec![44],
                    root_value_indices: vec![],
                    resume_arg_value_indices: vec![],
                    active_prompts: vec![],
                    caller_resume: CapturedCallerResume::TopLevel,
                }],
                consumed: false,
            })
            .unwrap();

        let clone_result = execute_outcome(
            JitOutcome::CloneSlice {
                func_idx: 0,
                record_idx: 0,
                values: vec![OwnedFrameSliceStore::encode_handle(&handle)],
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
            &mut store,
        )
        .unwrap();

        let cloned_handle = match clone_result {
            JitExecutionResult::CloneSlice { handle, .. } => handle,
            other => panic!("expected clone result, got {other:?}"),
        };
        assert_ne!(OwnedFrameSliceStore::encode_handle(&handle), OwnedFrameSliceStore::encode_handle(&cloned_handle));

        let resume_result = execute_outcome(
            JitOutcome::ResumeSlice {
                func_idx: 0,
                record_idx: 0,
                values: vec![OwnedFrameSliceStore::encode_handle(&cloned_handle), 99],
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
            &mut store,
        )
        .unwrap();

        match resume_result {
            JitExecutionResult::ResumeSlice { handle, args, .. } => {
                assert_eq!(
                    OwnedFrameSliceStore::encode_handle(&handle),
                    OwnedFrameSliceStore::encode_handle(&cloned_handle)
                );
                assert_eq!(args, vec![99]);
                assert!(store.slice(&handle).unwrap().consumed);
            }
            other => panic!("expected resume result, got {other:?}"),
        }
    }

    #[test]
    fn jit_capture_snapshot_can_resume_through_interpreter() {
        let mut b = FunctionBuilder::new("capture", &[], Some(Type::I64));
        let prompt = b.create_prompt();
        b.push_prompt(prompt);
        let _slice = b.capture_slice(prompt, &[]);
        b.pop_prompt(prompt);
        let resumed = b.iconst(Type::I64, 77);
        b.ret(resumed);
        let func = b.build();

        let jit = JitFunction::compile::<NanBox>(&func, &[]);
        let (module, _entry) = Module::from_function(func);
        let roots = NoGcRoots;
        let interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
        let mut store = OwnedFrameSliceStore::new();

        let handle = match execute_jit_function(&jit, &[], &mut store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, .. } => handle,
            other => panic!("expected capture result, got {other:?}"),
        };

        match resume_stored_slice_with_interpreter(&interp, &store, &handle, &[]).unwrap() {
            InterpResult::Value(v) => assert_eq!(v, 77),
            other => panic!("expected resumed interpreter value, got {other:?}"),
        }
    }

    #[test]
    fn jit_capture_snapshot_can_resume_natively() {
        let mut b = FunctionBuilder::new("capture", &[], Some(Type::I64));
        let prompt = b.create_prompt();
        b.push_prompt(prompt);
        let _slice = b.capture_slice(prompt, &[]);
        b.pop_prompt(prompt);
        let resumed = b.iconst(Type::I64, 77);
        b.ret(resumed);
        let func = b.build();

        let jit = JitFunction::compile::<NanBox>(&func, &[]);
        let mut store = OwnedFrameSliceStore::new();

        let (handle, record) = match execute_jit_function(&jit, &[], &mut store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
            other => panic!("expected capture result, got {other:?}"),
        };

        match resume_stored_slice_with_jit(&jit, &record, &store, &handle, &[]).unwrap() {
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
        let prompt = capture.create_prompt();
        capture.push_prompt(prompt);
        let _slice = capture.capture_slice(prompt, &[]);
        capture.pop_prompt(prompt);
        let resumed = capture.iconst(Type::I64, 77);
        capture.ret(resumed);
        mb.finish_func(f_capture, capture);

        let mut clone_resume = mb.define_func(f_clone_resume);
        let entry = clone_resume.entry_block();
        let slice = clone_resume.block_param(entry, 0);
        let cloned = clone_resume.clone_slice(slice);
        clone_resume.resume_slice(cloned, &[]);
        mb.finish_func(f_clone_resume, clone_resume);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let mut store = OwnedFrameSliceStore::new();

        let handle = match execute_jit_module_function(&jit, f_capture, &[], &mut store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, .. } => handle,
            other => panic!("expected capture result, got {other:?}"),
        };

        match execute_jit_module_function_to_terminal(
            &jit,
            f_clone_resume,
            &[OwnedFrameSliceStore::encode_handle(&handle)],
            &mut store,
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
        let prompt = main.create_prompt();
        main.push_prompt(prompt);
        let ret = main.call(f_inner, &[]).expect("inner call should return i64");
        main.pop_prompt(prompt);
        main.ret(ret);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let mut store = OwnedFrameSliceStore::new();

        let (handle, record) = match execute_jit_module_function(&jit, f_main, &[], &mut store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
            other => panic!("expected capture result, got {other:?}"),
        };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, &handle, &[]).unwrap() {
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
        let prompt = main.create_prompt();
        let _entry = main.entry_block();
        let normal = main.create_block(&[Type::I64]);
        let exception = main.create_block(&[]);
        main.push_prompt(prompt);
        main.invoke(f_inner, &[], normal, &[], exception, &[]);
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.ret(ret);
        main.switch_to_block(exception);
        let zero = main.iconst(Type::I64, 0);
        main.ret(zero);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let mut store = OwnedFrameSliceStore::new();

        let (handle, record) = match execute_jit_module_function(&jit, f_main, &[], &mut store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
            other => panic!("expected capture result, got {other:?}"),
        };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, &handle, &[]).unwrap() {
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
        let prompt = main.create_prompt();
        let normal = main.create_block(&[Type::I64, Type::I64]);
        let exception = main.create_block(&[Type::I64]);
        let normal_extra = main.iconst(Type::I64, 44);
        let exception_extra = main.iconst(Type::I64, 99);
        main.push_prompt(prompt);
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
        main.ret(ret);
        main.switch_to_block(exception);
        let exc = main.block_param(exception, 0);
        main.ret(exc);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let mut store = OwnedFrameSliceStore::new();

        let handle = match execute_jit_module_function(&jit, f_main, &[], &mut store).unwrap() {
            JitExecutionResult::CaptureSlice { handle, .. } => handle,
            other => panic!("expected capture result, got {other:?}"),
        };
        let snapshot = store.slice(&handle).unwrap();
        assert_eq!(snapshot.frames.len(), 2);
        match &snapshot.frames[1].caller_resume {
            CapturedCallerResume::FromInvoke {
                normal_args_vals,
                exception_args_vals,
                ..
            } => {
                assert_eq!(normal_args_vals, &vec![44]);
                assert_eq!(exception_args_vals, &vec![99]);
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
        let prompt = main.create_prompt();
        let entry = main.entry_block();
        let callee = main.block_param(entry, 0);
        let normal = main.create_block(&[Type::I64]);
        let exception = main.create_block(&[]);
        main.push_prompt(prompt);
        main.invoke_indirect(callee, &[], Some(Type::I64), normal, &[], exception, &[]);
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.ret(ret);
        main.switch_to_block(exception);
        let zero = main.iconst(Type::I64, 0);
        main.ret(zero);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let mut store = OwnedFrameSliceStore::new();
        let inner_ptr = jit.function_ptr(f_inner) as u64;

        let (handle, record) =
            match execute_jit_module_function(&jit, f_main, &[inner_ptr], &mut store).unwrap() {
                JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
                other => panic!("expected capture result, got {other:?}"),
            };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, &handle, &[]).unwrap() {
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
        let prompt = main.create_prompt();
        let entry = main.entry_block();
        let callee = main.block_param(entry, 0);
        let normal = main.create_block(&[Type::I64]);
        let exception = main.create_block(&[]);
        main.push_prompt(prompt);
        main.invoke(f_inner, &[callee], normal, &[], exception, &[]);
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.ret(ret);
        main.switch_to_block(exception);
        let ex = main.iconst(Type::I64, 222);
        main.ret(ex);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let mut store = OwnedFrameSliceStore::new();
        let throw_ptr = dynruntime_test_throw_exception_stub as usize as u64;

        let (handle, record) =
            match execute_jit_module_function(&jit, f_main, &[throw_ptr], &mut store).unwrap() {
                JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
                other => panic!("expected capture result, got {other:?}"),
            };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, &handle, &[]).unwrap() {
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
        let prompt = main.create_prompt();
        let entry = main.entry_block();
        let callee = main.block_param(entry, 0);
        let throw_ptr = main.block_param(entry, 1);
        let normal = main.create_block(&[Type::I64]);
        let exception = main.create_block(&[]);
        main.push_prompt(prompt);
        main.invoke_indirect(callee, &[throw_ptr], Some(Type::I64), normal, &[], exception, &[]);
        main.switch_to_block(normal);
        let ret = main.block_param(normal, 0);
        main.pop_prompt(prompt);
        main.ret(ret);
        main.switch_to_block(exception);
        let ex = main.iconst(Type::I64, 616);
        main.ret(ex);
        mb.finish_func(f_main, main);

        let module = mb.build();
        let jit = JitModule::compile::<NanBox>(&module, &[]);
        let mut store = OwnedFrameSliceStore::new();
        let inner_ptr = jit.function_ptr(f_inner) as u64;
        let throw_stub_ptr = dynruntime_test_throw_exception_stub as usize as u64;

        let (handle, record) =
            match execute_jit_module_function(&jit, f_main, &[inner_ptr, throw_stub_ptr], &mut store).unwrap() {
                JitExecutionResult::CaptureSlice { handle, record } => (handle, record),
                other => panic!("expected capture result, got {other:?}"),
            };

        match resume_stored_slice_with_jit_module(&jit, &record, &store, &handle, &[]).unwrap() {
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
        let heap = Heap::new::<Compact>(256);
        let last_payload = Cell::new(usize::MAX);
        let transport = CountingTransport {
            last_payload: &last_payload,
        };
        let session =
            JitSafepointSession::<crate::ptr_policy::LowBitPtrPolicy<3>, _>::new(&heap, transport, &[]);
        let mut frame = [0u64; 2];

        session.with_installed(|| {
            active_jit_safepoint_handler(frame.as_mut_ptr().cast::<u8>(), 7);
        });

        assert_eq!(last_payload.get(), 7);
    }
}
