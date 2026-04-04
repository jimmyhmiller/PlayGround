use std::cell::RefCell;
use std::marker::PhantomData;

use dynexec::{
    CapturedCallerResume, CapturedFrame, DefaultExecutionConfig, ExecutionConfig,
    FrameResumePoint, FrameSliceError, FrameSliceMode, FrameSliceSnapshot, FrameSliceStore,
    InMemoryFrameSliceStore, LayoutConfigDefaults, RootPrecision, RootStrategy, RootTransport,
    RootTransportKind, ValueLayout,
};
use dynvalue::Decoded;

use crate::ir::*;
use crate::types::Type;

/// Successful execution result.
#[derive(Debug, PartialEq)]
pub enum InterpResult {
    Value(u64),
    Void,
    Deopt {
        deopt_id: DeoptId,
        resume_point: u64,
        live_values: Vec<u64>,
    },
}

/// Errors (bugs/traps, not normal control flow).
#[derive(Debug)]
pub enum InterpError {
    Unreachable,
    UncaughtException(u64),
    UnknownExternFunc(String),
    DivideByZero,
    UnsupportedControl(String),
    FrameSlice(FrameSliceError),
}

impl From<FrameSliceError> for InterpError {
    fn from(value: FrameSliceError) -> Self {
        InterpError::FrameSlice(value)
    }
}

/// What extern callbacks return.
pub enum ExternCallResult {
    Value(Option<u64>),
    Exception(u64),
}

// ─── InterpRootManager ─────────────────────────────────────────────

/// Trait for managing GC roots across interpreter call frames.
///
/// The module interpreter calls these methods to maintain roots for all
/// active frames so the GC can find and update heap pointers.
pub trait InterpRootManager<
    L: ValueLayout,
    Roots: RootStrategy<L>,
    Transport: RootTransport<L, Roots>,
> {
    /// Push a new root frame with `gc_slot_count` slots. Returns a frame handle.
    fn push_frame(&self, gc_slot_count: usize) -> usize;
    /// Pop the most recent root frame.
    fn pop_frame(&self);
    /// Set a root slot value in a frame.
    fn set_root(&self, frame: usize, slot: usize, value: u64);
    /// Get a root slot value from a frame.
    fn get_root(&self, frame: usize, slot: usize) -> u64;
    /// Clear all slots in a frame (set to 0).
    fn clear_frame(&self, frame: usize);
    /// Trigger garbage collection.
    fn collect(&self);

    /// How precisely this root manager tracks interpreter frame values.
    fn root_precision(&self) -> RootPrecision {
        Roots::precision()
    }

    fn root_transport_kind(&self) -> RootTransportKind {
        Transport::kind()
    }
}

/// No-op root manager for when GC is not needed.
pub struct NoGcRoots;

impl<L, Roots, Transport> InterpRootManager<L, Roots, Transport> for NoGcRoots
where
    L: ValueLayout,
    Roots: RootStrategy<L>,
    Transport: RootTransport<L, Roots>,
{
    fn push_frame(&self, _gc_slot_count: usize) -> usize {
        0
    }
    fn pop_frame(&self) {}
    fn set_root(&self, _frame: usize, _slot: usize, _value: u64) {}
    fn get_root(&self, _frame: usize, _slot: usize) -> u64 {
        0
    }
    fn clear_frame(&self, _frame: usize) {}
    fn collect(&self) {}

    /// NoGcRoots has no actual root storage, so report PreciseSlots to avoid
    /// mapping all values to root slots. With ConservativeWords, sync_all_from_roots
    /// would overwrite caller frame values with 0 after internal calls.
    fn root_precision(&self) -> RootPrecision {
        RootPrecision::PreciseSlots
    }
}

// ─── ModuleInterpreter ─────────────────────────────────────────────

/// How to resume the caller when a callee frame returns or throws.
enum CallerResume {
    /// Top-level entry — return to the user.
    TopLevel,
    /// Called via Inst::Call — write return value and resume at inst_idx.
    FromCall { return_dest: Option<Value> },
    /// Called via Terminator::Invoke — jump to normal/exception block.
    FromInvoke {
        normal: BlockId,
        normal_args_vals: Vec<u64>,
        exception: BlockId,
        exception_args_vals: Vec<u64>,
        has_ret_param: bool,
    },
}

/// A call frame on the module interpreter's call stack.
struct CallFrame {
    func_idx: usize,
    vals: Vec<u64>,
    block_idx: usize,
    inst_idx: usize,
    root_frame: usize,
    val_to_slot: Vec<Option<usize>>,
    caller_resume: CallerResume,
    active_prompts: Vec<PromptId>,
    /// Stack memory for function-level stack slots. Pre-allocated based
    /// on `Function::stack_slots`. Each slot is 8-byte aligned.
    slot_memory: Vec<u64>,
}

/// What `execute_frame` produces when it needs to transfer control.
enum FrameAction {
    /// Internal function call — push a new frame.
    InternalCall {
        callee_idx: usize,
        args: Vec<u64>,
        caller_resume: CallerResume,
    },
    /// Function returned normally.
    Return(Option<u64>),
    /// Uncaught exception from an extern call.
    Exception(u64),
    /// Guard failure / deoptimization.
    Deopt {
        deopt_id: DeoptId,
        resume_point: u64,
        live_values: Vec<u64>,
    },
    CaptureSlice {
        prompt: PromptId,
        live: Vec<Value>,
        return_dest: Value,
    },
    ResumeSlice {
        slice_bits: u64,
        args: Vec<u64>,
    },
    AbortToPrompt {
        prompt: PromptId,
        args: Vec<u64>,
    },
}

/// Multi-function interpreter that executes [`Module`]s with an iterative call stack.
///
/// Supports internal function calls (IR-to-IR), extern calls, and GC root
/// management via [`InterpRootManager`].
///
/// The root manager `R` is stored by reference, so closures bound via `bind()`
/// can borrow it directly (no `Rc` needed).
pub struct ConfiguredModuleInterpreter<
    'a,
    Cfg: ExecutionConfig,
    R: InterpRootManager<Cfg::Layout, Cfg::Roots, Cfg::RootTransport> = NoGcRoots,
    S: FrameSliceStore = InMemoryFrameSliceStore,
> {
    module: &'a Module,
    roots: &'a R,
    frame_slices: RefCell<S>,
    externs: Vec<Option<Box<dyn Fn(&[u64]) -> ExternCallResult + 'a>>>,
    indirect_handler: Option<Box<dyn Fn(u64, &[u64]) -> ExternCallResult + 'a>>,
    _config: PhantomData<Cfg>,
}

pub type ModuleInterpreter<'a, S, R = NoGcRoots> =
    ConfiguredModuleInterpreter<'a, DefaultExecutionConfig<S>, R, InMemoryFrameSliceStore>;

impl<'a, Cfg: ExecutionConfig, R, S> ConfiguredModuleInterpreter<'a, Cfg, R, S>
where
    Cfg::Layout: LayoutConfigDefaults,
    R: InterpRootManager<Cfg::Layout, Cfg::Roots, Cfg::RootTransport>,
    S: FrameSliceStore,
{
    pub fn with_frame_slices(module: &'a Module, roots: &'a R, frame_slices: S) -> Self {
        let externs = (0..module.func_table.len()).map(|_| None).collect();
        ConfiguredModuleInterpreter {
            module,
            roots,
            frame_slices: RefCell::new(frame_slices),
            externs,
            indirect_handler: None,
            _config: PhantomData,
        }
    }

    pub fn new(module: &'a Module, roots: &'a R) -> Self
    where
        S: Default,
    {
        Self::with_frame_slices(module, roots, S::default())
    }

    /// Bind a closure to an extern function by FuncRef.
    pub fn bind(&mut self, fref: FuncRef, f: impl Fn(&[u64]) -> ExternCallResult + 'a) {
        self.externs[fref.index()] = Some(Box::new(f));
    }

    /// Bind a closure to an extern function by name.
    pub fn bind_by_name(&mut self, name: &str, f: impl Fn(&[u64]) -> ExternCallResult + 'a) {
        for (i, def) in self.module.func_table.iter().enumerate() {
            let func_name = match def {
                FuncDef::Internal(idx) => &self.module.functions[*idx].name,
                FuncDef::Extern(ef) => &ef.name,
            };
            if func_name == name {
                self.externs[i] = Some(Box::new(f));
                return;
            }
        }
        panic!("no func named '{}' in module", name);
    }

    /// Bind a handler for indirect calls.
    pub fn bind_indirect(&mut self, handler: impl Fn(u64, &[u64]) -> ExternCallResult + 'a) {
        self.indirect_handler = Some(Box::new(handler));
    }

    /// Run an entry function. Uses the root manager stored in this interpreter.
    pub fn run(&self, entry: FuncRef, args: &[u64]) -> Result<InterpResult, InterpError> {
        let roots = self.roots;
        let entry_idx = match &self.module.func_table[entry.index()] {
            FuncDef::Internal(idx) => *idx,
            FuncDef::Extern(_) => panic!("cannot run extern function as entry point"),
        };

        let mut stack: Vec<CallFrame> = Vec::new();
        self.push_internal_frame(&mut stack, entry_idx, args, CallerResume::TopLevel, roots);
        self.run_stack(stack, roots)
    }

    pub fn resume_snapshot(
        &self,
        snapshot: FrameSliceSnapshot,
        args: &[u64],
    ) -> Result<InterpResult, InterpError> {
        let roots = self.roots;
        let mut stack = Vec::new();
        restore_frame_slice::<Cfg::Layout, Cfg::Roots, Cfg::RootTransport, R>(
            &mut stack, snapshot, args, roots, &self.module.functions,
        )?;
        self.run_stack(stack, roots)
    }

    fn run_stack(&self, mut stack: Vec<CallFrame>, roots: &R) -> Result<InterpResult, InterpError> {
        loop {
            let action = self.execute_frame(stack.last_mut().unwrap(), roots)?;

            match action {
                FrameAction::InternalCall {
                    callee_idx,
                    args,
                    caller_resume,
                } => {
                    // Sync caller's GcPtr values to roots before pushing new frame
                    sync_all_to_roots(stack.last().unwrap(), roots);
                    self.push_internal_frame(&mut stack, callee_idx, &args, caller_resume, roots);
                }

                FrameAction::Return(ret_val) => {
                    let frame = stack.pop().unwrap();
                    roots.pop_frame();

                    if stack.is_empty() {
                        return Ok(match ret_val {
                            Some(v) => InterpResult::Value(v),
                            None => InterpResult::Void,
                        });
                    }

                    let caller = stack.last_mut().unwrap();
                    sync_all_from_roots(caller, roots);

                    match frame.caller_resume {
                        CallerResume::FromCall { return_dest } => {
                            if let (Some(dest), Some(val)) = (return_dest, ret_val) {
                                caller.vals[dest.index()] = val;
                            }
                            // caller continues at its inst_idx (already advanced)
                        }
                        CallerResume::FromInvoke {
                            normal,
                            normal_args_vals,
                            has_ret_param,
                            ..
                        } => {
                            let func = &self.module.functions[caller.func_idx];
                            let target_block = &func.blocks[normal.index()];
                            let mut param_idx = 0;
                            if has_ret_param {
                                if let Some(val) = ret_val {
                                    caller.vals[target_block.params[0].0.index()] = val;
                                }
                                param_idx = 1;
                            }
                            for (i, val) in normal_args_vals.iter().enumerate() {
                                caller.vals[target_block.params[param_idx + i].0.index()] = *val;
                            }
                            caller.block_idx = normal.index();
                            caller.inst_idx = 0;
                        }
                        CallerResume::TopLevel => unreachable!(),
                    }
                }

                FrameAction::Exception(exc) => {
                    // Unwind stack looking for an Invoke exception handler
                    loop {
                        let frame = stack.pop().unwrap();
                        roots.pop_frame();

                        match frame.caller_resume {
                            CallerResume::TopLevel => {
                                return Err(InterpError::UncaughtException(exc));
                            }
                            CallerResume::FromInvoke {
                                exception,
                                exception_args_vals,
                                ..
                            } => {
                                let caller = stack.last_mut().unwrap();
                                sync_all_from_roots(caller, roots);
                                let func = &self.module.functions[caller.func_idx];
                                let target_block = &func.blocks[exception.index()];
                                for (i, val) in exception_args_vals.iter().enumerate() {
                                    caller.vals[target_block.params[i].0.index()] = *val;
                                }
                                caller.block_idx = exception.index();
                                caller.inst_idx = 0;
                                break;
                            }
                            CallerResume::FromCall { .. } => {
                                // No exception handler, continue unwinding
                                continue;
                            }
                        }
                    }
                }

                FrameAction::Deopt {
                    deopt_id,
                    resume_point,
                    live_values,
                } => {
                    // Pop all frames and return deopt to the top-level caller
                    while let Some(_frame) = stack.pop() {
                        roots.pop_frame();
                    }
                    return Ok(InterpResult::Deopt {
                        deopt_id,
                        resume_point,
                        live_values,
                    });
                }

                FrameAction::CaptureSlice {
                    prompt,
                    live,
                    return_dest,
                } => {
                    let slice =
                        capture_frame_slice::<Cfg::Layout, Cfg::Roots, Cfg::RootTransport, R>(
                            &stack, prompt, &live, return_dest, roots,
                        )?;
                    let handle = self
                        .frame_slices
                        .borrow_mut()
                        .insert_slice(slice)
                        .map_err(InterpError::FrameSlice)?;
                    let bits = S::encode_handle(&handle);
                    stack
                        .last_mut()
                        .unwrap()
                        .vals[return_dest.index()] = bits;
                }

                FrameAction::ResumeSlice { slice_bits, args } => {
                    let handle = S::decode_handle(slice_bits).map_err(InterpError::FrameSlice)?;
                    let snapshot = {
                        let mut store = self.frame_slices.borrow_mut();
                        let snapshot = store.slice(&handle).map_err(InterpError::FrameSlice)?.clone();
                        store.mark_consumed(&handle).map_err(InterpError::FrameSlice)?;
                        snapshot
                    };
                    restore_frame_slice::<Cfg::Layout, Cfg::Roots, Cfg::RootTransport, R>(
                        &mut stack, snapshot, &args, roots, &self.module.functions,
                    )?;
                }

                FrameAction::AbortToPrompt { prompt, args } => {
                    let ret_val = args.first().copied();

                    // Find which frame owns the prompt (search from top of stack).
                    // The top frame is the one that just executed the abort terminator.
                    let prompt_frame_idx = {
                        let mut found = None;
                        for i in (0..stack.len()).rev() {
                            if stack[i].active_prompts.contains(&prompt) {
                                found = Some(i);
                                break;
                            }
                        }
                        found.expect("abort_to_prompt: no frame has the target prompt")
                    };

                    // Pop all frames above the prompt owner (the aborting frame and any in between).
                    while stack.len() > prompt_frame_idx + 1 {
                        stack.pop().unwrap();
                        roots.pop_frame();
                    }

                    // Now stack.last() is the frame that owns the prompt.
                    // Find the handler block by scanning function for PushPrompt with matching prompt.
                    let frame = stack.last_mut().unwrap();
                    sync_all_from_roots(frame, roots);

                    let func = &self.module.functions[frame.func_idx];

                    let mut handler_block = None;
                    for blk in &func.blocks {
                        for node in &blk.insts {
                            if let Inst::PushPrompt(p, h) = &node.inst {
                                if *p == prompt {
                                    handler_block = Some(*h);
                                    break;
                                }
                            }
                        }
                        if handler_block.is_some() { break; }
                    }
                    let handler_block = handler_block.expect(
                        "abort_to_prompt: could not find PushPrompt instruction for prompt in owning frame"
                    );

                    // Pop the prompt from active_prompts.
                    let pos = frame.active_prompts.iter().rposition(|p| *p == prompt).unwrap();
                    frame.active_prompts.remove(pos);

                    // Write the abort value into the handler block's first parameter.
                    let hb = &func.blocks[handler_block.index()];
                    if let Some(val) = ret_val {
                        if let Some((param, _)) = hb.params.first() {
                            frame.vals[param.index()] = val;
                        }
                    }

                    // Jump to handler block.
                    frame.block_idx = handler_block.index();
                    frame.inst_idx = 0;
                }
            }
        }
    }

    fn push_internal_frame(
        &self,
        stack: &mut Vec<CallFrame>,
        func_idx: usize,
        args: &[u64],
        caller_resume: CallerResume,
        roots: &R,
    ) {
        let func = &self.module.functions[func_idx];
        let val_to_slot = build_gc_slot_map(func, roots.root_precision());
        let gc_slots = count_gc_slots_from_map(&val_to_slot);
        let root_frame = roots.push_frame(gc_slots);

        let mut vals = vec![0u64; func.value_types.len()];
        for (i, (v, _)) in func.blocks[0].params.iter().enumerate() {
            vals[v.index()] = args[i];
        }

        // Pre-allocate stack slot memory (each slot is 8 bytes)
        let slot_memory = vec![0u64; func.stack_slots.len()];

        stack.push(CallFrame {
            func_idx,
            vals,
            block_idx: 0,
            inst_idx: 0,
            root_frame,
            val_to_slot,
            caller_resume,
            active_prompts: Vec::new(),
            slot_memory,
        });
    }

    /// Execute the current frame until it needs to transfer control (call, return, exception).
    fn execute_frame(&self, frame: &mut CallFrame, roots: &R) -> Result<FrameAction, InterpError> {
        let func = &self.module.functions[frame.func_idx];

        loop {
            let block = &func.blocks[frame.block_idx];

            // Execute instructions
            while frame.inst_idx < block.insts.len() {
                let node = &block.insts[frame.inst_idx];

                match &node.inst {
                    Inst::PushPrompt(prompt, _handler) => {
                        frame.active_prompts.push(*prompt);
                        frame.inst_idx += 1;
                        continue;
                    }

                    Inst::PopPrompt(prompt) => {
                        let popped = frame.active_prompts.pop();
                        assert_eq!(popped, Some(*prompt), "prompt stack mismatch in interpreter");
                        frame.inst_idx += 1;
                        continue;
                    }

                    Inst::CaptureSlice(prompt, live) => {
                        let return_dest = node.value.expect("capture_slice must produce a value");
                        frame.inst_idx += 1;
                        return Ok(FrameAction::CaptureSlice {
                            prompt: *prompt,
                            live: live.clone(),
                            return_dest,
                        });
                    }

                    Inst::CloneSlice(slice) => {
                        let dest = node.value.expect("clone_slice must produce a value");
                        let handle = S::decode_handle(frame.vals[slice.index()])
                            .map_err(InterpError::FrameSlice)?;
                        let cloned = self
                            .frame_slices
                            .borrow_mut()
                            .clone_slice(&handle)
                            .map_err(InterpError::FrameSlice)?;
                        frame.vals[dest.index()] = S::encode_handle(&cloned);
                        frame.inst_idx += 1;
                        continue;
                    }

                    Inst::StackAddr(slot) => {
                        let dest = node.value.expect("stack_addr must produce a value");
                        let ptr = unsafe {
                            frame.slot_memory.as_ptr().add(slot.index())
                        } as u64;
                        frame.vals[dest.index()] = ptr;
                        frame.inst_idx += 1;
                        continue;
                    }

                    Inst::Safepoint(live) => {
                        handle_safepoint(frame, live, roots);
                        frame.inst_idx += 1;
                        continue;
                    }

                    Inst::Call(fref, call_args) => {
                        match &self.module.func_table[fref.index()] {
                            FuncDef::Internal(callee_idx) => {
                                let callee_idx = *callee_idx;
                                let arg_vals =
                                    call_args.iter().map(|v| frame.vals[v.index()]).collect();
                                let return_dest = node.value;
                                frame.inst_idx += 1;
                                return Ok(FrameAction::InternalCall {
                                    callee_idx,
                                    args: arg_vals,
                                    caller_resume: CallerResume::FromCall { return_dest },
                                });
                            }
                            FuncDef::Extern(_) => {
                                let arg_vals: Vec<u64> =
                                    call_args.iter().map(|v| frame.vals[v.index()]).collect();
                                match self.call_extern(*fref, &arg_vals)? {
                                    ExternCallResult::Value(ret) => {
                                        if let (Some(dest), Some(val)) = (node.value, ret) {
                                            frame.vals[dest.index()] = val;
                                        }
                                    }
                                    ExternCallResult::Exception(exc) => {
                                        return Ok(FrameAction::Exception(exc));
                                    }
                                }
                            }
                        }
                        frame.inst_idx += 1;
                        continue;
                    }

                    Inst::CallIndirect(callee, call_args, _ret_ty) => {
                        let callee_val = frame.vals[callee.index()];
                        let arg_vals: Vec<u64> =
                            call_args.iter().map(|v| frame.vals[v.index()]).collect();
                        let handler = self.indirect_handler.as_ref().ok_or_else(|| {
                            InterpError::UnknownExternFunc("(indirect)".to_string())
                        })?;
                        match handler(callee_val, &arg_vals) {
                            ExternCallResult::Value(ret) => {
                                if let (Some(dest), Some(val)) = (node.value, ret) {
                                    frame.vals[dest.index()] = val;
                                }
                            }
                            ExternCallResult::Exception(exc) => {
                                return Ok(FrameAction::Exception(exc));
                            }
                        }
                        frame.inst_idx += 1;
                        continue;
                    }

                    other => {
                        let result = exec_non_call_inst::<Cfg::Layout>(
                            other,
                            &frame.vals,
                            |v| func.value_type(v),
                            &func.deopt_info,
                        )?;
                        if let Some(r) = result {
                            match r {
                                InstResult::Val(v) => {
                                    if let Some(dest) = node.value {
                                        frame.vals[dest.index()] = v;
                                    }
                                }
                                InstResult::Deopt {
                                    deopt_id,
                                    resume_point,
                                    live_values,
                                } => {
                                    return Ok(FrameAction::Deopt {
                                        deopt_id,
                                        resume_point,
                                        live_values,
                                    });
                                }
                            }
                        }
                        frame.inst_idx += 1;
                        continue;
                    }
                }
            }

            // Execute terminator
            match &block.terminator {
                Terminator::Ret(v) => {
                    return Ok(FrameAction::Return(Some(frame.vals[v.index()])));
                }
                Terminator::RetVoid => {
                    return Ok(FrameAction::Return(None));
                }
                Terminator::Jump(target, args) => {
                    transfer_args_in_frame(frame, *target, args, func);
                    frame.block_idx = target.index();
                    frame.inst_idx = 0;
                }
                Terminator::BrIf {
                    cond,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                } => {
                    if frame.vals[cond.index()] != 0 {
                        transfer_args_in_frame(frame, *then_block, then_args, func);
                        frame.block_idx = then_block.index();
                    } else {
                        transfer_args_in_frame(frame, *else_block, else_args, func);
                        frame.block_idx = else_block.index();
                    }
                    frame.inst_idx = 0;
                }
                Terminator::Switch {
                    val,
                    cases,
                    default_block,
                    default_args,
                } => {
                    let v = frame.vals[val.index()] as i64;
                    let mut matched = false;
                    for (case_val, target, case_args) in cases {
                        if v == *case_val {
                            transfer_args_in_frame(frame, *target, case_args, func);
                            frame.block_idx = target.index();
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        transfer_args_in_frame(frame, *default_block, default_args, func);
                        frame.block_idx = default_block.index();
                    }
                    frame.inst_idx = 0;
                }

                Terminator::Invoke {
                    func: fref,
                    args: call_args,
                    normal,
                    normal_args,
                    exception,
                    exception_args,
                } => match &self.module.func_table[fref.index()] {
                    FuncDef::Internal(callee_idx) => {
                        let callee_idx = *callee_idx;
                        let arg_vals = call_args.iter().map(|v| frame.vals[v.index()]).collect();
                        let callee_sig = &self.module.functions[callee_idx].sig;
                        let normal_args_vals =
                            normal_args.iter().map(|v| frame.vals[v.index()]).collect();
                        let exception_args_vals = exception_args
                            .iter()
                            .map(|v| frame.vals[v.index()])
                            .collect();
                        return Ok(FrameAction::InternalCall {
                            callee_idx,
                            args: arg_vals,
                            caller_resume: CallerResume::FromInvoke {
                                normal: *normal,
                                normal_args_vals,
                                exception: *exception,
                                exception_args_vals,
                                has_ret_param: callee_sig.ret.is_some(),
                            },
                        });
                    }
                    FuncDef::Extern(_) => {
                        let arg_vals: Vec<u64> =
                            call_args.iter().map(|v| frame.vals[v.index()]).collect();
                        match self.call_extern(*fref, &arg_vals)? {
                            ExternCallResult::Value(ret) => {
                                let target_block = &func.blocks[normal.index()];
                                let mut param_idx = 0;
                                if let Some(ret_val) = ret {
                                    if !target_block.params.is_empty() {
                                        frame.vals[target_block.params[0].0.index()] = ret_val;
                                        param_idx = 1;
                                    }
                                }
                                let extra: Vec<u64> =
                                    normal_args.iter().map(|v| frame.vals[v.index()]).collect();
                                for (i, val) in extra.iter().enumerate() {
                                    frame.vals[target_block.params[param_idx + i].0.index()] = *val;
                                }
                                frame.block_idx = normal.index();
                                frame.inst_idx = 0;
                            }
                            ExternCallResult::Exception(_exc) => {
                                transfer_args_in_frame(frame, *exception, exception_args, func);
                                frame.block_idx = exception.index();
                                frame.inst_idx = 0;
                            }
                        }
                    }
                },

                Terminator::InvokeIndirect {
                    callee,
                    args: call_args,
                    ret_ty: _,
                    normal,
                    normal_args,
                    exception,
                    exception_args,
                } => {
                    let callee_val = frame.vals[callee.index()];
                    let arg_vals: Vec<u64> =
                        call_args.iter().map(|v| frame.vals[v.index()]).collect();
                    let handler = self
                        .indirect_handler
                        .as_ref()
                        .ok_or_else(|| InterpError::UnknownExternFunc("(indirect)".to_string()))?;
                    match handler(callee_val, &arg_vals) {
                        ExternCallResult::Value(ret) => {
                            let target_block = &func.blocks[normal.index()];
                            let mut param_idx = 0;
                            if let Some(ret_val) = ret {
                                if !target_block.params.is_empty() {
                                    frame.vals[target_block.params[0].0.index()] = ret_val;
                                    param_idx = 1;
                                }
                            }
                            let extra: Vec<u64> =
                                normal_args.iter().map(|v| frame.vals[v.index()]).collect();
                            for (i, val) in extra.iter().enumerate() {
                                frame.vals[target_block.params[param_idx + i].0.index()] = *val;
                            }
                            frame.block_idx = normal.index();
                            frame.inst_idx = 0;
                        }
                        ExternCallResult::Exception(_exc) => {
                            transfer_args_in_frame(frame, *exception, exception_args, func);
                            frame.block_idx = exception.index();
                            frame.inst_idx = 0;
                        }
                    }
                }
                Terminator::ResumeSlice { slice, args } => {
                    let arg_vals = args.iter().map(|v| frame.vals[v.index()]).collect();
                    return Ok(FrameAction::ResumeSlice {
                        slice_bits: frame.vals[slice.index()],
                        args: arg_vals,
                    });
                }

                Terminator::AbortToPrompt { prompt, args } => {
                    let arg_vals = args.iter().map(|v| frame.vals[v.index()]).collect();
                    return Ok(FrameAction::AbortToPrompt {
                        prompt: *prompt,
                        args: arg_vals,
                    });
                }

                Terminator::Unreachable => {
                    return Err(InterpError::Unreachable);
                }
            }
        }
    }

    fn call_extern(&self, fref: FuncRef, args: &[u64]) -> Result<ExternCallResult, InterpError> {
        match &self.externs[fref.index()] {
            Some(f) => Ok(f(args)),
            None => Err(InterpError::UnknownExternFunc(
                self.module.func_name(fref).to_string(),
            )),
        }
    }
}

// ─── Shared instruction execution ──────────────────────────────────

enum InstResult {
    Val(u64),
    Deopt {
        deopt_id: DeoptId,
        resume_point: u64,
        live_values: Vec<u64>,
    },
}

/// Execute an instruction that is NOT Call, CallIndirect, or Safepoint.
/// Those must be handled by the caller (they need interpreter-level context).
fn exec_non_call_inst<S: ValueLayout>(
    inst: &Inst,
    vals: &[u64],
    val_type: impl Fn(Value) -> Type,
    deopt_info: &[DeoptInfo],
) -> Result<Option<InstResult>, InterpError> {
    let v = |val: &Value| vals[val.index()];
    let ty = |val: &Value| val_type(*val);

    match inst {
        // Constants
        Inst::Iconst(t, imm) => Ok(Some(InstResult::Val(mask(*imm as u64, *t)))),
        Inst::F64Const(f) => Ok(Some(InstResult::Val(f.to_bits()))),

        // Integer arithmetic
        Inst::Add(a, b) => {
            let res_ty = arith_result_type(ty(a), ty(b));
            Ok(Some(InstResult::Val(mask(v(a).wrapping_add(v(b)), res_ty))))
        }
        Inst::Sub(a, b) => {
            let res_ty = arith_result_type(ty(a), ty(b));
            Ok(Some(InstResult::Val(mask(v(a).wrapping_sub(v(b)), res_ty))))
        }
        Inst::Mul(a, b) => {
            let res_ty = arith_result_type(ty(a), ty(b));
            Ok(Some(InstResult::Val(mask(v(a).wrapping_mul(v(b)), res_ty))))
        }
        Inst::SDiv(a, b) => {
            let vb = v(b);
            if vb == 0 {
                return Err(InterpError::DivideByZero);
            }
            let res_ty = arith_result_type(ty(a), ty(b));
            let sa = sign_extend(v(a), res_ty);
            let sb = sign_extend(vb, res_ty);
            Ok(Some(InstResult::Val(mask(
                (sa.wrapping_div(sb)) as u64,
                res_ty,
            ))))
        }
        Inst::UDiv(a, b) => {
            let vb = v(b);
            if vb == 0 {
                return Err(InterpError::DivideByZero);
            }
            let res_ty = arith_result_type(ty(a), ty(b));
            Ok(Some(InstResult::Val(mask(v(a).wrapping_div(vb), res_ty))))
        }

        // Float arithmetic
        Inst::FAdd(a, b) => {
            let fa = f64::from_bits(v(a));
            let fb = f64::from_bits(v(b));
            Ok(Some(InstResult::Val((fa + fb).to_bits())))
        }
        Inst::FSub(a, b) => {
            let fa = f64::from_bits(v(a));
            let fb = f64::from_bits(v(b));
            Ok(Some(InstResult::Val((fa - fb).to_bits())))
        }
        Inst::FMul(a, b) => {
            let fa = f64::from_bits(v(a));
            let fb = f64::from_bits(v(b));
            Ok(Some(InstResult::Val((fa * fb).to_bits())))
        }
        Inst::FDiv(a, b) => {
            let fa = f64::from_bits(v(a));
            let fb = f64::from_bits(v(b));
            Ok(Some(InstResult::Val((fa / fb).to_bits())))
        }

        // Bitwise
        Inst::And(a, b) => Ok(Some(InstResult::Val(v(a) & v(b)))),
        Inst::Or(a, b) => Ok(Some(InstResult::Val(v(a) | v(b)))),
        Inst::Xor(a, b) => Ok(Some(InstResult::Val(mask(v(a) ^ v(b), ty(a))))),
        Inst::Shl(a, b) => Ok(Some(InstResult::Val(mask(v(a) << (v(b) & 63), ty(a))))),
        Inst::LShr(a, b) => Ok(Some(InstResult::Val(mask(v(a) >> (v(b) & 63), ty(a))))),
        Inst::AShr(a, b) => {
            let t = ty(a);
            let sa = sign_extend(v(a), t);
            Ok(Some(InstResult::Val(mask((sa >> (v(b) & 63)) as u64, t))))
        }

        // Unary
        Inst::Neg(val) => {
            let t = ty(val);
            Ok(Some(InstResult::Val(mask(0u64.wrapping_sub(v(val)), t))))
        }
        Inst::FNeg(val) => {
            let f = f64::from_bits(v(val));
            Ok(Some(InstResult::Val((-f).to_bits())))
        }
        Inst::Not(val) => {
            let t = ty(val);
            Ok(Some(InstResult::Val(mask(!v(val), t))))
        }

        // Comparison
        Inst::Icmp(op, a, b) => {
            let t = ty(a);
            let result = match op {
                CmpOp::Eq => v(a) == v(b),
                CmpOp::Ne => v(a) != v(b),
                CmpOp::Slt => sign_extend(v(a), t) < sign_extend(v(b), t),
                CmpOp::Sle => sign_extend(v(a), t) <= sign_extend(v(b), t),
                CmpOp::Sgt => sign_extend(v(a), t) > sign_extend(v(b), t),
                CmpOp::Sge => sign_extend(v(a), t) >= sign_extend(v(b), t),
                CmpOp::Ult => v(a) < v(b),
                CmpOp::Ule => v(a) <= v(b),
                CmpOp::Ugt => v(a) > v(b),
                CmpOp::Uge => v(a) >= v(b),
            };
            Ok(Some(InstResult::Val(result as u64)))
        }
        Inst::Fcmp(op, a, b) => {
            let fa = f64::from_bits(v(a));
            let fb = f64::from_bits(v(b));
            let result = match op {
                CmpOp::Eq => fa == fb,
                CmpOp::Ne => fa != fb,
                CmpOp::Slt | CmpOp::Ult => fa < fb,
                CmpOp::Sle | CmpOp::Ule => fa <= fb,
                CmpOp::Sgt | CmpOp::Ugt => fa > fb,
                CmpOp::Sge | CmpOp::Uge => fa >= fb,
            };
            Ok(Some(InstResult::Val(result as u64)))
        }

        // Conversions
        Inst::Sext(val, to) => {
            let from_ty = ty(val);
            let extended = sign_extend(v(val), from_ty) as u64;
            Ok(Some(InstResult::Val(mask(extended, *to))))
        }
        Inst::Zext(val, _to) => Ok(Some(InstResult::Val(v(val)))),
        Inst::Trunc(val, to) => Ok(Some(InstResult::Val(mask(v(val), *to)))),
        Inst::IntToFloat(val) => {
            let i = v(val) as i64;
            Ok(Some(InstResult::Val((i as f64).to_bits())))
        }
        Inst::FloatToInt(val) => {
            let f = f64::from_bits(v(val));
            Ok(Some(InstResult::Val(f as i64 as u64)))
        }
        Inst::Bitcast(val, _to) => Ok(Some(InstResult::Val(v(val)))),

        // Memory
        Inst::Load(load_ty, addr, offset) => {
            let ptr = (v(addr) as isize + *offset as isize) as *const u8;
            let result = unsafe {
                match load_ty.size_bytes() {
                    1 => std::ptr::read_unaligned(ptr) as u64,
                    4 => std::ptr::read_unaligned(ptr as *const u32) as u64,
                    8 => std::ptr::read_unaligned(ptr as *const u64),
                    _ => unreachable!(),
                }
            };
            Ok(Some(InstResult::Val(result)))
        }
        Inst::Store(val, addr, offset) => {
            let ptr = (v(addr) as isize + *offset as isize) as *mut u8;
            unsafe {
                match val_type(*val).size_bytes() {
                    1 => std::ptr::write_unaligned(ptr, v(val) as u8),
                    4 => std::ptr::write_unaligned(ptr as *mut u32, v(val) as u32),
                    8 => std::ptr::write_unaligned(ptr as *mut u64, v(val)),
                    _ => unreachable!(),
                }
            }
            Ok(None)
        }

        // Tagged values
        Inst::TagOf(val) => {
            let bits = v(val);
            let tag = match S::decode(bits) {
                Decoded::Tagged { tag, .. } => tag,
                Decoded::Float(_) => panic!("TagOf on unboxed float"),
            };
            Ok(Some(InstResult::Val(tag as u64)))
        }
        Inst::Payload(val) => Ok(Some(InstResult::Val(S::extract_payload(v(val))))),
        Inst::MakeTagged(tag, payload) => {
            Ok(Some(InstResult::Val(S::encode_tagged(*tag, v(payload)))))
        }
        Inst::IsTag(val, tag) => Ok(Some(InstResult::Val(S::has_tag(v(val), *tag) as u64))),

        // Select
        Inst::Select(cond, t, f) => Ok(Some(InstResult::Val(if v(cond) != 0 {
            v(t)
        } else {
            v(f)
        }))),

        // Overflow checking
        Inst::OverflowCheck(op, a, b) => {
            let va = v(a);
            let vb = v(b);
            let t = ty(a);
            let overflowed = match (op, t) {
                (OverflowOp::SAdd, Type::I8) => (va as u8 as i8).overflowing_add(vb as u8 as i8).1,
                (OverflowOp::SAdd, Type::I32) => {
                    (va as u32 as i32).overflowing_add(vb as u32 as i32).1
                }
                (OverflowOp::SAdd, _) => (va as i64).overflowing_add(vb as i64).1,
                (OverflowOp::SSub, Type::I8) => (va as u8 as i8).overflowing_sub(vb as u8 as i8).1,
                (OverflowOp::SSub, Type::I32) => {
                    (va as u32 as i32).overflowing_sub(vb as u32 as i32).1
                }
                (OverflowOp::SSub, _) => (va as i64).overflowing_sub(vb as i64).1,
                (OverflowOp::SMul, Type::I8) => (va as u8 as i8).overflowing_mul(vb as u8 as i8).1,
                (OverflowOp::SMul, Type::I32) => {
                    (va as u32 as i32).overflowing_mul(vb as u32 as i32).1
                }
                (OverflowOp::SMul, _) => (va as i64).overflowing_mul(vb as i64).1,
                (OverflowOp::UAdd, Type::I8) => (va as u8).overflowing_add(vb as u8).1,
                (OverflowOp::UAdd, Type::I32) => (va as u32).overflowing_add(vb as u32).1,
                (OverflowOp::UAdd, _) => va.overflowing_add(vb).1,
                (OverflowOp::USub, Type::I8) => (va as u8).overflowing_sub(vb as u8).1,
                (OverflowOp::USub, Type::I32) => (va as u32).overflowing_sub(vb as u32).1,
                (OverflowOp::USub, _) => va.overflowing_sub(vb).1,
                (OverflowOp::UMul, Type::I8) => (va as u8).overflowing_mul(vb as u8).1,
                (OverflowOp::UMul, Type::I32) => (va as u32).overflowing_mul(vb as u32).1,
                (OverflowOp::UMul, _) => va.overflowing_mul(vb).1,
            };
            Ok(Some(InstResult::Val(overflowed as u64)))
        }

        // Guard
        Inst::Guard(cond, deopt_id, live) => {
            if v(cond) != 0 {
                Ok(None)
            } else {
                let info = &deopt_info[deopt_id.index()];
                let live_values: Vec<u64> = live.iter().map(|val| v(val)).collect();
                Ok(Some(InstResult::Deopt {
                    deopt_id: *deopt_id,
                    resume_point: info.resume_point,
                    live_values,
                }))
            }
        }

        // Call/CallIndirect/Safepoint/StackAddr and frame-slice control must be handled by the caller
        Inst::Call(..)
        | Inst::CallIndirect(..)
        | Inst::Safepoint(..)
        | Inst::StackAddr(..)
        | Inst::PushPrompt(..)
        | Inst::PopPrompt(..)
        | Inst::CaptureSlice(..)
        | Inst::CloneSlice(..) => {
            panic!("Call/CallIndirect/Safepoint/StackAddr must be handled by the interpreter, not exec_non_call_inst");
        }
    }
}

// ─── GC root helpers ───────────────────────────────────────────────

/// Build a mapping from SSA value index to GC root slot index.
///
/// Only "base" GcPtr values get root slots. Values derived from pointer
/// arithmetic (Add, Sub, Mul, etc. on a GcPtr) are interior/derived pointers
/// that point inside objects, not at valid object headers. They must NOT be
/// GC roots — the GC would treat them as object pointers, read garbage
/// headers, and crash.
///
/// Base GcPtr values: block parameters, call/invoke results, loads, iconst,
/// select, and any other instruction where the GcPtr points to an object start.
///
/// Derived GcPtr values: arithmetic results where GcPtr type propagates
/// through Add/Sub/Mul/Div (the `arith_result_type` function).
fn build_gc_slot_map(func: &Function, precision: RootPrecision) -> Vec<Option<usize>> {
    if precision == RootPrecision::ConservativeWords {
        // Every value gets a root slot — needed for NanBox runtimes where
        // any I64 value might encode a heap pointer.
        let mut map = vec![None; func.value_types.len()];
        for i in 0..func.value_types.len() {
            map[i] = Some(i);
        }
        return map;
    }

    let mut derived = vec![false; func.value_types.len()];
    for block in &func.blocks {
        for node in &block.insts {
            if let Some(v) = node.value {
                if func.value_types[v.index()] == Type::GcPtr {
                    match &node.inst {
                        // All integer arithmetic ops that can produce GcPtr
                        // via arith_result_type propagation
                        Inst::Add(_, _)
                        | Inst::Sub(_, _)
                        | Inst::Mul(_, _)
                        | Inst::SDiv(_, _)
                        | Inst::UDiv(_, _) => {
                            derived[v.index()] = true;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let mut map = vec![None; func.value_types.len()];
    let mut slot = 0;
    for (i, ty) in func.value_types.iter().enumerate() {
        if ty.is_gc() && !derived[i] {
            map[i] = Some(slot);
            slot += 1;
        }
    }
    map
}

fn count_gc_slots_from_map(map: &[Option<usize>]) -> usize {
    map.iter().filter(|s| s.is_some()).count()
}

fn serialize_caller_resume(resume: &CallerResume) -> CapturedCallerResume {
    match resume {
        CallerResume::TopLevel => CapturedCallerResume::TopLevel,
        CallerResume::FromCall { return_dest } => CapturedCallerResume::FromCall {
            return_dest: return_dest.map(|value| value.index()),
        },
        CallerResume::FromInvoke {
            normal,
            normal_args_vals,
            exception,
            exception_args_vals,
            has_ret_param,
        } => CapturedCallerResume::FromInvoke {
            normal_block: normal.index(),
            normal_args_vals: normal_args_vals.clone(),
            exception_block: exception.index(),
            exception_args_vals: exception_args_vals.clone(),
            has_ret_param: *has_ret_param,
        },
    }
}

fn deserialize_caller_resume(resume: &CapturedCallerResume) -> CallerResume {
    match resume {
        CapturedCallerResume::TopLevel => CallerResume::TopLevel,
        CapturedCallerResume::FromCall { return_dest } => CallerResume::FromCall {
            return_dest: return_dest.map(Value::from_index),
        },
        CapturedCallerResume::FromInvoke {
            normal_block,
            normal_args_vals,
            exception_block,
            exception_args_vals,
            has_ret_param,
        } => CallerResume::FromInvoke {
            normal: BlockId::from_index(*normal_block),
            normal_args_vals: normal_args_vals.clone(),
            exception: BlockId::from_index(*exception_block),
            exception_args_vals: exception_args_vals.clone(),
            has_ret_param: *has_ret_param,
        },
    }
}

fn capture_frame_slice<L, Roots, Transport, R>(
    stack: &[CallFrame],
    prompt: PromptId,
    _live: &[Value],
    return_dest: Value,
    roots: &R,
) -> Result<FrameSliceSnapshot, InterpError>
where
    L: ValueLayout,
    Roots: RootStrategy<L>,
    Transport: RootTransport<L, Roots>,
    R: InterpRootManager<L, Roots, Transport>,
{
    let start_idx = stack
        .iter()
        .enumerate()
        .rfind(|(_, frame)| frame.active_prompts.contains(&prompt))
        .map(|(idx, _)| idx)
        .ok_or_else(|| {
            InterpError::UnsupportedControl(format!("capture_slice: prompt {:?} is not active", prompt))
        })?;

    let mut frames = Vec::new();
    for frame in &stack[start_idx..] {
        let mut vals = frame.vals.clone();
        for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
            if let Some(slot) = slot_opt {
                vals[i] = roots.get_root(frame.root_frame, *slot);
            }
        }
        let root_value_indices = frame
            .val_to_slot
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.map(|_| idx))
            .collect();
        frames.push(CapturedFrame {
            resume: FrameResumePoint {
                func_idx: frame.func_idx,
                block_idx: frame.block_idx,
                inst_idx: frame.inst_idx,
            },
            values: vals,
            root_value_indices,
            resume_arg_value_indices: Vec::new(),
            active_prompts: frame.active_prompts.iter().map(|p| p.0).collect(),
            caller_resume: serialize_caller_resume(&frame.caller_resume),
        });
    }
    if let Some(top) = frames.last_mut() {
        top.resume_arg_value_indices = vec![return_dest.index()];
    }

    Ok(FrameSliceSnapshot {
        prompt_id: prompt.0,
        mode: FrameSliceMode::OneShot,
        frames,
        consumed: false,
    })
}

fn restore_frame_slice<L, Roots, Transport, R>(
    stack: &mut Vec<CallFrame>,
    snapshot: FrameSliceSnapshot,
    args: &[u64],
    roots: &R,
    functions: &[Function],
) -> Result<(), InterpError>
where
    L: ValueLayout,
    Roots: RootStrategy<L>,
    Transport: RootTransport<L, Roots>,
    R: InterpRootManager<L, Roots, Transport>,
{
    while stack.pop().is_some() {
        roots.pop_frame();
    }

    for (frame_idx, captured) in snapshot.frames.iter().enumerate() {
        let func_idx = captured.resume.func_idx;
        let root_frame = roots.push_frame(captured.root_value_indices.len());
        let mut val_to_slot = vec![None; captured.values.len()];
        for (slot, &value_idx) in captured.root_value_indices.iter().enumerate() {
            val_to_slot[value_idx] = Some(slot);
        }
        let mut vals = captured.values.clone();
        if frame_idx + 1 == snapshot.frames.len() {
            for (idx, value_idx) in captured.resume_arg_value_indices.iter().copied().enumerate() {
                if let Some(arg) = args.get(idx) {
                    vals[value_idx] = *arg;
                }
            }
        }

        let frame = CallFrame {
            func_idx,
            vals,
            block_idx: captured.resume.block_idx,
            inst_idx: captured.resume.inst_idx,
            root_frame,
            val_to_slot,
            caller_resume: deserialize_caller_resume(&captured.caller_resume),
            active_prompts: captured
                .active_prompts
                .iter()
                .copied()
                .map(PromptId)
                .collect(),
            slot_memory: vec![0u64; functions[func_idx].stack_slots.len()],
        };
        sync_all_to_roots::<L, Roots, Transport, R>(&frame, roots);
        stack.push(frame);
    }

    Ok(())
}

/// Sync all GcPtr values from a frame into root slots.
fn sync_all_to_roots<L, Roots, Transport, R>(frame: &CallFrame, roots: &R)
where
    L: ValueLayout,
    Roots: RootStrategy<L>,
    Transport: RootTransport<L, Roots>,
    R: InterpRootManager<L, Roots, Transport>,
{
    roots.clear_frame(frame.root_frame);
    for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
        if let Some(slot) = slot_opt {
            roots.set_root(frame.root_frame, *slot, frame.vals[i]);
        }
    }
}

/// Read back all GcPtr values from root slots into a frame (GC may have forwarded them).
fn sync_all_from_roots<L, Roots, Transport, R>(frame: &mut CallFrame, roots: &R)
where
    L: ValueLayout,
    Roots: RootStrategy<L>,
    Transport: RootTransport<L, Roots>,
    R: InterpRootManager<L, Roots, Transport>,
{
    for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
        if let Some(slot) = slot_opt {
            frame.vals[i] = roots.get_root(frame.root_frame, *slot);
        }
    }
}

/// Handle a safepoint: sync live GcPtr values to roots, collect, read back.
fn handle_safepoint<L, Roots, Transport, R>(frame: &mut CallFrame, live: &[Value], roots: &R)
where
    L: ValueLayout,
    Roots: RootStrategy<L>,
    Transport: RootTransport<L, Roots>,
    R: InterpRootManager<L, Roots, Transport>,
{
    if roots.root_precision() == RootPrecision::ConservativeWords {
        // Sync ALL frame values as potential roots (NanBox-style).
        for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
            if let Some(slot) = slot_opt {
                roots.set_root(frame.root_frame, *slot, frame.vals[i]);
            }
        }
        roots.collect();
        for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
            if let Some(slot) = slot_opt {
                frame.vals[i] = roots.get_root(frame.root_frame, *slot);
            }
        }
    } else {
        roots.clear_frame(frame.root_frame);
        for v in live {
            if let Some(slot) = frame.val_to_slot[v.index()] {
                roots.set_root(frame.root_frame, slot, frame.vals[v.index()]);
            }
        }
        roots.collect();
        for v in live {
            if let Some(slot) = frame.val_to_slot[v.index()] {
                frame.vals[v.index()] = roots.get_root(frame.root_frame, slot);
            }
        }
    }
}

// ─── Shared helpers ────────────────────────────────────────────────

/// Transfer block arguments within a CallFrame.
fn transfer_args_in_frame(frame: &mut CallFrame, target: BlockId, args: &[Value], func: &Function) {
    let arg_vals: Vec<u64> = args.iter().map(|v| frame.vals[v.index()]).collect();
    let target_block = &func.blocks[target.index()];
    for (i, val) in arg_vals.iter().enumerate() {
        frame.vals[target_block.params[i].0.index()] = *val;
    }
}

/// Mask a u64 to the correct width for the given type.
fn mask(val: u64, ty: Type) -> u64 {
    match ty {
        Type::I8 => val & 0xFF,
        Type::I32 => val & 0xFFFF_FFFF,
        Type::I64 | Type::Ptr | Type::GcPtr | Type::F64 | Type::FrameSlice => val,
    }
}

/// Sign-extend a u64 based on source type.
fn sign_extend(val: u64, ty: Type) -> i64 {
    match ty {
        Type::I8 => val as u8 as i8 as i64,
        Type::I32 => val as u32 as i32 as i64,
        Type::I64 | Type::Ptr | Type::GcPtr | Type::FrameSlice => val as i64,
        Type::F64 => val as i64,
    }
}

/// Compute the result type for integer arithmetic (GcPtr > Ptr > int).
fn arith_result_type(a: Type, b: Type) -> Type {
    if a == Type::GcPtr || b == Type::GcPtr {
        Type::GcPtr
    } else if a == Type::Ptr || b == Type::Ptr {
        Type::Ptr
    } else {
        a
    }
}
