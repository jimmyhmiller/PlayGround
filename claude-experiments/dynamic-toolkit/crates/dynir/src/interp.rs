use std::cell::RefCell;
use std::marker::PhantomData;

use dynexec::{
    CapturedFrame, CodegenConfig, ContinuationStore, DefaultCodegenConfig,
    FrameResume, FrameResumePoint, FrameSliceError, FrameSliceMode, FrameSliceSnapshot,
    InterpFrameStore, LayoutConfigDefaults, RootPrecision,
    RootStrategy, RootTransport, RootTransportKind, ValueLayout,
    VecContinuationStore,
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

// ─── InterpStackRuntime ───────────────────────────────────────────

/// Internal frame for InterpStackRuntime.
struct InterpFrame {
    func_idx: usize,
    vals: Vec<u64>,
    slots: Vec<u64>,
    block_idx: usize,
    inst_idx: usize,
    resume: FrameResume,
    active_prompts: Vec<u32>,
    root_frame: usize,
    val_to_slot: Vec<Option<usize>>,
}


/// An `InterpFrameStore` backed by `Vec<InterpFrame>` with GC root management.
pub struct InterpStackRuntime<'a, L: ValueLayout, Roots: RootStrategy<L>, Transport: RootTransport<L, Roots>, R: InterpRootManager<L, Roots, Transport>> {
    stack: Vec<InterpFrame>,
    roots: &'a R,
    functions: &'a [Function],
    root_precision: RootPrecision,
    _phantom: PhantomData<(L, Roots, Transport)>,
}

impl<'a, L: ValueLayout, Roots: RootStrategy<L>, Transport: RootTransport<L, Roots>, R: InterpRootManager<L, Roots, Transport>> InterpStackRuntime<'a, L, Roots, Transport, R> {
    pub fn new(roots: &'a R, functions: &'a [Function]) -> Self {
        InterpStackRuntime {
            stack: Vec::new(),
            roots,
            functions,
            root_precision: roots.root_precision(),
            _phantom: PhantomData,
        }
    }

    fn sync_top_to_roots(&self) {
        if let Some(frame) = self.stack.last() {
            sync_frame_to_roots(frame, self.roots);
        }
    }

    fn sync_top_from_roots(&mut self) {
        if let Some(frame) = self.stack.last_mut() {
            sync_frame_from_roots(frame, self.roots);
        }
    }
}

fn sync_frame_to_roots<L: ValueLayout, Roots: RootStrategy<L>, Transport: RootTransport<L, Roots>, R: InterpRootManager<L, Roots, Transport>>(frame: &InterpFrame, roots: &R) {
    roots.clear_frame(frame.root_frame);
    for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
        if let Some(slot) = slot_opt {
            roots.set_root(frame.root_frame, *slot, frame.vals[i]);
        }
    }
}

fn sync_frame_from_roots<L: ValueLayout, Roots: RootStrategy<L>, Transport: RootTransport<L, Roots>, R: InterpRootManager<L, Roots, Transport>>(frame: &mut InterpFrame, roots: &R) {
    for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
        if let Some(slot) = slot_opt {
            frame.vals[i] = roots.get_root(frame.root_frame, *slot);
        }
    }
}

impl<'a, L: ValueLayout, Roots: RootStrategy<L>, Transport: RootTransport<L, Roots>, R: InterpRootManager<L, Roots, Transport>> InterpFrameStore for InterpStackRuntime<'a, L, Roots, Transport, R> {
    fn push_frame(
        &mut self,
        func_idx: usize,
        val_count: usize,
        slot_count: usize,
        args: &[(usize, u64)],
        resume: FrameResume,
    ) {
        // Sync current top frame's roots before pushing
        self.sync_top_to_roots();

        let func = &self.functions[func_idx];
        let val_to_slot = build_gc_slot_map(func, self.root_precision);
        let gc_slots = count_gc_slots_from_map(&val_to_slot);
        let root_frame = self.roots.push_frame(gc_slots);

        let mut vals = vec![0u64; val_count];
        for &(idx, val) in args {
            vals[idx] = val;
        }
        self.stack.push(InterpFrame {
            func_idx,
            vals,
            slots: vec![0u64; slot_count],
            block_idx: 0,
            inst_idx: 0,
            resume,
            active_prompts: Vec::new(),
            root_frame,
            val_to_slot,
        });
    }

    fn pop_frame(&mut self) -> FrameResume {
        let frame = self.stack.pop().expect("pop_frame on empty stack");
        self.roots.pop_frame();
        // Sync new top frame from roots (GC may have forwarded pointers)
        self.sync_top_from_roots();
        frame.resume
    }

    fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    fn get(&self, idx: usize) -> u64 {
        self.stack.last().unwrap().vals[idx]
    }

    fn set(&mut self, idx: usize, val: u64) {
        self.stack.last_mut().unwrap().vals[idx] = val;
    }

    fn slot_ptr(&self, slot_idx: usize) -> *const u64 {
        let frame = self.stack.last().unwrap();
        unsafe { frame.slots.as_ptr().add(slot_idx) }
    }

    fn func_idx(&self) -> usize {
        self.stack.last().unwrap().func_idx
    }

    fn block_idx(&self) -> usize {
        self.stack.last().unwrap().block_idx
    }

    fn set_block(&mut self, block: usize) {
        self.stack.last_mut().unwrap().block_idx = block;
    }

    fn inst_idx(&self) -> usize {
        self.stack.last().unwrap().inst_idx
    }

    fn set_inst(&mut self, inst: usize) {
        self.stack.last_mut().unwrap().inst_idx = inst;
    }

    fn advance_inst(&mut self) {
        self.stack.last_mut().unwrap().inst_idx += 1;
    }

    fn push_prompt(&mut self, prompt: u32) {
        self.stack.last_mut().unwrap().active_prompts.push(prompt);
    }

    fn pop_prompt(&mut self, prompt: u32) {
        let popped = self.stack.last_mut().unwrap().active_prompts.pop();
        assert_eq!(popped, Some(prompt));
    }

    fn find_prompt_depth(&self, prompt: u32) -> Option<usize> {
        self.stack
            .iter()
            .rposition(|f| f.active_prompts.contains(&prompt))
    }

    fn pop_frames_above(&mut self, depth: usize) {
        while self.stack.len() > depth + 1 {
            self.stack.pop();
            self.roots.pop_frame();
        }
        // Sync the new top frame from roots
        self.sync_top_from_roots();
    }

    fn capture_snapshot(&mut self, prompt: u32, resume_dest: usize) -> FrameSliceSnapshot {
        let start = self
            .stack
            .iter()
            .rposition(|f| f.active_prompts.contains(&prompt))
            .expect("capture: prompt not found");
        let frame_count = self.stack.len() - start;
        let mut frames = Vec::with_capacity(frame_count);
        for (i, f) in self.stack[start..].iter().enumerate() {
            let is_top = i + 1 == frame_count;
            // Sync values from roots to get up-to-date GC pointers
            let mut vals = f.vals.clone();
            for (vi, slot_opt) in f.val_to_slot.iter().enumerate() {
                if let Some(slot) = slot_opt {
                    vals[vi] = self.roots.get_root(f.root_frame, *slot);
                }
            }
            let root_value_indices: Vec<usize> = f.val_to_slot.iter().enumerate()
                .filter_map(|(i, s)| s.map(|_| i))
                .collect();
            frames.push(CapturedFrame {
                resume: FrameResumePoint {
                    func_idx: f.func_idx,
                    block_idx: f.block_idx,
                    inst_idx: f.inst_idx,
                },
                values: vals,
                root_value_indices,
                resume_arg_value_indices: if is_top { vec![resume_dest] } else { Vec::new() },
                active_prompts: f.active_prompts.clone(),
                caller_resume: f.resume.clone(),
            });
        }
        FrameSliceSnapshot {
            prompt_id: prompt,
            mode: FrameSliceMode::OneShot,
            frames,
            consumed: false,
        }
    }

    fn resume_snapshot(&mut self, snapshot: &FrameSliceSnapshot, args: &[u64]) {
        // Pop all existing frames
        while !self.stack.is_empty() {
            self.stack.pop();
            self.roots.pop_frame();
        }
        let frame_count = snapshot.frames.len();
        for (i, captured) in snapshot.frames.iter().enumerate() {
            let func_idx = captured.resume.func_idx;
            let mut vals = captured.values.clone();
            if i + 1 == frame_count {
                for (idx, &value_idx) in captured.resume_arg_value_indices.iter().enumerate() {
                    if let Some(&arg) = args.get(idx) {
                        vals[value_idx] = arg;
                    }
                }
            }
            // Rebuild val_to_slot from root_value_indices
            let mut val_to_slot = vec![None; vals.len()];
            for (slot, &value_idx) in captured.root_value_indices.iter().enumerate() {
                val_to_slot[value_idx] = Some(slot);
            }
            let gc_slots = count_gc_slots_from_map(&val_to_slot);
            let root_frame = self.roots.push_frame(gc_slots);
            let frame = InterpFrame {
                func_idx,
                vals,
                slots: vec![0u64; self.functions[func_idx].stack_slots.len()],
                block_idx: captured.resume.block_idx,
                inst_idx: captured.resume.inst_idx,
                resume: captured.caller_resume.clone(),
                active_prompts: captured.active_prompts.clone(),
                root_frame,
                val_to_slot,
            };
            sync_frame_to_roots(&frame, self.roots);
            self.stack.push(frame);
        }
    }

    fn needs_gc(&self) -> bool {
        false
    }

    fn collect_gc(&mut self) {
        // Sync current frame to roots, collect, sync back
        self.sync_top_to_roots();
        self.roots.collect();
        self.sync_top_from_roots();
    }

    fn safepoint(&mut self, live_indices: &[usize]) {
        let frame = self.stack.last_mut().unwrap();
        if self.root_precision == RootPrecision::ConservativeWords {
            // Sync ALL frame values as potential roots (NanBox-style).
            self.roots.clear_frame(frame.root_frame);
            for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
                if let Some(slot) = slot_opt {
                    self.roots.set_root(frame.root_frame, *slot, frame.vals[i]);
                }
            }
            self.roots.collect();
            for (i, slot_opt) in frame.val_to_slot.iter().enumerate() {
                if let Some(slot) = slot_opt {
                    frame.vals[i] = self.roots.get_root(frame.root_frame, *slot);
                }
            }
        } else {
            // Precise: only sync live GcPtr values
            self.roots.clear_frame(frame.root_frame);
            for &idx in live_indices {
                if let Some(slot) = frame.val_to_slot[idx] {
                    self.roots.set_root(frame.root_frame, slot, frame.vals[idx]);
                }
            }
            self.roots.collect();
            for &idx in live_indices {
                if let Some(slot) = frame.val_to_slot[idx] {
                    frame.vals[idx] = self.roots.get_root(frame.root_frame, slot);
                }
            }
        }
    }
}

// ─── ModuleInterpreter ─────────────────────────────────────────────

/// What `execute_frame` produces when it needs to transfer control.
enum FrameAction {
    /// Internal function call — push a new frame.
    InternalCall {
        callee_idx: usize,
        args: Vec<u64>,
        resume: FrameResume,
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
    AbortToPrompt {
        prompt: PromptId,
        args: Vec<u64>,
    },
}

/// Multi-function interpreter that executes [`Module`]s with an iterative call stack.
///
/// Supports internal function calls (IR-to-IR), extern calls, and GC root
/// management via [`InterpRootManager`].
pub struct ConfiguredModuleInterpreter<
    'a,
    Cfg: CodegenConfig,
    R: InterpRootManager<Cfg::Layout, Cfg::Roots, Cfg::RootTransport> = NoGcRoots,
> {
    module: &'a Module,
    roots: &'a R,
    conts: RefCell<VecContinuationStore>,
    externs: Vec<Option<Box<dyn Fn(&[u64]) -> ExternCallResult + 'a>>>,
    indirect_handler: Option<Box<dyn Fn(u64, &[u64]) -> ExternCallResult + 'a>>,
    _config: PhantomData<Cfg>,
}

pub type ModuleInterpreter<'a, S, R = NoGcRoots> =
    ConfiguredModuleInterpreter<'a, DefaultCodegenConfig<S>, R>;

impl<'a, Cfg: CodegenConfig, R> ConfiguredModuleInterpreter<'a, Cfg, R>
where
    Cfg::Layout: LayoutConfigDefaults,
    R: InterpRootManager<Cfg::Layout, Cfg::Roots, Cfg::RootTransport>,
{
    pub fn new(module: &'a Module, roots: &'a R) -> Self {
        let externs = (0..module.func_table.len()).map(|_| None).collect();
        ConfiguredModuleInterpreter {
            module,
            roots,
            conts: RefCell::new(VecContinuationStore::new()),
            externs,
            indirect_handler: None,
            _config: PhantomData,
        }
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

    /// Run an entry function. Creates an `InterpStackRuntime` internally.
    pub fn run(&self, entry: FuncRef, args: &[u64]) -> Result<InterpResult, InterpError> {
        let mut sr = InterpStackRuntime::new(self.roots, &self.module.functions);
        self.run_with_runtime(&mut sr, entry, args)
    }

    /// Run an entry function with a provided `InterpFrameStore`.
    pub fn run_with_runtime(
        &self,
        sr: &mut impl InterpFrameStore,
        entry: FuncRef,
        args: &[u64],
    ) -> Result<InterpResult, InterpError> {
        let entry_idx = match &self.module.func_table[entry.index()] {
            FuncDef::Internal(idx) => *idx,
            FuncDef::Extern(_) => panic!("cannot run extern function as entry point"),
        };

        let func = &self.module.functions[entry_idx];
        let param_args: Vec<(usize, u64)> = func.blocks[0]
            .params
            .iter()
            .enumerate()
            .map(|(i, (v, _))| (v.index(), args[i]))
            .collect();
        sr.push_frame(
            entry_idx,
            func.value_types.len(),
            func.stack_slots.len(),
            &param_args,
            FrameResume::TopLevel,
        );
        self.run_loop(sr)
    }

    pub fn resume_snapshot(
        &self,
        snapshot: FrameSliceSnapshot,
        args: &[u64],
    ) -> Result<InterpResult, InterpError> {
        let mut sr = InterpStackRuntime::new(self.roots, &self.module.functions);
        sr.resume_snapshot(&snapshot, args);
        self.run_loop(&mut sr)
    }

    fn run_loop(&self, sr: &mut impl InterpFrameStore) -> Result<InterpResult, InterpError> {
        let mut conts = self.conts.borrow_mut();
        loop {
            let action = self.execute_frame(sr, &mut *conts)?;

            match action {
                FrameAction::InternalCall {
                    callee_idx,
                    args,
                    resume,
                } => {
                    let func = &self.module.functions[callee_idx];
                    let param_args: Vec<(usize, u64)> = func.blocks[0]
                        .params
                        .iter()
                        .enumerate()
                        .map(|(i, (v, _))| (v.index(), args[i]))
                        .collect();
                    sr.push_frame(
                        callee_idx,
                        func.value_types.len(),
                        func.stack_slots.len(),
                        &param_args,
                        resume,
                    );
                }

                FrameAction::Return(ret_val) => {
                    let resume = sr.pop_frame();

                    if sr.is_empty() {
                        return Ok(match ret_val {
                            Some(v) => InterpResult::Value(v),
                            None => InterpResult::Void,
                        });
                    }

                    match resume {
                        FrameResume::FromCall { return_dest } => {
                            if let (Some(dest), Some(val)) = (return_dest, ret_val) {
                                sr.set(dest, val);
                            }
                        }
                        FrameResume::FromInvoke {
                            normal_block,
                            normal_args_vals,
                            has_ret_param,
                            ..
                        } => {
                            let func = &self.module.functions[sr.func_idx()];
                            let target_block = &func.blocks[normal_block];
                            let mut param_idx = 0;
                            if has_ret_param {
                                if let Some(val) = ret_val {
                                    sr.set(target_block.params[0].0.index(), val);
                                }
                                param_idx = 1;
                            }
                            for (i, val) in normal_args_vals.iter().enumerate() {
                                sr.set(target_block.params[param_idx + i].0.index(), *val);
                            }
                            sr.set_block(normal_block);
                            sr.set_inst(0);
                        }
                        FrameResume::TopLevel => unreachable!(),
                    }
                }

                FrameAction::Exception(exc) => {
                    loop {
                        let resume = sr.pop_frame();

                        match resume {
                            FrameResume::TopLevel => {
                                return Err(InterpError::UncaughtException(exc));
                            }
                            FrameResume::FromInvoke {
                                exception_block,
                                exception_args_vals,
                                ..
                            } => {
                                let func = &self.module.functions[sr.func_idx()];
                                let target_block = &func.blocks[exception_block];
                                for (i, val) in exception_args_vals.iter().enumerate() {
                                    sr.set(target_block.params[i].0.index(), *val);
                                }
                                sr.set_block(exception_block);
                                sr.set_inst(0);
                                break;
                            }
                            FrameResume::FromCall { .. } => {
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
                    while !sr.is_empty() {
                        sr.pop_frame();
                    }
                    return Ok(InterpResult::Deopt {
                        deopt_id,
                        resume_point,
                        live_values,
                    });
                }

                FrameAction::AbortToPrompt { prompt, args } => {
                    let ret_val = args.first().copied();

                    let depth = sr
                        .find_prompt_depth(prompt.index_u32())
                        .expect("abort_to_prompt: no frame has the target prompt");
                    sr.pop_frames_above(depth);
                    sr.pop_prompt(prompt.index_u32());

                    let func = &self.module.functions[sr.func_idx()];
                    let handler = find_handler(func, prompt);
                    let hb = &func.blocks[handler.index()];
                    if let Some(val) = ret_val {
                        if let Some((param, _)) = hb.params.first() {
                            sr.set(param.index(), val);
                        }
                    }
                    sr.set_block(handler.index());
                    sr.set_inst(0);
                }
            }
        }
    }

    /// Execute the current frame until it needs to transfer control.
    fn execute_frame(&self, sr: &mut impl InterpFrameStore, conts: &mut impl ContinuationStore) -> Result<FrameAction, InterpError> {
        let func_idx = sr.func_idx();
        let func = &self.module.functions[func_idx];

        loop {
            let block_idx = sr.block_idx();
            let block = &func.blocks[block_idx];

            // Execute instructions
            while sr.inst_idx() < block.insts.len() {
                let inst_idx = sr.inst_idx();
                let node = &block.insts[inst_idx];

                match &node.inst {
                    Inst::PushPrompt(prompt, _handler) => {
                        sr.push_prompt(prompt.index_u32());
                        sr.advance_inst();
                        continue;
                    }

                    Inst::PopPrompt(prompt) => {
                        sr.pop_prompt(prompt.index_u32());
                        sr.advance_inst();
                        continue;
                    }

                    Inst::CaptureSlice(prompt, _live) => {
                        let dest = node.value.expect("capture_slice must produce a value");
                        sr.advance_inst();
                        let snapshot = sr.capture_snapshot(prompt.index_u32(), dest.index());
                        let handle = conts.store_snapshot(snapshot)?;
                        sr.set(dest.index(), handle);
                        continue;
                    }

                    Inst::CloneSlice(slice) => {
                        let dest = node.value.expect("clone_slice must produce a value");
                        let handle = sr.get(slice.index());
                        let new_handle = conts.clone_snapshot(handle)?;
                        sr.set(dest.index(), new_handle);
                        sr.advance_inst();
                        continue;
                    }

                    Inst::StackAddr(slot) => {
                        let dest = node.value.expect("stack_addr must produce a value");
                        let ptr = sr.slot_ptr(slot.index()) as u64;
                        sr.set(dest.index(), ptr);
                        sr.advance_inst();
                        continue;
                    }

                    Inst::Safepoint(live) => {
                        let live_indices: Vec<usize> = live.iter().map(|v| v.index()).collect();
                        sr.safepoint(&live_indices);
                        sr.advance_inst();
                        continue;
                    }

                    Inst::Call(fref, call_args) => {
                        match &self.module.func_table[fref.index()] {
                            FuncDef::Internal(callee_idx) => {
                                let callee_idx = *callee_idx;
                                let arg_vals: Vec<u64> =
                                    call_args.iter().map(|v| sr.get(v.index())).collect();
                                let return_dest = node.value.map(|v| v.index());
                                sr.advance_inst();
                                return Ok(FrameAction::InternalCall {
                                    callee_idx,
                                    args: arg_vals,
                                    resume: FrameResume::FromCall { return_dest },
                                });
                            }
                            FuncDef::Extern(_) => {
                                let arg_vals: Vec<u64> =
                                    call_args.iter().map(|v| sr.get(v.index())).collect();
                                match self.call_extern(*fref, &arg_vals)? {
                                    ExternCallResult::Value(ret) => {
                                        if let (Some(dest), Some(val)) = (node.value, ret) {
                                            sr.set(dest.index(), val);
                                        }
                                    }
                                    ExternCallResult::Exception(exc) => {
                                        return Ok(FrameAction::Exception(exc));
                                    }
                                }
                            }
                        }
                        sr.advance_inst();
                        continue;
                    }

                    Inst::CallIndirect(callee, call_args, _ret_ty) => {
                        let callee_val = sr.get(callee.index());
                        let arg_vals: Vec<u64> =
                            call_args.iter().map(|v| sr.get(v.index())).collect();

                        // Try the callee value as a func_table index.
                        // This lets language runtimes store function indices in
                        // closures and call them via CallIndirect, re-entering
                        // the interpreter just like Call does.
                        let idx = callee_val as usize;
                        if idx < self.module.func_table.len() {
                            match &self.module.func_table[idx] {
                                FuncDef::Internal(callee_idx) => {
                                    let return_dest = node.value.map(|v| v.index());
                                    sr.advance_inst();
                                    return Ok(FrameAction::InternalCall {
                                        callee_idx: *callee_idx,
                                        args: arg_vals,
                                        resume: FrameResume::FromCall { return_dest },
                                    });
                                }
                                FuncDef::Extern(_) => {
                                    match self.call_extern(FuncRef(idx as u32), &arg_vals)? {
                                        ExternCallResult::Value(ret) => {
                                            if let (Some(dest), Some(val)) = (node.value, ret) {
                                                sr.set(dest.index(), val);
                                            }
                                        }
                                        ExternCallResult::Exception(exc) => {
                                            return Ok(FrameAction::Exception(exc));
                                        }
                                    }
                                }
                            }
                        } else if let Some(handler) = self.indirect_handler.as_ref() {
                            // Fallback: opaque indirect handler (e.g., for host FFI).
                            match handler(callee_val, &arg_vals) {
                                ExternCallResult::Value(ret) => {
                                    if let (Some(dest), Some(val)) = (node.value, ret) {
                                        sr.set(dest.index(), val);
                                    }
                                }
                                ExternCallResult::Exception(exc) => {
                                    return Ok(FrameAction::Exception(exc));
                                }
                            }
                        } else {
                            return Err(InterpError::UnknownExternFunc(
                                format!("call_indirect: callee {} is not a valid func_table index and no indirect handler is bound", callee_val)
                            ));
                        }
                        sr.advance_inst();
                        continue;
                    }

                    other => {
                        let result = exec_non_call_inst::<Cfg::Layout>(
                            other,
                            |v| sr.get(v.index()),
                            |v| func.value_type(v),
                            &func.deopt_info,
                        )?;
                        if let Some(r) = result {
                            match r {
                                InstResult::Val(v) => {
                                    if let Some(dest) = node.value {
                                        sr.set(dest.index(), v);
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
                        sr.advance_inst();
                        continue;
                    }
                }
            }

            // Execute terminator
            match &block.terminator {
                Terminator::Ret(v) => {
                    return Ok(FrameAction::Return(Some(sr.get(v.index()))));
                }
                Terminator::RetVoid => {
                    return Ok(FrameAction::Return(None));
                }
                Terminator::Jump(target, args) => {
                    transfer_args(sr, *target, args, func);
                    sr.set_block(target.index());
                    sr.set_inst(0);
                }
                Terminator::BrIf {
                    cond,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                } => {
                    if sr.get(cond.index()) != 0 {
                        transfer_args(sr, *then_block, then_args, func);
                        sr.set_block(then_block.index());
                    } else {
                        transfer_args(sr, *else_block, else_args, func);
                        sr.set_block(else_block.index());
                    }
                    sr.set_inst(0);
                }
                Terminator::Switch {
                    val,
                    cases,
                    default_block,
                    default_args,
                } => {
                    let v = sr.get(val.index()) as i64;
                    let mut matched = false;
                    for (case_val, target, case_args) in cases {
                        if v == *case_val {
                            transfer_args(sr, *target, case_args, func);
                            sr.set_block(target.index());
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        transfer_args(sr, *default_block, default_args, func);
                        sr.set_block(default_block.index());
                    }
                    sr.set_inst(0);
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
                        let arg_vals: Vec<u64> =
                            call_args.iter().map(|v| sr.get(v.index())).collect();
                        let callee_sig = &self.module.functions[callee_idx].sig;
                        let normal_args_vals: Vec<u64> =
                            normal_args.iter().map(|v| sr.get(v.index())).collect();
                        let exception_args_vals: Vec<u64> = exception_args
                            .iter()
                            .map(|v| sr.get(v.index()))
                            .collect();
                        return Ok(FrameAction::InternalCall {
                            callee_idx,
                            args: arg_vals,
                            resume: FrameResume::FromInvoke {
                                normal_block: normal.index(),
                                normal_args_vals,
                                exception_block: exception.index(),
                                exception_args_vals,
                                has_ret_param: callee_sig.ret.is_some(),
                            },
                        });
                    }
                    FuncDef::Extern(_) => {
                        let arg_vals: Vec<u64> =
                            call_args.iter().map(|v| sr.get(v.index())).collect();
                        match self.call_extern(*fref, &arg_vals)? {
                            ExternCallResult::Value(ret) => {
                                let target_block = &func.blocks[normal.index()];
                                let mut param_idx = 0;
                                if let Some(ret_val) = ret {
                                    if !target_block.params.is_empty() {
                                        sr.set(target_block.params[0].0.index(), ret_val);
                                        param_idx = 1;
                                    }
                                }
                                let extra: Vec<u64> =
                                    normal_args.iter().map(|v| sr.get(v.index())).collect();
                                for (i, val) in extra.iter().enumerate() {
                                    sr.set(
                                        target_block.params[param_idx + i].0.index(),
                                        *val,
                                    );
                                }
                                sr.set_block(normal.index());
                                sr.set_inst(0);
                            }
                            ExternCallResult::Exception(_exc) => {
                                transfer_args(sr, *exception, exception_args, func);
                                sr.set_block(exception.index());
                                sr.set_inst(0);
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
                    let callee_val = sr.get(callee.index());
                    let arg_vals: Vec<u64> =
                        call_args.iter().map(|v| sr.get(v.index())).collect();
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
                                    sr.set(target_block.params[0].0.index(), ret_val);
                                    param_idx = 1;
                                }
                            }
                            let extra: Vec<u64> =
                                normal_args.iter().map(|v| sr.get(v.index())).collect();
                            for (i, val) in extra.iter().enumerate() {
                                sr.set(
                                    target_block.params[param_idx + i].0.index(),
                                    *val,
                                );
                            }
                            sr.set_block(normal.index());
                            sr.set_inst(0);
                        }
                        ExternCallResult::Exception(_exc) => {
                            transfer_args(sr, *exception, exception_args, func);
                            sr.set_block(exception.index());
                            sr.set_inst(0);
                        }
                    }
                }

                Terminator::ResumeSlice { slice, args } => {
                    let slice_bits = sr.get(slice.index());
                    let arg_vals: Vec<u64> = args.iter().map(|v| sr.get(v.index())).collect();
                    let snapshot = conts.get_snapshot(slice_bits)?;
                    sr.resume_snapshot(snapshot, &arg_vals);
                    // After resume, the stack has been replaced; continue executing
                    return Ok(self.execute_frame(sr, conts)?);
                }

                Terminator::AbortToPrompt { prompt, args } => {
                    let arg_vals = args.iter().map(|v| sr.get(v.index())).collect();
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
    get_val: impl Fn(&Value) -> u64,
    val_type: impl Fn(Value) -> Type,
    deopt_info: &[DeoptInfo],
) -> Result<Option<InstResult>, InterpError> {
    let v = get_val;
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
fn build_gc_slot_map(func: &Function, precision: RootPrecision) -> Vec<Option<usize>> {
    if precision == RootPrecision::ConservativeWords {
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

// ─── Snapshot helpers ─────────────────────────────────────────────


// ─── Helpers ──────────────────────────────────────────────────────

/// Find the handler block for a given prompt in a function.
fn find_handler(func: &Function, prompt: PromptId) -> BlockId {
    for blk in &func.blocks {
        for node in &blk.insts {
            if let Inst::PushPrompt(p, h) = &node.inst {
                if *p == prompt {
                    return *h;
                }
            }
        }
    }
    panic!(
        "abort_to_prompt: could not find PushPrompt instruction for prompt in owning frame"
    );
}

/// Transfer block arguments via an InterpFrameStore.
fn transfer_args(sr: &mut impl InterpFrameStore, target: BlockId, args: &[Value], func: &Function) {
    let arg_vals: Vec<u64> = args.iter().map(|v| sr.get(v.index())).collect();
    let target_block = &func.blocks[target.index()];
    for (i, val) in arg_vals.iter().enumerate() {
        sr.set(target_block.params[i].0.index(), *val);
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
