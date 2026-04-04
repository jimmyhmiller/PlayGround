//! Generic interpreter parameterized by StackBackend.
//!
//! This is a single interpreter loop that works with any backend:
//! Vec<u64> frames, GC-allocated segments, or anything else that
//! implements StackBackend.

use dynexec::{CallerResume, FrameMeta, InterpFrame, StackBackend};
use dynir::ir::*;

enum Action {
    Call { callee_idx: usize, args: Vec<u64>, resume: CallerResume },
    Return(Option<u64>),
    Capture { prompt: PromptId, return_dest: usize },
    DoResume { handle: u64, args: Vec<u64> },
    Abort { prompt: PromptId, args: Vec<u64> },
    Clone { source_handle: u64, dest_idx: usize },
}

pub struct GenericInterpreter<'a, B: StackBackend> {
    module: &'a Module,
    pub backend: B,
}

impl<'a, B: StackBackend> GenericInterpreter<'a, B> {
    pub fn new(module: &'a Module, backend: B) -> Self {
        GenericInterpreter { module, backend }
    }

    pub fn run(&mut self, entry: FuncRef, args: &[u64]) -> u64 {
        let callee_idx = match self.module.func_table[entry.index()] {
            FuncDef::Internal(idx) => idx,
            FuncDef::Extern(_) => panic!("cannot call extern as entry"),
        };

        let mut stack: Vec<InterpFrame<B>> = Vec::new();
        self.push_frame(&mut stack, callee_idx, args, CallerResume::TopLevel);

        loop {
            let action = self.exec(stack.last_mut().unwrap());
            match action {
                Action::Call { callee_idx, args, resume } => {
                    self.push_frame(&mut stack, callee_idx, &args, resume);
                }
                Action::Return(val) => {
                    let frame = stack.pop().unwrap();
                    if stack.is_empty() {
                        return val.unwrap_or(0);
                    }
                    let caller = stack.last_mut().unwrap();
                    if let CallerResume::FromCall { return_dest: Some(dest) } = frame.meta.resume {
                        if let Some(v) = val {
                            B::set(&mut caller.segment, dest, v);
                        }
                    }
                }
                Action::Capture { prompt, return_dest } => {
                    let start = stack.iter().rposition(|f| {
                        f.meta.active_prompts.contains(&prompt.index_u32())
                    }).expect("capture: prompt not found");

                    let cont = self.backend.capture(&stack, start, return_dest);
                    let handle = self.backend.store_cont(cont);
                    let frame = stack.last_mut().unwrap();
                    B::set(&mut frame.segment, return_dest, handle);
                }
                Action::DoResume { handle, args } => {
                    let cont = self.backend.load_cont(handle);
                    let restored = self.backend.restore(&cont, &args);
                    stack.clear();
                    stack.extend(restored);
                }
                Action::Clone { source_handle, dest_idx } => {
                    let cont = self.backend.load_cont(source_handle);
                    let cloned = self.backend.clone_cont(&cont);
                    let new_handle = self.backend.store_cont(cloned);
                    let frame = stack.last_mut().unwrap();
                    B::set(&mut frame.segment, dest_idx, new_handle);
                }
                Action::Abort { prompt, args } => {
                    let ret_val = args.first().copied();
                    let idx = stack.iter().rposition(|f| {
                        f.meta.active_prompts.contains(&prompt.index_u32())
                    }).expect("abort: prompt not found");
                    while stack.len() > idx + 1 { stack.pop(); }

                    let frame = stack.last_mut().unwrap();
                    let func = &self.module.functions[frame.meta.func_idx];
                    let handler = find_handler(func, prompt);

                    let pos = frame.meta.active_prompts.iter()
                        .rposition(|&p| p == prompt.index_u32()).unwrap();
                    frame.meta.active_prompts.remove(pos);

                    let hb = &func.blocks[handler.index()];
                    if let Some(val) = ret_val {
                        if let Some((param, _)) = hb.params.first() {
                            B::set(&mut frame.segment, param.index(), val);
                        }
                    }
                    frame.meta.block_idx = handler.index();
                    frame.meta.inst_idx = 0;
                }
            }
        }
    }

    fn push_frame(
        &mut self,
        stack: &mut Vec<InterpFrame<B>>,
        func_idx: usize,
        args: &[u64],
        resume: CallerResume,
    ) {
        // Safepoint before allocation if the backend requests it
        if self.backend.needs_safepoint() {
            self.backend.safepoint(stack);
        }

        let func = &self.module.functions[func_idx];
        let val_count = func.value_types.len();
        let mut segment = self.backend.alloc_segment(val_count);
        for (i, (v, _)) in func.blocks[0].params.iter().enumerate() {
            B::set(&mut segment, v.index(), args[i]);
        }
        stack.push(InterpFrame {
            meta: FrameMeta {
                func_idx,
                val_count,
                block_idx: 0,
                inst_idx: 0,
                resume,
                active_prompts: Vec::new(),
            },
            segment,
        });
    }

    fn exec(&mut self, frame: &mut InterpFrame<B>) -> Action {
        let func = &self.module.functions[frame.meta.func_idx];

        macro_rules! g { ($v:expr) => { B::get(&frame.segment, $v.index()) }; }
        macro_rules! s { ($d:expr, $v:expr) => { B::set(&mut frame.segment, $d.index(), $v) }; }

        loop {
            let block = &func.blocks[frame.meta.block_idx];

            while frame.meta.inst_idx < block.insts.len() {
                let node = &block.insts[frame.meta.inst_idx];
                match &node.inst {
                    Inst::PushPrompt(p, _) => {
                        frame.meta.active_prompts.push(p.index_u32());
                    }
                    Inst::PopPrompt(p) => {
                        let popped = frame.meta.active_prompts.pop();
                        assert_eq!(popped, Some(p.index_u32()));
                    }
                    Inst::CaptureSlice(prompt, _) => {
                        let dest = node.value.unwrap();
                        frame.meta.inst_idx += 1;
                        return Action::Capture { prompt: *prompt, return_dest: dest.index() };
                    }
                    Inst::CloneSlice(v) => {
                        let handle = g!(*v);
                        if let Some(d) = node.value {
                            frame.meta.inst_idx += 1;
                            return Action::Clone { source_handle: handle, dest_idx: d.index() };
                        }
                    }
                    Inst::Iconst(_, n) => {
                        if let Some(d) = node.value { s!(d, *n as u64); }
                    }
                    Inst::Add(a, b) => {
                        let r = (g!(*a) as i64).wrapping_add(g!(*b) as i64);
                        if let Some(d) = node.value { s!(d, r as u64); }
                    }
                    Inst::Sub(a, b) => {
                        let r = (g!(*a) as i64).wrapping_sub(g!(*b) as i64);
                        if let Some(d) = node.value { s!(d, r as u64); }
                    }
                    Inst::Mul(a, b) => {
                        let r = (g!(*a) as i64).wrapping_mul(g!(*b) as i64);
                        if let Some(d) = node.value { s!(d, r as u64); }
                    }
                    Inst::SDiv(a, b) => {
                        let bv = g!(*b) as i64;
                        let r = if bv == 0 { 0 } else { (g!(*a) as i64).wrapping_div(bv) };
                        if let Some(d) = node.value { s!(d, r as u64); }
                    }
                    Inst::Icmp(op, a, b) => {
                        let av = g!(*a) as i64;
                        let bv = g!(*b) as i64;
                        let r: bool = match op {
                            CmpOp::Eq => av == bv, CmpOp::Ne => av != bv,
                            CmpOp::Slt => av < bv, CmpOp::Sle => av <= bv,
                            CmpOp::Sgt => av > bv, CmpOp::Sge => av >= bv,
                            _ => panic!("unsupported cmp"),
                        };
                        if let Some(d) = node.value { s!(d, r as u64); }
                    }
                    Inst::Trunc(v, _) => {
                        let val = g!(*v) & 0xFF;
                        if let Some(d) = node.value { s!(d, val); }
                    }
                    Inst::Zext(v, _) => {
                        let val = g!(*v);
                        if let Some(d) = node.value { s!(d, val); }
                    }
                    Inst::Call(fref, call_args) => {
                        let callee = match self.module.func_table[fref.index()] {
                            FuncDef::Internal(idx) => idx,
                            FuncDef::Extern(_) => panic!("extern not supported"),
                        };
                        let args: Vec<u64> = call_args.iter().map(|v| g!(*v)).collect();
                        let dest = node.value.map(|v| v.index());
                        frame.meta.inst_idx += 1;
                        return Action::Call {
                            callee_idx: callee, args,
                            resume: CallerResume::FromCall { return_dest: dest },
                        };
                    }
                    _ => {} // skip unsupported
                }
                frame.meta.inst_idx += 1;
            }

            match &block.terminator {
                Terminator::Ret(v) => return Action::Return(Some(g!(*v))),
                Terminator::RetVoid => return Action::Return(None),
                Terminator::Jump(target, args) => {
                    let vals: Vec<u64> = args.iter().map(|a| g!(*a)).collect();
                    let tb = &func.blocks[target.index()];
                    for (i, v) in vals.into_iter().enumerate() {
                        let (p, _) = tb.params[i];
                        s!(p, v);
                    }
                    frame.meta.block_idx = target.index();
                    frame.meta.inst_idx = 0;
                }
                Terminator::BrIf { cond, then_block, then_args, else_block, else_args } => {
                    let c = g!(*cond);
                    let (tgt, args) = if c != 0 {
                        (then_block, then_args)
                    } else {
                        (else_block, else_args)
                    };
                    let vals: Vec<u64> = args.iter().map(|a| g!(*a)).collect();
                    let tb = &func.blocks[tgt.index()];
                    for (i, v) in vals.into_iter().enumerate() {
                        let (p, _) = tb.params[i];
                        s!(p, v);
                    }
                    frame.meta.block_idx = tgt.index();
                    frame.meta.inst_idx = 0;
                }
                Terminator::ResumeSlice { slice, args } => {
                    let handle = g!(*slice);
                    let av: Vec<u64> = args.iter().map(|v| g!(*v)).collect();
                    return Action::DoResume { handle, args: av };
                }
                Terminator::AbortToPrompt { prompt, args } => {
                    let av: Vec<u64> = args.iter().map(|v| g!(*v)).collect();
                    return Action::Abort { prompt: *prompt, args: av };
                }
                Terminator::Unreachable => panic!("hit unreachable"),
                other => panic!("unsupported terminator: {:?}", other),
            }
        }
    }
}

fn find_handler(func: &Function, prompt: PromptId) -> BlockId {
    for blk in &func.blocks {
        for node in &blk.insts {
            if let Inst::PushPrompt(p, h) = &node.inst {
                if *p == prompt { return *h; }
            }
        }
    }
    panic!("no PushPrompt for prompt");
}
