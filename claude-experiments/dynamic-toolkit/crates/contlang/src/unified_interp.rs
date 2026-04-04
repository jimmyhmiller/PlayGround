//! Thin interpreter parameterized by UnifiedStackStrategy.
//!
//! The interpreter is just instruction dispatch. All stack management,
//! frame storage, and continuation handling is delegated to the StackRuntime.

use dynexec::{FrameResume, StackRuntime, UnifiedStackStrategy, StackConfig};
use dynir::ir::*;

pub fn interpret<S: UnifiedStackStrategy>(
    module: &Module,
    rt: &mut S::Runtime,
    entry: FuncRef,
    args: &[u64],
) -> u64 {
    let callee_idx = match module.func_table[entry.index()] {
        FuncDef::Internal(idx) => idx,
        FuncDef::Extern(_) => panic!("cannot call extern as entry"),
    };

    let func = &module.functions[callee_idx];
    let val_count = func.value_types.len();
    let param_args: Vec<(usize, u64)> = func.blocks[0].params.iter().enumerate()
        .map(|(i, (v, _))| (v.index(), args[i]))
        .collect();
    rt.push_frame(callee_idx, val_count, &param_args, FrameResume::TopLevel);

    'dispatch: loop {
        let func = &module.functions[rt.func_idx()];
        let block = &func.blocks[rt.block_idx()];

        // Execute instructions
        while rt.inst_idx() < block.insts.len() {
            let node = &block.insts[rt.inst_idx()];
            match &node.inst {
                Inst::PushPrompt(p, _) => {
                    rt.push_prompt(p.index_u32());
                }
                Inst::PopPrompt(p) => {
                    rt.pop_prompt(p.index_u32());
                }
                Inst::CaptureSlice(prompt, _) => {
                    let dest = node.value.unwrap();
                    rt.advance_inst();
                    let handle = rt.capture(prompt.index_u32(), dest.index());
                    rt.set(dest.index(), handle);
                    continue; // already advanced
                }
                Inst::CloneSlice(v) => {
                    let handle = rt.get(v.index());
                    let new_handle = rt.clone_continuation(handle);
                    if let Some(d) = node.value {
                        rt.set(d.index(), new_handle);
                    }
                }
                Inst::Iconst(_, n) => {
                    if let Some(d) = node.value { rt.set(d.index(), *n as u64); }
                }
                Inst::Add(a, b) => {
                    let r = (rt.get(a.index()) as i64).wrapping_add(rt.get(b.index()) as i64);
                    if let Some(d) = node.value { rt.set(d.index(), r as u64); }
                }
                Inst::Sub(a, b) => {
                    let r = (rt.get(a.index()) as i64).wrapping_sub(rt.get(b.index()) as i64);
                    if let Some(d) = node.value { rt.set(d.index(), r as u64); }
                }
                Inst::Mul(a, b) => {
                    let r = (rt.get(a.index()) as i64).wrapping_mul(rt.get(b.index()) as i64);
                    if let Some(d) = node.value { rt.set(d.index(), r as u64); }
                }
                Inst::SDiv(a, b) => {
                    let bv = rt.get(b.index()) as i64;
                    let r = if bv == 0 { 0 } else { (rt.get(a.index()) as i64).wrapping_div(bv) };
                    if let Some(d) = node.value { rt.set(d.index(), r as u64); }
                }
                Inst::Icmp(op, a, b) => {
                    let av = rt.get(a.index()) as i64;
                    let bv = rt.get(b.index()) as i64;
                    let r: bool = match op {
                        CmpOp::Eq => av == bv, CmpOp::Ne => av != bv,
                        CmpOp::Slt => av < bv, CmpOp::Sle => av <= bv,
                        CmpOp::Sgt => av > bv, CmpOp::Sge => av >= bv,
                        _ => panic!("unsupported cmp"),
                    };
                    if let Some(d) = node.value { rt.set(d.index(), r as u64); }
                }
                Inst::Trunc(v, _) => {
                    let val = rt.get(v.index()) & 0xFF;
                    if let Some(d) = node.value { rt.set(d.index(), val); }
                }
                Inst::Zext(v, _) => {
                    let val = rt.get(v.index());
                    if let Some(d) = node.value { rt.set(d.index(), val); }
                }
                Inst::Call(fref, call_args) => {
                    let callee_idx = match module.func_table[fref.index()] {
                        FuncDef::Internal(idx) => idx,
                        FuncDef::Extern(_) => panic!("extern not supported"),
                    };
                    let arg_vals: Vec<u64> = call_args.iter().map(|v| rt.get(v.index())).collect();
                    let dest = node.value.map(|v| v.index());
                    rt.advance_inst();
                    let callee_func = &module.functions[callee_idx];
                    let param_args: Vec<(usize, u64)> = callee_func.blocks[0].params.iter()
                        .enumerate()
                        .map(|(i, (v, _))| (v.index(), arg_vals[i]))
                        .collect();
                    rt.push_frame(
                        callee_idx,
                        callee_func.value_types.len(),
                        &param_args,
                        FrameResume::FromCall { return_dest: dest },
                    );
                    continue 'dispatch; // new frame — re-read func/block from top
                }
                _ => {} // skip unsupported
            }
            rt.advance_inst();
        }

        // Execute terminator
        let block = &func.blocks[rt.block_idx()];
        match &block.terminator {
            Terminator::Ret(v) => {
                let val = rt.get(v.index());
                let resume = rt.pop_frame();
                if rt.is_empty() {
                    return val;
                }
                if let FrameResume::FromCall { return_dest: Some(dest) } = resume {
                    rt.set(dest, val);
                }
            }
            Terminator::RetVoid => {
                let resume = rt.pop_frame();
                if rt.is_empty() {
                    return 0;
                }
                let _ = resume;
            }
            Terminator::Jump(target, args) => {
                let vals: Vec<u64> = args.iter().map(|a| rt.get(a.index())).collect();
                let tb = &func.blocks[target.index()];
                rt.set_block(target.index());
                rt.set_inst(0);
                for (i, v) in vals.into_iter().enumerate() {
                    let (p, _) = tb.params[i];
                    rt.set(p.index(), v);
                }
            }
            Terminator::BrIf { cond, then_block, then_args, else_block, else_args } => {
                let c = rt.get(cond.index());
                let (tgt, args) = if c != 0 {
                    (then_block, then_args)
                } else {
                    (else_block, else_args)
                };
                let vals: Vec<u64> = args.iter().map(|a| rt.get(a.index())).collect();
                let tb = &func.blocks[tgt.index()];
                rt.set_block(tgt.index());
                rt.set_inst(0);
                for (i, v) in vals.into_iter().enumerate() {
                    let (p, _) = tb.params[i];
                    rt.set(p.index(), v);
                }
            }
            Terminator::ResumeSlice { slice, args } => {
                let handle = rt.get(slice.index());
                let arg_vals: Vec<u64> = args.iter().map(|v| rt.get(v.index())).collect();
                rt.resume(handle, &arg_vals);
                // Stack replaced — loop continues from new top frame
            }
            Terminator::AbortToPrompt { prompt, args } => {
                let arg_vals: Vec<u64> = args.iter().map(|v| rt.get(v.index())).collect();
                let ret_val = arg_vals.first().copied();

                let depth = rt.find_prompt_depth(prompt.index_u32())
                    .expect("abort: prompt not found");
                rt.pop_frames_above(depth);

                // Find handler block from PushPrompt in the current function
                let func = &module.functions[rt.func_idx()];
                let handler = find_handler(func, *prompt);

                rt.pop_prompt(prompt.index_u32());

                let hb = &func.blocks[handler.index()];
                if let Some(val) = ret_val {
                    if let Some((param, _)) = hb.params.first() {
                        rt.set(param.index(), val);
                    }
                }
                rt.set_block(handler.index());
                rt.set_inst(0);
            }
            Terminator::Unreachable => panic!("hit unreachable"),
            other => panic!("unsupported terminator: {:?}", other),
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

/// Convenience: run with a specific strategy, creating the runtime automatically.
pub fn run<S: UnifiedStackStrategy>(
    module: &Module,
    entry: FuncRef,
    args: &[u64],
) -> u64 {
    let mut rt = S::create_runtime(StackConfig::default());
    interpret::<S>(module, &mut rt, entry, args)
}
