use crate::ir::*;
use crate::types::Type;
use std::fmt;

#[derive(Debug, Clone)]
pub enum VerifyError {
    UndefinedValue {
        value: Value,
        in_block: BlockId,
    },
    TypeMismatch {
        expected: Type,
        got: Type,
        context: String,
    },
    UnterminatedBlock {
        block: BlockId,
    },
    BranchArgCount {
        block: BlockId,
        target: BlockId,
        expected: usize,
        got: usize,
    },
    BranchArgType {
        block: BlockId,
        target: BlockId,
        arg: usize,
        expected: Type,
        got: Type,
    },
    ReturnTypeMismatch {
        expected: Option<Type>,
        got: Option<Type>,
    },
    DominanceViolation {
        value: Value,
        def_block: BlockId,
        use_block: BlockId,
    },
    EntryParamMismatch {
        expected: usize,
        got: usize,
    },
    InvalidPrompt {
        prompt: PromptId,
        context: String,
    },
    DuplicateValueDef {
        value: Value,
    },
    ContinuationsNotAllowed {
        context: String,
    },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyError::UndefinedValue { value, in_block } => {
                write!(f, "undefined value v{} used in bb{}", value.0, in_block.0)
            }
            VerifyError::TypeMismatch {
                expected,
                got,
                context,
            } => {
                write!(
                    f,
                    "type mismatch: expected {expected}, got {got} ({context})"
                )
            }
            VerifyError::UnterminatedBlock { block } => {
                write!(f, "block bb{} has no terminator", block.0)
            }
            VerifyError::BranchArgCount {
                block,
                target,
                expected,
                got,
            } => {
                write!(
                    f,
                    "bb{} -> bb{}: expected {expected} args, got {got}",
                    block.0, target.0
                )
            }
            VerifyError::BranchArgType {
                block,
                target,
                arg,
                expected,
                got,
            } => {
                write!(
                    f,
                    "bb{} -> bb{} arg {arg}: expected {expected}, got {got}",
                    block.0, target.0
                )
            }
            VerifyError::ReturnTypeMismatch { expected, got } => {
                write!(
                    f,
                    "return type mismatch: expected {expected:?}, got {got:?}"
                )
            }
            VerifyError::DominanceViolation {
                value,
                def_block,
                use_block,
            } => {
                write!(
                    f,
                    "v{} defined in bb{} does not dominate use in bb{}",
                    value.0, def_block.0, use_block.0
                )
            }
            VerifyError::EntryParamMismatch { expected, got } => {
                write!(f, "entry block params: expected {expected}, got {got}")
            }
            VerifyError::InvalidPrompt { prompt, context } => {
                write!(f, "invalid prompt prompt#{} ({context})", prompt.0)
            }
            VerifyError::DuplicateValueDef { value } => {
                write!(f, "value v{} defined more than once", value.0)
            }
            VerifyError::ContinuationsNotAllowed { context } => {
                write!(f, "continuation instruction not allowed: {context}")
            }
        }
    }
}

impl std::error::Error for VerifyError {}

/// Options controlling what the verifier accepts.
#[derive(Debug, Clone, Copy)]
pub struct VerifyOptions {
    /// Whether continuation instructions (PushPrompt, PopPrompt, CaptureSlice,
    /// CloneSlice, ResumeSlice, AbortToPrompt) are permitted.
    pub allow_continuations: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        VerifyOptions {
            allow_continuations: true,
        }
    }
}

/// Verify the structural integrity of a function.
pub fn verify(func: &Function) -> Result<(), Vec<VerifyError>> {
    verify_with(func, VerifyOptions::default())
}

/// Verify with explicit options.
pub fn verify_with(func: &Function, options: VerifyOptions) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();
    let n_blocks = func.blocks.len();

    // 1. Entry block params match function signature
    if func.blocks[0].params.len() != func.sig.params.len() {
        errors.push(VerifyError::EntryParamMismatch {
            expected: func.sig.params.len(),
            got: func.blocks[0].params.len(),
        });
    } else {
        for (i, ((_, pty), &sty)) in func.blocks[0]
            .params
            .iter()
            .zip(func.sig.params.iter())
            .enumerate()
        {
            if *pty != sty {
                errors.push(VerifyError::TypeMismatch {
                    expected: sty,
                    got: *pty,
                    context: format!("entry block param {i}"),
                });
            }
        }
    }

    // 2. Build value definition map: Value -> (BlockId, position)
    //    position: negative for block params, non-negative for instruction index
    let mut value_def_block = vec![None; func.value_types.len()];
    let mut value_def_pos = vec![0i32; func.value_types.len()];

    for (bi, block) in func.blocks.iter().enumerate() {
        let bid = BlockId(bi as u32);
        for (pi, (v, _)) in block.params.iter().enumerate() {
            if v.index() < value_def_block.len() {
                if value_def_block[v.index()].is_some() {
                    errors.push(VerifyError::DuplicateValueDef { value: *v });
                }
                value_def_block[v.index()] = Some(bid);
                value_def_pos[v.index()] = -(pi as i32) - 1; // negative = param
            }
        }
        for (ii, inst_node) in block.insts.iter().enumerate() {
            if let Some(v) = inst_node.value {
                if v.index() < value_def_block.len() {
                    if value_def_block[v.index()].is_some() {
                        errors.push(VerifyError::DuplicateValueDef { value: v });
                    }
                    value_def_block[v.index()] = Some(bid);
                    value_def_pos[v.index()] = ii as i32;
                }
            }
        }
    }

    // 3. Compute dominators (iterative algorithm)
    let doms = compute_dominators(func, n_blocks);

    // 4. Check each block
    for (bi, block) in func.blocks.iter().enumerate() {
        let bid = BlockId(bi as u32);

        // Check instructions
        for (ii, inst_node) in block.insts.iter().enumerate() {
            inst_node.inst.for_each_value(|v| {
                check_value_use(
                    v,
                    bid,
                    ii as i32,
                    &value_def_block,
                    &value_def_pos,
                    &doms,
                    &mut errors,
                );
            });

            // Check instruction-specific type constraints
            check_inst_types(func, &inst_node.inst, bid, options, &mut errors);
        }

        // Check terminator
        check_terminator(
            func,
            &block.terminator,
            bid,
            n_blocks,
            &value_def_block,
            &value_def_pos,
            &doms,
            options,
            &mut errors,
        );
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn check_value_use(
    v: Value,
    use_block: BlockId,
    use_pos: i32,
    def_block: &[Option<BlockId>],
    def_pos: &[i32],
    doms: &[u32],
    errors: &mut Vec<VerifyError>,
) {
    if v.index() >= def_block.len() {
        errors.push(VerifyError::UndefinedValue {
            value: v,
            in_block: use_block,
        });
        return;
    }
    let Some(db) = def_block[v.index()] else {
        errors.push(VerifyError::UndefinedValue {
            value: v,
            in_block: use_block,
        });
        return;
    };

    // Same block: def must come before use
    if db == use_block {
        let dp = def_pos[v.index()];
        // Block params (dp < 0) always dominate instructions (use_pos >= 0)
        if dp >= 0 && use_pos >= 0 && dp >= use_pos {
            errors.push(VerifyError::DominanceViolation {
                value: v,
                def_block: db,
                use_block,
            });
        }
        return;
    }

    // Different block: def block must dominate use block
    if !dominates(db, use_block, doms) {
        errors.push(VerifyError::DominanceViolation {
            value: v,
            def_block: db,
            use_block,
        });
    }
}

fn check_inst_types(func: &Function, inst: &Inst, _block: BlockId, options: VerifyOptions, errors: &mut Vec<VerifyError>) {
    let vt = |v: Value| func.value_type(v);
    match inst {
        Inst::Add(a, b)
        | Inst::Sub(a, b)
        | Inst::Mul(a, b)
        | Inst::SDiv(a, b)
        | Inst::UDiv(a, b)
        | Inst::And(a, b)
        | Inst::Or(a, b)
        | Inst::Xor(a, b)
        | Inst::Shl(a, b)
        | Inst::LShr(a, b)
        | Inst::AShr(a, b) => {
            let ta = vt(*a);
            let tb = vt(*b);
            // Allow pointer arithmetic: Ptr+I64, I64+Ptr
            let is_ptr_arith = (ta.is_ptr() && tb == Type::I64) || (ta == Type::I64 && tb.is_ptr());
            if !is_ptr_arith && ta != tb {
                errors.push(VerifyError::TypeMismatch {
                    expected: ta,
                    got: tb,
                    context: "integer binop operands".into(),
                });
            }
            if !is_ptr_arith && !ta.is_int() && !ta.is_ptr() {
                errors.push(VerifyError::TypeMismatch {
                    expected: Type::I64,
                    got: ta,
                    context: "integer binop requires int type".into(),
                });
            }
        }
        Inst::FAdd(a, b) | Inst::FSub(a, b) | Inst::FMul(a, b) | Inst::FDiv(a, b) => {
            if vt(*a) != Type::F64 || vt(*b) != Type::F64 {
                errors.push(VerifyError::TypeMismatch {
                    expected: Type::F64,
                    got: vt(*a),
                    context: "float binop requires f64".into(),
                });
            }
        }
        Inst::Icmp(_, a, b) => {
            if vt(*a) != vt(*b) || !vt(*a).is_int() {
                errors.push(VerifyError::TypeMismatch {
                    expected: vt(*a),
                    got: vt(*b),
                    context: "icmp requires matching int types".into(),
                });
            }
        }
        Inst::Select(c, t, e) => {
            if vt(*c) != Type::I8 {
                errors.push(VerifyError::TypeMismatch {
                    expected: Type::I8,
                    got: vt(*c),
                    context: "select cond".into(),
                });
            }
            if vt(*t) != vt(*e) {
                errors.push(VerifyError::TypeMismatch {
                    expected: vt(*t),
                    got: vt(*e),
                    context: "select arms".into(),
                });
            }
        }
        Inst::PushPrompt(prompt, _) | Inst::PopPrompt(prompt) | Inst::CaptureSlice(prompt, _) => {
            if !options.allow_continuations {
                errors.push(VerifyError::ContinuationsNotAllowed {
                    context: format!("{inst:?}"),
                });
            }
            if prompt.index() >= func.prompt_count as usize {
                errors.push(VerifyError::InvalidPrompt {
                    prompt: *prompt,
                    context: "instruction".into(),
                });
            }
        }
        Inst::CloneSlice(slice) => {
            if !options.allow_continuations {
                errors.push(VerifyError::ContinuationsNotAllowed {
                    context: "clone_slice".into(),
                });
            }
            if vt(*slice) != Type::FrameSlice {
                errors.push(VerifyError::TypeMismatch {
                    expected: Type::FrameSlice,
                    got: vt(*slice),
                    context: "clone_slice requires frameslice".into(),
                });
            }
        }
        _ => {} // Other instructions validated structurally
    }
}

fn check_terminator(
    func: &Function,
    term: &Terminator,
    block: BlockId,
    n_blocks: usize,
    def_block: &[Option<BlockId>],
    def_pos: &[i32],
    doms: &[u32],
    options: VerifyOptions,
    errors: &mut Vec<VerifyError>,
) {
    // Check value uses in terminator
    term.for_each_value(|v| {
        check_value_use(v, block, i32::MAX, def_block, def_pos, doms, errors);
    });

    // Check branch arg counts and types
    match term {
        Terminator::Ret(v) => {
            if let Some(ret_ty) = func.sig.ret {
                if func.value_type(*v) != ret_ty {
                    errors.push(VerifyError::ReturnTypeMismatch {
                        expected: Some(ret_ty),
                        got: Some(func.value_type(*v)),
                    });
                }
            } else {
                errors.push(VerifyError::ReturnTypeMismatch {
                    expected: None,
                    got: Some(func.value_type(*v)),
                });
            }
        }
        Terminator::RetVoid => {
            if func.sig.ret.is_some() {
                errors.push(VerifyError::ReturnTypeMismatch {
                    expected: func.sig.ret,
                    got: None,
                });
            }
        }
        Terminator::Jump(target, args) => {
            if target.index() < n_blocks {
                check_branch_args(func, block, *target, args, errors);
            }
        }
        Terminator::BrIf {
            then_block,
            then_args,
            else_block,
            else_args,
            ..
        } => {
            if then_block.index() < n_blocks {
                check_branch_args(func, block, *then_block, then_args, errors);
            }
            if else_block.index() < n_blocks {
                check_branch_args(func, block, *else_block, else_args, errors);
            }
        }
        Terminator::Switch {
            cases,
            default_block,
            default_args,
            ..
        } => {
            for (_, target, args) in cases {
                if target.index() < n_blocks {
                    check_branch_args(func, block, *target, args, errors);
                }
            }
            if default_block.index() < n_blocks {
                check_branch_args(func, block, *default_block, default_args, errors);
            }
        }
        Terminator::Invoke {
            func: fref,
            args,
            normal,
            normal_args,
            exception,
            exception_args,
        } => {
            let sig = &func.extern_funcs[fref.index()].sig;
            if sig.params.len() != args.len() {
                errors.push(VerifyError::TypeMismatch {
                    expected: Type::I64,
                    got: Type::I64,
                    context: format!(
                        "invoke arg count: expected {}, got {}",
                        sig.params.len(),
                        args.len()
                    ),
                });
            } else {
                for (i, (&param_ty, arg)) in sig.params.iter().zip(args.iter()).enumerate() {
                    let at = func.value_type(*arg);
                    if at != param_ty {
                        errors.push(VerifyError::TypeMismatch {
                            expected: param_ty,
                            got: at,
                            context: format!("invoke arg {i}"),
                        });
                    }
                }
            }
            if normal.index() < n_blocks {
                check_invoke_normal_args(func, block, sig.ret, *normal, normal_args, errors);
            }
            if exception.index() < n_blocks {
                check_branch_args(func, block, *exception, exception_args, errors);
            }
        }
        Terminator::InvokeIndirect {
            ret_ty,
            normal,
            normal_args,
            exception,
            exception_args,
            ..
        } => {
            if normal.index() < n_blocks {
                check_invoke_normal_args(func, block, *ret_ty, *normal, normal_args, errors);
            }
            if exception.index() < n_blocks {
                check_branch_args(func, block, *exception, exception_args, errors);
            }
        }
        Terminator::CaptureSlice { prompt, handler_block, resume_block } => {
            if !options.allow_continuations {
                errors.push(VerifyError::ContinuationsNotAllowed {
                    context: "capture_slice".into(),
                });
            }
            // handler_block expects one FrameSlice param; resume_block one I64
            let _ = prompt;
            if handler_block.index() < n_blocks {
                let hb = &func.blocks[handler_block.index()];
                if hb.params.len() != 1 {
                    errors.push(VerifyError::BranchArgCount {
                        block,
                        target: *handler_block,
                        expected: 0,
                        got: hb.params.len(),
                    });
                } else if hb.params[0].1 != Type::FrameSlice {
                    errors.push(VerifyError::TypeMismatch {
                        expected: Type::FrameSlice,
                        got: hb.params[0].1,
                        context: "capture_slice handler_block param".into(),
                    });
                }
            }
            if resume_block.index() < n_blocks {
                let rb = &func.blocks[resume_block.index()];
                if rb.params.len() != 1 {
                    errors.push(VerifyError::BranchArgCount {
                        block,
                        target: *resume_block,
                        expected: 0,
                        got: rb.params.len(),
                    });
                } else if rb.params[0].1 != Type::I64 {
                    errors.push(VerifyError::TypeMismatch {
                        expected: Type::I64,
                        got: rb.params[0].1,
                        context: "capture_slice resume_block param".into(),
                    });
                }
            }
        }
        Terminator::ResumeSlice { slice, return_block, return_args, .. } => {
            if !options.allow_continuations {
                errors.push(VerifyError::ContinuationsNotAllowed {
                    context: "resume_slice".into(),
                });
            }
            if func.value_type(*slice) != Type::FrameSlice {
                errors.push(VerifyError::TypeMismatch {
                    expected: Type::FrameSlice,
                    got: func.value_type(*slice),
                    context: "resume_slice requires frameslice".into(),
                });
            }
            // return_block's first param is the runtime-produced result
            // value of the resumed computation; `return_args` supplies any
            // additional params beyond that.
            if return_block.index() < n_blocks {
                let target = &func.blocks[return_block.index()];
                if target.params.is_empty() {
                    errors.push(VerifyError::BranchArgCount {
                        block,
                        target: *return_block,
                        expected: 1,
                        got: 0,
                    });
                } else {
                    let expected_extra = target.params.len() - 1;
                    if return_args.len() != expected_extra {
                        errors.push(VerifyError::BranchArgCount {
                            block,
                            target: *return_block,
                            expected: expected_extra,
                            got: return_args.len(),
                        });
                    }
                }
            }
        }
        Terminator::AbortToPrompt { prompt, .. } => {
            if !options.allow_continuations {
                errors.push(VerifyError::ContinuationsNotAllowed {
                    context: "abort_to_prompt".into(),
                });
            }
            if prompt.index() >= func.prompt_count as usize {
                errors.push(VerifyError::InvalidPrompt {
                    prompt: *prompt,
                    context: "abort_to_prompt".into(),
                });
            }
        }
        Terminator::Unreachable => {}
    }
}

fn check_branch_args(
    func: &Function,
    src: BlockId,
    target: BlockId,
    args: &[Value],
    errors: &mut Vec<VerifyError>,
) {
    let params = &func.blocks[target.index()].params;
    if params.len() != args.len() {
        errors.push(VerifyError::BranchArgCount {
            block: src,
            target,
            expected: params.len(),
            got: args.len(),
        });
        return;
    }
    for (i, ((_, pty), arg)) in params.iter().zip(args.iter()).enumerate() {
        let at = func.value_type(*arg);
        if at != *pty {
            errors.push(VerifyError::BranchArgType {
                block: src,
                target,
                arg: i,
                expected: *pty,
                got: at,
            });
        }
    }
}

/// Check branch args for invoke normal continuation.
/// The normal block's first param (if ret_ty is Some) receives the return value implicitly.
/// The remaining params must match normal_args.
fn check_invoke_normal_args(
    func: &Function,
    src: BlockId,
    ret_ty: Option<Type>,
    target: BlockId,
    args: &[Value],
    errors: &mut Vec<VerifyError>,
) {
    let params = &func.blocks[target.index()].params;
    let ret_param_count = if ret_ty.is_some() { 1 } else { 0 };
    if params.len() < ret_param_count {
        errors.push(VerifyError::BranchArgCount {
            block: src,
            target,
            expected: ret_param_count,
            got: params.len(),
        });
        return;
    }
    // Check return type matches first param
    if let Some(rty) = ret_ty {
        if params[0].1 != rty {
            errors.push(VerifyError::BranchArgType {
                block: src,
                target,
                arg: 0,
                expected: rty,
                got: params[0].1,
            });
        }
    }
    // Check remaining params match args
    let remaining_params = &params[ret_param_count..];
    if remaining_params.len() != args.len() {
        errors.push(VerifyError::BranchArgCount {
            block: src,
            target,
            expected: remaining_params.len() + ret_param_count,
            got: args.len() + ret_param_count,
        });
        return;
    }
    for (i, ((_, pty), arg)) in remaining_params.iter().zip(args.iter()).enumerate() {
        let at = func.value_type(*arg);
        if at != *pty {
            errors.push(VerifyError::BranchArgType {
                block: src,
                target,
                arg: i + ret_param_count,
                expected: *pty,
                got: at,
            });
        }
    }
}

// ── Dominator computation ──────────────────────────────────────

/// Compute immediate dominators using the iterative algorithm.
/// Returns a vec where `doms[i]` is the immediate dominator of block `i`.
/// Entry block (0) dominates itself.
fn compute_dominators(func: &Function, n: usize) -> Vec<u32> {
    if n == 0 {
        return vec![];
    }

    let preds = func.predecessors();

    // Compute reverse postorder
    let rpo = reverse_postorder(func, n);
    let mut rpo_num = vec![0u32; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_num[b as usize] = i as u32;
    }

    const UNDEFINED: u32 = u32::MAX;
    let mut doms = vec![UNDEFINED; n];
    doms[0] = 0; // entry dominates itself

    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo[1..] {
            // skip entry
            let b = b as usize;
            let mut new_idom = UNDEFINED;
            for &p in &preds[b] {
                let p = p.0;
                if doms[p as usize] == UNDEFINED {
                    continue;
                }
                if new_idom == UNDEFINED {
                    new_idom = p;
                } else {
                    new_idom = intersect(new_idom, p, &doms, &rpo_num);
                }
            }
            if new_idom != UNDEFINED && doms[b] != new_idom {
                doms[b] = new_idom;
                changed = true;
            }
        }
    }

    doms
}

fn intersect(mut a: u32, mut b: u32, doms: &[u32], rpo_num: &[u32]) -> u32 {
    while a != b {
        while rpo_num[a as usize] > rpo_num[b as usize] {
            a = doms[a as usize];
        }
        while rpo_num[b as usize] > rpo_num[a as usize] {
            b = doms[b as usize];
        }
    }
    a
}

fn dominates(a: BlockId, b: BlockId, doms: &[u32]) -> bool {
    let mut cur = b.0;
    loop {
        if cur == a.0 {
            return true;
        }
        let idom = doms[cur as usize];
        if idom == cur {
            return false; // reached entry without finding a
        }
        cur = idom;
    }
}

fn reverse_postorder(func: &Function, n: usize) -> Vec<u32> {
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    fn dfs(block: u32, func: &Function, visited: &mut Vec<bool>, order: &mut Vec<u32>) {
        if visited[block as usize] {
            return;
        }
        visited[block as usize] = true;
        for succ in func.blocks[block as usize].terminator.successors() {
            dfs(succ.0, func, visited, order);
        }
        order.push(block);
    }

    dfs(0, func, &mut visited, &mut order);
    order.reverse();
    order
}
