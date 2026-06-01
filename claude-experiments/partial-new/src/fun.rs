//! The classic Futamura demo, now with **residual function generation**.
//!
//! This client is an explicit **stack-machine VM** for a small first-order
//! language with functions and recursion. We specialize the VM against a fixed
//! object program, with the object program's input dynamic.
//!
//! Binding-time behavior:
//!   * The bytecode, the program counter, and the static call structure reduce
//!     away: the fetch/dispatch loop vanishes and ordinary calls **inline**.
//!   * A function recursing on a STATIC argument (`power(x, 5)`) unrolls
//!     completely. A `while` on a DYNAMIC condition residualizes a loop, with
//!     the loop variables generalized by the whistle (spanning the call stack).
//!   * Recursion whose *depth* is DYNAMIC (`fib(x)`) cannot be inlined. Such a
//!     call is cut into a **residual function call**: the callee is specialized
//!     once into its own residual function (keyed by the binding-time pattern of
//!     its arguments — polyvariant specialization), and the call site emits an
//!     `Op::Call`. The residual function may call itself, reproducing the
//!     recursion in the output.
//!
//! Remarkably this needs no engine changes: a residual call is a straight-line
//! op that yields a value, and "specialize the callee" is just another run of
//! `engine::specialize`. The whole multi-function orchestration lives here.

use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::collections::HashMap;

use crate::engine::{Client, Step};
use crate::imp::{BinOp, Cond, RExpr};
use crate::residual::{BlockId, Program, Terminator};

// ---------------------------------------------------------------------------
// Object-language AST (built directly by demos)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    /// local slot of the current function
    Var(usize),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// call function #fid with these argument expressions
    Call(usize, Vec<Expr>),
}

pub fn int(n: i64) -> Expr {
    Expr::Int(n)
}
pub fn var(slot: usize) -> Expr {
    Expr::Var(slot)
}
pub fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Bin(op, Box::new(a), Box::new(b))
}
pub fn call(fid: usize, args: Vec<Expr>) -> Expr {
    Expr::Call(fid, args)
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Set(usize, Expr),
    Print(Expr),
    Return(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
}

pub struct FunDef {
    pub name: &'static str,
    pub nslots: usize,
    pub slot_names: Vec<&'static str>,
    pub body: Vec<Stmt>,
}

// ---------------------------------------------------------------------------
// Stack bytecode
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Instr {
    PushConst(i64),
    Load(usize),
    Store(usize),
    Bin(BinOp),
    Jmp(usize),
    JmpIfZero(usize),
    Call(usize, usize),
    Ret,
    Print,
}

// ---------------------------------------------------------------------------
// Residual program: imperative ops over residual variables, plus functions.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Op {
    Assign { var: usize, expr: RExpr },
    Print(RExpr),
    /// `dst = call func(args)`
    Call {
        dst: usize,
        func: usize,
        args: Vec<RExpr>,
    },
    /// set this function's return value (followed by a Halt terminator)
    Return(RExpr),
}

pub struct FuncBody {
    pub params: Vec<usize>,
    pub body: Program<Op, Cond>,
}

pub struct ResidualProgram {
    pub funcs: Vec<FuncBody>,
    pub entry: usize,
}

impl ResidualProgram {
    pub fn block_count(&self) -> usize {
        self.funcs.iter().map(|f| f.body.blocks.len()).sum()
    }
    pub fn op_count(&self) -> usize {
        self.funcs.iter().map(|f| f.body.op_count()).sum()
    }
}

// ---------------------------------------------------------------------------
// Partially-static VM state
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Abs {
    Static(i64),
    Dynamic(RExpr),
}

impl Abs {
    fn into_rexpr(self) -> RExpr {
        match self {
            Abs::Static(k) => RExpr::Const(k),
            Abs::Dynamic(r) => r,
        }
    }
    fn to_rexpr(&self) -> RExpr {
        self.clone().into_rexpr()
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Frame {
    pc: usize,
    func: usize,
    locals: Vec<Abs>,
    ostack: Vec<Abs>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct State {
    frames: Vec<Frame>,
    /// Stack of pending dynamic-if joins: `(join_pc, modified_slots)`. Pushed on
    /// both arms of a dynamic branch; popped (and the modified slots materialized
    /// to merge the arms) when control reaches `join_pc`.
    pending_joins: Vec<(usize, Vec<usize>)>,
}

impl State {
    fn top(&self) -> &Frame {
        self.frames.last().expect("no active frame")
    }
    fn top_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("no active frame")
    }
    fn frame_funcs(&self) -> Vec<usize> {
        self.frames.iter().map(|f| f.func).collect()
    }
}

// ---------------------------------------------------------------------------
// Residual-function and residual-variable bookkeeping (interior mutability)
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash)]
enum SlotBt {
    Static(i64),
    Param,
}

#[derive(Default)]
struct FuncTable {
    /// (object function, argument binding-time pattern) -> residual function id
    sig_to_fid: HashMap<(usize, Vec<SlotBt>), usize>,
    entry_states: Vec<State>,
    params: Vec<Vec<usize>>,
    bodies: Vec<Option<Program<Op, Cond>>>,
    pending: VecDeque<usize>,
}

#[derive(Default)]
struct VarAlloc {
    names: Vec<String>,
    /// stable ids for generalized locals, keyed by (call-context, frame, slot)
    map: HashMap<(Vec<usize>, usize, usize), usize>,
}

const MAX_DEPTH: usize = 512;

pub struct Fun {
    code: Vec<Instr>,
    leaders: Vec<bool>,
    loop_head: Vec<bool>,
    /// per loop-header pc, the local slots assigned anywhere in the loop body.
    /// In a dynamically-controlled loop these are exactly the variables that
    /// must be dynamic at the header (a value we cannot track across an unknown
    /// iteration count), so we materialize them there as phi-like merges.
    loop_modified: Vec<Vec<usize>>,
    /// per `JmpIfZero` pc that heads a (non-loop) `if`, the join pc just past the
    /// `if` and the slots assigned in either arm.
    if_join: Vec<Option<(usize, Vec<usize>)>>,
    entries: Vec<usize>,
    nslots: Vec<usize>,
    slot_names: Vec<Vec<String>>,
    is_recursive: Vec<bool>,
    main: usize,
    alloc: RefCell<VarAlloc>,
    ftable: RefCell<FuncTable>,
}

impl Fun {
    pub fn new(program: &[FunDef]) -> Self {
        let mut code = Vec::new();
        let mut entries = vec![0usize; program.len()];
        let mut nslots = vec![0usize; program.len()];
        let mut ifs: Vec<(usize, usize)> = Vec::new(); // (branch pc, join pc)

        for (fid, f) in program.iter().enumerate() {
            entries[fid] = code.len();
            nslots[fid] = f.nslots;
            compile_stmts(&f.body, &mut code, &mut ifs);
            code.push(Instr::PushConst(0));
            code.push(Instr::Ret);
        }

        let mut leaders = vec![false; code.len()];
        let mut loop_head = vec![false; code.len()];
        let mut loop_modified: Vec<Vec<usize>> = vec![Vec::new(); code.len()];
        for (j, i) in code.iter().enumerate() {
            match i {
                Instr::Jmp(t) => {
                    leaders[*t] = true;
                    if *t <= j {
                        loop_head[*t] = true;
                        // Slots assigned anywhere in the loop body [header, back-edge].
                        let mut mods: Vec<usize> = code[*t..=j]
                            .iter()
                            .filter_map(|ins| match ins {
                                Instr::Store(slot) => Some(*slot),
                                _ => None,
                            })
                            .collect();
                        mods.sort_unstable();
                        mods.dedup();
                        loop_modified[*t] = mods;
                    }
                }
                Instr::JmpIfZero(t) => leaders[*t] = true,
                _ => {}
            }
        }

        let slot_names: Vec<Vec<String>> = program
            .iter()
            .map(|f| {
                (0..f.nslots)
                    .map(|s| {
                        f.slot_names
                            .get(s)
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| format!("v{s}"))
                    })
                    .collect()
            })
            .collect();

        let mut if_join: Vec<Option<(usize, Vec<usize>)>> = vec![None; code.len()];
        for (jz, end) in ifs {
            let mut mods: Vec<usize> = code[jz + 1..end]
                .iter()
                .filter_map(|ins| match ins {
                    Instr::Store(slot) => Some(*slot),
                    _ => None,
                })
                .collect();
            mods.sort_unstable();
            mods.dedup();
            if_join[jz] = Some((end, mods));
        }

        let is_recursive = compute_recursive(&entries, &code, program.len());

        let main = program
            .iter()
            .position(|f| f.name == "main")
            .expect("program needs a `main`");

        let me = Fun {
            code,
            leaders,
            loop_head,
            loop_modified,
            if_join,
            entries,
            nslots,
            slot_names,
            is_recursive,
            main,
            alloc: RefCell::new(VarAlloc::default()),
            ftable: RefCell::new(FuncTable::default()),
        };

        // Intern `main` as the residual entry function. Its single parameter is
        // the dynamic program input; remaining slots start as static 0.
        let mut entry_locals = vec![Abs::Static(0); me.nslots[me.main]];
        if !entry_locals.is_empty() {
            entry_locals[0] = Abs::Dynamic(RExpr::Const(0)); // marks slot 0 as a Param
        }
        let (fid, _) = me.intern(me.main, entry_locals);
        debug_assert_eq!(fid, 0, "main must be residual function 0");
        me
    }

    fn fresh_temp(&self) -> usize {
        let mut a = self.alloc.borrow_mut();
        let id = a.names.len();
        a.names.push(format!("t{id}"));
        id
    }

    fn fresh_named(&self, name: String) -> usize {
        let mut a = self.alloc.borrow_mut();
        let id = a.names.len();
        a.names.push(name);
        id
    }

    fn stable_var(&self, ctx: &[usize], frame_ix: usize, slot: usize, name: String) -> usize {
        let mut a = self.alloc.borrow_mut();
        let key = (ctx.to_vec(), frame_ix, slot);
        if let Some(&id) = a.map.get(&key) {
            return id;
        }
        let id = a.names.len();
        a.names.push(name);
        a.map.insert(key, id);
        id
    }

    fn slot_name(&self, func: usize, slot: usize) -> String {
        self.slot_names[func]
            .get(slot)
            .cloned()
            .unwrap_or_else(|| format!("v{slot}"))
    }

    /// Intern a residual function for object function `object_fid` entered with
    /// `entry_locals` (Static slots are baked in; Dynamic slots become params).
    /// Returns (residual fid, the residual argument expressions for Param slots).
    fn intern(&self, object_fid: usize, entry_locals: Vec<Abs>) -> (usize, Vec<RExpr>) {
        let mut sig = Vec::with_capacity(entry_locals.len());
        let mut arg_rexprs = Vec::new();
        for a in &entry_locals {
            match a {
                Abs::Static(k) => sig.push(SlotBt::Static(*k)),
                Abs::Dynamic(r) => {
                    sig.push(SlotBt::Param);
                    arg_rexprs.push(r.clone());
                }
            }
        }

        let key = (object_fid, sig);
        if let Some(&fid) = self.ftable.borrow().sig_to_fid.get(&key) {
            return (fid, arg_rexprs);
        }

        // Build a fresh residual function.
        let mut params = Vec::new();
        let mut locals = Vec::with_capacity(entry_locals.len());
        for (slot, a) in entry_locals.iter().enumerate() {
            match a {
                Abs::Static(k) => locals.push(Abs::Static(*k)),
                Abs::Dynamic(_) => {
                    let pv = self.fresh_named(self.slot_name(object_fid, slot));
                    params.push(pv);
                    locals.push(Abs::Dynamic(RExpr::Var(pv)));
                }
            }
        }
        let entry_state = State {
            frames: vec![Frame {
                pc: self.entries[object_fid],
                func: object_fid,
                locals,
                ostack: Vec::new(),
            }],
            pending_joins: Vec::new(),
        };

        let mut t = self.ftable.borrow_mut();
        let fid = t.bodies.len();
        t.sig_to_fid.insert(key, fid);
        t.entry_states.push(entry_state);
        t.params.push(params);
        t.bodies.push(None);
        t.pending.push_back(fid);
        (fid, arg_rexprs)
    }

    /// Drive specialization to completion, producing all residual functions.
    pub fn specialize_program(&self) -> ResidualProgram {
        loop {
            let fid = match self.ftable.borrow_mut().pending.pop_front() {
                Some(f) => f,
                None => break,
            };
            let entry = self.ftable.borrow().entry_states[fid].clone();
            let mut body = crate::engine::specialize(self, entry);
            crate::residual::simplify(&mut body);
            self.ftable.borrow_mut().bodies[fid] = Some(body);
        }

        let mut t = self.ftable.borrow_mut();
        let n = t.bodies.len();
        let mut funcs = Vec::with_capacity(n);
        for fid in 0..n {
            let body = t.bodies[fid].take().expect("function body not specialized");
            funcs.push(FuncBody {
                params: t.params[fid].clone(),
                body,
            });
        }
        ResidualProgram { funcs, entry: 0 }
    }

    /// Transfer control to `target`. If `target` is a dynamically-controlled
    /// loop header, materialize the loop's modified slots to stable residual
    /// variables first (emitting the phi assignments on this edge), so every
    /// path into the header arrives with the same binding-time signature and the
    /// header merges into a single block instead of peeling.
    fn jump_to(&self, target: usize, s: &State, out: &mut Vec<Op>) -> Step<Self> {
        let mut ns = s.clone();
        ns.top_mut().pc = target;

        // Loop header: materialize the loop's modified variables (phi).
        if self.loop_head[target] && self.dynamically_controlled(&ns) {
            let modified = self.loop_modified[target].clone();
            self.materialize(&mut ns, &modified, out);
        }

        // Dynamic-if join: materialize the if's modified variables to merge arms.
        while ns.pending_joins.last().map(|(end, _)| *end) == Some(target) {
            let (_, mods) = ns.pending_joins.pop().unwrap();
            self.materialize(&mut ns, &mods, out);
        }

        Step::Jump(ns)
    }

    /// Materialize the given top-frame slots to stable residual variables,
    /// emitting their current values on this edge. Idempotent: a slot already
    /// equal to its residual variable emits nothing.
    fn materialize(&self, ns: &mut State, slots: &[usize], out: &mut Vec<Op>) {
        let ctx = ns.frame_funcs();
        let fi = ns.frames.len() - 1;
        for &slot in slots {
            let cur = ns.frames[fi].locals[slot].clone();
            let name = self.slot_name(ns.frames[fi].func, slot);
            let id = self.stable_var(&ctx, fi, slot, name);
            if cur != Abs::Dynamic(RExpr::Var(id)) {
                out.push(Op::Assign {
                    var: id,
                    expr: cur.to_rexpr(),
                });
            }
            ns.frames[fi].locals[slot] = Abs::Dynamic(RExpr::Var(id));
        }
    }

    fn abs_bin(op: BinOp, a: Abs, b: Abs) -> Abs {
        match (a, b) {
            (Abs::Static(x), Abs::Static(y)) => Abs::Static(op.eval(x, y)),
            (x, y) => Abs::Dynamic(RExpr::Bin(
                op,
                Box::new(x.into_rexpr()),
                Box::new(y.into_rexpr()),
            )),
        }
    }

    fn dynamically_controlled(&self, s: &State) -> bool {
        if !self.loop_head[s.top().pc] {
            return false;
        }
        let mut f = s.top().clone();
        for _ in 0..256 {
            match &self.code[f.pc] {
                Instr::PushConst(k) => {
                    f.ostack.push(Abs::Static(*k));
                    f.pc += 1;
                }
                Instr::Load(slot) => {
                    f.ostack.push(f.locals[*slot].clone());
                    f.pc += 1;
                }
                Instr::Bin(op) => {
                    let b = f.ostack.pop().expect("dry-run bin");
                    let a = f.ostack.pop().expect("dry-run bin");
                    f.ostack.push(Fun::abs_bin(*op, a, b));
                    f.pc += 1;
                }
                Instr::JmpIfZero(_) => {
                    return matches!(f.ostack.last(), Some(Abs::Dynamic(_)));
                }
                _ => return true,
            }
        }
        true
    }
}

fn compute_recursive(entries: &[usize], code: &[Instr], nfuncs: usize) -> Vec<bool> {
    let mut calls: Vec<HashSet<usize>> = vec![HashSet::new(); nfuncs];
    for f in 0..nfuncs {
        let lo = entries[f];
        let hi = if f + 1 < nfuncs {
            entries[f + 1]
        } else {
            code.len()
        };
        for instr in &code[lo..hi] {
            if let Instr::Call(g, _) = instr {
                calls[f].insert(*g);
            }
        }
    }
    let mut is_rec = vec![false; nfuncs];
    for f in 0..nfuncs {
        let mut stack: Vec<usize> = calls[f].iter().copied().collect();
        let mut seen = HashSet::new();
        while let Some(g) = stack.pop() {
            if g == f {
                is_rec[f] = true;
                break;
            }
            if seen.insert(g) {
                stack.extend(calls[g].iter().copied());
            }
        }
    }
    is_rec
}

fn compile_stmts(stmts: &[Stmt], code: &mut Vec<Instr>, ifs: &mut Vec<(usize, usize)>) {
    for s in stmts {
        match s {
            Stmt::Set(slot, e) => {
                compile_expr(e, code);
                code.push(Instr::Store(*slot));
            }
            Stmt::Print(e) => {
                compile_expr(e, code);
                code.push(Instr::Print);
            }
            Stmt::Return(e) => {
                compile_expr(e, code);
                code.push(Instr::Ret);
            }
            Stmt::If(c, t, e) => {
                compile_expr(c, code);
                let jz = code.len();
                code.push(Instr::JmpIfZero(0));
                compile_stmts(t, code, ifs);
                let jmp = code.len();
                code.push(Instr::Jmp(0));
                let else_pc = code.len();
                compile_stmts(e, code, ifs);
                let end = code.len();
                patch(&mut code[jz], else_pc);
                patch(&mut code[jmp], end);
                ifs.push((jz, end));
            }
            Stmt::While(c, body) => {
                let head = code.len();
                compile_expr(c, code);
                let jz = code.len();
                code.push(Instr::JmpIfZero(0));
                compile_stmts(body, code, ifs);
                code.push(Instr::Jmp(head));
                let end = code.len();
                patch(&mut code[jz], end);
            }
        }
    }
}

fn compile_expr(e: &Expr, code: &mut Vec<Instr>) {
    match e {
        Expr::Int(k) => code.push(Instr::PushConst(*k)),
        Expr::Var(slot) => code.push(Instr::Load(*slot)),
        Expr::Bin(op, a, b) => {
            compile_expr(a, code);
            compile_expr(b, code);
            code.push(Instr::Bin(*op));
        }
        Expr::Call(fid, args) => {
            for a in args {
                compile_expr(a, code);
            }
            code.push(Instr::Call(*fid, args.len()));
        }
    }
}

fn patch(i: &mut Instr, target: usize) {
    match i {
        Instr::Jmp(t) | Instr::JmpIfZero(t) => *t = target,
        _ => unreachable!("patch on non-jump"),
    }
}

impl Client for Fun {
    type State = State;
    type Key = State;
    type Point = Vec<usize>;
    type Op = Op;
    type Cond = Cond;

    fn key(&self, s: &State) -> State {
        s.clone()
    }
    fn point(&self, s: &State) -> Vec<usize> {
        s.frames.iter().map(|f| f.pc).collect()
    }

    fn step(&self, s: &mut State, out: &mut Vec<Op>, at_entry: bool) -> Step<Self> {
        let pc = s.top().pc;

        if !at_entry && self.leaders[pc] {
            debug_assert!(
                s.top().ostack.is_empty(),
                "leader pc={pc} reached with non-empty operand stack"
            );
            return self.jump_to(pc, s, out);
        }

        match &self.code[pc] {
            Instr::PushConst(k) => {
                s.top_mut().ostack.push(Abs::Static(*k));
                s.top_mut().pc += 1;
                Step::Continue
            }
            Instr::Load(slot) => {
                let v = s.top().locals[*slot].clone();
                s.top_mut().ostack.push(v);
                s.top_mut().pc += 1;
                Step::Continue
            }
            Instr::Store(slot) => {
                let v = s.top_mut().ostack.pop().expect("store: empty stack");
                s.top_mut().locals[*slot] = v;
                s.top_mut().pc += 1;
                Step::Continue
            }
            Instr::Bin(op) => {
                let b = s.top_mut().ostack.pop().expect("bin: empty");
                let a = s.top_mut().ostack.pop().expect("bin: empty");
                s.top_mut().ostack.push(Fun::abs_bin(*op, a, b));
                s.top_mut().pc += 1;
                Step::Continue
            }
            Instr::Print => {
                let v = s.top_mut().ostack.pop().expect("print: empty");
                out.push(Op::Print(v.into_rexpr()));
                s.top_mut().pc += 1;
                Step::Continue
            }
            Instr::Jmp(t) => self.jump_to(*t, s, out),
            Instr::JmpIfZero(t) => {
                let cond = s.top_mut().ostack.pop().expect("jz: empty");
                match cond {
                    Abs::Static(k) => {
                        s.top_mut().pc = if k == 0 { *t } else { pc + 1 };
                        Step::Continue
                    }
                    Abs::Dynamic(r) => {
                        let mut tstate = s.clone();
                        tstate.top_mut().pc = *t;
                        let mut fstate = s.clone();
                        fstate.top_mut().pc = pc + 1;
                        // If this branch is a dynamic `if`, schedule the join: both
                        // arms must merge their differing assignments there.
                        if let Some((end, mods)) = &self.if_join[pc] {
                            tstate.pending_joins.push((*end, mods.clone()));
                            fstate.pending_joins.push((*end, mods.clone()));
                        }
                        Step::Branch {
                            cond: Cond::IsZero(r),
                            t: tstate,
                            f: fstate,
                        }
                    }
                }
            }
            Instr::Call(fid, nargs) => {
                let fid = *fid;
                let mut args = Vec::with_capacity(*nargs);
                for _ in 0..*nargs {
                    args.push(s.top_mut().ostack.pop().expect("call: missing arg"));
                }
                args.reverse();
                s.top_mut().pc = pc + 1; // resume after the call

                // Dynamic-depth recursion: a statically-recursive callee whose
                // arguments are ALL dynamic has no static quantity to bound it.
                // Cut it into a residual function call instead of inlining.
                let all_dynamic =
                    !args.is_empty() && args.iter().all(|a| matches!(a, Abs::Dynamic(_)));
                if self.is_recursive[fid] && all_dynamic {
                    let nsl = self.nslots[fid];
                    let mut entry_locals = args;
                    entry_locals.resize(nsl, Abs::Static(0));
                    let (res_fid, call_args) = self.intern(fid, entry_locals);
                    let dst = self.fresh_temp();
                    out.push(Op::Call {
                        dst,
                        func: res_fid,
                        args: call_args,
                    });
                    s.top_mut().ostack.push(Abs::Dynamic(RExpr::Var(dst)));
                    return Step::Continue;
                }

                // Otherwise inline: push a callee frame and keep specializing.
                if s.frames.len() >= MAX_DEPTH {
                    panic!(
                        "inlining call-stack depth exceeded; this is non-terminating \
                         static recursion (a recursive call with no static argument \
                         to bound it and no dynamic argument to cut it)."
                    );
                }
                let entry = self.entries[fid];
                let nsl = self.nslots[fid];
                let mut locals = vec![Abs::Static(0); nsl];
                for (i, a) in args.into_iter().enumerate() {
                    locals[i] = a;
                }
                s.frames.push(Frame {
                    pc: entry,
                    func: fid,
                    locals,
                    ostack: Vec::new(),
                });
                Step::Continue
            }
            Instr::Ret => {
                let v = s.top_mut().ostack.pop().expect("ret: empty");
                s.frames.pop();
                if s.frames.is_empty() {
                    // Returning from this residual function.
                    out.push(Op::Return(v.into_rexpr()));
                    Step::Halt
                } else {
                    s.top_mut().ostack.push(v);
                    Step::Continue
                }
            }
        }
    }

    fn whistle(&self, seen: &State, cand: &State) -> bool {
        seen != cand && self.dynamically_controlled(cand)
    }

    fn generalize(&self, seen: &State, from: &State, out: &mut Vec<Op>) -> State {
        let ctx = from.frame_funcs();
        let mut g = from.clone();
        for fi in 0..g.frames.len() {
            for slot in 0..g.frames[fi].locals.len() {
                if seen.frames[fi].locals[slot] != from.frames[fi].locals[slot] {
                    let name = self.slot_name(from.frames[fi].func, slot);
                    let id = self.stable_var(&ctx, fi, slot, name);
                    let rexpr = from.frames[fi].locals[slot].to_rexpr();
                    if rexpr != RExpr::Var(id) {
                        out.push(Op::Assign {
                            var: id,
                            expr: rexpr,
                        });
                    }
                    g.frames[fi].locals[slot] = Abs::Dynamic(RExpr::Var(id));
                }
            }
        }
        g
    }
}

// ---------------------------------------------------------------------------
// Pretty-printing and oracles
// ---------------------------------------------------------------------------

impl Fun {
    fn var_name(&self, id: usize) -> String {
        self.alloc.borrow().names[id].clone()
    }
    fn nvars(&self) -> usize {
        self.alloc.borrow().names.len()
    }

    fn fmt_rexpr(&self, e: &RExpr) -> String {
        match e {
            RExpr::Const(k) => k.to_string(),
            RExpr::Var(v) => self.var_name(*v),
            RExpr::Bin(op, a, b) => {
                format!("({} {} {})", self.fmt_rexpr(a), op.sym(), self.fmt_rexpr(b))
            }
        }
    }

    pub fn dump(&self, prog: &ResidualProgram) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        for (fid, f) in prog.funcs.iter().enumerate() {
            let params: Vec<String> = f.params.iter().map(|p| self.var_name(*p)).collect();
            let tag = if fid == prog.entry { " (entry)" } else { "" };
            writeln!(s, "fn f{fid}({}){tag}:", params.join(", ")).unwrap();
            for (i, b) in f.body.blocks.iter().enumerate() {
                let lead = if BlockId(i) == f.body.entry { " <-" } else { "" };
                writeln!(s, "  b{i}:{lead}").unwrap();
                for op in &b.ops {
                    match op {
                        Op::Assign { var, expr } => {
                            writeln!(s, "      {} := {}", self.var_name(*var), self.fmt_rexpr(expr))
                                .unwrap()
                        }
                        Op::Print(e) => writeln!(s, "      print {}", self.fmt_rexpr(e)).unwrap(),
                        Op::Call { dst, func, args } => {
                            let a: Vec<String> = args.iter().map(|e| self.fmt_rexpr(e)).collect();
                            writeln!(
                                s,
                                "      {} := call f{func}({})",
                                self.var_name(*dst),
                                a.join(", ")
                            )
                            .unwrap()
                        }
                        Op::Return(e) => writeln!(s, "      return {}", self.fmt_rexpr(e)).unwrap(),
                    }
                }
                match &b.term {
                    Terminator::Unset => writeln!(s, "      <unset>").unwrap(),
                    Terminator::Halt => writeln!(s, "      halt").unwrap(),
                    Terminator::Br(t) => writeln!(s, "      br b{}", t.0).unwrap(),
                    Terminator::Cond {
                        cond: Cond::IsZero(e),
                        t,
                        f,
                    } => writeln!(
                        s,
                        "      if {} == 0 -> b{} else b{}",
                        self.fmt_rexpr(e),
                        t.0,
                        f.0
                    )
                    .unwrap(),
                }
            }
        }
        s
    }

    /// Reference VM over concrete integers (the ground truth).
    pub fn run_reference(&self, input: i64) -> Vec<i64> {
        let mut locals = vec![0i64; self.nslots[self.main]];
        if !locals.is_empty() {
            locals[0] = input;
        }
        let mut frames: Vec<(usize, Vec<i64>, Vec<i64>)> =
            vec![(self.entries[self.main], locals, Vec::new())];
        let mut out = Vec::new();
        let mut budget = 100_000_000u64;
        loop {
            budget -= 1;
            if budget == 0 {
                panic!("reference VM exceeded step budget");
            }
            let pc = frames.last().unwrap().0;
            match &self.code[pc] {
                Instr::PushConst(k) => {
                    frames.last_mut().unwrap().2.push(*k);
                    frames.last_mut().unwrap().0 += 1;
                }
                Instr::Load(slot) => {
                    let v = frames.last().unwrap().1[*slot];
                    frames.last_mut().unwrap().2.push(v);
                    frames.last_mut().unwrap().0 += 1;
                }
                Instr::Store(slot) => {
                    let v = frames.last_mut().unwrap().2.pop().unwrap();
                    frames.last_mut().unwrap().1[*slot] = v;
                    frames.last_mut().unwrap().0 += 1;
                }
                Instr::Bin(op) => {
                    let f = frames.last_mut().unwrap();
                    let b = f.2.pop().unwrap();
                    let a = f.2.pop().unwrap();
                    f.2.push(op.eval(a, b));
                    f.0 += 1;
                }
                Instr::Print => {
                    let v = frames.last_mut().unwrap().2.pop().unwrap();
                    out.push(v);
                    frames.last_mut().unwrap().0 += 1;
                }
                Instr::Jmp(t) => frames.last_mut().unwrap().0 = *t,
                Instr::JmpIfZero(t) => {
                    let c = frames.last_mut().unwrap().2.pop().unwrap();
                    frames.last_mut().unwrap().0 = if c == 0 { *t } else { pc + 1 };
                }
                Instr::Call(fid, nargs) => {
                    let mut args = Vec::with_capacity(*nargs);
                    for _ in 0..*nargs {
                        args.push(frames.last_mut().unwrap().2.pop().unwrap());
                    }
                    args.reverse();
                    frames.last_mut().unwrap().0 = pc + 1;
                    let mut locals = vec![0i64; self.nslots[*fid]];
                    for (i, a) in args.into_iter().enumerate() {
                        locals[i] = a;
                    }
                    frames.push((self.entries[*fid], locals, Vec::new()));
                }
                Instr::Ret => {
                    let v = frames.last_mut().unwrap().2.pop().unwrap();
                    frames.pop();
                    if frames.is_empty() {
                        break;
                    }
                    frames.last_mut().unwrap().2.push(v);
                }
            }
        }
        out
    }

    /// Execute the residual program (functions, with residual calls).
    pub fn run_residual(&self, prog: &ResidualProgram, input: i64) -> Vec<i64> {
        let mut out = Vec::new();
        let nvars = self.nvars();
        self.run_func(prog, prog.entry, &[input], nvars, &mut out, 0);
        out
    }

    fn run_func(
        &self,
        prog: &ResidualProgram,
        fid: usize,
        args: &[i64],
        nvars: usize,
        out: &mut Vec<i64>,
        depth: usize,
    ) -> i64 {
        if depth > 1_000_000 {
            panic!("residual call depth exceeded (non-terminating residual recursion)");
        }
        let f = &prog.funcs[fid];
        let mut store = vec![0i64; nvars];
        for (i, p) in f.params.iter().enumerate() {
            store[*p] = args[i];
        }
        let mut retval = 0i64;
        let mut bid = f.body.entry;
        let mut budget = 100_000_000u64;
        loop {
            let block = &f.body.blocks[bid.0];
            for op in &block.ops {
                match op {
                    Op::Assign { var, expr } => store[*var] = eval_rexpr(expr, &store),
                    Op::Print(e) => out.push(eval_rexpr(e, &store)),
                    Op::Call { dst, func, args } => {
                        let avals: Vec<i64> = args.iter().map(|e| eval_rexpr(e, &store)).collect();
                        store[*dst] = self.run_func(prog, *func, &avals, nvars, out, depth + 1);
                    }
                    Op::Return(e) => retval = eval_rexpr(e, &store),
                }
            }
            match &block.term {
                Terminator::Halt => break,
                Terminator::Br(b) => bid = *b,
                Terminator::Cond {
                    cond: Cond::IsZero(e),
                    t,
                    f,
                } => bid = if eval_rexpr(e, &store) == 0 { *t } else { *f },
                Terminator::Unset => panic!("unset terminator in f{fid} b{}", bid.0),
            }
            budget -= 1;
            if budget == 0 {
                panic!("residual VM exceeded step budget");
            }
        }
        retval
    }
}

fn eval_rexpr(e: &RExpr, store: &[i64]) -> i64 {
    match e {
        RExpr::Const(k) => *k,
        RExpr::Var(v) => store[*v],
        RExpr::Bin(op, a, b) => op.eval(eval_rexpr(a, store), eval_rexpr(b, store)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fib_program() -> Vec<FunDef> {
        vec![
            FunDef {
                name: "fib",
                nslots: 1,
                slot_names: vec!["n"],
                body: vec![Stmt::If(
                    bin(BinOp::Lt, var(0), int(2)),
                    vec![Stmt::Return(var(0))],
                    vec![Stmt::Return(bin(
                        BinOp::Add,
                        call(0, vec![bin(BinOp::Sub, var(0), int(1))]),
                        call(0, vec![bin(BinOp::Sub, var(0), int(2))]),
                    ))],
                )],
            },
            FunDef {
                name: "main",
                nslots: 1,
                slot_names: vec!["x"],
                body: vec![Stmt::Print(call(0, vec![var(0)]))],
            },
        ]
    }

    #[test]
    fn dynamic_depth_recursion_becomes_residual_function() {
        let program = fib_program();
        let vm = Fun::new(&program);
        let prog = vm.specialize_program();
        // fib becomes a residual function distinct from main.
        assert!(prog.funcs.len() >= 2, "fib should be its own residual function");
        for x in 0..=15 {
            assert_eq!(
                vm.run_reference(x),
                vm.run_residual(&prog, x),
                "residual diverged from reference at fib({x})"
            );
        }
    }

    #[test]
    fn mutual_recursion_becomes_mutually_recursive_functions() {
        // is_even(n) = if n==0 then 1 else is_odd(n-1)
        // is_odd(n)  = if n==0 then 0 else is_even(n-1)
        // The cycle is detected, and each becomes a residual function that calls
        // the other.
        let program = vec![
            FunDef {
                name: "is_even",
                nslots: 1,
                slot_names: vec!["n"],
                body: vec![Stmt::If(
                    bin(BinOp::Eq, var(0), int(0)),
                    vec![Stmt::Return(int(1))],
                    vec![Stmt::Return(call(1, vec![bin(BinOp::Sub, var(0), int(1))]))],
                )],
            },
            FunDef {
                name: "is_odd",
                nslots: 1,
                slot_names: vec!["n"],
                body: vec![Stmt::If(
                    bin(BinOp::Eq, var(0), int(0)),
                    vec![Stmt::Return(int(0))],
                    vec![Stmt::Return(call(0, vec![bin(BinOp::Sub, var(0), int(1))]))],
                )],
            },
            FunDef {
                name: "main",
                nslots: 1,
                slot_names: vec!["x"],
                body: vec![Stmt::Print(call(0, vec![var(0)]))],
            },
        ];
        let vm = Fun::new(&program);
        let prog = vm.specialize_program();
        assert!(
            prog.funcs.len() >= 3,
            "main + is_even + is_odd should all be residual functions"
        );
        for x in 0..=12 {
            assert_eq!(
                vm.run_reference(x),
                vm.run_residual(&prog, x),
                "residual diverged from reference at is_even({x})"
            );
        }
    }

    #[test]
    fn dynamic_if_inside_dynamic_loop() {
        // main(x):
        //   s = 0; i = 0
        //   while i < x {           # dynamic loop (x dynamic, i loop-dynamic)
        //     if i < 3 { s = s + 1 } else { s = s + 10 }   # dynamic if, merges s
        //     i = i + 1
        //   }
        //   print s
        // Stresses the pending-join stack composing with loop-header merging.
        let program = vec![FunDef {
            name: "main",
            nslots: 3,
            slot_names: vec!["x", "s", "i"],
            body: vec![
                Stmt::Set(1, int(0)),
                Stmt::Set(2, int(0)),
                Stmt::While(
                    bin(BinOp::Lt, var(2), var(0)),
                    vec![
                        Stmt::If(
                            bin(BinOp::Lt, var(2), int(3)),
                            vec![Stmt::Set(1, bin(BinOp::Add, var(1), int(1)))],
                            vec![Stmt::Set(1, bin(BinOp::Add, var(1), int(10)))],
                        ),
                        Stmt::Set(2, bin(BinOp::Add, var(2), int(1))),
                    ],
                ),
                Stmt::Print(var(1)),
            ],
        }];
        let vm = Fun::new(&program);
        let prog = vm.specialize_program();
        for x in 0..=10 {
            assert_eq!(
                vm.run_reference(x),
                vm.run_residual(&prog, x),
                "residual diverged from reference at x={x}"
            );
        }
    }
}
