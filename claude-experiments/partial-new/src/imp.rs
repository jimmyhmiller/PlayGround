//! A tiny imperative while-language client for the generic engine.
//!
//! This is the second client, and its job is to prove the engine is genuinely
//! generic: it shares `engine.rs` and `residual.rs` with `bf.rs` unchanged. It
//! exercises everything BF did not:
//!
//!   * A real **partially-static store**: each variable is individually
//!     `Static(value)` or `Dynamic`. Static computation folds to constants;
//!     dynamic computation residualizes into expression trees.
//!   * **Static control flow** that picks branches and unrolls static loops to
//!     completion (a whole loop can vanish into a constant).
//!   * The **generalization whistle**: a loop with a *dynamic* exit condition
//!     would unroll forever (the loop counter keeps changing), so the engine
//!     generalizes the counter to dynamic, materializing its value on the edge.
//!     This is the machinery BF's finite key space never needed.

use std::fmt::Write as _;

use crate::engine::{Client, Step};
use crate::residual::{Program, Terminator};

// ---------------------------------------------------------------------------
// Source AST
// ---------------------------------------------------------------------------

pub type Var = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Lt,
    Eq,
}

impl BinOp {
    pub fn eval(self, a: i64, b: i64) -> i64 {
        match self {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Lt => (a < b) as i64,
            BinOp::Eq => (a == b) as i64,
        }
    }
    pub fn sym(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Lt => "<",
            BinOp::Eq => "==",
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    Var(Var),
    Bin(BinOp, Box<Expr>, Box<Expr>),
}

// Convenience constructors so demos read like programs.
pub fn int(n: i64) -> Expr {
    Expr::Int(n)
}
pub fn var(v: Var) -> Expr {
    Expr::Var(v)
}
pub fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Bin(op, Box::new(a), Box::new(b))
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Assign(Var, Expr),
    Print(Expr),
    Input(Var),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
}

// ---------------------------------------------------------------------------
// Flat bytecode (so the client mirrors BF's pc-driven shape)
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Instr {
    Assign(Var, Expr),
    Print(Expr),
    Input(Var),
    /// if eval(cond) == 0 goto target, else fall through
    JmpIfZero(Expr, usize),
    Jmp(usize),
    Halt,
}

fn compile(stmts: &[Stmt], code: &mut Vec<Instr>, ifs: &mut Vec<(usize, usize)>) {
    for s in stmts {
        match s {
            Stmt::Assign(v, e) => code.push(Instr::Assign(*v, e.clone())),
            Stmt::Print(e) => code.push(Instr::Print(e.clone())),
            Stmt::Input(v) => code.push(Instr::Input(*v)),
            Stmt::If(c, t, e) => {
                let jz = code.len();
                code.push(Instr::JmpIfZero(c.clone(), 0)); // patch -> else
                compile(t, code, ifs);
                let jmp = code.len();
                code.push(Instr::Jmp(0)); // patch -> end
                let else_pc = code.len();
                compile(e, code, ifs);
                let end_pc = code.len();
                patch_jz(&mut code[jz], else_pc);
                patch_jmp(&mut code[jmp], end_pc);
                ifs.push((jz, end_pc));
            }
            Stmt::While(c, body) => {
                let head = code.len();
                let jz = code.len();
                code.push(Instr::JmpIfZero(c.clone(), 0)); // patch -> end
                compile(body, code, ifs);
                code.push(Instr::Jmp(head));
                let end_pc = code.len();
                patch_jz(&mut code[jz], end_pc);
            }
        }
    }
}

fn patch_jz(i: &mut Instr, target: usize) {
    if let Instr::JmpIfZero(_, t) = i {
        *t = target;
    } else {
        unreachable!("patch_jz on non-JmpIfZero");
    }
}
fn patch_jmp(i: &mut Instr, target: usize) {
    if let Instr::Jmp(t) = i {
        *t = target;
    } else {
        unreachable!("patch_jmp on non-Jmp");
    }
}

// ---------------------------------------------------------------------------
// Residual program: imperative ops over named variables
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RExpr {
    Const(i64),
    Var(Var),
    Bin(BinOp, Box<RExpr>, Box<RExpr>),
}

#[derive(Debug)]
pub enum Op {
    Assign { var: Var, expr: RExpr },
    Print(RExpr),
    Input(Var),
}

#[derive(Debug)]
pub enum Cond {
    /// branch taken (the `t` edge) when this expression evaluates to 0
    IsZero(RExpr),
}

// ---------------------------------------------------------------------------
// Partially-static state
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum AbsVal {
    Static(i64),
    Dynamic,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct State {
    pc: usize,
    store: Vec<AbsVal>,
    /// Stack of pending dynamic-if joins `(join_pc, modified_vars)`, pushed on
    /// both arms of a dynamic branch and popped (merging the arms) at the join.
    pending_joins: Vec<(usize, Vec<Var>)>,
}

/// Abstract value of an expression: either fully known, or a residual tree.
enum Abs {
    S(i64),
    D(RExpr),
}

impl Abs {
    fn into_rexpr(self) -> RExpr {
        match self {
            Abs::S(k) => RExpr::Const(k),
            Abs::D(r) => r,
        }
    }
}

pub struct Imp {
    code: Vec<Instr>,
    names: Vec<String>,
    /// pcs that are jump targets, i.e. basic-block leaders.
    leaders: Vec<bool>,
    /// per loop-header pc (a `JmpIfZero` targeted by a backward `Jmp`), the
    /// variables assigned in the loop body. In a dynamically-controlled loop
    /// these must be dynamic at the header, so we materialize them there to
    /// merge every path into the header into one block (no peeling).
    loop_modified: Vec<Vec<Var>>,
    /// per `JmpIfZero` pc that heads a (non-loop) `if`, the join pc just past the
    /// `if` and the variables assigned in either arm.
    if_join: Vec<Option<(usize, Vec<Var>)>>,
}

impl Imp {
    pub fn new(names: &[&str], body: &[Stmt]) -> Self {
        let mut code = Vec::new();
        let mut ifs: Vec<(usize, usize)> = Vec::new();
        compile(body, &mut code, &mut ifs);
        code.push(Instr::Halt);
        let mut leaders = vec![false; code.len()];
        let mut loop_modified: Vec<Vec<Var>> = vec![Vec::new(); code.len()];
        let mut if_join: Vec<Option<(usize, Vec<Var>)>> = vec![None; code.len()];
        for (jz, end) in ifs {
            let mut mods: Vec<Var> = code[jz + 1..end]
                .iter()
                .filter_map(|ins| match ins {
                    Instr::Assign(v, _) | Instr::Input(v) => Some(*v),
                    _ => None,
                })
                .collect();
            mods.sort_unstable();
            mods.dedup();
            if_join[jz] = Some((end, mods));
        }
        for (j, i) in code.iter().enumerate() {
            match i {
                Instr::JmpIfZero(_, t) => leaders[*t] = true,
                Instr::Jmp(t) => {
                    leaders[*t] = true;
                    if *t <= j {
                        let mut mods: Vec<Var> = code[*t..=j]
                            .iter()
                            .filter_map(|ins| match ins {
                                Instr::Assign(v, _) | Instr::Input(v) => Some(*v),
                                _ => None,
                            })
                            .collect();
                        mods.sort_unstable();
                        mods.dedup();
                        loop_modified[*t] = mods;
                    }
                }
                _ => {}
            }
        }
        Imp {
            code,
            names: names.iter().map(|s| s.to_string()).collect(),
            leaders,
            loop_modified,
            if_join,
        }
    }

    /// Transfer to `target`. Materializes a dynamically-controlled loop header's
    /// modified variables (phi), and pops + materializes any dynamic-if join that
    /// ends at `target`, so all paths into a merge point arrive identically.
    fn jump_to(&self, target: usize, s: &State, out: &mut Vec<Op>) -> Step<Self> {
        let mut ns = s.clone();
        ns.pc = target;

        if !self.loop_modified[target].is_empty() && self.dynamically_controlled(&ns) {
            let modified = self.loop_modified[target].clone();
            self.materialize(&mut ns, &modified, out);
        }

        while ns.pending_joins.last().map(|(end, _)| *end) == Some(target) {
            let (_, mods) = ns.pending_joins.pop().unwrap();
            self.materialize(&mut ns, &mods, out);
        }

        Step::Jump(ns)
    }

    /// Materialize the given variables to dynamic, emitting their current static
    /// value on this edge (idempotent for already-dynamic variables).
    fn materialize(&self, ns: &mut State, vars: &[Var], out: &mut Vec<Op>) {
        for &v in vars {
            if let AbsVal::Static(k) = ns.store[v] {
                out.push(Op::Assign {
                    var: v,
                    expr: RExpr::Const(k),
                });
                ns.store[v] = AbsVal::Dynamic;
            }
        }
    }

    pub fn start(&self) -> State {
        // All variables begin life as the static value 0.
        State {
            pc: 0,
            store: vec![AbsVal::Static(0); self.names.len()],
            pending_joins: Vec::new(),
        }
    }

    fn eval(&self, e: &Expr, store: &[AbsVal]) -> Abs {
        match e {
            Expr::Int(k) => Abs::S(*k),
            Expr::Var(v) => match store[*v] {
                AbsVal::Static(k) => Abs::S(k),
                AbsVal::Dynamic => Abs::D(RExpr::Var(*v)),
            },
            Expr::Bin(op, a, b) => match (self.eval(a, store), self.eval(b, store)) {
                (Abs::S(x), Abs::S(y)) => Abs::S(op.eval(x, y)),
                (ra, rb) => Abs::D(RExpr::Bin(
                    *op,
                    Box::new(ra.into_rexpr()),
                    Box::new(rb.into_rexpr()),
                )),
            },
        }
    }

    /// Is the loop/branch at `state.pc` controlled by a *dynamic* condition?
    /// Only such recurrences need generalization; a static condition will
    /// eventually decide and let the loop unroll to completion.
    fn dynamically_controlled(&self, state: &State) -> bool {
        match &self.code[state.pc] {
            Instr::JmpIfZero(e, _) => matches!(self.eval(e, &state.store), Abs::D(_)),
            _ => false,
        }
    }
}

fn dyn_signature(store: &[AbsVal]) -> Vec<bool> {
    store.iter().map(|a| matches!(a, AbsVal::Dynamic)).collect()
}

impl Client for Imp {
    type State = State;
    type Key = State; // (pc, full partially-static store)
    type Point = usize; // pc alone
    type Op = Op;
    type Cond = Cond;

    fn key(&self, s: &State) -> State {
        s.clone()
    }
    fn point(&self, s: &State) -> usize {
        s.pc
    }

    fn step(&self, s: &mut State, out: &mut Vec<Op>, at_entry: bool) -> Step<Self> {
        // Force a block boundary when straight-line execution reaches a leader
        // (so back-edges land on a clean block start, never re-running code).
        if !at_entry && self.leaders[s.pc] {
            return self.jump_to(s.pc, s, out);
        }

        match &self.code[s.pc] {
            Instr::Assign(v, e) => {
                match self.eval(e, &s.store) {
                    Abs::S(k) => s.store[*v] = AbsVal::Static(k), // static: folded away
                    Abs::D(r) => {
                        out.push(Op::Assign { var: *v, expr: r });
                        s.store[*v] = AbsVal::Dynamic;
                    }
                }
                s.pc += 1;
                Step::Continue
            }
            Instr::Print(e) => {
                let r = self.eval(e, &s.store).into_rexpr();
                out.push(Op::Print(r)); // output is an effect: always residualized
                s.pc += 1;
                Step::Continue
            }
            Instr::Input(v) => {
                out.push(Op::Input(*v));
                s.store[*v] = AbsVal::Dynamic;
                s.pc += 1;
                Step::Continue
            }
            Instr::JmpIfZero(e, target) => match self.eval(e, &s.store) {
                Abs::S(k) => {
                    // Static condition: pick the branch now, no residual control.
                    s.pc = if k == 0 { *target } else { s.pc + 1 };
                    Step::Continue
                }
                Abs::D(r) => {
                    let mut t = s.clone();
                    t.pc = *target;
                    let mut f = s.clone();
                    f.pc = s.pc + 1;
                    if let Some((end, mods)) = &self.if_join[s.pc] {
                        t.pending_joins.push((*end, mods.clone()));
                        f.pending_joins.push((*end, mods.clone()));
                    }
                    Step::Branch {
                        cond: Cond::IsZero(r),
                        t,
                        f,
                    }
                }
            },
            Instr::Jmp(target) => self.jump_to(*target, s, out),
            Instr::Halt => Step::Halt,
        }
    }

    fn whistle(&self, seen: &State, cand: &State) -> bool {
        // Same point (guaranteed by grouping), genuinely different store, same
        // set of dynamic variables, and the recurrence is dynamically driven.
        seen.pc == cand.pc
            && seen.store != cand.store
            && dyn_signature(&seen.store) == dyn_signature(&cand.store)
            && self.dynamically_controlled(cand)
    }

    fn generalize(&self, seen: &State, from: &State, out: &mut Vec<Op>) -> State {
        // Make every variable that disagrees dynamic. If it was a static value
        // in `from`, materialize that value into the residual so the runtime
        // variable holds it when control reaches the generalized block.
        let mut store = from.store.clone();
        for v in 0..store.len() {
            if seen.store[v] != from.store[v] {
                if let AbsVal::Static(k) = from.store[v] {
                    out.push(Op::Assign {
                        var: v,
                        expr: RExpr::Const(k),
                    });
                }
                store[v] = AbsVal::Dynamic;
            }
        }
        State {
            pc: from.pc,
            store,
            pending_joins: from.pending_joins.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Oracles: a reference interpreter and a residual interpreter.
// ---------------------------------------------------------------------------

const STEP_BUDGET: u64 = 100_000_000;

impl Imp {
    /// Reference interpreter over concrete integer state (the ground truth).
    pub fn run_reference(&self, input: &[i64]) -> Vec<i64> {
        let mut store = vec![0i64; self.names.len()];
        let mut pc = 0usize;
        let mut out = Vec::new();
        let mut inp = input.iter().copied();
        let mut budget = STEP_BUDGET;
        loop {
            budget -= 1;
            if budget == 0 {
                panic!("reference interpreter exceeded step budget");
            }
            match &self.code[pc] {
                Instr::Assign(v, e) => {
                    store[*v] = eval_concrete(e, &store);
                    pc += 1;
                }
                Instr::Print(e) => {
                    out.push(eval_concrete(e, &store));
                    pc += 1;
                }
                Instr::Input(v) => {
                    store[*v] = inp.next().unwrap_or(0);
                    pc += 1;
                }
                Instr::JmpIfZero(e, t) => {
                    pc = if eval_concrete(e, &store) == 0 {
                        *t
                    } else {
                        pc + 1
                    };
                }
                Instr::Jmp(t) => pc = *t,
                Instr::Halt => break,
            }
        }
        out
    }

    /// Execute the residual program produced by the engine.
    pub fn run_residual(&self, prog: &Program<Op, Cond>, input: &[i64]) -> Vec<i64> {
        let mut store = vec![0i64; self.names.len()];
        let mut out = Vec::new();
        let mut inp = input.iter().copied();
        let mut bid = prog.entry;
        let mut budget = STEP_BUDGET;
        loop {
            let block = &prog.blocks[bid.0];
            for op in &block.ops {
                match op {
                    Op::Assign { var, expr } => store[*var] = eval_rexpr(expr, &store),
                    Op::Print(e) => out.push(eval_rexpr(e, &store)),
                    Op::Input(v) => store[*v] = inp.next().unwrap_or(0),
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
                Terminator::Unset => panic!("reached unset terminator in b{}", bid.0),
            }
            budget -= 1;
            if budget == 0 {
                panic!("residual interpreter exceeded step budget");
            }
        }
        out
    }

    /// Pretty-print a residual program using the source variable names.
    pub fn dump(&self, prog: &Program<Op, Cond>) -> String {
        let mut s = String::new();
        for (i, b) in prog.blocks.iter().enumerate() {
            let lead = if crate::residual::BlockId(i) == prog.entry {
                " (entry)"
            } else {
                ""
            };
            writeln!(s, "b{i}:{lead}").unwrap();
            for op in &b.ops {
                match op {
                    Op::Assign { var, expr } => {
                        writeln!(s, "    {} := {}", self.names[*var], self.fmt_rexpr(expr)).unwrap()
                    }
                    Op::Print(e) => writeln!(s, "    print {}", self.fmt_rexpr(e)).unwrap(),
                    Op::Input(v) => writeln!(s, "    input {}", self.names[*v]).unwrap(),
                }
            }
            match &b.term {
                Terminator::Unset => writeln!(s, "    <unset>").unwrap(),
                Terminator::Halt => writeln!(s, "    halt").unwrap(),
                Terminator::Br(t) => writeln!(s, "    br b{}", t.0).unwrap(),
                Terminator::Cond {
                    cond: Cond::IsZero(e),
                    t,
                    f,
                } => writeln!(
                    s,
                    "    if {} == 0 -> b{} else b{}",
                    self.fmt_rexpr(e),
                    t.0,
                    f.0
                )
                .unwrap(),
            }
        }
        s
    }

    fn fmt_rexpr(&self, e: &RExpr) -> String {
        match e {
            RExpr::Const(k) => k.to_string(),
            RExpr::Var(v) => self.names[*v].clone(),
            RExpr::Bin(op, a, b) => {
                format!("({} {} {})", self.fmt_rexpr(a), op.sym(), self.fmt_rexpr(b))
            }
        }
    }
}

fn eval_concrete(e: &Expr, store: &[i64]) -> i64 {
    match e {
        Expr::Int(k) => *k,
        Expr::Var(v) => store[*v],
        Expr::Bin(op, a, b) => op.eval(eval_concrete(a, store), eval_concrete(b, store)),
    }
}

fn eval_rexpr(e: &RExpr, store: &[i64]) -> i64 {
    match e {
        RExpr::Const(k) => *k,
        RExpr::Var(v) => store[*v],
        RExpr::Bin(op, a, b) => op.eval(eval_rexpr(a, store), eval_rexpr(b, store)),
    }
}
