//! A JS-subset partial evaluator with objects, arrays, and closures.
//!
//! The memory model is a **precise abstract heap of partially-static objects**
//! (partial escape analysis). Every tracked object/array/closure has a static
//! identity; references are `Ref(addr)` into the abstract heap, so aliasing and
//! mutation among static objects is exact. Objects consumed entirely at
//! specialization time are scalar-replaced away (they never appear in the
//! residual); a dynamic *primitive* stored in them residualizes.
//!
//! This is what makes the Futamura projection compelling on JS: an interpreter
//! whose AST is a static object tree specializes to compiled code, the objects,
//! property dispatch, and the interpreter itself all vanish.
//!
//! Scope of this subset:
//!   * Values: number (i64 here; full f64/NaN semantics deferred), string,
//!     boolean, undefined, and heap refs (object / array / closure).
//!   * Static object/array/closure structure folds; dynamic primitives
//!     residualize. Calls whose arguments are static (including static object
//!     refs) inline; closures capturing static values inline.
//!   * Out of scope (clear errors, not silent): an object/array/closure
//!     escaping into the residual as a runtime value; a dynamic property key or
//!     array index; dynamic-depth recursion (use `fun`).

use std::collections::HashMap;

use crate::engine::{Client, Step};
use crate::residual::{BlockId, Program, Terminator};

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

// The enums below describe the full JS-subset language surface; the demos do
// not exercise every form (e.g. `!==`, `Bool`/`Undefined` literals), so we
// allow the unused variants rather than trim the language.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bop {
    Add,
    Sub,
    Mul,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,  // ===
    Ne,  // !==
}

impl Bop {
    fn sym(self) -> &'static str {
        match self {
            Bop::Add => "+",
            Bop::Sub => "-",
            Bop::Mul => "*",
            Bop::Lt => "<",
            Bop::Le => "<=",
            Bop::Gt => ">",
            Bop::Ge => ">=",
            Bop::Eq => "===",
            Bop::Ne => "!==",
        }
    }
}

// ---------------------------------------------------------------------------
// Source AST
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum Expr {
    Num(i64),
    Str(String),
    Bool(bool),
    Undefined,
    Var(usize),                       // local slot
    Bin(Bop, Box<Expr>, Box<Expr>),
    Object(Vec<(String, Expr)>),      // { k: e, ... }
    Array(Vec<Expr>),
    Get(Box<Expr>, String),           // e.k
    Index(Box<Expr>, Box<Expr>),      // e[i]
    Func(usize),                      // reference a global function as a value
    Closure(usize, Vec<Expr>),        // make closure of function #fid capturing these
    Call(Box<Expr>, Vec<Expr>),       // callee(args)
}

pub fn num(n: i64) -> Expr {
    Expr::Num(n)
}
pub fn str_(s: &str) -> Expr {
    Expr::Str(s.to_string())
}
pub fn var(slot: usize) -> Expr {
    Expr::Var(slot)
}
pub fn bin(op: Bop, a: Expr, b: Expr) -> Expr {
    Expr::Bin(op, Box::new(a), Box::new(b))
}
pub fn obj(fields: Vec<(&str, Expr)>) -> Expr {
    Expr::Object(fields.into_iter().map(|(k, e)| (k.to_string(), e)).collect())
}
pub fn arr(es: Vec<Expr>) -> Expr {
    Expr::Array(es)
}
pub fn get(e: Expr, k: &str) -> Expr {
    Expr::Get(Box::new(e), k.to_string())
}
pub fn index(e: Expr, i: Expr) -> Expr {
    Expr::Index(Box::new(e), Box::new(i))
}
pub fn func(fid: usize) -> Expr {
    Expr::Func(fid)
}
pub fn closure(fid: usize, caps: Vec<Expr>) -> Expr {
    Expr::Closure(fid, caps)
}
pub fn call(callee: Expr, args: Vec<Expr>) -> Expr {
    Expr::Call(Box::new(callee), args)
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum Stmt {
    Let(usize, Expr),
    Set(usize, Expr),
    SetProp(Expr, String, Expr), // o.k = v
    Push(Expr, Expr),            // arr.push(v)
    Return(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    ExprStmt(Expr),
}

pub struct FuncDef {
    pub name: &'static str,
    /// total slots = captured ++ params ++ locals
    pub nslots: usize,
    /// how many leading slots are captured (filled from the closure)
    pub ncaptured: usize,
    /// how many slots after captures are parameters
    pub nparams: usize,
    #[allow(dead_code)] // documentation for demo readers
    pub slot_names: Vec<&'static str>,
    pub body: Vec<Stmt>,
}

// ---------------------------------------------------------------------------
// Bytecode
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Instr {
    PushNum(i64),
    PushStr(String),
    PushBool(bool),
    PushUndef,
    Load(usize),
    Store(usize),
    Bin(Bop),
    NewObject(Vec<String>),
    NewArray(usize),
    GetProp(String),
    SetPropOp(String),
    PushArr,
    GetIndex,
    PushFunc(usize),
    MakeClosure(usize, usize), // fid, ncaptures
    Call(usize),               // nargs
    Jmp(usize),
    JmpIfFalsy(usize),
    Ret,
    Pop,
}

// ---------------------------------------------------------------------------
// Residual IR (primitives only; objects fold away)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RExpr {
    Num(i64),
    Str(String),
    Bool(bool),
    Undef,
    Var(usize),
    Bin(Bop, Box<RExpr>, Box<RExpr>),
}

#[derive(Debug)]
pub enum Op {
    Assign { var: usize, expr: RExpr },
    Return(RExpr),
    /// Construct an object into a residual variable (a materialized escape).
    NewObject { dst: usize, fields: Vec<(String, RExpr)> },
    NewArray { dst: usize, elems: Vec<RExpr> },
    /// Append to a residual (runtime) array variable.
    PushOp { arr: usize, val: RExpr },
}

#[derive(Debug)]
pub enum Cond {
    Falsy(RExpr), // the `t` edge is taken when this value is falsy (0)
}

// ---------------------------------------------------------------------------
// Abstract values and heap
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Abs {
    Num(i64),
    Str(String),
    Bool(bool),
    Undef,
    Dyn(RExpr),
    Ref(usize),
}

impl Abs {
    fn is_dynamic(&self) -> bool {
        matches!(self, Abs::Dyn(_))
    }
    /// Convert a *primitive* abstract value to a residual expression. Panics on
    /// a heap reference; use `Js::materialize_value` for escape points.
    fn to_rexpr(&self) -> RExpr {
        match self {
            Abs::Num(n) => RExpr::Num(*n),
            Abs::Str(s) => RExpr::Str(s.clone()),
            Abs::Bool(b) => RExpr::Bool(*b),
            Abs::Undef => RExpr::Undef,
            Abs::Dyn(r) => r.clone(),
            Abs::Ref(_) => panic!(
                "a heap reference reached a primitive context (e.g. an arithmetic \
                 operand or a loop-carried object); unsupported in this subset"
            ),
        }
    }
    /// JS-ish truthiness for statics; None if it must be decided at runtime.
    fn truthy(&self) -> Option<bool> {
        match self {
            Abs::Num(n) => Some(*n != 0),
            Abs::Str(s) => Some(!s.is_empty()),
            Abs::Bool(b) => Some(*b),
            Abs::Undef => Some(false),
            Abs::Ref(_) => Some(true),
            Abs::Dyn(_) => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum HeapObj {
    Object(Vec<(String, Abs)>),
    Array(Vec<Abs>),
    Closure { fid: usize, captured: Vec<Abs> },
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
    heap: Vec<HeapObj>,
    pending_joins: Vec<(usize, Vec<usize>)>,
}

impl State {
    fn top(&self) -> &Frame {
        self.frames.last().unwrap()
    }
    fn top_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

fn compile_stmts(
    stmts: &[Stmt],
    code: &mut Vec<Instr>,
    ifs: &mut Vec<(usize, usize)>,
    heap_mods: &mut Vec<(usize, Vec<usize>)>,
) {
    for s in stmts {
        match s {
            Stmt::Let(slot, e) | Stmt::Set(slot, e) => {
                compile_expr(e, code);
                code.push(Instr::Store(*slot));
            }
            Stmt::SetProp(o, k, v) => {
                compile_expr(o, code);
                compile_expr(v, code);
                code.push(Instr::SetPropOp(k.clone()));
            }
            Stmt::Push(a, v) => {
                compile_expr(a, code);
                compile_expr(v, code);
                code.push(Instr::PushArr);
            }
            Stmt::Return(e) => {
                compile_expr(e, code);
                code.push(Instr::Ret);
            }
            Stmt::ExprStmt(e) => {
                compile_expr(e, code);
                code.push(Instr::Pop);
            }
            Stmt::If(c, t, e) => {
                compile_expr(c, code);
                let jz = code.len();
                code.push(Instr::JmpIfFalsy(0));
                compile_stmts(t, code, ifs, heap_mods);
                let jmp = code.len();
                code.push(Instr::Jmp(0));
                let else_pc = code.len();
                compile_stmts(e, code, ifs, heap_mods);
                let end = code.len();
                patch(&mut code[jz], else_pc);
                patch(&mut code[jmp], end);
                ifs.push((jz, end));
                // Which slots hold arrays/objects mutated in either arm? They
                // must escape *before* the branch so conditional mutations
                // residualize and the arms merge at the join.
                let mut hm = Vec::new();
                collect_heap_mods(t, &mut hm);
                collect_heap_mods(e, &mut hm);
                hm.sort_unstable();
                hm.dedup();
                heap_mods.push((jz, hm));
            }
            Stmt::While(c, body) => {
                let head = code.len();
                compile_expr(c, code);
                let jz = code.len();
                code.push(Instr::JmpIfFalsy(0));
                compile_stmts(body, code, ifs, heap_mods);
                code.push(Instr::Jmp(head));
                let end = code.len();
                patch(&mut code[jz], end);
            }
        }
    }
}

/// Local slots that are the base of a `push` or property write anywhere in
/// these statements (the heap objects they hold get mutated).
fn collect_heap_mods(stmts: &[Stmt], out: &mut Vec<usize>) {
    for s in stmts {
        match s {
            Stmt::Push(Expr::Var(slot), _) | Stmt::SetProp(Expr::Var(slot), _, _) => out.push(*slot),
            Stmt::If(_, t, e) => {
                collect_heap_mods(t, out);
                collect_heap_mods(e, out);
            }
            Stmt::While(_, b) => collect_heap_mods(b, out),
            _ => {}
        }
    }
}

fn compile_expr(e: &Expr, code: &mut Vec<Instr>) {
    match e {
        Expr::Num(n) => code.push(Instr::PushNum(*n)),
        Expr::Str(s) => code.push(Instr::PushStr(s.clone())),
        Expr::Bool(b) => code.push(Instr::PushBool(*b)),
        Expr::Undefined => code.push(Instr::PushUndef),
        Expr::Var(slot) => code.push(Instr::Load(*slot)),
        Expr::Bin(op, a, b) => {
            compile_expr(a, code);
            compile_expr(b, code);
            code.push(Instr::Bin(*op));
        }
        Expr::Object(fields) => {
            let mut keys = Vec::with_capacity(fields.len());
            for (k, v) in fields {
                compile_expr(v, code);
                keys.push(k.clone());
            }
            code.push(Instr::NewObject(keys));
        }
        Expr::Array(es) => {
            for e in es {
                compile_expr(e, code);
            }
            code.push(Instr::NewArray(es.len()));
        }
        Expr::Get(o, k) => {
            compile_expr(o, code);
            code.push(Instr::GetProp(k.clone()));
        }
        Expr::Index(a, i) => {
            compile_expr(a, code);
            compile_expr(i, code);
            code.push(Instr::GetIndex);
        }
        Expr::Func(fid) => code.push(Instr::PushFunc(*fid)),
        Expr::Closure(fid, caps) => {
            for c in caps {
                compile_expr(c, code);
            }
            code.push(Instr::MakeClosure(*fid, caps.len()));
        }
        Expr::Call(callee, args) => {
            compile_expr(callee, code);
            for a in args {
                compile_expr(a, code);
            }
            code.push(Instr::Call(args.len()));
        }
    }
}

fn patch(i: &mut Instr, target: usize) {
    match i {
        Instr::Jmp(t) | Instr::JmpIfFalsy(t) => *t = target,
        _ => unreachable!("patch on non-jump"),
    }
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

const MAX_DEPTH: usize = 512;

pub struct Js {
    code: Vec<Instr>,
    leaders: Vec<bool>,
    loop_head: Vec<bool>,
    loop_modified: Vec<Vec<usize>>,
    if_join: Vec<Option<(usize, Vec<usize>)>>,
    /// per `if` branch pc, the slots whose heap object is mutated in an arm;
    /// these escape to residual variables before a dynamic branch so the arms
    /// merge instead of path-splitting.
    if_heap_modified: Vec<Vec<usize>>,
    entries: Vec<usize>,
    nslots: Vec<usize>,
    ncaptured: Vec<usize>,
    nparams: Vec<usize>,
    is_recursive: Vec<bool>,
    main: usize,
}

impl Js {
    pub fn new(program: &[FuncDef]) -> Self {
        let mut code = Vec::new();
        let mut entries = vec![0; program.len()];
        let mut nslots = vec![0; program.len()];
        let mut ncaptured = vec![0; program.len()];
        let mut nparams = vec![0; program.len()];
        let mut ifs: Vec<(usize, usize)> = Vec::new();
        let mut heap_mods: Vec<(usize, Vec<usize>)> = Vec::new();

        for (fid, f) in program.iter().enumerate() {
            entries[fid] = code.len();
            nslots[fid] = f.nslots;
            ncaptured[fid] = f.ncaptured;
            nparams[fid] = f.nparams;
            compile_stmts(&f.body, &mut code, &mut ifs, &mut heap_mods);
            code.push(Instr::PushUndef);
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
                        let mut m: Vec<usize> = code[*t..=j]
                            .iter()
                            .filter_map(|ins| match ins {
                                Instr::Store(s) => Some(*s),
                                _ => None,
                            })
                            .collect();
                        m.sort_unstable();
                        m.dedup();
                        loop_modified[*t] = m;
                    }
                }
                Instr::JmpIfFalsy(t) => leaders[*t] = true,
                _ => {}
            }
        }

        let mut if_join: Vec<Option<(usize, Vec<usize>)>> = vec![None; code.len()];
        for (jz, end) in ifs {
            let mut m: Vec<usize> = code[jz + 1..end]
                .iter()
                .filter_map(|ins| match ins {
                    Instr::Store(s) => Some(*s),
                    _ => None,
                })
                .collect();
            m.sort_unstable();
            m.dedup();
            if_join[jz] = Some((end, m));
        }

        let mut if_heap_modified: Vec<Vec<usize>> = vec![Vec::new(); code.len()];
        for (jz, mods) in heap_mods {
            if_heap_modified[jz] = mods;
        }

        let is_recursive = compute_recursive(&entries, &code, program.len());
        let main = program.iter().position(|f| f.name == "main").expect("need main");

        Js {
            code,
            leaders,
            loop_head,
            loop_modified,
            if_join,
            if_heap_modified,
            entries,
            nslots,
            ncaptured,
            nparams,
            is_recursive,
            main,
        }
    }

    /// main's single parameter is the dynamic input (residual variable 0).
    pub fn start(&self) -> State {
        let mut locals = vec![Abs::Undef; self.nslots[self.main]];
        if self.nparams[self.main] >= 1 {
            locals[self.ncaptured[self.main]] = Abs::Dyn(RExpr::Var(self.input_var()));
        }
        State {
            frames: vec![Frame {
                pc: self.entries[self.main],
                func: self.main,
                locals,
                ostack: Vec::new(),
            }],
            heap: Vec::new(),
            pending_joins: Vec::new(),
        }
    }

    pub fn input_var(&self) -> usize {
        0
    }

    fn eval_bin(op: Bop, a: &Abs, b: &Abs) -> Abs {
        use Abs::*;
        match (a, b) {
            (Num(x), Num(y)) => match op {
                Bop::Add => Num(x + y),
                Bop::Sub => Num(x - y),
                Bop::Mul => Num(x * y),
                Bop::Lt => Bool(x < y),
                Bop::Le => Bool(x <= y),
                Bop::Gt => Bool(x > y),
                Bop::Ge => Bool(x >= y),
                Bop::Eq => Bool(x == y),
                Bop::Ne => Bool(x != y),
            },
            (Str(x), Str(y)) => match op {
                Bop::Add => Str(format!("{x}{y}")),
                Bop::Eq => Bool(x == y),
                Bop::Ne => Bool(x != y),
                _ => panic!("unsupported string operator {:?}", op),
            },
            (Bool(x), Bool(y)) => match op {
                Bop::Eq => Bool(x == y),
                Bop::Ne => Bool(x != y),
                _ => panic!("unsupported bool operator {:?}", op),
            },
            // Any dynamic operand: residualize. (Static prims convert to literals.)
            _ if a.is_dynamic() || b.is_dynamic() => {
                Dyn(RExpr::Bin(op, Box::new(a.to_rexpr()), Box::new(b.to_rexpr())))
            }
            _ => panic!("unsupported operand types for {:?}: {:?} {:?}", op, a, b),
        }
    }
}

fn compute_recursive(entries: &[usize], code: &[Instr], nfuncs: usize) -> Vec<bool> {
    use std::collections::HashSet;
    let mut calls: Vec<HashSet<usize>> = vec![HashSet::new(); nfuncs];
    for f in 0..nfuncs {
        let lo = entries[f];
        let hi = if f + 1 < nfuncs { entries[f + 1] } else { code.len() };
        for ins in &code[lo..hi] {
            match ins {
                Instr::PushFunc(g) | Instr::MakeClosure(g, _) => {
                    calls[f].insert(*g);
                }
                _ => {}
            }
        }
    }
    let mut rec = vec![false; nfuncs];
    for f in 0..nfuncs {
        let mut stack: Vec<usize> = calls[f].iter().copied().collect();
        let mut seen = HashSet::new();
        while let Some(g) = stack.pop() {
            if g == f {
                rec[f] = true;
                break;
            }
            if seen.insert(g) {
                stack.extend(calls[g].iter().copied());
            }
        }
    }
    rec
}

impl Client for Js {
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
            return self.jump_to(pc, s, out);
        }

        match &self.code[pc] {
            Instr::PushNum(n) => self.push(s, Abs::Num(*n)),
            Instr::PushStr(st) => self.push(s, Abs::Str(st.clone())),
            Instr::PushBool(b) => self.push(s, Abs::Bool(*b)),
            Instr::PushUndef => self.push(s, Abs::Undef),
            Instr::Load(slot) => {
                let v = s.top().locals[*slot].clone();
                self.push(s, v)
            }
            Instr::Store(slot) => {
                let v = s.top_mut().ostack.pop().unwrap();
                s.top_mut().locals[*slot] = v;
                self.advance(s)
            }
            Instr::Pop => {
                s.top_mut().ostack.pop();
                self.advance(s)
            }
            Instr::Bin(op) => {
                let b = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                let r = Js::eval_bin(*op, &a, &b);
                self.push(s, r)
            }
            Instr::NewObject(keys) => {
                let mut fields = Vec::with_capacity(keys.len());
                let n = keys.len();
                let base = s.top().ostack.len() - n;
                let vals: Vec<Abs> = s.top_mut().ostack.split_off(base);
                for (k, v) in keys.iter().zip(vals) {
                    fields.push((k.clone(), v));
                }
                let addr = s.heap.len();
                s.heap.push(HeapObj::Object(fields));
                self.push(s, Abs::Ref(addr))
            }
            Instr::NewArray(n) => {
                let base = s.top().ostack.len() - *n;
                let vals: Vec<Abs> = s.top_mut().ostack.split_off(base);
                let addr = s.heap.len();
                s.heap.push(HeapObj::Array(vals));
                self.push(s, Abs::Ref(addr))
            }
            Instr::GetProp(k) => {
                let o = s.top_mut().ostack.pop().unwrap();
                let v = self.get_prop(s, &o, k);
                self.push(s, v)
            }
            Instr::SetPropOp(k) => {
                let v = s.top_mut().ostack.pop().unwrap();
                let o = s.top_mut().ostack.pop().unwrap();
                match o {
                    Abs::Ref(addr) => {
                        if let HeapObj::Object(fields) = &mut s.heap[addr] {
                            if let Some(slot) = fields.iter_mut().find(|(fk, _)| fk == k) {
                                slot.1 = v;
                            } else {
                                fields.push((k.clone(), v));
                            }
                        } else {
                            panic!("SetProp on non-object");
                        }
                    }
                    _ => panic!("SetProp on dynamic/non-ref object (unsupported)"),
                }
                self.advance(s)
            }
            Instr::PushArr => {
                let v = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                match a {
                    // Static array: mutate the abstract heap (still scalar).
                    Abs::Ref(addr) => match &mut s.heap[addr] {
                        HeapObj::Array(elems) => elems.push(v),
                        _ => panic!("push on non-array"),
                    },
                    // Dynamic (escaped) array: residualize the push.
                    Abs::Dyn(RExpr::Var(id)) => {
                        let r = self.materialize_value(&v, &s.heap, out);
                        out.push(Op::PushOp { arr: id, val: r });
                    }
                    _ => panic!("push on a non-array value"),
                }
                self.advance(s)
            }
            Instr::GetIndex => {
                let i = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                let v = self.get_index(s, &a, &i);
                self.push(s, v)
            }
            Instr::PushFunc(fid) => {
                let addr = s.heap.len();
                s.heap.push(HeapObj::Closure {
                    fid: *fid,
                    captured: Vec::new(),
                });
                self.push(s, Abs::Ref(addr))
            }
            Instr::MakeClosure(fid, ncap) => {
                let base = s.top().ostack.len() - *ncap;
                let captured: Vec<Abs> = s.top_mut().ostack.split_off(base);
                let addr = s.heap.len();
                s.heap.push(HeapObj::Closure {
                    fid: *fid,
                    captured,
                });
                self.push(s, Abs::Ref(addr))
            }
            Instr::Call(nargs) => self.do_call(s, *nargs),
            Instr::Jmp(t) => self.jump_to(*t, s, out),
            Instr::JmpIfFalsy(t) => {
                let c = s.top_mut().ostack.pop().unwrap();
                match c.truthy() {
                    Some(falsy) => {
                        // falsy => take the jump (to else / past loop)
                        s.top_mut().pc = if !falsy { *t } else { pc + 1 };
                        Step::Continue
                    }
                    None => {
                        let r = c.to_rexpr();
                        // Arrays/objects mutated in either arm must escape to a
                        // residual variable *before* we split, so the conditional
                        // mutations residualize and both arms rejoin identically.
                        let hm = self.if_heap_modified[pc].clone();
                        if !hm.is_empty() {
                            self.materialize(s, &hm, out);
                        }
                        let mut tstate = s.clone();
                        tstate.top_mut().pc = *t; // falsy branch
                        let mut fstate = s.clone();
                        fstate.top_mut().pc = pc + 1; // truthy branch
                        // Only schedule a join merge when arms assign primitive
                        // slots differently. With nothing to merge (e.g. an empty
                        // else, or arms that only push to an already-escaped
                        // array), a marker would just make the arms' states differ
                        // by the un-popped stack and block the natural merge.
                        if let Some((end, mods)) = &self.if_join[pc] {
                            if !mods.is_empty() {
                                tstate.pending_joins.push((*end, mods.clone()));
                                fstate.pending_joins.push((*end, mods.clone()));
                            }
                        }
                        Step::Branch {
                            cond: Cond::Falsy(r),
                            t: tstate,
                            f: fstate,
                        }
                    }
                }
            }
            Instr::Ret => {
                let v = s.top_mut().ostack.pop().unwrap_or(Abs::Undef);
                s.frames.pop();
                if s.frames.is_empty() {
                    let r = self.materialize_value(&v, &s.heap, out);
                    out.push(Op::Return(r));
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
        // Only local primitive slots of the top frame are generalized (loop
        // induction variables). Heap divergence is out of scope.
        let mut g = from.clone();
        let fi = g.frames.len() - 1;
        let slots: Vec<usize> = (0..g.frames[fi].locals.len())
            .filter(|&slot| seen.frames[fi].locals[slot] != from.frames[fi].locals[slot])
            .collect();
        self.materialize(&mut g, &slots, out);
        g
    }
}

impl Js {
    fn push(&self, s: &mut State, v: Abs) -> Step<Js> {
        s.top_mut().ostack.push(v);
        self.advance(s)
    }
    fn advance(&self, s: &mut State) -> Step<Js> {
        s.top_mut().pc += 1;
        Step::Continue
    }

    fn get_prop(&self, s: &State, o: &Abs, k: &str) -> Abs {
        match o {
            Abs::Ref(addr) => match &s.heap[*addr] {
                HeapObj::Object(fields) => fields
                    .iter()
                    .find(|(fk, _)| fk == k)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(Abs::Undef),
                HeapObj::Array(elems) if k == "length" => Abs::Num(elems.len() as i64),
                _ => panic!("property {k} on non-object"),
            },
            _ => panic!(
                "property access on a dynamic value (unsupported: objects must be \
                 statically known)"
            ),
        }
    }

    fn get_index(&self, s: &State, a: &Abs, i: &Abs) -> Abs {
        let addr = match a {
            Abs::Ref(addr) => *addr,
            _ => panic!("index on a dynamic value (unsupported)"),
        };
        match (&s.heap[addr], i) {
            (HeapObj::Array(elems), Abs::Num(n)) => {
                elems.get(*n as usize).cloned().unwrap_or(Abs::Undef)
            }
            (HeapObj::Object(fields), Abs::Str(k)) => fields
                .iter()
                .find(|(fk, _)| fk == k)
                .map(|(_, v)| v.clone())
                .unwrap_or(Abs::Undef),
            (_, Abs::Dyn(_)) => panic!("dynamic index/key (unsupported: must be static)"),
            _ => panic!("bad index/key types"),
        }
    }

    fn do_call(&self, s: &mut State, nargs: usize) -> Step<Js> {
        let base = s.top().ostack.len() - nargs;
        let args: Vec<Abs> = s.top_mut().ostack.split_off(base);
        let callee = s.top_mut().ostack.pop().unwrap();

        let (fid, captured) = match callee {
            Abs::Ref(addr) => match &s.heap[addr] {
                HeapObj::Closure { fid, captured } => (*fid, captured.clone()),
                _ => panic!("call of a non-function"),
            },
            _ => panic!("call of a dynamic value (unsupported: callee must be known)"),
        };

        // Dynamic-depth recursion is out of scope for this subset.
        if self.is_recursive[fid] && args.iter().all(|a| a.is_dynamic()) && !args.is_empty() {
            panic!(
                "dynamic-depth recursion in the JS subset is unsupported; the \
                 `fun` client demonstrates residual function generation"
            );
        }
        if s.frames.len() >= MAX_DEPTH {
            panic!("inlining depth exceeded (non-terminating static recursion?)");
        }

        let mut locals = vec![Abs::Undef; self.nslots[fid]];
        for (i, c) in captured.into_iter().enumerate() {
            locals[i] = c;
        }
        for (i, a) in args.into_iter().enumerate() {
            let slot = self.ncaptured[fid] + i;
            if slot < locals.len() {
                locals[slot] = a;
            }
        }
        s.top_mut().pc += 1; // resume after the call
        s.frames.push(Frame {
            pc: self.entries[fid],
            func: fid,
            locals,
            ostack: Vec::new(),
        });
        Step::Continue
    }

    /// Materialize an abstract value into a residual expression, emitting
    /// construction ops for any heap object/array it references (escape). This
    /// is the second half of partial escape analysis: objects that cannot be
    /// scalar-replaced (because they become runtime values) are reconstructed in
    /// the residual. Nested objects materialize depth-first so each is defined
    /// before it is referenced.
    fn materialize_value(&self, v: &Abs, heap: &[HeapObj], out: &mut Vec<Op>) -> RExpr {
        match v {
            Abs::Ref(addr) => {
                let dst = Js::escape_var(*addr);
                self.materialize_into(dst, v, heap, out);
                RExpr::Var(dst)
            }
            _ => v.to_rexpr(),
        }
    }

    /// Materialize `v` into the residual variable `dst`. For a heap reference,
    /// emits the construction op (recursively materializing nested values).
    fn materialize_into(&self, dst: usize, v: &Abs, heap: &[HeapObj], out: &mut Vec<Op>) {
        match v {
            Abs::Ref(addr) => match &heap[*addr] {
                HeapObj::Object(fields) => {
                    let rf: Vec<(String, RExpr)> = fields
                        .iter()
                        .map(|(k, fv)| (k.clone(), self.materialize_value(fv, heap, out)))
                        .collect();
                    out.push(Op::NewObject { dst, fields: rf });
                }
                HeapObj::Array(elems) => {
                    let re: Vec<RExpr> =
                        elems.iter().map(|e| self.materialize_value(e, heap, out)).collect();
                    out.push(Op::NewArray { dst, elems: re });
                }
                HeapObj::Closure { .. } => panic!(
                    "a closure escaped into the residual as a runtime value; \
                     residual closures are not supported in this subset"
                ),
            },
            _ => out.push(Op::Assign { var: dst, expr: v.to_rexpr() }),
        }
    }

    fn escape_var(addr: usize) -> usize {
        // Disjoint from the input var (0) and the loop/if materialization band.
        100_000 + addr
    }

    fn dynamically_controlled(&self, s: &State) -> bool {
        if !self.loop_head[s.top().pc] {
            return false;
        }
        // Dry-run the loop condition under the current locals/heap.
        let mut probe = s.clone();
        for _ in 0..512 {
            let pc = probe.top().pc;
            match &self.code[pc] {
                Instr::PushNum(n) => probe.top_mut().ostack.push(Abs::Num(*n)),
                Instr::PushStr(st) => probe.top_mut().ostack.push(Abs::Str(st.clone())),
                Instr::PushBool(b) => probe.top_mut().ostack.push(Abs::Bool(*b)),
                Instr::PushUndef => probe.top_mut().ostack.push(Abs::Undef),
                Instr::Load(slot) => {
                    let v = probe.top().locals[*slot].clone();
                    probe.top_mut().ostack.push(v);
                }
                Instr::Bin(op) => {
                    let b = probe.top_mut().ostack.pop().unwrap();
                    let a = probe.top_mut().ostack.pop().unwrap();
                    probe.top_mut().ostack.push(Js::eval_bin(*op, &a, &b));
                }
                Instr::GetProp(k) => {
                    let o = probe.top_mut().ostack.pop().unwrap();
                    let v = self.get_prop(&probe, &o, k);
                    probe.top_mut().ostack.push(v);
                }
                Instr::GetIndex => {
                    let i = probe.top_mut().ostack.pop().unwrap();
                    let a = probe.top_mut().ostack.pop().unwrap();
                    let v = self.get_index(&probe, &a, &i);
                    probe.top_mut().ostack.push(v);
                }
                Instr::JmpIfFalsy(_) => {
                    return probe.top().ostack.last().map(|v| v.is_dynamic()).unwrap_or(true);
                }
                _ => return true,
            }
            probe.top_mut().pc += 1;
        }
        true
    }

    fn jump_to(&self, target: usize, s: &State, out: &mut Vec<Op>) -> Step<Js> {
        let mut ns = s.clone();
        ns.top_mut().pc = target;
        if self.loop_head[target] && self.dynamically_controlled(&ns) {
            // Materialize loop-modified primitive slots AND any heap-reference
            // slot (an array/object live across a dynamic loop must become a
            // residual value, else the abstract heap grows without bound).
            let fi = ns.frames.len() - 1;
            let mut slots = self.loop_modified[target].clone();
            for slot in 0..ns.frames[fi].locals.len() {
                if matches!(ns.frames[fi].locals[slot], Abs::Ref(_)) && !slots.contains(&slot) {
                    slots.push(slot);
                }
            }
            slots.sort_unstable();
            self.materialize(&mut ns, &slots, out);
        }
        while ns.pending_joins.last().map(|(e, _)| *e) == Some(target) {
            let (_, m) = ns.pending_joins.pop().unwrap();
            self.materialize(&mut ns, &m, out);
        }
        Step::Jump(ns)
    }

    fn materialize(&self, ns: &mut State, slots: &[usize], out: &mut Vec<Op>) {
        let fi = ns.frames.len() - 1;
        let func = ns.frames[fi].func;
        let heap = ns.heap.clone();
        for &slot in slots {
            let cur = ns.frames[fi].locals[slot].clone();
            // A stable residual variable per (function, slot) keeps loops
            // converging across iterations and generalization rounds. A heap
            // object is materialized into that variable (so its mutations in the
            // loop become residual ops and the abstract heap stops growing).
            let id = self.stable_id(func, slot);
            if cur != Abs::Dyn(RExpr::Var(id)) {
                self.materialize_into(id, &cur, &heap, out);
            }
            ns.frames[fi].locals[slot] = Abs::Dyn(RExpr::Var(id));
        }
    }

    fn stable_id(&self, func: usize, slot: usize) -> usize {
        // Reserve a deterministic band of ids for loop/if materialization,
        // disjoint from input vars (which are small).
        1_000 + func * 64 + slot
    }
}

// ---------------------------------------------------------------------------
// Oracles
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum JsVal {
    Num(i64),
    Str(String),
    Bool(bool),
    Undef,
    Ref(usize),
}

enum CHeap {
    Object(Vec<(String, JsVal)>),
    Array(Vec<JsVal>),
    Closure { fid: usize, captured: Vec<JsVal> },
}

impl Js {
    fn cbin(op: Bop, a: &JsVal, b: &JsVal) -> JsVal {
        match (a, b) {
            (JsVal::Num(x), JsVal::Num(y)) => match op {
                Bop::Add => JsVal::Num(x + y),
                Bop::Sub => JsVal::Num(x - y),
                Bop::Mul => JsVal::Num(x * y),
                Bop::Lt => JsVal::Bool(x < y),
                Bop::Le => JsVal::Bool(x <= y),
                Bop::Gt => JsVal::Bool(x > y),
                Bop::Ge => JsVal::Bool(x >= y),
                Bop::Eq => JsVal::Bool(x == y),
                Bop::Ne => JsVal::Bool(x != y),
            },
            (JsVal::Str(x), JsVal::Str(y)) => match op {
                Bop::Add => JsVal::Str(format!("{x}{y}")),
                Bop::Eq => JsVal::Bool(x == y),
                Bop::Ne => JsVal::Bool(x != y),
                _ => panic!("bad string op"),
            },
            (JsVal::Bool(x), JsVal::Bool(y)) => match op {
                Bop::Eq => JsVal::Bool(x == y),
                Bop::Ne => JsVal::Bool(x != y),
                _ => panic!("bad bool op"),
            },
            _ => panic!("bad operand types in reference interpreter"),
        }
    }

    fn ctruthy(v: &JsVal) -> bool {
        match v {
            JsVal::Num(n) => *n != 0,
            JsVal::Str(s) => !s.is_empty(),
            JsVal::Bool(b) => *b,
            JsVal::Undef => false,
            JsVal::Ref(_) => true,
        }
    }

    /// Reference interpreter (ground truth) over the bytecode.
    pub fn run_reference(&self, input: i64) -> DeepVal {
        let mut heap: Vec<CHeap> = Vec::new();
        let mut locals = vec![JsVal::Undef; self.nslots[self.main]];
        if self.nparams[self.main] >= 1 {
            locals[self.ncaptured[self.main]] = JsVal::Num(input);
        }
        let v = self.run_ref_fn(self.main, locals, &mut heap, 0);
        deep(&v, &heap)
    }

    fn run_ref_fn(&self, fid: usize, locals: Vec<JsVal>, heap: &mut Vec<CHeap>, depth: usize) -> JsVal {
        if depth > 100_000 {
            panic!("reference recursion too deep");
        }
        let mut locals = locals;
        let mut ostack: Vec<JsVal> = Vec::new();
        let mut pc = self.entries[fid];
        loop {
            match &self.code[pc] {
                Instr::PushNum(n) => ostack.push(JsVal::Num(*n)),
                Instr::PushStr(s) => ostack.push(JsVal::Str(s.clone())),
                Instr::PushBool(b) => ostack.push(JsVal::Bool(*b)),
                Instr::PushUndef => ostack.push(JsVal::Undef),
                Instr::Load(slot) => ostack.push(locals[*slot].clone()),
                Instr::Store(slot) => locals[*slot] = ostack.pop().unwrap(),
                Instr::Pop => {
                    ostack.pop();
                }
                Instr::Bin(op) => {
                    let b = ostack.pop().unwrap();
                    let a = ostack.pop().unwrap();
                    ostack.push(Js::cbin(*op, &a, &b));
                }
                Instr::NewObject(keys) => {
                    let base = ostack.len() - keys.len();
                    let vals: Vec<JsVal> = ostack.split_off(base);
                    let fields = keys.iter().cloned().zip(vals).collect();
                    let addr = heap.len();
                    heap.push(CHeap::Object(fields));
                    ostack.push(JsVal::Ref(addr));
                }
                Instr::NewArray(n) => {
                    let base = ostack.len() - n;
                    let vals: Vec<JsVal> = ostack.split_off(base);
                    let addr = heap.len();
                    heap.push(CHeap::Array(vals));
                    ostack.push(JsVal::Ref(addr));
                }
                Instr::GetProp(k) => {
                    let o = ostack.pop().unwrap();
                    let v = match (&o, k.as_str()) {
                        (JsVal::Ref(a), _) => match &heap[*a] {
                            CHeap::Object(f) => f
                                .iter()
                                .find(|(fk, _)| fk == k)
                                .map(|(_, v)| v.clone())
                                .unwrap_or(JsVal::Undef),
                            CHeap::Array(e) if k == "length" => JsVal::Num(e.len() as i64),
                            _ => panic!("bad prop"),
                        },
                        _ => panic!("prop on non-ref"),
                    };
                    ostack.push(v);
                }
                Instr::SetPropOp(k) => {
                    let v = ostack.pop().unwrap();
                    let o = ostack.pop().unwrap();
                    if let JsVal::Ref(a) = o {
                        if let CHeap::Object(f) = &mut heap[a] {
                            if let Some(slot) = f.iter_mut().find(|(fk, _)| fk == k) {
                                slot.1 = v;
                            } else {
                                f.push((k.clone(), v));
                            }
                        }
                    }
                }
                Instr::PushArr => {
                    let v = ostack.pop().unwrap();
                    let a = ostack.pop().unwrap();
                    if let JsVal::Ref(ad) = a {
                        if let CHeap::Array(e) = &mut heap[ad] {
                            e.push(v);
                        } else {
                            panic!("push on non-array");
                        }
                    } else {
                        panic!("push on non-ref");
                    }
                }
                Instr::GetIndex => {
                    let i = ostack.pop().unwrap();
                    let a = ostack.pop().unwrap();
                    let ad = match a {
                        JsVal::Ref(ad) => ad,
                        _ => panic!("index non-ref"),
                    };
                    let v = match (&heap[ad], &i) {
                        (CHeap::Array(e), JsVal::Num(ix)) => {
                            e.get(*ix as usize).cloned().unwrap_or(JsVal::Undef)
                        }
                        (CHeap::Object(f), JsVal::Str(k)) => f
                            .iter()
                            .find(|(fk, _)| fk == k)
                            .map(|(_, v)| v.clone())
                            .unwrap_or(JsVal::Undef),
                        _ => panic!("bad index/key"),
                    };
                    ostack.push(v);
                }
                Instr::PushFunc(fid) => {
                    let addr = heap.len();
                    heap.push(CHeap::Closure {
                        fid: *fid,
                        captured: Vec::new(),
                    });
                    ostack.push(JsVal::Ref(addr));
                }
                Instr::MakeClosure(fid, ncap) => {
                    let base = ostack.len() - ncap;
                    let captured: Vec<JsVal> = ostack.split_off(base);
                    let addr = heap.len();
                    heap.push(CHeap::Closure {
                        fid: *fid,
                        captured,
                    });
                    ostack.push(JsVal::Ref(addr));
                }
                Instr::Call(nargs) => {
                    let base = ostack.len() - nargs;
                    let args: Vec<JsVal> = ostack.split_off(base);
                    let callee = ostack.pop().unwrap();
                    let (cfid, captured) = match callee {
                        JsVal::Ref(a) => match &heap[a] {
                            CHeap::Closure { fid, captured } => (*fid, captured.clone()),
                            _ => panic!("call non-fn"),
                        },
                        _ => panic!("call non-ref"),
                    };
                    let mut clocals = vec![JsVal::Undef; self.nslots[cfid]];
                    for (i, c) in captured.into_iter().enumerate() {
                        clocals[i] = c;
                    }
                    for (i, a) in args.into_iter().enumerate() {
                        let slot = self.ncaptured[cfid] + i;
                        if slot < clocals.len() {
                            clocals[slot] = a;
                        }
                    }
                    let r = self.run_ref_fn(cfid, clocals, heap, depth + 1);
                    ostack.push(r);
                }
                Instr::Jmp(t) => {
                    pc = *t;
                    continue;
                }
                Instr::JmpIfFalsy(t) => {
                    let c = ostack.pop().unwrap();
                    pc = if !Js::ctruthy(&c) { *t } else { pc + 1 };
                    continue;
                }
                Instr::Ret => return ostack.pop().unwrap_or(JsVal::Undef),
            }
            pc += 1;
        }
    }

    /// Execute the residual program.
    pub fn run_residual(&self, prog: &Program<Op, Cond>, input: i64) -> DeepVal {
        let mut store: HashMap<usize, JsVal> = HashMap::new();
        let mut heap: Vec<CHeap> = Vec::new();
        store.insert(self.input_var(), JsVal::Num(input));
        let mut ret = JsVal::Undef;
        let mut bid = prog.entry;
        let mut budget = 100_000_000u64;
        loop {
            let block = &prog.blocks[bid.0];
            for op in &block.ops {
                match op {
                    Op::Assign { var, expr } => {
                        let v = eval_rexpr(expr, &store);
                        store.insert(*var, v);
                    }
                    Op::Return(e) => ret = eval_rexpr(e, &store),
                    Op::NewObject { dst, fields } => {
                        let f = fields
                            .iter()
                            .map(|(k, e)| (k.clone(), eval_rexpr(e, &store)))
                            .collect();
                        let addr = heap.len();
                        heap.push(CHeap::Object(f));
                        store.insert(*dst, JsVal::Ref(addr));
                    }
                    Op::NewArray { dst, elems } => {
                        let e = elems.iter().map(|e| eval_rexpr(e, &store)).collect();
                        let addr = heap.len();
                        heap.push(CHeap::Array(e));
                        store.insert(*dst, JsVal::Ref(addr));
                    }
                    Op::PushOp { arr, val } => {
                        let v = eval_rexpr(val, &store);
                        match store.get(arr).cloned() {
                            Some(JsVal::Ref(addr)) => match &mut heap[addr] {
                                CHeap::Array(e) => e.push(v),
                                _ => panic!("push on non-array"),
                            },
                            _ => panic!("push on undefined array var"),
                        }
                    }
                }
            }
            match &block.term {
                Terminator::Halt => break,
                Terminator::Br(b) => bid = *b,
                Terminator::Cond {
                    cond: Cond::Falsy(e),
                    t,
                    f,
                } => {
                    bid = if !Js::ctruthy(&eval_rexpr(e, &store)) {
                        *t
                    } else {
                        *f
                    }
                }
                Terminator::Unset => panic!("unset terminator"),
            }
            budget -= 1;
            if budget == 0 {
                panic!("residual exceeded budget");
            }
        }
        deep(&ret, &heap)
    }

    pub fn dump(&self, prog: &Program<Op, Cond>) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        for (i, b) in prog.blocks.iter().enumerate() {
            let lead = if BlockId(i) == prog.entry { " (entry)" } else { "" };
            writeln!(s, "b{i}:{lead}").unwrap();
            for op in &b.ops {
                match op {
                    Op::Assign { var, expr } => {
                        writeln!(s, "    v{var} := {}", fmt_rexpr(expr)).unwrap()
                    }
                    Op::Return(e) => writeln!(s, "    return {}", fmt_rexpr(e)).unwrap(),
                    Op::NewObject { dst, fields } => {
                        let fs: Vec<String> = fields
                            .iter()
                            .map(|(k, e)| format!("{k}: {}", fmt_rexpr(e)))
                            .collect();
                        writeln!(s, "    v{dst} := {{{}}}", fs.join(", ")).unwrap()
                    }
                    Op::NewArray { dst, elems } => {
                        let es: Vec<String> = elems.iter().map(fmt_rexpr).collect();
                        writeln!(s, "    v{dst} := [{}]", es.join(", ")).unwrap()
                    }
                    Op::PushOp { arr, val } => {
                        writeln!(s, "    v{arr}.push({})", fmt_rexpr(val)).unwrap()
                    }
                }
            }
            match &b.term {
                Terminator::Unset => writeln!(s, "    <unset>").unwrap(),
                Terminator::Halt => writeln!(s, "    halt").unwrap(),
                Terminator::Br(t) => writeln!(s, "    br b{}", t.0).unwrap(),
                Terminator::Cond {
                    cond: Cond::Falsy(e),
                    t,
                    f,
                } => writeln!(s, "    if !{} -> b{} else b{}", fmt_rexpr(e), t.0, f.0).unwrap(),
            }
        }
        s
    }
}

fn eval_rexpr(e: &RExpr, store: &HashMap<usize, JsVal>) -> JsVal {
    match e {
        RExpr::Num(n) => JsVal::Num(*n),
        RExpr::Str(s) => JsVal::Str(s.clone()),
        RExpr::Bool(b) => JsVal::Bool(*b),
        RExpr::Undef => JsVal::Undef,
        RExpr::Var(v) => store.get(v).cloned().unwrap_or(JsVal::Undef),
        RExpr::Bin(op, a, b) => Js::cbin(*op, &eval_rexpr(a, store), &eval_rexpr(b, store)),
    }
}

fn fmt_rexpr(e: &RExpr) -> String {
    match e {
        RExpr::Num(n) => n.to_string(),
        RExpr::Str(s) => format!("{s:?}"),
        RExpr::Bool(b) => b.to_string(),
        RExpr::Undef => "undefined".to_string(),
        RExpr::Var(v) => format!("v{v}"),
        RExpr::Bin(op, a, b) => format!("({} {} {})", fmt_rexpr(a), op.sym(), fmt_rexpr(b)),
    }
}

/// A fully-expanded value, for structural comparison of objects/arrays returned
/// by the reference interpreter and the residual (whose heap addresses differ).
#[derive(Clone, Debug, PartialEq)]
pub enum DeepVal {
    Num(i64),
    Str(String),
    Bool(bool),
    Undef,
    Object(Vec<(String, DeepVal)>),
    Array(Vec<DeepVal>),
    Closure(usize),
}

fn deep(v: &JsVal, heap: &[CHeap]) -> DeepVal {
    match v {
        JsVal::Num(n) => DeepVal::Num(*n),
        JsVal::Str(s) => DeepVal::Str(s.clone()),
        JsVal::Bool(b) => DeepVal::Bool(*b),
        JsVal::Undef => DeepVal::Undef,
        JsVal::Ref(a) => match &heap[*a] {
            CHeap::Object(f) => {
                DeepVal::Object(f.iter().map(|(k, v)| (k.clone(), deep(v, heap))).collect())
            }
            CHeap::Array(e) => DeepVal::Array(e.iter().map(|v| deep(v, heap)).collect()),
            CHeap::Closure { fid, .. } => DeepVal::Closure(*fid),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine;

    fn node(op: &str, fields: Vec<(&str, Expr)>) -> Expr {
        let mut all = vec![("op", str_(op))];
        all.extend(fields);
        obj(all)
    }

    /// The Futamura projection on JS: specialize an interpreter against a static
    /// AST (objects) with a dynamic environment. The interpreter and the AST
    /// objects must vanish, leaving residual arithmetic that matches the oracle.
    #[test]
    fn interpreter_specializes_away() {
        // eval(node, env)
        let eval = FuncDef {
            name: "eval",
            nslots: 2,
            ncaptured: 0,
            nparams: 2,
            slot_names: vec!["node", "env"],
            body: vec![Stmt::If(
                bin(Bop::Eq, get(var(0), "op"), str_("lit")),
                vec![Stmt::Return(get(var(0), "val"))],
                vec![Stmt::If(
                    bin(Bop::Eq, get(var(0), "op"), str_("var")),
                    vec![Stmt::Return(index(var(1), get(var(0), "name")))],
                    vec![Stmt::If(
                        bin(Bop::Eq, get(var(0), "op"), str_("add")),
                        vec![Stmt::Return(bin(
                            Bop::Add,
                            call(func(0), vec![get(var(0), "l"), var(1)]),
                            call(func(0), vec![get(var(0), "r"), var(1)]),
                        ))],
                        vec![Stmt::Return(bin(
                            Bop::Mul,
                            call(func(0), vec![get(var(0), "l"), var(1)]),
                            call(func(0), vec![get(var(0), "r"), var(1)]),
                        ))],
                    )],
                )],
            )],
        };
        // ast = (x + 3) * (x + x), env = { x: input }
        let ast = node(
            "mul",
            vec![
                (
                    "l",
                    node(
                        "add",
                        vec![
                            ("l", node("var", vec![("name", str_("x"))])),
                            ("r", node("lit", vec![("val", num(3))])),
                        ],
                    ),
                ),
                (
                    "r",
                    node(
                        "add",
                        vec![
                            ("l", node("var", vec![("name", str_("x"))])),
                            ("r", node("var", vec![("name", str_("x"))])),
                        ],
                    ),
                ),
            ],
        );
        let main = FuncDef {
            name: "main",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["input"],
            body: vec![Stmt::Return(call(
                func(0),
                vec![ast, obj(vec![("x", var(0))])],
            ))],
        };
        let program = [eval, main];
        let vm = Js::new(&program);
        let mut prog = engine::specialize(&vm, vm.start());
        crate::residual::simplify(&mut prog);

        // The interpreter and AST are gone: pure straight-line residual.
        assert_eq!(prog.blocks.len(), 1, "interpreter did not fully specialize away");

        for x in [-3, 0, 1, 7, 42] {
            assert_eq!(
                vm.run_reference(x),
                vm.run_residual(&prog, x),
                "residual diverged from reference at x={x}"
            );
        }
    }

    /// Partial escape analysis: an object that becomes the program's result is
    /// materialized into residual construction code, and round-trips deeply.
    #[test]
    fn object_escape_materializes() {
        // main(a) = return { pair: [a, a*a], meta: { x: a, neg: 0 - a } };
        let main = FuncDef {
            name: "main",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["a"],
            body: vec![Stmt::Return(obj(vec![
                ("pair", arr(vec![var(0), bin(Bop::Mul, var(0), var(0))])),
                (
                    "meta",
                    obj(vec![("x", var(0)), ("neg", bin(Bop::Sub, num(0), var(0)))]),
                ),
            ]))],
        };
        let vm = Js::new(&[main]);
        let prog = engine::specialize(&vm, vm.start());

        // The residual constructs at least one object (the escape happened).
        let has_newobj = prog
            .blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|op| matches!(op, Op::NewObject { .. }));
        assert!(has_newobj, "escaped object was not materialized");

        for a in [-5, 0, 4, 13] {
            let got = vm.run_residual(&prog, a);
            assert_eq!(vm.run_reference(a), got, "escape diverged at a={a}");
            // And it really is a structured object.
            assert!(matches!(got, DeepVal::Object(_)));
        }
    }

    /// `map`, written in the subset, specializes away over a static-length array
    /// with a known callback: the loop unrolls and the result array escapes.
    #[test]
    fn map_specializes_away() {
        let double = FuncDef {
            name: "double",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["x"],
            body: vec![Stmt::Return(bin(Bop::Mul, var(0), num(2)))],
        };
        let map = FuncDef {
            name: "map",
            nslots: 4,
            ncaptured: 0,
            nparams: 2,
            slot_names: vec!["xs", "f", "result", "i"],
            body: vec![
                Stmt::Let(2, arr(vec![])),
                Stmt::Let(3, num(0)),
                Stmt::While(
                    bin(Bop::Lt, var(3), get(var(0), "length")),
                    vec![
                        Stmt::Push(var(2), call(var(1), vec![index(var(0), var(3))])),
                        Stmt::Set(3, bin(Bop::Add, var(3), num(1))),
                    ],
                ),
                Stmt::Return(var(2)),
            ],
        };
        let main = FuncDef {
            name: "main",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["a"],
            body: vec![Stmt::Return(call(
                func(1),
                vec![
                    arr(vec![var(0), bin(Bop::Add, var(0), num(1)), num(7)]),
                    func(0),
                ],
            ))],
        };
        let vm = Js::new(&[double, map, main]);
        let prog = engine::specialize(&vm, vm.start());
        for a in [0, 3, 9] {
            let got = vm.run_residual(&prog, a);
            assert_eq!(vm.run_reference(a), got, "map diverged at a={a}");
            // [a*2, (a+1)*2, 14]
            assert_eq!(
                got,
                DeepVal::Array(vec![
                    DeepVal::Num(a * 2),
                    DeepVal::Num((a + 1) * 2),
                    DeepVal::Num(14)
                ])
            );
        }
    }

    /// A dynamic-length array built in a dynamic loop: the array escapes to a
    /// residual variable, the pushes residualize, and the loop converges (the
    /// abstract heap does not grow). The result is a true runtime array whose
    /// length depends on the input.
    #[test]
    fn dynamic_length_array_in_loop() {
        // main(n) = { let r = []; let i = 0; while i < n { r.push(i*i); i=i+1 } return r }
        let main = FuncDef {
            name: "main",
            nslots: 3,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["n", "r", "i"],
            body: vec![
                Stmt::Let(1, arr(vec![])),
                Stmt::Let(2, num(0)),
                Stmt::While(
                    bin(Bop::Lt, var(2), var(0)),
                    vec![
                        Stmt::Push(var(1), bin(Bop::Mul, var(2), var(2))),
                        Stmt::Set(2, bin(Bop::Add, var(2), num(1))),
                    ],
                ),
                Stmt::Return(var(1)),
            ],
        };
        let vm = Js::new(&[main]);
        let mut prog = engine::specialize(&vm, vm.start());
        crate::residual::simplify(&mut prog);

        // The residual is a finite loop (it did NOT unroll per input).
        assert!(prog.blocks.len() <= 6, "dynamic loop should not have unrolled");
        let has_push = prog
            .blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|op| matches!(op, Op::PushOp { .. }));
        assert!(has_push, "expected a residual push");

        for n in [0, 1, 6, 20] {
            let got = vm.run_residual(&prog, n);
            assert_eq!(vm.run_reference(n), got, "diverged at n={n}");
            let expected: Vec<DeepVal> = (0..n).map(|i| DeepVal::Num(i * i)).collect();
            assert_eq!(got, DeepVal::Array(expected));
        }
    }

    /// `filter` over dynamic elements: each push is conditional. The result
    /// array escapes before the branch, the pushes residualize, and the arms
    /// merge into a LINEAR residual (not 2^n paths).
    #[test]
    fn filter_dynamic_elements_is_linear() {
        let ispos = FuncDef {
            name: "ispos",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["x"],
            body: vec![Stmt::Return(bin(Bop::Gt, var(0), num(0)))],
        };
        let filter = FuncDef {
            name: "filter",
            nslots: 4,
            ncaptured: 0,
            nparams: 2,
            slot_names: vec!["xs", "pred", "result", "i"],
            body: vec![
                Stmt::Let(2, arr(vec![])),
                Stmt::Let(3, num(0)),
                Stmt::While(
                    bin(Bop::Lt, var(3), get(var(0), "length")),
                    vec![
                        Stmt::If(
                            call(var(1), vec![index(var(0), var(3))]),
                            vec![Stmt::Push(var(2), index(var(0), var(3)))],
                            vec![],
                        ),
                        Stmt::Set(3, bin(Bop::Add, var(3), num(1))),
                    ],
                ),
                Stmt::Return(var(2)),
            ],
        };
        // main(a) = filter([a, a-1, a+2, a-3], ispos)
        let main = FuncDef {
            name: "main",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["a"],
            body: vec![Stmt::Return(call(
                func(1),
                vec![
                    arr(vec![
                        var(0),
                        bin(Bop::Sub, var(0), num(1)),
                        bin(Bop::Add, var(0), num(2)),
                        bin(Bop::Sub, var(0), num(3)),
                    ]),
                    func(0),
                ],
            ))],
        };
        let vm = Js::new(&[ispos, filter, main]);
        let mut prog = engine::specialize(&vm, vm.start());
        crate::residual::simplify(&mut prog);

        // Linear, not exponential: one block per element decision, merged.
        assert!(
            prog.blocks.len() <= 12,
            "filter path-split instead of merging ({} blocks)",
            prog.blocks.len()
        );

        for a in [-5, 0, 2, 5, 100] {
            let got = vm.run_residual(&prog, a);
            assert_eq!(vm.run_reference(a), got, "filter diverged at a={a}");
            let expected: Vec<DeepVal> = [a, a - 1, a + 2, a - 3]
                .into_iter()
                .filter(|&v| v > 0)
                .map(DeepVal::Num)
                .collect();
            assert_eq!(got, DeepVal::Array(expected));
        }
    }
}
