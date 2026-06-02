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
    Null,
    This,                             // `this` — sound only in a non-inlined (residual) function
    Var(usize),                       // local slot
    Bin(Bop, Box<Expr>, Box<Expr>),
    Object(Vec<(String, Expr)>),      // { k: e, ... }
    Array(Vec<Expr>),
    Get(Box<Expr>, String),           // e.k
    Index(Box<Expr>, Box<Expr>),      // e[i]
    Func(usize),                      // reference a global function as a value
    Closure(usize, Vec<Expr>),        // make closure of function #fid capturing these
    Call(Box<Expr>, Vec<Expr>),       // callee(args)
    New(Box<Expr>, Vec<Expr>),        // new callee(args) — passes through as a residual
    /// An operation the evaluator does not model (e.g. `/`, `%`, `&&`, `~`).
    /// The operands are still specialized; the operation itself passes through
    /// verbatim into the residual. Must be a *pure* operation over *primitive*
    /// operands (a heap reference reaching one is a hard error). `op` is the
    /// JS operator token; arity is encoded by `args.len()` (1 = prefix unary,
    /// 2 = infix, 3 = `?:`).
    Opaque(String, Vec<Expr>),
    /// A free identifier the evaluator has no binding for (e.g. `Math`,
    /// `parseInt`, `console`). It is treated as an opaque runtime global and
    /// passes through into the residual verbatim.
    Global(String),
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
    SetProp(Expr, String, Expr),  // o.k = v
    SetIndex(Expr, Expr, Expr),   // arr[i] = v
    DeleteProp(Expr, String),     // delete o.k
    DeleteIndex(Expr, Expr),      // delete arr[i]
    Push(Expr, Expr),             // arr.push(v)
    Return(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    /// `for (...; cond; update) { body }`. Kept distinct from `While` so
    /// `continue` runs `update` before re-testing `cond` (the init is emitted
    /// before this statement).
    For { cond: Expr, body: Vec<Stmt>, update: Vec<Stmt> },
    /// Skip to the next iteration of the innermost enclosing loop.
    Continue,
    /// `switch (disc) { clauses }`. Real JS semantics: the discriminant is
    /// evaluated once, the first `case` strictly-equal (`===`) to it becomes the
    /// entry point, and execution falls through subsequent clauses until a
    /// `break` or the end. `default` may appear in any position.
    Switch(Expr, Vec<Clause>),
    /// Exit the innermost enclosing `switch` (or `while`).
    Break,
    /// `throw expr` — transfer control to the nearest enclosing `catch`,
    /// unwinding inlined call frames, binding the value to the catch parameter.
    Throw(Expr),
    /// `try { body } catch (e?) { handler }`. Exceptions are modeled as control
    /// flow: a `throw` reached during specialization becomes a jump to the
    /// catch. `finally` is not yet modeled.
    Try { body: Vec<Stmt>, catch_slot: Option<usize>, catch_body: Vec<Stmt> },
    ExprStmt(Expr),
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum Clause {
    Case(Expr, Vec<Stmt>),
    Default(Vec<Stmt>),
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
    PushNull,
    PushThis,
    Load(usize),
    Store(usize),
    Bin(Bop),
    NewObject(Vec<String>),
    NewArray(usize),
    GetProp(String),
    SetPropOp(String),
    SetIndexOp,
    DeletePropOp(String),
    DeleteIndexOp,
    PushArr,
    GetIndex,
    PushFunc(usize),
    MakeClosure(usize, usize), // fid, ncaptures
    Call(usize),               // nargs
    NewOp(usize),              // new callee(args), nargs — pops args then callee
    PushHandler { catch_pc: usize, exc_slot: Option<usize> }, // begin a try region
    PopHandler,                // end a try region (normal completion)
    Throw,                     // pop a value and unwind to the nearest handler
    PushGlobal(String),        // push an opaque runtime global by name
    OpaqueOp { op: String, arity: usize }, // unmodeled pure op over `arity` operands
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
    Null,
    This,
    Var(usize),
    Bin(Bop, Box<RExpr>, Box<RExpr>),
    /// Read element `index` of a residual (escaped) array variable.
    Index(Box<RExpr>, Box<RExpr>),
    /// An unmodeled pure operation passed through verbatim (e.g. `/`, `&&`,
    /// `~`). `op` is the JS operator token; `args.len()` gives the arity
    /// (1 = prefix unary, 2 = infix, 3 = `?:`). The Rust interpreters cannot
    /// evaluate these (that is the point); such programs are validated by Node.
    Opaque(String, Vec<RExpr>),
    /// A free runtime global by name (e.g. `Math`, `parseInt`).
    Global(String),
    /// Property read on a runtime value: `base.key`.
    Get(Box<RExpr>, String),
    /// A call of a runtime (unmodeled) callee. To keep effects single-evaluated
    /// and in program order, this only ever appears as the rhs of an `Op::Eval`;
    /// the call's value is then referenced as that variable.
    Call(Box<RExpr>, Vec<RExpr>),
    /// `new callee(args)` — like `Call`, only ever the rhs of an `Op::Eval`.
    New(Box<RExpr>, Vec<RExpr>),
    /// A reference to a generated residual function with its captured values
    /// bound: renders as `__rf{rfid}.bind(null, caps...)` (or `__rf{rfid}` when
    /// there are no captures).
    FnRef { rfid: usize, caps: Vec<RExpr> },
}

#[derive(Debug)]
pub enum Op {
    Assign { var: usize, expr: RExpr },
    Return(RExpr),
    /// Construct an object into a residual variable (a materialized escape).
    NewObject { dst: usize, fields: Vec<(String, RExpr)> },
    NewArray { dst: usize, elems: Vec<RExpr> },
    /// Append to a residual (runtime) array value (`arr` is any object
    /// expression: an escaped var, a global, a property chain).
    PushOp { arr: RExpr, val: RExpr },
    /// Write `val` to element `index` of a residual array/object value.
    SetIndex { arr: RExpr, index: RExpr, val: RExpr },
    /// Evaluate a possibly-effectful residual expression (an unmodeled call)
    /// once into `dst`, in program order. The value is referenced as `v{dst}`.
    Eval { dst: usize, expr: RExpr },
    /// Write `val` to property `key` of a residual object value.
    SetProp { obj: RExpr, key: String, val: RExpr },
    /// `delete <obj>.key` on a residual object value.
    DeleteProp { obj: RExpr, key: String },
    /// `delete <arr>[index]` on a residual array/object value.
    DeleteIndex { arr: RExpr, index: RExpr },
    /// `throw expr` that escaped all handlers (an uncaught exception on this path).
    Throw(RExpr),
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
    Null,
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
            Abs::Null => RExpr::Null,
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
            Abs::Null => Some(false),
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

/// An active `try`: where to transfer on `throw`, which slot binds the
/// exception, and how far to unwind (the frame depth and that frame's operand
/// stack height when the `try` was entered).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Handler {
    catch_pc: usize,
    exc_slot: Option<usize>,
    frame_depth: usize,
    ostack_depth: usize,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct State {
    frames: Vec<Frame>,
    heap: Vec<HeapObj>,
    pending_joins: Vec<(usize, Vec<usize>)>,
    /// Stack of active `try` handlers (innermost last).
    handlers: Vec<Handler>,
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

/// Threaded compiler state. `ifs`/`heap_mods`/`switch_ends` accumulate across
/// the whole program (global pcs); `next_slot`/`max_slot` are reset per function
/// and `breaks` is a stack of unpatched `break` jumps for enclosing constructs.
#[derive(Default)]
struct CompileAux {
    ifs: Vec<(usize, usize)>,
    heap_mods: Vec<(usize, Vec<usize>)>,
    /// pcs that are the join after a `switch`; marked leaders so the arms of a
    /// dynamic-discriminant switch merge there instead of path-splitting.
    switch_ends: Vec<usize>,
    /// stack of pending `break` jumps, one Vec per enclosing switch/loop.
    breaks: Vec<Vec<usize>>,
    /// stack of pending `continue` jumps, one Vec per enclosing *loop* (not
    /// switch, which `continue` skips past).
    continues: Vec<Vec<usize>>,
    /// `catch` entry pcs; marked leaders so a `throw` transferring there starts a
    /// fresh residual block.
    catch_pcs: Vec<usize>,
    /// next free local slot for switch-discriminant scratch (per function).
    next_slot: usize,
    /// high-water mark of slots used (becomes the function's nslots).
    max_slot: usize,
}

fn compile_stmts(stmts: &[Stmt], code: &mut Vec<Instr>, aux: &mut CompileAux) {
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
            Stmt::SetIndex(a, i, v) => {
                compile_expr(a, code);
                compile_expr(i, code);
                compile_expr(v, code);
                code.push(Instr::SetIndexOp);
            }
            Stmt::DeleteProp(o, k) => {
                compile_expr(o, code);
                code.push(Instr::DeletePropOp(k.clone()));
            }
            Stmt::DeleteIndex(a, i) => {
                compile_expr(a, code);
                compile_expr(i, code);
                code.push(Instr::DeleteIndexOp);
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
            Stmt::Break => {
                let j = code.len();
                code.push(Instr::Jmp(0));
                aux.breaks
                    .last_mut()
                    .expect("`break` outside of a switch or loop")
                    .push(j);
            }
            Stmt::If(c, t, e) => {
                compile_expr(c, code);
                let jz = code.len();
                code.push(Instr::JmpIfFalsy(0));
                compile_stmts(t, code, aux);
                let jmp = code.len();
                code.push(Instr::Jmp(0));
                let else_pc = code.len();
                compile_stmts(e, code, aux);
                let end = code.len();
                patch(&mut code[jz], else_pc);
                patch(&mut code[jmp], end);
                aux.ifs.push((jz, end));
                // Which slots hold arrays/objects mutated in either arm? They
                // must escape *before* the branch so conditional mutations
                // residualize and the arms merge at the join.
                let mut hm = Vec::new();
                collect_heap_mods(t, &mut hm);
                collect_heap_mods(e, &mut hm);
                hm.sort_unstable();
                hm.dedup();
                aux.heap_mods.push((jz, hm));
            }
            Stmt::While(c, body) => {
                let head = code.len();
                compile_expr(c, code);
                let jz = code.len();
                code.push(Instr::JmpIfFalsy(0));
                aux.breaks.push(Vec::new());
                aux.continues.push(Vec::new());
                compile_stmts(body, code, aux);
                code.push(Instr::Jmp(head));
                let end = code.len();
                patch(&mut code[jz], end);
                for b in aux.breaks.pop().unwrap() {
                    patch(&mut code[b], end);
                }
                // `continue` in a while re-tests the condition (the head).
                for c in aux.continues.pop().unwrap() {
                    patch(&mut code[c], head);
                }
            }
            Stmt::For { cond, body, update } => {
                let head = code.len();
                compile_expr(cond, code);
                let jz = code.len();
                code.push(Instr::JmpIfFalsy(0));
                aux.breaks.push(Vec::new());
                aux.continues.push(Vec::new());
                compile_stmts(body, code, aux);
                // `continue` jumps here, so the update still runs before re-test.
                let update_pc = code.len();
                compile_stmts(update, code, aux);
                code.push(Instr::Jmp(head));
                let end = code.len();
                patch(&mut code[jz], end);
                for b in aux.breaks.pop().unwrap() {
                    patch(&mut code[b], end);
                }
                for c in aux.continues.pop().unwrap() {
                    patch(&mut code[c], update_pc);
                }
            }
            Stmt::Continue => {
                let j = code.len();
                code.push(Instr::Jmp(0));
                aux.continues
                    .last_mut()
                    .expect("`continue` outside of a loop")
                    .push(j);
            }
            Stmt::Switch(disc, clauses) => compile_switch(disc, clauses, code, aux),
            Stmt::Throw(e) => {
                compile_expr(e, code);
                code.push(Instr::Throw);
            }
            Stmt::Try { body, catch_slot, catch_body } => {
                let ph = code.len();
                code.push(Instr::PushHandler { catch_pc: 0, exc_slot: *catch_slot });
                compile_stmts(body, code, aux);
                code.push(Instr::PopHandler);
                let jmp = code.len();
                code.push(Instr::Jmp(0)); // normal completion skips the catch
                let catch_pc = code.len();
                if let Instr::PushHandler { catch_pc: cp, .. } = &mut code[ph] {
                    *cp = catch_pc;
                }
                aux.catch_pcs.push(catch_pc);
                compile_stmts(catch_body, code, aux);
                let end = code.len();
                patch(&mut code[jmp], end);
            }
        }
    }
}

fn compile_switch(disc: &Expr, clauses: &[Clause], code: &mut Vec<Instr>, aux: &mut CompileAux) {
    // The discriminant is evaluated once into a fresh scratch slot, so it is not
    // recomputed per case test (JS evaluates it exactly once).
    let sd = aux.next_slot;
    aux.next_slot += 1;
    aux.max_slot = aux.max_slot.max(aux.next_slot);
    compile_expr(disc, code);
    code.push(Instr::Store(sd));

    // Dispatch: test each `case` in order; the first strict-equal match jumps
    // into that clause's body. Bodies are laid out in source order, so reaching
    // the end of one falls through into the next (JS fall-through); a `break`
    // jumps past the whole switch.
    let mut match_jmps: Vec<usize> = Vec::new(); // Jmp-into-body, one per Case (in order)
    let mut dispatch_jzs: Vec<usize> = Vec::new();
    for clause in clauses {
        if let Clause::Case(test, _) = clause {
            code.push(Instr::Load(sd));
            compile_expr(test, code);
            code.push(Instr::Bin(Bop::Eq));
            let jz = code.len();
            code.push(Instr::JmpIfFalsy(0)); // not equal -> next test
            dispatch_jzs.push(jz);
            match_jmps.push(code.len());
            code.push(Instr::Jmp(0)); // equal -> this clause body
            let next = code.len();
            patch(&mut code[jz], next);
        }
    }
    // No case matched: jump to `default` (wherever it is) if present, else past
    // the switch.
    let fallthrough = code.len();
    code.push(Instr::Jmp(0));

    // Clause bodies, in source order.
    aux.breaks.push(Vec::new());
    let mut default_label: Option<usize> = None;
    let mut case_i = 0;
    for clause in clauses {
        let label = code.len();
        match clause {
            Clause::Case(_, body) => {
                patch(&mut code[match_jmps[case_i]], label);
                case_i += 1;
                compile_stmts(body, code, aux);
            }
            Clause::Default(body) => {
                default_label = Some(label);
                compile_stmts(body, code, aux);
            }
        }
    }
    let end = code.len();
    patch(&mut code[fallthrough], default_label.unwrap_or(end));
    for b in aux.breaks.pop().unwrap() {
        patch(&mut code[b], end);
    }

    // The join after the switch must be a leader so the arms of a dynamic
    // discriminant merge there rather than path-splitting.
    aux.switch_ends.push(end);

    // For a dynamic discriminant, treat each dispatch test like an `if` whose
    // join is the switch end: slots/heap mutated in any clause escape before the
    // branch and the arms merge, keeping the residual linear (as with `filter`).
    let mut hm = Vec::new();
    for clause in clauses {
        match clause {
            Clause::Case(_, body) | Clause::Default(body) => collect_heap_mods(body, &mut hm),
        }
    }
    hm.sort_unstable();
    hm.dedup();
    for jz in dispatch_jzs {
        aux.ifs.push((jz, end));
        aux.heap_mods.push((jz, hm.clone()));
    }
}

/// Local slots that are the base of a `push` or property write anywhere in
/// these statements (the heap objects they hold get mutated).
fn collect_heap_mods(stmts: &[Stmt], out: &mut Vec<usize>) {
    for s in stmts {
        match s {
            Stmt::Push(Expr::Var(slot), _)
            | Stmt::SetProp(Expr::Var(slot), _, _)
            | Stmt::SetIndex(Expr::Var(slot), _, _)
            | Stmt::DeleteProp(Expr::Var(slot), _)
            | Stmt::DeleteIndex(Expr::Var(slot), _) => out.push(*slot),
            Stmt::If(_, t, e) => {
                collect_heap_mods(t, out);
                collect_heap_mods(e, out);
            }
            Stmt::While(_, b) => collect_heap_mods(b, out),
            Stmt::For { body, update, .. } => {
                collect_heap_mods(body, out);
                collect_heap_mods(update, out);
            }
            Stmt::Switch(_, clauses) => {
                for clause in clauses {
                    match clause {
                        Clause::Case(_, b) | Clause::Default(b) => collect_heap_mods(b, out),
                    }
                }
            }
            Stmt::Try { body, catch_body, .. } => {
                collect_heap_mods(body, out);
                collect_heap_mods(catch_body, out);
            }
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
        Expr::Null => code.push(Instr::PushNull),
        Expr::This => code.push(Instr::PushThis),
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
        Expr::New(callee, args) => {
            compile_expr(callee, code);
            for a in args {
                compile_expr(a, code);
            }
            code.push(Instr::NewOp(args.len()));
        }
        Expr::Opaque(op, args) => {
            for a in args {
                compile_expr(a, code);
            }
            code.push(Instr::OpaqueOp { op: op.clone(), arity: args.len() });
        }
        Expr::Global(name) => code.push(Instr::PushGlobal(name.clone())),
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
    /// Residual functions emitted for closures that ESCAPE into the residual
    /// (returned, stored, passed to unmodeled code) instead of inlining. Built
    /// during specialization via reentrant `engine::specialize`; rendered as
    /// top-level `function __rf{i}(...)` by the back-end. (Interior mutability:
    /// `step`/`materialize` take `&self`.)
    residual_fns: std::cell::RefCell<Vec<ResidualFn>>,
    /// Memo: source fid -> residual fid (captures are passed as runtime args, so
    /// one residual function per source function).
    fn_memo: std::cell::RefCell<HashMap<usize, usize>>,
}

/// A generated residual function: `function __rf{id}(cap.., param..) { body }`.
pub struct ResidualFn {
    pub ncaptured: usize,
    pub nparams: usize,
    pub body: Program<Op, Cond>,
}

impl Js {
    pub fn new(program: &[FuncDef]) -> Self {
        let mut code = Vec::new();
        let mut entries = vec![0; program.len()];
        let mut nslots = vec![0; program.len()];
        let mut ncaptured = vec![0; program.len()];
        let mut nparams = vec![0; program.len()];
        let mut aux = CompileAux::default();

        for (fid, f) in program.iter().enumerate() {
            entries[fid] = code.len();
            ncaptured[fid] = f.ncaptured;
            nparams[fid] = f.nparams;
            // Scratch slots for switch discriminants are appended after the
            // declared locals; the high-water mark becomes the real slot count.
            aux.next_slot = f.nslots;
            aux.max_slot = f.nslots;
            compile_stmts(&f.body, &mut code, &mut aux);
            nslots[fid] = aux.max_slot;
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
        // A switch join is reached by fall-through (no jump targets it), so mark
        // it a leader explicitly; otherwise dynamic-switch arms would not merge.
        for &e in &aux.switch_ends {
            if e < leaders.len() {
                leaders[e] = true;
            }
        }
        // A `catch` is entered by a (dynamic) `throw` transferring control, not a
        // static jump, so mark catch entries as leaders too.
        for &c in &aux.catch_pcs {
            if c < leaders.len() {
                leaders[c] = true;
            }
        }

        let mut if_join: Vec<Option<(usize, Vec<usize>)>> = vec![None; code.len()];
        for (jz, end) in aux.ifs {
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
        for (jz, mods) in aux.heap_mods {
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
            residual_fns: std::cell::RefCell::new(Vec::new()),
            fn_memo: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Residual functions generated during specialization (for the back-end).
    pub fn residual_fns(&self) -> std::cell::Ref<'_, Vec<ResidualFn>> {
        self.residual_fns.borrow()
    }

    /// Stable residual var for a residual function's i-th captured param / j-th
    /// real param. Disjoint id bands so distinct residual functions don't clash.
    fn residual_cap_var(rfid: usize, i: usize) -> usize {
        residual_cap_var_id(rfid, i)
    }
    fn residual_param_var(rfid: usize, j: usize) -> usize {
        residual_param_var_id(rfid, j)
    }

    /// Get-or-generate the residual function for source `fid`. Captures are
    /// passed as runtime args, so one residual function per source fid. The body
    /// is produced by a reentrant `engine::specialize` with captured/param slots
    /// bound to the residual function's parameter variables.
    fn residual_fn_for(&self, fid: usize) -> usize {
        if let Some(&r) = self.fn_memo.borrow().get(&fid) {
            return r;
        }
        let rfid = self.residual_fns.borrow().len();
        self.fn_memo.borrow_mut().insert(fid, rfid);
        // Reserve the slot so a (mutually) recursive closure can reference itself.
        self.residual_fns.borrow_mut().push(ResidualFn {
            ncaptured: self.ncaptured[fid],
            nparams: self.nparams[fid],
            body: Program { blocks: Vec::new(), entry: BlockId(0) },
        });

        let ncap = self.ncaptured[fid];
        let nparams = self.nparams[fid];
        let mut locals = vec![Abs::Undef; self.nslots[fid]];
        for i in 0..ncap {
            locals[i] = Abs::Dyn(RExpr::Var(Js::residual_cap_var(rfid, i)));
        }
        for j in 0..nparams {
            locals[ncap + j] = Abs::Dyn(RExpr::Var(Js::residual_param_var(rfid, j)));
        }
        let start = State {
            frames: vec![Frame { pc: self.entries[fid], func: fid, locals, ostack: Vec::new() }],
            heap: Vec::new(),
            pending_joins: Vec::new(),
            handlers: Vec::new(),
        };
        let body = crate::engine::specialize(self, start);
        self.residual_fns.borrow_mut()[rfid].body = body;
        rfid
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
            handlers: Vec::new(),
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
            // Strict equality involving statically-known null/undefined/other
            // primitives: decide by type+value identity.
            (_, _) if matches!(op, Bop::Eq | Bop::Ne) => {
                let eq = static_strict_eq(a, b);
                Bool(if op == Bop::Eq { eq } else { !eq })
            }
            _ => panic!("unsupported operand types for {:?}: {:?} {:?}", op, a, b),
        }
    }
}

/// JS strict equality (`===`) over statically-known values: equal iff same type
/// and same value (object identity for heap refs).
fn static_strict_eq(a: &Abs, b: &Abs) -> bool {
    match (a, b) {
        (Abs::Null, Abs::Null) | (Abs::Undef, Abs::Undef) => true,
        (Abs::Num(x), Abs::Num(y)) => x == y,
        (Abs::Str(x), Abs::Str(y)) => x == y,
        (Abs::Bool(x), Abs::Bool(y)) => x == y,
        (Abs::Ref(x), Abs::Ref(y)) => x == y,
        _ => false,
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
            Instr::PushNull => self.push(s, Abs::Null),
            Instr::PushThis => self.push(s, Abs::Dyn(RExpr::This)),
            Instr::PushGlobal(name) => self.push(s, Abs::Dyn(RExpr::Global(name.clone()))),
            Instr::PushHandler { catch_pc, exc_slot } => {
                let frame_depth = s.frames.len();
                let ostack_depth = s.top().ostack.len();
                s.handlers.push(Handler {
                    catch_pc: *catch_pc,
                    exc_slot: *exc_slot,
                    frame_depth,
                    ostack_depth,
                });
                self.advance(s)
            }
            Instr::PopHandler => {
                s.handlers.pop();
                self.advance(s)
            }
            Instr::Throw => {
                let v = s.top_mut().ostack.pop().unwrap();
                match s.handlers.pop() {
                    // Transfer to the nearest handler: unwind inlined frames,
                    // restore that frame's operand stack, bind the exception, and
                    // jump to the catch. This becomes ordinary control flow.
                    Some(h) => {
                        s.frames.truncate(h.frame_depth);
                        let f = s.frames.last_mut().unwrap();
                        f.ostack.truncate(h.ostack_depth);
                        if let Some(slot) = h.exc_slot {
                            f.locals[slot] = v;
                        }
                        f.pc = h.catch_pc;
                        Step::Continue
                    }
                    // No handler on this path: the exception is uncaught.
                    None => {
                        let heap = s.heap.clone();
                        let r = self.materialize_value(&v, &heap, out);
                        out.push(Op::Throw(r));
                        Step::Halt
                    }
                }
            }
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
                    // A write to a dynamic object residualizes (the value, if it
                    // is itself an object, escapes too).
                    Abs::Dyn(obj_r) => {
                        forbid_residual_in_try(s, "a property write");
                        let mut escaped = std::collections::HashSet::new();
                        let val = self.operand_rexpr(s, &v, &mut escaped, out);
                        out.push(Op::SetProp { obj: obj_r, key: k.clone(), val });
                    }
                    _ => panic!("SetProp on a non-object value (unsupported)"),
                }
                self.advance(s)
            }
            Instr::SetIndexOp => {
                let v = s.top_mut().ostack.pop().unwrap();
                let i = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                match a {
                    // Static array with a static index: mutate the abstract heap
                    // in place (the array stays scalar-replaced). Grow with
                    // `undefined` holes if the write is past the current end.
                    Abs::Ref(addr) => match (&mut s.heap[addr], &i) {
                        (HeapObj::Array(elems), Abs::Num(n)) => {
                            let n = *n as usize;
                            if n >= elems.len() {
                                elems.resize(n + 1, Abs::Undef);
                            }
                            elems[n] = v;
                        }
                        _ => panic!("indexed write on non-array or dynamic index of a static array"),
                    },
                    // Dynamic array/object: residualize the indexed write.
                    Abs::Dyn(arr_r) => {
                        forbid_residual_in_try(s, "an indexed write");
                        let r = self.materialize_value(&v, &s.heap.clone(), out);
                        out.push(Op::SetIndex { arr: arr_r, index: i.to_rexpr(), val: r });
                    }
                    _ => panic!("indexed write on a non-array value"),
                }
                self.advance(s)
            }
            Instr::DeletePropOp(k) => {
                let o = s.top_mut().ostack.pop().unwrap();
                match o {
                    // Static object: drop the field from the abstract heap.
                    Abs::Ref(addr) => {
                        if let HeapObj::Object(fields) = &mut s.heap[addr] {
                            fields.retain(|(fk, _)| fk != k);
                        } else {
                            panic!("delete of a property on a non-object");
                        }
                    }
                    // Dynamic object: residualize the delete.
                    Abs::Dyn(obj_r) => {
                        forbid_residual_in_try(s, "a property delete");
                        out.push(Op::DeleteProp { obj: obj_r, key: k.clone() });
                    }
                    _ => panic!("delete of a property on a non-object value"),
                }
                self.advance(s)
            }
            Instr::DeleteIndexOp => {
                let i = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                match a {
                    // Static array/object: a deleted element becomes a hole
                    // (reads as `undefined`), matching JS `delete arr[i]`.
                    Abs::Ref(addr) => match (&mut s.heap[addr], &i) {
                        (HeapObj::Array(elems), Abs::Num(n)) => {
                            let n = *n as usize;
                            if n < elems.len() {
                                elems[n] = Abs::Undef;
                            }
                        }
                        (HeapObj::Object(fields), Abs::Str(k)) => {
                            fields.retain(|(fk, _)| fk != k);
                        }
                        _ => panic!("delete of a dynamic index on a static container"),
                    },
                    Abs::Dyn(arr_r) => {
                        forbid_residual_in_try(s, "an indexed delete");
                        out.push(Op::DeleteIndex { arr: arr_r, index: i.to_rexpr() });
                    }
                    _ => panic!("delete of an index on a non-container value"),
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
                    // Dynamic array: residualize the push.
                    Abs::Dyn(arr_r) => {
                        forbid_residual_in_try(s, "an array push");
                        let r = self.materialize_value(&v, &s.heap.clone(), out);
                        out.push(Op::PushOp { arr: arr_r, val: r });
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
            Instr::Call(nargs) => self.do_call(s, *nargs, pc, out),
            Instr::NewOp(nargs) => {
                // `new callee(args)` passes through: bind once to a temp (it is
                // effectful and yields a fresh runtime object), referenced as a
                // dynamic value thereafter. Object args escape.
                let base = s.top().ostack.len() - *nargs;
                let args: Vec<Abs> = s.top_mut().ostack.split_off(base);
                let callee = s.top_mut().ostack.pop().unwrap();
                forbid_residual_in_try(s, "a `new` expression");
                let callee_r = match callee {
                    Abs::Dyn(r) => r,
                    // `new` of a known function/closure: it becomes a residual
                    // function (constructors use `this`, so they must not inline).
                    Abs::Ref(addr) => match &s.heap[addr] {
                        HeapObj::Closure { fid, captured } => {
                            let caps: Vec<RExpr> = captured
                                .clone()
                                .iter()
                                .map(|c| self.materialize_value(c, &s.heap.clone(), out))
                                .collect();
                            let rfid = self.residual_fn_for(*fid);
                            RExpr::FnRef { rfid, caps }
                        }
                        _ => panic!("`new` of a non-function object"),
                    },
                    other => panic!("`new` of a non-constructor value: {other:?}"),
                };
                let mut escaped = std::collections::HashSet::new();
                let mut arg_rs = Vec::with_capacity(args.len());
                for a in args {
                    arg_rs.push(self.operand_rexpr(s, &a, &mut escaped, out));
                }
                let dst = Js::opaque_call_var(pc);
                out.push(Op::Eval { dst, expr: RExpr::New(Box::new(callee_r), arg_rs) });
                s.top_mut().pc += 1;
                s.top_mut().ostack.push(Abs::Dyn(RExpr::Var(dst)));
                Step::Continue
            }
            Instr::OpaqueOp { op, arity } => {
                let base = s.top().ostack.len() - *arity;
                let operands: Vec<Abs> = s.top_mut().ostack.split_off(base);
                // The operation is unmodeled, so the result is always dynamic. A
                // heap object operand escapes (it could be mutated by the op).
                let mut escaped = std::collections::HashSet::new();
                let mut args = Vec::with_capacity(operands.len());
                for a in operands {
                    args.push(self.operand_rexpr(s, &a, &mut escaped, out));
                }
                self.push(s, Abs::Dyn(RExpr::Opaque(op.clone(), args)))
            }
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
                // A `return` out of a `try` discards that frame's handlers.
                let depth = s.frames.len();
                s.handlers.retain(|h| h.frame_depth < depth);
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
            // A property read on a dynamic value (a runtime global, a string,
            // an escaped object) passes through as an opaque `base.key` read.
            Abs::Dyn(r) => Abs::Dyn(RExpr::Get(Box::new(r.clone()), k.to_string())),
            _ => panic!("property access on {o:?} (unsupported)"),
        }
    }

    fn get_index(&self, s: &State, a: &Abs, i: &Abs) -> Abs {
        let addr = match a {
            Abs::Ref(addr) => *addr,
            // A dynamic (escaped) array still supports element reads: the read
            // residualizes to an indexed load. The element value is a runtime
            // value, so the result is dynamic.
            Abs::Dyn(r) => {
                return Abs::Dyn(RExpr::Index(Box::new(r.clone()), Box::new(i.to_rexpr())));
            }
            _ => panic!("index on a non-array value (unsupported)"),
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
            // A static array indexed by a dynamic value cannot pick a static
            // element; that case is unsupported (the array would need to escape).
            (_, Abs::Dyn(_)) => panic!("dynamic index into a static array (unsupported: must be static)"),
            _ => panic!("bad index/key types"),
        }
    }

    fn do_call(&self, s: &mut State, nargs: usize, pc: usize, out: &mut Vec<Op>) -> Step<Js> {
        let base = s.top().ostack.len() - nargs;
        let args: Vec<Abs> = s.top_mut().ostack.split_off(base);
        let callee = s.top_mut().ostack.pop().unwrap();

        let (fid, captured) = match callee {
            Abs::Ref(addr) => match &s.heap[addr] {
                HeapObj::Closure { fid, captured } => (*fid, captured.clone()),
                _ => panic!("call of a non-function"),
            },
            // A dynamic (unmodeled) callee: pass the call through. It may be
            // effectful, so bind it to a temp once, in program order, and use
            // the temp downstream (never duplicating the call). A heap object
            // argument escapes (the callee may read or mutate it).
            Abs::Dyn(callee_r) => {
                forbid_residual_in_try(s, "an unmodeled call");
                let mut escaped = std::collections::HashSet::new();
                let mut arg_rs = Vec::with_capacity(args.len());
                for a in args {
                    arg_rs.push(self.operand_rexpr(s, &a, &mut escaped, out));
                }
                let dst = Js::opaque_call_var(pc);
                out.push(Op::Eval { dst, expr: RExpr::Call(Box::new(callee_r), arg_rs) });
                s.top_mut().pc += 1;
                s.top_mut().ostack.push(Abs::Dyn(RExpr::Var(dst)));
                return Step::Continue;
            }
            _ => panic!("call of a non-callable value"),
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
                // A closure that escapes into the residual becomes a generated
                // residual function, referenced with its captured values bound.
                HeapObj::Closure { fid, captured } => {
                    let caps: Vec<RExpr> =
                        captured.iter().map(|c| self.materialize_value(c, heap, out)).collect();
                    let rfid = self.residual_fn_for(*fid);
                    out.push(Op::Assign { var: dst, expr: RExpr::FnRef { rfid, caps } });
                }
            },
            _ => out.push(Op::Assign { var: dst, expr: v.to_rexpr() }),
        }
    }

    fn escape_var(addr: usize) -> usize {
        // Disjoint from the input var (0) and the loop/if materialization band.
        100_000 + addr
    }

    /// Escape a heap object into the residual because it flows into unmodeled
    /// code (an opaque operator or call). Construction ops for the whole graph
    /// reachable from `addr` are emitted, and every reference to those objects
    /// throughout the state is replaced with the residual variable, so any later
    /// read or write of them residualizes (the unmodeled code may have mutated
    /// them). Returns the residual variable for `addr`. `escaped` tracks objects
    /// already escaped during this step so an aliased object passed twice is
    /// built only once. Closures cannot be residual values (hard error).
    fn escape(
        &self,
        s: &mut State,
        addr: usize,
        escaped: &mut std::collections::HashSet<usize>,
        out: &mut Vec<Op>,
    ) -> RExpr {
        forbid_residual_in_try(s, "an object escaping into unmodeled code");
        if escaped.contains(&addr) {
            return RExpr::Var(Js::escape_var(addr));
        }
        let heap = s.heap.clone();
        let root = self.materialize_value(&Abs::Ref(addr), &heap, out);
        let mut reach = std::collections::HashSet::new();
        Js::reachable(&s.heap, addr, &mut reach);
        for f in &mut s.frames {
            for l in &mut f.locals {
                invalidate_abs(l, &reach);
            }
            for o in &mut f.ostack {
                invalidate_abs(o, &reach);
            }
        }
        for h in &mut s.heap {
            match h {
                HeapObj::Object(fs) => fs.iter_mut().for_each(|(_, v)| invalidate_abs(v, &reach)),
                HeapObj::Array(es) => es.iter_mut().for_each(|v| invalidate_abs(v, &reach)),
                HeapObj::Closure { captured, .. } => {
                    captured.iter_mut().for_each(|v| invalidate_abs(v, &reach))
                }
            }
        }
        escaped.extend(reach);
        root
    }

    /// Heap addresses transitively reachable from `addr` (terminates on shared
    /// or cyclic graphs via the visited set).
    fn reachable(heap: &[HeapObj], addr: usize, out: &mut std::collections::HashSet<usize>) {
        if !out.insert(addr) {
            return;
        }
        let kids: Vec<usize> = match &heap[addr] {
            HeapObj::Object(fs) => fs.iter().filter_map(|(_, v)| as_ref_addr(v)).collect(),
            HeapObj::Array(es) => es.iter().filter_map(as_ref_addr).collect(),
            HeapObj::Closure { captured, .. } => captured.iter().filter_map(as_ref_addr).collect(),
        };
        for c in kids {
            Js::reachable(heap, c, out);
        }
    }

    /// Convert an operand to a residual expression, escaping it first if it is a
    /// heap object (so it can flow into unmodeled code).
    fn operand_rexpr(
        &self,
        s: &mut State,
        a: &Abs,
        escaped: &mut std::collections::HashSet<usize>,
        out: &mut Vec<Op>,
    ) -> RExpr {
        match a {
            Abs::Ref(addr) => self.escape(s, *addr, escaped, out),
            _ => a.to_rexpr(),
        }
    }

    /// Stable residual variable for the result of an unmodeled call at this
    /// bytecode pc. Keyed by pc so a call in a loop reuses one temp (reassigned
    /// each iteration) and specialization converges. Disjoint from the other
    /// id bands.
    fn opaque_call_var(pc: usize) -> usize {
        500_000 + pc
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
                Instr::PushNull => probe.top_mut().ostack.push(Abs::Null),
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
        if !slots.is_empty() {
            forbid_residual_in_try(ns, "a value materializing (a dynamic loop/branch)");
        }
        let fi = ns.frames.len() - 1;
        let func = ns.frames[fi].func;
        let heap = ns.heap.clone();
        // The slots are updated *simultaneously* (loop-carried / join merge), so a
        // primitive RHS that reads another slot's stable var must see its OLD
        // value. Emitting `vA = ...; vB = ...vA...` sequentially would let vB read
        // the already-updated vA, so primitive updates go through a temp first:
        // compute every new value (reading old stable vars), then copy to the
        // stable vars. Heap (Ref) slots are materialized directly; their dst ids
        // are disjoint and their constructions read the still-old stable vars.
        let mut copies: Vec<(usize, usize)> = Vec::new(); // (stable_id, temp_id)
        for &slot in slots {
            let cur = ns.frames[fi].locals[slot].clone();
            let id = self.stable_id(func, slot);
            if cur != Abs::Dyn(RExpr::Var(id)) {
                if matches!(cur, Abs::Ref(_)) {
                    self.materialize_into(id, &cur, &heap, out);
                } else {
                    let tmp = self.loop_tmp(func, slot);
                    out.push(Op::Assign { var: tmp, expr: cur.to_rexpr() });
                    copies.push((id, tmp));
                }
            }
            ns.frames[fi].locals[slot] = Abs::Dyn(RExpr::Var(id));
        }
        for (id, tmp) in copies {
            out.push(Op::Assign { var: id, expr: RExpr::Var(tmp) });
        }
    }

    fn stable_id(&self, func: usize, slot: usize) -> usize {
        // Reserve a deterministic band of ids for loop/if materialization,
        // disjoint from input vars (which are small).
        1_000 + func * 64 + slot
    }

    /// Scratch var for the parallel-update snapshot in `materialize`. Disjoint
    /// from the other id bands.
    fn loop_tmp(&self, func: usize, slot: usize) -> usize {
        700_000 + func * 256 + slot
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
    Null,
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
            (_, _) if matches!(op, Bop::Eq | Bop::Ne) => {
                let eq = match (a, b) {
                    (JsVal::Null, JsVal::Null) | (JsVal::Undef, JsVal::Undef) => true,
                    (JsVal::Num(x), JsVal::Num(y)) => x == y,
                    (JsVal::Str(x), JsVal::Str(y)) => x == y,
                    (JsVal::Bool(x), JsVal::Bool(y)) => x == y,
                    (JsVal::Ref(x), JsVal::Ref(y)) => x == y,
                    _ => false,
                };
                JsVal::Bool(if op == Bop::Eq { eq } else { !eq })
            }
            _ => panic!("bad operand types in reference interpreter"),
        }
    }

    fn ctruthy(v: &JsVal) -> bool {
        match v {
            JsVal::Num(n) => *n != 0,
            JsVal::Str(s) => !s.is_empty(),
            JsVal::Bool(b) => *b,
            JsVal::Undef => false,
            JsVal::Null => false,
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
                Instr::PushNull => ostack.push(JsVal::Null),
                Instr::PushThis => panic!(
                    "the Rust reference interpreter cannot evaluate `this`; validate \
                     with the Node oracle"
                ),
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
                Instr::SetIndexOp => {
                    let v = ostack.pop().unwrap();
                    let i = ostack.pop().unwrap();
                    let a = ostack.pop().unwrap();
                    match (a, i) {
                        (JsVal::Ref(ad), JsVal::Num(n)) => {
                            if let CHeap::Array(e) = &mut heap[ad] {
                                let n = n as usize;
                                if n >= e.len() {
                                    e.resize(n + 1, JsVal::Undef);
                                }
                                e[n] = v;
                            } else {
                                panic!("indexed write on non-array");
                            }
                        }
                        _ => panic!("indexed write needs a ref array and numeric index"),
                    }
                }
                Instr::DeletePropOp(k) => {
                    let o = ostack.pop().unwrap();
                    if let JsVal::Ref(ad) = o {
                        if let CHeap::Object(f) = &mut heap[ad] {
                            f.retain(|(fk, _)| fk != k);
                        } else {
                            panic!("delete property on non-object");
                        }
                    } else {
                        panic!("delete property on non-ref");
                    }
                }
                Instr::DeleteIndexOp => {
                    let i = ostack.pop().unwrap();
                    let a = ostack.pop().unwrap();
                    match (a, i) {
                        (JsVal::Ref(ad), JsVal::Num(n)) => match &mut heap[ad] {
                            CHeap::Array(e) => {
                                let n = n as usize;
                                if n < e.len() {
                                    e[n] = JsVal::Undef;
                                }
                            }
                            _ => panic!("delete index on non-array"),
                        },
                        (JsVal::Ref(ad), JsVal::Str(k)) => {
                            if let CHeap::Object(f) = &mut heap[ad] {
                                f.retain(|(fk, _)| *fk != k);
                            }
                        }
                        _ => panic!("delete index needs a ref and num/str index"),
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
                Instr::OpaqueOp { op, .. } => panic!(
                    "the Rust reference interpreter cannot evaluate the unmodeled \
                     operator `{op}`; validate pass-through programs with the Node oracle"
                ),
                Instr::PushGlobal(name) => panic!(
                    "the Rust reference interpreter cannot resolve the runtime global \
                     `{name}`; validate pass-through programs with the Node oracle"
                ),
                Instr::PushHandler { .. } | Instr::PopHandler | Instr::Throw => panic!(
                    "the Rust reference interpreter does not model exceptions; validate \
                     try/catch programs with the Node oracle"
                ),
                Instr::NewOp(_) => panic!(
                    "the Rust reference interpreter cannot evaluate `new`; validate \
                     pass-through programs with the Node oracle"
                ),
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
                        let v = eval_rexpr(expr, &store, &heap);
                        store.insert(*var, v);
                    }
                    Op::Return(e) => ret = eval_rexpr(e, &store, &heap),
                    Op::NewObject { dst, fields } => {
                        let f = fields
                            .iter()
                            .map(|(k, e)| (k.clone(), eval_rexpr(e, &store, &heap)))
                            .collect();
                        let addr = heap.len();
                        heap.push(CHeap::Object(f));
                        store.insert(*dst, JsVal::Ref(addr));
                    }
                    Op::NewArray { dst, elems } => {
                        let e = elems.iter().map(|e| eval_rexpr(e, &store, &heap)).collect();
                        let addr = heap.len();
                        heap.push(CHeap::Array(e));
                        store.insert(*dst, JsVal::Ref(addr));
                    }
                    Op::PushOp { arr, val } => {
                        let v = eval_rexpr(val, &store, &heap);
                        match eval_rexpr(arr, &store, &heap) {
                            JsVal::Ref(addr) => match &mut heap[addr] {
                                CHeap::Array(e) => e.push(v),
                                _ => panic!("push on non-array"),
                            },
                            _ => panic!("push on a non-array value"),
                        }
                    }
                    Op::SetIndex { arr, index, val } => {
                        let i = eval_rexpr(index, &store, &heap);
                        let v = eval_rexpr(val, &store, &heap);
                        let n = match i {
                            JsVal::Num(n) => n as usize,
                            _ => panic!("indexed write needs a numeric index"),
                        };
                        match eval_rexpr(arr, &store, &heap) {
                            JsVal::Ref(addr) => match &mut heap[addr] {
                                CHeap::Array(e) => {
                                    if n >= e.len() {
                                        e.resize(n + 1, JsVal::Undef);
                                    }
                                    e[n] = v;
                                }
                                _ => panic!("indexed write on non-array"),
                            },
                            _ => panic!("indexed write on a non-array value"),
                        }
                    }
                    Op::Eval { .. } => panic!(
                        "the Rust residual interpreter cannot evaluate an unmodeled \
                         call; validate pass-through programs with the Node oracle"
                    ),
                    Op::SetProp { obj, key, val } => {
                        let v = eval_rexpr(val, &store, &heap);
                        match eval_rexpr(obj, &store, &heap) {
                            JsVal::Ref(addr) => match &mut heap[addr] {
                                CHeap::Object(fields) => {
                                    if let Some(slot) = fields.iter_mut().find(|(k, _)| k == key) {
                                        slot.1 = v;
                                    } else {
                                        fields.push((key.clone(), v));
                                    }
                                }
                                _ => panic!("property write on non-object"),
                            },
                            _ => panic!("property write on a non-object value"),
                        }
                    }
                    Op::DeleteProp { obj, key } => match eval_rexpr(obj, &store, &heap) {
                        JsVal::Ref(addr) => {
                            if let CHeap::Object(fields) = &mut heap[addr] {
                                fields.retain(|(k, _)| k != key);
                            }
                        }
                        _ => panic!("property delete on a non-object value"),
                    },
                    Op::DeleteIndex { arr, index } => {
                        let i = eval_rexpr(index, &store, &heap);
                        match (eval_rexpr(arr, &store, &heap), i) {
                            (JsVal::Ref(addr), JsVal::Num(n)) => match &mut heap[addr] {
                                CHeap::Array(e) => {
                                    let n = n as usize;
                                    if n < e.len() {
                                        e[n] = JsVal::Undef;
                                    }
                                }
                                _ => panic!("indexed delete on non-array"),
                            },
                            _ => panic!("indexed delete needs a ref array and numeric index"),
                        }
                    }
                    Op::Throw(e) => {
                        let v = eval_rexpr(e, &store, &heap);
                        panic!("uncaught residual throw: {v:?}");
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
                    bid = if !Js::ctruthy(&eval_rexpr(e, &store, &heap)) {
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
                        writeln!(s, "    {}.push({})", fmt_rexpr(arr), fmt_rexpr(val)).unwrap()
                    }
                    Op::SetIndex { arr, index, val } => writeln!(
                        s,
                        "    {}[{}] := {}",
                        fmt_rexpr(arr),
                        fmt_rexpr(index),
                        fmt_rexpr(val)
                    )
                    .unwrap(),
                    Op::Eval { dst, expr } => {
                        writeln!(s, "    v{dst} := {}", fmt_rexpr(expr)).unwrap()
                    }
                    Op::SetProp { obj, key, val } => {
                        writeln!(s, "    {}.{key} := {}", fmt_rexpr(obj), fmt_rexpr(val)).unwrap()
                    }
                    Op::DeleteProp { obj, key } => {
                        writeln!(s, "    delete {}.{key}", fmt_rexpr(obj)).unwrap()
                    }
                    Op::DeleteIndex { arr, index } => {
                        writeln!(s, "    delete {}[{}]", fmt_rexpr(arr), fmt_rexpr(index)).unwrap()
                    }
                    Op::Throw(e) => writeln!(s, "    throw {}", fmt_rexpr(e)).unwrap(),
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

/// Guard the soundness boundary of the v1 exception model: a `try` body is only
/// handled when it fully specializes (modeled `throw` becomes control flow). A
/// residual operation that could throw at runtime would escape the goto-based
/// catch, so emitting one while a handler is active is a hard error for now.
fn forbid_residual_in_try(s: &State, what: &str) {
    if !s.handlers.is_empty() {
        panic!(
            "{what} inside an active try/catch is not yet supported; only a try \
             body that fully specializes (with modeled `throw`) is handled"
        );
    }
}

fn as_ref_addr(v: &Abs) -> Option<usize> {
    if let Abs::Ref(a) = v {
        Some(*a)
    } else {
        None
    }
}

/// Replace a reference to an escaped object with its residual variable.
fn invalidate_abs(v: &mut Abs, escaped: &std::collections::HashSet<usize>) {
    if let Abs::Ref(a) = v {
        if escaped.contains(a) {
            *v = Abs::Dyn(RExpr::Var(Js::escape_var(*a)));
        }
    }
}

fn eval_rexpr(e: &RExpr, store: &HashMap<usize, JsVal>, heap: &[CHeap]) -> JsVal {
    match e {
        RExpr::Num(n) => JsVal::Num(*n),
        RExpr::Str(s) => JsVal::Str(s.clone()),
        RExpr::Bool(b) => JsVal::Bool(*b),
        RExpr::Undef => JsVal::Undef,
        RExpr::Null => JsVal::Null,
        RExpr::This => panic!(
            "the Rust residual interpreter cannot evaluate `this`; validate with the Node oracle"
        ),
        RExpr::Var(v) => store.get(v).cloned().unwrap_or(JsVal::Undef),
        RExpr::Bin(op, a, b) => {
            Js::cbin(*op, &eval_rexpr(a, store, heap), &eval_rexpr(b, store, heap))
        }
        RExpr::Index(a, i) => {
            let arr = eval_rexpr(a, store, heap);
            let idx = eval_rexpr(i, store, heap);
            match (arr, idx) {
                (JsVal::Ref(addr), JsVal::Num(n)) => match &heap[addr] {
                    CHeap::Array(e) => e.get(n as usize).cloned().unwrap_or(JsVal::Undef),
                    _ => panic!("indexed read on non-array"),
                },
                _ => panic!("indexed read needs a ref array and numeric index"),
            }
        }
        RExpr::Opaque(op, _) => panic!(
            "the Rust residual interpreter cannot evaluate the unmodeled operator \
             `{op}`; validate pass-through programs with the Node oracle"
        ),
        RExpr::Global(name) => panic!(
            "the Rust residual interpreter cannot resolve the runtime global `{name}`; \
             validate pass-through programs with the Node oracle"
        ),
        RExpr::Get(..) | RExpr::Call(..) | RExpr::New(..) | RExpr::FnRef { .. } => panic!(
            "the Rust residual interpreter cannot evaluate an unmodeled property \
             read / call / new / residual function; validate with the Node oracle"
        ),
    }
}

/// Render an opaque (pass-through) operator and its already-rendered operands as
/// a parenthesized JS expression. Shared by the IR dump and the JS code emitter.
pub fn render_opaque(op: &str, parts: &[String]) -> String {
    match parts {
        [a] if op.chars().next().is_some_and(|c| c.is_alphabetic()) => format!("({op} {a})"),
        [a] => format!("({op}{a})"),
        [a, b] => format!("({a} {op} {b})"),
        [a, b, c] if op == "?:" => format!("({a} ? {b} : {c})"),
        _ => panic!("opaque operator `{op}` with unexpected arity {}", parts.len()),
    }
}

/// True for residual nodes usable as a member base / call callee without parens
/// (a reference-like chain such as `Math`, `v0`, `a.b`, `f()`).
pub fn rexpr_is_ref_like(e: &RExpr) -> bool {
    matches!(
        e,
        RExpr::Global(_) | RExpr::Var(_) | RExpr::Get(..) | RExpr::Call(..) | RExpr::This
    )
}

/// Render `base.key`, parenthesizing the base only when needed.
pub fn render_get(base: &RExpr, base_str: String, key: &str) -> String {
    if rexpr_is_ref_like(base) {
        format!("{base_str}.{key}")
    } else {
        format!("({base_str}).{key}")
    }
}

/// Render `callee(args)`, parenthesizing the callee only when needed.
pub fn render_call(callee: &RExpr, callee_str: String, args: &[String]) -> String {
    let c = if rexpr_is_ref_like(callee) { callee_str } else { format!("({callee_str})") };
    format!("{c}({})", args.join(", "))
}

/// Render `new callee(args)`, parenthesizing the callee only when needed.
pub fn render_new(callee: &RExpr, callee_str: String, args: &[String]) -> String {
    let c = if rexpr_is_ref_like(callee) { callee_str } else { format!("({callee_str})") };
    format!("new {c}({})", args.join(", "))
}

fn fmt_rexpr(e: &RExpr) -> String {
    match e {
        RExpr::Num(n) => n.to_string(),
        RExpr::Str(s) => format!("{s:?}"),
        RExpr::Bool(b) => b.to_string(),
        RExpr::Undef => "undefined".to_string(),
        RExpr::Null => "null".to_string(),
        RExpr::This => "this".to_string(),
        RExpr::Var(v) => format!("v{v}"),
        RExpr::Bin(op, a, b) => format!("({} {} {})", fmt_rexpr(a), op.sym(), fmt_rexpr(b)),
        RExpr::Index(a, i) => format!("{}[{}]", fmt_rexpr(a), fmt_rexpr(i)),
        RExpr::Opaque(op, args) => {
            let parts: Vec<String> = args.iter().map(fmt_rexpr).collect();
            render_opaque(op, &parts)
        }
        RExpr::Global(name) => name.clone(),
        RExpr::Get(o, k) => render_get(o, fmt_rexpr(o), k),
        RExpr::Call(callee, args) => {
            let a: Vec<String> = args.iter().map(fmt_rexpr).collect();
            render_call(callee, fmt_rexpr(callee), &a)
        }
        RExpr::New(callee, args) => {
            let a: Vec<String> = args.iter().map(fmt_rexpr).collect();
            render_new(callee, fmt_rexpr(callee), &a)
        }
        RExpr::FnRef { rfid, caps } => {
            let cs: Vec<String> = caps.iter().map(fmt_rexpr).collect();
            render_fnref(*rfid, &cs)
        }
    }
}

/// Render a residual function reference with its captures bound:
/// `__rf{rfid}` (no captures) or `__rf{rfid}.bind(null, caps...)`.
pub fn render_fnref(rfid: usize, caps: &[String]) -> String {
    if caps.is_empty() {
        format!("__rf{rfid}")
    } else {
        format!("__rf{rfid}.bind(null, {})", caps.join(", "))
    }
}

/// The residual variable name for a residual function's i-th capture param.
pub fn residual_cap_var_id(rfid: usize, i: usize) -> usize {
    900_000 + rfid * 64 + i
}
/// The residual variable name for a residual function's j-th real param.
pub fn residual_param_var_id(rfid: usize, j: usize) -> usize {
    800_000 + rfid * 64 + j
}

/// A fully-expanded value, for structural comparison of objects/arrays returned
/// by the reference interpreter and the residual (whose heap addresses differ).
#[derive(Clone, Debug, PartialEq)]
pub enum DeepVal {
    Num(i64),
    Str(String),
    Bool(bool),
    Undef,
    Null,
    Object(Vec<(String, DeepVal)>),
    Array(Vec<DeepVal>),
    Closure(usize),
}

fn deep(v: &JsVal, heap: &[CHeap]) -> DeepVal {
    match v {
        JsVal::Num(n) => DeepVal::Num(*n),
        JsVal::Str(s) => DeepVal::Str(s.clone()),
        JsVal::Bool(b) => DeepVal::Bool(*b),
        JsVal::Null => DeepVal::Null,
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

    /// The Futamura projection on JS, written with `switch`: the same expression
    /// interpreter as `interpreter_specializes_away`, but dispatching on
    /// `node.op` with a `switch` instead of a nested `if`-chain. Because the
    /// discriminant is static (it comes from the static AST), the entire switch
    /// dispatch folds away, leaving the same straight-line residual.
    #[test]
    fn switch_interpreter_specializes_away() {
        let eval = FuncDef {
            name: "eval",
            nslots: 2,
            ncaptured: 0,
            nparams: 2,
            slot_names: vec!["node", "env"],
            body: vec![Stmt::Switch(
                get(var(0), "op"),
                vec![
                    Clause::Case(str_("lit"), vec![Stmt::Return(get(var(0), "val"))]),
                    Clause::Case(
                        str_("var"),
                        vec![Stmt::Return(index(var(1), get(var(0), "name")))],
                    ),
                    Clause::Case(
                        str_("add"),
                        vec![Stmt::Return(bin(
                            Bop::Add,
                            call(func(0), vec![get(var(0), "l"), var(1)]),
                            call(func(0), vec![get(var(0), "r"), var(1)]),
                        ))],
                    ),
                    Clause::Case(
                        str_("mul"),
                        vec![Stmt::Return(bin(
                            Bop::Mul,
                            call(func(0), vec![get(var(0), "l"), var(1)]),
                            call(func(0), vec![get(var(0), "r"), var(1)]),
                        ))],
                    ),
                    Clause::Default(vec![Stmt::Return(num(0))]),
                ],
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
            body: vec![Stmt::Return(call(func(0), vec![ast, obj(vec![("x", var(0))])]))],
        };
        let program = [eval, main];
        let vm = Js::new(&program);
        let mut prog = engine::specialize(&vm, vm.start());
        crate::residual::simplify(&mut prog);

        assert_eq!(
            prog.blocks.len(),
            1,
            "switch-based interpreter did not fully specialize away"
        );
        for x in [-3, 0, 1, 7, 42] {
            assert_eq!(
                vm.run_reference(x),
                vm.run_residual(&prog, x),
                "residual diverged from reference at x={x}"
            );
        }
    }

    /// Real JS switch semantics over a *dynamic* discriminant: fall-through
    /// (empty `case 1` falls into `case 2`), `break`, and `default` all behave
    /// like JS, verified against the reference interpreter across inputs.
    #[test]
    fn switch_fallthrough_break_default() {
        // main(x) {
        //   let r;
        //   switch (x) {
        //     case 0: r = 10; break;
        //     case 1:               // falls through
        //     case 2: r = 20; break;
        //     default: r = 99;
        //   }
        //   return r;
        // }
        let main = FuncDef {
            name: "main",
            nslots: 2, // slot 0 = x (input), slot 1 = r
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["x", "r"],
            body: vec![
                Stmt::Switch(
                    var(0),
                    vec![
                        Clause::Case(num(0), vec![Stmt::Set(1, num(10)), Stmt::Break]),
                        Clause::Case(num(1), vec![]), // fall-through into case 2
                        Clause::Case(num(2), vec![Stmt::Set(1, num(20)), Stmt::Break]),
                        Clause::Default(vec![Stmt::Set(1, num(99))]),
                    ],
                ),
                Stmt::Return(var(1)),
            ],
        };
        let vm = Js::new(&[main]);
        let prog = engine::specialize(&vm, vm.start());
        for x in [-1, 0, 1, 2, 3, 5, 100] {
            assert_eq!(
                vm.run_reference(x),
                vm.run_residual(&prog, x),
                "switch diverged from reference at x={x}"
            );
        }
        // Spot-check the actual JS semantics we expect.
        assert_eq!(vm.run_residual(&prog, 0), DeepVal::Num(10));
        assert_eq!(vm.run_residual(&prog, 1), DeepVal::Num(20)); // fell through
        assert_eq!(vm.run_residual(&prog, 2), DeepVal::Num(20));
        assert_eq!(vm.run_residual(&prog, 7), DeepVal::Num(99)); // default
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

    // -----------------------------------------------------------------------
    // A Brainfuck interpreter, written in the JS subset, specialized away.
    // -----------------------------------------------------------------------

    /// Parse Brainfuck source into a nested AST of JS-subset objects:
    ///   {op:"seq",  body:[...]}     a sequence of instructions
    ///   {op:"add",  n}              `+`/`-` runs, folded to a delta
    ///   {op:"move", n}              `>`/`<` runs, folded to a delta
    ///   {op:"out"}                  `.`
    ///   {op:"in"}                   `,`
    ///   {op:"loop", seq:{op:"seq"}} `[ ... ]`
    /// This is real BF source going in (not a hand-built AST), the way the
    /// expression-interpreter demo feeds a static AST.
    fn parse_bf(src: &str) -> Expr {
        let chars: Vec<char> = src.chars().collect();
        let mut pos = 0;
        let body = parse_bf_seq(&chars, &mut pos, false);
        seq_node(body)
    }

    fn seq_node(body: Vec<Expr>) -> Expr {
        obj(vec![("op", str_("seq")), ("body", arr(body))])
    }

    fn parse_bf_seq(chars: &[char], pos: &mut usize, nested: bool) -> Vec<Expr> {
        let mut out = Vec::new();
        while *pos < chars.len() {
            match chars[*pos] {
                '+' | '-' => {
                    let mut n = 0i64;
                    while *pos < chars.len() && (chars[*pos] == '+' || chars[*pos] == '-') {
                        n += if chars[*pos] == '+' { 1 } else { -1 };
                        *pos += 1;
                    }
                    out.push(node("add", vec![("n", num(n))]));
                }
                '>' | '<' => {
                    let mut n = 0i64;
                    while *pos < chars.len() && (chars[*pos] == '>' || chars[*pos] == '<') {
                        n += if chars[*pos] == '>' { 1 } else { -1 };
                        *pos += 1;
                    }
                    out.push(node("move", vec![("n", num(n))]));
                }
                '.' => {
                    out.push(node("out", vec![]));
                    *pos += 1;
                }
                ',' => {
                    out.push(node("in", vec![]));
                    *pos += 1;
                }
                '[' => {
                    *pos += 1; // consume '['
                    let inner = parse_bf_seq(chars, pos, true);
                    assert_eq!(chars.get(*pos), Some(&']'), "unmatched '[' in BF source");
                    *pos += 1; // consume ']'
                    out.push(node("loop", vec![("seq", seq_node(inner))]));
                }
                ']' => {
                    assert!(nested, "unmatched ']' in BF source");
                    break; // caller consumes ']'
                }
                _ => *pos += 1, // ignore comments / whitespace
            }
        }
        out
    }

    /// `exec(node, tape, ptr, out, input, inptr)` interprets one BF AST node,
    /// returning `[ptr, inptr]` (the mutated cursor + input position). `tape` and
    /// `out` are arrays mutated in place; `ptr`/`inptr` are threaded as values.
    fn bf_exec() -> FuncDef {
        // slots: 0 node, 1 tape, 2 ptr, 3 out, 4 input, 5 inptr, 6 i, 7 r
        let tape_at_ptr = || index(var(1), var(2));
        // r = exec(child, tape, ptr, out, input, inptr); ptr = r[0]; inptr = r[1];
        let call_child = |child: Expr| {
            vec![
                Stmt::Set(7, call(func(0), vec![child, var(1), var(2), var(3), var(4), var(5)])),
                Stmt::Set(2, index(var(7), num(0))),
                Stmt::Set(5, index(var(7), num(1))),
            ]
        };
        let mut seq_body = vec![Stmt::Set(6, num(0))];
        let mut seq_loop = vec![];
        seq_loop.extend(call_child(index(get(var(0), "body"), var(6))));
        seq_loop.push(Stmt::Set(6, bin(Bop::Add, var(6), num(1))));
        seq_body.push(Stmt::While(
            bin(Bop::Lt, var(6), get(get(var(0), "body"), "length")),
            seq_loop,
        ));

        FuncDef {
            name: "exec",
            nslots: 8,
            ncaptured: 0,
            nparams: 6,
            slot_names: vec!["node", "tape", "ptr", "out", "input", "inptr", "i", "r"],
            body: vec![
                Stmt::Switch(
                    get(var(0), "op"),
                    vec![
                        // tape[ptr] += node.n
                        Clause::Case(
                            str_("add"),
                            vec![
                                Stmt::SetIndex(
                                    var(1),
                                    var(2),
                                    bin(Bop::Add, tape_at_ptr(), get(var(0), "n")),
                                ),
                                Stmt::Break,
                            ],
                        ),
                        // ptr += node.n
                        Clause::Case(
                            str_("move"),
                            vec![
                                Stmt::Set(2, bin(Bop::Add, var(2), get(var(0), "n"))),
                                Stmt::Break,
                            ],
                        ),
                        // out.push(tape[ptr])
                        Clause::Case(
                            str_("out"),
                            vec![Stmt::Push(var(3), tape_at_ptr()), Stmt::Break],
                        ),
                        // tape[ptr] = input[inptr]; inptr += 1
                        Clause::Case(
                            str_("in"),
                            vec![
                                Stmt::SetIndex(var(1), var(2), index(var(4), var(5))),
                                Stmt::Set(5, bin(Bop::Add, var(5), num(1))),
                                Stmt::Break,
                            ],
                        ),
                        // while (tape[ptr] !== 0) exec(node.seq, ...)
                        Clause::Case(
                            str_("loop"),
                            vec![
                                Stmt::While(
                                    bin(Bop::Ne, tape_at_ptr(), num(0)),
                                    call_child(get(var(0), "seq")),
                                ),
                                Stmt::Break,
                            ],
                        ),
                        // run each node in node.body
                        Clause::Case(str_("seq"), {
                            seq_body.push(Stmt::Break);
                            seq_body
                        }),
                    ],
                ),
                Stmt::Return(arr(vec![var(2), var(5)])),
            ],
        }
    }

    /// `main(input)`: set up an 8-cell tape, feed `input` to the BF `,`, run the
    /// program, return the output array.
    fn bf_main(program: Expr) -> FuncDef {
        // slots: 0 input, 1 tape, 2 out, 3 input_arr, 4 r
        FuncDef {
            name: "main",
            nslots: 5,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["input", "tape", "out", "input_arr", "r"],
            body: vec![
                Stmt::Let(1, arr(vec![num(0); 8])),
                Stmt::Let(2, arr(vec![])),
                Stmt::Let(3, arr(vec![var(0)])),
                Stmt::Let(
                    4,
                    call(func(0), vec![program, var(1), num(0), var(2), var(3), num(0)]),
                ),
                Stmt::Return(var(2)),
            ],
        }
    }

    /// The Futamura projection on a *Brainfuck* interpreter. The BF program reads
    /// the input into cell 0, adds 6, prints it; then computes 65 (= 13 * 5) with
    /// a static-count loop and prints that. Because every loop count is static
    /// (only the cell *value* is dynamic), the whole interpreter, the AST, the
    /// dispatch `switch`, the tape, and all loops specialize away, leaving a
    /// single residual block that builds `[input + 6, 65]`.
    #[test]
    fn brainfuck_interpreter_specializes_away() {
        let src = ",++++++.>+++++++++++++[->+++++<]>.";
        let program = parse_bf(src);
        let vm = Js::new(&[bf_exec(), bf_main(program)]);

        let mut prog = engine::specialize(&vm, vm.start());
        crate::residual::simplify(&mut prog);

        // The interpreter, AST, tape, switch dispatch and loops are all gone:
        // straight-line residual with no branches.
        assert_eq!(
            prog.blocks.len(),
            1,
            "BF interpreter did not fully specialize away:\n{}",
            vm.dump(&prog)
        );
        let has_newarray = prog
            .blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|op| matches!(op, Op::NewArray { .. }));
        assert!(has_newarray, "the output array should be materialized");

        // The residual matches the reference BF interpreter for every input, and
        // really does depend on the input (cell 0 = input + 6).
        for input in [0, 1, 5, 65, 200] {
            let got = vm.run_residual(&prog, input);
            assert_eq!(vm.run_reference(input), got, "BF residual diverged at input={input}");
            assert_eq!(got, DeepVal::Array(vec![DeepVal::Num(input + 6), DeepVal::Num(65)]));
        }
    }

    /// A second BF program with a static loop and pointer motion, to show the
    /// interpreter handles real BF control flow (not just a straight line).
    /// Computes 6 * 7 = 42 into a cell and prints it; no input, fully constant.
    #[test]
    fn brainfuck_constant_program() {
        // cell0 = 6; [ while cell0: cell1 += 7; cell0-- ]  => cell1 = 42; print cell1
        let src = "++++++[->+++++++<]>.";
        let program = parse_bf(src);
        let vm = Js::new(&[bf_exec(), bf_main(program)]);

        let mut prog = engine::specialize(&vm, vm.start());
        crate::residual::simplify(&mut prog);

        assert_eq!(prog.blocks.len(), 1, "constant BF program left control flow:\n{}", vm.dump(&prog));
        for input in [0, 9, 100] {
            let got = vm.run_residual(&prog, input);
            assert_eq!(vm.run_reference(input), got, "diverged at input={input}");
            assert_eq!(got, DeepVal::Array(vec![DeepVal::Num(42)]));
        }
    }
}
