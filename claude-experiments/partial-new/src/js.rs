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
    // Bitwise / shift. JS evaluates these with 32-bit integer semantics
    // (`ToInt32`/`ToUint32`), so they fold to constants on static integers,
    // which is essential for specializing bytecode interpreters whose dispatch
    // is `switch (pc & mask)`.
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^
    Shl,    // <<
    Shr,    // >>  (arithmetic / sign-propagating)
    UShr,   // >>> (logical / zero-fill)
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
            Bop::BitAnd => "&",
            Bop::BitOr => "|",
            Bop::BitXor => "^",
            Bop::Shl => "<<",
            Bop::Shr => ">>",
            Bop::UShr => ">>>",
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
    /// `place++` / `++place` / `place--` / `--place` used for its *value*. The
    /// place is read, written back as `place ± 1`, and the expression yields the
    /// old value (postfix) or the new value (prefix). The place (and any index)
    /// must be a *pure*, re-evaluable expression (a `Var`, or a `Get`/`Index`
    /// chain over pure bases): it is evaluated more than once, so a call/`new`/
    /// closure inside it is rejected at compile time.
    Update { place: Box<Expr>, op: Bop, prefix: bool },
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
    SetGlobal(String, Expr),      // name = v  (assignment to an undeclared/global name)
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
    /// If this function references `arguments`, the slot the lowerer reserved
    /// for it. The engine fills it with an array of the actual call arguments on
    /// the inline path (`arguments.length` / `arguments[i]` then read it).
    pub arguments_slot: Option<usize>,
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
    /// Duplicate the top of the operand stack. Used to compile short-circuit
    /// `&&`/`||`: the left value is both tested and (on the short-circuit path)
    /// kept as the result. Duplicating copies the abstract *value* (an effectful
    /// operand has already run and been bound to a temp), so no effect repeats.
    Dup,
    /// Bind the top-of-stack value to a fresh residual temp if it is dynamic, so
    /// it is stable across a following effect. Used for the result of a *postfix*
    /// `x++` whose place residualizes: the old value must be captured before the
    /// write, otherwise the residual re-reads the (already-incremented) place.
    Snapshot,
    Bin(Bop),
    NewObject(Vec<String>),
    NewArray(usize),
    GetProp(String),
    SetPropOp(String),
    SetIndexOp,
    DeletePropOp(String),
    DeleteIndexOp,
    SetGlobalOp(String),
    PushArr,
    GetIndex,
    PushFunc(usize),
    MakeClosure(usize, usize), // fid, ncaptures
    Call(usize),               // nargs
    NewOp(usize),              // new callee(args), nargs — pops args then callee
    /// Begin a try region. `catch_pc` is where the catch starts, `body_end` is
    /// the `PopHandler` pc (normal-completion boundary of the body), and `end` is
    /// the pc just past the catch (the join after the whole try). The latter two
    /// bound the body/catch sub-programs when a try must residualize.
    PushHandler { catch_pc: usize, exc_slot: Option<usize>, body_end: usize, end: usize },
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
    /// A method access (`recv.m` / `recv[i]`) that has been snapshotted by the
    /// freeze pass while pending as a CALLEE on the operand stack. `func` is the
    /// snapshotted method *value* (a frozen temp), `recv` the snapshotted
    /// receiver. As a plain value it decays to `func` (JS detaches `this` once a
    /// method is read into a value); as a call callee, `do_call` emits
    /// `func.call(recv, args)`, which preserves BOTH the pre-call snapshot
    /// (`func` was captured before any intervening effect) AND the correct
    /// `this`. Without this, freezing a pending method callee to a bare temp and
    /// calling it would run with `this === undefined` — an over-throw for
    /// `Function.prototype.call`/`.apply`, a wrong `this` for other methods.
    BoundMethod { func: Box<RExpr>, recv: Box<RExpr> },
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
    /// `name = expr` — write to a runtime global (an undeclared assignment).
    AssignGlobal { name: String, expr: RExpr },
    /// A residual `try { body } catch (vN?) { catch_body }`. Emitted when a `try`
    /// body cannot fully specialize (it contains an operation that may throw at
    /// runtime), so the catch must run at runtime. `body`/`catch_body` are nested
    /// residual programs; the exception binds to slot `catch_slot`.
    Try {
        body: Program<Op, Cond>,
        catch_slot: Option<usize>,
        catch_body: Program<Op, Cond>,
    },
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
    /// A modeled built-in value (e.g. a `TextDecoder` instance, or a bound
    /// method like `TextDecoder.decode`). `kind` selects the Rust model and
    /// `data` carries any captured static state. Built-ins fold at
    /// specialization time when their inputs are static; they have no residual
    /// form, so one reaching a runtime (escape) context is a hard error.
    Builtin { kind: String, data: Vec<Abs> },
}

/// The abstract heap: a *persistent* map from address to object. Persistent so
/// that cloning a `State` (at every branch, jump, loop probe, and memo insert)
/// shares structure instead of deep-copying (clone O(1), update O(log n)).
///
/// A *map* rather than a vector because the heap is garbage-collected: dead
/// objects are reclaimed (see `State::gc`) so it stays bounded by the live set.
/// Addresses come from a monotonic counter and are NEVER reused, so reclaiming
/// an object can never alias its address (and hence its `escape_var` residual
/// name) to a later, different object.
type Heap = im::OrdMap<usize, HeapObj>;

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

#[derive(Clone, Debug)]
pub struct State {
    frames: Vec<Frame>,
    heap: Heap,
    /// Next heap address to hand out. Monotonic: never reset, never reused, so a
    /// reclaimed address is gone for good and its `escape_var` name is unique.
    /// NOT part of the state's identity (see `PartialEq`/`Hash` below): it is an
    /// allocation-numbering detail, so two otherwise-equal states must memoize to
    /// the same residual block regardless of how many addresses they have spent.
    next_addr: usize,
    pending_joins: Vec<(usize, Vec<usize>)>,
    /// Stack of active `try` handlers (innermost last).
    handlers: Vec<Handler>,
}

impl PartialEq for State {
    fn eq(&self, o: &Self) -> bool {
        self.frames == o.frames
            && self.heap == o.heap
            && self.pending_joins == o.pending_joins
            && self.handlers == o.handlers
    }
}
impl Eq for State {}
impl std::hash::Hash for State {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.frames.hash(h);
        self.heap.hash(h);
        self.pending_joins.hash(h);
        self.handlers.hash(h);
    }
}

impl State {
    fn top(&self) -> &Frame {
        self.frames.last().unwrap()
    }
    fn top_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().unwrap()
    }

    /// Allocate `obj` at a fresh, never-reused address.
    fn alloc(&mut self, obj: HeapObj) -> usize {
        let addr = self.next_addr;
        self.next_addr += 1;
        self.heap.insert(addr, obj);
        addr
    }

    /// Reclaim heap objects unreachable from the frames so the abstract heap
    /// stays bounded by the live set (the append-only growth is what made
    /// specializing long static computations quadratic). Addresses are not
    /// renumbered: survivors keep their address, so no residual variable name
    /// shifts. Roots are every `Ref` in the frames' locals and operand stacks;
    /// reachability follows object/array/closure/built-in children.
    fn gc(&mut self) {
        let mut live = std::collections::HashSet::new();
        for f in &self.frames {
            for v in f.locals.iter().chain(f.ostack.iter()) {
                if let Abs::Ref(addr) = v {
                    Js::reachable(&self.heap, *addr, &mut live);
                }
            }
        }
        if live.len() != self.heap.len() {
            self.heap = self
                .heap
                .iter()
                .filter(|(addr, _)| live.contains(*addr))
                .map(|(k, v)| (*k, v.clone()))
                .collect();
        }
    }

    /// A cheap proxy for this state's memory footprint: the total number of
    /// residual-expression nodes held across all live abstract values (frame
    /// locals/operand stacks plus heap objects). Residualized abstract values
    /// (`Abs::Dyn`) carry `RExpr` trees that can grow without bound when a
    /// program branch-explodes (e.g. each `x++` wraps the prior tree), so this is
    /// what actually blows up — block and live-object counts stay small while the
    /// trees, cloned into every memoized state, dominate memory. Used only to
    /// enforce `SPEC_WEIGHT_BUDGET`.
    fn weight(&self) -> u64 {
        let mut w = 0u64;
        for f in &self.frames {
            for v in f.locals.iter().chain(f.ostack.iter()) {
                w += abs_weight(v);
            }
        }
        for (_, o) in self.heap.iter() {
            w += match o {
                HeapObj::Object(fs) => fs.iter().map(|(_, a)| abs_weight(a)).sum::<u64>(),
                HeapObj::Array(xs) | HeapObj::Builtin { data: xs, .. } => {
                    xs.iter().map(abs_weight).sum::<u64>()
                }
                HeapObj::Closure { captured, .. } => {
                    captured.iter().map(abs_weight).sum::<u64>()
                }
            };
        }
        w
    }
}

/// Number of `RExpr` nodes an abstract value carries (1 for any non-residual
/// value; the full tree size for `Abs::Dyn`). See `State::weight`.
fn abs_weight(a: &Abs) -> u64 {
    match a {
        Abs::Dyn(e) => rexpr_weight(e),
        _ => 1,
    }
}

/// Can evaluating this residual expression throw at runtime? A member/index read
/// throws when its base is null/undefined; a call/`new` can throw; `in` and
/// `instanceof` throw on a non-object operand. Used so a *discarded* expression
/// that can throw is still emitted for effect (a discarded one that cannot throw
/// is dropped). Conservative over-approximation is safe: emitting a discarded
/// expression that turns out not to throw is a harmless extra read.
fn may_throw(e: &RExpr) -> bool {
    match e {
        RExpr::Get(..) | RExpr::Index(..) | RExpr::Call(..) | RExpr::New(..) => true,
        RExpr::Bin(_, a, b) => may_throw(a) || may_throw(b),
        RExpr::Opaque(op, args) => {
            op == "in" || op == "instanceof" || args.iter().any(may_throw)
        }
        RExpr::FnRef { caps, .. } => caps.iter().any(may_throw),
        RExpr::BoundMethod { func, recv } => may_throw(func) || may_throw(recv),
        _ => false,
    }
}

fn rexpr_weight(e: &RExpr) -> u64 {
    match e {
        RExpr::Bin(_, a, b) | RExpr::Index(a, b) => 1 + rexpr_weight(a) + rexpr_weight(b),
        RExpr::Get(a, _) => 1 + rexpr_weight(a),
        RExpr::Opaque(_, xs) => 1 + xs.iter().map(rexpr_weight).sum::<u64>(),
        RExpr::Call(c, xs) | RExpr::New(c, xs) => {
            1 + rexpr_weight(c) + xs.iter().map(rexpr_weight).sum::<u64>()
        }
        RExpr::FnRef { caps, .. } => 1 + caps.iter().map(rexpr_weight).sum::<u64>(),
        RExpr::BoundMethod { func, recv } => 1 + rexpr_weight(func) + rexpr_weight(recv),
        _ => 1,
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
            Stmt::SetGlobal(name, v) => {
                compile_expr(v, code);
                code.push(Instr::SetGlobalOp(name.clone()));
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
                code.push(Instr::PushHandler {
                    catch_pc: 0,
                    exc_slot: *catch_slot,
                    body_end: 0,
                    end: 0,
                });
                compile_stmts(body, code, aux);
                let body_end = code.len();
                code.push(Instr::PopHandler);
                let jmp = code.len();
                code.push(Instr::Jmp(0)); // normal completion skips the catch
                let catch_pc = code.len();
                aux.catch_pcs.push(catch_pc);
                compile_stmts(catch_body, code, aux);
                let end = code.len();
                patch(&mut code[jmp], end);
                if let Instr::PushHandler { catch_pc: cp, body_end: be, end: en, .. } =
                    &mut code[ph]
                {
                    *cp = catch_pc;
                    *be = body_end;
                    *en = end;
                }
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
            // `&&`, `||`, `??`, and `?:` short-circuit: an operand that is only
            // *conditionally* evaluated must not have its side effects applied on
            // the path where it is skipped. When that operand is pure, the compact
            // pass-through `Opaque` is correct (and is what real inputs rely on for
            // a small residual); when it has side effects, compile real
            // short-circuit control flow so only the operand actually taken runs.
            let needs_branch = match op.as_str() {
                "&&" | "||" | "??" => args.len() == 2 && !is_pure(&args[1]),
                "?:" => args.len() == 3 && (!is_pure(&args[1]) || !is_pure(&args[2])),
                _ => false,
            };
            if needs_branch {
                compile_short_circuit(op, args, code);
            } else {
                for a in args {
                    compile_expr(a, code);
                }
                code.push(Instr::OpaqueOp { op: op.clone(), arity: args.len() });
            }
        }
        Expr::Global(name) => code.push(Instr::PushGlobal(name.clone())),
        Expr::Update { place, op, prefix } => compile_update(place, *op, *prefix, code),
    }
}

/// Compile a short-circuiting `&&`/`||`/`??`/`?:` whose conditionally-evaluated
/// operand has side effects, as real control flow so the skipped operand never
/// runs (and never applies its effects). Leaves the operator's value on the
/// stack. Used only when the compact pass-through `Opaque` form would be unsound;
/// see `compile_expr`.
fn compile_short_circuit(op: &str, args: &[Expr], code: &mut Vec<Instr>) {
    match op {
        // a && b  ≡  let t = a; t ? b : t
        "&&" => {
            compile_expr(&args[0], code);
            code.push(Instr::Dup);
            let jz = code.len();
            code.push(Instr::JmpIfFalsy(0)); // a falsy: keep a as the result
            code.push(Instr::Pop); // a truthy: drop the kept a, value is b
            compile_expr(&args[1], code);
            let end = code.len();
            patch(&mut code[jz], end);
        }
        // a || b  ≡  let t = a; t ? t : b
        "||" => {
            compile_expr(&args[0], code);
            code.push(Instr::Dup);
            let jz = code.len();
            code.push(Instr::JmpIfFalsy(0)); // a falsy: evaluate b
            let jend = code.len();
            code.push(Instr::Jmp(0)); // a truthy: keep a, skip b
            let els = code.len();
            code.push(Instr::Pop); // drop the kept a
            compile_expr(&args[1], code);
            let end = code.len();
            patch(&mut code[jz], els);
            patch(&mut code[jend], end);
        }
        // a ?? b  ≡  let t = a; (t == null) ? b : t   (`== null` is true iff
        // `t` is null or undefined — exactly the nullish test).
        "??" => {
            compile_expr(&args[0], code);
            code.push(Instr::Dup);
            code.push(Instr::PushNull);
            code.push(Instr::OpaqueOp { op: "==".to_string(), arity: 2 });
            let jz = code.len();
            code.push(Instr::JmpIfFalsy(0)); // (a == null) falsy: a not nullish, keep a
            code.push(Instr::Pop); // nullish: drop a, value is b
            compile_expr(&args[1], code);
            let jend = code.len();
            code.push(Instr::Jmp(0));
            let end = code.len();
            patch(&mut code[jz], end); // falsy lands here with a still on the stack
            patch(&mut code[jend], end);
        }
        // a ? b : c
        "?:" => {
            compile_expr(&args[0], code);
            let jz = code.len();
            code.push(Instr::JmpIfFalsy(0));
            compile_expr(&args[1], code); // then
            let jend = code.len();
            code.push(Instr::Jmp(0));
            let els = code.len();
            compile_expr(&args[2], code); // else
            let end = code.len();
            patch(&mut code[jz], els);
            patch(&mut code[jend], end);
        }
        _ => unreachable!("compile_short_circuit on {op}"),
    }
}

/// Is `e` safe to evaluate more than once (no observable side effect, same value
/// each time)? Used to guard the re-evaluation in `compile_update`.
fn is_pure(e: &Expr) -> bool {
    match e {
        Expr::Num(_) | Expr::Str(_) | Expr::Bool(_) | Expr::Undefined | Expr::Null
        | Expr::This | Expr::Var(_) | Expr::Func(_) | Expr::Global(_) => true,
        Expr::Bin(_, a, b) => is_pure(a) && is_pure(b),
        Expr::Get(o, _) => is_pure(o),
        Expr::Index(a, i) => is_pure(a) && is_pure(i),
        Expr::Opaque(_, args) => args.iter().all(is_pure),
        // Calls, `new`, closures, and object/array literals are not re-evaluable
        // (effects / fresh identity).
        _ => false,
    }
}

/// Compile `place++` / `++place` (and the `--` forms) used for its value. The
/// place is read once for the result, and the increment re-evaluates the (pure)
/// place to write back `place ± 1`, so no extra stack-shuffling instructions are
/// needed. Pre-increment re-reads after the store to yield the new value.
fn compile_update(place: &Expr, op: Bop, prefix: bool, code: &mut Vec<Instr>) {
    // Emit `place = place ± 1` (consumes nothing, leaves nothing).
    let emit_store = |code: &mut Vec<Instr>| match place {
        Expr::Var(slot) => {
            code.push(Instr::Load(*slot));
            // `++`/`--` coerce the place to a number first: `obj++` stores
            // `ToNumber(obj) + 1` (NaN), not `obj + 1` (string concat). `x - 0`
            // is `ToNumber(x)` and folds to nothing when `x` is already numeric.
            code.push(Instr::PushNum(0));
            code.push(Instr::Bin(Bop::Sub));
            code.push(Instr::PushNum(1));
            code.push(Instr::Bin(op));
            code.push(Instr::Store(*slot));
        }
        Expr::Get(o, k) => {
            assert!(is_pure(o), "update of a property on a non-pure base");
            compile_expr(o, code); // obj (for the write)
            compile_expr(o, code); // obj (for the read)
            code.push(Instr::GetProp(k.clone()));
            // `++`/`--` coerce the place to a number first: `obj++` stores
            // `ToNumber(obj) + 1` (NaN), not `obj + 1` (string concat). `x - 0`
            // is `ToNumber(x)` and folds to nothing when `x` is already numeric.
            code.push(Instr::PushNum(0));
            code.push(Instr::Bin(Bop::Sub));
            code.push(Instr::PushNum(1));
            code.push(Instr::Bin(op));
            code.push(Instr::SetPropOp(k.clone()));
        }
        Expr::Index(a, i) => {
            assert!(is_pure(a) && is_pure(i), "update of an index on a non-pure base");
            compile_expr(a, code); // arr (for the write)
            compile_expr(i, code); // index (for the write)
            compile_expr(a, code); // arr (for the read)
            compile_expr(i, code); // index (for the read)
            code.push(Instr::GetIndex);
            // `++`/`--` coerce the place to a number first: `obj++` stores
            // `ToNumber(obj) + 1` (NaN), not `obj + 1` (string concat). `x - 0`
            // is `ToNumber(x)` and folds to nothing when `x` is already numeric.
            code.push(Instr::PushNum(0));
            code.push(Instr::Bin(Bop::Sub));
            code.push(Instr::PushNum(1));
            code.push(Instr::Bin(op));
            code.push(Instr::SetIndexOp);
        }
        _ => unreachable!("compile_update place must be Var/Get/Index"),
    };

    if prefix {
        emit_store(code);
        compile_expr(place, code); // read the new value
    } else {
        compile_expr(place, code); // read the old value (the result)
        // The result of a *postfix* `x++`/`x--` is `ToNumber(old)`, not the raw
        // old value: `false--` yields 0 and `"3"--` yields 3, not `false`/`"3"`.
        // `x - 0` is exactly `ToNumber(x)` (and folds to a no-op when `x` is
        // already a number). The prefix form needs no coercion: its result is the
        // stored `place ± 1`, which the `Bin` already coerces.
        code.push(Instr::PushNum(0));
        code.push(Instr::Bin(Bop::Sub));
        // Capture it before the store: if the place residualizes, the store
        // mutates it, and an un-snapshotted dynamic old value would re-read the
        // new value at runtime.
        code.push(Instr::Snapshot);
        emit_store(code);
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

/// Upper bound on accumulated `State::weight` (residual-expression nodes summed
/// over residual block entries) in one specialization. This is the
/// memory-proportional bound: each memoized block clones its state, and the
/// abstract values' `RExpr` trees are what grow without bound in a
/// branch-exploding program.
///
/// The default suits direct CLI use on real inputs: the simple.js deobfuscation,
/// the hardest real input, accumulates ~141M and peaks under 700 MB, so 300M
/// clears it with headroom. The fuzzer only ever specializes small generated
/// programs (the largest observed weight is ~1.4k), so it lowers this to ~1M via
/// the `SPEC_WEIGHT_BUDGET` env var; a generated branch-exploder then refuses at
/// a few hundred MB instead of exhausting memory. (A blowup and simple.js are
/// within ~1.4x on every cheap count metric — steps, blocks, live heap — so only
/// a budget tuned to the workload separates them; there is no single threshold
/// that fits both, hence the env knob.)
const SPEC_WEIGHT_BUDGET: u64 = 300_000_000;

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
    /// Per function, the slot to fill with an `arguments` array on inline (if the
    /// function references `arguments`).
    arguments_slot: Vec<Option<usize>>,
    is_recursive: Vec<bool>,
    /// Whether each function contains a `try`. Such a function is never inlined:
    /// it is residualized as a function so a `return` in its body is a real
    /// function return, not a `return` from whatever inlined it (which would, for
    /// a residual `try`, exit the caller early).
    has_try: Vec<bool>,
    /// Source name of each function (for recognizing built-ins overridden by a
    /// user definition, e.g. a hand-rolled `TextDecoder`).
    names: Vec<String>,
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
    /// Stack of `pc`s at which the current (sub-)specialization must stop (treat
    /// as `Halt`). Used to bound a residual `try`'s body/catch sub-programs and
    /// the foldability probe to their region. Empty for the main run.
    halt_at: std::cell::RefCell<Vec<usize>>,
    /// Catch-`pc`s of residual `try`s currently being specialized (innermost
    /// last). A residual `try`'s `catch`/`body` is specialized as a fresh nested
    /// `specialize` call; if a `continue`/`break` inside it jumps back to an
    /// enclosing loop head, that nested call re-enters the *same* residual `try`,
    /// which would recurse forever (each nested call has its own memo, so it
    /// never converges). We detect the re-entry and raise a clear, bounded error
    /// instead of overflowing the stack. (Properly threading the non-local exit
    /// out of the residual sub-program is future work.)
    residual_try_stack: std::cell::RefCell<Vec<usize>>,
    /// True while running the foldability probe for a `try`: a would-be
    /// `forbid_residual_in_try` records taint instead of aborting.
    probing: std::cell::Cell<bool>,
    /// Set by the probe when the try body would emit a may-throw residual op, so
    /// the engine knows the try must residualize instead of fold inline.
    try_taint: std::cell::Cell<bool>,
    /// True while specializing a residual `try`'s body/catch: a write to a
    /// *top-frame* local emits a residual assignment immediately (in source
    /// order) rather than only updating the abstract local. This keeps effects
    /// ordered with respect to may-throw residual ops, so a throw partway through
    /// the body leaves the same partial state as the original. (Inlined callees,
    /// at greater frame depth, still fold normally.)
    eager_stores: std::cell::Cell<bool>,
    /// Total `step` calls during this specialization. A program whose driving
    /// branch-explodes (e.g. a depth-bounded recursion whose body forks on
    /// dynamic values inside a loop) can produce a finite but astronomically
    /// large residual, exhausting memory before any depth/loop bound trips. When
    /// the count crosses `SPEC_STEP_BUDGET` we stop with a clear, catchable
    /// refusal rather than OOM. This is a resource bound, not a soundness
    /// mechanism: it lives in the JS client, never the generic engine.
    spec_steps: std::cell::Cell<u64>,
    /// Count of *block entries* (a `step` with `at_entry`), i.e. residual blocks
    /// created (diagnostic).
    spec_blocks: std::cell::Cell<u64>,
    /// Accumulated `State::weight` (residual-expression nodes) across all block
    /// entries. Each memoized block holds a clone of its state, so total memory
    /// tracks the sum of per-state weights. A branch-exploding program grows its
    /// abstract values' `RExpr` trees without bound, so this sum dwarfs the
    /// largest legitimate program's even when block, step, and live-object counts
    /// look comparable. This is the budgeted metric.
    spec_weight: std::cell::Cell<u64>,
    /// The weight budget (defaults to `SPEC_WEIGHT_BUDGET`, override with the
    /// `SPEC_WEIGHT_BUDGET` env var for calibration).
    spec_budget: u64,
}

impl Js {
    /// Total `step` calls consumed by the last `specialize` over this client
    /// (diagnostic; used to calibrate the budgets).
    pub fn spec_steps_used(&self) -> u64 {
        self.spec_steps.get()
    }
    /// Residual blocks created by the last `specialize` (diagnostic).
    pub fn spec_blocks_used(&self) -> u64 {
        self.spec_blocks.get()
    }
    /// Accumulated state weight over the last `specialize` (the budgeted metric).
    pub fn spec_weight_used(&self) -> u64 {
        self.spec_weight.get()
    }
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
        let arguments_slot: Vec<Option<usize>> = program.iter().map(|f| f.arguments_slot).collect();
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
        let names = program.iter().map(|f| f.name.to_string()).collect();
        // A function "has a try" if its own code (not callees) contains a handler.
        let has_try: Vec<bool> = (0..program.len())
            .map(|fid| {
                let lo = entries[fid];
                let hi = entries.get(fid + 1).copied().unwrap_or(code.len());
                code[lo..hi].iter().any(|i| matches!(i, Instr::PushHandler { .. }))
            })
            .collect();

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
            arguments_slot,
            is_recursive,
            has_try,
            names,
            main,
            residual_fns: std::cell::RefCell::new(Vec::new()),
            fn_memo: std::cell::RefCell::new(HashMap::new()),
            halt_at: std::cell::RefCell::new(Vec::new()),
            residual_try_stack: std::cell::RefCell::new(Vec::new()),
            probing: std::cell::Cell::new(false),
            try_taint: std::cell::Cell::new(false),
            spec_steps: std::cell::Cell::new(0),
            spec_blocks: std::cell::Cell::new(0),
            spec_weight: std::cell::Cell::new(0),
            spec_budget: std::env::var("SPEC_WEIGHT_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SPEC_WEIGHT_BUDGET),
            eager_stores: std::cell::Cell::new(false),
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
        // A residualized function runs as a real JS function, so its `arguments`
        // reads the native `arguments`. But the residual binds the closure's
        // captures as *leading* parameters (`__rfN.bind(null, cap0, ...)`), so
        // native `arguments` also contains those captures. The real call args are
        // everything after them: `Array.prototype.slice.call(arguments, ncap)`.
        // (For a capture-free function `ncap == 0`, this is just the args.) The
        // expression is pure, so reading it per use is sound.
        if let Some(aslot) = self.arguments_slot[fid] {
            if aslot < locals.len() {
                let slice_call = RExpr::Get(
                    Box::new(RExpr::Get(
                        Box::new(RExpr::Get(
                            Box::new(RExpr::Global("Array".to_string())),
                            "prototype".to_string(),
                        )),
                        "slice".to_string(),
                    )),
                    "call".to_string(),
                );
                locals[aslot] = Abs::Dyn(RExpr::Call(
                    Box::new(slice_call),
                    vec![RExpr::Var(ARGUMENTS_VAR_ID), RExpr::Num(ncap as i64)],
                ));
            }
        }
        let start = State {
            frames: vec![Frame { pc: self.entries[fid], func: fid, locals, ostack: Vec::new() }],
            heap: Heap::new(),
            next_addr: 0,
            pending_joins: Vec::new(),
            handlers: Vec::new(),
        };
        // The body is an independent program: specialize it in a clean context.
        // `residual_fn_for` can be reached *while* the caller is mid-probe or
        // mid-residual-try (e.g. `f.call(x)` inside a `try`), and the inherited
        // `halt_at` / `probing` / `eager_stores` / residual-try state would
        // corrupt the body (it was producing an empty `__rf0`, silently dropping
        // a throwing call). Save, clear, specialize, restore.
        let saved_probing = self.probing.replace(false);
        let saved_taint = self.try_taint.get();
        let saved_eager = self.eager_stores.replace(false);
        let saved_halt = std::mem::take(&mut *self.halt_at.borrow_mut());
        let saved_try_stack = std::mem::take(&mut *self.residual_try_stack.borrow_mut());
        let body = crate::engine::specialize(self, start);
        self.probing.set(saved_probing);
        self.try_taint.set(saved_taint);
        self.eager_stores.set(saved_eager);
        *self.halt_at.borrow_mut() = saved_halt;
        *self.residual_try_stack.borrow_mut() = saved_try_stack;
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
            heap: Heap::new(),
            next_addr: 0,
            pending_joins: Vec::new(),
            handlers: Vec::new(),
        }
    }

    pub fn input_var(&self) -> usize {
        0
    }

    /// Fold a binary operator over abstract operands. Total: any combination it
    /// cannot fold to a static value (mixed/coerced primitive types, a dynamic
    /// operand, an unsupported operator on a known type) **residualizes** to a
    /// runtime `Bin`, so Node evaluates the real JS coercion semantics and the
    /// residual stays observationally equivalent. Heap references must be escaped
    /// to `Dyn` by the caller before reaching here (`to_rexpr` cannot serialize a
    /// live object); see `Instr::Bin`.
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
                // 32-bit integer semantics, matching JS: operands are truncated
                // to int32 (shift count masked to 5 bits); `>>>` is unsigned.
                Bop::BitAnd => Num(((*x as i32) & (*y as i32)) as i64),
                Bop::BitOr => Num(((*x as i32) | (*y as i32)) as i64),
                Bop::BitXor => Num(((*x as i32) ^ (*y as i32)) as i64),
                Bop::Shl => Num(((*x as i32) << ((*y as u32) & 31)) as i64),
                Bop::Shr => Num(((*x as i32) >> ((*y as u32) & 31)) as i64),
                Bop::UShr => Num(((*x as u32) >> ((*y as u32) & 31)) as i64),
            },
            (Str(x), Str(y)) if op == Bop::Add => Str(format!("{x}{y}")),
            (Str(x), Str(y)) if op == Bop::Eq => Bool(x == y),
            (Str(x), Str(y)) if op == Bop::Ne => Bool(x != y),
            (Bool(x), Bool(y)) if op == Bop::Eq => Bool(x == y),
            (Bool(x), Bool(y)) if op == Bop::Ne => Bool(x != y),
            // Any dynamic operand: residualize. (Static prims convert to literals.)
            _ if a.is_dynamic() || b.is_dynamic() => {
                Dyn(RExpr::Bin(op, Box::new(a.to_rexpr()), Box::new(b.to_rexpr())))
            }
            // Bitwise/shift over mixed static primitive types: JS `ToInt32`-coerces
            // both operands (e.g. the `x | 0` idiom, where a non-numeric string
            // coerces to 0). This is common in interpreter dispatch and keeps the
            // result static.
            _ if is_bitwise(op) => match (to_int32(a), to_int32(b)) {
                (Some(x), Some(y)) => Num(int32_bitwise(op, x, y)),
                // a non-coercible operand (a `Dyn` is already handled above):
                // fall through to residualize.
                _ => Dyn(RExpr::Bin(op, Box::new(a.to_rexpr()), Box::new(b.to_rexpr()))),
            },
            // Strict equality involving statically-known null/undefined/other
            // primitives: decide by type+value identity.
            (_, _) if matches!(op, Bop::Eq | Bop::Ne) => {
                let eq = static_strict_eq(a, b);
                Bool(if op == Bop::Eq { eq } else { !eq })
            }
            // Any other mixed-primitive combination (e.g. `1 - "y"` → NaN,
            // `1 + null` → 1, `"a" < "b"`, `undefined + 1` → NaN): JS resolves it
            // by coercion. We do not model every coercion statically (the number
            // model is integer-only), so residualize and let the runtime decide.
            // Always sound; a fold for the common cases can come later.
            _ => Dyn(RExpr::Bin(op, Box::new(a.to_rexpr()), Box::new(b.to_rexpr()))),
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

/// An atom needs no freezing: a residual variable or literal denotes a stable
/// value at a program point (a write elsewhere cannot change it).
fn rexpr_is_atom(e: &RExpr) -> bool {
    matches!(
        e,
        RExpr::Var(_)
            | RExpr::Num(_)
            | RExpr::Str(_)
            | RExpr::Bool(_)
            | RExpr::Undef
            | RExpr::Null
            | RExpr::This
    )
}

/// Does `e` read array `arr` in a way a write/push to it could change: an
/// element (`arr[_]`) or its `.length`?
fn reads_array(e: &RExpr, arr: &RExpr) -> bool {
    match e {
        RExpr::Index(b, _) => **b == *arr,
        RExpr::Get(b, k) => **b == *arr && k == "length",
        _ => false,
    }
}

/// Does `e` contain a subexpression satisfying `pred` (e.g. a read of a location
/// that is about to be written)?
fn rexpr_contains(e: &RExpr, pred: &dyn Fn(&RExpr) -> bool) -> bool {
    if pred(e) {
        return true;
    }
    match e {
        RExpr::Bin(_, a, b) => rexpr_contains(a, pred) || rexpr_contains(b, pred),
        RExpr::Index(a, b) => rexpr_contains(a, pred) || rexpr_contains(b, pred),
        RExpr::Get(o, _) => rexpr_contains(o, pred),
        RExpr::Opaque(_, args) | RExpr::Call(_, args) | RExpr::New(_, args) => {
            args.iter().any(|a| rexpr_contains(a, pred))
        }
        RExpr::FnRef { caps, .. } => caps.iter().any(|c| rexpr_contains(c, pred)),
        RExpr::BoundMethod { func, recv } => rexpr_contains(func, pred) || rexpr_contains(recv, pred),
        _ => false,
    }
}

/// Names inherited from `Object.prototype` that resolve to functions on any
/// plain object. Reading one off a static object (when it is not shadowed by an
/// own field) must residualize, since the abstract object models only own data
/// fields. Also includes `__proto__` (an accessor) and `constructor`.
/// JS `ToNumber` applied to a string, restricted to integers this model can
/// hold. JS trims whitespace, treats the empty/blank string as `0`, and parses a
/// numeric literal (else `NaN`). We fold only the integer cases (decimal); a
/// blank string is `0`, a fractional / non-numeric string yields `None` (it
/// would be `NaN` or a float, which stays residualized).
/// The ASCII members of JS's `trim` whitespace set (note `\v` = U+000B, which
/// Rust's `char::is_ascii_whitespace` omits but JS trims).
fn is_js_ascii_ws(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | ' ')
}

fn parse_js_int(s: &str) -> Option<i64> {
    let t = s.trim();
    if t.is_empty() {
        return Some(0);
    }
    t.parse::<i64>().ok()
}

fn is_object_proto_member(k: &str) -> bool {
    matches!(
        k,
        "toString"
            | "toLocaleString"
            | "valueOf"
            | "hasOwnProperty"
            | "isPrototypeOf"
            | "propertyIsEnumerable"
            | "constructor"
            | "__proto__"
            | "__defineGetter__"
            | "__defineSetter__"
            | "__lookupGetter__"
            | "__lookupSetter__"
    )
}

fn is_bitwise(op: Bop) -> bool {
    matches!(
        op,
        Bop::BitAnd | Bop::BitOr | Bop::BitXor | Bop::Shl | Bop::Shr | Bop::UShr
    )
}

fn int32_bitwise(op: Bop, x: i32, y: i32) -> i64 {
    match op {
        Bop::BitAnd => (x & y) as i64,
        Bop::BitOr => (x | y) as i64,
        Bop::BitXor => (x ^ y) as i64,
        Bop::Shl => (x << ((y as u32) & 31)) as i64,
        Bop::Shr => (x >> ((y as u32) & 31)) as i64,
        Bop::UShr => ((x as u32) >> ((y as u32) & 31)) as i64,
        _ => unreachable!("not a bitwise op: {op:?}"),
    }
}

/// JS `ToInt32` for static primitives: numbers truncate, booleans are 0/1,
/// null/undefined are 0, and a string is parsed as a number (non-numeric → 0).
/// `None` for a heap reference (object coercion is not modeled).
fn to_int32(a: &Abs) -> Option<i32> {
    match a {
        Abs::Num(n) => Some(*n as i32),
        Abs::Bool(b) => Some(*b as i32),
        Abs::Undef | Abs::Null => Some(0),
        Abs::Str(s) => {
            let t = s.trim();
            if t.is_empty() {
                Some(0)
            } else {
                Some(t.parse::<f64>().ok().filter(|f| f.is_finite()).map_or(0, |f| f as i32))
            }
        }
        Abs::Ref(_) | Abs::Dyn(_) => None,
    }
}

/// JS loose equality (`==`) restricted to cases we can decide statically and
/// soundly: same-type primitives (then `==` matches `===`), object identity, and
/// the null/undefined rule (they are loosely equal only to each other). Mixed
/// primitive types invoke coercion (`5 == "5"`), which we don't model, so those
/// return `None` and stay residual.
fn loose_eq_static(a: &Abs, b: &Abs) -> Option<bool> {
    use Abs::*;
    match (a, b) {
        (Dyn(_), _) | (_, Dyn(_)) => None,
        (Num(x), Num(y)) => Some(x == y),
        (Str(x), Str(y)) => Some(x == y),
        (Bool(x), Bool(y)) => Some(x == y),
        (Ref(x), Ref(y)) => Some(x == y),
        (Null | Undef, Null | Undef) => Some(true),
        // null/undefined are loosely equal only to each other.
        (Null | Undef, _) | (_, Null | Undef) => Some(false),
        // mixed primitive types (or object-vs-primitive) coerce: don't fold.
        _ => None,
    }
}

/// Fold an otherwise pass-through operator when its operands are static, so a
/// static computation stays static (essential for interpreter dispatch built
/// from `!`, `~`, `?:`, `&&`, `||`, `??`). Returns `None` to residualize.
fn fold_opaque(op: &str, xs: &[Abs]) -> Option<Abs> {
    use Abs::*;
    match (op, xs) {
        ("?:", [c, a, b]) => c.truthy().map(|t| if t { a.clone() } else { b.clone() }),
        ("!", [x]) => x.truthy().map(|t| Bool(!t)),
        ("~", [Num(n)]) => Some(Num((!(*n as i32)) as i64)),
        // `void e` yields undefined; fold only a static operand (a dynamic one
        // may have a side effect that must be preserved in the residual).
        ("void", [x]) if !x.is_dynamic() => Some(Undef),
        ("typeof", [x]) => match x {
            Num(_) => Some(Str("number".to_string())),
            Str(_) => Some(Str("string".to_string())),
            Bool(_) => Some(Str("boolean".to_string())),
            Undef => Some(Str("undefined".to_string())),
            Null => Some(Str("object".to_string())),
            _ => None, // object/function need the heap; dynamic is unknown
        },
        // Short-circuit operators yield one operand (both are already evaluated
        // here, so this matches JS only for side-effect-free branches, which is
        // the existing pass-through behavior; we just fold the result).
        ("&&", [a, b]) => a.truthy().map(|t| if t { b.clone() } else { a.clone() }),
        ("||", [a, b]) => a.truthy().map(|t| if t { a.clone() } else { b.clone() }),
        ("??", [a, b]) => match a {
            Null | Undef => Some(b.clone()),
            Dyn(_) => None,
            _ => Some(a.clone()),
        },
        ("==", [a, b]) => loose_eq_static(a, b).map(Bool),
        ("!=", [a, b]) => loose_eq_static(a, b).map(|e| Bool(!e)),
        _ => None,
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
        // Resource guard: bound the number of residual blocks so a
        // branch-exploding program fails with a clear refusal instead of
        // exhausting memory. Block count (not raw step count) is the bound,
        // because each block memoizes a full `State`/heap snapshot — that is
        // where the memory goes. (Not a soundness mechanism; see
        // `SPEC_BLOCK_BUDGET`.)
        self.spec_steps.set(self.spec_steps.get() + 1);
        if at_entry {
            self.spec_blocks.set(self.spec_blocks.get() + 1);
            let accum = self.spec_weight.get() + s.weight();
            self.spec_weight.set(accum);
            if accum > self.spec_budget {
                let budget = self.spec_budget;
                panic!(
                    "specialization budget exceeded ({budget} residual-expression nodes): the \
                     program drives the partial evaluator into runaway (branch-exploding) \
                     residual growth"
                );
            }
        }
        let pc = s.top().pc;
        // The foldability probe only needs to learn *whether* the region taints;
        // once it has, stop immediately instead of specializing the rest.
        if self.probing.get() && self.try_taint.get() {
            return Step::Halt;
        }
        // A bounded sub-specialization (a residual `try`'s body/catch, or the
        // foldability probe) stops when it reaches its region's end pc.
        if s.frames.len() == 1 {
            if self.halt_at.borrow().last() == Some(&pc) {
                // For a real residual-try sub-program (not the throwaway probe),
                // write any locals the body/catch modified back to their stable
                // variables so the continuation after the `try` observes them.
                if !self.probing.get() {
                    let slots: Vec<usize> = (0..s.top().locals.len()).collect();
                    self.materialize(s, &slots, out);
                }
                return Step::Halt;
            }
        }
        if std::env::var("JS_TRACE").is_ok() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n % 1_000_000 == 0 {
                eprintln!(
                    "step {n}: pc={pc} func={} depth={} heap={}",
                    s.top().func,
                    s.frames.len(),
                    s.heap.len()
                );
            }
        }
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
            Instr::PushHandler { catch_pc, exc_slot, body_end, end } => {
                let frame_depth = s.frames.len();
                let ostack_depth = s.top().ostack.len();
                let handler = Handler {
                    catch_pc: *catch_pc,
                    exc_slot: *exc_slot,
                    frame_depth,
                    ostack_depth,
                };
                // Decide whether the try body fully specializes (model `throw` as
                // control flow, exceptions vanish) or must residualize a runtime
                // `try`/`catch` (the body has a may-throw residual op). The probe
                // re-specializes the region; if it would trip the soundness guard,
                // residualize. (Already inside a probe: just push and continue.)
                if self.probing.get() || self.try_folds(s, &handler, *body_end, *end) {
                    s.handlers.push(handler);
                    self.advance(s)
                } else {
                    self.residualize_try(s, *catch_pc, *exc_slot, *body_end, *end, out)
                }
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
                // Two reasons to commit a top-frame store as a residual assignment
                // in program order (then read the slot back as that var):
                //  1. eager_stores: inside a residual `try`, the write must commit
                //     before any following may-throw op.
                //  2. a may-throw RHS (`x = a.b`, `x = a[i]` with a runtime/null
                //     base): the read happens *at the assignment* in the original,
                //     so its potential exception must be emitted there. Otherwise a
                //     dead store (`x` never read) drops the throw entirely, and a
                //     conditionally/late-read store delays it past intervening
                //     effects — both observable divergences. Emitting the whole RHS
                //     (not the inner access) keeps short-circuiting intact.
                let throws = matches!(&v, Abs::Dyn(e) if may_throw(e));
                let eager = self.eager_stores.get() && s.frames.len() == 1;
                if eager || throws {
                    // A may-throw store inside a source `try` must mark that try
                    // un-foldable (so it is residualized and the exception is
                    // caught at runtime); without this the emitted throw escapes
                    // the catch (an over-throw). `forbid_residual_in_try` taints
                    // the foldability probe; on the residualized path the handler
                    // set is empty, so this is a no-op there.
                    if throws {
                        self.forbid_residual_in_try(s, "a may-throw store");
                    }
                    let func = s.top().func;
                    let id = self.stable_id(func, *slot);
                    let heap = s.heap.clone();
                    let expr = self.materialize_value(&v, &heap, out);
                    // Reassigning this slot rebinds its residual var `id`. Any
                    // operand still on the stack that reads `id` must be snapshotted
                    // to its pre-reassignment value first: in `a[i]` where `i`
                    // mutates `a` (`a0[a0--]`), the base `a` was evaluated to
                    // `Var(id)` and pushed before `i` ran, so without this it would
                    // read the post-mutation value. (Unlike `freeze_readers`, a bare
                    // `Var(id)` IS frozen here — it is exactly the stale reference.)
                    let fi = s.frames.len() - 1;
                    for k in 0..s.frames[fi].ostack.len() {
                        if let Abs::Dyn(e) = &s.frames[fi].ostack[k] {
                            if rexpr_contains(e, &|x| matches!(x, RExpr::Var(v) if *v == id)) {
                                let fexpr = e.clone();
                                let fdst = Js::freeze_var(pc, 4096 + k);
                                out.push(Op::Eval { dst: fdst, expr: fexpr });
                                s.frames[fi].ostack[k] = Abs::Dyn(RExpr::Var(fdst));
                            }
                        }
                    }
                    out.push(Op::Assign { var: id, expr });
                    s.top_mut().locals[*slot] = Abs::Dyn(RExpr::Var(id));
                } else {
                    s.top_mut().locals[*slot] = v;
                }
                self.advance(s)
            }
            Instr::Snapshot => {
                let v = s.top_mut().ostack.pop().unwrap();
                // A dynamic value is bound to a stable temp so it survives a
                // following store unchanged; a static value needs no snapshot.
                let v = if let Abs::Dyn(expr) = v {
                    let dst = Js::snapshot_var(pc);
                    out.push(Op::Eval { dst, expr });
                    Abs::Dyn(RExpr::Var(dst))
                } else {
                    v
                };
                self.push(s, v)
            }
            Instr::Pop => {
                let v = s.top_mut().ostack.pop().unwrap();
                // A discarded expression that can throw (e.g. `undefined.length;`
                // as a statement) must still run for effect, in program order, or
                // the residual would silently skip the exception the original
                // raises. A discarded value that cannot throw is dropped as before.
                if let Abs::Dyn(rexpr) = &v {
                    if may_throw(rexpr) {
                        self.forbid_residual_in_try(s, "a discarded may-throw expression");
                        let dst = Js::discard_var(pc);
                        out.push(Op::Eval { dst, expr: rexpr.clone() });
                    }
                }
                self.advance(s)
            }
            Instr::Dup => {
                let v = s.top().ostack.last().cloned().expect("Dup on empty stack");
                self.push(s, v)
            }
            Instr::Bin(op) => {
                let b = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                // First try to fold with full JS coercion over static operands
                // (`ToPrimitive`/`ToNumber`/`ToString`): this collapses the
                // coercion soup that obfuscators like JSFuck are built from
                // (`+[]` → 0, `[]+[]` → "", `![]+[]` → "false", ...).
                if let Some(folded) = self.try_fold_bin_coerced(s, *op, &a, &b) {
                    return self.push(s, folded);
                }
                // Otherwise a heap object reaching a primitive context escapes to
                // a residual reference and the operator residualizes.
                let a = self.escape_if_ref(s, a, out);
                let b = self.escape_if_ref(s, b, out);
                let r = Js::eval_bin(*op, &a, &b);
                self.push(s, r)
            }
            Instr::NewObject(keys) => {
                let n = keys.len();
                let base = s.top().ostack.len() - n;
                let vals: Vec<Abs> = s.top_mut().ostack.split_off(base);
                let vals = self.emit_maythrow_elems(s, pc, vals, out);
                let fields = keys.iter().cloned().zip(vals).collect();
                let addr = s.alloc(HeapObj::Object(fields));
                self.push(s, Abs::Ref(addr))
            }
            Instr::NewArray(n) => {
                let base = s.top().ostack.len() - *n;
                let vals: Vec<Abs> = s.top_mut().ostack.split_off(base);
                let vals = self.emit_maythrow_elems(s, pc, vals, out);
                let addr = s.alloc(HeapObj::Array(vals));
                self.push(s, Abs::Ref(addr))
            }
            Instr::GetProp(k) => {
                let o = s.top_mut().ostack.pop().unwrap();
                // A property the abstract heap can't serve (an array method like
                // `.slice`, or anything on a function value such as `.prototype`)
                // makes the value escape to a residual variable; the access then
                // passes through as `base.key`. Object fields and array `.length`
                // stay modeled. A modeled built-in instance yields a bound method
                // (itself a built-in value) so the following call can fold.
                if let Abs::Ref(addr) = &o {
                    let addr = *addr;
                    let needs_escape = match &s.heap[&addr] {
                        // An own data field stays modeled (folds). A non-own key
                        // that names an `Object.prototype` member (`toString`,
                        // `valueOf`, `hasOwnProperty`, `constructor`, ...)
                        // resolves to an inherited function in real JS, which the
                        // own-fields-only model can't represent: escape so the
                        // read residualizes. A non-own, non-prototype key is a
                        // genuine miss (reads as `undefined`) and keeps folding.
                        HeapObj::Object(fields) => {
                            !fields.iter().any(|(fk, _)| fk == k) && is_object_proto_member(k)
                        }
                        HeapObj::Array(_) => k != "length",
                        HeapObj::Closure { .. } => true,
                        HeapObj::Builtin { kind, data } => {
                            let bound = HeapObj::Builtin {
                                kind: format!("{kind}.{k}"),
                                data: data.clone(),
                            };
                            let baddr = s.alloc(bound);
                            return self.push(s, Abs::Ref(baddr));
                        }
                    };
                    if needs_escape {
                        let mut escaped = std::collections::HashSet::new();
                        let base_r = self.escape(s, addr, &mut escaped, out);
                        let v = Abs::Dyn(RExpr::Get(Box::new(base_r), k.clone()));
                        return self.push(s, v);
                    }
                }
                let v = self.get_prop(s, &o, k);
                self.push(s, v)
            }
            Instr::SetPropOp(k) => {
                let v = s.top_mut().ostack.pop().unwrap();
                let o = s.top_mut().ostack.pop().unwrap();
                match o {
                    // Static object: update or append the field in the abstract
                    // heap (stays scalar-replaced). A may-throw RHS is committed
                    // first, so a dead object can't drop the exception.
                    Abs::Ref(addr) if matches!(&s.heap[&addr], HeapObj::Object(_)) => {
                        let v = self.emit_if_maythrow(s, pc, 0, v, out);
                        if let HeapObj::Object(fields) = s.heap.get_mut(&addr).unwrap() {
                            if let Some(slot) = fields.iter_mut().find(|(fk, _)| fk == k) {
                                slot.1 = v;
                            } else {
                                fields.push((k.clone(), v));
                            }
                        }
                    }
                    // A named-property write on a non-object ref (e.g. `arr.foo =
                    // v`, `arr.length = n`): the abstract heap doesn't model it,
                    // so the container escapes and the write residualizes.
                    Abs::Ref(addr) => {
                        let obj_r = self.escape(s, addr, &mut std::collections::HashSet::new(), out);
                        self.forbid_residual_in_try(s, "a property write");
                        let mut escaped = std::collections::HashSet::new();
                        let val = self.operand_rexpr(s, &v, &mut escaped, out);
                        let (o2, key) = (obj_r.clone(), k.clone());
                        self.freeze_readers(s, pc, out, &|e| {
                            matches!(e, RExpr::Get(b, kk) if **b == o2 && *kk == key)
                        });
                        out.push(Op::SetProp { obj: obj_r, key: k.clone(), val });
                    }
                    // A write to a dynamic object — or a primitive base, where JS
                    // either silently ignores the write (`(5).a = v`) or throws
                    // (`null.a = v`) — residualizes. The value, if itself an
                    // object, escapes too.
                    other => {
                        let obj_r = other.to_rexpr();
                        self.forbid_residual_in_try(s, "a property write");
                        let mut escaped = std::collections::HashSet::new();
                        let val = self.operand_rexpr(s, &v, &mut escaped, out);
                        let (o, key) = (obj_r.clone(), k.clone());
                        self.freeze_readers(s, pc, out, &|e| {
                            matches!(e, RExpr::Get(b, kk) if **b == o && *kk == key)
                        });
                        out.push(Op::SetProp { obj: obj_r, key: k.clone(), val });
                    }
                }
                self.advance(s)
            }
            Instr::SetIndexOp => {
                let v = s.top_mut().ostack.pop().unwrap();
                let i = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                let served = matches!(&a, Abs::Ref(_)) && self.index_is_static(s, &a, &i);
                match a {
                    // Static container with a static, type-matched index: mutate
                    // the abstract heap in place (it stays scalar-replaced). An
                    // array grows with `undefined` holes past its end; an object
                    // key write updates or appends a field (the `obj["k"] = v`
                    // form, equivalent to `obj.k = v`).
                    Abs::Ref(addr) if served => {
                        let v = self.emit_if_maythrow(s, pc, 0, v, out);
                        match (s.heap.get_mut(&addr).unwrap(), &i) {
                            (HeapObj::Array(elems), Abs::Num(n)) => {
                                let n = *n as usize;
                                if n >= elems.len() {
                                    elems.resize(n + 1, Abs::Undef);
                                }
                                elems[n] = v;
                            }
                            (HeapObj::Object(fields), Abs::Str(k)) => {
                                if let Some(slot) = fields.iter_mut().find(|(fk, _)| fk == k) {
                                    slot.1 = v;
                                } else {
                                    fields.push((k.clone(), v));
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                    // Static container with a dynamic index or a type-mismatched
                    // key (`arr[dyn] = v`, `arr["x"] = v`, `obj[5] = v`): the
                    // container escapes and the write residualizes.
                    Abs::Ref(addr) => {
                        let arr_r = self.escape(s, addr, &mut std::collections::HashSet::new(), out);
                        self.forbid_residual_in_try(s, "an indexed write");
                        let r = self.materialize_value(&v, &s.heap.clone(), out);
                        let idx = self.escape_if_ref(s, i, out);
                        let a = arr_r.clone();
                        self.freeze_readers(s, pc, out, &|e| reads_array(e, &a));
                        out.push(Op::SetIndex { arr: arr_r, index: idx.to_rexpr(), val: r });
                    }
                    // Dynamic array/object, or a primitive base (`(5)[0] = v` is
                    // a silent no-op, `null[0] = v` throws): residualize the
                    // indexed write. An object index escapes to a string key.
                    other => {
                        let arr_r = other.to_rexpr();
                        self.forbid_residual_in_try(s, "an indexed write");
                        let r = self.materialize_value(&v, &s.heap.clone(), out);
                        let idx = self.escape_if_ref(s, i, out);
                        let a = arr_r.clone();
                        self.freeze_readers(s, pc, out, &|e| reads_array(e, &a));
                        out.push(Op::SetIndex { arr: arr_r, index: idx.to_rexpr(), val: r });
                    }
                }
                self.advance(s)
            }
            Instr::DeletePropOp(k) => {
                let o = s.top_mut().ostack.pop().unwrap();
                match o {
                    // Static object: drop the field from the abstract heap.
                    Abs::Ref(addr) => {
                        if let HeapObj::Object(fields) = s.heap.get_mut(&addr).unwrap() {
                            fields.retain(|(fk, _)| fk != k);
                        } else {
                            panic!("delete of a property on a non-object");
                        }
                    }
                    // Dynamic object, or a primitive base (`delete (5).a` → true,
                    // a no-op): residualize the delete.
                    other => {
                        let obj_r = other.to_rexpr();
                        self.forbid_residual_in_try(s, "a property delete");
                        let (o, key) = (obj_r.clone(), k.clone());
                        self.freeze_readers(s, pc, out, &|e| {
                            matches!(e, RExpr::Get(b, kk) if **b == o && *kk == key)
                        });
                        out.push(Op::DeleteProp { obj: obj_r, key: k.clone() });
                    }
                }
                self.advance(s)
            }
            Instr::SetGlobalOp(name) => {
                self.forbid_residual_in_try(s, "a global assignment");
                let v = s.top_mut().ostack.pop().unwrap();
                let mut escaped = std::collections::HashSet::new();
                let expr = self.operand_rexpr(s, &v, &mut escaped, out);
                let nm = name.clone();
                self.freeze_readers(s, pc, out, &|e| matches!(e, RExpr::Global(n) if *n == nm));
                out.push(Op::AssignGlobal { name: name.clone(), expr });
                self.advance(s)
            }
            Instr::DeleteIndexOp => {
                let i = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                let served = matches!(&a, Abs::Ref(_)) && self.index_is_static(s, &a, &i);
                match a {
                    // Static array/object with a static, type-matched index: a
                    // deleted element becomes a hole (reads as `undefined`),
                    // matching JS `delete arr[i]`.
                    Abs::Ref(addr) if served => {
                        match (s.heap.get_mut(&addr).unwrap(), &i) {
                            (HeapObj::Array(elems), Abs::Num(n)) => {
                                let n = *n as usize;
                                if n < elems.len() {
                                    elems[n] = Abs::Undef;
                                }
                            }
                            (HeapObj::Object(fields), Abs::Str(k)) => {
                                fields.retain(|(fk, _)| fk != k);
                            }
                            _ => unreachable!(),
                        }
                    }
                    // Static container with a dynamic / mismatched index: it
                    // escapes and the delete residualizes.
                    Abs::Ref(addr) => {
                        let arr_r = self.escape(s, addr, &mut std::collections::HashSet::new(), out);
                        self.forbid_residual_in_try(s, "an indexed delete");
                        let idx = self.escape_if_ref(s, i, out);
                        let a = arr_r.clone();
                        self.freeze_readers(s, pc, out, &|e| reads_array(e, &a));
                        out.push(Op::DeleteIndex { arr: arr_r, index: idx.to_rexpr() });
                    }
                    // Dynamic array, or a primitive base (`delete (5)[0]` → true,
                    // a no-op): residualize the delete.
                    other => {
                        let arr_r = other.to_rexpr();
                        self.forbid_residual_in_try(s, "an indexed delete");
                        let idx = self.escape_if_ref(s, i, out);
                        let a = arr_r.clone();
                        self.freeze_readers(s, pc, out, &|e| reads_array(e, &a));
                        out.push(Op::DeleteIndex { arr: arr_r, index: idx.to_rexpr() });
                    }
                }
                self.advance(s)
            }
            Instr::PushArr => {
                let v = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                match a {
                    // Static array: mutate the abstract heap (still scalar).
                    Abs::Ref(addr) if matches!(&s.heap[&addr], HeapObj::Array(_)) => {
                        if let HeapObj::Array(elems) = s.heap.get_mut(&addr).unwrap() {
                            elems.push(v);
                        }
                    }
                    // A non-array static object reaching `.push` (`{}.push(v)`
                    // throws at runtime): escape and residualize so the runtime
                    // throws identically.
                    Abs::Ref(addr) => {
                        let arr_r = self.escape(s, addr, &mut std::collections::HashSet::new(), out);
                        self.forbid_residual_in_try(s, "an array push");
                        let r = self.materialize_value(&v, &s.heap.clone(), out);
                        let a = arr_r.clone();
                        self.freeze_readers(s, pc, out, &|e| reads_array(e, &a));
                        out.push(Op::PushOp { arr: arr_r, val: r });
                    }
                    // Dynamic array, or a primitive base (`(5).push(v)` throws):
                    // residualize the push.
                    other => {
                        let arr_r = other.to_rexpr();
                        self.forbid_residual_in_try(s, "an array push");
                        let r = self.materialize_value(&v, &s.heap.clone(), out);
                        let a = arr_r.clone();
                        self.freeze_readers(s, pc, out, &|e| reads_array(e, &a));
                        out.push(Op::PushOp { arr: arr_r, val: r });
                    }
                }
                self.advance(s)
            }
            Instr::GetIndex => {
                let i = s.top_mut().ostack.pop().unwrap();
                let a = s.top_mut().ostack.pop().unwrap();
                // If the abstract heap can't serve this index statically (a
                // dynamic index, or a type-mismatched key like `arr["x"]` /
                // `obj[5]`), the container and the index escape and the read
                // residualizes. An object used as an index coerces to a string
                // key at runtime, so it escapes too.
                let (a, i) = if self.index_is_static(s, &a, &i) {
                    (a, i)
                } else {
                    let a = self.escape_if_ref(s, a, out);
                    let i = self.escape_if_ref(s, i, out);
                    (a, i)
                };
                let v = self.get_index(s, &a, &i);
                self.push(s, v)
            }
            Instr::PushFunc(fid) => {
                let addr = s.alloc(HeapObj::Closure {
                    fid: *fid,
                    captured: Vec::new(),
                });
                self.push(s, Abs::Ref(addr))
            }
            Instr::MakeClosure(fid, ncap) => {
                let base = s.top().ostack.len() - *ncap;
                let captured: Vec<Abs> = s.top_mut().ostack.split_off(base);
                let addr = s.alloc(HeapObj::Closure {
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
                // A modeled built-in constructor (e.g. `new TextDecoder()`,
                // `new Uint8Array(staticBytes)`): fold to a modeled value when
                // inputs are static, instead of residualizing a `new` call.
                if let Some(name) = self.builtin_ctor_name(&callee, s) {
                    if let Some(v) = self.try_builtin_new(name, &args, s) {
                        s.top_mut().pc += 1;
                        s.top_mut().ostack.push(v);
                        return Step::Continue;
                    }
                }
                self.forbid_residual_in_try(s, "a `new` expression");
                let callee_r = match callee {
                    Abs::Dyn(r) => r,
                    // `new` of a known function/closure: it becomes a residual
                    // function (constructors use `this`, so they must not inline).
                    Abs::Ref(addr) => match &s.heap[&addr] {
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
                // Operators we pass through verbatim but can still *fold* when
                // their operands are static (`!`, `~`, `?:`, `&&`, `||`, `??`,
                // loose `==`/`!=`). This keeps a static computation static: an
                // interpreter's `pc = !flag ? L1 : L2` must not go dynamic just
                // because `!`/`?:` aren't arithmetic, or the dispatch loop
                // residualizes and unrolls forever.
                if let Some(folded) = fold_opaque(op, &operands) {
                    return self.push(s, folded);
                }
                // Otherwise the operation is unmodeled, so the result is dynamic.
                // A heap object operand escapes (it could be mutated by the op).
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
                        // Block boundary: reclaim heap garbage in each successor.
                        tstate.gc();
                        fstate.gc();
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
                // The return expression is evaluated *inside* any enclosing `try`,
                // so a may-throw return value must mark that try un-foldable (it is
                // residualized and the exception caught at runtime). Taint the
                // probe before the handlers are discarded below; a no-op when no
                // handler is active.
                if matches!(&v, Abs::Dyn(e) if may_throw(e)) {
                    self.forbid_residual_in_try(s, "a may-throw return value");
                }
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
        self.get_prop_opt(s, o, k).unwrap_or_else(|| {
            // An unmodeled property read on a static primitive (`(5).toString`,
            // `"".length`, `undefined.x`): residualize as an opaque `base.key`
            // read. Node then applies real JS semantics, including throwing for
            // `null`/`undefined` bases, so the residual stays equivalent. (Heap
            // refs are intercepted and escaped by `Instr::GetProp` before here.)
            Abs::Dyn(RExpr::Get(Box::new(o.to_rexpr()), k.to_string()))
        })
    }

    /// Resolve a property read that the abstract heap can serve, or `None` for an
    /// unmodeled case (e.g. an array method, a property on a primitive) that the
    /// caller must residualize (or, in the loop probe, treat as dynamic).
    fn get_prop_opt(&self, s: &State, o: &Abs, k: &str) -> Option<Abs> {
        match o {
            Abs::Ref(addr) => match &s.heap[addr] {
                HeapObj::Object(fields) => Some(
                    fields
                        .iter()
                        .find(|(fk, _)| fk == k)
                        .map(|(_, v)| v.clone())
                        .unwrap_or(Abs::Undef),
                ),
                HeapObj::Array(elems) if k == "length" => Some(Abs::Num(elems.len() as i64)),
                _ => None,
            },
            // A static string's `.length` folds (JSFuck uses it for digits).
            Abs::Str(st) if k == "length" => Some(Abs::Num(st.chars().count() as i64)),
            // A property read on a dynamic value (a runtime global, a string,
            // an escaped object) passes through as an opaque `base.key` read.
            Abs::Dyn(r) => Some(Abs::Dyn(RExpr::Get(Box::new(r.clone()), k.to_string()))),
            _ => None,
        }
    }

    fn get_index(&self, s: &State, a: &Abs, i: &Abs) -> Abs {
        // String character access (`"false"[1]` → "a") folds to a one-character
        // string, or `undefined` out of range — this is how JSFuck harvests
        // letters out of coerced primitive strings.
        if let (Abs::Str(st), Abs::Num(n)) = (a, i) {
            if *n >= 0 {
                return match st.chars().nth(*n as usize) {
                    Some(c) => Abs::Str(c.to_string()),
                    None => Abs::Undef,
                };
            }
        }
        let addr = match a {
            Abs::Ref(addr) => *addr,
            // A dynamic (escaped) array still supports element reads: the read
            // residualizes to an indexed load. The element value is a runtime
            // value, so the result is dynamic.
            Abs::Dyn(r) => {
                return Abs::Dyn(RExpr::Index(Box::new(r.clone()), Box::new(i.to_rexpr())));
            }
            // Indexing a primitive (`(5)[0]` → undefined, `"ab"[0]` → "a",
            // `null[0]` throws): residualize and let the runtime decide.
            _ => return Abs::Dyn(RExpr::Index(Box::new(a.to_rexpr()), Box::new(i.to_rexpr()))),
        };
        match (&s.heap[&addr], i) {
            (HeapObj::Array(elems), Abs::Num(n)) => {
                elems.get(*n as usize).cloned().unwrap_or(Abs::Undef)
            }
            (HeapObj::Object(fields), Abs::Str(k)) => fields
                .iter()
                .find(|(fk, _)| fk == k)
                .map(|(_, v)| v.clone())
                .unwrap_or(Abs::Undef),
            // A static container indexed by a value it can't serve statically (a
            // dynamic index, or a type-mismatched key such as `arr["x"]` /
            // `obj[5]`): the caller (`Instr::GetIndex`) escapes the container
            // first so the read residualizes. Reaching here means a ref slipped
            // through unescaped, which is a bug.
            _ => unreachable!("get_index: unescaped static container with unservable index"),
        }
    }

    /// JS `ToString` for a statically-known value (recursive for arrays). `None`
    /// if it would touch a dynamic value or a function (which we don't statically
    /// stringify). This is what lets JSFuck-style coercion soup fold to literals.
    fn to_string_static(&self, s: &State, v: &Abs) -> Option<String> {
        self.to_string_rec(s, v, &mut std::collections::HashSet::new())
    }

    fn to_string_rec(
        &self,
        s: &State,
        v: &Abs,
        on_stack: &mut std::collections::HashSet<usize>,
    ) -> Option<String> {
        Some(match v {
            Abs::Num(n) => n.to_string(),
            Abs::Str(st) => st.clone(),
            Abs::Bool(b) => b.to_string(),
            Abs::Null => "null".to_string(),
            Abs::Undef => "undefined".to_string(),
            Abs::Ref(addr) => {
                // A cycle (an array reachable from itself) makes `toString`
                // non-trivial in JS (`Array.join` emits "" for the revisited
                // entry). Rather than model that, bail so the `+`/coercion
                // residualizes and the runtime computes it. Always sound.
                if !on_stack.insert(*addr) {
                    return None;
                }
                let out = match &s.heap[addr] {
                    // `Array.prototype.toString` is `join(",")`, where `null` and
                    // `undefined` elements stringify to the empty string.
                    HeapObj::Array(elems) => {
                        let parts: Option<Vec<String>> = elems
                            .iter()
                            .map(|e| match e {
                                Abs::Null | Abs::Undef => Some(String::new()),
                                _ => self.to_string_rec(s, e, on_stack),
                            })
                            .collect();
                        parts?.join(",")
                    }
                    HeapObj::Object(_) => "[object Object]".to_string(),
                    _ => {
                        on_stack.remove(addr);
                        return None; // closure / builtin
                    }
                };
                on_stack.remove(addr);
                out
            }
            Abs::Dyn(_) => return None,
        })
    }

    /// JS `ToNumber` for a statically-known value, as an `i64`. `None` if dynamic,
    /// a function, or the result is not an integer this model can hold (e.g.
    /// `NaN` from `+undefined` or `+"x"`, or a fractional value) — those stay
    /// residualized, which is sound; folding them needs a richer number model.
    fn to_number_static(&self, s: &State, v: &Abs) -> Option<i64> {
        match v {
            Abs::Num(n) => Some(*n),
            Abs::Bool(b) => Some(if *b { 1 } else { 0 }),
            Abs::Null => Some(0),
            Abs::Undef => None, // NaN
            Abs::Str(st) => parse_js_int(st),
            Abs::Ref(_) => parse_js_int(&self.to_string_static(s, v)?),
            Abs::Dyn(_) => None,
        }
    }

    /// `ToPrimitive` (default hint, as used by `+`): a heap array/object becomes
    /// its string form; a primitive is itself. `None` if dynamic / a function.
    fn to_primitive_static(&self, s: &State, v: &Abs) -> Option<Abs> {
        match v {
            Abs::Ref(_) => Some(Abs::Str(self.to_string_static(s, v)?)),
            Abs::Dyn(_) => None,
            prim => Some(prim.clone()),
        }
    }

    /// Fold a binary operator that requires JS coercion over *static* operands
    /// (the cases `eval_bin` leaves alone): `+` with `ToPrimitive`/string-concat,
    /// numeric `-`/`*`, and relational comparisons. Returns `None` when an
    /// operand is dynamic or the numeric result isn't representable (so the
    /// caller residualizes). This is the core of JSFuck decoding.
    fn try_fold_bin_coerced(&self, s: &State, op: Bop, a: &Abs, b: &Abs) -> Option<Abs> {
        use Bop::*;
        if a.is_dynamic() || b.is_dynamic() {
            return None;
        }
        match op {
            Add => {
                let pa = self.to_primitive_static(s, a)?;
                let pb = self.to_primitive_static(s, b)?;
                // If either primitive is a string, `+` is string concatenation;
                // otherwise it is numeric addition.
                if matches!(pa, Abs::Str(_)) || matches!(pb, Abs::Str(_)) {
                    let sa = self.to_string_static(s, &pa)?;
                    let sb = self.to_string_static(s, &pb)?;
                    Some(Abs::Str(format!("{sa}{sb}")))
                } else {
                    Some(Abs::Num(self.to_number_static(s, &pa)?.wrapping_add(self.to_number_static(s, &pb)?)))
                }
            }
            Sub => Some(Abs::Num(
                self.to_number_static(s, a)?.wrapping_sub(self.to_number_static(s, b)?),
            )),
            Mul => Some(Abs::Num(
                self.to_number_static(s, a)?.wrapping_mul(self.to_number_static(s, b)?),
            )),
            Lt | Le | Gt | Ge => {
                let pa = self.to_primitive_static(s, a)?;
                let pb = self.to_primitive_static(s, b)?;
                if matches!(pa, Abs::Str(_)) && matches!(pb, Abs::Str(_)) {
                    let (sa, sb) = (self.to_string_static(s, &pa)?, self.to_string_static(s, &pb)?);
                    Some(Abs::Bool(match op {
                        Lt => sa < sb,
                        Le => sa <= sb,
                        Gt => sa > sb,
                        _ => sa >= sb,
                    }))
                } else {
                    let (na, nb) = (self.to_number_static(s, &pa)?, self.to_number_static(s, &pb)?);
                    Some(Abs::Bool(match op {
                        Lt => na < nb,
                        Le => na <= nb,
                        Gt => na > nb,
                        _ => na >= nb,
                    }))
                }
            }
            // Bitwise/shift already coerce via `to_int32` in `eval_bin`; strict
            // equality is decided there too.
            _ => None,
        }
    }

    /// Can `get_index` serve this `(container, index)` from the abstract heap
    /// without escaping? Used to decide when to escape before residualizing. An
    /// object used as an index coerces to a string key at runtime, so it always
    /// forces escape. A negative array index is a named property (not an
    /// element), which the element-list model can't serve.
    fn index_is_static(&self, s: &State, a: &Abs, i: &Abs) -> bool {
        if matches!(i, Abs::Ref(_)) {
            return false;
        }
        match a {
            Abs::Ref(addr) => match (&s.heap[addr], i) {
                (HeapObj::Array(_), Abs::Num(n)) => *n >= 0,
                (HeapObj::Object(_), Abs::Str(_)) => true,
                _ => false,
            },
            _ => true, // primitive/dynamic bases residualize purely
        }
    }

    // -- Built-in modeling -------------------------------------------------
    //
    // A small registry of standard library values the evaluator understands
    // semantically. When such a value is constructed or called with *static*
    // inputs, the evaluator computes the result directly (folding it to a
    // constant) instead of residualizing an opaque call. This is what lets,
    // e.g., `new TextDecoder().decode(staticBytes)` collapse to a string
    // literal. Adding a built-in is local: extend the match arms below.

    /// If `callee` names a modeled built-in constructor (whether a free global
    /// like `Uint8Array` or a user function shadowing one like a hand-rolled
    /// `TextDecoder`), return its canonical name.
    fn builtin_ctor_name(&self, callee: &Abs, s: &State) -> Option<&'static str> {
        let raw = match callee {
            Abs::Dyn(RExpr::Global(name)) => name.as_str(),
            Abs::Ref(addr) => match &s.heap[addr] {
                HeapObj::Closure { fid, .. } => self.names.get(*fid)?.as_str(),
                _ => return None,
            },
            _ => return None,
        };
        match raw {
            "TextDecoder" => Some("TextDecoder"),
            "Uint8Array" => Some("Uint8Array"),
            _ => None,
        }
    }

    /// Construct a modeled built-in (`new Name(args)`). Returns `None` (so the
    /// caller residualizes) when the inputs are not static enough to fold.
    fn try_builtin_new(&self, name: &str, args: &[Abs], s: &mut State) -> Option<Abs> {
        match name {
            "TextDecoder" => {
                let addr = s.alloc(HeapObj::Builtin { kind: "TextDecoder".into(), data: vec![] });
                Some(Abs::Ref(addr))
            }
            // `new Uint8Array(arr | length)` -> a static byte array (modeled as a
            // plain array of numbers so element/length reads work as usual).
            "Uint8Array" => {
                let bytes = self.uint8_init(args.first(), s)?;
                let addr = s.alloc(HeapObj::Array(bytes));
                Some(Abs::Ref(addr))
            }
            _ => None,
        }
    }

    /// The element list for `new Uint8Array(arg)`: a copy of a static numeric
    /// array (byte-masked), a zero-filled array for a numeric length, or empty.
    fn uint8_init(&self, arg: Option<&Abs>, s: &State) -> Option<Vec<Abs>> {
        match arg {
            None | Some(Abs::Undef) => Some(vec![]),
            Some(Abs::Num(n)) => Some(vec![Abs::Num(0); (*n).max(0) as usize]),
            Some(Abs::Ref(addr)) => match &s.heap[addr] {
                HeapObj::Array(elems) => {
                    let mut out = Vec::with_capacity(elems.len());
                    for e in elems {
                        match e {
                            Abs::Num(n) => out.push(Abs::Num(n & 0xff)),
                            _ => return None, // a non-static element: can't fold
                        }
                    }
                    Some(out)
                }
                _ => None,
            },
            _ => None, // dynamic argument
        }
    }

    /// Evaluate a modeled bound method (e.g. `TextDecoder.decode`). Returns
    /// `None` when arguments are not static.
    fn try_builtin_method(&self, kind: &str, _data: &[Abs], args: &[Abs], s: &State) -> Option<Abs> {
        match kind {
            "TextDecoder.decode" => {
                let bytes = self.static_bytes(args.first(), s)?;
                Some(Abs::Str(String::from_utf8_lossy(&bytes).into_owned()))
            }
            _ => None,
        }
    }

    /// Evaluate a modeled static method on a global (e.g. `String.fromCharCode`,
    /// `Math.floor`). Returns the folded value, or `None` to residualize (dynamic
    /// or non-numeric args, or a method this model doesn't cover — which then
    /// passes through to Node unchanged). Adding a static-method built-in is local:
    /// add an arm here.
    ///
    /// Math: only the **deterministic, integer-result** methods are modeled. In
    /// the i64 number model every value is an integer, so `floor`/`ceil`/`round`/
    /// `trunc` are the identity; `abs`/`sign`/`max`/`min` fold over static numbers.
    /// Float-producing methods (`sqrt`, `pow`, `hypot`, …) and the non-deterministic
    /// `random` are deliberately NOT modeled — folding them would be unsound or
    /// lossy, so they pass through to the runtime.
    fn try_builtin_static(&self, obj: &str, method: &str, args: &[Abs]) -> Option<Abs> {
        // A single static integer argument.
        let num1 = || match args.first() {
            Some(Abs::Num(n)) => Some(*n),
            _ => None,
        };
        match (obj, method) {
            ("String", "fromCharCode") => {
                let mut st = String::new();
                for a in args {
                    match a {
                        Abs::Num(n) => st.push(char::from_u32((*n as u32) & 0xffff)?),
                        _ => return None,
                    }
                }
                Some(Abs::Str(st))
            }
            // Integer-preserving rounding is the identity in the i64 model.
            ("Math", "floor") | ("Math", "ceil") | ("Math", "round") | ("Math", "trunc") => {
                num1().map(Abs::Num)
            }
            // `checked_abs` residualizes the lone overflow case (`abs(i64::MIN)`),
            // whose true JS value is a float we can't represent.
            ("Math", "abs") => num1()?.checked_abs().map(Abs::Num),
            ("Math", "sign") => num1().map(|n| Abs::Num(n.signum())),
            ("Math", "max") | ("Math", "min") => {
                // `max()`/`min()` with no args are ±Infinity (not representable);
                // require at least one arg and all-static-numeric args.
                let mut nums = Vec::with_capacity(args.len());
                for a in args {
                    match a {
                        Abs::Num(n) => nums.push(*n),
                        _ => return None,
                    }
                }
                let v = if method == "max" {
                    nums.iter().copied().max()?
                } else {
                    nums.iter().copied().min()?
                };
                Some(Abs::Num(v))
            }
            _ => None,
        }
    }

    /// Fold a deterministic `String.prototype` method called on a *static* ASCII
    /// string with static arguments. Returns the folded value, or `None` to
    /// residualize (dynamic args, non-ASCII, or a result this model can't hold).
    ///
    /// Restricting to ASCII makes the fold provably match the JS runtime: char
    /// indices then equal UTF-16 code-unit indices, ASCII case mapping equals
    /// JS's, and the ASCII whitespace set is exactly JS's minus characters the
    /// guard already excludes. `split` (which builds an array) is handled by the
    /// caller, which has `&mut State` to allocate. Validated by the Node oracle.
    fn try_string_method(&self, recv: &str, method: &str, args: &[Abs], s: &State) -> Option<Abs> {
        if !recv.is_ascii() {
            return None;
        }
        let chars: Vec<char> = recv.chars().collect();
        let n = chars.len() as i64;
        // JS `ToInteger` of argument `i`, or `default` if absent.
        let int_arg = |i: usize, default: i64| -> Option<i64> {
            match args.get(i) {
                None => Some(default),
                Some(a) => self.to_number_static(s, a),
            }
        };
        // A static ASCII string argument `i`.
        let str_arg = |i: usize| -> Option<String> {
            let st = self.to_string_static(s, args.get(i)?)?;
            st.is_ascii().then_some(st)
        };
        // JS `slice` clamp: negative counts from the end, then bound to [0, n].
        let slice_idx = |x: i64| -> usize {
            if x < 0 { (n + x).max(0) as usize } else { x.min(n) as usize }
        };
        // JS `substring`/clamp to [0, n].
        let clamp0 = |x: i64| -> usize { x.max(0).min(n) as usize };

        Some(match method {
            "charAt" => {
                let p = int_arg(0, 0)?;
                Abs::Str(if p >= 0 && p < n { chars[p as usize].to_string() } else { String::new() })
            }
            "at" => {
                let mut p = int_arg(0, 0)?;
                if p < 0 {
                    p += n;
                }
                if p >= 0 && p < n {
                    Abs::Str(chars[p as usize].to_string())
                } else {
                    Abs::Undef
                }
            }
            "charCodeAt" | "codePointAt" => {
                let p = int_arg(0, 0)?;
                if p >= 0 && p < n {
                    Abs::Num(chars[p as usize] as i64)
                } else {
                    return None; // out of range -> NaN / undefined: residualize
                }
            }
            "slice" => {
                let from = slice_idx(int_arg(0, 0)?);
                let to = slice_idx(int_arg(1, n)?);
                Abs::Str(if to > from { chars[from..to].iter().collect() } else { String::new() })
            }
            "substring" => {
                let a = clamp0(int_arg(0, 0)?);
                let b = clamp0(int_arg(1, n)?);
                let (from, to) = (a.min(b), a.max(b));
                Abs::Str(chars[from..to].iter().collect())
            }
            "toUpperCase" | "toLocaleUpperCase" => Abs::Str(recv.to_ascii_uppercase()),
            "toLowerCase" | "toLocaleLowerCase" => Abs::Str(recv.to_ascii_lowercase()),
            "trim" => Abs::Str(recv.trim_matches(is_js_ascii_ws).to_string()),
            "trimStart" => Abs::Str(recv.trim_start_matches(is_js_ascii_ws).to_string()),
            "trimEnd" => Abs::Str(recv.trim_end_matches(is_js_ascii_ws).to_string()),
            "indexOf" => {
                let needle = str_arg(0)?;
                let from = int_arg(1, 0)?.max(0).min(n) as usize;
                let hay: String = chars[from..].iter().collect();
                Abs::Num(match hay.find(&needle) {
                    // `find` is a byte offset; ASCII byte offset == char offset.
                    Some(b) => (from + b) as i64,
                    None => -1,
                })
            }
            "lastIndexOf" => {
                let needle = str_arg(0)?;
                Abs::Num(recv.rfind(&needle).map(|b| b as i64).unwrap_or(-1))
            }
            "includes" => Abs::Bool(recv.contains(&str_arg(0)?)),
            "startsWith" => Abs::Bool(recv.starts_with(&str_arg(0)?)),
            "endsWith" => Abs::Bool(recv.ends_with(&str_arg(0)?)),
            "repeat" => {
                let c = int_arg(0, 0)?;
                if c < 0 {
                    return None; // RangeError at runtime: residualize so it throws
                }
                Abs::Str(recv.repeat(c as usize))
            }
            "concat" => {
                let mut out = recv.to_string();
                for a in args {
                    out.push_str(&self.to_string_static(s, a)?);
                }
                Abs::Str(out)
            }
            _ => return None,
        })
    }

    /// Fold `"...".split(sep)` on a static ASCII string into a freshly allocated
    /// static array of substrings. `None` for dynamic/non-ASCII args (residualize).
    /// Matches JS: a string separator splits; `""` splits into characters; an
    /// absent separator yields a one-element array of the whole string.
    fn try_string_split(&self, recv: &str, args: &[Abs], s: &mut State) -> Option<Abs> {
        if !recv.is_ascii() {
            return None;
        }
        let parts: Vec<String> = match args.first() {
            // `"x".split()` -> ["x"] (separator undefined)
            None | Some(Abs::Undef) => vec![recv.to_string()],
            Some(a) => {
                let sep = self.to_string_static(s, a)?;
                if !sep.is_ascii() {
                    return None;
                }
                if sep.is_empty() {
                    recv.chars().map(|c| c.to_string()).collect()
                } else {
                    recv.split(&sep).map(|p| p.to_string()).collect()
                }
            }
        };
        let elems: Vec<Abs> = parts.into_iter().map(Abs::Str).collect();
        let addr = s.alloc(HeapObj::Array(elems));
        Some(Abs::Ref(addr))
    }

    /// A static byte vector from a numeric array (or a string's bytes); `None`
    /// when the value is dynamic or not byte-like.
    fn static_bytes(&self, arg: Option<&Abs>, s: &State) -> Option<Vec<u8>> {
        match arg {
            Some(Abs::Ref(addr)) => match &s.heap[addr] {
                HeapObj::Array(elems) => {
                    let mut out = Vec::with_capacity(elems.len());
                    for e in elems {
                        match e {
                            Abs::Num(n) => out.push((n & 0xff) as u8),
                            _ => return None,
                        }
                    }
                    Some(out)
                }
                _ => None,
            },
            Some(Abs::Str(st)) => Some(st.clone().into_bytes()),
            _ => None,
        }
    }

    fn do_call(&self, s: &mut State, nargs: usize, pc: usize, out: &mut Vec<Op>) -> Step<Js> {
        let base = s.top().ostack.len() - nargs;
        let args: Vec<Abs> = s.top_mut().ostack.split_off(base);
        let callee = s.top_mut().ostack.pop().unwrap();

        // A pending method-access callee that the freeze pass snapshotted (its
        // receiver was preserved as a `BoundMethod`): emit `func.call(recv, args)`.
        // This invokes the snapshotted method `func` with the snapshotted receiver
        // `recv` as `this` — the universal "uncurry this" form, correct even when
        // the method is itself `.call`/`.apply` (`(f.call).call(f, a, b)` ===
        // `f.call(a, b)`). Both `func` and `recv` are already frozen temps/atoms,
        // so neither needs escaping.
        if let Abs::Dyn(RExpr::BoundMethod { func, recv }) = &callee {
            let callee_fn = RExpr::Get(Box::new((**func).clone()), "call".to_string());
            let recv = (**recv).clone();
            self.forbid_residual_in_try(s, "an unmodeled call");
            self.freeze_before_opaque_call(s, pc, out);
            let mut escaped = std::collections::HashSet::new();
            let mut call_args = Vec::with_capacity(args.len() + 1);
            call_args.push(recv);
            for a in &args {
                call_args.push(self.operand_rexpr(s, a, &mut escaped, out));
            }
            let dst = Js::opaque_call_var(pc);
            out.push(Op::Eval { dst, expr: RExpr::Call(Box::new(callee_fn), call_args) });
            s.top_mut().pc += 1;
            s.top_mut().ostack.push(Abs::Dyn(RExpr::Var(dst)));
            return Step::Continue;
        }

        // A modeled built-in bound method (e.g. `TextDecoder.decode`): fold to a
        // constant when its arguments are static.
        if let Abs::Ref(addr) = &callee {
            if let HeapObj::Builtin { kind, data } = &s.heap[addr] {
                let (kind, data) = (kind.clone(), data.clone());
                // Static inputs: fold to a constant. Otherwise residualize the
                // built-in as a runtime call (e.g. `new TextDecoder().decode(x)`),
                // so a dynamic argument is handled rather than rejected.
                if let Some(v) = self.try_builtin_method(&kind, &data, &args, s) {
                    s.top_mut().pc += 1;
                    s.top_mut().ostack.push(v);
                    return Step::Continue;
                }
                let (base, method) = kind
                    .split_once('.')
                    .unwrap_or_else(|| panic!("built-in method without a receiver: {kind}"));
                self.forbid_residual_in_try(s, "a built-in call");
                self.freeze_before_opaque_call(s, pc, out);
                let mut escaped = std::collections::HashSet::new();
                let arg_rs: Vec<RExpr> =
                    args.iter().map(|a| self.operand_rexpr(s, a, &mut escaped, out)).collect();
                let recv = RExpr::New(Box::new(RExpr::Global(base.to_string())), vec![]);
                let dst = Js::opaque_call_var(pc);
                out.push(Op::Eval {
                    dst,
                    expr: RExpr::Call(
                        Box::new(RExpr::Get(Box::new(recv), method.to_string())),
                        arg_rs,
                    ),
                });
                s.top_mut().pc += 1;
                s.top_mut().ostack.push(Abs::Dyn(RExpr::Var(dst)));
                return Step::Continue;
            }
        }
        // A modeled static method on a global (e.g. `String.fromCharCode`): fold
        // when static, otherwise fall through to a pass-through call.
        if let Abs::Dyn(RExpr::Get(obj, method)) = &callee {
            if let RExpr::Global(g) = &**obj {
                if let Some(v) = self.try_builtin_static(g, method, &args) {
                    s.top_mut().pc += 1;
                    s.top_mut().ostack.push(v);
                    return Step::Continue;
                }
            }
            // A `String.prototype` method on a static string literal receiver
            // (`"abc".charAt(0)`, `"a,b".split(",")`, ...): the receiver survives
            // as `RExpr::Str` because `get_prop` residualizes primitive reads to
            // `base.key`. Fold deterministic methods so deobfuscated/JSFuck string
            // pipelines collapse to literals.
            if let RExpr::Str(recv) = &**obj {
                let recv = recv.clone();
                let method = method.clone();
                if method == "split" {
                    if let Some(v) = self.try_string_split(&recv, &args, s) {
                        s.top_mut().pc += 1;
                        s.top_mut().ostack.push(v);
                        return Step::Continue;
                    }
                } else if let Some(v) = self.try_string_method(&recv, &method, &args, s) {
                    s.top_mut().pc += 1;
                    s.top_mut().ostack.push(v);
                    return Step::Continue;
                }
            }
        }

        let (fid, captured) = match &callee {
            Abs::Ref(addr) if matches!(&s.heap[addr], HeapObj::Closure { .. }) => {
                match &s.heap[addr] {
                    HeapObj::Closure { fid, captured } => (*fid, captured.clone()),
                    _ => unreachable!(),
                }
            }
            // Any other callee passes the call through to the residual:
            //   - a dynamic (unmodeled) callee — an opaque, possibly effectful
            //     call;
            //   - a non-callable value (a non-closure heap object like `({})()`,
            //     or a primitive like `(5)()` / `undefined()`), which throws
            //     `TypeError` at runtime — the residual throws identically.
            // The call is bound to a temp once, in program order, so it is never
            // duplicated; the callee and any heap-object arguments escape (the
            // callee may read or mutate them).
            _ => {
                self.forbid_residual_in_try(s, "an unmodeled call");
                self.freeze_before_opaque_call(s, pc, out);
                let callee_r = self.escape_if_ref(s, callee, out).to_rexpr();
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
        };

        // A function that contains a `try` is never inlined: it becomes a
        // residual function and is called at runtime, so a `return` in its body
        // returns from *it*, not from whatever inlined it. (Inlining it and
        // residualizing the inner `try` would turn its `return` into a return
        // from the caller, exiting the caller early.)
        if self.has_try[fid] {
            self.forbid_residual_in_try(s, "a call to a try-containing function");
            self.freeze_before_opaque_call(s, pc, out);
            let mut escaped = std::collections::HashSet::new();
            // The residual function holds its captures *by reference* and may
            // mutate them when called, so each captured heap object ESCAPES (its
            // references throughout the caller's state are invalidated). Using
            // `materialize_value` here instead would leave the caller's abstract
            // copy live, so a later read would re-materialize the stale,
            // pre-call value — dropping mutations the callee made through the
            // capture. (This is the self-modifying-bytecode bug in simple.js: a
            // loader closure decrypts the captured byte array, but the reader
            // re-materialized the undecrypted literal.)
            let caps: Vec<RExpr> =
                captured.iter().map(|c| self.operand_rexpr(s, c, &mut escaped, out)).collect();
            let rfid = self.residual_fn_for(fid);
            let arg_rs: Vec<RExpr> =
                args.iter().map(|a| self.operand_rexpr(s, a, &mut escaped, out)).collect();
            let dst = Js::opaque_call_var(pc);
            out.push(Op::Eval {
                dst,
                expr: RExpr::Call(Box::new(RExpr::FnRef { rfid, caps }), arg_rs),
            });
            s.top_mut().pc += 1;
            s.top_mut().ostack.push(Abs::Dyn(RExpr::Var(dst)));
            return Step::Continue;
        }

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

        // Arguments are evaluated even when the inlined callee ignores them (an
        // unused parameter, an extra arg reachable only via `arguments`, or a body
        // that folds away entirely). Commit any may-throw argument for effect in
        // program order so its exception is not dropped (seed 10125:
        // `r0(3, input[input[-31]], 8)` with the result dead and `r0` ignoring its
        // args). The opaque-call arms already keep arg throws (they reside in the
        // residual call); only this inline path can drop them.
        let args: Vec<Abs> = args
            .into_iter()
            .enumerate()
            .map(|(i, a)| self.emit_if_maythrow(s, pc, i, a, out))
            .collect();

        let mut locals = vec![Abs::Undef; self.nslots[fid]];
        for (i, c) in captured.into_iter().enumerate() {
            locals[i] = c;
        }
        // If the callee references `arguments`, build it from the actual args
        // (an array-like the body reads via `.length` / `[i]`) before the args
        // are moved into the param slots.
        if let Some(aslot) = self.arguments_slot[fid] {
            let addr = s.alloc(HeapObj::Array(args.clone()));
            if aslot < locals.len() {
                locals[aslot] = Abs::Ref(addr);
            }
        }
        // Only the declared parameters bind to slots; any extra arguments are
        // reachable solely through `arguments` (and must not clobber locals or
        // the `arguments` slot).
        for (i, a) in args.into_iter().enumerate().take(self.nparams[fid]) {
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
    /// the residual.
    ///
    /// Construction is *two-phase* so that cyclic object graphs are sound: every
    /// object/array is first created as an empty **shell**, then its
    /// fields/elements/captures are **filled** in. Because the shells all exist
    /// before any fill runs, a closure can capture a cell that is filled later
    /// (objects are shared by reference), which is exactly the capture-by-
    /// reference boxing pattern (`var c = {value: ...}` captured by a closure
    /// whose own value is stored back into `c`).
    fn materialize_value(&self, v: &Abs, heap: &Heap, out: &mut Vec<Op>) -> RExpr {
        let mut seen = HashMap::new();
        // Acyclic graphs use the compact inline form (`{a: 1}`, `[1, 2]`); only a
        // genuinely cyclic graph needs the verbose two-phase shell+fill form.
        if matches!(v, Abs::Ref(a) if Self::graph_cyclic(heap, *a)) {
            let mut shells = Vec::new();
            let mut fills = Vec::new();
            let r = self.mat_value(v, heap, &mut shells, &mut fills, &mut seen);
            out.append(&mut shells);
            out.append(&mut fills);
            r
        } else {
            self.mat_inline(v, heap, out, &mut seen)
        }
    }

    /// Materialize `v` into the residual variable `dst`.
    fn materialize_into(&self, dst: usize, v: &Abs, heap: &Heap, out: &mut Vec<Op>) {
        let mut seen = HashMap::new();
        match v {
            Abs::Ref(addr) if Self::graph_cyclic(heap, *addr) => {
                seen.insert(*addr, dst); // a cycle back to this object reuses `dst`
                let mut shells = Vec::new();
                let mut fills = Vec::new();
                self.mat_obj(dst, *addr, heap, &mut shells, &mut fills, &mut seen);
                out.append(&mut shells);
                out.append(&mut fills);
            }
            Abs::Ref(addr) => {
                seen.insert(*addr, dst);
                self.mat_inline_obj(dst, *addr, heap, out, &mut seen);
            }
            _ => out.push(Op::Assign { var: dst, expr: v.to_rexpr() }),
        }
    }

    /// Is the heap graph rooted at `start` cyclic? (Then materialization must be
    /// two-phase so a closure can capture a cell whose value is filled later.)
    fn graph_cyclic(heap: &Heap, start: usize) -> bool {
        fn dfs(
            heap: &Heap,
            addr: usize,
            on_path: &mut std::collections::HashSet<usize>,
            done: &mut std::collections::HashSet<usize>,
        ) -> bool {
            if on_path.contains(&addr) {
                return true;
            }
            if !done.insert(addr) {
                return false;
            }
            on_path.insert(addr);
            let kids: Vec<usize> = match &heap[&addr] {
                HeapObj::Object(fs) => fs.iter().filter_map(|(_, v)| as_ref_addr(v)).collect(),
                HeapObj::Array(es) => es.iter().filter_map(as_ref_addr).collect(),
                HeapObj::Closure { captured, .. } => {
                    captured.iter().filter_map(as_ref_addr).collect()
                }
                HeapObj::Builtin { data, .. } => data.iter().filter_map(as_ref_addr).collect(),
            };
            for k in kids {
                if dfs(heap, k, on_path, done) {
                    return true;
                }
            }
            on_path.remove(&addr);
            false
        }
        let mut on_path = std::collections::HashSet::new();
        let mut done = std::collections::HashSet::new();
        dfs(heap, start, &mut on_path, &mut done)
    }

    /// Inline (compact) materialization for an acyclic graph: nested objects are
    /// constructed depth-first as object/array literals.
    fn mat_inline(
        &self,
        v: &Abs,
        heap: &Heap,
        out: &mut Vec<Op>,
        seen: &mut HashMap<usize, usize>,
    ) -> RExpr {
        match v {
            Abs::Ref(addr) => {
                if let Some(&var) = seen.get(addr) {
                    return RExpr::Var(var);
                }
                let dst = Js::escape_var(*addr);
                seen.insert(*addr, dst);
                self.mat_inline_obj(dst, *addr, heap, out, seen);
                RExpr::Var(dst)
            }
            _ => v.to_rexpr(),
        }
    }

    fn mat_inline_obj(
        &self,
        dst: usize,
        addr: usize,
        heap: &Heap,
        out: &mut Vec<Op>,
        seen: &mut HashMap<usize, usize>,
    ) {
        match &heap[&addr] {
            HeapObj::Object(fields) => {
                let rf: Vec<(String, RExpr)> = fields
                    .clone()
                    .into_iter()
                    .map(|(k, fv)| (k, self.mat_inline(&fv, heap, out, seen)))
                    .collect();
                out.push(Op::NewObject { dst, fields: rf });
            }
            HeapObj::Array(elems) => {
                let re: Vec<RExpr> = elems
                    .clone()
                    .into_iter()
                    .map(|e| self.mat_inline(&e, heap, out, seen))
                    .collect();
                out.push(Op::NewArray { dst, elems: re });
            }
            HeapObj::Closure { fid, captured } => {
                let (fid, captured) = (*fid, captured.clone());
                let caps: Vec<RExpr> =
                    captured.iter().map(|c| self.mat_inline(c, heap, out, seen)).collect();
                let rfid = self.residual_fn_for(fid);
                out.push(Op::Assign { var: dst, expr: RExpr::FnRef { rfid, caps } });
            }
            HeapObj::Builtin { kind, data } => {
                let expr = if let Some((base, method)) = kind.split_once('.') {
                    RExpr::Get(
                        Box::new(RExpr::New(Box::new(RExpr::Global(base.to_string())), vec![])),
                        method.to_string(),
                    )
                } else if data.is_empty() {
                    RExpr::New(Box::new(RExpr::Global(kind.clone())), vec![])
                } else {
                    panic!("cannot residualize built-in `{kind}` with captured state");
                };
                out.push(Op::Assign { var: dst, expr });
            }
        }
    }

    /// Convert an abstract value to a residual expression. `seen` maps each
    /// already-created heap address to the residual variable holding its shell,
    /// so a cyclic or shared graph references the existing variable.
    fn mat_value(
        &self,
        v: &Abs,
        heap: &Heap,
        shells: &mut Vec<Op>,
        fills: &mut Vec<Op>,
        seen: &mut HashMap<usize, usize>,
    ) -> RExpr {
        match v {
            Abs::Ref(addr) => {
                if let Some(&var) = seen.get(addr) {
                    return RExpr::Var(var);
                }
                let dst = Js::escape_var(*addr);
                seen.insert(*addr, dst);
                self.mat_obj(dst, *addr, heap, shells, fills, seen);
                RExpr::Var(dst)
            }
            _ => v.to_rexpr(),
        }
    }

    /// Emit the shell (into `shells`) and fill ops (into `fills`) for the heap
    /// object at `addr`, bound to residual variable `dst`.
    fn mat_obj(
        &self,
        dst: usize,
        addr: usize,
        heap: &Heap,
        shells: &mut Vec<Op>,
        fills: &mut Vec<Op>,
        seen: &mut HashMap<usize, usize>,
    ) {
        match &heap[&addr] {
            HeapObj::Object(fields) => {
                shells.push(Op::NewObject { dst, fields: vec![] });
                for (k, fv) in fields.clone() {
                    let rv = self.mat_value(&fv, heap, shells, fills, seen);
                    fills.push(Op::SetProp { obj: RExpr::Var(dst), key: k, val: rv });
                }
            }
            HeapObj::Array(elems) => {
                shells.push(Op::NewArray { dst, elems: vec![] });
                for (i, e) in elems.clone().into_iter().enumerate() {
                    let rv = self.mat_value(&e, heap, shells, fills, seen);
                    fills.push(Op::SetIndex {
                        arr: RExpr::Var(dst),
                        index: RExpr::Num(i as i64),
                        val: rv,
                    });
                }
            }
            // A closure that escapes becomes a generated residual function,
            // referenced with its captured values bound. Emitted in the fill
            // phase so the cells it captures already exist as shells.
            HeapObj::Closure { fid, captured } => {
                let (fid, captured) = (*fid, captured.clone());
                let caps: Vec<RExpr> = captured
                    .iter()
                    .map(|c| self.mat_value(c, heap, shells, fills, seen))
                    .collect();
                let rfid = self.residual_fn_for(fid);
                fills.push(Op::Assign { var: dst, expr: RExpr::FnRef { rfid, caps } });
            }
            // A modeled built-in escaping into runtime code is reconstructed as
            // its runtime form (`new TextDecoder()`, a bound method, ...).
            HeapObj::Builtin { kind, data } => {
                let expr = if let Some((base, method)) = kind.split_once('.') {
                    RExpr::Get(
                        Box::new(RExpr::New(Box::new(RExpr::Global(base.to_string())), vec![])),
                        method.to_string(),
                    )
                } else if data.is_empty() {
                    RExpr::New(Box::new(RExpr::Global(kind.clone())), vec![])
                } else {
                    panic!("cannot residualize built-in `{kind}` with captured state");
                };
                fills.push(Op::Assign { var: dst, expr });
            }
        }
    }

    fn escape_var(addr: usize) -> usize {
        // Disjoint from the input var (0) and the loop/if materialization band.
        100_000 + addr
    }

    /// If `v` is a live heap reference, escape it into the residual and return
    /// the resulting `Dyn` reference; otherwise return `v` unchanged. Used where
    /// an object reaches a context we don't model statically (a primitive
    /// operator operand, a mismatched index) and must become a runtime value.
    fn escape_if_ref(&self, s: &mut State, v: Abs, out: &mut Vec<Op>) -> Abs {
        if let Abs::Ref(addr) = v {
            let mut escaped = std::collections::HashSet::new();
            let r = self.escape(s, addr, &mut escaped, out);
            Abs::Dyn(r)
        } else {
            v
        }
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
        self.forbid_residual_in_try(s, "an object escaping into unmodeled code");
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
        s.heap = s
            .heap
            .iter()
            .map(|(addr, h)| {
                let mut h = h.clone();
                match &mut h {
                    HeapObj::Object(fs) => fs.iter_mut().for_each(|(_, v)| invalidate_abs(v, &reach)),
                    HeapObj::Array(es) => es.iter_mut().for_each(|v| invalidate_abs(v, &reach)),
                    HeapObj::Closure { captured, .. } => {
                        captured.iter_mut().for_each(|v| invalidate_abs(v, &reach))
                    }
                    HeapObj::Builtin { data, .. } => {
                        data.iter_mut().for_each(|v| invalidate_abs(v, &reach))
                    }
                }
                (*addr, h)
            })
            .collect();
        escaped.extend(reach);
        root
    }

    /// Heap addresses transitively reachable from `addr` (terminates on shared
    /// or cyclic graphs via the visited set).
    fn reachable(heap: &Heap, addr: usize, out: &mut std::collections::HashSet<usize>) {
        if !out.insert(addr) {
            return;
        }
        let kids: Vec<usize> = match &heap[&addr] {
            HeapObj::Object(fs) => fs.iter().filter_map(|(_, v)| as_ref_addr(v)).collect(),
            HeapObj::Array(es) => es.iter().filter_map(as_ref_addr).collect(),
            HeapObj::Closure { captured, .. } => captured.iter().filter_map(as_ref_addr).collect(),
            HeapObj::Builtin { data, .. } => data.iter().filter_map(as_ref_addr).collect(),
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

    /// At an array/object literal, emit each may-throw element for effect in
    /// left-to-right order (single-evaluated into a stable temp, the element then
    /// reading back as that temp). JS evaluates literal elements eagerly at the
    /// literal, so the exception must be raised there — otherwise a literal that
    /// folds (`[x, a.b].length`) or is discarded (a dead `var a = {k: a.b}`) drops
    /// the throw the original raises. A static element (no `Dyn`) cannot throw and
    /// is left untouched, so fully-static literals still fold.
    ///
    /// Known gap: a may-throw element that precedes a *separately-effectful*
    /// element in the same literal (`[a.b, sideEffectCall()]`) is emitted after
    /// that effect rather than before it (the effectful element was emitted during
    /// its own evaluation, before this runs). Correct element-eval-order emission
    /// would require a definitely-evaluated-context flag in the lowering; no test
    /// hits this today.
    fn emit_maythrow_elems(&self, s: &mut State, pc: usize, vals: Vec<Abs>, out: &mut Vec<Op>) -> Vec<Abs> {
        vals.into_iter()
            .enumerate()
            .map(|(i, v)| self.emit_if_maythrow(s, pc, i, v, out))
            .collect()
    }

    /// If `v` is a may-throw residual value, emit it for effect (single-evaluated
    /// into a stable temp keyed by (pc, i)) and return the temp; otherwise return
    /// `v` unchanged. Used wherever a value is *stored into the abstract heap*
    /// (array/object element, scalar-replaced field/index write): the heap slot
    /// may never be materialized (a dead object), so the exception the read
    /// raises in the original must be committed here, at the write.
    fn emit_if_maythrow(&self, s: &mut State, pc: usize, i: usize, v: Abs, out: &mut Vec<Op>) -> Abs {
        if matches!(&v, Abs::Dyn(e) if may_throw(e)) {
            self.forbid_residual_in_try(s, "a may-throw heap write/element");
            let dst = Js::elem_var(pc, i);
            out.push(Op::Eval { dst, expr: v.to_rexpr() });
            Abs::Dyn(RExpr::Var(dst))
        } else {
            v
        }
    }

    /// Stable residual variable for the result of an unmodeled call at this
    /// bytecode pc. Keyed by pc so a call in a loop reuses one temp (reassigned
    /// each iteration) and specialization converges. Disjoint from the other
    /// id bands.
    fn opaque_call_var(pc: usize) -> usize {
        500_000 + pc
    }

    /// Stable residual temp for a postfix-update snapshot (`Instr::Snapshot`).
    /// Keyed by pc so a snapshot in a loop reuses one temp and converges.
    fn snapshot_var(pc: usize) -> usize {
        600_000 + pc
    }

    /// Stable residual temp for a discarded-but-may-throw expression statement
    /// (`Instr::Pop`). Keyed by pc so a discard in a loop reuses one temp.
    fn discard_var(pc: usize) -> usize {
        700_000 + pc
    }

    /// Stable residual temp for a may-throw element of an array/object literal
    /// (`Instr::NewArray`/`NewObject`), single-evaluated at construction so its
    /// exception is preserved even when the literal is discarded or folds. Keyed
    /// by (pc, element index); placed well above the `freeze_var` band so it
    /// cannot collide (pc is far below 1e6 in practice, element index below 256).
    fn elem_var(pc: usize, i: usize) -> usize {
        1_000_000_000 + pc * 256 + i
    }

    /// Stable residual temp for freezing a stale reader at a mutation site.
    /// Keyed by (pc, idx) where idx identifies the local slot or operand-stack
    /// position, so it is deterministic and distinct per frozen value.
    fn freeze_var(pc: usize, idx: usize) -> usize {
        2_000_000 + pc * 8192 + idx
    }

    /// Before a residual write to some location, freeze every live value (in the
    /// frame's locals and operand stack) whose residual expression *reads* that
    /// location, binding it to a temp holding the pre-write value. Without this,
    /// such a value would re-read the mutated location at runtime and observe the
    /// new value (a single-evaluation/ordering bug), e.g. `t = a[i]; i--; use t`
    /// or `arr[i-2].push(arr[--i])`.
    /// Before an opaque call (which may mutate any heap location), snapshot every
    /// live frame value whose residual expression *reads* mutable state
    /// (`obj.k`, `arr[i]`, a global) into a temp holding the pre-call value. A
    /// `Dyn` value left as such an expression would otherwise be re-read after
    /// the call and see post-call state — the single-evaluation bug, e.g. a VM
    /// operand `stack[sp]` read once, then used after intervening calls moved
    /// `sp` or mutated `stack`.
    fn freeze_before_opaque_call(&self, s: &mut State, pc: usize, out: &mut Vec<Op>) {
        self.freeze_readers(s, pc, out, &|e| {
            matches!(e, RExpr::Get(..) | RExpr::Index(..) | RExpr::Global(..))
        });
    }

    fn freeze_readers(&self, s: &mut State, pc: usize, out: &mut Vec<Op>, reads: &dyn Fn(&RExpr) -> bool) {
        let fi = s.frames.len() - 1;
        for slot in 0..s.frames[fi].locals.len() {
            if let Abs::Dyn(e) = &s.frames[fi].locals[slot] {
                if !rexpr_is_atom(e) && rexpr_contains(e, reads) {
                    let expr = e.clone();
                    let dst = Js::freeze_var(pc, slot);
                    out.push(Op::Eval { dst, expr });
                    s.frames[fi].locals[slot] = Abs::Dyn(RExpr::Var(dst));
                }
            }
        }
        for i in 0..s.frames[fi].ostack.len() {
            if let Abs::Dyn(e) = &s.frames[fi].ostack[i] {
                if !rexpr_is_atom(e) && rexpr_contains(e, reads) {
                    let expr = e.clone();
                    let dst = Js::freeze_var(pc, 4096 + i);
                    // If this operand is a pending method-access callee
                    // (`recv.m` / `recv[i]`) with an *atom* receiver, snapshot the
                    // method VALUE to the temp (a value use then sees the same
                    // pre-call snapshot a plain freeze would give) but keep the
                    // receiver in a `BoundMethod` so a later call preserves `this`.
                    // Freezing it to a bare `Var` would detach the receiver and the
                    // call would run with `this === undefined` (over-throw for
                    // `.call`/`.apply`, wrong `this` otherwise). The atom-receiver
                    // restriction keeps the receiver stable without a second temp
                    // (the `freeze_var` id space has no third band); a non-atom
                    // receiver falls back to the plain freeze.
                    let bound_recv = match &expr {
                        RExpr::Get(base, _) | RExpr::Index(base, _) if rexpr_is_atom(base) => {
                            Some((**base).clone())
                        }
                        _ => None,
                    };
                    out.push(Op::Eval { dst, expr });
                    s.frames[fi].ostack[i] = match bound_recv {
                        Some(recv) => Abs::Dyn(RExpr::BoundMethod {
                            func: Box::new(RExpr::Var(dst)),
                            recv: Box::new(recv),
                        }),
                        None => Abs::Dyn(RExpr::Var(dst)),
                    };
                }
            }
        }
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
                    // A heap ref as an operand would have to escape (which a
                    // read-only probe cannot do); treat the condition as
                    // dynamically controlled.
                    if matches!(a, Abs::Ref(_)) || matches!(b, Abs::Ref(_)) {
                        return true;
                    }
                    probe.top_mut().ostack.push(Js::eval_bin(*op, &a, &b));
                }
                Instr::GetProp(k) => {
                    let o = probe.top_mut().ostack.pop().unwrap();
                    match self.get_prop_opt(&probe, &o, k) {
                        Some(v) => probe.top_mut().ostack.push(v),
                        None => return true, // unmodeled: conservatively dynamic
                    }
                }
                Instr::GetIndex => {
                    let i = probe.top_mut().ostack.pop().unwrap();
                    let a = probe.top_mut().ostack.pop().unwrap();
                    // A read the abstract heap can't serve statically would need
                    // to escape (impossible in a read-only probe): treat the
                    // loop as dynamically controlled.
                    if !self.index_is_static(&probe, &a, &i) {
                        return true;
                    }
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
        // Block boundary: reclaim heap garbage so the abstract heap stays bounded
        // by the live set (otherwise long static unrolling accumulates dead
        // objects forever, which is what made `simple.js` blow up).
        ns.gc();
        Step::Jump(ns)
    }

    fn materialize(&self, ns: &mut State, slots: &[usize], out: &mut Vec<Op>) {
        if !slots.is_empty() {
            self.forbid_residual_in_try(ns, "a value materializing (a dynamic loop/branch)");
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

    /// A residual operation inside an active `try` would, at runtime, be able to
    /// throw past our goto-based catch. The body must therefore residualize a
    /// real `try`/`catch`. During the foldability probe we only *record* that
    /// this happened (`try_taint`); the real run never reaches here on the fold
    /// path (the probe already steered it to `residualize_try`).
    fn forbid_residual_in_try(&self, s: &State, _what: &str) {
        if !s.handlers.is_empty() {
            if self.probing.get() {
                self.try_taint.set(true);
                return;
            }
            panic!(
                "residual operation inside an active try/catch reached the inline \
                 path; this should have been residualized (probe bug)"
            );
        }
    }

    /// The residual variable an emitted `catch` binds its exception to.
    fn exc_var(catch_pc: usize) -> usize {
        800_000 + catch_pc
    }

    /// Carve a single-frame state out of the top frame at `pc`, for bounded
    /// sub-specialization of a try region (the region only touches this frame,
    /// its callees, the shared heap, and globals). A `return` empties the frame
    /// and ends the sub-program; `halt_at` ends it at the region boundary.
    fn sole_frame_state(&self, s: &State, pc: usize) -> State {
        let mut f = s.top().clone();
        f.pc = pc;
        f.ostack.clear();
        State {
            frames: vec![f],
            heap: s.heap.clone(),
            next_addr: s.next_addr,
            pending_joins: Vec::new(),
            handlers: Vec::new(),
        }
    }

    /// Foldability probe: re-specialize the try region (body with throws modeled
    /// to the catch) on a clone, bounded at `end`. If a may-throw residual op
    /// would be emitted, `try_taint` is set. Returns true when the region folds
    /// (no runtime `try` needed).
    fn try_folds(&self, s: &State, handler: &Handler, _body_end: usize, end: usize) -> bool {
        let body_pc = s.top().pc + 1;
        let mut probe = self.sole_frame_state(s, body_pc);
        probe.handlers.push(Handler {
            catch_pc: handler.catch_pc,
            exc_slot: handler.exc_slot,
            frame_depth: 1,
            ostack_depth: 0,
        });
        self.probing.set(true);
        self.try_taint.set(false);
        self.halt_at.borrow_mut().push(end);
        let _ = crate::engine::specialize(self, probe);
        self.halt_at.borrow_mut().pop();
        self.probing.set(false);
        !self.try_taint.get()
    }

    /// Emit a residual `try`/`catch` for a body that can't fully fold. Everything
    /// the region can touch is first materialized to residual variables so the
    /// continuation sees the (now runtime) state; the body and catch are then
    /// specialized as nested programs over that fully-residual state.
    fn residualize_try(
        &self,
        s: &mut State,
        catch_pc: usize,
        exc_slot: Option<usize>,
        body_end: usize,
        end: usize,
        out: &mut Vec<Op>,
    ) -> Step<Js> {
        // A `continue`/`break` inside this residual `try` that targets an
        // enclosing loop makes the nested specialization below jump back to the
        // loop head and re-enter this same `try`, recursing without bound. Detect
        // that re-entry and fail with a clear, catchable error rather than
        // overflowing the stack. (The sound complete fix is to thread the
        // non-local exit out of the residual sub-program; not yet done.)
        if self.residual_try_stack.borrow().contains(&catch_pc) {
            panic!(
                "`continue`/`break` out of a residualized `try`/`catch` back into an \
                 enclosing loop is not yet supported (would not terminate)"
            );
        }
        self.residual_try_stack.borrow_mut().push(catch_pc);

        self.materialize_all(s, out);
        // Body/catch run with eager top-frame stores so their effects residualize
        // in source order (sound under may-throw ops).
        let prev_eager = self.eager_stores.replace(true);
        // Body: no handler (a runtime throw is caught by the emitted `catch`),
        // bounded at the `PopHandler` (normal completion).
        let body_pc = s.top().pc + 1;
        let body_state = self.sole_frame_state(s, body_pc);
        self.halt_at.borrow_mut().push(body_end);
        let body = crate::engine::specialize(self, body_state);
        self.halt_at.borrow_mut().pop();
        // Catch: the exception is a runtime value bound to a residual variable.
        let ev = Js::exc_var(catch_pc);
        let mut catch_state = self.sole_frame_state(s, catch_pc);
        if let Some(slot) = exc_slot {
            catch_state.frames[0].locals[slot] = Abs::Dyn(RExpr::Var(ev));
        }
        self.halt_at.borrow_mut().push(end);
        let catch_body = crate::engine::specialize(self, catch_state);
        self.halt_at.borrow_mut().pop();
        self.eager_stores.set(prev_eager);
        self.residual_try_stack.borrow_mut().pop();
        out.push(Op::Try { body, catch_slot: exc_slot.map(|_| ev), catch_body });
        s.top_mut().pc = end;
        Step::Continue
    }

    /// Materialize the whole top frame to its *stable* residual variables: escape
    /// every heap object it can reach (so aliases collapse to one residual var),
    /// then lift EVERY slot (not just static ones) to its `stable_id`. Lifting
    /// every slot — including already-dynamic ones — is what makes a residual
    /// `try` sound: the body/catch sub-programs and the continuation then all name
    /// a given slot the same way, so a modification the body writes back at its
    /// boundary (see the `halt_at` materialize in `step`) is read by the
    /// continuation. After this the frame is entirely dynamic.
    fn materialize_all(&self, s: &mut State, out: &mut Vec<Op>) {
        let fi = s.frames.len() - 1;
        let mut escaped = std::collections::HashSet::new();
        for slot in 0..s.frames[fi].locals.len() {
            if let Abs::Ref(addr) = s.frames[fi].locals[slot] {
                let _ = self.escape(s, addr, &mut escaped, out);
            }
        }
        let slots: Vec<usize> = (0..s.frames[fi].locals.len()).collect();
        self.materialize(s, &slots, out);
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
                Bop::BitAnd => JsVal::Num(((*x as i32) & (*y as i32)) as i64),
                Bop::BitOr => JsVal::Num(((*x as i32) | (*y as i32)) as i64),
                Bop::BitXor => JsVal::Num(((*x as i32) ^ (*y as i32)) as i64),
                Bop::Shl => JsVal::Num(((*x as i32) << ((*y as u32) & 31)) as i64),
                Bop::Shr => JsVal::Num(((*x as i32) >> ((*y as u32) & 31)) as i64),
                Bop::UShr => JsVal::Num(((*x as u32) >> ((*y as u32) & 31)) as i64),
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
                Instr::Snapshot => {} // identity for concrete execution
                Instr::Store(slot) => locals[*slot] = ostack.pop().unwrap(),
                Instr::Pop => {
                    ostack.pop();
                }
                Instr::Dup => {
                    let v = ostack.last().cloned().unwrap();
                    ostack.push(v);
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
                Instr::SetGlobalOp(name) => panic!(
                    "the Rust reference interpreter cannot assign the global `{name}`; \
                     validate with the Node oracle"
                ),
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
                    Op::AssignGlobal { name, .. } => panic!(
                        "the Rust residual interpreter cannot assign the global `{name}`; \
                         validate with the Node oracle"
                    ),
                    Op::Try { .. } => panic!(
                        "the Rust residual interpreter does not execute `try`/`catch`; \
                         validate residual exceptions with the Node oracle"
                    ),
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
                    Op::AssignGlobal { name, expr } => {
                        writeln!(s, "    {name} := {}", fmt_rexpr(expr)).unwrap()
                    }
                    Op::Try { body, catch_slot, catch_body } => {
                        let cs = catch_slot.map(|v| format!("v{v}")).unwrap_or_default();
                        writeln!(s, "    try {{ {} blocks }} catch ({cs}) {{ {} blocks }}",
                            body.blocks.len(), catch_body.blocks.len()).unwrap()
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

/// Guard the soundness boundary of the v1 exception model: a `try` body is only
/// handled when it fully specializes (modeled `throw` becomes control flow). A
/// residual operation that could throw at runtime would escape the goto-based
/// catch, so emitting one while a handler is active is a hard error for now.
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
        RExpr::Get(..) | RExpr::Call(..) | RExpr::New(..) | RExpr::FnRef { .. } | RExpr::BoundMethod { .. } => {
            panic!(
                "the Rust residual interpreter cannot evaluate an unmodeled property \
                 read / call / new / residual function / bound method; validate with the Node oracle"
            )
        }
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
        RExpr::Global(_)
            | RExpr::Var(_)
            | RExpr::Get(..)
            | RExpr::Call(..)
            | RExpr::This
            // Decays to its `func` (always a frozen `Var`), so it renders as a
            // ref-like primary — never needs wrapping as a member base.
            | RExpr::BoundMethod { .. }
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
        // A bound method decays to its (snapshotted) function value when not in
        // call position; only `do_call` reattaches the receiver.
        RExpr::BoundMethod { func, .. } => fmt_rexpr(func),
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

/// Sentinel residual-variable id for a residualized function's `arguments`.
/// A residual function is emitted as a real JS function, so its body can read
/// the native `arguments` object: the codegen renders this id as `arguments`
/// rather than `v<id>`. (Correct for capture-free functions, where the emitted
/// params are exactly the original call args.)
pub const ARGUMENTS_VAR_ID: usize = usize::MAX;

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
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
            body: vec![Stmt::Return(bin(Bop::Mul, var(0), num(2)))],
        };
        let map = FuncDef {
            name: "map",
            nslots: 4,
            ncaptured: 0,
            nparams: 2,
            slot_names: vec!["xs", "f", "result", "i"],
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
            body: vec![Stmt::Return(bin(Bop::Gt, var(0), num(0)))],
        };
        let filter = FuncDef {
            name: "filter",
            nslots: 4,
            ncaptured: 0,
            nparams: 2,
            slot_names: vec!["xs", "pred", "result", "i"],
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
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
            arguments_slot: None,
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
