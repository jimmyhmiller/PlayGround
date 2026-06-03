//! Generative differential fuzzer for the JS partial evaluator.
//!
//! Generates random programs in the lowerable subset (each defining
//! `function main(input)`), runs them through the partial evaluator to get a
//! residual, and checks the residual is *observationally equivalent* to the
//! original under Node (the same `assert_node_equiv` oracle the unit tests use,
//! batched). Any disagreement is a real specialization bug. Failing cases are
//! shrunk to a minimal reproducer.
//!
//!   cargo run -p js-frontend --release --bin fuzz -- [--seed N] [--count N]
//!                                                     [--batch N] [--no-shrink]
//!                                                     [--repro SEED]
//!
//! The generator is terminating by construction: loops are bounded (a static
//! `for` count, or a `while` over a reserved decrementing guard) and the call
//! graph is acyclic (a function may only call functions defined before it), so
//! the only nontermination the oracle should ever see is a partial-evaluator
//! bug. Node also runs every program under a wall-clock timeout as a backstop.

use std::fmt::Write as _;
use std::io::Write as _;
use std::panic;
use std::process::Command;

// ----------------------------------------------------------------------------
// PRNG: SplitMix64. Deterministic, seedable, no external crates.
// ----------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, n)` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn chance(&mut self, num: u32, den: u32) -> bool {
        (self.next_u64() % den as u64) < num as u64
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
    /// A small integer literal, biased toward the interesting small range.
    fn small_int(&mut self) -> i64 {
        const POOL: [i64; 14] = [0, 1, 2, 3, -1, -2, 5, 7, 8, 10, 16, 31, 32, 100];
        *self.pick(&POOL)
    }
}

// ----------------------------------------------------------------------------
// AST for the generated subset.
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Expr {
    Num(i64),
    Str(String),
    Bool(bool),
    Null,
    Var(String),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    Un(&'static str, Box<Expr>),
    Logical(&'static str, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Array(Vec<Expr>),
    Object(Vec<(String, Expr)>),
    Index(Box<Expr>, Box<Expr>),
    Member(Box<Expr>, String),
    /// Call a user function (by name) or a whitelisted deterministic builtin.
    Call(String, Vec<Expr>),
    /// `obj.method(args)` for a small deterministic builtin-method set.
    Method(Box<Expr>, &'static str, Vec<Expr>),
    /// `x++`, `++x`, `x--`, `--x` (target is always a plain variable).
    Update { var: String, prefix: bool, inc: bool },
    /// A function expression `function (params) { body; return ret; }`. Its body
    /// may reference (capture) and mutate outer variables — the case that
    /// stresses escape / capture-by-reference / effect ordering.
    Closure { params: Vec<String>, body: Vec<Stmt>, ret: Box<Expr> },
}

#[derive(Clone, Debug)]
enum Stmt {
    Var(String, Expr),
    AssignVar(String, Expr),
    AssignIndex(Expr, Expr, Expr),
    AssignMember(Expr, String, Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    /// `for (var I = 0; I < count; I = I + 1) { body }`
    For { var: String, count: i64, body: Vec<Stmt> },
    /// `var G = count; while (G > 0) { G = G - 1; body }` (guard reserved).
    While { guard: String, count: i64, body: Vec<Stmt> },
    Switch(Expr, Vec<(Option<i64>, Vec<Stmt>)>),
    TryCatch(Vec<Stmt>, String, Vec<Stmt>),
    Throw(Expr),
    Return(Expr),
    Break,
    Continue,
    Push(String, Expr),
    ExprStmt(Expr),
}

#[derive(Clone, Debug)]
struct Func {
    name: String,
    params: Vec<String>,
    body: Vec<Stmt>,
    ret: Expr,
}

#[derive(Clone, Debug)]
struct Prog {
    funcs: Vec<Func>,
    main_body: Vec<Stmt>,
    ret: Expr,
}

// ----------------------------------------------------------------------------
// Generator.
// ----------------------------------------------------------------------------

/// Deterministic, side-effect-free builtins it is safe to fold or pass through
/// (never time/IO/random, which would break observational equivalence).
const SAFE_CALLS: &[(&str, usize)] = &[
    ("Math.floor", 1),
    ("Math.abs", 1),
    ("Math.max", 2),
    ("Math.min", 2),
    ("Math.sign", 1),
    ("Math.trunc", 1),
    ("String", 1),
    ("Number", 1),
    ("Boolean", 1),
    ("parseInt", 1),
];
const SAFE_METHODS: &[&str] = &[
    "toString", "charAt", "charCodeAt", "at", "slice", "substring", "indexOf", "lastIndexOf",
    "includes", "startsWith", "endsWith", "toUpperCase", "toLowerCase", "trim", "repeat", "concat",
    "split",
];
const ARITH: &[&str] = &["+", "-", "*", "%", "/", "&", "|", "^", "<<", ">>", ">>>"];
const CMP: &[&str] = &["<", ">", "<=", ">=", "==", "===", "!=", "!=="];
const KEYS: &[&str] = &["a", "b", "c"];
const STRS: &[&str] = &["x", "y", "", "ab", "k"];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Scalar,
    Array,
    Object,
}

/// While generating a recursive function's body, the context needed to emit a
/// well-founded self-call: the function's own name, the (frozen) counter
/// parameter, and its arity. A self-call always passes `counter - 1` as the
/// first argument so the unrolling strictly decreases and terminates.
#[derive(Clone)]
struct RecCtx {
    name: String,
    counter: String,
    arity: usize,
}

struct Gen {
    rng: Rng,
    /// In-scope variables, with a loose kind hint (hints need not be sound;
    /// runtime agreement holds regardless, they only improve signal).
    vars: Vec<(String, Kind)>,
    funcs: Vec<(String, usize)>,
    /// Recursive functions (name, arity), callable from anywhere with a *static*
    /// counter so the partial evaluator fully unrolls them (a dynamic counter
    /// would hit the deliberate dynamic-recursion refusal). Kept separate from
    /// `funcs` so their call sites can force the static counter.
    rec_funcs: Vec<(String, usize)>,
    /// Set while generating a recursive function's body (drives self-calls).
    rec: Option<RecCtx>,
    /// Variables that must never be reassigned/updated (recursion counters), so
    /// the self-call's `counter - 1` always strictly decreases.
    frozen: Vec<String>,
    var_ctr: usize,
    loop_depth: usize,
    switch_depth: usize,
}

impl Gen {
    fn fresh_var(&mut self) -> String {
        let n = format!("a{}", self.var_ctr);
        self.var_ctr += 1;
        n
    }

    fn vars_of(&self, want: Option<Kind>) -> Vec<String> {
        self.vars
            .iter()
            .filter(|(_, k)| want.map_or(true, |w| *k == w))
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Variables eligible as an assignment/update *target*: everything in scope
    /// except frozen vars (recursion counters), which must keep their bound
    /// value so a `counter - 1` self-call stays well-founded.
    fn assignable_vars(&self) -> Vec<String> {
        self.vars
            .iter()
            .filter(|(n, _)| !self.frozen.contains(n))
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// An expression of roughly the given fuel; leaves get cheaper as fuel runs
    /// out so generation always terminates.
    fn expr(&mut self, fuel: u32) -> Expr {
        if fuel == 0 || self.rng.chance(1, 3) {
            return self.atom();
        }
        match self.rng.below(11) {
            0 => Expr::Bin(
                *self.rng.pick(ARITH),
                Box::new(self.expr(fuel.saturating_sub(1))),
                Box::new(self.expr(fuel.saturating_sub(1))),
            ),
            1 => Expr::Bin(
                *self.rng.pick(CMP),
                Box::new(self.expr(fuel.saturating_sub(1))),
                Box::new(self.expr(fuel.saturating_sub(1))),
            ),
            2 => Expr::Un(
                *self.rng.pick(&["-", "!", "~", "typeof", "void"]),
                Box::new(self.expr(fuel.saturating_sub(1))),
            ),
            3 => Expr::Logical(
                *self.rng.pick(&["&&", "||", "??"]),
                Box::new(self.expr(fuel.saturating_sub(1))),
                Box::new(self.expr(fuel.saturating_sub(1))),
            ),
            4 => Expr::Ternary(
                Box::new(self.expr(fuel.saturating_sub(1))),
                Box::new(self.expr(fuel.saturating_sub(1))),
                Box::new(self.expr(fuel.saturating_sub(1))),
            ),
            5 => {
                let n = 1 + self.rng.below(3);
                Expr::Array((0..n).map(|_| self.expr(fuel.saturating_sub(1))).collect())
            }
            6 => {
                let n = 1 + self.rng.below(3);
                let mut seen = vec![];
                let fields = (0..n)
                    .filter_map(|_| {
                        let k = self.rng.pick(KEYS).to_string();
                        if seen.contains(&k) {
                            return None;
                        }
                        seen.push(k.clone());
                        Some((k, self.expr(fuel.saturating_sub(1))))
                    })
                    .collect();
                Expr::Object(fields)
            }
            7 => {
                // index/member read off some base
                let base = self.base_expr(fuel.saturating_sub(1));
                if self.rng.chance(1, 2) {
                    Expr::Member(Box::new(base), self.rng.pick(KEYS).to_string())
                } else {
                    Expr::Index(Box::new(base), Box::new(self.expr(fuel.saturating_sub(1))))
                }
            }
            8 => self.call_expr(fuel),
            9 => {
                // .length / .toString() / method
                let base = self.base_expr(fuel.saturating_sub(1));
                if self.rng.chance(1, 2) {
                    Expr::Member(Box::new(base), "length".to_string())
                } else {
                    let m = *self.rng.pick(SAFE_METHODS);
                    // 0-2 args, so methods like charAt/slice/split get exercised
                    // (with a static-string base they should fold; otherwise
                    // residualize).
                    let nargs = self.rng.below(3);
                    let args = (0..nargs).map(|_| self.expr(fuel.saturating_sub(1))).collect();
                    Expr::Method(Box::new(base), m, args)
                }
            }
            _ => {
                // update expression on an existing scalar/any var
                let vs = self.assignable_vars();
                if vs.is_empty() {
                    self.atom()
                } else {
                    Expr::Update {
                        var: self.rng.pick(&vs).clone(),
                        prefix: self.rng.chance(1, 2),
                        inc: self.rng.chance(1, 2),
                    }
                }
            }
        }
    }

    fn call_expr(&mut self, fuel: u32) -> Expr {
        // Inside a recursive body, sometimes recurse. The counter argument is
        // forced to `counter - 1` (the counter is frozen, so this strictly
        // decreases), giving a well-founded recursion the evaluator unrolls.
        if let Some(rc) = self.rec.clone() {
            if self.rng.chance(2, 5) {
                let mut a = vec![Expr::Bin(
                    "-",
                    Box::new(Expr::Var(rc.counter.clone())),
                    Box::new(Expr::Num(1)),
                )];
                for _ in 1..rc.arity {
                    a.push(self.expr(fuel.saturating_sub(1)));
                }
                return Expr::Call(rc.name, a);
            }
        }
        // Outside a recursive body, sometimes call a recursive function with a
        // *static* small counter so the evaluator fully unrolls it (a dynamic
        // counter would hit the deliberate dynamic-recursion refusal). Data
        // arguments stay dynamic, threading runtime state through the unrolled
        // frames — the escape / effect-ordering coverage we're after.
        if self.rec.is_none() && !self.rec_funcs.is_empty() && self.rng.chance(1, 3) {
            let (name, arity) = self.rng.pick(&self.rec_funcs).clone();
            let mut a = vec![Expr::Num(1 + self.rng.below(4) as i64)];
            for _ in 1..arity {
                a.push(self.expr(fuel.saturating_sub(1)));
            }
            return Expr::Call(name, a);
        }
        // Prefer a user function if any are in scope.
        if !self.funcs.is_empty() && self.rng.chance(1, 2) {
            let (name, arity) = self.rng.pick(&self.funcs).clone();
            let args: Vec<Expr> =
                (0..arity).map(|_| self.expr(fuel.saturating_sub(1))).collect();
            // Sometimes invoke via `.apply`/`.call` (the obfuscated-VM idiom that
            // stresses residual functions, `arguments`, and effect ordering).
            match self.rng.below(5) {
                0 => Expr::Method(
                    Box::new(Expr::Var(name)),
                    "apply",
                    vec![Expr::Null, Expr::Array(args)],
                ),
                1 => {
                    let mut a = vec![Expr::Null];
                    a.extend(args);
                    Expr::Method(Box::new(Expr::Var(name)), "call", a)
                }
                _ => Expr::Call(name, args),
            }
        } else {
            let (name, arity) = *self.rng.pick(SAFE_CALLS);
            let args = (0..arity).map(|_| self.expr(fuel.saturating_sub(1))).collect();
            Expr::Call(name.to_string(), args)
        }
    }

    /// A base suitable for member/index access: bias toward an in-scope var.
    fn base_expr(&mut self, fuel: u32) -> Expr {
        let vs = self.vars_of(None);
        if !vs.is_empty() && self.rng.chance(3, 4) {
            Expr::Var(self.rng.pick(&vs).clone())
        } else {
            self.expr(fuel)
        }
    }

    fn atom(&mut self) -> Expr {
        let vs = self.vars_of(None);
        match self.rng.below(6) {
            0 if !vs.is_empty() => Expr::Var(self.rng.pick(&vs).clone()),
            1 => Expr::Num(self.rng.small_int()),
            2 => Expr::Str(self.rng.pick(STRS).to_string()),
            3 => Expr::Bool(self.rng.chance(1, 2)),
            4 => Expr::Null,
            _ => {
                // Always include `input` so programs actually depend on it.
                if self.rng.chance(1, 3) {
                    Expr::Var("input".to_string())
                } else {
                    Expr::Num(self.rng.small_int())
                }
            }
        }
    }

    fn block(&mut self, fuel: u32, n: usize) -> Vec<Stmt> {
        // `var` is function-scoped and hoisted, so vars introduced in a block
        // stay in scope (and reading one before its branch ran exercises
        // hoisting, which is exactly the kind of thing worth fuzzing).
        (0..n).map(|_| self.stmt(fuel)).collect()
    }

    fn stmt(&mut self, fuel: u32) -> Stmt {
        // Bias the menu by context (break/continue only inside loops, etc.).
        let r = self.rng.below(18);
        match r {
            12 | 13 if fuel > 0 => self.closure_decl(fuel),
            0 => {
                let v = self.fresh_var();
                let e = self.expr(fuel);
                let k = match &e {
                    Expr::Array(_) => Kind::Array,
                    Expr::Object(_) => Kind::Object,
                    _ => Kind::Scalar,
                };
                self.vars.push((v.clone(), k));
                Stmt::Var(v, e)
            }
            1 | 2 => {
                let vs = self.assignable_vars();
                if vs.is_empty() {
                    let v = self.fresh_var();
                    self.vars.push((v.clone(), Kind::Scalar));
                    Stmt::Var(v, self.expr(fuel))
                } else {
                    Stmt::AssignVar(self.rng.pick(&vs).clone(), self.expr(fuel))
                }
            }
            3 => {
                let base = self.base_expr(fuel.saturating_sub(1));
                Stmt::AssignIndex(base, self.expr(fuel.saturating_sub(1)), self.expr(fuel.saturating_sub(1)))
            }
            4 => {
                let base = self.base_expr(fuel.saturating_sub(1));
                Stmt::AssignMember(base, self.rng.pick(KEYS).to_string(), self.expr(fuel.saturating_sub(1)))
            }
            5 | 6 => {
                if fuel == 0 {
                    return self.simple_stmt(fuel);
                }
                let c = self.expr(fuel.saturating_sub(1));
                let t = { let n = 1 + self.rng.below(2); self.block(fuel.saturating_sub(1), n) };
                let e = if self.rng.chance(1, 2) {
                    { let n = 1 + self.rng.below(2); self.block(fuel.saturating_sub(1), n) }
                } else {
                    vec![]
                };
                Stmt::If(c, t, e)
            }
            7 => {
                if fuel == 0 {
                    return self.simple_stmt(fuel);
                }
                let v = self.fresh_var();
                self.vars.push((v.clone(), Kind::Scalar));
                let count = 1 + self.rng.below(4) as i64;
                self.loop_depth += 1;
                let body = { let n = 1 + self.rng.below(2); self.block(fuel.saturating_sub(1), n) };
                self.loop_depth -= 1;
                Stmt::For { var: v, count, body }
            }
            8 => {
                if fuel == 0 {
                    return self.simple_stmt(fuel);
                }
                let g = format!("g{}", self.var_ctr);
                self.var_ctr += 1;
                let count = 1 + self.rng.below(4) as i64;
                self.loop_depth += 1;
                let body = { let n = 1 + self.rng.below(2); self.block(fuel.saturating_sub(1), n) };
                self.loop_depth -= 1;
                Stmt::While { guard: g, count, body }
            }
            9 => {
                if fuel == 0 {
                    return self.simple_stmt(fuel);
                }
                let scrut = self.expr(fuel.saturating_sub(1));
                self.switch_depth += 1;
                let ncases = 1 + self.rng.below(3);
                let mut cases = vec![];
                for i in 0..ncases {
                    let label = if i + 1 == ncases && self.rng.chance(1, 2) {
                        None // default
                    } else {
                        Some(self.rng.small_int())
                    };
                    let body = { let n = 1 + self.rng.below(2); self.block(fuel.saturating_sub(1), n) };
                    cases.push((label, body));
                }
                self.switch_depth -= 1;
                Stmt::Switch(scrut, cases)
            }
            10 => {
                if fuel == 0 {
                    return self.simple_stmt(fuel);
                }
                let body = { let n = 1 + self.rng.below(2); self.block(fuel.saturating_sub(1), n) };
                let cv = self.fresh_var();
                self.vars.push((cv.clone(), Kind::Scalar));
                let catch = { let n = 1 + self.rng.below(2); self.block(fuel.saturating_sub(1), n) };
                Stmt::TryCatch(body, cv, catch)
            }
            11 => {
                let vs = self.vars_of(None);
                if !vs.is_empty() {
                    Stmt::Push(self.rng.pick(&vs).clone(), self.expr(fuel))
                } else {
                    self.simple_stmt(fuel)
                }
            }
            _ => self.simple_stmt(fuel),
        }
    }

    /// Declare a closure in a local var: `var f = function (params) { body;
    /// return ret; };`, registered as callable. Its body sees the OUTER
    /// variables (capture-by-reference) and may mutate them, and is sometimes
    /// wrapped in `try` (which forces residualization) — the exact shape that
    /// stresses escape / capture-mutation / effect ordering.
    fn closure_decl(&mut self, fuel: u32) -> Stmt {
        let name = self.fresh_var();
        let arity = self.rng.below(3);
        let id = self.var_ctr;
        let params: Vec<String> = (0..arity).map(|i| format!("c{id}p{i}")).collect();
        // Body sees outer vars (capture) plus the closure's own params.
        let saved_vars = self.vars.clone();
        let saved_ld = self.loop_depth;
        self.loop_depth = 0;
        for p in &params {
            self.vars.push((p.clone(), Kind::Scalar));
        }
        let inner = { let n = 1 + self.rng.below(3); self.block(fuel.saturating_sub(1), n) };
        let ret = self.expr(fuel.saturating_sub(1));
        self.vars = saved_vars;
        self.loop_depth = saved_ld;
        // Often wrap the body in try/catch: a `try`-containing closure is
        // residualized rather than inlined, exercising the residual-function +
        // capture-mutation paths.
        let body = if self.rng.chance(1, 2) {
            vec![Stmt::TryCatch(inner, format!("c{id}e"), vec![])]
        } else {
            inner
        };
        // The closure var is callable.
        self.vars.push((name.clone(), Kind::Scalar));
        self.funcs.push((name.clone(), arity));
        Stmt::Var(name, Expr::Closure { params, body, ret: Box::new(ret) })
    }

    fn simple_stmt(&mut self, fuel: u32) -> Stmt {
        match self.rng.below(10) {
            0 if self.loop_depth > 0 => Stmt::Break,
            1 if self.loop_depth > 0 => Stmt::Continue,
            2 => Stmt::Throw(self.expr(fuel)),
            3 => {
                let vs = self.assignable_vars();
                if !vs.is_empty() {
                    Stmt::Update(self.rng.pick(&vs).clone())
                } else {
                    Stmt::ExprStmt(self.expr(fuel))
                }
            }
            // Early `return` (exercises non-local exit through inlined frames,
            // loops, switches, and try/catch).
            4 | 5 => Stmt::Return(self.expr(fuel)),
            _ => Stmt::ExprStmt(self.expr(fuel)),
        }
    }

    fn func(&mut self, fuel: u32) -> Func {
        let name = format!("f{}", self.funcs.len());
        let arity = self.rng.below(3);
        let params: Vec<String> = (0..arity).map(|i| format!("p{i}")).collect();
        // Function body sees only its params (acyclic call graph: it can call
        // earlier funcs, which are already in self.funcs).
        let saved_vars = std::mem::take(&mut self.vars);
        let saved_ld = self.loop_depth;
        self.loop_depth = 0;
        for p in &params {
            self.vars.push((p.clone(), Kind::Scalar));
        }
        let body = { let n = 1 + self.rng.below(3); self.block(fuel, n) };
        let ret = self.expr(fuel);
        self.vars = saved_vars;
        self.loop_depth = saved_ld;
        Func { name, params, body, ret }
    }

    /// A self-recursive function `r{idx}(n, ...data)`. `n` is a frozen counter;
    /// the body begins with a base case `if (n <= 0) return <leaf>;`, and any
    /// self-call (emitted in `call_expr`) passes `n - 1`, so every well-founded
    /// call chain terminates. The data params thread arbitrary (often dynamic)
    /// state through the recursion, stressing escape / effect ordering across
    /// the unrolled frames.
    fn rec_func(&mut self, fuel: u32, idx: usize) -> Func {
        let name = format!("r{idx}");
        let arity = 2 + self.rng.below(2); // counter + 1..2 data params
        let counter = format!("rn{idx}");
        let mut params = vec![counter.clone()];
        for i in 1..arity {
            params.push(format!("rp{idx}_{i}"));
        }
        // A recursive function (acyclic otherwise): its body sees only its own
        // params plus earlier non-recursive funcs.
        let saved_vars = std::mem::take(&mut self.vars);
        let saved_ld = self.loop_depth;
        let saved_frozen = std::mem::take(&mut self.frozen);
        let saved_rec = self.rec.take();
        self.loop_depth = 0;
        for p in &params {
            self.vars.push((p.clone(), Kind::Scalar));
        }
        self.frozen.push(counter.clone());
        self.rec = Some(RecCtx { name: name.clone(), counter: counter.clone(), arity });
        // Base case first, so the recursion is well-founded by construction.
        let leaf = self.atom();
        let base = Stmt::If(
            Expr::Bin("<=", Box::new(Expr::Var(counter.clone())), Box::new(Expr::Num(0))),
            vec![Stmt::Return(leaf)],
            vec![],
        );
        let mut body = vec![base];
        let n = 1 + self.rng.below(3);
        body.extend(self.block(fuel, n));
        let ret = self.expr(fuel);
        self.vars = saved_vars;
        self.loop_depth = saved_ld;
        self.frozen = saved_frozen;
        self.rec = saved_rec;
        Func { name, params, body, ret }
    }

    fn program(&mut self) -> Prog {
        let nfuncs = self.rng.below(3);
        let mut funcs = vec![];
        for _ in 0..nfuncs {
            let f = self.func(3);
            self.funcs.push((f.name.clone(), f.params.len()));
            funcs.push(f);
        }
        // 0..2 recursive functions, generated after the plain ones (so a
        // recursive body may call an earlier plain func, but plain funcs never
        // forward-reference a recursive one).
        let nrec = self.rng.below(3);
        for i in 0..nrec {
            let f = self.rec_func(3, i);
            self.rec_funcs.push((f.name.clone(), f.params.len()));
            funcs.push(f);
        }
        // main(input)
        self.vars.push(("input".to_string(), Kind::Scalar));
        let main_body = { let n = 2 + self.rng.below(4); self.block(4, n) };
        let ret = self.expr(4);
        Prog { funcs, main_body, ret }
    }
}

// In `simple_stmt` we emit a bare `Update` statement; model it as its own node.
// (Declared here to keep the enum above readable.)
impl Stmt {
    #[allow(non_snake_case)]
    fn Update(var: String) -> Stmt {
        Stmt::ExprStmt(Expr::Update { var, prefix: false, inc: true })
    }
}

// ----------------------------------------------------------------------------
// JS printer. Fully parenthesizes operators so precedence is never in doubt.
// ----------------------------------------------------------------------------

fn base_js(e: &Expr) -> String {
    match e {
        Expr::Var(_) | Expr::Num(_) | Expr::Str(_) | Expr::Bool(_) | Expr::Null => expr_js(e),
        Expr::Member(..) | Expr::Index(..) | Expr::Call(..) | Expr::Method(..) => expr_js(e),
        _ => format!("({})", expr_js(e)),
    }
}

fn expr_js(e: &Expr) -> String {
    match e {
        Expr::Num(n) => n.to_string(),
        Expr::Str(s) => format!("{s:?}"),
        Expr::Bool(b) => b.to_string(),
        Expr::Null => "null".to_string(),
        Expr::Var(v) => v.clone(),
        Expr::Bin(op, a, b) => format!("({} {} {})", expr_js(a), op, expr_js(b)),
        Expr::Un(op, a) => {
            if *op == "typeof" || *op == "void" {
                format!("({} {})", op, expr_js(a))
            } else {
                format!("({}{})", op, expr_js(a))
            }
        }
        Expr::Logical(op, a, b) => format!("({} {} {})", expr_js(a), op, expr_js(b)),
        Expr::Ternary(c, t, e) => format!("({} ? {} : {})", expr_js(c), expr_js(t), expr_js(e)),
        Expr::Array(xs) => {
            let parts: Vec<String> = xs.iter().map(expr_js).collect();
            format!("[{}]", parts.join(", "))
        }
        Expr::Object(fs) => {
            let parts: Vec<String> =
                fs.iter().map(|(k, v)| format!("{k:?}: {}", expr_js(v))).collect();
            format!("{{{}}}", parts.join(", "))
        }
        Expr::Index(b, i) => format!("{}[{}]", base_js(b), expr_js(i)),
        Expr::Member(b, k) => format!("{}.{}", base_js(b), k),
        Expr::Call(name, args) => {
            let parts: Vec<String> = args.iter().map(expr_js).collect();
            format!("{}({})", name, parts.join(", "))
        }
        Expr::Method(b, m, args) => {
            let parts: Vec<String> = args.iter().map(expr_js).collect();
            format!("{}.{}({})", base_js(b), m, parts.join(", "))
        }
        Expr::Update { var, prefix, inc } => {
            let op = if *inc { "++" } else { "--" };
            if *prefix {
                format!("({}{})", op, var)
            } else {
                format!("({}{})", var, op)
            }
        }
        Expr::Closure { params, body, ret } => {
            let mut s = format!("(function ({}) {{\n", params.join(", "));
            stmts_js(&mut s, body, 0);
            let _ = write!(s, "return {};", expr_js(ret));
            s.push_str("})");
            s
        }
    }
}

fn stmts_js(s: &mut String, stmts: &[Stmt], ind: usize) {
    for st in stmts {
        stmt_js(s, st, ind);
    }
}

fn pad(ind: usize) -> String {
    "  ".repeat(ind)
}

fn stmt_js(s: &mut String, st: &Stmt, ind: usize) {
    let p = pad(ind);
    match st {
        Stmt::Var(v, e) => {
            let _ = writeln!(s, "{p}var {v} = {};", expr_js(e));
        }
        Stmt::AssignVar(v, e) => {
            let _ = writeln!(s, "{p}{v} = {};", expr_js(e));
        }
        Stmt::AssignIndex(b, i, e) => {
            let _ = writeln!(s, "{p}{}[{}] = {};", base_js(b), expr_js(i), expr_js(e));
        }
        Stmt::AssignMember(b, k, e) => {
            let _ = writeln!(s, "{p}{}.{} = {};", base_js(b), k, expr_js(e));
        }
        Stmt::If(c, t, e) => {
            let _ = writeln!(s, "{p}if ({}) {{", expr_js(c));
            stmts_js(s, t, ind + 1);
            if e.is_empty() {
                let _ = writeln!(s, "{p}}}");
            } else {
                let _ = writeln!(s, "{p}}} else {{");
                stmts_js(s, e, ind + 1);
                let _ = writeln!(s, "{p}}}");
            }
        }
        Stmt::For { var, count, body } => {
            let _ = writeln!(s, "{p}for (var {var} = 0; {var} < {count}; {var} = {var} + 1) {{");
            stmts_js(s, body, ind + 1);
            let _ = writeln!(s, "{p}}}");
        }
        Stmt::While { guard, count, body } => {
            let _ = writeln!(s, "{p}var {guard} = {count};");
            let _ = writeln!(s, "{p}while ({guard} > 0) {{");
            let _ = writeln!(s, "{}{guard} = {guard} - 1;", pad(ind + 1));
            stmts_js(s, body, ind + 1);
            let _ = writeln!(s, "{p}}}");
        }
        Stmt::Switch(scrut, cases) => {
            let _ = writeln!(s, "{p}switch ({}) {{", expr_js(scrut));
            for (label, body) in cases {
                match label {
                    Some(n) => {
                        let _ = writeln!(s, "{}case {n}: {{", pad(ind + 1));
                    }
                    None => {
                        let _ = writeln!(s, "{}default: {{", pad(ind + 1));
                    }
                }
                stmts_js(s, body, ind + 2);
                let _ = writeln!(s, "{}break;", pad(ind + 2));
                let _ = writeln!(s, "{}}}", pad(ind + 1));
            }
            let _ = writeln!(s, "{p}}}");
        }
        Stmt::TryCatch(b, cv, c) => {
            let _ = writeln!(s, "{p}try {{");
            stmts_js(s, b, ind + 1);
            let _ = writeln!(s, "{p}}} catch ({cv}) {{");
            stmts_js(s, c, ind + 1);
            let _ = writeln!(s, "{p}}}");
        }
        Stmt::Throw(e) => {
            let _ = writeln!(s, "{p}throw {};", expr_js(e));
        }
        Stmt::Return(e) => {
            let _ = writeln!(s, "{p}return {};", expr_js(e));
        }
        Stmt::Break => {
            let _ = writeln!(s, "{p}break;");
        }
        Stmt::Continue => {
            let _ = writeln!(s, "{p}continue;");
        }
        Stmt::Push(v, e) => {
            let _ = writeln!(s, "{p}{v}.push({});", expr_js(e));
        }
        Stmt::ExprStmt(e) => {
            let _ = writeln!(s, "{p}{};", expr_js(e));
        }
    }
}

fn prog_js(prog: &Prog) -> String {
    let mut s = String::new();
    for f in &prog.funcs {
        let _ = writeln!(s, "function {}({}) {{", f.name, f.params.join(", "));
        stmts_js(&mut s, &f.body, 1);
        let _ = writeln!(s, "  return {};", expr_js(&f.ret));
        let _ = writeln!(s, "}}");
    }
    let _ = writeln!(s, "function main(input) {{");
    stmts_js(&mut s, &prog.main_body, 1);
    let _ = writeln!(s, "  return {};", expr_js(&prog.ret));
    let _ = writeln!(s, "}}");
    s
}

// ----------------------------------------------------------------------------
// Oracle: emit the residual (catching panics) and diff against the original.
// ----------------------------------------------------------------------------

const INPUTS: &[i64] = &[0, 1, 2, -1, 3, 7, -5, 11];

enum Residual {
    Js(String),
    Rejected(String), // lowering said "unsupported": generator imperfection, skip
    Refused(String),  // a deliberate, documented evaluator refusal (not a bug)
    Panicked(String),
}

/// Is this panic message one of the partial evaluator's *deliberate* refusals
/// (a documented "not supported in this subset" boundary), as opposed to an
/// actual bug? These are sound: the evaluator stops rather than emit a wrong
/// answer. We count them but never report or shrink them as findings.
fn is_refusal(msg: &str) -> bool {
    msg.contains("dynamic-depth recursion")
        || msg.contains("non-terminating static recursion")
        || msg.contains("out of a residualized `try`")
        || msg.contains("specialization budget exceeded")
}

fn residual_of(src: &str) -> Residual {
    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| js_frontend::to_js(src)));
    match res {
        Ok(Ok(js)) => Residual::Js(js),
        Ok(Err(e)) => Residual::Rejected(e),
        Err(e) => {
            let msg = e
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".to_string());
            if is_refusal(&msg) {
                Residual::Refused(msg)
            } else {
                Residual::Panicked(msg)
            }
        }
    }
}

struct Finding {
    index: usize,
    input: i64,
    kind: String,
    orig: String,
    spec: String,
}

// Minimal hand-rolled JSON parsing for the comparator output (avoids a serde
// dependency). The comparator emits a flat array of flat objects.
mod minijson {
    use super::Finding;

    pub fn parse_findings(s: &str) -> Vec<Finding> {
        let v = parse(s);
        let arr = match v {
            Json::Arr(a) => a,
            _ => return vec![],
        };
        arr.into_iter()
            .filter_map(|o| {
                if let Json::Obj(fields) = o {
                    let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
                    Some(Finding {
                        index: get("index").and_then(num).unwrap_or(0.0) as usize,
                        input: get("input").and_then(num).unwrap_or(0.0) as i64,
                        kind: get("kind").and_then(string).unwrap_or_default(),
                        orig: get("orig").and_then(string).unwrap_or_default(),
                        spec: get("spec").and_then(string).unwrap_or_default(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    fn num(j: Json) -> Option<f64> {
        match j {
            Json::Num(n) => Some(n),
            Json::Str(s) => s.parse().ok(),
            _ => None,
        }
    }
    fn string(j: Json) -> Option<String> {
        match j {
            Json::Str(s) => Some(s),
            Json::Num(n) => Some(n.to_string()),
            Json::Bool(b) => Some(b.to_string()),
            Json::Null => Some("null".to_string()),
            _ => None,
        }
    }

    #[derive(Clone)]
    pub enum Json {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Json>),
        Obj(Vec<(String, Json)>),
    }

    pub fn parse(s: &str) -> Json {
        let mut p = P { b: s.as_bytes(), i: 0 };
        p.ws();
        p.value()
    }

    struct P<'a> {
        b: &'a [u8],
        i: usize,
    }
    impl<'a> P<'a> {
        fn ws(&mut self) {
            while self.i < self.b.len() && (self.b[self.i] as char).is_whitespace() {
                self.i += 1;
            }
        }
        fn value(&mut self) -> Json {
            self.ws();
            if self.i >= self.b.len() {
                return Json::Null;
            }
            match self.b[self.i] {
                b'[' => self.arr(),
                b'{' => self.obj(),
                b'"' => Json::Str(self.string()),
                b't' => {
                    self.i += 4;
                    Json::Bool(true)
                }
                b'f' => {
                    self.i += 5;
                    Json::Bool(false)
                }
                b'n' => {
                    self.i += 4;
                    Json::Null
                }
                _ => self.number(),
            }
        }
        fn arr(&mut self) -> Json {
            self.i += 1; // [
            let mut out = vec![];
            self.ws();
            if self.i < self.b.len() && self.b[self.i] == b']' {
                self.i += 1;
                return Json::Arr(out);
            }
            loop {
                out.push(self.value());
                self.ws();
                if self.i < self.b.len() && self.b[self.i] == b',' {
                    self.i += 1;
                    continue;
                }
                break;
            }
            if self.i < self.b.len() && self.b[self.i] == b']' {
                self.i += 1;
            }
            Json::Arr(out)
        }
        fn obj(&mut self) -> Json {
            self.i += 1; // {
            let mut out = vec![];
            self.ws();
            if self.i < self.b.len() && self.b[self.i] == b'}' {
                self.i += 1;
                return Json::Obj(out);
            }
            loop {
                self.ws();
                let k = self.string();
                self.ws();
                if self.i < self.b.len() && self.b[self.i] == b':' {
                    self.i += 1;
                }
                let v = self.value();
                out.push((k, v));
                self.ws();
                if self.i < self.b.len() && self.b[self.i] == b',' {
                    self.i += 1;
                    continue;
                }
                break;
            }
            if self.i < self.b.len() && self.b[self.i] == b'}' {
                self.i += 1;
            }
            Json::Obj(out)
        }
        fn string(&mut self) -> String {
            // assumes current char is '"'
            self.i += 1;
            let mut out = String::new();
            while self.i < self.b.len() {
                let c = self.b[self.i];
                self.i += 1;
                match c {
                    b'"' => break,
                    b'\\' => {
                        let e = self.b[self.i];
                        self.i += 1;
                        match e {
                            b'n' => out.push('\n'),
                            b't' => out.push('\t'),
                            b'r' => out.push('\r'),
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'u' => {
                                let hex = std::str::from_utf8(&self.b[self.i..self.i + 4]).unwrap_or("0000");
                                let cp = u32::from_str_radix(hex, 16).unwrap_or(0);
                                self.i += 4;
                                if let Some(ch) = char::from_u32(cp) {
                                    out.push(ch);
                                }
                            }
                            other => out.push(other as char),
                        }
                    }
                    _ => out.push(c as char),
                }
            }
            out
        }
        fn number(&mut self) -> Json {
            let start = self.i;
            while self.i < self.b.len()
                && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
            {
                self.i += 1;
            }
            let s = std::str::from_utf8(&self.b[start..self.i]).unwrap_or("0");
            Json::Num(s.parse().unwrap_or(0.0))
        }
    }
}

/// Run the batched Node comparator over the given (src, residual) cases.
fn run_comparator(cases: &[(String, String)], repo: &str) -> Vec<Finding> {
    // Build a JSON array by hand.
    let mut json = String::from("[");
    for (i, (src, residual)) in cases.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let inputs: Vec<String> = INPUTS.iter().map(|n| n.to_string()).collect();
        let _ = write!(
            json,
            "{{\"src\":{},\"residual\":{},\"inputs\":[{}]}}",
            js_str(src),
            js_str(residual),
            inputs.join(",")
        );
    }
    json.push(']');

    let tmp = std::env::temp_dir().join(format!("fuzzcases_{}.json", std::process::id()));
    std::fs::write(&tmp, &json).expect("write cases");
    let out = Command::new("node")
        .arg(format!("{repo}/tools/fuzzcmp.js"))
        .arg(&tmp)
        .output()
        .expect("spawn node");
    let _ = std::fs::remove_file(&tmp);
    if !out.status.success() {
        eprintln!("comparator failed: {}", String::from_utf8_lossy(&out.stderr));
        return vec![];
    }
    minijson::parse_findings(&String::from_utf8_lossy(&out.stdout))
}

/// JSON-encode a string.
fn js_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ----------------------------------------------------------------------------
// Shrinker: reduce a failing program to a minimal still-failing reproducer.
// ----------------------------------------------------------------------------

/// All one-step reductions of a statement list (remove one statement anywhere,
/// at any nesting depth).
fn reduce_stmts(stmts: &[Stmt]) -> Vec<Vec<Stmt>> {
    let mut out = vec![];
    // Remove statement i.
    for i in 0..stmts.len() {
        let mut v = stmts.to_vec();
        v.remove(i);
        out.push(v);
    }
    // Reduce inside statement i's sub-blocks.
    for i in 0..stmts.len() {
        for reduced in reduce_stmt(&stmts[i]) {
            let mut v = stmts.to_vec();
            v[i] = reduced;
            out.push(v);
        }
    }
    out
}

fn reduce_stmt(s: &Stmt) -> Vec<Stmt> {
    let mut out = vec![];
    match s {
        // A closure stored in a var: shrink its body/return.
        Stmt::Var(name, e @ Expr::Closure { .. }) => {
            for re in reduce_expr(e) {
                out.push(Stmt::Var(name.clone(), re));
            }
        }
        Stmt::If(c, t, e) => {
            for tt in reduce_stmts(t) {
                out.push(Stmt::If(c.clone(), tt, e.clone()));
            }
            for ee in reduce_stmts(e) {
                out.push(Stmt::If(c.clone(), t.clone(), ee));
            }
        }
        Stmt::For { var, count, body } => {
            for b in reduce_stmts(body) {
                out.push(Stmt::For { var: var.clone(), count: *count, body: b });
            }
        }
        Stmt::While { guard, count, body } => {
            for b in reduce_stmts(body) {
                out.push(Stmt::While { guard: guard.clone(), count: *count, body: b });
            }
        }
        Stmt::TryCatch(b, cv, c) => {
            for bb in reduce_stmts(b) {
                out.push(Stmt::TryCatch(bb, cv.clone(), c.clone()));
            }
            for cc in reduce_stmts(c) {
                out.push(Stmt::TryCatch(b.clone(), cv.clone(), cc));
            }
        }
        Stmt::Switch(scrut, cases) => {
            for ci in 0..cases.len() {
                // drop a whole case
                let mut v = cases.clone();
                v.remove(ci);
                if !v.is_empty() {
                    out.push(Stmt::Switch(scrut.clone(), v));
                }
                // reduce within a case
                for body in reduce_stmts(&cases[ci].1) {
                    let mut v = cases.clone();
                    v[ci].1 = body;
                    out.push(Stmt::Switch(scrut.clone(), v));
                }
            }
        }
        _ => {}
    }
    out
}

/// One-step program reductions.
fn reduce_prog(p: &Prog) -> Vec<Prog> {
    let mut out = vec![];
    // Drop a function.
    for i in 0..p.funcs.len() {
        let mut funcs = p.funcs.clone();
        funcs.remove(i);
        out.push(Prog { funcs, ..p.clone() });
    }
    // Reduce the main body.
    for body in reduce_stmts(&p.main_body) {
        out.push(Prog { main_body: body, ..p.clone() });
    }
    // Reduce a function body.
    for i in 0..p.funcs.len() {
        for body in reduce_stmts(&p.funcs[i].body) {
            let mut funcs = p.funcs.clone();
            funcs[i].body = body;
            out.push(Prog { funcs, ..p.clone() });
        }
    }
    // Simplify the return expression toward a leaf.
    for r in reduce_expr(&p.ret) {
        out.push(Prog { ret: r, ..p.clone() });
    }
    out
}

/// Replace an expression with a structurally smaller one (a child, or 0).
fn reduce_expr(e: &Expr) -> Vec<Expr> {
    let mut out = vec![];
    // A closure shrinks by reducing its body / return, keeping it a closure (so
    // a bug inside it survives), plus the generic "replace with 0".
    if let Expr::Closure { params, body, ret } = e {
        for b in reduce_stmts(body) {
            out.push(Expr::Closure { params: params.clone(), body: b, ret: ret.clone() });
        }
        for r in reduce_expr(ret) {
            out.push(Expr::Closure { params: params.clone(), body: body.clone(), ret: Box::new(r) });
        }
        out.push(Expr::Num(0));
        return out;
    }
    let children: Vec<Expr> = match e {
        Expr::Bin(_, a, b) | Expr::Logical(_, a, b) => vec![(**a).clone(), (**b).clone()],
        Expr::Un(_, a) => vec![(**a).clone()],
        Expr::Ternary(c, t, f) => vec![(**c).clone(), (**t).clone(), (**f).clone()],
        Expr::Index(b, i) => vec![(**b).clone(), (**i).clone()],
        Expr::Member(b, _) => vec![(**b).clone()],
        Expr::Method(b, _, _) => vec![(**b).clone()],
        Expr::Array(xs) => xs.clone(),
        Expr::Call(_, args) => args.clone(),
        _ => vec![],
    };
    out.extend(children);
    if !matches!(e, Expr::Num(0)) {
        out.push(Expr::Num(0));
    }
    out
}

// ----------------------------------------------------------------------------
// Driver.
// ----------------------------------------------------------------------------

struct Args {
    seed: u64,
    count: usize,
    batch: usize,
    shrink: bool,
    repro: Option<u64>,
    overflow: Option<u64>,
}

fn parse_args() -> Args {
    let mut a =
        Args { seed: 1, count: 5000, batch: 200, shrink: true, repro: None, overflow: None };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--seed" => {
                i += 1;
                a.seed = argv[i].parse().unwrap();
            }
            "--count" => {
                i += 1;
                a.count = argv[i].parse().unwrap();
            }
            "--batch" => {
                i += 1;
                a.batch = argv[i].parse().unwrap();
            }
            "--no-shrink" => a.shrink = false,
            "--repro" => {
                i += 1;
                a.repro = Some(argv[i].parse().unwrap());
            }
            "--overflow" => {
                i += 1;
                a.overflow = Some(argv[i].parse().unwrap());
            }
            other => eprintln!("ignoring unknown arg {other}"),
        }
        i += 1;
    }
    a
}

fn repo_root() -> String {
    // js-frontend/.. = repo root
    let mut p = std::env::current_dir().unwrap();
    if p.ends_with("js-frontend") {
        p.pop();
    }
    p.to_string_lossy().to_string()
}

/// A stable key for a panic message (strips value-specific tails like
/// `Num(0) Str("y")` or out-of-bounds indices) so distinct programs hitting the
/// same bug dedup together and shrinking can target the same failure.
fn panic_key(msg: &str) -> String {
    let cut = msg.find(':').map(|i| i + 1).unwrap_or(msg.len().min(40));
    msg[..cut.min(msg.len())].to_string()
}

/// Does this program still panic with the same failure class?
fn panics_with(p: &Prog, key: &str) -> bool {
    matches!(residual_of(&prog_js(p)), Residual::Panicked(m) if panic_key(&m) == key)
}

/// Shrink a panicking program to a minimal one that still panics the same way.
fn shrink_panic(mut p: Prog, _repo: &str, key: &str) -> Prog {
    let mut budget = 6000;
    loop {
        let mut improved = false;
        for cand in reduce_prog(&p) {
            if budget == 0 {
                return p;
            }
            budget -= 1;
            if panics_with(&cand, key) {
                p = cand;
                improved = true;
                break;
            }
        }
        if !improved {
            return p;
        }
    }
}

/// Does specializing this program abort the engine (stack overflow / OOM)? Runs
/// in a subprocess (the abort is uncatchable in-process) with a wall-clock
/// timeout so a reduction that merely loops forever doesn't hang the shrinker.
fn overflows(p: &Prog) -> bool {
    use std::os::unix::process::ExitStatusExt;
    let src = prog_js(p);
    let tmp = std::env::temp_dir().join(format!("fuzz_ovf_{}.js", std::process::id()));
    if std::fs::write(&tmp, &src).is_err() {
        return false;
    }
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = match Command::new(&exe).arg("--specialize-file").arg(&tmp).spawn() {
        Ok(c) => c,
        Err(_) => return false,
    };
    // Poll up to ~10s; an overflow aborts in well under a second.
    let mut aborted_by_signal = None;
    for _ in 0..1000 {
        match child.try_wait() {
            Ok(Some(status)) => {
                aborted_by_signal = Some(status.signal().is_some());
                break;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(10)),
            Err(_) => break,
        }
    }
    let result = match aborted_by_signal {
        Some(sig) => sig,        // terminated by a signal (SIGABRT from overflow)
        None => {
            let _ = child.kill(); // timed out (a non-overflow infinite loop)
            let _ = child.wait();
            false
        }
    };
    let _ = std::fs::remove_file(&tmp);
    result
}

/// Shrink a program that overflows the engine to a minimal one that still does.
fn shrink_overflow(mut p: Prog) -> Prog {
    let mut budget = 4000;
    loop {
        let mut improved = false;
        for cand in reduce_prog(&p) {
            if budget == 0 {
                return p;
            }
            budget -= 1;
            if overflows(&cand) {
                p = cand;
                improved = true;
                break;
            }
        }
        if !improved {
            return p;
        }
    }
}

/// Does this single program still diverge? Used during shrinking.
fn still_fails(p: &Prog, repo: &str) -> bool {
    let src = prog_js(p);
    match residual_of(&src) {
        Residual::Panicked(_) => true,
        Residual::Rejected(_) => false, // can't lower the reduced form: not a repro
        Residual::Refused(_) => false,  // a deliberate refusal: not a repro
        Residual::Js(js) => !run_comparator(&[(src, js)], repo).is_empty(),
    }
}

fn shrink(mut p: Prog, repo: &str) -> Prog {
    let mut budget = 4000;
    loop {
        let mut improved = false;
        for cand in reduce_prog(&p) {
            if budget == 0 {
                return p;
            }
            budget -= 1;
            if still_fails(&cand, repo) {
                p = cand;
                improved = true;
                break;
            }
        }
        if !improved {
            return p;
        }
    }
}

fn main() {
    // Generated programs are tiny (the largest observed specialization weight is
    // ~1.4k), so cap the partial evaluator's per-program weight budget far below
    // its CLI default. A generated branch-exploder then refuses cleanly at a few
    // hundred MB instead of driving the evaluator into a multi-GB blowup that
    // would OOM the fuzzer. 100k keeps ~70x headroom over any legitimate program
    // while bounding even the heaviest blowups (whose memory-per-weight is far
    // above normal) to well under a gigabyte. Respect an explicit override.
    if std::env::var_os("SPEC_WEIGHT_BUDGET").is_none() {
        std::env::set_var("SPEC_WEIGHT_BUDGET", "100000");
    }

    // Worker mode: specialize one JS file. Used by the overflow shrinker, which
    // runs this in a subprocess so an uncatchable stack overflow / OOM (which
    // `catch_unwind` cannot trap) shows up as the child dying by signal rather
    // than killing the fuzzer. Exits 0 whether `to_js` succeeds or returns an
    // `Err`; only an abort (overflow) is distinguishable, by the signal.
    let argv: Vec<String> = std::env::args().collect();
    if let Some(pos) = argv.iter().position(|a| a == "--specialize-file") {
        let path = &argv[pos + 1];
        let src = std::fs::read_to_string(path).expect("read specialize-file");
        let _ = js_frontend::to_js(&src);
        return;
    }

    let args = parse_args();
    let repo = repo_root();

    // Quiet panic output: we catch panics ourselves and report them cleanly.
    panic::set_hook(Box::new(|_| {}));

    // Shrink a known stack-overflow seed to a minimal reproducer (subprocess
    // based; does not need node).
    if let Some(seed) = args.overflow {
        let mut g = Gen {
            rng: Rng::new(seed),
            vars: vec![],
            funcs: vec![],
            rec_funcs: vec![],
            rec: None,
            frozen: vec![],
            var_ctr: 0,
            loop_depth: 0,
            switch_depth: 0,
        };
        let p = g.program();
        eprintln!("[generated seed {seed} OK; program has {} top-level stmts]", p.main_body.len());
        let _ = std::io::stderr().flush();
        if !overflows(&p) {
            println!("seed {seed} does not overflow (already fixed?)");
            return;
        }
        let minimal = shrink_overflow(p);
        println!("=== minimal stack-overflow reproducer (seed {seed}) ===\n{}", prog_js(&minimal));
        return;
    }

    if Command::new("node").arg("--version").output().is_err() {
        eprintln!("node is required for the differential oracle");
        std::process::exit(2);
    }

    if let Some(seed) = args.repro {
        let mut g = Gen {
            rng: Rng::new(seed),
            vars: vec![],
            funcs: vec![],
            rec_funcs: vec![],
            rec: None,
            frozen: vec![],
            var_ctr: 0,
            loop_depth: 0,
            switch_depth: 0,
        };
        let p = g.program();
        let src = prog_js(&p);
        println!("=== seed {seed} ===\n{src}");
        match residual_of(&src) {
            Residual::Js(js) => {
                println!("--- residual ---\n{js}");
                let f = run_comparator(&[(src, js)], &repo);
                println!("findings: {}", f.len());
                for x in &f {
                    println!("  input={} kind={} orig={} spec={}", x.input, x.kind, x.orig, x.spec);
                }
            }
            Residual::Rejected(e) => println!("rejected: {e}"),
            Residual::Refused(e) => println!("refused (deliberate): {e}"),
            Residual::Panicked(e) => println!("PANIC: {e}"),
        }
        return;
    }

    let mut generated = 0usize;
    let mut rejected = 0usize;
    let mut refused = 0usize;
    let mut panics: Vec<(u64, Prog, String)> = vec![];
    let mut divergences: Vec<(u64, Prog, Finding)> = vec![];

    let mut next_seed = args.seed;
    'outer: while generated < args.count {
        // Build a batch.
        let mut cases: Vec<(String, String)> = vec![];
        let mut seeds: Vec<u64> = vec![];
        let mut progs: Vec<Prog> = vec![];
        while cases.len() < args.batch && generated < args.count {
            let seed = next_seed;
            next_seed = next_seed.wrapping_add(1);
            generated += 1;
            let mut g = Gen {
                rng: Rng::new(seed),
                vars: vec![],
                funcs: vec![],
                rec_funcs: vec![],
                rec: None,
                frozen: vec![],
                var_ctr: 0,
                loop_depth: 0,
                switch_depth: 0,
            };
            let p = g.program();
            let src = prog_js(&p);
            // Trace seed before specializing so an *uncatchable* abort (stack
            // overflow / OOM in the engine, which catch_unwind cannot trap)
            // leaves the culprit seed as the last line on stderr.
            if std::env::var_os("FUZZ_TRACE").is_some() {
                eprintln!("trying {seed}");
                let _ = std::io::stderr().flush();
            }
            match residual_of(&src) {
                Residual::Js(js) => {
                    seeds.push(seed);
                    progs.push(p);
                    cases.push((src, js));
                }
                Residual::Rejected(_) => rejected += 1,
                Residual::Refused(_) => refused += 1,
                Residual::Panicked(e) => {
                    panics.push((seed, p, e.clone()));
                    eprintln!("\n!! PANIC seed={seed}: {e}");
                }
            }
        }
        if cases.is_empty() {
            continue;
        }

        let findings = run_comparator(&cases, &repo);
        for f in findings {
            let seed = seeds[f.index];
            let prog = progs[f.index].clone();
            eprintln!(
                "\n!! DIVERGENCE seed={seed} input={} kind={} orig={} spec={}",
                f.input, f.kind, f.orig, f.spec
            );
            divergences.push((seed, prog, f));
            if divergences.len() >= 25 {
                break 'outer;
            }
        }
        print!("\r  generated {generated}, rejected {rejected}, refused {refused}, panics {}, divergences {}   ",
            panics.len(), divergences.len());
        let _ = std::io::stdout().flush();
    }

    println!(
        "\n\n=== done: generated {generated}, rejected {rejected}, refused {refused}, panics {}, divergences {} ===",
        panics.len(),
        divergences.len()
    );

    // Report panics: one representative per distinct message, shrunk to a
    // minimal reproducer that still panics with the same message.
    let mut seen_panic = std::collections::BTreeSet::new();
    for (seed, prog, msg) in &panics {
        let key = panic_key(msg);
        if !seen_panic.insert(key.clone()) {
            continue;
        }
        println!("\n================ PANIC seed={seed} ================");
        println!("{msg}");
        let minimal = if args.shrink {
            shrink_panic(prog.clone(), &repo, &key)
        } else {
            prog.clone()
        };
        let src = prog_js(&minimal);
        println!("--- minimal reproducer ---\n{src}");
    }

    // Shrink and report divergences.
    for (seed, prog, f) in &divergences {
        println!("\n================ DIVERGENCE seed={seed} ================");
        println!("kind={} input={} orig={} spec={}", f.kind, f.input, f.orig, f.spec);
        let minimal = if args.shrink {
            shrink(prog.clone(), &repo)
        } else {
            prog.clone()
        };
        let src = prog_js(&minimal);
        println!("--- minimal reproducer ---\n{src}");
        if let Residual::Js(js) = residual_of(&src) {
            println!("--- residual ---\n{js}");
        }
    }
}
