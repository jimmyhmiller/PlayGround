//! Static type analysis for beagle's `num_*` specialization.
//!
//! Sound for **open-world** dynamic-language compilation: every fact in
//! `TypeInfo` is derivable from a single function body, so adding new
//! callers, new struct constructors, or new modules can never invalidate
//! a previously-emitted `num_*`. Cross-function inference (function
//! return types, parameter types from call sites, struct field types
//! from creation sites) was removed — it required closed-world
//! assumptions that don't hold for a JIT serving FFI / REPL / dynamic
//! dispatch / deserialized objects. A future tiered JIT can reintroduce
//! those facts speculatively, behind type guards with deopt.
//!
//! Two layers:
//!
//! 1. **Per-function analysis** (`analyze_types`) — for each top-level
//!    function, walk the body and compute the LUB of (init ∪ every
//!    reachable `Assignment`) for every let-mut binding. That's the only
//!    "whole-body" fact we need; reads of the let-mut see the conservative
//!    type even before the assignments have been lowered.
//!
//! 2. **Per-function scope** (`TypeEnv`) — pushed/popped in lockstep with
//!    `DynFunc`'s scope stack while lowering, so inlined frames don't leak
//!    types into the caller. `TypeEnv::type_of` is the single entry point
//!    used during lowering to query an expression's static type.
//!
//! The lattice (`Ty`) is deliberately minimal: just enough to recognize
//! NaN-boxed numbers and propagate that knowledge through let bindings,
//! local arithmetic, and array literal / push chains. Anything that
//! crosses a function call, a struct field, or an unknown identifier
//! collapses to `Unknown` and the call site emits the safe `dyn_*` form.

use std::collections::HashMap;

use crate::ast::{Ast, Pattern};

/// Static type lattice. See module docs for the rationale on minimality.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    /// Bottom: no value of this type exists yet. Used for empty array
    /// literals (`[]` is `Array<Bottom>`) and for fresh let-mut bindings
    /// before any assignment has been observed. `Bottom.lub(T) = T` for
    /// any `T`, which lets a `let mut xs = []` accumulate into
    /// `Array<T>` once `push(xs, x: T)` is observed.
    Bottom,
    Number,
    /// Names a struct type. Carried so future passes (e.g. ICs) can
    /// distinguish struct identities; we don't currently resolve fields
    /// off it because that would require closed-world struct-field
    /// inference.
    Object(String),
    /// Array with statically-known element type. Only assigned to array
    /// literals whose elements all share a type, and to chains of `push`
    /// that preserve it.
    Array(Box<Ty>),
    Unknown,
}

impl Ty {
    pub fn is_number(&self) -> bool {
        matches!(self, Ty::Number)
    }

    /// Pointwise least-upper-bound. Bottom is the identity; same-type
    /// joins (incl. matching object/array element) preserve type;
    /// mismatches collapse to `Unknown`.
    pub fn lub(&self, other: &Ty) -> Ty {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Ty::Bottom, t) | (t, Ty::Bottom) => t.clone(),
            (Ty::Array(a), Ty::Array(b)) => Ty::Array(Box::new(a.lub(b))),
            _ => Ty::Unknown,
        }
    }
}

impl From<&Ty> for dynlang::TypeHint {
    fn from(t: &Ty) -> dynlang::TypeHint {
        match t {
            Ty::Number => dynlang::TypeHint::Number,
            // Object / Array / Bottom / Unknown all collapse: the binop
            // dispatcher treats anything that isn't `Number` as needing
            // the conservative `dyn_*` form.
            _ => dynlang::TypeHint::Unknown,
        }
    }
}

/// Per-function type information. The only "whole-body" fact we infer
/// soundly without closed-world: for each let-mut binding, the LUB of
/// (its initializer ∪ every reachable assignment) within the function.
/// Used by lowering so reads of a let-mut see the conservative type even
/// before the assignments have been lowered.
#[derive(Default)]
pub struct TypeInfo {
    /// Per top-level function → (let-mut name → LUB of init + assigns).
    pub let_mut_types: HashMap<String, HashMap<String, Ty>>,
}

/// Per-function type-tracking scope used during lowering. Holds a
/// borrowed reference to whole-program `TypeInfo` and `globals`, plus a
/// scope stack mirroring `DynFunc::push_scope` / `pop_scope`.
pub struct TypeEnv<'a> {
    pub info: &'a TypeInfo,
    pub globals: &'a HashMap<String, Ast>,
    pub current_fn: String,
    /// Scope stack. Each `let` / `let mut` / inlined-arg binding records
    /// a type in the topmost scope at the same point its `def_var`
    /// happens.
    var_types: Vec<HashMap<String, Ty>>,
}

impl<'a> TypeEnv<'a> {
    /// Build a fresh env for one function's body. Params start as
    /// `Unknown` — sound under open-world (we don't get to assume
    /// anything about what callers pass).
    pub fn new(
        info: &'a TypeInfo,
        globals: &'a HashMap<String, Ast>,
        fname: String,
    ) -> Self {
        Self {
            info,
            globals,
            current_fn: fname,
            var_types: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.var_types.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.var_types.pop();
    }

    pub fn bind(&mut self, name: &str, ty: Ty) {
        if let Some(scope) = self.var_types.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    /// Resolve a let-mut variable's type. Tries the per-function table
    /// first (which reflects the LUB of init + every assignment in
    /// scope). Falls back to typing the initializer directly when no
    /// analysis entry exists — useful for synthetic helper let-muts
    /// inserted by lowering itself.
    pub fn let_mut_type(&self, name: &str, init: &Ast) -> Ty {
        if let Some(per_fn) = self.info.let_mut_types.get(&self.current_fn) {
            if let Some(t) = per_fn.get(name) {
                return t.clone();
            }
        }
        self.type_of(init)
    }

    /// Static type of an arbitrary expression. Conservative: when in
    /// doubt, returns `Unknown` and the call site falls back to `dyn_*`.
    pub fn type_of(&self, ast: &Ast) -> Ty {
        expr_type(ast, &self.var_types, self.globals)
    }
}

/// Per-function type analysis. Walks each top-level function body once
/// and collects the let-mut LUB table. Sound under open-world: the only
/// inputs are the function's own AST and its globals (which are
/// constants and themselves locally typed).
pub fn analyze_types(elements: &[Ast], globals: &HashMap<String, Ast>) -> TypeInfo {
    let mut info = TypeInfo::default();
    for el in elements {
        if let Ast::Function { name, body, .. } = el {
            let Some(fname) = name.as_ref() else { continue };
            let mut let_mut: HashMap<String, Ty> = HashMap::new();
            let env: Vec<HashMap<String, Ty>> = vec![HashMap::new()];
            collect_let_mut_types(body, &env, globals, &mut let_mut);
            info.let_mut_types.insert(fname.clone(), let_mut);
        }
    }
    info
}

/// Type one expression against a pre-built env. The single source of
/// truth for the lattice rules — `TypeEnv::type_of` delegates here.
///
/// Open-world rules: anything that depends on a fact across a function
/// boundary (return type of a user-defined call, type of a struct
/// field, type of an unknown identifier) returns `Unknown`. Builtins
/// have hard-coded return types because the runtime owns their
/// implementation.
fn expr_type(
    ast: &Ast,
    env: &Vec<HashMap<String, Ty>>,
    globals: &HashMap<String, Ast>,
) -> Ty {
    match ast {
        Ast::IntegerLiteral(..) | Ast::FloatLiteral(..) => Ty::Number,
        Ast::True(..) | Ast::False(..) | Ast::Null(..) | Ast::String(..) => Ty::Unknown,

        Ast::Identifier(name, _) => {
            if let Some(global) = globals.get(name) {
                return expr_type(global, env, globals);
            }
            for scope in env.iter().rev() {
                if let Some(t) = scope.get(name) {
                    return t.clone();
                }
            }
            Ty::Unknown
        }

        Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. } => {
            if expr_type(left, env, globals).is_number()
                && expr_type(right, env, globals).is_number()
            {
                Ty::Number
            } else {
                Ty::Unknown
            }
        }

        Ast::Condition { .. } | Ast::And { .. } | Ast::Or { .. } | Ast::Not { .. } => {
            Ty::Unknown
        }

        Ast::If { then, else_, .. } => {
            // Each branch is a Vec<Ast> with its own `let` chain. We
            // can't just type the last expression — `let` bindings made
            // earlier in the branch must be visible to it. Run a
            // statement-by-statement walk in a fresh scope for each.
            let tt = type_block_local(then, env, globals);
            let et = type_block_local(else_, env, globals);
            tt.lub(&et)
        }

        Ast::Let { value, .. } | Ast::LetMut { value, .. } => expr_type(value, env, globals),
        Ast::Assignment { value, .. } => expr_type(value, env, globals),

        Ast::StructCreation { name, .. } => Ty::Object(name.clone()),

        Ast::Array { array, .. } => {
            if array.is_empty() {
                return Ty::Array(Box::new(Ty::Bottom));
            }
            let mut acc = expr_type(&array[0], env, globals);
            for x in &array[1..] {
                acc = acc.lub(&expr_type(x, env, globals));
            }
            Ty::Array(Box::new(acc))
        }

        Ast::IndexOperator { array, .. } => {
            if let Ty::Array(elem) = expr_type(array, env, globals) {
                *elem
            } else {
                Ty::Unknown
            }
        }

        // Open-world: we can't trust an inferred struct-field type, so
        // every property read is Unknown. A future tiered JIT can
        // recover this by guarding on the IC's observed shape.
        Ast::PropertyAccess { .. } => Ty::Unknown,

        Ast::Call { name, args, .. } => {
            // Builtins have hard-coded return types — the runtime owns
            // their implementation, so this is sound under open-world.
            match name.as_str() {
                "cos" | "sin" | "to-float" | "to-number" | "length" | "core/time-now" => {
                    Ty::Number
                }
                "push" => {
                    // `push(arr, x)` returns Array<LUB(prior_elem, type_of(x))>.
                    if args.len() == 2 {
                        let arr_ty = expr_type(&args[0], env, globals);
                        let val_ty = expr_type(&args[1], env, globals);
                        if let Ty::Array(prior) = arr_ty {
                            return Ty::Array(Box::new(prior.lub(&val_ty)));
                        }
                        Ty::Array(Box::new(val_ty))
                    } else {
                        Ty::Unknown
                    }
                }
                // Open-world: no inferred return types for user fns.
                _ => Ty::Unknown,
            }
        }

        Ast::CallExpr { .. } => Ty::Unknown,

        Ast::While { .. } | Ast::Loop { .. } | Ast::For { .. } => Ty::Unknown,

        _ => Ty::Unknown,
    }
}

/// Type a sequence of statements in a fresh nested scope. Used for if /
/// while / for branches whose `let` bindings shouldn't leak past them.
/// Returns the type of the last statement.
fn type_block_local(
    body: &[Ast],
    env: &Vec<HashMap<String, Ty>>,
    globals: &HashMap<String, Ast>,
) -> Ty {
    let mut local: Vec<HashMap<String, Ty>> = env.clone();
    local.push(HashMap::new());
    let mut last = Ty::Unknown;
    for stmt in body {
        match stmt {
            Ast::Let { pattern, value, .. } | Ast::LetMut { pattern, value, .. } => {
                let ty = expr_type(value, &local, globals);
                if let Pattern::Identifier { name, .. } = pattern {
                    if let Some(scope) = local.last_mut() {
                        scope.insert(name.clone(), ty.clone());
                    }
                }
                last = ty;
            }
            other => last = expr_type(other, &local, globals),
        }
    }
    last
}

/// Collect let-mut narrowed types via a forward walk over the function
/// body. Threads a mutable env through so `let` bindings encountered
/// mid-body (including ones inside loop bodies) are visible when we
/// type subsequent expressions — including assignments to outer
/// let-muts whose RHS reads those inner lets.
fn collect_let_mut_types(
    body: &[Ast],
    seed_env: &Vec<HashMap<String, Ty>>,
    globals: &HashMap<String, Ast>,
    out: &mut HashMap<String, Ty>,
) {
    let mut env = seed_env.clone();
    for s in body {
        forward_let_mut(s, &mut env, globals, out);
    }
}

/// Walk the AST in lexical order. Maintains `env` as encountered
/// `let`/`let mut` bindings extend it. For each `let mut x = init`,
/// record/init `out[x] = init_type`. For each `Assignment(Identifier(x),
/// v)`, update `out[x] = out[x].lub(value_type)` AND `env[x] = out[x]`
/// so later reads of `x` see the running LUB.
fn forward_let_mut(
    ast: &Ast,
    env: &mut Vec<HashMap<String, Ty>>,
    globals: &HashMap<String, Ast>,
    out: &mut HashMap<String, Ty>,
) {
    match ast {
        Ast::Let { pattern, value, .. } => {
            forward_let_mut(value, env, globals, out);
            let ty = expr_type(value, env, globals);
            if let Pattern::Identifier { name, .. } = pattern {
                if let Some(scope) = env.last_mut() {
                    scope.insert(name.clone(), ty);
                }
            }
        }
        Ast::LetMut { pattern, value, .. } => {
            forward_let_mut(value, env, globals, out);
            let init_ty = expr_type(value, env, globals);
            if let Pattern::Identifier { name, .. } = pattern {
                let merged = match out.get(name) {
                    Some(prev) => prev.lub(&init_ty),
                    None => init_ty.clone(),
                };
                out.insert(name.clone(), merged.clone());
                if let Some(scope) = env.last_mut() {
                    scope.insert(name.clone(), merged);
                }
            }
        }
        Ast::Assignment { name, value, .. } => {
            forward_let_mut(value, env, globals, out);
            if let Ast::Identifier(n, _) = name.as_ref() {
                let v_ty = expr_type(value, env, globals);
                let merged = match out.get(n) {
                    Some(prev) => prev.lub(&v_ty),
                    None => v_ty,
                };
                out.insert(n.clone(), merged.clone());
                // Push the running LUB into env so subsequent reads of
                // the variable in this same pass see the wider type.
                for scope in env.iter_mut().rev() {
                    if scope.contains_key(n) {
                        scope.insert(n.clone(), merged);
                        break;
                    }
                }
            }
        }
        Ast::While { condition, body, .. } => {
            forward_let_mut(condition, env, globals, out);
            // Loops can iterate, so a let-mut written inside the body
            // can be read back at the top of the body. Two passes
            // suffice for the simple lattice we use (a third pass would
            // make no further progress).
            for _ in 0..2 {
                env.push(HashMap::new());
                for s in body {
                    forward_let_mut(s, env, globals, out);
                }
                env.pop();
            }
        }
        Ast::For { collection, body, .. } => {
            forward_let_mut(collection, env, globals, out);
            for _ in 0..2 {
                env.push(HashMap::new());
                for s in body {
                    forward_let_mut(s, env, globals, out);
                }
                env.pop();
            }
        }
        Ast::Loop { body, .. } => {
            for _ in 0..2 {
                env.push(HashMap::new());
                for s in body {
                    forward_let_mut(s, env, globals, out);
                }
                env.pop();
            }
        }
        Ast::If { condition, then, else_, .. } => {
            forward_let_mut(condition, env, globals, out);
            env.push(HashMap::new());
            for s in then {
                forward_let_mut(s, env, globals, out);
            }
            env.pop();
            env.push(HashMap::new());
            for s in else_ {
                forward_let_mut(s, env, globals, out);
            }
            env.pop();
        }
        _ => {
            walk_children(ast, |c| forward_let_mut(c, env, globals, out));
        }
    }
}

/// Generic AST child-walker: invokes `f` on every immediate child
/// expression. Used by `forward_let_mut` to avoid duplicating the
/// structural recursion. Doesn't recurse into nested function
/// definitions or other top-level forms — those are walked at the
/// program level.
fn walk_children(ast: &Ast, mut f: impl FnMut(&Ast)) {
    match ast {
        Ast::If { condition, then, else_, .. } => {
            f(condition);
            for x in then {
                f(x);
            }
            for x in else_ {
                f(x);
            }
        }
        Ast::While { condition, body, .. } => {
            f(condition);
            for x in body {
                f(x);
            }
        }
        Ast::For { collection, body, .. } => {
            f(collection);
            for x in body {
                f(x);
            }
        }
        Ast::Loop { body, .. }
        | Ast::Reset { body, .. }
        | Ast::Shift { body, .. }
        | Ast::Test { body, .. } => {
            for x in body {
                f(x);
            }
        }
        Ast::Condition { left, right, .. }
        | Ast::Add { left, right, .. }
        | Ast::Sub { left, right, .. }
        | Ast::Mul { left, right, .. }
        | Ast::Div { left, right, .. }
        | Ast::Modulo { left, right, .. }
        | Ast::ShiftLeft { left, right, .. }
        | Ast::ShiftRight { left, right, .. }
        | Ast::ShiftRightZero { left, right, .. }
        | Ast::BitWiseAnd { left, right, .. }
        | Ast::BitWiseOr { left, right, .. }
        | Ast::BitWiseXor { left, right, .. }
        | Ast::And { left, right, .. }
        | Ast::Or { left, right, .. } => {
            f(left);
            f(right);
        }
        Ast::Call { args, .. } | Ast::Recurse { args, .. } | Ast::TailRecurse { args, .. } => {
            for a in args {
                f(a);
            }
        }
        Ast::CallExpr { callee, args, .. } => {
            f(callee);
            for a in args {
                f(a);
            }
        }
        Ast::Let { value, .. } | Ast::LetMut { value, .. } | Ast::LetDynamic { value, .. } => {
            f(value);
        }
        Ast::Binding { value_expr, body, .. } => {
            f(value_expr);
            for s in body {
                f(s);
            }
        }
        Ast::Assignment { name, value, .. } => {
            f(name);
            f(value);
        }
        Ast::Not { expr, .. } => f(expr),
        Ast::PropertyAccess { object, property, .. } => {
            f(object);
            f(property);
        }
        Ast::IndexOperator { array, index, .. } => {
            f(array);
            f(index);
        }
        Ast::Array { array, .. } => {
            for x in array {
                f(x);
            }
        }
        Ast::StructCreation { fields, spread, .. } => {
            for (_, v) in fields {
                f(v);
            }
            if let Some(s) = spread {
                f(s);
            }
        }
        Ast::EnumCreation { fields, .. } => {
            for (_, v) in fields {
                f(v);
            }
        }
        Ast::MapLiteral { pairs, .. } => {
            for (k, v) in pairs {
                f(k);
                f(v);
            }
        }
        Ast::SetLiteral { elements, .. } => {
            for x in elements {
                f(x);
            }
        }
        Ast::Break { value, .. } | Ast::Return { value, .. } | Ast::Throw { value, .. } => {
            f(value);
        }
        Ast::StringInterpolation { parts, .. } => {
            for p in parts {
                if let crate::ast::StringInterpolationPart::Expression(e) = p {
                    f(e);
                }
            }
        }
        Ast::Try { body, catch_body, .. } => {
            for x in body {
                f(x);
            }
            for x in catch_body {
                f(x);
            }
        }
        Ast::Match { value, arms, .. } => {
            f(value);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    f(g);
                }
                for x in &arm.body {
                    f(x);
                }
            }
        }
        Ast::MultiArityFunction { cases, .. } => {
            for c in cases {
                for x in &c.body {
                    f(x);
                }
            }
        }
        Ast::Perform { value, .. } => f(value),
        Ast::Handle { handler_instance, body, .. } => {
            f(handler_instance);
            for x in body {
                f(x);
            }
        }
        Ast::Future { body, .. } => f(body),
        Ast::Use { alias, .. } => f(alias),

        // Leaves and top-level decls — no children we walk in this
        // helper.
        Ast::Program { .. }
        | Ast::Function { .. }
        | Ast::FunctionStub { .. }
        | Ast::Struct { .. }
        | Ast::StructField { .. }
        | Ast::Enum { .. }
        | Ast::EnumVariant { .. }
        | Ast::EnumStaticVariant { .. }
        | Ast::Protocol { .. }
        | Ast::Extend { .. }
        | Ast::ProtocolDispatch { .. }
        | Ast::IntegerLiteral(..)
        | Ast::FloatLiteral(..)
        | Ast::Identifier(..)
        | Ast::String(..)
        | Ast::Keyword(..)
        | Ast::True(..)
        | Ast::False(..)
        | Ast::Null(..)
        | Ast::Namespace { .. }
        | Ast::Continue { .. } => {}
    }
}
