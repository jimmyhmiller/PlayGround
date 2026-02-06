use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::ast::{BinOp, Expr};
use crate::opaque::{ArrayBufferState, DataViewState, OpaqueRegistry, TextDecoderState, Uint8ArrayState};
use crate::value::{array_from_vec, new_env, new_object, value_to_expr, Value};

/// Partial environment maps variables to either static values or dynamic expressions
pub type PEnv = Rc<RefCell<HashMap<String, PValue>>>;

// Thread-local storage for the opaque registry
thread_local! {
    static OPAQUE_REGISTRY: RefCell<Option<OpaqueRegistry>> = const { RefCell::new(None) };
    static GAS: RefCell<Option<usize>> = const { RefCell::new(None) };
}

/// Set gas limit for partial evaluation (for debugging infinite loops)
pub fn set_gas(limit: usize) {
    GAS.with(|g| *g.borrow_mut() = Some(limit));
}

/// Check and decrement gas, panic if exhausted
fn use_gas(expr: &Expr) {
    GAS.with(|g| {
        if let Some(ref mut gas) = *g.borrow_mut() {
            if *gas == 0 {
                let expr_type = match expr {
                    Expr::Int(_) => "Int",
                    Expr::Bool(_) => "Bool",
                    Expr::Var(_) => "Var",
                    Expr::Let(_, _, _) => "Let",
                    Expr::Fn(_, _) => "Fn",
                    Expr::Call(_, _) => "Call",
                    Expr::While(_, _) => "While",
                    Expr::Switch { .. } => "Switch",
                    Expr::Array(_) => "Array",
                    _ => "Other",
                };
                panic!("GAS EXHAUSTED on {}", expr_type);
            }
            *gas -= 1;
        }
    });
}

fn with_gas_disabled<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let saved = GAS.with(|g| g.borrow().clone());
    GAS.with(|g| *g.borrow_mut() = None);
    let result = f();
    GAS.with(|g| *g.borrow_mut() = saved);
    result
}

/// Set the opaque registry for the current thread
pub fn set_opaque_registry(registry: OpaqueRegistry) {
    OPAQUE_REGISTRY.with(|r| {
        *r.borrow_mut() = Some(registry);
    });
}

/// Clear the opaque registry for the current thread
pub fn clear_opaque_registry() {
    OPAQUE_REGISTRY.with(|r| {
        *r.borrow_mut() = None;
    });
}

/// Run a function with an opaque registry set, then clear it
pub fn with_opaque_registry<F, R>(registry: OpaqueRegistry, f: F) -> R
where
    F: FnOnce() -> R,
{
    set_opaque_registry(registry);
    let result = f();
    clear_opaque_registry();
    result
}

/// Try to handle a `new` expression using the registered handlers
fn try_opaque_new(ctor: &Expr, args_exprs: &[Expr], args_pvalues: &[PValue], env: &PEnv) -> Option<PValue> {
    OPAQUE_REGISTRY.with(|r| {
        if let Some(ref registry) = *r.borrow() {
            registry.try_handle_new(ctor, args_exprs, args_pvalues, env)
        } else {
            None
        }
    })
}

/// Try to handle a call expression using the registered handlers
#[allow(dead_code)]
fn try_opaque_call(callee: &Expr, args_exprs: &[Expr], args_pvalues: &[PValue], env: &PEnv) -> Option<PValue> {
    OPAQUE_REGISTRY.with(|r| {
        if let Some(ref registry) = *r.borrow() {
            registry.try_handle_call(callee, args_exprs, args_pvalues, env)
        } else {
            None
        }
    })
}

pub fn new_penv() -> PEnv {
    Rc::new(RefCell::new(HashMap::new()))
}

// Thread-local tracking of variables that have been mutated via Set.
// This is used to determine which captured variables should be treated as dynamic in closures.
thread_local! {
    static MUTATED_VARS: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

fn mark_var_mutated(name: &str) {
    MUTATED_VARS.with(|m| {
        m.borrow_mut().insert(name.to_string());
    });
}

fn is_var_mutated(name: &str) -> bool {
    MUTATED_VARS.with(|m| {
        m.borrow().contains(name)
    })
}

fn is_internal_name(name: &str) -> bool {
    let mut chars = name.chars();
    if chars.next() != Some('v') {
        return false;
    }
    chars.all(|c| c.is_ascii_digit())
}

/// Clear the mutated vars tracking - call at the start of a fresh partial evaluation
pub fn clear_mutated_vars() {
    MUTATED_VARS.with(|m| {
        m.borrow_mut().clear();
    });
}

pub fn penv_with_parent(parent: &PEnv) -> PEnv {
    // Return the same env instead of cloning - this ensures mutations
    // propagate across all nested scopes, matching JavaScript's var semantics
    parent.clone()
}

/// Check if a value is mutable (objects, arrays, and opaque values with state)
/// These need special handling to preserve variable references in residuals
fn is_mutable_value(v: &Value) -> bool {
    matches!(v,
        Value::Object(_) | Value::Array(_) |
        Value::Opaque { state: Some(_), .. }  // Opaque with state is mutable
    )
}

/// Collect all captured (non-local) variables that are mutated via set! in an expression.
/// This is used to determine which captured variables should be treated as dynamic.
/// Iterative implementation to avoid stack overflow.
fn collect_captured_mutations(expr: &Expr, local_vars: &[String], result: &mut HashSet<String>) {
    // Work items: (expression, local_vars for that context)
    let mut stack: Vec<(&Expr, Vec<String>)> = vec![(expr, local_vars.to_vec())];

    while let Some((e, locals)) = stack.pop() {
        match e {
            Expr::Set(name, value) => {
                // If setting a variable that's NOT local to this scope, it's a captured mutation
                if !locals.contains(name) {
                    result.insert(name.clone());
                }
                stack.push((value, locals));
            }

            Expr::Let(name, value, body) => {
                // Body gets new local binding
                let mut new_locals = locals.clone();
                new_locals.push(name.clone());
                stack.push((body, new_locals));
                // Value uses current locals
                stack.push((value, locals));
            }

            Expr::Fn(params, body) => {
                // At function boundary, reset locals to just params
                stack.push((body, params.clone()));
            }

            Expr::BinOp(_, l, r) => {
                stack.push((r, locals.clone()));
                stack.push((l, locals));
            }

            Expr::If(c, t, e) => {
                stack.push((e, locals.clone()));
                stack.push((t, locals.clone()));
                stack.push((c, locals));
            }

            Expr::Call(f, args) => {
                for a in args.iter().rev() {
                    stack.push((a, locals.clone()));
                }
                stack.push((f, locals));
            }

            Expr::Begin(exprs) => {
                for expr in exprs.iter().rev() {
                    stack.push((expr, locals.clone()));
                }
            }

            Expr::While(cond, body) => {
                stack.push((body, locals.clone()));
                stack.push((cond, locals));
            }

            Expr::For { init, cond, update, body } => {
                stack.push((body, locals.clone()));
                if let Some(u) = update {
                    stack.push((u, locals.clone()));
                }
                if let Some(c) = cond {
                    stack.push((c, locals.clone()));
                }
                if let Some(i) = init {
                    stack.push((i, locals));
                }
            }

            Expr::Array(elems) => {
                for elem in elems.iter().rev() {
                    stack.push((elem, locals.clone()));
                }
            }

            Expr::Index(arr, idx) => {
                stack.push((idx, locals.clone()));
                stack.push((arr, locals));
            }

            Expr::Len(arr) => {
                stack.push((arr, locals));
            }

            Expr::Object(props) => {
                for (_, v) in props.iter().rev() {
                    stack.push((v, locals.clone()));
                }
            }

            Expr::PropAccess(obj, _) => {
                stack.push((obj, locals));
            }

            Expr::PropSet(obj, _, val) => {
                stack.push((val, locals.clone()));
                stack.push((obj, locals));
            }

            Expr::ComputedAccess(obj, key) => {
                stack.push((key, locals.clone()));
                stack.push((obj, locals));
            }

            Expr::ComputedSet(obj, key, val) => {
                stack.push((val, locals.clone()));
                stack.push((key, locals.clone()));
                stack.push((obj, locals));
            }

            Expr::BitNot(inner) | Expr::LogNot(inner) | Expr::Throw(inner) | Expr::Return(inner) => {
                stack.push((inner, locals));
            }

            Expr::New(ctor, args) => {
                for a in args.iter().rev() {
                    stack.push((a, locals.clone()));
                }
                stack.push((ctor, locals));
            }

            Expr::Switch { discriminant, cases, default } => {
                if let Some(d) = default {
                    for stmt in d.iter().rev() {
                        stack.push((stmt, locals.clone()));
                    }
                }
                for (cv, body) in cases.iter().rev() {
                    for stmt in body.iter().rev() {
                        stack.push((stmt, locals.clone()));
                    }
                    stack.push((cv, locals.clone()));
                }
                stack.push((discriminant, locals));
            }

            Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
                if let Some(fb) = finally_block {
                    stack.push((fb, locals.clone()));
                }
                // Catch block may have catch_param as local
                let mut catch_locals = locals.clone();
                if let Some(param) = catch_param {
                    catch_locals.push(param.clone());
                }
                stack.push((catch_block, catch_locals));
                stack.push((try_block, locals));
            }

            // Literals, Var, Break, Continue, etc. - no mutations to collect
            Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Undefined
            | Expr::Null | Expr::Var(_) | Expr::Opaque(_) | Expr::Break | Expr::Continue => {}
        }
    }
}

/// Check if an expression contains set! operations on captured (non-local) variables.
/// This is used to detect closures that mutate their captured environment.
/// Iterative implementation to avoid stack overflow.
fn contains_captured_mutation(expr: &Expr, local_vars: &[String]) -> bool {
    // Work items: (expression, local_vars for that context)
    let mut stack: Vec<(&Expr, Vec<String>)> = vec![(expr, local_vars.to_vec())];

    while let Some((e, locals)) = stack.pop() {
        match e {
            Expr::Set(name, value) => {
                // If setting a variable that's NOT local to this scope, it's a captured mutation
                if !locals.contains(name) {
                    return true;
                }
                stack.push((value, locals));
            }

            Expr::Let(name, value, body) => {
                // Body gets new local binding
                let mut new_locals = locals.clone();
                new_locals.push(name.clone());
                stack.push((body, new_locals));
                // Value uses current locals
                stack.push((value, locals));
            }

            Expr::Fn(params, body) => {
                // IMPORTANT: At a function boundary, reset locals to just params.
                // Variables from enclosing scopes are "captured", not local.
                stack.push((body, params.clone()));
            }

            Expr::BinOp(_, l, r) => {
                stack.push((r, locals.clone()));
                stack.push((l, locals));
            }

            Expr::If(c, t, el) => {
                stack.push((el, locals.clone()));
                stack.push((t, locals.clone()));
                stack.push((c, locals));
            }

            Expr::Call(f, args) => {
                for a in args.iter().rev() {
                    stack.push((a, locals.clone()));
                }
                stack.push((f, locals));
            }

            Expr::Begin(exprs) => {
                for expr in exprs.iter().rev() {
                    stack.push((expr, locals.clone()));
                }
            }

            Expr::While(cond, body) => {
                stack.push((body, locals.clone()));
                stack.push((cond, locals));
            }

            Expr::For { init, cond, update, body } => {
                stack.push((body, locals.clone()));
                if let Some(u) = update {
                    stack.push((u, locals.clone()));
                }
                if let Some(c) = cond {
                    stack.push((c, locals.clone()));
                }
                if let Some(i) = init {
                    stack.push((i, locals));
                }
            }

            Expr::Array(elems) => {
                for elem in elems.iter().rev() {
                    stack.push((elem, locals.clone()));
                }
            }

            Expr::Index(arr, idx) => {
                stack.push((idx, locals.clone()));
                stack.push((arr, locals));
            }

            Expr::Len(arr) => {
                stack.push((arr, locals));
            }

            Expr::Object(props) => {
                for (_, v) in props.iter().rev() {
                    stack.push((v, locals.clone()));
                }
            }

            Expr::PropAccess(obj, _) => {
                stack.push((obj, locals));
            }

            Expr::PropSet(obj, _, val) => {
                stack.push((val, locals.clone()));
                stack.push((obj, locals));
            }

            Expr::ComputedAccess(obj, key) => {
                stack.push((key, locals.clone()));
                stack.push((obj, locals));
            }

            Expr::ComputedSet(obj, key, val) => {
                stack.push((val, locals.clone()));
                stack.push((key, locals.clone()));
                stack.push((obj, locals));
            }

            Expr::BitNot(inner) | Expr::LogNot(inner) | Expr::Throw(inner) | Expr::Return(inner) => {
                stack.push((inner, locals));
            }

            Expr::New(ctor, args) => {
                for a in args.iter().rev() {
                    stack.push((a, locals.clone()));
                }
                stack.push((ctor, locals));
            }

            Expr::Switch { discriminant, cases, default } => {
                if let Some(d) = default {
                    for stmt in d.iter().rev() {
                        stack.push((stmt, locals.clone()));
                    }
                }
                for (cv, body) in cases.iter().rev() {
                    for stmt in body.iter().rev() {
                        stack.push((stmt, locals.clone()));
                    }
                    stack.push((cv, locals.clone()));
                }
                stack.push((discriminant, locals));
            }

            Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
                if let Some(fb) = finally_block {
                    stack.push((fb, locals.clone()));
                }
                // Catch block may have catch_param as local
                let mut catch_locals = locals.clone();
                if let Some(param) = catch_param {
                    catch_locals.push(param.clone());
                }
                stack.push((catch_block, catch_locals));
                stack.push((try_block, locals));
            }

            // Literals and simple expressions don't contain mutations
            Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Var(_)
            | Expr::Undefined | Expr::Null | Expr::Opaque(_)
            | Expr::Break | Expr::Continue => {}
        }
    }

    false
}

/// A partially-evaluated value: either a known static value or a dynamic expression
#[derive(Clone, Debug)]
pub enum PValue {
    /// A fully-known static value that can be inlined anywhere (immutable: numbers, bools, strings)
    Static(Value),
    /// A dynamic expression that must be preserved in residual
    Dynamic(Expr),
    /// A return control-flow value carrying the actual result
    Return(Box<PValue>),
    /// A static value bound to a variable name - used for mutable values (objects, arrays)
    /// where we need to preserve the variable reference in residuals rather than inlining
    StaticNamed { name: String, value: Value },
    /// A dynamic expression bound to a variable name - residualizes to Var(name) but keeps
    /// the original expression for optimization purposes (e.g., idempotent mask elimination)
    DynamicNamed { name: String, expr: Expr },
}

/// Convert a PValue back to an expression (residualization)
pub fn residualize(pv: &PValue) -> Expr {
    match pv {
        PValue::Static(v) => value_to_expr(v),
        PValue::Dynamic(e) => e.clone(),
        PValue::Return(inner) => Expr::Return(Box::new(residualize(inner))),
        // For named static values, emit the variable reference instead of the value literal
        PValue::StaticNamed { name, .. } => Expr::Var(name.clone()),
        // For named dynamic values, emit the variable reference (keeps expr for optimization)
        PValue::DynamicNamed { name, .. } => Expr::Var(name.clone()),
    }
}

/// JavaScript-style truthiness check for static values
/// Returns Some(true) for truthy, Some(false) for falsy, None if not a static value
fn is_js_truthy(pv: &PValue) -> Option<bool> {
    match pv {
        PValue::Static(v) | PValue::StaticNamed { value: v, .. } => {
            Some(match v {
                Value::Bool(b) => *b,
                Value::Int(n) => *n != 0,
                Value::String(s) => !s.is_empty(),
                Value::Undefined | Value::Null => false,
                // Arrays, objects, closures, opaques are always truthy in JS
                Value::Array(_) | Value::Object(_) | Value::Closure { .. } | Value::Opaque { .. } => true,
            })
        }
        PValue::Dynamic(_) | PValue::DynamicNamed { .. } | PValue::Return(_) => None,
    }
}

/// Work items for the iterative partial evaluator
/// Each item either evaluates an expression or continues after child evaluation
enum PEWorkItem {
    /// Evaluate an expression, store result in the given slot
    Eval { expr: Expr, env: PEnv, result_slot: usize },

    /// Continue after evaluating binary operation operands
    ContBinOp { op: BinOp, left_slot: usize, right_slot: usize, result_slot: usize },

    /// Continue after evaluating condition - decide which branch to take
    ContIfCond { then_branch: Expr, else_branch: Expr, env: PEnv, cond_slot: usize, result_slot: usize },

    /// Continue after evaluating both branches (for dynamic condition)
    ContIfBothBranches { cond_slot: usize, then_slot: usize, else_slot: usize, result_slot: usize },

    /// Continue after evaluating let value - bind and evaluate body
    ContLetValue { name: String, body: Expr, outer_env: PEnv, value_slot: usize, result_slot: usize },

    /// Continue after evaluating let body - decide whether to emit let
    ContLetBody { name: String, value_slot: usize, body_slot: usize, result_slot: usize },

    /// Continue after evaluating function definition (PE the body)
    ContFnBody { params: Vec<String>, outer_env: PEnv, body_slot: usize, result_slot: usize, body_expr: Expr },

    /// Restore params in env after function body PE
    ContFnRestoreParams { params: Vec<String>, saved_params: Vec<(String, PValue)>, env: PEnv },

    /// Continue after evaluating call function and args
    ContCall { func_slot: usize, arg_slots: Vec<usize>, result_slot: usize, original_func_expr: Expr, original_args: Vec<Expr>, env: PEnv },

    /// Continue after evaluating inlined call body (unwrap return)
    ContCallBody { body_slot: usize, result_slot: usize },

    /// Continue after evaluating array elements
    ContArray { elem_slots: Vec<usize>, result_slot: usize },

    /// Continue after evaluating index operands
    ContIndex { arr_slot: usize, idx_slot: usize, result_slot: usize },

    /// Continue after evaluating length operand
    ContLen { arr_slot: usize, result_slot: usize },

    /// Continue after evaluating object properties
    ContObject { keys: Vec<String>, value_slots: Vec<usize>, result_slot: usize },

    /// Continue after evaluating prop access object
    ContPropAccess { obj_slot: usize, prop: String, result_slot: usize },

    /// Continue after evaluating prop set operands
    ContPropSet { obj_slot: usize, prop: String, value_slot: usize, result_slot: usize },

    /// Continue after evaluating computed access operands
    ContComputedAccess { obj_slot: usize, key_slot: usize, result_slot: usize },

    /// Continue after evaluating computed set operands
    ContComputedSet { obj_slot: usize, key_slot: usize, value_slot: usize, result_slot: usize },

    /// Continue after evaluating set value
    ContSet { name: String, value_slot: usize, result_slot: usize, env: PEnv },

    /// Continue after evaluating begin expressions
    ContBegin { expr_slots: Vec<usize>, result_slot: usize },

    /// Continue after evaluating bit not operand
    ContBitNot { inner_slot: usize, result_slot: usize },

    /// Continue after evaluating log not operand
    ContLogNot { inner_slot: usize, result_slot: usize },

    /// Continue after evaluating throw operand
    ContThrow { inner_slot: usize, result_slot: usize },

    /// Continue after evaluating return operand
    ContReturn { inner_slot: usize, result_slot: usize },

    /// Continue after evaluating new constructor and args
    ContNew { ctor_slot: usize, arg_slots: Vec<usize>, result_slot: usize, original_ctor: Expr, original_args: Vec<Expr>, env: PEnv },

    /// Continue after evaluating switch discriminant
    ContSwitchDiscriminant { disc_slot: usize, cases: Vec<(Expr, Vec<Expr>)>, default: Option<Vec<Expr>>, env: PEnv, result_slot: usize },

    /// Find matching case in switch (static discriminant) - iterative
    ContSwitchFindCase { disc_val: Value, cases: Vec<(Expr, Vec<Expr>)>, default: Option<Vec<Expr>>, env: PEnv, result_slot: usize, case_idx: usize },

    /// Continue after evaluating a case value to check for match
    ContSwitchCheckCase { disc_val: Value, case_val_slot: usize, case_body: Vec<Expr>, remaining_cases: Vec<(Expr, Vec<Expr>)>, default: Option<Vec<Expr>>, env: PEnv, result_slot: usize },

    /// Execute switch case body statements iteratively
    ContSwitchCaseBody { stmts: Vec<Expr>, stmt_idx: usize, env: PEnv, result_slot: usize, collected_residuals: Vec<Expr>, last_static: Option<Value> },

    /// Continue after evaluating a switch case body statement
    ContSwitchCaseBodyStmt { stmt_slot: usize, remaining_stmts: Vec<Expr>, env: PEnv, result_slot: usize, collected_residuals: Vec<Expr>, last_static: Option<Value> },

    /// Continue after evaluating while condition (first time)
    ContWhileCond { cond: Expr, body: Expr, env: PEnv, cond_slot: usize, result_slot: usize, initial_env_snapshot: HashMap<String, PValue> },

    /// Continue after evaluating while body (during unrolling)
    ContWhileBody { cond: Expr, body: Expr, env: PEnv, body_slot: usize, result_slot: usize, initial_env_snapshot: HashMap<String, PValue>, pre_iteration_snapshot: HashMap<String, PValue>, iterations: usize, unrolled_bodies: Vec<Expr> },

    /// Continue after evaluating for init
    ContForInit { cond: Option<Expr>, update: Option<Expr>, body: Expr, env: PEnv, result_slot: usize, init_expr: Option<Expr> },

    /// Continue after evaluating for condition
    ContForCond { cond: Option<Expr>, update: Option<Expr>, body: Expr, env: PEnv, cond_slot: usize, result_slot: usize, iterations: usize, init_expr: Option<Expr> },

    /// Continue after evaluating for body
    ContForBody { cond: Option<Expr>, update: Option<Expr>, body: Expr, env: PEnv, body_slot: usize, result_slot: usize, iterations: usize, init_expr: Option<Expr> },

    /// Continue after evaluating try block
    ContTryCatch { try_slot: usize, catch_param: Option<String>, catch_block: Expr, finally_block: Option<Expr>, env: PEnv, result_slot: usize },
}

/// The main partial evaluation function (iterative implementation)
pub fn partial_eval(expr: &Expr, env: &PEnv) -> PValue {
    // Use results vector to store intermediate results
    let mut results: Vec<Option<PValue>> = vec![None]; // Slot 0 is the final result
    let mut work_stack: Vec<PEWorkItem> = vec![
        PEWorkItem::Eval { expr: expr.clone(), env: env.clone(), result_slot: 0 }
    ];

    while let Some(work) = work_stack.pop() {
        match work {
            PEWorkItem::Eval { expr, env, result_slot } => {
                eval_expr(&expr, &env, result_slot, &mut results, &mut work_stack);
            }

            PEWorkItem::ContBinOp { op, left_slot, right_slot, result_slot } => {
                let left_pv = results[left_slot].take().unwrap();
                let right_pv = results[right_slot].take().unwrap();
                results[result_slot] = Some(partial_eval_binop(&op, left_pv, right_pv));
            }

            PEWorkItem::ContIfCond { then_branch, else_branch, env, cond_slot, result_slot } => {
                let cond_pv = results[cond_slot].take().unwrap();
                if matches!(cond_pv, PValue::Return(_)) {
                    results[result_slot] = Some(cond_pv);
                    continue;
                }
                match is_js_truthy(&cond_pv) {
                    Some(true) => {
                        // Static true - evaluate only then branch
                        work_stack.push(PEWorkItem::Eval { expr: then_branch, env, result_slot });
                    }
                    Some(false) => {
                        // Static false - evaluate only else branch
                        work_stack.push(PEWorkItem::Eval { expr: else_branch, env, result_slot });
                    }
                    None => {
                        // Dynamic condition - need to evaluate both branches
                        let then_slot = results.len();
                        results.push(None);
                        let else_slot = results.len();
                        results.push(None);
                        // Store cond result back for continuation
                        results[cond_slot] = Some(cond_pv);

                        work_stack.push(PEWorkItem::ContIfBothBranches { cond_slot, then_slot, else_slot, result_slot });
                        work_stack.push(PEWorkItem::Eval { expr: else_branch, env: env.clone(), result_slot: else_slot });
                        work_stack.push(PEWorkItem::Eval { expr: then_branch, env, result_slot: then_slot });
                    }
                }
            }

            PEWorkItem::ContIfBothBranches { cond_slot, then_slot, else_slot, result_slot } => {
                let cond_pv = results[cond_slot].take().unwrap();
                let then_pv = results[then_slot].take().unwrap();
                let else_pv = results[else_slot].take().unwrap();

                let then_residual = residualize(&then_pv);
                let else_residual = residualize(&else_pv);

                // If both branches are identical, simplify
                if then_residual == else_residual {
                    let cond_residual = residualize(&cond_pv);
                    if is_pure_expr(&cond_residual) {
                        results[result_slot] = Some(then_pv);
                    } else {
                        results[result_slot] = Some(PValue::Dynamic(Expr::Begin(vec![
                            cond_residual,
                            then_residual,
                        ])));
                    }
                } else {
                    results[result_slot] = Some(PValue::Dynamic(Expr::If(
                        Box::new(residualize(&cond_pv)),
                        Box::new(then_residual),
                        Box::new(else_residual),
                    )));
                }
            }

            PEWorkItem::ContLetValue { name, body, outer_env, value_slot, result_slot } => {
                let value_pv = results[value_slot].clone().unwrap();
                if matches!(&value_pv, PValue::Return(_)) {
                    results[result_slot] = Some(value_pv);
                    continue;
                }
                let new_env = penv_with_parent(&outer_env);

                // Determine the binding type
                let binding = match &value_pv {
                    PValue::Static(v) => {
                        if is_mutable_value(v) {
                            PValue::StaticNamed { name: name.clone(), value: v.clone() }
                        } else {
                            value_pv.clone()
                        }
                    }
                    PValue::StaticNamed { value, .. } => {
                        PValue::StaticNamed { name: name.clone(), value: value.clone() }
                    }
                    PValue::Dynamic(e) => {
                        PValue::DynamicNamed { name: name.clone(), expr: e.clone() }
                    }
                    PValue::DynamicNamed { expr, .. } => {
                        PValue::DynamicNamed { name: name.clone(), expr: expr.clone() }
                    }
                    PValue::Return(inner) => PValue::Return(Box::new(inner.as_ref().clone())),
                };
                if name == "v11" || name == "v9" || name == "v21" {
                    eprintln!("DEBUG: ContLetValue binding {} = {:?}, env_ptr={:?}", name, binding, new_env.as_ptr());
                }
                new_env.borrow_mut().insert(name.clone(), binding);

                let body_slot = results.len();
                results.push(None);

                work_stack.push(PEWorkItem::ContLetBody { name, value_slot, body_slot, result_slot });
                work_stack.push(PEWorkItem::Eval { expr: body, env: new_env, result_slot: body_slot });
            }

            PEWorkItem::ContLetBody { name, value_slot, body_slot, result_slot } => {
                let value_pv = results[value_slot].take().unwrap();
                let body_pv = results[body_slot].take().unwrap();

                let residual_body = residualize(&body_pv);
                // If body is just the bound variable, the let is redundant.
                if matches!(&residual_body, Expr::Var(var) if var == &name) {
                    results[result_slot] = Some(value_pv);
                    continue;
                }
                if uses_var(&residual_body, &name) {
                    let init_expr = match &value_pv {
                        PValue::Static(v) | PValue::StaticNamed { value: v, .. } => value_to_expr(v),
                        PValue::Dynamic(e) | PValue::DynamicNamed { expr: e, .. } => e.clone(),
                        PValue::Return(inner) => Expr::Return(Box::new(residualize(inner))),
                    };
                    results[result_slot] = Some(PValue::Dynamic(Expr::Let(
                        name,
                        Box::new(init_expr),
                        Box::new(residual_body),
                    )));
                } else {
                    results[result_slot] = Some(body_pv);
                }
            }

            PEWorkItem::ContFnBody { params, outer_env, body_slot, result_slot, body_expr } => {
                let body_pv = results[body_slot].take().unwrap();
                let pe_body = residualize(&body_pv);
                eprintln!("DEBUG ContFnBody: Creating closure, body starts with: {:?}", &pe_body.to_string().chars().take(100).collect::<String>());

                // Check if this closure mutates captured variables
                if contains_captured_mutation(&body_expr, &params) {
                    // Store the ORIGINAL body_expr, not pe_body. This allows us to
                    // PE the body fresh at call time with the actual runtime values.
                    results[result_slot] = Some(PValue::Dynamic(Expr::Fn(params, Box::new(body_expr))));
                } else {
                    results[result_slot] = Some(PValue::Static(Value::Closure {
                        params,
                        body: pe_body,
                        env: penv_to_env(&outer_env),
                    }));
                }
            }

            PEWorkItem::ContFnRestoreParams { params, saved_params, env } => {
                // Restore saved param bindings after function body PE
                let restored_params: HashSet<String> = saved_params.iter().map(|(p, _)| p.clone()).collect();
                for (param, saved) in saved_params {
                    env.borrow_mut().insert(param, saved);
                }
                // Remove params that weren't shadowing anything
                for param in &params {
                    if !restored_params.contains(param) {
                        env.borrow_mut().remove(param);
                    }
                }
            }

            PEWorkItem::ContCall { func_slot, arg_slots, result_slot, original_func_expr, original_args, env } => {
                let func_pv = results[func_slot].take().unwrap();
                let args_pv: Vec<PValue> = arg_slots.iter().map(|s| results[*s].take().unwrap()).collect();

                let call_result = handle_call(func_pv, args_pv, &env, &original_func_expr, &original_args, &mut results, &mut work_stack, result_slot);
                if let Some(pv) = call_result {
                    results[result_slot] = Some(pv);
                }
                // If call_result is None, it means we pushed more work items
            }

            PEWorkItem::ContCallBody { body_slot, result_slot } => {
                let body_pv = results[body_slot].take().unwrap();
                results[result_slot] = Some(unwrap_return_pv(body_pv));
            }

            PEWorkItem::ContArray { elem_slots, result_slot } => {
                let elem_pvs: Vec<PValue> = elem_slots.iter().map(|s| results[*s].take().unwrap()).collect();

                if elem_pvs.iter().all(|pv| matches!(pv, PValue::Static(_))) {
                    let values: Vec<Value> = elem_pvs
                        .into_iter()
                        .map(|pv| match pv {
                            PValue::Static(v) => v,
                            _ => unreachable!(),
                        })
                        .collect();
                    results[result_slot] = Some(PValue::Static(Value::Array(array_from_vec(values))));
                } else {
                    results[result_slot] = Some(PValue::Dynamic(Expr::Array(elem_pvs.iter().map(residualize).collect())));
                }
            }

            PEWorkItem::ContIndex { arr_slot, idx_slot, result_slot } => {
                let arr_pv = results[arr_slot].take().unwrap();
                let idx_pv = results[idx_slot].take().unwrap();
                results[result_slot] = Some(eval_index(arr_pv, idx_pv));
            }

            PEWorkItem::ContLen { arr_slot, result_slot } => {
                let arr_pv = results[arr_slot].take().unwrap();
                if let Some(cf) = control_flow_pv(&arr_pv) {
                    results[result_slot] = Some(cf);
                    continue;
                }
                results[result_slot] = Some(match &arr_pv {
                    PValue::StaticNamed { name, value: Value::Array(_), .. } if !is_internal_name(name) && is_var_mutated(name) => {
                        PValue::Dynamic(Expr::Len(Box::new(residualize(&arr_pv))))
                    }
                    PValue::Static(Value::Array(elements)) |
                    PValue::StaticNamed { value: Value::Array(elements), .. } => {
                        PValue::Static(Value::Int(elements.borrow().len() as i64))
                    }
                    _ => PValue::Dynamic(Expr::Len(Box::new(residualize(&arr_pv)))),
                });
            }

            PEWorkItem::ContObject { keys, value_slots, result_slot } => {
                let prop_pvs: Vec<(String, PValue)> = keys.into_iter()
                    .zip(value_slots.iter().map(|s| results[*s].take().unwrap()))
                    .collect();

                let all_static = prop_pvs.iter().all(|(_, pv)| {
                    matches!(pv, PValue::Static(_) | PValue::StaticNamed { .. })
                });

                if all_static {
                    let obj = new_object();
                    for (k, pv) in &prop_pvs {
                        match pv {
                            PValue::Static(v) => { obj.borrow_mut().insert(k.clone(), v.clone()); }
                            PValue::StaticNamed { value, .. } => { obj.borrow_mut().insert(k.clone(), value.clone()); }
                            _ => {}
                        }
                    }
                    results[result_slot] = Some(PValue::Static(Value::Object(obj)));
                } else {
                    let residual_props: Vec<(String, Expr)> = prop_pvs.iter()
                        .map(|(k, pv)| (k.clone(), residualize(pv)))
                        .collect();
                    results[result_slot] = Some(PValue::Dynamic(Expr::Object(residual_props)));
                }
            }

            PEWorkItem::ContPropAccess { obj_slot, prop, result_slot } => {
                let obj_pv = results[obj_slot].take().unwrap();
                results[result_slot] = Some(eval_prop_access(obj_pv, &prop));
            }

            PEWorkItem::ContPropSet { obj_slot, prop, value_slot, result_slot } => {
                let obj_pv = results[obj_slot].take().unwrap();
                let value_pv = results[value_slot].take().unwrap();
                results[result_slot] = Some(eval_prop_set(obj_pv, &prop, value_pv));
            }

            PEWorkItem::ContComputedAccess { obj_slot, key_slot, result_slot } => {
                let obj_pv = results[obj_slot].take().unwrap();
                let key_pv = results[key_slot].take().unwrap();
                results[result_slot] = Some(eval_computed_access(obj_pv, key_pv));
            }

            PEWorkItem::ContComputedSet { obj_slot, key_slot, value_slot, result_slot } => {
                let obj_pv = results[obj_slot].take().unwrap();
                let key_pv = results[key_slot].take().unwrap();
                let value_pv = results[value_slot].take().unwrap();
                results[result_slot] = Some(eval_computed_set(obj_pv, key_pv, value_pv));
            }

            PEWorkItem::ContSet { name, value_slot, result_slot, env } => {
                let value_pv = results[value_slot].take().unwrap();
                results[result_slot] = Some(eval_set(&name, value_pv, &env));
            }

            PEWorkItem::ContBegin { expr_slots, result_slot } => {
                let pvs: Vec<PValue> = expr_slots.iter().map(|s| results[*s].take().unwrap()).collect();
                results[result_slot] = Some(eval_begin(pvs));
            }

            PEWorkItem::ContBitNot { inner_slot, result_slot } => {
                let inner_pv = results[inner_slot].take().unwrap();
                results[result_slot] = Some(match inner_pv {
                    PValue::Static(Value::Int(n)) => PValue::Static(Value::Int(!n)),
                    _ => PValue::Dynamic(Expr::BitNot(Box::new(residualize(&inner_pv)))),
                });
            }

            PEWorkItem::ContLogNot { inner_slot, result_slot } => {
                let inner_pv = results[inner_slot].take().unwrap();
                results[result_slot] = Some(eval_log_not(inner_pv));
            }

            PEWorkItem::ContThrow { inner_slot, result_slot } => {
                let inner_pv = results[inner_slot].take().unwrap();
                results[result_slot] = Some(PValue::Dynamic(Expr::Throw(Box::new(residualize(&inner_pv)))));
            }

            PEWorkItem::ContReturn { inner_slot, result_slot } => {
                let inner_pv = results[inner_slot].take().unwrap();
                let return_inner = match inner_pv {
                    PValue::Return(nested) => *nested,
                    other => other,
                };
                results[result_slot] = Some(PValue::Return(Box::new(return_inner)));
            }

            PEWorkItem::ContNew { ctor_slot, arg_slots, result_slot, original_ctor, original_args, env } => {
                let ctor_pv = results[ctor_slot].take().unwrap();
                let args_pv: Vec<PValue> = arg_slots.iter().map(|s| results[*s].take().unwrap()).collect();

                // Try opaque registry handlers first
                if let Some(result) = try_opaque_new(&original_ctor, &original_args, &args_pv, &env) {
                    results[result_slot] = Some(result);
                } else {
                    let residual = Expr::New(
                        Box::new(residualize(&ctor_pv)),
                        args_pv.iter().map(residualize).collect(),
                    );
                    results[result_slot] = Some(PValue::Static(Value::Opaque {
                        label: "new expression".to_string(),
                        expr: residual,
                        state: None,
                    }));
                }
            }

            PEWorkItem::ContSwitchDiscriminant { disc_slot, cases, default, env, result_slot } => {
                let disc_pv = results[disc_slot].take().unwrap();
                eval_switch(disc_pv, cases, default, env, result_slot, &mut results, &mut work_stack);
            }

            PEWorkItem::ContSwitchFindCase { disc_val, cases, default, env, result_slot, case_idx } => {
                if case_idx >= cases.len() {
                    // No matching case found - execute default if present
                    if let Some(default_body) = default {
                        if default_body.is_empty() {
                            results[result_slot] = Some(PValue::Static(Value::Undefined));
                        } else {
                            // Start executing default body
                            work_stack.push(PEWorkItem::ContSwitchCaseBody {
                                stmts: default_body,
                                stmt_idx: 0,
                                env,
                                result_slot,
                                collected_residuals: Vec::new(),
                                last_static: None,
                            });
                        }
                    } else {
                        results[result_slot] = Some(PValue::Static(Value::Undefined));
                    }
                } else {
                    // Evaluate the case value at case_idx
                    let (case_val, case_body) = cases[case_idx].clone();
                    let remaining_cases: Vec<_> = cases.into_iter().skip(case_idx + 1).collect();

                    let case_val_slot = results.len();
                    results.push(None);

                    work_stack.push(PEWorkItem::ContSwitchCheckCase {
                        disc_val,
                        case_val_slot,
                        case_body,
                        remaining_cases,
                        default,
                        env: env.clone(),
                        result_slot,
                    });
                    work_stack.push(PEWorkItem::Eval { expr: case_val, env, result_slot: case_val_slot });
                }
            }

            PEWorkItem::ContSwitchCheckCase { disc_val, case_val_slot, case_body, remaining_cases, default, env, result_slot } => {
                let case_pv = results[case_val_slot].take().unwrap();
                if let PValue::Static(cv) = case_pv {
                    if cv == disc_val {
                        // Match found - execute this case body
                        if case_body.is_empty() {
                            results[result_slot] = Some(PValue::Static(Value::Undefined));
                        } else {
                            work_stack.push(PEWorkItem::ContSwitchCaseBody {
                                stmts: case_body,
                                stmt_idx: 0,
                                env,
                                result_slot,
                                collected_residuals: Vec::new(),
                                last_static: None,
                            });
                        }
                    } else {
                        // No match - continue to next case
                        let mut all_cases = vec![];
                        all_cases.extend(remaining_cases);
                        work_stack.push(PEWorkItem::ContSwitchFindCase {
                            disc_val,
                            cases: all_cases,
                            default,
                            env,
                            result_slot,
                            case_idx: 0,
                        });
                    }
                } else {
                    // Dynamic case value - can't match statically, continue to next case
                    let mut all_cases = vec![];
                    all_cases.extend(remaining_cases);
                    work_stack.push(PEWorkItem::ContSwitchFindCase {
                        disc_val,
                        cases: all_cases,
                        default,
                        env,
                        result_slot,
                        case_idx: 0,
                    });
                }
            }

            PEWorkItem::ContSwitchCaseBody { stmts, stmt_idx, env, result_slot, collected_residuals, last_static } => {
                if stmt_idx >= stmts.len() {
                    // Done with all statements
                    if collected_residuals.is_empty() {
                        results[result_slot] = Some(match last_static {
                            Some(v) => PValue::Static(v),
                            None => PValue::Static(Value::Undefined),
                        });
                    } else {
                        let filtered = filter_dead_code(collected_residuals);
                        if filtered.is_empty() {
                            results[result_slot] = Some(PValue::Static(Value::Undefined));
                        } else if filtered.len() == 1 {
                            results[result_slot] = Some(PValue::Dynamic(filtered.into_iter().next().unwrap()));
                        } else {
                            results[result_slot] = Some(PValue::Dynamic(Expr::Begin(filtered)));
                        }
                    }
                } else {
                    // Evaluate next statement
                    let stmt = stmts[stmt_idx].clone();
                    let remaining_stmts: Vec<_> = stmts.into_iter().skip(stmt_idx + 1).collect();

                    let stmt_slot = results.len();
                    results.push(None);

                    work_stack.push(PEWorkItem::ContSwitchCaseBodyStmt {
                        stmt_slot,
                        remaining_stmts,
                        env: env.clone(),
                        result_slot,
                        collected_residuals,
                        last_static,
                    });
                    work_stack.push(PEWorkItem::Eval { expr: stmt, env, result_slot: stmt_slot });
                }
            }

            PEWorkItem::ContSwitchCaseBodyStmt { stmt_slot, remaining_stmts, env, result_slot, mut collected_residuals, last_static } => {
                let pv = results[stmt_slot].take().unwrap();
                match &pv {
                    PValue::Return(inner) => {
                        if collected_residuals.is_empty() {
                            results[result_slot] = Some(PValue::Return(Box::new(inner.as_ref().clone())));
                        } else {
                            collected_residuals.push(Expr::Return(Box::new(residualize(inner))));
                            let filtered = filter_dead_code(collected_residuals);
                            if filtered.is_empty() {
                                results[result_slot] = Some(PValue::Static(Value::Undefined));
                            } else if filtered.len() == 1 {
                                results[result_slot] = Some(PValue::Dynamic(filtered.into_iter().next().unwrap()));
                            } else {
                                results[result_slot] = Some(PValue::Dynamic(Expr::Begin(filtered)));
                            }
                        }
                    }
                    PValue::Dynamic(e) | PValue::DynamicNamed { expr: e, .. } => {
                        let residual = residualize(&pv);
                        let is_break = matches!(e, Expr::Break | Expr::Continue);
                        let is_return = ends_with_return(&residual);
                        let is_throw = ends_with_throw(&residual);
                        if is_break {
                            // Break/continue inside a switch should not propagate outward.
                            // Stop executing this case body without emitting the break.
                            let filtered = filter_dead_code(collected_residuals);
                            if filtered.is_empty() {
                                results[result_slot] = Some(PValue::Static(Value::Undefined));
                            } else if filtered.len() == 1 {
                                results[result_slot] = Some(PValue::Dynamic(filtered.into_iter().next().unwrap()));
                            } else {
                                results[result_slot] = Some(PValue::Dynamic(Expr::Begin(filtered)));
                            }
                        } else if is_return || is_throw {
                            collected_residuals.push(residual);
                            let filtered = filter_dead_code(collected_residuals);
                            if filtered.is_empty() {
                                results[result_slot] = Some(PValue::Static(Value::Undefined));
                            } else if filtered.len() == 1 {
                                results[result_slot] = Some(PValue::Dynamic(filtered.into_iter().next().unwrap()));
                            } else {
                                results[result_slot] = Some(PValue::Dynamic(Expr::Begin(filtered)));
                            }
                        } else {
                            collected_residuals.push(residual);
                            // Continue with remaining statements
                            work_stack.push(PEWorkItem::ContSwitchCaseBody {
                                stmts: remaining_stmts,
                                stmt_idx: 0,
                                env,
                                result_slot,
                                collected_residuals,
                                last_static: None,
                            });
                        }
                    }
                    PValue::Static(v) | PValue::StaticNamed { value: v, .. } => {
                        // Continue with remaining statements
                        work_stack.push(PEWorkItem::ContSwitchCaseBody {
                            stmts: remaining_stmts,
                            stmt_idx: 0,
                            env,
                            result_slot,
                            collected_residuals,
                            last_static: Some(v.clone()),
                        });
                    }
                }
            }

            PEWorkItem::ContWhileCond { cond, body, env, cond_slot, result_slot, initial_env_snapshot } => {
                let cond_pv = results[cond_slot].take().unwrap();
                eval_while_after_cond(cond_pv, cond, body, env, result_slot, initial_env_snapshot, &mut results, &mut work_stack);
            }

            PEWorkItem::ContWhileBody { cond, body, env, body_slot, result_slot, initial_env_snapshot, pre_iteration_snapshot, iterations, unrolled_bodies } => {
                let body_pv = results[body_slot].take().unwrap();
                eval_while_after_body(body_pv, cond, body, env, result_slot, initial_env_snapshot, pre_iteration_snapshot, iterations, unrolled_bodies, &mut results, &mut work_stack);
            }

            PEWorkItem::ContForInit { cond, update, body, env, result_slot, init_expr } => {
                eval_for_after_init(cond, update, body, env, result_slot, init_expr, &mut results, &mut work_stack);
            }

            PEWorkItem::ContForCond { cond, update, body, env, cond_slot, result_slot, iterations, init_expr } => {
                let cond_pv = results[cond_slot].take().unwrap();
                eval_for_after_cond(cond_pv, cond, update, body, env, result_slot, iterations, init_expr, &mut results, &mut work_stack);
            }

            PEWorkItem::ContForBody { cond, update, body, env, body_slot, result_slot, iterations, init_expr } => {
                let body_pv = results[body_slot].take().unwrap();
                eval_for_after_body(body_pv, cond, update, body, env, result_slot, iterations, init_expr, &mut results, &mut work_stack);
            }

            PEWorkItem::ContTryCatch { try_slot, catch_param, catch_block, finally_block, env, result_slot } => {
                let try_pv = results[try_slot].take().unwrap();
                eval_try_catch(try_pv, catch_param, catch_block, finally_block, env, result_slot, &mut results, &mut work_stack);
            }
        }
    }

    results[0].take().unwrap()
}

/// Evaluate a single expression, either immediately or by pushing work items
fn eval_expr(expr: &Expr, env: &PEnv, result_slot: usize, results: &mut Vec<Option<PValue>>, work_stack: &mut Vec<PEWorkItem>) {
    use_gas(expr);
    match expr {
        // Literals are always static - immediate result
        Expr::Int(n) => {
            results[result_slot] = Some(PValue::Static(Value::Int(*n)));
        }
        Expr::Bool(b) => {
            results[result_slot] = Some(PValue::Static(Value::Bool(*b)));
        }
        Expr::String(s) => {
            results[result_slot] = Some(PValue::Static(Value::String(s.clone())));
        }
        Expr::Undefined => {
            results[result_slot] = Some(PValue::Static(Value::Undefined));
        }
        Expr::Null => {
            results[result_slot] = Some(PValue::Static(Value::Null));
        }
        Expr::Break => {
            results[result_slot] = Some(PValue::Dynamic(Expr::Break));
        }
        Expr::Continue => {
            results[result_slot] = Some(PValue::Dynamic(Expr::Continue));
        }
        Expr::Opaque(label) => {
            results[result_slot] = Some(PValue::Static(Value::Opaque {
                label: label.clone(),
                expr: Expr::Opaque(label.clone()),
                state: None,
            }));
        }

        // Variables: look up in env - immediate result
        Expr::Var(name) => {
            let pv = env.borrow()
                .get(name)
                .cloned()
                .unwrap_or_else(|| {
                    if name == "v11" || name == "v9" || name == "v21" || name == "v10" {
                        eprintln!("DEBUG: {} NOT IN ENV, returning Dynamic", name);
                    }
                    PValue::Dynamic(Expr::Var(name.clone()))
                });
            if name == "v11" || name == "v9" || name == "v21" || name == "v10" {
                eprintln!("DEBUG: Var lookup {} = {:?}", name, pv);
            }
            results[result_slot] = Some(pv);
        }

        // Binary operations - need to evaluate both operands first
        Expr::BinOp(op, left, right) => {
            let left_slot = results.len();
            results.push(None);
            let right_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContBinOp {
                op: op.clone(),
                left_slot,
                right_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**right).clone(),
                env: env.clone(),
                result_slot: right_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**left).clone(),
                env: env.clone(),
                result_slot: left_slot,
            });
        }

        // Conditionals - evaluate condition first, then decide which branch
        Expr::If(cond, then_branch, else_branch) => {
            let cond_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContIfCond {
                then_branch: (**then_branch).clone(),
                else_branch: (**else_branch).clone(),
                env: env.clone(),
                cond_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**cond).clone(),
                env: env.clone(),
                result_slot: cond_slot,
            });
        }

        // Let bindings - evaluate value first
        Expr::Let(name, value, body) => {
            let value_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContLetValue {
                name: name.clone(),
                body: (**body).clone(),
                outer_env: env.clone(),
                value_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**value).clone(),
                env: env.clone(),
                result_slot: value_slot,
            });
        }

        // Function definitions - DON'T PE the body at definition time!
        // Just store the function as-is. We'll PE the body at call time
        // when we have actual values for captured variables.
        Expr::Fn(params, body) => {
            // Store the function as a static opaque value so it can be
            // placed in arrays/objects and called later. We still PE the
            // body with params bound to dynamic vars to simplify static
            // control flow when the body is small enough.
            const MAX_FN_PE_NODES: usize = 600;
            let pe_body = if is_pure_expr(body.as_ref())
                && expr_size_with_limit(body.as_ref(), MAX_FN_PE_NODES) <= MAX_FN_PE_NODES
            {
                let fn_env: PEnv = Rc::new(RefCell::new(env.borrow().clone()));
                for param in params {
                    fn_env
                        .borrow_mut()
                        .insert(param.clone(), PValue::Dynamic(Expr::Var(param.clone())));
                }
                let pe = with_gas_disabled(|| partial_eval_sync(body, &fn_env));
                Box::new(residualize(&pe))
            } else {
                body.clone()
            };

            results[result_slot] = Some(PValue::Static(Value::Opaque {
                label: "Function".to_string(),
                expr: Expr::Fn(params.clone(), pe_body),
                state: None,
            }));
        }

        // Function calls - evaluate func and args, then handle call
        Expr::Call(func, args) => {
            // Check for method calls on opaque types first
            if let Expr::PropAccess(obj_expr, method) = func.as_ref() {
                // We need to handle this specially - evaluate obj and args first
                // then check for opaque method handling
                let obj_slot = results.len();
                results.push(None);
                let mut arg_slots = Vec::new();
                for _ in args {
                    arg_slots.push(results.len());
                    results.push(None);
                }

                // Push the method call continuation
                work_stack.push(PEWorkItem::ContCall {
                    func_slot: obj_slot, // We'll reuse this for the method call
                    arg_slots: arg_slots.clone(),
                    result_slot,
                    original_func_expr: (**func).clone(),
                    original_args: args.clone(),
                    env: env.clone(),
                });

                // Push arg evals (reverse order)
                for (i, arg) in args.iter().enumerate().rev() {
                    work_stack.push(PEWorkItem::Eval {
                        expr: arg.clone(),
                        env: env.clone(),
                        result_slot: arg_slots[i],
                    });
                }

                // Push obj eval
                work_stack.push(PEWorkItem::Eval {
                    expr: (**obj_expr).clone(),
                    env: env.clone(),
                    result_slot: obj_slot,
                });
            } else {
                let func_slot = results.len();
                results.push(None);
                let mut arg_slots = Vec::new();
                for _ in args {
                    arg_slots.push(results.len());
                    results.push(None);
                }

                work_stack.push(PEWorkItem::ContCall {
                    func_slot,
                    arg_slots: arg_slots.clone(),
                    result_slot,
                    original_func_expr: (**func).clone(),
                    original_args: args.clone(),
                    env: env.clone(),
                });

                // Push arg evals (reverse order)
                for (i, arg) in args.iter().enumerate().rev() {
                    work_stack.push(PEWorkItem::Eval {
                        expr: arg.clone(),
                        env: env.clone(),
                        result_slot: arg_slots[i],
                    });
                }

                // Push func eval
                work_stack.push(PEWorkItem::Eval {
                    expr: (**func).clone(),
                    env: env.clone(),
                    result_slot: func_slot,
                });
            }
        }

        // Arrays - evaluate all elements
        Expr::Array(elements) => {
            if elements.is_empty() {
                results[result_slot] = Some(PValue::Static(Value::Array(array_from_vec(vec![]))));
            } else {
                let mut elem_slots = Vec::new();
                for _ in elements {
                    elem_slots.push(results.len());
                    results.push(None);
                }

                work_stack.push(PEWorkItem::ContArray {
                    elem_slots: elem_slots.clone(),
                    result_slot,
                });

                for (i, elem) in elements.iter().enumerate().rev() {
                    work_stack.push(PEWorkItem::Eval {
                        expr: elem.clone(),
                        env: env.clone(),
                        result_slot: elem_slots[i],
                    });
                }
            }
        }

        // Array indexing
        Expr::Index(arr, idx) => {
            let arr_slot = results.len();
            results.push(None);
            let idx_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContIndex {
                arr_slot,
                idx_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**idx).clone(),
                env: env.clone(),
                result_slot: idx_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**arr).clone(),
                env: env.clone(),
                result_slot: arr_slot,
            });
        }

        // Array length
        Expr::Len(arr) => {
            let arr_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContLen {
                arr_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**arr).clone(),
                env: env.clone(),
                result_slot: arr_slot,
            });
        }

        // Objects
        Expr::Object(props) => {
            if props.is_empty() {
                results[result_slot] = Some(PValue::Static(Value::Object(new_object())));
            } else {
                let keys: Vec<String> = props.iter().map(|(k, _)| k.clone()).collect();
                let mut value_slots = Vec::new();
                for _ in props {
                    value_slots.push(results.len());
                    results.push(None);
                }

                work_stack.push(PEWorkItem::ContObject {
                    keys,
                    value_slots: value_slots.clone(),
                    result_slot,
                });

                for (i, (_, v)) in props.iter().enumerate().rev() {
                    work_stack.push(PEWorkItem::Eval {
                        expr: v.clone(),
                        env: env.clone(),
                        result_slot: value_slots[i],
                    });
                }
            }
        }

        // Property access
        Expr::PropAccess(obj, prop) => {
            let obj_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContPropAccess {
                obj_slot,
                prop: prop.clone(),
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**obj).clone(),
                env: env.clone(),
                result_slot: obj_slot,
            });
        }

        // Property set
        Expr::PropSet(obj, prop, value) => {
            let obj_slot = results.len();
            results.push(None);
            let value_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContPropSet {
                obj_slot,
                prop: prop.clone(),
                value_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**value).clone(),
                env: env.clone(),
                result_slot: value_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**obj).clone(),
                env: env.clone(),
                result_slot: obj_slot,
            });
        }

        // Computed access
        Expr::ComputedAccess(obj, key) => {
            let obj_slot = results.len();
            results.push(None);
            let key_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContComputedAccess {
                obj_slot,
                key_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**key).clone(),
                env: env.clone(),
                result_slot: key_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**obj).clone(),
                env: env.clone(),
                result_slot: obj_slot,
            });
        }

        // Computed set
        Expr::ComputedSet(obj, key, value) => {
            let obj_slot = results.len();
            results.push(None);
            let key_slot = results.len();
            results.push(None);
            let value_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContComputedSet {
                obj_slot,
                key_slot,
                value_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**value).clone(),
                env: env.clone(),
                result_slot: value_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**key).clone(),
                env: env.clone(),
                result_slot: key_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**obj).clone(),
                env: env.clone(),
                result_slot: obj_slot,
            });
        }

        // Set (mutation)
        Expr::Set(name, value) => {
            let value_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContSet {
                name: name.clone(),
                value_slot,
                result_slot,
                env: env.clone(),
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**value).clone(),
                env: env.clone(),
                result_slot: value_slot,
            });
        }

        // Begin (sequencing)
        Expr::Begin(exprs) => {
            if exprs.is_empty() {
                results[result_slot] = Some(PValue::Static(Value::Bool(false)));
            } else {
                // We need special handling for Begin to track which expressions produce dynamic results
                // For now, evaluate all expressions sequentially
                let mut expr_slots = Vec::new();
                for _ in exprs {
                    expr_slots.push(results.len());
                    results.push(None);
                }

                work_stack.push(PEWorkItem::ContBegin {
                    expr_slots: expr_slots.clone(),
                    result_slot,
                });

                for (i, e) in exprs.iter().enumerate().rev() {
                    work_stack.push(PEWorkItem::Eval {
                        expr: e.clone(),
                        env: env.clone(),
                        result_slot: expr_slots[i],
                    });
                }
            }
        }

        // Unary operations
        Expr::BitNot(inner) => {
            let inner_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContBitNot {
                inner_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**inner).clone(),
                env: env.clone(),
                result_slot: inner_slot,
            });
        }

        Expr::LogNot(inner) => {
            let inner_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContLogNot {
                inner_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**inner).clone(),
                env: env.clone(),
                result_slot: inner_slot,
            });
        }

        Expr::Throw(inner) => {
            let inner_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContThrow {
                inner_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**inner).clone(),
                env: env.clone(),
                result_slot: inner_slot,
            });
        }

        Expr::Return(inner) => {
            let inner_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContReturn {
                inner_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**inner).clone(),
                env: env.clone(),
                result_slot: inner_slot,
            });
        }

        // New expression
        Expr::New(ctor, args) => {
            let ctor_slot = results.len();
            results.push(None);
            let mut arg_slots = Vec::new();
            for _ in args {
                arg_slots.push(results.len());
                results.push(None);
            }

            work_stack.push(PEWorkItem::ContNew {
                ctor_slot,
                arg_slots: arg_slots.clone(),
                result_slot,
                original_ctor: (**ctor).clone(),
                original_args: args.clone(),
                env: env.clone(),
            });

            for (i, arg) in args.iter().enumerate().rev() {
                work_stack.push(PEWorkItem::Eval {
                    expr: arg.clone(),
                    env: env.clone(),
                    result_slot: arg_slots[i],
                });
            }
            work_stack.push(PEWorkItem::Eval {
                expr: (**ctor).clone(),
                env: env.clone(),
                result_slot: ctor_slot,
            });
        }

        // Switch
        Expr::Switch { discriminant, cases, default } => {
            let disc_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContSwitchDiscriminant {
                disc_slot,
                cases: cases.clone(),
                default: default.clone(),
                env: env.clone(),
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**discriminant).clone(),
                env: env.clone(),
                result_slot: disc_slot,
            });
        }

        // While loop
        Expr::While(cond, body) => {
            // Always evaluate condition first - if it's statically false,
            // we don't need to check body for free vars at all
            let initial_env_snapshot: HashMap<String, PValue> =
                env.borrow().iter().map(|(k, v)| (k.clone(), v.clone())).collect();

            let cond_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContWhileCond {
                cond: (**cond).clone(),
                body: (**body).clone(),
                env: env.clone(),
                cond_slot,
                result_slot,
                initial_env_snapshot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**cond).clone(),
                env: env.clone(),
                result_slot: cond_slot,
            });
        }

        // For loop
        Expr::For { init, cond, update, body } => {
            // Handle initialization if present
            if let Some(init_expr) = init {
                work_stack.push(PEWorkItem::ContForInit {
                    cond: cond.as_ref().map(|c| (**c).clone()),
                    update: update.as_ref().map(|u| (**u).clone()),
                    body: (**body).clone(),
                    env: env.clone(),
                    result_slot,
                    init_expr: Some((**init_expr).clone()),
                });
                let init_slot = results.len();
                results.push(None);
                work_stack.push(PEWorkItem::Eval {
                    expr: (**init_expr).clone(),
                    env: env.clone(),
                    result_slot: init_slot,
                });
            } else {
                // No init, go straight to condition
                eval_for_after_init(
                    cond.as_ref().map(|c| (**c).clone()),
                    update.as_ref().map(|u| (**u).clone()),
                    (**body).clone(),
                    env.clone(),
                    result_slot,
                    None,
                    results,
                    work_stack
                );
            }
        }

        // Try-catch
        Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
            let try_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContTryCatch {
                try_slot,
                catch_param: catch_param.clone(),
                catch_block: (**catch_block).clone(),
                finally_block: finally_block.as_ref().map(|fb| (**fb).clone()),
                env: env.clone(),
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: (**try_block).clone(),
                env: env.clone(),
                result_slot: try_slot,
            });
        }
    }
}

// Helper function for synchronous partial eval (used in some edge cases)
// partial_eval already has stacker protection, so this is just an alias
fn partial_eval_sync(expr: &Expr, env: &PEnv) -> PValue {
    partial_eval(expr, env)
}

// ============================================================================
// Helper functions for continuation processing
// ============================================================================

/// Evaluate index operation
fn eval_index(arr_pv: PValue, idx_pv: PValue) -> PValue {
    if let Some(cf) = control_flow_pv(&arr_pv) {
        return cf;
    }
    if let Some(cf) = control_flow_pv(&idx_pv) {
        return cf;
    }
    eprintln!("DEBUG eval_index: arr_pv type={}, idx_pv type={}",
        match &arr_pv {
            PValue::Static(_) => "Static",
            PValue::StaticNamed { .. } => "StaticNamed",
            PValue::Dynamic(_) => "Dynamic",
            PValue::DynamicNamed { .. } => "DynamicNamed",
            PValue::Return(_) => "Return",
        },
        match &idx_pv {
            PValue::Static(_) => "Static",
            PValue::StaticNamed { .. } => "StaticNamed",
            PValue::Dynamic(_) => "Dynamic",
            PValue::DynamicNamed { .. } => "DynamicNamed",
            PValue::Return(_) => "Return",
        }
    );
    if let PValue::StaticNamed { name, value: Value::Array(_), .. } = &arr_pv {
        if !is_internal_name(name) && is_var_mutated(name) {
            return PValue::Dynamic(Expr::Index(
                Box::new(Expr::Var(name.clone())),
                Box::new(residualize(&idx_pv)),
            ));
        }
    }
    // Try Uint8Array index access first
    if let (PValue::Static(Value::Opaque { state: Some(ref s), .. }), PValue::Static(Value::Int(i)))
        | (PValue::StaticNamed { value: Value::Opaque { state: Some(ref s), .. }, .. }, PValue::Static(Value::Int(i))) = (&arr_pv, &idx_pv)
    {
        if let Some(ua) = s.downcast_ref::<Uint8ArrayState>() {
            let i = *i;
            if i < 0 {
                return PValue::Static(Value::Undefined);
            }
            let i = i as usize;
            if i < ua.length {
                if let Some(byte) = ua.buffer.get(ua.byte_offset + i) {
                    return PValue::Static(Value::Int(byte as i64));
                }
            }
            return PValue::Static(Value::Undefined);
        }
    }

    // Extract array elements and index
    let elements = match &arr_pv {
        PValue::Static(Value::Array(e)) => Some(e),
        PValue::StaticNamed { value: Value::Array(e), .. } => Some(e),
        _ => None,
    };
    let index = match &idx_pv {
        PValue::Static(Value::Int(i)) => Some(*i),
        PValue::StaticNamed { value: Value::Int(i), .. } => Some(*i),
        _ => None,
    };

    match (elements, index) {
        (Some(elems), Some(i)) => {
            if i < 0 {
                PValue::Static(Value::Undefined)
            } else {
                let i = i as usize;
                let borrowed = elems.borrow();
                if i < borrowed.len() {
                    PValue::Static(borrowed[i].clone())
                } else {
                    PValue::Static(Value::Undefined)
                }
            }
        }
        _ => PValue::Dynamic(Expr::Index(
            Box::new(residualize(&arr_pv)),
            Box::new(residualize(&idx_pv)),
        )),
    }
}

/// Evaluate property access
fn eval_prop_access(obj_pv: PValue, prop: &str) -> PValue {
    if let Some(cf) = control_flow_pv(&obj_pv) {
        return cf;
    }
    if let PValue::StaticNamed { name, value: Value::Object(_), .. } = &obj_pv {
        if !is_internal_name(name) && is_var_mutated(name) {
            return PValue::Dynamic(Expr::PropAccess(
                Box::new(Expr::Var(name.clone())),
                prop.to_string(),
            ));
        }
    }
    if let PValue::StaticNamed { name, value: Value::Array(_), .. } = &obj_pv {
        if !is_internal_name(name) && is_var_mutated(name) {
            return PValue::Dynamic(Expr::PropAccess(
                Box::new(Expr::Var(name.clone())),
                prop.to_string(),
            ));
        }
    }

    match &obj_pv {
        PValue::Static(Value::Object(obj_ref)) |
        PValue::StaticNamed { value: Value::Object(obj_ref), .. } => {
            if let Some(v) = obj_ref.borrow().get(prop) {
                PValue::Static(v.clone())
            } else {
                PValue::Static(Value::Undefined)
            }
        }
        PValue::Static(Value::Array(arr)) |
        PValue::StaticNamed { value: Value::Array(arr), .. } if prop == "length" => {
            PValue::Static(Value::Int(arr.borrow().len() as i64))
        }
        PValue::Static(Value::String(s)) |
        PValue::StaticNamed { value: Value::String(s), .. } if prop == "length" => {
            PValue::Static(Value::Int(s.len() as i64))
        }
        // Handle opaque types with state
        PValue::Static(Value::Opaque { state: Some(ref s), expr, .. }) |
        PValue::StaticNamed { value: Value::Opaque { state: Some(ref s), expr, .. }, .. } => {
            // Try ArrayBuffer
            if let Some(ab) = s.downcast_ref::<ArrayBufferState>() {
                if prop == "byteLength" {
                    return PValue::Static(Value::Int(ab.buffer.len() as i64));
                }
            }
            // Try DataView
            if let Some(dv) = s.downcast_ref::<DataViewState>() {
                match prop {
                    "byteLength" => return PValue::Static(Value::Int(dv.byte_length as i64)),
                    "byteOffset" => return PValue::Static(Value::Int(dv.byte_offset as i64)),
                    "buffer" => {
                        return PValue::Static(Value::Opaque {
                            label: "ArrayBuffer".into(),
                            expr: Expr::PropAccess(Box::new(expr.clone()), "buffer".into()),
                            state: Some(Rc::new(ArrayBufferState { buffer: dv.buffer.clone() }) as Rc<dyn Any>),
                        });
                    }
                    _ => {}
                }
            }
            // Try Uint8Array
            if let Some(ua) = s.downcast_ref::<Uint8ArrayState>() {
                match prop {
                    "length" => return PValue::Static(Value::Int(ua.length as i64)),
                    "byteLength" => return PValue::Static(Value::Int(ua.length as i64)),
                    "byteOffset" => return PValue::Static(Value::Int(ua.byte_offset as i64)),
                    "buffer" => {
                        return PValue::Static(Value::Opaque {
                            label: "ArrayBuffer".into(),
                            expr: Expr::PropAccess(Box::new(expr.clone()), "buffer".into()),
                            state: Some(Rc::new(ArrayBufferState { buffer: ua.buffer.clone() }) as Rc<dyn Any>),
                        });
                    }
                    _ => {}
                }
            }
            PValue::Dynamic(Expr::PropAccess(Box::new(residualize(&obj_pv)), prop.to_string()))
        }
        _ => PValue::Dynamic(Expr::PropAccess(Box::new(residualize(&obj_pv)), prop.to_string())),
    }
}

/// Evaluate property set
fn eval_prop_set(obj_pv: PValue, prop: &str, value_pv: PValue) -> PValue {
    if let Some(cf) = control_flow_pv(&obj_pv) {
        return cf;
    }
    if let Some(cf) = control_flow_pv(&value_pv) {
        return cf;
    }
    let (obj_ref, obj_name) = match &obj_pv {
        PValue::Static(Value::Object(r)) => (Some(r.clone()), None),
        PValue::StaticNamed { name, value: Value::Object(r), .. } => (Some(r.clone()), Some(name.clone())),
        _ => (None, None),
    };
    let static_value = match &value_pv {
        PValue::Static(v) => Some(v.clone()),
        PValue::StaticNamed { value: v, .. } => Some(v.clone()),
        _ => None,
    };

    if let (Some(obj_r), Some(v)) = (&obj_ref, &static_value) {
        if let Some(name) = obj_name {
            if !is_internal_name(&name) {
                mark_var_mutated(&name);
                return PValue::Dynamic(Expr::PropSet(
                    Box::new(Expr::Var(name)),
                    prop.to_string(),
                    Box::new(residualize(&value_pv)),
                ));
            }
        }
        obj_r.borrow_mut().insert(prop.to_string(), v.clone());
        return PValue::Static(v.clone());
    }

    PValue::Dynamic(Expr::PropSet(
        Box::new(residualize(&obj_pv)),
        prop.to_string(),
        Box::new(residualize(&value_pv)),
    ))
}

/// Evaluate computed access
fn eval_computed_access(obj_pv: PValue, key_pv: PValue) -> PValue {
    if let Some(cf) = control_flow_pv(&obj_pv) {
        return cf;
    }
    if let Some(cf) = control_flow_pv(&key_pv) {
        return cf;
    }
    if let PValue::StaticNamed { name, value: Value::Object(_), .. } = &obj_pv {
        if !is_internal_name(name) && is_var_mutated(name) {
            return PValue::Dynamic(Expr::ComputedAccess(
                Box::new(Expr::Var(name.clone())),
                Box::new(residualize(&key_pv)),
            ));
        }
    }
    if let PValue::StaticNamed { name, value: Value::Array(_), .. } = &obj_pv {
        if !is_internal_name(name) && is_var_mutated(name) {
            return PValue::Dynamic(Expr::ComputedAccess(
                Box::new(Expr::Var(name.clone())),
                Box::new(residualize(&key_pv)),
            ));
        }
    }

    match (&obj_pv, &key_pv) {
        (PValue::Static(Value::Object(obj_ref)), PValue::Static(Value::String(k))) |
        (PValue::StaticNamed { value: Value::Object(obj_ref), .. }, PValue::Static(Value::String(k))) |
        (PValue::Static(Value::Object(obj_ref)), PValue::StaticNamed { value: Value::String(k), .. }) |
        (PValue::StaticNamed { value: Value::Object(obj_ref), .. }, PValue::StaticNamed { value: Value::String(k), .. }) => {
            if let Some(v) = obj_ref.borrow().get(k) {
                PValue::Static(v.clone())
            } else {
                PValue::Static(Value::Undefined)
            }
        }
        (PValue::Static(Value::Array(arr)), PValue::Static(Value::Int(i))) |
        (PValue::StaticNamed { value: Value::Array(arr), .. }, PValue::Static(Value::Int(i))) => {
            let i = *i as usize;
            let borrowed = arr.borrow();
            if i < borrowed.len() {
                PValue::Static(borrowed[i].clone())
            } else {
                PValue::Static(Value::Undefined)
            }
        }
        (PValue::Static(Value::Array(arr)), PValue::Static(Value::String(s))) |
        (PValue::StaticNamed { value: Value::Array(arr), .. }, PValue::Static(Value::String(s))) if s == "length" => {
            PValue::Static(Value::Int(arr.borrow().len() as i64))
        }
        _ => PValue::Dynamic(Expr::ComputedAccess(
            Box::new(residualize(&obj_pv)),
            Box::new(residualize(&key_pv)),
        )),
    }
}

/// Evaluate computed set
fn eval_computed_set(obj_pv: PValue, key_pv: PValue, value_pv: PValue) -> PValue {
    if let Some(cf) = control_flow_pv(&obj_pv) {
        return cf;
    }
    if let Some(cf) = control_flow_pv(&key_pv) {
        return cf;
    }
    if let Some(cf) = control_flow_pv(&value_pv) {
        return cf;
    }
    let static_value = match &value_pv {
        PValue::Static(v) => Some(v.clone()),
        PValue::StaticNamed { value: v, .. } => Some(v.clone()),
        _ => None,
    };

    // Try array with integer index
    let (arr_ref, arr_name) = match &obj_pv {
        PValue::Static(Value::Array(r)) => (Some(r.clone()), None),
        PValue::StaticNamed { name, value: Value::Array(r), .. } => (Some(r.clone()), Some(name.clone())),
        _ => (None, None),
    };
    let key_int = match &key_pv {
        PValue::Static(Value::Int(i)) => Some(*i),
        PValue::StaticNamed { value: Value::Int(i), .. } => Some(*i),
        _ => None,
    };

    if let (Some(arr), Some(i), Some(v)) = (arr_ref, key_int, static_value.clone()) {
        if i >= 0 {
            let i = i as usize;
            if let Some(name) = arr_name {
                if !is_internal_name(&name) {
                    mark_var_mutated(&name);
                    return PValue::Dynamic(Expr::ComputedSet(
                        Box::new(Expr::Var(name)),
                        Box::new(residualize(&key_pv)),
                        Box::new(residualize(&value_pv)),
                    ));
                }
            }
            let mut borrowed = arr.borrow_mut();
            if i >= borrowed.len() {
                borrowed.resize(i + 1, Value::Undefined);
            }
            borrowed[i] = v.clone();
            return PValue::Static(v);
        }
    }

    // Try object with string key
    let (obj_ref, obj_name) = match &obj_pv {
        PValue::Static(Value::Object(r)) => (Some(r.clone()), None),
        PValue::StaticNamed { name, value: Value::Object(r), .. } => (Some(r.clone()), Some(name.clone())),
        _ => (None, None),
    };
    let key_str = match &key_pv {
        PValue::Static(Value::String(k)) => Some(k.clone()),
        PValue::StaticNamed { value: Value::String(k), .. } => Some(k.clone()),
        _ => None,
    };

    match (obj_ref, key_str, static_value) {
        (Some(obj_r), Some(k), Some(v)) => {
            if let Some(name) = obj_name {
                if !is_internal_name(&name) {
                    mark_var_mutated(&name);
                    return PValue::Dynamic(Expr::ComputedSet(
                        Box::new(Expr::Var(name)),
                        Box::new(residualize(&key_pv)),
                        Box::new(residualize(&value_pv)),
                    ));
                }
            }
            obj_r.borrow_mut().insert(k, v.clone());
            PValue::Static(v)
        }
        _ => PValue::Dynamic(Expr::ComputedSet(
            Box::new(residualize(&obj_pv)),
            Box::new(residualize(&key_pv)),
            Box::new(residualize(&value_pv)),
        )),
    }
}

/// Evaluate set (mutation)
fn eval_set(name: &str, value_pv: PValue, env: &PEnv) -> PValue {
    if let Some(cf) = control_flow_pv(&value_pv) {
        return cf;
    }
    // Debug: trace v35, v23, v37 assignments
    if name == "v35" || name == "v23" || name == "v37" || name == "v32" || name == "v10" {
        match &value_pv {
            PValue::Static(v) => {
                if let Value::Int(n) = v {
                    eprintln!("DEBUG: {} assigned static value {}", name, n);
                } else {
                    eprintln!("DEBUG: {} assigned static non-int: {:?}", name, v);
                }
            }
            PValue::Dynamic(e) => {
                eprintln!("DEBUG: {} assigned DYNAMIC: {:?}", name, e);
            }
            PValue::Return(_) => {
                eprintln!("DEBUG: {} assigned RETURN", name);
            }
            _ => {
                eprintln!("DEBUG: {} assigned other: {:?}", name, value_pv);
            }
        }
    }

    // Only mark as mutated if this is a reassignment (variable already exists)
    // Initial assignments shouldn't be treated as mutations for closure capture purposes
    if env.borrow().contains_key(name) {
        mark_var_mutated(name);
    }

    if env.borrow().contains_key(name) {
        let was_dynamic = matches!(
            env.borrow().get(name),
            Some(PValue::Dynamic(_)) | Some(PValue::DynamicNamed { .. })
        );

        let binding = match &value_pv {
            PValue::Static(v) if is_mutable_value(v) => {
                PValue::StaticNamed { name: name.to_string(), value: v.clone() }
            }
            PValue::Static(v) => PValue::Static(v.clone()),
            PValue::StaticNamed { value, .. } => {
                PValue::StaticNamed { name: name.to_string(), value: value.clone() }
            }
            PValue::Dynamic(e) => {
                if name == "v11" {
                    eprintln!("DEBUG: v11 SET to DynamicNamed (reassign from Dynamic), expr variant: {:?}", std::mem::discriminant(e));
                }
                PValue::DynamicNamed { name: name.to_string(), expr: e.clone() }
            }
            PValue::DynamicNamed { expr, .. } => {
                if name == "v11" {
                    eprintln!("DEBUG: v11 SET to DynamicNamed (reassign from DynamicNamed)");
                }
                PValue::DynamicNamed { name: name.to_string(), expr: expr.clone() }
            }
            PValue::Return(inner) => PValue::Return(Box::new(inner.as_ref().clone())),
        };
        if name == "v9" || name == "v11" || name == "v21" || name == "v10" {
            eprintln!("DEBUG: {} reassignment (was_dynamic={}), binding={:?}, value_pv={:?}, env_ptr={:p}",
                name,
                was_dynamic,
                match &binding {
                    PValue::Static(_) => "Static",
                    PValue::StaticNamed { .. } => "StaticNamed",
                    PValue::Dynamic(_) => "Dynamic",
                    PValue::DynamicNamed { .. } => "DynamicNamed",
                    PValue::Return(_) => "Return",
                },
                match &value_pv {
                    PValue::Static(_) => "Static",
                    PValue::StaticNamed { .. } => "StaticNamed",
                    PValue::Dynamic(_) => "Dynamic",
                    PValue::DynamicNamed { .. } => "DynamicNamed",
                    PValue::Return(_) => "Return",
                },
                env.as_ptr());
        }
        env.borrow_mut().insert(name.to_string(), binding.clone());

        if is_internal_name(name) {
            if matches!(binding, PValue::Static(_) | PValue::StaticNamed { .. }) {
                return binding;
            }
        }

        match &binding {
            PValue::Static(v) if !is_mutable_value(v) && !was_dynamic => binding,
            _ => PValue::Dynamic(Expr::Set(name.to_string(), Box::new(residualize(&value_pv)))),
        }
    } else {
        // Variable not in env - this is a global assignment
        // Still insert it into the env so later accesses can find it
        let binding = match &value_pv {
            PValue::Static(v) if is_mutable_value(v) => {
                PValue::StaticNamed { name: name.to_string(), value: v.clone() }
            }
            PValue::Static(v) => PValue::Static(v.clone()),
            PValue::StaticNamed { value, .. } => {
                PValue::StaticNamed { name: name.to_string(), value: value.clone() }
            }
            PValue::Dynamic(e) => {
                PValue::DynamicNamed { name: name.to_string(), expr: e.clone() }
            }
            PValue::DynamicNamed { expr, .. } => {
                PValue::DynamicNamed { name: name.to_string(), expr: expr.clone() }
            }
            PValue::Return(inner) => PValue::Return(Box::new(inner.as_ref().clone())),
        };
        if name == "v9" || name == "v10" {
            eprintln!("DEBUG: {} global assignment, binding={:?}", name,
                match &binding {
                    PValue::Static(_) => "Static",
                    PValue::StaticNamed { .. } => "StaticNamed",
                    PValue::Dynamic(_) => "Dynamic",
                    PValue::DynamicNamed { .. } => "DynamicNamed",
                    PValue::Return(_) => "Return",
                });
        }
        env.borrow_mut().insert(name.to_string(), binding.clone());
        if is_internal_name(name) {
            if matches!(binding, PValue::Static(_) | PValue::StaticNamed { .. }) {
                return binding;
            }
        }
        PValue::Dynamic(Expr::Set(name.to_string(), Box::new(residualize(&value_pv))))
    }
}

/// Evaluate begin (sequencing)
fn eval_begin(pvs: Vec<PValue>) -> PValue {
    if pvs.is_empty() {
        return PValue::Static(Value::Bool(false));
    }

    let mut has_dynamic = false;
    let mut residual_exprs: Vec<Expr> = Vec::new();

    for (i, pv) in pvs.iter().enumerate() {
        let is_last = i == pvs.len() - 1;
        if let PValue::Return(inner) = pv {
            if residual_exprs.is_empty() {
                return PValue::Return(Box::new(inner.as_ref().clone()));
            }
            residual_exprs.push(Expr::Return(Box::new(residualize(inner))));
            let result_expr = if residual_exprs.len() == 1 {
                residual_exprs.pop().unwrap()
            } else {
                Expr::Begin(residual_exprs)
            };
            return PValue::Dynamic(result_expr);
        }

        let is_dynamic = matches!(pv, PValue::Dynamic(_) | PValue::DynamicNamed { .. });
        if is_dynamic {
            has_dynamic = true;
        }

        if has_dynamic {
            let residual = residualize(pv);
            let is_return = ends_with_return(&residual);
            let is_throw = ends_with_throw(&residual);
            if is_last || !is_pure_expr(&residual) || is_return || is_throw {
                residual_exprs.push(residual.clone());
            }
            if is_return || is_throw {
                let result_expr = if residual_exprs.len() == 1 {
                    residual_exprs.pop().unwrap()
                } else {
                    Expr::Begin(residual_exprs)
                };
                return PValue::Dynamic(result_expr);
            }
        }
    }

    if has_dynamic {
        if residual_exprs.is_empty() {
            PValue::Static(Value::Undefined)
        } else if residual_exprs.len() == 1 {
            PValue::Dynamic(residual_exprs.pop().unwrap())
        } else {
            PValue::Dynamic(Expr::Begin(residual_exprs))
        }
    } else {
        pvs.into_iter().last().unwrap_or(PValue::Static(Value::Bool(false)))
    }
}

/// Evaluate logical not
fn eval_log_not(inner_pv: PValue) -> PValue {
    if let Some(cf) = control_flow_pv(&inner_pv) {
        return cf;
    }
    match inner_pv {
        PValue::Static(Value::Bool(b)) => PValue::Static(Value::Bool(!b)),
        PValue::Static(Value::Int(0)) => PValue::Static(Value::Bool(true)),
        PValue::Static(Value::Int(_)) => PValue::Static(Value::Bool(false)),
        PValue::Static(Value::String(ref s)) if s.is_empty() => PValue::Static(Value::Bool(true)),
        PValue::Static(Value::String(_)) => PValue::Static(Value::Bool(false)),
        PValue::Static(Value::Undefined) => PValue::Static(Value::Bool(true)),
        PValue::Static(Value::Null) => PValue::Static(Value::Bool(true)),
        _ => {
            let inner_residual = residualize(&inner_pv);
            // Double negation elimination: !!x → x
            if let Expr::LogNot(inner_inner) = inner_residual {
                PValue::Dynamic(*inner_inner)
            } else {
                PValue::Dynamic(Expr::LogNot(Box::new(inner_residual)))
            }
        }
    }
}

/// Handle function call
fn handle_call(
    func_pv: PValue,
    args_pv: Vec<PValue>,
    env: &PEnv,
    original_func_expr: &Expr,
    original_args: &[Expr],
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
    result_slot: usize,
) -> Option<PValue> {
    // Debug: trace all function calls
    if let Expr::Var(ref name) = original_func_expr {
        if name == "v2" {
            let type_str = match &func_pv {
                PValue::Static(Value::Closure { .. }) => "Static Closure".to_string(),
                PValue::Static(_) => "Static other".to_string(),
                PValue::Dynamic(e) => format!("Dynamic({:?})", std::mem::discriminant(e)),
                PValue::StaticNamed { .. } => "StaticNamed".to_string(),
                PValue::DynamicNamed { expr, .. } => format!("DynamicNamed(expr={:?})", std::mem::discriminant(expr)),
                PValue::Return(_) => "Return".to_string(),
            };
            eprintln!("DEBUG handle_call: v2() called, func_pv type: {}", type_str);
        }
    }

    // Check for method calls on opaque types with state (DataView, TextDecoder, etc.)
    if let Expr::PropAccess(obj_expr, method) = original_func_expr {
        // The func_pv is actually the object (we evaluated the obj part)
        let obj_pv = func_pv;

        // Extract DataView state
        let dv_info: Option<(&DataViewState, Expr)> = match &obj_pv {
            PValue::Static(Value::Opaque { state: Some(ref s), expr, .. }) => {
                s.downcast_ref::<DataViewState>().map(|dv| (dv, expr.clone()))
            }
            PValue::StaticNamed { name, value: Value::Opaque { state: Some(ref s), .. } } => {
                s.downcast_ref::<DataViewState>().map(|dv| (dv, Expr::Var(name.clone())))
            }
            _ => None,
        };

        if let Some((dv, receiver_expr)) = dv_info {
            match method.as_str() {
                "getInt8" | "getUint8" => {
                    if let Some(PValue::Static(Value::Int(offset))) = args_pv.first() {
                        let offset = *offset as usize + dv.byte_offset;
                        if offset < dv.byte_offset + dv.byte_length {
                            if let Some(byte) = dv.buffer.get(offset) {
                                let value = if method == "getInt8" {
                                    (byte as i8) as i64
                                } else {
                                    byte as i64
                                };
                                return Some(PValue::Static(Value::Int(value)));
                            }
                        }
                    }
                }
                "setInt8" | "setUint8" => {
                    if args_pv.len() >= 2 {
                        if let (Some(PValue::Static(Value::Int(offset))), Some(PValue::Static(Value::Int(val)))) =
                            (args_pv.get(0), args_pv.get(1))
                        {
                            let offset = *offset as usize + dv.byte_offset;
                            if offset < dv.byte_offset + dv.byte_length {
                                dv.buffer.set(offset, *val as u8);
                                return Some(PValue::Static(Value::Undefined));
                            }
                        }
                    }
                }
                "getInt16" | "getUint16" => {
                    if let Some(PValue::Static(Value::Int(offset))) = args_pv.first() {
                        let little_endian = args_pv.get(1)
                            .map(|pv| matches!(pv, PValue::Static(Value::Bool(true))))
                            .unwrap_or(false);
                        let offset = *offset as usize + dv.byte_offset;
                        if offset + 1 < dv.byte_offset + dv.byte_length {
                            if let (Some(b0), Some(b1)) = (dv.buffer.get(offset), dv.buffer.get(offset + 1)) {
                                let value = if little_endian {
                                    (b1 as u16) << 8 | (b0 as u16)
                                } else {
                                    (b0 as u16) << 8 | (b1 as u16)
                                };
                                let value = if method == "getInt16" {
                                    (value as i16) as i64
                                } else {
                                    value as i64
                                };
                                return Some(PValue::Static(Value::Int(value)));
                            }
                        }
                    }
                }
                "setInt16" | "setUint16" => {
                    if args_pv.len() >= 2 {
                        if let (Some(PValue::Static(Value::Int(offset))), Some(PValue::Static(Value::Int(val)))) =
                            (args_pv.get(0), args_pv.get(1))
                        {
                            let little_endian = args_pv.get(2)
                                .map(|pv| matches!(pv, PValue::Static(Value::Bool(true))))
                                .unwrap_or(false);
                            let offset = *offset as usize + dv.byte_offset;
                            if offset + 1 < dv.byte_offset + dv.byte_length {
                                let val = *val as u16;
                                if little_endian {
                                    dv.buffer.set(offset, val as u8);
                                    dv.buffer.set(offset + 1, (val >> 8) as u8);
                                } else {
                                    dv.buffer.set(offset, (val >> 8) as u8);
                                    dv.buffer.set(offset + 1, val as u8);
                                }
                                return Some(PValue::Static(Value::Undefined));
                            }
                        }
                    }
                }
                "getInt32" | "getUint32" => {
                    if let Some(PValue::Static(Value::Int(offset))) = args_pv.first() {
                        let little_endian = args_pv.get(1)
                            .map(|pv| matches!(pv, PValue::Static(Value::Bool(true))))
                            .unwrap_or(false);
                        let offset = *offset as usize + dv.byte_offset;
                        if offset + 3 < dv.byte_offset + dv.byte_length {
                            if let (Some(b0), Some(b1), Some(b2), Some(b3)) =
                                (dv.buffer.get(offset), dv.buffer.get(offset + 1),
                                 dv.buffer.get(offset + 2), dv.buffer.get(offset + 3))
                            {
                                let value = if little_endian {
                                    (b3 as u32) << 24 | (b2 as u32) << 16 | (b1 as u32) << 8 | (b0 as u32)
                                } else {
                                    (b0 as u32) << 24 | (b1 as u32) << 16 | (b2 as u32) << 8 | (b3 as u32)
                                };
                                let value = if method == "getInt32" {
                                    (value as i32) as i64
                                } else {
                                    value as i64
                                };
                                return Some(PValue::Static(Value::Int(value)));
                            }
                        }
                    }
                }
                "setInt32" | "setUint32" => {
                    if args_pv.len() >= 2 {
                        if let (Some(PValue::Static(Value::Int(offset))), Some(PValue::Static(Value::Int(val)))) =
                            (args_pv.get(0), args_pv.get(1))
                        {
                            let little_endian = args_pv.get(2)
                                .map(|pv| matches!(pv, PValue::Static(Value::Bool(true))))
                                .unwrap_or(false);
                            let offset = *offset as usize + dv.byte_offset;
                            if offset + 3 < dv.byte_offset + dv.byte_length {
                                let val = *val as u32;
                                if little_endian {
                                    dv.buffer.set(offset, val as u8);
                                    dv.buffer.set(offset + 1, (val >> 8) as u8);
                                    dv.buffer.set(offset + 2, (val >> 16) as u8);
                                    dv.buffer.set(offset + 3, (val >> 24) as u8);
                                } else {
                                    dv.buffer.set(offset, (val >> 24) as u8);
                                    dv.buffer.set(offset + 1, (val >> 16) as u8);
                                    dv.buffer.set(offset + 2, (val >> 8) as u8);
                                    dv.buffer.set(offset + 3, val as u8);
                                }
                                return Some(PValue::Static(Value::Undefined));
                            }
                        }
                    }
                }
                _ => {}
            }
            // Fall through to residual for unhandled methods or dynamic args
            return Some(PValue::Dynamic(Expr::Call(
                Box::new(Expr::PropAccess(Box::new(receiver_expr.clone()), method.clone())),
                args_pv.iter().map(residualize).collect(),
            )));
        }

        // Try TextDecoder methods
        let td_info: Option<(&TextDecoderState, Expr)> = match &obj_pv {
            PValue::Static(Value::Opaque { state: Some(ref s), expr, .. }) => {
                s.downcast_ref::<TextDecoderState>().map(|td| (td, expr.clone()))
            }
            PValue::StaticNamed { name, value: Value::Opaque { state: Some(ref s), .. } } => {
                s.downcast_ref::<TextDecoderState>().map(|td| (td, Expr::Var(name.clone())))
            }
            _ => None,
        };

        if let Some((td, receiver_expr)) = td_info {
            if method == "decode" {
                if let Some(arg_pv) = args_pv.first() {
                    let ua_state: Option<&Uint8ArrayState> = match arg_pv {
                        PValue::Static(Value::Opaque { state: Some(ref s), .. }) => {
                            s.downcast_ref::<Uint8ArrayState>()
                        }
                        PValue::StaticNamed { value: Value::Opaque { state: Some(ref s), .. }, .. } => {
                            s.downcast_ref::<Uint8ArrayState>()
                        }
                        _ => None,
                    };

                    if let Some(ua) = ua_state {
                        let bytes: Vec<u8> = (0..ua.length)
                            .filter_map(|i| ua.buffer.get(ua.byte_offset + i))
                            .collect();

                        let decoded = match td.encoding.as_str() {
                            "utf-8" | "utf8" => String::from_utf8(bytes.clone()).ok(),
                            "ascii" | "us-ascii" => {
                                if bytes.iter().all(|&b| b < 128) {
                                    String::from_utf8(bytes.clone()).ok()
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };

                        if let Some(s) = decoded {
                            return Some(PValue::Static(Value::String(s)));
                        }
                    }
                }
            }
            return Some(PValue::Dynamic(Expr::Call(
                Box::new(Expr::PropAccess(Box::new(receiver_expr.clone()), method.clone())),
                args_pv.iter().map(residualize).collect(),
            )));
        }

        // Try Array methods (push, slice, etc.)
        let (arr_ref, arr_name): (Option<Rc<RefCell<Vec<Value>>>>, Option<String>) = match &obj_pv {
            PValue::Static(Value::Array(r)) => (Some(r.clone()), None),
            PValue::StaticNamed { name, value: Value::Array(r), .. } => (Some(r.clone()), Some(name.clone())),
            _ => (None, None),
        };

        if let Some(arr) = arr_ref {
            match method.as_str() {
                "push" => {
                    // Push all static arguments to the array
                    let all_static = args_pv.iter().all(|pv| matches!(pv, PValue::Static(_) | PValue::StaticNamed { .. }));
                    if all_static {
                        // push returns the new length
                        if let Some(name) = &arr_name {
                            if !is_internal_name(name) {
                                mark_var_mutated(name);
                                return Some(PValue::Dynamic(Expr::Call(
                                    Box::new(Expr::PropAccess(Box::new(Expr::Var(name.clone())), method.clone())),
                                    args_pv.iter().map(residualize).collect(),
                                )));
                            }
                        }
                        let mut borrowed = arr.borrow_mut();
                        for arg_pv in &args_pv {
                            let value = match arg_pv {
                                PValue::Static(v) => v.clone(),
                                PValue::StaticNamed { value: v, .. } => v.clone(),
                                _ => unreachable!(),
                            };
                            borrowed.push(value);
                        }
                        return Some(PValue::Static(Value::Int(borrowed.len() as i64)));
                    }
                    // Dynamic args - fall through to residualization
                }
                "slice" => {
                    // slice(start, end?) returns a new array
                    let start = match args_pv.get(0) {
                        Some(PValue::Static(Value::Int(n))) => Some(*n as usize),
                        Some(PValue::StaticNamed { value: Value::Int(n), .. }) => Some(*n as usize),
                        None => Some(0),
                        _ => None,
                    };
                    let borrowed = arr.borrow();
                    let end = match args_pv.get(1) {
                        Some(PValue::Static(Value::Int(n))) => Some(*n as usize),
                        Some(PValue::StaticNamed { value: Value::Int(n), .. }) => Some(*n as usize),
                        None => Some(borrowed.len()),
                        _ => None,
                    };
                    if let (Some(s), Some(e)) = (start, end) {
                        let s = s.min(borrowed.len());
                        let e = e.min(borrowed.len());
                        let sliced: Vec<Value> = borrowed[s..e].to_vec();
                        drop(borrowed);
                        return Some(PValue::Static(Value::Array(array_from_vec(sliced))));
                    }
                }
                "pop" => {
                    let mut borrowed = arr.borrow_mut();
                    let popped = borrowed.pop().unwrap_or(Value::Undefined);
                    if let Some(name) = &arr_name {
                        if !is_internal_name(name) {
                            mark_var_mutated(name);
                            return Some(PValue::Dynamic(Expr::Call(
                                Box::new(Expr::PropAccess(Box::new(Expr::Var(name.clone())), method.clone())),
                                args_pv.iter().map(residualize).collect(),
                            )));
                        }
                    }
                    return Some(PValue::Static(popped));
                }
                "shift" => {
                    let mut borrowed = arr.borrow_mut();
                    if borrowed.is_empty() {
                        if let Some(name) = &arr_name {
                            if !is_internal_name(name) {
                                mark_var_mutated(name);
                                return Some(PValue::Dynamic(Expr::Call(
                                    Box::new(Expr::PropAccess(Box::new(Expr::Var(name.clone())), method.clone())),
                                    args_pv.iter().map(residualize).collect(),
                                )));
                            }
                        }
                        return Some(PValue::Static(Value::Undefined));
                    }
                    let shifted = borrowed.remove(0);
                    if let Some(name) = &arr_name {
                        if !is_internal_name(name) {
                            mark_var_mutated(name);
                            return Some(PValue::Dynamic(Expr::Call(
                                Box::new(Expr::PropAccess(Box::new(Expr::Var(name.clone())), method.clone())),
                                args_pv.iter().map(residualize).collect(),
                            )));
                        }
                    }
                    return Some(PValue::Static(shifted));
                }
                "unshift" => {
                    let all_static = args_pv.iter().all(|pv| matches!(pv, PValue::Static(_) | PValue::StaticNamed { .. }));
                    if all_static {
                        if let Some(name) = &arr_name {
                            if !is_internal_name(name) {
                                mark_var_mutated(name);
                                return Some(PValue::Dynamic(Expr::Call(
                                    Box::new(Expr::PropAccess(Box::new(Expr::Var(name.clone())), method.clone())),
                                    args_pv.iter().map(residualize).collect(),
                                )));
                            }
                        }
                        let mut borrowed = arr.borrow_mut();
                        for (i, arg_pv) in args_pv.iter().enumerate() {
                            let value = match arg_pv {
                                PValue::Static(v) => v.clone(),
                                PValue::StaticNamed { value: v, .. } => v.clone(),
                                _ => unreachable!(),
                            };
                            borrowed.insert(i, value);
                        }
                        return Some(PValue::Static(Value::Int(borrowed.len() as i64)));
                    }
                }
                _ => {}
            }
            // Fall through to residualization for unhandled methods or dynamic args
            return Some(PValue::Dynamic(Expr::Call(
                Box::new(Expr::PropAccess(Box::new(residualize(&obj_pv)), method.clone())),
                args_pv.iter().map(residualize).collect(),
            )));
        }

        // Not a special opaque method - fall through to normal call handling
        // But we need to reconstruct the func_pv as the PropAccess
        let func_pv_reconstructed = eval_prop_access(obj_pv, method);
        return handle_normal_call(func_pv_reconstructed, args_pv, env, results, work_stack, result_slot);
    }

    handle_normal_call(func_pv, args_pv, env, results, work_stack, result_slot)
}

fn handle_normal_call(
    func_pv: PValue,
    args_pv: Vec<PValue>,
    env: &PEnv,
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
    result_slot: usize,
) -> Option<PValue> {
    // Debug: trace function calls
    if let PValue::Dynamic(Expr::Var(ref name)) = func_pv {
        if name == "v2" {
            eprintln!("DEBUG: v2 called with func_pv = Dynamic(Var(v2))");
        }
    }
    if let PValue::Static(Value::Closure { .. }) = func_pv {
        eprintln!("DEBUG: Calling a Static Closure");
    }
    if let PValue::Dynamic(Expr::Fn(_, _)) = func_pv {
        eprintln!("DEBUG: Calling a Dynamic Fn");
    }

    match func_pv {
        // Functions stored as opaque static values (e.g., in arrays/objects)
        PValue::Static(Value::Opaque { expr: Expr::Fn(params, body), .. }) => {
            if params.len() != args_pv.len() {
                return Some(PValue::Dynamic(Expr::Call(
                    Box::new(Expr::Fn(params, body)),
                    args_pv.iter().map(residualize).collect(),
                )));
            }

            // Save any existing bindings that would be shadowed by parameters
            let saved_bindings: Vec<_> = params.iter()
                .filter_map(|p| env.borrow().get(p).cloned().map(|v| (p.clone(), v)))
                .collect();

            // Bind parameters to arguments in the current env
            for (param, arg_pv) in params.iter().zip(args_pv.iter()) {
                env.borrow_mut().insert(param.clone(), arg_pv.clone());
            }

            let result = unwrap_return_pv(partial_eval_sync(&body, env));

            // Restore any shadowed bindings
            for (param, saved) in &saved_bindings {
                env.borrow_mut().insert(param.clone(), saved.clone());
            }
            // Remove parameters that weren't shadowing anything
            for param in &params {
                if !saved_bindings.iter().any(|(p, _)| p == param) {
                    env.borrow_mut().remove(param);
                }
            }

            Some(result)
        }

        PValue::Static(Value::Closure { params, body, env: closure_env }) => {
            if params.len() != args_pv.len() {
                return Some(PValue::Dynamic(Expr::Call(
                    Box::new(residualize(&PValue::Static(Value::Closure {
                        params,
                        body,
                        env: closure_env,
                    }))),
                    args_pv.iter().map(residualize).collect(),
                )));
            }

            // Check if this closure reads from outer variables that are mutated
            // If so, we should NOT inline the call during definition of another closure,
            // because the values might be different at actual call time.
            // We detect this by checking if the closure's env has variables marked as mutated.
            let free_vars = collect_free_vars(&body, &params);
            let closure_keys: Vec<_> = closure_env.borrow().keys().cloned().collect();
            let has_mutable_captures = free_vars
                .iter()
                .any(|k| is_var_mutated(k) && !is_internal_name(k) && closure_env.borrow().contains_key(k));
            eprintln!("DEBUG: Closure call - closure_keys: {:?}, has_mutable_captures: {}", closure_keys.iter().take(5).cloned().collect::<Vec<_>>(), has_mutable_captures);

            // Also check if the calling environment has variables that this closure needs
            // but which weren't captured (because they were out of scope at definition time)
            let needs_outer_vars = {
                let closure_borrowed = closure_env.borrow();
                free_vars.iter().any(|k| {
                    !closure_borrowed.contains_key(k)
                        && is_var_mutated(k)
                        && !is_internal_name(k)
                })
            };
            eprintln!("DEBUG: needs_outer_vars: {}", needs_outer_vars);

            if has_mutable_captures || needs_outer_vars {
                // Don't inline - the closure depends on mutable state
                // Residualize the call so it will be executed at actual call time
                eprintln!("DEBUG: NOT inlining closure call due to mutable captures/outer vars");
                return Some(PValue::Dynamic(Expr::Call(
                    Box::new(residualize(&PValue::Static(Value::Closure {
                        params,
                        body,
                        env: closure_env,
                    }))),
                    args_pv.iter().map(residualize).collect(),
                )));
            }

            // Inline the function: use closure's captured env (lexical scoping)
            let call_env = env_to_penv(&closure_env);

            // For variables that were captured as dynamic Var references (because they're mutated),
            // refresh their values from the current outer env. This allows the closure to see
            // the current static values of mutated outer variables.
            for (k, v) in call_env.borrow().clone().iter() {
                if matches!(v, PValue::Dynamic(Expr::Var(ref var_name)) if var_name == k) {
                    // This variable was captured as a dynamic reference - try to get current value
                    if let Some(current_val) = env.borrow().get(k) {
                        // Use the current value from outer env
                        call_env.borrow_mut().insert(k.clone(), current_val.clone());
                    }
                }
            }

            // Also add any variables from the calling env that aren't in the closure but are referenced
            // This handles the case where variables weren't in scope at definition time but are now
            for (k, v) in env.borrow().iter() {
                if !call_env.borrow().contains_key(k) {
                    call_env.borrow_mut().insert(k.clone(), v.clone());
                }
            }

            // Bind parameters to arguments
            for (param, arg_pv) in params.iter().zip(args_pv.iter()) {
                call_env.borrow_mut().insert(param.clone(), arg_pv.clone());
            }

            let body_slot = results.len();
            results.push(None);

            // Push evaluation of body, then unwrap any return
            work_stack.push(PEWorkItem::ContCallBody {
                body_slot,
                result_slot,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: body,
                env: call_env,
                result_slot: body_slot,
            });
            None // Indicates we pushed more work
        }

        PValue::Dynamic(Expr::Fn(params, body)) |
        PValue::DynamicNamed { expr: Expr::Fn(params, body), .. } => {
            eprintln!("DEBUG: Matched Dynamic/DynamicNamed Fn branch, params: {:?}, body starts with: {:?}",
                params,
                body.to_string().chars().take(60).collect::<String>());
            eprintln!("DEBUG: env has v9={}, v11={}, v21={}",
                env.borrow().contains_key("v9"),
                env.borrow().contains_key("v11"),
                env.borrow().contains_key("v21"));
            if params.len() != args_pv.len() {
                eprintln!("DEBUG: Param count mismatch");
                return Some(PValue::Dynamic(Expr::Call(
                    Box::new(Expr::Fn(params, body)),
                    args_pv.iter().map(residualize).collect(),
                )));
            }

            // At call time, we can inline even if free vars appear dynamic in the current
            // env, because the current env might be a PE snapshot that doesn't reflect
            // the actual runtime values. We'll just try to inline and if values aren't
            // available, the result will naturally be dynamic.

            // Save any existing bindings that would be shadowed by parameters
            let saved_bindings: Vec<_> = params.iter()
                .filter_map(|p| env.borrow().get(p).cloned().map(|v| (p.clone(), v)))
                .collect();

            // Bind parameters to arguments in the original env
            for (param, arg_pv) in params.iter().zip(args_pv.iter()) {
                env.borrow_mut().insert(param.clone(), arg_pv.clone());
            }

            // We need to evaluate the body and then restore the env
            // For now, evaluate synchronously (this is a limitation)
            let result = unwrap_return_pv(partial_eval_sync(&body, env));

            // Restore any shadowed bindings
            for (param, saved) in &saved_bindings {
                env.borrow_mut().insert(param.clone(), saved.clone());
            }

            // Remove parameters that weren't shadowing anything
            for param in &params {
                if !saved_bindings.iter().any(|(p, _)| p == param) {
                    env.borrow_mut().remove(param);
                }
            }

            Some(result)
        }

        _ => {
            // Truly dynamic function - emit residual call
            Some(PValue::Dynamic(Expr::Call(
                Box::new(residualize(&func_pv)),
                args_pv.iter().map(residualize).collect(),
            )))
        }
    }
}

/// Evaluate switch after discriminant is known
fn eval_switch(
    disc_pv: PValue,
    cases: Vec<(Expr, Vec<Expr>)>,
    default: Option<Vec<Expr>>,
    env: PEnv,
    result_slot: usize,
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
) {
    if let PValue::Return(inner) = &disc_pv {
        results[result_slot] = Some(PValue::Return(Box::new(inner.as_ref().clone())));
        return;
    }
    match &disc_pv {
        PValue::Static(disc_val) => {
            // Debug: log which static case value we're dispatching on
            if let Value::Int(n) = disc_val {
                let case_num = n & 31;
                eprintln!("DEBUG: Static switch dispatch case {} (disc_val={})", case_num, n);
            }
            // Find matching case - evaluate case values iteratively using work stack
            // We need to find the first case that matches, then execute its body
            work_stack.push(PEWorkItem::ContSwitchFindCase {
                disc_val: disc_val.clone(),
                cases,
                default,
                env,
                result_slot,
                case_idx: 0,
            });
        }
        _ => {
            eprintln!("DEBUG: Dynamic switch discriminant");
            //
            // Dynamic discriminant - emit residual switch WITHOUT deep evaluation
            // Mark ALL variables modified in ANY case as dynamic (this is already iterative)
            for (_, body) in &cases {
                for stmt in body {
                    mark_modified_vars_dynamic(stmt, &env);
                }
            }
            if let Some(def) = &default {
                for stmt in def {
                    mark_modified_vars_dynamic(stmt, &env);
                }
            }

            // Filter out empty cases (just break, or empty body) without evaluating
            let filtered_cases: Vec<(Expr, Vec<Expr>)> = cases
                .into_iter()
                .filter(|(_, body)| !is_empty_switch_case(body))
                .collect();

            // For dynamic discriminant, just keep the original expressions
            // Don't try to partially evaluate case bodies - this avoids deep recursion
            results[result_slot] = Some(PValue::Dynamic(Expr::Switch {
                discriminant: Box::new(residualize(&disc_pv)),
                cases: filtered_cases,
                default,
            }));
        }
    }
}

fn execute_switch_case_body(body: &[Expr], env: &PEnv) -> PValue {
    if body.is_empty() {
        return PValue::Static(Value::Undefined);
    }
    let mut residuals: Vec<Expr> = Vec::new();
    let mut last_static: Option<Value> = None;

    for stmt in body {
        let pv = partial_eval_sync(stmt, env);
        match &pv {
            PValue::Return(inner) => {
                if residuals.is_empty() {
                    return PValue::Return(Box::new(inner.as_ref().clone()));
                }
                residuals.push(Expr::Return(Box::new(residualize(inner))));
                return if residuals.len() == 1 {
                    PValue::Dynamic(residuals.pop().unwrap())
                } else {
                    PValue::Dynamic(Expr::Begin(residuals))
                };
            }
            PValue::Dynamic(e) | PValue::DynamicNamed { expr: e, .. } => {
                last_static = None;
                let is_break = matches!(e, Expr::Break | Expr::Continue);
                residuals.push(residualize(&pv));
                if is_break {
                    break;
                }
            }
            PValue::Static(v) => {
                last_static = Some(v.clone());
            }
            PValue::StaticNamed { .. } => {
                last_static = None;
            }
        }
    }

    if residuals.is_empty() {
        PValue::Static(last_static.unwrap_or(Value::Undefined))
    } else if residuals.len() == 1 {
        PValue::Dynamic(residuals.pop().unwrap())
    } else {
        PValue::Dynamic(Expr::Begin(residuals))
    }
}

/// While loop continuation after condition evaluation
fn eval_while_after_cond(
    cond_pv: PValue,
    cond: Expr,
    body: Expr,
    env: PEnv,
    result_slot: usize,
    initial_env_snapshot: HashMap<String, PValue>,
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
) {
    if matches!(cond_pv, PValue::Return(_)) {
        results[result_slot] = Some(cond_pv);
        return;
    }
    // Check if the condition is statically known to be truthy or falsy
    let static_truthiness = is_js_truthy(&cond_pv);

    match static_truthiness {
        Some(false) => {
            // Condition is statically falsy - loop never executes
            results[result_slot] = Some(PValue::Static(Value::Undefined));
        }
        Some(true) => {
            // Condition is statically truthy - start unrolling
            // We unroll until the condition becomes dynamic, not just because body has free vars
            let pre_iteration_snapshot: HashMap<String, PValue> =
                env.borrow().iter().map(|(k, v)| (k.clone(), v.clone())).collect();

            let body_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContWhileBody {
                cond: cond.clone(),
                body: body.clone(),
                env: env.clone(),
                body_slot,
                result_slot,
                initial_env_snapshot,
                pre_iteration_snapshot,
                iterations: 1,
                unrolled_bodies: Vec::new(),
            });
            work_stack.push(PEWorkItem::Eval {
                expr: body,
                env,
                result_slot: body_slot,
            });
        }
        None => {
            // Dynamic condition - emit residual
            mark_modified_vars_dynamic(&body, &env);
            let body_pv = partial_eval_sync(&body, &env);
            results[result_slot] = Some(PValue::Dynamic(Expr::While(
                Box::new(residualize(&cond_pv)),
                Box::new(residualize(&body_pv)),
            )));
        }
    }
}

/// While loop continuation after body evaluation
fn eval_while_after_body(
    body_pv: PValue,
    cond: Expr,
    body: Expr,
    env: PEnv,
    result_slot: usize,
    initial_env_snapshot: HashMap<String, PValue>,
    _pre_iteration_snapshot: HashMap<String, PValue>,
    iterations: usize,
    mut unrolled_bodies: Vec<Expr>,
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
) {
    const MAX_ITERATIONS: usize = 10000;

    // Helper to get the actual expression value
    let get_actual_expr = |pv: &PValue| -> Expr {
        match pv {
            PValue::Static(v) => value_to_expr(v),
            PValue::StaticNamed { value, .. } => value_to_expr(value),
            PValue::Dynamic(e) => e.clone(),
            PValue::DynamicNamed { expr, .. } => expr.clone(),
            PValue::Return(inner) => Expr::Return(Box::new(residualize(inner))),
        }
    };

    let emit_state_updates_between = |initial: &HashMap<String, PValue>,
                                       current: &HashMap<String, PValue>| -> Vec<Expr> {
        let mut updates = Vec::new();
        for (name, current_val) in current.iter() {
            if let Some(initial_val) = initial.get(name) {
                let current_expr = get_actual_expr(current_val);
                let initial_expr = get_actual_expr(initial_val);
                if current_expr != initial_expr {
                    updates.push(Expr::Set(name.clone(), Box::new(current_expr)));
                }
            }
        }
        updates
    };

    if let PValue::Return(inner) = &body_pv {
        if unrolled_bodies.is_empty() {
            results[result_slot] = Some(PValue::Return(Box::new(inner.as_ref().clone())));
        } else {
            unrolled_bodies.push(Expr::Return(Box::new(residualize(inner))));
            if unrolled_bodies.len() == 1 {
                results[result_slot] = Some(PValue::Dynamic(unrolled_bodies.into_iter().next().unwrap()));
            } else {
                results[result_slot] = Some(PValue::Dynamic(Expr::Begin(unrolled_bodies)));
            }
        }
        return;
    }

    // Accumulate the body into unrolled_bodies if it's dynamic
    let body_residual = residualize(&body_pv);
    if ends_with_return(&body_residual) || ends_with_throw(&body_residual) {
        results[result_slot] = Some(PValue::Dynamic(body_residual));
        return;
    }
    if matches!(body_pv, PValue::Dynamic(_) | PValue::DynamicNamed { .. }) {
        // Check for break - if body always breaks, we're done with this iteration
        if body_always_breaks(&body_residual) {
            unrolled_bodies.push(strip_trailing_break(body_residual));
            // Return accumulated bodies
            if unrolled_bodies.len() == 1 {
                results[result_slot] = Some(PValue::Dynamic(unrolled_bodies.into_iter().next().unwrap()));
            } else {
                results[result_slot] = Some(PValue::Dynamic(Expr::Begin(unrolled_bodies)));
            }
            return;
        }
        unrolled_bodies.push(body_residual);
    }

    let gas_enabled = GAS.with(|g| g.borrow().is_some());
    if !gas_enabled && iterations > MAX_ITERATIONS {
        // Hit iteration limit - emit residual while with accumulated bodies
        mark_modified_vars_dynamic(&body, &env);
        let body_pv = partial_eval_sync(&body, &env);
        let while_expr = Expr::While(
            Box::new(Expr::Bool(true)),
            Box::new(residualize(&body_pv)),
        );
        unrolled_bodies.push(while_expr);
        if unrolled_bodies.len() == 1 {
            results[result_slot] = Some(PValue::Dynamic(unrolled_bodies.into_iter().next().unwrap()));
        } else {
            results[result_slot] = Some(PValue::Dynamic(Expr::Begin(unrolled_bodies)));
        }
        return;
    }

    // Re-evaluate condition after body execution
    let new_cond_pv = partial_eval_sync(&cond, &env);
    if matches!(new_cond_pv, PValue::Return(_)) {
        results[result_slot] = Some(new_cond_pv);
        return;
    }
    let static_truthiness = is_js_truthy(&new_cond_pv);

    match static_truthiness {
        Some(false) => {
            // Loop terminates - return accumulated bodies or undefined
            if unrolled_bodies.is_empty() {
                results[result_slot] = Some(PValue::Static(Value::Undefined));
            } else if unrolled_bodies.len() == 1 {
                results[result_slot] = Some(PValue::Dynamic(unrolled_bodies.into_iter().next().unwrap()));
            } else {
                results[result_slot] = Some(PValue::Dynamic(Expr::Begin(unrolled_bodies)));
            }
        }
        Some(true) => {
            // Continue looping - evaluate body again
            let new_pre_iteration_snapshot: HashMap<String, PValue> =
                env.borrow().iter().map(|(k, v)| (k.clone(), v.clone())).collect();

            let body_slot = results.len();
            results.push(None);

            work_stack.push(PEWorkItem::ContWhileBody {
                cond: cond.clone(),
                body: body.clone(),
                env: env.clone(),
                body_slot,
                result_slot,
                initial_env_snapshot,
                pre_iteration_snapshot: new_pre_iteration_snapshot,
                iterations: iterations + 1,
                unrolled_bodies,
            });
            work_stack.push(PEWorkItem::Eval {
                expr: body,
                env,
                result_slot: body_slot,
            });
        }
        None => {
            // Condition became dynamic - emit residual while with accumulated bodies
            mark_modified_vars_dynamic(&body, &env);
            let body_pv = partial_eval_sync(&body, &env);
            let while_expr = Expr::While(
                Box::new(residualize(&new_cond_pv)),
                Box::new(residualize(&body_pv)),
            );

            unrolled_bodies.push(while_expr);
            if unrolled_bodies.len() == 1 {
                results[result_slot] = Some(PValue::Dynamic(unrolled_bodies.into_iter().next().unwrap()));
            } else {
                results[result_slot] = Some(PValue::Dynamic(Expr::Begin(unrolled_bodies)));
            }
        }
    }
}

/// For loop - after init evaluation
fn eval_for_after_init(
    cond: Option<Expr>,
    update: Option<Expr>,
    body: Expr,
    env: PEnv,
    result_slot: usize,
    init_expr: Option<Expr>,
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
) {
    // Evaluate condition
    let cond_pv = if let Some(ref c) = cond {
        partial_eval_sync(c, &env)
    } else {
        PValue::Static(Value::Bool(true))
    };

    let cond_slot = results.len();
    results.push(Some(cond_pv));

    eval_for_after_cond_impl(
        results[cond_slot].clone().unwrap(),
        cond,
        update,
        body,
        env,
        result_slot,
        0,
        init_expr,
        results,
        work_stack,
    );
}

fn eval_for_after_cond(
    cond_pv: PValue,
    cond: Option<Expr>,
    update: Option<Expr>,
    body: Expr,
    env: PEnv,
    result_slot: usize,
    iterations: usize,
    init_expr: Option<Expr>,
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
) {
    eval_for_after_cond_impl(cond_pv, cond, update, body, env, result_slot, iterations, init_expr, results, work_stack);
}

fn eval_for_after_cond_impl(
    cond_pv: PValue,
    cond: Option<Expr>,
    update: Option<Expr>,
    body: Expr,
    env: PEnv,
    result_slot: usize,
    iterations: usize,
    init_expr: Option<Expr>,
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
) {
    const MAX_ITERATIONS: usize = 10000;
    let gas_enabled = GAS.with(|g| g.borrow().is_some());

    if let PValue::Return(inner) = &cond_pv {
        results[result_slot] = Some(PValue::Return(Box::new(inner.as_ref().clone())));
        return;
    }

    let emit_residual_for = |results: &mut Vec<Option<PValue>>, result_slot: usize| {
        let init_pe = init_expr.as_ref().map(|e| Box::new(e.clone()));
        let cond_pe = cond.as_ref().map(|c| Box::new(c.clone()));
        let update_pe = update.as_ref().map(|u| Box::new(u.clone()));
        let body_pe = Box::new(body.clone());
        results[result_slot] = Some(PValue::Dynamic(Expr::For {
            init: init_pe,
            cond: cond_pe,
            update: update_pe,
            body: body_pe,
        }));
    };

    match &cond_pv {
        PValue::Static(Value::Bool(false)) => {
            results[result_slot] = Some(PValue::Static(Value::Undefined));
        }
        PValue::Static(Value::Bool(true)) => {
            if !gas_enabled && iterations > MAX_ITERATIONS {
                emit_residual_for(results, result_slot);
                return;
            }

            // Evaluate body
            let body_pv = partial_eval_sync(&body, &env);
            if let PValue::Return(inner) = &body_pv {
                results[result_slot] = Some(PValue::Return(Box::new(inner.as_ref().clone())));
                return;
            }
            let body_residual = residualize(&body_pv);
            if ends_with_return(&body_residual) || ends_with_throw(&body_residual) {
                results[result_slot] = Some(PValue::Dynamic(body_residual));
                return;
            }
            if matches!(body_pv, PValue::Dynamic(_)) {
                emit_residual_for(results, result_slot);
                return;
            }

            // Evaluate update if present
            if let Some(ref upd) = update {
                let upd_pv = partial_eval_sync(upd, &env);
                if let PValue::Return(inner) = &upd_pv {
                    results[result_slot] = Some(PValue::Return(Box::new(inner.as_ref().clone())));
                    return;
                }
                let upd_residual = residualize(&upd_pv);
                if ends_with_return(&upd_residual) || ends_with_throw(&upd_residual) {
                    results[result_slot] = Some(PValue::Dynamic(upd_residual));
                    return;
                }
                if matches!(upd_pv, PValue::Dynamic(_)) {
                    emit_residual_for(results, result_slot);
                    return;
                }
            }

            // Re-evaluate condition
            let new_cond = if let Some(ref c) = cond {
                partial_eval_sync(c, &env)
            } else {
                PValue::Static(Value::Bool(true))
            };

            match new_cond {
                PValue::Return(inner) => {
                    results[result_slot] = Some(PValue::Return(inner));
                }
                PValue::Static(Value::Bool(false)) => {
                    results[result_slot] = Some(PValue::Static(Value::Undefined));
                }
                PValue::Static(Value::Bool(true)) => {
                    // Continue looping - use recursive call for simplicity
                    eval_for_after_cond_impl(
                        new_cond,
                        cond,
                        update,
                        body,
                        env,
                        result_slot,
                        iterations + 1,
                        init_expr,
                        results,
                        work_stack,
                    );
                }
                _ => {
                    emit_residual_for(results, result_slot);
                }
            }
        }
        _ => {
            emit_residual_for(results, result_slot);
        }
    }
}

fn eval_for_after_body(
    body_pv: PValue,
    cond: Option<Expr>,
    update: Option<Expr>,
    body: Expr,
    env: PEnv,
    result_slot: usize,
    iterations: usize,
    init_expr: Option<Expr>,
    results: &mut Vec<Option<PValue>>,
    work_stack: &mut Vec<PEWorkItem>,
) {
    if let PValue::Return(inner) = body_pv {
        results[result_slot] = Some(PValue::Return(inner));
        return;
    }
    // For now, redirect to the after_cond_impl which handles the loop
    let new_cond = if let Some(ref c) = cond {
        partial_eval_sync(c, &env)
    } else {
        PValue::Static(Value::Bool(true))
    };

    eval_for_after_cond_impl(
        new_cond,
        cond,
        update,
        body,
        env,
        result_slot,
        iterations,
        init_expr,
        results,
        work_stack,
    );
}

/// Try-catch continuation
fn eval_try_catch(
    try_pv: PValue,
    catch_param: Option<String>,
    catch_block: Expr,
    finally_block: Option<Expr>,
    env: PEnv,
    result_slot: usize,
    results: &mut Vec<Option<PValue>>,
    _work_stack: &mut Vec<PEWorkItem>,
) {
    let finally_pv = finally_block.as_ref().map(|fb| partial_eval_sync(fb, &env));

    if let PValue::Return(inner) = &try_pv {
        if let Some(fpv) = finally_pv {
            let finally_residual = residualize(&fpv);
            if ends_with_return(&finally_residual) || ends_with_throw(&finally_residual) {
                results[result_slot] = Some(PValue::Dynamic(finally_residual));
            } else if is_pure_expr(&finally_residual) {
                results[result_slot] = Some(PValue::Return(Box::new(inner.as_ref().clone())));
            } else {
                results[result_slot] = Some(PValue::Dynamic(Expr::Begin(vec![
                    finally_residual,
                    Expr::Return(Box::new(residualize(inner))),
                ])));
            }
        } else {
            results[result_slot] = Some(PValue::Return(Box::new(inner.as_ref().clone())));
        }
        return;
    }

    let try_residual = residualize(&try_pv);
    if let Some(thrown_expr) = extract_throw_value(&try_residual) {
        // Evaluate catch in the current env to allow internal state updates.
        let mut saved_param: Option<(String, PValue)> = None;
        if let Some(ref param) = catch_param {
            if let Some(prev) = env.borrow().get(param).cloned() {
                saved_param = Some((param.clone(), prev));
            }
            let thrown_pv = partial_eval_sync(&thrown_expr, &env);
            env.borrow_mut().insert(param.clone(), thrown_pv);
        }

        let catch_pv = partial_eval_sync(&catch_block, &env);

        // Restore catch param binding if it existed.
        if let Some(ref param) = catch_param {
            if let Some((_, prev)) = saved_param {
                env.borrow_mut().insert(param.clone(), prev);
            } else {
                env.borrow_mut().remove(param);
            }
        }

        results[result_slot] = Some(catch_pv);
        return;
    }

    // If the try block can't throw (no explicit throw), ignore catch.
    if !may_throw_expr(&try_residual) {
        if let Some(fpv) = finally_pv {
            results[result_slot] = Some(PValue::Dynamic(Expr::Begin(vec![
                try_residual,
                residualize(&fpv),
            ])));
        } else {
            results[result_slot] = Some(try_pv);
        }
        return;
    }

    // If the try block is fully static, we can ignore the catch (no throw observed).
    if matches!(try_pv, PValue::Static(_) | PValue::StaticNamed { .. }) {
        if let Some(fpv) = finally_pv {
            results[result_slot] = Some(PValue::Dynamic(Expr::Begin(vec![
                try_residual,
                residualize(&fpv),
            ])));
        } else {
            results[result_slot] = Some(try_pv);
        }
        return;
    }

    // Evaluate catch in an isolated env so it doesn't mutate the outer env.
    let catch_env: PEnv = Rc::new(RefCell::new(env.borrow().clone()));
    if let Some(ref param) = catch_param {
        catch_env.borrow_mut().insert(param.clone(), PValue::Dynamic(Expr::Var(param.clone())));
    }
    let catch_pv = partial_eval_sync(&catch_block, &catch_env);

    results[result_slot] = Some(PValue::Dynamic(Expr::TryCatch {
        try_block: Box::new(residualize(&try_pv)),
        catch_param,
        catch_block: Box::new(residualize(&catch_pv)),
        finally_block: finally_pv.map(|pv| Box::new(residualize(&pv))),
    }));
}

/// Partial evaluation of binary operations
fn partial_eval_binop(op: &BinOp, left: PValue, right: PValue) -> PValue {
    if let Some(cf) = control_flow_pv(&left) {
        return cf;
    }
    if let Some(cf) = control_flow_pv(&right) {
        return cf;
    }
    // First, try to fold if both are static
    if let (PValue::Static(lv), PValue::Static(rv)) = (&left, &right) {
        if let Some(result) = eval_static_binop(op, lv, rv) {
            return PValue::Static(result);
        }
    }

    // Extract static int values for identity optimizations
    let left_int = match &left {
        PValue::Static(Value::Int(n)) => Some(*n),
        _ => None,
    };
    let right_int = match &right {
        PValue::Static(Value::Int(n)) => Some(*n),
        _ => None,
    };

    // Apply algebraic identity optimizations
    match op {
        // Addition: x + 0 = x, 0 + x = x
        BinOp::Add => {
            if right_int == Some(0) {
                return left;
            }
            if left_int == Some(0) {
                return right;
            }
        }
        // Subtraction: x - 0 = x
        BinOp::Sub => {
            if right_int == Some(0) {
                return left;
            }
        }
        // Multiplication: x * 0 = 0, 0 * x = 0, x * 1 = x, 1 * x = x
        BinOp::Mul => {
            if right_int == Some(0) || left_int == Some(0) {
                return PValue::Static(Value::Int(0));
            }
            if right_int == Some(1) {
                return left;
            }
            if left_int == Some(1) {
                return right;
            }
        }
        // Division: x / 1 = x
        BinOp::Div => {
            if right_int == Some(1) {
                return left;
            }
        }
        // Bitwise AND: x & 0 = 0, 0 & x = 0
        BinOp::BitAnd => {
            if right_int == Some(0) || left_int == Some(0) {
                return PValue::Static(Value::Int(0));
            }
            // Idempotent mask optimization: (& (& x m1) m2) where (m1 & m2) == m1 → (& x m1)
            // This works when the inner mask is already narrower than the outer mask
            if let Some(m2) = right_int {
                // Check if left is a DynamicNamed whose expr is (& _ m1)
                if let PValue::DynamicNamed { expr, .. } = &left {
                    if let Expr::BinOp(BinOp::BitAnd, _, inner_right) = expr {
                        if let Expr::Int(m1) = inner_right.as_ref() {
                            // If (m1 & m2) == m1, the outer mask doesn't narrow further
                            if (m1 & m2) == *m1 {
                                return left;
                            }
                        }
                    }
                }
            }
        }
        // Bitwise OR: x | 0 = x, 0 | x = x
        BinOp::BitOr => {
            if right_int == Some(0) {
                return left;
            }
            if left_int == Some(0) {
                return right;
            }
        }
        // Bitwise XOR: x ^ 0 = x, 0 ^ x = x
        BinOp::BitXor => {
            if right_int == Some(0) {
                return left;
            }
            if left_int == Some(0) {
                return right;
            }
        }
        // Shifts: x << 0 = x, x >> 0 = x, x >>> 0 = x
        BinOp::Shl | BinOp::Shr | BinOp::UShr => {
            if right_int == Some(0) {
                return left;
            }
        }
        // Logical AND short-circuit: (&&  true x) = x, (&& false x) = false
        BinOp::And => {
            if let PValue::Static(Value::Bool(b)) = &left {
                if *b {
                    return right; // true && x = x
                } else {
                    return PValue::Static(Value::Bool(false)); // false && x = false
                }
            }
            if let PValue::Static(Value::Bool(b)) = &right {
                if !*b {
                    return PValue::Static(Value::Bool(false)); // x && false = false
                }
                // x && true = x (but we keep it as is since x might not be a bool)
            }
        }
        // Logical OR short-circuit: (|| true x) = true, (|| false x) = x
        BinOp::Or => {
            if let PValue::Static(Value::Bool(b)) = &left {
                if *b {
                    return PValue::Static(Value::Bool(true)); // true || x = true
                } else {
                    return right; // false || x = x
                }
            }
            if let PValue::Static(Value::Bool(b)) = &right {
                if *b {
                    return PValue::Static(Value::Bool(true)); // x || true = true
                }
                // x || false = x (but we keep it as is since x might not be a bool)
            }
        }
        _ => {}
    }

    // Self-comparison optimization: x op x → known result for comparison operators
    // Only apply when both sides residualize to the same pure expression
    let left_residual = residualize(&left);
    let right_residual = residualize(&right);
    if is_pure_expr(&left_residual) && left_residual == right_residual {
        match op {
            BinOp::Eq | BinOp::Lte | BinOp::Gte => {
                return PValue::Static(Value::Bool(true)); // x == x, x <= x, x >= x
            }
            BinOp::NotEq | BinOp::Lt | BinOp::Gt => {
                return PValue::Static(Value::Bool(false)); // x != x, x < x, x > x
            }
            _ => {}
        }
    }

    // Reassociation: fold nested arithmetic with static constants
    // e.g., (+ (+ x 1) 2) → (+ x 3), (- (- x 1) 2) → (- x 3)
    if let Some(b) = right_int {
        if let Expr::BinOp(inner_op, inner_left, inner_right) = &left_residual {
            if let Expr::Int(a) = inner_right.as_ref() {
                // We have: op(inner_op(x, a), b) where a and b are static ints
                match (op, inner_op) {
                    // (+ (+ x a) b) → (+ x (a + b))
                    (BinOp::Add, BinOp::Add) => {
                        return PValue::Dynamic(Expr::BinOp(
                            BinOp::Add,
                            inner_left.clone(),
                            Box::new(Expr::Int(a.wrapping_add(b))),
                        ));
                    }
                    // (+ (- x a) b) → (+ x (b - a)) or (- x (a - b))
                    (BinOp::Add, BinOp::Sub) => {
                        let diff = b.wrapping_sub(*a);
                        if diff >= 0 {
                            return PValue::Dynamic(Expr::BinOp(
                                BinOp::Add,
                                inner_left.clone(),
                                Box::new(Expr::Int(diff)),
                            ));
                        } else {
                            return PValue::Dynamic(Expr::BinOp(
                                BinOp::Sub,
                                inner_left.clone(),
                                Box::new(Expr::Int(-diff)),
                            ));
                        }
                    }
                    // (- (+ x a) b) → (+ x (a - b)) or (- x (b - a))
                    (BinOp::Sub, BinOp::Add) => {
                        let diff = a.wrapping_sub(b);
                        if diff >= 0 {
                            return PValue::Dynamic(Expr::BinOp(
                                BinOp::Add,
                                inner_left.clone(),
                                Box::new(Expr::Int(diff)),
                            ));
                        } else {
                            return PValue::Dynamic(Expr::BinOp(
                                BinOp::Sub,
                                inner_left.clone(),
                                Box::new(Expr::Int(-diff)),
                            ));
                        }
                    }
                    // (- (- x a) b) → (- x (a + b))
                    (BinOp::Sub, BinOp::Sub) => {
                        return PValue::Dynamic(Expr::BinOp(
                            BinOp::Sub,
                            inner_left.clone(),
                            Box::new(Expr::Int(a.wrapping_add(b))),
                        ));
                    }
                    // (& (& x a) b) → (& x (a & b))
                    (BinOp::BitAnd, BinOp::BitAnd) => {
                        return PValue::Dynamic(Expr::BinOp(
                            BinOp::BitAnd,
                            inner_left.clone(),
                            Box::new(Expr::Int(a & b)),
                        ));
                    }
                    // (| (| x a) b) → (| x (a | b))
                    (BinOp::BitOr, BinOp::BitOr) => {
                        return PValue::Dynamic(Expr::BinOp(
                            BinOp::BitOr,
                            inner_left.clone(),
                            Box::new(Expr::Int(a | b)),
                        ));
                    }
                    // (^ (^ x a) b) → (^ x (a ^ b))
                    (BinOp::BitXor, BinOp::BitXor) => {
                        return PValue::Dynamic(Expr::BinOp(
                            BinOp::BitXor,
                            inner_left.clone(),
                            Box::new(Expr::Int(a ^ b)),
                        ));
                    }
                    _ => {}
                }
            }
        }
    }

    // Default: emit residual
    PValue::Dynamic(Expr::BinOp(
        op.clone(),
        Box::new(left_residual),
        Box::new(right_residual),
    ))
}

/// Mark variables that are modified (via set!) in an expression as dynamic
/// This is used when we can't unroll a loop - the modified vars become unknown
/// Iterative implementation to avoid stack overflow.
fn mark_modified_vars_dynamic(expr: &Expr, env: &PEnv) {
    let mut stack: Vec<&Expr> = vec![expr];

    while let Some(e) = stack.pop() {
        match e {
            Expr::Set(name, _) => {
                if env.borrow().contains_key(name) {
                    if name == "v9" || name == "v11" || name == "v21" {
                        eprintln!("DEBUG: mark_modified_vars_dynamic marking {} as Dynamic!", name);
                    }
                    env.borrow_mut().insert(name.clone(), PValue::Dynamic(Expr::Var(name.clone())));
                }
            }

            Expr::Begin(exprs) => {
                for expr in exprs.iter().rev() {
                    stack.push(expr);
                }
            }

            Expr::If(_, then_br, else_br) => {
                stack.push(else_br);
                stack.push(then_br);
            }

            Expr::While(_, body) => {
                stack.push(body);
            }

            Expr::For { body, update, .. } => {
                if let Some(upd) = update {
                    stack.push(upd);
                }
                stack.push(body);
            }

            Expr::Switch { cases, default, .. } => {
                if let Some(def) = default {
                    for stmt in def.iter().rev() {
                        stack.push(stmt);
                    }
                }
                for (_, body) in cases.iter().rev() {
                    for stmt in body.iter().rev() {
                        stack.push(stmt);
                    }
                }
            }

            Expr::TryCatch { try_block, catch_block, finally_block, .. } => {
                if let Some(fb) = finally_block {
                    stack.push(fb);
                }
                stack.push(catch_block);
                stack.push(try_block);
            }
            Expr::Return(inner) => {
                stack.push(inner);
            }

            _ => {}
        }
    }
}

/// Filter out pure expressions from a statement list, keeping only:
/// - Expressions with side effects
/// - The last expression (the return value)
fn filter_dead_code(exprs: Vec<Expr>) -> Vec<Expr> {
    if exprs.is_empty() {
        return exprs;
    }

    let mut result = Vec::new();
    let last_idx = exprs.len() - 1;

    for (i, expr) in exprs.into_iter().enumerate() {
        // Keep the last expression (return value) and any impure expressions
        if i == last_idx || !is_pure_expr(&expr) {
            result.push(expr);
        }
    }

    result
}

/// Check if an expression may throw (explicit throw only)
fn may_throw_expr(expr: &Expr) -> bool {
    let mut stack: Vec<&Expr> = vec![expr];

    while let Some(e) = stack.pop() {
        match e {
            Expr::Throw(_) => return true,

            Expr::BinOp(_, l, r) => {
                stack.push(r);
                stack.push(l);
            }
            Expr::If(cond, then_br, else_br) => {
                stack.push(else_br);
                stack.push(then_br);
                stack.push(cond);
            }
            Expr::Let(_, value, body) => {
                stack.push(body);
                stack.push(value);
            }
            Expr::Fn(_, body) => {
                stack.push(body);
            }
            Expr::Call(func, args) => {
                for arg in args.iter().rev() {
                    stack.push(arg);
                }
                stack.push(func);
            }
            Expr::Begin(exprs) => {
                for expr in exprs.iter().rev() {
                    stack.push(expr);
                }
            }
            Expr::While(cond, body) => {
                stack.push(body);
                stack.push(cond);
            }
            Expr::For { init, cond, update, body } => {
                stack.push(body);
                if let Some(u) = update {
                    stack.push(u);
                }
                if let Some(c) = cond {
                    stack.push(c);
                }
                if let Some(i) = init {
                    stack.push(i);
                }
            }
            Expr::Set(_, value) => {
                stack.push(value);
            }
            Expr::Array(elems) => {
                for elem in elems.iter().rev() {
                    stack.push(elem);
                }
            }
            Expr::Index(arr, idx) => {
                stack.push(idx);
                stack.push(arr);
            }
            Expr::Len(arr) => {
                stack.push(arr);
            }
            Expr::Object(props) => {
                for (_, v) in props.iter().rev() {
                    stack.push(v);
                }
            }
            Expr::PropAccess(obj, _) => {
                stack.push(obj);
            }
            Expr::PropSet(obj, _, value) => {
                stack.push(value);
                stack.push(obj);
            }
            Expr::ComputedAccess(obj, key) => {
                stack.push(key);
                stack.push(obj);
            }
            Expr::ComputedSet(obj, key, value) => {
                stack.push(value);
                stack.push(key);
                stack.push(obj);
            }
            Expr::BitNot(inner) | Expr::LogNot(inner) | Expr::Return(inner) => {
                stack.push(inner);
            }
            Expr::New(ctor, args) => {
                for a in args.iter().rev() {
                    stack.push(a);
                }
                stack.push(ctor);
            }
            Expr::Switch { discriminant, cases, default } => {
                if let Some(d) = default {
                    for stmt in d.iter().rev() {
                        stack.push(stmt);
                    }
                }
                for (cv, body) in cases.iter().rev() {
                    for stmt in body.iter().rev() {
                        stack.push(stmt);
                    }
                    stack.push(cv);
                }
                stack.push(discriminant);
            }
            Expr::TryCatch { try_block, catch_block, finally_block, .. } => {
                if let Some(fb) = finally_block {
                    stack.push(fb);
                }
                stack.push(catch_block);
                stack.push(try_block);
            }

            Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Var(_)
            | Expr::Undefined | Expr::Null | Expr::Break | Expr::Continue
            | Expr::Opaque(_) => {}
        }
    }

    false
}

fn extract_throw_value(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Throw(inner) => Some((**inner).clone()),
        Expr::Begin(exprs) => exprs.last().and_then(extract_throw_value),
        _ => None,
    }
}

/// Count nodes in an expression, stopping once the limit is exceeded
fn expr_size_with_limit(expr: &Expr, limit: usize) -> usize {
    let mut count: usize = 0;
    let mut stack: Vec<&Expr> = vec![expr];

    while let Some(e) = stack.pop() {
        count += 1;
        if count > limit {
            return count;
        }

        match e {
            Expr::BinOp(_, l, r) => {
                stack.push(r);
                stack.push(l);
            }
            Expr::If(cond, then_br, else_br) => {
                stack.push(else_br);
                stack.push(then_br);
                stack.push(cond);
            }
            Expr::Let(_, value, body) => {
                stack.push(body);
                stack.push(value);
            }
            Expr::Fn(_, body) => {
                stack.push(body);
            }
            Expr::Call(func, args) => {
                for arg in args.iter().rev() {
                    stack.push(arg);
                }
                stack.push(func);
            }
            Expr::Begin(exprs) => {
                for expr in exprs.iter().rev() {
                    stack.push(expr);
                }
            }
            Expr::While(cond, body) => {
                stack.push(body);
                stack.push(cond);
            }
            Expr::For { init, cond, update, body } => {
                stack.push(body);
                if let Some(u) = update {
                    stack.push(u);
                }
                if let Some(c) = cond {
                    stack.push(c);
                }
                if let Some(i) = init {
                    stack.push(i);
                }
            }
            Expr::Set(_, value) => {
                stack.push(value);
            }
            Expr::Array(elems) => {
                for elem in elems.iter().rev() {
                    stack.push(elem);
                }
            }
            Expr::Index(arr, idx) => {
                stack.push(idx);
                stack.push(arr);
            }
            Expr::Len(arr) => {
                stack.push(arr);
            }
            Expr::Object(props) => {
                for (_, v) in props.iter().rev() {
                    stack.push(v);
                }
            }
            Expr::PropAccess(obj, _) => {
                stack.push(obj);
            }
            Expr::PropSet(obj, _, value) => {
                stack.push(value);
                stack.push(obj);
            }
            Expr::ComputedAccess(obj, key) => {
                stack.push(key);
                stack.push(obj);
            }
            Expr::ComputedSet(obj, key, value) => {
                stack.push(value);
                stack.push(key);
                stack.push(obj);
            }
            Expr::BitNot(inner) | Expr::LogNot(inner) | Expr::Throw(inner) | Expr::Return(inner) => {
                stack.push(inner);
            }
            Expr::New(ctor, args) => {
                for a in args.iter().rev() {
                    stack.push(a);
                }
                stack.push(ctor);
            }
            Expr::Switch { discriminant, cases, default } => {
                if let Some(d) = default {
                    for stmt in d.iter().rev() {
                        stack.push(stmt);
                    }
                }
                for (cv, body) in cases.iter().rev() {
                    for stmt in body.iter().rev() {
                        stack.push(stmt);
                    }
                    stack.push(cv);
                }
                stack.push(discriminant);
            }
            Expr::TryCatch { try_block, catch_block, finally_block, .. } => {
                if let Some(fb) = finally_block {
                    stack.push(fb);
                }
                stack.push(catch_block);
                stack.push(try_block);
            }

            Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Var(_)
            | Expr::Undefined | Expr::Null | Expr::Break | Expr::Continue
            | Expr::Opaque(_) => {}
        }
    }

    count
}

/// Check if an expression ends with a return statement
fn ends_with_return(expr: &Expr) -> bool {
    match expr {
        Expr::Return(_) => true,
        Expr::Begin(exprs) => exprs.last().map_or(false, ends_with_return),
        Expr::Let(_, _, body) => ends_with_return(body),
        Expr::If(_, then_br, else_br) => ends_with_return(then_br) && ends_with_return(else_br),
        Expr::TryCatch { try_block, catch_block, .. } => {
            ends_with_return(try_block) && ends_with_return(catch_block)
        }
        _ => false,
    }
}

fn ends_with_throw(expr: &Expr) -> bool {
    match expr {
        Expr::Throw(_) => true,
        Expr::Begin(exprs) => exprs.last().map_or(false, ends_with_throw),
        Expr::Let(_, _, body) => ends_with_throw(body),
        Expr::If(_, then_br, else_br) => ends_with_throw(then_br) && ends_with_throw(else_br),
        Expr::TryCatch { try_block, catch_block, .. } => {
            ends_with_throw(try_block) && ends_with_throw(catch_block)
        }
        _ => false,
    }
}

/// Replace a trailing return with its inner value, preserving side effects
fn unwrap_return_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Return(inner) => *inner,
        Expr::Begin(mut exprs) => {
            if let Some(last) = exprs.pop() {
                let unwrapped = unwrap_return_expr(last);
                exprs.push(unwrapped);
                if exprs.len() == 1 {
                    exprs.pop().unwrap()
                } else {
                    Expr::Begin(exprs)
                }
            } else {
                Expr::Undefined
            }
        }
        Expr::Let(name, value, body) => {
            Expr::Let(name, value, Box::new(unwrap_return_expr(*body)))
        }
        Expr::If(cond, then_br, else_br) => {
            Expr::If(
                cond,
                Box::new(unwrap_return_expr(*then_br)),
                Box::new(unwrap_return_expr(*else_br)),
            )
        }
        Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
            Expr::TryCatch {
                try_block: Box::new(unwrap_return_expr(*try_block)),
                catch_param,
                catch_block: Box::new(unwrap_return_expr(*catch_block)),
                finally_block,
            }
        }
        other => other,
    }
}

/// If a PValue represents a return, unwrap it to the returned value
fn unwrap_return_pv(pv: PValue) -> PValue {
    match pv {
        PValue::Return(inner) => *inner,
        PValue::Dynamic(e) if ends_with_return(&e) => PValue::Dynamic(unwrap_return_expr(e)),
        PValue::DynamicNamed { expr, .. } if ends_with_return(&expr) => {
            PValue::Dynamic(unwrap_return_expr(expr))
        }
        other => other,
    }
}

fn control_flow_pv(pv: &PValue) -> Option<PValue> {
    match pv {
        PValue::Return(inner) => Some(PValue::Return(Box::new(inner.as_ref().clone()))),
        PValue::Dynamic(e) if ends_with_throw(e) => Some(PValue::Dynamic(e.clone())),
        PValue::DynamicNamed { expr, .. } if ends_with_throw(expr) => Some(PValue::Dynamic(expr.clone())),
        _ => None,
    }
}

/// Check if a switch case body is empty (contains only break or is empty)
/// Such cases can be removed since they have no effect
fn is_empty_switch_case(body: &[Expr]) -> bool {
    body.is_empty() || (body.len() == 1 && matches!(&body[0], Expr::Break))
}

/// Check if an expression always ends in a break statement
/// Used to optimize while loops with static true condition - if body always breaks,
/// the loop only executes once and we can eliminate the while wrapper
fn body_always_breaks(expr: &Expr) -> bool {
    match expr {
        Expr::Break => true,
        Expr::Begin(exprs) => {
            // A begin always breaks if its last expression always breaks
            exprs.last().map_or(false, body_always_breaks)
        }
        Expr::If(_, then_br, else_br) => {
            // An if always breaks if both branches always break
            body_always_breaks(then_br) && body_always_breaks(else_br)
        }
        _ => false,
    }
}

/// Strip trailing break from an expression (used when eliminating while loops)
fn strip_trailing_break(expr: Expr) -> Expr {
    match expr {
        Expr::Break => Expr::Undefined,
        Expr::Begin(mut exprs) => {
            if let Some(last) = exprs.pop() {
                let stripped = strip_trailing_break(last);
                if !matches!(stripped, Expr::Undefined) || exprs.is_empty() {
                    exprs.push(stripped);
                }
                if exprs.len() == 1 {
                    exprs.pop().unwrap()
                } else {
                    Expr::Begin(exprs)
                }
            } else {
                Expr::Undefined
            }
        }
        Expr::If(cond, then_br, else_br) => {
            Expr::If(
                cond,
                Box::new(strip_trailing_break(*then_br)),
                Box::new(strip_trailing_break(*else_br)),
            )
        }
        other => other,
    }
}

/// Check if an expression is pure (has no side effects)
/// Pure expressions can be safely eliminated if their result is unused
/// Iterative implementation to avoid stack overflow
fn is_pure_expr(expr: &Expr) -> bool {
    let mut stack: Vec<&Expr> = vec![expr];

    while let Some(e) = stack.pop() {
        match e {
            // Literals are pure - no need to check further
            Expr::Int(_) | Expr::Bool(_) | Expr::String(_)
            | Expr::Undefined | Expr::Null | Expr::Var(_) => {}

            // Function definitions are pure (they don't execute)
            Expr::Fn(_, _) => {}

            // Binary operations are pure if operands are pure
            Expr::BinOp(_, left, right) => {
                stack.push(right);
                stack.push(left);
            }

            // Unary operations are pure if operand is pure
            Expr::BitNot(e) | Expr::LogNot(e) => {
                stack.push(e);
            }

            // If is pure if all branches are pure
            Expr::If(cond, then_br, else_br) => {
                stack.push(else_br);
                stack.push(then_br);
                stack.push(cond);
            }

            // Array literals are pure if all elements are pure
            Expr::Array(elems) => {
                for elem in elems.iter().rev() {
                    stack.push(elem);
                }
            }
            // Object literals are treated as impure because they allocate
            Expr::Object(_) => {
                return false;
            }

            // Property access is pure (no mutation)
            Expr::PropAccess(obj, _) => {
                stack.push(obj);
            }
            Expr::ComputedAccess(obj, key) => {
                stack.push(key);
                stack.push(obj);
            }
            Expr::Index(arr, idx) => {
                stack.push(idx);
                stack.push(arr);
            }
            Expr::Len(arr) => {
                stack.push(arr);
            }

            // Everything else has side effects - return false immediately
            Expr::Call(_, _) | Expr::Set(_, _) | Expr::PropSet(_, _, _)
            | Expr::ComputedSet(_, _, _) | Expr::While(_, _) | Expr::For { .. }
            | Expr::Begin(_) | Expr::Switch { .. } | Expr::Break | Expr::Continue
            | Expr::Return(_) | Expr::Throw(_) | Expr::TryCatch { .. } | Expr::New(_, _)
            | Expr::Let(_, _, _) | Expr::Opaque(_) => {
                return false;
            }
        }
    }

    true
}

/// Check if an expression contains free variables (not defined in the environment)
/// Iterative implementation to avoid stack overflow
fn has_free_vars(expr: &Expr, env: &PEnv) -> bool {
    // Work items track the expression and any locally bound variables
    // Local bindings override the environment
    enum WorkItem<'a> {
        Check(&'a Expr, Vec<String>), // (expr, local_bindings)
    }

    let mut stack: Vec<WorkItem> = vec![WorkItem::Check(expr, Vec::new())];

    while let Some(item) = stack.pop() {
        let WorkItem::Check(e, local_bindings) = item;

        // Helper to check if a variable is bound (either in env or local bindings)
        let is_bound = |name: &str| -> bool {
            local_bindings.contains(&name.to_string()) || env.borrow().contains_key(name)
        };

        match e {
            Expr::Int(_) | Expr::Bool(_) | Expr::String(_)
            | Expr::Undefined | Expr::Null | Expr::Opaque(_)
            | Expr::Break | Expr::Continue => {}

            Expr::Var(name) => {
                if !is_bound(name) {
                    return true;
                }
            }

            Expr::BinOp(_, left, right) => {
                stack.push(WorkItem::Check(right, local_bindings.clone()));
                stack.push(WorkItem::Check(left, local_bindings));
            }

            Expr::BitNot(inner) | Expr::LogNot(inner) | Expr::Len(inner) | Expr::Throw(inner) | Expr::Return(inner) => {
                stack.push(WorkItem::Check(inner, local_bindings));
            }

            Expr::If(cond, then_br, else_br) => {
                stack.push(WorkItem::Check(else_br, local_bindings.clone()));
                stack.push(WorkItem::Check(then_br, local_bindings.clone()));
                stack.push(WorkItem::Check(cond, local_bindings));
            }

            Expr::Let(name, value, body) => {
                // Body gets the new binding
                let mut new_bindings = local_bindings.clone();
                new_bindings.push(name.clone());
                stack.push(WorkItem::Check(body, new_bindings));
                // Value uses current bindings
                stack.push(WorkItem::Check(value, local_bindings));
            }

            Expr::Fn(params, body) => {
                // Body gets params as bindings
                let mut new_bindings = local_bindings;
                new_bindings.extend(params.iter().cloned());
                stack.push(WorkItem::Check(body, new_bindings));
            }

            Expr::Call(func, args) => {
                for arg in args.iter().rev() {
                    stack.push(WorkItem::Check(arg, local_bindings.clone()));
                }
                stack.push(WorkItem::Check(func, local_bindings));
            }

            Expr::Array(elems) => {
                for elem in elems.iter().rev() {
                    stack.push(WorkItem::Check(elem, local_bindings.clone()));
                }
            }

            Expr::Index(arr, idx) => {
                stack.push(WorkItem::Check(idx, local_bindings.clone()));
                stack.push(WorkItem::Check(arr, local_bindings));
            }

            Expr::Object(props) => {
                for (_, v) in props.iter().rev() {
                    stack.push(WorkItem::Check(v, local_bindings.clone()));
                }
            }

            Expr::PropAccess(obj, _) => {
                stack.push(WorkItem::Check(obj, local_bindings));
            }

            Expr::PropSet(obj, _, val) => {
                stack.push(WorkItem::Check(val, local_bindings.clone()));
                stack.push(WorkItem::Check(obj, local_bindings));
            }

            Expr::ComputedAccess(obj, key) => {
                stack.push(WorkItem::Check(key, local_bindings.clone()));
                stack.push(WorkItem::Check(obj, local_bindings));
            }

            Expr::ComputedSet(obj, key, val) => {
                stack.push(WorkItem::Check(val, local_bindings.clone()));
                stack.push(WorkItem::Check(key, local_bindings.clone()));
                stack.push(WorkItem::Check(obj, local_bindings));
            }

            Expr::While(cond, body) => {
                stack.push(WorkItem::Check(body, local_bindings.clone()));
                stack.push(WorkItem::Check(cond, local_bindings));
            }

            Expr::For { init, cond, update, body } => {
                // Collect bindings introduced by a var-decl init (nested let chain).
                let mut for_bindings = local_bindings.clone();
                if let Some(i) = init.as_ref() {
                    let mut current = i.as_ref();
                    loop {
                        if let Expr::Let(name, _, body_expr) = current {
                            for_bindings.push(name.clone());
                            current = body_expr.as_ref();
                        } else {
                            break;
                        }
                    }
                }

                stack.push(WorkItem::Check(body, for_bindings.clone()));
                if let Some(u) = update {
                    stack.push(WorkItem::Check(u, for_bindings.clone()));
                }
                if let Some(c) = cond {
                    stack.push(WorkItem::Check(c, for_bindings.clone()));
                }
                if let Some(i) = init {
                    stack.push(WorkItem::Check(i, local_bindings));
                }
            }

            Expr::Set(name, value) => {
                if !is_bound(name) {
                    return true;
                }
                stack.push(WorkItem::Check(value, local_bindings));
            }

            Expr::Begin(exprs) => {
                for e in exprs.iter().rev() {
                    stack.push(WorkItem::Check(e, local_bindings.clone()));
                }
            }

            Expr::New(ctor, args) => {
                for arg in args.iter().rev() {
                    stack.push(WorkItem::Check(arg, local_bindings.clone()));
                }
                stack.push(WorkItem::Check(ctor, local_bindings));
            }

            Expr::Switch { discriminant, cases, default } => {
                if let Some(d) = default {
                    for e in d.iter().rev() {
                        stack.push(WorkItem::Check(e, local_bindings.clone()));
                    }
                }
                for (cv, body) in cases.iter().rev() {
                    for e in body.iter().rev() {
                        stack.push(WorkItem::Check(e, local_bindings.clone()));
                    }
                    stack.push(WorkItem::Check(cv, local_bindings.clone()));
                }
                stack.push(WorkItem::Check(discriminant, local_bindings));
            }

            Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
                if let Some(fb) = finally_block {
                    stack.push(WorkItem::Check(fb, local_bindings.clone()));
                }
                // Catch block may have catch_param bound
                let mut catch_bindings = local_bindings.clone();
                if let Some(param) = catch_param {
                    catch_bindings.push(param.clone());
                }
                stack.push(WorkItem::Check(catch_block, catch_bindings));
                stack.push(WorkItem::Check(try_block, local_bindings));
            }
        }
    }

    false
}

/// Evaluate a binary operation on static values
fn eval_static_binop(op: &BinOp, left: &Value, right: &Value) -> Option<Value> {
    match (op, left, right) {
        // Arithmetic
        (BinOp::Add, Value::Int(a), Value::Int(b)) => Some(Value::Int(a.wrapping_add(*b))),
        (BinOp::Sub, Value::Int(a), Value::Int(b)) => Some(Value::Int(a.wrapping_sub(*b))),
        (BinOp::Mul, Value::Int(a), Value::Int(b)) => Some(Value::Int(a.wrapping_mul(*b))),
        (BinOp::Div, Value::Int(a), Value::Int(b)) if *b != 0 => Some(Value::Int(a / b)),
        (BinOp::Mod, Value::Int(a), Value::Int(b)) if *b != 0 => Some(Value::Int(a % b)),

        // String concatenation
        (BinOp::Add, Value::String(a), Value::String(b)) => Some(Value::String(format!("{}{}", a, b))),
        (BinOp::Add, Value::String(a), Value::Int(b)) => Some(Value::String(format!("{}{}", a, b))),
        (BinOp::Add, Value::Int(a), Value::String(b)) => Some(Value::String(format!("{}{}", a, b))),

        // Comparison
        (BinOp::Lt, Value::Int(a), Value::Int(b)) => Some(Value::Bool(a < b)),
        (BinOp::Gt, Value::Int(a), Value::Int(b)) => Some(Value::Bool(a > b)),
        (BinOp::Lte, Value::Int(a), Value::Int(b)) => Some(Value::Bool(a <= b)),
        (BinOp::Gte, Value::Int(a), Value::Int(b)) => Some(Value::Bool(a >= b)),
        (BinOp::Eq, Value::Int(a), Value::Int(b)) => Some(Value::Bool(a == b)),
        (BinOp::NotEq, Value::Int(a), Value::Int(b)) => Some(Value::Bool(a != b)),
        (BinOp::Eq, Value::Bool(a), Value::Bool(b)) => Some(Value::Bool(a == b)),
        (BinOp::NotEq, Value::Bool(a), Value::Bool(b)) => Some(Value::Bool(a != b)),
        (BinOp::Eq, Value::String(a), Value::String(b)) => Some(Value::Bool(a == b)),
        (BinOp::NotEq, Value::String(a), Value::String(b)) => Some(Value::Bool(a != b)),
        (BinOp::Eq, Value::Undefined, Value::Undefined) => Some(Value::Bool(true)),
        (BinOp::Eq, Value::Null, Value::Null) => Some(Value::Bool(true)),
        (BinOp::Eq, Value::Undefined, Value::Null) => Some(Value::Bool(true)), // JS: undefined == null
        (BinOp::Eq, Value::Null, Value::Undefined) => Some(Value::Bool(true)),

        // Logical (short-circuit semantics handled at call site for dynamic cases)
        (BinOp::And, Value::Bool(a), Value::Bool(b)) => Some(Value::Bool(*a && *b)),
        (BinOp::Or, Value::Bool(a), Value::Bool(b)) => Some(Value::Bool(*a || *b)),

        // Bitwise operations
        (BinOp::BitAnd, Value::Int(a), Value::Int(b)) => Some(Value::Int(a & b)),
        (BinOp::BitOr, Value::Int(a), Value::Int(b)) => Some(Value::Int(a | b)),
        (BinOp::BitXor, Value::Int(a), Value::Int(b)) => Some(Value::Int(a ^ b)),
        (BinOp::Shl, Value::Int(a), Value::Int(b)) => {
            // JavaScript: converts to 32-bit int, shift amount masked to 5 bits
            let a32 = *a as i32;
            let shift = (*b as u32) & 0x1f;
            Some(Value::Int((a32 << shift) as i64))
        }
        (BinOp::Shr, Value::Int(a), Value::Int(b)) => {
            // JavaScript: signed right shift
            let a32 = *a as i32;
            let shift = (*b as u32) & 0x1f;
            Some(Value::Int((a32 >> shift) as i64))
        }
        (BinOp::UShr, Value::Int(a), Value::Int(b)) => {
            // JavaScript: unsigned right shift (zero-fill)
            let a32 = *a as u32;
            let shift = (*b as u32) & 0x1f;
            Some(Value::Int((a32 >> shift) as i64))
        }

        _ => None,
    }
}

/// Collect all free variables in an expression (not bound by local scopes)
fn collect_free_vars(expr: &Expr, bound_vars: &[String]) -> HashSet<String> {
    let mut free_vars = HashSet::new();
    let mut stack: Vec<(&Expr, HashSet<String>)> = vec![(expr, bound_vars.iter().cloned().collect())];

    while let Some((e, bound)) = stack.pop() {
        match e {
            Expr::Var(name) => {
                if !bound.contains(name) {
                    free_vars.insert(name.clone());
                }
            }
            Expr::Let(name, value, body) => {
                stack.push((value, bound.clone()));
                let mut new_bound = bound;
                new_bound.insert(name.clone());
                stack.push((body, new_bound));
            }
            Expr::Fn(params, body) => {
                let mut new_bound = bound;
                new_bound.extend(params.iter().cloned());
                stack.push((body, new_bound));
            }
            Expr::BinOp(_, l, r) => {
                stack.push((l, bound.clone()));
                stack.push((r, bound));
            }
            Expr::If(c, t, e) => {
                stack.push((c, bound.clone()));
                stack.push((t, bound.clone()));
                stack.push((e, bound));
            }
            Expr::Call(f, args) => {
                stack.push((f, bound.clone()));
                for arg in args {
                    stack.push((arg, bound.clone()));
                }
            }
            Expr::Index(arr, idx) => {
                stack.push((arr, bound.clone()));
                stack.push((idx, bound));
            }
            Expr::Array(elems) => {
                for elem in elems {
                    stack.push((elem, bound.clone()));
                }
            }
            Expr::Object(fields) => {
                for (_, v) in fields {
                    stack.push((v, bound.clone()));
                }
            }
            Expr::PropAccess(obj, _) => {
                stack.push((obj, bound));
            }
            Expr::PropSet(obj, _, val) => {
                stack.push((obj, bound.clone()));
                stack.push((val, bound));
            }
            Expr::ComputedAccess(obj, key) => {
                stack.push((obj, bound.clone()));
                stack.push((key, bound));
            }
            Expr::ComputedSet(obj, key, val) => {
                stack.push((obj, bound.clone()));
                stack.push((key, bound.clone()));
                stack.push((val, bound));
            }
            Expr::Set(name, val) => {
                if !bound.contains(name) {
                    free_vars.insert(name.clone());
                }
                stack.push((val, bound));
            }
            Expr::While(cond, body) => {
                stack.push((cond, bound.clone()));
                stack.push((body, bound));
            }
            Expr::For { init, cond, update, body } => {
                if let Some(i) = init {
                    stack.push((i, bound.clone()));
                }
                if let Some(c) = cond {
                    stack.push((c, bound.clone()));
                }
                if let Some(u) = update {
                    stack.push((u, bound.clone()));
                }
                stack.push((body, bound));
            }
            Expr::Begin(stmts) => {
                for stmt in stmts {
                    stack.push((stmt, bound.clone()));
                }
            }
            Expr::Switch { discriminant, cases, default } => {
                stack.push((discriminant, bound.clone()));
                for (val, stmts) in cases {
                    stack.push((val, bound.clone()));
                    for stmt in stmts {
                        stack.push((stmt, bound.clone()));
                    }
                }
                if let Some(def) = default {
                    for stmt in def {
                        stack.push((stmt, bound.clone()));
                    }
                }
            }
            Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
                stack.push((try_block, bound.clone()));
                let catch_bound = if let Some(param) = catch_param {
                    let mut b = bound.clone();
                    b.insert(param.clone());
                    b
                } else {
                    bound.clone()
                };
                stack.push((catch_block, catch_bound));
                if let Some(fb) = finally_block {
                    stack.push((fb, bound));
                }
            }
            Expr::Throw(e) | Expr::BitNot(e) | Expr::LogNot(e) | Expr::Len(e) | Expr::Return(e) => {
                stack.push((e, bound));
            }
            Expr::New(ctor, args) => {
                stack.push((ctor, bound.clone()));
                for arg in args {
                    stack.push((arg, bound.clone()));
                }
            }
            // Literals and other expressions without subexpressions
            Expr::Int(_) | Expr::Bool(_) | Expr::String(_)
            | Expr::Undefined | Expr::Null | Expr::Opaque(_)
            | Expr::Break | Expr::Continue => {}
        }
    }

    free_vars
}

/// Check if an expression uses a variable (iterative to avoid stack overflow)
fn uses_var(expr: &Expr, var: &str) -> bool {
    // Work item: (expression, bound_vars that shadow our target)
    // We track whether our target var is currently shadowed
    enum WorkItem<'a> {
        Check(&'a Expr, bool), // (expr, is_shadowed)
        PopShadow,             // Pop shadow context
    }

    let mut stack: Vec<WorkItem> = vec![WorkItem::Check(expr, false)];
    let mut shadow_depth = 0usize; // How deep we are in shadowing contexts

    while let Some(item) = stack.pop() {
        match item {
            WorkItem::PopShadow => {
                shadow_depth -= 1;
            }
            WorkItem::Check(e, is_shadowed) => {
                let shadowed = is_shadowed || shadow_depth > 0;

                match e {
                    Expr::Int(_) | Expr::Bool(_) | Expr::String(_)
                    | Expr::Undefined | Expr::Null | Expr::Opaque(_)
                    | Expr::Break | Expr::Continue => {}

                    Expr::Var(name) => {
                        if !shadowed && name == var {
                            return true;
                        }
                    }

                    Expr::BinOp(_, left, right) => {
                        stack.push(WorkItem::Check(right, shadowed));
                        stack.push(WorkItem::Check(left, shadowed));
                    }

                    Expr::BitNot(inner) | Expr::LogNot(inner) | Expr::Len(inner) | Expr::Throw(inner) | Expr::Return(inner) => {
                        stack.push(WorkItem::Check(inner, shadowed));
                    }

                    Expr::If(cond, then_br, else_br) => {
                        stack.push(WorkItem::Check(else_br, shadowed));
                        stack.push(WorkItem::Check(then_br, shadowed));
                        stack.push(WorkItem::Check(cond, shadowed));
                    }

                    Expr::Let(name, value, body) => {
                        // Value is checked without new shadow
                        // Body is checked with potential new shadow if name == var
                        if name == var && !shadowed {
                            // Variable is shadowed in body
                            stack.push(WorkItem::PopShadow);
                            stack.push(WorkItem::Check(body, false));
                            shadow_depth += 1;
                        } else {
                            stack.push(WorkItem::Check(body, shadowed));
                        }
                        stack.push(WorkItem::Check(value, shadowed));
                    }

                    Expr::Fn(params, body) => {
                        // If var is in params, it's shadowed in body
                        if params.contains(&var.to_string()) && !shadowed {
                            stack.push(WorkItem::PopShadow);
                            stack.push(WorkItem::Check(body, false));
                            shadow_depth += 1;
                        } else {
                            stack.push(WorkItem::Check(body, shadowed));
                        }
                    }

                    Expr::Call(func, args) => {
                        for arg in args.iter().rev() {
                            stack.push(WorkItem::Check(arg, shadowed));
                        }
                        stack.push(WorkItem::Check(func, shadowed));
                    }

                    Expr::Array(elems) => {
                        for elem in elems.iter().rev() {
                            stack.push(WorkItem::Check(elem, shadowed));
                        }
                    }

                    Expr::Index(arr, idx) => {
                        stack.push(WorkItem::Check(idx, shadowed));
                        stack.push(WorkItem::Check(arr, shadowed));
                    }

                    Expr::Object(props) => {
                        for (_, v) in props.iter().rev() {
                            stack.push(WorkItem::Check(v, shadowed));
                        }
                    }

                    Expr::PropAccess(obj, _) => {
                        stack.push(WorkItem::Check(obj, shadowed));
                    }

                    Expr::PropSet(obj, _, val) => {
                        stack.push(WorkItem::Check(val, shadowed));
                        stack.push(WorkItem::Check(obj, shadowed));
                    }

                    Expr::ComputedAccess(obj, key) => {
                        stack.push(WorkItem::Check(key, shadowed));
                        stack.push(WorkItem::Check(obj, shadowed));
                    }

                    Expr::ComputedSet(obj, key, val) => {
                        stack.push(WorkItem::Check(val, shadowed));
                        stack.push(WorkItem::Check(key, shadowed));
                        stack.push(WorkItem::Check(obj, shadowed));
                    }

                    Expr::While(cond, body) => {
                        stack.push(WorkItem::Check(body, shadowed));
                        stack.push(WorkItem::Check(cond, shadowed));
                    }

                    Expr::For { init, cond, update, body } => {
                        stack.push(WorkItem::Check(body, shadowed));
                        if let Some(u) = update {
                            stack.push(WorkItem::Check(u, shadowed));
                        }
                        if let Some(c) = cond {
                            stack.push(WorkItem::Check(c, shadowed));
                        }
                        if let Some(i) = init {
                            stack.push(WorkItem::Check(i, shadowed));
                        }
                    }

                    Expr::Set(name, value) => {
                        if !shadowed && name == var {
                            return true;
                        }
                        stack.push(WorkItem::Check(value, shadowed));
                    }

                    Expr::Begin(exprs) => {
                        for e in exprs.iter().rev() {
                            stack.push(WorkItem::Check(e, shadowed));
                        }
                    }

                    Expr::New(ctor, args) => {
                        for arg in args.iter().rev() {
                            stack.push(WorkItem::Check(arg, shadowed));
                        }
                        stack.push(WorkItem::Check(ctor, shadowed));
                    }

                    Expr::Switch { discriminant, cases, default } => {
                        if let Some(d) = default {
                            for e in d.iter().rev() {
                                stack.push(WorkItem::Check(e, shadowed));
                            }
                        }
                        for (cv, body) in cases.iter().rev() {
                            for e in body.iter().rev() {
                                stack.push(WorkItem::Check(e, shadowed));
                            }
                            stack.push(WorkItem::Check(cv, shadowed));
                        }
                        stack.push(WorkItem::Check(discriminant, shadowed));
                    }

                    Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
                        if let Some(fb) = finally_block {
                            stack.push(WorkItem::Check(fb, shadowed));
                        }
                        // Catch block - variable may be shadowed by catch_param
                        if catch_param.as_ref().map_or(false, |p| p == var) && !shadowed {
                            stack.push(WorkItem::PopShadow);
                            stack.push(WorkItem::Check(catch_block, false));
                            shadow_depth += 1;
                        } else {
                            stack.push(WorkItem::Check(catch_block, shadowed));
                        }
                        stack.push(WorkItem::Check(try_block, shadowed));
                    }
                }
            }
        }
    }

    false
}

/// Convert a PEnv to a regular Env (only keeping static values)
fn penv_to_env(penv: &PEnv) -> crate::value::Env {
    let env = new_env();
    for (k, v) in penv.borrow().iter() {
        if let PValue::Static(val) = v {
            env.borrow_mut().insert(k.clone(), val.clone());
        }
    }
    env
}

/// Convert a regular Env to a PEnv (all values become static)
fn env_to_penv(env: &crate::value::Env) -> PEnv {
    let penv = new_penv();
    for (k, v) in env.borrow().iter() {
        penv.borrow_mut().insert(k.clone(), PValue::Static(v.clone()));
    }
    penv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn pe(s: &str) -> PValue {
        let expr = parse(s).unwrap();
        partial_eval(&expr, &new_penv())
    }

    fn pe_with_dynamic(s: &str, dynamic_vars: &[&str]) -> PValue {
        let expr = parse(s).unwrap();
        let env = new_penv();
        for var in dynamic_vars {
            env.borrow_mut().insert(var.to_string(), PValue::Dynamic(Expr::Var(var.to_string())));
        }
        partial_eval(&expr, &env)
    }

    #[test]
    fn test_static_arithmetic() {
        match pe("(+ 1 2)") {
            PValue::Static(Value::Int(3)) => {}
            other => panic!("Expected Static(Int(3)), got {:?}", other),
        }
    }

    #[test]
    fn test_static_let() {
        match pe("(let x 5 (+ x 3))") {
            PValue::Static(Value::Int(8)) => {}
            other => panic!("Expected Static(Int(8)), got {:?}", other),
        }
    }

    #[test]
    fn test_static_if() {
        match pe("(if true 1 2)") {
            PValue::Static(Value::Int(1)) => {}
            other => panic!("Expected Static(Int(1)), got {:?}", other),
        }
    }

    #[test]
    fn test_dynamic_var() {
        match pe_with_dynamic("(+ x 1)", &["x"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "(+ x 1)");
            }
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    #[test]
    fn test_partial_eval_with_static_and_dynamic() {
        // (let a 3 (+ a x)) with x dynamic should give (+ 3 x)
        match pe_with_dynamic("(let a 3 (+ a x))", &["x"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "(+ 3 x)");
            }
            other => panic!("Expected Dynamic (+ 3 x), got {:?}", other),
        }
    }

    #[test]
    fn test_static_if_branch_elimination() {
        // Even with dynamic var, if condition is static, only one branch is taken
        match pe_with_dynamic("(if (< 1 2) (+ x 1) (+ x 2))", &["x"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "(+ x 1)");
            }
            other => panic!("Expected Dynamic (+ x 1), got {:?}", other),
        }
    }

    #[test]
    fn test_function_inlining() {
        // (let f (fn (y) (+ y 1)) (call f 5)) should give 6
        match pe("(let f (fn (y) (+ y 1)) (call f 5))") {
            PValue::Static(Value::Int(6)) => {}
            other => panic!("Expected Static(Int(6)), got {:?}", other),
        }
    }

    #[test]
    fn test_closure_specialization() {
        // (let a 10 (let f (fn (x) (+ x a)) (call f 5))) should give 15
        match pe("(let a 10 (let f (fn (x) (+ x a)) (call f 5)))") {
            PValue::Static(Value::Int(15)) => {}
            other => panic!("Expected Static(Int(15)), got {:?}", other),
        }
    }

    #[test]
    fn test_closure_with_dynamic_arg() {
        // (let a 10 (let f (fn (x) (+ x a)) (call f y))) with y dynamic should give (+ y 10)
        match pe_with_dynamic("(let a 10 (let f (fn (x) (+ x a)) (call f y)))", &["y"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "(+ y 10)");
            }
            other => panic!("Expected Dynamic (+ y 10), got {:?}", other),
        }
    }

    #[test]
    fn test_higher_order_function() {
        // make-adder example
        let code = "(let make-adder (fn (n) (fn (x) (+ x n)))
                      (let add5 (call make-adder 5)
                        (call add5 y)))";
        match pe_with_dynamic(code, &["y"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "(+ y 5)");
            }
            other => panic!("Expected Dynamic (+ y 5), got {:?}", other),
        }
    }

    #[test]
    fn test_array_static() {
        match pe("(index (array 10 20 30) 1)") {
            PValue::Static(Value::Int(20)) => {}
            other => panic!("Expected Static(Int(20)), got {:?}", other),
        }
    }

    #[test]
    fn test_array_len_static() {
        match pe("(len (array 1 2 3 4 5))") {
            PValue::Static(Value::Int(5)) => {}
            other => panic!("Expected Static(Int(5)), got {:?}", other),
        }
    }

    #[test]
    fn test_array_with_dynamic_index() {
        match pe_with_dynamic("(index (array 10 20 30) i)", &["i"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "(index (array 10 20 30) i)");
            }
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    #[test]
    fn test_while_static_unroll() {
        // Fully static while loop should be unrolled
        let code = "(let x 0 (begin (while (< x 5) (set! x (+ x 1))) x))";
        match pe(code) {
            PValue::Static(Value::Int(5)) => {}
            other => panic!("Expected Static(Int(5)), got {:?}", other),
        }
    }

    #[test]
    fn test_while_with_dynamic_cond() {
        // Dynamic condition should emit residual
        match pe_with_dynamic("(while (< x 5) (set! x (+ x 1)))", &["x"]) {
            PValue::Dynamic(e) => {
                assert!(e.to_string().contains("while"));
            }
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    #[test]
    fn test_set_static() {
        let code = "(let x 1 (begin (set! x 42) x))";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_begin_static() {
        let code = "(begin 1 2 3)";
        match pe(code) {
            PValue::Static(Value::Int(3)) => {}
            other => panic!("Expected Static(Int(3)), got {:?}", other),
        }
    }

    // =========================================================================
    // Object tests - these were missing entirely!
    // =========================================================================

    #[test]
    fn test_object_static_prop_set_and_access() {
        // Fully static: create object, set property, read it back
        let code = "(let obj (object) (begin (prop-set! obj \"x\" 42) (prop obj \"x\")))";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_object_prop_set_with_dynamic_value_preserves_variable() {
        // CRITICAL BUG TEST: When we have a static object bound to a variable,
        // and set a property to a dynamic value, the residual should reference
        // the variable name, NOT emit an object literal.
        //
        // Current broken behavior produces:
        //   (prop-set! (object) "x" dyn)
        //
        // Correct behavior should produce:
        //   (let obj (object) (prop-set! obj "x" dyn))
        let code = "(let obj (object) (prop-set! obj \"x\" dyn))";
        match pe_with_dynamic(code, &["dyn"]) {
            PValue::Dynamic(e) => {
                let result = e.to_string();
                assert!(
                    result.contains("let obj") || result.contains("prop-set! obj"),
                    "Expected variable 'obj' to be preserved in residual, got: {}",
                    result
                );
                // Should NOT contain an object literal where the variable should be
                assert!(
                    !result.contains("prop-set! (object)"),
                    "BUG: Object literal emitted instead of variable reference: {}",
                    result
                );
            }
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    #[test]
    fn test_object_multiple_props_some_dynamic() {
        // Set multiple properties, some static, some dynamic
        // Static props should be folded, dynamic should preserve var reference
        let code = "(let obj (object) (begin (prop-set! obj \"a\" 1) (prop-set! obj \"b\" dyn) obj))";
        match pe_with_dynamic(code, &["dyn"]) {
            PValue::Dynamic(e) => {
                let result = e.to_string();
                // The object should have 'a' folded in, but 'b' should use variable
                assert!(
                    !result.contains("prop-set! (object"),
                    "BUG: Object literal emitted instead of variable reference: {}",
                    result
                );
            }
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    // =========================================================================
    // Closure mutation tests - closures that capture mutable state
    // =========================================================================

    #[test]
    fn test_closure_with_mutable_captured_state() {
        // A closure that captures and mutates a variable CAN be inlined now that
        // we properly propagate mutations through the original environment.
        // The key is that each call sees the updated state from previous calls.
        //
        // First call: count 0 -> 1, returns 1
        // Second call: count 1 -> 2, returns 2
        // Begin returns last value: 2
        let code = "(let count 0
                      (let counter (fn () (begin (set! count (+ count 1)) count))
                        (begin (call counter) (call counter))))";
        match pe(code) {
            PValue::Static(Value::Int(2)) => {
                // Correct! Both calls executed, each saw the updated state
            }
            PValue::Static(Value::Int(1)) => {
                panic!("BUG: Closure was incorrectly inlined, causing both calls \
                        to see count=0 and return 1 instead of sharing state.");
            }
            other => panic!("Expected Static(Int(2)), got {:?}", other),
        }
    }

    #[test]
    fn test_closure_without_mutation_can_be_inlined() {
        // A closure that does NOT mutate captured state CAN be safely inlined
        let code = "(let x 10
                      (let add-x (fn (y) (+ y x))
                        (+ (call add-x 1) (call add-x 2))))";
        match pe(code) {
            PValue::Static(Value::Int(23)) => {} // (1+10) + (2+10) = 23
            other => panic!("Expected Static(Int(23)), got {:?}", other),
        }
    }

    #[test]
    fn test_iife_with_internal_state_preserved() {
        // IIFE that returns a closure which captures internal state
        // Since the inner closure mutates captured state, it must be preserved
        let code = "(let make-counter (fn ()
                      (let count 0
                        (fn () (begin (set! count (+ count 1)) count))))
                    (let counter (call make-counter)
                      (+ (call counter) (call counter))))";
        match pe(code) {
            PValue::Dynamic(e) => {
                let result = e.to_string();
                // The inner closure should be preserved
                assert!(
                    result.contains("fn ()") || result.contains("call"),
                    "Closure with mutable state should produce residual code, got: {}",
                    result
                );
            }
            PValue::Static(Value::Int(2)) => {
                panic!("BUG: Counter returned 1+1=2 instead of preserving the closure. \
                        Closure state mutation is not being detected.");
            }
            other => panic!("Expected Dynamic residual, got {:?}", other),
        }
    }

    // =========================================================================
    // Constant folding / optimization tests
    // =========================================================================

    #[test]
    fn test_bitwise_and_with_zero_folds() {
        // (& x 0) should always be 0, regardless of x
        match pe_with_dynamic("(& x 0)", &["x"]) {
            PValue::Static(Value::Int(0)) => {}
            other => panic!("Expected Static(Int(0)), got {:?}. \
                           Bitwise AND with 0 should always fold to 0.", other),
        }
    }

    #[test]
    fn test_bitwise_or_with_zero_folds() {
        // (| x 0) should be x
        match pe_with_dynamic("(| x 0)", &["x"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "x",
                    "Expected just 'x', got {}. OR with 0 should fold to the other operand.", e);
            }
            other => panic!("Expected Dynamic(Var(x)), got {:?}", other),
        }
    }

    #[test]
    fn test_multiply_by_zero_folds() {
        // (* x 0) should be 0
        match pe_with_dynamic("(* x 0)", &["x"]) {
            PValue::Static(Value::Int(0)) => {}
            other => panic!("Expected Static(Int(0)), got {:?}. \
                           Multiply by 0 should fold to 0.", other),
        }
    }

    #[test]
    fn test_multiply_by_one_folds() {
        // (* x 1) should be x
        match pe_with_dynamic("(* x 1)", &["x"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "x",
                    "Expected just 'x', got {}. Multiply by 1 should fold to the other operand.", e);
            }
            other => panic!("Expected Dynamic(Var(x)), got {:?}", other),
        }
    }

    #[test]
    fn test_add_zero_folds() {
        // (+ x 0) should be x
        match pe_with_dynamic("(+ x 0)", &["x"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "x",
                    "Expected just 'x', got {}. Add 0 should fold to the other operand.", e);
            }
            other => panic!("Expected Dynamic(Var(x)), got {:?}", other),
        }
    }

    #[test]
    fn test_subtract_zero_folds() {
        // (- x 0) should be x
        match pe_with_dynamic("(- x 0)", &["x"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "x",
                    "Expected just 'x', got {}. Subtract 0 should fold to the other operand.", e);
            }
            other => panic!("Expected Dynamic(Var(x)), got {:?}", other),
        }
    }

    #[test]
    fn test_closure_body_constants_folded_on_call() {
        // When a closure is CALLED, constants in the body should be folded
        let code = "(let f (fn () (+ 1 2)) (call f))";
        match pe(code) {
            PValue::Static(Value::Int(3)) => {}
            other => panic!("Expected Static(Int(3)), got {:?}", other),
        }
    }

    #[test]
    fn test_identity_optimizations_in_residual() {
        // When we have dynamic values, identity operations should still fold:
        // (- x 0) should become just x, even in more complex expressions
        let code = "(let f (fn (x) (- x 0)) (call f y))";
        match pe_with_dynamic(code, &["y"]) {
            PValue::Dynamic(e) => {
                // The (- x 0) inside should fold to just x, giving us just y
                assert_eq!(e.to_string(), "y",
                    "Expected 'y' after folding (- x 0), got: {}", e);
            }
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    #[test]
    fn test_array_computed_set_preserves_variable_reference() {
        // CRITICAL BUG TEST: When we have a static array bound to a variable,
        // and do a computed-set! with a dynamic index or value, the residual
        // should reference the variable name, NOT emit an array literal.
        //
        // Current broken behavior produces:
        //   (computed-set! (array) 0 dyn)
        //
        // Correct behavior should produce:
        //   (let arr (array) (computed-set! arr 0 dyn))

        // Construct AST directly since parser doesn't support computed-set!
        let arr_expr = Expr::Let(
            "arr".to_string(),
            Box::new(Expr::Array(vec![])),
            Box::new(Expr::ComputedSet(
                Box::new(Expr::Var("arr".to_string())),
                Box::new(Expr::Int(0)),
                Box::new(Expr::Var("dyn".to_string())),
            )),
        );

        let env = new_penv();
        env.borrow_mut().insert("dyn".to_string(), PValue::Dynamic(Expr::Var("dyn".to_string())));

        let pv = partial_eval(&arr_expr, &env);
        let result = residualize(&pv).to_string();

        // Should contain "arr" as the variable being set
        assert!(
            result.contains("computed-set! arr") || result.contains("let arr"),
            "BUG: Array literal emitted instead of variable reference.\n\
             Expected 'arr' to be preserved, got: {}",
            result
        );
        // Should NOT contain (computed-set! (array) ...) - that's the bug
        assert!(
            !result.contains("computed-set! (array)"),
            "BUG: Fresh array literal emitted instead of variable reference: {}",
            result
        );
    }

    #[test]
    fn test_array_computed_set_static_index_and_value() {
        // Test when both index and value are static - this should either:
        // 1. Mutate the static array in place (if we support that), OR
        // 2. Emit a residual with the variable reference preserved
        //
        // Bug scenario: When index and value are both static but array is StaticNamed,
        // we might be emitting (array) literal instead of the variable name.

        let arr_expr = Expr::Let(
            "lookup".to_string(),
            Box::new(Expr::Array(vec![])),
            Box::new(Expr::Begin(vec![
                Expr::ComputedSet(
                    Box::new(Expr::Var("lookup".to_string())),
                    Box::new(Expr::Int(21)),
                    Box::new(Expr::Int(34)),
                ),
                Expr::ComputedSet(
                    Box::new(Expr::Var("lookup".to_string())),
                    Box::new(Expr::Int(26)),
                    Box::new(Expr::Int(42)),
                ),
                Expr::Var("lookup".to_string()),
            ])),
        );

        let env = new_penv();
        let pv = partial_eval(&arr_expr, &env);
        let result = residualize(&pv).to_string();

        // Should NOT contain multiple (array) literals
        // Either the sets are folded into the array, or they use variable references
        let array_literal_count = result.matches("(array)").count();
        assert!(
            array_literal_count <= 1,
            "BUG: Multiple array literals created.\n\
             Found {} occurrences of '(array)' in: {}",
            array_literal_count, result
        );

        // If there are computed-set! calls, they should reference 'lookup', not '(array)'
        if result.contains("computed-set!") {
            assert!(
                !result.contains("computed-set! (array)"),
                "BUG: computed-set! uses (array) literal instead of variable: {}",
                result
            );
        }
    }

    #[test]
    fn test_array_multiple_computed_sets_preserve_variable() {
        // Multiple computed-set! operations on the same array should all
        // reference the same variable, not create fresh arrays
        let arr_expr = Expr::Let(
            "lookup".to_string(),
            Box::new(Expr::Array(vec![])),
            Box::new(Expr::Begin(vec![
                Expr::ComputedSet(
                    Box::new(Expr::Var("lookup".to_string())),
                    Box::new(Expr::Var("idx1".to_string())),
                    Box::new(Expr::Int(42)),
                ),
                Expr::ComputedSet(
                    Box::new(Expr::Var("lookup".to_string())),
                    Box::new(Expr::Var("idx2".to_string())),
                    Box::new(Expr::Int(99)),
                ),
                Expr::Var("lookup".to_string()),
            ])),
        );

        let env = new_penv();
        env.borrow_mut().insert("idx1".to_string(), PValue::Dynamic(Expr::Var("idx1".to_string())));
        env.borrow_mut().insert("idx2".to_string(), PValue::Dynamic(Expr::Var("idx2".to_string())));

        let pv = partial_eval(&arr_expr, &env);
        let result = residualize(&pv).to_string();

        // Count occurrences of "(array)" - should only appear once (in the let binding)
        let array_literal_count = result.matches("(array)").count();
        assert!(
            array_literal_count <= 1,
            "BUG: Multiple array literals created instead of using variable reference.\n\
             Found {} occurrences of '(array)' in: {}",
            array_literal_count, result
        );
    }

    #[test]
    fn test_array_set_then_computed_set_preserves_reference() {
        // BUG: When array is assigned via set! (not let), computed-set! loses
        // the variable reference.
        //
        // JavaScript pattern:
        //   var arr;      // declares arr as undefined
        //   arr = [];     // assigns empty array via set!
        //   arr[0] = val; // should use 'arr' not '(array)'
        //
        // The bug is that set! stores PValue::Static instead of StaticNamed

        let code = "(let arr undefined (begin (set! arr (array)) (computed-set! arr 0 dyn)))";
        let pv = pe_with_dynamic(code, &["dyn"]);
        let result = residualize(&pv).to_string();

        // Should reference 'arr', not emit fresh (array) literal
        assert!(
            !result.contains("computed-set! (array)"),
            "BUG: set! doesn't preserve StaticNamed for arrays.\n\
             computed-set! uses (array) literal instead of variable: {}",
            result
        );
    }
}
