//! Full partial evaluation with path exploration
//!
//! This module executes JavaScript with abstract values, exploring ALL codepaths
//! and building residual code for dynamic parts.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use swc_ecma_ast::*;

use crate::abstract_value::{AbstractValue, FunctionValue, JsValue};
use crate::parser::expr_as_ident;
use crate::emit;

/// Environment mapping variable names to abstract values
pub type Env = Rc<RefCell<HashMap<String, AbstractValue>>>;

/// Create a new empty environment
pub fn new_env() -> Env {
    Rc::new(RefCell::new(HashMap::new()))
}

/// Extract a single assignment from a statement: `target = expr;`
/// Returns (target_var_name, rhs_expr) if the statement is a simple assignment.
/// Handles both bare expression statements and single-statement blocks.
fn extract_single_assignment(stmt: &Stmt) -> Option<(String, &Expr)> {
    match stmt {
        Stmt::Expr(expr_stmt) => {
            if let Expr::Assign(assign) = expr_stmt.expr.as_ref() {
                if assign.op == AssignOp::Assign {
                    if let AssignTarget::Simple(SimpleAssignTarget::Ident(id)) = &assign.left {
                        return Some((id.id.sym.to_string(), &assign.right));
                    }
                }
            }
            None
        }
        Stmt::Block(block) => {
            if block.stmts.len() == 1 {
                extract_single_assignment(&block.stmts[0])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Deep clone an AbstractValue, creating new Rc<RefCell> for arrays and objects
fn deep_clone_value(val: &AbstractValue) -> AbstractValue {
    match val {
        AbstractValue::Known(JsValue::Array(arr)) => {
            // Deep clone each element
            let cloned_elements: Vec<AbstractValue> = arr.borrow()
                .iter()
                .map(|v| deep_clone_value(v))
                .collect();
            AbstractValue::Known(JsValue::Array(Rc::new(RefCell::new(cloned_elements))))
        }
        AbstractValue::Known(JsValue::Object(obj)) => {
            // Deep clone each property
            let cloned_props: HashMap<String, AbstractValue> = obj.borrow()
                .iter()
                .map(|(k, v)| (k.clone(), deep_clone_value(v)))
                .collect();
            AbstractValue::Known(JsValue::Object(Rc::new(RefCell::new(cloned_props))))
        }
        // Other values don't have interior mutability, regular clone is fine
        other => other.clone(),
    }
}

/// Clone an environment for path splitting (DEEP clone to avoid shared mutable state)
pub fn clone_env(env: &Env) -> Env {
    let env_map = env.borrow();
    let cloned_map: HashMap<String, AbstractValue> = env_map
        .iter()
        .map(|(k, v)| (k.clone(), deep_clone_value(v)))
        .collect();
    Rc::new(RefCell::new(cloned_map))
}

/// A traced operation - what we recorded during evaluation
#[derive(Debug, Clone)]
pub enum TracedOp {
    /// A statement that was executed with known values (can be removed)
    StaticStmt(Stmt),
    /// A statement that involves dynamic values (must keep)
    DynamicStmt(Stmt),
    /// A value was computed
    Value(AbstractValue),
    /// A branch was taken (both paths explored if dynamic)
    Branch { condition: AbstractValue, taken: bool },
    /// A function was called
    Call { callee: String, args: Vec<AbstractValue>, result: AbstractValue },
    /// An exception was thrown
    Throw(AbstractValue),
    /// TextDecoder was instantiated - MILESTONE!
    TextDecoderNew,
    /// TextDecoder.decode was called - MILESTONE!
    TextDecoderDecode(AbstractValue),
    /// A method was called on an object (for residual code)
    MethodCall {
        object: String,
        method: String,
        args: Vec<AbstractValue>,
    },
}

/// Pending ternary split information - used to propagate splits up to loop level
#[derive(Clone)]
pub struct PendingSplit {
    /// The condition expression (to re-evaluate for side effects)
    pub condition_expr: Box<Expr>,
    /// The variable being assigned
    pub target_var: String,
    /// The consequent expression (true branch)
    pub cons_expr: Box<Expr>,
    /// The alternate expression (false branch)
    pub alt_expr: Box<Expr>,
}

/// The evaluator state
pub struct Evaluator {
    /// Current environment
    pub env: Env,
    /// Trace of operations
    pub trace: Vec<TracedOp>,
    /// Maximum number of steps (very high!)
    pub max_steps: usize,
    /// Current step count
    pub steps: usize,
    /// Call stack depth
    pub call_depth: usize,
    /// Maximum call depth
    pub max_call_depth: usize,
    /// Global variables that are known to be dynamic
    pub dynamic_vars: Vec<String>,
    /// Debug mode - print progress
    pub debug: bool,
    /// Residual statements to emit
    pub residual: Vec<Stmt>,
    /// Loop iteration counts (for detecting infinite loops)
    pub loop_counts: HashMap<usize, usize>,
    /// Next loop ID
    pub next_loop_id: usize,
    /// Depth of callback specialization (>0 means inside callback)
    pub callback_depth: usize,
    /// Variables that should emit residual even when Known (closure captures)
    pub closure_vars: std::collections::HashSet<String>,
    /// Pending split that needs to be handled at loop level
    pub pending_split: Option<PendingSplit>,
    /// Whether we're currently executing a split (prevents nested splits)
    pub in_split_execution: bool,
    /// Variables to treat as closure captures during callback specialization
    pub closure_var_hints: Vec<String>,
}

impl Evaluator {
    pub fn new(env: Env) -> Self {
        Evaluator {
            env,
            trace: Vec::new(),
            max_steps: 1_000_000_000, // 1 billion steps!
            steps: 0,
            call_depth: 0,
            max_call_depth: 1000,
            dynamic_vars: Vec::new(),
            debug: true,
            residual: Vec::new(),
            loop_counts: HashMap::new(),
            next_loop_id: 0,
            callback_depth: 0,
            closure_vars: std::collections::HashSet::new(),
            pending_split: None,
            in_split_execution: false,
            closure_var_hints: Vec::new(),
        }
    }

    /// Mark a variable as dynamic (unknown at partial eval time)
    pub fn mark_dynamic(&mut self, name: &str) {
        self.dynamic_vars.push(name.to_string());
        self.env.borrow_mut().insert(name.to_string(), AbstractValue::dynamic(name));
    }

    /// Get a variable's value
    pub fn get_var(&self, name: &str) -> AbstractValue {
        self.env
            .borrow()
            .get(name)
            .cloned()
            .unwrap_or_else(|| AbstractValue::dynamic(name))
    }

    /// Set a variable's value
    pub fn set_var(&mut self, name: &str, value: AbstractValue) {
        self.env.borrow_mut().insert(name.to_string(), value);
    }

    /// Check if we should stop
    pub fn should_stop(&self) -> bool {
        self.steps >= self.max_steps
    }

    /// Count a step
    fn step(&mut self) -> bool {
        self.steps += 1;
        if self.steps % 1_000_000 == 0 && self.debug {
            eprintln!("Progress: {} million steps", self.steps / 1_000_000);
        }
        !self.should_stop()
    }

    /// Debug print
    fn debug(&self, msg: &str) {
        if self.debug && self.steps < 1000 {
            eprintln!("[step {}] {}", self.steps, msg);
        }
    }

    /// Recursively specialize all functions in a value (arrays, objects, etc.)
    /// This ensures functions inside data structures get specialized before emission
    fn specialize_value_deep(&mut self, value: AbstractValue) -> AbstractValue {
        match &value {
            // Recursively process arrays
            AbstractValue::Known(JsValue::Array(arr)) => {
                let specialized: Vec<AbstractValue> = arr
                    .borrow()
                    .iter()
                    .map(|v| self.specialize_value_deep(v.clone()))
                    .collect();
                AbstractValue::known_array(specialized)
            }
            // Recursively process objects
            AbstractValue::Known(JsValue::Object(obj)) => {
                let mut specialized = std::collections::HashMap::new();
                for (k, v) in obj.borrow().iter() {
                    specialized.insert(k.clone(), self.specialize_value_deep(v.clone()));
                }
                AbstractValue::known_object(specialized)
            }
            // Specialize functions
            AbstractValue::Known(JsValue::Function(_)) => {
                self.try_specialize_callback(value)
            }
            // Other values pass through unchanged
            _ => value,
        }
    }

    /// Try to specialize a callback function by tracing into it
    /// This transforms obfuscated while-switch VMs into straight-line code
    fn try_specialize_callback(&mut self, value: AbstractValue) -> AbstractValue {
        // Only specialize Known functions
        let (params, body) = match &value {
            AbstractValue::Known(JsValue::Function(FunctionValue::Known { params, body })) => {
                (params.clone(), body.clone())
            }
            _ => return value, // Not a function, return as-is
        };


        if self.debug {
            eprintln!("[step {}] Specializing callback with params: {:?}", self.steps, params);
        }

        // Save current residual state - we'll collect new statements for the callback
        let saved_residual = std::mem::take(&mut self.residual);

        // Save the current environment
        let saved_env = clone_env(&self.env);

        // Mark that we're inside callback specialization
        self.callback_depth += 1;
        let saved_closure_vars = std::mem::take(&mut self.closure_vars);
        for var in &self.closure_var_hints {
            self.closure_vars.insert(var.clone());
        }

        // Bind function parameters to dynamic values (e.g., 'event' parameter)
        for param in &params {
            self.set_var(param, AbstractValue::dynamic(param));
        }

        // Set up `arguments` as a known array with one dynamic element (the event)
        // This allows callbacks that use `arguments` instead of named params to specialize properly
        let event_arg = AbstractValue::dynamic("event");
        self.set_var("arguments", AbstractValue::known_array(vec![event_arg]));

        // Trace into the function body
        // This will execute the while-switch state machine with known state
        // and generate residual code for dynamic operations
        if self.call_depth >= self.max_call_depth {
            // Restore state and return original function
            self.residual = saved_residual;
            self.env = saved_env;
            self.callback_depth -= 1;
            self.closure_vars = saved_closure_vars;
            return value;
        }

        self.call_depth += 1;
        let _result = self.eval_block(&body);
        self.call_depth -= 1;

        // Collect the specialized statements
        let specialized_stmts = std::mem::replace(&mut self.residual, saved_residual);

        // Restore the environment and callback state
        self.env = saved_env;
        self.callback_depth -= 1;
        self.closure_vars = saved_closure_vars;

        if specialized_stmts.is_empty() {
            // No residual generated - the function was fully static
            // Return a function with an empty body (or original if preferred)
            if self.debug {
                eprintln!("[step {}] Callback fully specialized (no residual)", self.steps);
            }
            // Return a simple function that does nothing (all effects were static)
            let empty_body = BlockStmt {
                span: Default::default(),
                ctxt: Default::default(),
                stmts: vec![],
            };
            return AbstractValue::Known(JsValue::Function(FunctionValue::Known {
                params,
                body: empty_body,
            }));
        }

        // Check if we need to add an "event" parameter BEFORE cleanup
        // (because cleanup will simplify arguments[...] to event)
        let needs_event_param = uses_arguments(&specialized_stmts);

        // Clean up the specialized statements:
        // 1. Remove dead stores (assignments that get immediately overwritten)
        // 2. Simplify expressions
        let cleaned_stmts = cleanup_residual_stmts(specialized_stmts);

        if self.debug {
            eprintln!("[step {}] Callback specialized with {} residual statements (cleaned from original)",
                      self.steps, cleaned_stmts.len());
        }

        // Add "event" parameter if the callback uses arguments
        let final_params = if needs_event_param {
            vec!["event".to_string()]
        } else {
            params
        };

        // Build a new function with the specialized body
        let specialized_body = BlockStmt {
            span: Default::default(),
            ctxt: Default::default(),
            stmts: cleaned_stmts,
        };

        AbstractValue::Known(JsValue::Function(FunctionValue::Known {
            params: final_params,
            body: specialized_body,
        }))
    }

    /// Evaluate an expression to an abstract value
    pub fn eval_expr(&mut self, expr: &Expr) -> AbstractValue {
        if !self.step() {
            return AbstractValue::Top;
        }

        match expr {
            // Literals
            Expr::Lit(lit) => self.eval_lit(lit),

            // Variables
            Expr::Ident(id) => {
                // Handle special global identifiers
                match id.sym.as_ref() {
                    "undefined" => AbstractValue::known_undefined(),
                    "NaN" => AbstractValue::known_number(f64::NAN),
                    "Infinity" => AbstractValue::known_number(f64::INFINITY),
                    _ => {
                        let val = self.get_var(&id.sym);
                        self.debug(&format!("get {} = {}", id.sym, val));
                        val
                    }
                }
            }

            // Binary operations
            Expr::Bin(bin) => {
                let left = self.eval_expr(&bin.left);
                let right = self.eval_expr(&bin.right);
                self.eval_binop(bin.op, &left, &right)
            }

            // Unary operations
            Expr::Unary(unary) => {
                let arg = self.eval_expr(&unary.arg);
                self.eval_unop(unary.op, &arg)
            }

            // Assignment
            Expr::Assign(assign) => {
                let value = self.eval_expr(&assign.right);
                self.eval_assign(&assign.left, assign.op, value)
            }

            // Update (++, --)
            Expr::Update(update) => {
                self.eval_update(update)
            }

            // Array literal
            Expr::Array(arr) => {
                let elements: Vec<AbstractValue> = arr
                    .elems
                    .iter()
                    .map(|e| {
                        e.as_ref()
                            .map(|e| self.eval_expr(&e.expr))
                            .unwrap_or(AbstractValue::known_undefined())
                    })
                    .collect();
                AbstractValue::known_array(elements)
            }

            // Object literal
            Expr::Object(obj) => {
                let mut props = HashMap::new();
                for prop in &obj.props {
                    match prop {
                        PropOrSpread::Prop(prop) => {
                            match prop.as_ref() {
                                Prop::KeyValue(kv) => {
                                    let key = match &kv.key {
                                        PropName::Ident(id) => id.sym.to_string(),
                                        PropName::Str(s) => s.value.to_string(),
                                        PropName::Num(n) => n.value.to_string(),
                                        _ => continue,
                                    };
                                    let value = self.eval_expr(&kv.value);
                                    props.insert(key, value);
                                }
                                Prop::Method(method) => {
                                    let key = match &method.key {
                                        PropName::Ident(id) => id.sym.to_string(),
                                        _ => continue,
                                    };
                                    let params: Vec<String> = method.function.params
                                        .iter()
                                        .filter_map(|p| {
                                            if let Pat::Ident(id) = &p.pat {
                                                Some(id.sym.to_string())
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    if let Some(body) = &method.function.body {
                                        let func = AbstractValue::Known(JsValue::Function(
                                            FunctionValue::Known {
                                                params,
                                                body: body.clone(),
                                            }
                                        ));
                                        props.insert(key, func);
                                    }
                                }
                                Prop::Shorthand(id) => {
                                    let val = self.get_var(&id.sym);
                                    props.insert(id.sym.to_string(), val);
                                }
                                _ => {}
                            }
                        }
                        PropOrSpread::Spread(_) => {
                            // Handle spread later
                        }
                    }
                }
                AbstractValue::known_object(props)
            }

            // Member access (obj.prop or obj[key])
            Expr::Member(member) => {
                let obj = self.eval_expr(&member.obj);
                self.eval_member_access(&obj, &member.prop)
            }

            // Conditional (ternary)
            Expr::Cond(cond) => {
                let test = self.eval_expr(&cond.test);
                match test.is_truthy() {
                    Some(true) => self.eval_expr(&cond.cons),
                    Some(false) => self.eval_expr(&cond.alt),
                    None => {
                        // Dynamic condition - EXPLORE BOTH PATHS
                        if self.debug {
                            eprintln!("[step {}] Dynamic ternary condition: {}", self.steps, test);
                        }

                        // Save environment
                        let saved_env = clone_env(&self.env);

                        // Evaluate true branch
                        let cons_val = self.eval_expr(&cond.cons);
                        let true_env = clone_env(&self.env);

                        // Restore and evaluate false branch
                        self.env = saved_env;
                        let alt_val = self.eval_expr(&cond.alt);

                        // Merge environments (take the union, mark conflicting as dynamic)
                        self.merge_envs(&true_env);

                        // Build the actual ternary expression for residual
                        let cond_expr = Expr::Cond(CondExpr {
                            span: cond.span,
                            test: Box::new(test.to_expr()),
                            cons: Box::new(cons_val.to_expr()),
                            alt: Box::new(alt_val.to_expr()),
                        });
                        AbstractValue::dynamic_with_expr("cond_result", cond_expr)
                    }
                }
            }

            // Function call
            Expr::Call(call) => {
                self.eval_call(call)
            }

            // New expression
            Expr::New(new) => {
                self.eval_new(new)
            }

            // Sequence expression
            Expr::Seq(seq) => {
                let mut result = AbstractValue::known_undefined();
                for expr in &seq.exprs {
                    result = self.eval_expr(expr);
                }
                result
            }

            // Parenthesized
            Expr::Paren(paren) => self.eval_expr(&paren.expr),

            // Function expression
            Expr::Fn(fn_expr) => {
                let params: Vec<String> = fn_expr
                    .function
                    .params
                    .iter()
                    .filter_map(|p| {
                        if let Pat::Ident(id) = &p.pat {
                            Some(id.sym.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                if let Some(body) = &fn_expr.function.body {
                    AbstractValue::Known(JsValue::Function(FunctionValue::Known {
                        params,
                        body: body.clone(),
                    }))
                } else {
                    AbstractValue::dynamic("empty_fn")
                }
            }

            // Arrow function
            Expr::Arrow(arrow) => {
                let params: Vec<String> = arrow
                    .params
                    .iter()
                    .filter_map(|p| {
                        if let Pat::Ident(id) = p {
                            Some(id.sym.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                match &*arrow.body {
                    BlockStmtOrExpr::BlockStmt(body) => {
                        AbstractValue::Known(JsValue::Function(FunctionValue::Known {
                            params,
                            body: body.clone(),
                        }))
                    }
                    BlockStmtOrExpr::Expr(expr) => {
                        let body = BlockStmt {
                            span: Default::default(),
                            ctxt: Default::default(),
                            stmts: vec![Stmt::Return(ReturnStmt {
                                span: Default::default(),
                                arg: Some(expr.clone()),
                            })],
                        };
                        AbstractValue::Known(JsValue::Function(FunctionValue::Known {
                            params,
                            body,
                        }))
                    }
                }
            }

            // This
            Expr::This(_) => AbstractValue::dynamic("this"),

            _ => {
                AbstractValue::Top
            }
        }
    }

    /// Merge another environment into current (for path joining)
    fn merge_envs(&mut self, other: &Env) {
        let other_borrowed = other.borrow();
        let mut current = self.env.borrow_mut();

        for (name, other_val) in other_borrowed.iter() {
            if let Some(current_val) = current.get(name) {
                // If values differ, mark as dynamic
                if !values_equal(current_val, other_val) {
                    current.insert(name.clone(), AbstractValue::dynamic(name));
                }
            } else {
                current.insert(name.clone(), other_val.clone());
            }
        }
    }

    /// Evaluate a literal
    fn eval_lit(&self, lit: &Lit) -> AbstractValue {
        match lit {
            Lit::Num(n) => AbstractValue::known_number(n.value),
            Lit::Str(s) => AbstractValue::known_string(s.value.to_string()),
            Lit::Bool(b) => AbstractValue::known_bool(b.value),
            Lit::Null(_) => AbstractValue::known_null(),
            _ => AbstractValue::Top,
        }
    }

    /// Evaluate a binary operation
    fn eval_binop(&self, op: BinaryOp, left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
        match op {
            BinaryOp::Add => left.add(right),
            BinaryOp::Sub => left.sub(right),
            BinaryOp::Mul => left.mul(right),
            BinaryOp::Div => left.div(right),
            BinaryOp::Mod => left.rem(right),
            BinaryOp::BitAnd => left.bitand(right),
            BinaryOp::BitOr => left.bitor(right),
            BinaryOp::BitXor => left.bitxor(right),
            BinaryOp::LShift => left.lshift(right),
            BinaryOp::RShift => left.rshift(right),
            BinaryOp::Lt => left.lt(right),
            BinaryOp::LtEq => left.le(right),
            BinaryOp::Gt => left.gt(right),
            BinaryOp::GtEq => left.ge(right),
            BinaryOp::EqEq => left.eq(right),
            BinaryOp::EqEqEq => left.strict_eq(right),
            BinaryOp::NotEq => left.neq(right),
            BinaryOp::NotEqEq => left.neq(right),
            _ => AbstractValue::dynamic("binop_result"),
        }
    }

    /// Evaluate a unary operation
    fn eval_unop(&self, op: UnaryOp, arg: &AbstractValue) -> AbstractValue {
        match op {
            UnaryOp::Bang => arg.not(),
            UnaryOp::Minus => arg.neg(),
            UnaryOp::Tilde => arg.bitnot(),
            UnaryOp::Plus => {
                // +x converts to number
                match arg {
                    AbstractValue::Known(JsValue::Number(n)) => AbstractValue::known_number(*n),
                    AbstractValue::Known(JsValue::String(s)) => {
                        if let Ok(n) = s.parse::<f64>() {
                            AbstractValue::known_number(n)
                        } else {
                            AbstractValue::known_number(f64::NAN)
                        }
                    }
                    AbstractValue::Known(JsValue::Bool(b)) => {
                        AbstractValue::known_number(if *b { 1.0 } else { 0.0 })
                    }
                    _ => AbstractValue::dynamic("unary_plus"),
                }
            }
            UnaryOp::TypeOf => {
                match arg {
                    AbstractValue::Known(JsValue::Undefined) => AbstractValue::known_string("undefined".to_string()),
                    AbstractValue::Known(JsValue::Null) => AbstractValue::known_string("object".to_string()),
                    AbstractValue::Known(JsValue::Bool(_)) => AbstractValue::known_string("boolean".to_string()),
                    AbstractValue::Known(JsValue::Number(_)) => AbstractValue::known_string("number".to_string()),
                    AbstractValue::Known(JsValue::String(_)) => AbstractValue::known_string("string".to_string()),
                    AbstractValue::Known(JsValue::Function(_)) => AbstractValue::known_string("function".to_string()),
                    AbstractValue::Known(JsValue::Object(_)) => AbstractValue::known_string("object".to_string()),
                    AbstractValue::Known(JsValue::Array(_)) => AbstractValue::known_string("object".to_string()),
                    _ => AbstractValue::dynamic("typeof_result"),
                }
            }
            _ => AbstractValue::dynamic("unop_result"),
        }
    }

    /// Evaluate an assignment
    fn eval_assign(&mut self, target: &AssignTarget, op: AssignOp, value: AbstractValue) -> AbstractValue {
        match target {
            AssignTarget::Simple(SimpleAssignTarget::Ident(id)) => {
                let name = id.id.sym.to_string();
                let final_value = if op == AssignOp::Assign {
                    value
                } else {
                    let current = self.get_var(&name);
                    match op {
                        AssignOp::AddAssign => current.add(&value),
                        AssignOp::SubAssign => current.sub(&value),
                        AssignOp::MulAssign => current.mul(&value),
                        AssignOp::DivAssign => current.div(&value),
                        AssignOp::ModAssign => current.rem(&value),
                        AssignOp::BitAndAssign => current.bitand(&value),
                        AssignOp::BitOrAssign => current.bitor(&value),
                        AssignOp::BitXorAssign => current.bitxor(&value),
                        AssignOp::LShiftAssign => current.lshift(&value),
                        AssignOp::RShiftAssign => current.rshift(&value),
                        _ => value,
                    }
                };
                self.debug(&format!("set {} = {}", name, final_value));
                self.set_var(&name, final_value.clone());
                final_value
            }
            AssignTarget::Simple(SimpleAssignTarget::Member(member)) => {
                // Check if this is a closure variable that should emit residual
                let (is_closure_var, var_name) = if let Expr::Ident(id) = member.obj.as_ref() {
                    let name = id.sym.to_string();
                    (self.callback_depth > 0 && self.closure_vars.contains(&name), name)
                } else {
                    (false, String::new())
                };

                let obj = self.eval_expr(&member.obj);

                // Evaluate the index ONCE (before any use) to avoid double evaluation
                let evaluated_idx = match &member.prop {
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        Some(self.eval_expr(expr))
                    }
                    _ => None,
                };

                // Create a new MemberProp with the evaluated index
                let eval_prop = match (&member.prop, &evaluated_idx) {
                    (MemberProp::Computed(_), Some(idx)) => {
                        MemberProp::Computed(ComputedPropName {
                            span: Default::default(),
                            expr: Box::new(idx.to_expr()),
                        })
                    }
                    _ => member.prop.clone(),
                };

                let final_value = if op == AssignOp::Assign {
                    value
                } else {
                    // Compound assignment - need to read current value first
                    let current = self.eval_member_access(&obj, &eval_prop);
                    match op {
                        AssignOp::AddAssign => current.add(&value),
                        AssignOp::SubAssign => current.sub(&value),
                        AssignOp::MulAssign => current.mul(&value),
                        AssignOp::DivAssign => current.div(&value),
                        AssignOp::ModAssign => current.rem(&value),
                        AssignOp::BitAndAssign => current.bitand(&value),
                        AssignOp::BitOrAssign => current.bitor(&value),
                        AssignOp::BitXorAssign => current.bitxor(&value),
                        AssignOp::LShiftAssign => current.lshift(&value),
                        AssignOp::RShiftAssign => current.rshift(&value),
                        _ => value,
                    }
                };

                // For closure variables, emit residual even though we can evaluate the assignment
                // Skip undefined assignments - they're no-ops and just noise in residual
                if is_closure_var && !matches!(&final_value, AbstractValue::Known(JsValue::Undefined)) {
                    // Build the member assignment (using already-evaluated index)
                    let assign_expr = Expr::Assign(AssignExpr {
                        span: Default::default(),
                        op: AssignOp::Assign,
                        left: AssignTarget::Simple(SimpleAssignTarget::Member(MemberExpr {
                            span: Default::default(),
                            obj: Box::new(emit::ident(&var_name)),
                            prop: eval_prop.clone(),
                        })),
                        right: Box::new(final_value.to_expr()),
                    });

                    self.residual.push(emit::expr_stmt(assign_expr));
                }

                self.eval_member_set(&obj, &eval_prop, final_value.clone());
                final_value
            }
            _ => {
                AbstractValue::Top
            }
        }
    }

    /// Evaluate an update expression (++x, x++, etc.)
    fn eval_update(&mut self, update: &UpdateExpr) -> AbstractValue {
        if let Expr::Ident(id) = update.arg.as_ref() {
            let name = id.sym.to_string();
            let current = self.get_var(&name);

            let one = AbstractValue::known_number(1.0);
            let new_value = if update.op == UpdateOp::PlusPlus {
                current.add(&one)
            } else {
                current.sub(&one)
            };

            self.set_var(&name, new_value.clone());

            if update.prefix {
                new_value
            } else {
                current
            }
        } else if let Expr::Member(member) = update.arg.as_ref() {
            // Handle obj[idx]++ etc
            let obj = self.eval_expr(&member.obj);
            let current = self.eval_member_access(&obj, &member.prop);

            let one = AbstractValue::known_number(1.0);
            let new_value = if update.op == UpdateOp::PlusPlus {
                current.add(&one)
            } else {
                current.sub(&one)
            };

            self.eval_member_set(&obj, &member.prop, new_value.clone());

            if update.prefix {
                new_value
            } else {
                current
            }
        } else {
            AbstractValue::Top
        }
    }

    /// Evaluate member access
    fn eval_member_access(&mut self, obj: &AbstractValue, prop: &MemberProp) -> AbstractValue {
        match obj {
            AbstractValue::Known(JsValue::Array(arr)) => {
                match prop {
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        let index = self.eval_expr(expr);
                        if let Some(n) = index.as_number() {
                            let idx = n as usize;
                            let arr = arr.borrow();
                            if idx < arr.len() {
                                let result = arr[idx].clone();
                                return result;
                            }
                        }
                        AbstractValue::known_undefined()
                    }
                    MemberProp::Ident(id) if id.sym.as_ref() == "length" => {
                        AbstractValue::known_number(arr.borrow().len() as f64)
                    }
                    MemberProp::Ident(id) if id.sym.as_ref() == "push" => {
                        // Return a function that pushes
                        AbstractValue::Known(JsValue::Function(FunctionValue::Opaque("Array.push".to_string())))
                    }
                    MemberProp::Ident(id) if id.sym.as_ref() == "pop" => {
                        AbstractValue::Known(JsValue::Function(FunctionValue::Opaque("Array.pop".to_string())))
                    }
                    MemberProp::Ident(id) if id.sym.as_ref() == "slice" => {
                        AbstractValue::Known(JsValue::Function(FunctionValue::Opaque("Array.slice".to_string())))
                    }
                    MemberProp::Ident(id) if id.sym.as_ref() == "unshift" => {
                        AbstractValue::Known(JsValue::Function(FunctionValue::Opaque("Array.unshift".to_string())))
                    }
                    _ => AbstractValue::dynamic("arr_prop"),
                }
            }
            AbstractValue::Known(JsValue::Object(obj_map)) => {
                let key = match prop {
                    MemberProp::Ident(id) => id.sym.to_string(),
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        let k = self.eval_expr(expr);
                        if let Some(s) = k.as_string() {
                            s.clone()
                        } else if let Some(n) = k.as_number() {
                            n.to_string()
                        } else {
                            return AbstractValue::dynamic("obj_computed");
                        }
                    }
                    _ => return AbstractValue::dynamic("obj_prop"),
                };

                let obj = obj_map.borrow();
                obj.get(&key)
                    .cloned()
                    .unwrap_or(AbstractValue::known_undefined())
            }
            AbstractValue::Known(JsValue::String(s)) => {
                match prop {
                    MemberProp::Ident(id) if id.sym.as_ref() == "length" => {
                        AbstractValue::known_number(s.len() as f64)
                    }
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        let index = self.eval_expr(expr);
                        if let Some(n) = index.as_number() {
                            let idx = n as usize;
                            if idx < s.len() {
                                return AbstractValue::known_string(s.chars().nth(idx).unwrap().to_string());
                            }
                        }
                        AbstractValue::known_undefined()
                    }
                    _ => AbstractValue::dynamic("str_prop"),
                }
            }
            // Dynamic object - build the member access expression for residual
            AbstractValue::Dynamic(name, obj_expr) => {
                if self.debug {
                    eprintln!("[step {}] Dynamic member access on: {}", self.steps, obj);
                }

                // Build the member access AST
                let obj_ast = if let Some(e) = obj_expr {
                    *e.clone()
                } else {
                    emit::ident(name)
                };

                let member_expr = match prop {
                    MemberProp::Ident(id) => {
                        emit::member(obj_ast, &id.sym)
                    }
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        let idx = self.eval_expr(expr);
                        Expr::Member(MemberExpr {
                            span: Default::default(),
                            obj: Box::new(obj_ast),
                            prop: MemberProp::Computed(ComputedPropName {
                                span: Default::default(),
                                expr: Box::new(idx.to_expr()),
                            }),
                        })
                    }
                    _ => emit::ident(&format!("{}_prop", name)),
                };

                AbstractValue::dynamic_with_expr("member_access", member_expr)
            }
            // Catch-all for other known types (Undefined, Null, Bool, Number, Function)
            // that don't have meaningful member access
            _ => {
                AbstractValue::dynamic("member_access")
            }
        }
    }

    /// Set a member value
    fn eval_member_set(&mut self, obj: &AbstractValue, prop: &MemberProp, value: AbstractValue) {
        match obj {
            AbstractValue::Known(JsValue::Array(arr)) => {
                let index = match prop {
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        let idx = self.eval_expr(expr);
                        idx.as_number().map(|n| n as usize)
                    }
                    _ => None,
                };

                if let Some(idx) = index {
                    let mut arr = arr.borrow_mut();
                    while arr.len() <= idx {
                        arr.push(AbstractValue::known_undefined());
                    }
                    arr[idx] = value;
                }
            }
            AbstractValue::Known(JsValue::Object(obj_map)) => {
                let key = match prop {
                    MemberProp::Ident(id) => Some(id.sym.to_string()),
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        let k = self.eval_expr(expr);
                        if let Some(s) = k.as_string() {
                            Some(s.clone())
                        } else if let Some(n) = k.as_number() {
                            Some(n.to_string())
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(key) = key {
                    obj_map.borrow_mut().insert(key, value);
                }
            }
            // Dynamic object - only emit residual if it's a closure variable
            AbstractValue::Dynamic(name, obj_expr) => {
                // Only emit residual for closure variables that need to appear in output
                // Skip internal VM state that became dynamic after path splitting
                if self.callback_depth > 0 && !self.closure_vars.contains(name) {
                    // This is internal VM state - don't emit residual
                    return;
                }

                // Skip residual for undefined assignments - they're no-ops and just noise
                if let AbstractValue::Known(JsValue::Undefined) = &value {
                    return;
                }

                // Build the member access AST
                let obj_ast = if let Some(e) = obj_expr {
                    *e.clone()
                } else {
                    emit::ident(name)
                };

                let member_ast = match prop {
                    MemberProp::Ident(id) => {
                        emit::member(obj_ast, &id.sym)
                    }
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        let idx = self.eval_expr(expr);
                        Expr::Member(MemberExpr {
                            span: Default::default(),
                            obj: Box::new(obj_ast),
                            prop: MemberProp::Computed(ComputedPropName {
                                span: Default::default(),
                                expr: Box::new(idx.to_expr()),
                            }),
                        })
                    }
                    _ => return,
                };

                // Build assignment expression: obj.prop = value
                let assign_expr = Expr::Assign(AssignExpr {
                    span: Default::default(),
                    op: AssignOp::Assign,
                    left: AssignTarget::Simple(SimpleAssignTarget::Member(MemberExpr {
                        span: Default::default(),
                        obj: match &member_ast {
                            Expr::Member(m) => m.obj.clone(),
                            _ => Box::new(member_ast.clone()),
                        },
                        prop: match &member_ast {
                            Expr::Member(m) => m.prop.clone(),
                            _ => MemberProp::Ident(IdentName {
                                span: Default::default(),
                                sym: "unknown".into(),
                            }),
                        },
                    })),
                    right: Box::new(value.to_expr()),
                });

                let stmt = emit::expr_stmt(assign_expr);
                self.residual.push(stmt);
            }
            _ => {}
        }
    }

    /// Evaluate a function call
    fn eval_call(&mut self, call: &CallExpr) -> AbstractValue {
        let callee = match &call.callee {
            Callee::Expr(expr) => expr,
            _ => return AbstractValue::dynamic("super_call"),
        };

        // Check for IIFE: (function() { ... })()
        if let Expr::Paren(paren) = callee.as_ref() {
            if let Expr::Fn(fn_expr) = paren.expr.as_ref() {
                self.debug("Evaluating IIFE");
                return self.eval_iife(fn_expr, &call.args);
            }
        }

        // Direct function expression call
        if let Expr::Fn(fn_expr) = callee.as_ref() {
            self.debug("Evaluating direct function call");
            return self.eval_iife(fn_expr, &call.args);
        }

        // Method call: obj.method(args)
        if let Expr::Member(member) = callee.as_ref() {
            let obj = self.eval_expr(&member.obj);
            if let MemberProp::Ident(method) = &member.prop {
                return self.eval_method_call(&obj, &method.sym, &call.args);
            }
        }

        // Check for special built-in functions by name
        if let Some(name) = expr_as_ident(callee) {
            return self.eval_builtin_call(name, &call.args);
        }

        // General function call
        let callee_val = self.eval_expr(callee);
        let result = self.eval_function_value(&callee_val, &call.args);

        // If the callee was dynamic, emit the call as residual
        if callee_val.is_dynamic() {
            let arg_values: Vec<AbstractValue> = call.args.iter().map(|a| self.eval_expr(&a.expr)).collect();
            let arg_exprs: Vec<Expr> = arg_values.iter().map(|v| v.to_expr()).collect();
            let call_expr = emit::call(callee_val.to_expr(), arg_exprs);
            let stmt = emit::expr_stmt(call_expr);
            self.residual.push(stmt);
        }

        result
    }

    /// Evaluate an IIFE
    fn eval_iife(&mut self, fn_expr: &FnExpr, args: &[ExprOrSpread]) -> AbstractValue {
        if self.call_depth >= self.max_call_depth {
            self.debug("Max call depth reached");
            return AbstractValue::dynamic("max_depth");
        }

        self.call_depth += 1;

        // Bind parameters
        let params: Vec<String> = fn_expr.function.params
            .iter()
            .filter_map(|p| {
                if let Pat::Ident(id) = &p.pat {
                    Some(id.sym.to_string())
                } else {
                    None
                }
            })
            .collect();

        let arg_values: Vec<AbstractValue> = args
            .iter()
            .map(|a| self.eval_expr(&a.expr))
            .collect();

        for (i, param) in params.iter().enumerate() {
            let val = arg_values.get(i).cloned().unwrap_or(AbstractValue::known_undefined());
            self.set_var(param, val);
        }

        // Execute body
        let result = if let Some(body) = &fn_expr.function.body {
            self.eval_block(body).into_value()
        } else {
            AbstractValue::known_undefined()
        };

        self.call_depth -= 1;
        result
    }

    /// Evaluate a function value call
    fn eval_function_value(&mut self, callee: &AbstractValue, args: &[ExprOrSpread]) -> AbstractValue {
        match callee {
            AbstractValue::Known(JsValue::Function(fv)) => {
                match fv {
                    FunctionValue::Known { params, body } => {
                        if self.call_depth >= self.max_call_depth {
                            return AbstractValue::dynamic("max_depth");
                        }

                        self.call_depth += 1;

                        let arg_values: Vec<AbstractValue> = args
                            .iter()
                            .map(|a| self.eval_expr(&a.expr))
                            .collect();

                        for (i, param) in params.iter().enumerate() {
                            let val = arg_values.get(i).cloned().unwrap_or(AbstractValue::known_undefined());
                            self.set_var(param, val);
                        }

                        let result = self.eval_block(body).into_value();
                        self.call_depth -= 1;
                        result
                    }
                    FunctionValue::DispatchHandler(_, n) => {
                        self.debug(&format!("Call to dispatch handler {}", n));
                        AbstractValue::dynamic(&format!("opcode_{}", n))
                    }
                    FunctionValue::Opaque(name) => {
                        self.debug(&format!("Call to opaque function {}", name));
                        AbstractValue::dynamic(&format!("call_{}", name))
                    }
                }
            }
            _ => AbstractValue::dynamic("dyn_call"),
        }
    }

    /// Evaluate a built-in function call
    fn eval_builtin_call(&mut self, name: &str, args: &[ExprOrSpread]) -> AbstractValue {

        match name {
            "Array" => AbstractValue::known_array(vec![]),
            "Object" => AbstractValue::known_object(HashMap::new()),
            "TextDecoder" => {
                self.trace.push(TracedOp::TextDecoderNew);
                AbstractValue::dynamic("TextDecoder")
            }
            _ => {
                // Look up the variable - it might be a function we can call
                let callee_val = self.get_var(name);
                if matches!(&callee_val, AbstractValue::Known(JsValue::Function(_))) {
                    let result = self.eval_function_value(&callee_val, args);
                    result
                } else if let AbstractValue::Dynamic(_, Some(expr)) = &callee_val {
                    // Check if it's a ternary between two functions - we can split!
                    if let Expr::Cond(cond) = expr.as_ref() {
                        let cons_val = self.eval_expr(&cond.cons);
                        let alt_val = self.eval_expr(&cond.alt);

                        if matches!(&cons_val, AbstractValue::Known(JsValue::Function(_)))
                            && matches!(&alt_val, AbstractValue::Known(JsValue::Function(_)))
                        {
                            if self.debug {
                                eprintln!("[step {}] Splitting function call on ternary: cond ? {} : {}",
                                          self.steps, cons_val, alt_val);
                            }

                            // Split execution - evaluate both possibilities
                            let saved_env = clone_env(&self.env);
                            let saved_residual = std::mem::take(&mut self.residual);

                            // TRUE branch: call cons function
                            let true_result = self.eval_function_value(&cons_val, args);
                            let true_residual = std::mem::replace(&mut self.residual, saved_residual.clone());
                            let true_env = clone_env(&self.env);

                            // FALSE branch: call alt function
                            self.env = saved_env;
                            let _ = std::mem::take(&mut self.residual);
                            let false_result = self.eval_function_value(&alt_val, args);
                            let false_residual = std::mem::replace(&mut self.residual, saved_residual);

                            // Merge environments
                            self.merge_envs(&true_env);

                            // Emit residual if statement
                            if !true_residual.is_empty() || !false_residual.is_empty() {
                                let cond_val = self.eval_expr(&cond.test);
                                let if_stmt = Stmt::If(IfStmt {
                                    span: Default::default(),
                                    test: Box::new(cond_val.to_expr()),
                                    cons: Box::new(Stmt::Block(BlockStmt {
                                        span: Default::default(),
                                        ctxt: Default::default(),
                                        stmts: true_residual,
                                    })),
                                    alt: if false_residual.is_empty() {
                                        None
                                    } else {
                                        Some(Box::new(Stmt::Block(BlockStmt {
                                            span: Default::default(),
                                            ctxt: Default::default(),
                                            stmts: false_residual,
                                        })))
                                    },
                                });
                                self.residual.push(if_stmt);
                            }

                            // Return merged result - both branches returned null, so return null
                            // (If they're different, we'd need more sophisticated merging)
                            match (&true_result, &false_result) {
                                (AbstractValue::Known(JsValue::Null), AbstractValue::Known(JsValue::Null)) => {
                                    AbstractValue::known_null()
                                }
                                _ => AbstractValue::dynamic("split_call_result")
                            }
                        } else {
                            if self.debug {
                                eprintln!("[step {}] Unknown function call: {} (value: {})", self.steps, name, callee_val);
                            }
                            AbstractValue::dynamic(&format!("call_{}", name))
                        }
                    } else {
                        if self.debug {
                            eprintln!("[step {}] Unknown function call: {} (value: {})", self.steps, name, callee_val);
                        }
                        AbstractValue::dynamic(&format!("call_{}", name))
                    }
                } else {
                    if self.debug {
                        eprintln!("[step {}] Unknown function call: {} (value: {})", self.steps, name, callee_val);
                    }
                    AbstractValue::dynamic(&format!("call_{}", name))
                }
            }
        }
    }

    /// Evaluate a method call
    fn eval_method_call(&mut self, obj: &AbstractValue, method: &str, args: &[ExprOrSpread]) -> AbstractValue {
        self.debug(&format!("Method call: {}.{}", obj, method));

        // Check for TextDecoder.decode - MILESTONE!
        if method == "decode" {
            if let AbstractValue::Dynamic(name, _) = obj {
                if name == "TextDecoder" || name.contains("TextDecoder") {
                    let arg = args.first().map(|a| self.eval_expr(&a.expr)).unwrap_or(AbstractValue::Top);
                    self.trace.push(TracedOp::TextDecoderDecode(arg.clone()));

                    // Try to actually decode the bytes if they're known
                    if let Some(arr) = arg.as_array() {
                        let bytes: Vec<u8> = arr.borrow().iter().filter_map(|v| {
                            v.as_number().map(|n| n as u8)
                        }).collect();
                        if bytes.len() == arr.borrow().len() {
                            // All bytes are known, decode to string
                            if let Ok(s) = String::from_utf8(bytes.clone()) {
                                if self.debug {
                                    eprintln!("[step {}] TextDecoder decoded: {:?} ({} bytes)", self.steps, s, bytes.len());
                                }
                                return AbstractValue::known_string(s);
                            }
                        }
                    }
                    return AbstractValue::dynamic("decoded_string");
                }
            }
        }

        match (obj, method) {
            (AbstractValue::Known(JsValue::Array(arr)), "push") => {
                for arg in args {
                    let val = self.eval_expr(&arg.expr);
                    arr.borrow_mut().push(val);
                }
                AbstractValue::known_number(arr.borrow().len() as f64)
            }
            (AbstractValue::Known(JsValue::Array(arr)), "pop") => {
                arr.borrow_mut()
                    .pop()
                    .unwrap_or(AbstractValue::known_undefined())
            }
            (AbstractValue::Known(JsValue::Array(arr)), "unshift") => {
                let mut new_arr = Vec::new();
                for arg in args {
                    let val = self.eval_expr(&arg.expr);
                    new_arr.push(val);
                }
                let mut arr = arr.borrow_mut();
                new_arr.append(&mut arr.to_vec());
                *arr = new_arr;
                AbstractValue::known_number(arr.len() as f64)
            }
            (AbstractValue::Known(JsValue::Array(arr)), "slice") => {
                let arr = arr.borrow();
                AbstractValue::known_array(arr.clone())
            }
            (AbstractValue::Known(JsValue::Array(arr)), "length") => {
                AbstractValue::known_number(arr.borrow().len() as f64)
            }
            (AbstractValue::Known(JsValue::Function(FunctionValue::Known { params, body })), "apply") => {
                // fn.apply(thisArg, argsArray)
                if self.debug {
                    eprintln!("[step {}] Function.apply called with params: {:?}", self.steps, params);
                }
                if args.len() >= 2 {
                    let args_array = self.eval_expr(&args[1].expr);
                    if let Some(arr) = args_array.as_array() {
                        let arr = arr.borrow();
                        if self.call_depth >= self.max_call_depth {
                            return AbstractValue::dynamic("max_depth");
                        }
                        self.call_depth += 1;

                        for (i, param) in params.iter().enumerate() {
                            let val = arr.get(i).cloned().unwrap_or(AbstractValue::known_undefined());
                            self.set_var(param, val);
                        }

                        let result = self.eval_block(body).into_value();
                        self.call_depth -= 1;
                        return result;
                    }
                }
                AbstractValue::dynamic("apply_result")
            }
            // Check if the object has this method as a property
            (AbstractValue::Known(JsValue::Object(obj_map)), _) => {
                let method_func = {
                    let obj = obj_map.borrow();
                    obj.get(method).cloned()
                };

                if let Some(func_val) = method_func {
                    if self.debug {
                        eprintln!("[step {}] Found method {} on object, calling it: {}", self.steps, method, func_val);
                    }
                    return self.eval_function_value(&func_val, args);
                }
                AbstractValue::dynamic(&format!("method_{}", method))
            }
            // Dynamic object - emit residual code for the method call
            (AbstractValue::Dynamic(obj_name, obj_expr), _) => {
                // Evaluate arguments, recursively specializing any callback functions
                // This handles functions directly passed as args AND functions inside arrays/objects
                let arg_values: Vec<AbstractValue> = args.iter().map(|a| {
                    let val = self.eval_expr(&a.expr);
                    // Recursively specialize functions in the value
                    self.specialize_value_deep(val)
                }).collect();

                // Build the residual AST
                let obj_ast = if let Some(expr) = obj_expr {
                    *expr.clone()
                } else {
                    emit::ident(obj_name)
                };

                let arg_exprs: Vec<Expr> = arg_values.iter().map(|v| v.to_expr()).collect();
                let call_expr = emit::method_call(obj_ast, method, arg_exprs);

                // Record in trace for debugging
                self.trace.push(TracedOp::MethodCall {
                    object: obj_name.clone(),
                    method: method.to_string(),
                    args: arg_values,
                });

                // Emit as residual statement
                let stmt = emit::expr_stmt(call_expr.clone());
                self.residual.push(stmt);

                AbstractValue::dynamic_with_expr(&format!("{}_{}", obj_name, method), call_expr)
            }
            _ => AbstractValue::dynamic(&format!("method_{}", method)),
        }
    }

    /// Evaluate a new expression
    fn eval_new(&mut self, new: &NewExpr) -> AbstractValue {
        let callee = &new.callee;

        if let Some(name) = expr_as_ident(callee) {
            self.debug(&format!("new {}", name));
            match name {
                "Array" => {
                    let args = new.args.as_ref().map(|a| a.as_slice()).unwrap_or(&[]);
                    if args.is_empty() {
                        return AbstractValue::known_array(vec![]);
                    }
                    if args.len() == 1 {
                        let size = self.eval_expr(&args[0].expr);
                        if let Some(n) = size.as_number() {
                            let n = n as usize;
                            return AbstractValue::known_array(vec![AbstractValue::known_undefined(); n]);
                        }
                    }
                    AbstractValue::known_array(vec![])
                }
                "Object" => AbstractValue::known_object(HashMap::new()),
                "ArrayBuffer" => {
                    let args = new.args.as_ref().map(|a| a.as_slice()).unwrap_or(&[]);
                    if !args.is_empty() {
                        let size = self.eval_expr(&args[0].expr);
                        if let Some(n) = size.as_number() {
                            // Create an array of zeros
                            let n = n as usize;
                            return AbstractValue::known_array(vec![AbstractValue::known_number(0.0); n]);
                        }
                    }
                    AbstractValue::dynamic("ArrayBuffer")
                }
                "Uint8Array" => {
                    let args = new.args.as_ref().map(|a| a.as_slice()).unwrap_or(&[]);
                    if !args.is_empty() {
                        let arg = self.eval_expr(&args[0].expr);
                        // If given an ArrayBuffer (represented as array), wrap it
                        if let Some(arr) = arg.as_array() {
                            return AbstractValue::Known(JsValue::Array(arr));
                        }
                    }
                    AbstractValue::known_array(vec![])
                }
                "DataView" => {
                    let args = new.args.as_ref().map(|a| a.as_slice()).unwrap_or(&[]);
                    if !args.is_empty() {
                        let buffer = self.eval_expr(&args[0].expr);
                        // DataView wraps an ArrayBuffer
                        return buffer;
                    }
                    AbstractValue::dynamic("DataView")
                }
                "TextDecoder" => {
                    self.trace.push(TracedOp::TextDecoderNew);
                    // Return a special value that we can recognize later
                    AbstractValue::dynamic("TextDecoder_instance")
                }
                "Error" => {
                    let args = new.args.as_ref().map(|a| a.as_slice()).unwrap_or(&[]);
                    let msg = if !args.is_empty() {
                        self.eval_expr(&args[0].expr)
                    } else {
                        AbstractValue::known_string("".to_string())
                    };
                    let mut props = HashMap::new();
                    props.insert("message".to_string(), msg);
                    AbstractValue::known_object(props)
                }
                _ => AbstractValue::dynamic(&format!("new_{}", name)),
            }
        } else {
            AbstractValue::dynamic("new_dynamic")
        }
    }

    /// Evaluate a block statement
    pub fn eval_block(&mut self, block: &BlockStmt) -> StmtResult {
        self.eval_stmts(&block.stmts)
    }

    /// Evaluate a slice of statements, recording pending splits for loop-level handling
    fn eval_stmts(&mut self, stmts: &[Stmt]) -> StmtResult {
        let mut result = AbstractValue::known_undefined();

        for stmt in stmts.iter() {
            // Check if this is a ternary assignment pattern
            // Only consider splitting if:
            // 1. We're inside callback specialization (callback_depth > 0)
            // 2. We're not already in a split execution
            if self.callback_depth > 0 && !self.in_split_execution {
                if let Some((target_var, cond)) = self.detect_ternary_assignment_pattern(stmt) {
                    // SAVE environment BEFORE evaluating condition (which may have side effects)
                    let pre_condition_env = clone_env(&self.env);

                    // Evaluate condition to check if it's dynamic
                    // Note: this causes side effects if condition has them
                    let test = self.eval_expr(&cond.test);

                    match test.is_truthy() {
                        None => {
                            // Dynamic condition - RESTORE to pre-condition state
                            // Side effects need to happen fresh in each branch
                            self.env = pre_condition_env;

                            if self.debug {
                                eprintln!(
                                    "[step {}] Recording pending split for {} (storing expressions, env restored)",
                                    self.steps, target_var
                                );
                            }

                            // Store EXPRESSIONS, not evaluated values
                            // Each branch will re-evaluate with proper side effects
                            self.pending_split = Some(PendingSplit {
                                condition_expr: Box::new(cond.test.as_ref().clone()),
                                target_var: target_var.clone(),
                                cons_expr: Box::new(cond.cons.as_ref().clone()),
                                alt_expr: Box::new(cond.alt.as_ref().clone()),
                            });

                            // Assign a dynamic value so the while loop condition is dynamic
                            // (The actual value will be determined by the path split)
                            let cond_expr = Expr::Cond(CondExpr {
                                span: cond.span,
                                test: cond.test.clone(),
                                cons: cond.cons.clone(),
                                alt: cond.alt.clone(),
                            });
                            let dynamic_val = AbstractValue::dynamic_with_expr("cond_result", cond_expr);
                            self.set_var(&target_var, dynamic_val);

                            continue;
                        }
                        Some(true) => {
                            // Known true - side effects already applied, evaluate cons
                            let cons_val = self.eval_expr(&cond.cons);
                            self.set_var(&target_var, cons_val);
                            continue;
                        }
                        Some(false) => {
                            // Known false - side effects already applied, evaluate alt
                            let alt_val = self.eval_expr(&cond.alt);
                            self.set_var(&target_var, alt_val);
                            continue;
                        }
                    }
                }

                // Also detect: if (cond) { target = A; } else { target = B; }
                // This is the same split pattern but written as an if-statement.
                if let Some((target_var, test_expr, cons_rhs, alt_rhs)) = self.detect_if_assignment_pattern(stmt) {
                    let pre_condition_env = clone_env(&self.env);
                    let test = self.eval_expr(test_expr);

                    match test.is_truthy() {
                        None => {
                            // Dynamic condition - restore env to before condition evaluation
                            self.env = pre_condition_env;

                            // Evaluate both branch RHS NOW while variables (like pc) are still known.
                            // The expressions may reference variables that will become dynamic
                            // after the split, so we must resolve them here.
                            let cons_val = self.eval_expr(cons_rhs);
                            let alt_val = self.eval_expr(alt_rhs);

                            if self.debug {
                                eprintln!(
                                    "[step {}] Recording pending if-split for {} (cons={}, alt={})",
                                    self.steps, target_var, cons_val, alt_val
                                );
                            }

                            // Store pre-evaluated literal expressions
                            self.pending_split = Some(PendingSplit {
                                condition_expr: Box::new(test_expr.clone()),
                                target_var: target_var.clone(),
                                cons_expr: Box::new(cons_val.to_expr()),
                                alt_expr: Box::new(alt_val.to_expr()),
                            });

                            let cond_expr = Expr::Cond(CondExpr {
                                span: Default::default(),
                                test: Box::new(test_expr.clone()),
                                cons: Box::new(cons_val.to_expr()),
                                alt: Box::new(alt_val.to_expr()),
                            });
                            let dynamic_val = AbstractValue::dynamic_with_expr("cond_result", cond_expr);
                            self.set_var(&target_var, dynamic_val);

                            continue;
                        }
                        Some(true) => {
                            let cons_val = self.eval_expr(cons_rhs);
                            self.set_var(&target_var, cons_val);
                            continue;
                        }
                        Some(false) => {
                            let alt_val = self.eval_expr(alt_rhs);
                            self.set_var(&target_var, alt_val);
                            continue;
                        }
                    }
                }
            }

            // Normal execution
            match self.eval_stmt(stmt) {
                StmtResult::Value(val) => result = val,
                StmtResult::Return(val) => return StmtResult::Return(val),
                StmtResult::Break => return StmtResult::Break,
                StmtResult::Continue => return StmtResult::Continue,
                StmtResult::Throw(val) => return StmtResult::Throw(val),
                StmtResult::None => {}
            }
        }

        StmtResult::Value(result)
    }

    /// Detect if a statement is a ternary assignment pattern (without evaluating)
    /// Returns (target_var, ternary) if it matches the pattern
    fn detect_ternary_assignment_pattern<'a>(
        &self,
        stmt: &'a Stmt,
    ) -> Option<(String, &'a CondExpr)> {
        // Must be an expression statement
        let expr_stmt = match stmt {
            Stmt::Expr(e) => e,
            _ => return None,
        };

        // Must be a simple assignment
        let assign = match expr_stmt.expr.as_ref() {
            Expr::Assign(a) if a.op == AssignOp::Assign => a,
            _ => return None,
        };

        // Target must be a simple identifier
        let target_var = match &assign.left {
            AssignTarget::Simple(SimpleAssignTarget::Ident(id)) => id.id.sym.to_string(),
            _ => return None,
        };

        // RHS must be a ternary
        let cond = match assign.right.as_ref() {
            Expr::Cond(c) => c,
            _ => return None,
        };

        Some((target_var, cond))
    }

    /// Detect if a statement is an if-assignment pattern:
    ///   if (cond) { target = A; } else { target = B; }
    /// This is semantically equivalent to: target = cond ? A : B
    /// Returns (target_var, test_expr, cons_rhs_expr, alt_rhs_expr)
    fn detect_if_assignment_pattern<'a>(
        &self,
        stmt: &'a Stmt,
    ) -> Option<(String, &'a Expr, &'a Expr, &'a Expr)> {
        let if_stmt = match stmt {
            Stmt::If(i) => i,
            _ => return None,
        };

        let alt = match &if_stmt.alt {
            Some(alt) => alt,
            None => return None,
        };

        let (cons_target, cons_rhs) = extract_single_assignment(&if_stmt.cons)?;
        let (alt_target, alt_rhs) = extract_single_assignment(alt)?;

        if cons_target != alt_target {
            return None;
        }

        Some((cons_target, &if_stmt.test, cons_rhs, alt_rhs))
    }

    /// Evaluate a statement
    pub fn eval_stmt(&mut self, stmt: &Stmt) -> StmtResult {
        if !self.step() {
            return StmtResult::None;
        }

        match stmt {
            Stmt::Expr(expr_stmt) => {
                let val = self.eval_expr(&expr_stmt.expr);
                StmtResult::Value(val)
            }

            Stmt::Decl(decl) => {
                self.eval_decl(decl);
                StmtResult::None
            }

            Stmt::Return(ret) => {
                let value = ret
                    .arg
                    .as_ref()
                    .map(|e| self.eval_expr(e))
                    .unwrap_or(AbstractValue::known_undefined());
                StmtResult::Return(value)
            }

            Stmt::If(if_stmt) => {
                self.eval_if(if_stmt)
            }

            Stmt::While(while_stmt) => {
                self.eval_while(while_stmt)
            }

            Stmt::For(for_stmt) => {
                self.eval_for(for_stmt)
            }

            Stmt::ForIn(for_in) => {
                self.eval_for_in(for_in)
            }

            Stmt::Block(block) => {
                self.eval_block(block)
            }

            Stmt::Switch(switch) => {
                self.eval_switch(switch)
            }

            Stmt::Break(_) => StmtResult::Break,
            Stmt::Continue(_) => StmtResult::Continue,

            Stmt::Throw(throw) => {
                let value = self.eval_expr(&throw.arg);
                self.trace.push(TracedOp::Throw(value.clone()));
                StmtResult::Throw(value)
            }

            Stmt::Try(try_stmt) => {
                self.eval_try(try_stmt)
            }

            Stmt::Empty(_) => StmtResult::None,

            _ => StmtResult::None,
        }
    }

    /// Evaluate a declaration
    fn eval_decl(&mut self, decl: &Decl) {
        match decl {
            Decl::Var(var_decl) => {
                for decl in &var_decl.decls {
                    self.eval_var_declarator(decl);
                }
            }
            Decl::Fn(fn_decl) => {
                let name = fn_decl.ident.sym.to_string();
                let params: Vec<String> = fn_decl
                    .function
                    .params
                    .iter()
                    .filter_map(|p| {
                        if let Pat::Ident(id) = &p.pat {
                            Some(id.sym.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                if let Some(body) = &fn_decl.function.body {
                    let func = AbstractValue::Known(JsValue::Function(FunctionValue::Known {
                        params,
                        body: body.clone(),
                    }));
                    self.set_var(&name, func);
                }
            }
            _ => {}
        }
    }

    /// Evaluate a variable declarator
    fn eval_var_declarator(&mut self, decl: &VarDeclarator) {
        match &decl.name {
            Pat::Ident(id) => {
                let value = decl
                    .init
                    .as_ref()
                    .map(|e| self.eval_expr(e))
                    .unwrap_or(AbstractValue::known_undefined());
                self.set_var(&id.sym, value);
            }
            Pat::Array(arr) => {
                // Destructuring: var [a, b] = expr
                let value = decl
                    .init
                    .as_ref()
                    .map(|e| self.eval_expr(e))
                    .unwrap_or(AbstractValue::known_undefined());

                if let Some(arr_val) = value.as_array() {
                    let arr_val = arr_val.borrow();
                    for (i, elem) in arr.elems.iter().enumerate() {
                        if let Some(pat) = elem {
                            if let Pat::Ident(id) = pat {
                                let val = arr_val.get(i).cloned().unwrap_or(AbstractValue::known_undefined());
                                self.set_var(&id.sym, val);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Evaluate an if statement - EXPLORE BOTH PATHS if dynamic
    fn eval_if(&mut self, if_stmt: &IfStmt) -> StmtResult {
        let test = self.eval_expr(&if_stmt.test);

        match test.is_truthy() {
            Some(true) => self.eval_stmt(&if_stmt.cons),
            Some(false) => {
                if let Some(alt) = &if_stmt.alt {
                    self.eval_stmt(alt)
                } else {
                    StmtResult::None
                }
            }
            None => {
                // Dynamic condition - EXPLORE BOTH PATHS and emit residual if statement
                if self.debug {
                    eprintln!("[step {}] Dynamic if condition: {}", self.steps, test);
                }

                // Save environment
                let saved_env = clone_env(&self.env);

                // Collect residual from TRUE branch
                let saved_residual = std::mem::take(&mut self.residual);
                let true_result = self.eval_stmt(&if_stmt.cons);
                let true_residual = std::mem::replace(&mut self.residual, saved_residual);
                let true_env = clone_env(&self.env);

                // Restore env, collect residual from FALSE branch
                self.env = saved_env;
                let saved_residual = std::mem::take(&mut self.residual);
                let (false_result, false_residual) = if let Some(alt) = &if_stmt.alt {
                    let result = self.eval_stmt(alt);
                    let residual = std::mem::replace(&mut self.residual, saved_residual);
                    (result, residual)
                } else {
                    let _ = std::mem::replace(&mut self.residual, saved_residual);
                    (StmtResult::None, vec![])
                };

                // Merge environments
                self.merge_envs(&true_env);

                // Build and emit residual if statement (if either branch has residual)
                if !true_residual.is_empty() || !false_residual.is_empty() {
                    let residual_if = Stmt::If(IfStmt {
                        span: if_stmt.span,
                        test: Box::new(test.to_expr()),
                        cons: Box::new(Stmt::Block(BlockStmt {
                            span: Default::default(),
                            ctxt: Default::default(),
                            stmts: true_residual,
                        })),
                        alt: if false_residual.is_empty() {
                            None
                        } else {
                            Some(Box::new(Stmt::Block(BlockStmt {
                                span: Default::default(),
                                ctxt: Default::default(),
                                stmts: false_residual,
                            })))
                        },
                    });
                    self.residual.push(residual_if);
                }

                // Return based on results
                match (true_result, false_result) {
                    (StmtResult::Return(v1), StmtResult::Return(v2)) => {
                        // Both return - check if they're both known functions
                        // If so, we can potentially split on the call site
                        match (&v1, &v2) {
                            (AbstractValue::Known(JsValue::Function(_)),
                             AbstractValue::Known(JsValue::Function(_))) => {
                                // Both are known functions - create a dynamic value
                                // but store the condition for later splitting
                                // For now, build a ternary expression
                                let cond_expr = test.to_expr();
                                let ternary = Expr::Cond(CondExpr {
                                    span: Default::default(),
                                    test: Box::new(cond_expr),
                                    cons: Box::new(v1.to_expr()),
                                    alt: Box::new(v2.to_expr()),
                                });
                                StmtResult::Return(AbstractValue::dynamic_with_expr("if_return", ternary))
                            }
                            _ => {
                                // General case - just mark as dynamic
                                StmtResult::Return(AbstractValue::dynamic("if_return"))
                            }
                        }
                    }
                    (StmtResult::Return(_), _) | (_, StmtResult::Return(_)) => {
                        // One returns - could return
                        StmtResult::Value(AbstractValue::dynamic("maybe_return"))
                    }
                    _ => StmtResult::None
                }
            }
        }
    }

    /// Evaluate a while loop
    fn eval_while(&mut self, while_stmt: &WhileStmt) -> StmtResult {
        let loop_id = self.next_loop_id;
        self.next_loop_id += 1;

        let max_iterations = 10_000_000; // Per loop limit

        loop {
            let iteration = {
                let count = self.loop_counts.entry(loop_id).or_insert(0);
                *count += 1;
                *count
            };

            if iteration > max_iterations {
                self.debug(&format!("Loop {} hit max iterations", loop_id));
                break;
            }

            if self.should_stop() {
                break;
            }

            let test = self.eval_expr(&while_stmt.test);
            if self.debug && iteration <= 5 {
                eprintln!("[step {}] While loop {} iteration {}, test = {}", self.steps, loop_id, iteration, test);
            }
            match test.is_truthy() {
                Some(true) => {
                    match self.eval_stmt(&while_stmt.body) {
                        StmtResult::Break => break,
                        StmtResult::Return(v) => return StmtResult::Return(v),
                        StmtResult::Continue => continue,
                        _ => {}
                    }
                }
                Some(false) => break,
                None => {
                    // Dynamic condition - can't determine at partial eval time
                    if self.debug {
                        eprintln!("[step {}] Dynamic while condition in loop {}: {}", self.steps, loop_id, test);
                    }

                    // Check if we have a pending split that caused this
                    if let Some(split) = self.pending_split.take() {
                        if self.debug {
                            eprintln!(
                                "[step {}] Handling pending split for {} at while loop level",
                                self.steps, split.target_var
                            );
                        }

                        // Save state before split (this is BEFORE condition side effects)
                        let saved_env = clone_env(&self.env);
                        let saved_residual = std::mem::take(&mut self.residual);
                        let saved_in_split = self.in_split_execution;

                        // Mark that we're in a split execution to prevent nested splits
                        self.in_split_execution = true;

                        // === TRUE BRANCH ===
                        // Re-evaluate condition (with side effects) and cons expression
                        let cond_result = self.eval_expr(&split.condition_expr);
                        let cons_val = self.eval_expr(&split.cons_expr);
                        self.set_var(&split.target_var, cons_val);
                        // Reset loop counter for this branch
                        self.loop_counts.insert(loop_id, 0);
                        // Continue the while loop with the specialized value
                        let true_result = self.eval_while_body_until_done(while_stmt, loop_id, max_iterations);
                        let true_residual = std::mem::replace(&mut self.residual, saved_residual.clone());
                        let true_env = clone_env(&self.env);

                        // === FALSE BRANCH ===
                        // Restore to BEFORE condition side effects
                        self.env = saved_env;
                        let _ = std::mem::take(&mut self.residual);
                        // Re-evaluate condition (with side effects) and alt expression
                        let _ = self.eval_expr(&split.condition_expr);
                        let alt_val = self.eval_expr(&split.alt_expr);
                        self.set_var(&split.target_var, alt_val);
                        // Reset loop counter for this branch
                        self.loop_counts.insert(loop_id, 0);
                        // Continue the while loop with the specialized value
                        let _false_result = self.eval_while_body_until_done(while_stmt, loop_id, max_iterations);
                        let false_residual = std::mem::replace(&mut self.residual, saved_residual);

                        // Restore split execution flag
                        self.in_split_execution = saved_in_split;

                        // Merge environments
                        self.merge_envs(&true_env);

                        // Emit residual if-statement with condition result from true branch
                        if !true_residual.is_empty() || !false_residual.is_empty() {
                            if self.debug {
                                eprintln!(
                                    "[step {}] Emitting split if-statement: true={} stmts, false={} stmts",
                                    self.steps, true_residual.len(), false_residual.len()
                                );
                            }
                            let if_stmt = Stmt::If(IfStmt {
                                span: Default::default(),
                                test: Box::new(cond_result.to_expr()),
                                cons: Box::new(Stmt::Block(BlockStmt {
                                    span: Default::default(),
                                    ctxt: Default::default(),
                                    stmts: true_residual,
                                })),
                                alt: if false_residual.is_empty() {
                                    None
                                } else {
                                    Some(Box::new(Stmt::Block(BlockStmt {
                                        span: Default::default(),
                                        ctxt: Default::default(),
                                        stmts: false_residual,
                                    })))
                                },
                            });
                            self.residual.push(if_stmt);
                        }

                        // Handle return results
                        match true_result {
                            StmtResult::Return(v) => return StmtResult::Return(v),
                            _ => {}
                        }

                        break;
                    }

                    // No pending split - use original behavior
                    // If we're inside a callback being specialized AND not in split execution,
                    // emit residual while loop
                    if self.callback_depth > 0 && !self.in_split_execution {
                        // Explore the loop body to specialize it, then emit residual while loop
                        // Save current residual
                        let saved_residual = std::mem::take(&mut self.residual);

                        // Execute one iteration of the body to collect specialized residual
                        let _body_result = self.eval_stmt(&while_stmt.body);

                        // Collect the specialized body statements
                        let body_residual = std::mem::replace(&mut self.residual, saved_residual);

                        // Emit residual while loop
                        if !body_residual.is_empty() {
                            // Use specialized body
                            let residual_while = Stmt::While(WhileStmt {
                                span: while_stmt.span,
                                test: Box::new(test.to_expr()),
                                body: Box::new(Stmt::Block(BlockStmt {
                                    span: Default::default(),
                                    ctxt: Default::default(),
                                    stmts: body_residual,
                                })),
                            });
                            self.residual.push(residual_while);
                        } else {
                            // Body produced no residual - emit original while statement
                            // This preserves control flow that doesn't touch closure_vars
                            self.residual.push(Stmt::While(while_stmt.clone()));
                        }
                    }
                    break;
                }
            }
        }

        StmtResult::None
    }

    /// Continue executing a while loop body until done (for path splitting)
    /// This is called during split execution, so we don't emit residual while loops here
    fn eval_while_body_until_done(
        &mut self,
        while_stmt: &WhileStmt,
        loop_id: usize,
        max_iterations: usize,
    ) -> StmtResult {
        loop {
            let iteration = {
                let count = self.loop_counts.entry(loop_id).or_insert(0);
                *count += 1;
                *count
            };

            if iteration > max_iterations || self.should_stop() {
                break;
            }

            let test = self.eval_expr(&while_stmt.test);
            if self.debug && iteration <= 5 {
                eprintln!(
                    "[step {}] Split while loop {} iteration {}, test = {}",
                    self.steps, loop_id, iteration, test
                );
            }

            match test.is_truthy() {
                Some(true) => {
                    match self.eval_stmt(&while_stmt.body) {
                        StmtResult::Break => break,
                        StmtResult::Return(v) => return StmtResult::Return(v),
                        StmtResult::Continue => continue,
                        _ => {}
                    }
                }
                Some(false) => break,
                None => {
                    // Still dynamic during split execution
                    // Don't emit nested while loops - just break and let the outer split handle residual
                    if self.debug {
                        eprintln!(
                            "[step {}] Split while loop {} still dynamic (not emitting nested while): {}",
                            self.steps, loop_id, test
                        );
                    }
                    break;
                }
            }
        }

        StmtResult::None
    }

    /// Evaluate a for loop
    fn eval_for(&mut self, for_stmt: &ForStmt) -> StmtResult {
        // Initialize
        if let Some(init) = &for_stmt.init {
            match init {
                VarDeclOrExpr::VarDecl(var_decl) => {
                    self.eval_decl(&Decl::Var(var_decl.clone()));
                }
                VarDeclOrExpr::Expr(expr) => {
                    self.eval_expr(expr);
                }
            }
        }

        let loop_id = self.next_loop_id;
        self.next_loop_id += 1;
        let max_iterations = 100_000;

        loop {
            let iteration = {
                let count = self.loop_counts.entry(loop_id).or_insert(0);
                *count += 1;
                *count
            };

            if iteration > max_iterations || self.should_stop() {
                break;
            }

            // Test
            if let Some(test) = &for_stmt.test {
                let test_val = self.eval_expr(test);
                match test_val.is_truthy() {
                    Some(true) => {}
                    Some(false) => break,
                    None => {
                        self.debug("Dynamic for condition");
                        break;
                    }
                }
            }

            // Body
            match self.eval_stmt(&for_stmt.body) {
                StmtResult::Break => break,
                StmtResult::Return(v) => return StmtResult::Return(v),
                StmtResult::Continue => {}
                _ => {}
            }

            // Update
            if let Some(update) = &for_stmt.update {
                self.eval_expr(update);
            }
        }

        StmtResult::None
    }

    /// Evaluate a for-in loop
    fn eval_for_in(&mut self, for_in: &ForInStmt) -> StmtResult {
        let obj = self.eval_expr(&for_in.right);

        // Get the variable name
        let var_name = match &for_in.left {
            ForHead::VarDecl(var_decl) => {
                if let Some(decl) = var_decl.decls.first() {
                    if let Pat::Ident(id) = &decl.name {
                        Some(id.sym.to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            ForHead::Pat(pat) => {
                if let Pat::Ident(id) = pat.as_ref() {
                    Some(id.sym.to_string())
                } else {
                    None
                }
            }
            _ => None,
        };

        let var_name = match var_name {
            Some(n) => n,
            None => return StmtResult::None,
        };

        // Iterate over object keys
        if let Some(obj_map) = obj.as_object() {
            let keys: Vec<String> = obj_map.borrow().keys().cloned().collect();
            for key in keys {
                self.set_var(&var_name, AbstractValue::known_string(key));
                match self.eval_stmt(&for_in.body) {
                    StmtResult::Break => break,
                    StmtResult::Return(v) => return StmtResult::Return(v),
                    _ => {}
                }
            }
        } else if let Some(arr) = obj.as_array() {
            let len = arr.borrow().len();
            for i in 0..len {
                self.set_var(&var_name, AbstractValue::known_string(i.to_string()));
                match self.eval_stmt(&for_in.body) {
                    StmtResult::Break => break,
                    StmtResult::Return(v) => return StmtResult::Return(v),
                    _ => {}
                }
            }
        }

        StmtResult::None
    }

    /// Evaluate a switch statement
    fn eval_switch(&mut self, switch: &SwitchStmt) -> StmtResult {
        let discriminant = self.eval_expr(&switch.discriminant);

        // If discriminant is known, we can statically select the case
        if let Some(disc_val) = discriminant.as_number() {
            let disc_int = disc_val as i64;

            let mut matched = false;
            let mut fell_through = false;

            for case in &switch.cases {
                if let Some(test) = &case.test {
                    if !matched && !fell_through {
                        if let Some(case_val) = crate::parser::expr_as_number(test) {
                            if case_val as i64 == disc_int {
                                matched = true;
                            }
                        }
                    }
                } else {
                    // Default case
                    if !matched {
                        matched = true;
                    }
                }

                if matched || fell_through {
                    // Use eval_stmts to enable path splitting detection for ternary assignments
                    match self.eval_stmts(&case.cons) {
                        StmtResult::Break => return StmtResult::None,
                        StmtResult::Return(v) => return StmtResult::Return(v),
                        StmtResult::Continue => return StmtResult::Continue,
                        _ => {}
                    }
                    fell_through = true;
                }
            }
        } else {
            // Dynamic discriminant - need to explore all cases
            self.debug("Dynamic switch discriminant - exploring all cases");

            // Limit recursion depth to prevent stack overflow
            if self.call_depth > 50 {
                self.debug("Dynamic switch skipped - call depth too deep");
                return StmtResult::None;
            }

            // Save initial state
            let initial_env = clone_env(&self.env);

            for case in &switch.cases {
                // Restore to initial state for each case
                self.env = clone_env(&initial_env);

                for stmt in &case.cons {
                    match self.eval_stmt(stmt) {
                        StmtResult::Break => break,
                        StmtResult::Return(_) => break,
                        _ => {}
                    }
                }
            }

            // Final state is a merge (simplified - just use last)
        }

        StmtResult::None
    }

    /// Evaluate a try statement
    fn eval_try(&mut self, try_stmt: &TryStmt) -> StmtResult {
        // Evaluate try block
        let result = self.eval_block(&try_stmt.block);

        // If there was a throw, evaluate catch
        match result {
            StmtResult::Throw(thrown_value) => {
                if let Some(handler) = &try_stmt.handler {
                    // Bind the caught value to the parameter
                    if let Some(Pat::Ident(id)) = &handler.param {
                        self.set_var(&id.sym, thrown_value);
                    }

                    let catch_result = self.eval_block(&handler.body);

                    // Evaluate finally if present
                    if let Some(finally) = &try_stmt.finalizer {
                        let _ = self.eval_block(finally);
                    }

                    catch_result
                } else {
                    // No handler, propagate throw
                    if let Some(finally) = &try_stmt.finalizer {
                        let _ = self.eval_block(finally);
                    }
                    StmtResult::Throw(thrown_value)
                }
            }
            other => {
                // Evaluate finally if present
                if let Some(finally) = &try_stmt.finalizer {
                    let _ = self.eval_block(finally);
                }
                other
            }
        }
    }
}

/// Result of evaluating a statement
pub enum StmtResult {
    None,
    Value(AbstractValue),
    Return(AbstractValue),
    Break,
    Continue,
    Throw(AbstractValue),
}

impl StmtResult {
    /// Extract the value from the result, returning undefined for control flow results
    pub fn into_value(self) -> AbstractValue {
        match self {
            StmtResult::Value(v) => v,
            StmtResult::Return(v) => v,
            StmtResult::Throw(v) => v,
            _ => AbstractValue::known_undefined(),
        }
    }
}

/// Check if statements use the `arguments` object
fn uses_arguments(stmts: &[Stmt]) -> bool {
    use swc_ecma_ast::*;

    fn expr_uses_arguments(expr: &Expr) -> bool {
        match expr {
            Expr::Ident(id) => id.sym.as_ref() == "arguments",
            Expr::Member(m) => expr_uses_arguments(&m.obj),
            Expr::Call(c) => {
                if let Callee::Expr(e) = &c.callee {
                    if expr_uses_arguments(e) {
                        return true;
                    }
                }
                c.args.iter().any(|a| expr_uses_arguments(&a.expr))
            }
            Expr::Bin(b) => expr_uses_arguments(&b.left) || expr_uses_arguments(&b.right),
            Expr::Assign(a) => expr_uses_arguments(&a.right),
            _ => false,
        }
    }

    for stmt in stmts {
        if let Stmt::Expr(expr_stmt) = stmt {
            if expr_uses_arguments(&expr_stmt.expr) {
                return true;
            }
        }
    }
    false
}

/// Extract a numeric value from an expression (handles negative numbers)
fn extract_number(expr: &Expr) -> Option<i64> {
    use swc_ecma_ast::*;

    match expr {
        Expr::Lit(Lit::Num(n)) => Some(n.value as i64),
        Expr::Unary(unary) if unary.op == UnaryOp::Minus => {
            if let Expr::Lit(Lit::Num(n)) = unary.arg.as_ref() {
                Some(-(n.value as i64))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Get the assignment target as a string key for dead store elimination
fn get_assign_target_key(expr: &Expr) -> Option<String> {
    use swc_ecma_ast::*;

    if let Expr::Assign(assign) = expr {
        match &assign.left {
            AssignTarget::Simple(SimpleAssignTarget::Ident(id)) => {
                Some(id.sym.to_string())
            }
            AssignTarget::Simple(SimpleAssignTarget::Member(member)) => {
                // For member access like obj[0], create key "obj[0]"
                let obj_name = if let Expr::Ident(id) = member.obj.as_ref() {
                    id.sym.to_string()
                } else {
                    return None;
                };

                let prop_key = match &member.prop {
                    MemberProp::Computed(ComputedPropName { expr, .. }) => {
                        if let Some(n) = extract_number(expr.as_ref()) {
                            format!("[{}]", n)
                        } else {
                            return None; // Dynamic index, can't eliminate
                        }
                    }
                    MemberProp::Ident(id) => format!(".{}", id.sym),
                    _ => return None,
                };

                Some(format!("{}{}", obj_name, prop_key))
            }
            _ => None,
        }
    } else {
        None
    }
}

/// Simplify an expression (e.g., Date["now"].apply(Date, []) -> Date.now())
pub fn simplify_expr(expr: Expr) -> Expr {
    use swc_ecma_ast::*;

    match expr {
        // Handle assignments - simplify the RHS
        Expr::Assign(mut assign) => {
            assign.right = Box::new(simplify_expr(*assign.right));
            Expr::Assign(assign)
        }
        // Handle binary expressions - simplify both sides
        Expr::Bin(mut bin) => {
            bin.left = Box::new(simplify_expr(*bin.left));
            bin.right = Box::new(simplify_expr(*bin.right));
            Expr::Bin(bin)
        }
        // Simplify X["method"].apply(X, []) -> X.method()
        Expr::Call(call) => {
            if let Callee::Expr(callee_expr) = &call.callee {
                if let Expr::Member(member) = callee_expr.as_ref() {
                    // Check if it's .apply
                    if let MemberProp::Ident(prop) = &member.prop {
                        if prop.sym.as_ref() == "apply" {
                            // Check if the object is X["method"]
                            if let Expr::Member(inner_member) = member.obj.as_ref() {
                                // Get the method name
                                if let MemberProp::Computed(ComputedPropName { expr: method_expr, .. }) = &inner_member.prop {
                                    if let Expr::Lit(Lit::Str(method_name)) = method_expr.as_ref() {
                                        // Check if args are (X, [])
                                        if call.args.len() == 2 {
                                            if let Expr::Array(arr) = call.args[1].expr.as_ref() {
                                                if arr.elems.is_empty() {
                                                    // Transform to X.method()
                                                    return Expr::Call(CallExpr {
                                                        span: call.span,
                                                        ctxt: call.ctxt,
                                                        callee: Callee::Expr(Box::new(Expr::Member(MemberExpr {
                                                            span: Default::default(),
                                                            obj: inner_member.obj.clone(),
                                                            prop: MemberProp::Ident(IdentName {
                                                                span: Default::default(),
                                                                sym: method_name.value.clone(),
                                                            }),
                                                        }))),
                                                        args: vec![],
                                                        type_args: None,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Expr::Call(call)
        }
        // Simplify arguments[arguments.length - 1] -> event (when we add event param)
        Expr::Member(member) => {
            if let Expr::Ident(id) = member.obj.as_ref() {
                if id.sym.as_ref() == "arguments" {
                    // Replace with just "event"
                    return Expr::Ident(Ident {
                        span: Default::default(),
                        ctxt: Default::default(),
                        sym: "event".into(),
                        optional: false,
                    });
                }
            }
            Expr::Member(member)
        }
        other => other,
    }
}

/// Check if an assignment is to a negative array index (VM artifact)
fn is_negative_index_assign(expr: &Expr) -> bool {
    use swc_ecma_ast::*;

    if let Expr::Assign(assign) = expr {
        if let AssignTarget::Simple(SimpleAssignTarget::Member(member)) = &assign.left {
            if let MemberProp::Computed(ComputedPropName { expr, .. }) = &member.prop {
                if let Some(n) = extract_number(expr.as_ref()) {
                    return n < 0;
                }
            }
        }
    }
    false
}

/// Clean up residual statements by removing dead stores and simplifying expressions
pub fn cleanup_residual_stmts(stmts: Vec<Stmt>) -> Vec<Stmt> {
    use std::collections::HashMap;
    use swc_ecma_ast::*;

    // First pass: find which assignments are "dead" (overwritten later)
    let mut last_assign: HashMap<String, usize> = HashMap::new();

    for (i, stmt) in stmts.iter().enumerate() {
        if let Stmt::Expr(expr_stmt) = stmt {
            if let Some(key) = get_assign_target_key(&expr_stmt.expr) {
                last_assign.insert(key, i);
            }
        }
    }

    // Second pass: keep only non-dead stores and simplify expressions
    let mut result = Vec::new();

    for (i, stmt) in stmts.into_iter().enumerate() {
        if let Stmt::Expr(expr_stmt) = &stmt {
            // Check if this is a dead store
            if let Some(key) = get_assign_target_key(&expr_stmt.expr) {
                if let Some(&last_idx) = last_assign.get(&key) {
                    if i < last_idx {
                        // This assignment is overwritten later, skip it
                        continue;
                    }
                }
            }

            // Skip assignments to negative indices (VM artifacts)
            if is_negative_index_assign(&expr_stmt.expr) {
                continue;
            }

            // Check if this is a bare expression with no side effects worth keeping
            // (like just Date.now() with no assignment)
            if let Expr::Call(call) = expr_stmt.expr.as_ref() {
                if get_assign_target_key(&expr_stmt.expr).is_none() {
                    // Check if it's a call to a known side-effecting function
                    let is_side_effecting = if let Callee::Expr(callee) = &call.callee {
                        match callee.as_ref() {
                            // console.log, console.error, etc
                            Expr::Member(m) => {
                                if let Expr::Ident(id) = m.obj.as_ref() {
                                    id.sym.as_ref() == "console" || id.sym.as_ref() == "document"
                                } else {
                                    false
                                }
                            }
                            _ => false,
                        }
                    } else {
                        false
                    };

                    if !is_side_effecting {
                        // Bare function call without side effects - skip it
                        continue;
                    }
                }
            }
        }

        // Simplify the statement
        let simplified = match stmt {
            Stmt::Expr(expr_stmt) => {
                let simplified_expr = simplify_expr(*expr_stmt.expr);
                Stmt::Expr(ExprStmt {
                    span: expr_stmt.span,
                    expr: Box::new(simplified_expr),
                })
            }
            other => other,
        };

        result.push(simplified);
    }

    result
}

/// Check if two abstract values are equal (for merging)
fn values_equal(a: &AbstractValue, b: &AbstractValue) -> bool {
    match (a, b) {
        (AbstractValue::Known(JsValue::Number(n1)), AbstractValue::Known(JsValue::Number(n2))) => {
            (n1 - n2).abs() < f64::EPSILON || (n1.is_nan() && n2.is_nan())
        }
        (AbstractValue::Known(JsValue::String(s1)), AbstractValue::Known(JsValue::String(s2))) => {
            s1 == s2
        }
        (AbstractValue::Known(JsValue::Bool(b1)), AbstractValue::Known(JsValue::Bool(b2))) => {
            b1 == b2
        }
        (AbstractValue::Known(JsValue::Null), AbstractValue::Known(JsValue::Null)) => true,
        (AbstractValue::Known(JsValue::Undefined), AbstractValue::Known(JsValue::Undefined)) => true,
        (AbstractValue::Known(JsValue::Array(arr1)), AbstractValue::Known(JsValue::Array(arr2))) => {
            let arr1 = arr1.borrow();
            let arr2 = arr2.borrow();
            if arr1.len() != arr2.len() {
                return false;
            }
            arr1.iter().zip(arr2.iter()).all(|(a, b)| values_equal(a, b))
        }
        (AbstractValue::Known(JsValue::Object(obj1)), AbstractValue::Known(JsValue::Object(obj2))) => {
            let obj1 = obj1.borrow();
            let obj2 = obj2.borrow();
            if obj1.len() != obj2.len() {
                return false;
            }
            obj1.iter().all(|(k, v1)| {
                obj2.get(k).map_or(false, |v2| values_equal(v1, v2))
            })
        }
        // Functions are equal if they're the same reference (or same params/body)
        (AbstractValue::Known(JsValue::Function(f1)), AbstractValue::Known(JsValue::Function(f2))) => {
            // For simplicity, consider functions equal if both are the same variant
            // This is conservative - they might be "equal" but we won't mark as dynamic
            match (f1, f2) {
                (FunctionValue::Known { params: p1, .. }, FunctionValue::Known { params: p2, .. }) => {
                    // Same number of params is a reasonable proxy for "same function"
                    p1.len() == p2.len()
                }
                (FunctionValue::DispatchHandler(name1, n1), FunctionValue::DispatchHandler(name2, n2)) => name1 == name2 && n1 == n2,
                (FunctionValue::Opaque(n1), FunctionValue::Opaque(n2)) => n1 == n2,
                _ => false,
            }
        }
        (AbstractValue::Dynamic(n1, _), AbstractValue::Dynamic(n2, _)) => n1 == n2,
        (AbstractValue::Top, AbstractValue::Top) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_js;

    #[test]
    fn test_simple_eval() {
        let source = "var x = 1 + 2;";
        let module = parse_js(source).unwrap();

        let mut eval = Evaluator::new(new_env());
        eval.debug = false;
        for item in &module.body {
            if let ModuleItem::Stmt(stmt) = item {
                eval.eval_stmt(stmt);
            }
        }

        let x = eval.get_var("x");
        assert_eq!(x.as_number(), Some(3.0));
    }

    #[test]
    fn test_array_eval() {
        let source = r#"
            var arr = [1, 2, 3];
            arr.push(4);
            var len = arr.length;
        "#;
        let module = parse_js(source).unwrap();

        let mut eval = Evaluator::new(new_env());
        eval.debug = false;
        for item in &module.body {
            if let ModuleItem::Stmt(stmt) = item {
                eval.eval_stmt(stmt);
            }
        }

        let len = eval.get_var("len");
        assert_eq!(len.as_number(), Some(4.0));
    }
}
