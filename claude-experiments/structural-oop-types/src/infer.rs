//! Type inference for expressions
//!
//! This implements Hindley-Milner style type inference extended with:
//! - Row polymorphism for structural object types
//! - Equi-recursive types for self-reference (this)
//! - Let-polymorphism (generalization and instantiation)

use crate::expr::{Expr, ObjectField};
use crate::node::{Node, NodeId};
use crate::store::NodeStore;
use crate::unify::{unify, TypeError};
use std::collections::{HashMap, HashSet};

/// A type scheme: ∀vars. type
///
/// The `vars` are the IDs of type/row variables that are generalized.
/// When instantiating, these are replaced with fresh variables.
#[derive(Debug, Clone)]
pub struct Scheme {
    /// The IDs of generalized type variables
    pub type_vars: HashSet<u32>,
    /// The IDs of generalized row variables
    pub row_vars: HashSet<u32>,
    /// The type (containing references to the generalized variables)
    pub ty: NodeId,
}

impl Scheme {
    /// Create a monomorphic scheme (no generalized variables)
    pub fn mono(ty: NodeId) -> Self {
        Scheme {
            type_vars: HashSet::new(),
            row_vars: HashSet::new(),
            ty,
        }
    }

    /// Create a polymorphic scheme
    pub fn poly(type_vars: HashSet<u32>, row_vars: HashSet<u32>, ty: NodeId) -> Self {
        Scheme { type_vars, row_vars, ty }
    }
}

/// Type environment mapping variable names to their type schemes
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Variable bindings (now storing schemes for let-polymorphism)
    bindings: HashMap<String, Scheme>,
    /// The type of `this` in the current object (if any)
    self_type: Option<NodeId>,
}

impl TypeEnv {
    /// Create an empty environment
    pub fn new() -> Self {
        TypeEnv {
            bindings: HashMap::new(),
            self_type: None,
        }
    }

    /// Extend the environment with a monomorphic binding
    pub fn extend(&self, name: impl Into<String>, ty: NodeId) -> Self {
        let mut new_env = self.clone();
        new_env.bindings.insert(name.into(), Scheme::mono(ty));
        new_env
    }

    /// Extend the environment with a polymorphic scheme
    pub fn extend_scheme(&self, name: impl Into<String>, scheme: Scheme) -> Self {
        let mut new_env = self.clone();
        new_env.bindings.insert(name.into(), scheme);
        new_env
    }

    /// Look up a variable's scheme in the environment
    pub fn lookup(&self, name: &str) -> Option<&Scheme> {
        self.bindings.get(name)
    }

    /// Set the self type for object inference
    pub fn with_self(&self, self_type: NodeId) -> Self {
        let mut new_env = self.clone();
        new_env.self_type = Some(self_type);
        new_env
    }

    /// Get the self type (if inside an object)
    pub fn get_self(&self) -> Option<NodeId> {
        self.self_type
    }

    /// Get all type variable IDs that are free in the environment
    pub fn free_type_vars(&self, store: &NodeStore) -> HashSet<u32> {
        let mut free = HashSet::new();
        for scheme in self.bindings.values() {
            let (ty_vars, _) = free_vars_in_type(store, scheme.ty);
            for var in ty_vars {
                if !scheme.type_vars.contains(&var) {
                    free.insert(var);
                }
            }
        }
        if let Some(self_ty) = self.self_type {
            let (ty_vars, _) = free_vars_in_type(store, self_ty);
            free.extend(ty_vars);
        }
        free
    }

    /// Get all row variable IDs that are free in the environment
    pub fn free_row_vars(&self, store: &NodeStore) -> HashSet<u32> {
        let mut free = HashSet::new();
        for scheme in self.bindings.values() {
            let (_, row_vars) = free_vars_in_type(store, scheme.ty);
            for var in row_vars {
                if !scheme.row_vars.contains(&var) {
                    free.insert(var);
                }
            }
        }
        if let Some(self_ty) = self.self_type {
            let (_, row_vars) = free_vars_in_type(store, self_ty);
            free.extend(row_vars);
        }
        free
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Collect all free type and row variable IDs in a type
/// Returns (type_vars, row_vars)
fn free_vars_in_type(store: &NodeStore, ty: NodeId) -> (HashSet<u32>, HashSet<u32>) {
    let mut type_vars = HashSet::new();
    let mut row_vars = HashSet::new();
    let mut visited = HashSet::new();
    collect_free_vars(store, ty, &mut type_vars, &mut row_vars, &mut visited);
    (type_vars, row_vars)
}

fn collect_free_vars(
    store: &NodeStore,
    ty: NodeId,
    type_vars: &mut HashSet<u32>,
    row_vars: &mut HashSet<u32>,
    visited: &mut HashSet<NodeId>,
) {
    let resolved = store.find(ty);
    if visited.contains(&resolved) {
        return; // Handle cycles (equi-recursive types)
    }
    visited.insert(resolved);

    match store.get(resolved) {
        Node::Var { id, .. } => {
            type_vars.insert(*id);
        }
        Node::RowVar { id, .. } => {
            row_vars.insert(*id);
        }
        Node::Const { .. } => {}
        Node::Arrow { domain, codomain, .. } => {
            collect_free_vars(store, *domain, type_vars, row_vars, visited);
            collect_free_vars(store, *codomain, type_vars, row_vars, visited);
        }
        Node::Record { row, .. } => {
            collect_free_vars(store, *row, type_vars, row_vars, visited);
        }
        Node::RowEmpty { .. } => {}
        Node::RowExtend { presence, rest, .. } => {
            collect_free_vars(store, *presence, type_vars, row_vars, visited);
            collect_free_vars(store, *rest, type_vars, row_vars, visited);
        }
        Node::Present { ty, .. } => {
            collect_free_vars(store, *ty, type_vars, row_vars, visited);
        }
        Node::Absent { .. } => {}
    }
}

/// Generalize a type into a scheme by quantifying over variables not free in the environment
fn generalize(env: &TypeEnv, store: &NodeStore, ty: NodeId) -> Scheme {
    let env_type_free = env.free_type_vars(store);
    let env_row_free = env.free_row_vars(store);
    let (ty_type_vars, ty_row_vars) = free_vars_in_type(store, ty);

    // Quantify over variables that are in ty but not in env
    let type_vars: HashSet<u32> = ty_type_vars.difference(&env_type_free).copied().collect();
    let row_vars: HashSet<u32> = ty_row_vars.difference(&env_row_free).copied().collect();

    Scheme::poly(type_vars, row_vars, ty)
}

/// Instantiate a scheme by replacing quantified variables with fresh ones
fn instantiate(scheme: &Scheme, store: &mut NodeStore) -> NodeId {
    if scheme.type_vars.is_empty() && scheme.row_vars.is_empty() {
        // Monomorphic - no need to copy
        return scheme.ty;
    }

    // Create fresh type variables for each quantified type variable
    let mut type_subst: HashMap<u32, NodeId> = HashMap::new();
    for &var_id in &scheme.type_vars {
        let fresh = store.fresh_var("inst");
        type_subst.insert(var_id, fresh);
    }

    // Create fresh row variables for each quantified row variable
    let mut row_subst: HashMap<u32, NodeId> = HashMap::new();
    for &var_id in &scheme.row_vars {
        let fresh = store.fresh_row_var("inst");
        row_subst.insert(var_id, fresh);
    }

    // Copy the type, substituting quantified variables
    let mut copied: HashMap<NodeId, NodeId> = HashMap::new();
    copy_type(store, scheme.ty, &type_subst, &row_subst, &mut copied)
}

/// Copy a type, substituting variables according to the substitution maps
fn copy_type(
    store: &mut NodeStore,
    ty: NodeId,
    type_subst: &HashMap<u32, NodeId>,
    row_subst: &HashMap<u32, NodeId>,
    copied: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    let resolved = store.find(ty);

    // Check if already copied (handles cycles)
    if let Some(&already) = copied.get(&resolved) {
        return already;
    }

    match store.get(resolved).clone() {
        Node::Var { id, .. } => {
            if let Some(&fresh) = type_subst.get(&id) {
                fresh
            } else {
                resolved // Not quantified, return as-is
            }
        }
        Node::RowVar { id, .. } => {
            if let Some(&fresh) = row_subst.get(&id) {
                fresh
            } else {
                resolved
            }
        }
        Node::Const { .. } => resolved,
        Node::Arrow { domain, codomain, .. } => {
            // Pre-allocate to handle cycles
            let placeholder = store.fresh_var("copy");
            copied.insert(resolved, placeholder);

            let new_domain = copy_type(store, domain, type_subst, row_subst, copied);
            let new_codomain = copy_type(store, codomain, type_subst, row_subst, copied);
            let result = store.arrow(new_domain, new_codomain);

            // Update the placeholder to point to the real result
            store.union(placeholder, result);
            copied.insert(resolved, result);
            result
        }
        Node::Record { row, .. } => {
            let placeholder = store.fresh_var("copy");
            copied.insert(resolved, placeholder);

            let new_row = copy_type(store, row, type_subst, row_subst, copied);
            let result = store.record(new_row);

            store.union(placeholder, result);
            copied.insert(resolved, result);
            result
        }
        Node::RowEmpty { .. } => resolved,
        Node::RowExtend { field, presence, rest, .. } => {
            let placeholder = store.fresh_row_var("copy");
            copied.insert(resolved, placeholder);

            let new_presence = copy_type(store, presence, type_subst, row_subst, copied);
            let new_rest = copy_type(store, rest, type_subst, row_subst, copied);

            // Need to create the row extend node manually
            let result = store.add(Node::row_extend(&field, new_presence, new_rest));

            store.union(placeholder, result);
            copied.insert(resolved, result);
            result
        }
        Node::Present { ty: inner, .. } => {
            let new_inner = copy_type(store, inner, type_subst, row_subst, copied);
            store.present(new_inner)
        }
        Node::Absent { .. } => resolved,
    }
}

/// Inference error
#[derive(Debug, Clone)]
pub enum InferError {
    /// Unification failed
    UnifyError(TypeError),
    /// Undefined variable
    UndefinedVar(String),
    /// `this` used outside of an object
    ThisOutsideObject,
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferError::UnifyError(e) => write!(f, "Type error: {}", e),
            InferError::UndefinedVar(name) => write!(f, "Undefined variable: {}", name),
            InferError::ThisOutsideObject => write!(f, "`this` used outside of an object"),
        }
    }
}

impl std::error::Error for InferError {}

impl From<TypeError> for InferError {
    fn from(e: TypeError) -> Self {
        InferError::UnifyError(e)
    }
}

/// Infer the type of an expression
///
/// Returns the NodeId representing the inferred type.
pub fn infer(env: &TypeEnv, expr: &Expr, store: &mut NodeStore) -> Result<NodeId, InferError> {
    match expr {
        // === Literals ===
        Expr::Bool(_) => Ok(store.bool()),
        Expr::Int(_) => Ok(store.int()),
        Expr::String(_) => Ok(store.string()),

        // === Variables ===
        Expr::Var(name) => {
            let scheme = env
                .lookup(name)
                .ok_or_else(|| InferError::UndefinedVar(name.clone()))?;
            // Instantiate the scheme with fresh type variables
            Ok(instantiate(scheme, store))
        }

        // === Self reference ===
        Expr::This => env.get_self().ok_or(InferError::ThisOutsideObject),

        // === Lambda: λx. body has type α → β where body: β with x: α ===
        Expr::Lambda(param, body) => {
            let param_type = store.fresh_var("arg");
            let new_env = env.extend(param, param_type);
            let body_type = infer(&new_env, body, store)?;
            Ok(store.arrow(param_type, body_type))
        }

        // === Application: f(x) where f: α → β and x: α gives β ===
        Expr::App(func, arg) => {
            let func_type = infer(env, func, store)?;
            let arg_type = infer(env, arg, store)?;
            let result_type = store.fresh_var("result");

            // func_type should unify with (arg_type → result_type)
            let expected_func_type = store.arrow(arg_type, result_type);
            unify(store, func_type, expected_func_type)?;

            Ok(result_type)
        }

        // === If: both branches must have same type, condition must be bool ===
        Expr::If(cond, then_expr, else_expr) => {
            let cond_type = infer(env, cond, store)?;
            let then_type = infer(env, then_expr, store)?;
            let else_type = infer(env, else_expr, store)?;

            // Condition must be bool
            let bool_type = store.bool();
            unify(store, cond_type, bool_type)?;

            // Both branches must have same type
            unify(store, then_type, else_type)?;

            Ok(then_type)
        }

        // === Let: let x = e1 in e2 ===
        // With let-polymorphism: generalize value_type before binding
        Expr::Let(name, value, body) => {
            let value_type = infer(env, value, store)?;
            // Generalize: quantify over variables not free in env
            let scheme = generalize(env, store, value_type);
            let new_env = env.extend_scheme(name, scheme);
            infer(&new_env, body, store)
        }

        // === Let rec: let rec x = e1 in e2 ===
        // The variable is bound with a fresh type before inferring the value,
        // allowing recursive references. After inference, we generalize.
        Expr::LetRec(name, value, body) => {
            // Create a fresh type variable for the recursive binding
            let rec_type = store.fresh_var("rec");
            // Extend environment with the binding BEFORE inferring value (monomorphic for now)
            let rec_env = env.extend(name, rec_type);
            // Infer the value's type in the recursive environment
            let value_type = infer(&rec_env, value, store)?;
            // Unify the recursive type variable with the actual type
            unify(store, rec_type, value_type)?;
            // Now generalize for the body
            let scheme = generalize(env, store, rec_type);
            let body_env = env.extend_scheme(name, scheme);
            infer(&body_env, body, store)
        }

        // === Mutually recursive let: let rec x1 = e1 and x2 = e2 in body ===
        Expr::LetRecMutual(bindings, body) => {
            // Create fresh type variables for ALL bindings first
            let mut rec_types = Vec::new();
            let mut rec_env = env.clone();
            for (name, _) in bindings {
                let rec_type = store.fresh_var("rec");
                rec_env = rec_env.extend(name, rec_type);
                rec_types.push((name.clone(), rec_type));
            }

            // Now infer each value's type with all bindings in scope
            for (i, (_, value)) in bindings.iter().enumerate() {
                let value_type = infer(&rec_env, value, store)?;
                unify(store, rec_types[i].1, value_type)?;
            }

            // Generalize all bindings for the body
            let mut body_env = env.clone();
            for (name, rec_type) in &rec_types {
                let scheme = generalize(env, store, *rec_type);
                body_env = body_env.extend_scheme(name, scheme);
            }

            // Infer the body
            infer(&body_env, body, store)
        }

        // === Block with class definitions ===
        // { class A(x) { ... } class B(y) { ... } body }
        // Treats all classes as mutually recursive bindings.
        // Each class is converted to a constructor function.
        Expr::Block(classes, body) => {
            // Convert classes to bindings: class F(a,b) { fields } -> F = a => b => { fields }
            let bindings: Vec<(String, Expr)> = classes
                .iter()
                .map(|class| (class.name.clone(), class.to_lambda()))
                .collect();

            // Use the LetRecMutual logic
            let mut rec_types = Vec::new();
            let mut rec_env = env.clone();
            for (name, _) in &bindings {
                let rec_type = store.fresh_var("class");
                rec_env = rec_env.extend(name, rec_type);
                rec_types.push((name.clone(), rec_type));
            }

            for (i, (_, value)) in bindings.iter().enumerate() {
                let value_type = infer(&rec_env, &value, store)?;
                unify(store, rec_types[i].1, value_type)?;
            }

            let mut body_env = env.clone();
            for (name, rec_type) in &rec_types {
                let scheme = generalize(env, store, *rec_type);
                body_env = body_env.extend_scheme(name, scheme);
            }

            infer(&body_env, body, store)
        }

        // === Multi-argument call: f(a, b, c) -> ((f a) b) c ===
        Expr::Call(func, args) => {
            let mut result_type = infer(env, func, store)?;

            for arg in args {
                let arg_type = infer(env, arg, store)?;
                let new_result_type = store.fresh_var("result");
                let expected_func_type = store.arrow(arg_type, new_result_type);
                unify(store, result_type, expected_func_type)?;
                result_type = new_result_type;
            }

            Ok(result_type)
        }

        // === Object: { field1: e1, field2: e2, ...spread, ... } ===
        //
        // This is the key case for self-referential types.
        // We create a fresh type variable for `self`, then infer each method
        // with `self` bound to that variable. Finally, we unify `self` with
        // the actual record type, creating an equi-recursive type.
        //
        // For spreads: we require the spread expression to be a record type.
        // The resulting object has an open row (fresh row variable) when spreads
        // are present, allowing it to have additional fields from the spread.
        Expr::Object(fields) => {
            // Create a fresh type variable for `this`
            let self_var = store.fresh_var("self");

            // Create environment with `this` bound
            let obj_env = env.with_self(self_var);

            // Process fields and spreads
            let mut field_types = Vec::new();
            let mut has_spread = false;

            for field in fields {
                match field {
                    ObjectField::Spread(expr) => {
                        // Spread: require the expression to be a record type
                        let spread_type = infer(&obj_env, expr, store)?;
                        let spread_row = store.fresh_row_var("spread");
                        let expected_record = store.record(spread_row);
                        unify(store, spread_type, expected_record)?;
                        has_spread = true;
                    }
                    ObjectField::Field(name, body) => {
                        let method_type = infer(&obj_env, body, store)?;
                        field_types.push((name.clone(), method_type));
                    }
                }
            }

            // Build the record type: { method1: T1, method2: T2, ... }
            // If we have spreads, use an open row (fresh row var) as the tail
            // so the object can have additional fields from spreads.
            // If no spreads, use closed row (row_empty).
            let row_tail = if has_spread {
                store.fresh_row_var("rest")
            } else {
                store.row_empty()
            };

            let mut row = row_tail;
            for (name, ty) in field_types.into_iter().rev() {
                row = store.row_extend_present(&name, ty, row);
            }
            let record_type = store.record(row);

            // Unify self_var with the record type
            // This creates the equi-recursive type: μself. { ... | ρ }
            unify(store, self_var, record_type)?;

            Ok(record_type)
        }

        // === Field access: expr.field ===
        Expr::FieldAccess(obj, field) => {
            let obj_type = infer(env, obj, store)?;

            // obj_type should be a record with `field` present
            let field_type = store.fresh_var("field");
            let row_tail = store.fresh_row_var("row");
            let row = store.row_extend_present(field, field_type, row_tail);
            let expected_type = store.record(row);

            unify(store, obj_type, expected_type)?;

            Ok(field_type)
        }

        // === Equality: e1 == e2 where both have same type (int or string), returns bool ===
        Expr::Eq(left, right) => {
            let left_type = infer(env, left, store)?;
            let right_type = infer(env, right, store)?;

            // Both sides must have the same type
            unify(store, left_type, right_type)?;

            Ok(store.bool())
        }

        // === Boolean AND: e1 && e2 where both are bool, returns bool ===
        Expr::And(left, right) => {
            let left_type = infer(env, left, store)?;
            let right_type = infer(env, right, store)?;

            // Both sides must be bool
            let bool_type = store.bool();
            unify(store, left_type, bool_type)?;
            unify(store, right_type, bool_type)?;

            Ok(store.bool())
        }

        // === Boolean OR: e1 || e2 where both are bool, returns bool ===
        Expr::Or(left, right) => {
            let left_type = infer(env, left, store)?;
            let right_type = infer(env, right, store)?;

            // Both sides must be bool
            let bool_type = store.bool();
            unify(store, left_type, bool_type)?;
            unify(store, right_type, bool_type)?;

            Ok(store.bool())
        }

        // === Arithmetic: e1 op e2 where both are int, returns int ===
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) | Expr::Div(left, right) => {
            let left_type = infer(env, left, store)?;
            let right_type = infer(env, right, store)?;

            // Both sides must be int
            let int_type = store.int();
            unify(store, left_type, int_type)?;
            unify(store, right_type, int_type)?;

            Ok(store.int())
        }

        // === String concatenation: e1 ++ e2 where both are string, returns string ===
        Expr::Concat(left, right) => {
            let left_type = infer(env, left, store)?;
            let right_type = infer(env, right, store)?;

            // Both sides must be string
            let string_type = store.string();
            unify(store, left_type, string_type)?;
            unify(store, right_type, string_type)?;

            Ok(store.string())
        }
    }
}

/// Infer the type of an expression with an empty environment
pub fn infer_expr(expr: &Expr, store: &mut NodeStore) -> Result<NodeId, InferError> {
    infer(&TypeEnv::new(), expr, store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    #[test]
    fn test_infer_bool() {
        let mut store = NodeStore::new();
        let ty = infer_expr(&Expr::bool(true), &mut store).unwrap();
        // Should be bool
        let resolved = store.find(ty);
        match store.get(resolved) {
            crate::node::Node::Const { name } => assert_eq!(name, "bool"),
            _ => panic!("Expected bool constant"),
        }
    }

    #[test]
    fn test_infer_lambda() {
        let mut store = NodeStore::new();
        // λx. x should have type α → α
        let id = Expr::lambda("x", Expr::var("x"));
        let ty = infer_expr(&id, &mut store).unwrap();
        let resolved = store.find(ty);
        match store.get(resolved) {
            crate::node::Node::Arrow { domain, codomain, .. } => {
                // domain and codomain should be the same variable
                assert_eq!(store.find(*domain), store.find(*codomain));
            }
            _ => panic!("Expected arrow type"),
        }
    }

    #[test]
    fn test_infer_simple_object() {
        let mut store = NodeStore::new();
        // { x = true } should have type { x: bool | ρ }
        let obj = Expr::object(vec![("x", Expr::bool(true))]);
        let result = infer_expr(&obj, &mut store);
        assert!(result.is_ok());
    }

    #[test]
    fn test_infer_object_with_this() {
        let mut store = NodeStore::new();
        // { self_ref = this } - an object that refers to itself
        let obj = Expr::object(vec![("self_ref", Expr::this())]);
        let result = infer_expr(&obj, &mut store);
        assert!(result.is_ok());
    }
}
