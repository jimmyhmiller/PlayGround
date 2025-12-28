//! Type inference for expressions
//!
//! This implements Hindley-Milner style type inference extended with:
//! - Row polymorphism for structural object types
//! - Equi-recursive types for self-reference (this)

use crate::expr::Expr;
use crate::node::NodeId;
use crate::store::NodeStore;
use crate::unify::{unify, TypeError};
use std::collections::HashMap;

/// Type environment mapping variable names to their types
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Variable bindings
    bindings: HashMap<String, NodeId>,
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

    /// Extend the environment with a new binding
    pub fn extend(&self, name: impl Into<String>, ty: NodeId) -> Self {
        let mut new_env = self.clone();
        new_env.bindings.insert(name.into(), ty);
        new_env
    }

    /// Look up a variable in the environment
    pub fn lookup(&self, name: &str) -> Option<NodeId> {
        self.bindings.get(name).copied()
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
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
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

        // === Variables ===
        Expr::Var(name) => env
            .lookup(name)
            .ok_or_else(|| InferError::UndefinedVar(name.clone())),

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
        Expr::Let(name, value, body) => {
            let value_type = infer(env, value, store)?;
            let new_env = env.extend(name, value_type);
            infer(&new_env, body, store)
        }

        // === Let rec: let rec x = e1 in e2 ===
        // The variable is bound with a fresh type before inferring the value,
        // allowing recursive references.
        Expr::LetRec(name, value, body) => {
            // Create a fresh type variable for the recursive binding
            let rec_type = store.fresh_var("rec");
            // Extend environment with the binding BEFORE inferring value
            let rec_env = env.extend(name, rec_type);
            // Infer the value's type in the recursive environment
            let value_type = infer(&rec_env, value, store)?;
            // Unify the recursive type variable with the actual type
            unify(store, rec_type, value_type)?;
            // Infer the body with the recursive binding
            infer(&rec_env, body, store)
        }

        // === Mutually recursive let: let rec x1 = e1 and x2 = e2 in body ===
        Expr::LetRecMutual(bindings, body) => {
            // Create fresh type variables for ALL bindings first
            let mut rec_types = Vec::new();
            let mut rec_env = env.clone();
            for (name, _) in bindings {
                let rec_type = store.fresh_var("rec");
                rec_env = rec_env.extend(name, rec_type);
                rec_types.push(rec_type);
            }

            // Now infer each value's type with all bindings in scope
            for (i, (_, value)) in bindings.iter().enumerate() {
                let value_type = infer(&rec_env, value, store)?;
                unify(store, rec_types[i], value_type)?;
            }

            // Infer the body
            infer(&rec_env, body, store)
        }

        // === Object: { method1 = e1, method2 = e2, ... } ===
        //
        // This is the key case for self-referential types.
        // We create a fresh type variable for `self`, then infer each method
        // with `self` bound to that variable. Finally, we unify `self` with
        // the actual record type, creating an equi-recursive type.
        Expr::Object(methods) => {
            // Create a fresh type variable for `this`
            let self_var = store.fresh_var("self");

            // Create environment with `this` bound
            let obj_env = env.with_self(self_var);

            // Infer each method's type
            let mut field_types = Vec::new();
            for (name, body) in methods {
                let method_type = infer(&obj_env, body, store)?;
                field_types.push((name.clone(), method_type));
            }

            // Build the record type: { method1: T1, method2: T2, ... }
            // Objects have CLOSED rows - they have exactly the fields defined, no more.
            // Structural subtyping comes from function parameters having OPEN rows.
            let row_tail = store.row_empty();
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

        // === Equality: e1 == e2 where both are int, returns bool ===
        Expr::Eq(left, right) => {
            let left_type = infer(env, left, store)?;
            let right_type = infer(env, right, store)?;

            // Both sides must be int
            let int_type = store.int();
            unify(store, left_type, int_type)?;
            unify(store, right_type, int_type)?;

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
