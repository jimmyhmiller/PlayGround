//! Unification algorithm with equi-recursive type support
//!
//! This implements union-find based unification for types with row polymorphism.
//! Unlike traditional unification that fails on occurs-check violations,
//! we allow cycles to represent equi-recursive types.

use crate::node::{Node, NodeId};
use crate::store::NodeStore;
use std::collections::HashSet;
use std::fmt;

/// Errors that can occur during unification
#[derive(Debug, Clone)]
pub enum TypeError {
    /// Cannot unify two different constant types
    ConstantMismatch { expected: String, found: String },
    /// Cannot unify structurally incompatible types
    StructuralMismatch { expected: String, found: String },
    /// Row kinding error
    RowKindingError { message: String },
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeError::ConstantMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
            TypeError::StructuralMismatch { expected, found } => {
                write!(
                    f,
                    "Structural mismatch: cannot unify {} with {}",
                    expected, found
                )
            }
            TypeError::RowKindingError { message } => {
                write!(f, "Row kinding error: {}", message)
            }
        }
    }
}

impl std::error::Error for TypeError {}

/// Unification context tracking in-progress unifications for cycle detection
pub struct UnifyContext {
    /// Set of (n1, n2) pairs currently being unified
    /// If we encounter a pair already in this set, we have a cycle (equi-recursive type)
    in_progress: HashSet<(u32, u32)>,
}

impl UnifyContext {
    pub fn new() -> Self {
        UnifyContext {
            in_progress: HashSet::new(),
        }
    }
}

impl Default for UnifyContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Unify two types
///
/// This is the main entry point for unification.
pub fn unify(store: &mut NodeStore, n1: NodeId, n2: NodeId) -> Result<(), TypeError> {
    let mut ctx = UnifyContext::new();
    unify_with_ctx(store, n1, n2, &mut ctx)
}

/// Unify two types with a context for cycle detection
fn unify_with_ctx(
    store: &mut NodeStore,
    n1: NodeId,
    n2: NodeId,
    ctx: &mut UnifyContext,
) -> Result<(), TypeError> {
    let n1 = store.find(n1);
    let n2 = store.find(n2);

    // Already the same node
    if n1 == n2 {
        return Ok(());
    }

    // Normalize order for consistent cycle detection
    let (lo, hi) = if n1.0 < n2.0 { (n1.0, n2.0) } else { (n2.0, n1.0) };

    // Cycle detection for equi-recursive types
    // If we're already trying to unify these nodes, we have a recursive type
    if ctx.in_progress.contains(&(lo, hi)) {
        // This is not an error - it's an equi-recursive type!
        // Just return success and let the cycle exist in the graph
        return Ok(());
    }

    ctx.in_progress.insert((lo, hi));

    let result = unify_nodes(store, n1, n2, ctx);

    ctx.in_progress.remove(&(lo, hi));

    result
}

/// Unify two nodes (internal implementation)
fn unify_nodes(
    store: &mut NodeStore,
    n1: NodeId,
    n2: NodeId,
    ctx: &mut UnifyContext,
) -> Result<(), TypeError> {
    // Clone the nodes to avoid borrow issues
    let node1 = store.get(n1).clone();
    let node2 = store.get(n2).clone();

    match (&node1, &node2) {
        // Variable unifies with anything
        (Node::Var { .. }, _) => {
            store.union(n1, n2);
            Ok(())
        }
        (_, Node::Var { .. }) => {
            store.union(n2, n1);
            Ok(())
        }

        // Row variable unifies with row types
        (Node::RowVar { .. }, _) if is_row_like(&node2) => {
            store.union(n1, n2);
            Ok(())
        }
        (_, Node::RowVar { .. }) if is_row_like(&node1) => {
            store.union(n2, n1);
            Ok(())
        }

        // Constants must match exactly
        (Node::Const { name: name1 }, Node::Const { name: name2 }) => {
            if name1 == name2 {
                Ok(())
            } else {
                Err(TypeError::ConstantMismatch {
                    expected: name1.clone(),
                    found: name2.clone(),
                })
            }
        }

        // Arrow types: unify domain and codomain
        (
            Node::Arrow {
                domain: d1,
                codomain: c1,
                ..
            },
            Node::Arrow {
                domain: d2,
                codomain: c2,
                ..
            },
        ) => {
            store.union(n1, n2);
            unify_with_ctx(store, *d1, *d2, ctx)?;
            unify_with_ctx(store, *c1, *c2, ctx)?;
            Ok(())
        }

        // Record types: unify the rows
        (Node::Record { row: r1, .. }, Node::Record { row: r2, .. }) => {
            store.union(n1, n2);
            unify_row(store, *r1, *r2, ctx)?;
            Ok(())
        }

        // Present fields: unify the types
        (Node::Present { ty: t1, .. }, Node::Present { ty: t2, .. }) => {
            store.union(n1, n2);
            unify_with_ctx(store, *t1, *t2, ctx)?;
            Ok(())
        }

        // Absent fields unify with each other
        (Node::Absent { .. }, Node::Absent { .. }) => {
            store.union(n1, n2);
            Ok(())
        }

        // Empty rows unify with each other
        (Node::RowEmpty { .. }, Node::RowEmpty { .. }) => {
            store.union(n1, n2);
            Ok(())
        }

        // Row extensions are handled by unify_row
        (Node::RowExtend { .. }, _) | (_, Node::RowExtend { .. }) => {
            unify_row(store, n1, n2, ctx)
        }

        // Structural mismatch
        _ => Err(TypeError::StructuralMismatch {
            expected: node1.kind().to_string(),
            found: node2.kind().to_string(),
        }),
    }
}

/// Check if a node is a row-like type
fn is_row_like(node: &Node) -> bool {
    matches!(
        node,
        Node::RowEmpty { .. } | Node::RowExtend { .. } | Node::RowVar { .. }
    )
}

/// Unify two rows using Rémy's algorithm
///
/// This handles row polymorphism by:
/// 1. Collecting explicit fields from both rows
/// 2. Padding each row with fields from the other that it's missing
/// 3. Unifying the field types for matching fields
pub fn unify_row(
    store: &mut NodeStore,
    r1: NodeId,
    r2: NodeId,
    ctx: &mut UnifyContext,
) -> Result<(), TypeError> {
    let r1 = store.find(r1);
    let r2 = store.find(r2);

    if r1 == r2 {
        return Ok(());
    }

    let node1 = store.get(r1).clone();
    let node2 = store.get(r2).clone();

    match (&node1, &node2) {
        // Row variable unifies with any row
        (Node::RowVar { .. }, _) => {
            store.union(r1, r2);
            Ok(())
        }
        (_, Node::RowVar { .. }) => {
            store.union(r2, r1);
            Ok(())
        }

        // Empty rows unify
        (Node::RowEmpty { .. }, Node::RowEmpty { .. }) => {
            store.union(r1, r2);
            Ok(())
        }

        // Extension with extension: use Rémy's algorithm
        (Node::RowExtend { .. }, Node::RowExtend { .. }) => {
            // Get explicit fields from both rows
            let fields1 = explicits(store, r1)?;
            let fields2 = explicits(store, r2)?;

            // Fields in r1 but not r2
            let diff1: Vec<_> = fields1
                .iter()
                .filter(|f| !fields2.contains(f))
                .cloned()
                .collect();
            // Fields in r2 but not r1
            let diff2: Vec<_> = fields2
                .iter()
                .filter(|f| !fields1.contains(f))
                .cloned()
                .collect();

            // Create a fresh row variable for the common tail
            let common_tail = store.fresh_row_var("r");

            // Pad r1 with fields from diff2
            pad(store, r1, &diff2, common_tail)?;
            // Pad r2 with fields from diff1
            pad(store, r2, &diff1, common_tail)?;

            // Now unify matching fields
            let subgoals = find_subgoals(store, r1, r2, &fields1)?;
            for (f1, f2) in subgoals {
                unify_with_ctx(store, f1, f2, ctx)?;
            }

            Ok(())
        }

        // Extension with empty - only valid if extension has no explicit fields
        // or all fields are absent
        (Node::RowExtend { .. }, Node::RowEmpty { .. })
        | (Node::RowEmpty { .. }, Node::RowExtend { .. }) => {
            // For simplicity, we'll create fresh variables for any fields
            // and unify the tails
            Err(TypeError::RowKindingError {
                message: "Cannot unify row extension with empty row".to_string(),
            })
        }

        _ => Err(TypeError::RowKindingError {
            message: format!(
                "Expected row types, got {} and {}",
                node1.kind(),
                node2.kind()
            ),
        }),
    }
}

/// Get all explicit field names from a row
pub fn explicits(store: &NodeStore, row: NodeId) -> Result<Vec<String>, TypeError> {
    let mut fields = Vec::new();
    let mut current = store.find(row);

    loop {
        let node = store.get(current).clone();
        match node {
            Node::RowExtend { field, rest, .. } => {
                fields.push(field);
                current = store.find(rest);
            }
            Node::RowVar { .. } | Node::RowEmpty { .. } => {
                break;
            }
            _ => {
                return Err(TypeError::RowKindingError {
                    message: format!("Expected row type, got {}", node.kind()),
                });
            }
        }
    }

    Ok(fields)
}

/// Pad a row with new fields, pointing to a new tail
///
/// This finds the tail of the row (a row variable) and replaces it with
/// an extension containing the new fields.
///
/// IMPORTANT: If the tail is RowEmpty (closed row), we cannot add new fields.
/// This enforces that closed rows must have exactly the fields they define.
pub fn pad(
    store: &mut NodeStore,
    row: NodeId,
    new_fields: &[String],
    new_tail: NodeId,
) -> Result<(), TypeError> {
    // Find the tail of the row
    let tail = find_row_tail(store, row)?;

    // Check if the row is closed (tail is RowEmpty)
    let is_closed = matches!(store.get(tail), Node::RowEmpty { .. });

    if new_fields.is_empty() {
        // No new fields to add, just unify tails
        if is_closed {
            // Closed row - new_tail must also be empty or compatible
            // For now, just union and let later checks catch issues
            store.union(tail, new_tail);
        } else {
            store.union(tail, new_tail);
        }
        return Ok(());
    }

    // We have new fields to add
    if is_closed {
        // Cannot add fields to a closed row!
        return Err(TypeError::RowKindingError {
            message: format!(
                "Cannot add fields {:?} to a closed row (object doesn't have these fields)",
                new_fields
            ),
        });
    }

    // Build extension: new_fields... | new_tail
    let mut ext = new_tail;
    for field in new_fields.iter().rev() {
        // Create fresh type variable for the new field
        let field_var = store.fresh_var("f");
        ext = store.row_extend_present(field, field_var, ext);
    }

    // Union the tail with the extension
    store.union(tail, ext);

    Ok(())
}

/// Find the tail (final variable or empty) of a row
fn find_row_tail(store: &NodeStore, row: NodeId) -> Result<NodeId, TypeError> {
    let mut current = store.find(row);

    loop {
        let node = store.get(current).clone();
        match node {
            Node::RowExtend { rest, .. } => {
                current = store.find(rest);
            }
            Node::RowVar { .. } | Node::RowEmpty { .. } => {
                return Ok(current);
            }
            _ => {
                return Err(TypeError::RowKindingError {
                    message: format!("Expected row type, got {}", node.kind()),
                });
            }
        }
    }
}

/// Find a field's presence node in a row
fn find_field(store: &NodeStore, row: NodeId, field: &str) -> Result<NodeId, TypeError> {
    let mut current = store.find(row);

    loop {
        let node = store.get(current).clone();
        match node {
            Node::RowExtend {
                field: f,
                presence,
                rest,
                ..
            } => {
                if f == field {
                    return Ok(store.find(presence));
                }
                current = store.find(rest);
            }
            Node::RowVar { .. } | Node::RowEmpty { .. } => {
                return Err(TypeError::RowKindingError {
                    message: format!("Field {} not found in row", field),
                });
            }
            _ => {
                return Err(TypeError::RowKindingError {
                    message: format!("Expected row type, got {}", node.kind()),
                });
            }
        }
    }
}

/// Find pairs of presence nodes to unify for matching fields
fn find_subgoals(
    store: &NodeStore,
    r1: NodeId,
    r2: NodeId,
    fields: &[String],
) -> Result<Vec<(NodeId, NodeId)>, TypeError> {
    let mut subgoals = Vec::new();

    for field in fields {
        let p1 = find_field(store, r1, field)?;
        let p2 = find_field(store, r2, field)?;
        subgoals.push((p1, p2));
    }

    Ok(subgoals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_same_const() {
        let mut store = NodeStore::new();
        let int1 = store.int();
        let int2 = store.int();
        assert!(unify(&mut store, int1, int2).is_ok());
    }

    #[test]
    fn test_unify_different_const() {
        let mut store = NodeStore::new();
        let int = store.int();
        let bool = store.bool();
        assert!(unify(&mut store, int, bool).is_err());
    }

    #[test]
    fn test_unify_var_with_const() {
        let mut store = NodeStore::new();
        let var = store.fresh_var("x");
        let int = store.int();
        assert!(unify(&mut store, var, int).is_ok());
        // After unification, var should resolve to int
        let resolved = store.find(var);
        assert_eq!(resolved, store.find(int));
    }

    #[test]
    fn test_unify_arrow() {
        let mut store = NodeStore::new();
        let int = store.int();
        let bool = store.bool();
        let arr1 = store.arrow(int, bool);

        let x = store.fresh_var("x");
        let y = store.fresh_var("y");
        let arr2 = store.arrow(x, y);

        assert!(unify(&mut store, arr1, arr2).is_ok());
    }
}
