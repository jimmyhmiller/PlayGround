//! Pretty printing for types with cycle detection
//!
//! Converts the node graph back to readable type notation,
//! detecting cycles and introducing μ binders for recursive types.

use crate::node::{Node, NodeId};
use crate::store::NodeStore;
use std::collections::{HashMap, HashSet};

/// Pretty print a type node
pub fn display_type(store: &NodeStore, node_id: NodeId) -> String {
    let mut ctx = DisplayContext::new();
    ctx.display(store, node_id)
}

/// Context for pretty printing with cycle detection
struct DisplayContext {
    /// Nodes we're currently in the process of printing (for cycle detection)
    in_progress: HashSet<NodeId>,
    /// Nodes we've detected as recursive, mapped to their μ-variable names
    recursive_vars: HashMap<NodeId, String>,
    /// Counter for generating fresh μ-variable names
    var_counter: u32,
}

impl DisplayContext {
    fn new() -> Self {
        DisplayContext {
            in_progress: HashSet::new(),
            recursive_vars: HashMap::new(),
            var_counter: 0,
        }
    }

    fn fresh_mu_var(&mut self) -> String {
        let name = format!("α{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    fn display(&mut self, store: &NodeStore, node_id: NodeId) -> String {
        let node_id = store.find(node_id);

        // Check if we're in a cycle
        if self.in_progress.contains(&node_id) {
            // We've found a cycle - use or create a μ-variable
            if let Some(var) = self.recursive_vars.get(&node_id) {
                return var.clone();
            }
            // This shouldn't happen in normal use, but handle gracefully
            let var = self.fresh_mu_var();
            self.recursive_vars.insert(node_id, var.clone());
            return var;
        }

        // First pass: detect if this node leads to a cycle
        let is_recursive = self.detect_cycle(store, node_id);

        if is_recursive && !self.recursive_vars.contains_key(&node_id) {
            // This is a recursive type - assign it a μ-variable
            let var = self.fresh_mu_var();
            self.recursive_vars.insert(node_id, var.clone());

            // Format as μα. τ
            self.in_progress.insert(node_id);
            let body = self.display_node(store, node_id);
            self.in_progress.remove(&node_id);

            format!("μ{}. {}", var, body)
        } else {
            self.in_progress.insert(node_id);
            let result = self.display_node(store, node_id);
            self.in_progress.remove(&node_id);
            result
        }
    }

    /// Check if following this node would lead back to itself
    fn detect_cycle(&self, store: &NodeStore, start: NodeId) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        while let Some(node_id) = stack.pop() {
            let node_id = store.find(node_id);

            if !visited.insert(node_id) {
                // Already visited this node - cycle detected
                if node_id == store.find(start) {
                    return true;
                }
                continue;
            }

            // Add children to stack
            match store.get(node_id) {
                Node::Arrow { domain, codomain, .. } => {
                    stack.push(*domain);
                    stack.push(*codomain);
                }
                Node::Record { row, .. } => {
                    stack.push(*row);
                }
                Node::RowExtend { presence, rest, .. } => {
                    stack.push(*presence);
                    stack.push(*rest);
                }
                Node::Present { ty, .. } => {
                    stack.push(*ty);
                }
                _ => {}
            }
        }

        false
    }

    fn display_node(&mut self, store: &NodeStore, node_id: NodeId) -> String {
        let node = store.get(node_id).clone();

        match node {
            Node::Var { name, id, .. } => {
                if let Some(var) = self.recursive_vars.get(&node_id) {
                    var.clone()
                } else {
                    format!("{}{}", name, id)
                }
            }

            Node::Const { name } => name,

            Node::Arrow {
                domain, codomain, ..
            } => {
                let d = self.display(store, domain);
                let c = self.display(store, codomain);

                // Parenthesize domain if it's an arrow type
                let d_node = store.get(store.find(domain));
                let d_str = if matches!(d_node, Node::Arrow { .. }) {
                    format!("({})", d)
                } else {
                    d
                };

                format!("{} → {}", d_str, c)
            }

            Node::Record { row, .. } => {
                let row_str = self.display_row(store, row);
                format!("{{ {} }}", row_str)
            }

            Node::RowEmpty { .. } => String::new(),

            Node::RowExtend { .. } => self.display_row(store, node_id),

            Node::RowVar { name, id, .. } => {
                if let Some(var) = self.recursive_vars.get(&node_id) {
                    var.clone()
                } else {
                    format!("{}{}", name, id)
                }
            }

            Node::Present { ty, .. } => self.display(store, ty),

            Node::Absent { .. } => "⊥".to_string(),
        }
    }

    fn display_row(&mut self, store: &NodeStore, row_id: NodeId) -> String {
        let row_id = store.find(row_id);

        // Check for recursive row reference
        if self.in_progress.contains(&row_id) {
            if let Some(var) = self.recursive_vars.get(&row_id) {
                return format!("| {}", var);
            }
        }

        let node = store.get(row_id).clone();

        match node {
            Node::RowEmpty { .. } => String::new(),

            Node::RowVar { name, id, .. } => {
                if let Some(var) = self.recursive_vars.get(&row_id) {
                    format!("| {}", var)
                } else {
                    format!("| {}{}", name, id)
                }
            }

            Node::RowExtend {
                field,
                presence,
                rest,
                ..
            } => {
                let presence_str = self.display(store, presence);
                let rest_str = self.display_row(store, rest);

                if rest_str.is_empty() {
                    format!("{}: {}", field, presence_str)
                } else if rest_str.starts_with('|') {
                    format!("{}: {} {}", field, presence_str, rest_str)
                } else {
                    format!("{}: {}, {}", field, presence_str, rest_str)
                }
            }

            _ => format!("<invalid row: {:?}>", node.kind()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unify::unify;

    #[test]
    fn test_display_const() {
        let mut store = NodeStore::new();
        let int = store.int();
        assert_eq!(display_type(&store, int), "int");
    }

    #[test]
    fn test_display_arrow() {
        let mut store = NodeStore::new();
        let int = store.int();
        let bool = store.bool();
        let arrow = store.arrow(int, bool);
        assert_eq!(display_type(&store, arrow), "int → bool");
    }

    #[test]
    fn test_display_record() {
        let mut store = NodeStore::new();
        let int = store.int();
        let row_empty = store.row_empty();
        let row = store.row_extend_present("x", int, row_empty);
        let record = store.record(row);
        assert_eq!(display_type(&store, record), "{ x: int }");
    }

    #[test]
    fn test_display_recursive() {
        let mut store = NodeStore::new();
        // Create a recursive type: μα. { self: α }
        let self_var = store.fresh_var("self");
        let row_empty = store.row_empty();
        let row = store.row_extend_present("self_ref", self_var, row_empty);
        let record = store.record(row);

        // Unify self_var with the record to create the cycle
        unify(&mut store, self_var, record).unwrap();

        let displayed = display_type(&store, record);
        // Should contain μ and show the recursive structure
        assert!(displayed.contains("μ"), "Expected recursive type: {}", displayed);
    }
}
