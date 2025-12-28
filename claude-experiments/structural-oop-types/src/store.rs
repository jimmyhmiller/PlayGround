//! Node store - arena for type nodes and fresh variable generation

use crate::node::{Node, NodeId};
use crate::types::{FieldPresence, Row, Type};
use std::collections::HashMap;

/// Arena-based store for type nodes
pub struct NodeStore {
    /// All nodes in the store
    nodes: Vec<Node>,
    /// Counter for generating fresh variable ids
    next_var_id: u32,
}

impl NodeStore {
    /// Create a new empty store
    pub fn new() -> Self {
        NodeStore {
            nodes: Vec::new(),
            next_var_id: 0,
        }
    }

    /// Add a node to the store and return its id
    pub fn add(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Get a node by id
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Generate a fresh unique id for a variable
    fn fresh_id(&mut self) -> u32 {
        let id = self.next_var_id;
        self.next_var_id += 1;
        id
    }

    /// Create a fresh type variable
    pub fn fresh_var(&mut self, name: impl Into<String>) -> NodeId {
        let id = self.fresh_id();
        self.add(Node::var(name, id))
    }

    /// Create a fresh row variable
    pub fn fresh_row_var(&mut self, name: impl Into<String>) -> NodeId {
        let id = self.fresh_id();
        self.add(Node::row_var(name, id))
    }

    /// Create a constant type node
    pub fn constant(&mut self, name: impl Into<String>) -> NodeId {
        self.add(Node::constant(name))
    }

    /// Create a bool type node
    pub fn bool(&mut self) -> NodeId {
        self.constant("bool")
    }

    /// Create an int type node
    pub fn int(&mut self) -> NodeId {
        self.constant("int")
    }

    /// Create a string type node
    pub fn string(&mut self) -> NodeId {
        self.constant("string")
    }

    /// Create an arrow (function) type node
    pub fn arrow(&mut self, domain: NodeId, codomain: NodeId) -> NodeId {
        self.add(Node::arrow(domain, codomain))
    }

    /// Create a record type node
    pub fn record(&mut self, row: NodeId) -> NodeId {
        self.add(Node::record(row))
    }

    /// Create an empty row node
    pub fn row_empty(&mut self) -> NodeId {
        self.add(Node::row_empty())
    }

    /// Create a row extension node with a present field
    pub fn row_extend_present(
        &mut self,
        field: impl Into<String>,
        ty: NodeId,
        rest: NodeId,
    ) -> NodeId {
        let presence = self.add(Node::present(ty));
        self.add(Node::row_extend(field, presence, rest))
    }

    /// Create a row extension node with an absent field
    pub fn row_extend_absent(&mut self, field: impl Into<String>, rest: NodeId) -> NodeId {
        let presence = self.add(Node::absent());
        self.add(Node::row_extend(field, presence, rest))
    }

    /// Create a present field node
    pub fn present(&mut self, ty: NodeId) -> NodeId {
        self.add(Node::present(ty))
    }

    /// Create an absent field node
    pub fn absent(&mut self) -> NodeId {
        self.add(Node::absent())
    }

    /// Union-find: find the representative of a node
    ///
    /// Follows links until finding a node that points to NULL (its own representative).
    /// Does NOT perform path compression (to keep implementation simple).
    pub fn find(&self, mut id: NodeId) -> NodeId {
        loop {
            let node = self.get(id);
            if !node.has_link() {
                // Constants are their own representative
                return id;
            }
            let link = node.link().get();
            if link.is_null() {
                return id;
            }
            id = link;
        }
    }

    /// Union-find: union two nodes by making n1's representative point to n2's representative
    pub fn union(&self, n1: NodeId, n2: NodeId) {
        let r1 = self.find(n1);
        let r2 = self.find(n2);
        if r1 != r2 {
            let node = self.get(r1);
            if node.has_link() {
                node.link().set(r2);
            }
        }
    }

    /// Translate an abstract Type to a Node, returning the NodeId
    ///
    /// Uses a variable environment to ensure consistent variable handling.
    pub fn translate_type(&mut self, ty: &Type) -> NodeId {
        let mut env: HashMap<(String, u32), NodeId> = HashMap::new();
        self.translate_type_with_env(ty, &mut env)
    }

    fn translate_type_with_env(
        &mut self,
        ty: &Type,
        env: &mut HashMap<(String, u32), NodeId>,
    ) -> NodeId {
        match ty {
            Type::Var(name, id) => {
                let key = (name.clone(), *id);
                if let Some(&node_id) = env.get(&key) {
                    node_id
                } else {
                    let node_id = self.add(Node::var(name, *id));
                    env.insert(key, node_id);
                    node_id
                }
            }
            Type::Const(name) => self.constant(name),
            Type::Arrow(domain, codomain) => {
                let d = self.translate_type_with_env(domain, env);
                let c = self.translate_type_with_env(codomain, env);
                self.arrow(d, c)
            }
            Type::Record(row) => {
                let r = self.translate_row_with_env(row, env);
                self.record(r)
            }
            Type::Mu(var, body) => {
                // For Mu types, we create a fresh variable and add it to the env
                // before translating the body, creating the cycle
                let var_id = self.fresh_id();
                let node_id = self.add(Node::var(var, var_id));
                env.insert((var.clone(), var_id), node_id);
                let body_id = self.translate_type_with_env(body, env);
                // The variable should eventually be unified with the body
                // For now, we just return the body (the Mu is implicit)
                body_id
            }
        }
    }

    fn translate_row_with_env(
        &mut self,
        row: &Row,
        env: &mut HashMap<(String, u32), NodeId>,
    ) -> NodeId {
        match row {
            Row::Empty => self.row_empty(),
            Row::Var(name, id) => {
                let key = (name.clone(), *id);
                if let Some(&node_id) = env.get(&key) {
                    node_id
                } else {
                    let node_id = self.add(Node::row_var(name, *id));
                    env.insert(key, node_id);
                    node_id
                }
            }
            Row::Extend {
                field,
                presence,
                rest,
            } => {
                let rest_id = self.translate_row_with_env(rest, env);
                match presence {
                    FieldPresence::Present(ty) => {
                        let ty_id = self.translate_type_with_env(ty, env);
                        self.row_extend_present(field, ty_id, rest_id)
                    }
                    FieldPresence::Absent => self.row_extend_absent(field, rest_id),
                }
            }
        }
    }
}

impl Default for NodeStore {
    fn default() -> Self {
        Self::new()
    }
}
