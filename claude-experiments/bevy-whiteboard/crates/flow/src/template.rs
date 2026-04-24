use std::collections::BTreeMap;

use crate::expr::Expr;
use crate::rule::Rule;
use crate::value::Value;

/// One end of an edge in a template. Resolved at spawn time.
#[derive(Debug, Clone, Copy)]
pub enum EdgeEnd {
    /// The newly-created node.
    ThisInstance,
    /// The node that did the spawning (the firing node, or an explicit parent).
    Parent,
}

/// An edge declared inside a template. Created alongside the new
/// instance at spawn time.
#[derive(Debug, Clone)]
pub struct EdgeSpec {
    pub from: EdgeEnd,
    pub to: EdgeEnd,
    pub latency: Expr,
}

/// A reusable node shape: slots + rules + associated edges.
///
/// The edges capture how the instance wires into the surrounding
/// graph. Typical autoscaling template has: `Parent → ThisInstance`
/// (for forwarding work) and `ThisInstance → Parent` (for replies).
/// Anything that requires edges to *other* nodes than the parent has
/// to wire them manually after spawn.
#[derive(Debug, Clone)]
pub struct Template {
    pub name: String,
    /// Prefix for instance naming. Instances get names like `Worker_7`.
    pub node_name_prefix: String,
    /// Initial slot values for each instance.
    pub slots: BTreeMap<String, Value>,
    /// Rules copied into each instance.
    pub rules: Vec<Rule>,
    /// Edges created alongside each instance.
    pub edges: Vec<EdgeSpec>,
    /// Packets delivered into the new instance's inbox at
    /// instantiation time. Seeds the class's self-driven loops
    /// (e.g. a `tick(nil)` that a `rule tick` rearms each step).
    pub initial_packets: Vec<Value>,
}

impl Template {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            node_name_prefix: name.clone(),
            name,
            slots: BTreeMap::new(),
            rules: Vec::new(),
            edges: Vec::new(),
            initial_packets: Vec::new(),
        }
    }
    pub fn initial_packet(mut self, v: Value) -> Self {
        self.initial_packets.push(v);
        self
    }
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.node_name_prefix = prefix.into();
        self
    }
    pub fn slot(mut self, name: impl Into<String>, initial: Value) -> Self {
        self.slots.insert(name.into(), initial);
        self
    }
    pub fn rule(mut self, r: Rule) -> Self {
        self.rules.push(r);
        self
    }
    pub fn edge(mut self, from: EdgeEnd, to: EdgeEnd, latency: Expr) -> Self {
        self.edges.push(EdgeSpec { from, to, latency });
        self
    }
}
