use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::expr::Expr;
use crate::rule::{Rule, When};
use crate::value::Value;

/// True if the rule has at least one `When::Input` pattern. A rule
/// without one is "source-style" — it fires off slot/guard state
/// alone and must be re-tested every fire scan even when the inbox
/// is empty.
pub fn rule_has_input(r: &Rule) -> bool {
    r.when.iter().any(|w| matches!(w, When::Input { .. }))
}

/// One end of an edge in a template. Resolved at spawn time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeEnd {
    /// The newly-created node.
    ThisInstance,
    /// The node that did the spawning (the firing node, or an explicit parent).
    Parent,
    /// A node sharing the spawner's enclosing-compound prefix, addressed by
    /// its local (unqualified) name. Resolved at spawn time against the
    /// spawner's qualified name: spawner `ASG::Scaler` + `Sibling("LB")`
    /// → `ASG::LB`. Lets a dynamically-spawned worker wire itself to a
    /// fixed sibling (e.g. a load balancer) rather than only to its
    /// spawner. Hard-errors if no such sibling exists.
    Sibling(String),
}

/// An edge declared inside a template. Created alongside the new
/// instance at spawn time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSpec {
    pub from: EdgeEnd,
    pub to: EdgeEnd,
    pub latency: Expr,
}

/// A declared probe on a class. At read time, each part is evaluated
/// against the node's slots and concatenated into the displayed string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub label: String,
    pub parts: Vec<ProbePart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbePart {
    Literal(String),
    Hole(Expr),
}

/// A reusable node shape: slots + rules + associated edges.
///
/// The edges capture how the instance wires into the surrounding
/// graph. Typical autoscaling template has: `Parent → ThisInstance`
/// (for forwarding work) and `ThisInstance → Parent` (for replies).
/// Anything that requires edges to *other* nodes than the parent has
/// to wire them manually after spawn.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Probes declared on the class. Copied into every instance.
    pub probes: Vec<Probe>,
    /// Cached: does any rule fire without consuming an inbox packet?
    /// (i.e. a rule with no `When::Input` pattern, driven only by
    /// slot matches and guards.) The fire loop uses this to skip
    /// nodes with empty inboxes — a huge win on dense graphs where
    /// most nodes are quiescent at any given instant.
    #[serde(default, skip_serializing)]
    pub has_source_rule: bool,
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
            probes: Vec::new(),
            has_source_rule: false,
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
        if !rule_has_input(&r) {
            self.has_source_rule = true;
        }
        self.rules.push(r);
        self
    }

    /// Recompute `has_source_rule` from the current rule set. Called
    /// after a direct mutation to `rules` (the field is `pub` so
    /// callers can append; if they do, they must invalidate the
    /// cache by calling this — `Sim::add_rule_to_node` does it
    /// automatically).
    pub fn refresh_has_source_rule(&mut self) {
        self.has_source_rule = self.rules.iter().any(|r| !rule_has_input(r));
    }
    pub fn edge(mut self, from: EdgeEnd, to: EdgeEnd, latency: Expr) -> Self {
        self.edges.push(EdgeSpec { from, to, latency });
        self
    }
}
