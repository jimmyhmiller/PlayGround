// DOT (Graphviz) file format types

/// Parsed DOT graph
#[derive(Debug, Clone)]
pub struct DotGraph {
    /// Graph name (from "digraph NAME { ... }")
    pub name: String,
    /// Nodes in the graph
    pub nodes: Vec<DotNode>,
    /// Edges in the graph
    pub edges: Vec<DotEdge>,
}

/// A node in the DOT graph
#[derive(Debug, Clone)]
pub struct DotNode {
    /// Node identifier
    pub id: String,
    /// Optional label (from label="...")
    pub label: Option<String>,
    /// Whether this is a loop header (from loopheader="true")
    pub is_loop_header: bool,
    /// Whether this node has a backedge (from backedge="true")
    pub is_backedge: bool,
}

/// An edge in the DOT graph
#[derive(Debug, Clone)]
pub struct DotEdge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Optional edge label
    pub label: Option<String>,
}

impl DotGraph {
    pub fn new(name: String) -> Self {
        DotGraph {
            name,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
}

impl DotNode {
    pub fn new(id: String) -> Self {
        DotNode {
            id,
            label: None,
            is_loop_header: false,
            is_backedge: false,
        }
    }
}

impl DotEdge {
    pub fn new(from: String, to: String) -> Self {
        DotEdge {
            from,
            to,
            label: None,
        }
    }
}
