//! Runtime node representation for unification
//!
//! Like OCaml's `node` type but using indices into an arena instead of
//! mutable references for cleaner Rust semantics.

use std::cell::Cell;

/// Index into the node store
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const NULL: NodeId = NodeId(u32::MAX);

    pub fn is_null(self) -> bool {
        self == Self::NULL
    }
}

/// A node in the type graph
///
/// Each node has a `link` field for union-find operations.
/// When `link` is NULL, this node is its own representative.
/// Otherwise, follow `link` to find the representative.
#[derive(Debug, Clone)]
pub enum Node {
    /// Type variable
    Var {
        name: String,
        id: u32,
        link: Cell<NodeId>,
    },
    /// Constant/base type
    Const {
        name: String,
    },
    /// Function type
    Arrow {
        domain: NodeId,
        codomain: NodeId,
        link: Cell<NodeId>,
    },
    /// Record type
    Record {
        row: NodeId,
        link: Cell<NodeId>,
    },
    /// Row: empty
    RowEmpty {
        link: Cell<NodeId>,
    },
    /// Row: extension with a field
    RowExtend {
        field: String,
        presence: NodeId, // Points to a Presence node
        rest: NodeId,
        link: Cell<NodeId>,
    },
    /// Row variable
    RowVar {
        name: String,
        id: u32,
        link: Cell<NodeId>,
    },
    /// Field presence: present with type
    Present {
        ty: NodeId,
        link: Cell<NodeId>,
    },
    /// Field presence: absent
    Absent {
        link: Cell<NodeId>,
    },
}

impl Node {
    /// Get the link field for union-find
    pub fn link(&self) -> &Cell<NodeId> {
        match self {
            Node::Var { link, .. } => link,
            Node::Const { .. } => {
                // Constants don't have links - they're always their own representative
                // We use a thread-local or just panic
                panic!("Const nodes don't have links")
            }
            Node::Arrow { link, .. } => link,
            Node::Record { link, .. } => link,
            Node::RowEmpty { link } => link,
            Node::RowExtend { link, .. } => link,
            Node::RowVar { link, .. } => link,
            Node::Present { link, .. } => link,
            Node::Absent { link } => link,
        }
    }

    /// Check if this node has a link (for union-find)
    pub fn has_link(&self) -> bool {
        !matches!(self, Node::Const { .. })
    }

    /// Create a new type variable node
    pub fn var(name: impl Into<String>, id: u32) -> Self {
        Node::Var {
            name: name.into(),
            id,
            link: Cell::new(NodeId::NULL),
        }
    }

    /// Create a new constant node
    pub fn constant(name: impl Into<String>) -> Self {
        Node::Const { name: name.into() }
    }

    /// Create a new arrow node
    pub fn arrow(domain: NodeId, codomain: NodeId) -> Self {
        Node::Arrow {
            domain,
            codomain,
            link: Cell::new(NodeId::NULL),
        }
    }

    /// Create a new record node
    pub fn record(row: NodeId) -> Self {
        Node::Record {
            row,
            link: Cell::new(NodeId::NULL),
        }
    }

    /// Create an empty row node
    pub fn row_empty() -> Self {
        Node::RowEmpty {
            link: Cell::new(NodeId::NULL),
        }
    }

    /// Create a row extension node
    pub fn row_extend(field: impl Into<String>, presence: NodeId, rest: NodeId) -> Self {
        Node::RowExtend {
            field: field.into(),
            presence,
            rest,
            link: Cell::new(NodeId::NULL),
        }
    }

    /// Create a row variable node
    pub fn row_var(name: impl Into<String>, id: u32) -> Self {
        Node::RowVar {
            name: name.into(),
            id,
            link: Cell::new(NodeId::NULL),
        }
    }

    /// Create a present field node
    pub fn present(ty: NodeId) -> Self {
        Node::Present {
            ty,
            link: Cell::new(NodeId::NULL),
        }
    }

    /// Create an absent field node
    pub fn absent() -> Self {
        Node::Absent {
            link: Cell::new(NodeId::NULL),
        }
    }

    /// Get a short description of the node type for debugging
    pub fn kind(&self) -> &'static str {
        match self {
            Node::Var { .. } => "Var",
            Node::Const { .. } => "Const",
            Node::Arrow { .. } => "Arrow",
            Node::Record { .. } => "Record",
            Node::RowEmpty { .. } => "RowEmpty",
            Node::RowExtend { .. } => "RowExtend",
            Node::RowVar { .. } => "RowVar",
            Node::Present { .. } => "Present",
            Node::Absent { .. } => "Absent",
        }
    }
}
