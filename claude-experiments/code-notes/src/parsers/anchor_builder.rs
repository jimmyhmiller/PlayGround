use anyhow::{anyhow, Result};
use tree_sitter::{Node, Tree};

use crate::models::{CodeAnchor, NoteAnchor};

/// Builds a CodeAnchor from a tree-sitter node
pub struct AnchorBuilder<'a> {
    source: &'a str,
    file_path: String,
}

impl<'a> AnchorBuilder<'a> {
    pub fn new(source: &'a str, file_path: String) -> Self {
        Self { source, file_path }
    }

    /// Build a CodeAnchor from a tree-sitter node
    pub fn build_anchor(&self, node: Node) -> Result<CodeAnchor> {
        let node_text = node
            .utf8_text(self.source.as_bytes())
            .map_err(|e| anyhow!("Failed to extract node text: {}", e))?
            .to_string();

        let start_position = node.start_position();
        let line_number = start_position.row;
        let column = start_position.column;

        let ast_path = self.build_ast_path(node);
        let context = self.build_context(node);

        Ok(CodeAnchor {
            file_path: self.file_path.clone(),
            node_kind: node.kind().to_string(),
            node_text,
            line_number,
            column,
            ast_path,
            context,
        })
    }

    /// Build a NoteAnchor with alternative anchor points for resilience
    pub fn build_note_anchor(&self, node: Node, commit_hash: String) -> Result<NoteAnchor> {
        let primary = self.build_anchor(node)?;
        let alternatives = self.build_alternative_anchors(node)?;

        Ok(NoteAnchor {
            primary,
            alternatives,
            commit_hash,
            is_valid: true,
            migration_history: Vec::new(),
        })
    }

    /// Build alternative anchors for redundancy
    /// These include parent nodes and sibling nodes
    fn build_alternative_anchors(&self, node: Node) -> Result<Vec<CodeAnchor>> {
        let mut alternatives = Vec::new();

        // Add parent node as alternative
        if let Some(parent) = node.parent() {
            if let Ok(anchor) = self.build_anchor(parent) {
                alternatives.push(anchor);
            }
        }

        // Add named siblings as alternatives
        if let Some(parent) = node.parent() {
            let mut cursor = parent.walk();
            for sibling in parent.named_children(&mut cursor) {
                if sibling.id() != node.id() {
                    if let Ok(anchor) = self.build_anchor(sibling) {
                        alternatives.push(anchor);
                    }
                }
            }
        }

        Ok(alternatives)
    }

    /// Build the AST path from root to this node
    fn build_ast_path(&self, node: Node) -> Vec<(String, usize)> {
        let mut path = Vec::new();
        let mut current = Some(node);

        while let Some(n) = current {
            if let Some(parent) = n.parent() {
                // Find this node's index among parent's children
                let mut cursor = parent.walk();
                let index = parent
                    .children(&mut cursor)
                    .position(|child| child.id() == n.id())
                    .unwrap_or(0);

                path.push((n.kind().to_string(), index));
                current = Some(parent);
            } else {
                // Root node
                path.push((n.kind().to_string(), 0));
                current = None;
            }
        }

        path.reverse();
        path
    }

    /// Build context (parent nodes' kinds)
    fn build_context(&self, node: Node) -> Vec<String> {
        let mut context = Vec::new();
        let mut current = node.parent();

        while let Some(n) = current {
            context.push(n.kind().to_string());
            current = n.parent();
        }

        context
    }
}

/// Attempts to find a matching node for an anchor in a new tree
pub struct AnchorMatcher<'a> {
    source: &'a str,
    tree: &'a Tree,
}

impl<'a> AnchorMatcher<'a> {
    pub fn new(source: &'a str, tree: &'a Tree) -> Self {
        Self { source, tree }
    }

    /// Find the best matching node for a given anchor
    pub fn find_match(&self, anchor: &CodeAnchor) -> Option<(Node<'a>, f64)> {
        let root = self.tree.root_node();
        let mut candidates = Vec::new();

        // First, try to find nodes of the same kind
        self.find_candidates(root, &anchor.node_kind, &mut candidates);

        // Score each candidate
        let mut best_match = None;
        let mut best_score = 0.0;

        let builder = AnchorBuilder::new(self.source, anchor.file_path.clone());

        for candidate_node in candidates {
            if let Ok(candidate_anchor) = builder.build_anchor(candidate_node) {
                let score = anchor.match_confidence(&candidate_anchor);
                if score > best_score {
                    best_score = score;
                    best_match = Some(candidate_node);
                }
            }
        }

        best_match.map(|node| (node, best_score))
    }

    fn find_candidates(&self, node: Node<'a>, kind: &str, results: &mut Vec<Node<'a>>) {
        if node.kind() == kind {
            results.push(node);
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.find_candidates(child, kind, results);
        }
    }
}
