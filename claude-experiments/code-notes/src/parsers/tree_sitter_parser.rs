use anyhow::{anyhow, Result};
use tree_sitter::{Language, Parser, Tree};

/// Supported languages for parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
}

impl SupportedLanguage {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "js" | "jsx" => Some(Self::JavaScript),
            "ts" | "tsx" => Some(Self::TypeScript),
            _ => None,
        }
    }

    /// Get the tree-sitter language for this language
    pub fn tree_sitter_language(&self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::language(),
            Self::Python => tree_sitter_python::language(),
            Self::JavaScript => tree_sitter_javascript::language(),
            Self::TypeScript => tree_sitter_typescript::language_typescript(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
        }
    }
}

/// Wrapper around tree-sitter parser
pub struct CodeParser {
    parser: Parser,
    language: SupportedLanguage,
}

impl CodeParser {
    /// Create a new parser for the given language
    pub fn new(language: SupportedLanguage) -> Result<Self> {
        let mut parser = Parser::new();
        parser
            .set_language(&language.tree_sitter_language())
            .map_err(|e| anyhow!("Failed to set language: {}", e))?;

        Ok(Self { parser, language })
    }

    /// Parse source code
    pub fn parse(&mut self, source: &str) -> Result<Tree> {
        self.parser
            .parse(source, None)
            .ok_or_else(|| anyhow!("Failed to parse source code"))
    }

    /// Get the language this parser is configured for
    pub fn language(&self) -> SupportedLanguage {
        self.language
    }
}

/// Helper to find a node at a specific position
pub fn find_node_at_position<'a>(
    tree: &'a Tree,
    source: &str,
    line: usize,
    column: usize,
) -> Option<tree_sitter::Node<'a>> {
    let root = tree.root_node();

    // Convert line/column to byte offset
    let mut byte_offset = 0;
    for (idx, line_text) in source.lines().enumerate() {
        if idx == line {
            byte_offset += column.min(line_text.len());
            break;
        }
        byte_offset += line_text.len() + 1; // +1 for newline
    }

    find_smallest_node_at_byte(root, byte_offset)
}

fn find_smallest_node_at_byte<'a>(node: tree_sitter::Node<'a>, byte_offset: usize) -> Option<tree_sitter::Node<'a>> {
    if byte_offset < node.start_byte() || byte_offset > node.end_byte() {
        return None;
    }

    // Try to find a smaller child node
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(found) = find_smallest_node_at_byte(child, byte_offset) {
            return Some(found);
        }
    }

    // This is the smallest node containing the position
    Some(node)
}

/// Get all nodes of a specific kind in the tree
pub fn find_nodes_by_kind<'a>(
    node: tree_sitter::Node<'a>,
    kind: &str,
    results: &mut Vec<tree_sitter::Node<'a>>,
) {
    if node.kind() == kind {
        results.push(node);
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        find_nodes_by_kind(child, kind, results);
    }
}
