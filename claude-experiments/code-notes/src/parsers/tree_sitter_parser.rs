use anyhow::{anyhow, Result};
use tree_sitter::{Language, Parser, Tree};
use crate::parsers::language_registry::LanguageRegistry;

/// Wrapper around tree-sitter parser with dynamic language loading
pub struct CodeParser {
    parser: Parser,
    #[allow(dead_code)]
    language_name: String,
}

impl CodeParser {
    /// Create a new parser for the given language name
    pub fn new(language_name: &str, registry: &mut LanguageRegistry) -> Result<Self> {
        // Try to load from registry first
        let language = registry.get_language(language_name)
            .or_else(|_| {
                // Fall back to built-in languages
                Self::get_builtin_language(language_name)
            })?;

        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .map_err(|e| anyhow!("Failed to set language: {}", e))?;

        Ok(Self {
            parser,
            language_name: language_name.to_string(),
        })
    }

    /// Create a parser from a file extension
    pub fn from_extension(ext: &str, registry: &mut LanguageRegistry) -> Result<Self> {
        // Try registry first
        let language_name = registry.language_from_extension(ext)
            .or_else(|_| {
                // Fall back to built-in extension mapping
                Self::builtin_extension_to_language(ext)
                    .ok_or_else(|| anyhow!("No language found for extension '.{}'", ext))
            })?;

        Self::new(&language_name, registry)
    }

    /// Get a built-in language (for backward compatibility)
    fn get_builtin_language(name: &str) -> Result<Language> {
        match name {
            "rust" => Ok(tree_sitter_rust::language()),
            "python" => Ok(tree_sitter_python::language()),
            "javascript" => Ok(tree_sitter_javascript::language()),
            "typescript" => Ok(tree_sitter_typescript::language_typescript()),
            _ => Err(anyhow!("Language '{}' is not a built-in language. Install it with: code-notes lang install {}", name, name))
        }
    }

    /// Map file extension to built-in language name
    fn builtin_extension_to_language(ext: &str) -> Option<String> {
        match ext {
            "rs" => Some("rust".to_string()),
            "py" => Some("python".to_string()),
            "js" | "jsx" => Some("javascript".to_string()),
            "ts" | "tsx" => Some("typescript".to_string()),
            _ => None,
        }
    }

    /// Get the language name this parser is configured for
    #[allow(dead_code)]
    pub fn language_name(&self) -> &str {
        &self.language_name
    }

    /// Parse source code
    pub fn parse(&mut self, source: &str) -> Result<Tree> {
        self.parser
            .parse(source, None)
            .ok_or_else(|| anyhow!("Failed to parse source code"))
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

    let node = find_smallest_node_at_byte(root, byte_offset)?;

    // Find the most meaningful parent node
    // Prefer larger structural nodes over small modifiers/keywords
    find_best_anchor_node(node, column)
}

/// Find the best node to anchor to by walking up the tree
/// Prefers meaningful structural nodes over modifiers/keywords
fn find_best_anchor_node<'a>(mut node: tree_sitter::Node<'a>, column: usize) -> Option<tree_sitter::Node<'a>> {
    // If column is 0 (start of line), prefer larger parent nodes
    // This helps when anchoring to "pub fn foo" - we want function_item, not visibility_modifier
    if column == 0 {
        while let Some(parent) = node.parent() {
            // Walk up to find function_item, struct_item, impl_item, etc.
            // But stop at the root level (source_file, module, program, etc.)
            if matches!(parent.kind(), "source_file" | "module" | "program") {
                break;
            }

            // Prefer these structural nodes
            if matches!(
                parent.kind(),
                "function_item" | "struct_item" | "enum_item" | "impl_item" |
                "trait_item" | "mod_item" | "const_item" | "static_item" |
                "type_item" | "use_declaration" | "function_definition" |
                "class_definition" | "function_declaration" | "method_definition"
            ) {
                return Some(parent);
            }

            node = parent;
        }
    }

    Some(node)
}

fn find_smallest_node_at_byte<'a>(node: tree_sitter::Node<'a>, byte_offset: usize) -> Option<tree_sitter::Node<'a>> {
    if byte_offset < node.start_byte() || byte_offset > node.end_byte() {
        return None;
    }

    // Try to find a smaller NAMED child node first
    let mut cursor = node.walk();
    let mut best_named_child = None;
    for child in node.named_children(&mut cursor) {
        if let Some(found) = find_smallest_node_at_byte(child, byte_offset) {
            best_named_child = Some(found);
            break;
        }
    }

    if let Some(named) = best_named_child {
        return Some(named);
    }

    // If no named child found, try anonymous children
    let mut cursor2 = node.walk();
    for child in node.children(&mut cursor2) {
        if let Some(found) = find_smallest_node_at_byte(child, byte_offset) {
            return Some(found);
        }
    }

    // This node itself contains the position
    // Return it only if it's named
    if node.is_named() {
        Some(node)
    } else {
        None
    }
}

/// Get all nodes of a specific kind in the tree
#[allow(dead_code)]
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
