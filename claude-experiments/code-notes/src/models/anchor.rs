use serde::{Deserialize, Serialize};

/// Represents a way to anchor a note to a specific location in code
/// This uses tree-sitter concepts to be resilient across refactorings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodeAnchor {
    /// The file path relative to repository root
    pub file_path: String,

    /// The kind of AST node (e.g., "identifier", "function_definition")
    pub node_kind: String,

    /// The text content at the anchor point (for verification)
    pub node_text: String,

    /// Line number where the node starts (for quick lookup)
    pub line_number: usize,

    /// Column number where the node starts
    pub column: usize,

    /// Path through the AST from root to this node
    /// Each element is (node_kind, child_index)
    pub ast_path: Vec<(String, usize)>,

    /// Context around the node (parent nodes' kinds)
    pub context: Vec<String>,

    /// Semantic identifier (function name, class name, variable name, etc.)
    /// Extracted from the node for robust matching across refactorings
    pub semantic_id: Option<String>,

    /// Normalized version of node_text (whitespace removed, trimmed)
    /// Used for matching that ignores formatting changes
    pub normalized_text: String,
}

impl CodeAnchor {
    /// Normalize text by removing formatting differences
    /// This makes matching robust to whitespace-only changes
    pub fn normalize_text(text: &str) -> String {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string()
    }

    /// Calculate a confidence score for matching this anchor at a new location
    /// Returns 0.0 (no match) to 1.0 (perfect match)
    pub fn match_confidence(&self, candidate: &CodeAnchor) -> f64 {
        // Node kind must match as prerequisite
        if self.node_kind != candidate.node_kind {
            return 0.0;
        }

        // Strategy 1: Semantic identifier match (highest priority)
        // If both have semantic IDs and they match, this is a strong signal
        if let (Some(self_id), Some(cand_id)) = (&self.semantic_id, &candidate.semantic_id)
            && self_id == cand_id {
                // Same semantic identity - check if content is similar too
                if self.normalized_text == candidate.normalized_text {
                    return 1.0; // Perfect match - same name and same content
                } else {
                    return 0.85; // Same name but content changed (e.g., function body changed)
                }
            }

        // Strategy 2: Weighted scoring for structural similarity
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Normalized text match (ignores whitespace)
        if self.normalized_text == candidate.normalized_text {
            score += 10.0;
        } else if self.node_text == candidate.node_text {
            // Exact text match as fallback
            score += 8.0;
        }
        weight_sum += 10.0;

        // Node kind already checked above
        score += 5.0;
        weight_sum += 5.0;

        // AST path similarity
        let path_similarity = self.ast_path_similarity(&candidate.ast_path);
        score += path_similarity * 3.0;
        weight_sum += 3.0;

        // Context similarity
        let context_similarity = self.context_similarity(&candidate.context);
        score += context_similarity * 2.0;
        weight_sum += 2.0;

        score / weight_sum
    }

    fn ast_path_similarity(&self, other: &[(String, usize)]) -> f64 {
        if self.ast_path.is_empty() && other.is_empty() {
            return 1.0;
        }

        let min_len = self.ast_path.len().min(other.len());
        let max_len = self.ast_path.len().max(other.len());

        if max_len == 0 {
            return 1.0;
        }

        let mut matches = 0;
        for (i, item) in other.iter().enumerate().take(min_len) {
            if self.ast_path[i] == *item {
                matches += 1;
            }
        }

        matches as f64 / max_len as f64
    }

    fn context_similarity(&self, other: &[String]) -> f64 {
        if self.context.is_empty() && other.is_empty() {
            return 1.0;
        }

        let min_len = self.context.len().min(other.len());
        let max_len = self.context.len().max(other.len());

        if max_len == 0 {
            return 1.0;
        }

        let mut matches = 0;
        for (i, item) in other.iter().enumerate().take(min_len) {
            if self.context[i] == *item {
                matches += 1;
            }
        }

        matches as f64 / max_len as f64
    }
}

/// Represents where a note is anchored in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteAnchor {
    /// The primary anchor point
    pub primary: CodeAnchor,

    /// Alternative anchor points (for redundancy)
    pub alternatives: Vec<CodeAnchor>,

    /// The git commit hash when this anchor was created
    pub commit_hash: String,

    /// Whether this anchor has been successfully migrated to the current commit
    pub is_valid: bool,

    /// History of migrations
    pub migration_history: Vec<MigrationRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRecord {
    pub from_commit: String,
    pub to_commit: String,
    pub timestamp: i64,
    pub success: bool,
    pub confidence: Option<f64>,
}
