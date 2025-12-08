// Simple DOT file parser
//
// Supports:
// - digraph NAME { ... }
// - node declarations: A [label="...", loopheader="true", backedge="true"]
// - edge declarations: A -> B [label="..."]
// - Comments: // and /* */

use super::schema::{DotGraph, DotNode, DotEdge};
use std::collections::HashMap;

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse error at line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse a DOT file string into a DotGraph
pub fn parse_dot(input: &str) -> Result<DotGraph, ParseError> {
    let mut parser = DotParser::new(input);
    parser.parse()
}

struct DotParser<'a> {
    input: &'a str,
    pos: usize,
    line: usize,
}

impl<'a> DotParser<'a> {
    fn new(input: &'a str) -> Self {
        DotParser {
            input,
            pos: 0,
            line: 1,
        }
    }

    fn parse(&mut self) -> Result<DotGraph, ParseError> {
        self.skip_whitespace_and_comments();

        // Expect "digraph" or "graph"
        let keyword = self.expect_identifier()?;
        if keyword != "digraph" && keyword != "graph" {
            return Err(self.error(format!("Expected 'digraph' or 'graph', found '{}'", keyword)));
        }

        self.skip_whitespace_and_comments();

        // Graph name (optional)
        let name = if self.peek() != Some('{') {
            let name = self.expect_identifier_or_string()?;
            self.skip_whitespace_and_comments();
            name
        } else {
            "graph".to_string()
        };

        // Expect '{'
        self.expect_char('{')?;

        let mut graph = DotGraph::new(name);
        let mut node_map: HashMap<String, usize> = HashMap::new();

        // Parse statements until '}'
        loop {
            self.skip_whitespace_and_comments();

            if self.peek() == Some('}') {
                self.advance();
                break;
            }

            if self.is_eof() {
                return Err(self.error("Unexpected end of file, expected '}'".to_string()));
            }

            // Parse statement
            self.parse_statement(&mut graph, &mut node_map)?;
        }

        Ok(graph)
    }

    fn parse_statement(&mut self, graph: &mut DotGraph, node_map: &mut HashMap<String, usize>) -> Result<(), ParseError> {
        // Skip graph/node/edge default attributes
        let id = self.expect_identifier_or_string()?;

        // Skip subgraph keyword
        if id == "subgraph" {
            self.skip_subgraph()?;
            return Ok(());
        }

        // Skip graph/node/edge defaults
        if id == "graph" || id == "node" || id == "edge" {
            self.skip_whitespace_and_comments();
            if self.peek() == Some('[') {
                self.skip_attributes()?;
            }
            self.skip_optional_semicolon();
            return Ok(());
        }

        self.skip_whitespace_and_comments();

        // Check if this is an edge (->)
        if self.peek_str("->") || self.peek_str("--") {
            // Edge declaration
            self.advance();
            self.advance();
            self.skip_whitespace_and_comments();

            let target = self.expect_identifier_or_string()?;
            self.skip_whitespace_and_comments();

            // Ensure both nodes exist
            self.ensure_node(graph, node_map, &id);
            self.ensure_node(graph, node_map, &target);

            // Parse optional attributes
            let mut edge = DotEdge::new(id, target);
            if self.peek() == Some('[') {
                let attrs = self.parse_attributes()?;
                if let Some(label) = attrs.get("label") {
                    edge.label = Some(label.clone());
                }
            }

            graph.edges.push(edge);
        } else {
            // Node declaration
            self.ensure_node(graph, node_map, &id);

            // Parse optional attributes
            if self.peek() == Some('[') {
                let attrs = self.parse_attributes()?;
                let node_idx = *node_map.get(&id).unwrap();
                let node = &mut graph.nodes[node_idx];

                if let Some(label) = attrs.get("label") {
                    node.label = Some(label.clone());
                }
                if attrs.get("loopheader").map(|v| v == "true").unwrap_or(false) {
                    node.is_loop_header = true;
                }
                if attrs.get("backedge").map(|v| v == "true").unwrap_or(false) {
                    node.is_backedge = true;
                }
            }
        }

        self.skip_optional_semicolon();
        Ok(())
    }

    fn ensure_node(&mut self, graph: &mut DotGraph, node_map: &mut HashMap<String, usize>, id: &str) {
        if !node_map.contains_key(id) {
            let idx = graph.nodes.len();
            graph.nodes.push(DotNode::new(id.to_string()));
            node_map.insert(id.to_string(), idx);
        }
    }

    fn parse_attributes(&mut self) -> Result<HashMap<String, String>, ParseError> {
        let mut attrs = HashMap::new();

        self.expect_char('[')?;
        self.skip_whitespace_and_comments();

        while self.peek() != Some(']') && !self.is_eof() {
            // Parse key
            let key = self.expect_identifier()?;
            self.skip_whitespace_and_comments();

            // Expect '='
            self.expect_char('=')?;
            self.skip_whitespace_and_comments();

            // Parse value
            let value = self.expect_value()?;
            attrs.insert(key, value);

            self.skip_whitespace_and_comments();

            // Skip optional comma or semicolon
            if self.peek() == Some(',') || self.peek() == Some(';') {
                self.advance();
                self.skip_whitespace_and_comments();
            }
        }

        self.expect_char(']')?;
        Ok(attrs)
    }

    fn skip_attributes(&mut self) -> Result<(), ParseError> {
        self.expect_char('[')?;
        let mut depth = 1;
        while depth > 0 && !self.is_eof() {
            match self.peek() {
                Some('[') => depth += 1,
                Some(']') => depth -= 1,
                Some('"') => {
                    self.advance();
                    self.skip_string_content()?;
                    continue;
                }
                _ => {}
            }
            self.advance();
        }
        Ok(())
    }

    fn skip_subgraph(&mut self) -> Result<(), ParseError> {
        self.skip_whitespace_and_comments();

        // Skip optional name
        if self.peek() != Some('{') {
            self.expect_identifier_or_string()?;
            self.skip_whitespace_and_comments();
        }

        // Skip the subgraph body
        self.expect_char('{')?;
        let mut depth = 1;
        while depth > 0 && !self.is_eof() {
            match self.peek() {
                Some('{') => depth += 1,
                Some('}') => depth -= 1,
                Some('"') => {
                    self.advance();
                    self.skip_string_content()?;
                    continue;
                }
                _ => {}
            }
            self.advance();
        }
        Ok(())
    }

    fn expect_identifier(&mut self) -> Result<String, ParseError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        if self.pos == start {
            return Err(self.error("Expected identifier".to_string()));
        }

        Ok(self.input[start..self.pos].to_string())
    }

    fn expect_identifier_or_string(&mut self) -> Result<String, ParseError> {
        if self.peek() == Some('"') {
            self.advance();
            let s = self.expect_string_content()?;
            self.expect_char('"')?;
            Ok(s)
        } else {
            self.expect_identifier()
        }
    }

    fn expect_value(&mut self) -> Result<String, ParseError> {
        if self.peek() == Some('"') {
            self.advance();
            let s = self.expect_string_content()?;
            self.expect_char('"')?;
            Ok(s)
        } else {
            // Unquoted value (identifier or number)
            let start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' {
                    self.advance();
                } else {
                    break;
                }
            }
            if self.pos == start {
                return Err(self.error("Expected value".to_string()));
            }
            Ok(self.input[start..self.pos].to_string())
        }
    }

    fn expect_string_content(&mut self) -> Result<String, ParseError> {
        let mut result = String::new();
        while let Some(c) = self.peek() {
            match c {
                '"' => break,
                '\\' => {
                    self.advance();
                    if let Some(escaped) = self.peek() {
                        result.push(match escaped {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '\\' => '\\',
                            '"' => '"',
                            _ => escaped,
                        });
                        self.advance();
                    }
                }
                '\n' => {
                    self.line += 1;
                    result.push(c);
                    self.advance();
                }
                _ => {
                    result.push(c);
                    self.advance();
                }
            }
        }
        Ok(result)
    }

    fn skip_string_content(&mut self) -> Result<(), ParseError> {
        while let Some(c) = self.peek() {
            match c {
                '"' => return Ok(()),
                '\\' => {
                    self.advance();
                    self.advance(); // Skip escaped char
                }
                '\n' => {
                    self.line += 1;
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }
        Ok(())
    }

    fn expect_char(&mut self, expected: char) -> Result<(), ParseError> {
        if self.peek() == Some(expected) {
            self.advance();
            Ok(())
        } else {
            Err(self.error(format!("Expected '{}', found '{:?}'", expected, self.peek())))
        }
    }

    fn skip_optional_semicolon(&mut self) {
        self.skip_whitespace_and_comments();
        if self.peek() == Some(';') {
            self.advance();
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while let Some(c) = self.peek() {
                if c == '\n' {
                    self.line += 1;
                    self.advance();
                } else if c.is_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }

            // Skip // comments
            if self.peek_str("//") {
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            // Skip /* */ comments
            if self.peek_str("/*") {
                self.advance();
                self.advance();
                while !self.is_eof() {
                    if self.peek() == Some('\n') {
                        self.line += 1;
                    }
                    if self.peek_str("*/") {
                        self.advance();
                        self.advance();
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            // Skip # comments (common in some DOT variants)
            if self.peek() == Some('#') {
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            break;
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn peek_str(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.pos += c.len_utf8();
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn error(&self, message: String) -> ParseError {
        ParseError {
            message,
            line: self.line,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        let input = r#"
            digraph test {
                A -> B
                B -> C
            }
        "#;

        let graph = parse_dot(input).unwrap();
        assert_eq!(graph.name, "test");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn test_node_attributes() {
        let input = r#"
            digraph test {
                A [label="Entry", loopheader="true"]
                B [label="Loop Body", backedge="true"]
                A -> B
            }
        "#;

        let graph = parse_dot(input).unwrap();
        assert_eq!(graph.nodes[0].label, Some("Entry".to_string()));
        assert!(graph.nodes[0].is_loop_header);
        assert!(graph.nodes[1].is_backedge);
    }

    #[test]
    fn test_comments() {
        let input = r#"
            // This is a comment
            digraph test {
                A -> B // inline comment
                /* block comment */
                B -> C
            }
        "#;

        let graph = parse_dot(input).unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2);
    }
}
