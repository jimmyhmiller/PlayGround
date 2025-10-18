use crate::ast::{SExpr, Span};
use anyhow::{anyhow, Context, Result};
use tree_sitter::{Node, Parser, Tree};

pub struct ClojureParser {
    parser: Parser,
}

impl ClojureParser {
    const MAX_DEPTH: usize = 256;

    pub fn new() -> Result<Self> {
        let mut parser = Parser::new();
        let language: tree_sitter::Language = tree_sitter_clojure::LANGUAGE.into();
        parser
            .set_language(&language)
            .context("Failed to load Clojure grammar")?;
        Ok(ClojureParser { parser })
    }

    pub fn parse(&mut self, source: &str) -> Result<Tree> {
        self.parser
            .parse(source, None)
            .context("Failed to parse source code")
    }

    pub fn parse_to_sexpr(&mut self, source: &str) -> Result<Vec<SExpr>> {
        let mut depth = 0usize;
        for ch in source.chars() {
            match ch {
                '(' | '[' | '{' => {
                    depth += 1;
                    if depth > Self::MAX_DEPTH {
                        return Err(anyhow!("Source exceeds maximum nesting depth of {}", Self::MAX_DEPTH));
                    }
                }
                ')' | ']' | '}' => {
                    if depth > 0 {
                        depth -= 1;
                    }
                }
                _ => {}
            }
        }

        let tree = self.parse(source)?;
        let root = tree.root_node();
        let mut forms = Vec::new();

        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if let Some(sexpr) = self.node_to_sexpr(&child, source, 0)? {
                forms.push(sexpr);
            }
        }

        Ok(forms)
    }

    fn node_to_sexpr(&self, node: &Node, source: &str, depth: usize) -> Result<Option<SExpr>> {
        if depth >= Self::MAX_DEPTH {
            return Err(anyhow!("Exceeded maximum AST depth of {}", Self::MAX_DEPTH));
        }

        let span = Span::from_node(node);

        match node.kind() {
            "list_lit" => {
                let mut children = Vec::new();
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "(" && child.kind() != ")" {
                        if let Some(sexpr) = self.node_to_sexpr(&child, source, depth + 1)? {
                            children.push(sexpr);
                        }
                    }
                }
                Ok(Some(SExpr::List {
                    span,
                    open: '(',
                    close: ')',
                    children,
                }))
            }
            "vec_lit" => {
                let mut children = Vec::new();
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "[" && child.kind() != "]" {
                        if let Some(sexpr) = self.node_to_sexpr(&child, source, depth + 1)? {
                            children.push(sexpr);
                        }
                    }
                }
                Ok(Some(SExpr::List {
                    span,
                    open: '[',
                    close: ']',
                    children,
                }))
            }
            "map_lit" => {
                let mut children = Vec::new();
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "{" && child.kind() != "}" {
                        if let Some(sexpr) = self.node_to_sexpr(&child, source, depth + 1)? {
                            children.push(sexpr);
                        }
                    }
                }
                Ok(Some(SExpr::List {
                    span,
                    open: '{',
                    close: '}',
                    children,
                }))
            }
            "set_lit" => {
                let mut children = Vec::new();
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "#{" && child.kind() != "}" {
                        if let Some(sexpr) = self.node_to_sexpr(&child, source, depth + 1)? {
                            children.push(sexpr);
                        }
                    }
                }
                Ok(Some(SExpr::List {
                    span,
                    open: '{',
                    close: '}',
                    children,
                }))
            }
            "str_lit" => {
                if let Some(text) = node.utf8_text(source.as_bytes()).ok() {
                    Ok(Some(SExpr::String {
                        span,
                        value: text.to_string(),
                    }))
                } else {
                    Ok(None)
                }
            }
            "comment" => {
                if let Some(raw) = node.utf8_text(source.as_bytes()).ok() {
                    let text = raw.trim_end_matches(|c| c == '\n' || c == '\r').to_string();
                    Ok(Some(SExpr::Comment { span, text }))
                } else {
                    Ok(None)
                }
            }
            "sym_lit" | "kwd_lit" | "num_lit" | "bool_lit" | "nil_lit" | "char_lit" => {
                if let Some(text) = node.utf8_text(source.as_bytes()).ok() {
                    Ok(Some(SExpr::Atom {
                        span,
                        value: text.to_string(),
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => {
                // For other node types, try to recursively parse children
                if node.child_count() > 0 {
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        if let Some(sexpr) = self.node_to_sexpr(&child, source, depth + 1)? {
                            return Ok(Some(sexpr));
                        }
                    }
                }
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_list() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("(+ 1 2)").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::List { children, .. } => {
                assert_eq!(children.len(), 3);
            }
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_parse_nested_list() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("(defn foo [x] (+ x 1))").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::List { children, .. } => {
                assert_eq!(children.len(), 4);
            }
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_parse_vector() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("[1 2 3]").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::List { open, close, children, .. } => {
                assert_eq!(*open, '[');
                assert_eq!(*close, ']');
                assert_eq!(children.len(), 3);
            }
            _ => panic!("Expected vector"),
        }
    }

    #[test]
    fn test_parse_map() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("{:a 1 :b 2}").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::List { open, close, children, .. } => {
                assert_eq!(*open, '{');
                assert_eq!(*close, '}');
                assert_eq!(children.len(), 4);
            }
            _ => panic!("Expected map"),
        }
    }

    #[test]
    fn test_parse_empty_structures() {
        let mut parser = ClojureParser::new().unwrap();
        
        let forms = parser.parse_to_sexpr("()").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::List { children, .. } => assert_eq!(children.len(), 0),
            _ => panic!("Expected empty list"),
        }

        let forms = parser.parse_to_sexpr("[]").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::List { open, children, .. } => {
                assert_eq!(*open, '[');
                assert_eq!(children.len(), 0);
            }
            _ => panic!("Expected empty vector"),
        }

        let forms = parser.parse_to_sexpr("{}").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::List { open, children, .. } => {
                assert_eq!(*open, '{');
                assert_eq!(children.len(), 0);
            }
            _ => panic!("Expected empty map"),
        }
    }

    #[test]
    fn test_parse_strings() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(r#""hello world""#).unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::String { value, .. } => {
                assert_eq!(value, r#""hello world""#);
            }
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_parse_escaped_strings() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(r#""hello \"world\"""#).unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::String { value, .. } => {
                assert_eq!(value, r#""hello \"world\"""#);
            }
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_parse_multiline_string() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("\"hello\nworld\"").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::String { value, .. } => {
                assert_eq!(value, "\"hello\nworld\"");
            }
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_parse_comments() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("; this is a comment\n(+ 1 2)").unwrap();
        assert_eq!(forms.len(), 2);
        
        match &forms[0] {
            SExpr::Comment { text, .. } => {
                assert_eq!(text, "; this is a comment");
            }
            _ => panic!("Expected comment"),
        }

        match &forms[1] {
            SExpr::List { .. } => {},
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_parse_atoms() {
        let mut parser = ClojureParser::new().unwrap();

        // Symbol
        let forms = parser.parse_to_sexpr("symbol").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::Atom { value, .. } => assert_eq!(value, "symbol"),
            _ => panic!("Expected atom"),
        }

        // Keyword
        let forms = parser.parse_to_sexpr(":keyword").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::Atom { value, .. } => assert_eq!(value, ":keyword"),
            _ => panic!("Expected atom"),
        }

        // Number
        let forms = parser.parse_to_sexpr("42").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::Atom { value, .. } => assert_eq!(value, "42"),
            _ => panic!("Expected atom"),
        }

        // Boolean
        let forms = parser.parse_to_sexpr("true").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::Atom { value, .. } => assert_eq!(value, "true"),
            _ => panic!("Expected atom"),
        }

        // Nil
        let forms = parser.parse_to_sexpr("nil").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::Atom { value, .. } => assert_eq!(value, "nil"),
            _ => panic!("Expected atom"),
        }

        // Character
        let forms = parser.parse_to_sexpr("\\a").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::Atom { value, .. } => assert_eq!(value, "\\a"),
            _ => panic!("Expected atom"),
        }
    }

    #[test]
    fn test_parse_deeply_nested() {
        let mut parser = ClojureParser::new().unwrap();
        let source = "(((((inner)))))";
        let forms = parser.parse_to_sexpr(source).unwrap();
        assert_eq!(forms.len(), 1);

        // Navigate down the nesting
        let mut current = &forms[0];
        for _ in 0..5 {
            match current {
                SExpr::List { children, .. } => {
                    assert_eq!(children.len(), 1);
                    current = &children[0];
                }
                _ => panic!("Expected nested list"),
            }
        }

        // Should find the inner atom
        match current {
            SExpr::Atom { value, .. } => assert_eq!(value, "inner"),
            _ => panic!("Expected inner atom"),
        }
    }

    #[test]
    fn test_parse_mixed_structures() {
        let mut parser = ClojureParser::new().unwrap();
        let source = r#"(defn example [x y]
  {:result (+ x y)
   :data [x y :sum]})"#;
        let forms = parser.parse_to_sexpr(source).unwrap();
        assert_eq!(forms.len(), 1);

        match &forms[0] {
            SExpr::List { children, .. } => {
                assert_eq!(children.len(), 4); // defn, example, [x y], body
                
                // Check the vector
                match &children[2] {
                    SExpr::List { open, children, .. } => {
                        assert_eq!(*open, '[');
                        assert_eq!(children.len(), 2);
                    }
                    _ => panic!("Expected vector"),
                }

                // Check the map
                match &children[3] {
                    SExpr::List { open, children, .. } => {
                        assert_eq!(*open, '{');
                        assert_eq!(children.len(), 4); // :result, (+...), :data, [...]
                    }
                    _ => panic!("Expected map"),
                }
            }
            _ => panic!("Expected top-level list"),
        }
    }

    #[test]
    fn test_parse_multiple_top_level_forms() {
        let mut parser = ClojureParser::new().unwrap();
        let source = "(defn f1 []) (defn f2 []) (defn f3 [])";
        let forms = parser.parse_to_sexpr(source).unwrap();
        assert_eq!(forms.len(), 3);

        for form in &forms {
            match form {
                SExpr::List { children, .. } => {
                    assert_eq!(children.len(), 3); // defn, name, []
                    match &children[0] {
                        SExpr::Atom { value, .. } => assert_eq!(value, "defn"),
                        _ => panic!("Expected defn"),
                    }
                }
                _ => panic!("Expected list"),
            }
        }
    }

    #[test]
    fn test_parse_with_whitespace() {
        let mut parser = ClojureParser::new().unwrap();
        let source = "  (  +   1    2  )  ";
        let forms = parser.parse_to_sexpr(source).unwrap();
        assert_eq!(forms.len(), 1);

        match &forms[0] {
            SExpr::List { children, .. } => {
                assert_eq!(children.len(), 3);
            }
            _ => panic!("Expected list"),
        }
    }

    #[test] 
    fn test_parse_empty_input() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("").unwrap();
        assert_eq!(forms.len(), 0);
    }

    #[test]
    fn test_parse_whitespace_only() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("   \n  \t  ").unwrap();
        assert_eq!(forms.len(), 0);
    }

    #[test]
    fn test_parse_set_literal() {
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr("#{1 2 3}").unwrap();
        assert_eq!(forms.len(), 1);
        match &forms[0] {
            SExpr::List { open, close, children, .. } => {
                assert_eq!(*open, '{');
                assert_eq!(*close, '}');
                assert_eq!(children.len(), 3);
            }
            _ => panic!("Expected set"),
        }
    }

    #[test]
    fn test_parse_malformed_still_attempts() {
        let mut parser = ClojureParser::new().unwrap();
        // tree-sitter should still parse this and return what it can
        let result = parser.parse_to_sexpr("(defn incomplete");
        assert!(result.is_ok()); // Should not panic
    }
}
