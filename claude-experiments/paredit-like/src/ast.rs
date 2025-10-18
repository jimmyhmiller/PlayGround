use tree_sitter::Node;

#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl Position {
    pub fn from_point(point: tree_sitter::Point, offset: usize) -> Self {
        Position {
            line: point.row + 1, // Convert to 1-indexed
            column: point.column,
            offset,
        }
    }

    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Position { line, column, offset }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

impl Span {
    pub fn from_node(node: &Node) -> Self {
        Span {
            start: Position::from_point(node.start_position(), node.start_byte()),
            end: Position::from_point(node.end_position(), node.end_byte()),
        }
    }

    pub fn new(start: Position, end: Position) -> Self {
        Span { start, end }
    }

    pub fn contains_line(&self, line: usize) -> bool {
        self.start.line <= line && line <= self.end.line
    }

    pub fn contains_offset(&self, offset: usize) -> bool {
        self.start.offset <= offset && offset <= self.end.offset
    }

    pub fn len(&self) -> usize {
        self.end.offset - self.start.offset
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    List {
        span: Span,
        open: char,
        close: char,
        children: Vec<SExpr>,
    },
    Atom {
        span: Span,
        value: String,
    },
    String {
        span: Span,
        value: String,
    },
    Comment {
        span: Span,
        text: String,
    },
}

impl SExpr {
    pub fn span(&self) -> &Span {
        match self {
            SExpr::List { span, .. } => span,
            SExpr::Atom { span, .. } => span,
            SExpr::String { span, .. } => span,
            SExpr::Comment { span, .. } => span,
        }
    }

    pub fn find_at_line(&self, line: usize) -> Option<&SExpr> {
        if !self.span().contains_line(line) {
            return None;
        }

        if let SExpr::List { children, .. } = self {
            for child in children {
                if let Some(found) = child.find_at_line(line) {
                    return Some(found);
                }
            }
        }

        Some(self)
    }

    pub fn find_deepest_at_line(&self, line: usize) -> Option<&SExpr> {
        if !self.span().contains_line(line) {
            return None;
        }

        if let SExpr::List { children, .. } = self {
            for child in children {
                if let Some(found) = child.find_deepest_at_line(line) {
                    return Some(found);
                }
            }
        }

        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let pos = Position::new(1, 5, 10);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 5);
        assert_eq!(pos.offset, 10);
    }

    #[test]
    fn test_position_from_point() {
        let point = tree_sitter::Point { row: 0, column: 5 };
        let pos = Position::from_point(point, 10);
        assert_eq!(pos.line, 1); // Should be 1-indexed
        assert_eq!(pos.column, 5);
        assert_eq!(pos.offset, 10);
    }

    #[test]
    fn test_span_creation() {
        let start = Position::new(1, 0, 0);
        let end = Position::new(1, 10, 10);
        let span = Span::new(start, end);
        assert_eq!(span.len(), 10);
    }

    #[test]
    fn test_span_contains_line() {
        let start = Position::new(2, 0, 10);
        let end = Position::new(4, 5, 50);
        let span = Span::new(start, end);

        assert!(!span.contains_line(1));
        assert!(span.contains_line(2));
        assert!(span.contains_line(3));
        assert!(span.contains_line(4));
        assert!(!span.contains_line(5));
    }

    #[test]
    fn test_span_contains_offset() {
        let start = Position::new(1, 0, 10);
        let end = Position::new(1, 10, 20);
        let span = Span::new(start, end);

        assert!(!span.contains_offset(5));
        assert!(span.contains_offset(10));
        assert!(span.contains_offset(15));
        assert!(span.contains_offset(20));
        assert!(!span.contains_offset(25));
    }

    #[test]
    fn test_sexpr_span() {
        let span = Span::new(Position::new(1, 0, 0), Position::new(1, 5, 5));
        let atom = SExpr::Atom {
            span: span.clone(),
            value: "test".to_string(),
        };
        assert_eq!(atom.span(), &span);

        let string = SExpr::String {
            span: span.clone(),
            value: "\"test\"".to_string(),
        };
        assert_eq!(string.span(), &span);

        let comment = SExpr::Comment {
            span: span.clone(),
            text: "; test".to_string(),
        };
        assert_eq!(comment.span(), &span);

        let list = SExpr::List {
            span: span.clone(),
            open: '(',
            close: ')',
            children: vec![],
        };
        assert_eq!(list.span(), &span);
    }

    #[test]
    fn test_find_at_line_simple() {
        let span = Span::new(Position::new(2, 0, 10), Position::new(2, 5, 15));
        let atom = SExpr::Atom {
            span,
            value: "test".to_string(),
        };

        assert!(atom.find_at_line(2).is_some());
        assert!(atom.find_at_line(1).is_none());
        assert!(atom.find_at_line(3).is_none());
    }

    #[test]
    fn test_find_at_line_nested() {
        let outer_span = Span::new(Position::new(1, 0, 0), Position::new(3, 5, 25));
        let inner_span = Span::new(Position::new(2, 2, 10), Position::new(2, 7, 15));
        
        let inner_atom = SExpr::Atom {
            span: inner_span,
            value: "inner".to_string(),
        };

        let outer_list = SExpr::List {
            span: outer_span,
            open: '(',
            close: ')',
            children: vec![inner_atom],
        };

        // Should find the inner atom when searching on line 2
        let found = outer_list.find_at_line(2).unwrap();
        if let SExpr::Atom { value, .. } = found {
            assert_eq!(value, "inner");
        } else {
            panic!("Expected atom");
        }

        // Should find the outer list when searching on line 1
        let found = outer_list.find_at_line(1).unwrap();
        assert!(matches!(found, SExpr::List { .. }));
    }

    #[test]
    fn test_find_deepest_at_line() {
        let outer_span = Span::new(Position::new(1, 0, 0), Position::new(3, 5, 25));
        let middle_span = Span::new(Position::new(2, 1, 5), Position::new(2, 8, 20));
        let inner_span = Span::new(Position::new(2, 2, 10), Position::new(2, 7, 15));
        
        let inner_atom = SExpr::Atom {
            span: inner_span,
            value: "deepest".to_string(),
        };

        let middle_list = SExpr::List {
            span: middle_span,
            open: '(',
            close: ')',
            children: vec![inner_atom],
        };

        let outer_list = SExpr::List {
            span: outer_span,
            open: '(',
            close: ')',
            children: vec![middle_list],
        };

        // Should find the deepest atom on line 2
        let found = outer_list.find_deepest_at_line(2).unwrap();
        if let SExpr::Atom { value, .. } = found {
            assert_eq!(value, "deepest");
        } else {
            panic!("Expected deepest atom");
        }
    }

    #[test]
    fn test_find_deepest_at_line_no_children() {
        let span = Span::new(Position::new(1, 0, 0), Position::new(1, 5, 5));
        let atom = SExpr::Atom {
            span,
            value: "test".to_string(),
        };

        let found = atom.find_deepest_at_line(1).unwrap();
        if let SExpr::Atom { value, .. } = found {
            assert_eq!(value, "test");
        } else {
            panic!("Expected atom");
        }
    }

    #[test]
    fn test_find_at_line_none() {
        let span = Span::new(Position::new(5, 0, 50), Position::new(5, 5, 55));
        let atom = SExpr::Atom {
            span,
            value: "test".to_string(),
        };

        assert!(atom.find_at_line(1).is_none());
        assert!(atom.find_deepest_at_line(1).is_none());
    }
}
