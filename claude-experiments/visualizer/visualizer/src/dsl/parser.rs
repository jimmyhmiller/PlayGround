/// S-expression value — the AST for our DSL.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    String(String),
    Symbol(String),
    Keyword(String), // :x, :fill, etc.
    Bool(bool),
    List(Vec<Value>),
}

impl Value {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            Value::Symbol(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_keyword(&self) -> Option<&str> {
        match self {
            Value::Keyword(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) => Some(l),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at {}: {}", self.offset, self.message)
    }
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            // Skip line comments
            if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == b';' {
                while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn parse_string(&mut self) -> Result<Value, ParseError> {
        self.advance(); // skip opening "
        let start = self.pos;
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != b'"' {
            if self.input.as_bytes()[self.pos] == b'\\' {
                self.pos += 1; // skip escaped char
            }
            self.pos += 1;
        }
        if self.pos >= self.input.len() {
            return Err(ParseError {
                message: "unterminated string".into(),
                offset: start - 1,
            });
        }
        let s = self.input[start..self.pos].to_string();
        self.advance(); // skip closing "
        Ok(Value::String(s))
    }

    fn parse_atom(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;

        // Keywords start with :
        let is_keyword = self.peek() == Some(b':');
        if is_keyword {
            self.pos += 1;
        }

        // Read until whitespace or delimiter
        while self.pos < self.input.len() {
            let b = self.input.as_bytes()[self.pos];
            if b.is_ascii_whitespace() || b == b'(' || b == b')' || b == b'"' {
                break;
            }
            self.pos += 1;
        }

        let text = if is_keyword {
            &self.input[start + 1..self.pos]
        } else {
            &self.input[start..self.pos]
        };

        if text.is_empty() {
            return Err(ParseError {
                message: "empty atom".into(),
                offset: start,
            });
        }

        if is_keyword {
            return Ok(Value::Keyword(text.to_string()));
        }

        // Try to parse as number
        if let Ok(n) = text.parse::<f64>() {
            return Ok(Value::Number(n));
        }

        // Booleans
        if text == "true" {
            return Ok(Value::Bool(true));
        }
        if text == "false" {
            return Ok(Value::Bool(false));
        }

        // Otherwise it's a symbol
        Ok(Value::Symbol(text.to_string()))
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        self.skip_whitespace_and_comments();

        match self.peek() {
            None => Err(ParseError {
                message: "unexpected end of input".into(),
                offset: self.pos,
            }),
            Some(b'(') => self.parse_list(),
            Some(b'"') => self.parse_string(),
            Some(b')') => Err(ParseError {
                message: "unexpected ')'".into(),
                offset: self.pos,
            }),
            Some(_) => self.parse_atom(),
        }
    }

    fn parse_list(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        self.advance(); // skip (
        let mut items = Vec::new();

        loop {
            self.skip_whitespace_and_comments();
            match self.peek() {
                None => {
                    return Err(ParseError {
                        message: "unterminated list".into(),
                        offset: start,
                    });
                }
                Some(b')') => {
                    self.advance();
                    return Ok(Value::List(items));
                }
                _ => {
                    items.push(self.parse_value()?);
                }
            }
        }
    }
}

/// Parse a string into a list of top-level s-expressions.
pub fn parse(input: &str) -> Result<Vec<Value>, ParseError> {
    let mut parser = Parser::new(input);
    let mut values = Vec::new();

    loop {
        parser.skip_whitespace_and_comments();
        if parser.pos >= parser.input.len() {
            break;
        }
        values.push(parser.parse_value()?);
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_number() {
        let vals = parse("42").unwrap();
        assert_eq!(vals, vec![Value::Number(42.0)]);
    }

    #[test]
    fn test_parse_simple_list() {
        let vals = parse("(rect :x 10 :y 20)").unwrap();
        assert_eq!(
            vals,
            vec![Value::List(vec![
                Value::Symbol("rect".into()),
                Value::Keyword("x".into()),
                Value::Number(10.0),
                Value::Keyword("y".into()),
                Value::Number(20.0),
            ])]
        );
    }

    #[test]
    fn test_parse_nested() {
        let vals = parse("(+ 1 (* 2 3))").unwrap();
        assert_eq!(
            vals,
            vec![Value::List(vec![
                Value::Symbol("+".into()),
                Value::Number(1.0),
                Value::List(vec![
                    Value::Symbol("*".into()),
                    Value::Number(2.0),
                    Value::Number(3.0),
                ]),
            ])]
        );
    }

    #[test]
    fn test_parse_comment() {
        let vals = parse(";; hello\n42").unwrap();
        assert_eq!(vals, vec![Value::Number(42.0)]);
    }

    #[test]
    fn test_parse_string() {
        let vals = parse("\"hello\"").unwrap();
        assert_eq!(vals, vec![Value::String("hello".into())]);
    }
}
