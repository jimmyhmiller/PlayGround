//! Parser for the expression language (JavaScript-like syntax)
//!
//! Grammar (precedence from low to high):
//!   expr       ::= let_expr | ternary_expr
//!   let_expr   ::= 'let' IDENT '=' expr 'in' expr
//!   ternary_expr ::= or_expr ('?' expr ':' expr)?
//!   or_expr    ::= and_expr ('||' and_expr)*
//!   and_expr   ::= eq_expr ('&&' eq_expr)*
//!   eq_expr    ::= concat_expr ('==' concat_expr)?
//!   concat_expr ::= add_expr ('++' add_expr)*
//!   add_expr   ::= mul_expr (('+' | '-') mul_expr)*
//!   mul_expr   ::= arrow_expr (('*' | '/') arrow_expr)*
//!   arrow_expr ::= IDENT '=>' arrow_expr | paren_arrow | app_expr
//!   paren_arrow ::= '(' IDENT ')' '=>' arrow_expr
//!   app_expr   ::= primary (primary)*
//!   primary    ::= atom ('.' IDENT)*
//!   atom       ::= 'true' | 'false' | INT | STRING | IDENT | 'this' | object | '(' expr ')'
//!   object     ::= '{' (field (',' field)*)? '}'
//!   field      ::= IDENT ':' expr

use crate::expr::Expr;
use crate::lexer::{Lexer, Token, LexError};
use std::fmt;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error: {}", self.message)
    }
}

impl From<LexError> for ParseError {
    fn from(e: LexError) -> Self {
        ParseError {
            message: format!("Lexer error: {}", e),
        }
    }
}

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(input: &str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        Ok(Parser {
            tokens,
            position: 0,
        })
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::Eof)
    }

    fn peek(&self, offset: usize) -> &Token {
        self.tokens.get(self.position + offset).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let token = self.current().clone();
        if self.position < self.tokens.len() {
            self.position += 1;
        }
        token
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        let actual = self.advance();
        if actual == expected {
            Ok(())
        } else {
            Err(ParseError {
                message: format!("Expected {}, got {}", expected, actual),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.advance() {
            Token::Ident(name) => Ok(name),
            other => Err(ParseError {
                message: format!("Expected identifier, got {}", other),
            }),
        }
    }

    pub fn parse(&mut self) -> Result<Expr, ParseError> {
        let expr = self.parse_expr()?;
        if *self.current() != Token::Eof {
            return Err(ParseError {
                message: format!("Unexpected token after expression: {}", self.current()),
            });
        }
        Ok(expr)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        match self.current() {
            Token::Let => self.parse_let(),
            _ => self.parse_ternary_expr(),
        }
    }

    fn parse_ternary_expr(&mut self) -> Result<Expr, ParseError> {
        let cond = self.parse_or_expr()?;

        // Check for ternary: cond ? then : else
        if *self.current() == Token::Question {
            self.advance(); // consume '?'
            let then_expr = self.parse_expr()?;
            self.expect(Token::Colon)?;
            let else_expr = self.parse_expr()?;
            Ok(Expr::if_(cond, then_expr, else_expr))
        } else {
            Ok(cond)
        }
    }

    fn parse_or_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and_expr()?;

        while *self.current() == Token::OrOr {
            self.advance(); // consume '||'
            let right = self.parse_and_expr()?;
            left = Expr::or(left, right);
        }

        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_eq_expr()?;

        while *self.current() == Token::AndAnd {
            self.advance(); // consume '&&'
            let right = self.parse_eq_expr()?;
            left = Expr::and(left, right);
        }

        Ok(left)
    }

    fn parse_eq_expr(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_concat_expr()?;

        if *self.current() == Token::EqEq {
            self.advance(); // consume '=='
            let right = self.parse_concat_expr()?;
            Ok(Expr::eq(left, right))
        } else {
            Ok(left)
        }
    }

    fn parse_concat_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_add_expr()?;

        while *self.current() == Token::PlusPlus {
            self.advance(); // consume '++'
            let right = self.parse_add_expr()?;
            left = Expr::concat(left, right);
        }

        Ok(left)
    }

    fn parse_add_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_mul_expr()?;

        loop {
            match self.current() {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_mul_expr()?;
                    left = Expr::add(left, right);
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_mul_expr()?;
                    left = Expr::sub(left, right);
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_mul_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_arrow_expr()?;

        loop {
            match self.current() {
                Token::Star => {
                    self.advance();
                    let right = self.parse_arrow_expr()?;
                    left = Expr::mul(left, right);
                }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_arrow_expr()?;
                    left = Expr::div(left, right);
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_let(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::Let)?;

        // Check for 'rec' keyword
        let is_rec = if *self.current() == Token::Rec {
            self.advance();
            true
        } else {
            false
        };

        let name = self.expect_ident()?;
        self.expect(Token::Equals)?;
        let value = self.parse_expr()?;

        // Check for mutual recursion with 'and'
        if is_rec && *self.current() == Token::And {
            let mut bindings = vec![(name, value)];

            while *self.current() == Token::And {
                self.advance(); // consume 'and'
                let next_name = self.expect_ident()?;
                self.expect(Token::Equals)?;
                let next_value = self.parse_expr()?;
                bindings.push((next_name, next_value));
            }

            self.expect(Token::In)?;
            let body = self.parse_expr()?;
            return Ok(Expr::LetRecMutual(bindings, Box::new(body)));
        }

        self.expect(Token::In)?;
        let body = self.parse_expr()?;

        if is_rec {
            Ok(Expr::let_rec(name, value, body))
        } else {
            Ok(Expr::let_(name, value, body))
        }
    }

    fn parse_arrow_expr(&mut self) -> Result<Expr, ParseError> {
        // Check for: IDENT => expr
        // Arrow function body can contain ||, &&, ==, ternary, etc.
        if let Token::Ident(name) = self.current().clone() {
            if *self.peek(1) == Token::Arrow {
                self.advance(); // consume IDENT
                self.advance(); // consume '=>'
                // Parse body at ternary level to allow all operators including ternary
                let body = self.parse_ternary_expr()?;
                return Ok(Expr::lambda(name, body));
            }
        }

        // Check for: (IDENT) => expr
        if *self.current() == Token::LParen {
            if let Token::Ident(_) = self.peek(1) {
                if *self.peek(2) == Token::RParen && *self.peek(3) == Token::Arrow {
                    self.advance(); // consume '('
                    let name = self.expect_ident()?;
                    self.expect(Token::RParen)?;
                    self.expect(Token::Arrow)?;
                    // Parse body at ternary level to allow all operators including ternary
                    let body = self.parse_ternary_expr()?;
                    return Ok(Expr::lambda(name, body));
                }
            }
        }

        self.parse_app_expr()
    }

    fn parse_app_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;

        // Application is left-associative: f x y = (f x) y
        loop {
            match self.current() {
                // These can start a primary expression (but not an arrow function)
                Token::True
                | Token::False
                | Token::Int(_)
                | Token::String(_)
                | Token::This
                | Token::LBrace => {
                    let arg = self.parse_primary()?;
                    expr = Expr::app(expr, arg);
                }
                Token::Ident(_) => {
                    // Only parse as argument if not followed by =>
                    if *self.peek(1) != Token::Arrow {
                        let arg = self.parse_primary()?;
                        expr = Expr::app(expr, arg);
                    } else {
                        break;
                    }
                }
                Token::LParen => {
                    // Could be (expr) as argument or (x) => body
                    // Check if it's an arrow function
                    if let Token::Ident(_) = self.peek(1) {
                        if *self.peek(2) == Token::RParen && *self.peek(3) == Token::Arrow {
                            break; // It's an arrow function, stop parsing application
                        }
                    }
                    let arg = self.parse_primary()?;
                    expr = Expr::app(expr, arg);
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_atom()?;

        // Field access: expr.field.field...
        while *self.current() == Token::Dot {
            self.advance(); // consume '.'
            let field = self.expect_ident()?;
            expr = Expr::field(expr, field);
        }

        Ok(expr)
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        match self.current().clone() {
            Token::True => {
                self.advance();
                Ok(Expr::bool(true))
            }
            Token::False => {
                self.advance();
                Ok(Expr::bool(false))
            }
            Token::Int(n) => {
                self.advance();
                Ok(Expr::int(n))
            }
            Token::String(s) => {
                self.advance();
                Ok(Expr::string(s))
            }
            Token::Ident(name) => {
                self.advance();
                Ok(Expr::var(name))
            }
            Token::This => {
                self.advance();
                Ok(Expr::this())
            }
            Token::LBrace => self.parse_object(),
            Token::LParen => {
                self.advance(); // consume '('
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            other => Err(ParseError {
                message: format!("Unexpected token: {}", other),
            }),
        }
    }

    fn parse_object(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::LBrace)?;

        let mut fields = Vec::new();

        // Empty object
        if *self.current() == Token::RBrace {
            self.advance();
            return Ok(Expr::object(fields));
        }

        // Parse first field
        fields.push(self.parse_field()?);

        // Parse remaining fields
        while *self.current() == Token::Comma {
            self.advance(); // consume ','

            // Allow trailing comma
            if *self.current() == Token::RBrace {
                break;
            }

            fields.push(self.parse_field()?);
        }

        self.expect(Token::RBrace)?;
        Ok(Expr::object(fields))
    }

    fn parse_field(&mut self) -> Result<(String, Expr), ParseError> {
        let name = self.expect_ident()?;
        self.expect(Token::Colon)?;
        let value = self.parse_expr()?;
        Ok((name, value))
    }
}

/// Parse a string into an expression
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let mut parser = Parser::new(input)?;
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_literals() {
        assert!(matches!(parse("true").unwrap(), Expr::Bool(true)));
        assert!(matches!(parse("false").unwrap(), Expr::Bool(false)));
        assert!(matches!(parse("42").unwrap(), Expr::Int(42)));
    }

    #[test]
    fn test_parse_subtraction() {
        // Negative numbers are now parsed as (0 - n) or just subtraction
        let expr = parse("5 - 3").unwrap();
        assert!(matches!(expr, Expr::Sub(..)));
    }

    #[test]
    fn test_parse_variable() {
        assert!(matches!(parse("x").unwrap(), Expr::Var(ref s) if s == "x"));
    }

    #[test]
    fn test_parse_this() {
        assert!(matches!(parse("this").unwrap(), Expr::This));
    }

    #[test]
    fn test_parse_arrow_function() {
        let expr = parse("x => x").unwrap();
        assert!(matches!(expr, Expr::Lambda(..)));
    }

    #[test]
    fn test_parse_arrow_function_parens() {
        let expr = parse("(x) => x").unwrap();
        assert!(matches!(expr, Expr::Lambda(..)));
    }

    #[test]
    fn test_parse_nested_arrow() {
        let expr = parse("x => y => x").unwrap();
        assert!(matches!(expr, Expr::Lambda(..)));
    }

    #[test]
    fn test_parse_application() {
        let expr = parse("f x").unwrap();
        assert!(matches!(expr, Expr::App(..)));
    }

    #[test]
    fn test_parse_multiple_application() {
        let expr = parse("f x y").unwrap();
        // Should be (f x) y
        assert!(matches!(expr, Expr::App(..)));
    }

    #[test]
    fn test_parse_let() {
        let expr = parse("let x = 42 in x").unwrap();
        assert!(matches!(expr, Expr::Let(..)));
    }

    #[test]
    fn test_parse_ternary() {
        let expr = parse("true ? 1 : 2").unwrap();
        assert!(matches!(expr, Expr::If(..)));
    }

    #[test]
    fn test_parse_object() {
        let expr = parse("{ x: 42, y: true }").unwrap();
        assert!(matches!(expr, Expr::Object(..)));
    }

    #[test]
    fn test_parse_empty_object() {
        let expr = parse("{ }").unwrap();
        match expr {
            Expr::Object(fields) => assert!(fields.is_empty()),
            _ => panic!("Expected empty object"),
        }
    }

    #[test]
    fn test_parse_field_access() {
        let expr = parse("obj.x").unwrap();
        assert!(matches!(expr, Expr::FieldAccess(..)));
    }

    #[test]
    fn test_parse_chained_field_access() {
        let expr = parse("obj.x.y").unwrap();
        assert!(matches!(expr, Expr::FieldAccess(..)));
    }

    #[test]
    fn test_parse_object_with_this() {
        let expr = parse("{ self: this }").unwrap();
        assert!(matches!(expr, Expr::Object(..)));
    }

    #[test]
    fn test_parse_complex() {
        let input = r#"
            let isEmpty = s => s.isEmpty in
            let mySet = { isEmpty: true, contains: i => false } in
            isEmpty mySet
        "#;
        parse(input).unwrap();
    }

    #[test]
    fn test_parse_cook_style() {
        let input = r#"
            {
                isEmpty: true,
                contains: i => false,
                insert: i => this
            }
        "#;
        parse(input).unwrap();
    }

    #[test]
    fn test_parse_nested_ternary() {
        let expr = parse("a ? b ? 1 : 2 : 3").unwrap();
        assert!(matches!(expr, Expr::If(..)));
    }

    #[test]
    fn test_parse_arrow_in_object() {
        let expr = parse("{ f: x => x }").unwrap();
        assert!(matches!(expr, Expr::Object(..)));
    }

    #[test]
    fn test_application_with_arrow_arg() {
        let expr = parse("f (x => x)").unwrap();
        assert!(matches!(expr, Expr::App(..)));
    }
}
