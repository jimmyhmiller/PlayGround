//! Parser for the expression language (JavaScript-like syntax with classes)
//!
//! Grammar (precedence from low to high):
//!   program    ::= block | expr
//!   block      ::= '{' class_def+ expr '}'
//!   class_def  ::= 'class' IDENT params? '{' fields '}'
//!   params     ::= '(' IDENT (',' IDENT)* ')'
//!   fields     ::= (field (',' field)* ','?)?
//!   field      ::= IDENT ':' expr
//!
//!   expr       ::= ternary_expr
//!   ternary_expr ::= or_expr ('?' expr ':' expr)?
//!   or_expr    ::= and_expr ('||' and_expr)*
//!   and_expr   ::= eq_expr ('&&' eq_expr)*
//!   eq_expr    ::= add_expr ('==' add_expr)?
//!   add_expr   ::= mul_expr (('+' | '-') mul_expr)*
//!   mul_expr   ::= call_expr (('*' | '/') call_expr)*
//!   call_expr  ::= primary ('(' args? ')')?
//!   args       ::= expr (',' expr)*
//!   primary    ::= atom ('.' IDENT ('(' args? ')')?)*
//!   atom       ::= 'true' | 'false' | INT | STRING | IDENT | 'this' | object | '(' expr ')' | lambda
//!   lambda     ::= '(' params ')' '=>' expr | IDENT '=>' expr
//!   object     ::= '{' fields '}'

use crate::expr::{Expr, ClassDef, ObjectField};
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
        let expr = self.parse_program()?;
        if *self.current() != Token::Eof {
            return Err(ParseError {
                message: format!("Unexpected token after expression: {}", self.current()),
            });
        }
        Ok(expr)
    }

    /// Parse program: either a block with classes, a standalone class, or a plain expression
    fn parse_program(&mut self) -> Result<Expr, ParseError> {
        // Check if this is a block (starts with { followed by 'class')
        if *self.current() == Token::LBrace {
            if *self.peek(1) == Token::Class {
                return self.parse_block();
            }
        }

        // Check for standalone class definition (for REPL convenience)
        // class F(x) { ... } is equivalent to { class F(x) { ... } F }
        if *self.current() == Token::Class {
            let class_def = self.parse_class_def()?;
            let name = class_def.name.clone();
            // Return a block with just this class, and the class name as the body
            return Ok(Expr::Block(vec![class_def], Box::new(Expr::var(name))));
        }

        self.parse_expr()
    }

    /// Parse a block: { class_def+ expr }
    fn parse_block(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::LBrace)?;

        let mut classes = Vec::new();

        // Parse class definitions
        while *self.current() == Token::Class {
            classes.push(self.parse_class_def()?);
        }

        if classes.is_empty() {
            return Err(ParseError {
                message: "Block must contain at least one class definition".to_string(),
            });
        }

        // Parse the final expression
        let body = self.parse_expr()?;

        self.expect(Token::RBrace)?;

        Ok(Expr::Block(classes, Box::new(body)))
    }

    /// Parse a class definition: class Name(params) { fields }
    fn parse_class_def(&mut self) -> Result<ClassDef, ParseError> {
        self.expect(Token::Class)?;
        let name = self.expect_ident()?;

        // Parse optional parameters
        let params = if *self.current() == Token::LParen {
            self.parse_params()?
        } else {
            Vec::new()
        };

        // Parse the class body (fields, including spreads)
        self.expect(Token::LBrace)?;
        let fields = self.parse_class_fields()?;
        self.expect(Token::RBrace)?;

        Ok(ClassDef::new(name, params, fields))
    }

    /// Parse class fields: name: expr, ...spread, ...
    fn parse_class_fields(&mut self) -> Result<Vec<ObjectField>, ParseError> {
        let mut fields = Vec::new();

        // Empty fields
        if *self.current() == Token::RBrace {
            return Ok(fields);
        }

        // Parse first field
        fields.push(self.parse_object_field()?);

        // Parse remaining fields
        while *self.current() == Token::Comma {
            self.advance(); // consume ','

            // Allow trailing comma
            if *self.current() == Token::RBrace {
                break;
            }

            fields.push(self.parse_object_field()?);
        }

        Ok(fields)
    }

    /// Parse parameter list: (a, b, c)
    fn parse_params(&mut self) -> Result<Vec<String>, ParseError> {
        self.expect(Token::LParen)?;

        let mut params = Vec::new();

        if *self.current() != Token::RParen {
            params.push(self.expect_ident()?);

            while *self.current() == Token::Comma {
                self.advance(); // consume ','
                params.push(self.expect_ident()?);
            }
        }

        self.expect(Token::RParen)?;
        Ok(params)
    }

    /// Parse fields: name: expr, name: expr, ...
    fn parse_fields(&mut self) -> Result<Vec<(String, Expr)>, ParseError> {
        let mut fields = Vec::new();

        // Empty fields
        if *self.current() == Token::RBrace {
            return Ok(fields);
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

        Ok(fields)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_ternary_expr()
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
        let mut left = self.parse_call_expr()?;

        loop {
            match self.current() {
                Token::Star => {
                    self.advance();
                    let right = self.parse_call_expr()?;
                    left = Expr::mul(left, right);
                }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_call_expr()?;
                    left = Expr::div(left, right);
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse call expression: primary followed by optional calls and field access
    /// Handles: f(x), f(x).g(y), f.g(x).h, etc.
    fn parse_call_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_atom()?;

        loop {
            match self.current() {
                Token::LParen => {
                    // Make sure this isn't a lambda
                    if self.is_lambda_start() {
                        break;
                    }
                    let args = self.parse_args()?;
                    expr = Expr::Call(Box::new(expr), args);
                }
                Token::Dot => {
                    self.advance(); // consume '.'
                    let field = self.expect_ident()?;
                    expr = Expr::field(expr, field);
                    // Check for method call: .method(args)
                    if *self.current() == Token::LParen && !self.is_lambda_start() {
                        let args = self.parse_args()?;
                        expr = Expr::Call(Box::new(expr), args);
                    }
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    /// Check if current position starts a lambda: (params) => or single IDENT =>
    fn is_lambda_start(&self) -> bool {
        if *self.current() == Token::LParen {
            // Check for (IDENT, ...) => pattern
            // Find the matching )
            let mut depth = 0;
            let mut i = 0;
            loop {
                let token = self.peek(i);
                match token {
                    Token::LParen => depth += 1,
                    Token::RParen => {
                        depth -= 1;
                        if depth == 0 {
                            // Check if followed by =>
                            return *self.peek(i + 1) == Token::Arrow;
                        }
                    }
                    Token::Eof => return false,
                    _ => {}
                }
                i += 1;
                if i > 100 {
                    // Safety limit
                    return false;
                }
            }
        }
        false
    }

    /// Parse argument list: (expr, expr, ...)
    fn parse_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        self.expect(Token::LParen)?;

        let mut args = Vec::new();

        if *self.current() != Token::RParen {
            args.push(self.parse_expr()?);

            while *self.current() == Token::Comma {
                self.advance(); // consume ','
                args.push(self.parse_expr()?);
            }
        }

        self.expect(Token::RParen)?;
        Ok(args)
    }

    // Note: parse_primary is no longer used - field access is handled in parse_call_expr
    #[allow(dead_code)]
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        self.parse_atom()
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
                // Check for single-param lambda: x => expr
                if *self.peek(1) == Token::Arrow {
                    self.advance(); // consume IDENT
                    self.advance(); // consume '=>'
                    let body = self.parse_expr()?;
                    return Ok(Expr::lambda(name, body));
                }
                self.advance();
                Ok(Expr::var(name))
            }
            Token::This => {
                self.advance();
                Ok(Expr::this())
            }
            Token::LBrace => self.parse_object(),
            Token::LParen => {
                // Could be:
                // 1. Lambda with params: (a, b) => expr or (x) => expr
                // 2. Grouped expression: (expr)

                if self.is_lambda_start() {
                    return self.parse_lambda();
                }

                // Grouped expression
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

    /// Parse a lambda expression: (params) => expr
    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        let params = self.parse_params()?;
        self.expect(Token::Arrow)?;
        let body = self.parse_expr()?;

        // Build curried lambda: (a, b) => e  becomes  a => b => e
        let result = params.into_iter().rev().fold(body, |acc, param| {
            Expr::Lambda(param, Box::new(acc))
        });

        Ok(result)
    }

    fn parse_object(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::LBrace)?;

        let mut fields: Vec<ObjectField> = Vec::new();

        // Empty object
        if *self.current() == Token::RBrace {
            self.advance();
            return Ok(Expr::Object(fields));
        }

        // Parse first field (or spread)
        fields.push(self.parse_object_field()?);

        // Parse remaining fields
        while *self.current() == Token::Comma {
            self.advance(); // consume ','

            // Allow trailing comma
            if *self.current() == Token::RBrace {
                break;
            }

            fields.push(self.parse_object_field()?);
        }

        self.expect(Token::RBrace)?;
        Ok(Expr::Object(fields))
    }

    /// Parse an object field: either "name: expr" or "...expr"
    fn parse_object_field(&mut self) -> Result<ObjectField, ParseError> {
        // Check for spread: ...expr
        if *self.current() == Token::DotDotDot {
            self.advance(); // consume '...'
            let expr = self.parse_expr()?;
            return Ok(ObjectField::Spread(expr));
        }

        // Otherwise, it's a named field: name: expr
        let name = self.expect_ident()?;
        self.expect(Token::Colon)?;
        let value = self.parse_expr()?;
        Ok(ObjectField::Field(name, value))
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
    fn test_parse_multi_param_lambda() {
        let expr = parse("(x, y) => x").unwrap();
        // Should be curried: x => y => x
        match expr {
            Expr::Lambda(x, body) => {
                assert_eq!(x, "x");
                assert!(matches!(*body, Expr::Lambda(..)));
            }
            _ => panic!("Expected lambda"),
        }
    }

    #[test]
    fn test_parse_nested_arrow() {
        let expr = parse("x => y => x").unwrap();
        assert!(matches!(expr, Expr::Lambda(..)));
    }

    #[test]
    fn test_parse_call() {
        let expr = parse("f(x)").unwrap();
        assert!(matches!(expr, Expr::Call(..)));
    }

    #[test]
    fn test_parse_call_multiple_args() {
        let expr = parse("f(x, y, z)").unwrap();
        match expr {
            Expr::Call(_, args) => assert_eq!(args.len(), 3),
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_parse_call_no_args() {
        let expr = parse("f()").unwrap();
        match expr {
            Expr::Call(_, args) => assert!(args.is_empty()),
            _ => panic!("Expected Call"),
        }
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
    fn test_parse_method_call() {
        let expr = parse("obj.method(x, y)").unwrap();
        assert!(matches!(expr, Expr::Call(..)));
    }

    #[test]
    fn test_parse_object_with_this() {
        let expr = parse("{ self: this }").unwrap();
        assert!(matches!(expr, Expr::Object(..)));
    }

    #[test]
    fn test_parse_class_def() {
        let input = r#"
            {
                class Foo(x) {
                    value: x
                }
                Foo(42)
            }
        "#;
        let expr = parse(input).unwrap();
        assert!(matches!(expr, Expr::Block(..)));
    }

    #[test]
    fn test_parse_class_no_params() {
        let input = r#"
            {
                class Empty {
                    value: 42
                }
                Empty
            }
        "#;
        let expr = parse(input).unwrap();
        assert!(matches!(expr, Expr::Block(..)));
    }

    #[test]
    fn test_parse_multiple_classes() {
        let input = r#"
            {
                class A(x) { value: x }
                class B(y) { other: y }
                A(1)
            }
        "#;
        let expr = parse(input).unwrap();
        match expr {
            Expr::Block(classes, _) => assert_eq!(classes.len(), 2),
            _ => panic!("Expected Block"),
        }
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
    fn test_parse_call_with_lambda_arg() {
        let expr = parse("f(x => x)").unwrap();
        assert!(matches!(expr, Expr::Call(..)));
    }

    #[test]
    fn test_parse_chained_calls() {
        let expr = parse("f(x).g(y)").unwrap();
        assert!(matches!(expr, Expr::Call(..)));
    }

    #[test]
    fn test_parse_spread() {
        let expr = parse("{ ...x }").unwrap();
        match expr {
            Expr::Object(fields) => {
                assert_eq!(fields.len(), 1);
                assert!(matches!(fields[0], ObjectField::Spread(_)));
            }
            _ => panic!("Expected Object"),
        }
    }

    #[test]
    fn test_parse_spread_with_fields() {
        let expr = parse("{ ...x, y: 1, z: 2 }").unwrap();
        match expr {
            Expr::Object(fields) => {
                assert_eq!(fields.len(), 3);
                assert!(matches!(fields[0], ObjectField::Spread(_)));
                assert!(matches!(fields[1], ObjectField::Field(..)));
                assert!(matches!(fields[2], ObjectField::Field(..)));
            }
            _ => panic!("Expected Object"),
        }
    }

    #[test]
    fn test_parse_multiple_spreads() {
        let expr = parse("{ ...x, y: 1, ...z }").unwrap();
        match expr {
            Expr::Object(fields) => {
                assert_eq!(fields.len(), 3);
                assert!(matches!(fields[0], ObjectField::Spread(_)));
                assert!(matches!(fields[1], ObjectField::Field(..)));
                assert!(matches!(fields[2], ObjectField::Spread(_)));
            }
            _ => panic!("Expected Object"),
        }
    }
}
