use crate::ast::{BinOp, Expr};

#[derive(Debug, Clone, PartialEq)]
enum Token {
    LParen,
    RParen,
    Symbol(String),
    Int(i64),
}

struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer { input, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek_char() {
            self.pos += c.len_utf8();
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while let Some(c) = self.peek_char() {
                if c.is_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }
            // Skip line comments
            if self.peek_char() == Some(';') {
                while let Some(c) = self.peek_char() {
                    self.advance();
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> Option<Token> {
        self.skip_whitespace_and_comments();

        let c = self.peek_char()?;

        match c {
            '(' => {
                self.advance();
                Some(Token::LParen)
            }
            ')' => {
                self.advance();
                Some(Token::RParen)
            }
            '-' => {
                // Could be negative number or minus operator
                let start = self.pos;
                self.advance();
                if let Some(c) = self.peek_char() {
                    if c.is_ascii_digit() {
                        // Negative number
                        while let Some(c) = self.peek_char() {
                            if c.is_ascii_digit() {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        let s = &self.input[start..self.pos];
                        Some(Token::Int(s.parse().unwrap()))
                    } else {
                        // Just a minus symbol
                        Some(Token::Symbol("-".to_string()))
                    }
                } else {
                    Some(Token::Symbol("-".to_string()))
                }
            }
            c if c.is_ascii_digit() => {
                let start = self.pos;
                while let Some(c) = self.peek_char() {
                    if c.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
                let s = &self.input[start..self.pos];
                Some(Token::Int(s.parse().unwrap()))
            }
            _ => {
                // Symbol
                let start = self.pos;
                while let Some(c) = self.peek_char() {
                    if c.is_whitespace() || c == '(' || c == ')' {
                        break;
                    }
                    self.advance();
                }
                let s = &self.input[start..self.pos];
                Some(Token::Symbol(s.to_string()))
            }
        }
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(input: &str) -> Self {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        while let Some(token) = lexer.next_token() {
            tokens.push(token);
        }
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.pos);
        self.pos += 1;
        token
    }

    fn expect_lparen(&mut self) -> Result<(), String> {
        match self.advance() {
            Some(Token::LParen) => Ok(()),
            other => Err(format!("Expected '(', got {:?}", other)),
        }
    }

    fn expect_rparen(&mut self) -> Result<(), String> {
        match self.advance() {
            Some(Token::RParen) => Ok(()),
            other => Err(format!("Expected ')', got {:?}", other)),
        }
    }

    fn expect_symbol(&mut self) -> Result<String, String> {
        match self.advance() {
            Some(Token::Symbol(s)) => Ok(s.clone()),
            other => Err(format!("Expected symbol, got {:?}", other)),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() {
                    Ok(Expr::Int(*n))
                } else {
                    unreachable!()
                }
            }
            Some(Token::Symbol(s)) => {
                let s = s.clone();
                self.advance();
                match s.as_str() {
                    "true" => Ok(Expr::Bool(true)),
                    "false" => Ok(Expr::Bool(false)),
                    _ => Ok(Expr::Var(s)),
                }
            }
            Some(Token::LParen) => self.parse_compound(),
            Some(Token::RParen) => Err("Unexpected ')'".to_string()),
            None => Err("Unexpected end of input".to_string()),
        }
    }

    fn parse_compound(&mut self) -> Result<Expr, String> {
        self.expect_lparen()?;

        let head = self.expect_symbol()?;

        let result = match head.as_str() {
            // Arithmetic operators
            "+" => self.parse_binop(BinOp::Add),
            "-" => self.parse_binop(BinOp::Sub),
            "*" => self.parse_binop(BinOp::Mul),
            "/" => self.parse_binop(BinOp::Div),
            "%" => self.parse_binop(BinOp::Mod),

            // Comparison operators
            "<" => self.parse_binop(BinOp::Lt),
            ">" => self.parse_binop(BinOp::Gt),
            "<=" => self.parse_binop(BinOp::Lte),
            ">=" => self.parse_binop(BinOp::Gte),
            "==" => self.parse_binop(BinOp::Eq),
            "!=" => self.parse_binop(BinOp::NotEq),

            // Logical operators
            "&&" => self.parse_binop(BinOp::And),
            "||" => self.parse_binop(BinOp::Or),

            // Bitwise operators
            "&" => self.parse_binop(BinOp::BitAnd),
            "|" => self.parse_binop(BinOp::BitOr),
            "^" => self.parse_binop(BinOp::BitXor),
            "<<" => self.parse_binop(BinOp::Shl),
            ">>" => self.parse_binop(BinOp::Shr),
            ">>>" => self.parse_binop(BinOp::UShr),

            // Control flow
            "if" => {
                let cond = self.parse_expr()?;
                let then_branch = self.parse_expr()?;
                let else_branch = self.parse_expr()?;
                Ok(Expr::If(
                    Box::new(cond),
                    Box::new(then_branch),
                    Box::new(else_branch),
                ))
            }

            // Binding
            "let" => {
                let name = self.expect_symbol()?;
                let value = self.parse_expr()?;
                let body = self.parse_expr()?;
                Ok(Expr::Let(name, Box::new(value), Box::new(body)))
            }

            // Functions
            "fn" => {
                self.expect_lparen()?;
                let mut params = Vec::new();
                while self.peek() != Some(&Token::RParen) {
                    params.push(self.expect_symbol()?);
                }
                self.expect_rparen()?;
                let body = self.parse_expr()?;
                Ok(Expr::Fn(params, Box::new(body)))
            }

            "call" => {
                let func = self.parse_expr()?;
                let mut args = Vec::new();
                while self.peek() != Some(&Token::RParen) {
                    args.push(self.parse_expr()?);
                }
                Ok(Expr::Call(Box::new(func), args))
            }

            // New expression
            "new" => {
                let ctor = self.parse_expr()?;
                let mut args = Vec::new();
                while self.peek() != Some(&Token::RParen) {
                    args.push(self.parse_expr()?);
                }
                Ok(Expr::New(Box::new(ctor), args))
            }

            // Arrays
            "array" => {
                let mut elements = Vec::new();
                while self.peek() != Some(&Token::RParen) {
                    elements.push(self.parse_expr()?);
                }
                Ok(Expr::Array(elements))
            }

            "index" => {
                let arr = self.parse_expr()?;
                let idx = self.parse_expr()?;
                Ok(Expr::Index(Box::new(arr), Box::new(idx)))
            }

            "len" => {
                let arr = self.parse_expr()?;
                Ok(Expr::Len(Box::new(arr)))
            }

            "while" => {
                let cond = self.parse_expr()?;
                let body = self.parse_expr()?;
                Ok(Expr::While(Box::new(cond), Box::new(body)))
            }

            "set!" => {
                let name = self.expect_symbol()?;
                let value = self.parse_expr()?;
                Ok(Expr::Set(name, Box::new(value)))
            }

            "begin" => {
                let mut exprs = Vec::new();
                while self.peek() != Some(&Token::RParen) {
                    exprs.push(self.parse_expr()?);
                }
                Ok(Expr::Begin(exprs))
            }

            // Objects
            "object" => {
                // (object) or (object (key1 val1) (key2 val2) ...)
                let mut props = Vec::new();
                while self.peek() != Some(&Token::RParen) {
                    self.expect_lparen()?;
                    let key = self.expect_symbol()?;
                    let val = self.parse_expr()?;
                    self.expect_rparen()?;
                    props.push((key, val));
                }
                Ok(Expr::Object(props))
            }

            "prop" => {
                // (prop obj "key")
                let obj = self.parse_expr()?;
                let key = self.parse_string_literal()?;
                Ok(Expr::PropAccess(Box::new(obj), key))
            }

            "prop-set!" => {
                // (prop-set! obj "key" value)
                let obj = self.parse_expr()?;
                let key = self.parse_string_literal()?;
                let value = self.parse_expr()?;
                Ok(Expr::PropSet(Box::new(obj), key, Box::new(value)))
            }

            "computed-prop" => {
                // (computed-prop obj key-expr)
                let obj = self.parse_expr()?;
                let key = self.parse_expr()?;
                Ok(Expr::ComputedAccess(Box::new(obj), Box::new(key)))
            }

            "computed-set!" => {
                // (computed-set! obj key-expr value)
                let obj = self.parse_expr()?;
                let key = self.parse_expr()?;
                let value = self.parse_expr()?;
                Ok(Expr::ComputedSet(Box::new(obj), Box::new(key), Box::new(value)))
            }

            // Switch statement
            "switch" => {
                // (switch expr (case val body...) ... (default body...))
                let discriminant = self.parse_expr()?;
                let mut cases = Vec::new();
                let mut default = None;

                while self.peek() != Some(&Token::RParen) {
                    self.expect_lparen()?;
                    let case_head = self.expect_symbol()?;
                    match case_head.as_str() {
                        "case" => {
                            let case_val = self.parse_expr()?;
                            let mut body = Vec::new();
                            while self.peek() != Some(&Token::RParen) {
                                body.push(self.parse_expr()?);
                            }
                            self.expect_rparen()?;
                            cases.push((case_val, body));
                        }
                        "default" => {
                            let mut body = Vec::new();
                            while self.peek() != Some(&Token::RParen) {
                                body.push(self.parse_expr()?);
                            }
                            self.expect_rparen()?;
                            default = Some(body);
                        }
                        other => return Err(format!("Expected 'case' or 'default', got '{}'", other)),
                    }
                }

                Ok(Expr::Switch {
                    discriminant: Box::new(discriminant),
                    cases,
                    default,
                })
            }

            // For loop
            "for" => {
                // (for init cond update body)
                let init = self.parse_expr()?;
                let cond = self.parse_expr()?;
                let update = self.parse_expr()?;
                let body = self.parse_expr()?;

                // Convert special values to None
                let init = if matches!(&init, Expr::Bool(false)) { None } else { Some(Box::new(init)) };
                let cond = if matches!(&cond, Expr::Bool(true)) { None } else { Some(Box::new(cond)) };
                let update = if matches!(&update, Expr::Bool(false)) { None } else { Some(Box::new(update)) };

                Ok(Expr::For {
                    init,
                    cond,
                    update,
                    body: Box::new(body),
                })
            }

            // Break and Continue
            "break" => Ok(Expr::Break),
            "continue" => Ok(Expr::Continue),

            // Undefined and Null
            "undefined" => Ok(Expr::Undefined),
            "null" => Ok(Expr::Null),

            // Logical not
            "!" => {
                let inner = self.parse_expr()?;
                Ok(Expr::LogNot(Box::new(inner)))
            }

            // Bitwise not
            "~" => {
                let inner = self.parse_expr()?;
                Ok(Expr::BitNot(Box::new(inner)))
            }

            // Throw
            "throw" => {
                let inner = self.parse_expr()?;
                Ok(Expr::Throw(Box::new(inner)))
            }
            // Return
            "return" => {
                let inner = self.parse_expr()?;
                Ok(Expr::Return(Box::new(inner)))
            }

            // Try-catch
            "try" => {
                // (try body (catch param body) (finally body)?)
                let try_block = self.parse_expr()?;

                // Parse catch clause
                self.expect_lparen()?;
                let catch_head = self.expect_symbol()?;
                if catch_head != "catch" {
                    return Err(format!("Expected 'catch', got '{}'", catch_head));
                }

                // Catch can have optional param: (catch e body) or (catch body)
                let (catch_param, catch_block) = if let Some(Token::Symbol(_)) = self.peek() {
                    let param = self.expect_symbol()?;
                    let block = self.parse_expr()?;
                    (Some(param), block)
                } else {
                    let block = self.parse_expr()?;
                    (None, block)
                };
                self.expect_rparen()?;

                // Optional finally clause
                let finally_block = if self.peek() == Some(&Token::LParen) {
                    self.expect_lparen()?;
                    let finally_head = self.expect_symbol()?;
                    if finally_head != "finally" {
                        return Err(format!("Expected 'finally', got '{}'", finally_head));
                    }
                    let block = self.parse_expr()?;
                    self.expect_rparen()?;
                    Some(Box::new(block))
                } else {
                    None
                };

                Ok(Expr::TryCatch {
                    try_block: Box::new(try_block),
                    catch_param,
                    catch_block: Box::new(catch_block),
                    finally_block,
                })
            }

            // Opaque
            "opaque" => {
                let label = self.parse_string_literal()?;
                Ok(Expr::Opaque(label))
            }

            other => Err(format!("Unknown form: {}", other)),
        }?;

        self.expect_rparen()?;
        Ok(result)
    }

    fn parse_binop(&mut self, op: BinOp) -> Result<Expr, String> {
        let left = self.parse_expr()?;
        let right = self.parse_expr()?;
        Ok(Expr::BinOp(op, Box::new(left), Box::new(right)))
    }

    fn parse_string_literal(&mut self) -> Result<String, String> {
        // Expect a symbol that looks like "key" (with quotes)
        match self.advance() {
            Some(Token::Symbol(s)) => {
                if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                    Ok(s[1..s.len() - 1].to_string())
                } else {
                    // Allow unquoted property names too
                    Ok(s.clone())
                }
            }
            other => Err(format!("Expected string literal, got {:?}", other)),
        }
    }
}

pub fn parse(input: &str) -> Result<Expr, String> {
    let mut parser = Parser::new(input);
    parser.parse_expr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_int() {
        assert_eq!(parse("42").unwrap(), Expr::Int(42));
        assert_eq!(parse("-5").unwrap(), Expr::Int(-5));
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(parse("true").unwrap(), Expr::Bool(true));
        assert_eq!(parse("false").unwrap(), Expr::Bool(false));
    }

    #[test]
    fn test_parse_var() {
        assert_eq!(parse("x").unwrap(), Expr::Var("x".to_string()));
    }

    #[test]
    fn test_parse_binop() {
        let expr = parse("(+ 1 2)").unwrap();
        assert_eq!(
            expr,
            Expr::BinOp(BinOp::Add, Box::new(Expr::Int(1)), Box::new(Expr::Int(2)))
        );
    }

    #[test]
    fn test_parse_let() {
        let expr = parse("(let x 5 (+ x 1))").unwrap();
        assert_eq!(
            expr,
            Expr::Let(
                "x".to_string(),
                Box::new(Expr::Int(5)),
                Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Int(1))
                ))
            )
        );
    }

    #[test]
    fn test_parse_fn() {
        let expr = parse("(fn (x y) (+ x y))").unwrap();
        assert_eq!(
            expr,
            Expr::Fn(
                vec!["x".to_string(), "y".to_string()],
                Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Var("y".to_string()))
                ))
            )
        );
    }
}
