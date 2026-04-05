/// Recursive descent parser for Lox. Produces an AST.
use crate::ast::*;
use crate::scanner::{Scanner, Token, TokenType};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    had_error: bool,
    panic_mode: bool,
}

impl Parser {
    pub fn parse(source: &str) -> Option<Program> {
        let mut scanner = Scanner::new(source);
        let mut tokens = Vec::new();
        loop {
            let tok = scanner.scan_token();
            let is_eof = tok.token_type == TokenType::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }

        let mut parser = Parser {
            tokens,
            current: 0,
            had_error: false,
            panic_mode: false,
        };

        let mut stmts = Vec::new();
        while !parser.is_at_end() {
            if let Some(s) = parser.declaration() {
                stmts.push(s);
            }
        }

        if parser.had_error {
            None
        } else {
            Some(Program { stmts })
        }
    }

    // ── Declarations ─────────────────────────────────────────────

    fn declaration(&mut self) -> Option<Stmt> {
        let result = if self.match_token(TokenType::Class) {
            self.class_declaration()
        } else if self.match_token(TokenType::Fun) {
            self.fun_declaration()
        } else if self.match_token(TokenType::Var) {
            self.var_declaration()
        } else {
            self.statement()
        };

        match result {
            Some(s) => Some(s),
            None => {
                self.synchronize();
                None
            }
        }
    }

    fn class_declaration(&mut self) -> Option<Stmt> {
        let name = self.consume_identifier("Expect class name.")?;

        let superclass = if self.match_token(TokenType::Less) {
            Some(self.consume_identifier("Expect superclass name.")?)
        } else {
            None
        };

        self.consume(TokenType::LeftBrace, "Expect '{' before class body.")?;

        let mut methods = Vec::new();
        while !self.check(TokenType::RightBrace) && !self.is_at_end() {
            methods.push(self.fun_decl("method")?);
        }

        self.consume(TokenType::RightBrace, "Expect '}' after class body.")?;

        Some(Stmt::Class(ClassDecl {
            name,
            superclass,
            methods,
        }))
    }

    fn fun_declaration(&mut self) -> Option<Stmt> {
        let decl = self.fun_decl("function")?;
        Some(Stmt::Fun(decl))
    }

    fn fun_decl(&mut self, kind: &str) -> Option<FunDecl> {
        let name = self.consume_identifier(&format!("Expect {} name.", kind))?;
        self.consume(
            TokenType::LeftParen,
            &format!("Expect '(' after {} name.", kind),
        )?;

        let mut params = Vec::new();
        if !self.check(TokenType::RightParen) {
            loop {
                if params.len() >= 255 {
                    self.error_at_current("Can't have more than 255 parameters.");
                }
                params.push(self.consume_identifier("Expect parameter name.")?);
                if !self.match_token(TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenType::RightParen, "Expect ')' after parameters.")?;
        self.consume(
            TokenType::LeftBrace,
            &format!("Expect '{{' before {} body.", kind),
        )?;

        let body = self.block_stmts()?;

        Some(FunDecl { name, params, body })
    }

    fn var_declaration(&mut self) -> Option<Stmt> {
        let name = self.consume_identifier("Expect variable name.")?;
        let initializer = if self.match_token(TokenType::Equal) {
            Some(self.expression()?)
        } else {
            None
        };
        self.consume(TokenType::Semicolon, "Expect ';' after variable declaration.")?;
        Some(Stmt::Var(name, initializer))
    }

    // ── Statements ───────────────────────────────────────────────

    fn statement(&mut self) -> Option<Stmt> {
        if self.match_token(TokenType::Print) {
            self.print_statement()
        } else if self.match_token(TokenType::For) {
            self.for_statement()
        } else if self.match_token(TokenType::If) {
            self.if_statement()
        } else if self.match_token(TokenType::Return) {
            self.return_statement()
        } else if self.match_token(TokenType::While) {
            self.while_statement()
        } else if self.match_token(TokenType::LeftBrace) {
            let stmts = self.block_stmts()?;
            Some(Stmt::Block(stmts))
        } else {
            self.expression_statement()
        }
    }

    fn print_statement(&mut self) -> Option<Stmt> {
        let expr = self.expression()?;
        self.consume(TokenType::Semicolon, "Expect ';' after value.")?;
        Some(Stmt::Print(expr))
    }

    fn for_statement(&mut self) -> Option<Stmt> {
        self.consume(TokenType::LeftParen, "Expect '(' after 'for'.")?;

        let initializer = if self.match_token(TokenType::Semicolon) {
            None
        } else if self.match_token(TokenType::Var) {
            Some(self.var_declaration()?)
        } else {
            Some(self.expression_statement()?)
        };

        let condition = if !self.check(TokenType::Semicolon) {
            Some(self.expression()?)
        } else {
            None
        };
        self.consume(TokenType::Semicolon, "Expect ';' after loop condition.")?;

        let increment = if !self.check(TokenType::RightParen) {
            Some(self.expression()?)
        } else {
            None
        };
        self.consume(TokenType::RightParen, "Expect ')' after for clauses.")?;

        let mut body = self.statement()?;

        // Desugar: for → while
        if let Some(inc) = increment {
            body = Stmt::Block(vec![body, Stmt::Expr(inc)]);
        }

        let cond = condition.unwrap_or(Expr::Bool(true));
        body = Stmt::While(cond, Box::new(body));

        if let Some(init) = initializer {
            body = Stmt::Block(vec![init, body]);
        }

        Some(body)
    }

    fn if_statement(&mut self) -> Option<Stmt> {
        self.consume(TokenType::LeftParen, "Expect '(' after 'if'.")?;
        let condition = self.expression()?;
        self.consume(TokenType::RightParen, "Expect ')' after if condition.")?;

        let then_branch = self.statement()?;
        let else_branch = if self.match_token(TokenType::Else) {
            Some(Box::new(self.statement()?))
        } else {
            None
        };

        Some(Stmt::If(condition, Box::new(then_branch), else_branch))
    }

    fn return_statement(&mut self) -> Option<Stmt> {
        let value = if !self.check(TokenType::Semicolon) {
            Some(self.expression()?)
        } else {
            None
        };
        self.consume(TokenType::Semicolon, "Expect ';' after return value.")?;
        Some(Stmt::Return(value))
    }

    fn while_statement(&mut self) -> Option<Stmt> {
        self.consume(TokenType::LeftParen, "Expect '(' after 'while'.")?;
        let condition = self.expression()?;
        self.consume(TokenType::RightParen, "Expect ')' after condition.")?;
        let body = self.statement()?;
        Some(Stmt::While(condition, Box::new(body)))
    }

    fn expression_statement(&mut self) -> Option<Stmt> {
        let expr = self.expression()?;
        self.consume(TokenType::Semicolon, "Expect ';' after expression.")?;
        Some(Stmt::Expr(expr))
    }

    fn block_stmts(&mut self) -> Option<Vec<Stmt>> {
        let mut stmts = Vec::new();
        while !self.check(TokenType::RightBrace) && !self.is_at_end() {
            if let Some(s) = self.declaration() {
                stmts.push(s);
            }
        }
        self.consume(TokenType::RightBrace, "Expect '}' after block.")?;
        Some(stmts)
    }

    // ── Expressions (Pratt/precedence climbing) ──────────────────

    fn expression(&mut self) -> Option<Expr> {
        self.assignment()
    }

    fn assignment(&mut self) -> Option<Expr> {
        let expr = self.or()?;

        if self.match_token(TokenType::Equal) {
            let value = self.assignment()?;
            match expr {
                Expr::Var(name) => return Some(Expr::Assign(name, Box::new(value))),
                Expr::Get(obj, name) => return Some(Expr::Set(obj, name, Box::new(value))),
                _ => {
                    self.error("Invalid assignment target.");
                    return Some(value);
                }
            }
        }

        Some(expr)
    }

    fn or(&mut self) -> Option<Expr> {
        let mut expr = self.and()?;
        while self.match_token(TokenType::Or) {
            let right = self.and()?;
            expr = Expr::Logical(Box::new(expr), LogicalOp::Or, Box::new(right));
        }
        Some(expr)
    }

    fn and(&mut self) -> Option<Expr> {
        let mut expr = self.equality()?;
        while self.match_token(TokenType::And) {
            let right = self.equality()?;
            expr = Expr::Logical(Box::new(expr), LogicalOp::And, Box::new(right));
        }
        Some(expr)
    }

    fn equality(&mut self) -> Option<Expr> {
        let mut expr = self.comparison()?;
        while self.match_any(&[TokenType::BangEqual, TokenType::EqualEqual]) {
            let op = if self.previous().token_type == TokenType::EqualEqual {
                BinOp::Eq
            } else {
                BinOp::Ne
            };
            let right = self.comparison()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(right));
        }
        Some(expr)
    }

    fn comparison(&mut self) -> Option<Expr> {
        let mut expr = self.term()?;
        while self.match_any(&[
            TokenType::Greater,
            TokenType::GreaterEqual,
            TokenType::Less,
            TokenType::LessEqual,
        ]) {
            let op = match self.previous().token_type {
                TokenType::Greater => BinOp::Gt,
                TokenType::GreaterEqual => BinOp::Ge,
                TokenType::Less => BinOp::Lt,
                TokenType::LessEqual => BinOp::Le,
                _ => unreachable!(),
            };
            let right = self.term()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(right));
        }
        Some(expr)
    }

    fn term(&mut self) -> Option<Expr> {
        let mut expr = self.factor()?;
        while self.match_any(&[TokenType::Plus, TokenType::Minus]) {
            let op = if self.previous().token_type == TokenType::Plus {
                BinOp::Add
            } else {
                BinOp::Sub
            };
            let right = self.factor()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(right));
        }
        Some(expr)
    }

    fn factor(&mut self) -> Option<Expr> {
        let mut expr = self.unary()?;
        while self.match_any(&[TokenType::Star, TokenType::Slash]) {
            let op = if self.previous().token_type == TokenType::Star {
                BinOp::Mul
            } else {
                BinOp::Div
            };
            let right = self.unary()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(right));
        }
        Some(expr)
    }

    fn unary(&mut self) -> Option<Expr> {
        if self.match_any(&[TokenType::Bang, TokenType::Minus]) {
            let op = if self.previous().token_type == TokenType::Bang {
                UnaryOp::Not
            } else {
                UnaryOp::Neg
            };
            let right = self.unary()?;
            return Some(Expr::Unary(op, Box::new(right)));
        }
        self.call()
    }

    fn call(&mut self) -> Option<Expr> {
        let mut expr = self.primary()?;
        loop {
            if self.match_token(TokenType::LeftParen) {
                expr = self.finish_call(expr)?;
            } else if self.match_token(TokenType::Dot) {
                let name = self.consume_identifier("Expect property name after '.'.")?;
                expr = Expr::Get(Box::new(expr), name);
            } else {
                break;
            }
        }
        Some(expr)
    }

    fn finish_call(&mut self, callee: Expr) -> Option<Expr> {
        let mut args = Vec::new();
        if !self.check(TokenType::RightParen) {
            loop {
                if args.len() >= 255 {
                    self.error_at_current("Can't have more than 255 arguments.");
                }
                args.push(self.expression()?);
                if !self.match_token(TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenType::RightParen, "Expect ')' after arguments.")?;
        Some(Expr::Call(Box::new(callee), args))
    }

    fn primary(&mut self) -> Option<Expr> {
        if self.match_token(TokenType::False) {
            return Some(Expr::Bool(false));
        }
        if self.match_token(TokenType::True) {
            return Some(Expr::Bool(true));
        }
        if self.match_token(TokenType::Nil) {
            return Some(Expr::Nil);
        }
        if self.match_token(TokenType::Number) {
            let n: f64 = self.previous().lexeme.parse().unwrap();
            return Some(Expr::Number(n));
        }
        if self.match_token(TokenType::String) {
            let s = self.previous().lexeme.clone();
            let s = s[1..s.len() - 1].to_string(); // trim quotes
            return Some(Expr::String(s));
        }
        if self.match_token(TokenType::Super) {
            self.consume(TokenType::Dot, "Expect '.' after 'super'.")?;
            let method = self.consume_identifier("Expect superclass method name.")?;
            return Some(Expr::Super(method));
        }
        if self.match_token(TokenType::This) {
            return Some(Expr::This);
        }
        if self.match_token(TokenType::Identifier) {
            return Some(Expr::Var(self.previous().lexeme.clone()));
        }
        if self.match_token(TokenType::LeftParen) {
            let expr = self.expression()?;
            self.consume(TokenType::RightParen, "Expect ')' after expression.")?;
            return Some(Expr::Grouping(Box::new(expr)));
        }

        self.error_at_current("Expect expression.");
        None
    }

    // ── Helpers ──────────────────────────────────────────────────

    fn match_token(&mut self, t: TokenType) -> bool {
        if self.check(t) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn match_any(&mut self, types: &[TokenType]) -> bool {
        for &t in types {
            if self.check(t) {
                self.advance();
                return true;
            }
        }
        false
    }

    fn check(&self, t: TokenType) -> bool {
        self.peek().token_type == t
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn is_at_end(&self) -> bool {
        self.peek().token_type == TokenType::Eof
    }

    fn consume(&mut self, t: TokenType, message: &str) -> Option<()> {
        if self.check(t) {
            self.advance();
            Some(())
        } else {
            self.error_at_current(message);
            None
        }
    }

    fn consume_identifier(&mut self, message: &str) -> Option<String> {
        if self.check(TokenType::Identifier) {
            self.advance();
            Some(self.previous().lexeme.clone())
        } else {
            self.error_at_current(message);
            None
        }
    }

    fn error(&mut self, message: &str) {
        self.error_at(&self.tokens[self.current - 1].clone(), message);
    }

    fn error_at_current(&mut self, message: &str) {
        self.error_at(&self.tokens[self.current].clone(), message);
    }

    fn error_at(&mut self, token: &Token, message: &str) {
        if self.panic_mode {
            return;
        }
        self.panic_mode = true;
        eprint!("[line {}] Error", token.line);
        match token.token_type {
            TokenType::Eof => eprint!(" at end"),
            TokenType::Error => {}
            _ => eprint!(" at '{}'", token.lexeme),
        }
        eprintln!(": {}", message);
        self.had_error = true;
    }

    fn synchronize(&mut self) {
        self.panic_mode = false;
        while !self.is_at_end() {
            if self.previous().token_type == TokenType::Semicolon {
                return;
            }
            match self.peek().token_type {
                TokenType::Class
                | TokenType::Fun
                | TokenType::Var
                | TokenType::For
                | TokenType::If
                | TokenType::While
                | TokenType::Print
                | TokenType::Return => return,
                _ => {
                    self.advance();
                }
            }
        }
    }
}
