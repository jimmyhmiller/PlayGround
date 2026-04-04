use crate::lexer::{Token, TokenKind};

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Bool(bool),
    Var(String),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnaryNeg(Box<Expr>),
    Call(String, Vec<Expr>),
    If(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Block(Vec<Stmt>, Option<Box<Expr>>),
    /// reset { body } — installs a prompt around body
    Reset(Box<Expr>),
    /// capture() — captures the continuation up to the enclosing reset, returns FrameSlice
    Capture,
    /// abort(val) — aborts to enclosing reset with val as the result
    Abort(Box<Expr>),
    /// resume(cont, val) — resumes a captured continuation with val
    Resume(Box<Expr>, Box<Expr>),
    /// clone(cont) — clones a continuation for multi-shot use
    CloneCont(Box<Expr>),
    While(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Expr),
    Assign(String, Expr),
    Expr(Expr),
    Return(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    Int,
    Cont,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: ValType,
}

#[derive(Debug, Clone)]
pub struct Decl {
    pub name: String,
    pub params: Vec<Param>,
    pub ret_ty: ValType,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub decls: Vec<Decl>,
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn advance(&mut self) -> &Token {
        let t = &self.tokens[self.pos];
        self.pos += 1;
        t
    }

    fn expect(&mut self, kind: &TokenKind) {
        let got = self.advance();
        if &got.kind != kind {
            panic!("expected {:?}, got {:?} at pos {}", kind, got.kind, got.pos);
        }
    }

    fn expect_ident(&mut self) -> String {
        let t = self.advance();
        match &t.kind {
            TokenKind::Ident(s) => s.clone(),
            other => panic!("expected identifier, got {:?} at pos {}", other, t.pos),
        }
    }

    fn parse_program(&mut self) -> Program {
        let mut decls = Vec::new();
        while *self.peek() != TokenKind::Eof {
            decls.push(self.parse_decl());
        }
        Program { decls }
    }

    fn parse_param(&mut self) -> Param {
        let name = self.expect_ident();
        let ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            self.parse_type()
        } else {
            ValType::Int
        };
        Param { name, ty }
    }

    fn parse_type(&mut self) -> ValType {
        match self.peek() {
            TokenKind::Cont => {
                self.advance();
                ValType::Cont
            }
            TokenKind::Ident(s) if s == "int" => {
                self.advance();
                ValType::Int
            }
            _ => ValType::Int,
        }
    }

    fn parse_decl(&mut self) -> Decl {
        self.expect(&TokenKind::Fn);
        let name = self.expect_ident();
        self.expect(&TokenKind::LParen);
        let mut params = Vec::new();
        if *self.peek() != TokenKind::RParen {
            params.push(self.parse_param());
            while *self.peek() == TokenKind::Comma {
                self.advance();
                params.push(self.parse_param());
            }
        }
        self.expect(&TokenKind::RParen);
        let ret_ty = if *self.peek() == TokenKind::Arrow {
            self.advance();
            self.parse_type()
        } else {
            ValType::Int
        };
        let body = self.parse_block_expr();
        Decl { name, params, ret_ty, body }
    }

    fn parse_block_expr(&mut self) -> Expr {
        self.expect(&TokenKind::LBrace);
        let mut stmts = Vec::new();
        let mut tail = None;

        loop {
            if *self.peek() == TokenKind::RBrace {
                self.advance();
                break;
            }

            match self.peek().clone() {
                TokenKind::Let => {
                    stmts.push(self.parse_let_stmt());
                }
                TokenKind::Return => {
                    self.advance();
                    let e = self.parse_expr();
                    if *self.peek() == TokenKind::Semicolon {
                        self.advance();
                    }
                    stmts.push(Stmt::Return(e));
                }
                _ => {
                    let e = self.parse_expr();
                    if *self.peek() == TokenKind::Semicolon {
                        self.advance();
                        stmts.push(Stmt::Expr(e));
                    } else if *self.peek() == TokenKind::RBrace {
                        tail = Some(Box::new(e));
                        self.advance();
                        break;
                    } else {
                        stmts.push(Stmt::Expr(e));
                    }
                }
            }
        }

        Expr::Block(stmts, tail)
    }

    fn parse_let_stmt(&mut self) -> Stmt {
        self.expect(&TokenKind::Let);
        let name = self.expect_ident();
        // Optional type annotation: let x: cont = ...
        if *self.peek() == TokenKind::Colon {
            self.advance();
            let _ = self.parse_type(); // consume type, used for documentation only
        }
        self.expect(&TokenKind::Eq);
        let val = self.parse_expr();
        if *self.peek() == TokenKind::Semicolon {
            self.advance();
        }
        Stmt::Let(name, val)
    }

    fn parse_expr(&mut self) -> Expr {
        let e = self.parse_comparison();

        if *self.peek() == TokenKind::Eq {
            if let Expr::Var(name) = e {
                self.advance();
                let rhs = self.parse_expr();
                return Expr::Block(vec![Stmt::Assign(name, rhs)], None);
            }
        }

        e
    }

    fn parse_comparison(&mut self) -> Expr {
        let mut left = self.parse_additive();
        loop {
            let op = match self.peek() {
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::BangEq => BinOp::Ne,
                TokenKind::Lt => BinOp::Lt,
                TokenKind::LtEq => BinOp::Le,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::GtEq => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let right = self.parse_additive();
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }
        left
    }

    fn parse_additive(&mut self) -> Expr {
        let mut left = self.parse_multiplicative();
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative();
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }
        left
    }

    fn parse_multiplicative(&mut self) -> Expr {
        let mut left = self.parse_unary();
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary();
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }
        left
    }

    fn parse_unary(&mut self) -> Expr {
        if *self.peek() == TokenKind::Minus {
            self.advance();
            let e = self.parse_primary();
            return Expr::UnaryNeg(Box::new(e));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Expr {
        match self.peek().clone() {
            TokenKind::Int(n) => {
                self.advance();
                Expr::Int(n)
            }
            TokenKind::True => {
                self.advance();
                Expr::Bool(true)
            }
            TokenKind::False => {
                self.advance();
                Expr::Bool(false)
            }
            TokenKind::Ident(name) => {
                self.advance();
                if *self.peek() == TokenKind::LParen {
                    self.advance();
                    let mut args = Vec::new();
                    if *self.peek() != TokenKind::RParen {
                        args.push(self.parse_expr());
                        while *self.peek() == TokenKind::Comma {
                            self.advance();
                            args.push(self.parse_expr());
                        }
                    }
                    self.expect(&TokenKind::RParen);
                    Expr::Call(name, args)
                } else {
                    Expr::Var(name)
                }
            }
            TokenKind::LParen => {
                self.advance();
                let e = self.parse_expr();
                self.expect(&TokenKind::RParen);
                e
            }
            TokenKind::If => {
                self.advance();
                let cond = self.parse_expr();
                let then_body = self.parse_block_expr();
                let else_body = if *self.peek() == TokenKind::Else {
                    self.advance();
                    if *self.peek() == TokenKind::If {
                        Some(Box::new(self.parse_primary()))
                    } else {
                        Some(Box::new(self.parse_block_expr()))
                    }
                } else {
                    None
                };
                Expr::If(Box::new(cond), Box::new(then_body), else_body)
            }
            TokenKind::While => {
                self.advance();
                let cond = self.parse_comparison();
                let body = self.parse_block_expr();
                Expr::While(Box::new(cond), Box::new(body))
            }
            TokenKind::Reset => {
                self.advance();
                let body = self.parse_block_expr();
                Expr::Reset(Box::new(body))
            }
            TokenKind::Shift => {
                // capture() — keyword reused as "capture current continuation"
                self.advance();
                self.expect(&TokenKind::LParen);
                self.expect(&TokenKind::RParen);
                Expr::Capture
            }
            TokenKind::Resume => {
                self.advance();
                self.expect(&TokenKind::LParen);
                let cont = self.parse_expr();
                self.expect(&TokenKind::Comma);
                let val = self.parse_expr();
                self.expect(&TokenKind::RParen);
                Expr::Resume(Box::new(cont), Box::new(val))
            }
            TokenKind::Clone => {
                self.advance();
                self.expect(&TokenKind::LParen);
                let cont = self.parse_expr();
                self.expect(&TokenKind::RParen);
                Expr::CloneCont(Box::new(cont))
            }
            TokenKind::Return => {
                // abort(val) - we reuse 'return' keyword inside reset as abort
                // Actually let's keep abort as a function-like syntax
                // and handle it as Ident("abort")
                panic!("unexpected 'return' in expression position")
            }
            other => {
                let pos = self.tokens[self.pos].pos;
                panic!("unexpected token {:?} at pos {}", other, pos)
            }
        }
    }
}

pub fn parse(tokens: Vec<Token>) -> Program {
    let mut parser = Parser { tokens, pos: 0 };
    parser.parse_program()
}
