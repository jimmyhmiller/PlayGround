//! Recursive-descent parser.

use crate::ast::*;
use crate::lexer::{Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

pub fn parse(toks: Vec<Token>) -> Result<Program, String> {
    let mut p = Parser { toks, pos: 0 };
    let mut stmts = Vec::new();
    while !p.check(&Tok::Eof) {
        stmts.push(p.statement()?);
    }
    Ok(stmts)
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn line(&self) -> usize {
        self.toks[self.pos].line
    }

    fn check(&self, t: &Tok) -> bool {
        self.peek() == t
    }

    fn advance(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> Result<(), String> {
        if self.check(t) {
            self.advance();
            Ok(())
        } else {
            Err(format!(
                "line {}: expected {:?}, found {:?}",
                self.line(),
                t,
                self.peek()
            ))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.advance() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("line {}: expected identifier, found {:?}", self.line(), other)),
        }
    }

    // ----- statements ---------------------------------------------------

    fn statement(&mut self) -> Result<Stmt, String> {
        match self.peek() {
            Tok::Let => self.let_stmt(),
            Tok::If => self.if_stmt(),
            Tok::While => self.while_stmt(),
            Tok::For => self.for_stmt(),
            Tok::Break => {
                self.advance();
                Ok(Stmt::Break)
            }
            Tok::Continue => {
                self.advance();
                Ok(Stmt::Continue)
            }
            Tok::Ident(_) => {
                // Could be `name = expr` (assignment) or a bare expression.
                if self.peek_is_assignment() {
                    let name = self.ident()?;
                    self.eat(&Tok::Assign)?;
                    let value = self.expr()?;
                    Ok(Stmt::Assign { name, value })
                } else {
                    Ok(Stmt::Expr(self.expr()?))
                }
            }
            _ => Ok(Stmt::Expr(self.expr()?)),
        }
    }

    /// Lookahead: is the current identifier immediately followed by `=`?
    fn peek_is_assignment(&self) -> bool {
        matches!(self.toks.get(self.pos).map(|t| &t.tok), Some(Tok::Ident(_)))
            && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Assign))
    }

    fn let_stmt(&mut self) -> Result<Stmt, String> {
        self.eat(&Tok::Let)?;
        let name = self.ident()?;
        self.eat(&Tok::Assign)?;
        let value = self.expr()?;
        Ok(Stmt::Let { name, value })
    }

    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        self.eat(&Tok::LBrace)?;
        let mut stmts = Vec::new();
        while !self.check(&Tok::RBrace) && !self.check(&Tok::Eof) {
            stmts.push(self.statement()?);
        }
        self.eat(&Tok::RBrace)?;
        Ok(stmts)
    }

    fn if_stmt(&mut self) -> Result<Stmt, String> {
        self.eat(&Tok::If)?;
        let cond = self.expr()?;
        let then = self.block()?;
        let els = if self.check(&Tok::Else) {
            self.advance();
            if self.check(&Tok::If) {
                vec![self.if_stmt()?]
            } else {
                self.block()?
            }
        } else {
            Vec::new()
        };
        Ok(Stmt::If { cond, then, els })
    }

    fn while_stmt(&mut self) -> Result<Stmt, String> {
        self.eat(&Tok::While)?;
        let cond = self.expr()?;
        let body = self.block()?;
        Ok(Stmt::While { cond, body })
    }

    fn for_stmt(&mut self) -> Result<Stmt, String> {
        self.eat(&Tok::For)?;
        let var = self.ident()?;
        self.eat(&Tok::In)?;
        let iter = self.expr()?;
        let body = self.block()?;
        Ok(Stmt::For { var, iter, body })
    }

    // ----- expressions (precedence climbing) ----------------------------

    fn expr(&mut self) -> Result<Expr, String> {
        self.or_expr()
    }

    fn or_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.and_expr()?;
        while self.check(&Tok::Or) {
            self.advance();
            let right = self.and_expr()?;
            left = Expr::Binary(BinOp::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.not_expr()?;
        while self.check(&Tok::And) {
            self.advance();
            let right = self.not_expr()?;
            left = Expr::Binary(BinOp::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> Result<Expr, String> {
        if self.check(&Tok::Not) {
            self.advance();
            let e = self.not_expr()?;
            Ok(Expr::Unary(UnOp::Not, Box::new(e)))
        } else {
            self.cmp_expr()
        }
    }

    fn cmp_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.add_expr()?;
        loop {
            let op = match self.peek() {
                Tok::Eq => BinOp::Eq,
                Tok::Ne => BinOp::Ne,
                Tok::Lt => BinOp::Lt,
                Tok::Le => BinOp::Le,
                Tok::Gt => BinOp::Gt,
                Tok::Ge => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let right = self.add_expr()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn add_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.mul_expr()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.mul_expr()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn mul_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.unary()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.check(&Tok::Minus) {
            self.advance();
            let e = self.unary()?;
            Ok(Expr::Unary(UnOp::Neg, Box::new(e)))
        } else {
            self.primary()
        }
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Tok::Int(n) => {
                self.advance();
                Ok(Expr::Int(n))
            }
            Tok::Str(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Tok::True => {
                self.advance();
                Ok(Expr::Bool(true))
            }
            Tok::False => {
                self.advance();
                Ok(Expr::Bool(false))
            }
            Tok::Nil => {
                self.advance();
                Ok(Expr::Nil)
            }
            Tok::LParen => {
                self.advance();
                let e = self.expr()?;
                self.eat(&Tok::RParen)?;
                Ok(e)
            }
            Tok::LBracket => self.list_literal(),
            Tok::Ident(name) => {
                self.advance();
                if self.check(&Tok::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    if !self.check(&Tok::RParen) {
                        loop {
                            args.push(self.expr()?);
                            if self.check(&Tok::Comma) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.eat(&Tok::RParen)?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(format!("line {}: unexpected token {:?}", self.line(), other)),
        }
    }

    fn list_literal(&mut self) -> Result<Expr, String> {
        self.eat(&Tok::LBracket)?;
        let mut items = Vec::new();
        if !self.check(&Tok::RBracket) {
            loop {
                items.push(self.expr()?);
                if self.check(&Tok::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.eat(&Tok::RBracket)?;
        Ok(Expr::List(items))
    }
}
