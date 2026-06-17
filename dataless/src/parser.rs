//! Recursive-descent parser for programs (.dl) and declarations (.decl).
//! Keywords are plain identifiers matched contextually.

use crate::ast::*;
use crate::lexer::{Tok, Token};
use crate::repr::{Rep, Type};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

pub fn parse_program(toks: Vec<Token>) -> Result<Program, String> {
    let mut p = Parser { toks, pos: 0 };
    let mut stmts = Vec::new();
    p.skip_semis();
    while !p.at(&Tok::Eof) {
        stmts.push(p.statement()?);
        p.skip_semis();
    }
    Ok(stmts)
}

pub fn parse_decls(toks: Vec<Token>) -> Result<Decls, String> {
    let mut p = Parser { toks, pos: 0 };
    let mut decls = Vec::new();
    p.skip_semis();
    while !p.at(&Tok::Eof) {
        decls.push(p.decl()?);
        p.skip_semis();
    }
    Ok(decls)
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn line(&self) -> usize {
        self.toks[self.pos].line
    }

    fn at(&self, t: &Tok) -> bool {
        self.peek() == t
    }

    fn at_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }

    fn advance(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> Result<(), String> {
        if self.at(t) {
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

    fn eat_kw(&mut self, kw: &str) -> Result<(), String> {
        if self.at_kw(kw) {
            self.advance();
            Ok(())
        } else {
            Err(format!(
                "line {}: expected `{}`, found {:?}",
                self.line(),
                kw,
                self.peek()
            ))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.advance() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!(
                "line {}: expected identifier, found {:?}",
                self.line(),
                other
            )),
        }
    }

    fn skip_semis(&mut self) {
        while self.at(&Tok::Semi) {
            self.advance();
        }
    }

    // ===== statements ===================================================

    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        self.eat(&Tok::LBrace)?;
        self.skip_semis();
        let mut stmts = Vec::new();
        while !self.at(&Tok::RBrace) && !self.at(&Tok::Eof) {
            stmts.push(self.statement()?);
            self.skip_semis();
        }
        self.eat(&Tok::RBrace)?;
        Ok(stmts)
    }

    fn statement(&mut self) -> Result<Stmt, String> {
        if self.at_kw("let") {
            self.advance();
            let name = self.ident()?;
            self.eat(&Tok::Assign)?;
            let value = self.expr()?;
            return Ok(Stmt::Let(name, value));
        }
        if self.at_kw("print") {
            self.advance();
            self.eat(&Tok::LParen)?;
            let mut args = Vec::new();
            if !self.at(&Tok::RParen) {
                loop {
                    args.push(self.expr()?);
                    if self.at(&Tok::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.eat(&Tok::RParen)?;
            return Ok(Stmt::Print(args));
        }
        if self.at_kw("if") {
            return self.if_stmt();
        }
        if self.at_kw("while") {
            self.advance();
            let cond = self.expr()?;
            let body = self.block()?;
            return Ok(Stmt::While(cond, body));
        }
        if self.at_kw("repeat") {
            self.advance();
            let n = self.expr()?;
            self.eat_kw("with")?;
            let var = self.ident()?;
            let body = self.block()?;
            return Ok(Stmt::Repeat(n, var, body));
        }
        if self.at_kw("for") {
            self.advance();
            // `for each|all|every <coll> [such that (cond)] { body }`
            if self.at_kw("each") || self.at_kw("all") || self.at_kw("every") {
                self.advance();
            } else {
                return Err(format!("line {}: expected `each` after `for`", self.line()));
            }
            let coll = self.ident()?;
            let cond = if self.at_kw("such") {
                self.advance();
                self.eat_kw("that")?;
                self.eat(&Tok::LParen)?;
                let c = self.expr()?;
                self.eat(&Tok::RParen)?;
                Some(c)
            } else {
                None
            };
            let body = self.block()?;
            return Ok(Stmt::ForEach { coll, cond, body });
        }
        if self.at_kw("generate") {
            self.advance();
            let gen = self.ident()?;
            self.eat_kw("over")?;
            let coll = self.ident()?;
            self.eat_kw("such")?;
            self.eat_kw("that")?;
            self.eat(&Tok::LParen)?;
            let cond = self.expr()?;
            self.eat(&Tok::RParen)?;
            return Ok(Stmt::Generate { gen, coll, cond });
        }
        if self.at_kw("whenever") {
            self.advance();
            self.eat(&Tok::LParen)?;
            let cond = self.expr()?;
            self.eat(&Tok::RParen)?;
            let body = self.block()?;
            return Ok(Stmt::Whenever(cond, body));
        }
        if self.at_kw("delete") {
            self.advance();
            let e = self.expr()?;
            return Ok(Stmt::Delete(e));
        }

        // Otherwise: an assignment or a bare expression statement.
        let e = self.expr()?;
        if self.at(&Tok::Assign) {
            self.advance();
            let rhs = self.expr()?;
            return match e {
                Expr::Var(name) => Ok(Stmt::AssignVar(name, rhs)),
                Expr::Ref(name, idx) => Ok(Stmt::AssignRef(name, *idx, rhs)),
                _ => Err(format!("line {}: invalid assignment target", self.line())),
            };
        }
        Ok(Stmt::ExprStmt(e))
    }

    fn if_stmt(&mut self) -> Result<Stmt, String> {
        self.eat_kw("if")?;
        let cond = self.expr()?;
        let then = self.block()?;
        let els = if self.at_kw("else") {
            self.advance();
            if self.at_kw("if") {
                vec![self.if_stmt()?]
            } else {
                self.block()?
            }
        } else {
            Vec::new()
        };
        Ok(Stmt::If(cond, then, els))
    }

    // ===== expressions ==================================================

    fn expr(&mut self) -> Result<Expr, String> {
        self.or_expr()
    }

    fn or_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.and_expr()?;
        while self.at_kw("or") {
            self.advance();
            let right = self.and_expr()?;
            left = Expr::Binary(BinOp::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.not_expr()?;
        while self.at_kw("and") {
            self.advance();
            let right = self.not_expr()?;
            left = Expr::Binary(BinOp::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> Result<Expr, String> {
        if self.at_kw("not") {
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
        if self.at(&Tok::Minus) {
            self.advance();
            let e = self.unary()?;
            Ok(Expr::Unary(UnOp::Neg, Box::new(e)))
        } else {
            self.primary()
        }
    }

    fn primary(&mut self) -> Result<Expr, String> {
        // literals
        match self.peek().clone() {
            Tok::Int(n) => {
                self.advance();
                return Ok(Expr::Int(n));
            }
            Tok::Text(s) => {
                self.advance();
                return Ok(Expr::Text(s));
            }
            Tok::LParen => {
                self.advance();
                let e = self.expr()?;
                self.eat(&Tok::RParen)?;
                return Ok(e);
            }
            _ => {}
        }

        // keyword-led expression forms
        if self.at_kw("true") {
            self.advance();
            return Ok(Expr::Bool(true));
        }
        if self.at_kw("false") {
            self.advance();
            return Ok(Expr::Bool(false));
        }
        if self.at_kw("nil") {
            self.advance();
            return Ok(Expr::Nil);
        }
        if self.at_kw("it") {
            self.advance();
            return Ok(Expr::It);
        }
        if self.at_kw("size") {
            self.advance();
            self.eat(&Tok::LParen)?;
            let coll = self.ident()?;
            self.eat(&Tok::RParen)?;
            return Ok(Expr::Size(coll));
        }
        if self.at_kw("member") {
            self.advance();
            self.eat(&Tok::LParen)?;
            let coll = self.ident()?;
            self.eat(&Tok::Comma)?;
            let idx = self.expr()?;
            self.eat(&Tok::RParen)?;
            return Ok(Expr::MemberAt(coll, Box::new(idx)));
        }
        if self.at_kw("insert") {
            self.advance();
            if self.at_kw("after") {
                self.advance();
                let handle = self.expr()?;
                self.eat_kw("in")?;
                let coll = self.ident()?;
                return Ok(Expr::InsertAfter(coll, Box::new(handle)));
            }
            let coll = self.ident()?;
            return Ok(Expr::Insert(coll));
        }
        if self.at_kw("next") {
            self.advance();
            let gen = self.ident()?;
            return Ok(Expr::Next(gen));
        }
        if self.at_kw("there") || self.at_kw("exists") {
            if self.at_kw("there") {
                self.advance();
            }
            self.eat_kw("exists")?;
            let var = self.ident()?;
            self.eat_kw("in")?;
            let coll = self.ident()?;
            self.eat_kw("such")?;
            self.eat_kw("that")?;
            self.eat(&Tok::LParen)?;
            let cond = self.expr()?;
            self.eat(&Tok::RParen)?;
            return Ok(Expr::Exists {
                var,
                coll,
                cond: Box::new(cond),
            });
        }

        // identifier: either `name(index)` (canonical reference) or a variable
        if let Tok::Ident(_) = self.peek() {
            let name = self.ident()?;
            if self.at(&Tok::LParen) {
                self.advance();
                let idx = self.expr()?;
                self.eat(&Tok::RParen)?;
                return Ok(Expr::Ref(name, Box::new(idx)));
            }
            return Ok(Expr::Var(name));
        }

        Err(format!(
            "line {}: unexpected token {:?}",
            self.line(),
            self.peek()
        ))
    }

    // ===== declarations =================================================

    fn decl(&mut self) -> Result<Decl, String> {
        if self.at_kw("collection") {
            self.advance();
            let name = self.ident()?;
            self.eat_kw("as")?;
            let rep = self.representation()?;
            self.eat(&Tok::LBrace)?;
            self.skip_field_seps();
            let mut fields = Vec::new();
            while !self.at(&Tok::RBrace) && !self.at(&Tok::Eof) {
                let fname = self.ident()?;
                self.eat(&Tok::Colon)?;
                let ty = self.type_name()?;
                let init = if self.at(&Tok::Assign) {
                    self.advance();
                    Some(self.expr()?)
                } else {
                    None
                };
                fields.push(FieldDecl {
                    name: fname,
                    ty,
                    init,
                });
                self.skip_field_seps();
            }
            self.eat(&Tok::RBrace)?;
            return Ok(Decl::Collection { name, rep, fields });
        }
        if self.at_kw("computed") {
            self.advance();
            let name = self.ident()?;
            self.eat(&Tok::LParen)?;
            let param = self.ident()?;
            self.eat(&Tok::RParen)?;
            self.eat(&Tok::Assign)?;
            let body = self.expr()?;
            return Ok(Decl::Computed { name, param, body });
        }
        Err(format!(
            "line {}: expected `collection` or `computed`, found {:?}",
            self.line(),
            self.peek()
        ))
    }

    fn skip_field_seps(&mut self) {
        while self.at(&Tok::Comma) || self.at(&Tok::Semi) {
            self.advance();
        }
    }

    fn representation(&mut self) -> Result<Rep, String> {
        let word = self.ident()?;
        match word.to_ascii_uppercase().as_str() {
            "ARRAY" => Ok(Rep::Array),
            "LIST" => Ok(Rep::List),
            "DOUBLE_LIST" | "DOUBLELIST" => Ok(Rep::DoubleList),
            other => Err(format!("line {}: unknown representation `{}`", self.line(), other)),
        }
    }

    fn type_name(&mut self) -> Result<Type, String> {
        let word = self.ident()?;
        match word.to_ascii_lowercase().as_str() {
            "int" => Ok(Type::Int),
            "text" => Ok(Type::Text),
            "bool" => Ok(Type::Bool),
            other => Err(format!("line {}: unknown type `{}`", self.line(), other)),
        }
    }
}
