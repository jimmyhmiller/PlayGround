//! Recursive-descent parser: tokens → AST.

use super::ast::*;
use super::lexer::{Tok, Token};

pub fn parse(tokens: Vec<Token>) -> Result<Program, String> {
    let mut p = Parser { tokens, pos: 0 };
    let mut items = Vec::new();
    while !p.at(&Tok::Eof) {
        items.push(p.item()?);
    }
    Ok(Program { items })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }
    fn line(&self) -> usize {
        self.tokens[self.pos].line
    }
    fn at(&self, t: &Tok) -> bool {
        self.peek() == t
    }
    fn bump(&mut self) -> Tok {
        let t = self.tokens[self.pos].tok.clone();
        self.pos += 1;
        t
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, t: &Tok) -> Result<(), String> {
        if self.eat(t) {
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
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("line {}: expected identifier, found {other:?}", self.line())),
        }
    }

    fn item(&mut self) -> Result<Item, String> {
        match self.peek() {
            Tok::Struct => Ok(Item::Struct(self.struct_def()?)),
            Tok::Fn => Ok(Item::Fn(self.fn_def()?)),
            other => Err(format!("line {}: expected `struct` or `fn`, found {other:?}", self.line())),
        }
    }

    fn type_expr(&mut self) -> Result<TypeExpr, String> {
        if self.eat(&Tok::Amp) {
            return Ok(TypeExpr::Ref(self.ident()?));
        }
        if self.eat(&Tok::LParen) {
            self.expect(&Tok::RParen)?;
            return Ok(TypeExpr::Unit);
        }
        match self.ident()?.as_str() {
            "i64" => Ok(TypeExpr::I64),
            "bool" => Ok(TypeExpr::Bool),
            other => Err(format!(
                "line {}: unknown type `{other}` (write `&{other}` for a struct reference)",
                self.line()
            )),
        }
    }

    fn struct_def(&mut self) -> Result<StructDef, String> {
        self.expect(&Tok::Struct)?;
        let name = self.ident()?;
        self.expect(&Tok::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(&Tok::RBrace) {
            let fname = self.ident()?;
            self.expect(&Tok::Colon)?;
            let ty = self.type_expr()?;
            let default = if self.eat(&Tok::Eq) {
                Some(self.expr(true)?)
            } else {
                None
            };
            fields.push(FieldDef { name: fname, ty, default });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(StructDef { name, fields })
    }

    fn fn_def(&mut self) -> Result<FnDef, String> {
        self.expect(&Tok::Fn)?;
        let name = self.ident()?;
        self.expect(&Tok::LParen)?;
        let mut params = Vec::new();
        while !self.at(&Tok::RParen) {
            let pname = self.ident()?;
            self.expect(&Tok::Colon)?;
            let ty = self.type_expr()?;
            params.push(Param { name: pname, ty });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        let ret = if self.eat(&Tok::Arrow) {
            self.type_expr()?
        } else {
            TypeExpr::Unit
        };
        let body = self.block()?;
        Ok(FnDef { name, params, ret, body })
    }

    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect(&Tok::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(&Tok::RBrace) {
            stmts.push(self.stmt()?);
        }
        self.expect(&Tok::RBrace)?;
        Ok(stmts)
    }

    fn stmt(&mut self) -> Result<Stmt, String> {
        match self.peek() {
            Tok::Let => {
                self.bump();
                let name = self.ident()?;
                self.expect(&Tok::Eq)?;
                let value = self.expr(true)?;
                self.expect(&Tok::Semi)?;
                Ok(Stmt::Let { name, value })
            }
            Tok::Return => {
                self.bump();
                let value = self.expr(true)?;
                self.expect(&Tok::Semi)?;
                Ok(Stmt::Return(value))
            }
            Tok::Emit => {
                self.bump();
                self.expect(&Tok::LParen)?;
                let value = self.expr(true)?;
                self.expect(&Tok::RParen)?;
                self.expect(&Tok::Semi)?;
                Ok(Stmt::Emit(value))
            }
            Tok::Yield => {
                self.bump();
                self.expect(&Tok::Semi)?;
                Ok(Stmt::Yield)
            }
            Tok::If => {
                self.bump();
                let cond = self.expr(false)?; // no struct literal in a condition
                let then = self.block()?;
                let els = if self.eat(&Tok::Else) { self.block()? } else { Vec::new() };
                Ok(Stmt::If { cond, then, els })
            }
            Tok::While => {
                self.bump();
                let cond = self.expr(false)?;
                let body = self.block()?;
                Ok(Stmt::While { cond, body })
            }
            // assignment `name = expr;` or a bare expression statement
            Tok::Ident(_) if matches!(self.tokens[self.pos + 1].tok, Tok::Eq) => {
                let name = self.ident()?;
                self.expect(&Tok::Eq)?;
                let value = self.expr(true)?;
                self.expect(&Tok::Semi)?;
                Ok(Stmt::Assign { name, value })
            }
            _ => {
                let e = self.expr(true)?;
                self.expect(&Tok::Semi)?;
                Ok(Stmt::Expr(e))
            }
        }
    }

    // expr := comparison ; comparison := additive ((< | >) additive)* ;
    // additive := postfix ((+ | -) postfix)*
    fn expr(&mut self, allow_struct: bool) -> Result<Expr, String> {
        let mut left = self.additive(allow_struct)?;
        while matches!(self.peek(), Tok::Lt | Tok::Gt) {
            let op = if self.eat(&Tok::Lt) { BinOp::Lt } else { self.expect(&Tok::Gt).map(|_| ())?; BinOp::Gt };
            let right = self.additive(allow_struct)?;
            left = Expr::Binary { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn additive(&mut self, allow_struct: bool) -> Result<Expr, String> {
        let mut left = self.postfix(allow_struct)?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let right = self.postfix(allow_struct)?;
            left = Expr::Binary { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn postfix(&mut self, allow_struct: bool) -> Result<Expr, String> {
        let mut e = self.primary(allow_struct)?;
        while self.eat(&Tok::Dot) {
            let field = self.ident()?;
            e = Expr::Field { object: Box::new(e), field };
        }
        Ok(e)
    }

    fn primary(&mut self, allow_struct: bool) -> Result<Expr, String> {
        match self.peek().clone() {
            Tok::Int(n) => {
                self.bump();
                Ok(Expr::Int(n))
            }
            Tok::True => {
                self.bump();
                Ok(Expr::Bool(true))
            }
            Tok::False => {
                self.bump();
                Ok(Expr::Bool(false))
            }
            Tok::Amp => {
                // `&expr` — a reference-taking marker; structs are already
                // references, so it's accepted and ignored (Rust-flavored sugar).
                self.bump();
                self.primary(allow_struct)
            }
            Tok::LParen => {
                self.bump();
                if self.eat(&Tok::RParen) {
                    return Ok(Expr::Unit);
                }
                let e = self.expr(true)?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Tok::Ident(name) => {
                self.bump();
                if self.at(&Tok::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    while !self.at(&Tok::RParen) {
                        args.push(self.expr(true)?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(Expr::Call { name, args })
                } else if allow_struct && self.at(&Tok::LBrace) {
                    self.bump();
                    let mut fields = Vec::new();
                    while !self.at(&Tok::RBrace) {
                        let fname = self.ident()?;
                        self.expect(&Tok::Colon)?;
                        let value = self.expr(true)?;
                        fields.push((fname, value));
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RBrace)?;
                    Ok(Expr::StructLit { name, fields })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(format!("line {}: unexpected {other:?} in expression", self.line())),
        }
    }
}
