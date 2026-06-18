//! Recursive-descent parser for the v0.2 surface language.

use crate::ast::{Expr, Program, Stmt, StructDecl, Ty};
use crate::lexer::{lex, Tok};

pub fn parse(src: &str) -> Result<Program, String> {
    let toks = lex(src)?;
    let mut p = Parser { toks, i: 0 };
    p.program()
}

struct Parser {
    toks: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.i]
    }
    fn bump(&mut self) -> Tok {
        let t = self.toks[self.i].clone();
        self.i += 1;
        t
    }
    fn is_punc(&self, c: char) -> bool {
        matches!(self.peek(), Tok::Punc(x) if *x == c)
    }
    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }
    fn eat_punc(&mut self, c: char) -> Result<(), String> {
        match self.bump() {
            Tok::Punc(x) if x == c => Ok(()),
            t => Err(format!("expected {c:?}, got {t:?}")),
        }
    }
    fn eat_ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            t => Err(format!("expected identifier, got {t:?}")),
        }
    }

    fn program(&mut self) -> Result<Program, String> {
        let mut structs = Vec::new();
        let mut body = Vec::new();
        loop {
            match self.peek() {
                Tok::Eof => break,
                Tok::Ident(s) if s == "struct" => structs.push(self.struct_decl()?),
                _ => body.push(self.stmt()?),
            }
        }
        Ok(Program { structs, body })
    }

    fn struct_decl(&mut self) -> Result<StructDecl, String> {
        self.bump(); // struct
        let name = self.eat_ident()?;
        self.eat_punc('{')?;
        let mut fields = Vec::new();
        while !self.is_punc('}') {
            let f = self.eat_ident()?;
            self.eat_punc(':')?;
            let t = self.ty()?;
            fields.push((f, t));
            if self.is_punc(',') {
                self.bump();
            }
        }
        self.eat_punc('}')?;
        Ok(StructDecl { name, fields })
    }

    fn ty(&mut self) -> Result<Ty, String> {
        let name = self.eat_ident()?;
        match name.as_str() {
            "Int" => Ok(Ty::Int),
            "Unit" => Ok(Ty::Unit),
            "Own" => {
                self.eat_punc('<')?;
                let s = self.eat_ident()?;
                self.eat_punc('>')?;
                Ok(Ty::Own(s))
            }
            "Ptr" => {
                self.eat_punc('<')?;
                let s = self.eat_ident()?;
                self.eat_punc('>')?;
                Ok(Ty::Ptr(s))
            }
            other => Err(format!("unknown type `{other}`")),
        }
    }

    fn stmt(&mut self) -> Result<Stmt, String> {
        if self.is_kw("let") {
            self.bump();
            let name = self.eat_ident()?;
            let ann = if self.is_punc(':') {
                self.bump();
                Some(self.ty()?)
            } else {
                None
            };
            self.eat_punc('=')?;
            let rhs = self.expr()?;
            self.eat_punc(';')?;
            return Ok(Stmt::Let(name, ann, rhs));
        }
        if self.is_kw("free") {
            self.bump();
            let name = self.eat_ident()?;
            self.eat_punc(';')?;
            return Ok(Stmt::Free(name));
        }
        let e = self.expr()?;
        if let Expr::Field(base, fld) = e {
            if self.is_punc('=') {
                self.bump();
                let rhs = self.expr()?;
                self.eat_punc(';')?;
                return Ok(Stmt::Write(*base, fld, rhs));
            }
            self.eat_punc(';')?;
            return Ok(Stmt::Expr(Expr::Field(base, fld)));
        }
        self.eat_punc(';')?;
        Ok(Stmt::Expr(e))
    }

    fn expr(&mut self) -> Result<Expr, String> {
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.primary()?;
        while self.is_punc('.') {
            self.bump();
            let fld = self.eat_ident()?;
            e = Expr::Field(Box::new(e), fld);
        }
        Ok(e)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Tok::Int(n) => {
                self.bump();
                Ok(Expr::Int(n))
            }
            Tok::Ident(s) if s == "null" => {
                self.bump();
                Ok(Expr::Null)
            }
            Tok::Ident(s) if s == "unit" => {
                self.bump();
                Ok(Expr::Unit)
            }
            Tok::Ident(s) if s == "alloc" => {
                self.bump();
                let name = self.eat_ident()?;
                self.eat_punc('{')?;
                let mut fields = Vec::new();
                while !self.is_punc('}') {
                    let f = self.eat_ident()?;
                    self.eat_punc(':')?;
                    let e = self.expr()?;
                    fields.push((f, e));
                    if self.is_punc(',') {
                        self.bump();
                    }
                }
                self.eat_punc('}')?;
                Ok(Expr::Alloc(name, fields))
            }
            Tok::Ident(s) if s == "addr" => {
                self.bump();
                self.eat_punc('(')?;
                let n = self.eat_ident()?;
                self.eat_punc(')')?;
                Ok(Expr::AddrOf(n))
            }
            Tok::Ident(s) => {
                self.bump();
                Ok(Expr::Var(s))
            }
            Tok::Punc('(') => {
                self.bump();
                let e = self.expr()?;
                self.eat_punc(')')?;
                Ok(e)
            }
            t => Err(format!("unexpected token {t:?}")),
        }
    }
}
