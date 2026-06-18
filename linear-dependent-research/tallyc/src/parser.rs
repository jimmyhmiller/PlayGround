//! Recursive-descent parser for the v0 surface language.

use crate::ast::{Expr, Program, Stmt};
use crate::lexer::{lex, Tok};

pub fn parse(src: &str) -> Result<Program, String> {
    let toks = lex(src)?;
    let mut p = Parser { toks, i: 0 };
    p.block()
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

    fn block(&mut self) -> Result<Program, String> {
        let mut out = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            out.push(self.stmt()?);
        }
        Ok(out)
    }

    fn stmt(&mut self) -> Result<Stmt, String> {
        if let Tok::Ident(kw) = self.peek().clone() {
            if kw == "let" {
                self.bump();
                let name = self.eat_ident()?;
                self.eat_punc('=')?;
                let rhs = self.expr()?;
                self.eat_punc(';')?;
                return Ok(Stmt::Let(name, rhs));
            }
            if kw == "free" {
                self.bump();
                let name = self.eat_ident()?;
                self.eat_punc(';')?;
                return Ok(Stmt::Free(name));
            }
        }
        // expression statement, or a field-write if it is `path.fld = rhs;`
        let e = self.expr()?;
        if let Expr::Field(base, fld) = e {
            if self.is_punc('=') {
                self.bump();
                let rhs = self.expr()?;
                self.eat_punc(';')?;
                return Ok(Stmt::Write(*base, fld, rhs));
            }
            // not a write; reconstruct and finish as an expression statement
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
            Tok::Ident(s) if s == "alloc" => self.alloc(),
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

    fn alloc(&mut self) -> Result<Expr, String> {
        self.bump(); // alloc
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
        Ok(Expr::Alloc(fields))
    }
}
