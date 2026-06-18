//! Recursive-descent parser for the v0.3 surface language (functions added).

use crate::ast::{Expr, Func, Idx, Param, Program, Stmt, StructDecl, Ty};
use crate::lexer::{lex, Tok};
use crate::mult::Mult;

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
    fn peek2(&self) -> &Tok {
        &self.toks[(self.i + 1).min(self.toks.len() - 1)]
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
        let mut funcs = Vec::new();
        loop {
            match self.peek() {
                Tok::Eof => break,
                Tok::Ident(s) if s == "struct" => structs.push(self.struct_decl()?),
                Tok::Ident(s) if s == "fn" => funcs.push(self.func()?),
                t => return Err(format!("expected `struct` or `fn` at top level, got {t:?}")),
            }
        }
        Ok(Program { structs, funcs })
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

    fn func(&mut self) -> Result<Func, String> {
        self.bump(); // fn
        let name = self.eat_ident()?;
        self.eat_punc('(')?;
        let mut params = Vec::new();
        while !self.is_punc(')') {
            params.push(self.param()?);
            if self.is_punc(',') {
                self.bump();
            }
        }
        self.eat_punc(')')?;
        // optional `-> Ty`, default Unit
        let ret = if self.is_punc('-') || self.is_kw("->") {
            // tokens: '-' '>'  (lexer emits them separately)
            self.eat_arrow()?;
            self.ty()?
        } else {
            Ty::Unit
        };
        self.eat_punc('{')?;
        let (body, tail) = self.body()?;
        self.eat_punc('}')?;
        Ok(Func {
            name,
            params,
            ret,
            body,
            tail,
        })
    }

    fn eat_arrow(&mut self) -> Result<(), String> {
        // `->`  (the lexer doesn't special-case it; it is '-' then '>')
        self.eat_punc('-')?;
        self.eat_punc('>')
    }

    /// `[mult] name : Ty`, where mult ∈ {0, 1, w} is optional.
    fn param(&mut self) -> Result<Param, String> {
        let mult = self.opt_mult();
        let name = self.eat_ident()?;
        self.eat_punc(':')?;
        let ty = self.ty()?;
        Ok(Param { mult, name, ty })
    }

    fn opt_mult(&mut self) -> Option<Mult> {
        // a multiplicity is one of `0`, `1`, `w` AND must be followed by an
        // identifier (the parameter name), to avoid eating a name like `w`.
        let m = match self.peek() {
            Tok::Int(0) => Some(Mult::Zero),
            Tok::Int(1) => Some(Mult::One),
            Tok::Ident(s) if s == "w" => Some(Mult::Omega),
            _ => None,
        };
        if m.is_some() && matches!(self.peek2(), Tok::Ident(_)) {
            self.bump();
            m
        } else {
            None
        }
    }

    fn ty(&mut self) -> Result<Ty, String> {
        let name = self.eat_ident()?;
        match name.as_str() {
            "Int" => Ok(Ty::Int),
            "Unit" => Ok(Ty::Unit),
            "Nat" => Ok(Ty::Nat),
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
            "Vec" => {
                self.eat_punc('<')?;
                let i = self.idx()?;
                self.eat_punc('>')?;
                Ok(Ty::Vec(i))
            }
            other => Err(format!("unknown type `{other}`")),
        }
    }

    /// a length index: `k` | `n` | `n + k`
    fn idx(&mut self) -> Result<Idx, String> {
        match self.bump() {
            Tok::Int(k) => Ok(Idx::Lit(k as u32)),
            Tok::Ident(n) => {
                if self.is_punc('+') {
                    self.bump();
                    match self.bump() {
                        Tok::Int(k) => Ok(Idx::Var(n, k as u32)),
                        t => Err(format!("expected a literal after `+` in index, got {t:?}")),
                    }
                } else {
                    Ok(Idx::Var(n, 0))
                }
            }
            t => Err(format!("expected an index, got {t:?}")),
        }
    }

    /// a function body: zero or more `;`-terminated statements, then an optional
    /// trailing expression (the return value).
    fn body(&mut self) -> Result<(Vec<Stmt>, Option<Expr>), String> {
        let mut stmts = Vec::new();
        let mut tail = None;
        while !self.is_punc('}') {
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
                stmts.push(Stmt::Let(name, ann, rhs));
                continue;
            }
            if self.is_kw("free") {
                self.bump();
                let name = self.eat_ident()?;
                self.eat_punc(';')?;
                stmts.push(Stmt::Free(name));
                continue;
            }
            let e = self.expr()?;
            if let Expr::Field(base, fld) = e.clone() {
                if self.is_punc('=') {
                    self.bump();
                    let rhs = self.expr()?;
                    self.eat_punc(';')?;
                    stmts.push(Stmt::Write(*base, fld, rhs));
                    continue;
                }
            }
            // either a `;`-terminated expr statement, or the trailing tail expr
            if self.is_punc(';') {
                self.bump();
                stmts.push(Stmt::Expr(e));
            } else {
                tail = Some(e);
                break;
            }
        }
        Ok((stmts, tail))
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
                // function call `s( ... )` vs a plain variable
                if self.is_punc('(') {
                    self.bump();
                    let mut args = Vec::new();
                    while !self.is_punc(')') {
                        args.push(self.expr()?);
                        if self.is_punc(',') {
                            self.bump();
                        }
                    }
                    self.eat_punc(')')?;
                    Ok(Expr::Call(s, args))
                } else {
                    Ok(Expr::Var(s))
                }
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
