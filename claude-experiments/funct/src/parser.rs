//! Recursive-descent parser. Desugars `|>` pipes (with `_` holes), UFCS
//! method-call syntax, subjectless `match`, and string interpolation.

use crate::ast::*;
use crate::lexer::{lex, StrPart, Tok, Token};

pub fn parse(src: &str) -> Result<Program, String> {
    let toks = lex(src)?;
    let mut p = Parser { toks, pos: 0, no_struct: 0, no_bare_lambda: 0 };
    p.parse_program()
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    /// >0 while parsing a `match`/`if`/`while`/`for` subject: `TypeName {`
    /// > is then NOT a record-style constructor (the brace starts the body).
    no_struct: u32,
    /// >0 while parsing a match-arm guard: a bare `ident =>` is then the
    /// > arm's arrow, not a lambda (parenthesize lambdas in guards).
    no_bare_lambda: u32,
}

impl Parser {
    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).map(|t| &t.tok).unwrap_or(&Tok::Eof)
    }

    fn peek_at(&self, n: usize) -> &Tok {
        self.toks.get(self.pos + n).map(|t| &t.tok).unwrap_or(&Tok::Eof)
    }

    fn line(&self) -> u32 {
        self.toks.get(self.pos).map(|t| t.line).unwrap_or(0)
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks.get(self.pos).map(|t| t.tok.clone()).unwrap_or(Tok::Eof);
        self.pos += 1;
        t
    }

    fn at(&self, t: &Tok) -> bool {
        self.peek() == t
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
            Err(format!("line {}: expected {:?}, found {:?}", self.line(), t, self.peek()))
        }
    }

    fn skip_nl(&mut self) {
        while self.at(&Tok::Newline) {
            self.pos += 1;
        }
    }

    fn err<T>(&self, msg: &str) -> Result<T, String> {
        Err(format!("line {}: {}", self.line(), msg))
    }

    /// Entering a bracketed construct clears subject-position restrictions
    /// (`no_struct`, `no_bare_lambda`); restore on exit.
    fn save_flags(&mut self) -> (u32, u32) {
        let s = (self.no_struct, self.no_bare_lambda);
        self.no_struct = 0;
        self.no_bare_lambda = 0;
        s
    }

    fn restore_flags(&mut self, s: (u32, u32)) {
        self.no_struct = s.0;
        self.no_bare_lambda = s.1;
    }

    // ---------- program / items ----------

    fn parse_program(&mut self) -> Result<Program, String> {
        let mut items = Vec::new();
        self.skip_nl();
        while !self.at(&Tok::Eof) {
            items.push(self.parse_item()?);
            if !self.at(&Tok::Eof)
                && !self.eat(&Tok::Newline) {
                    return self.err(&format!("expected end of statement, found {:?}", self.peek()));
                }
            self.skip_nl();
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Result<Item, String> {
        // attributes come first: #[test]
        let mut attrs = Vec::new();
        while self.at(&Tok::Hash) {
            self.bump();
            self.expect(&Tok::LBracket)?;
            let name = match self.bump() {
                Tok::Ident(n) => n,
                t => return Err(format!("line {}: expected attribute name, found {:?}", self.line(), t)),
            };
            self.expect(&Tok::RBracket)?;
            if name != "test" {
                return self.err(&format!("unknown attribute `#[{}]` (known: #[test])", name));
            }
            attrs.push(name);
            self.skip_nl();
        }
        let exported = self.eat(&Tok::Export);
        if !attrs.is_empty() && !self.at(&Tok::Fn) {
            return self.err("attributes can only be applied to `fn` items");
        }
        if self.at(&Tok::Extern) {
            if exported {
                return self.err("`extern` declarations cannot be exported");
            }
            return self.parse_extern();
        }
        match self.peek() {
            Tok::Import => {
                if exported {
                    return self.err("`export import` is not supported; re-export by name instead");
                }
                self.parse_import()
            }
            Tok::Fn => {
                let def = self.parse_fn_def(exported, attrs)?;
                if def.attrs.iter().any(|a| a == "test") && !def.params.is_empty() {
                    return Err(format!(
                        "line {}: #[test] functions take no arguments",
                        def.line
                    ));
                }
                Ok(Item::Fn(def))
            }
            Tok::Type => Ok(Item::Type(self.parse_type_def()?)),
            Tok::Let => {
                let line = self.line();
                self.bump();
                if self.at(&Tok::Mut) {
                    return self.err("`mut` is not allowed at top level; use an atom: let x = atom(v)");
                }
                let pattern = self.parse_pattern()?;
                self.expect(&Tok::Assign)?;
                let expr = self.parse_expr()?;
                Ok(Item::Let { pattern, expr, exported, line })
            }
            _ => {
                if exported {
                    return self.err("only `fn`, `let` and `type` items can be exported");
                }
                let e = self.parse_expr()?;
                if matches!(
                    self.peek(),
                    Tok::Assign | Tok::PlusEq | Tok::MinusEq | Tok::StarEq | Tok::SlashEq | Tok::PercentEq
                ) {
                    return self.err("assignment is not allowed at top level (top-level bindings are immutable)");
                }
                Ok(Item::Expr(e))
            }
        }
    }

    /// `extern fn name(params)` / `extern let name` — host-interface
    /// declarations, no body/initializer.
    fn parse_extern(&mut self) -> Result<Item, String> {
        let line = self.line();
        self.expect(&Tok::Extern)?;
        if self.eat(&Tok::Let) {
            let name = match self.bump() {
                Tok::Ident(n) => n,
                t => return Err(format!("line {}: expected name after `extern let`, found {:?}", line, t)),
            };
            if self.at(&Tok::Assign) {
                return self.err("`extern let` declares a host-provided value and has no initializer");
            }
            return Ok(Item::ExternLet { name, line });
        }
        self.expect(&Tok::Fn)?;
        let name = match self.bump() {
            Tok::Ident(n) => n,
            t => return Err(format!("line {}: expected function name after `extern fn`, found {:?}", line, t)),
        };
        self.expect(&Tok::LParen)?;
        let mut params = Vec::new();
        while !self.at(&Tok::RParen) {
            match self.bump() {
                Tok::Ident(n) => params.push(n),
                Tok::Underscore => params.push("_".to_string()),
                t => return Err(format!("line {}: extern parameters are plain names, found {:?}", self.line(), t)),
            }
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        if self.at(&Tok::Assign) || self.at(&Tok::LBrace) {
            return self.err("`extern fn` declares a host function and has no body");
        }
        Ok(Item::Extern { name, params, line })
    }

    /// `import { a, b as c } from "path"` | `import "path" (as alias)?`
    /// There is deliberately NO wildcard form.
    fn parse_import(&mut self) -> Result<Item, String> {
        let line = self.line();
        self.expect(&Tok::Import)?;
        const NO_WILDCARD: &str =
            "wildcard imports are not allowed; list names explicitly (`import { a, b } from \"m\"`) \
             or import qualified (`import \"m\" as m`)";
        if self.at(&Tok::Star) {
            return self.err(NO_WILDCARD);
        }
        if self.eat(&Tok::LBrace) {
            let mut names = Vec::new();
            self.skip_nl();
            loop {
                if self.at(&Tok::Star) {
                    return self.err(NO_WILDCARD);
                }
                let name = match self.bump() {
                    Tok::Ident(n) => n,
                    t => return Err(format!("line {}: expected import name, found {:?}", self.line(), t)),
                };
                let alias = if self.eat(&Tok::As) {
                    match self.bump() {
                        Tok::Ident(n) => Some(n),
                        t => return Err(format!("line {}: expected alias after `as`, found {:?}", self.line(), t)),
                    }
                } else {
                    None
                };
                names.push((name, alias));
                self.skip_nl();
                if !self.eat(&Tok::Comma) {
                    break;
                }
                self.skip_nl();
            }
            self.skip_nl();
            self.expect(&Tok::RBrace)?;
            // contextual keyword `from`
            match self.bump() {
                Tok::Ident(k) if k == "from" => {}
                t => return Err(format!("line {}: expected `from` after import list, found {:?}", self.line(), t)),
            }
            let path = self.parse_import_path()?;
            if names.is_empty() {
                return self.err("import list must name at least one export");
            }
            Ok(Item::Import(ImportDef { path, kind: ImportKind::Named(names), line }))
        } else if matches!(self.peek(), Tok::Str(_)) {
            let path = self.parse_import_path()?;
            let alias = if self.eat(&Tok::As) {
                match self.bump() {
                    Tok::Ident(n) => Some(n),
                    t => return Err(format!("line {}: expected alias after `as`, found {:?}", self.line(), t)),
                }
            } else {
                None
            };
            Ok(Item::Import(ImportDef { path, kind: ImportKind::Qualified(alias), line }))
        } else {
            self.err("expected `{ names }` or a \"path\" string after `import`")
        }
    }

    fn parse_import_path(&mut self) -> Result<String, String> {
        let line = self.line();
        match self.bump() {
            Tok::Str(parts) => match parts.as_slice() {
                [StrPart::Lit(s)] => Ok(s.clone()),
                _ => Err(format!("line {}: import path must be a plain string (no interpolation)", line)),
            },
            t => Err(format!("line {}: expected a \"path\" string, found {:?}", line, t)),
        }
    }

    fn parse_fn_def(&mut self, exported: bool, attrs: Vec<String>) -> Result<FnDef, String> {
        let line = self.line();
        self.expect(&Tok::Fn)?;
        let name = match self.bump() {
            Tok::Ident(n) => n,
            t => return Err(format!("line {}: expected function name, found {:?}", line, t)),
        };
        self.expect(&Tok::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Tok::RParen)?;
        let body = if self.eat(&Tok::Assign) {
            self.parse_expr()?
        } else if self.at(&Tok::LBrace) {
            self.parse_block_forced()?
        } else {
            return self.err("expected `=` or `{` after function signature");
        };
        Ok(FnDef { name, params, body, exported, attrs, line })
    }

    fn parse_params(&mut self) -> Result<Vec<Pattern>, String> {
        let mut params = Vec::new();
        self.skip_nl();
        if self.at(&Tok::RParen) {
            return Ok(params);
        }
        loop {
            self.skip_nl();
            params.push(self.parse_pattern()?);
            self.skip_nl();
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.skip_nl();
        Ok(params)
    }

    fn parse_type_def(&mut self) -> Result<TypeDef, String> {
        let line = self.line();
        self.expect(&Tok::Type)?;
        let name = match self.bump() {
            Tok::TypeName(n) => n,
            t => return Err(format!("line {}: expected type name, found {:?}", line, t)),
        };
        // optional generic params: bare idents before `=`
        while matches!(self.peek(), Tok::Ident(_)) {
            self.bump();
        }
        self.expect(&Tok::Assign)?;
        self.skip_nl();
        self.eat(&Tok::VBar); // optional leading |
        let mut variants = Vec::new();
        loop {
            self.skip_nl();
            let tag = match self.bump() {
                Tok::TypeName(n) => n,
                t => return Err(format!("line {}: expected variant name, found {:?}", self.line(), t)),
            };
            let fields = if self.at(&Tok::LBrace) {
                self.bump();
                let mut fs = Vec::new();
                self.skip_nl();
                while !self.at(&Tok::RBrace) {
                    let fname = match self.bump() {
                        Tok::Ident(n) => n,
                        t => return Err(format!("line {}: expected field name, found {:?}", self.line(), t)),
                    };
                    if self.eat(&Tok::Colon) {
                        self.skip_type_expr()?;
                    }
                    fs.push(fname);
                    self.skip_nl();
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                    self.skip_nl();
                }
                self.skip_nl();
                self.expect(&Tok::RBrace)?;
                Some(fs)
            } else {
                None
            };
            variants.push(VariantDef { tag, fields });
            // newlines before `|` are already suppressed by the lexer, so a
            // newline here terminates the type definition
            if !self.eat(&Tok::VBar) {
                break;
            }
        }
        Ok(TypeDef { name, variants, line })
    }

    /// Type annotations are parsed and discarded (gradual typing, M7 deferred).
    fn skip_type_expr(&mut self) -> Result<(), String> {
        match self.bump() {
            Tok::TypeName(_) | Tok::Ident(_) => {}
            t => return Err(format!("line {}: expected type, found {:?}", self.line(), t)),
        }
        // optional simple type application: `List Int`, `Option a`, `[Int]`, `(..)`
        loop {
            match self.peek() {
                Tok::TypeName(_) | Tok::Ident(_) => {
                    self.bump();
                }
                Tok::LBracket => {
                    self.bump();
                    self.skip_type_expr()?;
                    self.expect(&Tok::RBracket)?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    // ---------- statements & blocks ----------

    /// Parse `{ stmt* expr? }` as a block (never a record).
    fn parse_block_forced(&mut self) -> Result<Expr, String> {
        let saved = self.save_flags();
        let r = self.parse_block_forced_inner();
        self.restore_flags(saved);
        r
    }

    fn parse_block_forced_inner(&mut self) -> Result<Expr, String> {
        let line = self.line();
        self.expect(&Tok::LBrace)?;
        let mut stmts: Vec<Stmt> = Vec::new();
        let mut tail: Option<Box<Expr>> = None;
        loop {
            self.skip_nl();
            if self.at(&Tok::RBrace) {
                break;
            }
            let stmt = self.parse_stmt()?;
            self.skip_nl();
            if self.at(&Tok::RBrace) {
                if let Stmt::Expr(e) = stmt {
                    tail = Some(Box::new(e));
                } else {
                    stmts.push(stmt);
                }
                break;
            }
            stmts.push(stmt);
        }
        self.expect(&Tok::RBrace)?;
        Ok(Expr { kind: ExprKind::Block(stmts, tail), line })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        let line = self.line();
        match self.peek() {
            Tok::Import => self.err("`import` is only allowed at the top level of a file"),
            Tok::Let => {
                self.bump();
                let mutable = self.eat(&Tok::Mut);
                let pattern = self.parse_pattern()?;
                if mutable && !matches!(pattern, Pattern::Bind(_)) {
                    return self.err("`let mut` requires a plain name, not a pattern");
                }
                self.expect(&Tok::Assign)?;
                let expr = self.parse_expr()?;
                Ok(Stmt::Let { mutable, pattern, expr, line })
            }
            Tok::While => {
                self.bump();
                self.no_struct += 1;
                let cond = self.parse_expr()?;
                self.no_struct -= 1;
                let body = self.parse_block_forced()?;
                Ok(Stmt::While { cond, body, line })
            }
            Tok::For => {
                self.bump();
                let pattern = self.parse_pattern()?;
                self.expect(&Tok::In)?;
                self.no_struct += 1;
                let iter = self.parse_expr()?;
                self.no_struct -= 1;
                let body = self.parse_block_forced()?;
                Ok(Stmt::For { pattern, iter, body, line })
            }
            Tok::Return => {
                self.bump();
                let expr = if self.at(&Tok::Newline) || self.at(&Tok::RBrace) || self.at(&Tok::Eof) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                Ok(Stmt::Return { expr, line })
            }
            Tok::Break => {
                self.bump();
                Ok(Stmt::Break { line })
            }
            Tok::Continue => {
                self.bump();
                Ok(Stmt::Continue { line })
            }
            _ => {
                let e = self.parse_expr()?;
                let op = match self.peek() {
                    Tok::Assign => Some(None),
                    Tok::PlusEq => Some(Some(BinOp::Add)),
                    Tok::MinusEq => Some(Some(BinOp::Sub)),
                    Tok::StarEq => Some(Some(BinOp::Mul)),
                    Tok::SlashEq => Some(Some(BinOp::Div)),
                    Tok::PercentEq => Some(Some(BinOp::Mod)),
                    _ => None,
                };
                if let Some(compound) = op {
                    let name = match &e.kind {
                        ExprKind::Ident(n) => n.clone(),
                        _ => return self.err("assignment target must be a `let mut` variable name"),
                    };
                    self.bump();
                    let rhs = self.parse_expr()?;
                    let expr = match compound {
                        None => rhs,
                        Some(op) => Expr {
                            kind: ExprKind::Binary {
                                op,
                                lhs: Box::new(e),
                                rhs: Box::new(rhs),
                            },
                            line,
                        },
                    };
                    Ok(Stmt::Assign { name, expr, line })
                } else {
                    Ok(Stmt::Expr(e))
                }
            }
        }
    }

    // ---------- expressions ----------

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_pipe()
    }

    fn parse_pipe(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_or()?;
        while self.at(&Tok::Pipe) {
            let line = self.line();
            self.bump();
            self.skip_nl();
            let rhs = self.parse_unary()?; // postfix-level: f, f(a), m.f(a), match {..}, lambda
            lhs = desugar_pipe(lhs, rhs, line)?;
        }
        Ok(lhs)
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_and()?;
        while self.at(&Tok::Or) {
            let line = self.line();
            self.bump();
            let rhs = self.parse_and()?;
            lhs = Expr { kind: ExprKind::Or(Box::new(lhs), Box::new(rhs)), line };
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_cmp()?;
        while self.at(&Tok::And) {
            let line = self.line();
            self.bump();
            let rhs = self.parse_cmp()?;
            lhs = Expr { kind: ExprKind::And(Box::new(lhs), Box::new(rhs)), line };
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_range()?;
        let op = match self.peek() {
            Tok::EqEq => Some(BinOp::Eq),
            Tok::NotEq => Some(BinOp::Ne),
            Tok::Lt => Some(BinOp::Lt),
            Tok::Le => Some(BinOp::Le),
            Tok::Gt => Some(BinOp::Gt),
            Tok::Ge => Some(BinOp::Ge),
            _ => None,
        };
        if let Some(op) = op {
            let line = self.line();
            self.bump();
            let rhs = self.parse_range()?;
            // non-associative: no chaining
            Ok(Expr { kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, line })
        } else {
            Ok(lhs)
        }
    }

    fn parse_range(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_add()?;
        let inclusive = match self.peek() {
            Tok::DotDot => false,
            Tok::DotDotEq => true,
            _ => return Ok(lhs),
        };
        let line = self.line();
        self.bump();
        let rhs = self.parse_add()?;
        Ok(Expr {
            kind: ExprKind::Range { lo: Box::new(lhs), hi: Box::new(rhs), inclusive },
            line,
        })
    }

    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            let line = self.line();
            self.bump();
            let rhs = self.parse_mul()?;
            lhs = Expr { kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, line };
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_pow()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Mod,
                _ => break,
            };
            let line = self.line();
            self.bump();
            let rhs = self.parse_pow()?;
            lhs = Expr { kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, line };
        }
        Ok(lhs)
    }

    fn parse_pow(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_unary()?;
        if self.at(&Tok::StarStar) {
            let line = self.line();
            self.bump();
            let rhs = self.parse_pow()?; // right-assoc
            return Ok(Expr {
                kind: ExprKind::Binary { op: BinOp::Pow, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                line,
            });
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        let line = self.line();
        match self.peek() {
            Tok::Not => {
                self.bump();
                let e = self.parse_unary()?;
                Ok(Expr { kind: ExprKind::Unary { op: UnOp::Not, operand: Box::new(e) }, line })
            }
            Tok::Minus => {
                self.bump();
                let e = self.parse_unary()?;
                Ok(Expr { kind: ExprKind::Unary { op: UnOp::Neg, operand: Box::new(e) }, line })
            }
            Tok::At => {
                self.bump();
                let e = self.parse_unary()?;
                Ok(Expr { kind: ExprKind::Deref(Box::new(e)), line })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_primary()?;
        loop {
            let line = self.line();
            match self.peek() {
                Tok::LParen => {
                    self.bump();
                    let args = self.parse_args()?;
                    self.expect(&Tok::RParen)?;
                    e = Expr { kind: ExprKind::Call { callee: Box::new(e), args }, line };
                }
                Tok::Dot => {
                    self.bump();
                    let name = match self.bump() {
                        Tok::Ident(n) => n,
                        t => return Err(format!("line {}: expected field/method name after '.', found {:?}", line, t)),
                    };
                    if self.at(&Tok::LParen) {
                        self.bump();
                        let args = self.parse_args()?;
                        self.expect(&Tok::RParen)?;
                        e = Expr { kind: ExprKind::MethodCall { recv: Box::new(e), name, args }, line };
                    } else {
                        e = Expr { kind: ExprKind::Field { recv: Box::new(e), name }, line };
                    }
                }
                Tok::LBracket => {
                    self.bump();
                    let saved = self.save_flags();
                    self.skip_nl();
                    let idx = self.parse_expr()?;
                    self.skip_nl();
                    self.restore_flags(saved);
                    self.expect(&Tok::RBracket)?;
                    e = Expr { kind: ExprKind::Index { recv: Box::new(e), index: Box::new(idx) }, line };
                }
                Tok::Question => {
                    self.bump();
                    e = Expr { kind: ExprKind::Try(Box::new(e)), line };
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, String> {
        let saved = self.save_flags();
        let r = self.parse_args_inner();
        self.restore_flags(saved);
        r
    }

    fn parse_args_inner(&mut self) -> Result<Vec<Expr>, String> {
        let mut args = Vec::new();
        self.skip_nl();
        if self.at(&Tok::RParen) {
            return Ok(args);
        }
        loop {
            self.skip_nl();
            args.push(self.parse_expr()?);
            self.skip_nl();
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.skip_nl();
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        let line = self.line();
        match self.peek().clone() {
            Tok::Int(i) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Int(i), line })
            }
            Tok::Float(f) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Float(f), line })
            }
            Tok::True => {
                self.bump();
                Ok(Expr { kind: ExprKind::Bool(true), line })
            }
            Tok::False => {
                self.bump();
                Ok(Expr { kind: ExprKind::Bool(false), line })
            }
            Tok::Str(parts) => {
                self.bump();
                self.parse_string(parts, line)
            }
            Tok::Underscore => {
                self.bump();
                Ok(Expr { kind: ExprKind::Ident("_".into()), line })
            }
            Tok::Ident(name) => {
                if self.peek_at(1) == &Tok::FatArrow && self.no_bare_lambda == 0 {
                    self.bump();
                    self.bump();
                    let body = self.parse_expr()?;
                    Ok(Expr {
                        kind: ExprKind::Lambda { params: vec![Pattern::Bind(name)], body: Box::new(body) },
                        line,
                    })
                } else {
                    self.bump();
                    Ok(Expr { kind: ExprKind::Ident(name), line })
                }
            }
            Tok::TypeName(tag) => {
                self.bump();
                if self.at(&Tok::LParen) {
                    self.bump();
                    let args = self.parse_args()?;
                    self.expect(&Tok::RParen)?;
                    Ok(Expr { kind: ExprKind::Variant { tag, payload: VariantCtor::Positional(args) }, line })
                } else if self.at(&Tok::LBrace) && self.no_struct == 0 {
                    self.bump();
                    let mut fields = Vec::new();
                    self.skip_nl();
                    while !self.at(&Tok::RBrace) {
                        let fname = match self.bump() {
                            Tok::Ident(n) => n,
                            t => return Err(format!("line {}: expected field name, found {:?}", self.line(), t)),
                        };
                        let val = if self.eat(&Tok::Colon) {
                            self.skip_nl();
                            self.parse_expr()?
                        } else {
                            Expr { kind: ExprKind::Ident(fname.clone()), line }
                        };
                        fields.push((fname, val));
                        self.skip_nl();
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                        self.skip_nl();
                    }
                    self.skip_nl();
                    self.expect(&Tok::RBrace)?;
                    Ok(Expr { kind: ExprKind::Variant { tag, payload: VariantCtor::Named(fields) }, line })
                } else {
                    Ok(Expr { kind: ExprKind::Variant { tag, payload: VariantCtor::Unit }, line })
                }
            }
            Tok::LParen => self.parse_paren(),
            Tok::LBracket => {
                self.bump();
                let saved = self.save_flags();
                let mut items = Vec::new();
                self.skip_nl();
                while !self.at(&Tok::RBracket) {
                    self.skip_nl();
                    items.push(self.parse_expr()?);
                    self.skip_nl();
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.skip_nl();
                self.restore_flags(saved);
                self.expect(&Tok::RBracket)?;
                Ok(Expr { kind: ExprKind::List(items), line })
            }
            Tok::LBrace => self.parse_brace(),
            Tok::If => self.parse_if(),
            Tok::Match => self.parse_match(),
            t => Err(format!("line {}: unexpected token {:?}", line, t)),
        }
    }

    fn parse_string(&mut self, parts: Vec<StrPart>, line: u32) -> Result<Expr, String> {
        if parts.len() == 1 {
            if let StrPart::Lit(s) = &parts[0] {
                return Ok(Expr { kind: ExprKind::Str(s.clone()), line });
            }
        }
        let mut out = Vec::new();
        for p in parts {
            match p {
                StrPart::Lit(s) => out.push(InterpPart::Lit(s)),
                StrPart::Interp(mut toks) => {
                    toks.push(Token { tok: Tok::Eof, line });
                    let mut sub = Parser { toks, pos: 0, no_struct: 0, no_bare_lambda: 0 };
                    let e = sub.parse_expr()?;
                    if !sub.at(&Tok::Eof) {
                        return Err(format!("line {}: trailing tokens in interpolation", line));
                    }
                    out.push(InterpPart::Expr(e));
                }
            }
        }
        Ok(Expr { kind: ExprKind::Interp(out), line })
    }

    /// `(` already peeked: unit, grouped expr, tuple, or lambda params.
    fn parse_paren(&mut self) -> Result<Expr, String> {
        let line = self.line();
        // lookahead: find matching ')' and check for '=>'
        if self.is_lambda_params() {
            self.expect(&Tok::LParen)?;
            let params = self.parse_params()?;
            self.expect(&Tok::RParen)?;
            self.expect(&Tok::FatArrow)?;
            let body = self.parse_expr()?;
            return Ok(Expr { kind: ExprKind::Lambda { params, body: Box::new(body) }, line });
        }
        self.expect(&Tok::LParen)?;
        self.skip_nl();
        if self.eat(&Tok::RParen) {
            return Ok(Expr { kind: ExprKind::Unit, line });
        }
        let saved = self.save_flags(); // parens reset subject-position restrictions
        let first = self.parse_expr()?;
        self.skip_nl();
        let result = if self.at(&Tok::Comma) {
            let mut items = vec![first];
            while self.eat(&Tok::Comma) {
                self.skip_nl();
                if self.at(&Tok::RParen) {
                    break;
                }
                items.push(self.parse_expr()?);
                self.skip_nl();
            }
            Expr { kind: ExprKind::Tuple(items), line }
        } else {
            first
        };
        self.restore_flags(saved);
        self.expect(&Tok::RParen)?;
        Ok(result)
    }

    /// Scan ahead from a `(` to decide whether this is lambda params.
    fn is_lambda_params(&self) -> bool {
        debug_assert!(self.at(&Tok::LParen));
        let mut depth = 0usize;
        let mut i = self.pos;
        while let Some(t) = self.toks.get(i) {
            match t.tok {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return self.toks.get(i + 1).map(|t| t.tok == Tok::FatArrow).unwrap_or(false);
                    }
                }
                Tok::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// `{` already peeked: record literal or block.
    fn parse_brace(&mut self) -> Result<Expr, String> {
        let line = self.line();
        // decide record vs block by lookahead (skipping newlines)
        let mut i = self.pos + 1;
        while self.toks.get(i).map(|t| t.tok == Tok::Newline).unwrap_or(false) {
            i += 1;
        }
        let is_record = match self.toks.get(i).map(|t| &t.tok) {
            Some(Tok::RBrace) | Some(Tok::DotDot) => true,
            Some(Tok::Ident(_)) => {
                let mut j = i + 1;
                while self.toks.get(j).map(|t| t.tok == Tok::Newline).unwrap_or(false) {
                    j += 1;
                }
                matches!(
                    self.toks.get(j).map(|t| &t.tok),
                    Some(Tok::Colon) | Some(Tok::Comma) | Some(Tok::RBrace)
                )
            }
            _ => false,
        };
        if !is_record {
            return self.parse_block_forced();
        }
        self.expect(&Tok::LBrace)?;
        self.skip_nl();
        let spread = if self.eat(&Tok::DotDot) {
            let base = self.parse_expr()?;
            self.skip_nl();
            self.eat(&Tok::Comma);
            self.skip_nl();
            Some(Box::new(base))
        } else {
            None
        };
        let mut fields = Vec::new();
        while !self.at(&Tok::RBrace) {
            let fname = match self.bump() {
                Tok::Ident(n) => n,
                t => return Err(format!("line {}: expected field name, found {:?}", self.line(), t)),
            };
            let val = if self.eat(&Tok::Colon) {
                self.skip_nl();
                self.parse_expr()?
            } else {
                Expr { kind: ExprKind::Ident(fname.clone()), line }
            };
            fields.push((fname, val));
            self.skip_nl();
            if !self.eat(&Tok::Comma) {
                break;
            }
            self.skip_nl();
        }
        self.skip_nl();
        self.expect(&Tok::RBrace)?;
        Ok(Expr { kind: ExprKind::Record { spread, fields }, line })
    }

    fn parse_if(&mut self) -> Result<Expr, String> {
        let line = self.line();
        self.expect(&Tok::If)?;
        self.no_struct += 1;
        let cond = self.parse_expr()?;
        self.no_struct -= 1;
        let then = self.parse_block_forced()?;
        let els = if self.eat(&Tok::Else) {
            if self.at(&Tok::If) {
                Some(Box::new(self.parse_if()?))
            } else {
                Some(Box::new(self.parse_block_forced()?))
            }
        } else {
            None
        };
        Ok(Expr {
            kind: ExprKind::If { cond: Box::new(cond), then: Box::new(then), els },
            line,
        })
    }

    fn parse_match(&mut self) -> Result<Expr, String> {
        let line = self.line();
        self.expect(&Tok::Match)?;
        let subject = if self.at(&Tok::LBrace) {
            None
        } else {
            self.no_struct += 1;
            let s = self.parse_expr()?;
            self.no_struct -= 1;
            Some(s)
        };
        self.expect(&Tok::LBrace)?;
        let saved = self.save_flags();
        let mut arms = Vec::new();
        loop {
            self.skip_nl();
            if self.at(&Tok::RBrace) {
                break;
            }
            let arm_line = self.line();
            let pattern = self.parse_pattern()?;
            let guard = if self.eat(&Tok::If) {
                self.no_bare_lambda += 1;
                let g = self.parse_expr()?;
                self.no_bare_lambda -= 1;
                Some(g)
            } else {
                None
            };
            self.expect(&Tok::FatArrow)?;
            self.skip_nl();
            // statement-only forms are handy as arm bodies: `3 => break`
            let body = match self.peek() {
                Tok::Break | Tok::Continue | Tok::Return => {
                    let stmt = self.parse_stmt()?;
                    Expr { kind: ExprKind::Block(vec![stmt], None), line: arm_line }
                }
                _ => self.parse_expr()?,
            };
            arms.push(Arm { pattern, guard, body, line: arm_line });
            // a comma, a newline, or the closing brace ends an arm
            if !self.eat(&Tok::Comma) && !self.at(&Tok::RBrace) && !self.at(&Tok::Newline) {
                return self.err(&format!("expected ',' or '}}' after match arm, found {:?}", self.peek()));
            }
        }
        self.restore_flags(saved);
        self.expect(&Tok::RBrace)?;
        if arms.is_empty() {
            return self.err("match must have at least one arm");
        }
        match subject {
            Some(s) => Ok(Expr {
                kind: ExprKind::Match { subject: Box::new(s), arms },
                line,
            }),
            None => {
                // subjectless match = matching function (spec §4.2)
                let subj = Expr { kind: ExprKind::Ident("__subject".into()), line };
                let m = Expr { kind: ExprKind::Match { subject: Box::new(subj), arms }, line };
                Ok(Expr {
                    kind: ExprKind::Lambda {
                        params: vec![Pattern::Bind("__subject".into())],
                        body: Box::new(m),
                    },
                    line,
                })
            }
        }
    }

    // ---------- patterns ----------

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        let first = self.parse_pattern_alt()?;
        if !self.at(&Tok::VBar) {
            return Ok(first);
        }
        let mut alts = vec![first];
        while self.eat(&Tok::VBar) {
            self.skip_nl();
            alts.push(self.parse_pattern_alt()?);
        }
        // or-pattern arms must bind the same names
        let mut names0 = Vec::new();
        alts[0].bound_names(&mut names0);
        names0.sort();
        for alt in &alts[1..] {
            let mut names = Vec::new();
            alt.bound_names(&mut names);
            names.sort();
            if names != names0 {
                return self.err("all sides of an or-pattern must bind the same names");
            }
        }
        Ok(Pattern::Or(alts))
    }

    fn parse_pattern_alt(&mut self) -> Result<Pattern, String> {
        let p = self.parse_pattern_primary()?;
        if self.eat(&Tok::As) {
            let name = match self.bump() {
                Tok::Ident(n) => n,
                t => return Err(format!("line {}: expected name after 'as', found {:?}", self.line(), t)),
            };
            return Ok(Pattern::As(Box::new(p), name));
        }
        Ok(p)
    }

    fn parse_pattern_primary(&mut self) -> Result<Pattern, String> {
        let line = self.line();
        match self.peek().clone() {
            Tok::Underscore => {
                self.bump();
                Ok(Pattern::Wildcard)
            }
            Tok::Int(_) | Tok::Minus => {
                let neg = self.eat(&Tok::Minus);
                let lo = match self.bump() {
                    Tok::Int(i) => {
                        if neg {
                            -i
                        } else {
                            i
                        }
                    }
                    Tok::Float(f) => {
                        return Ok(Pattern::LitFloat(if neg { -f } else { f }));
                    }
                    t => return Err(format!("line {}: expected number after '-', found {:?}", line, t)),
                };
                match self.peek() {
                    Tok::DotDot | Tok::DotDotEq => {
                        let inclusive = self.peek() == &Tok::DotDotEq;
                        self.bump();
                        let neg2 = self.eat(&Tok::Minus);
                        let hi = match self.bump() {
                            Tok::Int(i) => {
                                if neg2 {
                                    -i
                                } else {
                                    i
                                }
                            }
                            t => return Err(format!("line {}: expected int in range pattern, found {:?}", line, t)),
                        };
                        Ok(Pattern::Range { lo, hi, inclusive })
                    }
                    _ => Ok(Pattern::LitInt(lo)),
                }
            }
            Tok::Float(f) => {
                self.bump();
                Ok(Pattern::LitFloat(f))
            }
            Tok::True => {
                self.bump();
                Ok(Pattern::LitBool(true))
            }
            Tok::False => {
                self.bump();
                Ok(Pattern::LitBool(false))
            }
            Tok::Str(parts) => {
                self.bump();
                match parts.as_slice() {
                    [StrPart::Lit(s)] => Ok(Pattern::LitStr(s.clone())),
                    _ => self.err("string interpolation is not allowed in patterns"),
                }
            }
            Tok::Ident(n) => {
                self.bump();
                Ok(Pattern::Bind(n))
            }
            Tok::TypeName(tag) => {
                self.bump();
                if self.at(&Tok::LParen) {
                    self.bump();
                    let mut items = Vec::new();
                    self.skip_nl();
                    while !self.at(&Tok::RParen) {
                        items.push(self.parse_pattern()?);
                        self.skip_nl();
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                        self.skip_nl();
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(Pattern::VariantPos { tag, items })
                } else if self.at(&Tok::LBrace) {
                    self.bump();
                    let (fields, rest) = self.parse_field_patterns()?;
                    Ok(Pattern::VariantNamed { tag, fields, rest })
                } else {
                    Ok(Pattern::VariantPos { tag, items: vec![] })
                }
            }
            Tok::LParen => {
                self.bump();
                self.skip_nl();
                if self.eat(&Tok::RParen) {
                    return Ok(Pattern::LitUnit);
                }
                let first = self.parse_pattern()?;
                self.skip_nl();
                if self.at(&Tok::Comma) {
                    let mut items = vec![first];
                    while self.eat(&Tok::Comma) {
                        self.skip_nl();
                        if self.at(&Tok::RParen) {
                            break;
                        }
                        items.push(self.parse_pattern()?);
                        self.skip_nl();
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(Pattern::Tuple(items))
                } else {
                    self.expect(&Tok::RParen)?;
                    Ok(first)
                }
            }
            Tok::LBracket => {
                self.bump();
                let mut items = Vec::new();
                let mut rest: Option<Option<String>> = None;
                self.skip_nl();
                while !self.at(&Tok::RBracket) {
                    if self.eat(&Tok::DotDot) {
                        let name = match self.peek() {
                            Tok::Ident(n) => {
                                let n = n.clone();
                                self.bump();
                                Some(n)
                            }
                            _ => None,
                        };
                        rest = Some(name);
                        self.skip_nl();
                        break;
                    }
                    items.push(self.parse_pattern()?);
                    self.skip_nl();
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                    self.skip_nl();
                }
                self.skip_nl();
                self.expect(&Tok::RBracket)?;
                Ok(Pattern::List { items, rest })
            }
            Tok::LBrace => {
                self.bump();
                let (fields, rest) = self.parse_field_patterns()?;
                Ok(Pattern::Record { fields, rest })
            }
            t => Err(format!("line {}: unexpected token in pattern: {:?}", line, t)),
        }
    }

    /// Parse `field: pat, field, .. }` (after the opening brace).
    fn parse_field_patterns(&mut self) -> Result<(Vec<(String, Pattern)>, bool), String> {
        let mut fields = Vec::new();
        let mut rest = false;
        self.skip_nl();
        while !self.at(&Tok::RBrace) {
            if self.eat(&Tok::DotDot) {
                rest = true;
                self.skip_nl();
                break;
            }
            let fname = match self.bump() {
                Tok::Ident(n) => n,
                t => return Err(format!("line {}: expected field name in pattern, found {:?}", self.line(), t)),
            };
            let pat = if self.eat(&Tok::Colon) {
                self.parse_pattern()?
            } else {
                Pattern::Bind(fname.clone())
            };
            fields.push((fname, pat));
            self.skip_nl();
            if !self.eat(&Tok::Comma) {
                break;
            }
            self.skip_nl();
        }
        self.skip_nl();
        self.expect(&Tok::RBrace)?;
        Ok((fields, rest))
    }
}

/// Desugar `lhs |> rhs` per spec §4.1.
fn desugar_pipe(lhs: Expr, rhs: Expr, line: u32) -> Result<Expr, String> {
    fn is_hole(e: &Expr) -> bool {
        matches!(&e.kind, ExprKind::Ident(n) if n == "_")
    }
    match rhs.kind {
        ExprKind::Call { callee, mut args } => {
            if let Some(slot) = args.iter().position(is_hole) {
                args[slot] = lhs;
            } else {
                args.insert(0, lhs);
            }
            Ok(Expr { kind: ExprKind::Call { callee, args }, line })
        }
        ExprKind::MethodCall { recv, name, mut args } => {
            if let Some(slot) = args.iter().position(is_hole) {
                args[slot] = lhs;
            } else {
                args.insert(0, lhs);
            }
            Ok(Expr { kind: ExprKind::MethodCall { recv, name, args }, line })
        }
        ExprKind::Variant { tag, payload: VariantCtor::Unit } => {
            // `x |> Some` builds Some(x)
            Ok(Expr {
                kind: ExprKind::Variant { tag, payload: VariantCtor::Positional(vec![lhs]) },
                line,
            })
        }
        _ => Ok(Expr {
            kind: ExprKind::Call { callee: Box::new(rhs), args: vec![lhs] },
            line,
        }),
    }
}
