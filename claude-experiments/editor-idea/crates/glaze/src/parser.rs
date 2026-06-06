//! Recursive-descent + precedence-climbing parser for Glaze.

use crate::GlazeError;
use crate::ast::*;
use crate::lexer::{Tok, lex};

pub fn parse(src: &str) -> Result<Program, GlazeError> {
    let toks = lex(src)?;
    let prog = Parser { toks, pos: 0 }.program()?;
    validate_fns(&prog)?;
    Ok(prog)
}

/// Reject recursive user functions (they'd inline forever).
fn validate_fns(p: &Program) -> PResult<()> {
    use std::collections::HashSet;
    fn calls(e: &Expr, out: &mut Vec<String>) {
        match e {
            Expr::Call(n, a) => {
                out.push(n.clone());
                a.iter().for_each(|x| calls(x, out));
            }
            Expr::Unary(_, x) => calls(x, out),
            Expr::Bin(_, l, r) => {
                calls(l, out);
                calls(r, out);
            }
            Expr::Tern(c, a, b) => {
                calls(c, out);
                calls(a, out);
                calls(b, out);
            }
            _ => {}
        }
    }
    let fnames: HashSet<&str> = p.fns.iter().map(|f| f.name.as_str()).collect();
    fn dfs(name: &str, p: &Program, fnames: &HashSet<&str>, path: &mut Vec<String>) -> PResult<()> {
        if path.iter().any(|n| n == name) {
            return err(format!("recursive function `{}` is not allowed", name));
        }
        let Some(f) = p.fns.iter().find(|f| f.name == name) else {
            return Ok(());
        };
        path.push(name.to_string());
        let mut cs = Vec::new();
        calls(&f.body, &mut cs);
        for c in cs {
            if fnames.contains(c.as_str()) {
                dfs(&c, p, fnames, path)?;
            }
        }
        path.pop();
        Ok(())
    }
    for f in &p.fns {
        dfs(&f.name, p, &fnames, &mut Vec::new())?;
    }
    Ok(())
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

type PResult<T> = Result<T, GlazeError>;

fn err<T>(msg: impl Into<String>) -> PResult<T> {
    Err(GlazeError::Parse(msg.into()))
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos]
    }
    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].clone();
        self.pos += 1;
        t
    }
    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Tok::Newline) {
            self.pos += 1;
        }
    }
    fn expect(&mut self, t: &Tok) -> PResult<()> {
        if self.peek() == t {
            self.pos += 1;
            Ok(())
        } else {
            err(format!("expected {:?}, found {:?}", t, self.peek()))
        }
    }
    fn ident(&mut self) -> PResult<String> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            other => err(format!("expected identifier, found {:?}", other)),
        }
    }

    fn program(&mut self) -> PResult<Program> {
        let mut tokens = Vec::new();
        let mut fns = Vec::new();
        let mut styles = Vec::new();
        loop {
            self.skip_newlines();
            match self.peek() {
                Tok::Eof => break,
                // `token x = …` and top-level `let x = …` are both global values.
                Tok::Ident(k) if k == "token" || k == "let" => {
                    self.bump();
                    let name = self.ident()?;
                    self.expect(&Tok::Assign)?;
                    let value = self.expr(0)?;
                    tokens.push(TokenDef { name, value });
                }
                Tok::Ident(k) if k == "fn" => {
                    self.bump();
                    fns.push(self.fn_def()?);
                }
                Tok::Ident(k) if k == "style" => {
                    self.bump();
                    styles.push(self.style_def()?);
                }
                other => {
                    return err(format!("expected `token`, `let`, `fn`, or `style`, found {:?}", other));
                }
            }
        }
        Ok(Program { tokens, fns, styles })
    }

    fn fn_def(&mut self) -> PResult<FnDef> {
        let name = self.ident()?;
        self.expect(&Tok::LParen)?;
        let mut params = Vec::new();
        while !matches!(self.peek(), Tok::RParen) {
            params.push(self.ident()?);
            if matches!(self.peek(), Tok::Comma) {
                self.bump();
            }
        }
        self.expect(&Tok::RParen)?;
        self.expect(&Tok::Assign)?;
        let body = self.expr(0)?;
        Ok(FnDef { name, params, body })
    }

    fn style_def(&mut self) -> PResult<StyleDef> {
        let name = self.ident()?;
        let mut params = Vec::new();
        if matches!(self.peek(), Tok::LParen) {
            self.bump();
            while !matches!(self.peek(), Tok::RParen) {
                params.push(self.ident()?);
                if matches!(self.peek(), Tok::Comma) {
                    self.bump();
                }
            }
            self.expect(&Tok::RParen)?;
        }
        self.expect(&Tok::LBrace)?;
        let body = self.block_items()?;
        Ok(StyleDef { name, params, body })
    }

    /// Parse items until the closing `}` (which is consumed).
    fn block_items(&mut self) -> PResult<Vec<Item>> {
        let mut items = Vec::new();
        loop {
            self.skip_newlines();
            match self.peek().clone() {
                Tok::RBrace => {
                    self.bump();
                    break;
                }
                Tok::Eof => return err("unclosed `{`"),
                Tok::Colon => {
                    self.bump();
                    let state = self.ident()?;
                    self.expect(&Tok::LBrace)?;
                    let body = self.block_items()?;
                    items.push(Item::State { state, body });
                }
                Tok::Ident(name) if name == "let" => {
                    self.bump();
                    let var = self.ident()?;
                    self.expect(&Tok::Assign)?;
                    let value = self.expr(0)?;
                    items.push(Item::Let { name: var, value });
                }
                Tok::Ident(name) if name == "when" => {
                    self.bump();
                    let cond = self.expr(0)?;
                    self.expect(&Tok::LBrace)?;
                    let body = self.block_items()?;
                    items.push(Item::When { cond, body });
                }
                Tok::Ident(name) if name == "shader" => {
                    self.bump();
                    let body = self.shader_block()?;
                    items.push(Item::Shader { overlay: false, body });
                }
                Tok::Ident(name) if name == "overlay" => {
                    self.bump();
                    // `overlay shader { … }`
                    match self.peek() {
                        Tok::Ident(s) if s == "shader" => {
                            self.bump();
                            let body = self.shader_block()?;
                            items.push(Item::Shader { overlay: true, body });
                        }
                        other => return err(format!("expected `shader` after `overlay`, found {:?}", other)),
                    }
                }
                Tok::Ident(name) => {
                    self.bump();
                    let mut args = Vec::new();
                    while !matches!(self.peek(), Tok::Newline | Tok::RBrace | Tok::Eof) {
                        args.push(self.expr(0)?);
                    }
                    items.push(Item::Prop { name, args });
                }
                other => return err(format!("unexpected {:?} in style body", other)),
            }
        }
        Ok(items)
    }

    /// Parse a shader body: a sequence of `let NAME = expr` bindings and exactly
    /// one `emit expr`. Statements are newline-terminated.
    fn shader_block(&mut self) -> PResult<ShaderBody> {
        self.expect(&Tok::LBrace)?;
        let mut lets = Vec::new();
        let mut emit: Option<Expr> = None;
        loop {
            self.skip_newlines();
            match self.peek().clone() {
                Tok::RBrace => {
                    self.bump();
                    break;
                }
                Tok::Eof => return err("unclosed shader `{`"),
                Tok::Ident(kw) if kw == "let" => {
                    self.bump();
                    let name = self.ident()?;
                    self.expect(&Tok::Assign)?;
                    let value = self.expr(0)?;
                    lets.push((name, value));
                }
                Tok::Ident(kw) if kw == "emit" => {
                    self.bump();
                    if emit.is_some() {
                        return err("shader block has more than one `emit`");
                    }
                    emit = Some(self.expr(0)?);
                }
                other => return err(format!("expected `let` or `emit` in shader block, found {:?}", other)),
            }
        }
        let emit = emit.ok_or_else(|| GlazeError::Parse("shader block needs an `emit`".into()))?;
        Ok(ShaderBody { lets, emit })
    }

    // ---- expressions (precedence climbing) ----

    fn binding_power(op: &str) -> Option<u8> {
        match op {
            ">" | "<" | ">=" | "<=" | "==" => Some(10),
            "+" | "-" => Some(20),
            "*" | "/" => Some(30),
            _ => None,
        }
    }

    fn expr(&mut self, min_bp: u8) -> PResult<Expr> {
        let mut left = self.prefix()?;
        loop {
            match self.peek().clone() {
                Tok::Question if min_bp <= 1 => {
                    self.bump();
                    let a = self.expr(0)?;
                    self.expect(&Tok::Colon)?;
                    let b = self.expr(1)?;
                    left = Expr::Tern(Box::new(left), Box::new(a), Box::new(b));
                }
                Tok::Op(op) => match Self::binding_power(&op) {
                    Some(bp) if bp >= min_bp => {
                        self.bump();
                        let right = self.expr(bp + 1)?;
                        left = Expr::Bin(op, Box::new(left), Box::new(right));
                    }
                    _ => break,
                },
                _ => break,
            }
        }
        Ok(left)
    }

    fn prefix(&mut self) -> PResult<Expr> {
        match self.bump() {
            Tok::Num(v, u) => Ok(Expr::Num(v, u)),
            Tok::Hex(h) => Ok(Expr::Hex(h)),
            Tok::Op(ref o) if o == "-" => Ok(Expr::Unary('-', Box::new(self.expr(40)?))),
            Tok::LParen => {
                let e = self.expr(0)?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Tok::Ident(name) => {
                if matches!(self.peek(), Tok::LParen) {
                    self.bump();
                    if name == "oklch" || name == "oklab" {
                        let nums = self.color_args()?;
                        Ok(Expr::Color { space: name, nums })
                    } else {
                        let mut args = Vec::new();
                        while !matches!(self.peek(), Tok::RParen) {
                            args.push(self.expr(0)?);
                            if matches!(self.peek(), Tok::Comma) {
                                self.bump();
                            }
                        }
                        self.expect(&Tok::RParen)?;
                        Ok(Expr::Call(name, args))
                    }
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            other => err(format!("unexpected {:?} in expression", other)),
        }
    }

    /// `oklch(0.72 0.11 85 / 0.5)` — space-separated numbers, `/` before alpha (ignored).
    fn color_args(&mut self) -> PResult<Vec<f64>> {
        let mut nums = Vec::new();
        loop {
            match self.peek().clone() {
                Tok::RParen => {
                    self.bump();
                    break;
                }
                Tok::Num(v, _) => {
                    self.bump();
                    nums.push(v);
                }
                Tok::Op(ref o) if o == "/" => {
                    self.bump();
                }
                Tok::Comma => {
                    self.bump();
                }
                other => return err(format!("unexpected {:?} in color literal", other)),
            }
        }
        Ok(nums)
    }
}
