//! Recursive-descent parser: token stream → surface [`Module`].
//!
//! Grammar in `docs/language.md`. Pratt-style expression parsing with Rust
//! operator precedence. Errors are single-shot (no recovery in v0): the first
//! error aborts with a span.

use crate::ast::*;
use crate::lexer::{Kw, Span, TokKind, Token};

#[derive(Debug)]
pub struct ParseError {
    pub msg: String,
    pub span: Span,
}

pub fn parse_module(tokens: &[Token]) -> Result<Module, ParseError> {
    let mut p = Parser { toks: tokens.to_vec(), pos: 0, no_struct: 0 };
    let mut items = Vec::new();
    while !p.at(&TokKind::Eof) {
        items.push(p.item()?);
    }
    Ok(Module { items })
}

struct Parser {
    /// Owned so `expect_gt` can split a `>>` token in place into a `>` (it
    /// rewrites the current token to `Gt` and leaves the cursor for the next
    /// generic close to consume).
    toks: Vec<Token>,
    pos: usize,
    /// >0 while parsing a context where a trailing `{` struct literal is
    /// forbidden (if/while/match/for scrutinee), so `if x {` reads `x` as the
    /// condition, not `x { … }` as a struct literal.
    no_struct: u32,
}

type PResult<T> = Result<T, ParseError>;

impl Parser {
    // ---- token cursor ------------------------------------------------------
    fn peek(&self) -> &TokKind {
        &self.toks[self.pos].kind
    }
    fn peek_at(&self, n: usize) -> &TokKind {
        let i = (self.pos + n).min(self.toks.len() - 1);
        &self.toks[i].kind
    }
    fn span(&self) -> Span {
        self.toks[self.pos].span
    }
    fn prev_span(&self) -> Span {
        self.toks[self.pos.saturating_sub(1)].span
    }
    fn at(&self, k: &TokKind) -> bool {
        self.peek() == k
    }
    fn at_kw(&self, kw: Kw) -> bool {
        matches!(self.peek(), TokKind::Keyword(k) if *k == kw)
    }
    fn bump(&mut self) -> &Token {
        let t = &self.toks[self.pos];
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, k: &TokKind) -> bool {
        if self.at(k) { self.bump(); true } else { false }
    }
    fn eat_kw(&mut self, kw: Kw) -> bool {
        if self.at_kw(kw) { self.bump(); true } else { false }
    }
    fn expect(&mut self, k: &TokKind) -> PResult<Span> {
        if self.at(k) {
            let s = self.span();
            self.bump();
            Ok(s)
        } else {
            Err(self.error(format!("expected {:?}, found {:?}", k, self.peek())))
        }
    }
    fn expect_kw(&mut self, kw: Kw) -> PResult<Span> {
        if self.at_kw(kw) {
            let s = self.span();
            self.bump();
            Ok(s)
        } else {
            Err(self.error(format!("expected `{:?}`, found {:?}", kw, self.peek())))
        }
    }
    fn ident(&mut self) -> PResult<String> {
        match self.peek().clone() {
            TokKind::Ident(s) => { self.bump(); Ok(s) }
            // `self`/`Self` are sometimes used as idents in paths.
            other => Err(self.error(format!("expected identifier, found {:?}", other))),
        }
    }
    fn error(&self, msg: impl Into<String>) -> ParseError {
        ParseError { msg: msg.into(), span: self.span() }
    }

    // ---- items -------------------------------------------------------------
    fn item(&mut self) -> PResult<Item> {
        let start = self.span();
        let vis = self.eat_kw(Kw::Pub);
        let kind = match self.peek() {
            TokKind::Keyword(Kw::Fn) => ItemKind::Fn(self.fn_def(vis)?),
            TokKind::Keyword(Kw::Value) | TokKind::Keyword(Kw::Struct)
                if self.is_struct_ahead() =>
            {
                ItemKind::Struct(self.struct_def(vis)?)
            }
            TokKind::Keyword(Kw::Value) | TokKind::Keyword(Kw::Enum) => {
                ItemKind::Enum(self.enum_def(vis)?)
            }
            TokKind::Keyword(Kw::Struct) => ItemKind::Struct(self.struct_def(vis)?),
            TokKind::Keyword(Kw::Trait) => ItemKind::Trait(self.trait_def(vis)?),
            TokKind::Keyword(Kw::Impl) => ItemKind::Impl(self.impl_block()?),
            TokKind::Keyword(Kw::Type) => ItemKind::TypeAlias(self.type_alias(vis)?),
            TokKind::Keyword(Kw::Const) | TokKind::Keyword(Kw::Static) => {
                ItemKind::Const(self.const_def(vis)?)
            }
            TokKind::Keyword(Kw::Mod) => ItemKind::Mod(self.mod_def(vis)?),
            TokKind::Keyword(Kw::Use) => ItemKind::Use(self.use_decl()?),
            other => return Err(self.error(format!("expected item, found {:?}", other))),
        };
        Ok(Item { kind, span: start.to(self.prev_span()) })
    }

    /// Disambiguate `value struct` / `struct` from `value enum` / `enum`.
    fn is_struct_ahead(&self) -> bool {
        match self.peek() {
            TokKind::Keyword(Kw::Struct) => true,
            TokKind::Keyword(Kw::Value) => matches!(self.peek_at(1), TokKind::Keyword(Kw::Struct)),
            _ => false,
        }
    }

    fn fn_def(&mut self, vis: bool) -> PResult<FnDef> {
        let start = self.expect_kw(Kw::Fn)?;
        let name = self.ident()?;
        let generics = self.generics()?;
        self.expect(&TokKind::LParen)?;
        let (params, has_self) = self.params()?;
        self.expect(&TokKind::RParen)?;
        let ret = if self.eat(&TokKind::Arrow) { Some(self.ty()?) } else { None };
        let generics = self.maybe_where(generics)?;
        let body = self.block()?;
        Ok(FnDef { vis, name, generics, params, has_self, ret, body, span: start.to(self.prev_span()) })
    }

    fn params(&mut self) -> PResult<(Vec<Param>, bool)> {
        let mut params = Vec::new();
        let mut has_self = false;
        if self.at_kw(Kw::SelfValue) {
            self.bump();
            has_self = true;
            if !self.at(&TokKind::RParen) {
                self.expect(&TokKind::Comma)?;
            }
        }
        while !self.at(&TokKind::RParen) {
            let pstart = self.span();
            let is_mut = self.eat_kw(Kw::Mut);
            let name = self.ident()?;
            self.expect(&TokKind::Colon)?;
            let ty = self.ty()?;
            params.push(Param { is_mut, name, ty, span: pstart.to(self.prev_span()) });
            if !self.eat(&TokKind::Comma) {
                break;
            }
        }
        Ok((params, has_self))
    }

    fn struct_def(&mut self, vis: bool) -> PResult<StructDef> {
        let start = self.span();
        let is_value = self.eat_kw(Kw::Value);
        self.expect_kw(Kw::Struct)?;
        let name = self.ident()?;
        let generics = self.generics()?;
        let generics = self.maybe_where(generics)?;
        let body = if self.at(&TokKind::LBrace) {
            self.bump();
            let mut fields = Vec::new();
            while !self.at(&TokKind::RBrace) {
                fields.push(self.field_def()?);
                if !self.eat(&TokKind::Comma) {
                    break;
                }
            }
            self.expect(&TokKind::RBrace)?;
            StructBody::Named(fields)
        } else if self.at(&TokKind::LParen) {
            self.bump();
            let mut tys = Vec::new();
            while !self.at(&TokKind::RParen) {
                tys.push(self.ty()?);
                if !self.eat(&TokKind::Comma) {
                    break;
                }
            }
            self.expect(&TokKind::RParen)?;
            self.expect(&TokKind::Semi)?;
            StructBody::Tuple(tys)
        } else {
            self.expect(&TokKind::Semi)?;
            StructBody::Unit
        };
        Ok(StructDef { vis, is_value, name, generics, body, span: start.to(self.prev_span()) })
    }

    fn field_def(&mut self) -> PResult<FieldDef> {
        let start = self.span();
        let vis = self.eat_kw(Kw::Pub);
        let name = self.ident()?;
        self.expect(&TokKind::Colon)?;
        let ty = self.ty()?;
        Ok(FieldDef { vis, name, ty, span: start.to(self.prev_span()) })
    }

    fn enum_def(&mut self, vis: bool) -> PResult<EnumDef> {
        let start = self.span();
        let is_value = self.eat_kw(Kw::Value);
        self.expect_kw(Kw::Enum)?;
        let name = self.ident()?;
        let generics = self.generics()?;
        let generics = self.maybe_where(generics)?;
        self.expect(&TokKind::LBrace)?;
        let mut variants = Vec::new();
        while !self.at(&TokKind::RBrace) {
            let vstart = self.span();
            let vname = self.ident()?;
            let payload = if self.at(&TokKind::LParen) {
                self.bump();
                let mut tys = Vec::new();
                while !self.at(&TokKind::RParen) {
                    tys.push(self.ty()?);
                    if !self.eat(&TokKind::Comma) { break; }
                }
                self.expect(&TokKind::RParen)?;
                VariantPayload::Tuple(tys)
            } else if self.at(&TokKind::LBrace) {
                self.bump();
                let mut fields = Vec::new();
                while !self.at(&TokKind::RBrace) {
                    fields.push(self.field_def()?);
                    if !self.eat(&TokKind::Comma) { break; }
                }
                self.expect(&TokKind::RBrace)?;
                VariantPayload::Named(fields)
            } else {
                VariantPayload::None
            };
            variants.push(VariantDef { name: vname, payload, span: vstart.to(self.prev_span()) });
            if !self.eat(&TokKind::Comma) { break; }
        }
        self.expect(&TokKind::RBrace)?;
        Ok(EnumDef { vis, is_value, name, generics, variants, span: start.to(self.prev_span()) })
    }

    fn trait_def(&mut self, vis: bool) -> PResult<TraitDef> {
        let start = self.expect_kw(Kw::Trait)?;
        let name = self.ident()?;
        let generics = self.generics()?;
        let mut supertraits = Vec::new();
        if self.eat(&TokKind::Colon) {
            loop {
                supertraits.push(self.trait_ref()?);
                if !self.eat(&TokKind::Plus) { break; }
            }
        }
        let generics = self.maybe_where(generics)?;
        self.expect(&TokKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(&TokKind::RBrace) {
            items.push(self.trait_item()?);
        }
        self.expect(&TokKind::RBrace)?;
        Ok(TraitDef { vis, name, generics, supertraits, items, span: start.to(self.prev_span()) })
    }

    fn trait_item(&mut self) -> PResult<TraitItem> {
        if self.eat_kw(Kw::Type) {
            let n = self.ident()?;
            self.expect(&TokKind::Semi)?;
            return Ok(TraitItem::AssocType(n));
        }
        // A method: parse the signature, then either `;` (required) or a body.
        let start = self.expect_kw(Kw::Fn)?;
        let name = self.ident()?;
        let generics = self.generics()?;
        self.expect(&TokKind::LParen)?;
        let (params, has_self) = self.params()?;
        self.expect(&TokKind::RParen)?;
        let ret = if self.eat(&TokKind::Arrow) { Some(self.ty()?) } else { None };
        let generics = self.maybe_where(generics)?;
        if self.eat(&TokKind::Semi) {
            Ok(TraitItem::Required(FnSig {
                name, generics, params, has_self, ret, span: start.to(self.prev_span()),
            }))
        } else {
            let body = self.block()?;
            Ok(TraitItem::Provided(FnDef {
                vis: false, name, generics, params, has_self, ret, body,
                span: start.to(self.prev_span()),
            }))
        }
    }

    fn impl_block(&mut self) -> PResult<ImplBlock> {
        let start = self.expect_kw(Kw::Impl)?;
        let generics = self.generics()?;
        // `impl Trait for Type` vs `impl Type`. Parse a type; if `for` follows,
        // the parsed thing was the trait ref.
        let first = self.ty()?;
        let (trait_ref, self_ty) = if self.eat_kw(Kw::For) {
            let tr = type_to_trait_ref(first)
                .ok_or_else(|| self.error("expected a trait name before `for`"))?;
            (Some(tr), self.ty()?)
        } else {
            (None, first)
        };
        let generics = self.maybe_where(generics)?;
        self.expect(&TokKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(&TokKind::RBrace) {
            let vis = self.eat_kw(Kw::Pub);
            items.push(self.fn_def(vis)?);
        }
        self.expect(&TokKind::RBrace)?;
        Ok(ImplBlock { generics, trait_ref, self_ty, items, span: start.to(self.prev_span()) })
    }

    fn type_alias(&mut self, vis: bool) -> PResult<TypeAlias> {
        let start = self.expect_kw(Kw::Type)?;
        let name = self.ident()?;
        let generics = self.generics()?;
        self.expect(&TokKind::Eq)?;
        let ty = self.ty()?;
        self.expect(&TokKind::Semi)?;
        Ok(TypeAlias { vis, name, generics, ty, span: start.to(self.prev_span()) })
    }

    fn const_def(&mut self, vis: bool) -> PResult<ConstDef> {
        let start = self.span();
        // `const` or `static`.
        self.bump();
        let name = self.ident()?;
        self.expect(&TokKind::Colon)?;
        let ty = self.ty()?;
        self.expect(&TokKind::Eq)?;
        let value = self.expr()?;
        self.expect(&TokKind::Semi)?;
        Ok(ConstDef { vis, name, ty, value, span: start.to(self.prev_span()) })
    }

    fn mod_def(&mut self, vis: bool) -> PResult<ModDef> {
        let start = self.expect_kw(Kw::Mod)?;
        let name = self.ident()?;
        self.expect(&TokKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(&TokKind::RBrace) {
            items.push(self.item()?);
        }
        self.expect(&TokKind::RBrace)?;
        Ok(ModDef { vis, name, items, span: start.to(self.prev_span()) })
    }

    fn use_decl(&mut self) -> PResult<UsePath> {
        let start = self.expect_kw(Kw::Use)?;
        let mut segments = vec![self.ident()?];
        while self.eat(&TokKind::ColonColon) {
            segments.push(self.ident()?);
        }
        self.expect(&TokKind::Semi)?;
        Ok(UsePath { segments, span: start.to(self.prev_span()) })
    }

    // ---- generics ----------------------------------------------------------
    fn generics(&mut self) -> PResult<Generics> {
        let mut g = Generics::default();
        if !self.at(&TokKind::Lt) {
            return Ok(g);
        }
        self.bump(); // <
        while !self.at(&TokKind::Gt) {
            let pstart = self.span();
            let name = self.ident()?;
            let mut bounds = Vec::new();
            if self.eat(&TokKind::Colon) {
                loop {
                    bounds.push(self.trait_ref()?);
                    if !self.eat(&TokKind::Plus) { break; }
                }
            }
            g.params.push(TypeParam { name, bounds, span: pstart.to(self.prev_span()) });
            if !self.eat(&TokKind::Comma) { break; }
        }
        self.expect_gt()?;
        Ok(g)
    }

    fn maybe_where(&mut self, mut g: Generics) -> PResult<Generics> {
        if self.eat_kw(Kw::Where) {
            loop {
                let ty = self.ty()?;
                self.expect(&TokKind::Colon)?;
                let mut bounds = Vec::new();
                loop {
                    bounds.push(self.trait_ref()?);
                    if !self.eat(&TokKind::Plus) { break; }
                }
                g.where_clauses.push(WherePredicate { ty, bounds });
                if !self.eat(&TokKind::Comma) { break; }
                // A `{` ends the where clause.
                if self.at(&TokKind::LBrace) { break; }
            }
        }
        Ok(g)
    }

    fn trait_ref(&mut self) -> PResult<TraitRef> {
        let start = self.span();
        let path = self.path()?;
        let args = self.opt_type_args()?;
        Ok(TraitRef { path, args, span: start.to(self.prev_span()) })
    }

    // ---- types -------------------------------------------------------------
    fn ty(&mut self) -> PResult<Type> {
        let start = self.span();
        let kind = match self.peek() {
            TokKind::LParen => {
                self.bump();
                let mut tys = Vec::new();
                while !self.at(&TokKind::RParen) {
                    tys.push(self.ty()?);
                    if !self.eat(&TokKind::Comma) { break; }
                }
                self.expect(&TokKind::RParen)?;
                // `(T)` is just T; `()` is unit; `(A, B)` is a tuple.
                if tys.len() == 1 {
                    return Ok(tys.into_iter().next().unwrap());
                }
                TypeKind::Tuple(tys)
            }
            TokKind::LBracket => {
                self.bump();
                let elem = Box::new(self.ty()?);
                self.expect(&TokKind::Semi)?;
                let count = Box::new(self.expr()?);
                self.expect(&TokKind::RBracket)?;
                TypeKind::Array(elem, count)
            }
            TokKind::Keyword(Kw::Fn) => {
                self.bump();
                self.expect(&TokKind::LParen)?;
                let mut params = Vec::new();
                while !self.at(&TokKind::RParen) {
                    params.push(self.ty()?);
                    if !self.eat(&TokKind::Comma) { break; }
                }
                self.expect(&TokKind::RParen)?;
                let ret = if self.eat(&TokKind::Arrow) { Some(Box::new(self.ty()?)) } else { None };
                TypeKind::Fn(params, ret)
            }
            TokKind::Keyword(Kw::SelfType) => {
                self.bump();
                TypeKind::SelfType
            }
            TokKind::Ident(_) => {
                let path = self.path()?;
                let args = self.opt_type_args()?;
                TypeKind::Path(path, args)
            }
            other => return Err(self.error(format!("expected type, found {:?}", other))),
        };
        Ok(Type { kind, span: start.to(self.prev_span()) })
    }

    fn opt_type_args(&mut self) -> PResult<Vec<Type>> {
        if !self.at(&TokKind::Lt) {
            return Ok(Vec::new());
        }
        self.bump();
        let mut args = Vec::new();
        while !self.at(&TokKind::Gt) && !self.at(&TokKind::Shr) {
            args.push(self.ty()?);
            if !self.eat(&TokKind::Comma) { break; }
        }
        self.expect_gt()?;
        Ok(args)
    }

    /// Consume a `>`, splitting a `>>` token into a single `>` in place when
    /// closing nested generics (`Vec<Vec<T>>`).
    fn expect_gt(&mut self) -> PResult<()> {
        match self.peek() {
            TokKind::Gt => { self.bump(); Ok(()) }
            TokKind::Shr => {
                // Rewrite `>>` to `>` in the owned token stream and leave the
                // cursor so the enclosing generic's close consumes it.
                let sp = self.span();
                self.toks[self.pos].kind = TokKind::Gt;
                let _ = sp;
                Ok(())
            }
            other => Err(self.error(format!("expected `>`, found {:?}", other))),
        }
    }

    // ---- paths -------------------------------------------------------------
    fn path(&mut self) -> PResult<Path> {
        let start = self.span();
        let mut segments = vec![self.path_seg()?];
        while self.at(&TokKind::ColonColon)
            && matches!(self.peek_at(1), TokKind::Ident(_) | TokKind::Keyword(Kw::SelfType))
        {
            self.bump();
            segments.push(self.path_seg()?);
        }
        Ok(Path { segments, span: start.to(self.prev_span()) })
    }
    fn path_seg(&mut self) -> PResult<String> {
        match self.peek().clone() {
            TokKind::Ident(s) => { self.bump(); Ok(s) }
            TokKind::Keyword(Kw::SelfType) => { self.bump(); Ok("Self".to_string()) }
            other => Err(self.error(format!("expected path segment, found {:?}", other))),
        }
    }

    // ---- statements + blocks ----------------------------------------------
    fn block(&mut self) -> PResult<Block> {
        let start = self.expect(&TokKind::LBrace)?;
        let mut stmts = Vec::new();
        let mut tail = None;
        while !self.at(&TokKind::RBrace) {
            if self.at_kw(Kw::Let) {
                stmts.push(self.let_stmt()?);
                continue;
            }
            if self.is_item_start() {
                stmts.push(Stmt::Item(self.item()?));
                continue;
            }
            let e = self.expr()?;
            if self.eat(&TokKind::Semi) {
                stmts.push(Stmt::Expr(e));
            } else if self.at(&TokKind::RBrace) {
                tail = Some(e);
                break;
            } else if expr_is_block_like(&e) {
                // Block-like exprs used as statements need no `;`.
                stmts.push(Stmt::Expr(e));
            } else {
                return Err(self.error("expected `;` or `}` after expression"));
            }
        }
        self.expect(&TokKind::RBrace)?;
        Ok(Block { stmts, tail, span: start.to(self.prev_span()) })
    }

    fn is_item_start(&self) -> bool {
        matches!(
            self.peek(),
            TokKind::Keyword(
                Kw::Fn | Kw::Struct | Kw::Enum | Kw::Trait | Kw::Impl
                    | Kw::Type | Kw::Const | Kw::Static | Kw::Mod | Kw::Use | Kw::Value
            )
        )
    }

    fn let_stmt(&mut self) -> PResult<Stmt> {
        let start = self.expect_kw(Kw::Let)?;
        let pattern = self.pattern()?;
        let ty = if self.eat(&TokKind::Colon) { Some(self.ty()?) } else { None };
        let init = if self.eat(&TokKind::Eq) { Some(self.expr()?) } else { None };
        self.expect(&TokKind::Semi)?;
        Ok(Stmt::Let { pattern, ty, init, span: start.to(self.prev_span()) })
    }

    // ---- expressions (Pratt) ----------------------------------------------
    fn expr(&mut self) -> PResult<Expr> {
        self.expr_bp(0)
    }

    fn expr_bp(&mut self, min_bp: u8) -> PResult<Expr> {
        let start = self.span();
        let mut lhs = self.unary()?;

        loop {
            // assignment (right-assoc, lowest)
            if let Some(op) = self.assign_op() {
                if min_bp > 1 { break; }
                self.bump();
                let value = self.expr_bp(1)?;
                let span = start.to(self.prev_span());
                lhs = Expr { kind: Box::new(ExprKind::Assign { target: lhs, op, value }), span };
                continue;
            }
            // range
            if matches!(self.peek(), TokKind::DotDot | TokKind::DotDotEq) && min_bp <= 2 {
                let inclusive = matches!(self.peek(), TokKind::DotDotEq);
                self.bump();
                let hi = if self.starts_expr() { Some(self.expr_bp(3)?) } else { None };
                let span = start.to(self.prev_span());
                lhs = Expr {
                    kind: Box::new(ExprKind::Range { lo: Some(lhs), hi, inclusive }),
                    span,
                };
                continue;
            }

            let Some((op, l_bp, r_bp)) = self.bin_op() else { break };
            if l_bp < min_bp { break; }
            self.bump();
            let rhs = self.expr_bp(r_bp)?;
            let span = start.to(self.prev_span());
            lhs = Expr { kind: Box::new(ExprKind::Binary(op, lhs, rhs)), span };
        }
        Ok(lhs)
    }

    fn assign_op(&self) -> Option<Option<BinOp>> {
        Some(match self.peek() {
            TokKind::Eq => None,
            TokKind::PlusEq => Some(BinOp::Add),
            TokKind::MinusEq => Some(BinOp::Sub),
            TokKind::StarEq => Some(BinOp::Mul),
            TokKind::SlashEq => Some(BinOp::Div),
            TokKind::PercentEq => Some(BinOp::Rem),
            _ => return None,
        })
    }

    fn bin_op(&self) -> Option<(BinOp, u8, u8)> {
        // (op, left bp, right bp). Higher binds tighter.
        Some(match self.peek() {
            TokKind::OrOr => (BinOp::Or, 4, 5),
            TokKind::AndAnd => (BinOp::And, 6, 7),
            TokKind::EqEq => (BinOp::Eq, 8, 9),
            TokKind::Ne => (BinOp::Ne, 8, 9),
            TokKind::Lt => (BinOp::Lt, 8, 9),
            TokKind::Le => (BinOp::Le, 8, 9),
            TokKind::Gt => (BinOp::Gt, 8, 9),
            TokKind::Ge => (BinOp::Ge, 8, 9),
            TokKind::Pipe => (BinOp::BitOr, 10, 11),
            TokKind::Caret => (BinOp::BitXor, 12, 13),
            TokKind::Amp => (BinOp::BitAnd, 14, 15),
            TokKind::Shl => (BinOp::Shl, 16, 17),
            TokKind::Shr => (BinOp::Shr, 16, 17),
            TokKind::Plus => (BinOp::Add, 18, 19),
            TokKind::Minus => (BinOp::Sub, 18, 19),
            TokKind::Star => (BinOp::Mul, 20, 21),
            TokKind::Slash => (BinOp::Div, 20, 21),
            TokKind::Percent => (BinOp::Rem, 20, 21),
            _ => return None,
        })
    }

    fn unary(&mut self) -> PResult<Expr> {
        let start = self.span();
        let op = match self.peek() {
            TokKind::Minus => Some(UnOp::Neg),
            TokKind::Not => Some(UnOp::Not),
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let e = self.unary()?;
            let span = start.to(self.prev_span());
            return Ok(Expr { kind: Box::new(ExprKind::Unary(op, e)), span });
        }
        self.postfix()
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let start = self.span();
        let mut e = self.primary()?;
        loop {
            match self.peek() {
                TokKind::LParen => {
                    self.bump();
                    let args = self.call_args()?;
                    let span = start.to(self.prev_span());
                    e = Expr { kind: Box::new(ExprKind::Call(e, args)), span };
                }
                TokKind::Dot => {
                    self.bump();
                    match self.peek().clone() {
                        TokKind::Ident(name) => {
                            self.bump();
                            if self.at(&TokKind::LParen) {
                                self.bump();
                                let args = self.call_args()?;
                                let span = start.to(self.prev_span());
                                e = Expr {
                                    kind: Box::new(ExprKind::MethodCall {
                                        recv: e, method: name, args, span,
                                    }),
                                    span,
                                };
                            } else {
                                let span = start.to(self.prev_span());
                                e = Expr {
                                    kind: Box::new(ExprKind::Field {
                                        base: e, field: FieldAccess::Named(name),
                                    }),
                                    span,
                                };
                            }
                        }
                        TokKind::Int(n, _) => {
                            self.bump();
                            let span = start.to(self.prev_span());
                            e = Expr {
                                kind: Box::new(ExprKind::Field {
                                    base: e, field: FieldAccess::Tuple(n as u32),
                                }),
                                span,
                            };
                        }
                        other => return Err(self.error(format!("expected field, found {:?}", other))),
                    }
                }
                TokKind::LBracket => {
                    self.bump();
                    let index = self.expr()?;
                    self.expect(&TokKind::RBracket)?;
                    let span = start.to(self.prev_span());
                    e = Expr { kind: Box::new(ExprKind::Index { base: e, index }), span };
                }
                TokKind::Question => {
                    self.bump();
                    let span = start.to(self.prev_span());
                    e = Expr { kind: Box::new(ExprKind::Try(e)), span };
                }
                TokKind::Keyword(Kw::As) => {
                    self.bump();
                    let t = self.ty()?;
                    let span = start.to(self.prev_span());
                    e = Expr { kind: Box::new(ExprKind::Cast(e, t)), span };
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn call_args(&mut self) -> PResult<Vec<Expr>> {
        let mut args = Vec::new();
        while !self.at(&TokKind::RParen) {
            args.push(self.expr()?);
            if !self.eat(&TokKind::Comma) { break; }
        }
        self.expect(&TokKind::RParen)?;
        Ok(args)
    }

    fn primary(&mut self) -> PResult<Expr> {
        let start = self.span();
        let kind = match self.peek().clone() {
            TokKind::Int(n, sfx) => { self.bump(); ExprKind::Int(n, sfx) }
            TokKind::Float(f, sfx) => { self.bump(); ExprKind::Float(f, sfx) }
            TokKind::Str(s) => { self.bump(); ExprKind::Str(s) }
            TokKind::Char(c) => { self.bump(); ExprKind::Char(c) }
            TokKind::Keyword(Kw::True) => { self.bump(); ExprKind::Bool(true) }
            TokKind::Keyword(Kw::False) => { self.bump(); ExprKind::Bool(false) }
            TokKind::Keyword(Kw::If) => return self.if_expr(),
            TokKind::Keyword(Kw::Match) => return self.match_expr(),
            TokKind::Keyword(Kw::While) => return self.while_expr(),
            TokKind::Keyword(Kw::Loop) => return self.loop_expr(),
            TokKind::Keyword(Kw::For) => return self.for_expr(),
            TokKind::LBrace => return Ok(self.block_expr()?),
            TokKind::Keyword(Kw::Return) => {
                self.bump();
                let v = if self.starts_expr() { Some(self.expr()?) } else { None };
                ExprKind::Return(v)
            }
            TokKind::Keyword(Kw::Break) => {
                self.bump();
                let v = if self.starts_expr() { Some(self.expr()?) } else { None };
                ExprKind::Break(v)
            }
            TokKind::Keyword(Kw::Continue) => { self.bump(); ExprKind::Continue }
            TokKind::Pipe | TokKind::OrOr => return self.closure(),
            TokKind::LParen => {
                self.bump();
                if self.at(&TokKind::RParen) {
                    self.bump();
                    ExprKind::Unit
                } else {
                    let mut elems = vec![self.expr()?];
                    let mut is_tuple = false;
                    while self.eat(&TokKind::Comma) {
                        is_tuple = true;
                        if self.at(&TokKind::RParen) { break; }
                        elems.push(self.expr()?);
                    }
                    self.expect(&TokKind::RParen)?;
                    if is_tuple { ExprKind::Tuple(elems) }
                    else { return Ok(elems.into_iter().next().unwrap()); }
                }
            }
            TokKind::LBracket => {
                self.bump();
                if self.at(&TokKind::RBracket) {
                    self.bump();
                    ExprKind::Array(ArrayLit::Elems(Vec::new()))
                } else {
                    let first = self.expr()?;
                    if self.eat(&TokKind::Semi) {
                        let count = self.expr()?;
                        self.expect(&TokKind::RBracket)?;
                        ExprKind::Array(ArrayLit::Repeat(Box::new(first), Box::new(count)))
                    } else {
                        let mut elems = vec![first];
                        while self.eat(&TokKind::Comma) {
                            if self.at(&TokKind::RBracket) { break; }
                            elems.push(self.expr()?);
                        }
                        self.expect(&TokKind::RBracket)?;
                        ExprKind::Array(ArrayLit::Elems(elems))
                    }
                }
            }
            TokKind::Ident(_) | TokKind::Keyword(Kw::SelfValue) | TokKind::Keyword(Kw::SelfType) => {
                let path = self.path_or_self()?;
                // Struct literal: `Path { ... }` — but only when a struct lit is
                // allowed here (it always is in primary position in v0).
                if self.at(&TokKind::LBrace) && self.looks_like_struct_lit() {
                    self.bump();
                    let mut fields = Vec::new();
                    while !self.at(&TokKind::RBrace) {
                        let fstart = self.span();
                        let name = self.ident()?;
                        let value = if self.eat(&TokKind::Colon) { Some(self.expr()?) } else { None };
                        fields.push(FieldInit { name, value, span: fstart.to(self.prev_span()) });
                        if !self.eat(&TokKind::Comma) { break; }
                    }
                    self.expect(&TokKind::RBrace)?;
                    let span = start.to(self.prev_span());
                    ExprKind::StructLit { path, fields, span }
                } else {
                    ExprKind::Path(path)
                }
            }
            other => return Err(self.error(format!("expected expression, found {:?}", other))),
        };
        Ok(Expr { kind: Box::new(kind), span: start.to(self.prev_span()) })
    }

    fn path_or_self(&mut self) -> PResult<Path> {
        if self.at_kw(Kw::SelfValue) {
            let s = self.span();
            self.bump();
            return Ok(Path::single("self".to_string(), s));
        }
        self.path()
    }

    /// Heuristic: after a path, `{` begins a struct literal unless we're in a
    /// position where `{` starts a block (handled by callers gating struct lits
    /// out of `if`/`while`/`match` scrutinee via `no_struct` — v0 keeps it
    /// simple and always allows struct lits in primary expressions).
    fn looks_like_struct_lit(&self) -> bool {
        if self.no_struct > 0 {
            return false;
        }
        // `Ident {` or `Ident }` ... we accept `{ ident :` , `{ ident ,`,
        // `{ }`. This distinguishes from a following block in statement
        // position, which the block parser handles before reaching here.
        match self.peek_at(1) {
            TokKind::RBrace => true,
            TokKind::Ident(_) => matches!(self.peek_at(2), TokKind::Colon | TokKind::Comma | TokKind::RBrace),
            _ => false,
        }
    }

    fn closure(&mut self) -> PResult<Expr> {
        let start = self.span();
        let mut params = Vec::new();
        if self.eat(&TokKind::OrOr) {
            // empty params `||`
        } else {
            self.expect(&TokKind::Pipe)?;
            while !self.at(&TokKind::Pipe) {
                let pstart = self.span();
                let name = self.ident()?;
                let ty = if self.eat(&TokKind::Colon) { Some(self.ty()?) } else { None };
                params.push(ClosureParam { name, ty, span: pstart.to(self.prev_span()) });
                if !self.eat(&TokKind::Comma) { break; }
            }
            self.expect(&TokKind::Pipe)?;
        }
        let ret = if self.eat(&TokKind::Arrow) { Some(self.ty()?) } else { None };
        let body = self.expr()?;
        let span = start.to(self.prev_span());
        Ok(Expr { kind: Box::new(ExprKind::Closure { params, ret, body }), span })
    }

    fn block_expr(&mut self) -> PResult<Expr> {
        let start = self.span();
        let b = self.block()?;
        Ok(Expr { kind: Box::new(ExprKind::Block(b)), span: start.to(self.prev_span()) })
    }

    fn if_expr(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::If)?;
        let cond = self.expr_no_struct()?;
        let then_branch = self.block()?;
        let else_branch = if self.eat_kw(Kw::Else) {
            if self.at_kw(Kw::If) {
                Some(self.if_expr()?)
            } else {
                Some(self.block_expr()?)
            }
        } else {
            None
        };
        let span = start.to(self.prev_span());
        Ok(Expr { kind: Box::new(ExprKind::If { cond, then_branch, else_branch }), span })
    }

    fn match_expr(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::Match)?;
        let scrutinee = self.expr_no_struct()?;
        self.expect(&TokKind::LBrace)?;
        let mut arms = Vec::new();
        while !self.at(&TokKind::RBrace) {
            let astart = self.span();
            let pattern = self.pattern()?;
            let guard = if self.eat_kw(Kw::If) { Some(self.expr()?) } else { None };
            self.expect(&TokKind::FatArrow)?;
            let body = self.expr()?;
            self.eat(&TokKind::Comma);
            arms.push(MatchArm { pattern, guard, body, span: astart.to(self.prev_span()) });
        }
        self.expect(&TokKind::RBrace)?;
        let span = start.to(self.prev_span());
        Ok(Expr { kind: Box::new(ExprKind::Match { scrutinee, arms }), span })
    }

    fn while_expr(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::While)?;
        let cond = self.expr_no_struct()?;
        let body = self.block()?;
        let span = start.to(self.prev_span());
        Ok(Expr { kind: Box::new(ExprKind::While { cond, body }), span })
    }

    fn loop_expr(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::Loop)?;
        let body = self.block()?;
        let span = start.to(self.prev_span());
        Ok(Expr { kind: Box::new(ExprKind::Loop { body }), span })
    }

    fn for_expr(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::For)?;
        let pat = self.pattern()?;
        self.expect_kw(Kw::In)?;
        let iter = self.expr_no_struct()?;
        let body = self.block()?;
        let span = start.to(self.prev_span());
        Ok(Expr { kind: Box::new(ExprKind::For { pat, iter, body }), span })
    }

    /// Parse an expression but forbid a trailing struct literal at the top
    /// level (so `if x { }` parses `x` as the cond, not `x { }` as a struct
    /// lit). Implemented with a flag.
    fn expr_no_struct(&mut self) -> PResult<Expr> {
        self.no_struct += 1;
        let r = self.expr();
        self.no_struct -= 1;
        r
    }

    fn starts_expr(&self) -> bool {
        matches!(
            self.peek(),
            TokKind::Int(..) | TokKind::Float(..) | TokKind::Str(_) | TokKind::Char(_)
                | TokKind::Ident(_) | TokKind::LParen | TokKind::LBracket | TokKind::LBrace
                | TokKind::Minus | TokKind::Not | TokKind::Pipe | TokKind::OrOr
                | TokKind::Keyword(
                    Kw::True | Kw::False | Kw::If | Kw::Match | Kw::While | Kw::Loop
                        | Kw::For | Kw::Return | Kw::Break | Kw::Continue
                        | Kw::SelfValue | Kw::SelfType
                )
        )
    }

    // ---- patterns ----------------------------------------------------------
    fn pattern(&mut self) -> PResult<Pattern> {
        let start = self.span();
        let kind = match self.peek().clone() {
            TokKind::Ident(name) if name == "_" => { self.bump(); PatternKind::Wildcard }
            TokKind::Keyword(Kw::Mut) => {
                self.bump();
                let name = self.ident()?;
                PatternKind::Binding { is_mut: true, name }
            }
            TokKind::Int(n, _) => { self.bump(); PatternKind::Literal(LitPattern::Int(n)) }
            TokKind::Keyword(Kw::True) => { self.bump(); PatternKind::Literal(LitPattern::Bool(true)) }
            TokKind::Keyword(Kw::False) => { self.bump(); PatternKind::Literal(LitPattern::Bool(false)) }
            TokKind::Char(c) => { self.bump(); PatternKind::Literal(LitPattern::Char(c)) }
            TokKind::Str(s) => { self.bump(); PatternKind::Literal(LitPattern::Str(s)) }
            TokKind::LParen => {
                self.bump();
                let mut pats = Vec::new();
                while !self.at(&TokKind::RParen) {
                    pats.push(self.pattern()?);
                    if !self.eat(&TokKind::Comma) { break; }
                }
                self.expect(&TokKind::RParen)?;
                PatternKind::Tuple(pats)
            }
            TokKind::Ident(_) => {
                let path = self.path()?;
                if self.at(&TokKind::LParen) {
                    self.bump();
                    let mut payload = Vec::new();
                    while !self.at(&TokKind::RParen) {
                        payload.push(self.pattern()?);
                        if !self.eat(&TokKind::Comma) { break; }
                    }
                    self.expect(&TokKind::RParen)?;
                    PatternKind::Variant { path, payload }
                } else if self.at(&TokKind::LBrace) {
                    self.bump();
                    let mut fields = Vec::new();
                    let mut rest = false;
                    while !self.at(&TokKind::RBrace) {
                        if self.eat(&TokKind::DotDot) { rest = true; break; }
                        let fname = self.ident()?;
                        let fpat = if self.eat(&TokKind::Colon) {
                            self.pattern()?
                        } else {
                            Pattern {
                                kind: PatternKind::Binding { is_mut: false, name: fname.clone() },
                                span: self.prev_span(),
                            }
                        };
                        fields.push((fname, fpat));
                        if !self.eat(&TokKind::Comma) { break; }
                    }
                    self.expect(&TokKind::RBrace)?;
                    PatternKind::Struct { path, fields, rest }
                } else if path.is_single() {
                    // A bare lowercase-ish single ident is a binding; an
                    // upper-case one *could* be a unit variant. Resolution
                    // disambiguates; we record it as a Binding-or-variant by
                    // name and let resolve decide. We choose Variant if the
                    // name starts uppercase, else Binding.
                    let name = path.segments[0].clone();
                    if name.chars().next().is_some_and(|c| c.is_uppercase()) {
                        PatternKind::Variant { path, payload: Vec::new() }
                    } else {
                        PatternKind::Binding { is_mut: false, name }
                    }
                } else {
                    PatternKind::Variant { path, payload: Vec::new() }
                }
            }
            other => return Err(self.error(format!("expected pattern, found {:?}", other))),
        };
        Ok(Pattern { kind, span: start.to(self.prev_span()) })
    }

}

fn expr_is_block_like(e: &Expr) -> bool {
    matches!(
        &*e.kind,
        ExprKind::Block(_)
            | ExprKind::If { .. }
            | ExprKind::Match { .. }
            | ExprKind::While { .. }
            | ExprKind::Loop { .. }
            | ExprKind::For { .. }
    )
}

fn type_to_trait_ref(t: Type) -> Option<TraitRef> {
    match t.kind {
        TypeKind::Path(path, args) => Some(TraitRef { path, args, span: t.span }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn parse(src: &str) -> Module {
        let toks = lex(src).expect("lex");
        parse_module(&toks).unwrap_or_else(|e| panic!("parse error: {} at {:?}", e.msg, e.span))
    }

    #[test]
    fn parses_fib() {
        let m = parse(include_str!("../examples/fib.gcr"));
        assert_eq!(m.items.len(), 2);
        match &m.items[0].kind {
            ItemKind::Fn(f) => {
                assert_eq!(f.name, "fib");
                assert_eq!(f.params.len(), 1);
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn parses_types_example() {
        let m = parse(include_str!("../examples/types.gcr"));
        // value struct, struct, 2 enums, trait, 2 impls, 3 fns, main
        let kinds: Vec<&str> = m.items.iter().map(|i| match &i.kind {
            ItemKind::Fn(_) => "fn",
            ItemKind::Struct(_) => "struct",
            ItemKind::Enum(_) => "enum",
            ItemKind::Trait(_) => "trait",
            ItemKind::Impl(_) => "impl",
            _ => "other",
        }).collect();
        assert!(kinds.contains(&"trait"));
        assert_eq!(kinds.iter().filter(|k| **k == "impl").count(), 2);
        assert_eq!(kinds.iter().filter(|k| **k == "enum").count(), 2);
        // First item is the value struct Vec3.
        match &m.items[0].kind {
            ItemKind::Struct(s) => { assert!(s.is_value); assert_eq!(s.name, "Vec3"); }
            _ => panic!("expected value struct first"),
        }
    }

    #[test]
    fn nested_generics_close() {
        let m = parse("fn f() -> Vec<Vec<i64>> { x }");
        match &m.items[0].kind {
            ItemKind::Fn(f) => match &f.ret.as_ref().unwrap().kind {
                TypeKind::Path(p, args) => {
                    assert_eq!(p.last(), "Vec");
                    assert_eq!(args.len(), 1);
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn precedence() {
        // 1 + 2 * 3 == 7  parses as  (1 + (2*3)) == 7
        let m = parse("fn f() -> bool { 1 + 2 * 3 == 7 }");
        let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        match &*tail.kind {
            ExprKind::Binary(BinOp::Eq, lhs, _) => match &*lhs.kind {
                ExprKind::Binary(BinOp::Add, _, r) => {
                    assert!(matches!(&*r.kind, ExprKind::Binary(BinOp::Mul, _, _)));
                }
                _ => panic!("add not at top of lhs"),
            },
            _ => panic!("eq not at top"),
        }
    }

    #[test]
    fn if_cond_not_struct_lit() {
        // `if x { 1 } else { 2 }` — x is a path, not a struct literal.
        let m = parse("fn f(x: bool) -> i64 { if x { 1 } else { 2 } }");
        let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
        assert!(matches!(&*f.body.tail.as_ref().unwrap().kind, ExprKind::If { .. }));
    }

    #[test]
    fn match_and_try() {
        let m = parse(
            "fn f() -> i64 { let q = g()?; match q { Result::Ok(v) => v, _ => 0 } }",
        );
        let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
        assert!(matches!(&*f.body.tail.as_ref().unwrap().kind, ExprKind::Match { .. }));
    }

    #[test]
    fn closures() {
        let m = parse("fn f() -> i64 { let g = |x: i64| x + 1; g(2) }");
        let ItemKind::Fn(f) = &m.items[0].kind else { panic!() };
        let Stmt::Let { init: Some(init), .. } = &f.body.stmts[0] else { panic!() };
        assert!(matches!(&*init.kind, ExprKind::Closure { .. }));
    }
}
