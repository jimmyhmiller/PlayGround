//! Recursive-descent parser for the Flow DSL.

use super::ast::*;
use super::lex::{RawNamePart, Tok, Token};

pub fn parse(src: &str) -> Result<File, String> {
    let tokens = super::lex::lex(src)?;
    let mut p = Parser { toks: tokens, pos: 0 };
    p.parse_file()
}

/// Split a probe format string into literal + hole parts.
///
/// Holes are `{expr}`; the expression inside is re-lexed and re-parsed
/// using the full DSL expression grammar. `\{` escapes a literal `{`
/// (handled by the outer string lexer, which passes unknown escapes
/// through as their trailing character — so by the time we see the
/// string here, `\{` has already become `{` and the backslash is gone).
/// A literal `}` just writes itself; only `{` begins a hole.
pub(crate) fn parse_probe_format(s: &str) -> Result<Vec<ProbePart>, String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            if !cur.is_empty() {
                parts.push(ProbePart::Literal(std::mem::take(&mut cur)));
            }
            let mut expr_src = String::new();
            let mut depth = 1;
            let mut closed = false;
            while let Some(ec) = chars.next() {
                if ec == '{' { depth += 1; expr_src.push(ec); }
                else if ec == '}' {
                    depth -= 1;
                    if depth == 0 { closed = true; break; }
                    expr_src.push(ec);
                }
                else { expr_src.push(ec); }
            }
            if !closed {
                return Err("unterminated `{` in format string".into());
            }
            let expr = parse_expr_from_str(&expr_src)
                .map_err(|e| format!("inside `{{{}}}`: {}", expr_src, e))?;
            parts.push(ProbePart::Hole(expr));
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        parts.push(ProbePart::Literal(cur));
    }
    Ok(parts)
}

/// Parse a standalone expression (used by probe format interpolation).
fn parse_expr_from_str(src: &str) -> Result<Expr, String> {
    let tokens = super::lex::lex(src)?;
    let mut p = Parser { toks: tokens, pos: 0 };
    let e = p.parse_expr()?;
    if !matches!(p.peek(), Tok::Eof) {
        return Err(format!("{}: unexpected trailing tokens in expression", p.here()));
    }
    Ok(e)
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok { &self.toks[self.pos].kind }
    fn here(&self) -> String {
        let t = &self.toks[self.pos];
        format!("{}:{}", t.line, t.col)
    }
    fn bump(&mut self) -> Tok { let t = self.toks[self.pos].kind.clone(); self.pos += 1; t }
    fn expect(&mut self, expect: Tok) -> Result<(), String> {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(&expect) {
            self.bump();
            Ok(())
        } else {
            Err(format!("{}: expected {:?}, got {:?}", self.here(), expect, self.peek()))
        }
    }
    fn eat(&mut self, expect: &Tok) -> bool {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(expect) {
            self.bump();
            true
        } else { false }
    }
    fn ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("{}: expected identifier, got {:?}", self.here(), other)),
        }
    }

    /// Parse a name in any name-bearing position (node decls, edge
    /// endpoints, emit targets, port `inner` mappings). Accepts plain
    /// `Tok::Ident` and `Tok::IdentTpl(...)` (`Cell_{x}_{y}` shape) and
    /// returns a `NameTpl`. Hole bodies are re-parsed as expressions
    /// using the full DSL grammar.
    fn name_tpl(&mut self) -> Result<NameTpl, String> {
        let here = self.here();
        match self.bump() {
            Tok::Ident(s) => Ok(NameTpl::plain(s)),
            Tok::IdentTpl(parts) => {
                let mut out = Vec::with_capacity(parts.len());
                for p in parts {
                    out.push(match p {
                        RawNamePart::Lit(s) => NamePart::Literal(s),
                        RawNamePart::Hole(src) => NamePart::Hole(
                            parse_expr_from_str(&src)
                                .map_err(|e| format!("{}: inside `{{{}}}`: {}", here, src, e))?
                        ),
                    });
                }
                Ok(NameTpl { parts: out })
            }
            other => Err(format!("{}: expected name, got {:?}", here, other)),
        }
    }

    // -------- top-level --------

    fn parse_file(&mut self) -> Result<File, String> {
        let mut items = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            items.push(self.parse_item()?);
        }
        Ok(File { items })
    }

    fn parse_item(&mut self) -> Result<Item, String> {
        match self.peek() {
            Tok::Params => self.parse_params(),
            Tok::Node => self.parse_node(),
            Tok::Compound => self.parse_compound(),
            Tok::Edges => self.parse_edges(),
            Tok::Scenario => self.parse_scenario(),
            Tok::For => self.parse_for(),
            other => Err(format!("{}: expected top-level item, got {:?}", self.here(), other)),
        }
    }

    fn parse_params(&mut self) -> Result<Item, String> {
        self.expect(Tok::Params)?;
        self.expect(Tok::LBrace)?;
        let mut out = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            let name = self.ident()?;
            self.expect(Tok::Colon)?;
            let value = self.parse_expr()?;
            // optional trailing ; or ,
            let _ = self.eat(&Tok::Semi) || self.eat(&Tok::Comma);
            out.push(ParamDecl { name, value });
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Params(out))
    }

    fn parse_node(&mut self) -> Result<Item, String> {
        self.expect(Tok::Node)?;
        let name_tpl = self.name_tpl()?;

        // Instance form: `node NAME : CLASS { slot: expr; ... }`.
        // No rules / probes / on_spawn — those come from the class.
        // Instance names support `{...}` templates so they can be the
        // target of a surrounding `for` loop (the common case for
        // grids — see examples/life.flow).
        if self.eat(&Tok::Colon) {
            let class = self.ident()?;
            self.expect(Tok::LBrace)?;
            let mut overrides = Vec::new();
            while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
                let slot = self.ident()?;
                self.expect(Tok::Colon)?;
                let value = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi) || self.eat(&Tok::Comma);
                overrides.push((slot, value));
            }
            self.expect(Tok::RBrace)?;
            return Ok(Item::Instance(crate::dsl::ast::InstanceDecl { name: name_tpl, class, overrides }));
        }

        self.expect(Tok::LBrace)?;
        let mut slots = Vec::new();
        let mut rules = Vec::new();
        let mut on_spawn = Vec::new();
        let mut probes = Vec::new();
        loop {
            match self.peek() {
                Tok::Slots => { slots = self.parse_slots()?; }
                Tok::Rule => { rules.push(self.parse_rule()?); }
                Tok::OnSpawn => { on_spawn = self.parse_on_spawn()?; }
                Tok::Probes => { probes = self.parse_probes()?; }
                Tok::RBrace => break,
                other => return Err(format!("{}: expected slots/rule/on_spawn/probes, got {:?}", self.here(), other)),
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Node(NodeDecl { name: name_tpl, slots, rules, on_spawn, probes }))
    }

    /// Parse `probes { LABEL: "FMT" ... }` where FMT is a string that may
    /// contain `{expr}` interpolation holes. The expression inside each
    /// hole is re-lexed + re-parsed using the same DSL expression
    /// grammar. Trailing `,` / `;` between entries is optional.
    fn parse_probes(&mut self) -> Result<Vec<ProbeDecl>, String> {
        self.expect(Tok::Probes)?;
        self.expect(Tok::LBrace)?;
        let mut out = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            let label = self.ident()?;
            self.expect(Tok::Colon)?;
            let fmt_line = self.here();
            let fmt = match self.bump() {
                Tok::Str(s) => s,
                other => return Err(format!(
                    "{}: probe `{}`: expected \"format string\", got {:?}",
                    fmt_line, label, other
                )),
            };
            let parts = parse_probe_format(&fmt)
                .map_err(|e| format!("{}: probe `{}`: {}", fmt_line, label, e))?;
            let _ = self.eat(&Tok::Comma) || self.eat(&Tok::Semi);
            out.push(ProbeDecl { label, parts });
        }
        self.expect(Tok::RBrace)?;
        Ok(out)
    }

    /// Parse `on_spawn { … }` — per-instance bootstrap wiring.
    ///
    /// Two statement shapes:
    ///   `self -> self : LATENCY_EXPR`    — create a self-edge
    ///   `inject TAG` or `inject TAG(PAYLOAD)` — seed inbox packet
    ///
    /// Trailing `;` / `,` is optional between statements.
    fn parse_on_spawn(&mut self) -> Result<Vec<OnSpawnStmt>, String> {
        self.expect(Tok::OnSpawn)?;
        self.expect(Tok::LBrace)?;
        let mut out = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            match self.peek().clone() {
                Tok::Self_ => {
                    self.bump();
                    self.expect(Tok::Arrow)?;
                    // Only self -> self supported. Cross-instance wiring
                    // belongs at the top-level `edges` block or on the
                    // canvas file format.
                    if !matches!(self.peek(), Tok::Self_) {
                        return Err(format!(
                            "{}: on_spawn edges must be `self -> self`",
                            self.here()
                        ));
                    }
                    self.bump();
                    self.expect(Tok::Colon)?;
                    let latency = self.parse_expr()?;
                    let _ = self.eat(&Tok::Semi) || self.eat(&Tok::Comma);
                    out.push(OnSpawnStmt::SelfEdge { latency });
                }
                Tok::Inject => {
                    self.bump();
                    let tag = self.ident()?;
                    let payload = if self.eat(&Tok::LParen) {
                        let e = if matches!(self.peek(), Tok::RParen) {
                            None
                        } else {
                            Some(self.parse_expr()?)
                        };
                        self.expect(Tok::RParen)?;
                        e
                    } else { None };
                    let _ = self.eat(&Tok::Semi) || self.eat(&Tok::Comma);
                    out.push(OnSpawnStmt::Inject { tag, payload });
                }
                other => return Err(format!(
                    "{}: expected `self -> self : …` or `inject …`, got {:?}",
                    self.here(), other
                )),
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(out)
    }

    fn parse_slots(&mut self) -> Result<Vec<SlotDecl>, String> {
        self.expect(Tok::Slots)?;
        self.expect(Tok::LBrace)?;
        let mut out = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            let name = self.ident()?;
            self.expect(Tok::Colon)?;
            let ty = self.parse_slot_type()?;
            // Accept `=` or `:=` as the init marker. Omit for Samples (auto-init empty).
            let init = if self.eat(&Tok::Equals) || self.eat(&Tok::Assign) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            let _ = self.eat(&Tok::Semi) || self.eat(&Tok::Comma);
            out.push(SlotDecl { name, ty, init });
        }
        self.expect(Tok::RBrace)?;
        Ok(out)
    }

    fn parse_slot_type(&mut self) -> Result<SlotType, String> {
        let name = self.ident()?;
        match name.as_str() {
            "Int" => Ok(SlotType::Int),
            "Float" => Ok(SlotType::Float),
            "Bool" => Ok(SlotType::Bool),
            "String" => Ok(SlotType::Str),
            "Nil" => Ok(SlotType::Nil),
            "Any" => Ok(SlotType::Any),
            "Samples" => {
                // Samples(CAP)
                self.expect(Tok::LParen)?;
                let cap = match self.bump() {
                    Tok::Int(n) if n > 0 => n as u32,
                    other => return Err(format!("{}: Samples expects capacity int, got {:?}", self.here(), other)),
                };
                self.expect(Tok::RParen)?;
                Ok(SlotType::Samples(cap))
            }
            other => Err(format!("{}: unknown slot type `{}`", self.here(), other)),
        }
    }

    fn parse_rule(&mut self) -> Result<RuleDecl, String> {
        self.expect(Tok::Rule)?;
        let name = self.ident()?;
        self.expect(Tok::LBrace)?;
        let mut ons = Vec::new();
        let mut when = None;
        let mut body = Vec::new();
        loop {
            match self.peek() {
                Tok::On => {
                    self.bump();
                    let pat = self.parse_pattern()?;
                    let _ = self.eat(&Tok::Semi);
                    ons.push(pat);
                }
                Tok::When => {
                    self.bump();
                    let e = self.parse_expr()?;
                    let _ = self.eat(&Tok::Semi);
                    when = Some(e);
                }
                Tok::Do => {
                    self.bump();
                    if self.eat(&Tok::LBrace) {
                        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
                            body.push(self.parse_stmt()?);
                        }
                        self.expect(Tok::RBrace)?;
                    } else {
                        body.push(self.parse_stmt()?);
                    }
                }
                Tok::RBrace => break,
                other => return Err(format!("{}: expected on/when/do, got {:?}", self.here(), other)),
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(RuleDecl { name, ons, when, body })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        match self.peek() {
            Tok::Ident(_) => {
                let name = self.ident()?;
                // `_` as identifier is handled elsewhere, but for safety:
                if name == "_" { return Ok(Pattern::Wild); }
                if matches!(self.peek(), Tok::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Tok::RParen) {
                        args.push(self.parse_pattern()?);
                        while self.eat(&Tok::Comma) {
                            args.push(self.parse_pattern()?);
                        }
                    }
                    self.expect(Tok::RParen)?;
                    Ok(Pattern::Variant(name, args))
                } else {
                    // A bare identifier in pattern position = variable binding.
                    Ok(Pattern::Var(name))
                }
            }
            Tok::Int(_) | Tok::Float(_) | Tok::Str(_) | Tok::TimeNs(_) | Tok::True | Tok::False | Tok::Nil => {
                let e = self.parse_primary()?;
                Ok(Pattern::Lit(e))
            }
            Tok::Self_ => { self.bump(); Ok(Pattern::Lit(Expr::SelfRef)) }
            other => Err(format!("{}: expected pattern, got {:?}", self.here(), other)),
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match self.peek().clone() {
            Tok::Push => {
                self.bump();
                let slot = self.ident()?;
                self.expect(Tok::LArrow)?;
                let value = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Push { slot, value })
            }
            Tok::Pop => {
                self.bump();
                let slot = self.ident()?;
                self.expect(Tok::Arrow)?;
                let into = self.ident()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Pop { slot, into })
            }
            Tok::Drop => {
                self.bump();
                let slot = self.ident()?;
                let n = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::DropN { slot, n })
            }
            Tok::Emit => {
                self.bump();
                let payload = self.parse_expr()?;
                let (meta_ops, rp_op) = self.parse_emit_modifiers()?;
                self.expect(Tok::To)?;
                let target = self.parse_emit_target()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Emit { payload, target, meta_ops, rp_op })
            }
            Tok::EmitEach => {
                self.bump();
                let payload = self.parse_expr()?;
                let (meta_ops, rp_op) = self.parse_emit_modifiers()?;
                self.expect(Tok::To)?;
                let targets = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::EmitEach { payload, targets, meta_ops, rp_op })
            }
            Tok::Record => {
                self.bump();
                let name = self.ident()?;
                let value = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Record { name, value })
            }
            Tok::Error => {
                self.bump();
                let kind = match self.bump() {
                    Tok::Str(s) => s,
                    other => return Err(format!(
                        "{}: error: expected string literal kind, got {:?}",
                        self.here(), other
                    )),
                };
                let detail = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Error { kind, detail })
            }
            Tok::Spawn => {
                self.bump();
                let template = self.ident()?;
                self.expect(Tok::Arrow)?;
                let into = self.ident()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Spawn { template, into })
            }
            Tok::Ident(_) => {
                // `name := expr ;`
                let name = self.ident()?;
                self.expect(Tok::Assign)?;
                let value = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::SlotSet { slot: name, value })
            }
            other => Err(format!("{}: expected statement, got {:?}", self.here(), other)),
        }
    }

    /// Parse zero-or-more metadata / return_path modifiers that sit
    /// between an emit payload and its `to` target. At most one of
    /// {pushing, popping, return_path <expr>} is accepted per emit
    /// — later ones overwrite earlier ones at this layer but the
    /// lowerer rejects it if you care (we just let the last one win).
    ///
    ///   meta { k: EXPR, k: EXPR }
    ///   forget_meta { IDENT, IDENT }
    ///   pushing EXPR
    ///   popping
    ///   return_path EXPR
    fn parse_emit_modifiers(&mut self) -> Result<(Vec<MetaOp>, ReturnPathOp), String> {
        let mut meta_ops: Vec<MetaOp> = Vec::new();
        let mut rp_op = ReturnPathOp::Inherit;
        loop {
            match self.peek() {
                Tok::Meta => {
                    self.bump();
                    self.expect(Tok::LBrace)?;
                    while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
                        let key = self.ident()?;
                        self.expect(Tok::Colon)?;
                        let value = self.parse_expr()?;
                        meta_ops.push(MetaOp::Set { key, value });
                        let _ = self.eat(&Tok::Comma) || self.eat(&Tok::Semi);
                    }
                    self.expect(Tok::RBrace)?;
                }
                Tok::ForgetMeta => {
                    self.bump();
                    self.expect(Tok::LBrace)?;
                    while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
                        let key = self.ident()?;
                        meta_ops.push(MetaOp::Remove { key });
                        let _ = self.eat(&Tok::Comma) || self.eat(&Tok::Semi);
                    }
                    self.expect(Tok::RBrace)?;
                }
                Tok::Pushing => {
                    self.bump();
                    let e = self.parse_expr()?;
                    rp_op = ReturnPathOp::Push(e);
                }
                Tok::Popping => {
                    self.bump();
                    rp_op = ReturnPathOp::Pop;
                }
                Tok::ReturnPath => {
                    // `return_path EXPR` — only valid here as a modifier
                    // when NOT immediately followed by a `(` (that would
                    // be a function-call-style misuse; return_path takes
                    // no args as an expression). Since `return_path` in
                    // expression position is a nullary keyword, this
                    // modifier form is always `return_path EXPR`.
                    self.bump();
                    let e = self.parse_expr()?;
                    rp_op = ReturnPathOp::Replace(e);
                }
                _ => break,
            }
        }
        Ok((meta_ops, rp_op))
    }

    fn parse_emit_target(&mut self) -> Result<EmitTarget, String> {
        match self.peek().clone() {
            Tok::Self_ => { self.bump(); Ok(EmitTarget::Self_) }
            Tok::Out => {
                self.bump();
                let p = self.ident()?;
                Ok(EmitTarget::OutPort(p))
            }
            Tok::Port => {
                self.bump();
                let p = self.ident()?;
                Ok(EmitTarget::FromPort(p))
            }
            Tok::Ident(s) if s == "default" => { self.bump(); Ok(EmitTarget::Default) }
            Tok::Ident(_) | Tok::IdentTpl(_) => {
                let n = self.name_tpl()?;
                Ok(EmitTarget::Target(n))
            }
            Tok::LParen => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(Tok::RParen)?;
                Ok(EmitTarget::Dynamic(e))
            }
            other => Err(format!("{}: expected emit target, got {:?}", self.here(), other)),
        }
    }

    fn parse_compound(&mut self) -> Result<Item, String> {
        self.expect(Tok::Compound)?;
        let name = self.ident()?;
        // Optional compile-time params: `compound NAME (p: Int = 5, q: Bool) { ... }`.
        let params = if self.eat(&Tok::LParen) {
            let mut ps = Vec::new();
            while !matches!(self.peek(), Tok::RParen | Tok::Eof) {
                ps.push(self.parse_tpl_param()?);
                // Trailing comma is fine.
                if !self.eat(&Tok::Comma) { break; }
            }
            self.expect(Tok::RParen)?;
            ps
        } else {
            Vec::new()
        };
        self.expect(Tok::LBrace)?;
        let mut items = Vec::new();
        let mut in_ports = Vec::new();
        let mut out_ports = Vec::new();
        loop {
            match self.peek() {
                Tok::In => { self.bump(); self.parse_port_map(&mut in_ports)?; }
                Tok::Out => { self.bump(); self.parse_port_map(&mut out_ports)?; }
                Tok::RBrace => break,
                // Inside a compound body, anything that's a top-level
                // item is also valid: nested nodes, edges, sub-compounds,
                // `for` loops, scenarios, params.
                Tok::Params | Tok::Node | Tok::Compound | Tok::Edges
                | Tok::Scenario | Tok::For => {
                    items.push(self.parse_item()?);
                }
                other => return Err(format!(
                    "{}: expected in/out or item inside compound, got {:?}",
                    self.here(), other
                )),
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Compound(CompoundDecl { name, params, items, in_ports, out_ports }))
    }

    fn parse_tpl_param(&mut self) -> Result<TplParam, String> {
        let name = self.ident()?;
        self.expect(Tok::Colon)?;
        let ty_name = self.ident()?;
        let ty = match ty_name.as_str() {
            "Int" => CtType::Int,
            "Bool" => CtType::Bool,
            "String" => CtType::Str,
            "Float" => CtType::Float,
            other => return Err(format!(
                "{}: compound param `{}`: type must be Int|Bool|String|Float, got `{}`",
                self.here(), name, other
            )),
        };
        let default = if self.eat(&Tok::Equals) || self.eat(&Tok::Assign) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        // Optional range hint: `… in LO..HI`. Reuses the same
        // expression grammar + `..` token as `for IDENT in LO..HI`,
        // and shares the same compile-time-only semantics — bounds
        // must evaluate to integers at expand time. Today the hint
        // only surfaces in the UI (slider min/max); the engine
        // doesn't enforce the bound, so a deliberate value outside
        // it is still allowed (acts as a soft hint, not a constraint).
        let range = if self.eat(&Tok::In) {
            let lo = self.parse_expr()?;
            self.expect(Tok::DotDot)?;
            let hi = self.parse_expr()?;
            Some((lo, hi))
        } else {
            None
        };
        Ok(TplParam { name, ty, default, range })
    }

    fn parse_port_map(&mut self, out: &mut Vec<PortDecl>) -> Result<(), String> {
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            let port = self.ident()?;
            self.expect(Tok::Colon)?;
            let inner = self.name_tpl()?;
            let _ = self.eat(&Tok::Semi) || self.eat(&Tok::Comma);
            out.push(PortDecl { port, inner });
        }
        self.expect(Tok::RBrace)?;
        Ok(())
    }

    fn parse_edges(&mut self) -> Result<Item, String> {
        self.expect(Tok::Edges)?;
        self.expect(Tok::LBrace)?;
        let mut out = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            out.push(self.parse_edge_body_item()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Edges(out))
    }

    /// Parse one item inside an `edges { }` block — either an edge
    /// declaration or a `for` loop whose body produces more such items.
    fn parse_edge_body_item(&mut self) -> Result<EdgeBodyItem, String> {
        if matches!(self.peek(), Tok::For) {
            let bindings = self.parse_for_bindings()?;
            self.expect(Tok::LBrace)?;
            let mut body = Vec::new();
            while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
                body.push(self.parse_edge_body_item()?);
            }
            self.expect(Tok::RBrace)?;
            Ok(EdgeBodyItem::For(EdgeFor { bindings, body }))
        } else {
            let from = self.parse_endpoint()?;
            self.expect(Tok::Arrow)?;
            let to = self.parse_endpoint()?;
            self.expect(Tok::Colon)?;
            let latency = self.parse_expr()?;
            let _ = self.eat(&Tok::Semi);
            Ok(EdgeBodyItem::Edge(EdgeDecl { from, to, latency }))
        }
    }

    fn parse_endpoint(&mut self) -> Result<EdgeEndpoint, String> {
        let node = self.name_tpl()?;
        let port = if self.eat(&Tok::Dot) { Some(self.ident()?) } else { None };
        Ok(EdgeEndpoint { node, port })
    }

    /// Parse `for IDENT in LO..HI [, IDENT in LO..HI ...]` — bindings only,
    /// caller handles the body brace block. Used by both `parse_for` (top
    /// level / compound body) and `parse_edge_body_item` (inside `edges`).
    fn parse_for_bindings(&mut self) -> Result<Vec<ForBinding>, String> {
        self.expect(Tok::For)?;
        let mut bs = Vec::new();
        bs.push(self.parse_one_for_binding()?);
        while self.eat(&Tok::Comma) {
            bs.push(self.parse_one_for_binding()?);
        }
        Ok(bs)
    }

    fn parse_one_for_binding(&mut self) -> Result<ForBinding, String> {
        let name = self.ident()?;
        // `in` is a keyword (reused from compound port maps).
        if !self.eat(&Tok::In) {
            return Err(format!("{}: expected `in` after `for {}`", self.here(), name));
        }
        let lo = self.parse_expr()?;
        self.expect(Tok::DotDot)?;
        let hi = self.parse_expr()?;
        Ok(ForBinding { name, lo, hi })
    }

    fn parse_for(&mut self) -> Result<Item, String> {
        let bindings = self.parse_for_bindings()?;
        self.expect(Tok::LBrace)?;
        let mut body = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            body.push(self.parse_item()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::For(ItemFor { bindings, body }))
    }

    fn parse_scenario(&mut self) -> Result<Item, String> {
        self.expect(Tok::Scenario)?;
        // Optional name: `scenario foo { … }` — named scenarios live in
        // the sim's library without auto-scheduling; `scenario { … }`
        // stays back-compatible and auto-schedules as "main".
        let name = if matches!(self.peek(), Tok::Ident(_)) {
            Some(self.ident()?)
        } else {
            None
        };
        self.expect(Tok::LBrace)?;
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            self.expect(Tok::At)?;
            let at_ns = match self.bump() {
                Tok::TimeNs(t) => t,
                Tok::Int(n) => n as u64,
                other => return Err(format!("{}: expected time literal, got {:?}", self.here(), other)),
            };
            self.expect(Tok::Colon)?;
            let action = self.parse_scene_action()?;
            let _ = self.eat(&Tok::Semi);
            stmts.push(SceneStmt { at_ns, action });
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Scenario(ScenarioDecl { name, stmts }))
    }

    fn parse_scene_action(&mut self) -> Result<SceneAction, String> {
        match self.peek().clone() {
            Tok::Inject => {
                self.bump();
                let node = self.ident()?;
                self.expect(Tok::LArrow)?;
                let tag = self.ident()?;
                let payload = if self.eat(&Tok::LParen) {
                    let e = if matches!(self.peek(), Tok::RParen) {
                        None
                    } else {
                        Some(self.parse_expr()?)
                    };
                    self.expect(Tok::RParen)?;
                    e
                } else { None };
                Ok(SceneAction::Inject { node, tag, payload })
            }
            Tok::SetParam => {
                self.bump();
                let name = self.ident()?;
                if !(self.eat(&Tok::Equals) || self.eat(&Tok::Assign)) {
                    return Err(format!("{}: expected `=` or `:=` after param name", self.here()));
                }
                let value = self.parse_expr()?;
                Ok(SceneAction::SetParam { name, value })
            }
            Tok::SetSlot => {
                self.bump();
                let node = self.ident()?;
                self.expect(Tok::Dot)?;
                let slot = self.ident()?;
                if !(self.eat(&Tok::Equals) || self.eat(&Tok::Assign)) {
                    return Err(format!("{}: expected `=` or `:=` after slot name", self.here()));
                }
                let value = self.parse_expr()?;
                Ok(SceneAction::SetSlot { node, slot, value })
            }
            Tok::Kill => {
                self.bump();
                let node = self.ident()?;
                Ok(SceneAction::Kill { node })
            }
            other => Err(format!("{}: expected scenario action, got {:?}", self.here(), other)),
        }
    }

    // -------- expressions (Pratt) --------

    fn parse_expr(&mut self) -> Result<Expr, String> { self.parse_or() }
    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Tok::OrOr) {
            self.bump();
            let rhs = self.parse_and()?;
            lhs = Expr::Binary(BinOp::Or, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_eq()?;
        while matches!(self.peek(), Tok::AndAnd) {
            self.bump();
            let rhs = self.parse_eq()?;
            lhs = Expr::Binary(BinOp::And, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn parse_eq(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_cmp()?;
        while matches!(self.peek(), Tok::Eq | Tok::NEq) {
            let op = if matches!(self.peek(), Tok::Eq) { BinOp::Eq } else { BinOp::NEq };
            self.bump();
            let rhs = self.parse_cmp()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn parse_cmp(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_add()?;
        while matches!(self.peek(), Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge) {
            let op = match self.peek() {
                Tok::Lt => BinOp::Lt, Tok::Le => BinOp::Le,
                Tok::Gt => BinOp::Gt, Tok::Ge => BinOp::Ge,
                _ => unreachable!(),
            };
            self.bump();
            let rhs = self.parse_add()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_mul()?;
        while matches!(self.peek(), Tok::Plus | Tok::Minus) {
            let op = if matches!(self.peek(), Tok::Plus) { BinOp::Add } else { BinOp::Sub };
            self.bump();
            let rhs = self.parse_mul()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_unary()?;
        while matches!(self.peek(), Tok::Star | Tok::Slash | Tok::Percent) {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul, Tok::Slash => BinOp::Div, Tok::Percent => BinOp::Mod,
                _ => unreachable!(),
            };
            self.bump();
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }
    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Tok::Minus => { self.bump(); Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?))) }
            Tok::Bang => { self.bump(); Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_unary()?))) }
            _ => self.parse_pow(),
        }
    }
    fn parse_pow(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_primary()?;
        if matches!(self.peek(), Tok::Caret) {
            self.bump();
            let rhs = self.parse_unary()?;       // right-assoc
            Ok(Expr::Binary(BinOp::Pow, Box::new(lhs), Box::new(rhs)))
        } else { Ok(lhs) }
    }
    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Tok::Int(n) => { self.bump(); Ok(Expr::Int(n)) }
            Tok::Float(f) => { self.bump(); Ok(Expr::Float(f)) }
            Tok::Str(s) => { self.bump(); Ok(Expr::Str(s)) }
            Tok::TimeNs(ns) => { self.bump(); Ok(Expr::Int(ns as i64)) }
            Tok::True => { self.bump(); Ok(Expr::Bool(true)) }
            Tok::False => { self.bump(); Ok(Expr::Bool(false)) }
            Tok::Nil => { self.bump(); Ok(Expr::Nil) }
            Tok::Self_ => { self.bump(); Ok(Expr::SelfRef) }
            Tok::ReturnPath => { self.bump(); Ok(Expr::ReturnPath) }
            Tok::Meta => {
                // Only valid in expression position as `meta("key")`.
                // Anywhere else, `meta` is consumed earlier by
                // parse_emit_modifiers as a modifier keyword.
                self.bump();
                self.expect(Tok::LParen)?;
                let key = match self.bump() {
                    Tok::Str(s) => s,
                    other => return Err(format!(
                        "{}: meta(): expected string literal key, got {:?}",
                        self.here(), other
                    )),
                };
                self.expect(Tok::RParen)?;
                Ok(Expr::Meta(key))
            }
            Tok::Ident(s) if s == "now" => { self.bump(); Ok(Expr::Now) }
            Tok::Ident(s) if s == "param" => {
                self.bump();
                self.expect(Tok::LParen)?;
                let n = self.ident()?;
                self.expect(Tok::RParen)?;
                Ok(Expr::Param(n))
            }
            Tok::Ident(s) if s == "if" => {
                self.bump();
                let c = self.parse_expr()?;
                // `then` keyword-ish — accept Ident("then") or just continue
                let _ = match self.peek() { Tok::Ident(x) if x == "then" => { self.bump(); true }, _ => false };
                let t = self.parse_expr()?;
                let _ = match self.peek() { Tok::Ident(x) if x == "else" => { self.bump(); true }, _ => false };
                let e = self.parse_expr()?;
                Ok(Expr::If { cond: Box::new(c), then_: Box::new(t), else_: Box::new(e) })
            }
            Tok::Ident(_) => {
                let name = self.ident()?;
                if matches!(self.peek(), Tok::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Tok::RParen) {
                        args.push(self.parse_expr()?);
                        while self.eat(&Tok::Comma) {
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(Tok::RParen)?;
                    // A small closed set of built-in function names.
                    // Anything else is a variant constructor — this way
                    // variant tags can be any case (e.g., lowercase `work`).
                    let is_builtin = matches!(
                        name.as_str(),
                        "Exp" | "Uniform" | "Bernoulli"
                        | "len" | "mean" | "count_where"
                        | "out_neighbors" | "slot_of"
                        | "length" | "index" | "filter" | "map" | "reduce" | "argmin"
                        | "head" | "tail"
                        | "edge_last_sent"
                    );
                    if is_builtin {
                        Ok(Expr::FnCall(name, args))
                    } else {
                        let payload = match args.len() {
                            0 => None,
                            1 => Some(Box::new(args.into_iter().next().unwrap())),
                            _ => return Err(format!("{}: variant `{}` takes 0 or 1 payload (got {})",
                                self.here(), name, args.len())),
                        };
                        Ok(Expr::Variant(name, payload))
                    }
                } else {
                    // Bare identifier: slot or variable (resolved in lowering).
                    Ok(Expr::Name(name))
                }
            }
            Tok::LParen => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(Tok::RParen)?;
                Ok(e)
            }
            other => Err(format!("{}: expected primary expression, got {:?}", self.here(), other)),
        }
    }
}
