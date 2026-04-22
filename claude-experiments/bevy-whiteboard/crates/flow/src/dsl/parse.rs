//! Recursive-descent parser for the Flow DSL.

use super::ast::*;
use super::lex::{Tok, Token};

pub fn parse(src: &str) -> Result<File, String> {
    let tokens = super::lex::lex(src)?;
    let mut p = Parser { toks: tokens, pos: 0 };
    p.parse_file()
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
        let name = self.ident()?;
        self.expect(Tok::LBrace)?;
        let mut slots = Vec::new();
        let mut rules = Vec::new();
        loop {
            match self.peek() {
                Tok::Slots => { slots = self.parse_slots()?; }
                Tok::Rule => { rules.push(self.parse_rule()?); }
                Tok::RBrace => break,
                other => return Err(format!("{}: expected slots/rule, got {:?}", self.here(), other)),
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Node(NodeDecl { name, slots, rules }))
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
                self.expect(Tok::To)?;
                let target = self.parse_emit_target()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Emit { payload, target })
            }
            Tok::EmitEach => {
                self.bump();
                let payload = self.parse_expr()?;
                self.expect(Tok::To)?;
                let targets = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::EmitEach { payload, targets })
            }
            Tok::Respond => {
                self.bump();
                let payload = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Respond { payload })
            }
            Tok::Record => {
                self.bump();
                let name = self.ident()?;
                let value = self.parse_expr()?;
                let _ = self.eat(&Tok::Semi);
                Ok(Stmt::Record { name, value })
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

    fn parse_emit_target(&mut self) -> Result<EmitTarget, String> {
        match self.peek().clone() {
            Tok::Self_ => { self.bump(); Ok(EmitTarget::Self_) }
            Tok::Out => {
                self.bump();
                let p = self.ident()?;
                Ok(EmitTarget::OutPort(p))
            }
            Tok::Ident(s) if s == "default" => { self.bump(); Ok(EmitTarget::Default) }
            Tok::Ident(_) => {
                let n = self.ident()?;
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
        self.expect(Tok::LBrace)?;
        let mut in_ports = Vec::new();
        let mut out_ports = Vec::new();
        loop {
            match self.peek() {
                Tok::In => { self.bump(); self.parse_port_map(&mut in_ports)?; }
                Tok::Out => { self.bump(); self.parse_port_map(&mut out_ports)?; }
                Tok::RBrace => break,
                other => return Err(format!("{}: expected in/out, got {:?}", self.here(), other)),
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Compound(CompoundDecl { name, in_ports, out_ports }))
    }

    fn parse_port_map(&mut self, out: &mut Vec<PortDecl>) -> Result<(), String> {
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace | Tok::Eof) {
            let port = self.ident()?;
            self.expect(Tok::Colon)?;
            let inner = self.ident()?;
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
            let from = self.parse_endpoint()?;
            self.expect(Tok::Arrow)?;
            let to = self.parse_endpoint()?;
            self.expect(Tok::Colon)?;
            let latency = self.parse_expr()?;
            let _ = self.eat(&Tok::Semi);
            out.push(EdgeDecl { from, to, latency });
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Edges(out))
    }

    fn parse_endpoint(&mut self) -> Result<EdgeEndpoint, String> {
        let node = self.ident()?;
        let port = if self.eat(&Tok::Dot) { Some(self.ident()?) } else { None };
        Ok(EdgeEndpoint { node, port })
    }

    fn parse_scenario(&mut self) -> Result<Item, String> {
        self.expect(Tok::Scenario)?;
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
        Ok(Item::Scenario(stmts))
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
                        | "len" | "mean"
                        | "out_neighbors" | "slot_of"
                        | "length" | "index" | "filter" | "map" | "reduce"
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
