//! Lower a parsed DSL `File` into a live `Sim`.
//!
//! Responsibilities:
//!   - Resolve context-dependent identifiers (slot vs variable vs param).
//!   - Build `Pattern`, `Rule`, `Effect`, `Expr` internal types.
//!   - Create nodes (leaf & compound), edges (with ports), scenarios,
//!     register params.
//!
//! Identifier resolution rule (simple and local):
//!   - Inside a rule body, an `Expr::Name(x)` resolves to:
//!       * a variable bound by this rule's patterns, if one exists;
//!       * otherwise to a slot of the current node.
//!   - `param(name)` and `self` and `now` are explicit keywords.
//!   - Outside rules (e.g. edge latency), `Name(x)` resolves to a slot
//!     of the source node at emit time (delegated to the engine's
//!     `Slot` evaluator — semantically identical).

use std::collections::{BTreeMap, HashSet};

use super::ast;
use crate::{
    expr::{BinOp as IbinOp, Expr as Ie},
    rule::{Effect as Ieff, EmitTo as Iem, Rule as Irule, When as Iwhen},
    samples::Samples,
    sim::{NodeId, Sim},
    value::{Pattern as Ipat, Value},
};

/// Lower a parsed DSL file into a new `Sim` seeded from `seed`.
pub fn lower(file: &ast::File, seed: u64) -> Result<Sim, String> {
    let mut sim = Sim::new(seed);

    // Phase 1: params + leaf/compound node shells (by name).
    //
    // We must create leaves before compounds that reference them, and
    // params before edges that reference them. Parse order in the file
    // is preserved for reproducibility.
    let mut name_to_id: std::collections::HashMap<String, NodeId> = std::collections::HashMap::new();
    let mut pending_compounds: Vec<&ast::CompoundDecl> = Vec::new();
    let mut pending_edges: Vec<&ast::EdgeDecl> = Vec::new();
    let mut pending_scenarios: Vec<&Vec<ast::SceneStmt>> = Vec::new();

    for item in &file.items {
        match item {
            ast::Item::Params(ps) => {
                for p in ps {
                    // Params can reference other params — lower expr with
                    // empty bound-vars scope.
                    let e = lower_expr(&p.value, &HashSet::new())?;
                    sim.set_param(p.name.clone(), e);
                }
            }
            ast::Item::Node(n) => {
                let (slots, rules) = build_leaf(n)?;
                let id = sim.add_node(n.name.clone(), slots, rules);
                name_to_id.insert(n.name.clone(), id);
            }
            ast::Item::Compound(c) => pending_compounds.push(c),
            ast::Item::Edges(es) => {
                for e in es { pending_edges.push(e); }
            }
            ast::Item::Scenario(s) => pending_scenarios.push(s),
        }
    }

    // Phase 2: compound nodes (needs leaf ids).
    for c in pending_compounds {
        let mut in_ports = BTreeMap::new();
        for p in &c.in_ports {
            let inner = name_to_id.get(&p.inner)
                .ok_or_else(|| format!("compound `{}`: unknown inner node `{}`", c.name, p.inner))?;
            in_ports.insert(p.port.clone(), *inner);
        }
        let mut out_ports = BTreeMap::new();
        for p in &c.out_ports {
            let inner = name_to_id.get(&p.inner)
                .ok_or_else(|| format!("compound `{}`: unknown inner node `{}`", c.name, p.inner))?;
            out_ports.insert(p.port.clone(), *inner);
        }
        let id = sim.add_compound(c.name.clone(), in_ports, out_ports);
        name_to_id.insert(c.name.clone(), id);
    }

    // Phase 3: edges.
    for e in pending_edges {
        let from = name_to_id.get(&e.from.node)
            .ok_or_else(|| format!("edge from unknown node `{}`", e.from.node))?;
        let to = name_to_id.get(&e.to.node)
            .ok_or_else(|| format!("edge to unknown node `{}`", e.to.node))?;
        let latency = lower_expr(&e.latency, &HashSet::new())?;
        sim.add_edge_ports(*from, e.from.port.clone(), *to, e.to.port.clone(), latency);
    }

    // Phase 4: scenarios.
    for stmts in pending_scenarios {
        for s in stmts {
            let action = lower_scene_action(&s.action, &name_to_id)?;
            sim.schedule_action(s.at_ns, action);
        }
    }

    Ok(sim)
}

fn build_leaf(n: &ast::NodeDecl) -> Result<(BTreeMap<String, Value>, Vec<Irule>), String> {
    // Slots
    let mut slots = BTreeMap::new();
    for s in &n.slots {
        let v = lower_slot_init(s)?;
        slots.insert(s.name.clone(), v);
    }
    // Rules
    let mut rules = Vec::new();
    for r in &n.rules {
        rules.push(lower_rule(r, &slots)?);
    }
    Ok((slots, rules))
}

fn lower_slot_init(s: &ast::SlotDecl) -> Result<Value, String> {
    // If slot is Samples(cap), init yields an empty Samples.
    // Otherwise init is a literal expression we evaluate at load time.
    if let ast::SlotType::Samples(cap) = s.ty {
        return Ok(Value::Samples(Samples::new(cap as usize)));
    }
    let e = s.init.as_ref().ok_or_else(|| format!("slot `{}` missing initializer", s.name))?;
    lower_literal(e)
}

/// Fold an AST expression into a concrete Value, accepting only
/// literal shapes. Used for slot initializers.
fn lower_literal(e: &ast::Expr) -> Result<Value, String> {
    match e {
        ast::Expr::Int(n) => Ok(Value::Int(*n)),
        ast::Expr::Float(f) => Ok(Value::Float(*f)),
        ast::Expr::Bool(b) => Ok(Value::Bool(*b)),
        ast::Expr::Str(s) => Ok(Value::Str(s.clone())),
        ast::Expr::Nil => Ok(Value::Nil),
        ast::Expr::Variant(tag, payload) => {
            let p = match payload {
                None => Value::Nil,
                Some(box_inner) => lower_literal(box_inner)?,
            };
            Ok(Value::variant(tag.clone(), p))
        }
        other => Err(format!("slot init must be a literal, got {:?}", other)),
    }
}

fn lower_rule(r: &ast::RuleDecl, _node_slots: &BTreeMap<String, Value>) -> Result<Irule, String> {
    // Collect variables bound by patterns (for identifier resolution inside body).
    let mut bound: HashSet<String> = HashSet::new();
    for p in &r.ons { collect_pattern_bindings(p, &mut bound); }

    let mut rule = Irule::new(r.name.clone());
    // Only the FIRST `on` is an Input pattern; subsequent `on`s are SlotMatch
    // patterns (convention: `on slotname pattern` but we use Input-first-only
    // to keep the DSL aligned with current substrate).
    // For simplicity: all `on` patterns become Input patterns (engine already
    // picks the single Input per rule). If a user writes multiple ons in one
    // rule, they want them all checked — flag as a DSL error for now.
    if r.ons.len() > 1 {
        return Err(format!("rule `{}`: multiple `on` patterns not supported yet", r.name));
    }
    for p in &r.ons {
        rule = rule.when(Iwhen::input(lower_pattern(p)?));
    }
    if let Some(g) = &r.when {
        rule = rule.guard(lower_expr(g, &bound)?);
    }
    for stmt in &r.body {
        rule = rule.do_(lower_stmt(stmt, &mut bound)?);
    }
    Ok(rule)
}

fn collect_pattern_bindings(p: &ast::Pattern, out: &mut HashSet<String>) {
    match p {
        ast::Pattern::Var(n) => { out.insert(n.clone()); }
        ast::Pattern::Variant(_, args) => {
            for a in args { collect_pattern_bindings(a, out); }
        }
        _ => {}
    }
}

fn lower_pattern(p: &ast::Pattern) -> Result<Ipat, String> {
    Ok(match p {
        ast::Pattern::Wild => Ipat::Wild,
        ast::Pattern::Var(n) => Ipat::var(n.clone()),
        ast::Pattern::Lit(e) => Ipat::Lit(lower_literal(e)?),
        ast::Pattern::Variant(tag, args) => {
            let inner = match args.len() {
                0 => Ipat::wild(),
                1 => lower_pattern(&args[0])?,
                _ => return Err(format!("variant pattern `{}`: only 0 or 1 inner pattern supported", tag)),
            };
            Ipat::variant(tag.clone(), inner)
        }
    })
}

fn lower_stmt(s: &ast::Stmt, bound: &mut HashSet<String>) -> Result<Ieff, String> {
    Ok(match s {
        ast::Stmt::SlotSet { slot, value } => Ieff::SetSlot {
            slot: slot.clone(),
            value: lower_expr(value, bound)?,
        },
        ast::Stmt::Push { slot, value } => Ieff::SamplesPush {
            slot: slot.clone(),
            value: lower_expr(value, bound)?,
        },
        ast::Stmt::Pop { slot, into } => {
            // Pop binds a variable — add it to `bound` so subsequent
            // statements in this rule can reference it by name.
            bound.insert(into.clone());
            Ieff::SamplesPopOldestInto { slot: slot.clone(), into_var: into.clone() }
        }
        ast::Stmt::DropN { slot, n } => Ieff::SamplesDropOldest {
            slot: slot.clone(),
            n: lower_expr(n, bound)?,
        },
        ast::Stmt::Emit { payload, target } => {
            let p = lower_expr(payload, bound)?;
            let to = match target {
                ast::EmitTarget::Default => Iem::DefaultOut,
                ast::EmitTarget::Self_ => Iem::ToTargetExpr(Ie::self_ref()),
                ast::EmitTarget::OutPort(name) => Iem::ToOutPort(name.clone()),
                ast::EmitTarget::Target(name) => Iem::ToTarget(name.clone()),
                ast::EmitTarget::Dynamic(e) => Iem::ToTargetExpr(lower_expr(e, bound)?),
            };
            Ieff::Emit { payload: p, to }
        }
        ast::Stmt::EmitEach { payload, targets } => Ieff::EmitToEach {
            payload: lower_expr(payload, bound)?,
            targets: lower_expr(targets, bound)?,
        },
        ast::Stmt::Respond { payload } => Ieff::Respond { payload: lower_expr(payload, bound)? },
        ast::Stmt::Record { name, value } => Ieff::RecordMetric {
            name: name.clone(),
            value: lower_expr(value, bound)?,
        },
        ast::Stmt::Spawn { template, into } => {
            bound.insert(into.clone());
            Ieff::Spawn { template: template.clone(), into_var: Some(into.clone()) }
        }
    })
}

fn lower_expr(e: &ast::Expr, bound: &HashSet<String>) -> Result<Ie, String> {
    Ok(match e {
        ast::Expr::Int(n) => Ie::int(*n),
        ast::Expr::Float(f) => Ie::float(*f),
        ast::Expr::Bool(b) => Ie::bool(*b),
        ast::Expr::Str(s) => Ie::lit(Value::Str(s.clone())),
        ast::Expr::Nil => Ie::lit(Value::Nil),
        ast::Expr::Now => Ie::now(),
        ast::Expr::SelfRef => Ie::self_ref(),
        ast::Expr::Name(n) => {
            if bound.contains(n) {
                Ie::var(n.clone())
            } else {
                // Assume a slot of the current node. If it's wrong, the
                // engine will panic at eval time with a clear message.
                Ie::slot(n.clone())
            }
        }
        ast::Expr::Param(n) => Ie::param(n.clone()),
        ast::Expr::Variant(tag, payload) => {
            let inner = match payload {
                None => Ie::lit(Value::Nil),
                Some(p) => lower_expr(p, bound)?,
            };
            Ie::variant(tag.clone(), inner)
        }
        ast::Expr::FnCall(name, args) => lower_fn_call(name, args, bound)?,
        ast::Expr::Binary(op, l, r) => {
            let l = lower_expr(l, bound)?;
            let r = lower_expr(r, bound)?;
            let op = match op {
                ast::BinOp::Add => IbinOp::Add, ast::BinOp::Sub => IbinOp::Sub,
                ast::BinOp::Mul => IbinOp::Mul, ast::BinOp::Div => IbinOp::Div,
                ast::BinOp::Mod => IbinOp::Mod, ast::BinOp::Pow => IbinOp::Pow,
                ast::BinOp::Eq => IbinOp::Eq, ast::BinOp::NEq => IbinOp::Neq,
                ast::BinOp::Lt => IbinOp::Lt, ast::BinOp::Le => IbinOp::Le,
                ast::BinOp::Gt => IbinOp::Gt, ast::BinOp::Ge => IbinOp::Ge,
                ast::BinOp::And => IbinOp::And, ast::BinOp::Or => IbinOp::Or,
            };
            Ie::BinOp(op, Box::new(l), Box::new(r))
        }
        ast::Expr::Unary(op, e) => {
            let v = lower_expr(e, bound)?;
            match op {
                ast::UnOp::Not => Ie::not(v),
                ast::UnOp::Neg => Ie::sub(Ie::int(0), v),
            }
        }
        ast::Expr::If { cond, then_, else_ } => Ie::if_(
            lower_expr(cond, bound)?,
            lower_expr(then_, bound)?,
            lower_expr(else_, bound)?,
        ),
    })
}

fn lower_fn_call(name: &str, args: &[ast::Expr], bound: &HashSet<String>) -> Result<Ie, String> {
    // Routing primitives that need scope-extending semantics for their
    // closure body — handled before bulk arg lowering.
    match (name, args.len()) {
        ("out_neighbors", 0) => return Ok(Ie::out_neighbors()),
        ("slot_of", 2) => {
            let node = lower_expr(&args[0], bound)?;
            let slot = expect_str_lit(&args[1], "slot_of")?;
            return Ok(Ie::slot_of(node, slot));
        }
        ("length", 1) => {
            return Ok(Ie::length(lower_expr(&args[0], bound)?));
        }
        ("index", 2) => {
            return Ok(Ie::index(
                lower_expr(&args[0], bound)?,
                lower_expr(&args[1], bound)?,
            ));
        }
        ("filter", 3) => {
            let list = lower_expr(&args[0], bound)?;
            let bind = expect_str_lit(&args[1], "filter")?;
            let mut inner = bound.clone();
            inner.insert(bind.clone());
            let pred = lower_expr(&args[2], &inner)?;
            return Ok(Ie::filter(list, bind, pred));
        }
        ("map", 3) => {
            let list = lower_expr(&args[0], bound)?;
            let bind = expect_str_lit(&args[1], "map")?;
            let mut inner = bound.clone();
            inner.insert(bind.clone());
            let body = lower_expr(&args[2], &inner)?;
            return Ok(Ie::map(list, bind, body));
        }
        ("reduce", 5) => {
            let list = lower_expr(&args[0], bound)?;
            let elt = expect_str_lit(&args[1], "reduce (elt bind)")?;
            let acc = expect_str_lit(&args[2], "reduce (acc bind)")?;
            // init evaluates in the outer scope (no per-element binding yet).
            let init = lower_expr(&args[3], bound)?;
            let mut inner = bound.clone();
            inner.insert(elt.clone());
            inner.insert(acc.clone());
            let body = lower_expr(&args[4], &inner)?;
            return Ok(Ie::reduce(list, elt, acc, init, body));
        }
        _ => {}
    }

    let lowered: Result<Vec<Ie>, String> = args.iter().map(|a| lower_expr(a, bound)).collect();
    let mut a = lowered?;
    match (name, a.len()) {
        ("Exp", 1)   => Ok(Ie::exp_dist(a.pop().unwrap())),
        ("Uniform", 2) => {
            let hi = a.pop().unwrap();
            let lo = a.pop().unwrap();
            Ok(Ie::uniform_int(lo, hi))
        }
        ("Bernoulli", 1) => Ok(Ie::bernoulli(a.pop().unwrap())),
        ("len", 1) => {
            // Expects slot name; already Name/Slot.
            if let Ie::Slot(s) = a.pop().unwrap() { Ok(Ie::samples_len(s)) }
            else { Err(format!("len(): expected slot name")) }
        }
        ("mean", 1) => {
            if let Ie::Slot(s) = a.pop().unwrap() { Ok(Ie::samples_mean(s)) }
            else { Err(format!("mean(): expected slot name")) }
        }
        _ => Err(format!("unknown function `{}` with {} arg(s)", name, args.len())),
    }
}

fn expect_str_lit(e: &ast::Expr, ctx: &str) -> Result<String, String> {
    match e {
        ast::Expr::Str(s) => Ok(s.clone()),
        other => Err(format!("{}: expected a string literal bind name, got {:?}", ctx, other)),
    }
}

fn lower_scene_action(
    a: &ast::SceneAction,
    name_to_id: &std::collections::HashMap<String, NodeId>,
) -> Result<crate::scenario::Action, String> {
    use crate::scenario::Action;
    match a {
        ast::SceneAction::Inject { node, tag, payload } => {
            let nid = *name_to_id.get(node).ok_or_else(|| format!("inject: no node `{}`", node))?;
            let inner = match payload {
                None => Value::Nil,
                Some(e) => lower_literal(e)?,
            };
            Ok(Action::Inject { node: nid, payload: Value::variant(tag.clone(), inner), reply_to: None })
        }
        ast::SceneAction::SetParam { name, value } => {
            let e = lower_expr(value, &HashSet::new())?;
            Ok(Action::SetParam { name: name.clone(), value: e })
        }
        ast::SceneAction::SetSlot { node, slot, value } => {
            let nid = *name_to_id.get(node).ok_or_else(|| format!("set_slot: no node `{}`", node))?;
            let v = lower_literal(value)?;
            Ok(Action::SetSlot { node: nid, slot: slot.clone(), value: v })
        }
        ast::SceneAction::Kill { node } => {
            let nid = *name_to_id.get(node).ok_or_else(|| format!("kill: no node `{}`", node))?;
            Ok(Action::KillNode { node: nid })
        }
    }
}
