//! Expansion pass: the **single, explicit construction phase** that
//! turns generative source (`for`, parametric `compound`, `Cell_{x}_{y}`
//! interpolations, nested compound bodies) into a residual `ast::File`
//! that contains none of those constructs.
//!
//! Pipeline contract:
//!
//! ```text
//! source.flow ──parse──▶ AST_raw ──EXPAND──▶ AST_residual ──lower──▶ Sim
//! ```
//!
//! - The expander runs **once**, deterministically, before lowering.
//! - It uses its own tiny compile-time evaluator (`ct_eval`) over a
//!   subset of `ast::Expr`. Slots, params, runtime functions, RNG —
//!   none of those exist here. A `Slot(_)` reference in a `for` bound is
//!   rejected at expansion time.
//! - The output is a fully-flat `ast::File` whose names are all
//!   `NameTpl::plain`, whose compounds have empty `params`/`items`, and
//!   that contains no `Item::For` / `EdgeBodyItem::For`. `lower::lower`
//!   asserts each of those properties.
//! - Encapsulation: a compound's inner items become top-level after
//!   expansion, but their names are prefixed with `OuterName::` so they
//!   don't collide with siblings or cross-contaminate other compounds.
//!
//! Expansion is closed-world: same input → byte-identical residual.
//! That property is what lets us golden-file the expanded source in
//! tests.

use std::collections::{BTreeMap, HashMap};

use super::ast::{
    self, CompoundDecl, EdgeBodyItem, EdgeDecl, EdgeEndpoint, EdgeFor, Expr, File, ForBinding,
    Item, ItemFor, NameTpl, NamePart, NodeDecl, OnSpawnStmt, Pattern, PortDecl, ProbeDecl,
    ProbePart, RuleDecl, ScenarioDecl, SceneAction, SceneStmt, SlotDecl, Stmt, TplParam,
    EmitTarget, MetaOp, ReturnPathOp, BinOp, UnOp,
};

/// Compile-time value. Deliberately tiny — only what makes sense for
/// loop bounds and name interpolation.
///
/// Public so callers (e.g. the bevy canvas's compound-param inspector)
/// can walk a parsed AST and surface compound defaults without
/// re-implementing the evaluator.
#[derive(Debug, Clone)]
pub enum CtValue {
    Int(i64),
    Bool(bool),
    Str(String),
}

impl CtValue {
    fn type_name(&self) -> &'static str {
        match self {
            CtValue::Int(_) => "Int",
            CtValue::Bool(_) => "Bool",
            CtValue::Str(_) => "String",
        }
    }
    fn as_int(&self) -> Result<i64, String> {
        match self {
            CtValue::Int(n) => Ok(*n),
            other => Err(format!("expected Int, got {}", other.type_name())),
        }
    }
    fn to_display_string(&self) -> String {
        match self {
            CtValue::Int(n) => n.to_string(),
            CtValue::Bool(b) => b.to_string(),
            CtValue::Str(s) => s.clone(),
        }
    }
}

type Env = HashMap<String, CtValue>;

/// Class-name rewrite map: keys are the unqualified class names that
/// were declared inside the *enclosing* compound; values are the
/// fully-qualified names they got after prefixing. Lookups in nested
/// scopes fall back to outer scopes (innermost wins). When an
/// `Instance` decl references a class, we look it up here first; if
/// nothing matches, the reference stays as-is and lowering resolves
/// it against the global template registry.
type ClassScope = HashMap<String, String>;

/// Surface a compound's authoring-time params with their default
/// values evaluated. Walks the parsed AST (no expansion needed) so
/// callers introspecting compound metadata don't pay the full
/// expansion cost.
///
/// Compound names are emitted with their fully-qualified prefix
/// (e.g. `"Outer::Inner"`) — same shape as what `expand` produces and
/// what the engine carries on `CompoundBody`. Params whose default
/// expression fails to evaluate as compile-time data are skipped
/// (their default isn't representable as a [`CtValue`]) — they'll
/// surface in the regular expansion error path when the compound is
/// instantiated.
pub fn collect_compound_params(file: &File) -> Vec<CompoundParamSummary> {
    let mut out = Vec::new();
    let env = Env::new();
    walk_compound_summaries(&file.items, &env, "", &mut out);
    out
}

/// Per-compound authoring metadata exposed by [`collect_compound_params`].
#[derive(Debug, Clone)]
pub struct CompoundParamSummary {
    /// Fully-qualified compound name, matching `CompoundBody.name` in
    /// the lowered Sim.
    pub name: String,
    pub params: Vec<CompoundParamEntry>,
}

#[derive(Debug, Clone)]
pub struct CompoundParamEntry {
    pub name: String,
    pub ty: ast::CtType,
    /// Evaluated default. `None` if the param has no default declared
    /// or the default failed CT evaluation. In either case the
    /// inspector should display the param without a current value
    /// rather than fabricate one.
    pub default: Option<CtValue>,
    /// `[lo, hi)` range hint (DSL `… in LO..HI`) with both bounds
    /// already evaluated. `None` = no hint authored; UI falls back to
    /// a heuristic derived from the current value.
    pub range: Option<(CtValue, CtValue)>,
}

fn walk_compound_summaries(
    items: &[Item],
    env: &Env,
    name_prefix: &str,
    out: &mut Vec<CompoundParamSummary>,
) {
    for item in items {
        match item {
            Item::Compound(c) => {
                let qualified = prefix_str(name_prefix, &c.name);
                let mut params = Vec::with_capacity(c.params.len());
                for p in &c.params {
                    let default = p
                        .default
                        .as_ref()
                        .and_then(|e| ct_eval(e, env).ok());
                    let range = p.range.as_ref().and_then(|(lo, hi)| {
                        let lo = ct_eval(lo, env).ok()?;
                        let hi = ct_eval(hi, env).ok()?;
                        Some((lo, hi))
                    });
                    params.push(CompoundParamEntry {
                        name: p.name.clone(),
                        ty: p.ty.clone(),
                        default,
                        range,
                    });
                }
                out.push(CompoundParamSummary { name: qualified.clone(), params });
                let inner_prefix = format!("{}::", qualified);
                walk_compound_summaries(&c.items, env, &inner_prefix, out);
            }
            Item::For(ItemFor { body, .. }) => {
                // For-bindings within a non-compound for don't change
                // scoping — recurse so nested compounds inside loops
                // are still surfaced. We pass the *same* env because
                // loop vars aren't bound to specific values at this
                // metadata-collection layer; they'd shadow params,
                // which is fine because we already evaluated defaults
                // before recursing.
                walk_compound_summaries(body, env, name_prefix, out);
            }
            _ => {}
        }
    }
}

/// Re-expand **just one named compound** with explicit param overrides
/// instead of evaluating its declared defaults. Returns the residual
/// items belonging to that compound's interior — the inner nodes,
/// edges, scenarios, and the port-shim itself, with names already
/// prefixed by `<compound_name>::`.
///
/// Used by the live-edit path: when the user drags a slider on a
/// compound param, the canvas demolishes the compound's current
/// interior in the sim, calls this function to produce the new
/// interior items, and lowers them back into the same sim. None of
/// the surrounding canvas (top-level nodes, scenarios, viewport, sim
/// time, snapshot) is touched.
///
/// Override semantics: `overrides` is `param_name → CtValue`. Any
/// param the user didn't override falls back to its declared default
/// (same evaluation rules as `expand`). If the named compound doesn't
/// exist, returns `Err`.
pub fn expand_compound_subtree(
    file: &File,
    compound_name: &str,
    overrides: &BTreeMap<String, CtValue>,
) -> Result<Vec<Item>, String> {
    let mut out = Vec::new();
    let env = Env::new();
    let scope = ClassScope::new();
    let found = expand_named_compound(
        &file.items,
        &env,
        &scope,
        "",
        compound_name,
        overrides,
        &mut out,
    )?;
    if !found {
        return Err(format!(
            "expand_compound_subtree: no compound named `{}` in source",
            compound_name
        ));
    }
    Ok(out)
}

/// Walk the tree looking for a `Item::Compound` whose qualified name
/// matches `target`. When found, run the same expansion logic
/// `expand_compound` uses but with the user-supplied overrides folded
/// into the compound's param env. Returns `true` iff the target was
/// matched (so the caller can surface a helpful error otherwise).
fn expand_named_compound(
    items: &[Item],
    env: &Env,
    parent_scope: &ClassScope,
    name_prefix: &str,
    target: &str,
    overrides: &BTreeMap<String, CtValue>,
    out: &mut Vec<Item>,
) -> Result<bool, String> {
    for item in items {
        match item {
            Item::Compound(c) => {
                let qualified = prefix_str(name_prefix, &c.name);
                if qualified == target {
                    return expand_compound_with_overrides(
                        c,
                        env,
                        parent_scope,
                        name_prefix,
                        overrides,
                        out,
                    ).map(|_| true);
                }
                // Recurse into nested compounds in case `target` is
                // something like `Outer::Inner` and we need to walk
                // through `Outer` first.
                let mut inner_env = env.clone();
                for p in &c.params {
                    let v = bind_compound_param(p, env, &c.name)?;
                    inner_env.insert(p.name.clone(), v);
                }
                let inner_prefix = format!("{}::", qualified);
                let mut inner_scope = parent_scope.clone();
                collect_class_names(&c.items, &inner_prefix, &mut inner_scope);
                if expand_named_compound(
                    &c.items, &inner_env, &inner_scope,
                    &inner_prefix, target, overrides, out,
                )? {
                    return Ok(true);
                }
            }
            Item::For(ItemFor { body, .. }) => {
                // For loops at the file level still have to be
                // walked — a nested compound could live inside.
                // We don't iterate the for here because we're only
                // looking for a *named* compound, not generating
                // anything; the compound declaration itself doesn't
                // depend on for-binding values (those affect its
                // body, which we re-expand from inside `expand_compound_with_overrides`).
                if expand_named_compound(
                    body, env, parent_scope, name_prefix,
                    target, overrides, out,
                )? {
                    return Ok(true);
                }
            }
            _ => {}
        }
    }
    Ok(false)
}

/// Same shape as `expand_compound`, but binds the compound's params
/// from `(declared default ⊕ overrides)` instead of just defaults.
/// Mismatched override types (e.g. user supplies a Bool for an Int
/// param) are rejected; missing override + missing default is
/// rejected with the same error `bind_compound_param` already
/// produces.
fn expand_compound_with_overrides(
    c: &CompoundDecl,
    env: &Env,
    parent_scope: &ClassScope,
    name_prefix: &str,
    overrides: &BTreeMap<String, CtValue>,
    out: &mut Vec<Item>,
) -> Result<(), String> {
    let qualified_name = prefix_str(name_prefix, &c.name);
    let mut inner_env = env.clone();
    for p in &c.params {
        let v = if let Some(ov) = overrides.get(&p.name) {
            // Type-check the override. We don't promote (e.g. Int → Float).
            match (&p.ty, ov) {
                (ast::CtType::Int, CtValue::Int(_))
                | (ast::CtType::Bool, CtValue::Bool(_))
                | (ast::CtType::Str, CtValue::Str(_)) => {}
                _ => return Err(format!(
                    "compound `{}` param `{}`: override type {} doesn't match declared type {:?}",
                    c.name, p.name, ov.type_name(), p.ty
                )),
            }
            ov.clone()
        } else {
            bind_compound_param(p, env, &c.name)?
        };
        inner_env.insert(p.name.clone(), v);
    }

    let inner_prefix = format!("{}::", qualified_name);
    let mut inner_scope = parent_scope.clone();
    collect_class_names(&c.items, &inner_prefix, &mut inner_scope);
    expand_items(&c.items, &inner_env, &inner_scope, &inner_prefix, out)?;

    // Emit the compound's port-shim too. Same logic as the regular
    // `expand_compound`, kept inline to avoid threading overrides into
    // the existing function (its callers don't need overrides).
    let in_ports = resolve_ports(&c.in_ports, &inner_env, &inner_prefix, &qualified_name)?;
    let out_ports = resolve_ports(&c.out_ports, &inner_env, &inner_prefix, &qualified_name)?;
    out.push(Item::Compound(CompoundDecl {
        name: qualified_name,
        params: Vec::new(),
        items: Vec::new(),
        in_ports,
        out_ports,
    }));
    Ok(())
}

/// Run the expansion pass on a parsed file. Returns a residual `File`
/// whose `items` are flat (no `For`, no nested compound bodies, no
/// unresolved `{...}` interpolations).
pub fn expand(file: &File) -> Result<File, String> {
    let mut out = Vec::new();
    let env = Env::new();
    let scope = ClassScope::new();
    expand_items(&file.items, &env, &scope, "", &mut out)?;
    Ok(File { items: out })
}

/// Recurse over a list of items in the given compile-time environment
/// and `name_prefix` (the chain of enclosing-compound names joined by
/// `::`, terminated with `::` when non-empty). Appends residual items
/// to `out`. `class_scope` carries the class-rewrite map from the
/// surrounding compound (empty at file scope).
fn expand_items(
    items: &[Item],
    env: &Env,
    class_scope: &ClassScope,
    name_prefix: &str,
    out: &mut Vec<Item>,
) -> Result<(), String> {
    for item in items {
        match item {
            Item::Params(ps) => {
                let mut rewritten = Vec::with_capacity(ps.len());
                for p in ps {
                    rewritten.push(ast::ParamDecl {
                        name: p.name.clone(),
                        value: rewrite_expr_names(&p.value, env, name_prefix)?,
                    });
                }
                out.push(Item::Params(rewritten));
            }
            Item::Node(n) => out.push(Item::Node(expand_node(n, env, name_prefix)?)),
            Item::Instance(inst) => {
                // Instance names get template-resolved + prefixed;
                // class names look up via `class_scope` first
                // (compound-local class wins), then fall through to
                // the global template registry. Override expressions
                // are rewritten so they can reference compile-time
                // names (e.g. `node Cell_{x}_{y} : LifeCell { alive: x % 2 }`).
                let resolved = resolve_name(&inst.name, env, name_prefix)
                    .map_err(|e| format!("instance decl: {}", e))?;
                let class = class_scope.get(&inst.class).cloned().unwrap_or_else(||
                    inst.class.clone()
                );
                let mut overrides = Vec::with_capacity(inst.overrides.len());
                for (slot, e) in &inst.overrides {
                    overrides.push((slot.clone(), rewrite_expr_names(e, env, name_prefix)?));
                }
                out.push(Item::Instance(ast::InstanceDecl {
                    name: NameTpl::plain(resolved),
                    class,
                    overrides,
                }));
            }
            Item::Compound(c) => expand_compound(c, env, class_scope, name_prefix, out)?,
            Item::Edges(es) => {
                let mut residual: Vec<EdgeBodyItem> = Vec::new();
                expand_edge_items(es, env, name_prefix, &mut residual)?;
                out.push(Item::Edges(residual));
            }
            Item::Scenario(s) => out.push(Item::Scenario(expand_scenario(s, env, name_prefix)?)),
            Item::For(ItemFor { bindings, body }) => {
                iter_bindings(bindings, env, &mut |inner_env| {
                    expand_items(body, inner_env, class_scope, name_prefix, out)
                })?;
            }
        }
    }
    Ok(())
}

/// Walk the cartesian product of every for-binding's range, calling
/// `f(env)` for each tuple with the bindings inserted into `env`.
fn iter_bindings<F>(
    bindings: &[ForBinding],
    base_env: &Env,
    f: &mut F,
) -> Result<(), String>
where
    F: FnMut(&Env) -> Result<(), String>,
{
    fn rec<F>(
        bindings: &[ForBinding],
        idx: usize,
        env: &mut Env,
        f: &mut F,
    ) -> Result<(), String>
    where
        F: FnMut(&Env) -> Result<(), String>,
    {
        if idx == bindings.len() {
            return f(env);
        }
        let b = &bindings[idx];
        let lo = ct_eval(&b.lo, env)?.as_int()
            .map_err(|e| format!("for binding `{}`: lo: {}", b.name, e))?;
        let hi = ct_eval(&b.hi, env)?.as_int()
            .map_err(|e| format!("for binding `{}`: hi: {}", b.name, e))?;
        for i in lo..hi {
            let prev = env.insert(b.name.clone(), CtValue::Int(i));
            rec(bindings, idx + 1, env, f)?;
            match prev {
                Some(v) => { env.insert(b.name.clone(), v); }
                None => { env.remove(&b.name); }
            }
        }
        Ok(())
    }
    let mut env = base_env.clone();
    rec(bindings, 0, &mut env, f)
}

/// Expand a compound declaration. Two cases:
///
/// 1. Empty `items` and empty `params` — legacy port-rename shim.
///    Pass through unchanged after resolving inner-port names.
/// 2. Otherwise — a "rich" compound: bind defaults for any params
///    (compound declarations as singletons must use defaults; reusable
///    template instantiation arrives later as a follow-up), prefix all
///    inner items with `compound_name::`, recurse to expand them, then
///    emit (a) the residual inner items at top-level and (b) a
///    port-shim `Compound` referencing the prefixed inner names.
fn expand_compound(
    c: &CompoundDecl,
    env: &Env,
    parent_scope: &ClassScope,
    name_prefix: &str,
    out: &mut Vec<Item>,
) -> Result<(), String> {
    let qualified_name = prefix_str(name_prefix, &c.name);

    if c.params.is_empty() && c.items.is_empty() {
        // Legacy shim. Inner port names are resolved against the
        // *outer* environment / prefix — they refer to siblings the
        // user already declared at top level.
        let in_ports = resolve_ports(&c.in_ports, env, name_prefix, &qualified_name)?;
        let out_ports = resolve_ports(&c.out_ports, env, name_prefix, &qualified_name)?;
        out.push(Item::Compound(CompoundDecl {
            name: qualified_name,
            params: Vec::new(),
            items: Vec::new(),
            in_ports,
            out_ports,
        }));
        return Ok(());
    }

    // Bind compound params from defaults (singleton form). Reusable
    // template instantiation is a documented follow-up.
    let mut inner_env = env.clone();
    for p in &c.params {
        let v = bind_compound_param(p, env, &c.name)?;
        inner_env.insert(p.name.clone(), v);
    }

    let inner_prefix = format!("{}::", qualified_name);

    // Build the class scope for items declared *directly* inside this
    // compound. Walks `for` bodies to find `node` decls but does not
    // descend into nested compounds — those manage their own scope.
    // Each `node` declared inside the compound becomes a class scoped
    // to it; instance refs inside the compound rewrite to the
    // qualified class name.
    let mut inner_scope = parent_scope.clone();
    collect_class_names(&c.items, &inner_prefix, &mut inner_scope);

    expand_items(&c.items, &inner_env, &inner_scope, &inner_prefix, out)?;

    // Inner port names resolve against the *inner* environment (the
    // params and bindings of the compound) and the *inner* prefix
    // (so `Cell_0_0` inside `compound Life` becomes `Life::Cell_0_0`).
    let in_ports = resolve_ports(&c.in_ports, &inner_env, &inner_prefix, &qualified_name)?;
    let out_ports = resolve_ports(&c.out_ports, &inner_env, &inner_prefix, &qualified_name)?;
    out.push(Item::Compound(CompoundDecl {
        name: qualified_name,
        params: Vec::new(),
        items: Vec::new(),
        in_ports,
        out_ports,
    }));
    Ok(())
}

/// Walk items looking for `Item::Node` decls (the class-defining form).
/// Each one's unqualified name maps to its prefixed qualified name in
/// `out_scope`. Recurses through `Item::For` (those are still inside
/// the same compound's lexical scope after expansion) but stops at
/// `Item::Compound` boundaries — sub-compounds own their own scope.
fn collect_class_names(items: &[Item], inner_prefix: &str, out_scope: &mut ClassScope) {
    for item in items {
        match item {
            Item::Node(n) => {
                // Class definition uses the literal-prefix part of the
                // node's name template. If the user wrote a templated
                // class name like `node Type_{x}` they get one class
                // per loop iteration with a distinct qualified name —
                // we conservatively map only the no-hole prefix here.
                if let Some(plain) = n.name.as_plain() {
                    out_scope.insert(plain.to_string(), format!("{}{}", inner_prefix, plain));
                }
            }
            Item::For(ItemFor { body, .. }) => collect_class_names(body, inner_prefix, out_scope),
            _ => {}
        }
    }
}

fn bind_compound_param(p: &TplParam, env: &Env, owner: &str) -> Result<CtValue, String> {
    let default = p.default.as_ref().ok_or_else(|| format!(
        "compound `{}` param `{}` has no default and no instantiation argument was provided \
         (parametric instantiation is a planned follow-up)",
        owner, p.name
    ))?;
    let v = ct_eval(default, env)?;
    let bad = || format!(
        "compound `{}` param `{}`: default value type {} doesn't match declared type {:?}",
        owner, p.name, v.type_name(), p.ty
    );
    match (&p.ty, &v) {
        (ast::CtType::Int, CtValue::Int(_)) => {}
        (ast::CtType::Bool, CtValue::Bool(_)) => {}
        (ast::CtType::Str, CtValue::Str(_)) => {}
        _ => return Err(bad()),
    }
    Ok(v)
}

fn resolve_ports(
    ports: &[PortDecl],
    env: &Env,
    name_prefix: &str,
    owner: &str,
) -> Result<Vec<PortDecl>, String> {
    let mut out = Vec::with_capacity(ports.len());
    for p in ports {
        let inner = resolve_name(&p.inner, env, name_prefix)
            .map_err(|e| format!("compound `{}` port `{}`: {}", owner, p.port, e))?;
        out.push(PortDecl { port: p.port.clone(), inner: NameTpl::plain(inner) });
    }
    Ok(out)
}

fn expand_node(n: &NodeDecl, env: &Env, name_prefix: &str) -> Result<NodeDecl, String> {
    let resolved_name = resolve_name(&n.name, env, name_prefix)
        .map_err(|e| format!("node decl: {}", e))?;
    Ok(NodeDecl {
        name: NameTpl::plain(resolved_name),
        slots: n.slots.iter().map(|s| expand_slot(s, env, name_prefix)).collect::<Result<_, _>>()?,
        rules: n.rules.iter().map(|r| expand_rule(r, env, name_prefix)).collect::<Result<_, _>>()?,
        on_spawn: n.on_spawn.iter().map(|s| expand_on_spawn(s, env, name_prefix)).collect::<Result<_, _>>()?,
        probes: n.probes.iter().map(|p| expand_probe(p, env, name_prefix)).collect::<Result<_, _>>()?,
    })
}

fn expand_slot(s: &SlotDecl, env: &Env, name_prefix: &str) -> Result<SlotDecl, String> {
    Ok(SlotDecl {
        name: s.name.clone(),
        ty: s.ty.clone(),
        init: s.init.as_ref().map(|e| rewrite_expr_names(e, env, name_prefix)).transpose()?,
    })
}

fn expand_on_spawn(s: &OnSpawnStmt, env: &Env, name_prefix: &str) -> Result<OnSpawnStmt, String> {
    Ok(match s {
        OnSpawnStmt::SelfEdge { latency } => OnSpawnStmt::SelfEdge {
            latency: rewrite_expr_names(latency, env, name_prefix)?,
        },
        OnSpawnStmt::Inject { tag, payload } => OnSpawnStmt::Inject {
            tag: tag.clone(),
            payload: payload.as_ref().map(|p| rewrite_expr_names(p, env, name_prefix)).transpose()?,
        },
    })
}

fn expand_probe(p: &ProbeDecl, env: &Env, name_prefix: &str) -> Result<ProbeDecl, String> {
    let mut parts = Vec::with_capacity(p.parts.len());
    for part in &p.parts {
        parts.push(match part {
            ProbePart::Literal(s) => ProbePart::Literal(s.clone()),
            ProbePart::Hole(e) => ProbePart::Hole(rewrite_expr_names(e, env, name_prefix)?),
        });
    }
    Ok(ProbeDecl { label: p.label.clone(), parts })
}

fn expand_rule(r: &RuleDecl, env: &Env, name_prefix: &str) -> Result<RuleDecl, String> {
    Ok(RuleDecl {
        name: r.name.clone(),
        ons: r.ons.iter().map(|p| expand_pattern(p, env, name_prefix)).collect::<Result<_, _>>()?,
        when: r.when.as_ref().map(|e| rewrite_expr_names(e, env, name_prefix)).transpose()?,
        body: r.body.iter().map(|s| expand_stmt(s, env, name_prefix)).collect::<Result<_, _>>()?,
    })
}

fn expand_pattern(p: &Pattern, env: &Env, name_prefix: &str) -> Result<Pattern, String> {
    Ok(match p {
        Pattern::Wild | Pattern::Var(_) => p.clone(),
        Pattern::Lit(e) => Pattern::Lit(rewrite_expr_names(e, env, name_prefix)?),
        Pattern::Variant(tag, args) => Pattern::Variant(
            tag.clone(),
            args.iter().map(|a| expand_pattern(a, env, name_prefix)).collect::<Result<_, _>>()?,
        ),
    })
}

fn expand_stmt(s: &Stmt, env: &Env, name_prefix: &str) -> Result<Stmt, String> {
    Ok(match s {
        Stmt::SlotSet { slot, value } => Stmt::SlotSet {
            slot: slot.clone(),
            value: rewrite_expr_names(value, env, name_prefix)?,
        },
        Stmt::Push { slot, value } => Stmt::Push {
            slot: slot.clone(),
            value: rewrite_expr_names(value, env, name_prefix)?,
        },
        Stmt::Pop { slot, into } => Stmt::Pop { slot: slot.clone(), into: into.clone() },
        Stmt::DropN { slot, n } => Stmt::DropN {
            slot: slot.clone(),
            n: rewrite_expr_names(n, env, name_prefix)?,
        },
        Stmt::Emit { payload, target, meta_ops, rp_op } => Stmt::Emit {
            payload: rewrite_expr_names(payload, env, name_prefix)?,
            target: expand_emit_target(target, env, name_prefix)?,
            meta_ops: meta_ops.iter().map(|m| expand_meta_op(m, env, name_prefix)).collect::<Result<_, _>>()?,
            rp_op: expand_rp_op(rp_op, env, name_prefix)?,
        },
        Stmt::EmitEach { payload, targets, meta_ops, rp_op } => Stmt::EmitEach {
            payload: rewrite_expr_names(payload, env, name_prefix)?,
            targets: rewrite_expr_names(targets, env, name_prefix)?,
            meta_ops: meta_ops.iter().map(|m| expand_meta_op(m, env, name_prefix)).collect::<Result<_, _>>()?,
            rp_op: expand_rp_op(rp_op, env, name_prefix)?,
        },
        Stmt::Record { name, value } => Stmt::Record {
            name: name.clone(),
            value: rewrite_expr_names(value, env, name_prefix)?,
        },
        Stmt::Spawn { template, into } => Stmt::Spawn { template: template.clone(), into: into.clone() },
        Stmt::Error { kind, detail } => Stmt::Error {
            kind: kind.clone(),
            detail: rewrite_expr_names(detail, env, name_prefix)?,
        },
    })
}

fn expand_emit_target(t: &EmitTarget, env: &Env, name_prefix: &str) -> Result<EmitTarget, String> {
    Ok(match t {
        EmitTarget::Default => EmitTarget::Default,
        EmitTarget::Self_ => EmitTarget::Self_,
        EmitTarget::OutPort(s) => EmitTarget::OutPort(s.clone()),
        EmitTarget::FromPort(s) => EmitTarget::FromPort(s.clone()),
        EmitTarget::Target(name) => {
            let resolved = resolve_name(name, env, name_prefix)
                .map_err(|e| format!("emit target: {}", e))?;
            EmitTarget::Target(NameTpl::plain(resolved))
        }
        EmitTarget::Dynamic(e) => EmitTarget::Dynamic(rewrite_expr_names(e, env, name_prefix)?),
    })
}

fn expand_meta_op(m: &MetaOp, env: &Env, name_prefix: &str) -> Result<MetaOp, String> {
    Ok(match m {
        MetaOp::Set { key, value } => MetaOp::Set {
            key: key.clone(),
            value: rewrite_expr_names(value, env, name_prefix)?,
        },
        MetaOp::Remove { key } => MetaOp::Remove { key: key.clone() },
    })
}

fn expand_rp_op(r: &ReturnPathOp, env: &Env, name_prefix: &str) -> Result<ReturnPathOp, String> {
    Ok(match r {
        ReturnPathOp::Inherit => ReturnPathOp::Inherit,
        ReturnPathOp::Push(e) => ReturnPathOp::Push(rewrite_expr_names(e, env, name_prefix)?),
        ReturnPathOp::Pop => ReturnPathOp::Pop,
        ReturnPathOp::Replace(e) => ReturnPathOp::Replace(rewrite_expr_names(e, env, name_prefix)?),
    })
}

fn expand_edge_items(
    items: &[EdgeBodyItem],
    env: &Env,
    name_prefix: &str,
    out: &mut Vec<EdgeBodyItem>,
) -> Result<(), String> {
    for it in items {
        match it {
            EdgeBodyItem::Edge(d) => out.push(EdgeBodyItem::Edge(expand_edge(d, env, name_prefix)?)),
            EdgeBodyItem::For(EdgeFor { bindings, body }) => {
                iter_bindings(bindings, env, &mut |inner_env| {
                    expand_edge_items(body, inner_env, name_prefix, out)
                })?;
            }
        }
    }
    Ok(())
}

fn expand_edge(d: &EdgeDecl, env: &Env, name_prefix: &str) -> Result<EdgeDecl, String> {
    let from_name = resolve_name(&d.from.node, env, name_prefix)
        .map_err(|e| format!("edge from: {}", e))?;
    let to_name = resolve_name(&d.to.node, env, name_prefix)
        .map_err(|e| format!("edge to: {}", e))?;
    Ok(EdgeDecl {
        from: EdgeEndpoint { node: NameTpl::plain(from_name), port: d.from.port.clone() },
        to: EdgeEndpoint { node: NameTpl::plain(to_name), port: d.to.port.clone() },
        latency: rewrite_expr_names(&d.latency, env, name_prefix)?,
    })
}

fn expand_scenario(s: &ScenarioDecl, env: &Env, name_prefix: &str) -> Result<ScenarioDecl, String> {
    let mut stmts = Vec::with_capacity(s.stmts.len());
    for st in &s.stmts {
        stmts.push(SceneStmt {
            at_ns: st.at_ns,
            action: expand_scene_action(&st.action, env, name_prefix)?,
        });
    }
    Ok(ScenarioDecl { name: s.name.clone(), stmts })
}

fn expand_scene_action(a: &SceneAction, env: &Env, name_prefix: &str) -> Result<SceneAction, String> {
    Ok(match a {
        SceneAction::Inject { node, tag, payload } => SceneAction::Inject {
            node: prefix_str(name_prefix, node),
            tag: tag.clone(),
            payload: payload.as_ref().map(|p| rewrite_expr_names(p, env, name_prefix)).transpose()?,
        },
        SceneAction::SetParam { name, value } => SceneAction::SetParam {
            name: name.clone(),
            value: rewrite_expr_names(value, env, name_prefix)?,
        },
        SceneAction::SetSlot { node, slot, value } => SceneAction::SetSlot {
            node: prefix_str(name_prefix, node),
            slot: slot.clone(),
            value: rewrite_expr_names(value, env, name_prefix)?,
        },
        SceneAction::Kill { node } => SceneAction::Kill { node: prefix_str(name_prefix, node) },
    })
}

/// Apply the current `name_prefix` to a plain string. Used for items
/// where we don't carry a `NameTpl` (instance names, scene actions).
fn prefix_str(prefix: &str, name: &str) -> String {
    if prefix.is_empty() { name.to_string() } else { format!("{}{}", prefix, name) }
}

/// Resolve a `NameTpl` to a fully-qualified plain string.
///
/// Each `Hole(Expr)` is evaluated against `env` via `ct_eval` and
/// stringified; literals concatenate. The result is then prefixed with
/// `name_prefix` (joined by `::` already at the prefix's tail). A
/// fully literal `NameTpl::plain(s)` just returns `prefix + s`.
fn resolve_name(t: &NameTpl, env: &Env, name_prefix: &str) -> Result<String, String> {
    let mut s = String::new();
    for part in &t.parts {
        match part {
            NamePart::Literal(lit) => s.push_str(lit),
            NamePart::Hole(e) => s.push_str(&ct_eval(e, env)?.to_display_string()),
        }
    }
    Ok(prefix_str(name_prefix, &s))
}

/// Walk a runtime `ast::Expr`, substituting any `Name(x)` reference
/// where `x` is bound in the compile-time environment (a `for` binding
/// or a compound param) with the corresponding literal, and folding
/// constant subtrees (`Binary` / `Unary` / `If`) whose operands have
/// all reduced to literals.
///
/// Names that *aren't* in the CT environment are left intact — the
/// lowering pass will resolve them as slots, locals, or params at
/// engine load time. So the cell rule body `alive := alive + 1` keeps
/// `alive` as a slot reference, while `period_ns + 1` (where
/// `period_ns` is a CT param) collapses to a single Int literal.
///
/// Folding matters because slot initializers and instance overrides
/// must lower to literals via `lower_literal`. Without folding,
/// `if x == width / 2 && y == height / 2 then 1 else 0` would still
/// be an `If` expression after substitution and lowering would reject
/// it.
///
/// Note: if the user accidentally shadows a slot name with a CT name,
/// the CT name wins inside the compound's expansion. That's a
/// documented hazard; pick distinct names.
fn rewrite_expr_names(e: &Expr, env: &Env, name_prefix: &str) -> Result<Expr, String> {
    Ok(match e {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Nil
        | Expr::Now | Expr::SelfRef | Expr::Meta(_) | Expr::ReturnPath
        | Expr::Param(_) => e.clone(),
        Expr::Name(n) => {
            if let Some(v) = env.get(n) {
                ct_value_to_lit_expr(v)
            } else {
                Expr::Name(n.clone())
            }
        }
        Expr::Variant(tag, payload) => Expr::Variant(
            tag.clone(),
            match payload {
                None => None,
                Some(b) => Some(Box::new(rewrite_expr_names(b, env, name_prefix)?)),
            },
        ),
        Expr::FnCall(name, args) => Expr::FnCall(
            name.clone(),
            args.iter().map(|a| rewrite_expr_names(a, env, name_prefix)).collect::<Result<_, _>>()?,
        ),
        Expr::Binary(op, l, r) => fold_binary(
            *op,
            rewrite_expr_names(l, env, name_prefix)?,
            rewrite_expr_names(r, env, name_prefix)?,
        )?,
        Expr::Unary(op, x) => fold_unary(*op, rewrite_expr_names(x, env, name_prefix)?),
        Expr::If { cond, then_, else_ } => {
            let c = rewrite_expr_names(cond, env, name_prefix)?;
            let t = rewrite_expr_names(then_, env, name_prefix)?;
            let e = rewrite_expr_names(else_, env, name_prefix)?;
            match c {
                Expr::Bool(true) => t,
                Expr::Bool(false) => e,
                other => Expr::If {
                    cond: Box::new(other),
                    then_: Box::new(t),
                    else_: Box::new(e),
                },
            }
        }
    })
}

fn ct_value_to_lit_expr(v: &CtValue) -> Expr {
    match v {
        CtValue::Int(n) => Expr::Int(*n),
        CtValue::Bool(b) => Expr::Bool(*b),
        CtValue::Str(s) => Expr::Str(s.clone()),
    }
}

/// Constant-fold a binary op with two literal operands. Falls through
/// to a fresh `Binary` node if either side isn't a literal of a
/// foldable shape.
fn fold_binary(op: BinOp, l: Expr, r: Expr) -> Result<Expr, String> {
    let lv = expr_to_ct(&l);
    let rv = expr_to_ct(&r);
    if let (Some(lv), Some(rv)) = (lv, rv) {
        match ct_binop(op, lv, rv) {
            Ok(v) => return Ok(ct_value_to_lit_expr(&v)),
            Err(_) => {} // fall through to non-folded form
        }
    }
    Ok(Expr::Binary(op, Box::new(l), Box::new(r)))
}

fn fold_unary(op: UnOp, x: Expr) -> Expr {
    match (op, &x) {
        (UnOp::Neg, Expr::Int(n)) => Expr::Int(-n),
        (UnOp::Not, Expr::Bool(b)) => Expr::Bool(!b),
        _ => Expr::Unary(op, Box::new(x)),
    }
}

fn expr_to_ct(e: &Expr) -> Option<CtValue> {
    match e {
        Expr::Int(n) => Some(CtValue::Int(*n)),
        Expr::Bool(b) => Some(CtValue::Bool(*b)),
        Expr::Str(s) => Some(CtValue::Str(s.clone())),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// Compile-time evaluator
//
// Accepts a strict subset of `ast::Expr`. Anything outside the subset
// (slots, params, runtime functions, distributions) is rejected with a
// clear "not allowed at compile time" error, which is the contract we
// promised the user — `Slot(_)` in a `for` bound surfaces here, not at
// lowering time.

fn ct_eval(e: &Expr, env: &Env) -> Result<CtValue, String> {
    match e {
        Expr::Int(n) => Ok(CtValue::Int(*n)),
        Expr::Float(_) => Err("compile-time eval: floats are not supported".to_string()),
        Expr::Bool(b) => Ok(CtValue::Bool(*b)),
        Expr::Str(s) => Ok(CtValue::Str(s.clone())),
        Expr::Nil => Err("compile-time eval: `nil` is not a compile-time value".to_string()),
        Expr::Now | Expr::SelfRef
        | Expr::Meta(_) | Expr::ReturnPath
        | Expr::Param(_) | Expr::Variant(_, _) | Expr::FnCall(_, _) =>
            Err(format!(
                "compile-time eval: `{}` is a runtime construct and cannot appear in a `for` \
                 bound, name interpolation, or compound param default",
                ct_kind(e)
            )),
        Expr::Name(n) => env.get(n).cloned().ok_or_else(||
            format!("compile-time eval: unknown name `{}` (only `for` bindings and compound \
                     params are in scope)", n)),
        Expr::Binary(op, l, r) => {
            let lv = ct_eval(l, env)?;
            let rv = ct_eval(r, env)?;
            ct_binop(*op, lv, rv)
        }
        Expr::Unary(op, x) => {
            let v = ct_eval(x, env)?;
            match (op, v) {
                (UnOp::Neg, CtValue::Int(n)) => Ok(CtValue::Int(-n)),
                (UnOp::Not, CtValue::Bool(b)) => Ok(CtValue::Bool(!b)),
                (op, v) => Err(format!("compile-time eval: unary {:?} on {}", op, v.type_name())),
            }
        }
        Expr::If { cond, then_, else_ } => {
            let c = ct_eval(cond, env)?;
            match c {
                CtValue::Bool(true) => ct_eval(then_, env),
                CtValue::Bool(false) => ct_eval(else_, env),
                other => Err(format!("compile-time eval: `if` condition is {}, expected Bool", other.type_name())),
            }
        }
    }
}

fn ct_kind(e: &Expr) -> &'static str {
    match e {
        Expr::Now => "now",
        Expr::SelfRef => "self",
        Expr::Meta(_) => "meta(...)",
        Expr::ReturnPath => "return_path",
        Expr::Param(_) => "param(...)",
        Expr::Variant(_, _) => "variant constructor",
        Expr::FnCall(_, _) => "function call",
        Expr::Nil => "nil",
        _ => "expr",
    }
}

fn ct_binop(op: BinOp, l: CtValue, r: CtValue) -> Result<CtValue, String> {
    use CtValue::*;
    match (op, l, r) {
        (BinOp::Add, Int(a), Int(b)) => Ok(Int(a.wrapping_add(b))),
        (BinOp::Sub, Int(a), Int(b)) => Ok(Int(a.wrapping_sub(b))),
        (BinOp::Mul, Int(a), Int(b)) => Ok(Int(a.wrapping_mul(b))),
        (BinOp::Div, Int(a), Int(b)) => {
            if b == 0 { return Err("compile-time eval: divide by zero".to_string()); }
            Ok(Int(a / b))
        }
        (BinOp::Mod, Int(a), Int(b)) => {
            if b == 0 { return Err("compile-time eval: modulo by zero".to_string()); }
            Ok(Int(a.rem_euclid(b)))
        }
        (BinOp::Pow, Int(a), Int(b)) => {
            if b < 0 || b > u32::MAX as i64 {
                return Err(format!("compile-time eval: pow exponent out of range: {}", b));
            }
            Ok(Int(a.wrapping_pow(b as u32)))
        }

        (BinOp::Eq, Int(a), Int(b)) => Ok(Bool(a == b)),
        (BinOp::NEq, Int(a), Int(b)) => Ok(Bool(a != b)),
        (BinOp::Lt, Int(a), Int(b)) => Ok(Bool(a < b)),
        (BinOp::Le, Int(a), Int(b)) => Ok(Bool(a <= b)),
        (BinOp::Gt, Int(a), Int(b)) => Ok(Bool(a > b)),
        (BinOp::Ge, Int(a), Int(b)) => Ok(Bool(a >= b)),

        (BinOp::Eq, Bool(a), Bool(b)) => Ok(Bool(a == b)),
        (BinOp::NEq, Bool(a), Bool(b)) => Ok(Bool(a != b)),
        (BinOp::And, Bool(a), Bool(b)) => Ok(Bool(a && b)),
        (BinOp::Or,  Bool(a), Bool(b)) => Ok(Bool(a || b)),

        (BinOp::Eq, Str(a), Str(b)) => Ok(Bool(a == b)),
        (BinOp::NEq, Str(a), Str(b)) => Ok(Bool(a != b)),
        (BinOp::Add, Str(a), Str(b)) => Ok(Str(a + &b)),

        (op, l, r) => Err(format!(
            "compile-time eval: {:?} not defined for ({}, {})",
            op, l.type_name(), r.type_name()
        )),
    }
}
