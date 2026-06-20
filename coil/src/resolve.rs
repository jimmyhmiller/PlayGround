//! Module name resolution — the "qualify" pass, at the AST level.
//!
//! Each top-level form is parsed *with* its source module, then every definition
//! is renamed `module/name` and every reference rewritten to match. Working on
//! the AST (not raw forms) means the parser has already separated type positions,
//! value references, conventions, and locals — so resolution is a structured
//! walk with no special-form enumeration and no local-scope tracking (a `Var` is
//! always a local; a `Call.func` is always a global callable).
//!
//! Resolution of a bare name in module M: (1) M's own definitions, (2) an extern
//! (left bare — it is a C symbol), (3) a `:use`d module's exports, else left
//! as-is. A qualified `alias/x` resolves via M's `:as` aliases. `main` and
//! externs are never renamed.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::macros::{self, ExportMap, ImportMap, ModImports, TaggedForm, UseSpec};
use crate::parse;

/// What each module defines, split by namespace.
#[derive(Default)]
struct Defs {
    callables: HashSet<String>, // functions + sum-variant constructors
    types: HashSet<String>,     // struct + sum type names
    convs: HashSet<String>,
    externs: HashSet<String>, // C symbols — referenced bare, never renamed
}
type DefTable = HashMap<String, Defs>;

/// Parse every tagged form, qualify it under its module, and merge into one
/// whole-program `Program`.
pub fn resolve_program(
    tagged: Vec<TaggedForm>,
    imports: &ImportMap,
    exports: &ExportMap,
) -> Result<Program, crate::span::Diag> {
    // Parse each form alone so it keeps its module identity.
    let mut parsed: Vec<(Program, Option<String>)> = Vec::with_capacity(tagged.len());
    for (form, module) in tagged {
        let p = parse::parse_program(std::slice::from_ref(&form))?;
        parsed.push((p, module));
    }
    // Pass 1: collect each module's definition names.
    let mut table: DefTable = HashMap::new();
    for (p, module) in &parsed {
        if let Some(m) = module {
            let d = table.entry(m.clone()).or_default();
            for f in &p.funcs {
                if f.name != "main" {
                    d.callables.insert(f.name.clone());
                }
            }
            for s in &p.structs {
                d.types.insert(s.name.clone());
            }
            for s in &p.sums {
                d.types.insert(s.name.clone());
                for v in &s.variants {
                    d.callables.insert(v.name.clone());
                }
            }
            for c in p.conventions.keys() {
                if c != "c" {
                    d.convs.insert(c.clone());
                }
            }
            for e in &p.externs {
                d.externs.insert(e.name.clone());
            }
        }
    }
    // Pass 2: qualify each module's items and merge.
    let mut out = Program {
        conventions: HashMap::new(),
        structs: vec![],
        sums: vec![],
        externs: vec![],
        funcs: vec![],
        asserts: vec![],
    };
    for (mut p, module) in parsed {
        if let Some(m) = &module {
            qualify_program(&mut p, m, imports.get(m), &table, exports)?;
        }
        merge(&mut out, p);
    }
    Ok(out)
}

fn merge(out: &mut Program, p: Program) {
    for (k, v) in p.conventions {
        out.conventions.entry(k).or_insert(v); // dedups the default `c`
    }
    out.structs.extend(p.structs);
    out.sums.extend(p.sums);
    out.externs.extend(p.externs);
    out.funcs.extend(p.funcs);
    out.asserts.extend(p.asserts);
}

type Pick = fn(&Defs) -> &HashSet<String>;

fn qualify_program(
    p: &mut Program,
    m: &str,
    imps: Option<&ModImports>,
    table: &DefTable,
    exports: &ExportMap,
) -> Result<(), String> {
    let empty = HashSet::new();
    for f in &mut p.funcs {
        let tps: HashSet<String> = f.type_params.iter().cloned().collect();
        if f.name != "main" {
            f.name = format!("{m}.{}", f.name);
        }
        f.cc = resolve(&f.cc, m, imps, table, exports, |d| &d.convs)?;
        for param in &mut f.params {
            qualify_type(&mut param.ty, m, imps, table, &tps, exports)?;
        }
        qualify_type(&mut f.ret, m, imps, table, &tps, exports)?;
        for e in &mut f.body {
            qualify_expr(e, m, imps, table, &tps, exports)?;
        }
    }
    for s in &mut p.structs {
        let tps: HashSet<String> = s.type_params.iter().cloned().collect();
        s.name = format!("{m}.{}", s.name);
        for (_, ty) in &mut s.fields {
            qualify_type(ty, m, imps, table, &tps, exports)?;
        }
    }
    for s in &mut p.sums {
        let tps: HashSet<String> = s.type_params.iter().cloned().collect();
        s.name = format!("{m}.{}", s.name);
        for v in &mut s.variants {
            v.name = format!("{m}.{}", v.name);
            for (_, ty) in &mut v.fields {
                qualify_type(ty, m, imps, table, &tps, exports)?;
            }
        }
    }
    // conventions live in a name-keyed map, so rekey (except the default `c`).
    let convs = std::mem::take(&mut p.conventions);
    for (k, mut c) in convs {
        if k == "c" {
            p.conventions.insert(k, c);
        } else {
            let nk = format!("{m}.{k}");
            c.name = nk.clone();
            p.conventions.insert(nk, c);
        }
    }
    // externs keep their bare C names; only their types are qualified.
    for e in &mut p.externs {
        for t in &mut e.params {
            qualify_type(t, m, imps, table, &empty, exports)?;
        }
        qualify_type(&mut e.ret, m, imps, table, &empty, exports)?;
    }
    for a in &mut p.asserts {
        qualify_expr(&mut a.cond, m, imps, table, &empty, exports)?;
    }
    Ok(())
}

fn qualify_type(
    ty: &mut Type,
    m: &str,
    imps: Option<&ModImports>,
    table: &DefTable,
    tps: &HashSet<String>,
    exports: &ExportMap,
) -> Result<(), String> {
    match ty {
        Type::Struct(n) => {
            if !tps.contains(n) {
                *n = resolve(n, m, imps, table, exports, |d| &d.types)?;
            }
        }
        Type::App(n, args) => {
            if !tps.contains(n) {
                *n = resolve(n, m, imps, table, exports, |d| &d.types)?;
            }
            for a in args {
                qualify_type(a, m, imps, table, tps, exports)?;
            }
        }
        Type::Ptr(p) | Type::Ref(_, p) | Type::Array(p, _) | Type::Vec(p, _) => {
            qualify_type(p, m, imps, table, tps, exports)?
        }
        Type::Fn(cc, params, ret) => {
            *cc = resolve(cc, m, imps, table, exports, |d| &d.convs)?;
            for p in params {
                qualify_type(p, m, imps, table, tps, exports)?;
            }
            qualify_type(ret, m, imps, table, tps, exports)?;
        }
        Type::Int(..) | Type::Float(..) | Type::Bool => {}
    }
    Ok(())
}

fn qualify_expr(
    e: &mut Expr,
    m: &str,
    imps: Option<&ModImports>,
    table: &DefTable,
    tps: &HashSet<String>,
    exports: &ExportMap,
) -> Result<(), String> {
    let ty = |ty: &mut Type| qualify_type(ty, m, imps, table, tps, exports);
    let call = |name: &str, pick: Pick| resolve(name, m, imps, table, exports, pick);
    match e {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Var(_) => {}
        Expr::Zeroed(t) | Expr::SizeOf(t) | Expr::AlignOf(t) | Expr::OffsetOf(t, _) => ty(t)?,
        Expr::Borrow { place, .. } => qualify_expr(place, m, imps, table, tps, exports)?,
        Expr::Let { binds, body } => {
            for (_, _, v) in binds {
                qualify_expr(v, m, imps, table, tps, exports)?;
            }
            for e in body {
                qualify_expr(e, m, imps, table, tps, exports)?;
            }
        }
        Expr::Bin { lhs, rhs, .. } | Expr::Cmp { lhs, rhs, .. } => {
            qualify_expr(lhs, m, imps, table, tps, exports)?;
            qualify_expr(rhs, m, imps, table, tps, exports)?;
        }
        Expr::Not(x) | Expr::Load(x) | Expr::Free(x) => {
            qualify_expr(x, m, imps, table, tps, exports)?
        }
        Expr::If { cond, then, els } => {
            qualify_expr(cond, m, imps, table, tps, exports)?;
            qualify_expr(then, m, imps, table, tps, exports)?;
            qualify_expr(els, m, imps, table, tps, exports)?;
        }
        Expr::Do(es) => {
            for e in es {
                qualify_expr(e, m, imps, table, tps, exports)?;
            }
        }
        // Loop bodies and break values are ordinary expressions; labels are not
        // names to resolve.
        Expr::Loop { body, .. } => {
            for e in body {
                qualify_expr(e, m, imps, table, tps, exports)?;
            }
        }
        Expr::Break { value, .. } => {
            if let Some(v) = value {
                qualify_expr(v, m, imps, table, tps, exports)?;
            }
        }
        Expr::Continue { .. } => {}
        Expr::Call { func, type_args, args } => {
            *func = call(func, |d| &d.callables)?;
            for t in type_args {
                qualify_type(t, m, imps, table, tps, exports)?;
            }
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
        Expr::Alloc { ty: t, .. } => ty(t)?,
        Expr::Field { ptr, .. } | Expr::BitGet { ptr, .. } => {
            qualify_expr(ptr, m, imps, table, tps, exports)?
        }
        Expr::Store { ptr, val } => {
            qualify_expr(ptr, m, imps, table, tps, exports)?;
            qualify_expr(val, m, imps, table, tps, exports)?;
        }
        Expr::Index { ptr, idx } => {
            qualify_expr(ptr, m, imps, table, tps, exports)?;
            qualify_expr(idx, m, imps, table, tps, exports)?;
        }
        Expr::Cast { ty: t, expr } => {
            ty(t)?;
            qualify_expr(expr, m, imps, table, tps, exports)?;
        }
        Expr::BitSet { ptr, val, .. } => {
            qualify_expr(ptr, m, imps, table, tps, exports)?;
            qualify_expr(val, m, imps, table, tps, exports)?;
        }
        Expr::Construct { sum, variant, args } => {
            *sum = call(sum, |d| &d.types)?;
            *variant = call(variant, |d| &d.callables)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
        Expr::Match { scrut, arms } => {
            qualify_expr(scrut, m, imps, table, tps, exports)?;
            for arm in arms {
                arm.variant = call(&arm.variant, |d| &d.callables)?;
                qualify_expr(&mut arm.body, m, imps, table, tps, exports)?;
            }
        }
        Expr::FnPtrOf(name) => *name = call(name, |d| &d.callables)?,
        Expr::CallPtr { fp, args } => {
            qualify_expr(fp, m, imps, table, tps, exports)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
        Expr::LlvmIr { result, args, .. } => {
            ty(result)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
    }
    Ok(())
}

/// Resolve a reference name in module `m`, picking the relevant namespace via
/// `pick` (callables / types / convs). A cross-module reference to a name the
/// target module doesn't `(export …)` is private: an explicit `alias/name` is a
/// hard error; a `:use`d name is simply not imported.
fn resolve(
    name: &str,
    m: &str,
    imps: Option<&ModImports>,
    table: &DefTable,
    exports: &ExportMap,
    pick: Pick,
) -> Result<String, String> {
    // qualified `alias/rest`
    if let Some((alias, rest)) = name.split_once('/') {
        if let Some(target) = imps.and_then(|i| i.aliases.get(alias)) {
            if !macros::exports(exports, target, rest) {
                return Err(format!(
                    "in module '{m}': '{rest}' is private to module '{target}' (add it to that module's (export …))"
                ));
            }
            return Ok(format!("{target}.{rest}"));
        }
        return Ok(name.to_string());
    }
    // own module (or a bare extern — a C symbol)
    if let Some(d) = table.get(m) {
        if pick(d).contains(name) {
            return Ok(format!("{m}.{name}"));
        }
        if d.externs.contains(name) {
            return Ok(name.to_string());
        }
    }
    // `:use`d modules (only their exported names)
    if let Some(imp) = imps {
        for (target, spec) in &imp.uses {
            let used = match spec {
                UseSpec::All => true,
                UseSpec::Names(ns) => ns.iter().any(|n| n == name),
            };
            if used && macros::exports(exports, target, name) {
                if let Some(d) = table.get(target) {
                    if pick(d).contains(name) {
                        return Ok(format!("{target}.{name}"));
                    }
                }
            }
        }
    }
    Ok(name.to_string())
}
