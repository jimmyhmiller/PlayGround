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
        consts: vec![],
        traits: vec![],
        impls: vec![],
        statics: vec![],
    };
    for (mut p, module) in parsed {
        if let Some(m) = &module {
            qualify_program(&mut p, m, imports.get(m), &table, exports)?;
        }
        merge(&mut out, p)?;
    }
    Ok(out)
}

/// Two extern declarations are the *same* declaration (dedup-able) when they agree
/// on convention, parameter types, variadicness, and return type.
fn extern_eq(a: &Extern, b: &Extern) -> bool {
    a.cc == b.cc && a.params == b.params && a.variadic == b.variadic && a.ret == b.ret
}

fn merge(out: &mut Program, p: Program) -> Result<(), crate::span::Diag> {
    for (k, v) in p.conventions {
        out.conventions.entry(k).or_insert(v); // dedups the default `c`
    }
    out.structs.extend(p.structs);
    out.sums.extend(p.sums);
    // Dedup identical extern declarations across modules (the same libc symbol
    // declared by several libraries is allowed, as in C); only a same-name extern
    // with a *conflicting* signature is an error. Coil has no other extern dedup,
    // so this is where shared `(extern malloc …)`/`(extern abort …)` collisions
    // are resolved.
    for e in p.externs {
        match out.externs.iter().find(|x| x.name == e.name) {
            Some(prev) if extern_eq(prev, &e) => {} // identical redeclaration — drop
            Some(_) => {
                return Err(crate::span::Diag::new(format!(
                    "extern '{}' declared with conflicting signatures",
                    e.name
                )))
            }
            None => out.externs.push(e),
        }
    }
    out.funcs.extend(p.funcs);
    out.asserts.extend(p.asserts);
    // Traits + impls share one flat global namespace (v1): collected as-is.
    out.traits.extend(p.traits);
    out.impls.extend(p.impls);
    // Consts share one flat global namespace (like externs / C constants), so a
    // re-import of the same bindings is fine but two distinct definitions of the
    // same name collide loudly.
    for c in p.consts {
        if out.consts.iter().any(|x| x.name == c.name) {
            return Err(crate::span::Diag::new(format!(
                "const '{}' defined more than once",
                c.name
            )));
        }
        out.consts.push(c);
    }
    Ok(())
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
    // Const values are now expressions (possibly calling functions / using types).
    for c in &mut p.consts {
        if let Some(t) = &mut c.ty {
            qualify_type(t, m, imps, table, &empty, exports)?;
        }
        qualify_expr(&mut c.value, m, imps, table, &empty, exports)?;
    }
    // Traits: qualify the types in each method signature (Self stays unqualified,
    // treated as an in-scope type parameter). Trait names are a flat global
    // namespace in v1, so the name itself isn't renamed.
    for t in &mut p.traits {
        let tps: HashSet<String> = std::iter::once(t.self_param.clone()).collect();
        for meth in &mut t.methods {
            for param in &mut meth.params {
                qualify_type(&mut param.ty, m, imps, table, &tps, exports)?;
            }
            qualify_type(&mut meth.ret, m, imps, table, &tps, exports)?;
        }
    }
    // Impls: resolve the implementing type to its qualified name, and qualify each
    // method's signature + body like an ordinary function (the method name is left
    // bare — the checker mangles it to <Trait>$<Type>$<method>).
    for im in &mut p.impls {
        // A scalar impl target (`(impl Eq i64 …)`) keeps its spelling; a nominal
        // one resolves to its qualified type name.
        if !crate::ast::is_scalar_typename(&im.for_type) {
            let mut for_ty = Type::Struct(im.for_type.clone());
            qualify_type(&mut for_ty, m, imps, table, &empty, exports)?;
            if let Type::Struct(n) = for_ty {
                im.for_type = n;
            }
        }
        for meth in &mut im.methods {
            let tps: HashSet<String> = meth.type_params.iter().cloned().collect();
            for param in &mut meth.params {
                qualify_type(&mut param.ty, m, imps, table, &tps, exports)?;
            }
            qualify_type(&mut meth.ret, m, imps, table, &tps, exports)?;
            for e in &mut meth.body {
                qualify_expr(e, m, imps, table, &tps, exports)?;
            }
        }
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
        Type::Never => {}   // synthesized only; never appears in user-written types
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
        Type::Ptr(p) | Type::Ref(_, p) | Type::Array(p, _) | Type::Slice(p) | Type::Vec(p, _) => {
            qualify_type(p, m, imps, table, tps, exports)?
        }
        Type::Fn(cc, params, ret) => {
            *cc = resolve(cc, m, imps, table, exports, |d| &d.convs)?;
            for p in params {
                qualify_type(p, m, imps, table, tps, exports)?;
            }
            qualify_type(ret, m, imps, table, tps, exports)?;
        }
        Type::Int(..) | Type::Float(..) | Type::Bool | Type::Void => {}
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
    match &mut e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::CStr(_) | ExprKind::Var(_) => {}
        ExprKind::Zeroed(t) | ExprKind::SizeOf(t) | ExprKind::AlignOf(t) | ExprKind::OffsetOf(t, _) => ty(t)?,
        ExprKind::Borrow { place, .. } => qualify_expr(place, m, imps, table, tps, exports)?,
        // `SpillRef` is inserted by the checker, which runs after name
        // resolution; the resolver never encounters one.
        ExprKind::SpillRef(_) => unreachable!("SpillRef is produced after name resolution"),
        ExprKind::Let { binds, body } => {
            for (_, _, v) in binds {
                qualify_expr(v, m, imps, table, tps, exports)?;
            }
            for e in body {
                qualify_expr(e, m, imps, table, tps, exports)?;
            }
        }
        ExprKind::Bin { lhs, rhs, .. } | ExprKind::Cmp { lhs, rhs, .. } => {
            qualify_expr(lhs, m, imps, table, tps, exports)?;
            qualify_expr(rhs, m, imps, table, tps, exports)?;
        }
        ExprKind::Not(x) | ExprKind::Load(x) | ExprKind::Free(x) => {
            qualify_expr(x, m, imps, table, tps, exports)?
        }
        ExprKind::If { cond, then, els } => {
            qualify_expr(cond, m, imps, table, tps, exports)?;
            qualify_expr(then, m, imps, table, tps, exports)?;
            qualify_expr(els, m, imps, table, tps, exports)?;
        }
        ExprKind::Do(es) => {
            for e in es {
                qualify_expr(e, m, imps, table, tps, exports)?;
            }
        }
        // Loop bodies and break values are ordinary expressions; labels are not
        // names to resolve.
        ExprKind::Loop { body, .. } => {
            for e in body {
                qualify_expr(e, m, imps, table, tps, exports)?;
            }
        }
        ExprKind::Break { value, .. } => {
            if let Some(v) = value {
                qualify_expr(v, m, imps, table, tps, exports)?;
            }
        }
        ExprKind::Continue { .. } => {}
        ExprKind::Call { func, type_args, args } => {
            *func = call(func, |d| &d.callables)?;
            for t in type_args {
                qualify_type(t, m, imps, table, tps, exports)?;
            }
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
        ExprKind::Alloc { ty: t, .. } => ty(t)?,
        ExprKind::Field { ptr, .. } | ExprKind::BitGet { ptr, .. } => {
            qualify_expr(ptr, m, imps, table, tps, exports)?
        }
        ExprKind::Store { ptr, val } => {
            qualify_expr(ptr, m, imps, table, tps, exports)?;
            qualify_expr(val, m, imps, table, tps, exports)?;
        }
        ExprKind::Index { ptr, idx } => {
            qualify_expr(ptr, m, imps, table, tps, exports)?;
            qualify_expr(idx, m, imps, table, tps, exports)?;
        }
        ExprKind::Cast { ty: t, expr } => {
            ty(t)?;
            qualify_expr(expr, m, imps, table, tps, exports)?;
        }
        ExprKind::BitSet { ptr, val, .. } => {
            qualify_expr(ptr, m, imps, table, tps, exports)?;
            qualify_expr(val, m, imps, table, tps, exports)?;
        }
        ExprKind::Construct { sum, variant, args } => {
            *sum = call(sum, |d| &d.types)?;
            *variant = call(variant, |d| &d.callables)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
        ExprKind::Match { scrut, arms } => {
            qualify_expr(scrut, m, imps, table, tps, exports)?;
            for arm in arms {
                arm.variant = call(&arm.variant, |d| &d.callables)?;
                qualify_expr(&mut arm.body, m, imps, table, tps, exports)?;
            }
        }
        ExprKind::FnPtrOf(name) => *name = call(name, |d| &d.callables)?,
        ExprKind::CallPtr { fp, args } => {
            qualify_expr(fp, m, imps, table, tps, exports)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
        ExprKind::LlvmIr { result, args, .. } => {
            ty(result)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
        // The parser never produces a TraitCall (the checker does, post-resolve);
        // recurse into args for completeness.
        ExprKind::TraitCall { args, .. } => {
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports)?;
            }
        }
        ExprKind::Comptime(inner) => qualify_expr(inner, m, imps, table, tps, exports)?,
        ExprKind::StaticRef(_) => {} // checker-produced; no names to qualify
        ExprKind::TypeQuery { ty: t, .. } => ty(t)?,
        ExprKind::FieldMeta { ty: t, idx, .. } => {
            ty(t)?;
            qualify_expr(idx, m, imps, table, tps, exports)?;
        }
        ExprKind::FieldIndex { ty: t, name } => {
            ty(t)?;
            qualify_expr(name, m, imps, table, tps, exports)?;
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
