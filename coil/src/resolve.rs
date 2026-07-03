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
    traits: HashSet<String>,    // deftrait names
    convs: HashSet<String>,
    externs: HashSet<String>, // C symbols — referenced bare, never renamed
}
type DefTable = HashMap<String, Defs>;

/// The core namespace (the prelude): auto-referred into every module like Clojure's
/// `clojure.core`, so its traits/operators are in scope with no import.
const CORE: &str = "coil.core";

/// Parse every tagged form, qualify it under its module, and merge into one
/// whole-program `Program`.
/// `strict` enables the undefined-reference check (a still-bare callee that is
/// neither an extern nor a trait method is an error). It must be OFF for an
/// INTERMEDIATE resolve whose program still calls not-yet-generated definitions
/// (the pre-`meta` pass, and the macro-detection subset); ON for the final program.
pub fn resolve_program(
    tagged: Vec<TaggedForm>,
    imports: &ImportMap,
    exports: &ExportMap,
    strict: bool,
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
            for t in &p.traits {
                d.traits.insert(t.name.clone());
            }
            for c in p.conventions.keys() {
                if c != "c" {
                    d.convs.insert(c.clone());
                }
            }
            for e in &p.externs {
                d.externs.insert(e.name.clone());
                // An extern resolves like any callable, so a bare reference in its
                // module qualifies to `module.name` (its bare C symbol becomes the
                // link name at codegen). This keeps externs from leaking across
                // modules: `io`'s `write` is `io.write`, distinct from your `write`.
                d.callables.insert(e.name.clone());
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
        metas: vec![],
        exports: vec![],
    };
    // Names allowed to remain bare after resolution: trait-method names (the checker
    // resolves a method call against the caller's scope). Externs are now qualified
    // like callables, so they no longer need to stay bare. Anything else still-bare is
    // an undefined reference.
    let mut bare_ok: HashSet<String> = HashSet::new();
    for (p, _) in &parsed {
        for t in &p.traits {
            for meth in &t.methods {
                bare_ok.insert(meth.name.clone());
            }
        }
    }
    let bare_ok = strict.then_some(&bare_ok);
    for (mut p, module) in parsed {
        if let Some(m) = &module {
            qualify_program(&mut p, m, imports.get(m), &table, exports, bare_ok)?;
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
    out.exports.extend(p.exports);
    out.asserts.extend(p.asserts);
    out.metas.extend(p.metas);
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
    bare_ok: Option<&HashSet<String>>,
) -> Result<(), String> {
    let empty = HashSet::new();
    for f in &mut p.funcs {
        let tps: HashSet<String> = f.type_params.iter().cloned().collect();
        if f.name != "main" {
            f.name = format!("{m}.{}", f.name);
        }
        f.cc = resolve(&f.cc, m, imps, table, exports, |d| &d.convs)?;
        // Resolve each trait name in the bounds (`(T Eq)` → `coil.core.Eq`).
        for (_, traits) in &mut f.bounds {
            for tr in traits {
                *tr = resolve(tr, m, imps, table, exports, |d| &d.traits)?;
            }
        }
        for param in &mut f.params {
            qualify_type(&mut param.ty, m, imps, table, &tps, exports)?;
        }
        qualify_type(&mut f.ret, m, imps, table, &tps, exports)?;
        for e in &mut f.body {
            qualify_expr(e, m, imps, table, &tps, exports, bare_ok)?;
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
    // Qualify the extern's Coil name to `module.name` (like a function); its bare C
    // symbol — the last path component — becomes the LLVM/link name at codegen, so
    // several modules can declare the same libc symbol without a name clash.
    for e in &mut p.externs {
        e.name = format!("{m}.{}", e.name);
        for t in &mut e.params {
            qualify_type(t, m, imps, table, &empty, exports)?;
        }
        qualify_type(&mut e.ret, m, imps, table, &empty, exports)?;
    }
    // `(export-c foo)` names a function in this module — qualify it like a callable.
    for ex in &mut p.exports {
        ex.name = resolve(&ex.name, m, imps, table, exports, |d| &d.callables)?;
    }
    for a in &mut p.asserts {
        qualify_expr(&mut a.cond, m, imps, table, &empty, exports, bare_ok)?;
    }
    // Const values are now expressions (possibly calling functions / using types).
    for c in &mut p.consts {
        if let Some(t) = &mut c.ty {
            qualify_type(t, m, imps, table, &empty, exports)?;
        }
        qualify_expr(&mut c.value, m, imps, table, &empty, exports, bare_ok)?;
    }
    // `(meta EXPR)` — EXPR is a comptime expression; qualify it.
    for meta in &mut p.metas {
        qualify_expr(meta, m, imps, table, &empty, exports, bare_ok)?;
    }
    // Traits: rename the trait to `module.Trait` and qualify the types in each
    // method signature (Self and any extra trait type parameters stay
    // unqualified — they're in-scope parameters, not module types).
    for t in &mut p.traits {
        t.name = format!("{m}.{}", t.name);
        let tps: HashSet<String> = std::iter::once(t.self_param.clone())
            .chain(t.type_params.iter().cloned())
            .collect();
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
        // Resolve the trait reference (`(impl Eq i64 …)` → `coil.core.Eq`).
        im.trait_name = resolve(&im.trait_name, m, imps, table, exports, |d| &d.traits)?;
        // Qualify the implementing type (the impl's own type params stay
        // unqualified), then re-derive the dispatch base name from it —
        // scalars/`slice` keep their spelling, nominal names qualify.
        {
            let itps: HashSet<String> = im.type_params.iter().cloned().collect();
            qualify_type(&mut im.self_type, m, imps, table, &itps, exports)?;
            im.for_type = crate::ast::type_impl_name(&im.self_type)
                .expect("impl target validated at parse");
        }
        for meth in &mut im.methods {
            let tps: HashSet<String> = meth.type_params.iter().cloned().collect();
            for param in &mut meth.params {
                qualify_type(&mut param.ty, m, imps, table, &tps, exports)?;
            }
            qualify_type(&mut meth.ret, m, imps, table, &tps, exports)?;
            for e in &mut meth.body {
                qualify_expr(e, m, imps, table, &tps, exports, bare_ok)?;
            }
        }
    }
    Ok(())
}

fn qualify_quasi(
    q: &mut Quasi,
    m: &str,
    imps: Option<&ModImports>,
    table: &DefTable,
    tps: &HashSet<String>,
    exports: &ExportMap,
    bare_ok: Option<&HashSet<String>>,
) -> Result<(), String> {
    match q {
        Quasi::Lit(_) => {}
        Quasi::Unquote(e) | Quasi::Splice(e) => qualify_expr(e, m, imps, table, tps, exports, bare_ok)?,
        Quasi::List(items) | Quasi::Vector(items) => {
            for it in items {
                qualify_quasi(it, m, imps, table, tps, exports, bare_ok)?;
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
        Type::Code => {}    // comptime-only; no names to qualify
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
    bare_ok: Option<&HashSet<String>>,
) -> Result<(), String> {
    let ty = |ty: &mut Type| qualify_type(ty, m, imps, table, tps, exports);
    let call = |name: &str, pick: Pick| resolve(name, m, imps, table, exports, pick);
    match &mut e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::CStr(_) | ExprKind::Var(_) => {}
        ExprKind::Zeroed(t) | ExprKind::SizeOf(t) | ExprKind::AlignOf(t) | ExprKind::OffsetOf(t, _) => ty(t)?,
        ExprKind::Borrow { place, .. } => qualify_expr(place, m, imps, table, tps, exports, bare_ok)?,
        // `SpillRef` is inserted by the checker, which runs after name
        // resolution; the resolver never encounters one.
        ExprKind::SpillRef(_) => unreachable!("SpillRef is produced after name resolution"),
        ExprKind::Let { binds, body } => {
            for (_, _, v) in binds {
                qualify_expr(v, m, imps, table, tps, exports, bare_ok)?;
            }
            for e in body {
                qualify_expr(e, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::Bin { lhs, rhs, .. } | ExprKind::Cmp { lhs, rhs, .. } => {
            qualify_expr(lhs, m, imps, table, tps, exports, bare_ok)?;
            qualify_expr(rhs, m, imps, table, tps, exports, bare_ok)?;
        }
        // `(make-dyn Trait expr)`: qualify the trait name like a type reference,
        // resolve the inner expression. `MakeDyn` is produced after resolution.
        ExprKind::Erase { trait_name, inner } => {
            let mut tt = Type::Struct(std::mem::take(trait_name));
            ty(&mut tt)?;
            if let Type::Struct(n) = tt {
                *trait_name = n;
            }
            qualify_expr(inner, m, imps, table, tps, exports, bare_ok)?
        }
        ExprKind::MakeDyn { inner, .. } => {
            qualify_expr(inner, m, imps, table, tps, exports, bare_ok)?
        }
        // `DynDispatch` is produced by the checker, after resolution.
        ExprKind::DynDispatch { recv, args, .. } => {
            qualify_expr(recv, m, imps, table, tps, exports, bare_ok)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::Not(x) | ExprKind::Load(x) | ExprKind::Free(x) => {
            qualify_expr(x, m, imps, table, tps, exports, bare_ok)?
        }
        ExprKind::If { cond, then, els } => {
            qualify_expr(cond, m, imps, table, tps, exports, bare_ok)?;
            qualify_expr(then, m, imps, table, tps, exports, bare_ok)?;
            qualify_expr(els, m, imps, table, tps, exports, bare_ok)?;
        }
        ExprKind::Do(es) => {
            for e in es {
                qualify_expr(e, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        // Loop bodies and break values are ordinary expressions; labels are not
        // names to resolve.
        ExprKind::Loop { body, .. } => {
            for e in body {
                qualify_expr(e, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::Break { value, .. } => {
            if let Some(v) = value {
                qualify_expr(v, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::Continue { .. } => {}
        ExprKind::Call { func, type_args, args } => {
            *func = call(func, |d| &d.callables)?;
            // A still-bare callee (no module qualifier) that is neither a global
            // extern nor a trait-method name is an undefined reference — report it
            // here rather than letting it slip to a later pass. (Skipped on a
            // non-strict resolve, whose program may still call generated defs.)
            if let Some(ok) = bare_ok {
                if !func.contains('.') && !ok.contains(func) {
                    return Err(format!("in module '{m}': call to undefined function '{func}'"));
                }
            }
            for t in type_args {
                qualify_type(t, m, imps, table, tps, exports)?;
            }
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::Alloc { ty: t, .. } => ty(t)?,
        ExprKind::Field { ptr, .. } | ExprKind::BitGet { ptr, .. } => {
            qualify_expr(ptr, m, imps, table, tps, exports, bare_ok)?
        }
        ExprKind::Store { ptr, val } => {
            qualify_expr(ptr, m, imps, table, tps, exports, bare_ok)?;
            qualify_expr(val, m, imps, table, tps, exports, bare_ok)?;
        }
        ExprKind::Index { ptr, idx } => {
            qualify_expr(ptr, m, imps, table, tps, exports, bare_ok)?;
            qualify_expr(idx, m, imps, table, tps, exports, bare_ok)?;
        }
        ExprKind::Cast { ty: t, expr } => {
            ty(t)?;
            qualify_expr(expr, m, imps, table, tps, exports, bare_ok)?;
        }
        ExprKind::BitSet { ptr, val, .. } => {
            qualify_expr(ptr, m, imps, table, tps, exports, bare_ok)?;
            qualify_expr(val, m, imps, table, tps, exports, bare_ok)?;
        }
        ExprKind::Construct { sum, variant, args } => {
            *sum = call(sum, |d| &d.types)?;
            *variant = call(variant, |d| &d.callables)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::Match { scrut, arms } => {
            qualify_expr(scrut, m, imps, table, tps, exports, bare_ok)?;
            for arm in arms {
                arm.variant = call(&arm.variant, |d| &d.callables)?;
                qualify_expr(&mut arm.body, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::FnPtrOf(name) => *name = call(name, |d| &d.callables)?,
        ExprKind::CallPtr { fp, args } => {
            qualify_expr(fp, m, imps, table, tps, exports, bare_ok)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::LlvmIr { result, args, .. } => {
            ty(result)?;
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        // The parser never produces a TraitCall (the checker does, post-resolve);
        // recurse into args for completeness.
        ExprKind::TraitCall { args, .. } => {
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        ExprKind::Comptime(inner) => qualify_expr(inner, m, imps, table, tps, exports, bare_ok)?,
        ExprKind::StaticRef(_) => {} // checker-produced; no names to qualify
        ExprKind::TypeQuery { ty: t, .. } => ty(t)?,
        ExprKind::FieldMeta { ty: t, idx, .. } => {
            ty(t)?;
            qualify_expr(idx, m, imps, table, tps, exports, bare_ok)?;
        }
        ExprKind::FieldIndex { ty: t, name } => {
            ty(t)?;
            qualify_expr(name, m, imps, table, tps, exports, bare_ok)?;
        }
        // Quoted code is raw syntax — its names are data, not references.
        ExprKind::Quote(_) => {}
        ExprKind::CodeOp { args, .. } => {
            for a in args {
                qualify_expr(a, m, imps, table, tps, exports, bare_ok)?;
            }
        }
        // A quasiquote template is data; only its unquote holes are real exprs.
        ExprKind::Quasi(q) => qualify_quasi(q, m, imps, table, tps, exports, bare_ok)?,
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
    // own module (externs are in `callables`, so they qualify here too)
    if let Some(d) = table.get(m) {
        if pick(d).contains(name) {
            return Ok(format!("{m}.{name}"));
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
    // `coil.core` is auto-referred into every module (like clojure.core).
    if m != CORE {
        if let Some(d) = table.get(CORE) {
            if pick(d).contains(name) && macros::exports(exports, CORE, name) {
                return Ok(format!("{CORE}.{name}"));
            }
        }
    }
    Ok(name.to_string())
}
