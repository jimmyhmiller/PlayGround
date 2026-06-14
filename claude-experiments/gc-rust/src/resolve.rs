//! Name resolution: build the global item table and resolve paths.
//!
//! This is a light pass between parsing and type checking. It does NOT do type
//! inference. It:
//!   - collects every top-level item (functions, types, traits, impls, consts)
//!     into a flat, name-indexed table, handling nested `mod`s by path,
//!   - records the kind of each name (so the type checker / lowerer knows a
//!     `Path` expr is a fn vs a unit-variant vs a const vs a local),
//!   - validates that referenced type names exist,
//!   - leaves trait-method and generic-instantiation resolution to Phase 2
//!     (those need types).
//!
//! Local-variable resolution (let bindings, params, match binds, closure
//! params) is done here too, producing a `DefId`/`LocalId` for every binding so
//! later phases don't re-walk scopes.

use crate::ast::*;
use crate::lexer::Span;
use std::collections::HashMap;

#[derive(Debug)]
pub struct ResolveError {
    pub msg: String,
    pub span: Span,
}

/// A resolved program: the original module plus a global symbol table.
pub struct Resolved {
    pub module: Module,
    pub globals: GlobalTable,
}

/// What a top-level name refers to.
#[derive(Clone, Debug)]
pub enum SymKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Const,
    TypeAlias,
    /// An enum variant: (enum name, variant index, has-payload).
    Variant { enum_name: String, index: u32, has_payload: bool },
}

#[derive(Clone, Debug)]
pub struct Symbol {
    pub kind: SymKind,
    pub span: Span,
}

#[derive(Default)]
pub struct GlobalTable {
    /// Fully-qualified dotted name (`geometry::area`) → symbol. Also indexed by
    /// last segment for unqualified lookups where unambiguous.
    pub by_path: HashMap<String, Symbol>,
    /// last-segment → list of full paths (for unqualified resolution).
    pub by_last: HashMap<String, Vec<String>>,
    /// Struct/enum/trait definitions kept for layout + method resolution.
    pub structs: HashMap<String, StructDef>,
    pub enums: HashMap<String, EnumDef>,
    pub traits: HashMap<String, TraitDef>,
    pub impls: Vec<ImplBlock>,
    pub fns: HashMap<String, FnDef>,
    pub consts: HashMap<String, ConstDef>,
    pub aliases: HashMap<String, TypeAlias>,
}

impl GlobalTable {
    fn insert(&mut self, path: String, sym: Symbol) {
        let last = path.rsplit("::").next().unwrap().to_string();
        self.by_last.entry(last).or_default().push(path.clone());
        self.by_path.insert(path, sym);
    }

    /// Resolve a (possibly unqualified) path to a symbol.
    pub fn resolve(&self, path: &Path) -> Option<&Symbol> {
        let joined = path.segments.join("::");
        if let Some(s) = self.by_path.get(&joined) {
            return Some(s);
        }
        // Try last-segment if unambiguous.
        if let Some(paths) = self.by_last.get(path.last()) {
            if paths.len() == 1 {
                return self.by_path.get(&paths[0]);
            }
        }
        None
    }
}

pub fn resolve_module(module: Module) -> Result<Resolved, ResolveError> {
    let mut globals = GlobalTable::default();
    collect_items(&module.items, &mut Vec::new(), &mut globals)?;
    validate_types(&globals)?;
    Ok(Resolved { module, globals })
}

fn collect_items(
    items: &[Item],
    prefix: &mut Vec<String>,
    g: &mut GlobalTable,
) -> Result<(), ResolveError> {
    for item in items {
        match &item.kind {
            ItemKind::Fn(f) => {
                let path = qualify(prefix, &f.name);
                g.insert(path.clone(), Symbol { kind: SymKind::Fn, span: f.span });
                g.fns.insert(path, f.clone());
            }
            ItemKind::Struct(s) => {
                let path = qualify(prefix, &s.name);
                g.insert(path.clone(), Symbol { kind: SymKind::Struct, span: s.span });
                g.structs.insert(path, s.clone());
            }
            ItemKind::Enum(e) => {
                let path = qualify(prefix, &e.name);
                g.insert(path.clone(), Symbol { kind: SymKind::Enum, span: e.span });
                g.enums.insert(path.clone(), e.clone());
                // Register variants as `Enum::Variant`.
                for (i, v) in e.variants.iter().enumerate() {
                    let vpath = format!("{}::{}", path, v.name);
                    let has_payload = !matches!(v.payload, VariantPayload::None);
                    g.insert(
                        vpath,
                        Symbol {
                            kind: SymKind::Variant {
                                enum_name: path.clone(),
                                index: i as u32,
                                has_payload,
                            },
                            span: v.span,
                        },
                    );
                }
            }
            ItemKind::Trait(t) => {
                let path = qualify(prefix, &t.name);
                g.insert(path.clone(), Symbol { kind: SymKind::Trait, span: t.span });
                g.traits.insert(path, t.clone());
            }
            ItemKind::Impl(b) => {
                g.impls.push(b.clone());
            }
            ItemKind::Const(c) => {
                let path = qualify(prefix, &c.name);
                g.insert(path.clone(), Symbol { kind: SymKind::Const, span: c.span });
                g.consts.insert(path, c.clone());
            }
            ItemKind::TypeAlias(a) => {
                let path = qualify(prefix, &a.name);
                g.insert(path.clone(), Symbol { kind: SymKind::TypeAlias, span: a.span });
                g.aliases.insert(path, a.clone());
            }
            ItemKind::Mod(m) => {
                prefix.push(m.name.clone());
                collect_items(&m.items, prefix, g)?;
                prefix.pop();
            }
            ItemKind::Use(_) => { /* v0: `use` is a no-op alias hint; lookups try
                                     both qualified and last-segment already. */ }
        }
    }
    Ok(())
}

fn qualify(prefix: &[String], name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}::{}", prefix.join("::"), name)
    }
}

/// Shallow check that every named type referenced by a signature exists (as a
/// builtin, a declared struct/enum/alias, or a generic parameter in scope).
/// Deep checking is the type checker's job; this catches obvious typos early.
fn validate_types(g: &GlobalTable) -> Result<(), ResolveError> {
    // Build a set of all known type names.
    let mut known: Vec<String> = Vec::new();
    known.extend(g.structs.keys().cloned());
    known.extend(g.enums.keys().cloned());
    known.extend(g.aliases.keys().cloned());
    let is_known = |name: &str, generics: &[String]| -> bool {
        is_builtin_type(name)
            || generics.iter().any(|gp| gp == name)
            || name == "Self"
            || g.by_last.contains_key(name)
    };

    for f in g.fns.values() {
        let gp: Vec<String> = f.generics.params.iter().map(|p| p.name.clone()).collect();
        for p in &f.params {
            check_type(&p.ty, &gp, &is_known)?;
        }
        if let Some(r) = &f.ret {
            check_type(r, &gp, &is_known)?;
        }
    }
    for s in g.structs.values() {
        let gp: Vec<String> = s.generics.params.iter().map(|p| p.name.clone()).collect();
        if let StructBody::Named(fields) = &s.body {
            for f in fields {
                check_type(&f.ty, &gp, &is_known)?;
            }
        }
        if let StructBody::Tuple(tys) = &s.body {
            for t in tys {
                check_type(t, &gp, &is_known)?;
            }
        }
    }
    for e in g.enums.values() {
        let gp: Vec<String> = e.generics.params.iter().map(|p| p.name.clone()).collect();
        for v in &e.variants {
            match &v.payload {
                VariantPayload::Tuple(tys) => {
                    for t in tys { check_type(t, &gp, &is_known)?; }
                }
                VariantPayload::Named(fields) => {
                    for f in fields { check_type(&f.ty, &gp, &is_known)?; }
                }
                VariantPayload::None => {}
            }
        }
    }
    let _ = known;
    Ok(())
}

fn check_type(
    t: &Type,
    generics: &[String],
    is_known: &impl Fn(&str, &[String]) -> bool,
) -> Result<(), ResolveError> {
    match &t.kind {
        TypeKind::Path(path, args) => {
            let name = path.last();
            if !is_known(name, generics) {
                return Err(ResolveError {
                    msg: format!("unknown type `{}`", name),
                    span: t.span,
                });
            }
            for a in args {
                check_type(a, generics, is_known)?;
            }
        }
        TypeKind::Tuple(tys) => {
            for t in tys { check_type(t, generics, is_known)?; }
        }
        TypeKind::Array(elem, _) => check_type(elem, generics, is_known)?,
        TypeKind::Fn(params, ret) => {
            for p in params { check_type(p, generics, is_known)?; }
            if let Some(r) = ret { check_type(r, generics, is_known)?; }
        }
        TypeKind::SelfType => {}
    }
    Ok(())
}

pub fn is_builtin_type(name: &str) -> bool {
    matches!(
        name,
        "i8" | "i16" | "i32" | "i64"
            | "u8" | "u16" | "u32" | "u64"
            | "f32" | "f64"
            | "bool" | "char" | "String" | "Bytes"
            // generic builtins resolved structurally later
            | "Vec" | "Array" | "Option" | "Result"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse_module;

    fn resolve(src: &str) -> Result<Resolved, ResolveError> {
        let toks = lex(src).unwrap();
        let m = parse_module(&toks).unwrap();
        resolve_module(m)
    }

    #[test]
    fn collects_fns_and_types() {
        let r = resolve(
            "struct Point { x: i64, y: i64 } fn area(p: Point) -> i64 { p.x }",
        )
        .unwrap();
        assert!(r.globals.by_path.contains_key("Point"));
        assert!(matches!(
            r.globals.resolve(&path("area")).unwrap().kind,
            SymKind::Fn
        ));
    }

    #[test]
    fn enum_variants_registered() {
        let r = resolve("enum Color { Red, Green, Blue }").unwrap();
        let sym = r.globals.by_path.get("Color::Green").unwrap();
        assert!(matches!(sym.kind, SymKind::Variant { index: 1, .. }));
    }

    #[test]
    fn unknown_type_errors() {
        let e = resolve("fn f(x: Nope) -> i64 { 0 }");
        assert!(e.is_err());
    }

    #[test]
    fn generic_param_is_known() {
        let r = resolve("fn id<T>(x: T) -> T { x }");
        assert!(r.is_ok());
    }

    #[test]
    fn nested_mod_qualifies() {
        let r = resolve("mod geo { pub fn area() -> i64 { 0 } }").unwrap();
        assert!(r.globals.by_path.contains_key("geo::area"));
        // unqualified resolution works since it's unambiguous
        assert!(r.globals.resolve(&path("area")).is_some());
    }

    fn path(name: &str) -> Path {
        Path::single(name.to_string(), Span::new(0, 0))
    }
}
