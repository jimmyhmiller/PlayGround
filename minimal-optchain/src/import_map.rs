//! A backend-neutral port of Next.js's `import_analyzer::ImportMap`
//! (`crates/next-custom-transforms/src/transforms/import_analyzer.rs`) — the
//! "which local name is bound to which import" primitive that `cjs_optimizer`,
//! `named_import_transform`, `dynamic`, and friends all build on.
//!
//! The original is written straight against swc: its maps are keyed on
//! `Ident::to_id()` (`(Atom, SyntaxContext)`) and `is_import` matches on swc AST
//! nodes. Here the *same* logic is written once over [`Backend`], so it also
//! runs on oxc — where binding identity is a `SymbolId` from a separate
//! `Semantic` pass, not a ctxt baked into the ident. Nothing in this file names
//! swc or oxc; the whole divergence is behind [`Backend::ident_binding`] /
//! [`Backend::for_each_import`] / [`Backend::as_static_member`].

use std::collections::HashMap;

use crate::{Backend, Imported};

/// Maps every import-bound local to what it refers to. Generic over the backend
/// only through [`Backend::BindingId`] as the key type.
pub struct ImportMap<B: Backend> {
    /// local binding → (module specifier, original exported name)
    named: HashMap<B::BindingId, (String, String)>,
    /// local binding → module specifier, for `import * as ns`
    namespaces: HashMap<B::BindingId, String>,
}

impl<B: Backend> ImportMap<B> {
    /// Build the map by walking every `import` in `program`. `sem` must come
    /// from [`Backend::build_semantics`] on the same program.
    pub fn analyze<'a>(program: &B::Program<'a>, sem: &B::Semantics<'a>) -> Self {
        let mut named = HashMap::new();
        let mut namespaces = HashMap::new();
        B::for_each_import(program, sem, |b| match b.imported {
            Imported::Named(orig) => {
                named.insert(b.local, (b.source, orig));
            }
            Imported::Default => {
                named.insert(b.local, (b.source, "default".to_string()));
            }
            Imported::Namespace => {
                namespaces.insert(b.local, b.source);
            }
        });
        Self { named, namespaces }
    }

    /// Is `e` a reference to `orig_name` imported from `module`? Handles both a
    /// direct binding (`import { orig as e }` … `e`) and a namespace member
    /// (`import * as ns` … `ns.orig`) — the two cases the original `is_import`
    /// covers.
    pub fn is_import<'a>(
        &self,
        sem: &B::Semantics<'a>,
        e: &B::Expr<'a>,
        module: &str,
        orig_name: &str,
    ) -> bool {
        if let Some(id) = B::ident_binding(sem, e) {
            if let Some((src, name)) = self.named.get(&id) {
                return src == module && name == orig_name;
            }
        }
        if let Some((obj, prop)) = B::as_static_member(e) {
            if prop == orig_name {
                if let Some(id) = B::ident_binding(sem, obj) {
                    if let Some(src) = self.namespaces.get(&id) {
                        return src == module;
                    }
                }
            }
        }
        false
    }

    /// Resolve `e` to a stable `"source#name"` description, or `None` if it is
    /// not a reference to any import. Used by the cross-backend harness to prove
    /// oxc and swc agree; `#*` denotes a namespace binding itself.
    pub fn describe<'a>(&self, sem: &B::Semantics<'a>, e: &B::Expr<'a>) -> Option<String> {
        if let Some(id) = B::ident_binding(sem, e) {
            if let Some((src, name)) = self.named.get(&id) {
                return Some(format!("{src}#{name}"));
            }
            if let Some(src) = self.namespaces.get(&id) {
                return Some(format!("{src}#*"));
            }
        }
        if let Some((obj, prop)) = B::as_static_member(e) {
            if let Some(id) = B::ident_binding(sem, obj) {
                if let Some(src) = self.namespaces.get(&id) {
                    return Some(format!("{src}#{prop}"));
                }
            }
        }
        None
    }
}
