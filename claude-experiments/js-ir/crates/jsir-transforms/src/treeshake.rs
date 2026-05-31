//! Cross-module dead code elimination (tree-shaking).
//!
//! Given a set of ES modules (`path -> source`) and an entry, this:
//!  1. resolves the import graph from the entry and **drops modules nothing
//!     imports** (whole-file dead code);
//!  2. computes which **exports** of each module are actually imported, and
//!     un-exports the dead ones so the intra-module DCE can delete them;
//!  3. runs the per-module [`crate::eliminate_dead_code`] pass (pinning the
//!     live exports as roots so they survive);
//!  4. drops **imports** whose local binding ends up unused.
//!
//! This is intentionally sound: a module reachable by any import is kept (its
//! top-level code may have side effects), and a default/`export *` is kept
//! conservatively. The win is dropping unreachable modules and unused named
//! exports, the cross-file analogue of the single-file pass.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use jsir_ir::{Attr, Op};

/// The result of tree-shaking a project.
#[derive(Debug)]
pub struct TreeShakeResult {
    /// Rewritten source per surviving module (unreachable modules are absent).
    pub modules: BTreeMap<String, String>,
    pub stats: TreeShakeStats,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TreeShakeStats {
    pub modules_total: usize,
    pub modules_reachable: usize,
    pub modules_dropped: usize,
    pub dead_exports_removed: usize,
    pub dead_imports_removed: usize,
}

/// An `import` binding seen in a module.
#[derive(Debug, Clone)]
struct Import {
    /// Raw specifier text, e.g. `./math.js`.
    source: String,
    /// The name imported from the target module (`default` / `*` / a name).
    imported: ImportedName,
}

#[derive(Debug, Clone, PartialEq)]
enum ImportedName {
    Named(String),
    Default,
    Namespace,
}

/// Which exports of a module are consumed by the rest of the graph.
#[derive(Debug, Clone, Default)]
enum Used {
    #[default]
    None,
    Names(BTreeSet<String>),
    /// A namespace import (`import * as ns`) or the entry: everything is live.
    All,
}

impl Used {
    fn add(&mut self, name: String) {
        match self {
            Used::All => {}
            Used::None => *self = Used::Names(BTreeSet::from([name])),
            Used::Names(s) => {
                s.insert(name);
            }
        }
    }
    fn mark_all(&mut self) {
        *self = Used::All;
    }
    fn contains(&self, name: &str) -> bool {
        match self {
            Used::All => true,
            Used::None => false,
            Used::Names(s) => s.contains(name),
        }
    }
}

/// Tree-shake `sources` starting from `entry` (a key in `sources`).
pub fn tree_shake(sources: &BTreeMap<String, String>, entry: &str) -> Result<TreeShakeResult, String> {
    if !sources.contains_key(entry) {
        return Err(format!("entry module not found: {entry}"));
    }

    // 1. Parse + resolve the reachable graph from the entry (BFS over imports).
    let mut irs: BTreeMap<String, Op> = BTreeMap::new();
    let mut imports: BTreeMap<String, Vec<Import>> = BTreeMap::new();
    let mut queue: VecDeque<String> = VecDeque::from([entry.to_string()]);
    let mut reachable: BTreeSet<String> = BTreeSet::new();

    while let Some(path) = queue.pop_front() {
        if !reachable.insert(path.clone()) {
            continue;
        }
        let src = sources.get(&path).ok_or_else(|| format!("missing module: {path}"))?;
        let ir = jsir_swc::source_to_ir(src).map_err(|e| format!("{path}: {e}"))?;
        // Graph edges come from every import's source (including bare
        // side-effect imports with no specifiers).
        for source in import_sources(&ir) {
            if let Some(target) = resolve(&path, &source, sources) {
                if !reachable.contains(&target) {
                    queue.push_back(target);
                }
            }
        }
        imports.insert(path.clone(), collect_imports(&ir));
        irs.insert(path, ir);
    }

    // 2. For each reachable module, which of its exports are consumed?
    let mut used: BTreeMap<String, Used> = BTreeMap::new();
    // The entry's own exports are roots (it is the program being run).
    used.entry(entry.to_string()).or_default().mark_all();
    for (path, imps) in &imports {
        for imp in imps {
            let Some(target) = resolve(path, &imp.source, sources) else { continue };
            let u = used.entry(target).or_default();
            match &imp.imported {
                ImportedName::Namespace => u.mark_all(),
                ImportedName::Default => u.add("default".to_string()),
                ImportedName::Named(n) => u.add(n.clone()),
            }
        }
    }

    // 3. Rewrite each reachable module: strip dead exports, DCE, strip dead imports.
    let mut stats = TreeShakeStats {
        modules_total: sources.len(),
        modules_reachable: reachable.len(),
        modules_dropped: sources.len() - reachable.len(),
        ..Default::default()
    };
    let mut out = BTreeMap::new();
    for path in &reachable {
        let mut ir = irs.remove(path).unwrap();
        let module_used = used.get(path).cloned().unwrap_or_default();

        let (pinned, dead_exports) = strip_dead_exports(&mut ir, &module_used);
        stats.dead_exports_removed += dead_exports;

        let (mut ir, _dce) = crate::eliminate_dead_code_with_roots(&ir, &pinned);

        stats.dead_imports_removed += strip_dead_imports(&mut ir);

        let js = jsir_swc::ir_to_source(&ir).map_err(|e| format!("{path}: lift: {e}"))?;
        out.insert(path.clone(), js);
    }

    Ok(TreeShakeResult { modules: out, stats })
}

// ---------------------------------------------------------------------------
// Module-graph extraction
// ---------------------------------------------------------------------------

/// The statement ops of a `jsir.file`'s program body.
fn program_stmts(file: &Op) -> &[Op] {
    file.regions
        .first()
        .and_then(|r| r.blocks.first())
        .and_then(|b| b.ops.first()) // jsir.program
        .and_then(|p| p.regions.first())
        .and_then(|r| r.blocks.first())
        .map(|b| b.ops.as_slice())
        .unwrap_or(&[])
}

fn program_stmts_mut(file: &mut Op) -> Option<&mut Vec<Op>> {
    file.regions
        .first_mut()?
        .blocks
        .first_mut()?
        .ops
        .first_mut()? // jsir.program
        .regions
        .first_mut()?
        .blocks
        .first_mut()
        .map(|b| &mut b.ops)
}

/// The string value of an identifier / string-literal attribute.
fn attr_name(a: &Attr) -> Option<String> {
    match a {
        Attr::Identifier(i) => Some(i.name.clone()),
        Attr::StringLiteralKey(s) => Some(s.value.clone()),
        _ => None,
    }
}

/// Every `import` declaration's source specifier (one per declaration), used to
/// build module-graph edges regardless of whether the import binds any names.
fn import_sources(file: &Op) -> Vec<String> {
    program_stmts(file)
        .iter()
        .filter(|op| op.name == "jsir.import_declaration")
        .filter_map(|op| {
            op.attrs.iter().find_map(|(k, v)| match v {
                Attr::StringLiteralKey(s) if k == "source" => Some(s.value.clone()),
                _ => None,
            })
        })
        .collect()
}

fn collect_imports(file: &Op) -> Vec<Import> {
    let mut out = Vec::new();
    for op in program_stmts(file) {
        if op.name != "jsir.import_declaration" {
            continue;
        }
        let Some(source) = op.attrs.iter().find_map(|(k, v)| match v {
            Attr::StringLiteralKey(s) if k == "source" => Some(s.value.clone()),
            _ => None,
        }) else {
            continue;
        };
        let specs = op.attrs.iter().find_map(|(k, v)| match v {
            Attr::Array(items) if k == "specifiers" => Some(items),
            _ => None,
        });
        for spec in specs.into_iter().flatten() {
            if let Attr::ImportSpecifier(s) = spec {
                let imported = match s.kind {
                    jsir_ir::ImportSpecKind::Default => ImportedName::Default,
                    jsir_ir::ImportSpecKind::Namespace => ImportedName::Namespace,
                    jsir_ir::ImportSpecKind::Named => {
                        let n = s.imported.as_ref().and_then(attr_name).unwrap_or_else(|| s.local.name.clone());
                        ImportedName::Named(n)
                    }
                };
                out.push(Import { source: source.clone(), imported });
            }
        }
    }
    out
}

/// Names a declaration op binds (function/class id, or var declarator symbols).
fn decl_bound_names(decl: &Op) -> Vec<String> {
    match decl.name.as_str() {
        "jsir.function_declaration" | "jsir.class_declaration" => decl
            .attrs
            .iter()
            .find_map(|(k, v)| match v {
                Attr::Identifier(i) if k == "id" => Some(vec![i.name.clone()]),
                _ => None,
            })
            .unwrap_or_default(),
        "jsir.variable_declaration" => {
            let mut names = Vec::new();
            collect_defined_symbol_names(decl, &mut names);
            names
        }
        _ => Vec::new(),
    }
}

fn collect_defined_symbol_names(op: &Op, out: &mut Vec<String>) {
    if op.name == "jsir.variable_declarator" {
        if let Some(syms) = op.trivia.as_ref().and_then(|t| t.defined_symbols.as_ref()) {
            out.extend(syms.iter().map(|s| s.name.clone()));
        }
    }
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                collect_defined_symbol_names(o, out);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rewrites
// ---------------------------------------------------------------------------

/// Strip exports the rest of the graph never imports, returning the set of
/// live-export local names (to pin during DCE) and the count removed.
///
/// - `export <decl>` whose names are all dead: un-exported (the inner
///   declaration is spliced in, so DCE can remove it if locally unused).
/// - `export { a, b }`: dead specifiers dropped; if all dead, the export goes.
/// - `export default` / `export *`: kept conservatively; a named default's name
///   is pinned.
fn strip_dead_exports(file: &mut Op, used: &Used) -> (HashSet<String>, usize) {
    let mut pinned: HashSet<String> = HashSet::new();
    let mut removed = 0;
    let Some(stmts) = program_stmts_mut(file) else { return (pinned, 0) };

    let mut new_stmts: Vec<Op> = Vec::with_capacity(stmts.len());
    for op in std::mem::take(stmts) {
        match op.name.as_str() {
            "jsir.export_named_declaration" => {
                let has_decl = op
                    .regions
                    .first()
                    .and_then(|r| r.blocks.first())
                    .map(|b| !b.ops.is_empty())
                    .unwrap_or(false);
                if has_decl {
                    // `export function f` / `export const K`.
                    let decl = op.regions[0].blocks[0].ops[0].clone();
                    let names = decl_bound_names(&decl);
                    let any_live = names.iter().any(|n| used.contains(n));
                    if any_live {
                        for n in &names {
                            pinned.insert(n.clone());
                        }
                        new_stmts.push(op);
                    } else {
                        // Un-export: keep the declaration, drop the `export`.
                        for inner in op.regions[0].blocks[0].ops.clone() {
                            new_stmts.push(inner);
                        }
                        removed += 1;
                    }
                } else {
                    // `export { a as b, ... }` (possibly a re-export with source).
                    let is_reexport = op.attrs.iter().any(|(k, _)| k == "source");
                    let mut kept_specs: Vec<Attr> = Vec::new();
                    let mut dropped_any = false;
                    for (k, v) in &op.attrs {
                        if k != "specifiers" {
                            continue;
                        }
                        if let Attr::Array(items) = v {
                            for spec in items {
                                if let Attr::ExportSpecifier(s) = spec {
                                    let exported = attr_name(&s.exported).unwrap_or_default();
                                    if used.contains(&exported) {
                                        if !is_reexport {
                                            if let Some(local) = attr_name(&s.local) {
                                                pinned.insert(local);
                                            }
                                        }
                                        kept_specs.push(spec.clone());
                                    } else {
                                        dropped_any = true;
                                    }
                                } else {
                                    kept_specs.push(spec.clone());
                                }
                            }
                        }
                    }
                    if kept_specs.is_empty() && dropped_any {
                        removed += 1; // whole export gone
                    } else {
                        let mut newop = op.clone();
                        for (k, v) in newop.attrs.iter_mut() {
                            if k == "specifiers" {
                                *v = Attr::Array(kept_specs.clone());
                            }
                        }
                        if dropped_any {
                            removed += 1;
                        }
                        new_stmts.push(newop);
                    }
                }
            }
            "jsir.export_default_declaration" => {
                // Pin a named default function/class; keep the statement.
                if let Some(inner) = op.regions.first().and_then(|r| r.blocks.first()).and_then(|b| b.ops.first()) {
                    for n in decl_bound_names(inner) {
                        pinned.insert(n);
                    }
                }
                new_stmts.push(op);
            }
            _ => new_stmts.push(op),
        }
    }
    if let Some(stmts) = program_stmts_mut(file) {
        *stmts = new_stmts;
    }
    (pinned, removed)
}

/// Drop import specifiers whose local binding is unreferenced after DCE. A bare
/// side-effect import (`import './x'`, no specifiers) is always kept.
fn strip_dead_imports(file: &mut Op) -> usize {
    let mut referenced: HashSet<String> = HashSet::new();
    collect_referenced_names(file, &mut referenced);

    let mut removed = 0;
    let Some(stmts) = program_stmts_mut(file) else { return 0 };
    let mut new_stmts = Vec::with_capacity(stmts.len());
    for mut op in std::mem::take(stmts) {
        if op.name != "jsir.import_declaration" {
            new_stmts.push(op);
            continue;
        }
        let mut had_specs = false;
        let mut kept_any = false;
        for (k, v) in op.attrs.iter_mut() {
            if k != "specifiers" {
                continue;
            }
            if let Attr::Array(items) = v {
                if !items.is_empty() {
                    had_specs = true;
                }
                let before = items.len();
                items.retain(|spec| match spec {
                    Attr::ImportSpecifier(s) => referenced.contains(&s.local.name),
                    _ => true,
                });
                removed += before - items.len();
                kept_any = !items.is_empty();
            }
        }
        // A bare side-effect import (never had specifiers) is kept; an import
        // whose every specifier became dead is dropped entirely.
        if had_specs && !kept_any {
            continue;
        }
        new_stmts.push(op);
    }
    if let Some(stmts) = program_stmts_mut(file) {
        *stmts = new_stmts;
    }
    removed
}

fn collect_referenced_names(op: &Op, out: &mut HashSet<String>) {
    if op.name == "jsir.identifier" || op.name == "jsir.identifier_ref" {
        if let Some(sym) = op.trivia.as_ref().and_then(|t| t.referenced_symbol.as_ref()) {
            out.insert(sym.name.clone());
        }
    }
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                collect_referenced_names(o, out);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Module resolution
// ---------------------------------------------------------------------------

/// Resolve a relative specifier against the importer's path to a key in
/// `sources`. Bare specifiers (npm packages) and unresolved paths return `None`
/// (treated as external, left untouched).
fn resolve(importer: &str, specifier: &str, sources: &BTreeMap<String, String>) -> Option<String> {
    if !(specifier.starts_with("./") || specifier.starts_with("../")) {
        return None; // bare/external specifier
    }
    let mut segs: Vec<&str> = importer.split('/').collect();
    segs.pop(); // drop the importer file name -> its directory
    for part in specifier.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                segs.pop();
            }
            p => segs.push(p),
        }
    }
    let base = segs.join("/");
    for cand in [base.clone(), format!("{base}.js"), format!("{base}/index.js")] {
        if sources.contains_key(&cand) {
            return Some(cand);
        }
    }
    None
}
