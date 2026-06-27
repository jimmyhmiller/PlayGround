//! The module/import **loader**: read the source + the files it imports into one
//! flat, module-tagged list of top-level forms, building the import/export tables.
//!
//! This used to also be a compile-time Lisp (a `Value` interpreter for `defmacro`).
//! That is gone: macros are now ordinary Coil `[Code…] -> Code` functions, expanded
//! by the Stage-3 elaborator (`lib::expand_stage3_macros`) over the forms this
//! loader produces. So this module no longer interprets anything — it just loads.
//!
//! Pipeline position: `read → ►load◄ → expand-macros → parse → check → codegen`.
//!
//! Surface it understands:
//! * `(module NAME)` — namespace the following top-level defs under NAME.
//! * `(export a b c)` — restrict this module's public names to those listed.
//! * `(import "path" [:as alias] [:use *|[names]])` — load another module.
//! Everything else is collected raw (and macro-expanded later), tagged with its
//! module. The prelude (core traits + primitive impls) is auto-loaded into every
//! program.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::reader::{Sexp, SexpKind};

/// The compile target description (host triple, derived in `lib`).
pub struct TargetInfo {
    pub arch: String,
    pub os: String,
    pub triple: String,
    pub pointer_width: i64,
}

/// What an `import`'s `:use` clause brings into the importer's scope unqualified.
#[derive(Clone)]
pub enum UseSpec {
    /// `:use *` — all of the target module's definitions.
    All,
    /// `:use [a b c]` — just these names.
    Names(Vec<String>),
}

/// One module's imports: `:as` aliases (qualified access), `:use` clauses
/// (unqualified access), and `:reexport`ed targets (whose exported names this module
/// re-publishes as its own). Re-export is only consulted for `coil.core`, whose
/// names are auto-referred everywhere — so `(import "x" :reexport)` in the prelude
/// folds `x` into core.
#[derive(Default, Clone)]
pub struct ModImports {
    pub aliases: HashMap<String, String>, // alias -> target module
    pub uses: Vec<(String, UseSpec)>,     // (target module, what to bring in)
    pub reexports: Vec<String>,           // target modules whose exports are re-published
}

/// Per-module import table: importing module name -> its imports.
pub type ImportMap = HashMap<String, ModImports>;

/// Per-module export table: module -> the names it makes public via `(export …)`.
/// A module with *no* `(export …)` form is absent here, meaning **all public**
/// (the default); a present entry restricts visibility to the listed names.
pub type ExportMap = HashMap<String, HashSet<String>>;

/// Whether module `m` exports `name` to other modules (own-module access is
/// always allowed; absence from the table = everything public).
pub fn exports(exports: &ExportMap, m: &str, name: &str) -> bool {
    exports.get(m).map(|set| set.contains(name)).unwrap_or(true)
}

/// A top-level form tagged with the module it belongs to (`None` = flat, for a
/// file with no `(module …)` declaration).
pub type TaggedForm = (Sexp, Option<String>);

/// Load a program: auto-load the prelude, then the source and everything it
/// imports, into one module-tagged form list plus the import/export tables.
/// Performs no macro expansion — that is the Stage-3 elaborator's job.
pub fn load_program(
    forms: &[Sexp],
    _target: &TargetInfo,
) -> Result<(Vec<TaggedForm>, ImportMap, ExportMap), String> {
    let mut out: Vec<TaggedForm> = Vec::new();
    // Loaded modules, by name — an import loads a module once no matter how it is
    // reached (disk path or bundled stdlib), so the prelude's `control` and a user's
    // explicit `(import "lib/control.coil")` don't double-define it.
    let mut loaded: HashSet<String> = HashSet::new();
    let mut imports: ImportMap = HashMap::new();
    let mut exports: ExportMap = HashMap::new();
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    // The prelude (`coil.core`: traits, operator impls, and a re-export of `control`)
    // is compiled into the compiler and auto-loaded into every program — so operators
    // AND everyday control flow (while/for/cond/…) work unimported.
    const PRELUDE: &str = include_str!("prelude.coil");
    let prelude_forms = crate::reader::read_all_unspanned(PRELUDE)?;
    process_forms(&prelude_forms, &mut out, &base, &mut loaded, None, &mut imports, &mut exports)?;
    process_forms(forms, &mut out, &base, &mut loaded, None, &mut imports, &mut exports)?;
    Ok((out, imports, exports))
}

fn process_forms(
    forms: &[Sexp],
    out: &mut Vec<TaggedForm>,
    base_dir: &Path,
    loaded: &mut HashSet<String>,
    mut cur_module: Option<String>,
    imports: &mut ImportMap,
    exports: &mut ExportMap,
) -> Result<(), String> {
    for form in forms {
        match list_head(form) {
            // `(module NAME)` — namespace the following top-level defs under NAME.
            Some("module") => {
                let name = module_name(form)?;
                imports.entry(name.clone()).or_default();
                cur_module = Some(name);
            }
            // `(export a b c)` — restrict this module's public names to those listed.
            Some("export") => {
                if let Some(m) = &cur_module {
                    let set = exports.entry(m.clone()).or_default();
                    for it in &as_list(form)?[1..] {
                        let n = sym_name(it).ok_or("export: names must be symbols")?;
                        set.insert(n.to_string());
                    }
                } else {
                    return Err("export: only valid inside a (module …)".into());
                }
            }
            // `(import "path" [:as alias] [:use *|[names]])` — load another module.
            Some("import") => {
                // A file with no `(module …)` is never name-qualified, so imported
                // (non-macro) names would silently fail to resolve. Require a module
                // declaration before importing, with a clear error instead.
                if cur_module.is_none() {
                    return Err("import requires a (module …) declaration: add one \
                                (e.g. `(module app)`) at the top of the file, before any \
                                imports — otherwise imported types and functions cannot \
                                be resolved"
                        .into());
                }
                process_import(form, out, base_dir, loaded, &cur_module, imports, exports)?;
            }
            // Any other top-level form is collected raw, tagged with its module.
            _ => out.push((form.clone(), cur_module.clone())),
        }
    }
    Ok(())
}

/// Parse the name out of `(module NAME)`.
fn module_name(form: &Sexp) -> Result<String, String> {
    match as_list(form)?.get(1).and_then(sym_name) {
        Some(n) => Ok(n.to_string()),
        None => Err("module: expected a name, e.g. (module math/vec)".into()),
    }
}

/// Scan a freshly-read file's forms for its `(module NAME)` declaration.
fn declared_module(forms: &[Sexp]) -> Option<String> {
    forms.iter().find_map(|f| {
        (list_head(f) == Some("module")).then(|| module_name(f).ok()).flatten()
    })
}

#[allow(clippy::too_many_arguments)]
fn process_import(
    form: &Sexp,
    out: &mut Vec<TaggedForm>,
    base_dir: &Path,
    loaded: &mut HashSet<String>,
    cur_module: &Option<String>,
    imports: &mut ImportMap,
    exports: &mut ExportMap,
) -> Result<(), String> {
    let items = as_list(form)?;
    let path = match items.get(1).map(|s| &s.kind) {
        Some(SexpKind::Str(p)) => p,
        _ => return Err("import: expected (import \"path\" [:as alias])".into()),
    };
    // optional `:as alias`, `:use *` / `:use [names]`, and `:reexport`
    let mut alias: Option<String> = None;
    let mut use_spec: Option<UseSpec> = None;
    let mut reexport = false;
    let mut i = 2;
    while i < items.len() {
        match items.get(i).map(|s| &s.kind) {
            Some(SexpKind::Keyword(k)) if k == "as" => {
                let a = items.get(i + 1).and_then(sym_name).ok_or("import: :as expects an alias symbol")?;
                alias = Some(a.to_string());
                i += 2;
            }
            Some(SexpKind::Keyword(k)) if k == "reexport" => {
                reexport = true;
                i += 1;
            }
            Some(SexpKind::Keyword(k)) if k == "use" => {
                use_spec = Some(match items.get(i + 1).map(|s| &s.kind) {
                    Some(SexpKind::Sym(s)) if s == "*" => UseSpec::All,
                    Some(SexpKind::Vector(v)) => {
                        UseSpec::Names(v.iter().filter_map(sym_name).map(str::to_string).collect())
                    }
                    _ => return Err("import: :use expects * or [names]".into()),
                });
                i += 2;
            }
            _ => return Err("import: expected :as alias, :use *|[names], or :reexport".into()),
        }
    }
    // Resolve the source: a real file relative to the importing file wins (so disk
    // edits during development take effect); otherwise the BUNDLED stdlib by file
    // basename, so `(import "control.coil")` / `(import "objc.coil")` work from
    // anywhere with no `../../lib/` path.
    let basename = Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or(path);
    let (text, inc_dir) = match base_dir.join(path).canonicalize() {
        Ok(canon) => {
            let t = std::fs::read_to_string(&canon).map_err(|e| format!("import '{}': {e}", canon.display()))?;
            let dir = canon.parent().map(Path::to_path_buf).unwrap_or_else(|| base_dir.to_path_buf());
            (t, dir)
        }
        Err(_) => match crate::stdlib::lookup(basename) {
            // A bundled module's own nested imports resolve the same way (disk, then
            // bundle), so `base_dir` carries through.
            Some(src) => (src.to_string(), base_dir.to_path_buf()),
            None => return Err(format!(
                "import '{path}': not found (no such file, and not a bundled stdlib module)"
            )),
        },
    };
    // The included file's bytes are a *different* source than the one we render
    // diagnostics against, so its spans would point into the wrong file — strip them.
    let inc_forms = crate::reader::read_all_unspanned(&text)?;
    let modname = declared_module(&inc_forms)
        .ok_or_else(|| format!("import '{}': file has no (module …) declaration", path))?;
    // Record the import in the importing module's table. `:as` adds a qualified
    // alias (defaulting to the module's own name); `:use` brings names in bare.
    let importer = cur_module.clone().unwrap_or_default();
    let imp = imports.entry(importer).or_default();
    imp.aliases.insert(alias.unwrap_or_else(|| modname.clone()), modname.clone());
    if let Some(spec) = use_spec {
        imp.uses.push((modname.clone(), spec));
    }
    if reexport {
        imp.reexports.push(modname.clone());
    }
    // Load the module once, by name — no matter which path/bundle reached it.
    if loaded.insert(modname.clone()) {
        process_forms(&inc_forms, out, &inc_dir, loaded, None, imports, exports)
            .map_err(|e| trace(e, || format!("import \"{path}\"")))?;
    }
    Ok(())
}

/// Append a one-line frame to an error message (used for import traces).
pub fn trace(err: String, frame: impl FnOnce() -> String) -> String {
    format!("{err}\n  in {}", frame())
}

fn as_list(s: &Sexp) -> Result<&[Sexp], String> {
    match &s.kind {
        SexpKind::List(items) => Ok(items),
        _ => Err("expected a list".to_string()),
    }
}

fn list_head(s: &Sexp) -> Option<&str> {
    match &s.kind {
        SexpKind::List(items) => items.first().and_then(sym_name),
        _ => None,
    }
}

fn sym_name(s: &Sexp) -> Option<&str> {
    match &s.kind {
        SexpKind::Sym(s) => Some(s.as_str()),
        _ => None,
    }
}
