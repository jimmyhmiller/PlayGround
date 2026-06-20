//! User-defined macros via a small compile-time Lisp.
//!
//! Macros here are *real* macros, not template substitution: a `defmacro` body
//! is evaluated by a compile-time interpreter (this module), with the macro's
//! arguments bound to the **unevaluated** argument forms. The body computes —
//! recursing, building forms programmatically, calling helper functions — and
//! returns a form that is spliced in and re-expanded. This is what lets the
//! "higher-level ways of specifying things" be written in the language itself.
//!
//! Pipeline position: `read → ►expand◄ → parse → check → codegen`.
//!
//! Surface:
//! * `(defmacro name [params... & rest] body...)` — define a macro.
//! * `(def name expr)` — define a compile-time helper value/function.
//! * `(include "path")` — splice another file's macros/definitions in.
//! * `` `form ``, `~x`, `~@xs`, `'x` — quasiquote / unquote / splicing / quote.
//! * Inside a quasiquote, a symbol ending in `#` (e.g. `tmp#`) auto-gensyms to a
//!   fresh name, consistently within that one quasiquote — automatic hygiene for
//!   macro-introduced temporaries.
//! * A top-level `(do ...)` produced by expansion is spliced into several
//!   top-level forms, so one macro call can emit multiple definitions.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::reader::{Sexp, SexpKind};

// ---- compile-time values ------------------------------------------------

#[derive(Clone)]
enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Sym(String),
    Keyword(String),
    List(Vec<Value>),
    Vector(Vec<Value>),
    Closure(Rc<Closure>),
    Builtin(&'static str),
}

struct Closure {
    params: Vec<String>,
    rest: Option<String>,
    body: Vec<Value>,
    env: Env,
    /// The module this closure was defined in (for a macro: the module whose
    /// namespace its template's symbol references resolve in). `None` for the
    /// flat/global default and for helper lambdas.
    module: Option<String>,
}

type Env = Rc<RefCell<Scope>>;

struct Scope {
    vars: HashMap<String, Value>,
    parent: Option<Env>,
}

fn env_child(parent: &Env) -> Env {
    Rc::new(RefCell::new(Scope {
        vars: HashMap::new(),
        parent: Some(parent.clone()),
    }))
}

fn env_lookup(env: &Env, name: &str) -> Option<Value> {
    let s = env.borrow();
    if let Some(v) = s.vars.get(name) {
        Some(v.clone())
    } else if let Some(p) = &s.parent {
        env_lookup(p, name)
    } else {
        None
    }
}

fn env_define(env: &Env, name: &str, val: Value) {
    env.borrow_mut().vars.insert(name.to_string(), val);
}

// ---- public entry point -------------------------------------------------

/// Compile-time view of the target, exposed to macros so conventions/layouts
/// can branch per architecture.
pub struct TargetInfo {
    pub arch: String,
    pub os: String,
    pub triple: String,
    pub pointer_width: i64,
}

/// Expand all macros in a program, returning the macro-free top-level forms.
/// `(include "path")` forms pull in another file's macros and definitions;
/// paths resolve relative to the current working directory, with an include
/// guard so a file is processed at most once.
/// What an `import`'s `:use` clause brings into the importer's scope unqualified.
#[derive(Clone)]
pub enum UseSpec {
    /// `:use *` — all of the target module's definitions.
    All,
    /// `:use [a b c]` — just these names.
    Names(Vec<String>),
}

/// One module's imports: `:as` aliases (qualified access) and `:use` clauses
/// (unqualified access).
#[derive(Default, Clone)]
pub struct ModImports {
    pub aliases: HashMap<String, String>, // alias -> target module
    pub uses: Vec<(String, UseSpec)>,     // (target module, what to bring in)
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

/// Names defined in each module (functions, struct/sum types, sum variants,
/// conventions — *not* externs). Used to qualify macro-template references to
/// their defining module's namespace (referential hygiene).
type DefNames = HashMap<String, HashSet<String>>;

/// Thread-local context for macro-template name resolution: the def table, the
/// import table, and a stack of the module each currently-expanding macro was
/// defined in (top = innermost).
struct MacroCtx {
    stack: Vec<Option<String>>,
    defs: DefNames,
    imports: ImportMap,
    exports: ExportMap,
}
thread_local! {
    static MACRO_CTX: RefCell<Option<MacroCtx>> = const { RefCell::new(None) };
}
fn macro_ctx_init(defs: DefNames, imports: ImportMap, exports: ExportMap) {
    MACRO_CTX
        .with(|c| *c.borrow_mut() = Some(MacroCtx { stack: Vec::new(), defs, imports, exports }));
}
fn macro_ctx_clear() {
    MACRO_CTX.with(|c| *c.borrow_mut() = None);
}
fn macro_push(module: Option<String>) {
    MACRO_CTX.with(|c| {
        if let Some(ctx) = c.borrow_mut().as_mut() {
            ctx.stack.push(module);
        }
    });
}
fn macro_pop() {
    MACRO_CTX.with(|c| {
        if let Some(ctx) = c.borrow_mut().as_mut() {
            ctx.stack.pop();
        }
    });
}
/// Record an (expanded) form's definitions into the live def table, so later
/// macro expansions see macro-generated definitions too.
fn macro_note_defs(form: &Sexp, module: &str) {
    MACRO_CTX.with(|c| {
        if let Some(ctx) = c.borrow_mut().as_mut() {
            scan_defs(form, module, &mut ctx.defs);
        }
    });
}

/// If a macro-template symbol `s` names a definition reachable from the macro's
/// defining module (its own def, a `:use`d module's, or an `alias/x`), return
/// its absolute resolved name. Builtins, locals, and field names aren't in the
/// def table, so they return `None` and stay bare. This is the Clojure
/// syntax-quote rule: template references resolve where the macro was *defined*.
fn macro_qualify(s: &str) -> Result<Option<String>, String> {
    MACRO_CTX.with(|c| {
        let c = c.borrow();
        let ctx = match c.as_ref() {
            Some(ctx) => ctx,
            None => return Ok(None),
        };
        let module = match ctx.stack.last().and_then(|m| m.as_ref()) {
            Some(m) => m,
            None => return Ok(None),
        };
        // explicit `alias/name` — a private target name is a hard error
        if let Some((alias, rest)) = s.split_once('/') {
            match ctx.imports.get(module).and_then(|i| i.aliases.get(alias)) {
                Some(target) => {
                    if !exports(&ctx.exports, target, rest) {
                        return Err(format!("'{rest}' is private to module '{target}'"));
                    }
                    return Ok(Some(format!("{target}.{rest}")));
                }
                None => return Ok(None),
            }
        }
        if ctx.defs.get(module).is_some_and(|d| d.contains(s)) {
            return Ok(Some(format!("{module}.{s}")));
        }
        if let Some(imp) = ctx.imports.get(module) {
            for (target, spec) in &imp.uses {
                let used = match spec {
                    UseSpec::All => true,
                    UseSpec::Names(ns) => ns.iter().any(|n| n == s),
                };
                // `:use` only brings in *exported* names; a private one is skipped.
                if used
                    && ctx.defs.get(target).is_some_and(|d| d.contains(s))
                    && exports(&ctx.exports, target, s)
                {
                    return Ok(Some(format!("{target}.{s}")));
                }
            }
        }
        Ok(None)
    })
}

/// Record the definition names a top-level form introduces under `module`.
fn scan_defs(form: &Sexp, module: &str, defs: &mut DefNames) {
    let items = match &form.kind {
        SexpKind::List(i) => i,
        _ => return,
    };
    let name = items.get(1).and_then(sym_name);
    match items.first().and_then(sym_name) {
        Some("defn") | Some("defstruct") | Some("defcc") => {
            if let Some(n) = name {
                defs.entry(module.to_string()).or_default().insert(n.to_string());
            }
        }
        Some("defsum") => {
            let set = defs.entry(module.to_string()).or_default();
            if let Some(n) = name {
                set.insert(n.to_string());
            }
            for it in items.iter().skip(2) {
                if let SexpKind::List(v) = &it.kind {
                    if let Some(vn) = v.first().and_then(sym_name) {
                        set.insert(vn.to_string());
                    }
                }
            }
        }
        _ => {}
    }
}

/// Read + load the module graph (pass 1), then macro-expand (pass 2). Returns
/// the macro-free top-level forms (each tagged with its module) plus the import
/// table. AST-level name resolution (the qualify pass) happens later in
/// `resolve.rs`; macro-template references are resolved here, in pass 2, against
/// the macro's *defining* module.
pub fn expand_program(
    forms: &[Sexp],
    target: &TargetInfo,
) -> Result<(Vec<TaggedForm>, ImportMap, ExportMap), String> {
    let genv = global_env(target);
    let mut macros: HashMap<String, Value> = HashMap::new();
    let mut raw: Vec<TaggedForm> = Vec::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    let mut imports: ImportMap = HashMap::new();
    let mut defs: DefNames = HashMap::new();
    let mut exports: ExportMap = HashMap::new();
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    // Pass 1 — load: collect raw forms, build the def table, register macros + defs.
    process_forms(forms, &genv, &mut macros, &mut raw, &base, &mut visited, None, &mut imports, &mut defs, &mut exports)?;
    // Pass 2 — expand, with macro-template hygiene resolved against each macro's module.
    macro_ctx_init(defs, imports.clone(), exports.clone());
    let result: Result<Vec<TaggedForm>, String> = (|| {
        let mut out: Vec<TaggedForm> = Vec::new();
        for (form, module) in &raw {
            let expanded = expand_form(form, &macros, &genv)?;
            let start = out.len();
            splice_toplevel_tagged(expanded, &mut out, module);
            // Incremental namespace building (proper-lisp): a form's macro-generated
            // definitions become visible to *later* forms' macro expansions. Since
            // imports load before their users, a module is fully expanded — and its
            // macro-generated defs recorded — before another module's macros run.
            for (sf, m) in &out[start..] {
                if let Some(mm) = m {
                    macro_note_defs(sf, mm);
                }
            }
        }
        Ok(out)
    })();
    macro_ctx_clear();
    Ok((result?, imports, exports))
}

#[allow(clippy::too_many_arguments)]
fn process_forms(
    forms: &[Sexp],
    genv: &Env,
    macros: &mut HashMap<String, Value>,
    out: &mut Vec<TaggedForm>,
    base_dir: &Path,
    visited: &mut HashSet<PathBuf>,
    mut cur_module: Option<String>,
    imports: &mut ImportMap,
    defs: &mut DefNames,
    exports: &mut ExportMap,
) -> Result<(), String> {
    for form in forms {
        match list_head(form) {
            Some("defmacro") => {
                let (name, clo) = make_macro(form, genv, &cur_module)?;
                macros.insert(name, clo);
            }
            Some("def") => eval_toplevel_def(form, genv)?,
            // `(module NAME)` — every following top-level def in this file is
            // namespaced under NAME.
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
                process_import(form, genv, macros, out, base_dir, visited, &cur_module, imports, defs, exports)?;
            }
            // Any other top-level form is collected raw (expanded in pass 2) and
            // its definitions recorded for macro-hygiene resolution.
            _ => {
                if let Some(m) = &cur_module {
                    scan_defs(form, m, defs);
                }
                out.push((form.clone(), cur_module.clone()));
            }
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
    genv: &Env,
    macros: &mut HashMap<String, Value>,
    out: &mut Vec<TaggedForm>,
    base_dir: &Path,
    visited: &mut HashSet<PathBuf>,
    cur_module: &Option<String>,
    imports: &mut ImportMap,
    defs: &mut DefNames,
    exports: &mut ExportMap,
) -> Result<(), String> {
    let items = as_list(form)?;
    let path = match items.get(1).map(|s| &s.kind) {
        Some(SexpKind::Str(p)) => p,
        _ => return Err("import: expected (import \"path\" [:as alias])".into()),
    };
    // optional `:as alias` and `:use *` / `:use [names]`
    let mut alias: Option<String> = None;
    let mut use_spec: Option<UseSpec> = None;
    let mut i = 2;
    while i < items.len() {
        match items.get(i).map(|s| &s.kind) {
            Some(SexpKind::Keyword(k)) if k == "as" => {
                let a = items.get(i + 1).and_then(sym_name).ok_or("import: :as expects an alias symbol")?;
                alias = Some(a.to_string());
                i += 2;
            }
            Some(SexpKind::Keyword(k)) if k == "use" => {
                use_spec = Some(match items.get(i + 1).map(|s| &s.kind) {
                    Some(SexpKind::Sym(s)) if s == "*" => UseSpec::All,
                    Some(SexpKind::Vector(v)) => UseSpec::Names(
                        v.iter().filter_map(sym_name).map(str::to_string).collect(),
                    ),
                    _ => return Err("import: :use expects * or [names]".into()),
                });
                i += 2;
            }
            _ => return Err("import: expected :as alias or :use *|[names]".into()),
        }
    }
    // Resolve relative to the importing file (NOT the cwd) — every import resolves
    // the same way regardless of where `coil` is run.
    let full = base_dir.join(path);
    let canon = full.canonicalize().map_err(|e| format!("import '{}': {e}", full.display()))?;
    let text = std::fs::read_to_string(&canon).map_err(|e| format!("import '{}': {e}", canon.display()))?;
    // The included file's bytes are a *different* source than the one we render
    // diagnostics against, so its spans would point into the wrong file — strip
    // them (a proper multi-file SourceMap is future work).
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
    // Load the imported file once (its own (module …) form tags its defs).
    if visited.insert(canon.clone()) {
        let inc_dir = canon.parent().map(Path::to_path_buf).unwrap_or_else(|| base_dir.to_path_buf());
        process_forms(&inc_forms, genv, macros, out, &inc_dir, visited, None, imports, defs, exports)
            .map_err(|e| trace(e, || format!("import \"{path}\"")))?;
    }
    Ok(())
}

/// A top-level `(do a b c)` is spliced into separate top-level forms, each
/// tagged with the module it came from.
fn splice_toplevel_tagged(form: Sexp, out: &mut Vec<(Sexp, Option<String>)>, module: &Option<String>) {
    if let SexpKind::List(items) = &form.kind {
        if items.first().map(sym_name) == Some(Some("do")) {
            for child in &items[1..] {
                splice_toplevel_tagged(child.clone(), out, module);
            }
            return;
        }
    }
    out.push((form, module.clone()));
}


fn make_macro(form: &Sexp, genv: &Env, module: &Option<String>) -> Result<(String, Value), String> {
    let items = as_list(form)?;
    // (defmacro name [params] body...)
    let name = sym_name(&items[1])
        .ok_or("defmacro: name must be a symbol")?
        .to_string();
    let params_v = match items.get(2).map(|s| &s.kind) {
        Some(SexpKind::Vector(v)) => v,
        _ => return Err(format!("defmacro '{name}': expected parameter vector")),
    };
    let (params, rest) = parse_params(params_v)?;
    let body: Vec<Value> = items[3..].iter().map(sexp_to_value).collect();
    if body.is_empty() {
        return Err(format!("defmacro '{name}': empty body"));
    }
    let clo = Value::Closure(Rc::new(Closure {
        params,
        rest,
        body,
        env: genv.clone(),
        module: module.clone(),
    }));
    Ok((name, clo))
}

fn eval_toplevel_def(form: &Sexp, genv: &Env) -> Result<(), String> {
    let items = as_list(form)?;
    if items.len() != 3 {
        return Err("def: expected (def name expr)".to_string());
    }
    let name = sym_name(&items[1]).ok_or("def: name must be a symbol")?;
    let val = eval(&sexp_to_value(&items[2]), genv)?;
    env_define(genv, name, val);
    Ok(())
}

// ---- macro expansion walk ----------------------------------------------

/// Append a context frame to an error message, building a "stack trace" as the
/// error unwinds (deepest frame first). The leaf message stays on line 1; each
/// enclosing context adds an indented `in …` line below it.
pub fn trace(err: String, frame: impl FnOnce() -> String) -> String {
    format!("{err}\n  in {}", frame())
}

fn expand_form(form: &Sexp, macros: &HashMap<String, Value>, genv: &Env) -> Result<Sexp, String> {
    match &form.kind {
        SexpKind::List(items) if !items.is_empty() => {
            if let Some(name) = sym_name(&items[0]) {
                if let Some(m) = macros.get(name) {
                    let args: Vec<Value> = items[1..].iter().map(sexp_to_value).collect();
                    let frame = || format!("expansion of macro ({name} …)");
                    // While this macro expands, its template's symbol references
                    // resolve in *its* defining module (hygiene), not the use site.
                    let mac_mod = match m {
                        Value::Closure(c) => c.module.clone(),
                        _ => None,
                    };
                    macro_push(mac_mod);
                    let applied = apply(m.clone(), args);
                    macro_pop();
                    let result = applied.map_err(|e| trace(e, frame))?;
                    let result_sexp = value_to_sexp(&result).map_err(|e| trace(e, frame))?;
                    // re-expand the macro's output, attributing any error to this macro
                    return expand_form(&result_sexp, macros, genv).map_err(|e| trace(e, frame));
                }
            }
            // Re-expand children but keep this node's source span, so a list
            // whose head wasn't a macro still points at its origin.
            let children = items
                .iter()
                .map(|c| expand_form(c, macros, genv))
                .collect::<Result<_, _>>()?;
            Ok(Sexp::new(SexpKind::List(children), form.span))
        }
        SexpKind::Vector(items) => {
            let children = items
                .iter()
                .map(|c| expand_form(c, macros, genv))
                .collect::<Result<_, _>>()?;
            Ok(Sexp::new(SexpKind::Vector(children), form.span))
        }
        _ => Ok(form.clone()),
    }
}

// ---- the compile-time evaluator -----------------------------------------

fn eval(form: &Value, env: &Env) -> Result<Value, String> {
    match form {
        Value::Sym(s) => {
            env_lookup(env, s).ok_or_else(|| format!("compile-time: unbound symbol '{s}'"))
        }
        Value::List(items) => {
            if items.is_empty() {
                return Ok(Value::List(vec![]));
            }
            if let Value::Sym(head) = &items[0] {
                match head.as_str() {
                    "quote" => return Ok(items[1].clone()),
                    "quasiquote" => {
                        // Fresh auto-gensym scope per quasiquote: `tmp#` -> a
                        // consistent fresh symbol within this template.
                        let mut gs = HashMap::new();
                        return quasi(&items[1], env, 1, &mut gs);
                    }
                    "if" => {
                        let c = eval(&items[1], env)?;
                        return if truthy(&c) {
                            eval(&items[2], env)
                        } else if items.len() > 3 {
                            eval(&items[3], env)
                        } else {
                            Ok(Value::Bool(false))
                        };
                    }
                    "let" => return eval_let(items, env),
                    "lambda" | "fn" => return make_lambda(items, env),
                    "do" | "begin" => {
                        let mut last = Value::Bool(false);
                        for e in &items[1..] {
                            last = eval(e, env)?;
                        }
                        return Ok(last);
                    }
                    "and" => {
                        let mut last = Value::Bool(true);
                        for e in &items[1..] {
                            last = eval(e, env)?;
                            if !truthy(&last) {
                                return Ok(Value::Bool(false));
                            }
                        }
                        return Ok(last);
                    }
                    "or" => {
                        for e in &items[1..] {
                            let v = eval(e, env)?;
                            if truthy(&v) {
                                return Ok(v);
                            }
                        }
                        return Ok(Value::Bool(false));
                    }
                    _ => {}
                }
            }
            // application
            let f = eval(&items[0], env)?;
            let mut args = Vec::with_capacity(items.len() - 1);
            for a in &items[1..] {
                args.push(eval(a, env)?);
            }
            apply(f, args)
        }
        Value::Vector(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval(it, env)?);
            }
            Ok(Value::Vector(out))
        }
        other => Ok(other.clone()), // Int/Bool/Str/Keyword/Closure/Builtin self-evaluate
    }
}

fn eval_let(items: &[Value], env: &Env) -> Result<Value, String> {
    let binds = match items.get(1) {
        Some(Value::Vector(v)) => v,
        _ => return Err("let: expected binding vector".to_string()),
    };
    if binds.len() % 2 != 0 {
        return Err("let: bindings must be name/value pairs".to_string());
    }
    let local = env_child(env);
    for pair in binds.chunks(2) {
        let name = sym_str(&pair[0]).ok_or("let: binding name must be a symbol")?;
        let v = eval(&pair[1], &local)?;
        env_define(&local, &name, v);
    }
    let mut last = Value::Bool(false);
    for e in &items[2..] {
        last = eval(e, &local)?;
    }
    Ok(last)
}

fn make_lambda(items: &[Value], env: &Env) -> Result<Value, String> {
    let params_v = match items.get(1) {
        Some(Value::Vector(v)) => v,
        _ => return Err("lambda: expected parameter vector".to_string()),
    };
    let (params, rest) = parse_params_v(params_v)?;
    Ok(Value::Closure(Rc::new(Closure {
        params,
        rest,
        body: items[2..].to_vec(),
        env: env.clone(),
        module: None, // helper lambdas inherit the executing macro's module at run time
    })))
}

fn apply(f: Value, args: Vec<Value>) -> Result<Value, String> {
    match f {
        Value::Closure(c) => {
            let local = env_child(&c.env);
            match &c.rest {
                None => {
                    if args.len() != c.params.len() {
                        return Err(format!(
                            "compile-time: expected {} args, got {}",
                            c.params.len(),
                            args.len()
                        ));
                    }
                }
                Some(_) => {
                    if args.len() < c.params.len() {
                        return Err(format!(
                            "compile-time: expected at least {} args, got {}",
                            c.params.len(),
                            args.len()
                        ));
                    }
                }
            }
            for (name, val) in c.params.iter().zip(args.iter()) {
                env_define(&local, name, val.clone());
            }
            if let Some(rest) = &c.rest {
                env_define(&local, rest, Value::List(args[c.params.len()..].to_vec()));
            }
            let mut last = Value::Bool(false);
            for e in &c.body {
                last = eval(e, &local)?;
            }
            Ok(last)
        }
        Value::Builtin(name) => call_builtin(name, args),
        _ => Err("compile-time: value is not callable".to_string()),
    }
}

// ---- quasiquote ---------------------------------------------------------

fn quasi(
    form: &Value,
    env: &Env,
    depth: u32,
    gs: &mut HashMap<String, String>,
) -> Result<Value, String> {
    match form {
        // automatic hygiene: a template symbol ending in `#` becomes a fresh
        // gensym, the same one for every occurrence within this quasiquote.
        Value::Sym(s) if s.len() > 1 && s.ends_with('#') => {
            let g = gs
                .entry(s.clone())
                .or_insert_with(|| auto_gensym(s))
                .clone();
            Ok(Value::Sym(g))
        }
        // Referential hygiene: a template symbol that names a definition of the
        // macro's module is resolved to that module's namespace here. Builtins,
        // locals, and field names aren't definitions, so they stay bare.
        Value::Sym(s) => Ok(Value::Sym(macro_qualify(s)?.unwrap_or_else(|| s.clone()))),
        Value::List(items) => {
            // (unquote x) / (quasiquote x) as the whole form
            if items.len() == 2 {
                if sym_is(&items[0], "unquote") {
                    return if depth == 1 {
                        eval(&items[1], env)
                    } else {
                        Ok(Value::List(vec![
                            Value::Sym("unquote".into()),
                            quasi(&items[1], env, depth - 1, gs)?,
                        ]))
                    };
                }
                if sym_is(&items[0], "quasiquote") {
                    return Ok(Value::List(vec![
                        Value::Sym("quasiquote".into()),
                        quasi(&items[1], env, depth + 1, gs)?,
                    ]));
                }
            }
            Ok(Value::List(quasi_seq(items, env, depth, gs)?))
        }
        Value::Vector(items) => Ok(Value::Vector(quasi_seq(items, env, depth, gs)?)),
        other => Ok(other.clone()),
    }
}

fn auto_gensym(s: &str) -> String {
    let stem = &s[..s.len() - 1]; // drop the trailing '#'
    let n = GENSYM.fetch_add(1, Ordering::Relaxed);
    format!("{stem}__hy{n}")
}

/// Walk a sequence, handling `~@` splicing at the current depth.
fn quasi_seq(
    items: &[Value],
    env: &Env,
    depth: u32,
    gs: &mut HashMap<String, String>,
) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for it in items {
        if let Value::List(inner) = it {
            if inner.len() == 2 && sym_is(&inner[0], "unquote-splicing") {
                if depth == 1 {
                    match eval(&inner[1], env)? {
                        Value::List(xs) | Value::Vector(xs) => out.extend(xs),
                        _ => return Err("unquote-splicing of a non-list".to_string()),
                    }
                } else {
                    out.push(Value::List(vec![
                        Value::Sym("unquote-splicing".into()),
                        quasi(&inner[1], env, depth - 1, gs)?,
                    ]));
                }
                continue;
            }
        }
        out.push(quasi(it, env, depth, gs)?);
    }
    Ok(out)
}

// ---- builtins -----------------------------------------------------------

static GENSYM: AtomicU64 = AtomicU64::new(0);

fn global_env(target: &TargetInfo) -> Env {
    let env = Rc::new(RefCell::new(Scope {
        vars: HashMap::new(),
        parent: None,
    }));
    for name in [
        "+", "-", "*", "mod", "=", "<", ">", "<=", ">=", "list", "vector", "cons", "first", "rest",
        "nth", "count", "empty?", "concat", "not", "symbol", "name", "str", "gensym", "map",
        "list?", "vector?", "symbol?", "number?", "keyword?", "error",
    ] {
        env_define(&env, name, Value::Builtin(name));
    }
    env_define(&env, "true", Value::Bool(true));
    env_define(&env, "false", Value::Bool(false));
    // target as compile-time values, so macros can branch per architecture.
    env_define(&env, "target-arch", Value::Str(target.arch.clone()));
    env_define(&env, "target-os", Value::Str(target.os.clone()));
    env_define(&env, "target-triple", Value::Str(target.triple.clone()));
    env_define(&env, "target-pointer-width", Value::Int(target.pointer_width));
    env
}

fn call_builtin(name: &str, args: Vec<Value>) -> Result<Value, String> {
    let nint = |i: usize| as_int(&args[i]);
    match name {
        "+" => Ok(Value::Int(fold_ints(&args)?.iter().sum())),
        "*" => Ok(Value::Int(fold_ints(&args)?.iter().product())),
        "-" => {
            let v = fold_ints(&args)?;
            match v.as_slice() {
                [] => Err("-: needs at least one argument".into()),
                [x] => Ok(Value::Int(-x)),
                [x, rest @ ..] => Ok(Value::Int(rest.iter().fold(*x, |a, b| a - b))),
            }
        }
        "mod" => Ok(Value::Int(nint(0)? % nint(1)?)),
        "=" => Ok(Value::Bool(args.len() == 2 && val_eq(&args[0], &args[1]))),
        "<" => Ok(Value::Bool(nint(0)? < nint(1)?)),
        ">" => Ok(Value::Bool(nint(0)? > nint(1)?)),
        "<=" => Ok(Value::Bool(nint(0)? <= nint(1)?)),
        ">=" => Ok(Value::Bool(nint(0)? >= nint(1)?)),
        "list" => Ok(Value::List(args)),
        "vector" => Ok(Value::Vector(args)),
        "cons" => {
            let mut v = vec![args[0].clone()];
            v.extend(seq_items(&args[1])?);
            Ok(Value::List(v))
        }
        "first" => seq_items(&args[0])?
            .first()
            .cloned()
            .ok_or_else(|| "first: empty sequence".to_string()),
        "rest" => {
            let items = seq_items(&args[0])?;
            Ok(Value::List(items.iter().skip(1).cloned().collect()))
        }
        "nth" => {
            let items = seq_items(&args[0])?;
            let i = nint(1)? as usize;
            items
                .get(i)
                .cloned()
                .ok_or_else(|| "nth: index out of range".to_string())
        }
        "count" => Ok(Value::Int(seq_items(&args[0])?.len() as i64)),
        "empty?" => Ok(Value::Bool(seq_items(&args[0])?.is_empty())),
        "concat" => {
            let mut out = Vec::new();
            for a in &args {
                out.extend(seq_items(a)?);
            }
            Ok(Value::List(out))
        }
        "not" => Ok(Value::Bool(!truthy(&args[0]))),
        "symbol" => {
            let mut s = String::new();
            for a in &args {
                s.push_str(&text_of(a)?);
            }
            Ok(Value::Sym(s))
        }
        "name" => Ok(Value::Str(text_of(&args[0])?)),
        "str" => {
            let mut s = String::new();
            for a in &args {
                s.push_str(&text_of(a)?);
            }
            Ok(Value::Str(s))
        }
        "gensym" => {
            let n = GENSYM.fetch_add(1, Ordering::Relaxed);
            let prefix = args.first().map(text_of).transpose()?.unwrap_or_else(|| "g".into());
            Ok(Value::Sym(format!("{prefix}__{n}")))
        }
        "map" => {
            let f = args[0].clone();
            let items = seq_items(&args[1])?;
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(apply(f.clone(), vec![it])?);
            }
            Ok(Value::List(out))
        }
        "list?" => Ok(Value::Bool(matches!(args[0], Value::List(_)))),
        "vector?" => Ok(Value::Bool(matches!(args[0], Value::Vector(_)))),
        "symbol?" => Ok(Value::Bool(matches!(args[0], Value::Sym(_)))),
        "number?" => Ok(Value::Bool(matches!(args[0], Value::Int(_)))),
        "keyword?" => Ok(Value::Bool(matches!(args[0], Value::Keyword(_)))),
        // Abort macro expansion with a message. Lets a macro validate its
        // arguments and hard-error (instead of silently emitting wrong code) —
        // the building block for misuse checks like "(defer) only inside (scope)".
        "error" => Err(args
            .first()
            .and_then(|v| text_of(v).ok())
            .unwrap_or_else(|| "macro error".to_string())),
        other => Err(format!("compile-time: unknown builtin '{other}'")),
    }
}

// ---- helpers ------------------------------------------------------------

fn truthy(v: &Value) -> bool {
    !matches!(v, Value::Bool(false))
}

fn as_int(v: &Value) -> Result<i64, String> {
    match v {
        Value::Int(n) => Ok(*n),
        _ => Err("compile-time: expected a number".to_string()),
    }
}

fn fold_ints(args: &[Value]) -> Result<Vec<i64>, String> {
    args.iter().map(as_int).collect()
}

fn seq_items(v: &Value) -> Result<Vec<Value>, String> {
    match v {
        Value::List(xs) | Value::Vector(xs) => Ok(xs.clone()),
        _ => Err("compile-time: expected a list or vector".to_string()),
    }
}

fn text_of(v: &Value) -> Result<String, String> {
    Ok(match v {
        Value::Sym(s) | Value::Str(s) | Value::Keyword(s) => s.clone(),
        Value::Int(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => return Err("compile-time: cannot convert value to text".to_string()),
    })
}

fn val_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Sym(x), Value::Sym(y)) => x == y,
        (Value::Keyword(x), Value::Keyword(y)) => x == y,
        (Value::List(x), Value::List(y)) | (Value::Vector(x), Value::Vector(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| val_eq(p, q))
        }
        _ => false,
    }
}

fn sym_is(v: &Value, name: &str) -> bool {
    matches!(v, Value::Sym(s) if s == name)
}

fn sym_str(v: &Value) -> Option<String> {
    match v {
        Value::Sym(s) => Some(s.clone()),
        _ => None,
    }
}

fn parse_params_v(params: &[Value]) -> Result<(Vec<String>, Option<String>), String> {
    let mut names = Vec::new();
    let mut rest = None;
    let mut i = 0;
    while i < params.len() {
        let s = sym_str(&params[i]).ok_or("parameter must be a symbol")?;
        if s == "&" {
            let r = params
                .get(i + 1)
                .and_then(sym_str)
                .ok_or("expected a name after '&'")?;
            rest = Some(r);
            break;
        }
        names.push(s);
        i += 1;
    }
    Ok((names, rest))
}

// Sexp-level param parsing for defmacro (params come straight from the reader).
fn parse_params(params: &[Sexp]) -> Result<(Vec<String>, Option<String>), String> {
    let vs: Vec<Value> = params.iter().map(sexp_to_value).collect();
    parse_params_v(&vs)
}

// ---- Sexp <-> Value conversion -----------------------------------------

fn sexp_to_value(s: &Sexp) -> Value {
    match &s.kind {
        SexpKind::Int(n) => Value::Int(*n),
        SexpKind::Float(x) => Value::Float(*x),
        SexpKind::Sym(s) => Value::Sym(s.clone()),
        SexpKind::Keyword(k) => Value::Keyword(k.clone()),
        SexpKind::Str(s) => Value::Str(s.clone()),
        SexpKind::List(items) => Value::List(items.iter().map(sexp_to_value).collect()),
        SexpKind::Vector(items) => Value::Vector(items.iter().map(sexp_to_value).collect()),
    }
}

// Macro output is synthesized: it has no bytes in any source, so the forms are
// span-less (`DUMMY`). Diagnostics on macro-generated code therefore fall back
// to the bare message until macro-provenance spans (call site) are plumbed.
fn value_to_sexp(v: &Value) -> Result<Sexp, String> {
    match v {
        Value::Int(n) => Ok(Sexp::int(*n)),
        Value::Float(x) => Ok(Sexp::float(*x)),
        Value::Sym(s) => Ok(Sexp::sym(s.clone())),
        Value::Keyword(k) => Ok(Sexp::keyword(k.clone())),
        Value::List(items) => Ok(Sexp::list(
            items.iter().map(value_to_sexp).collect::<Result<_, _>>()?,
        )),
        Value::Vector(items) => Ok(Sexp::vector(
            items.iter().map(value_to_sexp).collect::<Result<_, _>>()?,
        )),
        Value::Bool(_) => Err("macro produced a boolean where a form was expected".to_string()),
        // A string is a valid form (`SexpKind::Str`): a quasiquoted string
        // literal, or a `(llvm-ir ... "BODY")` body assembled by a macro.
        Value::Str(s) => Ok(Sexp::string(s.clone())),
        Value::Closure(_) | Value::Builtin(_) => {
            Err("macro produced a function where a form was expected".to_string())
        }
    }
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
