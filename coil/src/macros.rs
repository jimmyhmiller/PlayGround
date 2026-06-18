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

use crate::reader::Sexp;

// ---- compile-time values ------------------------------------------------

#[derive(Clone)]
enum Value {
    Int(i64),
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
pub fn expand_program(forms: &[Sexp], target: &TargetInfo) -> Result<Vec<Sexp>, String> {
    let genv = global_env(target);
    let mut macros: HashMap<String, Value> = HashMap::new();
    let mut out: Vec<Sexp> = Vec::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    process_forms(forms, &genv, &mut macros, &mut out, &base, &mut visited)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn process_forms(
    forms: &[Sexp],
    genv: &Env,
    macros: &mut HashMap<String, Value>,
    out: &mut Vec<Sexp>,
    base_dir: &Path,
    visited: &mut HashSet<PathBuf>,
) -> Result<(), String> {
    for form in forms {
        match list_head(form) {
            Some("defmacro") => {
                let (name, clo) = make_macro(form, genv)?;
                macros.insert(name, clo);
            }
            Some("def") => eval_toplevel_def(form, genv)?,
            Some("include") => process_include(form, genv, macros, out, base_dir, visited)?,
            _ => {
                let expanded = expand_form(form, macros, genv)?;
                splice_toplevel(expanded, out);
            }
        }
    }
    Ok(())
}

fn process_include(
    form: &Sexp,
    genv: &Env,
    macros: &mut HashMap<String, Value>,
    out: &mut Vec<Sexp>,
    base_dir: &Path,
    visited: &mut HashSet<PathBuf>,
) -> Result<(), String> {
    let items = as_list(form)?;
    let path = match items.get(1) {
        Some(Sexp::Str(p)) => p,
        _ => return Err("include: expected a string path, e.g. (include \"lib/x.coil\")".into()),
    };
    let full = base_dir.join(path);
    let canon = full
        .canonicalize()
        .map_err(|e| format!("include '{}': {e}", full.display()))?;
    if !visited.insert(canon.clone()) {
        return Ok(()); // already included
    }
    let text = std::fs::read_to_string(&canon)
        .map_err(|e| format!("include '{}': {e}", canon.display()))?;
    let inc_forms = crate::reader::read_all(&text)?;
    let inc_dir = canon
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| base_dir.to_path_buf());
    process_forms(&inc_forms, genv, macros, out, &inc_dir, visited)
}

/// A top-level `(do a b c)` is spliced into separate top-level forms.
fn splice_toplevel(form: Sexp, out: &mut Vec<Sexp>) {
    if let Sexp::List(items) = &form {
        if items.first().map(sym_name) == Some(Some("do")) {
            for child in &items[1..] {
                splice_toplevel(child.clone(), out);
            }
            return;
        }
    }
    out.push(form);
}

fn make_macro(form: &Sexp, genv: &Env) -> Result<(String, Value), String> {
    let items = as_list(form)?;
    // (defmacro name [params] body...)
    let name = sym_name(&items[1])
        .ok_or("defmacro: name must be a symbol")?
        .to_string();
    let params_v = match items.get(2) {
        Some(Sexp::Vector(v)) => v,
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

fn expand_form(form: &Sexp, macros: &HashMap<String, Value>, genv: &Env) -> Result<Sexp, String> {
    match form {
        Sexp::List(items) if !items.is_empty() => {
            if let Some(name) = sym_name(&items[0]) {
                if let Some(m) = macros.get(name) {
                    let args: Vec<Value> = items[1..].iter().map(sexp_to_value).collect();
                    let result = apply(m.clone(), args)?;
                    let result_sexp = value_to_sexp(&result)?;
                    return expand_form(&result_sexp, macros, genv); // re-expand
                }
            }
            let children = items
                .iter()
                .map(|c| expand_form(c, macros, genv))
                .collect::<Result<_, _>>()?;
            Ok(Sexp::List(children))
        }
        Sexp::Vector(items) => {
            let children = items
                .iter()
                .map(|c| expand_form(c, macros, genv))
                .collect::<Result<_, _>>()?;
            Ok(Sexp::Vector(children))
        }
        other => Ok(other.clone()),
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
        "list?", "vector?", "symbol?", "number?", "keyword?",
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
    match s {
        Sexp::Int(n) => Value::Int(*n),
        Sexp::Sym(s) => Value::Sym(s.clone()),
        Sexp::Keyword(k) => Value::Keyword(k.clone()),
        Sexp::Str(s) => Value::Str(s.clone()),
        Sexp::List(items) => Value::List(items.iter().map(sexp_to_value).collect()),
        Sexp::Vector(items) => Value::Vector(items.iter().map(sexp_to_value).collect()),
    }
}

fn value_to_sexp(v: &Value) -> Result<Sexp, String> {
    match v {
        Value::Int(n) => Ok(Sexp::Int(*n)),
        Value::Sym(s) => Ok(Sexp::Sym(s.clone())),
        Value::Keyword(k) => Ok(Sexp::Keyword(k.clone())),
        Value::List(items) => Ok(Sexp::List(
            items.iter().map(value_to_sexp).collect::<Result<_, _>>()?,
        )),
        Value::Vector(items) => Ok(Sexp::Vector(
            items.iter().map(value_to_sexp).collect::<Result<_, _>>()?,
        )),
        Value::Bool(_) => Err("macro produced a boolean where a form was expected".to_string()),
        Value::Str(_) => Err("macro produced a string where a form was expected".to_string()),
        Value::Closure(_) | Value::Builtin(_) => {
            Err("macro produced a function where a form was expected".to_string())
        }
    }
}

fn as_list(s: &Sexp) -> Result<&[Sexp], String> {
    match s {
        Sexp::List(items) => Ok(items),
        _ => Err("expected a list".to_string()),
    }
}

fn list_head(s: &Sexp) -> Option<&str> {
    match s {
        Sexp::List(items) => items.first().and_then(sym_name),
        _ => None,
    }
}

fn sym_name(s: &Sexp) -> Option<&str> {
    match s {
        Sexp::Sym(s) => Some(s.as_str()),
        _ => None,
    }
}
