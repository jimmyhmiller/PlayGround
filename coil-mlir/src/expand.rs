//! The bootstrap expander: surface sugar → core forms (`Val → Val`).
//!
//! Per `mlir-lisp-design/AOT.md`, phase-0 expansion of the irreducible sugar is
//! provided by the compiler (host = Rust); user `defmacro`s are added later via
//! staging. This module implements the fixed rules: `defn`, `if` → `scf.if`,
//! `when`, `cond`, and `->` threading. Everything else passes through, so
//! op-calls and `(: v t)` reach `emit` unchanged (the emitter realizes those).
//!
//! Expansion is structural and runs to a fixpoint: a rule's output is itself
//! re-expanded, so nested sugar (e.g. an `if` inside a `defn` body) expands too.

use crate::value::Val;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, PartialEq)]
pub struct ExpandError(pub String);

impl std::fmt::Display for ExpandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expand error: {}", self.0)
    }
}
impl std::error::Error for ExpandError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ExpandError> {
    Err(ExpandError(msg.into()))
}

const MAX_DEPTH: usize = 512;
static GENSYM: AtomicU64 = AtomicU64::new(0);

fn gensym(prefix: &str) -> Val {
    let n = GENSYM.fetch_add(1, Ordering::Relaxed);
    Val::sym(format!("{prefix}{n}"))
}

/// Expand every top-level form.
pub fn expand_all(forms: &[Val]) -> Result<Vec<Val>, ExpandError> {
    forms.iter().map(|f| expand(f, 0)).collect()
}

/// Expand one form to a fixpoint.
pub fn expand(form: &Val, depth: usize) -> Result<Val, ExpandError> {
    if depth > MAX_DEPTH {
        return err("maximum macro expansion depth exceeded");
    }
    match form {
        Val::List(items) if !items.is_empty() => {
            if let Val::Sym(head) = &items[0] {
                let args = &items[1..];
                let expanded = match head.as_ref() {
                    "defn" => Some(expand_defn(args)?),
                    "if" => Some(expand_if(args)?),
                    "when" => Some(expand_when(args)?),
                    "cond" => Some(expand_cond(args)?),
                    "->" => Some(expand_thread(args)?),
                    _ => None,
                };
                if let Some(out) = expanded {
                    // re-expand the result (it may contain more sugar)
                    return expand(&out, depth + 1);
                }
            }
            // not a macro call: expand children
            let mut out = Vec::with_capacity(items.len());
            for it in items.iter() {
                out.push(expand(it, depth)?);
            }
            Ok(Val::list(out))
        }
        Val::Vec(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items.iter() {
                out.push(expand(it, depth)?);
            }
            Ok(Val::vector(out))
        }
        Val::Map(pairs) => {
            let mut out = Vec::with_capacity(pairs.len());
            for (k, v) in pairs.iter() {
                out.push((expand(k, depth)?, expand(v, depth)?));
            }
            Ok(Val::map(out))
        }
        other => Ok(other.clone()),
    }
}

// --- defn -------------------------------------------------------------------

/// `(defn name [(: p t) …] -> ret  body…)` → a core `func.func` op-form.
/// `(defn name [(: p t) …]          body…)` → void function.
fn expand_defn(args: &[Val]) -> Result<Val, ExpandError> {
    if args.len() < 2 {
        return err("defn needs at least a name and a parameter vector");
    }
    let name = args[0]
        .as_sym()
        .ok_or_else(|| ExpandError("defn name must be a symbol".into()))?
        .to_string();
    let params = match &args[1] {
        Val::Vec(v) => v.to_vec(),
        _ => return err("defn parameters must be a vector"),
    };

    // optional `-> ret`
    let (rets, body_start) = if args.len() > 3 && args[2].as_sym() == Some("->") {
        (vec![args[3].clone()], 4)
    } else {
        (vec![], 2)
    };
    let has_ret = !rets.is_empty();
    let body = args[body_start..].to_vec();
    if body.is_empty() {
        return err("defn needs at least one body expression");
    }

    // parameter types for the function signature
    let mut ptypes = Vec::with_capacity(params.len());
    for p in &params {
        match p {
            Val::List(it) if it.len() == 3 && it[0].as_sym() == Some(":") => {
                ptypes.push(it[2].clone());
            }
            _ => return err("defn parameter must be (: name type)"),
        }
    }

    let body = with_implicit_return(body, has_ret);

    // (block ^entryN [params…] body…)
    let entry = gensym("^entry");
    let mut block = vec![Val::sym("block"), entry, Val::vector(params)];
    block.extend(body);
    let region = Val::list(vec![Val::sym("region"), Val::list(block)]);

    // function_type uses `fn-type` (NOT `->`, which is the threading macro)
    let ftype = Val::list(vec![Val::sym("fn-type"), Val::vector(ptypes), Val::vector(rets)]);
    let attrs = Val::map(vec![
        (Val::keyword("sym_name"), Val::str(name)),
        (Val::keyword("function_type"), ftype),
        (Val::keyword("llvm.emit_c_interface"), Val::Bool(true)),
    ]);

    Ok(Val::list(vec![
        Val::sym("op"),
        Val::str("func.func"),
        Val::keyword("attrs"),
        attrs,
        Val::keyword("regions"),
        Val::vector(vec![region]),
    ]))
}

/// Wrap the last body expression in `func.return` when the function returns a
/// value (unless it already is a return). Void functions are left untouched.
fn with_implicit_return(mut body: Vec<Val>, has_ret: bool) -> Vec<Val> {
    if !has_ret {
        return body;
    }
    let last = body.pop().expect("non-empty body");
    let wrapped = if last.head_sym() == Some("func.return") {
        last
    } else {
        Val::list(vec![Val::sym("func.return"), last])
    };
    body.push(wrapped);
    body
}

// --- if / when / cond -------------------------------------------------------

/// `(if {:result T} c t e)` → value `scf.if`; `(if c t e?)` → statement `scf.if`.
fn expand_if(args: &[Val]) -> Result<Val, ExpandError> {
    let (attr_map, rest): (Option<&[(Val, Val)]>, &[Val]) = match args.first() {
        Some(Val::Map(m)) => (Some(m), &args[1..]),
        _ => (None, args),
    };
    if rest.len() < 2 {
        return err("if needs a condition and a then-branch");
    }
    let cond = rest[0].clone();
    let then = rest[1].clone();
    let els = rest.get(2).cloned();

    let result_ty = attr_map.and_then(|m| {
        m.iter()
            .find(|(k, _)| *k == Val::keyword("result"))
            .map(|(_, v)| v.clone())
    });

    if let Some(rt) = result_ty {
        // value if: each region yields its branch value
        let els = els.ok_or_else(|| ExpandError("if with :result needs an else-branch".into()))?;
        let then_r = region_yielding(Some(then));
        let else_r = region_yielding(Some(els));
        Ok(Val::list(vec![
            Val::sym("scf.if"),
            Val::map(vec![(Val::keyword("results"), Val::vector(vec![rt]))]),
            cond,
            then_r,
            else_r,
        ]))
    } else {
        // statement if: branches run for effect, regions yield nothing
        let then_r = region_stmt(then);
        let mut out = vec![Val::sym("scf.if"), cond, then_r];
        if let Some(e) = els {
            out.push(region_stmt(e));
        }
        Ok(Val::list(out))
    }
}

/// `(region (scf.yield v))` — a value-producing region.
fn region_yielding(v: Option<Val>) -> Val {
    let mut yield_op = vec![Val::sym("scf.yield")];
    if let Some(v) = v {
        yield_op.push(v);
    }
    Val::list(vec![Val::sym("region"), Val::list(yield_op)])
}

/// `(region expr (scf.yield))` — a statement region (effect then empty yield).
fn region_stmt(expr: Val) -> Val {
    Val::list(vec![
        Val::sym("region"),
        expr,
        Val::list(vec![Val::sym("scf.yield")]),
    ])
}

/// `(when c body…)` → statement `scf.if` with a single then-region.
fn expand_when(args: &[Val]) -> Result<Val, ExpandError> {
    if args.is_empty() {
        return err("when needs a condition");
    }
    let cond = args[0].clone();
    let mut region = vec![Val::sym("region")];
    region.extend(args[1..].iter().cloned());
    region.push(Val::list(vec![Val::sym("scf.yield")]));
    Ok(Val::list(vec![Val::sym("scf.if"), cond, Val::list(region)]))
}

/// `(cond t1 e1 t2 e2 … :else e)` → nested statement `if`s.
fn expand_cond(args: &[Val]) -> Result<Val, ExpandError> {
    if args.is_empty() {
        return Ok(Val::list(vec![Val::sym("do")]));
    }
    if args[0] == Val::keyword("else") {
        if args.len() != 2 {
            return err("cond :else needs exactly one expression");
        }
        return Ok(Val::list(vec![Val::sym("do"), args[1].clone()]));
    }
    if args.len() < 2 {
        return err("cond clause needs a test and an expression");
    }
    let test = args[0].clone();
    let expr = args[1].clone();
    let rest = expand_cond(&args[2..])?;
    Ok(Val::list(vec![Val::sym("if"), test, expr, rest]))
}

// --- threading --------------------------------------------------------------

/// `(-> x (f a) (g))` → `(g (f x a))`.
fn expand_thread(args: &[Val]) -> Result<Val, ExpandError> {
    if args.is_empty() {
        return err("-> needs an initial expression");
    }
    let mut acc = args[0].clone();
    for form in &args[1..] {
        acc = match form {
            Val::List(items) if !items.is_empty() => {
                let mut v = vec![items[0].clone(), acc];
                v.extend(items[1..].iter().cloned());
                Val::list(v)
            }
            other => Val::list(vec![other.clone(), acc]),
        };
    }
    Ok(acc)
}
