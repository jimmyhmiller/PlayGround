//! An R7RS-flavored Scheme frontend, built ENTIRELY on the core's public API.
//!
//! The whole point is the library/language split (see
//! `../docs/LIBRARY_LANGUAGE_SPLIT.md`). Everything reusable — the IR, the
//! execution tiers, the value model, GC, dispatch, the runtime — lives in
//! `microlang` (the core) and is untouched here. This crate contains ONLY
//! Scheme policy:
//!
//!   * a Scheme **reader** (its own lexical syntax: `#t`/`#f`, `'`) that builds
//!     core `Val` via the core's public constructors, and
//!   * a **desugarer** that rewrites Scheme special forms (`define`, `lambda`,
//!     `begin`, `let`, `cond`, ...) into the core's small form language
//!     (`def`, `fn`, `do`, `let`, `if`).
//!
//! It then hands the result to any core `CodeSpace` (interpreter, closure
//! compiler, or bytecode VM) — Scheme runs on all of them for free. Nothing
//! Scheme-specific is in the core; the core has no idea Scheme exists.
//!
//! This is a starting slice (define/lambda/let/cond/quote/aliases). Hygienic
//! macros, the numeric tower, and call/cc are the axes that will genuinely
//! stress the core next; per the operating procedure, each addition asks "core
//! mechanism or Scheme policy?" and defaults to living here.

use std::collections::HashMap;

use microlang::{CodeSpace, Runtime, Val, ValueModel};

mod syntax_rules;
use syntax_rules::SyntaxRules;

/// Format a core value in R7RS `write`/`display` notation. This is a frontend
/// concern (Scheme's external representation), distinct from the core's debug
/// `print`: booleans are `#t`/`#f`, the empty list is `()`.
pub fn write_value<M: ValueModel>(rt: &Runtime<M>, bits: u64) -> String {
    match rt.decode(bits) {
        Val::Int(i) => i.to_string(),
        Val::Float(f) => format!("{f}"),
        Val::Bool(b) => if b { "#t" } else { "#f" }.to_string(),
        Val::Nil => "()".to_string(),
        Val::Sym(s) => rt.sym_name(s).to_string(),
        Val::Ref(_) => {
            if rt.as_cons(bits).is_some() {
                let items = rt.list_to_vec(bits);
                let inner: Vec<String> = items.iter().map(|&x| write_value(rt, x)).collect();
                format!("({})", inner.join(" "))
            } else {
                "#<object>".to_string()
            }
        }
    }
}

/// Read a whole Scheme program into core `Val` forms.
pub fn read<M: ValueModel>(rt: &mut Runtime<M>, src: &str) -> Vec<u64> {
    let toks = tokenize(src);
    let mut p = 0;
    let mut out = Vec::new();
    while p < toks.len() {
        let (v, np) = read_form(rt, &toks, p);
        out.push(v);
        p = np;
    }
    out
}

/// Read, macro-expand (`syntax-rules`), desugar, and evaluate a program on the
/// given core execution tier. Macro expansion is a pure frontend pass; the core
/// never sees a `syntax-rules` macro.
pub fn run<M: ValueModel>(rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
    let forms = read(rt, src);
    let mut macros: HashMap<u32, SyntaxRules> = HashMap::new();
    let mut last = rt.encode(Val::Nil);
    for f in forms {
        // `(define-syntax name (syntax-rules ...))` registers a macro; it is a
        // compile-time definition with no runtime form.
        if let Some((name, sr_form)) = as_define_syntax(rt, f) {
            let sr = syntax_rules::parse(rt, sr_form);
            macros.insert(name, sr);
            continue;
        }
        let expanded = expand(rt, &macros, f);
        let core = desugar(rt, expanded);
        last = rt.eval_top(cs, core);
    }
    last
}

fn as_define_syntax<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<(u32, u64)> {
    if head_sym(rt, form).as_deref() == Some("define-syntax") {
        let items = rt.list_to_vec(form);
        if let Val::Sym(name) = rt.decode(items[1]) {
            return Some((name, items[2]));
        }
    }
    None
}

/// Fully expand `syntax-rules` macro uses in a form (recursively), leaving
/// everything else for `desugar`.
fn expand<M: ValueModel>(rt: &mut Runtime<M>, macros: &HashMap<u32, SyntaxRules>, form: u64) -> u64 {
    if rt.as_cons(form).is_none() {
        return form;
    }
    if let Some(head) = head_sym(rt, form) {
        if head == "quote" {
            return form; // never expand inside quote
        }
        if let Val::Sym(h) = rt.decode(rt.list_to_vec(form)[0]) {
            if let Some(sr) = macros.get(&h) {
                if let Some(expansion) = syntax_rules::apply(rt, sr, form) {
                    return expand(rt, macros, expansion); // re-expand the result
                }
            }
        }
    }
    let items = rt.list_to_vec(form);
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        out.push(expand(rt, macros, it));
    }
    rt.vec_to_list(&out)
}

// ── reader (Scheme syntax → core Val) ───────────────────────

enum Tok {
    L,
    R,
    Quote,
    Atom(String),
}

fn tokenize(src: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let cs: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        match c {
            c if c.is_whitespace() => i += 1,
            ';' => {
                while i < cs.len() && cs[i] != '\n' {
                    i += 1;
                }
            }
            '(' | '[' => {
                out.push(Tok::L);
                i += 1;
            }
            ')' | ']' => {
                out.push(Tok::R);
                i += 1;
            }
            '\'' => {
                out.push(Tok::Quote);
                i += 1;
            }
            _ => {
                let start = i;
                while i < cs.len()
                    && !cs[i].is_whitespace()
                    && !matches!(cs[i], '(' | ')' | '[' | ']' | '\'' | ';')
                {
                    i += 1;
                }
                out.push(Tok::Atom(cs[start..i].iter().collect()));
            }
        }
    }
    out
}

fn read_form<M: ValueModel>(rt: &mut Runtime<M>, toks: &[Tok], p: usize) -> (u64, usize) {
    match &toks[p] {
        Tok::L => {
            let mut items = Vec::new();
            let mut q = p + 1;
            while !matches!(toks.get(q), Some(Tok::R)) {
                assert!(q < toks.len(), "scheme reader: unbalanced (");
                let (v, nq) = read_form(rt, toks, q);
                items.push(v);
                q = nq;
            }
            (rt.vec_to_list(&items), q + 1)
        }
        Tok::R => panic!("scheme reader: unexpected )"),
        Tok::Quote => {
            let (v, nq) = read_form(rt, toks, p + 1);
            let q = rt.intern("quote");
            let qs = rt.encode(Val::Sym(q));
            (rt.vec_to_list(&[qs, v]), nq)
        }
        Tok::Atom(a) => (read_atom(rt, a), p + 1),
    }
}

fn read_atom<M: ValueModel>(rt: &mut Runtime<M>, a: &str) -> u64 {
    let v = if a == "#t" {
        Val::Bool(true)
    } else if a == "#f" {
        Val::Bool(false)
    } else if a == "nil" || a == "'()" {
        Val::Nil
    } else if let Ok(i) = a.parse::<i64>() {
        Val::Int(i)
    } else if let Ok(f) = a.parse::<f64>() {
        Val::Float(f)
    } else {
        Val::Sym(rt.intern(a))
    };
    rt.encode(v)
}

// ── desugar (Scheme forms → core forms) ─────────────────────

fn head_sym<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<String> {
    let (h, _) = rt.as_cons(form)?;
    if let Val::Sym(s) = rt.decode(h) {
        Some(rt.sym_name(s).to_string())
    } else {
        None
    }
}

fn sym<M: ValueModel>(rt: &mut Runtime<M>, name: &str) -> u64 {
    let s = rt.intern(name);
    rt.encode(Val::Sym(s))
}

/// Desugar one Scheme form into the core form language.
pub fn desugar<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let Some(head) = head_sym(rt, form) else {
        // atom, or a list whose head is not a symbol (an application)
        return if rt.as_cons(form).is_some() {
            desugar_app(rt, form)
        } else {
            form
        };
    };
    match head.as_str() {
        "define" => desugar_define(rt, form),
        "lambda" => {
            // (lambda (params) body...) -> (fn (params) body'...)
            let items = rt.list_to_vec(form);
            let params = items[1];
            let mut out = vec![sym(rt, "fn"), params];
            for &b in &items[2..] {
                out.push(desugar(rt, b));
            }
            rt.vec_to_list(&out)
        }
        "begin" => rebuild_head(rt, "do", form),
        "if" => rebuild_head(rt, "if", form),
        "let" => {
            // named let `(let loop (bindings) body)` vs ordinary parallel let
            if matches!(rt.decode(nth(rt, form, 1)), Val::Sym(_)) {
                desugar_named_let(rt, form)
            } else {
                desugar_let_parallel(rt, form)
            }
        }
        "let*" => desugar_let_star(rt, form),
        "letrec" => desugar_letrec(rt, form),
        "set!" => rebuild_head(rt, "set!", form), // core has `set!`; only the value desugars
        "cond" => desugar_cond(rt, form),
        "quote" => form, // keep the datum verbatim
        // variadic arithmetic folds into the core's binary prims
        "+" => fold(rt, form, "+", 0),
        "*" => fold(rt, form, "*", 1),
        "-" => desugar_minus(rt, form),
        // comparison ops in terms of the core's single `<`
        ">" => {
            let a = desugar(rt, nth(rt, form, 1));
            let b = desugar(rt, nth(rt, form, 2));
            let lt = sym(rt, "<");
            list(rt, &[lt, b, a])
        }
        "<=" => cmp_or_eq(rt, form, false),
        ">=" => cmp_or_eq(rt, form, true),
        "and" => {
            let args = desugared_args(rt, form);
            build_and(rt, &args)
        }
        "or" => {
            let args = desugared_args(rt, form);
            build_or(rt, &args)
        }
        "when" => desugar_when(rt, form, true),
        "unless" => desugar_when(rt, form, false),
        "case" => desugar_case(rt, form),
        _ => desugar_app(rt, form),
    }
}

// ── small builders ──────────────────────────────────────────

fn list<M: ValueModel>(rt: &mut Runtime<M>, items: &[u64]) -> u64 {
    rt.vec_to_list(items)
}
fn nth<M: ValueModel>(rt: &Runtime<M>, form: u64, i: usize) -> u64 {
    rt.list_to_vec(form)[i]
}
fn desugared_args<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> Vec<u64> {
    let items = rt.list_to_vec(form);
    items[1..].iter().map(|&a| desugar(rt, a)).collect()
}

fn fold<M: ValueModel>(rt: &mut Runtime<M>, form: u64, op: &str, identity: i64) -> u64 {
    let args = desugared_args(rt, form);
    if args.is_empty() {
        return rt.encode(Val::Int(identity));
    }
    let mut acc = args[0];
    for &a in &args[1..] {
        let o = sym(rt, op);
        acc = list(rt, &[o, acc, a]);
    }
    acc
}

fn desugar_minus<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let args = desugared_args(rt, form);
    if args.len() == 1 {
        // (- a) is negation: (- 0 a)
        let zero = rt.encode(Val::Int(0));
        let o = sym(rt, "-");
        return list(rt, &[o, zero, args[0]]);
    }
    let mut acc = args[0];
    for &a in &args[1..] {
        let o = sym(rt, "-");
        acc = list(rt, &[o, acc, a]);
    }
    acc
}

fn cmp_or_eq<M: ValueModel>(rt: &mut Runtime<M>, form: u64, ge: bool) -> u64 {
    // (<= a b) => (if (< b a) #f #t) ; (>= a b) => (if (< a b) #f #t)
    let a = desugar(rt, nth(rt, form, 1));
    let b = desugar(rt, nth(rt, form, 2));
    let lt = sym(rt, "<");
    let test = if ge { list(rt, &[lt, a, b]) } else { list(rt, &[lt, b, a]) };
    let f = rt.encode(Val::Bool(false));
    let t = rt.encode(Val::Bool(true));
    let iff = sym(rt, "if");
    list(rt, &[iff, test, f, t])
}

fn build_and<M: ValueModel>(rt: &mut Runtime<M>, args: &[u64]) -> u64 {
    match args.len() {
        0 => rt.encode(Val::Bool(true)),
        1 => args[0],
        _ => {
            let rest = build_and(rt, &args[1..]);
            let f = rt.encode(Val::Bool(false));
            let iff = sym(rt, "if");
            list(rt, &[iff, args[0], rest, f])
        }
    }
}

fn build_or<M: ValueModel>(rt: &mut Runtime<M>, args: &[u64]) -> u64 {
    match args.len() {
        0 => rt.encode(Val::Bool(false)),
        1 => args[0],
        _ => {
            // ((fn (%t) (if %t %t rest)) args[0]) — evaluate args[0] once
            let rest = build_or(rt, &args[1..]);
            let t = sym(rt, "%or-t");
            let params = list(rt, &[t]);
            let iff = sym(rt, "if");
            let body = list(rt, &[iff, t, t, rest]);
            let fnsym = sym(rt, "fn");
            let lam = list(rt, &[fnsym, params, body]);
            list(rt, &[lam, args[0]])
        }
    }
}

fn desugar_when<M: ValueModel>(rt: &mut Runtime<M>, form: u64, when: bool) -> u64 {
    let items = rt.list_to_vec(form);
    let cond = desugar(rt, items[1]);
    let mut body = vec![sym(rt, "do")];
    for &b in &items[2..] {
        body.push(desugar(rt, b));
    }
    let body = list(rt, &body);
    let nil = rt.encode(Val::Nil);
    let iff = sym(rt, "if");
    if when {
        list(rt, &[iff, cond, body, nil])
    } else {
        list(rt, &[iff, cond, nil, body])
    }
}

fn desugar_case<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    // (case key (datums body...)... (else body...)) ->
    //   ((lambda (%k) (if (or (= %k 'd)...) (begin body...) ...)) key)
    let items = rt.list_to_vec(form);
    let key = items[1];
    let kvar = sym(rt, "%case-key");
    let mut chain = rt.encode(Val::Nil);
    for &clause in items[2..].iter().rev() {
        let parts = rt.list_to_vec(clause);
        let mut body = vec![sym(rt, "begin")];
        body.extend_from_slice(&parts[1..]);
        let body = list(rt, &body);
        if head_sym(rt, clause).as_deref() == Some("else") {
            chain = body;
        } else {
            let mut ortest = vec![sym(rt, "or")];
            for &d in &rt.list_to_vec(parts[0]) {
                let q = sym(rt, "quote");
                let quoted = list(rt, &[q, d]);
                let eq = sym(rt, "=");
                ortest.push(list(rt, &[eq, kvar, quoted]));
            }
            let test = list(rt, &ortest);
            let iff = sym(rt, "if");
            chain = list(rt, &[iff, test, body, chain]);
        }
    }
    let params = list(rt, &[kvar]);
    let lambdasym = sym(rt, "lambda");
    let lam = list(rt, &[lambdasym, params, chain]);
    let app = list(rt, &[lam, key]);
    desugar(rt, app) // re-desugar the sugar we just emitted (or/if/lambda)
}

/// Rewrite the head symbol to `name` and desugar the argument forms.
fn rebuild_head<M: ValueModel>(rt: &mut Runtime<M>, name: &str, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let mut out = vec![sym(rt, name)];
    for &it in &items[1..] {
        out.push(desugar(rt, it));
    }
    rt.vec_to_list(&out)
}

/// An application: desugar every element, renaming a few Scheme names to their
/// core prims (`display`->`println`, `car`->`first`, ...).
fn desugar_app<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let mut out = Vec::with_capacity(items.len());
    for (i, &it) in items.iter().enumerate() {
        if i == 0 {
            if let Val::Sym(s) = rt.decode(it) {
                if let Some(core) = alias(rt.sym_name(s)) {
                    out.push(sym(rt, core));
                    continue;
                }
            }
        }
        out.push(desugar(rt, it));
    }
    rt.vec_to_list(&out)
}

fn alias(name: &str) -> Option<&'static str> {
    Some(match name {
        "display" => "println",
        "car" => "first",
        "cdr" => "rest",
        "null?" => "nil?",
        "equal?" => "=", // the core's `=` is structural equality
        // Scheme's call/cc maps to the core's escape-continuation mechanism
        // (escape-only for now — full multi-shot is a separate frontier).
        "call/cc" => "%callec",
        "call-with-current-continuation" => "%callec",
        _ => return None,
    })
}

fn desugar_define<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let target = items[1];
    if let Some((name, params)) = rt.as_cons(target) {
        // (define (name params...) body...) -> (def name (fn (params...) body'...))
        let mut fn_form = vec![sym(rt, "fn"), params];
        for &b in &items[2..] {
            fn_form.push(desugar(rt, b));
        }
        let lam = rt.vec_to_list(&fn_form);
        let def = sym(rt, "def");
        rt.vec_to_list(&[def, name, lam])
    } else {
        // (define name val) -> (def name val')
        let val = desugar(rt, items[2]);
        let def = sym(rt, "def");
        rt.vec_to_list(&[def, target, val])
    }
}

fn desugar_let_parallel<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    // R7RS `let` is PARALLEL: inits are evaluated in the OUTER scope. The classic
    // desugar is application of a lambda —
    //   (let ((a va) (b vb)) body...) -> ((lambda (a b) body'...) va' vb')
    // so va'/vb' cannot see a/b. (The core's own `let` is sequential = `let*`.)
    let items = rt.list_to_vec(form);
    let pairs = rt.list_to_vec(items[1]);
    let mut params = Vec::new();
    let mut args = Vec::new();
    for &pair in &pairs {
        let kv = rt.list_to_vec(pair);
        params.push(kv[0]);
        args.push(desugar(rt, kv[1]));
    }
    let params_list = rt.vec_to_list(&params);
    let mut fn_form = vec![sym(rt, "fn"), params_list];
    for &b in &items[2..] {
        fn_form.push(desugar(rt, b));
    }
    let lam = rt.vec_to_list(&fn_form);
    let mut app = vec![lam];
    app.extend(args);
    rt.vec_to_list(&app)
}

fn desugar_letrec<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    // (letrec ((f fe) (g ge)) body...) ->
    //   (let (f nil g nil) (set! f fe') (set! g ge') body'...)
    // Core `let` gives each binding a shared mutable cell; `set!` ties the knot,
    // so the closures see one another (mutual recursion).
    let items = rt.list_to_vec(form);
    let pairs = rt.list_to_vec(items[1]);
    let nil = rt.encode(Val::Nil);
    let mut binds = Vec::new();
    let mut sets = Vec::new();
    for &pair in &pairs {
        let kv = rt.list_to_vec(pair);
        binds.push(kv[0]);
        binds.push(nil);
        let set_s = sym(rt, "set!");
        let ve = desugar(rt, kv[1]);
        sets.push(list(rt, &[set_s, kv[0], ve]));
    }
    let bindlist = list(rt, &binds);
    let let_s = sym(rt, "let");
    let mut out = vec![let_s, bindlist];
    out.extend(sets);
    for &b in &items[2..] {
        out.push(desugar(rt, b));
    }
    list(rt, &out)
}

fn desugar_named_let<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    // (let loop ((i 0) (acc 0)) body) ->
    //   (letrec ((loop (lambda (i acc) body))) (loop 0 0))
    let items = rt.list_to_vec(form);
    let name = items[1];
    let pairs = rt.list_to_vec(items[2]);
    let mut params = Vec::new();
    let mut initargs = Vec::new();
    for &pair in &pairs {
        let kv = rt.list_to_vec(pair);
        params.push(kv[0]);
        initargs.push(kv[1]);
    }
    let paramlist = list(rt, &params);
    let lambda_s = sym(rt, "lambda");
    let mut lam = vec![lambda_s, paramlist];
    lam.extend_from_slice(&items[3..]);
    let lam = list(rt, &lam);
    let binding = list(rt, &[name, lam]);
    let bindings = list(rt, &[binding]);
    let mut call = vec![name];
    call.extend(initargs);
    let call = list(rt, &call);
    let letrec_s = sym(rt, "letrec");
    let letrec = list(rt, &[letrec_s, bindings, call]);
    desugar(rt, letrec)
}

fn desugar_let_star<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    // R7RS `let*` is SEQUENTIAL — exactly the core's `let`. Flatten pairs.
    let items = rt.list_to_vec(form);
    let pairs = rt.list_to_vec(items[1]);
    let mut flat = Vec::new();
    for &pair in &pairs {
        let kv = rt.list_to_vec(pair);
        flat.push(kv[0]);
        flat.push(desugar(rt, kv[1]));
    }
    let binds = rt.vec_to_list(&flat);
    let mut out = vec![sym(rt, "let"), binds];
    for &b in &items[2..] {
        out.push(desugar(rt, b));
    }
    rt.vec_to_list(&out)
}

fn desugar_cond<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    // (cond (t e...) ... (else e...)) -> nested (if t (do e'...) rest)
    let clauses = rt.list_to_vec(form);
    let mut result = rt.encode(Val::Nil);
    for &clause in clauses[1..].iter().rev() {
        let parts = rt.list_to_vec(clause);
        let mut body = vec![sym(rt, "do")];
        for &e in &parts[1..] {
            body.push(desugar(rt, e));
        }
        let body = rt.vec_to_list(&body);
        if head_sym(rt, clause).as_deref() == Some("else") {
            result = body;
        } else {
            let test = desugar(rt, parts[0]);
            let if_s = sym(rt, "if");
            result = rt.vec_to_list(&[if_s, test, body, result]);
        }
    }
    result
}
