//! A mini-Clojure frontend, built entirely on the microlang toolkit's PUBLIC API.
//!
//! The toolkit deliberately has NO macro system and NO Clojure data types — both
//! are frontend policy. So this frontend brings:
//!   * a Clojure reader (`reader`): `[] {} #{} :kw "str" \c '` -> toolkit values,
//!   * its own PROCEDURAL macro expander (`defmacro` + `expand`),
//!   * Clojure's data types as list-backed tagged `Record`s (Vector/Map/Set/
//!     Keyword) with `type-of`-driven dispatch, and
//!   * a `clojure.core` prelude — seqs, collections, HOFs — written in the
//!     language on top of the toolkit's cons/record/vector primitives.
//!
//! Collection LITERALS in code position are desugared by `expand` to constructor
//! calls (so `[a b]` evaluates its elements); under `quote` they stay literal.
//! Binding vectors (`fn`/`let`) are converted back to the toolkit's list form.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use microlang::value::Sym;
use microlang::{CodeSpace, Obj, Runtime, Val, ValueModel};

/// Monotonic counter for auto-gensym (`foo#`). Deterministic (no RNG/clock).
static GENSYM: AtomicU64 = AtomicU64::new(0);

mod compile;
mod core_src;
mod reader;
pub use reader::read_all;

use compile::Compiler;

/// Run a mini-Clojure program (the `clojure.core` prelude first). Returns the
/// last form's value.
pub fn run<M: ValueModel>(rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
    let mut macros: HashSet<Sym> = HashSet::new();
    let mut comp = Compiler::new(rt);
    run_src(rt, cs, &mut macros, &mut comp, core_src::CORE);
    // Route `(obj arg)` for a non-closure record (keyword/map/vector) through the
    // core `-apply-obj` dispatcher, so keywords/collections are callable.
    let apply_obj = rt.intern("-apply-obj");
    rt.set_apply_fn(apply_obj);
    run_src(rt, cs, &mut macros, &mut comp, src)
}

fn run_src<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    src: &str,
) -> u64 {
    let forms = reader::read_all(rt, src);
    let base = rt.root_depth();
    for &f in &forms {
        rt.push_root(f);
    }
    let mut last = rt.encode(Val::Nil);
    for i in 0..forms.len() {
        let f = rt.root_get(base + i);
        last = eval_form(rt, cs, macros, comp, f);
    }
    rt.truncate_roots(base);
    last
}

/// Expand, compile straight to `Ir`, and run it. The toolkit's `analyze` is
/// never used — this frontend owns the surface -> `Ir` lowering.
fn eval1<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &mut Compiler,
    form: u64,
) -> u64 {
    let expanded = expand(rt, cs, macros, form);
    let slot = rt.push_root(expanded);
    let ir = comp.compile(rt, rt.root_get(slot));
    rt.truncate_roots(slot);
    cs.eval_ir(cs, rt, &ir, &None)
}

fn eval_form<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    form: u64,
) -> u64 {
    // Real core.clj: `(def ^{:macro true} name (fn ...))` — reader wrapped the
    // name as `(-macro-meta name)`. Register the macro and define the fn.
    if let Some((name, newform)) = strip_def_macro_meta(rt, form) {
        macros.insert(name);
        return eval1(rt, cs, macros, comp, newform);
    }
    // Real core.clj registers a macro AFTER defining it: `(. (var foo) (setMacro))`.
    if let Some(name) = setmacro_target(rt, form) {
        macros.insert(name);
        return rt.encode(Val::Nil);
    }
    if let Some(name) = defmacro_name(rt, form) {
        // (defmacro name params body...) -> (def name (fn [&form &env params...] body...)).
        // Every macro fn gets the Clojure `&form`/`&env` hidden params (our macros
        // just ignore them), so our expander invokes all macros uniformly.
        let items = rt.list_to_vec(form);
        let form_s = sym(rt, "&form");
        let env_s = sym(rt, "&env");
        let orig = match record_field0(rt, items[2], reader::VECTOR) {
            Some(l) => rt.list_to_vec(l),
            None => rt.list_to_vec(items[2]),
        };
        let mut params = vec![form_s, env_s];
        params.extend(orig);
        let paramvec = make_vector(rt, params);
        let fn_sym = sym(rt, "fn");
        let mut fnform = vec![fn_sym, paramvec];
        fnform.extend_from_slice(&items[3..]);
        let lam = rt.vec_to_list(&fnform);
        let def_sym = sym(rt, "def");
        let defform = rt.vec_to_list(&[def_sym, items[1], lam]);
        let r = eval1(rt, cs, macros, comp, defform);
        macros.insert(name);
        return r;
    }
    eval1(rt, cs, macros, comp, form)
}

/// Fully macro-expand + desugar a form.
fn expand<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    form: u64,
) -> u64 {
    let slot = rt.push_root(form);
    // 1. head-expand while the head resolves to a macro
    loop {
        let f = rt.root_get(slot);
        let Some((head, _)) = rt.as_cons(f) else { break };
        let Val::Sym(hs) = rt.decode(head) else { break };
        if !macros.contains(&hs) {
            break;
        }
        let mfn = match rt.globals.get(&hs) {
            Some(v) => v.val,
            None => break,
        };
        // Clojure macro convention: (&form &env & args). Pass the whole form and
        // a nil env before the argument forms.
        let args = rt.list_to_vec(f);
        let nilv = rt.encode(Val::Nil);
        let mut margs = vec![f, nilv];
        margs.extend_from_slice(&args[1..]);
        let result = cs.invoke(cs, rt, mfn, &margs);
        rt.set_root(slot, result);
    }
    // 2. structural desugar / recurse
    let f = rt.root_get(slot);
    let out = if let Some((head, _)) = rt.as_cons(f) {
        if is_sym(rt, head, "quote") {
            f
        } else if is_sym(rt, head, "syntax-quote") {
            // ` template -> a form that BUILDS the data; then expand that.
            let inner = rt.list_to_vec(f)[1];
            let mut gs = HashMap::new();
            let built = syntax_quote(rt, inner, &mut gs);
            expand(rt, cs, macros, built)
        } else if is_sym(rt, head, "ns") || is_sym(rt, head, "in-ns") {
            rt.encode(Val::Nil) // namespaces are a no-op for now
        } else if is_sym(rt, head, "fn") || is_sym(rt, head, "fn*") {
            expand_fn(rt, cs, macros, f)
        } else if is_sym(rt, head, "let") || is_sym(rt, head, "let*") {
            rebuild_binder(rt, cs, macros, f)
        } else if is_sym(rt, head, "loop") || is_sym(rt, head, "loop*") {
            expand_loop(rt, cs, macros, f)
        } else if is_sym(rt, head, "defprotocol") {
            let d = desugar_defprotocol(rt, f);
            expand(rt, cs, macros, d)
        } else if is_sym(rt, head, "extend-type") {
            let d = desugar_extend_type(rt, f);
            expand(rt, cs, macros, d)
        } else if is_sym(rt, head, "deftype") {
            let d = desugar_deftype(rt, f);
            expand(rt, cs, macros, d)
        } else if is_sym(rt, head, "var") {
            // `(var x)` -> a first-class var value `(record 'Var 'x)`. (The
            // `(. (var x) (setMacro))` bootstrap pattern is intercepted earlier,
            // at eval-form level; this handles `var` in any other position.)
            let items = rt.list_to_vec(f);
            let rec = sym(rt, "record");
            let vtag = sym(rt, "Var");
            let tag_q = quote_form(rt, vtag);
            let name_q = quote_form(rt, items[1]);
            let g = rt.vec_to_list(&[rec, tag_q, name_q]);
            expand(rt, cs, macros, g)
        } else if is_sym(rt, head, "instance?") {
            let d = instance_rewrite(rt, f);
            expand(rt, cs, macros, d)
        } else if record_field0(rt, head, reader::KEYWORD).is_some() {
            // keyword in head position: (:k m) -> (get m :k)
            let items = rt.list_to_vec(f);
            let getsym = sym(rt, "get");
            let g = rt.vec_to_list(&[getsym, items[1], head]);
            expand(rt, cs, macros, g)
        } else if let Some(rw) = interop_rewrite(rt, f) {
            expand(rt, cs, macros, rw)
        } else {
            let items = rt.list_to_vec(f);
            let ex = expand_each(rt, cs, macros, &items);
            rt.vec_to_list(&ex)
        }
    } else if let Some(lst) = record_field0(rt, f, reader::VECTOR) {
        build_call(rt, cs, macros, "vector", lst)
    } else if let Some(lst) = record_field0(rt, f, reader::MAP) {
        build_call(rt, cs, macros, "hash-map", lst)
    } else if let Some(lst) = record_field0(rt, f, reader::SET) {
        build_call(rt, cs, macros, "hash-set", lst)
    } else {
        f // keyword / string / char / number / symbol — self-evaluating
    };
    rt.truncate_roots(slot);
    out
}

fn expand_each<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    items: &[u64],
) -> Vec<u64> {
    items.iter().map(|&it| expand(rt, cs, macros, it)).collect()
}

/// `(vector|hash-map|hash-set <elems>)` from a list of element forms.
fn build_call<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    ctor: &str,
    arglist: u64,
) -> u64 {
    let args = rt.list_to_vec(arglist);
    let fsym = sym(rt, ctor);
    let mut out = vec![fsym];
    out.extend(expand_each(rt, cs, macros, &args));
    rt.vec_to_list(&out)
}

/// `(let/loop [binds] body)` or `(fn [params] body)`: normalize the binding
/// VECTOR to the toolkit's binding LIST, desugaring DESTRUCTURING, then expand.
fn rebuild_binder<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    form: u64,
) -> u64 {
    let items = rt.list_to_vec(form);
    let bind_forms = match record_field0(rt, items[1], reader::VECTOR) {
        Some(lst) => rt.list_to_vec(lst),
        None => rt.list_to_vec(items[1]),
    };

    if is_sym(rt, items[0], "fn") || is_sym(rt, items[0], "fn*") {
        // Params: a destructuring param becomes a fresh param + a `let` in the
        // body binding the pattern to it. (Symbols and `&` pass through.)
        let mut params = Vec::new();
        let mut wrap = Vec::new(); // pattern, gensym pairs
        for &p in &bind_forms {
            if matches!(rt.decode(p), Val::Sym(_)) {
                params.push(p);
            } else {
                let g = gensym(rt, "p");
                params.push(g);
                wrap.push(p);
                wrap.push(g);
            }
        }
        let paramlist = rt.vec_to_list(&params);
        let body: Vec<u64> = items[2..].to_vec();
        let inner = if wrap.is_empty() {
            body
        } else {
            // (let [pat0 g0 ..] body...)
            let letk = sym(rt, "let");
            let bvec = make_vector(rt, wrap);
            let mut letf = vec![letk, bvec];
            letf.extend(body);
            vec![rt.vec_to_list(&letf)]
        };
        let mut out = vec![items[0], paramlist];
        out.extend(expand_each(rt, cs, macros, &inner));
        return rt.vec_to_list(&out);
    }

    // let / loop: destructure each (pattern, init) pair.
    let mut binds = Vec::new();
    let mut i = 0;
    while i + 1 < bind_forms.len() {
        let pairs = destructure(rt, bind_forms[i], bind_forms[i + 1]);
        binds.extend(pairs);
        i += 2;
    }
    let exbinds = expand_each(rt, cs, macros, &binds);
    let bindlist = rt.vec_to_list(&exbinds);
    let mut out = vec![items[0], bindlist];
    out.extend(expand_each(rt, cs, macros, &items[2..]));
    rt.vec_to_list(&out)
}

/// Desugar one binding `(pat, init)` into a flat `[sym expr sym expr ...]` list
/// of simple bindings. Handles symbols, sequential `[a b & r]` (nested), and
/// `{:keys [x y]}` map destructuring.
fn destructure<M: ValueModel>(rt: &mut Runtime<M>, pat: u64, init: u64) -> Vec<u64> {
    if matches!(rt.decode(pat), Val::Sym(_)) {
        return vec![pat, init];
    }
    if let Some(lst) = record_field0(rt, pat, reader::VECTOR) {
        let elems = rt.list_to_vec(lst);
        let t = gensym(rt, "vec");
        let mut binds = vec![t, init];
        let mut idx: i128 = 0;
        let mut k = 0;
        while k < elems.len() {
            if is_sym(rt, elems[k], "&") {
                let iv = int(rt, idx);
                let rest_expr = call2(rt, "drop", iv, t);
                binds.extend(destructure(rt, elems[k + 1], rest_expr));
                k += 2;
                continue;
            }
            if is_keyword(rt, elems[k], "as") {
                // [a b :as whole] -> whole bound to the entire collection.
                binds.push(elems[k + 1]);
                binds.push(t);
                k += 2;
                continue;
            }
            if !is_sym(rt, elems[k], "_") {
                let iv = int(rt, idx);
                let nth_expr = call2(rt, "nth", t, iv);
                binds.extend(destructure(rt, elems[k], nth_expr));
            }
            idx += 1;
            k += 1;
        }
        return binds;
    }
    if let Some(lst) = record_field0(rt, pat, reader::MAP) {
        let kvs = rt.list_to_vec(lst);
        let t = gensym(rt, "map");
        let mut binds = vec![t, init];
        // First locate `:or` defaults (a `{sym default …}` map) so any binding
        // can fall back when its key is absent.
        let mut defaults: Vec<(u64, u64)> = Vec::new();
        let mut k = 0;
        while k + 1 < kvs.len() {
            if is_keyword(rt, kvs[k], "or") {
                if let Some(dl) = record_field0(rt, kvs[k + 1], reader::MAP) {
                    let dkvs = rt.list_to_vec(dl);
                    let mut j = 0;
                    while j + 1 < dkvs.len() {
                        defaults.push((dkvs[j], dkvs[j + 1]));
                        j += 2;
                    }
                }
            }
            k += 2;
        }
        // `get t key`, or `(if (contains? t key) (get t key) default)` if the
        // symbol has an `:or` default.
        let get_with_default =
            |rt: &mut Runtime<M>, name: u64, key: u64| -> u64 {
                let ge = call2(rt, "get", t, key);
                for &(dn, dv) in &defaults {
                    if let (Val::Sym(a), Val::Sym(b)) = (rt.decode(dn), rt.decode(name)) {
                        if a == b {
                            let has = call2(rt, "contains?", t, key);
                            let ifs = sym(rt, "if");
                            return rt.vec_to_list(&[ifs, has, ge, dv]);
                        }
                    }
                }
                ge
            };
        let mut k = 0;
        while k + 1 < kvs.len() {
            let key = kvs[k];
            let val = kvs[k + 1];
            if is_keyword(rt, key, "keys") {
                // {:keys [x y]} -> x (get t :x), y (get t :y)  (honoring :or)
                if let Some(vl) = record_field0(rt, val, reader::VECTOR) {
                    for s in rt.list_to_vec(vl) {
                        let kw = keyword_expr(rt, s);
                        let ge = get_with_default(rt, s, kw);
                        binds.push(s);
                        binds.push(ge);
                    }
                }
            } else if is_keyword(rt, key, "as") {
                // {:as whole} -> whole bound to the entire map.
                binds.push(val);
                binds.push(t);
            } else if is_keyword(rt, key, "or") {
                // handled above
            } else {
                // {name :key} -> name (get t :key)  (honoring :or when name is a symbol)
                if matches!(rt.decode(key), Val::Sym(_)) {
                    let ge = get_with_default(rt, key, val);
                    binds.push(key);
                    binds.push(ge);
                } else {
                    let ge = call2(rt, "get", t, val);
                    binds.extend(destructure(rt, key, ge));
                }
            }
            k += 2;
        }
        return binds;
    }
    vec![pat, init]
}

// ─────────────────────────────────────────────────────────────────────────
// Protocols, on the toolkit's defmethod/dispatch axis. A protocol method
// dispatches on `type-of` the first arg (a record's tag: 'Vector/'Rect/…).
// ─────────────────────────────────────────────────────────────────────────

/// `(defprotocol P (m1 [this]) (m2 [this x]))` -> register each method NAME so
/// `(m1 x)` becomes a dispatch site (a sentinel `defmethod` per method; the real
/// impls come from `extend-type`). No-impl calls then error cleanly.
fn desugar_defprotocol<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let dok = sym(rt, "do");
    let mut out = vec![dok];
    let sentinel = sym(rt, "-protocol-default");
    for &spec in &items[2..] {
        let parts = rt.list_to_vec(spec);
        let m = parts[0];
        let params = parts[1]; // [this ...] vector record
        let nilv = rt.encode(Val::Nil);
        let fnf = mk_fn(rt, params, vec![nilv]);
        out.push(mk_defmethod(rt, m, sentinel, fnf));
    }
    rt.vec_to_list(&out)
}

/// `(extend-type T P (m1 [this] body) (m2 [this x] body) [P2 ...])` -> a
/// `(defmethod m T (fn [this ...] body))` per method. Protocol NAMES (bare
/// symbols) are grouping only, and skipped; method impls are lists.
fn desugar_extend_type<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let ty = items[1];
    let dok = sym(rt, "do");
    let mut out = vec![dok];
    for &item in &items[2..] {
        if matches!(rt.decode(item), Val::Sym(_)) {
            continue; // a protocol name — grouping only
        }
        let parts = rt.list_to_vec(item); // (m [this ...] body...)
        let m = parts[0];
        let params = parts[1];
        let fnf = mk_fn(rt, params, parts[2..].to_vec());
        out.push(mk_defmethod(rt, m, ty, fnf));
    }
    rt.vec_to_list(&out)
}

/// `(deftype T [f0 f1])` -> `(def ->T (fn [f0 f1] (record 'T f0 f1)))`. Methods
/// read fields with `(field this i)`.
fn desugar_deftype<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let tsym = items[1];
    let tname = match rt.decode(tsym) {
        Val::Sym(s) => rt.sym_name(s).to_string(),
        _ => panic!("deftype: name must be a symbol"),
    };
    let fields = match record_field0(rt, items[2], reader::VECTOR) {
        Some(l) => rt.list_to_vec(l),
        None => rt.list_to_vec(items[2]),
    };
    // (record 'T f0 f1 ...)
    let recsym = sym(rt, "record");
    let tag = quote_form(rt, tsym);
    let mut rec = vec![recsym, tag];
    rec.extend_from_slice(&fields);
    let reccall = rt.vec_to_list(&rec);
    // (fn [f0 f1] reccall)
    let paramvec = make_vector(rt, fields);
    let fnf = mk_fn(rt, paramvec, vec![reccall]);
    // (def ->T fnf)
    let ctor = sym(rt, &format!("->{tname}"));
    let defk = sym(rt, "def");
    rt.vec_to_list(&[defk, ctor, fnf])
}

fn mk_fn<M: ValueModel>(rt: &mut Runtime<M>, params: u64, body: Vec<u64>) -> u64 {
    let fnk = sym(rt, "fn");
    let mut out = vec![fnk, params];
    out.extend(body);
    rt.vec_to_list(&out)
}

fn mk_defmethod<M: ValueModel>(rt: &mut Runtime<M>, m: u64, ty: u64, imp: u64) -> u64 {
    let dm = sym(rt, "defmethod");
    rt.vec_to_list(&[dm, m, ty, imp])
}

/// `(loop [n init ...] body)` -> a self-recursive fn bound via `set!` (so `recur`
/// can tail-jump), then called. `recur` gets TCO from the toolkit trampoline.
///   (let [g nil] (set! g (fn [n ...] body[recur->g])) (g init ...))
fn expand_loop<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    form: u64,
) -> u64 {
    let items = rt.list_to_vec(form);
    let binds = match record_field0(rt, items[1], reader::VECTOR) {
        Some(l) => rt.list_to_vec(l),
        None => rt.list_to_vec(items[1]),
    };
    let mut names = Vec::new();
    let mut inits = Vec::new();
    let mut i = 0;
    while i + 1 < binds.len() {
        names.push(binds[i]);
        inits.push(binds[i + 1]);
        i += 2;
    }
    let g = gensym(rt, "loop");
    let body: Vec<u64> = items[2..].iter().map(|&b| replace_recur(rt, b, g)).collect();

    // (fn [names] body)
    let fnk = sym(rt, "fn");
    let paramvec = make_vector(rt, names);
    let mut fnform = vec![fnk, paramvec];
    fnform.extend(body);
    let fnform = rt.vec_to_list(&fnform);

    // (let [g nil] (set! g fnform) (g inits))
    let letk = sym(rt, "let");
    let nilv = rt.encode(Val::Nil);
    let bindvec = make_vector(rt, vec![g, nilv]);
    let setk = sym(rt, "set!");
    let setform = rt.vec_to_list(&[setk, g, fnform]);
    let mut call = vec![g];
    call.extend(inits);
    let callform = rt.vec_to_list(&call);
    let letform = rt.vec_to_list(&[letk, bindvec, setform, callform]);
    expand(rt, cs, macros, letform)
}

/// Replace `(recur ...)` with `(g ...)`, not descending into a nested `fn`/`loop`
/// (which rebinds the recur target).
fn replace_recur<M: ValueModel>(rt: &mut Runtime<M>, form: u64, g: u64) -> u64 {
    if let Some((h, _)) = rt.as_cons(form) {
        if is_sym(rt, h, "recur") {
            let items = rt.list_to_vec(form);
            let mut out = vec![g];
            out.extend_from_slice(&items[1..]);
            return rt.vec_to_list(&out);
        }
        if is_sym(rt, h, "fn") || is_sym(rt, h, "loop") || is_sym(rt, h, "quote") {
            return form;
        }
        let items = rt.list_to_vec(form);
        let out: Vec<u64> = items.iter().map(|&it| replace_recur(rt, it, g)).collect();
        return rt.vec_to_list(&out);
    }
    form
}

/// `(fn* name? [params] body)` or `(fn* name? ([p] b) ([p q] b) …)`: drop the
/// optional self-name, then either normalize to a single-arity `fn*` (binder
/// handles destructuring) or desugar multiple arities to one variadic fn that
/// dispatches on `(count args)`.
fn expand_fn<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    form: u64,
) -> u64 {
    let items = rt.list_to_vec(form);
    let mut i = 1;
    if i < items.len() && matches!(rt.decode(items[i]), Val::Sym(_)) {
        i += 1; // drop the self-name
    }
    // Single-arity: params are a `[..]` vector, a `(a b)` symbol-list (our legacy
    // style), or empty. Multi-arity: each element is a CLAUSE `([params] body)`
    // whose first element is a params container.
    let single = if i >= items.len() {
        true
    } else if record_field0(rt, items[i], reader::VECTOR).is_some() {
        true
    } else if let Some((h, _)) = rt.as_cons(items[i]) {
        matches!(rt.decode(h), Val::Sym(_))
    } else {
        true
    };
    if single {
        let params = items[i];
        let body = wrap_fn_recur(rt, params, &items[i + 1..]);
        let fnstar = sym(rt, "fn*");
        let mut norm = vec![fnstar, params];
        norm.extend(body);
        let normf = rt.vec_to_list(&norm);
        rebuild_binder(rt, cs, macros, normf)
    } else {
        let clauses: Vec<u64> = items[i..].to_vec();
        let desugared = multi_arity(rt, &clauses);
        expand(rt, cs, macros, desugared)
    }
}

/// A `fn` body that contains `recur` re-enters the fn — desugar to a `loop` over
/// the params (so `recur` rebinds them and tail-jumps via the loop trampoline).
fn wrap_fn_recur<M: ValueModel>(rt: &mut Runtime<M>, params: u64, body: &[u64]) -> Vec<u64> {
    let psyms = param_syms(rt, params);
    if psyms.is_empty() || !body.iter().any(|&b| has_recur(rt, b)) {
        return body.to_vec();
    }
    let loopsym = sym(rt, "loop");
    let mut binds = Vec::new();
    for &s in &psyms {
        binds.push(s);
        binds.push(s);
    }
    let bindvec = make_vector(rt, binds);
    let mut loopf = vec![loopsym, bindvec];
    loopf.extend_from_slice(body);
    vec![rt.vec_to_list(&loopf)]
}

/// The plain-symbol params (skipping `&`, keeping the rest name).
fn param_syms<M: ValueModel>(rt: &Runtime<M>, params: u64) -> Vec<u64> {
    let list = match record_field0(rt, params, reader::VECTOR) {
        Some(l) => rt.list_to_vec(l),
        None => rt.list_to_vec(params),
    };
    list.into_iter()
        .filter(|&p| matches!(rt.decode(p), Val::Sym(_)) && !is_sym(rt, p, "&"))
        .collect()
}

/// Does `ir` contain a `(recur …)` in this fn's scope (not inside a nested
/// `fn`/`loop`, which rebind the recur target)?
fn has_recur<M: ValueModel>(rt: &Runtime<M>, form: u64) -> bool {
    let Some((h, _)) = rt.as_cons(form) else { return false };
    if is_sym(rt, h, "recur") {
        return true;
    }
    if is_sym(rt, h, "fn") || is_sym(rt, h, "fn*") || is_sym(rt, h, "loop") || is_sym(rt, h, "quote") {
        return false;
    }
    rt.list_to_vec(form).iter().any(|&c| has_recur(rt, c))
}

/// Multiple arities -> `(fn* [& g] (let [n (count g)] (cond (= n k0) (let [pv0 g] b0) … true (let [pvV g] bV))))`.
/// Each arity's params destructure `g`; a variadic clause is the catch-all.
fn multi_arity<M: ValueModel>(rt: &mut Runtime<M>, clauses: &[u64]) -> u64 {
    let g = gensym(rt, "args");
    let n = gensym(rt, "n");
    let condsym = sym(rt, "cond");
    let mut cond = vec![condsym];
    for &clause in clauses {
        let parts = rt.list_to_vec(clause);
        let paramvec = parts[0];
        let params = record_field0(rt, paramvec, reader::VECTOR)
            .map(|l| rt.list_to_vec(l))
            .unwrap_or_default();
        let variadic = params.iter().any(|&p| is_sym(rt, p, "&"));
        let fixed = params.iter().position(|&p| is_sym(rt, p, "&")).unwrap_or(params.len());
        let test = if variadic {
            rt.encode(Val::Bool(true))
        } else {
            let k = int(rt, fixed as i128);
            call2(rt, "=", n, k)
        };
        let body = wrap_fn_recur(rt, paramvec, &parts[1..]);
        let letsym = sym(rt, "let");
        let bindvec = make_vector(rt, vec![paramvec, g]);
        let mut letf = vec![letsym, bindvec];
        letf.extend(body);
        let letform = rt.vec_to_list(&letf);
        cond.push(test);
        cond.push(letform);
    }
    let condform = rt.vec_to_list(&cond);
    let letsym = sym(rt, "let");
    let countg = call1(rt, "count", g);
    let bindn = make_vector(rt, vec![n, countg]);
    let letn = rt.vec_to_list(&[letsym, bindn, condform]);
    let fnstar = sym(rt, "fn*");
    let amp = sym(rt, "&");
    let paramvec = make_vector(rt, vec![amp, g]);
    rt.vec_to_list(&[fnstar, paramvec, letn])
}

// ─────────────────────────────────────────────────────────────────────────
// Java-interop shim: rewrite `(. Class …)` / `Class/method` / `.method` to our
// own primitives. We don't implement the JVM — we map each Java method to our
// reimplementation (the `-rt-*` host runtime). The table GROWS as we walk real
// clojure/core.clj; unknown interop panics loudly, naming the next gap.
// ─────────────────────────────────────────────────────────────────────────

fn interop_rewrite<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> Option<u64> {
    let items = rt.list_to_vec(form);
    let Val::Sym(hs) = rt.decode(items[0]) else { return None };
    let hname = rt.sym_name(hs).to_string();

    if hname == "." {
        let class = last_seg(&sym_str(rt, items[1]));
        if rt.as_cons(items[2]).is_some() {
            let call = rt.list_to_vec(items[2]);
            let method = sym_str(rt, call[0]);
            return Some(shim_call(rt, &class, &method, &call[1..]));
        }
        let method = sym_str(rt, items[2]);
        if items.len() > 3 {
            return Some(shim_call(rt, &class, &method, &items[3..]));
        }
        return Some(shim_field(rt, &class, &method));
    }
    if let Some(slash) = hname.find('/') {
        let class = last_seg(&hname[..slash]);
        let method = hname[slash + 1..].to_string();
        return Some(shim_call(rt, &class, &method, &items[1..]));
    }
    if hname.starts_with('.') && hname.len() > 1 {
        let method = hname[1..].to_string();
        return Some(shim_instance(rt, &method, items[1], &items[2..]));
    }
    None
}

fn sym_str<M: ValueModel>(rt: &Runtime<M>, v: u64) -> String {
    match rt.decode(v) {
        Val::Sym(s) => rt.sym_name(s).to_string(),
        _ => String::new(),
    }
}
fn last_seg(s: &str) -> String {
    s.rsplit('.').next().unwrap_or(s).to_string()
}

fn shim_call<M: ValueModel>(rt: &mut Runtime<M>, class: &str, method: &str, args: &[u64]) -> u64 {
    let head = match (class, method) {
        ("RT", "cons") => "%cons",
        ("RT", "first") => "-rt-first",
        ("RT", "next") => "-rt-next",
        ("RT", "more") => "-rt-rest",
        ("RT", "seq") => "-rt-seq",
        ("RT", "conj") => "-rt-conj",
        ("RT", "assoc") => "-rt-assoc",
        _ => panic!("unimplemented interop: {class}/{method}"),
    };
    let h = sym(rt, head);
    let mut out = vec![h];
    out.extend_from_slice(args);
    rt.vec_to_list(&out)
}

fn shim_field<M: ValueModel>(rt: &mut Runtime<M>, class: &str, field: &str) -> u64 {
    match (class, field) {
        ("PersistentList", "creator") => sym(rt, "-list"),
        _ => panic!("unimplemented interop field: {class}/{field}"),
    }
}

fn shim_instance<M: ValueModel>(rt: &mut Runtime<M>, method: &str, obj: u64, args: &[u64]) -> u64 {
    match method {
        "withMeta" => {
            let h = sym(rt, "-with-meta");
            let mut out = vec![h, obj];
            out.extend_from_slice(args);
            rt.vec_to_list(&out)
        }
        "meta" => call1(rt, "-meta", obj),
        _ => panic!("unimplemented interop instance method: .{method}"),
    }
}

fn gensym<M: ValueModel>(rt: &mut Runtime<M>, prefix: &str) -> u64 {
    let n = GENSYM.fetch_add(1, Ordering::Relaxed);
    let s = rt.intern(&format!("{prefix}__{n}__d"));
    rt.encode(Val::Sym(s))
}

fn int<M: ValueModel>(rt: &mut Runtime<M>, i: i128) -> u64 {
    rt.encode(Val::Int(i))
}

fn call2<M: ValueModel>(rt: &mut Runtime<M>, f: &str, a: u64, b: u64) -> u64 {
    let fs = sym(rt, f);
    rt.vec_to_list(&[fs, a, b])
}

/// `(record 'Keyword 'name)` — a keyword-construction form for `:name`.
fn keyword_expr<M: ValueModel>(rt: &mut Runtime<M>, name_sym: u64) -> u64 {
    let rec = sym(rt, "record");
    let kw = sym(rt, "Keyword");
    let tag = quote_form(rt, kw);
    let nm = quote_form(rt, name_sym);
    rt.vec_to_list(&[rec, tag, nm])
}

fn make_vector<M: ValueModel>(rt: &mut Runtime<M>, elems: Vec<u64>) -> u64 {
    let lst = rt.vec_to_list(&elems);
    let type_id = rt.intern(reader::VECTOR);
    let id = rt.alloc(Obj::Record { type_id, fields: vec![lst] });
    <M::R as microlang::Repr>::enc_ref(id)
}

fn is_keyword<M: ValueModel>(rt: &Runtime<M>, k: u64, name: &str) -> bool {
    match record_field0(rt, k, reader::KEYWORD) {
        Some(f0) => is_sym(rt, f0, name),
        None => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Syntax-quote: `template with ~unquote and ~@splice, plus foo# auto-gensym.
// Produces a FORM that builds the templated data at run time.
// ─────────────────────────────────────────────────────────────────────────

fn syntax_quote<M: ValueModel>(rt: &mut Runtime<M>, form: u64, gs: &mut HashMap<Sym, Sym>) -> u64 {
    match rt.decode(form) {
        // symbol -> 'symbol, or a per-template gensym for `foo#`
        Val::Sym(s) => {
            let name = rt.sym_name(s).to_string();
            if name.ends_with('#') {
                let g = match gs.get(&s) {
                    Some(&g) => g,
                    None => {
                        let n = GENSYM.fetch_add(1, Ordering::Relaxed);
                        let g = rt.intern(&format!("{}__{}__auto", &name[..name.len() - 1], n));
                        gs.insert(s, g);
                        g
                    }
                };
                let gv = rt.encode(Val::Sym(g));
                quote_form(rt, gv)
            } else {
                quote_form(rt, form)
            }
        }
        // self-evaluating literals stay as themselves
        Val::Int(_) | Val::Float(_) | Val::Bool(_) | Val::Nil => form,
        Val::Ref(_) => {
            // ~expr -> expr
            if let Some((h, _)) = rt.as_cons(form) {
                if is_sym(rt, h, "unquote") {
                    return rt.list_to_vec(form)[1];
                }
                // a list -> (concat <seq parts>)
                let items = rt.list_to_vec(form);
                return sq_concat(rt, &items, gs);
            }
            // [..] -> (vec (concat ..))
            if let Some(lst) = record_field0(rt, form, reader::VECTOR) {
                let items = rt.list_to_vec(lst);
                let inner = sq_concat(rt, &items, gs);
                return call1(rt, "vec", inner);
            }
            // {..} -> (hash-map 'k (sq v) ..)  (no splice in map position)
            if let Some(lst) = record_field0(rt, form, reader::MAP) {
                let items = rt.list_to_vec(lst);
                let hm = sym(rt, "hash-map");
                let mut out = vec![hm];
                for &it in &items {
                    out.push(syntax_quote(rt, it, gs));
                }
                return rt.vec_to_list(&out);
            }
            // keyword / string / char -> themselves (self-evaluating)
            form
        }
    }
}

/// `(concat (list e0) (list e1) ..)`, with `~@x` contributing `x` directly.
fn sq_concat<M: ValueModel>(rt: &mut Runtime<M>, items: &[u64], gs: &mut HashMap<Sym, Sym>) -> u64 {
    let concat = sym(rt, "concat");
    let mut parts = vec![concat];
    for &it in items {
        if let Some((h, _)) = rt.as_cons(it) {
            if is_sym(rt, h, "unquote-splice") {
                parts.push(rt.list_to_vec(it)[1]);
                continue;
            }
        }
        let one = syntax_quote(rt, it, gs);
        parts.push(call1(rt, "list", one));
    }
    rt.vec_to_list(&parts)
}

fn quote_form<M: ValueModel>(rt: &mut Runtime<M>, x: u64) -> u64 {
    let q = sym(rt, "quote");
    rt.vec_to_list(&[q, x])
}

fn call1<M: ValueModel>(rt: &mut Runtime<M>, f: &str, arg: u64) -> u64 {
    let fs = sym(rt, f);
    rt.vec_to_list(&[fs, arg])
}

fn record_field0<M: ValueModel>(rt: &Runtime<M>, form: u64, tag: &str) -> Option<u64> {
    if let Val::Ref(id) = rt.decode(form) {
        if let Obj::Record { type_id, fields } = &rt.heap[id as usize] {
            if rt.sym_name(*type_id) == tag {
                return fields.first().copied();
            }
        }
    }
    None
}

fn sym<M: ValueModel>(rt: &mut Runtime<M>, n: &str) -> u64 {
    let s = rt.intern(n);
    rt.encode(Val::Sym(s))
}

fn is_sym<M: ValueModel>(rt: &Runtime<M>, bits: u64, name: &str) -> bool {
    matches!(rt.decode(bits), Val::Sym(s) if rt.sym_name(s) == name)
}

/// `(instance? Class x)` -> `(%num-eq (type-of x) 'Tag)`, mapping the (possibly
/// package-qualified) Java class name to the runtime tag `type-of` reports.
/// Panics on an unknown class rather than silently returning a wrong answer.
fn instance_rewrite<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let Val::Sym(cs) = rt.decode(items[1]) else {
        panic!("instance?: first argument must be a class symbol");
    };
    let cname = rt.sym_name(cs).to_string();
    // Match on the simple (last) segment of a dotted class name.
    let simple = cname.rsplit('.').next().unwrap_or(&cname);
    let tag = match simple {
        "Symbol" => "Symbol",
        "Keyword" => "Keyword",
        "String" | "CharSequence" => "String",
        "Character" => "Char",
        "Long" | "Integer" | "Short" | "Byte" | "BigInteger" | "BigInt" => "Long",
        "Double" | "Float" | "BigDecimal" => "Double",
        "Boolean" => "Boolean",
        s if s.contains("Vector") => "Vector",
        s if s.contains("Map") => "Map",
        s if s.contains("Set") => "Set",
        s if s.contains("List") || s == "ISeq" || s == "Seqable" || s == "Cons" => "List",
        s if s == "IFn" || s == "AFn" || s == "Fn" => "Fn",
        _ => panic!("instance?: unmapped class `{cname}` (add it to instance_rewrite)"),
    };
    let numeq = sym(rt, "%num-eq");
    let typeof_sym = sym(rt, "type-of");
    let quote_sym = sym(rt, "quote");
    let tag_sym = sym(rt, tag);
    let tag_quoted = rt.vec_to_list(&[quote_sym, tag_sym]);
    let typeof_call = rt.vec_to_list(&[typeof_sym, items[2]]);
    rt.vec_to_list(&[numeq, typeof_call, tag_quoted])
}

/// `(def (-macro-meta name) val…)` -> `(name, (def name val…))`, else None.
fn strip_def_macro_meta<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> Option<(Sym, u64)> {
    let items = rt.list_to_vec(form);
    if items.len() < 2 || !is_sym(rt, items[0], "def") {
        return None;
    }
    let nparts = rt.list_to_vec(items[1]);
    if nparts.len() < 2 || !is_sym(rt, nparts[0], "-macro-meta") {
        return None;
    }
    let name = match rt.decode(nparts[1]) {
        Val::Sym(s) => s,
        _ => return None,
    };
    let mut newf = vec![items[0], nparts[1]];
    newf.extend_from_slice(&items[2..]);
    Some((name, rt.vec_to_list(&newf)))
}

/// Recognize real core.clj's macro registration `(. (var NAME) (setMacro))`
/// (and its `(. (var NAME) setMacro)` / `(.setMacro (var NAME))` variants),
/// returning NAME. This runs at eval-form level because marking a macro must
/// mutate the (frontend-owned) macro set that the NEXT form's expander reads.
fn setmacro_target<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Sym> {
    let items = rt.list_to_vec(form);
    if items.is_empty() {
        return None;
    }
    // Locate the receiver form and confirm the method is `setMacro`.
    let recv = if is_sym(rt, items[0], ".") {
        if items.len() < 3 {
            return None;
        }
        // items[2] is either `(setMacro)` or the bare symbol `setMacro`.
        let is_setmacro = match rt.as_cons(items[2]) {
            Some(_) => is_sym(rt, rt.list_to_vec(items[2])[0], "setMacro"),
            None => is_sym(rt, items[2], "setMacro"),
        };
        if !is_setmacro {
            return None;
        }
        items[1]
    } else if is_sym(rt, items[0], ".setMacro") {
        if items.len() < 2 {
            return None;
        }
        items[1]
    } else {
        return None;
    };
    // The receiver must be `(var NAME)` or a `#'NAME` var literal.
    let rv = rt.list_to_vec(recv);
    if rv.len() == 2 && is_sym(rt, rv[0], "var") {
        if let Val::Sym(name) = rt.decode(rv[1]) {
            return Some(name);
        }
    }
    None
}

fn defmacro_name<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Sym> {
    let (h, _) = rt.as_cons(form)?;
    if !is_sym(rt, h, "defmacro") {
        return None;
    }
    let items = rt.list_to_vec(form);
    match rt.decode(items[1]) {
        Val::Sym(name) => Some(name),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Clojure-flavored printing (the toolkit's `print` doesn't know these tags).
// ─────────────────────────────────────────────────────────────────────────

/// Render a value the way Clojure would: `[..]` vectors, `{..}` maps, `#{..}`
/// sets, `:kw` keywords, `(..)` lists.
pub fn clj_str<M: ValueModel>(rt: &Runtime<M>, v: u64) -> String {
    match rt.decode(v) {
        Val::Ref(id) => match &rt.heap[id as usize] {
            Obj::Record { type_id, fields } => {
                let tag = rt.sym_name(*type_id);
                match tag {
                    "Keyword" => format!(":{}", rt.print(fields[0])),
                    "Vector" => format!("[{}]", list_items(rt, fields[0], " ")),
                    "Set" => format!("#{{{}}}", list_items(rt, fields[0], " ")),
                    "Map" => format!("{{{}}}", map_items(rt, fields[0])),
                    _ => rt.print(v),
                }
            }
            Obj::Cons { .. } => format!("({})", list_items(rt, v, " ")),
            _ => rt.print(v),
        },
        _ => rt.print(v),
    }
}

fn list_items<M: ValueModel>(rt: &Runtime<M>, lst: u64, sep: &str) -> String {
    rt.list_to_vec(lst)
        .iter()
        .map(|&x| clj_str(rt, x))
        .collect::<Vec<_>>()
        .join(sep)
}

fn map_items<M: ValueModel>(rt: &Runtime<M>, kvs: u64) -> String {
    let items = rt.list_to_vec(kvs);
    items
        .chunks(2)
        .map(|kv| {
            let k = clj_str(rt, kv[0]);
            let v = kv.get(1).map(|&x| clj_str(rt, x)).unwrap_or_default();
            format!("{k} {v}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}
