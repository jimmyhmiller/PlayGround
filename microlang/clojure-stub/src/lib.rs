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
mod cljs_types_src;
mod clojure_data_json_src;
mod clojure_string_src;
mod core_src;
mod reader;
pub use reader::read_all;

use compile::Compiler;

/// Run a mini-Clojure program (the `clojure.core` prelude first). Returns the
/// last form's value. `require` searches the default load path (`$MICROLANG_PATH`,
/// colon-separated, else `.`).
pub fn run<M: ValueModel>(rt: &mut Runtime<M>, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
    run_with_paths(rt, cs, src, default_load_paths())
}

/// The load-path directories `require` searches (from `$MICROLANG_PATH`, else `.`).
pub fn default_load_paths() -> Vec<std::path::PathBuf> {
    match std::env::var("MICROLANG_PATH") {
        Ok(v) if !v.is_empty() => v.split(':').map(std::path::PathBuf::from).collect(),
        _ => vec![std::path::PathBuf::from(".")],
    }
}

/// Like [`run`], but with an explicit set of load-path directories for `require`.
pub fn run_with_paths<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    src: &str,
    load_paths: Vec<std::path::PathBuf>,
) -> u64 {
    let mut macros: HashSet<Sym> = HashSet::new();
    let mut comp = Compiler::new(rt);
    comp.set_load_paths(load_paths);
    run_src(rt, cs, &mut macros, &mut comp, core_src::CORE);
    // Persistent data structures ported from ClojureScript (EPL-1.0), loaded after
    // the core protocols/shim they build on. Redefines vector/vec/vector?.
    run_src(rt, cs, &mut macros, &mut comp, cljs_types_src::CLJS);
    // clojure.core + the cljs types loaded into `clojure.core`; user code from
    // here on runs in the `user` namespace. EVERY var is now ns-qualified, so the
    // frontend's own references to core helpers use their `clojure.core/…` names.
    comp.end_core_load();
    // `clojure.string` — bundled, but written ENTIRELY in the language (its `(ns
    // clojure.string)` form sets the ns + marks it loaded). Proof that the string
    // library is library code over one primitive, not builtins.
    run_src(rt, cs, &mut macros, &mut comp, clojure_string_src::CLOJURE_STRING);
    // `clojure.data.json` — a real library, also written entirely in the language
    // (loaded after clojure.string, which its writer uses for `join`).
    run_src(rt, cs, &mut macros, &mut comp, clojure_data_json_src::CLOJURE_DATA_JSON);
    comp.set_ns("user");
    // These are provided in-process; `require` must never look for them on disk.
    comp.mark_loaded("clojure.core");
    comp.mark_loaded("user");
    // Route `(obj arg)` for a non-closure record (keyword/map/vector) through the
    // core `-apply-obj` dispatcher, so keywords/collections are callable.
    let apply_obj = rt.intern("clojure.core/-apply-obj");
    rt.set_apply_fn(apply_obj);
    let result = run_src(rt, cs, &mut macros, &mut comp, src);
    // Force any lazy sequence in the final value so callers / the printer (which
    // can't invoke thunks) see a fully realized result.
    let slot = rt.push_root(result);
    let realize = rt.intern("clojure.core/-realize");
    let out = match rt.global(realize) {
        Some(rf) => cs.invoke(cs, rt, rf, &[rt.root_get(slot)]),
        None => rt.root_get(slot),
    };
    rt.truncate_roots(slot);
    out
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
    let expanded = expand(rt, cs, macros, comp, form);
    let slot = rt.push_root(expanded);
    let ir = comp.compile(rt, rt.root_get(slot));
    rt.truncate_roots(slot);
    let r = cs.eval_ir(cs, rt, &ir, &None);
    // A signal still pending at the top level is an UNCAUGHT throw/escape — a
    // program error that terminates (like an uncaught exception in real Clojure).
    // This is the boundary, not the throw/catch mechanism (which is signal-based).
    if rt.pending() {
        let sig = rt.take_signal();
        if sig.kind == 1 {
            panic!("uncaught throw: {}", rt.print(sig.value));
        }
        panic!("escape continuation invoked outside its (%callec) extent");
    }
    r
}

fn eval_form<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    form: u64,
) -> u64 {
    // Namespace declarations mutate the compiler's resolution state (and may LOAD
    // required files) and yield nil. Handled before macro/def checks.
    if let Some(r) = handle_ns_form(rt, cs, macros, comp, form) {
        return r;
    }
    // Real core.clj: `(def ^{:macro true} name (fn ...))` — reader wrapped the
    // name as `(-macro-meta name)`. Define the fn, then register the macro under
    // its RESOLVED (namespace-qualified) sym.
    if let Some((name, newform)) = strip_def_macro_meta(rt, form) {
        let r = eval1(rt, cs, macros, comp, newform);
        let q = comp.resolve_ref(rt, name);
        macros.insert(q);
        rt.set_var_flags(q, microlang::runtime::VAR_MACRO);
        return r;
    }
    // Real core.clj registers a macro AFTER defining it: `(. (var foo) (setMacro))`.
    if let Some(name) = setmacro_target(rt, form) {
        let q = comp.resolve_ref(rt, name);
        macros.insert(q);
        rt.set_var_flags(q, microlang::runtime::VAR_MACRO);
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
        let q = comp.resolve_ref(rt, name);
        macros.insert(q);
        rt.set_var_flags(q, microlang::runtime::VAR_MACRO);
        return r;
    }
    eval1(rt, cs, macros, comp, form)
}

/// Fully macro-expand + desugar a form.
fn expand<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    form: u64,
) -> u64 {
    let slot = rt.push_root(form);
    // 1. head-expand while the head resolves to a macro
    loop {
        let f = rt.root_get(slot);
        let Some((head, _)) = rt.as_cons(f) else { break };
        let Val::Sym(hs) = rt.decode(head) else { break };
        // Resolve the head to its namespace-qualified var; is that a macro? This
        // makes bare, fully-qualified, AND alias-qualified macro calls all work.
        let q = comp.resolve_ref(rt, hs);
        if !macros.contains(&q) {
            break;
        }
        let mfn = match rt.global(q) {
            Some(v) => v,
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
            // A quoted collection literal must build the runtime persistent type;
            // a plain quoted datum (symbols, lists of atoms) stays a literal.
            let datum = rt.list_to_vec(f)[1];
            if datum_has_coll(rt, datum) {
                let built = build_quote(rt, datum);
                expand(rt, cs, macros, comp, built)
            } else {
                f
            }
        } else if is_sym(rt, head, "syntax-quote") {
            // ` template -> a form that BUILDS the data; then expand that.
            let inner = rt.list_to_vec(f)[1];
            let mut gs = HashMap::new();
            let built = syntax_quote(rt, inner, &mut gs);
            expand(rt, cs, macros, comp, built)
        } else if is_sym(rt, head, "ns") || is_sym(rt, head, "in-ns") {
            rt.encode(Val::Nil) // namespaces are a no-op for now
        } else if is_sym(rt, head, "fn") || is_sym(rt, head, "fn*") {
            expand_fn(rt, cs, macros, comp, f)
        } else if is_sym(rt, head, "let") || is_sym(rt, head, "let*") {
            rebuild_binder(rt, cs, macros, comp, f)
        } else if is_sym(rt, head, "loop") || is_sym(rt, head, "loop*") {
            expand_loop(rt, cs, macros, comp, f)
        } else if is_sym(rt, head, "defprotocol") {
            let d = desugar_defprotocol(rt, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "extend-type") {
            let d = desugar_extend_type(rt, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "deftype") {
            let d = desugar_deftype(rt, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "var") {
            // `(var x)` / `#'x` -> a first-class Var handle `(record 'Var 'ns/x)`.
            // The name is RESOLVED so the handle keys the global table exactly as
            // an ordinary reference to `x` does — deref/alter-var-root operate by
            // that sym. (The `(. (var x) (setMacro))` bootstrap is intercepted
            // earlier, at eval-form level; this handles `var` elsewhere.)
            let items = rt.list_to_vec(f);
            let resolved = match rt.decode(items[1]) {
                Val::Sym(s) => rt.encode(Val::Sym(comp.resolve_ref(rt, s))),
                _ => items[1],
            };
            let rec = sym(rt, "record");
            let vtag = sym(rt, "Var");
            let tag_q = quote_form(rt, vtag);
            let name_q = quote_form(rt, resolved);
            let g = rt.vec_to_list(&[rec, tag_q, name_q]);
            expand(rt, cs, macros, comp, g)
        } else if let Some(qualified) = resolve_rewrite(rt, comp, f) {
            // `(resolve 'x)` / `(ns-resolve 'ns 'x)` with LITERAL symbols: namespace
            // resolution is compile-time, so rewrite to `(find-var 'qualified)`.
            let find = sym(rt, "find-var");
            let qsym = rt.encode(Val::Sym(qualified));
            let qform = quote_form(rt, qsym);
            let g = rt.vec_to_list(&[find, qform]);
            expand(rt, cs, macros, comp, g)
        } else if is_sym(rt, head, "instance?") {
            let d = instance_rewrite(rt, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "try") {
            // Desugar typed multi-catch into a single catch-all whose body is a
            // type-dispatch (ClojureScript's model); expand the result.
            let d = desugar_try(rt, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "binding") {
            // `(binding [*x* v] body…)` — thread-local dynamic-var bindings via a
            // `try`/`finally` that installs and (always) unwinds them.
            let d = desugar_binding(rt, comp, f);
            expand(rt, cs, macros, comp, d)
        } else if is_sym(rt, head, "with-redefs") {
            // `(with-redefs [f g] body…)` — temporarily rebind var ROOTS (for
            // testing), restoring them in a `finally`.
            let d = desugar_with_redefs(rt, f);
            expand(rt, cs, macros, comp, d)
        } else if record_field0(rt, head, reader::KEYWORD).is_some() {
            // keyword in head position: (:k m) -> (get m :k)
            let items = rt.list_to_vec(f);
            let getsym = sym(rt, "get");
            let g = rt.vec_to_list(&[getsym, items[1], head]);
            expand(rt, cs, macros, comp, g)
        } else if let Some(rw) = interop_rewrite(rt, f) {
            expand(rt, cs, macros, comp, rw)
        } else {
            let items = rt.list_to_vec(f);
            let ex = expand_each(rt, cs, macros, comp, &items);
            rt.vec_to_list(&ex)
        }
    } else if let Some(lst) = record_field0(rt, f, reader::VECTOR) {
        build_call(rt, cs, macros, comp, "vector", lst)
    } else if let Some(lst) = record_field0(rt, f, reader::MAP) {
        build_call(rt, cs, macros, comp, "hash-map", lst)
    } else if let Some(lst) = record_field0(rt, f, reader::SET) {
        build_call(rt, cs, macros, comp, "hash-set", lst)
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
    comp: &Compiler,
    items: &[u64],
) -> Vec<u64> {
    items.iter().map(|&it| expand(rt, cs, macros, comp, it)).collect()
}

/// `(vector|hash-map|hash-set <elems>)` from a list of element forms.
fn build_call<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    ctor: &str,
    arglist: u64,
) -> u64 {
    let args = rt.list_to_vec(arglist);
    let fsym = sym(rt, ctor);
    let mut out = vec![fsym];
    out.extend(expand_each(rt, cs, macros, comp, &args));
    rt.vec_to_list(&out)
}

/// `(let/loop [binds] body)` or `(fn [params] body)`: normalize the binding
/// VECTOR to the toolkit's binding LIST, desugaring DESTRUCTURING, then expand.
fn rebuild_binder<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    form: u64,
) -> u64 {
    let items = rt.list_to_vec(form);
    let bind_forms = binding_items(rt, items[1]).unwrap_or_else(|| rt.list_to_vec(items[1]));

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
        out.extend(expand_each(rt, cs, macros, comp, &inner));
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
    let exbinds = expand_each(rt, cs, macros, comp, &binds);
    let bindlist = rt.vec_to_list(&exbinds);
    let mut out = vec![items[0], bindlist];
    out.extend(expand_each(rt, cs, macros, comp, &items[2..]));
    rt.vec_to_list(&out)
}

/// Desugar one binding `(pat, init)` into a flat `[sym expr sym expr ...]` list
/// of simple bindings. Handles symbols, sequential `[a b & r]` (nested), and
/// `{:keys [x y]}` map destructuring.
fn destructure<M: ValueModel>(rt: &mut Runtime<M>, pat: u64, init: u64) -> Vec<u64> {
    if matches!(rt.decode(pat), Val::Sym(_)) {
        return vec![pat, init];
    }
    if let Some(elems) = binding_items(rt, pat) {
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
                if let Some(vl) = binding_items(rt, val) {
                    for s in vl {
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
    // `(extend-type nil …)`: `nil` reads as the value Nil, but `type-of nil`
    // reports the tag symbol `nil` — so extend against that symbol.
    let ty = if matches!(rt.decode(items[1]), Val::Nil) {
        sym(rt, "nil")
    } else {
        items[1]
    };
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

/// `(deftype T [f0 f1] Protocol (-m [this a] body) …)` — the full ClojureScript
/// `deftype` form. Emits `(def ->T (fn [f0 f1] (record 'T f0 f1)))` PLUS an inline
/// `-proto-method` per protocol method (type-indexed dispatch, same axis as
/// `extend-type`). Inside a method body the deftype's FIELD NAMES are in lexical
/// scope (bound from the instance = first param), matching cljs; multi-arity
/// methods (the same name repeated) become a multi-arity fn. The `Object` group
/// (host interop: toString/equiv/indexOf/…) is skipped. `^:mutable` is ignored
/// (the reader already drops the meta) — `set!` on such a "field" hits the local
/// binding, so caching-hash degrades to recomputation rather than erroring.
fn desugar_deftype<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let tsym = items[1];
    let tname = match rt.decode(tsym) {
        Val::Sym(s) => rt.sym_name(s).to_string(),
        _ => panic!("deftype: name must be a symbol"),
    };
    let fields = binding_items(rt, items[2]).unwrap_or_else(|| rt.list_to_vec(items[2]));

    // constructor: (def ->T (fn [fields] (record 'T fields)))
    let recsym = sym(rt, "record");
    let tag = quote_form(rt, tsym);
    let mut rec = vec![recsym, tag];
    rec.extend_from_slice(&fields);
    let reccall = rt.vec_to_list(&rec);
    let paramvec = make_vector(rt, fields.clone());
    let ctorfn = mk_fn(rt, paramvec, vec![reccall]);
    let ctorname = sym(rt, &format!("->{tname}"));
    let defk = sym(rt, "def");
    let ctordef = rt.vec_to_list(&[defk, ctorname, ctorfn]);

    // register the field-name order so `(.-field x)` resolves by name.
    let regsym = sym(rt, "%register-fields");
    let tq = quote_form(rt, tsym);
    let flist = rt.vec_to_list(&fields);
    let fq = quote_form(rt, flist);
    let regcall = rt.vec_to_list(&[regsym, tq, fq]);

    let dok = sym(rt, "do");
    let mut out = vec![dok, ctordef, regcall];

    // inline protocol methods: a bare symbol starts a protocol group (or `Object`);
    // lists are method impls, and consecutive same-name impls are one multi-arity fn.
    let mut i = 3;
    let mut in_object = false;
    while i < items.len() {
        let it = items[i];
        if matches!(rt.decode(it), Val::Sym(_)) {
            in_object = is_sym(rt, it, "Object");
            i += 1;
            continue;
        }
        let mname = rt.list_to_vec(it)[0];
        // gather this + following consecutive impls sharing the method name.
        let mut arities: Vec<(u64, Vec<u64>)> = Vec::new();
        loop {
            let p = rt.list_to_vec(items[i]);
            arities.push((p[1], p[2..].to_vec()));
            i += 1;
            if i >= items.len() || matches!(rt.decode(items[i]), Val::Sym(_)) {
                break;
            }
            if !same_sym(rt, rt.list_to_vec(items[i])[0], mname) {
                break;
            }
        }
        if in_object {
            continue; // host interop methods — not dispatched
        }
        let mfn = build_method_fn(rt, &fields, &arities);
        out.push(mk_defmethod(rt, mname, tsym, mfn));
    }
    rt.vec_to_list(&out)
}

fn same_sym<M: ValueModel>(rt: &Runtime<M>, a: u64, b: u64) -> bool {
    matches!((rt.decode(a), rt.decode(b)), (Val::Sym(x), Val::Sym(y)) if x == y)
}

/// Build a method fn (single- or multi-arity) whose body has the deftype's fields
/// bound from the instance (first param), except fields shadowed by a param name.
fn build_method_fn<M: ValueModel>(
    rt: &mut Runtime<M>,
    fields: &[u64],
    arities: &[(u64, Vec<u64>)],
) -> u64 {
    if arities.len() == 1 {
        let (params, body) = &arities[0];
        let wrapped = wrap_field_let(rt, fields, *params, body);
        return mk_fn(rt, *params, vec![wrapped]);
    }
    let fnk = sym(rt, "fn");
    let mut out = vec![fnk];
    for (params, body) in arities {
        let wrapped = wrap_field_let(rt, fields, *params, body);
        out.push(rt.vec_to_list(&[*params, wrapped]));
    }
    rt.vec_to_list(&out)
}

/// `(let [f0 (field inst 0) f1 (field inst 1) …] body…)` — binds each deftype
/// field (that a param doesn't shadow) to its slot on the instance.
fn wrap_field_let<M: ValueModel>(
    rt: &mut Runtime<M>,
    fields: &[u64],
    params: u64,
    body: &[u64],
) -> u64 {
    let plist = binding_items(rt, params).unwrap_or_else(|| rt.list_to_vec(params));
    if plist.is_empty() {
        let dok = sym(rt, "do");
        let mut out = vec![dok];
        out.extend_from_slice(body);
        return rt.vec_to_list(&out);
    }
    let inst = plist[0];
    let param_syms: std::collections::HashSet<Sym> = plist
        .iter()
        .filter_map(|&p| match rt.decode(p) {
            Val::Sym(s) => Some(s),
            _ => None,
        })
        .collect();
    let fieldk = sym(rt, "field");
    let mut binds = Vec::new();
    for (idx, &f) in fields.iter().enumerate() {
        if let Val::Sym(s) = rt.decode(f) {
            if param_syms.contains(&s) {
                continue; // a param shadows this field
            }
        }
        let idxv = rt.encode(Val::Int(idx as i128));
        let fieldcall = rt.vec_to_list(&[fieldk, inst, idxv]);
        binds.push(f);
        binds.push(fieldcall);
    }
    if binds.is_empty() {
        let dok = sym(rt, "do");
        let mut out = vec![dok];
        out.extend_from_slice(body);
        return rt.vec_to_list(&out);
    }
    let bindvec = make_vector(rt, binds);
    let letk = sym(rt, "let");
    let mut out = vec![letk, bindvec];
    out.extend_from_slice(body);
    rt.vec_to_list(&out)
}

fn mk_fn<M: ValueModel>(rt: &mut Runtime<M>, params: u64, body: Vec<u64>) -> u64 {
    let fnk = sym(rt, "fn");
    let mut out = vec![fnk, params];
    out.extend(body);
    rt.vec_to_list(&out)
}

fn mk_defmethod<M: ValueModel>(rt: &mut Runtime<M>, m: u64, ty: u64, imp: u64) -> u64 {
    // Internal protocol-impl form (type-indexed dispatch). Named distinctly from
    // the user-facing `defmethod` (multimethods), which is a library macro.
    let dm = sym(rt, "-proto-method");
    rt.vec_to_list(&[dm, m, ty, imp])
}

/// `(loop [n init ...] body)` -> a self-recursive fn bound via `set!` (so `recur`
/// can tail-jump), then called. `recur` gets TCO from the toolkit trampoline.
///   (let [g nil] (set! g (fn [n ...] body[recur->g])) (g init ...))
fn expand_loop<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &HashSet<Sym>,
    comp: &Compiler,
    form: u64,
) -> u64 {
    let items = rt.list_to_vec(form);
    let binds = match binding_items(rt, items[1]) {
        Some(l) => l,
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
    expand(rt, cs, macros, comp, letform)
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
    comp: &Compiler,
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
    } else if binding_items(rt, items[i]).is_some() {
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
        rebuild_binder(rt, cs, macros, comp, normf)
    } else {
        let clauses: Vec<u64> = items[i..].to_vec();
        let desugared = multi_arity(rt, &clauses);
        expand(rt, cs, macros, comp, desugared)
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
    let list = match binding_items(rt, params) {
        Some(l) => l,
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
        let params = binding_items(rt, paramvec).unwrap_or_default();
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
        // `Foo/bar` with an UPPERCASE leading segment is a host static call; a
        // lowercase prefix (`m/square`, `util.math/square`) is a namespace- or
        // alias-qualified VAR reference — leave it for the compiler to resolve.
        if class.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            let method = hname[slash + 1..].to_string();
            return Some(shim_call(rt, &class, &method, &items[1..]));
        }
    }
    // `(.-field x)` -> `(%field-by-name x 'field)` — ClojureScript field access on
    // a deftype instance, resolved through the field-name registry.
    if let Some(field) = hname.strip_prefix(".-") {
        if !field.is_empty() {
            let fsym = sym(rt, field);
            let fq = quote_form(rt, fsym);
            let fbn = sym(rt, "%field-by-name");
            return Some(rt.vec_to_list(&[fbn, items[1], fq]));
        }
    }
    if hname.starts_with('.') && hname.len() > 1 {
        let method = hname[1..].to_string();
        return Some(shim_instance(rt, &method, items[1], &items[2..]));
    }
    // Constructor `(Class. args…)` -> a record tagged by the class's simple name,
    // so `type-of` / typed `catch` can identify it (e.g. `(RuntimeException. m)`).
    if hname.ends_with('.') && hname.len() > 1 && hname != ".." {
        let simple = last_seg(&hname[..hname.len() - 1]);
        let rec = sym(rt, "record");
        let tag = sym(rt, &simple);
        let tag_q = quote_form(rt, tag);
        let mut out = vec![rec, tag_q];
        out.extend_from_slice(&items[1..]);
        return Some(rt.vec_to_list(&out));
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

/// `(-concat (list e0) (list e1) ..)`, with `~@x` contributing `x` directly.
/// Uses the EAGER `-concat` (not the lazy user `concat`): a macro must return a
/// realized code form the expander can splice, not a lazy seq.
fn sq_concat<M: ValueModel>(rt: &mut Runtime<M>, items: &[u64], gs: &mut HashMap<Sym, Sym>) -> u64 {
    let concat = sym(rt, "-concat");
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

/// Does this quoted datum contain a collection literal (vector/map/set) anywhere?
/// Such a datum can't stay a plain `(quote …)` literal, because collection
/// literals must evaluate to the runtime persistent types, not the reader's
/// list-backed records — so it is rebuilt with constructor calls instead.
fn datum_has_coll<M: ValueModel>(rt: &Runtime<M>, datum: u64) -> bool {
    if record_field0(rt, datum, reader::VECTOR).is_some()
        || record_field0(rt, datum, reader::MAP).is_some()
        || record_field0(rt, datum, reader::SET).is_some()
    {
        return true;
    }
    if rt.as_cons(datum).is_some() {
        return rt.list_to_vec(datum).into_iter().any(|e| datum_has_coll(rt, e));
    }
    false
}

/// Rebuild a quoted datum as code: vector/map/set literals become `(vector …)` /
/// `(hash-map …)` / `(hash-set …)` constructor calls (so they evaluate to the
/// persistent runtime types), a list containing a collection becomes `(list …)`,
/// and every leaf stays a `(quote leaf)`.
fn build_quote<M: ValueModel>(rt: &mut Runtime<M>, datum: u64) -> u64 {
    for (tag, ctor) in [
        (reader::VECTOR, "vector"),
        (reader::MAP, "hash-map"),
        (reader::SET, "hash-set"),
    ] {
        if let Some(lst) = record_field0(rt, datum, tag) {
            let elems = rt.list_to_vec(lst);
            let mut out = vec![sym(rt, ctor)];
            out.extend(elems.into_iter().map(|e| build_quote(rt, e)));
            return rt.vec_to_list(&out);
        }
    }
    if rt.as_cons(datum).is_some() && datum_has_coll(rt, datum) {
        let elems = rt.list_to_vec(datum);
        let mut out = vec![sym(rt, "list")];
        out.extend(elems.into_iter().map(|e| build_quote(rt, e)));
        return rt.vec_to_list(&out);
    }
    quote_form(rt, datum)
}

fn call1<M: ValueModel>(rt: &mut Runtime<M>, f: &str, arg: u64) -> u64 {
    let fs = sym(rt, f);
    rt.vec_to_list(&[fs, arg])
}

/// The elements of a binding/param vector, from EITHER the reader's list-backed
/// `Vector` record OR a runtime `PVec`. Syntax-quote and the `vector` constructor
/// both build binding forms (`(loop [~i 0] …)`, `(fn [x] …)`) as runtime PVecs,
/// so the expander must read them structurally here. Returns `None` if `form` is
/// neither representation.
fn binding_items<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Option<Vec<u64>> {
    if let Some(lst) = record_field0(rt, form, reader::VECTOR) {
        return Some(rt.list_to_vec(lst));
    }
    // A runtime vector built by a macro / syntax-quote. Two layouts coexist: the
    // load-time `PVec` (macros expanded while core.clj loads) and the cljs-ported
    // `PersistentVector` (user-time expansion, after cljs types load).
    if record_field0(rt, form, "PVec").is_some() {
        return Some(pvec_elems(rt, form));
    }
    if record_field0(rt, form, "PersistentVector").is_some() {
        return Some(persistent_vector_elems(rt, form));
    }
    None
}

/// Read a cljs `PersistentVector`'s elements: fields [meta cnt shift root tail
/// __hash], where `root` is a `VectorNode` (arr = its field 1) and `tail` is a
/// raw array. Mirrors cljs `unchecked-array-for`.
fn persistent_vector_elems<M: ValueModel>(rt: &Runtime<M>, pv: u64) -> Vec<u64> {
    let cnt = pvec_int_field(rt, pv, 1);
    let shift = pvec_int_field(rt, pv, 2);
    let root = pvec_ref_field(rt, pv, 3);
    let tail = pvec_ref_field(rt, pv, 4);
    let node_arr = |rt: &Runtime<M>, node: u64| pvec_ref_field(rt, node, 1);
    let tail_off = if cnt < 32 { 0 } else { ((cnt - 1) >> 5) << 5 };
    let mut out = Vec::with_capacity(cnt);
    for i in 0..cnt {
        let arr = if i >= tail_off {
            tail
        } else {
            let mut node = root;
            let mut level = shift;
            while level > 0 {
                node = arr_get(rt, node_arr(rt, node), (i >> level) & 31);
                level -= 5;
            }
            node_arr(rt, node)
        };
        out.push(arr_get(rt, arr, i & 31));
    }
    out
}

fn pvec_int_field<M: ValueModel>(rt: &Runtime<M>, pv: u64, i: usize) -> usize {
    let Val::Ref(id) = rt.decode(pv) else { panic!("PVec: not a record") };
    let Obj::Record { fields, .. } = &rt.heap()[id as usize] else { panic!("PVec: not a record") };
    match rt.decode(fields[i]) {
        Val::Int(n) => n as usize,
        _ => panic!("PVec: field {i} is not an int"),
    }
}
fn pvec_ref_field<M: ValueModel>(rt: &Runtime<M>, pv: u64, i: usize) -> u64 {
    let Val::Ref(id) = rt.decode(pv) else { panic!("PVec: not a record") };
    let Obj::Record { fields, .. } = &rt.heap()[id as usize] else { panic!("PVec: not a record") };
    fields[i]
}
fn arr_get<M: ValueModel>(rt: &Runtime<M>, arr: u64, i: usize) -> u64 {
    let Val::Ref(id) = rt.decode(arr) else { panic!("PVec node: not an array") };
    let Obj::Vector(v) = &rt.heap()[id as usize] else { panic!("PVec node: not an array") };
    v[i]
}
/// Read a PVec's logical elements by walking its trie (mirrors core's `-pv-nth`).
fn pvec_elems<M: ValueModel>(rt: &Runtime<M>, pv: u64) -> Vec<u64> {
    let cnt = pvec_int_field(rt, pv, 0);
    let shift = pvec_int_field(rt, pv, 1);
    let root = pvec_ref_field(rt, pv, 2);
    let tail = pvec_ref_field(rt, pv, 3);
    let tail_off = if cnt < 32 { 0 } else { ((cnt - 1) >> 5) << 5 };
    let mut out = Vec::with_capacity(cnt);
    for i in 0..cnt {
        let node = if i >= tail_off {
            tail
        } else {
            let mut n = root;
            let mut level = shift;
            while level > 0 {
                n = arr_get(rt, n, (i >> level) & 31);
                level -= 5;
            }
            n
        };
        out.push(arr_get(rt, node, i & 31));
    }
    out
}

fn record_field0<M: ValueModel>(rt: &Runtime<M>, form: u64, tag: &str) -> Option<u64> {
    if let Val::Ref(id) = rt.decode(form) {
        if let Obj::Record { type_id, fields } = &rt.heap()[id as usize] {
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

/// Strip a leading `(quote X)` wrapper, returning `X` (or the form unchanged).
fn unquote<M: ValueModel>(rt: &Runtime<M>, f: u64) -> u64 {
    if let Some((h, _)) = rt.as_cons(f) {
        if is_sym(rt, h, "quote") {
            return rt.list_to_vec(f)[1];
        }
    }
    f
}

/// If `f` is `(resolve 'x)` or `(ns-resolve 'ns 'x)` with LITERAL quoted symbols,
/// return the fully-qualified sym it resolves to (namespace resolution is a
/// compile-time operation over the current/ named namespace). Else `None`.
fn resolve_rewrite<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, f: u64) -> Option<Sym> {
    let (head, _) = rt.as_cons(f)?;
    let items = rt.list_to_vec(f);
    if is_sym(rt, head, "resolve") && items.len() == 2 {
        // `(resolve 'x)` -> resolve `x` in the CURRENT namespace.
        if let Val::Sym(s) = rt.decode(unquote(rt, items[1])) {
            return Some(comp.resolve_ref(rt, s));
        }
    } else if is_sym(rt, head, "ns-resolve") && items.len() == 3 {
        // `(ns-resolve 'ns 'x)` -> the qualified sym `ns/x` directly.
        let ns = sym_name_of(rt, items[1])?;
        let name = sym_name_of(rt, items[2])?;
        return Some(rt.intern(&format!("{ns}/{name}")));
    }
    None
}

/// The name of a (possibly quoted) symbol form, e.g. `foo.bar` or `'foo.bar`.
fn sym_name_of<M: ValueModel>(rt: &Runtime<M>, f: u64) -> Option<String> {
    match rt.decode(unquote(rt, f)) {
        Val::Sym(s) => Some(rt.sym_name(s).to_string()),
        _ => None,
    }
}

/// Handle a namespace declaration form (`ns`/`in-ns`/`require`/`use`/`alias`/
/// `refer`). These are COMPILE-TIME only: they mutate the compiler's resolution
/// state and produce nil. Returns `None` if `form` isn't such a declaration.
fn handle_ns_form<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    form: u64,
) -> Option<u64> {
    let (head, _) = rt.as_cons(form)?;
    let nil = rt.encode(Val::Nil);
    if is_sym(rt, head, "ns") {
        let items = rt.list_to_vec(form);
        if let Some(name) = items.get(1).and_then(|&f| sym_name_of(rt, f)) {
            comp.set_ns(&name);
            // This ns is being defined here (not from a file); don't re-load it.
            comp.mark_loaded(&name);
        }
        for &clause in items.get(2..).unwrap_or(&[]) {
            process_ns_clause(rt, cs, macros, comp, clause);
        }
        return Some(nil);
    }
    if is_sym(rt, head, "in-ns") {
        let items = rt.list_to_vec(form);
        if let Some(name) = items.get(1).and_then(|&f| sym_name_of(rt, f)) {
            comp.set_ns(&name);
            comp.mark_loaded(&name);
        }
        return Some(nil);
    }
    if is_sym(rt, head, "require") || is_sym(rt, head, "use") {
        let items = rt.list_to_vec(form);
        for &spec in &items[1..] {
            process_require_spec(rt, cs, macros, comp, spec);
        }
        return Some(nil);
    }
    if is_sym(rt, head, "alias") {
        let items = rt.list_to_vec(form);
        if let (Some(a), Some(real)) =
            (items.get(1).and_then(|&f| sym_name_of(rt, f)), items.get(2).and_then(|&f| sym_name_of(rt, f)))
        {
            comp.add_alias(&a, &real);
        }
        return Some(nil);
    }
    if is_sym(rt, head, "refer") {
        let items = rt.list_to_vec(form);
        if let Some(real) = items.get(1).and_then(|&f| sym_name_of(rt, f)) {
            refer_names(rt, comp, &items[2..], &real, "only");
        }
        return Some(nil);
    }
    None
}

/// A `(:require …)` / `(:use …)` clause inside an `ns` form.
fn process_ns_clause<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    clause: u64,
) {
    let items = rt.list_to_vec(clause);
    if items.is_empty() {
        return;
    }
    if is_keyword(rt, items[0], "require") || is_keyword(rt, items[0], "use") {
        for &spec in &items[1..] {
            process_require_spec(rt, cs, macros, comp, spec);
        }
    }
    // :refer-clojure / :import — core is auto-referred; we model no host imports.
}

/// A single require spec: `foo`, `[foo :as bar]`, or `[foo :refer [x y]]`. LOADS
/// the namespace's file (once) before wiring up any alias/refer.
fn process_require_spec<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    spec: u64,
) {
    let spec = unquote(rt, spec);
    // bare `(require 'foo)` — load it, nothing to alias/refer.
    if let Some(real) = sym_name_of(rt, spec) {
        ensure_loaded(rt, cs, macros, comp, &real);
        return;
    }
    let Some(elems) = binding_items(rt, spec) else { return };
    let Some(real) = elems.first().and_then(|&f| sym_name_of(rt, f)) else { return };
    ensure_loaded(rt, cs, macros, comp, &real);
    let mut k = 1;
    while k < elems.len() {
        if is_keyword(rt, elems[k], "as") && k + 1 < elems.len() {
            if let Some(alias) = sym_name_of(rt, elems[k + 1]) {
                comp.add_alias(&alias, &real);
            }
            k += 2;
        } else if is_keyword(rt, elems[k], "refer") && k + 1 < elems.len() {
            refer_names(rt, comp, &[elems[k + 1]], &real, "");
            k += 2;
        } else {
            k += 1;
        }
    }
}

/// Add refers for `[x y]` (a vector of names) under namespace `real`. When `kw`
/// is non-empty the list is preceded by that keyword marker (`:only [x y]`).
fn refer_names<M: ValueModel>(rt: &mut Runtime<M>, comp: &mut Compiler, forms: &[u64], real: &str, kw: &str) {
    let list = if kw.is_empty() {
        forms.first().copied()
    } else {
        forms.iter().position(|&f| is_keyword(rt, f, kw)).and_then(|i| forms.get(i + 1).copied())
    };
    let Some(list) = list else { return };
    if let Some(names) = binding_items(rt, unquote(rt, list)) {
        for nm in names {
            if let Some(short) = sym_name_of(rt, nm) {
                comp.add_refer(&short, &format!("{real}/{short}"));
            }
        }
    }
}

/// Load a required namespace from disk if it isn't already loaded. Maps ns
/// `foo.bar` to `foo/bar.clj` (`.cljc`/`.cljs` also tried; hyphens in a segment
/// become underscores, as Clojure munges file paths) and searches each load-path
/// directory. Errors clearly if the namespace can't be located.
fn ensure_loaded<M: ValueModel>(
    rt: &mut Runtime<M>,
    cs: &dyn CodeSpace<M>,
    macros: &mut HashSet<Sym>,
    comp: &mut Compiler,
    name: &str,
) {
    if comp.is_loaded(name) {
        return;
    }
    let rel = name.replace('.', "/").replace('-', "_");
    let mut found = None;
    for dir in comp.load_paths() {
        for ext in ["clj", "cljc", "cljs"] {
            let p = dir.join(format!("{rel}.{ext}"));
            if p.is_file() {
                found = Some(p);
                break;
            }
        }
        if found.is_some() {
            break;
        }
    }
    let Some(path) = found else {
        panic!(
            "require: cannot find namespace `{name}` (looked for `{rel}.clj` on load path {:?})",
            comp.load_paths()
        );
    };
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("require: failed reading {}: {e}", path.display()));
    // Mark loaded BEFORE running so a cyclic require terminates.
    comp.mark_loaded(name);
    let saved = comp.current_ns().to_string();
    run_src(rt, cs, macros, comp, &src);
    // The loaded file's `(ns …)` moved us into it; restore the requiring ns.
    comp.set_ns(&saved);
}

/// Map the simple (last dotted segment of a) Java/JS class name to the runtime
/// tag `type-of` reports, or `None` if we don't model it. Shared by `instance?`
/// and typed `catch`.
fn class_to_tag(simple: &str) -> Option<&'static str> {
    Some(match simple {
        "Symbol" => "Symbol",
        "Keyword" => "Keyword",
        "String" | "CharSequence" => "String",
        "Character" => "Char",
        "Long" | "Integer" | "Short" | "Byte" | "BigInteger" | "BigInt" => "Long",
        "Double" | "Float" | "BigDecimal" => "Double",
        "Boolean" => "Boolean",
        s if s.contains("Vector") => "PersistentVector",
        s if s.contains("Map") => "PersistentArrayMap",
        s if s.contains("Set") => "PersistentHashSet",
        s if s.contains("List") || s == "ISeq" || s == "Seqable" || s == "Cons" => "List",
        s if s == "IFn" || s == "AFn" || s == "Fn" => "Fn",
        _ => return None,
    })
}

/// `(instance? Class x)` -> `(%num-eq (type-of x) 'Tag)`, mapping the (possibly
/// package-qualified) class name to the runtime tag `type-of` reports. Panics on
/// an unknown class rather than silently returning a wrong answer.
fn instance_rewrite<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let Val::Sym(cs) = rt.decode(items[1]) else {
        panic!("instance?: first argument must be a class symbol");
    };
    let cname = rt.sym_name(cs).to_string();
    let simple = cname.rsplit('.').next().unwrap_or(&cname);
    // A deftype/record instance is tagged by its own simple name, so fall back to
    // that when the class isn't a built-in we map.
    let tag = class_to_tag(simple).unwrap_or(simple).to_string();
    let numeq = sym(rt, "%num-eq");
    let typeof_sym = sym(rt, "type-of");
    let quote_sym = sym(rt, "quote");
    let tag_sym = sym(rt, &tag);
    let tag_quoted = rt.vec_to_list(&[quote_sym, tag_sym]);
    let typeof_call = rt.vec_to_list(&[typeof_sym, items[2]]);
    rt.vec_to_list(&[numeq, typeof_call, tag_quoted])
}

/// Desugar `(try body… (catch Class e h…)… (catch :default e h…) (finally f…))`
/// into `(try* (do body…) EXC DISPATCH (do f…))` — a fixed-shape low-level form
/// the compiler maps to `Ir::Try`. DISPATCH is a nested-`if` that tests the
/// thrown value's runtime tag against each clause's class (ClojureScript's
/// `instanceof`-chain model), binds the clause's name, and re-`throw`s on no
/// match. A `:default` clause, or a base class (`Throwable`/`Exception`/`Error`/
/// `Object`), matches anything. `try*`/`EXC`/`DISPATCH`/`finally` are `nil` when
/// the corresponding part is absent.
fn desugar_try<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let mut body_forms: Vec<u64> = Vec::new();
    let mut catches: Vec<Vec<u64>> = Vec::new();
    let mut finally: Option<Vec<u64>> = None;
    for &f in &items[1..] {
        let head = rt.as_cons(f).map(|_| rt.list_to_vec(f)[0]);
        match head {
            Some(h) if is_sym(rt, h, "catch") => catches.push(rt.list_to_vec(f)),
            Some(h) if is_sym(rt, h, "finally") => finally = Some(rt.list_to_vec(f)[1..].to_vec()),
            _ => body_forms.push(f),
        }
    }
    let do_sym = sym(rt, "do");
    let nil = rt.encode(Val::Nil);
    let mut body = vec![do_sym];
    body.extend_from_slice(&body_forms);
    let body = rt.vec_to_list(&body);

    let (catchbind, dispatch) = if catches.is_empty() {
        (nil, nil)
    } else {
        let exc = gensym(rt, "exc");
        // Fold clauses back-to-front so the first listed clause is tried first.
        let throwsym = sym(rt, "throw");
        let mut chain = rt.vec_to_list(&[throwsym, exc]); // no clause matched -> re-throw
        for clause in catches.iter().rev() {
            // clause = (catch <class-or-:default> <bind> body…)
            let class = clause[1];
            let bind = clause[2];
            let test = catch_test(rt, class, exc);
            // (let (bind exc) body…)
            let letsym = sym(rt, "let");
            let bindlist = rt.vec_to_list(&[bind, exc]);
            let mut letf = vec![letsym, bindlist];
            letf.extend_from_slice(&clause[3..]);
            let handler = rt.vec_to_list(&letf);
            let ifsym = sym(rt, "if");
            chain = rt.vec_to_list(&[ifsym, test, handler, chain]);
        }
        (exc, chain)
    };

    let finally_form = match finally {
        Some(fs) => {
            let mut f = vec![do_sym];
            f.extend_from_slice(&fs);
            rt.vec_to_list(&f)
        }
        None => nil,
    };

    let trystar = sym(rt, "try*");
    rt.vec_to_list(&[trystar, body, catchbind, dispatch, finally_form])
}

/// `(binding [*x* v …] body…)` -> `(try* (do (%dyn-mark) (%dyn-bind 'ns/*x* v) …
/// body…) nil nil (%dyn-unwind))`. The `%dyn-mark` delimits this scope on the
/// per-thread binding stack; the `finally` `%dyn-unwind` pops it (so bindings
/// unwind even when the body throws). Each var is RESOLVED to its qualified
/// dynamic sym so it keys the stack the same way references (`%dyn-get`) do.
fn desugar_binding<M: ValueModel>(rt: &mut Runtime<M>, comp: &Compiler, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let pairs = binding_items(rt, items[1]).unwrap_or_else(|| rt.list_to_vec(items[1]));
    let quote = sym(rt, "quote");
    let dynbind = sym(rt, "%dyn-bind");
    let do_sym = sym(rt, "do");
    let nil = rt.encode(Val::Nil);

    let mut body = vec![do_sym];
    let mark = sym(rt, "%dyn-mark");
    body.push(rt.vec_to_list(&[mark]));
    let mut k = 0;
    while k + 1 < pairs.len() {
        // Resolve the var to its qualified dynamic sym (matching `%dyn-get`).
        let resolved = match rt.decode(pairs[k]) {
            Val::Sym(s) => rt.encode(Val::Sym(comp.resolve_ref(rt, s))),
            _ => pairs[k],
        };
        let qvar = rt.vec_to_list(&[quote, resolved]);
        body.push(rt.vec_to_list(&[dynbind, qvar, pairs[k + 1]]));
        k += 2;
    }
    body.extend_from_slice(&items[2..]);
    let body = rt.vec_to_list(&body);

    let unwind = {
        let u = sym(rt, "%dyn-unwind");
        rt.vec_to_list(&[u])
    };
    let trystar = sym(rt, "try*");
    rt.vec_to_list(&[trystar, body, nil, nil, unwind])
}

/// `(with-redefs [f newf …] body…)` -> save each var's current root, install the
/// replacements, run the body, and (in a `finally`) restore the roots. Operates on
/// var ROOTS by symbol (`(var f)` resolves the name), so it composes with the
/// namespace-qualified var model.
fn desugar_with_redefs<M: ValueModel>(rt: &mut Runtime<M>, form: u64) -> u64 {
    let items = rt.list_to_vec(form);
    let pairs = binding_items(rt, items[1]).unwrap_or_else(|| rt.list_to_vec(items[1]));
    let var_s = sym(rt, "var");
    let vget = sym(rt, "var-get");
    let vset = sym(rt, "var-set");
    let let_s = sym(rt, "let");
    let do_s = sym(rt, "do");
    let trystar = sym(rt, "try*");
    let nil = rt.encode(Val::Nil);

    let mut let_binds = Vec::new(); // [old_i (var-get (var name_i)) …]
    let mut sets = Vec::new(); // (var-set (var name_i) new_i)
    let mut restores = Vec::new(); // (var-set (var name_i) old_i)
    let mut k = 0;
    while k + 1 < pairs.len() {
        let (name, new) = (pairs[k], pairs[k + 1]);
        let old = gensym(rt, "old");
        let vf = |rt: &mut Runtime<M>| rt.vec_to_list(&[var_s, name]);
        let getf = {
            let v = vf(rt);
            rt.vec_to_list(&[vget, v])
        };
        let_binds.push(old);
        let_binds.push(getf);
        let setf = {
            let v = vf(rt);
            rt.vec_to_list(&[vset, v, new])
        };
        sets.push(setf);
        let restf = {
            let v = vf(rt);
            rt.vec_to_list(&[vset, v, old])
        };
        restores.push(restf);
        k += 2;
    }
    let finally = {
        let mut fin = vec![do_s];
        fin.extend(restores);
        rt.vec_to_list(&fin)
    };
    let body = {
        let mut b = vec![do_s];
        b.extend_from_slice(&items[2..]);
        rt.vec_to_list(&b)
    };
    let tryf = rt.vec_to_list(&[trystar, body, nil, nil, finally]);
    let bindvec = make_vector(rt, let_binds);
    let mut letform = vec![let_s, bindvec];
    letform.extend(sets);
    letform.push(tryf);
    rt.vec_to_list(&letform)
}

/// The test form for one catch clause: `true` for a catch-all (`:default` or a
/// base class), else `(%num-eq (type-of exc) 'Tag)`.
fn catch_test<M: ValueModel>(rt: &mut Runtime<M>, class: u64, exc: u64) -> u64 {
    let catch_all = if is_keyword(rt, class, "default") {
        true
    } else if let Val::Sym(s) = rt.decode(class) {
        let simple = last_seg(rt.sym_name(s));
        matches!(simple.as_str(), "Throwable" | "Exception" | "Error" | "Object")
    } else {
        false
    };
    if catch_all {
        return rt.encode(Val::Bool(true));
    }
    let Val::Sym(s) = rt.decode(class) else {
        panic!("catch: expected a class symbol or :default");
    };
    let simple = last_seg(rt.sym_name(s));
    let tag = class_to_tag(&simple).unwrap_or(&simple).to_string();
    let numeq = sym(rt, "%num-eq");
    let typeof_sym = sym(rt, "type-of");
    let typeof_call = rt.vec_to_list(&[typeof_sym, exc]);
    let tag_sym = sym(rt, &tag);
    let tag_q = quote_form(rt, tag_sym);
    rt.vec_to_list(&[numeq, typeof_call, tag_q])
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
        Val::Ref(id) => match &rt.heap()[id as usize] {
            Obj::Record { type_id, fields } => {
                let tag = rt.sym_name(*type_id);
                match tag {
                    "Keyword" => format!(":{}", rt.print(fields[0])),
                    "Vector" => format!("[{}]", list_items(rt, fields[0], " ")),
                    "Set" => format!("#{{{}}}", list_items(rt, fields[0], " ")),
                    "Map" => format!("{{{}}}", map_items(rt, fields[0])),
                    "Var" => format!("#'{}", rt.print(fields[0])),
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
