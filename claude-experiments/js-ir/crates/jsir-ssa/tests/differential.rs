//! **Instrument 2: the behavioral differential (incl. a fuzzer).**
//!
//! "Is the CFG *right*?" has a precise behavioral answer: interpreting the CFG
//! must compute the same thing the program does. We run each program three
//! ways and require agreement:
//!   1. Node on the original JavaScript,
//!   2. our interpreter on the SSA CFG ([`jsir_ssa::interp`]),
//! and we drive it with three sources of programs:
//!   - a **destructured-param battery** (validates the param lowering against
//!     Node, including object/array args),
//!   - a **closure red-spec battery** (closures that mutate a captured value;
//!     the CFG drops closure bodies today, so these *cannot* be interpreted —
//!     that red is the spec for the closure-body fix, see `CFG_FIDELITY.md`),
//!   - a **seeded fuzzer** that generates programs in the lowerable subset
//!     (simple + destructured params, arithmetic, if/else, bounded loops,
//!     object/array mutation) and diffs CFG-interp against Node at scale.
//!
//! A program that *fails to lower* is a coverage gap, not a failure (skipped).
//! A program that lowers but whose CFG-interp diverges from Node (or errors for
//! an unregistered reason) is a real fidelity bug and fails the test.

use jsir_ssa::interp::{self, Val};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Node harness (supports object/array arguments, unlike tests/oracle.rs).
// ---------------------------------------------------------------------------

const TAG_JS: &str = r#"
function __tag(v){
  if (v === undefined) return "u:";
  if (v === null) return "l:";
  if (typeof v === "boolean") return "b:"+v;
  if (typeof v === "number") return "n:"+(Number.isNaN(v)?"NaN":v===Infinity?"Infinity":v===-Infinity?"-Infinity":String(v));
  if (typeof v === "string") return "s:"+v;
  if (Array.isArray(v)) return "a:["+v.map(__tag).join(",")+"]";
  if (typeof v === "object"){var ks=Object.keys(v).sort();return "o:{"+ks.map(function(k){return k+"="+__tag(v[k]);}).join(",")+"}";}
  return "?:"+typeof v;
}
"#;

fn node_available() -> bool {
    std::process::Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// A JS source literal for an argument value (primitives, objects, arrays).
fn js_literal(v: &Val) -> String {
    match v {
        Val::Undef => "undefined".into(),
        Val::Null => "null".into(),
        Val::Bool(b) => b.to_string(),
        Val::Num(n) => {
            if n.is_nan() {
                "NaN".into()
            } else if n.is_infinite() {
                if *n > 0.0 { "Infinity".into() } else { "-Infinity".into() }
            } else {
                interp::js_num_to_string(*n)
            }
        }
        Val::Str(s) => format!("{s:?}"),
        Val::Arr(a) => {
            let elems = a.borrow().iter().map(js_literal).collect::<Vec<_>>().join(",");
            format!("[{elems}]")
        }
        Val::Obj(m) => {
            let props = m
                .borrow()
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), js_literal(v)))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{props}}}")
        }
        Val::Closure { .. } => unreachable!("closures are never passed as arguments"),
    }
}

fn run_node(src: &str, args: &[Val]) -> Option<String> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let arglist = args.iter().map(js_literal).collect::<Vec<_>>().join(", ");
    let program = format!("{TAG_JS}\n{src}\nconsole.log(__tag(f({arglist})));\n");
    let dir = std::env::temp_dir();
    let path = dir.join(format!("jsir_diff_{}_{}.js", std::process::id(), C.fetch_add(1, Ordering::Relaxed)));
    std::fs::File::create(&path).ok()?.write_all(program.as_bytes()).ok()?;
    let out = std::process::Command::new("node").arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

// ---------------------------------------------------------------------------
// The differential check for one (program, args).
// ---------------------------------------------------------------------------

/// Reasons CFG-interp can legitimately refuse a program *without* it being a
/// fidelity bug — each corresponds to a registered known loss / coverage gap.
/// Anything else is a real divergence and fails.
const KNOWN_INTERP_LOSSES: &[&str] = &[
    "interp: call unsupported",        // closures/calls: bodies dropped (KNOWN_LOSSY)
    "interp: free global",             // free globals not modeled
];

#[derive(Debug, PartialEq)]
enum Outcome {
    /// Node and CFG-interp agreed on the tagged result.
    Agree,
    /// The program did not lower (coverage gap) — not a fidelity failure.
    LowerBailed,
    /// CFG-interp refused for a registered known-loss reason (e.g. a closure).
    KnownLoss(String),
}

fn check(src: &str, args: &[Val]) -> Result<Outcome, String> {
    let mut cfg = match jsir_ssa::lower(src) {
        Ok(c) => c,
        Err(_) => return Ok(Outcome::LowerBailed),
    };
    jsir_ssa::ssa::construct(&mut cfg);
    let errs = jsir_ssa::verify::verify(&cfg);
    if !errs.is_empty() {
        return Err(format!("SSA verify failed: {}", errs.join("; ")));
    }
    let ours = match interp::run(&cfg, args) {
        Ok(v) => interp::tag(&v),
        Err(e) => {
            if KNOWN_INTERP_LOSSES.iter().any(|k| e.contains(k)) {
                return Ok(Outcome::KnownLoss(e));
            }
            return Err(format!("CFG-interp errored unexpectedly: {e}"));
        }
    };
    let node = match run_node(src, args) {
        Some(t) => t,
        None => return Ok(Outcome::LowerBailed), // node parse/runtime error: not our concern
    };
    if node != ours {
        return Err(format!("DIVERGENCE: node={node} ours={ours}"));
    }
    Ok(Outcome::Agree)
}

// ---------------------------------------------------------------------------
// Battery 1: destructured params (validates the param lowering vs Node).
// ---------------------------------------------------------------------------

fn obj(pairs: &[(&str, f64)]) -> Val {
    let mut m = BTreeMap::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), Val::Num(*v));
    }
    Val::Obj(Rc::new(RefCell::new(m)))
}
fn arr(xs: &[f64]) -> Val {
    Val::Arr(Rc::new(RefCell::new(xs.iter().map(|x| Val::Num(*x)).collect())))
}

#[test]
fn destructured_params_match_node() {
    if !node_available() {
        eprintln!("node unavailable; skipping");
        return;
    }
    let cases: &[(&str, Vec<Val>)] = &[
        ("function f({a,b}){ return a+b; }", vec![obj(&[("a", 3.0), ("b", 4.0)])]),
        ("function f({a,b}){ a=a*2; return a-b; }", vec![obj(&[("a", 5.0), ("b", 1.0)])]),
        ("function f([x,y]){ return x*y; }", vec![arr(&[3.0, 5.0])]),
        ("function f([x,y,z]){ return x+y+z; }", vec![arr(&[1.0, 2.0, 3.0])]),
        ("function f({a},c){ return a+c; }", vec![obj(&[("a", 10.0)]), Val::Num(2.0)]),
        ("function f({a,b},c){ let o={s:a+b+c}; o.s=o.s*2; return o.s; }", vec![obj(&[("a",1.0),("b",2.0)]), Val::Num(3.0)]),
        ("function f({a}){ let s=0; let i=0; while(i<a){ s=s+i; i=i+1; } return s; }", vec![obj(&[("a", 5.0)])]),
    ];
    let mut n = 0;
    for (src, args) in cases {
        match check(src, args) {
            Ok(Outcome::Agree) => n += 1,
            Ok(other) => panic!("[{src}] expected agreement, got {other:?}"),
            Err(e) => panic!("[{src}]: {e}"),
        }
    }
    eprintln!("destructured_params_match_node: {n} cases agreed with node");
}

// ---------------------------------------------------------------------------
// Battery 2: closures that mutate a captured value through a call. The closure
// body is now lowered into a nested CFG and the interpreter executes it with
// the captured object shared by reference, so the mutation is observable. This
// was the red spec for the closure-body fix; it is now GREEN (validated against
// Node), which is what lets the body loss be retired.
// ---------------------------------------------------------------------------

const CLOSURE_PROGRAMS: &[&str] = &[
    "function f(n){ let z={a:n}; let g=function(){ z.a=z.a+1; }; g(); return z.a; }",
    "function f(n){ let z={a:n}; let g=()=>{ z.a=z.a*2; }; g(); g(); return z.a; }",
    "function f(n){ let a=[n]; let push=()=>{ a[1]=a[0]+1; }; push(); return a[1]; }",
    "function f(n){ let g=function(q){ return q+n; }; return g(10); }",
    "function f(n){ let z={a:n,b:0}; let g=()=>{ z.b=z.a*3; }; g(); return z.a+z.b; }",
    "function f(n){ let s={v:0}; let add=function(x){ s.v=s.v+x; }; add(n); add(n); add(1); return s.v; }",
    "function f(n){ let o={c:n}; let inc=function(){ o.c=o.c+1; return o.c; }; let a=inc(); let b=inc(); return a+b+o.c; }",
    // Arrows WITH parameters (recovered from the arrow op's operands), including
    // destructured params (bound through the same machinery as `const {a}=x`).
    "function f(n){ let s={v:0}; let add=(x)=>{ s.v=s.v+x; }; add(n); add(n); add(1); return s.v; }",
    "function f(n){ let g=(a,b)=>{ return a+b+n; }; return g(1,2); }",
    "function f(n){ let g=({a})=>{ return a+n; }; return g({a:3}); }",
    "function f(n){ let g=([x,y])=>{ return x*y+n; }; return g([3,4]); }",
    "function f(n){ let g=(p,{a})=>{ return p+a+n; }; return g(10,{a:2}); }",
];

#[test]
fn spread_matches_node() {
    if !node_available() {
        eprintln!("node unavailable; skipping");
        return;
    }
    let cases: &[(&str, Vec<Val>)] = &[
        ("function f(a){ let b=[...a, 9]; return b[0]+b[1]+b[2]; }", vec![arr(&[1.0, 2.0])]),
        ("function f(a){ let b=[0, ...a, 100]; return b[0]+b[1]+b[2]+b[3]; }", vec![arr(&[3.0, 4.0])]),
        ("function f(o){ let r={...o, b:2}; return r.a+r.b; }", vec![obj(&[("a", 5.0)])]),
        ("function f(o){ let r={a:1, ...o}; return r.a; }", vec![obj(&[("a", 9.0)])]), // later spread overrides
        ("function f(a){ let g=function(x,y){ return x*y; }; return g(...a); }", vec![arr(&[3.0, 4.0])]),
    ];
    let mut n = 0;
    for (src, args) in cases {
        match check(src, args) {
            Ok(Outcome::Agree) => n += 1,
            Ok(other) => panic!("[{src}] expected agreement, got {other:?}"),
            Err(e) => panic!("[{src}]: {e}"),
        }
    }
    eprintln!("spread_matches_node: {n} cases agreed with node");
}

#[test]
fn closures_with_captured_mutation_match_node() {
    if !node_available() {
        eprintln!("node unavailable; skipping");
        return;
    }
    let mut n = 0;
    for src in CLOSURE_PROGRAMS {
        for &input in &[0.0, 1.0, 2.0, 5.0] {
            match check(src, &[Val::Num(input)]) {
                Ok(Outcome::Agree) => n += 1,
                Ok(other) => panic!(
                    "[{src}] f({input}): expected agreement, got {other:?} \
                     (closure bodies should now be executable)"
                ),
                Err(e) => panic!("[{src}] f({input}): {e}"),
            }
        }
    }
    eprintln!("closures_with_captured_mutation_match_node: {n} (program,input) pairs agreed with node");
}

// ---------------------------------------------------------------------------
// Battery 3: the fuzzer. Generates programs in the lowerable subset and diffs
// CFG-interp against Node. Catches "lowers but computes the wrong thing".
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

/// One generated parameter: a binding shape plus the names it introduces.
struct ParamSpec {
    /// The JS source for the param (e.g. `p0`, `{a,b}`, `[x,y]`).
    decl: String,
    /// The in-scope numeric variable names it binds.
    names: Vec<String>,
    /// Build a concrete argument for this param from a seed.
    arg: Val,
}

fn gen_param(rng: &mut Rng, idx: usize) -> ParamSpec {
    match rng.below(3) {
        0 => {
            let n = format!("p{idx}");
            let v = (rng.below(9)) as f64 - 4.0;
            ParamSpec { decl: n.clone(), names: vec![n], arg: Val::Num(v) }
        }
        1 => {
            let a = format!("a{idx}");
            let b = format!("b{idx}");
            let av = (rng.below(9)) as f64 - 4.0;
            let bv = (rng.below(9)) as f64 - 4.0;
            ParamSpec {
                decl: format!("{{{a},{b}}}"),
                names: vec![a.clone(), b.clone()],
                arg: obj(&[(a.as_str(), av), (b.as_str(), bv)]),
            }
        }
        _ => {
            let x = format!("x{idx}");
            let y = format!("y{idx}");
            let xv = (rng.below(9)) as f64 - 4.0;
            let yv = (rng.below(9)) as f64 - 4.0;
            ParamSpec {
                decl: format!("[{x},{y}]"),
                names: vec![x, y],
                arg: arr(&[xv, yv]),
            }
        }
    }
}

/// A numeric expression over in-scope names + small literals.
fn gen_expr(rng: &mut Rng, scope: &[String], depth: u32) -> String {
    if depth == 0 || scope.is_empty() || rng.below(3) == 0 {
        if scope.is_empty() || rng.below(2) == 0 {
            return format!("{}", rng.below(9));
        }
        return rng.pick(scope).clone();
    }
    let op = *rng.pick(&["+", "-", "*"]);
    let a = gen_expr(rng, scope, depth - 1);
    let b = gen_expr(rng, scope, depth - 1);
    format!("({a} {op} {b})")
}

fn gen_cond(rng: &mut Rng, scope: &[String]) -> String {
    let op = *rng.pick(&["<", "<=", ">", ">=", "===", "!=="]);
    let a = gen_expr(rng, scope, 1);
    let b = gen_expr(rng, scope, 1);
    format!("{a} {op} {b}")
}

/// Generate one program; returns (source, args).
fn gen_program(seed: u64) -> (String, Vec<Val>) {
    let mut rng = Rng(seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493));
    let nparams = 1 + rng.below(2) as usize;
    let mut scope: Vec<String> = Vec::new();
    let mut decls: Vec<String> = Vec::new();
    let mut args: Vec<Val> = Vec::new();
    for i in 0..nparams {
        let p = gen_param(&mut rng, i);
        decls.push(p.decl);
        scope.extend(p.names);
        args.push(p.arg);
    }
    let mut body = String::new();
    let nstmts = 2 + rng.below(4);
    for s in 0..nstmts {
        match rng.below(7) {
            4 => {
                // A closure that captures a fresh object and mutates it through
                // one or more calls; the mutation is then observed via a plain
                // local so it reaches the return value. Exercises closure-body
                // lowering (function vs arrow, 0 vs 1 param, captured mutation).
                let o = format!("o{s}");
                let g = format!("g{s}");
                let r = format!("r{s}");
                let init = gen_expr(&mut rng, &scope, 1);
                let step = gen_expr(&mut rng, &scope, 1);
                body.push_str(&format!("let {o} = {{ q: {init} }}; "));
                let arrow = rng.below(2) == 0;
                if rng.below(2) == 0 {
                    // one-parameter closure, called twice with different args
                    let (a1, a2) = (gen_expr(&mut rng, &scope, 1), gen_expr(&mut rng, &scope, 1));
                    if arrow {
                        body.push_str(&format!("let {g} = (d) => {{ {o}.q = {o}.q + d; }}; "));
                    } else {
                        body.push_str(&format!("let {g} = function(d) {{ {o}.q = {o}.q + d; }}; "));
                    }
                    body.push_str(&format!("{g}({a1}); {g}({a2}); "));
                } else {
                    // no-parameter closure capturing an outer expression
                    if arrow {
                        body.push_str(&format!("let {g} = () => {{ {o}.q = {o}.q + ({step}); }}; "));
                    } else {
                        body.push_str(&format!("let {g} = function() {{ {o}.q = {o}.q + ({step}); }}; "));
                    }
                    body.push_str(&format!("{g}(); {g}(); "));
                }
                body.push_str(&format!("let {r} = {o}.q; "));
                scope.push(r);
            }
            _ => match rng.below(5) {
            0 => {
                // local numeric binding
                let v = format!("v{s}");
                let e = gen_expr(&mut rng, &scope, 2);
                body.push_str(&format!("let {v} = {e}; "));
                scope.push(v);
            }
            1 => {
                // reassignment of an existing name
                if let Some(name) = scope.get(rng.below(scope.len() as u64) as usize).cloned() {
                    let e = gen_expr(&mut rng, &scope, 2);
                    body.push_str(&format!("{name} = {e}; "));
                }
            }
            2 => {
                // if/else assigning a fresh var on both arms
                let v = format!("v{s}");
                let c = gen_cond(&mut rng, &scope);
                let t = gen_expr(&mut rng, &scope, 1);
                let e = gen_expr(&mut rng, &scope, 1);
                body.push_str(&format!("let {v}; if ({c}) {{ {v} = {t}; }} else {{ {v} = {e}; }} "));
                scope.push(v);
            }
            3 => {
                // object with a mutated property
                let o = format!("o{s}");
                let e1 = gen_expr(&mut rng, &scope, 1);
                let e2 = gen_expr(&mut rng, &scope, 2);
                body.push_str(&format!("let {o} = {{ p: {e1}, q: 0 }}; {o}.q = {e2}; "));
                scope.push(format!("{o}.p"));
            }
            _ => {
                // bounded accumulation loop
                let acc = format!("acc{s}");
                let k = format!("k{s}");
                let bound = 1 + rng.below(4);
                let e = gen_expr(&mut rng, &scope, 1);
                body.push_str(&format!(
                    "let {acc} = 0; let {k} = 0; while ({k} < {bound}) {{ {acc} = {acc} + {e}; {k} = {k} + 1; }} "
                ));
                scope.push(acc);
            }
            },
        }
    }
    // Return a numeric expression over plain (non-member) names.
    let ret_scope: Vec<String> = scope.iter().filter(|n| !n.contains('.')).cloned().collect();
    let ret = gen_expr(&mut rng, &ret_scope, 2);
    let params = decls.join(", ");
    (format!("function f({params}) {{ {body}return {ret}; }}"), args)
}

#[test]
fn fuzz_cfg_matches_node() {
    if !node_available() {
        eprintln!("node unavailable; skipping fuzzer");
        return;
    }
    let count: u64 = std::env::var("JSIR_FUZZ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(250);
    let mut agreed = 0u64;
    let mut bailed = 0u64;
    let mut known_loss = 0u64;
    for seed in 0..count {
        let (src, args) = gen_program(seed);
        match check(&src, &args) {
            Ok(Outcome::Agree) => agreed += 1,
            Ok(Outcome::LowerBailed) => bailed += 1,
            Ok(Outcome::KnownLoss(_)) => known_loss += 1,
            Err(e) => panic!("FUZZ FAILURE @seed {seed}\n  src: {src}\n  args: {args:?}\n  {e}"),
        }
    }
    eprintln!(
        "fuzz_cfg_matches_node: {count} programs -> {agreed} agreed, {bailed} bailed (coverage), {known_loss} known-loss"
    );
    assert!(agreed > count / 2, "fuzzer agreed on only {agreed}/{count}; generator likely off-subset");
}
