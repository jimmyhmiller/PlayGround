//! **Soundness oracle** (no reference implementation needed): execute the
//! program for real and verify each analysis's abstract value over-approximates
//! every concrete runtime value observed at that point.
//!
//! Pipeline: lower -> run analysis -> for each safe r-value op, record its
//! source span + abstract value -> instrument the source so each such
//! expression logs its concrete value -> run under node -> assert every observed
//! concrete value lies in γ(abstract). A claimed-unreachable (`⊥`) point must
//! never log; a `⊤` claim is unfalsifiable (skipped).

use std::collections::HashMap;

use jsir_analyses::dataflow::run;
use jsir_analyses::dataflow::Lattice as _; // bring trait methods (bottom/render) into scope
use jsir_analyses::sign::Sign;
use jsir_analyses::{Const, ConstProp, Lattice, SignAnalysis};
use jsir_ir::{Op, ValueId};

/// IR ops whose value equals a safe, wrappable r-value expression.
const PROBE_OPS: &[&str] = &[
    "jsir.numeric_literal",
    "jsir.string_literal",
    "jsir.boolean_literal",
    "jsir.null_literal",
    "jsir.big_int_literal",
    "jsir.binary_expression",
    "jsir.unary_expression",
    "jsir.identifier",
    "jsir.parenthesized_expression",
    "jshir.logical_expression",
    "jshir.conditional_expression",
];

/// Collect `(start, end, value-id)` for every probeable op in the IR.
fn collect_probes(op: &Op, out: &mut Vec<(i64, i64, ValueId)>) {
    if PROBE_OPS.contains(&op.name.as_str()) {
        if let (Some(r), Some(t)) = (op.results.first(), op.trivia.as_ref()) {
            if let (Some(s), Some(e)) = (t.start, t.end) {
                out.push((s, e, *r));
            }
        }
    }
    for region in &op.regions {
        for block in &region.blocks {
            for inner in &block.ops {
                collect_probes(inner, out);
            }
        }
    }
}

/// Run the instrumented program under node and return `id -> [tagged values]`.
fn run_node(js: &str) -> Option<HashMap<String, Vec<String>>> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir();
    // Unique per call (tests run in parallel threads in one process).
    let uniq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("jsir_soundness_{}_{}.js", std::process::id(), uniq));
    std::fs::File::create(&path).ok()?.write_all(js.as_bytes()).ok()?;
    let out = std::process::Command::new("node").arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        return None; // program threw at runtime; skip it
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().find_map(|l| l.strip_prefix("__PROBES__"))?;
    serde_json::from_str(line).ok()
}

/// Result of checking one analysis on one program.
struct Report {
    /// Number of (concrete observation, abstract claim) pairs that were checked.
    checked: usize,
    /// Soundness violations: the abstract value did NOT cover a concrete value.
    violations: Vec<String>,
}

/// Run `analysis` on `src`, instrument, execute, and verify γ for every value.
/// `gamma` returns `Ok(())` if the tagged concrete value lies in γ(abstract),
/// or `Err(msg)` if it's a soundness violation. `is_top`/`is_bottom` identify
/// the unfalsifiable / unreachable claims.
fn check<A: jsir_analyses::Analysis>(
    src: &str,
    analysis: &A,
    gamma: impl Fn(&A::V, &str) -> Result<(), String>,
    is_top: impl Fn(&A::V) -> bool,
    is_bottom: impl Fn(&A::V) -> bool,
) -> Option<Report> {
    let op = jsir_swc::source_to_ir(src).ok()?;
    let values = run(analysis, &op).values;

    let mut probes = Vec::new();
    collect_probes(&op, &mut probes);

    // span -> probe id; probe id -> abstract value.
    let mut span_to_id: HashMap<(i64, i64), i64> = HashMap::new();
    let mut id_to_abs: HashMap<i64, A::V> = HashMap::new();
    for (i, (s, e, vid)) in probes.iter().enumerate() {
        let id = i as i64;
        span_to_id.insert((*s, *e), id);
        let abs = values.get(vid).cloned().unwrap_or_else(A::V::bottom);
        id_to_abs.insert(id, abs);
    }

    let js = jsir_swc::instrument::instrument_probes(src, &span_to_id)?;
    let log = run_node(&js)?;

    let mut report = Report { checked: 0, violations: Vec::new() };
    for (id, abs) in &id_to_abs {
        let obs = log.get(&id.to_string());
        if is_bottom(abs) {
            // Claimed unreachable: it must never have executed.
            if let Some(vals) = obs {
                if !vals.is_empty() {
                    report.violations.push(format!(
                        "claimed <bottom>/unreachable but ran, observed {vals:?}"
                    ));
                }
            }
            continue;
        }
        if is_top(abs) {
            continue; // unfalsifiable
        }
        let Some(vals) = obs else { continue }; // not exercised by this input
        for v in vals {
            report.checked += 1;
            if let Err(msg) = gamma(abs, v) {
                report.violations.push(format!("{} (claim={})", msg, abs.render()));
            }
        }
    }
    Some(report)
}

/// γ for constant propagation: a `Const` must equal the observed value exactly.
fn const_gamma(abs: &jsir_analyses::Lattice, obs: &str) -> Result<(), String> {
    let lattice = abs;
    let c = match lattice {
        jsir_analyses::Lattice::Const(c) => c,
        _ => return Ok(()),
    };
    let ok = match c {
        Const::Num(bits) => {
            let n = f64::from_bits(*bits);
            match obs.strip_prefix("n:") {
                Some("NaN") => n.is_nan(),
                Some(s) => s.parse::<f64>().map(|x| x == n || (x.is_nan() && n.is_nan())).unwrap_or(false),
                None => false,
            }
        }
        Const::Str(s) => obs.strip_prefix("s:") == Some(s.as_str()),
        Const::Bool(b) => obs == &format!("b:{b}"),
        Const::Null => obs == "l:",
        Const::BigInt { value, .. } => obs.strip_prefix("g:") == Some(value.as_str()),
        Const::RegExp { .. } => true, // not compared
    };
    if ok {
        Ok(())
    } else {
        Err(format!("UNSOUND: constprop claim does not cover observed {obs:?}"))
    }
}

/// γ for sign analysis: the observed number must have the claimed sign.
fn sign_gamma(abs: &Sign, obs: &str) -> Result<(), String> {
    let want = match abs {
        Sign::Pos | Sign::Neg | Sign::Zero => abs,
        _ => return Ok(()),
    };
    let n = match obs.strip_prefix("n:") {
        Some("NaN") | Some("Infinity") | Some("-Infinity") => {
            // NaN/Infinity have no finite sign claim here; treat as a miss only
            // if a finite sign was claimed.
            return Err(format!("UNSOUND: sign claim {} but observed {obs:?}", want.render()));
        }
        Some(s) => s.parse::<f64>().ok(),
        None => return Err(format!("UNSOUND: sign claim {} but observed non-number {obs:?}", want.render())),
    };
    let Some(n) = n else { return Ok(()) };
    let ok = match want {
        Sign::Pos => n > 0.0,
        Sign::Neg => n < 0.0,
        Sign::Zero => n == 0.0,
        _ => true,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("UNSOUND: sign claim {} but observed {obs:?}", want.render()))
    }
}

/// Representative programs exercising literals, folding, variables, and control
/// flow (the values whose soundness we want to confirm against real execution).
const PROGRAMS: &[&str] = &[
    "var a = 1 + 2; var b = a * 3; var c = a - b;",
    "var s = 'ab' + 'cd'; var n = (false + '')[1 + 1];",
    "var x = 5; if (x > 3) { x = x - 10; } else { x = x + 1; } var y = x;",
    "var p = 1; var i = 0; while (i < 4) { p = p * 2; i = i + 1; } var q = p;",
    "var t = true && (1 < 2); var u = 0 || 7; var v = null ?? 9;",
    "var k = 1; for (var j = 0; j < 3; j = j + 1) { k = k + j; } var m = k;",
    "var r = -3; var sgn = r < 0 ? -1 : 1; var z = r * sgn;",
    // String/strict/loose comparison folding (constprop comparison operators).
    "var a = 'x' === 'x'; var b = 'x' !== 'y'; var c = 'a' < 'b'; var d = 'b' >= 'b';",
    "var e = (1 == '1'); var f = (0 == ''); var g = (null == null); var h = ('ab' == 'ab');",
    "var i = ('production' !== 'production'); var j = i ? 'dev' : 'prod';",
];

fn node_available() -> bool {
    std::process::Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn constprop_is_sound() {
    if !node_available() {
        eprintln!("node not available; skipping soundness oracle");
        return;
    }
    let mut total = 0;
    let mut viol = Vec::new();
    for src in PROGRAMS {
        if let Some(r) = check(src, &ConstProp, const_gamma, |v| matches!(v, Lattice::Unknown), |v| matches!(v, Lattice::Uninitialized)) {
            total += r.checked;
            for v in r.violations {
                viol.push(format!("[{src}] {v}"));
            }
        }
    }
    eprintln!("constprop soundness: {total} concrete observations checked");
    assert!(viol.is_empty(), "constprop UNSOUND:\n{}", viol.join("\n"));
    assert!(total > 0, "no observations were checked (instrumentation/exec failed)");
}

#[test]
fn sign_is_sound() {
    if !node_available() {
        return;
    }
    let mut total = 0;
    let mut viol = Vec::new();
    for src in PROGRAMS {
        if let Some(r) = check(src, &SignAnalysis, sign_gamma, |v| matches!(v, Sign::Top), |v| matches!(v, Sign::Bottom)) {
            total += r.checked;
            for v in r.violations {
                viol.push(format!("[{src}] {v}"));
            }
        }
    }
    eprintln!("sign soundness: {total} concrete observations checked");
    assert!(viol.is_empty(), "sign UNSOUND:\n{}", viol.join("\n"));
    assert!(total > 0, "no observations were checked");
}

/// Negative control: a deliberately-unsound analysis (constant propagation that
/// folds `-` as `+`) MUST be caught by the oracle. This proves the oracle has
/// teeth — it isn't passing vacuously.
#[test]
fn oracle_catches_an_unsound_analysis() {
    if !node_available() {
        return;
    }
    let mut any_violation = false;
    let mut checked = 0;
    for src in PROGRAMS {
        if let Some(r) = check(src, &BrokenConstProp, const_gamma, |v| matches!(v, Lattice::Unknown), |v| matches!(v, Lattice::Uninitialized)) {
            checked += r.checked;
            if !r.violations.is_empty() {
                any_violation = true;
            }
        }
    }
    assert!(checked > 0);
    assert!(
        any_violation,
        "the soundness oracle FAILED to catch a deliberately-unsound analysis"
    );
}

/// Constant propagation with one bug: subtraction is computed as addition.
struct BrokenConstProp;
impl jsir_analyses::Analysis for BrokenConstProp {
    type V = Lattice;
    fn transfer(&self, op: &Op, cx: &mut jsir_analyses::dataflow::Transfer<Lattice>) {
        // Reuse the real transfer for everything...
        ConstProp.transfer(op, cx);
        // ...then corrupt `a - b` into `a + b`.
        if op.name == "jsir.binary_expression"
            && matches!(cx.attr("operator_"), Some(jsir_ir::Attr::Str(s)) if s == "-")
        {
            if let (Lattice::Const(Const::Num(a)), Lattice::Const(Const::Num(b))) = (cx.operand(0), cx.operand(1)) {
                let bad = f64::from_bits(a) + f64::from_bits(b);
                cx.set_result(Lattice::Const(Const::num(bad)));
            }
        }
    }
    fn is_true(&self, v: &Lattice) -> bool {
        ConstProp.is_true(v)
    }
    fn is_false(&self, v: &Lattice) -> bool {
        ConstProp.is_false(v)
    }
    fn is_nullish(&self, v: &Lattice) -> bool {
        ConstProp.is_nullish(v)
    }
    fn is_nonnullish(&self, v: &Lattice) -> bool {
        ConstProp.is_nonnullish(v)
    }
}
