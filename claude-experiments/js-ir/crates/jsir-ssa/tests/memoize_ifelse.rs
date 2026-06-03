//! Phase B Step 3: if/else (recursive region walk, RULE 2).
//!
//! The production `compile` path now memoizes structured `if`/`else` functions
//! by wrapping each branch-local reactive scope in its own memo guard. This test
//! pins the two things that matter:
//!
//!  1. SOUNDNESS — a scope whose allocation is mutated across a branch boundary
//!     (alloc-before-`if`, alloc-inside-branch, aliased-mutation, mutate-phi)
//!     MUST hard-error (`Err`) and stay unmemoized, never miscompile.
//!  2. CORRECTNESS — for the cases we DO memoize, the emitted JS computes the
//!     same value as the original across many prop sets AND is reference-stable
//!     when dependencies are unchanged (verified under Node).

use jsir_ssa::codegen;

fn node_available() -> bool {
    std::process::Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_node(program: &str) -> Option<String> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "jsir_ifelse_{}_{}.js",
        std::process::id(),
        C.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::File::create(&path).ok()?.write_all(program.as_bytes()).ok()?;
    let out = std::process::Command::new("node").arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        eprintln!("node error: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// `(cache_size, memo_block_count)` from compiled output (mirrors the gate's
/// `structure()`), or None if not memoized.
fn structure(code: &str) -> Option<(usize, usize)> {
    let n = code.split("_c(").nth(1)?.split(')').next()?.trim().parse::<usize>().ok()?;
    let block_count = code
        .match_indices("if (")
        .filter(|(i, _)| code[i + 4..].trim_start_matches(['(', ' ']).starts_with("$["))
        .count();
    Some((n, block_count))
}

// A persistent per-instance cache + a React.createElement stub, so the emitted
// code (which lowers JSX to createElement) runs under Node and we can observe
// reference stability across calls.
const PRELUDE: &str = r#"
const _e = Symbol('empty');
let __c = null;
function _c(n){ if(!__c) __c = new Array(n).fill(_e); return __c; }
const React = { createElement: (t, p, ...kids) => ({ type: t, props: p || {}, kids }) };
function __tag(v){
  if (v === undefined) return "u:";
  if (v === null) return "l:";
  if (typeof v !== "object") return typeof v + ":" + String(v);
  if (Array.isArray(v)) return "a:[" + v.map(__tag).join(",") + "]";
  const ks = Object.keys(v).sort();
  return "o:{" + ks.map(k => k + "=" + __tag(v[k])).join(",") + "}";
}
"#;

/// Compile `src` (must be a component) and assert it memoizes with `(cache,
/// blocks)`, then check value-equivalence and reference-stability under Node.
fn check_memoized(src: &str, want_structure: (usize, usize), prop_sets: &[&str]) {
    let memo = codegen::compile(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"));
    assert_eq!(
        structure(&memo),
        Some(want_structure),
        "structure mismatch for\n{src}\n--- memoized ---\n{memo}"
    );

    if !node_available() {
        eprintln!("node unavailable; skipping semantic check for [{src}]");
        return;
    }
    // The emitted module uses `import {c as _c}`; strip the import and rely on the
    // prelude's `_c`. The component keeps its name `Component`.
    let body = memo.lines().filter(|l| !l.trim_start().starts_with("import ")).collect::<Vec<_>>().join("\n");

    for ps in prop_sets {
        // (a) The reuse path (else-branch cache restore) must yield the SAME value
        //     as the recompute path (then-branch). We populate the cache, then run
        //     with a *fresh* cache, and compare tagged values.
        let recompute_vs_reuse = run_node(&format!(
            "{PRELUDE}\n{body}\n\
             const reuse1 = Component({ps}); const reuse2 = Component({ps});\n\
             __c = null;\n\
             const fresh = Component({ps});\n\
             console.log([__tag(reuse2) === __tag(fresh), reuse1 === reuse2].join(','));"
        ));
        assert_eq!(
            recompute_vs_reuse.as_deref(),
            Some("true,true"),
            "reuse path differs from recompute, or not reference-stable, on {ps} for\n{src}\n--- memoized ---\n{memo}"
        );
    }
}

/// Like [`check_memoized`] but ALSO compares the memoized value against the
/// original (non-memoized) source under Node — only valid for sources Node can
/// run directly (no JSX).
fn check_memoized_vs_original(src: &str, want_structure: (usize, usize), prop_sets: &[&str]) {
    check_memoized(src, want_structure, prop_sets);
    if !node_available() {
        return;
    }
    let memo = codegen::compile(src).unwrap();
    let body = memo.lines().filter(|l| !l.trim_start().starts_with("import ")).collect::<Vec<_>>().join("\n");
    for ps in prop_sets {
        let memo_val = run_node(&format!("{PRELUDE}\n{body}\nconsole.log(__tag(Component({ps})));"));
        let orig_val = run_node(&format!("{PRELUDE}\n{src}\nconsole.log(__tag(Component({ps})));"));
        assert_eq!(memo_val, orig_val, "memoized value != original on {ps} for\n{src}\n--- memoized ---\n{memo}");
    }
}

/// Assert `src` is NOT memoized — it must hard-error (deferred), leaving the
/// function un-memoized rather than miscompiled.
fn assert_bails(src: &str) {
    match codegen::compile(src) {
        Err(_) => {} // expected: deferred / unsound -> hard error
        Ok(out) => {
            // A pass-through (no cache) is also acceptable (some shapes round-trip
            // without memoizing). What is NOT acceptable is emitting a memo guard.
            assert!(
                structure(&out).is_none(),
                "expected NO memoization (unsound to memoize), but got one:\n{src}\n--- output ---\n{out}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// REQUIRED hard-error counterexamples (confirmed silent-wrong if memoized).
// ---------------------------------------------------------------------------

#[test]
fn alloc_before_if_then_mutate_in_branch_bails() {
    // `a` is allocated before the `if`; the branch mutates it. A single guard
    // around the alloc would NOT enclose the in-branch mutation.
    assert_bails("function Component(props){ const a=[]; if(props.c){ a.push(props.x); } return a; }");
}

#[test]
fn alloc_inside_branch_used_after_bails() {
    // `a` is allocated (and mutated) inside one branch, allocated in the other,
    // and consumed after the join. The scope's owned statements span the branch.
    assert_bails(
        "function Component(props){ let a; if(props.c){ a=[]; a.push(props.p0); } else { a=[]; } return Foo(a); }",
    );
}

#[test]
fn aliased_mutation_through_property_load_after_branch_bails() {
    // `x.y = y; x.y.push(...)` mutates the cached `{}` via a member load off it,
    // then the branch condition `x.y[0]` reads that mutated allocation. The value
    // selected by the branch (`z`) is therefore control-dependent on mutable state
    // we don't surface as a dependency, so memoizing it would be unsound. Must bail.
    assert_bails(
        "function Component(props){ const x={}; const y=[]; x.y=y; x.y.push(props.input); let z=0; if(x.y[0]){z=1;} return [z]; }",
    );
}

#[test]
fn mutate_phi_after_branch_bails() {
    // `x` is a phi (frozen-or-`{}`); `x.property = true` after the join mutates
    // whichever value flowed in — including the cached `{}`.
    assert_bails(
        "function Component(props){ let x; if(props.cond){ x=frozen(); } else { x={}; } x.property = true; return x; }",
    );
}

// ---------------------------------------------------------------------------
// Fully-enclosed if/else scopes: each branch builds an independent value whose
// allocation + (no) mutation live entirely in that branch.
// ---------------------------------------------------------------------------

#[test]
fn jsx_in_each_branch_memoizes_per_branch() {
    // Two branch-local JSX scopes, each with one reactive dep: cache=4, 2 blocks.
    check_memoized(
        "function Component(props) {\n  let el;\n  if (props.cond) {\n    el = <div a={props.a} />;\n  } else {\n    el = <span b={props.b} />;\n  }\n  return el;\n}",
        (4, 2),
        &[
            "{cond:true, a:1, b:2}",
            "{cond:false, a:1, b:2}",
            "{cond:true, a:9, b:2}",
            "{cond:false, a:9, b:5}",
        ],
    );
}

#[test]
fn obj_literal_in_each_branch_is_memoized_per_branch() {
    // `o = {x: props.a}` / `o = {y: props.b}` then `return o`. An escaping
    // allocation is always memoized (like the JSX case above), so each branch
    // gets its own per-branch object scope: cache=4, 2 blocks. Verified against
    // the official compiler (`_c(4)`), and behaviorally under Node. (An earlier
    // port mis-modeled bare objects as un-memoized; this asserts the corrected,
    // oracle-matching shape.)
    check_memoized_vs_original(
        "function Component(props) {\n  let o;\n  if (props.cond) {\n    o = {x: props.a};\n  } else {\n    o = {y: props.b};\n  }\n  return o;\n}",
        (4, 2),
        &[
            "{cond:true, a:1, b:2}",
            "{cond:false, a:1, b:2}",
            "{cond:true, a:9, b:2}",
            "{cond:false, a:9, b:5}",
        ],
    );
}

#[test]
fn straight_line_still_memoizes() {
    // Regression guard: the single-block path still memoizes an escaping JSX
    // element keyed on its reactive prop. (A returned *bare object* like
    // `const b = {y: props.p, z: a}; return b;` is intentionally NOT memoized
    // after the escape-analysis port — React emits it unchanged — so this guard
    // uses a JSX element, which React does memoize.)
    check_memoized(
        "function Component(props) { const el = <div p={props.p} />; return el; }",
        (2, 1),
        &["{p:1}", "{p:2}"],
    );
}

