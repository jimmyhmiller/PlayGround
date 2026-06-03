//! Golden / structural tests for the React-compiler analyses (mutable ranges and
//! reactive scopes). These assert the *shape* of the analysis (which values are
//! mutable, how scopes group) rather than runtime values.

use jsir_ssa::aliasing_ranges::analyze;
use jsir_ssa::mutability::Ranges;
use jsir_ssa::scopes::{self, Scope};
use jsir_ssa::{cfg::Value, lower, ssa};

fn analyzed(src: &str) -> (jsir_ssa::Cfg, Ranges) {
    let mut cfg = lower(src).expect("lower");
    ssa::construct(&mut cfg);
    let r = analyze(&cfg);
    (cfg, r)
}

/// Values that are mutated after their definition.
fn mutated_refs(r: &Ranges) -> Vec<Value> {
    let mut v: Vec<Value> = r.range.keys().copied().filter(|&v| r.is_mutable_after_def(v)).collect();
    v.sort();
    v
}

#[test]
fn pure_object_is_immutable() {
    // No store/call -> object never mutated -> freely memoizable.
    let (_c, r) = analyzed("function f(){ let x = {a:1, b:2}; return x.a + x.b; }");
    assert!(mutated_refs(&r).is_empty(), "pure object should be immutable");
}

#[test]
fn stored_object_is_mutable() {
    let (_c, r) = analyzed("function f(){ let x = {}; x.a = 1; return x; }");
    assert_eq!(mutated_refs(&r).len(), 1, "object with a store is mutable");
}

#[test]
fn mutated_array_is_mutable() {
    let (_c, r) = analyzed("function f(n){ let a=[]; let i=0; while(i<n){ a[i]=i; i=i+1; } return a; }");
    assert_eq!(mutated_refs(&r).len(), 1, "array mutated in loop is mutable");
}

#[test]
fn call_argument_is_treated_as_mutated() {
    // Sound over-approximation: passing a value to a call may mutate it.
    let (_c, r) = analyzed("function f(g){ let x = {}; g(x); return x; }");
    assert_eq!(mutated_refs(&r).len(), 1, "call argument conservatively mutable");
}

#[test]
fn capture_aliases_extend_range() {
    // `inner` is captured into `outer`, then `outer` is mutated; the alias means
    // `inner`'s mutable set must cover that mutation.
    let (_c, r) = analyzed("function f(){ let inner={}; let outer={x:inner}; outer.y=1; return outer; }");
    // Both inner and outer share an alias root and a non-trivial range.
    let muts = mutated_refs(&r);
    assert!(!muts.is_empty(), "captured-and-mutated objects are mutable: {muts:?}");
}

// --- reactive scopes ---

fn scopes_of(src: &str) -> Vec<Scope> {
    let (cfg, r) = analyzed(src);
    scopes::infer(&cfg, &r)
}

#[test]
fn pure_construction_makes_independent_scopes() {
    // Two pure objects -> two independent (pure) scopes; none mutable.
    let s = scopes_of("function f(p){ let a={x:p}; let b={y:p}; return [a,b]; }");
    assert!(s.iter().all(|sc| !sc.mutable), "pure construction has no mutable scopes: {s:?}");
    assert!(s.len() >= 2, "each pure object is its own scope: {s:?}");
}

#[test]
fn mutation_forms_single_scope() {
    // The object and both its stores collapse into one mutable scope.
    let s = scopes_of("function f(){ let x={}; x.a=1; x.b=2; return x; }");
    let mutable: Vec<&Scope> = s.iter().filter(|sc| sc.mutable).collect();
    assert_eq!(mutable.len(), 1, "one merged mutable scope: {s:?}");
    let m = mutable[0];
    assert!(m.end > m.start, "mutable scope spans the stores: {m:?}");
}

#[test]
fn dependency_inference_matches_memo_shape() {
    // `function f(a,b){ let style={color:a}; let el={size:b,props:style}; return el; }`
    // An escaping allocation is always memoized (React's `getMemoizationLevel`),
    // so this produces TWO scopes — `style` (dep `a`) and `el` (deps `b`,
    // `style`) — for a cache of 5 slots. Verified byte-for-byte against the
    // official compiler, which emits exactly `const $ = _c(5)` with these two
    // memo blocks. (An earlier port mis-modeled bare allocations as not memoized;
    // this asserts the corrected, oracle-matching shape.)
    let (cfg, r) = analyzed("function f(a,b){ let style={color:a}; let el={size:b,props:style}; return el; }");
    let infos = scopes::analyze(&cfg, &r);
    let emitted: Vec<_> = infos.iter().filter(|i| !i.outputs.is_empty()).collect();
    assert_eq!(emitted.len(), 2, "two memoized scopes (style, el): {infos:?}");
    let cache: usize = emitted.iter().map(|i| i.deps.len() + i.outputs.len()).sum();
    assert_eq!(cache, 5, "cache size matches React's _c(5): {infos:?}");
}

#[test]
fn single_dependency_consumer_merges() {
    // A consumer with exactly one reactive dep (its producer's output) folds into
    // that producer — they invalidate together. (Matches the React Compiler.)
    let (cfg, r) = analyzed("function f(p){ let a={x:p}; let b={y:a}; return b; }");
    let infos = scopes::analyze(&cfg, &r);
    let pure: Vec<_> = infos.iter().filter(|i| !i.scope.mutable).collect();
    assert_eq!(pure.len(), 1, "b (single dep a) merges into a: {infos:?}");
    assert_eq!(pure[0].scope.values.len(), 2, "scope owns both a and b");
}

#[test]
fn two_dependency_consumer_stays_separate() {
    // A consumer with two reactive deps stays its own scope, even if one dep is a
    // subset of the producer's — React keeps both identities independent.
    let (cfg, r) = analyzed("function f(p){ let a={x:p}; let b={y:p, z:a}; return b; }");
    let infos = scopes::analyze(&cfg, &r);
    let pure: Vec<_> = infos.iter().filter(|i| !i.scope.mutable).collect();
    assert_eq!(pure.len(), 2, "a and b stay separate (b has 2 deps): {infos:?}");
}

#[test]
fn ssa_then_analyses_never_panic() {
    // Broad smoke test: a variety of shapes lower, go to SSA, and analyze.
    for src in [
        "function f(n){ let s=0; let i=0; while(i<n){ s=s+i; i=i+1; } return s; }",
        "function f(a,b){ return a>b ? {hi:a} : {lo:b}; }",
        "function f(p){ let o={}; if(p){ o.a=1; } else { o.b=2; } return o; }",
        "function f(n){ let xs=[]; let i=0; while(i<n){ xs[i]={v:i}; i=i+1; } return xs; }",
    ] {
        let (cfg, r) = analyzed(src);
        let _ = scopes::infer(&cfg, &r);
        assert!(jsir_ssa::verify::verify(&cfg).is_empty(), "[{src}] verify failed");
    }
}
