//! Instrument 1 gate: the static fidelity audit over the React fixture corpus.
//!
//! Scans every fixture's base IR, collects the reachable op-kinds, classifies
//! each (faithful / hard-error / known-lossy / wrapper), and:
//!   1. fails on any internal-consistency violation (stale/dead registry rows);
//!   2. ratchets the full classification against a checked-in snapshot, so a
//!      new op-kind, or a construct silently changing fidelity bucket, must be
//!      reviewed and the snapshot regenerated on purpose.
//!
//! Regenerate the snapshot intentionally with `JSIR_FIDELITY_REGEN=1`.
//! Point at a corpus with `REACT_FIXTURES=/path/to/fixtures/compiler`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use jsir_ir::Op as IrOp;
use jsir_ssa::fidelity;

fn fixtures_dir() -> PathBuf {
    if let Ok(d) = std::env::var("REACT_FIXTURES") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(format!(
        "{home}/Documents/Code/open-source/react-rust-pr36173/compiler/packages/babel-plugin-react-compiler/src/__tests__/fixtures/compiler"
    ))
}

fn collect(op: &IrOp, out: &mut BTreeSet<String>) {
    out.insert(op.name.clone());
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                collect(o, out);
            }
        }
    }
}

fn reachable_op_kinds(dir: &PathBuf) -> BTreeSet<String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read fixtures {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "js").unwrap_or(false))
        .collect();
    files.sort();
    let mut kinds = BTreeSet::new();
    for p in &files {
        let Ok(src) = std::fs::read_to_string(p) else { continue };
        let Ok(ir) = jsir_swc::source_to_ir(&src) else { continue };
        collect(&ir, &mut kinds);
    }
    kinds
}

/// Representative minimal snippet exercising each not-faithful op-kind, plus the
/// observed lowering outcome. This makes the audit's `hard-error` / `ignored`
/// classification **empirical**: we assert the lowering really bails (resp.
/// really lowers) on each, instead of inferring it from `HANDLED_OP_KINDS`.
/// `true` = `lower()` must return `Err` (loud bail); `false` = must lower OK.
const BEHAVIOR_PROBES: &[(&str, &str, bool)] = &[
    ("jshir.break_statement", "function f(n){ while(n){ break; } return n; }", true),
    ("jshir.continue_statement", "function f(n){ while(n){ continue; } return n; }", true),
    ("jshir.do_while_statement", "function f(n){ do{ n=n-1; }while(n>0); return n; }", true),
    ("jshir.for_statement", "function f(n){ for(let i=0;i<n;i++){} return n; }", true),
    ("jshir.for_of_statement", "function f(a){ for(const x of a){} return a; }", true),
    ("jshir.for_in_statement", "function f(o){ for(const k in o){} return o; }", true),
    ("jshir.switch_statement", "function f(n){ switch(n){case 1:return 1;} return 0; }", true),
    ("jshir.labeled_statement", "function f(n){ outer: while(n){ break outer; } return n; }", true),
    ("jshir.try_statement", "function f(n){ try{return n;}catch(e){return 0;} }", true),
    ("jsir.throw_statement", "function f(n){ throw n; }", true),
    ("jsir.class_declaration", "function f(){ class C{} return C; }", true),
    ("jsir.await_expression", "async function f(n){ return await n; }", true),
    ("jsir.yield_expression", "function* f(n){ yield n; }", true),
    ("jsir.this_expression", "function f(){ return this.x; }", true),
    ("jsir.debugger_statement", "function f(n){ debugger; return n; }", true),
    ("jsir.reg_exp_literal", "function f(){ return /ab/g; }", true),
    ("jsir.meta_property", "function f(){ return new.target; }", true),
    ("jsir.none", "function f(){ let a=[1,,3]; return a; }", true),
    // The two silent miscompiles the audit surfaced, now loud bails:
    ("jsir.assignment_pattern_ref", "function f({a=1}){ return a; }", true),
    ("jsir.rest_element_ref", "function f(a,...rest){ return rest; }", true),
    // Faithful-by-omission: these lower without bailing and lose nothing.
    ("jsir.import_declaration", "import x from 'y'; function f(){ return x; }", false),
    ("jsir.directive", "function f(n){ 'use strict'; return n; }", false),
];

/// The classification must match observed lowering behavior, not an assumption.
#[test]
fn fidelity_classification_matches_behavior() {
    for (kind, src, must_bail) in BEHAVIOR_PROBES {
        let lowered = jsir_ssa::lower(src).is_ok();
        let bailed = !lowered;
        assert_eq!(
            bailed, *must_bail,
            "{kind}: probe `{src}` expected bail={must_bail} but lower() bailed={bailed}"
        );
        // Cross-check against the static classification.
        match fidelity::classify(kind) {
            fidelity::Fidelity::HardError => assert!(
                *must_bail,
                "{kind} classified HardError but its probe lowers OK (silent skip?)"
            ),
            fidelity::Fidelity::Ignored(_) | fidelity::Fidelity::Faithful => assert!(
                !*must_bail,
                "{kind} classified faithful/ignored but its probe bails"
            ),
            other => panic!("{kind}: unexpected classification {other:?} for a behavior probe"),
        }
    }
}

#[test]
fn fidelity_audit_corpus() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!("corpus dir {} missing; skipping fidelity audit", dir.display());
        return;
    }
    let kinds = reachable_op_kinds(&dir);
    assert!(
        kinds.len() > 50,
        "only {} op-kinds reachable; corpus likely not parsing (dir={})",
        kinds.len(),
        dir.display()
    );

    let audit = fidelity::audit(&kinds);

    // 1. No internal-consistency violations (stale / dead / double-handled rows).
    assert!(
        audit.violations.is_empty(),
        "fidelity registry violations:\n  {}",
        audit.violations.join("\n  ")
    );

    let (faithful, hard, lossy, wrapper) = audit.counts();
    eprintln!(
        "fidelity: {} op-kinds  ({faithful} faithful, {hard} hard-error, {lossy} known-lossy, {wrapper} wrapper)",
        kinds.len()
    );

    // 2. Snapshot ratchet.
    let snap_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fidelity_snapshot.txt");
    let actual = audit.snapshot();
    if std::env::var("JSIR_FIDELITY_REGEN").is_ok() || !snap_path.exists() {
        std::fs::write(&snap_path, &actual).expect("write snapshot");
        eprintln!("wrote fidelity snapshot to {}", snap_path.display());
        return;
    }
    let expected = std::fs::read_to_string(&snap_path).expect("read snapshot");
    if expected != actual {
        // Produce a readable diff of which op-kinds changed bucket / appeared.
        let exp: BTreeSet<&str> = expected.lines().collect();
        let act: BTreeSet<&str> = actual.lines().collect();
        let added: Vec<&&str> = act.difference(&exp).collect();
        let removed: Vec<&&str> = exp.difference(&act).collect();
        panic!(
            "fidelity classification changed (review, then JSIR_FIDELITY_REGEN=1 to accept):\n\
             + new/changed:\n   {}\n- gone/old:\n   {}",
            added.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n   "),
            removed.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("\n   "),
        );
    }
}
