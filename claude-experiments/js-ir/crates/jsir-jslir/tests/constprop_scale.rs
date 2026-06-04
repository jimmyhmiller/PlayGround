//! Scale sanity check for the constant-folding transform: fold every lowered
//! function body across the whole React fixture corpus, then lift + emit. Catches
//! folder/lift bugs the small equivalence tests can't (malformed literal ops,
//! ValueId reuse hazards, lift mismatches). Not an equivalence check — just that
//! folding never panics and always produces re-emittable JS.
//!
//! Ignored by default (it walks ~1700 files); run with
//! `cargo test -p jsir-jslir --test constprop_scale -- --ignored`.

use std::path::PathBuf;

use jsir_ir::Op;
use jsir_jslir::constprop::{constant_lattice, fold_constants, prune_constant_if_branches};
use jsir_jslir::dce::eliminate_dead_code;
use jsir_jslir::ssa::enter_ssa;

#[test]
#[ignore = "walks the full fixture corpus; run explicitly"]
fn fold_every_fixture_without_panicking() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../jsir-react-oracle/fixtures");
    let mut files = 0usize;
    let mut lowered_fns = 0usize;
    let mut folded_ops = 0usize;
    let mut reemitted = 0usize;

    for entry in std::fs::read_dir(&dir).expect("fixtures dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("js") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        // The corpus has TS/Flow files source_to_ir rejects; skip what won't parse.
        let Ok(file) = jsir_swc::source_to_ir(&src) else { continue };
        files += 1;

        let (mut jslir, _) = jsir_jslir::build_jslir(&file);
        let base = max_value(&jslir) + 1;
        fold_all(&mut jslir, base, &mut lowered_fns, &mut folded_ops);

        // Must still lift + emit cleanly after folding.
        let lifted = jsir_jslir::lift_jslir(&jslir);
        if jsir_swc::ir_to_source(&lifted).is_ok() {
            reemitted += 1;
        } else {
            panic!("re-emit failed after folding: {}", path.display());
        }
    }

    eprintln!(
        "fixtures parsed={files}, lowered fns folded over={lowered_fns}, ops folded={folded_ops}, re-emitted={reemitted}"
    );
    assert_eq!(reemitted, files, "every parsed fixture must re-emit after folding");
    assert!(folded_ops > 0, "the corpus should contain *some* foldable constants");
}

fn fold_all(op: &mut Op, base: u32, lowered_fns: &mut usize, folded_ops: &mut usize) {
    const FUNCTION_OPS: &[&str] = &[
        "jsir.function_declaration",
        "jsir.function_expression",
        "jsir.arrow_function_expression",
        "jsir.object_method",
        "jsir.class_method",
        "jsir.class_private_method",
    ];
    if FUNCTION_OPS.contains(&op.name.as_str()) {
        let idx = if op.name == "jsir.function_declaration" || op.name == "jsir.function_expression"
        {
            1
        } else {
            0
        };
        if let Some(b) = op.regions.get_mut(idx) {
            if jsir_jslir::dialect::region_is_cfg(b) {
                let mut next = base;
                if let Some(info) = enter_ssa(b, &mut next) {
                    *lowered_fns += 1;
                    let lattice = constant_lattice(b, &info);
                    *folded_ops += fold_constants(b, &lattice);
                    let lattice = constant_lattice(b, &info);
                    prune_constant_if_branches(b, &lattice);
                }
                // DCE runs regardless of SSA (it doesn't need def-use chains).
                eliminate_dead_code(b);
            }
        }
    }
    for r in &mut op.regions {
        for b in &mut r.blocks {
            for o in &mut b.ops {
                fold_all(o, base, lowered_fns, folded_ops);
            }
        }
    }
}

fn max_value(op: &Op) -> u32 {
    let mut m = op.results.iter().map(|v| v.0).max().unwrap_or(0);
    for r in &op.regions {
        for b in &r.blocks {
            m = m.max(b.args.iter().map(|v| v.0).max().unwrap_or(0));
            for o in &b.ops {
                m = m.max(max_value(o));
            }
        }
    }
    m
}
