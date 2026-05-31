//! Validate constant propagation against upstream's golden analysis fixtures
//! (`vendor/.../analyses/constant_propagation/tests/<case>/`).
//!
//! Upstream's annotated dump emits a `// %N = <value>` line for each
//! value-producing op in document order. We reproduce that ordered list of
//! values (ignoring the `%N` labels, whose numbering differs) and compare.

use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("vendor/jsir-upstream/maldoca/js/ir/analyses/constant_propagation/tests")
}

/// Extract the ordered `// %N = <value>` facts from a golden output file.
fn expected_facts(output: &str) -> Vec<String> {
    let mut facts = Vec::new();
    for line in output.lines() {
        // Strip the FileCheck prefix: `// JSHIR[-NEXT]:   <content>`.
        let content = match line.find(':') {
            Some(i) => line[i + 1..].trim_start(),
            None => continue,
        };
        // We want the annotation comments `// %N = value` (not the op lines).
        if let Some(rest) = content.strip_prefix("// %") {
            if let Some(eq) = rest.find(" = ") {
                facts.push(rest[eq + 3..].trim().to_string());
            }
        }
    }
    facts
}

#[test]
fn constant_propagation_matches_golden() {
    let dir = fixtures_dir();
    let mut total = 0;
    let mut passed = 0;
    let mut failures = Vec::new();

    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("fixtures dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let case = entry.path();
        let name = case.file_name().unwrap().to_string_lossy().to_string();
        let input = match std::fs::read_to_string(case.join("input.js")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let golden = match std::fs::read_to_string(case.join("output.generated.txt")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        total += 1;

        let expected = expected_facts(&golden);
        let result = std::panic::catch_unwind(|| {
            let op = jsir_swc::source_to_ir(&input).map_err(|e| format!("lower: {e}"))?;
            Ok::<_, String>(jsir_analyses::analyze_constants(&op))
        });
        let ours = match result {
            Ok(Ok(f)) => f,
            Ok(Err(e)) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
            Err(_) => {
                failures.push(format!("{name}: PANIC"));
                continue;
            }
        };

        if ours == expected {
            passed += 1;
        } else {
            let n = expected.len().max(ours.len());
            let mut diff = String::new();
            for i in 0..n {
                let e = expected.get(i).map(|s| s.as_str()).unwrap_or("<none>");
                let o = ours.get(i).map(|s| s.as_str()).unwrap_or("<none>");
                if e != o {
                    diff.push_str(&format!("\n    [{i}] upstream={e:?} ours={o:?}"));
                }
            }
            failures.push(format!("{name}: {} facts differ{diff}", expected.len()));
        }
    }

    eprintln!("constant_propagation golden: {passed}/{total} cases match");
    for f in &failures {
        eprintln!("  FAIL {f}");
    }
    // Lock in progress: the implemented constructs must keep passing.
    assert_eq!(
        passed, total,
        "constant propagation regressed: {passed}/{total}. Failures:\n{}",
        failures.join("\n")
    );
}

/// A self-consistency oracle that needs no reference implementation: over every
/// fixture, for *both* the constant-propagation and sign analyses, assert
///   (a) the engine reached a true fixpoint (no loop hit the iteration cap), and
///   (b) the produced lattice values obey the lattice laws (idempotent,
///       commutative, associative, bottom-identity, top-absorbing, monotone,
///       correct change-flag) — the algebraic preconditions for soundness.
/// This validates the generic engine + any lattice without upstream.
#[test]
fn dataflow_self_consistent() {
    use jsir_analyses::dataflow::{check_lattice_laws, run, Lattice};

    let dir = fixtures_dir();
    let mut checked = 0;
    let mut problems: Vec<String> = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("fixtures dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let case = entry.path();
        let name = case.file_name().unwrap().to_string_lossy().to_string();
        let Ok(input) = std::fs::read_to_string(case.join("input.js")) else { continue };
        let Ok(op) = jsir_swc::source_to_ir(&input) else { continue };
        checked += 1;

        // Constant propagation.
        let cp = run(&jsir_analyses::ConstProp, &op);
        if !cp.converged {
            problems.push(format!("{name}: constprop did not converge"));
        }
        let cp_vals: Vec<_> = cp.values.values().cloned().collect();
        let cp_laws = check_lattice_laws(&cp_vals);
        if !cp_laws.is_empty() {
            problems.push(format!("{name}: constprop lattice laws: {cp_laws:?}"));
        }

        // Sign analysis (a totally different lattice on the same engine).
        let sg = run(&jsir_analyses::SignAnalysis, &op);
        if !sg.converged {
            problems.push(format!("{name}: sign did not converge"));
        }
        let sg_vals: Vec<_> = sg.values.values().cloned().collect();
        let sg_laws = check_lattice_laws(&sg_vals);
        if !sg_laws.is_empty() {
            problems.push(format!("{name}: sign lattice laws: {sg_laws:?}"));
        }

        // Render is total (every produced value renders without panicking).
        let _ = cp_vals.iter().map(Lattice::render).collect::<Vec<_>>();
        let _ = sg_vals.iter().map(Lattice::render).collect::<Vec<_>>();
    }

    eprintln!("dataflow self-consistency: {checked} fixtures x 2 analyses");
    assert!(problems.is_empty(), "self-consistency failures:\n{}", problems.join("\n"));
}
