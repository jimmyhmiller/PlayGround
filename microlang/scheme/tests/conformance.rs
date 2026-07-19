//! R7RS conformance suite for the Scheme frontend.
//!
//! Doubles as the roadmap. Each case is `(area, setup, expr, expect, live)`:
//!   * `expect` is the R7RS `display` of `expr` (given `setup` at top level).
//!   * `live` = we claim to support it now; these MUST pass (regression guard).
//!   * `!live` = pending; run but not asserted, reported as the frontier. When a
//!     pending case starts passing, promote it to `live`.
//!
//! "Knowing we got it right" comes from an objective oracle: if Chicken Scheme
//! (`csi`) is installed, EVERY `expect` is validated against it. So a wrong
//! expected value fails loudly, and a live case is correct iff our frontend's
//! output equals the oracle's.

use std::io::Write;
use std::process::{Command, Stdio};

use microlang::{CekMachine, LowBitModel, Runtime};

#[path = "common/suite.rs"]
mod suite;
use suite::CASES;

fn our_output(setup: &str, expr: &str) -> Option<String> {
    let prog = format!("{setup} {expr}");
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(|| {
        let mut rt = Runtime::<LowBitModel>::new();
        let v = scheme::run(&mut rt, &CekMachine, &prog);
        scheme::write_value(&rt, v)
    })
    .ok();
    std::panic::set_hook(prev);
    out
}

fn csi_path() -> Option<String> {
    Command::new("csi").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().ok()?;
    Some("csi".to_string())
}

fn oracle_output(csi: &str, setup: &str, expr: &str) -> Option<String> {
    let program = format!("{setup}\n(display {expr})(newline)");
    let mut child = Command::new(csi)
        .arg("-q")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(program.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[test]
fn r7rs_conformance() {
    let csi = csi_path();
    if csi.is_none() {
        eprintln!("note: `csi` (Chicken Scheme) not found — expected values not oracle-validated");
    }

    let (mut live_ok, mut live_total) = (0, 0);
    let (mut pending_ok, mut pending_total) = (0, 0);
    let mut live_failures = Vec::new();
    let mut oracle_failures = Vec::new();
    let mut promote = Vec::new();

    for case in CASES {
        // Validate the expected value against the oracle (all cases).
        if let Some(csi) = &csi {
            if let Some(oracle) = oracle_output(csi, case.setup, case.expr) {
                if oracle != case.expect {
                    oracle_failures.push(format!(
                        "[{}] {}  oracle={:?} but expect={:?}",
                        case.area, case.expr, oracle, case.expect
                    ));
                }
            }
        }

        let ours = our_output(case.setup, case.expr);
        let ok = ours.as_deref() == Some(case.expect);
        if case.live {
            live_total += 1;
            if ok {
                live_ok += 1;
            } else {
                live_failures.push(format!(
                    "[{}] {}  got={:?} want={:?}",
                    case.area, case.expr, ours, case.expect
                ));
            }
        } else {
            pending_total += 1;
            if ok {
                pending_ok += 1;
                promote.push(format!("[{}] {}", case.area, case.expr));
            }
        }
    }

    println!("\nR7RS conformance:");
    println!("  live:    {live_ok}/{live_total} passing");
    println!("  pending: {pending_ok}/{pending_total} now pass (promote these to live)");
    if !promote.is_empty() {
        println!("  ready to promote:\n    {}", promote.join("\n    "));
    }

    assert!(
        oracle_failures.is_empty(),
        "expected values disagree with the Chicken Scheme oracle:\n{}",
        oracle_failures.join("\n")
    );
    assert!(
        live_failures.is_empty(),
        "LIVE conformance regressions:\n{}",
        live_failures.join("\n")
    );
}
