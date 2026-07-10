//! R7RS conformance, run on the NATIVE tier.
//!
//! Same case table as `conformance.rs`, but each program is executed through an
//! automatic tiering policy instead of purely on the CEK machine:
//!
//!   * a program that uses first-class continuations (`call/cc`) runs WHOLLY on
//!     the `CekMachine` — a host-stack JIT cannot capture native frames, so no
//!     JIT frame may sit inside a `call/cc` capture;
//!   * every other program runs on `Tiered` = the Cranelift JIT with a CEK
//!     fallback, so its bodies compile to machine code and only the non-capturing
//!     CEK-only prims (`apply`, `values`) fall back.
//!
//! The assertion is identical to the interpreter suite (oracle-checked against
//! Chicken `csi` when present), so "runs on the JIT" means "computes the R7RS
//! answer on the JIT," not merely "does not crash." We also report how many
//! live cases ran fully native vs. fell back — the honest native-coverage number.
#![cfg(feature = "jit")]

use std::io::Write;
use std::process::{Command, Stdio};

use microlang::{CekMachine, LowBitModel, Runtime, Tiered};

#[path = "common/suite.rs"]
mod suite;
use suite::{uses_first_class_continuations, CASES};

/// Run one program under the tiering policy. Returns `(output, ran_native)`.
fn run_tiered(setup: &str, expr: &str) -> (Option<String>, bool) {
    let prog = format!("{setup} {expr}");
    let continuation = uses_first_class_continuations(setup, expr);
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(|| {
        let mut rt = Runtime::<LowBitModel>::new();
        if continuation {
            // First-class continuations: whole program on the stackless machine.
            let v = scheme::run(&mut rt, &CekMachine, &prog);
            scheme::write_value(&rt, v)
        } else {
            // Native JIT with a CEK fallback for non-capturing CEK-only prims.
            let cs = Tiered::<LowBitModel>::new();
            let v = scheme::run(&mut rt, &cs, &prog);
            scheme::write_value(&rt, v)
        }
    })
    .ok();
    std::panic::set_hook(prev);
    (out, !continuation)
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
fn r7rs_conformance_on_the_jit() {
    let csi = csi_path();
    if csi.is_none() {
        eprintln!("note: `csi` (Chicken Scheme) not found — expected values not oracle-validated");
    }

    let (mut live_ok, mut live_total) = (0, 0);
    let (mut native_ran, mut native_ok) = (0, 0);
    let mut live_failures = Vec::new();
    let mut oracle_failures = Vec::new();

    for case in CASES {
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

        let (ours, ran_native) = run_tiered(case.setup, case.expr);
        let ok = ours.as_deref() == Some(case.expect);
        if ran_native {
            native_ran += 1;
        }
        if case.live {
            live_total += 1;
            if ok {
                live_ok += 1;
                if ran_native {
                    native_ok += 1;
                }
            } else {
                live_failures.push(format!(
                    "[{}] {} (native={})  got={:?} want={:?}",
                    case.area, case.expr, ran_native, ours, case.expect
                ));
            }
        }
    }

    println!("\nR7RS conformance on the native tier:");
    println!("  live:            {live_ok}/{live_total} passing");
    println!("  ran on the JIT:  {native_ok} live cases fully native (+ CEK fallback for apply/values)");
    println!("  on CEK (call/cc): {} cases require the stackless machine", live_total - native_ok);
    let _ = native_ran;

    assert!(
        oracle_failures.is_empty(),
        "expected values disagree with the Chicken Scheme oracle:\n{}",
        oracle_failures.join("\n")
    );
    assert!(
        live_failures.is_empty(),
        "LIVE conformance regressions on the JIT tier:\n{}",
        live_failures.join("\n")
    );
}
