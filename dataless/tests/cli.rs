//! End-to-end tests asserting that the implementation meets the paper's claims.

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_dataless")
}

fn ex(name: &str) -> String {
    format!("{}/examples/{}", env!("CARGO_MANIFEST_DIR"), name)
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

fn run(args: &[&str]) -> Run {
    let out = Command::new(bin()).args(args).output().expect("spawn dataless");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        ok: out.status.success(),
    }
}

/// The central claim: one program, several representations, identical output.
#[test]
fn shapes_identical_across_representations() {
    let r = run(&[
        "compare",
        &ex("shapes.dl"),
        "--decl",
        &ex("shapes_array.decl"),
        "--decl",
        &ex("shapes_list.decl"),
        "--decl",
        &ex("shapes_double.decl"),
    ]);
    assert!(r.ok, "compare failed:\n{}", r.stderr);
    assert!(
        r.stdout.contains("every representation produced identical output"),
        "stdout:\n{}",
        r.stdout
    );
    assert!(r.stdout.contains("total area: 9455"));
    // implied qualification produced the bare-field lines
    assert!(r.stdout.contains("side 30 area 900"));
}

/// Same program text, byte-identical output regardless of representation.
#[test]
fn output_bytes_independent_of_representation() {
    let a = run(&["run", &ex("shapes.dl"), "--decl", &ex("shapes_array.decl")]);
    let b = run(&["run", &ex("shapes.dl"), "--decl", &ex("shapes_list.decl")]);
    assert!(a.ok && b.ok);
    assert_eq!(a.stdout, b.stdout, "representation changed the output");
}

/// The representation must actually change the *cost*, or independence is empty.
#[test]
fn array_cheaper_than_list() {
    let steps = |decl: &str| -> u64 {
        let r = run(&["run", &ex("shapes.dl"), "--decl", &ex(decl), "--trace"]);
        r.stderr
            .lines()
            .find_map(|l| l.trim().strip_prefix("total positional/shift steps: "))
            .and_then(|n| n.trim().parse().ok())
            .unwrap_or_else(|| panic!("no step count in trace:\n{}", r.stderr))
    };
    let array = steps("shapes_array.decl");
    let list = steps("shapes_list.decl");
    assert!(
        array < list,
        "expected ARRAY ({}) cheaper than LIST ({})",
        array,
        list
    );
}

/// A data reference and a function reference are interchangeable by changing
/// only the declaration.
#[test]
fn stored_field_and_computed_function_agree() {
    let r = run(&[
        "compare",
        &ex("circle.dl"),
        "--decl",
        &ex("circle_stored.decl"),
        "--decl",
        &ex("circle_computed.decl"),
    ]);
    assert!(r.ok, "compare failed:\n{}", r.stderr);
    assert!(r.stdout.contains("every representation produced identical output"));
    assert!(r.stdout.contains("total area x100: 171"));
}

/// Generators, search expressions, STATE monitors, insert and delete.
#[test]
fn facilities_behave() {
    let r = run(&["run", &ex("accounts.dl"), "--decl", &ex("accounts.decl")]);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    let o = &r.stdout;
    // search expression
    assert!(o.contains("at least one account has >= 300"), "{}", o);
    // generator pulls advance independently
    assert!(o.contains("first wealthy: acct-2 = 200"), "{}", o);
    assert!(o.contains("next wealthy:  acct-3 = 300"), "{}", o);
    // STATE monitor fires only on false->true transitions (twice, not 3x)
    assert_eq!(o.matches("ALERT: watched account overdrawn").count(), 2, "{}", o);
    assert!(o.contains("overdrawn at -10"));
    assert!(o.contains("overdrawn at -1"));
    // structural ops
    assert!(o.contains("count after insert: 5"));
    assert!(o.contains("count after delete: 4"));
    assert!(o.contains("first account now: acct-2"));
}

/// The facilities program is itself representation-independent (incl. mutation).
#[test]
fn accounts_identical_across_representations() {
    let r = run(&[
        "compare",
        &ex("accounts.dl"),
        "--decl",
        &ex("accounts_array.decl"),
        "--decl",
        &ex("accounts_list.decl"),
        "--decl",
        &ex("accounts.decl"),
    ]);
    assert!(r.ok, "compare failed:\n{}", r.stderr);
    assert!(r.stdout.contains("every representation produced identical output"));
}
