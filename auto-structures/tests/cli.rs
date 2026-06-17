//! End-to-end tests: drive the compiled `autostruct` binary against the
//! example programs and assert both the inferred structures and the output.

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_autostruct")
}

fn run(args: &[&str]) -> String {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("failed to run autostruct");
    assert!(
        out.status.success(),
        "command {:?} failed:\n{}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn ex(name: &str) -> String {
    format!("{}/examples/{}.shape", env!("CARGO_MANIFEST_DIR"), name)
}

/// Just the lines printed by the program (after the report separator).
fn output_only(text: &str) -> String {
    match text.find("output:\n") {
        Some(i) => text[i + "output:\n".len()..].to_string(),
        None => text.to_string(),
    }
}

#[test]
fn infers_ordered_map_for_word_count() {
    let report = run(&["analyze", &ex("word_count")]);
    assert!(report.contains("collection `counts`"));
    assert!(report.contains("BTreeMap"), "expected ordered map, got:\n{}", report);
}

#[test]
fn word_count_output_is_sorted_and_correct() {
    let out = output_only(&run(&["run", &ex("word_count")]));
    let lines: Vec<&str> = out.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines, vec!["cat 2", "mat 1", "on 1", "ran 1", "sat 1", "the 3"]);
}

#[test]
fn infers_hashset_and_vec_for_dedup() {
    let report = run(&["analyze", &ex("dedup")]);
    assert!(report.contains("HashSet"), "report:\n{}", report);
    assert!(report.contains("Vec"), "report:\n{}", report);
}

#[test]
fn dedup_is_stable() {
    let out = output_only(&run(&["run", &ex("dedup")]));
    let lines: Vec<&str> = out.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines,
        vec!["unique count: 7", "3", "1", "4", "5", "9", "2", "6"]
    );
}

#[test]
fn infers_queue_for_bfs_and_visits_in_order() {
    let report = run(&["analyze", &ex("bfs")]);
    assert!(report.contains("VecDeque"), "report:\n{}", report);
    assert!(report.contains("HashMap"), "report:\n{}", report);

    let out = output_only(&run(&["run", &ex("bfs")]));
    let nums: Vec<&str> = out
        .lines()
        .filter(|l| l.chars().all(|c| c.is_ascii_digit()) && !l.is_empty())
        .collect();
    assert_eq!(nums, vec!["0", "1", "2", "3", "4", "5", "6"]);
}

#[test]
fn ordered_set_upgrades_to_btreeset() {
    let report = run(&["analyze", &ex("ordered_set")]);
    assert!(report.contains("BTreeSet"), "report:\n{}", report);
    let out = output_only(&run(&["run", &ex("ordered_set")]));
    assert!(out.contains("min: 3"));
    assert!(out.contains("max: 88"));
}

#[test]
fn naive_and_specialized_agree() {
    // Same program, two backends, identical observable behaviour.
    for name in ["word_count", "dedup", "bfs", "ordered_set", "stack"] {
        let spec = output_only(&run(&["run", &ex(name)]));
        let naive_full = run(&["run", &ex(name), "--naive"]);
        // --naive suppresses the report, so its stdout is pure output.
        assert_eq!(
            spec.trim(),
            naive_full.trim(),
            "backends disagree for {}",
            name
        );
    }
}
