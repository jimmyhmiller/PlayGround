use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use diffpack::bundler::{BuildUpdate, Bundler, DirectReachability};
use tempfile::tempdir;

#[test]
fn incremental_bundle_matches_a_clean_rebuild_after_structural_edits() {
    if Command::new("node").arg("--version").output().is_err() {
        return;
    }

    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("oracle/fixtures/incremental-edit");
    let workspace = tempdir().unwrap();
    copy_directory(&fixture.join("initial"), workspace.path());

    let entry = workspace.path().join("entry.js");
    let value = workspace.path().join("value.js");
    let incremental_output = workspace.path().join("incremental.cjs");
    let fresh_output = workspace.path().join("fresh.cjs");

    let (mut incremental, _) = Bundler::discover(&entry).unwrap();
    let mut reachability = incremental.direct_reachability();
    let mut reachable = reachability.reachable_modules();

    fs::copy(fixture.join("edited/entry.js"), &entry).unwrap();
    let entry_update = incremental.rebuild_path(&entry).unwrap();
    apply_update(&mut reachability, &mut reachable, &entry_update);
    assert_eq!(reachable.len(), 2, "the detached cycle must be retracted");

    fs::copy(fixture.join("edited/value.js"), &value).unwrap();
    let value_update = incremental.rebuild_path(&value).unwrap();
    apply_update(&mut reachability, &mut reachable, &value_update);
    incremental.emit(&reachable, &incremental_output).unwrap();

    let (fresh, _) = Bundler::discover(&entry).unwrap();
    let fresh_reachable = fresh.reachable_modules_direct();
    fresh.emit(&fresh_reachable, &fresh_output).unwrap();

    assert_eq!(reachable, fresh_reachable);
    assert_eq!(
        fs::read(&incremental_output).unwrap(),
        fs::read(&fresh_output).unwrap()
    );
    assert_eq!(run_node(&incremental_output), "value:2\n");
    assert_eq!(run_node(&fresh_output), "value:2\n");
}

fn apply_update(
    reachability: &mut DirectReachability,
    reachable: &mut BTreeSet<String>,
    update: &BuildUpdate,
) {
    let result = reachability.apply(&update.delta);
    for removed in result.removed {
        reachable.remove(&removed);
    }
    reachable.extend(result.added);
}

fn copy_directory(source: &Path, destination: &Path) {
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
    }
}

fn run_node(path: &Path) -> String {
    let output = Command::new("node").arg(path).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}
