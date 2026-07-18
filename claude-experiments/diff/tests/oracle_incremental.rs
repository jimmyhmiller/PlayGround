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

/// Incremental EMIT parity: after a leaf edit, the incrementally-emitted output
/// tree (the re-rendered chunk plus every cache-reused chunk) is byte-for-byte
/// identical to a clean from-scratch build of the same final filesystem state,
/// and exactly ONE chunk is re-rendered while the others are reused verbatim.
#[test]
fn incremental_emit_reuses_every_unchanged_chunk_and_matches_a_clean_build() {
    let workspace = tempdir().unwrap();
    let root = workspace.path();
    // entry + a form the main chunk; b + bleaf form a dynamic-import chunk. The
    // leaf `bleaf` lives only in the dynamic chunk, so editing it must re-render
    // that chunk alone and leave the main chunk untouched.
    write(root, "entry.js", "import { a } from \"./a.js\";\nimport(\"./b.js\").then((m) => console.log(a, m.b));\n");
    write(root, "a.js", "export const a = 10;\n");
    write(root, "b.js", "import { l } from \"./bleaf.js\";\nexport const b = l + 1;\n");
    write(root, "bleaf.js", "export const l = 100;\n");

    let entry = root.join("entry.js");
    let bleaf = root.join("bleaf.js");
    let incremental_dir = root.join("out-incremental");
    let fresh_dir = root.join("out-fresh");

    // Initial incremental build + emit: every chunk renders from scratch.
    let (mut incremental, _) = Bundler::discover(&entry).unwrap();
    let mut reachability = incremental.direct_reachability();
    let mut reachable = reachability.reachable_modules();
    let initial = incremental
        .emit(&reachable, &incremental_dir.join("bundle.js"))
        .unwrap();
    assert!(
        initial.rendered_chunks >= 2,
        "the initial build must render every chunk (got {})",
        initial.rendered_chunks
    );
    let main_before = fs::read(incremental_dir.join("bundle.js")).unwrap();

    // Edit the leaf, re-transform exactly it, then re-emit incrementally.
    fs::write(&bleaf, "export const l = 200;\n").unwrap();
    let update = incremental.rebuild_path(&bleaf).unwrap();
    assert_eq!(update.transformed_modules, 1, "a leaf edit re-transforms one module");
    apply_update(&mut reachability, &mut reachable, &update);
    let reemit = incremental
        .emit(&reachable, &incremental_dir.join("bundle.js"))
        .unwrap();
    assert_eq!(
        reemit.rendered_chunks, 1,
        "a leaf edit must re-render exactly one chunk, reusing the rest"
    );

    // The main chunk (which does not contain the leaf) is byte-identical across
    // the edit: it was served from the render cache, not re-rendered.
    let main_after = fs::read(incremental_dir.join("bundle.js")).unwrap();
    assert_eq!(main_before, main_after, "the unchanged main chunk must be reused verbatim");

    // A clean from-scratch build of the final filesystem state.
    let (fresh, _) = Bundler::discover(&entry).unwrap();
    let fresh_reachable = fresh.reachable_modules_direct();
    fresh.emit(&fresh_reachable, &fresh_dir.join("bundle.js")).unwrap();

    // The whole incrementally-emitted tree equals the clean build's, file for
    // file, byte for byte (the changed chunk AND every reused chunk).
    assert_eq!(reachable, fresh_reachable);
    let incremental_tree = read_tree(&incremental_dir);
    let fresh_tree = read_tree(&fresh_dir);
    assert!(
        incremental_tree.len() >= 2,
        "expected a multi-chunk output, got {} files",
        incremental_tree.len()
    );
    assert_eq!(
        incremental_tree, fresh_tree,
        "incremental emit must be byte-identical to a clean build of the final state"
    );
}

fn write(root: &Path, name: &str, contents: &str) {
    fs::write(root.join(name), contents).unwrap();
}

/// Reads every file under `dir` into a map of dir-relative path -> bytes.
fn read_tree(dir: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
    let mut tree = std::collections::BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in fs::read_dir(&current).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let relative = path.strip_prefix(dir).unwrap().to_path_buf();
                tree.insert(relative, fs::read(&path).unwrap());
            }
        }
    }
    tree
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

/// Measurement (not a gate): on a many-chunk graph, an incremental re-emit after
/// a leaf edit re-renders exactly one chunk and is dramatically faster than the
/// initial whole-bundle render, with the output byte-identical. Run with:
/// `cargo test --release --test oracle_incremental -- --ignored --nocapture`.
#[test]
#[ignore]
fn measure_incremental_emit_speedup() {
    use std::time::Instant;
    let workspace = tempdir().unwrap();
    let root = workspace.path();
    // 200 dynamically-imported chunks, each a small leaf, off one entry.
    let chunks = 200usize;
    let mut entry_src = String::new();
    for i in 0..chunks {
        write(root, &format!("leaf{i}.js"), &format!("export const v = {i};\nexport const w = v + 1;\n"));
        entry_src.push_str(&format!("import(\"./leaf{i}.js\").then((m) => console.log(m.v));\n"));
    }
    write(root, "entry.js", &entry_src);
    let entry = root.join("entry.js");

    let (mut bundler, _) = Bundler::discover(&entry).unwrap();
    let mut reachability = bundler.direct_reachability();
    let mut reachable = reachability.reachable_modules();

    let out = root.join("out/bundle.js");
    let t0 = Instant::now();
    let cold = bundler.emit(&reachable, &out).unwrap();
    let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Edit one leaf and re-emit with a warm cache.
    fs::write(root.join("leaf137.js"), "export const v = 9999;\nexport const w = v + 1;\n").unwrap();
    let update = bundler.rebuild_path(&root.join("leaf137.js")).unwrap();
    apply_update(&mut reachability, &mut reachable, &update);
    let t1 = Instant::now();
    let warm = bundler.emit(&reachable, &out).unwrap();
    let warm_ms = t1.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "chunks={chunks} cold: rendered {} chunks in {cold_ms:.2}ms | warm(leaf edit): rendered {} chunk(s) in {warm_ms:.2}ms | speedup {:.1}x",
        cold.rendered_chunks, warm.rendered_chunks, cold_ms / warm_ms
    );
    assert_eq!(warm.rendered_chunks, 1, "leaf edit re-renders exactly one chunk");
    assert!(cold.rendered_chunks >= chunks, "cold build renders every chunk");
}
