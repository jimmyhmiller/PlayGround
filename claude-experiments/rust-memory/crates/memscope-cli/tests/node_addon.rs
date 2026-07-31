//! Rust-in-a-Node-app capture, as a test.
//!
//! > **A Rust cdylib loaded by `node` records and symbolicates against its own
//! > `.node` image: concrete types, addon source locations, and a live set — with
//! > no `main`, no injection, and an executable (`node`) that has no Rust DWARF
//! > at all.**
//!
//! This is the shape that used to fail silently in three separate places, so all
//! three are asserted here rather than reasoned about:
//!
//! 1. The recording header must name **the module**, not `current_exe()`. Naming
//!    `node` sends the reader to a binary with no Rust debug info, and every
//!    frame comes back `[unknown]` — a recording that looks fine and says
//!    nothing.
//! 2. `MEMSCOPE_RECORD` must expand `{pid}`. A host that spawns workers hands the
//!    same value to every child, and they'd all write one file.
//! 3. `memscope check` must recognize a module: no `fn main()` to edit, and the
//!    injection/signing verdicts don't apply to it.
//!
//! The fixture (`crates/nodeaddon`) is a real N-API addon, loaded through
//! `process.dlopen` exactly as `require('./foo.node')` would.

use std::path::{Path, PathBuf};
use std::process::Command;

use memscope_replay::{read_recording, Timeline};

/// Calls into the addon. Each one keeps a `Box<Widget>` alive forever, so the
/// final live set is a known count — an addon-side leak we must be able to see.
const CALLS: usize = 200;

#[test]
fn node_addon_records_and_symbolicates_against_its_own_image() {
    let node = require_node();
    let root = workspace_root();
    let module = addon_module();
    let work_dir = fresh_dir("memscope-node-addon");

    // `{pid}` is the child's pid, which we don't know: that's the point of the
    // template, and finding exactly one expanded file is what proves it worked.
    let template = work_dir.join("addon-{pid}.mscope");
    let out = Command::new(&node)
        .arg(root.join("crates/nodeaddon/driver.js"))
        .arg(&module)
        .arg(CALLS.to_string())
        .env("MEMSCOPE_RECORD", &template)
        .output()
        .expect("failed to run node");
    assert!(
        out.status.success(),
        "node failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // 1. `{pid}` expanded — one file per process, named for the child.
    let recordings = mscope_files(&work_dir);
    assert_eq!(
        recordings.len(),
        1,
        "expected exactly one expanded recording in {}, got {recordings:?}",
        work_dir.display()
    );
    let recording = &recordings[0];
    let name = recording.file_name().unwrap().to_string_lossy().into_owned();
    assert!(
        !name.contains("{pid}"),
        "MEMSCOPE_RECORD did not expand {{pid}}: wrote {name}"
    );
    let child_pid: u32 = name
        .trim_start_matches("addon-")
        .trim_end_matches(".mscope")
        .parse()
        .unwrap_or_else(|e| panic!("{name} is not addon-<pid>.mscope: {e}"));
    assert_ne!(child_pid, std::process::id(), "{name} should carry node's pid");

    // 2. The recording symbolicates against the MODULE, not the host executable.
    let rec = read_recording(recording.to_str().unwrap()).expect("read recording");
    assert_eq!(rec.pid, child_pid, "header pid should match the expanded name");
    assert!(
        Path::new(&rec.exe).file_name() == module.file_name(),
        "recording should name the module ({}), not the host: {}",
        module.display(),
        rec.exe
    );

    // 3. Types recovered from the addon's own DWARF, through node.
    let labels: Vec<&str> = rec.sites.values().map(|s| s.label.as_str()).collect();
    assert!(
        labels.iter().any(|l| l.contains("Widget")),
        "no addon type recovered; got labels: {labels:?}"
    );

    // 4. Stacks land in the addon's source, not in `[unknown]`.
    let in_addon: Vec<&str> = rec
        .sites
        .values()
        .flat_map(|s| s.frames.iter())
        .filter(|f| f.func.contains("nodeaddon"))
        .map(|f| f.file.as_str())
        .collect();
    assert!(
        !in_addon.is_empty(),
        "no frame resolved into the addon; sites: {:?}",
        rec.sites.values().map(|s| (&s.label, &s.frames)).collect::<Vec<_>>()
    );
    assert!(
        in_addon.iter().any(|f| f.ends_with("nodeaddon/src/lib.rs")),
        "addon frames carry no source file: {in_addon:?}"
    );

    // 5. The live set at the end holds the addon's leak — one boxed Widget per
    //    call, still reachable when node exited.
    let timeline = Timeline::open(recording.to_str().unwrap(), &rec).expect("open timeline");
    let live = timeline.replay(|_, _, _, _| true).expect("replay");
    let boxed_widgets: u64 = live
        .agg_by_site()
        .into_iter()
        .filter(|(site, _)| {
            let l = rec.site_label(*site);
            l.contains("Widget") && !l.starts_with("Vec<")
        })
        .map(|(_, (count, _bytes))| count)
        .sum();
    assert_eq!(
        boxed_widgets, CALLS as u64,
        "expected one leaked Box<Widget> per call in the final live set"
    );

    // 6. Per-call metadata survives the addon path (meta! inside an N-API call).
    let calls_tagged = rec
        .meta
        .values()
        .filter(|kvs| kvs.iter().any(|(k, _)| k == "call"))
        .count();
    assert_eq!(calls_tagged, CALLS, "every call should have its own meta context");
}

/// Several host processes at once — the shape you get the moment an app uses a
/// worker pool, since one `MEMSCOPE_RECORD` value is inherited by all of them.
///
/// Two things have to hold: `{pid}` gives each worker its own file, and the
/// per-module `.dSYM` generation is safe under concurrency. Both failed before:
/// they shared one file, and racing `dsymutil` runs left a half-written bundle
/// that resolved every site to nothing without reporting an error.
///
/// Deliberately runs against its **own copy** of the module, so the dSYM starts
/// cold (that's the contended path) without disturbing the sibling tests.
#[test]
fn concurrent_host_workers_each_get_a_symbolicated_recording() {
    const WORKERS: usize = 4;
    let node = require_node();
    let root = workspace_root();
    let work_dir = fresh_dir("memscope-node-workers");

    let module = work_dir.join("worker.node");
    std::fs::copy(cdylib_path(&root), &module).expect("copy cdylib");

    let children: Vec<_> = (0..WORKERS)
        .map(|_| {
            Command::new(&node)
                .arg(root.join("crates/nodeaddon/driver.js"))
                .arg(&module)
                .arg("60")
                .env("MEMSCOPE_RECORD", work_dir.join("worker-{pid}.mscope"))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .expect("failed to spawn node worker")
        })
        .collect();
    for child in children {
        let out = child.wait_with_output().expect("wait for worker");
        assert!(out.status.success(), "worker failed: {}", String::from_utf8_lossy(&out.stderr));
    }

    let recordings = mscope_files(&work_dir);
    assert_eq!(
        recordings.len(),
        WORKERS,
        "each worker should have written its own recording, got {recordings:?}"
    );
    for rec_path in &recordings {
        let rec = read_recording(rec_path.to_str().unwrap()).expect("read worker recording");
        assert!(
            rec.sites.values().any(|s| s.label.contains("Widget")),
            "{} resolved no addon types (concurrent dSYM generation lost the race)",
            rec_path.display()
        );
    }
}

/// The **live** path from inside a host process: `MEMSCOPE_LIVE=1` brings the
/// agent up in the addon, and `memscope dump` gets a type-resolved heap out of a
/// running `node`.
///
/// This is the one that needed `TypeOracle::for_current_process` to stop meaning
/// `current_exe()`: the oracle is built *inside* the traced process, and inside
/// node that used to mean running `dsymutil` on node itself.
#[test]
fn live_agent_works_inside_a_node_process() {
    let node = require_node();
    let root = workspace_root();
    let module = addon_module();
    let work_dir = fresh_dir("memscope-node-live");
    let sock = work_dir.join("addon.sock");
    let dump = work_dir.join("dump.json");

    let mut child = Command::new(&node)
        .arg(root.join("crates/nodeaddon/driver.js"))
        .arg(&module)
        .args(["20", "hold"])
        .env("MEMSCOPE_LIVE", "1")
        .env("MEMSCOPE_SOCK", &sock)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn node");

    // Poll the condition that actually matters — a dump containing the addon's
    // types — rather than assuming an ordering between the agent binding its
    // socket, the addon doing work, and us asking. The `hold` loop keeps
    // allocating, so this converges as soon as all three have happened.
    let mut last = String::new();
    let got_types = wait_for(std::time::Duration::from_secs(30), || {
        if !sock.exists() {
            return false;
        }
        let out = Command::new(env!("CARGO_BIN_EXE_memscope"))
            .arg("dump")
            .arg("--sock")
            .arg(&sock)
            .arg("--out")
            .arg(&dump)
            .output()
            .expect("failed to run memscope dump");
        if !out.status.success() {
            last = String::from_utf8_lossy(&out.stderr).into_owned();
            return false;
        }
        last = std::fs::read_to_string(&dump).unwrap_or_default();
        last.contains("nodeaddon::Widget")
    });
    let _ = child.kill();
    let _ = child.wait();

    assert!(
        got_types,
        "no live dump from node ever carried the addon's types; last attempt:\n{}",
        &last[..last.len().min(2000)]
    );
}

/// The heap-dump trigger from inside a host process: `MEMSCOPE_HPROF_ON_EXIT`
/// writes a real JVM `.hprof` (MAT-openable) of the addon's heap, with the
/// addon's types as class names.
#[test]
fn hprof_on_exit_works_inside_a_node_process() {
    let node = require_node();
    let root = workspace_root();
    let module = addon_module();
    let work_dir = fresh_dir("memscope-node-hprof");

    let out = Command::new(&node)
        .arg(root.join("crates/nodeaddon/driver.js"))
        .arg(&module)
        .arg("100")
        .env("MEMSCOPE_HPROF_ON_EXIT", "1")
        .env("MEMSCOPE_HPROF_OUT", work_dir.join("addon-{pid}.hprof"))
        .output()
        .expect("failed to run node");
    assert!(out.status.success(), "node failed: {}", String::from_utf8_lossy(&out.stderr));

    // The dump is built in a forked child that outlives the host, so wait for it.
    let dumps = || {
        std::fs::read_dir(&work_dir)
            .expect("read work dir")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "hprof"))
            .collect::<Vec<_>>()
    };
    assert!(
        wait_for(std::time::Duration::from_secs(120), || {
            dumps().first().is_some_and(|p| {
                std::fs::metadata(p).map(|m| m.len() > 1024).unwrap_or(false)
            })
        }),
        "no .hprof appeared in {}: {}",
        work_dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );

    let dump = dumps().remove(0);
    let bytes = std::fs::read(&dump).expect("read hprof");
    assert!(
        bytes.starts_with(b"JAVA PROFILE 1.0.2\0"),
        "{} is not an hprof",
        dump.display()
    );
    // Class names live in the file as UTF8 records: the addon's type must be one.
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("nodeaddon::Widget"),
        "hprof from a node addon carries no addon type names"
    );
}

/// `memscope check` on the module file: it must read as a module, and *not* be
/// judged by the injection/signing verdict a `.node` would otherwise get.
#[test]
fn check_reports_a_dynamic_module() {
    let module = addon_module();

    let out = Command::new(env!("CARGO_BIN_EXE_memscope"))
        .arg("check")
        .arg(&module)
        .output()
        .expect("failed to run memscope check");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "check should exit 0 for a ready module:\n{stdout}");
    assert!(stdout.contains("dynamic module"), "check should name the kind:\n{stdout}");
    assert!(
        !stdout.contains("signing"),
        "signing is an injection verdict and must not be shown for a module:\n{stdout}"
    );
}

/// `memscope check <cdylib project dir>`: the artifact lives in the *workspace*
/// target dir, and the setup that matters is an init hook rather than `main`.
/// Both were invisible before: a built module reported as "none built yet".
#[test]
fn check_finds_a_cdylib_projects_module_and_init_hook() {
    let root = workspace_root();
    addon_module();

    let out = Command::new(env!("CARGO_BIN_EXE_memscope"))
        .arg("check")
        .arg(root.join("crates/nodeaddon"))
        .output()
        .expect("failed to run memscope check");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the fixture is fully set up, check should exit 0:\n{stdout}");
    assert!(stdout.contains("dynamic module"), "check should name the kind:\n{stdout}");
    assert!(
        !stdout.contains("none built yet"),
        "the module IS built (in the workspace target dir):\n{stdout}"
    );
    assert!(
        stdout.contains("memscope::init() present"),
        "check should see the init hook in the source:\n{stdout}"
    );
    assert!(
        stdout.contains("types resolve"),
        "debug info comes from the workspace profile, and must be found:\n{stdout}"
    );
}

/// `memscope run -- node app.js` is the trap: injection would record against
/// node's binary and resolve nothing. It must refuse and point at the module.
#[test]
fn run_refuses_to_inject_a_module_host() {
    let node = require_node();
    let out = Command::new(env!("CARGO_BIN_EXE_memscope"))
        .args(["run", "--on-exit", "--"])
        .arg(&node)
        .arg("-e")
        .arg("0")
        .output()
        .expect("failed to run memscope run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "run should refuse a module host:\n{stderr}");
    assert!(
        stderr.contains("not a Rust program") && stderr.contains(".node"),
        "the refusal should explain why and point at the module:\n{stderr}"
    );
}

// --- fixture plumbing --------------------------------------------------------

fn workspace_root() -> PathBuf {
    // crates/memscope-cli -> crates -> workspace root
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn target_dir(root: &Path) -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(d) => PathBuf::from(d),
        None => root.join("target"),
    }
}

/// node is a hard requirement of this test, not an optional extra: the whole
/// claim is about running under a real host. Fail loudly rather than skip —
/// a test that quietly passes when it didn't run is worse than no test.
fn require_node() -> PathBuf {
    let node = std::env::var("MEMSCOPE_TEST_NODE").unwrap_or_else(|_| "node".to_string());
    let ok = Command::new(&node)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    assert!(
        ok,
        "these tests drive a real Node process and need `{node}` on PATH \
         (or set MEMSCOPE_TEST_NODE=/path/to/node)"
    );
    PathBuf::from(node)
}

/// Build the fixture cdylib and return its path. Cheap to call repeatedly:
/// cargo is a no-op once it's up to date.
fn cdylib_path(root: &Path) -> PathBuf {
    static BUILT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    BUILT
        .get_or_init(|| {
            let status = Command::new(env!("CARGO"))
                .current_dir(root)
                .args(["build", "-p", "nodeaddon", "--release"])
                .status()
                .expect("failed to run cargo build for the addon");
            assert!(status.success(), "cargo build -p nodeaddon failed");

            let release = target_dir(root).join("release");
            let cdylib = if cfg!(target_os = "macos") {
                release.join("libnodeaddon.dylib")
            } else {
                release.join("libnodeaddon.so")
            };
            assert!(cdylib.is_file(), "{} was not built", cdylib.display());
            cdylib
        })
        .clone()
}

/// The `.node` module, built exactly once per test binary.
///
/// Tests in a binary run as concurrent threads, so this MUST NOT be per-test:
/// [`build_addon`] deletes and re-copies the module, and doing that under a
/// sibling test's live `node` process (or its `dsymutil` run) makes both flaky
/// for reasons that have nothing to do with what they assert.
fn addon_module() -> PathBuf {
    static MODULE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| build_addon(&workspace_root())).clone()
}

/// Build the addon and produce the `.node` file node will load.
///
/// The copy is deliberate: it's what napi-rs does, and it's the thing that used
/// to break symbolication (the dSYM has to be found next to the *loaded* file).
/// Stale artifacts are removed first so `dsymutil` runs against this build's
/// monomorphization hashes — a stale dSYM silently un-recovers every type.
fn build_addon(root: &Path) -> PathBuf {
    let cdylib = cdylib_path(root);
    let module = target_dir(root).join("release").join("nodeaddon.node");
    let _ = std::fs::remove_file(&module);
    let _ = std::fs::remove_dir_all(module.with_extension("node.dSYM"));
    std::fs::copy(&cdylib, &module).expect("copy cdylib to .node");
    module
}

fn fresh_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create work dir");
    dir
}

/// Poll `cond` until it holds or `limit` elapses. Waiting on the real condition
/// (a socket bound, a file written) keeps these tests from encoding a guessed
/// sleep, which is where "passes on my machine" comes from.
fn wait_for(limit: std::time::Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < limit {
        if cond() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    cond()
}

fn mscope_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read work dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "mscope"))
        .collect();
    files.sort();
    files
}
