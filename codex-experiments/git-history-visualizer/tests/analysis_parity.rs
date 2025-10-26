use git_history_visualizer::analysis::{AnalyzeConfig, analyze};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;

fn git(dir: &Path, args: &[&str], env: &[(&str, &str)]) {
    let status = Command::new("git")
        .args(args)
        .current_dir(dir)
        .envs(env.iter().cloned())
        .status()
        .expect("failed to run git command");
    assert!(
        status.success(),
        "git {:?} failed with status {:?}",
        args,
        status.code()
    );
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents.as_bytes()).expect("failed to write file");
}

fn append_file(path: &Path, contents: &str) {
    let mut existing = fs::read_to_string(path).expect("failed to read file");
    existing.push_str(contents);
    fs::write(path, existing).expect("failed to append file");
}

fn replace_line(path: &Path, old: &str, new: &str) {
    let mut existing = fs::read_to_string(path).expect("failed to read file");
    existing = existing.replace(old, new);
    fs::write(path, existing).expect("failed to rewrite file");
}

#[test]
fn analyzer_matches_known_fixture() {
    let repo_dir = TempDir::new().expect("tempdir");
    let repo_path = repo_dir.path();

    git(repo_path, &["init"], &[]);
    git(repo_path, &["config", "user.name", "tester"], &[]);
    git(
        repo_path,
        &["config", "user.email", "tester@example.com"],
        &[],
    );

    let src_file = repo_path.join("src.rs");
    write_file(&src_file, "fn main() {}\n");
    git(repo_path, &["add", "src.rs"], &[]);
    git(
        repo_path,
        &["commit", "-m", "initial"],
        &[
            ("GIT_AUTHOR_DATE", "2023-01-01T05:00:00 +0000"),
            ("GIT_COMMITTER_DATE", "2023-01-01T05:00:00 +0000"),
        ],
    );

    append_file(&src_file, "fn add(a: i32, b: i32) -> i32 { a + b }\n");
    git(repo_path, &["add", "src.rs"], &[]);
    git(
        repo_path,
        &["commit", "-m", "add function"],
        &[
            ("GIT_AUTHOR_DATE", "2023-01-02T00:00:00 +0000"),
            ("GIT_COMMITTER_DATE", "2023-01-02T00:00:00 +0000"),
        ],
    );

    let c_file = repo_path.join("c.c");
    write_file(&c_file, "#include <stdio.h>\nint main(){return 0;}\n");
    git(repo_path, &["add", "c.c"], &[]);
    git(
        repo_path,
        &["commit", "-m", "add c file"],
        &[
            ("GIT_AUTHOR_DATE", "2023-01-03T00:00:00 +0000"),
            ("GIT_COMMITTER_DATE", "2023-01-03T00:00:00 +0000"),
        ],
    );

    replace_line(&src_file, "fn main()", "fn main_updated()");
    git(repo_path, &["add", "src.rs"], &[]);
    git(
        repo_path,
        &["commit", "-m", "rename main fn"],
        &[
            ("GIT_AUTHOR_DATE", "2023-01-04T00:00:00 +0000"),
            ("GIT_COMMITTER_DATE", "2023-01-04T00:00:00 +0000"),
        ],
    );

    git(repo_path, &["rm", "c.c"], &[]);
    git(
        repo_path,
        &["commit", "-m", "remove c file"],
        &[
            ("GIT_AUTHOR_DATE", "2025-10-26T00:13:52 +0000"),
            ("GIT_COMMITTER_DATE", "2025-10-26T00:13:52 +0000"),
        ],
    );

    let out_dir = TempDir::new().expect("outdir");
    let config = AnalyzeConfig {
        repo: repo_path.to_path_buf(),
        cohort_format: "%Y".to_string(),
        interval_secs: 7 * 24 * 60 * 60,
        ignore_patterns: Vec::new(),
        only_patterns: Vec::new(),
        outdir: out_dir.path().to_path_buf(),
        branch: "main".to_string(),
        all_filetypes: false,
        ignore_whitespace: false,
        quiet: true,
        jobs: 1,
        opt: false,
    };

    analyze(config).expect("analysis should succeed");

    let expected: HashMap<&str, Value> = [
        (
            "cohorts.json",
            serde_json::json!({
                "y": [[4, 2], [0, 0]],
                "ts": ["2023-01-04T00:00:00", "2025-10-26T00:13:52"],
                "labels": ["Code added in 2023", "Code added in 2025"],
            }),
        ),
        (
            "exts.json",
            serde_json::json!({
                "y": [[2, 0], [2, 2]],
                "ts": ["2023-01-04T00:00:00", "2025-10-26T00:13:52"],
                "labels": [".c", ".rs"],
            }),
        ),
        (
            "authors.json",
            serde_json::json!({
                "y": [[4, 2]],
                "ts": ["2023-01-04T00:00:00", "2025-10-26T00:13:52"],
                "labels": ["tester"],
            }),
        ),
        (
            "dirs.json",
            serde_json::json!({
                "y": [[4, 2]],
                "ts": ["2023-01-04T00:00:00", "2025-10-26T00:13:52"],
                "labels": ["/"],
            }),
        ),
        (
            "domains.json",
            serde_json::json!({
                "y": [[4, 2]],
                "ts": ["2023-01-04T00:00:00", "2025-10-26T00:13:52"],
                "labels": ["example.com"],
            }),
        ),
    ]
    .into_iter()
    .collect();

    let out_path = out_dir.path();
    for (file, expected_value) in expected {
        let actual = serde_json::from_str::<Value>(
            &fs::read_to_string(PathBuf::from(out_path).join(file)).expect("read output"),
        )
        .expect("parse json");
        assert_eq!(expected_value, actual, "mismatch in output file {file}",);
    }

    let survival_path = out_path.join("survival.json");
    let survival: Value =
        serde_json::from_str(&fs::read_to_string(&survival_path).expect("read survival"))
            .expect("parse survival");
    let expected_survival = serde_json::json!({
        "5daff6f7e6f35b2fb5e167f2da6714685ae0fe1e": [[1672790400, 2], [1761437632, 0]],
        "4d3645a0e202ee87f8650f22524ad44a435f9ed0": [[1672790400, 1], [1761437632, 1]],
        "dc22b55b431b19f5a6808fc36adf0c7b7c3c78a7": [[1672790400, 1], [1761437632, 1]],
    });
    assert_eq!(expected_survival, survival, "survival output mismatch");
}
