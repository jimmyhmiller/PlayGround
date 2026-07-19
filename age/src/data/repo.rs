//! A cheap, bounded scan of a project's codebase on disk — language histogram,
//! file count, estimated LOC, git commits, and whether it has tests/README. This
//! is what gives each city a *biome* and codebase achievements.
//!
//! The walk skips heavy/generated directories, caps how many files it visits, and
//! estimates LOC from code-file bytes (so it never reads every file). Good enough
//! for thresholds; cached per path by the caller since a repo's character is stable.

use super::CodebaseInfo;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "build", "dist", ".next", "vendor", "Pods",
    "DerivedData", ".venv", "venv", "__pycache__", ".cargo", ".gradle", ".idea",
    ".vs", "bin", "obj", "out", ".terraform", "coverage", ".mypy_cache",
    ".pytest_cache", "Carthage", ".build", ".dart_tool", "elm-stuff", "_build",
];

/// Extensions we treat as "code" (for the language histogram / biome).
const CODE_EXTS: &[&str] = &[
    "rs", "swift", "m", "mm", "clj", "cljs", "cljc", "edn", "lisp", "el", "bg",
    "scm", "rkt", "js", "jsx", "ts", "tsx", "mjs", "cjs", "py", "c", "cc", "cpp",
    "cxx", "h", "hpp", "hh", "go", "rb", "java", "kt", "kts", "scala", "hs", "ml",
    "mli", "php", "cs", "sh", "bash", "zsh", "lua", "r", "jl", "ex", "exs", "erl",
    "sql", "vue", "svelte", "dart", "zig", "nim", "v",
];

const MAX_FILES: u32 = 15_000;
const AVG_BYTES_PER_LINE: u64 = 32;

pub fn scan(root: &Path) -> CodebaseInfo {
    let mut ci = CodebaseInfo::default();
    let mut ext_counts: HashMap<String, u32> = HashMap::new();
    let mut code_bytes: u64 = 0;
    let mut stack = vec![root.to_path_buf()];
    let mut visited: u32 = 0;

    while let Some(dir) = stack.pop() {
        if visited >= MAX_FILES {
            break;
        }
        let rd = match fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for ent in rd.flatten() {
            let name = ent.file_name();
            let name = name.to_string_lossy();
            let is_dir = ent.file_type().map(|f| f.is_dir()).unwrap_or(false);
            if is_dir {
                if name.starts_with('.') || SKIP_DIRS.contains(&name.as_ref()) {
                    continue;
                }
                stack.push(ent.path());
                continue;
            }
            visited += 1;
            ci.files += 1;
            let size = ent.metadata().map(|m| m.len()).unwrap_or(0);
            ci.bytes += size;

            let lower = name.to_ascii_lowercase();
            if lower == "readme" || lower.starts_with("readme.") {
                ci.has_readme = true;
            }
            if lower.contains("test") || lower.contains("spec") || lower.contains("_test.") {
                ci.has_tests = true;
            }
            if let Some(ext) = Path::new(name.as_ref()).extension().and_then(|e| e.to_str()) {
                let ext = ext.to_ascii_lowercase();
                if CODE_EXTS.contains(&ext.as_str()) {
                    *ext_counts.entry(ext).or_insert(0) += 1;
                    code_bytes += size;
                }
            }
        }
    }

    for t in ["tests", "test", "spec", "__tests__", "Tests"] {
        if root.join(t).is_dir() {
            ci.has_tests = true;
        }
    }

    let mut langs: Vec<(String, u32)> = ext_counts.into_iter().collect();
    langs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    langs.truncate(8);
    ci.languages = langs;
    ci.loc = code_bytes / AVG_BYTES_PER_LINE;
    ci.commits = git_commits(root);
    ci
}

/// `git rev-list --count HEAD`, or 0 if not a repo / git unavailable.
fn git_commits(root: &Path) -> u32 {
    if !root.join(".git").exists() {
        return 0;
    }
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-list", "--count", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(0)
        }
        _ => 0,
    }
}
