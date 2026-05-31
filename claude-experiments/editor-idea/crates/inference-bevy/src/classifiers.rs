//! Classifier prompts + result types.
//!
//! Each classifier is a (system prompt, user-template, output-shape)
//! triple. Keep them small and concrete — the smaller model behind a
//! cheap endpoint is fine for binary-ish decisions but easily confused
//! by ambiguous instructions.
//!
//! Design principle here: don't ask the model to do work we can do
//! cheaply in Rust. Directory facts (does `.git` exist, which build
//! manifests are present, the top-level entries) are extracted on the
//! Rust side and handed to the model as evidence. The model's job is
//! to weigh that evidence, not to play filename-detective on a bare
//! path string.

use serde::Deserialize;

use crate::llm::{self, LlmConfig, LlmError};

/// Output of [`classify_default_cwd`]. The model decides `good_default
/// = true` when the supplied directory facts add up to a stable
/// project root.
#[derive(Debug, Deserialize, Clone)]
pub struct DefaultCwdClassification {
    pub good_default: bool,
    /// 0.0 to 1.0. Caller should threshold (suggested >= 0.6) before
    /// surfacing the suggestion to the user.
    pub confidence: f32,
    /// One short sentence the user can see if we ever build a UI for
    /// these suggestions. Kept under ~120 chars by the prompt.
    pub reason: String,
}

/// Snapshot of directory state passed to the cwd-default classifier
/// as evidence. Computed cheaply with `std::fs` — no globbing, no
/// recursion beyond one level.
#[derive(Debug, Clone)]
pub struct CwdFacts {
    pub path: String,
    /// True if the path itself contains a `.git` directory (or file —
    /// git worktrees use a file). Doesn't walk upward; that's a
    /// separate signal.
    pub is_git_root: bool,
    /// Build manifests present at the top level, in canonical order.
    pub manifests: Vec<String>,
    /// Top-level entries (file + directory names), sorted, with `/`
    /// suffixed onto directories. Truncated to keep the prompt small.
    pub entries: Vec<String>,
    /// Total entry count *before* truncation — lets the model reason
    /// about whether a path is e.g. mostly empty.
    pub entries_total: usize,
}

const MAX_ENTRIES_IN_PROMPT: usize = 40;
const MAX_ENTRY_NAME_LEN: usize = 48;

/// The set of files we treat as evidence-of-project. Order is the
/// order they're listed in the prompt — most-distinctive first.
const KNOWN_MANIFESTS: &[&str] = &[
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "go.mod",
    "Gemfile",
    "build.zig",
    "CMakeLists.txt",
    "Makefile",
    "deno.json",
    "deno.jsonc",
    "tsconfig.json",
    "pnpm-workspace.yaml",
];

/// Read the directory at `cwd` and pack a [`CwdFacts`]. Returns a
/// minimal facts struct (path-only, everything else empty/false) if
/// the directory doesn't exist or can't be read — the classifier will
/// then quite reasonably say `good_default = false`.
pub fn gather_facts(cwd: &str) -> CwdFacts {
    let path = std::path::Path::new(cwd);
    if !path.is_dir() {
        return CwdFacts {
            path: cwd.to_string(),
            is_git_root: false,
            manifests: Vec::new(),
            entries: Vec::new(),
            entries_total: 0,
        };
    }
    let is_git_root = path.join(".git").exists();
    let mut all_entries: Vec<(String, bool)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(path) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            // Skip dotfiles other than `.git` — they're rarely
            // signal-rich and we'd rather use the slot for visible
            // entries.
            if name.starts_with('.') && name != ".git" {
                continue;
            }
            let is_dir = entry
                .file_type()
                .map(|t| t.is_dir())
                .unwrap_or(false);
            all_entries.push((name, is_dir));
        }
    }
    all_entries.sort_by(|a, b| a.0.cmp(&b.0));

    let manifests: Vec<String> = KNOWN_MANIFESTS
        .iter()
        .filter(|m| all_entries.iter().any(|(n, is_dir)| !is_dir && n == *m))
        .map(|m| (*m).to_string())
        .collect();

    let entries_total = all_entries.len();
    let entries: Vec<String> = all_entries
        .iter()
        .take(MAX_ENTRIES_IN_PROMPT)
        .map(|(name, is_dir)| {
            let mut n: String = name.chars().take(MAX_ENTRY_NAME_LEN).collect();
            if name.len() > n.len() {
                n.push('…');
            }
            if *is_dir {
                n.push('/');
            }
            n
        })
        .collect();

    CwdFacts {
        path: cwd.to_string(),
        is_git_root,
        manifests,
        entries,
        entries_total,
    }
}

const DEFAULT_CWD_SYSTEM: &str = "\
You decide whether a given directory is a sensible default working \
directory to open future terminals in for a given project.

Useful context about this user's setup:
- `~/Documents/Code/PlayGround` is the user's big monorepo-style root \
holding many independent projects side-by-side. A directory *inside* \
PlayGround is usually a real project, not a generic location, even if \
its parents include words like 'playground' or 'experiments'. Do not \
penalize PlayGround paths.
- The `project_name` field is just a human label the user picked; it \
may or may not match the directory name. Do NOT require name-to-path \
matching. Use the directory contents (below) as the real evidence.

Signals that argue FOR `good_default=true`:
- `is_git_root=true`, or any entry in `manifests` (Cargo.toml, \
package.json, pyproject.toml, go.mod, Makefile, CMakeLists.txt, …)
- Top-level entries that look like a project's working files (src/, \
README, tests/, scripts/, lockfiles, language-specific dirs).

Signals that argue AGAINST:
- The directory is the user's `$HOME` itself, or a system/temp dir \
(/tmp, /private/tmp), or `~/Downloads`.
- The directory is empty or contains only unrelated downloads/notes.

Respond with ONLY a JSON object matching this exact shape:
{\"good_default\": true|false, \"confidence\": 0.0..1.0, \"reason\": \"<one short sentence>\"}
No prose outside the JSON. Keep `reason` under 120 characters.";

/// Ask the model whether `cwd` is a reasonable default working
/// directory for the project named `project_name`. Blocking; call
/// from a thread.
///
/// `project_name` is included in the prompt for context only — the
/// classifier is instructed to ignore name-vs-path mismatches and
/// decide on directory contents.
pub fn classify_default_cwd(
    cfg: &LlmConfig,
    project_name: &str,
    cwd: &str,
) -> Result<DefaultCwdClassification, LlmError> {
    let facts = gather_facts(cwd);
    let user = format_user_message(project_name, &facts);
    llm::classify(cfg, DEFAULT_CWD_SYSTEM, &user)
}

fn format_user_message(project_name: &str, facts: &CwdFacts) -> String {
    let mut out = String::new();
    out.push_str(&format!("Project name (label only): {project_name}\n"));
    out.push_str(&format!("Candidate cwd: {}\n", facts.path));
    out.push_str(&format!("is_git_root: {}\n", facts.is_git_root));
    if facts.manifests.is_empty() {
        out.push_str("manifests: []\n");
    } else {
        out.push_str(&format!("manifests: {:?}\n", facts.manifests));
    }
    if facts.entries.is_empty() {
        out.push_str("entries: (none readable)\n");
    } else {
        out.push_str(&format!(
            "entries ({} shown of {} total):\n",
            facts.entries.len(),
            facts.entries_total
        ));
        for e in &facts.entries {
            out.push_str("  ");
            out.push_str(e);
            out.push('\n');
        }
    }
    out
}

// ---------- Command-suggestion classifier ----------

/// Output of [`classify_command_suggestion`]. The model decides whether
/// a command the user just ran is worth parking as a one-click
/// run-button on their canvas.
#[derive(Debug, Deserialize, Clone)]
pub struct CommandSuggestion {
    pub worth_suggesting: bool,
    /// 0.0 to 1.0. Caller thresholds before surfacing.
    pub confidence: f32,
    /// Short imperative label for the run-button, e.g. "Run tests",
    /// "Start dev server". Kept under ~40 chars by the prompt.
    pub title: String,
    /// One short sentence shown under the title in the drawer.
    pub reason: String,
}

const COMMAND_SUGGEST_SYSTEM: &str = "\
You decide whether a shell command the user just ran in a project \
terminal is worth saving as a one-click 'run-button' on their canvas, \
so they can re-run it later without retyping.

Say worth_suggesting=true for repeatable, project-meaningful tasks the \
user is likely to run again:
- build / compile (cargo build, make, npm run build, go build, zig build)
- test (cargo test, pytest, npm test, go test)
- run / dev servers (cargo run, npm run dev, ./script.sh, python main.py)
- lint / format / typecheck / benchmark / deploy
- longer pipelines or commands with meaningful flags the user tuned

Say worth_suggesting=false for one-off or navigational/inspection \
commands that make no sense as a saved button:
- navigation & inspection: cd, ls, pwd, cat, less, head, tail, find, \
grep, which, man, echo, clear, env, history
- interactive editors / pagers / REPLs: vim, nano, less, top, htop, \
python (bare), node (bare)
- version-control inspection: git status, git log, git diff, git branch
- throwaway one-liners, typos, or anything that already failed for a \
reason that won't change (note: a failing test command is still worth \
a button; a typo'd command is not).

Respond with ONLY a JSON object matching this exact shape:
{\"worth_suggesting\": true|false, \"confidence\": 0.0..1.0, \"title\": \"<short imperative label>\", \"reason\": \"<one short sentence>\"}
No prose outside the JSON. Keep `title` under 40 characters and \
`reason` under 120 characters.";

/// Ask the model whether `command` (run in `cwd`, project label
/// `project_name`, finishing with `exit_code`) is worth a saved
/// run-button. Blocking; call from a thread.
pub fn classify_command_suggestion(
    cfg: &LlmConfig,
    project_name: &str,
    command: &str,
    cwd: &str,
    exit_code: i32,
) -> Result<CommandSuggestion, LlmError> {
    let user = format!(
        "Project name (label only): {project_name}\n\
         cwd: {cwd}\n\
         exit_code: {exit_code}\n\
         command:\n  {command}\n"
    );
    llm::classify(cfg, COMMAND_SUGGEST_SYSTEM, &user)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_facts_on_missing_dir_is_safe() {
        let f = gather_facts("/no/such/dir/anywhere/x");
        assert_eq!(f.is_git_root, false);
        assert!(f.manifests.is_empty());
        assert!(f.entries.is_empty());
        assert_eq!(f.entries_total, 0);
    }

    #[test]
    fn gather_facts_finds_cargo_and_git() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join(".git")).unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        std::fs::create_dir(tmp.path().join("src")).unwrap();
        let f = gather_facts(tmp.path().to_str().unwrap());
        assert!(f.is_git_root);
        assert_eq!(f.manifests, vec!["Cargo.toml"]);
        // .git is special-cased and included; "src/" should have the
        // trailing slash marker.
        assert!(f.entries.iter().any(|e| e == "src/"));
        assert!(f.entries.iter().any(|e| e == ".git/"));
    }
}
