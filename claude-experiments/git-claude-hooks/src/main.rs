use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::{exit, Command, Stdio};

use serde_json::{json, Value};

struct Config {
    block_stash: bool,
    block_destructive: bool,
    protected_branches: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            block_stash: true,
            block_destructive: true,
            protected_branches: vec!["main".into(), "master".into()],
        }
    }
}

fn load_config() -> Config {
    let mut cfg = Config::default();
    let base = env::var("CLAUDE_PROJECT_DIR").unwrap_or_else(|_| ".".into());
    let path = PathBuf::from(base).join(".claude/hooks/git_guard.json");
    let Ok(text) = fs::read_to_string(&path) else {
        return cfg;
    };
    let Ok(v): Result<Value, _> = serde_json::from_str(&text) else {
        return cfg;
    };
    if let Some(b) = v.get("block_stash").and_then(Value::as_bool) {
        cfg.block_stash = b;
    }
    if let Some(b) = v.get("block_destructive").and_then(Value::as_bool) {
        cfg.block_destructive = b;
    }
    if let Some(arr) = v.get("protected_branches").and_then(Value::as_array) {
        cfg.protected_branches = arr
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
    }
    cfg
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("install") => {
            let target = args.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
            match cmd_install(&target) {
                Ok(()) => exit(0),
                Err(e) => {
                    eprintln!("git-guard install failed: {}", e);
                    exit(1);
                }
            }
        }
        Some("uninstall") => {
            let target = args.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
            match cmd_uninstall(&target) {
                Ok(()) => exit(0),
                Err(e) => {
                    eprintln!("git-guard uninstall failed: {}", e);
                    exit(1);
                }
            }
        }
        Some("help") | Some("--help") | Some("-h") => {
            print_help();
            exit(0);
        }
        Some("version") | Some("--version") | Some("-V") => {
            println!("git-guard {}", env!("CARGO_PKG_VERSION"));
            exit(0);
        }
        Some(other) => {
            eprintln!("git-guard: unknown subcommand '{}'\n", other);
            print_help();
            exit(2);
        }
        None => run_hook(),
    }
}

fn print_help() {
    println!(
        "git-guard {ver} — Claude Code PreToolUse hook for destructive git operations.

USAGE:
    git-guard                  Run as a Claude Code hook (reads JSON on stdin).
    git-guard install [DIR]    Wire git-guard into DIR/.claude/settings.json
                               and seed DIR/.claude/hooks/git_guard.json.
                               DIR defaults to the current directory.
    git-guard uninstall [DIR]  Remove the git-guard hook entry from
                               DIR/.claude/settings.json (config is kept).
    git-guard version          Print version.
    git-guard help             Show this help.",
        ver = env!("CARGO_PKG_VERSION")
    );
}

fn run_hook() {
    let mut input = String::new();
    if io::stdin().read_to_string(&mut input).is_err() {
        exit(0);
    }
    let v: Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(_) => exit(0),
    };
    if v.get("tool_name").and_then(Value::as_str) != Some("Bash") {
        exit(0);
    }
    let command = match v
        .pointer("/tool_input/command")
        .and_then(Value::as_str)
    {
        Some(c) => c,
        None => exit(0),
    };

    let cfg = load_config();

    if let Some(reason) = check_command(command, &cfg) {
        let out = json!({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": reason,
            }
        });
        println!("{}", out);
    }
    exit(0);
}

const DEFAULT_CONFIG: &str = r#"{
  "block_stash": true,
  "block_destructive": true,
  "protected_branches": ["main", "master"]
}
"#;

fn cmd_install(project_dir: &PathBuf) -> Result<(), String> {
    let project_dir = project_dir
        .canonicalize()
        .map_err(|e| format!("cannot resolve {}: {}", project_dir.display(), e))?;
    if !project_dir.is_dir() {
        return Err(format!("{} is not a directory", project_dir.display()));
    }
    let claude_dir = project_dir.join(".claude");
    let hooks_dir = claude_dir.join("hooks");
    let config_path = hooks_dir.join("git_guard.json");
    let settings_path = claude_dir.join("settings.json");

    fs::create_dir_all(&hooks_dir).map_err(|e| format!("creating {}: {}", hooks_dir.display(), e))?;

    if !config_path.exists() {
        fs::write(&config_path, DEFAULT_CONFIG)
            .map_err(|e| format!("writing {}: {}", config_path.display(), e))?;
        println!("seeded {}", config_path.display());
    } else {
        println!("kept existing {}", config_path.display());
    }

    let mut settings: Value = if settings_path.exists() {
        let text = fs::read_to_string(&settings_path)
            .map_err(|e| format!("reading {}: {}", settings_path.display(), e))?;
        if text.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&text)
                .map_err(|e| format!("parsing {}: {}", settings_path.display(), e))?
        }
    } else {
        json!({})
    };

    inject_hook(&mut settings)?;

    let pretty = serde_json::to_string_pretty(&settings)
        .map_err(|e| format!("serializing settings: {}", e))?;
    fs::write(&settings_path, format!("{}\n", pretty))
        .map_err(|e| format!("writing {}: {}", settings_path.display(), e))?;

    println!("wired hook into {}", settings_path.display());
    println!("Restart Claude Code in this project for the hook to take effect.");
    Ok(())
}

fn cmd_uninstall(project_dir: &PathBuf) -> Result<(), String> {
    let project_dir = project_dir
        .canonicalize()
        .map_err(|e| format!("cannot resolve {}: {}", project_dir.display(), e))?;
    let settings_path = project_dir.join(".claude/settings.json");
    if !settings_path.exists() {
        println!("no settings.json at {} — nothing to do", settings_path.display());
        return Ok(());
    }
    let text = fs::read_to_string(&settings_path)
        .map_err(|e| format!("reading {}: {}", settings_path.display(), e))?;
    let mut settings: Value = serde_json::from_str(&text)
        .map_err(|e| format!("parsing {}: {}", settings_path.display(), e))?;

    let removed = remove_hook(&mut settings);
    if !removed {
        println!("no git-guard hook entry found in {}", settings_path.display());
        return Ok(());
    }
    let pretty = serde_json::to_string_pretty(&settings)
        .map_err(|e| format!("serializing settings: {}", e))?;
    fs::write(&settings_path, format!("{}\n", pretty))
        .map_err(|e| format!("writing {}: {}", settings_path.display(), e))?;
    println!("removed git-guard hook from {}", settings_path.display());
    Ok(())
}

fn is_git_guard_command(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed == "git-guard" || trimmed.ends_with("/git-guard") || trimmed.starts_with("git-guard ")
}

fn is_legacy_python_git_guard(s: &str) -> bool {
    s.contains("git_guard.py")
}

fn inject_hook(settings: &mut Value) -> Result<(), String> {
    let root = settings.as_object_mut().ok_or("settings.json root must be an object")?;
    let hooks = root
        .entry("hooks".to_string())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or("settings.hooks must be an object")?;
    let pretooluse = hooks
        .entry("PreToolUse".to_string())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or("settings.hooks.PreToolUse must be an array")?;

    let bash_matcher = pretooluse.iter_mut().find(|m| {
        m.get("matcher").and_then(Value::as_str) == Some("Bash")
    });

    let matcher = match bash_matcher {
        Some(m) => m,
        None => {
            pretooluse.push(json!({ "matcher": "Bash", "hooks": [] }));
            pretooluse.last_mut().unwrap()
        }
    };

    let matcher_obj = matcher.as_object_mut().ok_or("PreToolUse entry must be an object")?;
    let hooks_arr = matcher_obj
        .entry("hooks".to_string())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or("PreToolUse[].hooks must be an array")?;

    let before = hooks_arr.len();
    hooks_arr.retain(|h| {
        let is_legacy = h.get("type").and_then(Value::as_str) == Some("command")
            && h.get("command")
                .and_then(Value::as_str)
                .map(is_legacy_python_git_guard)
                .unwrap_or(false);
        !is_legacy
    });
    if hooks_arr.len() != before {
        println!("removed legacy python git_guard.py hook entry");
    }

    let already = hooks_arr.iter().any(|h| {
        h.get("type").and_then(Value::as_str) == Some("command")
            && h.get("command")
                .and_then(Value::as_str)
                .map(is_git_guard_command)
                .unwrap_or(false)
    });
    if !already {
        hooks_arr.push(json!({ "type": "command", "command": "git-guard" }));
    }
    Ok(())
}

fn remove_hook(settings: &mut Value) -> bool {
    let Some(root) = settings.as_object_mut() else { return false; };
    let Some(hooks) = root.get_mut("hooks").and_then(Value::as_object_mut) else { return false; };
    let Some(pretooluse) = hooks.get_mut("PreToolUse").and_then(Value::as_array_mut) else { return false; };

    let mut changed = false;
    for matcher in pretooluse.iter_mut() {
        let Some(m) = matcher.as_object_mut() else { continue; };
        let Some(hooks_arr) = m.get_mut("hooks").and_then(Value::as_array_mut) else { continue; };
        let before = hooks_arr.len();
        hooks_arr.retain(|h| {
            !(h.get("type").and_then(Value::as_str) == Some("command")
                && h.get("command")
                    .and_then(Value::as_str)
                    .map(is_git_guard_command)
                    .unwrap_or(false))
        });
        if hooks_arr.len() != before {
            changed = true;
        }
    }
    // Prune empty matchers and empty PreToolUse
    pretooluse.retain(|m| {
        m.get("hooks").and_then(Value::as_array).map_or(true, |a| !a.is_empty())
    });
    if pretooluse.is_empty() {
        hooks.remove("PreToolUse");
    }
    if hooks.is_empty() {
        root.remove("hooks");
    }
    changed
}

fn check_command(command: &str, cfg: &Config) -> Option<String> {
    for segment in split_segments(command) {
        let tokens = tokenize(segment);
        if let Some(reason) = check_git_invocation(&tokens, cfg) {
            return Some(reason);
        }
    }
    None
}

fn split_segments(s: &str) -> Vec<&str> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        let two = if i + 1 < bytes.len() {
            &bytes[i..i + 2]
        } else {
            &[][..]
        };
        if two == b"&&" || two == b"||" {
            out.push(&s[start..i]);
            i += 2;
            start = i;
            continue;
        }
        if c == b';' || c == b'|' || c == b'\n' {
            out.push(&s[start..i]);
            i += 1;
            start = i;
            continue;
        }
        i += 1;
    }
    out.push(&s[start..]);
    out
}

fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if in_single {
            if c == '\'' {
                in_single = false;
            } else {
                cur.push(c);
            }
            continue;
        }
        if in_double {
            if c == '"' {
                in_double = false;
            } else if c == '\\' {
                if let Some(&n) = chars.peek() {
                    chars.next();
                    cur.push(n);
                }
            } else {
                cur.push(c);
            }
            continue;
        }
        match c {
            '\'' => in_single = true,
            '"' => in_double = true,
            ' ' | '\t' => {
                if !cur.is_empty() {
                    tokens.push(std::mem::take(&mut cur));
                }
            }
            '\\' => {
                if let Some(&n) = chars.peek() {
                    chars.next();
                    cur.push(n);
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

fn find_git_index(tokens: &[String]) -> Option<usize> {
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        // Skip env-var assignments like FOO=bar
        if t.contains('=') && !t.starts_with('-') && t.chars().next().map_or(false, |c| c.is_ascii_alphabetic() || c == '_') {
            let (name, _) = t.split_once('=').unwrap();
            if name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                i += 1;
                continue;
            }
        }
        // Skip common wrappers
        if matches!(t.as_str(), "sudo" | "nice" | "time" | "env" | "command" | "exec") {
            i += 1;
            continue;
        }
        if t == "git" {
            return Some(i);
        }
        return None;
    }
    None
}

fn skip_global_git_flags(args: &[String]) -> usize {
    let mut i = 0;
    while i < args.len() {
        let t = args[i].as_str();
        match t {
            "-c" | "-C" | "--git-dir" | "--work-tree" | "--namespace" | "--exec-path"
            | "--super-prefix" => {
                i += 2;
                continue;
            }
            _ => {}
        }
        if t.starts_with("--git-dir=")
            || t.starts_with("--work-tree=")
            || t.starts_with("--namespace=")
            || t.starts_with("--exec-path=")
            || t.starts_with("--super-prefix=")
            || t == "--no-pager"
            || t == "--paginate"
            || t == "-p"
            || t == "--bare"
            || t == "--no-replace-objects"
            || t == "--literal-pathspecs"
            || t == "--glob-pathspecs"
            || t == "--noglob-pathspecs"
            || t == "--icase-pathspecs"
        {
            i += 1;
            continue;
        }
        if t.starts_with('-') && t.len() > 1 {
            i += 1;
            continue;
        }
        break;
    }
    i
}

fn check_git_invocation(tokens: &[String], cfg: &Config) -> Option<String> {
    let gi = find_git_index(tokens)?;
    let after = &tokens[gi + 1..];
    let start = skip_global_git_flags(after);
    let subcmd = after.get(start)?.as_str();
    let args: Vec<&str> = after[start + 1..].iter().map(String::as_str).collect();

    match subcmd {
        "stash" if cfg.block_stash => check_stash(&args),
        "reset" if cfg.block_destructive => check_reset(&args),
        "clean" if cfg.block_destructive => check_clean(&args),
        "checkout" if cfg.block_destructive => check_checkout(&args),
        "restore" if cfg.block_destructive => check_restore(&args),
        "switch" if cfg.block_destructive => check_switch(&args),
        "branch" if cfg.block_destructive => check_branch(&args),
        "tag" if cfg.block_destructive => check_tag(&args),
        "push" => check_push(&args, cfg),
        "commit" => check_commit(&args, cfg),
        "rebase" if cfg.block_destructive => check_rebase(&args, cfg),
        "rm" if cfg.block_destructive => Some(format!(
            "git rm removes tracked files (and from working tree). Allow?"
        )),
        "mv" if cfg.block_destructive => Some(format!(
            "git mv renames/overwrites tracked files. Allow?"
        )),
        "filter-branch" | "filter-repo" if cfg.block_destructive => Some(format!(
            "git {} rewrites history irreversibly. Allow?",
            subcmd
        )),
        "reflog" if cfg.block_destructive => check_reflog(&args),
        "gc" if cfg.block_destructive => check_gc(&args),
        "worktree" if cfg.block_destructive => check_worktree(&args),
        "update-ref" if cfg.block_destructive => check_update_ref(&args),
        "replace" if cfg.block_destructive => Some(format!(
            "git replace rewrites object graph. Allow?"
        )),
        "submodule" if cfg.block_destructive => check_submodule(&args),
        "remote" if cfg.block_destructive => check_remote(&args),
        "notes" if cfg.block_destructive => check_notes(&args),
        "am" if cfg.block_destructive => check_am(&args),
        "cherry-pick" if cfg.block_destructive => check_cherry_pick(&args),
        "revert" if cfg.block_destructive => check_revert(&args),
        "merge" if cfg.block_destructive => check_merge(&args),
        _ => None,
    }
}

fn has_short_flag(args: &[&str], ch: char) -> bool {
    args.iter().any(|a| {
        a.starts_with('-') && !a.starts_with("--") && a.len() > 1 && a[1..].contains(ch)
    })
}

fn has_long_flag(args: &[&str], name: &str) -> bool {
    args.iter().any(|a| *a == name || a.starts_with(&format!("{}=", name)))
}

fn check_stash(args: &[&str]) -> Option<String> {
    let sub = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or("push");
    if matches!(sub, "list" | "show") {
        return None;
    }
    Some("git stash is blocked by git_guard. Allow this command?".into())
}

fn check_reset(args: &[&str]) -> Option<String> {
    for a in args {
        if matches!(*a, "--hard" | "--merge" | "--keep") {
            return Some(format!(
                "git reset {} can discard uncommitted work. Allow?",
                a
            ));
        }
    }
    None
}

fn check_clean(args: &[&str]) -> Option<String> {
    if has_short_flag(args, 'f') || has_long_flag(args, "--force") {
        return Some("git clean -f deletes untracked files. Allow?".into());
    }
    None
}

fn check_checkout(args: &[&str]) -> Option<String> {
    // Creating new branches is safe
    if has_long_flag(args, "--orphan") || args.iter().any(|a| matches!(*a, "-b" | "-B")) {
        return None;
    }
    if has_long_flag(args, "--force") || has_short_flag(args, 'f') {
        return Some("git checkout --force overwrites local changes. Allow?".into());
    }
    if let Some(pos) = args.iter().position(|a| *a == "--") {
        let paths = &args[pos + 1..];
        if !paths.is_empty() {
            return Some(format!(
                "git checkout -- {} discards uncommitted changes. Allow?",
                paths.join(" ")
            ));
        }
    }
    for a in args {
        if a.starts_with('-') {
            continue;
        }
        if is_tracked_file(a) {
            return Some(format!(
                "git checkout of '{}' may discard uncommitted file changes. Allow?",
                a
            ));
        }
    }
    None
}

fn check_restore(args: &[&str]) -> Option<String> {
    let positional_paths: Vec<&&str> = args
        .iter()
        .skip_while(|a| **a != "--")
        .skip(1)
        .chain(args.iter().filter(|a| !a.starts_with('-') && **a != "--"))
        .collect();
    let staged_only =
        has_long_flag(args, "--staged") || has_short_flag(args, 'S') && !has_long_flag(args, "--worktree");
    if staged_only {
        // Unstaging only; not destructive to working tree
        return None;
    }
    if positional_paths.is_empty() {
        return None;
    }
    Some("git restore discards working-tree changes. Allow?".into())
}

fn check_switch(args: &[&str]) -> Option<String> {
    if has_long_flag(args, "--discard-changes") {
        return Some("git switch --discard-changes drops uncommitted work. Allow?".into());
    }
    if args.iter().any(|a| *a == "-D") {
        return Some("git switch -D force-creates a branch, overwriting it. Allow?".into());
    }
    None
}

fn check_branch(args: &[&str]) -> Option<String> {
    if args
        .iter()
        .any(|a| matches!(*a, "-d" | "-D" | "--delete" | "--delete-force"))
    {
        return Some("git branch delete removes a branch. Allow?".into());
    }
    if args.iter().any(|a| matches!(*a, "-m" | "-M" | "--move")) {
        return Some("git branch -m/-M renames a branch (force may overwrite). Allow?".into());
    }
    None
}

fn check_tag(args: &[&str]) -> Option<String> {
    if args.iter().any(|a| matches!(*a, "-d" | "-D" | "--delete")) {
        return Some("git tag -d deletes a tag. Allow?".into());
    }
    if args.iter().any(|a| matches!(*a, "-f" | "--force")) {
        return Some("git tag -f overwrites an existing tag. Allow?".into());
    }
    None
}

fn check_push(args: &[&str], cfg: &Config) -> Option<String> {
    let forced = args.iter().any(|a| {
        *a == "-f"
            || *a == "--force"
            || *a == "--force-with-lease"
            || a.starts_with("--force-with-lease=")
            || *a == "--force-if-includes"
            || *a == "--mirror"
    });
    if forced {
        return Some("git push --force can overwrite remote history. Allow?".into());
    }
    if args.iter().any(|a| matches!(*a, "-d" | "--delete")) {
        return Some("git push --delete removes a remote ref. Allow?".into());
    }
    // refspec like `:branch` deletes a remote branch
    if args
        .iter()
        .any(|a| !a.starts_with('-') && a.starts_with(':') && a.len() > 1)
    {
        return Some("git push :ref deletes a remote branch. Allow?".into());
    }
    if let Some(branch) = get_current_branch() {
        if cfg.protected_branches.iter().any(|b| b == &branch) {
            return Some(format!(
                "git push on protected branch '{}' is gated. Allow?",
                branch
            ));
        }
    }
    None
}

fn check_commit(args: &[&str], cfg: &Config) -> Option<String> {
    if has_long_flag(args, "--amend") {
        return Some("git commit --amend rewrites the last commit. Allow?".into());
    }
    if let Some(branch) = get_current_branch() {
        if cfg.protected_branches.iter().any(|b| b == &branch) {
            return Some(format!(
                "git commit on protected branch '{}' is gated. Allow?",
                branch
            ));
        }
    }
    None
}

fn check_rebase(args: &[&str], cfg: &Config) -> Option<String> {
    // Aborts and continues are safe-to-resume operations
    if args
        .iter()
        .any(|a| matches!(*a, "--abort" | "--quit" | "--continue" | "--skip" | "--edit-todo" | "--show-current-patch"))
    {
        return None;
    }
    if let Some(branch) = get_current_branch() {
        if cfg.protected_branches.iter().any(|b| b == &branch) {
            return Some(format!(
                "git rebase on protected branch '{}' rewrites history. Allow?",
                branch
            ));
        }
    }
    Some("git rebase rewrites commits in the current branch. Allow?".into())
}

fn check_reflog(args: &[&str]) -> Option<String> {
    let sub = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or("");
    if matches!(sub, "expire" | "delete") {
        return Some(format!(
            "git reflog {} permanently drops recoverable history. Allow?",
            sub
        ));
    }
    None
}

fn check_gc(args: &[&str]) -> Option<String> {
    if args.iter().any(|a| *a == "--prune" || a.starts_with("--prune=")) {
        return Some("git gc --prune can discard unreachable objects. Allow?".into());
    }
    None
}

fn check_worktree(args: &[&str]) -> Option<String> {
    let sub = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or("");
    if matches!(sub, "remove" | "prune") {
        return Some(format!("git worktree {} deletes a worktree. Allow?", sub));
    }
    None
}

fn check_update_ref(args: &[&str]) -> Option<String> {
    if args.iter().any(|a| *a == "-d" || *a == "--delete") {
        return Some("git update-ref -d deletes a ref. Allow?".into());
    }
    Some("git update-ref modifies refs directly. Allow?".into())
}

fn check_submodule(args: &[&str]) -> Option<String> {
    let sub = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or("");
    if sub == "deinit" {
        return Some("git submodule deinit removes submodule working trees. Allow?".into());
    }
    None
}

fn check_remote(args: &[&str]) -> Option<String> {
    let sub = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or("");
    if matches!(sub, "remove" | "rm") {
        return Some("git remote remove deletes a remote. Allow?".into());
    }
    if sub == "prune" {
        return Some("git remote prune deletes stale remote refs. Allow?".into());
    }
    if sub == "set-url" {
        return Some("git remote set-url changes a remote URL. Allow?".into());
    }
    None
}

fn check_notes(args: &[&str]) -> Option<String> {
    let sub = args.iter().find(|a| !a.starts_with('-')).copied().unwrap_or("");
    if matches!(sub, "remove" | "prune") {
        return Some(format!("git notes {} deletes notes. Allow?", sub));
    }
    None
}

fn check_am(args: &[&str]) -> Option<String> {
    if args.iter().any(|a| *a == "--abort") {
        return Some("git am --abort discards the in-progress patch. Allow?".into());
    }
    None
}

fn check_cherry_pick(args: &[&str]) -> Option<String> {
    if args.iter().any(|a| *a == "--abort") {
        return Some("git cherry-pick --abort discards in-progress work. Allow?".into());
    }
    None
}

fn check_revert(args: &[&str]) -> Option<String> {
    if args.iter().any(|a| *a == "--abort") {
        return Some("git revert --abort discards in-progress work. Allow?".into());
    }
    None
}

fn check_merge(args: &[&str]) -> Option<String> {
    if args.iter().any(|a| *a == "--abort") {
        return Some("git merge --abort discards the in-progress merge. Allow?".into());
    }
    None
}

fn get_current_branch() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    Some(s.trim().to_string())
}

fn is_tracked_file(p: &str) -> bool {
    Command::new("git")
        .args(["ls-files", "--error-unmatch", p])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config::default()
    }

    fn check(s: &str) -> Option<String> {
        check_command(s, &cfg())
    }

    #[test]
    fn allows_non_git() {
        assert!(check("ls -la").is_none());
        assert!(check("echo hello && pwd").is_none());
    }

    #[test]
    fn blocks_stash() {
        assert!(check("git stash").is_some());
        assert!(check("git stash push -m foo").is_some());
        assert!(check("git stash pop").is_some());
        assert!(check("git stash drop").is_some());
        assert!(check("git stash list").is_none());
    }

    #[test]
    fn blocks_reset_hard() {
        assert!(check("git reset --hard HEAD~1").is_some());
        assert!(check("git reset --merge").is_some());
        assert!(check("git reset HEAD foo.txt").is_none());
    }

    #[test]
    fn blocks_clean_force() {
        assert!(check("git clean -fd").is_some());
        assert!(check("git clean --force").is_some());
        assert!(check("git clean -n").is_none());
    }

    #[test]
    fn blocks_branch_delete() {
        assert!(check("git branch -D feature").is_some());
        assert!(check("git branch -d old").is_some());
        assert!(check("git branch new").is_none());
    }

    #[test]
    fn blocks_push_force() {
        assert!(check("git push --force").is_some());
        assert!(check("git push -f origin main").is_some());
        assert!(check("git push --force-with-lease").is_some());
        assert!(check("git push origin :feature").is_some());
        assert!(check("git push --delete origin foo").is_some());
    }

    #[test]
    fn blocks_amend() {
        assert!(check("git commit --amend").is_some());
    }

    #[test]
    fn blocks_filter_branch() {
        assert!(check("git filter-branch -- --all").is_some());
        assert!(check("git filter-repo --invert-paths").is_some());
    }

    #[test]
    fn blocks_worktree_remove() {
        assert!(check("git worktree remove foo").is_some());
        assert!(check("git worktree list").is_none());
    }

    #[test]
    fn handles_chained_commands() {
        assert!(check("cd foo && git push --force").is_some());
        assert!(check("git status; git clean -fdx").is_some());
    }

    #[test]
    fn handles_env_prefix() {
        assert!(check("GIT_TRACE=1 git reset --hard").is_some());
    }

    #[test]
    fn handles_global_flags() {
        assert!(check("git -c color.ui=always reset --hard").is_some());
        assert!(check("git --no-pager clean -fd").is_some());
    }

    #[test]
    fn injects_into_empty_settings() {
        let mut s = json!({});
        inject_hook(&mut s).unwrap();
        let cmd = s
            .pointer("/hooks/PreToolUse/0/hooks/0/command")
            .and_then(Value::as_str);
        assert_eq!(cmd, Some("git-guard"));
        let matcher = s.pointer("/hooks/PreToolUse/0/matcher").and_then(Value::as_str);
        assert_eq!(matcher, Some("Bash"));
    }

    #[test]
    fn injects_into_existing_bash_matcher() {
        let mut s = json!({
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {"type": "command", "command": "other-hook"}
                        ]
                    }
                ]
            }
        });
        inject_hook(&mut s).unwrap();
        let arr = s
            .pointer("/hooks/PreToolUse/0/hooks")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(arr.len(), 2);
        assert!(arr.iter().any(|h| h.get("command").and_then(Value::as_str) == Some("git-guard")));
    }

    #[test]
    fn idempotent_inject() {
        let mut s = json!({});
        inject_hook(&mut s).unwrap();
        inject_hook(&mut s).unwrap();
        let arr = s.pointer("/hooks/PreToolUse/0/hooks").and_then(Value::as_array).unwrap();
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn uninstall_removes_entry_and_prunes() {
        let mut s = json!({});
        inject_hook(&mut s).unwrap();
        assert!(remove_hook(&mut s));
        assert!(s.pointer("/hooks").is_none());
    }

    #[test]
    fn uninstall_preserves_other_hooks() {
        let mut s = json!({
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {"type": "command", "command": "other-hook"},
                            {"type": "command", "command": "git-guard"}
                        ]
                    }
                ]
            }
        });
        assert!(remove_hook(&mut s));
        let arr = s.pointer("/hooks/PreToolUse/0/hooks").and_then(Value::as_array).unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(
            arr[0].get("command").and_then(Value::as_str),
            Some("other-hook")
        );
    }

    #[test]
    fn inject_replaces_legacy_python_hook() {
        let mut s = json!({
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {"type": "command", "command": "python3 $CLAUDE_PROJECT_DIR/.claude/hooks/git_guard.py"}
                        ]
                    }
                ]
            }
        });
        inject_hook(&mut s).unwrap();
        let arr = s
            .pointer("/hooks/PreToolUse/0/hooks")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(
            arr[0].get("command").and_then(Value::as_str),
            Some("git-guard")
        );
    }

    #[test]
    fn detects_absolute_path_invocations() {
        assert!(is_git_guard_command("git-guard"));
        assert!(is_git_guard_command("/usr/local/bin/git-guard"));
        assert!(is_git_guard_command("git-guard --foo"));
        assert!(!is_git_guard_command("some-other-tool"));
    }
}
