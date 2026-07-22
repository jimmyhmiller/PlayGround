//! `memscope check` / `setup` / `agent-setup` — where you stand, and the
//! guided path to ready.
//!
//! All three commands share one pipeline: `assess` gathers the facts (binary
//! symbols + project source), `plan` turns them into the ordered list of steps
//! still needed. `check` prints the facts and how many steps remain (exit 0 =
//! ready). `setup` prints ONLY the next step, and you re-run it after each one
//! — every re-run re-verifies from scratch, so it doubles as the "did I do it
//! right?" check. `agent-setup` prints the pending steps as a compact prompt an
//! AI agent can execute (`claude "$(memscope agent-setup)"`).
//!
//! Output discipline: no walls of text. One step at a time, a handful of lines
//! each, exact edits only.

use std::path::{Path, PathBuf};

use memscope_symbols::inspect::{inspect_binary, Allocator, BinaryFacts, DebugInfo, Injection};

// --- assessment --------------------------------------------------------------

/// What the project's *source* says, which the binary alone can't.
struct SourceFacts {
    /// (file, line, the `static …` declaration under `#[global_allocator]`)
    global_allocators: Vec<(PathBuf, usize, String)>,
    release_debug: bool,
    /// Cargo.toml already mentions a memscope dependency.
    memscope_dep: bool,
}

struct Assessment {
    /// The project directory, when the target was one.
    dir: Option<PathBuf>,
    source: Option<SourceFacts>,
    /// Inspected binary (the target itself, or the newest build in target/).
    bin: Option<PathBuf>,
    bin_profile: Option<&'static str>,
    facts: Option<BinaryFacts>,
}

fn assess(target: &str) -> Result<Assessment, String> {
    let path = Path::new(target);
    if path.is_file() {
        let facts = inspect_binary(path, true).map_err(|e| e.to_string())?;
        return Ok(Assessment {
            dir: None,
            source: None,
            bin: Some(path.to_path_buf()),
            bin_profile: None,
            facts: Some(facts),
        });
    }
    if !path.is_dir() {
        return Err(format!("{target}: not a file or directory"));
    }
    let manifest = path.join("Cargo.toml");
    if !manifest.exists() {
        return Err(format!(
            "{}: no Cargo.toml here — pass a cargo project directory or a built binary",
            path.display()
        ));
    }
    let manifest_text = std::fs::read_to_string(&manifest).map_err(|e| e.to_string())?;

    let mut global_allocators = Vec::new();
    scan_rs_files(path, &mut |file, text| {
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.contains("#[global_allocator]") && !line.trim_start().starts_with("//") {
                let decl = lines[i + 1..]
                    .iter()
                    .take(5)
                    .find(|l| l.contains("static"))
                    .map(|l| l.trim().to_string())
                    .unwrap_or_default();
                global_allocators.push((file.to_path_buf(), i + 1, decl));
            }
        }
    });

    let source = SourceFacts {
        global_allocators,
        release_debug: profile_has_debug(&manifest_text, "release"),
        memscope_dep: manifest_text.contains("memscope"),
    };

    let (bin, bin_profile, facts) = match newest_binary(path) {
        Some((b, profile)) => {
            let f = inspect_binary(&b, true).map_err(|e| e.to_string())?;
            (Some(b), Some(profile), Some(f))
        }
        None => (None, None, None),
    };
    Ok(Assessment { dir: Some(path.to_path_buf()), source: Some(source), bin, bin_profile, facts })
}

impl Assessment {
    fn memscope_installed(&self) -> bool {
        self.facts.as_ref().map(|f| f.memscope_linked).unwrap_or(false)
            || self
                .source
                .as_ref()
                .map(|s| s.global_allocators.iter().any(|(_, _, d)| d.contains("MemScope")))
                .unwrap_or(false)
    }

    /// The allocator memscope must wrap, if any: (display name, Rust type expr).
    fn custom_allocator(&self) -> Option<(String, String)> {
        if self.memscope_installed() {
            return None;
        }
        if let Some(BinaryFacts { allocator: Allocator::Named { name, wrap_type }, .. }) =
            &self.facts
        {
            return Some((name.to_string(), wrap_type.to_string()));
        }
        // Source-only detection: pull the inner type out of `static X: Ty = …;`.
        let (_, _, decl) = self.source.as_ref()?.global_allocators.first()?;
        let ty = decl
            .split_once(':')
            .map(|(_, rest)| rest.split('=').next().unwrap_or("").trim().to_string())
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| "YourAllocator".to_string());
        Some(("a custom allocator".to_string(), ty))
    }
}

// --- the plan ----------------------------------------------------------------

struct Step {
    title: &'static str,
    lines: Vec<String>,
    /// What to do after this step (always ends with re-running `memscope setup`).
    then: String,
    /// This step edits the user's code — the agent handoff applies.
    code_change: bool,
}

/// The ordered steps still needed. Empty = ready. `live` forces the in-process
/// agent integration even when injection alone would do.
fn plan(a: &Assessment, live: bool) -> Vec<Step> {
    let mut steps = Vec::new();
    let setup_cmd = if live { "memscope setup --live" } else { "memscope setup" };
    let installed = a.memscope_installed();
    let custom = a.custom_allocator();

    let debug_missing = match (&a.facts, &a.source) {
        // Trust the binary when we have one: it either has DWARF or it doesn't.
        (Some(f), _) => matches!(f.debug_info, DebugInfo::Absent { .. }),
        (None, Some(s)) => !s.release_debug,
        (None, None) => false,
    };

    // Integration (the only code-change step). Needed when a custom allocator
    // blinds injection, or when the user asked for the live agent.
    if !installed && (custom.is_some() || live) {
        let (_, wrap) = custom
            .clone()
            .unwrap_or(("".into(), "".into()));
        let mut lines = Vec::new();
        let mut n = 0;
        let mut item = |lines: &mut Vec<String>, s: String| {
            n += 1;
            lines.push(format!("{n}. {s}"));
        };
        if a.source.as_ref().map(|s| !s.memscope_dep).unwrap_or(true) {
            item(&mut lines, format!("Cargo.toml [dependencies]:  memscope = {{ path = \"{}\" }}", memscope_crate_path()));
        }
        if debug_missing {
            lines.push("   …and in the same file:      [profile.release] debug = true".to_string());
        }
        if wrap.is_empty() {
            item(&mut lines, "In your binary's main.rs:".to_string());
            lines.push("     #[global_allocator]".to_string());
            lines.push("     static GLOBAL: memscope::MemScope = memscope::MemScope::system();".to_string());
        } else {
            item(&mut lines, "Replace your #[global_allocator] static with the wrapped one:".to_string());
            lines.push(format!("     static GLOBAL: memscope::MemScope<{wrap}> = memscope::MemScope::new({wrap});"));
        }
        item(&mut lines, "First lines of fn main():".to_string());
        lines.push("     memscope::set_mode(memscope::Mode::Full);".to_string());
        lines.push("     memscope::start_agent().unwrap();".to_string());
        steps.push(Step {
            title: "install the tracking allocator",
            lines,
            then: format!("cargo build --release && {setup_cmd}"),
            code_change: true,
        });
    } else if debug_missing {
        // Only a standalone step when integration didn't already fold it in.
        steps.push(Step {
            title: "enable debug info (needed for type names)",
            lines: vec![
                "Add to Cargo.toml:".to_string(),
                "  [profile.release]".to_string(),
                "  debug = true".to_string(),
            ],
            then: format!("cargo build --release && {setup_cmd}"),
            code_change: true,
        });
    }

    if a.dir.is_some() && a.bin.is_none() && steps.is_empty() {
        steps.push(Step {
            title: "build the binary",
            lines: vec!["cargo build --release".to_string()],
            then: setup_cmd.to_string(),
            code_change: false,
        });
    }

    // Signing only matters on the injection path (no in-process agent).
    if !installed && !live && custom.is_none() {
        if let Some(BinaryFacts { injection: Injection::Blocked { reason, fix }, .. }) = &a.facts {
            steps.push(Step {
                title: "unblock injection (signature)",
                lines: vec![format!("{reason}:"), format!("  {fix}")],
                then: setup_cmd.to_string(),
                code_change: false,
            });
        }
    }

    steps
}

/// Path to the memscope crate for a `{ path = … }` dependency. The CLI is built
/// from this workspace, so the crate lives next to this crate's manifest.
fn memscope_crate_path() -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../memscope");
    std::fs::canonicalize(&p).unwrap_or(p).display().to_string()
}

fn ready_line(a: &Assessment) -> String {
    let target = a
        .bin
        .as_ref()
        .map(|b| b.display().to_string())
        .unwrap_or_else(|| "<your binary>".to_string());
    if a.memscope_installed() {
        "run your program, then:  memscope monitor   (or record: MEMSCOPE_RECORD=rec.mscope, then memscope analyze)".to_string()
    } else {
        format!("memscope run --on-exit -- {target}   (more: memscope help)")
    }
}

// --- commands ----------------------------------------------------------------

fn target_arg(args: &[String]) -> &str {
    args.iter().find(|a| !a.starts_with("--")).map(String::as_str).unwrap_or(".")
}

pub fn cmd_check(args: &[String]) -> Result<(), String> {
    let a = assess(target_arg(args))?;
    let live = args.iter().any(|x| x == "--live");

    let shown = a.bin.as_ref().or(a.dir.as_ref()).unwrap();
    println!("== memscope check: {} ==", shown.display());
    if let (Some(_), Some(profile)) = (&a.dir, a.bin_profile) {
        println!("  binary        newest {profile} build in target/");
    }
    if let Some(f) = &a.facts {
        match (&f.allocator, a.memscope_installed()) {
            (_, true) => println!("  allocator     memscope — already installed  ✓"),
            (Allocator::SystemDefault, _) => match a.custom_allocator() {
                Some((name, _)) => println!("  allocator     {name} (from source)  ✗ injection can't see it"),
                None => println!("  allocator     system default  ✓ injection sees everything"),
            },
            (Allocator::Named { name, .. }, _) => {
                println!("  allocator     {name}  ✗ bypasses malloc — injection can't see it")
            }
        }
        match &f.debug_info {
            DebugInfo::Present { source } => println!("  debug info    {source}  ✓ types resolve"),
            DebugInfo::Absent { .. } => println!("  debug info    none  ✗ dumps would be untyped"),
            DebugInfo::Unknown { detail } => println!("  debug info    ? {detail}"),
        }
        match &f.injection {
            Injection::Ok { detail } => println!("  signing       {detail}  ✓ injectable"),
            Injection::Blocked { reason, .. } => println!("  signing       ✗ {reason}"),
        }
    } else if let Some(s) = &a.source {
        println!("  binary        none built yet");
        match a.custom_allocator() {
            Some((name, _)) => println!("  allocator     {name} (from source)  ✗ injection can't see it"),
            None if a.memscope_installed() => println!("  allocator     memscope — already installed  ✓"),
            None => println!("  allocator     system default (no #[global_allocator] in source)  ✓"),
        }
        println!(
            "  debug info    [profile.release] debug {}",
            if s.release_debug { "= true  ✓" } else { "not set  ✗" }
        );
    }

    let steps = plan(&a, live);
    println!();
    if steps.is_empty() {
        println!("  ✓ ready — {}", ready_line(&a));
        if !a.memscope_installed() && !live {
            println!("    live monitor / marks / diff need 2 lines of code:  memscope setup --live");
        }
        Ok(())
    } else {
        let n = steps.len();
        println!(
            "  → {n} step{} needed — walk through {}:  memscope setup {}{}",
            if n == 1 { "" } else { "s" },
            if n == 1 { "it" } else { "them" },
            target_arg(args),
            if live { " --live" } else { "" },
        );
        // Scriptable: exit 1 until ready, so agents/loops can poll `check`.
        std::process::exit(1);
    }
}

pub fn cmd_setup(args: &[String]) -> Result<(), String> {
    let a = assess(target_arg(args))?;
    let live = args.iter().any(|x| x == "--live");
    let steps = plan(&a, live);

    if steps.is_empty() {
        println!("✓ ready — {}", ready_line(&a));
        if !a.memscope_installed() && !live {
            println!("  live monitor / marks / diff need 2 lines of code:  memscope setup --live");
        }
        return Ok(());
    }

    let step = &steps[0];
    println!("memscope setup — step 1 of {}: {}", steps.len(), step.title);
    println!();
    for l in &step.lines {
        println!("  {l}");
    }
    if step.code_change {
        println!();
        println!(
            "  or hand it to an agent:  claude \"$(memscope agent-setup {}{})\"",
            target_arg(args),
            if live { " --live" } else { "" }
        );
    }
    println!();
    println!("  then:  {}", step.then);
    Ok(())
}

/// Print the pending setup as a prompt for an AI agent. Stdout carries ONLY the
/// prompt (so `claude "$(memscope agent-setup)"` is clean); the usage hint goes
/// to stderr.
pub fn cmd_agent_setup(args: &[String]) -> Result<(), String> {
    let target = target_arg(args);
    let a = assess(target)?;
    let live = args.iter().any(|x| x == "--live");
    let steps = plan(&a, live);

    let dir = a
        .dir
        .clone()
        .or_else(|| a.bin.as_ref().and_then(|b| b.parent().map(|p| p.to_path_buf())))
        .unwrap_or_else(|| PathBuf::from("."));
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);

    if steps.is_empty() {
        println!("memscope is already set up in {} — nothing to do.", dir.display());
        return Ok(());
    }

    let me = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "memscope".to_string());

    println!("In {}: finish setting up the memscope memory profiler. Steps:", dir.display());
    for (i, step) in steps.iter().enumerate() {
        println!("{}. {}:", i + 1, step.title);
        for l in &step.lines {
            println!("   {l}");
        }
    }
    println!("Change nothing else. Verify when done: `cargo build --release` succeeds and");
    println!("`{me} check {}` exits 0 (it prints \"ready\").", dir.display());

    eprintln!();
    eprintln!("[memscope] pass the text above to an agent, e.g.:  claude \"$(memscope agent-setup {target})\"");
    Ok(())
}

/// Preflight for `memscope run`: refuse the two cases where the run would
/// silently produce a useless dump. Short messages; `setup` has the details.
pub fn preflight_run(bin: &Path, force: bool) -> Result<(), String> {
    // Target may be resolved via PATH (`memscope run -- ls`); only inspect
    // real paths, and never fail the run because inspection itself failed.
    if !bin.is_file() {
        return Ok(());
    }
    let facts = match inspect_binary(bin, false) {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };

    if let Allocator::Named { name, .. } = &facts.allocator {
        let msg = format!(
            "{} uses {name} — injection interposes malloc, so the dump would be (almost) empty.\n\
             next: memscope setup <its project dir>   (walks you through the 2-line fix; or pass --force)",
            bin.display()
        );
        if force {
            eprintln!("[memscope] WARNING (--force): {msg}");
        } else {
            return Err(msg);
        }
    }

    if let Injection::Blocked { reason, fix } = &facts.injection {
        let msg = format!("injection would be silently ignored: {reason}.\nfix: {fix}   (or pass --force)");
        if force {
            eprintln!("[memscope] WARNING (--force): {msg}");
        } else {
            return Err(msg);
        }
    }

    if facts.memscope_linked {
        eprintln!(
            "[memscope] note: {} already has memscope built in — you can attach directly (memscope monitor).",
            bin.display()
        );
    }
    if let DebugInfo::Absent { .. } = &facts.debug_info {
        eprintln!("[memscope] warning: no debug info — the dump will be untyped (memscope setup to fix).");
    }
    Ok(())
}

// --- project scanning helpers -----------------------------------------------

/// Walk `.rs` files under `dir`, skipping build output and VCS internals.
fn scan_rs_files(dir: &Path, cb: &mut dyn FnMut(&Path, &str)) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "target" || name == ".git" || name == "node_modules" {
                continue;
            }
            scan_rs_files(&path, cb);
        } else if name.ends_with(".rs") {
            if let Ok(text) = std::fs::read_to_string(&path) {
                cb(&path, &text);
            }
        }
    }
}

/// Does `[profile.<name>]` set debug info on? Naive TOML scan — good enough
/// for a diagnostic; anything ambiguous reads as "not set".
fn profile_has_debug(manifest: &str, profile: &str) -> bool {
    let header = format!("[profile.{profile}]");
    let mut in_section = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_section = t == header;
            continue;
        }
        if in_section && t.starts_with("debug") {
            let val = t.splitn(2, '=').nth(1).unwrap_or("").trim();
            return !matches!(val, "false" | "0" | "\"none\"");
        }
    }
    false
}

/// Newest executable in `target/{release,debug}` (top level only — that's
/// where cargo puts bin artifacts; `deps/` holds hashed intermediates).
fn newest_binary(dir: &Path) -> Option<(PathBuf, &'static str)> {
    use std::os::unix::fs::PermissionsExt;
    let mut best: Option<(PathBuf, &'static str, std::time::SystemTime)> = None;
    for profile in ["release", "debug"] {
        let d = dir.join("target").join(profile);
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() || meta.permissions().mode() & 0o111 == 0 {
                continue;
            }
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy();
                if matches!(&*ext, "d" | "dylib" | "so" | "rlib" | "dSYM") {
                    continue;
                }
            }
            let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if best.as_ref().map(|(_, _, t)| mtime > *t).unwrap_or(true) {
                best = Some((path, profile, mtime));
            }
        }
    }
    best.map(|(p, profile, _)| (p, profile))
}
