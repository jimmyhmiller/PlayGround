//! `memscope check` — the "where do I stand?" doctor, plus the preflight that
//! `memscope run` uses to refuse footguns.
//!
//! Point it at a binary or a cargo project directory and it answers, in order:
//! do you need code changes at all, will injection see your allocations, and
//! will types resolve — each with the exact next command or the exact lines to
//! add.

use std::path::{Path, PathBuf};

use memscope_symbols::inspect::{inspect_binary, Allocator, BinaryFacts, DebugInfo, Injection};

pub fn cmd_check(args: &[String]) -> Result<(), String> {
    let target = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(String::as_str)
        .unwrap_or(".");
    let path = Path::new(target);
    if path.is_dir() {
        check_project(path)
    } else if path.is_file() {
        let facts = inspect_binary(path, true).map_err(|e| e.to_string())?;
        render(&facts, None);
        Ok(())
    } else {
        Err(format!(
            "{target}: not a file or directory\nusage: memscope check [BINARY | CARGO_DIR]   (default: current directory)"
        ))
    }
}

/// What the project's *source* says, which the binary alone can't:
/// where a `#[global_allocator]` is declared and whether debug info is enabled.
struct SourceFacts {
    root: PathBuf,
    /// (file, line, the `static …` declaration under the attribute)
    global_allocators: Vec<(PathBuf, usize, String)>,
    release_debug: bool,
    /// Which profile the inspected binary came from, when checking a project.
    bin_profile: Option<&'static str>,
}

fn check_project(dir: &Path) -> Result<(), String> {
    let manifest = dir.join("Cargo.toml");
    if !manifest.exists() {
        return Err(format!(
            "{}: no Cargo.toml here — pass a cargo project directory or a built binary",
            dir.display()
        ));
    }
    let manifest_text = std::fs::read_to_string(&manifest).map_err(|e| e.to_string())?;

    let mut global_allocators = Vec::new();
    scan_rs_files(dir, &mut |file, text| {
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

    let mut source = SourceFacts {
        root: dir.to_path_buf(),
        global_allocators,
        release_debug: profile_has_debug(&manifest_text, "release"),
        bin_profile: None,
    };

    match newest_binary(dir) {
        Some((bin, profile)) => {
            let facts = inspect_binary(&bin, true).map_err(|e| e.to_string())?;
            source.bin_profile = Some(profile);
            render(&facts, Some(&source));
        }
        None => {
            println!("== memscope check: {} ==", dir.display());
            println!();
            println!("  no built binary found under {}/target", dir.display());
            render_source_only(&source);
        }
    }
    Ok(())
}

fn render_source_only(source: &SourceFacts) {
    for (file, line, decl) in &source.global_allocators {
        let rel = file.strip_prefix(&source.root).unwrap_or(file);
        if decl.contains("MemScope") {
            println!("  allocator     memscope already installed  ({}:{line})", rel.display());
        } else {
            println!("  allocator     custom #[global_allocator]  ({}:{line})  {decl}", rel.display());
        }
    }
    if source.global_allocators.is_empty() {
        println!("  allocator     system default (no #[global_allocator] in source)");
    }
    println!(
        "  debug info    [profile.release] debug {}",
        if source.release_debug { "= true  ✓ types will resolve" } else { "not set  ✗ release types won't resolve" }
    );
    println!();
    println!("VERDICT — build first, then re-run `memscope check` on the result:");
    println!("    cargo build --release && memscope check .");
    if !source.release_debug {
        println!();
        println!("  For typed release dumps, add to Cargo.toml first:");
        println!("    [profile.release]");
        println!("    debug = true");
    }
}

fn render(facts: &BinaryFacts, source: Option<&SourceFacts>) {
    println!("== memscope check: {} ==", facts.path.display());
    println!();
    if let Some(profile) = source.and_then(|s| s.bin_profile) {
        println!("  binary        newest {profile} build in target/");
    }

    // -- facts table ---------------------------------------------------------
    let src_alloc = source.and_then(|s| s.global_allocators.first());
    match (&facts.allocator, facts.memscope_linked) {
        (_, true) => {
            let loc = src_alloc
                .filter(|(_, _, d)| d.contains("MemScope"))
                .map(|(f, l, _)| {
                    let rel = source.map(|s| f.strip_prefix(&s.root).unwrap_or(f)).unwrap_or(f);
                    format!("  ({}:{l})", rel.display())
                })
                .unwrap_or_default();
            println!("  allocator     memscope (already installed){loc}");
        }
        (Allocator::SystemDefault, false) => match src_alloc {
            Some((f, l, decl)) => {
                let rel = source.map(|s| f.strip_prefix(&s.root).unwrap_or(f)).unwrap_or(f);
                println!("  allocator     ! custom #[global_allocator] at {}:{l}", rel.display());
                println!("                  {decl}");
            }
            None => println!("  allocator     system default              ✓ injection sees every allocation"),
        },
        (Allocator::Named { name, .. }, false) => {
            println!("  allocator     {name}                    ✗ bypasses malloc — injection would see (almost) nothing");
        }
    }

    match &facts.debug_info {
        DebugInfo::Present { source } => println!("  debug info    {source}  ✓ types will be recovered"),
        DebugInfo::Absent { detail } => println!("  debug info    none  ✗ dumps will be complete but untyped\n                  ({detail})"),
        DebugInfo::Unknown { detail } => println!("  debug info    ? {detail}"),
    }

    match &facts.injection {
        Injection::Ok { detail } => println!("  signing       {detail}  ✓ injection allowed"),
        Injection::Blocked { reason, .. } => println!("  signing       ✗ {reason}"),
    }

    if !facts.is_rust {
        println!("  language      not Rust (or stripped) — injection + dumps still work; type names may be C symbols");
    }

    // -- verdict -------------------------------------------------------------
    println!();
    let bin = facts.path.display();
    if facts.memscope_linked {
        println!("VERDICT — memscope is already in this program. No changes needed.");
        println!();
        println!("  Run it, then attach live:");
        println!("    memscope monitor        # live heap by type");
        println!("    memscope graph          # top retainers (retained size)");
        println!("  Or if it records to a file (record_to_file):");
        println!("    memscope analyze rec.mscope");
        finish_debug_note(facts, source);
        return;
    }

    // A custom allocator makes injection blind regardless of anything else.
    let custom = match &facts.allocator {
        Allocator::Named { name, wrap_type } => Some((*name, *wrap_type)),
        Allocator::SystemDefault => src_alloc
            .filter(|(_, _, d)| !d.contains("MemScope"))
            .map(|_| ("your custom allocator", "YourAllocator")),
    };
    if let Some((name, wrap_type)) = custom {
        println!("VERDICT — code changes needed (2 lines).");
        println!();
        println!("  This program's global allocator is {name}, which bypasses malloc, so");
        println!("  `memscope run` (malloc interposition) can't see its allocations.");
        println!("  Wrap it — memscope tracks, {name} still does the allocating:");
        println!();
        println!("    #[global_allocator]");
        println!("    static GLOBAL: memscope::MemScope<{wrap_type}> =");
        println!("        memscope::MemScope::new({wrap_type});");
        if wrap_type == "YourAllocator" {
            println!("    // YourAllocator = the type currently in your #[global_allocator]");
        }
        println!();
        println!("    fn main() {{");
        println!("        memscope::set_mode(memscope::Mode::Full);");
        println!("        memscope::start_agent().unwrap();     // then: memscope monitor");
        println!("        // or: memscope::record_to_file(\"rec.mscope\").unwrap();");
        println!("    }}");
        finish_debug_note(facts, source);
        return;
    }

    if let Injection::Blocked { reason, fix } = &facts.injection {
        println!("VERDICT — no code changes needed, but injection is blocked: {reason}.");
        println!();
        println!("  Fix, then dump the unmodified binary:");
        println!("    {fix}");
        println!("    memscope run --on-exit -- {bin}");
        finish_debug_note(facts, source);
        return;
    }

    println!("VERDICT — no code changes needed.");
    println!();
    println!("  Dump this program's heap, unmodified:");
    println!("    memscope run --on-exit  -- {bin}        # what was never freed, at exit");
    println!("    memscope run --after 5s -- {bin}        # steady state, 5s in");
    println!("    memscope run --at-bytes 50MB -- {bin}   # the moment the heap first hits 50MB");
    println!("    memscope run --out rec.mscope -- {bin}  # full allocation stream → analyze/flamegraph/perfetto");
    println!();
    println!("  Want live monitoring, checkpoints (mark), or diffs? That takes the 2-line agent:");
    println!("    #[global_allocator]");
    println!("    static GLOBAL: memscope::MemScope = memscope::MemScope::system();");
    println!("    // in main: memscope::set_mode(memscope::Mode::Full); memscope::start_agent().unwrap();");
    finish_debug_note(facts, source);
}

fn finish_debug_note(facts: &BinaryFacts, source: Option<&SourceFacts>) {
    if let DebugInfo::Absent { .. } = &facts.debug_info {
        println!();
        println!("  note: no debug info — everything above works, but types show as raw sizes.");
        if let Some(s) = source {
            if !s.release_debug {
                println!("  Fix in Cargo.toml (then rebuild):");
                println!("    [profile.release]");
                println!("    debug = true");
                return;
            }
        }
        println!("  Build with `debug = true` in the profile you ship (no nightly needed).");
    }
}

/// Preflight for `memscope run`: same facts, but as hard errors for the two
/// cases where the run would *silently* produce garbage (empty or missing
/// dump), and warnings for the rest. `--force` downgrades the errors.
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
            "{} uses {name} as its allocator — injection interposes malloc, so the dump would be (almost) empty.\n\
             Run `memscope check {}` for the 2-line fix (wrap it in MemScope::new).",
            bin.display(),
            bin.display()
        );
        if force {
            eprintln!("[memscope] WARNING (--force): {msg}");
        } else {
            return Err(format!("{msg}\nPass --force to run anyway."));
        }
    }

    if let Injection::Blocked { reason, fix } = &facts.injection {
        let msg = format!(
            "injection into {} will be silently ignored: {reason}.\nFix: {fix}",
            bin.display()
        );
        if force {
            eprintln!("[memscope] WARNING (--force): {msg}");
        } else {
            return Err(format!("{msg}\nPass --force to run anyway."));
        }
    }

    if facts.memscope_linked {
        eprintln!(
            "[memscope] note: {} already has memscope compiled in — you can attach directly \
             (memscope monitor) instead of injecting.",
            bin.display()
        );
    }
    if let DebugInfo::Absent { detail } = &facts.debug_info {
        eprintln!("[memscope] warning: no debug info ({detail}) — the dump will be untyped.");
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
