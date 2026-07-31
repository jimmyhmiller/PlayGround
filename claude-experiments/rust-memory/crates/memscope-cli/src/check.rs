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

use memscope_symbols::inspect::{
    inspect_binary, Allocator, BinaryFacts, BinaryKind, DebugInfo, Injection,
};

// --- assessment --------------------------------------------------------------

/// What the project's *source* says, which the binary alone can't.
struct SourceFacts {
    /// (file, line, the `static …` declaration under `#[global_allocator]`)
    global_allocators: Vec<(PathBuf, usize, String)>,
    release_debug: bool,
    /// Cargo.toml already mentions a memscope dependency.
    memscope_dep: bool,
    /// Cargo.toml builds a `cdylib` — a host-loaded module, not a program.
    cdylib: bool,
    /// A `memscope::init()` call is already in the source.
    init_called: bool,
    /// The file most likely to hold the module's init hook (`lib.rs`).
    lib_rs: Option<PathBuf>,
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
    let mut init_called = false;
    let mut lib_rs = None;
    scan_rs_files(path, &mut |file, text| {
        if file.file_name().is_some_and(|n| n == "lib.rs") && lib_rs.is_none() {
            lib_rs = Some(file.to_path_buf());
        }
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            if line.contains("#[global_allocator]") {
                let decl = lines[i + 1..]
                    .iter()
                    .take(5)
                    .find(|l| l.contains("static"))
                    .map(|l| l.trim().to_string())
                    .unwrap_or_default();
                global_allocators.push((file.to_path_buf(), i + 1, decl));
            }
            if line.contains("memscope::init(") {
                init_called = true;
            }
        }
    });

    let pkg = package_name(&manifest_text);
    let source = SourceFacts {
        global_allocators,
        // A workspace member usually has no `[profile.*]` of its own — cargo
        // takes the profile from the workspace root, so that's where to look
        // when the member says nothing.
        release_debug: profile_has_debug(&manifest_text, "release")
            || workspace_manifest(path)
                .is_some_and(|text| profile_has_debug(&text, "release")),
        memscope_dep: manifest_text.contains("memscope"),
        // Textual, deliberately: a cdylib is declared as `crate-type = [...]`
        // (or `crate_type`), and that word appearing at all is enough to tell us
        // "this project ships a module a host loads" — which is the only thing
        // the plan branches on. Cheaper and more robust than a TOML parse over
        // the many ways cargo lets you spell it.
        cdylib: manifest_text.contains("cdylib"),
        init_called,
        lib_rs,
    };

    // A cdylib project's artifact isn't in the exe search path: look for the
    // module (including a `.node` copy, which is where napi-rs leaves it).
    if source.cdylib {
        if let Some((m, profile)) = newest_module(path, pkg.as_deref()) {
            let facts = inspect_binary(&m, true).map_err(|e| e.to_string())?;
            return Ok(Assessment {
                dir: Some(path.to_path_buf()),
                source: Some(source),
                bin: Some(m),
                bin_profile: profile,
                facts: Some(facts),
            });
        }
    }

    let (bin, bin_profile, facts) = match newest_binary(path, &bin_names(path, &manifest_text)) {
        Some((b, profile)) => {
            let f = inspect_binary(&b, true).map_err(|e| e.to_string())?;
            (Some(b), Some(profile), Some(f))
        }
        None => (None, None, None),
    };
    Ok(Assessment { dir: Some(path.to_path_buf()), source: Some(source), bin, bin_profile, facts })
}

impl Assessment {
    /// Is the target a module a host process loads (Node `.node` addon, Python
    /// extension) rather than a program? Injection doesn't apply to these, and
    /// the setup lands in an init hook instead of `fn main()`.
    fn is_module(&self) -> bool {
        self.facts.as_ref().map(|f| f.kind == BinaryKind::DynamicModule).unwrap_or(false)
            || self.source.as_ref().map(|s| s.cdylib).unwrap_or(false)
    }

    /// The module's init hook is where `memscope::init()` goes; report whether
    /// it's already there (source-visible only, hence `Option`).
    fn init_installed(&self) -> Option<bool> {
        self.source.as_ref().map(|s| s.init_called)
    }

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
    // A module has no `main` to inject into and no `fn main()` to edit: the
    // in-process path is the ONLY path, so integration is never optional.
    let module = a.is_module();
    // napi-rs projects rebuild through npm, which also re-copies the `.node`;
    // mention it as a note rather than in the command line we tell them to run.
    let napi_note = "   (napi-rs: `npm run build` — it rebuilds and re-copies the .node)";

    let debug_missing = match (&a.facts, &a.source) {
        // Trust the binary when we have one: it either has DWARF or it doesn't.
        (Some(f), _) => matches!(f.debug_info, DebugInfo::Absent { .. }),
        (None, Some(s)) => !s.release_debug,
        (None, None) => false,
    };

    // Integration (the only code-change step). Needed when a custom allocator
    // blinds injection, when the user asked for the live agent, or always for a
    // module (there is no injection path for it).
    if !installed && (custom.is_some() || live || module) {
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
        let where_static = if module {
            let lib = a
                .source
                .as_ref()
                .and_then(|s| s.lib_rs.as_ref())
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "your lib.rs".to_string());
            format!("In {lib}:")
        } else {
            "In your binary's main.rs:".to_string()
        };
        if wrap.is_empty() {
            item(&mut lines, where_static);
            lines.push("     #[global_allocator]".to_string());
            lines.push("     static GLOBAL: memscope::MemScope = memscope::MemScope::system();".to_string());
        } else {
            item(&mut lines, "Replace your #[global_allocator] static with the wrapped one:".to_string());
            lines.push(format!("     static GLOBAL: memscope::MemScope<{wrap}> = memscope::MemScope::new({wrap});"));
        }
        if module {
            item(&mut lines, "In the same file, in your module's init hook:".to_string());
            lines.extend(init_hook_lines());
        } else {
            item(&mut lines, "First lines of fn main():".to_string());
            lines.push("     memscope::set_mode(memscope::Mode::Full);".to_string());
            lines.push("     memscope::start_agent().unwrap();".to_string());
        }
        if module {
            lines.push(napi_note.to_string());
        }
        steps.push(Step {
            title: "install the tracking allocator",
            lines,
            then: format!("cargo build --release && {setup_cmd}"),
            code_change: true,
        });
    } else if module && a.init_installed() == Some(false) {
        // Allocator in place but nothing turns tracking on: for a program that's
        // `fn main()`'s job, for a module it's the init hook — and we can see
        // from the source that it's missing.
        let mut lines = vec!["In your module's init hook:".to_string()];
        lines.extend(init_hook_lines());
        lines.push("   (env-driven: no MEMSCOPE_* vars set = no tracking, no cost)".to_string());
        lines.push(napi_note.to_string());
        steps.push(Step {
            title: "call memscope::init() from the module's init hook",
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
            title: if module { "build the module" } else { "build the binary" },
            lines: if module {
                vec!["cargo build --release".to_string(), napi_note.to_string()]
            } else {
                vec!["cargo build --release".to_string()]
            },
            then: setup_cmd.to_string(),
            code_change: false,
        });
    }

    // Signing only matters on the injection path (no in-process agent), which a
    // module never takes.
    if !installed && !live && !module && custom.is_none() {
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

/// The `memscope::init()` call, shown with the init hooks of the three ways
/// people build Rust modules for Node. One of these is always the right line;
/// which one depends on the binding crate, which we don't try to guess.
fn init_hook_lines() -> Vec<String> {
    vec![
        "     #[napi::module_init]".to_string(),
        "     fn memscope_init() { memscope::init(); }".to_string(),
        "   (neon: call memscope::init() first in your #[neon::main] fn;".to_string(),
        "    raw N-API: first line of napi_register_module_v1)".to_string(),
    ]
}

/// Newest module artifact for a cdylib project: a `.node` at the project root
/// (where napi-rs leaves its copy, and what node actually loads) or the cdylib
/// under `target/`. Returns the profile only when the artifact came from
/// `target/<profile>/` — a root `.node` doesn't say which profile built it.
///
/// `pkg` filters the `target/` search to *this* package's cdylib: a workspace
/// target dir is full of other members' `.dylib`s (memscope's own preload shim,
/// for one), and "newest" alone would pick the wrong one.
fn newest_module(dir: &Path, pkg: Option<&str>) -> Option<(PathBuf, Option<&'static str>)> {
    let mut best: Option<(PathBuf, Option<&'static str>, std::time::SystemTime)> = None;
    let mut consider = |path: PathBuf, profile: Option<&'static str>| {
        let Ok(meta) = std::fs::metadata(&path) else { return };
        if !meta.is_file() {
            return;
        }
        let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if best.as_ref().map(|(_, _, t)| mtime > *t).unwrap_or(true) {
            best = Some((path, profile, mtime));
        }
    };

    // Root-level `.node` first (it's what the host loads, so it's what must be
    // checked — its dSYM has to sit next to *it*, not next to the cdylib). Any
    // name goes here: the project root is unambiguously this package's.
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "node") {
                consider(path, None);
            }
        }
    }
    // cargo names a cdylib `lib<crate>.dylib`/`.so` (crate name = package name
    // with `-` → `_`); a hand-copied `.node` sits next to it.
    let stems: Vec<String> = pkg
        .map(|p| {
            let c = p.replace('-', "_");
            vec![c.clone(), format!("lib{c}")]
        })
        .unwrap_or_default();
    for target in target_dirs(dir) {
        for profile in ["release", "debug"] {
            let d = target.join(profile);
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for entry in rd.flatten() {
                let path = entry.path();
                let is_module = path
                    .extension()
                    .map(|e| e.to_string_lossy().to_ascii_lowercase())
                    .is_some_and(|e| matches!(&*e, "node" | "dylib" | "so"));
                if !is_module {
                    continue;
                }
                if !stems.is_empty() {
                    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
                    if !stem.is_some_and(|s| stems.contains(&s)) {
                        continue;
                    }
                }
                consider(path, Some(profile));
            }
        }
    }
    best.map(|(p, profile, _)| (p, profile))
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
    if a.is_module() {
        // The host is what you launch; the module records itself.
        return "run your host with the module loaded:  MEMSCOPE_RECORD=rec.mscope node your-app.js\n\
                then:  memscope analyze rec.mscope   (live view: MEMSCOPE_LIVE=1 + memscope monitor)"
            .to_string();
    }
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
    if a.is_module() {
        println!(
            "  kind          dynamic module (a host process loads it) — in-process capture, not injection"
        );
        if let (Some(_), Some(profile)) = (&a.dir, a.bin_profile) {
            println!("  module        newest {profile} build in target/");
        }
    } else if let (Some(_), Some(profile)) = (&a.dir, a.bin_profile) {
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
        // Signing is only about DYLD_INSERT_LIBRARIES into an executable; it says
        // nothing useful about a module, so don't print a line that invites the
        // wrong conclusion.
        if !a.is_module() {
            match &f.injection {
                Injection::Ok { detail } => println!("  signing       {detail}  ✓ injectable"),
                Injection::Blocked { reason, .. } => println!("  signing       ✗ {reason}"),
            }
        }
        if a.is_module() {
            match a.init_installed() {
                Some(true) => println!("  init hook     memscope::init() present  ✓"),
                Some(false) => println!("  init hook     no memscope::init() call  ✗ nothing turns tracking on"),
                // Binary-only target: the call can't be seen from symbols.
                None => {}
            }
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
        let target = target_arg(args);
        println!(
            "  → {n} step{} needed — walk through {}:  memscope setup{}{}",
            if n == 1 { "" } else { "s" },
            if n == 1 { "it" } else { "them" },
            if target == "." { String::new() } else { format!(" {target}") },
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

    print_next_step(&steps, live, None);
    Ok(())
}

/// The one-step-at-a-time view shared by `setup` and `run`'s preflight.
/// `then_override` replaces the step's own "then:" line (the run flow
/// auto-detects completion, so it shouldn't say "re-run memscope setup").
fn print_next_step(steps: &[Step], live: bool, then_override: Option<&str>) {
    let step = &steps[0];
    println!("memscope setup — step 1 of {}: {}", steps.len(), step.title);
    println!();
    for l in &step.lines {
        println!("  {l}");
    }
    if step.code_change {
        println!();
        println!(
            "  or, from your project dir, hand it to an agent:  claude \"$(memscope agent-setup{})\"",
            if live { " --live" } else { "" }
        );
    }
    println!();
    println!("  then:  {}", then_override.unwrap_or(&step.then));
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
    eprintln!("[memscope] pass the text above to an agent, e.g.:  claude \"$(memscope agent-setup)\"");
    Ok(())
}

/// Preflight for `memscope run`: refuse the two cases where the run would
/// silently produce a useless dump. Short messages; `setup` has the details.
/// How `memscope run` should drive the target.
pub enum RunMode {
    /// Inject the preload shim (`DYLD_INSERT_LIBRARIES` / `LD_PRELOAD`).
    Inject,
    /// The binary has memscope built in — same env contract, no injection
    /// (injecting would double-track every allocation).
    Integrated,
}

pub fn preflight_run(bin: &Path, force: bool, wait: bool) -> Result<RunMode, String> {
    // `memscope run -- node app.js` hands us a bare command name, so resolve it
    // through PATH exactly as the spawn will. Skipping that meant every check
    // below was silently bypassed for anything not typed as a path. A name we
    // can't resolve is left alone: let the exec report "not found".
    let resolved = match resolve_target(bin) {
        Some(p) => p,
        None => return Ok(RunMode::Inject),
    };
    let bin = resolved.as_path();
    // Never fail the run because inspection itself failed.
    let facts = match inspect_binary(bin, false) {
        Ok(f) => f,
        Err(_) => return Ok(RunMode::Inject),
    };

    // A module is not a program. You launch the *host* and let the module record
    // itself in-process; there is nothing here to exec.
    if facts.kind == BinaryKind::DynamicModule {
        return Err(format!(
            "{} is a dynamic module (a host process loads it), not a program to run.\n\
             Capture it in-process instead — 2 lines in the module, then run your host:\n  \
             memscope setup {}",
            bin.display(),
            bin.display()
        ));
    }

    // A non-Rust host that loads native modules: injecting it is the trap that
    // looks like it worked. The preload records against the *host* binary, so
    // the addon's frames symbolicate against a binary that has no Rust DWARF and
    // every one of them comes back `[unknown]` — plus you trace the host's own
    // allocator churn (all of V8) instead of the code you care about.
    if !facts.is_rust {
        if let Some(host) = module_host_name(bin) {
            let msg = format!(
                "{} is {host}, not a Rust program. If the Rust code you want is a module it loads\n\
                 (a .node addon, a native extension), injection records against {host}'s own binary —\n\
                 which has no Rust DWARF — so every module frame would come back [unknown].\n\
                 Instrument the module instead (it then records itself, whoever launches it):\n  \
                 memscope check path/to/your-module.node\n\
                 (--force injects anyway.)",
                bin.display()
            );
            if force {
                eprintln!("[memscope] WARNING (--force): {msg}");
            } else {
                return Err(msg);
            }
        }
    }

    if facts.memscope_linked {
        eprintln!(
            "[memscope] {} has memscope built in — running with the in-process agent (no injection).",
            bin.display()
        );
        return Ok(RunMode::Integrated);
    }

    if let Allocator::Named { name, .. } = &facts.allocator {
        if force {
            eprintln!(
                "[memscope] WARNING (--force): {} uses {name} — injection interposes malloc, \
                 so the dump will be (almost) empty.",
                bin.display()
            );
        } else {
            // Don't just refuse — explain why, start the setup process on the
            // spot, and then wait: the moment the rebuilt binary passes, run.
            println!(
                "{} uses {name} as its global allocator, which bypasses malloc — so `memscope run`\n\
                 (malloc interposition) would see (almost) nothing. This needs the 2-line setup:\n",
                bin.display()
            );
            let target = project_dir_of(bin)
                .map(|d| d.display().to_string())
                .unwrap_or_else(|| bin.display().to_string());
            let a = assess(&target)?;
            let steps = plan(&a, false);
            if steps.is_empty() {
                println!("✓ setup is already done — {}", ready_line(&a));
            } else {
                print_next_step(
                    &steps,
                    false,
                    Some("cargo build --release   (I'll detect the rebuild and launch automatically)"),
                );
            }
            return wait_for_setup(bin, wait);
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

    if let DebugInfo::Absent { .. } = &facts.debug_info {
        eprintln!("[memscope] warning: no debug info — the dump will be untyped (memscope setup to fix).");
    }
    Ok(RunMode::Inject)
}

/// The file `memscope run -- <target>` will actually execute: the path itself if
/// it names one, otherwise the first executable of that name on `PATH`.
fn resolve_target(bin: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    // Anything with a separator is a path, not a name to look up.
    if bin.components().count() > 1 {
        return bin.is_file().then(|| bin.to_path_buf());
    }
    if bin.is_file() {
        return Some(bin.to_path_buf());
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(bin)).find(|c| {
        std::fs::metadata(c)
            .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    })
}

/// Runtimes whose whole point is loading native modules. Only used to tell a
/// user "the Rust you want is probably in an addon, not in this binary" — a
/// non-Rust binary that *isn't* one of these (say a C program) is still a
/// perfectly good `run` target, just an untyped one.
fn module_host_name(bin: &Path) -> Option<&'static str> {
    let name = bin.file_name()?.to_string_lossy().to_ascii_lowercase();
    let stem = name.trim_end_matches(".exe");
    // python3.12, node, electron helper names, …
    const HOSTS: &[&str] = &["node", "deno", "bun", "electron", "python", "ruby", "php", "java"];
    HOSTS.iter().find(|h| stem == **h || stem.starts_with(*h)).copied()
}

/// Block until the binary passes preflight, then run it. Re-inspects every 2s
/// (a rebuild shows up automatically); Enter forces a check now and reports
/// what's still missing; Ctrl-C gives up. Skipped when stdin isn't a terminal
/// (scripts get the immediate error instead of a silent hang) unless `--wait`
/// asked for it explicitly.
fn wait_for_setup(bin: &Path, wait_flag: bool) -> Result<RunMode, String> {
    // SAFETY: plain isatty query.
    if !wait_flag && unsafe { libc::isatty(0) } != 1 {
        return Err("not launching (setup needed first; re-run when done, or --wait to auto-launch after the rebuild, or --force)".into());
    }
    eprintln!();
    eprintln!(
        "[memscope] waiting for setup — re-checking every 2s; Enter = check now, Ctrl-C = give up"
    );

    // Enter-to-check-now: a reader thread turns stdin lines into nudges.
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            match std::io::BufRead::read_line(&mut stdin.lock(), &mut line) {
                Ok(0) | Err(_) => break, // EOF — polling continues without us
                Ok(_) => {
                    if tx.send(()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    loop {
        let nudged = match rx.recv_timeout(std::time::Duration::from_secs(2)) {
            Ok(()) => true,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => false,
            // Reader gone (stdin EOF): keep polling on our own clock.
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                std::thread::sleep(std::time::Duration::from_secs(2));
                false
            }
        };
        // Mid-rebuild the binary may be missing or half-written — not ready yet.
        let facts = match inspect_binary(bin, false) {
            Ok(f) => f,
            Err(_) => continue,
        };
        if facts.memscope_linked {
            eprintln!("[memscope] ✓ rebuilt binary has memscope — launching");
            return Ok(RunMode::Integrated);
        }
        if matches!(facts.allocator, Allocator::SystemDefault) {
            eprintln!("[memscope] ✓ custom allocator is gone — launching with injection");
            return Ok(RunMode::Inject);
        }
        if nudged {
            let name = match &facts.allocator {
                Allocator::Named { name, .. } => name,
                Allocator::SystemDefault => unreachable!(),
            };
            eprintln!(
                "[memscope] not ready yet — {} still uses {name} and doesn't link memscope \
                 (did `cargo build --release` finish?)",
                bin.display()
            );
        }
    }
}

/// The cargo project a binary came from: nearest ancestor with a Cargo.toml
/// (e.g. `<project>/target/release/app` → `<project>`).
fn project_dir_of(bin: &Path) -> Option<PathBuf> {
    let abs = std::fs::canonicalize(bin).ok()?;
    let mut dir = abs.parent()?;
    for _ in 0..8 {
        if dir.join("Cargo.toml").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
    None
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

/// Where cargo may have put this project's artifacts, nearest first.
///
/// A workspace **member** has no `target/` of its own — everything lands in the
/// workspace root's. Without walking up, `check crates/my-addon` reports "none
/// built yet" for a module that's sitting right there, built.
fn target_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(explicit) = std::env::var_os("CARGO_TARGET_DIR") {
        let p = PathBuf::from(explicit);
        if p.is_dir() {
            dirs.push(p);
        }
    }
    let mut at = Some(dir);
    // Bounded walk: deep enough for `workspace/crates/member`, shallow enough
    // that we never wander into an unrelated project above the one asked about.
    for _ in 0..6 {
        let Some(d) = at else { break };
        let t = d.join("target");
        if t.is_dir() && !dirs.contains(&t) {
            dirs.push(t);
        }
        at = d.parent();
    }
    dirs
}

/// The enclosing workspace root's `Cargo.toml` text, if `dir` is a member.
/// Found by walking up to the nearest manifest that declares `[workspace]`.
fn workspace_manifest(dir: &Path) -> Option<String> {
    let mut at = dir.parent();
    for _ in 0..6 {
        let d = at?;
        if let Ok(text) = std::fs::read_to_string(d.join("Cargo.toml")) {
            if text.lines().any(|l| l.trim() == "[workspace]") {
                return Some(text);
            }
        }
        at = d.parent();
    }
    None
}

/// The `name = "…"` of the `[package]` section.
fn package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_package = t == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = t.strip_prefix("name") {
                let v = rest.trim_start().strip_prefix('=')?.trim();
                return Some(v.trim_matches('"').to_string());
            }
        }
    }
    None
}

/// Executable names this package can produce: `[[bin]] name = …`, every
/// `src/bin/*.rs`, and the package name itself when there's a `src/main.rs`.
///
/// Needed because a workspace target dir holds *every* member's output; without
/// filtering, `check` on one member would happily inspect another's binary.
fn bin_names(dir: &Path, manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_bin = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_bin = t == "[[bin]]";
            continue;
        }
        if in_bin {
            if let Some(rest) = t.strip_prefix("name") {
                if let Some(v) = rest.trim_start().strip_prefix('=') {
                    names.push(v.trim().trim_matches('"').to_string());
                }
            }
        }
    }
    if let Ok(rd) = std::fs::read_dir(dir.join("src").join("bin")) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "rs") {
                if let Some(stem) = p.file_stem() {
                    names.push(stem.to_string_lossy().into_owned());
                }
            }
        }
    }
    if dir.join("src").join("main.rs").is_file() {
        if let Some(pkg) = package_name(manifest) {
            names.push(pkg);
        }
    }
    names
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

/// Newest executable this package built, in any of its `target/{release,debug}`
/// dirs (top level only — `deps/` holds hashed intermediates).
///
/// `names` (from [`bin_names`]) keeps a workspace member from picking up a
/// sibling's binary out of the shared target dir. An empty `names` means "we
/// couldn't tell" and falls back to any executable, which is the old behavior.
fn newest_binary(dir: &Path, names: &[String]) -> Option<(PathBuf, &'static str)> {
    use std::os::unix::fs::PermissionsExt;
    let mut best: Option<(PathBuf, &'static str, std::time::SystemTime)> = None;
    for target in target_dirs(dir) {
        for profile in ["release", "debug"] {
            let d = target.join(profile);
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
                if !names.is_empty() {
                    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
                    let ours = stem.is_some_and(|s| {
                        names.iter().any(|n| n == &s || n.replace('-', "_") == s)
                    });
                    if !ours {
                        continue;
                    }
                }
                let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                if best.as_ref().map(|(_, _, t)| mtime > *t).unwrap_or(true) {
                    best = Some((path, profile, mtime));
                }
            }
        }
        // Nearest target dir that has one of our binaries wins; don't let a
        // stale copy further up shadow it.
        if best.is_some() {
            break;
        }
    }
    best.map(|(p, profile, _)| (p, profile))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_and_bin_names_come_off_the_manifest() {
        let manifest = "\
[package]\n\
name = \"my-addon\"\n\
version = \"0.1.0\"\n\
\n\
[lib]\n\
crate-type = [\"cdylib\"]\n\
\n\
[[bin]]\n\
name = \"helper\"\n\
path = \"src/helper.rs\"\n";
        assert_eq!(package_name(manifest).as_deref(), Some("my-addon"));
        assert_eq!(bin_names(Path::new("/nonexistent"), manifest), vec!["helper".to_string()]);
        // `name` under [[bin]] must not be mistaken for the package name.
        assert!(!bin_names(Path::new("/nonexistent"), manifest).contains(&"my-addon".to_string()));
    }

    #[test]
    fn profile_debug_scan() {
        assert!(profile_has_debug("[profile.release]\ndebug = true\n", "release"));
        assert!(profile_has_debug("[profile.release]\ndebug = 2\n", "release"));
        assert!(!profile_has_debug("[profile.release]\ndebug = false\n", "release"));
        // A setting in a *different* profile doesn't count.
        assert!(!profile_has_debug("[profile.dev]\ndebug = true\n", "release"));
        assert!(!profile_has_debug("[package]\nname = \"x\"\n", "release"));
    }

    #[test]
    fn module_hosts_are_recognized_by_name() {
        assert_eq!(module_host_name(Path::new("/opt/homebrew/bin/node")), Some("node"));
        assert_eq!(module_host_name(Path::new("/usr/bin/python3.12")), Some("python"));
        assert_eq!(module_host_name(Path::new("/usr/local/bin/bun")), Some("bun"));
        // A non-Rust binary that isn't a module host stays a valid `run` target.
        assert_eq!(module_host_name(Path::new("/bin/ls")), None);
        assert_eq!(module_host_name(Path::new("./my-c-program")), None);
    }
}
