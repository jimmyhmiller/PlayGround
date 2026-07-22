//! Binary fact-finding: everything `memscope check` (and `memscope run`'s
//! preflight) needs to know about a target binary *before* any tracking
//! happens — which global allocator it uses, whether DWARF type recovery will
//! work, and whether load-time injection is even possible on this platform.
//!
//! All of it is read straight off the binary (symbol table, sections, code
//! signature); nothing here executes the target.

use std::path::{Path, PathBuf};

use object::{Object, ObjectSection, ObjectSymbol};

use crate::load;

type DynErr = Box<dyn std::error::Error + Send + Sync>;

/// Which global allocator the binary's heap goes through. This decides the
/// single most important question: does `malloc` interposition (`memscope run`)
/// see the program's allocations at all?
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Allocator {
    /// No known alternative allocator linked in — the Rust default (`System`,
    /// i.e. libc `malloc`) is in effect. Injection sees everything.
    SystemDefault,
    /// A known allocator library is linked in (almost certainly the
    /// `#[global_allocator]`). Its allocations bypass `malloc`, so injection
    /// is blind — the program needs the 2-line `MemScope::new(inner)` wrap.
    Named {
        /// Display name, e.g. "mimalloc".
        name: &'static str,
        /// The Rust type to pass to `MemScope::new(...)`, e.g. "mimalloc::MiMalloc".
        wrap_type: &'static str,
    },
}

/// Whether DWARF for this binary can be located (embedded, or a `.dSYM`).
#[derive(Debug, Clone)]
pub enum DebugInfo {
    /// DWARF located — allocation sites will resolve to concrete types.
    Present { source: String },
    /// No DWARF anywhere we know to look — dumps will be complete but untyped.
    Absent { detail: String },
    /// Could not cheaply tell (macOS quick probe with no `.dSYM` yet:
    /// `dsymutil` may still be able to produce one from the debug map).
    Unknown { detail: String },
}

/// Whether load-time injection (`DYLD_INSERT_LIBRARIES` / `LD_PRELOAD`) will
/// be honored for this binary.
#[derive(Debug, Clone)]
pub enum Injection {
    Ok { detail: String },
    Blocked { reason: String, fix: String },
}

/// Everything we can tell about a binary without running it.
#[derive(Debug, Clone)]
pub struct BinaryFacts {
    pub path: PathBuf,
    /// Rust mangled symbols present (affects wording only — injection works on
    /// C programs too).
    pub is_rust: bool,
    /// memscope's tracking allocator (`memscope-core`) is compiled in.
    pub memscope_linked: bool,
    pub allocator: Allocator,
    pub debug_info: DebugInfo,
    pub injection: Injection,
}

/// Inspect `path`. With `generate_dsym` (the thorough `check` mode) a missing
/// macOS `.dSYM` is generated on the spot via `dsymutil`, so the debug-info
/// answer is definitive; without it (cheap `run` preflight) the macOS answer
/// may be [`DebugInfo::Unknown`].
pub fn inspect_binary(path: &Path, generate_dsym: bool) -> Result<BinaryFacts, DynErr> {
    let file = std::fs::File::open(path)
        .map_err(|e| -> DynErr { format!("cannot open {}: {e}", path.display()).into() })?;
    // SAFETY: read-only map of an on-disk build artifact; standard mmap caveat.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let obj = object::File::parse(&*mmap)
        .map_err(|e| -> DynErr { format!("{} is not a recognized binary: {e}", path.display()).into() })?;

    let mut is_rust = false;
    let mut memscope_linked = false;
    let mut allocator = Allocator::SystemDefault;
    for sym in obj.symbols().chain(obj.dynamic_symbols()) {
        let Ok(raw) = sym.name() else { continue };
        // Mach-O prepends one `_` to every C-level name; strip it so the same
        // matches work on ELF and Mach-O. (Rust markers use `contains`, which
        // is insensitive to the extra underscore.)
        let n = raw.strip_prefix('_').unwrap_or(raw);

        if !is_rust
            && (raw.contains("_ZN4core") || raw.contains("_ZN3std") || raw.contains("__rust_alloc")
                || raw.contains("rust_begin_unwind") || raw.contains("_RNv"))
        {
            is_rust = true;
        }
        // `memscope_core` (not just "memscope"): the CLI links memscope-replay /
        // -symbols without the tracking allocator, and must not read as "integrated".
        if !memscope_linked && raw.contains("memscope_core") {
            memscope_linked = true;
        }
        if allocator == Allocator::SystemDefault {
            allocator = match n {
                "mi_malloc" | "mi_free" => Allocator::Named { name: "mimalloc", wrap_type: "mimalloc::MiMalloc" },
                "je_malloc" | "je_mallocx" | "mallocx" => {
                    Allocator::Named { name: "jemalloc", wrap_type: "tikv_jemallocator::Jemalloc" }
                }
                "tc_malloc" => Allocator::Named { name: "tcmalloc", wrap_type: "tcmalloc::TCMalloc" },
                "sn_malloc" | "sn_free" => Allocator::Named { name: "snmalloc", wrap_type: "snmalloc::SnMalloc" },
                _ if n.contains("rjem_malloc") => {
                    Allocator::Named { name: "jemalloc (tikv)", wrap_type: "tikv_jemallocator::Jemalloc" }
                }
                _ => Allocator::SystemDefault,
            };
        }
    }

    let debug_info = probe_debug_info(path, &obj, generate_dsym);
    let injection = probe_injection(path);

    Ok(BinaryFacts { path: path.to_path_buf(), is_rust, memscope_linked, allocator, debug_info, injection })
}

fn probe_debug_info(path: &Path, obj: &object::File, generate_dsym: bool) -> DebugInfo {
    // Embedded DWARF (the normal Linux shape; also Mach-O object-style embeds).
    let embedded = obj
        .sections()
        .filter_map(|s| s.name().ok().map(|n| (n.to_string(), s.size())))
        .find(|(n, size)| n.ends_with("debug_info") && *size > 0);
    if let Some((_, size)) = embedded {
        return DebugInfo::Present { source: format!("embedded DWARF ({})", human_bytes(size)) };
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(dsym) = load::dsym_dwarf_path(path) {
            if dsym.exists() && !load::is_stale(&dsym, path) {
                return match dwarf_nonempty(&dsym) {
                    true => DebugInfo::Present { source: "fresh .dSYM next to the binary".into() },
                    false => DebugInfo::Absent {
                        detail: "the .dSYM has no DWARF — build with `debug = true`".into(),
                    },
                };
            }
        }
        if generate_dsym {
            return match load::find_or_make_dsym(path) {
                // dsymutil exits 0 even for a binary with no debug info at all —
                // it just writes an EMPTY dSYM. Presence isn't proof; look inside.
                Ok(p) if dwarf_nonempty(&p) => DebugInfo::Present { source: ".dSYM generated".into() },
                Ok(_) => DebugInfo::Absent {
                    detail: "binary has no debug info (dsymutil produced an empty .dSYM) — build with `debug = true`".into(),
                },
                Err(e) => DebugInfo::Absent { detail: format!("dsymutil could not produce DWARF: {e}") },
            };
        }
        return DebugInfo::Unknown {
            detail: "no .dSYM yet; one is generated automatically at first dump (run `memscope check` to verify now)".into(),
        };
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (path, generate_dsym);
        DebugInfo::Absent {
            detail: "no .debug_info section — build with `debug = true` in the used [profile.*]".into(),
        }
    }
}

/// Does this DWARF-bearing file actually contain a non-empty `.debug_info`?
#[cfg(target_os = "macos")]
fn dwarf_nonempty(path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else { return false };
    // SAFETY: read-only map of an on-disk build artifact.
    let Ok(mmap) = (unsafe { memmap2::Mmap::map(&file) }) else { return false };
    let Ok(obj) = object::File::parse(&*mmap) else { return false };
    obj.sections()
        .any(|s| s.name().map(|n| n.ends_with("debug_info")).unwrap_or(false) && s.size() > 0)
}

/// On macOS, `DYLD_INSERT_LIBRARIES` is silently ignored for setuid binaries
/// and for signatures with the hardened runtime / `restrict` flag (unless the
/// binary carries the allow-dyld-environment-variables entitlement). Detecting
/// that up front turns "the dump was mysteriously empty" into a clear message.
#[cfg(target_os = "macos")]
fn probe_injection(path: &Path) -> Injection {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.permissions().mode() & 0o6000 != 0 {
            return Injection::Blocked {
                reason: "binary is setuid/setgid — dyld ignores DYLD_INSERT_LIBRARIES".into(),
                fix: "run a non-setuid copy of the binary".into(),
            };
        }
    }
    let out = match std::process::Command::new("codesign").args(["-d", "--verbose=2"]).arg(path).output() {
        Ok(o) => o,
        // codesign missing: nothing to check; dev binaries are fine.
        Err(_) => return Injection::Ok { detail: "codesign not available; assuming injectable".into() },
    };
    let text = String::from_utf8_lossy(&out.stderr).into_owned();
    if text.contains("not signed at all") {
        return Injection::Ok { detail: "unsigned".into() };
    }
    let flags_line = text.lines().find(|l| l.contains("flags=")).unwrap_or("");
    let hardened = flags_line.contains("runtime");
    let restricted = flags_line.contains("restrict") || flags_line.contains("library-validation");
    if hardened || restricted {
        // The entitlement re-allows dyld env vars even under the hardened runtime.
        let ent = std::process::Command::new("codesign")
            .args(["-d", "--entitlements", "-"])
            .arg(path)
            .output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr)
            })
            .unwrap_or_default();
        if ent.contains("com.apple.security.cs.allow-dyld-environment-variables") {
            return Injection::Ok { detail: "hardened runtime, but allow-dyld-environment-variables entitlement present".into() };
        }
        return Injection::Blocked {
            reason: format!(
                "code signature blocks dyld env vars ({})",
                if hardened { "hardened runtime" } else { "restrict/library-validation flag" }
            ),
            fix: format!("codesign --remove-signature {}   # or rebuild without hardened runtime", path.display()),
        };
    }
    let detail = if flags_line.contains("linker-signed") || flags_line.contains("adhoc") {
        "ad-hoc signature (normal cargo build output)".to_string()
    } else {
        "signed, no hardened runtime".to_string()
    };
    Injection::Ok { detail }
}

#[cfg(not(target_os = "macos"))]
fn probe_injection(path: &Path) -> Injection {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.permissions().mode() & 0o6000 != 0 {
            return Injection::Blocked {
                reason: "binary is setuid/setgid — the loader ignores LD_PRELOAD".into(),
                fix: "run a non-setuid copy of the binary".into(),
            };
        }
    }
    Injection::Ok { detail: "LD_PRELOAD honored for normal binaries".into() }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 { format!("{n} B") } else { format!("{v:.1} {}", UNITS[u]) }
}
