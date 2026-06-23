//! Locating the bytes that actually contain DWARF for the running binary.
//!
//! * **Linux / ELF**: DWARF is embedded in the executable (with `debug = true`),
//!   so we read the executable itself.
//! * **macOS / Mach-O**: DWARF lives in a separate `.dSYM` bundle produced by
//!   `dsymutil`. We look for an existing one next to the binary and, if absent,
//!   generate it (a standard Xcode tool — no toolchain change to the user's
//!   build, just a post-build symbolication step).

use std::error::Error;
use std::path::{Path, PathBuf};

type DynErr = Box<dyn Error + Send + Sync>;

/// DWARF-bearing bytes for the current executable.
pub fn dwarf_bytes_for_current_exe() -> Result<Vec<u8>, DynErr> {
    let exe = std::env::current_exe()?;
    dwarf_bytes_for(&exe)
}

/// DWARF-bearing bytes for an arbitrary executable path.
pub fn dwarf_bytes_for(exe: &Path) -> Result<Vec<u8>, DynErr> {
    #[cfg(target_os = "macos")]
    {
        let dsym = find_or_make_dsym(exe)?;
        Ok(std::fs::read(dsym)?)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(std::fs::read(exe)?)
    }
}

#[cfg(target_os = "macos")]
fn dsym_dwarf_path(exe: &Path) -> Option<PathBuf> {
    let name = exe.file_name()?;
    let parent = exe.parent()?;
    let mut bundle = parent.to_path_buf();
    bundle.push(format!("{}.dSYM", name.to_string_lossy()));
    bundle.push("Contents");
    bundle.push("Resources");
    bundle.push("DWARF");
    bundle.push(name);
    Some(bundle)
}

#[cfg(target_os = "macos")]
fn find_or_make_dsym(exe: &Path) -> Result<PathBuf, DynErr> {
    let path = dsym_dwarf_path(exe)
        .ok_or_else(|| -> DynErr { "could not derive dSYM path from executable".into() })?;
    if path.exists() {
        return Ok(path);
    }
    // Generate it: `dsymutil <exe>` writes `<exe>.dSYM` alongside the binary.
    let status = std::process::Command::new("dsymutil")
        .arg(exe)
        .status()
        .map_err(|e| -> DynErr {
            format!(
                "no dSYM next to {} and failed to run dsymutil ({e}). \
                 Build with debuginfo and ensure dsymutil is on PATH.",
                exe.display()
            )
            .into()
        })?;
    if !status.success() {
        return Err(format!("dsymutil failed for {}", exe.display()).into());
    }
    if !path.exists() {
        return Err(format!(
            "dsymutil ran but no DWARF found at {}. Is the binary built with debuginfo?",
            path.display()
        )
        .into());
    }
    Ok(path)
}
