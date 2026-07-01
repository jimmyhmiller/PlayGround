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

/// The ASLR load slide (runtime address − link-time address) of the image that
/// contains memscope's own code. Captured once at record time and written into
/// the recording header so a reader can map a recorded return address back to a
/// static address for symbolication: `static = ip - slide`.
///
/// Crucially this resolves the image *memscope is compiled into* — which may be
/// the main executable OR a loaded dylib (e.g. a Node native addon / an injected
/// `.node`). Using `dladdr` on one of our own functions finds the right image;
/// using image 0 unconditionally would be wrong inside a dylib.
#[cfg(target_os = "macos")]
pub fn current_image_slide() -> u64 {
    self_image().map(|(slide, _)| slide).unwrap_or(0)
}

/// Path to the image (executable or dylib) that contains memscope's code. The
/// recorder writes this into the header instead of `current_exe()` so the reader
/// symbolicates against the right binary even when memscope lives in a dylib.
#[cfg(target_os = "macos")]
pub fn current_image_path() -> Option<std::path::PathBuf> {
    self_image().map(|(_, path)| path)
}

#[cfg(not(target_os = "macos"))]
pub fn current_image_path() -> Option<std::path::PathBuf> {
    None
}

/// Find the (slide, path) of the Mach-O image containing this function, via
/// `dladdr` (path + runtime base) joined to the matching dyld image's slide.
#[cfg(target_os = "macos")]
fn self_image() -> Option<(u64, std::path::PathBuf)> {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_void};

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }
    extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_vmaddr_slide(image_index: u32) -> isize;
        fn _dyld_get_image_name(image_index: u32) -> *const c_char;
    }

    // Probe with the address of this very function to land in memscope's image.
    let probe = self_image as *const c_void;
    let mut info: DlInfo = unsafe { std::mem::zeroed() };
    // SAFETY: dladdr fills `info` for a valid code address.
    if unsafe { dladdr(probe, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    let path = unsafe { CStr::from_ptr(info.dli_fname) }
        .to_string_lossy()
        .into_owned();

    // The slide comes from the dyld image whose name matches our path; that
    // accounts for a non-zero link-time base (dladdr's fbase alone would not).
    // SAFETY: plain libdyld queries over a valid index range.
    let slide = unsafe {
        let count = _dyld_image_count();
        let mut found = None;
        for i in 0..count {
            let name = _dyld_get_image_name(i);
            if !name.is_null() && CStr::from_ptr(name).to_string_lossy() == path {
                found = Some(_dyld_get_image_vmaddr_slide(i) as u64);
                break;
            }
        }
        // Fall back to the runtime base if the image table lookup misses.
        found.unwrap_or(info.dli_fbase as u64)
    };

    Some((slide, std::path::PathBuf::from(path)))
}

/// On ELF the recording stores raw runtime addresses and the reader resolves
/// against the (position-independent) executable; the slide is folded in there,
/// so record 0 here.
#[cfg(not(target_os = "macos"))]
pub fn current_image_slide() -> u64 {
    0
}

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

/// Like [`dwarf_bytes_for`], but **memory-maps** the DWARF-bearing file instead
/// of reading it into a `Vec`. The OS pages the (often ~1 GB) debug sections in
/// and out on demand, so they don't sit in resident heap — a prerequisite for
/// constant-memory symbolication. The returned `Mmap` derefs to `&[u8]`.
pub fn dwarf_mmap_for(exe: &Path) -> Result<memmap2::Mmap, DynErr> {
    #[cfg(target_os = "macos")]
    let path = find_or_make_dsym(exe)?;
    #[cfg(not(target_os = "macos"))]
    let path = exe.to_path_buf();

    let file = std::fs::File::open(&path)?;
    // SAFETY: the dSYM/binary is a stable on-disk file we only read; we accept
    // the standard mmap caveat that external truncation is UB (not a concern for
    // a build artifact we just located).
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(mmap)
}

/// True if `dsym` is older than `exe` (so it predates the current build).
#[cfg(target_os = "macos")]
fn is_stale(dsym: &Path, exe: &Path) -> bool {
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    match (mtime(dsym), mtime(exe)) {
        (Some(d), Some(e)) => d < e,
        // If we can't tell, assume stale and regenerate (correctness over speed).
        _ => true,
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
    // Reuse an existing dSYM only if it's at least as new as the binary.
    // Monomorphization hashes change on every rebuild, so a stale dSYM would
    // mismatch the running binary's symbols and silently break type recovery.
    if path.exists() && !is_stale(&path, exe) {
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
