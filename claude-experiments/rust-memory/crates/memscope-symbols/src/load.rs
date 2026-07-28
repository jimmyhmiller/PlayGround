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

/// The (slide, path) of the **main executable**, regardless of which image this
/// code lives in. This is what the preload shim must record: interposed
/// `malloc` calls come from the *target's* code, so the reader has to
/// symbolicate against the target — recording the shim dylib (what
/// [`current_image_path`] would return there) leaves every target frame
/// unresolvable.
#[cfg(target_os = "macos")]
pub fn main_image() -> Option<(u64, std::path::PathBuf)> {
    use std::ffi::CStr;
    use std::os::raw::c_char;
    extern "C" {
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_vmaddr_slide(image_index: u32) -> isize;
        fn _dyld_get_image_name(image_index: u32) -> *const c_char;
    }
    // Find the executable by *path*, not by image index — with
    // DYLD_INSERT_LIBRARIES in play, index 0 isn't reliably the main image.
    let exe = std::env::current_exe().ok()?;
    let exe_canon = std::fs::canonicalize(&exe).unwrap_or(exe);
    // SAFETY: plain libdyld queries over a valid index range.
    unsafe {
        for i in 0.._dyld_image_count() {
            let name = _dyld_get_image_name(i);
            if name.is_null() {
                continue;
            }
            let p = std::path::PathBuf::from(CStr::from_ptr(name).to_string_lossy().into_owned());
            let canon = std::fs::canonicalize(&p).unwrap_or(p);
            if canon == exe_canon {
                return Some((_dyld_get_image_vmaddr_slide(i) as u64, exe_canon));
            }
        }
    }
    None
}

/// ELF: the executable path with slide 0, matching the [`current_image_slide`]
/// convention (the reader folds the slide in).
#[cfg(not(target_os = "macos"))]
pub fn main_image() -> Option<(u64, std::path::PathBuf)> {
    std::env::current_exe().ok().map(|p| (0, p))
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
pub(crate) fn is_stale(dsym: &Path, exe: &Path) -> bool {
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    match (mtime(dsym), mtime(exe)) {
        (Some(d), Some(e)) => d < e,
        // If we can't tell, assume stale and regenerate (correctness over speed).
        _ => true,
    }
}

/// An exclusive `flock` on `<exe>.dSYM.lock`, held for as long as it lives.
///
/// Advisory and best-effort: if the lock file can't be created (read-only
/// directory) or `flock` fails, we proceed unlocked rather than refuse to
/// symbolicate — the same behavior as before this existed. `flock` is per open
/// file description, so this serializes threads within a process too.
#[cfg(target_os = "macos")]
struct DsymLock(Option<std::fs::File>);

// Declared here rather than pulling `libc` into this crate: memscope-symbols is
// linked into the traced process, and this is two constants and one call.
#[cfg(target_os = "macos")]
const LOCK_EX: std::os::raw::c_int = 2;
#[cfg(target_os = "macos")]
const LOCK_UN: std::os::raw::c_int = 8;
#[cfg(target_os = "macos")]
extern "C" {
    fn flock(fd: std::os::raw::c_int, operation: std::os::raw::c_int) -> std::os::raw::c_int;
}

#[cfg(target_os = "macos")]
impl DsymLock {
    fn acquire(exe: &Path) -> DsymLock {
        let mut path = exe.as_os_str().to_os_string();
        path.push(".dSYM.lock");
        let file = match std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(PathBuf::from(path))
        {
            Ok(f) => f,
            Err(_) => return DsymLock(None),
        };
        use std::os::unix::io::AsRawFd;
        // SAFETY: plain flock on a file descriptor we own; blocks until acquired.
        if unsafe { flock(file.as_raw_fd(), LOCK_EX) } != 0 {
            return DsymLock(None);
        }
        DsymLock(Some(file))
    }
}

#[cfg(target_os = "macos")]
impl Drop for DsymLock {
    fn drop(&mut self) {
        if let Some(f) = &self.0 {
            use std::os::unix::io::AsRawFd;
            // SAFETY: releasing our own lock; the fd is still open here.
            unsafe { flock(f.as_raw_fd(), LOCK_UN) };
        }
    }
}

/// The `<exe>.dSYM` bundle directory itself.
#[cfg(target_os = "macos")]
pub(crate) fn dsym_bundle_path(exe: &Path) -> Option<PathBuf> {
    let name = exe.file_name()?;
    Some(exe.parent()?.join(format!("{}.dSYM", name.to_string_lossy())))
}

#[cfg(target_os = "macos")]
pub(crate) fn dsym_dwarf_path(exe: &Path) -> Option<PathBuf> {
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
pub(crate) fn find_or_make_dsym(exe: &Path) -> Result<PathBuf, DynErr> {
    let path = dsym_dwarf_path(exe)
        .ok_or_else(|| -> DynErr { "could not derive dSYM path from executable".into() })?;
    // Reuse an existing dSYM only if it's at least as new as the binary.
    // Monomorphization hashes change on every rebuild, so a stale dSYM would
    // mismatch the running binary's symbols and silently break type recovery.
    if path.exists() && !is_stale(&path, exe) {
        return Ok(path);
    }

    // Serialize generation across processes. Several processes routinely need the
    // same dSYM at the same time — a Node app whose workers each dump, a reader
    // symbolicating while the target dumps — and concurrent `dsymutil` runs write
    // the same bundle, so one of them reads a half-written `.debug_info` and
    // resolves NOTHING (empty labels, no error). Whoever gets the lock builds it;
    // the others wait and then find it already fresh.
    let _guard = DsymLock::acquire(exe);
    if path.exists() && !is_stale(&path, exe) {
        return Ok(path);
    }
    // Generate to a private bundle, then move it into place, so `<exe>.dSYM`
    // NEVER exists in a half-written state. The fast path above only looks at
    // existence + mtime, so a bundle appearing while dsymutil is still filling it
    // in would be taken as ready — and a reader that mmaps it then resolves
    // nothing at all, with no error to explain why. A rename also leaves any
    // already-mmapped older bundle valid (the mapping holds the old inode).
    let bundle = dsym_bundle_path(exe)
        .ok_or_else(|| -> DynErr { "could not derive dSYM path from executable".into() })?;
    let mut staging = bundle.clone().into_os_string();
    staging.push(format!(".{}.tmp", std::process::id()));
    let staging = PathBuf::from(staging);
    let _ = std::fs::remove_dir_all(&staging);

    // Capture (rather than inherit) dsymutil's output — its "no debug symbols"
    // warning is our Absent verdict, not something to splat on the terminal.
    let status = std::process::Command::new("dsymutil")
        .arg(exe)
        .arg("-o")
        .arg(&staging)
        .output()
        .map(|o| o.status)
        .map_err(|e| -> DynErr {
            format!(
                "no dSYM next to {} and failed to run dsymutil ({e}). \
                 Build with debuginfo and ensure dsymutil is on PATH.",
                exe.display()
            )
            .into()
        })?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(format!("dsymutil failed for {}", exe.display()).into());
    }
    let staged_dwarf = staging
        .join("Contents")
        .join("Resources")
        .join("DWARF")
        .join(exe.file_name().unwrap_or_default());
    if !staged_dwarf.is_file() {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(format!(
            "dsymutil ran but no DWARF found at {}. Is the binary built with debuginfo?",
            path.display()
        )
        .into());
    }
    // Swap it in. We hold the lock, so no other memscope process is mid-generate;
    // a reader that catches the gap sees "no dSYM", takes the lock, waits for us,
    // and then finds the finished one.
    let _ = std::fs::remove_dir_all(&bundle);
    std::fs::rename(&staging, &bundle).map_err(|e| -> DynErr {
        let _ = std::fs::remove_dir_all(&staging);
        format!("could not move the generated dSYM into {}: {e}", bundle.display()).into()
    })?;
    if !path.exists() {
        return Err(format!("dSYM moved into place but {} is missing", path.display()).into());
    }
    Ok(path)
}
