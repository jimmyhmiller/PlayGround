//! Stack-frame classification: demangle, strip stdlib/runtime plumbing, and
//! locate the **boundary frame** — the first application frame entered from the
//! runtime — which is what tells you *where* an allocation came from.
//!
//! Shared by the flamegraph `--no-std` view, `diff`/`analyze` site locations, and
//! the profiler-noise filter (allocations made *inside* memscope's own DWARF
//! symbolication / recording machinery, which aren't part of the program under
//! study).

use crate::FrameMeta;

/// Demangle a recorded frame name: drop the `::h<hash>` suffix and `[<hash>]`
/// crate-disambiguators rustc emits.
pub fn clean_frame(f: &str) -> String {
    let mut s = f;
    if let Some(idx) = s.rfind("::h") {
        if s[idx + 3..].len() >= 8 && s[idx + 3..].bytes().all(|b| b.is_ascii_hexdigit()) {
            s = &s[..idx];
        }
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' {
            let mut inner = String::new();
            while let Some(&n) = chars.peek() {
                if n == ']' {
                    chars.next();
                    break;
                }
                inner.push(n);
                chars.next();
            }
            if !inner.chars().all(|c| c.is_ascii_hexdigit()) {
                out.push('[');
                out.push_str(&inner);
                out.push(']');
            }
        } else {
            out.push(c);
        }
    }
    if out.is_empty() {
        return "[unknown]".to_string();
    }
    out
}

/// True for stdlib/runtime plumbing frames (std/core/alloc, hashbrown/allocator
/// shim, lang-start, pthread/libc, the `Fn` shims). Expects a demangled name.
pub fn is_std_frame(name: &str) -> bool {
    let n = name.trim_start_matches(['<', '&', ' ']);
    const PREFIXES: &[&str] = &[
        "std::", "core::", "alloc::", "proc_macro::", "test::", "backtrace::",
        "hashbrown::", "allocator_api2::",
        "__rust", "_rust", "rustc", "__pthread", "_pthread", "pthread", "__libc",
        "libc::", "_main", "dyld", "start_",
    ];
    if PREFIXES.iter().any(|p| n.starts_with(p)) {
        return true;
    }
    name.contains(" as std::")
        || name.contains(" as core::")
        || name.contains(" as alloc::")
        || name.contains("rust_begin_short_backtrace")
        || name.contains("lang_start")
        || name.contains("catch_unwind")
        || name.contains("::panicking::")
        || name.contains("ops::function::Fn")
        || name.contains("call_once")
        || name == "_main"
        || name == "main"
        || name == "start"
        || name == "[unknown]"
}

/// True for frames inside memscope's own machinery or the DWARF/symbolication
/// stack it pulls in — allocations whose *innermost* such frame is one of these
/// are profiler overhead, not part of the program under study. Expects a
/// demangled name.
pub fn is_profiler_frame(name: &str) -> bool {
    let n = name.trim_start_matches(['<', '&', ' ']);
    const PREFIXES: &[&str] = &[
        "memscope", "gimli::", "addr2line::", "object::", "rustc_demangle::",
    ];
    PREFIXES.iter().any(|p| n.starts_with(p)) || n.contains("memscope")
}

/// The boundary frame: the first non-runtime frame (frames are recorded
/// innermost-first), i.e. the application code closest to the allocation.
/// `None` when every frame is stdlib/runtime.
pub fn boundary_frame(frames: &[FrameMeta]) -> Option<&FrameMeta> {
    frames.iter().find(|f| !is_std_frame(&clean_frame(&f.func)))
}

/// Format a frame as `func (file:line)` (or just `func` when no source info).
pub fn frame_location(f: &FrameMeta) -> String {
    if f.file.is_empty() {
        clean_frame(&f.func)
    } else {
        format!("{} ({}:{})", clean_frame(&f.func), f.file, f.line)
    }
}

/// True when this site's allocation originates *inside* the profiler/symbolication
/// machinery — its boundary (innermost application-or-library) frame is a
/// memscope/DWARF frame. Such sites are excluded from `analyze` findings.
pub fn is_profiler_origin(frames: &[FrameMeta]) -> bool {
    match boundary_frame(frames) {
        Some(f) => is_profiler_frame(&clean_frame(&f.func)),
        None => false,
    }
}
