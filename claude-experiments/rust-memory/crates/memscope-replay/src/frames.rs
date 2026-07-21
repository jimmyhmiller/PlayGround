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

/// Normalize a recovered *type* name for display: shorten stdlib paths to the
/// bare type (`alloc::boxed::Box` → `Box`) and drop defaulted allocator/hasher
/// params (`Box<T, Global>` → `Box<T>`), so a site reads `Vec<Box<Particle>>`
/// rather than `Vec<alloc::boxed::Box<serve::Particle, alloc::alloc::Global>>`.
/// User-crate paths are kept — they distinguish types; std paths never do.
///
/// Display-only: raw DWARF names remain the layout/graph join keys.
pub fn clean_type_name(name: &str) -> String {
    const STD_NS: &[&str] = &["alloc", "core", "std", "hashbrown"];
    let b = name.as_bytes();
    let mut out = String::with_capacity(name.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            // Read one path: `ident(::ident)*` (generic args are handled by the
            // outer loop — a path run stops at `<`, `,`, `>` etc.).
            let start = i;
            let mut first_seg_end = start;
            let mut last_seg;
            let mut j = i;
            loop {
                let seg = j;
                while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                    j += 1;
                }
                if seg == start {
                    first_seg_end = j;
                }
                last_seg = seg;
                // Continue only over `::ident`; `::{closure...}` stays literal.
                if j + 2 < b.len()
                    && b[j] == b':'
                    && b[j + 1] == b':'
                    && ((b[j + 2] as char).is_ascii_alphabetic() || b[j + 2] == b'_')
                {
                    j += 2;
                } else {
                    break;
                }
            }
            let first = &name[start..first_seg_end];
            if last_seg > start && STD_NS.contains(&first) {
                out.push_str(&name[last_seg..j]);
            } else {
                out.push_str(&name[start..j]);
            }
            i = j;
        } else {
            out.push(c);
            i += 1;
        }
    }
    // With paths shortened, defaulted generic params are recognizable bare
    // names; drop them (they're never the interesting part of the type).
    for noise in [", Global>", ", RandomState>", ", DefaultHashBuilder>"] {
        while let Some(p) = out.find(noise) {
            out.replace_range(p..p + noise.len(), ">");
        }
    }
    out
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

#[cfg(test)]
mod tests {
    use super::clean_type_name;

    #[test]
    fn shortens_std_paths_and_drops_default_params() {
        assert_eq!(
            clean_type_name("alloc::boxed::Box<serve::Particle, alloc::alloc::Global>"),
            "Box<serve::Particle>"
        );
        assert_eq!(
            clean_type_name("alloc::vec::Vec<alloc::string::String>"),
            "Vec<String>"
        );
        assert_eq!(
            clean_type_name(
                "std::collections::hash::map::HashMap<u64, serve::Session, std::hash::random::RandomState>"
            ),
            "HashMap<u64, serve::Session>"
        );
    }

    #[test]
    fn keeps_user_crate_paths_and_plain_types() {
        assert_eq!(clean_type_name("serve::Session"), "serve::Session");
        assert_eq!(clean_type_name("(u64, serve::Session)"), "(u64, serve::Session)");
        assert_eq!(clean_type_name("u8"), "u8");
        assert_eq!(clean_type_name("[u8; 4]"), "[u8; 4]");
        // A user type that happens to start like a std crate name is kept.
        assert_eq!(clean_type_name("alloc_tracker::Entry"), "alloc_tracker::Entry");
    }

    #[test]
    fn nested_wrappers_clean_recursively() {
        assert_eq!(
            clean_type_name(
                "alloc::sync::Arc<alloc::vec::Vec<alloc::boxed::Box<app::Node, alloc::alloc::Global>, alloc::alloc::Global>>"
            ),
            "Arc<Vec<Box<app::Node>>>"
        );
    }

    #[test]
    fn closures_and_non_path_syntax_survive() {
        assert_eq!(
            clean_type_name("{closure_env#1}<memscope_agent::start_at::{closure_env#0}, ()>"),
            "{closure_env#1}<memscope_agent::start_at::{closure_env#0}, ()>"
        );
        assert_eq!(clean_type_name("core::option::Option<&str>"), "Option<&str>");
    }
}
