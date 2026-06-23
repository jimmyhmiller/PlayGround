//! Picks the *semantic* allocation shape and element type from a resolved frame
//! chain.
//!
//! Wrappers nest: `Rc<T>`/`Arc<T>` allocate through `Box`, `String` allocates
//! through `Vec<u8>`. Within a single site's frames these wrappers appear as a
//! contiguous run of std-container frames between the raw allocator glue and the
//! user's code. Strategy:
//!   1. take the element type from the *innermost* container frame that pins a
//!      concrete `T` (most reliable — closest to the actual allocation);
//!   2. take the shape from the *outermost* container frame in that run (the
//!      semantic intent: an `Rc`, not the `Box` it uses internally).

use memscope_proto::AllocShape;

use crate::dwarf::FnTypeInfo;

/// One resolved frame as seen by the recognizer.
pub struct ResolvedFrame<'a> {
    pub fn_name: Option<&'a str>,
    pub info: Option<&'a FnTypeInfo>,
}

/// Classify a demangled function name into a container shape, or `None` if the
/// frame is allocator glue / user code / unrecognized.
fn classify(name: &str) -> Option<AllocShape> {
    // Order matters: check the more specific wrappers before Box/Vec.
    if name.contains("alloc::sync::Arc") || name.contains("sync::Arc<") {
        return Some(AllocShape::Arc);
    }
    if name.contains("alloc::rc::Rc") || name.contains("rc::Rc<") {
        return Some(AllocShape::Rc);
    }
    if name.contains("alloc::string::String") || name.contains("string::String") {
        return Some(AllocShape::StringBuf);
    }
    if name.contains("hashbrown")
        || name.contains("collections::hash")
        || name.contains("HashMap")
        || name.contains("HashSet")
    {
        return Some(AllocShape::HashTable);
    }
    if name.contains("raw_vec::RawVec")
        || name.contains("RawVecInner")
        || name.contains("alloc::vec::Vec")
        || name.contains("vec::Vec<")
    {
        return Some(AllocShape::Vec);
    }
    if name.contains("alloc::boxed::Box") || name.contains("box_new") || name.contains("boxed::Box")
    {
        return Some(AllocShape::Boxed);
    }
    None
}

/// Is this frame raw allocator plumbing we should treat as transparent (it sits
/// between the container frames and the actual malloc, and never names a useful
/// element type)?
fn is_allocator_glue(name: &str) -> bool {
    const GLUE: &[&str] = &[
        "__rust_alloc",
        "__rdl_alloc",
        "alloc::alloc::alloc",
        "alloc::alloc::Global",
        "alloc_impl",
        "exchange_malloc",
        "as core::alloc::global",
        "Allocator>::allocate",
        "finish_grow",
        "try_allocate_in",
        "current_memory",
        "memscope_core",
        "GlobalAlloc",
    ];
    GLUE.iter().any(|g| name.contains(g))
}

/// Result of recognition.
pub struct Recognized {
    pub shape: Option<AllocShape>,
    pub element_type: Option<String>,
}

/// Walk `frames` innermost-first and recover (shape, element type).
pub fn recognize(frames: &[ResolvedFrame]) -> Recognized {
    let mut element_type: Option<String> = None;
    let mut shape: Option<AllocShape> = None;
    let mut in_container_run = false;

    for f in frames {
        let Some(name) = f.fn_name else { continue };

        // Skip our own frames and allocator glue without breaking the run.
        if is_allocator_glue(name) {
            continue;
        }

        match classify(name) {
            Some(s) => {
                in_container_run = true;
                // Outermost shape wins: keep overwriting as we move outward.
                shape = Some(s);
                // Innermost element type wins: only set if still unknown.
                if element_type.is_none() {
                    if let Some(et) = f.info.and_then(|i| i.element_type()) {
                        element_type = Some(et.to_string());
                    }
                }
            }
            None => {
                // A non-container, non-glue frame: if we've already seen the
                // container run, we've now reached user/caller code — stop so a
                // later unrelated container frame can't hijack the shape.
                if in_container_run {
                    break;
                }
            }
        }
    }

    // Special case: a String/HashTable whose element type we couldn't pin still
    // has a well-known element.
    if element_type.is_none() {
        match shape {
            Some(AllocShape::StringBuf) => element_type = Some("u8".to_string()),
            _ => {}
        }
    }

    Recognized {
        shape,
        element_type,
    }
}
