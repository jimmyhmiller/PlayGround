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

    // `Box<[T]>` — a boxed slice, e.g. the `vec![x]` → `into_vec` path — is really
    // a `Vec<T>` backing buffer. Unwrap the slice to its element + a Vec shape so
    // it matches the `.collect()` path instead of surfacing as a boxed `[T]`.
    if shape == Some(AllocShape::Boxed) {
        let slice_elem = element_type.as_deref().and_then(|t| {
            t.strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'))
                .filter(|inner| !inner.contains(';')) // a slice `[T]`, not array `[T; N]`
                .map(str::to_string)
        });
        if let Some(inner) = slice_elem {
            element_type = Some(inner);
            shape = Some(AllocShape::Vec);
        }
    }

    Recognized {
        shape,
        element_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(name: &str, elem: Option<&str>) -> (String, Option<FnTypeInfo>) {
        let info = elem.map(|t| FnTypeInfo {
            concrete_name: None,
            template_params: vec![("T".to_string(), t.to_string())],
        });
        (name.to_string(), info)
    }

    fn run(frames: &[(String, Option<FnTypeInfo>)]) -> Recognized {
        let rframes: Vec<ResolvedFrame> = frames
            .iter()
            .map(|(n, i)| ResolvedFrame {
                fn_name: Some(n.as_str()),
                info: i.as_ref(),
            })
            .collect();
        recognize(&rframes)
    }

    #[test]
    fn box_of_widget() {
        let f = vec![
            frame("__rust_alloc", None),
            frame("alloc::alloc::alloc", None),
            frame("alloc::boxed::Box<demo::Widget>::new", Some("demo::Widget")),
            frame("demo::main", None),
        ];
        let r = run(&f);
        assert_eq!(r.shape, Some(AllocShape::Boxed));
        assert_eq!(r.element_type.as_deref(), Some("demo::Widget"));
    }

    #[test]
    fn vec_of_u64() {
        let f = vec![
            frame("alloc::raw_vec::RawVec<u64>::with_capacity_in", Some("u64")),
            frame("alloc::vec::Vec<u64>::with_capacity", Some("u64")),
            frame("app::run", None),
        ];
        let r = run(&f);
        assert_eq!(r.shape, Some(AllocShape::Vec));
        assert_eq!(r.element_type.as_deref(), Some("u64"));
    }

    #[test]
    fn boxed_slice_becomes_vec() {
        // `vec![x]` allocates `Box<[T]>` via `into_vec`; surface it as `Vec<T>`,
        // not a boxed slice `[T]` (which also collides with JVM array names).
        let f = vec![
            frame("alloc::boxed::Box<T>::new_uninit", Some("[String]")),
            frame("app::main", None),
        ];
        let r = run(&f);
        assert_eq!(r.shape, Some(AllocShape::Vec));
        assert_eq!(r.element_type.as_deref(), Some("String"));
    }

    #[test]
    fn boxed_fixed_array_is_left_alone() {
        // A genuine boxed fixed array `Box<[u8; 4]>` is not a slice — keep it.
        let f = vec![
            frame("alloc::boxed::Box<T>::new", Some("[u8; 4]")),
            frame("app::main", None),
        ];
        let r = run(&f);
        assert_eq!(r.shape, Some(AllocShape::Boxed));
        assert_eq!(r.element_type.as_deref(), Some("[u8; 4]"));
    }

    #[test]
    fn rc_wins_over_inner_box() {
        // Rc<T> allocates through Box internally; the semantic shape is Rc, but
        // the element type is still T.
        let f = vec![
            frame("alloc::boxed::Box<T>::new", Some("app::Node")),
            frame("alloc::rc::Rc<app::Node>::new", Some("app::Node")),
            frame("app::build", None),
        ];
        let r = run(&f);
        assert_eq!(r.shape, Some(AllocShape::Rc));
        assert_eq!(r.element_type.as_deref(), Some("app::Node"));
    }

    #[test]
    fn string_is_stringbuf_u8() {
        let f = vec![
            frame("alloc::raw_vec::RawVec<u8>::with_capacity_in", Some("u8")),
            frame("alloc::vec::Vec<u8>::with_capacity", Some("u8")),
            frame("alloc::string::String::with_capacity", None),
            frame("app::format_thing", None),
        ];
        let r = run(&f);
        assert_eq!(r.shape, Some(AllocShape::StringBuf));
        assert_eq!(r.element_type.as_deref(), Some("u8"));
    }

    #[test]
    fn unrelated_outer_container_does_not_hijack() {
        // After the container run ends at user code, a later (caller's) Vec frame
        // must not steal the shape.
        let f = vec![
            frame("alloc::boxed::Box<app::Leaf>::new", Some("app::Leaf")),
            frame("app::make_leaf", None),
            frame("alloc::vec::Vec<app::Leaf>::push", Some("app::Leaf")),
        ];
        let r = run(&f);
        assert_eq!(r.shape, Some(AllocShape::Boxed));
        assert_eq!(r.element_type.as_deref(), Some("app::Leaf"));
    }
}
