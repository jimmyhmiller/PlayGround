//! Safety-property tests: prove the default-deny guarantee structurally.
//!
//! The property that matters: **no private route is ever allowed without a
//! valid token, for any config and any request path.** We test the pure
//! decision function `Router::decide` exhaustively with proptest (generated
//! configs × paths × auth states), plus explicit regression cases for the
//! classic accidental-exposure bugs (prefix confusion, %2f traversal, trailing
//! slash, case).

use gatekeeper::config::Route;
use gatekeeper::route::{Decision, Router};

fn route(path: &str, public: bool) -> Route {
    Route {
        path: path.to_string(),
        static_dir: Some(std::path::PathBuf::from("/tmp")),
        proxy: None,
        public,
    }
}

// ---- Explicit regression tests for known accidental-exposure bugs ----

#[test]
fn private_route_denied_without_token() {
    let r = Router::new(vec![route("/metrics", false)]);
    assert_eq!(r.decide("/metrics", false), Decision::Unauthorized);
    assert!(matches!(
        r.decide("/metrics", true),
        Decision::Allow { private: true, .. }
    ));
}

#[test]
fn public_route_allowed_without_token() {
    let r = Router::new(vec![route("/blog", true)]);
    assert!(matches!(
        r.decide("/blog", false),
        Decision::Allow { private: false, .. }
    ));
}

#[test]
fn prefix_confusion_does_not_leak() {
    // /admin private must NOT match /administrator (which has no route -> deny).
    let r = Router::new(vec![route("/admin", false)]);
    assert_eq!(r.decide("/administrator", false), Decision::NoRoute);
    assert_eq!(r.decide("/admin", false), Decision::Unauthorized);
}

#[test]
fn traversal_is_rejected_not_rerouted() {
    // A public /static must not let an encoded ../ escape to reach elsewhere.
    let r = Router::new(vec![route("/static", true), route("/secret", false)]);
    assert_eq!(r.decide("/static/..%2f..%2fsecret", false), Decision::BadPath);
    assert_eq!(r.decide("/static/../secret", false), Decision::BadPath);
}

#[test]
fn longest_prefix_public_hole_under_private_parent() {
    // Intentional pattern: public subpath carved out of a private area.
    let r = Router::new(vec![route("/admin", false), route("/admin/pub", true)]);
    assert!(matches!(
        r.decide("/admin/pub/x", false),
        Decision::Allow { private: false, .. }
    ));
    assert_eq!(r.decide("/admin/secret", false), Decision::Unauthorized);
}

#[test]
fn trailing_slash_and_case() {
    let r = Router::new(vec![route("/admin", false)]);
    assert_eq!(r.decide("/admin/", false), Decision::Unauthorized); // still private
    assert_eq!(r.decide("/Admin", false), Decision::NoRoute); // case-sensitive
}

#[test]
fn unmatched_is_never_allowed() {
    let r = Router::new(vec![route("/a", true)]);
    assert_eq!(r.decide("/b", false), Decision::NoRoute);
    assert_eq!(r.decide("/b", true), Decision::NoRoute);
}

// ---- Property test: the core invariant over random inputs ----

mod prop {
    use super::*;
    use proptest::prelude::*;

    // Generate a path segment from a small alphabet (so overlaps/collisions are
    // likely and the matcher is stressed).
    fn seg() -> impl Strategy<Value = String> {
        proptest::collection::vec(proptest::sample::select(vec!['a', 'b', 'c', '/']), 0..6)
            .prop_map(|cs| cs.into_iter().collect())
    }

    fn route_path() -> impl Strategy<Value = String> {
        seg().prop_map(|s| {
            // Normalize to a valid route path: leading '/', no trailing '/',
            // no empty segments. Fall back to "/a" if it collapses to nothing.
            let cleaned: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).collect();
            if cleaned.is_empty() {
                "/a".to_string()
            } else {
                format!("/{}", cleaned.join("/"))
            }
        })
    }

    fn routes_strategy() -> impl Strategy<Value = Vec<(String, bool)>> {
        proptest::collection::vec((route_path(), any::<bool>()), 1..6).prop_map(|mut v| {
            // Dedup paths (the real config validator rejects duplicates); keep
            // first occurrence so the table is well-formed.
            let mut seen = std::collections::HashSet::new();
            v.retain(|(p, _)| seen.insert(p.clone()));
            v
        })
    }

    fn request_path() -> impl Strategy<Value = String> {
        // Arbitrary request paths, including traversal and percent-encoding.
        prop_oneof![
            seg().prop_map(|s| format!("/{s}")),
            Just("/a/../b".to_string()),
            Just("/a/..%2f..%2fb".to_string()),
            Just("/a/%2e%2e/b".to_string()),
            ".*".prop_map(|s: String| s.chars().take(40).collect()),
        ]
    }

    proptest! {
        /// THE invariant: a private route is allowed only when auth_ok is true.
        #[test]
        fn private_never_allowed_without_auth(
            routes in routes_strategy(),
            path in request_path(),
            auth_ok in any::<bool>(),
        ) {
            let table: Vec<Route> = routes.iter().map(|(p, pub_)| route(p, *pub_)).collect();
            let router = Router::new(table);
            let decision = router.decide(&path, auth_ok);

            if let Decision::Allow { route_index, private } = decision {
                // If it allowed a PRIVATE route, auth MUST have been ok.
                if private {
                    prop_assert!(auth_ok, "private route allowed without auth: path={path:?}");
                }
                // The allowed route's own public flag must agree with `private`.
                let allowed = &router.routes()[route_index];
                prop_assert_eq!(allowed.public, !private);
            }
        }

        /// Path normalization never yields a `..` component (no traversal can
        /// survive normalization).
        #[test]
        fn normalized_paths_have_no_dotdot(path in request_path()) {
            if let Some(norm) = Router::normalize(&path) {
                prop_assert!(!norm.split('/').any(|s| s == ".." || s == "."));
                prop_assert!(norm.starts_with('/'));
            }
        }
    }
}
