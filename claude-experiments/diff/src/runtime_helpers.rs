//! The JavaScript runtime helpers oxc's transforms call, served from inside the
//! binary.
//!
//! Some lowerings cannot be expressed as pure syntax. TypeScript's legacy decorators
//! are the case that brought this module into existence: `@Memoize() foo() {}` lowers
//! to a call to `__decorate`, a real function whose body implements TypeScript's
//! decorator protocol (including the `Reflect.decorate` handoff). oxc emits
//! `import _decorate from "<package>/helpers/decorate"` for it — its "inline the
//! helper" mode is unimplemented — so a package by that name has to resolve to
//! something.
//!
//! Pointing that at oxc's real `@oxc-project/runtime` would make every decorator-using
//! app need an `npm install` it never needed under Next or Vite, and would silently
//! bind the emit to whatever version of that package the app happens to have hoisted.
//! Instead diffpack claims a package name of its OWN, resolves it before the resolver
//! ever touches the filesystem, and answers from [`scripts/helpers`] — files embedded
//! at compile time, so the helper the transform was written against is the helper that
//! ships (see `scripts/helpers/README.md` for their provenance and license).

/// The package name diffpack hands oxc's helper loader. Scoped and diffpack-specific
/// so it cannot collide with anything installable, and so a `@diffpack/runtime/...`
/// specifier appearing in a stack trace is unambiguously ours.
pub const HELPER_PACKAGE: &str = "@diffpack/runtime";

/// The specifier prefix oxc builds from [`HELPER_PACKAGE`]: `<package>/helpers/<name>`.
const HELPER_SPECIFIER_PREFIX: &str = "@diffpack/runtime/helpers/";

/// The helper `specifier` names, or `None` when the specifier is not a helper at all.
/// A helper diffpack does not carry is `Some(None)`-shaped in the caller's terms: it
/// still IS a helper specifier, so the caller must refuse it by name rather than let
/// it fall through to the filesystem resolver and be reported as a missing package.
pub fn helper_name(specifier: &str) -> Option<&str> {
    specifier.strip_prefix(HELPER_SPECIFIER_PREFIX)
}

/// The ES module source for a helper specifier: an `export default` of the helper
/// function, which is the shape oxc's emitted `import _x from ...` reads.
///
/// `None` for a helper this build does not carry. Only the transforms diffpack
/// actually enables can request one, so the set here is exactly the reachable set,
/// not a partial copy of `@oxc-project/runtime`.
pub fn helper_source(specifier: &str) -> Option<&'static str> {
    match helper_name(specifier)? {
        "decorate" => Some(include_str!("../scripts/helpers/decorate.js")),
        "decorateMetadata" => Some(include_str!("../scripts/helpers/decorateMetadata.js")),
        "decorateParam" => Some(include_str!("../scripts/helpers/decorateParam.js")),
        _ => None,
    }
}

/// The refusal for a helper specifier with no embedded source. Nothing an app author
/// can install fixes this — the specifier names a diffpack-internal package — so the
/// message is addressed to whoever enabled the transform that asked for it.
pub fn unknown_helper_error(specifier: &str) -> String {
    let name = helper_name(specifier).unwrap_or(specifier);
    format!(
        "diffpack does not carry the oxc runtime helper `{name}` (requested as \
         \"{specifier}\"). A transform this build enables lowers to a call into it, so \
         the emitted code would reference a function that is not there. Vendor \
         `src/helpers/esm/{name}.js` from `@oxc-project/runtime` at the pinned \
         oxc_transformer version into scripts/helpers/ and register it in \
         src/runtime_helpers.rs (see scripts/helpers/README.md)."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three helpers the legacy-decorator lowering can emit are all carried, and
    /// each is an ES module with a default export — the exact shape oxc's
    /// `import _x from "<package>/helpers/<name>"` reads. A helper that resolved but
    /// exported nothing would bind `_x` to `undefined` and fail only when a decorated
    /// member is first evaluated.
    #[test]
    fn every_helper_the_decorator_lowering_emits_is_carried_and_default_exports() {
        for name in ["decorate", "decorateMetadata", "decorateParam"] {
            let specifier = format!("{HELPER_PACKAGE}/helpers/{name}");
            let source = helper_source(&specifier)
                .unwrap_or_else(|| panic!("no embedded source for the `{name}` helper"));
            assert!(
                source.contains("as default"),
                "the `{name}` helper must default-export the function oxc imports:\n{source}"
            );
        }
    }

    /// A helper specifier is claimed by this module even when the helper is unknown,
    /// so it can be refused by name instead of reaching the filesystem resolver and
    /// being reported as a missing npm package the user is told to install.
    #[test]
    fn an_unknown_helper_is_still_recognized_as_a_helper_and_refused_by_name() {
        let specifier = format!("{HELPER_PACKAGE}/helpers/objectSpread2");
        assert_eq!(helper_name(&specifier), Some("objectSpread2"));
        assert!(helper_source(&specifier).is_none());
        let error = unknown_helper_error(&specifier);
        assert!(error.contains("objectSpread2"), "{error}");
        assert!(error.contains("scripts/helpers"), "{error}");
    }

    /// An ordinary package is not mistaken for a helper.
    #[test]
    fn a_normal_specifier_is_not_a_helper() {
        assert_eq!(helper_name("react"), None);
        assert_eq!(helper_name("@diffpack/runtime-ish/helpers/decorate"), None);
    }
}
