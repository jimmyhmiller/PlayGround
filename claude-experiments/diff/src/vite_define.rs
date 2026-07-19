//! Vite `define`: compile-time replacement of global identifiers with a resolved
//! value (e.g. `__OC_WEB_VERSION__` -> `"1.1.4"`).
//!
//! Like `import.meta.env`, this is a Vite convention applied only when a build
//! opts in (the `build-app` path supplies the map, evaluated once by
//! [`crate::vite_config`]). A define replaces a *free* identifier reference only:
//! a local binding of the same name shadows it, exactly as esbuild/Vite behave, so
//! the rewrite is scope-aware rather than a blind text substitution.
//!
//! Both bare-identifier keys (`__OC_WEB_VERSION__`) and dotted member-expression
//! keys (`process.env.NODE_ENV`) are handled. The dotted form is what lets a
//! package's `if (process.env.NODE_ENV === 'production')` dispatch fold to a
//! literal comparison, which [`crate::dead_branch`] then resolves — the mechanism
//! that keeps React's development build out of a production bundle.

use std::collections::HashMap;
use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::Expression;
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_semantic::{Scoping, SemanticBuilder};
use oxc_span::{SourceType, Span};

/// Replaces free references to any `defines` key in `source` with its replacement
/// text, returning the rewritten source, or `None` when nothing is replaced.
pub fn transform(path: &Path, source: &str, defines: &[(String, String)]) -> Option<String> {
    let simple: Vec<(&str, &str)> = defines
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    if simple.is_empty() {
        return None;
    }
    // Cheap gate: no key even appears as a substring. A dotted key is matched
    // against the source verbatim, so `process . env . NODE_ENV` (whitespace
    // inside the member chain) is not gated in. That form does not occur in
    // published packages, and missing it only forgoes a substitution — it never
    // produces a wrong one.
    if !simple.iter().any(|(key, _)| source.contains(key)) {
        return None;
    }

    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let scoping = SemanticBuilder::new()
        .build(&parsed.program)
        .semantic
        .into_scoping();

    let map: HashMap<&str, &str> = simple.into_iter().collect();
    let mut collector = DefineCollector {
        scoping: &scoping,
        map: &map,
        edits: Vec::new(),
    };
    collector.visit_program(&parsed.program);
    if collector.edits.is_empty() {
        return None;
    }
    Some(apply_edits(source, collector.edits))
}

struct DefineCollector<'a> {
    scoping: &'a Scoping,
    map: &'a HashMap<&'a str, &'a str>,
    edits: Vec<(Span, String)>,
}

impl<'a> DefineCollector<'a> {
    /// Whether `identifier` is a free (unbound) reference. A resolved reference is
    /// a local binding that shadows the define.
    fn is_free(&self, identifier: &oxc_ast::ast::IdentifierReference<'a>) -> bool {
        identifier
            .reference_id
            .get()
            .and_then(|reference| self.scoping.get_reference(reference).symbol_id())
            .is_none()
    }

    /// The dotted path a static member chain spells (`process.env.NODE_ENV`), or
    /// `None` when the chain is not rooted in a plain identifier — which is what
    /// excludes `a.process.env.NODE_ENV` and any computed access. Only the ROOT of
    /// the chain is an identifier reference at all: `env` and `NODE_ENV` are
    /// property names, which can never be bound, so shadowing is decided entirely
    /// by whether that root identifier is free.
    fn member_path(&self, member: &oxc_ast::ast::StaticMemberExpression<'a>) -> Option<String> {
        let mut parts = vec![member.property.name.as_str()];
        let mut object = &member.object;
        loop {
            match object {
                Expression::StaticMemberExpression(inner) => {
                    parts.push(inner.property.name.as_str());
                    object = &inner.object;
                }
                Expression::Identifier(identifier) => {
                    if !self.is_free(identifier) {
                        return None;
                    }
                    parts.push(identifier.name.as_str());
                    parts.reverse();
                    return Some(parts.join("."));
                }
                _ => return None,
            }
        }
    }
}

impl<'a> Visit<'a> for DefineCollector<'_> {
    /// Matches a dotted key against the LONGEST member chain first, which is what
    /// esbuild and Vite do: with both `process.env` and `process.env.NODE_ENV`
    /// defined, the longer key wins. Because the visit runs outermost-in and
    /// returns without walking children on a hit, the outer (longer) chain is
    /// always tested before any prefix of it.
    fn visit_static_member_expression(
        &mut self,
        member: &oxc_ast::ast::StaticMemberExpression<'a>,
    ) {
        if let Some(path) = self.member_path(member)
            && let Some(&replacement) = self.map.get(path.as_str())
        {
            // Replace the WHOLE chain and stop descending, so the inner nodes are
            // never independently rewritten and the recorded spans never nest —
            // which `apply_edits` requires.
            self.edits.push((member.span, replacement.to_string()));
            return;
        }
        walk::walk_static_member_expression(self, member);
    }

    fn visit_identifier_reference(&mut self, identifier: &oxc_ast::ast::IdentifierReference<'a>) {
        if let Some(&replacement) = self.map.get(identifier.name.as_str())
            && self.is_free(identifier)
        {
            self.edits.push((identifier.span, replacement.to_string()));
        }
        walk::walk_identifier_reference(self, identifier);
    }
}

/// Applies non-overlapping `(span, replacement)` edits. Reference spans never
/// nest, so sorting by start and skipping any overlap keeps the output well-formed.
fn apply_edits(source: &str, mut edits: Vec<(Span, String)>) -> String {
    edits.sort_by_key(|(span, _)| span.start);
    let mut output = String::with_capacity(source.len());
    let mut cursor = 0_usize;
    for (span, replacement) in edits {
        let start = span.start as usize;
        let end = span.end as usize;
        if start < cursor {
            continue;
        }
        output.push_str(&source[cursor..start]);
        output.push_str(&replacement);
        cursor = end;
    }
    output.push_str(&source[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defines() -> Vec<(String, String)> {
        vec![
            ("__OC_WEB_VERSION__".to_string(), "\"1.1.4\"".to_string()),
            ("__DEBUG__".to_string(), "false".to_string()),
        ]
    }

    #[test]
    fn replaces_a_free_reference_with_its_value() {
        let out = transform(Path::new("m.ts"), "const v = __OC_WEB_VERSION__;", &defines()).unwrap();
        assert_eq!(out, "const v = \"1.1.4\";");
    }

    #[test]
    fn a_local_binding_shadows_the_define() {
        // A local `__DEBUG__` must NOT be replaced; the free one must be.
        let source = "function f(__DEBUG__) { return __DEBUG__; }\nconst g = __DEBUG__;";
        let out = transform(Path::new("m.ts"), source, &defines()).unwrap();
        assert!(out.contains("function f(__DEBUG__) { return __DEBUG__; }"), "{out}");
        assert!(out.contains("const g = false;"), "{out}");
    }

    #[test]
    fn a_property_access_is_not_a_define() {
        // `obj.__OC_WEB_VERSION__` is a property, not a global reference.
        let out = transform(Path::new("m.ts"), "const v = obj.__OC_WEB_VERSION__;", &defines());
        assert!(out.is_none(), "property access must not be replaced: {out:?}");
    }

    #[test]
    fn a_module_without_any_key_is_untouched() {
        assert!(transform(Path::new("m.ts"), "const x = 1;", &defines()).is_none());
    }

    fn node_env() -> Vec<(String, String)> {
        vec![(
            "process.env.NODE_ENV".to_string(),
            "\"production\"".to_string(),
        )]
    }

    #[test]
    fn a_dotted_key_replaces_the_whole_member_chain() {
        let out = transform(
            Path::new("m.ts"),
            "const e = process.env.NODE_ENV;",
            &node_env(),
        )
        .unwrap();
        assert_eq!(out, "const e = \"production\";");
    }

    #[test]
    fn a_dotted_key_folds_the_package_dispatch_condition() {
        // The shape every React-style package uses to pick its build. The point of
        // the substitution is that the result is a literal comparison the
        // dead-branch pass can resolve.
        let out = transform(
            Path::new("m.ts"),
            "if (process.env.NODE_ENV === 'production') { a(); } else { b(); }",
            &node_env(),
        )
        .unwrap();
        assert_eq!(
            out,
            "if (\"production\" === 'production') { a(); } else { b(); }"
        );
    }

    #[test]
    fn a_local_process_binding_shadows_a_dotted_define() {
        // Only the chain ROOT can be bound, so shadowing is decided there.
        let source = "function f(process) { return process.env.NODE_ENV; }";
        assert!(
            transform(Path::new("m.ts"), source, &node_env()).is_none(),
            "a local `process` must shadow the define"
        );
    }

    #[test]
    fn a_dotted_key_does_not_match_a_longer_chain_it_is_only_a_suffix_of() {
        // `a.process.env.NODE_ENV` is a property of `a`, not the global `process`.
        let out = transform(
            Path::new("m.ts"),
            "const e = a.process.env.NODE_ENV;",
            &node_env(),
        );
        assert!(out.is_none(), "suffix of a longer chain must not match: {out:?}");
    }

    #[test]
    fn the_longest_matching_dotted_key_wins() {
        let defines = vec![
            ("process.env".to_string(), "{}".to_string()),
            (
                "process.env.NODE_ENV".to_string(),
                "\"production\"".to_string(),
            ),
        ];
        let out = transform(
            Path::new("m.ts"),
            "const e = process.env.NODE_ENV;",
            &defines,
        )
        .unwrap();
        assert_eq!(out, "const e = \"production\";");
    }
}
