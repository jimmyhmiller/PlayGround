//! Vite `define`: compile-time replacement of global identifiers with a resolved
//! value (e.g. `__OC_WEB_VERSION__` -> `"1.1.4"`).
//!
//! Like `import.meta.env`, this is a Vite convention applied only when a build
//! opts in (the `build-app` path supplies the map, evaluated once by
//! [`crate::vite_config`]). A define replaces a *free* identifier reference only:
//! a local binding of the same name shadows it, exactly as esbuild/Vite behave, so
//! the rewrite is scope-aware rather than a blind text substitution.
//!
//! Only bare-identifier define keys are handled here; dotted keys such as
//! `process.env.NODE_ENV` are a member-expression form not yet supported (they are
//! simply left in place, never mis-replaced).

use std::collections::HashMap;
use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_semantic::{Scoping, SemanticBuilder};
use oxc_span::{SourceType, Span};

/// Replaces free references to any `defines` key in `source` with its replacement
/// text, returning the rewritten source, or `None` when nothing is replaced.
pub fn transform(path: &Path, source: &str, defines: &[(String, String)]) -> Option<String> {
    // Only bare-identifier keys apply here; skip any dotted key.
    let simple: Vec<(&str, &str)> = defines
        .iter()
        .filter(|(key, _)| !key.contains('.'))
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    if simple.is_empty() {
        return None;
    }
    // Cheap gate: no key even appears as a substring.
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

impl<'a> Visit<'a> for DefineCollector<'_> {
    fn visit_identifier_reference(&mut self, identifier: &oxc_ast::ast::IdentifierReference<'a>) {
        if let Some(&replacement) = self.map.get(identifier.name.as_str()) {
            // A define replaces only a free (unbound) reference; a resolved
            // reference is a local binding that shadows the define.
            let is_free = identifier
                .reference_id
                .get()
                .and_then(|reference| self.scoping.get_reference(reference).symbol_id())
                .is_none();
            if is_free {
                self.edits.push((identifier.span, replacement.to_string()));
            }
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

    #[test]
    fn dotted_keys_are_skipped() {
        let defines = vec![("process.env.NODE_ENV".to_string(), "\"production\"".to_string())];
        assert!(transform(Path::new("m.ts"), "const e = process.env.NODE_ENV;", &defines).is_none());
    }
}
