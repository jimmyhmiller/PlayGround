//! Vite's `import.meta.env` convention.
//!
//! `import.meta` is a JavaScript standard, but `import.meta.env` (with `MODE`,
//! `DEV`, `PROD`, `SSR`, `BASE_URL`, and `VITE_*` variables) is a **Vite
//! convention, not a standard**. So Diffpack treats it as an opt-in: it is applied
//! only when a build supplies [`ImportMetaEnv`] (via `BuildConfig::import_meta_env`,
//! which the TanStack/Vite `build-app` path sets). Generic bundling leaves
//! `import.meta.env` completely untouched.
//!
//! The rewrite matches Vite's static replacement, so dead branches fold:
//! `import.meta.env.KEY` becomes the value literal (`import.meta.env.PROD` -> `true`,
//! a `VITE_*` var -> its build-time string, an unknown key -> `undefined`), and a
//! bare `import.meta.env` (destructured, or accessed with a computed key) becomes
//! the full object literal so property access still yields the right value instead
//! of throwing on an `undefined` receiver.

use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::Expression;
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_span::{SourceType, Span};

use crate::transform::Target;

/// The `import.meta.env` values a build injects. `SSR` is not stored here: it is
/// derived from the [`Target`] each module is compiled for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportMetaEnv {
    /// `import.meta.env.BASE_URL`.
    pub base: String,
    /// `import.meta.env.MODE` (and the basis of `DEV`/`PROD`).
    pub mode: String,
    /// `VITE_*` variables captured at build time, as `(name, value)`.
    pub vite_vars: Vec<(String, String)>,
}

impl ImportMetaEnv {
    /// The JS-source value literal for a static `import.meta.env.KEY` access.
    fn value_for(&self, key: &str, target: Target) -> String {
        match key {
            "MODE" => json(&self.mode),
            "DEV" => "false".to_string(),
            "PROD" => "true".to_string(),
            "SSR" => bool_lit(target == Target::Server),
            "BASE_URL" => json(&self.base),
            _ => self
                .vite_vars
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| json(value))
                // An unknown key is `undefined`, exactly as reading a missing
                // property off Vite's env object would be.
                .unwrap_or_else(|| "undefined".to_string()),
        }
    }

    /// The object literal for a bare `import.meta.env` reference.
    fn object_literal(&self, target: Target) -> String {
        let mut parts = vec![
            format!("\"MODE\":{}", json(&self.mode)),
            "\"DEV\":false".to_string(),
            "\"PROD\":true".to_string(),
            format!("\"SSR\":{}", bool_lit(target == Target::Server)),
            format!("\"BASE_URL\":{}", json(&self.base)),
        ];
        for (name, value) in &self.vite_vars {
            parts.push(format!("{}:{}", json(name), json(value)));
        }
        format!("{{{}}}", parts.join(","))
    }
}

/// Rewrites `import.meta.env` references in `source` (a still-TypeScript/JSX module
/// at `path`) per Vite semantics, returning the rewritten source, or `None` when
/// the module has no `import.meta.env` to rewrite. Runs before the main transform,
/// source-to-source, like the route-split and server-fn rewrites.
pub fn transform(path: &Path, source: &str, options: &ImportMetaEnv, target: Target) -> Option<String> {
    // Cheap string gate before any parse: the vast majority of modules never
    // mention it.
    if !source.contains("import.meta.env") {
        return None;
    }
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut collector = EnvCollector {
        options,
        target,
        edits: Vec::new(),
    };
    collector.visit_program(&parsed.program);
    if collector.edits.is_empty() {
        return None;
    }
    Some(apply_edits(source, collector.edits))
}

struct EnvCollector<'o> {
    options: &'o ImportMetaEnv,
    target: Target,
    edits: Vec<(Span, String)>,
}

impl<'a> Visit<'a> for EnvCollector<'_> {
    fn visit_static_member_expression(
        &mut self,
        member: &oxc_ast::ast::StaticMemberExpression<'a>,
    ) {
        // `import.meta.env.KEY`: replace the whole access with the value literal,
        // and do not descend (the inner `import.meta.env` must not also be
        // rewritten to the object literal).
        if is_import_meta_env(&member.object) {
            let value = self.options.value_for(member.property.name.as_str(), self.target);
            self.edits.push((member.span, value));
            return;
        }
        // A bare `import.meta.env` (destructured, or the object of a computed
        // access `import.meta.env[expr]`): replace with the object literal.
        if member.property.name == "env" && is_import_meta(&member.object) {
            self.edits.push((member.span, self.options.object_literal(self.target)));
            return;
        }
        walk::walk_static_member_expression(self, member);
    }
}

/// Whether `expression` is the `import.meta` meta-property.
fn is_import_meta(expression: &Expression<'_>) -> bool {
    matches!(
        expression,
        Expression::MetaProperty(meta)
            if meta.meta.name == "import" && meta.property.name == "meta"
    )
}

/// Whether `expression` is `import.meta.env`.
fn is_import_meta_env(expression: &Expression<'_>) -> bool {
    matches!(
        expression,
        Expression::StaticMemberExpression(member)
            if member.property.name == "env" && is_import_meta(&member.object)
    )
}

fn json(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a JavaScript string cannot fail")
}

fn bool_lit(value: bool) -> String {
    if value { "true" } else { "false" }.to_string()
}

/// Applies non-overlapping `(span, replacement)` edits to `source`. The collector
/// never nests edits (it returns without descending once it rewrites a node), so
/// sorting by start offset and skipping any overlap keeps the output well-formed.
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

    fn env() -> ImportMetaEnv {
        ImportMetaEnv {
            base: "/".to_string(),
            mode: "production".to_string(),
            vite_vars: vec![("VITE_API".to_string(), "https://api.example.com".to_string())],
        }
    }

    fn run(source: &str, target: Target) -> String {
        transform(Path::new("m.tsx"), source, &env(), target).unwrap_or_else(|| source.to_string())
    }

    #[test]
    fn known_keys_fold_to_literals_for_dce() {
        let out = run("const a = import.meta.env.PROD; const b = import.meta.env.DEV; const c = import.meta.env.MODE;", Target::Client);
        assert!(out.contains("const a = true"), "{out}");
        assert!(out.contains("const b = false"), "{out}");
        assert!(out.contains("const c = \"production\""), "{out}");
    }

    #[test]
    fn ssr_depends_on_target() {
        assert!(run("x(import.meta.env.SSR)", Target::Server).contains("x(true)"));
        assert!(run("x(import.meta.env.SSR)", Target::Client).contains("x(false)"));
    }

    #[test]
    fn vite_var_uses_its_value_and_unknown_is_undefined() {
        let out = run("const u = import.meta.env.VITE_API; const m = import.meta.env.VITE_MISSING || '';", Target::Client);
        assert!(out.contains("const u = \"https://api.example.com\""), "{out}");
        assert!(out.contains("const m = undefined || ''"), "{out}");
    }

    #[test]
    fn bare_reference_becomes_the_object_so_access_never_throws() {
        // The oc-web failure mode: `import.meta.env` destructured/held, then a
        // property read. As an object literal the read yields undefined, not a
        // TypeError on an undefined receiver.
        let out = run("const e = import.meta.env; const url = e.VITE_MISSING || '';", Target::Client);
        assert!(out.contains("\"MODE\":\"production\""), "{out}");
        assert!(out.contains("\"PROD\":true"), "{out}");
        assert!(out.contains("\"VITE_API\":\"https://api.example.com\""), "{out}");
    }

    #[test]
    fn computed_access_replaces_the_receiver() {
        let out = run("const v = import.meta.env[key];", Target::Client);
        assert!(out.contains("[key]"), "{out}");
        assert!(out.contains("\"BASE_URL\":\"/\""), "receiver became the object: {out}");
    }

    #[test]
    fn a_module_without_import_meta_env_is_untouched() {
        assert!(transform(Path::new("m.ts"), "export const x = import.meta.url", &env(), Target::Server).is_none());
        assert!(transform(Path::new("m.ts"), "export const x = 1", &env(), Target::Server).is_none());
    }
}
