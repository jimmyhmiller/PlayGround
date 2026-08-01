//! `import()` with a variable in the specifier — webpack's "context module",
//! Rollup/Vite's "dynamic import vars".
//!
//! ```js
//! const { default: messages } = await import(`./locales/${locale}/${ns}.json`);
//! ```
//!
//! A bundler cannot leave that alone: the emitted chunk lives somewhere else on
//! disk, so a relative specifier computed at runtime resolves against the OUTPUT
//! directory and the file is simply not there. webpack answers this by globbing
//! the pattern at build time and emitting a request -> module map (a *context
//! module*); Rollup's `dynamic-import-vars` (which is what Vite ships) does the
//! same thing. Both are load-bearing in real apps — i18n bundles, app-store
//! loaders, plugin registries — so this rewrite is ALWAYS ON rather than being an
//! opt-in convention like [`crate::import_meta_glob`].
//!
//! The rewrite is source-to-source and runs before the module is parsed for
//! dependencies, so every match becomes a real dynamic-import edge through the
//! normal pipeline (its own chunk), exactly as if it had been written by hand.
//!
//! What is rewritten, and what deliberately is not:
//!
//! * A template literal whose first quasi starts with `./` or `../` is expanded.
//!   Each `${...}` becomes a single-path-segment wildcard (`*`), which is Rollup's
//!   rule; webpack's default `wrappedContextRegExp` is looser (`.*`, crossing `/`),
//!   and the narrower rule is the safe subset — it can only ever pull in FEWER
//!   files, and it bounds the filesystem walk to the depth the pattern writes out.
//! * A pattern that matches nothing is left untouched. There is no honest map to
//!   emit, and rewriting would only move the same failure to a different message.
//! * A template with no `${}` at all is turned into the plain string literal it
//!   already is. That is not an optimisation: the dependency scanner only records
//!   `import()` of a StringLiteral, so ``import(`./a.js`)`` was previously an
//!   invisible edge whose target never made it into the graph.
//! * A bare, absolute or URL-shaped pattern (``import(`@scope/${x}`)``,
//!   ``import(`${base}/x.js`)``) is left untouched: those are not importer-relative,
//!   so there is no directory to glob, and the runtime import is often genuinely
//!   dynamic (a URL). webpack warns here too and bundles nothing.
//!
//! The generated helper is a hoisted function declaration appended at the END of
//! the module, so it cannot disturb a directive prologue (`"use client"` must stay
//! the first statement) and is nonetheless callable from code above it. An
//! unmatched request rejects with a MODULE_NOT_FOUND error naming the request —
//! the same observable shape as a missing dynamic import, never a silent
//! `undefined`.

use std::collections::BTreeSet;
use std::path::Path;

use oxc_allocator::Allocator;
use oxc_ast::ast::{Expression, ImportExpression, TemplateLiteral};
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_span::Span;

/// Rewrites every variable `import()` in `source` (a still-TypeScript/JSX module
/// at `path`), returning the rewritten source or `None` when the module has none.
/// Runs before the main transform, source-to-source, like
/// [`crate::import_meta_glob::transform`].
pub fn transform(path: &Path, source: &str) -> Option<String> {
    // Cheap string gate before any parse: only a module that literally writes
    // `import(` followed by a backtick can contain one, and the vast majority of
    // modules do not.
    if !has_template_import(source) {
        return None;
    }
    let allocator = Allocator::default();
    let source_type = diffpack_core::parser::scan_source_type(path);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut collector = ContextCollector {
        path,
        source,
        edits: Vec::new(),
        helpers: Vec::new(),
        calls: 0,
    };
    collector.visit_program(&parsed.program);
    if collector.edits.is_empty() {
        return None;
    }
    let mut output = apply_edits(source, collector.edits);
    if !collector.helpers.is_empty() {
        output.push('\n');
        output.push_str(&collector.helpers.join("\n"));
        output.push('\n');
    }
    Some(output)
}

/// Whether `source` contains `import` `(` `` ` `` with only whitespace between —
/// the only spelling a template-literal dynamic import can have. A false positive
/// (the text inside a comment or a string) costs one parse and nothing else; the
/// scan must never produce a false NEGATIVE, which is why it tolerates whitespace
/// and newlines rather than matching the literal `"import(`"`.
fn has_template_import(source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut from = 0;
    while let Some(at) = source[from..].find("import") {
        let mut index = from + at + "import".len();
        from = from + at + 1;
        while bytes.get(index).is_some_and(|b| b.is_ascii_whitespace()) {
            index += 1;
        }
        if bytes.get(index) != Some(&b'(') {
            continue;
        }
        index += 1;
        while bytes.get(index).is_some_and(|b| b.is_ascii_whitespace()) {
            index += 1;
        }
        if bytes.get(index) == Some(&b'`') {
            return true;
        }
    }
    false
}

struct ContextCollector<'c> {
    path: &'c Path,
    source: &'c str,
    edits: Vec<(Span, String)>,
    /// Hoisted helper function declarations, in call order.
    helpers: Vec<String>,
    calls: usize,
}

impl<'a> Visit<'a> for ContextCollector<'_> {
    fn visit_import_expression(&mut self, import: &ImportExpression<'a>) {
        if let Expression::TemplateLiteral(template) = &import.source
            && let Some(edit) = self.rewrite(import, template)
        {
            self.edits.push(edit);
        }
        walk::walk_import_expression(self, import);
    }
}

impl ContextCollector<'_> {
    /// The edit for one `import(`...`)`, or `None` to leave the call as written.
    fn rewrite(
        &mut self,
        import: &ImportExpression<'_>,
        template: &TemplateLiteral<'_>,
    ) -> Option<(Span, String)> {
        let quasis: Vec<String> = template
            .quasis
            .iter()
            .map(|element| {
                element
                    .value
                    .cooked
                    .as_ref()
                    .map(|cooked| cooked.to_string())
                    .unwrap_or_else(|| element.value.raw.to_string())
            })
            .collect();

        // No interpolation: the specifier is already a constant. Replace just the
        // template with the string literal it is, so the dependency scanner (which
        // only records `import()` of a StringLiteral) can see the edge.
        if template.expressions.is_empty() {
            let literal = quasis.first()?;
            return Some((template.span, json(literal)));
        }

        // Only an importer-relative pattern has a directory to glob.
        let first = quasis.first()?;
        if !(first.starts_with("./") || first.starts_with("../")) {
            return None;
        }
        // A pattern must name a file, not a whole directory: without a static tail
        // the "extension" is a variable too and every file under the base matches.
        if quasis.last()?.is_empty() {
            return None;
        }

        // `./locales/${a}/${b}.json` -> `./locales/*/*.json`.
        let pattern = quasis.join("*");
        let matches = expand(self.path, &pattern)?;
        if matches.is_empty() {
            return None;
        }

        let index = self.calls;
        self.calls += 1;
        let name = format!("__diffpack_dynimport_{index}");
        let mut arms = String::new();
        for request in &matches {
            arms.push_str(&format!(
                "\n    case {0}: return import({0});",
                json(request)
            ));
        }
        let message = format!(
            "Cannot find module '\" + request + \"' \
             (no file matched the dynamic import pattern {} in {} at build time)",
            pattern.escape_debug(),
            self.path.display().to_string().escape_debug(),
        );
        self.helpers.push(format!(
            "function {name}(request) {{\n  switch (request) {{{arms}\n  }}\n  \
             return Promise.reject(Object.assign(new Error(\"{message}\"), \
             {{ code: \"MODULE_NOT_FOUND\" }}));\n}}"
        ));

        // Replace the whole `import(...)` with a call to the map, keeping the
        // template verbatim so its expressions still evaluate once, in order.
        let template_text = &self.source[template.span.start as usize..template.span.end as usize];
        Some((import.span, format!("{name}({template_text})")))
    }
}

/// Matches `pattern` (importer-relative, `*` = one path-segment fragment) against
/// the filesystem, returning the request strings — the pattern with each `*`
/// substituted — sorted and deduplicated. `None` when the pattern escapes above
/// the filesystem root or the importer has no parent directory.
fn expand(importer: &Path, pattern: &str) -> Option<Vec<String>> {
    let mut base = importer.parent()?.to_path_buf();
    let segments: Vec<&str> = pattern.split('/').collect();
    // Leading `.` / `..` navigation, kept verbatim so the request can be rebuilt.
    let mut prefix = Vec::new();
    let mut rest = &segments[..];
    while let Some(first) = rest.first() {
        match *first {
            "." => {
                prefix.push(".");
                rest = &rest[1..];
            }
            ".." => {
                base = base.parent()?.to_path_buf();
                prefix.push("..");
                rest = &rest[1..];
            }
            _ => break,
        }
    }
    if rest.is_empty() || prefix.is_empty() {
        return None;
    }
    let mut out = BTreeSet::new();
    walk_directory(&base, rest, &mut Vec::new(), &mut out);
    Some(
        out.into_iter()
            .map(|matched| format!("{}/{matched}", prefix.join("/")))
            .collect(),
    )
}

/// Recursively matches `pattern` segments against entries under `directory`,
/// collecting the matched paths relative to it.
fn walk_directory(
    directory: &Path,
    pattern: &[&str],
    sofar: &mut Vec<String>,
    out: &mut BTreeSet<String>,
) {
    let Some((segment, rest)) = pattern.split_first() else {
        return;
    };
    if !segment.contains('*') {
        // A literal segment needs no directory listing.
        let child = directory.join(segment);
        sofar.push((*segment).to_string());
        if rest.is_empty() {
            if child.is_file() {
                out.insert(sofar.join("/"));
            }
        } else if child.is_dir() {
            walk_directory(&child, rest, sofar, out);
        }
        sofar.pop();
        return;
    }
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // A wildcard never matches a dotfile, matching every glob convention.
        if name.starts_with('.') || !segment_matches(segment, &name) {
            continue;
        }
        let child = entry.path();
        sofar.push(name);
        if rest.is_empty() {
            if child.is_file() {
                out.insert(sofar.join("/"));
            }
        } else if child.is_dir() {
            walk_directory(&child, rest, sofar, out);
        }
        sofar.pop();
    }
}

/// `*` matches any run of characters within one path segment (never `/`, which
/// cannot appear in a directory entry name anyway).
fn segment_matches(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let first = parts[0];
    let last = parts[parts.len() - 1];
    if !name.starts_with(first) || name.len() < first.len() + last.len() {
        return false;
    }
    let mut rest = &name[first.len()..];
    for middle in &parts[1..parts.len() - 1] {
        if middle.is_empty() {
            continue;
        }
        match rest.find(middle) {
            Some(at) => rest = &rest[at + middle.len()..],
            None => return false,
        }
    }
    rest.ends_with(last)
}

fn json(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a JavaScript string cannot fail")
}

/// Applies non-overlapping `(span, replacement)` edits.
fn apply_edits(source: &str, mut edits: Vec<(Span, String)>) -> String {
    edits.sort_by_key(|(span, _)| (span.start, span.end));
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

    fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("src/locales/de")).unwrap();
        std::fs::create_dir_all(root.join("src/locales/fr")).unwrap();
        std::fs::create_dir_all(root.join("src/locales/.hidden")).unwrap();
        std::fs::write(root.join("src/locales/de/common.json"), "{}").unwrap();
        std::fs::write(root.join("src/locales/de/other.json"), "{}").unwrap();
        std::fs::write(root.join("src/locales/fr/common.json"), "{}").unwrap();
        std::fs::write(root.join("src/locales/.hidden/common.json"), "{}").unwrap();
        std::fs::write(root.join("src/plain.js"), "").unwrap();
        let importer = root.join("src/entry.ts");
        (directory, importer)
    }

    /// The defect this module exists for: cal.com's `packages/i18n/server.ts` does
    /// `await import(`./locales/${locale}/${ns}.json`)`, which resolved against the
    /// OUTPUT directory at runtime and threw ERR_MODULE_NOT_FOUND for every
    /// non-English locale.
    #[test]
    fn a_two_variable_pattern_expands_to_every_matching_file() {
        let (_directory, importer) = fixture();
        let source = "const m = await import(`./locales/${locale}/${ns}.json`);\n";
        let output = transform(&importer, source).expect("the pattern must be rewritten");
        assert!(
            output.contains("__diffpack_dynimport_0(`./locales/${locale}/${ns}.json`)"),
            "the call is routed through the map, keeping the template verbatim: {output}"
        );
        for expected in [
            "case \"./locales/de/common.json\": return import(\"./locales/de/common.json\");",
            "case \"./locales/de/other.json\": return import(\"./locales/de/other.json\");",
            "case \"./locales/fr/common.json\": return import(\"./locales/fr/common.json\");",
        ] {
            assert!(
                output.contains(expected),
                "missing arm {expected} in {output}"
            );
        }
        assert!(
            !output.contains(".hidden"),
            "a wildcard must not match a dotted directory: {output}"
        );
        assert!(
            output.contains("code: \"MODULE_NOT_FOUND\""),
            "an unmatched request must reject loudly, never resolve to undefined: {output}"
        );
    }

    #[test]
    fn a_static_tail_narrows_the_match() {
        let (_directory, importer) = fixture();
        let source = "import(`./locales/${l}/common.json`);\n";
        let output = transform(&importer, source).expect("the pattern must be rewritten");
        assert!(output.contains("\"./locales/de/common.json\""), "{output}");
        assert!(output.contains("\"./locales/fr/common.json\""), "{output}");
        assert!(
            !output.contains("other.json"),
            "the static tail excludes other.json: {output}"
        );
    }

    #[test]
    fn a_template_without_interpolation_becomes_a_string_literal() {
        let (_directory, importer) = fixture();
        let output = transform(&importer, "import(`./plain.js`);\n")
            .expect("a constant template must become a literal the scanner can see");
        assert_eq!(output.trim(), "import(\"./plain.js\");");
    }

    #[test]
    fn a_non_relative_pattern_is_left_alone() {
        let (_directory, importer) = fixture();
        // A bare specifier and a fully computed base are both genuinely dynamic;
        // webpack bundles nothing for them either.
        assert!(transform(&importer, "import(`@scope/${name}`);\n").is_none());
        assert!(transform(&importer, "import(`${base}/x.js`);\n").is_none());
    }

    #[test]
    fn a_pattern_matching_nothing_is_left_alone() {
        let (_directory, importer) = fixture();
        assert!(transform(&importer, "import(`./missing/${x}.js`);\n").is_none());
    }

    #[test]
    fn a_variable_extension_is_left_alone() {
        let (_directory, importer) = fixture();
        // `./locales/de/common${ext}` would otherwise match every file under the
        // base directory, which is not what the author asked for.
        assert!(transform(&importer, "import(`./locales/${l}/common${ext}`);\n").is_none());
    }

    #[test]
    fn a_use_client_directive_stays_first() {
        let (_directory, importer) = fixture();
        let source = "\"use client\";\nimport(`./locales/${l}/common.json`);\n";
        let output = transform(&importer, source).expect("rewritten");
        assert!(
            output.starts_with("\"use client\";"),
            "the helper must be appended, never prepended: {output}"
        );
        assert!(
            output.contains("function __diffpack_dynimport_0"),
            "{output}"
        );
    }

    #[test]
    fn several_patterns_get_distinct_helpers() {
        let (_directory, importer) = fixture();
        let source = "import(`./locales/${a}/common.json`); import(`./locales/${b}/other.json`);\n";
        let output = transform(&importer, source).expect("rewritten");
        assert!(
            output.contains("function __diffpack_dynimport_0"),
            "{output}"
        );
        assert!(
            output.contains("function __diffpack_dynimport_1"),
            "{output}"
        );
    }

    #[test]
    fn segment_matching_is_within_one_segment() {
        assert!(segment_matches("*", "de"));
        assert!(segment_matches("*.json", "common.json"));
        assert!(segment_matches("a*c", "abc"));
        assert!(!segment_matches("a*c", "ab"));
        assert!(!segment_matches("*.json", "common.jsonx"));
        assert!(segment_matches("common.json", "common.json"));
        assert!(!segment_matches("common.json", "other.json"));
    }
}
