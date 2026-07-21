//! Vite's `import.meta.glob` convention.
//!
//! `import.meta.glob('./dir/*.ts')` expands at build time to an object literal
//! mapping each matched path (as written relative to the importing module, e.g.
//! `./dir/a.ts`) to a `() => import('./dir/a.ts')` thunk; `{ eager: true }`
//! expands to static imports instead. Like `import.meta.env`, this is a **Vite
//! convention, not a standard**, so it is strictly opt-in: the rewrite runs only
//! when a build supplies [`ImportMetaGlob`] (via `BuildConfig::import_meta_glob`,
//! set by the Vite-convention build paths). Generic bundling leaves
//! `import.meta.glob` completely untouched — it then flows through as ordinary
//! `import.meta` (fine in ESM output, refused with a module-naming error in
//! CommonJS output).
//!
//! The rewrite is source-to-source and runs before the module is parsed for
//! dependencies, so every expansion becomes a real graph edge through the normal
//! pipeline: each lazy match is its own dynamic-import edge (its own chunk), and
//! eager matches are ordinary static imports. Malformed calls are hard errors
//! naming the file and construct — never a silently empty object. An empty MATCH
//! (a well-formed pattern that finds no files) is valid and yields `{}`, exactly
//! as in Vite.
//!
//! KNOWN LIMITATION (watch/dev): the glob is expanded when the importing module
//! is transformed, so ADDING or REMOVING a file that a glob matches does not
//! invalidate the importer — the map goes stale until the importer itself is
//! rebuilt. Vite's dev server re-globs on directory change; wiring that
//! filesystem-level invalidation into the dev server is future work. (EDITING a
//! matched file is fine: it is a normal module in the graph.)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    Argument, ArrayExpressionElement, CallExpression, Expression, ObjectPropertyKind, PropertyKey,
};
use oxc_ast_visit::{Visit, walk};
use oxc_parser::Parser;
use oxc_span::{SourceType, Span};

/// Build-level configuration for the glob rewrite. Presence of this value is the
/// opt-in gate (mirroring `ImportMetaEnv`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportMetaGlob {
    /// The project root that `/`-prefixed (root-relative) patterns resolve
    /// against — Vite's config root. Must be absolute.
    pub root: PathBuf,
}

/// Expands every `import.meta.glob(...)` call in `source` (a still-TypeScript/JSX
/// module at `path`), returning the rewritten source, `Ok(None)` when the module
/// has no glob calls, or a file-and-construct-naming error for a malformed call.
/// Runs before the main transform, source-to-source, like `import_meta_env`.
pub fn transform(
    path: &Path,
    source: &str,
    options: &ImportMetaGlob,
) -> Result<Option<String>, String> {
    // Cheap string gate before any parse: the vast majority of modules never
    // mention it.
    if !source.contains("import.meta.glob") {
        return Ok(None);
    }
    let allocator = Allocator::default();
    let source_type = SourceType::from_path(path).unwrap_or_default().with_module(true);
    let parsed = Parser::new(&allocator, source, source_type).parse();
    let mut collector = GlobCollector {
        path,
        options,
        edits: Vec::new(),
        prelude: Vec::new(),
        calls: 0,
        error: None,
    };
    collector.visit_program(&parsed.program);
    if let Some(error) = collector.error {
        return Err(error);
    }
    if collector.edits.is_empty() {
        return Ok(None);
    }
    let mut edits = collector.edits;
    if !collector.prelude.is_empty() {
        // Eager imports are inserted after the hashbang and directive prologue
        // ("use client" and friends must stay first), before any other code.
        let offset = parsed
            .program
            .directives
            .last()
            .map(|directive| directive.span.end)
            .or_else(|| parsed.program.hashbang.as_ref().map(|hashbang| hashbang.span.end))
            .unwrap_or(0);
        edits.push((Span::new(offset, offset), format!("\n{}\n", collector.prelude.join("\n"))));
    }
    Ok(Some(apply_edits(source, edits)))
}

/// The parsed second argument of a glob call.
#[derive(Default)]
struct GlobCallOptions {
    eager: bool,
    /// `{ import: '...' }`: `None` and `Some("*")` both mean the namespace.
    import: Option<String>,
    /// A validated loader query (`?raw` or `?url`), appended to each specifier.
    query: Option<String>,
}

struct GlobCollector<'c> {
    path: &'c Path,
    options: &'c ImportMetaGlob,
    edits: Vec<(Span, String)>,
    /// Static import statements hoisted for eager globs, in call order.
    prelude: Vec<String>,
    /// Calls rewritten so far; namespaces eager identifiers per call.
    calls: usize,
    error: Option<String>,
}

impl<'a> Visit<'a> for GlobCollector<'_> {
    fn visit_call_expression(&mut self, call: &CallExpression<'a>) {
        if self.error.is_some() {
            return;
        }
        if is_import_meta_glob(&call.callee) {
            match self.rewrite(call) {
                // The arguments are consumed by the rewrite; do not descend.
                Ok(replacement) => self.edits.push((call.span, replacement)),
                Err(error) => self.error = Some(error),
            }
            return;
        }
        walk::walk_call_expression(self, call);
    }
}

impl GlobCollector<'_> {
    /// Expands one call into its replacement object literal (recording any eager
    /// imports in the prelude), or a hard error naming this file.
    fn rewrite(&mut self, call: &CallExpression<'_>) -> Result<String, String> {
        let call_index = self.calls;
        self.calls += 1;

        let mut arguments = call.arguments.iter();
        let first = arguments.next().ok_or_else(|| {
            format!("{}: import.meta.glob requires a pattern argument", self.path.display())
        })?;
        let patterns = self.collect_patterns(first)?;
        let options = match arguments.next() {
            None => GlobCallOptions::default(),
            Some(argument) => self.parse_options(argument)?,
        };
        if arguments.next().is_some() {
            return Err(format!(
                "{}: import.meta.glob takes at most two arguments (patterns, options)",
                self.path.display()
            ));
        }

        let entries = self.expand(&patterns)?;

        let mut properties = Vec::with_capacity(entries.len());
        for (index, (key, specifier)) in entries.iter().enumerate() {
            let specifier = match &options.query {
                Some(query) => format!("{specifier}{query}"),
                None => specifier.clone(),
            };
            let value = if options.eager {
                let identifier = format!("__diffpack_glob_{call_index}_{index}");
                self.prelude
                    .push(import_statement(&identifier, options.import.as_deref(), &specifier));
                identifier
            } else {
                lazy_thunk(options.import.as_deref(), &specifier)
            };
            properties.push(format!("{}: {value}", json(key)));
        }
        // Parenthesized so a statement-position call does not turn into a block.
        Ok(format!("({{{}}})", properties.join(", ")))
    }

    /// The pattern strings of the first argument: a string literal or an array of
    /// string literals; anything else is a hard error (Vite requires literal
    /// patterns for the same reason — the expansion happens at build time).
    fn collect_patterns(&self, argument: &Argument<'_>) -> Result<Vec<String>, String> {
        let non_literal = || {
            format!(
                "{}: import.meta.glob patterns must be string literals \
                 (a string or an array of strings); dynamic patterns cannot be \
                 expanded at build time",
                self.path.display()
            )
        };
        match argument.as_expression() {
            Some(Expression::StringLiteral(literal)) => Ok(vec![literal.value.to_string()]),
            Some(Expression::ArrayExpression(array)) => {
                let mut patterns = Vec::with_capacity(array.elements.len());
                for element in &array.elements {
                    match element {
                        ArrayExpressionElement::StringLiteral(literal) => {
                            patterns.push(literal.value.to_string());
                        }
                        _ => return Err(non_literal()),
                    }
                }
                Ok(patterns)
            }
            _ => Err(non_literal()),
        }
    }

    /// The options object: an inline literal with only the supported keys.
    fn parse_options(&self, argument: &Argument<'_>) -> Result<GlobCallOptions, String> {
        let file = self.path.display();
        let Some(Expression::ObjectExpression(object)) = argument.as_expression() else {
            return Err(format!(
                "{file}: import.meta.glob options must be an inline object literal"
            ));
        };
        let mut options = GlobCallOptions::default();
        for property in &object.properties {
            let ObjectPropertyKind::ObjectProperty(property) = property else {
                return Err(format!(
                    "{file}: import.meta.glob options cannot use spread properties"
                ));
            };
            let key = match &property.key {
                PropertyKey::StaticIdentifier(identifier) => identifier.name.to_string(),
                PropertyKey::StringLiteral(literal) => literal.value.to_string(),
                _ => {
                    return Err(format!(
                        "{file}: import.meta.glob option keys must be static (no computed keys)"
                    ));
                }
            };
            match key.as_str() {
                "eager" => match &property.value {
                    Expression::BooleanLiteral(literal) => options.eager = literal.value,
                    _ => {
                        return Err(format!(
                            "{file}: import.meta.glob `eager` must be a boolean literal"
                        ));
                    }
                },
                "import" => match &property.value {
                    Expression::StringLiteral(literal) => {
                        options.import = Some(literal.value.to_string());
                    }
                    _ => {
                        return Err(format!(
                            "{file}: import.meta.glob `import` must be a string literal"
                        ));
                    }
                },
                "query" => match &property.value {
                    Expression::StringLiteral(literal) => {
                        // Vite accepts both `?raw` and `raw`; normalize to the
                        // `?`-prefixed loader form.
                        let value = literal.value.trim_start_matches('?');
                        if value != "raw" && value != "url" {
                            return Err(format!(
                                "{file}: import.meta.glob `query` value {:?} is not supported \
                                 (supported: '?raw', '?url')",
                                literal.value.as_str()
                            ));
                        }
                        options.query = Some(format!("?{value}"));
                    }
                    _ => {
                        return Err(format!(
                            "{file}: import.meta.glob `query` must be a string literal \
                             ('?raw' or '?url')"
                        ));
                    }
                },
                "exhaustive" => {
                    return Err(format!(
                        "{file}: import.meta.glob option `exhaustive` is not supported"
                    ));
                }
                other => {
                    return Err(format!(
                        "{file}: unknown import.meta.glob option `{other}` \
                         (supported: eager, import, query)"
                    ));
                }
            }
        }
        Ok(options)
    }

    /// Matches the patterns against the filesystem, returning sorted
    /// `(map_key, import_specifier)` pairs. Keys keep the pattern's form
    /// (`./`/`../` importer-relative, or `/` root-relative); specifiers are
    /// always importer-relative so the resolver treats them like hand-written
    /// imports.
    fn expand(&self, patterns: &[String]) -> Result<Vec<(String, String)>, String> {
        let importer_directory =
            absolute_segments(self.path.parent().unwrap_or_else(|| Path::new("/")));
        let importer = absolute_segments(self.path);
        let root = absolute_segments(&self.options.root);

        let mut positives = Vec::new();
        let mut negatives = Vec::new();
        for pattern in patterns {
            let parsed = self.parse_pattern(pattern, &importer_directory, &root)?;
            if parsed.negative {
                negatives.push(parsed);
            } else {
                positives.push(parsed);
            }
        }
        if positives.is_empty() {
            return Err(format!(
                "{}: import.meta.glob has only negative patterns; at least one \
                 positive pattern is required",
                self.path.display()
            ));
        }

        // First matching pattern (argument order) decides a file's key form;
        // BTreeMap dedups files matched by several patterns.
        let mut matched: BTreeMap<Vec<String>, KeyForm> = BTreeMap::new();
        for pattern in &positives {
            let mut files = Vec::new();
            collect_matches(&pattern.segments, &mut files);
            for file in files {
                // Vite excludes the importing module itself from its own globs.
                if file == importer {
                    continue;
                }
                matched.entry(file).or_insert(pattern.form);
            }
        }

        let mut entries = Vec::with_capacity(matched.len());
        for (file, form) in matched {
            if negatives
                .iter()
                .any(|negative| segments_match(&negative.segments, &file))
            {
                continue;
            }
            let specifier = relative_reference(&importer_directory, &file);
            let key = match form {
                KeyForm::Relative => specifier.clone(),
                // A `/`-pattern only yields files under root, so the strip holds.
                KeyForm::Root => format!("/{}", file[root.len()..].join("/")),
            };
            entries.push((key, specifier));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries)
    }

    /// Parses one pattern into absolute path segments (wildcards left in place).
    fn parse_pattern(
        &self,
        pattern: &str,
        importer_directory: &[String],
        root: &[String],
    ) -> Result<ParsedPattern, String> {
        let file = self.path.display();
        let (negative, rest) = match pattern.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, pattern),
        };
        if let Some(unsupported) =
            rest.chars().find(|c| matches!(c, '?' | '[' | ']' | '{' | '}' | '(' | ')' | '\\'))
        {
            return Err(format!(
                "{file}: import.meta.glob pattern {pattern:?} uses unsupported glob \
                 syntax {unsupported:?} (supported: `*` and `**`)"
            ));
        }
        let (base, form, body) = if let Some(body) = rest.strip_prefix('/') {
            (root, KeyForm::Root, body)
        } else if rest.starts_with("./") || rest.starts_with("../") {
            (importer_directory, KeyForm::Relative, rest)
        } else if negative {
            // Vite allows bare negatives (`'!**/excluded.ts'`); they match
            // relative to the importing module, i.e. an implicit `./`.
            (importer_directory, KeyForm::Relative, rest)
        } else {
            return Err(format!(
                "{file}: import.meta.glob pattern {pattern:?} must start with './', \
                 '../', or '/' (bare specifiers cannot be globbed)"
            ));
        };

        let mut segments: Vec<String> = base.to_vec();
        let mut saw_wildcard = false;
        for segment in body.split('/') {
            match segment {
                "" | "." => {}
                ".." => {
                    if saw_wildcard {
                        return Err(format!(
                            "{file}: import.meta.glob pattern {pattern:?} has a '..' \
                             segment after a wildcard, which cannot be expanded"
                        ));
                    }
                    if segments.pop().is_none() {
                        return Err(format!(
                            "{file}: import.meta.glob pattern {pattern:?} escapes the \
                             filesystem root"
                        ));
                    }
                }
                segment => {
                    if segment.contains('*') {
                        saw_wildcard = true;
                    }
                    segments.push(segment.to_string());
                }
            }
        }
        Ok(ParsedPattern { negative, form, segments })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KeyForm {
    /// `./` / `../` pattern: keys are importer-relative paths as written.
    Relative,
    /// `/` pattern: keys are project-root-relative, `/`-prefixed.
    Root,
}

struct ParsedPattern {
    negative: bool,
    form: KeyForm,
    /// Absolute path segments (no leading `/` entry), wildcards in place.
    segments: Vec<String>,
}

/// The static import statement binding `identifier` for an eager match.
fn import_statement(identifier: &str, import: Option<&str>, specifier: &str) -> String {
    let specifier = json(specifier);
    match import {
        None | Some("*") => format!("import * as {identifier} from {specifier};"),
        Some("default") => format!("import {identifier} from {specifier};"),
        Some(name) if is_identifier(name) => {
            format!("import {{ {name} as {identifier} }} from {specifier};")
        }
        // Arbitrary export names use the ES2022 string-literal import form.
        Some(name) => format!("import {{ {} as {identifier} }} from {specifier};", json(name)),
    }
}

/// The lazy `() => import(...)` thunk for a match, selecting the requested export.
fn lazy_thunk(import: Option<&str>, specifier: &str) -> String {
    let specifier = json(specifier);
    match import {
        None | Some("*") => format!("() => import({specifier})"),
        Some(name) => format!("() => import({specifier}).then((m) => m[{}])", json(name)),
    }
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '$')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

/// Whether `expression` is the `import.meta.glob` member.
fn is_import_meta_glob(expression: &Expression<'_>) -> bool {
    matches!(
        expression,
        Expression::StaticMemberExpression(member)
            if member.property.name == "glob"
                && matches!(
                    &member.object,
                    Expression::MetaProperty(meta)
                        if meta.meta.name == "import" && meta.property.name == "meta"
                )
    )
}

/// The absolute path as plain segments (`/a/b/c.ts` -> `["a","b","c.ts"]`).
fn absolute_segments(path: &Path) -> Vec<String> {
    use std::path::Component;
    let mut segments = Vec::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::Prefix(_) | Component::CurDir => {}
            Component::ParentDir => {
                segments.pop();
            }
            Component::Normal(segment) => segments.push(segment.to_string_lossy().into_owned()),
        }
    }
    segments
}

fn path_of(segments: &[String]) -> PathBuf {
    PathBuf::from(format!("/{}", segments.join("/")))
}

/// Walks the filesystem collecting files that match `pattern` (absolute segments
/// with `*`/`**` wildcards). A non-existent directory simply matches nothing.
fn collect_matches(pattern: &[String], out: &mut Vec<Vec<String>>) {
    // Anchor the walk at the fixed (wildcard-free) prefix so `./dir/*.ts` never
    // scans outside `dir`.
    let first_wildcard = pattern.iter().position(|segment| segment.contains('*'));
    let Some(first_wildcard) = first_wildcard else {
        // A literal path with no wildcard: it matches iff the file exists.
        if path_of(pattern).is_file() {
            out.push(pattern.to_vec());
        }
        return;
    };
    walk_directory(&pattern[..first_wildcard], &pattern[first_wildcard..], out);
}

/// Recursively matches `pattern` segments against entries under `directory`.
fn walk_directory(directory: &[String], pattern: &[String], out: &mut Vec<Vec<String>>) {
    let Some(segment) = pattern.first() else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(path_of(directory)) else {
        return;
    };
    let rest = &pattern[1..];
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        let mut child = directory.to_owned();
        child.push(name.clone());
        if segment == "**" {
            // `**` spans zero or more directories: descend keeping the `**`, and
            // also try the rest of the pattern at this level.
            if path.is_dir() {
                walk_directory(&child, pattern, out);
            }
            if rest.is_empty() {
                if path.is_file() {
                    out.push(child);
                }
            } else if segment_matches(&rest[0], &name) {
                if rest.len() == 1 {
                    if path.is_file() {
                        out.push(child);
                    }
                } else if path.is_dir() {
                    walk_directory(&child, &rest[1..], out);
                }
            }
        } else if segment_matches(segment, &name) {
            if rest.is_empty() {
                if path.is_file() {
                    out.push(child);
                }
            } else if path.is_dir() {
                walk_directory(&child, rest, out);
            }
        }
    }
}

/// Pure segment-list match (no filesystem), used for negative patterns.
fn segments_match(pattern: &[String], path: &[String]) -> bool {
    match pattern.first() {
        None => path.is_empty(),
        Some(segment) if segment == "**" => {
            // Zero directories, or consume one path segment and keep `**`.
            segments_match(&pattern[1..], path)
                || (!path.is_empty() && segments_match(pattern, &path[1..]))
        }
        Some(segment) => {
            !path.is_empty()
                && segment_matches(segment, &path[0])
                && segments_match(&pattern[1..], &path[1..])
        }
    }
}

/// `*`-wildcard match of one segment (no `/` crossing), iterative backtracking.
fn segment_matches(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    let (mut p, mut n) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while n < name.len() {
        if p < pattern.len() && pattern[p] == '*' {
            star = Some((p, n));
            p += 1;
        } else if p < pattern.len() && pattern[p] == name[n] {
            p += 1;
            n += 1;
        } else if let Some((star_p, star_n)) = star {
            p = star_p + 1;
            n = star_n + 1;
            star = Some((star_p, star_n + 1));
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == '*' {
        p += 1;
    }
    p == pattern.len()
}

/// `to` expressed relative to the `from` directory, always `./`- or
/// `../`-prefixed so the resolver treats it as a relative import.
fn relative_reference(from: &[String], to: &[String]) -> String {
    let common = from
        .iter()
        .zip(to.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let ups = from.len() - common;
    let rest = to[common..].join("/");
    if ups == 0 {
        format!("./{rest}")
    } else {
        format!("{}{rest}", "../".repeat(ups))
    }
}

fn json(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a JavaScript string cannot fail")
}

/// Applies non-overlapping `(span, replacement)` edits. Sorted by `(start, end)`
/// so the zero-length prelude insertion at an offset a call replacement also
/// starts at is emitted before, not skipped as an overlap.
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

    fn fixture() -> (tempfile::TempDir, PathBuf, ImportMetaGlob) {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("src/widgets")).unwrap();
        std::fs::write(root.join("src/widgets/alpha.ts"), "export default 'alpha';\n").unwrap();
        std::fs::write(root.join("src/widgets/beta.ts"), "export default 'beta';\n").unwrap();
        std::fs::write(root.join("src/widgets/skip.ts"), "export default 'skip';\n").unwrap();
        std::fs::write(root.join("src/entry.ts"), "").unwrap();
        let importer = root.join("src/entry.ts");
        let options = ImportMetaGlob { root: root.clone() };
        (directory, importer, options)
    }

    fn rewrite(source: &str) -> Result<Option<String>, String> {
        let (_directory, importer, options) = fixture();
        transform(&importer, source, &options)
    }

    #[test]
    fn lazy_glob_expands_to_sorted_dynamic_import_thunks() {
        let out = rewrite("const modules = import.meta.glob('./widgets/*.ts');")
            .unwrap()
            .unwrap();
        let alpha = out.find("\"./widgets/alpha.ts\": () => import(\"./widgets/alpha.ts\")");
        let beta = out.find("\"./widgets/beta.ts\": () => import(\"./widgets/beta.ts\")");
        assert!(alpha.is_some() && beta.is_some(), "{out}");
        assert!(alpha < beta, "keys are sorted: {out}");
    }

    #[test]
    fn eager_glob_hoists_static_imports_after_the_directive_prologue() {
        let out = rewrite(
            "'use client';\nexport const modules = import.meta.glob('./widgets/alpha.ts', { eager: true });",
        )
        .unwrap()
        .unwrap();
        assert!(out.starts_with("'use client';\n"), "{out}");
        assert!(
            out.contains("import * as __diffpack_glob_0_0 from \"./widgets/alpha.ts\";"),
            "{out}"
        );
        assert!(out.contains("\"./widgets/alpha.ts\": __diffpack_glob_0_0"), "{out}");
    }

    #[test]
    fn import_option_selects_the_export_in_both_modes() {
        let eager = rewrite(
            "const m = import.meta.glob('./widgets/alpha.ts', { eager: true, import: 'default' });",
        )
        .unwrap()
        .unwrap();
        assert!(eager.contains("import __diffpack_glob_0_0 from \"./widgets/alpha.ts\";"), "{eager}");
        let lazy =
            rewrite("const m = import.meta.glob('./widgets/alpha.ts', { import: 'setup' });")
                .unwrap()
                .unwrap();
        assert!(
            lazy.contains("() => import(\"./widgets/alpha.ts\").then((m) => m[\"setup\"])"),
            "{lazy}"
        );
    }

    #[test]
    fn query_appends_the_loader_suffix_to_each_specifier() {
        let out = rewrite("const m = import.meta.glob('./widgets/*.ts', { query: '?raw' });")
            .unwrap()
            .unwrap();
        assert!(out.contains("import(\"./widgets/alpha.ts?raw\")"), "{out}");
        // The map key stays the plain path, as in Vite.
        assert!(out.contains("\"./widgets/alpha.ts\": "), "{out}");
    }

    #[test]
    fn root_relative_pattern_keys_are_root_relative() {
        let out = rewrite("const m = import.meta.glob('/src/widgets/alpha.ts');")
            .unwrap()
            .unwrap();
        assert!(
            out.contains("\"/src/widgets/alpha.ts\": () => import(\"./widgets/alpha.ts\")"),
            "{out}"
        );
    }

    #[test]
    fn negative_pattern_excludes_and_valid_empty_match_yields_an_empty_object() {
        let out = rewrite("const m = import.meta.glob(['./widgets/*.ts', '!**/skip.ts']);")
            .unwrap()
            .unwrap();
        assert!(!out.contains("skip.ts"), "{out}");
        assert!(out.contains("alpha.ts") && out.contains("beta.ts"), "{out}");

        let empty = rewrite("const m = import.meta.glob('./nothing-here/*.ts');")
            .unwrap()
            .unwrap();
        assert!(empty.contains("const m = ({})"), "{empty}");
    }

    #[test]
    fn non_literal_pattern_is_a_hard_error_naming_the_file() {
        let error = rewrite("const m = import.meta.glob(pattern);").unwrap_err();
        assert!(error.contains("entry.ts"), "{error}");
        assert!(error.contains("string literal"), "{error}");
        let error = rewrite("const m = import.meta.glob(`./widgets/*.ts`);").unwrap_err();
        assert!(error.contains("string literal"), "{error}");
        let error = rewrite("const m = import.meta.glob(['./widgets/*.ts', other]);").unwrap_err();
        assert!(error.contains("string literal"), "{error}");
    }

    #[test]
    fn bare_specifier_pattern_is_a_hard_error_naming_the_file() {
        let error = rewrite("const m = import.meta.glob('widgets/*.ts');").unwrap_err();
        assert!(error.contains("entry.ts"), "{error}");
        assert!(error.contains("must start with './', '../', or '/'"), "{error}");
    }

    #[test]
    fn unsupported_options_are_hard_errors_naming_the_file() {
        let error =
            rewrite("const m = import.meta.glob('./widgets/*.ts', { unknown: 1 });").unwrap_err();
        assert!(error.contains("entry.ts"), "{error}");
        assert!(error.contains("unknown import.meta.glob option `unknown`"), "{error}");

        let error = rewrite("const m = import.meta.glob('./widgets/*.ts', { exhaustive: true });")
            .unwrap_err();
        assert!(error.contains("`exhaustive` is not supported"), "{error}");

        let error =
            rewrite("const m = import.meta.glob('./widgets/*.ts', { eager: flag });").unwrap_err();
        assert!(error.contains("`eager` must be a boolean literal"), "{error}");

        let error = rewrite("const m = import.meta.glob('./widgets/*.ts', { query: '?worker' });")
            .unwrap_err();
        assert!(error.contains("entry.ts"), "{error}");
        assert!(error.contains("\"?worker\" is not supported"), "{error}");
    }

    #[test]
    fn only_negative_patterns_is_a_hard_error() {
        let error = rewrite("const m = import.meta.glob('!./widgets/*.ts');").unwrap_err();
        assert!(error.contains("at least one positive pattern"), "{error}");
    }

    #[test]
    fn a_module_without_glob_calls_is_untouched() {
        assert_eq!(rewrite("export const x = import.meta.url;").unwrap(), None);
        assert_eq!(rewrite("export const x = 1;").unwrap(), None);
    }

    #[test]
    fn segment_wildcards_match_within_one_segment_only() {
        assert!(segment_matches("*.ts", "a.ts"));
        assert!(segment_matches("*.module.css", "button.module.css"));
        assert!(!segment_matches("*.ts", "a.tsx"));
        assert!(segment_matches("a*b*c", "aXbYc"));
        assert!(!segment_matches("a*b", "acb-d"));
        assert!(segment_matches("*", "anything"));
    }
}
