//! Oracle harness for the Rust JSIR reimplementation.
//!
//! Upstream ships golden conversion fixtures under
//! `vendor/jsir-upstream/maldoca/js/ir/conversion/tests/<case>/` as **LLVM
//! FileCheck** files: each line of real tool output is prefixed with
//! `// <PREFIX>:`, `// <PREFIX>-NEXT:` or `// <PREFIX>-EMPTY:` (no regex
//! patterns are used — see upstream `generate_tests.py`). This crate
//! reconstructs the byte-exact expected output by stripping those prefixes,
//! enumerates the corpus, and offers a byte-diff used by every milestone gate.

use std::path::{Path, PathBuf};

/// Locate the workspace root (the directory containing the top-level
/// `Cargo.toml` and `vendor/`). Derived from this crate's manifest dir.
pub fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/jsir-oracle
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent() // crates/
        .and_then(Path::parent) // root
        .expect("manifest dir should have two ancestors")
        .to_path_buf()
}

/// Path to the upstream conversion-test corpus.
pub fn conversion_tests_dir() -> PathBuf {
    workspace_root().join("vendor/jsir-upstream/maldoca/js/ir/conversion/tests")
}

/// A single golden conversion fixture (one directory under the corpus).
#[derive(Debug, Clone)]
pub struct Fixture {
    pub name: String,
    pub dir: PathBuf,
}

impl Fixture {
    fn read_check_file(&self, file: &str) -> Option<String> {
        let path = self.dir.join(file);
        let raw = std::fs::read_to_string(&path).ok()?;
        Some(strip_filecheck(&raw))
    }

    /// The hand-written input source for this case.
    pub fn input_js(&self) -> std::io::Result<String> {
        std::fs::read_to_string(self.dir.join("input.js"))
    }

    /// Expected JSIR AST JSON (output of `source2ast`), byte-exact.
    pub fn expected_ast_json(&self) -> Option<String> {
        self.read_check_file("ast.json")
    }

    /// Expected JSHIR text (output of `source2ast,ast2hir`), byte-exact.
    pub fn expected_jshir(&self) -> Option<String> {
        self.read_check_file("jshir.mlir")
    }

    /// Expected re-emitted source (output of the full round-trip), byte-exact.
    pub fn expected_output_js(&self) -> Option<String> {
        self.read_check_file("output.js")
    }
}

/// Enumerate every fixture directory in the corpus (those containing
/// `input.js`), sorted by name for deterministic test ordering.
pub fn list_fixtures() -> Vec<Fixture> {
    let dir = conversion_tests_dir();
    let mut fixtures = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return fixtures;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("input.js").is_file() {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            fixtures.push(Fixture { name, dir: path });
        }
    }
    fixtures.sort_by(|a, b| a.name.cmp(&b.name));
    fixtures
}

/// Reconstruct byte-exact tool output from an LLVM FileCheck golden file.
///
/// Layout produced by upstream `generate_tests.py`:
/// - line 0:        `// <PREFIX>:      <content>`  (colon + 6 spaces of padding)
/// - lines 1..N:    `// <PREFIX>-NEXT: <content>`  (`-NEXT:` + 1 space)
/// - empty lines:   `// <PREFIX>-EMPTY:`           (no content)
///
/// In every variant the marker that follows `<PREFIX>` is exactly 7 characters
/// (`:` + 6 spaces, `-NEXT:` + 1 space, or `-EMPTY:`), so content always begins
/// at a constant column: `len("// ") + PREFIX.len() + 7`.
pub fn strip_filecheck(raw: &str) -> String {
    // Detect the check prefix from the first non-empty line: it is the run of
    // characters after "// " up to the first ':' or '-'.
    let prefix_len = raw
        .lines()
        .find(|l| l.starts_with("// "))
        .map(|l| {
            l[3..]
                .find(|c| c == ':' || c == '-')
                .expect("FileCheck line must contain a marker")
        })
        .unwrap_or(0);

    let offset = 3 + prefix_len + 7;

    let mut out = String::new();
    let mut first = true;
    for line in raw.lines() {
        // Skip anything that isn't a check line (e.g. stray blank lines).
        if !line.starts_with("// ") {
            continue;
        }
        if !first {
            out.push('\n');
        }
        first = false;
        if line.len() > offset {
            out.push_str(&line[offset..]);
        }
        // else: an `-EMPTY:` marker -> empty line (nothing to push).
    }
    out
}

/// Produce a compact line-oriented diff between expected and actual. Returns
/// `None` when the two are byte-identical.
pub fn byte_diff(expected: &str, actual: &str) -> Option<String> {
    if expected == actual {
        return None;
    }
    let mut report = String::new();
    let exp: Vec<&str> = expected.lines().collect();
    let act: Vec<&str> = actual.lines().collect();
    let max = exp.len().max(act.len());
    for i in 0..max {
        let e = exp.get(i).copied();
        let a = act.get(i).copied();
        if e != a {
            report.push_str(&format!("line {}:\n", i + 1));
            report.push_str(&format!("  expected: {}\n", e.map(escape).unwrap_or_else(|| "<eof>".into())));
            report.push_str(&format!("  actual:   {}\n", a.map(escape).unwrap_or_else(|| "<eof>".into())));
        }
    }
    if exp.len() == act.len() && report.is_empty() {
        // Differs only in trailing newline / whitespace not visible line-by-line.
        report.push_str(&format!(
            "byte length differs: expected {} bytes, actual {} bytes (trailing whitespace?)\n",
            expected.len(),
            actual.len()
        ));
    }
    Some(report)
}

fn escape(s: &str) -> String {
    format!("{:?}", s)
}

/// Canonicalize a line the way LLVM FileCheck does by default (i.e. without
/// `--strict-whitespace`): trim leading/trailing horizontal whitespace and
/// collapse every internal run of spaces/tabs to a single space.
fn filecheck_canonical_line(line: &str) -> String {
    let trimmed = line.trim_matches([' ', '\t']);
    let mut out = String::with_capacity(trimmed.len());
    let mut prev_ws = false;
    for ch in trimmed.chars() {
        if ch == ' ' || ch == '\t' {
            if !prev_ws {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out
}

/// True iff `actual` would pass upstream's FileCheck verification of `expected`.
///
/// The upstream conversion tests run `FileCheck` WITHOUT `--strict-whitespace`
/// (see each `run.lit`), so horizontal-whitespace runs are canonicalized before
/// comparison. This models that exactly (line-by-line, since the golden files
/// hold full output lines). Passing this is the real definition of jsir
/// compatibility; byte-exactness is a strictly stronger property we also track.
pub fn filecheck_equivalent(expected: &str, actual: &str) -> bool {
    let e: Vec<String> = expected.lines().map(filecheck_canonical_line).collect();
    let a: Vec<String> = actual.lines().map(filecheck_canonical_line).collect();
    e == a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_present() {
        let fixtures = list_fixtures();
        assert!(
            fixtures.len() >= 40,
            "expected the full conversion corpus (~46 cases), found {}",
            fixtures.len()
        );
    }

    #[test]
    fn every_fixture_has_golden_files_that_strip_cleanly() {
        for f in list_fixtures() {
            // Each case must at least have input.js + ast.json + jshir.mlir.
            assert!(f.input_js().is_ok(), "{}: missing input.js", f.name);
            let ast = f
                .expected_ast_json()
                .unwrap_or_else(|| panic!("{}: missing ast.json", f.name));
            let hir = f
                .expected_jshir()
                .unwrap_or_else(|| panic!("{}: missing jshir.mlir", f.name));
            assert!(!ast.trim().is_empty(), "{}: empty ast.json", f.name);
            assert!(!hir.trim().is_empty(), "{}: empty jshir.mlir", f.name);
        }
    }

    #[test]
    fn strip_reconstructs_known_lines() {
        // From load_store_identifier/jshir.mlir.
        let raw = "\
// JSHIR:      \"jsir.file\"() <{comments = []}> ({
// JSHIR-NEXT:   \"jsir.program\"() <{source_type = \"script\"}> ({
// JSHIR-NEXT:     %0 = \"jsir.identifier_ref\"() <{name = \"a\"}> : () -> !jsir.any
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()";
        let out = strip_filecheck(raw);
        let mut lines = out.lines();
        assert_eq!(lines.next().unwrap(), "\"jsir.file\"() <{comments = []}> ({");
        assert_eq!(
            lines.next().unwrap(),
            "  \"jsir.program\"() <{source_type = \"script\"}> ({"
        );
        assert_eq!(
            lines.next().unwrap(),
            "    %0 = \"jsir.identifier_ref\"() <{name = \"a\"}> : () -> !jsir.any"
        );
        assert_eq!(lines.next().unwrap(), "  }, {");
        assert_eq!(lines.next().unwrap(), "  ^bb0:");
        assert_eq!(lines.next().unwrap(), "  }) : () -> ()");
        assert_eq!(lines.next().unwrap(), "}) : () -> ()");
        assert!(lines.next().is_none());
    }

    #[test]
    fn filecheck_canonicalizes_whitespace() {
        // FileCheck (no --strict-whitespace) treats whitespace runs as equal.
        assert!(filecheck_equivalent(
            "  id = #jsir<identifier <L 1 C 0>",
            "  id = #jsir<identifier   <L 1 C 0>"
        ));
        // ...but it does not ignore non-whitespace differences.
        assert!(!filecheck_equivalent(
            "id = #jsir<identifier <L 1 C 0>",
            "id = #jsir<identifier <L 1 C 1>"
        ));
    }

    #[test]
    fn strip_handles_empty_marker() {
        let raw = "// AST:      {\n// AST-EMPTY:\n// AST-NEXT: }";
        let out = strip_filecheck(raw);
        assert_eq!(out, "{\n\n}");
    }
}
