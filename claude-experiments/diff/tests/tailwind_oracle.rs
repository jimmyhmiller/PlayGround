//! Tailwind compile oracle: the reference Vite/Tailwind build's extracted
//! `app-*.css` is a byte-level spec. This test compiles the pinned app's
//! `src/styles/app.css` natively (scanning the app source for candidate classes,
//! exactly as the emit step does) and asserts that every utility class the app
//! uses yields the *same declarations* as the reference stylesheet.
//!
//! The comparison is per-class and declaration-set based (normalized for
//! whitespace and leading zeros, order-independent within a rule), because the
//! reference is minified by lightningcss while diffpack emits its own formatting;
//! the semantic declarations must match exactly.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use diffpack::tailwind::{compile, scan_class_candidates};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("integration/tanstack-start-reference")
}

/// Recursively scans a directory for class candidates (JS/TS/JSX), skipping
/// node_modules and dot-directories — the same policy as the emit step.
fn scan_dir(dir: &std::path::Path, out: &mut BTreeSet<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }
        if path.is_dir() {
            scan_dir(&path, out);
        } else if matches!(
            path.extension().and_then(|v| v.to_str()),
            Some("js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "html")
        ) && let Ok(src) = fs::read_to_string(&path)
        {
            scan_class_candidates(&src, out);
        }
    }
}

/// Normalizes a declaration block into a comparable, order-independent key.
fn normalize_decls(body: &str) -> String {
    let mut decls: Vec<String> = body
        .split(';')
        .map(|d| {
            d.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                // Strip leading zeros (`0.17` -> `.17`, `0.25rem` -> `.25rem`) so
                // lightningcss's minified numbers compare equal.
                .replace("0.", ".")
        })
        .filter(|d| !d.is_empty())
        .collect();
    decls.sort();
    decls.join(";")
}

/// Extracts a map from class name -> normalized declarations for every
/// `.class{…}`, `.class:hover{…}`, and `:where(.class>…){…}` rule in a stylesheet.
fn class_declaration_map(css: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let bytes = css.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Find a class or :where selector followed by a `{`.
        if bytes[i] == b'.' || css[i..].starts_with(":where(.") {
            // Read the selector up to the next `{` (no nested braces in selectors).
            if let Some(rel) = css[i..].find('{') {
                let selector = &css[i..i + rel];
                if let Some(open) = css[i..].find('{') {
                    let block_start = i + open + 1;
                    if let Some(close_rel) = css[block_start..].find('}') {
                        let body = &css[block_start..block_start + close_rel];
                        // Only record simple declaration blocks (no nested rules).
                        if !body.contains('{')
                            && let Some(class) = extract_class_name(selector)
                        {
                            map.insert(class, normalize_decls(body));
                        }
                        i = block_start + close_rel + 1;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    map
}

/// Extracts the (unescaped) class name from a selector like `.p-2`,
/// `.hover\:text-blue-600:hover`, or `:where(.space-y-2>:not(:last-child))`.
fn extract_class_name(selector: &str) -> Option<String> {
    let start = selector.find('.')? + 1;
    let rest = &selector[start..];
    let mut name = String::new();
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(&next) = chars.peek() {
                    name.push(next);
                    chars.next();
                }
            }
            // A pseudo-class / combinator / grouping char terminates the name.
            ':' | '>' | ' ' | ')' | '~' | '+' | '[' | ',' => break,
            _ => name.push(c),
        }
    }
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[test]
fn compiled_app_css_matches_the_reference_declarations_per_class() {
    let fixture = fixture();
    let app_css_path = fixture.join("src/styles/app.css");
    let reference_css_path = fixture.join(".output/public/assets/app-CgRaPnL3.css");
    if !app_css_path.exists() || !reference_css_path.exists() {
        // The reference Vite build must be present (`npm ci && npm run build`).
        eprintln!("skipping: reference build artifacts not present");
        return;
    }

    let app_css = fs::read_to_string(&app_css_path).unwrap();
    let reference = fs::read_to_string(&reference_css_path).unwrap();

    let mut candidates = BTreeSet::new();
    scan_dir(&fixture.join("src"), &mut candidates);

    // The app must use a non-trivial set of classes (guards against a broken scan
    // silently matching nothing).
    assert!(
        candidates.len() >= 30,
        "scanned only {} candidate classes; the source scan is broken",
        candidates.len()
    );

    let compiled = compile(&app_css, &candidates).expect("native compile must succeed");

    let reference_map = class_declaration_map(&reference);
    let compiled_map = class_declaration_map(&compiled);

    // Every class the app uses (present in the reference utilities layer) must
    // exist in diffpack's output with identical declarations.
    let mut checked = 0;
    let mut mismatches = Vec::new();
    for class in &candidates {
        // The escaped class name in a selector matches the raw candidate.
        let Some(reference_decls) = reference_map.get(class) else {
            // A candidate the reference did not emit (shouldn't happen for this
            // app) is out of scope for the per-class spec comparison.
            continue;
        };
        checked += 1;
        match compiled_map.get(class) {
            None => mismatches.push(format!("MISSING `{class}` (reference: {reference_decls})")),
            Some(mine) if mine != reference_decls => mismatches.push(format!(
                "DIFF `{class}`\n  reference: {reference_decls}\n  diffpack:  {mine}"
            )),
            _ => {}
        }
    }

    assert!(
        checked >= 30,
        "compared only {checked} classes against the reference; expected the full app class set"
    );
    assert!(
        mismatches.is_empty(),
        "{} class(es) diverged from the reference spec:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
fn compiled_app_css_has_no_fetchable_tailwindcss_import() {
    let fixture = fixture();
    let app_css_path = fixture.join("src/styles/app.css");
    if !app_css_path.exists() {
        return;
    }
    let app_css = fs::read_to_string(&app_css_path).unwrap();
    let mut candidates = BTreeSet::new();
    scan_dir(&fixture.join("src"), &mut candidates);
    let compiled = compile(&app_css, &candidates).unwrap();
    assert!(
        !compiled.to_lowercase().contains("@import"),
        "the compiled stylesheet must not contain an @import (that 404s in the browser)"
    );
    assert!(!compiled.contains("tailwindcss'"));
}
