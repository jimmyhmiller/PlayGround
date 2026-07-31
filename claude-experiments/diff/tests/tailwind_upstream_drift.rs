// Drift guard for the vendored upstream Tailwind data.
//
// Diffpack reimplements Tailwind v4's compile natively, driven by data lifted verbatim
// from the `tailwindcss` package (`src/tailwind_theme.css`,
// `src/tailwind_preflight.source.css`) plus the compiled preflight extracted from a real
// `@tailwindcss/vite` build (`src/tailwind_preflight.css`). Copied data drifts silently:
// upstream changed `--font-sans` and every app diffpack built kept shipping the old font
// stack, which only a real browser reading `getComputedStyle` ever noticed.
//
// Nothing here compares CLASS output — that is tests/tailwind_oracle.rs's job, and it
// structurally excludes the theme `:root` block and the preflight. This file guards
// exactly the surface that oracle cannot see.
//
// When tailwindcss is not installed, T2-T5 SKIP LOUDLY (never silently pass), and
// `DIFFPACK_REQUIRE_UPSTREAM=1` promotes every skip to a failure. T1 needs nothing but
// checked-in files, so no configuration leaves the guard entirely inert.

use std::path::{Path, PathBuf};

/// Why T2-T4 cannot run, and what supplies it.
const NO_UPSTREAM: &str = "no tailwindcss install found; run 'npm ci' in \
     integration/tanstack-start-reference (or set DIFFPACK_TAILWIND_UPSTREAM to a \
     tailwindcss package directory)";

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The checked-in fixture whose lockfile pins the Tailwind release diffpack vendors.
fn reference_fixture() -> PathBuf {
    manifest_dir().join("integration/tanstack-start-reference")
}

/// Reports that an upstream-requiring test could not run. Prints an unmistakable line
/// naming what is missing and the command that supplies it; panics instead when
/// `DIFFPACK_REQUIRE_UPSTREAM=1` (what CI and `check.sh` set), so "not installed" can
/// never read as green there.
fn skip(test: &str, reason: &str) {
    let message = format!("SKIP tailwind_upstream_drift::{test} — {reason}");
    if std::env::var_os("DIFFPACK_REQUIRE_UPSTREAM").is_some_and(|v| v == "1") {
        panic!("{message}");
    }
    eprintln!("{message}");
}

/// Locates an installed `tailwindcss` package to compare against.
///
/// `DIFFPACK_TAILWIND_UPSTREAM` wins; pointing it somewhere wrong is a HARD ERROR, not a
/// skip — an operator who configured the guard gets told the configuration is bad.
/// Otherwise the pinned reference fixture, then any v4 install in the e2e corpus.
fn upstream_tailwind_dir() -> Option<PathBuf> {
    if let Some(configured) = std::env::var_os("DIFFPACK_TAILWIND_UPSTREAM") {
        let dir = PathBuf::from(&configured);
        assert!(
            dir.join("theme.css").is_file(),
            "DIFFPACK_TAILWIND_UPSTREAM={} is not a tailwindcss package directory \
             (no theme.css in it)",
            dir.display()
        );
        return Some(dir);
    }
    let pinned = reference_fixture().join("node_modules/tailwindcss");
    if pinned.join("theme.css").is_file() {
        return Some(pinned);
    }
    let apps = std::fs::read_dir(manifest_dir().join("integration/e2e/apps")).ok()?;
    let mut candidates: Vec<PathBuf> = apps
        .flatten()
        .map(|entry| entry.path().join("node_modules/tailwindcss"))
        .filter(|dir| dir.join("theme.css").is_file())
        .filter(|dir| package_version(dir).is_some_and(|v| v.starts_with("4.")))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

fn package_version(package: &Path) -> Option<String> {
    let manifest = std::fs::read_to_string(package.join("package.json")).ok()?;
    // A hand-rolled read keeps this guard dependency-free; `"version"` appears once as a
    // top-level key in every npm manifest.
    let after = manifest.split_once("\"version\"")?.1;
    let open = after.find('"')? + 1;
    let rest = &after[open..];
    Some(rest[..rest.find('"')?].to_string())
}

/// The first differing byte offset plus surrounding context, so a failure says WHERE.
fn first_difference(vendored: &str, upstream: &str) -> String {
    let offset = vendored
        .as_bytes()
        .iter()
        .zip(upstream.as_bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| vendored.len().min(upstream.len()));
    format!(
        "first difference at byte {offset}\n  vendored: …{}…\n  upstream: …{}…",
        window(vendored, offset),
        window(upstream, offset)
    )
}

/// `text` around `offset`, clamped to char boundaries (the files are ASCII today, but a
/// panic inside a failure message would hide the failure).
fn window(text: &str, offset: usize) -> &str {
    let mut start = offset.saturating_sub(80);
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    let mut end = (offset + 160).min(text.len());
    while end > start && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[start.min(text.len())..end.max(start.min(text.len()))]
}

// --- T1: needs nothing beyond checked-in files -------------------------------------

#[test]
fn vendored_version_matches_the_pinned_lockfile() {
    let lock_path = reference_fixture().join("package-lock.json");
    let lock = std::fs::read_to_string(&lock_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", lock_path.display()));
    // The lockfile entry for the root `tailwindcss` dependency.
    let entry = lock
        .split_once("\"node_modules/tailwindcss\": {")
        .unwrap_or_else(|| {
            panic!(
                "{} has no `node_modules/tailwindcss` entry — the Tailwind pin moved and \
                 this guard's anchor must move with it",
                lock_path.display()
            )
        })
        .1;
    let after = entry.split_once("\"version\": \"").expect("a lockfile entry has a version").1;
    let pinned = &after[..after.find('"').unwrap()];

    let fixture = reference_fixture();
    let (lock, vendored) = (lock_path.display(), diffpack::tailwind::VERSION);
    assert_eq!(
        pinned, vendored,
        "{lock} pins tailwindcss v{pinned} but src/tailwind.rs vendors v{vendored}. \
         Re-vendor and bump `tailwind::VERSION`:\n  \
         cp {fixture}/node_modules/tailwindcss/theme.css     src/tailwind_theme.css\n  \
         cp {fixture}/node_modules/tailwindcss/preflight.css src/tailwind_preflight.source.css",
        fixture = fixture.display(),
    );
}

// --- T2-T5: need an installed tailwindcss ------------------------------------------

#[test]
fn upstream_version_matches_the_vendored_version() {
    let Some(upstream) = upstream_tailwind_dir() else {
        return skip("upstream_version_matches_the_vendored_version", NO_UPSTREAM);
    };
    let installed = package_version(&upstream)
        .unwrap_or_else(|| panic!("{}/package.json has no version", upstream.display()));
    assert_eq!(
        installed,
        diffpack::tailwind::VERSION,
        "the installed tailwindcss at {} is v{installed}, but src/tailwind.rs vendors \
         v{}. The byte comparisons below are only meaningful against the release the \
         banner claims, so re-vendor (see `tailwind::VERSION`) before trusting them.",
        upstream.display(),
        diffpack::tailwind::VERSION,
    );
}

#[test]
fn theme_css_is_verbatim_upstream() {
    let Some(upstream) = upstream_tailwind_dir() else {
        return skip("theme_css_is_verbatim_upstream", NO_UPSTREAM);
    };
    let path = upstream.join("theme.css");
    let expected = std::fs::read_to_string(&path).unwrap();
    let vendored = diffpack::tailwind::vendored_theme_css();
    assert!(
        vendored == expected,
        "src/tailwind_theme.css has drifted from upstream {}.\n{}\n\nrepair:\n  \
         cp {} src/tailwind_theme.css",
        path.display(),
        first_difference(vendored, &expected),
        path.display(),
    );
}

#[test]
fn preflight_source_is_verbatim_upstream() {
    let Some(upstream) = upstream_tailwind_dir() else {
        return skip("preflight_source_is_verbatim_upstream", NO_UPSTREAM);
    };
    let path = upstream.join("preflight.css");
    let expected = std::fs::read_to_string(&path).unwrap();
    let vendored = diffpack::tailwind::vendored_preflight_source_css();
    assert!(
        vendored == expected,
        "src/tailwind_preflight.source.css has drifted from upstream {}. The COMPILED \
         preflight src/tailwind_preflight.css was derived from the old source and is now \
         stale too — re-extract it (see the T5 test below).\n{}\n\nrepair:\n  \
         cp {} src/tailwind_preflight.source.css",
        path.display(),
        first_difference(vendored, &expected),
        path.display(),
    );
}

#[test]
fn v3_preflight_source_is_verbatim_upstream_v3() {
    // The legacy v3 dialect has its OWN base reset (v3 resets `border-color` to
    // gray-200, v4 to `currentColor`), vendored from a real tailwindcss@3 install in
    // the e2e corpus. Unlike the v4 preflight this needs no separately extracted
    // compiled form: v3 emits it through PostCSS with no lightningcss lowering, so the
    // engine resolves its `theme()` calls itself.
    let app = manifest_dir().join("integration/e2e/apps/next-blog-starter/node_modules/tailwindcss");
    let installed = app.join("package.json");
    let Ok(manifest) = std::fs::read_to_string(&installed) else {
        return skip(
            "v3_preflight_source_is_verbatim_upstream_v3",
            "no tailwindcss@3 install found; run 'npm ci' in \
             integration/e2e/apps/next-blog-starter",
        );
    };
    assert!(
        manifest.contains(&format!("\"version\": \"{}\"", diffpack::tailwind::V3_VERSION)),
        "integration/e2e/apps/next-blog-starter pins a tailwindcss other than the \
         vendored v{} — re-vendor src/tailwind_preflight_v3.source.css and bump \
         tailwind::V3_VERSION",
        diffpack::tailwind::V3_VERSION,
    );
    let path = app.join("src/css/preflight.css");
    let expected = std::fs::read_to_string(&path).unwrap();
    let vendored = diffpack::tailwind::vendored_preflight_v3_source_css();
    assert!(
        vendored == expected,
        "src/tailwind_preflight_v3.source.css has drifted from upstream {}.\n{}\n\nrepair:\n  \
         cp {} src/tailwind_preflight_v3.source.css",
        path.display(),
        first_difference(vendored, &expected),
        path.display(),
    );
}

#[test]
fn embedded_preflight_is_the_compiled_form_of_that_source() {
    // Upstream ships preflight as commented source; the browser gets it after
    // lightningcss has split `::file-selector-button` out of every selector list,
    // rewritten `::after` to `:after`, lowered `--theme(...)` to `var(...)` and
    // synthesized a `color-mix` fallback `@supports`. None of that is derivable in Rust,
    // so the only trustworthy oracle for the compiled form is a real reference build —
    // in which the embedded preflight must appear as a contiguous verbatim run.
    let reference = reference_fixture().join(".output/public/assets");
    let Some(stylesheet) = std::fs::read_dir(&reference).ok().and_then(|entries| {
        entries
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.extension().and_then(|e| e.to_str()) == Some("css"))
    }) else {
        return skip(
            "embedded_preflight_is_the_compiled_form_of_that_source",
            "no reference build stylesheet; run 'npm ci && npm run build' in integration/tanstack-start-reference",
        );
    };
    let built = std::fs::read_to_string(&stylesheet).unwrap();
    let vendored = diffpack::tailwind::vendored_preflight_css();
    assert!(
        built.contains(vendored),
        "src/tailwind_preflight.css is not a verbatim run of the reference build's \
         stylesheet {} — the compiled preflight has gone stale.\n\nrepair: rebuild the \
         reference (`npm ci && npm run build` in integration/tanstack-start-reference) \
         and lift the `@layer base{{…}}` run out of {} into src/tailwind_preflight.css.",
        stylesheet.display(),
        stylesheet.display(),
    );
}
