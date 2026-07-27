// Which Tailwind default theme a v4 entry compiles against.
//
// Tailwind's default tokens change between v4 releases (`--font-sans` did), so the
// compile must use the theme the app ACTUALLY has installed, not diffpack's vendored
// copy. Diffpack looked that up by joining `node_modules/tailwindcss` onto the
// CANDIDATE SCAN ROOT — the directory the entry declares via `@import 'tailwindcss'
// source(...)`, a source-tree concept unrelated to Node module resolution. It happened
// to work when the two coincided and silently fell back to the vendored theme when they
// did not: TanStack Start's `src/styles/app.css` with `source('../')` scans `src/`,
// which holds no `node_modules`, so the whole app shipped stale theme tokens.
//
// These tests pin the MECHANISM, not the token values: the installed theme carries a
// sentinel token that exists in no released Tailwind, so the assertion cannot be
// satisfied by re-copying an upstream file.
//
// Soft-skips when `target/release/diffpack` has not been built (same convention as
// tests/out_dir.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;

/// A theme token no released Tailwind defines. Present in the emitted stylesheet only
/// if the compile read the project's INSTALLED `theme.css`.
const SENTINEL_COLOR: &str = "#123456";

fn release_binary() -> Option<PathBuf> {
    let binary = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/diffpack");
    binary.is_file().then_some(binary)
}

/// Writes a project whose Tailwind entry lives at `css_rel` (relative to the root) and
/// declares `source('../')`, plus a module that applies `bg-sentinel`.
fn write_project(root: &Path, css_rel: &str, installed_tailwind: bool) {
    std::fs::write(
        root.join("index.html"),
        "<!doctype html><html><head><title>tailwind-theme</title></head>\
         <body><div id=\"app\"></div>\
         <script type=\"module\" src=\"/src/main.js\"></script></body></html>",
    )
    .unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    let css_path = root.join(css_rel);
    let css_dir = css_path.parent().unwrap();
    std::fs::create_dir_all(css_dir).unwrap();
    // The entry's own directory relative to `src/main.js`, so the import resolves
    // whichever layout the case uses.
    let import_specifier = format!(
        "./{}",
        css_rel.strip_prefix("src/").expect("the entry lives under src/")
    );
    std::fs::write(
        root.join("src/main.js"),
        format!(
            "import {import_specifier:?};\n\
             document.getElementById('app').className = 'bg-sentinel';\n"
        ),
    )
    .unwrap();
    std::fs::write(&css_path, "@import 'tailwindcss' source('../');\n").unwrap();

    if installed_tailwind {
        let package = root.join("node_modules/tailwindcss");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(
            package.join("package.json"),
            format!(
                "{{\"name\":\"tailwindcss\",\"version\":{:?}}}\n",
                diffpack::tailwind::VERSION
            ),
        )
        .unwrap();
        // The full default scale (so ordinary utilities still resolve) plus one token
        // that only this installed copy has.
        std::fs::write(
            package.join("theme.css"),
            format!(
                "{}\n@theme {{\n  --color-sentinel: {SENTINEL_COLOR};\n}}\n",
                diffpack::tailwind::vendored_theme_css()
            ),
        )
        .unwrap();
    }
}

/// Builds `root` and returns the emitted stylesheet's text.
fn build_and_read_css(binary: &Path, root: &Path) -> String {
    let output = Command::new(binary)
        .arg("build")
        .arg(root)
        .arg("--out-dir")
        .arg("dist")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "build failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mut css = String::new();
    collect_css(&root.join("dist"), &mut css);
    assert!(!css.is_empty(), "the build emitted no stylesheet");
    css
}

fn collect_css(dir: &Path, out: &mut String) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_css(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("css") {
            out.push_str(&std::fs::read_to_string(&path).unwrap());
            out.push('\n');
        }
    }
}

#[test]
fn installed_theme_is_used_when_node_modules_is_above_the_scan_root() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let project = tempdir().unwrap();
    // The TanStack Start layout: the entry is two levels down, so `source('../')`
    // names `src/` — which has no `node_modules`.
    write_project(project.path(), "src/styles/app.css", true);

    let css = build_and_read_css(&binary, project.path());

    assert!(
        css.contains(SENTINEL_COLOR),
        "the compile must use the project's INSTALLED tailwindcss/theme.css, resolved by \
         walking up from the stylesheet; it fell back to the vendored theme instead.\n\
         emitted stylesheet:\n{css}"
    );
}

#[test]
fn installed_theme_is_used_when_node_modules_sits_at_the_scan_root() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let project = tempdir().unwrap();
    // The TanStack Router quickstart layout: `source('../')` from `src/styles.css`
    // names the project root, which DOES hold `node_modules`. This case already
    // worked and must keep working.
    write_project(project.path(), "src/styles.css", true);

    let css = build_and_read_css(&binary, project.path());

    assert!(
        css.contains(SENTINEL_COLOR),
        "the installed theme must still be used when the scan root is the package root.\n\
         emitted stylesheet:\n{css}"
    );
}

#[test]
fn vendored_theme_is_the_fallback_when_nothing_is_installed() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let project = tempdir().unwrap();
    write_project(project.path(), "src/styles/app.css", false);

    let css = build_and_read_css(&binary, project.path());

    assert!(
        !css.contains(SENTINEL_COLOR),
        "no tailwindcss is installed; the sentinel token cannot appear"
    );
    // The vendored default scale is what compiled instead — the preflight it carries
    // is proof the theme resolved at all rather than the entry being copied through.
    assert!(
        css.contains("--default-font-family"),
        "the vendored default theme must be the fallback.\nemitted stylesheet:\n{css}"
    );
}
