// Which file extensions may contain JSX is a per-project rule, not a global one.
//
// Next.js compiles JSX in `.js` (its SWC loader enables jsx for everything that is
// not a plain `.ts`), and real Next apps rely on it — `pages/index.js` is the
// default Next page. diffpack used to parse every `.js` as plain JavaScript, so
// those pages were a FATAL parse error: the module returned a dummy program, which
// also meant no dependencies, so the page's whole subtree vanished from the graph.
//
// Vite/esbuild deliberately go the other way: `.js` is plain JavaScript there, and
// JSX in it is a syntax error the user fixes by renaming the file. These tests pin
// BOTH halves, so "fix the Next case" can never quietly become "JSX everywhere".
//
// Soft-skips when `target/release/diffpack` has not been built (same convention as
// tests/out_dir.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;

fn release_binary() -> Option<PathBuf> {
    let binary = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/diffpack");
    binary.is_file().then_some(binary)
}

/// An HTML-rooted web app whose entry module is a `.js` file containing JSX.
fn write_project(root: &Path) {
    std::fs::write(
        root.join("index.html"),
        "<!doctype html><html><head><title>jsx-in-js</title></head>\
         <body><div id=\"app\"></div>\
         <script type=\"module\" src=\"/src/main.js\"></script></body></html>",
    )
    .unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/main.js"),
        "export default function App() {\n    return <div>hello</div>;\n}\n",
    )
    .unwrap();
}

fn build(binary: &Path, root: &Path, vite: bool) -> (bool, String) {
    let mut command = Command::new(binary);
    command.arg("build").arg(root).arg("--out-dir").arg("dist");
    if vite {
        command.arg("--vite");
    }
    let output = command.output().unwrap();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), combined)
}

/// The Vite/generic half. The build must FAIL, and the message must name the file
/// and the rename — not oxc's bare "Unexpected JSX expression", which says nothing
/// about why the same source builds under Next.
#[test]
fn a_vite_build_rejects_jsx_in_a_js_module_with_an_actionable_message() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let project = tempdir().unwrap();
    write_project(project.path());

    for vite in [false, true] {
        let (success, output) = build(&binary, project.path(), vite);
        assert!(!success, "vite={vite}: build must fail, got:\n{output}");
        assert!(
            output.contains("src/main.js"),
            "vite={vite}: the message must name the file:\n{output}"
        );
        assert!(
            output.contains("JSX is not enabled for `.js` files"),
            "vite={vite}: the message must explain the rule:\n{output}"
        );
        assert!(
            output.contains("main.jsx"),
            "vite={vite}: the message must give the remedy:\n{output}"
        );
        assert!(
            !output.contains("Unexpected JSX expression"),
            "vite={vite}: the raw oxc message must be replaced:\n{output}"
        );
    }
}

/// The Next half, driven through the real pages-router adapter: the same JSX-in-`.js`
/// page compiles, AND its imports are discovered — the component it imports has to
/// reach the bundle, which is exactly what a fatal parse silently prevented.
#[test]
fn a_next_pages_build_compiles_jsx_in_js_and_keeps_the_importers_subtree() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let project = tempdir().unwrap();
    let root = project.path();
    // A real Next pages-router app needs `next` resolvable; reuse the workspace's
    // pinned corpus install rather than shipping a second node_modules.
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("integration/e2e/apps/next-pages-shallow-routing/node_modules");
    if !corpus.is_dir() {
        eprintln!("skipping: {} not installed", corpus.display());
        return;
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(&corpus, root.join("node_modules")).unwrap();
    #[cfg(not(unix))]
    {
        eprintln!("skipping: symlinked node_modules is unix-only");
        return;
    }

    std::fs::write(root.join("next.config.js"), "module.exports = {};\n").unwrap();
    std::fs::write(root.join("package.json"), "{\"name\":\"jsx-in-js\",\"private\":true}\n").unwrap();
    std::fs::create_dir_all(root.join("pages")).unwrap();
    std::fs::create_dir_all(root.join("components")).unwrap();
    std::fs::write(
        root.join("components/Gallery.js"),
        "export default function Gallery() {\n    return <p>diffpackGalleryMarker</p>;\n}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("pages/index.js"),
        "import Gallery from '../components/Gallery';\n\
         export default function Home() {\n    return <Gallery />;\n}\n",
    )
    .unwrap();

    let output = Command::new(&binary)
        .arg("build-app")
        .arg(root)
        .arg("production")
        .output()
        .unwrap();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "next build failed:\n{combined}");

    // The page parsed, so its dependency list was real and the imported component
    // was discovered. Before the fix the page produced NO dependencies at all.
    let public = root.join(".diffpack-output/public");
    let bundled: String = std::fs::read_dir(&public)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "js"))
        .map(|entry| std::fs::read_to_string(entry.path()).unwrap())
        .collect();
    assert!(
        bundled.contains("diffpackGalleryMarker"),
        "the JSX `.js` page's imported component must be in the bundle:\n{combined}"
    );
}
