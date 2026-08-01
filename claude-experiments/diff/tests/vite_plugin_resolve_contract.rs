// A Vite PLUGIN is where a non-React project's whole compilation contract comes
// from, and that contract is two things, not one.
//
// `@preact/preset-vite` returns BOTH from its `config()` hooks:
//   * `{ oxc: { jsx: { runtime: 'automatic', importSource: 'preact' } } }` — how JSX
//     lowers, which diffpack already honored, and
//   * `{ resolve: { alias: { react: 'preact/compat', 'react/jsx-runtime':
//     'preact/jsx-runtime', 'react-dom': 'preact/compat', ... } } }` — how the
//     react-named specifiers RESOLVE, which diffpack silently threw away.
//
// The second is not optional decoration. Preact apps import from `react` all the
// time (every third-party React component does, and `preact/compat` exists for
// exactly that), and `create-vite`'s preact templates do not depend on `react` at
// all. Dropping the alias therefore either fails a build that Vite completes — with
// the actively wrong advice `npm install react` — or, when `react` is present as a
// transitive dependency, silently resolves to REAL React and ships two rendering
// libraries in one bundle with hooks split between them.
//
// This drives the real, pinned `@preact/preset-vite` from the corpus rather than a
// hand-written stand-in, because the defect was in reading a real plugin's real
// `config()` return shape.
//
// Soft-skips when `target/release/diffpack` or the corpus install is absent (same
// convention as tests/jsx_extensions.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;

fn release_binary() -> Option<PathBuf> {
    let binary = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/diffpack");
    binary.is_file().then_some(binary)
}

/// The pinned `vite-preact` corpus install: real `preact` + real
/// `@preact/preset-vite`. Reused by symlink rather than installed a second time.
fn corpus_node_modules() -> Option<PathBuf> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/e2e/apps/vite-preact/node_modules");
    path.is_dir().then_some(path)
}

#[test]
fn a_preact_preset_aliases_react_to_preact_compat_and_lowers_jsx_to_preact() {
    let Some(binary) = release_binary() else {
        eprintln!("skipping: target/release/diffpack not built");
        return;
    };
    let Some(corpus) = corpus_node_modules() else {
        eprintln!("skipping: integration/e2e/apps/vite-preact/node_modules not installed");
        return;
    };
    #[cfg(not(unix))]
    {
        eprintln!("skipping: symlinked node_modules is unix-only");
        return;
    }

    let project = tempdir().unwrap();
    let root = project.path();
    #[cfg(unix)]
    std::os::unix::fs::symlink(&corpus, root.join("node_modules")).unwrap();
    std::fs::write(
        root.join("package.json"),
        "{\"name\":\"preact-alias\",\"private\":true,\"type\":\"module\"}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("vite.config.js"),
        "import { defineConfig } from 'vite'\n\
         import preact from '@preact/preset-vite'\n\
         export default defineConfig({ plugins: [preact()] })\n",
    )
    .unwrap();
    std::fs::write(
        root.join("index.html"),
        "<!doctype html><html><head><title>preact-alias</title></head>\
         <body><div id=\"app\"></div>\
         <script type=\"module\" src=\"/src/main.jsx\"></script></body></html>",
    )
    .unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    // `forwardRef` is only reachable through the preset's `react -> preact/compat`
    // alias: this project has no `react` in `node_modules`, exactly like every
    // create-vite preact scaffold.
    std::fs::write(
        root.join("src/main.jsx"),
        "import { render } from 'preact'\n\
         import { forwardRef } from 'react'\n\
         const Box = forwardRef((props, ref) => <div ref={ref}>boxed</div>)\n\
         export function App() { return <Box /> }\n\
         render(<App />, document.getElementById('app'))\n",
    )
    .unwrap();

    let output = Command::new(&binary)
        .arg("build")
        .arg(root)
        .arg("--vite")
        .arg("--out-dir")
        .arg("dist")
        .output()
        .unwrap();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success() && !combined.contains("cannot resolve \"react\""),
        "vite builds this project; the preset's `react -> preact/compat` alias must \
         reach the native resolver:\n{combined}"
    );

    let bundle: String = std::fs::read_dir(root.join("dist"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "js"))
        .map(|entry| std::fs::read_to_string(entry.path()).unwrap())
        .collect();
    assert!(
        bundle.contains("forwardRef"),
        "the aliased `preact/compat` module must be in the bundle:\n{combined}"
    );
    // The other half of the same plugin's contract: JSX lowers against preact, and
    // nothing in the bundle reaches for a react runtime this project does not have.
    assert!(
        bundle.contains("preact/jsx-runtime"),
        "the preset's `importSource: 'preact'` must lower JSX against preact:\n{bundle:.400}"
    );
    // Quoted, because `preact/jsx-runtime` CONTAINS `react/jsx-runtime` as a
    // substring; the emitted specifier map quotes each key.
    assert!(
        !bundle.contains("\"react/jsx-runtime\""),
        "no module may fall back to `react/jsx-runtime` in a preact project"
    );
}
