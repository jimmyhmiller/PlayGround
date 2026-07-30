// Hermetic Tier-1 corpus gate for the next app-router adapter (NO node, NO network).
//
// For each app under `integration/next-corpus/` this copies the tree to a tempdir and
// drives the crate's PUBLIC, node-free API — `is_app_router`, `configure` /
// `configure_dev` for client / react-server / ssr, and `write_prerender_plan` — then
// asserts the generated `.diffpack-next/` scaffold and every route's classified `kind`
// in `prerender-plan.json` against the app's committed `expected.json`. This is pure
// file IO: `configure` and `write_prerender_plan` only read app source and emit
// scaffold, they never resolve `react` or spawn node, so the whole route-discovery +
// classification + scaffold-generation surface is tested in-process, hermetically.
//
// An OPTIONAL gated sub-block additionally runs a REAL native build for one app when
// `target/release/diffpack` AND `integration/next-corpus/node_modules` both exist;
// otherwise it soft-skips (like the node gate in tests/oracle_incremental.rs).

use std::fs;
use std::path::{Path, PathBuf};

use diffpack::next_adapter::{configure, configure_dev, is_app_router, write_prerender_plan};
use serde_json::Value;
use tempfile::tempdir;

/// Recursively copy a directory tree (the corpus apps have nested dirs, unlike the
/// shallow copy_directory in tests/oracle_incremental.rs). Skips any generated
/// `.diffpack-*` output and `node_modules` so the copy is a clean source tree.
fn copy_directory(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if matches!(name_str.as_ref(), "node_modules" | ".diffpack-output" | ".diffpack-next" | ".next") {
            continue;
        }
        let from = entry.path();
        let to = destination.join(&name);
        if entry.file_type().unwrap().is_dir() {
            copy_directory(&from, &to);
        } else {
            fs::copy(&from, &to).unwrap();
        }
    }
}

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("integration/next-corpus")
}

/// Every corpus app: a direct child dir of the corpus root that carries an
/// `expected.json`. Sorted for a deterministic run order.
fn corpus_apps() -> Vec<PathBuf> {
    let mut apps: Vec<PathBuf> = fs::read_dir(corpus_dir())
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("expected.json").is_file())
        .collect();
    apps.sort();
    assert!(!apps.is_empty(), "no corpus apps found under {}", corpus_dir().display());
    apps
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

fn parse(path: &Path) -> Value {
    serde_json::from_str(&read(path)).unwrap_or_else(|error| panic!("invalid JSON {}: {error}", path.display()))
}

/// Drive one corpus app through the hermetic (node-free) adapter surface.
fn check_app(app_src: &Path) {
    let expected = parse(&app_src.join("expected.json"));
    let label = expected["name"].as_str().expect("expected.json has a name").to_string();
    let label = &label;
    let routes = expected["routes"].as_array().expect("expected.json has routes");
    let handlers = expected["handlers"].as_array().cloned().unwrap_or_default();
    let scaffold = &expected["scaffold"];

    // Copy to a tempdir so the committed tree is never dirtied and runs are parallel-safe
    // (configure writes .diffpack-next/ + write_prerender_plan writes into a temp out-dir).
    let workspace = tempdir().unwrap();
    let root = workspace.path().join(label);
    copy_directory(app_src, &root);

    // (1) app-router detection.
    assert!(is_app_router(&root), "[{label}] is_app_router must detect the app");

    // (2) configure() for every environment; each returns Some and re-writes the shared
    // .diffpack-next/ scaffold. Capture the client config for the alias assertions.
    let client = configure(&root, "client")
        .unwrap_or_else(|e| panic!("[{label}] configure(client) errored: {e}"))
        .unwrap_or_else(|| panic!("[{label}] configure(client) returned None (not detected)"));
    let react_server = configure(&root, "react-server")
        .unwrap_or_else(|e| panic!("[{label}] configure(react-server) errored: {e}"))
        .unwrap_or_else(|| panic!("[{label}] configure(react-server) returned None"));
    let ssr = configure(&root, "ssr")
        .unwrap_or_else(|e| panic!("[{label}] configure(ssr) errored: {e}"))
        .unwrap_or_else(|| panic!("[{label}] configure(ssr) returned None"));
    assert_eq!(client.environment, "client", "[{label}]");
    assert_eq!(react_server.environment, "react-server", "[{label}]");
    assert_eq!(ssr.environment, "ssr", "[{label}]");

    let adapter = root.join(".diffpack-next");
    let rsc = read(&adapter.join("rsc-entry.tsx"));
    let client_src = read(&adapter.join("client.tsx"));
    let ssr_src = read(&adapter.join("server.tsx"));

    // (3) structural scaffold markers that must exist for every app.
    for marker in [
        "const ROUTES = [",
        "function matchRoute(pathname)",
        "const ROUTE_HANDLERS = [",
        "Suspense",
        "ERROR_BOUNDARY",
        "function notFoundTree()",
    ] {
        assert!(rsc.contains(marker), "[{label}] rsc-entry missing marker `{marker}`");
    }
    assert!(client_src.contains("hydrateRoot(document"), "[{label}] client entry hydrates the document");
    assert!(
        client_src.contains("window.__diffpack_navigate"),
        "[{label}] client entry installs the soft-nav router"
    );
    assert!(ssr_src.contains("renderFlightToDocument"), "[{label}] ssr entry renders flight to document");
    assert!(adapter.join("image-manifest.ts").is_file(), "[{label}] image variant manifest is generated");

    // (4) every expected route path appears in the rsc ROUTES table.
    for route in routes {
        let path = route["path"].as_str().unwrap();
        assert!(
            rsc.contains(&format!("path: {path:?}")),
            "[{label}] rsc-entry ROUTES has no entry for `{path}`"
        );
    }

    // (5) scaffold-boundary interning is present iff the app declares it. `loading: M`/
    // `error: M` mark a real boundary module in a level (vs `loading: null`/`error: null`);
    // `APP_NOT_FOUND = M<i>` (vs `= null`) marks app/not-found.* wired into the 404 tree.
    assert_boundary(&rsc, "loading: M", scaffold["loading"].as_bool().unwrap(), label, "loading.tsx");
    assert_boundary(&rsc, "error: M", scaffold["error"].as_bool().unwrap(), label, "error.tsx");
    if scaffold["notFound"].as_bool().unwrap() {
        assert!(
            rsc.contains("const APP_NOT_FOUND = M"),
            "[{label}] expected app/not-found wired into the not-found tree (APP_NOT_FOUND = M<i>)"
        );
    } else {
        assert!(
            rsc.contains("const APP_NOT_FOUND = null"),
            "[{label}] expected NO app/not-found (APP_NOT_FOUND = null)"
        );
    }

    // (6) route handlers: each expected `route.ts` endpoint appears in ROUTE_HANDLERS
    // with each of its HTTP methods.
    for handler in &handlers {
        let path = handler["path"].as_str().unwrap();
        assert!(
            rsc.contains(&format!("path: {path:?}")),
            "[{label}] ROUTE_HANDLERS has no entry for `{path}`"
        );
        for method in handler["methods"].as_array().unwrap() {
            let method = method.as_str().unwrap();
            assert!(
                rsc.contains(&format!("{method}: H")),
                "[{label}] handler `{path}` is missing method `{method}` in ROUTE_HANDLERS"
            );
        }
    }
    if handlers.is_empty() {
        assert!(rsc.contains("const ROUTE_HANDLERS = [\n];"), "[{label}] expected NO route handlers");
    } else {
        assert!(rsc.contains("import * as H0 from"), "[{label}] a route handler must be namespace-imported");
    }

    // (7) next/* shims aliased to real generated files (assert on the client config).
    let aliases: std::collections::HashMap<_, _> = client.build.aliases.iter().cloned().collect();
    for spec in ["next/link", "next/image", "next/navigation", "next/headers", "next/server"] {
        let target = aliases
            .get(spec)
            .unwrap_or_else(|| panic!("[{label}] {spec} must be aliased"));
        assert!(Path::new(target).is_file(), "[{label}] {spec} shim file `{target}` must exist");
    }
    // Production config never turns HMR on and defines NODE_ENV=production.
    assert!(!client.build.hmr, "[{label}] production config keeps HMR off");
    assert!(
        client.build.defines.iter().any(|(k, v)| k == "process.env.NODE_ENV" && v == "\"production\""),
        "[{label}] production config defines NODE_ENV=production"
    );

    // (8) configure_dev: HMR on, NODE_ENV=development, production condition swapped, for
    // every environment.
    for environment in ["client", "react-server", "ssr"] {
        let dev = configure_dev(&root, environment, &diffpack::next_adapter::RouteScope::All)
            .unwrap_or_else(|e| panic!("[{label}] configure_dev({environment}) errored: {e}"))
            .unwrap_or_else(|| panic!("[{label}] configure_dev({environment}) returned None"));
        assert!(dev.build.hmr, "[{label}] dev {environment} turns HMR on");
        assert!(
            dev.build.defines.iter().any(|(k, v)| k == "process.env.NODE_ENV" && v == "\"development\""),
            "[{label}] dev {environment} defines NODE_ENV=development"
        );
        assert!(
            !dev.build.conditions.iter().any(|c| c == "production"),
            "[{label}] dev {environment} swaps the production resolve condition"
        );
    }

    // (9) write_prerender_plan → assert every route's kind / revalidate / gsp / reason
    // against expected.json. This reaches the classifier through the PUBLIC plan writer.
    let out = tempdir().unwrap();
    let count = write_prerender_plan(&root, out.path())
        .unwrap_or_else(|e| panic!("[{label}] write_prerender_plan errored: {e}"));
    assert_eq!(count, routes.len(), "[{label}] plan route count {count} != expected {}", routes.len());
    let plan = parse(&out.path().join("static/prerender-plan.json"));
    let plan = plan.as_array().expect("plan is an array");

    for route in routes {
        let path = route["path"].as_str().unwrap();
        let kind = route["kind"].as_str().unwrap();
        let entry = plan
            .iter()
            .find(|e| e["path"].as_str() == Some(path))
            .unwrap_or_else(|| panic!("[{label}] prerender plan has no route `{path}`"));
        assert_eq!(entry["kind"].as_str(), Some(kind), "[{label}] route `{path}` kind");

        if let Some(expected_reval) = route.get("revalidate").and_then(|r| r.as_u64()) {
            assert_eq!(
                entry.get("revalidate").and_then(|r| r.as_u64()),
                Some(expected_reval),
                "[{label}] route `{path}` revalidate (isr TTL)"
            );
        }
        if kind == "ssg" {
            if let Some(has_gsp) = route.get("hasGenerateStaticParams").and_then(|b| b.as_bool()) {
                assert_eq!(
                    entry.get("hasGenerateStaticParams").and_then(|b| b.as_bool()),
                    Some(has_gsp),
                    "[{label}] route `{path}` hasGenerateStaticParams"
                );
            }
            if let Some(dyn_params) = route.get("dynamicParams").and_then(|b| b.as_bool()) {
                assert_eq!(
                    entry.get("dynamicParams").and_then(|b| b.as_bool()),
                    Some(dyn_params),
                    "[{label}] route `{path}` dynamicParams"
                );
            }
        }
        if route.get("reasonPresent").and_then(|b| b.as_bool()) == Some(true) {
            let reason = entry.get("reason").and_then(|r| r.as_str()).unwrap_or("");
            assert!(
                !reason.is_empty(),
                "[{label}] dynamic route `{path}` must carry a non-empty reason (got: {reason:?})"
            );
        }
    }
}

/// A boundary is present iff the app declares it — `marker` (e.g. `loading: M`) must
/// appear when `present`, and must NOT appear when absent.
fn assert_boundary(rsc: &str, marker: &str, present: bool, label: &str, what: &str) {
    if present {
        assert!(rsc.contains(marker), "[{label}] expected a {what} boundary interned (`{marker}`)");
    } else {
        assert!(!rsc.contains(marker), "[{label}] expected NO {what} boundary, found `{marker}`");
    }
}

#[test]
fn corpus_classification_and_scaffold_hold_hermetically() {
    for app in corpus_apps() {
        check_app(&app);
    }
}

/// OPTIONAL gated build smoke: only when `target/release/diffpack` AND
/// `integration/next-corpus/node_modules` both exist does this run a REAL native build
/// for one representative app across the three graphs and assert emit success + at
/// least one client bundle. Otherwise it soft-skips — exactly like the node gate in
/// tests/oracle_incremental.rs.
#[test]
fn corpus_native_build_smoke_when_prereqs_present() {
    use std::process::Command;

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let diffpack = manifest.join("target/release/diffpack");
    let node_modules = corpus_dir().join("node_modules");
    if !diffpack.is_file() || !node_modules.is_dir() {
        // Prereqs absent: the hermetic Tier-1 test above is the real deliverable; the
        // node-backed native build lives in scripts/rsc/next-corpus-check.sh (Tier 2).
        eprintln!(
            "skip: corpus native-build smoke (need {} + {})",
            diffpack.display(),
            node_modules.display()
        );
        return;
    }

    // Build one representative app (real path, so react resolves via the parent-dir
    // node_modules) across the three graphs. Output goes to the gitignored
    // .diffpack-output under the app.
    let app = corpus_dir().join("blog-static");
    for environment in ["client", "react-server", "ssr"] {
        let status = Command::new(&diffpack)
            .arg("build-app")
            .arg(&app)
            .arg(environment)
            .arg("--no-minify")
            .status()
            .unwrap_or_else(|e| panic!("cannot run build-app {environment}: {e}"));
        assert!(status.success(), "build-app {environment} failed for blog-static");
    }

    // At least one client bundle was emitted under public/.
    let public = app.join(".diffpack-output/public");
    let js_count = fs::read_dir(&public)
        .unwrap_or_else(|e| panic!("no public dir {}: {e}", public.display()))
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("js"))
        .count();
    assert!(js_count > 0, "the client build emitted no public/*.js for blog-static");
}
