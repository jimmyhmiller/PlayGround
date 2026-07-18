//! The plugin-host sidecar resolves real project config. Gated on `node` and the
//! pinned fixture's installed dependencies; skips cleanly when unavailable.

use std::path::Path;
use std::process::Command;

use diffpack::host::{Sidecar, resolve_config};

fn node_available() -> bool {
    Command::new("node")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[test]
fn resolve_config_reports_the_tanstack_router_alias() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/tanstack-start-reference");
    if !node_available() || !root.join("node_modules/vite").exists() {
        eprintln!("skipping resolve_config test: node or fixture node_modules unavailable");
        return;
    }

    let config = resolve_config(&root, "client").unwrap();
    assert_eq!(config.environment, "client");
    assert!(
        config.environments.iter().any(|name| name == "ssr"),
        "expected an ssr environment among {:?}",
        config.environments
    );

    let router = config
        .build
        .aliases
        .iter()
        .find(|(find, _)| find == "#tanstack-router-entry")
        .expect("the router entry alias must be reported");
    assert!(
        router.1.replace('\\', "/").ends_with("src/router.tsx"),
        "router alias should point at the app router, got {}",
        router.1
    );
}

#[test]
fn the_long_lived_sidecar_resolves_and_loads_a_virtual_module() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/tanstack-start-reference");
    if !node_available() || !root.join("node_modules/vite").exists() {
        eprintln!("skipping sidecar serve test: node or fixture node_modules unavailable");
        return;
    }

    let sidecar = Sidecar::start(&root).unwrap();

    // The TanStack manifest is a plugin-generated virtual module (no file).
    let resolved = sidecar
        .resolve_id("ssr", "tanstack-start-manifest:v", None)
        .unwrap()
        .expect("a framework plugin must resolve the manifest virtual id");
    assert!(resolved.contains("tanstack-start-manifest"), "{resolved}");

    let code = sidecar
        .load("ssr", &resolved)
        .unwrap()
        .expect("a framework plugin must load the manifest virtual module");
    assert!(code.contains("tsrStartManifest"), "{code}");

    // Serialized requests keep working (the process is long-lived).
    let again = sidecar
        .resolve_id("ssr", "tanstack-start-manifest:v", None)
        .unwrap();
    assert_eq!(again.as_deref(), Some(resolved.as_str()));

    sidecar.shutdown();
}
