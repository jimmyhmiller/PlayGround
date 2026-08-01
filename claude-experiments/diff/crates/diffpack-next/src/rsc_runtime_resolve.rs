//! Where the RSC runtime (`react-server-dom-webpack`) comes from.
//!
//! diffpack's generated app-router entries import `react-server-dom-webpack/client`
//! (browser + SSR-of-flight) and `react-server-dom-webpack/server` (the react-server
//! graph, plus every `"use client"` proxy and `"use server"` registration the RSC
//! transform writes). **No real Next.js application depends on that package**: Next
//! never asks the app to install one, it vendors its own copy at
//! `next/dist/compiled/react-server-dom-webpack` and aliases the specifier onto it.
//!
//! Resolving the specifier from the APP therefore made a stock `create-next-app`
//! unbuildable until the user installed a package they had no reason to know about —
//! diffpack's own requirement charged to the app's dependency list. This module removes
//! that requirement by resolving the runtime the same way Next does:
//!
//! 1. the app's own `react-server-dom-webpack`, if it has one (an explicit dependency
//!    always wins — nothing is aliased and resolution is exactly what it always was);
//! 2. otherwise the copy `next` vendors, resolved THROUGH that copy's own
//!    `package.json` `exports` map under the build environment's conditions, so
//!    `/server` picks the real flight writer under `react-server`, `/client` picks
//!    `client.browser` in the browser and `client.node` on the server — byte-for-byte
//!    the files node would have chosen for an installed package.
//!
//! The vendored copy `require`s bare `react` / `react-dom`, so it binds to the app's
//! React exactly as an installed copy would; there is no second React in the graph.
//!
//! When neither copy exists there is nothing to alias and the specifier stays
//! unresolved — a fatal build diagnostic, not a silent gap. `bundler` renders that one
//! specially (see `unresolved_import_message`) so the message says the RSC runtime is
//! diffpack's requirement rather than the app's missing dependency.
//!
//! # React itself
//!
//! [`react_aliases`] answers the same question for `react` and `react-dom`, with one
//! difference that inverts the precedence: an App Router app's own `react` dependency
//! is NOT what `next build` compiles the app router against. Next aliases `react$`,
//! `react-dom$` and their entry points onto the copies IT vendors, in every layer —
//! browser, SSR and react-server alike (`createVendoredReactAliases` in
//! `next/dist/build/create-compiler-aliases.js`) — because the App Router is written
//! against React internals that only the exact React it ships provides.
//!
//! That is not a detail a bundler can skip. React 18.2's `react-server` export
//! condition resolves to `react.shared-subset.js`, whose entire body is
//! `throw Error("This entry point is not yet supported outside of experimental
//! channels")` — the App Router's flight render is IMPOSSIBLE against a stable React
//! 18, and an app on `react@18.2.0` (cal.com, for one) builds and renders under
//! `next build` only because Next never uses that copy. Resolving `react` from the
//! app there produces a bundle that dies at the first render, so honoring the app's
//! own dependency would be honoring the wrong contract.

use std::path::{Path, PathBuf};

use oxc_resolver::{ResolveOptions, Resolver};

/// The package whose flight runtime the app-router entries import.
pub const PACKAGE: &str = "react-server-dom-webpack";

pub const MISSING_RUNTIME_HELP: &str = "\n  this is diffpack's requirement, not your app's: diffpack's app-router entries need an RSC (flight) runtime.\n  It normally uses the copy `next` vendors at next/dist/compiled/react-server-dom-webpack; the installed `next` has none (or `next` is not installed).\n  install it:  npm install react-server-dom-webpack";

/// The subpaths diffpack's generated code imports. Each is aliased to the concrete
/// file the environment's conditions select; nothing else in the package is aliased,
/// so an unexpected subpath fails to resolve loudly instead of being redirected.
const SUBPATHS: [&str; 3] = ["client", "server", "static"];

/// Where Next keeps its vendored copy, relative to the `next` package root.
const VENDORED_RELATIVE: [&str; 3] = ["dist", "compiled", PACKAGE];

/// `(specifier, absolute file)` aliases pinning `react-server-dom-webpack/*` for a build
/// with these resolve `conditions`. Empty when the app installs its own copy (it wins) or
/// when no copy exists anywhere (the unresolved import is then the build's own error).
///
/// `client` selects the browser resolution rules (the `browser` alias field), matching
/// the default loader's client resolve options.
pub fn aliases(root: &Path, conditions: &[String], client: bool) -> Vec<(String, String)> {
    if installed_in_app(root) {
        return Vec::new();
    }
    let Some(vendored) = vendored_dir(root) else {
        return Vec::new();
    };
    let resolver = Resolver::new(vendored_resolve_options(&vendored, conditions, client));
    let mut aliases = Vec::new();
    for subpath in SUBPATHS {
        let specifier = format!("{PACKAGE}/{subpath}");
        // `modules` points at the directory that CONTAINS the vendored package, so the
        // bare specifier resolves through its package.json `exports` under `conditions`
        // — the whole point of going through the resolver instead of guessing a file.
        if let Ok(resolution) = resolver.resolve(root, &specifier) {
            aliases.push((
                specifier,
                resolution.full_path().to_string_lossy().into_owned(),
            ));
        }
    }
    aliases
}

/// The React packages a Next App Router build takes from the copy `next` vendors
/// rather than from the app's own `node_modules`. See the module docs.
const VENDORED_REACT_PACKAGES: [&str; 2] = ["react", "react-dom"];

/// `(specifier, absolute file)` aliases pinning `react` and `react-dom` — every entry
/// point they export — to the copies `next` vendors, resolved through those copies'
/// own `exports` maps under this environment's `conditions`.
///
/// The alias set is READ from the vendored `package.json`'s `exports` keys rather than
/// listed here, so it is whatever the installed Next actually ships and cannot drift
/// from it as React's entry points change.
///
/// Empty when the installed `next` vendors no React (nothing better to alias to, so
/// the app's own copy is used exactly as before) or when there is no `next` at all.
///
/// Longest specifier first: the default loader's alias table also rewrites prefix
/// matches, so `react/jsx-runtime` must be found as an exact alias before the bare
/// `react` entry can claim it as a prefix.
pub fn react_aliases(root: &Path, conditions: &[String], client: bool) -> Vec<(String, String)> {
    let mut aliases = Vec::new();
    for package in VENDORED_REACT_PACKAGES {
        let Some(vendored) = vendored_package_dir(root, package) else {
            continue;
        };
        let resolver = Resolver::new(vendored_resolve_options(&vendored, conditions, client));
        for specifier in exported_specifiers(package, &vendored) {
            let Ok(resolution) = resolver.resolve(root, &specifier) else {
                continue;
            };
            let file = resolution.full_path().to_string_lossy().into_owned();
            // The BY-PATH spelling of the same entry point, pinned to the same file.
            //
            // Next rewrites React's internal `require("react")` to
            // `require("next/dist/compiled/react")` inside the copies it vendors, and a
            // path INTO a package does not go through that package's `exports` map (a
            // subpath only does when the package is named as a package). So the
            // vendored `react-dom.react-server` would otherwise pull the CLIENT React
            // via its `main`, and the RSC render dies on React's own check: `The
            // "react" package in this environment is not configured correctly. The
            // "react-server" condition must be enabled`. Aliasing both spellings to one
            // file is what keeps exactly ONE React per environment in the graph.
            aliases.push((vendored_specifier(package, &specifier), file.clone()));
            aliases.push((specifier, file));
        }
    }
    // Longest first: the alias table also rewrites PREFIX matches, so every exact
    // entry must be seen before a shorter one could claim it as a prefix.
    aliases.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()).then(left.cmp(right)));
    aliases
}

/// Next's Node.js React Server layer aliases.
///
/// Native `.next` route entries execute inside Next's app-page runtime. They must use
/// the same vendored facade modules as that runtime, not merely resolve the underlying
/// compiled React packages to equivalent files. The facades are the module-identity
/// boundary shared by the renderer, Flight writer, and Next's route module.
pub fn native_next_rsc_aliases(next_root: &Path) -> Result<Vec<(String, String)>, String> {
    let entries = [
        (
            "react",
            "dist/server/route-modules/app-page/vendored/rsc/react.js",
        ),
        (
            "react/compiler-runtime",
            "dist/server/route-modules/app-page/vendored/rsc/react-compiler-runtime.js",
        ),
        (
            "react/jsx-dev-runtime",
            "dist/server/route-modules/app-page/vendored/rsc/react-jsx-dev-runtime.js",
        ),
        (
            "react/jsx-runtime",
            "dist/server/route-modules/app-page/vendored/rsc/react-jsx-runtime.js",
        ),
        (
            "react-dom",
            "dist/server/route-modules/app-page/vendored/rsc/react-dom.js",
        ),
        ("react-dom/client", "dist/compiled/react-dom/client.js"),
        ("react-dom/server", "dist/compiled/react-dom/server.node.js"),
        (
            "react-dom/server.browser",
            "dist/compiled/react-dom/server.browser.js",
        ),
        ("react-dom/static", "dist/compiled/react-dom/static.node.js"),
        (
            "react-server-dom-webpack/client",
            "dist/compiled/react-server-dom-webpack/client.node.js",
        ),
        (
            "react-server-dom-webpack/server",
            "dist/server/route-modules/app-page/vendored/rsc/react-server-dom-webpack-server.js",
        ),
        (
            "react-server-dom-webpack/server.node",
            "dist/server/route-modules/app-page/vendored/rsc/react-server-dom-webpack-server.js",
        ),
        (
            "react-server-dom-webpack/static",
            "dist/server/route-modules/app-page/vendored/rsc/react-server-dom-webpack-static.js",
        ),
    ];
    let mut aliases = Vec::with_capacity(entries.len());
    for (specifier, relative) in entries {
        let path = next_root.join(relative);
        if !path.is_file() {
            return Err(format!(
                "installed Next is missing its native RSC runtime entry {}",
                path.display()
            ));
        }
        aliases.push((specifier.to_string(), path.to_string_lossy().into_owned()));
    }
    aliases.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()).then(left.cmp(right)));
    Ok(aliases)
}

/// Next's Node.js SSR/client React layer aliases used while turning a Flight
/// response into HTML. These facades share React's dispatcher with the app-page
/// runtime that owns the render.
pub fn native_next_ssr_aliases(next_root: &Path) -> Result<Vec<(String, String)>, String> {
    let entries = [
        (
            "react",
            "dist/server/route-modules/app-page/vendored/ssr/react.js",
        ),
        (
            "react/compiler-runtime",
            "dist/server/route-modules/app-page/vendored/ssr/react-compiler-runtime.js",
        ),
        (
            "react/jsx-dev-runtime",
            "dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js",
        ),
        (
            "react/jsx-runtime",
            "dist/server/route-modules/app-page/vendored/ssr/react-jsx-runtime.js",
        ),
        (
            "react-dom",
            "dist/server/route-modules/app-page/vendored/ssr/react-dom.js",
        ),
        // The SSR module graph includes application dependencies which import
        // these public renderer entry points directly. They must stay on the
        // same vendored React ABI as Next's app-page runtime; falling back to
        // the application's ReactDOM (Cal currently carries React 18 here)
        // creates elements the React 19 route renderer cannot consume.
        ("react-dom/server", "dist/compiled/react-dom/server.node.js"),
        (
            "react-dom/server.browser",
            "dist/compiled/react-dom/server.browser.js",
        ),
        ("react-dom/static", "dist/compiled/react-dom/static.node.js"),
        (
            "react-server-dom-webpack/client",
            "dist/server/route-modules/app-page/vendored/ssr/react-server-dom-webpack-client.js",
        ),
    ];
    let mut aliases = Vec::with_capacity(entries.len());
    for (specifier, relative) in entries {
        let path = next_root.join(relative);
        if !path.is_file() {
            return Err(format!(
                "installed Next is missing its native SSR runtime entry {}",
                path.display()
            ));
        }
        aliases.push((specifier.to_string(), path.to_string_lossy().into_owned()));
    }
    aliases.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()).then(left.cmp(right)));
    Ok(aliases)
}

/// The shared-runtime replacements from Next's Node webpack configuration.
/// Next source imports these contexts relatively from several depths; every spelling
/// is redirected to the app-page runtime facade so RSC and SSR observe one provider.
pub fn native_next_context_aliases(next_root: &Path) -> Result<Vec<(String, String)>, String> {
    let entries = [
        (
            "../../../shared/lib/app-router-context.shared-runtime",
            "app-router-context",
        ),
        (
            "../../shared/lib/app-router-context.shared-runtime",
            "app-router-context",
        ),
        (
            "../../shared/lib/hooks-client-context.shared-runtime",
            "hooks-client-context",
        ),
        (
            "../../shared/lib/router-context.shared-runtime",
            "router-context",
        ),
        (
            "../../shared/lib/server-inserted-html.shared-runtime",
            "server-inserted-html",
        ),
    ];
    entries
        .into_iter()
        .map(|(specifier, module)| {
            let path = next_root.join(format!(
                "dist/server/route-modules/app-page/vendored/contexts/{module}.js"
            ));
            if !path.is_file() {
                return Err(format!(
                    "installed Next is missing its native context entry {}",
                    path.display()
                ));
            }
            Ok((specifier.to_string(), path.to_string_lossy().into_owned()))
        })
        .collect()
}

/// Next's Pages Node layer applies the `.shared-runtime` replacement rule to
/// the Pages runtime's context table.
pub fn native_pages_context_aliases(next_root: &Path) -> Result<Vec<(String, String)>, String> {
    let modules = [
        "app-router-context",
        "head-manager-context",
        "hooks-client-context",
        "html-context",
        "image-config-context",
        "loadable-context",
        "loadable",
        "router-context",
        "server-inserted-html",
    ];
    let mut aliases = Vec::new();
    for module in modules {
        let path = next_root.join(format!(
            "dist/server/route-modules/pages/vendored/contexts/{module}.js"
        ));
        if !path.is_file() {
            return Err(format!(
                "installed Next is missing its native Pages context entry {}",
                path.display()
            ));
        }
        let target = path.to_string_lossy().into_owned();
        for depth in 1..=5 {
            aliases.push((
                format!("{}shared/lib/{module}.shared-runtime", "../".repeat(depth)),
                target.clone(),
            ));
        }
    }
    aliases.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()).then(left.cmp(right)));
    Ok(aliases)
}

/// `react/jsx-runtime` -> `next/dist/compiled/react/jsx-runtime`: the same entry point
/// spelled as the path Next's own vendored code requires it by.
fn vendored_specifier(package: &str, specifier: &str) -> String {
    let subpath = specifier.strip_prefix(package).unwrap_or("");
    format!("next/dist/compiled/{package}{subpath}")
}

/// Every specifier the vendored package's `exports` map publishes: `"."` is the bare
/// package name, `"./x"` is `<package>/x`. `./package.json` is skipped — it is
/// metadata, and aliasing it would point a `require("react/package.json")` at the
/// VENDORED manifest, misreporting the app's React version.
///
/// A package with no `exports` map publishes only its main entry.
fn exported_specifiers(package: &str, vendored: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(vendored.join("package.json")) else {
        return Vec::new();
    };
    let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let Some(exports) = manifest
        .get("exports")
        .and_then(serde_json::Value::as_object)
    else {
        return vec![package.to_string()];
    };
    exports
        .keys()
        .filter(|key| key.as_str() != "./package.json")
        .filter_map(|key| match key.as_str() {
            "." => Some(package.to_string()),
            subpath => subpath
                .strip_prefix("./")
                .map(|subpath| format!("{package}/{subpath}")),
        })
        .collect()
}

/// The `next/dist/compiled/<package>` directory for `root`, when the installed `next`
/// ships one.
fn vendored_package_dir(root: &Path, package: &str) -> Option<PathBuf> {
    node_modules_ancestors(root).find_map(|node_modules| {
        let next = node_modules.join("next");
        if !next.join("package.json").is_file() {
            return None;
        }
        let candidate = next.join("dist").join("compiled").join(package);
        candidate
            .join("package.json")
            .is_file()
            .then_some(candidate)
    })
}

/// True when `react-server-dom-webpack` is installed for `root` (its own `node_modules`
/// or any ancestor's — npm/pnpm hoisting puts it either place).
fn installed_in_app(root: &Path) -> bool {
    node_modules_ancestors(root)
        .any(|node_modules| node_modules.join(PACKAGE).join("package.json").is_file())
}

/// The `next/dist/compiled/react-server-dom-webpack` directory for `root`, when the
/// installed `next` ships one.
fn vendored_dir(root: &Path) -> Option<PathBuf> {
    node_modules_ancestors(root).find_map(|node_modules| {
        let mut candidate = node_modules.join("next");
        if !candidate.join("package.json").is_file() {
            return None;
        }
        for segment in VENDORED_RELATIVE {
            candidate = candidate.join(segment);
        }
        candidate
            .join("package.json")
            .is_file()
            .then_some(candidate)
    })
}

/// Every `<ancestor>/node_modules` from `root` upward, nearest first.
fn node_modules_ancestors(root: &Path) -> impl Iterator<Item = PathBuf> + '_ {
    root.ancestors().map(|dir| dir.join("node_modules"))
}

/// Resolve an installed package using Node's nearest-ancestor `node_modules`
/// lookup. Native Next integration must support hoisted monorepos such as Cal.com,
/// where the application directory intentionally has no local `node_modules/next`.
pub fn installed_package_root(root: &Path, package: &str) -> Result<PathBuf, String> {
    node_modules_ancestors(root)
        .map(|node_modules| node_modules.join(package))
        .find(|candidate| candidate.join("package.json").is_file())
        .and_then(|candidate| candidate.canonicalize().ok())
        .ok_or_else(|| {
            format!(
                "cannot resolve installed package {package} from {}",
                root.display()
            )
        })
}

/// Resolve options that find the VENDORED package by its bare name: `modules` is the
/// directory containing it rather than `node_modules`, so `exports` applies exactly as
/// it would for an installed package. Conditions mirror `bundler::resolve_options` (the
/// environment's own, plus `import`/`default` so plain ESM still resolves).
fn vendored_resolve_options(
    vendored: &Path,
    conditions: &[String],
    client: bool,
) -> ResolveOptions {
    let container = vendored
        .parent()
        .expect("the vendored package always has a parent directory")
        .to_string_lossy()
        .into_owned();
    let mut condition_names = conditions.to_vec();
    for fallback in ["import", "default"] {
        if !condition_names.iter().any(|name| name == fallback) {
            condition_names.push(fallback.to_string());
        }
    }
    ResolveOptions {
        modules: vec![container],
        extensions: [".js", ".mjs", ".cjs", ".json"]
            .into_iter()
            .map(String::from)
            .collect(),
        condition_names,
        alias_fields: if client {
            vec![vec!["browser".into()]]
        } else {
            Vec::new()
        },
        main_fields: if client {
            vec!["browser".into(), "module".into(), "main".into()]
        } else {
            vec!["module".into(), "main".into()]
        },
        ..ResolveOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal stand-in for what Next vendors: the same `exports` shape (the
    /// conditions that matter — `react-server`, `node`, `browser`, `default`) over real
    /// files, so the resolver's choice is observable.
    fn write_vendored_next(root: &Path) {
        let vendored = root
            .join("node_modules/next")
            .join(VENDORED_RELATIVE.join("/"));
        std::fs::create_dir_all(&vendored).unwrap();
        std::fs::write(
            root.join("node_modules/next/package.json"),
            r#"{"name":"next","version":"16.0.0"}"#,
        )
        .unwrap();
        std::fs::write(
            vendored.join("package.json"),
            r#"{
              "name": "react-server-dom-webpack-builtin",
              "main": "index.js",
              "exports": {
                "./client": {
                  "node": "./client.node.js",
                  "browser": "./client.browser.js",
                  "default": "./client.browser.js"
                },
                "./server": {
                  "react-server": { "node": "./server.node.js" },
                  "default": "./server.js"
                },
                "./static": {
                  "react-server": { "node": "./static.node.js" },
                  "default": "./static.js"
                }
              }
            }"#,
        )
        .unwrap();
        for file in [
            "client.node.js",
            "client.browser.js",
            "server.node.js",
            "server.js",
            "static.node.js",
            "static.js",
            "index.js",
        ] {
            std::fs::write(vendored.join(file), "module.exports = {};\n").unwrap();
        }
    }

    fn names(aliases: &[(String, String)]) -> Vec<(String, String)> {
        aliases
            .iter()
            .map(|(specifier, path)| {
                (
                    specifier.clone(),
                    Path::new(path)
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                )
            })
            .collect()
    }

    #[test]
    fn falls_back_to_the_copy_next_vendors_when_the_app_has_none() {
        // FINDINGS item 3: a stock create-next-app has no `react-server-dom-webpack`.
        // The runtime must come from the copy Next itself ships, per environment.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write_vendored_next(root);

        let react_server = aliases(
            root,
            &["react-server".into(), "node".into(), "production".into()],
            false,
        );
        assert_eq!(
            names(&react_server),
            vec![
                (
                    "react-server-dom-webpack/client".into(),
                    "client.node.js".into()
                ),
                // The flight WRITER only exists under `react-server`; `server.js`
                // throws on import, so picking it would break every RSC render.
                (
                    "react-server-dom-webpack/server".into(),
                    "server.node.js".into()
                ),
                (
                    "react-server-dom-webpack/static".into(),
                    "static.node.js".into()
                ),
            ],
        );

        let browser = aliases(
            root,
            &["module".into(), "browser".into(), "production".into()],
            true,
        );
        assert_eq!(
            names(&browser)[0],
            (
                "react-server-dom-webpack/client".into(),
                "client.browser.js".into()
            ),
        );
    }

    #[test]
    fn an_app_that_installs_its_own_copy_keeps_it() {
        // An explicit dependency is never overridden: no alias at all, so resolution
        // is exactly what it was before the vendored fallback existed.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write_vendored_next(root);
        let own = root.join("node_modules").join(PACKAGE);
        std::fs::create_dir_all(&own).unwrap();
        std::fs::write(
            own.join("package.json"),
            r#"{"name":"react-server-dom-webpack"}"#,
        )
        .unwrap();

        assert!(
            aliases(root, &["node".into()], false).is_empty(),
            "the app's own react-server-dom-webpack must win over Next's vendored copy",
        );
    }

    /// A stand-in for `next/dist/compiled/react{,-dom}`: the same `exports` shape
    /// Next ships, where the `react-server` condition selects a DIFFERENT file from
    /// the default. Only the condition dispatch is modelled — that is the whole
    /// question this alias answers.
    fn write_vendored_react(root: &Path) {
        std::fs::create_dir_all(root.join("node_modules/next")).unwrap();
        std::fs::write(
            root.join("node_modules/next/package.json"),
            r#"{"name":"next","version":"16.0.0"}"#,
        )
        .unwrap();
        for (package, files) in [
            (
                "react",
                vec![
                    "index.js",
                    "react.react-server.js",
                    "jsx-runtime.js",
                    "jsx-runtime.react-server.js",
                ],
            ),
            ("react-dom", vec!["index.js", "react-dom.react-server.js"]),
        ] {
            let dir = root.join("node_modules/next/dist/compiled").join(package);
            std::fs::create_dir_all(&dir).unwrap();
            for file in &files {
                std::fs::write(dir.join(file), "module.exports = {};\n").unwrap();
            }
            let manifest = if package == "react" {
                r#"{"name":"react-builtin","main":"index.js","exports":{
                    ".":{"react-server":"./react.react-server.js","default":"./index.js"},
                    "./package.json":"./package.json",
                    "./jsx-runtime":{"react-server":"./jsx-runtime.react-server.js","default":"./jsx-runtime.js"}}}"#
            } else {
                r#"{"name":"react-dom-builtin","main":"index.js","exports":{
                    ".":{"react-server":"./react-dom.react-server.js","default":"./index.js"},
                    "./package.json":"./package.json"}}"#
            };
            std::fs::write(dir.join("package.json"), manifest).unwrap();
        }
    }

    /// React comes from the copy Next vendors, per layer — and the app's own
    /// `react` never wins, because an App Router app on React 18 has no working
    /// `react-server` entry at all.
    #[test]
    fn react_resolves_to_the_copy_next_vendors_under_each_layers_conditions() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write_vendored_react(root);
        // The app installs its own React, as every real app does. It must NOT win.
        let own = root.join("node_modules/react");
        std::fs::create_dir_all(&own).unwrap();
        std::fs::write(
            own.join("package.json"),
            r#"{"name":"react","main":"index.js"}"#,
        )
        .unwrap();
        std::fs::write(own.join("index.js"), "module.exports = {};\n").unwrap();

        let react_server = names(&react_aliases(
            root,
            &["react-server".into(), "node".into(), "production".into()],
            false,
        ));
        assert!(
            react_server.contains(&("react".into(), "react.react-server.js".into())),
            "{react_server:?}",
        );
        assert!(
            react_server.contains(&(
                "react/jsx-runtime".into(),
                "jsx-runtime.react-server.js".into()
            )),
            "the JSX runtime must follow the same condition as React itself: {react_server:?}",
        );

        let browser = names(&react_aliases(
            root,
            &["module".into(), "browser".into(), "production".into()],
            true,
        ));
        assert!(
            browser.contains(&("react".into(), "index.js".into())),
            "{browser:?}",
        );
    }

    /// Next rewrites React's internal `require("react")` to
    /// `require("next/dist/compiled/react")` inside the copies it vendors, and a
    /// path INTO a package does not go through that package's `exports`. Both
    /// spellings must therefore land on the SAME file, or the react-server
    /// `react-dom` pulls the client React and React's own configuration check
    /// throws.
    #[test]
    fn the_by_path_spelling_of_a_vendored_entry_points_at_the_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write_vendored_react(root);
        let aliases = react_aliases(root, &["react-server".into(), "node".into()], false);
        let find = |specifier: &str| {
            aliases
                .iter()
                .find(|(from, _)| from == specifier)
                .map(|(_, to)| to.clone())
                .unwrap_or_else(|| panic!("no alias for {specifier} in {aliases:?}"))
        };
        assert_eq!(find("react"), find("next/dist/compiled/react"));
        assert_eq!(find("react-dom"), find("next/dist/compiled/react-dom"),);
        // The alias table also rewrites PREFIX matches, so every longer specifier
        // must come first or the bare entry claims it.
        let lengths: Vec<usize> = aliases.iter().map(|(from, _)| from.len()).collect();
        assert!(
            lengths.windows(2).all(|pair| pair[0] >= pair[1]),
            "aliases must be longest-first: {aliases:?}",
        );
    }

    /// An app whose `next` vendors no React aliases nothing: there is nothing
    /// better to point at, so the app's own copy is used exactly as before.
    #[test]
    fn a_next_without_a_vendored_react_aliases_nothing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules/next")).unwrap();
        std::fs::write(
            dir.path().join("node_modules/next/package.json"),
            r#"{"name":"next","version":"16.0.0"}"#,
        )
        .unwrap();
        assert!(react_aliases(dir.path(), &["node".into()], false).is_empty());
    }

    #[test]
    fn no_copy_anywhere_aliases_nothing() {
        // Nothing to alias — the specifier stays unresolved, which is a FATAL build
        // diagnostic (bundler renders it with the RSC-runtime explanation).
        let dir = tempfile::tempdir().unwrap();
        assert!(aliases(dir.path(), &["node".into()], false).is_empty());
    }
}
