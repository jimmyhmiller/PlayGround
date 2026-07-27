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

use std::path::{Path, PathBuf};

use oxc_resolver::{ResolveOptions, Resolver};

/// The package whose flight runtime the app-router entries import.
pub const PACKAGE: &str = "react-server-dom-webpack";

/// The subpaths diffpack's generated code imports. Each is aliased to the concrete
/// file the environment's conditions select; nothing else in the package is aliased,
/// so an unexpected subpath fails to resolve loudly instead of being redirected.
const SUBPATHS: [&str; 2] = ["client", "server"];

/// Where Next keeps its vendored copy, relative to the `next` package root.
const VENDORED_RELATIVE: [&str; 3] = ["dist", "compiled", PACKAGE];

/// `(specifier, absolute file)` aliases pinning `react-server-dom-webpack/*` for a build
/// with these resolve `conditions`. Empty when the app installs its own copy (it wins) or
/// when no copy exists anywhere (the unresolved import is then the build's own error).
///
/// `client` selects the browser resolution rules (the `browser` alias field), matching
/// [`crate::bundler`]'s `resolve_options` for `Target::Client`.
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

/// True when `react-server-dom-webpack` is installed for `root` (its own `node_modules`
/// or any ancestor's — npm/pnpm hoisting puts it either place).
fn installed_in_app(root: &Path) -> bool {
    node_modules_ancestors(root).any(|node_modules| node_modules.join(PACKAGE).join("package.json").is_file())
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
        candidate.join("package.json").is_file().then_some(candidate)
    })
}

/// Every `<ancestor>/node_modules` from `root` upward, nearest first.
fn node_modules_ancestors(root: &Path) -> impl Iterator<Item = PathBuf> + '_ {
    root.ancestors().map(|dir| dir.join("node_modules"))
}

/// Resolve options that find the VENDORED package by its bare name: `modules` is the
/// directory containing it rather than `node_modules`, so `exports` applies exactly as
/// it would for an installed package. Conditions mirror `bundler::resolve_options` (the
/// environment's own, plus `import`/`default` so plain ESM still resolves).
fn vendored_resolve_options(vendored: &Path, conditions: &[String], client: bool) -> ResolveOptions {
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
                ("react-server-dom-webpack/client".into(), "client.node.js".into()),
                // The flight WRITER only exists under `react-server`; `server.js`
                // throws on import, so picking it would break every RSC render.
                ("react-server-dom-webpack/server".into(), "server.node.js".into()),
            ],
        );

        let browser = aliases(
            root,
            &["module".into(), "browser".into(), "production".into()],
            true,
        );
        assert_eq!(
            names(&browser)[0],
            ("react-server-dom-webpack/client".into(), "client.browser.js".into()),
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
        std::fs::write(own.join("package.json"), r#"{"name":"react-server-dom-webpack"}"#).unwrap();

        assert!(
            aliases(root, &["node".into()], false).is_empty(),
            "the app's own react-server-dom-webpack must win over Next's vendored copy",
        );
    }

    #[test]
    fn no_copy_anywhere_aliases_nothing() {
        // Nothing to alias — the specifier stays unresolved, which is a FATAL build
        // diagnostic (bundler renders it with the RSC-runtime explanation).
        let dir = tempfile::tempdir().unwrap();
        assert!(aliases(dir.path(), &["node".into()], false).is_empty());
    }
}
