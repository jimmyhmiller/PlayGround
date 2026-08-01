mod tests {
    use std::process::Command;

    use tempfile::tempdir;

    use super::*;

    fn synthesize_asset_url(
        source_path: PathBuf,
        base: &str,
        inline_limit: usize,
        image_shape: ImageImportShape,
    ) -> Result<SpecialModule, String> {
        diffpack_default_loader::module::asset_url(
            source_path,
            base,
            inline_limit,
            |source| transform_module(Path::new("diffpack-url-asset.js"), source, Target::Server),
            |path, bytes, public_name| match image_shape {
                ImageImportShape::Url => Ok(None),
                ImageImportShape::NextObject {
                    responsive_variants,
                } => diffpack_next::static_image::module(
                    path,
                    bytes,
                    public_name,
                    base,
                    responsive_variants,
                    |source| {
                        transform_module(
                            Path::new("diffpack-image-import.js"),
                            source,
                            Target::Server,
                        )
                    },
                ),
            },
        )
    }

    /// The hinted partition search must return EXACTLY `partition_point`'s
    /// answer for every (query, hint) pair — it is the thing that makes the
    /// composed source map identical to the single-cursor composition. Checked
    /// exhaustively over a readable array with duplicate positions, line
    /// boundaries, and gaps, for every hint from 0 past the end.
    #[test]
    fn partition_point_from_hint_matches_partition_point_for_every_hint() {
        let token = |line: u32, column: u32| MapToken {
            generated_line: line,
            generated_column: column,
            source_line: 0,
            source_column: 0,
            name: None,
        };
        let readable: Vec<(MapToken, DenseModuleId)> = [
            (0, 0),
            (0, 4),
            (0, 4),
            (0, 9),
            (1, 0),
            (3, 2),
            (3, 2),
            (3, 7),
            (7, 0),
            (7, 1),
        ]
        .into_iter()
        .map(|(line, column)| (token(line, column), 0))
        .collect();
        for query_line in 0..9u32 {
            for query_column in 0..11u32 {
                let position = (query_line, query_column);
                let expected = readable.partition_point(|(token, _)| {
                    (token.generated_line, token.generated_column) <= position
                });
                for hint in 0..=readable.len() + 2 {
                    assert_eq!(
                        partition_point_from_hint(&readable, position, hint),
                        expected,
                        "position {position:?} hint {hint}"
                    );
                }
            }
        }
        assert_eq!(
            partition_point_from_hint(&[] as &[(MapToken, DenseModuleId)], (5, 5), 3),
            0
        );
    }

    /// `node`, with an inherited terminal-colour override stripped.
    ///
    /// Many tests here execute an emitted chunk and compare its stdout
    /// byte-for-byte. `console.log(6)` prints `6` down a pipe but
    /// `\x1b[33m6\x1b[39m` when node believes it is writing to a terminal, and
    /// an inherited `FORCE_COLOR` (set by plenty of terminal wrappers and CI
    /// runners) makes it believe exactly that. Every such assertion then fails
    /// for a reason that has nothing to do with the bundler. Removing the
    /// variable — rather than setting `NO_COLOR`, which node ignores in its
    /// presence and warns about on stderr — makes the output environment-
    /// independent.
    fn node_command() -> Command {
        let mut command = Command::new("node");
        command.env_remove("FORCE_COLOR");
        command
    }

    #[test]
    fn node_is_spawned_without_inherited_terminal_colour() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        // The hazard is real: with FORCE_COLOR set, node writes ANSI escapes even
        // down a pipe, and every stdout comparison in this module would fail.
        let coloured = Command::new("node")
            .env("FORCE_COLOR", "3")
            .args(["-e", "console.log(6)"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&coloured.stdout).contains('\u{1b}'),
            "expected node to colour its output under FORCE_COLOR"
        );
        // node_command() unsets it, whatever the parent environment holds.
        assert!(
            node_command()
                .get_envs()
                .any(|(key, value)| key == std::ffi::OsStr::new("FORCE_COLOR") && value.is_none()),
            "node_command must remove FORCE_COLOR from the child environment"
        );
        let plain = node_command()
            .args(["-e", "console.log(6)"])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&plain.stdout), "6\n");
    }

    #[test]
    fn bundles_typescript_dynamic_import_and_a_package_into_executable_javascript() {
        if node_command().arg("--version").output().is_err() {
            return;
        }

        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/tiny-package");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            directory.path().join("entry.ts"),
            r#"
                import message from "tiny-package";
                import { add } from "./math.js";
                console.log(`${message}:${add(2, 3)}`);
                import("./lazy.js").then(({ lazy }) => console.log(lazy));
            "#,
        )
        .unwrap();
        fs::write(
            directory.path().join("math.ts"),
            "export const add = (a: number, b: number): number => a + b;",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy-loaded';",
        )
        .unwrap();
        fs::write(
            package.join("package.json"),
            r#"{"name":"tiny-package","type":"module","exports":"./index.js"}"#,
        )
        .unwrap();
        fs::write(package.join("index.js"), "export default 'package-ok';").unwrap();

        let entry = directory.path().join("entry.ts");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 4);
        bundler.emit(&reachable, &output).unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "package-ok:5\nlazy-loaded\n"
        );
    }

    /// A dual-published package: `exports` sends `import` and `require` to two
    /// different files, exactly as `pg-pool` (and most of npm) does.
    fn write_dual_package(directory: &Path, name: &str, esm_body: &str, cjs_body: &str) {
        let package = directory.join("node_modules").join(name);
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            format!(
                r#"{{"name":"{name}","exports":{{".":{{"import":"./esm.mjs","require":"./cjs.js"}}}}}}"#
            ),
        )
        .unwrap();
        fs::write(package.join("esm.mjs"), esm_body).unwrap();
        fs::write(package.join("cjs.js"), cjs_body).unwrap();
    }

    /// A `require(...)` call site must resolve under the `require` export
    /// condition. Resolving it under `import` hands back a Module namespace where
    /// the caller expects the CommonJS export, and `class extends <namespace>`
    /// throws `Class extends value [object Module] is not a constructor` — which
    /// is exactly how `pg`'s `require('pg-pool')` died.
    #[test]
    fn a_require_call_site_resolves_under_the_require_condition() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        write_dual_package(
            directory.path(),
            "dual",
            "export default class Esm {}\n",
            "class Cjs {}\nmodule.exports = Cjs;\n",
        );
        fs::write(
            directory.path().join("entry.js"),
            "const Base = require(\"dual\");\nclass Sub extends Base {}\nconsole.log(new Sub().constructor.name === \"Sub\" ? Base.name : \"wrong\");\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "Cjs\n");
    }

    /// The other half of the same rule: an `import` of the identical specifier
    /// still resolves under `import`.
    #[test]
    fn an_import_of_the_same_package_still_resolves_under_the_import_condition() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        write_dual_package(
            directory.path(),
            "dual",
            "export const which = \"esm\";\n",
            "exports.which = \"cjs\";\n",
        );
        fs::write(
            directory.path().join("entry.js"),
            "import { which } from \"dual\";\nconsole.log(which);\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "esm\n");
    }

    /// One module, one specifier, both syntaxes, two different files. Node loads
    /// two module instances here and the runtime map holds one target per
    /// specifier, so the build refuses rather than silently giving one call site
    /// the other's module.
    #[test]
    fn one_specifier_reached_both_ways_that_resolves_two_ways_is_a_hard_error() {
        let directory = tempdir().unwrap();
        write_dual_package(
            directory.path(),
            "dual",
            "export const which = \"esm\";\n",
            "exports.which = \"cjs\";\n",
        );
        fs::write(
            directory.path().join("entry.js"),
            "const eager = require(\"dual\");\nexport const lazy = import(\"dual\");\nconsole.log(eager, lazy);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (_, update) = discover(&entry).unwrap();
        let fatal = update
            .diagnostics
            .iter()
            .find(|diagnostic| {
                matches!(
                    diagnostic.kind,
                    DiagnosticKind::SpecifierResolvesTwoWays { .. }
                )
            })
            .expect("reaching one specifier both ways must be reported");
        assert!(fatal.is_fatal());
        assert!(fatal.message.contains("entry.js"), "{}", fatal.message);
        assert!(fatal.message.contains("esm.mjs"), "{}", fatal.message);
        assert!(fatal.message.contains("cjs.js"), "{}", fatal.message);
    }

    /// A package whose `exports` sends both conditions to the SAME file is not a
    /// conflict, so reaching it both ways is fine.
    #[test]
    fn one_specifier_reached_both_ways_that_resolves_the_same_way_is_not_an_error() {
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/single");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            r#"{"name":"single","exports":{".":{"import":"./index.js","require":"./index.js"}}}"#,
        )
        .unwrap();
        fs::write(package.join("index.js"), "exports.which = \"one\";\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const eager = require(\"single\");\nexport const lazy = import(\"single\");\nconsole.log(eager, lazy);\n",
        )
        .unwrap();
        let (_, update) = discover(&directory.path().join("entry.js")).unwrap();
        assert!(
            !update.diagnostics.iter().any(|diagnostic| matches!(
                diagnostic.kind,
                DiagnosticKind::SpecifierResolvesTwoWays { .. }
            )),
            "{:?}",
            update.diagnostics
        );
    }

    /// `export const p = import("./a")` holds a real dependency. The dependency
    /// scan used to stop at the `from` clause of an `export … from` and never
    /// look inside an exported declaration, so this module was bundled with no
    /// edge at all and the emitted `import()` threw MODULE_NOT_FOUND.
    #[test]
    fn a_dynamic_import_inside_an_exported_declaration_is_bundled_and_runs() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const value = \"lazy-value\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("map.js"),
            "export const Map = { a: import(\"./lazy.js\") };\nexport const run = async () => (await Map.a).value;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { run } from \"./map.js\";\nrun().then((v) => console.log(v));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(
            reachable.len(),
            3,
            "the dynamic target must be in the graph"
        );
        assert_eq!(bundle_and_run(directory.path()), "lazy-value\n");
    }

    /// A `require(...)` inside an exported declaration is the same hole.
    #[test]
    fn a_require_inside_an_exported_declaration_is_bundled_and_runs() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("dep.js"),
            "exports.value = \"dep-value\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "export const dep = require(\"./dep.js\");\nconsole.log(dep.value);\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "dep-value\n");
    }

    /// Bundles `entry.js` out of an already-populated directory and runs the
    /// result under Node, returning its stdout. The interop tests below are all
    /// "what does the emitted program actually print", which is the only level
    /// at which a runtime helper can be pinned.
    fn bundle_and_run(directory: &Path) -> String {
        let entry = directory.join("entry.js");
        let output = directory.join("dist/bundle.js");
        let (bundler, update) = discover(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        String::from_utf8_lossy(&executed.stdout).into_owned()
    }

    /// `import { missing } from "./legacy.cjs"` is a hard error in Node, and
    /// must not evaluate to `undefined` here either — not even when the module
    /// stamps the `__esModule` convention marker on itself, which is exactly
    /// the case the interop's own CommonJS marker used to wave through.
    #[test]
    fn a_named_import_a_commonjs_module_does_not_provide_is_a_hard_error() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("marked.cjs"),
            "exports.__esModule = true;\nexports.present = \"present-val\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                import { present, missingName } from "./marked.cjs";
                import marked from "./marked.cjs";
                console.log("present:" + present);
                console.log("default-is-module-exports:" + (marked.present === "present-val"));
                try {
                  console.log("missing:" + missingName);
                } catch (error) {
                  console.log("threw:" + (error instanceof SyntaxError));
                  console.log("message:" + error.message);
                }
            "#,
        )
        .unwrap();

        let stdout = bundle_and_run(directory.path());
        let lines = stdout.lines().collect::<Vec<_>>();
        assert_eq!(
            &lines[..3],
            [
                "present:present-val",
                "default-is-module-exports:true",
                "threw:true",
            ],
            "{stdout}"
        );
        // The error names both the module and the export, the way Node's does.
        let message = lines[3];
        assert!(message.contains("./marked.cjs"), "{message}");
        assert!(message.contains("missingName"), "{message}");
    }

    /// The `__esModule` interop. A CommonJS module that stamps the marker AND owns a
    /// `default` was compiled down from ESM (TypeScript / Babel / SWC output, which is
    /// most of npm), so `import X from` it must bind THAT default — not the exports
    /// object wrapping it. Binding the wrapper is silent until the value is used as
    /// what it claims to be: `next-auth/providers/credentials` is exactly this shape,
    /// and cal.com's next-auth config died on `o(...) is not a function` because the
    /// provider factory came back as `{ __esModule: true, default: fn }`.
    #[test]
    fn a_default_import_of_a_transpiled_commonjs_module_binds_its_default_export() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("provider.cjs"),
            "Object.defineProperty(exports, \"__esModule\", { value: true });\n\
             exports.default = Credentials;\n\
             exports.named = \"named-val\";\n\
             function Credentials(options) { return { id: \"credentials\", options }; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                import Credentials from "./provider.cjs";
                import { named } from "./provider.cjs";
                import * as ns from "./provider.cjs";
                console.log("typeof:" + typeof Credentials);
                console.log("call:" + Credentials({ a: 1 }).id);
                console.log("named:" + named);
                console.log("ns-default-is-the-same:" + (ns.default === Credentials));
            "#,
        )
        .unwrap();
        let stdout = bundle_and_run(directory.path());
        assert_eq!(
            stdout.lines().collect::<Vec<_>>(),
            [
                "typeof:function",
                "call:credentials",
                "named:named-val",
                "ns-default-is-the-same:true",
            ],
            "{stdout}"
        );
    }

    /// The negation of the rule above, so it stays a rule and not a guess: a CommonJS
    /// module with NO `__esModule` marker keeps Node's semantics — a default import
    /// binds `module.exports`, even when the object happens to carry a `default` key.
    #[test]
    fn a_default_import_of_an_unmarked_commonjs_module_still_binds_module_exports() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("plain.cjs"),
            "module.exports = { default: \"inner\", other: \"o\" };\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import plain from \"./plain.cjs\";\nconsole.log(\"default:\" + JSON.stringify(plain));\n",
        )
        .unwrap();
        assert_eq!(
            bundle_and_run(directory.path()),
            "default:{\"default\":\"inner\",\"other\":\"o\"}\n",
        );
    }

    /// `serverExternalPackages` (next.config): a listed package is NOT bundled into a
    /// server graph — it stays a runtime `require` from `node_modules`. Apps use the
    /// list precisely because bundling the package fails, so a build that ignores it
    /// turns working configuration into a fatal error: cal.com externalizes
    /// `rest-facade`, whose `require('superagent-proxy')` sits behind a runtime `if`
    /// and names a package that is deliberately not installed.
    #[test]
    fn a_server_external_package_is_not_bundled_and_its_own_imports_are_not_resolved() {
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/rest-facade");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            r#"{"name":"rest-facade","main":"index.js"}"#,
        )
        .unwrap();
        // The shape that makes the list necessary: an import of a package that is not
        // installed, reached only on a branch the app never takes.
        fs::write(
            package.join("index.js"),
            "exports.Client = function (o) { if (o.proxy) require('superagent-proxy'); };\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { Client } from \"rest-facade\";\nexport const c = Client;\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");

        // Bundled (the default), the uninstalled transitive import is a fatal diagnostic.
        let (_, update) = discover(&entry).unwrap();
        assert!(
            update.diagnostics.iter().any(|d| d.is_fatal()
                && matches!(d.kind, DiagnosticKind::UnresolvedImport { .. })
                && d.message.contains("superagent-proxy")),
            "without the list the uninstalled dependency is fatal: {:?}",
            update.diagnostics,
        );

        // Listed as a server external, the package is never resolved at all — so
        // nothing inside it can fail the build, and it is not a graph module.
        let config = BuildConfig {
            target: Target::Server,
            server_external_packages: vec!["rest-facade".to_string()],
            ..BuildConfig::default()
        };
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(
            update.diagnostics.iter().all(|d| !d.is_fatal()),
            "an externalized package cannot fail the build: {:?}",
            update.diagnostics,
        );
        assert!(
            !bundler
                .graph
                .ids
                .iter()
                .any(|id| id.contains("rest-facade")),
            "the external must not be a graph module: {:?}",
            bundler.graph.ids,
        );
    }

    /// A CLIENT graph must ignore the list: a browser has no `node_modules` to require
    /// from at runtime, so externalizing there would emit a chunk that dies on the
    /// throw-on-use stub with a zero build exit code.
    #[test]
    fn a_server_external_package_is_still_bundled_for_the_browser() {
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/jose");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            r#"{"name":"jose","main":"index.js"}"#,
        )
        .unwrap();
        fs::write(
            package.join("index.js"),
            "exports.sign = () => \"signed\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { sign } from \"jose\";\nexport const s = sign;\n",
        )
        .unwrap();
        let config = BuildConfig {
            target: Target::Client,
            server_external_packages: vec!["jose".to_string()],
            ..BuildConfig::default()
        };
        let (bundler, update) =
            discover_direct_with_config(&directory.path().join("entry.js"), &config).unwrap();
        assert!(
            update.diagnostics.iter().all(|d| !d.is_fatal()),
            "{:?}",
            update.diagnostics
        );
        assert!(
            bundler.graph.ids.iter().any(|id| id.contains("jose")),
            "the browser graph still bundles it: {:?}",
            bundler.graph.ids,
        );
    }

    /// The interop namespace copies `module.exports`' keys at wrap time, which
    /// in an ESM<->CJS cycle is a PARTIALLY populated object. A key the module
    /// assigns after that point must still be readable through a named import
    /// rather than being frozen out (or, worse, reported as not provided).
    #[test]
    fn a_commonjs_export_assigned_after_the_interop_wrap_is_still_visible() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("legacy.cjs"),
            "exports.early = \"early\";\nrequire(\"./esm.js\");\nexports.late = \"late\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("esm.js"),
            "import { early, late } from \"./legacy.cjs\";\nexport function read() { return early + \"/\" + late; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import \"./legacy.cjs\";\nimport { read } from \"./esm.js\";\nconsole.log(\"read:\" + read());\n",
        )
        .unwrap();

        assert_eq!(bundle_and_run(directory.path()), "read:early/late\n");
    }

    /// One CommonJS module has exactly one interop namespace: re-running the
    /// interop over the same `module.exports` (`export * as ns from` re-runs it
    /// on every read) must return the same object, and running it over a
    /// namespace it already produced must be a no-op instead of nesting a
    /// second `default` around it.
    #[test]
    fn the_commonjs_interop_namespace_is_stable_and_idempotent() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("legacy.cjs"),
            "exports.value = \"legacy-value\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("hub.js"),
            "export * as legacy from \"./legacy.cjs\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                import * as hub from "./hub.js";
                import * as direct from "./legacy.cjs";
                console.log("stable:" + (hub.legacy === hub.legacy));
                console.log("shared:" + (hub.legacy === direct));
                console.log("value:" + hub.legacy.value);
                console.log("not-nested:" + (hub.legacy.default.default === undefined));
            "#,
        )
        .unwrap();

        assert_eq!(
            bundle_and_run(directory.path()),
            "stable:true\nshared:true\nvalue:legacy-value\nnot-nested:true\n"
        );
    }

    #[test]
    fn url_asset_import_emits_a_content_hashed_file_and_exports_its_public_url() {
        let directory = tempdir().unwrap();
        let css = ".brand { color: red; }\n";
        fs::write(directory.path().join("styles.css"), css).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import url from './styles.css?url';\nconsole.log(url);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        // The entry plus the distinct `styles.css?url` asset module.
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 2, "{reachable:?}");
        bundler.emit(&reachable, &output).unwrap();

        // The bundle exports the asset's public URL, not the raw path.
        let bundle = fs::read_to_string(&output).unwrap();
        let url = bundle
            .lines()
            .find_map(|line| line.find("/assets/styles-").map(|start| &line[start..]))
            .and_then(|rest| rest.split('"').next())
            .expect("bundle should reference the hashed asset url");
        assert!(url.ends_with(".css"), "{url}");

        // The content-hashed asset file is copied next to the bundle with the
        // exact original bytes.
        let asset_name = url.trim_start_matches("/assets/");
        let asset_path = directory.path().join("dist/assets").join(asset_name);
        assert_eq!(fs::read_to_string(&asset_path).unwrap(), css);

        // A second, identical asset would hash to the same name (determinism).
        assert_eq!(
            asset_name,
            asset_public_name(Path::new("styles.css"), content_hash(css.as_bytes()))
        );

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(
                String::from_utf8_lossy(&executed.stdout),
                format!("{url}\n")
            );
        }
    }

    /// A module that is BOTH named-imported and bare-`require`d by the same
    /// importer keeps its whole-module demand: `require("m")` hands out
    /// `module.exports` wholesale, so the import statement's named list must not
    /// downgrade the demand and shake off exports the require observably reads.
    /// This is exactly the shape of the next adapter's lazy island pins (a
    /// require thunk beside a named import of `control-boundary`), where the
    /// downgrade shook off the island's `default` export and broke hydration.
    #[test]
    fn a_bare_require_beside_a_named_import_keeps_every_export() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("m.js"),
            "export default function island() { return \"DEFAULT\"; }\n\
             export const named = \"NAMED\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { named } from './m.js';\n\
             const pins = [() => require('./m.js')];\n\
             globalThis.__pins = pins;\n\
             console.log(named, typeof pins[0]().default === 'function' ? pins[0]().default() : 'MISSING');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), "NAMED DEFAULT\n");
        }
    }

    #[test]
    fn raw_import_inlines_the_file_contents_as_a_string() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("note.txt"), "hello from raw").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import raw from './note.txt?raw';\nconsole.log(raw);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(
                String::from_utf8_lossy(&executed.stdout),
                "hello from raw\n"
            );
        }
    }

    #[test]
    fn worker_query_import_emits_a_worker_chunk_and_references_its_public_url() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("job.js"),
            "self.onmessage = (event) => self.postMessage(event.data * 2);\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import JobWorker from './job.js?worker';\nconst worker = new JobWorker();\nconsole.log(worker);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let bundle = fs::read_to_string(&output).unwrap();
        // The worker URL placeholder must be fully substituted — a leftover
        // `__diffpack_worker__…__` would 404 at runtime.
        assert!(
            !bundle.contains("__diffpack_worker__"),
            "worker placeholder left in bundle:\n{bundle}"
        );
        // The bundle spawns a `Worker` at the emitted chunk's public URL.
        let url = bundle
            .lines()
            .find_map(|line| line.find("/assets/job-").map(|start| &line[start..]))
            .and_then(|rest| rest.split('"').next())
            .expect("bundle should reference the worker chunk url");
        assert!(url.ends_with(".worker.js"), "{url}");

        // The self-contained worker chunk is emitted next to the bundle and
        // carries the entry's code.
        let worker_path = directory
            .path()
            .join("dist/assets")
            .join(url.trim_start_matches("/assets/"));
        assert!(worker_path.is_file(), "missing {}", worker_path.display());
        let worker_code = fs::read_to_string(&worker_path).unwrap();
        assert!(
            worker_code.contains("postMessage"),
            "worker chunk should bundle the entry code:\n{worker_code}"
        );
    }

    #[test]
    fn worker_inline_combo_reports_a_specific_unimplemented_error() {
        let error = match diffpack_default_loader::module::query_module(
            &ResourceId::parse("/abs/job.js?worker&inline"),
            "/",
            0,
            None,
            |path, source| transform_module(path, source, Target::Client),
        ) {
            Err(error) => error,
            Ok(_) => panic!("?worker&inline should be refused"),
        };
        assert!(error.contains("?worker&inline"), "{error}");
        assert!(!error.contains("No such file or directory"), "{error}");
    }

    #[test]
    fn inline_query_import_embeds_the_asset_as_a_data_uri() {
        let directory = tempdir().unwrap();
        let png: &[u8] = b"\x89PNG\r\n\x1a\nfake-png-bytes";
        fs::write(directory.path().join("pixel.png"), png).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import pixel from './pixel.png?inline';\nconsole.log(pixel);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let bundle = fs::read_to_string(&output).unwrap();
        let expected = format!("data:image/png;base64,{}", base64_encode(png));
        assert!(
            bundle.contains(&expected),
            "bundle should embed the data URI:\n{bundle}"
        );
        // An inlined asset emits no separate file.
        let assets_dir = directory.path().join("dist/assets");
        assert!(
            !assets_dir.exists() || fs::read_dir(&assets_dir).unwrap().next().is_none(),
            "?inline must not emit a separate asset file"
        );
    }

    #[test]
    fn wasm_init_import_emits_the_module_and_a_default_initializer() {
        let directory = tempdir().unwrap();
        // A minimal well-formed WebAssembly module: the `\0asm` magic + version 1.
        let wasm: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        fs::write(directory.path().join("add.wasm"), wasm).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import init from './add.wasm?init';\ninit().then((instance) => console.log(instance));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let bundle = fs::read_to_string(&output).unwrap();
        // The instantiation helper is inlined.
        assert!(bundle.contains("WebAssembly"), "helper missing:\n{bundle}");
        assert!(bundle.contains("instantiate"), "helper missing:\n{bundle}");

        // The `.wasm` payload takes the content-hashed asset pipeline (default
        // inline limit is 0, so it is a real file, not a data URI) and the
        // initializer closes over its URL.
        let url = bundle
            .lines()
            .find_map(|line| line.find("/assets/add-").map(|start| &line[start..]))
            .and_then(|rest| rest.split('"').next())
            .expect("bundle should reference the hashed wasm url");
        assert!(url.ends_with(".wasm"), "{url}");
        let wasm_path = directory
            .path()
            .join("dist/assets")
            .join(url.trim_start_matches("/assets/"));
        assert_eq!(fs::read(&wasm_path).unwrap(), wasm);
    }

    #[test]
    fn wasm_init_inlines_a_small_module_as_a_data_uri() {
        let directory = tempdir().unwrap();
        let wasm: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        fs::write(directory.path().join("tiny.wasm"), wasm).unwrap();
        // A generous inline limit forces the payload into a `data:` URI.
        // Guard: `?init` only applies to `.wasm`.
        assert!(
            diffpack_default_loader::module::query_module(
                &ResourceId::parse("tiny.js?init"),
                "/",
                4096,
                None,
                |path, source| transform_module(path, source, Target::Server),
            )
            .is_err()
        );
        let path = directory.path().join("tiny.wasm");
        let module = diffpack_default_loader::module::query_module(
            &ResourceId::parse(&format!("{}?init", path.display())),
            "/",
            4096,
            None,
            |path, source| transform_module(path, source, Target::Server),
        )
        .unwrap()
        .unwrap();
        assert!(
            module.assets.is_empty(),
            "small wasm should inline, not emit a file"
        );
        assert!(
            module.code.contains("data:application/wasm;base64,"),
            "small wasm should be a data URI:\n{}",
            module.code
        );
    }

    #[test]
    fn default_asset_import_emits_a_hashed_file_and_exports_its_url() {
        let directory = tempdir().unwrap();
        let svg = "<svg></svg>";
        fs::write(directory.path().join("logo.svg"), svg).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import logo from './logo.svg';\nconsole.log(logo);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 2, "{reachable:?}");
        bundler.emit(&reachable, &output).unwrap();

        let bundle = fs::read_to_string(&output).unwrap();
        let url = bundle
            .lines()
            .find_map(|line| line.find("/assets/logo-").map(|start| &line[start..]))
            .and_then(|rest| rest.split('"').next())
            .expect("bundle should reference the hashed asset url");
        assert!(url.ends_with(".svg"), "{url}");
        let asset_path = directory
            .path()
            .join("dist/assets")
            .join(url.trim_start_matches("/assets/"));
        assert_eq!(fs::read_to_string(&asset_path).unwrap(), svg);
    }

    #[test]
    fn an_unrecognized_loader_query_reports_a_specific_error() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("thing.js"), "export const x = 1;").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import c from './thing.js?mystery';\nconsole.log(c);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match discover_direct(&entry) {
            Ok(_) => panic!("an unimplemented loader must fail the build, not silently succeed"),
            Err(error) => error,
        };
        assert!(
            error.contains("unrecognized loader query `?mystery`"),
            "{error}"
        );
        assert!(!error.contains("No such file or directory"), "{error}");
    }

    #[test]
    fn a_tsr_split_query_on_a_non_route_file_reports_a_specific_error() {
        // `?tsr-split` is implemented, but only for route files. Asking a plain
        // module to produce a split module is a clear error, not a silent empty
        // module or a filesystem crash.
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("thing.js"), "export const x = 1;").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import c from './thing.js?tsr-split=component';\nconsole.log(c);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match discover_tanstack_with_config(&entry, &BuildConfig::default()) {
            Ok(_) => panic!("a tsr-split on a non-route file must fail the build"),
            Err(error) => error,
        };
        assert!(error.contains("not a splittable route file"), "{error}");
        assert!(!error.contains("No such file or directory"), "{error}");
    }

    /// Every wording a reader would chase down the WRONG path: a JSX syntax error
    /// in their own file, or a resolution failure. An unhandled source is neither.
    fn assert_not_misreported(error: &str) {
        for misleading in [
            "Unexpected JSX expression",
            "cannot resolve",
            "unresolved",
            "npm install",
            "No such file or directory",
        ] {
            assert!(!error.contains(misleading), "{misleading}: {error}");
        }
    }

    #[test]
    fn a_vue_component_whose_compiler_is_missing_names_the_package_not_a_jsx_error() {
        // A `.vue` file is not JavaScript. Parsing it as JavaScript reports
        // `Unexpected JSX expression` on the app's `<template>`, blaming the app
        // for diffpack's own gap. It is compiled by the APP's OWN
        // `@vue/compiler-sfc`; this fixture project has no `node_modules` at all,
        // so the compile must fail loudly, naming the file and the package —
        // never fall back to reading the component as JavaScript.
        // (Requires `node` on PATH, as every diffpack build already does.)
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("App.vue"),
            "<script setup lang=\"ts\">\nconst greeting = 'hi';\n</script>\n\n\
             <template>\n  <h1>{{ greeting }}</h1>\n</template>\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import App from './App.vue';\nconsole.log(App);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match discover_direct(&entry) {
            Ok(_) => panic!("a `.vue` component has no compiler here; it must fail the build"),
            Err(error) => error,
        };
        assert!(error.contains("App.vue"), "{error}");
        assert!(error.contains("Vue single-file component"), "{error}");
        assert!(error.contains("@vue/compiler-sfc"), "{error}");
        assert_not_misreported(&error);
    }

    #[test]
    fn a_svelte_component_whose_compiler_is_missing_names_the_package() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("App.svelte"),
            "<script lang=\"ts\">\n  let count = 0;\n</script>\n\n<h1>{count}</h1>\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import App from './App.svelte';\nconsole.log(App);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match discover_direct(&entry) {
            Ok(_) => panic!("a `.svelte` component has no compiler here; it must fail the build"),
            Err(error) => error,
        };
        assert!(error.contains("App.svelte"), "{error}");
        assert!(error.contains("Svelte component"), "{error}");
        assert!(error.contains("svelte/compiler"), "{error}");
        assert_not_misreported(&error);
    }

    /// A build configured like a Vite project: a real project root (so
    /// root-absolute and `public/` imports resolve) and the client target.
    fn vite_like_config(root: &Path, aliases: Vec<(String, String)>) -> BuildConfig {
        BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases,
            conditions: vec!["module".into(), "browser".into()],
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            private_chunk_names: Vec::new(),
            target: Target::Client,
            server_external_packages: Vec::new(),
            source_policy: Arc::new(
                diffpack_default_loader::source_policy::NoSourceIntegrationPolicy,
            ),
            hmr: false,
            scss: diffpack_default_loader::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess {
                root: Some(root.to_path_buf()),
                postcss: None,
            },
            jsx_extensions: diffpack_core::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: diffpack_core::transform::JsxConfig::default(),
            source_maps: false,
        }
    }

    #[test]
    fn a_root_absolute_import_of_a_public_file_is_its_url_not_an_emitted_asset() {
        // Vite: `import icons from "/icons.svg"` is `<root>/icons.svg`, not the
        // filesystem path `/icons.svg`. With no such file in the root but one in
        // `public/`, the import is the file's PUBLIC URL — `public/` is copied to
        // the site root verbatim, so hashing and re-emitting it would mint a
        // second copy at a URL the app's own build never produces.
        // (Vue's SFC compiler emits exactly this import for a `<use href="/icons.svg#x">`.)
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("public")).unwrap();
        fs::write(root.join("public/icons.svg"), "<svg/>").unwrap();
        fs::write(
            root.join("entry.js"),
            "import icons from '/icons.svg';\nconsole.log(icons);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(
            code.contains("\"/icons.svg\"") || code.contains("'/icons.svg'"),
            "{code}"
        );
        // No hashed copy: the public file is served from the site root as-is.
        assert!(
            !root.join("dist/assets").exists(),
            "a public file must not be re-emitted"
        );
    }

    /// Write a package under `<root>/node_modules/<name>/` from `(relative path,
    /// contents)` pairs. `package.json` is one of the pairs, so the test owns the
    /// whole manifest (`browser`, `exports`, …).
    fn write_package_files(root: &Path, name: &str, files: &[(&str, &str)]) {
        let base = root.join("node_modules").join(name);
        for (relative, contents) in files {
            let path = base.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }
    }

    #[test]
    fn an_object_browser_field_remaps_a_packages_own_relative_import() {
        // The classic pre-`exports` substitution map in its OBJECT form: keys are
        // paths RELATIVE TO THE PACKAGE ROOT, and they rewrite the package's own
        // internal `./node.js` import — not just its entry point. axios ships
        // exactly this shape to keep `lib/adapters/http.js` (which imports `http`,
        // `https`, `zlib`, …) out of browser bundles. Honouring only the string
        // form drags the Node implementation, and every Node built-in it touches,
        // into the client graph.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package_files(
            root,
            "swappable",
            &[
                (
                    "package.json",
                    r#"{
                      "name": "swappable",
                      "version": "1.0.0",
                      "type": "module",
                      "main": "./index.js",
                      "exports": { ".": "./index.js" },
                      "browser": { "./lib/node.js": "./lib/browser.js" }
                    }"#,
                ),
                ("index.js", "export { impl } from './lib/node.js';\n"),
                (
                    "lib/node.js",
                    "import zlib from 'zlib';\nexport const impl = 'NODE_IMPL' + typeof zlib;\n",
                ),
                ("lib/browser.js", "export const impl = 'BROWSER_IMPL';\n"),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { impl } from 'swappable';\nconsole.log(impl);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(
            !code.contains("NODE_IMPL"),
            "the node variant must not be bundled: {code}"
        );
        assert_eq!(run_node(&output), "BROWSER_IMPL\n");
    }

    #[test]
    fn a_false_browser_field_entry_stubs_a_module_out_of_the_browser_graph() {
        // `"browser": { "./lib/node.js": false }` means "this module is empty in a
        // browser". webpack/Vite substitute an empty module; leaving the real one
        // in place pulls its Node built-ins into the client bundle.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package_files(
            root,
            "stubbable",
            &[
                (
                    "package.json",
                    r#"{
                      "name": "stubbable",
                      "version": "1.0.0",
                      "type": "module",
                      "main": "./index.js",
                      "exports": { ".": "./index.js" },
                      "browser": { "./lib/node.js": false }
                    }"#,
                ),
                (
                    "index.js",
                    "import * as node from './lib/node.js';\nexport const impl = typeof node.impl;\n",
                ),
                (
                    "lib/node.js",
                    "import zlib from 'zlib';\nexport const impl = 'NODE_IMPL' + typeof zlib;\n",
                ),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { impl } from 'stubbable';\nconsole.log(impl);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(
            !code.contains("NODE_IMPL"),
            "the stubbed module must not be bundled: {code}"
        );
        // webpack's semantics: the excluded module is an object with nothing on it.
        assert_eq!(run_node(&output), "undefined\n");
    }

    #[test]
    fn a_try_guarded_require_of_an_uninstalled_package_is_a_warning_not_a_build_error() {
        // `try { require("accelerator") } catch {}` is how packages with native or
        // platform-specific accelerators declare an optional dependency (`ws` ->
        // bufferutil/utf-8-validate, `pg` -> pg-native, `sharp` -> @img/*, jsdom ->
        // canvas). Node throws MODULE_NOT_FOUND at that `require` and the `catch`
        // supplies the fallback, so the program is CORRECT with the package absent.
        // Failing the build rejects code that runs.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package_files(
            root,
            "guarded",
            &[
                (
                    "package.json",
                    r#"{ "name": "guarded", "version": "1.0.0", "main": "./index.js" }"#,
                ),
                (
                    "index.js",
                    "let fast;\ntry { fast = require('accelerator'); } catch { fast = null; }\n\
                     module.exports.impl = fast ? 'FAST' : 'FALLBACK_OK';\n",
                ),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { impl } from 'guarded';\nconsole.log(impl);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(
            update
                .diagnostics
                .iter()
                .all(|diagnostic| !diagnostic.is_fatal()),
            "a guarded optional require must not be fatal: {:?}",
            update.diagnostics
        );
        // Reported, though: an omission nobody is told about is a silent fallback.
        assert!(
            update.diagnostics.iter().any(|diagnostic| matches!(
                &diagnostic.kind,
                DiagnosticKind::OptionalDependencyMissing { specifier, .. }
                    if specifier == "accelerator"
            )),
            "the omission must still be reported: {:?}",
            update.diagnostics
        );
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        // Node semantics preserved end to end: the require throws, the catch runs.
        assert_eq!(run_node(&output), "FALLBACK_OK\n");
    }

    #[test]
    fn an_unguarded_require_of_the_same_package_stays_a_fatal_build_error() {
        // The counterpart that keeps the rule honest. One reference outside a `try`
        // means some path really does need the module, so its absence still breaks
        // the artifact — a typo inside a package must not be laundered into a
        // warning just because the same name also appears in a guarded require.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package_files(
            root,
            "unguarded",
            &[
                (
                    "package.json",
                    r#"{ "name": "unguarded", "version": "1.0.0", "main": "./index.js" }"#,
                ),
                (
                    "index.js",
                    "let fast;\ntry { fast = require('accelerator'); } catch { fast = null; }\n\
                     const always = require('accelerator');\nmodule.exports.impl = always;\n",
                ),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { impl } from 'unguarded';\nconsole.log(impl);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        let _ = &bundler;
        assert!(
            update.diagnostics.iter().any(|diagnostic| matches!(
                &diagnostic.kind,
                DiagnosticKind::UnresolvedImport { specifier, .. } if specifier == "accelerator"
            )),
            "an unguarded reference must stay fatal: {:?}",
            update.diagnostics
        );
    }

    #[test]
    fn a_foreign_runtimes_scheme_specifier_is_classified_like_a_node_builtin() {
        // `node:fs` already means "the host provides this". Every other runtime uses
        // the same reserved shape for its own built-ins, and diffpack's rule was
        // accidentally Node-only — so `cloudflare:sockets` (imported by pg-cloudflare,
        // which `pg` pulls in) was reported as a missing package with the impossible
        // advice `npm install cloudflare:sockets`.
        assert_eq!(
            host_provided_scheme("cloudflare:sockets"),
            Some("cloudflare")
        );
        assert_eq!(host_provided_scheme("bun:ffi"), Some("bun"));
        assert_eq!(host_provided_scheme("node:fs"), Some("node"));
        // Resource URLs address bytes rather than naming a host module; diffpack
        // cannot load any of them, so they must keep failing.
        assert_eq!(host_provided_scheme("https://esm.sh/react"), None);
        assert_eq!(host_provided_scheme("data:text/javascript,export{}"), None);
        assert_eq!(host_provided_scheme("file:///tmp/x.js"), None);
        // Ordinary specifiers, and a Windows drive path, are not schemes.
        assert_eq!(host_provided_scheme("react"), None);
        assert_eq!(host_provided_scheme("./local.js"), None);
        assert_eq!(host_provided_scheme("@scope/pkg"), None);
        assert_eq!(host_provided_scheme("C:/project/src/x.js"), None);
        assert_eq!(host_provided_scheme("cloudflare:"), None);
    }

    #[test]
    fn a_foreign_runtime_module_is_external_on_a_server_graph_and_fatal_on_a_client_one() {
        // The consequence of the classification above, at both targets.
        let directory = tempdir().unwrap();
        let root = directory.path();
        // pg-cloudflare's shape: a package whose socket implementation reaches for the
        // Workers runtime module, pulled in unconditionally by its parent (`pg`).
        write_package_files(
            root,
            "workers-socket",
            &[
                (
                    "package.json",
                    r#"{ "name": "workers-socket", "version": "1.0.0", "main": "./index.js" }"#,
                ),
                (
                    "index.js",
                    "module.exports.connect = async () => (await import('cloudflare:sockets')).connect;\n",
                ),
            ],
        );
        fs::write(
            root.join("entry.js"),
            "import { connect } from 'workers-socket';\nexport { connect };\n",
        )
        .unwrap();
        let entry = root.join("entry.js");

        let mut server = vite_like_config(root, Vec::new());
        server.target = Target::Server;
        server.conditions = vec!["node".into()];
        let (_, update) = discover_direct_with_config(&entry, &server).unwrap();
        assert!(
            update
                .diagnostics
                .iter()
                .all(|diagnostic| !diagnostic.is_fatal()),
            "a host-provided module must not fail a server build: {:?}",
            update.diagnostics
        );
        assert!(
            update.diagnostics.iter().any(|diagnostic| matches!(
                &diagnostic.kind,
                DiagnosticKind::HostProvidedModule { specifier, .. }
                    if specifier == "cloudflare:sockets"
            )),
            "the external must still be reported: {:?}",
            update.diagnostics
        );

        // A browser has no host to provide it, so it stays fatal — same as `node:fs`.
        let client = vite_like_config(root, Vec::new());
        let (_, update) = discover_direct_with_config(&entry, &client).unwrap();
        assert!(
            update
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.is_fatal()),
            "a browser graph has no host runtime: {:?}",
            update.diagnostics
        );
    }

    #[test]
    fn a_root_absolute_import_prefers_a_file_in_the_project_root() {
        // `/lib/util.js` with `<root>/lib/util.js` present is that module, not a
        // public URL — Vite resolves root-relative first.
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("lib")).unwrap();
        fs::write(
            root.join("lib/util.js"),
            "export const value = 'ROOT_RELATIVE_OK';\n",
        )
        .unwrap();
        fs::write(
            root.join("entry.js"),
            "import { value } from '/lib/util.js';\nconsole.log(value);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let config = vite_like_config(root, Vec::new());
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(code.contains("ROOT_RELATIVE_OK"), "{code}");
    }

    #[test]
    fn a_dedupe_alias_still_resolves_a_subpath_through_the_package_exports_map() {
        // Vite's `resolve.dedupe` pins a package to `<root>/node_modules/<pkg>`,
        // which diffpack carries as a directory alias. A SUBPATH cannot be
        // answered by joining onto that directory: `svelte/internal/client` is a
        // key in the package's `exports` map, not a file at that path, so the
        // join produced a path that does not exist and the build failed on a
        // package that is installed.
        let directory = tempdir().unwrap();
        let root = directory.path();
        let package = root.join("node_modules/widget");
        fs::create_dir_all(package.join("src/internal")).unwrap();
        fs::write(
            package.join("package.json"),
            "{\"name\":\"widget\",\"exports\":{\".\":\"./src/index.js\",\
             \"./internal/client\":\"./src/internal/client-impl.js\"}}",
        )
        .unwrap();
        fs::write(package.join("src/index.js"), "export const main = 1;\n").unwrap();
        fs::write(
            package.join("src/internal/client-impl.js"),
            "export const internalValue = 'EXPORTS_SUBPATH_OK';\n",
        )
        .unwrap();
        fs::write(
            root.join("entry.js"),
            "import { internalValue } from 'widget/internal/client';\nconsole.log(internalValue);\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        let aliases = vec![("widget".to_string(), package.to_string_lossy().into_owned())];
        let config = vite_like_config(root, aliases);
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(code.contains("EXPORTS_SUBPATH_OK"), "{code}");
    }

    #[test]
    fn an_extension_no_loader_handles_is_named_not_parsed_as_javascript() {
        // diffpack does not know what a `.graphql` file is, and must say exactly
        // that rather than invent a compiler for it or parse it as JavaScript.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("schema.graphql"),
            "type Query { hello: String }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import schema from './schema.graphql';\nconsole.log(schema);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match discover_direct(&entry) {
            Ok(_) => panic!("no loader handles `.graphql`; it must fail the build"),
            Err(error) => error,
        };
        assert!(
            error.contains("no loader handles the `.graphql` extension"),
            "{error}"
        );
        assert!(error.contains("./schema.graphql?raw"), "{error}");
        assert!(!error.contains("compiler"), "{error}");
        assert_not_misreported(&error);
    }

    #[test]
    fn a_native_addon_is_reported_where_it_is_loaded_not_where_it_is_resolved() {
        // A `.node` addon resolves perfectly well. Failing it inside the resolver
        // printed `cannot resolve ...` plus `install it: npm install ...` for a
        // file sitting right there on disk.
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/native-addon");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            "{\"name\":\"native-addon\",\"main\":\"index.node\"}",
        )
        .unwrap();
        fs::write(package.join("index.node"), [0x7f, b'E', b'L', b'F']).unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import addon from 'native-addon';\nconsole.log(addon);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let error = match discover_direct(&entry) {
            Ok(_) => panic!("native code cannot go in a JavaScript bundle; it must fail the build"),
            Err(error) => error,
        };
        assert!(error.contains("index.node"), "{error}");
        assert!(error.contains("prebuilt native addon"), "{error}");
        assert_not_misreported(&error);
    }

    #[test]
    fn a_vue_file_still_loads_through_the_raw_loader() {
        // The query check runs BEFORE the extension table, so `?raw`/`?url` remain
        // the escape hatch for any extension diffpack cannot compile itself.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("App.vue"),
            "<template><h1>hi</h1></template>\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import source from './App.vue?raw';\nconsole.log(source);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(code.contains("<template><h1>hi</h1></template>"), "{code}");
    }

    #[test]
    fn mts_cts_and_extensionless_modules_still_build_as_javascript() {
        // The extension table is an ALLOW-list for JavaScript, so it must not
        // reject the JS-family extensions the resolver never adds implicitly
        // (`.mts`/`.cts`) or the extensionless files `node_modules` is full of.
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("a.mts"), "export const a = 1;\n").unwrap();
        fs::write(directory.path().join("b.cts"), "export const b = 2;\n").unwrap();
        fs::create_dir_all(directory.path().join("bin")).unwrap();
        fs::write(directory.path().join("bin/cli"), "export const c = 3;\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { a } from './a.mts';\nimport { b } from './b.cts';\n\
             import { c } from './bin/cli';\nconsole.log(a + b + c);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let output = directory.path().join("dist/bundle.js");
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
    }

    #[test]
    fn a_source_error_is_not_reported_as_a_dangling_reference() {
        // "Dangling references" describes what an UNRESOLVED IMPORT leaves behind.
        // Saying it for a module that never compiled points the reader at an
        // import that is perfectly fine.
        let source_only = [Diagnostic {
            kind: DiagnosticKind::Source { fatal: true },
            message: "App.vue: `.vue` is a Vue single-file component".into(),
        }];
        let error = partition_diagnostics(&source_only, "page `index`").unwrap_err();
        assert!(!error.contains("dangling"), "{error}");
        assert!(error.contains("would not match the source"), "{error}");

        let unresolved_only = [Diagnostic {
            kind: DiagnosticKind::UnresolvedImport {
                specifier: "left-pad".into(),
                importer: PathBuf::from("entry.js"),
            },
            message: "cannot resolve \"left-pad\"".into(),
        }];
        let error = partition_diagnostics(&unresolved_only, "page `index`").unwrap_err();
        assert!(error.contains("dangling references"), "{error}");
    }

    #[test]
    fn node_builtin_imports_are_left_external_and_run() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { basename } from 'node:path';\nimport { EOL } from 'node:os';\n\
             console.log(basename('/a/b/c.txt') + (EOL === '\\n' ? ':nl' : ':other'));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        // Externals are neither resolved nor diagnosed nor added to the graph.
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(
            reachable.len(),
            1,
            "only the entry is a graph module: {reachable:?}"
        );
        bundler.emit(&reachable, &output).unwrap();

        // The external require survives for the runtime to resolve. A static
        // import goes through `require.esm`, which calls that same `require` and
        // falls back to `__toESM` for a specifier the graph does not own.
        let bundle = fs::read_to_string(&output).unwrap();
        assert!(bundle.contains("require.esm(\"node:path\")"), "{bundle}");

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), "c.txt:nl\n");
        }
    }

    /// The browser build's node-builtin stub exists so that dead server code
    /// which leaked into the client graph still LOADS, and throws a
    /// specifically-named error only when it actually calls into the built-in.
    /// A named import is a read like any other: it must hand back the stub, not
    /// trip `__import`'s "does not provide an export" check — the stub is a
    /// Proxy whose shape is unknowable, so absence there proves nothing.
    #[test]
    fn a_named_import_of_a_node_builtin_in_a_browser_build_stubs_instead_of_throwing() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                import { readFileSync } from "node:fs";
                console.log("loaded:" + (typeof readFileSync));
                try {
                  readFileSync("/etc/hosts");
                } catch (error) {
                  console.log("called:" + error.message);
                }
            "#,
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.mjs");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "loaded:function\ncalled:node builtin node:fs is not available in the browser\n"
        );
    }

    /// A CommonJS module has exactly ONE ES namespace, whatever `module.exports`
    /// happens to be.
    ///
    /// `export * as ns from "cjs"` compiles to a getter, so the interop re-runs on
    /// every read of `ns`. `__cjsNamespaces` keys the wrapper by the `module.exports`
    /// object, which covers nothing when `module.exports = 42`: a WeakMap takes no
    /// primitive key, so every read minted a fresh namespace and `ns.legacy ===
    /// ns.legacy` was `false` where Node (and rolldown) say `true`.
    ///
    /// The identity that exists for every value shape is the MODULE, which is why a
    /// static import goes through `require.esm` (keyed by module id) rather than
    /// `__toESM(require(...))`. Caching by primitive VALUE instead would be a second
    /// wrong answer, and the second half of this test is what forbids it: two
    /// modules that each `module.exports = 42` are two namespaces.
    #[test]
    fn one_commonjs_module_has_one_namespace_even_when_its_exports_are_a_primitive() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(root.join("num.cjs"), "module.exports = 42;\n").unwrap();
        fs::write(root.join("other.cjs"), "module.exports = 42;\n").unwrap();
        fs::write(
            root.join("a.js"),
            "export * as legacy from \"./num.cjs\";\n",
        )
        .unwrap();
        fs::write(
            root.join("b.js"),
            "export * as legacy from \"./num.cjs\";\n",
        )
        .unwrap();
        fs::write(
            root.join("c.js"),
            "export * as legacy from \"./other.cjs\";\n",
        )
        .unwrap();
        fs::write(
            root.join("entry.js"),
            "import * as a from \"./a.js\";\n\
             import * as b from \"./b.js\";\n\
             import * as c from \"./c.js\";\n\
             console.log(\"stable:\" + (a.legacy === a.legacy));\n\
             console.log(\"shared:\" + (a.legacy === b.legacy));\n\
             console.log(\"distinct:\" + (a.legacy === c.legacy));\n\
             console.log(\"default:\" + a.legacy.default);\n",
        )
        .unwrap();

        let entry = root.join("entry.js");
        let output = root.join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        // Node's own answer for this program, unbundled.
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "stable:true\nshared:true\ndistinct:false\ndefault:42\n"
        );
    }

    /// A Node built-in reached from a BROWSER graph is a FATAL build diagnostic,
    /// not a silent external. Leaving it external emits a `require` no browser can
    /// satisfy: the build exits 0 and the page dies. The same specifier on a
    /// SERVER graph stays external and is not a diagnostic at all — Node resolves
    /// it. The classifier alone cannot tell these apart, which is why
    /// `resolve_dependencies` takes the `Target`.
    #[test]
    fn a_node_builtin_is_fatal_in_a_browser_build_and_external_in_a_server_build() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(
            &entry,
            "import { format } from \"url\";\nconsole.log(format({}));\n",
        )
        .unwrap();
        let config = |target| BuildConfig {
            target,
            ..BuildConfig::default()
        };

        let (_bundler, client) =
            discover_direct_with_config(&entry, &config(Target::Client)).unwrap();
        let fatal: Vec<_> = client
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_fatal())
            .collect();
        assert_eq!(fatal.len(), 1, "{:?}", client.diagnostics);
        assert!(
            matches!(
                &fatal[0].kind,
                DiagnosticKind::NodeBuiltinInBrowser { specifier, .. } if specifier == "url"
            ),
            "{:?}",
            fatal[0].kind
        );
        // The message names the built-in AND the file that imported it: the fix is
        // to stop pulling that file into the client graph.
        assert!(fatal[0].message.contains("\"url\""), "{}", fatal[0].message);
        assert!(
            fatal[0].message.contains("entry.js"),
            "{}",
            fatal[0].message
        );
        // And it must stop the build, not warn.
        assert!(partition_diagnostics(&client.diagnostics, "client build").is_err());

        let (_bundler, server) =
            discover_direct_with_config(&entry, &config(Target::Server)).unwrap();
        assert!(
            server
                .diagnostics
                .iter()
                .all(|diagnostic| !diagnostic.is_fatal()),
            "{:?}",
            server.diagnostics
        );
    }

    /// The browser `requireNative` fallback must not claim that an npm package is
    /// a "node builtin", and must not hand back a lazy stub for one. Every
    /// optional dependency in the ecosystem is loaded as
    /// `try { require(pkg) } catch {}`; returning a Proxy defeats the `catch`,
    /// smuggles the stub in as a real value, and throws later somewhere unrelated
    /// (this is exactly how `next-pages-framer-motion` died on
    /// `@emotion/is-prop-valid`). Node throws immediately for an absent module, so
    /// so do we — while a genuine Node built-in keeps the load-safe stub.
    #[test]
    fn a_non_builtin_runtime_require_throws_immediately_in_a_browser_build() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        // The specifier is assembled at runtime, exactly as framer-motion does it,
        // so the bundler never sees it as a static dependency and it reaches the
        // `requireNative` fallback.
        fs::write(
            directory.path().join("entry.js"),
            r#"
                const pkg = "@emotion/is-prop-" + "valid";
                let loaded = "fallback";
                try {
                  loaded = require(pkg).default;
                } catch (error) {
                  console.log("caught:" + error.message);
                }
                console.log("value:" + loaded);
            "#,
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.mjs");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        let stdout = String::from_utf8_lossy(&executed.stdout);
        // It threw (so the app's own `catch` ran and its fallback survived) ...
        assert!(stdout.contains("caught:"), "{stdout}");
        assert!(stdout.contains("value:fallback"), "{stdout}");
        // ... and it did NOT call an npm package a node builtin.
        assert!(
            !stdout.contains("node builtin"),
            "an npm package must not be reported as a Node built-in: {stdout}"
        );
        assert!(
            stdout.contains("@emotion/is-prop-valid"),
            "the error must name the specifier: {stdout}"
        );
    }

    /// `__dirname`/`__filename` in a BROWSER bundle. Node's ESM entry defines them
    /// from `import.meta.url`, but a browser chunk has no location to derive them
    /// from, so a bundled CommonJS package that reads one at module-init time died
    /// with `ReferenceError: __dirname is not defined` — and, because that runs
    /// during the entry's initialization, it took the WHOLE client bundle with it
    /// (this is exactly how `next-pages-shallow-routing` failed to hydrate: Next
    /// vendors an ncc-compiled `url` polyfill that does
    /// `__nccwpck_require__.ab = __dirname + "/"`). Webpack's `target: "web"`
    /// defines the same two names per module (its `node.__dirname` "mock" default),
    /// so this is what a browser build is supposed to do.
    #[test]
    fn a_browser_bundle_defines_dirname_for_a_bundled_commonjs_module() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        // The exact shape ncc-compiled packages emit at module scope.
        fs::write(
            directory.path().join("vendored.js"),
            r#"
                const base = __dirname + "/";
                module.exports = { base, file: __filename };
            "#,
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            r#"
                const vendored = require("./vendored.js");
                console.log("base:" + vendored.base);
                console.log("file:" + vendored.file);
            "#,
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.mjs");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let code = fs::read_to_string(&output).unwrap();
        assert!(
            code.contains("const __filename=\"/index.js\",__dirname=\"/\";"),
            "the browser factory must bind the two CommonJS location ambients: {code}"
        );
        // Only the module that reads them gets the binding — the entry does not.
        assert_eq!(
            code.matches("const __filename=\"/index.js\",__dirname=\"/\";")
                .count(),
            1,
            "the binding must be emitted per referencing module, not for every module"
        );

        // A `.mjs` file has no ambient `__dirname`, so running it proves the
        // binding is what makes the module load at all.
        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        let stdout = String::from_utf8_lossy(&executed.stdout);
        assert!(stdout.contains("base://"), "{stdout}");
        assert!(stdout.contains("file:/index.js"), "{stdout}");
    }

    #[test]
    fn a_configured_alias_resolves_to_its_target() {
        // The shape of TanStack's `#tanstack-router-entry` -> app router: a bare
        // `#`-specifier the plugin host aliases to a real file.
        let directory = tempdir().unwrap();
        let router = directory.path().join("router.tsx");
        fs::write(&router, "export const router = 1;\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { router } from '#tanstack-router-entry';\nconsole.log(router);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let config = BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: vec![(
                "#tanstack-router-entry".to_string(),
                router.to_string_lossy().into_owned(),
            )],
            ..BuildConfig::default()
        };
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);

        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 2, "{reachable:?}");
        assert!(
            reachable.iter().any(|id| id.contains("router.tsx")),
            "aliased import must resolve to the real router file: {reachable:?}"
        );
    }

    #[test]
    fn global_css_side_effect_imports_are_extracted_into_one_stylesheet() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("a.css"), ".a { color: red; }").unwrap();
        fs::write(directory.path().join("b.css"), ".b { color: blue; }").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './a.css';\nimport './b.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        // entry plus the two extracted stylesheets.
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 3, "{reachable:?}");
        bundler.emit(&reachable, &output).unwrap();

        // Both stylesheets land in one extracted file, in import order.
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        let a = css.find(".a { color: red; }").expect("a.css extracted");
        let b = css.find(".b { color: blue; }").expect("b.css extracted");
        assert!(a < b, "import order preserved: {css}");

        // The CSS is not left in the JavaScript bundle.
        let js = fs::read_to_string(&output).unwrap();
        assert!(!js.contains("color: red"), "{js}");

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&executed.stdout), "ok\n");
        }
    }

    /// The first `/assets/...` URL with the given stem referenced by `css`.
    /// The relative `assets/<stem>-<hash>.<ext>` reference inside an emitted
    /// stylesheet (CSS asset URLs are stylesheet-relative so any public base
    /// works).
    fn asset_url_in<'c>(css: &'c str, stem: &str) -> &'c str {
        let marker = format!("url(\"assets/{stem}-");
        let start = css
            .find(&marker)
            .unwrap_or_else(|| panic!("no assets/{stem}- reference in: {css}"));
        let url = &css[start + "url(\"".len()..];
        url.split('"').next().expect("the url is terminated")
    }

    #[test]
    fn css_module_import_exports_scoped_mapping_with_vite_default_and_named_exports() {
        // Vite's default CSS Modules behavior (no `css.modules` config): the
        // default export is the locals -> scoped-names object, AND every
        // identifier-safe local is also a named export. Non-identifier locals
        // (`btn-primary`) appear only in the default object.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("button.module.css"),
            ".btn { color: red; }\n\
             .btn:hover { color: blue; }\n\
             .btn-primary > .icon, .btn-primary::before { color: green; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles, { btn, icon } from './button.module.css';\n\
             console.log(styles.btn === btn, styles.icon === icon);\n\
             console.log(JSON.stringify(styles));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        // The stylesheet carries the scoped selectors and no unscoped local.
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("._btn_"), "{css}");
        assert!(css.contains(":hover"), "{css}");
        assert!(css.contains("._btn-primary_"), "{css}");
        assert!(!css.contains(".btn "), "unscoped local leaked: {css}");
        assert!(!css.contains(".btn:"), "unscoped local leaked: {css}");
        assert!(!css.contains(".icon"), "unscoped local leaked: {css}");

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            let stdout = String::from_utf8_lossy(&executed.stdout);
            let mut lines = stdout.lines();
            assert_eq!(
                lines.next(),
                Some("true true"),
                "named exports must alias the default mapping: {stdout}"
            );
            let mapping: serde_json::Value =
                serde_json::from_str(lines.next().expect("mapping line")).unwrap();
            let btn = mapping["btn"].as_str().expect("btn mapping");
            assert!(
                btn.starts_with("_btn_") && btn.len() == "_btn_".len() + 8,
                "scoped name format `_btn_<hash8>`: {btn}"
            );
            let primary = mapping["btn-primary"]
                .as_str()
                .expect("btn-primary mapping");
            assert!(primary.starts_with("_btn-primary_"), "{primary}");
            // The scoped selector in the emitted CSS is exactly the exported name.
            assert!(css.contains(&format!(".{btn}")), "{css}");
        }
    }

    #[test]
    fn css_module_global_escape_hatch_and_same_file_composes() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("card.module.css"),
            ":global(.theme-dark) .card { color: white; }\n\
             .base { padding: 4px; }\n\
             .fancy { composes: base; color: blue; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './card.module.css';\n\
             console.log(styles.fancy);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();

        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        // The :global(...) contents are unscoped and the wrapper is gone.
        assert!(css.contains(".theme-dark ._card_"), "{css}");
        assert!(!css.contains(":global"), "{css}");
        // composes never reaches the emitted CSS.
        assert!(!css.contains("composes"), "{css}");

        if node_command().arg("--version").output().is_ok() {
            let executed = node_command().arg(&output).output().unwrap();
            assert!(
                executed.status.success(),
                "{}",
                String::from_utf8_lossy(&executed.stderr)
            );
            let stdout = String::from_utf8_lossy(&executed.stdout);
            let names = stdout.trim().split(' ').collect::<Vec<_>>();
            assert_eq!(names.len(), 2, "self + composed: {stdout}");
            assert!(names[0].starts_with("_fancy_"), "{stdout}");
            assert!(names[1].starts_with("_base_"), "{stdout}");
            // Both classes exist in the emitted stylesheet.
            assert!(css.contains(&format!(".{}", names[0])), "{css}");
            assert!(css.contains(&format!(".{}", names[1])), "{css}");
        }
    }

    #[test]
    fn cross_file_composes_adds_a_dependency_edge_and_tracks_edits_incrementally() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let other = directory.path().join("other.module.css");
        fs::write(&other, ".bar { color: green; }").unwrap();
        fs::write(
            directory.path().join("main.module.css"),
            ".foo { composes: bar from './other.module.css'; color: red; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './main.module.css';\nconsole.log(styles.foo);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (mut bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        // The composes source is a real graph module (the dependency edge).
        let reachable = bundler.reachable_modules_direct();
        assert!(
            reachable.iter().any(|id| id.ends_with("other.module.css")),
            "composes must create a dependency edge: {reachable:?}"
        );
        bundler.emit(&reachable, &output).unwrap();
        let first =
            String::from_utf8(node_command().arg(&output).output().unwrap().stdout).unwrap();
        let first_names = first
            .trim()
            .split(' ')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(first_names.len(), 2, "{first}");
        assert!(first_names[0].starts_with("_foo_"), "{first}");
        assert!(first_names[1].starts_with("_bar_"), "{first}");

        // Editing the COMPOSED file re-derives through the incremental path:
        // its scoped name (content-hashed) changes, and the composer — whose
        // mapping resolves the foreign name at runtime through the module graph
        // — picks the new name up without itself being re-derived.
        fs::write(&other, ".bar { color: purple; }").unwrap();
        let update = bundler.rebuild_path(&other).unwrap();
        assert!(
            update
                .delta
                .changed
                .iter()
                .any(|id| id.ends_with("other.module.css")),
            "{update:?}"
        );
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let second =
            String::from_utf8(node_command().arg(&output).output().unwrap().stdout).unwrap();
        let second_names = second
            .trim()
            .split(' ')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(
            first_names[0], second_names[0],
            "the composer's own scoped name is unchanged"
        );
        assert_ne!(
            first_names[1], second_names[1],
            "the composed file's scoped name must move with its content"
        );
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains(&format!(".{}", second_names[1])), "{css}");
        assert!(css.contains("color: purple"), "{css}");
    }

    #[test]
    fn scss_global_stylesheet_compiles_through_the_css_pipeline() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("app.scss"),
            "$pad: 12px;\n#bar {\n  padding: $pad;\n  &:hover { color: red; }\n  \
             @media (min-width: 2 * 400px) { flex: 1; }\n}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './app.scss';\nconsole.log('ok');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("#bar {\n  padding: 12px;\n}"), "{css}");
        assert!(css.contains("#bar:hover {\n  color: red;\n}"), "{css}");
        assert!(
            css.contains("@media (min-width: 800px) {"),
            "nested media must bubble with the evaluated prelude: {css}"
        );
    }

    #[test]
    fn scss_module_scopes_compiled_css_and_exports_the_mapping() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("_theme.scss"),
            "$clr: #e6a459;\n@mixin pulse { animation: pulse 1s infinite; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("card.module.scss"),
            "@use './theme';\n.card { color: theme.$clr; @include theme.pulse; }\n\
             @keyframes pulse { 0% { opacity: 0.5; } }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './card.module.scss';\nconsole.log(styles.card);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let printed =
            String::from_utf8(node_command().arg(&output).output().unwrap().stdout).unwrap();
        let scoped = printed.trim();
        assert!(scoped.starts_with("_card_"), "{printed}");
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains(&format!(".{scoped}")), "{css}");
        assert!(css.contains("color: #e6a459"), "{css}");
        // The keyframes name AND the mixin-injected animation reference are
        // scoped consistently by the CSS Modules pass.
        assert!(css.contains("@keyframes _pulse_"), "{css}");
        assert!(css.contains("animation: _pulse_"), "{css}");
    }

    #[test]
    fn editing_a_used_scss_partial_rederives_the_importing_module() {
        let directory = tempdir().unwrap();
        let partial = directory.path().join("_theme.scss");
        fs::write(&partial, "$clr: red;\n").unwrap();
        fs::write(
            directory.path().join("app.scss"),
            "@use './theme';\n.x { color: theme.$clr; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './app.scss';\nconsole.log('ok');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (mut bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("color: red"), "{css}");
        // The partial is not a graph module itself, but it IS a recorded css
        // source: editing it must re-derive the importing .scss module.
        assert!(bundler.is_known_module(&partial), "partial must be known");
        fs::write(&partial, "$clr: blue;\n").unwrap();
        let update = bundler.rebuild_path(&partial).unwrap();
        assert!(
            update
                .delta
                .changed
                .iter()
                .any(|id| id.ends_with("app.scss")),
            "{update:?}"
        );
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("color: blue"), "{css}");
        assert!(!css.contains("color: red"), "{css}");
    }

    #[test]
    fn scss_unsupported_construct_is_a_hard_build_error() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("app.scss"), ".a { @extend .b; }\n").unwrap();
        fs::write(directory.path().join("entry.js"), "import './app.scss';\n").unwrap();
        let entry = directory.path().join("entry.js");
        let error = match discover_direct(&entry) {
            Err(error) => error,
            Ok(_) => panic!("@extend must fail the build"),
        };
        assert!(
            error.contains("@extend") && error.contains("app.scss"),
            "the error must name the construct and the file: {error}"
        );
    }

    #[test]
    fn a_missing_cross_file_composes_target_throws_at_runtime_instead_of_undefined() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("other.module.css"),
            ".present { color: green; }",
        )
        .unwrap();
        fs::write(
            directory.path().join("main.module.css"),
            ".foo { composes: missing from './other.module.css'; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './main.module.css';\nconsole.log(styles.foo);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, _) = discover_direct(&entry).unwrap();
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            !executed.status.success(),
            "a missing composes target must not silently yield undefined"
        );
        let stderr = String::from_utf8_lossy(&executed.stderr);
        assert!(
            stderr.contains("composes target \"missing\" is not exported by"),
            "{stderr}"
        );
    }

    #[test]
    fn css_import_statements_become_edges_with_dedup_ordering_and_media_wrap() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("base.css"), ".base { color: red; }\n").unwrap();
        fs::write(
            directory.path().join("cond.css"),
            ".cond { color: blue; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("a.css"),
            "@import './base.css';\n@import './cond.css' screen;\n.a { color: black; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("b.css"),
            "@import './base.css';\n.b { color: white; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './a.css';\nimport './b.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (mut bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        // entry, a.css, b.css, base.css (deduped once), cond.css?media=screen.
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 5, "{reachable:?}");
        assert!(
            reachable
                .iter()
                .any(|id| id.ends_with("cond.css?media=screen")),
            "{reachable:?}"
        );
        bundler.emit(&reachable, &output).unwrap();

        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(!css.contains("@import"), "no unresolved @import: {css}");
        assert_eq!(
            css.matches(".base ").count(),
            1,
            "the shared import is inlined exactly once: {css}"
        );
        // Imported-before-importer ordering.
        let base = css.find(".base").unwrap();
        let a = css.find(".a ").unwrap();
        let b = css.find(".b ").unwrap();
        assert!(base < a && base < b, "{css}");
        // The media-qualified import is wrapped.
        let media = css.find("@media screen").unwrap();
        let cond = css.find(".cond").unwrap();
        let close = css[media..].find('}').unwrap() + media;
        assert!(media < cond && cond < close, "{css}");

        // Editing the media-imported file re-derives its `?media` module even
        // though the bare path is not itself a module.
        fs::write(
            directory.path().join("cond.css"),
            ".cond { color: teal; }\n",
        )
        .unwrap();
        let cond_path = directory.path().join("cond.css");
        assert!(bundler.is_known_module(&cond_path));
        let update = bundler.rebuild_path(&cond_path).unwrap();
        assert_eq!(update.transformed_modules, 1, "{update:?}");
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("color: teal"), "{css}");
    }

    #[test]
    fn css_url_references_are_rewritten_to_hashed_assets_relative_to_each_file() {
        let directory = tempdir().unwrap();
        let sub = directory.path().join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("img.png"), b"png-bytes").unwrap();
        fs::write(
            sub.join("inner.css"),
            ".inner { background: url(./img.png); }\n",
        )
        .unwrap();
        fs::write(directory.path().join("photo.jpg"), b"jpg-bytes").unwrap();
        fs::write(
            directory.path().join("top.css"),
            "@import './sub/inner.css';\n\
             .top { background: url('./photo.jpg'); }\n\
             .keep { fill: url(#gradient); background: url(data:image/gif;base64,R0); }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './top.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();

        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        // The nested @import's url resolved relative to THAT file (sub/img.png),
        // and both references were rewritten to hashed public URLs.
        let img_url = asset_url_in(&css, "img");
        let photo_url = asset_url_in(&css, "photo");
        assert!(img_url.ends_with(".png"), "{img_url}");
        assert!(photo_url.ends_with(".jpg"), "{photo_url}");
        assert!(!css.contains("./img.png"), "{css}");
        assert!(!css.contains("./photo.jpg"), "{css}");
        // Skipped forms survive verbatim.
        assert!(css.contains("url(#gradient)"), "{css}");
        assert!(css.contains("url(data:image/gif;base64,R0)"), "{css}");
        // The assets landed on disk with the referenced bytes.
        let assets = directory.path().join("dist/assets");
        assert_eq!(
            fs::read(assets.join(img_url.trim_start_matches("assets/"))).unwrap(),
            b"png-bytes"
        );
        assert_eq!(
            fs::read(assets.join(photo_url.trim_start_matches("assets/"))).unwrap(),
            b"jpg-bytes"
        );
    }

    #[test]
    fn a_nested_media_import_inlines_relative_urls_and_reacts_to_nested_edits() {
        let directory = tempdir().unwrap();
        let sub = directory.path().join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("icon.png"), b"icon-bytes").unwrap();
        fs::write(
            sub.join("deep.css"),
            ".deep { background: url(./icon.png); }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("wrapped.css"),
            "@import './sub/deep.css';\n.wrapped { color: red; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("app.css"),
            "@import './wrapped.css' print;\n.app { color: blue; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './app.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (mut bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();

        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        // The whole wrapped file (including its nested import) is inside the
        // media block, and the nested file's url resolved relative to ITSELF.
        let media = css.find("@media print").unwrap();
        assert!(media < css.find(".deep").unwrap(), "{css}");
        assert!(media < css.find(".wrapped").unwrap(), "{css}");
        let icon_url = asset_url_in(&css, "icon");
        assert_eq!(
            fs::read(
                directory
                    .path()
                    .join("dist/assets")
                    .join(icon_url.trim_start_matches("assets/"))
            )
            .unwrap(),
            b"icon-bytes"
        );

        // An edit to the transitively INLINED nested file re-derives the media
        // module (tracked via css_source_files), even though neither deep.css
        // nor wrapped.css is a bare module.
        fs::write(sub.join("deep.css"), ".deep { color: orange; }\n").unwrap();
        let deep = sub.join("deep.css");
        assert!(bundler.is_known_module(&deep));
        let update = bundler.rebuild_path(&deep).unwrap();
        assert_eq!(update.transformed_modules, 1, "{update:?}");
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(css.contains("color: orange"), "{css}");
        let media = css.find("@media print").unwrap();
        assert!(
            media < css.find(".deep").unwrap(),
            "the edit stays wrapped: {css}"
        );
    }

    #[test]
    fn remote_css_imports_are_hoisted_to_the_top_of_the_emitted_stylesheet() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("fonts.css"),
            "@import url(https://example.com/font.css);\n.fonts { font-family: X; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './fonts.css';\nconsole.log('ok');\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let css = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(
            css.starts_with("@import url(https://example.com/font.css);"),
            "a remote @import is only valid before all rules, so it must be hoisted: {css}"
        );
        assert!(css.contains(".fonts"), "{css}");
    }

    #[test]
    fn unsupported_css_constructs_fail_the_build_with_specific_errors() {
        // A CSS module with an at-rule the scoper cannot handle confidently.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("odd.module.css"),
            "@tailwind base;\n.foo { color: red; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import styles from './odd.module.css';\nconsole.log(styles);\n",
        )
        .unwrap();
        let error = discover_direct(&directory.path().join("entry.js"))
            .map(|_| ())
            .unwrap_err();
        assert!(error.contains("unsupported at-rule `@tailwind`"), "{error}");
        assert!(error.contains("odd.module.css"), "{error}");

        // An @import form we do not support.
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("x.css"), ".x{}").unwrap();
        fs::write(
            directory.path().join("layered.css"),
            "@import './x.css' layer(base);\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './layered.css';\n",
        )
        .unwrap();
        let error = discover_direct(&directory.path().join("entry.js"))
            .map(|_| ())
            .unwrap_err();
        assert!(
            error.contains("layer(...) condition is not supported"),
            "{error}"
        );
        assert!(error.contains("layered.css"), "{error}");

        // A url() that resolves to nothing names the CSS file and the reference.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("broken.css"),
            ".a { background: url(./missing.png); }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './broken.css';\n",
        )
        .unwrap();
        let error = discover_direct(&directory.path().join("entry.js"))
            .map(|_| ())
            .unwrap_err();
        assert!(error.contains("url(./missing.png)"), "{error}");
        assert!(error.contains("broken.css"), "{error}");
    }

    /// A legacy Tailwind v3 entry (`@tailwind base/components/utilities`) compiles
    /// natively through the v4 pipeline (the directives expand to the same layers), so
    /// a real v3 app is styled instead of hard-erroring as it used to.
    #[test]
    fn a_tailwind_v3_entry_compiles_natively() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("v3.css"),
            "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './v3.css';\nexport const html = '<div class=\"underline\">x</div>';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        // The `@tailwind` directives are consumed (not shipped raw) and the scanned
        // utility is generated from the vendored v4 base theme.
        assert!(
            !stylesheet.contains("@tailwind"),
            "directives must not survive: {stylesheet}"
        );
        assert!(
            stylesheet.contains("underline") && stylesheet.contains("text-decoration"),
            "the scanned utility is generated for a v3 entry: {}",
            &stylesheet[..stylesheet.len().min(400)]
        );
    }

    /// A Tailwind v4 entry imported as a plain global stylesheet compiles
    /// through the native engine at emit time (previously a hard error that
    /// demanded `?url` — real apps, e.g. markpad, import it directly).
    #[test]
    fn a_globally_imported_tailwind_entry_is_compiled_at_emit() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("tw.css"), "@import 'tailwindcss';\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './tw.css';\nexport const html = '<div class=\"underline\">x</div>';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(
            !stylesheet.contains("@import 'tailwindcss'"),
            "the compiler invocation must not survive: {stylesheet}"
        );
        assert!(
            stylesheet.contains("underline") && stylesheet.contains("text-decoration"),
            "the scanned utility is generated: {}",
            &stylesheet[..stylesheet.len().min(400)]
        );
    }

    /// A Tailwind entry is the INPUT to a compiler, not a stylesheet that happens
    /// to be concatenated with its imports: `@theme`, `@utility` and plain CSS
    /// written in an `@import`ed file configure the SAME compile. Splitting the
    /// graph into separate stylesheet modules silently dropped every directive an
    /// imported file carried — an app whose design tokens live in an imported
    /// file lost its entire theme.
    #[test]
    fn an_imported_stylesheets_tailwind_directives_configure_the_entrys_compile() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("tokens.css"),
            "@theme {\n  --color-brand: #123456;\n}\n\
             @utility card-pad {\n  padding: 7px;\n}\n\
             @utility card-rule {\n  width: 3px;\n}\n\
             @property --ring-shade {\n  syntax: \"*\";\n  inherits: false;\n}\n\
             .from-tokens {\n  color: rebeccapurple;\n}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("tw.css"),
            "@import 'tailwindcss';\n@import './tokens.css';\n\
             .card {\n  @apply text-brand card-rule;\n}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './tw.css';\n\
             export const html = '<div class=\"card-pad from-tokens\">x</div>';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(
            !stylesheet.contains("@import"),
            "no @import may survive the compile: {stylesheet}"
        );
        // A `@theme` token from the imported file resolves an `@apply` in the entry.
        assert!(
            stylesheet.contains("#123456"),
            "the imported @theme token reaches the compile: {stylesheet}"
        );
        // An `@utility` from the imported file generates for a scanned candidate.
        assert!(
            stylesheet.contains("padding: 7px") || stylesheet.contains("padding:7px"),
            "the imported @utility generates: {stylesheet}"
        );
        // Plain rules in the imported file are emitted too.
        assert!(
            stylesheet.contains("rebeccapurple"),
            "the imported file's own rules are emitted: {stylesheet}"
        );
        // A standard block at-rule the compiler has no opinion about passes through.
        assert!(
            stylesheet.contains("@property --ring-shade"),
            "@property survives verbatim: {stylesheet}"
        );
        // And an `@apply` in the ENTRY resolves an `@utility` the IMPORTED file
        // defines — the two files are one compile, in both directions.
        assert!(
            stylesheet.contains("width: 3px") || stylesheet.contains("width:3px"),
            "the entry's @apply of an imported @utility expands: {stylesheet}"
        );
    }

    /// cal.com's exact shape: the entry itself is plain, and the `@plugin` lives in
    /// a stylesheet it `@import`s from another workspace package. The delegation
    /// gate reads the SPLICED entry, so the plugin is seen and the whole sheet is
    /// compiled by the app's own Tailwind — including a `@apply` of a utility only
    /// that plugin registers, which no native engine could answer.
    #[test]
    fn a_plugin_reached_through_an_import_delegates_the_whole_entry() {
        if Command::new("node").arg("--version").output().is_err() {
            return;
        }
        // A real Tailwind v4 install to delegate to; the corpus apps carry one.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let modules = fs::read_dir(repo.join("integration/e2e/apps"))
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|app| app.path().join("node_modules"))
            .find(|modules| modules.join("@tailwindcss/node/package.json").is_file());
        let Some(modules) = modules else {
            eprintln!("skipped: no corpus app has @tailwindcss/node installed");
            return;
        };
        let directory = tempdir().unwrap();
        let root = directory.path();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&modules, root.join("node_modules")).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(&modules, root.join("node_modules")).unwrap();
        fs::write(root.join("package.json"), "{\"name\":\"probe\"}\n").unwrap();
        fs::write(
            root.join("plugin.js"),
            "module.exports = function ({ addUtilities }) {\n\
               addUtilities({ '.probe-rule': { 'caret-color': 'rebeccapurple' } });\n\
             };\n",
        )
        .unwrap();
        // The plugin is declared HERE, one @import away from the entry.
        fs::write(
            root.join("tokens.css"),
            "@plugin './plugin.js';\n.from-tokens {\n  color: teal;\n}\n",
        )
        .unwrap();
        fs::write(
            root.join("tw.css"),
            "@import 'tailwindcss';\n@import './tokens.css';\n\
             .scroll-bar {\n  @apply probe-rule;\n}\n",
        )
        .unwrap();
        fs::write(
            root.join("entry.js"),
            "import './tw.css';\nexport const html = '<div class=\"flex\">x</div>';\n",
        )
        .unwrap();

        let entry = root.join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(root.join("dist/bundle.css")).unwrap();
        assert!(
            stylesheet.contains("rebeccapurple"),
            "the imported file's @plugin registered the utility the entry applies: {}",
            &stylesheet[..stylesheet.len().min(600)]
        );
        assert!(
            stylesheet.contains("color: teal") || stylesheet.contains("color:teal"),
            "the imported file's own rules survive the delegated compile"
        );
        assert!(
            stylesheet.contains("display: flex") || stylesheet.contains("display:flex"),
            "diffpack's class scan still drives the delegated compile"
        );
    }

    /// `@import "some-package"` in a Tailwind entry resolves through
    /// `node_modules` with the CSS `style` condition — the resolution Tailwind
    /// itself performs, and the only way to reach a stylesheet a package
    /// publishes as `exports: { ".": { "style": "./dist/x.css" } }`.
    #[test]
    fn a_bare_css_import_resolves_through_a_packages_style_export() {
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/some-tokens");
        fs::create_dir_all(package.join("dist")).unwrap();
        fs::write(
            package.join("package.json"),
            "{\"name\":\"some-tokens\",\"exports\":{\".\":{\"style\":\"./dist/tokens.css\"}}}",
        )
        .unwrap();
        fs::write(
            package.join("dist/tokens.css"),
            "@utility packaged-gap {\n  gap: 11px;\n}\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("tw.css"),
            "@import 'tailwindcss';\n@import \"some-tokens\";\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './tw.css';\nexport const html = '<div class=\"packaged-gap\">x</div>';\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        assert!(
            stylesheet.contains("gap: 11px") || stylesheet.contains("gap:11px"),
            "the package's published stylesheet reaches the compile: {stylesheet}"
        );
    }

    /// `@source` widens the candidate scan past the project root — how a monorepo
    /// app declares that its classes also live in sibling workspace packages —
    /// and `@source not` narrows it again. The path is anchored to the file that
    /// WROTE the directive, not to the entry that imported it.
    #[test]
    fn at_source_widens_and_at_source_not_narrows_the_candidate_scan() {
        let directory = tempdir().unwrap();
        let app = directory.path().join("app");
        fs::create_dir_all(app.join("styles")).unwrap();
        fs::create_dir_all(directory.path().join("shared/src")).unwrap();
        fs::create_dir_all(directory.path().join("shared/generated")).unwrap();
        fs::write(app.join("package.json"), "{\"name\":\"app\"}").unwrap();
        // Declared in an IMPORTED file one directory deeper, so a directive
        // anchored to the entry instead of its own file would resolve elsewhere.
        fs::write(
            app.join("styles/sources.css"),
            "@source \"../../shared/**/*.tsx\";\n\
             @source not \"../../shared/generated/**/*.tsx\";\n",
        )
        .unwrap();
        fs::write(
            app.join("tw.css"),
            "@import 'tailwindcss';\n@import './styles/sources.css';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("shared/src/widget.tsx"),
            "export const Widget = () => <div className=\"tracking-widest\" />;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("shared/generated/stale.tsx"),
            "export const Stale = () => <div className=\"tracking-tighter\" />;\n",
        )
        .unwrap();
        fs::write(
            app.join("entry.js"),
            "import './tw.css';\nexport const html = '<div />';\n",
        )
        .unwrap();
        let entry = app.join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = app.join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(app.join("dist/bundle.css")).unwrap();
        assert!(
            stylesheet.contains("tracking-widest"),
            "the @source-declared directory is scanned: {stylesheet}"
        );
        assert!(
            !stylesheet.contains("tracking-tighter"),
            "the `@source not` directory is excluded: {stylesheet}"
        );
        assert!(
            !stylesheet.contains("@source"),
            "the directive itself is consumed: {stylesheet}"
        );
    }

    #[test]
    fn rebuilds_only_the_changed_module_and_updates_live_reachability() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let value = directory.path().join("value.ts");
        let output = directory.path().join("bundle.js");
        fs::write(
            &entry,
            "import { value } from './value.js'; console.log(value);",
        )
        .unwrap();
        fs::write(&value, "export const value: number = 1;").unwrap();

        let (mut bundler, _) = discover(&entry).unwrap();
        let mut session = bundler.direct_reachability();
        let mut reachable = session.reachable_modules();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "1\n");

        fs::write(&value, "export const value: number = 2;").unwrap();
        let update = bundler.rebuild_path(&value).unwrap();
        assert_eq!(update.transformed_modules, 1);
        assert_eq!(update.delta.changed.len(), 1);
        let result = session.apply(&update.delta);
        for removed in result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(result.added);
        bundler.emit(&reachable, &output).unwrap();

        assert_eq!(run_node(&output), "2\n");
        assert_eq!(reachable.len(), 2);

        fs::write(&entry, "console.log('detached');").unwrap();
        let update = bundler.rebuild_path(&entry).unwrap();
        let result = session.apply(&update.delta);
        for removed in result.removed {
            reachable.remove(&removed);
        }
        reachable.extend(result.added);
        assert_eq!(reachable, bundler.reachable_modules_direct());
        assert_eq!(reachable.len(), 1);
    }

    #[test]
    fn resolves_typescript_path_aliases_from_the_nearest_tsconfig() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let source = directory.path().join("src");
        fs::create_dir_all(&source).unwrap();
        fs::write(
            directory.path().join("tsconfig.json"),
            r#"{"compilerOptions":{"paths":{"~/*":["./src/*"]}}}"#,
        )
        .unwrap();
        let entry = source.join("entry.ts");
        let output = directory.path().join("bundle.js");
        fs::write(
            &entry,
            "import { value } from '~/value'; console.log(value);",
        )
        .unwrap();
        fs::write(source.join("value.ts"), "export const value = 42;").unwrap();

        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 2);
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "42\n");
    }

    /// `experimentalDecorators` in the tsconfig that owns a file makes its
    /// `@decorator`s LOWER, and the `__decorate` helper the lowering calls comes
    /// from inside the binary — no `@oxc-project/runtime` install.
    ///
    /// A decorator is syntax no engine parses, so an unlowered one does not fail
    /// the build: it fails at LOAD, as `SyntaxError: Invalid or unexpected token`
    /// pointing into a minified line. Node executing the bundle (and observing the
    /// decorator's effect) is therefore the real assertion.
    #[test]
    fn legacy_decorators_lower_against_the_owning_tsconfig_and_bundle_their_helper() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("tsconfig.json"),
            r#"{"compilerOptions":{"experimentalDecorators":true}}"#,
        )
        .unwrap();
        // A method decorator that REPLACES the implementation, so the emitted
        // program is only correct if the decorator actually ran.
        fs::write(
            directory.path().join("entry.ts"),
            "function shout(_target: any, _key: string, descriptor: any) {\n\
             \x20 const inner = descriptor.value;\n\
             \x20 descriptor.value = function (...args: any[]) { return inner.apply(this, args).toUpperCase(); };\n\
             \x20 return descriptor;\n\
             }\n\
             class Greeter {\n\
             \x20 @shout\n\
             \x20 greet(name: string) { return 'hello ' + name; }\n\
             }\n\
             console.log(new Greeter().greet('world'));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.ts");
        let output = directory.path().join("bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(
            reachable.len(),
            2,
            "the entry plus the embedded __decorate helper: {reachable:?}"
        );
        bundler.emit(&reachable, &output).unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(
            !code.contains("@oxc-project/runtime"),
            "the helper must be served from the binary, not asked of the app: {code}"
        );
        node_check(&output);
        assert_eq!(run_node(&output), "HELLO WORLD\n");
    }

    /// Without `experimentalDecorators`, a decorator is a Stage 3 decorator, which
    /// this build cannot lower. Emitting it verbatim would produce a file no engine
    /// parses, so it is a FATAL diagnostic naming the file and the decorator —
    /// never a bundle that fails at load with a SyntaxError in minified output.
    #[test]
    fn a_stage_three_decorator_is_a_fatal_diagnostic_naming_the_file_and_decorator() {
        let directory = tempdir().unwrap();
        // A tsconfig that owns the file but says nothing about decorators: the
        // TypeScript default, which is Stage 3 semantics.
        fs::write(
            directory.path().join("tsconfig.json"),
            r#"{"compilerOptions":{}}"#,
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.ts"),
            "function logged(value: any, _context: any) { return value; }\n\
             class Greeter {\n\
             \x20 @logged\n\
             \x20 greet() { return 'hi'; }\n\
             }\n\
             console.log(new Greeter().greet());\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.ts");
        let (_, update) = discover_direct(&entry).unwrap();
        let fatal: Vec<&str> = update
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_fatal())
            .map(|diagnostic| diagnostic.message.as_str())
            .collect();
        assert_eq!(fatal.len(), 1, "{:?}", update.diagnostics);
        assert!(fatal[0].contains("entry.ts"), "{}", fatal[0]);
        assert!(fatal[0].contains("@logged"), "{}", fatal[0]);
        assert!(
            fatal[0].contains("experimentalDecorators"),
            "the message must name the setting that would lower it: {}",
            fatal[0]
        );
    }

    /// Writes a `node_modules` JSX-runtime package whose `jsx` factory records
    /// which runtime produced an element, so a bundle can be asked, per module,
    /// what import source its JSX was lowered against.
    fn write_jsx_runtime_package(root: &Path, name: &str) {
        let package = root.join("node_modules").join(name);
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            format!(
                r#"{{"name":"{name}","version":"1.0.0","exports":{{"./jsx-runtime":"./jsx-runtime.js"}}}}"#
            ),
        )
        .unwrap();
        fs::write(
            package.join("jsx-runtime.js"),
            format!(
                "export const Fragment = 'Fragment';\n\
                 export function jsx(tag, props) {{ return '{name}:' + tag; }}\n\
                 export const jsxs = jsx;\n"
            ),
        )
        .unwrap();
    }

    /// `compilerOptions.jsxImportSource` decides which package the automatic
    /// runtime is imported from, and it is read from the tsconfig that OWNS each
    /// file — through create-vite's solution-style root config (`{"files":[],
    /// "references":[...]}`, no `compilerOptions` at all), which a nearest-file
    /// read finds nothing in. Two files that the app's tsconfig does NOT own stay
    /// on react: a dependency's `.tsx` under `node_modules`, and diffpack's own
    /// generated `.diffpack-next/` sources, which live inside the project root and
    /// would otherwise be claimed by the app's `include`.
    #[test]
    fn jsx_import_source_comes_from_the_tsconfig_that_owns_each_file() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join(".diffpack-next")).unwrap();
        write_jsx_runtime_package(root, "myjsx");
        write_jsx_runtime_package(root, "react");
        // Solution-style: the root config carries no `compilerOptions` at all.
        fs::write(
            root.join("tsconfig.json"),
            r#"{"files":[],"references":[{"path":"./tsconfig.app.json"}]}"#,
        )
        .unwrap();
        // `**/*.tsx` is create-next-app's own `include`, and it reaches straight
        // into `.diffpack-next/` — which is why the guard there is not theoretical.
        fs::write(
            root.join("tsconfig.app.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"myjsx"},
                "include":["**/*.ts","**/*.tsx"]}"#,
        )
        .unwrap();
        fs::write(
            root.join("node_modules").join("vendor.tsx"),
            "export const vendor = <span />;\n",
        )
        .unwrap();
        fs::write(
            root.join(".diffpack-next").join("generated.tsx"),
            "export const generated = <main />;\n",
        )
        .unwrap();
        let entry = root.join("src").join("entry.tsx");
        fs::write(
            &entry,
            "import { vendor } from '../node_modules/vendor.tsx';\n\
             import { generated } from '../.diffpack-next/generated.tsx';\n\
             console.log(<div />, vendor, generated);\n",
        )
        .unwrap();

        let output = root.join("bundle.js");
        let (bundler, update) = discover_next_with_config(&entry, &BuildConfig::default()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "myjsx:div react:span react:main\n");
    }

    /// Per FILE, not per build: two sibling subtrees of ONE bundle, each with its
    /// own nearest config naming a different import source, must each be lowered
    /// against its own. A build-wide answer (first config found, or the entry's)
    /// silently hands one subtree the other's runtime, and nothing in the output
    /// says so — the JSX still compiles, it just calls into the wrong package.
    #[test]
    fn two_subtrees_with_different_nearest_configs_each_get_their_own_import_source() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_jsx_runtime_package(root, "myjsx");
        write_jsx_runtime_package(root, "react");
        fs::create_dir_all(root.join("packages/preactish")).unwrap();
        fs::create_dir_all(root.join("packages/reactish")).unwrap();
        // A JS project states its options in `jsconfig.json`, a TS one in
        // `tsconfig.json`; both shapes appear in one tree here on purpose.
        fs::write(
            root.join("packages/preactish/jsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"myjsx"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("packages/reactish/tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"react"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("packages/preactish/widget.jsx"),
            "export const widget = <span />;\n",
        )
        .unwrap();
        fs::write(
            root.join("packages/reactish/panel.tsx"),
            "export const panel = <section />;\n",
        )
        .unwrap();
        // The entry itself is under NEITHER config: it keeps oxc's react default.
        let entry = root.join("entry.jsx");
        fs::write(
            &entry,
            "import { widget } from './packages/preactish/widget.jsx';\n\
             import { panel } from './packages/reactish/panel.tsx';\n\
             console.log(<div />, widget, panel);\n",
        )
        .unwrap();

        let output = root.join("bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "react:div myjsx:span react:section\n");
    }

    /// A `.jsx` file under a tsconfig that TypeScript would NOT compile (no
    /// `allowJs`, so `include: ["src"]` does not claim it) still gets the project's
    /// import source. The bundler lowers the file whatever `tsc` would have done
    /// with it, and `preact/jsx-runtime` is the only runtime such a project has.
    #[test]
    fn a_jsx_file_gets_the_import_source_of_a_tsconfig_that_would_not_compile_it() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        write_jsx_runtime_package(root, "myjsx");
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"myjsx"},"include":["src"]}"#,
        )
        .unwrap();
        let entry = root.join("src").join("main.jsx");
        fs::write(&entry, "console.log(<div />);\n").unwrap();

        let output = root.join("bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "myjsx:div\n");
    }

    /// A JavaScript project states its compiler options in `jsconfig.json`. It is
    /// the only place such a project can put `jsxImportSource` at all, so a build
    /// that never reads it silently lowers the whole app against React.
    #[test]
    fn a_jsconfig_import_source_reaches_a_javascript_project() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        write_jsx_runtime_package(root, "myjsx");
        fs::write(
            root.join("jsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-jsx","jsxImportSource":"myjsx"}}"#,
        )
        .unwrap();
        let entry = root.join("src").join("main.jsx");
        fs::write(&entry, "console.log(<div />);\n").unwrap();

        let output = root.join("bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "myjsx:div\n");
    }

    /// ONE app, ONE JSX runtime. `create-next-app`'s tsconfig `include`s only
    /// `**/*.ts` and `**/*.tsx`, and Next compiles JSX in `.js` (and `.mdx`) too:
    /// under a type-checking ownership rule the `.tsx` modules take the configured
    /// import source while the `.js` ones silently take React — two runtimes in one
    /// bundle, and (for a preact app) one of them not installed.
    #[test]
    fn every_extension_in_one_project_lowers_against_the_same_import_source() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("app")).unwrap();
        write_jsx_runtime_package(root, "myjsx");
        write_jsx_runtime_package(root, "react");
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"preserve","jsxImportSource":"myjsx","allowJs":true},
                "include":["next-env.d.ts","**/*.ts","**/*.tsx"],
                "exclude":["node_modules"]}"#,
        )
        .unwrap();
        fs::write(
            root.join("app").join("legacy.js"),
            "export const Legacy = () => <span />;\n",
        )
        .unwrap();
        let entry = root.join("app").join("page.tsx");
        fs::write(
            &entry,
            "import { Legacy } from './legacy.js';\nconsole.log(<div />, Legacy());\n",
        )
        .unwrap();

        let output = root.join("bundle.js");
        // Next's rule: `.js` may contain JSX (`diffpack_core::parser::JsxExtensions::JsxInJavaScript`).
        let config = BuildConfig {
            jsx_extensions: diffpack_core::parser::JsxExtensions::JsxInJavaScript,
            ..BuildConfig::default()
        };
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();
        assert_eq!(run_node(&output), "myjsx:div myjsx:span\n");
    }

    /// A `jsx` value diffpack cannot honor names the tsconfig and the value, and
    /// stops the build — a silently mislowered module would be a bundle whose
    /// runtime import points at the wrong package.
    #[test]
    fn an_unsupported_tsconfig_jsx_value_is_a_named_hard_error() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("tsconfig.json"),
            r#"{"compilerOptions":{"jsx":"react-native-web"},"include":["src"]}"#,
        )
        .unwrap();
        let entry = root.join("src").join("entry.tsx");
        fs::write(&entry, "export const view = <div />;\n").unwrap();

        let Err(error) = discover_direct(&entry) else {
            panic!("an unsupported tsconfig `jsx` value must stop the build");
        };
        assert!(
            error.contains("tsconfig.json")
                && error.contains("react-native-web")
                && error.contains("entry.tsx"),
            "the error must name the tsconfig, the value and the file: {error}"
        );
    }

    #[test]
    fn a_minified_chunk_runs_identically_to_its_readable_form_and_is_smaller() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let a = directory.path().join("a.js");
        // Multi-line source with comments and whitespace, so a real whitespace/
        // syntax minification pass has something to collapse and drop.
        fs::write(
            &entry,
            concat!(
                "// entry comment\n",
                "import { a } from './a.js';\n",
                "import { b } from './b.js';\n",
                "\n",
                "function total(left, right) {\n",
                "    /* add the two operands */\n",
                "    const sum = left + right;\n",
                "    return sum;\n",
                "}\n",
                "\n",
                "console.log(total(a, b));\n",
            ),
        )
        .unwrap();
        fs::write(&a, "// module a\nexport const a = 1 + 2;\n").unwrap();
        fs::write(
            directory.path().join("b.js"),
            "// module b\nexport const b = 3;\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty());
        let reachable = bundler.reachable_modules_direct();

        // Emit the readable form.
        let readable = directory.path().join("readable.js");
        bundler
            .emit_with_options(&reachable, &readable, EmitOptions::default())
            .unwrap();
        let readable_code = fs::read_to_string(&readable).unwrap();

        // Emit the minified form (same graph, `minify: true`).
        let minified = directory.path().join("minified.js");
        bundler
            .emit_with_options(
                &reachable,
                &minified,
                EmitOptions {
                    minify: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let minified_code = fs::read_to_string(&minified).unwrap();

        // Behavior is identical: both run under node and print the same value.
        assert_eq!(run_node(&readable), "6\n");
        assert_eq!(
            run_node(&minified),
            run_node(&readable),
            "minified output must behave identically to the readable output"
        );

        // The minified bytes are genuinely smaller, have no comments, and are not
        // just the readable bytes passed through.
        assert!(
            minified_code.len() < readable_code.len(),
            "minified ({} bytes) must be smaller than readable ({} bytes)",
            minified_code.len(),
            readable_code.len(),
        );
        assert!(
            !minified_code.contains("entry comment")
                && !minified_code.contains("add the two operands")
                && !minified_code.contains("module a"),
            "minified output still carries comments: {minified_code}"
        );
        assert_ne!(
            minified_code, readable_code,
            "minify must actually transform the bytes"
        );
    }

    /// The build config every source-map test uses: the per-module maps the
    /// printer produces are only built when the build asks for them.
    fn source_map_config() -> BuildConfig {
        BuildConfig {
            source_maps: true,
            ..BuildConfig::default()
        }
    }

    /// A TypeScript module whose interesting identifiers sit BELOW erased,
    /// type-only statements — the exact shape a line-identity map gets wrong,
    /// because every erased line shifts the real code up by one.
    ///
    /// Returns `(source, marker_line, marker_column, call_line, call_column)`,
    /// all 0-based, for the `MARKER_ALPHA` literal and the `greet` call.
    fn typed_module_with_erased_lines() -> (&'static str, u32, u32, u32, u32) {
        // line 0: comment      line 1: interface   line 2: type alias  line 3: blank
        // line 4: export fn    line 5: const       line 6: return      line 7: }
        let source = concat!(
            "// a leading comment\n",
            "interface Props { label: string }\n",
            "type Unused = number\n",
            "\n",
            "export function greet(props: Props) {\n",
            "  const marker = \"MARKER_ALPHA\"\n",
            "  return props.label + marker + globalThis.who\n",
            "}\n",
        );
        (source, 5, 17, 4, 16)
    }

    /// Finds the 0-based (line, column) of `needle` in `text`, in UTF-16 columns.
    fn position_of(text: &str, needle: &str) -> (u32, u32) {
        let byte = text
            .find(needle)
            .unwrap_or_else(|| panic!("`{needle}` must be present in:\n{text}"));
        let prefix = &text[..byte];
        let line_start = prefix.rfind('\n').map_or(0, |newline| newline + 1);
        (
            prefix.matches('\n').count() as u32,
            diffpack_core::source_map::utf16_len(&text[line_start..byte]),
        )
    }

    /// A READABLE chunk's map must resolve a known identifier to the EXACT
    /// original line AND column it came from — not to the generated line's index,
    /// which is what a line-identity guess produces and what every erased
    /// TypeScript line above the identifier makes wrong.
    #[test]
    fn a_readable_chunk_map_resolves_a_known_identifier_to_its_exact_original_line_and_column() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        let a = directory.path().join("a.ts");
        let (source, marker_line, marker_column, greet_line, greet_column) =
            typed_module_with_erased_lines();
        fs::write(&a, source).unwrap();
        fs::write(
            &entry,
            "import { greet } from './a.ts';\nconsole.log(greet({ label: \"x\" }));\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let code = fs::read_to_string(&output).unwrap();
        let map_json = fs::read_to_string(directory.path().join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let table = map.generate_lookup_table();

        // The string literal in the EMITTED chunk resolves to the literal in a.ts:
        // line 5, column 17 — four lines below where a line-identity map would put
        // it, because the comment, the interface and the type alias all vanished.
        let (line, column) = position_of(&code, "\"MARKER_ALPHA\"");
        let token = map
            .lookup_token(&table, line, column)
            .expect("the literal's position must be mapped");
        assert_eq!(
            (
                token.get_source_id().and_then(|id| map.get_source(id)),
                token.get_src_line(),
                token.get_src_col(),
            ),
            (Some("diffpack:///a.ts"), marker_line, marker_column),
            "the emitted literal must resolve to a.ts {}:{marker_column}, got {:?}",
            marker_line + 1,
            (token.get_src_line(), token.get_src_col()),
        );

        // ...and so does the function NAME, at its own exact column.
        let (line, column) = position_of(&code, "greet(props)");
        let token = map
            .lookup_token(&table, line, column)
            .expect("the declaration's position must be mapped");
        assert_eq!(
            (token.get_src_line(), token.get_src_col()),
            (greet_line, greet_column),
            "the emitted `greet` declaration must resolve to a.ts {}:{greet_column}",
            greet_line + 1,
        );

        // Every mapped column must be a REAL column: a map that had given up and
        // pinned everything to column 0 would pass a line-only check.
        assert!(
            map.get_tokens().any(|token| token.get_src_col() > 0),
            "the map must carry real columns, not column 0 for everything"
        );

        // A bundler-synthesized line owns no original position and must be
        // EXPLICITLY unmapped rather than be attributed to whatever module is
        // nearby. Explicitly matters: omitting a token does not mark a line
        // unmapped, because a consumer resolves a position to the last mapping at
        // or before it anywhere in the file, so a line with nothing on it inherits
        // the previous line's origin.
        let (line, _) = position_of(&code, "console.log");
        let separator = line - 1;
        let marker = map
            .get_tokens()
            .find(|token| token.get_dst_line() == separator)
            .expect("the blank separator line must carry an explicit unmapped marker");
        assert_eq!(
            (marker.get_dst_col(), marker.get_source_id()),
            (0, None),
            "the marker must be a source-less segment at the start of the line, got {marker:?}"
        );
        assert!(
            map.lookup_token(&table, separator, 0)
                .is_none_or(|token| token.get_source_id().is_none()),
            "resolving the blank separator line must not name any original source"
        );
    }

    /// Every generated line of a readable chunk that no module accounts for must
    /// say so, with its own unmapped segment.
    ///
    /// This is the whole honesty mechanism, and leaving the token out does NOT
    /// achieve it: Node's `--enable-source-maps` (which is how `diffpack start`
    /// runs a server) and DevTools both binary-search the flattened mapping list
    /// and return the last entry at or before the queried position, IGNORING line
    /// boundaries. So a bundler-authored line with no segments resolves to
    /// whatever author code was mapped before it — which is how a frame inside the
    /// bundler's own `__require` came out attributed to a component, at a line and
    /// column that exist, in a file that has no such code.
    #[test]
    fn every_line_a_module_does_not_account_for_carries_an_explicit_unmapped_marker() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        let a = directory.path().join("a.ts");
        let (source, ..) = typed_module_with_erased_lines();
        fs::write(&a, source).unwrap();
        // A CommonJS dependency forces the full registry runtime into the chunk,
        // so the chunk really does interleave author code with bundler-authored
        // text — which is the situation the markers exist for.
        fs::write(
            directory.path().join("legacy.cjs"),
            "module.exports = { legacy: 1 };\n",
        )
        .unwrap();
        fs::write(
            &entry,
            "import { greet } from \"./a\";\nimport legacy from \"./legacy.cjs\";\n\
             console.log(greet({ label: \"x\" }), legacy);\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let code = fs::read_to_string(&output).unwrap();
        let map_json = fs::read_to_string(directory.path().join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let mut mapped_lines: HashSet<u32> = HashSet::new();
        let mut marked_lines: HashSet<u32> = HashSet::new();
        for token in map.get_tokens() {
            match token.get_source_id() {
                Some(_) => mapped_lines.insert(token.get_dst_line()),
                None => marked_lines.insert(token.get_dst_line()),
            };
        }
        let total = line_count(&code);
        let unaccounted: Vec<u32> = (0..total)
            .filter(|line| !mapped_lines.contains(line) && !marked_lines.contains(line))
            .collect();
        assert!(
            unaccounted.is_empty(),
            "generated lines {unaccounted:?} of a {total}-line chunk carry neither a mapping \
             nor an unmapped marker, so a consumer resolves them to the last mapping before \
             them:\n{code}"
        );
        assert!(
            !marked_lines.is_empty() && !mapped_lines.is_empty(),
            "the chunk must have both kinds of line — runtime/glue and module code — for this \
             to be testing anything (mapped: {}, marked: {})",
            mapped_lines.len(),
            marked_lines.len()
        );
        // The runtime's own `__require` is bundler-authored: resolving a position
        // in it must name no source at all.
        let table = map.generate_lookup_table();
        let (throw_line, throw_column) = position_of(&code, "Module is not loaded");
        assert!(
            map.lookup_token(&table, throw_line, throw_column)
                .is_none_or(|token| token.get_source_id().is_none()),
            "a position inside the bundler's own runtime must resolve to no original source"
        );
    }

    /// A `sources` label is a module's IDENTITY — DevTools' source tree and every
    /// error reporter dedupe on it — so it must carry the module's directory and
    /// stay the same in every chunk it appears in.
    ///
    /// The failure this locks out: a root computed per MAP collapses to the
    /// module's own directory whenever a chunk holds one module, and the label
    /// becomes a bare file name. On cal.com that turned nine different
    /// `pages/setup/index.tsx` files into one `diffpack:///Setup.tsx`, and thirty
    /// different `add.ts` files into one `diffpack:///add.ts`.
    #[test]
    fn same_named_modules_in_different_chunks_keep_distinct_directory_qualified_labels() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(root.join("package.json"), r#"{"name":"labels"}"#).unwrap();
        for area in ["alpha", "beta"] {
            fs::create_dir_all(root.join("src").join(area)).unwrap();
            fs::write(
                root.join("src").join(area).join("Setup.ts"),
                format!("export const AREA = \"{area}\";\n"),
            )
            .unwrap();
        }
        let entry = root.join("src").join("entry.ts");
        // Dynamic imports put each `Setup.ts` in its own chunk, which is exactly
        // when a per-chunk root degenerates to a bare file name.
        fs::write(
            &entry,
            "const both = [import(\"./alpha/Setup\"), import(\"./beta/Setup\")];\n\
             Promise.all(both).then((loaded) => console.log(loaded.length));\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = root.join("out").join("bundle.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let mut labels: Vec<String> = Vec::new();
        for file in fs::read_dir(root.join("out")).unwrap() {
            let path = file.unwrap().path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("map") {
                continue;
            }
            let json = fs::read_to_string(&path).unwrap();
            let map = SourceMap::from_json_string(&json).unwrap();
            labels.extend(map.get_sources().map(str::to_owned));
        }
        assert!(
            labels.contains(&"diffpack:///src/alpha/Setup.ts".to_string())
                && labels.contains(&"diffpack:///src/beta/Setup.ts".to_string()),
            "each module must be named by its path from the project root, got {labels:?}"
        );
    }

    /// A module OUTSIDE the project root (a package in a store elsewhere, a
    /// symlinked workspace, another volume) must never publish its absolute path:
    /// that names the machine and the user, and production maps are served to
    /// browsers. It must still be told apart from any other file, so the label
    /// keeps the path within its own package and disambiguates it.
    #[test]
    fn a_module_outside_the_project_root_is_labelled_without_leaking_its_absolute_path() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let root = directory.path();
        let project = root.join("project");
        fs::create_dir_all(project.join("src")).unwrap();
        fs::write(project.join("package.json"), r#"{"name":"project"}"#).unwrap();
        let outside = root.join("elsewhere").join("pkg");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("package.json"), r#"{"name":"faraway"}"#).unwrap();
        fs::write(
            outside.join("index.js"),
            "export const FAR = \"far\";\nexport function far(x) { return x + FAR; }\n",
        )
        .unwrap();
        let entry = project.join("src").join("entry.js");
        fs::write(
            &entry,
            "import { far } from \"../../elsewhere/pkg/index.js\";\nconsole.log(far(1));\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = project.join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let map_json = fs::read_to_string(project.join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let sources = map.get_sources().collect::<Vec<_>>();
        let leaked = root.to_string_lossy().to_string();
        assert!(
            sources.iter().all(|source| !source.contains(&leaked)
                && !source.contains("..")
                && source.starts_with("diffpack:///")),
            "no label may carry an absolute path or a traversal, got {sources:?} (root {leaked})"
        );
        assert!(
            sources
                .iter()
                .any(|source| source.contains("external/") && source.ends_with("pkg/index.js")),
            "the outside module must still be identifiable, got {sources:?}"
        );
        assert!(
            sources.contains(&"diffpack:///src/entry.js"),
            "a module INSIDE the project keeps its project-relative path, got {sources:?}"
        );
    }

    /// A module whose source diffpack REWROTE before parsing must not be
    /// presented as the file on disk: the map's positions index the rewritten
    /// text, so the label says so and the inlined content is that text.
    #[test]
    fn a_rewritten_source_is_labelled_and_carries_the_text_its_positions_index() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        // A Vite `define` rewrites the SOURCE before it is parsed, so every span
        // below the substitution is measured against text that is not on disk.
        fs::write(
            &entry,
            "const flag = __BUILD_FLAG__\nconsole.log(flag, globalThis.who)\n",
        )
        .unwrap();

        let config = BuildConfig {
            source_maps: true,
            source_policy: Arc::new(diffpack_vite_compat::source_policy::ViteSourcePolicy {
                defines: vec![("__BUILD_FLAG__".to_string(), "\"enabled\"".to_string())],
                ..Default::default()
            }),
            ..BuildConfig::default()
        };
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let map_json = fs::read_to_string(directory.path().join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let sources = map.get_sources().collect::<Vec<_>>();
        assert_eq!(
            sources,
            vec!["diffpack:///entry.ts?diffpack-generated=vite-replace&diffpack-graph=server"],
            "a rewritten source must be labelled as generated (and by which graph generated \
             it, since the same file rewrites differently per graph), not as the file on disk"
        );
        let content = map.get_source_content(0).expect("content must be inlined");
        assert!(
            content.contains("\"enabled\"") && !content.contains("__BUILD_FLAG__"),
            "sourcesContent must be the REWRITTEN text the positions were measured \
             against, got: {content}"
        );
    }

    /// DEV: the Fast Refresh instrumentation edits a module's lowered code AFTER
    /// the printer measured it — a whole line inserted at the top, and
    /// `import.meta.hot` rewritten in place. The map must move with it, or every
    /// position in a dev build is one line off.
    #[test]
    fn the_fast_refresh_instrumentation_moves_the_map_with_the_code_it_edits() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.tsx");
        // A component (so the Fast Refresh preamble is injected) that also reads
        // `import.meta.hot` (so the in-place rewrite runs too).
        let source = concat!(
            "// a comment\n",
            "type Props = { label: string }\n",
            "export function Widget(props: Props) {\n",
            "  const marker = \"MARKER_ALPHA\"\n",
            "  return marker + props.label\n",
            "}\n",
            "if (import.meta.hot) { import.meta.hot.accept() }\n",
            "console.log(Widget({ label: globalThis.who }))\n",
        );
        fs::write(&entry, source).unwrap();

        let config = BuildConfig {
            source_maps: true,
            hmr: true,
            target: Target::Client,
            ..BuildConfig::default()
        };
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map: true,
                    hmr: true,
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let code = fs::read_to_string(&output).unwrap();
        assert!(
            code.contains("$RefreshReg$"),
            "the module must have been instrumented, or this test proves nothing"
        );
        let map_json = fs::read_to_string(directory.path().join("out.js.map")).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let table = map.generate_lookup_table();

        let (expected_line, expected_column) = position_of(source, "\"MARKER_ALPHA\"");
        let (line, column) = position_of(&code, "\"MARKER_ALPHA\"");
        let token = map
            .lookup_token(&table, line, column)
            .expect("the literal must still be mapped after instrumentation");
        assert_eq!(
            (token.get_src_line(), token.get_src_col()),
            (expected_line, expected_column),
            "the instrumented module's map must still point at the original literal",
        );
    }

    /// Emitting a map from a bundler that was never asked to build the per-module
    /// maps is refused, loudly. There is no cheaper, guessed map to fall back to.
    #[test]
    fn emitting_a_source_map_without_the_per_module_maps_is_a_hard_error() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(&entry, "console.log(globalThis.who);\n").unwrap();
        let (bundler, _) = discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("out.js"),
                EmitOptions {
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .expect_err("a map with no measured positions must be refused");
        assert!(
            error.contains("source_maps"),
            "the refusal must name the setting that fixes it, got: {error}"
        );
    }

    /// A multi-chunk app whose modules are split across chunks by dynamic import,
    /// so the coverage assertions below run over MORE than the entry chunk.
    fn code_split_source_map_project(directory: &Path) -> PathBuf {
        let entry = directory.join("entry.ts");
        fs::write(
            &entry,
            "import { shared } from \"./shared\";\n\
             export async function boot(): Promise<string> {\n\
             \x20 const lazy = await import(\"./lazy\");\n\
             \x20 return shared() + lazy.lazily();\n\
             }\n",
        )
        .unwrap();
        fs::write(
            directory.join("shared.ts"),
            "interface Erased { gone: boolean }\n\
             type AlsoErased = string;\n\
             export function shared(): string {\n\
             \x20 return \"SHARED_MARKER\";\n\
             }\n",
        )
        .unwrap();
        fs::write(
            directory.join("lazy.ts"),
            "import { shared } from \"./shared\";\n\
             type Gone = number;\n\
             export function lazily(): string {\n\
             \x20 return \"LAZY_MARKER\" + shared();\n\
             }\n",
        )
        .unwrap();
        entry
    }

    /// Every JS file an emit writes either carries NO `sourceMappingURL` at all or
    /// carries one whose file was really written, and whose `file` field names the
    /// chunk it belongs to.
    ///
    /// A dangling `sourceMappingURL` is not a cosmetic defect: the browser fetches
    /// it on every load of the chunk and logs a failure, and a `file` field naming
    /// some other chunk sends a map consumer to the wrong bytes. Both are the kind
    /// of drift that appears the moment a second writer (here: the dev HMR
    /// micro-chunk) names its sidecar itself, which is why the naming lives in one
    /// place — [`Bundler::source_map_sidecar`].
    #[test]
    fn every_emitted_chunk_points_at_a_map_that_exists_and_names_itself() {
        for minify in [false, true] {
            let directory = tempdir().unwrap();
            let entry = code_split_source_map_project(directory.path());
            let (bundler, update) =
                discover_direct_with_config(&entry, &source_map_config()).unwrap();
            assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
            let reachable = bundler.reachable_modules_direct();
            let out_root = directory.path().join("out");
            bundler
                .emit_public(
                    &reachable,
                    &out_root,
                    EmitOptions {
                        source_map: true,
                        minify,
                        ..EmitOptions::default()
                    },
                )
                .unwrap();

            let public = out_root.join("public");
            let mut checked = 0;
            for entry in fs::read_dir(&public).unwrap() {
                let path = entry.unwrap().path();
                let name = path.file_name().unwrap().to_str().unwrap().to_string();
                if !name.ends_with(".js") {
                    continue;
                }
                checked += 1;
                let code = fs::read_to_string(&path).unwrap();
                let reference = code
                    .rsplit("//# sourceMappingURL=")
                    .next()
                    .filter(|_| code.contains("//# sourceMappingURL="))
                    .map(|tail| tail.trim().to_string())
                    .unwrap_or_else(|| {
                        panic!("{name} was emitted with source maps on but names no map")
                    });
                let map_path = public.join(&reference);
                assert!(
                    map_path.is_file(),
                    "{name} points at {reference}, which was never written — a browser \
                     fetches that on every load and gets a 404"
                );
                let map: serde_json::Value =
                    serde_json::from_str(&fs::read_to_string(&map_path).unwrap()).unwrap();
                assert_eq!(
                    map.get("file").and_then(|value| value.as_str()),
                    Some(name.as_str()),
                    "{reference} claims to describe a different chunk"
                );
            }
            assert!(
                checked > 1,
                "the fixture must emit MORE than one chunk (minify={minify}), or this \
                 proves nothing about chunks past the entry"
            );
        }
    }

    /// The dev HMR micro-chunk — the code the developer is editing RIGHT NOW — ships
    /// with its own map, and that map resolves back to the edited file.
    ///
    /// This is the most user-visible source map diffpack writes: it is the one a
    /// stack trace lands in seconds after a save. It previously shipped with none at
    /// all, so the hot-updated module was the one region of a dev session with no
    /// mapping.
    #[test]
    fn the_hmr_micro_chunk_ships_a_map_that_resolves_to_the_edited_source() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        let edited = directory.path().join("edited.ts");
        fs::write(
            &entry,
            "import { hot } from \"./edited\";\nconsole.log(hot());\n",
        )
        .unwrap();
        // Type-only lines above the marker, so a line-identity guess would be wrong.
        let source = "interface Erased { gone: boolean }\n\
                      type AlsoErased = string;\n\
                      \n\
                      export function hot(): string {\n\
                      \x20 return \"HOT_MARKER\";\n\
                      }\n";
        fs::write(&edited, source).unwrap();

        let config = BuildConfig {
            source_maps: true,
            hmr: true,
            target: Target::Client,
            ..BuildConfig::default()
        };
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let changed: BTreeSet<ModuleId> = reachable
            .iter()
            .filter(|id| id.contains("edited.ts"))
            .cloned()
            .collect();
        assert_eq!(changed.len(), 1, "the edited module must be in the graph");

        let chunk = directory.path().join("client.hmr.js");
        let wrote = bundler
            .write_hmr_chunk(
                &reachable,
                &changed,
                "client.js",
                EmitOptions {
                    source_map: true,
                    hmr: true,
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
                ModuleFormat::BrowserEsm,
                &chunk,
            )
            .unwrap();
        assert!(
            wrote,
            "the edited module is live, so a micro-chunk must render"
        );

        let code = fs::read_to_string(&chunk).unwrap();
        assert!(
            code.trim_end()
                .ends_with("//# sourceMappingURL=client.hmr.js.map"),
            "the micro-chunk must name its map, or the browser never loads it"
        );
        let map_path = directory.path().join("client.hmr.js.map");
        let map_json = fs::read_to_string(&map_path).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let table = map.generate_lookup_table();

        let (expected_line, expected_column) = position_of(source, "\"HOT_MARKER\"");
        let (line, column) = position_of(&code, "\"HOT_MARKER\"");
        let token = map
            .lookup_token(&table, line, column)
            .expect("the marker must be mapped in the micro-chunk");
        assert_eq!(
            (token.get_src_line(), token.get_src_col()),
            (expected_line, expected_column),
            "the micro-chunk's map must resolve to the edited file's real position, \
             not to the generated line number"
        );
    }

    /// With source maps OFF, the micro-chunk carries no dangling reference: no map
    /// file, and no `sourceMappingURL` for the browser to chase.
    #[test]
    fn the_hmr_micro_chunk_names_no_map_when_source_maps_are_off() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let edited = directory.path().join("edited.js");
        fs::write(
            &entry,
            "import { hot } from \"./edited\";\nconsole.log(hot());\n",
        )
        .unwrap();
        fs::write(&edited, "export function hot() {\n  return \"HOT\";\n}\n").unwrap();
        let config = BuildConfig {
            hmr: true,
            target: Target::Client,
            ..BuildConfig::default()
        };
        let (bundler, _) = discover_direct_with_config(&entry, &config).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let changed: BTreeSet<ModuleId> = reachable
            .iter()
            .filter(|id| id.contains("edited.js"))
            .cloned()
            .collect();
        let chunk = directory.path().join("client.hmr.js");
        assert!(
            bundler
                .write_hmr_chunk(
                    &reachable,
                    &changed,
                    "client.js",
                    EmitOptions {
                        source_map: false,
                        hmr: true,
                        format: ModuleFormat::BrowserEsm,
                        ..EmitOptions::default()
                    },
                    ModuleFormat::BrowserEsm,
                    &chunk,
                )
                .unwrap()
        );
        assert!(
            !fs::read_to_string(&chunk)
                .unwrap()
                .contains("sourceMappingURL")
        );
        assert!(!directory.path().join("client.hmr.js.map").exists());
    }

    /// `__dirname`/`__filename` in a SPLIT Node ESM build. Every chunk is its own ES
    /// module, so it does NOT close over the entry's bindings: a bundled CommonJS module
    /// that reads `__dirname` and lands behind a dynamic `import()` threw
    /// `ReferenceError: __dirname is not defined in ES module scope` the moment its chunk
    /// was loaded. That is not hypothetical — it is how cal.com's `pages/api/**` routes
    /// died: Prisma's generated client reads `__dirname`, and in the SSR graph it is
    /// reachable ONLY through those lazily-imported route chunks. The prelude therefore
    /// belongs on every Node ESM chunk, not just the entry.
    #[test]
    fn a_split_node_chunk_defines_dirname_for_a_bundled_commonjs_module() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        // The exact shape ncc-compiled / generated CJS packages emit at module scope.
        fs::write(
            directory.path().join("vendored.js"),
            "const base = __dirname + \"/\";\nmodule.exports = { base, file: __filename };\n",
        )
        .unwrap();
        // `lazy.js` is only reachable through the dynamic import, so it (and the CJS
        // module it pulls in) lands in a chunk of its own.
        fs::write(
            directory.path().join("lazy.js"),
            "const vendored = require(\"./vendored.js\");\nexport const where = vendored.base;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import(\"./lazy.js\").then(({ where }) => console.log(\"base:\" + where));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        let server_dir = output_root.join("server");
        let chunk = server_dir.join("server.chunk-1.mjs");
        assert!(chunk.is_file(), "the dynamic import lands in its own chunk");
        assert!(
            fs::read_to_string(&chunk)
                .unwrap()
                .contains("const __dirname = __diffpackDirname(__filename)"),
            "the split chunk must define __dirname from its own import.meta.url",
        );

        // Running it is the real proof: the chunk is imported at runtime, and without
        // the prelude that import rejects with a ReferenceError.
        let executed = node_command()
            .arg(server_dir.join("server.mjs"))
            .output()
            .unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        let stdout = String::from_utf8_lossy(&executed.stdout);
        // Compared against the CANONICAL path: macOS resolves the temp dir through
        // `/private`, and `import.meta.url` carries the resolved form.
        let canonical = server_dir.canonicalize().unwrap();
        assert!(
            stdout.contains(&format!("base:{}/", canonical.display())),
            "the chunk resolves __dirname to its own directory: {stdout}"
        );
    }

    /// A hot-updated module must land in the SAME environment its graph was emitted
    /// for. `__dirname`/`__filename` are the sharp edge: browser output substitutes
    /// the stubs `"/index.js"` and `"/"` (a browser has no CommonJS locations), while
    /// Node ESM output binds the entry's real values. Rendering a SERVER micro-chunk as
    /// browser output therefore swaps a server module's file paths for stubs the
    /// instant it is hot-updated — a fault that is invisible until an edit, and then
    /// only on a module that reads a file. `write_hmr_chunk` takes the format
    /// explicitly for this reason; this pins both sides of the choice.
    #[test]
    fn the_hmr_micro_chunk_renders_dirname_for_the_format_its_graph_was_emitted_for() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let edited = directory.path().join("edited.js");
        fs::write(
            &entry,
            "import { where } from \"./edited\";\nconsole.log(where());\n",
        )
        .unwrap();
        fs::write(
            &edited,
            "export function where() {\n  return __dirname;\n}\n",
        )
        .unwrap();
        let config = BuildConfig {
            hmr: true,
            ..BuildConfig::default()
        };
        let (bundler, _) = discover_direct_with_config(&entry, &config).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let changed: BTreeSet<ModuleId> = reachable
            .iter()
            .filter(|id| id.contains("edited.js"))
            .cloned()
            .collect();
        let options = EmitOptions {
            hmr: true,
            ..EmitOptions::default()
        };

        let browser = directory.path().join("client.hmr.js");
        assert!(
            bundler
                .write_hmr_chunk(
                    &reachable,
                    &changed,
                    "client.js",
                    options,
                    ModuleFormat::BrowserEsm,
                    &browser,
                )
                .unwrap()
        );
        assert!(
            fs::read_to_string(&browser)
                .unwrap()
                .contains("__dirname=\"/\""),
            "a browser micro-chunk must carry the browser CommonJS-location stubs",
        );

        let node = directory.path().join("server.hmr.mjs");
        assert!(
            bundler
                .write_hmr_chunk(
                    &reachable,
                    &changed,
                    "server.mjs",
                    options,
                    ModuleFormat::Esm,
                    &node,
                )
                .unwrap()
        );
        assert!(
            !fs::read_to_string(&node)
                .unwrap()
                .contains("__dirname=\"/\""),
            "a Node micro-chunk must NOT stub __dirname; it binds the entry's real value",
        );
    }

    #[test]
    fn a_minified_chunk_emits_a_composed_source_map_resolving_to_the_original_source() {
        use oxc_sourcemap::SourceMap;

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.ts");
        let a = directory.path().join("a.ts");
        // `a.ts` must still contribute generated bytes AFTER compression, or there
        // is no cross-module position left to sample. A single `const` would be
        // inlined into its one use and the module would vanish entirely (correctly
        // — that is what esbuild does too), so `a.ts` exports a function called
        // from two places, which the minifier keeps as a real binding. Its
        // interesting lines sit below erased TypeScript, so a line-identity map
        // resolves them four lines too high.
        let (source, marker_line, marker_column, greet_line, greet_column) =
            typed_module_with_erased_lines();
        fs::write(&a, source).unwrap();
        fs::write(
            &entry,
            "import { greet } from './a.ts';\nconsole.log(greet({ label: globalThis.who }));\nconsole.log(greet({ label: globalThis.other }));\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct_with_config(&entry, &source_map_config()).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let output = directory.path().join("out.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    minify: true,
                    source_map: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        // The emitted (minified) chunk references its sibling map.
        let code = fs::read_to_string(&output).unwrap();
        assert!(
            code.contains("//# sourceMappingURL=out.js.map"),
            "minified chunk must reference its sibling map: {code}"
        );
        // It is genuinely minified (no source comments/newlines-per-statement).
        assert!(
            !code.contains("MARKER_ALPHA\";\n"),
            "the chunk must be minified, got: {code}"
        );

        // The map is valid JSON listing the real original sources with their
        // content inlined, under project-relative, traversal-free labels.
        let map_path = directory.path().join("out.js.map");
        let map_json = fs::read_to_string(&map_path).unwrap();
        let map = SourceMap::from_json_string(&map_json).unwrap();
        let sources = map.get_sources().collect::<Vec<_>>();
        assert!(
            sources.iter().any(|source| source.ends_with("a.ts"))
                && sources.iter().any(|source| source.ends_with("entry.ts")),
            "sources must list the real original modules, got {sources:?}"
        );
        assert!(
            sources
                .iter()
                .all(|source| source.starts_with("diffpack:///") && !source.contains("..")),
            "source labels must be project-relative and traversal-free, got {sources:?}"
        );
        let a_index = sources
            .iter()
            .position(|source| source.ends_with("a.ts"))
            .expect("a.ts must be a source");
        let a_content = map.get_source_content(a_index as u32);
        assert!(
            a_content.is_some_and(|content| content.contains("MARKER_ALPHA")),
            "sourcesContent must carry the real a.ts source, got {a_content:?}"
        );

        let table = map.generate_lookup_table();
        // A sampled MINIFIED position — the string literal that came from a.ts —
        // decodes back to a.ts at an EXACT line and column. The minifier inlined
        // the `marker` constant into its use, so the honest answer is the USE
        // site, not the declaration: line 7, column 23 of a.ts. Under the
        // line-identity map every position on the minified chunk's single line
        // resolved to line 1 of whichever module owned readable line 0.
        let (inlined_line, inlined_column) = position_of(source, "marker + globalThis");
        assert_ne!(
            (inlined_line, inlined_column),
            (marker_line, marker_column),
            "the use site and the declaration must be distinguishable"
        );
        let (line, column) = position_of(&code, "MARKER_ALPHA");
        let token = map
            .lookup_token(&table, line, column.saturating_sub(1))
            .expect("the sampled minified position must be mapped");
        assert_eq!(
            (
                token.get_source_id().and_then(|id| map.get_source(id)),
                token.get_src_line(),
                token.get_src_col(),
            ),
            (Some("diffpack:///a.ts"), inlined_line, inlined_column),
            "the minified literal must resolve to the `marker` use at a.ts {}:{inlined_column}, got {:?}",
            inlined_line + 1,
            (token.get_src_line(), token.get_src_col()),
        );
        let _ = (marker_line, marker_column);

        // The MANGLED function binding resolves to the original declaration AND
        // recovers its original NAME — the whole point of `names` in a production
        // map, and something no line-granular map can provide.
        let mangled = code
            .split_once("function ")
            .map(|(_, rest)| {
                rest.chars()
                    .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                    .collect::<String>()
            })
            .expect("the minified chunk declares the hoisted function");
        assert_ne!(mangled, "greet", "the minifier must have renamed it");
        let (line, column) = position_of(&code, &format!("function {mangled}"));
        let token = map
            .lookup_token(&table, line, column + "function ".len() as u32)
            .expect("the mangled binding must be mapped");
        assert_eq!(
            (token.get_src_line(), token.get_src_col()),
            (greet_line, greet_column),
            "the mangled binding must resolve to a.ts {}:{greet_column}",
            greet_line + 1,
        );
        assert_eq!(
            token.get_name_id().and_then(|id| map.get_name(id)),
            Some("greet"),
            "the composed map must recover the original identifier for a mangled name"
        );
    }

    #[test]
    fn direct_reachability_collects_a_detached_cycle_locally() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let a = directory.path().join("a.js");
        fs::write(
            &entry,
            concat!(
                "import './a.js';\n",
                "import './leaf-0.js';\n",
                "import './leaf-1.js';\n",
                "import './leaf-2.js';\n",
                "import './leaf-3.js';\n",
                "import './leaf-4.js';\n",
                "import './leaf-5.js';\n",
                "import './leaf-6.js';\n",
                "import './leaf-7.js';\n",
            ),
        )
        .unwrap();
        fs::write(&a, "import './b.js';").unwrap();
        fs::write(directory.path().join("b.js"), "import './a.js';").unwrap();
        for index in 0..8 {
            fs::write(
                directory.path().join(format!("leaf-{index}.js")),
                format!("export const leaf = {index};"),
            )
            .unwrap();
        }

        let (mut bundler, _) = discover(&entry).unwrap();
        let mut direct = bundler.direct_reachability();
        fs::write(
            &entry,
            concat!(
                "import './leaf-0.js';\n",
                "import './leaf-1.js';\n",
                "import './leaf-2.js';\n",
                "import './leaf-3.js';\n",
                "import './leaf-4.js';\n",
                "import './leaf-5.js';\n",
                "import './leaf-6.js';\n",
                "import './leaf-7.js';\n",
            ),
        )
        .unwrap();

        let revision = bundler.rebuild_path(&entry).unwrap();
        let update = direct.apply(&revision.delta);

        assert_eq!(update.removed.len(), 2);
        assert!(!update.used_full_recompute);
        assert_eq!(
            direct.reachable_modules(),
            bundler.reachable_modules_direct()
        );
    }

    #[test]
    fn deleting_a_non_tree_edge_does_not_scan_or_change_reachability() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let a = directory.path().join("a.js");
        fs::write(&entry, "import './a.js'; import './b.js';").unwrap();
        fs::write(&a, "import './b.js';").unwrap();
        fs::write(directory.path().join("b.js"), "export const b = 1;").unwrap();

        let (mut bundler, _) = discover(&entry).unwrap();
        let mut direct = bundler.direct_reachability();
        fs::write(&a, "export const a = 1;").unwrap();
        let revision = bundler.rebuild_path(&a).unwrap();
        let update = direct.apply(&revision.delta);

        assert!(update.added.is_empty());
        assert!(update.removed.is_empty());
        assert!(!update.used_full_recompute);
        assert_eq!(
            direct.reachable_modules(),
            bundler.reachable_modules_direct()
        );
    }

    #[test]
    fn direct_reachability_falls_back_for_a_large_detached_subtree() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(&entry, "import './a.js';").unwrap();
        fs::write(directory.path().join("a.js"), "import './b.js';").unwrap();
        fs::write(directory.path().join("b.js"), "export const b = 1;").unwrap();

        let (mut bundler, _) = discover(&entry).unwrap();
        let mut direct = bundler.direct_reachability();
        fs::write(&entry, "export const entry = 1;").unwrap();
        let revision = bundler.rebuild_path(&entry).unwrap();
        let update = direct.apply(&revision.delta);

        assert!(update.used_full_recompute);
        assert_eq!(update.removed.len(), 2);
        assert_eq!(
            direct.reachable_modules(),
            bundler.reachable_modules_direct()
        );
    }

    #[test]
    fn emit_public_writes_a_client_layout_with_chunks_css_and_assets() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("style.css"), ".a { color: red; }").unwrap();
        fs::write(directory.path().join("logo.svg"), "<svg></svg>").unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './style.css';\nimport logo from './logo.svg';\n\
             console.log(logo);\nimport('./lazy.js').then(({ lazy }) => console.log(lazy));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let summary = bundler
            .emit_public(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        // A main chunk plus the dynamically imported chunk.
        assert!(
            summary.javascript_files >= 2,
            "expected the entry chunk and a dynamic chunk: {summary:?}"
        );
        assert_eq!(
            summary.css_files, 1,
            "one extracted stylesheet: {summary:?}"
        );
        assert_eq!(summary.asset_files, 1, "one hashed asset: {summary:?}");

        let public_dir = output_root.join("public");
        assert!(public_dir.join("client.js").is_file());
        assert!(public_dir.join("client.css").is_file());
        assert!(
            public_dir.join("assets").read_dir().unwrap().count() == 1,
            "the svg asset is copied under assets/"
        );
        // The summary counts exactly the files on disk.
        let on_disk = EmitSummary::of(&public_dir).unwrap();
        assert_eq!(on_disk.javascript_files, summary.javascript_files);
        assert_eq!(on_disk.css_files, summary.css_files);
        assert_eq!(on_disk.asset_files, summary.asset_files);

        // A re-emit rebuilds `public/` from scratch: a file that would no longer
        // be produced does not linger.
        let stale = public_dir.join("stale.js");
        fs::write(&stale, "// stale").unwrap();
        bundler
            .emit_public(&reachable, &output_root, EmitOptions::default())
            .unwrap();
        assert!(!stale.exists(), "re-emit must clear stale output");
    }

    /// REGRESSION. The client emit's prune deletes everything under `public/` the
    /// client graph did not itself write — correct for the client's own stale chunks,
    /// wrong for a file another graph published there. `public/rsc.css` is published by
    /// the REACT-SERVER graph, which in `diffpack dev` is built FIRST in the same
    /// process, so the client's prune deleted the sheet moments after it was written.
    /// The document kept linking `/rsc.css` (that link is guarded on the artifact beside
    /// the render bundle, which the prune never sees), `GET /rsc.css` returned the 404
    /// HTML shell, and `nosniff` made the browser reject it: cal.com and
    /// `integration/next-app-router` both rendered completely unstyled from a cold
    /// `diffpack dev`.
    ///
    /// A preserved path survives the prune; everything NOT preserved still goes, so the
    /// fix cannot degrade into "stop pruning".
    #[test]
    fn emit_public_prune_keeps_the_paths_this_emit_does_not_own() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("entry.js"), "console.log('app');\n").unwrap();
        let entry = directory.path().join("entry.js");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, _) = discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_public(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        // Stand in for what the react-server build preserves before the client emit runs.
        let public_dir = output_root.join("public");
        let rsc_css = public_dir.join("rsc.css");
        fs::write(&rsc_css, ".from-the-react-server-graph { color: red; }").unwrap();
        let stale = public_dir.join("stale.js");
        fs::write(&stale, "// stale").unwrap();

        let preserve = BTreeSet::from([rsc_css.clone()]);
        bundler
            .emit_public_preserving(&reachable, &output_root, EmitOptions::default(), &preserve)
            .unwrap();

        assert!(
            rsc_css.is_file(),
            "the client emit must not delete a file it does not own",
        );
        assert_eq!(
            fs::read_to_string(&rsc_css).unwrap(),
            ".from-the-react-server-graph { color: red; }",
            "preserving must leave the bytes alone, not just the path",
        );
        assert!(
            !stale.exists(),
            "an UNpreserved stale file must still be pruned",
        );
    }

    /// The client `public/` build must emit BROWSER-executable ESM: the entry
    /// `client.js` is injected by the SSR document as
    /// `<script type="module" src="/client.js">`, so a CommonJS `module.exports=…`
    /// entry throws `module is not defined` under the ESM goal and the app never
    /// hydrates. This builds a small app with a Node built-in external (forcing
    /// the shared registry runtime and thus the browser `requireNative` stub) and
    /// a dynamic import (a split chunk), emits it via `emit_public`, then LOADS
    /// the entry with `import()` under `node` (as an ESM oracle) and asserts the
    /// entry's top-level code ran — proving there is no `module is not defined`
    /// and no `node:module` import a browser could not resolve.
    #[test]
    fn emit_public_entry_loads_as_a_browser_es_module_under_node() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy-value';\n",
        )
        .unwrap();
        // `import os from 'node:os'` forces the runtime path (the flat path cannot
        // bind an external); it is used only inside a function, so module init
        // never calls the browser stub. The dynamic import forces a split chunk.
        fs::write(
            directory.path().join("entry.js"),
            "import os from 'node:os';\n\
             export function platform(){ return os.platform(); }\n\
             globalThis.__diffpack_client_ran = true;\n\
             import('./lazy.js').then((m) => { globalThis.__diffpack_lazy = m.lazy; });\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_public(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        let public_dir = output_root.join("public");
        let client = public_dir.join("client.js");
        // Every emitted client `.js` passes `node --check` under the ESM goal.
        for entry in fs::read_dir(&public_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|value| value.to_str()) == Some("js") {
                node_check(&path);
            }
        }
        // The browser entry has NO `node:module` import and DOES `export default`.
        let code = fs::read_to_string(&client).unwrap();
        assert!(
            !code.contains("node:module"),
            "browser ESM entry must not import node:module"
        );
        assert!(
            code.contains("export default"),
            "browser ESM entry must export a default"
        );

        // Load the entry as a real ES module. A CJS entry would throw
        // `module is not defined`; a `node:module` import would fail to resolve.
        let harness = public_dir.join("harness.mjs");
        fs::write(
            &harness,
            // The `setTimeout` lets the entry's `import('./lazy.js')` settle before
            // the split chunk's value is asserted. Loading is not enough: a flat
            // chunk consumed through the registry protocol resolves to `undefined`
            // rather than throwing, so a load-only assertion passes while the
            // dynamic import silently yields nothing.
            "import(process.argv[2]).then(() => new Promise((done) => setTimeout(done, 0))).then(() => { if (globalThis.__diffpack_client_ran !== true) { console.error('entry top-level did not run'); process.exit(3); } if (globalThis.__diffpack_lazy !== 'lazy-value') { console.error('SPLIT_CHUNK_VALUE:' + String(globalThis.__diffpack_lazy)); process.exit(5); } console.log('LOADED'); }).catch((e) => { console.error('LOAD_ERROR:' + e.message); process.exit(4); });\n",
        )
        .unwrap();
        let output = node_command().arg(&harness).arg(&client).output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("LOADED"),
            "client.js did not load as an ES module: stdout={stdout} stderr={stderr}"
        );
        assert!(
            !stderr.contains("module is not defined"),
            "`module is not defined` leaked: {stderr}"
        );
    }

    fn run_node(path: &Path) -> String {
        let output = node_command().arg(path).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    }

    /// Syntax-checks a file as JavaScript under the Node ESM goal. `node --check`
    /// is a build oracle only, never in the build path.
    fn node_check(path: &Path) {
        let output = node_command().arg("--check").arg(path).output().unwrap();
        assert!(
            output.status.success(),
            "node --check failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn emit_server_writes_an_mjs_layout_that_node_accepts() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("style.css"), ".a { color: red; }").unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';",
        )
        .unwrap();
        fs::write(
            directory.path().join("server.ts"),
            "import './style.css';\n\
             console.log('render');\n\
             import('./lazy.js').then(({ lazy }) => console.log(lazy));\n",
        )
        .unwrap();

        let entry = directory.path().join("server.ts");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let summary = bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        // The server entry plus its dynamically imported chunk, as `.mjs`.
        assert!(
            summary.javascript_files >= 2,
            "expected the server entry and a dynamic chunk: {summary:?}"
        );

        let server_dir = output_root.join("server");
        assert!(server_dir.join("server.mjs").is_file());
        assert!(
            server_dir.join("server.chunk-1.mjs").is_file(),
            "the dynamic import lands in an `.mjs` chunk"
        );
        // No stray `.js` in the server build: everything is Node ESM.
        assert_eq!(summary.output_dir, server_dir);

        // Every emitted `.mjs` must be syntactically valid under Node's ESM goal.
        for entry in fs::read_dir(&server_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|value| value.to_str()) == Some("mjs") {
                node_check(&path);
            }
        }

        // A re-emit rebuilds `server/` from scratch.
        let stale = server_dir.join("stale.mjs");
        fs::write(&stale, "// stale").unwrap();
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();
        assert!(!stale.exists(), "re-emit must clear stale output");
    }

    /// The server `.mjs` output must not merely pass `node --check`; it must
    /// EXECUTE under Node's ESM goal. This builds a small multi-module app with a
    /// static cross-module import, an external Node built-in (forcing the shared
    /// Top-level `await` cannot exist in CommonJS output or inside the factory
    /// runtime; both must be hard, module-naming errors (previously the build
    /// "succeeded" and emitted a bundle Node rejects at parse — the conformance
    /// suite's worst honesty finding). In single-chunk ESM output it is
    /// representable and must actually run.
    #[test]
    fn top_level_await_is_a_hard_error_in_cjs_and_runs_in_flat_esm() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { value } from './value.js';\nconsole.log('got:' + value);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("dist/out.js"),
                EmitOptions::default(),
            )
            .unwrap_err();
        assert!(error.contains("top-level await"), "{error}");
        assert!(error.contains("value.js"), "names the module: {error}");
        assert!(error.contains("--format esm"), "names the way out: {error}");

        let esm_out = directory.path().join("dist/out.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert_eq!(run_node(&esm_out), "got:tla-value\n");
    }

    /// Top-level `await` in a CODE-SPLIT ESM build. Splitting forces the registry
    /// runtime, whose factories used to be plain synchronous functions — so this
    /// whole graph was a hard "requires the single-chunk scope-hoisted ESM output"
    /// error, which is what cal.com's SSR bundle hit through
    /// `i18next-fs-backend` (`await import('node:fs')` at module scope).
    ///
    /// The awaiting module now renders as an `async` factory, and the property
    /// propagates up the static import edges: `value.js` awaits, so `middle.js`
    /// (which imports it) and the entry (which imports that) are async too, and
    /// each of their import sites awaits. The bundle must EXECUTE under Node and
    /// print the awaited value — not merely parse — and the dynamically imported
    /// chunk (which is what forces the split) must resolve to the finished
    /// namespace of its own async module.
    #[test]
    fn top_level_await_runs_in_a_code_split_esm_build() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        // A second hop, so the async property has to PROPAGATE rather than only
        // apply to the module that literally awaits.
        fs::write(
            directory.path().join("middle.js"),
            "import { value } from './value.js';\nexport const shouted = value.toUpperCase();\n",
        )
        .unwrap();
        // The dynamically imported module also awaits: `require.dynamic` must
        // resolve through the async path, or `lazy` reads back `undefined`.
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = await Promise.resolve('lazy-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { shouted } from './middle.js';\n\
             const { lazy } = await import('./lazy.js');\n\
             console.log('got:' + shouted + ':' + lazy);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert!(
            stats.written.len() >= 2,
            "the dynamic import must split off a chunk: {:?}",
            stats.written
        );
        for path in &stats.written {
            node_check(path);
        }
        let code = fs::read_to_string(&esm_out).unwrap();
        assert!(
            code.contains("async function(module,exports,require"),
            "an awaiting module renders as an async factory: {code}"
        );
        assert!(
            code.contains("await require.esmAsync(\"./value.js\")"),
            "the importer awaits its async dependency: {code}"
        );
        assert_eq!(run_node(&esm_out), "got:TLA-VALUE:lazy-value\n");
    }

    /// The same graph under the DEV (HMR) runtime, which used to reject it outright
    /// ("top-level await ... is not supported in a dev (HMR) build"). That refusal is
    /// what stopped `diffpack dev` on cal.com: its SSR graph reaches
    /// `i18next-fs-backend`, whose `readFile.js` does `await import('node:fs')` at
    /// module scope, so the whole dev server died before serving a request.
    ///
    /// The async machinery and the HMR machinery are independent and must compose:
    /// the HMR runtime has to publish `requireAsync` (a chunk whose root is async
    /// returns `__runtime.requireAsync(...)`), and its version-aware
    /// `require.dynamic` has to resolve through it. Node EXECUTING the bundle —
    /// including the dynamically imported, separately-chunked async module — is the
    /// assertion.
    #[test]
    fn top_level_await_runs_under_the_dev_hmr_runtime() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("middle.js"),
            "import { value } from './value.js';\nexport const shouted = value.toUpperCase();\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = await Promise.resolve('lazy-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { shouted } from './middle.js';\n\
             const { lazy } = await import('./lazy.js');\n\
             console.log('got:' + shouted + ':' + lazy);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    hmr: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert!(
            stats.written.len() >= 2,
            "the dynamic import must split off a chunk: {:?}",
            stats.written
        );
        for path in &stats.written {
            node_check(path);
        }
        let code = fs::read_to_string(&esm_out).unwrap();
        assert!(
            code.contains("requireAsync:__requireAsync,"),
            "the HMR runtime must publish requireAsync for an async chunk root: {code}"
        );
        assert!(
            code.contains("__requireAsync(chunk[1])"),
            "the version-aware dynamic require must resolve through the async path: {code}"
        );
        assert_eq!(run_node(&esm_out), "got:TLA-VALUE:lazy-value\n");
    }

    /// A hot update whose re-run reaches an ASYNC module must not report success (or
    /// publish a fresh SSR handler) until that module's top-level `await` has SETTLED.
    ///
    /// This is the exact hazard the old blanket refusal cited. `__require` returns a
    /// module's exports object synchronously in both cases — the object exists before
    /// the factory's first `await` — so a naive re-run looks like it worked while the
    /// module body is still suspended, and the dev server hands the next SSR request a
    /// half-initialised entry.
    ///
    /// The awaited work here is a TIMER, not a resolved promise, so no amount of
    /// microtask draining can accidentally make the assertion pass: only a real `await`
    /// of the module's pending initialisation does.
    #[test]
    fn a_hot_update_waits_for_an_async_modules_top_level_await() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let value_js = directory.path().join("value.js");
        fs::write(
            &value_js,
            "export const value = await new Promise(r => setTimeout(() => r('v1'), 20));\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { value } from './value.js';\n\
             (globalThis.__log ??= []).push(value);\n\
             export const label = value;\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let options = EmitOptions {
            format: ModuleFormat::Esm,
            hmr: true,
            ..EmitOptions::default()
        };
        let esm_out = directory.path().join("dist/out.mjs");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(&reachable, &esm_out, options)
            .unwrap();

        // The edit, exactly as the dev server applies one: re-discover, locate the
        // changed module's runtime id, and render the tiny register-only HMR chunk
        // carrying only its new factory.
        fs::write(
            &value_js,
            "export const value = await new Promise(r => setTimeout(() => r('v2'), 20));\n",
        )
        .unwrap();
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let changed = BTreeSet::from([ModuleId::from(
            value_js.canonicalize().unwrap().to_string_lossy().as_ref(),
        )]);
        let located = bundler.hmr_locate(&reachable, &changed, "out.mjs").unwrap();
        assert_eq!(located.len(), 1, "the edited module must be located");
        let runtime_id = located[0].runtime_id;
        let hmr_path = directory.path().join("dist/hmr-1.mjs");
        assert!(
            bundler
                .write_hmr_chunk(
                    &reachable,
                    &changed,
                    "out.mjs",
                    options,
                    ModuleFormat::Esm,
                    &hmr_path
                )
                .unwrap(),
            "the edited module is live, so it renders"
        );
        node_check(&hmr_path);

        // Drive the update the way the Node control endpoint does: register the new
        // factory, then `serverInvalidate`, which re-runs the entry in-process and
        // republishes the SSR handler.
        let harness = directory.path().join("dist/harness.mjs");
        fs::write(
            &harness,
            format!(
                "import './out.mjs';\n\
                 await import('./hmr-1.mjs?__diffpack_hmr=1');\n\
                 const rt = globalThis.__diffpack_hmr_runtime;\n\
                 await rt.serverInvalidate([{runtime_id}], []);\n\
                 console.log(JSON.stringify({{\n\
                 log: globalThis.__log,\n\
                 published: globalThis.__diffpack_ssr_entry.label,\n\
                 }}));\n"
            ),
        )
        .unwrap();
        assert_eq!(
            run_node(&harness),
            "{\"log\":[\"v1\",\"v2\"],\"published\":\"v2\"}\n",
            "the hot update must observe the re-run module's SETTLED top-level await"
        );
    }

    /// Every dev-client module's Fast Refresh registrations must be NAMESPACED by
    /// that module, so two modules that happen to define a same-named component are
    /// never mistaken for two versions of one component.
    ///
    /// oxc's refresh transform emits `$RefreshReg$(_c, "Widget")` — the local name
    /// only — and react-refresh keys families in ONE global map, so an unscoped id
    /// makes the second registration read as a hot update of the first. On cal.com
    /// that put hundreds of phantom updates in the queue before a single edit; the
    /// first real edit then swapped unrelated component types into the live tree and
    /// React's `scheduleRefresh` -> `flushSyncWork` loop never terminated, wedging the
    /// browser tab. This bundles two same-named components, RUNS the emitted dev
    /// bundle against a recording refresh runtime, and asserts the ids it registers
    /// are distinct and carry their own module's path.
    #[test]
    fn a_dev_client_bundle_scopes_every_fast_refresh_registration_to_its_module() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_jsx_runtime_package(root, "react");
        fs::write(
            root.join("list.jsx"),
            "export function Widget() { return <div>list</div>; }\n",
        )
        .unwrap();
        fs::write(
            root.join("table.jsx"),
            "export function Widget() { return <div>table</div>; }\n",
        )
        .unwrap();
        fs::write(
            root.join("entry.jsx"),
            "import { Widget as FromList } from './list.jsx';\n\
             import { Widget as FromTable } from './table.jsx';\n\
             (globalThis.__used ??= []).push(FromList, FromTable);\n",
        )
        .unwrap();
        let entry = root.join("entry.jsx");

        // The dev build: the bundler's own `hmr` flag is what turns the per-module
        // refresh instrumentation on (`build-app` never sets it).
        let dev_config = BuildConfig {
            hmr: true,
            target: Target::Client,
            ..BuildConfig::default()
        };
        let (bundler, update) = discover_direct_with_config(&entry, &dev_config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let dev_out = root.join("dev/out.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &dev_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    hmr: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();

        let dev_code = fs::read_to_string(&dev_out).unwrap();
        assert!(
            dev_code.contains("$RefreshReg$"),
            "the dev build must instrument its components: {dev_code}"
        );
        let harness = root.join("dev/harness.mjs");
        fs::write(
            &harness,
            "globalThis.window = globalThis;\n\
             const ids = [];\n\
             globalThis.$RefreshRuntime$ = {\n\
             register: (type, id) => ids.push(id),\n\
             createSignatureFunctionForTransform: () => (type) => type,\n\
             registerExportsForReactRefresh: () => {},\n\
             validateRefreshBoundaryAndEnqueueUpdate: () => undefined,\n\
             };\n\
             await import('./out.mjs');\n\
             console.log(JSON.stringify(ids));\n",
        )
        .unwrap();
        let registered: Vec<String> = serde_json::from_str(run_node(&harness).trim()).unwrap();
        assert_eq!(
            registered.len(),
            2,
            "both modules must register their component: {registered:?}"
        );
        assert_ne!(
            registered[0], registered[1],
            "same-named components in different modules must not share a family: {registered:?}"
        );
        for (module, id) in [("list.jsx", &registered[0]), ("table.jsx", &registered[1])] {
            assert!(
                id.contains(module) && id.ends_with(" Widget"),
                "the family id must be its own module plus the export name: {id}"
            );
        }

        // Production is untouched: no refresh instrumentation is emitted at all, so
        // none of this can reach a `build-app` bundle.
        let (production, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let production_out = root.join("dist/out.mjs");
        production
            .emit_with_options(
                &production.reachable_modules_direct(),
                &production_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let code = fs::read_to_string(&production_out).unwrap();
        assert!(
            !code.contains("$RefreshReg$") && !code.contains("$RefreshRuntime$"),
            "a production bundle must carry no Fast Refresh instrumentation: {code}"
        );
    }

    /// An imported binding must be initialized before ANY of the module's body runs,
    /// even a statement written ABOVE the import.
    ///
    /// `import` declarations are hoisted by the language: the spec instantiates and
    /// evaluates every requested module before the importer's body executes, so source
    /// position says nothing about when a binding becomes available. Babel's JSX-pragma
    /// output relies on it — `var __jsx = React.createElement;` is emitted above
    /// `import React from "react"` (next-i18next's `appWithTranslation.js` ships exactly
    /// that) — and lowering each import in place made the binding read `undefined`,
    /// failing with `TypeError: Cannot convert undefined or null to object` inside a
    /// render, on code that is perfectly valid ESM.
    #[test]
    fn an_import_binding_is_initialized_before_a_statement_written_above_it() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("dep.js"),
            "export default { make: () => 'made' };\nexport const named = 'named';\n",
        )
        .unwrap();
        // Babel's JSX-pragma shape verbatim: a body statement above the import.
        fs::write(
            directory.path().join("entry.js"),
            "const make = Dep.make;\n\
             import Dep, { named } from './dep.js';\n\
             console.log(make() + ':' + named);\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "made:named\n");
    }

    /// The same rule for a bare side-effect `import`: the requested module runs before
    /// the importer's body, not at the import statement's source position.
    #[test]
    fn a_side_effect_import_runs_before_the_body_that_precedes_it() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("effect.js"),
            "globalThis.__order = (globalThis.__order || '') + 'effect';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "globalThis.__order = (globalThis.__order || '') + 'body';\n\
             import './effect.js';\n\
             console.log(globalThis.__order);\n",
        )
        .unwrap();
        assert_eq!(bundle_and_run(directory.path()), "effectbody\n");
    }

    /// A specifier one module reaches BOTH statically and dynamically is not a
    /// code-split boundary, and moving it into a lazily-fetched chunk breaks the
    /// static reference.
    ///
    /// The shape is the ordinary lazy-component barrel:
    ///
    /// ```js
    /// export { default as Foo } from "./Foo";
    /// export const FooLazy = dynamic(() => import("./Foo"));
    /// ```
    ///
    /// Reading only the `import()` said "./Foo is a chunk root", so the module moved
    /// out of the entry's static closure — and then the `export … from` on the line
    /// above, which lowers to a synchronous registry lookup, threw
    /// `Module is not loaded: <id>` the first time the barrel evaluated.
    ///
    /// Node EXECUTING the bundle is the assertion; the chunk-count check pins that the
    /// build really is code-split (so the test cannot pass by not splitting at all).
    #[test]
    fn a_specifier_reached_both_statically_and_dynamically_is_not_split_off() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("shared.js"),
            "export const label = 'shared-label';\nexport default 'shared-default';\n",
        )
        .unwrap();
        // An unrelated module reached ONLY dynamically, so the build genuinely splits.
        fs::write(
            directory.path().join("only-lazy.js"),
            "export const only = 'only-lazy';\n",
        )
        .unwrap();
        // The barrel: a static re-export AND a dynamic import of the SAME specifier.
        fs::write(
            directory.path().join("barrel.js"),
            "export { label } from './shared.js';\n\
             export const lazyShared = () => import('./shared.js');\n\
             export const lazyOther = () => import('./only-lazy.js');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { label, lazyShared, lazyOther } from './barrel.js';\n\
             Promise.all([lazyShared(), lazyOther()]).then(([a, b]) =>\n\
             console.log(label + ':' + a.label + ':' + b.only));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert!(
            stats.written.len() >= 2,
            "the dynamic-only import must still split off a chunk: {:?}",
            stats.written
        );
        for path in &stats.written {
            node_check(path);
        }
        // The both-ways module belongs to the entry chunk; only the dynamic-ONLY one
        // may live in a split chunk.
        for path in &stats.written {
            if path == &esm_out {
                continue;
            }
            let chunk = fs::read_to_string(path).unwrap();
            assert!(
                !chunk.contains("shared-label"),
                "a statically-referenced module must not be moved into a lazy chunk: {}",
                path.display()
            );
        }
        assert_eq!(run_node(&esm_out), "shared-label:shared-label:only-lazy\n");
    }

    /// The same hole through a `require(...)`: a synchronous read of a module that is
    /// also `import()`ed elsewhere. `require` returns the exports immediately, so the
    /// target can never sit behind a chunk fetch.
    #[test]
    fn a_specifier_reached_by_require_and_by_dynamic_import_is_not_split_off() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("shared.js"),
            "exports.label = 'req-shared';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("only-lazy.js"),
            "export const only = 'only-lazy';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "export const eager = require('./shared.js').label;\n\
             export const lazy = () => import('./shared.js');\n\
             Promise.all([lazy(), import('./only-lazy.js')]).then(([a, b]) =>\n\
             console.log(eager + ':' + b.only));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        for path in &stats.written {
            node_check(path);
        }
        assert_eq!(run_node(&esm_out), "req-shared:only-lazy\n");
    }

    /// `export * from "./x"` where `./x` is tree-shaken away must not leave a
    /// runtime lookup for `./x` behind.
    ///
    /// The registry's miss path is how EXTERNALS work (`node:fs`, an uninstalled
    /// optional dependency), so a lookup for a module the bundle dropped does not
    /// fail the build — it becomes a raw `require("./x")` in the emitted file and
    /// throws MODULE_NOT_FOUND the moment the module is evaluated. `tslog` ships
    /// exactly this shape: an `interfaces.js` whose entire body is `export {}`,
    /// star-re-exported by its logger. Node EXECUTING the bundle is the assertion
    /// that matters; the byte check pins the cause.
    #[test]
    fn a_star_reexport_of_a_shaken_away_module_leaves_no_runtime_lookup() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        // `sideEffects: false` is what entitles dead-module elimination to drop a
        // module nothing demands an export of — the declaration every package that
        // ships this shape (tslog among them) makes.
        fs::write(
            directory.path().join("package.json"),
            r#"{"name":"star-reexport-fixture","sideEffects":false}"#,
        )
        .unwrap();
        // Type-only in spirit and empty in fact: nothing to export, no side
        // effects, so dead-module elimination is entitled to drop it entirely.
        fs::write(directory.path().join("interfaces.js"), "export {};\n").unwrap();
        fs::write(
            directory.path().join("logger.js"),
            "export * from './interfaces.js';\nexport const log = (message) => 'log:' + message;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { log } from './logger.js';\nconsole.log(log('ok'));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let out = directory.path().join("dist/out.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let code = fs::read_to_string(&out).unwrap();
        assert!(
            !code.contains("./interfaces.js"),
            "the dropped module must not be referenced by the emitted code: {code}"
        );
        assert_eq!(run_node(&out), "log:ok\n");
    }

    /// A bundle with no top-level `await` anywhere must be BYTE-IDENTICAL to what
    /// it was before async-module support: every async runtime line is gated on
    /// the build actually having one. Guards against the registry runtime growing
    /// dead weight (and against the async paths quietly turning on).
    #[test]
    fn a_build_without_top_level_await_emits_no_async_runtime() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import('./lazy.js').then(({ lazy }) => console.log(lazy));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, _) = discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let esm_out = directory.path().join("dist/out.mjs");
        let stats = bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        for path in &stats.written {
            let code = fs::read_to_string(path).unwrap();
            for marker in [
                "__pending",
                "__requireAsync",
                "require.esmAsync",
                "require.async",
                "async function(module,exports",
                "await (async()=>{",
            ] {
                assert!(
                    !code.contains(marker),
                    "{}: a build with no top-level await must not emit {marker}",
                    path.display()
                );
            }
        }
    }

    /// A CommonJS `require()` cannot wait for a module that top-level-`await`s
    /// (Node itself throws `ERR_REQUIRE_ASYNC_MODULE`), and neither can the lazy
    /// getter `export * as ns from` lowers to. Both must be hard errors naming
    /// BOTH modules, never a bundle that reads a half-initialised namespace.
    #[test]
    fn reaching_an_async_module_without_an_awaitable_import_is_a_hard_error() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';\n",
        )
        .unwrap();

        // (1) A CommonJS `require` of the awaiting module.
        fs::write(
            directory.path().join("entry.js"),
            "const { value } = require('./value.js');\n\
             console.log(value);\n\
             import('./lazy.js');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, _) = discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("dist/out.mjs"),
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap_err();
        assert!(error.contains("top-level await"), "{error}");
        assert!(
            error.contains("value.js"),
            "names the async module: {error}"
        );
        assert!(error.contains("entry.js"), "names the importer: {error}");
        assert!(
            error.contains("ERR_REQUIRE_ASYNC_MODULE"),
            "names Node's own diagnosis: {error}"
        );

        // (2) `export * as ns from` the awaiting module — a lazy getter.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("value.js"),
            "export const value = await Promise.resolve('tla-value');\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy';\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "export * as values from './value.js';\nimport('./lazy.js');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, _) = discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("dist/out.mjs"),
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap_err();
        assert!(error.contains("export * as"), "{error}");
        assert!(
            error.contains("value.js"),
            "names the async module: {error}"
        );
        assert!(error.contains("entry.js"), "names the importer: {error}");
    }

    /// `import.meta` is a syntax error anywhere in a CommonJS file, so CJS
    /// output must refuse; in ESM output it stays, resolving against the
    /// emitted chunk (the standard bundler semantic).
    #[test]
    fn import_meta_is_a_hard_error_in_cjs_and_survives_in_esm() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "console.log('url-kind:' + (import.meta.url.startsWith('file://')));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let error = bundler
            .emit_with_options(
                &reachable,
                &directory.path().join("dist/out.js"),
                EmitOptions::default(),
            )
            .unwrap_err();
        assert!(error.contains("import.meta"), "{error}");
        assert!(error.contains("entry.js"), "names the module: {error}");

        let esm_out = directory.path().join("dist/out.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &esm_out,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert_eq!(run_node(&esm_out), "url-kind:true\n");
    }

    /// Statement-level shaking must be TRANSITIVE: a pure helper (exported or
    /// not) referenced only by a dead export falls with it, through chains,
    /// while impure statements and everything they reference stay. Pinned by
    /// the realistic-corpus finding where non-exported helpers of dead exports
    /// made output 2.2x larger than esbuild's.
    #[test]
    fn shaking_drops_helpers_of_dead_exports_transitively() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lib.js"),
            "const DEEP_CONFIG = { step: 3 };\n\
             function deepHelper(value) { return value + DEEP_CONFIG.step; }\n\
             function midHelper(value) { return deepHelper(value) * 2; }\n\
             export function unusedTool(value) { return midHelper(value); }\n\
             const KEPT_BASE = 40;\n\
             export function usedTool(value) { return value + KEPT_BASE; }\n\
             console.log('lib-side-effect:' + usedTool(0));\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { usedTool } from './lib.js';\nconsole.log('result:' + usedTool(2));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let code = fs::read_to_string(&output).unwrap();
        for dead in ["unusedTool", "midHelper", "deepHelper", "DEEP_CONFIG"] {
            assert!(!code.contains(dead), "`{dead}` should be shaken:\n{code}");
        }
        for live in ["usedTool", "KEPT_BASE", "lib-side-effect"] {
            assert!(code.contains(live), "`{live}` must survive:\n{code}");
        }
        assert_eq!(run_node(&output), "lib-side-effect:40\nresult:42\n");
    }

    /// Vite's `assetsInlineLimit`: in Vite mode a small asset import yields a
    /// `data:` URI (no emitted file, no request); over the limit — or with the
    /// limit disabled (generic bundling) — it stays a hashed public file.
    #[test]
    fn small_assets_inline_as_data_uris_only_in_vite_mode() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("icon.svg"), "<svg xmlns='x'/>").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import icon from './icon.svg';\nconsole.log(icon.slice(0, 30));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");

        let inline_config = BuildConfig {
            asset_inline_limit: 4096,
            ..BuildConfig::default()
        };
        let (bundler, update) = discover_direct_with_config(&entry, &inline_config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        assert_eq!(run_node(&output), "data:image/svg+xml,%3csvg%20xm\n");
        assert!(
            !directory.path().join("dist/assets").exists(),
            "an inlined asset emits no file"
        );

        let (bundler, _) = discover_direct(&entry).unwrap();
        let reachable = bundler.reachable_modules_direct();
        let plain = directory.path().join("dist-plain/bundle.js");
        bundler
            .emit_with_options(&reachable, &plain, EmitOptions::default())
            .unwrap();
        assert!(
            run_node(&plain).starts_with("/assets/icon-"),
            "generic bundling keeps the hashed file URL"
        );
    }

    /// `new Worker(new URL('./x', import.meta.url))` bundles the worker entry
    /// as its own self-contained file under `assets/` and substitutes its
    /// public URL — shipping the raw specifier would 404 at runtime (found
    /// live on wall-go's minimax AI workers).
    #[test]
    fn module_workers_are_bundled_and_their_urls_substituted() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("logic.js"),
            "export function answer() { return 42; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("worker.js"),
            "import { answer } from './logic.js';\nself.postMessage(answer());\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });\nconsole.log('spawned:' + (w instanceof Object));\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::BrowserEsm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let code = fs::read_to_string(&output).unwrap();
        assert!(
            !code.contains("__diffpack_worker__"),
            "placeholder replaced: {code}"
        );
        assert!(!code.contains("./worker.js"), "raw specifier gone: {code}");
        let url_start = code
            .find("/assets/worker-")
            .expect("worker URL substituted");
        let url = code[url_start..].split(['"', '\'', '`']).next().unwrap();
        let emitted = directory
            .path()
            .join("dist")
            .join(url.trim_start_matches('/'));
        assert!(
            emitted.is_file(),
            "worker bundle emitted at {}",
            emitted.display()
        );
        let worker_code = fs::read_to_string(&emitted).unwrap();
        assert!(worker_code.contains("postMessage"), "{worker_code}");
        assert!(
            worker_code.contains("42"),
            "the worker's import is bundled in: {worker_code}"
        );
    }

    /// Side-effect imports must execute in IMPORT order, not module-id order.
    /// The entry imports `./bbb.js` before `./aaa.js`; alphabetical ordering
    /// would run `aaa` first, which is exactly the bug this pins down (the
    /// conformance suite's `order-side-effect-imports` finding).
    #[test]
    fn side_effect_imports_execute_in_import_order_not_id_order() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("aaa.js"), "console.log('aaa');\n").unwrap();
        fs::write(directory.path().join("bbb.js"), "console.log('bbb');\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './bbb.js';\nimport './aaa.js';\nconsole.log('entry');\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        assert_eq!(run_node(&output), "bbb\naaa\nentry\n");
    }

    /// The emitted stylesheet must follow the same execution order, because the
    /// CSS cascade breaks equal-specificity ties by document order: a rule from
    /// a stylesheet the entry imports FIRST must lose to a same-specificity rule
    /// imported later, no matter how the module paths sort. (Found live on the
    /// create-vite fixture: `App.css`'s `.counter` override lost to `index.css`
    /// because alphabetical order inverted the cascade.)
    #[test]
    fn extracted_css_follows_import_order_so_the_cascade_ties_break_correctly() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("zzz-base.css"),
            ".x { color: red; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("aaa-widget.css"),
            ".x { color: blue; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("aaa-widget.js"),
            "import './aaa-widget.css';\nexport const widget = 1;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import './zzz-base.css';\nimport { widget } from './aaa-widget.js';\nconsole.log(widget);\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_with_options(&reachable, &output, EmitOptions::default())
            .unwrap();
        let stylesheet = fs::read_to_string(directory.path().join("dist/bundle.css")).unwrap();
        let base_at = stylesheet.find("red").expect("base rule present");
        let widget_at = stylesheet.find("blue").expect("widget rule present");
        assert!(
            base_at < widget_at,
            "entry-imported stylesheet must precede the later component's:\n{stylesheet}"
        );
    }

    /// registry runtime), and a dynamic `import()` of a split chunk, emits it via
    /// the server path, then runs the entry under `node` and asserts both the
    /// static value and the dynamically-loaded chunk's value reach stdout.
    #[test]
    fn emit_server_mjs_executes_the_entry_and_dynamic_chunk_under_node() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("util.js"),
            "export const base = 10;\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "import os from 'node:os';\n\
             export const lazy = 'lazy-value';\n\
             export function describe(){ return typeof os.platform === 'function' ? 'has-os' : 'no-os'; }\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("server.ts"),
            "import path from 'node:path';\n\
             import { base } from './util.js';\n\
             console.log('base:' + base);\n\
             console.log('sep:' + (path.sep.length === 1));\n\
             import('./lazy.js').then((m) => { console.log('lazy:' + m.lazy + ':' + m.describe()); });\n",
        )
        .unwrap();

        let entry = directory.path().join("server.ts");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        let server_entry = output_root.join("server/server.mjs");
        assert!(
            output_root.join("server/server.chunk-1.mjs").is_file(),
            "the dynamic import lands in its own `.mjs` chunk"
        );
        // Actually run it: `module is not defined` would abort here, so a clean
        // stdout proves the emitted ESM genuinely executes.
        assert_eq!(
            run_node(&server_entry),
            "base:10\nsep:true\nlazy:lazy-value:has-os\n"
        );
    }

    /// A host that wants a FRESH module graph re-imports the entry under a new URL
    /// after dropping the runtime globals — the react-server `serve` worker's
    /// protocol. The registry lives on `globalThis`, so the new entry instance
    /// builds a new registry; every chunk it dynamically imports must therefore be
    /// a new instance too, or the chunk stays in Node's ESM cache, never re-runs
    /// its `__register`, and `__require` throws "Module is not loaded: <id>".
    #[test]
    fn a_fresh_entry_instance_gets_fresh_chunk_instances() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lazy.js"),
            "export const lazy = 'lazy-value';\n",
        )
        .unwrap();
        // The dynamic import fires during the entry's own evaluation, which is what
        // gets the chunk into the ESM cache before the re-import happens.
        fs::write(
            directory.path().join("server.ts"),
            "export const loaded = import('./lazy.js').then((m) => m.lazy);\n\
             export async function read(){ return await loaded; }\n",
        )
        .unwrap();

        let entry = directory.path().join("server.ts");
        let output_root = directory.path().join(".diffpack-output");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();
        assert!(
            output_root.join("server/server.chunk-1.mjs").is_file(),
            "the dynamic import must land in its own chunk for this test to mean anything"
        );

        let driver = directory.path().join("driver.mjs");
        fs::write(
            &driver,
            "import { pathToFileURL } from 'node:url';\n\
             const url = pathToFileURL(process.argv[2]).href;\n\
             const first = await import(url);\n\
             console.log('first:' + await (first.default || first).read());\n\
             for (const key of Object.keys(globalThis)) {\n\
               if (key.indexOf('__diffpack_runtime:') === 0) delete globalThis[key];\n\
             }\n\
             const second = await import(url + '?v=2');\n\
             console.log('second:' + await (second.default || second).read());\n",
        )
        .unwrap();
        let executed = node_command()
            .arg(&driver)
            .arg(output_root.join("server/server.mjs"))
            .output()
            .unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "first:lazy-value\nsecond:lazy-value\n"
        );
    }

    /// Polls `127.0.0.1:port` until it accepts a connection (or the attempts run
    /// out), then makes one `HTTP/1.0` GET and returns the full raw response.
    fn http_get_when_ready(port: u16, path: &str) -> String {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::time::Duration;
        let address = format!("127.0.0.1:{port}");
        for _ in 0..200 {
            if let Ok(mut stream) = TcpStream::connect(&address) {
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let request =
                    format!("GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
                stream.write_all(request.as_bytes()).unwrap();
                let mut response = Vec::new();
                stream.read_to_end(&mut response).unwrap();
                return String::from_utf8_lossy(&response).into_owned();
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("server on port {port} never accepted a connection");
    }

    /// The emitted `server/index.mjs` must BOOT under Node and serve: SSR through
    /// the app's fetch handler (resolved from `server.mjs`'s CJS-interop default
    /// export by `_ssr/ssr.mjs`), plus a hashed asset from the sibling `public/`
    /// directory. Node is the runtime oracle — the request round-trips over real
    /// TCP, exactly like the acceptance runner.
    #[test]
    fn emitted_index_mjs_boots_and_serves_ssr_and_static_under_node() {
        use std::process::Stdio;
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let server_dir = directory.path().join("server");
        let public_dir = directory.path().join("public");
        fs::create_dir_all(&server_dir).unwrap();
        fs::create_dir_all(&public_dir).unwrap();

        // A stand-in for the emitted server bundle: its default export mirrors the
        // real build's shape (`default.default.fetch`), so `_ssr/ssr.mjs` must peel
        // the interop layers to find the Web fetch handler.
        fs::write(
            server_dir.join("server.mjs"),
            "const fetch = async (request) => {\n\
             \tconst { pathname } = new URL(request.url);\n\
             \tif (pathname === '/hello') return new Response('SSR-BODY-OK', { status: 200, headers: { 'content-type': 'text/html' } });\n\
             \treturn new Response('missing', { status: 404, headers: { 'content-type': 'text/html' } });\n\
             };\n\
             export default { default: { fetch } };\n",
        )
        .unwrap();
        // The natively generated manifest module: a runtime-style default export
        // carrying the `tsrStartManifest` factory that `_ssr/router.mjs` unwraps.
        fs::write(
            server_dir.join("_tanstack-start-manifest_v.mjs"),
            "const tsrStartManifest = () => ({ routes: { __root__: { preloads: [] } } });\n\
             export default { tsrStartManifest };\n",
        )
        .unwrap();
        fs::write(public_dir.join("static.txt"), "STATIC-ASSET-OK").unwrap();

        diffpack_tanstack::runtime::write_server_entry(&server_dir, false).unwrap();
        assert!(server_dir.join("index.mjs").is_file());
        assert!(server_dir.join("_ssr/ssr.mjs").is_file());
        assert!(server_dir.join("_ssr/router.mjs").is_file());
        assert!(server_dir.join("_ssr/node-adapter.mjs").is_file());

        // Reserve a free port, then hand it to the booted server.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut child = node_command()
            .arg(server_dir.join("index.mjs"))
            .env("PORT", port.to_string())
            .env("HOST", "127.0.0.1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();

        let ssr = http_get_when_ready(port, "/hello");
        let asset = http_get_when_ready(port, "/static.txt");
        child.kill().ok();
        child.wait().ok();

        assert!(
            ssr.contains("200") && ssr.contains("SSR-BODY-OK"),
            "SSR response did not come from the handler: {ssr}"
        );
        assert!(
            asset.contains("200") && asset.contains("STATIC-ASSET-OK"),
            "static asset was not served from public/: {asset}"
        );
    }

    /// A minimal TanStack-style route app: a stub `@tanstack/react-router` (so no
    /// node_modules is needed), one route file with a split component, and an
    /// entry that imports it. Returns `(directory, entry, config)`.
    fn route_app_fixture() -> (tempfile::TempDir, PathBuf, BuildConfig) {
        let directory = tempdir().unwrap();
        let router_stub = directory.path().join("react-router.js");
        fs::write(
            &router_stub,
            "export const createFileRoute = () => (options) => options;\n\
             export const lazyRouteComponent = () => {};\n",
        )
        .unwrap();
        let routes = directory.path().join("routes");
        fs::create_dir(&routes).unwrap();
        fs::write(
            routes.join("foo.tsx"),
            "import { createFileRoute } from '@tanstack/react-router'\n\
             export const Route = createFileRoute('/foo')({\n  component: Foo,\n})\n\
             function Foo() {\n  return null\n}\n",
        )
        .unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(&entry, "import './routes/foo.tsx';\n").unwrap();

        let config = BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: vec![(
                "@tanstack/react-router".to_string(),
                router_stub.to_string_lossy().into_owned(),
            )],
            conditions: Vec::new(),
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            private_chunk_names: Vec::new(),
            target: Target::Server,
            server_external_packages: Vec::new(),
            source_policy: Arc::new(
                diffpack_default_loader::source_policy::NoSourceIntegrationPolicy,
            ),
            hmr: false,
            scss: diffpack_default_loader::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess::default(),
            jsx_extensions: diffpack_core::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: diffpack_core::transform::JsxConfig::default(),
            source_maps: false,
        };
        (directory, entry, config)
    }

    /// An app that imports ONE name (`publicValue`) from a `sideEffects:false`
    /// package whose other export wraps a value from a second `sideEffects:false`
    /// package in `createServerOnlyFn`. That second package (`@leaf/server`) is
    /// reachable only through the wrapper's reference to it — exactly the shape of
    /// the real `@tanstack/*` leak, where a bare-specifier `sideEffects:false`
    /// package carries the server-only `node:async_hooks` code. Returns
    /// `(directory, entry)`.
    fn server_leak_fixture() -> (tempfile::TempDir, PathBuf) {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(
            root.join("package.json"),
            r#"{"name":"leak-app","version":"0.0.0"}"#,
        )
        .unwrap();
        let package = |name: &str, module_source: &str| {
            let dir = root.join("node_modules").join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("package.json"),
                format!(
                    r#"{{"name":"{name}","version":"0.0.0","module":"index.js","sideEffects":false}}"#
                ),
            )
            .unwrap();
            fs::write(dir.join("index.js"), module_source).unwrap();
        };
        // The directive-helper stub.
        package(
            "@tanstack/start-fn-stubs",
            "export const createServerOnlyFn = (fn) => fn;\n",
        );
        // The server-only leaf package (stands in for start-storage-context).
        package(
            "@leaf/server",
            "export const serverThing = \"SERVER_ONLY_MARKER_9271\";\n",
        );
        // The `sideEffects:false` barrel importing one name from each.
        package(
            "@tanstack/core",
            "import { createServerOnlyFn } from \"@tanstack/start-fn-stubs\";\n\
             import { serverThing } from \"@leaf/server\";\n\
             export const getServerThing = createServerOnlyFn(() => serverThing);\n\
             export const publicValue = 42;\n",
        );
        let entry = root.join("entry.js");
        fs::write(
            &entry,
            "import { publicValue } from \"@tanstack/core\";\nconsole.log(publicValue);\n",
        )
        .unwrap();
        (directory, entry)
    }

    #[test]
    fn client_build_drops_server_only_package_reached_through_neutralized_wrapper() {
        let (_directory, entry) = server_leak_fixture();
        let config = |target| BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: Vec::new(),
            conditions: Vec::new(),
            main_fields: Vec::new(),
            virtual_modules: Vec::new(),
            private_chunk_names: Vec::new(),
            target,
            server_external_packages: Vec::new(),
            source_policy: Arc::new(
                diffpack_default_loader::source_policy::NoSourceIntegrationPolicy,
            ),
            hmr: false,
            scss: diffpack_default_loader::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess::default(),
            jsx_extensions: diffpack_core::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: diffpack_core::transform::JsxConfig::default(),
            source_maps: false,
        };

        // Client: `createServerOnlyFn(() => serverThing)` is neutralized to a
        // throwing stub, so `@leaf/server` is unreferenced and pruned by the
        // `sideEffects:false` shaking — the leaf never enters the client graph.
        let (client, _) = discover_tanstack_with_config(&entry, &config(Target::Client)).unwrap();
        let client_reachable = client.reachable_modules_direct();
        assert!(
            !client_reachable
                .iter()
                .any(|module| module.contains("@leaf/server")),
            "the server-only package must not be reachable in the client build: {client_reachable:?}"
        );

        // Server: no transform, the wrapper keeps its reference, so the leaf stays.
        let (server, _) = discover_tanstack_with_config(&entry, &config(Target::Server)).unwrap();
        let server_reachable = server.reachable_modules_direct();
        assert!(
            server_reachable
                .iter()
                .any(|module| module.contains("@leaf/server")),
            "the server-only package must remain reachable in the server build: {server_reachable:?}"
        );
    }

    #[test]
    fn client_route_manifest_attributes_split_chunks_to_route_ids() {
        let (_directory, entry, config) = route_app_fixture();
        let (bundler, update) = discover_tanstack_with_config(&entry, &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();

        let manifest = diffpack_tanstack::manifest::from_bundle_graph(
            &bundler
                .integration_manifest_graph(&reachable, "client.js")
                .unwrap(),
            "/",
        )
        .unwrap();
        // The root route maps to the entry chunk (which statically bundles it).
        assert_eq!(
            manifest
                .routes
                .get(diffpack_tanstack::manifest::ROOT_ROUTE_ID),
            Some(&vec!["client.js".to_string()])
        );
        // The route's split component becomes a dynamic chunk attributed to its
        // TanStack route id.
        let foo = manifest.routes.get("/foo").expect("route /foo is mapped");
        assert_eq!(foo.len(), 1, "one split chunk for /foo: {foo:?}");
        assert!(foo[0].starts_with("client.chunk-"), "{foo:?}");

        // The generated manifest source is the exact contract the server consumes.
        let source = manifest.to_start_manifest_source();
        assert!(
            source.contains(
                "const tsrStartManifest = () => ({ clientEntry: \"/client.js\", routes: {"
            ),
            "{source}"
        );
        assert!(
            source.contains(&format!("\"/foo\": {{ preloads: [\"/{}\"] }}", foo[0])),
            "{source}"
        );
    }

    #[test]
    fn a_registered_virtual_module_resolves_loads_and_names_its_chunk() {
        let directory = tempdir().unwrap();
        let entry = directory.path().join("server.ts");
        fs::write(
            &entry,
            "import('tanstack-start-manifest:v').then(({ tsrStartManifest }) => \
             console.log(tsrStartManifest()));\n",
        )
        .unwrap();

        let source =
            "const tsrStartManifest = () => ({ routes: {} });\nexport { tsrStartManifest };\n";
        let config = BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            aliases: Vec::new(),
            conditions: Vec::new(),
            main_fields: Vec::new(),
            virtual_modules: vec![(
                diffpack_tanstack::manifest::START_MANIFEST_SPECIFIER.to_string(),
                source.to_string(),
            )],
            private_chunk_names: vec![(
                diffpack_tanstack::manifest::START_MANIFEST_SPECIFIER.to_string(),
                "_tanstack-start-manifest_v{ext}".to_string(),
            )],
            target: Target::Server,
            server_external_packages: Vec::new(),
            source_policy: Arc::new(
                diffpack_default_loader::source_policy::NoSourceIntegrationPolicy,
            ),
            hmr: false,
            scss: diffpack_default_loader::sass::ScssOptions::default(),
            image_import_shape: ImageImportShape::Url,
            css_preprocess: CssPreprocess::default(),
            jsx_extensions: diffpack_core::parser::JsxExtensions::JsxAndTsxOnly,
            jsx: diffpack_core::transform::JsxConfig::default(),
            source_maps: false,
        };
        let (bundler, update) = discover_direct_with_config(&entry, &config).unwrap();
        // The previously-unresolvable specifier now resolves and loads: no gap.
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        assert!(
            bundler
                .all_modules()
                .contains(diffpack_tanstack::manifest::START_MANIFEST_SPECIFIER),
            "the virtual module is in the graph"
        );

        let reachable = bundler.reachable_modules_direct();
        let output_root = directory.path().join(".diffpack-output");
        bundler
            .emit_server(&reachable, &output_root, EmitOptions::default())
            .unwrap();

        // The manifest lands in its own descriptively named server chunk (the
        // acceptance gate matches server files containing `tanstack-start-manifest`).
        let manifest_chunk = output_root.join("server/_tanstack-start-manifest_v.mjs");
        assert!(manifest_chunk.is_file(), "manifest chunk is emitted");
        let emitted = fs::read_to_string(&manifest_chunk).unwrap();
        assert!(emitted.contains("tsrStartManifest"), "{emitted}");
        node_check(&manifest_chunk);
    }

    /// Writes a `sideEffects`-annotated package under `<root>/node_modules/<name>`.
    /// `files` is `(relative path, source)`; `side_effects` is the raw JSON value
    /// of the `package.json` `sideEffects` field (e.g. `"false"`, `"true"`,
    /// `r#"["*.css"]"#`).
    fn write_package(root: &Path, name: &str, side_effects: &str, files: &[(&str, &str)]) {
        let package = root.join("node_modules").join(name);
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            format!(
                "{{ \"name\": \"{name}\", \"version\": \"1.0.0\", \"main\": \"index.js\", \
                 \"sideEffects\": {side_effects} }}"
            ),
        )
        .unwrap();
        for (relative, source) in files {
            let path = package.join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, source).unwrap();
        }
    }

    #[test]
    fn dce_drops_a_barrel_reexported_module_no_live_module_uses() {
        // A `sideEffects:false` package whose barrel re-exports two modules; the
        // app uses only one. The unused re-exported module — and the
        // side-effectful module it pulls (which imports a Node built-in) — must be
        // dropped, exactly as Rollup/esbuild would.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package(
            root,
            "lib",
            "false",
            &[
                (
                    "index.js",
                    "export { used } from './used.js';\nexport { unused } from './unused.js';\n",
                ),
                ("used.js", "export const used = 'USED';\n"),
                (
                    "unused.js",
                    "import { AsyncLocalStorage } from 'node:async_hooks';\n\
                     const store = new AsyncLocalStorage();\n\
                     export const unused = store;\n",
                ),
            ],
        );
        fs::write(root.join("package.json"), r#"{ "name": "app" }"#).unwrap();
        let entry = root.join("entry.js");
        fs::write(&entry, "import { used } from 'lib';\nconsole.log(used);\n").unwrap();

        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let live = bundler.live_modules(&reachable);

        let contains =
            |set: &BTreeSet<String>, suffix: &str| set.iter().any(|id| id.ends_with(suffix));
        // The barrel is reachable AND remains reachable, but `unused.js` is dead.
        assert!(
            contains(&reachable, "lib/unused.js"),
            "reachable set: {reachable:?}"
        );
        assert!(
            !contains(&live, "lib/unused.js"),
            "the barrel-only, unused re-export must be dropped: {live:?}"
        );
        assert!(
            contains(&live, "lib/used.js"),
            "the used export must be kept: {live:?}"
        );
        assert!(
            contains(&live, "lib/index.js"),
            "the live barrel is kept: {live:?}"
        );

        // Emit and confirm the Node built-in the dead module pulled never ships.
        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let bundle = fs::read_to_string(&output).unwrap();
        assert!(
            !bundle.contains("node:async_hooks"),
            "the dropped module's Node built-in must not ship: {bundle}"
        );
        assert!(
            bundle.contains("USED"),
            "the used export must ship: {bundle}"
        );
        node_check(&output);
    }

    /// A module reached ONLY through a CommonJS `require()` must survive dead-module
    /// elimination.
    ///
    /// `sideEffects: false` authorizes dropping a module nothing demands, and demand was
    /// collected from `import` declarations alone — so a `require()`d module carried no
    /// demand whatsoever and was deleted. The `require` CALL survived, found nothing in
    /// the registry, and fell through to the external path: `MODULE_NOT_FOUND` under
    /// Node, and in the browser `Cannot require "…": it is not a Node built-in and was
    /// not included in the bundle`. That is exactly what killed hydration on every
    /// cal.com page, through
    /// `const { i18n } = require("@calcom/i18n/next-i18next.config")` in a
    /// `"sideEffects": false` workspace package.
    ///
    /// `require()` yields the whole `module.exports`, so the demand it places is the
    /// full namespace — there is no named subset to narrow it to.
    #[test]
    fn dce_keeps_a_module_reached_only_through_a_commonjs_require() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package(
            root,
            "config-pkg",
            "false",
            &[
                ("index.js", "module.exports = { unrelated: true };\n"),
                (
                    "settings.js",
                    "module.exports = { locales: ['en', 'fr'] };\n",
                ),
            ],
        );
        fs::write(root.join("package.json"), r#"{ "name": "app" }"#).unwrap();
        // The require sits in a module that ALSO has ESM structure, so the liveness
        // record is non-empty and the conservative "no captured structure" path (which
        // keeps every dependency) cannot be what saves it.
        fs::write(
            root.join("lib.js"),
            "const settings = require('config-pkg/settings.js');\n\
             export const locales = settings.locales;\n",
        )
        .unwrap();
        let entry = root.join("entry.js");
        fs::write(
            &entry,
            "import { locales } from './lib.js';\nconsole.log('locales:' + locales.join(','));\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let live = bundler.live_modules(&reachable);
        assert!(
            live.iter().any(|id| id.ends_with("config-pkg/settings.js")),
            "a require()d module must stay live even under sideEffects:false: {live:?}"
        );
        // The unrelated entry point of the same package is still droppable — the fix
        // must not degrade into "keep the whole package".
        assert!(
            !live.iter().any(|id| id.ends_with("config-pkg/index.js")),
            "only what is actually required is kept: {live:?}"
        );

        // Executing is the real assertion: the emitted `require` must find its target
        // in the registry rather than falling through to the host.
        let output = root.join("dist/bundle.mjs");
        bundler
            .emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    format: ModuleFormat::Esm,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        assert_eq!(run_node(&output), "locales:en,fr\n");
    }

    #[test]
    fn dce_keeps_a_side_effectful_module_and_a_used_module() {
        // Two packages: one `sideEffects:true` (its module runs for effect even if
        // nothing is imported from it) and one `sideEffects:false` whose export IS
        // used. Both must be kept.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package(
            root,
            "effectful",
            "true",
            &[("index.js", "globalThis.__EFFECT__ = true;\n")],
        );
        write_package(
            root,
            "pure",
            "false",
            &[("index.js", "export const value = 'PURE';\n")],
        );
        fs::write(root.join("package.json"), r#"{ "name": "app" }"#).unwrap();
        let entry = root.join("entry.js");
        fs::write(
            &entry,
            "import 'effectful';\nimport { value } from 'pure';\nconsole.log(value);\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let live = bundler.live_modules(&reachable);

        let contains =
            |set: &BTreeSet<String>, suffix: &str| set.iter().any(|id| id.ends_with(suffix));
        assert!(
            contains(&live, "effectful/index.js"),
            "a bare `import 'effectful'` of a sideEffects:true module must be kept: {live:?}"
        );
        assert!(
            contains(&live, "pure/index.js"),
            "a used sideEffects:false module must be kept: {live:?}"
        );
    }

    #[test]
    fn dce_drops_a_bare_side_effect_import_of_a_side_effect_free_module() {
        // `import './noop.js'` for effect, but `./noop.js`'s package declares
        // `sideEffects:false`, so the flag authorizes dropping the module (and its
        // Node-built-in import) entirely — matching Rollup/esbuild.
        let directory = tempdir().unwrap();
        let root = directory.path();
        write_package(
            root,
            "quiet",
            "false",
            &[
                ("index.js", "export const marker = 'QUIET';\n"),
                (
                    "noop.js",
                    "import { readFileSync } from 'node:fs';\nexport const noop = readFileSync;\n",
                ),
            ],
        );
        fs::write(root.join("package.json"), r#"{ "name": "app" }"#).unwrap();
        let entry = root.join("entry.js");
        // Import the package's `noop.js` purely for side effect.
        fs::write(&entry, "import 'quiet/noop.js';\nconsole.log('app');\n").unwrap();

        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let live = bundler.live_modules(&reachable);
        let contains =
            |set: &BTreeSet<String>, suffix: &str| set.iter().any(|id| id.ends_with(suffix));
        assert!(
            !contains(&live, "quiet/noop.js"),
            "a bare side-effect import of a sideEffects:false module must be droppable: {live:?}"
        );

        let output = root.join("dist/bundle.js");
        bundler.emit(&reachable, &output).unwrap();
        let bundle = fs::read_to_string(&output).unwrap();
        assert!(
            !bundle.contains("node:fs"),
            "the dropped side-effect module's Node built-in must not ship: {bundle}"
        );
        node_check(&output);
    }

    /// A build that opts into Vite conventions for `import.meta.glob`, rooted at
    /// `root` (the gate `config::derive_web_config --vite` and `build-app` set).
    fn glob_config(root: &Path) -> BuildConfig {
        BuildConfig {
            base: "/".to_string(),
            browser_process_shim: false,
            asset_inline_limit: 0,
            source_policy: Arc::new(diffpack_vite_compat::source_policy::ViteSourcePolicy {
                import_meta_glob: Some(diffpack_vite_compat::import_meta_glob::ImportMetaGlob {
                    root: root.canonicalize().unwrap(),
                }),
                ..Default::default()
            }),
            ..BuildConfig::default()
        }
    }

    /// A module reached BOTH by a static named import AND by a dynamic `import()` must
    /// have its WHOLE namespace demanded: `import()` hands its consumer the entire
    /// namespace and nothing here analyses what the consumer then reads off it.
    ///
    /// Such a module never becomes a chunk root (it is already in the entry's static
    /// closure), so it never receives the `all = true` that `render_runtime` gives a root,
    /// and it used to keep only the static importer's names. That is how the RSC control
    /// boundary lost its `default` export in a dev build — the generated browser entry
    /// imports `{ isControlFlowError }` from it statically and `import()`s it in a
    /// never-called island-pin thunk — after which the flight's client reference for it
    /// resolved to `undefined` and hydration died with "Element type is invalid. Received
    /// a promise that resolves to: undefined".
    ///
    /// This asserts the DEMAND RULE directly. The end-to-end symptom needs the RSC seam
    /// resolving a client reference by runtime id, which only the cal.com dev corpus
    /// exercises; a two-module fixture emits both exports for unrelated reasons and would
    /// pass either way.
    #[test]
    fn a_dynamic_import_demands_its_targets_whole_namespace() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::write(
            root.join("boundary.js"),
            "export function named(){return 1;}\nexport default function Boundary(){return 2;}\n",
        )
        .unwrap();
        // The shape of the generated browser entry: one named import, plus a never-called
        // thunk holding the dynamic import that pins the module into the graph.
        std::fs::write(
            root.join("entry.js"),
            "import { named } from \"./boundary.js\";\n\
             const pins = [() => import(\"./boundary.js\")];\n\
             globalThis.pins = pins;\n\
             globalThis.named = named;\n",
        )
        .unwrap();
        let (bundler, update) = discover_direct(&root.join("entry.js")).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let dense: Vec<DenseModuleId> = reachable
            .iter()
            .map(|id| bundler.graph.indices[id.as_str()])
            .collect();
        let demands = bundler.export_demands(&dense);
        let boundary = dense
            .iter()
            .copied()
            .find(|&d| bundler.graph.ids[d].ends_with("boundary.js"))
            .expect("the boundary module is reachable");
        assert!(
            demands[boundary].all,
            "the dynamic import must demand the whole namespace, not just the static \
             importer's names (it kept {:?})",
            demands[boundary].names,
        );
    }

    /// A split chunk loaded BEFORE the chunk that builds the runtime must register itself
    /// anyway, not throw.
    ///
    /// The document loads the entry and this route's chunks as separate scripts and nothing
    /// in HTML orders them (react-dom even marks its bootstrap tag `async`). While an early
    /// chunk threw "Diffpack runtime is not initialized", document order was load-bearing —
    /// and it broke three separate ways on cal.com, each time as "no page hydrates". The
    /// chunk now queues and the runtime drains the queue before the entry evaluates, which
    /// is the same shape webpack's `webpackChunk.push` gives it.
    ///
    /// Executed under node in BOTH orders, because the whole point is that neither is
    /// special.
    #[test]
    fn a_chunk_loaded_before_the_runtime_registers_instead_of_throwing() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::write(
            root.join("island.js"),
            "export const island = \"island-value\";\n",
        )
        .unwrap();
        std::fs::write(
            root.join("entry.js"),
            "globalThis.pin = [() => import(\"./island.js\")];\n\
             globalThis.loadIsland = async () => (await globalThis.pin[0]()).island;\n",
        )
        .unwrap();
        let (bundler, update) = discover_direct(&root.join("entry.js")).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let output_root = root.join(".out");
        let reachable = bundler.reachable_modules_direct();
        bundler
            .emit_public(&reachable, &output_root, EmitOptions::default())
            .unwrap();
        let public_dir = output_root.join("public");
        let chunk = fs::read_dir(&public_dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .find(|name| name.starts_with("client.chunk-"))
            .expect("the dynamic import produced a chunk");

        for (label, first, second) in [
            ("chunk first", chunk.as_str(), "client.js"),
            ("entry first", "client.js", chunk.as_str()),
        ] {
            let driver = public_dir.join(format!("drive-{}.mjs", label.replace(' ', "-")));
            fs::write(
                &driver,
                format!(
                    "import \"./{first}\";\nimport \"./{second}\";\n\
                     const value = await globalThis.loadIsland();\n\
                     if (value !== \"island-value\") throw new Error(\"got \" + value);\n\
                     console.log(\"ok\");\n"
                ),
            )
            .unwrap();
            let out = node_command().arg(&driver).output().unwrap();
            assert!(
                out.status.success() && String::from_utf8_lossy(&out.stdout).contains("ok"),
                "{label}: loading the chunk before the runtime must still register it\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }
    }

    /// The chunk-id -> URL table the RSC seam installs is built from EVERY split chunk,
    /// not just the ones that own a dynamic-import root.
    ///
    /// It used to be built from `chunk_names` (root -> chunk), which by construction skips
    /// a shared chunk extracted out of several roots' closures. The client-references
    /// manifest, meanwhile, names whichever chunk carries a client module's factory —
    /// shared or not — and the browser calls `__webpack_chunk_load__` with that id. So a
    /// client reference living in a shared chunk died with "__webpack_chunk_load__: unknown
    /// chunk id client.shared-16.js", and on cal.com that meant NO page hydrated once
    /// islands were split per chunk.
    #[test]
    fn the_emit_plan_records_every_chunk_file_including_shared_ones() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        // `shared.js` is reachable from BOTH dynamic roots, so the planner extracts it into
        // a chunk that owns no root — the shape `chunk_names` cannot describe.
        std::fs::write(root.join("shared.js"), "export const shared = 1;\n").unwrap();
        for name in ["a.js", "b.js"] {
            std::fs::write(
                root.join(name),
                "import { shared } from \"./shared.js\";\nexport default () => shared;\n",
            )
            .unwrap();
        }
        std::fs::write(
            root.join("entry.js"),
            "globalThis.load = [() => import(\"./a.js\"), () => import(\"./b.js\")];\n",
        )
        .unwrap();
        let (bundler, update) = discover_direct(&root.join("entry.js")).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        let reachable_dense: Vec<DenseModuleId> = reachable
            .iter()
            .map(|id| bundler.graph.indices[id.as_str()])
            .collect();
        let allowed: HashSet<DenseModuleId> = reachable_dense.iter().copied().collect();
        let plans = bundler.chunk_plan(&allowed, "client.js").unwrap();
        let shared_count = plans
            .iter()
            .filter(|plan| plan.file_name.contains(".shared-"))
            .count();
        assert!(
            shared_count > 0,
            "the fixture must produce a shared chunk: {plans:?}"
        );

        let plan = bundler.build_emit_plan(reachable_dense, allowed, &plans);
        assert_eq!(
            plan.chunk_files.len(),
            plans.len(),
            "every chunk's file name is recorded, shared ones included",
        );
        // The old source of the table, for contrast: it cannot see a rootless chunk.
        let root_named = chunk_names(&plans);
        assert!(
            plan.chunk_files
                .iter()
                .filter(|file| file.contains(".shared-"))
                .count()
                >= root_named
                    .values()
                    .filter(|file| file.contains(".shared-"))
                    .count(),
            "the recorded set must cover at least what root -> chunk covers",
        );
    }

    fn emitted_chunk_names(dist: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dist)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with("bundle.chunk-"))
            .collect();
        names.sort();
        names
    }

    #[test]
    fn import_meta_glob_lazy_matches_load_from_their_own_chunks_in_sorted_key_order() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let widgets = directory.path().join("widgets");
        fs::create_dir_all(&widgets).unwrap();
        // Written in reverse name order so sorted keys are the transform's doing.
        fs::write(widgets.join("beta.js"), "export const name = 'beta';\n").unwrap();
        fs::write(widgets.join("alpha.js"), "export const name = 'alpha';\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const modules = import.meta.glob('./widgets/*.js');\n\
             console.log(JSON.stringify(Object.keys(modules)));\n\
             Promise.all(Object.entries(modules).map(async ([key, load]) => `${key}=${(await load()).name}`))\n\
               .then((loaded) => console.log(loaded.join(',')));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) =
            discover_direct_with_config(&entry, &glob_config(directory.path())).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        // Each lazy match is its own dynamic-import graph edge, so its own chunk.
        let chunks = emitted_chunk_names(&directory.path().join("dist"));
        assert_eq!(chunks.len(), 2, "one chunk per lazy match: {chunks:?}");

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "[\"./widgets/alpha.js\",\"./widgets/beta.js\"]\n\
             ./widgets/alpha.js=alpha,./widgets/beta.js=beta\n"
        );
    }

    #[test]
    fn import_meta_glob_eager_with_default_import_binds_values_statically() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let widgets = directory.path().join("widgets");
        fs::create_dir_all(&widgets).unwrap();
        fs::write(widgets.join("alpha.js"), "export default 'A';\n").unwrap();
        fs::write(widgets.join("beta.js"), "export default 'B';\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const modules = import.meta.glob('./widgets/*.js', { eager: true, import: 'default' });\n\
             console.log(modules['./widgets/alpha.js'], modules['./widgets/beta.js']);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) =
            discover_direct_with_config(&entry, &glob_config(directory.path())).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        // Eager matches are static imports: everything lands in the entry chunk.
        let chunks = emitted_chunk_names(&directory.path().join("dist"));
        assert!(
            chunks.is_empty(),
            "eager glob must not split chunks: {chunks:?}"
        );

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&executed.stdout), "A B\n");
    }

    #[test]
    fn import_meta_glob_raw_query_routes_matches_through_the_raw_loader() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        let notes = directory.path().join("notes");
        fs::create_dir_all(&notes).unwrap();
        fs::write(notes.join("hello.txt"), "hello from glob raw").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const files = import.meta.glob('./notes/*.txt', { eager: true, import: 'default', query: '?raw' });\n\
             console.log(files['./notes/hello.txt']);\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) =
            discover_direct_with_config(&entry, &glob_config(directory.path())).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "hello from glob raw\n"
        );
    }

    #[test]
    fn import_meta_glob_pattern_array_unions_and_negative_pattern_excludes() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let directory = tempdir().unwrap();
        fs::create_dir_all(directory.path().join("a")).unwrap();
        fs::create_dir_all(directory.path().join("b")).unwrap();
        fs::write(directory.path().join("a/one.js"), "export const v = 1;\n").unwrap();
        fs::write(directory.path().join("a/skip.js"), "export const v = 0;\n").unwrap();
        fs::write(directory.path().join("b/two.js"), "export const v = 2;\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "const modules = import.meta.glob(['./a/*.js', './b/*.js', '!**/skip.js']);\n\
             console.log(JSON.stringify(Object.keys(modules)));\n",
        )
        .unwrap();

        let entry = directory.path().join("entry.js");
        let output = directory.path().join("dist/bundle.js");
        let (bundler, update) =
            discover_direct_with_config(&entry, &glob_config(directory.path())).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        bundler.emit(&reachable, &output).unwrap();

        let executed = node_command().arg(&output).output().unwrap();
        assert!(
            executed.status.success(),
            "{}",
            String::from_utf8_lossy(&executed.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&executed.stdout),
            "[\"./a/one.js\",\"./b/two.js\"]\n"
        );
    }

    #[test]
    fn without_the_vite_opt_in_import_meta_glob_is_left_untouched() {
        let directory = tempdir().unwrap();
        let widgets = directory.path().join("widgets");
        fs::create_dir_all(&widgets).unwrap();
        fs::write(widgets.join("alpha.js"), "export const name = 'alpha';\n").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "export const modules = import.meta.glob('./widgets/*.js');\n",
        )
        .unwrap();

        // No `import_meta_glob` in the config: generic bundling. The call must
        // survive to the module (no expansion, no graph edges), so the existing
        // import.meta-in-CommonJS honesty check refuses the CJS emit by name.
        let entry = directory.path().join("entry.js");
        let (bundler, update) = discover_direct(&entry).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(
            reachable.len(),
            1,
            "no glob edges without the opt-in: {reachable:?}"
        );
        let error = bundler
            .emit(&reachable, &directory.path().join("dist/bundle.js"))
            .unwrap_err();
        assert!(error.contains("import.meta"), "{error}");
        assert!(error.contains("entry.js"), "{error}");
    }

    #[test]
    fn asset_variant_public_name_appends_width_before_ext() {
        assert_eq!(
            asset_variant_public_name("shot-1a2b3c4d.png", 640),
            "shot-1a2b3c4d-640.png"
        );
        assert_eq!(asset_variant_public_name("noext", 32), "noext-32");
    }

    #[test]
    fn blur_data_url_is_a_tiny_decodable_data_uri() {
        let img = image::DynamicImage::new_rgb8(200, 100);
        let png = generate_blur_data_url(&img, "png").unwrap();
        assert!(png.starts_with("data:image/png;base64,"), "{png}");
        let jpeg = generate_blur_data_url(&img, "jpeg").unwrap();
        assert!(jpeg.starts_with("data:image/jpeg;base64,"), "{jpeg}");
        // A real payload but small (~8px-wide downscale), never a heavy full image.
        assert!(
            png.len() > 40 && png.len() < 4000,
            "tiny but real: {}",
            png.len()
        );
    }

    #[test]
    fn next_object_image_import_differs_from_vite_url_shape() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shot.png");
        image::DynamicImage::new_rgb8(300, 200).save(&path).unwrap();

        // NextObject: the module plans responsive variants to emit (object shape).
        let obj = synthesize_asset_url(
            path.clone(),
            "/",
            0,
            ImageImportShape::NextObject {
                responsive_variants: true,
            },
        )
        .unwrap();
        assert_eq!(obj.assets.len(), 1, "one emitted original");
        let variants = obj.assets[0]
            .image_variants
            .as_ref()
            .expect("NextObject plans responsive variants");
        assert_eq!(variants, &diffpack_next::next_adapter::variant_widths(300));
        assert!(
            variants.len() >= 2,
            "several responsive widths: {variants:?}"
        );
        assert!(
            obj.code.contains("variants"),
            "the object carries its ladder: {}",
            obj.code
        );

        // NextObject with optimization off (next.config `images.unoptimized` / a custom
        // loader): the SAME object shape — Next's static import always carries
        // src/width/height/blurDataURL — but no ladder is planned and the `variants`
        // key is OMITTED, which is what makes the shim render a raw <img src>. An empty
        // `{}` would be truthy and would silently keep the srcset path alive.
        let unopt = synthesize_asset_url(
            path.clone(),
            "/",
            0,
            ImageImportShape::NextObject {
                responsive_variants: false,
            },
        )
        .unwrap();
        assert_eq!(unopt.assets.len(), 1, "the original is still emitted");
        assert!(
            unopt.assets[0].image_variants.is_none(),
            "no variant file is planned when optimization is off",
        );
        assert!(
            unopt.code.contains("blurDataURL"),
            "blur is still generated: {}",
            unopt.code
        );
        assert!(
            unopt.code.contains("width"),
            "intrinsic size is still carried: {}",
            unopt.code
        );
        assert!(
            !unopt.code.contains("variants"),
            "the `variants` key is omitted, not emptied: {}",
            unopt.code,
        );

        // Url (Vite/TanStack/generic): bare URL string, NO variants planned. This
        // locks the no-regression guarantee for every non-Next build path.
        let url = synthesize_asset_url(path.clone(), "/", 0, ImageImportShape::Url).unwrap();
        assert_eq!(url.assets.len(), 1);
        assert!(
            url.assets[0].image_variants.is_none(),
            "Url mode stays bare-URL (Vite parity): no variants"
        );
    }

    // --- diagnostic fatality ------------------------------------------------
    //
    // The predicate that decides whether a build fails. It is structural (the
    // diagnostic's kind), not a substring match, so a new diagnostic kind has to
    // state its own fatality rather than inherit someone else's.

    #[test]
    fn an_unresolved_import_is_a_fatal_diagnostic() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { helper } from './does-not-exist.js';\nexport const value = helper;\n",
        )
        .unwrap();

        let (_, update) = discover_direct(&directory.path().join("entry.js")).unwrap();
        assert_eq!(update.diagnostics.len(), 1, "{:?}", update.diagnostics);
        let diagnostic = &update.diagnostics[0];
        assert!(matches!(
            &diagnostic.kind,
            DiagnosticKind::UnresolvedImport { specifier, .. }
                if specifier == "./does-not-exist.js"
        ));
        assert!(diagnostic.is_fatal());
        // The message must be actionable: it names the specifier, the importing
        // file, and (for a relative path) that no file matched.
        assert!(diagnostic.message.contains("./does-not-exist.js"));
        assert!(diagnostic.message.contains("entry.js"));
        assert!(diagnostic.message.contains("no file matched"));

        let error = partition_diagnostics(&update.diagnostics, "test build").unwrap_err();
        assert!(error.contains("test build"), "{error}");
        assert!(error.contains("./does-not-exist.js"), "{error}");
    }

    #[test]
    fn an_unresolved_bare_package_suggests_installing_it() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import x from '@scope/missing-pkg/sub';\nexport const value = x;\n",
        )
        .unwrap();

        let (_, update) = discover_direct(&directory.path().join("entry.js")).unwrap();
        assert_eq!(update.diagnostics.len(), 1, "{:?}", update.diagnostics);
        assert!(
            update.diagnostics[0]
                .message
                .contains("npm install @scope/missing-pkg"),
            "{}",
            update.diagnostics[0].message
        );
    }

    #[test]
    fn a_node_builtin_is_an_external_not_a_diagnostic() {
        // Locks in that making unresolved imports fatal can never start failing
        // builds over Node built-ins: they short-circuit before resolution and are
        // never diagnostics at all.
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { readFileSync } from 'node:fs';\nimport { join } from 'path';\n\
             export const read = (p) => readFileSync(join(p, 'x'));\n",
        )
        .unwrap();

        let config = BuildConfig {
            target: Target::Server,
            ..BuildConfig::default()
        };
        let (bundler, update) =
            discover_direct_with_config(&directory.path().join("entry.js"), &config).unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        assert!(partition_diagnostics(&update.diagnostics, "test build").is_ok());
        let reachable = bundler.reachable_modules_direct();
        assert_eq!(reachable.len(), 1);
    }

    #[test]
    fn an_unsupported_side_effects_glob_is_a_warning_and_the_build_succeeds() {
        // `"sideEffects": ["*.{css,scss}"]` is a common package.json idiom this
        // matcher cannot evaluate. The module is KEPT, so the bundle is correct —
        // only larger. Failing the build on it would reject apps that bundle fine.
        let directory = tempdir().unwrap();
        let package = directory.path().join("node_modules/braced-pkg");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("package.json"),
            r#"{"name":"braced-pkg","type":"module","exports":"./index.js","sideEffects":["*.{css,scss}"]}"#,
        )
        .unwrap();
        fs::write(package.join("index.js"), "export const value = 'ok';").unwrap();
        fs::write(
            directory.path().join("entry.js"),
            "import { value } from 'braced-pkg';\nconsole.log(value);\n",
        )
        .unwrap();

        let (bundler, update) = discover_direct(&directory.path().join("entry.js")).unwrap();
        let side_effects = update
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::SideEffectsGlob)
            .collect::<Vec<_>>();
        assert_eq!(side_effects.len(), 1, "{:?}", update.diagnostics);
        assert!(!side_effects[0].is_fatal());

        let warnings = partition_diagnostics(&update.diagnostics, "test build")
            .expect("an unsupported sideEffects glob must not fail the build");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("sideEffects"), "{}", warnings[0]);
        // And the build really does complete.
        let output = directory.path().join("dist/bundle.js");
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        assert!(output.is_file());
    }

    #[test]
    fn partition_diagnostics_reports_every_fatal_and_keeps_warnings_separate() {
        let diagnostics = vec![
            Diagnostic {
                kind: DiagnosticKind::UnresolvedImport {
                    specifier: "./a".into(),
                    importer: PathBuf::from("/app/one.js"),
                },
                message: "cannot resolve \"./a\"".into(),
            },
            Diagnostic {
                kind: DiagnosticKind::SideEffectsGlob,
                message: "unsupported `sideEffects` glob".into(),
            },
            Diagnostic {
                kind: DiagnosticKind::Source { fatal: false },
                message: "a benign oxc warning".into(),
            },
            Diagnostic {
                kind: DiagnosticKind::Source { fatal: true },
                message: "a real parse error".into(),
            },
        ];

        let error = partition_diagnostics(&diagnostics, "client build").unwrap_err();
        assert!(error.contains("2 fatal build diagnostic(s)"), "{error}");
        assert!(error.contains("cannot resolve \"./a\""), "{error}");
        assert!(error.contains("a real parse error"), "{error}");
        assert!(!error.contains("a benign oxc warning"), "{error}");

        let warnings = partition_diagnostics(&diagnostics[1..3], "client build").unwrap();
        assert_eq!(
            warnings,
            vec![
                "unsupported `sideEffects` glob".to_string(),
                "a benign oxc warning".to_string()
            ]
        );
    }

    /// FINDINGS #19. A legacy v3 app's design tokens are `theme.extend` ON TOP OF the
    /// v3 DEFAULT theme — a different palette, radius scale and type scale from v4's.
    /// The evaluator used to emit only the config's OWN keys, which diffpack then
    /// merged into the vendored v4 defaults, so every unmentioned token came out v4:
    /// `slate-400` as `oklch(...)` rather than `#94a3b8`, `rounded-full` as
    /// `calc(infinity * 1px)` rather than `9999px`. It now resolves the config through
    /// the app's own `tailwindcss/resolveConfig`.
    ///
    /// Runs against the pinned `next-blog-starter` e2e app (a real tailwindcss@3
    /// install); soft-skips when the corpus has not been fetched.
    #[test]
    fn v3_config_evaluator_resolves_the_full_v3_default_theme() {
        let app =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("integration/e2e/apps/next-blog-starter");
        let config = app.join("tailwind.config.ts");
        if !config.is_file() || !app.join("node_modules/tailwindcss").is_dir() {
            return;
        }
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let loader = std::env::temp_dir().join("diffpack-tailwind-config-eval-test.mjs");
        fs::write(&loader, include_str!("../../scripts/tailwind-config-eval.mjs")).unwrap();
        let output = node_command()
            .arg(&loader)
            .arg(&config)
            .current_dir(&app)
            .output()
            .unwrap();
        let theme = String::from_utf8_lossy(&output.stdout).to_string();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        // v3 DEFAULT tokens the config never mentions, in v3's own sRGB form.
        assert!(theme.contains("--color-slate-400: #94a3b8;"), "{theme}");
        assert!(theme.contains("--radius-full: 9999px;"), "{theme}");
        // The v3 preflight's border reset colour.
        assert!(
            theme.contains("--default-border-color: #e5e7eb;"),
            "{theme}"
        );
        // A `[size, { lineHeight }]` pair splits into the value + modifier tokens
        // instead of stringifying the modifier object into the font-size.
        assert!(theme.contains("--text-4xl: 2.25rem;"), "{theme}");
        assert!(
            theme.contains("--text-4xl--line-height: 2.5rem;"),
            "{theme}"
        );
        assert!(!theme.contains("[object Object]"), "{theme}");
        // The app's OWN tokens still win over the resolved defaults.
        assert!(theme.contains("--color-cyan: #79FFE1;"), "{theme}");
        assert!(
            theme.contains("--shadow-md: 0 8px 30px rgba(0, 0, 0, 0.12);"),
            "{theme}"
        );
        // v3 `columns.12` is a column COUNT, not a v4 `--container-12` width: emitting
        // it made `w-12` resolve against the container scale (100px, not 3rem).
        assert!(!theme.contains("--container-12:"), "{theme}");
        assert!(theme.contains("--spacing-12: 3rem;"), "{theme}");

        // `darkMode: "class"` carries across as the `dark` variant it defines. Without
        // it every `dark:` utility compiled into `@media (prefers-color-scheme: dark)`,
        // so the app painted its dark palette on a browser that merely preferred dark.
        assert!(
            theme.contains("@custom-variant dark (&:is(.dark *));"),
            "{theme}"
        );

        // The resolved v3 fontSize scale REPLACES the vendored v4 one. Merging left
        // v4's `--text-5xl--line-height: 1` in place, but this config sets
        // `fontSize: { '5xl': '2.5rem' }` — a bare string, i.e. no line-height at all.
        assert!(theme.contains("--text-*: initial;"), "{theme}");
        assert!(theme.contains("--text-5xl: 2.5rem;"), "{theme}");
        assert!(!theme.contains("--text-5xl--line-height"), "{theme}");
        // The reset comes first, so every size token after it survives.
        assert!(
            theme.find("--text-*: initial;").unwrap() < theme.find("--text-4xl: 2.25rem;").unwrap(),
            "{theme}"
        );
    }

    /// `darkMode` strategies the evaluator maps, and the hard error for one it does
    /// not: an untranslated strategy would silently fall back to the media query.
    #[test]
    fn v3_config_evaluator_maps_every_dark_mode_strategy() {
        if node_command().arg("--version").output().is_err() {
            return;
        }
        let loader = std::env::temp_dir().join("diffpack-tailwind-darkmode-eval-test.mjs");
        fs::write(&loader, include_str!("../../scripts/tailwind-config-eval.mjs")).unwrap();
        let dir = std::env::temp_dir().join("diffpack-tailwind-darkmode-configs");
        fs::create_dir_all(&dir).unwrap();
        let run = |name: &str, dark_mode: &str| {
            let config = dir.join(format!("{name}.cjs"));
            fs::write(
                &config,
                format!("module.exports = {{ {dark_mode} theme: {{}} }};\n"),
            )
            .unwrap();
            let out = node_command().arg(&loader).arg(&config).output().unwrap();
            (
                String::from_utf8_lossy(&out.stdout).to_string(),
                String::from_utf8_lossy(&out.stderr).to_string(),
                out.status.success(),
            )
        };

        let (media, _, ok) = run("media", "darkMode: 'media',");
        assert!(ok);
        assert!(!media.contains("@custom-variant"), "{media}");
        let (absent, _, ok) = run("absent", "");
        assert!(ok);
        assert!(!absent.contains("@custom-variant"), "{absent}");

        // v3's `class` strategy emits `<selector> &`; `selector` emits the
        // `:where(sel, sel *)` form that also matches the element itself.
        let (class, _, ok) = run("class", "darkMode: 'class',");
        assert!(ok);
        assert!(
            class.contains("@custom-variant dark (&:is(.dark *));"),
            "{class}"
        );
        let (named, _, ok) = run("named", "darkMode: ['class', '[data-mode=\"dark\"]'],");
        assert!(ok);
        assert!(
            named.contains("@custom-variant dark (&:is([data-mode=\"dark\"] *));"),
            "{named}"
        );
        let (selector, _, ok) = run("selector", "darkMode: 'selector',");
        assert!(ok);
        assert!(
            selector.contains("@custom-variant dark (&:where(.dark, .dark *));"),
            "{selector}"
        );

        // An unmapped strategy is a hard, named failure — never a silent fallback.
        let (_, stderr, ok) = run("variant", "darkMode: ['variant', '&:not(.light *)'],");
        assert!(
            !ok,
            "an unmapped darkMode strategy must fail the evaluation"
        );
        assert!(stderr.contains("darkMode"), "{stderr}");
    }

    #[test]
    fn external_provider_supplies_the_real_discovery_source() {
        struct Supplied {
            dependency: String,
        }
        impl diffpack_core::ModuleProvider for Supplied {
            fn name(&self) -> &str {
                "test:supplied"
            }

            fn load(
                &self,
                request: diffpack_core::LoadRequest<'_>,
            ) -> Result<Option<diffpack_core::LoadedSource>, diffpack_core::ProviderDiagnostic>
            {
                Ok(Some(diffpack_core::LoadedSource {
                    code: if request.id == self.dependency {
                        b"export const dependency = 1;\n".to_vec()
                    } else {
                        b"import { dependency } from 'provided:dep';\nimport runtime from 'provided:runtime';\nexport const supplied = ;\n"
                            .to_vec()
                    },
                    language: diffpack_core::SourceLanguage::JavaScript,
                    source_map: None,
                    watch_files: Vec::new(),
                }))
            }

            fn resolve(
                &self,
                request: diffpack_core::ResolveRequest<'_>,
            ) -> Result<diffpack_core::ResolveResult, diffpack_core::ProviderDiagnostic>
            {
                Ok(match request.specifier {
                    "provided:dep" => {
                        diffpack_core::ResolveResult::Resolved(self.dependency.clone())
                    }
                    "provided:runtime" => {
                        diffpack_core::ResolveResult::External("provided:runtime".into())
                    }
                    _ => diffpack_core::ResolveResult::NoMatch,
                })
            }

            fn transform(
                &self,
                request: diffpack_core::TransformRequest<'_>,
            ) -> Result<Option<diffpack_core::TransformOutput>, diffpack_core::ProviderDiagnostic>
            {
                let code = String::from_utf8(request.code.to_vec()).unwrap();
                Ok(Some(diffpack_core::TransformOutput {
                    code: code.replace("= ;", "= 1;").into_bytes(),
                    language: request.language,
                    source_map: None,
                    watch_files: Vec::new(),
                    emitted_assets: vec![diffpack_core::EmittedAsset {
                        name: Some("provider-note.txt".into()),
                        source: b"from provider".to_vec(),
                    }],
                }))
            }
        }

        let file = tempfile::NamedTempFile::new().unwrap();
        let dependency = tempfile::NamedTempFile::new().unwrap();
        fs::write(file.path(), "this is deliberately not javascript {{{").unwrap();
        let dependency = dependency.path().canonicalize().unwrap();
        let providers = diffpack_core::ProviderPipeline::new(vec![Box::new(Supplied {
            dependency: dependency.to_string_lossy().into_owned(),
        })]);
        let (bundler, update) = discover_direct_with_config_and_providers(
            file.path(),
            &BuildConfig::default(),
            providers,
        )
        .unwrap();
        assert_eq!(update.transformed_modules, 2);
        assert!(update.diagnostics.is_empty());
        let output = tempfile::tempdir().unwrap();
        let entry_output = output.path().join("entry.js");
        bundler
            .emit(&bundler.reachable_modules_direct(), &entry_output)
            .unwrap();
        assert_eq!(
            fs::read(output.path().join("assets/provider-note.txt")).unwrap(),
            b"from provider"
        );
    }

    #[test]
    fn external_provider_transforms_filesystem_sources() {
        struct TransformOnly;

        impl diffpack_core::ModuleProvider for TransformOnly {
            fn name(&self) -> &str {
                "test:transform-only"
            }

            fn transform(
                &self,
                request: diffpack_core::TransformRequest<'_>,
            ) -> Result<Option<diffpack_core::TransformOutput>, diffpack_core::ProviderDiagnostic>
            {
                let code = String::from_utf8(request.code.to_vec()).unwrap();
                if !code.contains("__PROVIDER_VALUE__") {
                    return Ok(None);
                }
                Ok(Some(diffpack_core::TransformOutput {
                    code: code.replace("__PROVIDER_VALUE__", "42").into_bytes(),
                    language: request.language,
                    source_map: None,
                    watch_files: Vec::new(),
                    emitted_assets: Vec::new(),
                }))
            }
        }

        let dir = tempdir().unwrap();
        let entry = dir.path().join("entry.js");
        fs::write(&entry, "globalThis.answer = __PROVIDER_VALUE__;\n").unwrap();
        let providers = diffpack_core::ProviderPipeline::new(vec![Box::new(TransformOnly)]);
        let (bundler, update) =
            discover_direct_with_config_and_providers(&entry, &BuildConfig::default(), providers)
                .unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let output = dir.path().join("bundle.js");
        bundler
            .emit(&bundler.reachable_modules_direct(), &output)
            .unwrap();
        let output = fs::read_to_string(output).unwrap();
        assert!(output.contains("42"), "{output}");
        assert!(!output.contains("__PROVIDER_VALUE__"), "{output}");
    }

    #[test]
    fn injected_compiler_policy_drives_parallel_discovery_and_serial_rebuild() {
        struct CountingCompiler(std::sync::Arc<std::sync::atomic::AtomicUsize>);

        impl diffpack_core::compiler::ModuleCompiler for CountingCompiler {
            fn compile(
                &self,
                request: diffpack_core::compiler::CompileRequest<'_>,
            ) -> diffpack_core::transform::TransformResult {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let source = request.source.replace("__POLICY__", "41");
                diffpack_core::compiler::transform_module_in_language(
                    request.path,
                    &source,
                    request.target,
                    request.refresh,
                    request.jsx,
                    request.project_config,
                    request.language,
                    request.source_maps,
                )
            }
        }

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        let dependency = directory.path().join("dependency.js");
        fs::write(
            &entry,
            "import { value } from './dependency.js'; globalThis.answer = value;\n",
        )
        .unwrap();
        fs::write(&dependency, "export const value = __POLICY__;\n").unwrap();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (mut bundler, update) = discover_direct_with_config_providers_and_compiler(
            &entry,
            &BuildConfig::default(),
            ProviderPipeline::default(),
            Arc::new(CountingCompiler(Arc::clone(&calls))),
        )
        .unwrap();
        assert_eq!(update.transformed_modules, 2);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);

        fs::write(&dependency, "export const value = __POLICY__ + 1;\n").unwrap();
        let update = bundler.rebuild_path(&dependency).unwrap();
        assert_eq!(update.transformed_modules, 1);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    #[test]
    fn injected_special_module_policy_can_claim_an_external_query() {
        struct CustomQuery;

        impl diffpack_default_loader::module_policy::SpecialModulePolicy for CustomQuery {
            fn query_module(
                &self,
                resource: &ResourceId,
                _target: Target,
                compile: &mut diffpack_default_loader::module_policy::SyntheticCompiler<'_>,
            ) -> Result<Option<SpecialModule>, String> {
                if resource.query.as_deref() != Some("custom") {
                    return Ok(None);
                }
                Ok(Some(diffpack_default_loader::module::virtual_module(
                    "export default 73;\n",
                    compile,
                )))
            }
        }

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(
            &entry,
            "import value from './value.js?custom'; globalThis.answer = value;\n",
        )
        .unwrap();
        fs::write(directory.path().join("value.js"), "ignored by policy\n").unwrap();
        let (bundler, update) = discover_with_policies(
            &entry,
            &BuildConfig::default(),
            ProviderPipeline::default(),
            Arc::new(diffpack_web::compiler::WebCompiler::core()),
            Arc::new(CustomQuery),
        )
        .unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        assert!(
            bundler
                .reachable_modules_direct()
                .iter()
                .any(|id| id.ends_with("value.js?custom"))
        );
    }

    #[test]
    fn injected_runtime_policy_contributes_entry_runtime_without_graph_access() {
        struct MarkerRuntime;
        #[derive(Debug)]
        struct MarkerSource;

        impl diffpack_core::runtime::RuntimeIntegrationPolicy for MarkerRuntime {
            fn configure(
                &self,
                _request: diffpack_core::runtime::RuntimePolicyRequest<'_>,
            ) -> diffpack_core::runtime::RuntimePolicyOutput {
                diffpack_core::runtime::RuntimePolicyOutput {
                    entry_preludes: vec!["globalThis.__externalRuntimePolicy=1;".into()],
                    ..Default::default()
                }
            }
        }

        impl diffpack_default_loader::source_policy::SourceIntegrationPolicy for MarkerSource {
            fn transform(
                &self,
                _path: &Path,
                source: &str,
                _target: Target,
            ) -> Result<Option<String>, String> {
                Ok(source
                    .contains("__SOURCE_POLICY__")
                    .then(|| source.replace("__SOURCE_POLICY__", "41")))
            }
        }

        let directory = tempdir().unwrap();
        let entry = directory.path().join("entry.js");
        fs::write(
            &entry,
            "module.exports = { value: __SOURCE_POLICY__ + 1 };\n",
        )
        .unwrap();
        let (bundler, update) = Bundler::discover_with_driver_policies(
            &entry,
            &BuildConfig {
                hmr: true,
                ..BuildConfig::default()
            },
            ProviderPipeline::default(),
            DriverPolicies {
                compiler: Arc::new(diffpack_web::compiler::WebCompiler::core()),
                special_modules: Arc::new(
                    diffpack_default_loader::module_policy::NoSpecialModulePolicy,
                ),
                runtime: Arc::new(MarkerRuntime),
                output: Arc::new(diffpack_default_loader::output::NoOutputIntegrationPolicy),
                source: Arc::new(MarkerSource),
            },
        )
        .unwrap();
        assert!(update.diagnostics.is_empty(), "{:?}", update.diagnostics);
        let output = directory.path().join("bundle.js");
        bundler
            .emit_with_options(
                &bundler.reachable_modules_direct(),
                &output,
                EmitOptions {
                    hmr: true,
                    ..EmitOptions::default()
                },
            )
            .unwrap();
        let emitted = fs::read_to_string(output).unwrap();
        assert!(
            emitted.contains("globalThis.__externalRuntimePolicy=1;"),
            "{emitted}"
        );
        assert!(emitted.contains("41 + 1"), "{emitted}");
    }

    #[test]
    fn provider_context_reports_development_for_hmr_builds() {
        struct RequiresDevelopment;

        impl diffpack_core::ModuleProvider for RequiresDevelopment {
            fn name(&self) -> &str {
                "test:requires-development"
            }

            fn transform(
                &self,
                request: diffpack_core::TransformRequest<'_>,
            ) -> Result<Option<diffpack_core::TransformOutput>, diffpack_core::ProviderDiagnostic>
            {
                if request.context.environment.mode != diffpack_core::BuildMode::Development {
                    return Err(diffpack_core::ProviderDiagnostic::new(
                        "expected development provider context",
                    ));
                }
                Ok(None)
            }
        }

        let file = tempfile::NamedTempFile::new().unwrap();
        fs::write(file.path(), "globalThis.providerMode = 'development';\n").unwrap();
        let providers = diffpack_core::ProviderPipeline::new(vec![Box::new(RequiresDevelopment)]);
        let config = BuildConfig {
            hmr: true,
            ..BuildConfig::default()
        };
        discover_direct_with_config_and_providers(file.path(), &config, providers).unwrap();
    }
}
