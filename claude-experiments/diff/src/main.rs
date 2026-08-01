use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc;
use std::time::Instant;

use diffpack::bundle_benchmark::{
    run_bundle_scale_direct, run_bundle_scale_direct_dependency_edit, run_bundle_scale_direct_live,
    run_bundle_scale_direct_live_dependency_edit, run_bundle_scale_direct_live_minified,
    run_bundle_scale_direct_live_minified_dependency_edit, write_live_scale_visualization,
};
use diffpack::bundler::EmitOptions;
use notify::{RecursiveMode, Watcher};

#[cfg(feature = "memory-accounting")]
#[global_allocator]
static GLOBAL_ALLOCATOR: diffpack_core::memory::TrackingAllocator =
    diffpack_core::memory::TrackingAllocator;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SourceMapChoice {
    #[default]
    Auto,
    On,
    Off,
}

impl SourceMapChoice {
    fn from_flags<'a>(flags: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        let mut on = false;
        let mut off = false;
        for flag in flags {
            match flag {
                "--sourcemap" => on = true,
                "--no-sourcemap" => off = true,
                _ => {}
            }
        }
        match (on, off) {
            (true, true) => {
                Err("--sourcemap and --no-sourcemap were both given; pass one or neither".into())
            }
            (true, false) => Ok(Self::On),
            (false, true) => Ok(Self::Off),
            (false, false) => Ok(Self::Auto),
        }
    }

    fn resolve(self, integration_default: bool) -> bool {
        match self {
            Self::Auto => integration_default,
            Self::On => true,
            Self::Off => false,
        }
    }
}

fn main() -> ExitCode {
    let started = Instant::now();
    // `build-app <root> production` -> "build-app production": the subcommand plus the
    // environment, skipping the project path (noise) and the flags.
    let words: Vec<String> = env::args()
        .skip(1)
        .filter(|word| !word.starts_with("--") && !word.contains('/') && word != ".")
        .take(2)
        .collect();
    let label = if words.is_empty() {
        "diffpack".to_string()
    } else {
        words.join(" ")
    };
    let result = run();
    diffpack_core::build_profile::report(&label, started.elapsed().as_secs_f64() * 1000.0);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    match arguments.next().as_deref().and_then(|value| value.to_str()) {
        Some("bundle-scale-direct") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(run_bundle_scale_direct(modules, imports)?, "direct");
            Ok(())
        }
        Some("bundle-scale-direct-deps") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(
                run_bundle_scale_direct_dependency_edit(modules, imports)?,
                "direct-dependency-edit",
            );
            Ok(())
        }
        Some("bundle-scale-direct-live") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(
                run_bundle_scale_direct_live(modules, imports)?,
                "direct-live",
            );
            Ok(())
        }
        Some("bundle-scale-direct-live-deps") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(
                run_bundle_scale_direct_live_dependency_edit(modules, imports)?,
                "direct-live-dependency-edit",
            );
            Ok(())
        }
        Some("bundle-scale-direct-live-minify") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(
                run_bundle_scale_direct_live_minified(modules, imports)?,
                "direct-live-minified",
            );
            Ok(())
        }
        Some("bundle-scale-direct-live-minify-deps") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            print_bundle_scale(
                run_bundle_scale_direct_live_minified_dependency_edit(modules, imports)?,
                "direct-live-minified-dependency-edit",
            );
            Ok(())
        }
        Some("bundle-scale-memory") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            let edits = parse_usize(arguments.next(), 100, "edit count")?;
            let minify = arguments.any(|value| value.to_str() == Some("--minify"));
            #[cfg(not(feature = "memory-accounting"))]
            {
                let _ = (modules, imports, edits, minify);
                #[allow(clippy::needless_return)]
                return Err("bundle-scale-memory needs the accounting build: \
                     cargo run --release --features memory-accounting -- bundle-scale-memory ... \
                     (production binaries carry no allocator override, so wall-time and \
                     memory are measured in separate runs)"
                    .into());
            }
            #[cfg(feature = "memory-accounting")]
            {
                let result = diffpack::bundle_benchmark::run_bundle_scale_memory(
                    modules, imports, edits, minify,
                )?;
                println!(
                    "modules,reachable,source_mb,build_peak_mb,retained_mb,bytes_per_module,edits,transformed_per_edit_max,edit_growth_kb,retained_after_drop_kb"
                );
                println!(
                    "{},{},{:.3},{:.3},{:.3},{:.1},{},{},{:.1},{:.1}",
                    result.modules,
                    result.reachable,
                    result.source_bytes as f64 / 1_000_000.0,
                    result.build_peak_bytes as f64 / 1_000_000.0,
                    result.retained_after_build_bytes as f64 / 1_000_000.0,
                    result.bytes_per_module,
                    result.edits,
                    result.transformed_per_edit_max,
                    result.retained_growth_over_edits_bytes as f64 / 1_000.0,
                    result.retained_after_drop_bytes as f64 / 1_000.0,
                );
                Ok(())
            }
        }
        Some("build") => {
            let project_root = arguments.next().ok_or_else(usage)?;
            let remaining = arguments.collect::<Vec<_>>();
            let has_flag = |flag: &str| remaining.iter().any(|value| value.to_str() == Some(flag));
            let vite = has_flag("--vite");
            let minify = !has_flag("--no-minify");
            // `diffpack build` is the Vite-shaped web build, and Vite's own
            // `build.sourcemap` default is `false` — so `Auto` is off here. (Honoring
            // a Vite config that DOES set `build.sourcemap` is a separate change; its
            // `'inline'` and `'hidden'` modes are emit shapes diffpack has no
            // representation for yet, and half-reading the field would silently treat
            // them as plain external maps.)
            let source_map =
                SourceMapChoice::from_flags(remaining.iter().filter_map(|value| value.to_str()))?
                    .resolve(false);
            let out_dir = remaining
                .iter()
                .position(|value| value.to_str() == Some("--out-dir"))
                .map(|index| {
                    remaining
                        .get(index + 1)
                        .map(PathBuf::from)
                        .ok_or_else(|| "--out-dir needs a directory argument".to_string())
                })
                .transpose()?;

            let root = Path::new(&project_root).canonicalize().map_err(|error| {
                format!(
                    "cannot open project root {}: {error}",
                    Path::new(&project_root).display()
                )
            })?;
            web_build(&root, out_dir, vite, minify, source_map)
        }
        Some("preview") => {
            // Serve a completed production build (`diffpack build` output) over HTTP,
            // the analogue of `vite preview`. Static files are served from the build
            // directory; a client-routed path with no matching file falls back to the
            // build's `index.html` (SPA fallback), exactly as `vite preview` does.
            let dir = arguments
                .next()
                .ok_or_else(|| "usage: diffpack preview <build-dir> [port]".to_string())?;
            let port = arguments
                .next()
                .and_then(|value| value.to_str().map(str::to_string))
                .map(|value| {
                    value
                        .parse::<u16>()
                        .map_err(|error| format!("invalid preview port: {error}"))
                })
                .transpose()?
                .unwrap_or(4173);
            let build_dir = Path::new(&dir).canonicalize().map_err(|error| {
                format!(
                    "cannot open build directory {} — run `diffpack build <root>` first ({error})",
                    Path::new(&dir).display()
                )
            })?;
            diffpack::dev_server::preview(&build_dir, port)
        }
        Some("build-app") => {
            let project_root = arguments.next().ok_or_else(usage)?;
            let remaining = arguments.collect::<Vec<_>>();
            // Production minification is the default for `build-app` (the real
            // emitted client/server chunks should ship minified); `--no-minify`
            // opts out for debugging the readable output.
            let no_minify = remaining
                .iter()
                .any(|value| value.to_str() == Some("--no-minify"));
            let minify = !no_minify;
            // Production source maps, composed through the minify pass. With NO flag
            // the app's own framework config decides, per graph — for a Next app that
            // is `next build`'s policy: server maps always, browser maps only under
            // `productionBrowserSourceMaps` (see `next_adapter::default_source_maps`),
            // which is what makes a diffpack build comparable with a `next build` of
            // the same app. `--sourcemap` / `--no-sourcemap` force it either way.
            let source_map =
                SourceMapChoice::from_flags(remaining.iter().filter_map(|value| value.to_str()))?;
            // `--server-dir=<name>`: which directory under `.diffpack-output/` a
            // server-like graph emits into (default `server`). The production
            // orchestrator points the react-server graph at `rsc-render` so it can
            // build concurrently with the ssr graph, which owns `server/`. The
            // `=` spelling is deliberate: a separate value token would be
            // indistinguishable from the positional environment argument.
            let server_dir_name = remaining
                .iter()
                .filter_map(|value| value.to_str())
                .find_map(|value| value.strip_prefix("--server-dir="))
                .unwrap_or("server")
                .to_string();
            let environment = remaining
                .iter()
                .find(|value| !value.to_string_lossy().starts_with("--"))
                .and_then(|value| value.to_str().map(str::to_string))
                .unwrap_or_else(|| "client".to_string());

            // `static` — the SSG prerender phase (Full SSG). It builds NO graph: it
            // reuses the three already-emitted bundles (client / react-server render /
            // ssr) + their manifests, re-runs native route classification to write the
            // prerender plan, then spawns the node prerenderer (the app's own React
            // runtime — the explicitly-allowed oracle, exactly as the orchestrator
            // renders) to write `.html` + `.rsc` for every static/SSG route.
            if environment == "static" {
                let static_export = remaining
                    .iter()
                    .any(|value| value.to_str() == Some("--static-export"));
                return build_static(Path::new(&project_root), static_export);
            }

            // `production` — the one-command production build: run every graph in
            // order and lay them out into a single coherent, servable output
            // (`public/` + `server/` + `rsc-render/` + manifests + the production
            // server entry), so `diffpack start <out>` boots a deployable app. This
            // replaces the previous manual client -> react-server -> `cp` -> ssr shell
            // dance and is the only supported build->deploy path for a DYNAMIC app.
            // `build_production` itself dispatches pages-router vs app-router vs SPA.
            if environment == "production" {
                return build_production(Path::new(&project_root), &remaining);
            }

            // Next.js PAGES-router apps take a dedicated classic-SSR path (not the
            // RSC spine): a `pages/` project with no app-router `app/page` builds a
            // client hydration bundle (`build-app <root> client`) or a Node SSR bundle
            // (`build-app <root> ssr`). Checked after `static`/`production` (handled
            // above) and BEFORE the app-router/TanStack config so a pages project
            // never falls into the RSC path. App-router wins on a hybrid (checked
            // first) so a project with both `app/` and `pages/` builds as app-router.
            if !diffpack_next::next_adapter::is_app_router(Path::new(&project_root))
                && diffpack_next::next_pages::is_pages_router(Path::new(&project_root))
            {
                return build_pages_app(Path::new(&project_root), &environment, minify, source_map);
            }

            // Next.js app-router apps have no TanStack/src entry; their "entry" is
            // the app-router file convention (`app/layout.tsx` wrapping
            // `app/page.tsx`). The next adapter detects such a project, scaffolds the
            // three RSC entries (+ minimal `next/*` shims) under `.diffpack-next/`,
            // and returns a ready config; a non-Next project returns `None` and falls
            // back to the TanStack `derive_config` path unchanged.
            let configure_stage = diffpack_core::build_profile::stage("adapter/configure");
            let (mut config, is_next_app) = match diffpack::config::configure_next_app(
                Path::new(&project_root),
                &environment,
            )? {
                Some(next_config) => {
                    println!(
                        "next app-router adapter: scaffolded .diffpack-next/ for environment={environment}"
                    );
                    (next_config, true)
                }
                None => (
                    diffpack::config::derive_config(Path::new(&project_root), &environment)?,
                    false,
                ),
            };
            // A real per-module source map costs a second print per module, so it
            // is produced only when the emit will actually write `.map` files. The
            // adapter already set the framework's own default for this graph; a CLI
            // flag overrides it, absence of one keeps it.
            config.build.source_maps = source_map.resolve(config.build.source_maps);
            let source_map = config.build.source_maps;
            drop(configure_stage);
            let entry = config
                .entry
                .clone()
                .ok_or_else(|| format!("no {environment} entry found for the app"))?;
            let output_root = Path::new(&project_root).join(".diffpack-output");

            // Natively generate `src/routeTree.gen.ts` from `src/routes/` BEFORE
            // discovery, so the bundler consumes a diffpack-generated route tree
            // instead of one produced by TanStack Router's Vite plugin. This is a
            // build-emit step off the incremental hot path (mirroring native
            // manifest generation), so the thesis guards are unaffected. A
            // non-file-routed project (no `src/routes`) is a no-op.
            if !is_next_app {
                if let Some(route_count) =
                    diffpack_tanstack::route_tree::generate_for_project(Path::new(&project_root))?
                {
                    println!(
                        "generated src/{} natively ({route_count} route(s))",
                        diffpack_tanstack::route_tree::ROUTE_TREE_FILE,
                    );
                }
            }

            // A server build's TanStack manifest module (`tanstack-start-manifest:v`)
            // maps each route to the CLIENT build's emitted chunk URLs, so it is
            // generated natively from the client build's persisted route/chunk
            // manifest. Register it as a virtual module before discovery, so the
            // server's `router-manifest.js` import resolves and loads it. A missing
            // client manifest is a hard, specific error (run the client build first)
            // rather than a silent empty manifest.
            // RSC server actions — client transport. A `"use server"` module built
            // for the client is rewritten into `createServerReference(id, callServer)`
            // stubs importing `callServer` from `#diffpack-call-server`; register the
            // embedded transport under that specifier so the stub resolves. Harmless
            // (unreachable) when the app has no `"use server"` module.
            if is_next_app {
                config.build.virtual_modules.push((
                    diffpack_next::rsc::CALL_SERVER_SPECIFIER.to_string(),
                    diffpack_next::rsc::call_server_module_source().to_string(),
                ));
            }

            if config.environment != "client" {
                if !is_next_app {
                    let client_manifest_path =
                        output_root.join(diffpack_tanstack::manifest::CLIENT_MANIFEST_FILE);
                    let client_manifest = diffpack_tanstack::manifest::ClientRouteManifest::read(
                        &client_manifest_path,
                    )?;
                    config.build.virtual_modules.push((
                        diffpack_tanstack::manifest::START_MANIFEST_SPECIFIER.to_string(),
                        client_manifest.to_start_manifest_source(),
                    ));
                    // The sibling dev-only virtual module `loadVirtualModule.js`
                    // statically references (only used under TSS_DEV_SERVER, but its
                    // `import()` literal must still resolve). Register it too so the
                    // server build resolves cleanly on react-start versions that emit it.
                    config.build.virtual_modules.push((
                        diffpack_tanstack::manifest::INJECTED_HEAD_SCRIPTS_SPECIFIER.to_string(),
                        diffpack_tanstack::manifest::injected_head_scripts_module_source(),
                    ));
                    println!(
                        "loaded client route manifest ({} routes) from {}",
                        client_manifest.routes.len(),
                        client_manifest_path.display(),
                    );

                    // Server functions: register the native server-fn resolver module
                    // (`#tanstack-start-server-fn-resolver`) that `getServerFnById`
                    // dispatches through. It is generated from a pre-scan of the app
                    // source for `createServerFn(...).handler(...)` declarations, keyed
                    // by the same deterministic function id the server transform bakes
                    // into each handler — so an HTTP server-fn request reaches exactly
                    // the registered handler. Registered before discovery so the
                    // subpath import resolves to it instead of the framework's fake
                    // (undefined-returning) resolver.
                    let server_fns = diffpack_tanstack::server_fn::scan_project_server_fns(
                        Path::new(&project_root),
                    )?;
                    config.build.virtual_modules.push((
                        diffpack_tanstack::server_fn::RESOLVER_SPECIFIER.to_string(),
                        diffpack_tanstack::server_fn::generate_resolver_module(&server_fns),
                    ));
                    println!(
                        "registered {} server function(s) in the native server-fn resolver",
                        server_fns.len(),
                    );
                }

                if is_next_app {
                    // RSC server actions — server dispatch. Register the generated action
                    // resolver (`#diffpack-rsc-action-resolver`) that `getServerActionById`
                    // dispatches through, keyed by the same `"<moduleId>#<name>"` id the
                    // client stub and the server registration derive, plus the embedded
                    // `handleServerAction` endpoint (`#diffpack-rsc-action-handler`). The
                    // resolver is generated from a pre-scan of the app source for
                    // `"use server"` modules. Registered before discovery so the subpath
                    // imports resolve to the native modules.
                    let server_actions =
                        diffpack_next::rsc::scan_project_server_actions(Path::new(&project_root))?;
                    config.build.virtual_modules.push((
                        diffpack_next::rsc::ACTION_RESOLVER_SPECIFIER.to_string(),
                        diffpack_next::rsc::generate_action_resolver_module(&server_actions),
                    ));
                    config.build.virtual_modules.push((
                        diffpack_next::rsc::ACTION_HANDLER_SPECIFIER.to_string(),
                        diffpack_next::rsc::action_handler_module_source().to_string(),
                    ));
                    println!(
                        "registered {} server action(s) in the native rsc action resolver",
                        server_actions.len(),
                    );

                    // RSC flight — SSR consumer manifest (Manifest #2). The SSR pass
                    // consumes the flight stream with `createFromReadableStream`, which
                    // resolves the client references it carries through this manifest.
                    // It is derived natively from the client build's Manifest #1 (the
                    // client-references manifest), so the SSR graph resolves each
                    // client reference to the real module under diffpack's one runtime-id
                    // scheme. Registered under `#diffpack-rsc-ssr-consumer-manifest`; an
                    // app with no `"use client"` module gets an empty (but valid) map.
                    let client_references_path =
                        output_root.join(diffpack_next::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
                    let client_references = diffpack_next::rsc::ClientReferencesManifest::read(
                        &client_references_path,
                    )?;
                    config.build.virtual_modules.push((
                        diffpack_next::rsc::SSR_CONSUMER_MANIFEST_SPECIFIER.to_string(),
                        client_references.to_ssr_consumer_manifest_module(None),
                    ));
                    println!(
                        "registered the rsc ssr consumer manifest ({} client reference(s))",
                        client_references.entries.len(),
                    );
                }
            }

            println!(
                "app: environment={} ({} aliases), entry={}",
                config.environment,
                config.build.aliases.len(),
                entry.display(),
            );
            let mut rebuilt_for_async_islands = false;
            let (bundler, reachable, warnings) = loop {
                let discover_stage = diffpack_core::build_profile::stage("graph/discover");
                let (bundler, update) = if is_next_app {
                    diffpack::bundler::discover_next_with_config(&entry, &config.build)?
                } else {
                    diffpack::bundler::discover_tanstack_with_config(&entry, &config.build)?
                };
                drop(discover_stage);
                // A fatal diagnostic (an unresolved import, a source error) means the
                // chunk this build would write is already broken, so it is not written
                // at all. Only the non-fatal diagnostics survive as warnings.
                let warnings = diffpack::bundler::partition_diagnostics(
                    &update.diagnostics,
                    &format!("{} build", config.environment),
                )?;
                let reachability_stage = diffpack_core::build_profile::stage("graph/reachability");
                let reachable = bundler.reachable_modules_direct();
                drop(reachability_stage);
                // Client islands are pinned LAZILY (bundled + registered, evaluated on
                // demand by the RSC seam) except the async-tainted ones, which must be
                // evaluated at entry boot. That set is a fact of the discovered graph;
                // when it differs from what the entries on disk were generated with,
                // regenerate them and rediscover. Steady state (a recorded, unchanged
                // set) takes this branch zero times.
                if matches!(config.environment.as_str(), "client" | "ssr")
                    && diffpack::config::reconcile_next_async_islands(
                        Path::new(&project_root),
                        &config.environment,
                        &bundler,
                        &reachable,
                    )?
                {
                    if rebuilt_for_async_islands {
                        return Err(
                            "the async-island set did not stabilize after regenerating the \
                             entries once; this is a diffpack bug"
                                .to_string(),
                        );
                    }
                    rebuilt_for_async_islands = true;
                    println!(
                        "async island set changed; regenerating the entries and rediscovering"
                    );
                    diffpack::config::configure_next_app(
                        Path::new(&project_root),
                        &config.environment,
                    )?;
                    continue;
                }
                break (bundler, reachable, warnings);
            };
            println!(
                "reachable {} modules; {} warning(s)",
                reachable.len(),
                warnings.len()
            );
            for warning in &warnings {
                println!("  warning: {warning}");
            }

            // Emit the environment natively. The `client` environment writes the
            // browser `public/` layout (`.js` chunks + CSS + copied static
            // files) plus the route/chunk manifest the server build consumes; the
            // server environments (`ssr`/`nitro`) write the Node ESM `server/`
            // layout (`.mjs` chunks) including the natively generated
            // `tanstack-start-manifest` chunk and the Node HTTP runtime entry
            // (`server/index.mjs` plus its `_ssr/` adapter, SSR, and router
            // modules) that boots the SSR handler and serves the `public/` assets.
            let emit_options = EmitOptions {
                minify,
                source_map,
                ..EmitOptions::default()
            };
            if config.environment == "client" {
                // BOTH manifests the server-like graphs consume are pure functions
                // of the discovered graph (live-module refinement + the chunk
                // plan), not of any emitted bytes — so they are computed and
                // published FIRST. The production orchestrator watches for them
                // and starts the react-server and ssr builds the moment they
                // exist, overlapping those whole builds with this one's
                // render/minify tail. Each is staged and renamed into place so a
                // watcher can never read a half-written manifest.
                let manifests_stage = diffpack_core::build_profile::stage("emit/client-manifests");
                let client_manifest = diffpack_tanstack::manifest::from_bundle_graph(
                    &bundler.integration_manifest_graph(&reachable, "client.js")?,
                    "/",
                )?;
                let client_manifest_path =
                    output_root.join(diffpack_tanstack::manifest::CLIENT_MANIFEST_FILE);
                let client_references = diffpack_next::rsc::client_references_from_bundle_graph(
                    &bundler.integration_manifest_graph(&reachable, "client.js")?,
                );
                let client_references_path =
                    output_root.join(diffpack_next::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
                std::fs::create_dir_all(&output_root)
                    .map_err(|error| format!("cannot create {}: {error}", output_root.display()))?;
                let publish = |write: &dyn Fn(&Path) -> Result<(), String>,
                               path: &Path|
                 -> Result<(), String> {
                    let staged = path.with_extension(format!("staged-{}", std::process::id()));
                    write(&staged)?;
                    std::fs::rename(&staged, path)
                        .map_err(|error| format!("cannot publish {}: {error}", path.display()))
                };
                publish(&|path| client_manifest.write(path), &client_manifest_path)?;
                publish(
                    &|path| client_references.write(path),
                    &client_references_path,
                )?;
                drop(manifests_stage);
                let emit_stage = diffpack_core::build_profile::stage("emit/public");
                let summary = bundler.emit_public(&reachable, &output_root, emit_options)?;
                drop(emit_stage);
                let copy_stage = diffpack_core::build_profile::stage("emit/copy-static-public");
                let static_files = diffpack::config::copy_static_public(
                    Path::new(&project_root),
                    &summary.output_dir,
                )?;
                drop(copy_stage);
                // next/image (Slice J): emit downscaled responsive variants for every
                // raster under `public/` into `<public>/_diffpack-image/`. The runtime
                // shim's `srcset` points at these static files (no image server). A
                // no-op for a non-Next project (no public images / no app-router).
                //
                // An app whose next.config turns Next's optimizer off (or replaces it
                // with its own loader) gets NO variants — the emitted `<img>` can never
                // reference one. Reported, never silent: skipping this is a large chunk
                // of a build on an image-heavy app and must be visible in the log.
                if let diffpack_next::next_adapter::ImageOptimization::Disabled(reason) =
                    diffpack_next::next_adapter::ImageOptimization::for_project(Path::new(
                        &project_root,
                    ))
                {
                    println!(
                        "next/image: {reason}, so no build-time variants are generated \
                         (every <Image> renders a plain <img src> with no srcset, as under next build)"
                    );
                }
                let scan_stage = diffpack_core::build_profile::stage("image/scan-public");
                let public_images =
                    diffpack_next::next_adapter::scan_public_images(Path::new(&project_root))?;
                drop(scan_stage);
                if !public_images.is_empty() {
                    let variants_stage = diffpack_core::build_profile::stage("image/emit-variants");
                    let variants = diffpack_next::next_adapter::emit_image_variants(
                        Path::new(&project_root),
                        &summary.output_dir,
                        &public_images,
                    )?;
                    drop(variants_stage);
                    if variants > 0 {
                        println!(
                            "emitted {variants} next/image variant file(s) under {}/_diffpack-image",
                            summary.output_dir.display(),
                        );
                    }
                }
                // Metadata IMAGE file conventions (app/icon.png, app/favicon.ico, ...):
                // copy them to the served public/ so their build-emitted head links
                // resolve. Zero per-request cost (served by the static-asset path).
                let meta_images = diffpack_next::next_adapter::emit_metadata_images(
                    Path::new(&project_root),
                    &summary.output_dir,
                )?;
                if meta_images > 0 {
                    println!(
                        "copied {meta_images} metadata image file(s) to {}",
                        summary.output_dir.display(),
                    );
                }
                // `next/font/local`: copy the app's own font files to the hashed URLs the
                // generated @font-face rules already point at. Driven by the manifest the
                // adapter wrote while generating that CSS, so the two cannot disagree.
                let fonts = diffpack_next::next_font::emit_font_assets(
                    Path::new(&project_root),
                    &summary.output_dir,
                )?;
                if fonts > 0 {
                    println!(
                        "emitted {fonts} next/font/local file(s) under {}/{}",
                        summary.output_dir.display(),
                        diffpack_next::next_font::FONT_ASSET_DIR,
                    );
                }
                println!(
                    "wrote {} ({} client reference(s))",
                    client_references_path.display(),
                    client_references.entries.len(),
                );
                println!(
                    "emitted {}: {} public .js, {} .css, {} asset(s), {} static file(s)",
                    summary.output_dir.display(),
                    summary.javascript_files,
                    summary.css_files,
                    summary.asset_files,
                    static_files,
                );
                println!(
                    "wrote {} ({} routes mapped to client chunks)",
                    client_manifest_path.display(),
                    client_manifest.routes.len(),
                );
            } else {
                let emit_stage = diffpack_core::build_profile::stage("emit/server");
                let summary = bundler.emit_server_into(
                    &reachable,
                    &output_root.join(&server_dir_name),
                    emit_options,
                )?;
                drop(emit_stage);
                // The react-server graph is authoritative for the app's CSS (Server
                // Components render there, so its CSS-Module class scoping matches the
                // flight-rendered classNames). Preserve its compiled `server.css` to
                // the served, non-pruned `public/rsc.css` (the SSR build would
                // otherwise prune it from `server/`); the adapter links it into the
                // document head. Next injects the route's stylesheets the same way.
                //
                // The FILE NAME is the shared constant the render entry's head-link
                // guard reads (`RSC_EMITTED_CSS_FILE`): the entry links `/rsc.css` iff
                // this same file sits beside it, so the link and the artifact are one
                // fact and cannot disagree. Absent = the graph compiled no CSS = no
                // link, which is why nothing is copied and nothing is reported.
                if config.environment == "react-server" {
                    let css = output_root
                        .join(&server_dir_name)
                        .join(diffpack_next::next_adapter::RSC_EMITTED_CSS_FILE);
                    if css.is_file() {
                        let dest = output_root
                            .join("public")
                            .join(diffpack_next::next_adapter::RSC_CSS_URL.trim_start_matches('/'));
                        if let Some(parent) = dest.parent() {
                            std::fs::create_dir_all(parent).map_err(|error| {
                                format!("cannot create {}: {error}", parent.display())
                            })?;
                        }
                        std::fs::copy(&css, &dest).map_err(|error| {
                            format!(
                                "cannot preserve react-server CSS to {}: {error}",
                                dest.display()
                            )
                        })?;
                        println!("preserved react-server CSS -> {}", dest.display());
                    }
                }
                // Persist THIS server build's own client-references manifest (its
                // runtime ids + hosting chunks for every `"use client"` module). The
                // SSR-of-flight pass needs it: the flight carries the CLIENT build's
                // ids, but the SSR graph resolves references through its OWN registry
                // under different ids. Joining the client manifest (client ids) with
                // this one (this build's ids) on the shared canonical module id yields
                // the `ssrModuleMapping` (Manifest #2's `moduleMap` keyed by client id
                // -> this build's id). Written under a build-distinct file name so the
                // react-server render build and the SSR build do not clobber each
                // other's manifest: both server-like graphs emit into this one output
                // root and the ssr pass runs last, so a shared name would lose the
                // react-server graph's set — the set that says which client references
                // a flight can actually carry.
                let server_references = diffpack_next::rsc::client_references_from_bundle_graph(
                    &bundler.integration_manifest_graph(&reachable, "server.mjs")?,
                );
                let server_references_path =
                    output_root.join(if config.environment == "react-server" {
                        diffpack_next::rsc::REACT_SERVER_REFERENCES_MANIFEST_FILE
                    } else {
                        diffpack_next::rsc::SERVER_REFERENCES_MANIFEST_FILE
                    });
                server_references.write(&server_references_path)?;
                println!(
                    "wrote {} ({} client reference(s) under this build's ids)",
                    server_references_path.display(),
                    server_references.entries.len(),
                );
                println!(
                    "emitted {}: {} server .mjs, {} .css, {} asset(s)",
                    summary.output_dir.display(),
                    summary.javascript_files,
                    summary.css_files,
                    summary.asset_files,
                );
                println!(
                    "  server graph gate (>= 35 .mjs): {} .mjs emitted -> {}",
                    summary.javascript_files,
                    if summary.javascript_files >= 35 {
                        "PASS"
                    } else {
                        "not yet"
                    }
                );
            }
            Ok(())
        }
        Some("bundle") => {
            let entry = arguments.next().ok_or_else(usage)?;
            let remaining = arguments.collect::<Vec<_>>();
            let output = remaining
                .first()
                .filter(|value| !value.to_string_lossy().starts_with("--"))
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("dist/bundle.js"));
            let flags = remaining;
            // A bare module entry has no framework config to consult, so `Auto` here
            // is off — the same default Rollup/esbuild use for a library bundle.
            let source_map =
                SourceMapChoice::from_flags(flags.iter().filter_map(|value| value.to_str()))?
                    .resolve(false);
            let minify = flags.iter().any(|value| value.to_str() == Some("--minify"));
            // `--format esm` emits real ES modules (top-level `import`/`export`,
            // native dynamic `import()`), where `import.meta` and top-level
            // `await` are representable. The default stays CommonJS.
            let format = match flags
                .iter()
                .position(|value| value.to_str() == Some("--format"))
            {
                Some(index) => match flags.get(index + 1).and_then(|value| value.to_str()) {
                    Some("esm") => diffpack::bundler::ModuleFormat::Esm,
                    Some("cjs") => diffpack::bundler::ModuleFormat::Cjs,
                    other => {
                        return Err(format!(
                            "--format must be `esm` or `cjs`, got {:?}",
                            other.unwrap_or("nothing")
                        ));
                    }
                },
                None => diffpack::bundler::ModuleFormat::Cjs,
            };
            let profile = env::var_os("DIFFPACK_PROFILE_FRONTEND").is_some();
            let discover_started = Instant::now();
            let (bundler, update) = diffpack::bundler::discover_direct_with_config(
                Path::new(&entry),
                &diffpack::bundler::BuildConfig {
                    source_maps: source_map,
                    ..Default::default()
                },
            )?;
            if profile {
                eprintln!(
                    "discover: {:.1} ms",
                    discover_started.elapsed().as_secs_f64() * 1000.0
                );
            }
            for warning in diffpack::bundler::partition_diagnostics(&update.diagnostics, "bundle")?
            {
                eprintln!("warning: {warning}");
            }
            let phase_started = Instant::now();
            let reachable = bundler.reachable_modules_direct();
            if profile {
                eprintln!(
                    "reachability: {:.1} ms",
                    phase_started.elapsed().as_secs_f64() * 1000.0
                );
            }
            let phase_started = Instant::now();
            bundler.emit_with_options(
                &reachable,
                &output,
                EmitOptions {
                    source_map,
                    minify,
                    format,
                    ..Default::default()
                },
            )?;
            if profile {
                eprintln!(
                    "emit: {:.1} ms",
                    phase_started.elapsed().as_secs_f64() * 1000.0
                );
            }
            println!(
                "bundled {} modules to {} (transformed {})",
                reachable.len(),
                output.display(),
                update.transformed_modules
            );
            Ok(())
        }
        Some("visualize") => {
            let entry = arguments.next().ok_or_else(usage)?;
            let output = arguments
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("diffpack-graph.html"));
            let (bundler, update) = diffpack::bundler::discover_direct(Path::new(&entry))?;
            for warning in
                diffpack::bundler::partition_diagnostics(&update.diagnostics, "visualization")?
            {
                eprintln!("warning: {warning}");
            }
            let reachable = bundler.reachable_modules_direct();
            let graph = bundler.visualization_graph(&reachable);
            diffpack_web::visualizer::write_visualization(&graph, &output)?;
            println!(
                "visualized {} modules and {} imports at {}",
                graph.nodes.len(),
                graph.edges.len(),
                output.display()
            );
            Ok(())
        }
        Some("visualize-scale") => {
            let modules = parse_usize(arguments.next(), 10_000, "module count")?;
            let imports = parse_usize(arguments.next(), 4, "imports per module")?;
            let output = arguments
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("target/diffpack-large-graph.html"));
            let (nodes, edges, reachable) =
                write_live_scale_visualization(modules, imports, &output)?;
            println!(
                "visualized {nodes} cached modules, {edges} imports, and {reachable} reachable modules at {}",
                output.display()
            );
            Ok(())
        }
        Some("watch") => {
            let entry = arguments.next().ok_or_else(usage)?;
            let output = arguments
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("dist/bundle.js"));
            watch_bundle(Path::new(&entry), &output)
        }
        Some("dev") => {
            let project_root = arguments.next().ok_or_else(usage)?;
            let remaining = arguments.collect::<Vec<_>>();
            let no_minify = remaining
                .iter()
                .any(|value| value.to_str() == Some("--no-minify"));
            // Dev source maps default ON so a runtime-error stack frame in the error
            // overlay resolves back to the original source; `--no-sourcemap` opts out.
            // (Production `build-app` with no flag follows the framework's own
            // policy per graph — for a Next app that is server maps always, browser
            // maps only under `productionBrowserSourceMaps`, which is what makes a
            // diffpack build comparable with `next build`. See
            // `next_adapter::default_source_maps`.)
            let source_map = !remaining
                .iter()
                .any(|value| value.to_str() == Some("--no-sourcemap"));
            // Optional explicit port; positional non-flag argument, default 3000.
            let port = remaining
                .iter()
                .find(|value| !value.to_string_lossy().starts_with("--"))
                .and_then(|value| value.to_str())
                .map(|value| {
                    value
                        .parse::<u16>()
                        .map_err(|error| format!("invalid dev port: {error}"))
                })
                .transpose()?
                .unwrap_or(3000);
            diffpack::dev_server::run(diffpack::dev_server::DevOptions {
                project_root: PathBuf::from(&project_root),
                port,
                minify: !no_minify,
                source_map,
            })
        }
        // Debug/oracle helpers: expose the RSC `"use server"` transforms and the
        // generated action resolver as RAW ESM on stdout (no other output), so an
        // oracle can exercise the exact Rust transform outputs against the real
        // `react-server-dom-webpack` runtime without a full app build. These print
        // node-runnable module source; they are inspection tools, not a build path.
        Some("rsc-transform") => {
            let file = arguments.next().ok_or_else(|| {
                "usage: diffpack rsc-transform <file> <client|server|client-ref>".to_string()
            })?;
            let which = arguments.next().ok_or_else(|| {
                "usage: diffpack rsc-transform <file> <client|server|client-ref>".to_string()
            })?;
            let path = Path::new(&file);
            let source = std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            let out = match which.to_str() {
                Some("client") => diffpack_next::rsc::transform_use_server_client(path, &source)
                    .ok_or_else(|| format!("{} is not a \"use server\" module", path.display()))?,
                Some("server") => diffpack_next::rsc::transform_use_server_server(path, &source)?
                    .ok_or_else(|| {
                    format!("{} is not a \"use server\" module", path.display())
                })?,
                // The REACT-SERVER-graph rewrite of a `"use client"` module: its
                // real code never reaches the server; each export becomes a client
                // reference the flight render serializes via the manifest.
                Some("client-ref") => diffpack_next::rsc::transform_use_client_server(
                    path, &source,
                )?
                .ok_or_else(|| format!("{} is not a \"use client\" module", path.display()))?,
                other => {
                    return Err(format!(
                        "unknown rsc-transform target {:?}; expected client|server|client-ref",
                        other
                    ));
                }
            };
            print!("{out}");
            Ok(())
        }
        // Debug/oracle helper: print the RSC SSR consumer manifest (Manifest #2)
        // derived natively from a build's emitted `client-references-manifest.json`
        // (Manifest #1), so an oracle's `createFromReadableStream` resolves the
        // client references in a flight stream through the same manifest diffpack
        // wires into the SSR build. Reads `<output-dir>/client-references-manifest.json`.
        Some("rsc-ssr-manifest") => {
            let output_dir = arguments.next().ok_or_else(|| {
                "usage: diffpack rsc-ssr-manifest <.diffpack-output dir>".to_string()
            })?;
            let manifest_path =
                Path::new(&output_dir).join(diffpack_next::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
            let manifest = diffpack_next::rsc::ClientReferencesManifest::read(&manifest_path)?;
            let value = manifest.to_ssr_consumer_manifest_json(None);
            print!(
                "{}",
                serde_json::to_string_pretty(&value)
                    .map_err(|error| format!("cannot serialize ssr consumer manifest: {error}"))?
            );
            Ok(())
        }
        Some("rsc-resolver") => {
            let root = arguments
                .next()
                .ok_or_else(|| "usage: diffpack rsc-resolver <project-root>".to_string())?;
            let entries = diffpack_next::rsc::scan_project_server_actions(Path::new(&root))?;
            print!(
                "{}",
                diffpack_next::rsc::generate_action_resolver_module(&entries)
            );
            Ok(())
        }
        // Conformance helper: for EACH Tailwind candidate class on stdin (one per
        // line), compile it in isolation against `@import 'tailwindcss'` and print one
        // NDJSON line `{"class":…,"ok":bool,"css":…,"error":…}`. Per-class isolation is
        // required because the compiler hard-errors on an unsupported utility, which
        // would otherwise abort a whole batch. Used by the Tailwind conformance harness
        // to compare diffpack's native compiler against Tailwind's own test suite.
        Some("tailwind") => {
            use std::io::Read as _;
            let mut input = String::new();
            std::io::stdin()
                .read_to_string(&mut input)
                .map_err(|error| format!("cannot read stdin: {error}"))?;
            for line in input.lines() {
                let class = line.trim();
                if class.is_empty() {
                    continue;
                }
                let mut set = std::collections::BTreeSet::new();
                set.insert(class.to_string());
                let value = match diffpack_default_loader::tailwind::compile(
                    "@import 'tailwindcss';\n",
                    &set,
                ) {
                    Ok(css) => serde_json::json!({ "class": class, "ok": true, "css": css }),
                    Err(error) => {
                        serde_json::json!({ "class": class, "ok": false, "error": error })
                    }
                };
                println!("{value}");
            }
            Ok(())
        }
        // Boot the production server for a `diffpack build-app <root> production`
        // output. Runs the emitted orchestrator (`next-server.mjs`) in PRODUCTION mode
        // (a persistent react-server worker pool — no per-request Node spawn, no dev
        // re-imports), serving the built app. For a non-Next (TanStack/SPA) output it
        // boots `server/index.mjs`.
        Some("start") => {
            let output_dir = arguments
                .next()
                .ok_or_else(|| "usage: diffpack start <.diffpack-output dir> [port]".to_string())?;
            let port = arguments
                .next()
                .and_then(|value| value.to_str().map(str::to_string))
                .unwrap_or_else(|| "3000".to_string());
            let out = Path::new(&output_dir);
            let next_server = out.join("next-server.mjs");
            let pages_server = out.join("pages-server.mjs");
            let entry = if next_server.exists() {
                next_server
            } else if pages_server.exists() {
                pages_server
            } else {
                out.join("server/index.mjs")
            };
            if !entry.exists() {
                return Err(format!(
                    "no production server entry in {} — run `diffpack build-app <root> production` first",
                    out.display()
                ));
            }
            // The next/image optimizer (`/_next/image`) in the emitted orchestrator
            // shells back to THIS binary for a native resize/re-encode the build did not
            // precompute (the `image` crate — no Node image dep). Hand it our path.
            let mut command = std::process::Command::new("node");
            command.arg(&entry).arg(out).arg(&port);
            if let Ok(exe) = std::env::current_exe() {
                command.env("DIFFPACK_BIN", exe);
            }
            // `next start` loads next.config in the process that serves, so whatever the
            // config puts in `process.env` — for cal.com its whole `.env`, `DATABASE_URL`
            // included — is part of the environment the app's server code runs under.
            // Diffpack evaluated the config in a build-time child, so the delta it
            // recorded is replayed here. Variables THIS process already defines are left
            // alone: that is what `dotenv.config()` itself does, and it keeps an explicit
            // `DATABASE_URL=… diffpack start` from being silently overridden by a value
            // baked at build time.
            for (key, value) in diffpack_next::next_adapter::config_env_from_output(out) {
                if std::env::var_os(&key).is_none() {
                    command.env(key, value);
                }
            }
            let status = command.status().map_err(|error| {
                format!("cannot start node server ({}): {error}", entry.display())
            })?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("production server exited with {status}"))
            }
        }
        // Native next/image runtime optimizer worker. The emitted next orchestrator
        // (`scripts/rsc/next-server.mjs`) shells to this only for a width/quality the
        // build did not precompute — it answers everything else straight from the
        // build-emitted variants, so prerendered/static pages never invoke it. Reads the source image bytes on
        // stdin, resizes (never upscales) to `--width`, re-encodes to `--format`
        // (`png`|`jpeg`) at `--quality`, and writes the optimized bytes to stdout.
        Some("optimize-image") => optimize_image(arguments.collect::<Vec<_>>()),
        _ => Err(usage()),
    }
}

/// The native next/image runtime optimizer: decode the source raster on stdin, resize
/// (downscale only — never upscale past the intrinsic width) to `--width`, re-encode to
/// `--format` (`png` preserves alpha; `jpeg` re-compresses at `--quality`), and stream the
/// optimized bytes to stdout. This is the on-the-fly path the emitted orchestrator shells
/// to for a REMOTE src, or a width/quality the build did not precompute; a `/public` or
/// static-import raster at a build-emitted width is served from disk instead.
fn optimize_image(args: Vec<std::ffi::OsString>) -> Result<(), String> {
    use std::io::{Read, Write};

    let mut width: Option<u32> = None;
    let mut quality: u8 = 75;
    let mut format = String::from("jpeg");
    let flags: Vec<String> = args
        .iter()
        .filter_map(|a| a.to_str().map(str::to_string))
        .collect();
    let mut it = flags.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--width" => {
                width = Some(
                    it.next()
                        .and_then(|v| v.parse::<u32>().ok())
                        .filter(|w| *w > 0)
                        .ok_or_else(|| {
                            "optimize-image: --width needs a positive integer".to_string()
                        })?,
                );
            }
            "--quality" => {
                quality = it
                    .next()
                    .and_then(|v| v.parse::<u8>().ok())
                    .filter(|q| *q >= 1 && *q <= 100)
                    .ok_or_else(|| {
                        "optimize-image: --quality needs an integer 1-100".to_string()
                    })?;
            }
            "--format" => {
                format = it
                    .next()
                    .cloned()
                    .ok_or_else(|| "optimize-image: --format needs png|jpeg".to_string())?;
            }
            other => return Err(format!("optimize-image: unknown flag {other}")),
        }
    }
    let width = width.ok_or_else(|| "optimize-image: --width is required".to_string())?;

    let mut input = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input)
        .map_err(|error| format!("optimize-image: cannot read stdin: {error}"))?;
    if input.is_empty() {
        return Err("optimize-image: empty input on stdin".to_string());
    }
    let decoded = image::load_from_memory(&input)
        .map_err(|error| format!("optimize-image: cannot decode input image: {error}"))?;
    let (w, h) = (decoded.width().max(1), decoded.height().max(1));
    // Downscale only: clamp the requested width to the intrinsic width so we never
    // upscale (matching Next's optimizer, which caps at the source dimensions).
    let target_w = width.min(w);
    let target_h = ((h as u64 * target_w as u64) / w as u64).max(1) as u32;
    let resized = if target_w == w {
        decoded
    } else {
        decoded.resize(target_w, target_h, image::imageops::FilterType::Lanczos3)
    };

    let mut out = Vec::new();
    match format.as_str() {
        "png" => {
            let mut cursor = std::io::Cursor::new(&mut out);
            resized
                .write_to(&mut cursor, image::ImageFormat::Png)
                .map_err(|error| format!("optimize-image: cannot encode png: {error}"))?;
        }
        "jpeg" | "jpg" => {
            let rgb = resized.to_rgb8();
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
            encoder
                .encode(
                    rgb.as_raw(),
                    rgb.width(),
                    rgb.height(),
                    image::ExtendedColorType::Rgb8,
                )
                .map_err(|error| format!("optimize-image: cannot encode jpeg: {error}"))?;
        }
        other => {
            return Err(format!(
                "optimize-image: unsupported --format {other} (supported: png, jpeg)"
            ));
        }
    }
    std::io::stdout()
        .write_all(&out)
        .map_err(|error| format!("optimize-image: cannot write stdout: {error}"))?;
    Ok(())
}

/// `diffpack build` — an HTML-rooted web build. Single-page by default (the
/// project's `index.html`); MULTI-PAGE when the Vite config sets
/// `build.rollupOptions.input` to several HTML entries. Every page is bundled into a
/// shared output dir (its entry chunk named `<input-name>.js`, its extracted
/// stylesheet `<input-name>.css`, assets deduped by content hash), stale files are
/// pruned once across all pages, and — when `build.manifest` is set — a Vite-shaped
/// `manifest.json` mapping each entry to its emitted files is written.
fn web_build(
    root: &Path,
    out_dir: Option<PathBuf>,
    vite: bool,
    minify: bool,
    source_map: bool,
) -> Result<(), String> {
    let mut config = diffpack::config::derive_web_config(root, vite)?;
    // See `build-app`: the per-module map is paid for only when maps are emitted.
    config.build.source_maps = source_map;

    // The page set: the configured `rollupOptions.input` HTML entries, or the
    // single-`index.html` default. A non-HTML input is a hard, specific error (a
    // bare JS/library entry belongs to `diffpack bundle`), never a silent skip.
    let pages: Vec<(String, PathBuf)> = if config.inputs.is_empty() {
        let html = root.join("index.html");
        if !html.is_file() {
            return Err(format!(
                "{} has no index.html; `diffpack build` bundles an HTML-rooted web app \
                 (use `diffpack bundle <entry>` for a bare module entry, or set \
                 build.rollupOptions.input for a multi-page build)",
                root.display()
            ));
        }
        vec![("index".to_string(), html)]
    } else {
        let mut pages = Vec::new();
        for (name, path) in &config.inputs {
            if path.extension().and_then(|ext| ext.to_str()) != Some("html") {
                return Err(format!(
                    "build.rollupOptions.input `{name}` -> {} is not an .html entry; \
                     diffpack's multi-page build takes HTML inputs (a bare JS/library \
                     entry is `diffpack bundle <entry>`)",
                    path.display()
                ));
            }
            if !path.is_file() {
                return Err(format!(
                    "build.rollupOptions.input `{name}` -> {} does not exist",
                    path.display()
                ));
            }
            pages.push((name.clone(), path.clone()));
        }
        pages
    };

    // Vite resolves `build.outDir` against the project root, and so does diffpack:
    // an explicit relative `--out-dir` is root-relative, exactly like the `dist`
    // default. `Path::join` with an absolute argument yields that path unchanged,
    // so an absolute `--out-dir` still lands where the caller asked.
    //
    // Precedence, highest first: the `--out-dir` argument, then the project's
    // `vite.config` `build.outDir` (already root-resolved by `derive_web_config`),
    // then Vite's `dist` default. An explicit command-line argument always beats a
    // config file.
    let out_dir = match out_dir {
        Some(out_dir) => root.join(out_dir),
        None => config.out_dir.clone().unwrap_or_else(|| root.join("dist")),
    };
    let emit_options = EmitOptions {
        minify,
        source_map,
        ..EmitOptions::default()
    };
    println!(
        "web build{}: {} page(s) -> {}",
        if config.vite { " (vite mode)" } else { "" },
        pages.len(),
        out_dir.display(),
    );

    let mut written = BTreeSet::new();
    let mut manifest_pages = Vec::new();
    let mut total_js = 0usize;
    let mut total_css = 0usize;
    for (name, html_path) in &pages {
        let html = diffpack_web::html_entry::parse_file(html_path)?;
        let html_origin = html_path.display().to_string();
        // Each HTML page carries exactly one local module-script entry (an inline or
        // multi-entry document is a hard error naming the file).
        let entry_script = match html.module_scripts.as_slice() {
            [only] => only,
            [] => {
                return Err(format!(
                    "{html_origin}: no local <script type=\"module\" src> entry found"
                ));
            }
            many => {
                let sources = many
                    .iter()
                    .map(|script| script.src.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "{html_origin}: {} module script entries ({sources}); a single HTML \
                     page must have exactly one module-script entry",
                    many.len()
                ));
            }
        };
        // A root-absolute src (`/src/main.tsx`) is project-root-relative; anything
        // else is relative to the HTML document itself.
        let entry_path = match entry_script.src.strip_prefix('/') {
            Some(rest) => root.join(rest),
            None => html_path
                .parent()
                .expect("an HTML file has a parent directory")
                .join(&entry_script.src),
        };
        let entry = entry_path.canonicalize().map_err(|error| {
            format!(
                "{html_origin}: module script src \"{}\" does not resolve ({}: {error})",
                entry_script.src,
                entry_path.display()
            )
        })?;

        let (bundler, update) = diffpack::bundler::discover_web_with_config(&entry, &config.build)?;
        // A web build fails on unresolved imports: an artifact with dangling
        // references is not a successful build.
        for warning in diffpack::bundler::partition_diagnostics(
            &update.diagnostics,
            &format!("page `{name}` ({html_origin})"),
        )? {
            eprintln!("warning: {warning}");
        }
        let reachable = bundler.reachable_modules_direct();

        let entry_file = format!("{name}.js");
        let (summary, page_written) =
            bundler.emit_web_written(&reachable, &out_dir, &entry_file, emit_options)?;
        written.extend(page_written);
        total_js += 1;

        // Whether THIS page produced an extracted stylesheet (`<name>.css`).
        let css_file = format!("{name}.css");
        let has_css = out_dir.join(&css_file).is_file();
        if has_css {
            total_css += 1;
        }

        let mut injection = diffpack_web::html_entry::HeadInjection {
            script_urls: vec![format!("{}{entry_file}", config.base)],
            stylesheet_urls: Vec::new(),
        };
        if has_css {
            injection
                .stylesheet_urls
                .push(format!("{}{css_file}", config.base));
        }
        let built_html = diffpack_web::html_entry::apply_base(
            &html.rewrite(&html_origin, &injection)?,
            &config.base,
        );
        // The output document keeps the source HTML's own file name.
        let html_name = html_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("{html_origin}: HTML path is not UTF-8"))?;
        let html_out = out_dir.join(html_name);
        std::fs::write(&html_out, built_html)
            .map_err(|error| format!("cannot write {}: {error}", html_out.display()))?;
        written.insert(html_out.clone());

        // The manifest key is the entry HTML's path relative to the project root.
        let key = html_path
            .strip_prefix(root)
            .unwrap_or(html_path)
            .to_string_lossy()
            .replace('\\', "/");
        let src = entry
            .strip_prefix(root)
            .ok()
            .map(|rel| rel.to_string_lossy().replace('\\', "/"));
        manifest_pages.push(diffpack_vite_compat::vite_manifest::PageRecord {
            key,
            file: entry_file,
            css: if has_css { vec![css_file] } else { Vec::new() },
            src,
        });
        let _ = summary;
    }

    // Prune once across every page's written set (a shared asset written by one page
    // is never deleted by another), removing only stale files from a prior build.
    diffpack_default_loader::output::prune_output(&out_dir, &written)?;

    // The `public/` passthrough directory is a Vite convention.
    let static_files = if config.vite {
        diffpack::config::copy_static_public(root, &out_dir)?
    } else {
        0
    };

    // The Vite build manifest (`build.manifest`).
    let mut manifest_note = String::new();
    if config.emit_manifest {
        let manifest = diffpack_vite_compat::vite_manifest::render(&manifest_pages);
        let manifest_path = out_dir.join(&config.manifest_name);
        if let Some(parent) = manifest_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        std::fs::write(&manifest_path, manifest)
            .map_err(|error| format!("cannot write {}: {error}", manifest_path.display()))?;
        manifest_note = format!(", manifest {}", config.manifest_name);
    }

    // `optimizeDeps.exclude`: Diffpack bundles every dependency natively (no
    // pre-bundle step), so an exclusion is satisfied by construction — reported so it
    // is never silently ignored.
    if !config.optimize_deps_exclude.is_empty() {
        println!(
            "optimizeDeps.exclude: {} dep(s) ({}) — diffpack does not pre-bundle, so exclusion is inherent",
            config.optimize_deps_exclude.len(),
            config.optimize_deps_exclude.join(", "),
        );
    }

    println!(
        "emitted {}: {} page(s), {total_js} entry .js, {total_css} .css, {} static file(s){manifest_note}",
        out_dir.display(),
        pages.len(),
        static_files,
    );
    Ok(())
}

/// Recursively copy `src` into `dest` (used to publish the react-server bundle from
/// `server/` to `rsc-render/` before the ssr build overwrites `server/`).
fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest)
        .map_err(|error| format!("cannot create {}: {error}", dest.display()))?;
    for entry in
        std::fs::read_dir(src).map_err(|error| format!("cannot read {}: {error}", src.display()))?
    {
        let entry =
            entry.map_err(|error| format!("cannot read entry in {}: {error}", src.display()))?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        let kind = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", from.display()))?;
        if kind.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).map_err(|error| {
                format!(
                    "cannot copy {} -> {}: {error}",
                    from.display(),
                    to.display()
                )
            })?;
        }
    }
    Ok(())
}

/// The pages-router per-environment build (`build-app <root> client|ssr`). Classic
/// (non-RSC) React SSR: the `client` environment emits the browser hydration bundle
/// to `public/`; any other environment emits the Node ESM SSR bundle to
/// `server/server.mjs` (whose `handleRequest` the pages orchestrator drives). The
/// `production` one-command build (`build_production`) chains client -> ssr and
/// writes the orchestrator.
fn build_pages_app(
    project_root: &Path,
    environment: &str,
    minify: bool,
    source_map: SourceMapChoice,
) -> Result<(), String> {
    let mut config = diffpack::config::configure_next_pages(project_root, environment, false)?
        .ok_or_else(|| {
            "next pages-router configure returned None for a pages project".to_string()
        })?;
    // See `build-app`: the per-module map is paid for only when maps are emitted, and
    // with no CLI flag the framework default the adapter chose stands.
    config.build.source_maps = source_map.resolve(config.build.source_maps);
    let source_map = config.build.source_maps;
    println!(
        "next pages-router adapter: scaffolded {} for environment={environment}",
        diffpack_next::next_pages::ADAPTER_DIR,
    );
    let entry = config
        .entry
        .clone()
        .ok_or_else(|| format!("no {environment} entry found for the pages app"))?;
    let output_root = project_root.join(".diffpack-output");

    println!(
        "pages app: environment={} ({} aliases), entry={}",
        config.environment,
        config.build.aliases.len(),
        entry.display(),
    );
    let (bundler, update) = diffpack::bundler::discover_next_with_config(&entry, &config.build)?;
    let warnings = diffpack::bundler::partition_diagnostics(
        &update.diagnostics,
        &format!("pages {} build", config.environment),
    )?;
    let reachable = bundler.reachable_modules_direct();
    println!(
        "reachable {} modules; {} warning(s)",
        reachable.len(),
        warnings.len()
    );
    for warning in &warnings {
        println!("  warning: {warning}");
    }

    let emit_options = EmitOptions {
        minify,
        source_map,
        ..EmitOptions::default()
    };
    if config.environment == "client" {
        let summary = bundler.emit_public(&reachable, &output_root, emit_options)?;
        let static_files = diffpack::config::copy_static_public(project_root, &summary.output_dir)?;
        println!(
            "emitted {}: {} public .js, {} .css, {} asset(s), {} static file(s)",
            summary.output_dir.display(),
            summary.javascript_files,
            summary.css_files,
            summary.asset_files,
            static_files,
        );
    } else {
        let summary = bundler.emit_server(&reachable, &output_root, emit_options)?;
        println!(
            "emitted {}: {} server .mjs, {} .css, {} asset(s)",
            summary.output_dir.display(),
            summary.javascript_files,
            summary.css_files,
            summary.asset_files,
        );
    }
    Ok(())
}

/// Run the pages-router SSG prerenderer against a completed `.diffpack-output`: writes
/// the `pages-prerender.mjs` driver, spawns node on it (the app's own bundled React —
/// the same oracle the orchestrator uses; the graph bundling stayed native Rust), and
/// leaves `prerender.json` next to the SSR bundle. A prerender failure fails the build
/// (never silently skipped).
fn pages_prerender(output_root: &Path) -> Result<(), String> {
    let driver = output_root.join("pages-prerender.mjs");
    std::fs::write(&driver, diffpack_next::next_pages::PRERENDER_DRIVER)
        .map_err(|error| format!("cannot write {}: {error}", driver.display()))?;
    let status = std::process::Command::new("node")
        .arg(&driver)
        .arg(output_root)
        .status()
        .map_err(|error| format!("cannot spawn node for the pages SSG prerenderer: {error}"))?;
    if !status.success() {
        return Err(format!(
            "the pages SSG prerenderer (node {}) failed with {status}",
            driver.display()
        ));
    }
    Ok(())
}

/// `diffpack build-app <root> production` — the one-command production build. Builds
/// every graph in order (self-invoking the per-environment build so each runs exactly
/// as it does standalone) and assembles a single deployable output.
fn build_production(project_root: &Path, flags: &[std::ffi::OsString]) -> Result<(), String> {
    let exe = std::env::current_exe()
        .map_err(|error| format!("cannot locate the diffpack binary: {error}"))?;
    let root = project_root.to_string_lossy().to_string();
    let passthrough: Vec<String> = flags
        .iter()
        .filter_map(|f| f.to_str())
        .filter(|f| f.starts_with("--") && *f != "--static-export")
        .map(str::to_string)
        .collect();
    // Rejects a contradictory pair here, before any child runs, instead of letting
    // three children fail one after another.
    let source_map_choice = SourceMapChoice::from_flags(flags.iter().filter_map(|f| f.to_str()))?;
    let run = |environment: &str| -> Result<(), String> {
        // Each environment is a child process with its own stage table; this stage is
        // the parent's view of it, so the production table always accounts for the
        // whole wall clock even though the detail lives in the children.
        let _stage = match environment {
            "client" => diffpack_core::build_profile::stage("build/client"),
            "react-server" => diffpack_core::build_profile::stage("build/react-server"),
            "ssr" => diffpack_core::build_profile::stage("build/ssr"),
            _ => diffpack_core::build_profile::stage("build/other"),
        };
        let status = std::process::Command::new(&exe)
            .arg("build-app")
            .arg(&root)
            .arg(environment)
            .args(&passthrough)
            .status()
            .map_err(|error| format!("cannot run build-app {environment}: {error}"))?;
        if !status.success() {
            return Err(format!("build-app {environment} failed ({status})"));
        }
        Ok(())
    };
    let out = project_root.join(".diffpack-output");
    if !diffpack_next::next_adapter::is_app_router(project_root)
        && diffpack_next::next_pages::is_pages_router(project_root)
    {
        println!("=== production build (next pages-router): client -> ssr ===");
        run("client")?;
        run("ssr")?;
        // SSG prerender: run every getStaticProps (and getStaticPaths) page ONCE at
        // build time and write `prerender.json`; the orchestrator seeds its ISR cache
        // from it so static pages are served with zero per-request data fetch.
        println!("=== prerender (SSG getStaticProps) ===");
        pages_prerender(&out)?;
        // Emit the pages orchestrator (`pages-server.mjs`): plain Node that imports
        // the SSR bundle's `handleRequest` and serves the client `public/` assets.
        std::fs::write(
            out.join("pages-server.mjs"),
            diffpack_next::next_pages::ORCHESTRATOR,
        )
        .map_err(|error| format!("cannot write pages production server: {error}"))?;
        println!(
            "\nproduction build complete -> {}\n  serve it:  diffpack start {} [port]",
            out.display(),
            out.display()
        );
        return Ok(());
    }
    if diffpack_next::next_adapter::is_app_router(project_root) {
        println!("=== production build (next app-router): client -> (react-server || ssr) ===");
        // The react-server and ssr builds depend only on the CLIENT build's two
        // manifests — pure graph facts the client publishes (atomically, via
        // rename) BEFORE its emit — never on the client's emitted bytes or on
        // each other's output. So all three overlap: the client build streams
        // its logs live; the moment both manifests exist the other two spawn,
        // react-server emitting straight into `rsc-render/` (`--server-dir`,
        // which also retires the publish-copy that used to sit between them)
        // while ssr owns `server/`. Their outputs are captured and replayed
        // after the client logs so the three builds' reporting never
        // interleaves. Stale manifests from an earlier build are removed first,
        // or the server builds would launch against the previous graph.
        let rsc_render = out.join("rsc-render");
        let client_manifest_path = out.join(diffpack_tanstack::manifest::CLIENT_MANIFEST_FILE);
        let client_references_path = out.join(diffpack_next::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
        let _ = std::fs::remove_file(&client_manifest_path);
        let _ = std::fs::remove_file(&client_references_path);
        let client_stage = diffpack_core::build_profile::stage("build/client");
        let mut client_child = std::process::Command::new(&exe)
            .arg("build-app")
            .arg(&root)
            .arg("client")
            .args(&passthrough)
            .spawn()
            .map_err(|error| format!("cannot run build-app client: {error}"))?;
        // Wait for the manifests (or for the client to fail first).
        loop {
            if client_manifest_path.is_file() && client_references_path.is_file() {
                break;
            }
            match client_child.try_wait() {
                Ok(Some(status)) if !status.success() => {
                    return Err(format!("build-app client failed ({status})"));
                }
                Ok(Some(_)) => break,
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(5)),
                Err(error) => {
                    return Err(format!("cannot wait for build-app client: {error}"));
                }
            }
        }
        let rsc_stage = diffpack_core::build_profile::stage("build/react-server");
        let rsc_child = std::process::Command::new(&exe)
            .arg("build-app")
            .arg(&root)
            .arg("react-server")
            .arg("--server-dir=rsc-render")
            .args(&passthrough)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|error| format!("cannot run build-app react-server: {error}"))?;
        let ssr_stage = diffpack_core::build_profile::stage("build/ssr");
        let ssr_child = std::process::Command::new(&exe)
            .arg("build-app")
            .arg(&root)
            .arg("ssr")
            .args(&passthrough)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|error| format!("cannot run build-app ssr: {error}"))?;
        // Drain both captured children on their own threads so neither can stall
        // on a full pipe while the parent is still waiting for the client.
        let rsc_thread = std::thread::spawn(move || rsc_child.wait_with_output());
        let ssr_thread = std::thread::spawn(move || ssr_child.wait_with_output());
        let client_status = client_child
            .wait()
            .map_err(|error| format!("cannot wait for build-app client: {error}"))?;
        drop(client_stage);
        // Every child is always reaped (and its logs always surfaced), even when
        // an earlier one failed — a failure report that swallows another build's
        // output would hide half the story.
        let rsc_output = rsc_thread
            .join()
            .map_err(|_| "the react-server drain thread panicked".to_string())?
            .map_err(|error| format!("cannot wait for build-app react-server: {error}"))?;
        drop(rsc_stage);
        let ssr_output = ssr_thread
            .join()
            .map_err(|_| "the ssr drain thread panicked".to_string())?
            .map_err(|error| format!("cannot wait for build-app ssr: {error}"))?;
        drop(ssr_stage);
        print!("{}", String::from_utf8_lossy(&rsc_output.stdout));
        eprint!("{}", String::from_utf8_lossy(&rsc_output.stderr));
        print!("{}", String::from_utf8_lossy(&ssr_output.stdout));
        eprint!("{}", String::from_utf8_lossy(&ssr_output.stderr));
        if !client_status.success() {
            return Err(format!("build-app client failed ({client_status})"));
        }
        if !rsc_output.status.success() {
            return Err(format!(
                "build-app react-server failed ({})",
                rsc_output.status
            ));
        }
        if !ssr_output.status.success() {
            return Err(format!("build-app ssr failed ({})", ssr_output.status));
        }
        // Publish the react-server graph's emitted assets (content-hashed images
        // and their build-emitted responsive variants from static image imports)
        // into the SERVED `public/assets/`. A static image import
        // (`import img from './x.png'`) is referenced only by Server Components, so
        // its variants are emitted in the react-server build, not the client one;
        // merging them here (before the prerender copies `public/` -> `static/`)
        // makes the `<img>`'s `/assets/...` srcset URLs resolve. Names are
        // content-hashed, so this is a copy of new files only (zero per-request
        // cost, no image server). A no-op when the graph emitted no assets.
        let rsc_assets = rsc_render.join("assets");
        if rsc_assets.is_dir() {
            copy_dir_recursive(&rsc_assets, &out.join("public/assets"))?;
        }
        std::fs::write(
            out.join("next-server.mjs"),
            include_str!("../scripts/rsc/next-server.mjs"),
        )
        .map_err(|error| format!("cannot write production server: {error}"))?;
        write_ssr_module_map(&out)?;
        // instrumentation.{ts,js}: the app's boot hook (register() runs once at server
        // startup — OpenTelemetry/Sentry-style). Write the generated boot-entry wrapper
        // (which CALLS register at module load) then bundle it NATIVELY (self-invoke our
        // own `bundle` subcommand, ESM) to <out>/instrumentation.mjs; the orchestrator
        // dynamic-imports it once before listen (see next-server.mjs). Build-time only,
        // zero per-request cost.
        if let Some(wrapper) =
            diffpack_next::next_adapter::write_instrumentation_wrapper(project_root)?
        {
            println!("=== instrumentation (register() boot hook) ===");
            let instr_out = out.join("instrumentation.mjs");
            // `instrumentation.mjs` runs in the SERVER process — it is exactly the code
            // whose stack traces are unreadable without a map (OpenTelemetry/Sentry
            // boot hooks), and it never reaches a browser. The `bundle` subcommand has
            // no framework config to consult, so the server policy resolved here is
            // passed explicitly; without it this was the one server artifact of a
            // `--sourcemap` build that shipped with no map at all.
            let instrumentation_maps = source_map_choice.resolve(true);
            let status = std::process::Command::new(&exe)
                .arg("bundle")
                .arg(&wrapper)
                .arg(&instr_out)
                .arg("--format")
                .arg("esm")
                .args(instrumentation_maps.then_some("--sourcemap"))
                .status()
                .map_err(|error| {
                    format!(
                        "cannot bundle instrumentation ({}): {error}",
                        wrapper.display()
                    )
                })?;
            if !status.success() {
                return Err(format!(
                    "bundling instrumentation ({}) failed ({status})",
                    wrapper.display(),
                ));
            }
        }
        // Prerender static / SSG / ISR routes so the orchestrator serves them from the
        // cache (instant, no per-request render) instead of rendering every request.
        // Dynamic routes are recorded, never dropped. A prerender failure fails the
        // build (naming the route).
        println!("=== prerender (static / SSG / ISR) ===");
        next_prerender(project_root, &out, false)?;
        println!(
            "\nproduction build complete -> {}\n  serve it:  diffpack start {} [port]",
            out.display(),
            out.display()
        );
    } else {
        println!("=== production build: client -> ssr ===");
        run("client")?;
        run("ssr")?;
        println!(
            "\nproduction build complete -> {}\n  serve it:  diffpack start {} [port]  (node {}/server/index.mjs)",
            out.display(),
            out.display(),
            out.display()
        );
    }
    Ok(())
}

fn usage() -> String {
    "usage: diffpack build <project-root> [--vite] [--out-dir <dir>] [--no-minify] [--sourcemap] | diffpack preview <build-dir> [port] | diffpack build-app <project-root> [client|react-server|ssr|nitro|static] [--no-minify] [--sourcemap|--no-sourcemap] [--static-export] | diffpack dev <project-root> [port] [--no-minify] [--no-sourcemap] | diffpack bundle <entry> [output] [--sourcemap] [--minify] [--format esm|cjs] | diffpack visualize <entry> [output.html] | diffpack visualize-scale [modules] [imports-per-module] [output.html] | diffpack watch <entry> [output] | diffpack bundle-scale-direct [modules] [imports-per-module] | diffpack bundle-scale-direct-deps [modules] [imports-per-module] | diffpack bundle-scale-direct-live [modules] [imports-per-module] | diffpack bundle-scale-direct-live-deps [modules] [imports-per-module]".into()
}

fn print_bundle_scale(result: diffpack::bundle_benchmark::BundleScaleResult, mode: &str) {
    println!(
        "mode,frontend_threads,modules,edges,initial_reachable,final_reachable,source_mb,bundle_mb,bundle_bytes,runtime_value,generate_ms,discover_transform_resolve_ms,initial_reachability_ms,initial_emit_ms,edit_transform_resolve_ms,edit_reachability_ms,edit_emit_ms,transformed_on_edit,read_cpu_ms,transform_cpu_ms,lower_cpu_ms,resolve_cpu_ms"
    );
    println!(
        "{},{},{},{},{},{},{:.3},{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3}",
        mode,
        result.worker_threads,
        result.modules,
        result.generated_edges,
        result.initial_reachable,
        result.final_reachable,
        result.source_bytes as f64 / 1_000_000.0,
        result.bundle_bytes as f64 / 1_000_000.0,
        result.bundle_bytes,
        result
            .runtime_value
            .map_or_else(String::new, |value| value.to_string()),
        result.generate_ms,
        result.discover_transform_resolve_ms,
        result.initial_reachability_ms,
        result.initial_emit_ms,
        result.edit_transform_resolve_ms,
        result.edit_reachability_ms,
        result.edit_emit_ms,
        result.transformed_on_edit,
        result.frontend_read_cpu_ms,
        result.frontend_transform_cpu_ms,
        result.frontend_lower_cpu_ms,
        result.frontend_resolve_cpu_ms,
    );
}

/// The embedded SSG node scripts (guest code kept in REAL files under scripts/rsc/,
/// embedded so the built binary is self-contained; written next to the output so the
/// sibling `import "./next-render-core.mjs"` resolves regardless of cwd).
const NEXT_RENDER_CORE_MJS: &str = include_str!("../scripts/rsc/next-render-core.mjs");
const NEXT_PRERENDER_MJS: &str = include_str!("../scripts/rsc/next-prerender.mjs");

/// The manifest-join module both the orchestrator (`next-server.mjs`) and the render
/// seam (`next-render-core.mjs`) import as a SIBLING, so it must be written next to
/// whichever of them lands in the output dir. Shared, not duplicated, so the rule for
/// which client references are resolvable is stated in exactly one place.
pub(crate) fn write_ssr_module_map(output_root: &Path) -> Result<(), String> {
    let path = output_root.join(diffpack_next::rsc::SSR_MODULE_MAP_FILE);
    std::fs::write(&path, include_str!("../scripts/rsc/ssr-module-map.mjs"))
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

/// `diffpack build-app <root> static` — the SSG prerender phase. Builds NO graph:
/// reuses the three already-emitted bundles + manifests, re-runs native route
/// classification to write the prerender plan, then spawns the node prerenderer (the
/// app's own React runtime — the explicitly-allowed oracle) to write `.html` + `.rsc`
/// for every static/SSG route. Dynamic routes are recorded, never dropped.
fn build_static(project_root: &Path, static_export: bool) -> Result<(), String> {
    let output_root = project_root.join(".diffpack-output");

    // Hard-error (naming which) if any of the five inputs is missing — mirrors the
    // orchestrator's fail() checks. The SSG render reuses these verbatim.
    for (label, rel) in [
        ("react-server render bundle", "rsc-render/server.mjs"),
        ("SSR bundle", "server/server.mjs"),
        (
            "client-references manifest",
            diffpack_next::rsc::CLIENT_REFERENCES_MANIFEST_FILE,
        ),
        (
            "react-server-references manifest",
            diffpack_next::rsc::REACT_SERVER_REFERENCES_MANIFEST_FILE,
        ),
        (
            "ssr-references manifest",
            diffpack_next::rsc::SERVER_REFERENCES_MANIFEST_FILE,
        ),
    ] {
        let p = output_root.join(rel);
        if !p.exists() {
            return Err(format!(
                "{label} not found at {} — run the client -> react-server (cp -> rsc-render) -> ssr builds first",
                p.display(),
            ));
        }
    }

    next_prerender(project_root, &output_root, static_export)?;
    println!(
        "next SSG: prerendered static routes -> {}",
        output_root.join("static").display(),
    );
    Ok(())
}

/// Classify every app-router route and prerender the static / SSG / ISR ones to
/// `<output_root>/static/*.html` + `*.rsc` (+ `prerender-manifest.json`). Shared by
/// `build-app static` and `build-app production` — the latter serves these from a cache
/// (with ISR revalidation) instead of rendering them per request. Assumes the client /
/// react-server (`rsc-render/`) / ssr (`server/`) bundles are already built.
fn next_prerender(
    project_root: &Path,
    output_root: &Path,
    static_export: bool,
) -> Result<(), String> {
    // Native route classification -> the machine-readable prerender plan.
    let plan_stage = diffpack_core::build_profile::stage("prerender/classify-routes");
    let route_count = diffpack_next::next_adapter::write_prerender_plan(project_root, output_root)?;
    drop(plan_stage);
    println!(
        "next SSG/ISR: classified {route_count} route(s) -> {}",
        output_root.join("static/prerender-plan.json").display(),
    );

    // Materialize the embedded node scripts next to the output.
    let core_path = output_root.join("next-render-core.mjs");
    let prerender_path = output_root.join("next-prerender.mjs");
    std::fs::write(&core_path, NEXT_RENDER_CORE_MJS)
        .map_err(|error| format!("cannot write {}: {error}", core_path.display()))?;
    std::fs::write(&prerender_path, NEXT_PRERENDER_MJS)
        .map_err(|error| format!("cannot write {}: {error}", prerender_path.display()))?;
    write_ssr_module_map(output_root)?;

    // Spawn the prerenderer (the app's own React runtime; the bundling stays native
    // Rust). Its stdout/stderr stream straight through; a nonzero exit fails the build.
    let mut command = std::process::Command::new("node");
    command.arg(&prerender_path).arg(output_root);
    // The environment evaluating `next.config` produced. `next build` prerenders inside
    // the process that loaded the config, so a route's `getStaticProps`/server component
    // sees whatever the config put in `process.env` (for cal.com, its entire `.env`).
    command.envs(diffpack_next::next_adapter::config_env_from_manifest(
        project_root,
    ));
    if static_export {
        command.arg("--static-export");
    }
    let render_stage = diffpack_core::build_profile::stage("prerender/render-routes");
    let status = command
        .status()
        .map_err(|error| format!("cannot spawn node for the SSG prerenderer: {error}"))?;
    drop(render_stage);
    if !status.success() {
        return Err(format!(
            "the SSG prerenderer (node {}) failed with {status}",
            prerender_path.display(),
        ));
    }
    Ok(())
}

fn watch_bundle(entry: &Path, output: &Path) -> Result<(), String> {
    let (mut bundler, initial) = diffpack::bundler::discover_direct(entry)?;
    // The initial build is a hard error: there is nothing worth watching over a
    // graph that cannot produce a loadable artifact.
    let initial_warnings = diffpack::bundler::partition_diagnostics(&initial.diagnostics, "watch")?;
    let mut session = bundler.direct_reachability();
    let mut reachable = session.reachable_modules();
    bundler.emit(&reachable, output)?;
    println!(
        "watching {} ({} modules); wrote {}",
        bundler.watch_root().display(),
        reachable.len(),
        output.display()
    );
    for warning in initial_warnings {
        eprintln!("warning: {warning}");
    }

    let (events, receiver) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = events.send(event);
    })
    .map_err(|error| format!("cannot create filesystem watcher: {error}"))?;
    watcher
        .watch(&bundler.watch_root(), RecursiveMode::Recursive)
        .map_err(|error| format!("cannot start filesystem watcher: {error}"))?;

    loop {
        let event = receiver
            .recv()
            .map_err(|_| "filesystem watcher stopped".to_string())?
            .map_err(|error| format!("filesystem watch error: {error}"))?;
        // `rebuild_started` is stamped AFTER the OS watcher delivered the event, so
        // `rebuild=<ms>` below is the pure in-process rebuild (read + transform +
        // reachability + emit) and EXCLUDES OS change-detection latency — the
        // apples-to-apples equivalent of esbuild's `context.rebuild()` and
        // rolldown's `event.duration`, which likewise measure only the rebuild.
        let rebuild_started = Instant::now();
        let paths = event.paths.into_iter().collect::<BTreeSet<_>>();
        for path in paths {
            if !is_module_path(&path) {
                continue;
            }
            let update = bundler.rebuild_path(&path)?;
            if update.delta.edge_updates.is_empty() && update.delta.changed.is_empty() {
                continue;
            }
            let reachability_started = Instant::now();
            let result = session.apply(&update.delta);
            let reachability_ms = reachability_started.elapsed().as_secs_f64() * 1_000.0;
            for module in result.removed {
                reachable.remove(&module);
            }
            reachable.extend(result.added);
            // A bad edit must not kill the watcher, and must not overwrite the last
            // good artifact with a broken one: report it, skip the emit, and keep
            // watching so the next save can fix it.
            let warnings =
                match diffpack::bundler::partition_diagnostics(&update.diagnostics, "rebuild") {
                    Ok(warnings) => warnings,
                    Err(error) => {
                        eprintln!("error: {error}");
                        continue;
                    }
                };
            bundler.emit(&reachable, output)?;
            let rebuild_ms = rebuild_started.elapsed().as_secs_f64() * 1_000.0;
            println!(
                "rebuilt {}: reachable={} transformed={} reachability={:.3}ms rebuild={:.3}ms",
                path.display(),
                reachable.len(),
                update.transformed_modules,
                reachability_ms,
                rebuild_ms
            );
            for warning in warnings {
                eprintln!("warning: {warning}");
            }
        }
    }
}

fn is_module_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "json")
    )
}

fn parse_usize(
    value: Option<std::ffi::OsString>,
    default: usize,
    description: &str,
) -> Result<usize, String> {
    value.map_or(Ok(default), |value| {
        value
            .to_string_lossy()
            .parse()
            .map_err(|error| format!("invalid {description}: {error}"))
    })
}
