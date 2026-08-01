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

    fn override_value(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::On => Some(true),
            Self::Off => Some(false),
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
            // `productionBrowserSourceMaps` / `experimental.serverSourceMaps`
            // (see `next_adapter::default_source_maps`),
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
            let references_only = remaining
                .iter()
                .any(|value| value.to_str() == Some("--references-only"));

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
                return diffpack_next::production::build_static(
                    Path::new(&project_root),
                    static_export,
                );
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
                return diffpack_next::production::build_pages_environment(
                    Path::new(&project_root),
                    &environment,
                    minify,
                    source_map.override_value(),
                );
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

            if is_next_app {
                diffpack_next::profile::prepare_build(
                    Path::new(&project_root),
                    &output_root,
                    &mut config,
                )?;
            } else {
                diffpack_tanstack::profile::prepare_build(
                    Path::new(&project_root),
                    &output_root,
                    &mut config,
                )?;
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
            if is_next_app {
                if references_only {
                    diffpack_next::profile::emit_server_references(
                        &output_root,
                        &config,
                        &bundler,
                        &reachable,
                    )?;
                } else {
                    diffpack_next::profile::emit_build(
                        Path::new(&project_root),
                        &output_root,
                        &config,
                        &bundler,
                        &reachable,
                        emit_options,
                        &server_dir_name,
                    )?;
                }
            } else {
                diffpack_tanstack::profile::emit_build(
                    Path::new(&project_root),
                    &output_root,
                    &config,
                    &bundler,
                    &reachable,
                    emit_options,
                    &server_dir_name,
                )?;
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
    let vite_profile = if vite {
        Some(diffpack::config::derive_vite_web_config(root)?)
    } else {
        None
    };
    let mut config = match &vite_profile {
        Some(profile) => profile.web.clone(),
        None => diffpack::config::derive_web_config(root)?,
    };
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
        if vite { " (vite mode)" } else { "" },
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
        manifest_pages.push(diffpack_web::config::EmittedPage {
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
    let static_files = if vite_profile
        .as_ref()
        .is_some_and(|profile| profile.copy_public_dir)
    {
        diffpack::config::copy_static_public(root, &out_dir)?
    } else {
        0
    };

    // The Vite build manifest (`build.manifest`).
    let mut manifest_note = String::new();
    if let Some(manifest_name) = vite_profile
        .as_ref()
        .map(|profile| profile.write_manifest(&out_dir, &manifest_pages))
        .transpose()?
        .flatten()
    {
        manifest_note = format!(", manifest {manifest_name}");
    }

    // `optimizeDeps.exclude`: Diffpack bundles every dependency natively (no
    // pre-bundle step), so an exclusion is satisfied by construction — reported so it
    // is never silently ignored.
    let optimize_deps_exclude = vite_profile
        .as_ref()
        .map(|profile| profile.optimize_deps_exclude.as_slice())
        .unwrap_or_default();
    if !optimize_deps_exclude.is_empty() {
        println!(
            "optimizeDeps.exclude: {} dep(s) ({}) — diffpack does not pre-bundle, so exclusion is inherent",
            optimize_deps_exclude.len(),
            optimize_deps_exclude.join(", "),
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
        diffpack_next::production::prerender_pages(&out)?;
        // Emit the pages orchestrator (`pages-server.mjs`): plain Node that imports
        // the SSR bundle's `handleRequest` and serves the client `public/` assets.
        diffpack_next::production::write_pages_server(&out)?;
        println!(
            "\nproduction build complete -> {}\n  serve it:  diffpack start {} [port]",
            out.display(),
            out.display()
        );
        return Ok(());
    }
    if diffpack_next::next_adapter::is_app_router(project_root) {
        println!("=== production build (next app-router): client -> (react-server || ssr) ===");
        // The react-server and ssr builds depend only on the CLIENT build's
        // client-reference manifest — a pure graph fact published atomically, via
        // rename) BEFORE its emit — never on the client's emitted bytes or on
        // each other's output. So all three overlap: the client build streams
        // its logs live; the moment both manifests exist the other two spawn,
        // react-server emitting straight into `rsc-render/` (`--server-dir`,
        // which also retires the publish-copy that used to sit between them)
        // while ssr owns `server/`. Their outputs are captured and replayed
        // after the client logs so the three builds' reporting never
        // interleaves. Stale manifests from an earlier build are removed first,
        // or the server builds would launch against the previous graph.
        let client_references_path = out.join(diffpack_next::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
        let _ = std::fs::remove_file(&client_references_path);
        let client_stage = diffpack_core::build_profile::stage("build/client");
        let mut client_child = std::process::Command::new(&exe)
            .arg("build-app")
            .arg(&root)
            .arg("client")
            .args(&passthrough)
            .spawn()
            .map_err(|error| format!("cannot run build-app client: {error}"))?;
        // Wait for the client-reference manifest (or for the client to fail first).
        loop {
            if client_references_path.is_file() {
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
        diffpack_next::production::assemble_server(&out)?;
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
            let instrumentation_maps = source_map_choice.resolve(false);
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
        diffpack_next::production::prerender_app(project_root, &out, false)?;
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
