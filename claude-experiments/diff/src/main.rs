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
use diffpack::bundler::{Bundler, EmitOptions};
use notify::{RecursiveMode, Watcher};

fn main() -> ExitCode {
    match run() {
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
        Some("build") => {
            let project_root = arguments.next().ok_or_else(usage)?;
            let remaining = arguments.collect::<Vec<_>>();
            let has_flag = |flag: &str| remaining.iter().any(|value| value.to_str() == Some(flag));
            let vite = has_flag("--vite");
            let minify = !has_flag("--no-minify");
            let source_map = has_flag("--sourcemap");
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

            let root = Path::new(&project_root)
                .canonicalize()
                .map_err(|error| {
                    format!("cannot open project root {}: {error}", Path::new(&project_root).display())
                })?;
            let html_path = root.join("index.html");
            if !html_path.is_file() {
                return Err(format!(
                    "{} has no index.html; `diffpack build` bundles an HTML-rooted web app \
                     (use `diffpack bundle <entry>` for a bare module entry)",
                    root.display()
                ));
            }
            let html = diffpack::html_entry::parse_file(&html_path)?;
            let html_origin = html_path.display().to_string();
            let entry_script = match html.module_scripts.as_slice() {
                [] => {
                    return Err(format!(
                        "{html_origin}: no local <script type=\"module\" src> entry found"
                    ));
                }
                [only] => only,
                many => {
                    let sources = many
                        .iter()
                        .map(|script| script.src.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(format!(
                        "{html_origin}: {} module script entries ({sources}); multiple HTML \
                         entries are not supported yet",
                        many.len()
                    ));
                }
            };
            // A root-absolute src (`/src/main.tsx`) is project-root-relative, as
            // in a dev server's URL space; anything else is relative to the HTML
            // document itself.
            let entry_path = match entry_script.src.strip_prefix('/') {
                Some(rest) => root.join(rest),
                None => html_path
                    .parent()
                    .expect("an HTML file has a parent directory")
                    .join(&entry_script.src),
            };
            let entry = entry_path.canonicalize().map_err(|error| {
                format!(
                    "{html_origin}: module script src \"{}\" does not resolve \
                     ({}: {error})",
                    entry_script.src,
                    entry_path.display()
                )
            })?;

            let config = diffpack::config::derive_web_config(&root, vite)?;
            if config.vite && config.base != "/" {
                // Asset-URL synthesis does not apply a custom base yet; building
                // anyway would emit URLs the page cannot load.
                return Err(format!(
                    "vite `base` is {:?}, but diffpack does not yet apply a non-root base \
                     to emitted asset URLs",
                    config.base
                ));
            }
            println!(
                "web build{}: entry={}",
                if config.vite { " (vite mode)" } else { "" },
                entry.display()
            );
            let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config.build)?;
            // A generic web build fails on unresolved imports — an artifact with
            // dangling references is not a successful build.
            if !update.diagnostics.is_empty() {
                let mut message = format!(
                    "{} unresolved import(s):",
                    update.diagnostics.len()
                );
                for diagnostic in &update.diagnostics {
                    message.push_str("\n  ");
                    message.push_str(diagnostic);
                }
                return Err(message);
            }
            let reachable = bundler.reachable_modules_direct();
            println!("reachable {} modules", reachable.len());

            let out_dir = out_dir.unwrap_or_else(|| root.join("dist"));
            let emit_options = EmitOptions {
                minify,
                source_map,
                ..EmitOptions::default()
            };
            let summary = bundler.emit_web(&reachable, &out_dir, "index.js", emit_options)?;
            // The `public/` passthrough directory is a Vite convention.
            let static_files = if config.vite {
                diffpack::config::copy_static_public(&root, &out_dir)?
            } else {
                0
            };
            let mut injection = diffpack::html_entry::HeadInjection {
                script_urls: vec![format!("{}index.js", config.base)],
                stylesheet_urls: Vec::new(),
            };
            if summary.css_files > 0 {
                injection.stylesheet_urls.push(format!("{}index.css", config.base));
            }
            let built_html = html.rewrite(&html_origin, &injection)?;
            let html_out = out_dir.join("index.html");
            std::fs::write(&html_out, built_html)
                .map_err(|error| format!("cannot write {}: {error}", html_out.display()))?;
            println!(
                "emitted {}: {} .js, {} .css, {} asset(s), {} static file(s), index.html",
                summary.output_dir.display(),
                summary.javascript_files,
                summary.css_files,
                summary.asset_files,
                static_files,
            );
            Ok(())
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
            // Production source maps, composed through the minify pass, are opt-in
            // per build (`--sourcemap`) so the default acceptance/benchmark path is
            // unchanged. When set, both the client `public/` and server `server/`
            // emits ship a sibling `.map` per chunk resolving minified positions
            // back to the original source.
            let source_map = remaining
                .iter()
                .any(|value| value.to_str() == Some("--sourcemap"));
            let environment = remaining
                .iter()
                .find(|value| !value.to_string_lossy().starts_with("--"))
                .and_then(|value| value.to_str().map(str::to_string))
                .unwrap_or_else(|| "client".to_string());
            let mut config =
                diffpack::config::derive_config(Path::new(&project_root), &environment)?;
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
            if let Some(route_count) =
                diffpack::route_tree::generate_for_project(Path::new(&project_root))?
            {
                println!(
                    "generated src/{} natively ({route_count} route(s))",
                    diffpack::route_tree::ROUTE_TREE_FILE,
                );
            }

            // A server build's TanStack manifest module (`tanstack-start-manifest:v`)
            // maps each route to the CLIENT build's emitted chunk URLs, so it is
            // generated natively from the client build's persisted route/chunk
            // manifest. Register it as a virtual module before discovery, so the
            // server's `router-manifest.js` import resolves and loads it. A missing
            // client manifest is a hard, specific error (run the client build first)
            // rather than a silent empty manifest.
            if config.environment != "client" {
                let client_manifest_path =
                    output_root.join(diffpack::manifest::CLIENT_MANIFEST_FILE);
                let client_manifest =
                    diffpack::manifest::ClientRouteManifest::read(&client_manifest_path)?;
                config.build.virtual_modules.push((
                    diffpack::manifest::START_MANIFEST_SPECIFIER.to_string(),
                    client_manifest.to_start_manifest_source(),
                ));
                // The sibling dev-only virtual module `loadVirtualModule.js`
                // statically references (only used under TSS_DEV_SERVER, but its
                // `import()` literal must still resolve). Register it too so the
                // server build resolves cleanly on react-start versions that emit it.
                config.build.virtual_modules.push((
                    diffpack::manifest::INJECTED_HEAD_SCRIPTS_SPECIFIER.to_string(),
                    diffpack::manifest::injected_head_scripts_module_source(),
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
                let server_fns =
                    diffpack::server_fn::scan_project_server_fns(Path::new(&project_root))?;
                config.build.virtual_modules.push((
                    diffpack::server_fn::RESOLVER_SPECIFIER.to_string(),
                    diffpack::server_fn::generate_resolver_module(&server_fns),
                ));
                println!(
                    "registered {} server function(s) in the native server-fn resolver",
                    server_fns.len(),
                );
            }

            println!(
                "app: environment={} ({} aliases), entry={}",
                config.environment,
                config.build.aliases.len(),
                entry.display(),
            );
            let (bundler, update) =
                Bundler::discover_direct_with_config(&entry, &config.build)?;
            let reachable = bundler.reachable_modules_direct();
            println!(
                "reachable {} modules; {} diagnostic(s)",
                reachable.len(),
                update.diagnostics.len()
            );
            for diagnostic in &update.diagnostics {
                println!("  known gap: {diagnostic}");
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
                let summary = bundler.emit_public(&reachable, &output_root, emit_options)?;
                let static_files = diffpack::config::copy_static_public(
                    Path::new(&project_root),
                    &summary.output_dir,
                )?;
                // Persist the route -> client chunk mapping so the server build can
                // generate the TanStack manifest from real emitted chunk URLs.
                let client_manifest =
                    bundler.client_route_manifest(&reachable, "client.js", "/")?;
                let client_manifest_path =
                    output_root.join(diffpack::manifest::CLIENT_MANIFEST_FILE);
                client_manifest.write(&client_manifest_path)?;
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
                let summary = bundler.emit_server(&reachable, &output_root, emit_options)?;
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
            let source_map = flags
                .iter()
                .any(|value| value.to_str() == Some("--sourcemap"));
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
            let (bundler, update) = Bundler::discover_direct(Path::new(&entry))?;
            if profile {
                eprintln!("discover: {:.1} ms", discover_started.elapsed().as_secs_f64() * 1000.0);
            }
            if !update.diagnostics.is_empty() {
                return Err(format!(
                    "bundle produced {} diagnostic(s); first: {}",
                    update.diagnostics.len(),
                    update.diagnostics[0]
                ));
            }
            let phase_started = Instant::now();
            let reachable = bundler.reachable_modules_direct();
            if profile {
                eprintln!("reachability: {:.1} ms", phase_started.elapsed().as_secs_f64() * 1000.0);
            }
            let phase_started = Instant::now();
            bundler.emit_with_options(&reachable, &output, EmitOptions { source_map, minify, format, ..Default::default() })?;
            if profile {
                eprintln!("emit: {:.1} ms", phase_started.elapsed().as_secs_f64() * 1000.0);
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
            let (bundler, update) = Bundler::discover_direct(Path::new(&entry))?;
            if !update.diagnostics.is_empty() {
                return Err(format!(
                    "visualization produced {} diagnostic(s); first: {}",
                    update.diagnostics.len(),
                    update.diagnostics[0]
                ));
            }
            let reachable = bundler.reachable_modules_direct();
            let graph = bundler.visualization_graph(&reachable);
            diffpack::visualizer::write_visualization(&graph, &output)?;
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
            let source_map = remaining
                .iter()
                .any(|value| value.to_str() == Some("--sourcemap"));
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
        _ => Err(usage()),
    }
}

fn usage() -> String {
    "usage: diffpack build <project-root> [--vite] [--out-dir <dir>] [--no-minify] [--sourcemap] | diffpack build-app <project-root> [client|ssr|nitro] [--no-minify] [--sourcemap] | diffpack dev <project-root> [port] [--no-minify] [--sourcemap] | diffpack bundle <entry> [output] [--sourcemap] [--minify] [--format esm|cjs] | diffpack visualize <entry> [output.html] | diffpack visualize-scale [modules] [imports-per-module] [output.html] | diffpack watch <entry> [output] | diffpack bundle-scale-direct [modules] [imports-per-module] | diffpack bundle-scale-direct-deps [modules] [imports-per-module] | diffpack bundle-scale-direct-live [modules] [imports-per-module] | diffpack bundle-scale-direct-live-deps [modules] [imports-per-module]".into()
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
    let (mut bundler, initial) = Bundler::discover_direct(entry)?;
    let mut session = bundler.direct_reachability();
    let mut reachable = session.reachable_modules();
    bundler.emit(&reachable, output)?;
    println!(
        "watching {} ({} modules); wrote {}",
        bundler.watch_root().display(),
        reachable.len(),
        output.display()
    );
    for diagnostic in initial.diagnostics {
        eprintln!("diagnostic: {diagnostic}");
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
            bundler.emit(&reachable, output)?;
            println!(
                "rebuilt {}: reachable={} transformed={} reachability={:.3}ms",
                path.display(),
                reachable.len(),
                update.transformed_modules,
                reachability_ms
            );
            for diagnostic in update.diagnostics {
                eprintln!("diagnostic: {diagnostic}");
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
