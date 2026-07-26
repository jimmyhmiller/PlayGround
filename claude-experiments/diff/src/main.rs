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
            #[cfg(not(feature = "memory-accounting"))]
            {
                let _ = (modules, imports, edits, minify);
                #[allow(clippy::needless_return)]
                return Err(
                    "bundle-scale-memory needs the accounting build: \
                     cargo run --release --features memory-accounting -- bundle-scale-memory ... \
                     (production binaries carry no allocator override, so wall-time and \
                     memory are measured in separate runs)"
                        .into(),
                );
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
            let built_html = diffpack::html_entry::apply_base(
                &html.rewrite(&html_origin, &injection)?,
                &config.base,
            );
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
            if environment == "production" {
                return build_production(Path::new(&project_root), &remaining);
            }

            // Next.js app-router apps have no TanStack/src entry; their "entry" is
            // the app-router file convention (`app/layout.tsx` wrapping
            // `app/page.tsx`). The next adapter detects such a project, scaffolds the
            // three RSC entries (+ minimal `next/*` shims) under `.diffpack-next/`,
            // and returns a ready config; a non-Next project returns `None` and falls
            // back to the TanStack `derive_config` path unchanged.
            let mut config = match diffpack::next_adapter::configure(
                Path::new(&project_root),
                &environment,
            )? {
                Some(next_config) => {
                    println!(
                        "next app-router adapter: scaffolded .diffpack-next/ for environment={environment}"
                    );
                    next_config
                }
                None => diffpack::config::derive_config(Path::new(&project_root), &environment)?,
            };
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
            // RSC server actions — client transport. A `"use server"` module built
            // for the client is rewritten into `createServerReference(id, callServer)`
            // stubs importing `callServer` from `#diffpack-call-server`; register the
            // embedded transport under that specifier so the stub resolves. Harmless
            // (unreachable) when the app has no `"use server"` module.
            config.build.virtual_modules.push((
                diffpack::rsc::CALL_SERVER_SPECIFIER.to_string(),
                diffpack::rsc::call_server_module_source().to_string(),
            ));

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

                // RSC server actions — server dispatch. Register the generated action
                // resolver (`#diffpack-rsc-action-resolver`) that `getServerActionById`
                // dispatches through, keyed by the same `"<moduleId>#<name>"` id the
                // client stub and the server registration derive, plus the embedded
                // `handleServerAction` endpoint (`#diffpack-rsc-action-handler`). The
                // resolver is generated from a pre-scan of the app source for
                // `"use server"` modules. Registered before discovery so the subpath
                // imports resolve to the native modules.
                let server_actions =
                    diffpack::rsc::scan_project_server_actions(Path::new(&project_root))?;
                config.build.virtual_modules.push((
                    diffpack::rsc::ACTION_RESOLVER_SPECIFIER.to_string(),
                    diffpack::rsc::generate_action_resolver_module(&server_actions),
                ));
                config.build.virtual_modules.push((
                    diffpack::rsc::ACTION_HANDLER_SPECIFIER.to_string(),
                    diffpack::rsc::action_handler_module_source().to_string(),
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
                    output_root.join(diffpack::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
                let client_references =
                    diffpack::rsc::ClientReferencesManifest::read(&client_references_path)?;
                config.build.virtual_modules.push((
                    diffpack::rsc::SSR_CONSUMER_MANIFEST_SPECIFIER.to_string(),
                    client_references.to_ssr_consumer_manifest_module(None),
                ));
                println!(
                    "registered the rsc ssr consumer manifest ({} client reference(s))",
                    client_references.entries.len(),
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
                // next/image (Slice J): emit downscaled responsive variants for every
                // raster under `public/` into `<public>/_diffpack-image/`. The runtime
                // shim's `srcset` points at these static files (no image server). A
                // no-op for a non-Next project (no public images / no app-router).
                let public_images =
                    diffpack::next_adapter::scan_public_images(Path::new(&project_root))?;
                if !public_images.is_empty() {
                    let variants = diffpack::next_adapter::emit_image_variants(
                        Path::new(&project_root),
                        &summary.output_dir,
                        &public_images,
                    )?;
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
                let meta_images = diffpack::next_adapter::emit_metadata_images(
                    Path::new(&project_root),
                    &summary.output_dir,
                )?;
                if meta_images > 0 {
                    println!(
                        "copied {meta_images} metadata image file(s) to {}",
                        summary.output_dir.display(),
                    );
                }
                // Persist the route -> client chunk mapping so the server build can
                // generate the TanStack manifest from real emitted chunk URLs.
                let client_manifest =
                    bundler.client_route_manifest(&reachable, "client.js", "/")?;
                let client_manifest_path =
                    output_root.join(diffpack::manifest::CLIENT_MANIFEST_FILE);
                client_manifest.write(&client_manifest_path)?;
                // Persist the client-references manifest (Manifest #1 / bundlerConfig)
                // so the react-server render can resolve each `"use client"` `$$id` to
                // its client runtime id + hosting chunk. Regenerated on every client
                // emit (the ids are build-derived; the moduleId key is stable).
                let client_references =
                    bundler.client_references_manifest(&reachable, "client.js")?;
                let client_references_path = output_root
                    .join(diffpack::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
                client_references.write(&client_references_path)?;
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
                let summary = bundler.emit_server(&reachable, &output_root, emit_options)?;
                // The react-server graph is authoritative for the app's CSS (Server
                // Components render there, so its CSS-Module class scoping matches the
                // flight-rendered classNames). Preserve its compiled `server.css` to
                // the served, non-pruned `public/rsc.css` (the SSR build would
                // otherwise prune it from `server/`); the adapter links it into the
                // document head. Next injects the route's stylesheets the same way.
                if config.environment == "react-server" {
                    let css = output_root.join("server/server.css");
                    if css.is_file() {
                        let dest = output_root.join("public/rsc.css");
                        if let Some(parent) = dest.parent() {
                            std::fs::create_dir_all(parent).map_err(|error| {
                                format!("cannot create {}: {error}", parent.display())
                            })?;
                        }
                        std::fs::copy(&css, &dest).map_err(|error| {
                            format!("cannot preserve react-server CSS to {}: {error}", dest.display())
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
                // other's manifest.
                let server_references =
                    bundler.client_references_manifest(&reachable, "server.mjs")?;
                let server_references_path = output_root
                    .join(diffpack::rsc::SERVER_REFERENCES_MANIFEST_FILE);
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
            // Dev source maps default ON so a runtime-error stack frame in the error
            // overlay resolves back to the original source; `--no-sourcemap` opts out.
            // (Production `build-app` keeps its opt-in `--sourcemap` default-off.)
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
                Some("client") => diffpack::rsc::transform_use_server_client(path, &source)
                    .ok_or_else(|| {
                        format!("{} is not a \"use server\" module", path.display())
                    })?,
                Some("server") => diffpack::rsc::transform_use_server_server(path, &source)?
                    .ok_or_else(|| {
                        format!("{} is not a \"use server\" module", path.display())
                    })?,
                // The REACT-SERVER-graph rewrite of a `"use client"` module: its
                // real code never reaches the server; each export becomes a client
                // reference the flight render serializes via the manifest.
                Some("client-ref") => diffpack::rsc::transform_use_client_server(path, &source)
                    .ok_or_else(|| {
                        format!("{} is not a \"use client\" module", path.display())
                    })?,
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
            let manifest_path = Path::new(&output_dir)
                .join(diffpack::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
            let manifest =
                diffpack::rsc::ClientReferencesManifest::read(&manifest_path)?;
            let value = manifest.to_ssr_consumer_manifest_json(None);
            print!(
                "{}",
                serde_json::to_string_pretty(&value)
                    .map_err(|error| format!("cannot serialize ssr consumer manifest: {error}"))?
            );
            Ok(())
        }
        Some("rsc-resolver") => {
            let root = arguments.next().ok_or_else(|| {
                "usage: diffpack rsc-resolver <project-root>".to_string()
            })?;
            let entries =
                diffpack::rsc::scan_project_server_actions(Path::new(&root))?;
            print!(
                "{}",
                diffpack::rsc::generate_action_resolver_module(&entries)
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
                let value = match diffpack::tailwind::compile("@import 'tailwindcss';\n", &set) {
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
            let entry = if next_server.exists() {
                next_server
            } else {
                out.join("server/index.mjs")
            };
            if !entry.exists() {
                return Err(format!(
                    "no production server entry in {} — run `diffpack build-app <root> production` first",
                    out.display()
                ));
            }
            let status = std::process::Command::new("node")
                .arg(&entry)
                .arg(out)
                .arg(&port)
                .status()
                .map_err(|error| format!("cannot start node server ({}): {error}", entry.display()))?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("production server exited with {status}"))
            }
        }
        _ => Err(usage()),
    }
}

/// Recursively copy `src` into `dest` (used to publish the react-server bundle from
/// `server/` to `rsc-render/` before the ssr build overwrites `server/`).
fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest)
        .map_err(|error| format!("cannot create {}: {error}", dest.display()))?;
    for entry in std::fs::read_dir(src)
        .map_err(|error| format!("cannot read {}: {error}", src.display()))?
    {
        let entry = entry.map_err(|error| format!("cannot read entry in {}: {error}", src.display()))?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        let kind = entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", from.display()))?;
        if kind.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)
                .map_err(|error| format!("cannot copy {} -> {}: {error}", from.display(), to.display()))?;
        }
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
    let run = |environment: &str| -> Result<(), String> {
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
    if diffpack::next_adapter::is_app_router(project_root) {
        println!("=== production build (next app-router): client -> react-server -> ssr ===");
        run("client")?;
        run("react-server")?;
        // Publish the react-server bundle to `rsc-render/` BEFORE the ssr build
        // overwrites `server/` — the orchestrator reads the two from distinct dirs.
        let server = out.join("server");
        let rsc_render = out.join("rsc-render");
        let _ = std::fs::remove_dir_all(&rsc_render);
        copy_dir_recursive(&server, &rsc_render)?;
        run("ssr")?;
        std::fs::write(
            out.join("next-server.mjs"),
            include_str!("../scripts/rsc/next-server.mjs"),
        )
        .map_err(|error| format!("cannot write production server: {error}"))?;
        // instrumentation.{ts,js}: the app's boot hook (register() runs once at server
        // startup — OpenTelemetry/Sentry-style). Write the generated boot-entry wrapper
        // (which CALLS register at module load) then bundle it NATIVELY (self-invoke our
        // own `bundle` subcommand, ESM) to <out>/instrumentation.mjs; the orchestrator
        // dynamic-imports it once before listen (see next-server.mjs). Build-time only,
        // zero per-request cost.
        if let Some(wrapper) = diffpack::next_adapter::write_instrumentation_wrapper(project_root)? {
            println!("=== instrumentation (register() boot hook) ===");
            let instr_out = out.join("instrumentation.mjs");
            let status = std::process::Command::new(&exe)
                .arg("bundle")
                .arg(&wrapper)
                .arg(&instr_out)
                .arg("--format")
                .arg("esm")
                .status()
                .map_err(|error| format!("cannot bundle instrumentation ({}): {error}", wrapper.display()))?;
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
    "usage: diffpack build <project-root> [--vite] [--out-dir <dir>] [--no-minify] [--sourcemap] | diffpack build-app <project-root> [client|react-server|ssr|nitro|static] [--no-minify] [--sourcemap] [--static-export] | diffpack dev <project-root> [port] [--no-minify] [--no-sourcemap] | diffpack bundle <entry> [output] [--sourcemap] [--minify] [--format esm|cjs] | diffpack visualize <entry> [output.html] | diffpack visualize-scale [modules] [imports-per-module] [output.html] | diffpack watch <entry> [output] | diffpack bundle-scale-direct [modules] [imports-per-module] | diffpack bundle-scale-direct-deps [modules] [imports-per-module] | diffpack bundle-scale-direct-live [modules] [imports-per-module] | diffpack bundle-scale-direct-live-deps [modules] [imports-per-module]".into()
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

/// `diffpack build-app <root> static` — the SSG prerender phase. Builds NO graph:
/// reuses the three already-emitted bundles + manifests, re-runs native route
/// classification to write the prerender plan, then spawns the node prerenderer (the
/// app's own React runtime — the explicitly-allowed oracle) to write `.html` + `.rsc`
/// for every static/SSG route. Dynamic routes are recorded, never dropped.
fn build_static(project_root: &Path, static_export: bool) -> Result<(), String> {
    let output_root = project_root.join(".diffpack-output");

    // Hard-error (naming which) if any of the four inputs is missing — mirrors the
    // orchestrator's fail() checks. The SSG render reuses these verbatim.
    for (label, rel) in [
        ("react-server render bundle", "rsc-render/server.mjs"),
        ("SSR bundle", "server/server.mjs"),
        ("client-references manifest", "client-references-manifest.json"),
        ("ssr-references manifest", "server-references-manifest.json"),
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
fn next_prerender(project_root: &Path, output_root: &Path, static_export: bool) -> Result<(), String> {
    // Native route classification -> the machine-readable prerender plan.
    let route_count = diffpack::next_adapter::write_prerender_plan(project_root, output_root)?;
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

    // Spawn the prerenderer (the app's own React runtime; the bundling stays native
    // Rust). Its stdout/stderr stream straight through; a nonzero exit fails the build.
    let mut command = std::process::Command::new("node");
    command.arg(&prerender_path).arg(output_root);
    if static_export {
        command.arg("--static-export");
    }
    let status = command
        .status()
        .map_err(|error| format!("cannot spawn node for the SSG prerenderer: {error}"))?;
    if !status.success() {
        return Err(format!(
            "the SSG prerenderer (node {}) failed with {status}",
            prerender_path.display(),
        ));
    }
    Ok(())
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
