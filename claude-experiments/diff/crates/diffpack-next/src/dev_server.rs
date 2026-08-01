//! Next App Router development orchestration.

pub mod next {
    use std::collections::{BTreeSet, HashMap};
    use std::fs;
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command};
    use std::sync::Arc;
    use std::sync::mpsc::{self, Receiver};
    use std::time::{Duration, Instant, SystemTime};

    use diffpack_default_loader::driver::{
        EmitCancel, EmitOptions, EmitSummary, ModuleFormat, StylesheetEmit, partition_diagnostics,
    };
    use diffpack_default_loader::driver_config::EnvironmentConfig as AppConfig;
    use diffpack_web::config;
    use diffpack_web::dev_build::{DevOptions, EnvBuild, EnvCounters, source_watch_roots};
    use diffpack_web::dev_control::{post_json, push_client as hmr_push_client};
    use diffpack_web::runtime::{free_port, wait_for_node};
    use diffpack_web::watch::{
        EventEpoch, coalesce_batch, is_config_file, is_dependency_or_generated, is_module_path,
        is_stylesheet_path, start_paths_into as start_watcher_paths_into,
        uncovered_roots as uncovered_watch_roots,
    };
    use diffpack_web::websocket::HmrHub;
    use notify::RecursiveMode;

    use crate::lazy_routes::LazyRoutes;
    use crate::next_adapter::RouteScope;

    fn reachable_ids(build: &EnvBuild) -> BTreeSet<String> {
        build.reachable_ids()
    }

    fn serve_proxy(
        listener: TcpListener,
        node_port: u16,
        hub: HmrHub,
        refresh_runtime: Arc<String>,
        static_dir: Option<Arc<PathBuf>>,
        lazy: Option<Arc<LazyRoutes>>,
    ) {
        let route_gate = lazy.map(|gate| gate as Arc<dyn diffpack_web::node_proxy::RouteGate>);
        diffpack_web::node_proxy::serve(
            listener,
            node_port,
            hub,
            refresh_runtime,
            static_dir,
            route_gate,
        );
    }

    fn reconcile_next_async_islands(
        root: &Path,
        environment: &str,
        bundler: &diffpack_default_loader::driver::Bundler,
        reachable: &BTreeSet<String>,
    ) -> Result<bool, String> {
        crate::next_adapter::reconcile_async_islands_from_tainted(
            root,
            environment,
            &bundler.async_tainted_modules(reachable),
        )
    }

    fn register_next_server_virtual_modules(
        config: &mut AppConfig,
        project_root: &Path,
    ) -> Result<(), String> {
        let server_actions = crate::rsc::scan_project_server_actions(project_root)?;
        config.build.virtual_modules.push((
            crate::rsc::ACTION_RESOLVER_SPECIFIER.to_string(),
            crate::rsc::generate_action_resolver_module(&server_actions),
        ));
        config.build.virtual_modules.push((
            crate::rsc::ACTION_HANDLER_SPECIFIER.to_string(),
            crate::rsc::action_handler_module_source().to_string(),
        ));
        config.build.virtual_modules.push((
            crate::rsc::CALL_SERVER_SPECIFIER.to_string(),
            crate::rsc::call_server_module_source().to_string(),
        ));
        Ok(())
    }

    /// The next orchestrator, embedded so the dev server is self-contained (it does
    /// not need to locate the diffpack repo at runtime). It is plain Node that wires
    /// the three emitted bundles + manifests into an HTTP app; it reads only absolute
    /// paths derived from its `<output-dir>` argv, so running it from the output dir
    /// is equivalent to running it from the repo.
    const NEXT_SERVER_MJS: &str = include_str!("dev_assets/next-server.mjs");

    /// Entry point: build the three graphs, boot the orchestrator, put the HMR proxy
    /// in front, and drive the incremental rebuild loop. Blocks until the watcher
    /// stops or an unsupported edit is hit (a hard error).
    pub fn run_next(options: &DevOptions, project_root: &Path) -> Result<(), String> {
        let output_root = project_root.join(".diffpack-output");
        // The react-server graph emits here (isolated from the ssr `server/` bundle),
        // then is copied to `<out>/rsc-render` where the orchestrator reads it.
        let rsc_root = output_root.join(".rsc");
        let emit_options = EmitOptions {
            minify: false,
            source_map: options.source_map,
            hmr: true,
            ..EmitOptions::default()
        };

        // Fast Refresh runtime (must be present — a missing dep is a hard error now,
        // not a broken update later).
        let refresh_runtime = Arc::new(diffpack_web::hmr::find_refresh_runtime(project_root)?);
        let hub = HmrHub::default();

        // Everything the orchestrator needs beyond the graph outputs, written
        // before ANY boot (warm or cold) so both paths run the same bytes.
        let next_server_script = output_root.join("next-server.mjs");
        // Takes the output dir explicitly because the background fill emits into a SHADOW
        // dir (see the fill below) that is then swapped in wholesale, and it has to carry
        // the same orchestrator bytes.
        let write_orchestrator_scripts = |dir: &Path| -> Result<(), String> {
            fs::create_dir_all(dir)
                .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
            let script = dir.join("next-server.mjs");
            fs::write(&script, NEXT_SERVER_MJS)
                .map_err(|error| format!("cannot write {}: {error}", script.display()))?;
            // The orchestrator imports this as a sibling (the one place that joins the
            // three references manifests into the divergent-id ssrModuleMapping).
            let ssr_module_map = dir.join(crate::rsc::SSR_MODULE_MAP_FILE);
            fs::write(
                &ssr_module_map,
                include_str!("dev_assets/ssr-module-map.mjs"),
            )
            .map_err(|error| format!("cannot write {}: {error}", ssr_module_map.display()))?;
            Ok(())
        };
        let boot_node = |port: u16| -> Result<Child, String> {
            let mut node = spawn_next_node(&next_server_script, project_root, &output_root, port)?;
            wait_for_node(port).inspect_err(|_| {
                let _ = node.kill();
            })?;
            Ok(node)
        };
        let start_proxy = |node_port: u16, lazy: Option<Arc<LazyRoutes>>| -> Result<(), String> {
            // The diffpack reverse proxy: serves the HMR client + Fast Refresh
            // runtime, upgrades the WS channel, and injects the preamble into
            // every served HTML document.
            let proxy_listener = TcpListener::bind(("127.0.0.1", options.port))
                .map_err(|error| format!("cannot bind dev port {}: {error}", options.port))?;
            let hub = hub.clone();
            let refresh_runtime = Arc::clone(&refresh_runtime);
            // Serve the emitted client `public/` (chunks + assets) directly from the
            // proxy, so a browser HMR chunk fetch skips the Node orchestrator hop.
            let static_dir = Some(Arc::new(output_root.join("public")));
            std::thread::Builder::new()
                .name("diffpack-dev-next-proxy".into())
                .spawn(move || {
                    serve_proxy(
                        proxy_listener,
                        node_port,
                        hub,
                        refresh_runtime,
                        static_dir,
                        lazy,
                    )
                })
                .map_err(|error| format!("cannot start proxy thread: {error}"))?;
            Ok(())
        };

        // LAZY COMPILATION, OPT IN. Instead of compiling every route before answering
        // anything, the dev server can put the proxy up first, let the first request say
        // which route it wants, compile that, and fill in the rest behind it. The pattern
        // table comes from discovery alone (a directory walk), so a request can be matched
        // to a route before a single module is compiled.
        //
        // OFF by default, on measurement rather than taste. On cal.com it makes the first
        // document faster (6.2s -> 5.4s) but a second route clicked during the ~7s
        // background fill much slower (6.3s -> 12.7s), because widening the route scope
        // renumbers every runtime module id and so costs a full second build. The eager
        // path keeps every other win of this work (island pins derived from the
        // react-server graph, react-server built first, the parallel Tailwind scan) with no
        // such window. When ids become stable the fill turns incremental and this default
        // should flip — see docs/DEV_LAZY_ROUTES.md §5.
        //
        //   DIFFPACK_DEV_LAZY=1     pages compile on demand, endpoints stay eager
        //   DIFFPACK_DEV_LAZY=api   endpoints compile on demand too (measured SLOWER on
        //                           cal.com, whose server render calls its own API)
        let lazy_mode = std::env::var("DIFFPACK_DEV_LAZY").unwrap_or_default();
        let lazy = match lazy_mode.as_str() {
            "1" | "pages" | "api" => crate::next_adapter::discover_route_patterns(project_root)?
                .map(|patterns| Arc::new(LazyRoutes::new(patterns))),
            "" | "0" => None,
            other => {
                return Err(format!(
                    "DIFFPACK_DEV_LAZY={other:?} is not a mode; use 1 (lazy pages), api (lazy pages + endpoints), or 0 (the default, compile everything up front)"
                ));
            }
        };
        let cold_started = Instant::now();
        write_orchestrator_scripts(&output_root)?;
        let node_port = free_port()?;
        // The proxy binds BEFORE anything is compiled when lazy: a request arriving now is
        // what tells the first build which route to compile, and it blocks in
        // `LazyRoutes::ensure` until that build lands.
        if lazy.is_some() {
            start_proxy(node_port, lazy.clone())?;
            println!(
                "[dev] diffpack dev server (next app-router) on http://127.0.0.1:{} — routes compile on demand",
                options.port,
            );
        }

        // The first build's scope: what the first visitor asked for, or the whole app when
        // nobody asks within the grace period (nothing is waiting, so there is no reason
        // to compile a subset).
        let first_scope = match &lazy {
            Some(lazy) => first_build_scope(lazy),
            None => RouteScope::All,
        };
        println!("[dev] next: building graphs ({})...", first_scope.label());
        let graphs_started = Instant::now();
        let first = build_all_graphs(
            project_root,
            &output_root,
            &rsc_root,
            emit_options,
            &first_scope,
        )
        .inspect_err(|error| {
            // Release anything blocked on this build with the real error rather
            // than leaving the connection hanging until the wait times out.
            if let Some(lazy) = &lazy {
                lazy.failed(error);
            }
        })?;
        let (mut client, mut react_server, mut ssr) = (first.client, first.react_server, first.ssr);
        let graphs_ms = graphs_started.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "[dev] next: {} | wall {graphs_ms:.0}ms | {} islands pinned",
            graph_timing_label(&first.timings),
            crate::next_adapter::recorded_islands(project_root).len(),
        );

        let boot_started = Instant::now();
        let mut node = boot_node(node_port).inspect_err(|error| {
            if let Some(lazy) = &lazy {
                lazy.failed(error);
            }
        })?;
        let boot_ms = boot_started.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "[dev] next orchestrator listening on 127.0.0.1:{node_port} (boot {boot_ms:.0}ms)"
        );
        let cold_ms = cold_started.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "[dev] next cold start: {cold_ms:.0}ms total (graphs {graphs_ms:.0} + boot {boot_ms:.0})",
        );
        // A dev server never reaches main's exit-time report (it blocks in the watch loop
        // until killed), so the cold start prints its own stage table under
        // `DIFFPACK_PROFILE=1`. Without this the startup breakdown is unobservable in the
        // one process where startup is the whole question.
        diffpack_core::build_profile::report("dev cold start", cold_ms);
        // Whatever was waiting for this build can go now.
        if let Some(lazy) = &lazy {
            lazy.landed(&first_scope, node_port);
        }
        if lazy.is_none() {
            start_proxy(node_port, None)?;
            println!(
                "[dev] diffpack dev server (next app-router) on http://127.0.0.1:{} (proxying node :{node_port})",
                options.port
            );
        }

        // THE FILL: compile the routes the first build left out, so a later navigation is
        // instant instead of paying for its own build. It runs right after the first page
        // has been served — getting that page out was the whole point of the first build,
        // and the fill is what makes the second one free.
        //
        // It emits into a SHADOW output dir rather than over the live one. The live
        // orchestrator serves its bundles out of `<out>` and the proxy serves chunks
        // straight off `<out>/public`, so emitting the fill there would rewrite, mid-render,
        // the exact files the page being displayed is running from — which is not a
        // theoretical risk: doing that made every request between the first build and the
        // fill's end fail. The finished shadow dir is renamed into place instead, which is
        // atomic enough that only the swap itself (a few milliseconds, with requests held)
        // is a window at all.
        //
        // The swap does need the orchestrator restarted and open browsers reloaded once:
        // growing the module set moves runtime module ids, so a page holding the old ids
        // cannot resolve the new bundles' client references. Said out loud in the log,
        // because a reload nobody explained looks like a bug.
        // START WATCHING NOW, before the fill. The watch loop cannot run until the graphs
        // are final, but the WATCHER can, and it has to: the fill takes seconds, and an
        // edit made during it would otherwise be dropped on the floor — the browser shows
        // stale output with nothing said. Events queue in this channel and the loop drains
        // them the moment it starts.
        let app_dir = crate::next_adapter::app_dir(project_root).ok_or_else(|| {
            format!(
                "next dev: {} has no app/ or src/app directory",
                project_root.display()
            )
        })?;
        let watch_roots_for = |client: &EnvBuild, react_server: &EnvBuild, ssr: &EnvBuild| {
            let mut roots = source_watch_roots(project_root, &[client, react_server, ssr]);
            if !roots
                .iter()
                .any(|(root, mode)| *mode == RecursiveMode::Recursive && app_dir.starts_with(root))
            {
                roots.push((app_dir.clone(), RecursiveMode::Recursive));
            }
            if !roots.iter().any(|(root, _)| root == project_root) {
                roots.push((project_root.to_path_buf(), RecursiveMode::NonRecursive));
            }
            roots
        };
        let announce_roots = |roots: &[(PathBuf, RecursiveMode)]| {
            for (root, mode) in roots {
                println!(
                    "[dev] watching {}{}",
                    root.display(),
                    if *mode == RecursiveMode::Recursive {
                        ""
                    } else {
                        " (top level only)"
                    }
                );
            }
        };
        let (watch_events, receiver) = mpsc::channel();
        let mut watch_roots = watch_roots_for(&client, &react_server, &ssr);
        announce_roots(&watch_roots);
        start_watcher_paths_into(&watch_roots, watch_events.clone())?;

        let mut live_scope = first_scope.clone();
        if first_scope != RouteScope::All {
            // Let the page that triggered the first build actually finish first. A whole-app
            // compile saturates every core, and starting it while the first render is in
            // flight is measurable in exactly the number this whole change exists to
            // improve (cal.com: 1.4s render -> 2.5s). Capped so a page that polls forever
            // cannot postpone the fill for good.
            if let Some(lazy) = &lazy {
                lazy.wait_for_quiet(FILL_QUIET, FILL_QUIET_BUDGET);
            }
            let fill_started = Instant::now();
            println!("[dev] next: compiling the remaining routes in the background...");
            let fill_root = path_with_suffix_local(&output_root, ".fill");
            let fill_rsc_root = fill_root.join(".rsc");
            if fill_root.exists() {
                fs::remove_dir_all(&fill_root)
                    .map_err(|error| format!("cannot clear {}: {error}", fill_root.display()))?;
            }
            match build_all_graphs(
                project_root,
                &fill_root,
                &fill_rsc_root,
                emit_options,
                &RouteScope::All,
            )
            .and_then(|built| write_orchestrator_scripts(&fill_root).map(|()| built))
            {
                Ok(filled) => {
                    let build_ms = fill_started.elapsed().as_secs_f64() * 1_000.0;
                    let swap_started = Instant::now();
                    if let Some(lazy) = &lazy {
                        lazy.begin_swap();
                    }
                    let _ = node.kill();
                    let _ = node.wait();
                    swap_output_dir(&fill_root, &output_root)?;
                    client = filled.client;
                    react_server = filled.react_server;
                    ssr = filled.ssr;
                    live_scope = RouteScope::All;
                    node = boot_node(node_port)?;
                    if let Some(lazy) = &lazy {
                        lazy.landed(&RouteScope::All, node_port);
                    }
                    hub.broadcast_reload();
                    println!(
                        "[dev] next: all routes compiled in {build_ms:.0}ms | {} | swapped in {:.0}ms, reloading open browsers once (module ids moved)",
                        graph_timing_label(&filled.timings),
                        swap_started.elapsed().as_secs_f64() * 1_000.0,
                    );
                }
                // A fill failure must NOT take down a dev server that is already serving the
                // route it was asked for. Keep serving it, say exactly what is degraded (a
                // request for any other route now fails instead of waiting forever), and let
                // the next edit — which rebuilds from scratch — recover.
                Err(error) => {
                    eprintln!(
                        "[dev] next: compiling the remaining routes FAILED: {error}\n\
                         [dev] next: still serving {}; other routes will report this error until an edit rebuilds",
                        first_scope.label(),
                    );
                    if let Some(lazy) = &lazy {
                        lazy.failed(&error);
                    }
                }
            }
        }

        // The fill compiled the rest of the app, which can reach source directories the
        // first build never did. Watch the ones the running watcher does not already cover;
        // the two feed one channel (see `start_watcher_paths_into`).
        let extra_roots =
            uncovered_watch_roots(&watch_roots_for(&client, &react_server, &ssr), &watch_roots);
        if !extra_roots.is_empty() {
            announce_roots(&extra_roots);
            start_watcher_paths_into(&extra_roots, watch_events.clone())?;
            watch_roots.extend(extra_roots);
        }
        // The shallowest watched directory; everything watched lives under it.
        let watch_base = watch_roots
            .iter()
            .map(|(root, _)| root.clone())
            .min_by_key(|root| root.components().count())
            .unwrap_or_else(|| project_root.to_path_buf());

        let result = next_watch_loop(
            &receiver,
            project_root,
            &output_root,
            &rsc_root,
            &next_server_script,
            node_port,
            &mut node,
            &mut client,
            &mut react_server,
            &mut ssr,
            &hub,
            emit_options,
            &watch_base,
            &live_scope,
        );
        let _ = node.kill();
        let _ = node.wait();
        result
    }

    /// Build the client graph (browser bundle + RSC seam + Manifest #1), leaving the
    /// bundler alive. Dev config: development React (Fast Refresh hook) + HMR
    /// instrumentation.
    fn build_next_client(
        project_root: &Path,
        output_root: &Path,
        options: EmitOptions,
        scope: &RouteScope,
    ) -> Result<EnvBuild, String> {
        let build =
            discover_next_env_reconciled(project_root, "client", options, scope, &|config| {
                // The client needs `#diffpack-call-server` so a `"use server"` client stub
                // resolves its transport (harmless/unreachable when there is no action).
                config.build.virtual_modules.push((
                    crate::rsc::CALL_SERVER_SPECIFIER.to_string(),
                    crate::rsc::call_server_module_source().to_string(),
                ));
                Ok(())
            })?;
        emit_next_client(&build, project_root, output_root)?;
        Ok(build)
    }

    /// [`discover_next_env`] with the async-island reconcile loop: client islands
    /// are pinned lazily except the recorded async set; when discovery shows the
    /// recorded set is stale, the entries are regenerated (a fresh
    /// `configure_dev` pass rewrites them) and the graph rediscovered once.
    fn discover_next_env_reconciled(
        project_root: &Path,
        environment: &str,
        options: EmitOptions,
        scope: &RouteScope,
        prepare: &dyn Fn(&mut AppConfig) -> Result<(), String>,
    ) -> Result<EnvBuild, String> {
        let mut regenerated = false;
        loop {
            let mut config = next_config_dev(project_root, environment, scope)?;
            prepare(&mut config)?;
            let build = discover_next_env(config, environment, options)?;
            if !reconcile_next_async_islands(
                project_root,
                environment,
                &build.bundler,
                &build.reachable,
            )? {
                return Ok(build);
            }
            if regenerated {
                return Err(format!(
                    "dev {environment}: the async-island set did not stabilize after \
                     regenerating the entries once; this is a diffpack bug"
                ));
            }
            regenerated = true;
            println!(
                "[dev] async island set changed; regenerating the {environment} entry and rediscovering"
            );
        }
    }

    /// Build the react-server graph (flight render/action bundle) into the isolated
    /// `.rsc` root, then copy it to `<out>/rsc-render` and preserve its CSS.
    fn build_next_react_server(
        project_root: &Path,
        output_root: &Path,
        rsc_root: &Path,
        options: EmitOptions,
        scope: &RouteScope,
    ) -> Result<EnvBuild, String> {
        let mut config = next_config_dev(project_root, "react-server", scope)?;
        register_next_server_virtual_modules(&mut config, project_root)?;
        let build = discover_next_env(config, "react-server", options)?;
        emit_next_react_server(&build, output_root, rsc_root, EmitCancel::never())?;
        // The emit just wrote this graph's client-references manifest, whose keys are
        // exactly the islands a flight from these routes can reference. Record them: the
        // client and ssr graphs are configured next and pin precisely that set instead of
        // every `"use client"` file in the tree.
        crate::next_adapter::write_referenced_islands(
            project_root,
            &output_root.join(crate::rsc::REACT_SERVER_REFERENCES_MANIFEST_FILE),
        )?;
        Ok(build)
    }

    /// Build the ssr-of-flight graph into `<out>/server`, leaving the bundler alive.
    fn build_next_ssr(
        project_root: &Path,
        output_root: &Path,
        options: EmitOptions,
        scope: &RouteScope,
    ) -> Result<EnvBuild, String> {
        let build = discover_next_env_reconciled(project_root, "ssr", options, scope, &|config| {
            register_next_server_virtual_modules(config, project_root)
        })?;
        emit_next_ssr(&build, output_root, EmitCancel::never())?;
        Ok(build)
    }

    /// Scaffold `.diffpack-next/` + derive the dev [`AppConfig`] for one environment
    /// (a Next app-router project is guaranteed here — the caller already detected
    /// it, so `None` is an internal invariant break, a clear error not a silent skip).
    fn next_config_dev(
        project_root: &Path,
        environment: &str,
        scope: &RouteScope,
    ) -> Result<AppConfig, String> {
        crate::next_adapter::configure_app_router_dev(project_root, environment, scope)?.ok_or_else(
            || {
                format!(
                    "next dev: {} is not an app-router project (environment={environment})",
                    project_root.display()
                )
            },
        )
    }

    /// Discover one environment's graph and wrap it in a long-lived [`EnvBuild`].
    fn discover_next_env(
        config: AppConfig,
        label: &str,
        options: EmitOptions,
    ) -> Result<EnvBuild, String> {
        let mut config = config;
        config.build.source_maps = options.source_map;
        let entry = config
            .entry
            .clone()
            .ok_or_else(|| format!("no {label} entry found for the next app"))?;
        let (bundler, update) = crate::compiler::discover(&entry, &config.build)?;
        for warning in partition_diagnostics(&update.diagnostics, &format!("dev {label} build"))? {
            println!("[dev] warning: {warning}");
        }
        let session = bundler.direct_reachability();
        let reachable = session.reachable_modules();
        Ok(EnvBuild {
            bundler,
            session,
            reachable,
            options,
        })
    }

    /// Emit the client `public/` (chunks + CSS + copied static files + next/image
    /// variants) and persist both manifests the server graphs consume.
    fn emit_next_client(
        client: &EnvBuild,
        project_root: &Path,
        output_root: &Path,
    ) -> Result<EmitSummary, String> {
        let reachable = reachable_ids(client);
        // `public/rsc.css` belongs to the REACT-SERVER graph, which is built first in dev
        // and preserves its compiled sheet there. Without this the client's prune — which
        // deletes everything under `public/` the client graph did not itself write —
        // removed it, and the document went on linking `/rsc.css` (that link is guarded on
        // `rsc-render/server.css`, which the prune never touches) while `GET /rsc.css`
        // 404ed: an unstyled page, served with `X-Content-Type-Options: nosniff` so the
        // browser rejected the 404's HTML outright.
        //
        // Preserved on exactly the condition that makes the document link it — the sheet
        // sitting beside the render bundle — so the link and the artifact stay one fact,
        // and a react-server graph that stops compiling CSS still has its stale sheet
        // pruned.
        let mut preserve = BTreeSet::new();
        if output_root
            .join("rsc-render")
            .join(crate::next_adapter::RSC_EMITTED_CSS_FILE)
            .is_file()
        {
            preserve.insert(
                output_root
                    .join("public")
                    .join(crate::next_adapter::RSC_CSS_URL.trim_start_matches('/')),
            );
        }
        let summary = client.bundler.emit_public_preserving(
            &reachable,
            output_root,
            client.options,
            &preserve,
        )?;
        config::copy_static_public(project_root, &summary.output_dir)?;
        // next/image: emit downscaled responsive variants for every public raster.
        let images = crate::next_adapter::scan_public_images(project_root)?;
        if !images.is_empty() {
            crate::next_adapter::emit_image_variants(project_root, &summary.output_dir, &images)?;
        }
        // next/font/local: copy the app's font files to the hashed URLs the generated
        // @font-face rules point at, so dev serves the same face the build does.
        crate::next_font::emit_font_assets(project_root, &summary.output_dir)?;
        // Manifest #1 (client-references): the react-server render resolves each
        // `"use client"` `$$id` through it, and the orchestrator joins it with the
        // ssr manifest to build the SSR consumer manifest.
        let client_references = crate::rsc::client_references_from_bundle_graph(
            &client
                .bundler
                .integration_manifest_graph(&reachable, "client.js")?,
        );
        client_references.write(&output_root.join(crate::rsc::CLIENT_REFERENCES_MANIFEST_FILE))?;
        Ok(summary)
    }

    /// The LEAN client re-emit for a Fast Refresh hot update: incrementally re-render
    /// only the chunk(s) whose bytes changed (via `emit_public`), and NOTHING else.
    /// A same-graph island text edit does not move module ids, so the client-references
    /// / route manifests are byte-identical, the copied `public/` static assets are
    /// unchanged, and the next/image variants (e.g. a downscaled `hero.png`) do not
    /// need re-encoding. Keeping all of that OFF the hot-update critical path is what
    /// makes the edit-to-update latency competitive; the full emit (`emit_next_client`)
    /// still runs afterward, off the critical path, so a subsequent full document load
    /// remains correct.
    fn emit_next_client_hmr(
        client: &EnvBuild,
        output_root: &Path,
        cancel: EmitCancel<'_>,
    ) -> Result<(usize, bool), String> {
        let reachable = reachable_ids(client);
        client.bundler.emit_public_incremental_cancellable(
            &reachable,
            output_root,
            client.options,
            cancel,
        )
    }

    /// Emit the react-server graph into `<rsc_root>/server`, preserve its compiled CSS
    /// to `<out>/public/rsc.css`, write its own client-references manifest, and copy
    /// the bundle to `<out>/rsc-render` (where the orchestrator spawns it per GET).
    fn emit_next_react_server(
        env: &EnvBuild,
        output_root: &Path,
        rsc_root: &Path,
        cancel: EmitCancel<'_>,
    ) -> Result<(EmitSummary, bool), String> {
        let reachable = reachable_ids(env);
        let (summary, cancelled) = env.bundler.emit_server_into_cancellable(
            &reachable,
            &rsc_root.join("server"),
            env.options,
            cancel,
        )?;
        // Abandoned: the emitted tree is a partial rewrite of chunks whose patched
        // versions on disk are already current, so nothing downstream of it runs —
        // no CSS preserve, no manifest, and above all no copy of a half-written
        // bundle over the one the render worker loads.
        if cancelled {
            return Ok((summary, true));
        }
        // The react-server graph is authoritative for CSS; preserve it to the served,
        // non-pruned public/rsc.css. The render entry links it iff this same
        // `RSC_EMITTED_CSS_FILE` sits beside it (it is copied to `rsc-render/` with the
        // bundle below), so the <link> and the artifact are one fact.
        let css = rsc_root
            .join("server")
            .join(crate::next_adapter::RSC_EMITTED_CSS_FILE);
        if css.is_file() {
            let dest = output_root
                .join("public")
                .join(crate::next_adapter::RSC_CSS_URL.trim_start_matches('/'));
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
            }
            copy_file_if_changed(&css, &dest).map_err(|error| {
                format!(
                    "cannot preserve react-server CSS to {}: {error}",
                    dest.display()
                )
            })?;
        }
        // This build's OWN client-references manifest (its ids) — written beside the
        // ssr build's at `<out>` under the react-server-distinct file name, so the two
        // server-like graphs cannot clobber each other and the render seam reads the
        // SAME pair of files in dev as it does after a production build. It is also
        // the set that decides which client references a flight can carry, which the
        // seam validates the ssr manifest against.
        let server_references = crate::rsc::client_references_from_bundle_graph(
            &env.bundler
                .integration_manifest_graph(&reachable, "server.mjs")?,
        );
        server_references
            .write(&output_root.join(crate::rsc::REACT_SERVER_REFERENCES_MANIFEST_FILE))?;
        // Copy the fresh bundle to rsc-render (the orchestrator's per-request child).
        replace_dir(&rsc_root.join("server"), &output_root.join("rsc-render"))?;
        Ok((summary, false))
    }

    /// Emit the ssr-of-flight graph into `<out>/server` and write ITS client-references
    /// manifest (its ids) to `<out>` — the manifest the orchestrator reads as the SSR
    /// half of the divergent-id module map.
    fn emit_next_ssr(
        env: &EnvBuild,
        output_root: &Path,
        cancel: EmitCancel<'_>,
    ) -> Result<(EmitSummary, bool), String> {
        let reachable = reachable_ids(env);
        let (summary, cancelled) = env.bundler.emit_server_into_cancellable(
            &reachable,
            &output_root.join("server"),
            env.options,
            cancel,
        )?;
        if cancelled {
            return Ok((summary, true));
        }
        let server_references = crate::rsc::client_references_from_bundle_graph(
            &env.bundler
                .integration_manifest_graph(&reachable, "server.mjs")?,
        );
        server_references.write(&output_root.join(crate::rsc::SERVER_REFERENCES_MANIFEST_FILE))?;
        Ok((summary, false))
    }

    /// Where per-edit server micro-chunks are written, under the output dir so the
    /// watcher already ignores it and no graph's emit prunes it.
    const HOT_DIR: &str = ".hot";

    /// One pushed micro-chunk: the file a live Node process imports, and the runtime
    /// ids whose factories it re-registers.
    #[derive(Clone)]
    struct HotUpdate {
        chunk: PathBuf,
        ids: Vec<usize>,
    }

    /// The two server graphs a batch may have touched, each with the module ids that
    /// changed in it.
    struct HotGraphs<'a> {
        ssr: Option<(&'a EnvBuild, &'a BTreeSet<String>)>,
        react_server: Option<(&'a EnvBuild, &'a BTreeSet<String>)>,
    }

    /// The dev server → orchestrator hot-update channel: the Next analogue of the
    /// TanStack control endpoint ([`hmr_reload_server`]), at the granularity that
    /// actually changes.
    ///
    /// Node caches an ES module by URL forever, so the only way an edit reaches an
    /// already-loaded server graph is a URL Node has never seen. Re-importing the
    /// ENTRY under a fresh `?v=` cannot be that URL: the entry reaches its split
    /// chunks through query-less `import("./server.chunk-N.mjs")`, which Node answers
    /// from its cache — so the fresh entry binds stale factories (on cal.com's 69-chunk
    /// react-server graph that was not even silent: the fresh runtime's id table and
    /// the cached chunks' registrations disagreed and the worker died with
    /// `Module is not loaded: <id>`).
    ///
    /// So each edit gets its own tiny register-only chunk holding ONLY the changed
    /// modules, at a filename no previous edit used. The live runtime imports it,
    /// re-registers exactly those factories, and `serverInvalidate` re-runs them and
    /// their importers up to the entry — leaving React and every untouched dependency
    /// cached. Per edit that is ~1 KB of new module, not a graph-wide re-import: the
    /// inverse failure (re-importing the whole SSR graph per keystroke, and leaking a
    /// copy of it every time) is what the granularity choice avoids.
    struct HotChannel {
        /// Monotonic, so each micro-chunk gets a URL Node has never resolved.
        seq: u64,
        /// react-server updates that are NOT yet on disk in full. The orchestrator
        /// replays these into any worker it respawns, so a crash-respawn in the window
        /// between the hot push and the deferred full re-emit is still caught up.
        pending_react_server: Vec<HotUpdate>,
        /// Micro-chunk files the orchestrator may still hold a reference to. Deleted
        /// once a later push has told it a shorter list.
        live_files: Vec<PathBuf>,
    }

    impl HotChannel {
        fn new(output_root: &Path) -> Result<Self, String> {
            let mut channel = Self {
                seq: 0,
                pending_react_server: Vec::new(),
                live_files: Vec::new(),
            };
            channel.reset(output_root)?;
            Ok(channel)
        }

        /// Drop every pending update and clear the directory. Called when all three
        /// graphs were just re-emitted from scratch and the orchestrator restarted, so
        /// disk IS the truth and no replay is owed.
        fn reset(&mut self, output_root: &Path) -> Result<(), String> {
            self.pending_react_server.clear();
            self.live_files.clear();
            let dir = output_root.join(HOT_DIR);
            if dir.exists() {
                fs::remove_dir_all(&dir)
                    .map_err(|error| format!("cannot clear {}: {error}", dir.display()))?;
            }
            fs::create_dir_all(&dir)
                .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
            Ok(())
        }

        /// The deferred full re-emit has landed: `rsc-render/` on disk now contains
        /// every react-server edit pushed so far, so a respawned worker needs no
        /// replay. The files stay until the next push tells the orchestrator to drop
        /// them (it is still holding the list it was last sent).
        fn mark_react_server_on_disk(&mut self) {
            self.pending_react_server.clear();
        }

        /// The sequence number of the most recent push — names its micro-chunks
        /// (`.hot/ssr.<seq>.mjs`, `.hot/rsc.<seq>.mjs`) for the disk patcher.
        fn last_seq(&self) -> u64 {
            self.seq
        }

        /// Render a micro-chunk per changed server graph and apply them to the live
        /// Node processes, returning a log fragment. Returns `Err` — never a quiet
        /// no-op — if the orchestrator could not apply one: the alternative is serving
        /// HTML that disagrees with the editor.
        fn push(
            &mut self,
            node_port: u16,
            output_root: &Path,
            graphs: HotGraphs<'_>,
            profile: &mut EditProfile,
        ) -> Result<String, String> {
            if graphs.ssr.is_none() && graphs.react_server.is_none() {
                return Ok("server: no change".to_string());
            }
            self.seq += 1;
            let dir = output_root.join(HOT_DIR);
            fs::create_dir_all(&dir)
                .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
            let ssr = profile
                .stage("hot-render-ssr", || {
                    graphs
                        .ssr
                        .map(|(env, ids)| {
                            Self::render(env, ids, &dir.join(format!("ssr.{}.mjs", self.seq)))
                        })
                        .transpose()
                })?
                .flatten();
            let react_server = profile
                .stage("hot-render-rsc", || {
                    graphs
                        .react_server
                        .map(|(env, ids)| {
                            Self::render(env, ids, &dir.join(format!("rsc.{}.mjs", self.seq)))
                        })
                        .transpose()
                })?
                .flatten();

            let mut pending = self.pending_react_server.clone();
            if let Some(update) = &react_server {
                pending.push(update.clone());
            }
            let payload = serde_json::json!({
                "ssr": ssr.as_ref().map(Self::json),
                "reactServer": react_server.as_ref().map(Self::json),
                "pendingReactServer": pending.iter().map(Self::json).collect::<Vec<_>>(),
            })
            .to_string();
            profile.stage("hot-post", || {
                post_json(node_port, "/__diffpack_dev/hot", &payload)
            })?;

            // The orchestrator now holds exactly `pending`; anything it no longer
            // references (including the SSR chunk, which it imported synchronously
            // during the POST above and never replays) can go.
            let keep = pending
                .iter()
                .map(|update| update.chunk.clone())
                .collect::<BTreeSet<_>>();
            for file in std::mem::take(&mut self.live_files) {
                if !keep.contains(&file) {
                    // The micro-chunk carries a source-map sidecar (`<chunk>.map`) —
                    // the code being edited right now is exactly the code whose stack
                    // traces matter — so retire the pair together.
                    let mut map = file.clone().into_os_string();
                    map.push(".map");
                    let _ = fs::remove_file(&file);
                    let _ = fs::remove_file(PathBuf::from(map));
                }
            }
            self.live_files = keep.iter().cloned().collect();
            if let Some(update) = &ssr {
                self.live_files.push(update.chunk.clone());
            }
            self.pending_react_server = pending;

            Ok(format!(
                "server hot: ssr {} module(s), react-server {} module(s)",
                ssr.as_ref().map_or(0, |update| update.ids.len()),
                react_server.as_ref().map_or(0, |update| update.ids.len()),
            ))
        }

        /// Render one graph's micro-chunk. `None` when no changed module is live in
        /// that graph (e.g. it was tree-shaken away), which is a real "nothing to
        /// apply", not a swallowed failure.
        fn render(
            env: &EnvBuild,
            changed: &BTreeSet<String>,
            path: &Path,
        ) -> Result<Option<HotUpdate>, String> {
            let reachable = reachable_ids(env);
            // Both server graphs emit through `emit_server`, whose entry chunk is
            // `server.mjs`; the ids must be the ones that emit assigned or the
            // registration would bind against the wrong modules.
            let located = env.bundler.hmr_locate(&reachable, changed, "server.mjs")?;
            if located.is_empty() {
                return Ok(None);
            }
            // Node ESM, matching `emit_server`: a module that references `__dirname` /
            // `__filename` binds the entry's REAL values there, where browser output
            // would hand it `"/index.js"` / `"/"` stubs. Getting this wrong would only
            // show up after a hot update, on a server module that reads a file.
            if !env.bundler.write_hmr_chunk(
                &reachable,
                changed,
                "server.mjs",
                env.options,
                ModuleFormat::Esm,
                path,
            )? {
                return Ok(None);
            }
            Ok(Some(HotUpdate {
                chunk: path.to_path_buf(),
                ids: located.iter().map(|located| located.runtime_id).collect(),
            }))
        }

        fn json(update: &HotUpdate) -> serde_json::Value {
            serde_json::json!({ "chunk": update.chunk.to_string_lossy(), "ids": update.ids })
        }
    }

    /// How long the watch loop waits for another file event before running one step
    /// of chunk COMPACTION (the full re-emit of a touched graph).
    ///
    /// Compaction is housekeeping, not correctness: after every edit the on-disk
    /// chunks are already patched in place with the edit's own micro-chunk (see
    /// [`ChunkPatcher`]), so a full reload or a respawned worker reads CURRENT code at
    /// all times. Compaction only rewrites the accumulated patch tail into a pristine
    /// chunk (restoring real source maps for patched regions and bounding file
    /// growth).
    ///
    /// It used to wait TEN SECONDS, because it cost ~0.7s of loop-thread time per
    /// graph and an edit landing inside that window had to wait it out. A long idle
    /// only makes that collision rarer — at any steady typing cadence slower than the
    /// idle it happens on every edit, which is exactly the cliff that was measured
    /// (an edit at a ~1/sec cadence taking ~1s instead of 50ms). The fix is not a
    /// bigger number: the pass now carries an [`EventEpoch`] and is ABANDONED within
    /// a millisecond or two of a file event (see [`EmitCancel`]),
    /// keeping its debt for the next quiet moment. With the collision cost gone, the
    /// idle can be short — disk catches up right after a pause instead of ten seconds
    /// later — and it is short.
    const SETTLE_MS: u64 = 750;

    /// Stage-by-stage timing of one edit, appended to the `[dev]` summary line when
    /// `DIFFPACK_DEV_PROFILE=1`. The headline edit number is a single total; when it
    /// regresses there is no way to aim a fix without knowing which stage grew, and
    /// guessing has already cost one round of wasted work.
    #[derive(Default)]
    struct EditProfile {
        on: bool,
        stages: Vec<(&'static str, f64)>,
    }

    impl EditProfile {
        fn new() -> Self {
            Self {
                on: std::env::var_os("DIFFPACK_DEV_PROFILE").is_some(),
                stages: Vec::new(),
            }
        }

        /// Time `body`, recording it under `name`. Always runs `body` — the profile is
        /// observation only, so an unprofiled run takes exactly the same path.
        fn stage<T>(&mut self, name: &'static str, body: impl FnOnce() -> T) -> T {
            if !self.on {
                return body();
            }
            let started = Instant::now();
            let value = body();
            self.stages
                .push((name, started.elapsed().as_secs_f64() * 1_000.0));
            value
        }

        fn note(&mut self, name: &'static str, ms: f64) {
            if self.on {
                self.stages.push((name, ms));
            }
        }

        /// ` | profile: a=1.2ms b=3.4ms`, or empty when profiling is off.
        fn label(&self) -> String {
            if !self.on || self.stages.is_empty() {
                return String::new();
            }
            let mut merged: Vec<(&'static str, f64)> = Vec::new();
            for (name, ms) in &self.stages {
                match merged.iter_mut().find(|(seen, _)| seen == name) {
                    Some((_, total)) => *total += ms,
                    None => merged.push((name, *ms)),
                }
            }
            let body = merged
                .iter()
                .map(|(name, ms)| format!("{name}={ms:.1}ms"))
                .collect::<Vec<_>>()
                .join(" ");
            format!(" | profile: {body}")
        }
    }

    /// How long ago `path` was written, in milliseconds — the file-event detection
    /// latency (FSEvents delivery + the coalesce window + dedup) that precedes every
    /// stage the rebuild itself can see. Measured against the edited file's own mtime
    /// so it covers the part of the edit budget the dev server spends before it has
    /// even woken up. `None` when the file has no readable mtime (e.g. deleted).
    fn detection_lag_ms(path: &Path) -> Option<f64> {
        let modified = std::fs::metadata(path).ok()?.modified().ok()?;
        Some(
            SystemTime::now()
                .duration_since(modified)
                .ok()?
                .as_secs_f64()
                * 1_000.0,
        )
    }

    /// Which graphs owe disk a full chunk re-emit. See the `owed` binding in
    /// [`next_watch_loop`] for why this is deferred and coalesced rather than run
    /// inside the edit.
    #[derive(Clone, Copy, Default)]
    struct OwedEmits {
        client: bool,
        ssr: bool,
        react_server: bool,
    }

    impl OwedEmits {
        fn any(self) -> bool {
            self.client || self.ssr || self.react_server
        }

        fn label(self) -> String {
            let names = [
                self.client.then_some("client"),
                self.ssr.then_some("ssr"),
                self.react_server.then_some("react-server"),
            ];
            names.into_iter().flatten().collect::<Vec<_>>().join(" + ")
        }
    }

    /// Re-render the full chunks of every graph that owes disk one, so `public/` and
    /// `rsc-render/` match what the live processes are already running. Clears the debt
    /// only for graphs that succeeded, so a failing emit is retried rather than lost.
    /// Millisecond-cost patching of the big on-disk chunks after a hot update, so
    /// disk is never stale between edits and the deferred full re-emit becomes pure
    /// COMPACTION rather than a correctness requirement.
    ///
    /// Every dev chunk (main and split, all three graphs) has the same shape: one
    /// IIFE that registers its module factories into the globalThis-keyed registry
    /// and ends with a fixed tail —
    /// `if(import.meta...__diffpack_hmr...)return __runtime; return __runtime.require(N); })(); export default ...` —
    /// and registration is last-wins (`Object.assign` in `__register`). A hot
    /// micro-chunk is itself such an IIFE over ONLY the changed modules, with its
    /// own scope-isolated consts. So splicing the micro-chunk's IIFE into a chunk
    /// file immediately BEFORE the tail makes a fresh evaluation of that file run
    /// base registrations, then the patch registrations (overriding the stale
    /// factories), then the entry — byte-cheap (a few KB write) and semantically
    /// the same replay the live runtime already performed. A full page reload or a
    /// respawned worker then reads current code straight from disk.
    ///
    /// The sidecar map gets one explicit UNMAPPED (single-field) segment per
    /// appended line, inserted at the splice line's position in `mappings` — never
    /// omitted, because a consumer resolves an omitted position to the previous
    /// mapping and would attribute patched code to some unrelated module (the
    /// doctrine of `compose_source_map`). Single-field segments carry no source
    /// index, so nothing downstream in the delta chain needs re-basing. Real maps
    /// for the patched region come back at compaction; live debugging of the
    /// edited module meanwhile uses the micro-chunk's own map, which is the one
    /// the running page actually loaded.
    struct ChunkPatcher {
        path: PathBuf,
        map_path: PathBuf,
        /// Byte offset where the next patch inserts (start of the sentinel tail).
        splice: u64,
        tail: Vec<u8>,
        /// Line index of the splice point (number of `\n` before it).
        splice_line: usize,
        /// Byte offset in the MAP file where the next `A;` markers insert, or None
        /// when map patching is unavailable (map missing/unparseable — chunk
        /// patching still proceeds; the map is then stale-but-unshifted only for
        /// the tail lines, which are unmapped anyway).
        map_splice: Option<u64>,
    }

    impl ChunkPatcher {
        const SENTINEL: &'static [u8] =
            b"\nif(import.meta&&import.meta.url&&import.meta.url.indexOf(\"__diffpack_hmr\")";

        fn open(path: &Path) -> Result<Self, String> {
            let bytes = fs::read(path)
                .map_err(|error| format!("cannot read {} for patching: {error}", path.display()))?;
            let splice = find_last(&bytes, Self::SENTINEL).ok_or_else(|| {
                format!(
                    "{} has no hmr sentinel tail; is this a dev (hmr) chunk?",
                    path.display()
                )
            })? + 1; // keep the preceding newline with the base
            let tail = bytes[splice..].to_vec();
            let splice_line = bytes[..splice].iter().filter(|&&b| b == b'\n').count();
            let map_path = path_with_suffix_local(path, ".map");
            let map_splice = Self::map_offset(&map_path, splice_line);
            Ok(Self {
                path: path.to_path_buf(),
                map_path,
                splice: splice as u64,
                tail,
                splice_line,
                map_splice,
            })
        }

        /// The byte offset in the map file right after `splice_line` semicolons of
        /// its `mappings` string — where per-appended-line unmapped markers insert.
        fn map_offset(map_path: &Path, splice_line: usize) -> Option<u64> {
            let bytes = fs::read(map_path).ok()?;
            let key = b"\"mappings\":\"";
            let start = find_last(&bytes, key)? + key.len();
            let mut seen = 0usize;
            let mut at = start;
            while at < bytes.len() && seen < splice_line {
                match bytes[at] {
                    b';' => seen += 1,
                    b'"' => break, // mappings ended before the splice line: insert here
                    _ => {}
                }
                at += 1;
            }
            Some(at as u64)
        }

        /// Extract the registration IIFE from a micro-chunk's source: everything
        /// from its `const __diffpackEntry=(()=>{` (the file prelude above it is
        /// import statements, illegal mid-file) through the end, minus the
        /// `//# sourceMappingURL=` line (the base chunk has its own).
        fn patch_body(micro_chunk_source: &str) -> Result<String, String> {
            let start = micro_chunk_source
                .find("const __diffpackEntry=(()=>{")
                .ok_or_else(|| "micro-chunk has no registration IIFE".to_string())?;
            let body = &micro_chunk_source[start..];
            // Cut at the IIFE's own close. The micro-chunk's trailing
            // `export default __diffpackEntry;` and sourceMappingURL comment must
            // NOT be carried along: an `export` statement spliced inside the base
            // chunk's IIFE is a SyntaxError that would corrupt the whole file.
            let close = body
                .rfind("})();")
                .ok_or_else(|| "micro-chunk registration IIFE never closes".to_string())?;
            let mut body = body[..close + "})();".len()].to_string();
            body.push('\n');
            Ok(body)
        }

        fn append(&mut self, micro_chunk_source: &str) -> Result<(), String> {
            use std::io::{Seek, SeekFrom, Write};
            let body = Self::patch_body(micro_chunk_source)?;
            let lines = body.bytes().filter(|&b| b == b'\n').count();
            let mut file = fs::OpenOptions::new()
                .write(true)
                .open(&self.path)
                .map_err(|error| format!("cannot open {}: {error}", self.path.display()))?;
            file.seek(SeekFrom::Start(self.splice))
                .and_then(|_| file.write_all(body.as_bytes()))
                .and_then(|_| file.write_all(&self.tail))
                .and_then(|_| file.set_len(self.splice + (body.len() + self.tail.len()) as u64))
                .map_err(|error| format!("cannot patch {}: {error}", self.path.display()))?;
            self.splice += body.len() as u64;
            self.splice_line += lines;
            if let Some(map_at) = self.map_splice {
                // `A` = one explicit unmapped segment per appended line.
                let markers = "A;".repeat(lines);
                let patched = (|| -> std::io::Result<()> {
                    let mut map = fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&self.map_path)?;
                    map.seek(SeekFrom::Start(map_at))?;
                    let mut rest = Vec::new();
                    std::io::Read::read_to_end(&mut map, &mut rest)?;
                    map.seek(SeekFrom::Start(map_at))?;
                    map.write_all(markers.as_bytes())?;
                    map.write_all(&rest)?;
                    Ok(())
                })();
                match patched {
                    Ok(()) => self.map_splice = Some(map_at + markers.len() as u64),
                    Err(error) => {
                        // The chunk is patched; losing map markers only mis-shifts
                        // the (unmapped) tail lines. Do not fail the edit for it.
                        eprintln!(
                            "[dev] map patch failed for {} ({error}); disabling map patching until compaction",
                            self.map_path.display()
                        );
                        self.map_splice = None;
                    }
                }
            }
            Ok(())
        }
    }

    fn find_last(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() || haystack.len() < needle.len() {
            return None;
        }
        (0..=haystack.len() - needle.len())
            .rev()
            .find(|&i| &haystack[i..i + needle.len()] == needle)
    }

    fn path_with_suffix_local(path: &Path, suffix: &str) -> PathBuf {
        let mut s = path.as_os_str().to_owned();
        s.push(suffix);
        PathBuf::from(s)
    }

    /// Apply one micro-chunk to every on-disk chunk that HOSTS one of the changed
    /// modules (last-wins registration makes the patch authoritative wherever the
    /// module's home chunk is evaluated — main, prerequisite, or lazily loaded).
    /// Patchers are cached per file and dropped when the graph re-emits.
    #[allow(clippy::too_many_arguments)]
    fn append_hot_patch(
        patchers: &mut HashMap<PathBuf, ChunkPatcher>,
        env: &EnvBuild,
        changed_ids: &BTreeSet<String>,
        entry_name: &str,
        chunk_dir: &Path,
        micro_chunk_path: &Path,
    ) -> Result<usize, String> {
        let source = fs::read_to_string(micro_chunk_path).map_err(|error| {
            format!(
                "cannot read micro-chunk {}: {error}",
                micro_chunk_path.display()
            )
        })?;
        let located = env
            .bundler
            .hmr_locate(&reachable_ids(env), changed_ids, entry_name)?;
        let files: BTreeSet<PathBuf> = located
            .iter()
            .map(|l| chunk_dir.join(&l.chunk_file))
            .collect();
        for file in &files {
            if !patchers.contains_key(file) {
                patchers.insert(file.clone(), ChunkPatcher::open(file)?);
            }
            patchers
                .get_mut(file)
                .expect("just inserted")
                .append(&source)?;
        }
        Ok(files.len())
    }

    /// Flushes ONE owed graph per call, so the loop returns to its event channel
    /// between graphs: an edit that lands mid-flush waits for at most a single
    /// graph's emit, not the whole set. The caller keeps invoking on quiet settle
    /// timeouts until `owed.any()` is false. Returns `(which graph, rendered)`.
    fn flush_owed_step(
        owed: &mut OwedEmits,
        output_root: &Path,
        rsc_root: &Path,
        client: &EnvBuild,
        react_server: &EnvBuild,
        ssr: &EnvBuild,
        // Abandoned the moment a file event arrives: this is housekeeping, and the
        // developer's next keystroke outranks it. A graph whose pass was abandoned
        // stays owed, so the next quiet moment picks it up again.
        epoch: EventEpoch,
    ) -> Result<FlushStep, String> {
        let signal = move || epoch.superseded();
        let cancel = EmitCancel::when(&signal);
        if owed.client {
            let (rendered, cancelled) = emit_next_client_hmr(client, output_root, cancel)?;
            owed.client = cancelled;
            return Ok(FlushStep {
                which: "client",
                rendered,
                cancelled,
            });
        }
        if owed.ssr {
            let (summary, cancelled) = emit_next_ssr(ssr, output_root, cancel)?;
            owed.ssr = cancelled;
            return Ok(FlushStep {
                which: "ssr",
                rendered: summary.rendered_chunks,
                cancelled,
            });
        }
        if owed.react_server {
            let (summary, cancelled) =
                emit_next_react_server(react_server, output_root, rsc_root, cancel)?;
            owed.react_server = cancelled;
            return Ok(FlushStep {
                which: "react-server",
                rendered: summary.rendered_chunks,
                cancelled,
            });
        }
        Ok(FlushStep {
            which: "nothing",
            rendered: 0,
            cancelled: false,
        })
    }

    /// One compaction step's outcome: which graph it was for, how many chunks it
    /// re-rendered, and whether it was abandoned before finishing.
    struct FlushStep {
        which: &'static str,
        rendered: usize,
        cancelled: bool,
    }

    /// How long the loop waits for another file event before recompiling the
    /// STYLESHEETS (see [`refresh_next_stylesheets`]).
    ///
    /// A stylesheet edit does not wait for this at all — it is delivered inline on
    /// the edit, because the sheet IS that edit's visible result. This idle covers
    /// the other direction: under Tailwind, editing a component's class names
    /// changes the generated sheet even though no `.css` file was touched, and
    /// scanning candidates + recompiling costs real loop time. Waiting out a short
    /// typing pause keeps that off a burst of keystrokes while still restyling well
    /// inside a second of the last one.
    const CSS_SETTLE_MS: u64 = 400;

    /// Recompile ONLY the stylesheets and hand any that moved to open browsers as an
    /// in-place `<link>` swap.
    ///
    /// The sheets used to be written solely by the full environment re-emit, so a css
    /// edit inherited that pass's schedule: when compaction moved to a 10 s idle
    /// (chunk patching made it housekeeping), a css edit went with it and took 11.5 s
    /// to reach the browser. Nothing about a stylesheet needs a chunk render, so this
    /// runs the stylesheet pipeline on its own — the react-server graph is
    /// authoritative for the app's CSS and its sheet is preserved to the served
    /// `public/rsc.css` (and mirrored beside the `rsc-render` bundle, whose entry
    /// links it iff that file sits beside it), while the client graph writes its sheet
    /// straight into `public/`.
    ///
    /// Abandoned — and reported as such, so the caller keeps it owed — as soon as a
    /// file event arrives: recompiling a monorepo's sheet means rescanning its class
    /// candidates and running the app's own Tailwind, which is real loop time, and a
    /// result computed from superseded sources is worth nothing anyway.
    fn refresh_next_stylesheets(
        output_root: &Path,
        rsc_root: &Path,
        client: &EnvBuild,
        react_server: &EnvBuild,
        css_prints: &mut Vec<(String, Option<u64>)>,
        hub: &HmrHub,
        epoch: EventEpoch,
    ) -> Result<StylesheetRefresh, String> {
        let signal = move || epoch.superseded();
        let cancel = EmitCancel::when(&signal);
        if matches!(
            client.bundler.emit_stylesheet_only(
                &reachable_ids(client),
                &output_root.join("public").join("client.js"),
                cancel,
            )?,
            StylesheetEmit::Cancelled
        ) {
            return Ok(StylesheetRefresh::Abandoned);
        }
        let react_server_sheet = react_server.bundler.emit_stylesheet_only(
            &reachable_ids(react_server),
            &rsc_root.join("server").join("server.mjs"),
            cancel,
        )?;
        if matches!(react_server_sheet, StylesheetEmit::Cancelled) {
            return Ok(StylesheetRefresh::Abandoned);
        }
        if let StylesheetEmit::Written(sheet) = react_server_sheet {
            let served = output_root
                .join("public")
                .join(crate::next_adapter::RSC_CSS_URL.trim_start_matches('/'));
            if let Some(parent) = served.parent() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
            }
            copy_file_if_changed(&sheet, &served).map_err(|error| {
                format!(
                    "cannot preserve react-server CSS to {}: {error}",
                    served.display()
                )
            })?;
            // Keep the copy the per-request worker loads in step with the served one,
            // so a respawn in this window links a sheet whose bytes are current.
            let beside_render = output_root
                .join("rsc-render")
                .join(crate::next_adapter::RSC_EMITTED_CSS_FILE);
            if beside_render.exists() {
                copy_file_if_changed(&sheet, &beside_render).map_err(|error| {
                    format!(
                        "cannot mirror react-server CSS to {}: {error}",
                        beside_render.display()
                    )
                })?;
            }
        }
        let now = next_stylesheet_fingerprints(output_root);
        let changed: Vec<String> = now
            .iter()
            .zip(css_prints.iter())
            .filter(|(new, old)| new.1.is_some() && new.1 != old.1)
            .map(|(new, _)| new.0.clone())
            .collect();
        *css_prints = now;
        if !changed.is_empty() {
            diffpack_web::hmr::push_css_update(&changed, hub);
        }
        Ok(StylesheetRefresh::Done(changed.len()))
    }

    /// What a stylesheet refresh did: how many sheets it pushed, or that it gave way
    /// to an edit and is still owed.
    enum StylesheetRefresh {
        Done(usize),
        Abandoned,
    }

    /// Content fingerprints of the stylesheets the served document links, keyed by
    /// their public hrefs. Compared across refreshes: a changed sheet is delivered to
    /// open browsers via the HMR client's in-place `<link>` swap (`push_css`) — the
    /// Next topology previously rebuilt these sheets on a css edit and then never
    /// told anyone (KNOWN_ISSUES #7).
    fn next_stylesheet_fingerprints(output_root: &Path) -> Vec<(String, Option<u64>)> {
        ["/rsc.css", "/client.css"]
            .iter()
            .map(|href| {
                let file = output_root
                    .join("public")
                    .join(href.trim_start_matches('/'));
                let print = fs::read(&file).ok().map(|bytes| {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    bytes.hash(&mut hasher);
                    hasher.finish()
                });
                (href.to_string(), print)
            })
            .collect()
    }

    /// The incremental rebuild loop. Classifies each coalesced batch and applies the
    /// smallest correct update: a structural change → full rebuild + orchestrator
    /// restart + reload; an island edit → client+ssr rebuild + Fast Refresh WS update;
    /// a server-component edit → react-server rebuild + in-place RSC refresh. Both
    /// server graphs are hot-patched through [`HotChannel`] BEFORE the browser push, so
    /// the next full document load renders the edit.
    #[allow(clippy::too_many_arguments)]
    fn next_watch_loop(
        receiver: &Receiver<notify::Result<notify::Event>>,
        project_root: &Path,
        output_root: &Path,
        rsc_root: &Path,
        next_server_script: &Path,
        node_port: u16,
        node: &mut Child,
        client: &mut EnvBuild,
        react_server: &mut EnvBuild,
        ssr: &mut EnvBuild,
        hub: &HmrHub,
        emit_options: EmitOptions,
        // The topmost watched directory: dependency/generated-output exclusion is
        // judged relative to it (see `is_dependency_or_generated`).
        watch_base: &Path,
        // The routes the live graphs were built from; a rebuild must reproduce the SAME
        // scope, or an edit would silently widen or narrow what is served.
        scope: &RouteScope,
    ) -> Result<(), String> {
        // Publish the base the watch sources judge relevance against, so their event
        // COUNT (what deferred work cancels on) matches what this loop rebuilds for.
        diffpack_web::watch::set_base(watch_base);
        // Last processed `(mtime, len)` per path, so the FSEvents + fast-poller pair
        // (see `start_watcher_paths`) never rebuilds twice for one edit: whichever
        // source fires first is handled and recorded; the other's later echo reads the
        // same signature and is dropped.
        let mut processed: HashMap<PathBuf, (SystemTime, u64)> = HashMap::new();
        // Whether a build-error overlay is currently shown, so the next good rebuild
        // clears it (build-ok).
        let mut build_error_showing = false;
        // The chunk-granular hot-update channel into the live orchestrator + worker.
        // Constructing it clears `<out>/.hot` — the three graphs were just emitted from
        // scratch, so any micro-chunk left by a previous session is already superseded.
        let mut hot = HotChannel::new(output_root)?;
        // Per-chunk-file disk patchers (see ChunkPatcher). Dropped for a graph's
        // files whenever that graph re-emits (compaction or structural rebuild),
        // because a fresh emit rewrites the file under them.
        let mut patchers: HashMap<PathBuf, ChunkPatcher> = HashMap::new();
        // Pre-open patchers for the three main chunks so the first edit does not
        // pay their one-time init (a read+scan of ~60MB of chunk+map, ~0.7s
        // measured on cal.com). Failures are fine — the edit path re-opens lazily.
        for main in [
            output_root.join("public/client.js"),
            output_root.join("server/server.mjs"),
            output_root.join("rsc-render/server.mjs"),
        ] {
            if let Ok(patcher) = ChunkPatcher::open(&main) {
                patchers.insert(main, patcher);
            }
        }
        // Which graphs owe disk a full chunk re-emit. STICKY across iterations: the
        // re-emit is what makes a browser FULL RELOAD and a respawned react-server
        // worker correct, and nothing in a hot update needs it, so it runs when typing
        // STOPS rather than between two keystrokes. Re-rendering cal.com's 18.6 MB
        // client entry and 20.8 MB SSR entry costs ~1.2s; doing that inside every
        // keystroke's loop iteration would make a burst of edits queue behind it and
        // turn a 166ms hot update into a 1.2s one. Coalescing is sound because the emit
        // reads the CURRENT graph state, so one run after the burst covers every edit
        // in it.
        let mut owed = OwedEmits::default();
        // The served stylesheets as browsers last saw them.
        let mut css_prints = next_stylesheet_fingerprints(output_root);
        // Whether an edit may have moved the generated stylesheet WITHOUT touching a
        // `.css` file (a Tailwind class name added to a component). Cleared by the
        // short-idle recompile below; a real stylesheet edit never sets it, because
        // that one is delivered inline on the edit.
        let mut css_owed = false;
        loop {
            let first = if css_owed || owed.any() {
                // Two pending kinds of work, two very different idles: a stylesheet
                // that may have moved is recompiled after a short typing pause (the
                // browser is showing stale styling until it lands), while chunk
                // compaction waits for a long one (it is housekeeping, and costs ~1s
                // of loop time during which an edit would queue). Whichever is
                // pending picks this wait; both flush ONE step per quiet timeout, so
                // the loop returns to its channel between them.
                let idle = if css_owed { CSS_SETTLE_MS } else { SETTLE_MS };
                match receiver.recv_timeout(Duration::from_millis(idle)) {
                    Ok(event) => event,
                    Err(mpsc::RecvTimeoutError::Timeout) if css_owed => {
                        let started = Instant::now();
                        match refresh_next_stylesheets(
                            output_root,
                            rsc_root,
                            client,
                            react_server,
                            &mut css_prints,
                            hub,
                            EventEpoch::now(),
                        ) {
                            // Gave way to an edit: still owed, so the next quiet moment
                            // tries again against the newer sources. Logged under
                            // `DIFFPACK_DEV_PROFILE`, because "the sheet did not update
                            // and nothing was said" is otherwise indistinguishable from
                            // a bug.
                            Ok(StylesheetRefresh::Abandoned) => {
                                if std::env::var_os("DIFFPACK_DEV_PROFILE").is_some() {
                                    println!(
                                        "[dev] css refresh gave way to an edit after {:.1}ms (still owed)",
                                        started.elapsed().as_secs_f64() * 1_000.0,
                                    );
                                }
                            }
                            Ok(StylesheetRefresh::Done(0)) => css_owed = false,
                            Ok(StylesheetRefresh::Done(sheets)) => {
                                css_owed = false;
                                println!(
                                    "[dev] css hot-swap -> {sheets} sheet(s), {} browser(s) in {:.1}ms",
                                    hub.client_count(),
                                    started.elapsed().as_secs_f64() * 1_000.0,
                                );
                            }
                            Err(error) => {
                                css_owed = false;
                                eprintln!("[dev] stylesheet refresh failed: {error}");
                            }
                        }
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let started = Instant::now();
                        match flush_owed_step(
                            &mut owed,
                            output_root,
                            rsc_root,
                            client,
                            react_server,
                            ssr,
                            EventEpoch::now(),
                        ) {
                            Ok(step) => {
                                // Disk now matches the live processes for the
                                // react-server graph, so a worker respawned from here
                                // on needs no replay. Only a pass that FINISHED can say
                                // that: an abandoned one wrote a subset of the chunks.
                                if step.which == "react-server" && !step.cancelled {
                                    hot.mark_react_server_on_disk();
                                }
                                // The compacted emit rewrote chunk files out from under
                                // their patchers. Dropping a patcher only drops a cached
                                // file offset — the next patch reopens the file as it now
                                // stands — so doing it for the whole graph directory is
                                // correct whether the pass finished or was abandoned
                                // part-way through it.
                                let dir = match step.which {
                                    "client" => Some(output_root.join("public")),
                                    "ssr" => Some(output_root.join("server")),
                                    "react-server" => Some(output_root.join("rsc-render")),
                                    _ => None,
                                };
                                if let Some(dir) = dir {
                                    patchers.retain(|path, _| !path.starts_with(&dir));
                                }
                                if step.cancelled {
                                    println!(
                                        "[dev] compaction of {} dropped after {:.1}ms — an edit arrived; still owed: {}",
                                        step.which,
                                        started.elapsed().as_secs_f64() * 1_000.0,
                                        owed.label(),
                                    );
                                } else {
                                    println!(
                                        "[dev] compacted {} in {:.1}ms | rendered_chunks={}{}",
                                        step.which,
                                        started.elapsed().as_secs_f64() * 1_000.0,
                                        step.rendered,
                                        if owed.any() { " | more owed" } else { "" },
                                    );
                                }
                                // Compaction does not deliver CSS: it re-renders chunks
                                // from graph state the stylesheet pass already read, so
                                // the sheets it writes are the bytes browsers were sent
                                // on the edit itself. Delivery lives on the edit path
                                // and the short css idle above.
                                css_prints = next_stylesheet_fingerprints(output_root);
                            }
                            Err(error) => {
                                eprintln!("[dev] build error (kept serving): {error}");
                                hub.broadcast_build_error(&error);
                                build_error_showing = true;
                            }
                        }
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
                }
            } else {
                match receiver.recv() {
                    Ok(event) => event,
                    Err(_) => return Ok(()),
                }
            };
            let paths = coalesce_batch(receiver, first);
            let changed = paths
                .into_iter()
                .filter(|path| is_module_path(path))
                // Never react to a dependency or to generated output. The watch roots
                // are derived from the compiled module set, which in a monorepo means
                // watching a tree that CONTAINS `node_modules` and diffpack's own
                // `.diffpack-output` / `.diffpack-next`. Those are unknown to every
                // graph, so a write there would be classified as a new module — a
                // structural change — and a rebuild's own emit would retrigger the
                // rebuild, forever.
                .filter(|path| !is_dependency_or_generated(path, watch_base))
                .filter(|path| {
                    // Drop a path whose content signature is unchanged since we last
                    // processed it (a duplicate event from the other watch source). A
                    // path we cannot stat (e.g. just deleted) has no signature and is
                    // kept — a real change to react to.
                    match std::fs::metadata(path).and_then(|m| Ok((m.modified()?, m.len()))) {
                        Ok(sig) => {
                            if processed.get(path) == Some(&sig) {
                                false
                            } else {
                                processed.insert(path.clone(), sig);
                                true
                            }
                        }
                        Err(_) => true,
                    }
                })
                .collect::<BTreeSet<_>>();
            if changed.is_empty() {
                continue;
            }

            // A config-file edit can't be live re-derived; warn loudly, keep serving.
            if changed.iter().any(|path| is_config_file(path)) {
                println!(
                    "[dev] WARNING: a config file changed (next.config.* / package.json / tsconfig). Live config re-derivation is not implemented — the dev server is STILL USING THE STARTUP CONFIG. Restart `diffpack dev` to apply it."
                );
            }

            let known_by_any = |path: &Path| {
                client.bundler.is_known_module(path)
                    || react_server.bundler.is_known_module(path)
                    || ssr.bundler.is_known_module(path)
            };
            // A new module (exists, unknown to every graph) shifts ids across the whole
            // partition and cannot be hot-patched — full rebuild all three + restart +
            // reload (re-scans islands/routes too, so a new island/route is picked up).
            let structural = changed
                .iter()
                .any(|path| path.exists() && !known_by_any(path));
            if structural {
                let started = Instant::now();
                rebuild_all(
                    BuildPaths {
                        project_root,
                        output_root,
                        rsc_root,
                    },
                    client,
                    react_server,
                    ssr,
                    emit_options,
                    scope,
                )?;
                // Everything was re-emitted from scratch: no deferred debt survives, and
                // the worker replay list (whose micro-chunks the new bundles already
                // contain) is discharged.
                owed = OwedEmits::default();
                hot.reset(output_root)?;
                // A structural change means files appeared or vanished, so each graph's
                // per-file candidate scan no longer describes the tree.
                client.bundler.invalidate_tailwind_scan();
                react_server.bundler.invalidate_tailwind_scan();
                ssr.bundler.invalidate_tailwind_scan();
                // The rebuild wrote the sheets and the browser is reloading onto them,
                // so nothing is owed and the baseline moves with it — otherwise the
                // next idle pass would push a sheet the reload already fetched.
                css_owed = false;
                css_prints = next_stylesheet_fingerprints(output_root);
                restart_next_node(
                    node,
                    next_server_script,
                    project_root,
                    output_root,
                    node_port,
                )?;
                hub.broadcast_reload();
                println!(
                    "[dev] next structural change ({} file(s)) in {:.1}ms | full rebuild + reload",
                    changed.len(),
                    started.elapsed().as_secs_f64() * 1_000.0,
                );
                continue;
            }

            let started = Instant::now();
            let mut profile = EditProfile::new();
            // How much of the budget was already spent before this loop woke up.
            if let Some(lag) = changed
                .iter()
                .filter_map(|path| detection_lag_ms(path))
                .fold(None, |worst: Option<f64>, lag| {
                    Some(worst.map_or(lag, |worst| worst.max(lag)))
                })
            {
                profile.note("detect", lag);
            }
            let mut island_ids: BTreeSet<String> = BTreeSet::new();
            let mut ssr_ids: BTreeSet<String> = BTreeSet::new();
            let mut react_server_ids: BTreeSet<String> = BTreeSet::new();
            let mut client_c = EnvCounters::default();
            let mut server_c = EnvCounters::default();
            let mut server_reload = false;
            let mut graph_changed = false;

            // Catch edit-time build errors (e.g. a syntax error in the edited island or
            // server component) and surface them in the browser overlay instead of
            // killing the dev server; keep serving and clear the overlay on the next
            // good rebuild. The full rebuilds (`rebuild_all`) stay hard errors.
            // A non-stylesheet edit can add or remove a Tailwind class, which changes
            // what EVERY graph's sheet compiles to — the scan reads the source tree,
            // not the graph, so the file's owning graph is irrelevant. Only the edited
            // file is re-tokenized (~1 ms), not the tree (~660 ms on this app), because
            // a full rescan on the loop thread is exactly what made a JS edit collide
            // with the stylesheet pass.
            for path in changed.iter().filter(|path| !is_stylesheet_path(path)) {
                client.bundler.refresh_tailwind_scan_path(path);
                react_server.bundler.refresh_tailwind_scan_path(path);
                ssr.bundler.refresh_tailwind_scan_path(path);
            }

            let batch = (|| -> Result<(), String> {
                for path in &changed {
                    let is_island = profile.stage("classify", || {
                        let source = fs::read_to_string(path).unwrap_or_default();
                        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                        crate::rsc::detect_directive(&canonical, &source)
                            == Some(crate::rsc::RscDirective::Client)
                    });
                    if is_island && client.bundler.is_known_module(path) {
                        // Island edit — CRITICAL PATH: re-transform the changed module in
                        // the client graph AND in the SSR-of-flight graph, which is the
                        // one that renders this island into the served HTML. A same-graph
                        // island edit cannot move ids, so the manifests, the copied
                        // `public/` static assets and the next/image variants are all
                        // unchanged — none of that is re-run here (only a structural
                        // change, handled by `rebuild_all`, touches them).
                        let rebuilt = profile.stage("rebuild-client", || client.rebuild(path))?;
                        island_ids.extend(rebuilt.changed_ids.iter().cloned());
                        graph_changed |= rebuilt.graph_changed;
                        client_c.add(&rebuilt, 0);
                        owed.client = true;
                        if ssr.bundler.is_known_module(path) {
                            let rebuilt = profile.stage("rebuild-ssr", || ssr.rebuild(path))?;
                            ssr_ids.extend(rebuilt.changed_ids.iter().cloned());
                            graph_changed |= rebuilt.graph_changed;
                            server_c.add(&rebuilt, 0);
                            owed.ssr = true;
                        }
                    } else if react_server.bundler.is_known_module(path) {
                        // Server-component edit: re-transform ONLY the react-server graph.
                        // The persistent dev worker is hot-patched below, so the next
                        // `?__rsc=1` flight fetch renders this code — no worker respawn.
                        let rebuilt =
                            profile.stage("rebuild-react-server", || react_server.rebuild(path))?;
                        react_server_ids.extend(rebuilt.changed_ids.iter().cloned());
                        graph_changed |= rebuilt.graph_changed;
                        server_c.add(&rebuilt, 0);
                        owed.react_server = true;
                        server_reload = true;
                    } else {
                        // Known to neither the client nor the react-server graph as an
                        // editable leaf (e.g. a shared module the partition places
                        // elsewhere). Reconciling it precisely is out of scope for the
                        // fixture — force a correct full rebuild rather than guess.
                        graph_changed = true;
                    }
                }
                Ok(())
            })();
            if let Err(error) = batch {
                eprintln!("[dev] build error (kept serving): {error}");
                hub.broadcast_build_error(&error);
                build_error_showing = true;
                continue;
            }

            // A graph-structure change (import added/removed) re-partitions chunks and
            // shifts ids — the hot update's ESM re-import would fail to bind. A full
            // rebuild + reload is the correct, non-crashing path; it re-emits all three
            // graphs, so it also clears everything the hot channel was carrying.
            if graph_changed {
                rebuild_all(
                    BuildPaths {
                        project_root,
                        output_root,
                        rsc_root,
                    },
                    client,
                    react_server,
                    ssr,
                    emit_options,
                    scope,
                )?;
                // Every graph was just emitted from scratch: disk IS the truth, so the
                // deferred debt, the worker replay list, and every chunk patcher are
                // all discharged.
                owed = OwedEmits::default();
                hot.reset(output_root)?;
                patchers.clear();
                client.bundler.invalidate_tailwind_scan();
                react_server.bundler.invalidate_tailwind_scan();
                ssr.bundler.invalidate_tailwind_scan();
                // Same as the structural path: the sheets are on disk and the reload
                // will fetch them, so the baseline moves and nothing stays owed.
                css_owed = false;
                css_prints = next_stylesheet_fingerprints(output_root);
                restart_next_node(
                    node,
                    next_server_script,
                    project_root,
                    output_root,
                    node_port,
                )?;
                hub.broadcast_reload();
                println!(
                    "[dev] next rebuilt {} file(s) in {:.1}ms | graph changed -> full reload",
                    changed.len(),
                    started.elapsed().as_secs_f64() * 1_000.0,
                );
                continue;
            }

            // SERVER FIRST, on the critical path: hand each changed server graph a
            // micro-chunk holding ONLY its changed modules and have the live Node
            // processes swap them in place. This is what makes the next full document
            // load fresh, and it must happen BEFORE the browser is told anything —
            // otherwise a user who reloads the instant Fast Refresh lands sees the old
            // HTML. It is chunk-granular by construction: the micro-chunk carries the
            // changed modules and nothing else, so no unrelated chunk is re-imported
            // and no module leaks per keystroke beyond the ~1 KB update itself.
            let hot_result = hot.push(
                node_port,
                output_root,
                HotGraphs {
                    ssr: (!ssr_ids.is_empty()).then_some((&*ssr, &ssr_ids)),
                    react_server: (!react_server_ids.is_empty())
                        .then_some((&*react_server, &react_server_ids)),
                },
                &mut profile,
            );
            let hot_note = match hot_result {
                Ok(note) => note,
                Err(error) => {
                    // A hot update that did not land means the server would serve stale
                    // HTML for this edit. Say so loudly (overlay + log) rather than
                    // pushing a browser update that disagrees with the server.
                    let error = format!("dev server hot update failed: {error}");
                    eprintln!("[dev] {error}");
                    hub.broadcast_build_error(&error);
                    build_error_showing = true;
                    continue;
                }
            };

            // STYLESHEETS, still on the critical path: when a `.css` source was
            // edited, its recompiled sheet IS this edit's visible result, so it is
            // compiled and pushed here rather than waiting for any idle — the sheet
            // used to be a side-product of the deferred full re-emit, which put a css
            // edit 10s behind the keystroke. Every other edit only MIGHT have moved
            // the sheet (a Tailwind class name added to a component): worth a
            // candidate rescan, not worth it on a keystroke, so it is owed to the
            // short css idle at the top of the loop instead.
            let (css_note, css_failed) = if changed.iter().any(|path| is_stylesheet_path(path)) {
                let epoch = EventEpoch::now();
                match profile.stage("css-emit", || {
                    refresh_next_stylesheets(
                        output_root,
                        rsc_root,
                        client,
                        react_server,
                        &mut css_prints,
                        hub,
                        epoch,
                    )
                }) {
                    // A newer edit landed while this sheet was compiling, so this
                    // result describes sources nobody has any more. The newer edit
                    // owns the sheet now; if it was not itself a stylesheet edit, the
                    // short css idle picks the work up.
                    Ok(StylesheetRefresh::Abandoned) => {
                        css_owed = true;
                        (
                            " | stylesheet superseded by a newer edit".to_string(),
                            false,
                        )
                    }
                    Ok(StylesheetRefresh::Done(0)) => {
                        (" | stylesheet unchanged".to_string(), false)
                    }
                    Ok(StylesheetRefresh::Done(sheets)) => (
                        format!(
                            " | css hot-swap -> {sheets} sheet(s), {} browser(s)",
                            hub.client_count()
                        ),
                        false,
                    ),
                    Err(error) => {
                        // The browser would otherwise keep showing the old styling with
                        // nothing said. Surface it in the overlay; a later good edit
                        // clears it.
                        eprintln!("[dev] stylesheet refresh failed: {error}");
                        hub.broadcast_build_error(&error);
                        build_error_showing = true;
                        (" | STYLESHEET REFRESH FAILED".to_string(), true)
                    }
                }
            } else {
                css_owed = true;
                (String::new(), false)
            };

            // Then the browser push — the user-visible edit-to-update event.
            let update = if !island_ids.is_empty() {
                // State-preserving React Fast Refresh (no reload). Push a MICRO-CHUNK
                // (only the changed modules) so the browser re-parses ~1 KB, not the
                // ~1 MB entry chunk; served directly off disk by the proxy.
                let pushed = profile.stage("push-client", || {
                    hmr_push_client(client, &island_ids, hub, Some(output_root))
                });
                // And splice the same micro-chunk into the browser chunks on disk,
                // so a full reload loads current code without waiting for compaction.
                profile.stage("patch-disk-client", || {
                    if let Err(error) = append_hot_patch(
                        &mut patchers,
                        client,
                        &island_ids,
                        "client.js",
                        &output_root.join("public"),
                        &output_root.join("public/client.hmr.js"),
                    ) {
                        eprintln!("[dev] client disk patch skipped: {error}");
                    }
                });
                pushed
            } else if server_reload {
                // Server-component edit: an in-place RSC refresh (no full page reload).
                // The client refetches the current route's flight and diff-renders it;
                // the react-server worker is already running the new code (above), and
                // client-island state is preserved by React reconciliation.
                hub.broadcast_rsc_refresh();
                "server component -> in-place RSC refresh (no reload)".to_string()
            } else {
                "no visible change".to_string()
            };
            // DISK, right after the pushes and byte-cheaply: splice the same
            // micro-chunks into
            // the on-disk chunks that host the changed modules (see ChunkPatcher).
            // After this, a full reload or worker respawn reads CURRENT code from
            // disk, and the deferred re-emit below is pure compaction. A patch
            // failure is not an edit failure: it just means disk is as stale as it
            // always was pre-patching, and compaction restores it.
            profile.stage("patch-disk", || {
                let seq = hot.last_seq();
                let mut patched = 0usize;
                if !ssr_ids.is_empty() {
                    match append_hot_patch(
                        &mut patchers,
                        ssr,
                        &ssr_ids,
                        "server.mjs",
                        &output_root.join("server"),
                        &output_root.join(HOT_DIR).join(format!("ssr.{seq}.mjs")),
                    ) {
                        Ok(n) => patched += n,
                        Err(error) => eprintln!("[dev] ssr disk patch skipped: {error}"),
                    }
                }
                if !react_server_ids.is_empty() {
                    match append_hot_patch(
                        &mut patchers,
                        react_server,
                        &react_server_ids,
                        "server.mjs",
                        &output_root.join("rsc-render"),
                        &output_root.join(HOT_DIR).join(format!("rsc.{seq}.mjs")),
                    ) {
                        Ok(n) => patched += n,
                        Err(error) => eprintln!("[dev] rsc disk patch skipped: {error}"),
                    }
                }
                patched
            });

            let update_ms = started.elapsed().as_secs_f64() * 1_000.0;

            // The full chunk re-emit each touched graph now owes DISK is NOT run here.
            // Nothing in the update just delivered needs it — the browser has the
            // micro-chunk, both server graphs are hot-patched — and it costs ~1.2s on
            // cal.com, which inside the loop would make every keystroke of a burst queue
            // behind the previous one's re-emit. It runs at the top of the loop as soon
            // as `SETTLE_MS` passes with no new event, coalesced across the whole burst.
            // Until it lands, `hot.pending_react_server` keeps the worker replay list
            // non-empty, so a react-server worker respawned in that window is still
            // caught up from the micro-chunks.
            if build_error_showing && !css_failed {
                hub.broadcast_build_ok();
                build_error_showing = false;
            }

            println!(
                "[dev] next rebuilt {} file(s) | update in {update_ms:.1}ms | client transformed={} changed={} | server transformed={} changed={} | {hot_note} | {update}{css_note} | disk re-emit owed: {}{}",
                changed.len(),
                client_c.transformed,
                client_c.changed,
                server_c.transformed,
                server_c.changed,
                if owed.any() {
                    owed.label()
                } else {
                    "none".to_string()
                },
                profile.label(),
            );
        }
    }

    /// How long the first build waits for a request to tell it which route to compile.
    /// Short: a dev server is normally started with a browser already open or about to be,
    /// and if nobody asks, compiling the whole app is exactly the right thing to do.
    const FIRST_REQUEST_GRACE: Duration = Duration::from_millis(750);

    /// How quiet the server has to be before the fill starts, and how long the fill will
    /// wait for that quiet before starting anyway.
    const FILL_QUIET: Duration = Duration::from_millis(400);
    const FILL_QUIET_BUDGET: Duration = Duration::from_secs(20);

    /// How long to keep collecting wants after the first one, so a page load that fires a
    /// document plus several API calls compiles them in ONE build instead of one per
    /// request. Small enough to stay invisible against a multi-second build.
    const WANT_COALESCE_MS: u64 = 60;

    /// The scope of the first build: whatever the first requests asked for, or the whole
    /// app if nothing arrived within [`FIRST_REQUEST_GRACE`].
    ///
    /// HTTP endpoints (`route.ts`, `pages/api/**`) are compiled EAGERLY even under
    /// `DIFFPACK_DEV_LAZY=1`. A page whose API answers 404 is a broken app, not a page that
    /// is still compiling, and on cal.com the document immediately reads a next-auth
    /// session and several tRPC queries. `DIFFPACK_DEV_LAZY=api` makes them lazy as well
    /// and those requests WAIT rather than be answered wrongly — but it measured SLOWER
    /// than compiling everything up front (10.5s to the first document against 6.2s),
    /// because cal.com's server render calls the app's own API over HTTP, so the render
    /// itself sits waiting for an endpoint build. Kept because an app whose pages do not
    /// call their own API would win from it; do not reach for it without measuring.
    fn first_build_scope(lazy: &LazyRoutes) -> RouteScope {
        let lazy_endpoints = std::env::var("DIFFPACK_DEV_LAZY").as_deref() == Ok("api");
        println!("[dev] next: lazy route compilation is ON; waiting for the first request");
        let Some(wanted) = wait_for_first_wants(lazy) else {
            println!(
                "[dev] next: no request within {}ms — compiling the whole app",
                FIRST_REQUEST_GRACE.as_millis(),
            );
            return RouteScope::All;
        };
        let (pages, endpoints) = lazy.partition_wanted(wanted);
        if lazy_endpoints {
            RouteScope::pages_and_endpoints(pages, endpoints)
        } else {
            RouteScope::pages(pages)
        }
    }

    /// Block until a request registers a want, then keep collecting for
    /// [`WANT_COALESCE_MS`]. `None` means either nothing arrived within the grace period or
    /// a request matched no pattern at all — both of which mean "compile everything".
    fn wait_for_first_wants(lazy: &LazyRoutes) -> Option<BTreeSet<String>> {
        lazy.wait_for_first_wants(FIRST_REQUEST_GRACE, Duration::from_millis(WANT_COALESCE_MS))
    }

    /// One graph's contribution to a cold start / rebuild, for the log line.
    struct GraphTiming {
        ms: f64,
        modules: usize,
    }

    /// The three environment builds a pass produces, in the order they are always named:
    /// client, react-server, ssr.
    struct Graphs {
        client: EnvBuild,
        react_server: EnvBuild,
        ssr: EnvBuild,
        /// What each one cost, for the log line.
        timings: [(&'static str, GraphTiming); 3],
    }

    /// Build all three graphs for `scope`, in the order their facts flow.
    ///
    /// REACT-SERVER FIRST, then client and ssr concurrently. The react-server graph is
    /// what decides which islands exist as client references at all, so building it first
    /// lets the other two pin exactly that set (see [`REFERENCED_ISLANDS_FILE`]) instead
    /// of every `"use client"` file in the project. The client and ssr graphs read no
    /// output of each other's — client emits `public/`, ssr emits `server/` — so they run
    /// on two threads, and neither reads the react-server graph's output at build time.
    ///
    /// (The pre-existing order was client first, because TanStack Start's server graph
    /// imports a virtual module derived from the client build. A Next graph cannot import
    /// it; see `register_next_server_virtual_modules`.)
    fn build_all_graphs(
        project_root: &Path,
        output_root: &Path,
        rsc_root: &Path,
        emit_options: EmitOptions,
        scope: &RouteScope,
    ) -> Result<Graphs, String> {
        let started = Instant::now();
        let react_server =
            build_next_react_server(project_root, output_root, rsc_root, emit_options, scope)?;
        let react_server_timing = GraphTiming {
            ms: started.elapsed().as_secs_f64() * 1_000.0,
            modules: react_server.reachable.len(),
        };
        let (client_result, ssr_result) = std::thread::scope(|threads| {
            let client = threads.spawn(|| {
                let started = Instant::now();
                let build = build_next_client(project_root, output_root, emit_options, scope);
                (build, started.elapsed().as_secs_f64() * 1_000.0)
            });
            let ssr = threads.spawn(|| {
                let started = Instant::now();
                let build = build_next_ssr(project_root, output_root, emit_options, scope);
                (build, started.elapsed().as_secs_f64() * 1_000.0)
            });
            (client.join(), ssr.join())
        });
        let (client_result, client_ms) =
            client_result.map_err(|_| "the client build thread panicked".to_string())?;
        let (ssr_result, ssr_ms) =
            ssr_result.map_err(|_| "the ssr build thread panicked".to_string())?;
        let client = client_result?;
        let ssr = ssr_result?;
        let timings = [
            ("react-server", react_server_timing),
            (
                "client",
                GraphTiming {
                    ms: client_ms,
                    modules: client.reachable.len(),
                },
            ),
            (
                "ssr",
                GraphTiming {
                    ms: ssr_ms,
                    modules: ssr.reachable.len(),
                },
            ),
        ];
        Ok(Graphs {
            client,
            react_server,
            ssr,
            timings,
        })
    }

    /// The one-line summary of a [`build_all_graphs`] pass.
    fn graph_timing_label(timings: &[(&'static str, GraphTiming); 3]) -> String {
        timings
            .iter()
            .map(|(name, timing)| format!("{name} {:.0}ms ({} modules)", timing.ms, timing.modules))
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Where a rebuild reads its inputs from and writes its outputs to. Grouped because the
    /// four travel together everywhere a rebuild happens.
    #[derive(Clone, Copy)]
    struct BuildPaths<'a> {
        project_root: &'a Path,
        output_root: &'a Path,
        rsc_root: &'a Path,
    }

    /// Re-discover and re-emit all three graphs from scratch (used for structural /
    /// graph-changing edits, where module ids shift across the partition).
    fn rebuild_all(
        paths: BuildPaths<'_>,
        client: &mut EnvBuild,
        react_server: &mut EnvBuild,
        ssr: &mut EnvBuild,
        emit_options: EmitOptions,
        scope: &RouteScope,
    ) -> Result<(), String> {
        let fresh = build_all_graphs(
            paths.project_root,
            paths.output_root,
            paths.rsc_root,
            emit_options,
            scope,
        )?;
        *client = fresh.client;
        *react_server = fresh.react_server;
        *ssr = fresh.ssr;
        Ok(())
    }

    /// Spawn the next orchestrator (`node next-server.mjs <output-dir> <port>`) with
    /// `DIFFPACK_NEXT_DEV=1` so it re-imports the SSR bundle on change. stdout/stderr
    /// are inherited so orchestrator/worker errors surface in the dev console; stdin is
    /// a PIPE this process holds open. The orchestrator exits when that stdin closes
    /// (`next-server.mjs` watches for it), so when `diffpack dev` dies for ANY reason —
    /// including SIGKILL, where no Rust cleanup runs — the OS closes the pipe and the
    /// orchestrator (and, in turn, its persistent worker) shuts down instead of
    /// orphaning. The returned [`Child`] owns the write end; keeping it alive keeps the
    /// pipe open for the orchestrator's lifetime.
    fn spawn_next_node(
        script: &Path,
        project_root: &Path,
        output_root: &Path,
        port: u16,
    ) -> Result<Child, String> {
        let mut command = Command::new("node");
        command
            .arg(script)
            .arg(output_root)
            .arg(port.to_string())
            // `next dev` loads next.config in the process that then serves, so the
            // config's `process.env` side effects (cal.com's whole `.env`, loaded by a
            // `dotenv.config()` call in `next.config.ts`) are the environment the app's
            // server code runs under. Diffpack evaluates the config in a child process,
            // so the delta it recorded is handed to the orchestrator here — and through
            // it to the SSR worker.
            .envs(crate::next_adapter::config_env_from_manifest(project_root))
            .env("DIFFPACK_NEXT_DEV", "1")
            // The app's own working directory, as when a developer runs `next dev` from
            // it: server code that resolves a path relative to the cwd (a locale
            // directory, an on-disk template) must find it where the app expects.
            .current_dir(project_root);
        // The next/image optimizer (`/_next/image`) in the orchestrator shells back to
        // this binary for a native resize the build did not precompute.
        if let Ok(exe) = std::env::current_exe() {
            command.env("DIFFPACK_BIN", exe);
        }
        command
            .stdin(std::process::Stdio::piped())
            .spawn()
            .map_err(|error| {
                format!(
                    "cannot start next orchestrator ({}): {error}",
                    script.display()
                )
            })
    }

    /// Kill the orchestrator and spawn a fresh one on the same port (used for
    /// structural rebuilds, where the react-server/ssr bundles are re-derived).
    fn restart_next_node(
        node: &mut Child,
        script: &Path,
        project_root: &Path,
        output_root: &Path,
        port: u16,
    ) -> Result<(), String> {
        let _ = node.kill();
        let _ = node.wait();
        *node = spawn_next_node(script, project_root, output_root, port)?;
        wait_for_node(port)
    }

    /// Replace `dest` with a fresh recursive copy of `src` (used to publish the freshly
    /// emitted react-server bundle from `.rsc/server` to `rsc-render`).
    /// Move `fresh` into `live`'s place by RENAME, not by copy: `live` is a 626 MB tree on
    /// cal.com and the requests being held during the swap are waiting on it, so the swap
    /// has to be O(1). The displaced tree is renamed aside and deleted on a background
    /// thread, since nothing reads it any more.
    fn swap_output_dir(fresh: &Path, live: &Path) -> Result<(), String> {
        let displaced = path_with_suffix_local(live, ".old");
        if displaced.exists() {
            fs::remove_dir_all(&displaced)
                .map_err(|error| format!("cannot clear {}: {error}", displaced.display()))?;
        }
        if live.exists() {
            fs::rename(live, &displaced).map_err(|error| {
                format!(
                    "cannot move {} aside to {}: {error}",
                    live.display(),
                    displaced.display(),
                )
            })?;
        }
        fs::rename(fresh, live).map_err(|error| {
            format!(
                "cannot move {} into {}: {error}",
                fresh.display(),
                live.display()
            )
        })?;
        if displaced.exists() {
            let _ = std::thread::Builder::new()
                .name("diffpack-dev-output-cleanup".into())
                .spawn(move || {
                    let _ = fs::remove_dir_all(&displaced);
                });
        }
        Ok(())
    }

    fn replace_dir(src: &Path, dest: &Path) -> Result<(), String> {
        if dest.exists() {
            fs::remove_dir_all(dest)
                .map_err(|error| format!("cannot clear {}: {error}", dest.display()))?;
        }
        copy_dir_recursive(src, dest)
    }

    fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
        fs::create_dir_all(dest)
            .map_err(|error| format!("cannot create {}: {error}", dest.display()))?;
        let read =
            fs::read_dir(src).map_err(|error| format!("cannot read {}: {error}", src.display()))?;
        for entry in read {
            let entry = entry.map_err(|error| format!("cannot read {}: {error}", src.display()))?;
            let from = entry.path();
            let to = dest.join(entry.file_name());
            let file_type = entry
                .file_type()
                .map_err(|error| format!("cannot stat {}: {error}", from.display()))?;
            if file_type.is_dir() {
                copy_dir_recursive(&from, &to)?;
            } else {
                copy_file_if_changed(&from, &to)?;
            }
        }
        Ok(())
    }

    /// `fs::copy`, except a destination already holding the source's exact bytes
    /// is left alone. Skipping the write keeps the destination's mtime stable
    /// across a rebuild that reproduced it — which is what lets the dev warm
    /// start prove "the rebuild changed nothing" from an mtime snapshot — and
    /// costs a read the copy was going to do anyway.
    fn copy_file_if_changed(from: &Path, to: &Path) -> Result<(), String> {
        let source =
            fs::read(from).map_err(|error| format!("cannot read {}: {error}", from.display()))?;
        if fs::read(to).ok().as_deref() == Some(source.as_slice()) {
            return Ok(());
        }
        fs::write(to, source).map_err(|error| {
            format!(
                "cannot copy {} -> {}: {error}",
                from.display(),
                to.display()
            )
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A chunk patch must land BEFORE the sentinel tail (so the entry invoke
        /// still runs last), preserve the tail byte-for-byte, stack across multiple
        /// appends in edit order, and mark every appended line explicitly unmapped
        /// in the sidecar map at the splice line's position — never omitted, or a
        /// consumer would attribute patched code to the previous module's mapping.
        #[test]
        fn chunk_patcher_splices_before_the_tail_and_marks_the_map() {
            let dir = tempfile::tempdir().unwrap();
            let chunk = dir.path().join("client.js");
            let base = "const __diffpackEntry=(()=>{\n\
                        const __newModules={1:function(){}};\n\
                        __runtime.register(__newModules,__newMaps,__newChunks);\n\
                        if(import.meta&&import.meta.url&&import.meta.url.indexOf(\"__diffpack_hmr\")>=0)return __runtime;\n\
                        return __runtime.require(1);\n\
                        })();\n\
                        export default __diffpackEntry;\n";
            fs::write(&chunk, base).unwrap();
            // Map with one entry per line of the chunk (7 lines) — all "A" markers.
            let map = format!("{}.map", chunk.display());
            fs::write(&map, r#"{"version":3,"file":"client.js","names":[],"sources":[],"sourcesContent":[],"mappings":"A;A;A;A;A;A;A"}"#).unwrap();

            let micro = "import { x } from \"node:url\";\n\
                         const __diffpackEntry=(()=>{\n\
                         const __newModules={1:function(){/*patched*/}};\n\
                         __runtime.register(__newModules,__newMaps,__newChunks);\n\
                         if(import.meta&&import.meta.url&&import.meta.url.indexOf(\"__diffpack_hmr\")>=0)return __runtime;\n\
                         return __runtime.require(1);\n\
                         })();\n\
                         export default __diffpackEntry;\n\
                         //# sourceMappingURL=hot.mjs.map\n";

            let mut patcher = ChunkPatcher::open(&chunk).unwrap();
            patcher.append(micro).unwrap();
            patcher
                .append(&micro.replace("patched", "patched-again"))
                .unwrap();

            let out = fs::read_to_string(&chunk).unwrap();
            // Both patches present, in order, and before the sentinel tail.
            let first = out.find("/*patched*/").expect("first patch");
            let second = out.find("patched-again").expect("second patch");
            let tail = out.rfind("if(import.meta&&import.meta.url").unwrap();
            assert!(first < second && second < tail, "order: {out}");
            // The file prelude of the micro-chunk (an import statement, illegal
            // mid-file) must have been stripped.
            assert!(!out.contains("node:url"), "{out}");
            // The original tail survives byte-for-byte at the end.
            assert!(out.ends_with("export default __diffpackEntry;\n"), "{out}");
            // Each patch is a nested IIFE, so its consts cannot collide.
            assert_eq!(out.matches("const __diffpackEntry=(()=>{").count(), 3);
            // The micro-chunk's own `export default` must NOT be spliced in: an
            // export inside the base IIFE is a SyntaxError (caught live on
            // cal.com's server.mjs before this assertion existed). Exactly the
            // base's one export survives.
            assert_eq!(out.matches("export default").count(), 1, "{out}");

            // The map gained one explicit unmapped marker per appended line, at the
            // splice line (line 3 = index of the sentinel line), not at the end.
            let mapped = fs::read_to_string(&map).unwrap();
            let mappings = mapped
                .split("\"mappings\":\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap();
            let base_lines = 3; // lines before the sentinel in the base
            let patch_lines: usize = out.lines().count() - base.lines().count();
            let entries: Vec<&str> = mappings.split(';').collect();
            assert_eq!(
                entries.len(),
                7 + patch_lines,
                "one entry per line: {mappings}"
            );
            assert!(
                entries[base_lines..base_lines + patch_lines]
                    .iter()
                    .all(|e| *e == "A"),
                "appended lines are explicit unmapped markers: {mappings}"
            );
        }

        /// The embedded orchestrator must take its dev freshness from the hot-update
        /// channel and NOT from polling a bundle's mtime.
        ///
        /// The mtime cache is the exact defect this replaced, and it is the kind that
        /// reads as harmless: an island edit re-emits only the chunk that HOSTS the
        /// changed module, so `server/server.mjs`'s own mtime never moves and the cached
        /// module is returned forever — a `curl` after an edit served the OLD string for
        /// the life of the process while every browser-side HMR gate stayed green. The
        /// end-to-end proof is `scripts/rsc/next-dev-fresh-check.sh`; this asserts the
        /// shape at unit-test speed so a reintroduced poll is caught immediately.
        #[test]
        fn the_embedded_orchestrator_takes_dev_freshness_from_the_hot_channel_not_an_mtime_poll() {
            assert!(
                NEXT_SERVER_MJS.contains("/__diffpack_dev/hot"),
                "the orchestrator must expose the dev hot-update endpoint",
            );
            assert!(
                NEXT_SERVER_MJS.contains("serverInvalidate"),
                "a hot update must drive the live runtime's serverInvalidate",
            );
            assert!(
                !NEXT_SERVER_MJS.contains("statSync(ssrEntry).mtimeMs"),
                "the orchestrator must not key its SSR module cache on the entry's mtime",
            );
            assert!(
                !NEXT_SERVER_MJS.contains("ssrEntry).href + \"?v=\""),
                "re-importing the SSR entry under a query cannot bust its chunks' ESM cache",
            );
        }

        /// The deferred full re-emit is tracked as DEBT, not run inside the edit. The
        /// bookkeeping has to survive a batch that touches several graphs and has to
        /// report exactly what it owes, because that log line is how a developer sees
        /// which half of the update is still pending.
        #[test]
        fn owed_emits_accumulate_across_graphs_and_report_what_is_pending() {
            let mut owed = OwedEmits::default();
            assert!(!owed.any(), "nothing is owed before an edit");
            assert_eq!(owed.label(), "");
            owed.client = true;
            owed.ssr = true;
            assert!(owed.any());
            assert_eq!(
                owed.label(),
                "client + ssr",
                "an island edit owes both browser-facing graphs"
            );
            owed.react_server = true;
            assert_eq!(owed.label(), "client + ssr + react-server");
        }
    }
}
