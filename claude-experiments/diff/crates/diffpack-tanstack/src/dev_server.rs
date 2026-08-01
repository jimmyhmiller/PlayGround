//! `diffpack dev`: a long-lived, live-rebuild development server with
//! state-preserving Hot Module Replacement (React Fast Refresh + in-process
//! server hot reload).
//!
//! This is where Diffpack's incremental thesis becomes observable. The `build-app`
//! CLI is a cold process per invocation, so its already-incremental emit path (the
//! per-chunk render cache, incremental `emit_public`/`emit_server`) never gets
//! exercised across edits. The dev server keeps a `Bundler` (plus its reachability
//! session) alive PER ENVIRONMENT and re-emits on file change, so a leaf edit
//! re-transforms exactly one module and re-renders exactly one chunk from a
//! long-lived process — the payoff being a sub-budget hot update the browser
//! applies without losing state.
//!
//! Architecture (all native Rust; Node runs only the app's own SSR runtime, never
//! the build):
//!
//! 1. Build the client environment (`emit_public` + persist
//!    `client-manifest.json`) then the server environment (register the TanStack
//!    manifest + server-fn resolver virtual modules, `emit_server`) exactly as
//!    `build-app` does, but keep both bundlers alive and emit with
//!    `EmitOptions { hmr: true, .. }` (the HMR runtime, version-aware dynamic
//!    import, and Fast Refresh instrumentation from the Web HMR layer). The mandatory
//!    client-before-server order is preserved (the server manifest needs the
//!    finished client chunk URLs).
//! 2. Boot the emitted `server/index.mjs` as a child Node process on an internal
//!    loopback port, with a sibling in-process HMR control port (the app's own SSR
//!    runtime, never restarted on a normal edit).
//! 3. Put a diffpack-native reverse proxy in front on the public dev port. It
//!    serves the Fast Refresh runtime, upgrades the WebSocket HMR channel
//!    (`/__diffpack_hmr/ws`), and injects the Fast Refresh preamble + HMR client
//!    into served HTML. Node only runs the app.
//! 4. Watch the source tree with `notify`, coalescing duplicate/atomic-save
//!    events. On a module edit: incrementally rebuild the client bundler ->
//!    incremental `emit_public` -> re-persist the client manifest -> incrementally
//!    rebuild the server bundler -> `emit_server`, then (A) hot-reload the server
//!    IN-PROCESS by POSTing the changed module ids/chunks to its control endpoint,
//!    and (B/C) push a targeted update over the WebSocket channel so the browser
//!    re-imports exactly the changed chunk and applies the accept / React Fast
//!    Refresh protocol — preserving component state, no page reload, no Node
//!    restart. A route-file add/rename/remove re-derives the route tree and falls
//!    back to a Node restart + full reload (module ids shift); a server change that
//!    cannot be hot-applied falls back to a full page reload so the browser stays
//!    correct.
//!
//! SCOPE. Handled: content edits to existing modules (state-preserving HMR);
//! adding a new file and importing it, plus route-file add/rename/remove and any
//! edit that grows/shrinks the graph (full rebuild or reload — the module ids
//! shift, so these cannot be hot-patched); and BOTH app shapes — a TanStack Start
//! app (client + Node SSR, in-process server HMR) and a plain Vite HTML-entry SPA
//! (single client environment, static serving, no Node child; see the Web SPA
//! module). Deferred, with clear hard errors rather than silent partial handling:
//! CSS hot-swap without reload (a `.css` edit reloads today), config-change
//! handling, and error overlays. An edit class this slice does not handle is a hard
//! error naming what is unsupported, never a silent/partial rebuild.

use std::collections::BTreeSet;
use std::net::TcpListener;
use std::path::Path;
use std::process::Child;
use std::sync::Arc;
use std::sync::mpsc::Receiver;

pub use diffpack_web::dev_build::DevOptions;
use diffpack_web::dev_build::{EnvBuild, EnvCounters};
use diffpack_web::dev_control::{json_string, post_json, push_client as hmr_push_client};
pub use diffpack_web::dev_proxy;
pub use diffpack_web::preview::preview;
use diffpack_web::runtime::{free_port, restart_node, spawn_node, wait_for_node};
use diffpack_web::watch::{
    coalesce_batch, display_relative, is_module_path, source_dir as src_dir, start as start_watcher,
};
use diffpack_web::websocket::HmrHub;
use std::time::Instant;

use crate::config::{AppConfig, derive_config, set_development_mode};
use crate::manifest::{self, ClientRouteManifest};
use crate::server_fn;
use diffpack_default_loader::driver::{EmitOptions, EmitSummary, partition_diagnostics};

/// Run the dev server. Detects the app kind and dispatches: a TanStack Start app
/// (client + Node SSR runtime, in-process server HMR) or a plain Vite HTML-entry
/// SPA (single client environment, static serving, no Node). Blocks, serving until
/// the filesystem watcher stops or an unsupported edit is encountered (a hard
/// error).
pub fn run_tanstack(options: DevOptions) -> Result<(), String> {
    let project_root = options.project_root.canonicalize().map_err(|error| {
        format!(
            "cannot open project root {}: {error}",
            options.project_root.display()
        )
    })?;

    let output_root = project_root.join(".diffpack-output");
    let emit_options = EmitOptions {
        // Dev builds are never minified: HMR re-imports a chunk and reads
        // `import.meta.url`, and Fast Refresh instrumentation is appended as
        // readable JS. (Production `build-app` keeps its default-on minify.)
        minify: false,
        source_map: options.source_map,
        hmr: true,
        ..EmitOptions::default()
    };

    // 0. Natively generate `src/routeTree.gen.ts` from `src/routes/` so dev — like
    // build-app — consumes a diffpack-generated route tree, not one produced by
    // TanStack Router's Vite plugin.
    if let Some(route_count) = crate::route_tree::generate_for_project(&project_root)? {
        println!("[dev] generated route tree ({route_count} routes)");
    }

    // 1. Initial full build: client then server (order is load-bearing — the
    // server manifest reads the client's finished chunk map).
    println!("[dev] building client...");
    let mut client = build_client(&project_root, &output_root, emit_options)?;
    println!("[dev] building server...");
    let mut server = build_server(&project_root, &output_root, emit_options)?;

    // 2. Boot the emitted Node SSR runtime on an internal loopback port, with a
    // sibling control port so a server edit hot-reloads the runtime in-process
    // (Increment A) instead of restarting the Node child.
    let node_port = free_port()?;
    let control_port = free_port()?;
    let index_mjs = output_root.join("server/index.mjs");
    let mut node = spawn_node(&index_mjs, node_port, control_port)?;
    wait_for_node(node_port)?;
    println!(
        "[dev] node SSR runtime listening on 127.0.0.1:{node_port} (hmr control :{control_port})"
    );

    // The React Fast Refresh runtime (bundled by @vitejs/plugin-react); served to
    // the browser. Loaded once at startup so a missing dep is a clear hard error
    // now, not a broken update later.
    let refresh_runtime = Arc::new(diffpack_web::hmr::find_refresh_runtime(&project_root)?);

    // 3. Reverse proxy on the public dev port. It serves the HMR client assets,
    // upgrades the WebSocket HMR channel, and injects the Fast Refresh preamble.
    let hub = HmrHub::default();
    let proxy_listener = TcpListener::bind(("127.0.0.1", options.port))
        .map_err(|error| format!("cannot bind dev port {}: {error}", options.port))?;
    {
        let hub = hub.clone();
        let refresh_runtime = Arc::clone(&refresh_runtime);
        std::thread::Builder::new()
            .name("diffpack-dev-proxy".into())
            .spawn(move || {
                diffpack_web::node_proxy::serve(
                    proxy_listener,
                    node_port,
                    hub,
                    refresh_runtime,
                    None,
                    None,
                )
            })
            .map_err(|error| format!("cannot start proxy thread: {error}"))?;
    }
    println!(
        "[dev] diffpack dev server on http://127.0.0.1:{} (proxying node :{node_port})",
        options.port
    );

    // 4. Watch the source tree and drive the incremental rebuild loop.
    let watch_root = project_root.join(src_dir(&project_root));
    let receiver = start_watcher(&watch_root)?;
    println!("[dev] watching {}", watch_root.display());

    let result = watch_loop(
        &receiver,
        &project_root,
        &output_root,
        &index_mjs,
        node_port,
        control_port,
        &mut node,
        &mut client,
        &mut server,
        &hub,
    );
    // Always reap the child before returning.
    let _ = node.kill();
    let _ = node.wait();
    result
}

/// The blocking rebuild loop. Each coalesced batch of filesystem events is
/// classified; a supported edit incrementally rebuilds both environments,
/// restarts the Node child, and broadcasts a reload. An unsupported edit class is
/// a hard error naming what is missing.
#[allow(clippy::too_many_arguments)]
fn watch_loop(
    receiver: &Receiver<notify::Result<notify::Event>>,
    project_root: &Path,
    output_root: &Path,
    index_mjs: &Path,
    node_port: u16,
    control_port: u16,
    node: &mut Child,
    client_env: &mut EnvBuild,
    server_env: &mut EnvBuild,
    hub: &HmrHub,
) -> Result<(), String> {
    // Whether a build-error overlay is currently shown in the browser, so the next
    // good rebuild clears it (build-ok).
    let mut build_error_showing = false;
    loop {
        // Block for the first event, then coalesce a short burst (atomic saves
        // fire create+modify+rename in quick succession).
        let first = match receiver.recv() {
            Ok(event) => event,
            Err(_) => return Ok(()), // watcher dropped: clean shutdown.
        };
        let paths = coalesce_batch(receiver, first);

        let changed = paths
            .into_iter()
            .filter(|path| is_module_path(path))
            // The generated route tree is diffpack-owned now: it is regenerated
            // from `src/routes`, so an event on it is transient self-output, never
            // a user edit to react to.
            .filter(|path| {
                path.file_name().and_then(|name| name.to_str())
                    != Some(crate::route_tree::ROUTE_TREE_FILE)
            })
            .collect::<BTreeSet<_>>();
        if changed.is_empty() {
            continue;
        }

        // Two edit classes shift module ids across the whole graph and so cannot be
        // hot-patched — both are handled by a full rebuild + reload rather than a
        // crash:
        //   * a route-file add/rename/remove, which mutates the route tree (native
        //     regeneration first), and
        //   * a NEW non-route file (exists, unknown to both graphs) — adding a
        //     module is a normal dev action; the subsequent import of it is picked
        //     up by the re-discovery here (or, if imported later, by the
        //     graph-changed reload in the incremental path below).
        let route_mutation = changed
            .iter()
            .any(|path| is_route_tree_mutation(path, project_root, client_env, server_env));
        let new_file = changed.iter().any(|path| {
            path.exists()
                && !client_env.bundler.is_known_module(path)
                && !server_env.bundler.is_known_module(path)
        });
        if route_mutation || new_file {
            let started = Instant::now();
            if route_mutation {
                crate::route_tree::generate_for_project(project_root)?;
            }
            *client_env = build_client(project_root, output_root, client_env.options)?;
            *server_env = build_server(project_root, output_root, server_env.options)?;
            // Re-derives both graphs from scratch; the module ids shift, so restart
            // the Node child (rare) and full-reload the browser.
            restart_node(node, index_mjs, node_port, control_port)?;
            hub.broadcast_reload();
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let reason = if route_mutation {
                "route tree changed"
            } else {
                "new file(s)"
            };
            println!(
                "[dev] {reason} ({} file(s)) in {elapsed_ms:.1}ms | full rebuild + reload pushed",
                changed.len(),
            );
            continue;
        }

        let started = Instant::now();
        let mut client = EnvCounters::default();
        let mut server_c = EnvCounters::default();
        let mut touched = false;
        // Whether an edit grew/shrank either graph (added/removed an import). Such an
        // edit re-partitions the chunks and cannot be hot-patched — reload instead.
        let mut graph_changed = false;
        // Accumulate the changed module ids per environment so, after re-emit, one
        // targeted HMR update covers the whole coalesced batch.
        let mut client_changed_ids: BTreeSet<String> = BTreeSet::new();
        let mut server_changed_ids: BTreeSet<String> = BTreeSet::new();

        // Catch edit-time build errors (e.g. a syntax error in the edited module) and
        // surface them in the browser overlay instead of killing the dev server; the
        // loop keeps serving and clears the overlay on the next good rebuild. The
        // initial and full/route rebuilds stay hard errors (fail fast).
        let batch = (|| -> Result<(), String> {
            for path in &changed {
                classify_edit(path, project_root, client_env, server_env)?;
            }
            for path in &changed {
                // Rebuild whichever environment(s) actually own the module. A route
                // module is in both graphs; a client-only or server-only module is in
                // just one. (An unknown path here is a delete of an already-unreachable
                // file — it belongs to neither graph, so both branches skip it.)
                if client_env.bundler.is_known_module(path) {
                    let rebuilt = client_env.rebuild(path)?;
                    let summary = emit_client(client_env, project_root, output_root)?;
                    client_changed_ids.extend(rebuilt.changed_ids.iter().cloned());
                    graph_changed |= rebuilt.graph_changed;
                    client.add(&rebuilt, summary.rendered_chunks);
                    touched = true;
                }
                if server_env.bundler.is_known_module(path) {
                    let rebuilt = server_env.rebuild(path)?;
                    let summary = server_env.bundler.emit_server(
                        &reachable_ids(server_env),
                        output_root,
                        server_env.options,
                    )?;
                    server_changed_ids.extend(rebuilt.changed_ids.iter().cloned());
                    graph_changed |= rebuilt.graph_changed;
                    server_c.add(&rebuilt, summary.rendered_chunks);
                    touched = true;
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
        if build_error_showing {
            hub.broadcast_build_ok();
            build_error_showing = false;
        }

        if !touched {
            continue;
        }

        // A graph-structure change (import added/removed) re-partitions chunks:
        // re-emit already ran above, so a full reload picks up the new partition
        // correctly where a hot update's ESM re-import would fail to bind.
        if graph_changed {
            hub.broadcast_reload();
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            println!(
                "[dev] rebuilt {} file(s) in {elapsed_ms:.1}ms | graph changed (import added/removed) -> full reload",
                changed.len(),
            );
            continue;
        }

        // INCREMENT A: hot-reload the server in-process (no Node restart) by
        // POSTing the changed server module ids + chunks to the live runtime's
        // control endpoint, which invalidates their cache and bumps the chunk
        // versions so the next SSR request re-evaluates them.
        let server_reload = hmr_reload_server(server_env, &server_changed_ids, control_port);

        // INCREMENTS B/C: push a targeted client HMR update over WebSocket. The
        // browser re-imports the changed chunk (register-only) and applies the
        // accept/Fast-Refresh protocol, preserving state. If no browser is
        // connected there is nothing to push.
        let client_update = hmr_push_client(client_env, &client_changed_ids, hub, None);

        // Fall back to a full page reload only when the server change could not be
        // hot-applied (e.g. a statically-bundled server module), so the browser
        // still reflects it — correct, not a crash.
        let mut server_note = server_reload.summary;
        if server_reload.needs_reload {
            hub.broadcast_reload();
            server_note.push_str(" (fell back to full reload)");
        }

        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        // Per-edit incremental instrumentation, exercised live from a long-lived
        // process. `changed` is the sharp incremental-transform signal (exactly
        // one module's content changed for a leaf/route-component edit) and
        // `rendered_chunks` is the incremental-emit signal (exactly one chunk
        // re-rendered). Printed as a stable, parseable line the browser oracle
        // asserts on.
        println!(
            "[dev] rebuilt {} file(s) in {elapsed_ms:.1}ms | client transformed={} changed={} rendered_chunks={} | server transformed={} changed={} rendered_chunks={} | {client_update} | server: {server_note}",
            changed.len(),
            client.transformed,
            client.changed,
            client.rendered_chunks,
            server_c.transformed,
            server_c.changed,
            server_c.rendered_chunks,
        );
    }
}

/// Outcome of a server hot-reload attempt.
struct ServerReload {
    summary: String,
    needs_reload: bool,
}

/// INCREMENT A: hot-reload the server in-process. POSTs the changed server module
/// ids and their chunk files to the emitted server's control endpoint, which
/// invalidates the runtime cache and bumps chunk versions so the next SSR request
/// re-evaluates the changed subtree. The Node process (PID) is never restarted.
fn hmr_reload_server(
    server_env: &EnvBuild,
    changed_ids: &BTreeSet<String>,
    control_port: u16,
) -> ServerReload {
    if changed_ids.is_empty() {
        return ServerReload {
            summary: "no server change".to_string(),
            needs_reload: false,
        };
    }
    let located =
        match server_env
            .bundler
            .hmr_locate(&reachable_ids(server_env), changed_ids, "server.mjs")
        {
            Ok(located) => located,
            Err(error) => {
                return ServerReload {
                    summary: format!("locate failed: {error}"),
                    needs_reload: true,
                };
            }
        };
    if located.is_empty() {
        return ServerReload {
            summary: "no located server modules".to_string(),
            needs_reload: true,
        };
    }
    let ids = located.iter().map(|l| l.runtime_id).collect::<Vec<_>>();
    // Chunk version keys match the runtime's `__chunks` map, which stores relative
    // `./server.chunk-N.mjs` names. The entry (`server.mjs`) has no dynamic-import
    // version to bump; only real split chunks are versioned.
    let chunks = located
        .iter()
        .filter(|l| l.chunk_file != "server.mjs")
        .map(|l| format!("./{}", l.chunk_file))
        .collect::<BTreeSet<_>>();
    let entry_touched = located.iter().any(|l| l.chunk_file == "server.mjs");
    let payload = format!(
        "{{\"ids\":[{}],\"chunks\":[{}]}}",
        ids.iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(","),
        chunks
            .iter()
            .map(|chunk| json_string(chunk))
            .collect::<Vec<_>>()
            .join(","),
    );
    match post_json(control_port, "/__diffpack_hmr", &payload) {
        Ok(_) => ServerReload {
            summary: format!(
                "hot-reloaded {} module(s) in-process{}",
                ids.len(),
                if entry_touched {
                    " (entry module changed; a full reload will pick it up)"
                } else {
                    ""
                }
            ),
            // A statically-bundled entry module cannot be re-imported per request,
            // so pair it with a browser reload to stay correct.
            needs_reload: entry_touched,
        },
        Err(error) => ServerReload {
            summary: format!("control POST failed: {error}"),
            needs_reload: true,
        },
    }
}

/// INCREMENTS B/C: push a targeted client HMR update over the WebSocket channel.
/// Returns a short log fragment describing what was pushed.
fn classify_edit(
    path: &Path,
    project_root: &Path,
    _client: &EnvBuild,
    _server: &EnvBuild,
) -> Result<(), String> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");

    let is_config = matches!(name, "vite.config.ts" | "vite.config.js" | "package.json")
        || name.starts_with("tsconfig")
        || name.starts_with("diffpack.config");
    if is_config {
        return Err(format!(
            "unsupported dev edit: config file {} changed. Config-change handling (re-deriving aliases/conditions/virtual modules) is not implemented by the dev server; restart `diffpack dev` to pick it up.",
            display_relative(path, project_root)
        ));
    }

    Ok(())
}

/// Whether `path` is a route-tree-mutating change: a route-extension file under
/// `<src>/routes` that is NOT a known module in either graph (a new file, or a
/// deleted/renamed one). An edit to an existing route module is a normal
/// incremental edit, not a mutation.
fn is_route_tree_mutation(
    path: &Path,
    project_root: &Path,
    client: &EnvBuild,
    server: &EnvBuild,
) -> bool {
    let routes_dir = project_root.join(src_dir(project_root)).join("routes");
    if !path.starts_with(&routes_dir) {
        return false;
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if !["tsx", "ts", "jsx", "js"].contains(&extension) {
        return false;
    }
    !client.bundler.is_known_module(path) && !server.bundler.is_known_module(path)
}

/// Build the client environment fresh (mirrors `build-app <root> client`) and
/// leave the bundler alive.
fn build_client(
    project_root: &Path,
    output_root: &Path,
    options: EmitOptions,
) -> Result<EnvBuild, String> {
    let mut config = derive_config(project_root, "client")?;
    // DEV-ONLY: instrument the client graph for HMR / React Fast Refresh, and
    // select the dependencies' development builds.
    set_development_mode(&mut config);
    // Dev serves source maps by default (`--no-sourcemap` opts out), so the real
    // per-module map is produced exactly when the emit will write one.
    config.build.source_maps = options.source_map;
    let entry = config
        .entry
        .clone()
        .ok_or_else(|| "no client entry found for the app".to_string())?;
    let (bundler, update) = crate::compiler::discover(&entry, &config.build)?;
    for warning in partition_diagnostics(&update.diagnostics, "dev client build")? {
        println!("[dev] warning: {warning}");
    }
    let session = bundler.direct_reachability();
    let reachable = session.reachable_modules();
    let build = EnvBuild {
        bundler,
        session,
        reachable,
        options,
    };
    emit_client(&build, project_root, output_root)?;
    Ok(build)
}

/// Emit the client `public/` layout, copy static files, and persist the route ->
/// client-chunk manifest the server build consumes. Shared by the initial build
/// and every incremental rebuild.
fn emit_client(
    client: &EnvBuild,
    project_root: &Path,
    output_root: &Path,
) -> Result<EmitSummary, String> {
    let reachable = reachable_ids(client);
    let summary = client
        .bundler
        .emit_public(&reachable, output_root, client.options)?;
    diffpack_web::config::copy_static_public(project_root, &summary.output_dir)?;
    let client_manifest = crate::manifest::from_bundle_graph(
        &client
            .bundler
            .integration_manifest_graph(&reachable, "client.js")?,
        "/",
    )?;
    client_manifest.write(&output_root.join(manifest::CLIENT_MANIFEST_FILE))?;
    Ok(summary)
}

/// Build the server environment fresh (mirrors `build-app <root> ssr`) and leave
/// the bundler alive. Must run AFTER [`build_client`] so `client-manifest.json`
/// exists.
fn build_server(
    project_root: &Path,
    output_root: &Path,
    options: EmitOptions,
) -> Result<EnvBuild, String> {
    let mut config = derive_config(project_root, "ssr")?;
    // DEV-ONLY: emit the version-aware dynamic import + in-process control endpoint
    // so a server edit hot-reloads without restarting Node.
    set_development_mode(&mut config);
    register_server_virtual_modules(&mut config, project_root, output_root)?;
    config.build.source_maps = options.source_map;
    let entry = config
        .entry
        .clone()
        .ok_or_else(|| "no ssr entry found for the app".to_string())?;
    let (bundler, update) = crate::compiler::discover(&entry, &config.build)?;
    for warning in partition_diagnostics(&update.diagnostics, "dev server build")? {
        println!("[dev] warning: {warning}");
    }
    let session = bundler.direct_reachability();
    let reachable = session.reachable_modules();
    let build = EnvBuild {
        bundler,
        session,
        reachable,
        options,
    };
    let reachable = reachable_ids(&build);
    build
        .bundler
        .emit_server(&reachable, output_root, options)?;
    Ok(build)
}

/// Register the two build-output-dependent virtual modules the server graph
/// needs: the TanStack start manifest (from the client's persisted chunk map) and
/// the native server-fn resolver (from a scan of the project's `createServerFn`
/// handlers). Mirrors `build-app`'s server path.
fn register_server_virtual_modules(
    config: &mut AppConfig,
    project_root: &Path,
    output_root: &Path,
) -> Result<(), String> {
    // TanStack Start's dev-time head-script manifest, derived from the CLIENT build's
    // output — which is what makes the client build have to run first. Only a
    // `@tanstack/start-server-core` graph can import it, so the Next server graphs
    // register the rest of these modules without it (see
    // `register_next_server_virtual_modules`) and are free to build before the client.
    let client_manifest_path = output_root.join(manifest::CLIENT_MANIFEST_FILE);
    let client_manifest = ClientRouteManifest::read(&client_manifest_path)?;
    config.build.virtual_modules.push((
        manifest::START_MANIFEST_SPECIFIER.to_string(),
        client_manifest.to_start_manifest_source(),
    ));
    let server_fns = server_fn::scan_project_server_fns(project_root)?;
    config.build.virtual_modules.push((
        server_fn::RESOLVER_SPECIFIER.to_string(),
        server_fn::generate_resolver_module(&server_fns),
    ));
    Ok(())
}

fn reachable_ids(build: &EnvBuild) -> BTreeSet<String> {
    build.reachable_ids()
}
