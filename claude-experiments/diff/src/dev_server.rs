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
//!    import, and Fast Refresh instrumentation from [`crate::hmr`]). The mandatory
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
//! (single client environment, static serving, no Node child; see the [`spa`]
//! module). Deferred, with clear hard errors rather than silent partial handling:
//! CSS hot-swap without reload (a `.css` edit reloads today), config-change
//! handling, and error overlays. An edit class this slice does not handle is a hard
//! error naming what is unsupported, never a silent/partial rebuild.

use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use notify::{RecursiveMode, Watcher};

use crate::bundler::{Bundler, DirectReachability, EmitOptions, EmitSummary};
use crate::config::{self, AppConfig};
use crate::manifest::{self, ClientRouteManifest};
use crate::server_fn;

/// Options for `diffpack dev`.
pub struct DevOptions {
    pub project_root: PathBuf,
    /// Public port the browser connects to (the diffpack reverse proxy).
    pub port: u16,
    /// Whether emitted chunks are minified (matches `build-app`'s default-on).
    pub minify: bool,
    /// Whether emitted chunks carry composed source maps.
    pub source_map: bool,
}

/// One long-lived environment build (client or server): the bundler, its
/// persistent reachability session, and the current reachable set. Kept alive
/// across edits so a rebuild is incremental.
struct EnvBuild {
    bundler: Bundler,
    session: DirectReachability,
    reachable: BTreeSet<String>,
    options: EmitOptions,
}

impl EnvBuild {
    /// Incrementally rebuild after `path` changed, apply the reachability delta,
    /// and return `(transformed, changed)`: how many modules were re-evaluated and
    /// how many actually changed content. Emit is left to the caller since the
    /// client and server have different emit shapes.
    fn rebuild(&mut self, path: &Path) -> Result<Rebuilt, String> {
        let update = self.bundler.rebuild_path(path)?;
        let transformed = update.transformed_modules;
        let changed = update.delta.changed.len();
        let result = self.session.apply(&update.delta);
        // A change to the REACHABLE SET (a module added or removed — e.g. an edit
        // that introduces an `import` of a newly-created file, or drops the last
        // import of one) re-partitions the chunks and shifts runtime ids. That
        // cannot be hot-patched: the browser's ESM re-import would fail to bind the
        // new/removed exports. The caller reloads instead when this is set.
        let graph_changed = !result.added.is_empty() || !result.removed.is_empty();
        for module in result.removed {
            self.reachable.remove(&module);
        }
        self.reachable.extend(result.added);
        // An edit that introduces an unresolved import would hot-patch a chunk that
        // throws on load. Fail the rebuild instead; the caller catches this and
        // shows the browser overlay, keeping the last good bundle served.
        for warning in crate::bundler::partition_diagnostics(&update.diagnostics, "rebuild")? {
            eprintln!("[dev] warning: {warning}");
        }
        Ok(Rebuilt {
            transformed,
            changed,
            changed_ids: update.delta.changed.clone(),
            graph_changed,
        })
    }
}

/// Aggregated per-edit counters for one environment across all files touched in
/// a single coalesced batch.
#[derive(Default)]
struct EnvCounters {
    transformed: usize,
    changed: usize,
    rendered_chunks: usize,
}

impl EnvCounters {
    fn add(&mut self, rebuilt: &Rebuilt, rendered_chunks: usize) {
        self.transformed += rebuilt.transformed;
        self.changed += rebuilt.changed;
        self.rendered_chunks += rendered_chunks;
    }
}

/// Per-edit rebuild counts for one environment.
#[derive(Default)]
struct Rebuilt {
    /// Modules re-evaluated (the edited module plus any derived virtual siblings
    /// and newly-discovered dependencies).
    transformed: usize,
    /// Of those, how many actually changed content (the sharp incremental
    /// signal: a route-component edit changes exactly the one split chunk's
    /// module, not the reference module that no longer holds the body).
    changed: usize,
    /// The canonical ids of the modules whose content changed, so the dev server
    /// can push a targeted HMR update for exactly them.
    changed_ids: BTreeSet<String>,
    /// Whether the reachable set changed (a module added or removed). When set, the
    /// chunk partition and runtime ids shift, so the edit must be applied by a full
    /// reload rather than a hot update.
    graph_changed: bool,
}

/// The HMR broadcast fan-out over WebSocket. Each connected browser's upgraded
/// socket is held here; an update or reload writes one text frame to every one,
/// pruning any that error (a closed tab).
#[derive(Clone, Default)]
struct HmrHub {
    clients: Arc<Mutex<Vec<TcpStream>>>,
}

impl HmrHub {
    fn register(&self, stream: TcpStream) {
        self.clients.lock().unwrap().push(stream);
    }

    /// Send one JSON message to every connected browser as a WebSocket text frame.
    fn send(&self, json: &str) {
        let frame = ws_text_frame(json.as_bytes());
        let mut clients = self.clients.lock().unwrap();
        clients.retain_mut(|stream| {
            stream
                .write_all(&frame)
                .and_then(|()| stream.flush())
                .is_ok()
        });
    }

    /// Push a full-page reload to every connected browser.
    fn broadcast_reload(&self) {
        self.send(r#"{"type":"reload"}"#);
    }

    /// Push an in-place RSC refresh (a server-component edit): each browser refetches
    /// the current route's flight (`?__rsc=1`) and diff-renders it through the client
    /// Router — no full document reload, client-island state preserved.
    fn broadcast_rsc_refresh(&self) {
        self.send(r#"{"type":"rsc-refresh"}"#);
    }

    /// Surface an edit-time Rust build error in the browser overlay instead of
    /// crashing dev. The message is JSON-escaped for safe embedding.
    fn broadcast_build_error(&self, message: &str) {
        self.send(&format!(
            "{{\"type\":\"build-error\",\"message\":{}}}",
            json_string(message)
        ));
    }

    /// Signal the browser to clear the build-error overlay after a good rebuild.
    fn broadcast_build_ok(&self) {
        self.send(r#"{"type":"build-ok"}"#);
    }

    fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }
}

/// Frame a server->client WebSocket text message (RFC 6455): FIN + text opcode,
/// unmasked, with the minimal length encoding.
fn ws_text_frame(payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x81); // FIN=1, opcode=0x1 (text)
    let len = payload.len();
    if len < 126 {
        frame.push(len as u8);
    } else if len < 65536 {
        frame.push(126);
        frame.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(len as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    frame
}

/// The RFC 6455 `Sec-WebSocket-Accept` value for a client key.
fn ws_accept(key: &str) -> String {
    let mut input = key.to_string();
    input.push_str("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    base64_encode(&sha1(input.as_bytes()))
}

/// Minimal SHA-1 (RFC 3174), enough for the WebSocket handshake.
fn sha1(message: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let ml = (message.len() as u64).wrapping_mul(8);
    let mut data = message.to_vec();
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&ml.to_be_bytes());
    for block in data.chunks_exact(64) {
        let mut w = [0u32; 80];
        for (index, word) in block.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..80 {
            w[index] = (w[index - 3] ^ w[index - 8] ^ w[index - 14] ^ w[index - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (index, &word) in w.iter().enumerate() {
            let (f, k) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (index, word) in h.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Standard base64 encoding (no line wrapping).
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Run the dev server. Detects the app kind and dispatches: a TanStack Start app
/// (client + Node SSR runtime, in-process server HMR) or a plain Vite HTML-entry
/// SPA (single client environment, static serving, no Node). Blocks, serving until
/// the filesystem watcher stops or an unsupported edit is encountered (a hard
/// error).
pub fn run(options: DevOptions) -> Result<(), String> {
    let project_root = options
        .project_root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", options.project_root.display()))?;

    // A Next.js app-router app has no TanStack/src entry; its "entry" is the
    // app-router file convention. It needs a THIRD dev topology (three RSC graphs +
    // the next orchestrator), so dispatch it first, before the TanStack/SPA split.
    if crate::next_adapter::is_app_router(&project_root) {
        return next::run_next(&options, &project_root);
    }

    // App-kind detection. A TanStack Start app renders through its own Node SSR
    // runtime (`@tanstack/react-start`) and has no static `index.html`; a plain
    // Vite SPA is rooted at an `index.html` with a `<script type="module">` entry
    // and no SSR framework. The two need different dev topologies, so pick here.
    let has_start = project_root
        .join("node_modules/@tanstack/react-start")
        .exists();
    let index_html = project_root.join("index.html");
    if !has_start && index_html.is_file() {
        return spa::run_spa(&options, &project_root, &index_html);
    }

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
    let refresh_runtime = Arc::new(crate::hmr::find_refresh_runtime(&project_root)?);

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
            .spawn(move || serve_proxy(proxy_listener, node_port, hub, refresh_runtime, None))
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
            let reason = if route_mutation { "route tree changed" } else { "new file(s)" };
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
    let located = match server_env.bundler.hmr_locate(
        &reachable_ids(server_env),
        changed_ids,
        "server.mjs",
    ) {
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
        ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(","),
        chunks
            .iter()
            .map(|chunk| json_string(chunk))
            .collect::<Vec<_>>()
            .join(","),
    );
    match post_control(control_port, &payload) {
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
fn hmr_push_client(
    client_env: &EnvBuild,
    changed_ids: &BTreeSet<String>,
    hub: &HmrHub,
    // When Some(output_root), render a MICRO-CHUNK holding only the changed modules to
    // `<output_root>/public/client.hmr.js` and point the browser at that (~1 KB)
    // instead of re-importing the whole entry chunk (~1 MB of app + React, whose
    // re-parse dominates the browser-side hot update). None keeps the legacy behaviour
    // of re-importing the located full chunk(s).
    micro_chunk: Option<&Path>,
) -> String {
    if changed_ids.is_empty() {
        return "client: no change".to_string();
    }
    let reachable = reachable_ids(client_env);
    let located = match client_env
        .bundler
        .hmr_locate(&reachable, changed_ids, "client.js")
    {
        Ok(located) => located,
        Err(error) => {
            hub.broadcast_reload();
            return format!("client: locate failed ({error}); reloaded");
        }
    };
    if located.is_empty() {
        return "client: no located modules".to_string();
    }
    let ids = located.iter().map(|l| l.runtime_id).collect::<Vec<_>>();

    // The chunk(s) the browser re-imports to pick up the new factories.
    let chunks: BTreeSet<String> = if let Some(output_root) = micro_chunk {
        // The micro-chunk is written WITH its source map (see `write_hmr_chunk`) —
        // the edited module's stack traces are the whole point of a hot update.
        match client_env.bundler.write_hmr_chunk(
            &reachable,
            changed_ids,
            "client.js",
            client_env.options,
            // The client graph emits browser ESM; the micro-chunk must match it or a
            // hot-updated module's `__dirname`/`__filename` would not be the browser
            // stubs the rest of the chunk uses.
            crate::bundler::ModuleFormat::BrowserEsm,
            &output_root.join("public/client.hmr.js"),
        ) {
            Ok(true) => std::iter::once("/client.hmr.js".to_string()).collect(),
            // No live changed module rendered — nothing to push (defensive; located
            // was non-empty, so this is unexpected but not a crash).
            Ok(false) => return "client: no live changed module for micro-chunk".to_string(),
            Err(error) => {
                hub.broadcast_reload();
                return format!("client: micro-chunk render/write failed ({error}); reloaded");
            }
        }
    } else {
        located.iter().map(|l| format!("/{}", l.chunk_file)).collect()
    };

    let message = format!(
        "{{\"type\":\"update\",\"ids\":[{}],\"chunks\":[{}]}}",
        ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(","),
        chunks
            .iter()
            .map(|chunk| json_string(chunk))
            .collect::<Vec<_>>()
            .join(","),
    );
    hub.send(&message);
    format!(
        "client: hmr update -> {} module(s) in {} chunk(s), {} browser(s)",
        ids.len(),
        chunks.len(),
        hub.client_count()
    )
}

/// JSON-encode a string as a JS/JSON string literal.
fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

/// Minimal HTTP POST to the emitted server's loopback control endpoint.
fn post_control(control_port: u16, json: &str) -> Result<(), String> {
    post_json(control_port, "/__diffpack_hmr", json)
}

/// Minimal loopback HTTP POST of a JSON body. Non-200 (and any transport failure) is
/// an error carrying the server's reason — a dev control channel that silently no-ops
/// is exactly how stale output ships.
fn post_json(port: u16, path: &str, json: &str) -> Result<(), String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .map_err(|error| format!("cannot reach 127.0.0.1:{port}{path}: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .ok();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
        json.len()
    );
    stream
        .write_all(request.as_bytes())
        .and_then(|()| stream.flush())
        .map_err(|error| format!("cannot send control request: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("cannot read control response: {error}"))?;
    let head = String::from_utf8_lossy(&response);
    if head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200") {
        Ok(())
    } else {
        // Carry the BODY, not just the status line: the endpoint answers a failure with
        // the reason (usually a stack from the module it could not apply), and dropping
        // it turns a diagnosable error into "something went wrong".
        let (status, body) = head
            .split_once("\r\n\r\n")
            .map_or((head.as_ref(), ""), |(head, body)| {
                (head.lines().next().unwrap_or("<no status>"), body)
            });
        Err(format!("control endpoint returned {status}: {body}"))
    }
}

/// Classify an edited path and hard-error on any class this slice does not yet
/// handle, naming exactly what is unsupported.
///
/// By the time this runs, the two structural classes are already handled earlier
/// in the loop (a route-tree mutation and a brand-new non-route file both take the
/// full-rebuild + reload path), and a delete of an already-unreachable file is
/// harmlessly skipped by the incremental loop. What remains genuinely unsupported
/// is a config-file change (re-deriving aliases/conditions/virtual modules), which
/// is a clear hard error rather than a silent stale build.
fn classify_edit(
    path: &Path,
    project_root: &Path,
    _client: &EnvBuild,
    _server: &EnvBuild,
) -> Result<(), String> {
    let name = path.file_name().and_then(|value| value.to_str()).unwrap_or("");

    let is_config = matches!(
        name,
        "vite.config.ts" | "vite.config.js" | "package.json"
    ) || name.starts_with("tsconfig")
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
    let extension = path.extension().and_then(|value| value.to_str()).unwrap_or("");
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
    let mut config = config::derive_config(project_root, "client")?;
    // DEV-ONLY: instrument the client graph for HMR / React Fast Refresh, and
    // select the dependencies' development builds.
    config::set_development_mode(&mut config);
    // Dev serves source maps by default (`--no-sourcemap` opts out), so the real
    // per-module map is produced exactly when the emit will write one.
    config.build.source_maps = options.source_map;
    let entry = config
        .entry
        .clone()
        .ok_or_else(|| "no client entry found for the app".to_string())?;
    let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config.build)?;
    for warning in crate::bundler::partition_diagnostics(&update.diagnostics, "dev client build")? {
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
    config::copy_static_public(project_root, &summary.output_dir)?;
    let client_manifest = client
        .bundler
        .client_route_manifest(&reachable, "client.js", "/")?;
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
    let mut config = config::derive_config(project_root, "ssr")?;
    // DEV-ONLY: emit the version-aware dynamic import + in-process control endpoint
    // so a server edit hot-reloads without restarting Node.
    config::set_development_mode(&mut config);
    register_server_virtual_modules(&mut config, project_root, output_root)?;
    config.build.source_maps = options.source_map;
    let entry = config
        .entry
        .clone()
        .ok_or_else(|| "no ssr entry found for the app".to_string())?;
    let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config.build)?;
    for warning in crate::bundler::partition_diagnostics(&update.diagnostics, "dev server build")? {
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
    build.bundler.emit_server(&reachable, output_root, options)?;
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

    // RSC server actions: the native action resolver + endpoint + client transport,
    // mirroring `build-app`'s server path. The resolver is keyed by the same
    // `"<moduleId>#<name>"` id the `"use server"` transform bakes into the client
    // stub and the server registration.
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

fn reachable_ids(build: &EnvBuild) -> BTreeSet<String> {
    build.reachable.clone()
}

/// Reserve an ephemeral loopback port for the Node child by binding and
/// immediately dropping a listener, returning its number.
fn free_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("cannot reserve a port for the node runtime: {error}"))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| format!("cannot read reserved port: {error}"))
}

/// Spawn the emitted `server/index.mjs` under Node on `port` (loopback only). The
/// `control_port` is passed through `DIFFPACK_HMR_CONTROL_PORT` so the emitted
/// server starts its in-process HMR control endpoint (dev builds only).
fn spawn_node(index_mjs: &Path, port: u16, control_port: u16) -> Result<Child, String> {
    Command::new("node")
        .arg(index_mjs)
        .env("PORT", port.to_string())
        .env("HOST", "127.0.0.1")
        .env("DIFFPACK_HMR_CONTROL_PORT", control_port.to_string())
        .spawn()
        .map_err(|error| format!("cannot start node SSR runtime ({}): {error}", index_mjs.display()))
}

/// Kill the current Node child, spawn a fresh one on the same ports, and wait for
/// it to accept connections. Used only for edit classes that cannot be hot-swapped
/// in-process (a route-tree mutation / full rebuild), never for a normal edit.
fn restart_node(node: &mut Child, index_mjs: &Path, port: u16, control_port: u16) -> Result<(), String> {
    let _ = node.kill();
    let _ = node.wait();
    *node = spawn_node(index_mjs, port, control_port)?;
    wait_for_node(port)
}

/// Poll a loopback port until Node is accepting connections (or time out).
fn wait_for_node(port: u16) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Err(format!("node SSR runtime did not listen on 127.0.0.1:{port} within 15s"))
}

/// Accept loop for the diffpack-native reverse proxy. Each connection is handled
/// on its own thread: it serves the HMR client assets, upgrades the WebSocket HMR
/// channel (held open in the hub), and forwards every other request to the Node
/// child with the Fast Refresh preamble injected into any HTML response.
fn serve_proxy(
    listener: TcpListener,
    node_port: u16,
    hub: HmrHub,
    refresh_runtime: Arc<String>,
    // When set, a GET whose path maps to an existing regular file under this dir is
    // served DIRECTLY off disk, bypassing the Node orchestrator hop. This keeps the
    // browser's HMR chunk fetch (`/client.js?__diffpack_hmr=1`) off the critical path
    // of a Node round-trip; route documents (no matching file) still forward to Node.
    static_dir: Option<Arc<PathBuf>>,
) {
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let hub = hub.clone();
        let refresh_runtime = Arc::clone(&refresh_runtime);
        let static_dir = static_dir.clone();
        let _ = std::thread::Builder::new()
            .name("diffpack-dev-conn".into())
            .spawn(move || {
                if let Err(error) = handle_connection(
                    stream,
                    node_port,
                    &hub,
                    &refresh_runtime,
                    static_dir.as_deref().map(|p| p.as_path()),
                ) {
                    // A dropped browser connection is normal; log at a low volume.
                    let _ = error;
                }
            });
    }
}

/// The served path for the React Fast Refresh runtime (imported by the preamble).
const REFRESH_RUNTIME_PATH: &str = "/__diffpack_hmr/refresh-runtime.js";
/// The WebSocket HMR channel path.
const WS_PATH: &str = "/__diffpack_hmr/ws";

fn handle_connection(
    mut stream: TcpStream,
    node_port: u16,
    hub: &HmrHub,
    refresh_runtime: &str,
    static_dir: Option<&Path>,
) -> Result<(), String> {
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|error| format!("cannot clone client socket: {error}"))?,
    );
    let (request_line, headers) = read_head(&mut reader)?;
    let (method, target) = parse_request_line(&request_line)?;
    let path = target.split('?').next().unwrap_or(&target);

    // The WebSocket HMR channel: complete the RFC 6455 handshake and hand the
    // upgraded socket to the hub, which pushes update/reload frames.
    if path == WS_PATH {
        if let Some((_, key)) = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("sec-websocket-key"))
        {
            let accept = ws_accept(key.trim());
            let response = format!(
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
            );
            stream
                .write_all(response.as_bytes())
                .and_then(|()| stream.flush())
                .map_err(|error| format!("cannot complete websocket handshake: {error}"))?;
            hub.send_to(&stream, r#"{"type":"connected"}"#);
            hub.register(stream);
            return Ok(());
        }
        return Ok(());
    }

    // The Fast Refresh runtime, served as an ES module the preamble imports.
    if path == REFRESH_RUNTIME_PATH {
        write_js(&mut stream, refresh_runtime)?;
        return Ok(());
    }

    // Direct static serving (dev fast path): a GET for an existing regular file under
    // the emitted `public/` dir is served straight off disk, skipping the Node hop.
    // This is what a browser HMR chunk fetch (`/client.js?__diffpack_hmr=1&t=…`) hits,
    // so the re-imported chunk lands without a round-trip through the orchestrator.
    // App-router routes (`/`, `/about`) have no matching file and fall through to Node.
    if let Some(file) = static_dir
        .filter(|_| method == "GET")
        .and_then(|dir| resolve_static_file(dir, path))
    {
        write_file(&mut stream, &file)?;
        return Ok(());
    }

    // Read the request body (for server-fn POSTs) so it forwards intact.
    let body = read_body(&mut reader, &headers)?;
    let upstream = forward_to_node(node_port, &method, &target, &headers, &body)?;
    let response = maybe_inject_hmr(upstream);
    stream
        .write_all(&response)
        .map_err(|error| format!("cannot write response to client: {error}"))?;
    stream.flush().ok();
    Ok(())
}

/// Write a JavaScript module response (dev; no caching).
fn write_js(stream: &mut TcpStream, body: &str) -> Result<(), String> {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/javascript; charset=utf-8\r\nCache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .and_then(|()| stream.write_all(body.as_bytes()))
        .and_then(|()| stream.flush())
        .map_err(|error| format!("cannot write js response: {error}"))
}

/// Resolve a URL path to an existing regular file under `dir`, or `None`. Guards
/// against path traversal and empty segments, and returns only regular files — so an
/// app-router route path (which has no matching file on disk) falls through to Node.
fn resolve_static_file(dir: &Path, url_path: &str) -> Option<PathBuf> {
    let rel = url_path.trim_start_matches('/');
    if rel.is_empty() || rel.split('/').any(|seg| seg == ".." || seg == "." || seg.is_empty()) {
        return None;
    }
    let candidate = dir.join(rel);
    candidate.is_file().then_some(candidate)
}

/// Serve a file off disk as a dev HTTP response (no caching, `Connection: close`).
fn write_file(stream: &mut TcpStream, file: &Path) -> Result<(), String> {
    let bytes =
        std::fs::read(file).map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    let content_type = match file.extension().and_then(|value| value.to_str()) {
        Some("js" | "mjs" | "cjs") => "application/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("json" | "map") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("avif") => "image/avif",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("wasm") => "application/wasm",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    };
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|()| stream.write_all(&bytes))
        .and_then(|()| stream.flush())
        .map_err(|error| format!("cannot write file response: {error}"))
}

impl HmrHub {
    /// Send one JSON message to a single socket (used right after the handshake).
    fn send_to(&self, mut stream: &TcpStream, json: &str) {
        let frame = ws_text_frame(json.as_bytes());
        let _ = stream.write_all(&frame).and_then(|()| stream.flush());
    }
}

/// Forward a request to the Node child (forcing `Connection: close` and stripping
/// `Accept-Encoding` so the response is unencoded and framed by EOF), and return
/// the full raw upstream response bytes.
fn forward_to_node(
    node_port: u16,
    method: &str,
    target: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<UpstreamResponse, String> {
    let mut upstream = TcpStream::connect(("127.0.0.1", node_port))
        .map_err(|error| format!("cannot reach node runtime on :{node_port}: {error}"))?;
    let mut request = format!("{method} {target} HTTP/1.1\r\n");
    for (name, value) in headers {
        let lower = name.to_ascii_lowercase();
        // Drop hop-by-hop / framing headers we set ourselves, and encoding so the
        // upstream response is plain text we can inject into.
        if matches!(lower.as_str(), "connection" | "accept-encoding" | "content-length" | "transfer-encoding") {
            continue;
        }
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    request.push_str("Connection: close\r\n");
    request.push_str("Accept-Encoding: identity\r\n");
    request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    request.push_str("\r\n");

    upstream
        .write_all(request.as_bytes())
        .and_then(|()| upstream.write_all(body))
        .and_then(|()| upstream.flush())
        .map_err(|error| format!("cannot send request to node: {error}"))?;

    let mut raw = Vec::new();
    upstream
        .read_to_end(&mut raw)
        .map_err(|error| format!("cannot read node response: {error}"))?;
    parse_response(raw)
}

/// A parsed upstream HTTP response split into its status line, headers, and
/// fully-decoded (de-chunked) body.
struct UpstreamResponse {
    status_line: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

/// Split a raw upstream response into status line, headers, and decoded body
/// (de-chunking transfer-encoding: chunked; otherwise the bytes after the header
/// block, which are complete because the request forced `Connection: close`).
fn parse_response(raw: Vec<u8>) -> Result<UpstreamResponse, String> {
    let split = find_subsequence(&raw, b"\r\n\r\n")
        .ok_or_else(|| "malformed node response (no header terminator)".to_string())?;
    let head = std::str::from_utf8(&raw[..split])
        .map_err(|error| format!("non-utf8 response headers from node: {error}"))?;
    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| "empty node response".to_string())?
        .to_string();
    let mut headers = Vec::new();
    let mut chunked = false;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim().to_string();
            let value = value.trim().to_string();
            if name.eq_ignore_ascii_case("transfer-encoding")
                && value.to_ascii_lowercase().contains("chunked")
            {
                chunked = true;
            }
            headers.push((name, value));
        }
    }
    let raw_body = &raw[split + 4..];
    let body = if chunked {
        decode_chunked(raw_body)?
    } else {
        raw_body.to_vec()
    };
    Ok(UpstreamResponse {
        status_line,
        headers,
        body,
    })
}

/// If the upstream response is HTML, inject the Fast Refresh preamble + WebSocket
/// HMR client, then re-serialize with a correct `Content-Length`, no chunked
/// framing, and `Connection: close`.
fn maybe_inject_hmr(mut response: UpstreamResponse) -> Vec<u8> {
    let content_type = response
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.to_ascii_lowercase())
        .unwrap_or_default();
    let is_html = content_type.contains("text/html");
    if is_html {
        response.body = inject_into_html(&response.body);
    } else if is_server_error(&response.status_line) && content_type.contains("text/plain") {
        // A dev SSR crash: the upstream Node server returns a 5xx with a plain-text
        // error (no HTML document), which the HTML injection above would skip. Wrap
        // the error in a minimal HTML document carrying the HMR preamble + a trigger
        // that shows it in the error overlay, so an SSR failure surfaces the same way
        // a build or runtime error does. Dev/proxy-only — `build-app` never runs this.
        let error_text = String::from_utf8_lossy(&response.body).into_owned();
        response.body = synthesize_ssr_error_document(&error_text).into_bytes();
        set_content_type_html(&mut response.headers);
    }

    let mut out = Vec::new();
    out.extend_from_slice(response.status_line.as_bytes());
    out.extend_from_slice(b"\r\n");
    for (name, value) in &response.headers {
        let lower = name.to_ascii_lowercase();
        // We re-frame the body ourselves, so drop the upstream framing/connection
        // headers and any content-encoding (we forced identity upstream).
        if matches!(lower.as_str(), "content-length" | "transfer-encoding" | "connection" | "content-encoding") {
            continue;
        }
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(format!("Content-Length: {}\r\n", response.body.len()).as_bytes());
    out.extend_from_slice(b"Connection: close\r\n\r\n");
    out.extend_from_slice(&response.body);
    out
}

/// Whether an HTTP status line (`HTTP/1.1 500 Internal Server Error`) is a 5xx.
fn is_server_error(status_line: &str) -> bool {
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.chars().next())
        == Some('5')
}

/// Rewrite (or add) the `Content-Type` header to HTML, used when wrapping a plain-text
/// SSR error in an HTML overlay document.
fn set_content_type_html(headers: &mut Vec<(String, String)>) {
    let mut found = false;
    for (name, value) in headers.iter_mut() {
        if name.eq_ignore_ascii_case("content-type") {
            *value = "text/html; charset=utf-8".to_string();
            found = true;
        }
    }
    if !found {
        headers.push(("Content-Type".to_string(), "text/html; charset=utf-8".to_string()));
    }
}

/// Minimal `text/html` document that carries the HMR preamble (so the overlay client
/// is installed) and a trigger that renders the SSR error in the overlay. The error
/// text lives in a hidden `<pre>` the trigger reads via `textContent`, so it never
/// needs JS-string escaping — only HTML-escaping into the element.
fn synthesize_ssr_error_document(error: &str) -> String {
    format!(
        "<!doctype html><html><head>{preamble}</head><body><pre id=\"__diffpack_ssr_error\" style=\"display:none\">{escaped}</pre><script>if(window.__diffpackOverlay)window.__diffpackOverlay.showBuild({{message:document.getElementById(\"__diffpack_ssr_error\").textContent}});</script></body></html>",
        preamble = hmr_preamble(),
        escaped = html_escape(error),
    )
}

/// HTML-escape text for safe embedding in an element's content.
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Insert the Fast Refresh preamble + WebSocket HMR client at the TOP of `<head>`.
/// It is a blocking classic `<script src>` (loading the Refresh runtime as
/// `window.$RefreshRuntime$`) followed by an inline classic `<script>` that injects
/// the DevTools hook and sets the Refresh globals — both run synchronously during
/// parse, before the app's deferred/async entry module, and both remove themselves
/// so React 19 hydrates a `<head>` identical to what it server-rendered.
fn inject_into_html(body: &[u8]) -> Vec<u8> {
    let Ok(html) = std::str::from_utf8(body) else {
        // Non-utf8 HTML is not something we produce; leave it untouched.
        return body.to_vec();
    };
    let snippet = hmr_preamble();
    if let Some(position) = find_case_insensitive(html, "<head>") {
        let at = position + "<head>".len();
        let mut out = String::with_capacity(html.len() + snippet.len());
        out.push_str(&html[..at]);
        out.push_str(&snippet);
        out.push_str(&html[at..]);
        return out.into_bytes();
    }
    if let Some(position) = rfind_case_insensitive(html, "</body>") {
        let mut out = String::with_capacity(html.len() + snippet.len());
        out.push_str(&html[..position]);
        out.push_str(&snippet);
        out.push_str(&html[position..]);
        return out.into_bytes();
    }
    let mut out = html.to_string();
    out.push_str(&snippet);
    out.into_bytes()
}

/// The blocking `<script src>` for the Fast Refresh runtime plus the inline classic
/// preamble/WS client. Both are classic scripts so they run in document order,
/// synchronously, before the async entry module.
fn hmr_preamble() -> String {
    format!(
        "<script src=\"{REFRESH_RUNTIME_PATH}\"></script><script>{}</script><script>{}</script>",
        crate::hmr::client_script(WS_PATH),
        crate::hmr::overlay_script(),
    )
}

// --- small HTTP helpers (std-only; no dependency needed for a dev proxy) ------

/// Read the request/response head: the start line plus header lines, up to the
/// blank line that terminates the header block.
fn read_head(reader: &mut impl BufRead) -> Result<(String, Vec<(String, String)>), String> {
    let mut start_line = String::new();
    // Skip any stray leading blank lines, then read the request line.
    loop {
        start_line.clear();
        let read = reader
            .read_line(&mut start_line)
            .map_err(|error| format!("cannot read request line: {error}"))?;
        if read == 0 {
            return Err("client closed before sending a request".to_string());
        }
        if !start_line.trim().is_empty() {
            break;
        }
    }
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .map_err(|error| format!("cannot read header line: {error}"))?;
        if read == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
    }
    Ok((start_line.trim_end_matches(['\r', '\n']).to_string(), headers))
}

/// Read a request body based on its `Content-Length` (0 when absent).
fn read_body(reader: &mut impl Read, headers: &[(String, String)]) -> Result<Vec<u8>, String> {
    let length = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; length];
    if length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|error| format!("cannot read request body: {error}"))?;
    }
    Ok(body)
}

fn parse_request_line(line: &str) -> Result<(String, String), String> {
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "empty request line".to_string())?
        .to_string();
    let target = parts
        .next()
        .ok_or_else(|| "request line has no target".to_string())?
        .to_string();
    Ok((method, target))
}

/// Decode an HTTP/1.1 `chunked` transfer-encoding body into its raw bytes.
fn decode_chunked(mut input: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = find_subsequence(input, b"\r\n")
            .ok_or_else(|| "truncated chunk size line".to_string())?;
        let size_line = std::str::from_utf8(&input[..line_end])
            .map_err(|_| "non-utf8 chunk size".to_string())?;
        // A chunk size may carry extensions after a ';'.
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|error| format!("bad chunk size {size_hex:?}: {error}"))?;
        input = &input[line_end + 2..];
        if size == 0 {
            break;
        }
        if input.len() < size {
            return Err("truncated chunk data".to_string());
        }
        out.extend_from_slice(&input[..size]);
        input = &input[size..];
        // Each chunk's data is followed by CRLF.
        if input.len() >= 2 {
            input = &input[2..];
        }
    }
    Ok(out)
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let haystack = haystack.to_ascii_lowercase();
    haystack.find(&needle.to_ascii_lowercase())
}

fn rfind_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let haystack = haystack.to_ascii_lowercase();
    haystack.rfind(&needle.to_ascii_lowercase())
}

// --- watch helpers ------------------------------------------------------------

fn start_watcher(root: &Path) -> Result<Receiver<notify::Result<notify::Event>>, String> {
    start_watcher_paths(&[(root.to_path_buf(), RecursiveMode::Recursive)])
}

/// Start a filesystem watcher over several `(path, mode)` roots on one channel.
/// The SPA path watches `src` recursively AND the project root non-recursively, so
/// root-level files (`index.html`, `vite.config.*`) are seen without recursing into
/// `node_modules`.
/// The source trees `diffpack dev` must watch, derived from the modules the build
/// ACTUALLY compiled rather than from a convention directory.
///
/// Watching only `app/` was wrong for every application bigger than an example: a Next
/// app keeps its components in `components/`, `modules/`, `lib/`, `src/`, and in a
/// monorepo in sibling workspace packages. Editing any of those produced no event at
/// all — the dev server sat there while nothing happened, which is worse than a slow
/// rebuild because it looks like the tool is working. cal.com renders its login form
/// from `apps/web/modules/auth/login-view.tsx` and its design system from
/// `packages/ui/**`; neither is under `app/`.
///
/// Every reachable module's directory is collected and then reduced to the ONE level
/// below their deepest common ancestor: for a monorepo that is `<repo>/apps` and
/// `<repo>/packages`, for a standalone app `<app>/app`, `<app>/components`, … The
/// common ancestor itself is watched non-recursively, so a loose file sitting directly
/// in it (cal.com's root `i18n.json`, which its i18n config reads) is covered without
/// one such file collapsing the whole tree into a single enormous root.
///
/// Modules inside `node_modules` are excluded (a dependency is not edited during a dev
/// session, and the trees are enormous), as is anything inside a dot directory
/// (`.diffpack-next` shims and other generated output — regenerating them during a
/// rebuild must not look like a user edit and retrigger one).
/// Whether `path` is inside an installed dependency or inside generated output — the
/// two kinds of file a source watcher must never treat as a user edit. One rule covers
/// both: `node_modules`, and any dot directory (`.diffpack-output`, `.diffpack-next`,
/// `.next`, `.git`, `.turbo`). Application source never lives in either.
///
/// Only the part of the path BELOW `base` is examined. A project can perfectly well
/// live inside a dot directory itself (`~/.config/site`, a CI checkout under
/// `/.builds/…`); judging the absolute path would then exclude every file in it and
/// the dev server would silently stop reacting to anything.
fn is_dependency_or_generated(path: &Path, base: &Path) -> bool {
    let relative = path.strip_prefix(base).unwrap_or(path);
    relative.components().any(|component| {
        let name = component.as_os_str();
        name == "node_modules" || name.as_encoded_bytes().starts_with(b".")
    })
}

fn source_watch_roots(
    project_root: &Path,
    envs: [&EnvBuild; 3],
) -> Vec<(PathBuf, RecursiveMode)> {
    let mut directories: BTreeSet<PathBuf> = BTreeSet::new();
    for env in envs {
        for id in &env.reachable {
            // A module id can carry a `?query` (route splits, virtual variants); the
            // file on disk is the part before it.
            let path = Path::new(id.split('?').next().unwrap_or(id.as_str()));
            if !path.is_absolute() {
                continue;
            }
            let excluded = path.components().any(|component| {
                let name = component.as_os_str();
                name == "node_modules" || name.as_encoded_bytes().starts_with(b".")
            });
            if excluded {
                continue;
            }
            if let Some(parent) = path.parent() {
                directories.insert(parent.to_path_buf());
            }
        }
    }
    if directories.is_empty() {
        return vec![(project_root.to_path_buf(), RecursiveMode::Recursive)];
    }
    // The deepest common ancestor, never allowed above the project's own parent: a
    // module linked in from somewhere else entirely must not turn `/` (or `/Users`)
    // into a watch root.
    let mut common = directories
        .iter()
        .cloned()
        .reduce(|a, b| {
            a.ancestors()
                .find(|ancestor| b.starts_with(ancestor))
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("/"))
        })
        .unwrap_or_else(|| project_root.to_path_buf());
    let floor = project_root.parent().unwrap_or(project_root);
    if !floor.starts_with(&common) || common.components().count() < 2 {
        common = project_root.to_path_buf();
    }

    let mut roots: BTreeSet<PathBuf> = BTreeSet::new();
    for directory in &directories {
        // A directory outside the common ancestor (only reachable when the ancestor was
        // clamped above) is watched on its own; otherwise take the one segment below it.
        match directory.strip_prefix(&common) {
            Ok(relative) => {
                if let Some(first) = relative.components().next() {
                    roots.insert(common.join(first));
                }
            }
            Err(_) => {
                roots.insert(directory.clone());
            }
        }
    }
    let mut out = vec![(common, RecursiveMode::NonRecursive)];
    out.extend(roots.into_iter().map(|root| (root, RecursiveMode::Recursive)));
    out
}

fn start_watcher_paths(
    roots: &[(PathBuf, RecursiveMode)],
) -> Result<Receiver<notify::Result<notify::Event>>, String> {
    let (events, receiver) = mpsc::channel();
    // TWO event sources feed one channel:
    //
    // 1. The OS-native backend (FSEvents on macOS, inotify on Linux) — the RELIABLE
    //    backstop. It never misses an edit, but macOS FSEvents carries a ~13.5ms fixed
    //    detect latency even at `latency: 0` (measured), which dominates HMR latency.
    // 2. A tight custom POLLER (below) — the FAST path. It stats the (small) source
    //    tree every 2ms comparing full-resolution mtime+len, so it usually detects an
    //    edit in ~1-2ms, well under the FSEvents floor.
    //
    // The poller only needs to be fast-WHEN-it-fires, never complete: FSEvents
    // guarantees nothing is missed (unlike notify's own `PollWatcher`, which we found
    // silently drops ~70% of rapid edits). Whichever source sees an edit first drives
    // the rebuild; the watch loop dedups the other source's echo by (mtime, len), so a
    // double-detection never causes a second rebuild.
    let mut watcher = notify::recommended_watcher({
        let events = events.clone();
        move |event| {
            let _ = events.send(event);
        }
    })
    .map_err(|error| format!("cannot create filesystem watcher: {error}"))?;
    for (path, mode) in roots {
        watcher
            .watch(path, *mode)
            .map_err(|error| format!("cannot start filesystem watcher on {}: {error}", path.display()))?;
    }
    // Leak the watcher so it lives for the whole process (dropping it stops watching).
    Box::leak(Box::new(watcher));
    spawn_supplement_poller(roots.to_vec(), events);
    Ok(receiver)
}

/// The fast supplementary poller (see [`start_watcher_paths`]). Walks the watched
/// roots every 2ms, and on any file whose full-resolution `(mtime, len)` changed
/// since the last scan, sends a synthetic modify event down the same channel as the
/// OS watcher. Detection latency is ~half the interval — far under FSEvents' floor —
/// while FSEvents remains the never-miss backstop. Deliberately reliable where
/// notify's `PollWatcher` was not: full-nanosecond `mtime` + `len`, no rounding.
fn spawn_supplement_poller(
    roots: Vec<(PathBuf, RecursiveMode)>,
    events: Sender<notify::Result<notify::Event>>,
) {
    let _ = std::thread::Builder::new()
        .name("diffpack-fast-poll".into())
        .spawn(move || {
            let mut snapshot: HashMap<PathBuf, (SystemTime, u64)> = HashMap::new();
            let mut first = true;
            loop {
                let scan_started = Instant::now();
                let mut current: HashMap<PathBuf, (SystemTime, u64)> = HashMap::new();
                for (root, mode) in &roots {
                    scan_root(root, *mode, &mut current);
                }
                if !first {
                    for (path, sig) in &current {
                        if snapshot.get(path) != Some(sig) {
                            // A synthetic modify event; only `paths` is read downstream.
                            let event = notify::Event::new(notify::EventKind::Modify(
                                notify::event::ModifyKind::Any,
                            ))
                            .add_path(path.clone());
                            if events.send(Ok(event)).is_err() {
                                return; // receiver gone: dev server shutting down.
                            }
                        }
                    }
                }
                snapshot = current;
                first = false;
                // SELF-THROTTLING. The 2ms interval is tuned for a small source tree,
                // where a scan costs microseconds. A real application's watch set is
                // orders of magnitude larger (cal.com compiles ~6k modules across a
                // monorepo), and a fixed 2ms interval there would burn a whole core
                // scanning. The poller is only a LATENCY optimization — the OS watcher
                // is the never-miss backstop — so it caps itself at a ~20% duty cycle:
                // small trees keep the 2ms floor and their ~1ms detection, large ones
                // degrade gracefully toward the OS watcher's latency instead of
                // degrading the machine.
                let scan = scan_started.elapsed();
                std::thread::sleep(std::cmp::max(Duration::from_millis(2), scan * 4));
            }
        });
}

/// Collect `(mtime, len)` for every regular file under `root` into `out`. Recurses
/// when `mode` is recursive, but never descends into `node_modules` or the diffpack
/// output dir (nothing there is a user source edit, and both are large).
fn scan_root(root: &Path, mode: RecursiveMode, out: &mut HashMap<PathBuf, (SystemTime, u64)>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            if mode == RecursiveMode::Recursive {
                let name = entry.file_name();
                // `node_modules` and every DOT directory. The latter covers `.git`,
                // `.diffpack-output`, `.next`, `.turbo`, `.yarn` and any other tool
                // cache in one rule: application source never lives in a dot directory,
                // and those caches are both large and constantly rewritten, so polling
                // them costs a great deal and can never detect a user edit. The OS
                // watcher still sees them.
                if name == "node_modules" || name.as_encoded_bytes().starts_with(b".") {
                    continue;
                }
                scan_root(&path, mode, out);
            }
        } else if let Ok(meta) = entry.metadata() {
            let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            out.insert(path, (mtime, meta.len()));
        }
    }
}

fn collect_paths(event: notify::Result<notify::Event>) -> Vec<PathBuf> {
    match event {
        Ok(event) => event.paths,
        Err(_) => Vec::new(),
    }
}

/// Coalesce a burst of filesystem events into one batch, given the already-received
/// `first` event. Blocks only until a short QUIET period elapses with no new event —
/// an atomic save fires create+modify+rename within a few ms, so a quiet window
/// collapses that burst into a single rebuild, yet a lone edit (the common case, and
/// what HMR latency is judged on) returns after just one quiet window instead of a
/// fixed long debounce. Hard-capped so a pathological continuous event stream cannot
/// starve the rebuild. Returns every path seen in the burst (unfiltered — the caller
/// filters to module paths).
fn coalesce_batch(
    receiver: &Receiver<notify::Result<notify::Event>>,
    first: notify::Result<notify::Event>,
) -> Vec<PathBuf> {
    // 2ms quiet: FSEvents is created with latency 0 + NoDefer (events delivered
    // ASAP), so a single in-place write is one event and this window returns almost
    // immediately; an atomic save's create+rename burst still coalesces (its
    // inter-event gap is sub-millisecond). Far shorter than the old fixed 60ms window
    // that taxed every single edit with a full wait for a second event that never came.
    const QUIET: Duration = Duration::from_millis(2);
    const CAP: Duration = Duration::from_millis(250);
    let mut paths = collect_paths(first);
    let cap_at = Instant::now() + CAP;
    loop {
        let window = QUIET.min(cap_at.saturating_duration_since(Instant::now()));
        if window.is_zero() {
            break;
        }
        match receiver.recv_timeout(window) {
            Ok(event) => paths.extend(collect_paths(event)),
            Err(_) => break, // quiet window elapsed (or channel closed): burst done.
        }
    }
    paths
}

fn is_module_path(path: &Path) -> bool {
    // Ignore build output and editor scratch files.
    if path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(".diffpack-output" | ".diffpack-next" | "node_modules" | ".git")
        )
    }) {
        return false;
    }
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
    {
        return false;
    }
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "json" | "css" | "scss" | "sass" | "less")
    )
}

/// Whether a path is a build-config file whose edit would change derived
/// aliases/defines/base (not a source module). Handled explicitly by the dev loop
/// so it is neither mis-treated as a source module nor silently ignored.
fn is_config_file(path: &Path) -> bool {
    let name = path.file_name().and_then(|value| value.to_str()).unwrap_or("");
    name.starts_with("vite.config.")
        || name == "package.json"
        || name.starts_with("tsconfig")
        || name.starts_with("diffpack.config")
}

fn src_dir(project_root: &Path) -> String {
    // Mirror config::derive_config's srcDirectory handling by watching `src`
    // (its default) — the watch root only needs to cover editable source.
    if project_root.join("src").is_dir() {
        "src".to_string()
    } else {
        ".".to_string()
    }
}

fn display_relative(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// `diffpack dev` for a plain Vite HTML-entry SPA: a single client environment,
/// static serving, and the same WebSocket + React Fast Refresh HMR the TanStack
/// path uses — but NO Node child (an SPA has no SSR runtime). This is the
/// generalization that makes `diffpack dev` a drop-in for the everyday Vite
/// dev-server workflow (see docs/NEXT_STEPS.md #2), and where the low incremental
/// diff time becomes a daily, browser-visible advantage.
///
/// Topology: emit the browser-ESM client into `.diffpack-output` (with
/// `hmr: true`), then run diffpack's own static server on the public dev port. It
/// serves the emitted chunks/assets, upgrades the WebSocket HMR channel, serves the
/// Fast Refresh runtime, and returns the app document (the rewritten `index.html`
/// with the HMR preamble injected) for `/` and any client-routed path. On a source
/// edit it incrementally rebuilds the client, re-emits, and pushes a targeted
/// WebSocket update so the browser applies the accept / Fast Refresh protocol in
/// place — no reload, no state loss.
mod spa {
    use super::*;
    use crate::html_entry::{self, HeadInjection};
    use std::fs;

    /// Entry point: build the SPA client, start the static+HMR server, and drive
    /// the incremental rebuild loop. Blocks until the watcher stops or an
    /// unsupported edit is hit (a hard error).
    pub fn run_spa(
        options: &DevOptions,
        project_root: &Path,
        index_html: &Path,
    ) -> Result<(), String> {
        let output_root = project_root.join(".diffpack-output");
        // Vite conventions (aliases, base, import.meta.env, sass additionalData)
        // apply when the project has a Vite config; otherwise it is a bare web
        // build. Either way dev mode selects the development dependency builds and
        // turns on HMR instrumentation.
        let vite = [
            "vite.config.ts",
            "vite.config.js",
            "vite.config.mjs",
            "vite.config.mts",
            "vite.config.cjs",
            "vite.config.cts",
        ]
        .iter()
        .any(|name| project_root.join(name).is_file());
        let mut config = crate::config::derive_web_config(project_root, vite)?;
        crate::config::set_web_development_mode(&mut config);
        let base = config.base.clone();

        // Resolve the single module-script entry from index.html (mirrors the
        // `diffpack build` HTML-entry resolution).
        let html = html_entry::parse_file(index_html)?;
        let html_origin = index_html.display().to_string();
        let entry_script = match html.module_scripts.as_slice() {
            [only] => only,
            [] => {
                return Err(format!(
                    "{html_origin}: no local <script type=\"module\" src> entry found"
                ));
            }
            many => {
                return Err(format!(
                    "{html_origin}: {} module script entries; multiple HTML entries are not supported by `diffpack dev` yet",
                    many.len()
                ));
            }
        };
        let entry_path = match entry_script.src.strip_prefix('/') {
            Some(rest) => project_root.join(rest),
            None => index_html
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

        let emit_options = EmitOptions {
            // Same rationale as the TanStack dev path: never minify (HMR re-imports
            // chunks and reads readable Fast Refresh instrumentation).
            minify: false,
            source_map: options.source_map,
            hmr: true,
            ..EmitOptions::default()
        };

        println!(
            "[dev] building SPA client{}...",
            if vite { " (vite mode)" } else { "" }
        );
        let mut client = discover_spa_client(&entry, &config, emit_options)?;

        let (served, _) = emit_spa(&client, project_root, &output_root, &html, &html_origin, &base)?;
        let served_html = Arc::new(Mutex::new(served));

        let refresh_runtime = Arc::new(crate::hmr::find_refresh_runtime(project_root)?);
        // `server.proxy` rules (from the Vite config), shared with each connection.
        let proxy = Arc::new(config.proxy.clone());
        if !proxy.is_empty() {
            println!(
                "[dev] server.proxy: {} rule(s) ({})",
                proxy.len(),
                proxy
                    .iter()
                    .map(|rule| format!("{} -> {}", rule.context, rule.target))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
        let hub = HmrHub::default();
        let listener = TcpListener::bind(("127.0.0.1", options.port))
            .map_err(|error| format!("cannot bind dev port {}: {error}", options.port))?;
        {
            let hub = hub.clone();
            let refresh_runtime = Arc::clone(&refresh_runtime);
            let served_html = Arc::clone(&served_html);
            let output_root = output_root.clone();
            let base = base.clone();
            let proxy = Arc::clone(&proxy);
            std::thread::Builder::new()
                .name("diffpack-dev-spa".into())
                .spawn(move || serve_spa(listener, output_root, base, served_html, hub, refresh_runtime, proxy))
                .map_err(|error| format!("cannot start SPA server thread: {error}"))?;
        }
        println!(
            "[dev] diffpack dev server (SPA) on http://127.0.0.1:{}",
            options.port
        );

        // Watch `src` recursively for module edits, and the project root
        // non-recursively so `index.html` and `vite.config.*` edits are seen
        // (without recursing into node_modules). When there is no `src` dir the
        // recursive root already covers the project root.
        let src_watch = project_root.join(src_dir(project_root));
        let mut watch_roots = vec![(src_watch.clone(), RecursiveMode::Recursive)];
        if src_watch != *project_root {
            watch_roots.push((project_root.to_path_buf(), RecursiveMode::NonRecursive));
        }
        let receiver = start_watcher_paths(&watch_roots)?;
        println!("[dev] watching {}", src_watch.display());

        spa_watch_loop(
            &receiver,
            project_root,
            &output_root,
            index_html,
            &entry,
            &config,
            emit_options,
            html,
            &html_origin,
            &base,
            &mut client,
            &served_html,
            &hub,
        )
    }

    /// (Re)discover the SPA client graph from the entry and wrap it in a fresh
    /// `EnvBuild`. Used for the initial build and for a structural rebuild (a
    /// new-file event, whose module ids shift the whole graph).
    fn discover_spa_client(
        entry: &Path,
        config: &crate::config::WebConfig,
        emit_options: EmitOptions,
    ) -> Result<EnvBuild, String> {
        let mut build_config = config.build.clone();
        build_config.source_maps = emit_options.source_map;
        let (bundler, update) = Bundler::discover_direct_with_config(entry, &build_config)?;
        // The initial build is a hard error: a dev server with nothing loadable to
        // serve should say so, not start and hand the browser a broken chunk.
        for warning in
            crate::bundler::partition_diagnostics(&update.diagnostics, "dev client build")?
        {
            println!("[dev] warning: {warning}");
        }
        let session = bundler.direct_reachability();
        let reachable = session.reachable_modules();
        Ok(EnvBuild {
            bundler,
            session,
            reachable,
            options: emit_options,
        })
    }

    /// Emit the SPA client into `output_root` (the site root), copy the static
    /// `public/` passthrough, and build the served `index.html`: the original
    /// document with its module script rewritten to the emitted `index.js`, the
    /// stylesheet linked when the build produced CSS, and the configured `base`
    /// applied. The HMR preamble is NOT baked in here — it is injected per response
    /// so it always reflects the current runtime. Returns `(served_html, summary)`.
    fn emit_spa(
        client: &EnvBuild,
        project_root: &Path,
        output_root: &Path,
        html: &html_entry::HtmlEntry,
        html_origin: &str,
        base: &str,
    ) -> Result<(String, EmitSummary), String> {
        let reachable = reachable_ids(client);
        let summary = client
            .bundler
            .emit_web(&reachable, output_root, "index.js", client.options)?;
        crate::config::copy_static_public(project_root, output_root)?;
        let mut injection = HeadInjection {
            script_urls: vec![format!("{base}index.js")],
            stylesheet_urls: Vec::new(),
        };
        if summary.css_files > 0 {
            injection.stylesheet_urls.push(format!("{base}index.css"));
        }
        let served = html_entry::apply_base(&html.rewrite(html_origin, &injection)?, base);
        Ok((served, summary))
    }

    /// The incremental rebuild loop for the SPA: coalesce a burst of filesystem
    /// events, incrementally rebuild the client, re-emit, and push a targeted HMR
    /// update. A new file (an existing module the graph never reached) triggers a
    /// structural rebuild + reload, so adding modules works.
    #[allow(clippy::too_many_arguments)]
    fn spa_watch_loop(
        receiver: &Receiver<notify::Result<notify::Event>>,
        project_root: &Path,
        output_root: &Path,
        index_html: &Path,
        entry: &Path,
        config: &crate::config::WebConfig,
        emit_options: EmitOptions,
        mut html: html_entry::HtmlEntry,
        html_origin: &str,
        base: &str,
        client: &mut EnvBuild,
        served_html: &Arc<Mutex<String>>,
        hub: &HmrHub,
    ) -> Result<(), String> {
        // Fingerprint of the emitted stylesheet, so a CSS-only edit is detected and
        // hot-swapped (the <link> is replaced in place) instead of reloading.
        let mut css_fingerprint = stylesheet_fingerprint(output_root);
        // The canonical index.html path, so an edit to it is recognized and the
        // served document re-parsed.
        let index_html_canon = index_html.canonicalize().unwrap_or_else(|_| index_html.to_path_buf());
        // Whether a build-error overlay is currently shown, so the next good rebuild
        // clears it (build-ok).
        let mut build_error_showing = false;
        loop {
            let first = match receiver.recv() {
                Ok(event) => event,
                Err(_) => return Ok(()),
            };
            let paths = coalesce_batch(receiver, first);

            // An index.html edit is not a module edit: re-parse the document and
            // rebuild the served HTML (new title/meta/entry), then reload. The
            // reload re-fetches every chunk, so this also covers any module edits
            // coalesced in the same batch.
            let index_edited = paths
                .iter()
                .any(|path| path.canonicalize().map(|c| c == index_html_canon).unwrap_or(false));
            if index_edited {
                match html_entry::parse_file(index_html) {
                    Ok(fresh) => {
                        html = fresh;
                        let (fresh_doc, _) =
                            emit_spa(client, project_root, output_root, &html, html_origin, base)?;
                        *served_html.lock().unwrap() = fresh_doc;
                        css_fingerprint = stylesheet_fingerprint(output_root);
                        hub.broadcast_reload();
                        println!("[dev] index.html changed -> re-parsed document + reload");
                    }
                    // A transient parse error (mid-edit save) must not kill the
                    // server; keep serving the last-good document.
                    Err(error) => eprintln!("[dev] index.html parse error (kept previous): {error}"),
                }
                continue;
            }

            // A config-file edit changes derived aliases/defines/base; live
            // re-derivation is not implemented, so warn LOUDLY (never silently) and
            // keep serving the startup config rather than mis-treating the config as
            // a source module or killing the server.
            if paths.iter().any(|path| is_config_file(path)) {
                println!(
                    "[dev] WARNING: a config file changed (vite.config.* / package.json / tsconfig). Live config re-derivation (aliases/defines/base) is not implemented — the dev server is STILL USING THE CONFIG FROM STARTUP. Restart `diffpack dev` to apply it."
                );
            }

            let changed = paths
                .into_iter()
                .filter(|path| is_module_path(path) && !is_config_file(path))
                .collect::<BTreeSet<_>>();
            if changed.is_empty() {
                continue;
            }

            // A file the graph never reached that now EXISTS is a new file: adding a
            // module (and, in the same or a later save, an import to it) is a normal
            // dev action, not an error. The new file shifts module ids across the
            // whole graph, so it cannot be hot-patched — re-discover the client from
            // the entry, re-emit, and full-reload the browser. (An unknown file that
            // does NOT exist is a delete of something already unreachable — nothing
            // to do.)
            let has_new_file = changed
                .iter()
                .any(|path| path.exists() && !client.bundler.is_known_module(path));
            if has_new_file {
                let started = Instant::now();
                *client = discover_spa_client(entry, config, emit_options)?;
                let (fresh_doc, summary) =
                    emit_spa(client, project_root, output_root, &html, html_origin, base)?;
                *served_html.lock().unwrap() = fresh_doc;
                css_fingerprint = stylesheet_fingerprint(output_root);
                hub.broadcast_reload();
                let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                println!(
                    "[dev] new file(s) in {} change(s) -> structural rebuild ({} modules, {} chunks) + reload in {elapsed_ms:.1}ms",
                    changed.len(),
                    client.reachable.len(),
                    summary.rendered_chunks,
                );
                continue;
            }

            // Every remaining changed path is a known module (an unknown-and-missing
            // path was a delete of an unreachable file — skip it).
            let known = changed
                .iter()
                .filter(|path| client.bundler.is_known_module(path))
                .cloned()
                .collect::<BTreeSet<_>>();
            if known.is_empty() {
                continue;
            }

            let started = Instant::now();
            let mut counters = EnvCounters::default();
            let mut changed_ids: BTreeSet<String> = BTreeSet::new();
            let mut graph_changed = false;
            // Catch edit-time build errors (e.g. a syntax error) and show them in the
            // browser overlay instead of killing the dev server; keep serving and
            // clear the overlay on the next good rebuild. Structural rebuilds above
            // stay hard errors (fail fast).
            let batch = (|| -> Result<(), String> {
                for path in &known {
                    let rebuilt = client.rebuild(path)?;
                    let (fresh_doc, summary) =
                        emit_spa(client, project_root, output_root, &html, html_origin, base)?;
                    *served_html.lock().unwrap() = fresh_doc;
                    changed_ids.extend(rebuilt.changed_ids.iter().cloned());
                    graph_changed |= rebuilt.graph_changed;
                    counters.add(&rebuilt, summary.rendered_chunks);
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

            let new_css = stylesheet_fingerprint(output_root);
            let css_changed = new_css != css_fingerprint;
            css_fingerprint = new_css;

            // An edit that grew or shrank the reachable set (added/removed an import
            // of another module) re-partitions the chunks — reload rather than push a
            // hot update whose ESM re-import would fail to bind the new/removed
            // exports. Otherwise apply the change surgically: a changed stylesheet is
            // hot-SWAPPED in place (no reload), and changed JS modules take the Fast
            // Refresh path. A pure CSS edit therefore preserves ALL component state.
            let client_update = if graph_changed {
                hub.broadcast_reload();
                "client: graph changed (import added/removed) -> full reload".to_string()
            } else {
                let js_ids = changed_ids
                    .iter()
                    .filter(|id| !is_css_module_id(id))
                    .cloned()
                    .collect::<BTreeSet<_>>();
                let mut notes = Vec::new();
                if css_changed {
                    let hrefs = vec![format!("{base}index.css")];
                    push_css(&hrefs, hub);
                    notes.push(format!(
                        "css hot-swap -> {} sheet(s), {} browser(s)",
                        hrefs.len(),
                        hub.client_count()
                    ));
                }
                if !js_ids.is_empty() {
                    notes.push(push_spa_update(client, &js_ids, base, hub));
                }
                if notes.is_empty() {
                    "client: no visible change".to_string()
                } else {
                    notes.join(" | ")
                }
            };
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            // Same parseable instrumentation shape as the TanStack path (minus the
            // server environment), so the browser oracle can assert incrementality
            // and the low-diff-time budget live.
            println!(
                "[dev] rebuilt {} file(s) in {elapsed_ms:.1}ms | client transformed={} changed={} rendered_chunks={} | {client_update}",
                known.len(),
                counters.transformed,
                counters.changed,
                counters.rendered_chunks,
            );
        }
    }

    /// Push a targeted client HMR update over the WebSocket channel for the SPA:
    /// locate the changed modules' chunk files (entry `index.js`), build the
    /// A cheap content fingerprint of the emitted `index.css`, or `None` when the
    /// build produced no stylesheet. Compared across edits to detect a CSS-only
    /// change that should be hot-swapped rather than reloaded.
    fn stylesheet_fingerprint(output_root: &Path) -> Option<u64> {
        let bytes = fs::read(output_root.join("index.css")).ok()?;
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        Some(hasher.finish())
    }

    /// Whether a module id names a stylesheet source (its factory contributes to the
    /// extracted CSS, not to a JS chunk), so it is applied via a CSS hot-swap rather
    /// than a JS Fast Refresh update.
    fn is_css_module_id(id: &str) -> bool {
        let path = id.split('?').next().unwrap_or(id);
        [".css", ".scss", ".sass", ".less"]
            .iter()
            .any(|ext| path.ends_with(ext))
    }

    /// Broadcast a CSS hot-swap for the given stylesheet hrefs (the browser replaces
    /// each matching `<link>` in place, preserving all component state).
    fn push_css(hrefs: &[String], hub: &HmrHub) {
        let list = hrefs
            .iter()
            .map(|href| json_string(href))
            .collect::<Vec<_>>()
            .join(",");
        hub.send(&format!("{{\"type\":\"css\",\"hrefs\":[{list}]}}"));
    }

    /// base-prefixed chunk URLs the browser re-imports, and broadcast. Returns a
    /// short log fragment.
    fn push_spa_update(
        client: &EnvBuild,
        changed_ids: &BTreeSet<String>,
        base: &str,
        hub: &HmrHub,
    ) -> String {
        if changed_ids.is_empty() {
            return "client: no change".to_string();
        }
        let located = match client
            .bundler
            .hmr_locate(&reachable_ids(client), changed_ids, "index.js")
        {
            Ok(located) => located,
            Err(error) => {
                hub.broadcast_reload();
                return format!("client: locate failed ({error}); reloaded");
            }
        };
        if located.is_empty() {
            return "client: no located modules".to_string();
        }
        let ids = located.iter().map(|l| l.runtime_id).collect::<Vec<_>>();
        let chunks = located
            .iter()
            .map(|l| format!("{base}{}", l.chunk_file))
            .collect::<BTreeSet<_>>();
        let message = format!(
            "{{\"type\":\"update\",\"ids\":[{}],\"chunks\":[{}]}}",
            ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(","),
            chunks.iter().map(|chunk| json_string(chunk)).collect::<Vec<_>>().join(","),
        );
        hub.send(&message);
        format!(
            "client: hmr update -> {} module(s) in {} chunk(s), {} browser(s)",
            ids.len(),
            chunks.len(),
            hub.client_count()
        )
    }

    /// Accept loop for the SPA static+HMR server. Each connection is handled on its
    /// own thread.
    fn serve_spa(
        listener: TcpListener,
        output_root: PathBuf,
        base: String,
        served_html: Arc<Mutex<String>>,
        hub: HmrHub,
        refresh_runtime: Arc<String>,
        proxy: Arc<Vec<crate::vite_config::ProxyRule>>,
    ) {
        for connection in listener.incoming() {
            let Ok(stream) = connection else { continue };
            let output_root = output_root.clone();
            let base = base.clone();
            let served_html = Arc::clone(&served_html);
            let hub = hub.clone();
            let refresh_runtime = Arc::clone(&refresh_runtime);
            let proxy = Arc::clone(&proxy);
            let _ = std::thread::Builder::new()
                .name("diffpack-dev-spa-conn".into())
                .spawn(move || {
                    let _ = handle_spa_connection(
                        stream,
                        &output_root,
                        &base,
                        &served_html,
                        &hub,
                        &refresh_runtime,
                        &proxy,
                    );
                });
        }
    }

    fn handle_spa_connection(
        mut stream: TcpStream,
        output_root: &Path,
        base: &str,
        served_html: &Arc<Mutex<String>>,
        hub: &HmrHub,
        refresh_runtime: &str,
        proxy: &[crate::vite_config::ProxyRule],
    ) -> Result<(), String> {
        let mut reader = BufReader::new(
            stream
                .try_clone()
                .map_err(|error| format!("cannot clone client socket: {error}"))?,
        );
        let (request_line, headers) = read_head(&mut reader)?;
        let (method, target) = parse_request_line(&request_line)?;
        let path = target.split('?').next().unwrap_or(&target);
        let head_only = method.eq_ignore_ascii_case("HEAD");

        // `server.proxy`: a request whose path matches a rule's context is forwarded
        // to the rule's target and the upstream response streamed straight back. This
        // is checked BEFORE static/SPA handling so a `/api` proxy is never shadowed by
        // the SPA fallback document.
        if let Some(rule) = super::dev_proxy::match_rule(proxy, path) {
            let body = read_body(&mut reader, &headers)?;
            match super::dev_proxy::forward(rule, &method, &target, &headers, &body) {
                Ok(response) => {
                    stream
                        .write_all(&response)
                        .and_then(|()| stream.flush())
                        .map_err(|error| format!("cannot write proxied response: {error}"))?;
                }
                Err(error) => {
                    eprintln!("[dev] proxy error for {path}: {error}");
                    write_response(
                        &mut stream,
                        "502 Bad Gateway",
                        "text/plain; charset=utf-8",
                        error.as_bytes(),
                        head_only,
                    )?;
                }
            }
            return Ok(());
        }

        // The WebSocket HMR channel (shared framing with the TanStack path).
        if path == WS_PATH {
            if let Some((_, key)) = headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case("sec-websocket-key"))
            {
                let accept = ws_accept(key.trim());
                let response = format!(
                    "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
                );
                stream
                    .write_all(response.as_bytes())
                    .and_then(|()| stream.flush())
                    .map_err(|error| format!("cannot complete websocket handshake: {error}"))?;
                hub.send_to(&stream, r#"{"type":"connected"}"#);
                hub.register(stream);
                return Ok(());
            }
            return Ok(());
        }

        // The Fast Refresh runtime the preamble imports.
        if path == REFRESH_RUNTIME_PATH {
            write_js(&mut stream, refresh_runtime)?;
            return Ok(());
        }

        // A static file under the emitted site root (chunks, css, assets, copied
        // public/ files). Base-prefix stripped; path traversal rejected.
        if let Some(file) = resolve_static(output_root, base, path) {
            if file.is_file() {
                let bytes = fs::read(&file)
                    .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
                return write_response(&mut stream, "200 OK", content_type(&file), &bytes, head_only);
            }
            // A path that names a concrete file (has an extension) but is missing is
            // a real 404 — not the SPA document.
            if looks_like_file(path) {
                return write_response(
                    &mut stream,
                    "404 Not Found",
                    "text/plain; charset=utf-8",
                    b"not found",
                    head_only,
                );
            }
        }

        // SPA fallback: the app document with the HMR preamble injected fresh.
        let document = served_html.lock().unwrap().clone();
        let injected = inject_into_html(document.as_bytes());
        write_response(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            &injected,
            head_only,
        )
    }

    /// Map a request path to a file under the emitted site root, or `None` for the
    /// document root. Strips the configured `base`, rejects `..` traversal.
    fn resolve_static(output_root: &Path, base: &str, path: &str) -> Option<PathBuf> {
        // Strip the base prefix (base always ends with `/`). A path outside the
        // base is not a static file here.
        let relative = if base == "/" {
            path.trim_start_matches('/')
        } else if let Some(rest) = path.strip_prefix(base) {
            rest
        } else if path == base.trim_end_matches('/') {
            ""
        } else {
            path.trim_start_matches('/')
        };
        if relative.is_empty() {
            return None;
        }
        // Reject anything that could escape the output root.
        if relative.split('/').any(|segment| segment == ".." || segment == ".") {
            return None;
        }
        Some(output_root.join(relative))
    }

    /// Whether a request path names a concrete file (its last segment has an
    /// extension) rather than a client route.
    fn looks_like_file(path: &str) -> bool {
        path.rsplit('/').next().is_some_and(|last| last.contains('.'))
    }

    /// The MIME type for an emitted file, by extension.
    fn content_type(path: &Path) -> &'static str {
        match path.extension().and_then(|value| value.to_str()) {
            Some("js" | "mjs" | "cjs") => "application/javascript; charset=utf-8",
            Some("css") => "text/css; charset=utf-8",
            Some("html") => "text/html; charset=utf-8",
            Some("json" | "map") => "application/json; charset=utf-8",
            Some("svg") => "image/svg+xml",
            Some("png") => "image/png",
            Some("jpg" | "jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("avif") => "image/avif",
            Some("ico") => "image/x-icon",
            Some("woff2") => "font/woff2",
            Some("woff") => "font/woff",
            Some("ttf") => "font/ttf",
            Some("otf") => "font/otf",
            Some("wasm") => "application/wasm",
            Some("txt") => "text/plain; charset=utf-8",
            _ => "application/octet-stream",
        }
    }

    /// Write a complete HTTP/1.1 response (dev; no caching, `Connection: close`).
    fn write_response(
        stream: &mut TcpStream,
        status: &str,
        content_type: &str,
        body: &[u8],
        head_only: bool,
    ) -> Result<(), String> {
        let header = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream
            .write_all(header.as_bytes())
            .map_err(|error| format!("cannot write response head: {error}"))?;
        if !head_only {
            stream
                .write_all(body)
                .map_err(|error| format!("cannot write response body: {error}"))?;
        }
        stream
            .flush()
            .map_err(|error| format!("cannot flush response: {error}"))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn resolve_static_maps_under_root_at_default_base() {
            let root = Path::new("/out");
            assert_eq!(resolve_static(root, "/", "/index.js"), Some(root.join("index.js")));
            assert_eq!(
                resolve_static(root, "/", "/assets/logo-abc123.svg"),
                Some(root.join("assets/logo-abc123.svg"))
            );
            // The document root maps to None (served as the SPA fallback, not a file).
            assert_eq!(resolve_static(root, "/", "/"), None);
        }

        #[test]
        fn resolve_static_strips_a_non_root_base() {
            let root = Path::new("/out");
            assert_eq!(
                resolve_static(root, "/app/", "/app/index.js"),
                Some(root.join("index.js"))
            );
            // The base root itself (no trailing file) is the document.
            assert_eq!(resolve_static(root, "/app/", "/app"), None);
        }

        #[test]
        fn resolve_static_rejects_path_traversal() {
            let root = Path::new("/out");
            assert_eq!(resolve_static(root, "/", "/../etc/passwd"), None);
            assert_eq!(resolve_static(root, "/", "/assets/../../secret"), None);
        }

        #[test]
        fn looks_like_file_distinguishes_assets_from_routes() {
            assert!(looks_like_file("/index.js"));
            assert!(looks_like_file("/assets/x.png"));
            // A client route has no extension in its last segment.
            assert!(!looks_like_file("/about"));
            assert!(!looks_like_file("/users/42"));
        }

        #[test]
        fn config_files_are_recognized_and_modules_are_not() {
            use super::super::is_config_file;
            assert!(is_config_file(Path::new("/app/vite.config.ts")));
            assert!(is_config_file(Path::new("/app/vite.config.mts")));
            assert!(is_config_file(Path::new("/app/package.json")));
            assert!(is_config_file(Path::new("/app/tsconfig.app.json")));
            // Source modules and the app's own JSON data are NOT config.
            assert!(!is_config_file(Path::new("/app/src/App.tsx")));
            assert!(!is_config_file(Path::new("/app/src/data.json")));
        }

        #[test]
        fn content_type_covers_the_emitted_kinds() {
            assert_eq!(content_type(Path::new("a/index.js")), "application/javascript; charset=utf-8");
            assert_eq!(content_type(Path::new("a/index.css")), "text/css; charset=utf-8");
            assert_eq!(content_type(Path::new("a/logo.svg")), "image/svg+xml");
            assert_eq!(content_type(Path::new("a/hero.png")), "image/png");
            assert_eq!(content_type(Path::new("a/blob.bin")), "application/octet-stream");
        }
    }
}

/// `diffpack dev` for a Next.js app-router app (Slice K / spec Slice 5). A Next app
/// has no TanStack/src entry — its "entry" is the app-router file convention — so it
/// needs a third dev topology: the SAME three RSC graphs the production `build-app`
/// path builds (client / react-server / ssr), kept alive per-environment, served by
/// the emitted next orchestrator (`scripts/rsc/next-server.mjs`, embedded here and
/// written into the output dir), with the diffpack reverse proxy in front injecting
/// the Fast Refresh + WebSocket HMR preamble into every served document.
///
/// Two edit classes, both browser-visible:
///  * a `"use client"` island edit → rebuild the client + ssr graphs and push a
///    state-preserving React Fast Refresh update over the WebSocket (no reload, hook
///    state preserved) — the island is a generic refresh boundary, so no `hmr.rs`
///    change is needed;
///  * a Server-Component edit (page/layout, no directive) → rebuild ONLY the
///    react-server graph into an isolated `.rsc` root, copy it to `rsc-render`, and
///    broadcast a reload. The orchestrator spawns a FRESH react-server child per GET,
///    so the reload (and a fresh `curl`) show the new server-rendered content. This
///    is honestly a full reload, not in-place HMR — a server component has no client
///    runtime to hot-swap; documented, not dressed up.
///
/// The `.rsc` indirection is load-bearing: the react-server and ssr graphs both emit
/// a `server/` dir, so re-emitting the react-server graph on a server-component edit
/// would clobber the ssr bundle the orchestrator holds. Emitting react-server to
/// `<out>/.rsc/server` and copying to `<out>/rsc-render` keeps them separate.
mod next {
    use super::*;
    use std::fs;

    /// The next orchestrator, embedded so the dev server is self-contained (it does
    /// not need to locate the diffpack repo at runtime). It is plain Node that wires
    /// the three emitted bundles + manifests into an HTTP app; it reads only absolute
    /// paths derived from its `<output-dir>` argv, so running it from the output dir
    /// is equivalent to running it from the repo.
    const NEXT_SERVER_MJS: &str = include_str!("../scripts/rsc/next-server.mjs");

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
        let refresh_runtime = Arc::new(crate::hmr::find_refresh_runtime(project_root)?);
        let hub = HmrHub::default();

        // Everything the orchestrator needs beyond the graph outputs, written
        // before ANY boot (warm or cold) so both paths run the same bytes.
        let next_server_script = output_root.join("next-server.mjs");
        let write_orchestrator_scripts = || -> Result<(), String> {
            fs::create_dir_all(&output_root)
                .map_err(|error| format!("cannot create {}: {error}", output_root.display()))?;
            fs::write(&next_server_script, NEXT_SERVER_MJS).map_err(|error| {
                format!("cannot write {}: {error}", next_server_script.display())
            })?;
            // The orchestrator imports this as a sibling (the one place that joins the
            // three references manifests into the divergent-id ssrModuleMapping).
            let ssr_module_map = output_root.join(crate::rsc::SSR_MODULE_MAP_FILE);
            fs::write(&ssr_module_map, include_str!("../scripts/rsc/ssr-module-map.mjs"))
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
        let start_proxy = |node_port: u16| -> Result<(), String> {
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
                    serve_proxy(proxy_listener, node_port, hub, refresh_runtime, static_dir)
                })
                .map_err(|error| format!("cannot start proxy thread: {error}"))?;
            Ok(())
        };

        // Build order is load-bearing only for the client (its manifests feed the
        // server graphs). The react-server and ssr graphs read those manifests and
        // never each other's output — react-server emits into `.rsc`/`rsc-render`,
        // ssr into `server/` — so the two build concurrently, which is most of the
        // difference between a ~8.5s and a ~6s cold start on cal.com.
        println!("[dev] next: building client graph...");
        let mut client = build_next_client(project_root, &output_root, emit_options)?;
        println!("[dev] next: building react-server and ssr graphs (concurrently)...");
        let (react_server_result, ssr_result) = std::thread::scope(|scope| {
            let react_server = scope.spawn(|| {
                build_next_react_server(project_root, &output_root, &rsc_root, emit_options)
            });
            let ssr = scope.spawn(|| build_next_ssr(project_root, &output_root, emit_options));
            (react_server.join(), ssr.join())
        });
        let mut react_server = react_server_result
            .map_err(|_| "the react-server build thread panicked".to_string())??;
        let mut ssr = ssr_result.map_err(|_| "the ssr build thread panicked".to_string())??;

        write_orchestrator_scripts()?;
        let node_port = free_port()?;
        let mut node = boot_node(node_port)?;
        println!("[dev] next orchestrator listening on 127.0.0.1:{node_port}");
        start_proxy(node_port)?;
        println!(
            "[dev] diffpack dev server (next app-router) on http://127.0.0.1:{} (proxying node :{node_port})",
            options.port
        );

        // Watch the app dir recursively (all convention files live there) + the project
        // root non-recursively (next.config.*), without recursing into node_modules.
        // `run_next` is only reached after `is_app_router` said yes, so a missing app dir
        // is an invariant break, not a case to fall back from.
        let app_dir = crate::next_adapter::app_dir(project_root).ok_or_else(|| {
            format!("next dev: {} has no app/ or src/app directory", project_root.display())
        })?;
        let mut watch_roots = source_watch_roots(project_root, [&client, &react_server, &ssr]);
        if !watch_roots
            .iter()
            .any(|(root, mode)| *mode == RecursiveMode::Recursive && app_dir.starts_with(root))
        {
            watch_roots.push((app_dir.clone(), RecursiveMode::Recursive));
        }
        if !watch_roots.iter().any(|(root, _)| root == project_root) {
            watch_roots.push((project_root.to_path_buf(), RecursiveMode::NonRecursive));
        }
        for (root, mode) in &watch_roots {
            println!(
                "[dev] watching {}{}",
                root.display(),
                if *mode == RecursiveMode::Recursive { "" } else { " (top level only)" }
            );
        }
        // The shallowest watched directory; everything watched lives under it.
        let watch_base = watch_roots
            .iter()
            .map(|(root, _)| root.clone())
            .min_by_key(|root| root.components().count())
            .unwrap_or_else(|| project_root.to_path_buf());
        let receiver = start_watcher_paths(&watch_roots)?;

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
    ) -> Result<EnvBuild, String> {
        let build = discover_next_env_reconciled(project_root, "client", options, &|config| {
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
        prepare: &dyn Fn(&mut AppConfig) -> Result<(), String>,
    ) -> Result<EnvBuild, String> {
        let mut regenerated = false;
        loop {
            let mut config = next_config_dev(project_root, environment)?;
            prepare(&mut config)?;
            let build = discover_next_env(config, environment, options)?;
            if !crate::next_adapter::reconcile_async_islands(
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
    ) -> Result<EnvBuild, String> {
        let mut config = next_config_dev(project_root, "react-server")?;
        register_server_virtual_modules(&mut config, project_root, output_root)?;
        let build = discover_next_env(config, "react-server", options)?;
        emit_next_react_server(&build, output_root, rsc_root)?;
        Ok(build)
    }

    /// Build the ssr-of-flight graph into `<out>/server`, leaving the bundler alive.
    fn build_next_ssr(
        project_root: &Path,
        output_root: &Path,
        options: EmitOptions,
    ) -> Result<EnvBuild, String> {
        let build = discover_next_env_reconciled(project_root, "ssr", options, &|config| {
            register_server_virtual_modules(config, project_root, output_root)
        })?;
        emit_next_ssr(&build, output_root)?;
        Ok(build)
    }

    /// Scaffold `.diffpack-next/` + derive the dev [`AppConfig`] for one environment
    /// (a Next app-router project is guaranteed here — the caller already detected
    /// it, so `None` is an internal invariant break, a clear error not a silent skip).
    fn next_config_dev(project_root: &Path, environment: &str) -> Result<AppConfig, String> {
        crate::next_adapter::configure_dev(project_root, environment)?.ok_or_else(|| {
            format!(
                "next dev: {} is not an app-router project (environment={environment})",
                project_root.display()
            )
        })
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
        let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config.build)?;
        for warning in
            crate::bundler::partition_diagnostics(&update.diagnostics, &format!("dev {label} build"))?
        {
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
        let summary = client
            .bundler
            .emit_public(&reachable, output_root, client.options)?;
        config::copy_static_public(project_root, &summary.output_dir)?;
        // next/image: emit downscaled responsive variants for every public raster.
        let images = crate::next_adapter::scan_public_images(project_root)?;
        if !images.is_empty() {
            crate::next_adapter::emit_image_variants(project_root, &summary.output_dir, &images)?;
        }
        // next/font/local: copy the app's font files to the hashed URLs the generated
        // @font-face rules point at, so dev serves the same face the build does.
        crate::next_font::emit_font_assets(project_root, &summary.output_dir)?;
        // The route -> client-chunk manifest the server build's start-manifest reads.
        let client_manifest = client
            .bundler
            .client_route_manifest(&reachable, "client.js", "/")?;
        client_manifest.write(&output_root.join(manifest::CLIENT_MANIFEST_FILE))?;
        // Manifest #1 (client-references): the react-server render resolves each
        // `"use client"` `$$id` through it, and the orchestrator joins it with the
        // ssr manifest to build the SSR consumer manifest.
        let client_references = client
            .bundler
            .client_references_manifest(&reachable, "client.js")?;
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
    fn emit_next_client_hmr(client: &EnvBuild, output_root: &Path) -> Result<usize, String> {
        let reachable = reachable_ids(client);
        client
            .bundler
            .emit_public_incremental(&reachable, output_root, client.options)
    }

    /// Emit the react-server graph into `<rsc_root>/server`, preserve its compiled CSS
    /// to `<out>/public/rsc.css`, write its own client-references manifest, and copy
    /// the bundle to `<out>/rsc-render` (where the orchestrator spawns it per GET).
    fn emit_next_react_server(
        env: &EnvBuild,
        output_root: &Path,
        rsc_root: &Path,
    ) -> Result<EmitSummary, String> {
        let reachable = reachable_ids(env);
        let summary = env.bundler.emit_server(&reachable, rsc_root, env.options)?;
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
                format!("cannot preserve react-server CSS to {}: {error}", dest.display())
            })?;
        }
        // This build's OWN client-references manifest (its ids) — written beside the
        // ssr build's at `<out>` under the react-server-distinct file name, so the two
        // server-like graphs cannot clobber each other and the render seam reads the
        // SAME pair of files in dev as it does after a production build. It is also
        // the set that decides which client references a flight can carry, which the
        // seam validates the ssr manifest against.
        let server_references = env
            .bundler
            .client_references_manifest(&reachable, "server.mjs")?;
        server_references
            .write(&output_root.join(crate::rsc::REACT_SERVER_REFERENCES_MANIFEST_FILE))?;
        // Copy the fresh bundle to rsc-render (the orchestrator's per-request child).
        replace_dir(&rsc_root.join("server"), &output_root.join("rsc-render"))?;
        Ok(summary)
    }

    /// Emit the ssr-of-flight graph into `<out>/server` and write ITS client-references
    /// manifest (its ids) to `<out>` — the manifest the orchestrator reads as the SSR
    /// half of the divergent-id module map.
    fn emit_next_ssr(env: &EnvBuild, output_root: &Path) -> Result<EmitSummary, String> {
        let reachable = reachable_ids(env);
        let summary = env.bundler.emit_server(&reachable, output_root, env.options)?;
        let server_references = env
            .bundler
            .client_references_manifest(&reachable, "server.mjs")?;
        server_references.write(&output_root.join(crate::rsc::SERVER_REFERENCES_MANIFEST_FILE))?;
        Ok(summary)
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
            let ssr = profile.stage("hot-render-ssr", || {
                graphs
                    .ssr
                    .map(|(env, ids)| {
                        Self::render(env, ids, &dir.join(format!("ssr.{}.mjs", self.seq)))
                    })
                    .transpose()
            })?
            .flatten();
            let react_server = profile.stage("hot-render-rsc", || {
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
                crate::bundler::ModuleFormat::Esm,
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

    /// How long the watch loop waits for another file event before flushing the full
    /// chunk re-emit it owes disk. Short enough that a developer who stops typing and
    /// reaches for the reload button finds disk already current; long enough that a
    /// burst of keystrokes coalesces into ONE re-emit instead of one per keystroke.
    const SETTLE_MS: u64 = 150;

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
        Some(SystemTime::now().duration_since(modified).ok()?.as_secs_f64() * 1_000.0)
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
    fn flush_owed(
        owed: &mut OwedEmits,
        output_root: &Path,
        rsc_root: &Path,
        client: &EnvBuild,
        react_server: &EnvBuild,
        ssr: &EnvBuild,
    ) -> Result<usize, String> {
        let mut rendered_chunks = 0;
        if owed.client {
            rendered_chunks += emit_next_client_hmr(client, output_root)?;
            owed.client = false;
        }
        if owed.ssr {
            rendered_chunks += emit_next_ssr(ssr, output_root)?.rendered_chunks;
            owed.ssr = false;
        }
        if owed.react_server {
            rendered_chunks +=
                emit_next_react_server(react_server, output_root, rsc_root)?.rendered_chunks;
            owed.react_server = false;
        }
        Ok(rendered_chunks)
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
    ) -> Result<(), String> {
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
        loop {
            let first = if owed.any() {
                // Deferred work outstanding: wait only for the settle window. Nothing
                // new means typing stopped — flush disk now.
                match receiver.recv_timeout(Duration::from_millis(SETTLE_MS)) {
                    Ok(event) => event,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let started = Instant::now();
                        let flushed = owed;
                        match flush_owed(&mut owed, output_root, rsc_root, client, react_server, ssr)
                        {
                            Ok(rendered_chunks) => {
                                // Disk now matches the live processes for the
                                // react-server graph, so a worker respawned from here
                                // on needs no replay.
                                if flushed.react_server {
                                    hot.mark_react_server_on_disk();
                                }
                                println!(
                                    "[dev] next settled: full re-emit of {} in {:.1}ms | rendered_chunks={rendered_chunks}",
                                    flushed.label(),
                                    started.elapsed().as_secs_f64() * 1_000.0,
                                );
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
                    project_root,
                    output_root,
                    rsc_root,
                    client,
                    react_server,
                    ssr,
                    emit_options,
                )?;
                // Everything was re-emitted from scratch: no deferred debt survives, and
                // the worker replay list (whose micro-chunks the new bundles already
                // contain) is discharged.
                owed = OwedEmits::default();
                hot.reset(output_root)?;
                restart_next_node(node, next_server_script, project_root, output_root, node_port)?;
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
            if let Some(lag) = changed.iter().filter_map(|path| detection_lag_ms(path)).fold(
                None,
                |worst: Option<f64>, lag| Some(worst.map_or(lag, |worst| worst.max(lag))),
            ) {
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
                    project_root,
                    output_root,
                    rsc_root,
                    client,
                    react_server,
                    ssr,
                    emit_options,
                )?;
                // Every graph was just emitted from scratch: disk IS the truth, so the
                // deferred debt and the worker replay list are both discharged.
                owed = OwedEmits::default();
                hot.reset(output_root)?;
                restart_next_node(node, next_server_script, project_root, output_root, node_port)?;
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

            // Then the browser push — the user-visible edit-to-update event.
            let update = if !island_ids.is_empty() {
                // State-preserving React Fast Refresh (no reload). Push a MICRO-CHUNK
                // (only the changed modules) so the browser re-parses ~1 KB, not the
                // ~1 MB entry chunk; served directly off disk by the proxy.
                profile.stage("push-client", || {
                    hmr_push_client(client, &island_ids, hub, Some(output_root))
                })
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
            if build_error_showing {
                hub.broadcast_build_ok();
                build_error_showing = false;
            }

            println!(
                "[dev] next rebuilt {} file(s) | update in {update_ms:.1}ms | client transformed={} changed={} | server transformed={} changed={} | {hot_note} | {update} | disk re-emit owed: {}{}",
                changed.len(),
                client_c.transformed,
                client_c.changed,
                server_c.transformed,
                server_c.changed,
                if owed.any() { owed.label() } else { "none".to_string() },
                profile.label(),
            );
        }
    }

    /// Re-discover and re-emit all three graphs from scratch (used for structural /
    /// graph-changing edits, where module ids shift across the partition).
    fn rebuild_all(
        project_root: &Path,
        output_root: &Path,
        rsc_root: &Path,
        client: &mut EnvBuild,
        react_server: &mut EnvBuild,
        ssr: &mut EnvBuild,
        emit_options: EmitOptions,
    ) -> Result<(), String> {
        *client = build_next_client(project_root, output_root, emit_options)?;
        *react_server = build_next_react_server(project_root, output_root, rsc_root, emit_options)?;
        *ssr = build_next_ssr(project_root, output_root, emit_options)?;
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
                format!("cannot start next orchestrator ({}): {error}", script.display())
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
        let read = fs::read_dir(src)
            .map_err(|error| format!("cannot read {}: {error}", src.display()))?;
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
        let source = fs::read(from)
            .map_err(|error| format!("cannot read {}: {error}", from.display()))?;
        if fs::read(to).ok().as_deref() == Some(source.as_slice()) {
            return Ok(());
        }
        fs::write(to, source).map_err(|error| {
            format!("cannot copy {} -> {}: {error}", from.display(), to.display())
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

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
            assert_eq!(owed.label(), "client + ssr", "an island edit owes both browser-facing graphs");
            owed.react_server = true;
            assert_eq!(owed.label(), "client + ssr + react-server");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injects_hmr_client_into_head() {
        let html = b"<!doctype html><html><head><title>x</title></head><body><div id=\"root\"></div></body></html>";
        let out = inject_into_html(html);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("$RefreshRuntime$"));
        assert!(text.contains("WebSocket"));
        // The dev error overlay is injected alongside the HMR client.
        assert!(text.contains("__diffpackOverlay"), "overlay must be injected: {text}");
        // Injected inside <head>, before the title, so it runs before app modules.
        let head = text.find("<head>").unwrap();
        let snippet = text.find("$RefreshRuntime$").unwrap();
        let title = text.find("<title>").unwrap();
        assert!(head < snippet && snippet < title, "snippet must sit at the top of <head>: {text}");
    }

    #[test]
    fn preamble_is_a_blocking_runtime_script_before_the_inline_client() {
        let html = b"<!doctype html><html><head><title>x</title></head><body></body></html>";
        let out = inject_into_html(html);
        let text = String::from_utf8(out).unwrap();
        let runtime = text.find(REFRESH_RUNTIME_PATH).unwrap();
        let inline = text.find("WebSocket").unwrap();
        // The blocking runtime <script src> precedes the inline client, both classic
        // (no type=module) so they run in order before the async app entry.
        assert!(runtime < inline, "runtime script must precede the inline client");
        assert!(!text.contains("type=\"module\">"), "preamble scripts must be classic");
    }

    #[test]
    fn injects_before_body_when_no_head() {
        let html = b"<html><body><p>hi</p></body></html>";
        let out = inject_into_html(html);
        let text = String::from_utf8(out).unwrap();
        let snippet = text.find("$RefreshRuntime$").unwrap();
        let close_body = text.find("</body>").unwrap();
        assert!(snippet < close_body);
    }

    #[test]
    fn websocket_accept_matches_rfc6455_example() {
        // The canonical example from RFC 6455 section 1.3.
        assert_eq!(ws_accept("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn decodes_a_chunked_body() {
        // "Wiki" + "pedia" in two chunks, then a zero terminator.
        let raw = b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n";
        assert_eq!(decode_chunked(raw).unwrap(), b"Wikipedia");
    }

    #[test]
    fn parses_a_plain_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 5\r\n\r\nhello".to_vec();
        let parsed = parse_response(raw).unwrap();
        assert_eq!(parsed.status_line, "HTTP/1.1 200 OK");
        assert_eq!(parsed.body, b"hello");
    }

    #[test]
    fn non_html_response_is_not_injected() {
        let response = UpstreamResponse {
            status_line: "HTTP/1.1 200 OK".to_string(),
            headers: vec![("Content-Type".to_string(), "application/javascript".to_string())],
            body: b"console.log(1)".to_vec(),
        };
        let out = maybe_inject_hmr(response);
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("$RefreshRuntime$"));
        assert!(text.contains("Content-Length: 14"));
    }

    #[test]
    fn a_500_text_plain_becomes_an_html_overlay_document() {
        // A dev SSR crash returns a 5xx text/plain error with no HTML document. The
        // proxy wraps it in a minimal HTML doc carrying the HMR preamble + overlay so
        // the failure is surfaced the same way a build/runtime error is.
        let response = UpstreamResponse {
            status_line: "HTTP/1.1 500 Internal Server Error".to_string(),
            headers: vec![("Content-Type".to_string(), "text/plain".to_string())],
            body: b"ReferenceError: boom is not defined\n    at render (/server.js:1:1)".to_vec(),
        };
        let out = maybe_inject_hmr(response);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("text/html"), "must be re-typed as HTML: {text}");
        assert!(text.contains("__diffpackOverlay"), "must carry the overlay: {text}");
        assert!(text.contains("showBuild"), "must trigger the overlay: {text}");
        assert!(
            text.contains("ReferenceError: boom is not defined"),
            "must embed the SSR error text: {text}"
        );
    }

    #[test]
    fn non_5xx_text_plain_is_left_untouched() {
        // A normal (2xx) text/plain response must not be wrapped in an overlay doc.
        let response = UpstreamResponse {
            status_line: "HTTP/1.1 200 OK".to_string(),
            headers: vec![("Content-Type".to_string(), "text/plain".to_string())],
            body: b"plain body".to_vec(),
        };
        let out = maybe_inject_hmr(response);
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("__diffpackOverlay"), "2xx text/plain must not be wrapped: {text}");
        assert!(text.contains("plain body"));
    }
}

/// `diffpack preview` — serve a completed `diffpack build` output over HTTP, the
/// analogue of `vite preview`. Static files are served from `build_dir`; a
/// client-routed path with no matching file falls back to `index.html` (SPA
/// fallback), exactly as `vite preview` does. Blocks forever (a server); the caller
/// backgrounds and kills it. This is a production-preview server, not the dev
/// server: no HMR, no rebuild, no watch.
pub fn preview(build_dir: &Path, port: u16) -> Result<(), String> {
    let index = build_dir.join("index.html");
    if !index.is_file() {
        return Err(format!(
            "{} has no index.html — run `diffpack build <root>` first",
            build_dir.display()
        ));
    }
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|error| format!("cannot bind preview port {port}: {error}"))?;
    println!(
        "diffpack preview serving {} on http://127.0.0.1:{port}",
        build_dir.display()
    );
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let build_dir = build_dir.to_path_buf();
        let _ = std::thread::Builder::new()
            .name("diffpack-preview-conn".into())
            .spawn(move || {
                let _ = handle_preview_connection(stream, &build_dir);
            });
    }
    Ok(())
}

fn handle_preview_connection(mut stream: TcpStream, build_dir: &Path) -> Result<(), String> {
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|error| format!("cannot clone preview socket: {error}"))?,
    );
    let (request_line, _headers) = read_head(&mut reader)?;
    let (method, target) = parse_request_line(&request_line)?;
    let head_only = method.eq_ignore_ascii_case("HEAD");
    let path = target.split('?').next().unwrap_or(&target);

    // Map the request path to a file under the build dir, rejecting `..` traversal.
    let relative = path.trim_start_matches('/');
    let traversal = relative
        .split('/')
        .any(|segment| segment == ".." || segment == ".");
    if !traversal && !relative.is_empty() {
        let candidate = build_dir.join(relative);
        if candidate.is_file() {
            let bytes = std::fs::read(&candidate)
                .map_err(|error| format!("cannot read {}: {error}", candidate.display()))?;
            return write_preview_response(
                &mut stream,
                "200 OK",
                preview_content_type(&candidate),
                &bytes,
                head_only,
            );
        }
        // A concrete file (has an extension) that is missing is a real 404, not the
        // SPA document.
        if relative.rsplit('/').next().is_some_and(|last| last.contains('.')) {
            return write_preview_response(
                &mut stream,
                "404 Not Found",
                "text/plain; charset=utf-8",
                b"not found",
                head_only,
            );
        }
    }

    // SPA fallback: the build's index.html.
    let index = build_dir.join("index.html");
    let bytes = std::fs::read(&index)
        .map_err(|error| format!("cannot read {}: {error}", index.display()))?;
    write_preview_response(
        &mut stream,
        "200 OK",
        "text/html; charset=utf-8",
        &bytes,
        head_only,
    )
}

fn write_preview_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    head_only: bool,
) -> Result<(), String> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .map_err(|error| format!("cannot write preview response head: {error}"))?;
    if !head_only {
        stream
            .write_all(body)
            .map_err(|error| format!("cannot write preview response body: {error}"))?;
    }
    stream
        .flush()
        .map_err(|error| format!("cannot flush preview response: {error}"))
}

fn preview_content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|value| value.to_str()) {
        Some("js" | "mjs" | "cjs") => "application/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("json" | "map") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("avif") => "image/avif",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("wasm") => "application/wasm",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// The native dev proxy (`server.proxy`). A dev request whose path matches a rule's
/// context is forwarded verbatim to the rule's target and the upstream response is
/// streamed straight back — the same pass-through Vite's `http-proxy` performs for a
/// simple `{ '/api': 'http://localhost:3001' }` rule.
pub mod dev_proxy {
    use super::*;
    use crate::vite_config::ProxyRule;

    /// The first rule whose context matches `path`. Vite matches a context as a path
    /// prefix; a leading `^` (a regex anchor, the common `'^/api'` form) is treated
    /// as an anchored prefix, which is the same set for a plain prefix pattern.
    pub fn match_rule<'a>(rules: &'a [ProxyRule], path: &str) -> Option<&'a ProxyRule> {
        rules.iter().find(|rule| {
            let context = rule.context.strip_prefix('^').unwrap_or(&rule.context);
            !context.is_empty() && path.starts_with(context)
        })
    }

    /// Split a `http://host:port` (or `ws://...`) target into `(host, port)`. The
    /// scheme is stripped; a missing port defaults to 80. A `wss`/`https` target
    /// defaults to 443, though the native proxy speaks plain HTTP (a TLS upstream is
    /// a documented limitation, surfaced by the caller, not a silent misconnect).
    pub fn target_host_port(target: &str) -> Result<(String, u16), String> {
        let (scheme, rest) = match target.split_once("://") {
            Some((scheme, rest)) => (scheme, rest),
            None => ("http", target),
        };
        // Drop any path/query on the target authority.
        let authority = rest.split(['/', '?']).next().unwrap_or(rest);
        let default_port = if scheme == "https" || scheme == "wss" {
            443
        } else {
            80
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) => (
                host.to_string(),
                port.parse::<u16>()
                    .map_err(|error| format!("bad proxy target port in {target:?}: {error}"))?,
            ),
            None => (authority.to_string(), default_port),
        };
        if host.is_empty() {
            return Err(format!("proxy target {target:?} has no host"));
        }
        Ok((host, port))
    }

    /// Forward a request to the rule's target and return the raw upstream response
    /// bytes (status line + headers + body) to write straight back to the client.
    /// `changeOrigin` rewrites the forwarded `Host` header to the target's host, as
    /// Vite does. The original `path_and_query` is forwarded unchanged (a `rewrite`
    /// function is not expressible natively and was surfaced at config time).
    pub fn forward(
        rule: &ProxyRule,
        method: &str,
        path_and_query: &str,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Result<Vec<u8>, String> {
        let (host, port) = target_host_port(&rule.target)?;
        let mut upstream = TcpStream::connect((host.as_str(), port)).map_err(|error| {
            format!(
                "dev proxy cannot reach {} ({host}:{port}): {error}",
                rule.target
            )
        })?;
        let mut request = format!("{method} {path_and_query} HTTP/1.1\r\n");
        let mut wrote_host = false;
        for (name, value) in headers {
            let lower = name.to_ascii_lowercase();
            // Rewrite Host under changeOrigin; force Connection: close and identity
            // encoding so the response framing is a single, complete read.
            if lower == "host" {
                wrote_host = true;
                if rule.change_origin {
                    request.push_str(&format!("Host: {host}:{port}\r\n"));
                    continue;
                }
            }
            if lower == "connection" || lower == "accept-encoding" {
                continue;
            }
            request.push_str(&format!("{name}: {value}\r\n"));
        }
        if !wrote_host {
            request.push_str(&format!("Host: {host}:{port}\r\n"));
        }
        request.push_str("Connection: close\r\n");
        request.push_str("Accept-Encoding: identity\r\n");
        request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        upstream
            .write_all(request.as_bytes())
            .and_then(|()| upstream.write_all(body))
            .and_then(|()| upstream.flush())
            .map_err(|error| format!("dev proxy cannot send to {}: {error}", rule.target))?;
        let mut response = Vec::new();
        upstream
            .read_to_end(&mut response)
            .map_err(|error| format!("dev proxy cannot read from {}: {error}", rule.target))?;
        Ok(response)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn rule(context: &str, target: &str) -> ProxyRule {
            ProxyRule {
                context: context.to_string(),
                target: target.to_string(),
                change_origin: false,
                ws: false,
            }
        }

        #[test]
        fn matches_prefix_and_regex_anchored_context() {
            let rules = vec![rule("/api", "http://localhost:3001")];
            assert!(match_rule(&rules, "/api/users").is_some());
            assert!(match_rule(&rules, "/other").is_none());
            let anchored = vec![rule("^/socket", "http://localhost:4000")];
            assert!(match_rule(&anchored, "/socket/io").is_some());
        }

        #[test]
        fn parses_target_host_and_port() {
            assert_eq!(
                target_host_port("http://localhost:3001").unwrap(),
                ("localhost".to_string(), 3001)
            );
            assert_eq!(
                target_host_port("http://example.com").unwrap(),
                ("example.com".to_string(), 80)
            );
            assert_eq!(
                target_host_port("https://api.example.com/base").unwrap(),
                ("api.example.com".to_string(), 443)
            );
        }

        #[test]
        fn forwards_a_request_to_a_live_upstream_and_returns_the_response() {
            // A throwaway in-process upstream returns a fixed response, proving the
            // proxy connects, forwards, and passes the upstream bytes back verbatim.
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let port = listener.local_addr().unwrap().port();
            let handle = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                // Read the request head (up to blank line) so we can assert the path.
                let (line, _headers) = read_head(&mut reader).unwrap();
                let body = b"{\"ok\":true}";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.write_all(body).unwrap();
                stream.flush().unwrap();
                line
            });
            let rule = rule("/api", &format!("http://127.0.0.1:{port}"));
            let raw = forward(
                &rule,
                "GET",
                "/api/users?limit=1",
                &[("Host".to_string(), "127.0.0.1".to_string())],
                b"",
            )
            .unwrap();
            let text = String::from_utf8_lossy(&raw);
            assert!(text.starts_with("HTTP/1.1 200 OK"), "{text}");
            assert!(text.contains("{\"ok\":true}"), "{text}");
            let upstream_request_line = handle.join().unwrap();
            assert!(
                upstream_request_line.contains("/api/users?limit=1"),
                "upstream saw the original path+query: {upstream_request_line}"
            );
        }
    }
}
