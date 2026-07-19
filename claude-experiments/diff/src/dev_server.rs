//! `diffpack dev`: a long-lived, live-rebuild development server with full-page
//! browser reload.
//!
//! This is where Diffpack's incremental thesis becomes observable. The `build-app`
//! CLI is a cold process per invocation, so its already-incremental emit path (the
//! per-chunk render cache, incremental `emit_public`/`emit_server`) never gets
//! exercised across edits. The dev server keeps a `Bundler` (plus its reachability
//! session) alive PER ENVIRONMENT and re-emits on file change, so a leaf edit
//! re-transforms exactly one module and re-renders exactly one chunk from a
//! long-lived process.
//!
//! Architecture (all native Rust; Node runs only the app's own SSR runtime, never
//! the build):
//!
//! 1. Build the client environment (`emit_public` + persist
//!    `client-manifest.json`) then the server environment (register the TanStack
//!    manifest + server-fn resolver virtual modules, `emit_server`) exactly as
//!    `build-app` does, but keep both bundlers alive. The mandatory
//!    client-before-server order is preserved (the server manifest needs the
//!    finished client chunk URLs).
//! 2. Boot the emitted `server/index.mjs` as a child Node process on an internal
//!    loopback port (the app's own SSR runtime).
//! 3. Put a diffpack-native reverse proxy in front on the public dev port. It
//!    forwards every request to the Node child and injects a tiny SSE live-reload
//!    client into served HTML. The reload channel (`/__diffpack_dev/events`) is
//!    diffpack's; Node only runs the app.
//! 4. Watch the source tree with `notify`, coalescing duplicate/atomic-save
//!    events. On a module edit: incrementally rebuild the client bundler ->
//!    incremental `emit_public` -> re-persist the client manifest -> incrementally
//!    rebuild the server bundler -> `emit_server` -> restart the Node child ->
//!    push a full-page reload over the live-reload channel.
//!
//! SCOPE (full-page live reload only). Deferred, with clear hard errors rather
//! than silent partial handling: React Fast Refresh / state-preserving HMR, CSS
//! hot-swap without reload, route-tree regeneration on add/rename, new-file
//! handling, config-change handling, WebSocket-driven partial updates, and error
//! overlays. An edit class this slice does not handle is a hard error naming what
//! is unsupported, never a silent/partial rebuild.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
        for module in result.removed {
            self.reachable.remove(&module);
        }
        self.reachable.extend(result.added);
        for diagnostic in &update.diagnostics {
            eprintln!("[dev] diagnostic: {diagnostic}");
        }
        Ok(Rebuilt {
            transformed,
            changed,
            changed_ids: update.delta.changed.clone(),
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

/// Run the dev server. Blocks, serving until the filesystem watcher stops or an
/// unsupported edit is encountered (a hard error).
pub fn run(options: DevOptions) -> Result<(), String> {
    let project_root = options
        .project_root
        .canonicalize()
        .map_err(|error| format!("cannot open project root {}: {error}", options.project_root.display()))?;
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
            .spawn(move || serve_proxy(proxy_listener, node_port, hub, refresh_runtime))
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
    loop {
        // Block for the first event, then coalesce a short burst (atomic saves
        // fire create+modify+rename in quick succession).
        let first = match receiver.recv() {
            Ok(event) => event,
            Err(_) => return Ok(()), // watcher dropped: clean shutdown.
        };
        let mut paths = collect_paths(first);
        let deadline = Instant::now() + Duration::from_millis(60);
        while let Ok(remaining) = deadline.checked_duration_since(Instant::now()).map_or(Err(mpsc::RecvTimeoutError::Timeout), |timeout| receiver.recv_timeout(timeout)) {
            paths.extend(collect_paths(remaining));
        }

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

        // A route-file add/rename/remove mutates the route tree. Regenerate it
        // natively and fully rebuild both environments (re-discovering the graph
        // from the new tree), then push a full-page reload. State-preserving graph
        // extension / partial HMR is still deferred, but this replaces the prior
        // hard-error crash with a correct full reload.
        let route_mutation = changed
            .iter()
            .any(|path| is_route_tree_mutation(path, project_root, client_env, server_env));
        if route_mutation {
            let started = Instant::now();
            crate::route_tree::generate_for_project(project_root)?;
            *client_env = build_client(project_root, output_root, client_env.options)?;
            *server_env = build_server(project_root, output_root, server_env.options)?;
            // A route-tree mutation re-derives both graphs from scratch; the module
            // ids shift, so this class cannot be hot-patched — restart the Node
            // child (rare, only on add/rename/remove) and full-reload the browser.
            restart_node(node, index_mjs, node_port, control_port)?;
            hub.broadcast_reload();
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            println!(
                "[dev] route tree changed ({} file(s)) in {elapsed_ms:.1}ms | regenerated + full rebuild + reload pushed",
                changed.len(),
            );
            continue;
        }

        for path in &changed {
            classify_edit(path, project_root, client_env, server_env)?;
        }

        let started = Instant::now();
        let mut client = EnvCounters::default();
        let mut server_c = EnvCounters::default();
        let mut touched = false;
        // Accumulate the changed module ids per environment so, after re-emit, one
        // targeted HMR update covers the whole coalesced batch.
        let mut client_changed_ids: BTreeSet<String> = BTreeSet::new();
        let mut server_changed_ids: BTreeSet<String> = BTreeSet::new();

        for path in &changed {
            // Rebuild whichever environment(s) actually own the module. A route
            // module is in both graphs; a client-only or server-only module is in
            // just one.
            if client_env.bundler.is_known_module(path) {
                let rebuilt = client_env.rebuild(path)?;
                let summary = emit_client(client_env, project_root, output_root)?;
                client_changed_ids.extend(rebuilt.changed_ids.iter().cloned());
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
                server_c.add(&rebuilt, summary.rendered_chunks);
                touched = true;
            }
        }

        if !touched {
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
        let client_update = hmr_push_client(client_env, &client_changed_ids, hub);

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
) -> String {
    if changed_ids.is_empty() {
        return "client: no change".to_string();
    }
    let located = match client_env.bundler.hmr_locate(
        &reachable_ids(client_env),
        changed_ids,
        "client.js",
    ) {
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
        .map(|l| format!("/{}", l.chunk_file))
        .collect::<BTreeSet<_>>();
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
    let mut stream = TcpStream::connect(("127.0.0.1", control_port))
        .map_err(|error| format!("cannot reach hmr control on :{control_port}: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .ok();
    let request = format!(
        "POST /__diffpack_hmr HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
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
        Err(format!(
            "control endpoint returned: {}",
            head.lines().next().unwrap_or("<no status>")
        ))
    }
}

/// Classify an edited path and hard-error on any class this slice does not yet
/// handle, naming exactly what is unsupported. A supported edit is a content edit
/// to a module already present in the client or server graph.
fn classify_edit(
    path: &Path,
    project_root: &Path,
    client: &EnvBuild,
    server: &EnvBuild,
) -> Result<(), String> {
    let name = path.file_name().and_then(|value| value.to_str()).unwrap_or("");

    // Config changes require re-deriving the whole build config (aliases,
    // conditions, virtual modules) and a full rebuild — not handled by the
    // live-reload slice.
    let is_config = matches!(
        name,
        "vite.config.ts" | "vite.config.js" | "package.json"
    ) || name.starts_with("tsconfig")
        || name.starts_with("diffpack.config");
    if is_config {
        return Err(format!(
            "unsupported dev edit: config file {} changed. Config-change handling (re-deriving aliases/conditions/virtual modules) is not implemented by the full-page-reload dev slice; restart `diffpack dev` to pick it up.",
            display_relative(path, project_root)
        ));
    }

    // (Route-file add/rename/remove is handled earlier in the loop by native
    // route-tree regeneration + full rebuild, and the diffpack-generated
    // `routeTree.gen.ts` is filtered out as self-output — so neither reaches
    // here.)

    // A NON-route module in neither graph is a new/deleted file the build never
    // reached. A route file here is handled earlier (route-tree mutation); a
    // non-route new file needs graph extension from a new root, still deferred.
    if !client.bundler.is_known_module(path) && !server.bundler.is_known_module(path) {
        let what = if path.exists() { "new file" } else { "deleted file" };
        return Err(format!(
            "unsupported dev edit: {what} {} is not in the client or server module graph. Non-route new-file / graph-extension handling is not implemented by the full-page-reload dev slice (route-file add/rename/remove IS handled via native route-tree regeneration; edits to existing modules only otherwise).",
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
    let entry = config
        .entry
        .clone()
        .ok_or_else(|| "no client entry found for the app".to_string())?;
    let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config.build)?;
    for diagnostic in &update.diagnostics {
        println!("[dev] client known gap: {diagnostic}");
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
    let entry = config
        .entry
        .clone()
        .ok_or_else(|| "no ssr entry found for the app".to_string())?;
    let (bundler, update) = Bundler::discover_direct_with_config(&entry, &config.build)?;
    for diagnostic in &update.diagnostics {
        println!("[dev] server known gap: {diagnostic}");
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
fn serve_proxy(listener: TcpListener, node_port: u16, hub: HmrHub, refresh_runtime: Arc<String>) {
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let hub = hub.clone();
        let refresh_runtime = Arc::clone(&refresh_runtime);
        let _ = std::thread::Builder::new()
            .name("diffpack-dev-conn".into())
            .spawn(move || {
                if let Err(error) = handle_connection(stream, node_port, &hub, &refresh_runtime) {
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
    let is_html = response
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.to_ascii_lowercase().contains("text/html"))
        .unwrap_or(false);
    if is_html {
        response.body = inject_into_html(&response.body);
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
        "<script src=\"{REFRESH_RUNTIME_PATH}\"></script><script>{}</script>",
        crate::hmr::client_script(WS_PATH)
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
    let (events, receiver) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = events.send(event);
    })
    .map_err(|error| format!("cannot create filesystem watcher: {error}"))?;
    watcher
        .watch(root, RecursiveMode::Recursive)
        .map_err(|error| format!("cannot start filesystem watcher on {}: {error}", root.display()))?;
    // Leak the watcher so it lives for the whole process (dropping it stops
    // watching). The dev server runs until killed.
    Box::leak(Box::new(watcher));
    Ok(receiver)
}

fn collect_paths(event: notify::Result<notify::Event>) -> Vec<PathBuf> {
    match event {
        Ok(event) => event.paths,
        Err(_) => Vec::new(),
    }
}

fn is_module_path(path: &Path) -> bool {
    // Ignore build output and editor scratch files.
    if path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(".diffpack-output" | "node_modules" | ".git")
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
        Some("js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" | "json" | "css")
    )
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
}
