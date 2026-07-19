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
}

/// The reload broadcast fan-out. Each connected SSE client's socket is held here;
/// a reload writes one `data: reload` event to every one, pruning any that error.
#[derive(Clone, Default)]
struct ReloadHub {
    clients: Arc<Mutex<Vec<TcpStream>>>,
}

impl ReloadHub {
    fn register(&self, stream: TcpStream) {
        self.clients.lock().unwrap().push(stream);
    }

    /// Push a full-page reload to every connected browser. Sockets that error
    /// (the tab was closed) are dropped.
    fn broadcast_reload(&self) {
        let mut clients = self.clients.lock().unwrap();
        clients.retain_mut(|stream| stream.write_all(b"data: reload\n\n").and_then(|()| stream.flush()).is_ok());
    }
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
        minify: options.minify,
        source_map: options.source_map,
        ..EmitOptions::default()
    };

    // 1. Initial full build: client then server (order is load-bearing — the
    // server manifest reads the client's finished chunk map).
    println!("[dev] building client...");
    let mut client = build_client(&project_root, &output_root, emit_options)?;
    println!("[dev] building server...");
    let mut server = build_server(&project_root, &output_root, emit_options)?;

    // 2. Boot the emitted Node SSR runtime on an internal loopback port.
    let node_port = free_port()?;
    let index_mjs = output_root.join("server/index.mjs");
    let mut node = spawn_node(&index_mjs, node_port)?;
    wait_for_node(node_port)?;
    println!("[dev] node SSR runtime listening on 127.0.0.1:{node_port}");

    // 3. Reverse proxy on the public dev port, injecting the SSE reload client.
    let hub = ReloadHub::default();
    let proxy_listener = TcpListener::bind(("127.0.0.1", options.port))
        .map_err(|error| format!("cannot bind dev port {}: {error}", options.port))?;
    {
        let hub = hub.clone();
        std::thread::Builder::new()
            .name("diffpack-dev-proxy".into())
            .spawn(move || serve_proxy(proxy_listener, node_port, hub))
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
    node: &mut Child,
    client_env: &mut EnvBuild,
    server_env: &mut EnvBuild,
    hub: &ReloadHub,
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
            .collect::<BTreeSet<_>>();
        if changed.is_empty() {
            continue;
        }

        for path in &changed {
            classify_edit(path, project_root, client_env, server_env)?;
        }

        let started = Instant::now();
        let mut client = EnvCounters::default();
        let mut server_c = EnvCounters::default();
        let mut touched = false;

        for path in &changed {
            // Rebuild whichever environment(s) actually own the module. A route
            // module is in both graphs; a client-only or server-only module is in
            // just one.
            if client_env.bundler.is_known_module(path) {
                let rebuilt = client_env.rebuild(path)?;
                let summary = emit_client(client_env, project_root, output_root)?;
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
                server_c.add(&rebuilt, summary.rendered_chunks);
                touched = true;
            }
        }

        if !touched {
            continue;
        }

        // Restart the Node SSR child so the new server chunks are loaded, then
        // push a full-page reload to every connected browser.
        restart_node(node, index_mjs, node_port)?;
        hub.broadcast_reload();

        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        // Per-edit incremental instrumentation, exercised live from a long-lived
        // process. `changed` is the sharp incremental-transform signal (exactly
        // one module's content changed for a leaf/route-component edit) and
        // `rendered_chunks` is the incremental-emit signal (exactly one chunk
        // re-rendered). Printed as a stable, parseable line the browser oracle
        // asserts on.
        println!(
            "[dev] rebuilt {} file(s) in {elapsed_ms:.1}ms | client transformed={} changed={} rendered_chunks={} | server transformed={} changed={} rendered_chunks={} | reload pushed",
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

    // The generated route tree changes when routes are added/removed/renamed;
    // regenerating it (and its virtual route modules) is deferred.
    if name == "routeTree.gen.ts" {
        return Err(format!(
            "unsupported dev edit: {} changed. Route-tree regeneration (add/rename/remove a route) is not implemented by the full-page-reload dev slice.",
            display_relative(path, project_root)
        ));
    }

    // A module in neither graph is a NEW file (or a file the build never
    // reached). Handling it needs route-tree regeneration / graph extension from
    // a new root, which this slice defers.
    if !client.bundler.is_known_module(path) && !server.bundler.is_known_module(path) {
        // A deleted file also lands here (canonicalize fails / not in graph). Both
        // add and remove are the deferred route-tree-mutation class.
        let what = if path.exists() { "new file" } else { "deleted file" };
        return Err(format!(
            "unsupported dev edit: {what} {} is not in the client or server module graph. New-file / route-tree add/rename/remove handling is not implemented by the full-page-reload dev slice (edits to existing modules only).",
            display_relative(path, project_root)
        ));
    }

    Ok(())
}

/// Build the client environment fresh (mirrors `build-app <root> client`) and
/// leave the bundler alive.
fn build_client(
    project_root: &Path,
    output_root: &Path,
    options: EmitOptions,
) -> Result<EnvBuild, String> {
    let config = config::derive_config(project_root, "client")?;
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

/// Spawn the emitted `server/index.mjs` under Node on `port` (loopback only).
fn spawn_node(index_mjs: &Path, port: u16) -> Result<Child, String> {
    Command::new("node")
        .arg(index_mjs)
        .env("PORT", port.to_string())
        .env("HOST", "127.0.0.1")
        .spawn()
        .map_err(|error| format!("cannot start node SSR runtime ({}): {error}", index_mjs.display()))
}

/// Kill the current Node child, spawn a fresh one on the same port, and wait for
/// it to accept connections.
fn restart_node(node: &mut Child, index_mjs: &Path, port: u16) -> Result<(), String> {
    let _ = node.kill();
    let _ = node.wait();
    *node = spawn_node(index_mjs, port)?;
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
/// on its own thread: SSE reload subscribers are held open in the hub; every
/// other request is forwarded to the Node child with a live-reload script injected
/// into any HTML response.
fn serve_proxy(listener: TcpListener, node_port: u16, hub: ReloadHub) {
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let hub = hub.clone();
        let _ = std::thread::Builder::new()
            .name("diffpack-dev-conn".into())
            .spawn(move || {
                if let Err(error) = handle_connection(stream, node_port, &hub) {
                    // A dropped browser connection is normal; log at a low volume.
                    let _ = error;
                }
            });
    }
}

// The injected script SELF-REMOVES synchronously (like TanStack Start's own
// inline bootstrap scripts) so it leaves no foreign DOM node in the React-hydrated
// document — the EventSource closure keeps running after the node is gone, but
// React sees a head/body identical to what it server-rendered, so there is no
// hydration mismatch.
const RELOAD_SNIPPET: &str = "<script>(function(){var s=document.currentScript;try{var es=new EventSource(\"/__diffpack_dev/events\");es.onmessage=function(e){if(e.data===\"reload\"){es.close();location.reload();}};}catch(_){}if(s)s.remove();})();</script>";

/// The SSE endpoint the injected client subscribes to for reload events.
const EVENTS_PATH: &str = "/__diffpack_dev/events";

fn handle_connection(mut stream: TcpStream, node_port: u16, hub: &ReloadHub) -> Result<(), String> {
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|error| format!("cannot clone client socket: {error}"))?,
    );
    let (request_line, headers) = read_head(&mut reader)?;
    let (method, target) = parse_request_line(&request_line)?;

    if target == EVENTS_PATH {
        // Establish the SSE stream and hand the socket to the reload hub. The
        // handler returns; the hub keeps the socket alive and writes to it on the
        // next reload.
        let response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n: connected\n\n";
        stream
            .write_all(response.as_bytes())
            .map_err(|error| format!("cannot open SSE stream: {error}"))?;
        stream.flush().ok();
        hub.register(stream);
        return Ok(());
    }

    // Read the request body (for server-fn POSTs) so it forwards intact.
    let body = read_body(&mut reader, &headers)?;
    let upstream = forward_to_node(node_port, &method, &target, &headers, &body)?;
    let response = maybe_inject_reload(upstream);
    stream
        .write_all(&response)
        .map_err(|error| format!("cannot write response to client: {error}"))?;
    stream.flush().ok();
    Ok(())
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

/// If the upstream response is HTML, inject the SSE reload client before the last
/// `</body>` (or append it), then re-serialize with a correct `Content-Length`,
/// no chunked framing, and `Connection: close`.
fn maybe_inject_reload(mut response: UpstreamResponse) -> Vec<u8> {
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

/// Insert the reload snippet into HTML. Placed inside `<head>` (immediately after
/// the opening tag) because React 19 hydration is tolerant of extra hoistable
/// `<head>` script nodes, so the injection does not cause a hydration mismatch.
/// Falls back to before `</body>`, then to appending, if there is no `<head>`.
fn inject_into_html(body: &[u8]) -> Vec<u8> {
    let Ok(html) = std::str::from_utf8(body) else {
        // Non-utf8 HTML is not something we produce; leave it untouched.
        return body.to_vec();
    };
    if let Some(position) = find_case_insensitive(html, "<head>") {
        let at = position + "<head>".len();
        let mut out = String::with_capacity(html.len() + RELOAD_SNIPPET.len());
        out.push_str(&html[..at]);
        out.push_str(RELOAD_SNIPPET);
        out.push_str(&html[at..]);
        return out.into_bytes();
    }
    if let Some(position) = rfind_case_insensitive(html, "</body>") {
        let mut out = String::with_capacity(html.len() + RELOAD_SNIPPET.len());
        out.push_str(&html[..position]);
        out.push_str(RELOAD_SNIPPET);
        out.push_str(&html[position..]);
        return out.into_bytes();
    }
    let mut out = html.to_string();
    out.push_str(RELOAD_SNIPPET);
    out.into_bytes()
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
    fn injects_reload_client_into_head() {
        let html = b"<!doctype html><html><head><title>x</title></head><body><div id=\"root\"></div></body></html>";
        let out = inject_into_html(html);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("EventSource"));
        // Injected inside <head>, before the title.
        let head = text.find("<head>").unwrap();
        let snippet = text.find("EventSource").unwrap();
        let title = text.find("<title>").unwrap();
        assert!(head < snippet && snippet < title, "snippet must sit at the top of <head>: {text}");
    }

    #[test]
    fn injects_before_body_when_no_head() {
        let html = b"<html><body><p>hi</p></body></html>";
        let out = inject_into_html(html);
        let text = String::from_utf8(out).unwrap();
        let snippet = text.find("EventSource").unwrap();
        let close_body = text.find("</body>").unwrap();
        assert!(snippet < close_body);
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
        let out = maybe_inject_reload(response);
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("EventSource"));
        assert!(text.contains("Content-Length: 14"));
    }
}
