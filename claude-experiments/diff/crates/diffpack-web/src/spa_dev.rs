//! Development server orchestration for plain HTML-entry applications.

pub mod spa {
    use std::collections::BTreeSet;
    use std::net::TcpListener;
    use std::path::Path;
    use std::sync::mpsc::Receiver;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use diffpack_default_loader::driver::{EmitOptions, EmitSummary, partition_diagnostics};
    use notify::RecursiveMode;

    use crate::dev_build::{DevOptions, EnvBuild, EnvCounters};
    use crate::html_entry::{self, HeadInjection};
    use crate::watch::{
        coalesce_batch, is_config_file, is_module_path, source_dir as src_dir,
        start_paths as start_watcher_paths,
    };
    use crate::websocket::HmrHub;

    /// Entry point: build the SPA client, start the static+HMR server, and drive
    /// the incremental rebuild loop. Blocks until the watcher stops or an
    /// unsupported edit is hit (a hard error).
    pub fn run_spa(
        options: &DevOptions,
        project_root: &Path,
        index_html: &Path,
        mut config: crate::config::WebConfig,
        profile_name: &str,
    ) -> Result<(), String> {
        let output_root = project_root.join(".diffpack-output");
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

        println!("[dev] building SPA client{}...", profile_name);
        let mut client = discover_spa_client(&entry, &config, emit_options)?;

        let (served, _) = emit_spa(
            &client,
            project_root,
            &output_root,
            &html,
            &html_origin,
            &base,
        )?;
        let served_html = Arc::new(Mutex::new(served));

        let refresh_runtime = Arc::new(crate::hmr::find_refresh_runtime(project_root)?);
        // Resolved proxy rules, shared with each connection.
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
                .spawn(move || {
                    crate::spa_server::serve(
                        listener,
                        output_root,
                        base,
                        served_html,
                        hub,
                        refresh_runtime,
                        proxy,
                    )
                })
                .map_err(|error| format!("cannot start SPA server thread: {error}"))?;
        }
        println!(
            "[dev] diffpack dev server (SPA) on http://127.0.0.1:{}",
            options.port
        );

        // Watch `src` recursively for module edits, and the project root
        // non-recursively so `index.html` and resolved configuration edits are seen
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
        let (bundler, update) = crate::compiler::discover(entry, &build_config)?;
        // The initial build is a hard error: a dev server with nothing loadable to
        // serve should say so, not start and hand the browser a broken chunk.
        for warning in partition_diagnostics(&update.diagnostics, "dev client build")? {
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
        let reachable = client.reachable.clone();
        let summary =
            client
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
        let mut css_fingerprint = crate::hmr::stylesheet_fingerprint(output_root, "index.css");
        // The canonical index.html path, so an edit to it is recognized and the
        // served document re-parsed.
        let index_html_canon = index_html
            .canonicalize()
            .unwrap_or_else(|_| index_html.to_path_buf());
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
            let index_edited = paths.iter().any(|path| {
                path.canonicalize()
                    .map(|c| c == index_html_canon)
                    .unwrap_or(false)
            });
            if index_edited {
                match html_entry::parse_file(index_html) {
                    Ok(fresh) => {
                        html = fresh;
                        let (fresh_doc, _) =
                            emit_spa(client, project_root, output_root, &html, html_origin, base)?;
                        *served_html.lock().unwrap() = fresh_doc;
                        css_fingerprint =
                            crate::hmr::stylesheet_fingerprint(output_root, "index.css");
                        hub.broadcast_reload();
                        println!("[dev] index.html changed -> re-parsed document + reload");
                    }
                    // A transient parse error (mid-edit save) must not kill the
                    // server; keep serving the last-good document.
                    Err(error) => {
                        eprintln!("[dev] index.html parse error (kept previous): {error}")
                    }
                }
                continue;
            }

            // A config-file edit changes derived aliases/defines/base; live
            // re-derivation is not implemented, so warn LOUDLY (never silently) and
            // keep serving the startup config rather than mis-treating the config as
            // a source module or killing the server.
            let profile_config_changed = |path: &Path| {
                let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
                is_config_file(&path) || config.configuration_files.contains(&path)
            };
            if paths.iter().any(|path| profile_config_changed(path)) {
                println!(
                    "[dev] WARNING: a build configuration file changed. Live profile re-derivation is not implemented — the dev server is STILL USING THE CONFIG FROM STARTUP. Restart `diffpack dev` to apply it."
                );
            }

            let changed = paths
                .into_iter()
                .filter(|path| is_module_path(path) && !profile_config_changed(path))
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
                css_fingerprint = crate::hmr::stylesheet_fingerprint(output_root, "index.css");
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

            let new_css = crate::hmr::stylesheet_fingerprint(output_root, "index.css");
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
                    .filter(|id| !crate::hmr::is_stylesheet_module(id))
                    .cloned()
                    .collect::<BTreeSet<_>>();
                let mut notes = Vec::new();
                if css_changed {
                    let hrefs = vec![format!("{base}index.css")];
                    crate::hmr::push_css_update(&hrefs, hub);
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
            .hmr_locate(&client.reachable, changed_ids, "index.js")
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
        hub.send(&message);
        format!(
            "client: hmr update -> {} module(s) in {} chunk(s), {} browser(s)",
            ids.len(),
            chunks.len(),
            hub.client_count()
        )
    }

    fn json_string(value: &str) -> String {
        serde_json::to_string(value).expect("serializing a string cannot fail")
    }
}
