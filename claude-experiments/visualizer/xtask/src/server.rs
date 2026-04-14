use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::{
    extract::{Path as AxumPath, State},
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
    Router,
};
use futures_util::stream::{Stream, StreamExt};
use notify::{EventKind, RecursiveMode, Watcher};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;
use axum::http::HeaderValue;

#[derive(Clone, Debug)]
struct SceneUpdate {
    name: String,
    content: String,
}

#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<SceneUpdate>,
    scenes_dir: PathBuf,
}

pub async fn run(web_dir: PathBuf, scenes_dir: PathBuf, port: u16) {
    let (tx, _) = broadcast::channel::<SceneUpdate>(16);
    let state = AppState {
        tx: tx.clone(),
        scenes_dir: scenes_dir.clone(),
    };

    spawn_watcher(scenes_dir.clone(), tx.clone());

    let app = Router::new()
        .route("/events", get(sse_handler))
        .route("/scenes", get(list_scenes))
        .route("/scenes/:name", get(get_scene))
        .fallback_service(ServeDir::new(&web_dir))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache, no-store, must-revalidate"),
        ))
        .with_state(state);

    let addr = ("127.0.0.1", port);
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
    eprintln!("Serving {} on http://127.0.0.1:{port}", web_dir.display());
    eprintln!("Watching scenes in {}", scenes_dir.display());
    eprintln!("  Write to any .dsl file in that directory to push updates");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("serve failed");
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("Shutting down");
}

fn spawn_watcher(scenes_dir: PathBuf, tx: broadcast::Sender<SceneUpdate>) {
    std::thread::spawn(move || {
        let (notify_tx, notify_rx) =
            std::sync::mpsc::channel::<notify::Result<notify::Event>>();
        let mut watcher = match notify::recommended_watcher(move |res| {
            let _ = notify_tx.send(res);
        }) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("failed to create watcher: {e}");
                return;
            }
        };

        if let Err(e) = watcher.watch(&scenes_dir, RecursiveMode::NonRecursive) {
            eprintln!("failed to watch {}: {e}", scenes_dir.display());
            return;
        }

        let mut last_sent: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        for res in notify_rx {
            let event = match res {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("watch error: {e}");
                    continue;
                }
            };

            if !matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                continue;
            }

            // Debounce — small delay to avoid reading mid-write
            std::thread::sleep(Duration::from_millis(30));

            for path in &event.paths {
                let Some(name) = scene_name_from_path(path) else {
                    continue;
                };
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        if last_sent.get(&name).map(String::as_str) == Some(content.as_str()) {
                            continue;
                        }
                        let _ = tx.send(SceneUpdate {
                            name: name.clone(),
                            content: content.clone(),
                        });
                        last_sent.insert(name, content);
                    }
                    Err(e) => eprintln!("failed to read {}: {e}", path.display()),
                }
            }
        }
    });
}

fn scene_name_from_path(path: &Path) -> Option<String> {
    if path.extension().and_then(|e| e.to_str()) != Some("dsl") {
        return None;
    }
    path.file_stem().and_then(|s| s.to_str()).map(String::from)
}

async fn list_scenes(State(state): State<AppState>) -> String {
    let mut names: Vec<String> = std::fs::read_dir(&state.scenes_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| scene_name_from_path(&e.path()))
                .collect()
        })
        .unwrap_or_default();
    names.sort();

    // Return as a simple JSON array
    let quoted: Vec<String> = names
        .iter()
        .map(|n| format!("\"{}\"", n.replace('"', "\\\"")))
        .collect();
    format!("[{}]", quoted.join(","))
}

async fn get_scene(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> String {
    let path = state.scenes_dir.join(format!("{name}.dsl"));
    std::fs::read_to_string(path).unwrap_or_default()
}

async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.tx.subscribe();

    let live = BroadcastStream::new(rx).filter_map(|res| async move {
        res.ok().map(|update| {
            // Event with named event type and JSON-encoded data
            let data = format!(
                "{{\"name\":\"{}\",\"content\":{}}}",
                update.name.replace('"', "\\\""),
                json_escape_string(&update.content)
            );
            Ok(Event::default().data(data))
        })
    });

    Sse::new(live).keep_alive(KeepAlive::default())
}

fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
