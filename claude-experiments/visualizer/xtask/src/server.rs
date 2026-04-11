use std::convert::Infallible;
use std::path::PathBuf;
use std::time::Duration;

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
    Router,
};
use futures_util::stream::{Stream, StreamExt};
use notify::{EventKind, RecursiveMode, Watcher};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::services::ServeDir;

#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<String>,
    dsl_path: PathBuf,
}

pub async fn run(web_dir: PathBuf, dsl_path: PathBuf, port: u16) {
    let (tx, _) = broadcast::channel::<String>(16);
    let state = AppState {
        tx: tx.clone(),
        dsl_path: dsl_path.clone(),
    };

    // Spawn file-watcher thread. Uses notify's blocking recommended_watcher,
    // which runs the callback off the main thread.
    spawn_watcher(dsl_path.clone(), tx.clone());

    let app = Router::new()
        .route("/events", get(sse_handler))
        .route("/scene.dsl", get(dsl_handler))
        .fallback_service(ServeDir::new(&web_dir))
        .with_state(state);

    let addr = ("127.0.0.1", port);
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
    eprintln!("Serving {} on http://127.0.0.1:{port}", web_dir.display());
    eprintln!("Watching {}", dsl_path.display());
    eprintln!("  POST nothing — just write to {} to push updates", dsl_path.display());

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("serve failed");
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("Shutting down");
}

fn spawn_watcher(path: PathBuf, tx: broadcast::Sender<String>) {
    std::thread::spawn(move || {
        // notify sends events on its own thread; we receive via std mpsc
        let (notify_tx, notify_rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
        let mut watcher = match notify::recommended_watcher(move |res| {
            let _ = notify_tx.send(res);
        }) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("failed to create watcher: {e}");
                return;
            }
        };

        // Watch the parent dir so we still get events if the file is replaced (atomic writes)
        let watch_target = path.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
        if let Err(e) = watcher.watch(&watch_target, RecursiveMode::NonRecursive) {
            eprintln!("failed to watch {}: {e}", watch_target.display());
            return;
        }

        // Send initial content on startup
        if let Ok(content) = std::fs::read_to_string(&path) {
            let _ = tx.send(content);
        }

        let mut last_sent: Option<String> = None;
        for res in notify_rx {
            let event = match res {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("watch error: {e}");
                    continue;
                }
            };

            // Only react to events touching our specific file
            let touches_file = event.paths.iter().any(|p| {
                p.file_name() == path.file_name()
                    || p.canonicalize().ok() == path.canonicalize().ok()
            });
            if !touches_file {
                continue;
            }

            if !matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                continue;
            }

            // Debounce: small delay so we don't read mid-write
            std::thread::sleep(Duration::from_millis(30));

            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    if last_sent.as_deref() == Some(content.as_str()) {
                        continue;
                    }
                    let _ = tx.send(content.clone());
                    last_sent = Some(content);
                }
                Err(e) => eprintln!("failed to read {}: {e}", path.display()),
            }
        }
    });
}

async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.tx.subscribe();

    // Send the current file content immediately on connect
    let initial = std::fs::read_to_string(&state.dsl_path).unwrap_or_default();
    let initial_event = futures_util::stream::once(async move {
        Ok(Event::default().data(initial))
    });

    let live = BroadcastStream::new(rx).filter_map(|res| async move {
        res.ok().map(|content| Ok(Event::default().data(content)))
    });

    let stream = initial_event.chain(live);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn dsl_handler(State(state): State<AppState>) -> String {
    std::fs::read_to_string(&state.dsl_path).unwrap_or_default()
}
