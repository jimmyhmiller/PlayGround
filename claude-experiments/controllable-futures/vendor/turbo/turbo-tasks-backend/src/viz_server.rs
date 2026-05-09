//! Embedded HTTP server for the visualizer control center.
//!
//! Provides endpoints for:
//! - Controlling task scheduling (pause/resume/step/breakpoints)
//! - Live inspection of task state (cells, children, dependencies)

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::Method,
    routing::{delete, get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

use crate::viz::{
    ActiveTask, CellDetail, CellInfo, DebugEvent, TaskDepsInfo, TaskGraph, TaskStateInfo,
    VizBackendAccess, VizController,
};

/// Shared state for the HTTP server.
pub(crate) struct ServerState {
    pub controller: Arc<VizController>,
    pub backend_access: Arc<dyn VizBackendAccess>,
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct StatusResponse {
    paused: bool,
    pending_count: usize,
    breakpoints: Vec<BreakpointInfo>,
}

#[derive(Serialize)]
struct BreakpointInfo {
    id: u64,
    pattern: String,
    enabled: bool,
}

#[derive(Deserialize)]
struct StepRequest {
    count: Option<usize>,
}

#[derive(Deserialize)]
struct StepTaskRequest {
    task_id: u64,
}

#[derive(Serialize)]
struct StepResponse {
    released: usize,
}

#[derive(Serialize)]
struct PendingTaskInfo {
    task_id: u64,
    name: String,
    hit_breakpoint: Option<u64>,
}

#[derive(Deserialize)]
struct AddBreakpointRequest {
    pattern: String,
}

#[derive(Serialize)]
struct AddBreakpointResponse {
    id: u64,
}

#[derive(Deserialize)]
struct ToggleBreakpointRequest {
    enabled: bool,
}

#[derive(Serialize)]
struct TaskInfo {
    task_id: u64,
    name: String,
}

#[derive(Serialize)]
struct ChildInfo {
    task_id: u64,
    name: String,
}

#[derive(Serialize)]
struct OkResponse {
    ok: bool,
}

// ---------------------------------------------------------------------------
// Control endpoints
// ---------------------------------------------------------------------------

async fn get_status(State(state): State<Arc<ServerState>>) -> Json<StatusResponse> {
    let debugger = &state.controller.debugger;
    let bps = debugger.list_breakpoints();
    Json(StatusResponse {
        paused: debugger.is_paused(),
        pending_count: debugger.pending_count(),
        breakpoints: bps
            .into_iter()
            .map(|(id, pattern, enabled)| BreakpointInfo {
                id,
                pattern,
                enabled,
            })
            .collect(),
    })
}

async fn post_pause(State(state): State<Arc<ServerState>>) -> Json<OkResponse> {
    state.controller.debugger.set_paused(true);
    Json(OkResponse { ok: true })
}

async fn post_resume(State(state): State<Arc<ServerState>>) -> Json<OkResponse> {
    state.controller.debugger.resume();
    Json(OkResponse { ok: true })
}

async fn post_step(
    State(state): State<Arc<ServerState>>,
    Json(body): Json<StepRequest>,
) -> Json<StepResponse> {
    let count = body.count.unwrap_or(1);
    let released = state.controller.debugger.release_count(count);
    Json(StepResponse { released })
}

async fn post_step_task(
    State(state): State<Arc<ServerState>>,
    Json(body): Json<StepTaskRequest>,
) -> Json<OkResponse> {
    let ok = state.controller.debugger.release_specific(body.task_id);
    Json(OkResponse { ok })
}

async fn get_pending(State(state): State<Arc<ServerState>>) -> Json<Vec<PendingTaskInfo>> {
    let pending = state.controller.debugger.pending_list();
    Json(
        pending
            .into_iter()
            .map(|(task_id, name, hit_breakpoint)| PendingTaskInfo {
                task_id,
                name,
                hit_breakpoint,
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Step to idle endpoint
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct StepToIdleRequest {
    max: Option<usize>,
}

async fn post_step_to_idle(
    State(state): State<Arc<ServerState>>,
    Json(body): Json<StepToIdleRequest>,
) -> Json<StepResponse> {
    let max = body.max.unwrap_or(100);
    let released = state.controller.debugger.release_until_idle(max);
    Json(StepResponse { released })
}

// ---------------------------------------------------------------------------
// Active tasks endpoint
// ---------------------------------------------------------------------------

async fn get_active_tasks(State(state): State<Arc<ServerState>>) -> Json<Vec<ActiveTask>> {
    Json(state.controller.debugger.active_task_list())
}

// ---------------------------------------------------------------------------
// Breakpoint endpoints
// ---------------------------------------------------------------------------

async fn get_breakpoints(State(state): State<Arc<ServerState>>) -> Json<Vec<BreakpointInfo>> {
    let bps = state.controller.debugger.list_breakpoints();
    Json(
        bps.into_iter()
            .map(|(id, pattern, enabled)| BreakpointInfo {
                id,
                pattern,
                enabled,
            })
            .collect(),
    )
}

async fn post_breakpoint(
    State(state): State<Arc<ServerState>>,
    Json(body): Json<AddBreakpointRequest>,
) -> Json<AddBreakpointResponse> {
    let id = state.controller.debugger.add_breakpoint(body.pattern);
    Json(AddBreakpointResponse { id })
}

async fn delete_breakpoint(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<u64>,
) -> Json<OkResponse> {
    let ok = state.controller.debugger.remove_breakpoint(id);
    Json(OkResponse { ok })
}

async fn post_toggle_breakpoint(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<u64>,
    Json(body): Json<ToggleBreakpointRequest>,
) -> Json<OkResponse> {
    let ok = state
        .controller
        .debugger
        .toggle_breakpoint(id, body.enabled);
    Json(OkResponse { ok })
}

// ---------------------------------------------------------------------------
// Live inspection endpoints
// ---------------------------------------------------------------------------

async fn get_task(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<u64>,
) -> Json<Option<TaskInfo>> {
    let name = state.backend_access.get_task_description(id);
    Json(name.map(|name| TaskInfo { task_id: id, name }))
}

async fn get_task_cells(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<u64>,
) -> Json<Vec<CellInfo>> {
    Json(state.backend_access.list_task_cells(id))
}

async fn get_task_children(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<u64>,
) -> Json<Vec<ChildInfo>> {
    let children = state.backend_access.list_task_children(id);
    Json(
        children
            .into_iter()
            .map(|(task_id, name)| ChildInfo { task_id, name })
            .collect(),
    )
}

async fn get_task_deps(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<u64>,
) -> Json<TaskDepsInfo> {
    Json(state.backend_access.list_task_dependencies(id))
}

// ---------------------------------------------------------------------------
// Task graph endpoint
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GraphQuery {
    depth: Option<usize>,
}

async fn get_task_graph(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<u64>,
    Query(query): Query<GraphQuery>,
) -> Json<TaskGraph> {
    let depth = query.depth.unwrap_or(2);
    Json(state.backend_access.get_task_graph(id, depth))
}

// ---------------------------------------------------------------------------
// Task state info endpoint
// ---------------------------------------------------------------------------

async fn get_task_state_info(
    State(state): State<Arc<ServerState>>,
    Path(id): Path<u64>,
) -> Json<Option<TaskStateInfo>> {
    Json(state.backend_access.get_task_state_info(id))
}

// ---------------------------------------------------------------------------
// Cell detail endpoint
// ---------------------------------------------------------------------------

async fn get_cell_detail(
    State(state): State<Arc<ServerState>>,
    Path((id, cell_index)): Path<(u64, u32)>,
) -> Json<Option<CellDetail>> {
    Json(state.backend_access.get_cell_detail(id, cell_index))
}

// ---------------------------------------------------------------------------
// Event log endpoints
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EventsQuery {
    since: Option<u64>,
}

async fn get_events(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<EventsQuery>,
) -> Json<Vec<DebugEvent>> {
    let since = query.since.unwrap_or(0);
    Json(state.controller.debugger.events_since(since))
}

// ---------------------------------------------------------------------------
// Task search endpoint
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct SearchResult {
    task_id: u64,
    name: String,
}

async fn get_search(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<SearchQuery>,
) -> Json<Vec<SearchResult>> {
    let limit = query.limit.unwrap_or(10);
    let results = state.backend_access.search_tasks(&query.q, limit);
    Json(
        results
            .into_iter()
            .map(|(task_id, name)| SearchResult { task_id, name })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Router and server startup
// ---------------------------------------------------------------------------

pub(crate) fn build_router(state: Arc<ServerState>) -> Router {
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_origin(Any)
        .allow_headers(Any);

    Router::new()
        // Control endpoints
        .route("/api/control/status", get(get_status))
        .route("/api/control/pause", post(post_pause))
        .route("/api/control/resume", post(post_resume))
        .route("/api/control/step", post(post_step))
        .route("/api/control/step-task", post(post_step_task))
        .route("/api/control/step-to-idle", post(post_step_to_idle))
        .route("/api/control/pending", get(get_pending))
        .route("/api/control/active-tasks", get(get_active_tasks))
        // Breakpoint endpoints
        .route("/api/control/breakpoints", get(get_breakpoints))
        .route("/api/control/breakpoints", post(post_breakpoint))
        .route("/api/control/breakpoints/:id", delete(delete_breakpoint))
        .route(
            "/api/control/breakpoints/:id/toggle",
            post(post_toggle_breakpoint),
        )
        // Event log endpoint
        .route("/api/control/events", get(get_events))
        // Live inspection endpoints
        .route("/api/live/task/:id", get(get_task))
        .route("/api/live/task/:id/cells", get(get_task_cells))
        .route("/api/live/task/:id/children", get(get_task_children))
        .route("/api/live/task/:id/deps", get(get_task_deps))
        .route("/api/live/task/:id/graph", get(get_task_graph))
        .route("/api/live/task/:id/state", get(get_task_state_info))
        .route(
            "/api/live/task/:id/cells/:cell_index",
            get(get_cell_detail),
        )
        .route("/api/live/search", get(get_search))
        .layer(cors)
        .with_state(state)
}

/// Start the embedded viz server on a background thread with its own tokio runtime.
pub(crate) fn spawn_viz_server(
    controller: Arc<VizController>,
    backend_access: Arc<dyn VizBackendAccess>,
) {
    let port: u16 = std::env::var("TURBO_TASKS_VIZ_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3311);

    std::thread::Builder::new()
        .name("viz-server".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create viz-server tokio runtime");

            rt.block_on(async move {
                let state = Arc::new(ServerState {
                    controller,
                    backend_access,
                });
                let app = build_router(state);
                let listener = match tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!(
                            "[turbo-tasks-viz] Failed to bind viz server on port {port}: {e}"
                        );
                        return;
                    }
                };
                eprintln!(
                    "[turbo-tasks-viz] Control center server listening on http://0.0.0.0:{port}"
                );
                if let Err(e) = axum::serve(listener, app).await {
                    eprintln!("[turbo-tasks-viz] Server error: {e}");
                }
            });
        })
        .expect("Failed to spawn viz-server thread");
}
