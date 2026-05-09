use std::{collections::HashMap, path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use clap::Parser;
use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

#[derive(Parser)]
#[command(name = "turbo-tasks-viz-server")]
#[command(about = "HTTP server for visualizing turbo-tasks backend events")]
struct Cli {
    /// Path to the SQLite viz database
    db_path: PathBuf,

    /// Port to listen on
    #[arg(long, default_value = "5748")]
    port: u16,
}

struct AppState {
    conn: Mutex<Connection>,
}

// --- Response types ---

#[derive(Serialize)]
struct SummaryResponse {
    total_tasks: i64,
    total_events: i64,
    total_edges: i64,
    min_timestamp_us: Option<i64>,
    max_timestamp_us: Option<i64>,
}

#[derive(Serialize)]
struct TaskRow {
    task_id: i64,
    name: String,
    is_transient: bool,
    created_seq: i64,
    created_us: i64,
}

#[derive(Serialize)]
struct TaskDetail {
    task_id: i64,
    name: String,
    is_transient: bool,
    created_seq: i64,
    created_us: i64,
    event_count: i64,
    incoming_edge_count: i64,
    outgoing_edge_count: i64,
}

#[derive(Serialize)]
struct EventRow {
    seq: i64,
    timestamp_us: i64,
    kind: i64,
    task_id: i64,
    data: Option<String>,
}

#[derive(Serialize)]
struct EdgeRow {
    id: i64,
    source_task: i64,
    target_task: i64,
    edge_type: i64,
    created_seq: i64,
    removed_seq: Option<i64>,
}

#[derive(Serialize)]
struct TaskStateRow {
    task_id: i64,
    state: i64,
    seq: i64,
    timestamp_us: i64,
}

#[derive(Serialize)]
struct GraphResponse {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

#[derive(Serialize)]
struct GraphNode {
    id: i64,
    name: String,
    is_transient: bool,
}

#[derive(Serialize)]
struct GraphEdge {
    source: i64,
    target: i64,
    edge_type: i64,
}

#[derive(Serialize)]
struct HistogramBucket {
    bucket_start_us: i64,
    count: i64,
}

#[derive(Serialize)]
struct TaskTypeStats {
    name: String,
    task_count: i64,
    execution_count: i64,
}

#[derive(Serialize)]
struct TaskSearchResult {
    task_id: i64,
    name: String,
    is_transient: bool,
}

/// An execution span: a paired TaskStarted → TaskCompleted.
#[derive(Serialize)]
struct Span {
    task_id: i64,
    name: String,
    start_us: i64,
    end_us: i64,
    stale: bool,
    /// Parent task via child edge (edge_type=0), if any.
    parent_task_id: Option<i64>,
}

#[derive(Serialize)]
struct TaskTypeGroup {
    name: String,
    task_count: i64,
    execution_count: i64,
    total_duration_us: i64,
    example_task_id: i64,
    first_execution_us: Option<i64>,
    last_execution_us: Option<i64>,
}

#[derive(Serialize)]
struct StatsResponse {
    total_tasks: i64,
    total_events: i64,
    total_executions: i64,
    total_invalidations: i64,
    task_type_count: i64,
}

// --- Query params ---

#[derive(Deserialize)]
struct PaginationParams {
    limit: Option<i64>,
    offset: Option<i64>,
    sort: Option<String>,
    filter: Option<String>,
}

#[derive(Deserialize)]
struct TimelineParams {
    from_us: Option<i64>,
    to_us: Option<i64>,
    kind: Option<i64>,
    limit: Option<i64>,
    task_id: Option<i64>,
}

#[derive(Deserialize)]
struct GraphParams {
    root: Option<i64>,
    depth: Option<i64>,
    edge_types: Option<String>,
}

#[derive(Deserialize)]
struct HistogramParams {
    bucket_us: Option<i64>,
    from_us: Option<i64>,
    to_us: Option<i64>,
}

#[derive(Deserialize)]
struct EdgeParams {
    direction: Option<String>,
    #[serde(rename = "type")]
    edge_type: Option<String>,
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    limit: Option<i64>,
}

#[derive(Deserialize)]
struct SpansParams {
    from_us: Option<i64>,
    to_us: Option<i64>,
    limit: Option<i64>,
    name_filter: Option<String>,
}

type AppError = (StatusCode, String);

fn db_err(e: impl std::fmt::Display) -> AppError {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

// --- Handlers ---

async fn get_summary(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let total_tasks: i64 = conn
        .query_row("SELECT COUNT(*) FROM tasks", [], |r| r.get(0))
        .map_err(db_err)?;
    let total_events: i64 = conn
        .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))
        .map_err(db_err)?;
    let total_edges: i64 = conn
        .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))
        .map_err(db_err)?;
    let min_ts: Option<i64> = conn
        .query_row("SELECT MIN(timestamp_us) FROM events", [], |r| r.get(0))
        .map_err(db_err)?;
    let max_ts: Option<i64> = conn
        .query_row("SELECT MAX(timestamp_us) FROM events", [], |r| r.get(0))
        .map_err(db_err)?;

    Ok(Json(SummaryResponse {
        total_tasks,
        total_events,
        total_edges,
        min_timestamp_us: min_ts,
        max_timestamp_us: max_ts,
    }))
}

async fn get_tasks(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let limit = params.limit.unwrap_or(100).min(10000);
    let offset = params.offset.unwrap_or(0);

    let order = match params.sort.as_deref() {
        Some("name") => "name ASC",
        Some("created") => "created_us ASC",
        _ => "task_id ASC",
    };

    let (query, filter_params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
        if let Some(filter) = &params.filter {
            (
                format!(
                    "SELECT task_id, name, is_transient, created_seq, created_us FROM tasks \
                     WHERE name LIKE ?1 ORDER BY {order} LIMIT ?2 OFFSET ?3"
                ),
                vec![
                    Box::new(format!("%{filter}%")),
                    Box::new(limit),
                    Box::new(offset),
                ],
            )
        } else {
            (
                format!(
                    "SELECT task_id, name, is_transient, created_seq, created_us FROM tasks \
                     ORDER BY {order} LIMIT ?1 OFFSET ?2"
                ),
                vec![Box::new(limit), Box::new(offset)],
            )
        };

    let mut stmt = conn.prepare(&query).map_err(db_err)?;
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = filter_params.iter().map(|p| &**p).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok(TaskRow {
                task_id: row.get(0)?,
                name: row.get(1)?,
                is_transient: row.get::<_, i64>(2)? != 0,
                created_seq: row.get(3)?,
                created_us: row.get(4)?,
            })
        })
        .map_err(db_err)?;

    let tasks: Vec<TaskRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(tasks))
}

async fn get_task(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<i64>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;

    let task = conn
        .query_row(
            "SELECT task_id, name, is_transient, created_seq, created_us FROM tasks WHERE task_id = ?1",
            params![task_id],
            |row| {
                Ok(TaskRow {
                    task_id: row.get(0)?,
                    name: row.get(1)?,
                    is_transient: row.get::<_, i64>(2)? != 0,
                    created_seq: row.get(3)?,
                    created_us: row.get(4)?,
                })
            },
        )
        .map_err(db_err)?;

    let event_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE task_id = ?1",
            params![task_id],
            |r| r.get(0),
        )
        .map_err(db_err)?;
    let incoming_edge_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM edges WHERE target_task = ?1",
            params![task_id],
            |r| r.get(0),
        )
        .map_err(db_err)?;
    let outgoing_edge_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM edges WHERE source_task = ?1",
            params![task_id],
            |r| r.get(0),
        )
        .map_err(db_err)?;

    Ok(Json(TaskDetail {
        task_id: task.task_id,
        name: task.name,
        is_transient: task.is_transient,
        created_seq: task.created_seq,
        created_us: task.created_us,
        event_count,
        incoming_edge_count,
        outgoing_edge_count,
    }))
}

async fn get_task_timeline(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<i64>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let mut stmt = conn
        .prepare(
            "SELECT task_id, state, seq, timestamp_us FROM task_states \
             WHERE task_id = ?1 ORDER BY seq ASC",
        )
        .map_err(db_err)?;
    let rows = stmt
        .query_map(params![task_id], |row| {
            Ok(TaskStateRow {
                task_id: row.get(0)?,
                state: row.get(1)?,
                seq: row.get(2)?,
                timestamp_us: row.get(3)?,
            })
        })
        .map_err(db_err)?;
    let states: Vec<TaskStateRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(states))
}

async fn get_task_edges(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<i64>,
    Query(params): Query<EdgeParams>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;

    let direction = params.direction.as_deref().unwrap_or("out");
    let filter_col = match direction {
        "in" => "target_task",
        _ => "source_task",
    };

    let edge_type_filter = match params.edge_type.as_deref() {
        Some("child") => Some(0),
        Some("output_dep") => Some(1),
        Some("cell_dep") => Some(2),
        _ => None,
    };

    let (query, query_params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
        if let Some(et) = edge_type_filter {
            (
                format!(
                    "SELECT id, source_task, target_task, edge_type, created_seq, removed_seq \
                     FROM edges WHERE {filter_col} = ?1 AND edge_type = ?2 ORDER BY created_seq ASC"
                ),
                vec![Box::new(task_id), Box::new(et)],
            )
        } else {
            (
                format!(
                    "SELECT id, source_task, target_task, edge_type, created_seq, removed_seq \
                     FROM edges WHERE {filter_col} = ?1 ORDER BY created_seq ASC"
                ),
                vec![Box::new(task_id)],
            )
        };

    let mut stmt = conn.prepare(&query).map_err(db_err)?;
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = query_params.iter().map(|p| &**p).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok(EdgeRow {
                id: row.get(0)?,
                source_task: row.get(1)?,
                target_task: row.get(2)?,
                edge_type: row.get(3)?,
                created_seq: row.get(4)?,
                removed_seq: row.get(5)?,
            })
        })
        .map_err(db_err)?;
    let edges: Vec<EdgeRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(edges))
}

async fn get_timeline(
    State(state): State<Arc<AppState>>,
    Query(params): Query<TimelineParams>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let limit = params.limit.unwrap_or(1000).min(100000);

    let mut conditions = Vec::new();
    let mut query_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut param_idx = 1;

    if let Some(from) = params.from_us {
        conditions.push(format!("timestamp_us >= ?{param_idx}"));
        query_params.push(Box::new(from));
        param_idx += 1;
    }
    if let Some(to) = params.to_us {
        conditions.push(format!("timestamp_us <= ?{param_idx}"));
        query_params.push(Box::new(to));
        param_idx += 1;
    }
    if let Some(kind) = params.kind {
        conditions.push(format!("kind = ?{param_idx}"));
        query_params.push(Box::new(kind));
        param_idx += 1;
    }
    if let Some(task_id) = params.task_id {
        conditions.push(format!("task_id = ?{param_idx}"));
        query_params.push(Box::new(task_id));
        param_idx += 1;
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    let query = format!(
        "SELECT seq, timestamp_us, kind, task_id, data FROM events \
         {where_clause} ORDER BY seq ASC LIMIT ?{param_idx}"
    );
    query_params.push(Box::new(limit));

    let mut stmt = conn.prepare(&query).map_err(db_err)?;
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = query_params.iter().map(|p| &**p).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok(EventRow {
                seq: row.get(0)?,
                timestamp_us: row.get(1)?,
                kind: row.get(2)?,
                task_id: row.get(3)?,
                data: row.get(4)?,
            })
        })
        .map_err(db_err)?;
    let events: Vec<EventRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(events))
}

async fn get_graph(
    State(state): State<Arc<AppState>>,
    Query(params): Query<GraphParams>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let max_depth = params.depth.unwrap_or(3).min(10);

    let edge_types: Vec<i64> = params
        .edge_types
        .as_deref()
        .unwrap_or("0,1,2")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let Some(root) = params.root else {
        return Err((StatusCode::BAD_REQUEST, "root parameter required".into()));
    };

    // BFS from root
    let mut visited: HashMap<i64, GraphNode> = HashMap::new();
    let mut graph_edges: Vec<GraphEdge> = Vec::new();
    let mut frontier: Vec<i64> = vec![root];

    let edge_type_placeholders: String = edge_types.iter().map(|_| "?").collect::<Vec<_>>().join(",");

    for _depth in 0..max_depth {
        if frontier.is_empty() {
            break;
        }

        let mut next_frontier: Vec<i64> = Vec::new();

        for &node_id in &frontier {
            if visited.contains_key(&node_id) {
                continue;
            }

            // Get node info
            let node = conn.query_row(
                "SELECT task_id, name, is_transient FROM tasks WHERE task_id = ?1",
                params![node_id],
                |row| {
                    Ok(GraphNode {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        is_transient: row.get::<_, i64>(2)? != 0,
                    })
                },
            );
            if let Ok(node) = node {
                visited.insert(node_id, node);
            }

            // Get outgoing edges
            let query = format!(
                "SELECT source_task, target_task, edge_type FROM edges \
                 WHERE source_task = ?1 AND edge_type IN ({edge_type_placeholders})"
            );
            let mut stmt = conn.prepare(&query).map_err(db_err)?;
            let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(node_id)];
            for et in &edge_types {
                params_vec.push(Box::new(*et));
            }
            let params_refs: Vec<&dyn rusqlite::types::ToSql> =
                params_vec.iter().map(|p| &**p).collect();
            let rows = stmt
                .query_map(params_refs.as_slice(), |row| {
                    Ok(GraphEdge {
                        source: row.get(0)?,
                        target: row.get(1)?,
                        edge_type: row.get(2)?,
                    })
                })
                .map_err(db_err)?;
            for edge in rows.flatten() {
                if !visited.contains_key(&edge.target) {
                    next_frontier.push(edge.target);
                }
                graph_edges.push(edge);
            }
        }

        frontier = next_frontier;
    }

    // Add any remaining frontier nodes
    for &node_id in &frontier {
        if !visited.contains_key(&node_id) {
            let node = conn.query_row(
                "SELECT task_id, name, is_transient FROM tasks WHERE task_id = ?1",
                params![node_id],
                |row| {
                    Ok(GraphNode {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        is_transient: row.get::<_, i64>(2)? != 0,
                    })
                },
            );
            if let Ok(node) = node {
                visited.insert(node_id, node);
            }
        }
    }

    Ok(Json(GraphResponse {
        nodes: visited.into_values().collect(),
        edges: graph_edges,
    }))
}

async fn get_stats(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;

    let total_tasks: i64 = conn
        .query_row("SELECT COUNT(*) FROM tasks", [], |r| r.get(0))
        .map_err(db_err)?;
    let total_events: i64 = conn
        .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))
        .map_err(db_err)?;
    // kind=2 is TaskStarted
    let total_executions: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE kind = 2",
            [],
            |r| r.get(0),
        )
        .map_err(db_err)?;
    // kind=4 is TaskInvalidated
    let total_invalidations: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE kind = 4",
            [],
            |r| r.get(0),
        )
        .map_err(db_err)?;
    let task_type_count: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT name) FROM tasks",
            [],
            |r| r.get(0),
        )
        .map_err(db_err)?;

    Ok(Json(StatsResponse {
        total_tasks,
        total_events,
        total_executions,
        total_invalidations,
        task_type_count,
    }))
}

async fn get_timeline_histogram(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HistogramParams>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let bucket_us = params.bucket_us.unwrap_or(1_000_000); // default 1 second

    let mut conditions = Vec::new();
    let mut query_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut param_idx = 1;

    if let Some(from) = params.from_us {
        conditions.push(format!("timestamp_us >= ?{param_idx}"));
        query_params.push(Box::new(from));
        param_idx += 1;
    }
    if let Some(to) = params.to_us {
        conditions.push(format!("timestamp_us <= ?{param_idx}"));
        query_params.push(Box::new(to));
        param_idx += 1;
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    let query = format!(
        "SELECT (timestamp_us / ?{param_idx}) * ?{param_idx} AS bucket_start, COUNT(*) AS cnt \
         FROM events {where_clause} GROUP BY bucket_start ORDER BY bucket_start ASC"
    );
    query_params.push(Box::new(bucket_us));

    let mut stmt = conn.prepare(&query).map_err(db_err)?;
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = query_params.iter().map(|p| &**p).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok(HistogramBucket {
                bucket_start_us: row.get(0)?,
                count: row.get(1)?,
            })
        })
        .map_err(db_err)?;
    let buckets: Vec<HistogramBucket> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(buckets))
}

async fn get_stats_by_type(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    // kind=2 is TaskStarted
    let mut stmt = conn
        .prepare(
            "SELECT t.name, COUNT(DISTINCT t.task_id) AS task_count, \
             COUNT(e.seq) AS execution_count \
             FROM tasks t LEFT JOIN events e ON t.task_id = e.task_id AND e.kind = 2 \
             GROUP BY t.name ORDER BY execution_count DESC LIMIT 200",
        )
        .map_err(db_err)?;
    let rows = stmt
        .query_map([], |row| {
            Ok(TaskTypeStats {
                name: row.get(0)?,
                task_count: row.get(1)?,
                execution_count: row.get(2)?,
            })
        })
        .map_err(db_err)?;
    let stats: Vec<TaskTypeStats> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(stats))
}

async fn search_tasks(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let limit = params.limit.unwrap_or(20).min(100);
    let mut stmt = conn
        .prepare(
            "SELECT task_id, name, is_transient FROM tasks \
             WHERE name LIKE ?1 ORDER BY name ASC LIMIT ?2",
        )
        .map_err(db_err)?;
    let pattern = format!("%{}%", params.q);
    let rows = stmt
        .query_map(params![pattern, limit], |row| {
            Ok(TaskSearchResult {
                task_id: row.get(0)?,
                name: row.get(1)?,
                is_transient: row.get::<_, i64>(2)? != 0,
            })
        })
        .map_err(db_err)?;
    let results: Vec<TaskSearchResult> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(results))
}

async fn get_spans(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SpansParams>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let limit = params.limit.unwrap_or(10000).min(100000);

    // Build execution spans by pairing TaskStarted (kind=2) with TaskCompleted (kind=3)
    // We join against the tasks table to get the name, and use a self-join on task_states
    // to pair start→complete transitions.
    let mut conditions = vec!["s.state = 2".to_string()]; // Started
    let mut query_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut param_idx = 1;

    if let Some(from) = params.from_us {
        conditions.push(format!("s.timestamp_us >= ?{param_idx}"));
        query_params.push(Box::new(from));
        param_idx += 1;
    }
    if let Some(to) = params.to_us {
        conditions.push(format!("c.timestamp_us <= ?{param_idx}"));
        query_params.push(Box::new(to));
        param_idx += 1;
    }
    if let Some(ref name_filter) = params.name_filter {
        conditions.push(format!("t.name LIKE ?{param_idx}"));
        query_params.push(Box::new(format!("%{name_filter}%")));
        param_idx += 1;
    }

    let where_clause = format!("WHERE {}", conditions.join(" AND "));

    // For each TaskStarted state entry, find the next TaskCompleted for that same task
    // (the one with the smallest seq > start.seq and state = 3).
    // Also join with edges to find the parent task (edge_type=0 = child edge,
    // where source_task is the parent and target_task is the child).
    let query = format!(
        "SELECT s.task_id, t.name, s.timestamp_us AS start_us, \
         COALESCE(c.timestamp_us, s.timestamp_us + 1) AS end_us, \
         CASE WHEN ev.data LIKE '%\"stale\":true%' THEN 1 ELSE 0 END AS stale, \
         parent_edge.source_task AS parent_task_id \
         FROM task_states s \
         JOIN tasks t ON t.task_id = s.task_id \
         LEFT JOIN task_states c ON c.task_id = s.task_id AND c.state = 3 \
           AND c.seq = (SELECT MIN(c2.seq) FROM task_states c2 \
                        WHERE c2.task_id = s.task_id AND c2.state = 3 AND c2.seq > s.seq) \
         LEFT JOIN events ev ON ev.task_id = s.task_id AND ev.kind = 3 AND ev.seq = c.seq \
         LEFT JOIN edges parent_edge ON parent_edge.target_task = s.task_id \
           AND parent_edge.edge_type = 0 \
           AND parent_edge.id = (SELECT MIN(pe2.id) FROM edges pe2 \
                                  WHERE pe2.target_task = s.task_id AND pe2.edge_type = 0) \
         {where_clause} \
         ORDER BY s.timestamp_us ASC \
         LIMIT ?{param_idx}"
    );
    query_params.push(Box::new(limit));

    let mut stmt = conn.prepare(&query).map_err(db_err)?;
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = query_params.iter().map(|p| &**p).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok(Span {
                task_id: row.get(0)?,
                name: row.get(1)?,
                start_us: row.get(2)?,
                end_us: row.get(3)?,
                stale: row.get::<_, i64>(4)? != 0,
                parent_task_id: row.get(5)?,
            })
        })
        .map_err(db_err)?;
    let spans: Vec<Span> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(spans))
}

async fn get_task_type_groups(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<impl IntoResponse, AppError> {
    let conn = state.conn.lock().await;
    let limit = params.limit.unwrap_or(200).min(1000);
    let offset = params.offset.unwrap_or(0);

    let (filter_clause, query_params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
        if let Some(ref filter) = params.filter {
            (
                "WHERE t.name LIKE ?1".to_string(),
                vec![Box::new(format!("%{filter}%"))],
            )
        } else {
            (String::new(), vec![])
        };

    let order = match params.sort.as_deref() {
        Some("tasks") => "task_count DESC",
        Some("duration") => "total_duration_us DESC",
        Some("name") => "t.name ASC",
        _ => "execution_count DESC",
    };

    let start_param = query_params.len() + 1;
    let query = format!(
        "SELECT t.name, COUNT(DISTINCT t.task_id) AS task_count, \
         COUNT(s.seq) AS execution_count, \
         COALESCE(SUM(CASE WHEN c.timestamp_us IS NOT NULL THEN c.timestamp_us - s.timestamp_us ELSE 0 END), 0) AS total_duration_us, \
         MIN(t.task_id) AS example_task_id, \
         MIN(s.timestamp_us) AS first_execution_us, \
         MAX(COALESCE(c.timestamp_us, s.timestamp_us)) AS last_execution_us \
         FROM tasks t \
         LEFT JOIN task_states s ON s.task_id = t.task_id AND s.state = 2 \
         LEFT JOIN task_states c ON c.task_id = s.task_id AND c.state = 3 \
           AND c.seq = (SELECT MIN(c2.seq) FROM task_states c2 \
                        WHERE c2.task_id = s.task_id AND c2.state = 3 AND c2.seq > s.seq) \
         {filter_clause} \
         GROUP BY t.name \
         ORDER BY {order} \
         LIMIT ?{start_param} OFFSET ?{}",
        start_param + 1
    );

    let mut all_params = query_params;
    all_params.push(Box::new(limit));
    all_params.push(Box::new(offset));

    let mut stmt = conn.prepare(&query).map_err(db_err)?;
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = all_params.iter().map(|p| &**p).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok(TaskTypeGroup {
                name: row.get(0)?,
                task_count: row.get(1)?,
                execution_count: row.get(2)?,
                total_duration_us: row.get(3)?,
                example_task_id: row.get(4)?,
                first_execution_us: row.get(5)?,
                last_execution_us: row.get(6)?,
            })
        })
        .map_err(db_err)?;
    let groups: Vec<TaskTypeGroup> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(groups))
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let conn = Connection::open_with_flags(
        &cli.db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed to open database at {:?}: {}", cli.db_path, e);
        std::process::exit(1);
    });

    let state = Arc::new(AppState {
        conn: Mutex::new(conn),
    });

    let app = Router::new()
        .route("/api/summary", get(get_summary))
        .route("/api/tasks", get(get_tasks))
        .route("/api/task/{id}", get(get_task))
        .route("/api/task/{id}/timeline", get(get_task_timeline))
        .route("/api/task/{id}/edges", get(get_task_edges))
        .route("/api/timeline", get(get_timeline))
        .route("/api/graph", get(get_graph))
        .route("/api/stats", get(get_stats))
        .route("/api/timeline/histogram", get(get_timeline_histogram))
        .route("/api/stats/by_type", get(get_stats_by_type))
        .route("/api/tasks/search", get(search_tasks))
        .route("/api/spans", get(get_spans))
        .route("/api/task_types", get(get_task_type_groups))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    eprintln!(
        "turbo-tasks-viz-server listening on http://localhost:{}",
        cli.port
    );
    eprintln!("Database: {:?}", cli.db_path);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
