//! Website-visit analytics as a gatekeeper function.
//!
//! Queries the `PageView` entities in datalog-db (loaded from Axiom/Vercel by the
//! `axiom-reload` job) and returns aggregated JSON. Every endpoint accepts an
//! optional time window via query params:
//!
//! - `?days=N`   — last N days (from now)
//! - `?from=MS&to=MS` — explicit epoch-millisecond bounds (either may be omitted)
//!
//! Endpoints (path after the route prefix):
//!
//! - `/` or `/summary` — totals: views, unique IPs, distinct paths, time span
//! - `/by-day`         — daily view counts (bucketed)
//! - `/top-paths`      — most-visited paths        (`?n=` limit, default 20)
//! - `/top-referrers`  — top referrers
//! - `/by-status` `/by-region` `/by-source` `/by-method` `/by-host` — breakdowns
//! - `/recent`         — last N raw pageviews       (`?n=` default 50, max 500)
//!
//! The handler opens a fresh datalog-db connection per request (cheap, local
//! socket). Any datalog error is returned as a 502 JSON body, never a panic (and
//! the SDK would catch a panic anyway).

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use datalog_client::{Client, QueryResult};
use gatekeeper_fn::{handler, Request, Response};
use serde_json::{json, Value};

/// datalog-db address; overridable via env for non-default deployments.
fn datalog_addr() -> String {
    std::env::var("DATALOG_ADDR").unwrap_or_else(|_| "127.0.0.1:5557".to_string())
}

const DAY_MS: i64 = 86_400_000;

#[handler]
fn analytics(req: Request) -> Response {
    let params = parse_query(req.query());
    let window = Window::from_params(&params);

    let mut db = match Client::connect(&datalog_addr()) {
        Ok(c) => c,
        Err(e) => return err(502, &format!("datalog connect failed: {e}")),
    };

    // Route on the path after the gate prefix. Trailing slash tolerated.
    let path = req.path().trim_end_matches('/');
    let result = match path {
        "" | "/summary" => summary(&mut db, &window),
        "/by-day" => by_day(&mut db, &window),
        "/top-paths" => top_field(&mut db, &window, "path", limit(&params, 20)),
        "/top-referrers" => top_field(&mut db, &window, "referer", limit(&params, 20)),
        "/by-status" => top_field(&mut db, &window, "status", limit(&params, 50)),
        "/by-region" => top_field(&mut db, &window, "region", limit(&params, 50)),
        "/by-source" => top_field(&mut db, &window, "source", limit(&params, 50)),
        "/by-method" => top_field(&mut db, &window, "method", limit(&params, 20)),
        "/by-host" => top_field(&mut db, &window, "host", limit(&params, 50)),
        "/recent" => recent(&mut db, &window, limit(&params, 50).min(500)),
        other => return err(404, &format!("unknown analytics endpoint {other:?}")),
    };

    match result {
        Ok(body) => Response::json(body.to_string()),
        Err(e) => err(502, &format!("query failed: {e}")),
    }
}

/// A time window applied to every query as `ts >= from` / `ts <= to` filters.
struct Window {
    from: Option<i64>,
    to: Option<i64>,
}

impl Window {
    fn from_params(p: &BTreeMap<String, String>) -> Window {
        // ?days=N takes precedence as a convenience; else explicit from/to.
        if let Some(days) = p.get("days").and_then(|d| d.parse::<i64>().ok()) {
            let now = now_ms();
            return Window {
                from: Some(now - days * DAY_MS),
                to: None,
            };
        }
        Window {
            from: p.get("from").and_then(|s| s.parse().ok()),
            to: p.get("to").and_then(|s| s.parse().ok()),
        }
    }

    fn describe(&self) -> Value {
        json!({"from": self.from, "to": self.to})
    }
}

/// Build a `where` array for a PageView query.
///
/// The window bounds are applied as field-level predicates on `ts`. A single
/// clause field carries only one predicate, so when BOTH bounds are present we
/// emit two PageView clauses over the same entity `?e` (datalog ANDs clauses
/// sharing a variable): one with `ts >= from`, one with `ts <= to`. With one or
/// no bound, a single clause suffices. `extra` field bindings (e.g.
/// `{"path": "?p"}`) are merged onto the first clause so projections still work.
fn where_pageview(window: &Window, extra: Value) -> Vec<Value> {
    let base = |ts_pat: Value, with_extra: bool| -> Value {
        let mut clause = json!({"bind": "?e", "type": "PageView"});
        clause["ts"] = ts_pat;
        if with_extra {
            if let Some(obj) = extra.as_object() {
                for (k, v) in obj {
                    clause[k] = v.clone();
                }
            }
        }
        clause
    };

    match (window.from, window.to) {
        (Some(from), Some(to)) => vec![
            base(json!({"var": "?ts", "gte": from}), true),
            base(json!({"lte": to}), false),
        ],
        (Some(from), None) => vec![base(json!({"var": "?ts", "gte": from}), true)],
        (None, Some(to)) => vec![base(json!({"var": "?ts", "lte": to}), true)],
        // No window: bind ts so callers that aggregate on ?ts still have it.
        (None, None) => vec![base(json!("?ts"), true)],
    }
}

/// `/summary` — headline numbers for the window.
fn summary(db: &mut Client, window: &Window) -> Result<Value, String> {
    let res = db
        .query(json!({
            "find": [
                {"agg": "count", "var": "*"},
                {"agg": "count", "var": "?ts"},
                {"agg": "min", "var": "?ts"},
                {"agg": "max", "var": "?ts"},
            ],
            "where": where_pageview(window, json!({})),
        }))
        .map_err(|e| e.to_string())?;

    // Unique IPs need a distinct count over ?ip (separate query so the IP bind
    // doesn't drop rows with no ip).
    let uniq_ip = db
        .query(json!({
            "find": [{"agg": "count", "var": "distinct ?ip"}],
            "where": where_pageview(window, json!({"ip": "?ip"})),
        }))
        .map_err(|e| e.to_string())?;
    let uniq_path = db
        .query(json!({
            "find": [{"agg": "count", "var": "distinct ?p"}],
            "where": where_pageview(window, json!({"path": "?p"})),
        }))
        .map_err(|e| e.to_string())?;

    let row = res.rows.first().cloned().unwrap_or_default();
    Ok(json!({
        "window": window.describe(),
        "total_views": row.first().cloned().unwrap_or(json!(0)),
        "unique_ips": scalar_or_zero(&uniq_ip),
        "distinct_paths": scalar_or_zero(&uniq_path),
        "earliest_ts": row.get(2).cloned().unwrap_or(Value::Null),
        "latest_ts": row.get(3).cloned().unwrap_or(Value::Null),
    }))
}

/// `/by-day` — view counts grouped per UTC day (bucketed on ts).
fn by_day(db: &mut Client, window: &Window) -> Result<Value, String> {
    let res = db
        .query(json!({
            "find": [
                {"agg": "bucket", "var": format!("?ts, {DAY_MS}")},
                {"agg": "count", "var": "*"},
            ],
            "where": where_pageview(window, json!({})),
            "order_by": [{"var": format!("bucket(?ts, {DAY_MS})")}],
        }))
        .map_err(|e| e.to_string())?;

    // Each bucket key is the epoch-day-start in ms; expose ms + ISO date.
    let days: Vec<Value> = res
        .rows
        .iter()
        .map(|r| {
            let bucket_ms = r.first().and_then(|v| v.as_i64()).unwrap_or(0);
            json!({
                "day_start_ms": bucket_ms,
                "date": iso_date(bucket_ms),
                "views": r.get(1).cloned().unwrap_or(json!(0)),
            })
        })
        .collect();
    Ok(json!({"window": window.describe(), "by_day": days}))
}

/// Generic "top values of FIELD by count", e.g. top paths/referrers/regions.
fn top_field(
    db: &mut Client,
    window: &Window,
    field: &str,
    n: usize,
) -> Result<Value, String> {
    let res = db
        .query(json!({
            "find": ["?v", {"agg": "count", "var": "*"}],
            "where": where_pageview(window, json!({field: "?v"})),
            "order_by": [{"var": "count(*)", "desc": true}],
            "limit": n,
        }))
        .map_err(|e| e.to_string())?;

    let items: Vec<Value> = res
        .rows
        .iter()
        .map(|r| {
            json!({
                "value": r.first().cloned().unwrap_or(Value::Null),
                "views": r.get(1).cloned().unwrap_or(json!(0)),
            })
        })
        .collect();
    Ok(json!({"window": window.describe(), "field": field, "top": items}))
}

/// `/recent` — the last N raw pageviews, newest first.
fn recent(db: &mut Client, window: &Window, n: usize) -> Result<Value, String> {
    // Bind the fields we want to surface. Missing optional fields would drop a
    // row if bound positionally, so we bind only ts + the common ones and let
    // the engine return nulls where absent.
    let res = db
        .query(json!({
            "find": ["?ts", "?path", "?referer", "?region", "?status"],
            "where": where_pageview(
                window,
                json!({"path": "?path", "referer": "?referer", "region": "?region", "status": "?status"}),
            ),
            "order_by": [{"var": "?ts", "desc": true}],
            "limit": n,
        }))
        .map_err(|e| e.to_string())?;

    let items: Vec<Value> = res
        .rows
        .iter()
        .map(|r| {
            json!({
                "ts": r.first().cloned().unwrap_or(Value::Null),
                "path": r.get(1).cloned().unwrap_or(Value::Null),
                "referer": r.get(2).cloned().unwrap_or(Value::Null),
                "region": r.get(3).cloned().unwrap_or(Value::Null),
                "status": r.get(4).cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    Ok(json!({"window": window.describe(), "count": items.len(), "recent": items}))
}

// ---- helpers ----

fn parse_query(q: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for pair in q.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        m.insert(k.to_string(), percent_decode(v));
    }
    m
}

/// Minimal percent-decoding for query values (the gate already rejected path
/// traversal; this is just for `from`/`to` style values and referrers).
fn percent_decode(s: &str) -> String {
    let bytes = s.replace('+', " ");
    let bytes = bytes.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn limit(p: &BTreeMap<String, String>, default: usize) -> usize {
    p.get("n")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn scalar_or_zero(res: &QueryResult) -> Value {
    res.scalar().cloned().unwrap_or(json!(0))
}

fn err(status: u16, msg: &str) -> Response {
    Response::new(status, json!({"error": msg}).to_string().into_bytes())
        .header("Content-Type", "application/json")
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// epoch-ms -> `YYYY-MM-DD` (UTC). Inline civil-date math (no chrono dep).
fn iso_date(ms: i64) -> String {
    let days = ms.div_euclid(DAY_MS);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}
