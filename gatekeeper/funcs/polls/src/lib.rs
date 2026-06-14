//! Slack `/poll` app analytics as a gatekeeper function.
//!
//! Queries the `Poll` and `PollVote` entities in datalog-db (loaded from the
//! Axiom `vercel` dataset by the `axiom-reload` job — see that crate's `poll`
//! module) and returns aggregated JSON. This is the poll counterpart to the
//! `analytics` function, which does the same for `PageView`s.
//!
//! Data model (both keyed by `callback_id`):
//! - `Poll`     — one per poll creation (`INSERT INTO poll` log line): question,
//!                option_count, anonymous, team_id, ts.
//! - `PollVote` — one per vote (`UPDATE poll` log line). Each is a SNAPSHOT of
//!                the whole poll AFTER that vote, so `total_votes` is cumulative
//!                and one row == one vote event. The max `total_votes` for a
//!                `callback_id` is that poll's peak participation.
//!
//! Every endpoint accepts an optional time window:
//! - `?days=N`        — last N days (from now)
//! - `?from=MS&to=MS` — explicit epoch-millisecond bounds (either may be omitted)
//!
//! Endpoints (path after the route prefix, e.g. `/polls`):
//! - `/` or `/summary`  — totals: polls created, votes cast, distinct teams,
//!                        option/vote stats, time span
//! - `/by-day`          — polls created per UTC day
//! - `/votes-by-day`    — vote events per UTC day
//! - `/top-teams`       — teams ranked by polls created (`?n=`)
//! - `/top-polls`       — polls ranked by peak vote count (`?n=`)
//! - `/questions`       — recent poll questions + their peak votes (`?n=`)
//! - `/timeline`        — polls-created (and votes) series over time, for graphs
//!                        (`?bucket=day|hour`)
//! - `/recent`          — last N polls created (`?n=`)
//!
//! A fresh datalog-db connection is opened per request. Any datalog error is a
//! 502 JSON body, never a panic.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use datalog_client::{Client, QueryResult};
use gatekeeper_fn::{describe, handler, Description, Endpoint, Param, Request, Response};
use serde_json::{json, Value};

/// Self-description for the gate's `/describe` catalog. Paths are relative to
/// the route prefix the gate mounts this under (e.g. `/polls`).
#[describe]
fn describe() -> Description {
    let window = |e: Endpoint| -> Endpoint {
        e.param(Param::int("days", "window: only the last N days").default("(all time)"))
            .param(Param::new("from", "epoch-ms", "window start (inclusive)"))
            .param(Param::new("to", "epoch-ms", "window end (inclusive)"))
    };

    Description::new("polls", "Slack /poll app analytics from datalog-db Poll/PollVote")
        .endpoint(window(
            Endpoint::get("/summary", "headline totals for the window")
                .example("/summary?days=7")
                .returns(
                    "{ polls_created, votes_cast, distinct_teams, total_options, \
                     avg_options, earliest_ts, latest_ts }",
                ),
        ))
        .endpoint(window(
            Endpoint::get("/by-day", "polls created per UTC day")
                .example("/by-day?days=30")
                .returns("{ by_day: [{ date, day_start_ms, polls }] }"),
        ))
        .endpoint(window(
            Endpoint::get("/votes-by-day", "vote events per UTC day")
                .returns("{ by_day: [{ date, day_start_ms, votes }] }"),
        ))
        .endpoint(window(
            Endpoint::get("/top-teams", "teams ranked by polls created")
                .param(Param::int("n", "how many").default("20"))
                .returns("{ top: [{ team_id, polls }] }"),
        ))
        .endpoint(window(
            Endpoint::get("/top-polls", "polls ranked by current vote count")
                .param(Param::int("n", "how many").default("20"))
                .returns("{ top: [{ callback_id, question, votes, options }] }"),
        ))
        .endpoint(window(
            Endpoint::get("/questions", "recent poll questions with current votes")
                .param(Param::int("n", "how many").default("50"))
                .returns("{ count, questions: [{ ts, question, options, votes, anonymous, callback_id }] }"),
        ))
        .endpoint(window(
            Endpoint::get("/timeline", "polls-created & votes series over time, for graphs")
                .param(Param::string("bucket", "granularity: 'day' or 'hour'").default("day"))
                .example("/timeline?days=14&bucket=day")
                .returns(
                    "{ bucket_ms, buckets: [{date, bucket_start_ms}], \
                     polls: [{date, count}], votes: [{date, count}] }",
                ),
        ))
        .endpoint(window(
            Endpoint::get("/recent", "the last N polls created, newest first")
                .param(Param::int("n", "how many (max 500)").default("50"))
                .returns(
                    "{ count, recent: [{ ts, question, options, anonymous, team_id, callback_id }] }",
                ),
        ))
}

fn datalog_addr() -> String {
    std::env::var("DATALOG_ADDR").unwrap_or_else(|_| "127.0.0.1:5557".to_string())
}

const DAY_MS: i64 = 86_400_000;

#[handler]
fn polls(req: Request) -> Response {
    let params = parse_query(req.query());
    let window = Window::from_params(&params);

    let mut db = match Client::connect(&datalog_addr()) {
        Ok(c) => c,
        Err(e) => return err(502, &format!("datalog connect failed: {e}")),
    };

    let path = req.path().trim_end_matches('/');
    let result = match path {
        "" | "/summary" => summary(&mut db, &window),
        "/by-day" => created_by_day(&mut db, &window),
        "/votes-by-day" => votes_by_day(&mut db, &window),
        "/top-teams" => top_teams(&mut db, &window, limit(&params, 20)),
        "/top-polls" => top_polls(&mut db, &window, limit(&params, 20)),
        "/questions" => questions(&mut db, &window, limit(&params, 50)),
        "/timeline" => timeline(&mut db, &window, &params),
        "/recent" => recent(&mut db, &window, limit(&params, 50).min(500)),
        other => return err(404, &format!("unknown polls endpoint {other:?}")),
    };

    match result {
        Ok(body) => Response::json(body.to_string()),
        Err(e) => err(502, &format!("query failed: {e}")),
    }
}

/// A time window applied to every query as `ts >= from` / `ts <= to`.
struct Window {
    from: Option<i64>,
    to: Option<i64>,
}

impl Window {
    fn from_params(p: &BTreeMap<String, String>) -> Window {
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

/// Build a `where` array for a query over `entity_type` (`Poll` or `PollVote`),
/// applying the window as `ts` predicates. When both bounds are present we emit
/// two clauses over the same `?e` (datalog ANDs clauses sharing a variable),
/// since one clause field carries only one predicate. `extra` field bindings are
/// merged onto the first clause so projections still work. Mirrors the analytics
/// function's `where_pageview`.
fn where_type(entity_type: &str, window: &Window, extra: Value) -> Vec<Value> {
    let base = |ts_pat: Value, with_extra: bool| -> Value {
        let mut clause = json!({"bind": "?e", "type": entity_type});
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
        (None, None) => vec![base(json!("?ts"), true)],
    }
}

/// `/summary` — headline numbers for the window.
fn summary(db: &mut Client, window: &Window) -> Result<Value, String> {
    // Poll-creation stats: count, option totals, time span.
    let polls = db
        .query(json!({
            "find": [
                {"agg": "count", "var": "*"},
                {"agg": "sum", "var": "?opts"},
                {"agg": "min", "var": "?ts"},
                {"agg": "max", "var": "?ts"},
            ],
            "where": where_type("Poll", window, json!({"option_count": "?opts"})),
        }))
        .map_err(|e| e.to_string())?;
    let row = polls.rows.first().cloned().unwrap_or_default();
    let polls_created = row.first().and_then(|v| v.as_i64()).unwrap_or(0);
    let total_options = row.get(1).and_then(|v| v.as_i64()).unwrap_or(0);

    // Distinct teams that created a poll in the window.
    let teams = db
        .query(json!({
            "find": [{"agg": "count", "var": "distinct ?t"}],
            "where": where_type("Poll", window, json!({"team_id": "?t"})),
        }))
        .map_err(|e| e.to_string())?;

    // Vote events in the window (one PollVote row == one vote).
    let votes = db
        .query(json!({
            "find": [{"agg": "count", "var": "*"}],
            "where": where_type("PollVote", window, json!({})),
        }))
        .map_err(|e| e.to_string())?;

    let avg_options = if polls_created > 0 {
        json!((total_options as f64) / (polls_created as f64))
    } else {
        json!(0)
    };

    Ok(json!({
        "window": window.describe(),
        "polls_created": polls_created,
        "votes_cast": scalar_or_zero(&votes),
        "distinct_teams": scalar_or_zero(&teams),
        "total_options": total_options,
        "avg_options": avg_options,
        "earliest_ts": row.get(2).cloned().unwrap_or(Value::Null),
        "latest_ts": row.get(3).cloned().unwrap_or(Value::Null),
    }))
}

/// `/by-day` — polls created per UTC day (bucketed on ts).
fn created_by_day(db: &mut Client, window: &Window) -> Result<Value, String> {
    bucketed_count(db, window, "Poll", DAY_MS, "polls")
}

/// `/votes-by-day` — vote events per UTC day.
fn votes_by_day(db: &mut Client, window: &Window) -> Result<Value, String> {
    bucketed_count(db, window, "PollVote", DAY_MS, "votes")
}

/// Shared helper: count entities of `entity_type` per `bucket_ms`, returning
/// `{ by_day: [{ day_start_ms, date, <count_key> }] }`.
fn bucketed_count(
    db: &mut Client,
    window: &Window,
    entity_type: &str,
    bucket_ms: i64,
    count_key: &str,
) -> Result<Value, String> {
    let res = db
        .query(json!({
            "find": [
                {"agg": "bucket", "var": format!("?ts, {bucket_ms}")},
                {"agg": "count", "var": "*"},
            ],
            "where": where_type(entity_type, window, json!({})),
            "order_by": [{"var": format!("bucket(?ts, {bucket_ms})")}],
        }))
        .map_err(|e| e.to_string())?;

    let days: Vec<Value> = res
        .rows
        .iter()
        .map(|r| {
            let bucket_ms = r.first().and_then(|v| v.as_i64()).unwrap_or(0);
            json!({
                "day_start_ms": bucket_ms,
                "date": iso_date(bucket_ms),
                count_key: r.get(1).cloned().unwrap_or(json!(0)),
            })
        })
        .collect();
    Ok(json!({"window": window.describe(), "by_day": days}))
}

/// `/top-teams` — teams ranked by number of polls created.
fn top_teams(db: &mut Client, window: &Window, n: usize) -> Result<Value, String> {
    let res = db
        .query(json!({
            "find": ["?t", {"agg": "count", "var": "*"}],
            "where": where_type("Poll", window, json!({"team_id": "?t"})),
            "order_by": [{"var": "count(*)", "desc": true}],
            "limit": n,
        }))
        .map_err(|e| e.to_string())?;

    let items: Vec<Value> = res
        .rows
        .iter()
        .map(|r| {
            json!({
                "team_id": r.first().cloned().unwrap_or(Value::Null),
                "polls": r.get(1).cloned().unwrap_or(json!(0)),
            })
        })
        .collect();
    Ok(json!({"window": window.describe(), "top": items}))
}

/// `/top-polls` — polls ranked by CURRENT vote count. The loader upserts each
/// poll's `total_votes`/`last_vote_ts` to the latest vote snapshot as it ingests
/// the `UPDATE poll` log lines, so `Poll.total_votes` already holds the true
/// current tally — no peak/approximation, no per-poll subquery. This is the SQL
/// `SELECT ... ORDER BY total_votes DESC LIMIT n` it reads like.
///
/// The window is applied to the poll's CREATION time (`ts`): "top polls created
/// in the last N days". (Use no window for the all-time leaderboard.)
fn top_polls(db: &mut Client, window: &Window, n: usize) -> Result<Value, String> {
    let res = db
        .query(json!({
            "find": ["?cb", "?q", "?tv", "?opts"],
            "where": where_type(
                "Poll",
                window,
                json!({"callback_id": "?cb", "question": "?q", "total_votes": "?tv", "option_count": "?opts"}),
            ),
            "order_by": [{"var": "?tv", "desc": true}],
            "limit": n,
        }))
        .map_err(|e| e.to_string())?;

    let items: Vec<Value> = res
        .rows
        .iter()
        .map(|r| {
            json!({
                "callback_id": r.first().cloned().unwrap_or(Value::Null),
                "question": r.get(1).cloned().unwrap_or(Value::Null),
                "votes": r.get(2).cloned().unwrap_or(json!(0)),
                "options": r.get(3).cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    Ok(json!({"window": window.describe(), "top": items}))
}

/// `/questions` — recent poll questions with their option count, anonymity, and
/// current vote count, newest first. `total_votes` is the live tally the loader
/// maintains on the Poll, so this needs no per-poll subquery.
fn questions(db: &mut Client, window: &Window, n: usize) -> Result<Value, String> {
    let res = db
        .query(json!({
            "find": ["?ts", "?q", "?opts", "?votes", "?anon", "?cb"],
            "where": where_type(
                "Poll",
                window,
                json!({"question": "?q", "option_count": "?opts", "total_votes": "?votes", "anonymous": "?anon", "callback_id": "?cb"}),
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
                "question": r.get(1).cloned().unwrap_or(Value::Null),
                "options": r.get(2).cloned().unwrap_or(Value::Null),
                "votes": r.get(3).cloned().unwrap_or(json!(0)),
                "anonymous": r.get(4).cloned().unwrap_or(Value::Null),
                "callback_id": r.get(5).cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    Ok(json!({"window": window.describe(), "count": items.len(), "questions": items}))
}

/// `/timeline` — two aligned series over time: polls created and votes cast per
/// bucket, for a stacked/dual line graph. Shares one time axis (`buckets`).
fn timeline(
    db: &mut Client,
    window: &Window,
    params: &BTreeMap<String, String>,
) -> Result<Value, String> {
    let bucket_ms = match params.get("bucket").map(|s| s.as_str()) {
        Some("hour") => 3_600_000,
        _ => DAY_MS,
    };

    let poll_pts = bucketed_series(db, window, "Poll", bucket_ms)?;
    let vote_pts = bucketed_series(db, window, "PollVote", bucket_ms)?;

    // Shared time axis: every bucket either series touched.
    let mut all: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    for (b, _) in poll_pts.iter().chain(vote_pts.iter()) {
        all.insert(*b);
    }
    let buckets: Vec<Value> = all
        .iter()
        .map(|b| json!({"bucket_start_ms": b, "date": iso_date(*b)}))
        .collect();

    let to_points = |pts: &[(i64, i64)]| -> Vec<Value> {
        pts.iter()
            .map(|(b, c)| json!({"bucket_start_ms": b, "date": iso_date(*b), "count": c}))
            .collect()
    };

    Ok(json!({
        "window": window.describe(),
        "bucket_ms": bucket_ms,
        "buckets": buckets,
        "polls": to_points(&poll_pts),
        "votes": to_points(&vote_pts),
    }))
}

/// One bucketed count series for `entity_type`: `(bucket_start_ms, count)` sorted
/// by time.
fn bucketed_series(
    db: &mut Client,
    window: &Window,
    entity_type: &str,
    bucket_ms: i64,
) -> Result<Vec<(i64, i64)>, String> {
    let res = db
        .query(json!({
            "find": [
                {"agg": "bucket", "var": format!("?ts, {bucket_ms}")},
                {"agg": "count", "var": "*"},
            ],
            "where": where_type(entity_type, window, json!({})),
            "order_by": [{"var": format!("bucket(?ts, {bucket_ms})")}],
        }))
        .map_err(|e| e.to_string())?;
    Ok(res
        .rows
        .iter()
        .map(|r| {
            (
                r.first().and_then(|v| v.as_i64()).unwrap_or(0),
                r.get(1).and_then(|v| v.as_i64()).unwrap_or(0),
            )
        })
        .collect())
}

/// `/recent` — the last N polls created, newest first.
fn recent(db: &mut Client, window: &Window, n: usize) -> Result<Value, String> {
    let res = db
        .query(json!({
            "find": ["?ts", "?q", "?opts", "?anon", "?team", "?cb"],
            "where": where_type(
                "Poll",
                window,
                json!({"question": "?q", "option_count": "?opts", "anonymous": "?anon", "team_id": "?team", "callback_id": "?cb"}),
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
                "question": r.get(1).cloned().unwrap_or(Value::Null),
                "options": r.get(2).cloned().unwrap_or(Value::Null),
                "anonymous": r.get(3).cloned().unwrap_or(Value::Null),
                "team_id": r.get(4).cloned().unwrap_or(Value::Null),
                "callback_id": r.get(5).cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    Ok(json!({"window": window.describe(), "count": items.len(), "recent": items}))
}

// ---- helpers (shared shape with the analytics function) ----

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
