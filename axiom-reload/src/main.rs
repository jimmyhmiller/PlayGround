//! axiom-reload — incrementally pull new Vercel pageviews from Axiom into
//! datalog-db.
//!
//! Idempotent by construction: we ask datalog-db for the newest `ts` it already
//! holds (the high-water-mark), then fetch only Axiom events strictly newer than
//! that, in disjoint time windows, and assert them as `PageView` entities. Run
//! it every N minutes and it keeps the DB current without ever duplicating a row
//! — there is nothing to dedup because we never re-fetch a window we've crossed.
//!
//! This replaces the old two-script Python flow (full fixed-window dump + load,
//! which duplicated on re-run) with one self-contained Rust binary. No Python.
//!
//! Sources/credentials match the existing setup:
//! - Axiom org `jimmyhmiller-hcux`, dataset `vercel`, APL endpoint
//!   `https://api.axiom.co/v1/datasets/_apl?format=tabular`.
//! - Token from `$AXIOM_API_KEY` (a `xapt-` PAT with query scope).
//! - datalog-db at `127.0.0.1:5557` (override with `$DATALOG_ADDR`).

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use datalog_client::Client;
use serde_json::{json, Value};

const AXIOM_URL: &str = "https://api.axiom.co/v1/datasets/_apl?format=tabular";
const AXIOM_ORG: &str = "jimmyhmiller-hcux";
const DATASET: &str = "vercel";
const WINDOW_MS: i64 = 3_600_000; // 1 hour; busiest hour ~2.4k rows, well under cap
const ROW_LIMIT: i64 = 50_000; // APL default cap is 1000; raise it explicitly
/// If the DB is empty, how far back to seed from (Axiom retention is ~30 days).
const SEED_LOOKBACK_MS: i64 = 31 * 86_400_000;

/// Axiom field -> PageView field. String fields. (Matches the Python loader's
/// FIELD_MAP so the schema and existing 228k rows stay consistent.)
const STRING_FIELDS: &[(&str, &str)] = &[
    ("request.host", "host"),
    ("request.path", "path"),
    ("vercel.route", "route"),
    ("request.method", "method"),
    ("request.ip", "ip"),
    ("request.referer", "referer"),
    ("request.userAgent", "userAgent"),
    ("vercel.region", "region"),
    ("vercel.source", "source"),
    ("request.vercelCache", "cache"),
    ("vercel.projectName", "project"),
    ("level", "level"),
    ("request.id", "requestId"),
];
const INT_FIELDS: &[(&str, &str)] = &[
    ("request.statusCode", "status"),
    ("report.maxMemoryUsedMb", "memoryMb"),
];
const FLOAT_FIELDS: &[(&str, &str)] = &[("report.durationMs", "durationMs")];

fn main() {
    if let Err(e) = run() {
        eprintln!("axiom-reload: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let token = std::env::var("AXIOM_API_KEY")
        .map_err(|_| "AXIOM_API_KEY not set (need a xapt- PAT with query scope)".to_string())?;
    let datalog_addr =
        std::env::var("DATALOG_ADDR").unwrap_or_else(|_| "127.0.0.1:5557".to_string());

    let mut db = Client::connect(&datalog_addr)
        .map_err(|e| format!("connecting to datalog-db at {datalog_addr}: {e}"))?;

    // High-water-mark: newest ts already in the DB. Fetch strictly after it.
    let hwm = db
        .max_i64("PageView", "ts")
        .map_err(|e| format!("querying max(ts): {e}"))?;
    let now = now_ms();
    let start = match hwm {
        Some(ts) => ts + 1, // strictly newer than what we have
        None => now - SEED_LOOKBACK_MS,
    };
    match hwm {
        Some(ts) => println!(
            "axiom-reload: high-water-mark ts={ts} ({}), fetching newer events",
            iso(ts)
        ),
        None => println!("axiom-reload: DB empty, seeding from {}", iso(start)),
    }
    if start >= now {
        println!("axiom-reload: already up to date (nothing newer than now)");
        return Ok(());
    }

    // Walk disjoint [win_start, win_end) windows from the HWM to now. Each window
    // is fetched exactly once; no overlap means no dedup needed.
    let mut total_asserted = 0usize;
    let mut win_start = start;
    while win_start < now {
        let win_end = (win_start + WINDOW_MS).min(now);
        let events = fetch_window(&token, win_start, win_end)
            .map_err(|e| format!("fetching [{}, {}): {e}", iso(win_start), iso(win_end)))?;
        let ops: Vec<Value> = events.iter().filter_map(event_to_op).collect();
        let n = ops.len();
        if n > 0 {
            db.transact(ops)
                .map_err(|e| format!("asserting {n} PageViews: {e}"))?;
            total_asserted += n;
        }
        println!(
            "  {} .. {}: {} events ({} asserted, total {})",
            iso(win_start),
            iso(win_end),
            events.len(),
            n,
            total_asserted
        );
        win_start = win_end;
    }

    println!("axiom-reload: DONE — asserted {total_asserted} new PageView(s)");
    Ok(())
}

/// Fetch all events in `[start_ms, end_ms)` from Axiom, returning them as a list
/// of flat field maps (column-major Axiom response flattened to row dicts).
fn fetch_window(token: &str, start_ms: i64, end_ms: i64) -> Result<Vec<BTreeMap<String, Value>>, String> {
    let apl = format!("['{DATASET}'] | sort by _time asc | limit {ROW_LIMIT}");
    let body = json!({
        "apl": apl,
        "startTime": iso(start_ms),
        "endTime": iso(end_ms),
    });

    let resp = http_post_json(token, &body)?;
    let rows = tabular_rows(&resp);
    if rows.len() as i64 >= ROW_LIMIT {
        return Err(format!(
            "window [{}, {}) hit row cap {ROW_LIMIT}; shrink WINDOW_MS",
            iso(start_ms),
            iso(end_ms)
        ));
    }
    Ok(rows)
}

/// POST an APL query to Axiom with retries on transient errors.
fn http_post_json(token: &str, body: &Value) -> Result<Value, String> {
    let mut last_err = String::new();
    for attempt in 0..6 {
        let result = ureq::post(AXIOM_URL)
            .set("Authorization", &format!("Bearer {token}"))
            .set("X-AXIOM-ORG-ID", AXIOM_ORG)
            .set("Content-Type", "application/json")
            .send_json(body.clone());
        match result {
            Ok(resp) => {
                return resp
                    .into_json::<Value>()
                    .map_err(|e| format!("parsing Axiom response: {e}"));
            }
            Err(ureq::Error::Status(code, resp)) => {
                let retryable = matches!(code, 429 | 500 | 502 | 503 | 504);
                let detail = resp
                    .into_string()
                    .unwrap_or_default()
                    .chars()
                    .take(300)
                    .collect::<String>();
                last_err = format!("HTTP {code}: {detail}");
                if !retryable || attempt == 5 {
                    return Err(last_err);
                }
            }
            Err(e) => {
                last_err = e.to_string();
                if attempt == 5 {
                    return Err(last_err);
                }
            }
        }
        // Exponential backoff: 1s, 2s, 4s, ... using a busy sleep-free wait.
        std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
    }
    Err(format!("exhausted retries: {last_err}"))
}

/// Flatten Axiom's tabular (column-major) response into row maps. Empty windows
/// come back with no `tables`/`columns`; handle that as zero rows.
fn tabular_rows(resp: &Value) -> Vec<BTreeMap<String, Value>> {
    let Some(tables) = resp.get("tables").and_then(|t| t.as_array()) else {
        return Vec::new();
    };
    let Some(table) = tables.first() else {
        return Vec::new();
    };
    let Some(columns) = table.get("columns").and_then(|c| c.as_array()) else {
        return Vec::new();
    };
    if columns.is_empty() {
        return Vec::new();
    }
    let fields: Vec<String> = table
        .get("fields")
        .and_then(|f| f.as_array())
        .map(|a| {
            a.iter()
                .map(|f| {
                    f.get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default();

    let n_rows = columns.first().and_then(|c| c.as_array()).map_or(0, |a| a.len());
    let mut rows = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let mut row = BTreeMap::new();
        for (j, field) in fields.iter().enumerate() {
            if let Some(col) = columns.get(j).and_then(|c| c.as_array()) {
                if let Some(v) = col.get(i) {
                    if !v.is_null() {
                        row.insert(field.clone(), v.clone());
                    }
                }
            }
        }
        rows.push(row);
    }
    rows
}

/// Convert one raw Axiom event into a datalog-db `assert PageView` op, or `None`
/// if it has no usable `_time` (can't be indexed as a pageview).
fn event_to_op(ev: &BTreeMap<String, Value>) -> Option<Value> {
    let ms = ev.get("_time").and_then(parse_axiom_time)?;
    let mut data = serde_json::Map::new();
    data.insert("ts".into(), json!(ms));
    data.insert("day".into(), json!(ms / 86_400_000));

    for (src, dst) in STRING_FIELDS {
        if let Some(s) = ev.get(*src).and_then(|v| v.as_str()) {
            if !s.is_empty() {
                data.insert((*dst).into(), json!(s));
            }
        }
    }
    for (src, dst) in INT_FIELDS {
        if let Some(n) = ev.get(*src).and_then(as_i64_loose) {
            data.insert((*dst).into(), json!(n));
        }
    }
    for (src, dst) in FLOAT_FIELDS {
        if let Some(n) = ev.get(*src).and_then(|v| v.as_f64()) {
            data.insert((*dst).into(), json!(n));
        }
    }
    Some(json!({"assert": "PageView", "data": Value::Object(data)}))
}

/// Parse an Axiom `_time` (ISO-8601, e.g. `2026-06-10T14:50:05.653Z`, or already
/// a number of ms) into epoch milliseconds.
fn parse_axiom_time(v: &Value) -> Option<i64> {
    if let Some(n) = v.as_i64() {
        return Some(n);
    }
    if let Some(f) = v.as_f64() {
        return Some(f as i64);
    }
    let s = v.as_str()?;
    parse_iso_ms(s)
}

/// Minimal RFC3339/ISO-8601 parser for the `YYYY-MM-DDTHH:MM:SS[.fff]Z` shape
/// Axiom emits, returning epoch milliseconds. We avoid a chrono dependency: the
/// format is fixed and UTC ('Z').
fn parse_iso_ms(s: &str) -> Option<i64> {
    // Split date and time on 'T'.
    let (date, rest) = s.split_once('T')?;
    let mut dp = date.split('-');
    let year: i64 = dp.next()?.parse().ok()?;
    let month: i64 = dp.next()?.parse().ok()?;
    let day: i64 = dp.next()?.parse().ok()?;

    // rest like "14:50:05.653Z" or "14:50:05Z".
    let rest = rest.strip_suffix('Z').unwrap_or(rest);
    let (hms, frac) = match rest.split_once('.') {
        Some((a, b)) => (a, b),
        None => (rest, ""),
    };
    let mut tp = hms.split(':');
    let hour: i64 = tp.next()?.parse().ok()?;
    let min: i64 = tp.next()?.parse().ok()?;
    let sec: i64 = tp.next()?.parse().ok()?;
    // Milliseconds from the fractional part (pad/truncate to 3 digits).
    let mut millis = 0i64;
    if !frac.is_empty() {
        let mut digits = frac.to_string();
        digits.truncate(3);
        while digits.len() < 3 {
            digits.push('0');
        }
        millis = digits.parse().ok()?;
    }

    let days = days_from_civil(year, month, day);
    let secs = days * 86_400 + hour * 3600 + min * 60 + sec;
    Some(secs * 1000 + millis)
}

/// Days since the Unix epoch for a civil (proleptic Gregorian) date. Howard
/// Hinnant's `days_from_civil` algorithm.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Coerce a JSON number-or-numeric-string into i64 (Axiom sometimes stringifies).
fn as_i64_loose(v: &Value) -> Option<i64> {
    if let Some(n) = v.as_i64() {
        return Some(n);
    }
    if let Some(f) = v.as_f64() {
        return Some(f as i64);
    }
    v.as_str().and_then(|s| s.parse().ok())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Format epoch-ms as the ISO-8601 string Axiom's APL `startTime`/`endTime`
/// expect (`YYYY-MM-DDTHH:MM:SS.000Z`).
fn iso(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let millis = ms.rem_euclid(1000);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let hour = tod / 3600;
    let min = (tod % 3600) / 60;
    let sec = tod % 60;
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}.{millis:03}Z")
}

/// Inverse of [`days_from_civil`]: epoch-day -> (year, month, day).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_roundtrip_through_parse() {
        // A known instant: 2026-06-10T14:50:05.653Z
        let ms = 1_781_103_005_653;
        assert_eq!(iso(ms), "2026-06-10T14:50:05.653Z");
        assert_eq!(parse_iso_ms("2026-06-10T14:50:05.653Z"), Some(ms));
    }

    #[test]
    fn parse_without_fraction() {
        assert_eq!(
            parse_iso_ms("2026-06-10T14:50:05Z"),
            Some(1_781_103_005_000)
        );
    }

    #[test]
    fn epoch_zero() {
        assert_eq!(iso(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(parse_iso_ms("1970-01-01T00:00:00.000Z"), Some(0));
    }
}
