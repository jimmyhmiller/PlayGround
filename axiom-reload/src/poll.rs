//! poll — incrementally pull Slack `/poll` app events from the Axiom `vercel`
//! dataset into datalog-db as `Poll` and `PollVote` entities.
//!
//! The `/poll` Slack slash-command app logs its DB writes to Vercel (and thus
//! Axiom) as `message` lines. Two of those lines carry the structured signal we
//! want, and the JSON `info` payload embedded in each is real JSON even though
//! the surrounding log text is Node's `util.inspect` output:
//!
//! - **Poll creation** — `INSERT INTO poll (team_id, callback_id, info) ...`
//!   with `values: ['<team_id>', '<callback_id>', '{"question":...,"options":
//!   [{"value":..,"votes":[],"index":..}],"anonymous":..}']`.
//! - **A vote** — `UPDATE poll SET info = $1::jsonb WHERE callback_id = $2`
//!   with `values: ['<info-json-AFTER-the-vote>', '<callback_id>']`. The `info`
//!   here is the FULL poll state after that vote, so each option carries its
//!   current `votes` array (Slack user ids). One UPDATE = one vote; the latest
//!   UPDATE for a `callback_id` is the poll's final tally.
//!
//! We model them as two entity types, both keyed by `callback_id`:
//! - `Poll`     — one per INSERT: callback_id, team_id, question, option_count,
//!                anonymous, ts, day.
//! - `PollVote` — one per UPDATE (a vote snapshot): callback_id, ts, day,
//!                total_votes (sum over all options' votes arrays), option_count,
//!                question.
//!
//! Idempotency mirrors the PageView loader: a per-type high-water-mark on `ts`
//! (we take the min of the two, so neither type ever lags), disjoint hourly
//! windows from there to now, and assert-only ops. Re-running adds ~0.

use std::collections::BTreeMap;

use datalog_client::Client;
use serde_json::{json, Value};

use crate::{http_post_json, iso, now_ms, parse_axiom_time, tabular_rows};

const DATASET: &str = "vercel";
const WINDOW_MS: i64 = 3_600_000; // 1 hour, same as the PageView loader
const ROW_LIMIT: i64 = 50_000;
const DAY_MS: i64 = 86_400_000;
/// If the DB has no polls yet, how far back to seed (Axiom retention is ~30d).
const SEED_LOOKBACK_MS: i64 = 31 * DAY_MS;

/// Entry point, called after the PageView reload in `main`. Shares the same
/// `AXIOM_API_KEY` / `DATALOG_ADDR` env and datalog-db connection conventions.
pub fn run() -> Result<(), String> {
    let token = std::env::var("AXIOM_API_KEY")
        .map_err(|_| "AXIOM_API_KEY not set (need a xapt- PAT with query scope)".to_string())?;
    let datalog_addr =
        std::env::var("DATALOG_ADDR").unwrap_or_else(|_| "127.0.0.1:5557".to_string());

    let mut db = Client::connect(&datalog_addr)
        .map_err(|e| format!("connecting to datalog-db at {datalog_addr}: {e}"))?;

    // High-water-mark: the OLDER of the two types' newest ts, so whichever type
    // is behind gets caught up (a window may yield both Polls and PollVotes).
    // On the very first run neither type exists yet; datalog-db errors on an
    // unknown type rather than returning None, so treat that as "empty" — the
    // first assert below creates the type.
    let hwm_poll = max_ts_or_empty(&mut db, "Poll")?;
    let hwm_vote = max_ts_or_empty(&mut db, "PollVote")?;
    let hwm = match (hwm_poll, hwm_vote) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    let now = now_ms();
    let start = match hwm {
        Some(ts) => ts + 1,
        None => now - SEED_LOOKBACK_MS,
    };
    match hwm {
        Some(ts) => println!(
            "poll-reload: high-water-mark ts={ts} ({}), fetching newer poll events",
            iso(ts)
        ),
        None => println!("poll-reload: no polls yet, seeding from {}", iso(start)),
    }
    if start >= now {
        println!("poll-reload: already up to date");
        return Ok(());
    }

    let mut polls = 0usize;
    let mut votes = 0usize;
    let mut win_start = start;
    while win_start < now {
        let win_end = (win_start + WINDOW_MS).min(now);
        let events = fetch_window(&token, win_start, win_end)
            .map_err(|e| format!("fetching [{}, {}): {e}", iso(win_start), iso(win_end)))?;
        let mut ops = Vec::new();
        for ev in &events {
            match event_to_op(ev) {
                Some(Op::Poll(v)) => {
                    polls += 1;
                    ops.push(v);
                }
                // A vote emits TWO ops: the append-only PollVote event AND an
                // upsert of the parent Poll's latest tally (keyed on
                // callback_id). Since events are processed in ascending ts and
                // datalog upserts the unique key in place, the Poll ends up
                // holding the LATEST vote snapshot — the true "current votes".
                Some(Op::Vote { vote, poll_upsert }) => {
                    votes += 1;
                    ops.push(vote);
                    ops.push(poll_upsert);
                }
                None => {}
            }
        }
        let n = ops.len();
        if n > 0 {
            db.transact(ops)
                .map_err(|e| format!("asserting {n} poll ops: {e}"))?;
        }
        println!(
            "  {} .. {}: {} events ({} poll ops; running totals {} polls, {} votes)",
            iso(win_start),
            iso(win_end),
            events.len(),
            n,
            polls,
            votes
        );
        win_start = win_end;
    }

    println!("poll-reload: DONE — asserted {polls} Poll(s), {votes} PollVote(s)");
    Ok(())
}

/// `max(ts)` for a type, treating an "unknown entity type" server error (the
/// type was never created) the same as an empty table: `None`. Other errors
/// propagate.
fn max_ts_or_empty(db: &mut Client, type_name: &str) -> Result<Option<i64>, String> {
    match db.max_i64(type_name, "ts") {
        Ok(v) => Ok(v),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("unknown entity type") {
                Ok(None)
            } else {
                Err(format!("querying max({type_name}.ts): {msg}"))
            }
        }
    }
}

/// Fetch the `/poll` events in `[start_ms, end_ms)` whose `message` is one of the
/// two structured lines we parse. We filter server-side to keep windows small:
/// the `INSERT INTO poll` and `UPDATE poll` lines only.
fn fetch_window(
    token: &str,
    start_ms: i64,
    end_ms: i64,
) -> Result<Vec<BTreeMap<String, Value>>, String> {
    // Only the two write-log lines carry the JSON we parse. Filtering here (not
    // client-side) keeps each window well under the row cap.
    let apl = format!(
        "['{DATASET}'] \
         | where message contains 'INSERT INTO poll' or message contains 'UPDATE poll' \
         | sort by _time asc | limit {ROW_LIMIT}"
    );
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

/// A parsed op (or pair of ops) to transact.
enum Op {
    /// A poll creation -> one `Poll` upsert.
    Poll(Value),
    /// A vote -> an append-only `PollVote` event PLUS a `Poll` upsert that
    /// updates the parent poll's current tally (`total_votes`/`last_vote_ts`).
    Vote { vote: Value, poll_upsert: Value },
}

/// Classify and parse one Axiom event into poll ops, or `None` if it has no
/// usable time or isn't one of the two log lines we model.
fn event_to_op(ev: &BTreeMap<String, Value>) -> Option<Op> {
    let ms = ev.get("_time").and_then(parse_axiom_time)?;
    let msg = ev.get("message").and_then(|v| v.as_str())?;

    if msg.contains("INSERT INTO poll") {
        return parse_insert(msg, ms).map(Op::Poll);
    }
    if msg.contains("UPDATE poll") {
        return parse_update(msg, ms);
    }
    None
}

/// Parse an `INSERT INTO poll` line. The `values: [ ... ]` block holds, in order,
/// `'<team_id>'`, `'<callback_id>'`, `'<info-json>'`. team_id and callback_id are
/// single-quoted SQL string literals; info is the JSON object. We pull the two
/// uuid-ish literals and the embedded JSON object.
fn parse_insert(msg: &str, ms: i64) -> Option<Value> {
    let info = extract_info_json(msg)?;
    // The two single-quoted literals before the JSON: team_id then callback_id.
    // Take the last two single-quoted tokens that appear before the `{` of info.
    let (team_id, callback_id) = two_quoted_before_brace(msg)?;
    let (question, option_count, anonymous, total_votes) = poll_fields(&info);

    let mut data = serde_json::Map::new();
    data.insert("callback_id".into(), json!(callback_id));
    data.insert("team_id".into(), json!(team_id));
    data.insert("ts".into(), json!(ms));
    data.insert("day".into(), json!(ms / DAY_MS));
    if let Some(q) = question {
        data.insert("question".into(), json!(q));
    }
    data.insert("option_count".into(), json!(option_count));
    data.insert("anonymous".into(), json!(anonymous));
    // A freshly-inserted poll has no votes; record it for symmetry with PollVote.
    data.insert("total_votes".into(), json!(total_votes));
    Some(json!({"assert": "Poll", "data": Value::Object(data)}))
}

/// Parse an `UPDATE poll` line: `values: ['<info-json-after-vote>', '<callback_id>']`.
/// The info JSON is the FULL poll state after the vote, so summing each option's
/// `votes` array length gives the running vote total. One UPDATE = one vote.
///
/// Emits two ops:
/// - a `PollVote` (append-only event row, for vote-over-time series), and
/// - a `Poll` upsert keyed on `callback_id` that overwrites the poll's
///   `total_votes`/`last_vote_ts` with THIS snapshot. Because windows are walked
///   in ascending ts and datalog upserts the unique key in place, the Poll row
///   converges to the latest vote tally — the value `/top-polls` ranks on.
fn parse_update(msg: &str, ms: i64) -> Option<Op> {
    let info = extract_info_json(msg)?;
    // The callback_id is the single-quoted literal AFTER the JSON object.
    let callback_id = quoted_after_brace(msg)?;
    let (question, option_count, _anon, total_votes) = poll_fields(&info);

    let mut vote = serde_json::Map::new();
    vote.insert("callback_id".into(), json!(callback_id));
    vote.insert("ts".into(), json!(ms));
    vote.insert("day".into(), json!(ms / DAY_MS));
    vote.insert("total_votes".into(), json!(total_votes));
    vote.insert("option_count".into(), json!(option_count));
    if let Some(q) = &question {
        vote.insert("question".into(), json!(q));
    }

    // Poll upsert: callback_id is the unique key, so this updates the existing
    // poll (created by its INSERT) rather than inserting a new one. We set the
    // vote-derived fields; the creation-only fields (team_id, anonymous, ts) are
    // left as the INSERT set them.
    let mut upsert = serde_json::Map::new();
    upsert.insert("callback_id".into(), json!(callback_id));
    upsert.insert("total_votes".into(), json!(total_votes));
    upsert.insert("last_vote_ts".into(), json!(ms));
    // option_count can grow if the poll was edited; keep it current.
    upsert.insert("option_count".into(), json!(option_count));
    // The vote snapshot also carries the question. Set it so polls whose INSERT
    // predates Axiom's 30-day retention (old, still-active polls we only see via
    // votes) still get a question. For polls with an INSERT in range this just
    // re-asserts the same value (a no-op).
    if let Some(q) = &question {
        upsert.insert("question".into(), json!(q));
    }

    Some(Op::Vote {
        vote: json!({"assert": "PollVote", "data": Value::Object(vote)}),
        poll_upsert: json!({"assert": "Poll", "data": Value::Object(upsert)}),
    })
}

/// Pull derived fields out of a parsed poll `info` object:
/// `(question, option_count, anonymous, total_votes)`.
fn poll_fields(info: &Value) -> (Option<String>, i64, bool, i64) {
    let question = info
        .get("question")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let anonymous = info
        .get("anonymous")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let options = info.get("options").and_then(|v| v.as_array());
    let option_count = options.map(|a| a.len() as i64).unwrap_or(0);
    let total_votes = options
        .map(|a| {
            a.iter()
                .map(|o| {
                    o.get("votes")
                        .and_then(|v| v.as_array())
                        .map(|a| a.len() as i64)
                        .unwrap_or(0)
                })
                .sum()
        })
        .unwrap_or(0);
    (question, option_count, anonymous, total_votes)
}

/// Find and parse the embedded poll `info` JSON object inside a log message.
///
/// The surrounding message is Node's `util.inspect` output (which also starts
/// with `{`), so we can't just take the first brace. The info value is the JSON
/// object literal that begins with `{"` (a quoted key follows immediately). We
/// scan each `{"` occurrence, brace-match it (respecting JSON strings so braces
/// in option text don't miscount), and return the first slice that parses.
fn extract_info_json(msg: &str) -> Option<Value> {
    info_json_span(msg).map(|(v, _start, _end)| v)
}

/// Like [`extract_info_json`] but also returns the byte span `[start, end)` of
/// the matched object in `msg`, so callers can inspect the text after it.
fn info_json_span(msg: &str) -> Option<(Value, usize, usize)> {
    let bytes = msg.as_bytes();
    let mut search = 0;
    while let Some(rel) = msg[search..].find("{\"") {
        let start = search + rel;
        if let Some(end) = match_brace(bytes, start) {
            if let Ok(v) = serde_json::from_str::<Value>(&msg[start..end]) {
                return Some((v, start, end));
            }
        }
        search = start + 1;
    }
    None
}

/// Given `bytes[start] == b'{'`, return the index just past the matching `}`,
/// respecting JSON string literals so braces inside strings don't miscount.
fn match_brace(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for i in start..bytes.len() {
        let c = bytes[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// The two single-quoted SQL string literals immediately before the info JSON
/// object — i.e. team_id then callback_id in an INSERT. Returns them in source
/// order. Only literals before the info object are considered, so option text
/// inside the JSON can't be mistaken for a SQL literal.
fn two_quoted_before_brace(msg: &str) -> Option<(String, String)> {
    let (_info, start, _end) = info_json_span(msg)?;
    let lits = single_quoted_literals(&msg[..start]);
    let n = lits.len();
    if n < 2 {
        return None;
    }
    Some((lits[n - 2].clone(), lits[n - 1].clone()))
}

/// The first single-quoted SQL string literal AFTER the info JSON object —
/// i.e. the callback_id in an UPDATE.
fn quoted_after_brace(msg: &str) -> Option<String> {
    let (_info, _start, end) = info_json_span(msg)?;
    // The info JSON is wrapped in a single-quoted SQL literal: `'{...}'`. The
    // closing `'` sits just past `end`; skip up to and including it so the scan
    // doesn't pair that dangling quote with the next literal's opening quote.
    let rest = &msg[end..];
    let after = match rest.find('\'') {
        Some(q) => &rest[q + 1..],
        None => rest,
    };
    single_quoted_literals(after).into_iter().next()
}

/// Extract all single-quoted literals from a slice of `util.inspect` text.
/// Slack ids and uuids never contain single quotes, so a simple paired-quote
/// scan suffices (no escape handling needed for the id/uuid literals we want).
fn single_quoted_literals(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\'' {
            if let Some(rel) = s[i + 1..].find('\'') {
                let lit = &s[i + 1..i + 1 + rel];
                out.push(lit.to_string());
                i = i + 1 + rel + 1;
                continue;
            } else {
                break;
            }
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const INSERT_MSG: &str = "Querying {\n  text: '\\n' +\n    '      INSERT INTO poll (team_id, callback_id, info)\\n' +\n    '      VALUES ($1, $2, $3::jsonb);\\n' +\n    '    ',\n  values: [\n    'e12f3a04-912e-4184-92c7-b029f787e217',\n    '60200d93-87f2-4ec8-a452-d4ec37d2c3ac',\n    '{\"question\":\"are you ok to TA the above response from management?\",\"options\":[{\"value\":\"yes\",\"votes\":[],\"index\":0},{\"value\":\"no\",\"votes\":[],\"index\":1}],\"anonymous\":false}'\n  ]\n}";

    const UPDATE_MSG: &str = "Querying {\n  text: '\\n' +\n    '      UPDATE poll\\n' +\n    '      SET info = $1::jsonb\\n' +\n    '      WHERE callback_id = $2;\\n' +\n    '    ',\n  values: [\n    '{\"options\":[{\"index\":0,\"value\":\"Yes\",\"votes\":[\"U0A1PQNCUSX\",\"U044BFT1FB5\"]},{\"index\":1,\"value\":\"No\",\"votes\":[\"U01072943K2\"]}],\"question\":\"Are we ok\",\"anonymous\":false}',\n    '23c5a632-684e-4fc8-acc4-a0c68c59de6a'\n  ]\n}";

    #[test]
    fn parse_insert_extracts_fields() {
        let op = parse_insert(INSERT_MSG, 1_000).expect("insert parses");
        let d = &op["data"];
        assert_eq!(d["callback_id"], json!("60200d93-87f2-4ec8-a452-d4ec37d2c3ac"));
        assert_eq!(d["team_id"], json!("e12f3a04-912e-4184-92c7-b029f787e217"));
        assert_eq!(d["option_count"], json!(2));
        assert_eq!(d["anonymous"], json!(false));
        assert_eq!(d["total_votes"], json!(0));
        assert_eq!(
            d["question"],
            json!("are you ok to TA the above response from management?")
        );
    }

    #[test]
    fn parse_update_counts_votes() {
        let op = parse_update(UPDATE_MSG, 2_000).expect("update parses");
        let (vote, upsert) = match op {
            Op::Vote { vote, poll_upsert } => (vote, poll_upsert),
            _ => panic!("expected a Vote op"),
        };

        // The append-only PollVote event row.
        let d = &vote["data"];
        assert_eq!(vote["assert"], json!("PollVote"));
        assert_eq!(d["callback_id"], json!("23c5a632-684e-4fc8-acc4-a0c68c59de6a"));
        // 2 + 1 = 3 votes across the two options.
        assert_eq!(d["total_votes"], json!(3));
        assert_eq!(d["option_count"], json!(2));
        assert_eq!(d["question"], json!("Are we ok"));

        // The Poll upsert: keyed on callback_id, carries the latest tally and
        // vote time, and must NOT clobber creation-only fields (no ts/team_id).
        let u = &upsert["data"];
        assert_eq!(upsert["assert"], json!("Poll"));
        assert_eq!(u["callback_id"], json!("23c5a632-684e-4fc8-acc4-a0c68c59de6a"));
        assert_eq!(u["total_votes"], json!(3));
        assert_eq!(u["last_vote_ts"], json!(2_000));
        // The question rides along (helps polls whose INSERT predates retention).
        assert_eq!(u["question"], json!("Are we ok"));
        // But creation-only fields must NOT be set by a vote upsert.
        assert!(u.get("ts").is_none(), "upsert must not set ts");
        assert!(u.get("team_id").is_none(), "upsert must not set team_id");
    }

    #[test]
    fn brace_matching_survives_braces_in_text() {
        // Option text containing a brace must not break depth counting.
        let info = extract_info_json(
            "x {\"question\":\"a }{ b\",\"options\":[{\"value\":\"v\",\"votes\":[],\"index\":0}],\"anonymous\":true} 'cb'",
        )
        .expect("parses");
        assert_eq!(info["question"], json!("a }{ b"));
    }
}
