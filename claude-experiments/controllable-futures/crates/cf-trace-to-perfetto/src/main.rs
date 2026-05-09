//! Convert a cf-runtime JSONL trace into Chrome trace event format
//! (the format Perfetto and chrome://tracing both ingest).
//!
//! Mapping decisions:
//!
//! - **Each tokio task = its own track.** tid = 10000 + task_id. Spans
//!   and polls that ran on the task all stack on this lane, so you see
//!   "Build span > ResolveModule span > poll(123ns)" nested visually.
//!
//! - **Each worker thread = its own track.** tid = 1 + worker_idx. We
//!   ALSO emit each poll on its worker's lane, mirrored from the task
//!   lane, so you can see worker-level utilization independently.
//!
//! - **Wake edges = flow events.** Each Wake gets a `s` (start) on the
//!   source task and `f` (finish) on the target task; Perfetto draws an
//!   arrow between them when you select either endpoint.
//!
//! - **Spawn = instant event.** Single point, with parent metadata.
//!
//! - **Allocations = counter track** (`ph: "C"`). Per-build cumulative
//!   bytes, so you see allocation rate as a graph at the top.
//!
//! - **Spans and polls = duration events** (`ph: "X"` with `dur`).
//!
//! Format output as `{"traceEvents": [...]}` so it loads in chrome://
//! tracing as a top-level JSON. Perfetto also accepts that shape.
//!
//! Usage: `cf-trace-to-perfetto input.jsonl > output.json`

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input: Box<dyn BufRead> = if args.len() > 1 {
        Box::new(BufReader::new(
            std::fs::File::open(&args[1]).expect("open input"),
        ))
    } else {
        Box::new(BufReader::new(std::io::stdin()))
    };
    let stdout = std::io::stdout();
    let lock = stdout.lock();
    let mut emitter = Emitter::new(std::io::BufWriter::new(lock));

    let mut anchor_ns: Option<u64> = None;
    let to_us = |ns: u64, anchor: u64| -> f64 {
        (ns.saturating_sub(anchor)) as f64 / 1000.0
    };

    let mut span_start: HashMap<u64, SpanState> = HashMap::new();
    let mut poll_start: HashMap<u64, u64> = HashMap::new();
    let mut alloc_total_bytes: i64 = 0;

    // Pre-emit metadata so the tracks have human-friendly names.
    // Worker tracks are tid 1..N (assume up to 32). Task tracks are
    // tid 10000+task_id, named when we see a Spawned event.
    for w in 0..32 {
        let m = format!(
            r#"{{"ph":"M","name":"thread_name","pid":1,"tid":{},"args":{{"name":"worker {}"}}}}"#,
            w + 1,
            w
        );
        emitter.emit(&m);
    }
    emitter.emit(r#"{"ph":"M","name":"process_name","pid":1,"args":{"name":"turbopack-cli"}}"#);

    let mut line_buf = String::new();
    let mut reader = input;
    loop {
        line_buf.clear();
        let n = reader.read_line(&mut line_buf).expect("read");
        if n == 0 {
            break;
        }
        let line = line_buf.trim();
        if line.is_empty() {
            continue;
        }
        let evt = match parse(line) {
            Some(e) => e,
            None => continue,
        };
        if anchor_ns.is_none() {
            anchor_ns = Some(evt.ts_ns);
        }
        let anchor = anchor_ns.unwrap();
        let ts = to_us(evt.ts_ns, anchor);

        match evt.kind {
            Kind::Spawned { name, parent } => {
                // Name the task track.
                if let Some(tid) = evt.task.map(task_tid) {
                    let m = format!(
                        r#"{{"ph":"M","name":"thread_name","pid":1,"tid":{},"args":{{"name":"task #{} {}"}}}}"#,
                        tid,
                        evt.task.unwrap(),
                        json_escape(&name),
                    );
                    emitter.emit(&m);
                    // Instant event for the spawn itself.
                    let parent_str = parent
                        .map(|p| format!(r#","parent":{p}"#))
                        .unwrap_or_default();
                    let m = format!(
                        r#"{{"ph":"i","name":"spawn","cat":"task","ts":{:.3},"pid":1,"tid":{},"s":"t","args":{{"name":"{}"{}}}}}"#,
                        ts,
                        tid,
                        json_escape(&name),
                        parent_str,
                    );
                    emitter.emit(&m);
                }
            }
            Kind::PollStart => {
                if let Some(t) = evt.task {
                    poll_start.insert(t, evt.ts_ns);
                }
            }
            Kind::PollEnd { resched, d_ns } => {
                if let Some(t) = evt.task {
                    if let Some(start) = poll_start.remove(&t) {
                        let start_us = to_us(start, anchor);
                        let dur_us = (d_ns as f64) / 1000.0;
                        let resched_str = if resched { "1" } else { "0" };
                        // On task lane:
                        let m = format!(
                            r#"{{"ph":"X","name":"poll","cat":"poll","ts":{:.3},"dur":{:.3},"pid":1,"tid":{},"args":{{"resched":{}}}}}"#,
                            start_us,
                            dur_us,
                            task_tid(t),
                            resched_str,
                        );
                        emitter.emit(&m);
                        // Mirror onto the worker lane if known:
                        if let Some(w) = evt.worker {
                            let m = format!(
                                r#"{{"ph":"X","name":"poll #{}","cat":"poll","ts":{:.3},"dur":{:.3},"pid":1,"tid":{},"args":{{"task":{}}}}}"#,
                                t,
                                start_us,
                                dur_us,
                                w + 1,
                                t,
                            );
                            emitter.emit(&m);
                        }
                    }
                }
            }
            Kind::Wake { from_task } => {
                if let (Some(src), Some(tgt)) = (from_task, evt.task) {
                    // Flow events: shared id, source then target.
                    let id = evt.seq;
                    let m1 = format!(
                        r#"{{"ph":"s","name":"wake","cat":"wake","id":{},"ts":{:.3},"pid":1,"tid":{}}}"#,
                        id,
                        ts,
                        task_tid(src),
                    );
                    emitter.emit(&m1);
                    let m2 = format!(
                        r#"{{"ph":"f","name":"wake","cat":"wake","id":{},"ts":{:.3},"pid":1,"tid":{},"bp":"e"}}"#,
                        id,
                        ts + 0.001,
                        task_tid(tgt),
                    );
                    emitter.emit(&m2);
                }
            }
            Kind::State { from, to } => {
                if let Some(t) = evt.task {
                    let m = format!(
                        r#"{{"ph":"i","name":"state","cat":"state","ts":{:.3},"pid":1,"tid":{},"s":"t","args":{{"from":"{}","to":"{}"}}}}"#,
                        ts,
                        task_tid(t),
                        from,
                        to,
                    );
                    emitter.emit(&m);
                }
            }
            Kind::Completed => {
                if let Some(t) = evt.task {
                    let m = format!(
                        r#"{{"ph":"i","name":"completed","cat":"task","ts":{:.3},"pid":1,"tid":{},"s":"t"}}"#,
                        ts,
                        task_tid(t),
                    );
                    emitter.emit(&m);
                }
            }
            Kind::SpanEnter {
                id,
                name,
                target,
                parent,
                fields,
            } => {
                span_start.insert(
                    id,
                    SpanState {
                        name,
                        target,
                        parent,
                        fields,
                        start_ns: evt.ts_ns,
                        task: evt.task,
                    },
                );
            }
            Kind::SpanExit { id } => {
                if let Some(s) = span_start.remove(&id) {
                    let dur_us = ((evt.ts_ns - s.start_ns) as f64) / 1000.0;
                    let start_us = to_us(s.start_ns, anchor);
                    // Tid: prefer the originating task; fall back to a
                    // generic "spans" lane (tid 9999) if no task was
                    // attributed.
                    let tid = s.task.map(task_tid).unwrap_or(9999);
                    let parent_arg = s
                        .parent
                        .map(|p| format!(r#","parent_span":{p}"#))
                        .unwrap_or_default();
                    let fields_arg = if s.fields.is_empty() {
                        String::new()
                    } else {
                        format!(r#","fields":"{}""#, json_escape(&s.fields))
                    };
                    let m = format!(
                        r#"{{"ph":"X","name":"{}","cat":"span","ts":{:.3},"dur":{:.3},"pid":1,"tid":{},"args":{{"target":"{}"{}{}}}}}"#,
                        json_escape(&s.name),
                        start_us,
                        dur_us,
                        tid,
                        json_escape(&s.target),
                        parent_arg,
                        fields_arg,
                    );
                    emitter.emit(&m);
                }
            }
            Kind::SpanEvent {
                target,
                level,
                msg,
                in_span: _,
            } => {
                let tid = evt.task.map(task_tid).unwrap_or(9999);
                let m = format!(
                    r#"{{"ph":"i","name":"log","cat":"log","ts":{:.3},"pid":1,"tid":{},"s":"t","args":{{"level":"{}","target":"{}","msg":"{}"}}}}"#,
                    ts,
                    tid,
                    level,
                    json_escape(&target),
                    json_escape(&msg),
                );
                emitter.emit(&m);
            }
            Kind::SpanAllocs {
                bytes,
                count: _,
                id: _,
            } => {
                alloc_total_bytes = alloc_total_bytes.saturating_add(bytes);
                let m = format!(
                    r#"{{"ph":"C","name":"alloc_bytes","ts":{:.3},"pid":1,"args":{{"bytes":{}}}}}"#,
                    ts, alloc_total_bytes,
                );
                emitter.emit(&m);
            }
            Kind::User { cat, detail } => {
                let tid = evt.task.map(task_tid).unwrap_or(9999);
                let m = format!(
                    r#"{{"ph":"i","name":"{}","cat":"user","ts":{:.3},"pid":1,"tid":{},"s":"t","args":{{"detail":"{}"}}}}"#,
                    json_escape(&cat),
                    ts,
                    tid,
                    json_escape(&detail),
                );
                emitter.emit(&m);
            }
            Kind::Control { msg } => {
                let m = format!(
                    r#"{{"ph":"i","name":"control","cat":"control","ts":{:.3},"pid":1,"tid":0,"s":"g","args":{{"msg":"{}"}}}}"#,
                    ts,
                    json_escape(&msg),
                );
                emitter.emit(&m);
            }
            _ => {}
        }
    }

    emitter.finalize();
    eprintln!("[cf-trace-to-perfetto] done");
}

/// Wraps the output writer to handle the Chrome-trace JSON envelope:
/// opens with `{"traceEvents":[`, separates events with commas, closes
/// with `]}`. Avoids `serde_json` so the converter is single-file and
/// pulls zero deps.
struct Emitter<W: Write> {
    w: W,
    started: bool,
    finalized: bool,
}

impl<W: Write> Emitter<W> {
    fn new(mut w: W) -> Self {
        w.write_all(b"{\"traceEvents\":[\n").unwrap();
        Self {
            w,
            started: false,
            finalized: false,
        }
    }
    fn emit(&mut self, obj: &str) {
        if self.started {
            self.w.write_all(b",\n").unwrap();
        }
        self.started = true;
        self.w.write_all(obj.as_bytes()).unwrap();
    }
    fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        self.w.write_all(b"\n]}\n").unwrap();
        self.w.flush().unwrap();
    }
}

impl<W: Write> Drop for Emitter<W> {
    fn drop(&mut self) {
        self.finalize();
    }
}

fn task_tid(task_id: u64) -> u64 {
    10_000 + task_id
}

#[derive(Debug)]
struct SpanState {
    name: String,
    target: String,
    parent: Option<u64>,
    fields: String,
    start_ns: u64,
    task: Option<u64>,
}

#[derive(Debug)]
struct ParsedEvent {
    seq: u64,
    ts_ns: u64,
    task: Option<u64>,
    worker: Option<u64>,
    kind: Kind,
}

#[derive(Debug)]
enum Kind {
    Spawned { name: String, parent: Option<u64> },
    PollStart,
    PollEnd { resched: bool, d_ns: u64 },
    Wake { from_task: Option<u64> },
    State { from: String, to: String },
    Completed,
    SpanEnter { id: u64, name: String, target: String, parent: Option<u64>, fields: String },
    SpanExit { id: u64 },
    SpanEvent { target: String, level: String, msg: String, in_span: Option<u64> },
    SpanAllocs { id: u64, bytes: i64, count: i64 },
    User { cat: String, detail: String },
    Control { msg: String },
    Other,
}

/// Bare-bones JSON line parser tuned to our writer's exact shape.
/// Avoids serde_json — the JSONL has a fixed schema we control, and
/// pulling serde into a tiny converter is overkill.
fn parse(line: &str) -> Option<ParsedEvent> {
    // Each line is: {"seq":N,"ts_ns":N[,"task":N][,"worker":N],"kind":{...}}
    // We use a tolerant scanner.
    let seq = json_num(line, "seq")?;
    let ts_ns = json_num(line, "ts_ns")?;
    let task = json_num(line, "task");
    let worker = json_num(line, "worker");
    // Find the kind object.
    let kind_start = line.find(r#""kind":"#)? + r#""kind":"#.len();
    let kind_str = &line[kind_start..];
    let k = json_str(kind_str, "k")?;
    let kind = match k.as_str() {
        "spawn" => Kind::Spawned {
            name: json_str(kind_str, "name").unwrap_or_default(),
            parent: json_num(kind_str, "parent"),
        },
        "poll_start" => Kind::PollStart,
        "poll_end" => Kind::PollEnd {
            resched: json_bool(kind_str, "resched").unwrap_or(false),
            d_ns: json_num(kind_str, "d_ns").unwrap_or(0),
        },
        "wake" => Kind::Wake {
            from_task: json_num(kind_str, "from_task"),
        },
        "state" => Kind::State {
            from: json_str(kind_str, "from").unwrap_or_default(),
            to: json_str(kind_str, "to").unwrap_or_default(),
        },
        "completed" => Kind::Completed,
        "span_enter" => Kind::SpanEnter {
            id: json_num(kind_str, "id").unwrap_or(0),
            name: json_str(kind_str, "name").unwrap_or_default(),
            target: json_str(kind_str, "target").unwrap_or_default(),
            parent: json_num(kind_str, "parent"),
            fields: json_str(kind_str, "fields").unwrap_or_default(),
        },
        "span_exit" => Kind::SpanExit {
            id: json_num(kind_str, "id").unwrap_or(0),
        },
        "span_event" => Kind::SpanEvent {
            target: json_str(kind_str, "target").unwrap_or_default(),
            level: json_str(kind_str, "level").unwrap_or_default(),
            msg: json_str(kind_str, "msg").unwrap_or_default(),
            in_span: json_num(kind_str, "in_span"),
        },
        "span_allocs" => Kind::SpanAllocs {
            id: json_num(kind_str, "id").unwrap_or(0),
            bytes: json_inum(kind_str, "bytes").unwrap_or(0),
            count: json_inum(kind_str, "count").unwrap_or(0),
        },
        "user" => Kind::User {
            cat: json_str(kind_str, "cat").unwrap_or_default(),
            detail: json_str(kind_str, "detail").unwrap_or_default(),
        },
        "control" => Kind::Control {
            msg: json_str(kind_str, "msg").unwrap_or_default(),
        },
        _ => Kind::Other,
    };
    Some(ParsedEvent {
        seq,
        ts_ns,
        task,
        worker,
        kind,
    })
}

/// Find `"key":N` and parse N as u64.
fn json_num(s: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\":");
    let pos = s.find(&needle)? + needle.len();
    let rest = &s[pos..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    rest[..end].parse().ok()
}

/// Find `"key":N` where N may be negative. Parse as i64.
fn json_inum(s: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{key}\":");
    let pos = s.find(&needle)? + needle.len();
    let rest = &s[pos..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '-'))
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    rest[..end].parse().ok()
}

fn json_bool(s: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\":");
    let pos = s.find(&needle)? + needle.len();
    let rest = &s[pos..];
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Find `"key":"value"` and decode value (with our writer's escapes).
fn json_str(s: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let pos = s.find(&needle)? + needle.len();
    let rest = &s[pos..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    // \uXXXX
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(n) = u32::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(n) {
                            out.push(c);
                        }
                    }
                }
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    None
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
