//! Generates `samples/complex.chrome.json` — a realistic-looking webserver-style
//! trace with deep nesting, multiple categories, repeated quick calls, and
//! occasional outliers. Run from the workspace root:
//!
//!     cargo run -p flame-viewer --example gen_complex
//!
//! Roughly 12k events, depth up to 14, ~50ms total wall time at µs resolution.

use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Clone, Copy)]
struct Cat(&'static str);
const APP: Cat = Cat("app");
const IO: Cat = Cat("io");
const CPU: Cat = Cat("cpu");
const GPU: Cat = Cat("gpu");
const DB: Cat = Cat("db");
const NET: Cat = Cat("net");
const ALLOC: Cat = Cat("alloc");

struct Gen<W: Write> {
    out: W,
    t_us: f64,
    pid: i64,
    tid: i64,
    first: bool,
    rng: u64,
}

impl<W: Write> Gen<W> {
    fn new(out: W, pid: i64, tid: i64) -> Self {
        Self { out, t_us: 0.0, pid, tid, first: true, rng: 0xC0FFEE_u64 }
    }

    /// xorshift — deterministic so the trace is reproducible.
    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    fn rand_range(&mut self, lo: f64, hi: f64) -> f64 {
        let r = (self.rand() as f64) / (u64::MAX as f64);
        lo + r * (hi - lo)
    }

    fn comma(&mut self) {
        if self.first {
            self.first = false;
        } else {
            writeln!(self.out, ",").unwrap();
        }
    }

    fn meta(&mut self, name: &str, key: &str, value: &str) {
        self.comma();
        write!(
            self.out,
            r#"{{"name":"{name}","ph":"M","pid":{},"tid":{},"args":{{"{}":"{}"}}}}"#,
            self.pid, self.tid, key, value
        )
        .unwrap();
    }

    fn instant(&mut self, name: &str, cat: Cat) {
        self.comma();
        write!(
            self.out,
            r#"{{"name":"{name}","cat":"{}","ph":"i","ts":{},"pid":{},"tid":{}}}"#,
            cat.0, self.t_us, self.pid, self.tid
        )
        .unwrap();
    }

    fn complete(&mut self, name: &str, cat: Cat, dur_us: f64) {
        self.comma();
        write!(
            self.out,
            r#"{{"name":"{}","cat":"{}","ph":"X","ts":{},"dur":{},"pid":{},"tid":{}}}"#,
            name, cat.0, self.t_us, dur_us, self.pid, self.tid
        )
        .unwrap();
        self.t_us += dur_us;
    }

    fn begin(&mut self, name: &str, cat: Cat) {
        self.comma();
        write!(
            self.out,
            r#"{{"name":"{name}","cat":"{}","ph":"B","ts":{},"pid":{},"tid":{}}}"#,
            cat.0, self.t_us, self.pid, self.tid
        )
        .unwrap();
    }

    fn end(&mut self) {
        self.comma();
        write!(
            self.out,
            r#"{{"ph":"E","ts":{},"pid":{},"tid":{}}}"#,
            self.t_us, self.pid, self.tid
        )
        .unwrap();
    }

    fn advance(&mut self, dur_us: f64) {
        self.t_us += dur_us;
    }

    /// Convenience: emit a random-duration complete slice in one borrow.
    fn complete_rand(&mut self, name: &str, cat: Cat, lo: f64, hi: f64) {
        let dur = self.rand_range(lo, hi);
        self.complete(name, cat, dur);
    }

    fn advance_rand(&mut self, lo: f64, hi: f64) {
        let dur = self.rand_range(lo, hi);
        self.advance(dur);
    }

    /// Reset to t=0 on a new tid, ready to lay down a second thread's events.
    /// Chrome JSON allows out-of-order events (each carries its own ts).
    fn switch_thread(&mut self, tid: i64) {
        self.tid = tid;
        self.t_us = 0.0;
    }
}

fn db_query<W: Write>(g: &mut Gen<W>, label: &str, depth: u8) {
    g.begin(label, DB);
    g.advance_rand(0.5, 4.0);
    g.complete_rand("plan", CPU, 0.2, 1.5);
    g.complete_rand("acquire_conn", NET, 0.1, 0.8);
    // Some queries fan out into many tiny lookups.
    let n_lookups = (g.rand_range(3.0, 24.0) as usize).max(1);
    for i in 0..n_lookups {
        g.complete_rand(if i % 3 == 0 { "btree_seek" } else { "page_read" }, IO, 0.05, 0.4);
    }
    let recurse = g.rand_range(0.0, 1.0);
    if depth > 0 && recurse > 0.7 {
        db_query(g, "subquery", depth - 1);
    }
    g.complete_rand("decode_rows", CPU, 0.4, 2.0);
    g.advance_rand(0.1, 0.5);
    g.end();
}

fn render<W: Write>(g: &mut Gen<W>) {
    g.begin("render_frame", GPU);
    g.advance(0.2);

    g.begin("layout", CPU);
    for _ in 0..4 {
        g.complete_rand("measure_node", CPU, 0.05, 0.3);
    }
    g.complete_rand("flex_solve", CPU, 0.5, 1.6);
    g.complete_rand("text_shape", CPU, 0.3, 1.2);
    g.advance(0.1);
    g.end();

    g.begin("paint", GPU);
    g.complete("clear", GPU, 0.05);
    let n_draws = (g.rand_range(8.0, 24.0) as usize).max(1);
    for _ in 0..n_draws {
        g.complete_rand("draw_call", GPU, 0.05, 0.4);
    }
    g.complete_rand("compose", GPU, 0.4, 1.1);
    g.advance(0.1);
    g.end();

    g.complete_rand("vsync_wait", GPU, 0.5, 2.5);
    g.end();
}

fn alloc_burst<W: Write>(g: &mut Gen<W>) {
    g.begin("alloc_burst", ALLOC);
    let n = (g.rand_range(20.0, 60.0) as usize).max(1);
    for _ in 0..n {
        g.complete_rand("malloc", ALLOC, 0.02, 0.15);
    }
    g.complete_rand("gc_scan", ALLOC, 1.0, 3.5);
    g.complete_rand("gc_sweep", ALLOC, 0.5, 2.0);
    g.advance(0.05);
    g.end();
}

fn handle_request<W: Write>(g: &mut Gen<W>, route: &str) {
    g.begin("handle_request", APP);
    g.instant("request_received", NET);
    g.advance(0.3);

    g.begin("parse_http", NET);
    g.complete_rand("read_headers", IO, 0.2, 0.8);
    g.complete_rand("read_body", IO, 0.4, 2.5);
    g.complete_rand("validate", CPU, 0.1, 0.6);
    g.advance(0.1);
    g.end();

    g.begin("auth", APP);
    g.complete_rand("decode_jwt", CPU, 0.3, 1.2);
    db_query(g, "lookup_user", 1);
    g.complete_rand("check_perms", CPU, 0.2, 0.7);
    g.advance(0.1);
    g.end();

    g.begin("route_dispatch", APP);
    g.complete_rand(route, APP, 0.05, 0.2);
    let n_db = (g.rand_range(2.0, 4.0) as usize).max(1);
    for _ in 0..n_db {
        db_query(g, "query", 2);
    }
    let burst = g.rand_range(0.0, 1.0);
    if burst > 0.6 {
        alloc_burst(g);
    }
    g.complete_rand("serialize_json", CPU, 0.6, 2.4);
    g.advance(0.1);
    g.end();

    g.begin("write_response", NET);
    g.complete_rand("encode", CPU, 0.2, 0.9);
    g.complete_rand("send", IO, 0.3, 1.4);
    g.advance(0.05);
    g.end();

    g.advance(0.2);
    g.end();
}

fn main() -> std::io::Result<()> {
    let path = "samples/complex.chrome.json";
    let f = File::create(path)?;
    let mut out = BufWriter::new(f);
    writeln!(out, r#"{{"traceEvents":["#)?;

    {
        let mut g = Gen::new(&mut out, 1, 1);

        // Process + per-thread metadata.
        g.meta("process_name", "name", "demo-server");
        g.meta("thread_name", "name", "main");
        g.switch_thread(2);
        g.meta("thread_name", "name", "worker");
        g.switch_thread(3);
        g.meta("thread_name", "name", "render");
        g.switch_thread(4);
        g.meta("thread_name", "name", "background");

        let routes = [
            "GET /index", "GET /users", "POST /login", "GET /api/items",
            "POST /api/order", "GET /static/app.js", "GET /healthz",
            "POST /upload", "GET /api/items/:id", "DELETE /api/items/:id",
        ];

        // --- main thread: HTTP request handling ---
        g.switch_thread(1);
        for i in 0..80 {
            handle_request(&mut g, routes[i % routes.len()]);
            g.advance_rand(0.1, 0.8);
        }
        let main_total = g.t_us;
        eprintln!("main:       ~{} µs", main_total as u64);

        // --- worker thread: batch jobs ---
        g.switch_thread(2);
        for i in 0..40 {
            g.begin("batch_job", APP);
            g.complete_rand("dequeue", IO, 0.2, 0.6);
            db_query(&mut g, "fetch_batch", 2);
            for _ in 0..3 {
                g.complete_rand("transform", CPU, 0.5, 2.0);
            }
            g.complete_rand("write_result", IO, 0.4, 1.5);
            g.advance(0.2);
            g.end();
            g.advance_rand(0.5, 2.0);
            if i % 9 == 8 {
                alloc_burst(&mut g);
            }
        }
        eprintln!("worker:     ~{} µs", g.t_us as u64);

        // --- render thread: paint at ~16µs intervals (pretend 60fps in scaled µs) ---
        g.switch_thread(3);
        for _ in 0..50 {
            render(&mut g);
            g.advance_rand(2.0, 4.0);
        }
        eprintln!("render:     ~{} µs", g.t_us as u64);

        // --- background thread: telemetry + GC ---
        g.switch_thread(4);
        for i in 0..30 {
            g.begin("telemetry_flush", NET);
            g.complete_rand("collect", CPU, 0.5, 1.5);
            g.complete_rand("compress", CPU, 0.8, 2.5);
            g.complete_rand("send", NET, 1.0, 4.0);
            g.advance(0.1);
            g.end();
            g.advance_rand(1.0, 3.0);
            if i % 5 == 4 {
                alloc_burst(&mut g);
            }
        }
        eprintln!("background: ~{} µs", g.t_us as u64);
    }

    writeln!(out, "\n]}}")?;
    eprintln!("wrote {path}");
    Ok(())
}
