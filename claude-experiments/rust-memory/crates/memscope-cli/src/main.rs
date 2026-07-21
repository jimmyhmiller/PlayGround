//! `memscope` CLI — a terminal consumer for the in-process agent.
//!
//! Subcommands:
//!   monitor [--sock P] [--interval MS]   live, type-resolved heap monitor
//!   dump    [--sock P] [--out FILE]       one type-resolved heap dump
//!   events  [--sock P]                    raw allocation event stream
//!   mode    <off|full|sampled> [--rate N] switch the agent's recording mode
//!   show    <FILE>                        explore a saved dump posthoc (no agent)
//!
//! The agent prints its socket path on startup (default /tmp/memscope-<pid>.sock,
//! or $MEMSCOPE_SOCK). Pass it with --sock, or omit to auto-detect a single
//! running agent's socket in /tmp.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

use memscope_proto::{ClientMsg, HeapGraph, ServerMsg, Snapshot};
use memscope_replay::{
    analyze, boundary_frame, clean_frame, clean_type_name, frame_location, is_std_frame,
    label_for, read_recording, read_recording_raw, site_stats, stream_events, Finding,
    FrameMeta, RecEvent, Recording,
    SiteStats, Timeline,
};

// Read-time symbolication is allocation-heavy; mimalloc returns freed memory to
// the OS more readily than the system allocator (which retains it in magazines).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let rest = &args[args.len().min(1)..];

    let result = match cmd {
        "monitor" => cmd_monitor(rest),
        "dump" => cmd_dump(rest),
        "events" => cmd_events(rest),
        "mode" => cmd_mode(rest),
        "show" => cmd_show(rest),
        "graph" => cmd_graph(rest),
        "paths" => cmd_paths(rest),
        "replay" => cmd_replay(rest),
        "perfetto" => cmd_perfetto(rest),
        "flamegraph" => cmd_flamegraph(rest),
        "flamechart" => cmd_flamechart(rest),
        "marks" => cmd_marks(rest),
        "diff" => cmd_diff(rest),
        "analyze" => cmd_analyze(rest),
        "query" => cmd_query(rest),
        "run" => cmd_run(rest),
        "dump-pid" => cmd_dump_pid(rest),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => {
            eprintln!("unknown command: {other}\n");
            print_help();
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn print_help() {
    eprintln!(
        "memscope — JVM-style memory tooling for Rust\n\n\
         USAGE:\n  \
         memscope monitor [--sock P] [--interval MS]   live type-resolved heap monitor\n  \
         memscope dump    [--sock P] [--out FILE]       one type-resolved heap dump\n  \
         memscope events  [--sock P]                    raw allocation event stream\n  \
         memscope mode    <off|full|sampled> [--rate N] switch recording mode\n  \
         memscope show    <FILE>                        explore a saved dump posthoc\n  \
         memscope graph   [--sock P] [--limit N]        heap reference graph: top retainers\n  \
         memscope paths   <hexaddr> [--sock P]          who retains/references an allocation\n  \
         memscope replay  <FILE>                        read a record_to_file recording\n  \
         memscope perfetto <FILE> [--out trace.json]    convert a recording to a Perfetto trace\n  \
         memscope flamegraph <FILE> [--format chrome|folded] [--by bytes|count] [--live] [--no-std]\n  \
         \x20                       [--group-by KEY] [--filter KEY=VAL]   pivot/filter by meta!() metadata\n  \
         \x20                                            allocation flame graph by call stack (aggregated)\n  \
         memscope flamechart <FILE> [--no-std]          allocation flame CHART (full timeline, every allocation)\n  \
         \x20                                            (--no-std strips std/core/alloc/runtime frames)\n  \
         memscope marks   <FILE> [--json]               list checkpoints (memscope::mark) + heap size at each\n  \
         memscope diff    <FILE> <A> <B> [--json]       diff the live set between two checkpoints (A->B)\n  \
         \x20                                            A/B are mark labels, or `start`/`end` for the stream ends\n  \
         memscope analyze <FILE> [--json] [--top N]     ranked memory findings (leaks/churn/realloc/short-lived)\n  \
         memscope query   <FILE> (--site N | --type T) [--field stack|lifetimes|stats|sites] [--json]\n  \
         \x20                                            drill into one finding: call stack, lifetime histogram, stats\n  \
         memscope run     [--out PATH] [--on-exit] [--after DUR] [--at-bytes N] -- <prog> [args...]\n  \
         \x20                                            run an UNMODIFIED binary under memscope, dump its heap to .hprof\n  \
         \x20                                            (no DYLD/LD_PRELOAD setup needed; DUR e.g. 2s/500ms, N e.g. 5MB)\n  \
         memscope dump-pid <PID>                        trigger a heap dump in a process started by `memscope run`\n"
    );
}

// --- argument helpers --------------------------------------------------------

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn positional(args: &[String]) -> Option<&str> {
    args.iter().find(|a| !a.starts_with("--")).map(String::as_str)
}

/// Find the socket: explicit --sock, else $MEMSCOPE_SOCK, else the single
/// **live** `/tmp/memscope-*.sock`. Agents name their socket after their pid;
/// a socket whose process is gone is a leftover (nothing unlinks it on crash or
/// kill), so it's skipped — and removed, so it can't shadow the next session.
fn resolve_sock(args: &[String]) -> Result<String, String> {
    if let Some(s) = flag(args, "--sock") {
        return Ok(s.to_string());
    }
    if let Ok(s) = std::env::var("MEMSCOPE_SOCK") {
        return Ok(s);
    }
    let mut live = Vec::new();
    let mut stale = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/tmp") {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("memscope-") && name.ends_with(".sock") {
                let path = e.path().to_string_lossy().into_owned();
                // Keep sockets we can't attribute to a pid (custom names): we
                // can't prove them dead, and we must never delete those.
                match socket_pid(&name) {
                    Some(pid) if !pid_alive(pid) => stale.push(path),
                    _ => live.push(path),
                }
            }
        }
    }
    for s in &stale {
        let _ = std::fs::remove_file(s);
    }
    if !stale.is_empty() {
        eprintln!(
            "[cleaned up {} stale socket(s) from exited processes: {}]",
            stale.len(),
            stale.join(", ")
        );
    }
    match live.len() {
        1 => Ok(live.pop().unwrap()),
        0 => Err(
            "no live agent socket found; is the traced process running? \
             (pass --sock <path> — the agent prints it on start)"
                .into(),
        ),
        _ => Err(format!(
            "multiple live agent sockets found, pass --sock <path>:\n  {}",
            live.join("\n  ")
        )),
    }
}

/// The pid embedded in an agent's default socket name, `memscope-<pid>.sock`.
fn socket_pid(name: &str) -> Option<i32> {
    name.strip_prefix("memscope-")?.strip_suffix(".sock")?.parse().ok()
}

/// Does a process with this pid exist? (`kill(pid, 0)`: EPERM still means it
/// exists, just isn't ours.)
fn pid_alive(pid: i32) -> bool {
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

// --- protocol client ---------------------------------------------------------

struct Client {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Client {
    fn connect(path: &str) -> Result<Self, String> {
        Self::connect_io(path).map_err(|e| {
            format!(
                "could not connect to agent socket {path}: {e} \
                 (is the traced process still running?)"
            )
        })
    }

    fn connect_io(path: &str) -> std::io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        let reader = BufReader::new(stream.try_clone()?);
        let mut c = Client { reader, writer: stream };
        // Consume the Hello.
        if let Ok(ServerMsg::Hello { pid, agent_version }) = c.recv() {
            eprintln!("[connected to agent pid {pid}, v{agent_version}]");
        }
        Ok(c)
    }

    fn send(&mut self, msg: &ClientMsg) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(msg).map_err(std::io::Error::other)?;
        line.push(b'\n');
        self.writer.write_all(&line)?;
        self.writer.flush()
    }

    fn recv(&mut self) -> std::io::Result<ServerMsg> {
        let mut line = String::new();
        let n = self.reader.read_line(&mut line)?;
        if n == 0 {
            return Err(std::io::Error::other("agent closed connection"));
        }
        serde_json::from_str(line.trim_end()).map_err(std::io::Error::other)
    }

    fn snapshot(&mut self) -> std::io::Result<Snapshot> {
        self.send(&ClientMsg::GetSnapshot)?;
        loop {
            match self.recv()? {
                ServerMsg::Snapshot(s) => return Ok(*s),
                ServerMsg::Error(e) => eprintln!("[agent] {e}"),
                _ => continue,
            }
        }
    }

    fn stats(&mut self) -> std::io::Result<memscope_proto::StatsView> {
        self.send(&ClientMsg::GetStats)?;
        loop {
            match self.recv()? {
                ServerMsg::Stats(s) => return Ok(s),
                ServerMsg::Error(e) => eprintln!("[agent] {e}"),
                _ => continue,
            }
        }
    }

    fn graph(&mut self) -> std::io::Result<HeapGraph> {
        self.send(&ClientMsg::GetGraph)?;
        loop {
            match self.recv()? {
                ServerMsg::Graph(g) => return Ok(*g),
                ServerMsg::Error(e) => eprintln!("[agent] {e}"),
                _ => continue,
            }
        }
    }
}

// --- formatting --------------------------------------------------------------

fn human_bytes(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", U[i])
    }
}

fn mode_name(m: u8) -> &'static str {
    match m {
        1 => "Full",
        2 => "Sampled",
        _ => "Off",
    }
}

/// Aggregate a snapshot's live set by recovered type; returns rows sorted by
/// bytes desc: (label, count, bytes).
fn by_type(snap: &Snapshot) -> Vec<(String, u64, u64)> {
    let type_name: HashMap<u32, &str> =
        snap.types.iter().map(|t| (t.id, t.name.as_str())).collect();
    let site: HashMap<u32, (&memscope_proto::SiteInfo,)> =
        snap.sites.iter().map(|s| (s.id, (s,))).collect();

    // Key = a readable label combining shape + type (falls back to site id).
    let mut agg: HashMap<String, (u64, u64)> = HashMap::new();
    for l in &snap.live {
        let label = if let Some((s,)) = site.get(&l.site.0) {
            let ty = type_name.get(&s.ty.0).copied();
            match (s.shape, ty) {
                (Some(shape), Some(ty)) => format!("{shape:?}<{}>", clean_type_name(ty)),
                (Some(shape), None) => format!("{shape:?}<?>"),
                (None, Some(ty)) => clean_type_name(ty),
                (None, None) => format!("<site {}>", l.site.0),
            }
        } else {
            "<no site>".to_string()
        };
        let e = agg.entry(label).or_default();
        e.0 += 1;
        e.1 += l.size;
    }
    let mut rows: Vec<(String, u64, u64)> = agg.into_iter().map(|(k, (c, b))| (k, c, b)).collect();
    rows.sort_by_key(|(_, _, b)| std::cmp::Reverse(*b));
    rows
}

fn print_type_table(snap: &Snapshot, limit: usize) {
    let rows = by_type(snap);
    let scale = snap.sample_scale;
    println!(
        "{:>8}  {:>12}  {}",
        "count", "bytes", "type (shape<element>)"
    );
    println!("{}", "─".repeat(64));
    for (label, count, bytes) in rows.iter().take(limit) {
        let (c, b) = if scale > 1.0 {
            (((*count as f64) * scale) as u64, ((*bytes as f64) * scale) as u64)
        } else {
            (*count, *bytes)
        };
        println!("{:>8}  {:>12}  {}", c, human_bytes(b), label);
    }
    if rows.len() > limit {
        println!("  … and {} more types", rows.len() - limit);
    }
}

// --- commands ----------------------------------------------------------------

fn cmd_monitor(args: &[String]) -> Result<(), String> {
    let sock = resolve_sock(args)?;
    let interval_ms: u64 = flag(args, "--interval")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let mut client = Client::connect(&sock)?;

    loop {
        let stats = client.stats().map_err(|e| e.to_string())?;
        let snap = client.snapshot().map_err(|e| e.to_string())?;
        // Clear screen, home cursor.
        print!("\x1b[2J\x1b[H");
        println!("memscope live monitor — {sock}");
        println!(
            "mode={}  rate={}  live={}  total allocs={}  total alloc'd={}  dropped events={}",
            mode_name(stats.mode),
            stats.sample_rate,
            human_bytes(stats.live_bytes),
            stats.total_allocs,
            human_bytes(stats.total_alloc_bytes),
            stats.dropped_events,
        );
        println!(
            "live allocations: {}   distinct sites: {}   (sample scale {:.0})\n",
            snap.live.len(),
            snap.sites.len(),
            snap.sample_scale,
        );
        print_type_table(&snap, 20);
        println!("\n(refreshing every {interval_ms} ms — Ctrl-C to quit)");
        let _ = std::io::stdout().flush();
        std::thread::sleep(std::time::Duration::from_millis(interval_ms));
    }
}

fn cmd_dump(args: &[String]) -> Result<(), String> {
    let sock = resolve_sock(args)?;
    let mut client = Client::connect(&sock)?;
    let snap = client.snapshot().map_err(|e| e.to_string())?;
    render_dump(&snap);
    if let Some(out) = flag(args, "--out") {
        let json = serde_json::to_vec_pretty(&snap).map_err(|e| e.to_string())?;
        std::fs::write(out, json).map_err(|e| e.to_string())?;
        eprintln!("\n[saved type-resolved dump to {out} — explore later with `memscope show {out}`]");
    }
    Ok(())
}

fn cmd_show(args: &[String]) -> Result<(), String> {
    let file = positional(args).ok_or("usage: memscope show <FILE>")?;
    let bytes = std::fs::read(file).map_err(|e| format!("{file}: {e}"))?;
    if bytes.starts_with(b"MSCP") {
        return Err(format!(
            "{file} is a .mscope recording, not a heap dump — use `memscope replay {file}` \
             (show reads the JSON written by `memscope dump --out`)"
        ));
    }
    let snap: Snapshot = serde_json::from_slice(&bytes).map_err(|e| {
        format!("{file}: not a heap dump written by `memscope dump --out` ({e})")
    })?;
    render_dump(&snap);
    Ok(())
}

fn render_dump(snap: &Snapshot) {
    println!("== memscope heap dump ==");
    println!(
        "live allocations: {}   live bytes: {}   sites: {}   types: {}   scale: {:.0}\n",
        snap.live.len(),
        human_bytes(snap.total_live_bytes),
        snap.sites.len(),
        snap.types.len(),
        snap.sample_scale,
    );
    print_type_table(snap, 30);

    // Per-site detail with the first user-ish frame.
    println!("\n== top allocation sites ==");
    let mut site_rows: Vec<(&memscope_proto::SiteInfo, u64, u64)> = snap
        .sites
        .iter()
        .map(|s| {
            let (c, b) = snap
                .live
                .iter()
                .filter(|l| l.site.0 == s.id)
                .fold((0u64, 0u64), |(c, b), l| (c + 1, b + l.size));
            (s, c, b)
        })
        .collect();
    site_rows.sort_by_key(|(_, _, b)| std::cmp::Reverse(*b));

    let type_name: HashMap<u32, &str> =
        snap.types.iter().map(|t| (t.id, t.name.as_str())).collect();

    for (s, c, b) in site_rows.iter().take(12) {
        let label = match (s.shape, type_name.get(&s.ty.0)) {
            (Some(shape), Some(ty)) => format!("{shape:?}<{}>", clean_type_name(ty)),
            (Some(shape), None) => format!("{shape:?}<?>"),
            (None, Some(ty)) => clean_type_name(ty),
            (None, None) => "<unknown>".to_string(),
        };
        println!("\n  {label}  —  {c} live, {}", human_bytes(*b));
        for f in s.frames.iter().take(6) {
            let func = f.function.as_deref().map(clean_frame).unwrap_or_else(|| "?".into());
            let loc = match (&f.file, f.line) {
                (Some(file), Some(line)) => format!("  ({}:{})", short_path(file), line),
                _ => String::new(),
            };
            let inl = if f.inlined { " [inlined]" } else { "" };
            println!("      {func}{inl}{loc}");
        }
    }
}

fn short_path(p: &str) -> &str {
    // Trim everything before the last `src/` for readability.
    if let Some(idx) = p.rfind("/src/") {
        &p[idx + 1..]
    } else {
        p.rsplit('/').next().unwrap_or(p)
    }
}

fn cmd_events(args: &[String]) -> Result<(), String> {
    let sock = resolve_sock(args)?;
    let mut client = Client::connect(&sock)?;
    eprintln!("[streaming raw events — Ctrl-C to quit]");
    loop {
        client
            .send(&ClientMsg::PollEvents { max: 256 })
            .map_err(|e| e.to_string())?;
        let evs = loop {
            match client.recv().map_err(|e| e.to_string())? {
                ServerMsg::Events(e) => break e,
                ServerMsg::Error(e) => eprintln!("[agent] {e}"),
                _ => continue,
            }
        };
        for ev in &evs {
            println!(
                "seq={:<8} {:<12} addr={:#014x} size={:<8} site={} thr={}",
                ev.seq,
                format!("{:?}", ev.kind),
                ev.addr,
                ev.size,
                ev.site.0,
                ev.thread,
            );
        }
        if evs.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

fn node_label(n: &memscope_proto::GraphNode) -> String {
    match (n.shape, n.ty.as_deref()) {
        (Some(shape), Some(ty)) => format!("{shape:?}<{}>", clean_type_name(ty)),
        (Some(shape), None) => format!("{shape:?}<?>"),
        (None, Some(ty)) => clean_type_name(ty),
        (None, None) => "<unknown>".to_string(),
    }
}

fn cmd_graph(args: &[String]) -> Result<(), String> {
    let sock = resolve_sock(args)?;
    let limit: usize = flag(args, "--limit").and_then(|s| s.parse().ok()).unwrap_or(25);
    let mut client = Client::connect(&sock)?;
    let g = client.graph().map_err(|e| e.to_string())?;
    render_graph(&g, limit);
    Ok(())
}

fn render_graph(g: &HeapGraph, limit: usize) {
    let n_roots = g.roots.len();
    println!("== heap reference graph ==");
    println!(
        "nodes: {}   edges: {}   roots: {}   opaque(unwalked): {}   total: {}\n",
        g.nodes.len(),
        g.edges.len(),
        n_roots,
        g.opaque_nodes,
        human_bytes(g.total_bytes),
    );

    // Top retainers: nodes whose subtree (dominated set) is largest. These are
    // the "if this died, N bytes would be freed" candidates — the leak suspects.
    let mut idx: Vec<usize> = (0..g.nodes.len()).collect();
    idx.sort_by_key(|&i| std::cmp::Reverse(g.nodes[i].retained_size));
    println!("TOP RETAINERS (retained = bytes freed if this allocation died):");
    println!(
        "  {:>12}  {:>10}  {:>4}  {:>10}  {}",
        "retained", "self", "out", "addr", "type"
    );
    println!("  {}", "─".repeat(78));
    for &i in idx.iter().take(limit) {
        let nd = &g.nodes[i];
        println!(
            "  {:>12}  {:>10}  {:>4}  {:#012x}  {}",
            human_bytes(nd.retained_size),
            human_bytes(nd.size),
            nd.out_degree,
            nd.addr,
            node_label(nd),
        );
    }

    // Aggregate retained by type for a "dominator-by-type" summary, counting
    // only top-level (root) retainers to avoid double counting nested subtrees.
    let mut by_type: HashMap<String, (u64, u64)> = HashMap::new();
    for &r in &g.roots {
        let nd = &g.nodes[r as usize];
        let e = by_type.entry(node_label(nd)).or_default();
        e.0 += 1;
        e.1 += nd.retained_size;
    }
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
    println!("\nRETAINED BY ROOT TYPE (top-level owners):");
    println!("  {:>8}  {:>12}  {}", "count", "retained", "type");
    for (label, (count, bytes)) in rows.iter().take(15) {
        println!("  {count:>8}  {:>12}  {label}", human_bytes(*bytes));
    }
    println!("\n(use `memscope paths <hexaddr>` to see what retains a specific allocation)");
}

fn cmd_paths(args: &[String]) -> Result<(), String> {
    let addr_s = positional(args).ok_or("usage: memscope paths <hexaddr>")?;
    let addr = parse_hex(addr_s).ok_or_else(|| format!("bad address '{addr_s}' (use hex, e.g. 0x10abc)"))?;
    let sock = resolve_sock(args)?;
    let mut client = Client::connect(&sock)?;
    let g = client.graph().map_err(|e| e.to_string())?;

    let Some(i) = g.nodes.iter().position(|nd| nd.addr == addr) else {
        return Err(format!("{addr:#x} is not a tracked live allocation"));
    };
    let nd = &g.nodes[i];
    println!("{:#012x}  {}", nd.addr, node_label(nd));
    println!(
        "  self {}   retained {}   in-degree {}   out-degree {}",
        human_bytes(nd.size),
        human_bytes(nd.retained_size),
        nd.in_degree,
        nd.out_degree,
    );

    // Dominator chain upward: the unique ownership path that, if broken, frees
    // this allocation.
    println!("\nDOMINATOR CHAIN (who exclusively keeps it alive, nearest first):");
    let mut cur = nd.idom;
    let mut steps = 0;
    while cur >= 0 && steps < 64 {
        let p = &g.nodes[cur as usize];
        println!("  ← {:#012x}  {}", p.addr, node_label(p));
        cur = p.idom;
        steps += 1;
    }
    if nd.idom < 0 {
        println!("  (this allocation is itself a root — referenced by no tracked allocation)");
    } else {
        println!("  ← <roots>");
    }

    // Direct referrers (all in-edges), which may exceed the single dominator.
    let referrers: Vec<&memscope_proto::GraphEdge> =
        g.edges.iter().filter(|e| e.to as usize == i).collect();
    println!("\nDIRECT REFERRERS ({}):", referrers.len());
    for e in referrers.iter().take(20) {
        let from = &g.nodes[e.from as usize];
        println!(
            "  {:#012x} +{:#x}  {}",
            from.addr,
            e.offset,
            node_label(from)
        );
    }
    if referrers.len() > 20 {
        println!("  … and {} more", referrers.len() - 20);
    }
    Ok(())
}

/// Reference reader for a `record_to_file` recording. Detects the format
/// (compact binary `.mscope` vs newline-JSON), replays the event stream,
/// reconstructs the live set, and summarizes it — a starting point for viewers.
/// `memscope replay <FILE>` — summarize a recording: event counts, peak/final
/// live heap, and the final live set by recovered type.
///
/// Streams both formats through the shared reader, so it costs one pass and
/// `O(live set)` memory whatever the recording's size.
fn cmd_replay(args: &[String]) -> Result<(), String> {
    let file = positional(args).ok_or("usage: memscope replay <FILE>")?;

    // Definitions first (cheap: event payloads are seeked past), then the stream.
    let mut rec = read_recording_raw(file)?;
    rec.resolve_sites_compact();

    let mut live: HashMap<u64, (u64, u32)> = HashMap::new();
    let (mut allocs, mut frees, mut peak, mut cur) = (0u64, 0u64, 0u64, 0u64);
    let mut stream = stream_events(file)?;
    for e in stream.by_ref() {
        match e.kind {
            memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow => {
                if let Some((prev, _)) = live.insert(e.addr, (e.size, e.site)) {
                    cur = cur.saturating_sub(prev);
                }
                allocs += 1;
                cur += e.size;
                peak = peak.max(cur);
            }
            memscope_proto::EventKind::Dealloc => {
                if let Some((sz, _)) = live.remove(&e.addr) {
                    cur = cur.saturating_sub(sz);
                }
                frees += 1;
            }
            // Metadata / checkpoint markers aren't allocations.
            memscope_proto::EventKind::MetaEnter
            | memscope_proto::EventKind::MetaExit
            | memscope_proto::EventKind::Mark => {}
        }
    }
    if let Some(err) = stream.error() {
        return Err(err.to_string());
    }

    println!("== memscope recording: {file} ==");
    println!("  pid {}   exe {}", rec.pid, rec.exe);
    println!(
        "  events: {allocs} alloc, {frees} free   peak live: {}   final live: {} allocations, {}",
        human_bytes(peak),
        live.len(),
        human_bytes(cur)
    );
    print_live_by_type(&live, &rec);
    Ok(())
}

fn print_live_by_type(live: &HashMap<u64, (u64, u32)>, rec: &Recording) {
    let mut by_type: HashMap<String, (u64, u64)> = HashMap::new();
    for (sz, site) in live.values() {
        let e = by_type.entry(rec.site_label(*site).to_string()).or_default();
        e.0 += 1;
        e.1 += sz;
    }
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
    println!("\n  FINAL LIVE HEAP BY TYPE:");
    println!("  {:>8}  {:>12}  {}", "count", "bytes", "type");
    for (label, (count, bytes)) in rows.iter().take(25) {
        println!("  {count:>8}  {:>12}  {label}", human_bytes(*bytes));
    }
}

/// Parse repeated `--filter KEY=VAL` arguments.
fn parse_filters(args: &[String]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--filter" {
            if let Some(kv) = it.next() {
                if let Some((k, v)) = kv.split_once('=') {
                    out.push((k.to_string(), v.to_string()));
                }
            }
        }
    }
    out
}

/// Tracks each thread's active metadata context stack while streaming events, so
/// an allocation can be attributed to the `memscope::meta!` scope it happened in.
///
/// Replaces materializing a metadata map per event (which was one `BTreeMap` per
/// allocation, parallel to the whole stream): the stacks are `O(threads x nesting
/// depth)` and the merged map is built on demand, only for allocations that
/// survive the filters.
struct MetaTracker<'a> {
    meta: &'a HashMap<u32, Vec<(String, String)>>,
    stacks: HashMap<u32, Vec<u32>>,
}

impl<'a> MetaTracker<'a> {
    fn new(meta: &'a HashMap<u32, Vec<(String, String)>>) -> Self {
        MetaTracker { meta, stacks: HashMap::new() }
    }

    /// Feed one event; enter/exit events update the calling thread's stack.
    fn observe(&mut self, e: &RecEvent) {
        match e.kind {
            memscope_proto::EventKind::MetaEnter => {
                self.stacks.entry(e.thread).or_default().push(e.site);
            }
            memscope_proto::EventKind::MetaExit => {
                if let Some(s) = self.stacks.get_mut(&e.thread) {
                    // Pop the matching id (LIFO; tolerate slight disorder).
                    if let Some(pos) = s.iter().rposition(|&m| m == e.site) {
                        s.remove(pos);
                    } else {
                        s.pop();
                    }
                }
            }
            _ => {}
        }
    }

    /// The metadata active on `thread` right now (outer -> inner).
    fn active(&self, thread: u32) -> std::collections::BTreeMap<String, String> {
        let mut m = std::collections::BTreeMap::new();
        if let Some(s) = self.stacks.get(&thread) {
            for mid in s {
                if let Some(kvs) = self.meta.get(mid) {
                    for (k, v) in kvs {
                        m.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        m
    }
}

/// Convert a recording into a Perfetto / Chrome JSON trace.
///
/// Emits **everything** by default: an async slice for every allocation's
/// lifetime (alloc -> free, named by recovered type), plus live-byte counters for
/// the total and for *every* type. Nothing is capped, sampled, or truncated.
///
/// Counters are written **only when a value changes**. A Perfetto counter track
/// holds its value between samples, so re-stating an unchanged number is pure
/// redundancy — and it is the expensive kind: an allocation touches exactly one
/// type, so writing every type on every event costs `events × types` where the
/// information content is `events`. On a 41M-event recording that was 95 GB of
/// which ~99.5% was the same numbers restated. Skipping unchanged values
/// reconstructs a bit-identical curve (`tests/perfetto_counters.rs` pins that).
///
/// Opt out only if you deliberately want a smaller artifact: `--no-slices` drops
/// per-allocation lifetimes, `--max-slices N` caps them, `--top-types N` keeps
/// only the N largest-peaking types. All three *lose data* and say so on stderr.
///
/// Open the result at https://ui.perfetto.dev.
// `flush_counters!` writes back the values it emitted; on the final flush those
// write-backs are dead, which is inherent to the macro rather than a mistake.
#[allow(unused_assignments)]
fn cmd_perfetto(args: &[String]) -> Result<(), String> {
    let file = positional(args).ok_or(
        "usage: memscope perfetto <FILE> [--out trace.json] [--no-slices] [--max-slices N] [--top-types N]",
    )?;
    let out = flag(args, "--out").unwrap_or("trace.json").to_string();
    let want_slices = !args.iter().any(|a| a == "--no-slices");
    let max_slices: Option<usize> = flag(args, "--max-slices").and_then(|s| s.parse().ok());
    let top_types: Option<usize> = flag(args, "--top-types").and_then(|s| s.parse().ok());

    let mut rec = read_recording_raw(file)?;
    rec.resolve_sites_compact();

    // Counter tracks are keyed by *type label*, not by site: many call sites
    // allocate the same type, and Perfetto merges same-named tracks, so emitting
    // one track per site makes co-named tracks fight over a single curve. Intern
    // labels and fold every site of a type into one counter.
    let mut labels: Vec<String> = Vec::new();
    let mut label_id: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut label_of_site: HashMap<u32, u32> = HashMap::new();
    for (&site, info) in &rec.sites {
        let id = match label_id.get(info.label.as_str()) {
            Some(&id) => id,
            None => {
                let id = labels.len() as u32;
                labels.push(info.label.clone());
                label_id.insert(info.label.as_str(), id);
                id
            }
        };
        label_of_site.insert(site, id);
    }

    // Every type gets a counter unless --top-types asks for fewer. When it does,
    // rank by each type's peak live bytes (a first streaming pass — a counter
    // track has to be chosen before its samples are written).
    let dropped_types = match top_types {
        None => 0,
        Some(n) => {
            let keep = top_labels_by_peak_bytes(file, &label_of_site, labels.len(), n)?;
            let dropped = labels.len().saturating_sub(keep.len());
            let keep: std::collections::HashSet<u32> = keep.into_iter().collect();
            for id in 0..labels.len() as u32 {
                if !keep.contains(&id) {
                    labels[id as usize].clear(); // empty label == not tracked
                }
            }
            dropped
        }
    };

    use std::io::Write as _;
    let f = std::fs::File::create(&out).map_err(|e| e.to_string())?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let us = |ts: u64| (ts as f64) / 1000.0; // ns -> µs for Chrome format

    write!(w, "{{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n").map_err(|e| e.to_string())?;
    write!(w, "{{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{{\"name\":\"heap\"}}}}")
        .map_err(|e| e.to_string())?;

    // Live allocations we're tracking, so a free can close its slice / discount
    // its bytes. Bounded by the live set, not the event count.
    struct LiveEntry {
        size: u64,
        site: u32,
        slice: Option<u64>,
    }
    let mut live: HashMap<u64, LiveEntry> = HashMap::new();
    let mut seen_threads: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut total: u64 = 0;
    let mut slice_id: u64 = 0;
    let mut slices_emitted = 0usize;
    let mut slices_dropped = 0usize;
    let mut events = 0u64;
    let mut last_ts = 0u64;

    // Counter state, indexed by label id. `emitted` is the value last *written*
    // to the trace; a label is written only when `cur` diverges from it. u64::MAX
    // is the "never written" sentinel, so every track opens with a real sample.
    let mut cur: Vec<u64> = vec![0; labels.len()];
    let mut emitted: Vec<u64> = vec![u64::MAX; labels.len()];
    let mut total_emitted: u64 = u64::MAX;
    // Labels touched since the last flush. Bounded by the events in one
    // timestamp, so a flush costs the changes rather than a scan of all labels.
    let mut dirty: Vec<u32> = Vec::new();
    let mut is_dirty: Vec<bool> = vec![false; labels.len()];
    let mut samples = 0u64;
    // The timestamp whose accumulated state is pending. Counters flush when the
    // clock advances, so a sample is the value at the *end* of its timestamp;
    // the format can hold one point per track per ts, so that is full fidelity.
    let mut pending: Option<u64> = None;

    // Write every counter whose value moved since it was last written, at `ts`.
    // Unchanged tracks are skipped: Perfetto holds a counter's value until the
    // next sample, so re-stating it adds bytes and no information.
    macro_rules! flush_counters {
        ($ts:expr) => {{
            let ts = $ts;
            if total != total_emitted {
                write!(
                    w,
                    ",\n{{\"ph\":\"C\",\"name\":\"live_bytes\",\"ts\":{:.3},\"pid\":1,\"args\":{{\"bytes\":{}}}}}",
                    us(ts), total
                )
                .map_err(|e| e.to_string())?;
                total_emitted = total;
                samples += 1;
            }
            for id in dirty.drain(..) {
                let i = id as usize;
                is_dirty[i] = false;
                if labels[i].is_empty() || cur[i] == emitted[i] {
                    continue; // untracked (--top-types), or value did not move
                }
                write!(
                    w,
                    ",\n{{\"ph\":\"C\",\"name\":\"live: {}\",\"ts\":{:.3},\"pid\":1,\"args\":{{\"bytes\":{}}}}}",
                    esc(&labels[i]), us(ts), cur[i]
                )
                .map_err(|e| e.to_string())?;
                emitted[i] = cur[i];
                samples += 1;
            }
        }};
    }

    let mut stream = stream_events(file)?;
    for e in stream.by_ref() {
        events += 1;
        last_ts = e.ts_nanos.max(last_ts);
        // The clock advanced, so everything accumulated at the previous
        // timestamp is final — write it before this event mutates the state.
        match pending {
            Some(ts) if ts != e.ts_nanos => flush_counters!(ts),
            _ => {}
        }
        pending = Some(e.ts_nanos);
        if seen_threads.insert(e.thread) {
            write!(
                w,
                ",\n{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{},\"args\":{{\"name\":\"thread {}\"}}}}",
                e.thread, e.thread
            )
            .map_err(|e| e.to_string())?;
        }
        match e.kind {
            memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow => {
                total += e.size;
                if let Some(&id) = label_of_site.get(&e.site) {
                    cur[id as usize] += e.size;
                    if !is_dirty[id as usize] {
                        is_dirty[id as usize] = true;
                        dirty.push(id);
                    }
                }
                let slice = if want_slices && max_slices.is_none_or(|m| slices_emitted < m) {
                    let id = slice_id;
                    slice_id += 1;
                    slices_emitted += 1;
                    write!(
                        w,
                        ",\n{{\"ph\":\"b\",\"cat\":\"alloc\",\"name\":\"{}\",\"id\":{},\"ts\":{:.3},\"pid\":1,\"tid\":{},\"args\":{{\"size\":{},\"addr\":\"{:#x}\"}}}}",
                        esc(rec.site_label(e.site)), id, us(e.ts_nanos), e.thread, e.size, e.addr
                    )
                    .map_err(|e| e.to_string())?;
                    Some(id)
                } else {
                    if want_slices {
                        slices_dropped += 1;
                    }
                    None
                };
                live.insert(e.addr, LiveEntry { size: e.size, site: e.site, slice });
            }
            memscope_proto::EventKind::Dealloc => {
                if let Some(entry) = live.remove(&e.addr) {
                    total = total.saturating_sub(entry.size);
                    if let Some(&id) = label_of_site.get(&entry.site) {
                        let c = &mut cur[id as usize];
                        *c = c.saturating_sub(entry.size);
                        if !is_dirty[id as usize] {
                            is_dirty[id as usize] = true;
                            dirty.push(id);
                        }
                    }
                    if let Some(id) = entry.slice {
                        write!(
                            w,
                            ",\n{{\"ph\":\"e\",\"cat\":\"alloc\",\"name\":\"\",\"id\":{},\"ts\":{:.3},\"pid\":1,\"tid\":{}}}",
                            id, us(e.ts_nanos), e.thread
                        )
                        .map_err(|e| e.to_string())?;
                    }
                }
            }
            memscope_proto::EventKind::MetaEnter
            | memscope_proto::EventKind::MetaExit
            | memscope_proto::EventKind::Mark => {}
        }
    }
    if let Some(err) = stream.error() {
        return Err(err.to_string());
    }
    // The final timestamp's state never saw a clock advance to flush it.
    if let Some(ts) = pending {
        flush_counters!(ts);
    }

    // Close out slices for allocations still live at the end of the trace.
    let end = us(last_ts + 1_000_000);
    for entry in live.values() {
        if let Some(id) = entry.slice {
            write!(
                w,
                ",\n{{\"ph\":\"e\",\"cat\":\"alloc\",\"name\":\"\",\"id\":{},\"ts\":{:.3},\"pid\":1,\"tid\":0}}",
                id, end
            )
            .map_err(|e| e.to_string())?;
        }
    }
    write!(w, "\n]}}").map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;

    let tracked = labels.iter().filter(|l| !l.is_empty()).count();
    println!("wrote Perfetto trace: {out}");
    println!(
        "  {events} events  ->  {slices_emitted} allocation slices + live_bytes counter \
         + {tracked} per-type counter(s), {samples} counter samples  ({} threads)",
        seen_threads.len()
    );
    // Anything lost is stated loudly: a trace that silently dropped most of the
    // data reads exactly like one that covered it.
    if dropped_types > 0 {
        println!(
            "  WARNING: --top-types kept {tracked} of {} types; {dropped_types} type(s) have NO counter \
             and their bytes are absent from the per-type tracks (live_bytes is still exact). \
             Drop the flag to emit every type.",
            labels.len()
        );
    }
    if slices_dropped > 0 {
        println!(
            "  WARNING: {slices_dropped} allocations have NO slice (--max-slices); drop the flag for all of them"
        );
    }
    if !want_slices {
        println!("  WARNING: --no-slices — per-allocation lifetimes omitted, counters only");
    }
    println!("  open it at https://ui.perfetto.dev  (Open trace file)");
    Ok(())
}

/// The `n` type labels whose live bytes peak highest, via one streaming pass.
///
/// Only used when `--top-types` explicitly asks for a truncated trace. Ranking is
/// per *label*, not per site: a type allocated from a thousand call sites holds
/// its bytes across a thousand site ids, so site-ranking buries exactly the types
/// that dominate the heap. Peaks are per label for the same reason.
fn top_labels_by_peak_bytes(
    file: &str,
    label_of_site: &HashMap<u32, u32>,
    n_labels: usize,
    n: usize,
) -> Result<Vec<u32>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut live: HashMap<u64, (u64, u32)> = HashMap::new();
    let mut cur: Vec<u64> = vec![0; n_labels];
    let mut peak: Vec<u64> = vec![0; n_labels];
    let mut stream = stream_events(file)?;
    for e in stream.by_ref() {
        match e.kind {
            memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow => {
                if let Some(&id) = label_of_site.get(&e.site) {
                    cur[id as usize] += e.size;
                    peak[id as usize] = peak[id as usize].max(cur[id as usize]);
                    live.insert(e.addr, (e.size, id));
                }
            }
            memscope_proto::EventKind::Dealloc => {
                if let Some((size, id)) = live.remove(&e.addr) {
                    let c = &mut cur[id as usize];
                    *c = c.saturating_sub(size);
                }
            }
            _ => {}
        }
    }
    if let Some(err) = stream.error() {
        return Err(err.to_string());
    }
    let mut rows: Vec<(u32, u64)> =
        peak.into_iter().enumerate().map(|(i, p)| (i as u32, p)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    rows.truncate(n);
    Ok(rows.into_iter().map(|(id, _)| id).collect())
}


/// Build a site's root→leaf path: cleaned frames (recorded innermost-first, so
/// reversed to put the outermost/app frame at the root), then the recovered type
/// as the leaf.
///
/// With `no_std`, drop stdlib/runtime plumbing but **keep the boundary frame** —
/// the first std frame entered directly from application code (e.g.
/// `Vec::with_capacity`, `<str as ToOwned>::to_owned`, `HashMap::insert`), which
/// is what tells you *how* the allocation happened. Deeper std plumbing
/// (`RawVec`, `Global::alloc_impl`, `__rust_alloc`) and the root runtime
/// (`lang_start`, `_main`) are removed.
fn site_to_path(frames: &[FrameMeta], label: &str, no_std: bool) -> Vec<String> {
    // Recorded innermost-first; reverse so the outermost/app frame is the root.
    let rev: Vec<&FrameMeta> = frames.iter().rev().collect();
    let cleaned: Vec<String> = rev.iter().map(|f| clean_frame(&f.func)).collect();
    let mut p: Vec<String> = Vec::with_capacity(cleaned.len() + 1);
    for (i, f) in cleaned.iter().enumerate() {
        let keep = if !no_std {
            true
        } else if !is_std_frame(f) {
            true // application / dependency frame
        } else {
            // A std frame: keep it only at an app -> std boundary (its caller,
            // the frame below it toward the root, is application code).
            i > 0 && !is_std_frame(&cleaned[i - 1])
        };
        if keep {
            p.push(frame_label(f, &rev[i].file, rev[i].line));
        }
    }
    p.push(format!("[{label}]"));
    p
}

/// Display name for a frame: the (cleaned) function, with the source location
/// appended when known — `heapster::scan_segment (segment.rs:142)`.
fn frame_label(func: &str, file: &str, line: u32) -> String {
    if line > 0 && !file.is_empty() {
        let base = file.rsplit('/').next().unwrap_or(file);
        format!("{func} ({base}:{line})")
    } else {
        func.to_string()
    }
}


/// A node in the allocation flame tree (merged call stacks, weighted by bytes).
/// Edges are keyed by an *interned* frame-name id, not an owned `String`: the
/// same function name recurs in millions of tree positions, so interning keeps
/// the tree to `u32` keys + each unique name stored once (the difference between
/// ~2 GB and tens of GB for a million-site recording).
#[derive(Default)]
struct Flame {
    bytes: u64,
    samples: u64,
    children: std::collections::BTreeMap<u32, Flame>,
}

/// Interns frame-name strings to stable ids for the flame tree.
#[derive(Default)]
struct Interner {
    map: HashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    fn id(&mut self, s: String) -> u32 {
        if let Some(&i) = self.map.get(&s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.clone());
        self.map.insert(s, i);
        i
    }
    fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }
}

/// Build an allocation flame graph by call stack: aggregate every allocation's
/// bytes onto its captured stack, then emit it either as folded stacks
/// (`--format folded`, universal) or as a Chrome trace of nested synchronous
/// duration events (`--format chrome`, default — width = bytes), which standard
/// flame-graph importers render directly.
fn cmd_flamegraph(args: &[String]) -> Result<(), String> {
    let file_path = positional(args).ok_or("usage: memscope flamegraph <FILE> [--out F] [--format chrome|folded] [--by bytes|count] [--live] [--no-std] [--group-by KEY] [--filter KEY=VAL]")?;
    let format = flag(args, "--format").unwrap_or("chrome");
    if format != "chrome" && format != "folded" {
        return Err(format!("unknown --format {format:?} (chrome|folded)"));
    }
    let by = flag(args, "--by").unwrap_or("bytes");
    if by != "bytes" && by != "count" {
        return Err(format!("unknown --by {by:?} (bytes|count)"));
    }
    let by_count = by == "count";
    let live_only = args.iter().any(|a| a == "--live");
    let no_std = args.iter().any(|a| a == "--no-std" || a == "--exclude-std");
    let group_by = flag(args, "--group-by").map(|s| s.to_string());
    let filters = parse_filters(args);
    let default_out = if format == "folded" {
        "alloc.folded"
    } else {
        "alloc-flamegraph.json"
    };
    let out = flag(args, "--out").unwrap_or(default_out).to_string();

    // Constant-memory: read sites unresolved, symbolicate the unique IPs once,
    // then resolve each site's frames transiently and fold them into the
    // (prefix-merging) tree — never holding every site's full stack at once.
    let rec = read_recording_raw(file_path)?;
    let unique_ips: Vec<u64> = {
        let mut set: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for ips in rec.raw_sites.values() {
            set.extend(ips.iter().copied());
        }
        set.into_iter().collect()
    };
    let resolver = memscope_symbols::SiteResolver::build(
        std::path::Path::new(&rec.exe),
        rec.slide,
        &unique_ips,
    )
    .map_err(|e| e.to_string())?;
    // Resolve one site to (frames, label), transiently (caller drops it).
    let site_frames = |site: u32| -> Option<(Vec<FrameMeta>, String)> {
        let ips = rec.raw_sites.get(&site)?;
        let r = resolver.resolve_site(ips);
        let frames: Vec<FrameMeta> = r
            .frames
            .into_iter()
            .map(|fr| FrameMeta {
                func: fr.function.unwrap_or_default(),
                file: fr.file.unwrap_or_default(),
                line: fr.line.unwrap_or(0),
            })
            .collect();
        Some((frames, label_for(r.shape, r.element_type)))
    };
    let meta_active = group_by.is_some() || !filters.is_empty();

    let mut root = Flame::default();
    let mut interner = Interner::default();
    if meta_active {
        // Per-allocation pass: each alloc carries its correlated metadata, so the
        // same call site can land under different group values.
        //
        // With `--live` we can't fold an allocation until we know it survives, so
        // its weight and group are parked in the live set (bounded by live
        // allocations) and folded at the end. Otherwise we fold as we stream.
        let mut tracker = MetaTracker::new(&rec.meta);
        let mut pending: HashMap<u64, (u32, u64, std::collections::BTreeMap<String, String>)> =
            HashMap::new();
        let mut folded: Vec<(u32, u64, std::collections::BTreeMap<String, String>)> = Vec::new();
        let mut stream = stream_events(file_path)?;
        for e in stream.by_ref() {
            tracker.observe(&e);
            match e.kind {
                memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow => {}
                memscope_proto::EventKind::Dealloc if live_only => {
                    pending.remove(&e.addr);
                    continue;
                }
                _ => continue,
            }
            let m = tracker.active(e.thread);
            if !filters.iter().all(|(k, v)| m.get(k).map(|x| x == v).unwrap_or(false)) {
                continue;
            }
            let w = if by_count { 1 } else { e.size };
            if w == 0 {
                continue;
            }
            if live_only {
                pending.insert(e.addr, (e.site, w, m));
            } else {
                folded.push((e.site, w, m));
            }
        }
        if let Some(err) = stream.error() {
            return Err(err.to_string());
        }
        for entry in pending.into_values() {
            folded.push(entry);
        }
        for (site, w, m) in folded {
            let Some((frames, label)) = site_frames(site) else { continue };
            root.bytes += w;
            root.samples += 1;
            let mut node = &mut root;
            if let Some(gk) = &group_by {
                let gv = m.get(gk).cloned().unwrap_or_else(|| "<none>".to_string());
                let id = interner.id(format!("{gk}={gv}"));
                node = node.children.entry(id).or_default();
                node.bytes += w;
                node.samples += 1;
            }
            for name in site_to_path(&frames, &label, no_std) {
                let id = interner.id(name);
                node = node.children.entry(id).or_default();
                node.bytes += w;
                node.samples += 1;
            }
        }
    } else {
        // Fast path: aggregate per site (no metadata needed) while streaming.
        let mut live: HashMap<u64, (u64, u32)> = HashMap::new(); // for --live
        let mut site_bytes: HashMap<u32, u64> = HashMap::new();
        let mut site_count: HashMap<u32, u64> = HashMap::new();
        let mut stream = stream_events(file_path)?;
        for e in stream.by_ref() {
            match e.kind {
                memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow => {
                    if live_only {
                        live.insert(e.addr, (e.size, e.site));
                    } else {
                        *site_bytes.entry(e.site).or_default() += e.size;
                        *site_count.entry(e.site).or_default() += 1;
                    }
                }
                memscope_proto::EventKind::Dealloc if live_only => {
                    live.remove(&e.addr);
                }
                _ => {}
            }
        }
        if let Some(err) = stream.error() {
            return Err(err.to_string());
        }
        if live_only {
            for (size, site) in live.values() {
                *site_bytes.entry(*site).or_default() += *size;
                *site_count.entry(*site).or_default() += 1;
            }
        }
        // Stream over sites that actually allocated; resolve each transiently.
        let counts = if by_count { &site_count } else { &site_bytes };
        for (&site, &w) in counts {
            if w == 0 {
                continue;
            }
            let Some((frames, label)) = site_frames(site) else { continue };
            root.bytes += w;
            root.samples += 1;
            let mut node = &mut root;
            for name in site_to_path(&frames, &label, no_std) {
                let id = interner.id(name);
                node = node.children.entry(id).or_default();
                node.bytes += w;
                node.samples += 1;
            }
        }
    }

    // Stream the output straight to disk — never build the (multi-GB) serialized
    // form in memory.
    use std::io::Write as _;
    let file = std::fs::File::create(&out).map_err(|e| e.to_string())?;
    let mut w = std::io::BufWriter::new(file);

    if format == "folded" {
        fold(&root, &mut Vec::new(), by_count, &interner, &mut w).map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())?;
        println!("wrote folded stacks: {out}");
        println!("  pipe to inferno-flamegraph / flamegraph.pl, or load at speedscope.app");
        return Ok(());
    }

    // Chrome trace: nested B/E duration events; ts/dur in "bytes" so each frame's
    // width is its total allocated bytes.
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    write!(w, "{{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n").map_err(|e| e.to_string())?;
    write!(
        w,
        "{{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{{\"name\":\"allocations by stack\"}}}}"
    )
    .map_err(|e| e.to_string())?;
    emit_flame(&root, "all allocations", 0, &esc, &interner, &mut w).map_err(|e| e.to_string())?;
    write!(w, "\n]}}").map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;
    let unit = if by_count { "allocations" } else { "bytes" };
    println!("wrote allocation flame graph (Chrome trace): {out}");
    println!(
        "  {} total {unit} across {} stacks; width = {unit}{}",
        root.bytes,
        root.samples,
        if live_only { " (live only)" } else { "" }
    );
    println!("  open in your flame-graph viewer (Chrome trace importer), or ui.perfetto.dev");
    Ok(())
}

/// Build a flame *chart* (timeline, NOT aggregated): each allocation is a stack
/// sample placed at its time; per thread, consecutive samples are diffed so a run
/// of identical stacks merges into one slice. The x-axis is time — you scrub it to
/// see what was allocating when. Emitted as nested synchronous `B`/`E` events
/// (nesting from event order, so call order survives).
///
/// Written straight to disk as the stream is read: the per-thread open stack is
/// the only state carried, so memory is `O(threads x stack depth + sites)`.
fn cmd_flamechart(args: &[String]) -> Result<(), String> {
    let file = positional(args)
        .ok_or("usage: memscope flamechart <FILE> [--out F] [--no-std]")?;
    let out = flag(args, "--out").unwrap_or("alloc-flamechart.json").to_string();
    let no_std = args.iter().any(|a| a == "--no-std" || a == "--exclude-std");

    let rec = read_recording(file)?;

    use std::io::Write as _;
    let f = std::fs::File::create(&out).map_err(|e| e.to_string())?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let us = |t: u64| (t as f64) / 1000.0;

    write!(w, "{{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n").map_err(|e| e.to_string())?;
    write!(
        w,
        "{{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{{\"name\":\"allocation timeline\"}}}}"
    )
    .map_err(|e| e.to_string())?;

    /// A thread's currently-open frame stack, and the bytes charged through each
    /// open frame (reported as `args.bytes` when the frame closes).
    #[derive(Default)]
    struct Open {
        names: Vec<String>,
        bytes: Vec<u64>,
        last_vt: u64,
    }

    // Cleaned root->leaf path per site, built lazily so only sites that actually
    // allocate cost anything.
    let mut site_path: HashMap<u32, Vec<String>> = HashMap::new();
    let mut threads: std::collections::BTreeMap<u32, Open> = std::collections::BTreeMap::new();
    let mut total_samples = 0u64;
    let mut slices = 0u64;
    let mut vt: u64 = 0;

    let mut stream = stream_events(file)?;
    for e in stream.by_ref() {
        vt = (vt + 1).max(e.ts_nanos);
        if !matches!(
            e.kind,
            memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow
        ) {
            continue;
        }
        total_samples += 1;
        let path = site_path.entry(e.site).or_insert_with(|| {
            let info = rec.sites.get(&e.site);
            match info {
                Some(i) => site_to_path(&i.frames, &i.label, no_std),
                None => Vec::new(),
            }
        });

        let first_seen = !threads.contains_key(&e.thread);
        let open = threads.entry(e.thread).or_default();
        if first_seen {
            write!(
                w,
                ",\n{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{},\"args\":{{\"name\":\"thread {}\"}}}}",
                e.thread, e.thread
            )
            .map_err(|e| e.to_string())?;
        }

        // Common prefix with the currently-open stack.
        let mut common = 0;
        while common < open.names.len()
            && common < path.len()
            && open.names[common] == path[common]
        {
            common += 1;
        }
        // Close diverged frames (deepest first), reporting their total bytes.
        while open.names.len() > common {
            let name = open.names.pop().unwrap();
            let bytes = open.bytes.pop().unwrap();
            write!(
                w,
                ",\n{{\"ph\":\"E\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{:.3},\"pid\":1,\"tid\":{},\"args\":{{\"bytes\":{}}}}}",
                esc(&name), us(vt), e.thread, bytes
            )
            .map_err(|e| e.to_string())?;
        }
        // Open new frames (shallowest first).
        for name in &path[common..] {
            write!(
                w,
                ",\n{{\"ph\":\"B\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{:.3},\"pid\":1,\"tid\":{}}}",
                esc(name), us(vt), e.thread
            )
            .map_err(|e| e.to_string())?;
            open.names.push(name.clone());
            open.bytes.push(0);
            slices += 1;
        }
        // This allocation passes through every currently-open frame.
        for b in open.bytes.iter_mut() {
            *b += e.size;
        }
        open.last_vt = vt;
    }
    if let Some(err) = stream.error() {
        return Err(err.to_string());
    }

    // Close whatever remains open, at each thread's last timestamp.
    let thread_count = threads.len();
    for (tid, mut open) in threads {
        let end = open.last_vt + 1;
        while let Some(name) = open.names.pop() {
            let bytes = open.bytes.pop().unwrap();
            write!(
                w,
                ",\n{{\"ph\":\"E\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{:.3},\"pid\":1,\"tid\":{},\"args\":{{\"bytes\":{}}}}}",
                esc(&name), us(end), tid, bytes
            )
            .map_err(|e| e.to_string())?;
        }
    }
    write!(w, "\n]}}").map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;

    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!("wrote allocation flame chart (timeline): {out}  ({})", human_bytes(bytes));
    println!(
        "  {total_samples} allocations across {thread_count} threads -> {slices} slices (every allocation, adjacent identical stacks merged)"
    );
    // A flame *chart* is per-allocation by definition, so a churn-heavy recording
    // makes a file no viewer will open. Say so rather than let it surprise
    // someone — `flamegraph` (aggregated) or `perfetto` (counters) scale instead.
    if bytes > 512 * 1024 * 1024 {
        println!("  NOTE: that is large for a trace viewer. A flame chart keeps every allocation;");
        println!("        for a churn-heavy recording try `memscope flamegraph` (aggregated by stack)");
        println!("        or `memscope perfetto` (live-bytes counters) instead.");
    }
    println!("  open in your flame-graph viewer (Chrome trace) — x-axis is time, scrub to explore");
    Ok(())
}

fn fold<W: std::io::Write>(
    node: &Flame,
    stack: &mut Vec<u32>,
    by_count: bool,
    interner: &Interner,
    w: &mut W,
) -> std::io::Result<()> {
    // Self weight = node weight minus children weight (allocations terminating here).
    let child_sum: u64 = node
        .children
        .values()
        .map(|c| if by_count { c.samples } else { c.bytes })
        .sum();
    let self_w = if by_count { node.samples } else { node.bytes }.saturating_sub(child_sum);
    if self_w > 0 && !stack.is_empty() {
        for (i, id) in stack.iter().enumerate() {
            if i > 0 {
                w.write_all(b";")?;
            }
            w.write_all(interner.name(*id).as_bytes())?;
        }
        writeln!(w, " {self_w}")?;
    }
    for (id, child) in &node.children {
        stack.push(*id);
        fold(child, stack, by_count, interner, w)?;
        stack.pop();
    }
    Ok(())
}

/// Stream this subtree as Chrome-trace B/E events to `w`. Each event is written
/// with a leading `,\n` (the caller emits a comma-free event first), so nothing
/// accumulates in memory.
fn emit_flame<W: std::io::Write>(
    node: &Flame,
    name: &str,
    x: u64,
    esc: &impl Fn(&str) -> String,
    interner: &Interner,
    w: &mut W,
) -> std::io::Result<()> {
    // Begin/End pair (not a single X): a linear chain has parent and child with
    // the *same* [ts,dur], which X-importers can't nest — they fall back to
    // alphabetical. B/E nests by event order, so call order is preserved.
    let wgt = node.bytes.max(1);
    write!(
        w,
        ",\n{{\"ph\":\"B\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{},\"pid\":1,\"tid\":0,\"args\":{{\"bytes\":{},\"allocs\":{}}}}}",
        esc(name), x, node.bytes, node.samples
    )?;
    let mut cx = x;
    for (child_id, child) in &node.children {
        emit_flame(child, interner.name(*child_id), cx, esc, interner, w)?;
        cx += child.bytes.max(1);
    }
    write!(
        w,
        ",\n{{\"ph\":\"E\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{},\"pid\":1,\"tid\":0}}",
        esc(name),
        x + wgt
    )?;
    Ok(())
}

fn parse_hex(s: &str) -> Option<u64> {
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

fn cmd_mode(args: &[String]) -> Result<(), String> {
    let mode = positional(args).ok_or("usage: memscope mode <off|full|sampled> [--rate N]")?;
    let code = match mode {
        "off" => 0u8,
        "full" => 1,
        "sampled" => 2,
        other => return Err(format!("unknown mode '{other}' (off|full|sampled)")),
    };
    let sock = resolve_sock(args)?;
    let mut client = Client::connect(&sock)?;
    if let Some(rate) = flag(args, "--rate").and_then(|s| s.parse::<u32>().ok()) {
        client.send(&ClientMsg::SetSampleRate(rate)).map_err(|e| e.to_string())?;
        let _ = client.recv();
    }
    client.send(&ClientMsg::SetMode(code)).map_err(|e| e.to_string())?;
    let stats = loop {
        match client.recv().map_err(|e| e.to_string())? {
            ServerMsg::Stats(s) => break s,
            ServerMsg::Error(e) => eprintln!("[agent] {e}"),
            _ => continue,
        }
    };
    println!(
        "mode set to {} (rate {})",
        mode_name(stats.mode),
        stats.sample_rate
    );
    Ok(())
}

// --- marks & diff (snapshot exploration) -------------------------------------

/// A short JSON string escape for the small set of fields we emit.
fn jesc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push(' '),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Locate a site in source as `func (file:line)`: the first *application* frame
/// (frames are recorded innermost-first, so this is the boundary out of the
/// std/alloc/runtime plumbing into user code), falling back to the innermost
/// frame if every frame is runtime. Empty when the site has no resolved frames.
fn site_loc(rec: &Recording, site: u32) -> String {
    let Some(info) = rec.sites.get(&site) else { return String::new() };
    // The application boundary frame, else the innermost frame as a fallback.
    boundary_frame(&info.frames)
        .or_else(|| info.frames.first())
        .map(frame_location)
        .unwrap_or_default()
}

/// `memscope marks <FILE> [--json]` — list checkpoints and the heap size at each.
fn cmd_marks(args: &[String]) -> Result<(), String> {
    let file = positional(args).ok_or("usage: memscope marks <FILE> [--json]")?;
    let json = args.iter().any(|a| a == "--json");
    let mut rec = read_recording_raw(file)?;
    rec.resolve_sites_compact();
    let tl = Timeline::open(file, &rec)?;

    // One streaming pass over the events, summarizing the live set at each mark
    // as we reach it — the alternative (a replay per mark) re-reads the whole
    // recording once per checkpoint.
    let mut rows: Vec<(memscope_replay::MarkPoint, u64, Vec<(String, u64)>)> = Vec::new();
    tl.for_each_mark(|m, live| {
        let mut by_type: HashMap<String, u64> = HashMap::new();
        for a in live.iter() {
            *by_type.entry(rec.site_label(a.site).to_string()).or_default() += a.size;
        }
        let mut top: Vec<(String, u64)> = by_type.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        top.truncate(3);
        rows.push((m.clone(), live.total_bytes(), top));
    })?;

    if json {
        let mut out = String::from("{\"marks\":[");
        for (i, (m, bytes, top)) in rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let top_json: Vec<String> =
                top.iter().map(|(t, b)| format!("[{},{}]", jesc(t), b)).collect();
            out.push_str(&format!(
                "{{\"label\":{},\"ts_ms\":{},\"live_bytes\":{},\"top\":[{}]}}",
                jesc(&m.label),
                m.ts_nanos / 1_000_000,
                bytes,
                top_json.join(",")
            ));
        }
        out.push_str("]}");
        println!("{out}");
        return Ok(());
    }

    if rows.is_empty() {
        println!("no marks in {file} (call memscope::mark(\"label\") in the program)");
        return Ok(());
    }
    println!("== checkpoints in {file} ==");
    println!("   {:>10}  {:>12}  label   (top types)", "ts(ms)", "live");
    println!("   ────────────────────────────────────────────────────────────");
    for (m, bytes, top) in &rows {
        let top_s: Vec<String> =
            top.iter().map(|(t, b)| format!("{} {}", human_bytes(*b), t)).collect();
        println!(
            "   {:>10}  {:>12}  {}   [{}]",
            m.ts_nanos / 1_000_000,
            human_bytes(*bytes),
            m.label,
            top_s.join(", ")
        );
    }
    Ok(())
}

/// One endpoint of a diff: a named position in the stream, and when it happened.
struct Endpoint {
    name: String,
    ts_nanos: u64,
    /// Exclusive event index this endpoint sits at.
    upto: usize,
    /// Total live bytes here — filled in by the windowed replay.
    live_bytes: u64,
}

/// Resolve `start` / `end` / a mark label to a position in the stream.
///
/// This only *locates* the endpoint; reconstructing both live sets is one
/// combined pass (see [`Timeline::window`]) rather than a replay per endpoint.
fn resolve_endpoint(tl: &Timeline, name: &str) -> Result<Endpoint, String> {
    match name {
        "start" => Ok(Endpoint { name: name.into(), ts_nanos: 0, upto: 0, live_bytes: 0 }),
        "end" => Ok(Endpoint {
            name: name.into(),
            ts_nanos: tl.last_ts(),
            upto: tl.event_count(),
            live_bytes: 0,
        }),
        label => {
            let m = tl.find(label).ok_or_else(|| {
                format!("no checkpoint labeled {label:?} (try `memscope marks <FILE>`)")
            })?;
            // The mark itself is a no-op; replaying through its index is exact.
            Ok(Endpoint {
                name: label.into(),
                ts_nanos: m.ts_nanos,
                upto: m.index + 1,
                live_bytes: 0,
            })
        }
    }
}

/// One per-site delta in a diff.
struct DiffRow {
    site: u32,
    label: String,
    loc: String,
    delta_count: i64,
    delta_bytes: i64,
    born: u64,
    freed: u64,
    b_count: u64,
}

const DIFF_CAP: usize = 50;

/// `memscope diff <FILE> <A> <B> [--json]` — diff the live set between two
/// checkpoints (set-diff by site), with born/freed-in-window attribution.
fn cmd_diff(args: &[String]) -> Result<(), String> {
    let json = args.iter().any(|a| a == "--json");
    let pos: Vec<&str> = args.iter().filter(|a| !a.starts_with("--")).map(String::as_str).collect();
    let (file, a_name, b_name) = match pos.as_slice() {
        [f, a, b, ..] => (*f, *a, *b),
        _ => return Err("usage: memscope diff <FILE> <A> <B> [--json]  (A/B = mark label, or start/end)".into()),
    };

    // Definitions only, symbolicated compactly (a diff needs the type label and
    // the boundary frame, nothing deeper).
    let mut rec = read_recording_raw(file)?;
    rec.resolve_sites_compact();
    let tl = Timeline::open(file, &rec)?;
    let mut a = resolve_endpoint(&tl, a_name)?;
    let mut b = resolve_endpoint(&tl, b_name)?;
    if a.upto > b.upto {
        return Err(format!(
            "{a_name} occurs after {b_name} in the stream; pass them in chronological order"
        ));
    }

    // Both endpoints' live sets *and* the window's born/freed, in one pass.
    let w = tl.window(a.upto, b.upto)?;
    a.live_bytes = w.a_bytes;
    b.live_bytes = w.b_bytes;

    let mut sites: Vec<u32> = w.a_by_site.keys().chain(w.b_by_site.keys()).copied().collect();
    sites.sort_unstable();
    sites.dedup();

    let mut grew: Vec<DiffRow> = Vec::new();
    let mut shrank: Vec<DiffRow> = Vec::new();
    let mut unchanged = 0u32;
    for s in sites {
        let (ac, ab) = w.a_by_site.get(&s).copied().unwrap_or((0, 0));
        let (bc, bb) = w.b_by_site.get(&s).copied().unwrap_or((0, 0));
        let dc = bc as i64 - ac as i64;
        let db = bb as i64 - ab as i64;
        if dc == 0 && db == 0 {
            unchanged += 1;
            continue;
        }
        let (born, freed) = w.born_freed.get(&s).copied().unwrap_or((0, 0));
        let row = DiffRow {
            site: s,
            label: rec.site_label(s).to_string(),
            loc: site_loc(&rec, s),
            delta_count: dc,
            delta_bytes: db,
            born,
            freed,
            b_count: bc,
        };
        if db >= 0 {
            grew.push(row);
        } else {
            shrank.push(row);
        }
    }
    grew.sort_by(|x, y| y.delta_bytes.cmp(&x.delta_bytes));
    shrank.sort_by(|x, y| x.delta_bytes.cmp(&y.delta_bytes));
    let grew_trunc = grew.len() > DIFF_CAP;
    let shrank_trunc = shrank.len() > DIFF_CAP;
    grew.truncate(DIFF_CAP);
    shrank.truncate(DIFF_CAP);

    let net = b.live_bytes as i64 - a.live_bytes as i64;
    let window_ms = b.ts_nanos.saturating_sub(a.ts_nanos) / 1_000_000;

    if json {
        print_diff_json(&a, &b, &grew, &shrank, unchanged, net, window_ms, grew_trunc, shrank_trunc);
    } else {
        print_diff_text(&a, &b, &grew, &shrank, unchanged, net, window_ms, grew_trunc, shrank_trunc);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_diff_text(
    a: &Endpoint,
    b: &Endpoint,
    grew: &[DiffRow],
    shrank: &[DiffRow],
    unchanged: u32,
    net: i64,
    window_ms: u64,
    grew_trunc: bool,
    shrank_trunc: bool,
) {
    let signed = |n: i64| if n >= 0 { format!("+{}", human_bytes(n as u64)) } else { format!("-{}", human_bytes((-n) as u64)) };
    println!("== diff {} -> {}  ({} ms window) ==", a.name, b.name, window_ms);
    println!(
        "   live: {} -> {}   net retained: {}",
        human_bytes(a.live_bytes),
        human_bytes(b.live_bytes),
        signed(net)
    );
    let print_section = |title: &str, rows: &[DiffRow], trunc: bool| {
        if rows.is_empty() {
            return;
        }
        println!("\n   {title}");
        println!("   {:>8}  {:>11}  {:>6} {:>6}  type / site", "Δcount", "Δbytes", "born", "freed");
        println!("   ───────────────────────────────────────────────────────────────────");
        for r in rows {
            let leak = if r.delta_count > 0 && r.freed == 0 { "  ← never freed" } else { "" };
            let loc = if r.loc.is_empty() { String::new() } else { format!("  @ {}", r.loc) };
            println!(
                "   {:>+8}  {:>11}  {:>6} {:>6}  {}{}{}",
                r.delta_count,
                signed(r.delta_bytes),
                r.born,
                r.freed,
                r.label,
                loc,
                leak
            );
        }
        if trunc {
            println!("   … (truncated to top {DIFF_CAP})");
        }
    };
    print_section("GREW (live in B, not freed):", grew, grew_trunc);
    print_section("SHRANK:", shrank, shrank_trunc);
    println!("\n   {unchanged} type/site groups unchanged");
}

#[allow(clippy::too_many_arguments)]
fn print_diff_json(
    a: &Endpoint,
    b: &Endpoint,
    grew: &[DiffRow],
    shrank: &[DiffRow],
    unchanged: u32,
    net: i64,
    window_ms: u64,
    grew_trunc: bool,
    shrank_trunc: bool,
) {
    let row_json = |r: &DiffRow| -> String {
        format!(
            "{{\"type\":{},\"site\":{},\"site_id\":{},\"delta_count\":{},\"delta_bytes\":{},\"born_in_window\":{},\"freed_in_window\":{},\"still_live\":{}}}",
            jesc(&r.label),
            jesc(&r.loc),
            r.site,
            r.delta_count,
            r.delta_bytes,
            r.born,
            r.freed,
            r.b_count
        )
    };
    let grew_j: Vec<String> = grew.iter().map(row_json).collect();
    let shrank_j: Vec<String> = shrank.iter().map(row_json).collect();
    println!(
        "{{\"v\":1,\
         \"a\":{{\"label\":{},\"ts_ms\":{},\"live_bytes\":{}}},\
         \"b\":{{\"label\":{},\"ts_ms\":{},\"live_bytes\":{}}},\
         \"window_ms\":{},\"net_retained_delta\":{},\
         \"grew\":[{}],\"shrank\":[{}],\"unchanged_groups\":{},\
         \"truncated\":{{\"grew\":{},\"shrank\":{}}}}}",
        jesc(&a.name),
        a.ts_nanos / 1_000_000,
        a.live_bytes,
        jesc(&b.name),
        b.ts_nanos / 1_000_000,
        b.live_bytes,
        window_ms,
        net,
        grew_j.join(","),
        shrank_j.join(","),
        unchanged,
        grew_trunc,
        shrank_trunc
    );
}

/// `memscope analyze <FILE> [--json] [--top N]` — ranked memory findings.
fn cmd_analyze(args: &[String]) -> Result<(), String> {
    let file = positional(args).ok_or("usage: memscope analyze <FILE> [--json] [--top N]")?;
    let json = args.iter().any(|a| a == "--json");
    let top: usize = flag(args, "--top").and_then(|s| s.parse().ok()).unwrap_or(20);

    // Constant-memory path: read sites unresolved, then symbolicate the unique
    // IPs once and keep only a compact per-site result. Avoids materializing
    // every site's full stack (tens of GB for a million-site recording).
    let mut rec = read_recording_raw(file)?;
    rec.resolve_sites_compact();
    let findings = analyze(stream_events(file)?, &rec);
    let total = findings.len();
    let shown = total.min(top);

    if json {
        let mut out = String::from("{\"v\":1,\"findings\":[");
        for (i, f) in findings.iter().take(shown).enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&finding_json(f));
        }
        out.push_str(&format!(
            "],\"summary\":{{\"total\":{total},\"shown\":{shown}}},\"truncated\":{}}}",
            total > shown
        ));
        println!("{out}");
        return Ok(());
    }

    if findings.is_empty() {
        println!("no findings in {file} — no leaks, churn, realloc-thrash, or short-lived boxes detected");
        return Ok(());
    }
    println!("== memory findings: {file}  ({total} found, showing {shown}) ==");
    for (i, f) in findings.iter().take(shown).enumerate() {
        let loc = &f.location;
        let ev: Vec<String> = f.evidence.iter().map(|(k, v)| format!("{k}={v}")).collect();
        println!(
            "\n[{}] {}  severity {:.2}  confidence {:.2}",
            i + 1,
            f.detector.to_uppercase(),
            f.severity,
            f.confidence
        );
        println!("    {}", f.title);
        if !loc.is_empty() {
            println!("    @ {loc}");
        }
        println!("    fix: {} — {}", f.fix_class, f.suggestion);
        println!("    evidence: {}", ev.join(" "));
    }
    if total > shown {
        println!("\n   … {} more (raise --top to see them)", total - shown);
    }
    Ok(())
}

fn finding_json(f: &Finding) -> String {
    let ev: Vec<String> =
        f.evidence.iter().map(|(k, v)| format!("{}:{}", jesc(k), jesc(v))).collect();
    format!(
        "{{\"detector\":{},\"severity\":{:.4},\"confidence\":{:.2},\"title\":{},\"type\":{},\"site\":{},\"site_loc\":{},\"fix_class\":{},\"suggestion\":{},\"evidence\":{{{}}}}}",
        jesc(f.detector),
        f.severity,
        f.confidence,
        jesc(&f.title),
        jesc(&f.ty),
        f.site,
        jesc(&f.location),
        jesc(f.fix_class),
        jesc(&f.suggestion),
        ev.join(",")
    )
}

// --- run: dump an unmodified binary via injection -----------------------------

/// Locate the preload dylib: `$MEMSCOPE_PRELOAD_LIB`, else a sibling of this
/// executable (both land in `target/<profile>/`).
fn preload_lib() -> Result<String, String> {
    let name = if cfg!(target_os = "macos") {
        "libmemscope_preload.dylib"
    } else {
        "libmemscope_preload.so"
    };
    if let Ok(p) = std::env::var("MEMSCOPE_PRELOAD_LIB") {
        return Ok(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        let sib = exe.with_file_name(name);
        if sib.exists() {
            return Ok(sib.to_string_lossy().into_owned());
        }
    }
    Err(format!("could not find {name}; set MEMSCOPE_PRELOAD_LIB to its path"))
}

/// Parse a duration like `2s`, `500ms`, or a bare number of seconds.
/// Mirrors memscope-preload's `MEMSCOPE_HPROF_AT_BYTES` parser, so a value the
/// CLI accepts is exactly one the preload will honor.
fn parse_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix("GB").or_else(|| s.strip_suffix("G")) {
        (n, 1u64 << 30)
    } else if let Some(n) = s.strip_suffix("MB").or_else(|| s.strip_suffix("M")) {
        (n, 1 << 20)
    } else if let Some(n) = s.strip_suffix("KB").or_else(|| s.strip_suffix("K")) {
        (n, 1 << 10)
    } else {
        (s, 1)
    };
    num.trim().parse::<f64>().ok().map(|v| (v * mult as f64) as u64)
}

fn parse_dur(s: &str) -> Option<std::time::Duration> {
    let s = s.trim();
    if let Some(ms) = s.strip_suffix("ms") {
        ms.trim().parse::<u64>().ok().map(std::time::Duration::from_millis)
    } else if let Some(sec) = s.strip_suffix('s') {
        sec.trim().parse::<f64>().ok().map(std::time::Duration::from_secs_f64)
    } else {
        s.parse::<f64>().ok().map(std::time::Duration::from_secs_f64)
    }
}

/// `memscope run [--out PATH] [--on-exit] [--after DUR] [--at-bytes N] -- <prog> [args]`.
fn cmd_run(args: &[String]) -> Result<(), String> {
    let split = args.iter().position(|a| a == "--");
    let (opts, cmd): (&[String], &[String]) = match split {
        Some(i) => (&args[..i], &args[i + 1..]),
        None => return Err("usage: memscope run [opts] -- <program> [args...]  (note the `--`)".into()),
    };
    if cmd.is_empty() {
        return Err("no program given after `--`".into());
    }

    let out = flag(opts, "--out");
    let on_exit = opts.iter().any(|a| a == "--on-exit");
    let after = flag(opts, "--after");
    let at_bytes = flag(opts, "--at-bytes");
    // Validate trigger values *before* spawning the target — a bad flag must not
    // launch (and fully run) the program with no trigger armed.
    let after_dur = match after {
        Some(d) => {
            Some(parse_dur(d).ok_or_else(|| format!("bad --after duration {d:?} (e.g. 2s, 500ms)"))?)
        }
        None => None,
    };
    if let Some(n) = at_bytes {
        parse_bytes(n).ok_or_else(|| format!("bad --at-bytes value {n:?} (e.g. 5MB, 200KB, 1048576)"))?;
    }
    // Default trigger: dump at exit if nothing else was requested.
    let default_on_exit = !on_exit && after.is_none() && at_bytes.is_none();

    let lib = preload_lib()?;
    let env_var = if cfg!(target_os = "macos") { "DYLD_INSERT_LIBRARIES" } else { "LD_PRELOAD" };

    // Infer whether the caller wants an alloc-stream (.mscope) or a live-heap dump
    // (.hprof). We do this from the extension of --out: if it ends with .mscope,
    // .json, or .jsonl we produce a full churn trace via MEMSCOPE_RECORD (perfetto-ready).
    // Otherwise we keep the legacy HPROF behavior. When only --at-bytes/--after is
    // used we still default to HPROF; the new path is explicitly opt-in via .mscope/.json.
    let wants_mscope = out
        .map(|o| {
            let lo = o.to_ascii_lowercase();
            lo.ends_with(".mscope")
                || lo.ends_with(".json")
                || lo.ends_with(".jsonl")
                || lo.ends_with(".trace")
        })
        .unwrap_or(false);

    let mut child = std::process::Command::new(&cmd[0]);
    child.args(&cmd[1..]);
    child.env(env_var, &lib);
    if wants_mscope {
        // Full alloc-stream. memscope-preload now honors MEMSCOPE_RECORD and starts a
        // FileRecorder directly, so we get a self-contained .mscope with raw sites + slide.
        // That file is what `memscope perfetto|flamegraph|flamechart|analyze|diff` consume.
        if let Some(o) = out {
            child.env("MEMSCOPE_RECORD", o);
            // Also keep the HPROF env pointed at the same stem for `--on-exit` diagnostics,
            // but don't require it — the .mscope path is the primary artifact.
            child.env("MEMSCOPE_HPROF_ON_EXIT", "1");
        }
    } else {
        if let Some(o) = out {
            child.env("MEMSCOPE_HPROF_OUT", o);
        }
        if on_exit || default_on_exit {
            child.env("MEMSCOPE_HPROF_ON_EXIT", "1");
        }
        if let Some(n) = at_bytes {
            child.env("MEMSCOPE_HPROF_AT_BYTES", n);
        }
    }

    if wants_mscope {
        eprintln!(
            "[memscope] launching `{}` under memscope ({env_var}) — recording full alloc-stream to {}",
            cmd.join(" "),
            out.unwrap_or("/tmp/memscope-{pid}-0.mscope")
        );
    } else {
        eprintln!("[memscope] launching `{}` under memscope ({env_var})", cmd.join(" "));
    }
    let mut handle = child.spawn().map_err(|e| format!("failed to launch {}: {e}", cmd[0]))?;
    let pid = handle.id();

    // Timed trigger: signal the child after the delay (it stays running).
    if let Some(dur) = after_dur {
        std::thread::spawn(move || {
            std::thread::sleep(dur);
            eprintln!("[memscope] --after elapsed — signalling pid {pid} for a dump");
            unsafe {
                libc::kill(pid as i32, libc::SIGUSR1);
            }
        });
    }

    let status = handle.wait().map_err(|e| e.to_string())?;
    // Give the dumper thread a moment to finish an exit/threshold dump, or the
    // file recorder to flush the tail of the trace (important for short programs).
    std::thread::sleep(std::time::Duration::from_millis(if wants_mscope { 600 } else { 300 }));

    let expected = match out {
        Some(o) => o.replace("{pid}", &pid.to_string()).replace("{n}", "0"),
        None => {
            if wants_mscope {
                format!("/tmp/memscope-{pid}-0.mscope")
            } else {
                format!("/tmp/memscope-{pid}-0.hprof")
            }
        }
    };
    if std::path::Path::new(&expected).exists() {
        if wants_mscope {
            eprintln!("[memscope] alloc-stream trace: {expected}");
            eprintln!("           perfetto: memscope perfetto {expected} --out trace.json && open https://ui.perfetto.dev");
            eprintln!("           flamegraph: memscope flamegraph {expected} --out fg.json");
            eprintln!("           analyze:   memscope analyze {expected}");
        } else {
            eprintln!("[memscope] heap dump: {expected}");
            eprintln!("           analyze it:  memscope analyze {expected}   (or open in MAT / heapster)");
        }
    } else {
        eprintln!("[memscope] no dump found at {expected} — was a trigger reached? (--on-exit/--after/--at-bytes)");
        if wants_mscope {
            eprintln!("           note: MEMSCOPE_RECORD path should have been written by the preload lib; check that {env_var} injection worked");
        }
    }
    std::process::exit(status.code().unwrap_or(0));
}

/// `memscope dump-pid <PID>` — trigger a dump in a process started by `run`.
fn cmd_dump_pid(args: &[String]) -> Result<(), String> {
    let pid: i32 = positional(args)
        .and_then(|s| s.parse().ok())
        .ok_or("usage: memscope dump-pid <PID>  (the process must have been started by `memscope run`)")?;
    let rc = unsafe { libc::kill(pid, libc::SIGUSR1) };
    if rc != 0 {
        return Err(format!("could not signal pid {pid} (not running, or not yours)"));
    }
    eprintln!("[memscope] requested heap dump from pid {pid} (written by that process to its MEMSCOPE_HPROF_OUT)");
    Ok(())
}

// --- query: bounded drill-down into a finding ---------------------------------

/// Lifetime histogram buckets (upper bound in ms, exclusive; last is the tail).
const LIFETIME_BUCKETS: &[(&str, u64)] = &[
    ("<1ms", 1),
    ("1-2ms", 2),
    ("2-5ms", 5),
    ("5-10ms", 10),
    ("10-100ms", 100),
    ("100ms-1s", 1000),
    (">=1s", u64::MAX),
];

/// `memscope query <FILE> (--site N | --type T) [--field F] [--json]`.
fn cmd_query(args: &[String]) -> Result<(), String> {
    let file = positional(args).ok_or(
        "usage: memscope query <FILE> (--site N | --type T) [--field stack|lifetimes|stats|sites] [--json]",
    )?;
    let json = args.iter().any(|a| a == "--json");
    let field = flag(args, "--field").unwrap_or("stats");
    let site_arg: Option<u32> = flag(args, "--site").and_then(|s| s.parse().ok());
    let type_arg: Option<&str> = flag(args, "--type");
    if site_arg.is_none() && type_arg.is_none() {
        return Err("query needs --site N or --type T".into());
    }

    // Definitions only + compact symbolication; the events are folded straight
    // off the stream, so this is bounded by site count, not recording size.
    let mut rec = read_recording_raw(file)?;
    rec.resolve_sites_compact();
    let stats = site_stats(stream_events(file)?, &rec);

    // The matching per-site stats (one for --site, all of a type for --type).
    let mut matched: Vec<&SiteStats> = stats
        .iter()
        .filter(|s| match (site_arg, type_arg) {
            (Some(id), _) => s.site == id,
            (_, Some(ty)) => s.label == ty,
            _ => false,
        })
        .collect();
    // `--type` falls back to substring match when nothing matches exactly, so
    // `--type StringBuf` finds `StringBuf<u8>` (the labels other commands print
    // are generic — nobody should have to retype them character-perfect).
    if matched.is_empty() {
        if let (None, Some(ty)) = (site_arg, type_arg) {
            let sub: Vec<&SiteStats> =
                stats.iter().filter(|s| s.label.contains(ty)).collect();
            let mut labels: Vec<&str> = sub.iter().map(|s| s.label.as_str()).collect();
            labels.sort_unstable();
            labels.dedup();
            match labels.len() {
                0 => {}
                1 => matched = sub,
                _ => {
                    return Err(format!(
                        "--type {ty:?} is ambiguous in {file}; it matches:\n  {}",
                        labels.join("\n  ")
                    ))
                }
            }
        }
    }
    if matched.is_empty() {
        return Err(match (site_arg, type_arg) {
            (Some(id), _) => format!("no site {id} in {file}"),
            (_, Some(ty)) => {
                let mut labels: Vec<&str> = stats.iter().map(|s| s.label.as_str()).collect();
                labels.sort_unstable();
                labels.dedup();
                format!(
                    "no allocations of type {ty:?} in {file}; types present:\n  {}",
                    labels.join("\n  ")
                )
            }
            _ => unreachable!(),
        });
    }

    match field {
        "stack" => {
            // `--field stack` is the one view that wants a site's *whole* stack,
            // not just its boundary frame. Re-resolve the handful of matched
            // sites in full — bounded by `matched.len()`, not by site count.
            let ids: Vec<u32> = matched.iter().map(|s| s.site).collect();
            let matched: Vec<SiteStats> = matched.into_iter().cloned().collect();
            for id in &ids {
                rec.sites.remove(id);
            }
            rec.resolve_sites(&ids);
            let matched: Vec<&SiteStats> = matched.iter().collect();
            query_stack(&rec, &matched, json)
        }
        "lifetimes" => query_lifetimes(&matched, json),
        "sites" => query_sites(&rec, &matched, json),
        "stats" => query_stats(&matched, json),
        other => Err(format!("unknown --field {other:?} (stack|lifetimes|stats|sites)")),
    }
}

/// Combined per-site stats into one aggregate (for `--type`).
fn aggregate(matched: &[&SiteStats]) -> SiteStats {
    let mut it = matched.iter();
    let mut acc = (*it.next().unwrap()).clone();
    for s in it {
        acc.alloc_count += s.alloc_count;
        acc.alloc_bytes += s.alloc_bytes;
        acc.realloc_count += s.realloc_count;
        acc.free_count += s.free_count;
        acc.lifetime_sum_ms += s.lifetime_sum_ms;
        acc.lifetime_sample.extend_from_slice(&s.lifetime_sample);
        acc.live_count += s.live_count;
        acc.live_bytes += s.live_bytes;
    }
    acc
}

fn query_stats(matched: &[&SiteStats], json: bool) -> Result<(), String> {
    let a = aggregate(matched);
    let median = a.median_lifetime_ms();
    if json {
        println!(
            "{{\"type\":{},\"sites\":{},\"alloc_count\":{},\"alloc_bytes\":{},\"realloc_count\":{},\"free_count\":{},\"live_count\":{},\"live_bytes\":{},\"median_lifetime_ms\":{}}}",
            jesc(&a.label),
            matched.len(),
            a.alloc_count,
            a.alloc_bytes,
            a.realloc_count,
            a.free_count,
            a.live_count,
            a.live_bytes,
            median.map(|m| m.to_string()).unwrap_or("null".into())
        );
        return Ok(());
    }
    println!("== {} ({} site(s)) ==", a.label, matched.len());
    println!("   allocations : {} ({})", a.alloc_count, human_bytes(a.alloc_bytes));
    println!("   reallocs    : {}", a.realloc_count);
    println!("   freed       : {}", a.free_count);
    println!("   live now    : {} ({})", a.live_count, human_bytes(a.live_bytes));
    if let Some(m) = median {
        println!("   median life : {m} ms");
    }
    Ok(())
}

fn query_lifetimes(matched: &[&SiteStats], json: bool) -> Result<(), String> {
    let a = aggregate(matched);
    if a.lifetime_sample.is_empty() {
        if json {
            println!("{{\"type\":{},\"lifetimes\":[],\"note\":\"nothing freed\"}}", jesc(&a.label));
        } else {
            println!("{}: nothing from this type was observed to free", a.label);
        }
        return Ok(());
    }
    let mut counts = vec![0u64; LIFETIME_BUCKETS.len()];
    for &lt in &a.lifetime_sample {
        let i = LIFETIME_BUCKETS.iter().position(|(_, hi)| lt < *hi).unwrap_or(LIFETIME_BUCKETS.len() - 1);
        counts[i] += 1;
    }
    let total: u64 = counts.iter().sum();
    if json {
        let buckets: Vec<String> = LIFETIME_BUCKETS
            .iter()
            .zip(&counts)
            .map(|((label, _), c)| format!("{{\"bucket\":{},\"count\":{}}}", jesc(label), c))
            .collect();
        println!(
            "{{\"type\":{},\"sample\":{},\"buckets\":[{}]}}",
            jesc(&a.label),
            total,
            buckets.join(",")
        );
        return Ok(());
    }
    println!("== lifetime histogram: {} ({} freed sampled) ==", a.label, total);
    let max = counts.iter().copied().max().unwrap_or(1).max(1);
    for ((label, _), c) in LIFETIME_BUCKETS.iter().zip(&counts) {
        let bar = "█".repeat((*c as usize * 30 / max as usize).max(if *c > 0 { 1 } else { 0 }));
        println!("   {label:>9}  {c:>7}  {bar}");
    }
    Ok(())
}

fn query_sites(rec: &Recording, matched: &[&SiteStats], json: bool) -> Result<(), String> {
    let mut rows: Vec<(&SiteStats, String)> =
        matched.iter().map(|s| (*s, site_loc(rec, s.site))).collect();
    // Site id breaks byte ties so repeated runs print the same order (the stats
    // come out of a HashMap, whose iteration order varies run to run).
    rows.sort_by(|a, b| b.0.alloc_bytes.cmp(&a.0.alloc_bytes).then(a.0.site.cmp(&b.0.site)));
    if json {
        let sites: Vec<String> = rows
            .iter()
            .map(|(s, loc)| {
                format!(
                    "{{\"site\":{},\"loc\":{},\"alloc_count\":{},\"alloc_bytes\":{},\"live_bytes\":{}}}",
                    s.site, jesc(loc), s.alloc_count, s.alloc_bytes, s.live_bytes
                )
            })
            .collect();
        println!("{{\"sites\":[{}]}}", sites.join(","));
        return Ok(());
    }
    println!("== {} sites of this type ==", rows.len());
    for (s, loc) in &rows {
        println!(
            "   site {:>6}  {:>10} in {:>7} allocs  @ {}",
            s.site,
            human_bytes(s.alloc_bytes),
            s.alloc_count,
            loc
        );
    }
    Ok(())
}

fn query_stack(rec: &Recording, matched: &[&SiteStats], json: bool) -> Result<(), String> {
    // Use the biggest matching site (most allocations) as the representative.
    let s = matched.iter().max_by_key(|s| s.alloc_bytes).unwrap();
    let Some(info) = rec.sites.get(&s.site) else {
        return Err(format!("no resolved frames for site {}", s.site));
    };
    // Frames are innermost-first; show application frames with file:line, and
    // mark the runtime plumbing rather than hiding it entirely.
    let frames: Vec<(String, bool)> = info
        .frames
        .iter()
        .map(|f| (frame_location(f), is_std_frame(&clean_frame(&f.func))))
        .collect();
    if json {
        let arr: Vec<String> = frames
            .iter()
            .map(|(loc, is_std)| format!("{{\"frame\":{},\"std\":{}}}", jesc(loc), is_std))
            .collect();
        println!(
            "{{\"site\":{},\"type\":{},\"stack\":[{}]}}",
            s.site,
            jesc(&s.label),
            arr.join(",")
        );
        return Ok(());
    }
    println!("== stack for site {} ({}) ==", s.site, s.label);
    println!("   (innermost first; · = application frame, std/runtime dimmed)");
    for (loc, is_std) in &frames {
        let marker = if *is_std { "  " } else { "· " };
        println!("   {marker}{loc}");
    }
    Ok(())
}

#[cfg(test)]
mod meta_tests {
    use super::*;
    use memscope_proto::EventKind::*;

    fn ev(kind: memscope_proto::EventKind, thread: u32, site: u32, addr: u64) -> RecEvent {
        RecEvent { kind, addr, size: 8, ts_nanos: 0, site, thread }
    }

    fn meta_tables() -> HashMap<u32, Vec<(String, String)>> {
        let mut m = HashMap::new();
        m.insert(1, vec![("subsystem".into(), "physics".into())]);
        m.insert(2, vec![("shard".into(), "3".into())]);
        m.insert(3, vec![("subsystem".into(), "io".into())]);
        m
    }

    /// Feed events to a [`MetaTracker`] and capture the metadata active at each
    /// allocation — the streaming equivalent of the old materialized
    /// metadata-per-event array.
    fn meta_at_allocs(
        events: &[RecEvent],
        tables: &HashMap<u32, Vec<(String, String)>>,
    ) -> HashMap<u64, std::collections::BTreeMap<String, String>> {
        let mut tracker = MetaTracker::new(tables);
        let mut out = HashMap::new();
        for e in events {
            tracker.observe(e);
            if e.kind == Alloc {
                out.insert(e.addr, tracker.active(e.thread));
            }
        }
        out
    }

    #[test]
    fn correlate_nesting_and_per_thread() {
        // thread 10 nests ctx1(physics) > ctx2(shard=3); thread 20 runs ctx3(io)
        // concurrently and interleaved in the global stream.
        let events = vec![
            ev(MetaEnter, 10, 1, 0),
            ev(MetaEnter, 20, 3, 0),
            ev(MetaEnter, 10, 2, 0),
            ev(Alloc, 10, 99, 0x1), // {subsystem:physics, shard:3}
            ev(Alloc, 20, 99, 0x2), // {subsystem:io} (other thread unaffected)
            ev(MetaExit, 10, 2, 0),
            ev(Alloc, 10, 99, 0x3), // {subsystem:physics}
            ev(MetaExit, 10, 1, 0),
            ev(Alloc, 10, 99, 0x4), // {}
        ];
        let m = meta_at_allocs(&events, &meta_tables());
        assert_eq!(m[&0x1].get("subsystem").map(String::as_str), Some("physics"));
        assert_eq!(m[&0x1].get("shard").map(String::as_str), Some("3"));
        assert_eq!(m[&0x2].get("subsystem").map(String::as_str), Some("io"));
        assert!(m[&0x2].get("shard").is_none()); // thread 20 never saw shard
        assert_eq!(m[&0x3].get("subsystem").map(String::as_str), Some("physics"));
        assert!(m[&0x3].get("shard").is_none()); // ctx2 exited
        assert!(m[&0x4].is_empty()); // all scopes exited
    }

    #[test]
    fn inner_scope_overrides_outer_key() {
        let mut tables = HashMap::new();
        tables.insert(1, vec![("subsystem".to_string(), "outer".to_string())]);
        tables.insert(2, vec![("subsystem".to_string(), "inner".to_string())]);
        let events = vec![
            ev(MetaEnter, 1, 1, 0),
            ev(MetaEnter, 1, 2, 0),
            ev(Alloc, 1, 9, 0x1), // inner wins (merged outer->inner)
        ];
        let m = meta_at_allocs(&events, &tables);
        assert_eq!(m[&0x1].get("subsystem").map(String::as_str), Some("inner"));
    }

    #[test]
    fn parse_filters_multiple() {
        let args: Vec<String> = ["--filter", "subsystem=io", "--x", "--filter", "shard=3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let f = parse_filters(&args);
        assert_eq!(f, vec![("subsystem".into(), "io".into()), ("shard".into(), "3".into())]);
    }
}
