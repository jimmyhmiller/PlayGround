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
    analyze, boundary_frame, clean_frame, frame_location, is_std_frame, label_for, read_meta_value,
    read_recording, read_recording_raw, site_stats, Finding, FrameMeta, RecEvent, Recording,
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
/// `/tmp/memscope-*.sock` if exactly one exists.
fn resolve_sock(args: &[String]) -> Result<String, String> {
    if let Some(s) = flag(args, "--sock") {
        return Ok(s.to_string());
    }
    if let Ok(s) = std::env::var("MEMSCOPE_SOCK") {
        return Ok(s);
    }
    let mut found = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/tmp") {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("memscope-") && name.ends_with(".sock") {
                found.push(e.path().to_string_lossy().into_owned());
            }
        }
    }
    match found.len() {
        1 => Ok(found.pop().unwrap()),
        0 => Err("no agent socket found; pass --sock <path> (agent prints it on start)".into()),
        _ => Err(format!(
            "multiple agent sockets found, pass --sock <path>:\n  {}",
            found.join("\n  ")
        )),
    }
}

// --- protocol client ---------------------------------------------------------

struct Client {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Client {
    fn connect(path: &str) -> std::io::Result<Self> {
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
                (Some(shape), Some(ty)) => format!("{shape:?}<{ty}>"),
                (Some(shape), None) => format!("{shape:?}<?>"),
                (None, Some(ty)) => ty.to_string(),
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
    let mut client = Client::connect(&sock).map_err(|e| e.to_string())?;

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
    let mut client = Client::connect(&sock).map_err(|e| e.to_string())?;
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
    let bytes = std::fs::read(file).map_err(|e| e.to_string())?;
    let snap: Snapshot = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
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
            (Some(shape), Some(ty)) => format!("{shape:?}<{ty}>"),
            (Some(shape), None) => format!("{shape:?}<?>"),
            (None, Some(ty)) => ty.to_string(),
            (None, None) => "<unknown>".to_string(),
        };
        println!("\n  {label}  —  {c} live, {}", human_bytes(*b));
        for f in s.frames.iter().take(6) {
            let func = f.function.as_deref().unwrap_or("?");
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
    let mut client = Client::connect(&sock).map_err(|e| e.to_string())?;
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
        (Some(shape), Some(ty)) => format!("{shape:?}<{ty}>"),
        (Some(shape), None) => format!("{shape:?}<?>"),
        (None, Some(ty)) => ty.to_string(),
        (None, None) => "<unknown>".to_string(),
    }
}

fn cmd_graph(args: &[String]) -> Result<(), String> {
    let sock = resolve_sock(args)?;
    let limit: usize = flag(args, "--limit").and_then(|s| s.parse().ok()).unwrap_or(25);
    let mut client = Client::connect(&sock).map_err(|e| e.to_string())?;
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
    let mut client = Client::connect(&sock).map_err(|e| e.to_string())?;
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
fn cmd_replay(args: &[String]) -> Result<(), String> {
    use std::io::Read;
    let file = positional(args).ok_or("usage: memscope replay <FILE>")?;
    let mut magic = [0u8; 4];
    {
        let mut f = std::fs::File::open(file).map_err(|e| e.to_string())?;
        let _ = f.read(&mut magic);
    }
    if memscope_proto::recfmt::is_binary(&magic) {
        replay_binary(file)
    } else {
        replay_json(file)
    }
}

/// Replay a compact binary `.mscope` recording (streaming).
fn replay_binary(file: &str) -> Result<(), String> {
    use memscope_proto::recfmt;
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(file).map_err(|e| e.to_string())?);

    // Streaming primitives.
    let mut b1 = [0u8; 1];
    let mut b2 = [0u8; 2];
    let mut b4 = [0u8; 4];
    macro_rules! rd {
        ($buf:expr) => {
            f.read_exact(&mut $buf).is_ok()
        };
    }
    let rd_str = |f: &mut std::io::BufReader<std::fs::File>| -> Option<String> {
        let mut l = [0u8; 2];
        f.read_exact(&mut l).ok()?;
        let n = u16::from_le_bytes(l) as usize;
        let mut s = vec![0u8; n];
        f.read_exact(&mut s).ok()?;
        Some(String::from_utf8_lossy(&s).into_owned())
    };

    // Header.
    if !rd!(b4) || b4 != recfmt::MAGIC {
        return Err("not a memscope binary recording".into());
    }
    let _ = rd!(b2); // version
    let _ = rd!(b2); // flags
    let _ = rd!(b4); // pid
    let pid = u32::from_le_bytes(b4);
    let exe = rd_str(&mut f).unwrap_or_default();
    // v2+ carries the load slide (for read-time symbolication of raw sites).
    let mut b8 = [0u8; 8];
    let slide = if rd!(b8) { u64::from_le_bytes(b8) } else { 0 };

    let mut site_label: HashMap<u32, String> = HashMap::new();
    // Raw sites are symbolicated after the stream is read (see below).
    let mut raw_sites: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut live: HashMap<u64, (u64, u32)> = HashMap::new();
    let (mut allocs, mut frees, mut peak, mut cur) = (0u64, 0u64, 0u64, 0u64);

    while rd!(b1) {
        match b1[0] {
            recfmt::TAG_SITE => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let site = u32::from_le_bytes(b4);
                f.read_exact(&mut b1).map_err(|e| e.to_string())?;
                let ty = if b1[0] == 1 {
                    rd_str(&mut f)
                } else {
                    None
                };
                f.read_exact(&mut b1).map_err(|e| e.to_string())?;
                let shape = recfmt::shape_from_code(b1[0]);
                f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                let nframes = u16::from_le_bytes(b2);
                for _ in 0..nframes {
                    let _ = rd_str(&mut f); // function
                    let _ = rd_str(&mut f); // file
                    f.read_exact(&mut b4).ok(); // line
                    f.read_exact(&mut b1).ok(); // inlined
                }
                let label = match (shape, ty) {
                    (Some(sh), Some(t)) => format!("{sh:?}<{t}>"),
                    (Some(sh), None) => format!("{sh:?}<?>"),
                    (None, Some(t)) => t,
                    (None, None) => "<no type>".into(),
                };
                site_label.insert(site, label);
            }
            recfmt::TAG_RSITE => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let site = u32::from_le_bytes(b4);
                f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                let n = u16::from_le_bytes(b2) as usize;
                let mut ips = Vec::with_capacity(n);
                let mut ipb = [0u8; 8];
                for _ in 0..n {
                    f.read_exact(&mut ipb).map_err(|e| e.to_string())?;
                    ips.push(u64::from_le_bytes(ipb));
                }
                raw_sites.insert(site, ips);
            }
            recfmt::TAG_EVENTS => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let count = u32::from_le_bytes(b4);
                let mut rec = vec![0u8; recfmt::EVENT_BYTES * count as usize];
                f.read_exact(&mut rec).map_err(|e| e.to_string())?;
                let mut r = recfmt::Reader::new(&rec);
                for _ in 0..count {
                    let Some(e) = r.decode_event() else { break };
                    match e.kind {
                        memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow => {
                            live.insert(e.addr, (e.size, e.site.0));
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
            }
            recfmt::TAG_KEY => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let _ = rd_str(&mut f);
            }
            recfmt::TAG_MARK => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let _ = rd_str(&mut f); // label
            }
            recfmt::TAG_META => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                for _ in 0..u16::from_le_bytes(b2) {
                    f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    read_meta_value(&mut f);
                }
            }
            other => return Err(format!("corrupt recording: unknown tag {other}")),
        }
    }

    // Symbolicate raw sites once, against the binary's dSYM (off the traced
    // process). Best-effort: a missing dSYM just leaves labels unresolved.
    if !raw_sites.is_empty() {
        let sites: Vec<(u32, Vec<u64>)> = raw_sites.into_iter().collect();
        if let Ok(resolved) =
            memscope_symbols::resolve_raw_sites(std::path::Path::new(&exe), slide, &sites)
        {
            for (id, r) in resolved {
                site_label.insert(id, label_for(r.shape, r.element_type));
            }
        }
    }

    println!("== memscope binary recording: {file} ==");
    println!("  pid {pid}   exe {exe}");
    println!(
        "  events: {allocs} alloc, {frees} free   peak live: {}   final live: {} allocations, {}",
        human_bytes(peak),
        live.len(),
        human_bytes(cur)
    );
    print_live_by_type(&live, &site_label);
    Ok(())
}

fn print_live_by_type(live: &HashMap<u64, (u64, u32)>, site_label: &HashMap<u32, String>) {
    let mut by_type: HashMap<String, (u64, u64)> = HashMap::new();
    for (sz, site) in live.values() {
        let label = site_label
            .get(site)
            .cloned()
            .unwrap_or_else(|| "<no site>".to_string());
        let e = by_type.entry(label).or_default();
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

fn replay_json(file: &str) -> Result<(), String> {
    use std::io::BufRead;
    let f = std::fs::File::open(file).map_err(|e| e.to_string())?;
    let rdr = std::io::BufReader::new(f);

    let mut site_label: HashMap<u64, String> = HashMap::new();
    let mut raw_sites: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut exe = String::new();
    let mut slide = 0u64;
    // addr -> (size, site)
    let mut live: HashMap<u64, (u64, u64)> = HashMap::new();
    let (mut allocs, mut frees, mut peak_bytes, mut cur_bytes) = (0u64, 0u64, 0u64, 0u64);
    let mut header = String::new();

    for line in rdr.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("v").is_some() {
            header = line.clone();
            exe = v.get("exe").and_then(|x| x.as_str()).unwrap_or("").to_string();
            slide = v.get("slide").and_then(|x| x.as_u64()).unwrap_or(0);
            continue;
        }
        if let Some(id) = v.get("rsite").and_then(|x| x.as_u64()) {
            let ips = v
                .get("ips")
                .and_then(|x| x.as_array())
                .map(|arr| arr.iter().filter_map(|n| n.as_u64()).collect())
                .unwrap_or_default();
            raw_sites.insert(id as u32, ips);
            continue;
        }
        if let Some(site) = v.get("site").and_then(|x| x.as_u64()) {
            let ty = v.get("ty").and_then(|x| x.as_str()).unwrap_or("?");
            let shape = v.get("shape").and_then(|x| x.as_str());
            let label = match shape {
                Some(sh) => format!("{sh}<{ty}>"),
                None => ty.to_string(),
            };
            site_label.insert(site, label);
            continue;
        }
        match v.get("k").and_then(|x| x.as_str()) {
            Some("A") | Some("R") => {
                let a = v["a"].as_u64().unwrap_or(0);
                let sz = v["sz"].as_u64().unwrap_or(0);
                let s = v.get("s").and_then(|x| x.as_u64()).unwrap_or(u64::MAX);
                live.insert(a, (sz, s));
                allocs += 1;
                cur_bytes += sz;
                peak_bytes = peak_bytes.max(cur_bytes);
            }
            Some("D") => {
                let a = v["a"].as_u64().unwrap_or(0);
                if let Some((sz, _)) = live.remove(&a) {
                    cur_bytes = cur_bytes.saturating_sub(sz);
                }
                frees += 1;
            }
            _ => {}
        }
    }

    // Symbolicate raw sites once, against the binary's dSYM (off the traced
    // process). Best-effort: a missing dSYM just leaves labels unresolved.
    if !raw_sites.is_empty() {
        let sites: Vec<(u32, Vec<u64>)> = raw_sites.into_iter().collect();
        if let Ok(resolved) =
            memscope_symbols::resolve_raw_sites(std::path::Path::new(&exe), slide, &sites)
        {
            for (id, r) in resolved {
                site_label.insert(id as u64, label_for(r.shape, r.element_type));
            }
        }
    }

    println!("== memscope recording: {file} ==");
    println!("  {header}");
    println!(
        "  events: {} alloc, {} free   peak live: {}   final live: {} allocations, {}",
        allocs,
        frees,
        human_bytes(peak_bytes),
        live.len(),
        human_bytes(cur_bytes),
    );

    // Final live set aggregated by recovered type.
    let mut by_type: HashMap<String, (u64, u64)> = HashMap::new();
    for (_, (sz, site)) in &live {
        let label = site_label
            .get(site)
            .cloned()
            .unwrap_or_else(|| "<no site>".to_string());
        let e = by_type.entry(label).or_default();
        e.0 += 1;
        e.1 += sz;
    }
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
    println!("\n  FINAL LIVE HEAP BY TYPE:");
    println!("  {:>8}  {:>12}  {}", "count", "bytes", "type");
    for (label, (count, bytes)) in rows.iter().take(20) {
        println!("  {count:>8}  {:>12}  {label}", human_bytes(*bytes));
    }
    Ok(())
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

/// Indices of alloc events whose allocation is still live at the end (address
/// not subsequently freed). Handles address reuse by keeping the latest index.
fn live_indices(events: &[RecEvent]) -> std::collections::HashSet<usize> {
    let mut by_addr: HashMap<u64, usize> = HashMap::new();
    for (i, e) in events.iter().enumerate() {
        match e.kind {
            memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow => {
                by_addr.insert(e.addr, i);
            }
            memscope_proto::EventKind::Dealloc => {
                by_addr.remove(&e.addr);
            }
            _ => {}
        }
    }
    by_addr.values().copied().collect()
}

/// Correlate metadata onto allocations: replay events per thread in order,
/// maintaining each thread's context stack, and return — parallel to `events` —
/// the merged metadata map active at each event (only meaningful for allocs).
fn correlate_meta(
    events: &[RecEvent],
    meta: &HashMap<u32, Vec<(String, String)>>,
) -> Vec<std::collections::BTreeMap<String, String>> {
    use std::collections::BTreeMap;
    let mut stacks: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut out: Vec<BTreeMap<String, String>> = Vec::with_capacity(events.len());
    for e in events {
        match e.kind {
            memscope_proto::EventKind::MetaEnter => {
                stacks.entry(e.thread).or_default().push(e.site);
                out.push(BTreeMap::new());
            }
            memscope_proto::EventKind::MetaExit => {
                if let Some(s) = stacks.get_mut(&e.thread) {
                    // Pop the matching id (LIFO; tolerate slight disorder).
                    if let Some(pos) = s.iter().rposition(|&m| m == e.site) {
                        s.remove(pos);
                    } else {
                        s.pop();
                    }
                }
                out.push(BTreeMap::new());
            }
            _ => {
                // Merge the thread's active context frames (outer -> inner).
                let mut m = BTreeMap::new();
                if let Some(s) = stacks.get(&e.thread) {
                    for mid in s {
                        if let Some(kvs) = meta.get(mid) {
                            for (k, v) in kvs {
                                m.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
                out.push(m);
            }
        }
    }
    out
}

/// Convert a recording into a Perfetto / Chrome JSON trace: a live-heap-bytes
/// counter (overall + per type) over time, plus an async slice per allocation
/// lifetime (alloc -> free), named by recovered type. Open it at
/// https://ui.perfetto.dev.
fn cmd_perfetto(args: &[String]) -> Result<(), String> {
    let file = positional(args).ok_or("usage: memscope perfetto <FILE> [--out trace.json]")?;
    let out = flag(args, "--out").unwrap_or("trace.json").to_string();

    let rec = read_recording(file)?;
    let labels = &rec.sites;
    let events = &rec.events;

    // Build the trace incrementally.
    let mut te: Vec<String> = Vec::new();
    let label_of =
        |site: u32| -> &str { labels.get(&site).map(|s| s.label.as_str()).unwrap_or("<unknown>") };
    let us = |ts: u64| (ts as f64) / 1000.0; // ns -> µs for Chrome format

    // Process / thread metadata.
    te.push(r#"{"name":"process_name","ph":"M","pid":1,"args":{"name":"heap"}}"#.to_string());
    let mut threads: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for e in events {
        threads.insert(e.thread);
    }
    for t in &threads {
        te.push(format!(
            "{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{t},\"args\":{{\"name\":\"thread {t}\"}}}}"
        ));
    }

    // Live-bytes counters (total + per type) + async lifetime slices.
    let mut live: HashMap<u64, (u64, u64, u64)> = HashMap::new(); // addr -> (size, slice_id, alloc_ts)
    let mut total: u64 = 0;
    let mut per_type: HashMap<String, u64> = HashMap::new();
    let mut slice_id: u64 = 0;
    let mut slices_emitted = 0usize;
    let mut last_ts = 0u64;
    let mut last_ctr_ts = u64::MAX; // dedup counter samples to one per (ms) timestamp
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");

    for e in events {
        last_ts = e.ts_nanos.max(last_ts);
        match e.kind {
            memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow => {
                let label = label_of(e.site).to_string();
                total += e.size;
                *per_type.entry(label.clone()).or_default() += e.size;
                let id = slice_id;
                slice_id += 1;
                live.insert(e.addr, (e.size, id, e.ts_nanos));
                slices_emitted += 1;
                te.push(format!(
                    "{{\"ph\":\"b\",\"cat\":\"alloc\",\"name\":\"{}\",\"id\":{},\"ts\":{:.3},\"pid\":1,\"tid\":{},\"args\":{{\"size\":{},\"addr\":\"{:#x}\"}}}}",
                    esc(&label), id, us(e.ts_nanos), e.thread, e.size, e.addr
                ));
            }
            memscope_proto::EventKind::Dealloc => {
                if let Some((size, id, _)) = live.remove(&e.addr) {
                    total = total.saturating_sub(size);
                    te.push(format!(
                        "{{\"ph\":\"e\",\"cat\":\"alloc\",\"name\":\"\",\"id\":{},\"ts\":{:.3},\"pid\":1,\"tid\":{}}}",
                        id, us(e.ts_nanos), e.thread
                    ));
                }
            }
            memscope_proto::EventKind::MetaEnter
            | memscope_proto::EventKind::MetaExit
            | memscope_proto::EventKind::Mark => {}
        }
        // One live_bytes counter sample per distinct timestamp (the clock is
        // ms-granular, so this collapses the within-ms burst to one point).
        if e.ts_nanos != last_ctr_ts {
            te.push(format!(
                "{{\"ph\":\"C\",\"name\":\"live_bytes\",\"ts\":{:.3},\"pid\":1,\"args\":{{\"bytes\":{}}}}}",
                us(e.ts_nanos), total
            ));
            last_ctr_ts = e.ts_nanos;
        }
    }
    // Close out still-live allocations at the end of the trace.
    let end = us(last_ts + 1_000_000);
    for (_addr, (_size, id, _ts)) in &live {
        te.push(format!(
            "{{\"ph\":\"e\",\"cat\":\"alloc\",\"name\":\"\",\"id\":{},\"ts\":{:.3},\"pid\":1,\"tid\":0}}",
            id, end
        ));
    }

    let json = format!("{{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n{}\n]}}", te.join(",\n"));
    std::fs::write(&out, json).map_err(|e| e.to_string())?;

    println!("wrote Perfetto trace: {out}");
    println!(
        "  {} events  ->  {} async slices (every allocation) + live_bytes counter  ({} threads)",
        events.len(),
        slices_emitted,
        threads.len()
    );
    println!("  open it at https://ui.perfetto.dev  (Open trace file)");
    Ok(())
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
    let file = positional(args).ok_or("usage: memscope flamegraph <FILE> [--out F] [--format chrome|folded] [--by bytes|count] [--live] [--no-std] [--group-by KEY] [--filter KEY=VAL]")?;
    let format = flag(args, "--format").unwrap_or("chrome");
    let by_count = flag(args, "--by") == Some("count");
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
    let rec = read_recording_raw(file)?;
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
    let events = &rec.events;
    let meta_active = group_by.is_some() || !filters.is_empty();

    let mut root = Flame::default();
    let mut interner = Interner::default();
    if meta_active {
        // Per-allocation pass: each alloc carries its correlated metadata, so the
        // same call site can land under different group values.
        let metas = correlate_meta(events, &rec.meta);
        let live_idx = if live_only { Some(live_indices(events)) } else { None };
        for (i, e) in events.iter().enumerate() {
            if !matches!(
                e.kind,
                memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow
            ) {
                continue;
            }
            if let Some(ls) = &live_idx {
                if !ls.contains(&i) {
                    continue;
                }
            }
            let m = &metas[i];
            if !filters
                .iter()
                .all(|(k, v)| m.get(k).map(|x| x == v).unwrap_or(false))
            {
                continue;
            }
            let Some((frames, label)) = site_frames(e.site) else { continue };
            let w = if by_count { 1 } else { e.size };
            if w == 0 {
                continue;
            }
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
        // Fast path: aggregate per site (no metadata needed).
        let mut live: HashMap<u64, (u64, u32)> = HashMap::new(); // for --live
        let mut site_bytes: HashMap<u32, u64> = HashMap::new();
        let mut site_count: HashMap<u32, u64> = HashMap::new();
        for e in events {
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
fn cmd_flamechart(args: &[String]) -> Result<(), String> {
    let file = positional(args)
        .ok_or("usage: memscope flamechart <FILE> [--out F] [--no-std]")?;
    let out = flag(args, "--out").unwrap_or("alloc-flamechart.json").to_string();
    let no_std = args.iter().any(|a| a == "--no-std" || a == "--exclude-std");

    let rec = read_recording(file)?;
    let sites = &rec.sites;
    let events = &rec.events;

    // Cleaned root->leaf path per site (reversed frames + the type as the leaf).
    let mut site_path: HashMap<u32, Vec<String>> = HashMap::new();
    for (site, info) in sites {
        site_path.insert(*site, site_to_path(&info.frames, &info.label, no_std));
    }

    // Every allocation, in order, grouped per thread — no sampling, no caps.
    // Each sample carries (virtual time, site, allocation size).
    let mut per_thread: std::collections::BTreeMap<u32, Vec<(u64, u32, u64)>> =
        std::collections::BTreeMap::new();
    let mut vt: u64 = 0;
    for e in events {
        vt = (vt + 1).max(e.ts_nanos);
        if matches!(
            e.kind,
            memscope_proto::EventKind::Alloc | memscope_proto::EventKind::ReallocGrow
        ) {
            per_thread.entry(e.thread).or_default().push((vt, e.site, e.size));
        }
    }

    let mut te: Vec<String> = Vec::new();
    te.push(r#"{"name":"process_name","ph":"M","pid":1,"args":{"name":"allocation timeline"}}"#.into());
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let us = |t: u64| (t as f64) / 1000.0;
    let mut total_samples = 0u64;

    for (tid, samples) in &per_thread {
        te.push(format!(
            "{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{tid},\"args\":{{\"name\":\"thread {tid}\"}}}}"
        ));
        // Currently-open stack of frame names. We emit B(egin)/E(nd) pairs (not
        // X) so nesting comes from event order, not timestamp containment — a
        // linear chain otherwise produces identical [ts,dur] intervals that
        // importers can't nest (they fall back to alphabetical).
        let mut open: Vec<&str> = Vec::new();
        // Bytes allocated through each open frame (its `args.bytes`, emitted on E).
        let mut open_bytes: Vec<u64> = Vec::new();
        let empty: Vec<String> = Vec::new();
        for (vt, site, size) in samples {
            total_samples += 1;
            let path = site_path.get(site).unwrap_or(&empty);
            // Common prefix with the currently-open stack.
            let mut common = 0;
            while common < open.len()
                && common < path.len()
                && open[common] == path[common].as_str()
            {
                common += 1;
            }
            // Close diverged frames (deepest first), reporting their total bytes.
            while open.len() > common {
                let name = open.pop().unwrap();
                let bytes = open_bytes.pop().unwrap();
                te.push(format!(
                    "{{\"ph\":\"E\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{:.3},\"pid\":1,\"tid\":{},\"args\":{{\"bytes\":{}}}}}",
                    esc(name), us(*vt), tid, bytes
                ));
            }
            // Open new frames (shallowest first).
            for name in &path[common..] {
                te.push(format!(
                    "{{\"ph\":\"B\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{:.3},\"pid\":1,\"tid\":{}}}",
                    esc(name), us(*vt), tid
                ));
                open.push(name.as_str());
                open_bytes.push(0);
            }
            // This allocation passes through every currently-open frame.
            for b in open_bytes.iter_mut() {
                *b += size;
            }
        }
        // Close whatever remains at the thread's last timestamp.
        let end = samples.last().map(|(vt, _, _)| *vt + 1).unwrap_or(1);
        while let Some(name) = open.pop() {
            let bytes = open_bytes.pop().unwrap();
            te.push(format!(
                "{{\"ph\":\"E\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{:.3},\"pid\":1,\"tid\":{},\"args\":{{\"bytes\":{}}}}}",
                esc(name), us(end), tid, bytes
            ));
        }
    }

    let json = format!(
        "{{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n{}\n]}}",
        te.join(",\n")
    );
    std::fs::write(&out, json).map_err(|e| e.to_string())?;
    println!("wrote allocation flame chart (timeline): {out}");
    println!(
        "  {} allocations across {} threads -> {} slices (every allocation, adjacent identical stacks merged)",
        total_samples,
        per_thread.len(),
        te.iter().filter(|s| s.contains("\"ph\":\"B\"")).count()
    );
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
    let mut client = Client::connect(&sock).map_err(|e| e.to_string())?;
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
    let rec = read_recording(file)?;
    let tl = Timeline::new(&rec);

    // For each mark, reconstruct the live set and its top types by bytes.
    let rows: Vec<(memscope_replay::MarkPoint, u64, Vec<(String, u64)>)> = tl
        .marks()
        .iter()
        .map(|m| {
            let st = tl.state_at_index(m.index + 1, None);
            let mut by_type: HashMap<String, u64> = HashMap::new();
            for a in &st.live {
                *by_type.entry(rec.site_label(a.site).to_string()).or_default() += a.size;
            }
            let mut top: Vec<(String, u64)> = by_type.into_iter().collect();
            top.sort_by(|a, b| b.1.cmp(&a.1));
            top.truncate(3);
            (m.clone(), st.total_live_bytes, top)
        })
        .collect();

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

/// One endpoint of a diff: a named live-set state plus where it sits in time.
struct Endpoint {
    name: String,
    ts_nanos: u64,
    state: memscope_replay::LiveState,
}

/// Resolve `start` / `end` / a mark label to a reconstructed [`Endpoint`].
fn resolve_endpoint(tl: &Timeline, rec: &Recording, name: &str) -> Result<Endpoint, String> {
    match name {
        "start" => Ok(Endpoint {
            name: name.into(),
            ts_nanos: 0,
            state: tl.state_at_index(0, None),
        }),
        "end" => {
            let ts = rec.events.last().map(|e| e.ts_nanos).unwrap_or(0);
            Ok(Endpoint { name: name.into(), ts_nanos: ts, state: tl.state_at_end() })
        }
        label => {
            let state = tl
                .state_at(label)
                .ok_or_else(|| format!("no checkpoint labeled {label:?} (try `memscope marks <FILE>`)"))?;
            let ts = state.at.as_ref().map(|m| m.ts_nanos).unwrap_or(0);
            Ok(Endpoint { name: label.into(), ts_nanos: ts, state })
        }
    }
}

/// Aggregate a live set by site id -> (count, bytes).
fn agg_by_site(state: &memscope_replay::LiveState) -> HashMap<u32, (u64, u64)> {
    let mut m: HashMap<u32, (u64, u64)> = HashMap::new();
    for a in &state.live {
        let e = m.entry(a.site).or_default();
        e.0 += 1;
        e.1 += a.size;
    }
    m
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

    let rec = read_recording(file)?;
    let tl = Timeline::new(&rec);
    let a = resolve_endpoint(&tl, &rec, a_name)?;
    let b = resolve_endpoint(&tl, &rec, b_name)?;
    if a.state.upto > b.state.upto {
        return Err(format!(
            "{a_name} occurs after {b_name} in the stream; pass them in chronological order"
        ));
    }

    // Born/freed per site within the window [a.upto, b.upto).
    let born_freed = tl.window_born_freed(&a.state, b.state.upto);

    let agg_a = agg_by_site(&a.state);
    let agg_b = agg_by_site(&b.state);
    let mut sites: Vec<u32> = agg_a.keys().chain(agg_b.keys()).copied().collect();
    sites.sort_unstable();
    sites.dedup();

    let mut grew: Vec<DiffRow> = Vec::new();
    let mut shrank: Vec<DiffRow> = Vec::new();
    let mut unchanged = 0u32;
    for s in sites {
        let (ac, ab) = agg_a.get(&s).copied().unwrap_or((0, 0));
        let (bc, bb) = agg_b.get(&s).copied().unwrap_or((0, 0));
        let dc = bc as i64 - ac as i64;
        let db = bb as i64 - ab as i64;
        if dc == 0 && db == 0 {
            unchanged += 1;
            continue;
        }
        let (born, freed) = born_freed.get(&s).copied().unwrap_or((0, 0));
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

    let net = b.state.total_live_bytes as i64 - a.state.total_live_bytes as i64;
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
        human_bytes(a.state.total_live_bytes),
        human_bytes(b.state.total_live_bytes),
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
        a.state.total_live_bytes,
        jesc(&b.name),
        b.ts_nanos / 1_000_000,
        b.state.total_live_bytes,
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
    let findings = analyze(&rec);
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
    // Default trigger: dump at exit if nothing else was requested.
    let default_on_exit = !on_exit && after.is_none() && at_bytes.is_none();

    let lib = preload_lib()?;
    let env_var = if cfg!(target_os = "macos") { "DYLD_INSERT_LIBRARIES" } else { "LD_PRELOAD" };

    let mut child = std::process::Command::new(&cmd[0]);
    child.args(&cmd[1..]);
    child.env(env_var, &lib);
    if let Some(o) = out {
        child.env("MEMSCOPE_HPROF_OUT", o);
    }
    if on_exit || default_on_exit {
        child.env("MEMSCOPE_HPROF_ON_EXIT", "1");
    }
    if let Some(n) = at_bytes {
        child.env("MEMSCOPE_HPROF_AT_BYTES", n);
    }

    eprintln!("[memscope] launching `{}` under memscope ({env_var})", cmd.join(" "));
    let mut handle = child.spawn().map_err(|e| format!("failed to launch {}: {e}", cmd[0]))?;
    let pid = handle.id();

    // Timed trigger: signal the child after the delay (it stays running).
    if let Some(d) = after {
        let dur = parse_dur(d).ok_or_else(|| format!("bad --after duration {d:?}"))?;
        std::thread::spawn(move || {
            std::thread::sleep(dur);
            eprintln!("[memscope] --after elapsed — signalling pid {pid} for a dump");
            unsafe {
                libc::kill(pid as i32, libc::SIGUSR1);
            }
        });
    }

    let status = handle.wait().map_err(|e| e.to_string())?;
    // Give the dumper thread a moment to finish an exit/threshold dump.
    std::thread::sleep(std::time::Duration::from_millis(300));

    let expected = match out {
        Some(o) => o.replace("{pid}", &pid.to_string()).replace("{n}", "0"),
        None => format!("/tmp/memscope-{pid}-0.hprof"),
    };
    if std::path::Path::new(&expected).exists() {
        eprintln!("[memscope] heap dump: {expected}");
        eprintln!("           analyze it:  memscope analyze {expected}   (or open in MAT / heapster)");
    } else {
        eprintln!("[memscope] no dump found at {expected} — was a trigger reached? (--on-exit/--after/--at-bytes)");
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

    let rec = read_recording(file)?;
    let stats = site_stats(&rec);

    // The matching per-site stats (one for --site, all of a type for --type).
    let matched: Vec<&SiteStats> = stats
        .iter()
        .filter(|s| match (site_arg, type_arg) {
            (Some(id), _) => s.site == id,
            (_, Some(ty)) => s.label == ty,
            _ => false,
        })
        .collect();
    if matched.is_empty() {
        return Err(match (site_arg, type_arg) {
            (Some(id), _) => format!("no site {id} in {file}"),
            (_, Some(ty)) => format!("no allocations of type {ty:?} in {file}"),
            _ => unreachable!(),
        });
    }

    match field {
        "stack" => query_stack(&rec, &matched, json),
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
    rows.sort_by(|a, b| b.0.alloc_bytes.cmp(&a.0.alloc_bytes));
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

    #[test]
    fn correlate_nesting_and_per_thread() {
        // thread 10 nests ctx1(physics) > ctx2(shard=3); thread 20 runs ctx3(io)
        // concurrently and interleaved in the global stream.
        let events = vec![
            ev(MetaEnter, 10, 1, 0),
            ev(MetaEnter, 20, 3, 0),
            ev(MetaEnter, 10, 2, 0),
            ev(Alloc, 10, 99, 0x1), // idx 3: {subsystem:physics, shard:3}
            ev(Alloc, 20, 99, 0x2), // idx 4: {subsystem:io} (other thread unaffected)
            ev(MetaExit, 10, 2, 0),
            ev(Alloc, 10, 99, 0x3), // idx 6: {subsystem:physics}
            ev(MetaExit, 10, 1, 0),
            ev(Alloc, 10, 99, 0x4), // idx 8: {}
        ];
        let m = correlate_meta(&events, &meta_tables());
        assert_eq!(m[3].get("subsystem").map(String::as_str), Some("physics"));
        assert_eq!(m[3].get("shard").map(String::as_str), Some("3"));
        assert_eq!(m[4].get("subsystem").map(String::as_str), Some("io"));
        assert!(m[4].get("shard").is_none()); // thread 20 never saw shard
        assert_eq!(m[6].get("subsystem").map(String::as_str), Some("physics"));
        assert!(m[6].get("shard").is_none()); // ctx2 exited
        assert!(m[8].is_empty()); // all scopes exited
    }

    #[test]
    fn inner_scope_overrides_outer_key() {
        let mut tables = HashMap::new();
        tables.insert(1, vec![("subsystem".to_string(), "outer".to_string())]);
        tables.insert(2, vec![("subsystem".to_string(), "inner".to_string())]);
        let events = vec![
            ev(MetaEnter, 1, 1, 0),
            ev(MetaEnter, 1, 2, 0),
            ev(Alloc, 1, 9, 0x1), // idx 2: inner wins (merged outer->inner)
        ];
        let m = correlate_meta(&events, &tables);
        assert_eq!(m[2].get("subsystem").map(String::as_str), Some("inner"));
    }

    #[test]
    fn live_indices_handles_free_and_reuse() {
        let events = vec![
            ev(Alloc, 1, 0, 0xA),   // 0
            ev(Alloc, 1, 0, 0xB),   // 1
            ev(Dealloc, 1, 0, 0xA), // frees 0xA
            ev(Alloc, 1, 0, 0xA),   // 3: address reused -> this one is live
        ];
        let live = live_indices(&events);
        assert!(!live.contains(&0)); // freed
        assert!(live.contains(&1)); // never freed
        assert!(live.contains(&3)); // the reuse
        assert_eq!(live.len(), 2);
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
