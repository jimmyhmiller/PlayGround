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
         \x20                                            (--no-std strips std/core/alloc/runtime frames)\n"
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

    let mut site_label: HashMap<u32, String> = HashMap::new();
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
                        // Metadata markers aren't allocations.
                        memscope_proto::EventKind::MetaEnter | memscope_proto::EventKind::MetaExit => {}
                    }
                }
            }
            recfmt::TAG_KEY => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let _ = rd_str(&mut f);
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

/// A decoded event from a recording (either format).
struct RecEvent {
    kind: memscope_proto::EventKind,
    addr: u64,
    size: u64,
    ts_nanos: u64,
    site: u32,
    thread: u32,
}

/// Read one encoded `MetaValue` from a stream, returning its display string
/// (also used to skip the bytes when the value isn't needed).
fn read_meta_value(f: &mut impl std::io::Read) -> Option<String> {
    let mut t = [0u8; 1];
    f.read_exact(&mut t).ok()?;
    let mut b8 = [0u8; 8];
    match t[0] {
        0 => {
            let mut l = [0u8; 2];
            f.read_exact(&mut l).ok()?;
            let n = u16::from_le_bytes(l) as usize;
            let mut s = vec![0u8; n];
            f.read_exact(&mut s).ok()?;
            Some(String::from_utf8_lossy(&s).into_owned())
        }
        1 => {
            f.read_exact(&mut b8).ok()?;
            Some(i64::from_le_bytes(b8).to_string())
        }
        2 => {
            f.read_exact(&mut b8).ok()?;
            Some(u64::from_le_bytes(b8).to_string())
        }
        3 => {
            f.read_exact(&mut b8).ok()?;
            Some(f64::from_bits(u64::from_le_bytes(b8)).to_string())
        }
        4 => {
            let mut b1 = [0u8; 1];
            f.read_exact(&mut b1).ok()?;
            Some((b1[0] != 0).to_string())
        }
        _ => None,
    }
}

/// One resolved stack frame: function name + source location.
#[derive(Default, Clone)]
struct FrameMeta {
    func: String,
    file: String,
    line: u32,
}

/// Resolved info for one allocation site: its type label and its call stack
/// (innermost first — as recorded).
#[derive(Default, Clone)]
struct SiteInfo {
    label: String,
    frames: Vec<FrameMeta>,
}

/// A parsed recording: per-site info, per-context metadata, and the events.
#[derive(Default)]
struct Recording {
    sites: HashMap<u32, SiteInfo>,
    /// meta context id -> resolved [(key, value)] pairs.
    meta: HashMap<u32, Vec<(String, String)>>,
    events: Vec<RecEvent>,
}

/// Read a recording (binary `.mscope` or JSON) into sites, metadata, and events.
fn read_recording(file: &str) -> Result<Recording, String> {
    use std::io::Read;
    let mut magic = [0u8; 4];
    {
        let mut f = std::fs::File::open(file).map_err(|e| e.to_string())?;
        let _ = f.read(&mut magic);
    }
    if memscope_proto::recfmt::is_binary(&magic) {
        read_recording_binary(file)
    } else {
        read_recording_json(file)
    }
}

fn read_recording_binary(file: &str) -> Result<Recording, String> {
    use memscope_proto::recfmt;
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(file).map_err(|e| e.to_string())?);
    let mut b1 = [0u8; 1];
    let mut b2 = [0u8; 2];
    let mut b4 = [0u8; 4];
    let rd_str = |f: &mut std::io::BufReader<std::fs::File>| -> Option<String> {
        let mut l = [0u8; 2];
        f.read_exact(&mut l).ok()?;
        let n = u16::from_le_bytes(l) as usize;
        let mut s = vec![0u8; n];
        f.read_exact(&mut s).ok()?;
        Some(String::from_utf8_lossy(&s).into_owned())
    };
    if f.read_exact(&mut b4).is_err() || b4 != recfmt::MAGIC {
        return Err("not a memscope binary recording".into());
    }
    let _ = f.read_exact(&mut b2);
    let _ = f.read_exact(&mut b2);
    let _ = f.read_exact(&mut b4);
    let _exe = rd_str(&mut f);

    let mut labels: HashMap<u32, SiteInfo> = HashMap::new();
    let mut events: Vec<RecEvent> = Vec::new();
    let mut key_names: HashMap<u32, String> = HashMap::new();
    let mut meta: HashMap<u32, Vec<(String, String)>> = HashMap::new();
    while f.read_exact(&mut b1).is_ok() {
        match b1[0] {
            recfmt::TAG_KEY => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let id = u32::from_le_bytes(b4);
                let name = rd_str(&mut f).unwrap_or_default();
                key_names.insert(id, name);
            }
            recfmt::TAG_META => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let id = u32::from_le_bytes(b4);
                f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                let mut kvs = Vec::new();
                for _ in 0..u16::from_le_bytes(b2) {
                    f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                    let kid = u32::from_le_bytes(b4);
                    let val = read_meta_value(&mut f).unwrap_or_default();
                    let key = key_names.get(&kid).cloned().unwrap_or_else(|| kid.to_string());
                    kvs.push((key, val));
                }
                meta.insert(id, kvs);
            }
            recfmt::TAG_SITE => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let site = u32::from_le_bytes(b4);
                f.read_exact(&mut b1).map_err(|e| e.to_string())?;
                let ty = if b1[0] == 1 { rd_str(&mut f) } else { None };
                f.read_exact(&mut b1).map_err(|e| e.to_string())?;
                let shape = recfmt::shape_from_code(b1[0]);
                f.read_exact(&mut b2).map_err(|e| e.to_string())?;
                let mut frames = Vec::new();
                for _ in 0..u16::from_le_bytes(b2) {
                    let func = rd_str(&mut f).unwrap_or_default();
                    let file = rd_str(&mut f).unwrap_or_default();
                    f.read_exact(&mut b4).ok();
                    let line = u32::from_le_bytes(b4);
                    let _ = f.read_exact(&mut b1); // inlined
                    frames.push(FrameMeta { func, file, line });
                }
                labels.insert(
                    site,
                    SiteInfo {
                        label: label_for(shape, ty),
                        frames,
                    },
                );
            }
            recfmt::TAG_EVENTS => {
                f.read_exact(&mut b4).map_err(|e| e.to_string())?;
                let count = u32::from_le_bytes(b4);
                let mut rec = vec![0u8; recfmt::EVENT_BYTES * count as usize];
                f.read_exact(&mut rec).map_err(|e| e.to_string())?;
                let mut r = recfmt::Reader::new(&rec);
                for _ in 0..count {
                    let Some(e) = r.decode_event() else { break };
                    events.push(RecEvent {
                        kind: e.kind,
                        addr: e.addr,
                        size: e.size,
                        ts_nanos: e.ts_nanos,
                        site: e.site.0,
                        thread: e.thread,
                    });
                }
            }
            other => return Err(format!("corrupt recording: unknown tag {other}")),
        }
    }
    Ok(Recording {
        sites: labels,
        meta,
        events,
    })
}

fn read_recording_json(file: &str) -> Result<Recording, String> {
    use std::io::BufRead;
    let rdr = std::io::BufReader::new(std::fs::File::open(file).map_err(|e| e.to_string())?);
    let mut labels: HashMap<u32, SiteInfo> = HashMap::new();
    let mut events: Vec<RecEvent> = Vec::new();
    let mut meta: HashMap<u32, Vec<(String, String)>> = HashMap::new();
    for line in rdr.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("v").is_some() {
            continue;
        }
        // Metadata context definition: {"meta":id,"kv":{k:v,...}}
        if let Some(id) = v.get("meta").and_then(|x| x.as_u64()) {
            if let Some(obj) = v.get("kv").and_then(|x| x.as_object()) {
                let kvs = obj
                    .iter()
                    .map(|(k, val)| {
                        let vs = match val {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        (k.clone(), vs)
                    })
                    .collect();
                meta.insert(id as u32, kvs);
            }
            continue;
        }
        if let Some(site) = v.get("site").and_then(|x| x.as_u64()) {
            let ty = v.get("ty").and_then(|x| x.as_str()).map(String::from);
            let shape = v.get("shape").and_then(|x| x.as_str());
            let label = match (shape, &ty) {
                (Some(sh), Some(t)) => format!("{sh}<{t}>"),
                (Some(sh), None) => format!("{sh}<?>"),
                (None, Some(t)) => t.clone(),
                (None, None) => "<no type>".into(),
            };
            let frames = v
                .get("frames")
                .and_then(|f| f.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|fr| {
                            let a = fr.as_array()?;
                            Some(FrameMeta {
                                func: a.first()?.as_str()?.to_string(),
                                file: a.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string(),
                                line: a.get(2).and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            labels.insert(site as u32, SiteInfo { label, frames });
            continue;
        }
        let kind = match v.get("k").and_then(|x| x.as_str()) {
            Some("A") => memscope_proto::EventKind::Alloc,
            Some("R") => memscope_proto::EventKind::ReallocGrow,
            Some("D") => memscope_proto::EventKind::Dealloc,
            Some("M") => memscope_proto::EventKind::MetaEnter,
            Some("m") => memscope_proto::EventKind::MetaExit,
            _ => continue,
        };
        events.push(RecEvent {
            kind,
            addr: v.get("a").and_then(|x| x.as_u64()).unwrap_or(0),
            size: v.get("sz").and_then(|x| x.as_u64()).unwrap_or(0),
            ts_nanos: v.get("ts").and_then(|x| x.as_u64()).unwrap_or(0),
            site: v.get("s").and_then(|x| x.as_u64()).unwrap_or(u32::MAX as u64) as u32,
            thread: v.get("t").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        });
    }
    Ok(Recording {
        sites: labels,
        meta,
        events,
    })
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

fn label_for(shape: Option<memscope_proto::AllocShape>, ty: Option<String>) -> String {
    match (shape, ty) {
        (Some(sh), Some(t)) => format!("{sh:?}<{t}>"),
        (Some(sh), None) => format!("{sh:?}<?>"),
        (None, Some(t)) => t,
        (None, None) => "<no type>".into(),
    }
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
            memscope_proto::EventKind::MetaEnter | memscope_proto::EventKind::MetaExit => {}
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

/// Is this a Rust stdlib / language-runtime frame (std/core/alloc, the lang-start
/// + panic machinery, `FnOnce`/`Fn` shims, pthread/libc/C entry)? Used by
/// `--no-std` to collapse the boilerplate and leave only application frames.
fn is_std_frame(name: &str) -> bool {
    let n = name.trim_start_matches(['<', '&', ' ']);
    const PREFIXES: &[&str] = &[
        "std::", "core::", "alloc::", "proc_macro::", "test::", "backtrace::",
        // hashbrown / allocator_api2 are std's own HashMap + allocator shim —
        // separate crates, but pure stdlib plumbing for our purposes.
        "hashbrown::", "allocator_api2::",
        "__rust", "_rust", "rustc", "__pthread", "_pthread", "pthread", "__libc",
        "libc::", "_main", "dyld", "start_",
    ];
    if PREFIXES.iter().any(|p| n.starts_with(p)) {
        return true;
    }
    // Trait impls written `<T as std::…>` / `… as core::…` and the runtime shims.
    name.contains(" as std::")
        || name.contains(" as core::")
        || name.contains(" as alloc::")
        || name.contains("rust_begin_short_backtrace")
        || name.contains("lang_start")
        || name.contains("catch_unwind")
        || name.contains("::panicking::")
        || name.contains("ops::function::Fn")
        || name.contains("call_once")
        || name == "_main"
        || name == "main"
        || name == "start"
        || name == "[unknown]"
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

/// Strip a trailing rustc hash (`::h0123abcd…`) and crate disambiguators from a
/// frame name for readable flame labels.
fn clean_frame(f: &str) -> String {
    let mut s = f;
    if let Some(idx) = s.rfind("::h") {
        if s[idx + 3..].len() >= 8 && s[idx + 3..].bytes().all(|b| b.is_ascii_hexdigit()) {
            s = &s[..idx];
        }
    }
    // Drop crate-hash decorations like `alloc[5fb2…]` -> `alloc`.
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' {
            // skip a `[...]` disambiguator only if it looks like a hash
            let mut inner = String::new();
            while let Some(&n) = chars.peek() {
                if n == ']' {
                    chars.next();
                    break;
                }
                inner.push(n);
                chars.next();
            }
            if !inner.chars().all(|c| c.is_ascii_hexdigit()) {
                out.push('[');
                out.push_str(&inner);
                out.push(']');
            }
        } else {
            out.push(c);
        }
    }
    if out.is_empty() {
        return "[unknown]".to_string();
    }
    out
}

/// A node in the allocation flame tree (merged call stacks, weighted by bytes).
#[derive(Default)]
struct Flame {
    bytes: u64,
    samples: u64,
    children: std::collections::BTreeMap<String, Flame>,
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

    let rec = read_recording(file)?;
    let sites = &rec.sites;
    let events = &rec.events;
    let meta_active = group_by.is_some() || !filters.is_empty();

    let mut root = Flame::default();
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
            let Some(info) = sites.get(&e.site) else { continue };
            let w = if by_count { 1 } else { e.size };
            if w == 0 {
                continue;
            }
            root.bytes += w;
            root.samples += 1;
            let mut node = &mut root;
            if let Some(gk) = &group_by {
                let gv = m.get(gk).cloned().unwrap_or_else(|| "<none>".to_string());
                node = node.children.entry(format!("{gk}={gv}")).or_default();
                node.bytes += w;
                node.samples += 1;
            }
            for name in site_to_path(&info.frames, &info.label, no_std) {
                node = node.children.entry(name).or_default();
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
        for (site, info) in sites {
            let w = if by_count {
                *site_count.get(site).unwrap_or(&0)
            } else {
                *site_bytes.get(site).unwrap_or(&0)
            };
            if w == 0 {
                continue;
            }
            root.bytes += w;
            root.samples += 1;
            let mut node = &mut root;
            for name in site_to_path(&info.frames, &info.label, no_std) {
                node = node.children.entry(name).or_default();
                node.bytes += w;
                node.samples += 1;
            }
        }
    }

    if format == "folded" {
        let mut out_s = String::new();
        fold(&root, &mut Vec::new(), by_count, &mut out_s);
        std::fs::write(&out, out_s).map_err(|e| e.to_string())?;
        println!("wrote folded stacks: {out}");
        println!("  pipe to inferno-flamegraph / flamegraph.pl, or load at speedscope.app");
        return Ok(());
    }

    // Chrome trace: nested X (complete) duration events; ts/dur in "bytes" so the
    // width of each frame is its total allocated bytes.
    let mut te: Vec<String> = Vec::new();
    te.push(r#"{"name":"process_name","ph":"M","pid":1,"args":{"name":"allocations by stack"}}"#.into());
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    emit_flame(&root, "all allocations", 0, &esc, &mut te);

    let json = format!(
        "{{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n{}\n]}}",
        te.join(",\n")
    );
    std::fs::write(&out, json).map_err(|e| e.to_string())?;
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

fn fold(node: &Flame, stack: &mut Vec<String>, by_count: bool, out: &mut String) {
    // Self weight = node weight minus children weight (allocations terminating here).
    let child_sum: u64 = node
        .children
        .values()
        .map(|c| if by_count { c.samples } else { c.bytes })
        .sum();
    let self_w = if by_count { node.samples } else { node.bytes }.saturating_sub(child_sum);
    if self_w > 0 && !stack.is_empty() {
        out.push_str(&stack.join(";"));
        out.push(' ');
        out.push_str(&self_w.to_string());
        out.push('\n');
    }
    for (name, child) in &node.children {
        stack.push(name.clone());
        fold(child, stack, by_count, out);
        stack.pop();
    }
}

fn emit_flame(
    node: &Flame,
    name: &str,
    x: u64,
    esc: &impl Fn(&str) -> String,
    te: &mut Vec<String>,
) {
    // Begin/End pair (not a single X): a linear chain has parent and child with
    // the *same* [ts,dur], which X-importers can't nest — they fall back to
    // alphabetical. B/E nests by event order, so call order is preserved.
    let w = node.bytes.max(1);
    te.push(format!(
        "{{\"ph\":\"B\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{},\"pid\":1,\"tid\":0,\"args\":{{\"bytes\":{},\"allocs\":{}}}}}",
        esc(name), x, node.bytes, node.samples
    ));
    let mut cx = x;
    for (child_name, child) in &node.children {
        emit_flame(child, child_name, cx, esc, te);
        cx += child.bytes.max(1);
    }
    te.push(format!(
        "{{\"ph\":\"E\",\"name\":\"{}\",\"cat\":\"alloc\",\"ts\":{},\"pid\":1,\"tid\":0}}",
        esc(name),
        x + w
    ));
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
