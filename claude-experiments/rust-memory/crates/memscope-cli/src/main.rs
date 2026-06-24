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
         memscope replay  <FILE>                        read a record_to_file recording\n"
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

/// Reference reader for a `record_to_file` recording: replays the newline-JSON
/// event stream, reconstructs the live set, and summarizes it. A starting point
/// for building richer viewers — the format is documented in memscope-agent.
fn cmd_replay(args: &[String]) -> Result<(), String> {
    use std::io::BufRead;
    let file = positional(args).ok_or("usage: memscope replay <FILE>")?;
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
