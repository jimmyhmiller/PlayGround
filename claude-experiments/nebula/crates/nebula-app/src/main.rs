//! `nebula` — GPU graph viewer CLI.
//!
//! Builds a graph (generated or loaded), seeds positions, and hands it to the
//! GPU viewer. All layout and rendering then happen on the GPU.

use std::time::Instant;

use nebula_core::{formats, generate, Graph, Pos};
use nebula_layout::{Layout, RandomLayout};
use nebula_render::{App, ColorMode, FilterOp, LayoutSettings, RunOptions};

const HELP: &str = "\
nebula — the GPU graph viewer

USAGE:
    nebula [GENERATOR] [OPTIONS]

GENERATORS (pick one; default: --ba 50000 3):
    --grid <W> <H>            2D lattice, W*H nodes
    --random <N> <M>          Erdos-Renyi: N nodes, M random edges
    --ba <N> <M>              Barabasi-Albert scale-free: N nodes, m=M attachments
    --blocks <N> <K> <M>      Stochastic block model: N nodes, K communities, M edges
    --geo <N> <R>             Random geometric: N nodes, connect within radius R (0..1)
    --file <PATH>             Load a graph file (edge-list/CSV/mtx/DIMACS/GML/DOT/JSON/adjacency)

OPTIONS:
    --color <MODE>            uniform|components|degree|pagerank|coloring|communities
    --k <FLOAT>               Optimal edge length (default 30)
    --seed <INT>              RNG seed (default 42)
    --dt <FLOAT>              Simulation timestep
    --substeps <INT>          Sim substeps per frame (default 1)
    --paused                  Start with the simulation paused
    --labels                  Show node labels (graphs up to 50k nodes)
    --select <INDEX>          Preselect a node
    --no-edges / --no-nodes   Hide edges / nodes
    --frames <N>              Exit after N frames (headless)
    --screenshot <PATH>       Save a PNG of the final frame (headless)
    --help-overlay            Start with the controls overlay visible
    -h, --help                Show this help

CONTROLS (in the window):
    drag pan / scroll zoom / F fit / click a node to inspect / double-click to focus
    1..6           color: uniform / components / degree / pagerank / coloring / communities
    R / G / O      re-seed random / grid / circle
    E / N          toggle edges / nodes      L   labels      S   save screenshot
    + / -          node size                 [ / ]  edge brightness
    space pause / H help / Tab hud / C clear selection / Esc quit
";

struct Args {
    generator: Gen,
    k: f32,
    seed: u64,
    dt: Option<f32>,
    substeps: u32,
    frames: Option<u64>,
    screenshot: Option<String>,
    color: ColorMode,
    /// Start colored by this node attribute (resolved to an index after load).
    color_attr: Option<String>,
    /// Startup "show only" filter, raw "KEY:OP:VALUE" (resolved after load).
    filter: Option<String>,
    /// Start in the hierarchical (layered DAG) layout.
    hierarchical: bool,
    /// Start in the radial (concentric DAG) layout.
    radial: bool,
    paused: bool,
    no_edges: bool,
    no_nodes: bool,
    select: Option<u32>,
    help_overlay: bool,
    labels: bool,
    aggregate: bool,
    node_size: f32,
}

enum Gen {
    Grid(u32, u32),
    Random(u64, u64),
    Ba(u64, u32),
    Blocks(u64, u32, u64),
    Geo(u64, f32),
    File(String),
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprint!("{HELP}");
            if !msg.is_empty() {
                eprintln!("\nerror: {msg}");
            }
            std::process::exit(if msg.is_empty() { 0 } else { 2 });
        }
    };

    // --- Build the graph ---------------------------------------------------
    let t0 = Instant::now();
    let mut node_labels: Option<Vec<String>> = None;
    let mut node_attrs: Option<Vec<Vec<(String, String)>>> = None;
    let (mut graph, what): (Graph, String) = match &args.generator {
        Gen::Grid(w, h) => (generate::grid_2d(*w, *h), format!("grid {w}x{h}")),
        Gen::Random(n, m) => (
            generate::erdos_renyi_m(*n, *m, args.seed),
            format!("erdos-renyi n={n} m={m}"),
        ),
        Gen::Ba(n, m) => (
            generate::barabasi_albert(*n, *m, args.seed),
            format!("barabasi-albert n={n} m={m}"),
        ),
        Gen::Blocks(n, k, m) => (
            generate::stochastic_blocks(*n, *k, *m, 0.03, args.seed),
            format!("blocks n={n} k={k} m={m}"),
        ),
        Gen::Geo(n, r) => (
            generate::random_geometric(*n, *r, args.seed),
            format!("geometric n={n} r={r}"),
        ),
        Gen::File(path) => {
            let loaded = formats::load(path)?;
            log::info!("detected format: {}", loaded.format);
            let label = format!(
                "{} [{}]",
                std::path::Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or(path),
                loaded.format
            );
            node_labels = Some(loaded.labels);
            if !loaded.attrs.is_empty() {
                node_attrs = Some(loaded.attrs);
            }
            (loaded.graph, label)
        }
    };
    // Attribute keys in first-seen order (matches the viewer's own ordering),
    // used to resolve --color-attr and --filter by name.
    let mut keys: Vec<&str> = Vec::new();
    if let Some(attrs) = node_attrs.as_ref() {
        for node in attrs {
            for (k, _) in node {
                if !keys.iter().any(|s| s == k) {
                    keys.push(k);
                }
            }
        }
    }

    // Resolve --color-attr to an attribute color index.
    let mut color_mode = args.color;
    if let Some(key) = &args.color_attr {
        match keys.iter().position(|k| *k == key) {
            Some(i) => color_mode = ColorMode::Attribute(i),
            None => anyhow::bail!("attribute '{key}' not found; available: {}", keys.join(", ")),
        }
    }

    // Resolve --filter "KEY:OP:VALUE" (op = contains/eq/ne/gt/ge/lt/le).
    let mut filter = None;
    if let Some(spec) = &args.filter {
        let mut parts = spec.splitn(3, ':');
        let key = parts.next().unwrap_or("");
        let op_name = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("").to_string();
        let idx = keys
            .iter()
            .position(|k| *k == key)
            .ok_or_else(|| anyhow::anyhow!("filter attribute '{key}' not found"))?;
        let op = FilterOp::from_name(op_name)
            .ok_or_else(|| anyhow::anyhow!("unknown filter op '{op_name}'"))?;
        filter = Some((idx, op, value));
    }
    log::info!(
        "built {what}: {} nodes, {} edges in {:.2}s",
        graph.num_nodes(),
        graph.num_edges(),
        t0.elapsed().as_secs_f32()
    );

    // Build CSR up front (needed by GPU spring forces + algorithms).
    let t1 = Instant::now();
    graph.ensure_csr();
    log::info!("built CSR in {:.2}s", t1.elapsed().as_secs_f32());

    // --- Seed positions ----------------------------------------------------
    let n = graph.num_nodes() as usize;
    let mut positions: Vec<Pos> = vec![[0.0; 2]; n];
    let seed_extent = args.k * (n.max(1) as f32).sqrt();
    RandomLayout { extent: seed_extent }.place(&graph, &mut positions, args.seed);

    // --- Launch viewer -----------------------------------------------------
    let mut settings = LayoutSettings {
        k: args.k,
        substeps: args.substeps,
        running: !args.paused,
        ..Default::default()
    };
    if let Some(dt) = args.dt {
        settings.dt = dt;
    }

    let opts = RunOptions {
        title: format!("nebula · {what}"),
        k: args.k,
        settings,
        max_frames: args.frames,
        screenshot: args.screenshot,
        color_mode,
        draw_edges: !args.no_edges,
        draw_nodes: !args.no_nodes,
        select: args.select,
        show_help: args.help_overlay,
        show_labels: args.labels,
        aggregate: args.aggregate,
        node_size: args.node_size,
        filter,
        start_hierarchical: args.hierarchical,
        start_radial: args.radial,
    };

    let app = App::with_labels(graph, positions, opts, node_labels, node_attrs);
    app.run()
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1).peekable();
    let mut generator: Option<Gen> = None;
    let mut k = 30.0f32;
    let mut seed = 42u64;
    let mut dt = None;
    let mut substeps = 1u32;
    let mut frames = None;
    let mut screenshot = None;
    let mut color = ColorMode::Uniform;
    let mut color_attr: Option<String> = None;
    let mut filter: Option<String> = None;
    let mut hierarchical = false;
    let mut radial = false;
    let mut paused = false;
    let mut no_edges = false;
    let mut no_nodes = false;
    let mut select = None;
    let mut help_overlay = false;
    let mut labels = false;
    let mut aggregate = false;
    let mut node_size = 3.0f32;

    fn next_u64(it: &mut impl Iterator<Item = String>, name: &str) -> Result<u64, String> {
        it.next()
            .ok_or_else(|| format!("{name} needs a value"))?
            .parse()
            .map_err(|_| format!("{name}: expected integer"))
    }
    fn next_u32(it: &mut impl Iterator<Item = String>, name: &str) -> Result<u32, String> {
        Ok(next_u64(it, name)? as u32)
    }
    fn next_f32(it: &mut impl Iterator<Item = String>, name: &str) -> Result<f32, String> {
        it.next()
            .ok_or_else(|| format!("{name} needs a value"))?
            .parse()
            .map_err(|_| format!("{name}: expected number"))
    }

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--grid" => {
                generator = Some(Gen::Grid(next_u32(&mut it, "--grid W")?, next_u32(&mut it, "--grid H")?))
            }
            "--random" => {
                generator = Some(Gen::Random(next_u64(&mut it, "--random N")?, next_u64(&mut it, "--random M")?))
            }
            "--ba" => {
                generator = Some(Gen::Ba(next_u64(&mut it, "--ba N")?, next_u32(&mut it, "--ba M")?))
            }
            "--blocks" => {
                generator = Some(Gen::Blocks(
                    next_u64(&mut it, "--blocks N")?,
                    next_u32(&mut it, "--blocks K")?,
                    next_u64(&mut it, "--blocks M")?,
                ))
            }
            "--geo" => {
                generator = Some(Gen::Geo(next_u64(&mut it, "--geo N")?, next_f32(&mut it, "--geo R")?))
            }
            "--file" => {
                generator = Some(Gen::File(it.next().ok_or("--file needs a path")?))
            }
            "--k" => k = next_f32(&mut it, "--k")?,
            "--seed" => seed = next_u64(&mut it, "--seed")?,
            "--dt" => dt = Some(next_f32(&mut it, "--dt")?),
            "--substeps" => substeps = next_u32(&mut it, "--substeps")?,
            "--paused" => paused = true,
            "--no-edges" => no_edges = true,
            "--no-nodes" => no_nodes = true,
            "--select" => select = Some(next_u32(&mut it, "--select")?),
            "--help-overlay" => help_overlay = true,
            "--labels" => labels = true,
            "--aggregate" => aggregate = true,
            "--node-size" => node_size = next_f32(&mut it, "--node-size")?,
            "--frames" => frames = Some(next_u64(&mut it, "--frames")?),
            "--screenshot" => screenshot = Some(it.next().ok_or("--screenshot needs a path")?),
            "--color" => {
                let name = it.next().ok_or("--color needs a mode")?;
                color = match name.as_str() {
                    "uniform" => ColorMode::Uniform,
                    "components" => ColorMode::Components,
                    "degree" => ColorMode::Degree,
                    "pagerank" => ColorMode::PageRank,
                    "coloring" => ColorMode::Coloring,
                    "communities" => ColorMode::Communities,
                    other => return Err(format!("unknown color mode: {other}")),
                };
            }
            "--color-attr" => {
                color_attr = Some(it.next().ok_or("--color-attr needs an attribute name")?);
            }
            "--filter" => {
                filter = Some(it.next().ok_or("--filter needs KEY:OP:VALUE")?);
            }
            "--layout" => {
                let name = it.next().ok_or("--layout needs a name")?;
                match name.as_str() {
                    "hierarchical" | "layered" | "dag" => {
                        hierarchical = true;
                        radial = false;
                    }
                    "radial" | "concentric" => {
                        radial = true;
                        hierarchical = false;
                    }
                    "force" | "force-directed" => {
                        hierarchical = false;
                        radial = false;
                    }
                    other => return Err(format!("unknown layout: {other}")),
                }
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        generator: generator.unwrap_or(Gen::Ba(50_000, 3)),
        k,
        seed,
        dt,
        substeps,
        frames,
        screenshot,
        color,
        color_attr,
        filter,
        hierarchical,
        radial,
        paused,
        no_edges,
        no_nodes,
        select,
        help_overlay,
        labels,
        aggregate,
        node_size,
    })
}
