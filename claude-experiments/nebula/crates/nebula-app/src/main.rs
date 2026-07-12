//! `nebula` — GPU graph viewer CLI.
//!
//! Builds a graph (generated or loaded), seeds positions, and hands it to the
//! GPU viewer. All layout and rendering then happen on the GPU.

use std::time::Instant;

use nebula_core::{generate, io, Graph, Pos};
use nebula_layout::{Layout, RandomLayout};
use nebula_render::{App, ColorMode, LayoutSettings, RunOptions};

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
    --file <PATH>             Load an edge-list file (\"u v\" per line)

OPTIONS:
    --k <FLOAT>               Optimal edge length (default 30)
    --seed <INT>              RNG seed (default 42)
    --dt <FLOAT>              Simulation timestep
    --substeps <INT>          Sim substeps per frame (default 1)
    -h, --help                Show this help

CONTROLS (in the window):
    drag           pan            scroll        zoom
    space          pause/resume   F             fit view
    1..6           color by: uniform / components / degree / pagerank / coloring / communities
    E / N          toggle edges / nodes
    + / -          node size      [ / ]         edge brightness
    Esc            quit
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
            let (g, _) = io::load_edge_list(path)?;
            let label = format!("file {path}");
            (g, label)
        }
    };
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
        color_mode: args.color,
    };

    let app = App::new(graph, positions, opts);
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
    })
}
