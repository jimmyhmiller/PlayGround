// Generator for Game-of-Life-on-Flow whiteboards. Produces a complete
// `.whiteboard/` directory: manifest.json, main.flow, visual.json, and
// components/life.flow (the LifeCell + LifeClock templates).
//
// Topology is always 8-neighbor toroidal — the cell rule design assumes
// every cell receives exactly 8 reports per tick, so non-toroidal grids
// would deadlock at the edges.

use std::fs;
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Generate a Game-of-Life whiteboard for the Flow simulator")]
struct Args {
    /// Grid width (cells across).
    #[arg(long)]
    width: usize,

    /// Grid height (cells down).
    #[arg(long)]
    height: usize,

    /// Initial pattern: blinker | glider | single | all-on | random:DENSITY
    #[arg(long, default_value = "blinker")]
    pattern: String,

    /// Tick period in milliseconds. Must exceed clock-latency + cell-latency
    /// with comfortable margin or generations will overlap.
    #[arg(long, default_value_t = 200)]
    period_ms: u64,

    /// Latency on Clock -> Cell edges, in milliseconds.
    #[arg(long, default_value_t = 1)]
    clock_latency_ms: u64,

    /// Latency on Cell -> Cell (neighbor report) edges, in milliseconds.
    #[arg(long, default_value_t = 1)]
    cell_latency_ms: u64,

    /// Pixel spacing between cells in the visual layout.
    #[arg(long, default_value_t = 40.0)]
    spacing: f32,

    /// Random seed for `--pattern random:DENSITY`.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Output `.whiteboard/` directory. Created if missing; refuses to
    /// overwrite an existing non-empty directory unless --force.
    #[arg(long)]
    out: PathBuf,

    /// Overwrite the output directory if it already exists.
    #[arg(long, default_value_t = false)]
    force: bool,
}

fn main() {
    let args = Args::parse();
    if args.width == 0 || args.height == 0 {
        eprintln!("error: --width and --height must be > 0");
        std::process::exit(1);
    }
    if args.period_ms <= args.clock_latency_ms + args.cell_latency_ms {
        eprintln!(
            "error: --period-ms ({}) must exceed clock-latency ({}) + cell-latency ({}) \
             or generations will overlap",
            args.period_ms, args.clock_latency_ms, args.cell_latency_ms
        );
        std::process::exit(1);
    }

    let alive = build_pattern(&args.pattern, args.width, args.height, args.seed)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

    if args.out.exists() {
        let entries: Vec<_> = fs::read_dir(&args.out)
            .map(|d| d.collect::<Result<Vec<_>, _>>().unwrap_or_default())
            .unwrap_or_default();
        if !entries.is_empty() && !args.force {
            eprintln!(
                "error: --out `{}` exists and is non-empty; pass --force to overwrite",
                args.out.display()
            );
            std::process::exit(1);
        }
    }
    fs::create_dir_all(&args.out).expect("create out dir");
    fs::create_dir_all(args.out.join("components")).expect("create components dir");

    fs::write(
        args.out.join("manifest.json"),
        manifest_json(&args.pattern, args.width, args.height),
    )
    .expect("write manifest.json");
    fs::write(
        args.out.join("components").join("life.flow"),
        LIFE_TEMPLATE,
    )
    .expect("write components/life.flow");
    fs::write(
        args.out.join("main.flow"),
        main_flow(
            args.width,
            args.height,
            &alive,
            args.period_ms,
            args.clock_latency_ms,
            args.cell_latency_ms,
        ),
    )
    .expect("write main.flow");
    fs::write(
        args.out.join("visual.json"),
        visual_json(args.width, args.height, args.spacing),
    )
    .expect("write visual.json");

    println!(
        "wrote {} ({}x{}, {} alive cells, pattern={})",
        args.out.display(),
        args.width,
        args.height,
        alive.iter().filter(|&&b| b).count(),
        args.pattern,
    );
}

// ────────────────────────────────────────────────────────────────────
// Patterns
// ────────────────────────────────────────────────────────────────────

fn build_pattern(name: &str, w: usize, h: usize, seed: u64) -> Result<Vec<bool>, String> {
    let mut alive = vec![false; w * h];
    let idx = |x: usize, y: usize| -> usize { y * w + x };
    let cx = w / 2;
    let cy = h / 2;

    if let Some(rest) = name.strip_prefix("random:") {
        let density: f64 = rest
            .parse()
            .map_err(|_| format!("bad density in `{}`", name))?;
        if !(0.0..=1.0).contains(&density) {
            return Err(format!("density must be in [0,1], got {}", density));
        }
        let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        for cell in alive.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / ((1u64 << 31) as f64);
            *cell = u < density;
        }
        return Ok(alive);
    }

    match name {
        "single" => {
            alive[idx(cx, cy)] = true;
        }
        "blinker" => {
            // Horizontal 3-in-a-row centered. On a 5x5+ toroidal grid this
            // oscillates between horizontal and vertical period-2.
            if w < 3 {
                return Err("blinker needs width >= 3".into());
            }
            alive[idx((cx + w - 1) % w, cy)] = true;
            alive[idx(cx, cy)] = true;
            alive[idx((cx + 1) % w, cy)] = true;
        }
        "glider" => {
            // Classic 5-cell glider, top-left of the live region at (1,0).
            //   . X .
            //   . . X
            //   X X X
            if w < 5 || h < 5 {
                return Err("glider needs width and height >= 5".into());
            }
            alive[idx(1, 0)] = true;
            alive[idx(2, 1)] = true;
            alive[idx(0, 2)] = true;
            alive[idx(1, 2)] = true;
            alive[idx(2, 2)] = true;
        }
        "all-on" => {
            for cell in alive.iter_mut() {
                *cell = true;
            }
        }
        other => return Err(format!("unknown pattern `{}`", other)),
    }
    Ok(alive)
}

// ────────────────────────────────────────────────────────────────────
// File contents
// ────────────────────────────────────────────────────────────────────

const LIFE_TEMPLATE: &str = r#"# Conway's Game of Life — self-ticking cell template.
#
# Each cell drives its own period via `self -> self : period_ns` plus an
# initial `inject tick(nil)`. No global clock node is needed:
# synchronicity falls out of the engine's ordering rules — every cell's
# initial tick fires at T=0, every cell broadcasts at the same instant,
# every report arrives at T=cell_latency, every B3/S23 update happens
# in the same step. The next ticks all arrive at T=period_ns
# simultaneously, so the cycle stays in lockstep across the grid.
#
# `out_neighbors()` already excludes self-loops, so the report broadcast
# correctly fans only to the 8 toroidal neighbors and not back to the
# self-edge used for the period.
#
# Rule order:
#   on_tick           — period elapsed: reset accumulator + broadcast
#   on_report_partial — count < 8: accumulate
#   on_report_final   — count == 8: apply B3/S23 to `alive`

node LifeCell {
    slots {
        alive: Int = 0
        period_ns: Int = 200000000
        reports_seen: Int = 0
        live_neighbors: Int = 0
    }
    on_spawn {
        self -> self : period_ns
        inject tick(nil)
    }

    rule on_tick {
        on tick(_)
        do {
            reports_seen := 0
            live_neighbors := 0
            emit_each report(alive) to out_neighbors()
            emit tick(nil) to self
        }
    }

    rule on_report_partial {
        on report(v)
        when reports_seen + 1 < 8
        do {
            reports_seen := reports_seen + 1
            live_neighbors := live_neighbors + v
        }
    }

    rule on_report_final {
        on report(v)
        when reports_seen + 1 == 8
        do {
            reports_seen := reports_seen + 1
            live_neighbors := live_neighbors + v
            alive := if (alive == 1 && (live_neighbors == 2 || live_neighbors == 3)) || (alive == 0 && live_neighbors == 3) then 1 else 0
        }
    }
}
"#;

fn manifest_json(pattern: &str, w: usize, h: usize) -> String {
    format!(
        r#"{{
  "format_version": 1,
  "canvas": {{
    "name": "Game of Life {w}x{h} ({pattern})",
    "description": "Conway's Game of Life implemented as a grid of LifeCell gadgets connected by 8-neighbor toroidal edges, driven by a global LifeClock that fans out a pulse every period. Each cell broadcasts its `alive` to neighbors and accumulates 8 reports per generation."
  }}
}}
"#
    )
}

fn cell_name(x: usize, y: usize) -> String {
    format!("Cell_{}_{}", x, y)
}

fn main_flow(
    w: usize,
    h: usize,
    alive: &[bool],
    period_ms: u64,
    _clock_latency_ms: u64,
    cell_latency_ms: u64,
) -> String {
    let mut s = String::new();
    s.push_str("# Auto-generated by flow-life-gen. Do not hand-edit.\n");
    s.push_str(&format!(
        "# Grid: {}x{} toroidal, period={}ms, cell_latency={}ms.\n",
        w, h, period_ms, cell_latency_ms
    ));
    s.push_str(
        "# Cells are self-ticking — each one's `on_spawn` block sets up\n\
         # a self-edge with `period_ns` latency and injects the initial\n\
         # `tick(nil)`. No global clock node is needed; the engine's\n\
         # ordering rules keep all cells in lockstep.\n\n",
    );

    let period_ns = period_ms * 1_000_000;
    for y in 0..h {
        for x in 0..w {
            let init = if alive[y * w + x] { 1 } else { 0 };
            s.push_str(&format!(
                "node {} : LifeCell {{ alive: {}, period_ns: {} }}\n",
                cell_name(x, y),
                init,
                period_ns,
            ));
        }
    }
    s.push_str("\nedges {\n");

    // 8-neighbor toroidal wiring. Each cell emits to all 8 neighbors;
    // the corresponding inbound edges come from the other cells' outbound
    // declarations, so we don't double-declare.
    let neighbors: [(isize, isize); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];
    for y in 0..h {
        for x in 0..w {
            for (dx, dy) in neighbors {
                let nx = ((x as isize + dx).rem_euclid(w as isize)) as usize;
                let ny = ((y as isize + dy).rem_euclid(h as isize)) as usize;
                s.push_str(&format!(
                    "    {} -> {} : {}ms\n",
                    cell_name(x, y),
                    cell_name(nx, ny),
                    cell_latency_ms
                ));
            }
        }
    }
    s.push_str("}\n");
    s
}

fn visual_json(w: usize, h: usize, spacing: f32) -> String {
    let cell_size = (spacing * 0.6).max(8.0);
    let mut s = String::from("{\n  \"format_version\": 1,\n");

    // Class block: declares the LifeCell shape (small square) and binary-slot
    // paint behaviour (black when alive == 1, paper when 0). flow-bevy reads
    // this generically; nothing about Game of Life lives in the engine.
    s.push_str("  \"classes\": {\n");
    s.push_str(&format!(
        "    \"LifeCell\": {{ \"shape\": {{ \"kind\": \"square\", \"size\": {:.1} }}, \"paint\": {{ \"kind\": \"binary_slot\", \"slot\": \"alive\", \"on\": \"#111111\", \"off\": \"#f5f5f0\" }} }}\n",
        cell_size
    ));
    s.push_str("  },\n");

    s.push_str("  \"nodes\": {\n");
    let cx = (w as f32 - 1.0) * spacing * 0.5;
    let cy = (h as f32 - 1.0) * spacing * 0.5;

    let total = w * h;
    for y in 0..h {
        for x in 0..w {
            let px = (x as f32) * spacing - cx;
            let py = -((y as f32) * spacing - cy); // flip so y=0 is top
            let i = y * w + x;
            let last = i + 1 == total;
            s.push_str(&format!(
                "    \"{}\": {{ \"pos\": [{:.1}, {:.1}] }}{}\n",
                cell_name(x, y),
                px,
                py,
                if last { "" } else { "," }
            ));
        }
    }
    s.push_str("  },\n  \"viewport\": { \"pos\": [0.0, 0.0], \"zoom\": 1.0 },\n");
    // Life canvases ship with edges + packets hidden by default — at any
    // grid worth visualizing they'd otherwise drown out the cell state.
    // Users press `H` to toggle.
    s.push_str("  \"hide_all\": true\n}\n");
    s
}
