//! Headless layout-cost bench: times one full simulation step, with no window
//! and no rendering, so layout cost is isolated from raster/blend cost.
//!
//! Usage: layout_bench <nodes> <edges> [steps] [--no-farfield] [--dump PATH]
//!
//! `--no-farfield` empties the COM pyramid level table (num_levels = 0), which
//! makes the `forces` pass skip the far-field interaction-list walk entirely.
//! Diffing the two runs attributes the step time to the far field.
//!
//! `--dump PATH` reads positions back after the timed steps and writes them as
//! raw f32 pairs, so two builds can be compared numerically. Pixel-diffing
//! screenshots cannot separate a systematic force error from chaos amplifying a
//! rounding difference; comparing positions after a *single* step can.

use nebula_core::{generate, Pos};
use nebula_layout::{GridLayout, Layout, RandomLayout};
use nebula_render::layout_gpu::{LayoutGpu, LayoutSettings};
use nebula_render::scene::GpuGraph;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nodes: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(500_000);
    let edges: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(5_000_000);
    let steps: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let no_far = args.iter().any(|a| a == "--no-farfield");
    let dump = args.iter().position(|a| a == "--dump").map(|i| args[i + 1].clone());
    // Seeding on a grid makes node *index* correlate with node *position*, so a
    // workgroup's threads land in neighbouring cells and hit the same pyramid
    // entries. Random seeding destroys that correlation. Diffing the two bounds
    // what a cell-sorted node order could win, without building the sort.
    let grid_seed = args.iter().any(|a| a == "--grid-seed");
    let verify_sort = args.iter().any(|a| a == "--verify-sort");
    // Run N untimed steps before timing. A random seed spreads nodes evenly
    // (~1 node/cell), but a real layout collapses into clusters whose cores hold
    // far more — and near-field cost grows with occupancy. Timing from a random
    // seed measures the easy regime the app is only in for its first few seconds.
    let settle: u32 = args
        .iter()
        .position(|a| a == "--settle")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(0);
    // Force the spatial grid dimension. The derived value clamps at 2048, which
    // past ~2M nodes makes cells wider than k and inflates near-field work.
    let grid_dim: Option<u32> = args
        .iter()
        .position(|a| a == "--grid-dim")
        .map(|i| args[i + 1].parse().unwrap());

    let t = Instant::now();
    let mut graph = generate::stochastic_blocks(nodes, 8, edges, 0.01, 42);
    graph.ensure_csr();
    eprintln!(
        "graph {} nodes {} edges built in {:.1}s",
        graph.num_nodes(),
        graph.num_edges(),
        t.elapsed().as_secs_f32()
    );

    let k = 30.0f32;
    let mut positions: Vec<Pos> = vec![[0.0, 0.0]; graph.num_nodes() as usize];
    // Both seeds cover the same extent (k*sqrt(n)), so cell occupancy is
    // comparable and the only variable is index<->position correlation.
    if grid_seed {
        GridLayout { spacing: k }.place(&graph, &mut positions, 42);
    } else {
        RandomLayout { extent: k * (graph.num_nodes().max(1) as f32).sqrt() }
            .place(&graph, &mut positions, 42);
    }

    pollster::block_on(async move {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("no adapter");
        let mut limits = adapter.limits();
        limits.max_buffer_size = adapter.limits().max_buffer_size;
        limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bench"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .expect("no device");

        let mut gpu_graph =
            GpuGraph::upload_with_grid_dim(&device, &queue, &graph, &positions, k, grid_dim);
        drop(graph);
        eprintln!(
            "grid {}x{} (cell {:.1} vs k {:.0}), {} pyramid levels",
            gpu_graph.grid_dim,
            gpu_graph.grid_dim,
            gpu_graph.world_size / gpu_graph.grid_dim as f32,
            k,
            gpu_graph.pyr_levels.len(),
        );
        if no_far {
            gpu_graph.pyr_levels.clear();
        }
        let mut settings = LayoutSettings::default();
        // The sort runs on the positions at the top of a step, but `integrate`
        // moves them before we can read them back — a node legitimately drifts a
        // cell and looks misplaced. dt = 0 freezes integration so the positions we
        // read are the ones the sort actually binned. Forces still run.
        if verify_sort {
            settings.dt = 0.0;
        }
        let layout = LayoutGpu::new(&device, &gpu_graph, &settings);

        // Warm up (shader compile, first-touch of lazily-resident buffers).
        // Skipped when dumping: warmup steps are real steps, and an accuracy
        // comparison wants exactly `steps` of divergence, not steps + 3.
        let warmup = if dump.is_some() { 0 } else { 3 };
        for _ in 0..warmup {
            let mut enc = device.create_command_encoder(&Default::default());
            layout.step(&mut enc);
            queue.submit([enc.finish()]);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        if settle > 0 {
            // Cool alpha across the settling run the way the app does, so the
            // layout actually contracts into clusters instead of jittering hot.
            let t = Instant::now();
            let mut s = settings;
            s.set_cooling_for(nodes);
            for _ in 0..settle {
                s.alpha *= 1.0 - s.alpha_decay;
                layout.update_settings(&queue, &s);
                let mut enc = device.create_command_encoder(&Default::default());
                layout.step(&mut enc);
                queue.submit([enc.finish()]);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            eprintln!(
                "settled {settle} steps in {:.1}s (alpha now {:.4})",
                t.elapsed().as_secs_f32(),
                s.alpha
            );
            layout.update_settings(&queue, &settings);
        }

        let t = Instant::now();
        for _ in 0..steps {
            let mut enc = device.create_command_encoder(&Default::default());
            layout.step(&mut enc);
            queue.submit([enc.finish()]);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let per_step = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        println!(
            "{} nodes {} edges farfield={} gridseed={} dim={} settle={} -> {:.2} ms/step ({:.1} steps/s)",
            nodes,
            edges,
            !no_far,
            grid_seed,
            gpu_graph.grid_dim,
            settle,
            per_step,
            1000.0 / per_step
        );

        // Prove the counting sort is a correct permutation. A broken prefix sum
        // still produces plausible timings and a plausible-looking layout, so
        // this checks the structure directly rather than trusting the output.
        if verify_sort {
            let read = |buf: &wgpu::Buffer, bytes: u64| -> Vec<u8> {
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("verify_readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut enc = device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
                queue.submit([enc.finish()]);
                staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let v = staging.slice(..).get_mapped_range().to_vec();
                staging.unmap();
                v
            };
            let n = nodes as usize;
            let dim = gpu_graph.grid_dim as u64;
            let cells = (dim * dim) as usize;
            let as_u32 = |b: Vec<u8>| bytemuck::cast_slice::<u8, u32>(&b).to_vec();
            let order = as_u32(read(&gpu_graph.node_order, nodes * 4));
            let starts = as_u32(read(&gpu_graph.cell_starts, cells as u64 * 4));
            let counts = as_u32(read(&gpu_graph.grid_counts, cells as u64 * 4));
            let pos_bytes = read(&gpu_graph.positions, nodes * 8);
            let pos = bytemuck::cast_slice::<u8, [f32; 2]>(&pos_bytes);

            // 1. node_order is a permutation of 0..N.
            let mut seen = vec![0u8; n];
            for &id in &order {
                assert!((id as usize) < n, "node_order holds out-of-range id {id}");
                seen[id as usize] += 1;
            }
            let dupes = seen.iter().filter(|&&c| c > 1).count();
            let missing = seen.iter().filter(|&&c| c == 0).count();
            assert_eq!((dupes, missing), (0, 0), "node_order is not a permutation");

            // 2. Counts sum to N and the runs tile [0, N) contiguously in order.
            let total: u64 = counts.iter().map(|&c| c as u64).sum();
            assert_eq!(total, nodes, "counts sum to {total}, expected {nodes}");
            let mut expect = 0u32;
            for c in 0..cells {
                assert_eq!(starts[c], expect, "cell {c} starts at {} not {expect}", starts[c]);
                expect += counts[c];
            }

            // 3. Every node sits in the run of the cell its position maps to.
            //
            // Re-deriving the cell on the CPU cannot be exact: Metal's f32 divide
            // is not required to be correctly rounded, so a node sitting on a cell
            // boundary can floor to either side. Those disagreements are always
            // *adjacent* and vanishingly rare, and they are harmless — the GPU is
            // self-consistent (count_grid and scatter_nodes call the same
            // cell_coord, which checks 1 and 2 confirm), and the near field reads a
            // 3x3 neighbourhood anyway. A genuinely broken sort would scatter nodes
            // into unrelated runs, so distant mismatches are the real signal.
            let world = gpu_graph.world_size;
            let cs = world / dim as f32;
            let half = world * 0.5;
            let mut adjacent = 0u64;
            let mut distant = 0u64;
            for c in 0..cells {
                let (s, e) = (starts[c] as usize, (starts[c] + counts[c]) as usize);
                for &id in &order[s..e] {
                    let p = pos[id as usize];
                    let cx = (((p[0] + half) / cs).floor() as i64).clamp(0, dim as i64 - 1);
                    let cy = (((p[1] + half) / cs).floor() as i64).clamp(0, dim as i64 - 1);
                    if (cy * dim as i64 + cx) as usize == c {
                        continue;
                    }
                    let (dx, dy) = ((c as i64 % dim as i64) - cx, (c as i64 / dim as i64) - cy);
                    if dx.abs() <= 1 && dy.abs() <= 1 {
                        adjacent += 1;
                    } else {
                        distant += 1;
                        assert!(distant < 5, "node {id} is in run {c}, far from cell {cx},{cy}");
                    }
                }
            }
            // `distant == 0` is the real check — a broken sort puts nodes in
            // unrelated runs. The adjacent count is a secondary sanity bound: it
            // scales with the number of cell boundaries (finer grids have more
            // nodes sitting on one), so allow up to 0.1%.
            assert_eq!(distant, 0, "{distant} nodes scattered into unrelated runs");
            assert!(
                adjacent * 1_000 < nodes.max(1_000),
                "{adjacent}/{nodes} boundary disagreements — too many to be f32 rounding"
            );
            let occupied = counts.iter().filter(|&&c| c > 0).count();
            let maxc = counts.iter().max().unwrap();
            eprintln!(
                "sort VERIFIED: {n} ids permuted exactly once, runs tile [0,{n}) in cell order,\n  \
                 every node in its own cell ({adjacent} on a cell boundary, 0 distant)\n  \
                 {occupied}/{cells} cells occupied, max {maxc} nodes/cell (cap {})",
                gpu_graph.grid_cap
            );
        }

        if let Some(path) = dump {
            let bytes = nodes * 8;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pos_readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&gpu_graph.positions, 0, &staging, 0, bytes);
            queue.submit([enc.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let data = staging.slice(..).get_mapped_range().to_vec();
            std::fs::write(&path, &data).unwrap();
            eprintln!("dumped {} positions -> {}", nodes, path);
        }
    });
}
