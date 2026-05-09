//! Host binary: spawns a runtime, populates it with a workload, and runs the
//! egui inspector on the main thread. Two subcommands:
//!  - `demo` — synthetic mix of sleeping / cpu / fan-out tasks (default)
//!  - `mini-redis` — the real test app (wired in once cf-tokio shim is ready)

use cf_runtime::{time::sleep, Runtime};
use clap::{Parser, Subcommand};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "cf-host", about = "controllable-futures host")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Number of worker threads.
    #[arg(long, default_value_t = 4)]
    workers: usize,
    /// Skip the UI; useful for headless smoke testing.
    #[arg(long)]
    no_ui: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Synthetic workload — good for first-light testing of the UI.
    Demo,
    /// Run the vendored mini-redis on top of cf-runtime via cf-tokio shim.
    /// Not yet wired up.
    MiniRedis {
        #[arg(long, default_value_t = 6379u16)]
        port: u16,
    },
}

fn main() {
    let cli = Cli::parse();
    let rt = Runtime::new(cli.workers);

    match cli.cmd.unwrap_or(Cmd::Demo) {
        Cmd::Demo => spawn_demo(&rt),
        Cmd::MiniRedis { port } => spawn_mini_redis(&rt, port),
    }

    if cli.no_ui {
        // Run forever — the host is the long-running process. Process stays
        // alive until killed (Ctrl-C). Periodically print event counts so
        // the operator can confirm the runtime is doing work.
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let log = rt.handle().log();
            let reg = rt.handle().registry();
            eprintln!(
                "[cf-host] events={} tasks={}",
                log.len(),
                reg.snapshot().len()
            );
        }
    }

    let handle = rt.handle();
    if let Err(e) = cf_ui::run(handle) {
        eprintln!("ui error: {e}");
    }
}

/// Spawn the vendored mini-redis server as a top-level task on cf-runtime.
/// The server listens on the loopback interface; clients connect with any
/// stock Redis client (e.g. `redis-cli -p 6379 SET foo bar`).
fn spawn_mini_redis(rt: &Runtime, port: u16) {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,mini_redis=debug")),
        )
        .init();
    let _h = rt.spawn("mini-redis-bind", async move {
        let listener = match cf_tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("bind error: {e}");
                return;
            }
        };
        eprintln!("mini-redis listening on 127.0.0.1:{port}");
        // ctrl_c never resolves in our shim — server runs until host exits.
        mini_redis_cf::server::run(listener, cf_tokio::signal::ctrl_c()).await;
    });
}

/// Synthetic workload: a generator that fan-outs short tasks at a steady rate,
/// plus a few long-running ones. Designed to keep the task list lively and
/// give the operator something to play with (pause individual tasks, switch
/// to manual mode, etc.).
fn spawn_demo(rt: &Runtime) {
    // A long-running producer that periodically spawns short workers.
    rt.spawn("producer", {
        let h = rt.handle();
        async move {
            let mut n = 0u64;
            loop {
                sleep(Duration::from_millis(200)).await;
                n += 1;
                let id = n;
                h.spawn(format!("worker-{id}"), async move {
                    // Each worker sleeps a variable amount, then "computes".
                    let dur = Duration::from_millis(50 + (id % 7) * 30);
                    sleep(dur).await;
                    let mut acc = 0u64;
                    for i in 0..50_000 {
                        acc = acc.wrapping_add(i ^ id);
                    }
                    acc
                });
                if n >= 200 {
                    break;
                }
            }
        }
    });

    // A "cpu" task that yields between bursts so we see lots of polls.
    rt.spawn("cpu-bursts", async {
        for _ in 0..40 {
            // Burn a small amount of CPU.
            let mut x = 0u64;
            for i in 0..200_000 {
                x = x.wrapping_add(i);
            }
            std::hint::black_box(x);
            // Yield to let the scheduler pick someone else.
            sleep(Duration::from_millis(5)).await;
        }
    });

    // A "leaky" task that holds wakes pending without finishing — handy for
    // demoing the suspended/wake-pending UI state.
    rt.spawn("slow-tail", async {
        for _ in 0..30 {
            sleep(Duration::from_millis(500)).await;
        }
    });
}
