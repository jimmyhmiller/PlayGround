#![feature(future_join)]
#![feature(min_specialization)]

use std::{cell::RefCell, path::Path, thread::available_parallelism, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{Registry, layer::SubscriberExt, util::SubscriberInitExt};
use turbo_tasks_malloc::TurboMalloc;
use turbopack_cli::arguments::Arguments;
use turbopack_trace_utils::{
    exit::ExitHandler,
    filter_layer::FilterLayer,
    raw_trace::RawTraceLayer,
    trace_writer::TraceWriter,
    tracing_presets::{
        TRACING_OVERVIEW_TARGETS, TRACING_TURBO_TASKS_TARGETS, TRACING_TURBOPACK_TARGETS,
    },
};

#[global_allocator]
static ALLOC: TurboMalloc = TurboMalloc;

fn main() {
    thread_local! {
        static LAST_SWC_ATOM_GC_TIME: RefCell<Option<Instant>> = const { RefCell::new(None) };
    }

    let mut rt = tokio::runtime::Builder::new_multi_thread();
    rt.enable_all()
        .on_thread_stop(|| {
            TurboMalloc::thread_stop();
        })
        .on_thread_park(|| {
            LAST_SWC_ATOM_GC_TIME.with_borrow_mut(|cell| {
                use std::time::Duration;

                if cell.is_none_or(|t| t.elapsed() > Duration::from_secs(2)) {
                    swc_core::ecma::atoms::hstr::global_atom_store_gc();
                    *cell = Some(Instant::now());
                }
            });
        });

    let args = Arguments::parse();

    let worker_threads = args
        .worker_threads()
        .map(|v| {
            if v == 0 {
                panic!("--worker-threads=0 is invalid, you must use at least one thread.");
            } else {
                v
            }
        })
        .unwrap_or_else(|| available_parallelism().map(|n| n.get()).unwrap_or(1));

    rt.worker_threads(worker_threads);
    rt.max_blocking_threads(usize::MAX - worker_threads);

    #[cfg(not(codspeed))]
    rt.disable_lifo_slot();

    // cf-runtime integration: when CF_RUNTIME_UI=1, register an Observer
    // *before* building the tokio runtime so all spawn/poll/wake events
    // are captured. Then run egui on the main thread and the tokio
    // runtime on a worker thread.
    // CF_RUNTIME_DUMP=1: run turbopack with hooks but no UI, then print
    // a summary of the event stream to stderr. Diagnostic mode.
    if std::env::var("CF_RUNTIME_DUMP").ok().as_deref() == Some("1") {
        let observer = cf_runtime::hooks::Observer::new();
        cf_runtime::hooks::register(observer.clone());
        // Wire turbo-tasks-malloc as the per-span allocation source.
        cf_runtime::hooks::register_alloc_snapshot(|| {
            let c = TurboMalloc::allocation_counters();
            (
                c.allocations as u64,
                c.deallocations as u64,
                c.allocation_count as u64,
                c.deallocation_count as u64,
            )
        });
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        let _ = tracing_subscriber::registry()
            .with(cf_tracing_layer::CfLayer::new())
            .try_init();
        let anchor_instant = std::time::Instant::now();
        let anchor_sys = std::time::SystemTime::now();
        let runtime = rt.build().unwrap();
        let _ = runtime.block_on(main_inner(args));
        let log = observer.log.snapshot();
        if let Ok(path) = std::env::var("CF_RUNTIME_TRACE_OUT") {
            match std::fs::File::create(&path) {
                Ok(mut f) => {
                    let mut bw = std::io::BufWriter::new(&mut f);
                    if let Err(e) = cf_runtime::event::export::write_jsonl(
                        &log,
                        anchor_instant,
                        anchor_sys,
                        &mut bw,
                    ) {
                        eprintln!("[cf-dump] trace write error: {e}");
                    } else {
                        eprintln!("[cf-dump] trace written to {path} ({} events)", log.len());
                    }
                }
                Err(e) => eprintln!("[cf-dump] open {path}: {e}"),
            }
        }
        let mut starts = 0;
        let mut ends = 0;
        let mut waited_starts: std::collections::HashMap<cf_runtime::TaskId, u64> =
            std::collections::HashMap::new();
        let mut paired = 0;
        let mut orphan_starts = 0;
        let mut orphan_ends = 0;
        for e in &log {
            match &e.kind {
                cf_runtime::EventKind::PollStart => {
                    starts += 1;
                    if let Some(t) = e.task {
                        *waited_starts.entry(t).or_insert(0) += 1;
                    }
                }
                cf_runtime::EventKind::PollEnd { .. } => {
                    ends += 1;
                    if let Some(t) = e.task {
                        let c = waited_starts.entry(t).or_insert(0);
                        if *c > 0 {
                            *c -= 1;
                            paired += 1;
                        } else {
                            orphan_ends += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        for (_t, c) in &waited_starts {
            orphan_starts += c;
        }
        let mut span_enter = 0;
        let mut span_exit = 0;
        let mut span_event = 0;
        let mut alloc_events = 0u64;
        let mut total_bytes_delta: i64 = 0;
        let mut total_count_delta: i64 = 0;
        let mut max_bytes_span: (i64, &'static str) = (0, "");
        let mut span_names: std::collections::HashMap<&'static str, u32> =
            std::collections::HashMap::new();
        for e in &log {
            match &e.kind {
                cf_runtime::EventKind::SpanEnter { name, .. } => {
                    span_enter += 1;
                    *span_names.entry(name).or_insert(0) += 1;
                }
                cf_runtime::EventKind::SpanExit { .. } => span_exit += 1,
                cf_runtime::EventKind::SpanEvent { .. } => span_event += 1,
                cf_runtime::EventKind::SpanAllocs { bytes_delta, count_delta, .. } => {
                    alloc_events += 1;
                    total_bytes_delta += bytes_delta;
                    total_count_delta += count_delta;
                    if bytes_delta.abs() > max_bytes_span.0.abs() {
                        max_bytes_span = (*bytes_delta, "<see span_id>");
                    }
                }
                _ => {}
            }
        }
        eprintln!(
            "[cf-dump] {} events: {} PollStart, {} PollEnd; {} paired, {} unmatched starts, {} unmatched ends; {} span enters, {} span exits, {} span events",
            log.len(), starts, ends, paired, orphan_starts, orphan_ends,
            span_enter, span_exit, span_event,
        );
        let resources = observer.resources.snapshot();
        let mut by_kind: std::collections::HashMap<&'static str, u32> =
            std::collections::HashMap::new();
        for r in &resources {
            *by_kind.entry(r.kind.label()).or_insert(0) += 1;
        }
        eprintln!(
            "[cf-dump] alloc events: {} ({:+} bytes net, {:+} count net, max single-span: {:+} bytes)",
            alloc_events, total_bytes_delta, total_count_delta, max_bytes_span.0
        );
        eprintln!("[cf-dump] resources currently registered: {}", resources.len());
        let mut rows: Vec<_> = by_kind.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        for (kind, n) in rows {
            eprintln!("  {n:>4} {kind}");
        }
        let mut top_spans: Vec<_> = span_names.into_iter().collect();
        top_spans.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("[cf-dump] top span names:");
        for (name, count) in top_spans.iter().take(10) {
            eprintln!("  {count:>5} {name}");
        }
        // Print first 30 events
        for e in log.iter().take(30) {
            eprintln!("  seq={} task={:?} worker={:?} {:?}", e.seq, e.task, e.worker, e.kind);
        }
        return;
    }

    if std::env::var("CF_RUNTIME_UI").ok().as_deref() == Some("1") {
        let observer = cf_runtime::hooks::Observer::new();
        if std::env::var("CF_RUNTIME_PAUSE_AT_START").ok().as_deref() != Some("0") {
            observer.controller.pause();
        }
        cf_runtime::hooks::register(observer.clone());
        cf_runtime::hooks::register_alloc_snapshot(|| {
            let c = TurboMalloc::allocation_counters();
            (
                c.allocations as u64,
                c.deallocations as u64,
                c.allocation_count as u64,
                c.deallocation_count as u64,
            )
        });

        // Install our tracing-subscriber layer so every turbopack span
        // (Build, ResolveModule, ParseJs, Transform, EmitAsset, ...)
        // lands in the same EventLog as tokio's poll/wake events. This
        // is the cross-layer bridge that makes the UI semantic instead
        // of just runtime-noise.
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        let _ = tracing_subscriber::registry()
            .with(cf_tracing_layer::CfLayer::new())
            .try_init();
        let log = observer.log.clone();
        let registry = observer.registry.clone();

        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_for_thread = done.clone();

        // Build runtime + run main_inner on a separate thread.
        let runtime = rt.build().unwrap();
        let trace_out = std::env::var("CF_RUNTIME_TRACE_OUT").ok();
        let anchor_instant = std::time::Instant::now();
        let anchor_sys = std::time::SystemTime::now();
        std::thread::Builder::new()
            .name("turbopack-tokio".into())
            .spawn(move || {
                let res = runtime.block_on(main_inner(args));
                eprintln!(
                    "[cf-host] turbopack build finished: {res:?} \
                     events={} tasks={}",
                    log.len(),
                    registry.snapshot().len()
                );
                // Optional: dump trace to disk so it can be reopened
                // later (see the `replay` mode below).
                if let Some(path) = trace_out {
                    let snap = log.snapshot();
                    match std::fs::File::create(&path) {
                        Ok(mut f) => {
                            let mut bw = std::io::BufWriter::new(&mut f);
                            if let Err(e) = cf_runtime::event::export::write_jsonl(
                                &snap,
                                anchor_instant,
                                anchor_sys,
                                &mut bw,
                            ) {
                                eprintln!("[cf-host] trace write error: {e}");
                            } else {
                                eprintln!("[cf-host] trace written to {path} ({} events)", snap.len());
                            }
                        }
                        Err(e) => eprintln!("[cf-host] open {path}: {e}"),
                    }
                }
                done_for_thread.store(true, std::sync::atomic::Ordering::Release);
            })
            .unwrap();

        // Drive the cf-runtime UI on the main thread. Blocks until the
        // window is closed; the spawned tokio thread keeps running but
        // we exit the process on window-close to keep behavior simple.
        let observer_for_ui = observer.clone();
        if let Err(e) = cf_ui::run_observer(observer_for_ui) {
            eprintln!("cf-ui error: {e}");
        }
        if !done.load(std::sync::atomic::Ordering::Acquire) {
            eprintln!("[cf-host] window closed before turbopack finished — exiting");
        }
        return;
    }

    rt.build().unwrap().block_on(main_inner(args)).unwrap();
}

async fn main_inner(args: Arguments) -> Result<()> {
    let exit_handler = ExitHandler::listen();

    let trace = std::env::var("TURBOPACK_TRACING").ok();
    if let Some(mut trace) = trace.filter(|v| !v.is_empty()) {
        // Trace presets
        match trace.as_str() {
            "overview" | "1" => {
                trace = TRACING_OVERVIEW_TARGETS.join(",");
            }
            "turbopack" => {
                trace = TRACING_TURBOPACK_TARGETS.join(",");
            }
            "turbo-tasks" => {
                trace = TRACING_TURBO_TASKS_TARGETS.join(",");
            }
            _ => {}
        }

        let subscriber = Registry::default();

        let subscriber = subscriber.with(FilterLayer::try_new(&trace).unwrap());

        let internal_dir = args
            .dir()
            .unwrap_or_else(|| Path::new("."))
            .join(".turbopack");
        std::fs::create_dir_all(&internal_dir)
            .context("Unable to create .turbopack directory")
            .unwrap();
        let trace_file = internal_dir.join("trace.log");
        let trace_writer = std::fs::File::create(trace_file).unwrap();
        let (trace_writer, guard) = TraceWriter::new(trace_writer);
        let subscriber = subscriber.with(RawTraceLayer::new(trace_writer));

        exit_handler
            .on_exit(async move { tokio::task::spawn_blocking(|| drop(guard)).await.unwrap() });

        subscriber.init();
    }

    match args {
        Arguments::Build(args) => turbopack_cli::build::build(&args).await,
        Arguments::Dev(args) => turbopack_cli::dev::start_server(&args).await,
    }
}
