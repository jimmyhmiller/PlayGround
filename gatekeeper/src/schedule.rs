//! Scheduled jobs: run configured commands on a fixed interval from inside the
//! gate, so the periodic things you want to manage live in one place (the same
//! config, the same service) as the routes.
//!
//! Each job runs on its own thread: optionally once at start, then every
//! `interval` *after the previous run finishes* (so a slow job never overlaps
//! itself). Each run's start, exit status, and duration are logged; stdout/stderr
//! are inherited so they land in the gate's journal.
//!
//! ## Hot reload
//!
//! The job set is rebuilt on SIGHUP like routes. We don't try to diff jobs;
//! instead the scheduler carries a `generation` counter. On reload we bump it and
//! spawn a fresh set of job threads for the new generation; threads from older
//! generations notice the bump and exit at their next wake. This keeps the model
//! dead simple and guarantees the running jobs always match the current config.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::{self, Job};

/// Owns the scheduler state. Held by the gate; `reload` is called at boot and on
/// every config reload with the current job list.
pub struct Scheduler {
    /// Bumped on every (re)load. Job threads compare against it and exit when a
    /// newer generation exists, so stale jobs stop themselves.
    generation: Arc<AtomicU64>,
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler {
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Replace the running job set with `jobs`. Spawns one thread per job under a
    /// fresh generation; previously-running job threads observe the new
    /// generation and exit at their next wake (between runs). Safe to call
    /// repeatedly (boot + each reload).
    pub fn reload(&self, jobs: &[Job]) {
        // Claim the next generation. Any thread whose `my_gen` != current exits.
        let my_gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;

        if jobs.is_empty() {
            println!("gatekeeper: scheduler: no jobs configured");
            return;
        }
        println!(
            "gatekeeper: scheduler: starting {} job(s) (generation {my_gen})",
            jobs.len()
        );
        for job in jobs {
            let job = job.clone();
            let gen = Arc::clone(&self.generation);
            // Interval was validated at config load, so unwrap is safe; fall back
            // defensively to 5 min if that ever changes.
            let interval = config::parse_duration(&job.every)
                .unwrap_or_else(|_| Duration::from_secs(300));
            std::thread::spawn(move || run_job_loop(job, interval, gen, my_gen));
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// The per-job loop. Runs the command, then sleeps `interval` in short slices so
/// a generation change (reload) is noticed promptly. Exits when its generation
/// is superseded.
fn run_job_loop(job: Job, interval: Duration, generation: Arc<AtomicU64>, my_gen: u64) {
    if job.run_at_start {
        if !still_current(&generation, my_gen) {
            return;
        }
        run_once(&job);
    }
    loop {
        if !sleep_interruptible(interval, &generation, my_gen) {
            // Superseded by a newer generation (reload) — stop this thread.
            return;
        }
        run_once(&job);
    }
}

/// Execute the job's command once, inheriting stdio (so output flows to the
/// journal) and merging the job's `env`. Logs start, then exit status + wall
/// time. A failure to spawn or a non-zero exit is logged, not fatal — the
/// scheduler keeps the job on its schedule.
fn run_once(job: &Job) {
    let (exe, args) = job.command.split_first().expect("validated non-empty command");
    println!("gatekeeper: job {:?}: starting `{}`", job.name, job.command.join(" "));
    let start = Instant::now();

    let mut cmd = Command::new(exe);
    cmd.args(args);
    for (k, v) in &job.env {
        cmd.env(k, v);
    }

    match cmd.status() {
        Ok(status) => {
            let secs = start.elapsed().as_secs_f64();
            if status.success() {
                println!(
                    "gatekeeper: job {:?}: ok in {secs:.1}s",
                    job.name
                );
            } else {
                eprintln!(
                    "gatekeeper: job {:?}: FAILED ({}) in {secs:.1}s",
                    job.name, status
                );
            }
        }
        Err(e) => {
            eprintln!("gatekeeper: job {:?}: could not start: {e}", job.name);
        }
    }
}

/// Sleep `dur` in 500ms slices, returning `false` early if this thread's
/// generation has been superseded (so a reload stops jobs without waiting out a
/// long interval). Returns `true` if the full duration elapsed.
fn sleep_interruptible(dur: Duration, generation: &Arc<AtomicU64>, my_gen: u64) -> bool {
    let deadline = Instant::now() + dur;
    while Instant::now() < deadline {
        if !still_current(generation, my_gen) {
            return false;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        std::thread::sleep(remaining.min(Duration::from_millis(500)));
    }
    still_current(generation, my_gen)
}

fn still_current(generation: &Arc<AtomicU64>, my_gen: u64) -> bool {
    generation.load(Ordering::SeqCst) == my_gen
}
