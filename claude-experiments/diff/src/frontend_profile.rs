use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ENABLED: OnceLock<bool> = OnceLock::new();
static READ_NS: AtomicU64 = AtomicU64::new(0);
static TRANSFORM_NS: AtomicU64 = AtomicU64::new(0);
static LOWER_NS: AtomicU64 = AtomicU64::new(0);
static RESOLVE_NS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
pub(crate) enum Phase {
    Read,
    Transform,
    Lower,
    Resolve,
}

#[derive(Debug, Default)]
pub(crate) struct FrontendProfile {
    pub read_ms: f64,
    pub transform_ms: f64,
    pub lower_ms: f64,
    pub resolve_ms: f64,
}

pub(crate) fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

pub(crate) fn finish(phase: Phase, started: Option<Instant>) {
    let Some(started) = started else {
        return;
    };
    let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    counter(phase).fetch_add(nanos, Ordering::Relaxed);
}

pub(crate) fn reset() {
    for counter in [&READ_NS, &TRANSFORM_NS, &LOWER_NS, &RESOLVE_NS] {
        counter.store(0, Ordering::Relaxed);
    }
}

pub(crate) fn snapshot() -> FrontendProfile {
    FrontendProfile {
        read_ms: milliseconds(&READ_NS),
        transform_ms: milliseconds(&TRANSFORM_NS),
        lower_ms: milliseconds(&LOWER_NS),
        resolve_ms: milliseconds(&RESOLVE_NS),
    }
}

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("DIFFPACK_PROFILE_FRONTEND").is_some())
}

fn counter(phase: Phase) -> &'static AtomicU64 {
    match phase {
        Phase::Read => &READ_NS,
        Phase::Transform => &TRANSFORM_NS,
        Phase::Lower => &LOWER_NS,
        Phase::Resolve => &RESOLVE_NS,
    }
}

fn milliseconds(counter: &AtomicU64) -> f64 {
    counter.load(Ordering::Relaxed) as f64 / 1_000_000.0
}
