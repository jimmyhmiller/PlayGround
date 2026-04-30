//! Sim/render decoupling: snapshot + driver.
//!
//! The renderer never touches the live `Sim` — it reads an immutable
//! `RenderSnapshot` published once per tick. Mutations go through
//! `SimDriver` as commands.
//!
//! Two driver modes:
//!
//! - `Direct`: Sim lives inline on the main thread. `with_sim_mut` is
//!   synchronous. Used by integration tests that need deterministic,
//!   single-stepping control.
//! - `Worker`: Sim lives on a background thread. `with_sim_mut` ships
//!   a closure through a channel and blocks on a oneshot reply (used
//!   only for user-interactive sites that need a return value, e.g.
//!   palette drops). Hot per-frame mutations use `send_command`
//!   (fire-and-forget). Pause/multiplier are atomics so the worker
//!   reacts within one yield.
//!
//! Snapshot publication is the same in both modes: after each command
//! batch + tick, `make_snapshot(&sim, &mut prev_log_index)` builds a
//! fresh `Arc<RenderSnapshot>` and stores it in `ArcSwap`. Readers
//! load the latest with one atomic.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use bevy::prelude::*;
use flow::{CapturePolicy, EdgeId, Event, NodeId, Sim, SnapshotRing, Time, Value};
use flow::sim::Scheduled;
use flow::timeline::Timeline;

use crate::rewind_strategy::{
    initial_strategy as initial_rewind_strategy, RewindContext, RewindStrategy,
    RewindStrategyDispatch, RewindStrategyKind,
};

// ─────────────────────────────────────────────────────────────────
// Snapshot data
// ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct NodeView {
    pub name: String,
    pub slots: BTreeMap<String, Value>,
    pub class_name: Option<String>,
    /// Pre-computed probe outputs for label rendering. Cheap because
    /// most nodes have zero probes; only declared probes show up here.
    pub probe_readings: Vec<(String, String)>,
    /// All declared probe label names (even those with no current
    /// reading). Used by the probe-placement tool to know what's
    /// available to attach to.
    pub probe_labels: Vec<String>,
}

#[derive(Clone)]
pub struct EdgeView {
    pub from: NodeId,
    pub to: NodeId,
    pub from_port: Option<String>,
    pub to_port: Option<String>,
}

/// Per-tick view of the sim that the renderer reads.
///
/// Frozen between ticks. The renderer never sees mid-tick mutation —
/// `make_snapshot` clones at a quiescent point.
///
/// Events are *not* on the snapshot — the worker can publish many
/// snapshots between renders, and the latest would clobber the
/// earlier ones' event lists. They flow through a separate channel
/// (see `SimEventRx`) so nothing is lost regardless of snapshot
/// publish/consume cadence.
pub struct RenderSnapshot {
    pub now_ns: Time,
    pub nodes: BTreeMap<NodeId, NodeView>,
    pub edges: BTreeMap<EdgeId, EdgeView>,
    pub error_counts: BTreeMap<String, u64>,
    pub perf_samples: Vec<(&'static str, f64)>,
    /// Cloned timeline for the timeline strip UI. Cheap — typical
    /// boards have <20 timeline events.
    pub timeline: Timeline,
    /// Class names registered in the sim. Used by `has_class` checks.
    pub class_names: BTreeSet<String>,
    /// Increments every time the visual layer needs to recompute
    /// from the sim's event log: rewinds and topology resets
    /// (LoadExample / canvas load) both bump it. The visual layer
    /// watches this and on every bump (a) resets its strategy state
    /// and (b) replays `replay_events` with the correct
    /// `sim_now → visual_now` time mapping, reconstructing exactly
    /// the on-screen state the user originally saw at that sim time.
    pub rewind_epoch: u64,
    /// Sim-ns of every snapshot the user can scrub to, anchor first.
    /// Sourced from `SnapshotRing::marker_times_ns`. The HUD uses
    /// these to draw scrub-strip markers.
    pub rewind_markers_ns: Vec<u64>,
    /// Strategy-computed `PacketEmitted` events the visual layer
    /// should replay on the next rewind-epoch bump. Held in an `Arc`
    /// so per-frame snapshot publishes are cheap pointer bumps
    /// rather than `Vec` clones — important because the same Arc
    /// rides on every snapshot until the next rewind replaces it.
    /// Empty until the first rewind / topology reset; updated by the
    /// driver each time the active `RewindStrategy` produces a new
    /// set.
    pub replay_events: Arc<Vec<Event>>,
}

impl RenderSnapshot {
    pub fn empty() -> Self {
        Self {
            now_ns: 0,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            error_counts: BTreeMap::new(),
            perf_samples: Vec::new(),
            timeline: Timeline::default(),
            class_names: BTreeSet::new(),
            rewind_epoch: 0,
            rewind_markers_ns: Vec::new(),
            replay_events: Arc::new(Vec::new()),
        }
    }

    pub fn has_class(&self, name: &str) -> bool {
        self.class_names.contains(name)
    }
}

/// Build a snapshot from the live sim. Drains `perf_samples` out of
/// the sim. Clones: nodes, edges, error_counts, timeline.
///
/// `rewind_epoch` is forwarded straight onto the snapshot; the visual
/// layer watches it to detect rewinds. `rewind_markers_ns` is the
/// snapshot ring's marker list, included so the HUD can render scrub
/// strip ticks without a separate read of the ring.
/// Build a render snapshot, attaching rewind metadata. The
/// strategy-computed `replay_events` is only populated *on the
/// rewind itself* (passed in via `replay_events`); regular
/// per-frame snapshots leave it empty so we don't pay the cost of
/// derivation on every publish.
pub fn make_snapshot_with(
    sim: &mut Sim,
    rewind_epoch: u64,
    rewind_markers_ns: Vec<u64>,
    replay_events: Arc<Vec<Event>>,
) -> RenderSnapshot {
    let mut snap = make_snapshot(sim);
    snap.rewind_epoch = rewind_epoch;
    snap.rewind_markers_ns = rewind_markers_ns;
    snap.replay_events = replay_events;
    snap
}

/// Plain snapshot — no rewind metadata. Kept for sites that don't own
/// a `SnapshotRing` (e.g. tests).
pub fn make_snapshot(sim: &mut Sim) -> RenderSnapshot {
    let mut nodes = BTreeMap::new();
    for (nid, node) in sim.nodes.iter() {
        let class_name = sim.class_name(*nid).map(|s| s.to_owned());
        let probe_labels = sim.probe_labels(*nid);
        let probe_readings = sim.probe_readings(*nid);
        nodes.insert(*nid, NodeView {
            name: node.name.clone(),
            slots: node.slots.clone(),
            class_name,
            probe_readings,
            probe_labels,
        });
    }

    let mut edges = BTreeMap::new();
    for (eid, edge) in sim.edges.iter() {
        edges.insert(*eid, EdgeView {
            from: edge.from,
            to: edge.to,
            from_port: edge.from_port.clone(),
            to_port: edge.to_port.clone(),
        });
    }

    let class_names: BTreeSet<String> = sim
        .templates
        .iter()
        .map(|t| t.name.clone())
        .collect();

    let perf_samples: Vec<(&'static str, f64)> = sim.drain_perf_samples().collect();

    RenderSnapshot {
        now_ns: sim.now_ns,
        nodes,
        edges,
        error_counts: sim.error_counts.clone(),
        perf_samples,
        timeline: sim.timeline.clone(),
        class_names,
        rewind_epoch: 0,
        rewind_markers_ns: Vec::new(),
        replay_events: Arc::new(Vec::new()),
    }
}

/// Drain new events from the sim's ring log, advancing `prev_log_index`.
/// Mirrors the old `bridge::collect_new_events` ring-skip behavior.
pub fn drain_new_events(sim: &Sim, prev_log_index: &mut u64) -> Vec<Event> {
    let total = sim.log.total_recorded;
    if total <= *prev_log_index {
        *prev_log_index = total;
        return Vec::new();
    }
    let ring_len = sim.log.events.len() as u64;
    let first_resident = total.saturating_sub(ring_len);
    let skip_to = (*prev_log_index).max(first_resident);
    let start = (skip_to - first_resident) as usize;
    let mut out = Vec::with_capacity((total - skip_to) as usize);
    for ev in sim.log.events.iter().skip(start) {
        out.push(ev.clone());
    }
    *prev_log_index = total;
    out
}

// ─────────────────────────────────────────────────────────────────
// Commands
// ─────────────────────────────────────────────────────────────────

/// A mutation to apply to the sim, eventually. Closure-based: callers
/// move the operation into a `Box<dyn FnOnce>`, and the worker (or
/// Direct driver) runs it on the owned `Sim`.
///
/// We picked closures over typed enum variants because:
///  * the existing API surface (`sim.user_edit_slot`, `sim.inject`,
///    `sim.add_edge`, `gadgets::spawn(&mut sim, …)`, etc.) is large
///    and irregular — typed variants would be a big enum that just
///    re-encodes those calls;
///  * we don't need replay or network sync (out of scope per design);
///  * closures keep the call site readable: `send(|sim| sim.foo())`.
pub struct SimCommand(pub Box<dyn FnOnce(&mut Sim) + Send + 'static>);

impl SimCommand {
    pub fn new<F: FnOnce(&mut Sim) + Send + 'static>(f: F) -> Self {
        Self(Box::new(f))
    }
}

// ─────────────────────────────────────────────────────────────────
// Driver
// ─────────────────────────────────────────────────────────────────

/// Shared atomics so the worker reacts to pause/multiplier changes
/// without waiting for the command queue to drain. Multiplier is
/// stored as `f64::to_bits` in an `AtomicU64`.
#[derive(Clone)]
pub struct SimControl {
    pub multiplier_bits: Arc<AtomicU64>,
    pub paused: Arc<AtomicBool>,
    /// Tick budget for one-shot stepping. Worker checks before each
    /// idle yield: if non-zero, advances by exactly that many ns and
    /// then resets it to zero (regardless of pause state).
    pub step_once_ns: Arc<AtomicU64>,
}

impl SimControl {
    pub fn new(multiplier: f64) -> Self {
        Self {
            multiplier_bits: Arc::new(AtomicU64::new(multiplier.to_bits())),
            paused: Arc::new(AtomicBool::new(false)),
            step_once_ns: Arc::new(AtomicU64::new(0)),
        }
    }
    pub fn multiplier(&self) -> f64 {
        f64::from_bits(self.multiplier_bits.load(Ordering::Relaxed))
    }
    pub fn set_multiplier(&self, m: f64) {
        self.multiplier_bits.store(m.to_bits(), Ordering::Relaxed);
    }
    pub fn paused(&self) -> bool { self.paused.load(Ordering::Relaxed) }
    pub fn set_paused(&self, p: bool) { self.paused.store(p, Ordering::Relaxed); }
    pub fn request_step(&self, ns: u64) {
        self.step_once_ns.store(ns, Ordering::Relaxed);
    }
}

pub enum SimDriver {
    Direct {
        sim: Sim,
        ring: SnapshotRing,
        rewind_epoch: u64,
        snapshot: Arc<ArcSwap<RenderSnapshot>>,
        prev_log_index: u64,
        events_tx: mpsc::Sender<Event>,
        control: SimControl,
        rewind_strategy: RewindStrategyDispatch,
        /// Strategy-computed replay events for the next snapshot
        /// publish. Set by `rewind` / `reset_history`; drained by
        /// `republish`. We stash it here rather than inline-publishing
        /// because the bridge's per-frame `republish_after_mut` would
        /// otherwise overwrite the post-rewind snapshot with an
        /// empty replay list before the host could consume it.
        pending_replay_events: Arc<Vec<Event>>,
    },
    Worker {
        tx: mpsc::Sender<WorkerMsg>,
        snapshot: Arc<ArcSwap<RenderSnapshot>>,
        control: SimControl,
        shutdown: Arc<AtomicBool>,
        thread: Option<JoinHandle<()>>,
    },
}

pub enum WorkerMsg {
    Cmd(SimCommand),
    /// Run the closure, send result through the oneshot reply, then
    /// republish a snapshot. Used by `with_sim_mut` from interactive
    /// callers that need an immediate return value (palette drops,
    /// canvas load, test harness in worker mode).
    CmdReply(Box<dyn FnOnce(&mut Sim) + Send + 'static>),
    /// Rewind the sim to `target_ns` using the newest captured
    /// snapshot at or before that time (falling back to the anchor at
    /// t=0). After restore the worker runs forward to `target_ns`,
    /// bumps the rewind epoch, and republishes.
    Rewind { target_ns: Time },
    /// Discard all captured snapshots (anchor + ring) and re-anchor
    /// from the live sim. Sites that wholesale-replace the sim
    /// (`LoadExample`, canvas load) must send this so future rewinds
    /// don't restore the pre-replace state.
    ResetHistory,
    /// Swap the worker's active rewind strategy. Takes effect on
    /// the next rewind.
    SetRewindStrategy(RewindStrategyKind),
}

/// Default ring size (number of cadence-captured snapshots beyond the
/// sticky anchor). At the default 250ms cadence this gives ~16s of
/// scrubback before captures start rolling off.
pub const DEFAULT_SNAPSHOT_RING_CAP: usize = 64;

impl SimDriver {
    /// In-process synchronous driver. Used by tests.
    /// Returns the driver alongside the receiver end of the event
    /// channel — wrap that in `SimEventRx` and put both into the
    /// world.
    pub fn direct(sim: Sim, multiplier: f64) -> (Self, mpsc::Receiver<Event>) {
        let snapshot = Arc::new(ArcSwap::from_pointee(RenderSnapshot::empty()));
        let prev_log_index = sim.log.total_recorded;
        let (events_tx, events_rx) = mpsc::channel::<Event>();
        let mut ring = SnapshotRing::new(DEFAULT_SNAPSHOT_RING_CAP);
        // Seed the anchor immediately so rewind to 0 is always possible.
        ring.capture(&sim);
        let mut me = Self::Direct {
            sim,
            ring,
            rewind_epoch: 0,
            snapshot,
            prev_log_index,
            events_tx,
            control: SimControl::new(multiplier),
            rewind_strategy: initial_rewind_strategy(),
            pending_replay_events: Arc::new(Vec::new()),
        };
        me.republish();
        (me, events_rx)
    }

    /// Spawn a worker thread that owns the `Sim`. Used by the app.
    /// Returns the driver and the event receiver (same pattern as
    /// `direct`).
    pub fn worker(sim: Sim, multiplier: f64) -> (Self, mpsc::Receiver<Event>) {
        let snapshot = Arc::new(ArcSwap::from_pointee(RenderSnapshot::empty()));
        let control = SimControl::new(multiplier);
        let shutdown = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<WorkerMsg>();
        let (events_tx, events_rx) = mpsc::channel::<Event>();
        let thread = {
            let snapshot = snapshot.clone();
            let control = control.clone();
            let shutdown = shutdown.clone();
            thread::Builder::new()
                .name("flow-sim-worker".into())
                .spawn(move || worker_loop(sim, snapshot, rx, events_tx, control, shutdown))
                .expect("spawn sim worker")
        };
        let me = Self::Worker {
            tx, snapshot, control, shutdown, thread: Some(thread),
        };
        (me, events_rx)
    }

    pub fn snapshot(&self) -> Arc<RenderSnapshot> {
        match self {
            Self::Direct { snapshot, .. } | Self::Worker { snapshot, .. } => {
                snapshot.load_full()
            }
        }
    }

    pub fn control(&self) -> &SimControl {
        match self {
            Self::Direct { control, .. } | Self::Worker { control, .. } => control,
        }
    }

    /// Fire-and-forget mutation. Don't use this if you need the
    /// closure's return value — use `with_sim_mut` instead.
    pub fn send_command(&mut self, cmd: SimCommand) {
        match self {
            Self::Direct { sim, .. } => {
                (cmd.0)(sim);
                self.republish();
            }
            Self::Worker { tx, .. } => {
                tx.send(WorkerMsg::Cmd(cmd)).expect("worker channel closed");
            }
        }
    }

    /// Synchronous read-modify access. In Direct mode, runs inline.
    /// In Worker mode, ships through the channel and blocks on a
    /// reply — incurs at most one tick of latency. Use sparingly,
    /// only for sites that genuinely need a return value (canvas
    /// load, palette drop returning the new NodeId).
    pub fn with_sim_mut<R, F>(&mut self, f: F) -> R
    where
        R: Send + 'static,
        F: FnOnce(&mut Sim) -> R + Send + 'static,
    {
        match self {
            Self::Direct { sim, .. } => {
                let r = f(sim);
                self.republish();
                r
            }
            Self::Worker { tx, .. } => {
                let (reply_tx, reply_rx) = mpsc::sync_channel::<R>(1);
                let job: Box<dyn FnOnce(&mut Sim) + Send + 'static> =
                    Box::new(move |sim| {
                        let r = f(sim);
                        // Receiver may have hung up if caller was
                        // dropped mid-await; ignore the SendError
                        // so the worker keeps running.
                        let _ = reply_tx.send(r);
                    });
                tx.send(WorkerMsg::CmdReply(job)).expect("worker channel closed");
                reply_rx.recv().expect("worker dropped reply channel")
            }
        }
    }

    /// In Direct mode only: drain new events to channel + republish
    /// snapshot, attaching any pending strategy-computed replay
    /// events. In Worker mode this is a no-op because the worker
    /// republishes on its own cadence.
    fn republish(&mut self) {
        if let Self::Direct {
            sim,
            snapshot,
            prev_log_index,
            events_tx,
            ring,
            rewind_epoch,
            pending_replay_events,
            ..
        } = self {
            let new_events = drain_new_events(sim, prev_log_index);
            for ev in new_events {
                let _ = events_tx.send(ev);
            }
            // Stamp the current pending replay set onto the
            // snapshot. Cheap (Arc clone) — does NOT drain, so
            // subsequent per-frame republishes re-attach the same
            // Arc until a new rewind/reset replaces it. The visual
            // layer only consumes on `rewind_epoch` changes, so
            // restamping a stale Arc is harmless.
            let replay = pending_replay_events.clone();
            let snap = make_snapshot_with(
                sim,
                *rewind_epoch,
                ring.marker_times_ns(),
                replay,
            );
            snapshot.store(Arc::new(snap));
        }
    }

    /// Direct mode only: advance the sim by `dt_sim_ns`. Used by the
    /// bridge's per-frame advance system when running synchronously.
    /// In Worker mode the worker advances itself.
    pub fn advance_direct(&mut self, dt_sim_ns: u64) {
        if let Self::Direct { sim, ring, .. } = self {
            if dt_sim_ns == 0 { return; }
            let target = sim.now_ns.saturating_add(dt_sim_ns);
            sim.run_until(target);
            ring.auto_capture(sim, CapturePolicy::DEFAULT);
            self.republish();
        }
    }

    /// Swap the active rewind strategy. Takes effect on the next
    /// rewind. Mode-aware: in Direct mode replaces the field
    /// inline, in Worker mode ships a message.
    pub fn set_rewind_strategy(&mut self, kind: RewindStrategyKind) {
        match self {
            Self::Direct { rewind_strategy, .. } => {
                *rewind_strategy = kind.build();
            }
            Self::Worker { tx, .. } => {
                tx.send(WorkerMsg::SetRewindStrategy(kind))
                    .expect("worker channel closed");
            }
        }
    }

    /// Throw away all captured snapshots and re-anchor from the live
    /// sim. Call this after wholesale-replacing the sim (LoadExample,
    /// canvas load) so future rewinds don't restore the pre-replace
    /// state. Bumps the rewind epoch so the visual layer also resets.
    pub fn reset_history(&mut self) {
        match self {
            Self::Direct { sim, ring, rewind_epoch, prev_log_index, .. } => {
                ring.entries.clear();
                ring.anchor = None;
                ring.capture(sim);
                *prev_log_index = sim.log.total_recorded;
                *rewind_epoch += 1;
                self.republish();
            }
            Self::Worker { tx, .. } => {
                tx.send(WorkerMsg::ResetHistory)
                    .expect("worker channel closed");
            }
        }
    }

    /// Rewind to `target_ns`. Works in both Direct and Worker mode.
    /// Returns the actual sim time after the rewind (which equals
    /// `target_ns` if a snapshot at or before `target_ns` was found,
    /// otherwise leaves the sim unchanged and returns its current
    /// `now_ns`).
    ///
    /// Bumps `rewind_epoch` so the visual layer recomputes its
    /// state from the rewound sim's event log on the next frame —
    /// that's how on-screen visuals snap to exactly what was on
    /// screen at the rewound moment.
    pub fn rewind(&mut self, target_ns: Time) -> Time {
        match self {
            Self::Direct {
                sim,
                ring,
                prev_log_index,
                rewind_epoch,
                rewind_strategy,
                pending_replay_events,
                ..
            } => {
                let Some(snap) = ring.latest_before_ns(target_ns).cloned() else {
                    return sim.now_ns;
                };
                sim.restore_from(snap.sim);

                let start_idx = sim.log.total_recorded;
                let in_flight_at_snap: Vec<Scheduled> =
                    sim.in_flight.iter().map(|r| r.0.clone()).collect();
                let edges_at_snap = sim.edges.clone();

                if sim.now_ns < target_ns {
                    sim.run_until(target_ns);
                }
                let end_idx = sim.log.total_recorded;

                let events = rewind_strategy.compute_replay(&RewindContext {
                    sim,
                    in_flight_at_snap: &in_flight_at_snap,
                    edges_at_snap: &edges_at_snap,
                    start_idx,
                    end_idx,
                    target_ns,
                });
                *pending_replay_events = Arc::new(events);

                *prev_log_index = sim.log.total_recorded;
                *rewind_epoch += 1;
                let now = sim.now_ns;
                self.republish();
                now
            }
            Self::Worker { tx, .. } => {
                tx.send(WorkerMsg::Rewind { target_ns })
                    .expect("worker channel closed");
                target_ns
            }
        }
    }

    pub fn is_worker(&self) -> bool { matches!(self, Self::Worker { .. }) }

    /// Direct mode only: borrow the live sim. Panics in Worker mode —
    /// the sim lives on another thread there. Use `with_sim_mut` for
    /// mutations that need to work in both modes.
    pub fn sim(&self) -> &Sim {
        match self {
            Self::Direct { sim, .. } => sim,
            Self::Worker { .. } => panic!(
                "SimDriver::sim() called in Worker mode — sim lives on the worker thread. \
                 Use `with_sim_mut` instead."
            ),
        }
    }

    pub fn sim_mut(&mut self) -> &mut Sim {
        match self {
            Self::Direct { sim, .. } => sim,
            Self::Worker { .. } => panic!(
                "SimDriver::sim_mut() called in Worker mode — sim lives on the worker thread. \
                 Use `with_sim_mut` instead."
            ),
        }
    }

    /// After mutating via `sim_mut()`, call this to refresh the
    /// snapshot so subsequent reads see the change. Direct mode only.
    pub fn republish_after_mut(&mut self) {
        self.republish();
    }
}

impl Drop for SimDriver {
    fn drop(&mut self) {
        if let Self::Worker { shutdown, thread, .. } = self {
            shutdown.store(true, Ordering::Relaxed);
            if let Some(t) = thread.take() {
                // Best-effort join; if the worker is wedged we'd
                // rather not hang shutdown forever.
                let _ = t.join();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Worker loop
// ─────────────────────────────────────────────────────────────────

fn worker_loop(
    mut sim: Sim,
    snapshot_out: Arc<ArcSwap<RenderSnapshot>>,
    rx: mpsc::Receiver<WorkerMsg>,
    events_tx: mpsc::Sender<Event>,
    control: SimControl,
    shutdown: Arc<AtomicBool>,
) {
    let mut prev_log_index = sim.log.total_recorded;
    let mut last_tick = Instant::now();
    let mut ring = SnapshotRing::new(DEFAULT_SNAPSHOT_RING_CAP);
    // Seed anchor at t=0 before any work runs.
    ring.capture(&sim);
    let mut rewind_epoch: u64 = 0;
    let mut rewind_strategy = initial_rewind_strategy();
    // Strategy-computed replay set rides on every snapshot publish
    // until the next rewind replaces it. The visual layer only
    // consumes on `rewind_epoch` changes.
    let mut pending_replay: Arc<Vec<Event>> = Arc::new(Vec::new());

    let publish = |sim: &mut Sim,
                   prev_log_index: &mut u64,
                   ring: &SnapshotRing,
                   rewind_epoch: u64,
                   replay_events: Arc<Vec<Event>>| {
        let new_events = drain_new_events(sim, prev_log_index);
        for ev in new_events {
            // Receiver hung up if main thread is shutting down — fine.
            let _ = events_tx.send(ev);
        }
        let snap = make_snapshot_with(
            sim,
            rewind_epoch,
            ring.marker_times_ns(),
            replay_events,
        );
        snapshot_out.store(Arc::new(snap));
    };

    let do_rewind = |sim: &mut Sim,
                     prev_log_index: &mut u64,
                     ring: &SnapshotRing,
                     rewind_epoch: &mut u64,
                     strategy: &RewindStrategyDispatch,
                     target_ns: Time|
     -> Option<Arc<Vec<Event>>> {
        let snap = ring.latest_before_ns(target_ns).cloned()?;
        sim.restore_from(snap.sim);

        let start_idx = sim.log.total_recorded;
        let in_flight_at_snap: Vec<Scheduled> =
            sim.in_flight.iter().map(|r| r.0.clone()).collect();
        let edges_at_snap = sim.edges.clone();

        if sim.now_ns < target_ns {
            sim.run_until(target_ns);
        }
        let end_idx = sim.log.total_recorded;

        let replay = strategy.compute_replay(&RewindContext {
            sim,
            in_flight_at_snap: &in_flight_at_snap,
            edges_at_snap: &edges_at_snap,
            start_idx,
            end_idx,
            target_ns,
        });
        // Forward play after this point doesn't re-feed the visual
        // ingest with the post-snap delta — the strategy already
        // included it.
        *prev_log_index = sim.log.total_recorded;
        *rewind_epoch += 1;
        Some(Arc::new(replay))
    };

    publish(&mut sim, &mut prev_log_index, &ring, rewind_epoch, pending_replay.clone());

    loop {
        if shutdown.load(Ordering::Relaxed) { break; }

        // Drain commands first so Pause / LoadCanvas / EditSlot take
        // effect before we advance time.
        let mut applied_cmd = false;
        let mut rewound = false;
        loop {
            match rx.try_recv() {
                Ok(WorkerMsg::Cmd(cmd)) => {
                    (cmd.0)(&mut sim);
                    applied_cmd = true;
                }
                Ok(WorkerMsg::CmdReply(job)) => {
                    job(&mut sim);
                    applied_cmd = true;
                }
                Ok(WorkerMsg::Rewind { target_ns }) => {
                    if let Some(events) = do_rewind(
                        &mut sim,
                        &mut prev_log_index,
                        &ring,
                        &mut rewind_epoch,
                        &rewind_strategy,
                        target_ns,
                    ) {
                        pending_replay = events;
                        rewound = true;
                        last_tick = Instant::now();
                    }
                }
                Ok(WorkerMsg::ResetHistory) => {
                    ring.entries.clear();
                    ring.anchor = None;
                    ring.capture(&sim);
                    prev_log_index = sim.log.total_recorded;
                    rewind_epoch += 1;
                    applied_cmd = true;
                }
                Ok(WorkerMsg::SetRewindStrategy(kind)) => {
                    rewind_strategy = kind.build();
                    applied_cmd = true;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }

        // One-shot step request takes precedence over pause.
        let step = control.step_once_ns.swap(0, Ordering::Relaxed);
        let advanced = if step > 0 {
            let target = sim.now_ns.saturating_add(step);
            sim.run_until(target);
            last_tick = Instant::now();
            true
        } else if control.paused() {
            false
        } else {
            let now = Instant::now();
            let elapsed = now - last_tick;
            let dt_sim_ns = (elapsed.as_secs_f64() * control.multiplier() * 1e9) as u64;
            if dt_sim_ns > 0 {
                let target = sim.now_ns.saturating_add(dt_sim_ns);
                sim.run_until(target);
                last_tick = now;
                true
            } else {
                false
            }
        };

        if advanced {
            ring.auto_capture(&sim, CapturePolicy::DEFAULT);
        }

        if advanced || applied_cmd || rewound {
            publish(&mut sim, &mut prev_log_index, &ring, rewind_epoch, pending_replay.clone());
        } else {
            // Idle. Block on the channel with a small timeout so we
            // wake promptly on a new command but also tick the wall
            // clock often enough that the next dt_sim_ns has data.
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok(msg) => {
                    match msg {
                        WorkerMsg::Cmd(c) => (c.0)(&mut sim),
                        WorkerMsg::CmdReply(j) => j(&mut sim),
                        WorkerMsg::Rewind { target_ns } => {
                            if let Some(events) = do_rewind(
                                &mut sim,
                                &mut prev_log_index,
                                &ring,
                                &mut rewind_epoch,
                                &rewind_strategy,
                                target_ns,
                            ) {
                                pending_replay = events;
                            }
                            last_tick = Instant::now();
                        }
                        WorkerMsg::ResetHistory => {
                            ring.entries.clear();
                            ring.anchor = None;
                            ring.capture(&sim);
                            prev_log_index = sim.log.total_recorded;
                            rewind_epoch += 1;
                        }
                        WorkerMsg::SetRewindStrategy(kind) => {
                            rewind_strategy = kind.build();
                        }
                    }
                    publish(&mut sim, &mut prev_log_index, &ring, rewind_epoch, pending_replay.clone());
                }
                Err(mpsc::RecvTimeoutError::Timeout) => { /* loop and retry tick */ }
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Bevy resources
// ─────────────────────────────────────────────────────────────────

/// Owns the driver. Mutable access (`ResMut`) is the write/spawn path
/// for everyone who needs to send commands or call `with_sim_mut`.
///
/// In Direct mode, `Deref`/`DerefMut` give direct access to the
/// underlying `Sim` — convenient for tests that need raw access. In
/// Worker mode the deref panics (sim isn't on this thread); use
/// `with_sim_mut` instead.
#[derive(Resource)]
pub struct SimDriverRes(pub SimDriver);

impl std::ops::Deref for SimDriverRes {
    type Target = Sim;
    fn deref(&self) -> &Sim { self.0.sim() }
}
impl std::ops::DerefMut for SimDriverRes {
    fn deref_mut(&mut self) -> &mut Sim {
        // After mutation through the deref, the snapshot will be
        // stale until something republishes. Tests currently advance
        // via `advance_direct`, which republishes; pure mutations
        // without an advance won't show up in the snapshot until
        // someone calls `republish_after_mut`. Fine for tests.
        self.0.sim_mut()
    }
}

/// Latest published snapshot, refreshed once per frame at the start
/// of `Update` (see `sync_snapshot_system`). Read by every visual /
/// inspector / probe system that previously read `flow.sim.*`.
#[derive(Resource, Clone)]
pub struct SimSnapshotRes(pub Arc<RenderSnapshot>);

impl Default for SimSnapshotRes {
    fn default() -> Self { Self(Arc::new(RenderSnapshot::empty())) }
}

/// Receiver end of the sim → bridge event channel. Bridge drains
/// this each frame into `NewEvents`. Lives outside the snapshot so
/// burst-mode worker publishing can't lose events.
///
/// Wrapped in `Mutex` because `mpsc::Receiver` is `!Sync`, but Bevy
/// requires `Resource: Sync`. Only one system reads it (bridge), so
/// the lock is uncontended.
#[derive(Resource)]
pub struct SimEventRx(pub Mutex<mpsc::Receiver<Event>>);

/// SystemSet for the per-frame snapshot publish step. Every system
/// that reads `SimSnapshotRes` should run `.after(SnapshotReady)` so
/// it sees this frame's view, not last frame's.
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct SnapshotReady;

/// Per-frame system: pull the latest snapshot off the driver into the
/// resource so every downstream system in the same frame sees a
/// consistent view.
pub fn sync_snapshot_system(
    mut driver: ResMut<SimDriverRes>,
    mut snap: ResMut<SimSnapshotRes>,
) {
    // In Direct mode, republish first — tests (and other in-process
    // code) often mutate `Sim` through the `DerefMut` impl without
    // calling `republish_after_mut`. Republishing here makes those
    // mutations visible without ceremony. Cheap: just clones BTreeMaps.
    // No-op in Worker mode (the worker self-publishes).
    if !driver.0.is_worker() {
        driver.0.republish_after_mut();
    }
    snap.0 = driver.0.snapshot();
}
