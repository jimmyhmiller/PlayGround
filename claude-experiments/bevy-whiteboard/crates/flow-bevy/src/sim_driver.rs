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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use bevy::prelude::*;
use flow::sim::Scheduled;
use flow::timeline::Timeline;
use flow::{CapturePolicy, Edge, EdgeId, Event, NodeId, Sim, SnapshotRing, Time, Value};

use crate::rewind::{do_rewind, sim_max_edge_latency_ns};

// ─────────────────────────────────────────────────────────────────
// Snapshot data
// ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct NodeView {
    pub name: String,
    pub slots: BTreeMap<String, Value>,
    pub class_name: Option<String>,
    pub probe_readings: Vec<(String, String)>,
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
/// Events flow through a separate channel (`SimEventRx`) so nothing
/// is lost regardless of snapshot publish/consume cadence.
pub struct RenderSnapshot {
    pub now_ns: Time,
    pub nodes: BTreeMap<NodeId, NodeView>,
    pub edges: BTreeMap<EdgeId, EdgeView>,
    pub error_counts: BTreeMap<String, u64>,
    pub perf_samples: Vec<(&'static str, f64)>,
    pub timeline: Timeline,
    pub class_names: BTreeSet<String>,
    /// Bumps every time the visual layer needs to drop derived state
    /// — rewinds and topology resets (LoadExample / canvas load) both
    /// trigger it. The visual layer compares against its own
    /// `RewindEpochSeen` and calls `VisualStrategy::invalidate` on a
    /// change.
    pub rewind_epoch: u64,
    /// Sim-ns of every snapshot the user can scrub to, anchor first.
    /// HUD uses these to draw scrub-strip markers.
    pub rewind_markers_ns: Vec<u64>,
    /// Full edge map for the visual layer's `SimMirror` projection
    /// and any other strategy that resolves `Scheduled.edge → (from,
    /// to)`. Wrapped in `Arc` so per-frame snapshot publishes are
    /// cheap.
    pub edges_full: Arc<BTreeMap<EdgeId, Edge>>,
    /// Currently in-flight scheduled deliveries. `SimMirror` reads
    /// this directly; event-driven strategies ignore it.
    pub in_flight: Arc<Vec<Scheduled>>,
    /// Largest non-self-loop edge latency in the live sim. Visual
    /// strategies combine this with `k` to size their rewind
    /// lookback window (see `VisualStrategy::rewind_lookback_ns`).
    /// 0 means "no non-self-loop edges yet" — strategies fall back
    /// to a static ceiling.
    pub max_edge_lat_ns: u64,
    /// Trajectory `PacketEmitted` events the visual layer should
    /// feed through `frame_into` immediately after invalidating, on
    /// the rewind-epoch bump. Empty otherwise. Held in an `Arc` so
    /// per-frame snapshot publishes are cheap pointer bumps.
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
            edges_full: Arc::new(BTreeMap::new()),
            in_flight: Arc::new(Vec::new()),
            max_edge_lat_ns: 0,
            replay_events: Arc::new(Vec::new()),
        }
    }

    pub fn has_class(&self, name: &str) -> bool {
        self.class_names.contains(name)
    }
}

/// Build a render snapshot, attaching rewind metadata. The
/// `replay_events` Arc is only populated *on the rewind itself*; per-
/// frame publishes attach an empty Arc so the visual layer doesn't
/// keep re-feeding stale events.
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
        nodes.insert(
            *nid,
            NodeView {
                name: node.name.clone(),
                slots: node.slots.clone(),
                class_name,
                probe_readings,
                probe_labels,
            },
        );
    }

    let mut edges = BTreeMap::new();
    for (eid, edge) in sim.edges.iter() {
        edges.insert(
            *eid,
            EdgeView {
                from: edge.from,
                to: edge.to,
                from_port: edge.from_port.clone(),
                to_port: edge.to_port.clone(),
            },
        );
    }

    let class_names: BTreeSet<String> = sim.templates.iter().map(|t| t.name.clone()).collect();

    let perf_samples: Vec<(&'static str, f64)> = sim.drain_perf_samples().collect();

    let edges_full = Arc::new(sim.edges.clone());
    let in_flight: Vec<Scheduled> = sim.in_flight.iter().map(|r| r.0.clone()).collect();
    let max_edge_lat_ns = sim_max_edge_latency_ns(sim);

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
        edges_full,
        in_flight: Arc::new(in_flight),
        max_edge_lat_ns,
        replay_events: Arc::new(Vec::new()),
    }
}

/// Drain new events from the sim's ring log, advancing `prev_log_index`.
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
    pub step_once_ns: Arc<AtomicU64>,
    /// Sim-ns of lookback the active visual strategy needs on rewind.
    /// The host computes this as `strategy.rewind_lookback_ns(k,
    /// max_edge_lat_ns)` each frame and stores it here so the worker
    /// can read it on rewind without having to know which strategy
    /// is active or what `k` is.
    pub rewind_lookback_ns_bits: Arc<AtomicU64>,
}

impl SimControl {
    pub fn new(multiplier: f64) -> Self {
        Self {
            multiplier_bits: Arc::new(AtomicU64::new(multiplier.to_bits())),
            paused: Arc::new(AtomicBool::new(false)),
            step_once_ns: Arc::new(AtomicU64::new(0)),
            rewind_lookback_ns_bits: Arc::new(AtomicU64::new(0)),
        }
    }
    pub fn multiplier(&self) -> f64 {
        f64::from_bits(self.multiplier_bits.load(Ordering::Relaxed))
    }
    pub fn set_multiplier(&self, m: f64) {
        self.multiplier_bits.store(m.to_bits(), Ordering::Relaxed);
    }
    pub fn paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }
    pub fn set_paused(&self, p: bool) {
        self.paused.store(p, Ordering::Relaxed);
    }
    pub fn request_step(&self, ns: u64) {
        self.step_once_ns.store(ns, Ordering::Relaxed);
    }
    pub fn rewind_lookback_ns(&self) -> u64 {
        self.rewind_lookback_ns_bits.load(Ordering::Relaxed)
    }
    pub fn set_rewind_lookback_ns(&self, ns: u64) {
        self.rewind_lookback_ns_bits.store(ns, Ordering::Relaxed);
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
        /// Replay events from the most recent rewind. Re-attached on
        /// every `republish` so the bridge's per-frame
        /// `republish_after_mut` doesn't overwrite the post-rewind
        /// snapshot with an empty replay list before the host can
        /// consume it. The visual layer only consumes on
        /// `rewind_epoch` changes, so restamping a stale Arc is
        /// harmless.
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
    CmdReply(Box<dyn FnOnce(&mut Sim) + Send + 'static>),
    Rewind { target_ns: Time },
    ResetHistory,
}

pub const DEFAULT_SNAPSHOT_RING_CAP: usize = 64;

impl SimDriver {
    pub fn direct(sim: Sim, multiplier: f64) -> (Self, mpsc::Receiver<Event>) {
        let snapshot = Arc::new(ArcSwap::from_pointee(RenderSnapshot::empty()));
        let prev_log_index = sim.log.total_recorded;
        let (events_tx, events_rx) = mpsc::channel::<Event>();
        let mut ring = SnapshotRing::new(DEFAULT_SNAPSHOT_RING_CAP);
        ring.capture(&sim);
        let mut me = Self::Direct {
            sim,
            ring,
            rewind_epoch: 0,
            snapshot,
            prev_log_index,
            events_tx,
            control: SimControl::new(multiplier),
            pending_replay_events: Arc::new(Vec::new()),
        };
        me.republish();
        (me, events_rx)
    }

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
            tx,
            snapshot,
            control,
            shutdown,
            thread: Some(thread),
        };
        (me, events_rx)
    }

    pub fn snapshot(&self) -> Arc<RenderSnapshot> {
        match self {
            Self::Direct { snapshot, .. } | Self::Worker { snapshot, .. } => snapshot.load_full(),
        }
    }

    pub fn control(&self) -> &SimControl {
        match self {
            Self::Direct { control, .. } | Self::Worker { control, .. } => control,
        }
    }

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
                let job: Box<dyn FnOnce(&mut Sim) + Send + 'static> = Box::new(move |sim| {
                    let r = f(sim);
                    let _ = reply_tx.send(r);
                });
                tx.send(WorkerMsg::CmdReply(job))
                    .expect("worker channel closed");
                reply_rx.recv().expect("worker dropped reply channel")
            }
        }
    }

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
        } = self
        {
            let new_events = drain_new_events(sim, prev_log_index);
            for ev in new_events {
                let _ = events_tx.send(ev);
            }
            let replay = pending_replay_events.clone();
            let snap = make_snapshot_with(sim, *rewind_epoch, ring.marker_times_ns(), replay);
            snapshot.store(Arc::new(snap));
        }
    }

    pub fn advance_direct(&mut self, dt_sim_ns: u64) {
        if let Self::Direct { sim, ring, .. } = self {
            if dt_sim_ns == 0 {
                return;
            }
            let target = sim.now_ns.saturating_add(dt_sim_ns);
            sim.run_until(target);
            ring.auto_capture(sim, CapturePolicy::DEFAULT);
            self.republish();
        }
    }

    pub fn reset_history(&mut self) {
        match self {
            Self::Direct {
                sim,
                ring,
                rewind_epoch,
                prev_log_index,
                ..
            } => {
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

    /// Rewind to `target_ns`. Bumps `rewind_epoch` so the visual
    /// layer recomputes its state. Returns the actual sim time after
    /// the rewind (which equals `target_ns` if a snapshot covers
    /// `target_ns - lookback`, otherwise leaves the sim unchanged
    /// and returns its current `now_ns`).
    pub fn rewind(&mut self, target_ns: Time) -> Time {
        match self {
            Self::Direct {
                sim,
                ring,
                prev_log_index,
                rewind_epoch,
                pending_replay_events,
                control,
                ..
            } => {
                let lookback = control.rewind_lookback_ns();
                let Some(events) =
                    do_rewind(sim, ring, prev_log_index, rewind_epoch, target_ns, lookback)
                else {
                    return sim.now_ns;
                };
                *pending_replay_events = events;
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

    pub fn is_worker(&self) -> bool {
        matches!(self, Self::Worker { .. })
    }

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

    pub fn republish_after_mut(&mut self) {
        self.republish();
    }
}

impl Drop for SimDriver {
    fn drop(&mut self) {
        if let Self::Worker {
            shutdown, thread, ..
        } = self
        {
            shutdown.store(true, Ordering::Relaxed);
            if let Some(t) = thread.take() {
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
    ring.capture(&sim);
    let mut rewind_epoch: u64 = 0;
    let mut pending_replay: Arc<Vec<Event>> = Arc::new(Vec::new());

    let publish = |sim: &mut Sim,
                   prev_log_index: &mut u64,
                   ring: &SnapshotRing,
                   rewind_epoch: u64,
                   replay_events: Arc<Vec<Event>>| {
        let new_events = drain_new_events(sim, prev_log_index);
        for ev in new_events {
            let _ = events_tx.send(ev);
        }
        let snap = make_snapshot_with(sim, rewind_epoch, ring.marker_times_ns(), replay_events);
        snapshot_out.store(Arc::new(snap));
    };

    publish(
        &mut sim,
        &mut prev_log_index,
        &ring,
        rewind_epoch,
        pending_replay.clone(),
    );

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let mut applied_cmd = false;
        let mut rewound = false;
        let mut latest_rewind: Option<Time> = None;
        loop {
            match rx.try_recv() {
                Ok(WorkerMsg::Cmd(cmd)) => {
                    (cmd.0)(&mut sim);
                    applied_cmd = true;
                }
                Ok(WorkerMsg::CmdReply(job)) => {
                    job(&mut sim);
                    applied_cmd = true;
                    last_tick = Instant::now();
                }
                Ok(WorkerMsg::Rewind { target_ns }) => {
                    latest_rewind = Some(target_ns);
                }
                Ok(WorkerMsg::ResetHistory) => {
                    ring.entries.clear();
                    ring.anchor = None;
                    ring.capture(&sim);
                    prev_log_index = sim.log.total_recorded;
                    rewind_epoch += 1;
                    applied_cmd = true;
                    latest_rewind = None;
                    last_tick = Instant::now();
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }
        if let Some(target_ns) = latest_rewind {
            let lookback = control.rewind_lookback_ns();
            if let Some(events) = do_rewind(
                &mut sim,
                &ring,
                &mut prev_log_index,
                &mut rewind_epoch,
                target_ns,
                lookback,
            ) {
                pending_replay = events;
                rewound = true;
                last_tick = Instant::now();
            }
        }

        let step = control.step_once_ns.swap(0, Ordering::Relaxed);
        let advanced = if step > 0 {
            let target = sim.now_ns.saturating_add(step);
            sim.run_until(target);
            last_tick = Instant::now();
            true
        } else if rewound {
            false
        } else if control.paused() {
            // Keep `last_tick` current while paused. Otherwise, on
            // resume, `elapsed = now - last_tick` would span the
            // entire pause window and the worker would jump the sim
            // forward by however long the user was paused.
            last_tick = Instant::now();
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
            publish(
                &mut sim,
                &mut prev_log_index,
                &ring,
                rewind_epoch,
                pending_replay.clone(),
            );
        } else {
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok(msg) => {
                    let mut latest_rewind: Option<Time> = None;
                    let process = |msg: WorkerMsg,
                                   sim: &mut Sim,
                                   ring: &mut SnapshotRing,
                                   prev_log_index: &mut u64,
                                   rewind_epoch: &mut u64,
                                   latest_rewind: &mut Option<Time>| {
                        match msg {
                            WorkerMsg::Cmd(c) => (c.0)(sim),
                            WorkerMsg::CmdReply(j) => j(sim),
                            WorkerMsg::Rewind { target_ns } => {
                                *latest_rewind = Some(target_ns);
                            }
                            WorkerMsg::ResetHistory => {
                                ring.entries.clear();
                                ring.anchor = None;
                                ring.capture(sim);
                                *prev_log_index = sim.log.total_recorded;
                                *rewind_epoch += 1;
                                *latest_rewind = None;
                            }
                        }
                    };
                    process(
                        msg,
                        &mut sim,
                        &mut ring,
                        &mut prev_log_index,
                        &mut rewind_epoch,
                        &mut latest_rewind,
                    );
                    loop {
                        match rx.try_recv() {
                            Ok(m) => process(
                                m,
                                &mut sim,
                                &mut ring,
                                &mut prev_log_index,
                                &mut rewind_epoch,
                                &mut latest_rewind,
                            ),
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => return,
                        }
                    }
                    if let Some(target_ns) = latest_rewind {
                        let lookback = control.rewind_lookback_ns();
                        if let Some(events) = do_rewind(
                            &mut sim,
                            &ring,
                            &mut prev_log_index,
                            &mut rewind_epoch,
                            target_ns,
                            lookback,
                        ) {
                            pending_replay = events;
                        }
                    }
                    last_tick = Instant::now();
                    publish(
                        &mut sim,
                        &mut prev_log_index,
                        &ring,
                        rewind_epoch,
                        pending_replay.clone(),
                    );
                }
                Err(mpsc::RecvTimeoutError::Timeout) => { /* retry tick */ }
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Bevy resources
// ─────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct SimDriverRes(pub SimDriver);

impl std::ops::Deref for SimDriverRes {
    type Target = Sim;
    fn deref(&self) -> &Sim {
        self.0.sim()
    }
}
impl std::ops::DerefMut for SimDriverRes {
    fn deref_mut(&mut self) -> &mut Sim {
        self.0.sim_mut()
    }
}

#[derive(Resource, Clone)]
pub struct SimSnapshotRes(pub Arc<RenderSnapshot>);

impl Default for SimSnapshotRes {
    fn default() -> Self {
        Self(Arc::new(RenderSnapshot::empty()))
    }
}

#[derive(Resource)]
pub struct SimEventRx(pub Mutex<mpsc::Receiver<Event>>);

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct SnapshotReady;

pub fn sync_snapshot_system(
    mut driver: ResMut<SimDriverRes>,
    mut snap: ResMut<SimSnapshotRes>,
) {
    if !driver.0.is_worker() {
        driver.0.republish_after_mut();
    }
    snap.0 = driver.0.snapshot();
}
