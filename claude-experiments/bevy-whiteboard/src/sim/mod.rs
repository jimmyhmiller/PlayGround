//! Pure simulation core. No Bevy types, no rendering, no coordinates.
//!
//! Each node is a named ordered list of [`Step`] primitives plus a shared
//! runtime. A packet threads through the list; instant steps run to
//! completion and blocking steps ([`Step::Buffer`], [`Step::Process`])
//! suspend the packet until a later tick. The six built-in kinds
//! (Generator / Worker / Sink / Router / Queue / Client) are just preset
//! step-lists constructed by the `add_*` methods; custom user components
//! are the same thing under a different name.
//!
//! All time is stored as [`TimeNs`] (nanoseconds, u64). Simulation is
//! event-driven: [`Sim::advance_ns`] jumps to the next scheduled event
//! rather than ticking. Emissions cascade through routers/queues
//! synchronously in the same event; `SimEvent::Traveled` is emitted once
//! per hop for visuals.

use std::collections::{HashMap, HashSet, VecDeque};

pub type NodeId = u64;
pub type EdgeId = u64;
pub type PacketId = u64;

/// A duration or instant in nanoseconds.
pub type TimeNs = u64;

pub const NS_PER_US: u64 = 1_000;
pub const NS_PER_MS: u64 = 1_000_000;
pub const NS_PER_S: u64 = 1_000_000_000;

pub fn rate_to_period_ns(rate_per_sec: f64) -> TimeNs {
    if rate_per_sec <= 0.0 {
        0
    } else {
        ((NS_PER_S as f64) / rate_per_sec).round().max(1.0) as u64
    }
}

pub fn period_ns_to_rate(period_ns: TimeNs) -> f64 {
    if period_ns == 0 {
        0.0
    } else {
        NS_PER_S as f64 / period_ns as f64
    }
}

pub fn parse_duration_ns(s: &str) -> Option<TimeNs> {
    let (num, unit) = split_number_unit(s.trim())?;
    let n: f64 = num.parse().ok()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    let mul: f64 = match unit.to_ascii_lowercase().as_str() {
        "" | "ns" => 1.0,
        "us" | "µs" | "μs" => NS_PER_US as f64,
        "ms" => NS_PER_MS as f64,
        "s" | "sec" | "secs" => NS_PER_S as f64,
        _ => return None,
    };
    let ns = (n * mul).round();
    if ns > u64::MAX as f64 {
        return None;
    }
    Some(ns as u64)
}

pub fn parse_rate_pps(s: &str) -> Option<f64> {
    let s = s.trim();
    let (num_part, denom_part) = match s.split_once('/') {
        Some((a, b)) => (a.trim(), Some(b.trim())),
        None => (s, None),
    };
    let num_part = num_part.trim_end_matches("pps").trim();
    let (num_str, si_mul) = split_si_prefix(num_part);
    let n: f64 = num_str.trim().parse().ok()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    let denom_secs: f64 = match denom_part {
        None | Some("s") | Some("sec") | Some("") => 1.0,
        Some("ms") => 1e-3,
        Some("us") | Some("µs") | Some("μs") => 1e-6,
        Some("ns") => 1e-9,
        _ => return None,
    };
    Some(n * si_mul / denom_secs)
}

fn split_number_unit(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut saw_digit = false;
    let mut saw_dot = false;
    let mut saw_exp = false;
    while i < bytes.len() {
        let c = bytes[i];
        let is_sign = (c == b'+' || c == b'-')
            && (i == 0 || (saw_exp && bytes[i - 1].eq_ignore_ascii_case(&b'e')));
        let is_exp = (c.eq_ignore_ascii_case(&b'e')) && saw_digit && !saw_exp;
        let is_dot = c == b'.' && !saw_dot && !saw_exp;
        if c.is_ascii_digit() {
            saw_digit = true;
        } else if is_sign || is_dot || is_exp {
            if is_dot {
                saw_dot = true;
            }
            if is_exp {
                saw_exp = true;
            }
        } else {
            break;
        }
        i += 1;
    }
    if !saw_digit {
        return None;
    }
    Some((&s[..i], s[i..].trim()))
}

fn split_si_prefix(s: &str) -> (&str, f64) {
    let s = s.trim();
    if s.is_empty() {
        return (s, 1.0);
    }
    let last = s.chars().last().unwrap();
    let mul = match last {
        'k' | 'K' => 1e3,
        'M' => 1e6,
        'G' => 1e9,
        'T' => 1e12,
        _ => return (s, 1.0),
    };
    (&s[..s.len() - last.len_utf8()], mul)
}

/// Abstract color identity. The sim doesn't care how colors are encoded at
/// render time; it only compares for equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Color(pub u32);

/// A single packet traveling through the sim. Carries an id (so
/// Clients can correlate replies) and an optional reply-address (origin
/// client + return path). The first node along a request's path whose
/// step-list contains a responder step consumes the reply-address and
/// synthesizes the response; everyone downstream sees a plain packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub id: PacketId,
    pub color: Color,
    pub reply: Option<ReplyAddress>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplyAddress {
    pub client: NodeId,
    pub return_path: Vec<EdgeId>,
    pub sent_at_ns: TimeNs,
}

/// Legacy reply-mode enum kept for external code that still matches on
/// it. `OneWay` = `reply.is_none()`, `Request` = `reply.is_some()` with a
/// non-consumed address, `Response` = runtime-internal (never exposed on
/// a stationary packet; responses travel via `deliver_response` and
/// aren't stored on nodes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplyMode {
    OneWay,
    Request { return_path: Vec<EdgeId> },
    Response { return_path: Vec<EdgeId> },
}

impl Packet {
    pub fn oneway(id: PacketId, color: Color) -> Self {
        Self { id, color, reply: None }
    }

    pub fn request(
        id: PacketId,
        color: Color,
        client: NodeId,
        sent_at_ns: TimeNs,
    ) -> Self {
        Self {
            id,
            color,
            reply: Some(ReplyAddress {
                client,
                return_path: Vec::new(),
                sent_at_ns,
            }),
        }
    }

    pub fn is_response(&self) -> bool {
        false // responses are transient, never stored on a node
    }
}

// ---- Node kind / steps ---------------------------------------------------

/// A coarse label used by the UI to pick icons, inspector fields, and
/// palette categories. Does NOT drive simulation behavior — only the
/// step-list does. New step-list shapes that don't correspond to one of
/// the built-ins use `Custom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Generator,
    Worker,
    Sink,
    Router,
    Queue,
    Client,
    Custom,
    /// Scripted process: its behavior is an ordered list of [`StepRow`]s
    /// rather than the lower-level [`Step`] primitives. Rows advance
    /// sequentially; each row can have its own outbound edge(s) tagged
    /// with the row index (see [`Edge::from_row`]).
    Steps,
}

/// A lifecycle row inside a [`NodeKind::Steps`] container. Variants
/// mirror the subset of [`NodeKind`]s that have sensible "step-mode"
/// behavior (i.e. the ones whose normal behavior maps cleanly onto
/// "do a thing when triggered, then signal done"). A Steps container's
/// `step_rows` is an ordered list of these; the container auto-loops
/// back to row 0 after the last row completes.
///
/// Think of a `StepRow` as "this row IS a Client" (or Worker, etc.)
/// configured for single-shot operation instead of self-rate emission.
#[derive(Clone, Debug)]
pub enum StepRow {
    /// Behaves like a Client: on row entry, fires a request packet
    /// of `color` down this row's outbound edge, waits for the
    /// response to walk back, then advances. Rate is irrelevant —
    /// emission is driven by row entry.
    Client { color: Color },
    /// Behaves like a Worker: on row entry, holds the token for
    /// `duration_ns` sim time, then advances. `color` is reserved
    /// for future color-match gating; unused for now.
    Worker { duration_ns: TimeNs, color: Color },
}

/// Simulation primitive. A node's behavior is an ordered list of these.
#[derive(Clone, Debug)]
pub enum Step {
    /// Source: synthesize a packet every `period_ns`. `one_way = false`
    /// means emit a Request carrying a reply-address back to this node.
    EmitAtRate {
        period_ns: TimeNs,
        color: Color,
        one_way: bool,
    },
    /// Pull-semantics inbound: the node actively fetches from a source
    /// upstream (queue, buffered node). Used by Worker. Nodes with
    /// `PullInbound` reject pushes — the upstream has to have a
    /// drain-to-ready-puller step (`ForwardOut`/`Buffer`-drain) to hand
    /// off.
    PullInbound,
    /// Push-semantics inbound: the node accepts packets delivered onto
    /// it by an upstream's `ForwardOut`.
    AcceptInbound,
    /// Drop the packet (raise `Lost`) if its color doesn't match.
    MatchColor { color: Color },
    /// Hold packets in a FIFO until the next step can accept. Drain
    /// triggers when a downstream puller is ready, or when this node's
    /// own next step is an instant forward.
    Buffer { capacity: usize },
    /// Hold one packet for `duration_ns` sim time, then advance.
    Process { duration_ns: TimeNs },
    /// Hand the packet to an outbound edge. Strategy picks which edge
    /// when multiple match.
    ForwardOut { strategy: ForwardStrategy },
    /// If the packet carries a reply-address, synthesize a response
    /// back to its client *now* and strip the address. The packet
    /// continues through subsequent steps as plain data (no reply
    /// obligation). Non-request packets pass through untouched.
    RespondImmediate,
    /// Same as `RespondImmediate` but conceptually triggered "after
    /// work is done." Semantically identical to RespondImmediate today
    /// — the distinction is purely positional in the step list (before
    /// vs after a `Process`).
    RespondOnComplete,
    /// Terminal absorb. Stats: increments `sink_total` / per-color.
    Consume,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardStrategy {
    /// Round-robin over ALL outbound edges, regardless of downstream
    /// readiness. Downstream is responsible for rejecting packets it
    /// can't accept. Used by sources (Generator, Client): the source
    /// fires blindly and any drop is recorded at the receiver. If the
    /// source has no outbound edges at all, it drops at self.
    BlindRoundRobin,
    /// Round-robin over READY outbound edges only. If none is ready,
    /// drop-at-self. Router semantics.
    ReadyRoundRobin,
    /// First ready outbound edge, else silently consume (no drop event
    /// and no counter tick). Worker-post-process semantics: if there is
    /// no downstream to hand to, the work is simply done.
    ForwardOrConsume,
}

// ---- Node runtime --------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub name: String,
    pub kind: NodeKind,
    pub steps: Vec<Step>,
    /// Primary color (UI tint + palette grouping). When a step carries a
    /// `color` field it is authoritative for that step, but `color` here
    /// mirrors the dominant one for UI convenience.
    pub color: Color,
    pub down: bool,

    // Source state -----------------------------------------------------
    /// Count of emissions scheduled so far for the first `EmitAtRate`.
    pub emit_scheduled: u64,
    pub outbound_cursor: usize,

    // Blocking step state ----------------------------------------------
    /// Packet currently held in a `Process` step (if any).
    pub holding: Option<Packet>,
    /// Sim time at which `holding` began processing.
    pub started_at_ns: TimeNs,
    /// Packets buffered in the node's single `Buffer` step.
    pub buffer: VecDeque<Packet>,

    // Routing cursors --------------------------------------------------
    pub cursor_per_color: HashMap<Color, usize>,

    // Stats ------------------------------------------------------------
    pub emitted: u32,
    pub processed: u32,
    pub dropped: u32,
    pub total_in: u32,
    pub total_out: u32,
    pub lost: u32,
    pub max_depth: u32,
    pub sink_total: u32,
    pub sink_per_color: HashMap<Color, u32>,

    // Client-specific stats (only meaningful if kind == Client) --------
    pub sent: u32,
    pub received: u32,
    pub outstanding: HashMap<PacketId, TimeNs>,
    pub rtt_sum_ns: u64,
    pub rtt_count: u32,

    // Composite membership ---------------------------------------------
    /// If non-empty, this node is a composite: it visually groups the
    /// listed member node ids (which still live in the outer sim). The
    /// composite itself acts as a routing waypoint: external pushes
    /// land on the composite and its `ForwardOut` fans them into
    /// internal edges that reach the real members. Edges leaving the
    /// composite's `output_port` are rendered as leaving the composite
    /// boundary but are plain sim edges from the member.
    pub contains: Vec<NodeId>,
    /// Member node that receives incoming pushes via the composite.
    pub input_port: Option<NodeId>,
    /// Member node whose outgoing edges are rendered as leaving the
    /// composite boundary (pure presentation; the sim edge is just a
    /// regular outbound from this member).
    pub output_port: Option<NodeId>,
    /// If this node is a member of a composite, the composite's id.
    pub parent: Option<NodeId>,

    // Steps-specific state -------------------------------------------------
    /// Lifecycle script for a [`NodeKind::Steps`] node. Empty for every
    /// other kind.
    pub step_rows: Vec<StepRow>,
    /// Index of the currently-executing row. `None` means the script
    /// finished (ran off the end without a Loop row) and the node is
    /// dormant.
    pub current_row: Option<usize>,
    /// Sim time at which the current row began executing. Drives Delay
    /// row completion.
    pub row_started_ns: TimeNs,
    /// When a Call row has fired a request and is waiting for the
    /// response, holds the request's packet id so `deliver_response`
    /// can match it and advance the row.
    pub row_awaiting: Option<PacketId>,
}

impl Default for NodeKind {
    fn default() -> Self {
        NodeKind::Custom
    }
}

impl Default for Color {
    fn default() -> Self {
        Color(0)
    }
}

impl Node {
    pub fn new(name: impl Into<String>, kind: NodeKind, color: Color, steps: Vec<Step>) -> Self {
        Self {
            name: name.into(),
            kind,
            color,
            steps,
            ..Default::default()
        }
    }

    // --- Accessors used by UI / tests (read) --------------------------

    /// Emission period in ns from the first `EmitAtRate` step, or 0.
    pub fn emit_period_ns(&self) -> TimeNs {
        self.steps
            .iter()
            .find_map(|s| match s {
                Step::EmitAtRate { period_ns, .. } => Some(*period_ns),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Processing duration of the first `Process` step, or 0.
    pub fn processing_ns(&self) -> TimeNs {
        self.steps
            .iter()
            .find_map(|s| match s {
                Step::Process { duration_ns } => Some(*duration_ns),
                _ => None,
            })
            .unwrap_or(0)
    }

    pub fn buffer_capacity(&self) -> usize {
        self.steps
            .iter()
            .find_map(|s| match s {
                Step::Buffer { capacity } => Some(*capacity),
                _ => None,
            })
            .unwrap_or(0)
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn has_step(&self, matches: impl Fn(&Step) -> bool) -> bool {
        self.steps.iter().any(matches)
    }

    pub fn is_source(&self) -> bool {
        self.has_step(|s| matches!(s, Step::EmitAtRate { .. }))
    }

    pub fn is_pulling(&self) -> bool {
        self.has_step(|s| matches!(s, Step::PullInbound))
    }

    pub fn accepts_push(&self) -> bool {
        self.has_step(|s| matches!(s, Step::AcceptInbound))
    }

    // --- Mutators used by UI (write) ----------------------------------

    /// Update the first `EmitAtRate` step's period. Rebases emit_scheduled
    /// to the current now so the next emission is `period_ns` in the
    /// future, not retroactive.
    pub fn set_emit_period_ns(&mut self, new_period: TimeNs, now_ns: TimeNs) {
        for s in self.steps.iter_mut() {
            if let Step::EmitAtRate { period_ns, .. } = s {
                *period_ns = new_period;
                self.emit_scheduled = if new_period == 0 { 0 } else { now_ns / new_period };
                return;
            }
        }
    }

    /// Update the first `Process` step's duration. Does not affect a
    /// packet currently being processed.
    pub fn set_processing_ns(&mut self, new_ns: TimeNs) {
        for s in self.steps.iter_mut() {
            if let Step::Process { duration_ns } = s {
                *duration_ns = new_ns.max(1);
                return;
            }
        }
    }

    /// Update the color stamped on emitted packets (first EmitAtRate)
    /// and the node's primary color. Does not affect packets already in
    /// flight or color-match steps (those keep their own color).
    pub fn set_emit_color(&mut self, new_color: Color) {
        self.color = new_color;
        for s in self.steps.iter_mut() {
            if let Step::EmitAtRate { color, .. } = s {
                *color = new_color;
                return;
            }
        }
    }

    pub fn set_match_color(&mut self, new_color: Color) {
        self.color = new_color;
        for s in self.steps.iter_mut() {
            match s {
                Step::MatchColor { color } => *color = new_color,
                Step::EmitAtRate { color, .. } => *color = new_color,
                _ => {}
            }
        }
    }
}

// ---- Sim -----------------------------------------------------------------

#[derive(Default, Clone, Debug)]
pub struct Sim {
    next_node_id: u64,
    next_edge_id: u64,
    next_packet_id: u64,
    pub nodes: HashMap<NodeId, Node>,
    pub edges: HashMap<EdgeId, Edge>,
    outbound: HashMap<NodeId, Vec<EdgeId>>,
    pub now_ns: TimeNs,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    /// Who initiates transfer across this edge. `Push` is the default:
    /// the source decides when to fire (Generator tick, Queue drain,
    /// Worker post-process Forward). `Pull` inverts that — the
    /// destination decides: when it is ready, it asks the source for
    /// one packet. A Queue→Worker edge in `Pull` mode means the Worker
    /// reaches back for work when idle, rather than the Queue pushing.
    pub mode: EdgeMode,
    /// If the source node is a [`NodeKind::Steps`], which row this
    /// edge emerges from. `None` means the edge is anchored at the
    /// node as a whole (legacy) or the source isn't a Steps node.
    pub from_row: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeMode {
    Push,
    Pull,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimEvent {
    Traveled {
        edge: EdgeId,
        color: Color,
        is_response: bool,
    },
    Processed {
        node: NodeId,
        color: Color,
    },
    Lost {
        at: NodeId,
        color: Color,
        reason: LostReason,
    },
    ResponseReceived {
        client: NodeId,
        color: Color,
        rtt_ns: TimeNs,
    },
    /// A Steps container finished its last row and wrapped its
    /// `current_row` back to 0. The renderer uses this to animate a
    /// loop-back dot traveling along the container's return arc.
    StepsLooped {
        node: NodeId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LostReason {
    NoReadyOutbound,
    QueueFull,
    RouterStarved,
    WorkerBusy,
    SinkRejected,
}

// ---- Construction / presets ---------------------------------------------

impl Sim {
    pub fn new() -> Self {
        Self::default()
    }

    fn fresh_node(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    fn fresh_packet_id(&mut self) -> PacketId {
        let id = self.next_packet_id;
        self.next_packet_id += 1;
        id
    }

    pub fn add_generator(&mut self, color: Color, period_ns: TimeNs) -> NodeId {
        let id = self.fresh_node();
        let mut node = Node::new(
            "Generator",
            NodeKind::Generator,
            color,
            vec![
                Step::EmitAtRate {
                    period_ns,
                    color,
                    one_way: true,
                },
                Step::ForwardOut {
                    strategy: ForwardStrategy::BlindRoundRobin,
                },
            ],
        );
        node.emit_scheduled = if period_ns == 0 { 0 } else { self.now_ns / period_ns };
        self.nodes.insert(id, node);
        id
    }

    pub fn add_client(&mut self, color: Color, period_ns: TimeNs) -> NodeId {
        let id = self.fresh_node();
        let mut node = Node::new(
            "Client",
            NodeKind::Client,
            color,
            vec![
                Step::EmitAtRate {
                    period_ns,
                    color,
                    one_way: false,
                },
                Step::ForwardOut {
                    strategy: ForwardStrategy::BlindRoundRobin,
                },
            ],
        );
        node.emit_scheduled = if period_ns == 0 { 0 } else { self.now_ns / period_ns };
        self.nodes.insert(id, node);
        id
    }

    pub fn add_worker(&mut self, color: Color, processing_ns: TimeNs) -> NodeId {
        let id = self.fresh_node();
        // Worker is a PULLER: it fetches from an upstream buffer when
        // idle. Pushes into a worker are rejected (the old behavior
        // where generators could push directly into a worker is kept as
        // a compatibility path — see `deliver_push`).
        let node = Node::new(
            "Worker",
            NodeKind::Worker,
            color,
            vec![
                Step::PullInbound,
                Step::MatchColor { color },
                Step::Process {
                    duration_ns: processing_ns.max(1),
                },
                Step::RespondOnComplete,
                Step::ForwardOut {
                    strategy: ForwardStrategy::ForwardOrConsume,
                },
            ],
        );
        self.nodes.insert(id, node);
        id
    }

    pub fn add_sink(&mut self, color: Color) -> NodeId {
        let id = self.fresh_node();
        let node = Node::new(
            "Sink",
            NodeKind::Sink,
            color,
            vec![
                Step::AcceptInbound,
                Step::MatchColor { color },
                Step::Consume,
            ],
        );
        self.nodes.insert(id, node);
        id
    }

    pub fn add_router(&mut self) -> NodeId {
        let id = self.fresh_node();
        let node = Node::new(
            "Router",
            NodeKind::Router,
            Color(0),
            vec![
                Step::AcceptInbound,
                Step::ForwardOut {
                    strategy: ForwardStrategy::ReadyRoundRobin,
                },
            ],
        );
        self.nodes.insert(id, node);
        id
    }

    pub fn add_queue(&mut self, color: Color, capacity: usize) -> NodeId {
        let id = self.fresh_node();
        let node = Node::new(
            "Queue",
            NodeKind::Queue,
            color,
            vec![
                Step::AcceptInbound,
                Step::MatchColor { color },
                Step::RespondImmediate,
                Step::Buffer { capacity },
                Step::ForwardOut {
                    strategy: ForwardStrategy::ReadyRoundRobin,
                },
            ],
        );
        self.nodes.insert(id, node);
        id
    }

    /// Spawn a [`NodeKind::Steps`] node with the given script. The
    /// node starts dormant if `rows` is empty; otherwise its
    /// `current_row` is set to 0 so the first row fires on the next
    /// `advance_ns` call.
    pub fn add_steps(&mut self, color: Color, rows: Vec<StepRow>) -> NodeId {
        let id = self.fresh_node();
        let mut node = Node::new("Steps", NodeKind::Steps, color, Vec::new());
        node.step_rows = rows;
        node.current_row = if node.step_rows.is_empty() { None } else { Some(0) };
        node.row_started_ns = self.now_ns;
        self.nodes.insert(id, node);
        id
    }

    /// Replace the row script for a Steps node and restart at row 0.
    /// No-op if the node isn't a Steps node.
    pub fn set_step_rows(&mut self, id: NodeId, rows: Vec<StepRow>) {
        let now = self.now_ns;
        let Some(n) = self.nodes.get_mut(&id) else { return };
        if n.kind != NodeKind::Steps {
            return;
        }
        n.step_rows = rows;
        n.current_row = if n.step_rows.is_empty() { None } else { Some(0) };
        n.row_started_ns = now;
        n.row_awaiting = None;
    }

    pub fn connect(&mut self, from: NodeId, to: NodeId) -> EdgeId {
        self.connect_with_mode(from, to, EdgeMode::Push)
    }

    pub fn connect_pull(&mut self, from: NodeId, to: NodeId) -> EdgeId {
        self.connect_with_mode(from, to, EdgeMode::Pull)
    }

    pub fn connect_with_mode(&mut self, from: NodeId, to: NodeId, mode: EdgeMode) -> EdgeId {
        self.connect_full(from, to, mode, None)
    }

    /// Like `connect` but tags the edge as emerging from a specific
    /// row of a [`NodeKind::Steps`] source. Edge rendering (tail
    /// anchor) and the Steps runtime (which edge fires for a Call row)
    /// both key off this field.
    pub fn connect_from_row(&mut self, from: NodeId, to: NodeId, from_row: usize) -> EdgeId {
        self.connect_full(from, to, EdgeMode::Push, Some(from_row))
    }

    fn connect_full(
        &mut self,
        from: NodeId,
        to: NodeId,
        mode: EdgeMode,
        from_row: Option<usize>,
    ) -> EdgeId {
        let id = self.next_edge_id;
        self.next_edge_id += 1;
        self.edges.insert(id, Edge { from, to, mode, from_row });
        self.outbound.entry(from).or_default().push(id);
        id
    }

    pub fn set_edge_mode(&mut self, id: EdgeId, mode: EdgeMode) {
        if let Some(e) = self.edges.get_mut(&id) {
            e.mode = mode;
        }
    }

    pub fn set_generator_period_ns(&mut self, id: NodeId, period_ns: TimeNs) {
        let now = self.now_ns;
        if let Some(n) = self.nodes.get_mut(&id) {
            if n.kind == NodeKind::Generator {
                n.set_emit_period_ns(period_ns, now);
            }
        }
    }

    pub fn set_client_period_ns(&mut self, id: NodeId, period_ns: TimeNs) {
        let now = self.now_ns;
        if let Some(n) = self.nodes.get_mut(&id) {
            if n.kind == NodeKind::Client {
                n.set_emit_period_ns(period_ns, now);
            }
        }
    }

    pub fn set_worker_processing_ns(&mut self, id: NodeId, processing_ns: TimeNs) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.set_processing_ns(processing_ns);
        }
    }

    pub fn set_emitter_color(&mut self, id: NodeId, color: Color) {
        if let Some(n) = self.nodes.get_mut(&id) {
            match n.kind {
                NodeKind::Generator | NodeKind::Client => n.set_emit_color(color),
                _ => {}
            }
        }
    }

    /// Remove a node and every edge incident to it. Returns the ids of
    /// removed edges. Held/buffered packets are silently discarded.
    pub fn remove_node(&mut self, id: NodeId) -> Vec<EdgeId> {
        let mut removed = Vec::new();
        let to_drop: Vec<EdgeId> = self
            .edges
            .iter()
            .filter_map(|(eid, e)| (e.from == id || e.to == id).then_some(*eid))
            .collect();
        for eid in to_drop {
            self.remove_edge_inner(eid);
            removed.push(eid);
        }
        self.outbound.remove(&id);
        self.nodes.remove(&id);
        removed
    }

    pub fn remove_edge(&mut self, id: EdgeId) -> bool {
        self.remove_edge_inner(id)
    }

    /// Pull a set of selected nodes (and the edges entirely between them)
    /// into a new composite node. External edges touching the selection
    /// are rewired to connect the outer world to the composite via
    /// auto-inserted entry/exit gateways inside. Returns the new
    /// composite's outer id plus the ids of outer edges that were
    /// removed (so the caller can despawn the corresponding Bevy edge
    /// entities) and the ids of fresh outer edges created to replace
    /// them.
    pub fn group_into_composite(
        &mut self,
        selected: &HashSet<NodeId>,
        name: impl Into<String>,
        color: Color,
    ) -> Option<GroupResult> {
        if selected.len() < 2 || !selected.iter().all(|id| self.nodes.contains_key(id)) {
            return None;
        }

        // Bucket all incident edges.
        let mut internal_edges: Vec<EdgeId> = Vec::new();
        let mut ext_in: Vec<EdgeId> = Vec::new();
        let mut ext_out: Vec<EdgeId> = Vec::new();
        for (eid, e) in &self.edges {
            let from_sel = selected.contains(&e.from);
            let to_sel = selected.contains(&e.to);
            match (from_sel, to_sel) {
                (true, true) => internal_edges.push(*eid),
                (false, true) => ext_in.push(*eid),
                (true, false) => ext_out.push(*eid),
                (false, false) => {}
            }
        }

        // Pick a stable, deterministic ordering for selection. input
        // port is the first selected node (by id) that is the target
        // of any external-in edge; else first selected node. Output
        // port is analogous for external-out.
        let mut sel_sorted: Vec<NodeId> = selected.iter().copied().collect();
        sel_sorted.sort_unstable();
        let input_port = ext_in
            .iter()
            .filter_map(|eid| self.edges.get(eid).map(|e| e.to))
            .min()
            .unwrap_or_else(|| sel_sorted[0]);
        let output_port = ext_out
            .iter()
            .filter_map(|eid| self.edges.get(eid).map(|e| e.from))
            .min()
            .unwrap_or_else(|| sel_sorted[sel_sorted.len() - 1]);

        // Create composite node in outer sim. Its steps forward
        // incoming packets into internal edges leading to member
        // nodes. Output packets from members leave via ordinary
        // outbound edges — no special routing; we only redirect the
        // external-out SIM edge so it still points at the real
        // external target (the composite is purely visual for the
        // output side).
        let composite_id = self.fresh_node();
        let mut comp = Node::new(
            name,
            NodeKind::Custom,
            color,
            vec![
                Step::AcceptInbound,
                Step::ForwardOut {
                    strategy: ForwardStrategy::BlindRoundRobin,
                },
            ],
        );
        comp.contains = sel_sorted.clone();
        comp.input_port = Some(input_port);
        comp.output_port = Some(output_port);

        // Tag each member with its parent.
        for id in &sel_sorted {
            if let Some(n) = self.nodes.get_mut(id) {
                n.parent = Some(composite_id);
            }
        }

        self.nodes.insert(composite_id, comp);

        // Rewire external-in edges: each X → A_selected becomes
        // X → composite, and we add an internal composite → A_selected
        // edge so the composite's ForwardOut pushes packets to the
        // correct member.
        let mut removed_outer_edges: Vec<EdgeId> = Vec::new();
        let mut new_outer_edges: Vec<EdgeId> = Vec::new();
        // Track (original_target → new-from-composite edge id) so we
        // don't double-create composite→A if multiple external edges
        // landed on the same A.
        let mut comp_to_member_added: HashSet<NodeId> = HashSet::new();
        for eid in &ext_in {
            let Some(e) = self.edges.get(eid).cloned() else { continue };
            self.remove_edge_inner(*eid);
            removed_outer_edges.push(*eid);
            new_outer_edges.push(self.connect(e.from, composite_id));
            if comp_to_member_added.insert(e.to) {
                self.connect(composite_id, e.to);
            }
        }
        // External-out edges: leave the sim edge as (B_selected → Y)
        // untouched. They'll be rendered as leaving the composite
        // boundary via the Bevy layer's awareness of composite
        // membership.

        Some(GroupResult {
            composite: composite_id,
            absorbed_nodes: sel_sorted,
            removed_outer_edges,
            new_outer_edges,
        })
    }

    fn remove_edge_inner(&mut self, id: EdgeId) -> bool {
        let Some(edge) = self.edges.remove(&id) else {
            return false;
        };
        if let Some(outs) = self.outbound.get_mut(&edge.from) {
            outs.retain(|e| *e != id);
        }
        true
    }
}

// ---- Stepping ------------------------------------------------------------

impl Sim {
    pub fn advance_secs(&mut self, seconds: f32) -> Vec<SimEvent> {
        let ns = (seconds.max(0.0) as f64 * NS_PER_S as f64).round() as u64;
        self.advance_ns(ns)
    }

    /// Advance sim time by `dt_ns` nanoseconds.
    pub fn advance_ns(&mut self, dt_ns: TimeNs) -> Vec<SimEvent> {
        let deadline = self.now_ns.saturating_add(dt_ns);
        let mut events = Vec::new();
        for _ in 0..10_000_000 {
            let Some(t) = self.next_event_ns() else {
                self.now_ns = deadline;
                break;
            };
            if t > deadline {
                self.now_ns = deadline;
                break;
            }
            self.now_ns = t;
            self.process_due(&mut events);
        }
        events
    }

    pub fn next_event_ns(&self) -> Option<TimeNs> {
        let mut min: Option<TimeNs> = None;
        for (nid, node) in &self.nodes {
            if node.down {
                continue;
            }
            let period = node.emit_period_ns();
            if period > 0 {
                let t = (node.emit_scheduled + 1).saturating_mul(period);
                min = Some(min.map_or(t, |m| m.min(t)));
            }
            if node.holding.is_some() {
                let t = node.started_at_ns.saturating_add(node.processing_ns());
                min = Some(min.map_or(t, |m| m.min(t)));
            }
            // Steps-node row scheduling. A row only contributes an
            // event when the sim can actually advance it:
            // - Client row: has an outbound edge for this row and isn't
            //   already waiting on a response.
            // - Worker row: always schedules its deadline.
            // - Wrap past end: auto-loop runs synchronously inside
            //   `tick_steps_nodes`, so no separate event.
            // Dormant (empty script) or stuck-without-edge Clients
            // contribute nothing so `advance_ns` can exit its loop.
            if node.kind == NodeKind::Steps {
                if let Some(i) = node.current_row {
                    if i < node.step_rows.len() {
                        match &node.step_rows[i] {
                            StepRow::Client { .. }
                                if node.row_awaiting.is_none()
                                    && self.steps_row_has_outbound(*nid, i) =>
                            {
                                min = Some(min.map_or(self.now_ns, |m| m.min(self.now_ns)));
                            }
                            StepRow::Worker { duration_ns, .. } => {
                                let t = node.row_started_ns.saturating_add(*duration_ns);
                                min = Some(min.map_or(t, |m| m.min(t)));
                            }
                            _ => {}
                        }
                    } else {
                        // Past last row but not empty: auto-loop
                        // fires synchronously on next tick.
                        if !node.step_rows.is_empty() {
                            min = Some(min.map_or(self.now_ns, |m| m.min(self.now_ns)));
                        }
                    }
                }
            }
        }
        min
    }

    fn steps_row_has_outbound(&self, nid: NodeId, row: usize) -> bool {
        self.outbound
            .get(&nid)
            .map(|outs| {
                outs.iter().any(|eid| {
                    self.edges.get(eid).and_then(|e| e.from_row) == Some(row)
                })
            })
            .unwrap_or(false)
    }

    fn process_due(&mut self, events: &mut Vec<SimEvent>) {
        self.complete_processing(events);
        self.drain_buffers(events);
        self.emit_due(events);
        self.service_pullers(events);
        self.tick_steps_nodes(events);
    }

    /// Advance every Steps node as far as it can go right now.
    ///
    /// - A Client row fires one request on its outbound edge and
    ///   stops (waits for response).
    /// - A Worker row holds the token for `duration_ns`; advance
    ///   when the deadline has passed.
    /// - After the last row, the container auto-loops: `current_row`
    ///   wraps back to 0 and we emit a `StepsLooped` event so the
    ///   renderer can animate the return hop.
    ///
    /// Inner loop is bounded — a script of all-instant rows (empty
    /// set for now, but guards against future additions) would spin.
    fn tick_steps_nodes(&mut self, events: &mut Vec<SimEvent>) {
        let now = self.now_ns;
        let mut ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|(id, n)| (n.kind == NodeKind::Steps && !n.down).then_some(*id))
            .collect();
        ids.sort_unstable();

        for nid in ids {
            for _ in 0..1024 {
                let Some(node) = self.nodes.get(&nid) else { break };
                let Some(i) = node.current_row else { break };
                if node.step_rows.is_empty() {
                    break;
                }
                if i >= node.step_rows.len() {
                    // Past last row — auto-loop to 0 and signal the
                    // visual so it can draw the return animation.
                    if let Some(n) = self.nodes.get_mut(&nid) {
                        n.current_row = Some(0);
                        n.row_started_ns = now;
                        n.row_awaiting = None;
                    }
                    events.push(SimEvent::StepsLooped { node: nid });
                    continue;
                }
                let row = node.step_rows[i].clone();
                match row {
                    StepRow::Client { color } => {
                        if node.row_awaiting.is_some() {
                            // Request sent, response in flight.
                            break;
                        }
                        let edge_for_row = self
                            .outbound
                            .get(&nid)
                            .and_then(|outs| {
                                outs.iter().copied().find(|eid| {
                                    self.edges
                                        .get(eid)
                                        .and_then(|e| e.from_row)
                                        == Some(i)
                                })
                            });
                        let Some(edge_id) = edge_for_row else { break };
                        let pid = self.fresh_packet_id();
                        let packet = Packet::request(pid, color, nid, now);
                        if let Some(n) = self.nodes.get_mut(&nid) {
                            n.row_awaiting = Some(pid);
                            n.sent += 1;
                            n.outstanding.insert(pid, now);
                        }
                        self.travel_forward(edge_id, packet, events);
                        break;
                    }
                    StepRow::Worker { duration_ns, .. } => {
                        if now < node.row_started_ns.saturating_add(duration_ns) {
                            break;
                        }
                        self.advance_step_row(nid);
                        continue;
                    }
                }
            }
        }
    }

    fn advance_step_row(&mut self, nid: NodeId) {
        let now = self.now_ns;
        if let Some(n) = self.nodes.get_mut(&nid) {
            n.current_row = n.current_row.map(|i| i + 1);
            n.row_started_ns = now;
            n.row_awaiting = None;
        }
    }

    /// Update the `duration_ns` of the Worker row at `row` inside a
    /// Steps script. Silently no-ops if the id isn't a Steps node,
    /// the row is out of range, or the row isn't a Worker.
    pub fn set_steps_worker_duration_ns(&mut self, id: NodeId, row: usize, ns: TimeNs) {
        let Some(n) = self.nodes.get_mut(&id) else { return };
        if n.kind != NodeKind::Steps {
            return;
        }
        if let Some(StepRow::Worker { duration_ns, .. }) = n.step_rows.get_mut(row) {
            *duration_ns = ns.max(1);
        }
    }

    /// Append a row to a Steps node's script. If the node was
    /// dormant (empty script), starts the script at row 0. Returns
    /// the new row's index, or `None` if the node isn't a Steps node.
    pub fn push_step_row(&mut self, id: NodeId, row: StepRow) -> Option<usize> {
        let now = self.now_ns;
        let n = self.nodes.get_mut(&id)?;
        if n.kind != NodeKind::Steps {
            return None;
        }
        n.step_rows.push(row);
        let idx = n.step_rows.len() - 1;
        if n.current_row.is_none() {
            n.current_row = Some(0);
            n.row_started_ns = now;
            n.row_awaiting = None;
        }
        Some(idx)
    }

    /// Pull-side execution: find every node that is currently ready
    /// to accept (its step list gates pass) and has at least one
    /// Pull-mode inbound edge. Ask each such source to yield one
    /// packet (currently only a buffered source, i.e. a Queue, can
    /// yield). If yielded, the packet travels across the pull edge to
    /// the puller's usual arrival path. At most one pull per puller
    /// per tick — re-entry via the next `process_due` call handles
    /// steady draining as the puller becomes idle again.
    fn service_pullers(&mut self, events: &mut Vec<SimEvent>) {
        let mut pullers: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|(id, _)| {
                let has_pull = self
                    .edges
                    .values()
                    .any(|e| e.to == *id && e.mode == EdgeMode::Pull);
                has_pull.then_some(*id)
            })
            .collect();
        pullers.sort_unstable();

        for pid in pullers {
            let Some(color) = self.nodes.get(&pid).map(|n| n.color) else { continue };
            if !self.is_ready_now(pid, color) {
                continue;
            }
            let mut pull_in: Vec<EdgeId> = self
                .edges
                .iter()
                .filter_map(|(eid, e)| {
                    (e.to == pid && e.mode == EdgeMode::Pull).then_some(*eid)
                })
                .collect();
            pull_in.sort_unstable();
            for eid in pull_in {
                let Some(edge) = self.edges.get(&eid).cloned() else { continue };
                if let Some(packet) = self.try_yield_buffered(edge.from, color) {
                    self.travel_forward(eid, packet, events);
                    break;
                }
            }
        }
    }

    /// If `source` is a buffered node (has a `Buffer` step and a
    /// non-empty buffer whose front matches `color`), pop and return
    /// the front packet. Otherwise `None`. This is the only yield
    /// pathway today — generators and other sources are not
    /// pull-responsive yet.
    fn try_yield_buffered(&mut self, source: NodeId, color: Color) -> Option<Packet> {
        let n = self.nodes.get_mut(&source)?;
        if n.down {
            return None;
        }
        if !n.steps.iter().any(|s| matches!(s, Step::Buffer { .. })) {
            return None;
        }
        if n.buffer.front().map(|p| p.color) != Some(color) {
            return None;
        }
        let p = n.buffer.pop_front()?;
        n.total_out += 1;
        Some(p)
    }

    // --- Source emission --------------------------------------------------

    fn emit_due(&mut self, events: &mut Vec<SimEvent>) {
        let now = self.now_ns;
        let mut ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|(id, n)| {
                if !n.down
                    && n.steps
                        .iter()
                        .any(|s| matches!(s, Step::EmitAtRate { period_ns, .. } if *period_ns > 0))
                {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        ids.sort_unstable();

        for nid in ids {
            let (n_to_emit, color, one_way) = {
                let Some(node) = self.nodes.get_mut(&nid) else {
                    continue;
                };
                let (period, color, one_way) = node
                    .steps
                    .iter()
                    .find_map(|s| match s {
                        Step::EmitAtRate {
                            period_ns,
                            color,
                            one_way,
                        } if *period_ns > 0 => Some((*period_ns, *color, *one_way)),
                        _ => None,
                    })
                    .unwrap();
                let expected = now / period;
                let n = expected.saturating_sub(node.emit_scheduled) as u32;
                node.emit_scheduled = expected;
                (n, color, one_way)
            };

            for _ in 0..n_to_emit {
                let pid = self.fresh_packet_id();
                let packet = if one_way {
                    Packet::oneway(pid, color)
                } else {
                    Packet::request(pid, color, nid, now)
                };
                if !one_way {
                    if let Some(node) = self.nodes.get_mut(&nid) {
                        node.sent += 1;
                        node.outstanding.insert(pid, now);
                    }
                }
                // Source node runs its step list from just after
                // EmitAtRate. In practice that means: start at the
                // `ForwardOut` step.
                let start_idx = self
                    .nodes
                    .get(&nid)
                    .and_then(|n| {
                        n.steps
                            .iter()
                            .position(|s| matches!(s, Step::EmitAtRate { .. }))
                            .map(|i| i + 1)
                    })
                    .unwrap_or(0);
                let outcome = self.run_steps_from(nid, packet, start_idx, events);
                if let StepOutcome::Dropped { reason, color } = outcome {
                    if let Some(node) = self.nodes.get_mut(&nid) {
                        node.dropped += 1;
                    }
                    events.push(SimEvent::Lost {
                        at: nid,
                        color,
                        reason,
                    });
                } else if matches!(outcome, StepOutcome::Forwarded) {
                    if let Some(node) = self.nodes.get_mut(&nid) {
                        node.emitted += 1;
                    }
                }
            }
        }
    }

    // --- Worker / Process completion --------------------------------------

    fn complete_processing(&mut self, events: &mut Vec<SimEvent>) {
        let now = self.now_ns;
        let mut ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|(id, n)| {
                if n.holding.is_some() && now >= n.started_at_ns + n.processing_ns() {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        ids.sort_unstable();

        for nid in ids {
            let packet = match self.nodes.get_mut(&nid) {
                Some(n) => match n.holding.take() {
                    Some(p) => p,
                    None => continue,
                },
                None => continue,
            };
            let color = packet.color;
            // Find the Process step index and resume from after it.
            let resume_idx = match self
                .nodes
                .get(&nid)
                .and_then(|n| n.steps.iter().position(|s| matches!(s, Step::Process { .. })))
            {
                Some(i) => i + 1,
                None => continue,
            };
            let _ = self.run_steps_from(nid, packet, resume_idx, events);
            if let Some(n) = self.nodes.get_mut(&nid) {
                n.processed += 1;
            }
            events.push(SimEvent::Processed { node: nid, color });
        }
    }

    // --- Buffer drain -----------------------------------------------------

    fn drain_buffers(&mut self, events: &mut Vec<SimEvent>) {
        let mut ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|(id, n)| (!n.buffer.is_empty() && !n.down).then_some(*id))
            .collect();
        ids.sort_unstable();

        for nid in ids {
            loop {
                let front_color = match self.nodes.get(&nid) {
                    Some(n) if !n.down => n.buffer.front().map(|p| p.color),
                    _ => None,
                };
                let Some(color) = front_color else { break };

                let candidates = self.ready_outbound_candidates(nid, color);
                if candidates.is_empty() {
                    break;
                }
                let pick = {
                    let Some(n) = self.nodes.get_mut(&nid) else { break };
                    let cur = n.cursor_per_color.entry(color).or_insert(0);
                    let i = *cur % candidates.len();
                    *cur = (i + 1) % candidates.len();
                    i
                };
                let edge_id = candidates[pick];
                let packet = match self.nodes.get_mut(&nid) {
                    Some(n) => match n.buffer.pop_front() {
                        Some(p) => {
                            n.total_out += 1;
                            p
                        }
                        None => break,
                    },
                    None => break,
                };
                self.travel_forward(edge_id, packet, events);
            }
        }
    }

    // --- Step executor ----------------------------------------------------

    fn run_steps_from(
        &mut self,
        nid: NodeId,
        mut packet: Packet,
        mut idx: usize,
        events: &mut Vec<SimEvent>,
    ) -> StepOutcome {
        loop {
            let step = match self
                .nodes
                .get(&nid)
                .and_then(|n| n.steps.get(idx).cloned())
            {
                Some(s) => s,
                None => return StepOutcome::Consumed,
            };

            match step {
                Step::EmitAtRate { .. }
                | Step::PullInbound
                | Step::AcceptInbound => {
                    // Entry-style steps — skipped when advancing.
                    idx += 1;
                }
                Step::MatchColor { color } => {
                    if packet.color != color {
                        return StepOutcome::Dropped {
                            reason: drop_reason_for(self.nodes.get(&nid)),
                            color: packet.color,
                        };
                    }
                    idx += 1;
                }
                Step::Buffer { capacity } => {
                    let Some(n) = self.nodes.get_mut(&nid) else {
                        return StepOutcome::Consumed;
                    };
                    if n.buffer.len() >= capacity {
                        return StepOutcome::Dropped {
                            reason: LostReason::QueueFull,
                            color: packet.color,
                        };
                    }
                    n.buffer.push_back(packet);
                    n.total_in += 1;
                    let depth = n.buffer.len() as u32;
                    if depth > n.max_depth {
                        n.max_depth = depth;
                    }
                    return StepOutcome::Buffered;
                }
                Step::Process { duration_ns: _ } => {
                    let Some(n) = self.nodes.get_mut(&nid) else {
                        return StepOutcome::Consumed;
                    };
                    if n.holding.is_some() {
                        return StepOutcome::Dropped {
                            reason: LostReason::WorkerBusy,
                            color: packet.color,
                        };
                    }
                    n.holding = Some(packet);
                    n.started_at_ns = self.now_ns;
                    return StepOutcome::Processing;
                }
                Step::RespondImmediate | Step::RespondOnComplete => {
                    if let Some(addr) = packet.reply.take() {
                        self.deliver_response(nid, &addr, packet.id, packet.color, events);
                    }
                    idx += 1;
                }
                Step::ForwardOut { strategy } => {
                    let (candidates, empty_is_drop) = match strategy {
                        ForwardStrategy::BlindRoundRobin => {
                            // Blind push: round-robin across all
                            // PUSH-mode outbound (readiness ignored;
                            // pull edges are excluded — the receiver
                            // will come fetch when ready). Only drop
                            // if the push-outbound set is empty.
                            let outs: Vec<EdgeId> = self
                                .outbound
                                .get(&nid)
                                .map(|v| {
                                    v.iter()
                                        .copied()
                                        .filter(|eid| {
                                            self.edges
                                                .get(eid)
                                                .map(|e| e.mode == EdgeMode::Push)
                                                .unwrap_or(false)
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();
                            (outs, true)
                        }
                        ForwardStrategy::ReadyRoundRobin => {
                            (self.ready_outbound_candidates(nid, packet.color), true)
                        }
                        ForwardStrategy::ForwardOrConsume => {
                            (self.ready_outbound_candidates(nid, packet.color), false)
                        }
                    };
                    if candidates.is_empty() {
                        if empty_is_drop {
                            return StepOutcome::Dropped {
                                reason: no_ready_reason_for(self.nodes.get(&nid)),
                                color: packet.color,
                            };
                        } else {
                            return StepOutcome::Consumed;
                        }
                    }
                    let pick = match strategy {
                        ForwardStrategy::BlindRoundRobin
                        | ForwardStrategy::ReadyRoundRobin => {
                            let Some(n) = self.nodes.get_mut(&nid) else {
                                return StepOutcome::Consumed;
                            };
                            let cur = n.cursor_per_color.entry(packet.color).or_insert(0);
                            let i = *cur % candidates.len();
                            *cur = (i + 1) % candidates.len();
                            i
                        }
                        ForwardStrategy::ForwardOrConsume => 0,
                    };
                    let edge_id = candidates[pick];
                    self.travel_forward(edge_id, packet, events);
                    return StepOutcome::Forwarded;
                }
                Step::Consume => {
                    if let Some(n) = self.nodes.get_mut(&nid) {
                        *n.sink_per_color.entry(packet.color).or_insert(0) += 1;
                        n.sink_total += 1;
                    }
                    return StepOutcome::Consumed;
                }
            }
        }
    }

    /// Hop a packet across an edge forward and deliver at the far end.
    fn travel_forward(
        &mut self,
        edge_id: EdgeId,
        mut packet: Packet,
        events: &mut Vec<SimEvent>,
    ) {
        let Some(edge) = self.edges.get(&edge_id) else {
            return;
        };
        let to = edge.to;
        events.push(SimEvent::Traveled {
            edge: edge_id,
            color: packet.color,
            is_response: false,
        });
        if let Some(ref mut addr) = packet.reply {
            addr.return_path.push(edge_id);
        }
        self.deliver_push(to, packet, events);
    }

    /// Deliver a packet that was pushed to `node` from upstream. Enters
    /// the node at its first `AcceptInbound` step; if the node is
    /// pull-only, the push is rejected.
    fn deliver_push(&mut self, node: NodeId, packet: Packet, events: &mut Vec<SimEvent>) {
        let color = packet.color;
        let start_idx = match self.nodes.get(&node) {
            Some(n) => {
                // Accept point: first AcceptInbound step, else (for
                // compatibility with pull-workers that still need to
                // function when wired directly from a generator) the
                // first MatchColor / Process step.
                if let Some(i) = n.steps.iter().position(|s| matches!(s, Step::AcceptInbound)) {
                    i + 1
                } else if let Some(i) = n.steps.iter().position(|s| {
                    matches!(
                        s,
                        Step::MatchColor { .. } | Step::Process { .. } | Step::Buffer { .. }
                    )
                }) {
                    i
                } else {
                    0
                }
            }
            None => return,
        };
        let outcome = self.run_steps_from(node, packet, start_idx, events);
        if let StepOutcome::Dropped { reason, color: c } = outcome {
            if let Some(n) = self.nodes.get_mut(&node) {
                match reason {
                    LostReason::QueueFull => n.lost += 1,
                    _ => n.dropped += 1,
                }
            }
            events.push(SimEvent::Lost {
                at: node,
                color: c,
                reason,
            });
        }
    }

    /// Walk a response back along the recorded path to the originating
    /// client and close out the RTT.
    fn deliver_response(
        &mut self,
        at: NodeId,
        addr: &ReplyAddress,
        packet_id: PacketId,
        color: Color,
        events: &mut Vec<SimEvent>,
    ) {
        let mut cursor = at;
        let mut path = addr.return_path.clone();
        while let Some(edge_id) = path.pop() {
            let Some(edge) = self.edges.get(&edge_id) else {
                return;
            };
            if edge.to != cursor {
                return;
            }
            events.push(SimEvent::Traveled {
                edge: edge_id,
                color,
                is_response: true,
            });
            cursor = edge.from;
        }
        let now = self.now_ns;
        if cursor == addr.client {
            if let Some(n) = self.nodes.get_mut(&cursor) {
                match n.kind {
                    NodeKind::Client => {
                        if let Some(send_time) = n.outstanding.remove(&packet_id) {
                            let rtt = now.saturating_sub(send_time);
                            n.received += 1;
                            n.rtt_sum_ns = n.rtt_sum_ns.saturating_add(rtt);
                            n.rtt_count += 1;
                            events.push(SimEvent::ResponseReceived {
                                client: cursor,
                                color,
                                rtt_ns: rtt,
                            });
                        }
                    }
                    NodeKind::Steps => {
                        // If this response matches the Call row we're
                        // blocked on, clear the wait and advance. RTT
                        // stats are tracked the same way as for Client
                        // so the inspector has a consistent readout.
                        let send_time = n.outstanding.remove(&packet_id);
                        if n.row_awaiting == Some(packet_id) {
                            n.row_awaiting = None;
                            n.current_row = n.current_row.map(|i| i + 1);
                            n.row_started_ns = now;
                        }
                        if let Some(st) = send_time {
                            let rtt = now.saturating_sub(st);
                            n.received += 1;
                            n.rtt_sum_ns = n.rtt_sum_ns.saturating_add(rtt);
                            n.rtt_count += 1;
                            events.push(SimEvent::ResponseReceived {
                                client: cursor,
                                color,
                                rtt_ns: rtt,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // --- Readiness --------------------------------------------------------

    /// Push-mode outbound edges whose destination is ready for
    /// `color` right now. Pull edges are deliberately excluded — only
    /// the pull-side executor uses them.
    fn ready_outbound_candidates(&self, source: NodeId, color: Color) -> Vec<EdgeId> {
        let Some(outs) = self.outbound.get(&source) else {
            return Vec::new();
        };
        outs.iter()
            .copied()
            .filter(|eid| {
                let e = &self.edges[eid];
                e.mode == EdgeMode::Push && self.is_ready_now(e.to, color)
            })
            .collect()
    }

    pub fn is_ready_now(&self, node: NodeId, color: Color) -> bool {
        let mut visited = HashSet::new();
        self.ready_now_inner(node, color, &mut visited)
    }

    fn ready_now_inner(
        &self,
        node: NodeId,
        color: Color,
        visited: &mut HashSet<NodeId>,
    ) -> bool {
        if !visited.insert(node) {
            return false;
        }
        let Some(n) = self.nodes.get(&node) else {
            return false;
        };
        if n.down {
            return false;
        }
        // Walk steps to determine ready-now.
        for step in &n.steps {
            match step {
                Step::AcceptInbound | Step::PullInbound | Step::EmitAtRate { .. } => continue,
                Step::MatchColor { color: c } => {
                    if *c != color {
                        return false;
                    }
                }
                Step::Buffer { capacity } => {
                    return n.buffer.len() < *capacity;
                }
                Step::Process { .. } => {
                    return n.holding.is_none();
                }
                Step::RespondImmediate | Step::RespondOnComplete => continue,
                Step::ForwardOut { .. } => {
                    // Pass-through: ready iff some downstream is ready.
                    let Some(outs) = self.outbound.get(&node) else {
                        return false;
                    };
                    return outs.iter().any(|eid| {
                        let e = &self.edges[eid];
                        self.ready_now_inner(e.to, color, visited)
                    });
                }
                Step::Consume => return true,
            }
        }
        false
    }

    // --- Analytical rates -------------------------------------------------

    pub fn analytical_edge_rates(&self) -> HashMap<EdgeId, f64> {
        let mut colors: Vec<Color> = self
            .nodes
            .values()
            .filter_map(|n| {
                n.steps.iter().find_map(|s| match s {
                    Step::EmitAtRate {
                        period_ns, color, ..
                    } if !n.down && *period_ns > 0 => Some(*color),
                    _ => None,
                })
            })
            .collect();
        colors.sort_unstable();
        colors.dedup();

        let mut color_rates: HashMap<(EdgeId, Color), f64> = HashMap::new();
        let iters = self.nodes.len() * 3 + 8;
        for _ in 0..iters {
            let mut changed = false;
            for color in &colors {
                let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
                for nid in node_ids {
                    let push = self.push_out_color(nid, *color, &color_rates);
                    let accepting_outs: Vec<EdgeId> = self
                        .outbound
                        .get(&nid)
                        .map(|os| {
                            os.iter()
                                .copied()
                                .filter(|eid| {
                                    self.target_accepts_color(self.edges[eid].to, *color)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let caps: Vec<f64> = accepting_outs
                        .iter()
                        .map(|eid| {
                            self.accept_rate_color(self.edges[eid].to, *color, &color_rates)
                        })
                        .collect();
                    let distributed = water_fill(push, &caps);
                    for (eid, new) in accepting_outs.iter().zip(distributed.iter()) {
                        let key = (*eid, *color);
                        let old = color_rates.get(&key).copied().unwrap_or(0.0);
                        if (new - old).abs() > 1e-6 {
                            color_rates.insert(key, *new);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut rates: HashMap<EdgeId, f64> = HashMap::new();
        for ((eid, _), r) in &color_rates {
            *rates.entry(*eid).or_insert(0.0) += r;
        }
        for eid in self.edges.keys() {
            rates.entry(*eid).or_insert(0.0);
        }
        rates
    }

    fn push_out_color(
        &self,
        id: NodeId,
        color: Color,
        cr: &HashMap<(EdgeId, Color), f64>,
    ) -> f64 {
        let Some(n) = self.nodes.get(&id) else {
            return 0.0;
        };
        if n.down {
            return 0.0;
        }
        let inbound: f64 = self
            .edges
            .iter()
            .filter(|(_, e)| e.to == id)
            .map(|(eid, _)| cr.get(&(*eid, color)).copied().unwrap_or(0.0))
            .sum();
        // Source?
        if let Some((period, c)) = n.steps.iter().find_map(|s| match s {
            Step::EmitAtRate {
                period_ns, color, ..
            } if *period_ns > 0 => Some((*period_ns, *color)),
            _ => None,
        }) {
            if c == color {
                return NS_PER_S as f64 / period as f64;
            } else {
                return 0.0;
            }
        }
        // Worker-like: capped by processing rate and color match.
        if let Some(proc_ns) = n.steps.iter().find_map(|s| match s {
            Step::Process { duration_ns } => Some(*duration_ns),
            _ => None,
        }) {
            let color_ok = n
                .steps
                .iter()
                .find_map(|s| match s {
                    Step::MatchColor { color } => Some(*color),
                    _ => None,
                })
                .map(|c| c == color)
                .unwrap_or(true);
            if !color_ok || proc_ns == 0 {
                return 0.0;
            }
            return inbound.min(NS_PER_S as f64 / proc_ns as f64);
        }
        // Queue / router / buffer pass-through: gated by color if a
        // MatchColor step exists.
        let color_ok = n
            .steps
            .iter()
            .find_map(|s| match s {
                Step::MatchColor { color } => Some(*color),
                _ => None,
            })
            .map(|c| c == color)
            .unwrap_or(true);
        if !color_ok {
            return 0.0;
        }
        // Sinks don't push out.
        if n.steps.iter().any(|s| matches!(s, Step::Consume)) {
            return 0.0;
        }
        inbound
    }

    fn target_accepts_color(&self, id: NodeId, color: Color) -> bool {
        let Some(n) = self.nodes.get(&id) else {
            return false;
        };
        let mc = n.steps.iter().find_map(|s| match s {
            Step::MatchColor { color } => Some(*color),
            _ => None,
        });
        match mc {
            Some(c) => c == color,
            None => {
                // No MatchColor (router): accept if any downstream accepts.
                if let Some(outs) = self.outbound.get(&id) {
                    outs.iter()
                        .any(|eid| self.target_accepts_color(self.edges[eid].to, color))
                } else {
                    false
                }
            }
        }
    }

    fn accept_rate_color(
        &self,
        id: NodeId,
        color: Color,
        _cr: &HashMap<(EdgeId, Color), f64>,
    ) -> f64 {
        let Some(n) = self.nodes.get(&id) else {
            return 0.0;
        };
        if n.down {
            return 0.0;
        }
        let color_ok = n
            .steps
            .iter()
            .find_map(|s| match s {
                Step::MatchColor { color } => Some(*color),
                _ => None,
            })
            .map(|c| c == color)
            .unwrap_or(true);
        if !color_ok {
            return 0.0;
        }
        // Worker-like cap
        if let Some(proc_ns) = n.steps.iter().find_map(|s| match s {
            Step::Process { duration_ns } => Some(*duration_ns),
            _ => None,
        }) {
            if proc_ns == 0 {
                return 0.0;
            }
            return NS_PER_S as f64 / proc_ns as f64;
        }
        // Queue / Router / Sink: unlimited accept rate.
        f64::INFINITY
    }
}

/// Result of `Sim::group_into_composite`. Contains the new composite's
/// outer id plus the edge-id deltas the caller needs to reconcile Bevy
/// entities with.
#[derive(Debug)]
pub struct GroupResult {
    pub composite: NodeId,
    pub absorbed_nodes: Vec<NodeId>,
    pub removed_outer_edges: Vec<EdgeId>,
    pub new_outer_edges: Vec<EdgeId>,
}

// ---- Outcomes & helpers --------------------------------------------------

enum StepOutcome {
    Forwarded,
    Consumed,
    Buffered,
    Processing,
    Dropped { reason: LostReason, color: Color },
}

fn drop_reason_for(node: Option<&Node>) -> LostReason {
    match node.map(|n| n.kind) {
        Some(NodeKind::Sink) => LostReason::SinkRejected,
        Some(NodeKind::Queue) => LostReason::QueueFull,
        Some(NodeKind::Worker) => LostReason::WorkerBusy,
        _ => LostReason::NoReadyOutbound,
    }
}

fn no_ready_reason_for(node: Option<&Node>) -> LostReason {
    match node.map(|n| n.kind) {
        Some(NodeKind::Router) => LostReason::RouterStarved,
        _ => LostReason::NoReadyOutbound,
    }
}

/// Distribute `push` across sibling edges whose destinations accept at
/// rates `caps[i]`. Each edge starts with an equal share; any edge whose
/// share exceeds its cap is pinned at the cap and the overflow is
/// redistributed across the remaining edges — iterated until either all
/// flow is placed or every edge is at its cap.
fn water_fill(push: f64, caps: &[f64]) -> Vec<f64> {
    let n = caps.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }
    let mut remaining = push.max(0.0);
    let eps = 1e-12;
    loop {
        let active: Vec<usize> = (0..n).filter(|&i| caps[i] - out[i] > eps).collect();
        if active.is_empty() || remaining <= eps {
            break;
        }
        let share = remaining / active.len() as f64;
        let min_head = active
            .iter()
            .map(|&i| caps[i] - out[i])
            .fold(f64::INFINITY, f64::min);
        if share <= min_head + eps {
            for &i in &active {
                out[i] += share;
            }
            break;
        }
        for &i in &active {
            out[i] += min_head;
        }
        remaining -= min_head * active.len() as f64;
    }
    out
}

// ---- Tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Color = Color(0xE54C4C);
    const YELLOW: Color = Color(0xF2B233);
    const BLUE: Color = Color(0x4080E8);

    fn add_gen(sim: &mut Sim, color: Color, rate_per_sec: f64) -> NodeId {
        sim.add_generator(color, rate_to_period_ns(rate_per_sec))
    }

    fn add_worker_secs(sim: &mut Sim, color: Color, secs: f64) -> NodeId {
        let ns = (secs * NS_PER_S as f64).round() as u64;
        sim.add_worker(color, ns.max(1))
    }

    fn advance_secs(sim: &mut Sim, secs: f64) -> Vec<SimEvent> {
        let ns = (secs * NS_PER_S as f64).round() as u64;
        sim.advance_ns(ns)
    }

    fn worker_processed(sim: &Sim, id: NodeId) -> u32 {
        sim.nodes[&id].processed
    }

    fn worker_handled(sim: &Sim, id: NodeId) -> u32 {
        let n = &sim.nodes[&id];
        n.processed + n.holding.is_some() as u32
    }

    fn sink_total(sim: &Sim, id: NodeId) -> u32 {
        sim.nodes[&id].sink_total
    }

    fn queue_depth(sim: &Sim, id: NodeId) -> usize {
        sim.nodes[&id].buffer.len()
    }

    #[test]
    fn generator_with_no_outbound_emits_nothing() {
        let mut sim = Sim::new();
        add_gen(&mut sim, RED, 10.0);
        let events = advance_secs(&mut sim, 1.0);
        let traveled = events
            .iter()
            .filter(|e| matches!(e, SimEvent::Traveled { .. }))
            .count();
        assert_eq!(traveled, 0);
    }

    #[test]
    fn generator_to_sink_color_match() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 3.0);
        let s = sim.add_sink(RED);
        sim.connect(g, s);
        advance_secs(&mut sim, 1.0);
        assert_eq!(sink_total(&sim, s), 3);
    }

    #[test]
    fn sink_rejects_wrong_color() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 5.0);
        let s = sim.add_sink(YELLOW);
        sim.connect(g, s);
        advance_secs(&mut sim, 1.0);
        assert_eq!(sink_total(&sim, s), 0);
        assert_eq!(sim.nodes[&g].dropped, 0);
        assert_eq!(sim.nodes[&s].dropped, 5);
    }

    #[test]
    fn worker_processes_at_its_rate() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 100.0);
        let w = add_worker_secs(&mut sim, RED, 0.5);
        sim.connect(g, w);
        advance_secs(&mut sim, 2.0);
        assert_eq!(worker_handled(&sim, w), 4);
    }

    #[test]
    fn queue_smooths_burst() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 4.0);
        let q = sim.add_queue(RED, usize::MAX);
        let w = add_worker_secs(&mut sim, RED, 0.5);
        sim.connect(g, q);
        sim.connect(q, w);
        advance_secs(&mut sim, 2.0);
        let processed = worker_processed(&sim, w);
        let depth = queue_depth(&sim, q);
        let in_worker = if sim.nodes[&w].holding.is_some() { 1 } else { 0 };
        assert_eq!(processed + depth as u32 + in_worker, 8);
        assert!(processed >= 3 && processed <= 4);
    }

    #[test]
    fn four_workers_scale_throughput() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 8.0);
        let q = sim.add_queue(RED, usize::MAX);
        sim.connect(g, q);
        let mut workers = Vec::new();
        for _ in 0..4 {
            let w = add_worker_secs(&mut sim, RED, 0.5);
            sim.connect(q, w);
            workers.push(w);
        }
        advance_secs(&mut sim, 5.0);

        let total_processed: u32 = workers.iter().map(|id| worker_processed(&sim, *id)).sum();
        let in_workers: u32 = workers
            .iter()
            .map(|id| sim.nodes[id].holding.is_some() as u32)
            .sum();
        let depth = queue_depth(&sim, q) as u32;
        let emitted = sim.nodes[&g].emitted;
        assert_eq!(total_processed + in_workers + depth, emitted);
        assert!(depth <= 2, "queue depth {depth} grew unexpectedly");
        assert!(total_processed >= 35, "processed only {total_processed}");
    }

    #[test]
    fn queue_rejects_wrong_color_at_generator() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 5.0);
        let q = sim.add_queue(YELLOW, usize::MAX);
        sim.connect(g, q);
        advance_secs(&mut sim, 1.0);
        assert_eq!(queue_depth(&sim, q), 0);
    }

    #[test]
    fn router_skips_wrong_color_queues() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 10.0);
        let r = sim.add_router();
        sim.connect(g, r);
        let q_yellow = sim.add_queue(YELLOW, usize::MAX);
        let q_red = sim.add_queue(RED, usize::MAX);
        sim.connect(r, q_yellow);
        sim.connect(r, q_red);
        advance_secs(&mut sim, 1.0);
        assert_eq!(queue_depth(&sim, q_yellow), 0);
        assert_eq!(queue_depth(&sim, q_red), 10);
    }

    #[test]
    fn router_round_robins_matching_color_downstream() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 10.0);
        let r = sim.add_router();
        sim.connect(g, r);
        let w1 = add_worker_secs(&mut sim, RED, 0.001);
        let w2 = add_worker_secs(&mut sim, RED, 0.001);
        sim.connect(r, w1);
        sim.connect(r, w2);
        advance_secs(&mut sim, 1.0);
        let h1 = worker_handled(&sim, w1);
        let h2 = worker_handled(&sim, w2);
        assert_eq!(h1 + h2, 10);
        assert!((h1 as i32 - h2 as i32).abs() <= 1);
    }

    #[test]
    fn down_worker_is_bypassed_by_router() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 4.0);
        let r = sim.add_router();
        sim.connect(g, r);
        let w_down = add_worker_secs(&mut sim, RED, 0.001);
        let w_up = add_worker_secs(&mut sim, RED, 0.001);
        if let Some(n) = sim.nodes.get_mut(&w_down) {
            n.down = true;
        }
        sim.connect(r, w_down);
        sim.connect(r, w_up);
        advance_secs(&mut sim, 1.0);
        assert_eq!(worker_handled(&sim, w_down), 0);
        assert_eq!(worker_handled(&sim, w_up), 4);
    }

    #[test]
    fn mixed_colors_each_routed_to_its_sink() {
        let mut sim = Sim::new();
        let g_red = add_gen(&mut sim, RED, 3.0);
        let g_yellow = add_gen(&mut sim, YELLOW, 5.0);
        let r = sim.add_router();
        sim.connect(g_red, r);
        sim.connect(g_yellow, r);
        let s_red = sim.add_sink(RED);
        let s_yellow = sim.add_sink(YELLOW);
        sim.connect(r, s_red);
        sim.connect(r, s_yellow);
        advance_secs(&mut sim, 1.0);
        assert_eq!(sink_total(&sim, s_red), 3);
        assert_eq!(sink_total(&sim, s_yellow), 5);
    }

    #[test]
    fn queue_backpressures_slow_worker() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 10.0);
        let q = sim.add_queue(RED, usize::MAX);
        let w = add_worker_secs(&mut sim, RED, 1.0);
        sim.connect(g, q);
        sim.connect(q, w);
        advance_secs(&mut sim, 2.0);
        let processed = worker_processed(&sim, w);
        let depth = queue_depth(&sim, q) as u32;
        let in_worker = sim.nodes[&w].holding.is_some() as u32;
        assert_eq!(processed + depth + in_worker, 20);
        assert!(processed >= 1 && processed <= 2);
        assert!(depth >= 17);
    }

    #[test]
    fn router_drops_when_no_downstream() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 5.0);
        let r = sim.add_router();
        sim.connect(g, r);
        let events = advance_secs(&mut sim, 1.0);
        assert_eq!(sim.nodes[&g].dropped, 0);
        let router_starved = events
            .iter()
            .filter(|e| matches!(e, SimEvent::Lost { at, reason, .. } if *at == r && *reason == LostReason::RouterStarved))
            .count();
        assert_eq!(router_starved, 5);
    }

    #[test]
    fn router_cycle_does_not_hang() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 3.0);
        let r1 = sim.add_router();
        let r2 = sim.add_router();
        sim.connect(g, r1);
        sim.connect(r1, r2);
        sim.connect(r2, r1);
        advance_secs(&mut sim, 0.1);
    }

    #[test]
    fn generator_overrun_emits_lost_events() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 10.0);
        let w = add_worker_secs(&mut sim, RED, 0.5);
        sim.connect(g, w);
        let events = advance_secs(&mut sim, 1.0);

        let lost_count = events
            .iter()
            .filter(|e| matches!(e, SimEvent::Lost { .. }))
            .count();
        assert!(lost_count >= 7, "expected ≥7 lost events, got {lost_count}");
        assert_eq!(sim.nodes[&g].dropped, 0);
        let worker_dropped = sim.nodes[&w].dropped;
        assert_eq!(lost_count as u32, worker_dropped);
    }

    #[test]
    fn bounded_queue_overrun_drops_packets() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 20.0);
        let q = sim.add_queue(RED, 3);
        sim.connect(g, q);
        let events = advance_secs(&mut sim, 1.0);
        let total_lost = events
            .iter()
            .filter(|e| matches!(e, SimEvent::Lost { .. }))
            .count();
        assert!(total_lost >= 15, "expected ≥15 Lost events, got {total_lost}");
    }

    #[test]
    fn worker_forwards_to_sink() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 2.0);
        let w = add_worker_secs(&mut sim, RED, 0.1);
        let s = sim.add_sink(RED);
        sim.connect(g, w);
        sim.connect(w, s);
        advance_secs(&mut sim, 1.0);
        let total = sim.nodes[&s].sink_total;
        assert!(total >= 1, "sink received nothing (got {total})");
    }

    #[test]
    fn worker_chain_through_queue() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 4.0);
        let w1 = add_worker_secs(&mut sim, RED, 0.1);
        let q = sim.add_queue(RED, usize::MAX);
        let w2 = add_worker_secs(&mut sim, RED, 0.1);
        let s = sim.add_sink(RED);
        sim.connect(g, w1);
        sim.connect(w1, q);
        sim.connect(q, w2);
        sim.connect(w2, s);
        advance_secs(&mut sim, 2.0);
        let total = sim.nodes[&s].sink_total;
        assert!(total >= 5, "expected ≥5 through the chain, got {total}");
    }

    #[test]
    fn three_colors_independent_routing() {
        let mut sim = Sim::new();
        let g_r = add_gen(&mut sim, RED, 2.0);
        let g_y = add_gen(&mut sim, YELLOW, 2.0);
        let g_b = add_gen(&mut sim, BLUE, 2.0);
        let r = sim.add_router();
        sim.connect(g_r, r);
        sim.connect(g_y, r);
        sim.connect(g_b, r);
        let s_r = sim.add_sink(RED);
        let s_y = sim.add_sink(YELLOW);
        let s_b = sim.add_sink(BLUE);
        sim.connect(r, s_r);
        sim.connect(r, s_y);
        sim.connect(r, s_b);
        advance_secs(&mut sim, 1.0);
        assert_eq!(sink_total(&sim, s_r), 2);
        assert_eq!(sink_total(&sim, s_y), 2);
        assert_eq!(sink_total(&sim, s_b), 2);
    }

    #[test]
    fn nanosecond_scale_generator() {
        let mut sim = Sim::new();
        let g = sim.add_generator(RED, 1);
        let w = sim.add_worker(RED, 1);
        let s = sim.add_sink(RED);
        sim.connect(g, w);
        sim.connect(w, s);
        sim.advance_ns(1_000);
        let total = sink_total(&sim, s);
        assert!(total >= 400, "expected ≥400 at ns scale, got {total}");
    }

    #[test]
    fn analytical_rates_gen_to_sink() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 5.0);
        let s = sim.add_sink(RED);
        let e = sim.connect(g, s);
        let rates = sim.analytical_edge_rates();
        assert!((rates[&e] - 5.0).abs() < 1e-6, "got {}", rates[&e]);
    }

    #[test]
    fn analytical_rates_bottleneck_at_worker() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 10.0);
        let q = sim.add_queue(RED, usize::MAX);
        let w = add_worker_secs(&mut sim, RED, 0.5);
        let s = sim.add_sink(RED);
        let e_gq = sim.connect(g, q);
        let e_qw = sim.connect(q, w);
        let e_ws = sim.connect(w, s);
        let rates = sim.analytical_edge_rates();
        assert!((rates[&e_gq] - 10.0).abs() < 1e-6);
        assert!((rates[&e_qw] - 2.0).abs() < 1e-6);
        assert!((rates[&e_ws] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn analytical_rates_router_splits_equally() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 10.0);
        let r = sim.add_router();
        let w1 = add_worker_secs(&mut sim, RED, 0.5);
        let w2 = add_worker_secs(&mut sim, RED, 0.5);
        let s = sim.add_sink(RED);
        sim.connect(g, r);
        let e_r1 = sim.connect(r, w1);
        let e_r2 = sim.connect(r, w2);
        sim.connect(w1, s);
        sim.connect(w2, s);
        let rates = sim.analytical_edge_rates();
        assert!((rates[&e_r1] - 2.0).abs() < 1e-6, "got {}", rates[&e_r1]);
        assert!((rates[&e_r2] - 2.0).abs() < 1e-6, "got {}", rates[&e_r2]);
    }

    #[test]
    fn analytical_rates_queue_redistributes_to_fast_sibling() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 9.0);
        let q = sim.add_queue(RED, usize::MAX);
        let w_slow = add_worker_secs(&mut sim, RED, 0.5);
        let w_fast = add_worker_secs(&mut sim, RED, 0.1);
        let s = sim.add_sink(RED);
        let e_gq = sim.connect(g, q);
        let e_qs = sim.connect(q, w_slow);
        let e_qf = sim.connect(q, w_fast);
        sim.connect(w_slow, s);
        sim.connect(w_fast, s);
        let rates = sim.analytical_edge_rates();
        assert!((rates[&e_gq] - 9.0).abs() < 1e-6);
        assert!((rates[&e_qs] - 2.0).abs() < 1e-6);
        assert!((rates[&e_qf] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn water_fill_redistributes_overflow() {
        assert_eq!(water_fill(9.0, &[2.0, 10.0]), vec![2.0, 7.0]);
        assert_eq!(water_fill(20.0, &[2.0, 10.0]), vec![2.0, 10.0]);
        assert_eq!(water_fill(4.0, &[10.0, 10.0]), vec![2.0, 2.0]);
        assert_eq!(water_fill(0.0, &[5.0, 5.0]), vec![0.0, 0.0]);
        assert_eq!(water_fill(5.0, &[]), Vec::<f64>::new());
        let r = water_fill(10.0, &[f64::INFINITY, 2.0]);
        assert!((r[0] - 8.0).abs() < 1e-9 && (r[1] - 2.0).abs() < 1e-9, "{r:?}");
    }

    #[test]
    fn analytical_rates_color_aware_router() {
        let mut sim = Sim::new();
        let g_red = add_gen(&mut sim, RED, 2.0);
        let g_yel = add_gen(&mut sim, YELLOW, 10.0);
        let r = sim.add_router();
        sim.connect(g_red, r);
        sim.connect(g_yel, r);

        let q_red = sim.add_queue(RED, usize::MAX);
        let q_yel = sim.add_queue(YELLOW, usize::MAX);
        let e_rqr = sim.connect(r, q_red);
        let e_rqy = sim.connect(r, q_yel);

        let w = add_worker_secs(&mut sim, YELLOW, 0.1);
        let s = sim.add_sink(YELLOW);
        let e_qw = sim.connect(q_yel, w);
        sim.connect(w, s);

        let rates = sim.analytical_edge_rates();
        assert!((rates[&e_rqr] - 2.0).abs() < 0.5);
        assert!((rates[&e_rqy] - 10.0).abs() < 0.5);
        assert!((rates[&e_qw] - 10.0).abs() < 0.5);
    }

    #[test]
    fn slow_worker_drops_at_worker_not_generator() {
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 10.0);
        let w = add_worker_secs(&mut sim, RED, 0.5);
        sim.connect(g, w);
        advance_secs(&mut sim, 1.0);

        assert_eq!(sim.nodes[&g].dropped, 0);
        assert!(sim.nodes[&g].emitted >= 10);
        assert!(sim.nodes[&w].dropped >= 7);
    }

    #[test]
    fn parse_duration_examples() {
        assert_eq!(parse_duration_ns("400us"), Some(400_000));
        assert_eq!(parse_duration_ns("1.5ms"), Some(1_500_000));
        assert_eq!(parse_duration_ns("3s"), Some(3_000_000_000));
        assert_eq!(parse_duration_ns("100ns"), Some(100));
        assert_eq!(parse_duration_ns("100"), Some(100));
        assert_eq!(parse_duration_ns("2 sec"), Some(2_000_000_000));
        assert_eq!(parse_duration_ns("5µs"), Some(5_000));
        assert_eq!(parse_duration_ns("nope"), None);
        assert_eq!(parse_duration_ns("1pc"), None);
        assert_eq!(parse_duration_ns("-3ms"), None);
    }

    #[test]
    fn parse_rate_examples() {
        assert!((parse_rate_pps("10").unwrap() - 10.0).abs() < 1e-9);
        assert!((parse_rate_pps("10/s").unwrap() - 10.0).abs() < 1e-9);
        assert!((parse_rate_pps("1.2k").unwrap() - 1200.0).abs() < 1e-6);
        assert!((parse_rate_pps("1.2M/s").unwrap() - 1.2e6).abs() < 1e-3);
        assert!((parse_rate_pps("1.2Gpps").unwrap() - 1.2e9).abs() < 1e-3);
        assert!((parse_rate_pps("1/ms").unwrap() - 1e3).abs() < 1e-9);
        assert!((parse_rate_pps("1/us").unwrap() - 1e6).abs() < 1e-3);
        assert!((parse_rate_pps("1/ns").unwrap() - 1e9).abs() < 1e-3);
        assert_eq!(parse_rate_pps("junk"), None);
    }

    #[test]
    fn client_worker_request_response_roundtrip() {
        let mut sim = Sim::new();
        let c = sim.add_client(RED, rate_to_period_ns(2.0));
        let w = add_worker_secs(&mut sim, RED, 0.1);
        let e = sim.connect(c, w);
        let events = advance_secs(&mut sim, 2.1);

        let client = &sim.nodes[&c];
        assert_eq!(client.sent, 4);
        assert_eq!(client.received, 4);
        let rtt_avg = if client.rtt_count > 0 {
            client.rtt_sum_ns / client.rtt_count as u64
        } else {
            0
        };
        assert_eq!(rtt_avg, 100 * NS_PER_MS);

        let fwd = events.iter().filter(|ev| matches!(ev, SimEvent::Traveled { edge, is_response: false, .. } if *edge == e)).count();
        let bwd = events.iter().filter(|ev| matches!(ev, SimEvent::Traveled { edge, is_response: true, .. } if *edge == e)).count();
        let processed = events.iter().filter(|ev| matches!(ev, SimEvent::Processed { .. })).count();
        let response_events = events.iter().filter(|ev| matches!(ev, SimEvent::ResponseReceived { .. })).count();
        assert_eq!(fwd, 4);
        assert_eq!(bwd, 4);
        assert_eq!(processed, 4);
        assert_eq!(response_events, 4);

        let worker = &sim.nodes[&w];
        assert!(worker.holding.is_none());
        assert_eq!(worker.dropped, 0);
        assert_eq!(worker.processed, 4);
    }

    #[test]
    fn mixed_ns_and_ms_scales_coexist() {
        let mut sim = Sim::new();
        let g_ns = sim.add_generator(RED, 1_000);
        let s_ns = sim.add_sink(RED);
        sim.connect(g_ns, s_ns);

        let g_ms = sim.add_generator(YELLOW, NS_PER_MS);
        let s_ms = sim.add_sink(YELLOW);
        sim.connect(g_ms, s_ms);

        sim.advance_ns(10 * NS_PER_MS);
        assert_eq!(sink_total(&sim, s_ns), 10_000);
        assert_eq!(sink_total(&sim, s_ms), 10);
    }

    #[test]
    fn grouping_worker_and_sink_preserves_throughput() {
        // Gen → Worker → Sink, group Worker+Sink into a composite.
        // Observable behavior: the Sink's sink_total (still addressable
        // on the outer sim, since members live there) must match the
        // un-grouped baseline within ±1.
        let mut sim = Sim::new();
        let g = sim.add_generator(RED, rate_to_period_ns(2.0));
        let w = add_worker_secs(&mut sim, RED, 0.1);
        let s = sim.add_sink(RED);
        sim.connect(g, w);
        sim.connect(w, s);
        let mut baseline = sim.clone();
        advance_secs(&mut baseline, 2.0);
        let baseline_sunk = baseline.nodes[&s].sink_total;

        let mut selected: HashSet<NodeId> = HashSet::new();
        selected.insert(w);
        selected.insert(s);
        sim.group_into_composite(&selected, "Pipeline", RED).unwrap();
        advance_secs(&mut sim, 2.0);

        let sunk = sim.nodes[&s].sink_total;
        assert!(
            sunk >= baseline_sunk.saturating_sub(1) && sunk <= baseline_sunk + 1,
            "grouped sink {} vs baseline {}",
            sunk,
            baseline_sunk
        );
    }

    #[test]
    fn grouping_gen_gen_router_all_flows_through() {
        // Two generators and a router, all grouped. Their flow must
        // still reach downstream sinks unchanged: the composite adds no
        // new sim behavior beyond routing its external-in edges into
        // the members it contains. Gens are self-sourced (no external
        // in) so they fire as before.
        let mut sim = Sim::new();
        let g1 = add_gen(&mut sim, RED, 5.0);
        let g2 = add_gen(&mut sim, YELLOW, 5.0);
        let r = sim.add_router();
        let s_r = sim.add_sink(RED);
        let s_y = sim.add_sink(YELLOW);
        sim.connect(g1, r);
        sim.connect(g2, r);
        sim.connect(r, s_r);
        sim.connect(r, s_y);

        let mut selected: HashSet<NodeId> = HashSet::new();
        selected.insert(g1);
        selected.insert(g2);
        selected.insert(r);
        sim.group_into_composite(&selected, "Sources", RED).unwrap();

        advance_secs(&mut sim, 1.0);
        assert_eq!(sim.nodes[&s_r].sink_total, 5);
        assert_eq!(sim.nodes[&s_y].sink_total, 5);
    }

    #[test]
    fn grouping_queue_worker_probe_stays_visible() {
        // Gen → Queue → Worker → Sink. Group {Queue, Worker}. The
        // queue's buffer depth, in/out stats, and the worker's
        // processed count must all still be directly readable on the
        // outer sim (because members live in the outer sim after
        // grouping — no hidden inner sim).
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 4.0);
        let q = sim.add_queue(RED, usize::MAX);
        let w = add_worker_secs(&mut sim, RED, 0.5);
        let s = sim.add_sink(RED);
        sim.connect(g, q);
        sim.connect(q, w);
        sim.connect(w, s);

        let mut sel: HashSet<NodeId> = HashSet::new();
        sel.insert(q);
        sel.insert(w);
        sim.group_into_composite(&sel, "QWpair", RED).unwrap();

        advance_secs(&mut sim, 2.0);
        // Probing Q: its stats are still on the outer sim node.
        let queue_total_in = sim.nodes[&q].total_in;
        let queue_total_out = sim.nodes[&q].total_out;
        let worker_processed = sim.nodes[&w].processed;
        let sink_total = sim.nodes[&s].sink_total;
        assert!(
            queue_total_in >= 7,
            "queue total_in too low: {queue_total_in}"
        );
        assert!(
            worker_processed >= 3,
            "worker processed too low: {worker_processed}"
        );
        assert_eq!(
            queue_total_out, worker_processed + sim.nodes[&w].holding.is_some() as u32,
            "queue out == worker in-flight+processed"
        );
        assert!(sink_total >= 3, "sink too low: {sink_total}");
    }

    #[test]
    fn grouping_client_queue_worker_acks_immediately() {
        // Client → Queue → Worker, group Queue+Worker. Queue still
        // acks immediately; Client sees fast responses despite slow
        // Worker.
        let mut sim = Sim::new();
        let c = sim.add_client(RED, rate_to_period_ns(2.0));
        let q = sim.add_queue(RED, usize::MAX);
        let w = add_worker_secs(&mut sim, RED, 0.5);
        sim.connect(c, q);
        sim.connect(q, w);
        let mut sel: HashSet<NodeId> = HashSet::new();
        sel.insert(q);
        sel.insert(w);
        sim.group_into_composite(&sel, "QW", RED).unwrap();

        advance_secs(&mut sim, 1.0);
        let client = &sim.nodes[&c];
        assert_eq!(client.sent, 2);
        assert_eq!(client.received, 2, "queue should ack both immediately");
    }

    #[test]
    fn pull_edge_drains_queue_to_worker() {
        // Gen → Queue [push], Queue → Worker [PULL]. With a pull
        // edge, the queue does NOT push to the worker; the worker
        // initiates. Worker is idle, queue has packets, so on each
        // tick the worker pulls one and starts processing.
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 4.0);
        let q = sim.add_queue(RED, usize::MAX);
        let w = add_worker_secs(&mut sim, RED, 0.5);
        sim.connect(g, q);
        sim.connect_pull(q, w); // pull edge
        advance_secs(&mut sim, 2.0);
        let processed = sim.nodes[&w].processed;
        let depth = sim.nodes[&q].buffer.len() as u32;
        let in_worker = sim.nodes[&w].holding.is_some() as u32;
        assert_eq!(processed + depth + in_worker, 8);
        assert!(processed >= 3, "worker processed only {processed}");
    }

    #[test]
    fn pull_edge_waits_when_worker_busy() {
        // Gen → Queue [push], Queue → Worker [PULL]. Worker is slow;
        // queue should back up because the pull-worker only yanks
        // when idle.
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 10.0);
        let q = sim.add_queue(RED, usize::MAX);
        let w = add_worker_secs(&mut sim, RED, 1.0);
        sim.connect(g, q);
        sim.connect_pull(q, w);
        advance_secs(&mut sim, 2.0);
        let processed = sim.nodes[&w].processed;
        let depth = sim.nodes[&q].buffer.len() as u32;
        assert!(processed <= 2, "expected ≤2 processed, got {processed}");
        assert!(depth >= 17, "expected queue to back up, depth {depth}");
    }

    #[test]
    fn pull_edge_ignores_generator_push() {
        // Gen → Worker with a PULL edge. Generator fires but the
        // pull-only edge means the generator's BlindRoundRobin does
        // NOT push across it — the edge is dead to pushes. Worker,
        // having no buffer to pull from, gets nothing. Gen drops
        // every emission as "no push outbound."
        let mut sim = Sim::new();
        let g = add_gen(&mut sim, RED, 5.0);
        let w = add_worker_secs(&mut sim, RED, 0.1);
        sim.connect_pull(g, w);
        advance_secs(&mut sim, 1.0);
        assert_eq!(sim.nodes[&w].processed, 0);
        assert!(sim.nodes[&g].dropped >= 4);
    }

    #[test]
    fn steps_client_row_roundtrips_and_loops() {
        // Steps: [Client RED] → Worker. Auto-loops after the Client
        // row completes: each loop issues a new request.
        let mut sim = Sim::new();
        let steps = sim.add_steps(
            RED,
            vec![StepRow::Client { color: RED }],
        );
        let w = add_worker_secs(&mut sim, RED, 0.05);
        sim.connect_from_row(steps, w, 0);
        advance_secs(&mut sim, 0.5);
        let n = &sim.nodes[&steps];
        assert!(n.sent >= 5, "expected ≥5 roundtrips at 50ms each, got {}", n.sent);
        assert!(n.sent - n.received <= 1, "too many in-flight: {}/{}", n.received, n.sent);
    }

    #[test]
    fn steps_worker_row_waits_sim_time() {
        // Steps: [Worker 200ms, Client RED]. First Client fire only
        // after the Worker row's dwell has elapsed.
        let mut sim = Sim::new();
        let steps = sim.add_steps(
            RED,
            vec![
                StepRow::Worker {
                    duration_ns: 200 * NS_PER_MS,
                    color: RED,
                },
                StepRow::Client { color: RED },
            ],
        );
        let w = add_worker_secs(&mut sim, RED, 0.01);
        sim.connect_from_row(steps, w, 1);
        advance_secs(&mut sim, 0.1);
        assert_eq!(sim.nodes[&steps].sent, 0, "worker row should still be holding");
        advance_secs(&mut sim, 0.2);
        assert!(sim.nodes[&steps].sent >= 1, "expected ≥1 send after worker row");
    }

    #[test]
    fn steps_empty_script_is_dormant() {
        let mut sim = Sim::new();
        let steps = sim.add_steps(RED, vec![]);
        advance_secs(&mut sim, 1.0);
        let n = &sim.nodes[&steps];
        assert_eq!(n.sent, 0);
        assert_eq!(n.current_row, None);
    }

    #[test]
    fn steps_auto_loop_emits_event() {
        let mut sim = Sim::new();
        let steps = sim.add_steps(
            RED,
            vec![StepRow::Worker {
                duration_ns: 50 * NS_PER_MS,
                color: RED,
            }],
        );
        let events = advance_secs(&mut sim, 0.3);
        let loops = events
            .iter()
            .filter(|e| matches!(e, SimEvent::StepsLooped { node } if *node == steps))
            .count();
        // ~6 loops at 50ms each in 300ms.
        assert!(loops >= 4, "expected ≥4 loop events, got {loops}");
    }

    #[test]
    fn queue_immediately_acks_client_request() {
        // A Client sends Requests into a Queue whose step list includes
        // RespondImmediate. The client should see the response
        // immediately (RTT ≈ edge travel = 0ns in this model) — not
        // wait for a worker downstream to finish. The Worker pulling
        // from the queue does NOT send a second response: the queue's
        // RespondImmediate consumed the reply-address.
        let mut sim = Sim::new();
        let c = sim.add_client(RED, rate_to_period_ns(2.0));
        let q = sim.add_queue(RED, usize::MAX);
        let w = add_worker_secs(&mut sim, RED, 0.5); // slow
        sim.connect(c, q);
        sim.connect(q, w);
        advance_secs(&mut sim, 1.0);
        let client = &sim.nodes[&c];
        // At 2 req/s, over 1s we send 2. Queue acks each immediately.
        assert_eq!(client.sent, 2);
        assert_eq!(client.received, 2, "queue should ack both immediately");
    }
}
