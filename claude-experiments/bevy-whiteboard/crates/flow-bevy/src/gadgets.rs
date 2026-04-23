//! Gadgets — prebuilt Flow node shapes. Each function returns a new
//! Flow `NodeId` in the supplied `Sim`.
//!
//! Design: a gadget is just a set of slots + rules that emulate the
//! classic kind. The user can later edit the slots (and eventually the
//! rules) — the gadget is just the starting point.
//!
//! Every gadget's slot+rule shape is authored as a DSL string (see
//! `GADGETS_DSL`) compiled once at startup. The `gen_*` helpers clone
//! the pre-lowered template, override per-instance slots (e.g. `color`),
//! add the node, then wire edges and initial injections in Rust — that
//! second step is per-instance glue the template can't own.
//!
//! Live params (`install_default_params`) still live in Rust since the
//! DSL's `params { }` block runs at load time in a throwaway sim.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use bevy::prelude::*;
use flow::{Expr, NodeId, Rule, Sim, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Kind {
    #[default]
    Generator,
    Client,
    BackoffClient,
    Worker,
    Router,
    Queue,
    Sink,
}

/// What a kind exposes to external observers (probes, the inspector,
/// anything that wants to read node state without hard-coding slot names).
/// The reader closure is `fn(&flow::Node) -> String` — simple fn pointer so
/// it's `Copy + 'static` and can live directly inside a component.
///
/// A new kind declares a slice of specs; the probe UI iterates the slice
/// and spawns one readout per entry. Adding a new readable without
/// touching probes.rs is the whole point of this separation.
#[derive(Clone, Copy)]
pub struct ProbeSpec {
    /// Short uppercase-able label shown beside the value (e.g. "RATE",
    /// "FILL"). Kept to 1-2 words so it fits the canvas annotation.
    pub label: &'static str,
    /// Read the current value off a sim node and format it. Called every
    /// frame while a probe for this spec is mounted — keep it cheap.
    pub read: fn(&flow::Node) -> String,
}

/// Every kind's probe-readable stats. Ordered by "how often you want to
/// see this" — the Probe tool stacks them top-down above the node in
/// this order, so the first spec ends up closest to the node body.
pub fn probes_for_kind(kind: Kind) -> &'static [ProbeSpec] {
    match kind {
        Kind::Generator | Kind::Client => &[
            ProbeSpec { label: "rate",    read: read_rate },
            ProbeSpec { label: "emitted", read: read_emitted },
        ],
        Kind::BackoffClient => &[
            ProbeSpec { label: "rate",    read: read_rate },
            ProbeSpec { label: "emitted", read: read_emitted },
            ProbeSpec { label: "backoff", read: read_backoff_ms },
        ],
        Kind::Worker => &[
            ProbeSpec { label: "served",  read: read_served },
            ProbeSpec { label: "service", read: read_service_ms },
        ],
        Kind::Queue => &[
            ProbeSpec { label: "fill",    read: read_queue_len },
            ProbeSpec { label: "waiting", read: read_queue_waiter },
        ],
        Kind::Sink => &[
            ProbeSpec { label: "total", read: read_sink_count },
        ],
        Kind::Router => &[
            // Router has no state of its own, so no default specs. The
            // empty slice means the Probe tool silently skips routers.
        ],
    }
}

// ---- stock readers. New kinds add their own or reuse these. ------------

fn read_rate(node: &flow::Node) -> String {
    match node.slots.get("period_ns") {
        Some(Value::Int(p)) if *p > 0 => {
            let r = 1_000_000_000.0 / *p as f64;
            if r >= 10.0 { format!("{:.0}/s", r) } else { format!("{:.1}/s", r) }
        }
        _ => "—".into(),
    }
}

fn read_emitted(node: &flow::Node) -> String {
    match node.slots.get("emitted") {
        Some(Value::Int(i)) => format!("{}", i),
        _ => "—".into(),
    }
}

fn read_served(node: &flow::Node) -> String {
    match node.slots.get("served") {
        Some(Value::Int(i)) => format!("{}", i),
        _ => "—".into(),
    }
}

fn read_service_ms(node: &flow::Node) -> String {
    match node.slots.get("service_ns") {
        Some(Value::Int(ns)) => format!("{}ms", ns / 1_000_000),
        _ => "—".into(),
    }
}

fn read_queue_len(node: &flow::Node) -> String {
    match node.slots.get("len") {
        Some(Value::Int(i)) => format!("{}", i),
        _ => "—".into(),
    }
}

fn read_queue_waiter(node: &flow::Node) -> String {
    match node.slots.get("waiting") {
        Some(Value::Samples(s)) => format!("{}", s.len()),
        _ => "0".into(),
    }
}

fn read_sink_count(node: &flow::Node) -> String {
    match node.slots.get("count") {
        Some(Value::Int(i)) => format!("{}", i),
        _ => "—".into(),
    }
}

fn read_backoff_ms(node: &flow::Node) -> String {
    match node.slots.get("backoff_ns") {
        Some(Value::Int(ns)) => {
            if *ns == 0 { "—".into() }
            else if *ns >= 1_000_000_000 { format!("{:.1}s", *ns as f64 / 1e9) }
            else { format!("{}ms", ns / 1_000_000) }
        }
        _ => "—".into(),
    }
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Generator => "Gen",
            Kind::Client => "Client",
            Kind::BackoffClient => "BackoffClient",
            Kind::Worker => "Worker",
            Kind::Router => "Router",
            Kind::Queue => "Queue",
            Kind::Sink => "Sink",
        }
    }

    pub fn hotkey(self) -> char {
        match self {
            Kind::Generator => 'g',
            Kind::Client => 'c',
            Kind::BackoffClient => 'b',
            Kind::Worker => 'w',
            Kind::Router => 'r',
            Kind::Queue => 'q',
            Kind::Sink => 's',
        }
    }

    pub fn color(self, theme: &crate::theme::Theme) -> Color {
        crate::theme::kind_color(theme, self)
    }

    /// Unicode glyph that approximates the kind's iconic shape in palette
    /// buttons. Geometric-only (no emoji) so they render in line with the
    /// primary typeface.
    pub fn glyph(self) -> &'static str {
        match self {
            Kind::Generator => "◉",
            Kind::Client => "☻",
            Kind::BackoffClient => "↻",
            Kind::Worker => "▣",
            Kind::Router => "⊕",
            Kind::Queue => "▦",
            Kind::Sink => "▽",
        }
    }

    pub fn size(self) -> Vec2 {
        match self {
            Kind::Generator | Kind::Client | Kind::BackoffClient | Kind::Sink => Vec2::new(60.0, 60.0),
            Kind::Worker | Kind::Router => Vec2::new(70.0, 50.0),
            Kind::Queue => Vec2::new(90.0, 40.0),
        }
    }
}

/// Install common live params once into a fresh Sim. Called on sim init.
pub fn install_default_params(sim: &mut Sim) {
    sim.set_param("network",      Expr::int(1_000_000));          // 1 ms
    sim.set_param("service_mean", Expr::int(25_000_000));         // 25 ms
    sim.set_param("emit_period",  Expr::int(100_000_000));        // 100 ms
    sim.set_param("client_timeout", Expr::int(2_000_000_000));    // 2 s
    sim.set_param("queue_cap",    Expr::int(64));
}

/// Spawn a gadget of the given kind tagged with a data-palette slot index.
/// Every kind except [`Kind::Router`] stores `color = Int(slot)` in its sim
/// slots so the router can filter by it. Routers are left untagged because
/// they're the thing *doing* the matching — pinning them to a colour would
/// just make them filter on it.
pub fn spawn(sim: &mut Sim, kind: Kind, name: &str, slot: usize) -> NodeId {
    match kind {
        Kind::Generator     => gen_generator(sim, name, slot),
        Kind::Client        => gen_client(sim, name, slot),
        Kind::BackoffClient => gen_backoff_client(sim, name, slot),
        Kind::Worker    => gen_worker(sim, name, slot),
        Kind::Router    => gen_router(sim, name),
        Kind::Queue     => gen_queue(sim, name, slot),
        Kind::Sink      => gen_sink(sim, name, slot),
    }
}

// ============================================================================
// DSL-authored gadget templates
// ============================================================================
//
// The slot+rule shape of every gadget kind is written as plain DSL, compiled
// once, and stashed as `(slots, rules)` pairs keyed by node name. Per-instance
// setup (slot overrides, edge wiring, initial injections) is done in Rust
// below — that's the only stuff that can't be expressed in a generic
// template.

const GADGETS_DSL: &str = r#"
    node Generator {
        slots {
            emitted:   Int = 0
            color:     Int = 0
            period_ns: Int = 500000000
            # 1 = up (emitting), 0 = down (self-tick keeps ticking so the
            # loop can resume when up flips back, but packets are skipped).
            up:        Int = 1
        }
        rule tick {
            on tick(_)
            when up == 1
            do {
                emitted := emitted + 1
                record emitted emitted
                emit packet(color) to default
                emit tick(nil)     to self
            }
        }
        rule tick_idle {
            on tick(_)
            when up == 0
            do { emit tick(nil) to self }
        }
    }

    node Client {
        slots {
            emitted:   Int = 0
            in_flight: Int = 0
            failed:    Int = 0
            sent_at:   Samples(1024)
            completed: Int = 0
            color:     Int = 0
            period_ns: Int = 500000000
            up:        Int = 1
        }
        rule fire {
            on tick(_)
            when up == 1
            do {
                emitted   := emitted + 1
                in_flight := in_flight + 1
                push sent_at <- now
                # Push self onto return_path so the response walks back —
                # preserved automatically across intermediate hops (routers,
                # queues) that don't touch return_path.
                emit req(color) pushing self to default
                emit tick(nil)  to self
            }
        }
        rule fire_idle {
            on tick(_)
            when up == 0
            do { emit tick(nil) to self }
        }
        rule on_resp {
            on resp(_)
            do {
                in_flight := in_flight - 1
                completed := completed + 1
                pop sent_at -> t
                record rtt_ns now - t
            }
        }
        # A downstream down-node replies with resp_error instead of
        # resp. Balance in_flight / sent_at just like a success, but
        # count toward `failed` and record a request_failed error so
        # the failure is visible on the HUD.
        rule on_resp_error {
            on resp_error(_)
            do {
                in_flight := in_flight - 1
                failed    := failed + 1
                pop sent_at -> t
                error "request_failed" "client: request failed (downstream down)"
            }
        }
    }

    # Like Client, but delays its next tick with fixed exponential backoff
    # on resp_error and resets the delay on resp. Self-edge latency is
    # period_ns + backoff_ns, so backoff_ns = 0 recovers plain Client
    # timing. The doubling is deterministic (no jitter) on purpose — this
    # is the shape that produces a thundering herd when many clients
    # share an outage and come back into sync.
    node BackoffClient {
        slots {
            emitted:        Int = 0
            in_flight:      Int = 0
            failed:         Int = 0
            sent_at:        Samples(1024)
            completed:      Int = 0
            color:          Int = 0
            period_ns:      Int = 500000000
            backoff_ns:     Int = 0
            max_backoff_ns: Int = 8000000000
            up:             Int = 1
        }
        rule fire {
            on tick(_)
            when up == 1
            do {
                emitted   := emitted + 1
                in_flight := in_flight + 1
                push sent_at <- now
                emit req(color) pushing self to default
                emit tick(nil)  to self
            }
        }
        rule fire_idle {
            on tick(_)
            when up == 0
            do { emit tick(nil) to self }
        }
        # Success clears the backoff window so the next tick fires at
        # the base period again.
        rule on_resp {
            on resp(_)
            do {
                in_flight  := in_flight - 1
                completed  := completed + 1
                backoff_ns := 0
                pop sent_at -> t
                record rtt_ns now - t
            }
        }
        # First failure after a success: seed backoff to one period.
        rule on_resp_error_seed {
            on resp_error(_)
            when backoff_ns == 0
            do {
                in_flight  := in_flight - 1
                failed     := failed + 1
                backoff_ns := period_ns
                pop sent_at -> t
                error "request_failed" "client: request failed (downstream down)"
            }
        }
        # Subsequent failure, still under cap: double.
        rule on_resp_error_double {
            on resp_error(_)
            when backoff_ns > 0 && backoff_ns * 2 <= max_backoff_ns
            do {
                in_flight  := in_flight - 1
                failed     := failed + 1
                backoff_ns := backoff_ns * 2
                pop sent_at -> t
                error "request_failed" "client: request failed (downstream down)"
            }
        }
        # Doubling would exceed the cap: clamp to max_backoff_ns.
        rule on_resp_error_cap {
            on resp_error(_)
            when backoff_ns > 0 && backoff_ns * 2 > max_backoff_ns
            do {
                in_flight  := in_flight - 1
                failed     := failed + 1
                backoff_ns := max_backoff_ns
                pop sent_at -> t
                error "request_failed" "client: request failed (downstream down)"
            }
        }
    }

    node Worker {
        slots {
            served:      Int = 0
            color:       Int = 0
            service_ns:  Int = 50000000
            busy:        Int = 0
            # A small FIFO of queued requests, modeled like a naive
            # `listen(backlog)` socket. Each slot stores the req's
            # `return_path` as a List(NodeRef). `backlog_cap` is the
            # listen backlog (how many reqs the OS would hold for us
            # before dropping new connections); the Samples ring
            # itself is sized generously above that so push never
            # silently evicts — the `reject_full` rule is the only
            # way a req gets rejected for capacity reasons.
            backlog:     Samples(16)
            backlog_cap: Int = 5
            up:          Int = 1
            upstream:    Any = nil
            downstream:  Any = nil
        }
        # Rule ORDER is load-bearing on the Worker: completion rules
        # (finish_req_*) come BEFORE acceptance rules (serve /
        # enqueue) so a `done_req` arriving in the same sim tick as
        # a new `req` advances the queue first, letting the fresh
        # req be admitted to a freshly-drained slot rather than
        # spuriously bounced.
        #
        # Service time is the self-loop edge latency: `serve` sets
        # `busy := 1` and emits a `done_req` to self; when it
        # arrives, `finish_req_*` dispatches the reply and — if a
        # queued req was waiting — emits the NEXT `done_req` using
        # that queued req's stored return_path. The self-loop is
        # the only place the service delay lives, so the worker
        # reads as single-threaded (one service in flight) with a
        # bounded accept queue behind it, matching a naive Python
        # HTTP server's shape.

        # Something was queued: finish this one, pop the head of
        # the backlog, and kick off its service. `busy` stays 1
        # because we're immediately starting the next job — no
        # idle window between queued reqs.
        rule finish_req_next {
            on done_req(_)
            when up == 1 && len(backlog) > 0
            do {
                served := served + 1
                record served served
                pop backlog -> next_path
                emit done_req(nil) return_path next_path to self
                emit resp(nil) popping to (head(return_path))
            }
        }
        # Nothing queued: finish and go idle.
        rule finish_req_idle {
            on done_req(_)
            when up == 1 && len(backlog) == 0
            do {
                busy := 0
                served := served + 1
                record served served
                emit resp(nil) popping to (head(return_path))
            }
        }
        # Worker went down during the service window: the in-flight
        # reply becomes an error, same shape the client sees for any
        # other node-down failure. `busy` clears so the worker is a
        # fresh machine when it comes back up. Queued entries stay
        # in `backlog`; `resume_work` drains them once up flips 1.
        rule finish_req_crashed {
            on done_req(_)
            when up == 0
            do {
                busy := 0
                error "node_down" "worker: crashed mid-req, reply lost"
                emit resp_error(nil) popping to (head(return_path))
            }
        }
        # Serve: strict on colour, accept only when fully idle (no
        # in-flight service AND empty backlog so we don't overtake
        # queued reqs that were stranded by a prior crash).
        rule serve {
            on req(c)
            when c == color && up == 1 && busy == 0 && len(backlog) == 0
            do {
                busy := 1
                emit done_req(nil) to self
            }
        }
        # Already servicing and the backlog has room: stash this
        # req's return_path so `finish_req_next` can dispatch it
        # when its turn comes up.
        rule enqueue {
            on req(c)
            when c == color && up == 1 && len(backlog) < backlog_cap
            do {
                push backlog <- return_path
            }
        }
        # Accept queue is full — reject with resp_error so the
        # client can back off. "worker_full" is the error kind
        # that surfaces in the panel.
        rule reject_full {
            on req(c)
            when c == color && up == 1 && len(backlog) >= backlog_cap
            do {
                error "worker_full" "worker: backlog full, rejecting request"
                emit resp_error(nil) popping to (head(return_path))
            }
        }
        # Down: any inbound req is immediately failed back to the
        # client via `resp_error`, and a `node_down` error is logged.
        # Takes precedence over colour-mismatch rejection because the
        # down state is the more actionable signal for the operator.
        rule serve_down {
            on req(_)
            when up == 0
            do {
                error "node_down" "worker: request to down node"
                emit resp_error(nil) popping to (head(return_path))
            }
        }
        rule serve_reject {
            on req(_)
            do { error "color_mismatch" "worker: wrong-colour req rejected" }
        }
        rule start_service {
            on packet(p)
            when busy == 0 && p == color && up == 1
            do {
                busy := 1
                emit done(p) to self
            }
        }
        # Crash model: while down, every inbound packet is dropped on
        # the floor with a `node_down` error, regardless of `busy`.
        # No busy==0 gate here — if the worker crashed mid-service,
        # packets that pile up behind the stuck `busy` flag still get
        # rejected rather than queuing forever.
        rule start_service_down {
            on packet(_)
            when up == 0
            do { error "node_down" "worker: packet to down node" }
        }
        rule start_service_reject {
            on packet(_)
            when busy == 0
            do { error "color_mismatch" "worker: wrong-colour packet rejected" }
        }
        # `done` completes the in-flight packet — fires only when up.
        rule done {
            on done(p)
            when up == 1
            do {
                busy   := 0
                served := served + 1
                record served served
                emit packet(p)   to (downstream)
                emit pull(self)  to (upstream)
            }
        }
        # Crash: the `done(p)` self-packet that was scheduled before
        # the crash arrives and gets consumed + discarded. Clears
        # `busy` so when the worker comes back up it's a fresh machine
        # (no resumed work, no leaked downstream emission). The
        # packet the worker was servicing is simply lost — that's
        # what a real crash looks like.
        rule done_crashed {
            on done(_)
            when up == 0
            do {
                busy := 0
                error "node_down" "worker: crashed mid-service, packet lost"
            }
        }
        # Resume hook: the UI injects `resume(nil)` when `up` flips
        # 0→1. Drain any queued reqs first (reqs stashed during the
        # outage are still valid — their clients are still waiting
        # for a reply), then kick the pull loop. We pop ONE queued
        # req here; subsequent queued reqs are picked up by
        # `finish_req_next` as each service completes.
        rule resume_drain {
            on resume(_)
            when up == 1 && busy == 0 && len(backlog) > 0
            do {
                busy := 1
                pop backlog -> next_path
                emit done_req(nil) return_path next_path to self
            }
        }
        rule resume_pull {
            on resume(_)
            when up == 1 && busy == 0
            do { emit pull(self) to (upstream) }
        }
        # Resume arrived while the worker is still busy (a service
        # was in flight but hasn't timed out yet) or still down.
        # Consume silently — the normal done/done_crashed path will
        # handle the in-flight packet, and a subsequent resume on
        # the next up toggle will re-kick the loop if needed.
        rule resume_noop {
            on resume(_)
            do { }
        }
    }

    node Router {
        slots {
            up: Int = 1
        }
        # Fan-out by colour. `packet` is fire-and-forget: no reply is
        # expected, so return_path passes through untouched.
        rule forward {
            on packet(p)
            when up == 1
            do {
                emit_each packet(p) to filter(out_neighbors(), "n",
                    slot_of(n, "color") == p)
            }
        }
        rule forward_down {
            on packet(_)
            when up == 0
            do { error "node_down" "router: packet to down router" }
        }
        # Request forwarding: the Router opts into the reply path by
        # pushing itself onto return_path. The downstream Worker pops
        # back to us (not to the original Client); we then relay the
        # response one more hop via `forward_resp`. This makes
        # Client→Router→Worker a proper two-edge req/resp chain with
        # no triangle reply edge.
        rule forward_req {
            on req(p)
            when up == 1
            do {
                emit_each req(p) pushing self to filter(out_neighbors(), "n",
                    slot_of(n, "color") == p)
            }
        }
        rule forward_req_down {
            on req(_)
            when up == 0
            do {
                error "node_down" "router: request to down router"
                emit resp_error(nil) popping to (head(return_path))
            }
        }
        # Relay a response back along the chain. Pops ourselves off
        # the head we pushed on forward, emits to whoever's now at
        # head (the next hop back toward the client). Works for any
        # depth of nested routers because each one pushes on req and
        # pops on resp. Stays active even when the router is down
        # so an in-flight response from a still-up downstream doesn't
        # get stranded.
        rule forward_resp {
            on resp(x)
            do {
                emit resp(x) popping to (head(return_path))
            }
        }
        rule forward_resp_error {
            on resp_error(x)
            do {
                emit resp_error(x) popping to (head(return_path))
            }
        }
    }

    node Queue {
        slots {
            len:     Int = 0
            color:   Int = 0
            up:      Int = 1
            waiting: Samples(256)
        }
        # Colour-strict on both inbound variants. Router-matched
        # traffic always reaches the right queue, but direct wiring
        # from a wrong-colour source surfaces a `color_mismatch` on
        # the error panel rather than silently piling up.
        rule enqueue {
            on packet(c)
            when c == color && up == 1
            do { len := len + 1 }
        }
        rule enqueue_down {
            on packet(_)
            when up == 0
            do { error "node_down" "queue: packet to down queue" }
        }
        rule enqueue_reject {
            on packet(_)
            do { error "color_mismatch" "queue: wrong-colour packet rejected" }
        }
        # Client-facing variant: `req(c)` is a request TO queue
        # something. Bump the depth counter AND ack the client by
        # popping its frame off return_path and emitting resp back.
        # The ack travels along the existing Client→Queue edge in
        # reverse (engine's reverse-route fallback).
        rule enqueue_req {
            on req(c)
            when c == color && up == 1
            do {
                len := len + 1
                emit resp(nil) popping to (head(return_path))
            }
        }
        rule enqueue_req_down {
            on req(_)
            when up == 0
            do {
                error "node_down" "queue: request to down queue"
                emit resp_error(nil) popping to (head(return_path))
            }
        }
        rule enqueue_req_reject {
            on req(_)
            do { error "color_mismatch" "queue: wrong-colour req rejected" }
        }
        # Wake-tick keeps ticking either way so the queue resumes
        # flushing the moment `up` flips back to 1. The flushing
        # branch is gated so a down queue never emits to a waiter.
        rule wake_tick_flush {
            on wake(_)
            when len > 0 && len(waiting) > 0 && up == 1
            do {
                pop waiting -> consumer
                len := len - 1
                emit packet(color) to (consumer)
                emit wake(nil)     to self
            }
        }
        rule wake_tick_idle {
            on wake(_)
            when len == 0 || len(waiting) == 0 || up == 0
            do {
                emit wake(nil) to self
            }
        }
        # Pulls arriving while down get stashed rather than served —
        # consumers resume naturally once `up` flips and `wake_tick_flush`
        # drains the `waiting` list.
        rule on_pull {
            on pull(consumer)
            when len > 0 && up == 1
            do {
                len := len - 1
                emit packet(color) to (consumer)
            }
        }
        rule on_pull_stash {
            on pull(consumer)
            when len == 0 || up == 0
            do {
                push waiting <- consumer
            }
        }
    }

    node Sink {
        slots {
            count: Int = 0
            color: Int = 0
            up:    Int = 1
        }
        # Colour-strict: only matching-colour `packet(c)` bumps the
        # counter. Wrong colours, or any other variant, are rejected
        # with a `color_mismatch` error.
        rule absorb {
            on packet(c)
            when c == color && up == 1
            do {
                count := count + 1
                record absorbed count
            }
        }
        rule absorb_down {
            on packet(_)
            when up == 0
            do { error "node_down" "sink: packet to down sink" }
        }
        rule absorb_reject_packet {
            on packet(_)
            do { error "color_mismatch" "sink: wrong-colour packet rejected" }
        }
    }
"#;

/// Map from template node name → `(slots, rules)` pre-lowered once.
fn gadget_templates() -> &'static HashMap<String, (BTreeMap<String, Value>, Vec<Rule>)> {
    static CACHE: OnceLock<HashMap<String, (BTreeMap<String, Value>, Vec<Rule>)>> =
        OnceLock::new();
    CACHE.get_or_init(|| {
        let sim = flow::dsl::load(GADGETS_DSL, 0)
            .expect("gadget DSL must compile");
        sim.nodes
            .into_values()
            .map(|n| (n.name.clone(), (n.slots, n.rules)))
            .collect()
    })
}

/// Fetch a fresh clone of a template by node name (as written in the DSL).
fn template(name: &str) -> (BTreeMap<String, Value>, Vec<Rule>) {
    gadget_templates()
        .get(name)
        .unwrap_or_else(|| panic!("no gadget template `{}`", name))
        .clone()
}

// ------------- kind implementations -------------

/// Generator: periodically emits `packet(color)` to any downstream.
/// Self-loop edge latency reads the node's own `period_ns` slot so each
/// generator ticks at its own rate. The `color` slot carries the data-
/// palette index, and emitted packets carry that value in their payload
/// so downstream routers can route by it.
fn gen_generator(sim: &mut Sim, name: &str, slot: usize) -> NodeId {
    let (mut slots, rules) = template("Generator");
    slots.insert("color".into(), Value::Int(slot as i64));
    let id = sim.add_node(name, slots, rules);
    // Self-loop carries the period. Latency reads `period_ns` off the
    // source node's slots, re-evaluated each emission — live rate
    // changes via slot writes propagate automatically.
    sim.add_edge(id, id, Expr::slot("period_ns"));
    // Inject the initial tick so the loop starts.
    sim.inject(id, Value::variant("tick", Value::Nil));
    id
}

/// Client: like Generator but sends request/response pairs and tracks RTT.
/// Re-emits on response.
fn gen_client(sim: &mut Sim, name: &str, slot: usize) -> NodeId {
    let (mut slots, rules) = template("Client");
    slots.insert("color".into(), Value::Int(slot as i64));
    let id = sim.add_node(name, slots, rules);
    sim.add_edge(id, id, Expr::slot("period_ns"));
    sim.inject(id, Value::variant("tick", Value::Nil));
    id
}

/// BackoffClient: Client variant whose self-tick latency is
/// `period_ns + backoff_ns`, so `resp_error` can extend the next tick
/// by doubling `backoff_ns` (capped at `max_backoff_ns`). On `resp`
/// the backoff resets to 0 and timing collapses back to the base period.
fn gen_backoff_client(sim: &mut Sim, name: &str, slot: usize) -> NodeId {
    let (mut slots, rules) = template("BackoffClient");
    slots.insert("color".into(), Value::Int(slot as i64));
    let id = sim.add_node(name, slots, rules);
    sim.add_edge(
        id,
        id,
        Expr::add(Expr::slot("period_ns"), Expr::slot("backoff_ns")),
    );
    sim.inject(id, Value::variant("tick", Value::Nil));
    id
}

/// Worker: serialised service of one request at a time. Pulls from
/// `upstream`, forwards to `downstream` — both slots are set by the
/// Connect handler at wire-up time.
fn gen_worker(sim: &mut Sim, name: &str, slot: usize) -> NodeId {
    let (mut slots, rules) = template("Worker");
    slots.insert("color".into(), Value::Int(slot as i64));
    let id = sim.add_node(name, slots, rules);
    // Self-loop latency is the service time — `done` deliveries use this
    // edge, so changing `service_ns` live alters the service window on
    // the next packet.
    sim.add_edge(id, id, Expr::slot("service_ns"));
    id
}

/// Router: color-matched fan-out via DSL. For each inbound `packet(c)`
/// or `req(c)`, emit to every outbound neighbour whose `color` slot
/// equals `c`. Packets whose colour doesn't match any neighbour are
/// silently dropped (empty target list). The router itself has no
/// colour.
fn gen_router(sim: &mut Sim, name: &str) -> NodeId {
    let (slots, rules) = template("Router");
    sim.add_node(name, slots, rules)
}

/// Queue: pull-source with a depth counter. We don't store the actual
/// packet payloads — a simulation of queue depth only needs the count.
/// On pull we emit `packet(color)` using the queue's own colour slot.
fn gen_queue(sim: &mut Sim, name: &str, slot: usize) -> NodeId {
    let (mut slots, rules) = template("Queue");
    slots.insert("color".into(), Value::Int(slot as i64));
    let id = sim.add_node(name, slots, rules);
    // Self-loop wake-tick at 1ms keeps the queue responsive.
    sim.add_edge(id, id, Expr::int(1_000_000));
    sim.inject(id, Value::variant("wake", Value::Nil));
    id
}

/// Sink: absorbs anything, counts it, never emits.
fn gen_sink(sim: &mut Sim, name: &str, slot: usize) -> NodeId {
    let (mut slots, rules) = template("Sink");
    slots.insert("color".into(), Value::Int(slot as i64));
    sim.add_node(name, slots, rules)
}
