//! Gadgets — prebuilt Flow node shapes that map to the original
//! bevy-whiteboard kinds. Each function returns a new Flow `NodeId`
//! in the supplied `Sim`.
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

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Generator => "Gen",
            Kind::Client => "Client",
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
            Kind::Worker => "▣",
            Kind::Router => "⊕",
            Kind::Queue => "▦",
            Kind::Sink => "▽",
        }
    }

    pub fn size(self) -> Vec2 {
        match self {
            Kind::Generator | Kind::Client | Kind::Sink => Vec2::new(60.0, 60.0),
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
        Kind::Generator => gen_generator(sim, name, slot),
        Kind::Client    => gen_client(sim, name, slot),
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
        }
        rule tick {
            on tick(_)
            do {
                emitted := emitted + 1
                record emitted emitted
                emit packet(color) to default
                emit tick(nil)     to self
            }
        }
    }

    node Client {
        slots {
            in_flight: Int = 0
            sent_at:   Samples(1024)
            completed: Int = 0
            color:     Int = 0
            period_ns: Int = 500000000
        }
        rule fire {
            on tick(_)
            do {
                in_flight := in_flight + 1
                push sent_at <- now
                emit req(color) to default
                emit tick(nil)  to self
            }
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
    }

    node Worker {
        slots {
            served:     Int = 0
            color:      Int = 0
            service_ns: Int = 50000000
            busy:       Int = 0
            upstream:   Any = nil
            downstream: Any = nil
        }
        rule serve {
            on req(_)
            do {
                served := served + 1
                record served served
                respond resp(nil)
            }
        }
        rule start_service {
            on packet(p)
            when busy == 0
            do {
                busy := 1
                emit done(p) to self
            }
        }
        rule done {
            on done(p)
            do {
                busy   := 0
                served := served + 1
                record served served
                emit packet(p)   to (downstream)
                emit pull(self)  to (upstream)
            }
        }
    }

    node Router {
        rule forward {
            on packet(p)
            do {
                emit_each packet(p) to filter(out_neighbors(), "n",
                    slot_of(n, "color") == p)
            }
        }
        rule forward_req {
            on req(p)
            do {
                emit_each req(p) to filter(out_neighbors(), "n",
                    slot_of(n, "color") == p)
            }
        }
    }

    node Queue {
        slots {
            len:     Int = 0
            color:   Int = 0
            waiting: Samples(256)
        }
        rule enqueue {
            on packet(_)
            do { len := len + 1 }
        }
        rule wake_tick_flush {
            on wake(_)
            when len > 0 && len(waiting) > 0
            do {
                pop waiting -> consumer
                len := len - 1
                emit packet(color) to (consumer)
                emit wake(nil)     to self
            }
        }
        rule wake_tick_idle {
            on wake(_)
            when len == 0 || len(waiting) == 0
            do {
                emit wake(nil) to self
            }
        }
        rule on_pull {
            on pull(consumer)
            when len > 0
            do {
                len := len - 1
                emit packet(color) to (consumer)
            }
        }
        rule on_pull_stash {
            on pull(consumer)
            when len == 0
            do {
                push waiting <- consumer
            }
        }
    }

    node Sink {
        slots {
            count: Int = 0
            color: Int = 0
        }
        rule absorb {
            on _
            do {
                count := count + 1
                record absorbed count
            }
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
    sim.inject(id, Value::variant("tick", Value::Nil), None);
    id
}

/// Client: like Generator but sends request/response pairs and tracks RTT.
/// Re-emits on response.
fn gen_client(sim: &mut Sim, name: &str, slot: usize) -> NodeId {
    let (mut slots, rules) = template("Client");
    slots.insert("color".into(), Value::Int(slot as i64));
    let id = sim.add_node(name, slots, rules);
    sim.add_edge(id, id, Expr::slot("period_ns"));
    sim.inject(id, Value::variant("tick", Value::Nil), None);
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
    sim.inject(id, Value::variant("wake", Value::Nil), None);
    id
}

/// Sink: absorbs anything, counts it, never emits.
fn gen_sink(sim: &mut Sim, name: &str, slot: usize) -> NodeId {
    let (mut slots, rules) = template("Sink");
    slots.insert("color".into(), Value::Int(slot as i64));
    sim.add_node(name, slots, rules)
}
