//! Gadgets — prebuilt Flow node shapes. Each function returns a new
//! Flow `NodeId` in the supplied `Sim`.
//!
//! Design: a gadget is just a set of slots + rules that emulate the
//! classic kind. The user can later edit the slots (and eventually the
//! rules) — the gadget is just the starting point.
//!
//! Every gadget's **full behaviour** — slots, rules, self-edges, and
//! any bootstrap inbox packets — is authored as DSL in `GADGETS_DSL`.
//! The DSL's `on_spawn { }` block carries the self-edges and initial
//! `inject`s so the class-based instantiate path produces a working
//! instance with no Rust glue. The only per-instance override still
//! done in Rust is `color` (the data-palette slot index), because
//! that value is a runtime arg the DSL class doesn't know.
//!
//! Live params (`install_default_params`) still live in Rust since the
//! DSL's `params { }` block runs at load time in a throwaway sim.

use std::sync::Once;

use bevy::prelude::*;
use flow::{Expr, NodeId, Sim};

/// True if this node's class declares a `color` slot. Drives whether a
/// node participates in data-palette tagging: gadgets that opt in (by
/// declaring `color: Int = 0` in their DSL) get a slot value written
/// here and a `NodeColors` entry; gadgets that don't (Router, LifeCell,
/// etc.) stay untagged and let their own visual control the look.
pub fn has_color_slot(sim: &Sim, id: NodeId) -> bool {
    // Direct hit: legacy monolithic gadgets had `color` on the node.
    if let Some(n) = sim.nodes.get(&id) {
        if n.slots.contains_key("color") { return true; }
    }
    // Composite: check the spawning compound class's params for `color`.
    // The slot itself may live as `match` on an inner Filter (Router /
    // Sink), not as `color` on an inner node — so a slot-name walk
    // isn't enough; we want the *class declared a color parameter*
    // signal.
    if let Some(class_name) = sim.compound_class_of.get(&id) {
        if let Some(c) = sim.compound_templates.get(class_name) {
            return c.params.iter().any(|p| p.name == "color");
        }
    }
    false
}

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
    /// A load-balanced, self-scaling worker fleet. Spawns and destroys
    /// `ScalableWorker` members at runtime based on fleet load. Drill in
    /// to see the load balancer fanning out to the live workers.
    AutoScalingGroup,
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
            Kind::AutoScalingGroup => "ASG",
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
            Kind::AutoScalingGroup => 'a',
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
            Kind::AutoScalingGroup => "⊞",
        }
    }

    /// ASCII fallback glyph for the bitmap-label atlas (which only
    /// pre-rasterises printable ASCII). One uppercase initial.
    pub fn glyph_ascii(self) -> char {
        match self {
            Kind::Generator => 'G',
            Kind::Client => 'C',
            Kind::BackoffClient => 'B',
            Kind::Worker => 'W',
            Kind::Router => 'R',
            Kind::Queue => 'Q',
            Kind::Sink => 'S',
            Kind::AutoScalingGroup => 'A',
        }
    }

    /// Port on which this kind's compound emits *forward* data
    /// (requests, packet pass-through). `None` for terminal kinds.
    /// `wire_flow_edge` uses this as the `from_port` on the user-
    /// drawn forward edge.
    pub fn forward_output_port(self) -> Option<&'static str> {
        match self {
            Kind::Generator | Kind::Client | Kind::BackoffClient
            | Kind::Router | Kind::Queue | Kind::Worker
            | Kind::AutoScalingGroup => Some("output"),
            Kind::Sink => None,
        }
    }

    /// Port on which this kind's compound receives *forward* data.
    /// `None` for sources that don't take input (Generator) and for
    /// pure originators (Client / BackoffClient — those only consume
    /// replies, not fresh requests).
    pub fn forward_input_port(self) -> Option<&'static str> {
        match self {
            Kind::Worker | Kind::AutoScalingGroup => Some("request"),
            Kind::Router | Kind::Queue | Kind::Sink => Some("input"),
            Kind::Generator | Kind::Client | Kind::BackoffClient => None,
        }
    }

    /// Port on which this kind's compound emits *reverse* data (resp,
    /// resp_error) — used to attach the auto-wired reverse edge in
    /// `wire_flow_edge` when both endpoints participate in reply
    /// traffic. `None` for kinds that never reply.
    pub fn reverse_output_port(self) -> Option<&'static str> {
        match self {
            Kind::Worker | Kind::Router | Kind::Queue => Some("response"),
            Kind::Generator | Kind::Client | Kind::BackoffClient | Kind::Sink
            | Kind::AutoScalingGroup => None,
        }
    }

    /// Port on which this kind's compound receives *reverse* data
    /// (replies travelling back along the user-drawn edge). `None`
    /// for kinds that never observe replies (Generator, Sink).
    pub fn reverse_input_port(self) -> Option<&'static str> {
        match self {
            Kind::Client | Kind::BackoffClient | Kind::Router
            | Kind::Queue | Kind::Worker => Some("reply"),
            Kind::Generator | Kind::Sink | Kind::AutoScalingGroup => None,
        }
    }

    pub fn size(self) -> Vec2 {
        match self {
            Kind::Generator | Kind::Client | Kind::BackoffClient | Kind::Sink => Vec2::new(60.0, 60.0),
            Kind::Worker | Kind::Router => Vec2::new(70.0, 50.0),
            Kind::Queue => Vec2::new(90.0, 40.0),
            // Larger box — it visually contains a fleet.
            Kind::AutoScalingGroup => Vec2::new(110.0, 70.0),
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

/// Spawn a gadget of the given kind. For monolithic classes (current
/// default), the palette slot index is written into the spawned node's
/// `color` slot iff the class declares one. For compound classes (when
/// `gadget_class_name` returns a `*Composite`), the slot is threaded
/// through as a compile-time `color` param override at expansion time.
pub fn spawn(sim: &mut Sim, kind: Kind, name: &str, slot: usize) -> NodeId {
    ensure_gadget_classes_registered(sim);
    let class = gadget_class_name(kind);
    if sim.compound_templates.contains_key(class) {
        let mut overrides = std::collections::BTreeMap::new();
        if compound_has_color_param(sim, class) {
            overrides.insert(
                "color".to_string(),
                flow::dsl::expand::CtValue::Int(slot as i64),
            );
        }
        return sim
            .instantiate_compound(class, name, &overrides)
            .unwrap_or_else(|e| panic!("instantiate compound gadget `{}`: {}", class, e));
    }
    let id = sim
        .instantiate(class, name)
        .unwrap_or_else(|e| panic!("instantiate gadget `{}`: {}", class, e));
    if let Some(node) = sim.nodes.get_mut(&id) {
        if node.slots.contains_key("color") {
            node.slots.insert("color".into(), flow::Value::Int(slot as i64));
        }
    }
    id
}

/// True if the named compound class declares a `color` compile-time
/// param. Used by `spawn` to decide whether to thread the palette
/// slot through as an override. Cheap: just inspects the stored AST.
fn compound_has_color_param(sim: &Sim, class: &str) -> bool {
    sim.compound_templates
        .get(class)
        .map(|c| c.params.iter().any(|p| p.name == "color"))
        .unwrap_or(false)
}

/// Same mapping as `gadget_class_name`, exposed so other modules (e.g.
/// `edges::wire_flow_edge`) can check `sim.compound_templates.contains_key(...)`
/// before tagging an edge with compound ports.
pub fn compound_class_name_for(kind: Kind) -> &'static str {
    gadget_class_name(kind)
}

fn gadget_class_name(kind: Kind) -> &'static str {
    // Palette spawns the primitive-built composite classes. Tests that
    // pattern-match against monolithic node shapes use
    // `Sim::compound_outermost` (event log matching) and
    // `Sim::read_slot_resolved` / `write_slot_resolved` (slot access)
    // to walk into the compound's children.
    match kind {
        Kind::Generator     => "GeneratorComposite",
        Kind::Client        => "ClientComposite",
        Kind::BackoffClient => "BackoffClientComposite",
        Kind::Worker        => "WorkerComposite",
        Kind::Router        => "RouterComposite",
        Kind::Queue         => "QueueComposite",
        Kind::Sink          => "SinkComposite",
        // A leaf controller class (not a `*Composite`): it spawns the
        // real composites at runtime rather than containing them.
        Kind::AutoScalingGroup => "AutoScalingGroup",
    }
}

/// Register the gadget DSL classes on this sim the first time a gadget
/// is spawned into it. Keyed per-sim via the sim's `templates` map
/// (no-op if already populated).
fn ensure_gadget_classes_registered(sim: &mut Sim) {
    // Any one class name is enough as a presence check — they're
    // registered as a batch.
    if sim.has_class("Generator") {
        return;
    }
    flow::dsl::register_classes(sim, GADGETS_DSL)
        .expect("gadget DSL must compile");
    install_back_compat_aliases(sim);
}

/// Register the seven palette-kind aliases (`Generator`, `Client`, …,
/// `Sink`) so authored whiteboards and tests that write
/// `node x : Worker { color: 0 }` keep working after the monolithic
/// .flow files were deleted. Each alias just copies the corresponding
/// `*Composite` entry from `compound_templates`. Skip silently if a
/// composite of that base name doesn't exist (e.g. Cache/Saga/TPC have
/// no monolithic alias; they were always named `*Composite`).
pub fn install_back_compat_aliases(sim: &mut Sim) {
    const ALIASES: &[(&str, &str)] = &[
        ("Generator",      "GeneratorComposite"),
        ("Client",         "ClientComposite"),
        ("BackoffClient",  "BackoffClientComposite"),
        ("Worker",         "WorkerComposite"),
        ("Router",         "RouterComposite"),
        ("Queue",          "QueueComposite"),
        ("Sink",           "SinkComposite"),
        // Experimental gadgets that used to ship as monolithic .flow
        // files. The composite versions cover the same role.
        ("Cache",          "CacheComposite"),
        ("CircuitBreaker", "CircuitBreakerComposite"),
        ("Saga",           "SagaStepComposite"),
        ("TPCCoordinator", "TPCCoordinatorComposite"),
        ("TPCParticipant", "TPCParticipantComposite"),
    ];
    for (alias, source) in ALIASES {
        if sim.has_class(alias) { continue; }
        let Some(decl) = sim.compound_templates.get(*source).cloned() else { continue };
        sim.compound_templates.insert(alias.to_string(), decl);
    }
}

/// One-shot validation: parse the gadget DSL at startup so a syntax
/// error surfaces immediately instead of on first gadget spawn.
pub fn validate_gadget_dsl() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let mut sim = Sim::new(0);
        flow::dsl::register_classes(&mut sim, GADGETS_DSL)
            .expect("gadget DSL must compile");
    });
}

// ============================================================================
// DSL-authored gadget templates
// ============================================================================
//
// Each gadget's slots, rules, self-edges, and bootstrap injections live in
// its own `.flow` file under `src/gadgets/`. They're pulled in via
// `include_str!` and concatenated at compile time into a single DSL source
// that `register_classes` consumes.

pub const GADGETS_DSL: &str = concat!(
    // Composable primitives — the irreducible set (18 primitives).
    // Every higher-level gadget (Worker, Client, Cache, CircuitBreaker,
    // Saga, TPC, Game of Life) is expressed as a compound of these in
    // `gadgets/composite/*.flow`. The three return-path-aware
    // primitives (Stamp, Reply, Pump) replace the old hand-written
    // `L`/`R`/`Acc` adapter nodes inside composites. Tally is a
    // counter whose slot is named `len` rather than `count`.
    include_str!("gadgets/primitives/tick.flow"),      "\n",
    include_str!("gadgets/primitives/counter.flow"),   "\n",
    include_str!("gadgets/primitives/filter.flow"),    "\n",
    include_str!("gadgets/primitives/switch.flow"),    "\n",
    include_str!("gadgets/primitives/threshold.flow"), "\n",
    include_str!("gadgets/primitives/window.flow"),    "\n",
    include_str!("gadgets/primitives/delay.flow"),     "\n",
    include_str!("gadgets/primitives/fanout.flow"),    "\n",
    include_str!("gadgets/primitives/coin.flow"),      "\n",
    include_str!("gadgets/primitives/buffer.flow"),    "\n",
    include_str!("gadgets/primitives/service.flow"),   "\n",
    include_str!("gadgets/primitives/aggregator.flow"),"\n",
    include_str!("gadgets/primitives/egress.flow"),    "\n",
    // Unified Constant — replaces ConstantPacket + ConstantSignal. The
    // `out_kind` slot picks which envelope kind the packet ships with
    // ("packet" by default, "signal" for control-plane pulses).
    include_str!("gadgets/primitives/constant.flow"), "\n",
    // Return-path-aware primitives. Stamp pushes/pops self; Reply
    // turns a serviced packet into a `resp` back to head(rp); Pump
    // re-emits `pull(nil)` to upstream after each forward.
    include_str!("gadgets/primitives/stamp.flow"),     "\n",
    include_str!("gadgets/primitives/reply.flow"),     "\n",
    include_str!("gadgets/primitives/pump.flow"),      "\n",
    // Pass-through counter exposing a `len` slot (vs Counter's
    // `count`) — for composites whose surface state is a "depth".
    include_str!("gadgets/primitives/tally.flow"),     "\n",
    // Composites — each higher-level gadget redefined as a compound of
    // primitives. Registered as classes so whiteboards (and the palette)
    // can spawn them with `sim.instantiate("WorkerComposite", "w1")`.
    include_str!("gadgets/composite/sink.flow"),            "\n",
    include_str!("gadgets/composite/generator.flow"),       "\n",
    include_str!("gadgets/composite/client.flow"),          "\n",
    include_str!("gadgets/composite/backoff_client.flow"),  "\n",
    include_str!("gadgets/composite/worker.flow"),          "\n",
    include_str!("gadgets/composite/queue.flow"),           "\n",
    include_str!("gadgets/composite/router.flow"),          "\n",
    include_str!("gadgets/composite/cache.flow"),           "\n",
    include_str!("gadgets/composite/circuit_breaker.flow"), "\n",
    include_str!("gadgets/composite/saga.flow"),            "\n",
    include_str!("gadgets/composite/tpc.flow"),             "\n",
    include_str!("gadgets/composite/life.flow"),            "\n",
    // Autoscaling: a single top-level leaf that load-balances across and
    // grows/shrinks a fleet of workers it clones from its `template`
    // slot (default `WorkerComposite`). Pluggable — it treats whatever
    // it spawns as a black-box request/response gadget.
    include_str!("gadgets/composite/auto_scaling_group.flow"), "\n",
);

// ============================================================================
// Composite gadgets — every gadget redefined as a compound of primitives.
// ============================================================================
//
// These are *not* in GADGETS_DSL because the compound machinery isn't a
// registerable class (yet). Each constant is meant to be concatenated
// into a DSL scene that uses the composite as a singleton compound
// instance. The test suite under `tests/composite_gadgets.rs` exercises
// each one against behavioural fixtures.

pub const SINK_COMPOSITE: &str            = include_str!("gadgets/composite/sink.flow");
pub const GENERATOR_COMPOSITE: &str       = include_str!("gadgets/composite/generator.flow");
pub const CLIENT_COMPOSITE: &str          = include_str!("gadgets/composite/client.flow");
pub const BACKOFF_CLIENT_COMPOSITE: &str  = include_str!("gadgets/composite/backoff_client.flow");
pub const WORKER_COMPOSITE: &str          = include_str!("gadgets/composite/worker.flow");
pub const QUEUE_COMPOSITE: &str           = include_str!("gadgets/composite/queue.flow");
pub const ROUTER_COMPOSITE: &str          = include_str!("gadgets/composite/router.flow");
pub const CACHE_COMPOSITE: &str           = include_str!("gadgets/composite/cache.flow");
pub const CIRCUIT_BREAKER_COMPOSITE: &str = include_str!("gadgets/composite/circuit_breaker.flow");
pub const SAGA_COMPOSITE: &str            = include_str!("gadgets/composite/saga.flow");
pub const TPC_COMPOSITE: &str             = include_str!("gadgets/composite/tpc.flow");
pub const LIFE_CELL_COMPOSITE: &str       = include_str!("gadgets/composite/life.flow");
