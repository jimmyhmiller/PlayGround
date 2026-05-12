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
use flow::{Expr, NodeId, Sim, Value};

/// True if this node's class declares a `color` slot. Drives whether a
/// node participates in data-palette tagging: gadgets that opt in (by
/// declaring `color: Int = 0` in their DSL) get a slot value written
/// here and a `NodeColors` entry; gadgets that don't (Router, LifeCell,
/// etc.) stay untagged and let their own visual control the look.
pub fn has_color_slot(sim: &Sim, id: NodeId) -> bool {
    sim.nodes
        .get(&id)
        .map_or(false, |n| n.slots.contains_key("color"))
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

/// Spawn a gadget of the given kind, writing the user-chosen palette
/// slot index into its `color` slot iff the class's DSL declares one.
/// Gadgets without a `color` slot (Router, LifeCell, etc.) stay
/// untagged — they're either matchers (Router filters *by* colour, so
/// pinning one to a colour would just narrow what it routes) or they
/// own their own visual (LifeCell paints itself from `alive`).
pub fn spawn(sim: &mut Sim, kind: Kind, name: &str, slot: usize) -> NodeId {
    ensure_gadget_classes_registered(sim);
    let class = gadget_class_name(kind);
    let id = sim
        .instantiate(class, name)
        .unwrap_or_else(|e| panic!("instantiate gadget `{}`: {}", class, e));
    if let Some(node) = sim.nodes.get_mut(&id) {
        if node.slots.contains_key("color") {
            node.slots.insert("color".into(), Value::Int(slot as i64));
        }
    }
    id
}

fn gadget_class_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Generator => "Generator",
        Kind::Client => "Client",
        Kind::BackoffClient => "BackoffClient",
        Kind::Worker => "Worker",
        Kind::Router => "Router",
        Kind::Queue => "Queue",
        Kind::Sink => "Sink",
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
    include_str!("gadgets/generator.flow"),      "\n",
    include_str!("gadgets/client.flow"),         "\n",
    include_str!("gadgets/backoff_client.flow"), "\n",
    include_str!("gadgets/worker.flow"),         "\n",
    include_str!("gadgets/router.flow"),         "\n",
    include_str!("gadgets/queue.flow"),          "\n",
    include_str!("gadgets/sink.flow"),           "\n",
    // Experimental gadgets — registered as classes so whiteboard files
    // can `node Foo : Cache { ... }` them. Not yet in the Kind enum;
    // unknown classes default to a Worker-shaped node visually.
    include_str!("gadgets/cache.flow"),           "\n",
    include_str!("gadgets/circuit_breaker.flow"), "\n",
    include_str!("gadgets/saga.flow"),            "\n",
    include_str!("gadgets/tpc.flow"),             "\n",
    // Composable primitives. Each is a tiny single-purpose gadget
    // intended to be wired into bigger gadgets visually. Together
    // they cover everything the higher-level gadgets above can do.
    include_str!("gadgets/primitives/tick.flow"),      "\n",
    include_str!("gadgets/primitives/counter.flow"),   "\n",
    include_str!("gadgets/primitives/filter.flow"),    "\n",
    include_str!("gadgets/primitives/switch.flow"),    "\n",
    include_str!("gadgets/primitives/threshold.flow"), "\n",
    include_str!("gadgets/primitives/window.flow"),    "\n",
    include_str!("gadgets/primitives/delay.flow"),     "\n",
    include_str!("gadgets/primitives/stamp.flow"),     "\n",
    include_str!("gadgets/primitives/unstamp.flow"),   "\n",
    include_str!("gadgets/primitives/fanout.flow"),    "\n",
    include_str!("gadgets/primitives/coin.flow"),      "\n",
    include_str!("gadgets/primitives/buffer.flow"),    "\n",
    // Variant adapters and aggregate primitives — the additional set
    // that lets us build every higher-level gadget (Worker, Client,
    // Cache, CircuitBreaker, Saga, TPC, Game of Life) as a compound
    // of primitives. See `gadgets/composite/*.flow` for the recipes.
    include_str!("gadgets/primitives/lift.flow"),      "\n",
    include_str!("gadgets/primitives/lower.flow"),     "\n",
    include_str!("gadgets/primitives/reply.flow"),     "\n",
    include_str!("gadgets/primitives/service.flow"),   "\n",
    include_str!("gadgets/primitives/aggregator.flow"),"\n",
    include_str!("gadgets/primitives/egress.flow"),    "\n",
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
