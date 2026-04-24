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
    ensure_gadget_classes_registered(sim);
    let class = gadget_class_name(kind);
    let id = sim
        .instantiate(class, name)
        .unwrap_or_else(|e| panic!("instantiate gadget `{}`: {}", class, e));
    if !matches!(kind, Kind::Router) {
        if let Some(node) = sim.nodes.get_mut(&id) {
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
    if sim.templates.contains_key("Generator") {
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
);
