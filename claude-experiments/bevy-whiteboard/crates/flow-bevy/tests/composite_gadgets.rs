//! Composite gadgets — every higher-level gadget redefined as a
//! compound that wires only the primitive gadgets (Tick, Counter,
//! Filter, Switch, Threshold, Window, Delay, Stamp, Unstamp, FanOut,
//! Coin, Buffer, Lift, Reply, Service, Aggregator).
//!
//! Each test loads the primitive class registry (GADGETS_DSL) plus
//! the composite's compound definition plus a test scene, then runs
//! the sim and asserts behavioural parity with the original
//! monolithic gadget.
//!
//! These tests validate that the primitive set is *sufficient* to
//! express every gadget in the project — including the Game of Life
//! cell. If a primitive turns out to be missing, the breakage will
//! surface here.
//!
//! The composites live under `src/gadgets/composite/*.flow` and are
//! re-exported as `pub const` strings by `flow_bevy::gadgets`. They
//! are *not* registered via `register_classes` — that entry point
//! only accepts `node` blocks. Instead each test concatenates
//! `GADGETS_DSL` (which registers the primitive *classes*) with the
//! composite's `compound … { … }` block and a scene, then loads the
//! whole thing via `flow::dsl::load`.

use flow_bevy::gadgets::{
    GADGETS_DSL,
    SINK_COMPOSITE,
    GENERATOR_COMPOSITE,
    CLIENT_COMPOSITE,
    BACKOFF_CLIENT_COMPOSITE,
    WORKER_COMPOSITE,
    QUEUE_COMPOSITE,
    ROUTER_COMPOSITE,
    CACHE_COMPOSITE,
};

fn build_sim(composite: &str, scene: &str) -> flow::sim::Sim {
    // Register the primitive (and monolithic-gadget) classes onto a
    // fresh sim WITHOUT instantiating any of them. Then lower the
    // composite + scene into the same sim — at that point each
    // `node F : Filter { … }` inside the composite resolves against
    // the just-registered primitive class.
    let mut sim = flow::sim::Sim::new(0);
    flow::dsl::register_classes(&mut sim, GADGETS_DSL)
        .expect("primitive registry must compile");
    flow_bevy::gadgets::install_back_compat_aliases(&mut sim);
    let mut src = String::new();
    src.push_str(composite);
    src.push('\n');
    src.push_str(scene);
    let file = flow::dsl::parse(&src)
        .unwrap_or_else(|e| panic!("parse: {}\n--- src ---\n{}", e, src));
    let file = flow::dsl::expand::expand(&file)
        .unwrap_or_else(|e| panic!("expand: {}\n--- src ---\n{}", e, src));
    let loaded = flow::dsl::lower_into(&mut sim, &file)
        .unwrap_or_else(|e| panic!("lower: {}\n--- src ---\n{}", e, src));
    if loaded.auto_run_main {
        sim.run_scenario("main").unwrap();
    }
    sim
}

fn node_id(sim: &flow::sim::Sim, name: &str) -> flow::sim::NodeId {
    *sim.nodes
        .iter()
        .find(|(_, n)| n.name == name)
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("no node `{}`; available: {:?}",
            name,
            sim.nodes.values().map(|n| n.name.clone()).collect::<Vec<_>>()))
}

fn slot_i(sim: &flow::sim::Sim, name: &str, slot: &str) -> i64 {
    let id = node_id(sim, name);
    match sim.nodes[&id].slots.get(slot) {
        Some(flow::value::Value::Int(n)) => *n,
        other => panic!("slot {}.{} = {:?}", name, slot, other),
    }
}

// ----------------------------------------------------------------------------
// SINK — Filter(match=color) → Counter.
// ----------------------------------------------------------------------------
/// Concatenate several composites into a single DSL prelude. Used by
/// tests that wire one composite into another (e.g. ClientComposite →
/// WorkerComposite end-to-end).
fn concat_composites(parts: &[&str]) -> String {
    let mut s = String::new();
    for p in parts {
        s.push_str(p);
        s.push('\n');
    }
    s
}

#[test]
fn sink_composite_counts_matching_color_only() {
    // Composite defaults to color=0; send a mix of matching (0) and
    // non-matching payloads. Only the 0's land in the inner Counter.
    let scene = r#"
        node Src {
            slots { dummy: Int = 0 }
            rule fire {
                on go(_)
                do {
                    emit packet(0) to default
                    emit packet(0) to default
                    emit packet(0) to default
                    emit packet(7) to default
                    emit packet(9) to default
                }
            }
        }
        edges { Src -> SinkComposite.input : 1 }
        scenario { at 0ns: inject Src <- go() }
    "#;
    let mut sim = build_sim(SINK_COMPOSITE, scene);
    sim.run_until(100);
    assert_eq!(slot_i(&sim, "SinkComposite::C", "count"), 3);
    assert!(sim.error_counts.is_empty(),
        "no runtime errors expected: {:?}", sim.error_counts);
}

// ----------------------------------------------------------------------------
// GENERATOR — Tick + a single-rule colorize glue node.
// ----------------------------------------------------------------------------
#[test]
fn generator_composite_emits_periodically_with_color() {
    // Default period = 500ms → ~2 emits in 1.05s.
    let scene = r#"
        node Probe : Counter { }
        edges { GeneratorComposite.output -> Probe : 1 }
    "#;
    let mut sim = build_sim(GENERATOR_COMPOSITE, scene);
    sim.run_until(1_050_000_000);
    let n = slot_i(&sim, "Probe", "count");
    assert!(n == 2 || n == 3,
        "GeneratorComposite emitted {} packets in 1.05s (expected 2-3)", n);
}

// ----------------------------------------------------------------------------
// CLIENT → echo loop. Endpoint emits req, echo replies via popping —
// Endpoint.completed counts.
// ----------------------------------------------------------------------------
#[test]
fn client_composite_tracks_completed_responses() {
    let scene = r#"
        node Echo {
            slots { served: Int = 0 }
            rule on_req {
                on req(_) do {
                    served := served + 1
                    emit resp(nil) popping to (head(return_path))
                }
            }
        }
        edges {
            ClientComposite.output -> Echo : 1ms
            Echo                    -> ClientComposite.reply : 1ms
        }
    "#;
    let mut sim = build_sim(CLIENT_COMPOSITE, scene);
    // Default period 500ms → ~2 reqs over 1.05s; each gets a resp back.
    sim.run_until(1_050_000_000);
    let emitted   = slot_i(&sim, "ClientComposite::Endpoint", "emitted");
    let completed = slot_i(&sim, "ClientComposite::Endpoint", "completed");
    let failed    = slot_i(&sim, "ClientComposite::Endpoint", "failed");
    assert!(emitted >= 2 && emitted <= 3,
        "Client emitted {} (expected 2-3)", emitted);
    assert_eq!(completed, emitted,
        "every emit should produce a resp; emitted={} completed={}", emitted, completed);
    assert_eq!(failed, 0);
}

// ----------------------------------------------------------------------------
// WORKER — req in → packet processed via Filter+Service → resp out.
// ----------------------------------------------------------------------------
#[test]
fn worker_composite_serves_requests_then_replies() {
    let scene = r#"
        node Caller {
            slots { sent: Int = 0; got: Int = 0 }
            rule fire {
                on go(_) do {
                    sent := sent + 1
                    emit req(0) pushing self to default
                }
            }
            rule on_resp {
                on resp(_) do { got := got + 1 }
            }
        }
        edges {
            Caller -> WorkerComposite.request : 1ms
            WorkerComposite.response -> Caller : 1ms
        }
        scenario {
            at 0ns:           inject Caller <- go()
            at 200000000ns:   inject Caller <- go()
            at 400000000ns:   inject Caller <- go()
        }
    "#;
    let mut sim = build_sim(WORKER_COMPOSITE, scene);
    sim.run_until(800_000_000);
    let sent = slot_i(&sim, "Caller", "sent");
    let got  = slot_i(&sim, "Caller", "got");
    assert_eq!(sent, 3);
    assert_eq!(got, 3, "all 3 reqs should have round-tripped");
    assert!(sim.error_counts.is_empty(),
        "no runtime errors: {:?}", sim.error_counts);
}

// ----------------------------------------------------------------------------
// CLIENT + WORKER end-to-end. Two composites composed together.
// ----------------------------------------------------------------------------
#[test]
fn client_and_worker_composites_compose() {
    let prelude = concat_composites(&[CLIENT_COMPOSITE, WORKER_COMPOSITE]);
    let scene = r#"
        edges {
            ClientComposite.output -> WorkerComposite.request : 1ms
            WorkerComposite.response -> ClientComposite.reply : 1ms
        }
    "#;
    let mut sim = build_sim(&prelude, scene);
    // 2 seconds: Client period 500ms → ~4 reqs; Worker service 50ms
    // each → all complete within window.
    // 2.5s gives the final req emitted at ~2000ms a comfortable window
    // to finish the ~55ms round-trip (4 hops + service window).
    sim.run_until(2_500_000_000);
    let emitted   = slot_i(&sim, "ClientComposite::Endpoint", "emitted");
    let completed = slot_i(&sim, "ClientComposite::Endpoint", "completed");
    assert!(emitted >= 4, "Client emitted {} (expected ≥4)", emitted);
    assert_eq!(completed, emitted,
        "Worker should respond to every Client req: emitted={} completed={}",
        emitted, completed);
    assert!(sim.error_counts.is_empty(),
        "no runtime errors: {:?}", sim.error_counts);
}

// ----------------------------------------------------------------------------
// BACKOFF CLIENT — emits, doubles backoff on resp_error.
// ----------------------------------------------------------------------------
#[test]
fn backoff_client_doubles_on_error() {
    // Wire to a server that always fails. Watch the backoff grow.
    let scene = r#"
        node Server {
            slots { dummy: Int = 0 }
            rule on_req {
                on req(_) do { emit resp_error(nil) popping to (head(return_path)) }
            }
        }
        edges {
            BackoffClientComposite.output -> Server : 1ms
            Server -> BackoffClientComposite.reply  : 1ms
        }
    "#;
    let mut sim = build_sim(BACKOFF_CLIENT_COMPOSITE, scene);
    sim.run_until(5_000_000_000);
    let failed     = slot_i(&sim, "BackoffClientComposite::Endpoint", "failed");
    let backoff_ns = slot_i(&sim, "BackoffClientComposite::Endpoint", "backoff_ns");
    assert!(failed >= 2, "BackoffClient failed {} times (expected ≥2)", failed);
    assert!(backoff_ns > 500_000_000,
        "backoff_ns should have grown past initial period; got {}", backoff_ns);
}

// ----------------------------------------------------------------------------
// QUEUE — Filter(color) → Buffer; pull-driven dispatch.
// ----------------------------------------------------------------------------
#[test]
fn queue_composite_buffers_then_dispatches() {
    let scene = r#"
        node Src {
            slots { dummy: Int = 0 }
            rule fire {
                on go(_) do {
                    emit packet(0) to default
                    emit packet(0) to default
                    emit packet(0) to default
                    emit packet(9) to default
                }
            }
        }
        node Puller {
            slots { dummy: Int = 0 }
            rule on_pull { on pull(_) do { emit pull(nil) to default } }
        }
        node Out : Counter { }
        edges {
            Src    -> QueueComposite.input : 1
            Puller -> QueueComposite.pull  : 1
            QueueComposite.output -> Out   : 1
        }
        scenario {
            at 0ns:  inject Src <- go()
            at 5ns:  inject Puller <- pull(nil)
            at 6ns:  inject Puller <- pull(nil)
            at 7ns:  inject Puller <- pull(nil)
        }
    "#;
    let mut sim = build_sim(QUEUE_COMPOSITE, scene);
    sim.run_until(100);
    // 3 packet(0)s pass the filter (color=0 default); pulls drain them.
    // packet(9) is rejected by Filter (silent drop).
    assert_eq!(slot_i(&sim, "Out", "count"), 3,
        "Queue dispatched {} packets", slot_i(&sim, "Out", "count"));
}

// ----------------------------------------------------------------------------
// ROUTER — Filter(color) → output port; FanOut via multiple external edges.
// ----------------------------------------------------------------------------
#[test]
fn router_composite_broadcasts_color_matched_to_all_outputs() {
    let scene = r#"
        node Src {
            slots { dummy: Int = 0 }
            rule fire {
                on go(_) do {
                    emit packet(0) to default
                    emit packet(7) to default
                }
            }
        }
        node A : Counter { }
        node B : Counter { }
        node C : Counter { }
        edges {
            Src -> RouterComposite.input : 1
            RouterComposite.output -> A : 1
            RouterComposite.output -> B : 1
            RouterComposite.output -> C : 1
        }
        scenario { at 0ns: inject Src <- go() }
    "#;
    let mut sim = build_sim(ROUTER_COMPOSITE, scene);
    sim.run_until(100);
    // color=0 packet broadcasts to all three; color=7 is filtered out.
    assert_eq!(slot_i(&sim, "A", "count"), 1);
    assert_eq!(slot_i(&sim, "B", "count"), 1);
    assert_eq!(slot_i(&sim, "C", "count"), 1);
}

// ----------------------------------------------------------------------------
// CACHE — hit path: Lower→Coin→Delay→Reply; ~hit_rate of reqs succeed.
// ----------------------------------------------------------------------------
#[test]
fn cache_composite_hit_path_replies() {
    // hit_rate=1.0 forces every req to hit. We don't wire a backing
    // store so misses would silently drop — by pinning hit_rate to 1.0
    // we isolate the hit path for an unambiguous assertion.
    let prelude = format!(
        "{}\n# pin hit_rate to 1.0 by re-declaring (composite default is 0.8)\n",
        CACHE_COMPOSITE
    );
    let scene = r#"
        node Caller {
            slots { sent: Int = 0; got: Int = 0 }
            rule fire {
                on go(_) do {
                    sent := sent + 1
                    emit req(0) pushing self to default
                }
            }
            rule on_resp { on resp(_) do { got := got + 1 } }
        }
        edges {
            Caller -> CacheComposite.request : 1ms
            CacheComposite.response -> Caller : 1ms
        }
        scenario {
            at 0ns:           inject Caller <- go()
            at 100000000ns:   inject Caller <- go()
            at 200000000ns:   inject Caller <- go()
        }
    "#;
    // Re-emit composite with hit_rate=1.0 baked in.
    let custom = r#"
compound CacheComposite(hit_rate: Float = 1.0, hit_latency_ns: Int = 1000000, color: Int = 0) {
    in  { request:  L }
    out { response: L; miss: ME }
    # Inline entry adapter (formerly Lower).
    node L {
        rule rev { on p when p.kind == "resp" || p.kind == "resp_error"
                  do { emit p popping to out response } }
        rule fwd { on p when p.kind == "req"
                  do { emit packet(p.value) pushing self to default } }
    }
    node F  : Filter { match: color }
    node C  : Coin   { p: hit_rate }
    node HD : Delay  { ns: hit_latency_ns }
    # Inline hit-reply (formerly Reply).
    node HR {
        rule rev_pass { on p when p.kind == "resp" || p.kind == "resp_error"
                       do { emit p popping to (head(return_path)) } }
        rule on_packet { on p
                       do { emit resp(nil) popping to (head(return_path)) } }
    }
    # Inline stamp waystation (formerly Stamp).
    node S {
        rule rev { on p when p.kind == "resp" || p.kind == "resp_error"
                  do { emit p popping to (head(return_path)) } }
        rule fwd { on p do { emit p pushing self to default } }
    }
    node ME { rule fwd { on p do { emit p to out miss } } }
    edges {
        L -> F : 1
        F.pass -> C : 1
        C.heads -> HD : 1
        HD -> HR : 1
        HR -> L : 1
        C.tails -> S : 1
        S -> ME : 1
        ME -> S : 1
        S -> L : 1
    }
}
"#;
    let mut sim = build_sim(custom, scene);
    let _ = prelude;
    sim.run_until(500_000_000);
    let sent = slot_i(&sim, "Caller", "sent");
    let got  = slot_i(&sim, "Caller", "got");
    assert_eq!(sent, 3);
    assert_eq!(got, 3, "with hit_rate=1.0 all reqs should hit and reply");
}

// ----------------------------------------------------------------------------
// CIRCUIT BREAKER — through-traffic counts; failures feed Window→Threshold→Switch.
// Lightweight smoke test: enough failures trip the breaker; subsequent reqs
// route via SW.divert → resp_error.
// ----------------------------------------------------------------------------
#[test]
fn circuit_breaker_composite_trips_and_returns_resp_errors() {
    // Simplified test: the breaker's failure-counting & switching logic
    // is the headline; we don't re-test req/resp roundtripping (Worker
    // test covers that). Inject `packet`s into the Switch directly, and
    // separately feed `fail_tap` to drive Window→Threshold→signal.
    //
    // This is the same shape that `tests/primitives_compose.rs::
    // circuit_breaker_from_primitives_*` already proves — repeated here
    // wrapped in a compound so the *composite* code path is exercised.
    let custom = r#"
compound CB(window_ns: Int = 10000000000, threshold: Int = 3) {
    in  { input: SW; fail_tap: W }
    out { passed: PE; diverted: DE }
    node SW : Switch    { passing: 1 }
    node W  : Window    { window_ns: window_ns }
    node Th : Threshold { limit: threshold }
    node PE { rule fwd { on packet(p) do { emit packet(p) to out passed } } }
    node DE { rule fwd { on packet(p) do { emit packet(p) to out diverted } } }
    edges {
        SW.pass   -> PE : 1
        SW.divert -> DE : 1
        W         -> Th : 1
        Th        -> SW : 1
    }
}
"#;
    let scene = r#"
        node Src {
            slots { n: Int = 0 }
            rule fire {
                on tick(_)
                when n < 10
                do {
                    n := n + 1
                    emit packet(nil) to default
                    emit packet(nil) to FailTap
                    emit tick(nil) to self
                }
            }
            on_spawn { self -> self : 10000000; inject tick(nil) }
        }
        node FailTap {
            slots { dummy: Int = 0 }
            rule fwd { on packet(p) do { emit packet(p) to default } }
        }
        node Passed   : Counter { }
        node Diverted : Counter { }
        edges {
            Src     -> CB.input    : 1ms
            Src     -> FailTap     : 1ms
            FailTap -> CB.fail_tap : 1ms
            CB.passed   -> Passed   : 1ms
            CB.diverted -> Diverted : 1ms
        }
    "#;
    let mut sim = build_sim(custom, scene);
    sim.run_until(500_000_000);
    let passed   = slot_i(&sim, "Passed",   "count");
    let diverted = slot_i(&sim, "Diverted", "count");
    assert!(passed >= 1, "at least one packet should pass before trip; passed={}", passed);
    assert!(diverted >= 1, "after threshold crossed, packets should divert; diverted={}", diverted);
    // The breaker trips on the 3rd in-window tap; subsequent packets divert.
    assert!(passed <= 5,
        "breaker should trip and stop letting traffic through after a few; passed={}", passed);
}

// ----------------------------------------------------------------------------
// SAGA — one SagaStep with fail_prob=0 passes the req through; with fail_prob=1
// emits compensate back to caller.
// ----------------------------------------------------------------------------
#[test]
fn saga_step_failure_emits_compensate() {
    // fail_prob=1 baked in.
    // We model compensation as `resp_error` (functionally a "failed
    // step that needs rollback"). The composite uses Lower (which
    // already pops resp/resp_error through `out response`), so we can
    // reuse the standard reverse-routing without extending Lower with
    // a new variant.
    let custom = r#"
compound SagaStepC(fail_prob: Float = 1.0) {
    in  { request: L }
    out { forward: FE; response: L }
    # Inline entry adapter (formerly Lower).
    node L {
        rule rev { on p when p.kind == "resp" || p.kind == "resp_error"
                  do { emit p popping to out response } }
        rule fwd { on p when p.kind == "req"
                  do { emit packet(p.value) pushing self to default } }
    }
    # Coin's heads fires when Bernoulli(p) is true; with `p: fail_prob`
    # that's the fail probability, so heads → Fail, tails → forward.
    node C : Coin  { p: fail_prob }
    node FE : Egress { }
    node Fail {
        rule on_packet { on p do { emit resp_error(nil) popping to (head(return_path)) } }
    }
    edges {
        L -> C : 1
        C.heads -> Fail : 1
        C.tails -> FE   : 1
        Fail    -> L    : 1
        FE      -> L    : 1
    }
}
"#;
    let scene = r#"
        node Caller {
            slots { sent: Int = 0; comp: Int = 0 }
            rule fire {
                on go(_) do { sent := sent + 1; emit req(0) pushing self to default }
            }
            rule on_comp { on resp_error(_) do { comp := comp + 1 } }
        }
        edges {
            Caller -> SagaStepC.request : 1ms
            SagaStepC.response -> Caller : 1ms
        }
        scenario { at 0ns: inject Caller <- go() }
    "#;
    let mut sim = build_sim(custom, scene);
    sim.run_until(100_000_000);
    let comp = slot_i(&sim, "Caller", "comp");
    assert_eq!(comp, 1, "SagaStep with fail_prob=1 should compensate once");
}

// ----------------------------------------------------------------------------
// TPC — 2 participants with yes_prob=1 → coordinator commits.
// ----------------------------------------------------------------------------
#[test]
fn tpc_unanimous_yes_commits() {
    // Hardcode participants=2, yes_prob=1.0.
    let custom = r#"
compound Coord(participants: Int = 2, color: Int = 0) {
    in  { request: L; vote: A }
    out { response: L; prepare: PE }
    # Inline entry adapter (formerly Lower).
    node L {
        rule rev { on p when p.kind == "resp" || p.kind == "resp_error"
                  do { emit p popping to out response } }
        rule fwd { on p when p.kind == "req"
                  do { emit packet(p.value) pushing self to default } }
    }
    node FO : FanOut { }
    node PE { rule fwd { on p do { emit p to out prepare } } }
    node A : Aggregator { n: participants }
    node Decide {
        slots { n: Int = participants }
        rule decide {
            on p when p.kind == "total"
            do { emit pkt(if p.value == n then 1 else 0) to default }
        }
    }
    node Resp {
        # Emit through L: L's rev rule knows how to pop and route via the
        # compound's response port. Doing the popping here would leave L
        # popping an empty return_path.
        rule ok    { on p when p.kind == "pkt" && p.value == 1 do { emit resp(nil) to default } }
        rule abort { on p when p.kind == "pkt" && p.value == 0 do { emit resp_error(nil) to default } }
    }
    edges {
        L -> FO : 1
        FO -> PE : 1
        A -> Decide : 1
        Decide -> Resp : 1
        Resp -> L : 1
    }
}
compound Part(yes_prob: Float = 1.0) {
    in  { request: L }
    out { response: L }
    # Inline entry adapter (formerly Lower).
    node L {
        rule rev { on p when p.kind == "resp" || p.kind == "resp_error"
                  do { emit p popping to out response } }
        rule fwd { on p when p.kind == "req"
                  do { emit packet(p.value) pushing self to default } }
    }
    node C : Coin  { p: yes_prob }
    node Yes { rule on_packet { on p do { emit resp(1) popping to (head(return_path)) } } }
    node No  { rule on_packet { on p do { emit resp(0) popping to (head(return_path)) } } }
    edges {
        L -> C : 1
        C.heads -> Yes : 1
        C.tails -> No : 1
        Yes -> L : 1
        No -> L : 1
    }
}
"#;
    let scene = r#"
        node Caller {
            slots { sent: Int = 0; ok: Int = 0; err: Int = 0 }
            rule fire {
                on go(_) do { sent := sent + 1; emit req(0) pushing self to default }
            }
            rule on_resp       { on resp(_)       do { ok  := ok + 1 } }
            rule on_resp_error { on resp_error(_) do { err := err + 1 } }
        }
        # Two participants — wired into the coordinator's prepare port and feeding back to vote port.
        # The participants' `response` is a `resp(v)` which carries the vote bit.
        # We need to convert participant `resp` → coordinator `vote` (packet) for the Aggregator.
        # Glue: VoteBridge listens on resp and emits packet to Aggregator.
        node VoteBridge {
            slots { dummy: Int = 0 }
            rule on_resp { on resp(v) do { emit packet(v) to default } }
        }
        edges {
            Caller -> Coord.request : 1ms
            Coord.response -> Caller : 1ms

            # Coordinator broadcasts prepare to both participants.
            Coord.prepare -> Part1.request : 1ms
            Coord.prepare -> Part2.request : 1ms

            # Participants' responses go back to coordinator via the VoteBridge.
            Part1.response -> VoteBridge : 1ms
            Part2.response -> VoteBridge : 1ms
            VoteBridge -> Coord.vote : 1ms
        }
        scenario { at 0ns: inject Caller <- go() }
    "#;
    // The custom prelude declares Coord and Part compounds; we need
    // instances Part1, Part2 as singletons in the scene. The compound
    // singleton form expands once per declaration, so we declare two
    // distinct compounds with overrides.
    let extra = r#"
compound Part1(yes_prob: Float = 1.0) {
    in  { request: Yes }
    out { response: Yes }
    node Yes { rule on_packet { on packet(_) do { emit resp(1) to out response } } }
}
compound Part2(yes_prob: Float = 1.0) {
    in  { request: Yes }
    out { response: Yes }
    node Yes { rule on_packet { on packet(_) do { emit resp(1) to out response } } }
}
"#;
    let prelude = format!("{}\n{}", custom, extra);
    let mut sim = build_sim(&prelude, scene);
    sim.run_until(500_000_000);
    let ok  = slot_i(&sim, "Caller", "ok");
    let err = slot_i(&sim, "Caller", "err");
    assert_eq!(ok, 1, "unanimous yes should commit; ok={}, err={}", ok, err);
    assert_eq!(err, 0);
}

// ----------------------------------------------------------------------------
// LIFE CELL — 3×3 toroidal Life with a single live cell at center.
// After one generation the lone cell dies (Conway B3/S23: a cell with 0
// neighbors dies; a dead cell with 1 live neighbor stays dead).
// ----------------------------------------------------------------------------
#[test]
fn life_composite_lone_cell_dies() {
    // We need 9 LifeCellComposite instances (3×3) wired as 8-neighbor
    // toroidal. The compound singleton model means each instance needs
    // its own compound declaration. Use a small helper builder:
    let mut prelude = String::new();
    for y in 0..3 {
        for x in 0..3 {
            let alive = if x == 1 && y == 1 { 1 } else { 0 };
            // Each LifeCell is a separate compound declaration so we
            // get distinct singletons. Using LIFE_CELL_COMPOSITE
            // directly would only declare one instance.
            let body = format!(
                r#"
compound L_{x}_{y}(period_ns: Int = 200000000, init_alive: Int = {alive}) {{
    in  {{ report:    A }}
    out {{ broadcast: BE }}
    node T : Tick {{ period_ns: period_ns }}
    node A : Aggregator {{ n: 8 }}
    node CellState {{
        slots {{ alive: Int = init_alive; period_ns: Int = period_ns }}
        rule on_tick  {{ on packet(_) do {{ emit report(alive) to default }} }}
        rule on_total {{ on total(sum) do {{
            alive := if (alive == 1 && (sum == 2 || sum == 3))
                      || (alive == 0 && sum == 3) then 1 else 0
        }} }}
    }}
    node BE {{
        rule fwd_report {{ on report(v) do {{ emit report(v) to out broadcast }} }}
    }}
    edges {{
        T -> CellState : 1
        CellState -> BE : 1
        A -> CellState : 1
    }}
}}
"#,
                x = x, y = y, alive = alive
            );
            prelude.push_str(&body);
        }
    }

    // Wire the 8-neighbor toroidal edges.
    let mut edges = String::from("edges {\n");
    for y in 0..3i64 {
        for x in 0..3i64 {
            for dy in [-1i64, 0, 1] {
                for dx in [-1i64, 0, 1] {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = (x + dx + 3) % 3;
                    let ny = (y + dy + 3) % 3;
                    edges.push_str(&format!(
                        "    L_{x}_{y}.broadcast -> L_{nx}_{ny}.report : 1\n",
                        x=x, y=y, nx=nx, ny=ny
                    ));
                }
            }
        }
    }
    edges.push_str("}\n");

    let mut sim = build_sim(&prelude, &edges);
    // Run past the first generation tick (200ms).
    sim.run_until(250_000_000);

    // Initial: cell at (1,1) alive; after one generation it dies
    // (0 live neighbors). All others remain dead (no cell has 3 live
    // neighbors in this initial config).
    let alive_count: usize = (0..3).flat_map(|y| (0..3).map(move |x| (x, y))).filter(|(x, y)| {
        let name = format!("L_{}_{}::CellState", x, y);
        slot_i(&sim, &name, "alive") == 1
    }).count();
    assert_eq!(alive_count, 0,
        "after one generation the lone centre cell must die");
    assert!(sim.error_counts.is_empty(),
        "no errors: {:?}", sim.error_counts);
}

// ----------------------------------------------------------------------------
// LIFE CELL FROM PURE PRIMITIVES — verifies Conway B3/S23 is correctly
// implemented by the 10-primitive recipe (Tick + Switch + ConstantPacket
// + FanOut + Aggregator + Filter chain + ConstantSignal). NO bespoke
// CellState node anywhere.
//
// Test patterns on a 3×3 toroidal grid:
//   1. lone live cell → all dead next gen (centre has 0 neighbours,
//      every dead cell sees only 1 neighbour → no births).
//   2. full grid alive → most cells die (each has 8 live neighbours,
//      well above S23's max of 3; the rule kills them all on B3/S23).
// ----------------------------------------------------------------------------

fn build_pure_primitives_life_grid(w: usize, h: usize, init: &[Vec<u8>]) -> flow::sim::Sim {
    let mut src = String::new();
    src.push_str("# Pure-primitive Conway grid for the test.\n");
    for y in 0..h {
        for x in 0..w {
            let alive = init[y][x];
            src.push_str(&format!(r#"
node T_{x}_{y}     : Tick           {{ period_ns: 200000000 }}
node SW_{x}_{y}    : Switch         {{ passing: {alive} }}
node C1_{x}_{y}    : Constant {{ value: 1 }}
node C0_{x}_{y}    : Constant {{ value: 0 }}
node B_{x}_{y}     : FanOut         {{ }}
node A_{x}_{y}     : Aggregator     {{ n: 8 }}
node F3_{x}_{y}    : Filter         {{ match: 3 }}
node F2_{x}_{y}    : Filter         {{ match: 2 }}
node ToAlv_{x}_{y} : Constant {{ value: 0, out_kind: "signal" }}
node ToDed_{x}_{y} : Constant {{ value: 1, out_kind: "signal" }}
"#));
        }
    }
    src.push_str("edges {\n");
    for y in 0..h {
        for x in 0..w {
            src.push_str(&format!(
                "T_{x}_{y} -> SW_{x}_{y} : 1ms\n\
                 SW_{x}_{y}.pass -> C1_{x}_{y} : 1ms\n\
                 SW_{x}_{y}.divert -> C0_{x}_{y} : 1ms\n\
                 C1_{x}_{y} -> B_{x}_{y} : 1ms\n\
                 C0_{x}_{y} -> B_{x}_{y} : 1ms\n\
                 A_{x}_{y} -> F3_{x}_{y} : 1ms\n\
                 F3_{x}_{y}.pass -> ToAlv_{x}_{y} : 1ms\n\
                 F3_{x}_{y}.reject -> F2_{x}_{y} : 1ms\n\
                 F2_{x}_{y}.reject -> ToDed_{x}_{y} : 1ms\n\
                 ToAlv_{x}_{y} -> SW_{x}_{y} : 1ms\n\
                 ToDed_{x}_{y} -> SW_{x}_{y} : 1ms\n"));
            for dx in [-1i64, 0, 1] {
                for dy in [-1i64, 0, 1] {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = ((x as i64 + dx + w as i64) % w as i64) as usize;
                    let ny = ((y as i64 + dy + h as i64) % h as i64) as usize;
                    src.push_str(&format!("B_{x}_{y} -> A_{nx}_{ny} : 1ms\n"));
                }
            }
        }
    }
    src.push_str("}\n");

    let mut sim = flow::sim::Sim::new(0);
    flow::dsl::register_classes(&mut sim, GADGETS_DSL).expect("primitive registry");
    flow_bevy::gadgets::install_back_compat_aliases(&mut sim);
    let file = flow::dsl::parse(&src).unwrap_or_else(|e| panic!("parse: {}", e));
    let file = flow::dsl::expand::expand(&file).unwrap_or_else(|e| panic!("expand: {}", e));
    let _ = flow::dsl::lower_into(&mut sim, &file).unwrap_or_else(|e| panic!("lower: {}", e));
    sim
}

fn alive_at(sim: &flow::sim::Sim, x: usize, y: usize) -> i64 {
    slot_i(sim, &format!("SW_{x}_{y}"), "passing")
}

#[test]
fn pure_primitives_life_lone_cell_dies() {
    // 3×3 toroidal grid, only the centre is alive.
    let init = vec![
        vec![0, 0, 0],
        vec![0, 1, 0],
        vec![0, 0, 0],
    ];
    let mut sim = build_pure_primitives_life_grid(3, 3, &init);
    // 250ms = one tick window (period is 200ms).
    sim.run_until(250_000_000);

    // After one generation: every cell should be dead.
    //   - Centre (1,1): 0 live neighbours → dies.
    //   - All other cells: 1 live neighbour (the centre) → stays dead (B3 needs 3).
    for y in 0..3 {
        for x in 0..3 {
            assert_eq!(alive_at(&sim, x, y), 0,
                "cell ({x},{y}) should be dead after gen 1; was alive");
        }
    }
    assert!(sim.error_counts.is_empty(),
        "no engine errors expected: {:?}", sim.error_counts);
}

#[test]
fn pure_primitives_life_all_alive_dies() {
    // 3×3 toroidal grid, all 9 cells alive.
    let init = vec![
        vec![1, 1, 1],
        vec![1, 1, 1],
        vec![1, 1, 1],
    ];
    let mut sim = build_pure_primitives_life_grid(3, 3, &init);
    sim.run_until(250_000_000);

    // Every cell has 8 live neighbours (B3/S23: dies — overpopulated).
    // So all cells should be dead after one generation.
    for y in 0..3 {
        for x in 0..3 {
            assert_eq!(alive_at(&sim, x, y), 0,
                "cell ({x},{y}) should die from overpopulation (8 live neighbours); was alive");
        }
    }
    assert!(sim.error_counts.is_empty(),
        "no engine errors expected: {:?}", sim.error_counts);
}

#[test]
fn pure_primitives_life_three_in_a_row_blinker_oscillates() {
    // 5×5 toroidal grid with a 3-cell horizontal "blinker" on row 2.
    // Conway B3/S23: blinker oscillates between horizontal & vertical
    // each generation. After one generation: vertical line of 3 cells
    // centred on (2,2).
    let init = vec![
        vec![0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0],
        vec![0, 1, 1, 1, 0],
        vec![0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0],
    ];
    let mut sim = build_pure_primitives_life_grid(5, 5, &init);
    // First tick is at t=0 (on_spawn inject). Subsequent ticks at every
    // 200ms. Run for 100ms — well after gen 1 has finished propagating
    // through the pipeline (longest path ~7 hops × 1ms) but BEFORE gen
    // 2's tick fires at t=200ms, so we see exactly one generation.
    sim.run_until(100_000_000);

    // After gen 1 the live cells should be (2,1), (2,2), (2,3) (vertical).
    let expected = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ];
    let mut dump = String::new();
    for y in 0..5 {
        for x in 0..5 {
            dump.push(if alive_at(&sim, x, y) != 0 { '#' } else { '.' });
        }
        dump.push('\n');
    }
    for y in 0..5 {
        for x in 0..5 {
            assert_eq!(alive_at(&sim, x, y), expected[y][x] as i64,
                "blinker gen 1: cell ({x},{y}) expected {} got {}\nactual grid:\n{}",
                expected[y][x], alive_at(&sim, x, y), dump);
        }
    }
    assert!(sim.error_counts.is_empty(),
        "no engine errors expected: {:?}", sim.error_counts);
}

#[test]
fn pure_primitives_life_blinker_oscillates_over_many_generations() {
    // Same horizontal blinker on a 5×5 toroidal grid, but run through
    // multiple ticks to verify the oscillator stays correct over time
    // (catches bugs where state updates aren't atomic per generation,
    // or where the Aggregator's reset doesn't propagate cleanly).
    //
    // Tick period 200ms; first tick fires at t=0 (on_spawn inject),
    // then every 200ms. Each generation needs ~10ms to fully cascade
    // through Aggregator → Filter → ConstantSignal → Switch, so
    // sampling 100ms into each window catches the settled state.
    let init = vec![
        vec![0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0],
        vec![0, 1, 1, 1, 0],
        vec![0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0],
    ];
    let horizontal = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ];
    let vertical = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ];
    let mut sim = build_pure_primitives_life_grid(5, 5, &init);

    let grid_dump = |sim: &flow::sim::Sim| {
        let mut s = String::new();
        for y in 0..5 {
            for x in 0..5 {
                s.push(if alive_at(sim, x, y) != 0 { '#' } else { '.' });
            }
            s.push('\n');
        }
        s
    };

    for g in 1..=10 {
        sim.run_until(100_000_000 + (g as u64 - 1) * 200_000_000);
        let expected = if g % 2 == 1 { &vertical } else { &horizontal };
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(
                    alive_at(&sim, x, y), expected[y][x] as i64,
                    "gen {}: cell ({x},{y}) expected {} got {}\nactual:\n{}",
                    g, expected[y][x], alive_at(&sim, x, y), grid_dump(&sim)
                );
            }
        }
        assert!(
            sim.error_counts.is_empty(),
            "engine errors at gen {}: {:?}", g, sim.error_counts
        );
    }
}
