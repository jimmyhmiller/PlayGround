//! Full-fidelity sim snapshot: captures the complete `Sim` state
//! (nodes, edges, slots, in-flight packets, pending scenario actions,
//! RNG state, event log) as a serde-serialised envelope.
//!
//! A snapshot restored into a fresh process produces bitwise-identical
//! future evolution to the original run from that point onward — every
//! self-loop bootstrap packet, every random draw, every scheduled
//! action is preserved. This is the piece that makes canvas-level
//! "frozen scenario starting points" actually work: resuming a
//! generator-driven topology from a snapshot keeps generating packets,
//! workers resume mid-service, clients get their pending replies.
//!
//! The envelope wraps `Sim` with a small header (label, version) so
//! format evolution has a place to hang migrations off.

use serde::{Deserialize, Serialize};

use crate::sim::{Sim, Time};

/// On-disk snapshot envelope. Always versioned so loaders can refuse or
/// migrate older files deliberately.
#[derive(Clone, Serialize, Deserialize)]
pub struct SimSnapshot {
    #[serde(default = "default_format_version")]
    pub format_version: u32,
    /// Freeform label shown in the UI when picking a snapshot.
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub description: String,
    /// Sim time at which the snapshot was taken, in ns. Redundant with
    /// `sim.now_ns` but surfaced at the top so tooling can inspect
    /// without deserialising the whole blob.
    pub sim_time_ns: Time,
    /// The full serialised sim.
    pub sim: Sim,
}

fn default_format_version() -> u32 { 1 }

impl SimSnapshot {
    /// Capture the complete state of `sim` into a snapshot.
    pub fn capture(sim: &Sim, label: impl Into<String>) -> Self {
        SimSnapshot {
            format_version: 1,
            label: label.into(),
            description: String::new(),
            sim_time_ns: sim.now_ns,
            sim: sim.clone(),
        }
    }

    /// Consume the snapshot, yielding the stored `Sim`. Callers that
    /// want to keep the envelope around can clone `snap.sim` instead.
    pub fn into_sim(self) -> Sim {
        self.sim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    /// A sim with a Generator-like self-loop. The `tick` packet lives
    /// only on the self-edge's in-flight queue — if snapshot restore
    /// loses in-flight packets the loop dies and `hits` stops growing.
    const SRC: &str = r#"
        node Gen {
            slots {
                hits:      Int = 0
                period_ns: Int = 100000000
            }
            on_spawn {
                self -> self : period_ns
                inject tick(nil)
            }
            rule on_tick {
                on tick(_)
                do {
                    hits := hits + 1
                    emit tick(nil) to self
                }
            }
        }
    "#;

    #[test]
    fn roundtrip_through_json_preserves_state() {
        let mut sim = crate::dsl::load(SRC, 0).unwrap();
        sim.run_until(1_000_000_000); // 1s → 10 ticks at 100ms period
        let id = sim.node_by_name("Gen").unwrap();
        let hits_before = sim.nodes[&id].slots["hits"].clone();

        let snap = SimSnapshot::capture(&sim, "at-1s");
        let json = serde_json::to_string(&snap).unwrap();
        let restored: SimSnapshot = serde_json::from_str(&json).unwrap();
        let restored_sim = restored.into_sim();

        // Slots preserved.
        let id_r = restored_sim.node_by_name("Gen").unwrap();
        assert_eq!(restored_sim.nodes[&id_r].slots["hits"], hits_before);
        assert_eq!(restored_sim.now_ns, 1_000_000_000);
    }

    #[test]
    fn restored_sim_continues_generating() {
        // The teeth of the full-snapshot contract: after restore, the
        // generator's self-loop must keep firing. This would FAIL under
        // a minimal-state snapshot that didn't capture the in-flight
        // tick packet.
        let mut sim = crate::dsl::load(SRC, 0).unwrap();
        sim.run_until(500_000_000); // 0.5s → 5 ticks
        let id = sim.node_by_name("Gen").unwrap();
        let hits_at_500ms = sim.nodes[&id].slots["hits"].as_int().unwrap();
        assert!(hits_at_500ms > 0, "sanity: generator should have ticked");

        let snap = SimSnapshot::capture(&sim, "at-500ms");
        let json = serde_json::to_string(&snap).unwrap();
        let mut restored = serde_json::from_str::<SimSnapshot>(&json)
            .unwrap()
            .into_sim();

        // Run another 500ms. If the in-flight tick packet was lost on
        // round-trip, hits would stay at hits_at_500ms forever.
        let _ = restored.run_until(1_000_000_000);
        let id_r = restored.node_by_name("Gen").unwrap();
        let hits_at_1s = restored.nodes[&id_r].slots["hits"].as_int().unwrap();
        assert!(
            hits_at_1s > hits_at_500ms,
            "generator stopped after restore (hits_at_500ms={}, hits_at_1s={})",
            hits_at_500ms,
            hits_at_1s
        );
    }

    #[test]
    fn restored_sim_matches_original_future() {
        // Determinism: restoring from a snapshot and running N more
        // nanoseconds should produce the same state as continuing the
        // original sim for N more nanoseconds. This is what makes
        // snapshots a real "pause/resume" mechanism rather than a
        // best-effort reseed.
        let mut a = crate::dsl::load(SRC, 0).unwrap();
        a.run_until(300_000_000);

        let snap = SimSnapshot::capture(&a, "mid");
        let json = serde_json::to_string(&snap).unwrap();
        let mut b = serde_json::from_str::<SimSnapshot>(&json)
            .unwrap()
            .into_sim();

        // Continue both for the same amount of sim time.
        a.run_until(1_000_000_000);
        b.run_until(1_000_000_000);

        let id_a = a.node_by_name("Gen").unwrap();
        let id_b = b.node_by_name("Gen").unwrap();
        assert_eq!(a.nodes[&id_a].slots["hits"], b.nodes[&id_b].slots["hits"]);
        assert_eq!(a.now_ns, b.now_ns);
    }

    #[test]
    fn rng_state_is_preserved() {
        // A rule that consumes RNG via a stochastic edge latency. If
        // the RNG stream resets on restore, the two sims will diverge.
        const STOCH: &str = r#"
            node Gen {
                slots { hits: Int = 0, mean_ns: Int = 10_000_000 }
                on_spawn {
                    self -> self : Exp(mean_ns)
                    inject tick(nil)
                }
                rule on_tick {
                    on tick(_)
                    do {
                        hits := hits + 1
                        emit tick(nil) to self
                    }
                }
            }
        "#;
        let mut a = crate::dsl::load(STOCH, 42).unwrap();
        a.run_until(50_000_000);

        let snap = SimSnapshot::capture(&a, "mid");
        let json = serde_json::to_string(&snap).unwrap();
        let mut b = serde_json::from_str::<SimSnapshot>(&json)
            .unwrap()
            .into_sim();

        a.run_until(200_000_000);
        b.run_until(200_000_000);

        let id_a = a.node_by_name("Gen").unwrap();
        let id_b = b.node_by_name("Gen").unwrap();
        // Stochastic latencies — only matching hits prove the RNG
        // stream resumed at the right point.
        assert_eq!(
            a.nodes[&id_a].slots["hits"],
            b.nodes[&id_b].slots["hits"],
            "RNG-driven sim diverged across snapshot/restore"
        );
    }

    #[test]
    fn pending_scenario_actions_survive_roundtrip() {
        // A scenario with actions scheduled AFTER the snapshot moment.
        // Those pending actions live on `sim.pending_actions` (the
        // BinaryHeap) — if round-trip loses them, the late inject
        // never lands and hits stays at 2.
        const WITH_SCEN: &str = r#"
            node C {
                slots { hits: Int = 0 }
                rule on_ping { on ping(_) do { hits := hits + 1 } }
            }
            scenario {
                at 0ns:   inject C <- ping(nil)
                at 100ns: inject C <- ping(nil)
                at 500ns: inject C <- ping(nil)
            }
        "#;
        let mut sim = crate::dsl::load(WITH_SCEN, 0).unwrap();
        sim.run_until(200); // delivers first two; third still pending
        let id = sim.node_by_name("C").unwrap();
        assert_eq!(sim.nodes[&id].slots["hits"], Value::Int(2));

        let snap = SimSnapshot::capture(&sim, "mid");
        let json = serde_json::to_string(&snap).unwrap();
        let mut restored = serde_json::from_str::<SimSnapshot>(&json)
            .unwrap()
            .into_sim();

        restored.run_until(1000);
        let id_r = restored.node_by_name("C").unwrap();
        assert_eq!(
            restored.nodes[&id_r].slots["hits"],
            Value::Int(3),
            "pending action at 500ns lost during round-trip"
        );
    }
}
