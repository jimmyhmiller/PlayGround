//! A synthetic [`WorldSource`] for demos, screenshots, and testing the renderer
//! without touching real logs. Activity gently evolves over time so villagers come
//! and go. Run with `--mock`.

use super::{CityInfo, SessionInfo, WorldSnapshot, WorldSource};
use crate::util::{now_unix, Rng};

const PROJECTS: &[(&str, &str)] = &[
    ("beagle", "claude-opus-4-8"),
    ("coil", "claude-opus-4-8"),
    ("tallyc", "claude-sonnet-4-6"),
    ("datalog-db", "claude-opus-4-8"),
    ("v4-calories", "claude-sonnet-4-6"),
    ("jim-editor", "claude-opus-4-8"),
    ("simd-lang", "claude-haiku-4-5"),
    ("partial-new", "claude-sonnet-4-6"),
    ("js-ir", "claude-opus-4-8"),
    ("ion-layout", "claude-haiku-4-5"),
];

const TITLES: &[&str] = &[
    "Fix Never regression in defer/scope",
    "Port kernel-reviewer regressions",
    "Aggregate reference model design",
    "Write-barrier race in concurrent GC",
    "Scale-anchored deficit model",
    "Soft word-wrap in the editor",
    "SIMD token-boundary classifier",
    "Partial-eval the bytecode VM",
    "Tree-shaking across modules",
    "Iongraph parity sweep",
];

pub struct MockSource {
    /// Advances every poll so activity looks alive.
    tick: u64,
}

impl MockSource {
    pub fn new() -> MockSource {
        MockSource { tick: 0 }
    }
}

impl WorldSource for MockSource {
    fn name(&self) -> &str {
        "mock"
    }

    fn poll(&mut self) -> WorldSnapshot {
        self.tick = self.tick.wrapping_add(1);
        let now = now_unix();
        let mut cities = Vec::new();

        for (ci, (proj, model)) in PROJECTS.iter().enumerate() {
            let mut rng = Rng::seeded(&(proj, "city"));
            let n_sessions = 2 + rng.below(5);
            let mut sessions = Vec::new();
            for si in 0..n_sessions {
                let mut srng = Rng::seeded(&(proj, si, "sess"));
                // A couple of sessions per project are "live" and grow each tick.
                let live = (ci + si) as u64 % 4 == self.tick % 4;
                let base_msgs = 4 + srng.below(120) as u32;
                let grow = if live { (self.tick % 30) as u32 } else { 0 };
                let user = base_msgs / 6 + 1;
                let assistant = base_msgs + grow;
                let last_active = if live {
                    now - srng.range(0.0, 8.0) as f64
                } else {
                    now - srng.range(120.0, 86_400.0) as f64
                };
                sessions.push(SessionInfo {
                    id: format!("{proj}-{si}"),
                    title: Some(TITLES[(ci + si) % TITLES.len()].to_string()),
                    model: Some(model.to_string()),
                    user_messages: user,
                    assistant_messages: assistant,
                    tool_uses: assistant * 2,
                    first_active: Some(now - 3600.0 * (si as f64 + 1.0)),
                    last_active: Some(last_active),
                    git_branch: Some("master".to_string()),
                });
            }
            cities.push(CityInfo {
                id: format!("mock:{proj}"),
                name: proj.to_string(),
                path: Some(format!("/Users/demo/Code/{proj}")),
                sessions,
            });
        }

        cities.sort_by(|a, b| b.total_messages().cmp(&a.total_messages()));
        WorldSnapshot { cities, captured_at: now }
    }
}
