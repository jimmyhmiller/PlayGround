//! A synthetic [`WorldSource`] for demos, screenshots, and testing the renderer
//! without touching real logs. Activity gently evolves over time so villagers come
//! and go. Run with `--mock`.

use super::{CityInfo, CodebaseInfo, SessionInfo, ToolCounts, WorldSnapshot, WorldSource};
use crate::util::{now_unix, Rng};

/// (name, model, dominant language) — language picks the biome.
const PROJECTS: &[(&str, &str, &str)] = &[
    ("beagle", "claude-opus-4-8", "rs"),
    ("jim-editor", "claude-opus-4-8", "rs"),
    ("datalog-db", "claude-opus-4-8", "rs"),
    ("clojure-jvm", "claude-opus-4-8", "clj"),
    ("coil", "claude-opus-4-8", "rs"),
    ("tallyc", "claude-sonnet-4-6", "rs"),
    ("v4-calories", "claude-sonnet-4-6", "swift"),
    ("portable-node", "claude-haiku-4-5", "js"),
    ("ion-layout", "claude-haiku-4-5", "ts"),
    ("simd-lang", "claude-haiku-4-5", "rs"),
    ("partial-new", "claude-sonnet-4-6", "rs"),
    ("pyscan", "claude-sonnet-4-6", "py"),
    ("legacy-engine", "claude-sonnet-4-6", "c"),
    ("field-notes", "claude-haiku-4-5", "md"),
];

/// Tool whose name we lean on to set a session's building type.
const DOM_TOOLS: &[&str] = &["Bash", "Edit", "Read", "Task", "WebFetch", "Grep", "TodoWrite"];

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
        let today = (now / 86_400.0) as i32;
        let mut cities = Vec::new();

        for (ci, (proj, model, lang)) in PROJECTS.iter().enumerate() {
            // Spread projects across the full range of ages: tiny outposts to
            // sprawling metropolises.
            let mult = 0.2_f32 * 2f32.powi((ci % 7) as i32);
            let n_sessions = (1.0 + mult * 2.0).round().min(13.0) as usize;
            let mut sessions = Vec::new();
            for si in 0..n_sessions {
                let mut srng = Rng::seeded(&(proj, si, "sess"));
                let live = (ci + si) as u64 % 4 == self.tick % 4;
                let base_msgs = (10.0 + mult * srng.range(20.0, 120.0)) as u32;
                let grow = if live { (self.tick % 30) as u32 } else { 0 };
                let user = base_msgs / 6 + 1;
                let assistant = base_msgs + grow;

                // Tool mix with a dominant category -> building type.
                let dom = DOM_TOOLS[(ci + si) % DOM_TOOLS.len()];
                let mut tools = ToolCounts::default();
                let n_tools = assistant * 2;
                for k in 0..n_tools {
                    if k % 5 < 3 {
                        tools.add(dom);
                    } else {
                        tools.add(["Read", "Edit", "Bash"][(k as usize) % 3]);
                    }
                }

                // Some sessions use a different model -> model variety per city.
                let smodel = if si % 3 == 1 && ci % 2 == 0 {
                    "claude-sonnet-4-6"
                } else {
                    model
                };

                // Activity spread across days; some night-owl sessions.
                let ndays = (1.0 + mult * 3.0).round().min(40.0) as i32;
                let day = today - srng.below(ndays.max(1) as usize) as i32;
                let hour = if si % 4 == 0 { srng.below(4) } else { 9 + srng.below(12) };
                let last_active = if live {
                    now - srng.range(0.0, 8.0) as f64
                } else {
                    let hi: f32 = mult * 600_000.0 + 120.0;
                    now - srng.range(120.0, hi) as f64
                };

                sessions.push(SessionInfo {
                    id: format!("{proj}-{si}"),
                    title: Some(TITLES[(ci + si) % TITLES.len()].to_string()),
                    model: Some(smodel.to_string()),
                    user_messages: user,
                    assistant_messages: assistant,
                    tool_uses: tools.total,
                    tools,
                    total_tokens: assistant as u64 * 1500,
                    hours_mask: 1 << (hour.min(23)),
                    days: vec![day],
                    first_active: Some(now - 86_400.0 * ndays as f64),
                    last_active: Some(last_active),
                    git_branch: Some("master".to_string()),
                });
            }

            let files = (40.0 * mult).max(4.0) as u32;
            let codebase = Some(CodebaseInfo {
                languages: vec![(lang.to_string(), files), ("md".into(), 3), ("toml".into(), 1)],
                files: files + 12,
                loc: (files as u64) * 220,
                bytes: (files as u64) * 7000,
                commits: (mult * 90.0) as u32,
                has_readme: true,
                has_tests: ci % 2 == 0,
            });

            cities.push(CityInfo {
                id: format!("mock:{proj}"),
                name: proj.to_string(),
                path: Some(format!("/Users/demo/Code/{proj}")),
                sessions,
                codebase,
            });
        }

        cities.sort_by(|a, b| b.total_messages().cmp(&a.total_messages()));
        WorldSnapshot { cities, captured_at: now }
    }
}
