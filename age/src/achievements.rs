//! Achievements a codebase can earn — evaluated from real metrics only. If a
//! signal isn't available (e.g. the repo wasn't scanned, so `loc == 0`), the
//! achievement simply can't unlock; nothing is ever faked.
//!
//! Each unlocked achievement raises a monument in the city and lists in the
//! inspector. See `DESIGN.md` for the full vision.

use crate::data::{CityInfo, ToolCounts};

/// Flattened, render-ready metrics for one city.
#[derive(Clone, Default)]
pub struct Metrics {
    pub sessions: usize,
    pub messages: u32,
    pub tools: ToolCounts,
    pub tokens: u64,
    pub longest_session: u32,
    pub active_days: usize,
    pub hours_mask: u32,
    pub has_opus: bool,
    pub has_sonnet: bool,
    pub has_haiku: bool,
    pub model_count: usize,
    pub loc: u64,
    pub files: u32,
    pub commits: u32,
    pub languages: usize,
    pub has_readme: bool,
    pub has_tests: bool,
    pub age_days: f64,
    pub scanned: bool,
}

impl Metrics {
    pub fn from_city(c: &CityInfo, now: f64) -> Metrics {
        let fams = c.model_families();
        let cb = c.codebase.as_ref();
        Metrics {
            sessions: c.sessions.len(),
            messages: c.total_messages(),
            tools: c.tools(),
            tokens: c.total_tokens(),
            longest_session: c.longest_session_messages(),
            active_days: c.active_days(),
            hours_mask: c.hours_mask(),
            has_opus: fams.contains("opus"),
            has_sonnet: fams.contains("sonnet"),
            has_haiku: fams.contains("haiku"),
            model_count: fams.len(),
            loc: cb.map(|c| c.loc).unwrap_or(0),
            files: cb.map(|c| c.files).unwrap_or(0),
            commits: cb.map(|c| c.commits).unwrap_or(0),
            languages: cb.map(|c| c.languages.len()).unwrap_or(0),
            has_readme: cb.map(|c| c.has_readme).unwrap_or(false),
            has_tests: cb.map(|c| c.has_tests).unwrap_or(false),
            age_days: c.first_active().map(|f| (now - f) / 86_400.0).unwrap_or(0.0),
            scanned: cb.is_some(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cat {
    Activity,
    Craft,
    Codebase,
    Mastery,
    Time,
    Wealth,
}

pub struct AchievementDef {
    /// Stable id for referencing/saving (part of the catalog API; not all UIs use it).
    #[allow(dead_code)]
    pub id: &'static str,
    pub name: &'static str,
    /// One-line description, shown in fuller achievement views.
    #[allow(dead_code)]
    pub desc: &'static str,
    pub cat: Cat,
    pub test: fn(&Metrics) -> bool,
}

const NIGHT: u32 = 0b0001_1111; // hours 0..=4
const MORNING: u32 = 0b0001_1110_0000; // hours 5..=8

#[rustfmt::skip]
pub const CATALOG: &[AchievementDef] = &[
    // --- Activity ---------------------------------------------------------
    a("founding",  "Founding",      "Hold your first session",        Cat::Activity, |m| m.sessions >= 1),
    a("hamlet",    "Hamlet",        "10 sessions",                    Cat::Activity, |m| m.sessions >= 10),
    a("bustling",  "Bustling",      "50 sessions",                    Cat::Activity, |m| m.sessions >= 50),
    a("capital",   "Great Capital", "150 sessions",                   Cat::Activity, |m| m.sessions >= 150),
    a("chatty",    "Chatty",        "1,000 messages",                 Cat::Activity, |m| m.messages >= 1_000),
    a("verbose",   "Verbose",       "5,000 messages",                 Cat::Activity, |m| m.messages >= 5_000),
    a("epic",      "Epic Saga",     "20,000 messages",                Cat::Activity, |m| m.messages >= 20_000),
    a("marathon",  "Marathon",      "A single 400-message session",   Cat::Activity, |m| m.longest_session >= 400),
    // --- Craft ------------------------------------------------------------
    a("smith",     "Smith",         "500 Bash commands",              Cat::Craft, |m| m.tools.bash >= 500),
    a("architect", "Architect",     "1,000 edits",                    Cat::Craft, |m| m.tools.edit >= 1_000),
    a("scholar",   "Scholar",       "1,000 file reads",               Cat::Craft, |m| m.tools.read >= 1_000),
    a("seeker",    "Seeker",        "300 searches",                   Cat::Craft, |m| m.tools.search >= 300),
    a("general",   "General",       "30 subagents deployed",          Cat::Craft, |m| m.tools.task >= 30),
    a("navigator", "Navigator",     "50 web lookups",                 Cat::Craft, |m| m.tools.web >= 50),
    a("toolsmith", "Toolsmith",     "5,000 tool uses",                Cat::Craft, |m| m.tools.total >= 5_000),
    // --- Codebase ---------------------------------------------------------
    a("sapling",   "Sapling",       "Codebase on disk",               Cat::Codebase, |m| m.scanned && m.files > 0),
    a("grove",     "Grove",         "10k lines of code",              Cat::Codebase, |m| m.loc >= 10_000),
    a("oldgrowth", "Old Growth",    "100k lines of code",             Cat::Codebase, |m| m.loc >= 100_000),
    a("sprawl",    "Sprawl",        "1,000 files",                    Cat::Codebase, |m| m.files >= 1_000),
    a("polyglot",  "Polyglot",      "4+ languages",                   Cat::Codebase, |m| m.languages >= 4),
    a("committed", "Committed",     "200 git commits",                Cat::Codebase, |m| m.commits >= 200),
    a("prolific",  "Prolific",      "1,000 git commits",              Cat::Codebase, |m| m.commits >= 1_000),
    a("ancient",   "Ancient",       "Active for 6+ months",           Cat::Codebase, |m| m.age_days >= 180.0),
    a("tested",    "Tested",        "Has a test suite",               Cat::Codebase, |m| m.has_tests),
    a("documented","Documented",    "Has a README",                   Cat::Codebase, |m| m.has_readme),
    // --- Mastery ----------------------------------------------------------
    a("opus",      "Opus Adept",    "Worked with Opus",               Cat::Mastery, |m| m.has_opus),
    a("sonnet",    "Sonnet Adept",  "Worked with Sonnet",             Cat::Mastery, |m| m.has_sonnet),
    a("haiku",     "Haiku Adept",   "Worked with Haiku",              Cat::Mastery, |m| m.has_haiku),
    a("triumvir",  "Triumvirate",   "Used Opus, Sonnet and Haiku",    Cat::Mastery, |m| m.model_count >= 3),
    // --- Time -------------------------------------------------------------
    a("nightowl",  "Night Owl",     "Worked between midnight and 4am",Cat::Time, |m| m.hours_mask & NIGHT != 0),
    a("earlybird", "Early Bird",    "Worked between 5am and 8am",     Cat::Time, |m| m.hours_mask & MORNING != 0),
    a("veteran",   "Veteran",       "Active across 30+ days",         Cat::Time, |m| m.active_days >= 30),
    // --- Wealth -----------------------------------------------------------
    a("rich",      "Rich",          "10M tokens spent",               Cat::Wealth, |m| m.tokens >= 10_000_000),
    a("tycoon",    "Tycoon",        "100M tokens spent",              Cat::Wealth, |m| m.tokens >= 100_000_000),
];

const fn a(
    id: &'static str,
    name: &'static str,
    desc: &'static str,
    cat: Cat,
    test: fn(&Metrics) -> bool,
) -> AchievementDef {
    AchievementDef { id, name, desc, cat, test }
}

/// All achievements this city has unlocked, in catalog order.
pub fn unlocked(m: &Metrics) -> Vec<&'static AchievementDef> {
    CATALOG.iter().filter(|d| (d.test)(m)).collect()
}
