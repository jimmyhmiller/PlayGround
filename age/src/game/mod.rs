//! The game/simulation layer. Turns a [`WorldSnapshot`] into a living map:
//! projects become cities, sessions become buildings, and activity becomes
//! villagers who wander their town and occasionally trek to another project.
//!
//! It owns no rendering and no I/O — it consumes snapshots and advances motion.

use crate::achievements::{self, AchievementDef, Metrics};
use crate::data::{CityInfo, CodebaseInfo, SessionInfo, Tool, WorldSnapshot};
use crate::render::assets;
use crate::util::{hash64, Rng};
use raylib::prelude::Vector2;
use std::collections::{HashMap, HashSet};

/// How recently (seconds) a session must have been touched to count as "live".
pub const LIVE_WINDOW: f64 = 600.0;

const GRID_COLS: i32 = 9;
const CELL: f32 = 380.0;
const MAX_VILLAGERS: usize = 8;

// ---- Civilization metadata derived from a city's data ------------------------

/// The "age" of a city — how far the project has advanced. Drives footprint,
/// building count, town-center sprite and walls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Outpost,
    Hamlet,
    Village,
    Town,
    City,
    Metropolis,
}

impl Tier {
    pub fn from_score(s: u64) -> Tier {
        match s {
            0..=49 => Tier::Outpost,
            50..=499 => Tier::Hamlet,
            500..=1_999 => Tier::Village,
            2_000..=7_999 => Tier::Town,
            8_000..=24_999 => Tier::City,
            _ => Tier::Metropolis,
        }
    }
    pub fn index(self) -> usize {
        [Tier::Outpost, Tier::Hamlet, Tier::Village, Tier::Town, Tier::City, Tier::Metropolis]
            .iter()
            .position(|&t| t == self)
            .unwrap_or(0)
    }
    pub fn name(self) -> &'static str {
        ["Outpost", "Hamlet", "Village", "Town", "City", "Metropolis"][self.index()]
    }
    /// Max session-houses to render at this tier (bigger cities, more buildings).
    pub fn max_houses(self) -> usize {
        [2, 4, 7, 10, 13, 16][self.index()]
    }
    /// Town center is a proper castle keep from Town upward (else a hut/house).
    pub fn has_keep(self) -> bool {
        self.index() >= 3
    }
    /// A wall ring appears at City and Metropolis.
    pub fn has_walls(self) -> bool {
        self.index() >= 4
    }
    /// Town-center sprite scale grows with tier.
    pub fn keep_scale(self) -> f32 {
        [2.6, 3.0, 3.4, 3.8, 4.3, 4.9][self.index()]
    }
}

/// The biome a city sits in, themed by its dominant language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Biome {
    Forge,
    Coast,
    Forest,
    Port,
    Steppe,
    Stone,
    Plains,
    Vale,
    Heartland,
}

impl Biome {
    pub fn from_lang(lang: &str) -> Biome {
        match lang {
            "rs" | "zig" | "v" | "nim" => Biome::Forge,
            "swift" | "m" | "mm" => Biome::Coast,
            "clj" | "cljs" | "cljc" | "edn" | "lisp" | "el" | "bg" | "scm" | "rkt" => Biome::Forest,
            "js" | "ts" | "jsx" | "tsx" | "mjs" | "cjs" | "vue" | "svelte" => Biome::Port,
            "py" => Biome::Steppe,
            "c" | "cc" | "cpp" | "cxx" | "h" | "hpp" | "hh" => Biome::Stone,
            "go" => Biome::Plains,
            "md" | "txt" | "org" | "rst" => Biome::Vale,
            _ => Biome::Heartland,
        }
    }
    /// Stable fallback when the project's language is unknown.
    pub fn from_hash(id: &str) -> Biome {
        const OPTS: [Biome; 7] = [
            Biome::Forge, Biome::Coast, Biome::Forest, Biome::Port, Biome::Steppe, Biome::Stone,
            Biome::Plains,
        ];
        OPTS[(hash64(&id) % OPTS.len() as u64) as usize]
    }
    pub fn name(self) -> &'static str {
        match self {
            Biome::Forge => "Forge Hills",
            Biome::Coast => "The Coast",
            Biome::Forest => "Enchanted Forest",
            Biome::Port => "Trade Port",
            Biome::Steppe => "Grass Steppe",
            Biome::Stone => "Ancient Stone",
            Biome::Plains => "Open Plains",
            Biome::Vale => "Scholars' Vale",
            Biome::Heartland => "Heartland",
        }
    }
}

/// How recently the city was active, as a "season" tint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Season {
    HighSummer,
    Summer,
    LateSummer,
    Autumn,
    Winter,
    Dormant,
}

impl Season {
    pub fn from_idle(idle_secs: f64) -> Season {
        const DAY: f64 = 86_400.0;
        match idle_secs {
            s if s < 600.0 => Season::HighSummer,
            s if s < DAY => Season::Summer,
            s if s < 7.0 * DAY => Season::LateSummer,
            s if s < 30.0 * DAY => Season::Autumn,
            s if s < 182.0 * DAY => Season::Winter,
            _ => Season::Dormant,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Season::HighSummer => "High Summer",
            Season::Summer => "Summer",
            Season::LateSummer => "Late Summer",
            Season::Autumn => "Autumn",
            Season::Winter => "Winter",
            Season::Dormant => "Dormant",
        }
    }
}

pub struct World {
    pub cities: Vec<City>,
    index: HashMap<String, usize>,
    occupied: HashSet<i32>,
    pub now: f64,
    /// Bounds of the placed cities, for centering the camera initially.
    pub extent: (Vector2, Vector2),
}

pub struct City {
    pub id: String,
    pub name: String,
    pub path: Option<String>,
    pub pos: Vector2,
    pub buildings: Vec<Building>,
    pub villagers: Vec<Villager>,
    pub sessions: Vec<SessionInfo>,
    pub total_messages: u32,
    pub total_tools: u32,
    pub live: usize,
    pub last_active: Option<f64>,
    // Derived "civilization" metadata.
    pub tier: Tier,
    pub biome: Biome,
    pub season: Season,
    pub codebase: Option<CodebaseInfo>,
    pub metrics: Metrics,
    pub achievements: Vec<&'static AchievementDef>,
}

pub struct Building {
    pub session_id: String,
    pub pos: Vector2, // base (feet) point in world space
    pub is_town_center: bool,
    pub preset: usize,
    pub model: Option<String>,
    pub title: Option<String>,
    pub live: bool,
    pub messages: u32,
    pub smoke_t: f32,
    /// Tier of the owning city (drives the town-center sprite/scale).
    pub tier: Tier,
    /// Dominant tool for this session -> specialized building type.
    pub tool: Tool,
}

pub struct Villager {
    pub pos: Vector2,
    pub target: Vector2,
    pub home: Vector2,
    pub speed: f32,
    pub sprite: i32,
    pub wander_radius: f32,
    pub anim_t: f32,
    pub on_trip: bool,
    rng: Rng,
}

impl World {
    pub fn new() -> World {
        World {
            cities: Vec::new(),
            index: HashMap::new(),
            occupied: HashSet::new(),
            now: 0.0,
            extent: (Vector2::new(0.0, 0.0), Vector2::new(CELL, CELL)),
        }
    }

    /// Reconcile the world with a fresh snapshot, preserving city positions and
    /// villager motion across polls.
    pub fn sync(&mut self, snap: &WorldSnapshot) {
        self.now = snap.captured_at;
        let mut keep: HashSet<String> = HashSet::new();

        for info in &snap.cities {
            keep.insert(info.id.clone());
            match self.index.get(&info.id).copied() {
                Some(i) => {
                    let pos = self.cities[i].pos;
                    update_city(&mut self.cities[i], info, pos, self.now);
                }
                None => {
                    let pos = self.assign_cell(&info.id);
                    let mut city = City {
                        id: info.id.clone(),
                        name: info.name.clone(),
                        path: info.path.clone(),
                        pos,
                        buildings: Vec::new(),
                        villagers: Vec::new(),
                        sessions: Vec::new(),
                        total_messages: 0,
                        total_tools: 0,
                        live: 0,
                        last_active: None,
                        tier: Tier::Outpost,
                        biome: Biome::Heartland,
                        season: Season::Dormant,
                        codebase: None,
                        metrics: Metrics::default(),
                        achievements: Vec::new(),
                    };
                    update_city(&mut city, info, pos, self.now);
                    self.index.insert(info.id.clone(), self.cities.len());
                    self.cities.push(city);
                }
            }
        }

        // Drop cities no longer present (rare — projects seldom vanish).
        if self.cities.iter().any(|c| !keep.contains(&c.id)) {
            self.cities.retain(|c| keep.contains(&c.id));
            self.index.clear();
            for (i, c) in self.cities.iter().enumerate() {
                self.index.insert(c.id.clone(), i);
            }
        }

        self.recompute_extent();
    }

    /// Advance villager motion and building animations by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        // Snapshot city centers so villagers can pick travel destinations.
        let centers: Vec<Vector2> = self.cities.iter().map(|c| c.pos).collect();
        for (ci, city) in self.cities.iter_mut().enumerate() {
            for b in &mut city.buildings {
                if b.live {
                    b.smoke_t += dt;
                }
            }
            for v in &mut city.villagers {
                v.step(dt, ci, &centers);
            }
        }
    }

    fn recompute_extent(&mut self) {
        if self.cities.is_empty() {
            return;
        }
        let mut min = Vector2::new(f32::MAX, f32::MAX);
        let mut max = Vector2::new(f32::MIN, f32::MIN);
        for c in &self.cities {
            min.x = min.x.min(c.pos.x);
            min.y = min.y.min(c.pos.y);
            max.x = max.x.max(c.pos.x);
            max.y = max.y.max(c.pos.y);
        }
        self.extent = (min, max);
    }

    /// Find the city nearest to a world point within `radius`, if any.
    pub fn pick_city(&self, world: Vector2, radius: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, c) in self.cities.iter().enumerate() {
            let d = c.pos.distance_to(world);
            if d <= radius && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Claim the first free grid cell for a new city, jittered for an organic look.
    fn assign_cell(&mut self, id: &str) -> Vector2 {
        let mut cell = 0;
        while self.occupied.contains(&cell) {
            cell += 1;
        }
        self.occupied.insert(cell);
        let col = cell % GRID_COLS;
        let row = cell / GRID_COLS;
        let mut rng = Rng::seeded(&id);
        let jx = rng.range(-CELL * 0.22, CELL * 0.22);
        let jy = rng.range(-CELL * 0.22, CELL * 0.22);
        Vector2::new(col as f32 * CELL + jx, row as f32 * CELL + jy)
    }
}

/// Rebuild a city's stats, buildings and villager population from fresh info,
/// keeping existing building positions and villager motion stable.
fn update_city(city: &mut City, info: &CityInfo, center: Vector2, now: f64) {
    city.name = info.name.clone();
    city.path = info.path.clone();
    city.sessions = info.sessions.clone();
    city.total_messages = info.total_messages();
    city.total_tools = info.total_tool_uses();
    city.live = info.live_sessions(now, LIVE_WINDOW);
    city.last_active = info.last_active();

    // --- Derived civilization metadata ---------------------------------------
    city.tier = Tier::from_score(info.activity_score());
    city.biome = info
        .codebase
        .as_ref()
        .and_then(|c| c.dominant_lang())
        .map(Biome::from_lang)
        .unwrap_or_else(|| Biome::from_hash(&info.id));
    let idle = city.last_active.map(|t| (now - t).max(0.0)).unwrap_or(f64::INFINITY);
    city.season = Season::from_idle(idle);
    city.codebase = info.codebase.clone();
    city.metrics = Metrics::from_city(info, now);
    city.achievements = achievements::unlocked(&city.metrics);

    // --- Buildings: town center + one house per session (capped by tier) -----
    let tier = city.tier;
    let mut existing: HashMap<String, Vector2> =
        city.buildings.iter().map(|b| (b.session_id.clone(), b.pos)).collect();
    let mut buildings = Vec::new();
    buildings.push(Building {
        session_id: format!("__center__{}", info.id),
        pos: center,
        is_town_center: true,
        preset: 0,
        model: info.sessions.first().and_then(|s| s.model.clone()),
        title: Some(info.name.clone()),
        live: city.live > 0,
        messages: city.total_messages,
        smoke_t: 0.0,
        tier,
        tool: Tool::Edit,
    });

    for (i, s) in info.sessions.iter().take(tier.max_houses()).enumerate() {
        // Ring of houses around the town center, deterministic per session.
        let mut rng = Rng::seeded(&(&info.id, &s.id));
        let pos = *existing.entry(s.id.clone()).or_insert_with(|| {
            let ang = (i as f32) * 2.399_963 + rng.range(0.0, 0.6); // golden-angle spread
            let rad = 70.0 + (i as f32) * 7.0 + rng.range(0.0, 24.0);
            Vector2::new(center.x + ang.cos() * rad, center.y + ang.sin() * rad * 0.7 + 46.0)
        });
        buildings.push(Building {
            session_id: s.id.clone(),
            pos,
            is_town_center: false,
            preset: 1 + (hash64(&s.id) as usize % (assets::HOUSES.len() - 1)),
            model: s.model.clone(),
            title: s.title.clone(),
            live: s.is_live(now, LIVE_WINDOW),
            messages: s.total_messages(),
            smoke_t: 0.0,
            tier,
            tool: s.tools.dominant(),
        });
    }
    // Sort by y so nearer buildings draw last (painter's order).
    buildings.sort_by(|a, b| a.pos.y.partial_cmp(&b.pos.y).unwrap_or(std::cmp::Ordering::Equal));
    city.buildings = buildings;

    // --- Villagers: population reflects how busy the project is --------------
    let recently_active =
        city.last_active.map_or(false, |t| now - t < 86_400.0); // active in last day
    let target = if info.total_messages() == 0 {
        0
    } else {
        (1 + city.live * 2 + if recently_active { 1 } else { 0 }).min(MAX_VILLAGERS)
    };

    while city.villagers.len() > target {
        city.villagers.pop();
    }
    while city.villagers.len() < target {
        let idx = city.villagers.len();
        let mut rng = Rng::seeded(&(&info.id, idx, "villager"));
        let model = info.sessions.get(idx % info.sessions.len().max(1)).and_then(|s| s.model.clone());
        let sprite = assets::villager_sprite_for_model(model.as_deref(), idx);
        let off = Vector2::new(rng.range(-60.0, 60.0), rng.range(-30.0, 60.0));
        let start = Vector2::new(center.x + off.x, center.y + off.y + 40.0);
        city.villagers.push(Villager {
            pos: start,
            target: start,
            home: Vector2::new(center.x, center.y + 50.0),
            speed: rng.range(26.0, 46.0) + city.live as f32 * 6.0,
            sprite,
            wander_radius: 70.0 + rng.range(0.0, 50.0),
            anim_t: rng.range(0.0, 6.28),
            on_trip: false,
            rng,
        });
    }
    // If liveliness changed, nudge speeds so live towns bustle.
    for v in &mut city.villagers {
        let bustle = 26.0 + city.live as f32 * 8.0;
        if v.speed < bustle {
            v.speed = bustle;
        }
    }
}

impl Villager {
    fn step(&mut self, dt: f32, my_city: usize, centers: &[Vector2]) {
        self.anim_t += dt * (2.0 + self.speed * 0.05);
        let to = self.target - self.pos;
        let dist = to.length();
        if dist < 4.0 {
            self.pick_target(my_city, centers);
        } else {
            let dir = Vector2::new(to.x / dist, to.y / dist);
            self.pos = self.pos + dir * (self.speed * dt);
        }
    }

    fn pick_target(&mut self, my_city: usize, centers: &[Vector2]) {
        // Returning from a trip → head home.
        if self.on_trip {
            self.on_trip = false;
            self.target = self.home;
            return;
        }
        // Occasionally set off on a trade trip to another project.
        let roll = self.rng.next_f32();
        if roll < 0.06 && centers.len() > 1 {
            let mut dest = self.rng.below(centers.len());
            if dest == my_city {
                dest = (dest + 1) % centers.len();
            }
            self.on_trip = true;
            self.target = Vector2::new(centers[dest].x, centers[dest].y + 50.0);
            return;
        }
        // Otherwise wander near home.
        let a = self.rng.range(0.0, 6.2831);
        let r = self.rng.range(10.0, self.wander_radius);
        self.target = Vector2::new(self.home.x + a.cos() * r, self.home.y + a.sin() * r * 0.6);
    }
}
