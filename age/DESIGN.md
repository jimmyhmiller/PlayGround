# Age of Models — Design: Scale & Variety

The pitch: **every project is a civilization.** The more (and more recently) you work
in it, the more its city grows, advances through ages, and earns monuments. The *kind*
of work you do and the *codebase itself* shape what the city looks like. Two projects
should never look the same at a glance — you should be able to read your whole dev life
as a map.

This doc is the north star. Not all of it is built yet; see **Status** at the bottom.

---

## The substrate — signals we can actually read

Everything below is derived from real data, never invented. Two sources:

**Session logs** (`~/.claude/projects/*/*.jsonl`)
- sessions; user vs assistant messages; longest single session
- per-tool counts: Bash, Edit, Write, Read, Grep/Glob, Task (subagents), WebFetch/Search, Task* (todos), …
- tokens (from `usage`): input/output/cache
- models used; git branch
- timestamps → first/last active, distinct active **days**, active **hours-of-day**

**The codebase on disk** (each session records its `cwd`)
- language histogram (file extensions) → dominant language
- file count, estimated lines of code, repo bytes
- git commit count, repo age
- presence of tests, README/docs

If a signal isn't available (e.g. repo not scanned), the feature degrades gracefully —
we never fake an achievement or a stat.

---

## 1. Ages — vertical scale

Like Age of Empires' Dark → Imperial progression, a city advances through **tiers** as
its cumulative activity grows. Tier drives footprint, building tier, density, walls, and
how many wonders it can hold. This is where "scale" lives: a 50-message experiment is a
lonely outpost; a 20k-message magnum opus is a walled metropolis.

| Tier | Name        | Rough gate (activity score¹) | Look |
|------|-------------|------------------------------|------|
| 0 | Outpost     | < 50      | one hut, a campfire |
| 1 | Hamlet      | 50–500    | 2–3 houses, a path |
| 2 | Village     | 500–2k    | houses + market + fence |
| 3 | Town        | 2k–8k     | castle keep, more houses, partial wall |
| 4 | City        | 8k–25k    | bigger keep, full wall ring, a wonder slot |
| 5 | Metropolis  | > 25k     | citadel, towers, multiple wonders, dense |

¹ activity score = messages + 2·tool_uses (work is weighted over chatter).

---

## 2. Biomes — horizontal variety ("the different areas")

Each city sits in a biome themed by its **dominant language**. Biome sets the ground
tint and the decoration palette around the city, so the map reads as distinct regions:
the Lisp forests, the Rust forge-hills, the Swift coasts.

| Dominant lang | Biome | Ground | Decorations |
|---------------|-------|--------|-------------|
| Rust          | Forge Hills      | warm slate/orange | rocks, anvils, embers |
| Swift / ObjC  | The Coast        | pale sand/teal    | light, airy, few trees |
| Clojure/Lisp/Beagle | Enchanted Forest | deep green   | dense trees, mushrooms |
| JS/TS         | Trade Port       | bright green      | market stalls, fences |
| Python        | Grass Steppe     | golden green      | scattered trees |
| C / C++       | Ancient Stone    | grey-green        | ruins, boulders |
| Go            | Open Plains      | teal-green        | sparse, tidy |
| Markdown/docs | Scholars' Vale   | parchment         | signposts, quiet |
| (unknown/mixed) | Heartland      | default green     | balanced |

Biome is **stable** per project (language rarely flips), so cities keep their identity.
Even unscanned projects get a biome via a stable hash so nothing looks empty.

---

## 3. Seasons — temporal variety

Recency tints the whole city, so the map shows the *rhythm* of your attention:

| Last active | Season | Effect |
|-------------|--------|--------|
| < 10 min    | High Summer | full color, banners, smoke, busy villagers |
| < 1 day     | Summer      | full color |
| 1–7 days    | Late Summer | slightly warm |
| 1–4 weeks   | Autumn      | orange/desaturated tint, leaves |
| 1–6 months  | Winter      | cool/pale tint, fewer villagers |
| > 6 months  | Dormant     | grey, overgrown, ruins creeping in |

Biome × Season is the core variety engine: a freshly-active Rust project and a dormant
one read completely differently.

---

## 4. Buildings — work-type variety

The *mix of tools* a project uses determines which specialized buildings appear, so a
research-heavy project looks different from a build-heavy one. Houses (sessions) get a
type from their dominant tool:

| Dominant tool | Building | Flavor |
|---------------|----------|--------|
| Bash          | Forge / Smithy | running commands = crafting; smoke, anvil |
| Edit / Write  | Workshop (house) | building code; scaffolding |
| Read / Grep   | Library / Archive | research; signpost, quiet |
| Task (subagents) | Barracks | deploying agent armies; banners, soldiers |
| WebFetch/Search | Harbor / Trade post | reaching outside; market |
| Todo/planning | Town Hall | planning office |

The town center itself upgrades with tier (hut → keep → citadel).

---

## 5. Units

- **Model → unit class:** Opus = knight, Sonnet = peasant, Haiku = ranger, others = mage.
- **Subagents (Task) → squads** that march out from a barracks on "expeditions."
- **Specialists** cluster near matching buildings (smiths by forges, scholars by libraries).
- Live session → villagers move with purpose (faster, more of them); dormant → still.

---

## 6. Resources

The four classic resources, mapped to real metrics and shown as stockpiles + a readout:

- **Food** = messages (the conversation that feeds the town)
- **Wood** = edits + writes (construction)
- **Gold** = tokens (raw spend)
- **Stone** = git commits (permanence)

---

## 7. Achievements & Monuments — "achievements for a codebase"

Achievements are earned from real metrics. Each unlocked one raises a **monument** in the
city (a statue, obelisk, treasure, anvil…) and lists in the city inspector. They're how a
codebase shows off. Grouped:

**Activity** — Founding (1st session) · Hamlet (10 sessions) · Bustling (50) ·
Great Capital (150) · Chatty (1k msgs) · Verbose (5k) · Epic Saga (20k) ·
Marathon (a single 400-msg session)

**Craft** — Smith (500 Bash) · Architect (1k edits) · Scholar (1k reads) ·
Seeker (300 searches) · General (30 subagents) · Navigator (50 web) · Toolsmith (5k tools)

**Codebase** — Sapling (repo found) · Grove (10k LOC) · Old Growth (100k LOC) ·
Sprawl (1k files) · Polyglot (4+ languages) · Committed (200 commits) ·
Prolific (1k commits) · Ancient (6+ months old) · Tested (has tests) · Documented (has README)

**Mastery** — Opus/Sonnet/Haiku adept · Triumvirate (all three models)

**Time** — Night Owl (work 0–4am) · Early Bird (5–8am) · Veteran (active 30+ distinct days)

**Wealth** — Rich (10M tokens) · Tycoon (100M tokens)

We only ship achievements we can truthfully evaluate from available data; if a signal is
missing the achievement simply can't unlock (it never silently passes).

---

## 8. Events (future)

- Live session → **festival** (extra flags, fireworks, bustle).
- Recent errors/aborts in the log → a building **on fire** / smoking.
- A brand-new project → a **founding** animation.
- A very long running session → an **epic** marker.

---

## Status

- ✅ Base map: projects → cities, sessions → buildings, villagers, live banners, inspector.
- ✅ Data enrichment: per-tool counts, tokens, active days/hours, and a cached repo
  scan (languages, files, LOC, commits, tests/README).
- ✅ **Ages (tiers)** — cities grow Outpost → Metropolis, scaling footprint, building
  count, town-center keep, and walls.
- ✅ **Biomes** — dominant language themes each city's plaza colour + decorations.
- ✅ **Seasons** — recency tints the whole city (High Summer → Dormant).
- ✅ **Tool-typed buildings** — dominant tool adds a prop (forge anvil, library
  signpost, barracks banner, harbor stall, …).
- ✅ **Achievements + monuments** — 34-achievement catalog evaluated from real
  metrics; unlocked ones become a trophy shelf in front of the city and a badge
  wall in the inspector.
- ✅ **Resources readout** — food/wood/gold/stone in the inspector.
- 🧊 Later: subagent squads/expeditions, fire-on-error events, festivals, minimap, sound.
