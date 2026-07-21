# General-bundler status

Updated: 2026-07-21

Diffpack's goal expanded beyond the TanStack Start target: a **general
bundler** that is a drop-in replacement for **Vite without plugins** (popular
plugins get native reimplementations, as TanStack Router/Start and Tailwind v4
already did), later Nitro, and eventually Turbopack. The performance thesis:
fastest cold builds, and incremental edit times substantially below every
competitor. This doc is the umbrella scoreboard; details live in the linked
docs.

## What landed 2026-07-21

- **`diffpack build <root>` — HTML-entry web build** (`src/html_entry.rs`,
  `src/config.rs::derive_web_config`). `index.html` is the entry; the built
  document is rewritten Vite-style. **Vite conventions are strictly opt-in via
  `--vite`** (config `define`/`base` evaluation, native `.env`/`VITE_*` stack
  in `src/env_file.rs`, `import.meta.env`, `public/` passthrough); generic
  builds get none of them.
- **Second pinned reference app**: `integration/vite-react-reference` (stock
  `create-vite` react-ts, versions pinned). Gates: `acceptance.mjs` 15/15 for
  BOTH the Vite reference build and diffpack; `browser-check.mjs` 7/7 in real
  headless Chromium (React mounts, counter state works, images load, CSS
  custom properties resolve, zero console errors/failed requests).
- **Execution-order correctness fix** (was silently wrong): import order is
  now execution order. Two root causes — `collect_dependencies` alphabetized
  specifiers, and `static_execution_order` seeded from every module in id
  order instead of from the entry. Found independently by the conformance
  suite (`order-side-effect-imports`) and live on the vite fixture as an
  inverted CSS cascade (`App.css` overrides losing to `index.css`). Pinned by
  two regression tests in `src/bundler.rs`.
- **Conformance suite** (`conformance/`, `docs/CONFORMANCE.md`): 48 executable
  fixtures, Node ground truth, vs pinned Rolldown 1.2.0 + esbuild 0.28.1.
  Score moved 34 → **40 pass** tonight: the ordering fix, spec-ordered
  namespace keys (`__seal` sorts), CJS-globals modules excluded from flat ESM
  output, and honest handling of top-level await / `import.meta` (below).
  Remaining: 5 wrong-output (4 shared with a reference bundler;
  `cjs-esmodule-marker` is diffpack-specific), 2 honest build errors
  (TLA + code splitting), 1 shared runtime error. Rolldown scores 41,
  esbuild 39 on the same matrix.
- **Top-level await + `import.meta` honesty**: `bundle --format esm` emits
  real ESM where single-chunk TLA and `import.meta` genuinely execute;
  CommonJS output refuses both with module-naming errors instead of emitting
  bundles Node rejects at parse (previously a silent invalid-output bug).
  `import.meta.env` `DEV`/`PROD` now follow `MODE` (dev builds no longer fold
  `if (import.meta.env.DEV)` away).
- **CSS Modules + `@import`/`url()`** (`src/css.rs`): scoping with
  `composes` (incl. cross-file with real graph edges), `:global`/`:local`,
  keyframes; `@import` inlining with media wrapping and dedup; `url()` assets
  content-hashed and rewritten; remote imports hoisted. Unsupported constructs
  are file+construct-naming hard errors. All thesis guards stayed green.
- **Competitive benchmark harness** (`bench/`, `docs/COMPETITIVE_BENCHMARKS.md`):
  vs esbuild, Rolldown, rspack, Vite; tiny + realistic corpora at 1k/10k
  modules plus the real app; runtime-verified pairs only.

## Scoreboard vs the stated goals

1. **Faster than every bundler cold** — *not yet true, now measurable*:
   diffpack wins tiny-1k/tiny-10k/realistic-10k, but esbuild wins realistic-1k
   by ~1.5x and Rolldown ties realistic-10k. Output size also loses ~2.7x to
   esbuild on realistic corpora (conservative tree shaking) — that gap is the
   cold-throughput and bytes frontier.
2. **Incremental edits far below competitors** — *holds everywhere measured*:
   2.5–3.4x vs the best rival per corpus (diffpack's number even includes
   watch latency rivals exclude), ~20x vs Rolldown on the oracle bench; peak
   RSS lowest in every cell.
3. **Tests that prove correctness** — conformance suite + two pinned real
   apps + thesis guards + byte-parity oracle. The suite already caught two
   real silent-wrong-output bug families on day one.

## Next (ordered)

1. Remaining conformance gaps: `cjs-esmodule-marker` interop rule, factory
   `this`/`__filename` CJS ambients, TLA across split chunks.
2. Vite surface: `import.meta.glob`, multiple HTML entries, content-hashed
   chunk names, non-root `base` for asset URLs, dev server for `diffpack
   build` projects (HMR already exists for TanStack dev).
3. Cold-build/bytes frontier: deeper tree shaking toward esbuild's output
   size, then re-measure `bench/`.
4. Nitro via the Vite Environment API / module runner; then the Next/RSC
   (Turbopack) target with a pinned reference app and persistent caching.
