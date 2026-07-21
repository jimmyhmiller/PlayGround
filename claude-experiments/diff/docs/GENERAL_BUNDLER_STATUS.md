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

- **`import.meta.glob` natively** (`src/import_meta_glob.rs`): build-time
  expansion with Vite semantics — lazy `() => import(...)` per match (each its
  own chunk), `eager`, `import: 'default'/'name'`, `query: '?raw'/'?url'`
  through the existing loaders, pattern arrays and negative patterns, sorted
  keys for byte-reproducible output. Strictly behind the same Vite-convention
  opt-in as `import.meta.env` (`BuildConfig::import_meta_glob`; set by
  `--vite`/`build-app`); generic builds leave the call untouched. Malformed
  calls (non-literal patterns, bare specifiers, unknown options, unsupported
  queries) are file-naming hard errors, never a silent `{}`. Known limitation:
  adding/removing a matched file during watch does not re-expand the importer
  (dev-server glob invalidation is future work).

## Scoreboard vs the stated goals

1. **Faster than every bundler cold** — *not yet true, now measurable*:
   diffpack wins tiny-1k/tiny-10k/realistic-10k, but esbuild wins realistic-1k
   by ~1.5x and Rolldown ties realistic-10k. The output-size half of this gap
   FLIPPED on 2026-07-21: transitive statement-level shaking (liveness
   fixpoint over removable pure declarations, non-exported helpers included)
   took the realistic corpora from ~2.7x larger than esbuild to ~1.9x
   SMALLER, runtime-verified (see the benchmarks doc addendum). Cold wall time then
   fell 2-2.6x on 2026-07-21 (4-thread pool cap removed, pipelined discovery,
   allocation accounting moved behind the `memory-accounting` cargo feature
   so production builds have NO allocator override at all): realistic-1k
   17.4 ms vs esbuild's 23.7, realistic-10k 183 ms vs rolldown's 263 — every
   measured cold cell leads, measured on the clean default binary; memory is
   measured in separate feature-built runs. Pending: a full harness re-run to replace the old
   tables.
2. **Incremental edits far below competitors** — *holds everywhere measured*:
   2.5–3.4x vs the best rival per corpus (diffpack's number even includes
   watch latency rivals exclude), ~20x vs Rolldown on the oracle bench; peak
   RSS lowest in every cell.
3. **Tests that prove correctness** — conformance suite + two pinned real
   apps + thesis guards + byte-parity oracle. The suite already caught two
   real silent-wrong-output bug families on day one.

## Next (ordered)

1. Remaining conformance gaps: `cjs-esmodule-marker` interop rule, factory
   `this`/`__filename` CJS ambients, TLA across split chunks. (Transitive
   statement-level shaking landed 2026-07-21 — `shake_module_code` fixpoint,
   pinned by `shaking_drops_helpers_of_dead_exports_transitively`.)
2. Vite surface: `import.meta.glob`, multiple HTML entries, content-hashed
   chunk names, non-root `base` for asset URLs, dev server for `diffpack
   build` projects (HMR already exists for TanStack dev).
3. Cold-build/bytes frontier: deeper tree shaking toward esbuild's output
   size, then re-measure `bench/`.
4. Nitro via the Vite Environment API / module runner; then the Next/RSC
   (Turbopack) target with a pinned reference app and persistent caching.
