# OSS app validation

Real open-source Vite apps as proving grounds for the Vite-drop-in goal
(`diffpack build <root> --vite`). Round 1: 2026-07-21.

## Dev workflow / HMR — where low diff times pay off daily (2026-07-22)

`vite build` parity is the cold-start story; the *daily* story is `vite` the
dev server, and that is where diffpack's sub-10ms incremental diff should be
the visible win. `diffpack dev` now covers both app shapes:

- **Vite HTML-entry SPA** (the everyday `npm create vite` shape): a single
  client environment, diffpack's own static server, WebSocket HMR + React Fast
  Refresh, **no Node process**. Gate: `integration/vite-react-reference/
  dev-check.mjs` — 10/10. A `src/App.tsx` heading edit hot-swaps in place with
  **hook state preserved** (a live `useState` counter keeps its value across the
  update; the same `<h1>` node is reconciled; the page never reloads) with the
  whole incremental rebuild measured live at ~6ms.
- **TanStack Start** (client + Node SSR): state-preserving Fast Refresh on the
  client AND in-process SSR hot reload (the Node PID never restarts). Gate:
  `integration/tanstack-start-reference/dev-check.mjs` — 12/12, ~16ms per edit.

Both gates assert the sharp incremental signals live from the long-lived
process (exactly one module re-transformed, one chunk re-rendered) and a
low-diff-time budget, so a regression that quietly turned a hot update into a
full rebuild or a reload fails the gate. The SPA oracle additionally gates CSS
hot-swap without reload and the add-a-new-file-then-import flow, both
state-preserving.

**Fast Refresh is universal and native.** Per-component instrumentation is oxc's
built-in React Refresh transform (the `react-refresh/babel` equivalent, no Node),
paired with a diffpack accept footer that feeds the boundary check plain
data-property export copies — necessary because diffpack's registry exposes named
exports as live-binding getters, and `@vitejs/plugin-react` v4's runtime rejects
getter descriptors (forcing a reload for every `export const Foo`).

**Spot-checked on a real OSS app (2026-07-22).** `diffpack dev` on
`reduxjs/redux-essentials-example-app` (~1000 modules, RTK, `vite.config.mts`,
plugin-react v4): serves, mounts, and hot-updates a `src/components/Navbar.tsx`
named-export component **in place with state preserved** (no reload) in ~20ms.
Before the transform + getter-copy fix this fell back to a full reload. Promoting
this into a committed, reusable multi-app HMR harness (redux-essentials, markpad,
wall-go, ...) is the next generality step (NEXT_STEPS.md #2).

## Working end-to-end (browser-verified against the Vite reference build)

- **reduxjs/redux-essentials-example-app** — Vite+React+RTK+TS, 1002 modules.
  Renders identically to `vite build` in headless Chromium, zero console
  errors. Surfaced and fixed three capability gaps, each regression-tested:
  dangling-`else` corruption in dead-branch folding (hit RTK's
  `else if (NODE_ENV !== "production")` guards — would have broken every RTK
  app), `browser`-field resolution for client builds, and `resolve.alias`
  (string finds, Vite exact-or-prefix semantics) from vite.config.

## Triaged, ranked by adoption value (pinned commits in the triage report)

| App | Status | Gap(s) |
| --- | --- | --- |
| markpad (CodeMirror markdown editor) | **WORKING** — builds + browser-parity vs reference (13/13 computed properties, zero console errors) | was: Tailwind v4 global entry (wired) + engine coverage (~30 utility families, before/after/focus/breakpoint/group-hover variants, class-candidate dataflow scanner — all landed with 16 new pattern tests) |
| chebyshev-calculator (antd math tool) | **WORKING** — builds (518 modules) + browser-parity vs reference (1600/1600 computed properties across 40 elements, zero diffpack-only console errors) | was: Sass compilation (native `src/sass.rs` landed: variables, nesting/`&`, mixins, `@use`, arithmetic/`calc()` simplification) |
| swift-calc | silent-fallback BUG + out-of-scope gap | raw `@tailwind` v3 directives shipped uncompiled with exit 0 — must be a hard error; PostCSS/Tailwind-v3 pipeline itself likely not worth building |
| app-fire-calculator | 2 gaps | Tailwind v4 global entry; `virtual:pwa-register/react` (vite-plugin-pwa) |
| the-last-pawn | ONE gap left (sass landed: 11 `*.module.scss` + `additionalData` `@use` theme compile; 479/480 computed properties match the reference) | public-rooted URLs (`/fonts/...` in css `url()`, `/favicon-*.png` in index.html) are not rewritten with the non-root `base` the way Vite does — the remaining 404s and the single 2px style delta (font fallback) all trace to it |
| wall-go | **WORKING** — builds, renders, and the emitted AI worker bundle boots as a real module worker (zero console errors; only failure is offline Google Analytics, identical in the reference) | was: non-root base (landed), root-relative alias (landed), Tailwind `@custom-variant` + top-level `@apply`/`@keyframes`/`@media` passthrough (landed), module workers via `new Worker(new URL(...))` — bundled as self-contained `assets/` files, deduped per entry, placeholder-substituted URLs, worker-asset graphs hard-error (landed) |

## Fix queue (ordered)

1. DONE — Tailwind v4 entry as plain global import, compiled at emit.
2. DONE — hard error on Tailwind v3 `@tailwind` directives (was a silent
   broken page with exit 0).
3. Non-root `base` applied to emitted asset/chunk URLs (first wall for 2 of 6
   randomly-drawn apps; GitHub Pages is ubiquitous).
4. Root-relative alias/tsconfig targets (`/src/*`) resolved against the
   project root.
5. DONE — native Sass subset (`src/sass.rs`): variables (+scopes, `!default`),
   nesting with `&` everywhere, nested `@media` bubbling, `@mixin`/`@include`
   (args + defaults), `@use` (namespaces, `as *`, root-relative `/src/...`,
   `_partial` convention), scss `@import` in importer scope with url rebasing,
   arithmetic/`sqrt`, dart-sass `calc()` simplification, Vite
   `css.preprocessorOptions.scss.additionalData` (string form) evaluated from
   vite.config. `.scss` compiles to CSS first, then flows through the existing
   global/module CSS loaders; partials are recorded in `css_source_files` so
   edits invalidate. Everything else (control flow, `@extend`, interpolation,
   placeholders, Sass-only functions, `with (...)`, indented syntax) is a hard
   error naming file + construct. Unlocked chebyshev fully; the-last-pawn now
   only lacks base-prefixing of public-rooted URLs (see table).
6. Module workers (`new Worker(new URL(...), {type:"module"})`).

Also flagged: one NONDETERMINISTIC `Unexpected token` on a valid ESM file
(rc-virtual-list) that vanished on 30+ reruns — possible parse/ingest race;
keep an eye out, add stress coverage when reproducible. UPDATE: not
reproducible in 150 consecutive builds of the 1002-module redux app on the
current binary (post-ordering-fix); left open but downgraded.


## Behavioral parity: ALL FIVE APPS FULLY GREEN (2026-07-22)

`integration/app-parity/` run: every exact step of every app passes all five
channels (normalized DOM, all computed styles, pixel screenshots,
console/network, storage); wall-go's AI-reply invariant holds. Closing the
board took, beyond the wall-go Tailwind coverage round: `assetsInlineLimit`
(sub-4KB assets inline as data URIs in Vite mode — SVGs percent-encoded
byte-identically to Vite's encoder incl. inter-tag whitespace stripping,
everything else base64; generic builds keep hashed files), and harness rule
[N12] (pure-numeric `calc()` in custom properties evaluates — the two
reference bundlers themselves disagree textually there, rolldown-vite
preserving `calc(1.5 / 1)` where esbuild folds it). Known flake, documented:
a blocked third-party analytics retry can land in different steps run-to-run
on wall-go; it is a timing artifact of the sandbox, visible with its
explanation when it occurs.
