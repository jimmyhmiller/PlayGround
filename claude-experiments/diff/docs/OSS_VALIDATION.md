# OSS app validation

Real open-source Vite apps as proving grounds for the Vite-drop-in goal
(`diffpack build <root> --vite`). Round 1: 2026-07-21.

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
