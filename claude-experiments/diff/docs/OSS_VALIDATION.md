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
| chebyshev-calculator (antd math tool) | ONE gap, proven (517 modules build with scss bypassed) | Sass compilation |
| swift-calc | silent-fallback BUG + out-of-scope gap | raw `@tailwind` v3 directives shipped uncompiled with exit 0 — must be a hard error; PostCSS/Tailwind-v3 pipeline itself likely not worth building |
| app-fire-calculator | 2 gaps | Tailwind v4 global entry; `virtual:pwa-register/react` (vite-plugin-pwa) |
| the-last-pawn | 2 gaps | non-root `base`; sass + CSS-modules composed (`*.module.scss`, `additionalData`) |
| wall-go | 4 gaps | non-root base; Tailwind v4 entry; root-relative alias target `/src/*` not resolved against project root; `new Worker(new URL(...))` |

## Fix queue (ordered)

1. DONE — Tailwind v4 entry as plain global import, compiled at emit.
2. DONE — hard error on Tailwind v3 `@tailwind` directives (was a silent
   broken page with exit 0).
3. Non-root `base` applied to emitted asset/chunk URLs (first wall for 2 of 6
   randomly-drawn apps; GitHub Pages is ubiquitous).
4. Root-relative alias/tsconfig targets (`/src/*`) resolved against the
   project root.
5. Sass (fully unlocks chebyshev; composes with CSS modules for the-last-pawn).
6. Module workers (`new Worker(new URL(...), {type:"module"})`).

Also flagged: one NONDETERMINISTIC `Unexpected token` on a valid ESM file
(rc-virtual-list) that vanished on 30+ reruns — possible parse/ingest race;
keep an eye out, add stress coverage when reproducible.
