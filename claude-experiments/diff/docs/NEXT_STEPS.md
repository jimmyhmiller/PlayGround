# Next steps

Updated: 2026-07-22. Prioritized continuation plan; current state is in
[GENERAL_BUNDLER_STATUS.md](GENERAL_BUNDLER_STATUS.md) and
[OSS_VALIDATION.md](OSS_VALIDATION.md).

## 1. One-command gate + fresh benchmarks (first — cheap, protects everything)

The verification battery — 324 lib tests, clippy on both feature sets, the
memory-accounting guard build, TanStack 13/13 + browser checks, the
create-vite fixture, the conformance suite, and the five-app behavioral
parity suite — is run as a hand-driven sequence today. Consolidate it into a
single `check.sh` so every future change runs the whole wall by default.

The competitive-benchmark tables in COMPETITIVE_BENCHMARKS.md predate the
sass engine, worker scanning, the Tailwind engine growth, and asset
inlining; the doc carries spot-check addenda but the headline tables should
be regenerated (`bench/run.mjs`) so the "fastest cold / fastest
incremental / smaller output" claims are measured against the binary that
actually exists.

## 2. `diffpack dev` for generic Vite apps — the biggest adoption gap

Today diffpack replaces `vite build`; Vite is primarily a **dev tool**. The
long-lived dev server (incremental rebuild, client HMR over WebSocket, React
Fast Refresh, in-process server hot reload) exists but is wired only to the
TanStack `build-app` path. Generalize it to HTML-entry projects:

- serve `index.html` with the injected reload/HMR client;
- module-graph HMR + CSS hot-swap for `diffpack build`-shaped apps;
- close the documented watch-invalidation gaps (glob re-expansion on file
  add/remove, Tailwind candidate rescans, sass partial edits already work);
- dev-mode `import.meta.env` (`DEV: true` path is already correct).

This is the milestone that makes "drop-in replacement" literally true for
the daily workflow.

## 3. Widen the ecosystem net

One triage round produced six apps; five are fully parity-green. The
playbook (triage agent → ranked gap list → fix → parity gate) is proven and
cheap. Run a second batch of 10–15 real Vite apps to either confirm the
surface is genuinely general or produce the next fix list.

Also decide app-fire-calculator's `vite-plugin-pwa` question: a no-op
`virtual:pwa-register` shim yields a working non-offline app but diverges
from reference behavior — make it an explicit opt-in flag or a documented
hard error, never a silent shim.

## 4. Production-grade output details

- **Content-hashed chunk filenames** — emitted entry/chunks are
  `index.js`-style today; real deploys need immutable-cache names (this also
  feeds the manifest/preload story).
- **CSS-side `url()` data-URI inlining** — JS asset imports inline below
  `assetsInlineLimit`; CSS references still emit files (documented
  asymmetry; close it).
- **Multiple HTML entries** (currently a named hard error).
- **Library mode** (`build.lib`) and additional output formats.

## 5. Conformance tail

40/48 on the node-ground-truth suite. Bounded remaining work: the one
diffpack-specific wrong-output (`cjs-esmodule-marker` interop rule — match
esbuild/rolldown's extension-aware `__esModule` handling), factory
`this`/`__filename` CJS ambients, and top-level await across split chunks
(currently an honest build error).

## 6. Strategic arcs (in order)

1. **Nitro via the Vite Environment API / module runner** — closest to
   what exists (TanStack Start already sits on Nitro; the dev server's SSR
   staleness note names the module-runner need).
2. **Next/RSC + persistent caching** — the Turbopack half of the original
   charter: `"use client"`/`"use server"` boundary graphs (the server-fn
   splitting machinery generalizes), a pinned Next reference app with the
   same gate discipline, and an on-disk persistent cache — which doubles as
   another cold-start weapon.

## Standing invariants (do not regress)

- Every measured cold cell leads (realistic-1k 17.4ms vs esbuild 23.7;
  realistic-10k 183ms vs rolldown 263) on the clean no-allocator-override
  binary; wall/memory measured in separate runs (`memory-accounting`
  feature builds only for the guards).
- Incremental edits ~8ms, 1 module re-transformed / 1 chunk re-rendered per
  leaf edit (asserted thesis guards).
- Realistic-corpus output ~1.9x smaller than esbuild (transitive
  statement-level shaking).
- Behavioral parity suite (`integration/app-parity/run.mjs`) exits 0 across
  all five apps, five channels per step.
- Hard errors name the construct; no silent fallbacks; node/Chrome are test
  oracles only, never in the build path.
