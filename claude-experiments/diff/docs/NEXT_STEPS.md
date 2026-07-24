# Next steps

Updated: 2026-07-22. Prioritized continuation plan; current state is in
[GENERAL_BUNDLER_STATUS.md](GENERAL_BUNDLER_STATUS.md) and
[OSS_VALIDATION.md](OSS_VALIDATION.md).

## 1. One-command gate + fresh benchmarks (first — cheap, protects everything)

- DONE — **`./check.sh`**. Tiered one-command gate: Tier 1 Rust (build +
  `cargo test --lib`, now 332/332 with the macOS sass temp-dir failure
  root-caused and fixed + `clippy -D warnings`) is a hard gate; Tier 2 runs the
  SPA and TanStack dev-server HMR oracles and the offline HMR generality harness;
  `--full` adds conformance + five-app behavioral parity + reference build
  acceptance. Missing node/Chrome/deps SKIP with a reason (or FAIL under
  `--strict`). `--fast` is Rust-only. Green by default. Wire CI to
  `./check.sh --strict` (with deps + Chrome provisioned) per PR, and the network
  HMR sweep (`hmr-harness/run-all.sh`) nightly.

Still open here: **regenerate the competitive-benchmark tables.**

- DONE — **regenerated the competitive-benchmark tables**
  (COMPETITIVE_BENCHMARKS.md, full `bench/run.mjs` re-run 2026-07-23 on Apple
  M2 Max, commit `382c3eb0d`). On this machine diffpack now leads **cold** on all
  four synthetic corpora (realistic-1k 28.3 ms vs esbuild 36.5; realistic-10k
  201.4 ms vs rolldown 276), has the lowest peak RSS everywhere (2.3-4.4x), and
  realistic-corpus **output is ~1.9x smaller than esbuild** (transitive shaking,
  now default), and **incremental leads on all four corpora** (2.2-3.9x) once the
  measurement was fixed to be fair: the bench was timing diffpack's edit-to-emit
  wall *through the OS watcher* against competitors' watch-free in-process
  rebuilds, unfairly adding macOS FSEvents detection (~18 ms) to diffpack's number
  alone. `diffpack watch` now reports its own post-detection `rebuild=<ms>` and the
  bench parses that — the same watch-free quantity esbuild's `rebuild()` and
  rolldown's `event.duration` report. Bench harness also runs on macOS now:
  `peakRss` uses `vtime -m`, `collectMeta` no longer requires `/proc/cpuinfo`.

## 2. `diffpack dev` for generic Vite apps — the biggest adoption gap

Today diffpack replaces `vite build`; Vite is primarily a **dev tool**. The
long-lived dev server (incremental rebuild, client HMR over WebSocket, React
Fast Refresh, in-process server hot reload) existed but was wired only to the
TanStack `build-app` path.

- DONE — **generic Vite HTML-entry SPA dev**. `diffpack dev` now detects the
  app kind (a TanStack Start app via `@tanstack/react-start`; otherwise a plain
  Vite SPA rooted at `index.html`) and, for an SPA, runs a single client
  environment with NO Node child: diffpack's own static server serves the
  emitted chunks/assets and the app document (rewritten `index.html` + injected
  HMR preamble), upgrades the WebSocket HMR channel, and on a source edit
  incrementally rebuilds → re-emits → pushes a targeted Fast Refresh update. The
  HMR machinery (`emit_web` with `hmr:true`, `hmr_locate`, the Fast Refresh
  footer) was already environment-agnostic — the SPA path reuses it wholesale.
  Gated by `integration/vite-react-reference/dev-check.mjs`: 10/10, proving a
  `src/App.tsx` heading edit hot-swaps in place with **hook state preserved**
  (a live `useState` counter keeps its value; the same `<h1>` node updates; no
  reload) in ~6ms. Dev-mode `import.meta.env` and development dependency builds
  are selected via `config::set_web_development_mode`.

- DONE — **CSS hot-swap without reload**. A `.css`/`.scss`/`.sass` edit is
  fingerprinted and pushed as a `{type:"css"}` message; the client clones the
  matching `<link>`, cache-busts it, and drops the old node on load — no reload,
  all component state preserved. Gated in the SPA oracle.
- DONE — **new-file / graph-extension handling**. A new file (or any edit that
  grows/shrinks the graph) takes a structural rebuild + reload instead of the
  old hard-error crash; adding a component and importing it just works. Both app
  shapes. Gated in the SPA oracle.
- DONE — **universal state-preserving Fast Refresh**. oxc's native React Refresh
  transform (no Node) now injects per-component `$RefreshReg$`/`$RefreshSig$`, and
  the accept footer feeds the boundary check plain data-property export copies so
  react-refresh v4's getter-descriptor guard no longer forces a reload for
  `export const Foo` components. Proven state-preserving on the real
  redux-essentials app (~1000 modules, plugin-react v4), which previously fell
  back to a reload.

Remaining for full daily-workflow parity:

- DONE — **`index.html` edits** (SPA). The watcher now covers the project root
  non-recursively; an index.html edit re-parses the document and rebuilds the
  served HTML (title/meta/entry) + reload. Config-file edits
  (`vite.config.*`/`package.json`/`tsconfig`) emit a loud warning that live
  re-derivation isn't implemented and the startup config is still in effect
  (restart to apply) — no longer silently ignored or mis-treated as a module.
  New-file glob re-expansion is covered by the new-file structural rebuild
  (re-discovery re-runs `import.meta.glob`).
- **Tailwind candidate rescan on a source edit** — a source edit that introduces
  a new utility class needs the Tailwind entry re-scanned on the incremental
  emit; unverified for the dev hot path. Full live **config re-derivation** (esp.
  a `base` change) is the other deferred piece.
- DONE — committed the reusable **HMR generality harness**
  (`integration/hmr-harness/`, agent-browser-driven, no per-app puppeteer). Three
  diverse real apps pass state-preserving Fast Refresh: vite-react-reference
  (v6/default), redux-essentials (v4.2/named), the-last-pawn (v4.3/default,
  non-root base + sass modules). Adding the-last-pawn surfaced and fixed
  `find_refresh_runtime`'s handling of plugin-react >= 4.3's SPLIT runtime layout
  (react-refresh core + refreshUtils, with a local process shim). Still open:
  **@vitejs/plugin-react-swc** apps (chebyshev-calculator) need the SWC runtime's
  different API surface wired in; more apps are one `.conf` each.

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
   another cold-start weapon. **STARTED** — the sliced plan is in
   [RSC_PLAN.md](RSC_PLAN.md), and Slice 0 (module-directive detection,
   `src/rsc.rs::detect_directive`, AST-based + unit-tested) has landed. Next
   are Slice 1 (`"use server"` generalization of `server_fn.rs`) and Slice 2
   (the `"use client"` boundary + client-reference manifest); both reuse the
   existing deterministic-id/resolver/manifest machinery and gate without the
   flight runtime.

## Standing invariants (do not regress)

- Every measured cold cell leads on the clean no-allocator-override binary
  (Apple M2 Max 2026-07-23: realistic-1k 28.3ms vs esbuild 36.7; realistic-10k
  202.7ms vs rolldown 276.1. Ryzen/Linux earlier: 17.4ms / 183ms). Absolute ms are
  machine-specific; the invariant is the *ranking*. wall/memory measured in
  separate runs (`memory-accounting` feature builds only for the guards).
- Every measured incremental cell leads too (fair measurement — all tools report
  the pure rebuild, excluding OS watch detection; diffpack via its own
  post-detection `rebuild=<ms>`): 2.2-3.9x faster than the best rival. 1 module
  re-transformed / 1 chunk re-rendered per leaf edit (asserted thesis guards).
- Realistic-corpus output ~1.9x smaller than esbuild (transitive
  statement-level shaking).
- Behavioral parity suite (`integration/app-parity/run.mjs`) exits 0 across
  all five apps, five channels per step.
- Hard errors name the construct; no silent fallbacks; node/Chrome are test
  oracles only, never in the build path.
