# Known issues

Living list. Every entry names its evidence; nothing here is folklore. When an issue is
fixed, move it to the "Resolved" section with the commit rather than deleting it.

## Open

### 1. Residual auto-animate ghost on `next/dynamic` tab swaps (~1 in 3)

cal.com's "Should not allow enabling both recurring event and offer seats" passes ~2 in 3
runs on a diffpack build and 6/6 on the reference. The failure signature is the same
duplicated `[data-testid=offer-seats-toggle]` that the (fixed) Suspense-boundary bug
produced, but it survives the fix: the swap now lands in ONE mutation batch, byte-for-byte
the reference's pattern, yet an `@formkit/auto-animate` exit-ghost with a running WAAPI
animation still appears intermittently. This is a second, distinct defect.

- It is a Heisenbug: every probe that perturbs the test (init scripts, waits between
  steps) suppresses it; Playwright's trace zip truncates when the test dies on timeout.
- Next diagnostic (designed, not yet run): compile a MutationObserver into diffpack's
  emitted client entry so the failing DOM state is captured from inside the page, then
  diff the commit sequence against the reference on the failing iteration only.
- Evidence: day-2 agent report (18-run measurement), STATUS_2026-07-28.md §11.

### 2. No navigation discard queue

Next's `dispatchAction` (`app-router-instance.js:143-154`) marks a pending action
`discarded` when a new NAVIGATE arrives, so a superseded navigation's state is never
applied. diffpack's `navigate()` is a bare async function: two rapid navigations (or a
navigation racing cal.com's effect-driven `router.replace()` in `useTypedQuery`) both
fetch and both commit, in COMPLETION order — last-to-finish wins, not last-requested.

- A discard guard was implemented and measured against known failures: no observable
  effect, so it was deliberately reverted rather than shipped unproven. The divergence is
  real regardless and should be fixed on its own merits, ideally by adopting Next's
  action-queue shape (vendor where separable; MIT attribution).

### 3. `pushState` fires before the commit; Next fires it after

Next pushes history from `HistoryUpdater`'s `useInsertionEffect` after the navigation
commits (`app-router.js:38-71`). diffpack calls `history.pushState` synchronously inside
the transition callback, before the render. During a slow navigation the address bar
leads the DOM. Not implicated in any current test failure; a contract divergence to close
when touching the router, preferably together with issue 2.

### 4. Build CPU +21% / peak RSS +18% after the correctness fixes

Day-2 vs day-1 builds: CPU 23.7s -> 28.6s, peak RSS 1.67GB -> 1.98GB. Attributed to
`pages/api` routes now being bundled into the SSR graph (more modules; map count
763 -> 1,140) plus the styled-jsx / context-module transform scans. Wall time did NOT
regress (parallel headroom absorbed it: 8.6s -> 8.3s), so this is a recovery target, not
a regression alarm. Profile before cutting; the §9 stage profiler and hinted-search work
show where the levers live.

### 5. `ssr: false` dynamic imports use a mounted-gate, not `BailoutToCSR`

Same observable outcome for everything tested (server + first client paint render the
fallback, chunk swaps in after mount), but the mechanism differs from Next's
(`BailoutToCSR` thrown under the boundary). If a future app branches on React's bailout
error specifically, this will surface. Documented in `next_dynamic_shim`'s comment.

### 6. Dev cold start is 5.9s; sub-1-second requires lazy per-route compilation

The remaining cold-start floor is client graph + max(react-server, ssr) built
concurrently + orchestrator boot. Getting under 1s means building routes lazily on first
visit (what `next dev` does), which is an architecture project, deliberately deferred.
The persistent warm-start cache prototype (1.66s restarts) was removed by request —
no caching approaches for now.

### 7. The dev browser bundle is one 17.8 MB file: islands are pinned with static `require`

Every page in dev downloads every `"use client"` island in the app. On cal.com's
`/auth/login` that is the app store, the booker, all of settings and every payment
component — 17.8 MB of JS for a login form.

The cause is one line. `island_pins` (`src/next_adapter.rs`) records each island's
reachability edge as `() => require(path)` inside a never-called closure. `require` is a
STATIC edge, and `Bundler::chunk_plan` splits only on dynamic-import roots, so all 229
lazy islands and their transitive dependencies land in the main chunk. Nothing else is
missing: `client_references_manifest` already fills each entry's `chunks` from the same
chunk plan, the browser seam already installs a real `__webpack_chunk_load__`, and split
chunks already self-register into the runtime. The pins simply never ask for a split.

Measured, with the pins flipped to `import()` (`PinKind::DynamicChunk`, cal.com,
`/auth/login`, cold cache, gzip on both sides):

| | static pins (today) | per-island chunks | Turbopack |
|---|---:|---:|---:|
| main chunk | 17.8 MB | **1.2 MB** | n/a |
| on the wire | 5.16 MB | **1.55 MB** | 1.69 MB |
| decoded | 18.35 MB | **3.75 MB** | 9.20 MB |
| manifest entries carrying chunks | 2 of 469 | 468 of 469 | n/a |

So the split is byte-for-byte better than Turbopack on the same page. It is NOT enabled
because hydration then breaks: the browser leaves each client reference's module chunk
BLOCKED and the render reads it as an element type, dying with "Element type is invalid.
Received a promise that resolves to: undefined. Lazy element type must resolve to a class
or function." Only 3 of the route's references (the error boundaries) ever reach
`requireModule`; chunk loads all succeed (4 issued, 0 failed) and a manual
`__webpack_chunk_load__(chunk)` then `__webpack_require__(id)` in the page resolves the
island correctly — so the seam works and the sequencing does not.

- `await`ing the flight decode before `hydrateRoot` does NOT fix it (tried, reverted).
- The reference does not rely on sequencing at all: `next dev` emits a `<script>` tag per
  route chunk in the document, so every reference is registered before hydration starts.
  The likely fix is the same shape — have the render inject the route's chunk list and
  load it before `hydrateRoot`, rather than depending on React's blocked-chunk path.
- Bisected: with `PinKind::StaticRequire` both `/auth/login` and `/pro/30min` hydrate and
  are interactive; with `DynamicChunk` neither is. Evidence in this session's transcript.

### 8. Dev responses were uncompressed (fixed for assets, open for the document)

Assets now carry gzip and a validator, which took `/auth/login` from 18.86 MB to 5.16 MB
on the wire. Two leaks remain: the Node-forwarded HTML document is still sent
uncompressed (487 KB, against Turbopack's 97 KB gzipped) because it is proxied as a raw
response buffer rather than re-framed, and `write_js` (the Fast Refresh runtime, 21 KB)
has no compression either.

## Reference-side failures (NOT diffpack bugs — do not spend time here)

- `cannot book same slot multiple times`: fails identically on `next start`; the
  quick-availability feature keys off `NEXT_PUBLIC_IS_E2E`, a build-time define neither
  side bakes. Fixable only by building BOTH sides with the flag.
- `can select 'display on booking page' option when multiple organizer input type are
  present`: react-select strict-mode violation in cal.com itself; fails on the reference.
- cal.com's own bugs found en route (upstream material, not ours to fix): a literal
  `${t("switch_monthly")}` string in `Header.tsx` JSX; the same-slot feature-flag issue
  above.

## Resolved (with commit)

| Commit | Issue |
|---|---|
| `b0df5053f` | Dev: hot pushes queued behind the deferred full re-emit (~1s edit-to-DOM at 1Hz cadence; FULLY resolved in `08f72d3c8` by in-place chunk patching: each edit's micro-chunk is spliced into the on-disk host chunks in ~140ms, the full re-emit became 10s-idle compaction, and edit-to-DOM measures 48-53ms flat at every cadence including the old resonance points — no residual) |
| `b0df5053f` | Dev: global CSS edits rebuilt but never reached open browsers in the Next topology (fingerprinted at settle and delivered via the HMR client's in-place link swap) |
| this commit | Dev: a CSS edit inherited the DEFERRED pass's schedule, because the sheet was only ever written by the full re-emit — when compaction moved to a 10s idle (`08f72d3c8`) a css edit went with it and measured **11.49s** edit-to-applied against Turbopack's 7.56s, the one axis where diffpack lost. Now: the sheet is compiled on the edit itself off a chunk-free stylesheet emit (`Bundler::emit_stylesheet_only`), compiled sheets are cached on (entry text, candidates, theme) so compaction no longer repeats the edit's own node compile, and the candidate scan is kept per file so an edit re-tokenizes one file instead of re-reading ~4,000. Measured: **546ms vs Turbopack's 7.35s (13.5x)**, and a Tailwind class added to a *component* reaches the served sheet ~234ms after the short css idle (was ~906ms). |
| this commit | Dev: any deferred pass could hold the loop for the length of a chunk render, so an edit landing inside it waited (the "contention cliff" the 10s idle only made rarer — at a steady ~1/sec cadence it hit every edit: detection lag 400-736ms). Compaction and the stylesheet pass now carry an `EmitCancel` and are ABANDONED within a millisecond or two of a file event, keeping their debt; the idle dropped 10s -> 750ms. Measured at a 1/sec cadence: detection lag <=64ms, edit-to-DOM 48ms flat. |
| `75a2a630f` | `next/dynamic` wrapped every call in `<Suspense>`; Next's `Loadable` rule (`!ssr \|\| !!loading`) gives the default call NO boundary — the extra fallback commit made tab swaps two commits and minted auto-animate ghosts (the deterministic variant) |
| `0c180b351` | Soft navigation swapped in a still-streaming flight; a suspended transition commits nothing — stalled navigations presented as flaky tests |
| `c3af226ac` | Tailwind class-composition helper arguments not treated as class positions |
| `7b7a8e99b` | `"use cache"` reused values across arguments JSON cannot represent (stale event-type poisoning) |
| `0adb497cb` / `8d517a38f` | Static assets lacked validators/gzip; compression ran per-request |
| `e7ec0ad84` | `import()` with a variable specifier was emitted verbatim instead of expanding into a context module (non-English locales 500'd) |
| `38ba92eec` | Page Server Component `searchParams` ignored the request query |
| `771133425` | `pages/api` ran outside the SSR layer; route hooks died across soft nav |
| `e2f5720a9` | Navigation hooks returned fresh identities per render (unbounded render loop on the booker) |
| `24e3f0bc3` (sweep) | styled-jsx not compiled (hydration mismatch on every provider-wrapped route); `NEXT_PUBLIC_*` never inlined (undefined env in client bundles); mixed import+require demand downgrade shook off live exports |
