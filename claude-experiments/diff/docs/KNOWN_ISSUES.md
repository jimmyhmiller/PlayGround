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

### 7. Dev ships one chunk per island now, which is too many chunks

FIXED: the dev browser bundle is no longer one 17.8 MB file (see the Resolved table).
What remains is the granularity. Per-island chunks mean `/auth/login` fetches 112 JS
files and `/apps` 165, against Turbopack's 36 and 49, and the dev server answers each
with `Connection: close` — so every chunk costs a fresh TCP connection, six at a time.

That shows up in two places. On the wire, gzip does worse on many small files than on a
few big ones (`/auth/login`: 2.60 MB across 118 requests vs Turbopack's 1.69 MB across
40, even though we send LESS decoded code — 8.09 MB vs 9.20 MB). And the load event
lands later (669 ms vs 226 ms) despite the smaller payload.

Two fixes, both independent of the split itself:

- Coalesce chunks below a minimum size, which is what webpack and Turbopack do. The
  planner already groups by reachability label; a post-pass merging small groups into
  their common parent would cut request count by roughly an order of magnitude.
- HTTP keep-alive in the dev server. Every response currently ends the connection.

Measured, cold cache, gzip on both sides, after the split:

| route | diffpack wire / decoded / requests | Turbopack wire / decoded / requests |
|---|---|---|
| `/auth/login` | 2.60 MB / 8.09 MB / 118 | 1.69 MB / 9.20 MB / 40 |
| `/pro/30min` | **1.15 MB / 2.15 MB / 34** | 2.75 MB / 15.84 MB / 62 |
| `/apps` | 4.09 MB / 10.51 MB / 251 | 3.04 MB / 15.06 MB / 136 |

### 8. Two dev responses still go out uncompressed

Assets are compressed now (see the Resolved table), but two are not. The Node-forwarded
HTML document is proxied as a raw response buffer rather than re-framed, so it ships
uncompressed — 487 KB against Turbopack's 97 KB gzipped, which makes it the single
biggest item on the wire for a page whose chunks are all small. And `write_js` (the Fast
Refresh runtime, 21 KB) has no compression path at all.

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
| this commit | Dev: the browser bundle was ONE 17.8 MB `client.js`, so every page downloaded every `"use client"` island in the app (cal.com's login form pulled the app store, the booker, all of settings and every payment component) and re-parsed all of it on each full navigation: 693 ms of V8 compile against Turbopack's 33 ms, and no code cache at that size. `island_pins` recorded each island's edge as `() => require(path)` — a STATIC edge, and `chunk_plan` splits only on dynamic-import roots. Dev now pins with `import()` (`PinKind::DynamicChunk`; production keeps the static requires, where whole-graph DCE already shrinks the one chunk). Two real bugs had to be fixed for the split to work at all: the RSC seam's chunk-id -> URL table was built from `chunk_names` (root -> chunk) and so omitted every rootless shared chunk, and a module reached by BOTH a static named import and a dynamic `import()` never became a chunk root and therefore kept only the static importer's names — which shook `default` off the RSC control boundary and killed hydration on every page with "Element type is invalid. Received a promise that resolves to: undefined". Measured: `client.js` 17.8 MB -> 1.2 MB; `/auth/login` 18.86 MB -> 2.60 MB on the wire and 18.35 MB -> 8.09 MB decoded (Turbopack: 1.69 MB / 9.20 MB); `/pro/30min` 1.15 MB / 2.15 MB against Turbopack's 2.75 MB / 15.84 MB. Hydration is fiber-for-fiber with the reference on `/auth/login`, `/pro/30min` and `/apps` (858/859, 891/911, 3283/3284). |
| this commit | Dev served every asset uncompressed and with no validator: `/auth/login` put 18.86 MB on the wire against Turbopack's 2.75 MB, 0 of 8 responses compressed against 50 of 62. Assets now carry gzip (level 1, memoised per file size+mtime) and a weak ETag, so an unchanged chunk revalidates into a 304. |
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
