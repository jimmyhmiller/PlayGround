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

### 7. Per-island chunking is too granular — the request count is now the bottleneck

Both modes split per island and the document declares the route's chunks (see the Resolved
table), which is correct but coarse-grained in the wrong direction: a page loads one script
per island. cal.com's booker pulls 209 JS files where Turbopack pulls 54.

Production, cold cache, quiet machine, JS on the wire / decoded / files:

| route | monolith (before) | per-island (now) | Turbopack |
|---|---|---|---|
| `/auth/login` | 2.00 MB / 7.93 MB / 2 | 889 KB / 3.33 MB / 109 | **482 KB / 1.48 MB / 30** |
| `/pro/30min` | 2.00 MB / 7.93 MB / 2 | 1.46 MB / 5.14 MB / 209 | **1.02 MB / 3.09 MB / 54** |
| `/apps` | 2.00 MB / 7.93 MB / 2 | 1.16 MB / 4.22 MB / 162 | **813 KB / 2.43 MB / 47** |

Load event: 441 / 391 / 477 ms, against the monolith's 591 / 579 / 497 ms and Turbopack's
89 / 112 / 177 ms. So the split is strictly better than what it replaced on both bytes and
load, and still 1.4-1.8x Turbopack's wire bytes and 3-5x its load event — and the shape of
the gap says requests, not payload.

Turbopack's chunks are grouped by package and app directory (`node_modules_@radix-ui_…`,
`apps_web_app_(booking-page-wrapper)_…`) with a minimum size, which is why it needs 30-54 of
them. Two fixes:

- Coalesce groups below a minimum size, merging by shared reachability label or common path
  prefix. The planner already groups by label, so this is a post-pass over `chunk_plan`.
- HTTP keep-alive in the DEV server, which still sends `Connection: close` on every
  response, so each chunk costs a fresh connection six at a time. Production is served by
  Node's http server, which already keeps them alive.

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
| this commit | A `redirect()` thrown BEHIND a Suspense boundary was dropped on the buffered dev path: cal.com's logged-out `/settings/my-account/profile` answered **200 with a broken document** where the reference answers **307 -> /auth/login**. The redirect reaches the orchestrator only on the stream's END meta, which is genuinely too late for a STREAMED response, but the buffered path drains the whole flight before rendering a document, so nothing had been sent and it was still actionable — it just was not being read. What made this hard to see is what the browser reported: React aborted the errored render, then its dev performance flush measured a component with no end time and threw "Performance.measure: Given attribute end cannot be negative" out of `flushComponentPerformance`, which looks like a timing bug and is collateral damage. Both sides now answer 307 identically. Pinned by a contract test asserting the check sits between draining the flight and rendering the document. |
| this commit | Chunk registration required the runtime to ALREADY EXIST (`if(!__runtime)throw`), which made HTML script order load-bearing and broke three separate ways on cal.com — each one presenting as "no page hydrates". A split chunk now QUEUES its registration when the runtime is not up and the runtime drains the queue before the entry evaluates, the same shape webpack's `webpackChunk.push` has; the boot handshake is symmetric too (the tag sets a flag and calls if defined, the entry checks the flag). Pinned by a node-executed test that loads a chunk and its runtime in BOTH orders and fails on the old behaviour. Result: cal.com's suite went 59 passed / 2 failed -> **60 passed / 1 failed, and that one failure is the `NEXT_PUBLIC_IS_E2E` same-slot test which is red on the reference too** — no diffpack-attributable failures left. |
| this commit | Dev AND production shipped one monolithic client bundle to every route (17.8 MB in dev, 8,116 KB minified in production, the same 7.93 MB decoded on every page). Both now split per island, with the route's client-reference chunks DECLARED BY THE DOCUMENT as plain (deferred, ordered) module scripts ahead of a boot call — the same property `next build` gets from emitting a `<script>` per route chunk. Which references a route resolved is recorded exactly by proxying the object React resolves them through (`moduleMap[clientId]`), so there is no wire parsing and no static over-approximation. Three ordering facts, each one a red run of cal.com's own suite: lazy discovery hydrates islands with no handlers attached (a theme option clicked too early leaves its form clean and the submit button `disabled` for 240s); an entry that awaits its own fetches hydrates after DOMContentLoaded (the theme test reads `<html class>` right after DCL and saw `notranslate light`); and react-dom's `bootstrapModules` stamps `async` on the entry tag, which is unordered against the chunk scripts, so a chunk can run before the runtime it registers into exists and throw (all three theme tests). Result on a quiet machine, same 65 tests: **59 passed / 4 skipped / 2 failed with the split, against 57 / 4 / 4 with the monolith** — the split passes MORE, and the two failure sets barely overlap because the booking-flow family is flaky on both. `client.js` 17.8 MB -> 1.2 MB (dev) and 8,116 KB -> 344 KB (production); per page 7.93 MB -> 3.33 MB decoded on login. |
| this commit | PRODUCTION shipped the same monolith dev did — `client.js` 8,116 KB, the identical 7.93 MB of JS decoded on EVERY route — and the split turned out to need nothing but the pin flip. The chunk-list plumbing built for dev (a proxy over the manifest React resolves references with, the list injected into the document, the browser entry loading it before hydrating, `x-diffpack-chunks` for soft navigation) was diagnosed against a tree that still had the two real bugs: with the seam's chunk table complete and the control boundary's `default` no longer shaken away, React's own flight client preloads a reference's chunk before requiring the module, in the streaming document as well as the buffered one. All of that machinery is DELETED. Production `client.js` 8,116 KB -> 344 KB, and per page: `/auth/login` 2.00 MB -> 165 KB of JS on the wire, `/pro/30min` -> 155 KB, `/apps` -> 1.16 MB. Hydration verified fiber-for-fiber against the reference on all three routes (858/909/3283 with matching interactive-element counts) with zero console errors on two of them. Dev got faster too once the redundant preload went (81 files / 511 ms against 112 / 669 ms). |
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
