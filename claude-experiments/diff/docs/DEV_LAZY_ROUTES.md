# Dev cold start: lazy per-route compilation

Written 2026-07-30. Machine: Apple M2 Max, 12 cores, 64 GB, macOS 26.5.2, Node v26.5.0. App under test: cal.com (`/tmp/dpe2e/calcom/apps/web`, 229 page routes, 231 `"use client"` files), pinned at `3894f37e14eae5082770f35ff1fde72110c0e6b6`.

This closes the item `STATUS_2026-07-28.md` §9.1 deferred: *"Getting under 1 second requires building routes lazily on first visit rather than building all three whole-app graphs up front; that is real architecture, deliberately deferred."*

It was built. Sub-1-second was **not** reached, and the honest reason is in section 3: on cal.com the app's own HTTP API surface, which a page is useless without, is ~85% of the server graph, so route laziness cannot remove it from the first build.

## 1. What changed

Three independent changes, each measured separately in section 2.

**(a) Island pins come from the react-server graph, not a filesystem walk.** A pin exists to put a `"use client"` module into the client and SSR graphs so the client reference the flight carries for it resolves. Island discovery is a project walk, which cannot know whether any route references a file, so it pinned every `"use client"` file in the tree: 231 on cal.com, of which the whole app references 101. The react-server graph knows the exact set — a `"use client"` module in it *is* the reference boundary — so its emitted client-references manifest is now recorded (`.diffpack-next/referenced-islands.json`) and the client/ssr passes pin precisely that. This required the build order to change (below). DEV ONLY: a production build pins from its own walk, because it emits the client graph first and must not depend on whether a dev server ever ran in the tree.

**(b) The build order is react-server first, then client and ssr concurrently.** Previously client first, because TanStack Start's server graph imports a virtual module derived from the client build. A Next graph cannot import it, so `register_next_server_virtual_modules` leaves it out and the Next server graphs no longer wait for the client. This is what makes (a) possible, and it also happens to be faster: the react-server graph no longer contends with the ssr graph for cores.

**(c) Routes compile on demand.** `RouteScope` decides which routes the generated entries import. The dev server:

1. discovers the full route/handler/endpoint pattern table (a directory walk, no compilation) and binds the proxy — **accepting connections in ~50 ms**;
2. waits up to 750 ms for a request, which names the route to compile (nothing arrives → compile the whole app, since nobody is waiting);
3. compiles that page plus **all** HTTP endpoints, boots the orchestrator, and releases the request;
4. once the server goes quiet, compiles the rest of the app into a *shadow* output dir, swaps it in by rename, and reloads open browsers once.

A request for a route that is not compiled yet **waits**; it is never answered 404. `DIFFPACK_DEV_LAZY=0` restores the eager whole-app cold start; `DIFFPACK_DEV_LAZY=api` also makes endpoints lazy (measured in section 3 — it is slower on cal.com, and the comment on `first_build_scope` says why).

**(d) The Tailwind candidate scan is parallel and stops stat-ing every file.** Not route laziness, but it was the largest fixed cost left: reading and tokenizing thousands of sources ran serially, and `ScanSkip::skips` canonicalized every directory entry to check whether it was the build's output root — a file can never be a directory, so that check now runs for directories only.

## 2. Measured results

Medians of 3 samples per mode, interleaved, `.diffpack-next` / `.diffpack-output*` wiped before every sample (true cold starts), leaked orchestrators killed before each. Raw samples included; the harness is `scratchpad/devbench.mjs` in the session dir.

| Axis | before (2026-07-28) | eager today | lazy today | lazy vs before |
|---|---:|---:|---:|---|
| Accepting connections | 7,742 ms | 4,934 ms | **49 ms** | — |
| First build + orchestrator boot | 7,742 ms | 4,934 ms | **4,096 ms** | 1.89x |
| **First document 200 (`/auth/login`)** | 9,162 ms | 6,196 ms | **5,379 ms** | **1.70x** |
| Second route 200 (`/pro`), requested immediately | ~9,300 ms | 6,306 ms | 12,705 ms | **0.73x (worse)** |
| Whole app compiled | 7,742 ms | 4,934 ms | 11,687 ms | — |
| Warm `/auth/login` | 30 ms | 27 ms | 42 ms | — |
| Warm `/pro` | 222 ms | 49 ms | 28 ms | — |
| Islands pinned into client+ssr | 231 | 104 | **13** | 17.8x fewer |

Raw samples (ms): eager first-document 6196 / 6310 / 6083; lazy 5433 / 5379 / 5290. Eager first-build 4934 / 5043 / 4829; lazy 4147 / 4096 / 4002. Lazy whole-app-compiled 11687 / 11799 / 11562.

Attribution of the first-document number: 9,162 → 6,196 ms comes from (a) + (b) + (d) and applies to the eager path too; 6,196 → 5,379 ms is route laziness.

Per-graph cost of the first build, lazy (one page + all endpoints): react-server 2,092 ms / 7,706 modules; client 1,276 ms / 3,617 modules; ssr 2,513 ms / 13,933 modules. The same three at whole-app scope: 2,753 / 9,812; 2,389 / 6,098; 3,698 / 15,601.

Tailwind scan, cal.com (`DIFFPACK_PROFILE=1`): candidate scan 911 ms → 348 ms (walk 287 ms, parallel read+tokenize ~60 ms); `emit/css` 1,120 ms → 579 ms. This does not move the dev cold start much because the CSS work already ran under `rayon::join` beside the chunk renders, but it *is* the critical path for a stylesheet edit.

The edit loop is unchanged: island edit 11.6 ms to updated DOM, server-component edit 15.9 ms (`bench-dev-hmr.mjs`, vs Turbopack 21.4 / 36.2).

## 3. Why this is 1.7x and not 8x

The measurement that decided the design: compiling **one page and nothing else** costs 2,992 ms of graph work, and compiling **one page plus every HTTP endpoint** costs 4,606 ms. The whole app is 5,923 ms. So of the ~5.9 s, the 228 pages nobody asked for are worth about 1 s; the app's own API surface is worth about 2 s; the rest is fixed cost.

Endpoints cannot be left out. Two independent reasons, both observed:

* cal.com's login document immediately reads a next-auth session and several tRPC queries. With endpoints uncompiled those requests 404, and the page reports a broken app rather than one still compiling.
* Worse, the **server render itself** calls the app's own API over HTTP. With `DIFFPACK_DEV_LAZY=api` the first document measured **10,521 ms** — slower than eager — because the render sat in flight waiting for an endpoint build that had not started. (That also produced a real deadlock: the fill waited for the server to go idle while the in-flight render waited for the fill. `wait_for_quiet` now returns immediately when anything is already blocked on a wider build, and a test pins it.)

## 4. The costs, stated

* **A second route requested during the fill waits for the whole fill** (12.7 s vs 6.3 s eager). The window is the ~7 s after the first document; after it, every route is instant.
* **Open browsers reload once** when the fill lands, and the log says so. Growing the module set moves runtime module ids, so a page holding the old ids cannot resolve the new bundles' client references.
* **The swap is a rename, with requests held** (~480 ms measured, dominated by killing and rebooting the orchestrator). Chunk fetches bypass the wait and could 404 inside that window; the reload that immediately follows repairs it.
* **The fill costs a full second build** (~6.5 s of CPU). It re-parses everything because a fresh `Bundler` has no module cache and runtime ids are assigned by enumeration over the reachable set, so a wider scope renumbers every module.

## 5. The next lever, if this is worth more

Every cost in section 4 traces to one fact: **runtime module ids are ordinal** (`enumerate()` over the sorted reachable set), so adding a module renumbers all of them. With stable ids:

* the fill could widen the *existing* bundlers via `rebuild_path` on the generated entry, paying only for the new modules instead of ~6.5 s;
* it could emit only the new chunks, leaving existing chunk files byte-identical;
* no orchestrator restart, no browser reload, no shadow dir, no swap window.

That would make lazy compilation strictly better than eager on every axis rather than a trade. It is a real piece of work: id allocation, an incremental entry rewrite, and node-side on-demand chunk loading.
