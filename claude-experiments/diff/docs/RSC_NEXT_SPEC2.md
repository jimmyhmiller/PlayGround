# RSC_NEXT_SPEC2 — Closing the remaining Next.js app-router gaps to a working stock `create-next-app`, plus the Turbopack benchmark

Authoritative implementation spec. Supersedes the "REMAINING gaps" half of
`docs/RSC_NEXT_GAP.md` §4. The spine (three RSC graphs, both manifests, the
`__webpack_*` seam, whole-document SSR-of-flight, browser hydration, the
`"use server"` action round-trip) is **built and gated** — do NOT redo it. This
spec only closes the framework-mapping layer on top of it.

Everything here is native Rust in the build path; **Node + Chrome are TEST
ORACLES only**. Browser work uses `agent-browser` (absolute paths). No stub may
hide failure — every placeholder throws a clear error naming what is missing.
`cargo build --release` + `cargo test --lib` + `clippy -D warnings` stay green
after every slice, and the full `scripts/rsc/next-check.sh` battery only grows.

---

## 0. Ground truth this spec builds on (verified in-repo)

- **Orchestrator** `scripts/rsc/next-server.mjs` (166 lines): `runReactServer(args, stdinBody)`
  at L83 spawns `node rsc-render/server.mjs <args>` **fresh per request**, `stdio:["pipe","pipe","pipe"]`,
  rejects on non-zero exit, resolves stdout (the flight bytes). The `POST /_action/`
  branch (L118) already passes a stdin body. The GET route-render branch (L145) computes
  the per-route flight for **every** GET via `runReactServer(["render", pathname, manifest])`
  then SSRs it with `renderFlightToDocument`. The static-asset check (L133) keys on
  `url.pathname`, so a query string never collides with an asset. MIME map at L105.
- **Adapter** `src/next_adapter.rs` (998 lines): `detect_app_router` (L49, PRIVATE),
  `Route` struct (L144), `discover_routes`/`discover_routes_dir` (L158/166) — today
  **skips** `[param]`/`@slot` dirs and does NOT collect loading/error/not-found;
  `scan_client_islands`/`scan_islands_dir` (L306/314) discover `"use client"` modules
  under `app/` **but explicitly skip `.diffpack-next/`**; `configure(root, environment)`
  (L377) scaffolds `.diffpack-next/` via `write_if_changed`, pins islands into client+ssr
  graphs, returns an `AppConfig` with `hmr=false`, `NODE_ENV="production"`, and
  `production` resolve conditions; the generated entries `rsc_entry_module` (L513),
  `ssr_entry_module` (L673), `client_entry_module` (L754); `island_pins` (L656); shims
  `next_link_shim` (L808, a plain server `<a>`, no directive), `next_image_shim` (L822,
  plain `<img>`), `next_navigation_shim` (L836, `useParams`/`redirect`/`notFound`
  hard-error), `next_headers_shim` (L883, hard-errors); aliases at L451-454.
- **Client seam**: `Target::Client` installs `globalThis.__webpack_require__/.u/__webpack_chunk_load__`
  automatically (`src/bundler.rs`). `react-server-dom-webpack/client` (browser) exports
  `createFromFetch, createFromReadableStream, createServerReference, encodeReply,
  registerServerReference, createTemporaryReferenceSet`. `createFromFetch(fetch(...),{callServer})`
  resolves client refs through that global seam with **no manifest** — the same transport
  the action round-trip already uses (`src/rsc_runtime/call_server.js`).
- **Directive detection is per-module-source** (`src/transform.rs:206`, `detect_directive(path, source)`),
  path/alias-agnostic: a generated `"use client"` shim under `.diffpack-next/` becomes a
  flight client reference exactly like `app/Counter.tsx`, keyed by its canonical path
  (`module_reference_id`, `src/rsc.rs:90`).
- **React 19 head hoisting** of `<title>/<meta>/<link>/<style>` from anywhere in the tree
  is proven (gates 1b/1c). `react-dom` `preload(src,{as,imageSrcSet,...})` hoists a
  `<link rel=preload>`.
- **AsyncLocalStorage propagates through the flight render** into async Server Components
  when the store wraps BOTH `renderToReadableStream(...)` AND the stream drain (verified
  probe: `ALS-PROPAGATES: YES`).
- **`redirect()`/`notFound()`** are digest-throwing errors captured by
  `renderToReadableStream`'s `onError`; the stream drains to completion, the child exits 0.
  Exact digests: `NEXT_REDIRECT;<type>;<url>;<status>;` (type default `replace`, status 307);
  `NEXT_HTTP_ERROR_FALLBACK;404`.
- **Bench**: `scripts/bench-next.mjs` already exists and runs green; `bench/util.mjs`
  exports `median, round, timeProcess, peakRss, outputBytes, removePaths`.

**Shared infrastructure decision (reconciles dives #2 and #4.7):** dynamic-segment
matching, the fd-3 control channel, and params delivery are ONE mechanism, built once
in Slice 2 and extended in Slice 3 — not two parallel implementations. Client-side
hooks (`useParams`/`usePathname`/`useSearchParams`) read **React contexts** fed
identically on SSR and client (dive #3's approach), NOT `window` globals — window
globals don't exist during SSR and cause hydration mismatch.

---

## Ordered slices

Each slice is independently landable and independently gated. Order respects
dependencies: Slice 1 is self-contained; Slice 2 establishes routing + the fd-3
control channel; Slice 3 extends both with the request context; Slice 4 is
independent; Slice 5 needs a buildable app; Slice 6 is the culmination; Slice 7
is measurement.

| # | Gap | Depends on |
|---|-----|-----------|
| 1 | 4.6 client-side soft navigation | spine only |
| 2 | 4.5-tail dynamic segments + loading/error/not-found + real 404 | spine only |
| 3 | 4.7 next/navigation + next/headers request context | Slice 2 (segments, fd-3) |
| 4 | 4.2 next/image fidelity | spine only |
| 5 | 4.8 the Next dev server | Slices 1–4 buildable |
| 6 | stock-template culmination (unmodified `create-next-app`) | Slices 1–4 |
| 7 | Turbopack benchmark | Slice 6 (and independent) |

---

## Slice 1 — Client-side soft navigation (gap 4.6)

**Goal.** `next/link` clicks fetch the target route's flight and diff-render it into
the live tree instead of doing a full document load; `history.back()` does the same.
Built entirely on the already-proven flight transport (`createFromFetch` + `callServer`
over the client `__webpack_*` seam) — no new protocol.

### Files & changes

1. **`scripts/rsc/next-server.mjs`** — raw-flight endpoint. In the GET route-render
   branch (after the static-asset check, before the SSR call ~L145):
   ```js
   const flight = await runReactServer(["render", url.pathname, clientManifestPath]);
   if (url.searchParams.has("__rsc")) {
     res.writeHead(200, { "content-type": "text/x-component" });
     res.end(flight);
     return;
   }
   const doc = await renderFlightToDocument(new Uint8Array(flight), serverConsumerManifest, flight.toString("base64"));
   res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
   res.end(doc);
   ```
   Zero new render logic — the flight is already computed for every GET. The static
   check runs first so `?__rsc` never shadows an asset.

2. **`src/next_adapter.rs` `next_link_shim()`** (L808) — make it a `"use client"`
   intercepting component. Prepend `"use client";\n`. Render the same `<a href>` but
   with an `onClick` that, for a plain left-click (`e.button===0`, no meta/ctrl/shift/alt,
   not already `defaultPrevented`) on an internal (`/`-prefixed string) href, calls the
   user's `onClick` first, then `e.preventDefault()` + `window.__diffpack_navigate(resolved)`.
   Graceful real fallback (NO preventDefault → real full load) for modified clicks,
   external/non-string hrefs, or when `window.__diffpack_navigate` is absent (pre-hydration).
   Keep destructuring `replace/prefetch/scroll` out of `...rest`. Update the shim's
   comment (it now DOES soft-navigate).

3. **`src/next_adapter.rs` `configure()`** (L377) — pin the link shim into the client &
   ssr graphs (NOT the react-server graph, where it must stay a client reference). Because
   `scan_islands_dir` skips `.diffpack-next/`, add it explicitly: after writing
   `shims_dir.join("link.tsx")`, compute its canonical path and append it to the `islands`
   vec that is passed to `ssr_entry_module`/`client_entry_module` and `island_pins`. This
   makes the flight's `next/link` client reference resolve to real code and hydrate. The
   client-reference id is `module_reference_id` = `canonicalize(link.tsx)`, the same path
   the react-server render resolves the `next/link` alias to → manifest ids match.

4. **`src/next_adapter.rs` `client_entry_module()`** (L754) — replace the `Root`/`use(tree)`
   + `hydrateRoot(document, createElement(Root,{tree}))` tail with a **client Router**:
   - extend the react import to `{ use, useState, useEffect, useTransition, createElement }`;
   - `Router({ initialTree })`: `const [tree,setTree]=useState(initialTree); const [,startTransition]=useTransition();`
   - in `useEffect(()=>{...},[])` define
     ```js
     function navigate(to, { push = true } = {}) {
       const href = typeof to === "string" ? to : to.href;
       const replace = typeof to === "object" && to.replace;
       const sep = href.includes("?") ? "&" : "?";
       const next = createFromReadableStream((fetch(href + sep + "__rsc=1")).then(r=>r.body).catch()...); // use createFromFetch
       startTransition(() => {
         setTree(next);
         if (push) history[replace ? "replaceState" : "pushState"](null, "", href);
       });
     }
     window.__diffpack_navigate = navigate;
     const onpop = () => navigate(location.pathname + location.search, { push: false });
     window.addEventListener("popstate", onpop);
     return () => window.removeEventListener("popstate", onpop);
     ```
     Implement `next` with `createFromFetch(fetch(href + sep + "__rsc=1"), { callServer })`
     (already-imported `callServer`).
   - `Router` returns `use(tree)`.
   - Boot: `hydrateRoot(document, createElement(Router, { initialTree: createFromReadableStream(stream, { callServer }) }))`.
   `use(pendingThenable)` suspends; `startTransition` keeps the old document visible until
   the new flight resolves (soft-nav UX). React 19 reconciles the swapped `<html>/<head>/<body>`
   in place. Update the adapter unit test's `hydrateRoot(document` assertion to
   `createElement(Router` if the exact string changes.

5. **`integration/next-app-router/app/about/page.tsx`** — add `import Link from "next/link"`
   and a `<Link id="home-link" href="/">Home</Link>` inside `<main id="about">`, keeping the
   existing "About page (app-router route)" marker, so soft-nav is testable both directions.

6. **`docs/RSC_NEXT_GAP.md` §4.6** → CLOSED (raw-flight `?__rsc=1`, `"use client"` intercepting
   `next/link`, `useState`+`useTransition` Router swapping the whole-document flight tree).
   Note follow-ups OUT of scope: prefetch cache, scroll restoration, `useRouter().push`
   delegating to `window.__diffpack_navigate`.

### GATE (add Gate 5 to `scripts/rsc/next-check.sh`)

- **(a) Flight round-trip (curl):** `flight="$(curl -s "$base/about?__rsc=1")"` — assert it
  does NOT contain `<!DOCTYPE html>`, is non-empty (raw flight rows), and
  `curl -sI "$base/about?__rsc=1" | grep -qi 'content-type: text/x-component'`. Assert the
  SAME path WITHOUT `?__rsc=1` still returns the full HTML document (no gate-1d regression).
- **(b) Real-browser soft nav (agent-browser):** `agent-browser open "$base/"`; wait `#about-link`;
  set sentinel `agent-browser eval 'window.__softnav="kept"'`; `agent-browser click "#about-link"`;
  poll for `#about` text; assert (single eval)
  `window.__softnav==="kept" && location.pathname==="/about" && !!document.querySelector("#app-shell")`
  — the sentinel surviving proves NO reload; `#app-shell` proves the root layout was preserved
  (diff-render). Then `agent-browser eval 'history.back()'`; poll for `#heading` "Server:from-server";
  assert `window.__softnav==="kept" && location.pathname==="/"`.

Exit 0 = per-route flight fetched over `?__rsc=1` and diff-rendered into the live tree
without a reload, forward via intercepted `next/link` and backward via history.

**Risk & fallback.** If the aliased `"use client"` Link ever regresses, the drop-in
fallback is **event delegation**: the Router installs one `document.addEventListener("click", …)`
intercepting internal `<a>` clicks, and Link stays a plain server `<a>` with zero
graph/manifest changes. Functionally identical; keep it in pocket, do not build both.

---

## Slice 2 — Dynamic segments + loading/error/not-found + a real 404 (gap 4.5-tail)

**Goal.** `app/blog/[slug]/page.tsx` matches `/blog/hello` with `params.slug==="hello"`;
`loading.tsx`/`error.tsx`/`not-found.tsx` conventions compose around the segment; an
unmatched path returns a real HTTP 404 rendering `app/not-found.tsx` (or a built-in default)
under the root layout. Establishes the **fd-3 control channel** and **params delivery** that
Slice 3 extends.

### Files & changes

1. **`src/next_adapter.rs` route model.** Add
   `enum Seg { Static(String), Dynamic(String), CatchAll(String), OptionalCatchAll(String) }`
   and `parse_segment(&str)->Seg` (`[x]`→Dynamic, `[...x]`→CatchAll, `[[...x]]`→OptionalCatchAll,
   `(group)` stripped, `@slot`/`(.)` still SKIPPED — documented gap, never mis-served). `Route`
   gains `segments: Vec<Seg>` (keep a display `url_path`), PARALLEL boundary slots
   `loading/error/not_found: Vec<Option<PathBuf>>` indexed like the layout chain plus leaf-level
   ones; `discover_routes` returns a top-level `app_not_found: Option<PathBuf>`.

2. **`discover_routes_dir`** (L166): remove the `has_dynamic` skip; map surviving path
   components through `parse_segment`; in each dir `first_existing(dir, "loading"|"error"|"not-found")`
   into the parallel slots; detect app-root `not-found`. **Sort routes by specificity**:
   fewest catch-alls, then fewest dynamics, then longer segment count, then lexicographic —
   so a literal `/blog/new` beats `/blog/[slug]` (add a unit test).

3. **`rsc_entry_module`** (L513):
   - intern loading/error/not-found modules alongside page/layouts.
   - Emit each ROUTE with its `segments` pattern + per-level boundary indices.
   - Add JS `matchRoute(pathname)`: split into non-empty parts; Static matches one part
     exactly, Dynamic one part, CatchAll the tail; returns `{ route, params }` or `null`.
   - Rewrite `documentTree(pathname)` to return `{ tree, status, params }`:
     `const m = matchRoute(pathname); if (!m) return { tree: notFoundTree(), status: 404, params: {} };`
     where `notFoundTree()` wraps `app/not-found.tsx` (or a default
     `createElement('main',{id:'not-found'},'404 — This page could not be found.')`) in the
     root layout + head items. On hit compose boundaries **inner→outer**:
     `children = createElement(page, { params: Promise.resolve(m.params), searchParams: Promise.resolve({}) })`;
     wrap with NotFoundBoundary (if present), then `<Suspense fallback={<Loading/>}>` (if `loading`),
     then the generated client `<ErrorBoundary fallback={UserError}>` (if `error`), then each
     `<Layout params>` root-last. loading/not-found compose directly in the server tree; error
     uses a generated `"use client"` wrapper (below).
   - `main()` render op:
     ```js
     const { tree, status, params } = documentTree(pathname);
     writeMeta({ status: status || 200, params: params || {} });
     const stream = renderToReadableStream(tree, bundlerConfig);
     await streamToStdout(stream);
     ```
     `function writeMeta(m){ try { require('node:fs').writeSync(3, JSON.stringify(m)); } catch {} }`
     — guarded so a standalone/action invocation without fd 3 no-ops (a clear path, not a silent stub).

4. **Generated `.diffpack-next/error-boundary.tsx`** via a new `error_boundary_module()`:
   `"use client";` a class component with `getDerivedStateFromError` rendering
   `createElement(props.fallback, { error: this.state.error, reset: () => this.setState({error:null}) })`
   else `props.children`. Write it in `configure()` and — because `scan_islands_dir` skips
   `.diffpack-next/` — push its canonical path into the `islands` vec BEFORE `island_pins`
   so client+ssr bundle+register it and its client reference resolves.

5. **`next_navigation_shim`** (L836): rewrite `useParams()` body from the hard-error to
   `if (typeof window === 'undefined') throw new Error('useParams() during server render unsupported (client only)'); return window.__DIFFPACK_PARAMS__ || {}`. (Slice 3 replaces this
   window read with a React context; this interim keeps the shim honest.)

6. **`ssr_entry_module`** (L673) + `renderFlightToDocument`: add a `params` argument; append
   `window.__DIFFPACK_PARAMS__=<json>;` to `bootstrapScriptContent` so the browser has params.

7. **`scripts/rsc/next-server.mjs`**: spawn with `stdio:["pipe","pipe","pipe","pipe"]`; collect
   `child.stdio[3]` into a `meta` buffer; on close parse `{status,params}` (default `{status:200,params:{}}`);
   `runReactServer` resolves `{ flight, status, params }`. In the GET branch:
   `const {flight,status,params}=await runReactServer([...]); ... res.writeHead(status||200, {...}); res.end(await renderFlightToDocument(new Uint8Array(flight), serverConsumerManifest, flight.toString('base64'), params));`
   The `?__rsc=1` branch (Slice 1) returns `flight` regardless of status (raw flight is
   status-agnostic; the client Router handles a 404 body tree). The action path leaves fd-3 unused.
   **Do not reject on non-zero exit for a 404** — the child exits 0; status travels only on fd-3.

8. **Fixture** `integration/next-app-router/app`:
   - `blog/[slug]/page.tsx`: `export default async function Post({params}:{params:Promise<{slug:string}>}){const {slug}=await params; return <main id="post">post: {slug}</main>}`
   - `blog/[slug]/loading.tsx`: `export default function Loading(){return <main id="post-loading">Loading post…</main>}`
   - `error-demo/page.tsx`: server component that `throw new Error('boom-from-server')`
   - `error-demo/error.tsx`: `"use client"; export default function Err({error,reset}){return <main id="error-demo">error caught: {error.message}<button id="reset" onClick={reset}>retry</button></main>}`
   - `not-found.tsx`: `export default function NotFound(){return <main id="not-found">404 — page not found</main>}`
   Confirm the fixture still passes `next build` (stock conventions only).

9. **`docs/RSC_NEXT_GAP.md` §4.5** → dynamic segments + loading/error/not-found + 404 CLOSED;
   `@slot`/`(.)` parallel/intercepting routes remain documented gaps (still skipped, never mis-served).

### GATE (add to `scripts/rsc/next-check.sh`)

- **1e (dynamic segment):** `curl $base/blog/hello` renders a full document under `id="app-shell"`
  containing `post: hello`; `curl $base/blog/world` contains `post: world` (per-request extraction,
  not a hardcode).
- **1f (real 404):** `curl -s -o /dev/null -w '%{http_code}' $base/no/such/path` == `404`; body
  contains `404 — page not found`; body does NOT contain `from-server` (no fall-through to index).
- **1g (error boundary):** `curl $base/error-demo` returns HTTP 200 full document containing
  `error caught: boom-from-server` (throwing Server Component caught by the client boundary, render
  did not crash, child exit 0). Also grep the generated `.diffpack-next/rsc-entry.tsx` for the
  error-boundary import + `Suspense` usage (structural proof of composition).
- **1h (loading non-breaking):** generated rsc-entry contains `Suspense` wrapping the blog page;
  `curl $base/blog/hello` still renders `post: hello`.
- Rust: extend the `next_adapter` unit test — a `[slug]` dir yields a `Dynamic` segment and the
  generated source contains `matchRoute`.

**Honesty constraints (do NOT violate).**
- The SSR entry uses `onAllReady` (waits for all Suspense), so a `loading` fallback is NEVER in the
  final HTML — gate `loading` **structurally + non-breaking only**; true fallback-in-HTML is streaming
  (Slice 5 / gap 4.8), documented as such.
- `error.tsx` MUST exist for `error-demo` — a throw with NO enclosing client boundary makes
  `renderToReadableStream` call `onError` and the child fails; the 404 path must be handled by
  `matchRoute`-miss → not-found tree, never by throwing.
- Pass `params` as `Promise.resolve(params)`; the fixture page is `async` and `await`s it (Next 16 shape).

---

## Slice 3 — `next/navigation` + `next/headers` real request context (gap 4.7)

**Goal.** `redirect('/about')` issues a real HTTP 307; `await cookies()`/`await headers()` read the
actual request inside async Server Components; `useParams()/usePathname()/useSearchParams()` return the
right values on BOTH SSR and client with zero hydration warnings. Reuses Slice 2's fd-3 channel and
segment matching; adds an AsyncLocalStorage request context and React contexts for the client hooks.

### Files & changes

1. **Generated `.diffpack-next/request-context.ts`** (new `request_context_module()`):
   `import { AsyncLocalStorage } from 'node:async_hooks'; export const requestAls = new AsyncLocalStorage();`
   Because rsc-entry + the shims are all bundled into the ONE react-server graph, they share this
   single instance (Next's `workUnitAsyncStorage` analogue). Verify the graph dedupes it (same
   absolute path from both importers).

2. **Generated `.diffpack-next/hooks-context.ts`** (new `hooks_context_module()`): three
   `React.createContext` exports — `PathParamsContext`, `PathnameContext`, `SearchParamsContext`.

3. **`rsc_entry_module`** render op — read a request-context JSON from **stdin** before rendering
   (`{ url, headers: [[k,v]...], cookie }`); match the pattern to bind `params`; build
   `store = { url: new URL(url), headers: new Headers(headers), cookieHeader: cookie, params }`;
   wrap render AND drain in one store:
   ```js
   let control = {};
   await requestAls.run(store, async () => {
     const stream = renderToReadableStream(tree, bundlerConfig, {
       onError(e){ const d=e&&e.digest||""; if(d.startsWith("NEXT_REDIRECT;")){const p=d.split(";"); control.redirect=p.slice(2,-2).join(";"); control.status=Number(p.at(-2))||307;} else if(d==="NEXT_HTTP_ERROR_FALLBACK;404"){control.notFound=true;} return d||undefined; }
     });
     await streamToStdout(stream);
   });
   writeMeta({ status: control.status || status || 200, params, ...control });
   ```
   The store MUST enclose both the call and the drain (verified requirement) or late async
   components lose it. Pass `params: Promise.resolve(params)` to the page AND to layouts at/below
   the dynamic level.

4. **`next_headers_shim`** (L883): replace the throwing bodies with **async** functions reading
   `requestAls.getStore()` (imported from `../request-context`), hard-erroring ONLY when the store
   is absent (called outside a request):
   - `cookies()` → a `RequestCookies`-like object (`.get(name)→{name,value}|undefined`, `.getAll()`,
     `.has()`) parsed from `store.cookieHeader`;
   - `headers()` → `store.headers` (read-only `Headers`);
   - `draftMode()` → `{ isEnabled:false, enable(){throw}, disable(){throw} }` (faithful: no draft cookie).
   Keep them `async` — Next 16 requires `await`.

5. **`next_navigation_shim`** (L836): SERVER branch (`typeof window==='undefined'`):
   `redirect(url,type)` throws `Object.assign(new Error('NEXT_REDIRECT'),{digest:\`NEXT_REDIRECT;${type??'replace'};${url};307;\`})`;
   `notFound()` throws `{digest:'NEXT_HTTP_ERROR_FALLBACK;404'}`. `useParams/usePathname/useSearchParams`
   `useContext` the hooks-context module (NOT window) so they work in SSR and hydrate identically.
   `useRouter().push/replace` delegate to `window.__diffpack_navigate` (Slice 1) when present,
   else `location.assign` fallback.

6. **`ssr_entry_module` + `client_entry_module`**: wrap the rendered/hydrated root in the three
   providers fed from `params`/parsed url (SSR: from fd-3 params carried through `renderFlightToDocument`;
   client: from the injected `window.__DIFFPACK_PARAMS__`/`__DIFFPACK_URL__`). Both graphs must feed
   the SAME values or hydration warns (gate 7 asserts zero console errors).

7. **`scripts/rsc/next-server.mjs`**: for the render op write the request-context JSON to the child's
   stdin (`{ url:req.url, headers:[...Object.entries(req.headers)], cookie:req.headers.cookie }`).
   After close, branch on fd-3 control: `redirect` → `res.writeHead(control.status||307,{location:control.redirect}); res.end()` (do NOT SSR the flight); `notFound` → 404 not-found tree; else pass
   `control.params` + parsed url into `renderFlightToDocument`. The `?__rsc=1` branch (Slice 1),
   when control carries a redirect, responds `{ "content-type":"application/json" }` `{__redirect:url}`
   so the client Router can `history` + follow (refinement; keep the raw-flight default otherwise).

8. **Fixture**: `app/go/page.tsx` server component calling `redirect('/about')`; extend
   `blog/[slug]/page.tsx` to `const c = await cookies(); ... theme: {c.get('theme')?.value}` and embed a
   `"use client"` island rendering `useParams().slug`. Keep `next build`-valid.

9. **`docs/RSC_NEXT_GAP.md` §4.7** → CLOSED.

### GATE (add to `scripts/rsc/next-check.sh`)

- **Gate 6 (redirect):** `curl -sI "$base/go"` returns `307` + `location: /about`; `curl -sL "$base/go"`
  lands on the /about document.
- **Gate 7 (cookies):** `curl -s --cookie 'theme=dark' "$base/blog/hello"` HTML contains `theme: dark`
  (requestAls carried the cookie into `await cookies()`).
- **Gate 8 (useParams SSR+client):** `curl -s "$base/blog/hello"` contains `slug: hello` (SSR via
  PathParamsContext); then agent-browser open `$base/blog/hello`, assert the client island's
  `useParams().slug` text is `hello` after hydration with **zero console errors/warnings**.

**Risks.** ALS lifetime (wrap call+drain); the orchestrator must decide redirect/notFound purely from
fd-3 and NOT SSR a flight whose root errored; client hooks must use React contexts (not window) fed
identically on both sides; `cookies/headers/draftMode` must be `async`; guard `writeSync(3,…)` for
standalone runs; literal routes must beat dynamic ones (don't let `/about` match `/[slug]`).

---

## Slice 4 — `next/image` fidelity (gap 4.2)

**Goal.** Faithful, server-free `next/image`: SVG/`unoptimized` srcs render with the raw src and NO
srcset (matching Next), `priority` emits a React-hoisted `<link rel=preload as=image>`, and raster
srcs get a real responsive `srcset` pointing at build-emitted variant files (no image-optimization
server; optimization happens at build time, output is plain static files). Two layers, ONE slice.

### Layer A — runtime shim (no new crate) — `getImgProps` port

**`src/next_adapter.rs` `next_image_shim()`** (L822): rewrite as a port of Next's `getImgProps`,
running in all three graphs. Import `{ createElement }` from `react` and `{ preload }` from `react-dom`.
- Classify: UNOPTIMIZED if src ends `.svg`, is `data:`/`blob:`, the `unoptimized` prop is set, or there
  is no build-manifest entry for the src. Optimized only for raster with a manifest entry.
- UNOPTIMIZED path (the stock-template path): `<img src={rawSrc} width height decoding="async" loading fetchPriority>`
  with NO srcSet/sizes — byte-faithful to Next's SVG handling.
- OPTIMIZED path: compute widths+descriptor via getWidths (deviceSizes `[640,750,828,1080,1200,1920,2048,3840]`,
  imageSizes `[16,32,48,64,96,128,256,384]`; `sizes` present → `w` descriptors filtered `>= deviceSizes[0]*minVwRatio`;
  no `sizes` + numeric width → `[snap(w), snap(2w)]` `x` descriptors, capped at intrinsic — no upscaling);
  build `srcSet` from manifest variant URLs; `sizes` passthrough; src = largest variant.
- `decoding="async"` always; `loading = priority ? "eager" : (loading ?? "lazy")`; `fetchPriority` passthrough.
- PRIORITY: `preload(rawSrc, { as:"image", imageSrcSet, imageSizes, fetchPriority:"high" })`. **Verify `preload`
  is exported under the react-server condition; if not, hoist `createElement("link",{rel:"preload",as:"image",…})`
  instead** — both are React-hoisted; pick the one that works, never a silent no-op.
- No-silent-stub: a raster src with neither a manifest entry NOR an unoptimized marker MUST THROW naming the src.

Read the manifest via `import MANIFEST from "#diffpack-next-image-manifest"` (aliased in `configure`,
same pattern as `#diffpack-rsc-action-handler`).

### Layer B — build-time variant emit + manifest (raster only)

- **`Cargo.toml`**: `image = { version = "0.25", default-features = false, features = ["png","jpeg"] }`
  (pure Rust, build-time only).
- **New `emit_image_variants(public_dir, out_public) -> Manifest`** in `src/next_adapter.rs`, called from
  the public-copy step (`src/main.rs`): decode each raster under `public/`, read intrinsic w/h, downscale
  to each getWidths candidate `<= intrinsic`, write `out/public/_diffpack-image/<hash>-<w>.<ext>`, collect
  `{ width, height, variants:[{w,url}], unoptimized:false }`; SVGs get `{ unoptimized:true }` (no decode).
  Serialize into a generated module written in `configure`, aliased `#diffpack-next-image-manifest`.
- **`scripts/rsc/next-server.mjs` MIME** (L105): add `.png/.jpg/.jpeg/.webp/.gif`.

### Fixture

`integration/next-app-router/`: check in a tiny `public/hero.png` (≥1080px wide so ≥2 deviceSizes apply);
`app/page.tsx` renders `<Image id="hero" src="/hero.png" alt="hero" width={1200} height={300} sizes="(max-width: 600px) 100vw, 600px" priority />` and `<Image id="logo" src="/next.svg" alt="logo" width={100} height={20} />`.

### GATE (add Gate 1i to `scripts/rsc/next-check.sh`)

`html=$(curl -s "$base/")` and assert:
- RASTER: `id="hero"` `<img>` has `srcset` with ≥2 candidates matching `/_diffpack-image/[^ ]+\.png [0-9]+w`;
  `sizes="(max-width: 600px) 100vw, 600px"`; `decoding="async"`; `fetchpriority="high"`. `curl -sI` the largest
  variant URL → HTTP 200 `image/png` (variants are real files).
- PRELOAD: head contains `<link rel="preload" as="image"` for the hero (`imagesrcset` or
  `href="/hero.png"`+`fetchpriority="high"`).
- SVG UNOPTIMIZED: `id="logo"` `<img>` has raw `src="/next.svg"`, NO `srcset`, `decoding="async"`.

**Risks.** Verify `react-dom` `preload` under the react-server condition (fall back to a hoisted `<link>`).
Keep `image` features minimal. srcset is faithful-*shape*, not byte-exact to Turbopack. Static image
imports (`import x from './x.png'` → `{src,width,height,blurDataURL}`) and blur placeholder are OUT —
document in §4.2, do not silently skip.

---

## Slice 5 — The Next dev server (`diffpack dev <next-app>`) (gap 4.8)

**Goal.** `diffpack dev integration/next-app-router` boots a dev server with Fast Refresh for `"use client"`
islands (state-preserving, no reload) and a correct full-reload for server-component edits, reusing the
existing dev machinery in `src/dev_server.rs`. Today a Next app falls through to the TanStack path and
errors (no `src/client.tsx`) — add a third topology.

### Files & changes

1. **`src/next_adapter.rs`** — expose detection + a dev config variant:
   - `pub fn is_app_router(root: &Path) -> bool` wrapping `detect_app_router(...).is_some()`.
   - Add a `dev: bool` param to `configure` (or `configure_dev`): when dev, `build.hmr=true`, NODE_ENV
     define `"development"`, and map `production`→`development` in `build.conditions` for all three
     environments (mirroring `config::set_web_development_mode`). Prod call sites pass `false` — output
     byte-identical. Run all three graphs in development so the SSR/react-server React matches the client
     React at hydration (avoid a dev/prod hydration split).

2. **`src/dev_server.rs`** — dispatch + `mod next` (model on the existing `mod spa`):
   - In `run()`, before the `has_start`/`index_html` check:
     `if crate::next_adapter::is_app_router(&project_root) { return next::run_next(&options, &project_root); }`.
   - Extend `is_module_path` to ALSO exclude `.diffpack-next/` (so the adapter's own scaffold writes don't
     re-enter the watch loop).
   - `mod next`:
     a. `build_next_{client,react_server,ssr}(root, out)` — each: `configure(root, env, /*dev*/true)`, register
        the client `call-server` virtual module (client) or the full RSC server virtual-module set (react-server+ssr,
        reuse/generalize `register_server_virtual_modules` to take an output root), discover, wrap in
        `EnvBuild{ options: EmitOptions{ minify:false, hmr:true, source_map, .. } }`. **Emit to non-colliding
        roots**: client → `<out>` (public/, manifests); react-server → `<out>/.rsc` then copy `.rsc/server`
        → `<out>/rsc-render` and preserve `.rsc/server/server.css` → `<out>/public/rsc.css`; ssr → `<out>`
        (server/). This is load-bearing: react-server and ssr both target `server/`, so a server-component
        re-emit would clobber the ssr bundle next-server.mjs holds in memory without the `.rsc` indirection.
     b. Initial build in order client → react-server → ssr.
     c. `spawn_node` `scripts/rsc/next-server.mjs` with `[<out>, <port>]` and env `DIFFPACK_NEXT_DEV=1`; `wait_for_node`.
     d. `HmrHub` + `find_refresh_runtime` + `serve_proxy` on `options.port` (injects the Fast Refresh preamble +
        WS client into every served HTML document, exactly as the TanStack/SPA paths).
     e. Watch `app/` recursively + root non-recursively via `start_watcher_paths`.
     f. `next_watch_loop` (coalesce ~60ms, filter `is_module_path`), classify each changed path:
        - **client island** (`"use client"`, known to the client bundler): rebuild client + ssr EnvBuilds,
          re-emit, re-persist `client-references-manifest.json`, `hmr_push_client` a state-preserving Fast
          Refresh update (islands are generic refresh boundaries; no `hmr.rs` change). If `graph_changed`,
          `broadcast_reload` instead.
        - **server component** (page/layout, no directive, known to the react-server bundler only): rebuild+emit
          only the react-server graph to `.rsc`, copy → rsc-render, preserve rsc.css, `hub.broadcast_reload()`.
          next-server.mjs spawns a FRESH react-server child per GET → new content on reload. (Honest minimum:
          without a soft-nav flight-diff this is a full reload, not HMR — document it, do not dress it up.)
        - **structural** (new/renamed/deleted page/layout/island/route): re-run `configure`, full rebuild all
          three, `restart_node`, `broadcast_reload`.

3. **`scripts/rsc/next-server.mjs`** — `const DEV = process.env.DIFFPACK_NEXT_DEV === "1";`. When DEV, resolve
   `renderFlightToDocument` per request by re-importing `ssrEntry` with `?v=<statSync(ssrEntry).mtimeMs>` when the
   mtime changed (cache keyed by mtime) — closes in-process SSR staleness after an island edit + manual refresh.
   The react-server child is already fresh per request. Prod path (single top-level import) untouched.

4. **`scripts/rsc/next-dev-check.sh`** (new) + `integration/next-app-router/dev-check.mjs` (model on
   `integration/tanstack-start-reference/dev-check.mjs`): build release; `diffpack dev integration/next-app-router <port>`;
   agent-browser open `/`.

5. **`docs/RSC_NEXT_GAP.md` §4.8** → dev server CLOSED for Fast-Refresh islands + server-component reload;
   streaming Suspense shell / middleware / Next manifests remain documented gaps.

### GATE (new `scripts/rsc/next-dev-check.sh`, wired into `check.sh` Tier-3 near the next-check row)

1. Start `diffpack dev integration/next-app-router <port>`; agent-browser opens `/`; assert hydrated
   (island interactive) and the served HTML carries the diffpack WS/Fast-Refresh preamble.
2. **Island state-preserving update:** click `#inc` (5→6), edit `app/Counter.tsx`'s rendered label
   (`count: `→`tally: `); within the diff budget the WS pushes `{type:"update"}`; assert the label became
   `tally: 6` on the SAME live node (state preserved, NO reload).
3. **Server-component reload:** edit `app/page.tsx`'s `from-server` string; assert the new string appears
   in the browser AND in a fresh `curl /` (per-request react-server child re-rendered it server-side).
4. Restore `app/Counter.tsx` + `app/page.tsx` in a `finally`. Production `next-check.sh` stays green;
   `cargo build --release` + `cargo test --lib` + clippy pass.

**Highest risk (verify in-browser):** Fast Refresh through the flight — the island is mounted via a client
reference resolved through `__webpack_require__`; Fast Refresh instruments the same module. If the seam
instance and the Fast-Refresh-registered family align, state preserves; if not, it falls back to reload.
Only the browser gate proves it — if it reloads instead of preserving state, chase family-key vs runtime-id
alignment, do not silent-pass. Also verify the RSDW `development` build resolves under react@19.2.4; if not,
keep react-server/ssr in production while the client is development and watch for a hydration warning.

---

## Slice 6 — The stock-template culmination (unmodified `create-next-app`)

**Goal.** Build and serve `integration/next-app-router/authentic-create-next-app/` **VERBATIM** (no edits to
its `app/`) under diffpack, and prove SSR + hydration + `next/font` + `next/image` + a real 404 in a real
browser. This is the headline: the unmodified default template running under diffpack. All prior slices exist
to make this pass.

The default template uses: `next/font/google` (§4.1 CLOSED), `next/image` with SVG srcs + `priority`
(Slice 4 Layer A), `globals.css` + `page.module.css` (§4.3 CLOSED), `export const metadata` (§4.4 CLOSED),
a single `/` route (Slice 2's routing) and no `app/not-found.tsx` (Slice 2's default 404 body). It has NO
client island and NO server action — simpler than the hand-authored fixture in those respects.

### Files & changes

- No adapter source changes expected beyond Slices 1–4 — this slice is a VALIDATION gate that exercises the
  authentic template. If it surfaces a real gap (e.g. a `next.config.ts` shape, a favicon route, an unhandled
  `next/*` specifier), fix it in the adapter and note it; do NOT edit `authentic-create-next-app/app/`.
- `detect_app_router` must accept the authentic `next.config.{ts,mjs,js}` + `app/page.tsx`.
- The default 404 (Slice 2) applies since the template has no `app/not-found.tsx`.

### GATE (new `scripts/rsc/next-authentic-check.sh`)

Build the authentic dir (client → react-server → cp → ssr, `--no-minify`), boot next-server.mjs, then assert:
- SSR: `curl -s "$base/"` returns `<!DOCTYPE html>` with the template's marquee text and the root layout.
- `next/font`: the Geist `@import`/font CSS is hoisted into `<head>` and the font-variable class is on `<html>`.
- Metadata: the template's `<title>` ("Create Next App") is rendered.
- CSS: `page.module.css` scoped class on the element matches a rule in the served stylesheet.
- `next/image`: the template's SVG images (`/next.svg`, `/vercel.svg`) render as `<img src="/next.svg">` with
  NO srcset; the `priority` image emits a hoisted `<link rel="preload" as="image">`.
- 404: `curl -s -o /dev/null -w '%{http_code}' "$base/nope"` == `404`.
- Real browser: agent-browser open `/`, assert the document hydrated with ZERO console errors/warnings.

Wire this as a `rsc_gate` row in `check.sh`. Exit 0 = the unmodified stock `create-next-app` default builds,
renders, and hydrates under diffpack.

---

## Slice 7 — The Turbopack benchmark

**Goal.** A fair, HONEST diffpack-vs-Turbopack(-vs-webpack) `next build` comparison on the pinned fixture.
`scripts/bench-next.mjs` already exists and runs green — this slice wires it in and documents it honestly.

### Files & changes

- **`scripts/bench-next.mjs`** (exists): three cases `diffpack-next` / `next-turbopack` / `next-webpack` on
  `integration/next-app-router`. Fresh process each run; wipe `.next` / `.diffpack-output` before EVERY run
  (true cold, cache included); median of N (default 5) + one uncounted warmup; `NEXT_TELEMETRY_DISABLED=1`.
  diffpack RSS = max over the 3 invocations (never the shell wrapper). Every build passes a verify-or-EXCLUDE
  gate (next: `.next/server/app/index.html` prerendered; diffpack: `public/client.js` + `server/server.mjs`).
  Restores `next-env.d.ts`/`tsconfig.json` in a finally.
- **`docs/COMPETITIVE_BENCHMARKS.md`**: replace the "## Turbopack … future work" paragraph with
  "## Turbopack (measured)" carrying the labeled table + the caveat block + the env row (M2 Max, next 16.2.11).
- Optionally `check.sh`: a non-blocking bench row (or leave bench manual — it is measurement, not correctness).

### The mandatory honesty framing (do NOT drop)

`next build` type-checks, lints, generates route types, prerenders every static route to HTML+RSC, and emits
manifests + build traces; diffpack does NONE of these (renders per-request). Every next row is labeled
"FULL framework build incl. typecheck + lint + SSG"; the diffpack row "bundles the 3 RSC graphs; NO
typecheck/lint/SSG — renders per-request". Present as end-to-end "time to a deployable build", **NEVER** a bare
"N× faster than Turbopack". The diffpack build's CORRECTNESS is separately gated by `scripts/rsc/next-check.sh`
(+ Slice 6's authentic gate), so "fast" is backed by a passing functional gate, not a broken build.

Reference numbers already measured on this machine (median of 3, cold, cache wiped):

| case | measures | wall median | peak RSS | client out | server out |
|------|----------|-------------|----------|-----------|-----------|
| diffpack-next | bundles only — NO typecheck/lint/SSG (per-request render) | 54 ms | 21 MB | 545 KB (8f) | 530 KB (6f) |
| next-turbopack | FULL framework build (typecheck+lint+SSG) via Turbopack | 2572 ms | 569 MB | 712 KB (19f) | 4170 KB (133f) |
| next-webpack | FULL framework build (typecheck+lint+SSG) via webpack | 7967 ms | 457 MB | 958 KB (26f) | 789 KB (63f) |

`next/font/google` fetches Google fonts at build on the NEXT side only (diffpack rewrites the macro locally) —
note the network dependence. diffpack runs `--no-minify` (parity with next-check.sh); add a minified-diffpack
size row only if size is the headline.

### GATE

`node scripts/bench-next.mjs --runs 3` exits 0, prints the labeled table + the caveat, saves
`bench/results/next-results.json`, every case passes its verify gate (a failed build is EXCLUDED with its
reason, never silently timed), and the working tree stays clean.

---

## Cross-slice integration notes

- **fd-3 channel** is introduced in Slice 2 (`{status,params}`) and EXTENDED in Slice 3
  (`{status,params,redirect?,notFound?}`) — one channel, one `writeMeta`, guarded for standalone runs.
  `runReactServer` gains a 4th stdio pipe once (Slice 2) and never rejects on exit code for a 404.
- **`"use client"` generated modules** (Slice 1 Link, Slice 2 error-boundary) both need explicit pinning
  into the `islands` vec before `island_pins`, because `scan_islands_dir` skips `.diffpack-next/`. Same
  mechanism, applied twice.
- **Client hooks** end on React contexts (Slice 3), not window globals — Slice 2's interim
  `window.__DIFFPACK_PARAMS__` read in `useParams` is replaced by the `PathParamsContext` read in Slice 3.
- **`?__rsc=1`** (Slice 1) coexists with fd-3 status (Slice 2): raw flight is status-agnostic; a redirect
  (Slice 3) turns the `?__rsc=1` response into `{__redirect:url}` JSON for the client Router.
- **Emit roots** (Slice 5): the `.rsc` indirection that keeps react-server from clobbering the ssr `server/`
  bundle is dev-only; the production build order (client → react-server → cp → ssr) is unchanged.
- After each slice: `docs/RSC_NEXT_GAP.md` flips the corresponding §4.x to CLOSED with the honest remaining
  refinements listed; `./check.sh` stays green.
