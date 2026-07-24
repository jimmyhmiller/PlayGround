# Slice F — a real Next.js app-router app under diffpack: what works, and the honest gaps

This is the Slice F deliverable (per `docs/RSC_SPEC.md` §3 / R5): take a **real
Next.js app-router app**, build and serve it under diffpack using the RSC spine
from Slices A–E, get as far as possible end-to-end **in a real browser**
(`agent-browser`), and record precisely what works and what is still missing to a
fully working `create-next-app`.

Everything below is evidence-based: the numbers and errors come from actually
running `create-next-app`, `next build`, and diffpack against the fixture in
`integration/next-app-router/`.

---

## 0. TL;DR

- A **real `create-next-app --app` project** was created (`next@16.2.11`,
  `react@19.2.4`). Its pristine default is preserved verbatim under
  `integration/next-app-router/authentic-create-next-app/`.
- diffpack now has a **native Next app-router adapter** (`src/next_adapter.rs`):
  it detects the app-router file conventions, scaffolds the three RSC entries +
  minimal `next/*` shims under `.diffpack-next/`, and the proven RSC spine builds
  all three graphs unchanged. No Node/Vite/Next in the build path.
- A real app-router app (Server Component root layout wrapping an async Server
  Component page, a `next/link`, a `"use client"` island, and a `"use server"`
  action) — one that **`next build` accepts unchanged** — is built entirely by
  diffpack and **renders + hydrates + runs a server action end-to-end in a real
  browser**. Gate: `scripts/rsc/next-check.sh` (exit 0).
- **CULMINATION (done):** the **UNMODIFIED** `create-next-app --app` default
  (preserved verbatim at `authentic-create-next-app/`: `next/font/google`
  Geist + Geist_Mono, Metadata, `globals.css` + CSS Modules, `next/image` on the
  stock SVGs) now **builds with the diffpack RSC spine + native next adapter,
  server-renders the full document, its styles apply, and it hydrates cleanly in a
  real browser** — with its `app/` untouched (checksum-verified). Gate:
  `scripts/rsc/next-authentic-check.sh` (exit 0; A1–A7). See §6.
- The remaining gaps to a *fully working* `create-next-app` default were precise
  and listed in §4 — all now CLOSED (§4.1–4.8).

---

## 1. The real Next.js app (the oracle)

`npx create-next-app@latest next-app-router --app --ts` produced a genuine Next 16
app-router project. Confirmed real by `next build` (Turbopack): compiles, generates
static routes `/`, `/_not-found`, `/about`. The pristine default (`app/page.tsx`
with `next/image` + CSS modules, `app/layout.tsx` with `next/font` + Metadata) is
kept under `authentic-create-next-app/` so the gap analysis is against the *actual*
template, not a strawman.

The app diffpack builds (`app/`) is a real app-router app that **also passes
`next build` unchanged** (verified): `app/layout.tsx` (Server Component root layout,
owns `<html>`), `app/page.tsx` (async Server Component: `await`s data, renders a
`next/link`, embeds the island, passes the action in), `app/Counter.tsx`
(`"use client"` `useState` island), `app/actions.ts` (`"use server"` `increment`),
`app/about/page.tsx` (a second route). It exercises the full RSC spine without
depending on the Next-compiler-only features that are the documented gaps.

React pairing: `create-next-app` ships `react@19.2.4`; diffpack's RSC transforms
import `react-server-dom-webpack`, so the fixture pins the **matching**
`react-server-dom-webpack@19.2.4` (its export surface under the `react-server`
condition — `createClientModuleProxy`, `registerServerReference`,
`renderToReadableStream`, `createFromReadableStream`, `createServerReference`,
`encodeReply` — matches the transforms exactly; verified by probe).

---

## 2. The native app-router adapter (`src/next_adapter.rs`)

The RSC spine is framework-neutral — it drives off `"use client"` / `"use server"`
directives and a canonical `module_reference_id`, not off any TanStack- or
Next-specific entry. What a `create-next-app` project lacks is the **entries**: Next
has no `src/client.tsx`/`src/server.ts`/`src/rsc-entry.tsx`; its "entry" is the
app-router file convention. Next's own runtime composes
`<RootLayout><Page/></RootLayout>` and renders that tree — the adapter does the same
natively:

1. **Detect** an app-router project: `next.config.*` + `app/page.{tsx,jsx,ts,js}`.
2. **Scaffold** `.diffpack-next/` (gitignored): the react-server render/action entry
   (composes `<RootLayout><Page/></RootLayout>` → flight), the SSR-of-flight entry
   (reconstructs the flight and renders the **whole document** — the RootLayout owns
   `<html>` — with react-dom's bootstrap options so the client script + inlined
   flight are part of the stream and hydration matches), and the browser entry
   (reconstructs the inlined flight, `hydrateRoot(document, …)`). Every `"use
   client"` module under `app/` is discovered and pinned into the client + SSR
   graphs so the flight's client references resolve to real code.
3. **`next/*` shims** (aliased to real generated files): `next/link` → server `<a>`,
   `next/image` → plain `<img>`, `next/navigation` (browser-primitive router,
   hard-errors where it can't be faithful), `next/headers` (hard-errors — no request
   context). Unshimmed `next/*` is intentionally **not** aliased to a silent no-op;
   it fails at resolve naming the specifier (repo no-silent-stub rule).
4. **Return an `AppConfig`** per environment (client / react-server / ssr) with the
   right target, resolve conditions, aliases, and the `NODE_ENV=production` define
   (keeps React's dev build out of the bundle).

Generating entry/shim source as Rust strings follows the existing precedent of
`rsc::generate_action_resolver_module` and `server_fn::generate_resolver_module`:
diffpack-authored build glue, not guest source hidden in a string. Wired in
`main.rs` `build-app` **before** `derive_config`; a non-Next project returns `None`
and the TanStack path is unchanged.

---

## 3. What works end-to-end in a real browser

`scripts/rsc/next-check.sh` (native Rust build; Node + Chrome only as the oracle;
`agent-browser` drives Chrome). All four gates pass, exit 0:

1. **App-router SSR of the full document.** `curl /` returns
   `<!DOCTYPE html><html lang="en">…<div id="app-shell"><main id="page"><h1
   id="heading">Server:from-server</h1><p><a href="/about" id="about-link">About</a>
   </p><div id="island"><span id="counter">count: 5</span>…` — proving the
   app-router **layout wraps the page**, the async Server Component ran on the
   server, the `next/link` rendered a server `<a>`, and the `"use client"` island's
   initial state is present. react-dom injects the client bootstrap module + the
   inlined flight.
2. **The boundary holds.** The client bundle contains no `actions.ts` reference and
   no `n + 1` server-action body.
3. **Hydration + the `__webpack_*` seam.** In Chrome, clicking `#inc` moves the
   island's `useState` count 5 → 6 — the flight's client reference resolved to the
   real diffpack-bundled `Counter` and hydrated. **Zero console errors/warnings**
   (clean whole-document hydration).
4. **The `"use server"` action round-trips.** Clicking `#server-inc` calls the
   action the Server Component passed in: `encodeReply` → `POST /_action/` →
   `decodeReply` → dispatch → `renderToReadableStream` → `createFromFetch`, and the
   result renders (`increment(6)` → `server: 7`). Local state survives the call.

So: a **real Next.js app-router app** (Server Component layout + page, `next/link`,
a hydrating `"use client"` island, a `"use server"` action) is **built entirely by
diffpack** (three native graphs) and **works end-to-end in a real browser**.

---

## 4. The precise remaining gaps to a fully working `create-next-app`

Building the **untouched** default template (`authentic-create-next-app/`) under
diffpack surfaces these, in order:

### 4.1 `next/font/google` (build-time macro) — CLOSED
**Done.** `src/next_font.rs` rewrites each `Geist({...})`/`localFont({...})` call to
the static `{ className, variable, style }` object and drops the throwing import
(wired into the transform pipeline, gated on a `next/font` string check). The
app-router adapter (`collect_app_fonts` + `next_font::generate_css`) generates the
companion CSS — a Google Fonts `@import` for the real webfont plus the CSS-variable
class from the call's `variable` option — and the render entry injects it as a
React-19-hoisted `<style href precedence>`, so `${font.variable}` on `<html>`
resolves and the real font loads. The hand-authored fixture now uses `next/font`
exactly as the stock template; `scripts/rsc/next-check.sh` gate 1b asserts the
hoisted Geist `@import` + variable class in the served `<head>`. (Self-hosting the
font files, Next's default, is the remaining refinement over the `@import`.)

<details><summary>original analysis</summary>

#### (was) Hard blocker on the default template: `next/font/google` (build-time macro)
`app/layout.tsx` does `const geistSans = Geist({subsets:["latin"]})`. diffpack
bundles the real `next/font/google` module, but at render it throws
`TypeError: __import(...) is not a function`. **Why:** `next/font` is a
**Next-compiler macro** — the npm module is a build-time placeholder; Next's SWC
loader *replaces* each `Geist({...})` call at build time with generated
`@font-face` CSS + a hashed class name (optionally self-hosting the fetched font).
There is no runtime module to call. **To close:** a diffpack `next/font` transform
that, at build time, evaluates the `Geist(...)`/`localFont(...)` call, fetches
(google) or reads (local) the font, emits the CSS, and rewrites the binding to
`{ className, variable, style }`. Not shimmable at runtime.
</details>

### 4.2 `next/image` — CLOSED (native, server-free)
**Done (Slice J).** The `next/image` shim (`next_adapter::next_image_shim`) is a
faithful port of Next's `getImgProps`, running in all three graphs:
- **Raster** srcs (`png`/`jpeg`) under `public/` get a real responsive `srcSet`
  built from Next's `deviceSizes`/`imageSizes` (via a `getWidths` port honouring
  `sizes` vw-ratios or `width`-based `1x`/`2x` descriptors), pointing at
  build-emitted static variant files under `/_diffpack-image/`. There is **no
  image-optimization server**: the pure-Rust `image` crate decodes each raster at
  build time and writes downscaled variants (`emit_image_variants`, called from the
  client build's public-copy step in `main.rs`), keyed by a deterministic hash the
  generated manifest (`.diffpack-next/image-manifest.ts`) agrees on. No upscaling
  past intrinsic width.
- **SVG / `data:` / `blob:` / `unoptimized`** srcs render the raw `src` with **no**
  `srcSet` — byte-faithful to Next's SVG handling.
- **`priority`** renders a `<link rel="preload" as="image">` (with `imageSrcSet`/
  `imageSizes`/`fetchPriority=high`) that React 19 hoists into `<head>`; `decoding`
  defaults to `async`, `loading` to `lazy` (or eager under `priority`).
- **No silent stub:** a local raster path with no manifest entry **throws** naming
  the src (a real build gap, never a degraded `<img>`).

Note: React 19 emits these attribute names camelCase in the HTML string
(`srcSet`/`fetchPriority`/`imageSrcSet`) — the browser normalizes attribute names
case-insensitively, exactly as Next itself renders under React 19. Gated by
`next-check.sh` gate 1i (raster `srcset` ≥2 variant candidates + `sizes` +
`decoding` + `fetchpriority`, the largest variant a real `200 image/png`, a hoisted
priority preload link, and the SVG rendered raw with no `srcset`). **Remaining
(documented, not silently skipped):** static image imports (`import x from
'./x.png'` → `{src,width,height,blurDataURL}`), the blur placeholder, and `webp`/
`gif`/`avif` optimization (registered `unoptimized` — raw passthrough — since this
build compiles only the `image` crate's `png`+`jpeg` decoders).

### 4.3 CSS: `import "./globals.css"` + CSS Modules — CLOSED
**Done.** The react-server graph is authoritative for CSS (Server Components render
there, so its CSS-Module class scoping matches the flight-rendered classNames). Its
compiled `server.css` (globals + scoped modules) is preserved to the served,
non-pruned `public/rsc.css` by the build (`main.rs`), and the render entry links it
via a React-19-hoisted `<link rel="stylesheet" href="/rsc.css" precedence>`
(injected only when the app imports CSS — `next_adapter::app_has_css`). Gated by
`next-check.sh` gate 1c: the module class applied to the element (`_page_2e9f2b1b`)
must match a rule in the served stylesheet, and `globals.css` must be present. The
hand-authored fixture now imports `globals.css` + a `page.module.css` exactly like
the stock template. (Per-route CSS splitting/dedup across many routes is a later
refinement; a single app stylesheet covers the current single-route surface.)

<details><summary>original analysis (now closed)</summary>
diffpack compiled CSS but the app-router adapter did not inject the layout's global
stylesheet nor thread the per-graph CSS-Modules asset into the document `<head>`.
</details>

### 4.4 Metadata API — ignored
**CLOSED for the static `title`/`description` subset.** `next_adapter::scan_metadata`
reads `export const metadata = { title, description }` (page overriding layout) and
the render entry emits `<title>`/`<meta name="description">`, which React 19 hoists
into `<head>`. Gated by `next-check.sh` gate 1b (asserts the `<title>`). Remaining:
`generateMetadata()` (async), OpenGraph/icons/other fields, and title templates — a
mechanical extension of the same scan.

### 4.5 App-router routing surface — multi-route + nested layouts + dynamic segments + loading/error/404 DONE
**Done (static + dynamic routes + nested layouts + per-request matching + boundaries + real 404).**
The adapter discovers EVERY `app/**/page.{ext}` route (`discover_routes`), each with
its root→leaf level chain (nested `layout.{ext}` + `loading.{ext}` + `error.{ext}`
per level) and resolved metadata, and emits a ROUTE TABLE (each route's parsed
segment pattern) into the react-server render entry. `parse_segment` classifies
`[param]`→Dynamic, `[...x]`→CatchAll, `[[...x]]`→OptionalCatchAll, `(group)`→stripped;
routes are sorted most-specific-first so a literal segment beats a dynamic one.
The `render <pathname>` op's `matchRoute` matches the pathname to the most-specific
route, captures dynamic params, and composes the route boundaries inner→outer (page →
`<Suspense fallback={loading}>` → generated client `<ErrorBoundary fallback={error}>`
→ layouts root-last), delivering `params` to the page (`params: Promise.resolve(...)`,
Next 16 shape) and — via `window.__DIFFPACK_PARAMS__` injected in the SSR bootstrap —
to client `useParams()`. An unmatched path renders a real HTTP **404** (`app/not-found.tsx`
wrapped in the root layout; status carried to the orchestrator over an fd-3
sidechannel), never the index tree. Route groups `(group)` are stripped from the URL.
Gated by `next-check.sh` gates 1d (second route `/about`), 1e (dynamic `[slug]`:
`/blog/hello`→`post: hello`, `/blog/world`→`post: world`, per-request), 1f (real 404:
`/no/such/path` → HTTP 404 + `app/not-found.tsx`, no `from-server` fall-through),
1g (a throwing Server Component contained by the client ErrorBoundary — HTTP 200,
root layout intact, and the client `error.tsx` fallback recovers after hydration in a
real browser), and 1h (`loading.tsx` composes a Suspense, non-breaking).
**Honest deviations / remaining gaps:**
- Under `NODE_ENV=production` React **sanitizes** the Server-Component error message
  and defers the error-boundary recovery to the client, so the SSR HTML carries the
  empty Suspense placeholder and the real thrown text ("boom-from-server") is omitted
  — it surfaces on the client as a generic message; the dev server (§4.8) would show
  the real one. Gate 1g asserts the boundary caught + rendered its fallback, not the
  literal text.
- Because SSR uses `onAllReady`, a `loading` fallback is never in the final static
  HTML (true fallback-in-HTML is streaming, §4.8) — gated structurally + non-breaking.
- Per-route `not-found` boundaries (catching `notFound()` throws) need the request
  context — deferred with `next/navigation` (§4.7). The app-root 404 is done.
- Parallel/intercepting routes (`@slot`, `(.)`) and `template`/`route` conventions
  remain gaps (still skipped, never mis-served). `route_tree.rs` (TanStack) models
  much of the same mapping.

### 4.6 Client-side soft navigation — CLOSED
**Done.** `next/link` is now a `"use client"` intercepting component
(`next_adapter::next_link_shim`): a plain left-click on an internal href is handed to
the client Router (`window.__diffpack_navigate`) instead of a full document load
(modified clicks / external hrefs / pre-hydration fall through to a real navigation,
no `preventDefault`). The orchestrator serves each route's RAW flight at
`GET /<path>?__rsc=1` (`content-type: text/x-component`) — zero new render logic, the
flight is already computed for every GET, and the static-asset check runs first so
`?__rsc=1` never shadows an asset. The browser entry (`client_entry_module`) is now a
client Router: `useState(initialTree)` + `useTransition`, a `navigate()` that
`createFromFetch(fetch(href + "?__rsc=1"), { callServer })` — the same
`__webpack_*` seam + `callServer` transport the action round-trip uses, no manifest —
`setTree(next)` inside a transition (keeping the old document visible until the new
flight resolves) and updates `history`; a `popstate` listener soft-navigates on
back/forward. React 19 reconciles the swapped `<html>/<head>/<body>` in place. The
link shim is pinned into the client + ssr island lists (it stays a client reference in
the react-server graph) so its client reference resolves and it hydrates. Gated by
`next-check.sh` gate 5a (raw `?__rsc=1` flight is `text/x-component`, not a document;
plain `/about` still full HTML) and gate 5b (real browser: clicking `#about-link`
soft-navigates to `/about` with a page-scoped window sentinel + the `#app-shell` root
layout surviving = no reload + diff-render, history updated; `history.back()`
soft-navigates home). **Out of scope (documented follow-ups):** prefetch cache, scroll
restoration, and `useRouter().push` delegating to `window.__diffpack_navigate`.

### 4.7 `next/navigation`, `next/headers` — CLOSED
**Done.** A per-request context (an `AsyncLocalStorage` in the generated
`.diffpack-next/request-context.ts`) is established by the react-server render entry:
the orchestrator sends the request `{ url, headers, cookie }` to the render child on
stdin, the render binds it (plus the matched dynamic-segment `params`) into a store,
and wraps BOTH `renderToReadableStream(...)` AND the stream drain in `requestAls.run`
so late async Server Components keep the store. On that spine:
- **`next/headers`** `cookies()`/`headers()` are real `async` functions reading
  `requestAls.getStore()` — `await cookies()` returns a `RequestCookies`-like object
  (`.get(name)→{name,value}`, `.getAll()`, `.has()`) parsed from the request Cookie
  header; `await headers()` returns the request `Headers`. Called outside a request
  (no store) they **hard-error** naming the missing context (not silently empty).
  `draftMode()` is faithfully always-disabled (no response-cookie plumbing).
- **`redirect()`/`permanentRedirect()`** on the server throw Next's `NEXT_REDIRECT`
  digest; the render's `onError` captures it and reports it on the fd-3 control
  channel, and the orchestrator issues a **real HTTP 307/308** (`location:`), never
  SSRing the errored tree. `notFound()` throws the `NEXT_HTTP_ERROR_FALLBACK;404`
  digest → the orchestrator serves the real 404 not-found tree. On the client both
  delegate to the soft-nav router (`window.__diffpack_navigate`).
- **`useParams()`/`usePathname()`/`useSearchParams()`** read React **contexts**
  (`.diffpack-next/hooks-context.ts`) fed identically by the SSR entry (from the
  matched params + parsed url) and the client entry (from the injected
  `window.__DIFFPACK_PARAMS__`/`__DIFFPACK_URL__`), so they resolve on BOTH SSR and
  the browser with zero hydration mismatch — not window globals (which don't exist
  during SSR). `useRouter().push/replace` delegate to `window.__diffpack_navigate`
  (soft nav) when present, else browser navigation.
Gated by `next-check.sh` gates 6 (server `redirect('/about')` → real 307, followed to
the About doc), 7 (`await cookies()` reads `theme=dark` → `theme: dark`; no cookie →
`theme: none`), and 8 (`useParams().slug` → `hello` on the SSR HTML AND on the
hydrated client with no console hydration mismatch). **Remaining refinements:** the
`hooks-context` is created once at hydration and not re-fed on a soft navigation
(param updates across `?__rsc=1` navigations), and server actions run without the
request context (cookies/headers inside a `"use server"` action hard-error) — bounded
follow-ups on the same channel.

### 4.8 The Next server / dev server — dev server CLOSED (Fast-Refresh islands + server-component reload)
**Done (Slice K).** `diffpack dev integration/next-app-router` boots a real dev
server for the Next app-router app, reusing the existing dev machinery
(`src/dev_server.rs` + `src/hmr.rs`). A Next app is dispatched to a THIRD dev
topology (`src/dev_server.rs::next`) before the TanStack/SPA split
(`next_adapter::is_app_router`): the SAME three RSC graphs the production build uses
(client / react-server / ssr) are built via `next_adapter::configure_dev` (development
React — whose renderer alone exposes the Fast Refresh hook — + HMR instrumentation +
`NODE_ENV=development`), kept alive per-environment, and served by the embedded next
orchestrator (`scripts/rsc/next-server.mjs`, now `DIFFPACK_NEXT_DEV`-aware: it
re-imports the SSR bundle with a fresh `?v=<mtime>` on change), with the diffpack
reverse proxy in front injecting the WebSocket HMR + React Fast Refresh preamble into
every served document. Two edit classes, both browser-proven:
- **`"use client"` island edit → state-preserving Fast Refresh.** The client + ssr
  graphs rebuild and a WebSocket `update` is pushed (no reload); the island is a
  generic refresh boundary (no `hmr.rs` change), so editing `app/Counter.tsx`'s label
  swaps it on the SAME live node while the `useState` count survives — Fast Refresh
  works through the flight-resolved client reference.
- **Server-Component edit → correct reload.** Only the react-server graph rebuilds,
  into an isolated `<out>/.rsc/server` root then copied to `<out>/rsc-render` (the
  `.rsc` indirection keeps it from clobbering the ssr `server/` bundle the
  orchestrator holds); the orchestrator spawns a fresh react-server child per GET, so
  a `broadcast_reload` (and a fresh `curl`) show the newly server-rendered content.
  This is honestly a full reload, not in-place HMR (a server component has no client
  runtime to hot-swap) — documented, not dressed up.
Structural edits (a new island/route, or an import add/remove that shifts the chunk
partition) full-rebuild all three graphs + restart the orchestrator + reload. Gated by
`scripts/rsc/next-dev-check.sh` (wired into `check.sh`): D1 boot + SSR + injected HMR
preamble, D2 island hydration (count 5→6), D3 island edit → `tally: 6` on the same
node with the page sentinel surviving (no reload) + a pushed WS `hmr update`, D4
server-component edit → the new text in the browser AND a fresh `curl`. The fixture
files are always restored.

**Remaining (documented, not silently skipped):** this is not `next start`/`next dev`
proper — no streaming Suspense shell (SSR uses `onAllReady`, §4.5), no middleware, no
`next dev`-style overlay, no Next manifests (`build-manifest`, `app-paths-manifest`,
`react-loadable-manifest`, `next-font-manifest`), and a server-component edit is a
reload rather than a per-segment flight-diff (the soft-nav transport in §4.6 exists but
is not yet wired into the dev reload path). Live config re-derivation
(`next.config.*`) is not implemented (warned, not silently stale).

---

## 6. The culmination — the UNMODIFIED create-next-app default

The whole effort's target: take the **pristine** `create-next-app --app` default,
touch **nothing** in its `app/`, and have diffpack build + render + hydrate it. Done.

`scripts/rsc/next-authentic-check.sh` copies `authentic-create-next-app/app/` +
`next.config.ts` **byte-for-byte** into a fresh temp build dir (never editing the
preserved source — it re-checksums the source after the whole build and fails on any
drift), points `node_modules` at the working fixture's pinned installs
(`react`/`react-dom`/`react-server-dom-webpack@19.2.4`/`next`), and restores the two
standard create-next-app static SVGs (`public/next.svg`, `public/vercel.svg`) the
page references — the preserved git snapshot captured only `app/` + `next.config.ts`,
not the `public/` static assets, which are not code. It then builds all three RSC
graphs natively (Rust; **0 diagnostics** each), boots the Node orchestrator (oracle
only), and asserts, in a **real browser** (agent-browser):

- **A1** the untouched `app/layout.tsx` + `app/page.tsx` SSR the **full app-router
  document** (`<!DOCTYPE html>`, RootLayout owns `<html lang="en">`, the page's
  Server-Component content present);
- **A2** `next/font/google` is macro-rewritten for **both** `Geist` **and**
  `Geist_Mono`, the font CSS (`@import` + variable classes) hoisted into `<head>`,
  and `${geistSans.variable} ${geistMono.variable}` both resolved onto `<html>`;
- **A3** the stock **Metadata** (`<title>Create Next App</title>` + the description
  meta) rendered;
- **A4** `globals.css` + the page **CSS Module** served via `/rsc.css`, the scoped
  class on the element matching a real rule (scoping agrees);
- **A5** `next/image` renders **both** SVGs raw (`src="/next.svg"`/`/vercel.svg"`,
  **no** `srcset` — unoptimized, byte-faithful to Next under React 19), the
  `priority` image hoists a `<link rel="preload" as="image">`, both assets serve 200;
- **A6** the page **hydrates cleanly** — `hydrateRoot(document, …)` commits a React
  fiber onto the document (`__reactContainer$…`/`__reactFiber$…` present) with the
  inlined flight consumed — and the console has **zero** errors/warnings/mismatch;
- **A7** the styles **apply** in the browser (the CSS-Module `<main>` computes
  `max-width: 800px`, the page container computes `display: flex`, 2 `<img>`).

Verified visually too: the rendered page is the pixel-faithful create-next-app
landing (Next.js logo, the Geist "To get started…" heading, dark theme from
`globals.css`, the CSS-Module-styled Deploy Now / Documentation buttons).

**Honest scope of this gate:** it exercises the stock template's feature surface
(`next/font`, Metadata, `globals.css` + CSS Modules, `next/image` on SVGs, whole-
document SSR-of-flight + browser hydration). It is a *static-export-shaped* render
(the orchestrator is the same Node oracle the other gates use, not `next start`);
the dynamic/interactive surface (soft-nav, dynamic segments, `next/navigation`/
`next/headers`, server actions, the dev server) is covered by the hand-authored
fixture in `next-check.sh` / `next-dev-check.sh` (§4.5–4.8). The two standard SVGs
are restored from the fixture (identical bytes) because the preserved snapshot did
not include `public/`; nothing in `app/` or `next.config.ts` is added or changed.

Reproduce: `bash scripts/rsc/next-authentic-check.sh` (exit 0). Wired into
`check.sh` alongside the other RSC/Next gates.

---

## 5. Honest bottom line

The **RSC protocol spine is done and proven against real React 19.2.4 + real
`react-server-dom-webpack@19.2.4`** on a real Next app-router app: three native
graphs, both manifests, the `__webpack_*` seam, whole-document SSR-of-flight,
browser hydration, and the server-action round-trip all work end-to-end in a real
browser. What stands between this and an unmodified `create-next-app` is the
**Next-framework mapping layer**, not new protocol: a `next/font` build-time
transform (§4.1, the one hard blocker on the stock template), CSS/metadata head
injection (§4.3–4.4), the full app-router routing surface + soft navigation
(§4.5–4.6), and the Next server surface (§4.8). Each is a bounded, separately
gatable slice on top of the working spine.

Reproduce: `scripts/rsc/next-check.sh` (the passing browser gate).
