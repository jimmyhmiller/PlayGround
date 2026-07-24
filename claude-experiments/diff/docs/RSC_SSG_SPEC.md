# RSC_SSG_SPEC — Full SSG (build-time prerender) + Dev HMR benchmark

Authoritative implementation spec, synthesized from three parallel deep-dives, all
verified against the real `next@16.2.11` build on disk in `integration/next-app-router`
and against the diffpack render pipeline source. This spec supersedes ad-hoc plans for
these two goals. Read `docs/RSC_NEXT_GAP.md`, `docs/RSC_SPEC.md`, `docs/RSC_NEXT_SPEC2.md`
first; this builds ON the existing RSC spine and does NOT redo it.

Two goals:

- **(A) Full SSG** — build-time static prerender of every static route to on-disk
  `.html` + `.rsc` files, `generateStaticParams` enumeration for dynamic routes,
  static/dynamic classification, a genuinely dumb static file server (zero per-request
  render, zero child processes), plus a hybrid serve mode in the existing orchestrator.
  This closes the last honest boundary: serving becomes pure files for static routes.
- **(B) Dev HMR benchmark** — `diffpack dev` vs `next dev --turbopack` edit-to-update
  latency + startup, browser-observed, honestly caveated. Measurement, not correctness;
  non-blocking.

---

## 0. Ground truth (verified, do NOT re-derive)

**How Next classifies the UNMODIFIED fixture** (ran `next build --turbopack`, read
`prerender-manifest.json`, `find .next/server/app -name '*.html'`):

| Route            | Next legend | Why                                                        |
|------------------|-------------|-----------------------------------------------------------|
| `/`              | ○ Static    | no request reads, no dynamic segment                      |
| `/about`         | ○ Static    | same                                                      |
| `/blog/[slug]`   | ƒ Dynamic   | reads `await cookies()` AND has no `generateStaticParams` |
| `/go`            | ƒ Dynamic   | `export const dynamic = "force-dynamic"`                  |
| `/error-demo`    | ƒ Dynamic   | `export const dynamic = "force-dynamic"`                  |

Adding a clean SSG route (`generateStaticParams` returning N sets, no request reads)
yields `● /docs/[slug]` → one `<route>.html` + `<route>.rsc` per param set;
`prerender-manifest.json` lists the concrete pathnames under `routes` and the pattern
under `dynamicRoutes` with a `fallback` (`null` = dynamicParams true → on-demand;
`false` = dynamicParams false → unlisted = 404).

**Emitted layout Next uses** (per static route, under `.next/server/app/`): `<route>.html`
(full document with flight inlined), `<route>.rsc` (RAW flight — exactly what diffpack
already serves at `?__rsc=1`), `<route>.meta`, `<route>.segments/*` (PPR segment cache —
**out of scope**). Root `/` → `index.html`/`index.rsc`. No `<link rel="expect" blocking="render">`
in the non-PPR prerendered HTML (grep confirmed absent) — a full `onAllReady` prerender
ships a complete, hydration-ready document, which is exactly what diffpack's SSR entry
already produces.

**diffpack's per-request pipeline IS the SSG render, verbatim** (`scripts/rsc/next-server.mjs`
GET handler L169–267):
1. build `serverConsumerManifest` (divergent-id `ssrModuleMapping`, L63–81);
2. `runReactServer(["render", pathname, clientManifestPath], reqCtx)` spawns the
   `rsc-render/server.mjs` child → raw flight bytes on stdout + `{status,params,redirect,notFound}`
   on fd 3 (`writeMeta`, `next_adapter.rs` L1130);
3. `renderFlightToDocument(new Uint8Array(flight), serverConsumerManifest, flight.toString("base64"), params, {pathname,search})`
   (`ssr_entry_module`, `next_adapter.rs` L1227) → full HTML document via `onAllReady`,
   inlining `window.__DIFFPACK_FLIGHT__=<base64>` + params + url;
4. `?__rsc=1` (L253) serves the SAME raw flight buffer as `text/x-component`.

Moving steps 2–4 to build time and writing the outputs to files = SSG. Reuse them
verbatim; do NOT reimplement.

**Build order today** (`scripts/rsc/next-check.sh` L50–72):
`build-app <root> client` → `build-app <root> react-server` → `rm -rf out/rsc-render; cp -r out/server out/rsc-render` → `build-app <root> ssr` → boot orchestrator.
At serve time: `rsc-render/server.mjs` = react-server render child; `server/server.mjs` = SSR bundle;
both manifests (`client-references-manifest.json`, `server-references-manifest.json`) on disk.

**Key source anchors** (line numbers approximate; grep by symbol):
- `src/next_adapter.rs`: `struct Route` (L279), `enum Seg` (L172), `parse_segment` (L193),
  `segments_js` (L247), `scan_metadata` (L114), `discover_routes_dir` (L334),
  `rsc_entry_module` (L858, route table at L884, default-only imports `import M{i}` at L913,
  op dispatch `main()` at L1080, `render`/`action` ops, final `throw new Error("... unknown op")` L1155),
  `ssr_entry_module` / `renderFlightToDocument` (L1185/L1227), `next_headers_shim` (L1891,
  the `requestAls.getStore()` empty-store throw at L1916/L1934).
- `src/main.rs`: `build-app` dispatch (L253), environment parse (L271), the
  `environment == "client"` vs `else` graph-build split (L451/L510), `usage()` (L789).
- `src/config.rs`: `derive_config` conditions/entry match (L74/L87).
- `scripts/rsc/next-server.mjs`: `fail()` (L37), manifest load + `serverConsumerManifest`
  (L46–81), `getRenderFlightToDocument` (L86–104), `runReactServer` (L111–151), GET handler
  (L169–267).
- `check.sh`: `rsc_gate` (L87), the 7 rsc rows (L94–100). Battery is 13/13 today; SSG adds row 14.

**Explicitly out of scope** (must WARN/hard-error, never silently mishandle): PPR segment
cache (`.segments/`), ISR (`export const revalidate` — needs a running server + incremental
cache; `revalidate:0` maps to force-dynamic; any other value → clear WARN "prerendered once,
will not revalidate"), nested multi-dynamic-segment `generateStaticParams` BFS merge
(>1 dynamic segment in a chain each exporting `generateStaticParams` → HARD ERROR until the
merge is implemented), `force-static` fake-empty request reads (MVP: hard error, documented).

---

## Naming / layout decisions (fixed for all slices)

- Fourth build phase: **`diffpack build-app <root> static`** (parallels `client`/`react-server`/`ssr`).
  Optional flag **`--static-export`** = pure-export strictness (any non-prerenderable route
  HARD-ERRORS naming it, matching Next `output: 'export'` E87).
- Output dir: **`.diffpack-output/static/`**. Root `/` → `static/index.html` + `static/index.rsc`;
  `/about` → `static/about.html` + `static/about.rsc`; `/products/a` → `static/products/a.html` + `static/products/a.rsc`.
- Plan file (Rust → node): **`.diffpack-output/static/prerender-plan.json`** (input to the prerenderer).
- Manifest file (node → world): **`.diffpack-output/static/prerender-manifest.json`** (what was
  written + what was skipped-with-reason).
- Shared node render seam: **`scripts/rsc/next-render-core.mjs`** (imported by the orchestrator,
  the prerenderer, and — optionally — dev).
- Prerenderer: **`scripts/rsc/next-prerender.mjs`**. Dumb static server: **`scripts/rsc/next-static-serve.mjs`**.
- SSG gate: **`scripts/rsc/next-ssg-check.sh`** (check.sh row 14).
- RouteKind enum: **`Static | ForceStatic | Ssg | Dynamic`**.

---

## SLICE 1 — SSG fixture route + Rust route classification (pure, unit-tested)

**Goal:** a native source-scan classifier that reproduces Next's fixture result exactly,
plus a real SSG route to classify. No node, no I/O beyond reading page sources.

**Fixture (new files, must ALSO pass `next build`):**
- `integration/next-app-router/app/products/[id]/page.tsx` — a plain async Server Component:
  `export function generateStaticParams() { return [{ id: "a" }, { id: "b" }]; }` and a default
  component rendering e.g. `<main id="product">product: {id}</main>` (reads `params`, NO
  cookies/headers/searchParams). This is the SSG-enumeration proof. Do NOT touch `/blog/[slug]`
  (gates depend on its cookie-reading dynamic behavior) or the `authentic-create-next-app/` subtree.

**`src/next_adapter.rs`:**
- Add `enum RouteKind { Static, ForceStatic, Ssg, Dynamic }` and, on `struct Route` (L279),
  fields: `kind: RouteKind`, `has_generate_static_params: bool`, `dynamic_config: Option<String>`
  (`"force-dynamic"|"force-static"|"error"`), `dynamic_params: bool` (default `true`),
  `revalidate: Option<String>`.
- Add `fn scan_route_config(page: &Path, source: &str) -> RouteConfig` next to `scan_metadata`
  (L114) — a substring/regex source scan (same shape as `scan_metadata`/`scan_next_font`) for:
  `export ... generateStaticParams`, `export const dynamic =`, `export const dynamicParams =`,
  `export const revalidate =`. Also flag references to `next/headers` `cookies()`/`headers()`/`draftMode()`
  and the `searchParams` prop.
- Add `fn classify_route(route_has_dynamic_segment: bool, cfg: &RouteConfig) -> RouteKind`
  applying, in order:
  1. `dynamic == "force-dynamic"` (or `revalidate == "0"`) → `Dynamic`;
  2. `dynamic == "force-static"` or `"error"` → `ForceStatic`;
  3. reads request state at top level (cookies/headers/draftMode/searchParams) → `Dynamic`;
  4. has a Dynamic/CatchAll/OptionalCatchAll segment: `has_generate_static_params` → `Ssg`, else → `Dynamic`;
  5. otherwise → `Static`.
  Conservative default when the scan is ambiguous: `Dynamic` (skip), never `Static`.
  If `revalidate` is present and not `"0"`: still classify by the above but set a WARN flag
  (surfaced at build time), never treat as permanently static silently.
- Populate these in `discover_routes_dir` (L334) where each `Route` is pushed (read the page
  source once). Emit `kind`/`dynamic_params`/`has_generate_static_params` into the ROUTES table
  in `rsc_entry_module` (L884) so the render child sees them too (a `kind: "static"|"ssg"|"dynamic"|"forceStatic"`
  string field per route entry).

**GATE (Slice 1):** `cargo build --release && cargo test --lib` — a NEW `#[test]` in
`next_adapter.rs` runs discovery+classification on `integration/next-app-router` and asserts:
`/` and `/about` → `Static`; `/blog/[slug]`, `/go`, `/error-demo` → `Dynamic`;
`/products/[id]` → `Ssg`. Plus `clippy -D warnings` clean. `./check.sh` stays 13/13.
Manual: `cd integration/next-app-router && ./node_modules/.bin/next build` still succeeds
(the new fixture route is valid Next).

---

## SLICE 2 — `staticparams` op + prerender-plan emission (Rust codegen)

**Goal:** enumerate `generateStaticParams` in the app's own React runtime, and hand the node
side a machine-readable plan.

**`src/next_adapter.rs` — `rsc_entry_module`:**
- For every `Ssg` route, ALSO emit a namespace import beside its default import:
  `import * as NS{i} from "<page>";` (pages are default-only today, L913 — `generateStaticParams`
  is a NAMED export, so a namespace binding is required). Build a small generated
  `STATIC_PARAM_ROUTES` table mapping `url_path` → the `NS{i}` namespace.
- Add a `staticparams` op to `main()` (L1080), BEFORE the final `throw`:
  ```
  if (op === "staticparams") {
    const routePath = rest[0];
    const ns = STATIC_PARAM_ROUTES[routePath];
    if (!ns) throw new Error(`rsc-entry staticparams: route ${JSON.stringify(routePath)} is not an Ssg route`);
    if (typeof ns.generateStaticParams !== "function")
      throw new Error(`rsc-entry staticparams: route ${routePath} has no generateStaticParams export`);
    const combos = await ns.generateStaticParams({ params: Promise.resolve({}) });
    if (!Array.isArray(combos)) throw new Error(`rsc-entry staticparams: ${routePath} generateStaticParams did not return an array`);
    process.stdout.write(JSON.stringify(combos));
    return;
  }
  ```
  Keep the existing final `throw new Error("... unknown op ...")` (L1155).
- **Nested-segment guard:** if a route chain has >1 dynamic segment each exporting
  `generateStaticParams`, `classify_route`/plan emission HARD-ERRORS naming the route
  ("nested generateStaticParams BFS merge not implemented"). The fixture has only
  single-dynamic-segment SSG.

**`src/next_adapter.rs` — plan writer:** add
`pub fn write_prerender_plan(project_root: &Path, out_dir: &Path) -> Result<usize, String>`
that re-runs discovery+classification and writes `.diffpack-output/static/prerender-plan.json`:
```json
[
  { "path": "/", "kind": "static", "segments": [], "file": "index" },
  { "path": "/about", "kind": "static", "segments": [{"k":"static","v":"about"}], "file": "about" },
  { "path": "/products/[id]", "kind": "ssg", "hasGenerateStaticParams": true,
    "dynamicParams": true, "segments": [{"k":"static","v":"products"},{"k":"dynamic","v":"id"}] },
  { "path": "/blog/[slug]", "kind": "dynamic", "reason": "reads request state; no generateStaticParams", "segments": [...] },
  { "path": "/go", "kind": "dynamic", "reason": "force-dynamic", "segments": [...] }
]
```
`segments` reuses the exact `segments_js` shape (`[{k,v}]`) so the node side substitutes params
by walking segments. Return the route count.

**GATE (Slice 2):**
1. NEW `#[test]`: rsc-entry source contains `import * as NS` for the products route, a
   `STATIC_PARAM_ROUTES` table, and the `staticparams` op branch.
2. Runtime enumerate: after `build-app <fx> react-server` + `cp server rsc-render`, run
   `node "$out/rsc-render/server.mjs" staticparams /products/[id] "$out/client-references-manifest.json"`
   → stdout is exactly `[{"id":"a"},{"id":"b"}]`.
3. `write_prerender_plan` produces `static/prerender-plan.json` with `/` `Static`, `/products/[id]`
   `Ssg`, `/blog/[slug]`+`/go`+`/error-demo` `Dynamic` (with `reason`).
`./check.sh` stays 13/13; `clippy -D warnings` clean.

---

## SLICE 3 — Shared render core refactor (node, behavior-preserving)

**Goal:** ONE seam for manifest-load + flight-render, shared by the orchestrator and the
prerenderer, with ZERO behavior change to the orchestrator (this is a pure extraction).

**New `scripts/rsc/next-render-core.mjs`** — extract from `next-server.mjs` and export:
- `loadManifests(outputDir)` → `{ clientRefs, ssrRefs, serverConsumerManifest, clientManifestPath }`
  (the L46–81 logic: read both manifest files, build the divergent-id `moduleMap` +
  `serverConsumerManifest`).
- `getRenderFlightToDocument(ssrEntry, { dev })` → the memoized dynamic import of the SSR bundle
  returning `renderFlightToDocument` (L86–104, `pickRender` included; the `?v=mtime` cache-bust
  matters for dev).
- `makeRunReactServer(rscRenderEntry)` → a `runReactServer(args, stdinBody)` closure identical to
  L111–151 (spawn child, collect stdout flight + fd-3 meta JSON, reject on nonzero).

**Refactor `scripts/rsc/next-server.mjs`** to import and use these three (delete the inlined
copies). No functional change — same HTTP behavior, same bytes.

**GATE (Slice 3):** the THREE existing gates must stay green unchanged — run
`scripts/rsc/next-check.sh`, `scripts/rsc/next-authentic-check.sh`, `scripts/rsc/next-dev-check.sh`
(all pass). `./check.sh` stays 13/13. This slice adds no new gate; its correctness IS
"the orchestrator behaves identically after extraction."

---

## SLICE 4 — The prerenderer + build wiring (files on disk)

**Goal:** `build-app <root> static` writes `.html` + `.rsc` for every static/SSG concrete
pathname, copies `public/`, and writes `prerender-manifest.json`. Dynamic routes are SKIPPED
and recorded with a reason — never silently dropped.

**New `scripts/rsc/next-prerender.mjs`** — `node next-prerender.mjs <outputDir> [--static-export]`
(the app's own React runtime — the explicitly-allowed oracle, same as the orchestrator):
1. `const { serverConsumerManifest, clientManifestPath } = loadManifests(outputDir)`;
   `const render = await getRenderFlightToDocument(join(outputDir,"server","server.mjs"), { dev:false })`;
   `const runReactServer = makeRunReactServer(join(outputDir,"rsc-render","server.mjs"))`.
2. Read `static/prerender-plan.json`. For each route:
   - `Static`/`ForceStatic` → `pathnames = [url_path]`.
   - `Ssg` → `const combos = JSON.parse(await runReactServer(["staticparams", route.path, clientManifestPath]).then(r=>bufToString(r.flight)))`;
     substitute each combo into `route.segments` to build concrete pathnames (dynamic → one part;
     catch-all → `string[]` joined with `/`, each part `encodeURIComponent`d for the URL; file path
     mirrors the URL path). Record the concrete pathnames.
   - `Dynamic` → SKIP; push `{ path, skipped: route.reason || "dynamic" }` into the manifest.
3. For each concrete pathname: `const { flight, params } = await runReactServer(["render", pathname, clientManifestPath], "" /* EMPTY ctx */)`;
   `const html = await render(new Uint8Array(flight), serverConsumerManifest, bufToBase64(flight), params, { pathname, search: "" })`.
   Write `<outputDir>/static/<file>.html` and `<outputDir>/static/<file>.rsc` (raw flight buffer).
   `/` → `index`. **Any render error propagates and FAILS the prerender naming the pathname** — no
   `try/catch` swallow. If the error carries digest `DIFFPACK_DYNAMIC_BAILOUT` (see below), the
   message additionally says "route <p> read request state (cookies/headers) but was classified
   static — mark it Dynamic"; still a hard failure, never a silent demotion.
4. Copy `<outputDir>/public/` → `<outputDir>/static/` (so `/rsc.css`, `/_diffpack-image/*`, and
   all assets are colocated and the pages are self-contained under a dumb server).
5. Write `<outputDir>/static/prerender-manifest.json`:
   `{ static: ["/","/about","/products/a","/products/b"], dynamic: [{path,reason}...], generatedAt }`.
6. `--static-export`: if ANY route is `Dynamic`/`ForceStatic`-with-request-reads, EXIT NONZERO
   naming the offending route(s) ("`/blog/[slug]` has no generateStaticParams; a static export
   cannot serve it").

**`src/next_adapter.rs` — tag the bailout:** in `next_headers_shim` cookies()/headers() (L1916/L1934),
set `err.digest = "DIFFPACK_DYNAMIC_BAILOUT"` on the thrown Error (so the prerenderer's message
can distinguish a classifier gap from a generic bundle/render failure). This does NOT change the
existing throw-when-outside-request behavior — it only tags it.

**`src/main.rs` — `build-app` `static` branch:** add an early branch (before the graph-build
split at L451) for `environment == "static"`. It builds NO graph. It:
1. HARD-ERRORS if any of `rsc-render/server.mjs`, `server/server.mjs`,
   `client-references-manifest.json`, `server-references-manifest.json` is missing — naming which,
   and "run the client → react-server (cp → rsc-render) → ssr builds first" (mirrors the
   orchestrator's `fail()` checks);
2. calls `next_adapter::write_prerender_plan(project_root, out_dir)`;
3. spawns `node <repo>/scripts/rsc/next-prerender.mjs <out_dir> [--static-export]` via
   `std::process::Command`, streams its stderr, and returns `Err` on nonzero exit.
Add `static` to the `usage()` string (L789). (This keeps the bundling native Rust; the node
process renders React exactly as the orchestrator does — the app's own runtime.)

**GATE (Slice 4):** after `client → react-server (cp→rsc-render) → ssr → build-app <fx> static --no-minify`:
- `static/index.html`, `static/about.html`, `static/products/a.html`, `static/products/b.html` EXIST;
  each is a full document (`<!DOCTYPE html>` + `id="app-shell"` + inlined `__DIFFPACK_FLIGHT__`);
  `products/a.html` body contains `product: a`, `products/b.html` contains `product: b`.
- `static/index.rsc`, `static/about.rsc`, `static/products/a.rsc` EXIST, non-empty, raw flight
  (does NOT start with `<!DOCTYPE`).
- Exactly two products files (one per `generateStaticParams` combo); NO `static/products/[id].html`.
- Dynamic routes NOT prerendered: no `static/blog/**.html`, no `static/go.html`, no
  `static/error-demo.html`. `static/prerender-manifest.json` records `/blog/[slug]`, `/go`,
  `/error-demo` under `dynamic` with a reason.
- `static/rsc.css` present (public/ colocated).
- `./check.sh` stays green.

---

## SLICE 5 — Dumb static server + browser hydration + soft-nav (THE honesty proof) — check.sh row 14

**Goal:** a static file server that imports NEITHER RSC bundle and spawns NO child, serving the
prerendered files, and a real browser that hydrates + soft-navigates from them.

**New `scripts/rsc/next-static-serve.mjs`** — `node next-static-serve.mjs <outputDir>/static [port]`,
pure `fs`:
- `GET /` → `index.html`; `GET /x` → `x.html`; `GET /x/y` → `x/y.html`.
- `GET /x?__rsc=1` → `x.rsc` with `content-type: text/x-component` (the prerendered soft-nav source).
- `GET /<asset>` (has a file extension / exists under static/) → the file with its MIME.
- Unknown path with NO prerendered file: consult `prerender-manifest.json`; if the path matches a
  route recorded `dynamic` → `501` with a CLEAR body ("route <p> is dynamic; a pure static export
  cannot serve it — use the orchestrator (next-server.mjs)"). Otherwise `404` with a clear body.
- `GET /x?__rsc=1` for a dynamic route → `404` (no `.rsc` on disk) so the client Router falls back
  to a full navigation (which also 404s on a pure static site — honest scope).
- `POST /_action/` → `501` ("no server on a static export").
- It imports NEITHER `rsc-render/server.mjs` NOR `server/server.mjs`, and calls NO `spawn`/`exec`.
  This is the load-bearing property (enforced structurally by the gate grep AND by design).

**New `scripts/rsc/next-ssg-check.sh`** (wired into `check.sh` as row 14 via `rsc_gate`):
1. Native build: `client → react-server (cp→rsc-render) → ssr → build-app <fx> static`.
2. FILES-ON-DISK — the Slice-4 assertions (index/about/products a+b `.html`+`.rsc`, dynamic skipped).
3. STRUCTURAL — grep-assert `next-static-serve.mjs` contains no `spawn(`, no `child_process`, and
   no `import`/`require` of `rsc-render` or `server/server`.
4. DUMB SERVE — `node scripts/rsc/next-static-serve.mjs "$out/static" 0`:
   - `curl /` and `curl /about` return the prerendered HTML and are BYTE-EQUAL to the on-disk
     `static/index.html` / `static/about.html`;
   - `curl /products/a` byte-equals `static/products/a.html` and contains `product: a`;
   - `curl '/about?__rsc=1'` returns `content-type: text/x-component` raw flight (no `<!DOCTYPE`);
   - `curl /blog/anything` returns `501` (or `404`) with the clear message — NEVER index HTML,
     NEVER a render;
   - `curl '/blog/x?__rsc=1'` returns `404`.
5. REAL BROWSER (agent-browser, ABSOLUTE paths; `agent-browser skills get core --full` first):
   - open the static-served `/`; assert the document HYDRATED (`document.__reactContainer$*` /
     a `__reactFiber$` present) with ZERO console errors/warnings;
   - click `#inc` → island count `5 → 6` (hydration from a PRERENDERED file);
   - set `window.__softnav = "kept"`, click the `/about` link → soft-nav fetches `/about?__rsc=1`
     (the prerendered `.rsc`) and diff-renders: `[window.__softnav, location.pathname, !!document.querySelector("#app-shell")].join("|") === "kept|/about|true"`;
   - `history.back()` restores `/`.

**GATE (Slice 5):** `scripts/rsc/next-ssg-check.sh` exits 0 (all of the above). Wire it into
`check.sh` after L100 as `rsc_gate "Next SSG (prerender + dumb static serve + hydrate + soft-nav)" scripts/rsc/next-ssg-check.sh integration/next-app-router`.
Battery becomes 14/14. The three prior gates + `next-dev` stay green.

---

## SLICE 6 (recommended) — `--static-export` strictness + hybrid orchestrator serve

**Goal:** prove no-silent-drop, and let the EXISTING orchestrator serve prerendered files with
zero per-request render for static routes (= `next start`), falling through to per-request render
only for dynamic routes (the documented hybrid surface).

**`--static-export` enforcement** (already coded in Slice 4's prerenderer + main.rs branch): this
slice adds the gate assertion.

**`scripts/rsc/next-server.mjs` — hybrid serve:** at boot, load `static/prerender-manifest.json`
once (if present). In the GET handler, AFTER the static-asset check (L188) and BEFORE the render
spawn (L200): map `url.pathname` → `static/<file>.html` (root → `index.html`); if that file exists,
serve it directly (`text/html`, zero render); likewise `?__rsc=1` → the prerendered `.rsc`. For a
path NOT prerendered, fall through to the existing per-request render child unchanged (dynamic +
on-demand). No change to the render child.

**GATE (Slice 6):** extend `next-ssg-check.sh` (or a sibling section):
- `diffpack build-app <fx> static --static-export` EXITS NONZERO with a message NAMING `/blog/[slug]`
  (and/or `/go`) as un-prerenderable — proving no silent drop.
- Boot `next-server.mjs`; `curl /` and `curl /about` return HTML BYTE-EQUAL to the on-disk
  `static/*.html` (proves served-from-file, not re-rendered); `curl /blog/hello` still renders
  per-request with cookie support (existing next-check gate 7 behavior unbroken).
`./check.sh` stays green (14/14).

---

## SLICE 7 — Dev HMR benchmark: `diffpack dev` vs `next dev --turbopack` (goal B, non-blocking)

**Goal:** honest, browser-observed edit-to-update latency + startup for both dev servers on the
shared fixture. Measurement only — it does NOT gate correctness (that is `next-dev-check.sh` D1–D4).

**Verified primitives** (M2 Max, Node v26.5.0): `next dev --turbopack` has TWO startup numbers —
"Ready" ~165ms (accepts a request) vs first-byte ~1700ms (routes compile ON DEMAND at first
request); `diffpack dev` front-loads all three graphs at boot → first-byte ~334ms, no lazy-compile
spike. Both servers render DOM-compatible pages with the SAME selectors (`#counter`, `#heading`,
`#inc`). Browser `performance.timeOrigin + performance.now()` and Node `Date.now()` read the SAME
OS wall clock (verified ~ms parity) — so a cross-process `t0(Node) → t1(browser)` delta is FAIR on
one machine.

**New `scripts/bench-dev-hmr.mjs`** (a scaffold already drafted at
`scripts/bench-dev-hmr.mjs` per the dev deep-dive — finalize + run):
- Snapshot `app/Counter.tsx` + `app/page.tsx`; restore in a `finally` AND between samples (a leaked
  nonce edit would corrupt later gates — mirror the `next-dev-check.sh` trap pattern).
- STARTUP: 5 cold starts per server (wipe `.next` / `.diffpack-output` before each), record BOTH
  "ready" (a request to a nonexistent path returns) and "first-byte" (200 on `/`). Report medians.
- WARM HMR, two edit classes, measured via a SELF-TIMESTAMPING MutationObserver in the page (NOT
  harness-side polling — `agent-browser eval` spawns a CLI per call, tens of ms, which must stay
  OUT of the timed path):
  - **client-text** (Fast Refresh, state-preserving on BOTH): arm an observer that stamps
    `window.__mark = performance.timeOrigin + performance.now()` the instant `#counter` text
    contains a UNIQUE per-sample nonce; `t0 = Date.now()`; write the nonce edit to `Counter.tsx`;
    read `window.__mark` back → `delta = mark - t0` (readback latency does NOT enter the number);
    restore. `path = "hot"`.
  - **server-text** (`#heading` in `page.tsx`): diffpack does a FULL RELOAD (fresh react-server
    child per GET — hook state lost) which destroys the observer, so stamp `t1` from the FRESH
    document's `performance.timeOrigin + navigation.responseEnd` (`path = "reload"`); Turbopack
    does an RSC refresh (streams a new flight, reconciles, no reload — `path = "hot"`). Report
    latency AND semantics side by side; equal ms ≠ equal outcome.
  - Quarantine the FIRST edit per class as "cold-first-compile" (Turbopack pays route compile here;
    diffpack already paid at boot); then 20 warm samples → median / p95 / min / max.
- Write `bench/results/dev-hmr-results.json`; the two servers run sequentially on the same fixture.
- Server specs: `diffpack dev . <port>`; `next dev --turbopack --port <port>` with
  `NEXT_TELEMETRY_DISABLED=1`.

**docs/COMPETITIVE_BENCHMARKS.md** — add a "Dev HMR (edit-to-update)" section. Pull numbers from
`bench/results/dev-hmr-results.json` (do NOT hand-type). Print VERBATIM the caveats: (1) different
HMR models — client edit both state-preserving, server edit diffpack=reload vs Turbopack=RSC-refresh;
(2) Turbopack "Ready" vs first-byte gap from lazy compile, diffpack front-loads; (3) cross-process
wall-clock validity only on one machine; (4) we observe DOM MUTATION not paint, and INCLUDE
fs/watcher latency deliberately (real UX) so the number is strictly > diffpack's internal
post-detection rebuild ms parsed from `[dev] next rebuilt ... in X.Xms` (`dev_server.rs` ~L2629);
(5) machine/OS/Node/date labeled like the existing cold-build section. NEVER a bare "N× faster
than Turbopack".

**GATE (Slice 7):** `cargo build --release && (cd integration/next-app-router && npm install) && node scripts/bench-dev-hmr.mjs`
completes with, in `bench/results/dev-hmr-results.json`, for BOTH servers: a startup block
(median ready + first-byte) and hmr blocks for "client-text (Fast Refresh)" and
"server-text (RSC / reload)" each with a warm `{median,p95,min,max}` over 20 samples + a separate
cold-first, AND the two edited fixture files byte-identical to their originals afterward. NOT wired
as a hard check.sh threshold (machine-dependent); optionally add a SKIP-if-no-node/chrome liveness
row running `--samples 3 --starts 2` for non-regression only. `next-dev-check.sh` (D1–D4) remains
the correctness backbone; `./check.sh` stays green.

---

## Cross-cutting HARD RULES (repo owner — enforce in every slice)

1. **No stub hides failure.** A misclassified-static route that reads request state must FAIL
   LOUDLY at prerender naming the pathname (empty ctx → tagged `DIFFPACK_DYNAMIC_BAILOUT` throw) —
   never a silent try/catch demotion. The prerenderer lets every non-bailout render error propagate
   and aborts the build naming the pathname + stderr. Only the specific bailout digest (and the
   no-gsp-on-dynamic-segment case) may classify a route dynamic; anything else ABORTS.
2. **Build/prerender stays native Rust.** The node prerender process renders React (the app's own
   runtime) exactly as the orchestrator already does — the explicitly-allowed oracle. The bundling
   of all three graphs stays native Rust; do NOT turn `build-app` into a node-spawning bundler.
3. **The dumb server is load-bearing.** `next-static-serve.mjs` imports neither RSC bundle and
   spawns no child — enforced by the gate grep AND by design. If it ever renders per-request, the
   core SSG claim is false.
4. **Byte-identity to the live orchestrator.** Prerendered HTML must equal what `next-server.mjs`
   produces for the same route under an EMPTY context (guaranteed by reusing `renderFlightToDocument`
   + `onAllReady` verbatim via the shared core; no `rel=expect` shell — verified absent in Next's
   non-PPR prerender). The Slice-3 extraction must NOT change orchestrator behavior — re-run
   next-check/next-authentic/next-dev to prove it.
5. **Honest scope for dynamic routes.** A pure static export serves only its static routes; a
   soft-nav to a dynamic route 404s the `.rsc` → client full-nav → 404 (a static host's default).
   Dynamic routes remain the documented hybrid-orchestrator surface (Slice 6). Never paper over
   with a hidden render path.
6. **Keep the tree building + green at all times.** `cargo build --release` + `cargo test --lib` +
   `clippy -D warnings` pass after every slice; `./check.sh` goes 13/13 → 14/14 and stays there.
   ISR/PPR/nested-gsp/force-static-empty-reads are OUT OF SCOPE and must WARN or HARD-ERROR, never
   silently mishandle.

---

## Slice → gate summary

| Slice | Deliverable                                              | Gate (exact)                                                                 |
|-------|---------------------------------------------------------|------------------------------------------------------------------------------|
| 1     | SSG fixture route + Rust classification                 | `cargo test --lib` classify test (fixture matches Next); `next build` OK      |
| 2     | `staticparams` op + `prerender-plan.json`               | enumerate → `[{"id":"a"},{"id":"b"}]`; plan has correct kinds; unit test      |
| 3     | `next-render-core.mjs` extraction                       | next-check + next-authentic + next-dev unchanged-green                        |
| 4     | `next-prerender.mjs` + `build-app <root> static`        | `.html`+`.rsc` on disk for static+SSG; dynamic skipped in manifest           |
| 5     | `next-static-serve.mjs` + browser hydrate/soft-nav      | `next-ssg-check.sh` (dumb server, byte-equal, hydrate, soft-nav) → check.sh 14 |
| 6     | `--static-export` strictness + hybrid orchestrator      | build errors naming `/blog/[slug]`; orchestrator serves prerendered byte-equal |
| 7     | `bench-dev-hmr.mjs` + COMPETITIVE_BENCHMARKS Dev-HMR     | JSON has startup+hmr blocks for both servers; fixtures restored (non-blocking) |
