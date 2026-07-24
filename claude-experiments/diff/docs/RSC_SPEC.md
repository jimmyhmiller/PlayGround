# RSC Implementation Spec — React Server Components + Server Actions on diffpack

Authoritative, ordered implementation spec for React Server Components and server
actions in diffpack, targeting the **real** `react-server-dom-webpack` protocol
(the same wire format Next.js app-router uses). It reconciles the five research
deep-dives into one buildable plan. It supersedes the sliced sketch in
`docs/RSC_PLAN.md` for everything after Slice 0 / Slice 2a (which remain DONE).

The end state of the gated slices (R1–R4) is a **minimal-but-real RSC app** that:
renders a Server Component to a flight stream, server-renders that flight to HTML,
hydrates a `"use client"` island in a **real browser**, and runs a `"use server"`
action end-to-end over the wire — all verified by `agent-browser`. R5 is the
extension path to an actual Next.js app.

---

## 0. Ground truth already in the tree (do not redo)

- `src/rsc.rs`
  - `detect_directive(path, source) -> Option<RscDirective>` (AST prologue, gated). DONE.
  - `module_reference_id(path) -> String` — canonical absolute path; the **opaque
    moduleId** shared by every graph. This is the single id authority for RSC.
  - `transform_use_client_server(path, source) -> Option<String>` — rewrites a
    `"use client"` module for the react-server graph into
    `createClientModuleProxy("<moduleId>")` re-exports. DONE + runtime-verified
    against `react-server-dom-webpack@experimental`.
  - `module_exports(path, source)` — export-name enumeration (reuse for all RSC transforms).
- `src/server_fn.rs` — the *exact shape* RSC server-references generalize: per-target
  split, deterministic id, generated resolver virtual module, `scan_project_*`
  walk, `apply_edits`/`quote` helpers, cross-graph id-agreement test. REUSE its
  machinery; do not reinvent.
- `src/transform.rs`
  - `Target` enum (`Client` | `Server`, `Server` = `#[default]`), threaded
    config → `Bundler.target` → `transform_module_with_options` with **zero extra
    plumbing**. A new variant rides these channels automatically.
  - `transform_module_with_options` — the source-override pipeline: `route_split`
    then `server_fn` each produce `Option<String>` and rebind `source`, then the
    rewritten source flows through the normal parse/lower so its imports are
    collected as real deps. RSC transforms hook in the same way.
- `src/config.rs` — `derive_config(root, env)`: per-env `conditions` (the resolve
  export-condition set, consumed only in `bundler.rs::resolve_options`) and
  `target`. `ENVIRONMENTS: [&str;3]` is cosmetic (not iterated on the build path).
- `src/bundler.rs` — `render_runtime` (registry: `globalThis["__diffpack_runtime:<entry>"]`
  = `{register, require, ...}`, `require(runtimeId)` returns a module's exports;
  `chunk_names: HashMap<DenseModuleId,String>`, `runtime_ids: &[Option<usize>]`);
  `BROWSER_GLOBALS_PRELUDE` prepended to the main BrowserEsm chunk (the seam-install
  point); split chunks self-register via `__runtime.register(...)` on import.
- `src/main.rs` (`build-app`) + `src/dev_server.rs` — the manifest-as-virtual-module
  precedent: client build persists `ClientRouteManifest`; server build reads it and
  injects `START_MANIFEST_SPECIFIER`; server-fn resolver injected as a virtual
  module from a project pre-scan. The RSC manifests follow this pattern exactly.
- `src/server_runtime/*.mjs` — real embedded runtime files via `include_str!`
  (per the no-source-in-strings rule). All new RSC runtime JS lives in real
  `.js`/`.mjs` files under `src/rsc_runtime/`, embedded the same way.

### Pinned dependencies (install in every RSC fixture; `node_modules` gitignored)

`react`, `react-dom`, `react-server-dom-webpack` all at
`0.0.0-experimental-28cd4bb0-20260723` (the versions the contract below was
verified against). The oracle MUST fail loudly if they are missing, never skip.

---

## 1. The verified contract (reconciled — this overrides RSC_PLAN.md)

RSC needs a **third build graph** and **two manifests**. Both were empirically
verified against the pinned package; the reconciled shapes are:

### Directives → per-graph transform

| module | react-server graph | ssr + client graphs |
|---|---|---|
| `"use client"` | replaced by `createClientModuleProxy(moduleId)` re-exports (`rsc::transform_use_client_server`) — **no component code reaches the server** | bundled as **real code**, registered in the registry under its runtime id |
| `"use server"` | `registerServerReference(fn, moduleId, name)` footer (keeps real body) | client: `createServerReference("moduleId#name", callServer)` stubs (drops body); ssr(server): same as react-server (keeps + registers) |

The id is **`module_reference_id(path)`** (canonical path) everywhere; a server
reference `$$id` is `"<moduleId>#<name>"`. Both sides derive it from the same
function, so client stub ↔ server registration ↔ resolver key agree (the exact
invariant `server_fn.rs` already guarantees and tests).

### Condition boundary (mandatory, non-negotiable)

- `react-server-dom-webpack/server` resolves to the real writer **only** under the
  `react-server` export condition; the default `./server.js` **throws at import**.
- `react` under the `react-server` condition is a **different React** than the
  SSR/client React. The two cannot coexist in one module graph.
- ⇒ the react-server render is its **own bundle** (Target::ReactServer, conditions
  include `react-server`) and runs in its **own worker/child process**. diffpack's
  build stays native Rust — it only *emits* that bundle + the manifest; it never
  runs the flight render itself.

### Manifest #1 — SERVER RENDER manifest (`bundlerConfig`, 2nd arg to `renderToReadableStream`)

Flat, keyed by the **bare moduleId** (the `$$id` prefix). `resolveClientReferenceMetadata`
looks up `config[$$id]`; on miss it splits `$$id` at the **last `#`** and uses the
prefix as key, suffix as export name. So keying by bare moduleId is correct and
sufficient:

```json
{ "<moduleId>": { "id": <clientRuntimeId>, "chunks": [<chunkId>, "<chunkFile>", ...], "name": "*" } }
```

- **`id` is the CLIENT build's numeric runtime id** (`runtime_ids[dense]`), because
  it is serialized raw into the flight (`N:I[<id>,<chunks>,<export>]`) and consumed
  by the **browser** where `bundlerConfig` is `null` and metadata passes straight
  through to `__webpack_require__(id)`. `__webpack_require__ === __rt.require`, which
  takes exactly that numeric runtime id. (Resolves the finding-2 vs finding-5
  disagreement: **key = canonical path; value.id = numeric client runtime id.** Do
  NOT set value.id to the path — that would force the seam to do a path→id lookup.)
- **`chunks`** is a FLAT even-length `[chunkId, chunkFile, ...]`. `[]` when the
  module lands in the main entry chunk (already loaded). Otherwise the single
  hosting split chunk `[chunkId, chunkFile]` — diffpack chunk headers self-load
  their prerequisites, so listing only the host is complete. Use `chunkFile` as the
  `chunkId` too (or emit an explicit `chunkId→file` map global).
- **`name: "*"`** is safe: the real export name arrives via the `$$id` split; the
  manifest entry's own `name` is only read on the node-SSR-consumer path.

Because the moduleId (canonical path) is identical across graphs, the same manifest
bridges the react-server render and the client bundle.

### Manifest #2 — SSR CONSUMER manifest (`serverConsumerManifest`, for `createFromReadableStream` on the node SSR pass)

```json
{ "moduleMap": { "<id>": { "<export>": {"id","chunks","name"}, "*": {...} } },
  "serverModuleMap": { "<moduleId>": {"id","chunks","name"} } | null,
  "moduleLoading": { "prefix": "", "crossOrigin": null } }
```

- `moduleLoading` is **required** (reading `.prefix` off `undefined` crashes).
- `moduleMap` is keyed by the render-emitted `id` (flight `metadata[0]`), NESTED by
  export. If the SSR graph and the client graph share the same runtime-id scheme
  (they should — same `module_reference_id`, same registry), `moduleMap` is an
  **identity map** and can be generated mechanically from the client manifest.
- `serverModuleMap` resolves `"use server"` refs embedded in flight; `null` when the
  app has no server actions in the rendered tree.

### The webpack seam (browser globals over diffpack's registry)

The pinned build reads **three** globals, all of which must be functions **before**
`react-server-dom-webpack/client.browser` evaluates (it dereferences
`__webpack_require__.u` at module-init). Install them in the client **main-chunk
prelude** (same slot as `BROWSER_GLOBALS_PRELUDE`), never inside a factory:

```js
globalThis.__diffpack_chunkFilenames = { /* chunkId: "publicPath/chunkFile" */ };
var __rt = globalThis["__diffpack_runtime:<entry>"];
function __webpack_require__(id){ return __rt.require(id); }
__webpack_require__.u = function(c){ return globalThis.__diffpack_chunkFilenames[c]; };
globalThis.__webpack_require__ = __webpack_require__;
globalThis.__webpack_get_script_filename__ = __webpack_require__.u;
globalThis.__webpack_chunk_load__ = function(c){
  var f = globalThis.__diffpack_chunkFilenames[c];
  if (f === undefined) throw new Error("__webpack_chunk_load__: unknown chunk id " + c); // HARD ERROR, no silent fallback
  return import(f); // diffpack chunk self-registers on import; import() is URL-cached
};
```

### Runtime APIs (verified export surfaces)

- server (react-server cond): `renderToReadableStream(model, clientManifest, opts?)`,
  `renderToPipeableStream`, `registerServerReference(fn, moduleId, name)`,
  `createClientModuleProxy`, `decodeReply(body, serverMap, opts?)`, `decodeAction`.
- client (browser): `createFromReadableStream(stream, opts?)` (`opts.callServer`,
  `opts.serverConsumerManifest`), `createFromFetch`, `createServerReference(id, callServer)`,
  `encodeReply(value) -> Promise<string|FormData>` (string for JSON-able args;
  FormData when args carry Blob/File/functions/server-refs).
- `createServerReference(id, callServer)` returns `action(...args)` whose body is
  `callServer(id, args)`; the app supplies `callServer` as the transport.

---

## 2. Ordered slices

Each slice keeps `cargo build --release` and `cargo test --release --lib` green and
lands behind the stated GATE. Do them in order; R1→R2→R3→R4 build the app, R5 extends
to Next. Every stub is a **hard error naming the construct** (repo rule).

---

### Slice R1 — Introduce the react-server build environment

**Goal:** a third specialization graph that resolves `react-server-dom-webpack/server`
under the `react-server` condition and turns `"use client"` modules into client
references, leaving `"use server"` as a *hard error* until R2.

**Changes (file-level):**

1. `src/transform.rs` — add `ReactServer` to `enum Target` (keep `#[default]` on
   `Server`), doc: "the RSC graph; use-client → client-reference re-exports,
   use-server → server references; resolves under the `react-server` export
   condition; server-like for tanstack env helpers and node resolution."
2. `src/transform.rs::transform_module_with_options` — at the TOP (after the JSON
   early-return, **before** `route_split` and `server_fn`), add an RSC block gated
   on `target == Target::ReactServer`, using the existing `Option<String>`
   source-override pattern:
   - `detect_directive == Client` → `source = rsc::transform_use_client_server(...)`
     (`.expect` is wrong here; on `None` fall through — a use-client module always
     returns `Some`, but guard anyway).
   - `detect_directive == Server` → `rsc::transform_use_server_server(...)` which in
     R1 returns `Err("react-server build: 'use server' server-reference transform
     not yet implemented for <path>")`; route the `Err` through the same
     diagnostics-return shape the `server_fn` `Err` arm at ~line 204 uses (this fn
     returns `TransformResult`, not `Result` — do NOT use `?`).
   - Ordering matters: the use-client rewrite must run before `route_split` (a
     use-client route module becomes a single client reference, not a split module).
3. `src/transform.rs` — the isomorphic `match self.target` (~line 1531): add
   `Target::ReactServer => server`. Unreachable in practice (`apply_env_transform`
   returns early for non-`Client`) but required to compile.
4. `src/server_fn.rs` — treat react-server as server:
   - `match target` at ~124 and ~165: `Target::Server => ...` becomes
     `Target::Server | Target::ReactServer => ...`.
   - `if target == Target::Server` (export-unexported guard, ~178): change to
     `if target != Target::Client`.
   - `if target == Target::Client` (client DCE sweep, ~157): unchanged (correct).
5. `src/config.rs::derive_config`:
   - `conditions` match: add before `_`:
     `"react-server" => ["react-server","node","production","wasm","unwasm"].as_slice(),`
   - `target` match: add `"react-server" => Target::ReactServer,` before `_`.
   - `entry` match: `"react-server" => server_entry.clone()` for now (a dedicated
     RSC render entry is emitted by R4's fixture; a missing entry already hard-errors
     in main.rs). Add a `// R4: dedicated flight-render entry` note.
   - Extend `ENVIRONMENTS` to include `"react-server"` (cosmetic).
6. `src/rsc.rs` — add `pub fn transform_use_server_server(path, source) -> Result<Option<String>, String>`
   returning the R1 hard error `Err(...)` for a `"use server"` module (real impl in R2),
   `Ok(None)` otherwise.

**GATE:**
```
cargo build --release          # compiler enumerates every missed match arm
cargo test --release --lib     # no regressions
```
Plus a resolution smoke test (fixture with `react-server-dom-webpack@experimental`
installed):
```
./target/release/diffpack build-app integration/rsc-reference react-server --no-minify
node scripts/rsc/assert-react-server-env.mjs integration/rsc-reference/.diffpack-output
```
`assert-react-server-env.mjs` asserts: (a) the build resolved
`react-server-dom-webpack/server` (its react-server subpath appears in the emitted
graph — proves the condition wiring; if it 404s the whole env is inert), and (b) a
`"use client"` module's emitted server code contains `createClientModuleProxy` and
**not** the component body (e.g. no `useState`). A `"use server"` module in the
react-server build produces the hard error (asserted via the build's diagnostics).

---

### Slice R2 — `"use server"` generalization + server-action RPC runtime

**Goal:** module-level `"use server"` becomes real RSC server references, with a
working client→server round-trip over HTTP. RSC-native
(`registerServerReference`/`createServerReference` + `encodeReply`/`decodeReply`),
id = `"<moduleId>#<name>"` (NOT server_fn's SHA-256 — the react protocol fixes the
id shape and both sides must derive it identically).

**Changes:**

1. `src/server_fn.rs` — make `apply_edits` and `quote` `pub(crate)` (or lift both to
   a new `src/js_edit.rs` and re-export). No behavior change.
2. `src/rsc.rs`:
   - `pub fn action_reference_id(path, name) -> String` = `format!("{}#{}", module_reference_id(path), name)`.
   - `pub fn transform_use_server_client(path, source) -> Option<String>` (Server
     only): fresh module —
     `import { createServerReference } from "react-server-dom-webpack/client";`
     `import { callServer } from "#diffpack-call-server";` then per export
     `export const NAME = createServerReference("<moduleId#NAME>", callServer);`
     (default → `export default createServerReference("<moduleId#default>", callServer);`).
     Drops all bodies + server-only imports.
   - `transform_use_server_server(path, source) -> Result<Option<String>, String>`
     (replaces the R1 stub): prepend
     `import { registerServerReference as __rsr } from "react-server-dom-webpack/server";`,
     keep source verbatim, append `__rsr(NAME, "<moduleId>", "NAME");` per **named**
     export; default via `apply_edits` rewriting `export default <expr>` →
     `export default __rsr(<expr>, "<moduleId>", "default");`. **Hard-error** on
     function-level inline `"use server"` (closure actions) and anonymous default
     exports — naming the construct, never a silent pass-through.
   - `pub fn scan_project_server_actions(root) -> Result<Vec<ActionEntry>, String>`
     (copy `server_fn::scan_directory`/`is_module_file`, gate on
     `source.contains("use server")` + `detect_directive == Server`, enumerate
     `module_exports` as action names, sort by id, dedup).
   - `pub fn generate_action_resolver_module(entries) -> String`:
     `const manifest = { "<moduleId#name>": { importer: () => import("<path>"), exportName: "<name>" }, ... };`
     + `async function getServerActionById(id){ const i=manifest[id]; if(!i) throw new Error("diffpack rsc: no server action registered for id "+id); const m=await i.importer(); const fn=m[i.exportName]; if(typeof fn!=="function") throw new Error("diffpack rsc: action id "+id+" is not a function export "+i.exportName); return fn; }`
     + exports. Reuse `quote`.
   - Constants: `pub const ACTION_RESOLVER_SPECIFIER = "#diffpack-rsc-action-resolver";`
     `pub const CALL_SERVER_SPECIFIER = "#diffpack-call-server";`
     `pub const ACTION_HANDLER_SPECIFIER = "#diffpack-rsc-action-handler";`
     `pub const ACTION_ENDPOINT = "/_action/";` header `x-diffpack-action-id`.
3. `src/rsc_runtime/call_server.js` (NEW, `include_str!`):
   ```js
   import { encodeReply, createFromFetch } from "react-server-dom-webpack/client";
   export async function callServer(id, args) {
     const body = await encodeReply(args);
     const isForm = typeof body !== "string";
     const response = fetch("/_action/", {
       method: "POST",
       headers: { "x-diffpack-action-id": id, ...(isForm ? {} : { "content-type": "application/json" }) },
       body,
     });
     return createFromFetch(response, { callServer });
   }
   ```
4. `src/rsc_runtime/action_handler.js` (NEW, `include_str!`):
   ```js
   import { decodeReply, renderToReadableStream } from "react-server-dom-webpack/server";
   import { getServerActionById } from "#diffpack-rsc-action-resolver";
   export async function handleServerAction(request, clientManifest = {}) {
     const id = request.headers.get("x-diffpack-action-id");
     if (!id) throw new Error("diffpack rsc: POST to /_action/ missing x-diffpack-action-id header");
     const ct = request.headers.get("content-type") || "";
     const body = ct.includes("multipart/form-data") ? await request.formData() : await request.text();
     const args = await decodeReply(body, {});
     const fn = await getServerActionById(id);
     const result = await fn.apply(null, args);
     const stream = renderToReadableStream(result, clientManifest);
     return new Response(stream, { headers: { "content-type": "text/x-component" } });
   }
   ```
5. `src/transform.rs` — after the `server_fn` block, add an RSC use-server block
   gated on `source.contains("use server")` + `detect_directive == Server`: `Client`
   → `transform_use_server_client`; `Server | ReactServer` → `transform_use_server_server`
   (propagate `Err` through the diagnostics-return shape). This is the generic
   directive; the `createServerFn` path stays as-is for TanStack modules.
6. `src/main.rs` (`build-app`, alongside the server-fn resolver ~line 338) and
   `src/dev_server.rs` (~line 908): register `ACTION_RESOLVER_SPECIFIER` from
   `generate_action_resolver_module(scan_project_server_actions(root)?)`, and the two
   runtime virtual modules `CALL_SERVER_SPECIFIER` / `ACTION_HANDLER_SPECIFIER` from
   the `include_str!` sources. Print the count like the server-fn line. The Rust dev
   server only **forwards** POSTs (as today); the `/_action/` endpoint lives in the
   emitted server runtime.

**GATE:**
```
cargo test --release --lib     # rsc unit tests: id agreement, body-drop, footer-register, default, hard-errors
node scripts/rsc/action-roundtrip.mjs   # runs BOTH conditions (template: scratchpad/rsc-probe)
```
`action-roundtrip.mjs` builds a `"use server"` fixture through the client + server
transforms and asserts: (a) client stub id === server `$$id` === resolver key; and
(b) end-to-end `encodeReply(args)` → `handleServerAction` (decodeReply → getServerActionById
→ apply → renderToReadableStream) → `createFromReadableStream` returns the action's
real result. It must fail loudly if the RSC deps are absent.

---

### Slice R3 — Client-references manifest + the `__webpack_*` seam

**Goal:** emit Manifest #1 from the client build and install the browser seam over
the registry, so a flight stream that contains a `"use client"` child resolves to
the real client module in the browser.

**Changes:**

1. `src/rsc.rs` — `ClientReferenceEntry { id: usize, chunks: Vec<ChunkRef>, name: String }`,
   `ClientReferencesManifest` (map keyed by `module_reference_id`), serde to the
   `{ "<moduleId>": {id,chunks,name} }` JSON. Unit-test the shape against the verified
   example `{"/app/src/Counter.tsx":{"id":42,"chunks":["cc","<file>"],"name":"*"}}`.
2. `src/bundler.rs` — `fn client_references_manifest(&self, runtime_ids, chunk_names, reachable) -> ClientReferencesManifest`:
   iterate reachable modules, filter to `detect_directive(path, source) == Client`,
   emit `{ id: runtime_ids[dense], chunks: [] if in main entry chunk else [chunkId, chunkFile], name: "*" }`.
3. `src/main.rs` — after the client emit (next to the `client_route_manifest` write
   ~line 389), persist `client-references-manifest.json` to `.diffpack-output`
   (new `manifest.rs` constant + read/write, mirroring `ClientRouteManifest`).
   Regenerate on every client emit (ids are build-derived; the **key** is stable).
4. `src/bundler.rs::render_runtime` — when the client build contains any
   `"use client"` module (gate strictly, so non-RSC output is byte-identical),
   prepend the **seam prelude** (section 1) to the main `BrowserEsm` chunk alongside
   `BROWSER_GLOBALS_PRELUDE`, and emit `globalThis.__diffpack_chunkFilenames = {…}`
   from `chunk_names`. Must be in the entry prelude (before the module-graph IIFE),
   because `client.browser` reads `__webpack_require__.u` at module-init.
5. React-server build (R4's render entry) reads `client-references-manifest.json`
   and passes it as `renderToReadableStream`'s 2nd arg. Inject it as a virtual module
   in `main.rs` for the react-server env, same pattern as `START_MANIFEST_SPECIFIER`.

**GATE:**
```
cargo test --release --lib     # manifest JSON shape + cross-build id-agreement unit test
node scripts/rsc/seam-proof.mjs   # two-process proof (template: scratchpad/rsc-seam-proof)
```
`seam-proof.mjs`: server (`--conditions=react-server`) renders flight **with** the
manifest → browser-side (no react-server condition) reconstructs + SSR-renders
through the seam → asserts the HTML contains the client component's **real** output
(e.g. `REAL-COUNTER:from-server`). Asserts `$$id` base === manifest key.

---

### Slice R4 — Three flight entries → the minimal RSC app milestone (browser-verified)

**Goal:** the full end-to-end app: Server Component → flight → SSR HTML → browser
hydration of a `"use client"` island → a `"use server"` action round-trip, all in a
real browser via `agent-browser`.

**Fixture — `integration/rsc-reference/`** (install `react`/`react-dom`/`react-server-dom-webpack@experimental`):
- `src/app/page.tsx` — async Server Component; `await`s a trivial data fetch;
  imports `"use client"` `<Counter initial={5}/>`; imports the `"use server"`
  `increment` action and passes it to `<Counter>`.
- `src/app/Counter.tsx` — `"use client"`, `useState` counter with a `<button>` that
  (a) increments local state (proves hydration) and (b) calls the passed server
  action and shows its result (proves the action round-trip).
- `src/app/actions.ts` — `"use server"`, `export async function increment(n){ return n + 1 }`
  (server-only body; MUST be absent from the client bundle).
- `src/server.mjs` — the emitted Node server the fixture runs (build stays Rust; this
  is emitted output run on Node as the oracle). Routes:
  - `GET /` → run the **react-server render entry** in a worker/child started with
    `--conditions=react-server`, get the flight stream; feed it to the **SSR entry**
    (`createFromReadableStream(stream, {serverConsumerManifest})` → `React.use` →
    `react-dom/server` `renderToReadableStream` → HTML shell) that **inlines the raw
    flight** into the HTML (bootstrap script pushing chunks to a global array) and
    boots the browser bundle.
  - `POST /_action/` → `handleServerAction(request, clientManifest)`.
- `src/client.mjs` — the browser entry: seam prelude already installed by the runtime
  (R3); `createFromReadableStream(inlinedFlight, { callServer })` → `hydrateRoot`.

**Three entries diffpack emits** (one bundle each, pinned to its condition):
1. **react-server render** — Target::ReactServer; `"use client"` → client refs;
   imports `renderToReadableStream` + the render manifest. Runs in a worker with the
   `react-server` condition.
2. **SSR** — Target::Server (normal React); `"use client"` bundled as real code +
   registered; imports `createFromReadableStream` (`react-server-dom-webpack/client`)
   + `react-dom/server`.
3. **browser** — Target::Client; seam prelude; `createFromReadableStream`
   (`react-server-dom-webpack/client.browser`) + `hydrateRoot`.

**Changes:** the fixture files above; `config.rs` react-server `entry` now maps to
`src/app/page.tsx`'s render wrapper (or a dedicated `src/rsc-entry.tsx`); the
`serverConsumerManifest` generated as a virtual module (identity `moduleMap` from the
client manifest; `serverModuleMap` from `scan_project_server_actions`; `moduleLoading:{prefix:""}`).

**GATE (real browser via `agent-browser`):**
```
scripts/rsc/rsc-check.sh integration/rsc-reference
```
which: builds all three graphs with `diffpack build-app`; boots `src/server.mjs`;
then `agent-browser open` the app and asserts, in order:
1. SSR HTML (pre-hydration `curl /`) contains the Server Component text AND the
   client island's initial state (`count: 5`) — proves flight render + SSR-of-flight.
2. The client bundle contains **no** server-only code (grep the emitted client chunk:
   no `increment` body, no action internals) — proves the boundary.
3. After load, clicking the button increments the count — proves hydration + the
   `__webpack_*` seam + `hydrateRoot`.
4. The button's server-action call returns and its result renders — proves
   `encodeReply` → `/_action/` → `decodeReply` → dispatch → flight → `createFromFetch`
   end-to-end.

`cargo build --release` + `cargo test --release --lib` stay green.

---

## 3. Extension to a real Next.js app (Slice R5 — notes, staged)

R1–R4 build the **portable RSC protocol spine** — the same
`react-server-dom-webpack` wire format, manifests, and `__webpack_*` seam Next uses
internally. What remains for an actual `next build` app is a **mapping layer**, not
new protocol:

- **App-router file conventions** (`layout`/`page`/`loading`/`error`/`not-found`/
  `route`/`template`, `[param]`, `(groups)`, `@slots`, `(.)intercepts`) → a route
  tree. diffpack already generates a TanStack route tree from `src/routes/`
  (`route_tree.rs`); this is a known mapping, not a new engine.
- **`next/*` shims:** `next/navigation` (`useRouter`/`usePathname`/`redirect`/
  `notFound`), `next/link`, `next/image`, `next/font`, `next/headers`.
- **The Next server:** request routing, streaming the SSR shell + flight, client-side
  **soft-navigation** (fetch flight per segment), and server-action dispatch via a
  `server-reference-manifest` (diffpack's `scan_project_server_actions` + resolver is
  the same shape).
- **The second consumer manifest** (`ssrModuleMapping`) only if the SSR graph's module
  ids diverge from the browser's. Keeping one `module_reference_id` scheme across
  graphs makes it an identity map (already the plan) — but a real Next build with
  divergent ids needs it explicitly. Do NOT claim Next parity until the SSR-of-flight
  graph + this manifest are exercised against a pinned Next canary app.
- **Metadata API**, `middleware`, and the various Next manifests
  (`build-manifest`, `app-paths-manifest`, `react-loadable-manifest`,
  `next-font-manifest`) as the app surface demands.

**Sequencing:** land R1→R2→R3→R4 (hand-built fixture, fully browser-gated), then pin a
minimal `create-next-app` app-router project and graduate the gate to it one
convention at a time, reusing the route-tree mapping.

---

## 4. Key risks (carry forward)

- **Condition split is unavoidable.** The react-server render is its own bundle AND
  its own process/worker with `--conditions=react-server`. Never unify it with SSR.
  If the bundled writer still throws at runtime, spawn the worker with the flag.
- **Manifest id coupling.** Manifest #1's `id`/`chunks` are CLIENT-build artifacts
  serialized for the SERVER render. Regenerate after every client emit; stability
  rests on the canonical-path **key**, not the build-derived id.
- **Seam init order.** `client.browser` reads `__webpack_require__.u` at module-init —
  the seam MUST be in the entry prelude, before the module-graph IIFE. Getting this
  wrong throws at import.
- **Version drift.** The three required globals (incl. `__webpack_get_script_filename__`)
  are specific to the pinned build. Pin the deps in every fixture; the oracle runs
  against the same pin the app uses; re-verify on any bump.
- **No silent fallbacks.** `__webpack_chunk_load__` unknown-id, missing manifest
  entry, missing action id, and the R1 use-server stub all THROW named errors. A
  silent no-op here surfaces as an unrelated React hydration error much later.
- **`encodeReply` FormData path.** Args carrying Blob/File/functions/server-refs
  serialize to FormData, not a string; the endpoint branches on content-type and uses
  `request.formData()`. If the server runtime lacks a Fetch `Request` with
  `.formData()`, hard-error clearly rather than mis-parse.
- **Scope guard.** Module-level `"use server"` only. Function-level inline
  `"use server"` and anonymous default exports hard-error (name the construct).
