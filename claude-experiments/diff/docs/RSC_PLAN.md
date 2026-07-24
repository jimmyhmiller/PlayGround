# Next.js / RSC support — implementation plan

The Turbopack half of the original charter: React Server Components and server
actions, built on the `"use client"` / `"use server"` module boundaries, with a
pinned Next.js reference app under the same gate discipline as the rest, and an
on-disk persistent cache. This document is the grounded, sliced plan; each slice
lands behind an oracle, as everything else here does.

## The model, and what diffpack already has

RSC rests on two directive-marked module boundaries plus a serialization runtime:

- **`"use server"`** — a module (or function) whose exports are *server actions*.
  The client build ships a thin RPC stub keyed by a stable id; the server build
  keeps the real body and registers it under that id; a resolver dispatches HTTP
  calls. **Diffpack already implements exactly this shape** for TanStack's
  `createServerFn` (`src/server_fn.rs`: per-target rewrite, deterministic
  SHA-256 id, generated resolver module). RSC's bare `"use server"` is that
  machinery generalized from the `createServerFn().handler()` call to *any*
  exported function in a directive-marked module.
- **`"use client"`** — a *client boundary*. In the React Server graph, importing
  it must not pull its code server-side; the server instead emits a **client
  reference** (a stable id + the client chunk hosting the real module) and
  serializes that into the payload. The client build bundles the real module and
  resolves the reference. This is the INVERSE of the server-fn split.
- **The RSC flight payload** — the server renders the Server Component tree to a
  serialized stream in which client references stand in for `"use client"`
  subtrees; the client reconstructs the tree and mounts the real client
  components. This is the react-server-dom-{webpack,turbopack} runtime.

Diffpack also already has the load-bearing infrastructure the boundaries need: a
per-environment build (client vs server) with independent reachability and emit
(`bundler.rs`, `dev_server.rs`), content-addressed splitting, and a
build-output-dependent manifest mechanism (the client route manifest + the
server-fn resolver are generated from a scan and injected as virtual modules).

## Target: React-compatible (react-server-dom-webpack)

The chosen target is React's real RSC protocol, so the boundaries must match
`react-server-dom-webpack` exactly. The contract, verified against the installed
`react-server-dom-webpack@experimental`:

- **Client reference** (server sees this for a `"use client"` export):
  `{ $$typeof: Symbol.for("react.client.reference"), $$id: "<moduleId>#<export>",
  $$async }`. Produced by `createClientModuleProxy("<moduleId>")` (whose property
  access yields the per-export reference) or `registerClientReference(impl, id,
  name)`, both from `react-server-dom-webpack/server` under the `react-server`
  export condition.
- **Server reference** (for a `"use server"` export): `registerServerReference(fn,
  "<moduleId>", "<name>")` on the server; `createServerReference("<moduleId>#<name>",
  callServer)` on the client.
- **Client-references manifest** (`bundlerConfig`): `{ "<moduleId>": { id, chunks,
  name } }`, where `id` is passed to `__webpack_require__(id)` and `chunks` is a
  flat `[chunkId, chunkFilename, ...]` for `__webpack_chunk_load__`. The client's
  `resolveClientReference` splits `$$id` at `#`, looks up the module entry, and
  loads it. Diffpack must expose `__webpack_require__` / `__webpack_chunk_load__`
  globals over its registry (the integration seam) and emit this manifest.
- **Flight runtime**: `react-server-dom-webpack/server`
  (`renderToReadableStream`/`renderToPipeableStream`) encodes the tree with client
  references; `.../client` (`createFromReadableStream`) reconstructs it, resolving
  references through the manifest + the `__webpack_*` seam.

## Slices (each gated)

- **Slice 0 — directive detection. DONE.** `src/rsc.rs::detect_directive`
  robustly identifies a module's `"use client"` / `"use server"` prologue
  directive (AST-based: ignores non-prologue occurrences, comments, strings;
  quote/whitespace-insensitive). Unit-tested. Every later slice starts by asking
  this what a module is.

- **Slice 2a — `"use client"` server transform. DONE + runtime-verified.**
  `src/rsc.rs::transform_use_client_server` rewrites a `"use client"` module for
  the React Server build into `createClientModuleProxy("<moduleId>")` re-exports
  (one per original export), so NONE of the component code reaches the server and
  each export becomes a real client reference. `module_reference_id` is the shared
  id (canonical path). Unit-tested (shape), AND validated against the installed
  `react-server-dom-webpack@experimental`: the emitted module's exports come back
  with `$$typeof === Symbol.for("react.client.reference")` and
  `$$id === "<moduleId>#<export>"` under `node --conditions=react-server`.
  Remaining in Slice 2: emit the client-references **manifest** and wire the
  transform into a React Server build target (below).

- **Slice 1 — `"use server"` generalization.** Extend the server-fn rewrite so a
  `"use server"` module's *every* exported function is split (client RPC stub /
  server handler / resolver entry) using the same deterministic-id + resolver
  contract `server_fn.rs` already has. Gate: a fixture module with bare
  `"use server"` exports produces a client stub that round-trips to the server
  handler (extend the existing server-fn oracle).

- **Slice 2 — `"use client"` boundary.** Introduce the React Server target
  (distinct from SSR): in it, a `"use client"` module is replaced by
  client-reference exports (`{ $$typeof: Symbol.for("react.client.reference"),
  $$id, $$chunk }`) and recorded in a **client-reference manifest**; in the client
  target it is bundled as real code and registered so the runtime can resolve the
  reference by id. Gate: an oracle asserting a `"use client"` module is absent
  from the server graph (its code never reaches the server bundle) yet present in
  the client graph, and that the reference id matches on both sides — the same
  cross-graph-agreement property the server-fn id already guarantees.

- **Slice 3 — RSC flight runtime.** Wire a react-server-dom-* runtime: server
  renders the Server Component tree to the flight stream with client references
  in place; client bootstraps from the stream and mounts client components via the
  manifest. Reuse `react-server-dom-webpack` (or `-turbopack`) rather than
  reimplement React's serializer. Gate: a minimal RSC app renders a server
  component containing a `"use client"` interactive child; headless-browser oracle
  asserts SSR HTML + hydration + the client component's interactivity, with zero
  server code in the client bundle.

- **Slice 4 — pinned reference app + gate.** Full Next.js `next build` is a large
  surface (app-router compile, route generation, its own compiler). Start the gate
  on a **minimal hand-built RSC app** (Slice 3's fixture, grown) to exercise the
  boundary + flight machinery under the harness discipline; then graduate to a
  pinned Next.js app once the boundaries hold, treating Next's conventions
  (app-router file routing, layouts, metadata) as the next mapping layer — much of
  which the route-tree generation already models for TanStack.

- **Slice 5 — persistent on-disk cache.** The other Turbopack weapon and a second
  cold-start lever: persist the per-module transform + per-chunk render caches to
  disk keyed by content hash, so a cold `diffpack build` reuses unchanged work
  across process invocations. Independent of RSC; sequenced last so the RSC graph
  shape is settled before it is cached.

## Sequencing note

Slices 1 and 2 are the boundary atoms and are the highest-leverage next work: they
reuse the server-fn id/resolver/manifest machinery almost directly and are fully
gatable without the flight runtime. Slice 3 is the largest (the React serializer
integration) and is where a real runtime dependency enters. Do 1 → 2 → 3 in order;
4 and 5 follow.
