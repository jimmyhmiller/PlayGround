# Diffpack TanStack host architecture (hybrid, port-incrementally)

Updated: 2026-07-17

This document is the contract for making Diffpack build the pinned TanStack Start
application (`integration/tanstack-start-reference`). It is the spec that the
feature-iteration workflow implements against. Read it before touching host code.

## Decision

Chosen strategy: **hybrid, port incrementally**.

- **Phase 1** — Diffpack's Rust core owns the whole build pipeline (discovery,
  resolution, linking, chunking, CSS/asset handling, manifests, SSR server
  assembly, emit). The *framework* transforms it does not yet have — TanStack
  Start, TanStack Router, `@vitejs/plugin-react`, `@tailwindcss/vite` — are
  executed as their real JavaScript through a **Node sidecar** that Diffpack
  drives hook-by-hook. Vite/Rsbuild/Rollup/Nitro are **not** in the execution
  path; only the individual framework plugin hooks run, called by Diffpack.
- **Phase 2..n** — each hosted JS hook is replaced by a native Rust
  implementation behind the same plugin interface. The acceptance oracle gates
  every swap, so the build stays green the whole way. When the last hook is
  ported, the sidecar is deleted.

### Why not "just host all of Vite's plugins"

`resolveConfig` for this app yields **77 plugins across 3 environments**
(`client`, `ssr`, `nitro`). Most are Vite/Rollup *internals*
(`vite:optimized-deps`, `alias`, `vite:import-analysis`, asset/CSS/html/build
plugins, Rollup chunking). Hosting those would be re-embedding Vite and would
put Vite in the execution path, violating the acceptance gate. So the sidecar
hosts **only the framework plugins' hooks**; every capability Vite internals
provide (query IDs, `?url`/`?raw`, CSS extraction, asset hashing, import
analysis, HTML/manifest emission, chunking) is Diffpack-native from the start.

## The reference contract (what "done" means)

`integration/tanstack-start-reference/acceptance.mjs` runs 13 gates against an
output directory. Reference (`.output`) passes 13/13; Diffpack must produce
`.diffpack-output` that passes 13/13. Gates, grouped:

1. Output exists; ≥15 public `.js`; ≥1 public `.css`; ≥35 server `.mjs`.
2. `public/favicon.ico`, `public/site.webmanifest` copied.
3. `server/index.mjs` present; server artifacts contain
   `tanstack-start-manifest`, `ssr`, `router`.
4. Boot `server/index.mjs`; over HTTP:
   - `GET /` → 200 `text/html` containing `Welcome Home!!!`, `<script`,
     `stylesheet`.
   - `GET /customScript.js` → 200 `application/javascript` containing
     `Hello from customScript.js!` (a server route).
   - `GET /definitely-not-a-route` → 404 `text/html` containing
     `The page you are looking for does not exist.`

These gates are coarse (counts + HTTP behavior), not byte-parity. Byte/behavior
parity against the reference is a separate, stricter program (see the checklist
`P1: correctness program`); the host must not *depend* on coarseness — build the
real thing.

## Build environments

`resolveConfig` reports three environments. Diffpack models them explicitly:

- `client` — browser ESM chunks under `public/`. Server-only code must never
  enter here.
- `ssr` — server-rendering modules that import the client graph's server view.
- `nitro`/`server` — the Node server runtime that renders full documents,
  serves hashed assets, runs server routes/functions, and emits
  `server/index.mjs`.

The same source module may transform differently and have different dependency
edges per environment (client-only vs server-only boundaries). The module graph
is therefore keyed by `(environment, resolved_id)`.

## Module identity

A module ID is `(environment, resource_path, query, fragment)`. Query-bearing
IDs are first-class:

- `styles/app.css?url` — asset-URL module; loader returns a JS module exporting
  the hashed public URL string.
- `route.tsx?tsr-split=component` — a virtual route-split module produced by the
  router plugin.
- `foo.svg?raw`, `foo.png` — raw/asset imports.

Query parsing plus a resource/query/fragment split has landed natively in
`src/resource_id.rs` (`ResourceId`). The resolver resolves only the path
component and re-attaches the original query, so `app.css` and `app.css?url` are
distinct graph keys, and the load frontier splits the query off before touching
disk. A query-bearing id whose loader is unimplemented now returns a specific
error naming the loader and resource instead of a misleading file-not-found
crash. The loaders themselves (`?url`, `?raw`, `?tsr-split`) are the remaining
work; the identity split that everything downstream depends on is in place.

## The plugin container

Diffpack (Rust) owns the pipeline and calls hooks in Rollup order. For Phase 1
each hook is dispatched to the sidecar; for ported hooks it dispatches to Rust.

Hooks, per environment, in order:

1. `config` / `configResolved` — once per build. Sidecar returns the resolved
   environment list, entry points, and per-environment ordered plugin list.
2. `buildStart`.
3. `resolveId(id, importer, { environment })` → resolved id | external | null.
   Diffpack's oxc_resolver runs first for bare/relative specifiers; plugins can
   override (virtual modules, `?tsr-split`, server-fn stubs).
4. `load(id, { environment })` → code (+ optional map) | null. Physical files
   are read by Diffpack; virtual modules come from plugins.
5. `transform(code, id, { environment })` → code (+ map) | null, applied in
   plugin order. React JSX/Fast-Refresh, TanStack route-split, server-fn
   client/ssr rewrites, tailwind, etc. run here.
6. `buildEnd`.
7. `renderChunk` / `generateBundle` — Diffpack owns chunking; plugins may
   post-process and contribute manifest fragments.

Results are cached by `(plugin_version, environment, options_hash, module_id,
source_hash)`. A plugin declares watch files / invalidation deps.

### Isolation invariant

A client-environment build must never contain a server-only transform's output
(server-function implementations, secrets, filesystem/db code). The container
tags every module with the environment that produced it and refuses to let a
server-only module enter a client chunk. This is a hard error, not a warning.

## The sidecar protocol

A long-lived Node process (`host/sidecar.mjs`, shipped with Diffpack, run inside
the target project so it resolves the project's own plugin versions). Framing:
newline-delimited JSON over stdin/stdout (one request, one response), plus a
side channel for large `code` payloads if needed later.

Requests (Rust → Node):

- `{ "op": "resolveConfig", "root", "mode", "command" }`
  → `{ environments: [{ name, plugins: [name...], entries: [...] }], ... }`
- `{ "op": "resolveId", "env", "id", "importer" }`
  → `{ id, external } | null`
- `{ "op": "load", "env", "id" }` → `{ code, map } | null`
- `{ "op": "transform", "env", "id", "code" }` → `{ code, map } | null`
- `{ "op": "shutdown" }`

The sidecar builds one Vite `PluginContainer`-equivalent per environment from
`resolveConfig`, but **only invokes the framework plugins**; a deny-list drops
Vite/Rollup internal plugins (`vite:*`, `alias`, `commonjs`, asset/css/html/
build/`optimized-deps` families) so their behavior is Diffpack-native. The
deny-list is explicit and tested: an unrecognized non-framework plugin that
would otherwise silently no-op must surface as a diagnostic, never a silent
skip.

Every unimplemented op/hook returns a structured error that names exactly what
is missing. No silent nulls that could be mistaken for "plugin declined".

## Native capabilities Diffpack must own in Phase 1

These are the Vite-internal behaviors the host replaces natively. Each is an
independently workable feature slice, oracle-observable:

1. Query-bearing module IDs (`?url`, `?raw`, `?tsr-split`, generic query).
2. Asset pipeline: content-hash, copy to `public/assets/`, return public URL;
   rewrite references after hashing.
3. CSS: global side-effect imports, CSS Modules, `@import`/`url()`, production
   extraction + dedupe, associate CSS with chunks/routes.
4. Multi-entry, per-environment chunk graphs with stable content hashes and
   configurable naming.
5. Client / SSR / router / server-function / asset **manifests** in the exact
   shape TanStack Start's server runtime reads.
6. SSR server assembly: `server/index.mjs` that renders full documents via the
   Start server entry, injects scripts/styles/preloads/hydration data from
   manifests, serves hashed assets, runs server routes (`customScript.js`), and
   returns a rendered 404.
7. Env-var handling: `.env`, `import.meta.env`, public prefixes, no unprefixed
   secret in client output.

## Port ledger (Phase 2+)

Each row starts `hosted` (JS via sidecar) and moves to `native` (Rust), oracle
green at every step. Keep this table honest — it is the migration's ground
truth.

| Transform | Source plugin | Status |
| --- | --- | --- |
| React JSX + automatic runtime | `@vitejs/plugin-react` | hosted |
| React Fast Refresh (dev) | `@vitejs/plugin-react` | hosted |
| Route tree generation | `@tanstack/router-plugin` | hosted |
| Automatic route code-splitting (`?tsr-split`) | `@tanstack/router-plugin` | hosted |
| Server-function client stub | `tanstack-start-core::server-fn:client` | hosted |
| Server-function ssr/server registry | `tanstack-start-core::server-fn:*` | hosted |
| Compiler virtual modules | `tanstack-start-core:compiler-virtual-module` | hosted |
| Import protection (client/server leak guard) | `tanstack-start-core:import-protection` | hosted |
| Start manifest generation | `tanstack-start-core` | hosted |
| Tailwind scan + generate | `@tailwindcss/vite` | hosted |

## Workflow-driven development

Feature work is driven by `.claude/workflows/diffpack-feature-iteration.js`. One
iteration: triage the next smallest oracle-observable slice from the checklist +
this doc, implement it (native or host), and gate on `cargo test`, `cargo clippy
-D warnings`, and `npm run acceptance:status` (gate count must not regress). The
status doc (`docs/TANSTACK_IMPLEMENTATION_STATUS.md`) records the current gate
count and the next blocker after every iteration.

Rules:

- Stubs must throw a clear, specific error (never return a plausible-but-wrong
  value). A half-built hook that returns `null` silently is forbidden.
- No gate is "passed" by loosening the oracle. The oracle only gets stricter.
- Every fallback the linker takes becomes a named fixture before it is trusted.
- Server code and secrets in any client artifact is a release blocker and has a
  standing gate.
