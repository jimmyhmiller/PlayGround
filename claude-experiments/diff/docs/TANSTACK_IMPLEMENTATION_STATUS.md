# TanStack implementation status

Updated: 2026-07-17

## Pinned reference

The first production target is the official TanStack Start `start-basic`
application imported at TanStack Router commit
`00d00b3155e6e8bdbdae806ff12de129c4915d86`.

The fixture lives in
`integration/tanstack-start-reference` and pins exact versions of TanStack
Start, TanStack Router, React, Vite, Nitro, TypeScript, and the related build
plugins. Its npm lockfile is committed intentionally.

Reference commands:

```console
cd integration/tanstack-start-reference
npm ci
npm run build
npm run acceptance:reference
```

## Reference artifact contract

The pinned Vite/Nitro build currently produces:

- 27 public JavaScript chunks;
- one extracted CSS asset;
- 46 server modules;
- a Node server entry;
- separate SSR/router artifacts;
- a TanStack Start manifest;
- copied public assets.

The acceptance runner verifies 13 artifact and HTTP gates, including:

- SSR of `/` with the expected application content, stylesheet, and client
  script;
- execution of the `/customScript.js` server route;
- a rendered 404 response;
- required client, server, manifest, CSS, and static-asset output.

Current status:

```text
reference: 13/13 TanStack production gates passed
diffpack:    2/13 TanStack production gates passed
```

Diffpack now emits the client `public/` layout (see "Native client `public/`
emit" below), so the `output directory` and `extracted CSS` gates pass. The
remaining eleven gates need more browser chunks (route splitting) and the whole
server/SSR build, tracked below.

The Diffpack result is deliberately not softened: it does not yet emit the
required `.diffpack-output` client/server application layout. Run the
non-failing readiness report with:

```console
npm run acceptance:status
```

Once Diffpack has an application build command, CI should use the strict
`npm run acceptance:diffpack` gate.

## First core compatibility probe

Bundling the fixture's generated `src/router.tsx` initially failed on imports
such as `~/components/DefaultCatchBoundary`. Diffpack now enables Oxc Resolver's
nearest-`tsconfig.json` discovery and has a runtime test for TypeScript `paths`
aliases.

After that fix, discovery advances to the next real blocker:

```text
cannot read src/styles/app.css?url: No such file or directory
```

This identified the next implementation slice:

1. Treat query-bearing module IDs as a resource path plus loader query. (done)
2. Implement `?url` asset modules. (done)
3. Copy content-hashed assets and return their public URL from the synthetic
   JavaScript module. (done)
4. Add global CSS and CSS-module loaders. (global CSS done; CSS modules next)
5. Carry emitted CSS/assets into the client and SSR manifests.

## Query-aware module identity and the `?url` loader (landed)

`src/resource_id.rs` (`ResourceId`) splits a specifier/id into
`(path, query, fragment)` and round-trips it losslessly. Resolution now resolves
only the path component and re-attaches the query, so `app.css` and
`app.css?url` are distinct graph keys. The load frontier dispatches a
query-bearing id before touching disk:

- `?url` reads the target, content-hashes it, copies it to
  `<output>/assets/<stem>-<hash>.<ext>`, and synthesizes an ES module
  (`export default "/assets/..."`) run through the real transformer so it links
  like any other module. Asset copy happens in `Bundler::emit_assets`, deduped
  by public name.
- `?raw` inlines the file's contents as a default string export.
- A default asset import by extension (`import logo from "./logo.svg"` for
  images/fonts/SVG/media) emits a content-hashed file and exports its URL, just
  like `?url`.
- `?tsr-split` and unrecognized queries fail with a specific, actionable error
  (never a misleading filesystem read failure).

All loaders route through one `load_special_module` dispatch (query loader,
stylesheet, or asset) shared by the parallel and incremental load paths.

Tests: `url_asset_import_emits_a_content_hashed_file_and_exports_its_public_url`
(asserts the emitted asset bytes, the exported URL, and Node execution) and
`an_unimplemented_loader_query_reports_a_specific_error`.

With `?url` handled, bundling `src/router.tsx` traverses past `app.css?url` and
advances to the next blocker deep in the TanStack packages:

```text
cannot resolve "#tanstack-router-entry": The specifiers must be a non-empty string. Received ""
```

That is a Node `package.json` `imports` (subpath `#…`) / TanStack plugin virtual
entry concern, separate from the CSS/asset pipeline.

## Global CSS extraction (landed)

A bare stylesheet import (`import "./app.css"`) is a side effect with no
bindings. The load frontier detects a `.css` path (no query) and produces an
empty JavaScript module that carries the stylesheet text; `Bundler::emit_css`
concatenates every reachable module's CSS in execution order and writes it beside
the bundle as `<output_stem>.css`. The raw CSS never leaks into the JavaScript
bundle. Test:
`global_css_side_effect_imports_are_extracted_into_one_stylesheet`.

Note: the pinned app imports its only stylesheet as `~/styles/app.css?url` (an
asset URL, handled by the `?url` loader), and that CSS is Tailwind source that
the reference build compiles via `@tailwindcss/vite`. So Tailwind compilation is
a framework-plugin (host) concern; the generic global-CSS and (next) CSS-module
loaders are core pipeline capabilities the checklist requires independently.
Remaining CSS work: CSS Modules (scoped names + export map), `@import`/`url()`
rewriting, and carrying CSS into route manifests.

## Plugin host: config bridge (landed)

The plugin-host phase has begun. A Node sidecar (`host/sidecar.mjs`, embedded via
`include_str!` and run from a temp file) answers build-time questions that need
the project's own JavaScript plugins. It does NOT run a Vite/Rollup build; it
reports config. `src/host.rs::resolve_config(root, env)` runs
`vite.resolveConfig`, extracts the environment's string->file resolver aliases
(the entry aliases such as `#tanstack-router-entry` -> the app router;
regex/function aliases are counted as skipped, never silently dropped), and
returns a `BuildConfig`. Config is fetched once per build, off the per-edit
incremental path, so the thesis guards (`docs/THESIS_GUARDS.md`) are unaffected
and stay green.

`diffpack bundle-with-host <entry> <project-root> [env]` builds through diffpack
with the host config. With aliases applied, discovery of `src/router.tsx`
advances from a handful of modules to **206 reachable modules** of the real app
graph. Remaining diagnostics identify the next slices:

- Node built-ins (`node:stream`, `node:async_hooks`, `node:stream/web`) must be
  treated as externals, not resolved/bundled (generic capability).
- Virtual modules the plugins generate (`tanstack-start-manifest:v`,
  `virtual:...`) need the sidecar `load`/`resolveId` hooks (next host slice, a
  long-lived request/response loop keyed by module id + source hash so
  incrementality is preserved).

## Node built-in externals (landed)

A specifier that is a Node built-in (`node:stream`, or a bare builtin like `fs`,
`async_hooks`) is external: `is_external_specifier` recognizes it, resolution
skips it (no diagnostic, not added to the graph), and its `require(...)` is left
in the output for the runtime to resolve. The flat fast-path strips import
bindings and cannot bind an external, so a module with externals renders through
the runtime path; the runtime's `require` now falls back to the host's native
`require` for any specifier not in the module map. Tests:
`node_builtins_are_recognized_as_externals` and
`node_builtin_imports_are_left_external_and_run` (runs under Node). The byte-parity
oracle stays green, so the linker change did not disturb determinism.

Effect on the app: `bundle-with-host router.tsx` drops from 14 diagnostics to
**1** (206 modules reachable). The sole remaining blocker is a plugin-generated
virtual module:

```text
cannot resolve "tanstack-start-manifest:v": Cannot find module 'tanstack-start-manifest:v'
```

That needs the sidecar `resolveId`/`load` hooks (the next host slice): a
long-lived request/response loop keyed by module id + source hash so per-module
invalidation and the incrementality guards are preserved across the boundary.

## Plugin host: long-lived resolveId/load hooks (landed)

The sidecar gained a long-lived `serve` mode: newline-delimited JSON
request/response over stdin/stdout, running the framework plugins' `resolveId`
and `load` hooks on demand (Vite/Rollup internals denied). `src/host.rs::Sidecar`
is the Rust client (requests serialized behind a mutex; one process reused for the
whole build, including incremental rebuilds), and `HostBridge` binds it to one
environment. The bundler consults the host ONLY when native resolution/loading
fails (`ResolutionCache.host`), so the common path stays native and the thesis
guards are unaffected (all green). Virtual/plugin-generated modules resolve
through `resolveId` and load through `load`, transformed like any module.

Verified in isolation: `tests/host_sidecar.rs` starts the sidecar and
resolves+loads the `tanstack-start-manifest:v` virtual module. Wired end to end,
`bundle-with-host` now loads that virtual module and traverses its dependencies.

Known limitation surfaced by loading it: the TanStack manifest is
**build-output-dependent**. The content the dev-server plugin returns references
dev-relative route imports (`./routes/__root`, ...) that do not map to a
production static graph, so those imports do not resolve. Producing correct
manifest content requires client/server build separation and ordering (the client
build's real chunk outputs feed the server manifest). That build-graph work, not
the host integration, is the next milestone; the `resolveId`/`load` bridge it
needs is now in place.

## Architectural boundary: build-output-dependent modules

Applying environment-specific resolve conditions (client `browser`, server `node`)
is correct and landed, but it did not remove the manifest from the client graph,
and investigation reached a real boundary worth recording before more slices.

The sidecar runs a Vite **dev** server and calls `resolveId`/`load` manually. This
is perfect for per-module, mode-agnostic transforms (JSX, route splitting,
server-fn rewrites). It is NOT sufficient for **build-output-dependent** modules
such as `tanstack-start-manifest:v`: the manifest's correct content is derived
from the *finished client build* (real chunk file names/hashes and route->chunk
mapping). A dev server returns dev-mode content whose route references
(`./routes/__root`, ...) do not exist in a production static graph, so they cannot
resolve. Producing correct content this way would require driving the plugins
through a full production build lifecycle (buildStart -> transform -> render ->
generateBundle) -- i.e. running Rollup -- which puts the build tool back in the
execution path and breaks the incremental-graph thesis.

The fork (needs a decision):

- **A / native generation (recommended):** Diffpack owns discovery, chunking, and
  emit for both environments, and **natively generates** the few build-output-
  dependent artifacts (the manifests) from its own chunk graph. The host stays for
  per-module transforms only. Preserves incrementality and keeps the build tool
  out of the path; costs a native implementation of the manifest format and
  route->chunk mapping.
- **B / build lifecycle in the sidecar:** run the real Rollup/Vite production build
  in the sidecar for these artifacts. Correct output for free, but embeds the
  build tool and forfeits the incremental thesis for that work.

Recommendation: A. Concretely, the next slices are: build and chunk the client
environment natively and emit `public/` (client chunks + CSS + assets); do the
same for the server environment; then generate the client/SSR/router/start
manifests natively from those outputs. The `resolveId`/`load` host bridge (for
transforms) and the environment conditions are the pieces already in place.

## NATIVE PIVOT: no Vite, no Rolldown, ever (decided)

Decision (hard requirement): Diffpack is a **replacement** for Vite/Rolldown, not
a host for them. The Node/Vite sidecar (dev server + resolveConfig) is **removed**
entirely (`src/host.rs`, `host/sidecar.mjs`, `tests/host_sidecar.rs` deleted).
Everything is implemented natively in Rust.

Native config replaces `resolveConfig`: `src/config.rs::derive_config(root, env)`
derives the entry aliases (`#tanstack-router-entry` -> `src/router.tsx`;
`#tanstack-start-entry`/`virtual:tanstack-start-{client,server}-entry` ->
user file or the framework default-entry) and per-environment resolve conditions
(client `[module,browser,production]`, server `[node,production,...]`) from
convention + the filesystem. Reading `srcDirectory` out of `vite.config.ts` is a
plain text read of one value, not a dependency on Vite; a native Diffpack config
format supersedes it later.

`diffpack build-app <root> [env]` builds an environment natively. The client
build reaches **220 modules** of the real app with **1** remaining diagnostic:
`tanstack-start-manifest:v`, which must be **generated natively** from Diffpack's
own chunk graph (it cannot be loaded — it is build-output-dependent). All thesis
guards stay green; the native resolver + tsconfig `paths` + externals + loaders do
the rest.

### Remaining native work (the framework transforms/generation)

- Route splitting (`?tsr-split`): split route components into lazy chunks (was
  `@tanstack/router-plugin`).
- Server functions: client stub + server registry (was `tanstack-start-core`).
- Manifest generation: client/SSR/router/start manifests from the chunk graph.
- Server (SSR) build + a Node server entry that renders documents, serves hashed
  assets, runs server routes, and 404s.
- CSS: Tailwind compilation for `app.css` (was `@tailwindcss/vite`).

These are the native reimplementations that move the 13 acceptance gates.

## Native client `public/` emit (landed)

`diffpack build-app <root> client` now **emits a real client build** to
`<root>/.diffpack-output/public/` instead of only printing diagnostics. It runs
the existing linker/emit machinery over the real 220-module app graph:

- `Bundler::emit_public(reachable, output_root, options)` rebuilds `public/` from
  scratch (no stale files linger), emits the entry chunk `public/client.js` and
  its dynamic-import chunks (`public/client.chunk-N.js`), extracts CSS beside the
  entry (`public/client.css`), and copies content-hashed assets under
  `public/assets/`. It returns a `PublicBuildSummary` whose counts are read back
  from disk, so the printed summary always matches reality.
- Only the `client` environment emits today; a non-client environment stops after
  discovery rather than pretending to write a bundle (the server/SSR build is the
  next milestone).
- `emit_public` is a build-time entry point off the incremental hot path, so the
  thesis guards (`docs/THESIS_GUARDS.md`) are unaffected and stay green.

On the pinned reference the client build currently produces **4 public `.js`**
(the entry chunk plus three dynamic-import chunks), **0 extracted `.css`**, and
**1 asset** (`app-<hash>.css`, the Tailwind source imported as `~/styles/app.css?url`).
Every emitted `.js` passes `node --check` (used only as a test oracle, never in
the build path). Acceptance moves from `0/13` to `2/13` (`output directory` and
`extracted CSS`; the CSS gate counts the `app.css` asset).

Test: `bundler::tests::emit_public_writes_a_client_layout_with_chunks_css_and_assets`
builds a small app (CSS side effect + asset import + dynamic import), emits to a
temp `public/`, and asserts the chunk/CSS/asset files, the summary/disk match, and
that a re-emit clears stale output.

### The manifest gap, handled honestly

The single remaining discovery diagnostic is still
`tanstack-start-manifest:v` (unresolvable — it is build-output-dependent and must
be generated natively). Its true cause is a **client/server isolation gap**: the
generated route tree statically imports the server-only API route
`src/routes/api/users.ts`, which imports `@tanstack/react-start/server`, pulling
`react-start-server -> start-server-core -> createStartHandler -> router-manifest`
into the client graph. `router-manifest.js` then `import()`s the virtual manifest.

The correct fix is to strip the route module's `server: {...}` block (and server
functions) in the client transform so those server-only imports never survive —
but that is the route-splitting / server-function transform work listed above, too
large for this slice. So per the honest-fallback policy, `build-app` **emits what
builds** and reports the manifest as a clearly-labelled `known gap` rather than
inventing a placeholder. The leaked server modules are dead code on the client;
the dangling `import("tanstack-start-manifest:v")` lives only inside the unused
`getStartManifest` path. No silent placeholder is emitted for the manifest.

Next native step: the client route-module transform (strip `server`/`.server`
blocks + server functions), which both eliminates this leak and — via `?tsr-split`
route splitting — lifts the browser-chunk count toward the `>= 15` gate.
