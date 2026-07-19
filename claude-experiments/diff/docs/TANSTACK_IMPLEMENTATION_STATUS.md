# TanStack implementation status

Updated: 2026-07-18 (second real app validated; client server-fn leak fixed)

## Second real app + client server-function leak (landed)

Validated diffpack against a **second, independent official Start app** (the
`start-counter` example, not the pinned fixture): it builds and runs end to end
(SSR loader via a GET server fn, a POST `.validator()` server fn mutating a file
through `node:fs`, hydration, `router.invalidate()` re-render), which is the first
evidence the pipeline generalizes beyond the fixture's exact routes.

That test surfaced a real bug the fixture had hidden: **server-only code leaked into
the client bundle**. `start-counter` defines its server functions inline in the
route module with a top-level `import * as fs from 'node:fs'` and private helpers
used only by the handlers. The client transform replaced each handler with an RPC
stub but left those now-dead top-level imports/helpers in place, so the browser
shipped the server logic and a `require("node:fs")` that only survived because the
client runtime tolerated it. The fixture never hit this because its server fns just
`fetch()` external URLs with no server-only imports.

- **Fix: source-level tree shaking in the client server-fn transform.** After
  stubbing handlers, `src/server_fn.rs` now sweeps every module-level declaration
  the handlers were the sole users of. Reachability is computed by the shared
  `src/js_reachability.rs` (`dead_declaration_spans`): live roots are the module's
  exports and side-effect statements; a binding referenced only from inside a
  removed handler argument is dead. Only side-effect-free declarations are swept
  (function/class declarations and imports, or `const`s with pure initializers), so
  a declaration whose initializer could have a side effect is conservatively kept,
  and a bare side-effect import (`import './x.css'`) is always retained.
- **One reachability implementation.** The generic module-level reachability
  helpers were extracted from `route_split.rs` into `js_reachability.rs` and are now
  shared by both the route splitter and the server-fn sweep, so the two cannot drift
  on what "reachable" means.
- **Verified.** start-counter's client bundle is clean (no `node:fs`, no server
  logic) and the app still works end to end; the fixture stays 13/13 with its client
  clean; full `cargo test` + clippy green. Regression is guarded at the unit level
  by `client_build_sweeps_server_only_code_the_handler_orphaned` (and siblings) in
  `src/server_fn.rs`, which encode the inline-server-fn shape directly.

## Native route-tree generation from `src/routes/` (landed)

## Native route-tree generation from `src/routes/` (landed)

Diffpack now **natively generates `routeTree.gen.ts` from `src/routes/`**, removing
the last TanStack-Vite-plugin-produced build-path input. Before this slice the
committed plugin-generated route tree was consumed as-is; the file-route convention
itself was never owned by Diffpack, and a route-file add/rename/remove in the dev
server was a hard-error crash.

- **A native file-route classifier + tree builder.** `src/route_tree.rs` walks
  `src/routes`, classifies every filename against the full file-route convention
  (index, flat dot-nesting, directory nesting, pathless `_layout`, dynamic `$param`,
  escaped `[.]`, trailing-`_` nesting opt-out), builds the parent/child graph
  (`full_id`/`path`/parent edges matching the reference exactly), and emits a
  runtime-complete `routeTree.gen.ts`: imports + the
  `Import.update({ id, path, getParentRoute })` chain + `_addFileChildren` assembly
  + the exported `routeTree`. The type-only `declare module` blocks are omitted
  because they are stripped at build.
- **Hard errors name the file.** An unclassifiable filename, a missing `__root`, and
  a missing pathless-layout parent are each a hard, file-naming error, never a
  silent skip or a wrong-valued placeholder.
- **Wired as a build-emit step, off the incremental hot path.** Route-tree
  generation runs in `build-app` (before discovery, for both the client and the ssr
  build) and at dev startup, so the thesis guards are unaffected and stay green.
- **Dev server: the route-tree crash becomes a real path.** A route-file
  add/rename/remove now natively regenerates the tree, fully rebuilds both envs, and
  full-page reloads (state-preserving HMR still deferred). The diffpack-owned
  `routeTree.gen.ts` is filtered as self-output so it cannot self-trigger the
  watcher.

Verified (all six gate groups reproduced independently, green):

1. `cargo test --release`: 95 lib tests incl. 6 new `route_tree` (including a
   semantic-equivalence test parsed from the committed reference — non-gameable —
   and the unclassifiable-filename hard error), 4 `oracle_incremental` (1
   `#[ignore]`d), 2 tailwind, 1 thesis_memory — all pass.
2. `cargo clippy --release --all-targets -- -D warnings`: clean.
3. `build-app` client + ssr, strict `npm run acceptance:diffpack`: **13/13**.
   Verified diffpack GENUINELY regenerates `routeTree.gen.ts` (output differs from
   the committed TanStack file in ordering + one var name) and the build consumes
   the regenerated file.
4. Thesis guards: `thesis_memory` PASS, `bundle_benchmark::thesis_guards` 3/3 PASS
   (leaf edit = 1 module/chunk), `oracle_incremental` byte parity PASS;
   `bundle-scale-memory` `transformed_per_edit_max`=1, `bytes_per_module`=3464.5,
   flat.
5. Browser (real headless Chrome): hydration 7/7, routes-check 60/60, dev-check
   10/10, `async_hooks` leak empty. Independently reproduced a LIVE route-add in
   dev: adding `smoke.tsx` took the server 14 -> 15 routes with
   `route tree changed... regenerated + full rebuild + reload pushed` and no crash.
6. Benchmark non-regression: diffpack incremental content-edit **7.75 ms** vs
   Rolldown **150 ms**; cold 168 ms vs 144 ms — no meaningful regression (this slice
   is off the incremental hot path).

No test/oracle files were modified; no vite/rolldown/node in the diffpack build
path (`route_tree.rs` is pure Rust; the only vite mentions are comments); no stubs
returning wrong values (four hard errors name the missing files); no server-only
leak.

**Next remaining gap.** State-preserving HMR for a route-file change: today an
add/rename/remove regenerates the tree and full-page reloads rather than swapping
the route in place. That, plus the other still-deferred dev increments (React Fast
Refresh, CSS hot-swap without reload, WebSocket-driven partial updates, an
in-browser error overlay), remain the open dev-experience surface.

## Dev server: long-lived live-rebuild + full-page browser reload (landed)

`diffpack dev <project-root> [port]` is a long-lived development server that keeps
a `Bundler` (plus its reachability session) alive PER ENVIRONMENT and re-emits on
file change. This is where the already-landed incremental emit (per-chunk render
cache, incremental `emit_public`/`emit_server`) is finally exercised across edits
from a single process — the payoff the incremental-emit note said "only lands once
the dev server + HMR slice keeps a `Bundler` alive across edits and re-emits on
file change." Scope is **full-page live reload only** (see deferred list below).

- **Long-lived, client-before-server builds.** On startup it runs the client build
  (`emit_public` + persist `client-manifest.json`) then the server build (register
  the `tanstack-start-manifest:v` and `#tanstack-start-server-fn-resolver` virtual
  modules, `emit_server`) exactly as `build-app` does, but retains BOTH bundlers.
  The mandatory client-before-server order is preserved (the server manifest needs
  the finished client chunk URLs).
- **Node runs the app; the reload channel is diffpack's.** The emitted
  `server/index.mjs` (the app's own SSR runtime, NOT the build path) boots as a
  child Node process on an internal loopback port. A diffpack-native reverse proxy
  (`src/dev_server.rs`, std-only HTTP: no new dependency) sits on the public dev
  port, forwards every request to the Node child, and injects a tiny SSE
  live-reload client into served HTML. The injected `<script>` self-removes
  synchronously (like TanStack Start's own inline bootstrap scripts) so it leaves
  no foreign DOM node for React to reconcile — hydration stays clean.
- **Incremental rebuild loop.** `notify` watches `src/`, coalescing atomic-save
  bursts. On a module edit: incrementally `rebuild_path` the client bundler ->
  incremental `emit_public` -> re-persist `client-manifest.json` -> incrementally
  `rebuild_path` the server bundler -> `emit_server` -> restart the Node SSR child
  -> push a full-page reload over the SSE channel.
- **Derived-virtual-module invalidation (correctness fix).** A route file's actual
  component/loader bodies live in its `?tsr-split=<target>` virtual chunks, which
  are separate graph nodes. `rebuild_path` now re-derives every virtual sibling of
  an edited physical file (same path, query-bearing id), so a route-component edit
  actually updates the split chunk on disk instead of leaving it stale. Without
  this the reference module (which no longer holds the body) reported "unchanged"
  and the served output never changed — a real incremental-correctness bug the dev
  oracle caught.
- **Render cache keyed on EMITTED output, not source (incrementality fix).** Each
  module now carries a `code_hash` (hash of its transformed output) distinct from
  its source `hash`; the per-chunk render cache and the "changed" signal key on
  `code_hash`. So a route-component edit — whose body is split into its own chunk,
  leaving the large entry (`client.js`) reference module byte-identical — reuses
  the entry chunk and re-renders ONLY the one split chunk. Before this fix the
  source-hash key needlessly re-rendered the 43k-line entry on every route edit.
- **Live incremental instrumentation.** Each edit prints a parseable line —
  `client transformed=N changed=M rendered_chunks=K | server ...` — where `changed`
  is the sharp incremental-transform signal and `rendered_chunks` the
  incremental-emit signal. For a leaf/route-component edit it is `changed=1`,
  `rendered_chunks=1`: the incremental-emit thesis guard exercised LIVE from a
  long-lived process for the first time.
- **Unsupported edit classes are hard errors, not silent/partial rebuilds.** A
  config change (`vite.config.ts`/`tsconfig*`/`package.json`), a route-tree change
  (`routeTree.gen.ts`), or a new/deleted file (a module in neither graph) is a
  clear hard error naming exactly what is unsupported. Deferred as documented
  follow-ons: React Fast Refresh / state-preserving HMR, CSS hot-swap without
  reload, route-tree regeneration on add/rename, WebSocket-driven partial updates,
  error overlays. This slice is full-page live reload only.

Verified (all six gate groups, green):

1. `cargo test --release`: 89 lib (incl. 5 new `dev_server::tests` — HTML reload
   injection, chunked decode, response parse) + 4 `oracle_incremental` (1
   `#[ignore]`d) + 2 tailwind + 1 thesis_memory — all pass.
2. `cargo clippy --release --all-targets -- -D warnings`: clean (exit 0).
3. Fresh `build-app . client` (27 chunks, `client-manifest.json` with 14 routes) +
   `. ssr` (37 server `.mjs`), strict `npm run acceptance:diffpack --strict`:
   **13/13** (dev server is additive; cold `build-app` output is byte-identical — a
   cold process starts with an empty render cache, so the `code_hash` key change
   never alters emitted bytes).
4. Thesis guards: `thesis_memory` PASS, all three `bundle_benchmark::thesis_guards`
   PASS, `oracle_incremental` byte parity PASS.
5. Browser (real headless Chrome via puppeteer-core + system Chrome): the new
   `dev-check.mjs` (`npm run browser:dev`) is **10/10** — it starts `diffpack dev`,
   asserts `Welcome Home!!!` SSR + hydration + injected reload client, EDITS
   `src/routes/index.tsx`'s greeting, awaits the automatic full-page reload (the
   in-page window probe is cleared, proving a real document reload not a partial
   swap), and asserts the NEW greeting is server-rendered AND still hydrates clean
   with zero JS errors and zero server leaks. In the same run the live dev-loop
   instrumentation from the long-lived process reads
   `client transformed=2 changed=1 rendered_chunks=1`: the edit re-transformed the
   edited module plus its derived `?tsr-split` sibling (transformed=2), changed
   exactly one module's emitted output (`changed=1`), and re-rendered exactly one
   client chunk (`rendered_chunks=1`) — the incremental-emit thesis exercised LIVE.
   On the clean-rebuilt output the existing browser gates are unchanged: hydration
   7/7, serverfn 7/7, tailwind 9/9, routes 60/60;
   `grep -rl async_hooks .diffpack-output/public` empty; `index.tsx` restored.
6. Benchmark non-regression: `bundle-scale-memory 2000 4 200` -> 3322.3
   bytes/module (baseline ~3313; the +~9 bytes is the new per-module `code_hash`
   field — well under the 16000 guard), `transformed_per_edit_max`=1, edit growth
   0.2 KB. `oracle/benchmark.mjs` diffpack incremental **8 ms** vs Rolldown
   incremental **150 ms** (~19x), diffpack cold 167 ms vs Rolldown cold 144 ms —
   no regression.

**Next remaining gap.** This slice is full-page live reload only. The next dev-
experience increments are React Fast Refresh / state-preserving HMR (swap a module
without a document reload), CSS hot-swap without reload, route-tree regeneration on
file add/rename/delete (today a hard error), WebSocket-driven partial updates, and
an in-browser error overlay. Each is a currently-hard-errored edit class becoming a
real incremental path.

## Production source maps, composed through the minify pass (landed)

`build-app` can now emit the pinned production client/SSR chunks MINIFIED **with**
source maps (`--sourcemap`), so a browser stack trace or DevTools breakpoint in a
minified chunk maps back to the original source file. The former hard error
(`minify + source_map` rejected in `emit_with_options`) is removed and replaced by
a real composition.

- **Two honest maps, composed.** The linker already produced module-granularity
  readable-generated -> original mappings (`ModuleMapping`). The minify pass
  (`minify_chunk_code_with_map`) now enables Oxc codegen's own source map on the
  re-print, giving a minified -> readable-generated map. `Bundler::compose_source_map`
  walks every minified token, resolves its readable-generated line (binary search
  over the sorted readable ranges) to the owning original module and source line,
  and emits a combined minified -> ORIGINAL map. A minified position that lands in
  a synthetic bundler region (runtime wrapper, export footer, browser prelude)
  with no owning module is left UNMAPPED — an honest gap, never a fabricated wrong
  origin. Per-token TS/JSX column fidelity is NOT claimed (the readable map is
  line-granular at column 0); the map stays honestly coarse rather than
  precise-but-wrong. A chunk whose minified map resolves into no original module at
  all is a hard error naming the chunk.
- **Project-relative, non-leaking sources.** `sources` are emitted as
  `diffpack:///<project-relative>` labels (common-ancestor root stripped; no
  absolute-path leak, no `..` traversal), with the real module text inlined as
  `sourcesContent`. Queries/fragments are preserved so distinct graph keys
  (`app.css` vs `app.css?url`) stay distinct sources.
- **Keyed into the render cache.** `source_map` is folded into `chunk_render_key`
  alongside `minify`, so a source-mapped chunk is a distinct cache entry and a leaf
  edit re-composes exactly the one changed chunk's map, reusing every other chunk's
  bytes AND its `.map` byte-for-byte (the emit thesis guard is preserved).
- **Per-chunk `.map` emit.** `emit_with_options` writes a sibling `.map` + a
  `//# sourceMappingURL` for EVERY chunk (entry + dynamic/route-split), records each
  in `EmitStats.written`, and `prune_stale_files` protects live maps.
- **Opt-in.** `build-app <root> client|ssr --sourcemap` sets `EmitOptions.source_map`
  for both the client `emit_public` and server `emit_server` paths. The default
  acceptance/benchmark invocation (no flag) is byte-identical to before.

On the pinned app: `build-app . client --sourcemap` emits 27 `.js` + 27 `.map`;
`build-app . ssr --sourcemap` emits 34 top-level `.mjs` + 33 `.map` (the four
hand-authored server-runtime templates — `index.mjs` and `_ssr/*` — are static
files, not rendered chunks, so they legitimately carry no composed map).

Verified (all six gate groups, on the source-mapped build):

1. `cargo test --release`: 84 lib + 4 `oracle_incremental` (1 `#[ignore]`d) + 2
   tailwind + 1 thesis_memory — all pass, incl. the new positive
   `a_minified_chunk_emits_a_composed_source_map_resolving_to_the_original_source`
   (replacing the old hard-error test) and
   `source_mapped_incremental_emit_reuses_every_unchanged_chunk_and_map`.
2. `cargo clippy --release --all-targets -- -D warnings`: clean.
3. `build-app . client` + `. ssr` (default, no maps), strict
   `npm run acceptance:diffpack`: **13/13**.
4. Thesis guards: `thesis_memory` PASS, both emit guards PASS, `oracle_incremental`
   byte parity PASS — now including a source-mapped leaf edit re-rendering exactly
   ONE chunk (and re-composing exactly its map) with the rest reused verbatim.
5. Browser (real headless Chrome, `sourcemap-check.mjs`): 5/5 — 27/27 chunk maps
   valid + project-relative + content-inlined, with a unique-token strong decode in
   all 27; the app hydrates clean with maps present; and a GENUINE runtime error
   thrown from a minified chunk has its stack frame decoded through the composed map
   to `router-core/dist/esm/router.js:268`. Plus routes 60/60, hydration 7/7,
   serverfn 7/7, tailwind 9/9; `grep -rl async_hooks .diffpack-output/public` empty
   for BOTH the default and the sourcemapped build (27 `.map` for 27 chunks).
6. Benchmark non-regression: `bundle-scale-memory 2000 4 200` → 3314.1 bytes/module
   (baseline ~3313), `transformed_per_edit_max`=1, edit growth 0.2 KB.
   `oracle/benchmark.mjs` diffpack incremental **7.5 ms** vs Rolldown **149 ms** —
   no meaningful regression (source maps are off the default path, which stays
   byte-identical to before).

**Next remaining gap.** Per-token TS/JSX column fidelity: the composed map is
line-granular at column 0 because the readable-generated map it composes through is
line-granular, so a minified position resolves to the right original *line* but not
the exact original *column*. Making the map column-precise needs the linker to
carry token-level (not module-line-level) `ModuleMapping` spans. The dev-server/HMR
slice that keeps a `Bundler` alive across edits is still the other open
production-hardening surface.

## Production minification of the emitted client/server chunks (landed)

Updated: 2026-07-18 (production minification of emitted chunks landed)

## Production minification of the emitted client/server chunks (landed)

`build-app` previously emitted the real 27-chunk TanStack production bundles
UNMINIFIED (`EmitOptions::default()`); the only `minify` code was a
constant-fold toy gated on `dynamic_roots.is_empty()` that never ran for the
real multi-chunk app. Production minification is now real and on by default.

- **A real per-chunk minify pass.** After a chunk's clean, valid JS `code` is
  produced (`render_chunk_cached`), when `EmitOptions.minify` is set the finished
  bytes are re-parsed with `oxc_parser` and re-emitted with
  `oxc_codegen::Codegen` under `CodegenOptions::minify()` (comments dropped,
  whitespace collapsed, literals shortened). This is a final, self-contained pass
  on already-clean JS, so it never touches the marker-based linker. A parse
  failure on the generated chunk is a HARD, chunk-naming error
  (`minify_chunk_code`), never a silent passthrough of the unminified bytes.
- **SCOPE (explicit):** whitespace/syntax minification only. Cross-module
  identifier mangling + dead-code compression (oxc_minifier over a combined AST)
  is DEFERRED to the P0 linker-IR evolution and is NOT attempted here — any case
  that would need it is left readable, never half-mangled.
- **Keyed into the render cache.** `minify` is folded into `chunk_render_key`, so
  a leaf edit re-minifies exactly the one changed chunk and reuses every other
  chunk byte-for-byte (the emit thesis guard is preserved — see the new
  `a_leaf_edit_reminifies_exactly_one_chunk_with_a_bounded_cache` guard).
- **Retired the dead constant-fold special-case.** `render_folded_constants` and
  its `dynamic_roots.is_empty()` gate are gone; minify now applies uniformly to
  the entry chunk AND every dynamic/route-split chunk, so it actually runs on the
  real app.
- **Wired into the acceptance path.** `build-app <root> client|ssr` minifies by
  default (`--no-minify` opts out for debugging). Client browser chunks and the
  server `.mjs` chunks are minified; the server `.mjs` still execute under Node
  ESM (13/13 acceptance boots the server and serves every route). `minify` +
  `source_map` together is now supported (`--sourcemap`): the map is composed
  through the minify pass — see "Production source maps, composed through the
  minify pass" at the top.

Measured reduction on the pinned app (total client `.js` bytes): **11,438,413 →
9,762,330 = 14.7% smaller** (entry `client.js` 43,468 lines collapse to 248).
The reduction is whitespace/syntax-only because ~20% of the client bytes are
already-minified vendored React production builds and this pass does no mangling
or DCE (the reference's larger reduction comes from exactly the mangling/DCE +
shared-vendor chunking deferred above). Server minifies analogously.

Verified (all six gate groups, on the MINIFIED build):

1. `cargo test --release`: 84 lib + 3 `oracle_incremental` (1 `#[ignore]`d) + 2
   tailwind + 1 thesis_memory — all pass, incl. the new
   `a_minified_chunk_runs_identically_to_its_readable_form_and_is_smaller`,
   `minify_and_source_maps_together_is_a_hard_error`, and
   `minified_incremental_emit_reuses_every_unchanged_chunk_and_matches_a_clean_build`.
2. `cargo clippy --release --all-targets -- -D warnings`: clean.
3. `build-app . client` + `build-app . ssr` (minified), strict
   `npm run acceptance:diffpack`: **13/13**.
4. Thesis guards: `thesis_memory` PASS, both emit guards
   (`a_leaf_edit_rerenders...` and the new `a_leaf_edit_reminifies...`) PASS, and
   `oracle_incremental` proves minified clean-vs-incremental byte parity (1 chunk
   re-minified, the rest reused verbatim).
5. Browser (real headless Chrome, minified output): routes **60/60**, hydration
   **7/7**, serverfn **7/7**, tailwind **9/9**;
   `grep -rl async_hooks .diffpack-output/public` empty.
6. Benchmark non-regression: `bundle-scale-memory 2000 4 200` → 3313.0
   bytes/module (baseline ~3312.4; `--minify` 3314.3 — no regression),
   `transformed_per_edit_max`=1, edit growth 0.2 KB. `oracle/benchmark.mjs`
   incremental edit **7.4 ms** (~20x faster than Rolldown's 148 ms), cold 161 ms
   — no regression.

Source maps composed through the minify pass have since landed (see the top
section); `minify` + `source_map` is no longer a hard error.

## Incremental-emit history

Updated: 2026-07-18 (incremental-emit wiring completed)

## Native Tailwind v4 CSS compilation (landed)

`src/styles/app.css` is raw Tailwind v4 source (`@import 'tailwindcss'` +
`@apply`). The pinned reference compiles it with `@tailwindcss/vite`; Diffpack
now does it natively, killing the `/assets/tailwindcss` 404 so the app renders
fully styled.

`src/tailwind.rs` is a general, pattern-based v4 utility engine, NOT a lookup
table of the app's classes. It is driven by faithful reference data embedded as
real files: the published default theme (`src/tailwind_theme.css`) and the
resolved v4 preflight (`src/tailwind_preflight.css`). The engine parses the CSS
AST, expands `@apply` (splitting `dark:` applies into a `prefers-color-scheme`
media rule), honors `@layer base`, tree-shakes theme tokens to those actually
referenced, generates one utility rule per scanned class with `dark:`/`hover:`
variant handling, and inlines the framework so no fetchable `tailwindcss` URL
survives. Any utility or variant the engine cannot produce is a hard error
naming the exact token, never a silent skip.

`tailwind::compile` is wired into `emit_assets` (a build-emit step, off the
incremental transform hot path, called only from `bundler.rs:851`) for BOTH the
client `public/` and server `server/` CSS paths, using class candidates scanned
from the app source root declared by `@import 'tailwindcss' source('../')`.
Per-class declarations match the reference app `app-CgRaPnL3.css` 39/39.

Verified (all 6 gate groups reproduced independently):

1. `cargo test --release`: 81 lib + 1 oracle_incremental + 2 tailwind_oracle +
   1 thesis_memory, all pass.
2. `cargo clippy --release --all-targets -D warnings`: clean (after forcing a
   re-lint of the changed files).
3. Native `build-app` client+ssr, then `npm run acceptance:diffpack --strict`:
   13/13.
4. Thesis guards: `thesis_memory` PASS, `bundle_benchmark` thesis_guards
   leaf-edit=1-module PASS, `oracle_incremental` byte parity PASS.
5. Browser: `tailwind-check.mjs` 9/9 (real Chrome + real server: computed body
   colors match theme tokens via probe in light AND dark, `font-black`=900,
   `gap`=24px, stylesheet 200 `text/css`, no `@import` survives, zero
   `/assets/tailwindcss` 404s); `hydration-check.mjs` 7/7;
   `serverfn-check.mjs` 7/7; `routes-check.mjs` 60/60 (every route direct-loads
   with the expected SSR content and hydrates clean — see "All-routes
   direct-load + hydration" below); `grep -rl async_hooks .diffpack-output/public`
   empty.
6. `bundle-scale-memory`: 3450.9 bytes/module (baseline 3449.7),
   `transformed_per_edit_max`=1, edit growth 0.2KB — no regression.

No existing test, oracle, or acceptance script was modified or deleted; the
embedded theme/preflight are real reference CSS files; unhandled utilities are
genuine hard errors; the build path is native Diffpack (Node/Chrome are test
oracles only).

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
diffpack:  13/13 TanStack production gates passed
```

**MILESTONE (2026-07-18): diffpack passes all 13 gates, matching the reference
Vite/Nitro build gate-for-gate — fully native, with NO Vite, Rolldown, or Node in
the build path.** `build-app <root> client` then `build-app <root> ssr` emits a
`.diffpack-output/` whose `server/index.mjs` boots under Node and serves the real
app: `GET /` returns SSR'd `Welcome Home!!!` (+ hydration `<script>` + stylesheet
link), `GET /customScript.js` runs the server route, `GET /<missing>` renders the
404. Verified by booting the server and fetching each route (React SSR through the
app's own handler, not hardcoded). The whole pipeline — resolution, tsconfig
paths, loaders (`?url`/`?raw`/assets/CSS), Node externals, route splitting
(`?tsr-split`), native manifest generation, executable ESM server output, and the
`node:http`↔fetch runtime entry — is native Rust on Oxc, under the incremental +
low-memory thesis guards (all green).

Diffpack now emits the client `public/` layout (route-split browser chunks +
extracted CSS + static assets) and the server `server/` layout (Node ESM `.mjs`
chunks) including the **natively generated `tanstack-start-manifest` chunk**;
see "Native client `public/` emit", "Native server (SSR) `.mjs` emit", and
"Native manifest generation" below. The passing gates are `output directory`,
`browser chunks` (28 chunks), `extracted CSS`, the two required static assets,
and `server artifact: tanstack-start-manifest`. The remaining seven gates need
the server graph to reach `>= 35` modules and the whole SSR runtime
(`server/index.mjs`, `ssr`/`router` runtime artifacts, HTTP serving), tracked
below.

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

On the pinned reference the client build produces **28 public `.js`** (the entry
chunk plus dynamic-import chunks; the count rose from 4 to 17 to 28 as `?tsr-split`
route splitting was implemented and then generalized to all targets), **0
extracted `.css`**, and **1 asset** (`app-<hash>.css`, the Tailwind source
imported as `~/styles/app.css?url`). Every emitted `.js` passes `node --check`
(used only as a test oracle, never in the build path). With the server emit added
this slice reached `5/13`; native manifest generation (below) then took it to
`6/13` (`output directory`, `browser chunks`, `extracted CSS`, the two required
static assets, and `server artifact: tanstack-start-manifest`).

Note: `EmitSummary` (formerly `PublicBuildSummary`) is now shared by the client
and server emits and counts `.js` and `.mjs` modules; its `output_dir` field was
`public_dir`.

Test: `bundler::tests::emit_public_writes_a_client_layout_with_chunks_css_and_assets`
builds a small app (CSS side effect + asset import + dynamic import), emits to a
temp `public/`, and asserts the chunk/CSS/asset files, the summary/disk match, and
that a re-emit clears stale output.

### The manifest gap, handled honestly

(Historical: this diagnostic is now resolved for the server build — see "Native
manifest generation". The client build still surfaces it, because
`client_route_manifest` cannot exist before the client build finishes, and the
client does not consume the manifest anyway; the note below records the original
leak analysis.)

The single remaining discovery diagnostic was
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

## Generalized route splitting (all TanStack targets, landed)

`?tsr-split` previously split only the `component` property; every other target
was a hard error. `src/route_split.rs` now splits every property TanStack Start
lazy-loads, each into its own chunk with the same wrapper shape:

- `component`, `errorComponent`, `notFoundComponent`, `pendingComponent` are
  wrapped with `lazyRouteComponent($$split<Target>Importer, '<target>')`;
- `loader` is wrapped with `lazyFn($$splitLoaderImporter, 'loader')`.

This matches TanStack Start's actual output (the reference `_ssr` chunks include
`loader` splits, e.g. `deferred` → 3 chunks, `posts._postId` → 4). Root routes
stay unsplit, matching the plugin's `unsplittableCreateRouteFns`
(`createRootRoute`/`createRootRouteWithContext`). The reference-file rewrite
removes each split-out definition and any import only it used, across all splits
at once, and emits a single combined `import { lazyFn, lazyRouteComponent }`.
`build_split_module` synthesizes each target's virtual module by extracting that
property's value (and its transitive module-level deps) and re-exporting it under
its canonical name; an unrecognized target is a hard error.

Effect: the pinned app's browser chunks rise from 17 to **28** and the server
chunks from 19 to **30**. Tests:
`splits_every_target_property_into_its_own_lazy_chunk` and
`an_unrecognized_split_target_is_a_hard_error`.

## Native server (SSR) `.mjs` emit (landed)

`diffpack build-app <root> ssr` (and `nitro`) now **emits a real server build**
to `<root>/.diffpack-output/server/` as Node ESM `.mjs` modules, mirroring the
client `public/` emit:

- `Bundler::emit_server(reachable, output_root, options)` rebuilds `server/` from
  scratch, emits the entry chunk `server/server.mjs` and its dynamic-import
  chunks (`server/server.chunk-N.mjs`), extracts CSS, and copies content-hashed
  assets under `server/assets/`. It shares one `emit_environment` helper with the
  client emit; the entry filename's extension (`.mjs`) flows onto every chunk.
- The renamed `EmitSummary` counts `.js` **and** `.mjs` modules from disk, so the
  printed summary always matches reality.
- `emit_server` is a build-time entry point off the incremental hot path, so the
  thesis guards stay green (verified: `a_leaf_edit_retransforms_exactly_one_module`
  and `the_incremental_graph_stays_low_memory` pass unchanged).

On the pinned reference the server build currently produces **30 server `.mjs`**
(the entry chunk + 29 dynamic chunks: 24 route-split chunks across the 5 targets +
5 framework chunks) and **1 asset**. **Every emitted `.mjs` passes `node --check`**
under Node's ESM goal (used only as a build oracle, never in the build path).
Test: `emit_server_writes_an_mjs_layout_that_node_accepts` builds a small app,
emits to a temp `server/`, asserts the `.mjs` entry + dynamic chunk and stale-file
clearing, and runs `node --check` on every emitted `.mjs`.

## Native manifest generation (landed)

`tanstack-start-manifest:v` — the virtual module TanStack Start's
`@tanstack/start-server-core/dist/esm/router-manifest.js` imports to map each
route to the client asset URLs it must preload/script-inject — is now
**generated natively from Diffpack's own client chunk graph**. It was the single
remaining discovery diagnostic; the SSR build now reports **0 diagnostics**.

The module is build-output-dependent (a route's preloads are the *client* build's
emitted chunk URLs), so it is a cross-environment coordination between the two
builds:

- **Client build persists a route → chunk map.** `Bundler::client_route_manifest`
  derives it from the same dynamic-import roots the emit assigns to chunks (the
  single source of truth is a shared `Bundler::dynamic_roots` helper, so the
  recorded chunk file names are exactly the files on disk). Each `?tsr-split=*`
  chunk is attributed to its route's TanStack id (the `createFileRoute` string
  argument, extracted by `route_split::route_id`); a route with several split
  properties lists all its chunks; `__root__` maps to the entry chunk `client.js`,
  which statically bundles the root route and all shared code. `build-app <root>
  client` writes it as `.diffpack-output/client-manifest.json`.
- **Server build reads that map and generates the module natively.**
  `manifest::ClientRouteManifest::to_start_manifest_source` emits the exact
  `const tsrStartManifest = () => ({ routes: { … } });` / `export { tsrStartManifest }`
  contract `getStartManifest` consumes (`preloads` per route, plus the entry
  `scripts` tag on `__root__`). `build-app <root> ssr` reads
  `client-manifest.json`, generates the source, and registers it in
  `BuildConfig.virtual_modules`. The bundler resolves the specifier to a virtual
  id and loads it from that source (new `synthesize_virtual_module` path, shared
  by the parallel and incremental loaders). It is dynamically imported, so it
  becomes its own chunk, emitted with a descriptive name
  (`server/_tanstack-start-manifest_v.mjs`) that satisfies the
  `server artifact: tanstack-start-manifest` gate.

Honesty guarantees, not gamed: a **missing `client-manifest.json` is a hard,
specific error** (`run \`diffpack build-app <root> client\` before the server
build`), never a silent empty manifest; a `?tsr-split` chunk whose route id
cannot be derived is a hard error, never a silently dropped preload. This imposes
a build order — **client before server** — which is the correct dependency
direction (the server manifest needs the finished client chunk URLs).

Difference from the reference, by design: Diffpack's preload URLs are its own
emitted client chunk URLs (`/client.js`, `/client.chunk-N.js`), not the
reference's shared-vendor `/assets/*` layout. The manifest reflects Diffpack's
actual chunking model (each route's split chunks are self-contained), which is the
correct, honest mapping for the files Diffpack emits — not a transliteration of
Vite/Rollup's chunk graph.

Manifest generation is a build-emit step off the per-edit incremental hot path
(`client_route_manifest` runs only from `build-app`, never from `rebuild_path`),
so the thesis guards stay green (verified:
`a_leaf_edit_retransforms_exactly_one_module`,
`the_incremental_graph_stays_low_memory`).

Tests: `manifest::tests` (JSON round-trip, missing-file error, the exact
`tsrStartManifest` source contract), `route_split::tests::
extracts_the_route_id_from_the_factory_argument`, and two bundler end-to-end
tests — `client_route_manifest_attributes_split_chunks_to_route_ids` (a route
app's split chunks map to their route ids) and
`a_registered_virtual_module_resolves_loads_and_names_its_chunk` (the specifier
resolves with no diagnostic, lands in the graph, and emits a `node --check`-valid
`_tanstack-start-manifest_v.mjs` chunk).

Known limitation carried forward: the emitted server chunks (the manifest chunk
included) still render the shared CJS-semantics runtime
(`module.exports`/`require`) as syntactically-valid ESM — it passes `node --check`
but does not *execute* under Node's ESM goal. Making the server chunks run is the
SSR-runtime slice below.

### The `>= 35` server-graph gate: 31 of 35, and what the last 4 need

The `server graph` acceptance gate wants `>= 35` `.mjs` under `server/`; Diffpack
emits 31 (the manifest chunk added one). This is deliberately reported as a
partial, not gamed. The remaining modules come from work outside this foundation
slice, each of which the reference build produces and Diffpack does not yet:

- **`tsr-shared` shared-chunk extraction.** The reference hoists sub-components
  referenced across split boundaries into their own shared chunks (e.g.
  `NotFound`, `PostError`). Diffpack currently inlines those into the split chunk
  that uses them. This is the next route-splitting increment.
- **The SSR runtime.** `server/index.mjs` (the Node HTTP entry) and the
  `ssr`/`router` runtime chunks are still gaps. These are the `server/index.mjs`,
  server artifact (`ssr`/`router`), and HTTP gates. The
  `tanstack-start-manifest` artifact is now produced (see "Native manifest
  generation" above).

The runtime the server chunks use is still the shared CJS-semantics runtime
(`module.exports`/`require`) rendered as syntactically-valid ESM: it passes
`node --check` but is not yet an executable ESM server. Making it *run* under Node
ESM (real `import`/`export` linkage, or a package-type marker + host `require`) is
part of the SSR-runtime slice, not this foundation.

### Precise next native step

1. **Native manifest generation** — DONE (see "Native manifest generation"
   above): `tanstack-start-manifest:v` is generated from Diffpack's own chunk
   graph, resolving the last discovery diagnostic and adding the
   `tanstack-start-manifest` server artifact.
2. **The SSR `server/index.mjs` runtime** — a native Node ESM server entry that
   renders documents, serves the hashed assets, runs server routes, and 404s,
   producing the `ssr`/`router` artifacts and flipping the three HTTP gates. The
   prerequisite the runtime must fix first: the server chunks (manifest included)
   currently render the shared **CJS-semantics runtime** (`module.exports` /
   `require`) as valid-syntax ESM that passes `node --check` but does **not
   execute** under Node's ESM goal. The SSR slice must make them real ESM (genuine
   `import`/`export` linkage, or a `package.json` `type` marker + host `require`)
   so `server/index.mjs` can actually import the manifest and route chunks at
   runtime.

`tsr-shared` extraction can land alongside these to carry the server graph past
`>= 35`.

## Browser hydration verified (2026-07-18)

Beyond the 13 acceptance gates (which only check server HTML/HTTP), the client
now genuinely hydrates in a real browser. A real bug the gates missed: the client
emitted CJS (`module.exports=...`) but the SSR loads it as
`<script type="module">`, so it threw `module is not defined` at load and never
hydrated. Fixed with a native `ModuleFormat::BrowserEsm` (no `node:module`/
`createRequire`; a throw-on-USE Proxy for any dead leaked externals so the module
still loads; a `globalThis.process.env.NODE_ENV ||= "production"` prelude for React).
Also fixed a latent browser-fatal bug: `?tsr-split` modules had dropped their
dependency edges, so their `require("react/jsx-runtime")` had no runtime map entry.

Verified in headless Chrome (puppeteer-core, test oracle only): `Welcome Home!!!`
present, `window.__TSR_ROUTER__` set (client executed + hydrated), no
`module is not defined`, zero uncaught page errors, client-side SPA navigation
works. 13/13 acceptance still holds; thesis guards + oracle green.

### Server-context / `async_hooks` client leak — FIXED

Server-only code no longer leaks into the client bundle. `grep -rl async_hooks
.diffpack-output/public` returns nothing (the reference client also has zero);
the server build still legitimately carries `async_hooks`.

**Root cause.** TanStack Start ships environment-neutral runtime stubs
(`createServerOnlyFn`, `createClientOnlyFn`, `createIsomorphicFn`,
`createMiddleware`) and expects the build tool to specialize them per
environment (its `@tanstack/start-plugin-core` `handleEnvOnly` /
`handleCreateIsomorphicFn` / `handleCreateMiddleware` compiler passes). Without
that specialization the *server* branches stay live on the client and keep their
references to `getStartContext` / `getRequestHeaders`, which pull
`@tanstack/start-storage-context` and `@tanstack/start-server-core` (and thus
`node:async_hooks`) into the client graph. `sideEffects:false` tree-shaking could
not drop them because the references were real. Two client leak paths existed:
(a) `start-client-core`'s `getGlobalStartContext` / `getRouterInstance` /
`getStartOptions` / `getStartContextServerOnly` (isomorphic / server-only
wrappers), and (b) the `/api/users` route's `createMiddleware().server(fn)` using
`getRequestHeaders`, reached statically from `routeTree.gen`.

**Fix.** A native, faithful client transform (`apply_env_transform` in
`src/transform.rs`, gated by `Target::Client`, threaded through `BuildConfig` /
`config.rs`): on the client it replaces `createServerOnlyFn(fn)` with a throwing
stub, collapses `createIsomorphicFn().client(a).server(b)` to `a`, and strips
`.server`/`.validator`/`.inputValidator` from `createMiddleware` chains. Deleting
those references makes the server imports genuinely unused, so diffpack's
existing `sideEffects:false` pruning drops them (and `node:async_hooks` with
them). Server builds keep the neutral runtime stubs (no transform), so the server
graph is byte-identical to before. Verified in headless Chrome: home hydrates,
SPA nav to `/posts` runs with **no server-context / `async_hooks` console
error**; 13/13 acceptance, thesis guards, and the incremental oracle stay green.

### Remaining gaps toward full production (documented, distinct from format)
1. **Tailwind compilation** — DONE (see "Native Tailwind v4 CSS compilation" at
   the top): `app.css` compiles natively, the `/assets/tailwindcss` 404 is gone,
   and the app renders fully styled (`tailwind-check.mjs` 9/9 in light and dark).
2. **Server-function RPC (`createServerFn`)** — DONE (commit `b97462617`,
   "native server functions (createServerFn RPC)"). `createServerFn(...).handler(fn)`
   is rewritten natively to the client/SSR RPC form (`createClientRpc`/
   `createSsrRpc` + a server-fn dispatch manifest) the reference emits, so calling
   a server fn resolves its real value on both the SSR and the client-navigation
   path. `serverfn-check.mjs` is 7/7: `/posts` and `/users` render their fetched
   data on direct load AND on client-side SPA navigation (the server-fn HTTP RPC
   path). The doc entry that called this "the next remaining gap" was stale.

## All-routes direct-load + hydration (verified 2026-07-18)

Every route of the pinned app now direct-loads with the expected server-rendered
content AND hydrates cleanly in real headless Chrome. The oracle is
`integration/tanstack-start-reference/routes-check.mjs` (puppeteer-core + system
Chrome, test-only, NEVER referenced by the build): it boots the emitted
`.diffpack-output/server/index.mjs` once and DIRECT-LOADS each route URL, asserting
the four invariants used by the existing checks for every one:

1. the expected SSR text is present in the INITIAL server HTML (a raw `fetch`
   before any client JS runs — proving the content is truly server-rendered);
2. `window.__TSR_ROUTER__` is set after load (the client bundle executed and
   hydrated);
3. zero uncaught page/JS errors;
4. no server-only leak in the console (`async_hooks` / `No Start context` /
   `module is not defined`).

Routes covered (60/60 gates): `/` (home), `/posts` and `/users` (layout-loader
lists, ≥10 items), `/posts/1` and `/posts/1/deep` (dynamic-param `fetchPost`
server-fn loaders + the `posts_` layout opt-out), `/users/1` (dynamic-param
loader through the nested `/api/users/$userId` server route), `/deferred` (React
streaming: the awaited `person` in the initial HTML, then both `Await`/`Suspense`
boundaries — `deferred-person` "Tanner Linsley", `deferred-stuff` "Hello
deferred!" — resolve via the stream), `/redirect` (a `beforeLoad`-thrown
`redirect({ to: '/posts' })` returns a 307 with `Location: /posts` and a direct
browser load lands on `/posts` with the list rendered), `/route-a` and `/route-b`
(the pathless nested layout chain `_pathlessLayout` "I'm a layout" >
`_nested-layout` "I'm a nested layout" > leaf "I'm A!"/"I'm B!"), and a
known-missing path (the app's `NotFound` 404 renders and still hydrates).

Triage found **no native bug**: every route already server-renders correct
content and hydrates through the native SSR runtime, route-split client chunks,
dynamic-param loaders, server-fn RPC, streamed Suspense, and the SSR redirect
Response path. The oracle reads rendered text via `textContent` (not `innerText`)
so CSS transforms like `uppercase` (the `NotFound` buttons) don't mutate the
asserted text.

Wired into the browser gate group via `package.json` scripts
(`browser:hydration`, `browser:serverfn`, `browser:tailwind`, `browser:routes`,
and `browser:all` running all four). Verified alongside the unchanged
`hydration-check.mjs` (7/7), `serverfn-check.mjs` (7/7), `tailwind-check.mjs`
(9/9), and `grep -rl async_hooks .diffpack-output/public` empty. All six gate
groups stay green with no regression; no Rust/build-path code was changed by this
slice (build output is byte-identical to the baseline — the fix surface was the
oracle coverage itself, which now proves the whole route surface).

## All routes hydrate cleanly (verified 2026-07-18)

With Tailwind and server functions landed, every route of the pinned app was
browser-verified (headless Chrome). Direct-load of `/`, `/posts`, `/posts/1`,
`/users`, `/users/1`, `/deferred`, `/route-a`, `/route-b`, `/posts/1/deep`: all
return HTTP 200, hydrate (`window.__TSR_ROUTER__` set), and produce ZERO
JavaScript console/page errors. The app is functionally complete on the pinned
target: builds, styles, SSRs, hydrates, navigates, and fetches data over server
functions, all natively.

Remaining production-readiness work is now dev-experience and hardening (not
app-breaking): production minification, source maps, a dev server + HMR, a
second representative app for breadth, secret scanning, and the perf/soak program.
These are driven by the `diffpack-production-hardening` workflow. Incremental emit
(previously listed here) is now DONE — see "Incremental chunk emit" below.

## Incremental chunk emit (landed)

The module graph was already incremental (a leaf edit re-transforms exactly one
module), but emit was not: `emit_environment` `fs::remove_dir_all`'d the whole
output tree and `emit_with_options` re-ran `render_best` on every chunk from
scratch on every build. Diffpack exists as a REPLACEMENT for Vite/Rolldown
because of its incremental graph, so this closed the one place that promise was
still unfulfilled — turning the already-incremental graph into already-incremental
OUTPUT.

- **Per-chunk render cache** (`Bundler::render_cache`, an interior-mutable
  `Mutex` so emit stays `&self`). Each chunk is keyed by `chunk_render_key`: its
  ordered dense-module ids, each member's transformed-content hash, and —
  restricted to the chunk's own members and the targets they reference —
  `format`, `is_main`, the root, `runtime_ids`, `chunk_names`, and aggregated
  export demands. The key deliberately excludes the graph-wide
  `runtime_ids`/export-demand vectors, so a leaf edit shifts neither for any
  chunk that excludes the leaf: those chunks keep their key and are reused
  byte-for-byte, while the one chunk containing the leaf is re-rendered. `emit_with_options`
  routes every chunk through `render_chunk_cached` (was a direct `render_best`):
  a hit returns the stored `RenderedBundle` verbatim, identical to a fresh
  `render_best`; a miss renders and populates. A chunk whose key cannot be
  formed hits a distinct `None` branch that renders rather than returning a
  silent wrong value — the absent case is hard-encoded, not a placeholder.
- **Incremental file sync, not a wipe.** `emit_environment` stopped
  `fs::remove_dir_all`-ing the output tree. `emit_with_options` now writes only
  the chunks/CSS/assets whose bytes changed (`write_if_changed`), records every
  kept file, and returns `EmitStats { rendered_chunks, written }`. The
  environment emit (`emit_public`/`emit_server`) deletes only files no longer in
  that kept set (`prune_stale_files`) — preserving the "no stale files linger"
  guarantee via atomic per-file writes instead of nuking unchanged chunks. Server
  runtime files are protected from the prune.
- **Bounded cache.** Each emit evicts every entry not in the live chunk set, so
  retained bytes stay flat across a long edit sequence (guarded by the memory
  thresholds in `docs/THESIS_GUARDS.md`).

Verified (all six gate groups reproduced independently, green):

1. `cargo test --release`: 82 lib + `oracle_incremental` (2 pass, 1 `#[ignore]`d)
   + 2 tailwind + 1 thesis_memory, all pass. The new
   `incremental_emit_reuses_every_unchanged_chunk_and_matches_a_clean_build` and
   `bundle_benchmark::thesis_guards::a_leaf_edit_rerenders_exactly_one_chunk_with_a_bounded_cache`
   both execute and pass (verified by name).
2. `cargo clippy --release --all-targets -D warnings`: exit 0, clean.
3. Rebuilt `build-app` client (27 chunks) + ssr (37 `.mjs`); strict
   `npm run acceptance:diffpack`: 13/13.
4. Thesis guards: `thesis_memory` PASS (`rendered_chunks_per_edit_max == 1`,
   `render_cache_entries <= 1` over 200 edits), the `bundle_benchmark` guard
   PASS, and `oracle_incremental` proves full multi-chunk output-tree byte parity
   AND the main chunk reused verbatim across a leaf edit. The pre-existing
   "re-emit must clear stale output" tests confirm `prune_stale_files` genuinely
   deletes stale files on a warm re-emit (no `remove_dir_all` regression).
5. Browser (real headless Chrome): routes 60/60, hydration 7/7, serverfn 7/7,
   tailwind 9/9; `grep -rl async_hooks .diffpack-output/public` empty.
6. Benchmark non-regression: `bundle-scale-memory 2000 4 200` →
   3312.4 bytes/module (< 16000 guard), `transformed_per_edit_max`=1, edit growth
   0.2 KB. `oracle/benchmark.mjs` incremental edit **7.78 ms** vs **158 ms** cold
   — ~19x faster than Rolldown on edits.

No Node/Rolldown/spawn entered the `src` build path; no output was faked; no test
was loosened or deleted (additions only, plus a benign stats-capture and import
widening). HEAD was a partial baseline that already carried the
struct/field/prune scaffolding; the working tree completes the wiring
(`render_best` → `render_chunk_cached`, dropping `remove_dir_all` for
`write_if_changed` + `prune_stale_files`).

The incremental emit is on the future dev-server/HMR hot path; the `build-app`
CLI is a cold process per invocation, so it still renders every chunk once (and
prunes any stale output), but the byte-identical result is unchanged.

**Next remaining gap.** RESOLVED. The payoff (a leaf edit re-rendering one chunk
in single-digit ms) now lands from a long-lived process: the `diffpack dev` server
keeps client and server `Bundler`s alive across edits and re-emits on file change,
verified LIVE (`client transformed=2 changed=1 rendered_chunks=1`). See "Dev
server: long-lived live-rebuild + full-page browser reload" at the top. Production
minification and source maps have also landed (see their sections above).
