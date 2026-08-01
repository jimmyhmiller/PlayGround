# Native Next.js + Diffpack status and next steps

Last updated: 2026-08-01

## Goal

Make Diffpack a real Next.js bundler selection, using Next's normal configuration
and server/runtime machinery. Completion means all of the following are true:

- `next build --diffpack`, `next start`, and `next dev --diffpack` work without an
  application-specific `next.config` workaround.
- App Router, App Routes, Pages Router, Pages APIs, and mixed-router applications
  use Diffpack-produced artifacts through Next's native project/endpoint contract.
- Development supports correct invalidation and genuine HMR where Next normally
  preserves state, with explicit full-reload behavior only where Next requires it.
- Cal.com builds and runs through the native Next integration and at least 60 of
  its selected E2E tests pass.
- Production and development performance are benchmarked against Turbopack on the
  same machine and checkout. Material regressions are fixed rather than waived.
- The binding remains a transport layer; Next semantics stay in `diffpack-next`,
  and framework-neutral compilation stays in lower crates.

This goal is **in progress**. Production fixture support is substantially working;
native development/HMR, Cal.com certification, and comparative benchmarks remain
release blockers.

## Architecture

```text
Next CLI / build / dev / start
              |
              v
integrations/next-bindings/binding.cjs
  raw Next project, endpoint, and subscription shapes
              |
              v
integrations/next-bindings/crate
  versioned process protocol; no Next artifact semantics
              |
              v
crates/diffpack-next
  route inventory, installed-Next templates, profiles, manifests,
  native .next artifacts, development integration policy
              |
              v
diffpack-web -> diffpack-default-loader -> diffpack-core
  framework-neutral graph, transforms, chunks, emission, and HMR mechanism
```

The dependency direction is intentional:

- Next continues to parse ordinary `next.config.*`, perform type checking and
  static generation, write global build metadata, and serve the result.
- The JavaScript binding translates Next's unstable raw binding ABI into a small,
  versioned Diffpack protocol. It must not discover routes or synthesize manifests.
- `diffpack-next` owns Next route kinds, official template expansion, runtime
  aliases, client/server profiles, and `.next` layout.
- Lower crates do not know about Next route types or manifest formats.

## Verified current status

### Selection and configuration

- The local Next checkout exposes `--diffpack` for build and development.
- `DIFFPACK_BINDINGS` selects the integration while it is under development.
- Applications do not need to modify `next.config.*` to make the tested fixture
  build. The standard Next config is evaluated by Next, and relevant output config
  is transported to the native adapter.
- Diffpack delegates compiler transforms to the installed Next native SWC binding;
  it replaces project and endpoint compilation, not unrelated SWC functionality.

### Typed route and artifact boundary

`NextRouteArtifactKind` distinguishes:

- `AppPage`
- `ImplicitAppPage`
- `AppRoute`
- `PagesPage`
- `PagesApi`

Pages-only and hybrid projects are supported by the same public inventory. The
standalone Pages adapter exposes typed route artifacts without leaking its private
matcher representation into the binding or native artifact writer.

### Native production build

Verified on a mixed fixture using stock `next build --diffpack`:

- App page `/`
- implicit App Router not-found handling
- App Routes `/api/health` and `/api/ping`
- Pages page `/about`
- Pages API `/api/legacy`

The build completes type checking, page-data collection, static generation, and
final route optimization. The prior missing `/_error` client-file warnings are
gone.

Server entries use the installed Next version's official templates:

- App Page entry runtime and a real loader tree
- App Route template
- Pages template, including `_app`, `_document`, and `_error`
- Pages API template

Next's production server has been verified to return:

- HTTP 200 with real App Router/Flight HTML for `/`
- the expected JSON from App and Pages API routes
- HTTP 200 with Pages Router HTML for `/about`
- a correctly rendered HTTP 404 document

The Pages browser graph is emitted separately from the App Router Flight graph.
Its manifest entry is attached to `/_app`, `/_error`, and each Pages route. Because
Next's stock `_document` loads Pages files as classic deferred scripts, the adapter
converts Diffpack's eagerly executing browser-ESM entry into a classic-script goal
using a narrow, validated transformation. It hard-errors if ESM-only syntax remains.
The resulting artifact passes `node --check` and explicit checks for top-level
`import`, `export`, and `import.meta`.

### Tests passed

- All 297 `diffpack-next` library tests pass.
- The focused Pages adapter suite passes.
- Typed Pages-only and mixed route-inventory tests pass.
- The binding crate builds, and its raw App/Pages endpoint shape tests exist.
- The real mixed production fixture builds without warnings.

These results do not certify native development/HMR or Cal.com yet.

## Development and HMR: current truth

There is development scaffolding, but full support is **not verified and should not
be described as complete**.

The current bridge can:

- start a long-lived watcher;
- run an initial native artifact build;
- watch relevant source/config extensions while excluding outputs and
  `node_modules`;
- rebuild after a filesystem event;
- republish entrypoints and endpoint-change notifications;
- send a server-HMR event whose current meaning is `restart`.

Important limitations in the current implementation:

- A change invokes the production build path again rather than maintaining
  incremental client, SSR, RSC, Pages, and route-entry graphs.
- `projectHmrEvents` currently returns only an empty issues event.
- `projectHmrChunkNamesSubscribe` returns no chunk names.
- endpoint responses report no client paths.
- the bridge has no real browser update payload or module accept/dispose graph.
- The README still describes the old production-only milestone and must be updated
  after the development contract is real.
- A sandboxed attempt to launch `next dev --diffpack` could not bind a port. A later
  escalated launch was interrupted before yielding evidence, so cold-start dev
  rendering has not yet been certified.

In short: the watcher may be capable of triggering a server restart, but this is
not genuine HMR and is not sufficient for the goal.

## Ordered next steps

### 1. Establish the native development baseline

Run the mixed fixture with `next dev --diffpack` on an approved localhost port and
capture:

- cold-start time to listening;
- first-request status and rendered content for App and Pages routes;
- App Route and Pages API responses;
- browser assets requested by each router;
- server and browser console diagnostics;
- shutdown behavior and watcher-process cleanup.

Do not proceed on the assumption that production `.next` artifacts are valid dev
artifacts. Record every raw binding call Next makes in development and compare its
expected return shape to Turbopack's generated binding declarations.

Acceptance: all mixed-fixture routes work through `next dev --diffpack`, with no
uncaught server/browser errors and no leaked child process after shutdown.

### 2. Specify the development protocol explicitly

Replace the loose newline response stream with typed events such as:

```text
initial-build { generation, routes, artifacts, issues }
entrypoints-changed { generation, added, changed, removed, issues }
endpoint-changed { generation, endpoint-id, server-paths, client-paths, issues }
hmr-update { generation, target, chunks, update-payload, issues }
server-restart { generation, reason, issues }
```

Every event needs a monotonic generation so stale asynchronous results cannot win.
Endpoint IDs must be stable across rebuilds. Errors must be recoverable development
issues where Next expects recovery, not process termination.

Acceptance: protocol serialization tests cover initial success, recoverable compile
error, recovery, route add/remove, simultaneous edits, and shutdown.

### 3. Keep incremental graphs alive

Move development compilation out of `production_request`. Maintain independent
incremental graph state for:

- App Router client
- App Router SSR consumer
- React Server Components
- App Page and App Route native entries
- Pages client
- Pages server pages and APIs

Use Diffpack's existing watcher/rebuild and HMR mechanisms rather than rebuilding
the whole application for each notify event. Coalesce bursts without discarding a
change that arrives during compilation. Keep generated artifacts out of the watched
input set.

Acceptance: editing one leaf page recompiles only affected modules/entries; route
addition and deletion update the entrypoint inventory without restarting the bridge.

### 4. Implement Next's endpoint subscriptions faithfully

Complete and test:

- `endpointServerChangedSubscribe`
- `endpointClientChangedSubscribe`
- `projectHmrChunkNamesSubscribe`
- `projectHmrEvents`
- `projectAllHmrEvents`
- compilation issue and update-info subscriptions

`endpointWriteToDisk` must return the exact server and client paths used by that
endpoint, including shared registries and route-specific client chunks. It must not
guess solely from a route-kind prefix.

Acceptance: contract tests exercise the binding through Next's real dev server, not
only direct calls to `binding.__diffpack` helpers.

### 5. Deliver real browser HMR

Wire Diffpack's module update payloads to the protocol Next consumes. Preserve
component state for an accepted React edit and force a full reload for changes that
invalidate server/runtime boundaries. Cover at least:

- client component text/style edit with state preservation;
- client component syntax error and recovery;
- server component edit with refreshed Flight data;
- App Route and Pages API edit;
- Pages component and `_app` edit;
- CSS module/global CSS edit;
- route add, rename, and removal;
- environment/config edit requiring restart;
- server action edit;
- simultaneous edits while a build is active.

Acceptance: an automated browser test observes DOM updates and retained state where
appropriate, and asserts when a reload or server restart is expected. Merely seeing
new HTML after manually refreshing does not pass.

### 6. Cover standard Next options

Build a compatibility matrix from Next's actual normalized config, prioritizing
options used by Cal.com and options that alter compilation or output:

- `basePath`, `assetPrefix`, `trailingSlash`, redirects, rewrites, and headers;
- `pageExtensions`, `src/app`, `src/pages`, and mixed routers;
- standard environment precedence and `NEXT_PUBLIC_*` exposure;
- `images`, `next/font`, MDX, CSS/Sass/PostCSS/Tailwind, and static assets;
- `output`, source maps, server externals, aliases, and package transpilation;
- dynamic routes, catch-all routes, middleware/proxy, Edge declarations, ISR,
  server actions, and cache behavior.

Unsupported options must fail with a precise diagnostic. They must never be silently
ignored.

### 7. Certify Cal.com

Use the prepared checkout at `/tmp/dpe2e/calcom` (pinned revision documented in
`STATUS_2026-07-28.md`). Do not clone or reinstall it.

Cal.com currently installs Next 16.2.3 while the local development checkout is a
16.3 canary. Before certification, make the integration selectable from Cal.com's
ordinary `next` invocation without application config changes and explicitly verify
the raw binding ABI against 16.2.3. Do not silently replace application dependencies
without recording the exact test setup.

Run production and development modes against the restored pristine database using
the safeguards in `scripts/calcom-e2e.sh`. The native integration needs its own
harness mode; the existing script currently exercises Diffpack's standalone Next
server, not `next build/dev --diffpack`.

Acceptance:

- native production build and `next start` pass;
- native `next dev --diffpack` passes the same selected routes;
- at least 60 selected Cal.com Playwright tests pass;
- no Diffpack-attributable failure is hidden as skipped/flaky;
- failures reproduced on the Turbopack reference are recorded separately.

### 8. Benchmark and fix regressions

Extend `scripts/bench-calcom.mjs` with a native-Next Diffpack mode. Use the same
checkout, environment, database state, routes, edit, browser, and sampling method.
Measure at minimum:

- cold production build wall time and peak process-tree RSS;
- production output size and per-route transferred/decoded JavaScript;
- dev time to listening and first usable document;
- warm page navigation;
- client edit to updated DOM;
- server component edit to updated DOM;
- CSS edit to visible update;
- incremental CPU/RSS behavior over repeated edits.

Report distributions or best-of-N only when the existing benchmark protocol calls
for them. If Diffpack is materially slower, profile the responsible phase and fix it
before completion; do not redefine the benchmark.

### 9. Finish documentation and cleanup

When behavior is verified:

- update `integrations/next-bindings/README.md` with build and dev usage;
- document the protocol and supported Next versions/options;
- add a native integration test harness and CI/release-gate commands;
- remove stale production-only wording and temporary fixture assumptions;
- keep generated adapter directories and `.next` artifacts out of source control;
- run crate-boundary, public-API, extraction, workspace, binding, fixture, Cal.com,
  and benchmark gates.

## Completion checklist

- [x] Explicit Next bundler selection
- [x] Standard-config mixed production fixture
- [x] Native App Page and implicit App entries
- [x] Native App Routes
- [x] Native Pages APIs
- [x] Native Pages server pages
- [x] Pages hydration artifact and manifest integration
- [x] Clean mixed `next build --diffpack`
- [x] Mixed `next start` HTTP verification
- [ ] Native `next dev --diffpack` cold-start verification
- [ ] Incremental development graphs
- [ ] Client HMR with state preservation
- [ ] Server/RSC update propagation
- [ ] Pages HMR
- [ ] Route add/remove and error recovery
- [ ] Standard Next option compatibility matrix
- [ ] Native Cal.com production build/start
- [ ] Native Cal.com development run
- [ ] At least 60 native Cal.com E2E passes
- [ ] Native Diffpack versus Turbopack benchmarks
- [ ] Material performance regressions fixed
- [ ] Final integration/protocol/user documentation

The project should not be declared complete until every unchecked release blocker
above has evidence attached to it.
