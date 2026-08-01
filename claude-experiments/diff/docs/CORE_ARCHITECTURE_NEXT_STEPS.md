# Core architecture next steps

The crate extraction is structurally successful: the framework-independent
graph/compiler/linker/chunk/rendering machinery lives in `diffpack-core`, the
root package is a CLI/composition layer, integration crates no longer depend on
the root package, and the dependency rules are executable. Cal.com's production
build and 60-test Playwright gate demonstrate that the extracted layers compose
at real-application scale.

This document originally covered the work required to turn that successful
extraction into a deliberate, stable extension architecture. That work is now
complete. The priorities below remain as the design record and acceptance
contract for future changes.

## Completion record

Completed on 2026-08-01:

- Web is framework-neutral; Vite owns config evaluation, environment exposure,
  public-directory behavior, and manifest compatibility through a Web adapter.
- Next owns its environment/source policy and has no Vite dependency. TanStack's
  intentional Vite dependency is confined to its integration layer.
- Runtime contributions are named and owned, exclusive capabilities conflict
  explicitly, required capabilities validate before rendering, and every final
  browser/server profile has a composition snapshot.
- `diffpack-default-loader::BuildEngine` is the supported external facade. The
  workspace external-provider example exercises resolution, virtual loading,
  transforms, emitted assets, watches, incremental rebuilds, externals, warnings,
  and fatal diagnostics.
- The supported API and stability policy are documented in
  [PUBLIC_API.md](PUBLIC_API.md) and [EXTENDING_DIFFPACK.md](EXTENDING_DIFFPACK.md).
  `cargo public-api` is recorded in `public-api.snapshot` and checked by the
  extraction gates.
- Next and TanStack profile preparation/emission, Vite manifest emission, and
  Next production layout/prerender assembly live in their integration crates.
  The root selects profiles and orchestrates their commands.

Validation at completion:

- `./scripts/check-extraction.sh phase` passes all 775 extracted-crate unit tests.
- `./scripts/check-extraction.sh final` passes the full workspace, corpus, rustdoc,
  dependency-boundary, and public-API gates.
- The pinned Cal.com reference and Diffpack production builds both pass the heavy
  build-only gate after running Cal.com's declared static-input preparation task.
- The long Cal.com Playwright run remains a separate release gate. Its first
  attempt was invalidated by the corpus build's deliberately dead database URL;
  a correctly configured rerun was stopped by request rather than blocking this
  architecture cleanup.

## Desired end state

```text
                         diffpack (CLI)
                                |
          +---------------------+---------------------+
          |                     |                     |
     diffpack-next       diffpack-tanstack    diffpack-vite-compat
          |                     |                     |
          +---------------------+---------------------+
                                |
                         diffpack-web
                                |
                   diffpack-default-loader
                                |
                         diffpack-core
```

This diagram describes semantic ownership, not a requirement that every crate
depend on every layer immediately below it. Integrations may depend directly on
lower layers. The important constraints are:

- `diffpack-core` has no filesystem, package-manager, browser, dev-server, or
  framework semantics.
- `diffpack-default-loader` has no Vite, Next, TanStack, HTML, or dev-server
  semantics.
- `diffpack-web` has no Vite configuration semantics.
- Next and TanStack do not reuse behavior merely because it currently has a
  Vite-shaped implementation.
- The root package selects and composes profiles; it does not implement them.

## Priority 1: remove Vite semantics from `diffpack-web` — complete

### Problem

`diffpack-web` currently depends on `diffpack-vite-compat`. Its configuration
layer reads Vite config, Vite environment files, `import.meta.env`,
`import.meta.glob`, and Vite proxy rules. That makes the nominally neutral web
layer a Vite web profile rather than a reusable browser layer.

### Target ownership

Keep in `diffpack-web`:

- HTML discovery and rewriting.
- Browser runtime and output behavior.
- Static file serving, preview, proxy transport, HMR, WebSocket, and watcher
  primitives.
- Neutral proxy configuration types such as route prefix, target URL, and
  rewrite behavior.
- A neutral `WebBuildConfig` containing already-resolved browser settings.

Keep or move into `diffpack-vite-compat`:

- Vite config evaluation.
- Translation from Vite's `server.proxy` shape into neutral web proxy rules.
- Vite `.env` loading and exposure rules.
- `import.meta.env`, `import.meta.glob`, `define`, Vite aliases, and manifest
  conventions.
- Construction of the final Vite web build profile.

### Implementation slices

1. Introduce neutral proxy and web configuration records in `diffpack-web`.
2. Change web servers and compilers to consume those records without reading a
   Vite config themselves.
3. Add a Vite-to-web configuration adapter in `diffpack-vite-compat`.
4. Move Vite-specific config tests out of `diffpack-web`.
5. Remove the `diffpack-vite-compat` dependency from
   `crates/diffpack-web/Cargo.toml`.

### Acceptance criteria

- `cargo tree -p diffpack-web` contains no `diffpack-vite-compat`.
- A non-Vite HTML/SPA fixture can construct and run a web build without a Vite
  config file.
- Existing Vite build, preview, proxy, HMR, environment, and manifest tests pass.
- The dependency-boundary script rejects a future Web-to-Vite edge.

## Priority 2: remove Vite-shaped utilities from Next and TanStack — complete

### Problem

Next currently reuses `ViteSourcePolicy` and Vite environment parsing. TanStack
legitimately consumes Vite compatibility as part of TanStack Start's toolchain,
but it also reaches directly into Vite config helpers. Some of the reused code
is generic mechanism with a Vite name; some is genuinely Vite policy. The two
must be separated before the public boundaries stabilize.

### Classify before moving

Move generic mechanism downward only when its semantics are framework-neutral:

- `.env` syntax parsing can live in `diffpack-default-loader`.
- Generic AST replacement driven by an explicit define table can live in the
  default loader or core compiler support.
- Generic source inclusion/exclusion and transform pipelines can live in the
  default loader.
- Vite mode precedence, `VITE_` exposure, config evaluation, glob semantics,
  and manifest formats remain in `diffpack-vite-compat`.
- Next environment precedence, public-variable exposure, and defines remain in
  `diffpack-next` even if they use neutral parsing machinery.

### Implementation slices

1. Extract a neutral environment-file parser without Vite naming or precedence.
2. Give Next a `NextSourcePolicy` assembled from Next-owned semantics and
   neutral transforms.
3. Replace direct `ViteSourcePolicy` use in the App and Pages Router profiles.
4. Put TanStack's Vite interpretation behind one TanStack-owned configuration
   adapter instead of scattering Vite helper calls through route discovery.
5. Remove `diffpack-vite-compat` from `diffpack-next/Cargo.toml`.

### Acceptance criteria

- `cargo tree -p diffpack-next` contains no `diffpack-vite-compat`.
- Next `.env`, `next.config` environment values, browser exposure, and server
  secrecy have dedicated tests under `diffpack-next`.
- TanStack's dependency on Vite compatibility is localized to its profile/config
  adapter and is covered by integration tests.
- No API with `Vite` in its name is used to implement Next semantics.

## Priority 3: make policy composition explicit and safe — complete

### Problem

`RuntimePolicyChain` currently combines vectors and replaces optional fields
when later policies provide a value. This is convenient, but it makes accidental
omission and accidental override difficult to distinguish. The missing browser
`process.env` prelude found by Cal.com was an example: every crate compiled and
the unit suites passed, but the composed Next browser profile was incomplete.

The special-module and output policy chains should be audited under the same
principle. Ordering is part of behavior and must be named and tested.

### Target API

Prefer contributions over mutable, last-writer-wins records. A runtime policy
should contribute named capabilities such as:

- Browser process compatibility.
- Native-require behavior.
- Framework compatibility prelude.
- HMR protocol/runtime.
- Entry initialization.

Composition should return either a complete validated runtime plan or a typed
conflict/missing-capability error. Multiple append-only preludes are valid;
multiple exclusive implementations of the same capability are not valid unless
the profile explicitly selects one.

### Implementation slices

1. Give every runtime contribution a stable name and ownership label.
2. Replace silent optional-field replacement with conflict detection.
3. Add a validation step after composition and before graph discovery/emission.
4. Document ordering semantics for runtime, special-module, source, and output
   policy chains.
5. Add final-profile snapshot tests for plain Web, Vite, Next client, Next
   server, TanStack client, and TanStack server environments.

### Required regression tests

- Every browser profile that requests process compatibility emits it in both
  registry and flat output.
- Browser process initialization appears before framework code that reads it.
- A duplicate exclusive runtime capability fails with a named diagnostic.
- HMR and production profiles cannot accidentally share development-only
  preludes.
- Reordering two policies changes a snapshot or fails validation; it never
  changes behavior silently.

### Acceptance criteria

- The Cal.com `process is not defined` regression is reproducible by a small
  composition test that now passes.
- Invalid policy combinations fail before rendering any chunks.
- A profile test can explain the origin and order of every emitted prelude.

## Priority 4: provide one ergonomic external-tool facade — complete

### Problem

`ModuleProvider`, `ProviderPipeline`, `ModuleCompiler`, and the policy contracts
are sufficient building blocks, but an external tool must currently understand
too much of the default driver's internal assembly. This is a capable internal
extension system, not yet a small and intentional SDK.

### Target API

Add a builder owned by the lowest crate that can construct it without upward
dependencies. The exact name is open, but the shape should resemble:

```rust,ignore
let engine = BuildEngine::builder(project_root)
    .environment(environment)
    .compiler(compiler)
    .provider(my_provider)
    .source_policy(source_policy)
    .runtime_policy(runtime_policy)
    .build()?;

let result = engine.discover(entry)?;
```

The facade should make common extension work easy while retaining lower-level
APIs for built-in integrations.

### External example crate

Add a workspace example or test crate that depends on public crates exactly as
an outside tool would. It must demonstrate:

1. Resolving a custom scheme or virtual specifier.
2. Loading a virtual JavaScript or TypeScript module.
3. Transforming an ordinary source module.
4. Emitting an asset from a transform.
5. Adding an extra watch file.
6. Invalidating that file and observing the correct incremental graph delta.
7. Returning a warning and a fatal diagnostic.
8. Declaring an external module without adding it to graph discovery.

The example must not import private root modules or reach into driver fields.

### Acceptance criteria

- A new provider can be integrated without modifying Diffpack source.
- Cold discovery and incremental rebuild use identical provider ordering.
- Built-in virtual/query loaders have documented precedence relative to external
  providers.
- Source-map support is either implemented end to end or rejected explicitly;
  maps are never silently discarded.
- The example is compiled and executed in CI.

## Priority 5: define and reduce the public surface — complete

### Problem

The extracted crates currently expose many implementation modules directly.
That was useful during migration, but treating all of them as stable would make
future refactoring unnecessarily expensive. There are also similarly named
types at different abstraction levels, including multiple source-language and
dense-module-id representations.

### Work

1. Inventory every `pub` item consumed outside its defining crate.
2. Introduce small crate-root facades for the intended stable contracts.
3. Change implementation modules to `pub(crate)` where practical.
4. Consolidate or clearly distinguish duplicated concepts such as source
   language, module identity, dense IDs, and transform outputs.
5. Add rustdoc examples for provider, compiler, profile, and incremental APIs.
6. Decide whether semver stability is intended now or only after another
   iteration; state that explicitly in crate documentation.

### Acceptance criteria

- Public API documentation shows the supported path for an external tool
  without exposing the internal driver layout.
- `cargo public-api` or an equivalent recorded API snapshot can detect accidental
  expansion and breaking changes.
- Framework crates consume stable lower-layer facades rather than arbitrary
  internal modules.

## Priority 6: finish root composition cleanup — complete

The root package is now correctly a CLI/composition layer, but `src/main.rs`
remains large. Line count alone is not a defect; framework behavior in that file
would be. Audit it by responsibility rather than splitting it mechanically.

Move remaining profile-owned behavior to the relevant crate when it includes:

- Framework manifest construction.
- Framework virtual-module registration.
- Framework-specific output layout.
- Framework route or runtime decisions.

Keep in the root:

- CLI parsing and command dispatch.
- Selection of Web/Vite/Next/TanStack profiles.
- User-facing progress and diagnostics.
- High-level orchestration across explicitly selected profiles.

Add tests that call each integration's public build entry point directly, so
the CLI is not the only proven composition route.

## Continuous validation

Run after every slice:

```sh
./scripts/check-extraction.sh slice <changed-crate>
```

Run after each priority is completed:

```sh
./scripts/check-extraction.sh phase
cargo tree -p diffpack-core
cargo tree -p diffpack-default-loader
cargo tree -p diffpack-web
cargo tree -p diffpack-vite-compat
cargo tree -p diffpack-next
cargo tree -p diffpack-tanstack
git diff --check
```

Run before declaring the architecture stable:

```sh
./scripts/check-extraction.sh final
node integration/e2e/run.mjs --heavy next-calcom --build-only --jobs 1
scripts/calcom-e2e.sh prod bench/results/calcom-e2e-core-architecture
```

The minimum real-application gate is the current Cal.com result: at least 60
passing Playwright tests. The remaining same-slot availability timeout should
be tracked separately and must not be hidden by reducing the selected suite.

## Recommended sequence

1. Remove Web-to-Vite dependency.
2. Remove Next-to-Vite dependency and localize TanStack's Vite adapter.
3. Harden policy composition and add composed-profile tests.
4. Add the external `BuildEngine` facade and example provider.
5. Reduce and document the public API.
6. Audit remaining root composition and rerun the complete real-app gates.

Do not combine these into one large migration. Each priority should land with a
clear dependency diff, focused tests, and no compatibility regression.

## Architecture definition of done

All architecture criteria below are satisfied. The final real-application
browser suite is intentionally tracked as a release-validation gate, as noted in
the completion record, rather than unfinished ownership work.

- Core and default-loader remain free of browser and framework semantics.
- Web no longer depends on Vite compatibility.
- Next no longer depends on Vite compatibility.
- TanStack's Vite dependency is isolated behind a single adapter.
- Policy composition detects conflicts and missing required capabilities.
- External providers have one documented, end-to-end tested entry point.
- Public APIs are intentionally selected rather than migration-era exposure.
- Root remains CLI/composition only.
- Workspace, corpus, and Cal.com gates meet or exceed the pre-cleanup baseline.

## Native Next.js bundler integration

The native integration follows one dependency direction:

```text
Next CLI/build/dev orchestration
  -> @diffpack/next-bindings (versioned transport and raw binding shapes)
  -> diffpack-next (Next routes, entry templates, manifests, and artifacts)
  -> diffpack-web/default-loader/core (framework-neutral compilation)
```

Next continues to load ordinary `next.config.*`, run type checking, collect
route data, perform static generation, write its global manifests, and serve
with `next start`. `--diffpack` selects Diffpack explicitly; applications do not
need a custom Next configuration.

The per-route server artifact must be a Diffpack-compiled instance of Next's
official App Page or App Route entry template. A module that only copies a
pre-rendered document is not an acceptable production substitute: Next's build
workers call the entry renderer ABI (`patchFetch`, metadata construction,
segment collection, prerender helpers, and the route handler), and development
adds subscriptions and HMR on the same endpoint identity. Manifest and client
asset translation remains in `diffpack-next`; the JavaScript binding must not
accumulate filesystem or route semantics.

Production integration progress on 2026-08-01:

- Diffpack compiles Next's official App Page entry ABI, including implicit
  not-found and global-error entries, and emits the native `.next/server/app`
  artifacts and client-reference manifests consumed by Next build workers.
- Diffpack mechanically expands the installed Next version's official App
  Route template using the same route fields as `next-app-loader`. The binding
  transports standard `nextConfig.output` instead of requiring an
  application-specific configuration escape hatch.
- A zero-custom-config App Router fixture completes `next build --diffpack`,
  static generation, and `next start`; its page and a real `GET /api/ping`
  route both return 200 through Next's production server.
- Mixed Pages Router artifacts, Cal.com validation, native development
  subscriptions/HMR, and comparative benchmarks remain release blockers.

Completion requires native `next build --diffpack`, `next start`, and
`next dev --diffpack`, standard Next option coverage, mixed App/Pages Router
support, at least 60 passing Cal.com E2E tests, and production/dev benchmarks
against the existing bundler with material regressions fixed.

The live implementation status, verified evidence, and ordered remaining work are
tracked in [NATIVE_NEXT_DIFFPACK_STATUS.md](NATIVE_NEXT_DIFFPACK_STATUS.md).
