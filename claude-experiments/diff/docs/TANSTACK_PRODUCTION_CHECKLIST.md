# Diffpack production checklist for a TanStack application

Updated: 2026-07-17

## Target

Make Diffpack capable of building and developing a real **TanStack Start React**
application, initially for a Node.js production server. TanStack Start is a
full-stack target: it requires client and server builds, full-document SSR,
streaming, server functions, server routes, route generation, route-aware code
splitting, manifests, assets, and deployment output. It currently integrates
through Vite or Rsbuild, so merely bundling the application's browser entry is
not sufficient.

The first production claim should be deliberately narrow:

> Diffpack can develop, build, test, and deploy a pinned TanStack Start basic
> application on Node.js with React, TypeScript, file-based routes, automatic
> route splitting, server functions, server routes, CSS, static assets, SSR,
> hydration, source maps, and deterministic production output.

Do not claim general Vite, Rolldown, or TanStack Start compatibility until the
corresponding gates below pass.

## Current baseline

- [x] Oxc parsing and TypeScript/JSX transformation.
- [x] Persistent dense module graph and incremental reachability.
- [x] Static imports, literal dynamic imports, basic CommonJS, and JSON.
- [x] Basic tree shaking and package `sideEffects: false` support.
- [x] Conservative flat linking with runtime fallback.
- [x] Basic code splitting and source maps.
- [x] Deterministic clean-versus-incremental oracle.
- [x] Initial cross-module constant folding without generated-code reparsing.
- [x] Interactive linker graph visualizer.
- [ ] Complete JavaScript module semantics.
- [ ] General minification through a retained/combined Oxc AST.
- [ ] Multi-entry, client/server, SSR, or framework-aware builds.
- [ ] CSS and general asset pipeline.
- [ ] Plugin host, development server, HMR, or production deployment adapter.

## P0: freeze the compatibility target

- [x] Pin an exact TanStack Start, TanStack Router, React, and Node version in a
      repository-owned integration fixture.
- [x] Import the official `start-basic` example as the first acceptance app.
- [x] Record the reference Vite build's client files, server files, manifests,
      HTTP behavior, rendered HTML, and route chunks.
- [ ] Add browser-level reference gates for hydration and client navigation.
- [ ] Add one representative application containing nested routes, parameters,
      loaders, a server function, a server route, CSS modules, a global
      stylesheet, images, environment variables, and an npm dependency.
- [ ] Decide whether the initial integration will:
  - [ ] implement a focused Vite-plugin compatibility host, or
  - [ ] port the TanStack Start and Router plugin transformations to a native
        Diffpack plugin.
- [ ] Write an explicit unsupported-feature policy. Unsupported syntax, plugin
      hooks, runtimes, and asset types must fail clearly rather than silently
      producing approximate output.

### Acceptance gate

- [ ] `diffpack dev`, `diffpack build`, and `diffpack start` work on the pinned
      application without keeping Vite or Rsbuild in the execution path.

## P0: build-tool and plugin contract

TanStack file routing and automatic route splitting are build-time plugin
features. The Router plugin transforms route files and creates virtual modules
such as split route component modules. Start adds environment-specific
transforms and manifests.

- [ ] Define a stable Diffpack plugin API with at least:
  - [ ] configuration resolution;
  - [ ] `resolveId`-style virtual and physical module resolution;
  - [ ] `load` for physical and virtual modules;
  - [ ] ordered `transform` hooks with source maps;
  - [ ] build-start and build-end hooks;
  - [ ] chunk/output inspection and manifest generation;
  - [ ] development-server middleware;
  - [ ] file-watch and hot-update hooks.
- [ ] Preserve plugin ordering. TanStack Router transformations must run before
      the React transformation where its documented configuration requires it.
- [ ] Treat query-bearing module IDs as first-class IDs, including route split
      IDs such as `route.tsx?tsr-split=component`.
- [ ] Cache plugin results by plugin version, environment, options, module ID,
      and source hash.
- [ ] Let plugins declare watch files and invalidation dependencies.
- [ ] Ensure one plugin cannot accidentally leak client-only or server-only
      transformations into another build environment.
- [ ] Provide a JavaScript API usable from `diffpack.config.ts`.
- [ ] Add plugin timeouts, cancellation, structured diagnostics, and panic/error
      isolation.

### Acceptance gate

- [ ] The pinned TanStack Router/Start transformations produce the same virtual
      route modules and server-function boundaries as the reference build.

## P0: complete module resolution

- [ ] Implement Node ESM and CommonJS package resolution semantics.
- [ ] Complete `package.json` `exports` and `imports`, nested conditions,
      patterns, self-references, and error behavior.
- [ ] Support configurable conditions for browser, development, production,
      import, require, node, and custom environments.
- [ ] Complete `main`, `module`, browser mappings, extension aliases, directory
      imports, and package type handling.
- [ ] Handle symlinks and monorepo/workspace packages predictably.
- [x] Support automatically discovered TypeScript path aliases.
- [ ] Support an explicitly configured TypeScript project root.
- [ ] Support aliases, externals, built-in Node modules, and browser shims.
- [ ] Preserve query strings and fragments where the loader/plugin pipeline
      needs them.
- [ ] Implement complete `sideEffects` boolean and glob-pattern semantics.
- [ ] Add resolver conformance fixtures for pnpm, npm, Yarn, and common
      monorepo layouts.

## P0: JavaScript and module correctness

- [ ] Give every binding a stable symbol identity rather than relying on text
      names in the flat linker.
- [ ] Implement ESM live bindings, mutation, and namespace object semantics.
- [ ] Implement cycles, temporal dead zones, and specification-correct module
      instantiation/evaluation order.
- [ ] Complete default, named, aliased, namespace, star, and ambiguous
      re-export behavior.
- [ ] Complete ESM/CommonJS interop, including callable/default namespace
      behavior and mixed cycles.
- [ ] Support top-level `await` and async chunk/module execution.
- [ ] Support `import.meta`, `import.meta.url`, and environment-defined
      `import.meta.env` behavior.
- [ ] Correctly classify side effects involving getters, computed properties,
      exceptions, coercion, classes, decorators, and module initialization.
- [ ] Preserve directive prologues and shebangs where appropriate.
- [ ] Handle `eval` and `with` conservatively.
- [ ] Add explicit environment rules for Node built-ins and browser globals.

### Acceptance gate

- [ ] Diffpack agrees with Node and the pinned reference bundler on a large
      ESM/CJS/cycle/re-export corpus, including runtime status, stdout, stderr,
      exported values, and initialization errors.

## P0: evolve the linker IR

- [ ] Replace marker-oriented generated strings with a compact structured
      per-module linker IR containing:
  - [ ] stable symbols and scopes;
  - [ ] imports, exports, re-exports, and symbol demands;
  - [ ] statement declarations and references;
  - [ ] side-effect classifications;
  - [ ] execution-order constraints;
  - [ ] retained Oxc AST nodes or an Oxc-reconstructable representation;
  - [ ] source spans for diagnostics and maps.
- [ ] Keep the incremental invalidation unit at the module level.
- [ ] Keep symbol/statement data in compact local arrays rather than turning
      every AST node into a global incremental fact.
- [ ] Build one combined Oxc program per output chunk from retained statements.
- [ ] Run Oxc compression, mangling, and minifying code generation on that
      structured program without reparsing generated JavaScript.
- [ ] Preserve a conservative runtime fallback for constructs the flat linker
      cannot yet prove safe.

## P0: client/server environment builds

TanStack Start code is isomorphic by default and then separated by build-time
environment rules.

- [ ] Represent build environments explicitly: client, SSR server, and
      production server runtime.
- [ ] Allow the same source module to have different transformed outputs and
      dependency edges in different environments.
- [ ] Implement client-only and server-only module boundaries.
- [ ] Guarantee that server secrets, database clients, filesystem code, and
      server-function implementations cannot enter client chunks.
- [ ] Produce browser RPC stubs for server functions.
- [ ] Produce the server-function implementation registry with deterministic,
      collision-safe function IDs.
- [ ] Preserve server function middleware, validators, methods, responses,
      cancellation, and serialization behavior.
- [ ] Compile server routes and API handlers into the server build.
- [ ] Support static replacement of `process.env.NODE_ENV` in production server
      builds while allowing a configuration to retain runtime lookup.
- [ ] Implement `.env`, mode-specific files, public prefixes, `process.env`, and
      `import.meta.env` without exposing unprefixed secrets to the browser.
- [ ] Add a build-time secret scanner over client output and source maps.

### Acceptance gate

- [ ] A fixture importing a secret-bearing server module from a client-visible
      server function contains no secret or server implementation in any
      client artifact or source map.

## P0: route generation and route-aware splitting

- [ ] Watch the route directory and generate a deterministic typed route tree.
- [ ] Support flat, directory, pathless, layout, index, parameter, splat, and
      escaped route conventions.
- [ ] Detect duplicate and conflicting route paths with useful diagnostics.
- [ ] Transform file routes into reference modules and virtual lazy modules.
- [ ] Support automatic splitting of route components, error components,
      pending components, and not-found components.
- [ ] Preserve critical route configuration and loaders in the eager graph by
      default.
- [ ] Support `.lazy.tsx` routes and explicit dynamic imports.
- [ ] Support configurable split groupings.
- [ ] Invalidate only the affected virtual route modules when a route changes.
- [ ] Update route types and route manifests after add, remove, and rename.

### Acceptance gate

- [ ] Initial navigation, preload-on-intent, direct navigation to every route,
      client navigation, 404 handling, and lazy chunk failure recovery match the
      reference application.

## P0: chunk graph and production output

- [ ] Support multiple entries and environment-specific entries.
- [ ] Build deterministic shared chunks across route and application entries.
- [ ] Keep server-only and client-only chunk graphs isolated.
- [ ] Implement stable content hashes and configurable file naming.
- [ ] Rewrite static and dynamic asset references after final hashing.
- [ ] Emit browser-native ESM chunks and Node-compatible server ESM.
- [ ] Implement public/base paths, relative deployment paths, and preload URLs.
- [ ] Generate client, SSR, route, server-function, and asset manifests required
      by TanStack Start.
- [ ] Include CSS and preload relationships in route manifests.
- [ ] Remove stale files atomically without deleting unrelated user output.
- [ ] Guarantee byte-identical output for clean and equivalent incremental
      builds.

## P0: React and JSX

- [ ] Verify React 19 automatic JSX runtime behavior.
- [ ] Support development JSX metadata separately from production output.
- [ ] Support Fast Refresh boundaries and signature generation in development.
- [ ] Preserve React component identity across accepted hot updates.
- [ ] Handle JSX in dependencies, TypeScript decorators/configuration where
      required, and mixed JS/TS module graphs.
- [ ] Test hydration with nested Suspense and streaming boundaries.
- [ ] Treat React Server Components as explicitly unsupported initially; the
      current TanStack Start RSC feature remains experimental and adds a
      separate Flight/build-environment pipeline.

## P0: CSS and assets

TanStack Start expects its build tool to provide CSS behavior and adds
SSR-aware route asset discovery on top.

- [ ] Support global CSS side-effect imports.
- [ ] Support CSS Modules with deterministic scoped names and TypeScript-friendly
      default exports.
- [ ] Support CSS `@import`, `url()`, dependency resolution, and source maps.
- [ ] Extract production CSS, deduplicate it, and associate it with chunks and
      matched routes.
- [ ] Support `?url`, `?raw`, and asset/file imports.
- [ ] Support images, fonts, SVG, JSON, text, and binary assets.
- [ ] Hash and copy assets while preserving correct URLs under configured base
      paths.
- [ ] Feed stylesheet and asset information into SSR manifests for link tags,
      preload hints, and optional CSS inlining.
- [ ] Prevent flash, duplicated styles, and hydration mismatches in SSR.
- [ ] Define an extension point for PostCSS and Tailwind instead of embedding a
      large CSS ecosystem directly into the core.

## P0: SSR, streaming, and hydration

- [ ] Produce separate client and server entry modules.
- [ ] Render full HTML documents through the Start server entry.
- [ ] Preserve React streaming and Web `ReadableStream` behavior.
- [ ] Inject client entry scripts, route chunks, stylesheets, module preloads,
      and hydration data from manifests.
- [ ] Escape serialized state safely against HTML/script injection.
- [ ] Support redirects, status codes, response headers, errors, and not-found
      responses during SSR.
- [ ] Abort rendering and loader/server-function work when requests disconnect.
- [ ] Verify that server-rendered HTML hydrates without warnings in a real
      browser.
- [ ] Test slow Suspense boundaries and streamed server-function data.
- [ ] Make source maps useful for both browser and server stack traces.

## P1: development server and HMR

- [ ] Add an HTTP development server with HTML fallback and static assets.
- [ ] Add a WebSocket update channel.
- [ ] Build a module graph for HMR accept/decline boundaries distinct from pure
      production reachability.
- [ ] Implement React Fast Refresh.
- [ ] Implement CSS hot replacement without a page reload.
- [ ] Update route trees and virtual modules when route files are added,
      removed, or renamed.
- [ ] Restart or invalidate server modules safely after server-side edits.
- [ ] Preserve client state when the update boundary accepts; reload otherwise.
- [ ] Coalesce duplicate filesystem events and handle atomic-save rename
      patterns.
- [ ] Recover from syntax and resolution errors without restarting the server.
- [ ] Display browser and terminal error overlays with source locations.
- [ ] Support cancellation so obsolete rebuilds never overwrite newer output.

### Acceptance gate

- [ ] A scripted browser suite edits components, CSS, loaders, routes, server
      functions, and dependencies and verifies the correct refresh/reload and
      runtime result after every edit.

## P1: Node deployment

- [ ] Emit a documented production directory layout for client assets and a
      Node server entry.
- [ ] Export a fetch-style server entry compatible with Start's request model.
- [ ] Serve immutable hashed client assets with correct cache headers.
- [ ] Forward dynamic requests to the server entry.
- [ ] Handle shutdown, request cancellation, uncaught errors, and source-mapped
      stack traces.
- [ ] Support a standalone Node/Docker deployment first.
- [ ] Add Nitro/Vite-environment compatibility only after the standalone Node
      target is reliable.
- [ ] Treat Cloudflare, Bun, Netlify, Vercel, and other adapters as separate
      certification targets rather than assuming Node output is portable.

## P1: persistent incremental operation

- [ ] Separate caches for reading, parsing, transformation, plugin output,
      resolution, linker IR, chunking, minification, and emitted files.
- [ ] Version all persistent cache formats by Diffpack, Oxc, plugin, target,
      environment, and relevant configuration versions.
- [ ] Track environment variables and configuration files as build inputs.
- [ ] Invalidate virtual modules and manifests precisely.
- [ ] Avoid rebuilding the client graph for a server-only edit when no shared
      interface changes.
- [ ] Avoid rechunking unrelated routes after a leaf edit.
- [ ] Bound retained old revisions and memory growth during long watch sessions.
- [ ] Test clean builds against long arbitrary edit sequences ending in the same
      filesystem state.

## P1: minification and source maps

- [ ] Feed combined chunk ASTs into Oxc's compressor and mangler.
- [ ] Preserve public/exported names where framework/runtime contracts require
      them.
- [ ] Apply property mangling only with an explicit safe configuration.
- [ ] Compose source maps through plugin transforms, TypeScript/JSX lowering,
      linking, compression, and code generation.
- [ ] Preserve legal comments and license banners.
- [ ] Test minified and unminified output independently.
- [ ] Run browser and server runtime parity gates on minified artifacts.
- [ ] Validate source maps with stack traces, breakpoints, and mapped source
      contents rather than only checking JSON shape.

## P1: diagnostics and developer experience

- [ ] Produce code-framed syntax, transform, resolution, plugin, route, CSS,
      linking, and deployment diagnostics.
- [ ] Include importer chains and environment names in resolution errors.
- [ ] Explain client/server boundary violations and identify the import path that
      leaked the module.
- [ ] Explain why a module or symbol is retained using linker demand paths.
- [ ] Extend the graph visualizer with client/server environment switching,
      chunk membership, HMR boundaries, and before/after edit comparison.
- [ ] Add `--analyze`, JSON stats, metafile output, and bundle-size budgets.
- [ ] Add configuration schema validation and unknown-option errors.
- [ ] Add stable exit codes and machine-readable diagnostics.

## P1: correctness program

- [ ] Turn every current fallback boundary into a named fixture.
- [ ] Import relevant esbuild, Rollup, Rolldown, Node ESM, and Oxc conformance
      cases where licensing permits.
- [ ] Cross-execute Diffpack, the pinned reference build, Node, and a
      browser for applicable cases.
- [ ] Generate random acyclic and cyclic module graphs with imports, exports,
      mutations, and edit sequences.
- [ ] Fuzz parser-to-linker metadata extraction and chunk rendering.
- [ ] Test all supported platforms and case-sensitivity behaviors.
- [ ] Add large-project memory, file-descriptor, and long-watch soak tests.
- [ ] Make nondeterminism a hard CI failure.
- [ ] Make clean-versus-incremental artifact and runtime parity a hard CI
      failure for every production fixture.

## P1: security and operational hardening

- [ ] Prevent path traversal outside configured roots except explicit package
      resolution and allowed filesystem paths.
- [ ] Escape generated HTML, manifests, source-map paths, and runtime strings.
- [ ] Never expose server source or secrets in client output or browser source
      maps.
- [ ] Make server-function IDs deterministic, collision-safe, and
      non-path-leaking in production.
- [ ] Verify same-origin/CSRF assumptions required by Start server functions.
- [ ] Bound input sizes, recursion depth, plugin execution, and diagnostic volume.
- [ ] Avoid following unsafe symlink loops.
- [ ] Use atomic output and cache writes.
- [ ] Audit dependencies and define a vulnerability response/update policy.
- [ ] Document which code executes at build time and what filesystem/network
      authority plugins receive.

## P1: performance gates

- [ ] Benchmark the pinned real TanStack application, not only synthetic graphs.
- [ ] Measure cold build, warm build, component edit, CSS edit, route add/remove,
      server-function edit, dependency edit, and configuration edit.
- [ ] Measure time to browser update separately from filesystem detection.
- [ ] Track peak RSS, retained watch memory, CPU time, file reads, and output
      writes.
- [ ] Require runtime and artifact parity before accepting any timing.
- [ ] Compare unminified with unminified and minified with minified.
- [ ] Establish performance regression budgets in CI.
- [ ] Profile before adding finer-grained invalidation; keep module-level units
      unless a real workload proves a smaller unit is worthwhile.

## P2: packaging, compatibility, and release

- [ ] Publish prebuilt binaries for supported macOS, Linux, and Windows targets.
- [ ] Publish a versioned JavaScript API package and TypeScript definitions.
- [ ] Define configuration and plugin API stability policies.
- [ ] Define minimum Node, browser, and operating-system versions.
- [ ] Add migration and release notes.
- [ ] Add reproducible release builds, checksums, provenance, and signing.
- [ ] Add crash reporting that is opt-in and strips project source/secrets.
- [ ] Create a compatibility matrix for TanStack Start/Router releases.
- [ ] Run the pinned application against new TanStack and Oxc versions before
      declaring them supported.

## Milestones

### M1 — production TanStack Router SPA

- [ ] File-based route generation works.
- [ ] React/TSX, CSS, assets, environment variables, and browser ESM work.
- [ ] Automatic route code splitting and browser navigation work.
- [ ] Development server and React/CSS HMR work.
- [ ] Production static deployment passes browser tests.

This is useful before full TanStack Start support and avoids SSR/server-function
complexity.

### M2 — production TanStack Start on Node

- [ ] Client/server environment builds work.
- [ ] SSR, streaming, hydration, server functions, and server routes work.
- [ ] All required manifests and route assets are correct.
- [ ] Standalone Node deployment passes HTTP and browser acceptance tests.
- [ ] Client output is proven free of server code and secrets.

### M3 — production hardening

- [ ] Long watch sessions, large projects, malformed inputs, and arbitrary edit
      sequences remain correct and bounded.
- [ ] Persistent cache, cancellation, atomic output, diagnostics, security, and
      cross-platform CI are complete.
- [ ] The real-app performance suite consistently meets its budgets.

### M4 — broader ecosystem compatibility

- [ ] Additional deployment adapters are certified independently.
- [ ] Wider plugin compatibility is added from measured application demand.
- [ ] Experimental React Server Components are considered only after their
      TanStack interface stabilizes.

## Release-blocking definition of done

All of the following must be true before calling Diffpack production ready for
the declared TanStack target:

- [ ] The pinned application builds without Vite/Rsbuild in the execution path.
- [ ] Every route passes direct-load, SSR, hydration, and client-navigation tests.
- [ ] Server functions and server routes pass browser and direct HTTP tests.
- [ ] No server-only implementation or secret appears in client artifacts.
- [ ] Clean and incremental builds produce equivalent manifests, artifacts, and
      runtime behavior after randomized edit sequences.
- [ ] Minified production output passes the same runtime suite.
- [ ] Source maps correctly map browser and server stack traces.
- [ ] Development HMR and error recovery pass scripted edit tests.
- [ ] Node deployment survives restart, concurrent requests, streaming,
      cancellation, and graceful shutdown tests.
- [ ] The supported/unsupported surface is documented and unsupported inputs
      fail explicitly.
- [ ] Performance, memory, determinism, security, and soak-test budgets pass in
      CI.

## Official requirements used for this checklist

- [TanStack Start overview](https://tanstack.com/start/latest/docs/framework/react/overview)
- [TanStack Router file-based routing](https://tanstack.com/router/latest/docs/routing/file-based-routing)
- [TanStack Router automatic code splitting](https://tanstack.com/router/latest/docs/framework/react/guide/automatic-code-splitting)
- [TanStack Start server functions](https://tanstack.com/start/latest/docs/framework/react/guide/server-functions)
- [TanStack Start server routes](https://tanstack.com/start/latest/docs/framework/react/guide/server-routes)
- [TanStack Start execution model](https://tanstack.com/start/latest/docs/framework/react/guide/execution-model)
- [TanStack Start CSS styling](https://tanstack.com/start/latest/docs/framework/react/guide/css-styling)
- [TanStack Start environment variables](https://tanstack.com/start/latest/docs/framework/react/guide/environment-variables)
- [TanStack Start hosting](https://tanstack.com/start/latest/docs/framework/react/guide/hosting)
