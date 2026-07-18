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
diffpack:    0/13 TanStack production gates passed
```

The Diffpack result is deliberately not softened: it does not yet emit the
required `.diffpack-output` client/server application layout. Run the
non-failing readiness report with:

```console
npm run acceptance:status
```

Once Diffpack has an application build command, CI should use the strict
`npm run acceptance:diffpack` gate.

## Foundation note (differential-dataflow pivot)

The bundler now sits on the `graph.rs` + `dataflow.rs` (differential-dataflow /
timely) module-graph foundation, not the earlier flat-linker milestone build.

### tsconfig `paths` alias resolution (re-landed)

TypeScript `paths` alias resolution (the `~/*` -> `./src/*` mapping in the
fixture `tsconfig.json`) has been re-landed on the current dataflow foundation.
`resolve_options()` in `src/bundler.rs` now sets
`tsconfig: Some(TsconfigDiscovery::Auto)`, so oxc_resolver discovers the nearest
`tsconfig.json` to each importing file and satisfies `paths` on the path
component that the query-aware `ResourceId` split already isolates (aliases
resolve on the path only; any loader query is re-attached to the resolved id by
`module_id_with_resource`). No project-root plumbing and no change to the
`DirectoryResolutionCache` path/query split.

A Rust integration test
(`bundler::tests::tsconfig_paths_alias_resolves_to_the_real_file`) builds a temp
project with a `~/*` -> `./src/*` `paths` mapping and asserts the aliased
specifier resolves to the real file; it fails without the change and passes
with it.

Bundling `src/router.tsx` no longer emits any `~/`-prefixed
`Cannot find module` diagnostics (previously 25 diagnostics led by
`~/components/DefaultCatchBoundary`; that count is now 0). The graph now
traverses far enough to hit the next, different blocker — a loader gap rather
than an alias failure:

```text
error: loader `?url` is not yet implemented (requested for <fixture>/src/styles/app.css)
```

So the next slice is the `?url` asset loader (step 2 below), which the
query-aware module identity work already isolated.

## Query-aware module identity (landed)

Independent of the alias work, query-bearing module ids are now handled
natively. Bundling would otherwise crash when a query-bearing id
(`src/styles/app.css?url`) reached a raw file read:

```text
cannot read src/styles/app.css?url: No such file or directory
```

Step 1 below now lands. `src/resource_id.rs` introduces a native, query-aware
`ResourceId` that splits a specifier/id into `(path, query, fragment)` at the
resolution boundary and round-trips it back losslessly. The resolver
(`DirectoryResolutionCache::resolve`) resolves only the path component through
the filesystem and re-attaches the original query to form the module id, so an
`app.css` import and an `app.css?url` import are distinct modules. The load
frontier (`load_uncached`/`load_module`) splits the query off before touching
disk; a query-bearing id no longer crashes with a misleading file-not-found and
instead produces a specific, actionable error naming the loader and resource:

```text
loader `?url` is not yet implemented (requested for src/styles/app.css)
```

The remaining implementation slice:

1. Treat query-bearing module IDs as a resource path plus loader query. (done)
2. Implement `?url` asset modules.
3. Copy content-hashed assets and return their public URL from the synthetic
   JavaScript module.
4. Add global CSS and CSS-module loaders.
5. Carry emitted CSS/assets into the client and SSR manifests.

That work is necessary independently of the larger TanStack plugin host and is
the next smallest end-to-end step through the real application graph.
