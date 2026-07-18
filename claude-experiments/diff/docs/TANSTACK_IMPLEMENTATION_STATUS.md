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
- `?raw`, `?tsr-split`, and unrecognized queries fail with a specific,
  actionable error (never a misleading filesystem read failure).

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
