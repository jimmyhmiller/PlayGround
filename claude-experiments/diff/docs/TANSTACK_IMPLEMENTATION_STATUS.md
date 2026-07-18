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

This identifies the next implementation slice:

1. Treat query-bearing module IDs as a resource path plus loader query.
2. Implement `?url` asset modules.
3. Copy content-hashed assets and return their public URL from the synthetic
   JavaScript module.
4. Add global CSS and CSS-module loaders.
5. Carry emitted CSS/assets into the client and SSR manifests.

That work is necessary independently of the larger TanStack plugin host and is
the next smallest end-to-end step through the real application graph.
