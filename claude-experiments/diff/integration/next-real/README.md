# next-real: real OSS Next.js app-router acceptance corpus

This directory vendors small, pinned, **MIT-licensed** third-party Next.js
**app-router** applications and gates diffpack against them, so that
"full support" is proven against genuine open-source code rather than only a
hand-authored fixture.

## What is here

| App | Source (vercel/next.js `examples/`) | Exercises |
|-----|--------------------------------------|-----------|
| `hello-world` | `examples/hello-world` | minimal `app/` + root layout + static page |
| `mdx` | `examples/mdx` | metadata, `next/font`, `next/image`, CSS Modules, MDX-at-build, nested route |

All apps are pinned to vercel/next.js SHA
`367e5215bf6fbf7a894d940df61e664b08538e22`. See `apps.json` (the single
source of truth: source URL, SHA, features, smoke assertions, and per-app
build frontier) and `NOTICE` (per-app license/copyright).

The app **source** is committed (it is small). `node_modules/` and build
output are gitignored — deps are installed once at test time (see
`.gitignore`).

## Hermeticity

Every vendored app is verified to have **no request-time network, database,
or secret access**: it reads only local files at build time. The only
external references are static anchor `href`s in rendered HTML, which the
build and smoke test never fetch. This keeps the acceptance gate hermetic:
after a one-time `npm install`, nothing touches the network.

## The gate

`scripts/rsc/next-real-check.sh` (wired into `check.sh`) builds each app
with `diffpack build-app <app> production`, and — when the build succeeds —
boots `scripts/rsc/next-server.mjs` and curls each app's `smoke[]` routes,
asserting a 200 and the expected body substring. It **auto-skips** when node
or installed deps are absent.

## Current frontier (honest status)

As of the pinned diffpack tree, **neither app builds**: this diffpack line is
the TanStack Start toolchain and has **no Next.js app-router support**.
`build-app` derives its entry only from TanStack conventions
(`src/router.tsx`, `src/client.tsx`, `src/server.ts`, the
`@tanstack/react-start` default-entry) and fails at config derivation with:

```
error: no production entry found for the app
```

before any app-specific feature is reached. This is the whole point of the
corpus: each app that fails to build is a **recorded frontier signal**
(`apps.json` → each app's `frontier` field), defining the next feature work,
never a silent drop. The gate is forward-compatible: once an app-router
adapter lands and the build succeeds, the same script serves and smokes it
with no further change.

## Adding an app

1. Append an entry to `apps.json` (name, subdir, features, `smoke[]`,
   `frontier`).
2. Vendor its pinned source under `integration/next-real/<name>/` and copy
   the upstream LICENSE; add a `NOTICE` stanza.
3. Keep it hermetic — swap any request-time network/DB for committed local
   fixtures, or pick another app. Never paper over a genuinely unsupported
   feature.
