# next-corpus — a hermetic multi-app corpus for the next app-router adapter

A committed corpus of small, real Next.js app-router apps that exercise DISTINCT
slices of `src/next_adapter.rs` (route discovery + classification + scaffold
generation). It exists so a regression in any classifier arm or generated-entry
shape is caught by more than the single hand-authored `integration/next-app-router`
fixture.

## The two tiers

- Tier 1 (`tests/next_corpus.rs`, runs inside `cargo test`, NO node, NO network):
  for each app it copies the tree to a tempdir and drives the crate's PUBLIC,
  node-free API — `is_app_router`, `configure`/`configure_dev` for client /
  react-server / ssr, and `write_prerender_plan` — then asserts the generated
  `.diffpack-next/` scaffold and every route's classified `kind` in
  `prerender-plan.json` against the app's committed `expected.json`. This is the
  real deliverable: it is pure file IO (milliseconds per app) and needs no
  `npm install`, because `configure` and `write_prerender_plan` only read app
  source and emit scaffold — they never resolve `react` or spawn node.

- Tier 2 (`scripts/rsc/next-corpus-check.sh`, gated in `check.sh`, needs node):
  installs the pinned deps once here, builds each app's three graphs natively,
  boots `scripts/rsc/next-server.mjs`, and curl-smokes SSR / 404 / redirect /
  `?__rsc=1` per app. Auto-skips when node is absent.

## The apps

- `blog-static` — nested layout + route group `(marketing)/about` + dynamic
  `[slug]` with `generateStaticParams` (SSG) + metadata + `loading.tsx`.
- `shop-isr` — `export const revalidate` listing (ISR) + `products/[id]` SSG with
  `generateStaticParams` + `dynamicParams` + next/image (a committed SVG) + CSS.
- `dashboard-dynamic` — `force-dynamic` route, a route reading `cookies()`/
  `headers()`, a `searchParams` page, and `app/go` calling `redirect()`.
- `docs-catchall` — optional catch-all `[[...slug]]` (SSG) from a local docs map,
  `not-found.tsx`, an `error.tsx` boundary route that throws in a Server Component,
  and `app/api/{health,echo}/route.ts` HTTP handlers.

## Hermeticity contract (NO network at request time)

Every app's data is a LOCAL TypeScript array/map. No `fetch`, no remote images, no
network at build or request time. Any image is committed into that app's `public/`.
This is what lets Tier 2 build + serve + assert exact route kinds with zero flakiness
and zero external dependencies. `node_modules` is installed as a setup step (pinned
versions), never committed.
