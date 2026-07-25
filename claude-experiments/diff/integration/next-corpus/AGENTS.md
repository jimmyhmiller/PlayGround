# AGENTS — next-corpus contract

If you edit anything under `integration/next-corpus/`, keep these invariants or the
gate WILL fail (by design):

1. Hermetic. NO network at build or request time. Data is LOCAL TypeScript arrays /
   maps only. No `fetch(`, no `http(s)://` request-scoped calls, no remote images.
   Any image is a committed file in that app's `public/`. `next-corpus-check.sh`
   greps every app for a request-scope `fetch(`/`http(` and FAILS if one appears.

2. Each app is a real app-router tree: `next.config.*` at the root + `app/page.*`
   at the app root (the adapter's detection anchor) + whatever slice it exercises.

3. Every app owns an `expected.json` — the classification + scaffold ORACLE for the
   Tier-1 test (`tests/next_corpus.rs`). Its shape:

   ```json
   {
     "name": "<app dir>",
     "routes": [
       { "path": "/", "kind": "static" },
       { "path": "/blog/[slug]", "kind": "ssg",
         "hasGenerateStaticParams": true, "dynamicParams": true },
       { "path": "/", "kind": "isr", "revalidate": 5 },
       { "path": "/go", "kind": "dynamic", "reasonPresent": true }
     ],
     "handlers": [ { "path": "/api/health", "methods": ["GET"] } ],
     "scaffold": { "loading": true, "error": false, "notFound": false }
   }
   ```

   - `kind` is exactly one of `static` / `forceStatic` / `ssg` / `isr` / `dynamic`
     (the strings `RouteKind::as_str` emits into `prerender-plan.json`).
   - `revalidate` (ISR seconds), `hasGenerateStaticParams` + `dynamicParams` (SSG),
     and `reasonPresent` (dynamic routes carry a non-empty reason) are asserted
     against the plan the adapter emits TODAY. If you change a page's config exports
     you MUST update its `expected.json` in lockstep.
   - `handlers` capture each `route.ts` endpoint's URL path + HTTP methods.
   - `scaffold` flags assert whether a `loading.tsx` / `error.tsx` boundary is
     interned and whether `app/not-found.*` is wired into the not-found tree.

4. If you add / remove a route or handler, update `routes` / `handlers` so the count
   matches — the test asserts the plan's route SET equals `expected.json`.
