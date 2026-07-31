# integration/e2e — the real-application truth test

diffpack claims to build Next.js (app router and pages router), Vite, and
TanStack Start applications. This suite exists to find out whether that is
true, against real third-party applications rather than fixtures written to
pass.

Nothing here is part of any build. It is test-only tooling.

## What it does

For every app in `corpus.json`:

1. **Materialize** it from a pinned upstream git SHA into `apps/<id>/`. The
   application source is never edited. (A few entries are instead **first-party**
   fixtures copied from `fixtures/` — see below; and a **monorepo** entry is
   materialized whole and built in place — see below.)
2. **Build it twice** from that one untouched source tree — once with the app's
   own toolchain (`next build` / `vite build`), once with diffpack
   (`diffpack build-app <root> production` / `diffpack build <root> --vite`).
3. **Serve both** deployments on their own ports.
4. **Drive both** with byte-identical scripts in the same real browser
   (`agent-browser`), under an injected determinism shim that seeds
   `Math.random`, `crypto`, and the wall clock.
5. **Compare** the two observation records.

The app's own toolchain is the oracle. A difference is a diffpack defect unless
the reference build is itself broken, in which case the app is excluded from
comparison and that fact is recorded — never quietly dropped.

## Observation channels

Both sides run the same probe (`lib/probe.mjs`) and the results are compared
field by field (`lib/compare.mjs`):

| channel | what must match |
| --- | --- |
| `text` | `<title>`, `<html lang>`, the full rendered `innerText`, the heading outline |
| `structure` | the depth-tagged element outline of everything under `<body>` |
| `attributes` | per-element text and a fixed attribute set (`href`, `src`, `alt`, `value`, `aria-*`, …) |
| `styles` | 44 computed properties on every element, matched pairwise by position |
| `layout` | each element's rendered box, within 1px |
| `assets` | every `<img>` actually decoded (`naturalWidth > 0`); stylesheets present |
| `links` | the set of internal link targets |
| `hydration` | React fibers attached on the diffpack build wherever they are on the reference |
| `interaction` | the same clickable elements, clicked in the same order, producing the same result |
| `navigation` | following the first internal link lands on the same route with the same content |
| `errors` | console errors, uncaught page errors, and failed requests — any class present on diffpack but not on the reference |

Content hashes, chunk names, and `_next/static` paths are normalized away:
those legitimately differ between bundlers. CSS Module class names are not
compared as strings either — the *computed styles* they produce are, which is
what a user can actually perceive.

Two more things are declared per app rather than inferred, so that an exclusion
is always auditable:

- **`volatile`** — regexes for values the app itself renders differently on
  every request. The determinism shim seeds `Math.random` and freezes `Date` in
  the *browser*, which cannot reach a server render, so an app that prints the
  server's clock differs no matter what the bundlers do. Each entry needs a
  `volatileNote` saying why. An **undeclared** difference is always a finding.
- **`resetFiles`** — server-side state the app persists (a counter written to
  disk). Without resetting it, whichever side is driven second inherits the
  first side's mutations.

There is also a raw-document channel that does not go through the browser at
all, because a browser's parser silently *recovers* from a `<script>` spliced
into the middle of a tag — the DOM can look almost clean while the served bytes
are corrupt. Each run reports whether any route was actually served by the
streaming SSR path, so a corpus of prerendered pages cannot quietly stand in as
coverage for streaming.

## Running it

```sh
node integration/e2e/fetch.mjs                 # materialize + install the corpus (slow, once)
node integration/e2e/run.mjs                   # build, serve, drive, compare everything
node integration/e2e/run.mjs next-mdx          # one app (id substring)
node integration/e2e/run.mjs --no-build        # reuse the previous builds
node integration/e2e/run.mjs --build-only      # stop before the browser phase

node integration/e2e/fetch.mjs --heavy         # include the heavy apps (gigabytes, ~20 min)
node integration/e2e/run.mjs --heavy next-calcom
```

Requires `node`, `npm`, and `agent-browser`. The diffpack release binary must
already be built (`cargo build --release`).

Exit code is 0 only when every app that built produced no failing finding.

A finding carries one of three severities, and `lib/compare.mjs::isFailure`
decides which of them count. `fail` is an observed difference and counts.
`error` means the comparison could not be made at all — one side produced no
record for the route — and counts, because an unobservable route is the loudest
thing the suite can see, not the quietest. `info` is a difference in diffpack's
favour, such as an error only the reference produces; it is recorded in full and
never charged.

## Output

- `results/SUMMARY.md` — one row per app
- `results/report.json` — every finding, in full
- `results/<id>/` — per-app evidence: both build logs, both server logs, the raw
  probe record per route per side, the interaction and navigation transcripts,
  and a screenshot of each side

Findings are never truncated on disk. The console rendering shows the first few
per route and always states how many more there are.

## Monorepo apps

Most entries are one directory of a pinned checkout, copied to `apps/<id>/` and
installed there with npm. Some real applications cannot be expressed that way at
all. cal.com's `apps/web` imports 20 sibling workspace packages that resolve only
from the repository root, its package manager is yarn 4 through corepack, and its
`next.config.ts` throws unless three environment variables are set.

Such an entry declares:

```json
"monorepo": { "appDir": "apps/web", "packageManager": "corepack yarn", "installAt": "root" }
```

and is then materialized **whole** and built **in place** in `.cache/<source>`:
the repository is not copied to `apps/<id>/`. A workspace's install is only valid
in the tree its package manager installed into, and that tree is gigabytes — a
per-fetch copy would be neither cheap nor faithful. `lib/apps.mjs` answers three
questions for every entry, and `fetch.mjs` and `run.mjs` both use those answers:
where the tree is (`materializedRootOf`), where the two builds run (`appDirOf`),
and where `node_modules` belongs (`installDirOf` — a workspace hoists to the
root, so asking `apps/web` whether it is installed would answer "no" about a
perfectly installed workspace). The app's toolchain is invoked through the
package manager the entry declares, because `npx --no-install next build` cannot
find a binary yarn linked into the *root's* `node_modules/.bin`.

The source such an entry points at declares `"sparse": []` — the whole
repository, not a cone.

## Declared environment

An entry may declare `"env"`, with an `"envNote"` saying why:

```json
"env": { "NEXTAUTH_SECRET": "diffpack-e2e-not-a-real-secret", … }
```

Those values are written to the app's `.env` before it is built (at the
materialized root, which for a monorepo is the repository root — cal.com's config
loads `../../.env` itself; `"envFile"` overrides the path) and recorded verbatim
in `DIFFPACK_E2E_PROVENANCE.json` alongside the dependency pins. A build that
works only because of something ambient in one developer's shell is neither
reproducible nor auditable, so nothing may be ambient. Because the values are
recorded in a file meant to be read, they must be obvious dummies:
`lib/corpus-monorepo.test.mjs` fails on any declared value that does not look
like one, and a value that cannot be written literally (a quote, a newline) is
refused rather than escaped into something the app would misread.

cal.com's `DATABASE_URL` points at a database that does not exist, deliberately.
The reference build is only an oracle if it needs no live service; a build that
quietly started talking to one would be measuring something else.

## Heavy apps

`"heavy": true` marks an entry whose materialization is measured in gigabytes and
whose install is measured in tens of minutes — cal.com is a 349 MB checkout, 3.4 GB
of `node_modules`, and about 20 minutes. It is **excluded from a default
`fetch.mjs` and a default `run.mjs`**, including when it is named by a filter:
naming it is not consent to a 20-minute install, `--heavy` is. Both commands
print what they left out, because a corpus that silently shrinks lies about its
own coverage. Each heavy entry carries a `"heavyNote"` stating the cost.

## Licenses, and what is never vendored

Every third-party app is **cloned** into the gitignored `.cache/` and built
there. Nothing from any of them is copied into diffpack's own tree, and diffpack
redistributes none of it. That distinction is what lets this corpus use whatever
license an upstream ships, and it is written into each monorepo app's provenance
rather than left implied.

A pinned SHA's license is a fact about *that tree*, not about the project's
reputation — cal.com's platform was AGPL-3.0 with a proprietary `/ee` for years,
and at the SHA pinned here the repository has been relicensed and its `LICENSE`
is MIT. So for a checkout that includes its `LICENSE`, `fetch.mjs` verifies the
file against the id the corpus claims and **fails** rather than write a
provenance file that misdescribes what was cloned.

## First-party fixtures

Two corpus entries are not third-party. They declare
`"firstParty": "fixtures/<dir>"` instead of a `source`/`subdir`, and are copied from
`fixtures/` rather than cloned. They exist only where no pinned upstream example
exercises the behaviour **at all** — today, MDX: between them Vercel's `next-mdx` and
`next-pages-mdx` use no GFM, no remark/rehype plugin, no frontmatter, and an empty
`mdx-components` override map, so almost the whole MDX surface was passing the corpus
untested.

A first-party fixture is still built twice, served twice, driven twice and compared on
the same eleven channels, with the app's own `next build` as the oracle — a differential
test, not a self-assertion. What it cannot do is stand as third-party evidence, since its
author knew what diffpack supports. Each materialized copy says so in its
`DIFFPACK_E2E_PROVENANCE.json` (`"origin": "first-party"`), each corpus entry carries a
`firstPartyReason`, and `lib/corpus-mdx.test.mjs` fails if a fixture stops exercising the
features it was added for. See `fixtures/README.md`.

## What the corpus modifies

The applications are used as published, with one exception, recorded in each
app's `DIFFPACK_E2E_PROVENANCE.json`:

1. **Dependency version pins.** The upstream examples declare `next: "latest"`
   alongside `react: "^18"`, which is not reproducible and not mutually
   consistent (Next 16 requires React 19). `corpus.json`'s `pins` fix the
   versions.

Nothing is installed beyond what an app declares. There used to be a second
exception — `react-server-dom-webpack`, added to every app-router app because
diffpack could not otherwise resolve its own generated entries. That was a
diffpack requirement charged to the app's dependency list, and a recorded
finding; diffpack now resolves the flight runtime from the copy `next` vendors,
so the apps install exactly what they ship with.

For the reference build only, and only as a **retry after the app fails to build
as published**, the harness tries two more things, in order. First `--webpack`,
when the failure is Next 16's refusal to build a webpack-shaped config with
Turbopack — the flag is also passed on the first attempt when the config text
itself calls `webpack()`, but a config *plugin* like `@next/mdx` installs that
function at runtime, where no amount of reading the file can see it, so the
retry keys off Next's own diagnosis instead. Turbopack states that refusal two
different ways and both count: "This build is using Turbopack, with a `webpack`
config", and — when a loader rule's options hold live functions, which is what
`createMDX({ options: { remarkPlugins: [...] } })` produces — "loader … does not
have serializable options". Then type and lint gates are
disabled (`typescript.ignoreBuildErrors`, `eslint.ignoreDuringBuilds`); several
pinned examples are stale against current Next types, and a type error is not
what this suite measures — the oracle only has to be a *running* app. Every
attempt's output is kept in the build log. Everything written for the relaxed
retry is removed before diffpack sees the app, so an app with no `next.config`
still reaches diffpack with no `next.config`.

An entry may also *declare* that relaxation with `"relaxChecks": true` and a
`"relaxChecksNote"` saying why, in which case it is applied on the first attempt
and recorded in the app's provenance file. Only `next-calcom` declares it — the
tree has a pre-existing tRPC type collision unrelated to bundling, and a
ten-minute build spent rediscovering that buys nothing. Nothing about the app's
runtime behaviour is relaxed, on either side.

That wrapper has to preserve the app's config exactly. Next accepts a config
object *and* a `(phase, { defaultConfig }) => config` function, and the wrapper
used to spread the loaded value — which yields `{}` for a function config, so for
apps like cal.com it did not disable two checks, it silently replaced the whole
config with two checks and built a different application. `lib/apps.test.mjs`
evaluates the bytes the wrapper writes, for both shapes and for CommonJS.
