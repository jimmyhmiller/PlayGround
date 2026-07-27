# What the real-application e2e suite found

Every item below was produced by building a pinned, unmodified third-party
application twice — once with its own toolchain, once with diffpack — and
comparing the two running deployments. Each has a reproduction command.

Status is as of the corpus run after two rounds of fixes, a third round on the
TEST TOOLING itself — where several of these items turned out to be defects in
the suite rather than in diffpack — and the per-app rounds since. `results/`
holds the evidence: `results/<id>/findings.json` exists only for an app that was
actually built, served and driven on both sides, and its contents are that run's
findings. The scoreboard is at the bottom and says exactly how it is counted and
what has and has not been re-measured.

One app is still failing (`next-i18n-routing`, item 21) and it is listed as
failing for the first time here. Nothing regressed: the suite was scoring a
route it could not observe at all as a pass (item 32).

Two items below are struck through rather than deleted. A differential suite
earns its keep by being trusted, and a suite that quietly removes the
differences it got wrong is not auditable — so a withdrawn finding keeps its
number, says it was withdrawn, and is re-filed under "Differences that turned
out NOT to be diffpack defects".

## Blocking: apps diffpack cannot build at all

### 1. A Next.js project is not recognized without a `next.config` file

`next.config` is optional in Next.js. `src/next_adapter.rs::detect_app_router`
and `src/next_pages.rs::is_pages_router` both return early unless one exists,
so diffpack rejects the project outright:

```
error: no client entry found for the app
```

11 of the 24 pinned vercel/next.js examples have no `next.config` and all 11
fail, across both routers: `radix-ui`, `with-sass`, `with-xstate`,
`with-context-api`, `with-web-worker`, `with-absolute-imports`,
`with-dynamic-import`, `with-typescript` (pages), `with-shallow-routing`
(pages), `with-framer-motion` (pages), `i18n-routing`.

```sh
./target/release/diffpack build-app integration/e2e/apps/next-radix-ui production
```

**FIXED.** Detection now keys off a `next` dependency in `package.json` or a
`next.config.*`, with one shared definition of what a Next config file is (three
disagreeing lists were collapsed). All 7 previously-rejected apps retried are
now detected; 4 build outright, 3 progress to the deeper defects below (items
3a and 3b).

### 2. `src/app` is not recognized as the app router

`detect_app_router` only looks at `<root>/app`. Next also supports
`<root>/src/app`, which `blog-starter` and `with-zustand` use.
`src/next_pages.rs::pages_dir` already handles `src/pages`, so the two routers
disagree with each other.

**FIXED** for detection, and the same fix repaired a second `src/app` defect
nobody had noticed: the dev server watched `<root>/app` unconditionally, so
every `src/app` project got zero HMR.

### 3a. `src/app` client components never reach the React Client Manifest

Uncovered by fixing item 2. `next-with-zustand` now builds all three graphs, and
then the react-server render fails for every client component:

```
Could not find the module ".../src/lib/StoreProvider.tsx#default" in the React Client Manifest.
```

**FIXED**, and the title was wrong about both the cause and the scope. The two
graphs never disagreed about the id (both call `rsc::module_reference_id`, a
canonical absolute path). The client component simply never entered the client
graph: island discovery walked `app_dir`, not the project, so every `"use client"`
module that is a SIBLING of the app dir was invisible — `src/components` beside
`src/app`, but equally `components/` beside `app/` (`next-with-jotai`,
`next-with-context-api`, `next-with-styled-components` failed identically and are
not `src/`-rooted). It is unbuilt-graph, not mis-keying.

The two project scans over the same directive concept — `"use client"` islands and
`"use server"` actions — disagreed on both root and skip-list. They are now one
walk (`rsc::walk_project_modules`, rooted at the project), which the `next/font`
and stylesheet-import scans share too; those had the same app-rooted bug (a
`next/font` call or a CSS import outside `app/` was silently ignored).

Fixing the CSS scan exposed a second defect underneath: the react-server build
compiles the served `/rsc.css`, but the `"use client"` transform replaced the
module body with a `createClientModuleProxy` shell and dropped its
`import "./clock.css"` with it — so `has_css` linked a stylesheet the build never
emitted (a 404, and an unstyled page). The proxy now carries the module's
stylesheet imports over, as Next collects a client component's CSS into the
route's stylesheet.

Still discovery-by-filesystem-walk, not by graph reachability: every `"use client"`
file in the tree is pinned even if no route imports it, and a `"use client"` module
inside `node_modules` is not discoverable as an ENTRY. (It still reaches the
manifest when a pinned island imports it — `jotai/esm/react.mjs` does — because the
manifest is derived from the client graph's reachable set, not from the island list.)

```sh
./target/release/diffpack build-app integration/e2e/apps/next-with-zustand production
node integration/e2e/run.mjs next-with-zustand
```

Verified here: `next-with-zustand`, `next-with-jotai`, `next-with-context-api`
and `next-with-styled-components` all build, and the last three now render
**identically** to their own Next build — including `styled-components`, whose
SSR style registry is a genuinely hard case.

### 3b. SSR-of-flight rejects a real client library's module shape

Also uncovered by fixing item 2. `next-radix-ui` builds all three graphs, then
the prerenderer dies:

```
SyntaxError: Module does not provide an export named default
```

A genuine third-party dependency (`@radix-ui/react-dropdown-menu` /
`@radix-ui/react-icons`) is emitted with an ESM/CJS interop shape Node rejects.

```sh
./target/release/diffpack build-app integration/e2e/apps/next-radix-ui production
```

**FIXED** (`src/bundler.rs`, the emitted `__toESM`). Nothing radix-specific: the
module was `node_modules/tslib/tslib.js`, the UMD build that `react-remove-scroll`
and `use-sidecar` pull in, and any TypeScript package published with
`importHelpers` has the same shape. `__toESM` short-circuited on `value.__esModule`
and handed such a module's raw `exports` back as if it were an ES namespace.
`__esModule` cannot carry that meaning — it is a convention marker any CommonJS
file may stamp on its own `exports`, and tslib does, without ever defining
`default`; the un-branded object then hit `__import`'s strict ESM export check and
threw. The decision now runs on `__esmNamespaces`, a `WeakSet` only
`__esmNamespace` (diffpack's own ESM emit) can add to, plus a null-prototype
`Symbol.toStringTag === "Module"` test for a namespace the host produced. Once a
module is known CommonJS, Node's rule applies: `default` is `module.exports`.

That is one bug with two faces, and the silent one was already recorded:
`conformance/fixtures/cjs-esmodule-marker` (marker **with** a `default` key) was
`WRONG-OUTPUT` for diffpack while rolldown and esbuild passed, because `default`
resolved to `exports.default` instead of `module.exports`. It now passes.
`cjs-esmodule-marker-no-default` (the tslib half — marker, no `default`) covers the
crash, and `esm-missing-default-throws` pins the guard that a genuinely missing ESM
`default` must still throw rather than quietly become `undefined`.

The server graph is where this bites because the client graph resolves tslib
through the `module` condition (`tslib.es6.mjs`, pure ESM); the server's
`["node","production"]` conditions route to `modules/index.js`, the ESM file whose
first statement imports the CJS build.

`next-radix-ui` now builds, prerenders 1/1 page, serves and hydrates; the only
remaining channel is `styles` (Tailwind, items 8 and 19 below).

### 3. diffpack requires a dependency the application does not have

Building any real app-router app fails unless `react-server-dom-webpack` is
installed into the app. No real Next.js app depends on it — Next vendors its
own copy — and the generated entries under `.diffpack-next/` import it
directly. Worse, the missing module is reported as a non-fatal "known gap", the
build exits 0, and the failure surfaces much later as a `MODULE_NOT_FOUND`
stack trace out of generated code during prerender.

The e2e corpus installed this package into every app-router app so the rest of
the surface could be tested at all.

**FIXED.** diffpack now resolves the flight runtime the way Next does, in
`src/rsc_runtime_resolve.rs`: the app's own `react-server-dom-webpack` if it has
one (an explicit dependency always wins and nothing is aliased), otherwise the
copy `next` vendors at `next/dist/compiled/react-server-dom-webpack`, resolved
*through that copy's own `package.json` `exports`* under the build
environment's conditions — so `/server` picks the real flight writer under
`react-server`, `/client` picks `client.browser` in the browser and
`client.node` on the server, byte-for-byte the files node would have chosen for
an installed package. The vendored copy `require`s bare `react`/`react-dom`, so
it binds to the app's React; there is no second React in the graph.

One patch Next applies to its vendored copy had to be matched: its **node**
builds read `globalThis.__next_require__` where the npm package reads
`__webpack_require__` (its browser build still reads `__webpack_require__`).
The SSR-of-flight seam now installs both names over the same registry.

If neither copy exists the specifier stays unresolved — a fatal diagnostic, not
a silent gap — and it now says whose requirement it is:

```
  cannot resolve "react-server-dom-webpack/server": Cannot find module …
    imported by /…/.diffpack-next/rsc-entry.tsx
                (generated by diffpack, not by your app)
    this is diffpack's requirement, not your app's: diffpack's app-router
    entries need an RSC (flight) runtime.
    It normally uses the copy `next` vendors at
    next/dist/compiled/react-server-dom-webpack; the installed `next` has none.
```

The package was removed from every corpus app (`package.json` **and**
`node_modules`) and `integration/e2e/fetch.mjs` no longer installs it. Verified
with it uninstalled: `next-hello-world`, `next-active-class-name`,
`next-with-zustand` and `next-with-context-api` build, serve and compare
identically to their own toolchain — including the hydration, interaction and
client-navigation channels. `scripts/rsc/next-missing-dep-check.sh` now gates
both halves: phase A asserts the fixture builds with the package absent, phase B
keeps the original invariant (an import that resolves to nothing is fatal, names
its importer, and writes no output) using a package that genuinely does not
exist.

### 4. Unresolved imports are non-fatal on the `build-app` path

`src/main.rs:392` prints `known gap: <diagnostic>` for every unresolved import
and carries on. The web (`diffpack build`) path gets this right — it fails with
`error: page 'index' ...: N unresolved import(s)` naming each one. The
`build-app` path (Next, TanStack) does not, and ships a bundle that crashes at
runtime instead.

**FIXED**, structurally rather than by flipping a flag: diagnostics are now a
typed `DiagnosticKind` carrying fatality, built from oxc's real `Severity`
(three sites previously collapsed severity into a string). Every fatal
diagnostic is reported at once, and a `sideEffects` glob warning stays
non-fatal on purpose — with a test asserting the build still emits — so the fix
cannot overcorrect into rejecting valid projects. The message now says what to
do, and marks generated files as diffpack's own rather than the user's:

```
error: client build: 2 fatal build diagnostic(s). An artifact with dangling
references would crash at runtime, so no output was written.

  cannot resolve "react-server-dom-webpack/client": Cannot find module …
    imported by /…/app/.diffpack-next/client.tsx
                (generated by diffpack, not by your app)
    install it:  npm install react-server-dom-webpack
```

A dedicated gate (`scripts/rsc/next-missing-dep-check.sh`, wired into
`check.sh`) asserts the build fails, names the package and importer, and writes
no `public/client.js`. Re-run here independently: PASS. It also fixed a wrong
message — the browser stub used to call a missing npm package a "node builtin".
(Item 3 later removed `react-server-dom-webpack` as the gate's subject; the
invariant is unchanged and now exercised with a package that genuinely does not
exist.)

### 4a. A passing gate printed a spurious FAIL, on the wrong line

`scripts/rsc/_gate-prelude.sh`'s ERR net had two defects of its own, both of
which made a green run look red:

* Bash raises `ERR` even when errexit is OFF, so the deliberate `set +e` window
  in `next-missing-dep-check.sh` — which runs a build it EXPECTS to fail and
  then asserts on the status — printed `FAIL: … aborted (exit 1)` on a run that
  PASSED. The trap now reports only while errexit is in force (`case $- in *e*`),
  which is exactly when an abort is unhandled.
* `$LINENO` inside a MULTI-LINE trap string is offset by that reference's own
  line index within the string on bash 3.2 — what macOS ships, and what the
  prelude explicitly targets — so every abort was reported as `line + 1`.
  Capturing `__ln=$LINENO` on the FIRST line of the trap string is correct on
  3.2 and on bash 4/5 alike, with no version test.

**FIXED**, with two cases added to `scripts/rsc/tests/gate-prelude-selftest.sh`
(8 cases): a `set +e` window must produce no output, and the reported line
number is asserted exactly rather than by pattern.

## Correctness: wrong output from a build that succeeded

### 5. Inline flight scripts are written into the middle of HTML tokens

The SSR document served for a stock `create-next-app` page contains:

```html
<img src="/vercel.s<script>(self.__DF_FLIGHT=…).push([1,"…"])</script>vg" alt="Vercel logomark" …>
```

The `<script>` was flushed into the byte stream in the middle of an attribute
value, splitting `src="/vercel.svg"` and destroying the element. The generated
SSR entry (`renderFlightToStream` in `src/next_adapter.rs`) pipes React's
output into a `PassThrough` and then re-reads it with `for await`; a readable
stream re-chunks freely, so the boundaries it hands back are not the boundaries
React wrote at.

This is the cause of two failing gates that were already in the tree:
`scripts/rsc/next-authentic-check.sh` (A5) and `scripts/rsc/next-check.sh`
(next/image).

It is **size-dependent**, which makes it harder to trust rather than easier:
small streamed pages fit in a single write and come out intact (`/blog/hello`
on the fixture serves 12 flight scripts with no corruption), while the larger
stock `create-next-app` page splits. A bug that only appears once a page grows
past a buffer boundary is exactly the kind that reaches production.

**FIXED**, and the investigation corrected the proposed fix. react-dom fills a
fixed 2048-byte view and calls `destination.write()` the moment it is full — so
*React's own write boundaries are not token boundaries either*, and injecting
after each write on a `Writable` we own is equally corrupt. The only token-safe
boundary is the end of a flush cycle, which react-dom signals through
`destination.flush()`. The SSR entry now pipes into
`src/next_runtime/flight_sink.js` (a real file, spliced in with `include_str!`),
which injects only from that hook and hard-errors if a react-dom build never
calls it.

Two structural tests came out of it, both independently re-run here:

- `scripts/rsc/html-integrity.mjs` — a raw-document scanner for a `<script>`
  inside an open tag. This channel matters because **a browser's parser recovers
  from the corruption**, so a DOM-based probe can look almost clean while the
  served bytes are broken. It is now a channel in this suite too.
- `scripts/rsc/ssr-stream-integrity.mjs` — renders a tree built to cross a view
  boundary inside `src="…"`, with a **control that renders the old shape and
  must fail**:

```
OK: control: the old PassThrough + for-await shape splits a tag (1 hit(s)) — the test can fail
OK: createFlightSink never writes a <script> inside an open tag
OK: react-dom drove more than one flush cycle (5) — the boundary hook is live
OK: the flight stays interleaved (first script at 43555, last boundary at 48465, of 49277)
```

### 6. `--out-dir` is resolved against the working directory

`diffpack build <root> --out-dir dist-diffpack` writes `$PWD/dist-diffpack`,
not `<root>/dist-diffpack`, while the default (`root.join("dist")`) is
root-relative. Vite resolves `build.outDir` against the project root.

```sh
cd . && ./target/release/diffpack build integration/vite-react-reference --vite --out-dir dist-diffpack
# writes ./dist-diffpack
```

**FIXED** — one site now owns the semantic (`root.join(out_dir_or_default)`), an
absolute `--out-dir` still passes through, `README.md` documents the rule, and
`tests/out_dir.rs` drives the real binary from an unrelated working directory.
Verified here independently: the output now lands under the project root.

### 7. ~~SSG prerenders a route it classifies as dynamic~~ — the gate is wrong, not the classifier

`scripts/rsc/next-ssg-check.sh` fails with `a blog/*.html was prerendered
(dynamic must be skipped)`, but investigation showed the classifier is right and
the assertion is not. The plan marks `/blog/[slug]` `dynamic` ("reads request
state") and does not prerender it; the single file the gate trips over is
`static/blog/post.html`, produced by `app/blog/post/page.mdx` — a fully static
sibling route that Next itself would prerender. The gate's
`ls "$static/blog"/*.html` matches any file under `blog/`, static or not.

This is a test defect, not a product defect.

### 8. The native Tailwind theme has drifted from upstream Tailwind

`tanstack-start-basic` renders with a different font stack than its own build
produces. Real `tailwindcss@4.3.3` sets

```css
--font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, …
```

while `src/tailwind_theme.css` still carries

```css
--font-sans: ui-sans-serif, system-ui, sans-serif, …
```

10 elements differ in computed `font-family`. A native reimplementation of a
third-party tool drifts silently unless something pins it to the version the
app actually installs.

**FIXED** — the stale token was the symptom; the defect was the lookup.
Diffpack already preferred the app's own installed `theme.css`, but resolved it
by joining `node_modules/tailwindcss` onto the CANDIDATE SCAN ROOT (the
directory `@import 'tailwindcss' source(...)` names) — a source-tree concept
unrelated to Node module resolution. `src/styles/app.css` with `source('../')`
scans `src/`, which holds no `node_modules`, so every TanStack Start app fell
back to the vendored copy without a word. The lookup now walks up from the
STYLESHEET, which is what module resolution is defined against (and is also
what makes pnpm's nested layout and a monorepo root install resolve). The
vendored theme was re-copied verbatim, `tailwind::VERSION` is now the one
definition of the release every vendored artifact came from, a version
mismatch between the installed package and the vendored data is warned about
by name, and the resolved theme is folded into the stylesheet's content hash so
the fix does not ship new bytes under an old immutable URL.

Two guards, because a re-copied file drifts again: `tests/tailwind_theme_resolution.rs`
pins the mechanism (a sentinel token that exists in no released Tailwind must
reach the output from three different project layouts), and
`tests/tailwind_upstream_drift.rs` byte-compares the vendored theme and
preflight against the installed `tailwindcss` and against the reference build.
Its lockfile-vs-`VERSION` check needs nothing but checked-in files, so it can
never be inert; the rest skip loudly and `DIFFPACK_REQUIRE_UPSTREAM=1` (set by
`check.sh`) turns those skips into failures.

### 9. The JSX import source is hardcoded to `react/jsx-runtime`

create-vite's `preact-ts` template sets
`compilerOptions.jsxImportSource: "preact"`; diffpack ignores it and fails with
`cannot resolve "react/jsx-runtime"`. `jsxImportSource`/`jsxFactory` are
bundler configuration, not a plugin concern.

**FIXED.** How JSX is lowered is now a per-file contract
(`transform::JsxConfig`), resolved from the tsconfig that OWNS the file and
layered under `vite.config`'s `esbuild.*` / `oxc.jsx`, with a file-level
`@jsxImportSource` pragma still on top. Ownership is what matters, not
proximity: create-vite's root config is solution-style (`{"files":[],
"references":[...]}`, no `compilerOptions` at all), so a nearest-`tsconfig.json`
read finds nothing — the resolver follows `extends` and `references` to the
config that actually claims `src/`. Two classes of file are deliberately left on
react: anything under `node_modules` (which gets no tsconfig at all), and
diffpack's own generated `.diffpack-next/` sources, which live inside the project
root and would otherwise be claimed by Next's `"include": ["**/*.tsx"]`.
`vite-preact-ts` now builds and passes every channel.

### 10. `.vue` / `.svelte` files produce a misleading error

```
error: page `index` (…/vite-vue-ts/index.html): 1 fatal build diagnostic(s). An
artifact with dangling references would crash at runtime, so no output was written.

  …/vite-vue-ts/src/App.vue: Unexpected JSX expression
```

The import resolved fine. The file is a single-file component that needs a
compiler diffpack does not have — which is a legitimate boundary, but the error
must say so rather than report a JSX syntax error, under a header about dangling
references, in someone else's language.

```sh
./target/release/diffpack build integration/e2e/apps/vite-vue-ts --vite \
  --out-dir integration/e2e/apps/vite-vue-ts/dist-diffpack
```

**FIXED**, by removing the branch that made it possible. `load_special_module`'s
loader table used to end in a bare `None`, and `None` means "read this as
JavaScript" — so "unknown extension" and "JavaScript" were the *same branch*, and
oxc dutifully reported a JSX error on a Vue `<template>`. The JS family is now an
explicit allow-list (`.js/.jsx/.mjs/.cjs/.ts/.tsx/.mts/.cts`, `.json`, `.md/.mdx`,
plus extensionless files, which `node_modules` is full of); everything else is
named. Two classes, because the honest remedy differs: formats diffpack
recognizes and cannot compile name the compiler that would be required (`.vue`,
`.svelte`, `.astro`, `.marko`, `.riot`, `.imba`, `.civet`, `.coffee`, `.res`,
`.re`, `.elm`), and everything else says only that no loader handles the
extension and points at `?raw`/`?url`, which still work because the query check
runs first.

```
error: …/vite-vue-ts/src/App.vue: `.vue` is a Vue single-file component, not JavaScript
  compiling it requires the Vue SFC compiler (@vue/compiler-sfc, normally run by
  @vitejs/plugin-vue); diffpack hosts no JS plugins and has no built-in `.vue` compiler
  the file was found on disk: this is neither a missing import nor a JavaScript syntax error
  build this project with its own toolchain instead
```

Two adjacent misreports in the same channel went with it. A `.node` native addon
used to fail *inside the resolver*, so a file that was found on disk printed
`cannot resolve "x": native module "x" is not supported` followed by
`install it: npm install x`; it is now a loader-level error like any other
unbundlable file. And `partition_diagnostics`'s "an artifact with dangling
references would crash at runtime" was printed for every fatality class —
dangling references are what an *unresolved import* leaves behind, so a source
error now says the emitted code would not match the source.

Known limitation: the parallel load path returns the first error in sorted-path
order, so one offending file is named per run rather than all of them. Reporting
every source at once needs a fatal `DiagnosticKind` for unhandled sources and a
diagnostics channel on `SpecialModule`.

Superseded for `.vue`/`.svelte` by finding 25: those two are now COMPILED, by the
app's own compiler, rather than merely named. The table still names `.astro`,
`.marko`, `.riot`, `.imba`, `.civet`, `.coffee`, `.res`, `.re` and `.elm`.

### 25. Vue and Svelte single-file components cannot be built at all

Finding 10 made the failure honest; it was still a failure. `vite-vue-ts` and
`vite-svelte-ts` were the last two apps in the corpus that produced no artifact:

```
error: …/vite-vue-ts/src/App.vue: `.vue` is a Vue single-file component, not JavaScript
error: …/vite-svelte-ts/src/App.svelte: `.svelte` is a Svelte component, not JavaScript
```

"A Vite app" includes Vue and Svelte apps, so this was a hole in the drop-in
claim, not a boundary.

```sh
node integration/e2e/run.mjs vite-vue-ts vite-svelte-ts
```

**FIXED.** Both compilers ship as ordinary npm packages the app already depends
on (`@vue/compiler-sfc` under `@vitejs/plugin-vue`, `svelte/compiler` under
`@sveltejs/vite-plugin-svelte`), so a component is compiled by the APP's OWN
copy — `node` spawned with cwd = the project root, the same shape
`crate::less_stylus` already uses for Less/Stylus and `crate::postcss` for
PostCSS. diffpack reimplements neither compiler and still hosts no JS plugin.

New: `src/sfc.rs` + `src/sfc_runner.mjs`. The runner mirrors what each Vite
plugin does for a production client build — for Vue: `parse`, `compileScript`
with `inlineTemplate` and `genDefaultAs: "_sfc_main"`, `compileTemplate` when
there is no `<script setup>`, `compileStyleAsync` per `<style>` block, the
`__scopeId` / `_export_sfc` tail; for Svelte: `compile` with
`{ generate: "client", dev: false, hmr: false, css: "external" }` plus the
project's `svelte.config.js` `preprocess`/`compilerOptions`.

The compiler's output is deliberately **not** final. It still carries the
component's own imports and, for a Vue `<script lang="ts">`, TypeScript
annotations, so it is fed through the ordinary module pipeline: the imports
become real graph edges and the types are stripped by the same transform a `.ts`
module takes. That is exactly what `@vitejs/plugin-vue` does with its own output
(`transformWithOxc(..., { lang: "ts" })`). Because a `.vue` extension cannot say
which language the *generated* code is in, the transform gained an explicit
`SourceLanguage` parameter (`FromPath` for every module read off disk). Style
blocks come back as plain CSS and go through the same `load_stylesheet_from_text`
pipeline a hand-written stylesheet takes, so `url(...)` rebasing, PostCSS and
`@import` edges all apply.

Verified against the app's own toolchain, not just against the harness: for a
scoped-style SFC, diffpack derives the same component id as `vite build`
(`.card[data-v-eefd2ead]` on both sides) and the same Svelte scope class
(`.card.svelte-1n46o8q` on both sides).

Two real resolver defects surfaced on the way, each with its own regression test:

* **A root-absolute import was a hard error.** Vue's asset-URL transform emits
  `import _imports_0 from "/icons.svg"` for `<use href="/icons.svg#x">`. Vite
  resolves `/x` against the project root, and — when the file lives in
  `public/` — yields its public URL, because `public/` is copied to the site root
  verbatim. diffpack handed `/icons.svg` to the filesystem resolver and failed.
  Now `DirectoryResolutionCache::resolve_root_absolute` tries `<root>/x`, then
  `<root>/public/x` as a new `?public-url` loader whose module is exactly the
  URL — no second, hashed copy of a file the site already serves.
* **`resolve.dedupe` broke subpath imports of any package with an `exports`
  map.** dedupe is carried as a directory alias (`svelte` →
  `<root>/node_modules/svelte`), and a subpath was answered by joining onto it —
  but `svelte/internal/client` is a *key in the exports map*, not a file at that
  path, so `svelte/internal/client` failed to resolve in an app where svelte is
  installed. An alias that fails now retries the specifier as written from the
  project root, which is what dedupe means and lets the package's own `exports`
  decide. It only runs where the build would otherwise have failed.

Both apps now pass every channel, and `vite-react-ts` is unchanged.

### 14. JSX in a `.js` file is a parse error, so entire Next pages vanish

The most consequential defect in this round, and it was **invisible until
unresolved imports became fatal** (item 4). Next.js compiles JSX in `.js` — its
SWC loader treats `.js` as JSX-capable — and a large fraction of real Next apps
rely on it. diffpack does not. Before item 4, that produced a *successful* build
of an empty application:

```
reachable 19 modules; 2 diagnostic(s)
  known gap: pages/about.js: Unexpected JSX expression
  known gap: pages/index.js: Unexpected JSX expression
emitted .diffpack-output/public: 1 public .js
```

Exit 0, both of the app's pages silently dropped. Two corpus apps hit it
(`next-pages-shallow-routing`, `next-pages-framer-motion`).

The rule has to depend on the project kind: Vite deliberately rejects JSX in
`.js` and tells you to rename the file, and diffpack matching Vite there is
correct.

```sh
./target/release/diffpack build-app integration/e2e/apps/next-pages-shallow-routing production
```

**FIXED**, as a per-project rule rather than a global constant. A new
`JsxExtensions` kind is threaded from the project config to the one parse that
reports diagnostics: Next projects treat `.js`/`.mjs`/`.cjs` as JSX-capable,
Vite projects keep Vite's refusal. `.ts`/`.mts`/`.cts` stay JSX-free under both
(`<T>x` is a type assertion there).

Two things the fix got right beyond the ask. Auxiliary parses (the directive
probe, export scan, `define` folding, `next/font`) now use the *widest* rule on
the stated grounds that a scan narrower than the real parse is a silent wrong
answer while a wider one is harmless — which removed a whole class of latent
bugs without threading a flag through fifteen signatures. And `is_refresh_boundary`
in `src/hmr.rs` was hardcoded to `["jsx","tsx"]`, so a Next `.js` component would
have compiled and then full-page-reloaded on every edit.

Verified here: `next-pages-shallow-routing`, `next-pages-framer-motion` and
`next-strict-csp` all build, and a Vite project with JSX in `.js` is still
refused — with a message that explains itself and admits its own limits:

```
JSX is not enabled for `.js` files. Vite/esbuild parse `.js` as plain JavaScript
on purpose and diffpack matches that; `esbuild.include`/`esbuild.loader` is not
honored. Rename it to `main.jsx`. (A Next.js project, which does allow JSX in
`.js`, is detected automatically.)
```

**FIXED**, as a per-project rule rather than a global loosening.
`parser::JsxExtensions` is now the single definition of which extensions may
hold JSX — `JsxAndTsxOnly` (Vite/esbuild, the default) or `NextJs` (`.js`,
`.mjs`, `.cjs` too, matching Next's SWC loader, which enables jsx for everything
that is not a plain `.ts`). It rides on `BuildConfig`, so only the two Next
adapters opt in; `.ts` stays JSX-free under both kinds, because there `<T>x` is
a type assertion. `check.sh` now fails if `SourceType::from_path` reappears
outside `src/parser.rs`, so a second copy of the rule cannot drift back in.

Two things the original report understated:

* The diagnostic count **under-reported** the damage. framer-motion listed three
  files, but `components/Gallery.js` and `components/SingleImage.js` contain JSX
  too — they were never even discovered, because a fatal parse returns a dummy
  program, so the importing page contributed *no dependencies* and its whole
  subtree left the graph. The app now reaches 424 modules (was 0 emitted).
* Every *auxiliary* parse of the same file (the `"use client"` directive probe,
  export enumeration, `define`/dead-branch folding, `next/font`) had the same
  blind spot, and each answers "nothing here" rather than failing. Those now go
  through `parser::scan_source_type`, deliberately the widest rule: a scan that
  is narrower than the module's real parse is a silent wrong answer, while one
  that is wider is harmless (JSX parsing accepts a superset, and if the
  project's rule rejects the file the main parse fails the build anyway).

The Vite half is now an actionable error instead of oxc's bare
`Unexpected JSX expression`, and it does not point at `esbuild.include` /
`esbuild.loader`, which diffpack genuinely does not read:

```
src/main.js: JSX is not enabled for `.js` files. Vite/esbuild parse `.js` as
plain JavaScript on purpose and diffpack matches that;
`esbuild.include`/`esbuild.loader` is not honored. Rename it to `main.jsx`.
(A Next.js project, which does allow JSX in `.js`, is detected automatically.)
```

Dev is fixed with it: `hmr::is_refresh_boundary` hardcoded `["jsx","tsx"]`, so a
Next `.js` component would have built and then full-page-reloaded on every edit
instead of hot-swapping. It uses the same rule now.

Regressions: `tests/jsx_extensions.rs` (a Vite project whose entry `.js` holds
JSX must fail with that message, and a Next pages project whose `.js` page
imports a `.js` component must build *with the component in the bundle*), plus
unit tests in `src/parser.rs`, `src/transform.rs` and `src/hmr.rs`.

## Behavioural defects — apps that build, then render differently

These only became visible once the detection fix let 14 more apps reach the
compiler. Every one is a user-visible difference from what the app's own
toolchain produces.

### 15. `next/link` with an object `href` is not interpolated

`next-pages-typescript` renders **every** dynamic user link as the literal route
pattern:

```
reference: /users/101  /users/102  /users/103  /users/104
diffpack:  /users/[id] /users/[id] /users/[id] /users/[id]
```

The app passes `href={{ pathname: '/users/[id]', query: { id } }}`. Every link on
the page is dead.

**FIXED**, and the title named only half of it: the shim also dropped `as`
entirely (`as: _as`), which is the form `next-pages-typescript` actually uses
(`<Link href="/users/[id]" as={`/users/${data.id}`}>`). Both halves of Next's
`resolveHref` now exist, in one dependency-free module (`scripts/pages/pages-url.js`)
that `next/link` and both build entries share: an object `href` has its dynamic
segments interpolated from `query` with the consumed keys removed and the leftovers
appended, and `as` is the displayed URL when present. A dynamic segment with no
query value is a LOUD error naming the href and the segment — rendering the literal
`[id]` was the defect, so it is not a fallback that survives.

### 16. Pages-router built-in i18n is not implemented, and fails silently

`next-pages-i18n-routing` renders with every locale value empty:

```
reference: Current locale: en  Default locale: en  Configured locales: ["en","fr","nl"]  Locale switcher: fr nl
diffpack:  Current locale:     Default locale:     Configured locales:                   Locale switcher:
```

`<html lang>` is empty, the locale switcher's `<li><a>` elements are absent
entirely (19 vs 15 elements), and the `/fr` and `/nl` routes do not exist. The
build reports no error — it just serves an app with the feature missing.

**FIXED — implemented, not error-gated.** Sub-path locale routing turned out to be
a contained change, so the feature is real rather than a refusal. The pages adapter
now reads `next.config`'s `i18n` (through the same single config eval the app router
uses) and bakes `{ locales, defaultLocale, localeDetection }` into both generated
manifests. From there: the request path is split on a leading `/<locale>` before
route matching (the default locale stays unprefixed), the active locale reaches
`getStaticProps` / `getServerSideProps` / `getStaticPaths`, `router.locale` /
`locales` / `defaultLocale` (server and client), `__NEXT_DATA__`, `<html lang>`,
and `next/link`'s `addLocale` prefixing; SSG prerenders every locale under
locale-prefixed URLs and the ISR cache keys on the locale-prefixed path; and the
bare root honours `NEXT_LOCALE` / `Accept-Language` detection with Next's redirect.

The one sub-feature NOT implemented is `i18n.domains` (domain-based locale
routing), and it is a HARD BUILD ERROR naming the key rather than a silent gap:

```
next.config: `i18n.domains` (domain-based locale routing) is NOT implemented by
diffpack's pages-router adapter. Remove it, or serve the app with `next start`.
`i18n.locales` / `i18n.defaultLocale` (sub-path locale routing) ARE supported.
```

### 17. JSX text whitespace between elements is dropped

`next-with-redux` renders the counter as `-0+` where Next renders `- 0 +`, and
`Add AmountAdd AsyncAdd If Odd` where Next renders them separated. This survives
interaction too — every click produces the same divergence — so it is the
rendered output, not a timing artifact.

**FIXED**, pinned by `src/transform.rs::jsx_text_whitespace_follows_the_jsx_rules`
(the JSX rule: a text run is dropped only when it contains a newline and is
otherwise whitespace; an interior space between elements on one line survives).
`next-with-redux` last measured **0 findings on every channel**
(`results/next-with-redux/findings.json`).

### 18. `next/font` produces no size-adjusted fallback, so layout shifts

Next generates a local fallback face with matched metrics:

```
reference: font-family: Inter, "Inter Fallback"
diffpack:  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif
```

That is the whole point of `next/font` — avoiding layout shift — and the
measured layout does differ (`next-blog-starter` pages are 5px taller under
diffpack). Affects `next-mdx`, `next-blog-starter`, `next-github-pages`.

**CLOSED on all three.** `next-github-pages` was the first re-measured: both
sides compute `font-family: Inter, "Inter Fallback"` on every element, one
stylesheet each, every channel green. `next-mdx` and `next-blog-starter` were
re-measured in the rounds after and are **0 findings** each
(`results/next-mdx/findings.json`, `results/next-blog-starter/findings.json`) —
this was the last channel either of them was failing on.

### 19. Tailwind utilities silently do not apply, breaking a real control

On `next-blog-starter`, **194 computed-style differences across 60 elements**,
and the theme-switcher button is not merely styled differently — it is gone:

```
button  display:  block    (reference)  vs  inline-block (diffpack)
button  position: absolute (reference)  vs  static       (diffpack)
button  box:      26x26    (reference)  vs  0x0          (diffpack)
interaction: 1 interactive element on reference, 0 on diffpack
```

A user cannot click it. Separately, diffpack emits raw `oklch()` where the real
Tailwind build emits `rgb()` (`oklch(0.704 0.04 256.788)` vs
`rgb(148, 163, 184)`), so colours differ on any browser resolving them
differently.

**FIXED (root cause: the whole v3 dialect compiled as v4).** `next-blog-starter`
is a Tailwind **v3** app (`@tailwind base|components|utilities` +
`tailwind.config.ts`, `tailwindcss@3.4.19`). Diffpack compiled it entirely
against v4 reference data: the v4 default theme (hence `oklch()`), v4's built-in
literals (`rounded-full` = `calc(infinity * 1px)` = 33554432px, not `9999px`),
and v4's preflight (which resets `border-color` to `currentColor`; v3 resets it
to gray-200 `#e5e7eb`, which is what 125 of the diffs were). Now:
`scripts/tailwind-config-eval.mjs` resolves the config through the app's OWN
`tailwindcss/resolveConfig`, so the emitted `@theme` is v3's real resolved theme
(defaults included), and a v3 entry gets the vendored v3 preflight
(`src/tailwind_preflight_v3.source.css`) with its `theme()` calls resolved.
Style diffs across the three routes: **297 -> 160**, of which 136 are item 18
(`next/font` fallback) and 5 are a `next/image` gap (`color: transparent`).
Layout on `/`: 17 differing elements -> 0.

The residue this item recorded as STILL OPEN — v3-vs-v4 utility *semantics*
rather than theme data — is now closed too. v3 emits
`.text-4xl{line-height:2.5rem}` where v4 emits
`line-height:var(--tw-leading, var(--text-4xl--line-height))`, so a `leading-*`
on the same element won in diffpack but lost to a later `md:text-4xl` in the
reference (4 elements, 40px vs 45px); and v3's `box-shadow` composition has 3
slots where v4's has 5. Both are keyed off the same `Dialect` and pinned by
`src/tailwind.rs::v3_entry_line_height_follows_source_order_not_v4s_tw_leading`,
`v3_font_size_without_a_line_height_emits_only_font_size` and
`v3_entry_box_shadow_composes_three_slots_not_v4s_five`. `next-blog-starter`
last measured **0 findings on every channel**
(`results/next-blog-starter/findings.json`).

### 20. ~~Styles are injected into an app that has none~~ — WITHDRAWN

This was never a diffpack defect. It was the harness serving the app at the
wrong URL prefix, and it is written up in full under "Differences that turned
out NOT to be diffpack defects" below, where a false positive belongs. The
number is kept so the earlier rounds' references still resolve.

### 21. `next-i18n-routing` (app router) serves, then redirects both routes off the host — STILL OPEN

Originally recorded as `serve-failed`: the app-router locale app never answered
on its port. That much is out of date — the server comes up and the suite
compares the app. What it finds is worse than a dead port, and the suite was
hiding it (see item 32).

`results/next-i18n-routing/probe-_en_US-diffpack.json` records `"record": null`
for **both** of the app's routes, with the network log showing where the browser
went:

```
GET http://127.0.0.1:58483/en-US   -> (no status)
GET http://localhost/en/en-US      -> (no status)
```

The document capture for that route is the single line
`<!-- e2e: fetch failed: fetch failed -->`. diffpack's locale redirect sends the
browser to an **absolute** `http://localhost/…` URL — the wrong host, no port —
and with the locale prefix **doubled** (`/en/en-US`). Neither route can be
observed at all, so there is nothing to compare on any channel.

This is **not fixed here** and is not owned by this cluster's work. It is
recorded with its evidence and, as of item 32, it is scored as the failure it
is. Reproduce with:

```sh
node integration/e2e/run.mjs next-i18n-routing
```

### 22. A render error takes down the entire production server

Found by adding `with-strict-csp` — the first app in the corpus that is
genuinely *dynamic* (`await headers()` in the page opts out of prerendering), so
it is the first to exercise the streaming SSR path end to end rather than being
served as a static file.

Its react-server render fails (item 23), and the failure handling then does
this:

```
next-ssr stream onError: An error occurred in the Server Components render…
Error [ERR_HTTP_HEADERS_SENT]: Cannot write headers after they are sent to the client
    at Server.<anonymous> (.diffpack-output/next-server.mjs:1427:9)
error: production server exited with exit status: 1
```

The error path calls `res.writeHead(500)` on a response whose headers already
went out with the shell, which throws, and the throw is unhandled — **one bad
request kills the Node process**. A production server must survive a failing
render.

**FIXED.** The emitted server's request handler now goes through `failRequest`
(`scripts/rsc/next-server.mjs:78`), which never throws: once `res.headersSent ||
res.writableEnded` the status line is spent, so it ENDS the response — a
truncated document rather than a hung socket or a dead process — and only writes
a 500 when the shell has not gone out. Reporting the failure failing is caught
too and drops the connection. Alongside it: process-level `uncaughtException` /
`unhandledRejection` handlers that log rather than exit, and `error` listeners on
both halves of an in-flight streaming response. Pinned in `src/next_adapter.rs`
(`the_streaming_renderer_reports_a_post_shell_error_without_writing_headers` and
the assertions above it, which fail if the catch writes a status line
unconditionally).

### 23. `headers()` + `next/script` fails the Server Components render

`with-strict-csp` builds cleanly and then fails every request. The app is small:
a middleware that sets `x-nonce`, a page that does `await headers()` and renders
`<Script nonce={…}>`.

```sh
node integration/e2e/run.mjs next-strict-csp
```

**FIXED.** The app builds, serves, streams and hydrates. Re-verified here on its
own: `2/2 compared app(s) behave identically to their own toolchain`, zero
findings on every channel, and it is the route the suite reports as exercising
the streaming SSR path (`streaming SSR exercised on 1 route(s):
next-strict-csp/`) — so item 5's inline-flight-script channel is now covered by a
genuinely dynamic app, which closes the verifier note about that too.

Its last failing channel was `hydration`, and that one was the HARNESS's fault,
not diffpack's: see the `next-strict-csp` entry under "Differences that turned
out NOT to be diffpack defects".

### 24. v3 gradients interpolate in oklab, and `space-*` margins land on the wrong edge

`next-radix-ui` — the last non-image difference left in the corpus — showed 5
computed-style differences across 4 elements:

```
div  background-image  reference "linear-gradient(to right, rgb(6, 182, 212), rgb(59, 130, 246))"
                       diffpack  "linear-gradient(to right in oklab, rgb(6, 182, 212) 0%, rgb(59, 130, 246) 100%)"
h1   margin-top        reference 16px  diffpack 0px
h1   margin-bottom     reference 0px   diffpack 16px
```

```sh
node integration/e2e/run.mjs next-radix-ui
```

**FIXED**, and both halves are the same root cause as item 19: the app pins
`tailwindcss@3.4.9` and diffpack compiled its *utility semantics* the v4 way.
Item 19 fixed the theme data and the preflight; these are three more places
where the two versions genuinely disagree, now keyed off the existing
`Dialect` in `src/tailwind.rs`.

*Gradients.* v4 routes the direction through `--tw-gradient-position` and
interpolates `in oklab`; v3 writes it straight into `background-image` and
interpolates in sRGB. v4 registers `--tw-gradient-{from,via,to}-position` at
`0%`/`50%`/`100%`; v3 declares them EMPTY, which is why the reference gradient
carries no stop percentages at all. v3 also has no `--tw-gradient-via` or
`--tw-gradient-via-stops` (`via-*` inlines its colour), each stop carries its own
position (`<color> var(--tw-gradient-from-position)`), and `from-*`/`via-*` fade
`--tw-gradient-to` to the same colour at zero alpha. Those empty defaults cannot
be expressed with `@property` — a registered `syntax: "*"` property with no
initial value is guaranteed-invalid, so `var()`-ing it would poison every stop
declaration — so under v3 they are emitted as a plain
`*,::before,::after,::backdrop{--tw-gradient-from-position: ;…}` rule and the
whole gradient group is left unregistered (v3's `--tw-gradient-from` holds a
`<color> <position>` pair, which v4's `syntax: "<color>"` registration would
reject outright).

*Margins.* v4's `space-*`/`divide-*` select every child but the LAST and push on
the trailing edge; v3 selects every child but the FIRST
(`> :not([hidden]) ~ :not([hidden])`) and pushes on the leading one. Both give
the same gaps, but the margin lands on a different edge of a different element,
which is exactly what the `h1`/`button` diffs were. `divide-*` rides the same
selector, so its widths were flipped too (they were not observable in this app,
but changing the selector without the edges would have introduced the bug); v3
also has no `--tw-border-style` slot and normalizes a `0` value to `0px`.

Regression tests (`src/tailwind.rs`):
`v3_entry_gradients_interpolate_in_srgb_with_no_stop_positions`,
`v3_entry_space_utilities_style_the_leading_edge_of_every_child_but_the_first`,
`v3_entry_divide_utilities_share_the_v3_child_selector_and_edges`,
`v3_transparent_color_matches_upstreams_transparent_to`. Each asserts the v3
shape against what `tailwindcss@3.4.9` itself emits for the same classes, and
each also asserts the v4 shape is untouched.

`next-radix-ui`: **5 findings -> 0**, every channel green. `next-blog-starter`
(the corpus's only other v3 app) re-run and still green.

NOT fixed, and out of scope here: a `/<pct>` opacity modifier on a v3 colour
still compiles to v4's `color-mix(in oklab, …)` rather than v3's
`rgb(r g b / .5)`. Those two do not resolve to the same colour in general. No
app in the corpus exercises it, so it is recorded rather than guessed at.

### 31. MDX wraps a stand-alone component in a paragraph — and the app was never compared at all

`next-pages-mdx` was the one app in the corpus that was never measured against
anything. Its own toolchain could not build it, so there was no oracle:

```
ERROR: This build is using Turbopack, with a `webpack` config and no `turbopack` config.
   NOTE: your `webpack` config may have been added by a configuration plugin.
> Build error occurred
Error: Call retries were exceeded   { type: 'WorkerError' }
```

Two defects, one on each side of the comparison.

**The harness could not build the reference.** Next 16 builds with Turbopack by
default and refuses a config that carries a `webpack` function without a
`turbopack` one; `--webpack` is the way through, and the harness already passed
it — but it decided by grepping the config TEXT for `webpack(`. This app's
config is `module.exports = withMDX({ pageExtensions: [...] })`: the string
"webpack" appears nowhere, because `@next/mdx` installs that function at
*runtime*. A textual probe cannot see a config plugin, and no pin change or
newer `@next/mdx` was needed — Next names the case precisely in its own error,
so the retry now reads the answer back off the failed build instead of trying to
out-guess it statically (`integration/e2e/lib/apps.mjs`, `needsWebpackFlag` +
the three-attempt `buildReference`). The app is built as published, unmodified,
with its original `@next/mdx@^9.1.1` / `@mdx-js/loader@^1.5.1` pins intact.
Pinned by `integration/e2e/lib/apps.test.mjs`
(`a plugin-added webpack config is recognized from the failed build's output`),
which also asserts an unrelated failure is NOT retried as a builder-choice
problem.

**Then diffpack's MDX differed.** With the oracle finally available, the page
compared as:

```
reference: 0:div|1:h1|1:p|1:button
diffpack:  0:div|1:h1|1:p|1:p|2:button
```

The page's last line is `<Button>👋 Hello</Button>`. To micromark — and so to
markdown-rs, which ports it — a JSX line is "flow" only when the line holds
nothing but tags, so `<Button>text</Button>` parses as a *paragraph* containing
an `MdxJsxTextElement`, and emitting that tree literally produces
`<p><Button>…</Button></p>`. No MDX compiler produces that: `@mdx-js/mdx` runs a
pass (`remark-mark-and-unravel`) that replaces a paragraph whose children are
only JSX elements / MDX expressions and whitespace with those children, promoted
to flow. Checked against both ends of the version range rather than assumed —
`@mdx-js/loader@1.6.22` (what this app pins) emits
`<h1><p><button>` as siblings, and `@mdx-js/mdx@3` emits
`…<_components.p>…</_components.p>{"\n"}<Button>{"👋 Hello"}</Button>`. Every
MDX page that drops a component on its own line was getting the extra, invalid
wrapper.

**FIXED** — `src/mdx.rs::unravel`, applied at every depth (a paragraph inside a
list item or blockquote unravels too, like the `unist-util-visit` walk it
mirrors). Five regression tests in `src/mdx.rs` pin both directions: the
stand-alone component, several components on one line, a stand-alone MDX
expression, unravel reaching a nested blockquote, and — the load-bearing
negative — prose around an *inline* component keeping its paragraph.

```sh
node integration/e2e/run.mjs next-pages-mdx
```

`next-pages-mdx`: never compared -> **1 failing finding -> 0**, every channel
green, plus a driven interaction.

### 34. `createMDX`'s remark/rehype plugin options were read by nothing

Not found by a comparison — neither pinned MDX app configures a plugin, so the
suite could not have caught it. Found by inspection, and it is the silent-
degradation class this codebase treats as a defect on sight.

`@next/mdx` is configured like this in real apps:

```js
const withMDX = createMDX({
  options: { remarkPlugins: [remarkGfm], rehypePlugins: [rehypeSlug] },
});
```

`grep -an "remarkPlugins\|rehypePlugins" src/*.rs scripts/rsc/next-config-eval.mjs`
returned nothing. The config evaluator shimmed `@next/mdx` down to its
`pageExtensions` merge and threw the options away, so an app asking for GFM got
plain CommonMark: no tables, no strikethrough, no autolinks, no heading ids —
no build warning, no runtime error, just a page that renders differently from
what its author wrote.

**FIXED**, and by running the app's OWN pipeline rather than reimplementing
plugins in Rust. A unified plugin is an arbitrary JavaScript function over an
mdast/hast; the only faithful answer is to execute it.

*Reading them.* `scripts/rsc/next-config-eval.mjs` now captures the
`createMDX(pluginOptions)` object and reports an `mdx` block (each plugin's
identity and whether it was given options, plus `providerImportSource`,
`extension`, `experimental.mdxRs`, and any other option key). Capture covers
BOTH module systems — a `next.config.mjs` doing `import createMDX from
"@next/mdx"` never touches `Module._load`, and that path used to load the real
package un-shimmed and lose the options entirely.

*Reporting them.* Every Next build now prints what was configured and what
became of it, on both routers:

```
[next.config] @next/mdx: experimental.mdxRs; no remark/rehype/recma plugins — compiled by diffpack's native MDX compiler
[next.config] @next/mdx: remarkPlugins: [remarkGfm]; rehypePlugins: [rehypeSlug] — .mdx/.md files are compiled with the app's own @mdx-js/mdx pipeline so these run
```

*Running them.* When a config asks for anything the native emitter cannot do,
`src/mdx.rs` hands the file to `src/mdx_runner.mjs`, which re-evaluates
`next.config` (plugin values are live functions and cannot cross a JSON
boundary), compiles with the app's own `@mdx-js/mdx` at `jsx: true`, and returns
JSX into the normal oxc + RSC pipeline. The app's `mdx-components.*` is passed as
`providerImportSource`, standing in for `@next/mdx`'s
`next-mdx-import-source-file` webpack alias, so element overrides keep working.
Verified on a scratch app: GFM `<table>`/`<del>` and `rehype-slug`'s
`id="hello-world"` all reach the prerendered HTML, with an `h1` override applied.

*Refusing.* No `@mdx-js/mdx` installed, an ESM config the shim never saw, a
plugin that throws — each is a fatal build diagnostic naming the file and the
plugins. There is no fall-back to the plugin-free compiler.

Apps with no plugins configured (both pinned MDX fixtures) keep the native Rust
compiler and spawn no node process: `createMDX()` and
`createMDX({ extension: /\.mdx?$/ })` ask for nothing it cannot already do.

```sh
node integration/e2e/run.mjs next-mdx
node integration/e2e/run.mjs next-pages-mdx
```

`next-mdx`: pass -> **pass** (0 findings). `next-pages-mdx`: pass -> **pass**
(0 findings). Five regression tests in `src/mdx.rs` pin it: the options are
parsed and named; a plugin-free config still takes the native path; a configured
plugin reaches the app's compiler with `jsx: true` and the right provider; an
ESM config is captured too; and a missing app pipeline is a hard error naming
plugin and file.

> Superseded in part by item 35: `remark-gfm` on its own is no longer handed to
> the app's pipeline, because the native compiler now implements GFM. Every other
> plugin still routes exactly as described above.

### 35. No GitHub-Flavoured Markdown: tables, strikethrough, task lists, autolinks

Also not found by a comparison — neither pinned MDX app writes GFM, so the suite
could not have caught it. Found by direct probe against a scratch app:

```
input:  ~~struck~~                        rendered: literal "~~struck~~"
input:  | a | b |\n| - | - |\n| 1 | 2 |   rendered: no <table>, a paragraph of pipes
input:  - [ ] task                        rendered: a plain bullet
input:  www.example.com                   rendered: prose
```

Item 34 made a configured `remark-gfm` reach the app's own `@mdx-js/mdx`, which
answers the question of correctness but at the price of a node process per MDX
file — and only when the app has `@mdx-js/mdx` installed at all (it is an
*optional* peer of `@next/mdx`). GFM is what most people mean by "markdown", so
it is worth implementing rather than delegating.

**FIXED** — `src/mdx.rs` implements GFM natively, behind the app's own opt-in.

*The signal.* `MdxConfig::wants_gfm` is true only when `next.config` configures
`remark-gfm` (as a specifier, or by the plugin function's `remarkGfm`/`gfm`
name) with no options object. Nothing else turns GFM on. This matters in both
directions: `@next/mdx` does not enable GFM by default either, so an app that did
not ask for it must keep getting plain CommonMark, and rendering a table where
the author's own build renders pipes would be the same defect pointing the other
way. `[remarkGfm, {singleTilde: false}]` still defers to the app's pipeline —
only a plugin's *identity* survives the config eval, and `singleTilde` changes
what parses, so guessing would be a silent divergence.

*The implementation.* Six markdown-rs constructs (`gfm_table`,
`gfm_strikethrough`, `gfm_task_list_item`, `gfm_autolink_literal`,
`gfm_footnote_definition`, `gfm_label_start_footnote`) plus the emitter work they
imply: `<table>`/`<thead>`/`<tbody>` with per-column alignment and rows padded or
truncated to the header width, `<del>`, `className="contains-task-list"` +
`task-list-item` + a disabled checkbox pushed into the item's first paragraph,
autolink literals (including `www.` -> `http://www.`), and footnotes — a trailing
`<section data-footnotes>` in first-reference order, with one numbered `↩`
back-reference per reference. `Table`, `Delete`, `FootnoteDefinition` and
`FootnoteReference` used to be listed by name in the emitter's hard-error path;
they are now emitted.

*How the expected output was established.* Not from memory: a 45-case corpus was
compiled by BOTH diffpack and the real `@mdx-js/mdx@3` + `remark-gfm` at
`{jsx: true, outputFormat: "program"}` — the exact call `mdx_runner.mjs` makes —
and diffed. 44 of 45 are byte-identical; the 45th differs only in JSX-source
entity spelling (`&amp;` vs `&` inside an attribute string, which JSX decodes to
the same value). That oracle caught five things that would otherwise have shipped
wrong, three of them in code that predates this item:

- table alignment reaches the DOM as `style={{textAlign}}`, not the `align`
  attribute the hast carries — `hast-util-to-estree` rewrites it;
- a footnote's back-reference separator must be merged into the preceding text
  run, because two adjacent React text children are server-rendered with an
  `<!-- -->` marker between them;
- **tight lists were not unwrapping their paragraphs** — every bullet list was
  rendering `<li><p>text</p></li>` instead of `<li>text</li>`, with the
  block-level gaps that implies;
- **fenced code was losing the closing fence's newline**, which is significant
  inside a `<pre>`;
- **link/image URLs were not percent-normalized**, so `/café` reached the DOM as
  raw UTF-8 rather than `/caf%C3%A9`.

Verified end to end on a scratch copy of `next-mdx` with `remark-gfm` configured:
the build logs
`[next.config] @next/mdx: remark-gfm (native GFM) — compiled by diffpack's native MDX compiler`
and the prerendered HTML contains
`<th style="text-align:left">`, `<del>`, `class="contains-task-list"`,
`<input type="checkbox" ...>` and `<a href="http://www.example.com">`.

Eighteen regression tests in `src/mdx.rs` pin it, including the two negatives
that keep the opt-in honest: `commonmark_leaves_every_gfm_construct_alone` and
`remark_gfm_with_options_still_defers_to_the_apps_own_pipeline`.

```sh
node integration/e2e/run.mjs next-mdx
```

`next-mdx`: pass -> **pass** (0 findings). `next-pages-mdx` and
`next-blog-starter` re-run as insurance for the three shared-path fixes above:
both **pass**.

*What is deliberately still missing.* `mdast-util-to-hast` puts `{"\n"}`
whitespace text nodes between block children; this emitter has never emitted them
for any construct, GFM or otherwise, and that has always been invisible to all
eleven channels (`next-mdx`'s own two-block pages would fail otherwise). Left
alone rather than fixed only for the new constructs, which would be inconsistent.

> Superseded by item 36: the corpus could not see it because the corpus did not
> write MDX that exposes it. A fixture that does was added, the difference showed
> up in the text channel on the first run, and the separators are now emitted.

### 36. MDX: the corpus was green because it exercised almost nothing

Every MDX feature above (items 31, 34, 35) was verified by hand, on scratch apps,
and then guarded by a corpus that could not have caught a regression in any of
them. The two pinned MDX apps are:

- `next-mdx` — `app/message.mdx` is a heading and one sentence, and its
  `mdx-components.tsx` is `const components: MDXComponents = {}`. The file
  exists, is loaded, and overrides nothing, so the override path — the one thing
  `mdx-components` is for — was never observed.
- `next-pages-mdx` — one import, one heading, one component.

Between them: no GFM, no remark/rehype plugin, no frontmatter, no `export const`,
no MDX-hosted client component, no non-empty override. `2/2 pass` meant almost
nothing about MDX.

**No pinned third-party app covers it.** Vercel's examples are the only pinned
MDX apps in reach (the corpus's three sources are `vercel/next.js`,
`TanStack/router`, `vitejs/vite`), and these two are all of them. So this is
recorded as what it is and fixed with FIRST-PARTY fixtures, kept honest by being
built by BOTH toolchains — `next build` is still the oracle, so they are
differential tests, not self-assertions — and by being marked first-party
everywhere they appear: `"firstParty"` + a mandatory `firstPartyReason` in
`corpus.json`, `"origin": "first-party"` with an explicit `caveat` in each
materialized `DIFFPACK_E2E_PROVENANCE.json`, and a section in the suite README.

**Two fixtures, one per MDX compiler** (`integration/e2e/fixtures/`):

| fixture | covers | compiler exercised |
| --- | --- | --- |
| `next-mdx-features` (app router) | non-empty `mdx-components` overrides (`h1`, `table`, `del`, `a`), GFM (aligned table, strikethrough, task list, `www.` autolink, footnote), a component imported into MDX, a `"use client"` component inside an MDX route, `export const` read inside the file and by the importing module, `.mdx` used as a component AND as a route | diffpack's native Rust emitter (`remark-gfm` and nothing else) |
| `next-pages-mdx-plugins` (pages router) | YAML frontmatter stripped and exposed (`# {frontmatter.title}`), `remark-frontmatter` + `remark-mdx-frontmatter` + `rehype-slug` + `rehype-autolink-headings`, imports/JSX/`export const` in a pages-router MDX page | the app's own `@mdx-js/mdx` through `src/mdx_runner.mjs` |

Frontmatter is split off deliberately rather than for convenience: `next build`
has no opinion about frontmatter unless `remark-frontmatter` is configured (it
renders `---` as a thematic break), so frontmatter cannot be *compared* at all
without the plugin fixture — and configuring any plugin beyond a bare
`remark-gfm` routes the whole app to the app's own pipeline, which would have
left the native emitter untested again. Each fixture pins one branch, and
`lib/corpus-mdx.test.mjs` asserts that the native one keeps configuring nothing
but `remarkGfm`.

**Three defects fell out immediately.**

*(a) The suite could not build an MDX app that configures a plugin.* The very
first run reported `next-mdx-features` as `reference-build-failed` — the same
class as item 31, with wording that shares not one word with it:

```
Error: loader …/@next/mdx/mdx-js-loader.js for match "{*,next-mdx-rule}"
does not have serializable options.
```

Turbopack runs loaders out of process, so a loader rule may only carry JSON;
`createMDX({ options: { remarkPlugins: [remarkGfm] } })` puts live *functions*
there, which is how every MDX app configures a plugin. Such an app is buildable
only with `--webpack`, and `needsWebpackFlag` recognized only the other phrasing.
**FIXED** — `integration/e2e/lib/apps.mjs` (`TURBOPACK_UNSERIALIZABLE_LOADER`),
pinned by `lib/apps.test.mjs`
(`a loader whose options Turbopack cannot serialize is a --webpack retry too`),
which also asserts an unrelated loader failure is NOT retried as a builder
choice. Without this, an entire class of real MDX app silently reads as "the app
cannot be compared" instead of "diffpack was never asked".

*(b) MDX block separators were missing — the gap item 35 left open.* With the
fixture comparing, the text channel reported:

```
reference: … badge inside an MDX route clicked 0 times …
diffpack:  … badge inside an MDX routeclicked 0 times …
```

`mdast-util-to-hast`'s `wrap()` puts a `{"\n"}` text child between block
children, and `hast-util-to-estree` emits it as an explicit expression child (a
literal newline would be stripped by JSX). diffpack emitted none. Where the
neighbours are block-level a browser collapses the difference, which is why it
survived item 35 — but two flow-level *inline* elements (`<Badge/>` then
`<Counter/>`, ordinary MDX) run together into one word.

**FIXED** — `src/mdx.rs`: `EOL` between root children, the loose wrap for
blockquotes (`wrap_loose`) and lists, `mdast-util-to-hast`'s exact `listItem`
rule (`eol_before_list_item_child`: no leading newline for a tight item's
unwrapped first paragraph, trailing newline unless the last child is one), and
the footnote section's own newlines. A node that renders nothing where it is
written (a footnote *definition*) is dropped rather than separated — emitting a
separator for it put a stray `{"\n"}` in front of every footnotes section.

Established against the real thing, not from memory: a 47-case corpus was
compiled by BOTH diffpack and `@mdx-js/mdx@3` + `remark-gfm` at
`{jsx: true, outputFormat: "program"}` — the exact call `mdx_runner.mjs` makes.
**44 of 47 are now byte-identical**; the three that differ are the same JSX
spelled differently (`{ 1 + 1 }` vs `{1 + 1}`, a preserved `{/* comment */}` vs
`{}`, `<li></li>` vs `<li />`). Before the fix it was 38 of 47, every miss a
missing separator. Five new tests in `src/mdx.rs` pin it, including the negative
(`a_table_carries_no_separators_at_all`) and the dropped-definition case; seven
existing tests were updated from the old separator-free output.

*(c) `export const metadata` from an MDX route is not buildable by Next itself.*
Not a diffpack defect, recorded because it looks like one: with `@next/mdx` the
MDX module resolves its provider through `next-mdx-import-source-file`, and Next
then refuses with "You are attempting to export `metadata` from a component
marked with `use client`". The fixture therefore does not do it, and says why.

```sh
cargo build --release && cargo test --release --lib
cargo clippy --release --all-targets -- -D warnings
node --test integration/e2e/lib/*.test.mjs
node integration/e2e/run.mjs next-mdx next-pages-mdx next-mdx-features next-pages-mdx-plugins
```

`next-mdx-features`: reference-build-failed → 1 failing finding →
**pass (0 findings)**, with a driven interaction and a client-side navigation.
`next-pages-mdx-plugins`: new → **pass (0 findings)**. `next-mdx`: pass →
**pass**. `next-pages-mdx`: pass → **pass**.

**What is still not covered, stated plainly.** Frontmatter through diffpack's
NATIVE compiler is not compared by anything. diffpack strips YAML frontmatter and
turns `title`/`description` into `export const metadata`; `next build` without
`remark-frontmatter` renders the same source as a thematic break plus prose. That
is a real difference between diffpack and an app's own toolchain, it is
deliberate (and the behaviour most authors expect), and no app in the corpus can
observe it because any app that adds the plugin moves to the other compiler.
Recorded here rather than papered over.

### 37. `next-pages-mdx`'s reference build — re-verified, not a gap

Recorded because the opposite was believed. `next-pages-mdx` was for a while the
one app the suite never compared: `next build` failed with
`Error: Call retries were exceeded { type: 'WorkerError' }` under Turbopack, so
there was no oracle, and the app sat in the corpus looking like an unexplained
diffpack gap. Item 31 fixed the harness, and this round re-confirmed it against
the tree as it stands:

```sh
node integration/e2e/run.mjs next-pages-mdx
```

The reference builds **as published** — original `@next/mdx@^9.1.1` /
`@mdx-js/loader@^1.5.1` / `@mdx-js/react@^1.6.18` pins, no `corpus.json` pin
change, no source edit — on the second attempt, with `--webpack`, which the
harness now derives from Next's own diagnosis
(`results/next-pages-mdx/build-reference.log` holds both attempts). It then
compares clean: `results/next-pages-mdx/findings.json` is `[]`. It is a working
oracle, it stays in the corpus, and nothing about it needed removing.

### Further defects raised by the adversarial verifiers

Each fix was re-verified by an independent agent whose job was to refute it. All
four survived (gates green, tests confirmed load-bearing by reverting the
production change), but the verifiers surfaced these, now queued:

- **Node built-ins in a *browser* build still degrade silently.**
  `resolve_dependencies` takes no `Target`, so `is_external_specifier` treats a
  node built-in as external even for the browser and emits a runtime stub —
  the same bug class item 4 just closed for unresolved imports.

  **FIXED**, in the two halves the report asked for.

  *Classification.* `resolve_dependencies` now takes the `Target`. On
  `Target::Client` a Node built-in is a new fatal `DiagnosticKind::NodeBuiltinInBrowser`
  naming the built-in and the file that imported it; on a server target it stays
  external and silent, as before. The message states the gap outright: diffpack
  does not implement webpack/Next-style browser polyfills for Node built-ins.

  *The message.* The browser `requireNative` fallback used to call *everything*
  it could not resolve a "node builtin" and hand back a lazy throw-on-use Proxy.
  That is how `next-pages-framer-motion` died: framer-motion loads its optional
  `@emotion/is-prop-valid` as `require("@emotion/is-prop-" + "valid")` inside a
  `try/catch`, the minifier folds the concatenation, the Proxy is returned
  instead of throwing, the `catch` never runs, the stub is installed as
  `isPropValid`, and the *render* blows up with a false claim about a Node
  built-in. `requireNative` now classifies against the same builtin list the Rust
  side uses: a genuine built-in keeps the load-safe stub, anything else throws
  immediately — which is what Node does and what the universal
  `try { require(optional) } catch {}` idiom is written against. framer-motion's
  `/` now matches Next (one unrelated client-navigation difference remains).

  Cost, stated plainly: `next-pages-shallow-routing` imports `format` from `url`
  in a *client* page. Next polyfills that for the browser; diffpack does not, and
  now says so and fails the build instead of rendering the page with the
  formatted URL missing (which is what it was doing). Browser polyfills for Node
  built-ins remain unimplemented.
- **`vite.config`'s `build.outDir` is never read.** `WebConfig` has no such
  field, so an app configuring `build: { outDir: "build" }` silently gets
  `dist/`.

  **FIXED.** The evaluator emits `build.outDir` and `build.assetsDir`,
  `ResolvedViteConfig` and `WebConfig` carry them, and `web_build` resolves
  `outDir` against the project root exactly as Vite does. Precedence, highest
  first: `--out-dir`, then `build.outDir`, then Vite's `dist` default — an
  explicit command-line argument still wins. `build.assetsDir` is *not*
  implemented (the emitters hardcode `assets/`); the default is accepted and any
  other value is a named hard error rather than a build whose every asset URL is
  wrong.
- **`bundle_benchmark.rs` still uses `!diagnostics.is_empty()`**, so a benign
  non-fatal warning would fail a benchmark build.

  **FIXED.** All three sites now go through `partition_diagnostics`, so only a
  fatal diagnostic fails the benchmark; warnings are printed.
- **The new gate prelude's `ERR` trap fires inside a deliberate `set +e`
  window**, printing a spurious `FAIL:` line on a *passing* gate — and reports
  `line + 1` on bash 3.2, which is what macOS ships and what the prelude
  explicitly targets.

  **FIXED** — see item 4a. Re-verified independently here on bash 3.2.57:
  `bash scripts/rsc/next-missing-dep-check.sh` exits 0 with no `FAIL:` line, and
  the self-test's two cases are load-bearing — swapping the real prelude for the
  naive one-liner makes case 7 fail (`stderr did not match /^$/; got: FAIL:
  …case.sh:6 aborted (exit 1) running: false`), and fixing only the `case $-`
  half makes case 8 fail (`must name line 4 (the grep); got: …lineno.sh:7`, the
  +3 offset being the `$LINENO` reference's own index in the trap string).
- **The SSG gate's new set invariant has a hole**: it asserts plan→manifest
  coverage but not manifest→plan provenance; the verifier planted extra
  prerendered pages in both manifest and disk and the gate stayed green.
- **`scriptsInsideTags` only detects `<script` between `<` and `>`** — an
  injection landing inside a `<script>`, `<style>`, or comment body would not
  be flagged.
- **`NEXT_CONFIG_EXTS` precedence** puts `cjs` before `ts`; Next's own order is
  `js, mjs, ts, cjs, mts, cts`.
- **`PAGE_EXTS` routes `.md`/`.mdx` unconditionally** instead of honouring the
  `pageExtensions` config that enables them, so a stray `pages/NOTES.md` becomes
  a route.
- **This suite does not yet exercise the streaming SSR path.** The verifier
  noticed that `next-hello-world`'s captured document contains zero
  `__DF_FLIGHT` scripts — it is prerendered. The corpus needs an app-router app
  with a genuinely dynamic route before it can claim to cover item 5.

  **CLOSED.** `next-strict-csp` is that app (`await headers()` opts its page out
  of prerendering) and it now passes every channel, so the run reports
  `streaming SSR exercised on 1 route(s): next-strict-csp/`. Its served document
  carries three `__DF_FLIGHT` script pushes, checked against the raw-document
  integrity channel.

## Differences that turned out NOT to be diffpack defects

Recorded because a differential suite is only trustworthy if its false
positives are written down too.

- **`next-with-redux`'s running CSS animation.** Its logo carries
  `animation: logo-float infinite 3s`, so the style channel sampled `transform`
  mid-flight and it drifted every run on *both* sides
  (`matrix(1,0,0,1,0,2.63664)` vs `2.16572`; different numbers the next run).
  The determinism shim now pauses CSS animations at a fixed negative delay, so
  both deployments sit at the same phase. This does not weaken the channel: an
  animation diffpack failed to emit still differs, because the reference sits at
  the 1ms phase while the unstyled side sits at the identity transform.
  Verified on `next-pages-framer-motion` too — JS-driven animation never goes
  through CSS animation and is untouched. The app now passes.
- **Asset content hashes.** Vite emits `hero-CLDdwZDr.png` (base64url), diffpack
  emits a hex hash. The probe's URL normalizer only stripped hex, so every
  `<img src>` looked different. Hashes are now dropped entirely on both sides.
- **`next-with-zustand`'s server clock.** Its only remaining difference was
  `08:18:27` vs `08:18:28` — the page renders the *server's* time, and the
  determinism shim only reaches the browser. Now declared as `volatile` in
  `corpus.json` with a written reason; the app passes cleanly.
- **`next-with-redux`'s running CSS animation.** Its logo carries
  `animation: logo-float infinite 3s`, and the style channel sampled it
  mid-flight, so `transform` drifted every run on *both* sides
  (`matrix(1,0,0,1,0,2.63664)` vs `2.16572`, different numbers next run). The
  determinism shim now pauses CSS animations at a fixed negative delay, putting
  both deployments at the same phase. It does not weaken the channel: an
  animation diffpack failed to emit still differs, because the reference sits at
  the 1ms phase while the unstyled side sits at the identity transform. Verified
  against `next-pages-framer-motion` too — JS-driven animation is untouched.
- **`tanstack-start-counter`'s persisted counter.** Its server function writes
  `count.txt` in the project root, so whichever side was driven second started
  from the first side's value and the interaction channel flagged it. Apps with
  server-side persistence now declare `resetFiles` in `corpus.json`, and the
  runner resets that state before each side is driven.
- **`/blog/post` being prerendered** — see item 7.
- **`next-github-pages`'s "injected styles" (item 20).** Filed as "diffpack
  injects styles into an app that ships none": the reference rendered in the
  browser default (`font-family: Times`) while diffpack rendered in
  `Inter, ui-sans-serif, system-ui, …`, and the finding said a bundler must not
  add styling the application never asked for. **The premise was false and the
  oracle was at fault.** The app *does* ship a stylesheet — `app/layout.tsx`
  calls `Inter({ subsets: ['latin'] })` and applies `inter.className` to
  `<body>`, and `next build` emits a CSS chunk for it. The reference never
  loaded it: the app sets `output: 'export'` **and** `basePath:
  '/gh-pages-test'`, which bakes that prefix into every emitted asset URL, while
  the suite served `out/` at `/`. Every `/gh-pages-test/_next/…` request missed,
  and `startStaticServer`'s last-resort `index.html` fallback answered those
  misses with an HTML document — so the reference page loaded no CSS and no JS
  at all (the 8 `SyntaxError: Unexpected token '<'` page errors recorded
  alongside it were that, and the dead client-side navigation was too). Acting
  on the finding as written would have deleted a feature the app asks for.
  Fixed in `integration/e2e/lib/apps.mjs`: a static export is served under its
  own `basePath`, and the `index.html` fallback no longer applies to a path with
  a file extension, so a missing asset 404s instead of silently becoming HTML.
  `next-github-pages` went from 3 failing findings (styles, layout, navigation)
  to **0** and passes every channel. Pinned by `integration/e2e/lib/apps.test.mjs`.
- **`next-strict-csp`'s "unhydrated" page (was the app's only failing channel).**
  The `hydration` channel reported "React hydrated on the reference build but not
  on the diffpack build". React had hydrated both. `probe.mjs` answered "did React
  hydrate?" by scanning `document.body.querySelectorAll("*")` — the body's element
  *descendants* — which is a different question, and this page is where the two
  come apart: its component tree is a single `next/script
  strategy="afterInteractive"`, which renders no DOM. Checked in a real browser
  with both servers up, `<html>` and `<body>` carry `__reactFiber$…` /
  `__reactProps$…` on **both** builds; the only fibre-bearing descendant on the
  reference side was React's own streaming slot (`<div hidden>`), a node
  `isScaffolding` excludes from every other channel — so the suite was comparing
  framework scaffolding and calling it hydration. The predicate now also
  considers `document.documentElement` and `document.body` themselves
  (`probe.mjs`, `REACT_FIBER_SOURCE`). It does not make an unhydrated page look
  hydrated: nothing but React writes those keys, and two negative controls built
  from that same diffpack bundle — the served document with its `<script
  type="module" src="/client.js">` removed, and the whole bundle shipped with
  `client.js` throwing on its first statement — both still report false in a real
  browser. Pinned by `integration/e2e/lib/probe.test.mjs`, which evaluates the
  shipped predicate source over synthetic documents including that negative case.

## Test-harness defects found along the way

### 11. `scripts/rsc/next-check.sh` fails silently

Under `set -euo pipefail`, `hero_img="$(echo "$html" | grep -oiE '…' | head -1)"`
aborts the whole script with no output when the grep does not match. The gate
reports failure with no reason at all.

### 12. `integration/app-parity` cannot be run

`apps/wall-go.mjs` (and its siblings) point at
`/tmp/claude-1000/-home-jimmyhmiller-…/scratchpad/oss-triage/wall-go` — a
scratch directory from a previous session on a different machine. The five-app
behavioral parity gate that `docs/OSS_VALIDATION.md` cites as evidence cannot
be reproduced.

### 13. `scripts/rsc/next-real-check.sh` was never exercised

It skips whenever `integration/next-real/*/node_modules` is absent, which is
the committed state.

### 32. A route the suite could not observe at all was scored as a pass

The suite has three severities. `fail` is an observed difference. `info` is a
difference in diffpack's FAVOUR — an error only the reference produces — and is
recorded, not charged. `error` means the comparison could not be made at all;
today it has exactly one producer, `compare.mjs`'s "probe missing", raised when
one side never produced a record for the route.

`run.mjs` counted `fail` alone, in all five places it counts: the per-route
line, the printed differences, the `pass`/`differs` verdict, the summary table's
failing count, and its channel list. `error` was charged to nobody. So an app
whose routes diffpack could not serve at all printed `OK` per route, `pass` in
the table, and was counted in the headline "N/N compared app(s) behave
identically to their own toolchain".

That is how item 21 stayed invisible: `next-i18n-routing` has `"record": null`
for **both** of its routes and was scored a clean pass — and on the strength of
that it was listed under "What genuinely works" below. A false pass costs more
than a false difference, because nobody goes looking for it.

**FIXED.** `isFailure` is one exported predicate in
`integration/e2e/lib/compare.mjs`, used at all five sites, and it charges
`error` as well as `fail`. Pinned by the new
`integration/e2e/lib/compare.test.mjs`, which also pins the negative: an
`info`-severity finding stays uncharged, because charging it would invert the
suite's meaning.

Consequence, stated rather than left to be discovered: `next-i18n-routing`'s
last recorded run is `differs`, not `pass`, and this changes the scoreboard
below. It was not re-measured here.

### 33. `cargo test --lib` inherited the terminal's colour setting

Not an e2e-harness defect, but the same class, and it cost a red gate before it
was found. Roughly 30 tests in `src/bundler.rs` execute an emitted chunk under
`node` and compare its stdout byte-for-byte. With `FORCE_COLOR` in the
environment — plenty of terminal wrappers and CI runners set it — node writes
ANSI escapes even down a pipe, and `console.log(6)` arrives as
`\x1b[33m6\x1b[39m`. Four tests failed, in a way that looked like a bundler
regression and had nothing to do with the bundler.

**FIXED**: every one of those spawns now goes through a `node_command()` helper
that removes `FORCE_COLOR` from the child environment. Not `NO_COLOR` — node
ignores `NO_COLOR` when `FORCE_COLOR` is set and warns about it on stderr, which
would have broken stderr assertions instead. Pinned by
`src/bundler.rs::node_is_spawned_without_inherited_terminal_colour`, which first
demonstrates the hazard is real (node under `FORCE_COLOR` does emit escapes)
before asserting the helper removes it.

## Round-2 fixes that did NOT survive verification

Each fix was re-checked by an independent agent whose instruction was to refute
it. Three did not survive, and are recorded here rather than counted as done.

### `src/app` island discovery — REFUTED (introduces a wider defect)

The headline symptom is genuinely fixed and both new tests fail on a targeted
revert. But widening the island scan root from `app/` to the whole project
multiplied an existing divergence: `has_css` (`src/next_adapter.rs:1890`) is a
raw substring scan for `.css"` / `.css'`, and has no relationship to what the
react-server build actually emits (`src/main.rs:497` copies the stylesheet only
`if css.is_file()`, a silent skip).

Proven with an app containing **no stylesheet at all**, whose only occurrence of
the string is `export const THEME = "theme.css";` in a script: the served
document emits `<link rel="stylesheet" href="/rsc.css">` and `GET /rsc.css`
returns **404**. Reverting only the scan-root line removes it.

Second regression: every `"use client"` file in the tree is now a hard build
dependency, so an unreachable one with an unresolvable import fails the whole
build where it previously succeeded. No allowance for dead code, `examples/`, or
`__tests__`.

### `__toESM` CommonJS interop — REFUTED (new silent wrong value)

The crash is genuinely fixed, the conformance case genuinely flips, and the
verifier built an independent oracle confirming the new rule matches Vite 8's
own runtime on both client and server. But:

```js
import { missingName } from "<CJS module carrying the __esModule marker>"
```

now yields `undefined` and exit 0, where node, rolldown and esbuild all raise a
hard `SyntaxError`. **No fixture covers the missing-named-export-from-CJS case**,
so the suite cannot see it. Latent alongside it: `__isESM` does not recognise
`__toESM`'s own output, so a second application would double-wrap where the old
code was idempotent.

#### REPAIRED (the interop rule itself is unchanged)

`default` is still `module.exports` and the ESM/CJS decision still runs on
`__esmNamespaces`, never on `__esModule`. What changed is everything the wrapper
did *around* that rule (`src/bundler.rs`, the emitted runtime):

* **The named-export check no longer exempts CommonJS.** The wrapper used to be
  stamped `__diffpackCJS` and `__import` returned `namespace[name]` for anything
  carrying that stamp — that is where the `undefined` came from. The stamp is
  gone; `__import` consults the wrapper's own keys, then reads through to the
  live `module.exports`, and throws `SyntaxError` when the name is on neither.
  The message now names the module and the export the way Node's does:
  `The requested module "./marked.cjs" does not provide an export named "missingName"`.
  Verified: `conformance/fixtures/cjs-missing-named-throws` — Node exits 1 with
  no output, diffpack now matches it, and **rolldown and esbuild are the ones
  that are `WRONG` on this fixture** (both print `reached:undefined`, exit 0).
  The FINDINGS text above overstated their behaviour; only Node hard-errors.
* **Reading through to the live `module.exports` is what makes that safe.** The
  wrapper's keys are copied at wrap time, and in an ESM<->CJS cycle that is a
  *partially populated* object, so a purely snapshot-based strict check would
  have converted the second (unproven) suspicion into hard build breakage. A key
  the module assigns after the wrap is now visible: `import { late } from
  "./legacy.cjs"` in a cycle reads `late` (pre-repair `undefined`; rolldown
  agrees with the new behaviour, esbuild still prints `undefined`).
* **The latent double-wrap was real, and observable.** `export * as ns from
  "./legacy.cjs"` re-runs `__toESM` on *every read of `ns`*, so it minted a fresh
  namespace each time: `ns.legacy === ns.legacy` was **false**, against `true`
  from Node, rolldown and esbuild. The wrapper is now cached per
  `module.exports` (`__cjsNamespaces`) and recognised by `__isESM`
  (`__cjsInterops`), so one CommonJS module has exactly one namespace and
  re-wrapping is a no-op. Pinned by
  `conformance/fixtures/cjs-namespace-identity`.

One thing the strict check would have broken, caught before it shipped and fixed
in the same pass: the browser build's node-builtin stub
(`src/bundler.rs`, `requireNative` for `ModuleFormat::BrowserEsm`) is a `Proxy`
whose whole contract is "property reads succeed so dead server code still LOADS;
the moment it CALLS in, throw `node builtin X is not available in the browser`".
A `Proxy` with only a `get` trap answers `hasOwnProperty` from its target, so
`import { readFileSync } from "node:fs"` in a client graph started throwing
`does not provide an export named "readFileSync"` instead of handing back the
stub — absence proves nothing about an object whose shape is unknowable. The
stub now carries `getOwnPropertyDescriptor` and `has` traps that answer for any
name it would `get` (deferring to the target, so the function's own
non-configurable `prototype` still reports honestly and `Object.keys` stays
empty), and the `then`/`Symbol.toPrimitive`/iterator probes stay absent.
Pinned by `a_named_import_of_a_node_builtin_in_a_browser_build_stubs_instead_of_throwing`.

Still a snapshot, deliberately, and this is the precise boundary: the *enumerable
keys* of a CommonJS namespace (`Object.keys(ns)`, `import * as ns` member reads,
`export * from "./legacy.cjs"`) are refreshed on every `__toESM` of that
`module.exports`, but not after the last one. A key first assigned after that
point is readable through a named import and invisible to enumeration. Making it
fully live needs a `Proxy` per CommonJS namespace, and `__import` is on the
hottest path in the emitted code (every reference to an imported binding), so
that trade was declined. esbuild behaves identically here; rolldown is live.

Gate after the repair: `cargo test --release --lib` 537/537, clippy clean,
conformance **45 pass of 52** (was 43 of 50 — two new fixtures, both passing, no
change to the pre-existing failure set), `next-radix-ui` builds, prerenders,
serves and hydrates as before.

### `jsxImportSource` — REFUTED (incomplete, undisclosed)

It fixes the case it was measured on (`vite-preact-ts` passes end to end), but
only for **TypeScript-extension files under a tsconfig that claims them**.
Still silently on `react/jsx-runtime`:

- `.jsx` / `.js` files — `find_tsconfig` only returns a config that *claims* the
  file, and TypeScript's `include` does not claim `.jsx` without `allowJs`
- `jsconfig.json`, which is never read at all
- create-vite's **JS-flavour** preact template, which still fails with the
  original `cannot resolve "react/jsx-runtime"`

The same ownership rule splits a single Next app: `create-next-app`'s tsconfig
includes only `**/*.ts`/`**/*.tsx`, so after the JSX-in-`.js` fix its `.tsx`
modules get the configured import source and its `.js` modules silently get
React. One of the added tests also passes with the production change stubbed
out, so it pins nothing.

#### REPAIRED (the precedence is unchanged; the ownership question was wrong)

`vite.config` still beats the project config, which still beats nothing, and a
file-level pragma still beats both. What changed is WHICH config configures a
file, and a second source of the answer that was missing entirely.

* **Applicability, not type-checking** (`src/jsx_project_config.rs`, new; used by
  `bundler.rs::jsx_config_for`). `find_tsconfig` answers *would `tsc` compile this
  file as part of this project?*, and that question excludes exactly the files
  this one is about. The new rule: the nearest `tsconfig.json` **or
  `jsconfig.json`** whose `files`/`include`/`exclude` would cover the file *if its
  extension were any member of the JS/TS family* (`.ts .tsx .mts .cts .js .jsx
  .mjs .cjs`, plus `.md`/`.mdx`, which compile to JSX). `include: ["src"]` now
  covers `src/app.jsx`; `include: ["**/*.ts","**/*.tsx"]` — `create-next-app`'s own
  — now covers the app's `.js` and `.mdx` modules, so one app has ONE runtime.
  `extends`, solution-style `references` and `node_modules` exclusion are still
  the resolver's, loaded through it unchanged; a config that cannot apply is
  skipped and the walk continues to an ancestor that can. A malformed config is a
  hard error naming both the config and the file it would have configured.
* **`jsconfig.json` is read.** It is the only place a JavaScript project can write
  `jsx`/`jsxImportSource` at all, and both Next and the TS language service read
  it. `tsconfig.json` wins in the same directory.
* **A Vite PLUGIN can set the runtime, and now does.** create-vite's JS preact
  template (`template-preact`) has no tsconfig and no jsconfig: its
  `jsxImportSource` comes from `@preact/preset-vite`'s `config()` hook. The
  evaluator (`src/vite_config_evaluator.mjs`) now runs plugin `config()` hooks the
  way Vite's `runConfigHook` does — `enforce` order, `apply` respected, results
  merged over the config with the plugin as the override — and merges them
  **deeply**, because `@vitejs/plugin-react` sets the runtime from one plugin and
  Fast Refresh from a second and a shallow merge drops the first. Only the JSX
  keys (`oxc`/`esbuild`) are merged; see the boundary below. A hook that throws is
  reported by name (plugin + config file) and does not take the rest of the config
  with it.
* **The test that pinned nothing now fails when the production change is stubbed.**
  `a_jsx_import_source_pragma_beats_the_configured_source` asserts the pragma wins
  AND that a sibling module with no pragma takes the configured source — the
  second half is what `jsx_config.apply` is responsible for.

New corpus app: **`vite-preact`** — create-vite's JS-flavour preact template, the
JS twin of `vite-preact-ts`. It failed with the original
`cannot resolve "react/jsx-runtime"` for `src/app.jsx` and `src/main.jsx`; it now
builds, serves and behaves identically to Vite (`/` plus an interaction).

Deliberate boundary, stated rather than discovered later: only `oxc`/`esbuild` are
taken from plugin `config()` hooks. Everything else diffpack reads from a Vite
config (`define`, `resolve.alias`, `build.rollupOptions.input`, `server.proxy`) is
a USER surface; a plugin's contribution to those describes machinery (virtual
modules, dev middleware, SSR entry rewriting) a native build does not run.

Gate after the repair: `cargo build --release` clean, `cargo test --release --lib`
552/552, `cargo clippy --release --all-targets -D warnings` clean, and the vite +
tanstack corpus 8/8 comparable apps pass (`vite-preact-ts`, `vite-react-ts`,
`vite-preact`, `vite-vanilla-ts`, all four tanstack apps; `.vue`/`.svelte` remain
the declared boundary).

### On the test suite's stability

A verifier reported `cargo test --release --lib` as flaky-red (2–9 varying
failures). On a quiescent tree I could not reproduce it: **18 consecutive clean
runs, 520/520**. Every observed failure — mine and theirs — happened while
another agent was concurrently editing and rebuilding the same tree, and one
verifier explicitly traced its own mid-session failures to that. Treated as
measurement interference, worth one confirmation run in CI.

One later red run had a different and reproducible cause, now fixed and pinned:
`FORCE_COLOR` inherited from the shell made node colour the stdout that ~30
bundler tests compare byte-for-byte. See item 33.

## Scoreboard

| | at the start | after the two fix rounds | after the harness round | now |
| --- | --- | --- | --- | --- |
| apps behaving identically to their own toolchain | 7 | 17 | 20 | **36** |
| apps that still differ | — | — | — | **1** (`next-i18n-routing`, item 21) |
| apps diffpack could not build at all | 17 | 2 | **2** (`.vue`, `.svelte`) → **0** after finding 25 compiled both | **0** |
| apps that could never be compared (reference unbuildable) | — | 1 | 1 | **0** |
| corpus size | 9 | 35 | 35 | 37 |
| of which FIRST-PARTY (written here, still built by both toolchains) | 0 | 0 | 0 | **2** (item 36) |

**How the last column is measured, exactly.** It is not a fresh whole-corpus run
— that takes ~40 minutes and is not run casually. It is the per-app evidence
already on disk: `results/<id>/findings.json` is written only after a comparison
actually happened, so its presence means the app built on both sides, served on
both sides, and was driven in the browser; its contents are that run's findings.
All 37 apps have one. Counting them under the corrected scoring of item 32:
**36 with zero charged findings, 1 (`next-i18n-routing`) with two unobservable
routes.** Two of the 36 are the first-party MDX fixtures of item 36 — still
compared against their own `next build`, but not third-party evidence, so they
are counted separately in the row above and are not claimed as "real
third-party applications diffpack builds". Two apps additionally carry uncharged `info` findings (an error the
reference produces and diffpack does not) — those are recorded, not held
against diffpack.

Read that as "each app's last recorded measurement", not "one simultaneous
measurement of all 37". The runs span the fix rounds; the four apps measured
against the tree as it stands now are `next-mdx`, `next-pages-mdx`,
`next-mdx-features` and `next-pages-mdx-plugins` (item 36). A whole-corpus run is
what would turn the column into a single observation, and it is the obvious next
step — item 36 changed `src/mdx.rs`, and while only `.md`/`.mdx` sources reach
that code (no other app in the corpus has one), that is an argument, not a
measurement.

The `20` column was `17 + preact + github-pages + strict-csp`. `preact` joined
with the `jsxImportSource` repair; the other two, and **neither was a diffpack
defect** — `next-github-pages` was the suite serving a `basePath`'d static
export at `/`, `next-strict-csp` was the suite asking "did the app render
elements into `<body>`?" and reporting the answer as "did React hydrate?". Both
are written up under "Differences that turned out NOT to be diffpack defects",
not counted as fixes to the bundler.

Note what moved in the OTHER direction this round. `next-i18n-routing` is a new
entry in the "still differ" row, and no regression put it there: it was always
broken, and the suite was scoring an unobservable route as a pass (item 32).
The count went up because the measurement got honest.

Read the first row with the section above in mind: three of the six round-2
fixes were refuted, and two of them (`src/app` islands, `__toESM`) are part of
why apps now build. Those apps really do build and render correctly — the
refutations are about *new* defects the fixes introduced elsewhere, not about
the wins being false. The honest position is that the corpus moved a long way
and three fixes need another pass before they are safe to keep.

The two remaining build failures were the Vue and Svelte templates. That row is
now **0**: finding 25 compiles both with the app's own installed compiler
(`@vue/compiler-sfc` / `svelte/compiler`), and `vite-vue-ts` and `vite-svelte-ts`
pass every channel. Every app in the corpus builds.

The one app that still differs is item 21, with its evidence and a reproduction.
Two apps additionally carry `info`-severity findings — an error the reference
produces and diffpack does not (`next-with-dynamic-import`, `next-i18n-routing`).
Those are recorded and deliberately not charged; item 32 says why.

Two apps reported `serve-failed` in the full parallel run and pass when run
alone (`next-with-web-worker` passes outright; `next-radix-ui` builds, serves
and is measurable). That was the harness starving a cold server under
`--jobs 3`, not a bundler defect — the readiness budget has been raised so a
slow start is never reported as a failure.

## What genuinely works

Real third-party applications that diffpack builds, serves, and runs with **no
observable difference** from their own toolchain across every channel — text,
structure, attributes, computed styles, layout, assets, links, hydration,
interaction, client-side navigation, raw-document integrity, and the browser's
error channels:

**Next.js app router** (19) — `hello-world`, `basic-css`, `active-class-name`,
`blog-starter` (src/app, Tailwind v3, `next/font`), `image-component`, `mdx`,
`radix-ui`, `with-redux`, `with-zustand` (src/app), `with-xstate`, `with-jotai`,
`with-context-api`, `with-sass`, `with-styled-components` (SSR style registry),
`with-dynamic-import`, `with-web-worker`, `with-absolute-imports`,
`github-pages` (`output: 'export'` under a `basePath`, served as a real static
export), `strict-csp` (the only genuinely DYNAMIC app here — `await headers()`
opts its page out of prerendering, so this is the one that exercises streaming
SSR, inline flight scripts, middleware, and a nonce'd strict CSP end to end)

**Next.js pages router** (5) — `pages-typescript`, `pages-mdx` (MDX pages via
`@next/mdx` + `pageExtensions`; the app that had no oracle at all until item
31), `pages-shallow-routing`, `pages-i18n-routing` (built-in sub-path locale
routing), `pages-framer-motion`

**TanStack Start** (3) — `start-basic`, `start-counter` (SSR + a server function
driven through a real click), `start-tailwind-v4`

**Vite** (7) — `tanstack-router-quickstart` (two routes, built through the Vite
path), `react-ts`, `vanilla-ts`, `preact-ts`, `preact` (both preact templates: a
non-React JSX runtime, named by a tsconfig in the TS one and by a Vite plugin
ALONE in the JS one), `vue-ts`, `svelte-ts` (both compiled with the app's own
installed SFC compiler)

That is 34 of the 35 THIRD-PARTY apps in the corpus. The one that is not here is
`next-i18n-routing` (item 21), and it used to be — it was listed as working on
the strength of a run the suite mis-scored, which is what item 32 fixes.

The corpus also holds two first-party fixtures (`next-mdx-features`,
`next-pages-mdx-plugins`, item 36), which pass every channel against their own
`next build` but are deliberately not listed above: they were written here, so
they are evidence about diffpack's MDX compilers, not about applications nobody
here wrote.

### MDX, specifically — what is verified and by what

MDX support (`src/mdx.rs`, hooked at the generic transform choke point in
`src/transform.rs`, so both routers get it) had accumulated as folklore across
items 31/34/35: verified by hand on scratch apps, guarded by a corpus that
exercised almost none of it (item 36). This is the written-down version. Every
row is observed by a differential run — diffpack's output compared against the
same app's `next build` — not by a diffpack-authored expectation.

| behaviour | verified by |
| --- | --- |
| `.mdx` as a **route** on the app router | `next-mdx-features` `/docs`, `next-mdx` |
| `.mdx` as a **route** on the pages router | `next-pages-mdx`, `next-pages-mdx-plugins` `/` |
| `.mdx` imported as a plain **component** (not a route) | `next-mdx-features` `/` (`app/intro.mdx`), `next-mdx` |
| `import` inside MDX, and the imported component used as **JSX** | all four MDX apps |
| a `"use client"` component inside an MDX route (client boundary from MDX) | `next-mdx-features` — driven, one click, hydrated on both sides |
| `export const` from MDX, read inside the file and by the importing module | `next-mdx-features` (`revision`, `audience`) |
| `mdx-components` **overrides** (`useMDXComponents`), non-empty | `next-mdx-features` — `h1`, `table` (a wrapper element, not just a class), `del`, `a`, each observed through `data-testid` |
| **GFM**: aligned table, strikethrough, task list, autolink literal, footnotes | `next-mdx-features` `/docs` (native Rust compiler, opted in by `remark-gfm`) |
| GFM **off** by default (an app that did not configure `remark-gfm` gets CommonMark) | `next-mdx`, `next-pages-mdx` + `commonmark_leaves_every_gfm_construct_alone` |
| block **separators** (`{"\n"}` between block children) | item 36(b): 44/47 cases byte-identical to `@mdx-js/mdx@3`, plus the text channel on `next-mdx-features` |
| **YAML frontmatter** stripped and exposed to the page | `next-pages-mdx-plugins` (`remark-frontmatter` + `remark-mdx-frontmatter`) |
| remark/rehype **plugins** actually running (the app's own `@mdx-js/mdx`) | `next-pages-mdx-plugins` (`rehype-slug` + `rehype-autolink-headings` reach the DOM) |
| a configured plugin with **no** app pipeline installed | a hard build error naming plugin and file — never a quiet plain-CommonMark compile (`configured_plugins_without_an_app_pipeline_are_a_hard_error`) |

Known and deliberate, restated so it is not folklore either: diffpack's native
compiler strips YAML frontmatter (and turns `title`/`description` into
`export const metadata`) where `next build` without `remark-frontmatter` renders
it as a thematic break. No app can compare that, for the reason given in item 36.
`with-styled-components` and `preact-ts` are worth singling out: both were hard
build failures at the start of this work.

Every name above is backed by that app's `results/<id>/findings.json`, and those
runs span three fix rounds rather than one simultaneous measurement — see the
scoreboard note.

`github-pages` and `strict-csp` are worth singling out for the opposite reason.
Neither needed a change to diffpack: the suite was measuring them wrong, and had
been reporting a defect against diffpack for each. That is the failure mode a
differential suite has to be most suspicious of — a false difference costs more
than a missed one, because someone acts on it. Both are now pinned by harness
tests (`integration/e2e/lib/apps.test.mjs`, `integration/e2e/lib/probe.test.mjs`),
and `check.sh` runs those tests plus the shell-gate prelude's self-test, which
nothing gated on before.
