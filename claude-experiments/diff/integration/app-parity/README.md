# app-parity — behavioral differential testing of diffpack vs Vite

Proves (or disproves) that a diffpack-built app **behaves identically** to its
reference Vite build: the same scenario is driven against both dists in the
same headless Chromium, and five observation channels are compared after every
step. Anything that can't be made exactly comparable must be declared
`mode: "invariant"` with a written reason.

Test-only tooling. Nothing here is part of any build.

## Run

```sh
# whole suite (exits nonzero on any DIFF/ERROR, including invariant failures)
node integration/app-parity/run.mjs

# one app
node integration/app-parity/run.mjs wall-go
```

Outputs:

- per-app per-step PASS/DIFF table on stdout
- `report.json` — full machine-readable results
- `artifacts/<app>/` — for every DIFF step: both screenshots, normalized DOM
  dumps, a `-channels.json` with the exact diffs, and a `-pixeldiff.png`
  overlay when >0.1% of pixels differ. Invariant steps always keep their
  screenshots so the nondeterminism claim can be audited.

Dependencies are borrowed from `../vite-react-reference` (puppeteer-core via
`createRequire`, `static-server.mjs`). Chromium is resolved from `$CHROME`,
`~/.cache/ms-playwright/chromium-1194/`, or system paths.

App configs live in `apps/<name>.mjs`: dist locations, base path (base-path
apps are served from a temp root with the dist symlinked under the base, e.g.
`/wall-go/`), init stubs (markpad's `__TAURI_INTERNALS__`), fixed storage
seeds, and a 3–8 step scenario. Each step's `run(page)` executes identically
on both sides; optional `probe(page)` returns a JSON value that is compared
exactly as an extra channel.

## Comparison channels (per step)

1. **Normalized DOM** — `document.body.outerHTML` after in-page
   normalization (see N4–N6) plus text normalization (N1–N3, N8, whitespace
   collapsing).
2. **Full computed styles** — `getComputedStyle` over *all* properties of
   *every* element under body, matched pairwise by structural DOM path
   (tag + child index). Numeric tolerance 0.5 (px-ish values); colors compare
   per channel with tolerance 1/255 (see `styleValuesMatch`).
3. **Screenshot** — full-page PNG both sides, pixel-diffed inside a Chromium
   utility page (native PNG decode, no extra deps) with per-channel tolerance
   3; the % of differing pixels is reported and any nonzero % (or page
   dimension mismatch) is a DIFF. Canvas-rendered content (chebyshev's plots)
   is covered by this channel.
4. **Console + network** — console messages and the multiset of first-party
   request URLs; third-party requests are dropped from the URL multiset but
   their hostnames are recorded and compared (N7, N9, and the third-party
   failure rule below).
5. **Storage** — localStorage + sessionStorage JSON after each step.

## Determinism controls

- CSS injected before page scripts:
  `*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}`
  plus `prefers-reduced-motion: reduce` emulation. (JS-driven animation —
  framer-motion, WAAPI — is not stopped by this; steps use generous settle
  waits instead.)
- Fixed 1280×800 viewport, `--force-device-scale-factor=1`.
- `Math.random` replaced with mulberry32 (fixed seed, same on both sides) via
  `evaluateOnNewDocument`, so seeded app randomness aligns (this makes
  the-last-pawn's enemy spawning exactly comparable). **This does not reach
  Web Workers** — worker randomness must use `mode: "invariant"`.
- Per-app fixed storage seeds (e.g. redux-essentials' faker seed date,
  wall-go's theme/language) written before page scripts run.
- Pages are `bringToFront()`ed before running/capturing each side — headless
  Chromium throttles rAF/timers/rendering on background tabs.

### `mode: "invariant"`

Where true nondeterminism remains, a step declares `mode: "invariant"` with a
mandatory `why` and a `check` returning named predicates evaluated against
both sides. Exact channel comparison is skipped for that step; the invariants
must hold on both sides. Currently only wall-go's AI-reply step is invariant:
the AI's move choice calls `Math.random` *inside its Web Worker*, which init
scripts cannot seed.

## Normalization rules (each is a claim that a difference is benign)

Every rule is generic across apps. If a rule can mask a class of real bugs,
the caveat and the compensating channel are stated.

- **[N1] content-hashed asset names** — `name-B4x9Zk2p.ext` / `name-7375e725.ext`
  → `name.ext` (8+ chars of `[A-Za-z0-9_-]` before a known asset extension).
  Hashes are content-addressing, not behavior. Vite's hashes are base64-ish
  and may contain `-`, so the match is leftmost-greedy; multi-dash names like
  `inter-v13-latin-700-HASH.woff2` collapse further (`inter.woff2`) on *both*
  sides — loses some name sensitivity, never invents a diff, and request
  counts are still compared.
- **[N2] server origin** — the two dists are served on different ephemeral
  ports; `http://127.0.0.1:<port>` → `ORIGIN`. Harness plumbing, not behavior.
- **[N3] bundle source positions** — `ORIGIN/...:line:col` in messages →
  position stripped. Line/col inside a bundle is a property of bundle layout;
  message identity is what's compared.
- **[N4] CSS-module scoped class names** — tokens of the scoped shape
  (`_local_1lpfs_154` vite, `_local_7375e725` diffpack) are reduced to the
  stable local-name stem by iteratively stripping trailing `_<hash>`/`_<line>`
  segments. Scoped-name spelling differs between bundlers *by construction*.
  Symmetric; worst case merges snake_case names (loses sensitivity only).
- **[N5] class attribute ordering** — class lists are sorted before compare;
  emission order of class tokens is toolchain-dependent, the class *set* is
  what CSS matches against.
- **[N6] style="" serialization** — inline styles are re-serialized through
  the CSSOM per declaration and sorted; declaration order and vendor
  serialization quirks are not behavior. Values are preserved.
- **[N7] console/network as multisets** — async interleaving of logs and
  fetches is scheduler timing; counts and contents must still match exactly.
- **[N8] v4 UUIDs** → `UUID`. Crypto-random runtime identifiers (e.g. MSW's
  service-worker client id) cannot be bundler-determined and never repeat
  across runs. Caveat: an app rendering a *seeded* UUID would lose
  sensitivity here.
- **[N9] first-party request URLs compared by basename** (after N1), and the
  `.worker.js` chunk suffix folds to `.js` — emitted directory layout and
  chunk naming (`/assets/index-HASH.js` vs `/index.js`,
  `AIWorker-HASH.js` vs `AIWorker-HASH.worker.js`) are bundler packaging
  choices. A wrong path that fails to resolve still surfaces as a first-party
  404/failure, so base-path bugs are not masked.
- **[N10] CSS custom-property value serialization** — custom-prop computed
  values are the *as-authored token text*, so minifier rewrites leak through:
  quote style (`'x'`/`"x"`), hex shortening (`#fff`), leading zeros (`.4`),
  hex↔rgb() color form (8-bit alpha quantization: `rgb(0 0 0/0.045)` ↔
  `#0000000b`), `s`↔`ms`. Values are canonicalized before compare; anything
  differing beyond quantization is still a DIFF (this is what caught the
  `--font-sans` divergence — it was NOT normalized away).
- **[N11] custom properties present on one side only** — recorded in the
  channel output (`customPropsOnlyRef/…Diffpack`) as a note, not a DIFF.
  Observed cause: the two CSS pipelines emit different sets of *unused*
  Tailwind theme variables. A custom prop that is actually consumed cannot
  hide here: its consumption appears in some resolved (non-custom) property,
  and all of those are compared. The counts are printed in every styles
  detail line so this is never silent.
- **third-party request failures** are excluded from the failures channel
  (this sandbox has no outbound network; beacon timing varies) — but the
  attempted third-party hostnames are recorded and compared on both sides.

## Interpreting results

A DIFF is a real, reproducible behavioral difference between the two builds —
the suite deliberately does not widen tolerances to make apps pass. See
`report.json` and `artifacts/` for the current findings (at the time of
writing: markpad's Tailwind default `--font-sans` divergence, chebyshev's
missing small-asset data-URI inlining, wall-go's missing Tailwind utility
CSS, and diffpack's runtime `process.env` shim observable through i18next).
