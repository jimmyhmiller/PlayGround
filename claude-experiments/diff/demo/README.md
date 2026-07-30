# The side-by-side demo

Two dev servers over one cal.com checkout — `diffpack dev` and `next dev --turbopack`
— framed next to each other in one page, with a clock per side and a button per
scenario. Press a button, watch both clocks run, watch the change land in one frame
long before the other.

On a new machine, install Rust, Node + Corepack, Docker and Docker Compose v2, then
run the one-time setup from the diffpack repository root:

```sh
./demo/setup-calcom.sh
```

The setup builds diffpack, materializes cal.com at its pinned commit, installs the
workspace, starts Postgres, applies migrations and seeds the `pro` user plus its
`30min`/`60min` event types. cal.com is the heavy corpus entry: budget roughly 3.4 GB
and about 20 minutes for a cold install.

After setup:

```sh
node demo/server.mjs
open http://localhost:4321       # macOS; otherwise open this URL in a browser
```

The setup checkout lives at `integration/e2e/.cache/calcom`, which is gitignored and
automatically selected by `demo/server.mjs`. Pass `--app /another/cal.com` to use a
different checkout.

Then wait for both panes to say `ready` (roughly 8 s and 13 s from cold on an M-series
laptop, with both booting at once) and start pressing buttons.

## What it is

```
                  ┌──────────────── http://localhost:4321 ────────────────┐
                  │  scenario buttons                                     │
                  ├───────────────────────────┬───────────────────────────┤
                  │ diffpack   0.169 s        │ Turbopack   3.34 s        │
                  │            19.8x faster   │             +3.17 s       │
                  │ ┌───────── iframe ──────┐ │ ┌───────── iframe ──────┐ │
   one write   →  │ │  localhost:3000       │ │ │  localhost:3001       │ │
   two watchers   │ │  [island #1]          │ │ │  [island #1]          │ │
                  │ └───────────────────────┘ │ └───────────────────────┘ │
                  ├───────────────────────────┴───────────────────────────┤
                  │ results, every run kept   │ both dev servers' output  │
                  └───────────────────────────────────────────────────────┘
```

Every scenario is **one write to one real cal.com source file**. Both dev servers
watch that same tree, so the two sides are reacting to the identical event — which is
what makes the two clocks comparable. Each edit plants a visible badge carrying a
fresh token; a small probe inside each frame (`demo/probe.js`) posts the token set to
the dashboard, and each side's clock stops when *that* side's frame shows the new
token. The number on screen is edit → visibly updated page, in a real browser frame.
The component scenarios show large fixed labels such as `SHARED CLIENT EDIT #1`,
incrementing on every press; the CSS scenario shows `GLOBAL CSS EDIT #1` plus a
matching page ring.

Each landing also says **how** the change arrived: `hot` if the running page was
patched, `reload` if that side replaced the document. Both bundlers fall back to a
full reload when a hot update cannot be applied, and a reload costs the user
everything on screen, so a fast `reload` is not plainly better than a slower `hot`.
The probe stamps every message with an id unique to its document instance, and a
landing from a new id is a reload.

## Scenarios

| button | what it writes | what it exercises |
|---|---|---|
| `island` | `modules/auth/login-view.tsx` | leaf client component, hot-swapped into a running page |
| `server component` | `app/(use-page-wrapper)/auth/login/page.tsx` | RSC re-render, no reload |
| `shared client component` | `components/PageWrapperAppDir.tsx` | a module many routes import, measured on the booker |
| `global stylesheet` | `styles/globals.css` | Tailwind recompile, then the compiled sheet reaching the browser |
| `island ×5 @1/s` | the island file, 5 times, 1 s apart, no settle gap | the contended path — and how many of the 5 states each side ever displayed |
| route chips | nothing | first response for a route, cold or warm |
| `cold start both` | nothing | wipe every output tree (`.diffpack-output`, `.diffpack-next`, `.next`), reboot both servers, race to first paint |
| `production build race` | nothing | `next build --turbopack` vs `diffpack build-app . production`, both from scratch |
| `reset sources` | restores pristine source | undo every scenario edit |

The route chips say **cold** or **warm** against the compiled Next route, not the
URL: `/pro/60min` right after `/pro/30min` is a new URL but the *same* compiled route,
and calling that cold would credit both bundlers with work neither of them does.

Navigation and editing are deliberately separate phases. A route chip only changes the
two iframe URLs; it never writes a source file. A scenario button automatically
navigates both frames when needed, waits for both documents to load and settle, and
then performs exactly one source edit—the named scenario. There are no hidden priming
edits or automatic retries.

This makes the interaction literal and easy to inspect. It also means an edit pressed
immediately after navigation can expose a dev server whose HMR connection is not ready
yet; that outcome remains visible as `TIMEOUT` rather than being masked by a warm-up
edit.

The sustained-edit scenario reports something the single-shot ones cannot. Its edits
land on a fixed 1 s cadence whether or not a side has caught up, and the source only
ever holds one token — so a side still building edit 2 when edit 3 arrives never
displays edit 2 at all. That shows up as **not shown**, and the summary row counts how
many of the five states each side actually put on screen.

## What it is honest about

* **Both dev servers run at once**, so they contend for the machine. That is the price
  of a live side-by-side. Isolated, interleaved, multi-sample numbers come from
  `node scripts/bench-calcom.mjs`.
* **Which side is spawned first alternates every race.** Spawns are sequential, so the
  first process started gets a moment of an uncontended machine; a fixed order would
  hand the same side that moment on every single run. The log says who went first.
* **The probe polls every 8 ms**, so each reading carries up to 8 ms of quantisation —
  identical on both sides.
* **A no-op edit is refused.** If a badge or anchor has gone missing the server errors
  out instead of reporting a suspiciously fast time.
* **Nothing is dropped.** Every run lands in the results table; a side that never
  showed the change reads `not shown`, `TIMEOUT` or `FAILED` rather than vanishing.
* **A non-zero exit is `FAILED`, not a timeout.** A build that crashes on one side must
  not be readable as the other side being fast.
* **`ready` means the real page.** A boot counts as ready on a 200 whose body carries
  `Cal.diy` (`--ready-marker`) *and* a closed `</html>`. A dev server that answers an
  error shell, or a document it never finishes, would otherwise be handed boot time it
  did not earn; the bare 200s are counted and reported.
* **`next build` gets `--turbopack` explicitly.** Next 16 already defaults to it, but
  that default is version- and config-dependent (and Next hard-exits when it
  auto-selects Turbopack for a project with a webpack config and no turbopack config),
  so the flag is passed rather than assumed.
* **Both builds skip type checking and lint.** The demo temporarily wraps the checkout's
  real `next.config.ts` and sets `typescript.ignoreBuildErrors` and
  `eslint.ignoreDuringBuilds`. diffpack does no type checking, so leaving `tsc` inside
  `next build` would time a compiler only one side runs.
* **Build memory is the summed RSS of the whole process tree**, sampled every 250 ms.
  `/usr/bin/time -l`'s `maximum resident set size` is `ru_maxrss`, the largest *single*
  process, and both sides are trees (diffpack runs client, react-server and ssr
  concurrently; `next build` spawns workers), so that number under-reports both, by
  different amounts. It is still shown, labelled, on its own row. A spike shorter than
  the sampling interval is missed, which is why the sample count is printed.
* **The build race runs both builds concurrently**, so both numbers are inflated
  relative to an isolated build; the ratio is the point, and the isolated,
  one-at-a-time, interleaved numbers come from
  `node scripts/bench-calcom.mjs --only build`.

## The fairness rules are gated

```
node demo/dashboard.test.mjs      # drives dashboard.html in jsdom, server stubbed
node demo/racing-order.test.mjs   # who goes first really does alternate
node scripts/tree-rss.test.mjs    # the build-memory sampler, against a real process tree
```

Neither needs a dev server, a browser or a cal.com build; `./check.sh` runs both. They
hold the rules above in place: that navigation performs no source edit, that pressing a
scenario writes exactly one edit, that `hot` and `reload` are distinguished,
that a non-zero exit reads `FAILED` and never `TIMEOUT`, and that a state neither side
displayed keeps its row. The memory test
measures the gap the sampler exists to close: on a tree holding 600 MiB across three
processes, sampling reports ~730 MiB and `ru_maxrss` reports ~164 MiB.

## Known asymmetry, not yet fixed

diffpack evaluates `next.config` with the phase `phase-production-server` even in dev,
while `next dev` uses `phase-development-server` (both visible in
`demo/logs/{dp,tp}.log`). For cal.com the phase only reaches a log line, so nothing
about this comparison turns on it, but an app that branches on the phase would have the
two dev servers serving configs that differ. The fix belongs in diffpack
(`scripts/rsc/next-config-eval.mjs` hardcodes the phase), not in the demo.

## What it touches in the checkout, and puts back

At startup, snapshotted and restored on exit (including SIGINT/SIGTERM):

* `apps/web/app/layout.tsx` — one `<script src="/diffpack-demo-probe.js">` tag.
* `apps/web/next.config.ts` — under `DIFFPACK_DEMO=1` only, drops `X-Frame-Options`
  so the framed pages are not blanked, and aligns the production race's type/lint
  policy. The pristine config is temporarily held in
  `next.config.__diffpack_demo_base__.ts`; both files are restored on exit.
* `apps/web/public/diffpack-demo-probe.js` — a copy of `demo/probe.js`, deleted on exit.
* The four scenario source files above.

The probe returns immediately when it is not inside a frame, so the tag is inert in an
ordinary dev session.

Each side also gets its own `NEXT_PUBLIC_WEBAPP_URL` / `NEXT_PUBLIC_WEBSITE_URL` /
`NEXTAUTH_URL` matching its own port. cal.com bakes those into the browser bundle;
left at the checkout's `:3000` the Turbopack page would call the diffpack server's
API, and the frame would not be showing its own bundler end to end.

## Flags

```
node demo/server.mjs --app /path/to/calcom     # override the auto-detected checkout
                     --dp-port 3000            # diffpack dev
                     --tp-port 3001            # next dev --turbopack
                     --port 4321               # the dashboard
                     --ready-path /auth/login  # the URL a boot waits for a 200 on
                     --ready-marker Cal.diy    # text that 200 must carry to count as ready
                     --no-boot                 # attach to dev servers you started yourself
```

Full dev-server output is kept in `demo/logs/{dp,tp}.log`; the dashboard's log pane is
the same stream with the ANSI escapes taken off.
