# The side-by-side demo

Two dev servers over one cal.com checkout — `diffpack dev` and `next dev --turbopack`
— framed next to each other in one page, with a clock per side and a button per
scenario. Press a button, watch both clocks run, watch the change land in one frame
long before the other.

```
cargo build --release
node demo/server.mjs
open http://localhost:4321
```

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

## Scenarios

| button | what it writes | what it exercises |
|---|---|---|
| `island` | `modules/auth/login-view.tsx` | leaf client component, hot-swapped into a running page |
| `server component` | `app/(use-page-wrapper)/auth/login/page.tsx` | RSC re-render, no reload |
| `shared client component` | `components/PageWrapperAppDir.tsx` | a module many routes import, measured on the booker |
| `global stylesheet` | `styles/globals.css` | Tailwind recompile, then the compiled sheet reaching the browser |
| `island ×5 @1/s` | the island file, 5 times, 1 s apart, no settle gap | the contended path — and how many of the 5 states each side ever displayed |
| route chips | nothing | first response for a route, cold or warm |
| `cold start both` | nothing | wipe both output trees, reboot both servers, race to first paint |
| `production build race` | nothing | `next build` vs `diffpack build-app`, both from scratch |
| `reset sources` | restores pristine source | undo every scenario edit |

The route chips say **cold** or **warm** against the compiled Next route, not the
URL: `/pro/60min` right after `/pro/30min` is a new URL but the *same* compiled route,
and calling that cold would credit both bundlers with work neither of them does.

The sustained-edit scenario reports something the single-shot ones cannot. Its edits
land on a fixed 1 s cadence whether or not a side has caught up, and the source only
ever holds one token — so a side still building edit 2 when edit 3 arrives never
displays edit 2 at all. That shows up as **not shown**, and the summary row counts how
many of the five states each side actually put on screen.

## What it is honest about

* **Both dev servers run at once**, so they contend for the machine. That is the price
  of a live side-by-side. Isolated, interleaved, multi-sample numbers come from
  `node scripts/bench-calcom.mjs`.
* **The probe polls every 8 ms**, so each reading carries up to 8 ms of quantisation —
  identical on both sides.
* **A no-op edit is refused.** If a badge or anchor has gone missing the server errors
  out instead of reporting a suspiciously fast time.
* **Nothing is dropped.** Every run lands in the results table; a side that never
  showed the change reads `not shown` or `TIMEOUT` rather than vanishing.
* **The build race runs both builds concurrently**, so both numbers are inflated
  relative to an isolated build; the ratio is the point, and the isolated numbers are
  in the benchmark script.

## What it touches in the checkout, and puts back

At startup, snapshotted and restored on exit (including SIGINT/SIGTERM):

* `apps/web/app/layout.tsx` — one `<script src="/diffpack-demo-probe.js">` tag.
* `apps/web/next.config.ts` — under `DIFFPACK_DEMO=1` only, drops `X-Frame-Options`
  so the framed pages are not blanked. cal.com sends `DENY` on `/auth/*`.
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
node demo/server.mjs --app /tmp/dpe2e/calcom   # the cal.com checkout
                     --dp-port 3000            # diffpack dev
                     --tp-port 3001            # next dev --turbopack
                     --port 4321               # the dashboard
                     --ready-path /auth/login  # the URL a boot waits for a 200 on
                     --no-boot                 # attach to dev servers you started yourself
```

Full dev-server output is kept in `demo/logs/{dp,tp}.log`; the dashboard's log pane is
the same stream with the ANSI escapes taken off.
