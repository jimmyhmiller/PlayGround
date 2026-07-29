# Known issues

Living list. Every entry names its evidence; nothing here is folklore. When an issue is
fixed, move it to the "Resolved" section with the commit rather than deleting it.

## Open

### 1. Residual auto-animate ghost on `next/dynamic` tab swaps (~1 in 3)

cal.com's "Should not allow enabling both recurring event and offer seats" passes ~2 in 3
runs on a diffpack build and 6/6 on the reference. The failure signature is the same
duplicated `[data-testid=offer-seats-toggle]` that the (fixed) Suspense-boundary bug
produced, but it survives the fix: the swap now lands in ONE mutation batch, byte-for-byte
the reference's pattern, yet an `@formkit/auto-animate` exit-ghost with a running WAAPI
animation still appears intermittently. This is a second, distinct defect.

- It is a Heisenbug: every probe that perturbs the test (init scripts, waits between
  steps) suppresses it; Playwright's trace zip truncates when the test dies on timeout.
- Next diagnostic (designed, not yet run): compile a MutationObserver into diffpack's
  emitted client entry so the failing DOM state is captured from inside the page, then
  diff the commit sequence against the reference on the failing iteration only.
- Evidence: day-2 agent report (18-run measurement), STATUS_2026-07-28.md §11.

### 2. No navigation discard queue

Next's `dispatchAction` (`app-router-instance.js:143-154`) marks a pending action
`discarded` when a new NAVIGATE arrives, so a superseded navigation's state is never
applied. diffpack's `navigate()` is a bare async function: two rapid navigations (or a
navigation racing cal.com's effect-driven `router.replace()` in `useTypedQuery`) both
fetch and both commit, in COMPLETION order — last-to-finish wins, not last-requested.

- A discard guard was implemented and measured against known failures: no observable
  effect, so it was deliberately reverted rather than shipped unproven. The divergence is
  real regardless and should be fixed on its own merits, ideally by adopting Next's
  action-queue shape (vendor where separable; MIT attribution).

### 3. `pushState` fires before the commit; Next fires it after

Next pushes history from `HistoryUpdater`'s `useInsertionEffect` after the navigation
commits (`app-router.js:38-71`). diffpack calls `history.pushState` synchronously inside
the transition callback, before the render. During a slow navigation the address bar
leads the DOM. Not implicated in any current test failure; a contract divergence to close
when touching the router, preferably together with issue 2.

### 4. Build CPU +21% / peak RSS +18% after the correctness fixes

Day-2 vs day-1 builds: CPU 23.7s -> 28.6s, peak RSS 1.67GB -> 1.98GB. Attributed to
`pages/api` routes now being bundled into the SSR graph (more modules; map count
763 -> 1,140) plus the styled-jsx / context-module transform scans. Wall time did NOT
regress (parallel headroom absorbed it: 8.6s -> 8.3s), so this is a recovery target, not
a regression alarm. Profile before cutting; the §9 stage profiler and hinted-search work
show where the levers live.

### 5. `ssr: false` dynamic imports use a mounted-gate, not `BailoutToCSR`

Same observable outcome for everything tested (server + first client paint render the
fallback, chunk swaps in after mount), but the mechanism differs from Next's
(`BailoutToCSR` thrown under the boundary). If a future app branches on React's bailout
error specifically, this will surface. Documented in `next_dynamic_shim`'s comment.

### 6. Dev cold start is 5.9s; sub-1-second requires lazy per-route compilation

The remaining cold-start floor is client graph + max(react-server, ssr) built
concurrently + orchestrator boot. Getting under 1s means building routes lazily on first
visit (what `next dev` does), which is an architecture project, deliberately deferred.
The persistent warm-start cache prototype (1.66s restarts) was removed by request —
no caching approaches for now.

### 7. Global CSS edits rebuild but never reach open browsers (Next dev topology)

Editing `styles/globals.css` under `diffpack dev` recompiles the sheet (the marker
appears in the served `/rsc.css` on the next request) but no cache-busted link swap or
reload is pushed, so an open page keeps the stale stylesheet forever. Measured: marker
never applied in 30s, twice; the served sheet had it. Turbopack delivers the same edit
in ~7.4s (its own Tailwind recompile dominates). The SPA/Vite dev path HAS a css
hot-swap (`push_css`); the Next topology never wires it. Fix shape: fingerprint the
emitted sheets after each settle and push the swap (or a reload) when they change.

### 8. Edits within ~1.6s of a previous edit contend with the deferred full re-emit

Clean-path edit-to-DOM is ~50ms (island) / ~50ms (server component) / ~69ms (shared
client component). But an edit arriving while the PREVIOUS edit's deferred full
re-emit is still running (client+ssr ~1.6s, react-server ~1.0s on cal.com) measures
~1.0s to the DOM — a 20x penalty at a ~1-edit-per-second cadence (measured: 112ms
first edit, then 986-1068ms steady). The §2.3 coalescing handles bursts *within* the
150ms settle window; the gap is edits landing *during* the re-emit itself. Turbopack
has no equivalent cliff (flat ~420ms at the same cadence). Fix shape: let hot pushes
preempt/interleave with the background re-emit instead of queueing behind it.

## Reference-side failures (NOT diffpack bugs — do not spend time here)

- `cannot book same slot multiple times`: fails identically on `next start`; the
  quick-availability feature keys off `NEXT_PUBLIC_IS_E2E`, a build-time define neither
  side bakes. Fixable only by building BOTH sides with the flag.
- `can select 'display on booking page' option when multiple organizer input type are
  present`: react-select strict-mode violation in cal.com itself; fails on the reference.
- cal.com's own bugs found en route (upstream material, not ours to fix): a literal
  `${t("switch_monthly")}` string in `Header.tsx` JSX; the same-slot feature-flag issue
  above.

## Resolved (with commit)

| Commit | Issue |
|---|---|
| `75a2a630f` | `next/dynamic` wrapped every call in `<Suspense>`; Next's `Loadable` rule (`!ssr \|\| !!loading`) gives the default call NO boundary — the extra fallback commit made tab swaps two commits and minted auto-animate ghosts (the deterministic variant) |
| `0c180b351` | Soft navigation swapped in a still-streaming flight; a suspended transition commits nothing — stalled navigations presented as flaky tests |
| `c3af226ac` | Tailwind class-composition helper arguments not treated as class positions |
| `7b7a8e99b` | `"use cache"` reused values across arguments JSON cannot represent (stale event-type poisoning) |
| `0adb497cb` / `8d517a38f` | Static assets lacked validators/gzip; compression ran per-request |
| `e7ec0ad84` | `import()` with a variable specifier was emitted verbatim instead of expanding into a context module (non-English locales 500'd) |
| `38ba92eec` | Page Server Component `searchParams` ignored the request query |
| `771133425` | `pages/api` ran outside the SSR layer; route hooks died across soft nav |
| `e2f5720a9` | Navigation hooks returned fresh identities per render (unbounded render loop on the booker) |
| `24e3f0bc3` (sweep) | styled-jsx not compiled (hydration mismatch on every provider-wrapped route); `NEXT_PUBLIC_*` never inlined (undefined env in client bundles); mixed import+require demand downgrade shook off live exports |
