# HMR generality harness

Proves `diffpack dev` delivers **state-preserving React Fast Refresh** on real
apps — not just the pinned fixtures — by driving a real browser over an edit and
asserting the page did NOT reload (component state survives). This is the
daily-workflow payoff of diffpack's low diff times, exercised on arbitrary Vite
SPAs.

It is deliberately dependency-light. Driving the browser uses
[`agent-browser`](https://www.npmjs.com/package/agent-browser) — a **global CLI**,
so there is no per-app `puppeteer-core` dependency to install into every app under
test. The only inputs are the built `diffpack` binary, `agent-browser`, and the
app's own `npm install`.

## One-time setup

```bash
# 1. agent-browser (global) + its browser binary
npm install -g agent-browser
agent-browser install            # add --with-deps on Linux CI

# 2. the diffpack binary (the harness auto-builds it if missing)
cargo build --release
```

`hmr-check.sh` also resolves `npx agent-browser` if the global bin is absent, and
runs `agent-browser install` itself if the browser binary is missing — so a CI job
only needs `npm i -g agent-browser` (plus `--with-deps` once on Linux).

## Run

```bash
# One app
integration/hmr-harness/hmr-check.sh integration/hmr-harness/apps/redux-essentials.conf

# Everything (clones the network apps)
integration/hmr-harness/run-all.sh

# Fast, network-free smoke (in-repo apps only)
integration/hmr-harness/run-all.sh --offline
```

Exit code is 0 only if every selected app preserved state. `--keep` retains cloned
app dirs for debugging.

## What each run does

1. Resolve the app: an in-repo `LOCAL_DIR`, or a shallow clone of `REPO` pinned to
   `COMMIT` (reproducible).
2. `npm install` if `node_modules` is absent.
3. Start `diffpack dev` on a free port; wait for it to serve.
4. Open the app in `agent-browser`, wait for `MOUNT_TEXT` (proves it rendered),
   then tag the live page with a probe + a console-error trap.
5. Edit `EDIT_FILE` (replace `FIND` with a unique new string) and wait for the new
   text to appear (proves the hot update landed).
6. Assert: for `STATE=preserve`, the page-scoped probe survived (Fast Refresh, no
   reload) and no uncaught JS errors fired. Restore the file, tear everything down.

## Adding an app

Drop a `apps/<name>.conf` (KEY=VALUE; quote values with spaces):

```sh
NAME=my-app
REPO=https://github.com/owner/repo.git   # or LOCAL_DIR=path/relative/to/repo/root
COMMIT=<pin>                             # strongly recommended for reproducibility
INSTALL="npm install --no-audit --no-fund"
EDIT_FILE=src/components/Header.tsx       # a component file with visible text
FIND="Original Heading"                   # exact substring to change
REPLACE="Edited Heading"                  # base for the new text (a stamp is appended)
MOUNT_TEXT="Original Heading"             # text present on first render (default: FIND)
STATE=preserve                            # preserve = expect Fast Refresh (default)
# IGNORE_ERR="some other console noise"   # optional, joined with the built-in ignores
```

## Coverage

| app | plugin | export style | notes |
| --- | --- | --- | --- |
| vite-react-reference | @vitejs/plugin-react v6 (self-contained runtime) | default | offline, in-repo |
| redux-essentials | @vitejs/plugin-react v4.2 (self-contained runtime) | named (`export const`) | RTK, `vite.config.mts` |
| the-last-pawn | @vitejs/plugin-react v4.3 (SPLIT runtime: react-refresh core + refreshUtils) | default | non-root `base`, sass CSS modules, page transitions |

These span both @vitejs/plugin-react runtime layouts and both export styles. Adding
the-last-pawn is what surfaced the split-layout runtime handling in
`src/hmr.rs::find_refresh_runtime`.

## Notes / limits

- Pick a component whose edit is a Fast Refresh boundary (a module whose exports
  are all components). Editing a route/util module legitimately reloads —
  set `STATE=reload` for those.
- Missing static assets (favicon, etc.) are ignored as console noise, matching the
  fixture oracles; add app-specific noise via `IGNORE_ERR`.
- `agent-browser` sessions are namespaced per run, so parallel runs don't collide.
- **`@vitejs/plugin-react-swc` is not yet supported by the dev refresh runtime**
  (it ships a self-contained ESM runtime with a different API surface —
  `getRefreshReg` rather than `register`, a trailing `export {…}` block). An app on
  the SWC plugin (e.g. chebyshev-calculator) needs `find_refresh_runtime` +
  `refresh_runtime_source` extended before it can be added here.
