#!/usr/bin/env bash
# The "diffpack drove react-dom wrong" log signatures, in ONE place — sourced by the
# production gate (next-check.sh) and the dev gate (next-dev-ssr-api-check.sh), which
# render the same SSR entry through two different orchestrator paths (buffered
# `renderFlightToDocument` in dev, streaming `renderFlightToStream` in production).
#
# These are NOT "any error in the server log". A gate that banned those would fail on
# integration/next-app-router's /error-demo, whose Server Component throws on purpose.
# Each pattern here can only mean the ENTRY misused the API:
#
#   * "only supports piping to one writable stream" — `pipe()` called twice on one
#     `renderToPipeableStream` result. react-dom's ready callbacks are not fire-once:
#     when the last work to finish is a Suspense boundary still holding abortable
#     fallback tasks, `finishedTask` reaches `completeAll` -> `onAllReady` from the
#     nested abort AND from its own tail. Rendering /error-demo under `diffpack dev`
#     reproduces it exactly.
#   * "ERR_HTTP_HEADERS_SENT" — a second `res.writeHead` on a response whose head is
#     already on the wire. The streaming path writes the head from `onShellReady`, so a
#     re-entered callback there does not log noise, it throws in the request handler.
#   * "This is a bug in React" — React's own invariant text ("There can only be one
#     root segment", ...). Reached only by feeding React a state it says is impossible.
#
# React reports the first as a RECOVERABLE error, which is why every content assertion
# in these gates kept passing while cal.com logged it once per request for months.
REACT_DOM_MISUSE_RE='only supports piping to one writable stream|ERR_HTTP_HEADERS_SENT|This is a bug in React'

# assert_no_react_dom_misuse <server-log> <what-was-rendered>
# Requires the sourcing gate's `fail()` (from _gate-prelude.sh).
assert_no_react_dom_misuse() {
  local log="$1" what="$2" hits
  hits="$(grep -aE "$REACT_DOM_MISUSE_RE" "$log" || true)"
  [ -z "$hits" ] || {
    echo "$hits"
    fail "the SSR entry misused react-dom's server API while rendering $what (see above). The documents still rendered — React reports this class as recoverable — which is why no content assertion catches it."
  }
}
