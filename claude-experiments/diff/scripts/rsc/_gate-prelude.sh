#!/usr/bin/env bash
# Shared prelude for the scripts/rsc/*.sh gates: strict mode, a loud ERR trap, and fail().
# Sourced (never executed) as the first statement of every gate that sets -e.
#
# Every gate runs under `set -Eeuo pipefail`, so a failing command aborts the script. That
# abort is otherwise SILENT for the single most common gate idiom:
#
#     hero="$(echo "$html" | grep -oiE '<img[^>]*id="hero"[^>]*>' | head -1)"
#     [ -n "$hero" ] || fail "next/image: the raster hero <img id=hero> was not rendered"
#
# A standalone assignment takes its command substitution's exit status as its own, so a
# `grep` that legitimately matches nothing (exit 1, propagated past a succeeding `wc`/`head`
# by `pipefail`) kills the script on the ASSIGNMENT line — the named `fail` on the next line
# is unreachable dead code, and the operator sees exit 1 with ZERO diagnostics.
#
# The ERR trap below makes that impossible: any unhandled failure names file:line and the
# command that died. It is a NET, not a substitute for a named failure — the right shape for
# a capture that may legitimately match nothing is still to neutralize it and assert:
#
#     hero="$(echo "$html" | grep -oiE -m1 '<img[^>]*id="hero"[^>]*>' || true)"
#     [ -n "$hero" ] || fail "next/image: the raster hero <img id=hero> was not rendered"
#
# (`grep -m1` rather than `| head -1` also avoids `head` closing the pipe early and killing
# a SUCCEEDING grep with SIGPIPE/141.) scripts/rsc/lint-gates.sh enforces both this prelude
# and that capture shape; scripts/rsc/tests/gate-prelude-selftest.sh tests the net itself.
set -Eeuo pipefail

# `-E` propagates the trap into functions, subshells and command substitutions, so the
# report names the line that actually died rather than the caller. Bash raises ERR twice for
# a failing capture (once inside the substitution subshell for the innermost command, once
# for the enclosing assignment); reporting only at BASH_SUBSHELL 0 keeps the outer report —
# the one that shows the whole assignment — and drops the duplicate. The trap only REPORTS:
# `set -e` performs the exit and preserves the status, including through a gate's own
# `trap cleanup EXIT`. (BASH_SUBSHELL is bash 3.2-safe; BASHPID is not, and macOS ships 3.2.)
#
# Two things the obvious one-liner gets WRONG, both fixed here and both covered by
# tests/gate-prelude-selftest.sh:
#
#   * The ERR trap fires even when errexit is OFF. A gate that opens a deliberate
#     `set +e` window around an intentionally-failing command (next-missing-dep-check.sh
#     runs a build it EXPECTS to fail, then asserts on its status) would print
#     `FAIL: ... aborted (exit 1)` on a run that PASSES. `case $- in *e*)` reports only
#     while errexit is actually in force, which is exactly when the abort is unhandled.
#     ERR does not fire for a command in an `if`/`&&`/`||` condition, so this test does
#     not need to model "errexit on but suspended".
#
#   * `$LINENO` inside a MULTI-LINE trap string is offset by that reference's own line
#     index within the string on bash 3.2 — what macOS ships — so a two-line trap
#     reported every failure as `line + 1`. Capturing `__ln=$LINENO` on the FIRST line
#     of the trap string is correct on 3.2 and on bash 4/5 alike, with no version test.
#     $BASH_COMMAND/$BASH_SOURCE are captured alongside it so the later lines cannot
#     clobber them.
trap '__st=$?; __ln=$LINENO; __cmd=$BASH_COMMAND; __src=${BASH_SOURCE[0]}
      case $- in
        *e*) if [ "$BASH_SUBSHELL" = 0 ]; then
               echo "FAIL: $__src:$__ln aborted (exit $__st) running: $__cmd" >&2
             fi ;;
      esac' ERR

fail() { echo "FAIL: $*" >&2; exit 1; }
