#!/usr/bin/env bash
# Self-test for scripts/rsc/_gate-prelude.sh — the net that guarantees "a failing gate
# always prints something" is itself TESTED, not assumed.
#
# Each case builds a throwaway script that sources the real prelude, runs it, and asserts on
# the exit status + stderr. Case 1 is the exact regression: before the prelude existed, a
# capture whose `grep` matched nothing aborted the gate with exit 1 and ZERO output, leaving
# the named assertion on the following line unreachable.
# Exit 0 = self-test PASS.
set -Eeuo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
prelude="$here/../_gate-prelude.sh"
[ -f "$prelude" ] || { echo "FAIL: prelude not found at $prelude" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cases=0
check() { # <name> <expected-status> <stderr-ERE> <script-body>
  cases=$((cases + 1))
  local name="$1" want="$2" pat="$3" body="$4" got err
  { printf 'source %q\n%s\n' "$prelude" "$body"; } > "$work/case.sh"
  err="$(bash "$work/case.sh" 2>&1 >/dev/null)" && got=0 || got=$?
  [ "$got" = "$want" ] || { echo "FAIL: [$name] exit $got, want $want; stderr: $err" >&2; exit 1; }
  echo "$err" | grep -qE "$pat" \
    || { echo "FAIL: [$name] stderr did not match /$pat/; got: ${err:-<EMPTY>}" >&2; exit 1; }
  echo "ok: $name"
}

# 1. THE REGRESSION: a no-match capture aborts — but now names file:line and the command.
#    Without the prelude this exits 1 having printed absolutely nothing.
check "no-match capture is reported, not silent" 1 '^FAIL: .*case\.sh:[0-9]+ aborted \(exit 1\)' '
html="<div>no img here</div>"
hero="$(echo "$html" | grep -oiE "<img[^>]*id=\"hero\"[^>]*>" | head -1)"
[ -n "$hero" ] || fail "the hero <img> was not rendered"
echo REACHED'

# 2. Exactly ONE report: bash raises ERR for both the substitution subshell and the enclosing
#    assignment; the BASH_SUBSHELL guard must drop the duplicate.
cases=$((cases + 1))
printf 'source %q\nv="$(grep nope /dev/null)"\n' "$prelude" > "$work/dup.sh"
n="$(bash "$work/dup.sh" 2>&1 >/dev/null | grep -c 'aborted (exit' || true)"
[ "$n" = "1" ] || { echo "FAIL: expected exactly 1 abort report, got $n" >&2; exit 1; }
echo "ok: a single abort produces exactly one report"

# 3. The neutralized shape reaches its NAMED assertion instead of aborting anonymously.
check "|| true lets the named fail run" 1 '^FAIL: the hero <img> was not rendered$' '
html="<div>no img here</div>"
hero="$(echo "$html" | grep -oiE -m1 "<img[^>]*id=\"hero\"[^>]*>" || true)"
[ -n "$hero" ] || fail "the hero <img> was not rendered"
echo REACHED'

# 4. The status survives a gate's own `trap cleanup EXIT` (gates all install one).
check "exit status survives an EXIT trap" 1 'aborted \(exit 1\)' '
cleanup() { kill 999999 2>/dev/null || true; }
trap cleanup EXIT
grep -q nope /dev/null
echo REACHED'

# 5. set -e exemptions still hold: `[ -e missing ] && fail` must NOT abort or report, or
#    every existing "assert this file is absent" line in the gates would false-fail.
check "and-list probe does not false-report" 0 '^$' '
[ -e /definitely/missing ] && fail "must not fire"
for f in a b c; do [ -e "/definitely/missing/$f" ] && fail "must not fire in a loop either"; done
echo OK'

# 6. A passing gate stays silent on stderr — the net must not add noise to green runs.
check "no report on success" 0 '^$' '
v="$(echo hello | grep -o hello || true)"
[ "$v" = "hello" ] || fail "unexpected"'

# 7. REGRESSION: a deliberate `set +e` window (next-missing-dep-check.sh runs a build it
#    EXPECTS to fail, then asserts on the status) must not print a spurious FAIL. The net
#    reports UNHANDLED aborts; inside `set +e` the failure is handled by the next line.
check "no report inside a deliberate set +e window" 0 '^$' '
set +e
false
status=$?
set -e
[ "$status" = 1 ] || fail "the intentionally-failing command did not fail"
echo OK'

# 8. REGRESSION: the reported line is the line that DIED. `$LINENO` inside a multi-line
#    trap string is offset by its own line index within the string on bash 3.2 (what macOS
#    ships, and what the prelude targets), so this asserts the exact number, not a pattern.
cases=$((cases + 1))
{ printf 'source %q\n' "$prelude"; printf 'echo pad\n'; printf 'echo pad\n'; printf 'grep -q nope /dev/null\n'; } \
  > "$work/lineno.sh"
lineno_err="$(bash "$work/lineno.sh" 2>&1 >/dev/null || true)"
echo "$lineno_err" | grep -qE 'lineno\.sh:4 aborted' \
  || { echo "FAIL: the abort report must name line 4 (the grep); got: ${lineno_err:-<EMPTY>}" >&2; exit 1; }
echo "ok: the abort report names the line that actually died"

echo "PASS: gate prelude self-test ($cases cases) — no gate abort can be silent"
