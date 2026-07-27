#!/usr/bin/env bash
# Meta-gate: the GATES themselves must not be able to fail silently.
#
# Under `set -e`, a standalone capture `v="$(... grep ...)"` takes the substitution's exit
# status as its own, so a legitimately-empty match aborts the script ON THE ASSIGNMENT —
# before the named `fail` on the next line can run. The operator then sees exit 1 with no
# diagnostics and the assertion is unreachable dead code. This linter hard-fails, naming
# file:line, when a gate reintroduces that shape or drops the shared safety net.
#
# Checks, over every scripts/rsc/*.sh that enables `set -e`:
#   1. it sources _gate-prelude.sh (strict mode + the ERR trap + fail());
#   2. it does not re-declare its own strict mode / fail(), which would shadow the prelude;
#   3. no capture from a command that legitimately matches nothing (grep/ls/find/agent-browser)
#      is left unneutralized by `|| true`.
# Exit 0 = lint PASS.
# Strict mode, the ERR net (no abort is ever silent) and fail() — see _gate-prelude.sh.
source "$(dirname "$0")/_gate-prelude.sh"

cd "$(dirname "$0")"
prelude="_gate-prelude.sh"
status=0
note() { echo "LINT: $*" >&2; status=1; }

for f in *.sh; do
  [ "$f" = "$prelude" ] && continue
  # Only gates that enable -e can suffer the silent-abort class.
  grep -qE '^[[:space:]]*set -[A-Za-z]*e' "$f" || grep -q "$prelude" "$f" || continue

  grep -q "source \"\$(dirname \"\$0\")/$prelude\"" "$f" \
    || note "$f: does not source $prelude (no ERR net — an abort here would be silent)"
  # A duplicated strict-mode line means the file drifted back to its own prelude. A bare
  # `set -e` is NOT flagged: gates legitimately restore it after a `set +e` window around a
  # command that is EXPECTED to fail (e.g. next-missing-dep-check.sh's negative build).
  while IFS=: read -r n _; do
    [ -n "$n" ] && note "$f:$n: re-declares full strict mode; the prelude already sets -Eeuo pipefail"
  done < <(grep -nE '^[[:space:]]*set -[Ee]*[a-z]*uo[[:space:]]+pipefail[[:space:]]*$' "$f" || true)
  while IFS=: read -r n _; do
    [ -n "$n" ] && note "$f:$n: re-declares fail(); the prelude already defines it"
  done < <(grep -n '^[[:space:]]*fail() {' "$f" || true)

  # The silent-abort shape: NAME="$( ... grep|ls|find|agent-browser ... )" with no `|| true`.
  # `if`/`while` conditions and captures inside && / || lists are ERR-exempt, so only match a
  # statement that STARTS the line (the standalone-assignment case).
  #
  # Escape hatch: `|| true` is WRONG where an empty result would compare equal and silently
  # retire the assertion (e.g. checksum-drift captures — a truncated hash on both sides
  # matches). Such a line must instead assert its preconditions explicitly and carry a
  # preceding `# lint-gates: allow <reason>` line. The reason is mandatory, and the ERR trap
  # still reports the line loudly if it ever does fail, so nothing here can be silent.
  while IFS=: read -r n line; do
    case "$line" in
      *'|| true'*) continue ;;
    esac
    prev="$(sed -n "$((n - 1))p" "$f")"
    case "$prev" in
      *'# lint-gates: allow '?*) continue ;;
      *'# lint-gates: allow'*)
        note "$f:$((n - 1)): '# lint-gates: allow' with no reason — state why '|| true' is wrong here"
        continue ;;
    esac
    note "$f:$n: unneutralized capture (a no-match aborts before the next line's named fail); \
append '|| true' inside \$( ) and assert on the result: ${line#"${line%%[![:space:]]*}"}"
  done < <(grep -nE '^[A-Za-z_][A-Za-z_0-9]*="\$\(' "$f" \
             | grep -E 'grep |ls |find |agent-browser ' || true)
done

# shellcheck also flags the `local x="$(...)"` status-masking class; advisory when present.
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -S warning -e SC1090,SC1091 ./*.sh || note "shellcheck reported warnings (see above)"
else
  echo "note: shellcheck not installed — skipping the supplementary static check"
fi

[ "$status" = 0 ] || fail "gate lint found silent-failure shapes above; fix them (never weaken the gate)"
echo "PASS: every scripts/rsc gate sources the ERR net and has no unneutralized captures"
