#!/usr/bin/env bash
# CLI GATE — the driver's user-facing contract: argv handling, exit codes, fmt.
#
# Every case here is a bug that shipped, because the CLI surface had no test at all
# while the compiler core had several. The recurring shape was silent permissiveness:
# a flag ignored, a missing file read as empty, a crash reported as success. If any of
# these regress, the compiler still passes gate-full — that is precisely the gap.
#
# Usage: selfhost/oracle/gate-cli.sh [path-to-coil]      (default: ./coil)
set -uo pipefail
cd "$(dirname "$0")/../.."                       # repo root
COIL="${1:-./coil}"
[ -x "$COIL" ] || { echo "no coil at $COIL"; exit 1; }
COIL="$(cd "$(dirname "$COIL")" && pwd)/$(basename "$COIL")"

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
FAIL=0
ok()   { echo "  ok   — $1"; }
bad()  { echo "  FAIL — $1"; echo "         $2"; FAIL=1; }

# expect_rc <want> <name> <cmd...>
expect_rc() {
  local want=$1 name=$2; shift 2
  local out; out=$("$@" 2>&1); local rc=$?
  [ "$rc" = "$want" ] && ok "$name" || bad "$name" "want rc=$want got rc=$rc: $out"
}
# expect_out <regex> <name> <cmd...>
expect_out() {
  local want=$1 name=$2; shift 2
  local out; out=$("$@" 2>&1)
  echo "$out" | grep -qE "$want" && ok "$name" || bad "$name" "want /$want/, got: $out"
}

printf '(defn main [] (-> i64) 7)\n'                     > "$T/seven.coil"
printf '(extern abort :cc c [] (-> void))\n(defn main [] (-> i64) (abort) 0)\n' > "$T/abort.coil"
printf '(defn a [] (-> i64)    1)\n'                     > "$T/messy1.coil"
printf '(defn b [] (-> i64)    2)\n'                     > "$T/messy2.coil"

echo "== exit codes (a crash must not look like success) =="
expect_rc 7   "run propagates a normal exit code"        "$COIL" run "$T/seven.coil"
expect_rc 134 "run propagates SIGABRT as 128+signo"      "$COIL" run "$T/abort.coil"
expect_out "signal 6 \(SIGABRT\)" "a signal death names itself" "$COIL" run "$T/abort.coil"

echo "== a file the user named must exist =="
expect_rc 1 "build: missing file is an error"            "$COIL" build "$T/nope.coil" -o "$T/x"
expect_out "no such file" "build: missing file is named" "$COIL" build "$T/nope.coil" -o "$T/x"
expect_rc 2 "fmt: missing file is an error (not 'unformatted')" "$COIL" fmt --check "$T/nope.coil"
"$COIL" fmt --write "$T/ghost.coil" >/dev/null 2>&1
[ -e "$T/ghost.coil" ] && bad "fmt --write must not fabricate a file" "it created one" \
                        || ok  "fmt --write does not fabricate a file"

echo "== flags are position-independent, unknown ones are errors =="
expect_rc 7 "build: flags BEFORE the file"               "$COIL" run "$T/seven.coil"
"$COIL" build -o "$T/a" "$T/seven.coil" >/dev/null 2>&1 && "$T/a"; rc=$?
[ "$rc" = 7 ] && ok "build -o <out> <file> (Unix order)" || bad "build -o <out> <file>" "rc=$rc"
expect_rc 1 "unknown flag is rejected"                   "$COIL" build "$T/seven.coil" -o "$T/b" --frobnicate
expect_out "unknown flag" "unknown flag is named"        "$COIL" build "$T/seven.coil" -o "$T/b" --frobnicate
expect_rc 1 "missing -o exits 1 (not SIGABRT)"           "$COIL" build "$T/seven.coil"
expect_rc 1 "bogus --target is rejected"                 "$COIL" build "$T/seven.coil" -o "$T/c" --target not-a-real-triple

echo "== fmt formats EVERY file it is given =="
out=$("$COIL" fmt --check "$T/messy1.coil" "$T/messy2.coil" 2>&1)
n=$(echo "$out" | grep -c "not formatted")
[ "$n" = 2 ] && ok "fmt --check reports both files" || bad "fmt --check multi-file" "named $n of 2: $out"
expect_rc 2 "fmt on a directory is an error"             "$COIL" fmt "$T"

echo "== per-subcommand help =="
for c in build run fmt new emit-ir; do
  expect_out "usage: coil $c" "$c --help"                "$COIL" "$c" --help
done

echo "== project mode honors flags =="
mkdir -p "$T/proj/src"
printf '[package]\nname  = "proj"\nentry = "src/main.coil"\n' > "$T/proj/Coil.toml"
printf '(module app)\n(defn main [] (-> i64) 3)\n'            > "$T/proj/src/main.coil"
( cd "$T/proj" && "$COIL" build >/dev/null 2>&1 )
[ -x "$T/proj/proj" ] && ok "project build" || bad "project build" "no ./proj"
( cd "$T/proj" && "$COIL" run >/dev/null 2>&1 ); [ $? = 3 ] && ok "project run propagates exit code" \
                                                            || bad "project run" "want 3"
# the headline case: --target wasm32 used to print `wrote proj`, exit 0, and emit a Mach-O
( cd "$T/proj" && rm -f proj && "$COIL" build --target wasm32-unknown-unknown >/dev/null 2>&1 )
if file "$T/proj/proj" 2>/dev/null | grep -q WebAssembly; then
  ok "project --target wasm32 emits WebAssembly"
else
  bad "project --target wasm32" "got: $(file "$T/proj/proj" 2>/dev/null | sed 's/.*: //')"
fi
( cd "$T/proj" && rm -f proj && "$COIL" build -o "$T/proj/elsewhere" >/dev/null 2>&1 )
[ -x "$T/proj/elsewhere" ] && ok "project -o is honored" || bad "project -o" "not written"
( cd "$T/proj" && "$COIL" build --target not-a-real-triple >/dev/null 2>&1 )
[ $? = 1 ] && ok "project bogus --target is rejected" || bad "project bogus --target" "want rc=1"

echo
[ "$FAIL" = 0 ] && echo "gate-cli: PASS" || echo "gate-cli: FAIL"
exit $FAIL
