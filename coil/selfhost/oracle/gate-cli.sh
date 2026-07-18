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

echo "== a compile that cannot finish must SAY SO, not hang or crash =="
# These all used to die with zero output: no message, no location, nothing naming the
# construct — the worst possible failure for a mistake a typo can cause.
cat > "$T/runaway.coil" <<'EOF'
(module rw)
(defstruct Box [T] [(v T)])
(defn grow [T] [(x T)] (-> i64)
  (let [b (alloc-stack (Box T))] (store! (field b v) x) (grow (load b))))
(defn main [] (-> i64) (grow 1))
EOF
out=$(timeout 30 "$COIL" build "$T/runaway.coil" -o "$T/rw" 2>&1); rc=$?
[ "$rc" = 1 ] && ok "runaway monomorphization errors (was: infinite hang)" \
              || bad "runaway monomorphization" "rc=$rc (124=still hanging)"
echo "$out" | grep -q "never reaches a fixpoint" && ok "…and explains the growth" \
                                                 || bad "runaway message" "$(echo "$out" | head -1)"

# deep macro-generated nesting: `cond` expands to nested ifs, so 800 clauses — an
# ordinary bytecode dispatch table — is 800 levels deep, and used to segfault.
{
  printf '(module dp)\n(import "control.coil" :use *)\n(defn f [(x i64)] (-> i64) (cond '
  i=0; while [ $i -lt 800 ]; do printf '(icmp-eq x %d) %d ' $i $i; i=$((i+1)); done
  printf -- '-1))\n(defn main [] (-> i64) (f 5))\n'
} > "$T/deep.coil"
expect_rc 5 "an 800-clause cond builds and runs (was: SIGSEGV)" "$COIL" run "$T/deep.coil"

echo "== independent mistakes are reported in ONE pass, across phases =="
# resolve used to abort on its first error, so a batch of mistakes surfaced one
# recompile at a time — and the one that survived was the span-less resolve error.
cat > "$T/multi.coil" <<'EOF'
(module m)
(defn f [(a i64)] (-> i64) a)
(defn g [] (-> i64) (f 1.5))
(defn h [] (-> i64) (f 1 2))
(defn i2 [] (-> i64) (nosuchfn 3))
(defn j [] (-> i64) (f "x"))
(defn main [] (-> i64) 0)
EOF
out=$("$COIL" build "$T/multi.coil" -o "$T/x" 2>&1)
echo "$out" | grep -q "4 errors" && ok "4 independent errors in one pass (1 resolve + 3 type)" \
                                 || bad "multi-error report" "no '4 errors': $(echo "$out" | tail -1)"
# a resolve error with NO type errors must still fail the build, not reach codegen
cat > "$T/resonly.coil" <<'EOF'
(module m)
(defn ok [(x i64)] (-> i64) (+ x 1))
(defn bad [] (-> i64) (nosuchfn 3))
(defn main [] (-> i64) (ok 5))
EOF
expect_rc 1 "a resolve-only error still fails the build" "$COIL" build "$T/resonly.coil" -o "$T/x"

echo "== trait bounds on type params are resolved and enforced =="
# unknown trait in a bound: error at the DEFINITION (was: accepted, or mis-blamed at the call)
printf '(module m)\n(defn f [(T NoSuchTrait)] [(x T)] (-> i64) 0)\n(defn main [] (-> i64) 0)\n' > "$T/badtrait.coil"
expect_rc 1 "unknown trait in a bound is rejected" "$COIL" build "$T/badtrait.coil" -o "$T/x"
expect_out "unknown trait" "…and named" "$COIL" build "$T/badtrait.coil" -o "$T/x"
# defstruct bound ENFORCED at instantiation (was: silently ignored)
printf '(module m)\n(defstruct Box [(T Eq)] [(v T)])\n(defn u [] (-> i64) (let [b (alloc-stack (Box f64))] 0))\n(defn main [] (-> i64) (u))\n' > "$T/structbound.coil"
expect_rc 1 "defstruct bound enforced ((Box f64), f64 has no Eq)" "$COIL" build "$T/structbound.coil" -o "$T/x"
# defsum now PARSES a bound (was: 'expected symbol') and enforces it
printf '(module m)\n(defsum Opt [(T Eq)] (Non) (Som [(v T)]))\n(defn u [] (-> i64) (let [o (alloc-stack (Opt f64))] 0))\n(defn main [] (-> i64) (u))\n' > "$T/sumbound.coil"
expect_rc 1 "defsum bound parses + enforced" "$COIL" build "$T/sumbound.coil" -o "$T/x"
# and the valid instantiations still compile
printf '(module m)\n(defstruct Box [(T Eq)] [(v T)])\n(defn main [] (-> i64) (let [b (alloc-stack (Box i64))] (store! (field b v) 7) 0))\n' > "$T/okbound.coil"
expect_rc 0 "a satisfied bound ((Box i64)) still compiles" "$COIL" build "$T/okbound.coil" -o "$T/x"

echo "== store! yields unit (std-12): effect-only stores type-check without a wrapping do =="
# was: `store!` took the STORED VALUE's type, so `(if c (store! p ptr) 0)` was a type error
# (then=(ptr i64) vs else=i64) and every non-i64 effect-only store needed `(do (store! …) 0)`.
# Now store! is unit (canonical i64 0): the bare form checks, and the store still happens.
# FAILS on the seed (branch-type error, build exits 1); PASSES here (runs, returns 7).
cat > "$T/std12.coil" <<'EOF'
(module m)
(defn main [] (-> i64)
  (let [n (alloc-stack i64) pp (alloc-stack (ptr i64))]
    (store! n 7)
    (if (icmp-eq 1 1) (store! pp n) 0)   ; stores a (ptr i64); store! yields unit i64 0
    (load (load pp))))
EOF
expect_rc 7 "a bare effect-only (if c (store! p ptr) 0) builds+runs (was: branch-type error)" "$COIL" run "$T/std12.coil"

echo "== type ascription (: value type) supplies a stranded type arg + enforces it (gen-9) =="
# was: a local could not be annotated, so a generic value whose type argument only the
# RETURN position could fix — e.g. (Okk 5), where the error type E is undetermined —
# could not be bound in a let ("cannot infer type argument 'E'"), and `(: …)` itself was
# a parse error ("expected a symbol head, got :"). Now `(: value type)` checks `value`
# against `type`, flowing the expected type IN. FAILS on the seed (parse error → run ≠ 5);
# PASSES here (runs, returns 5 — the payload of (Okk 5)).
cat > "$T/gen9.coil" <<'EOF'
(module m)
(defsum Res [T E] (Okk [(v T)]) (Errr [(e E)]))
(defn via-let [] (-> (Res i64 bool))
  (let [r (: (Okk 5) (Res i64 bool))] r))
(defn main [] (-> i64) (match (via-let) (Okk [v] v) (Errr [e] 0)))
EOF
expect_rc 5 "ascription supplies a let binding's stranded type arg (was: parse error)" "$COIL" run "$T/gen9.coil"
# and inference now flows across the let on its OWN: when the let's tail is the binding
# and the let has an expected type, the binding's value is checked against it — so the
# bare `(let [r (Okk 5)] r)` needs no annotation. FAILS on the seed ("cannot infer type
# argument 'E'"), PASSES here (returns 5).
cat > "$T/gen9auto.coil" <<'EOF'
(module m)
(defsum Res [T E] (Okk [(v T)]) (Errr [(e E)]))
(defn via-let [] (-> (Res i64 bool))
  (let [r (Okk 5)] r))
(defn main [] (-> i64) (match (via-let) (Okk [v] v) (Errr [e] 0)))
EOF
expect_rc 5 "inference flows the return type into a returned let binding (was: 'cannot infer E')" "$COIL" run "$T/gen9auto.coil"
# and it TYPE-CHECKS, it is NOT a silent cast: a value whose type is not the ascribed one
# is rejected with a located error naming both types.
printf '(module m)\n(defn main [] (-> i64) (: true i64))\n' > "$T/gen9bad.coil"
expect_rc 1 "ascription rejects a mismatched value (not a numeric cast)" "$COIL" build "$T/gen9bad.coil" -o "$T/x"
expect_out "has type bool but expected i64" "…naming both the actual and ascribed type" "$COIL" build "$T/gen9bad.coil" -o "$T/x"

echo "== (target-arch) reflects --target, not a hardcoded host constant =="
# was: the constant "aarch64" — so a macro branching on it baked the host branch into a
# cross-compiled wasm build, silently.
cat > "$T/tgt.coil" <<'EOF'
(module t)
(defn arch [] (-> Code)
  (if (code-eq (target-arch) `wasm32) `1 (if (code-eq (target-arch) `aarch64) `2 `3)))
(defn main [] (-> i64) (arch))
EOF
expect_out "i64\) 2" "target-arch = aarch64 on the host"          "$COIL" expand "$T/tgt.coil"
expect_out "i64\) 1" "target-arch = wasm32 under --target wasm32" "$COIL" expand "$T/tgt.coil" --target wasm32-unknown-unknown
expect_out "i64\) 3" "target-arch = x86_64 under --target x86_64" "$COIL" expand "$T/tgt.coil" --target x86_64-apple-macosx11.0.0

echo "== object emission on the DEFAULT (LLVM) backend =="
# Nothing else covers this: gate-full stops at emit-ir, and arm64/gate-run.sh only
# exercises --backend arm64. So `:shim` — a naked trampoline, i.e. INLINE ASM, and the
# language's headline calling-convention-as-a-type feature — silently could not build on
# the default backend at all (LLVM aborts without an AsmParser). A committed example
# (examples/shim.coil) failed to build and no gate noticed.
for e in examples/shim.coil examples/everything.coil; do
  out=$("$COIL" run "$e" 2>&1); rc=$?
  [ "$rc" = 42 ] && ok "$e builds+runs on the LLVM backend (42)" \
                 || bad "$e on the LLVM backend" "rc=$rc: $(echo "$out" | head -1)"
done

echo
[ "$FAIL" = 0 ] && echo "gate-cli: PASS" || echo "gate-cli: FAIL"
exit $FAIL
