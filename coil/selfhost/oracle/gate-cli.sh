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

echo "== check mode: typecheck/compile with no object (diag-12) =="
# `build -o /dev/null` USED to SIGABRT with a bare 'LLVMTargetMachineEmitToFile ...
# Operation not permitted' (exit 134) because /dev is unwritable. It now routes to the
# compile-check path: full front-end, no object emitted, real exit codes. FAILS on the
# seed (SIGABRT 134, and no `check` command at all).
printf '(defn main [] (-> i64) (bad-fn 3))\n' > "$T/broken.coil"
expect_rc 0 "build -o /dev/null on a good program exits 0 (was SIGABRT 134)"  "$COIL" build "$T/seven.coil"  -o /dev/null
expect_rc 1 "build -o /dev/null on a broken program exits 1"                  "$COIL" build "$T/broken.coil" -o /dev/null
expect_out "undefined function 'bad-fn'" "build -o /dev/null reports a LOCATED error, not an LLVM abort" \
  "$COIL" build "$T/broken.coil" -o /dev/null
expect_rc 0 "check: a good program exits 0"              "$COIL" check "$T/seven.coil"
expect_rc 1 "check: a broken program exits 1"            "$COIL" check "$T/broken.coil"
expect_out "undefined function 'bad-fn'" "check names the located error"      "$COIL" check "$T/broken.coil"
# check emits NO object: the .o path must not appear.
"$COIL" check "$T/seven.coil" >/dev/null 2>&1
[ ! -e "$T/seven.o" ] && ok "check writes no object file" || bad "check writes no object" "$T/seven.o exists"
# A genuinely non-writable -o is a CLEAR error + exit 1, not a SIGABRT.
expect_rc 1 "build -o into a read-only location is a clear error, not SIGABRT" \
  "$COIL" build "$T/seven.coil" -o /System/nope
expect_out "could not write object file" "non-writable -o names the path + a remedy" \
  "$COIL" build "$T/seven.coil" -o /System/nope
expect_out "usage: coil check" "check --help documents itself"                "$COIL" check --help

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

echo "== tool-12: the Coil.toml reader is STRICT — unknown sections/keys are LOCATED errors =="
# The reader knows exactly five sections (package/build/link/cc/run) and a fixed key
# set. `[dependencies]` used to be silently accepted (inviting the false inference that
# deps resolve) and `entrypoint` typos were swallowed. Now each is a located hard error
# naming the offending line. FAILS on the seed (which builds clean, rc=0); PASSES here.
mkdir -p "$T/strict/src"
printf '(module app)\n(defn main [] (-> i64) 3)\n' > "$T/strict/src/main.coil"
# unknown section [dependencies]
printf '[package]\nname  = "s"\nentry = "src/main.coil"\n\n[dependencies]\nfoo = "1.0"\n' > "$T/strict/Coil.toml"
( cd "$T/strict" && "$COIL" build >/dev/null 2>&1 ); [ $? = 1 ] \
  && ok "[dependencies] is rejected (was: silently ignored)" \
  || bad "strict [dependencies]" "want rc=1"
out=$( cd "$T/strict" && "$COIL" build 2>&1 )
echo "$out" | grep -qE "Coil.toml:5: unknown section \[dependencies\]" \
  && ok "…and the error is located at the section line" \
  || bad "strict [dependencies] location" "got: $out"
# typo'd key `entrypoint`
printf '[package]\nname  = "s"\nentrypoint = "src/main.coil"\n' > "$T/strict/Coil.toml"
out=$( cd "$T/strict" && "$COIL" build 2>&1 ); rc=$?
[ "$rc" = 1 ] && ok "a typo'd key (entrypoint) is rejected (was: swallowed)" \
              || bad "strict typo key" "want rc=1 got rc=$rc: $out"
echo "$out" | grep -qE "Coil.toml:3: unknown key 'entrypoint' in \[package\]" \
  && ok "…and the error names the key + section + line" \
  || bad "strict typo key location" "got: $out"
# a valid manifest still builds
printf '[package]\nname  = "s"\nentry = "src/main.coil"\n' > "$T/strict/Coil.toml"
( cd "$T/strict" && rm -f s && "$COIL" build >/dev/null 2>&1 )
[ -x "$T/strict/s" ] && ok "a valid manifest still builds" || bad "strict valid manifest" "no ./s"

echo "== tool-1: a relative import resolves against the IMPORTING FILE's directory, not the CWD =="
# The layout `coil new` scaffolds (src/main.coil) must be able to import a sibling
# (src/util.coil). Under the old CWD-relative rule this failed with
# `import 'util.coil': not found` — a scaffolded project could not split into files.
# FAILS on a pre-tool-1 compiler (import not found → build fails, rc≠42); PASSES here.
mkdir -p "$T/sib/src"
printf '[package]\nname  = "sib"\nentry = "src/main.coil"\n'                             > "$T/sib/Coil.toml"
printf '(module util)\n(defn forty-two [] (-> i64) 42)\n'                                > "$T/sib/src/util.coil"
printf '(module app)\n(import "util.coil" :use *)\n(defn main [] (-> i64) (forty-two))\n' > "$T/sib/src/main.coil"
( cd "$T/sib" && "$COIL" run >/dev/null 2>&1 ); [ $? = 42 ] \
  && ok "project src/main.coil imports sibling src/util.coil" \
  || bad "sibling import (project mode)" "want rc=42 (was: import 'util.coil' not found)"
# The base is the FILE's directory, NOT the CWD: build the entry from an unrelated CWD.
( cd "$T" && "$COIL" run "$T/sib/src/main.coil" >/dev/null 2>&1 ); [ $? = 42 ] \
  && ok "sibling import resolves from ANY cwd (file-relative, not cwd-relative)" \
  || bad "sibling import (arbitrary cwd)" "want rc=42"
# The PRELUDE + bundled libs are self-contained: their imports resolve to the BUNDLED
# stdlib, never to same-named decoys sitting in the entry file's directory. A naive
# file-relative switch made the prelude's control.coil->print->io chain resolve to
# examples/io.coil (a demo), silently dropping the io library from every build — the
# emit-ir change the prior attempt could not explain. `println` here comes only from
# the prelude, so it breaks if the chain loads the decoy instead of bundled io.
mkdir -p "$T/dec"
printf '(module control)\n(defn decoy [] (-> i64) 0)\n'           > "$T/dec/control.coil"
printf '(module io)\n(defn decoy [] (-> i64) 0)\n'                > "$T/dec/io.coil"
printf '(module app)\n(defn main [] (-> i64) (println "hi") 7)\n' > "$T/dec/main.coil"
( cd "$T/dec" && "$COIL" run "$T/dec/main.coil" >/dev/null 2>&1 ); [ $? = 7 ] \
  && ok "the prelude reaches the BUNDLED stdlib despite same-named decoys in the entry dir" \
  || bad "bundled prelude self-contained" "want rc=7 (println from bundled print/io, not the decoy)"

echo "== diag-10: (:use [name]) naming a symbol the module does NOT export is a located error =="
# util exports only `good`; `secret` is private. `(import … :use [secret])` used to be
# silently accepted (build succeeded, rc=0) — the bogus name evaporated instead of erroring.
# Now it is a located import-site error naming the importer, the symbol, and the target module.
# FAILS on the seed (which builds it clean, rc=0); PASSES here.
mkdir -p "$T/use"
printf '(module util)\n(export good)\n(defn good [] (-> i64) 42)\n(defn secret [] (-> i64) 99)\n' > "$T/use/util.coil"
printf '(module app)\n(import "util.coil" :use [secret])\n(defn main [] (-> i64) 0)\n'              > "$T/use/nono.coil"
expect_rc  1 "a non-exported :use name is rejected (was: silently accepted)" "$COIL" build "$T/use/nono.coil" -o "$T/use/x"
expect_out "'secret', which module 'util' does not export" "…and the error names the symbol + module" \
  "$COIL" build "$T/use/nono.coil" -o "$T/use/x"
# and the legitimate exported name still builds
printf '(module app)\n(import "util.coil" :use [good])\n(defn main [] (-> i64) (good))\n' > "$T/use/yes.coil"
expect_rc 42 "an exported :use name still resolves" "$COIL" run "$T/use/yes.coil"

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

echo "== qualified trait-method calls (A::go) recover a same-name collision (gen-6) =="
# was: two traits declaring the same method name were an UNFIXABLE collision — `(go x)`
# errored "ambiguous / none in scope" and there was no syntax to pick one. Now a
# `Trait::method` head pins dispatch to the named trait. FAILS on the seed ('A::go' is a
# call to an undefined function → rc=1), PASSES here (dispatches to A's impl → 111).
cat > "$T/gen6.coil" <<'EOF'
(deftrait A [Self] (go [(x Self)] (-> i64)))
(deftrait B [Self] (go [(x Self)] (-> i64)))
(impl A i64 (go [(x i64)] (-> i64) 111))
(impl B i64 (go [(x i64)] (-> i64) 222))
(defn main [] (-> i64) (A::go 5))
EOF
expect_rc 111 "A::go selects trait A's impl (was: undefined function 'A::go')" "$COIL" run "$T/gen6.coil"
sed 's/(A::go 5)/(B::go 5)/' "$T/gen6.coil" > "$T/gen6b.coil"
expect_rc 222 "B::go selects trait B's impl" "$COIL" run "$T/gen6b.coil"
# an unknown qualifier/method is a located trait-method error, not a bare undefined-function
printf '(deftrait A [Self] (go [(x Self)] (-> i64)))\n(impl A i64 (go [(x i64)] (-> i64) 1))\n(defn main [] (-> i64) (A::nope 5))\n' > "$T/gen6bad.coil"
expect_out "names no trait method 'nope'" "A::nope is a located trait-method error" "$COIL" build "$T/gen6bad.coil" -o "$T/x"
# the collision error stops recommending `:use` (useless when the traits are your own) and
# points at the qualified escape hatch instead. FAILS on the seed (its message says `:use`).
cat > "$T/gen6col.coil" <<'EOF'
(module app)
(deftrait A [Self] (go [(x Self)] (-> i64)))
(deftrait B [Self] (go [(x Self)] (-> i64)))
(impl A i64 (go [(x i64)] (-> i64) 111))
(impl B i64 (go [(x i64)] (-> i64) 222))
(defn main [] (-> i64) (go 5))
EOF
expect_out "call it qualified" "a collision error advertises Trait::method, not :use" "$COIL" build "$T/gen6col.coil" -o "$T/x"

echo "== supertraits: (deftrait D [Self] :requires [Base] …) (gen-8) =="
# was: supertraits and associated-type params shared the ONE `[Self …]` vector, so a
# supertrait written there was silently accepted as an extra type parameter, and there
# was no way to say "every impl of D must also impl Base". Now `:requires [Base …]` is a
# SEPARATE clause the checker ENFORCES. FAILS on the seed (`:requires` is unknown there,
# so the keyword parses as a bad trait method → build error), PASSES here (runs → 11).
cat > "$T/super_ok.coil" <<'EOF'
(module app)
(deftrait Animal [Self] (legs [(x Self)] (-> i64)))
(deftrait Pet [Self] :requires [Animal] (name [(x Self)] (-> i64)))
(defstruct Dog [(n i64)])
(impl Animal Dog (legs [(x Dog)] (-> i64) 4))
(impl Pet Dog (name [(x Dog)] (-> i64) 7))
(defn main [] (-> i64) (let [d (alloc-stack Dog)] (iadd (Animal::legs (load d)) (Pet::name (load d)))))
EOF
expect_rc 11 "a :requires supertrait builds+runs when the base is impl'd (was: parse error)" "$COIL" run "$T/super_ok.coil"
# the supertrait is ENFORCED: impl Pet without impl Animal is a located error naming the
# base. Both the seed and this build exit 1 here, so the TEETH is the message: the seed
# fails to PARSE `:requires` ("trait method must be…"), this build names the supertrait.
cat > "$T/super_missing.coil" <<'EOF'
(module app)
(deftrait Animal [Self] (legs [(x Self)] (-> i64)))
(deftrait Pet [Self] :requires [Animal] (name [(x Self)] (-> i64)))
(defstruct Dog [(n i64)])
(impl Pet Dog (name [(x Dog)] (-> i64) 7))
(defn main [] (-> i64) 0)
EOF
expect_rc 1 "impl Pet without impl Animal is rejected" "$COIL" build "$T/super_missing.coil" -o "$T/x"
expect_out "supertrait 'app.Animal'" "…and the error names the missing supertrait" "$COIL" build "$T/super_missing.coil" -o "$T/x"
# the OLD ambiguous form — a supertrait smuggled into the `[Self …]` vector — is now a
# located error pointing at `:requires` (was: silently an associated-type parameter, so
# the seed builds it clean, rc=0). The extra param NAMES a trait, which is the tell.
cat > "$T/super_ambig.coil" <<'EOF'
(module app)
(deftrait Animal [Self] (legs [(x Self)] (-> i64)))
(deftrait Pet [Self Animal] (name [(x Self)] (-> i64)))
(defn main [] (-> i64) 0)
EOF
expect_rc 1 "a trait name in the [Self …] vector is rejected" "$COIL" build "$T/super_ambig.coil" -o "$T/x"
expect_out "if you meant a supertrait" "…and the error points at :requires" "$COIL" build "$T/super_ambig.coil" -o "$T/x"
# a genuine associated-type parameter (NOT a trait name) is still fine — the guard only
# fires on names that actually resolve to a trait, so `[Self K E]` (à la Get) still works.
printf '(module app)\n(deftrait Grab [Self K E] (grab [(x Self) (k K)] (-> E)))\n(defn main [] (-> i64) 0)\n' > "$T/super_assoc.coil"
expect_rc 0 "associated-type params [Self K E] still parse (not mistaken for supertraits)" "$COIL" build "$T/super_assoc.coil" -o "$T/x"

echo "== parameterized traits are usable as bounds; extra params are associated types (gen-1) =="
# was: a bound over a PARAMETERIZED trait (Get/Set/Push/Pop or any [Self …] trait) was a
# hard error — "trait 'Pop' takes type parameters — bounds over parameterized traits aren't
# supported yet" — so there was NO generic code over any collection. Now the non-Self params
# are ASSOCIATED TYPES determined by the impl (one impl per type): the bounded body checks
# against them, and mono resolves each to the impl's concrete type. FAILS on the seed (build
# exits 1, "aren't supported yet"), PASSES here.
# Pop is (deftrait Pop [Self E] (pop! [(xs (mut Self))] (-> (Option E)))); E is return-only.
cat > "$T/gen1pop.coil" <<'EOF'
(module app)
(import "arraylist.coil" :use *)
(import "alloc.coil" :use *)
(import "result.coil" :use *)
(defn drain-count [(C Pop)] [(xs (mut C))] (-> i64)
  (let [(mut n) 0]
    (loop (match (pop! (mut xs)) (None [] (break)) (Some [v] (store! n (+ (load n) 1)))))
    (load n)))
(defn main [] (-> i64)
  (let [a (malloc-allocator) (mut xs) (al-new [i64] a)]
    (al-push! (mut xs) 10) (al-push! (mut xs) 20) (al-push! (mut xs) 30)
    (drain-count (mut xs))))
EOF
expect_rc 3 "Pop used as a bound drains a concrete ArrayList (was: 'aren't supported yet')" "$COIL" run "$T/gen1pop.coil"
# a user's own parameterized trait, two impls, associated element type read off each impl
cat > "$T/gen1custom.coil" <<'EOF'
(module app)
(import "result.coil" :use *)
(deftrait Container [Self Elem]
  (head [(c Self)] (-> (Option Elem)))
  (size [(c Self)] (-> i64)))
(defstruct IntBox [(v i64)])
(impl Container IntBox (head [(c IntBox)] (-> (Option i64)) (Some (load (field c v)))) (size [(c IntBox)] (-> i64) 1))
(defstruct Empty [(x i64)])
(impl Container Empty (head [(c Empty)] (-> (Option i64)) (None)) (size [(c Empty)] (-> i64) 0))
(defn describe [(C Container)] [(c C)] (-> i64)
  (match (head c) (Some [v] (+ 100 (size c))) (None [] (size c))))
(defn main [] (-> i64)
  (let [b (alloc-stack IntBox) e (alloc-stack Empty)]
    (store! (field b v) 7) (store! (field e x) 0)
    (+ (describe (load b)) (describe (load e)))))
EOF
expect_rc 101 "a user parameterized trait as a bound dispatches per-impl (IntBox=101, Empty=0)" "$COIL" run "$T/gen1custom.coil"
# calling a parameterized method on an UNBOUNDED type param is a located definition-time error
printf '(module m)\n(import "result.coil" :use *)\n(deftrait Box2 [Self E] (peek [(c Self)] (-> (Option E))))\n(defn bad [T] [(x T)] (-> i64) (match (peek x) (Some [v] 1) (None [] 0)))\n(defn main [] (-> i64) 0)\n' > "$T/gen1unb.coil"
expect_out "'T' is not bounded by 'm.Box2'" "an unbounded param calling a parameterized method is located" "$COIL" build "$T/gen1unb.coil" -o "$T/x"
# an associated type in ARGUMENT position (a Get key, not return) renders readably in the
# mismatch — `<C as Get>::K`, not the internal mangling. The seed can't reach this message
# (it errors "aren't supported yet" first), so this is the teeth.
printf '(module app)\n(import "arraylist.coil" :use *)\n(defn first-elem [(C Get)] [(xs C)] (-> i64) (get xs 0))\n(defn main [] (-> i64) 0)\n' > "$T/gen1assoc.coil"
expect_out "expected <C as Get>::K" "an associated type renders as <C as Trait>::Param in diagnostics" "$COIL" build "$T/gen1assoc.coil" -o "$T/x"

echo "== one Iterator/Iterable protocol: (for x (iter coll)); (in map) fixed (gen-1 · std-11 · std-4) =="
# was: iteration was four unrelated per-collection macros (slice-for/al-for/hm-for/for-in
# via len+get), there was NO Iterator trait, and `(for-in [k (in map)])` iterated GARBAGE —
# a map's `get` takes a KEY, so `(get m i)` by slot index is nonsense (std-4). Now one
# coil.core Iterator/Iterable protocol drives `(for x (iter coll))` uniformly over slices,
# lists and maps; a map yields its keys, so `(in map)` is correct.
# (for x (iter slice)) — the unified surface. FAILS on the seed (no protocol → rc≠20).
cat > "$T/it-slice.coil" <<'EOF'
(module app)
(import "slice.coil" :use *)
(import "control.coil" :use *)
(defn main [] (-> i64)
  (let [arr (alloc-stack (array i64 4)) (mut s) 0]
    (store! (index arr 0) 2) (store! (index arr 1) 4) (store! (index arr 2) 6) (store! (index arr 3) 8)
    (for x (iter (slice-new (index arr 0) 4)) (store! s (iadd (load s) x)))
    (load s)))
EOF
expect_rc 20 "(for x (iter slice)) drives the Iterator protocol (was: no such protocol)" "$COIL" run "$T/it-slice.coil"
# std-4: (for-in [k (in map)]) now iterates the map's KEYS correctly (was: get-by-index
# garbage → 'arithmetic on different types i64 vs (Option i64)' on the seed).
cat > "$T/it-map.coil" <<'EOF'
(module app)
(import "hashmap.coil" :use *)
(import "alloc.coil" :use *)
(import "control.coil" :use *)
(defn main [] (-> i64)
  (let [a (malloc-allocator) (mut hm) (hm-new-scalar [i64 i64] a) (mut ksum) 0]
    (hm-put! (mut hm) 40 1) (hm-put! (mut hm) 60 2)
    (for-in [k (in hm)] (store! ksum (iadd (load ksum) k)))
    (load ksum)))
EOF
expect_rc 100 "(in map) iterates the map's keys (std-4: was get-by-index garbage)" "$COIL" run "$T/it-map.coil"
# a generic bounded on the Iterator trait consumes ANY iterator (Item abstract) — the
# associated-type bound (gen-1) composing with the protocol. FAILS on the seed (rc≠3).
cat > "$T/it-generic.coil" <<'EOF'
(module app)
(import "arraylist.coil" :use *)
(import "alloc.coil" :use *)
(import "control.coil" :use *)
(defn count-iter [(I Iterator)] [(it (mut I))] (-> i64)
  (let [(mut n) 0]
    (loop (match (next (mut it)) (None [] (break)) (Some [x] (store! n (iadd (load n) 1)))))
    (load n)))
(defn main [] (-> i64)
  (let [a (malloc-allocator) (mut xs) (al-new [i64] a)]
    (al-push! (mut xs) 7) (al-push! (mut xs) 8) (al-push! (mut xs) 9)
    (let [(mut it) (iter xs)] (count-iter (mut it)))))
EOF
expect_rc 3 "a generic (I Iterator) consumes any iterator via the protocol" "$COIL" run "$T/it-generic.coil"
# the collapsed per-collection macros still work: slice-for/al-for now expand through the
# unified protocol (equivalent element iteration).
cat > "$T/it-alfor.coil" <<'EOF'
(module app)
(import "arraylist.coil" :use *)
(import "alloc.coil" :use *)
(import "control.coil" :use *)
(defn main [] (-> i64)
  (let [a (malloc-allocator) (mut xs) (al-new [i64] a) (mut s) 0]
    (al-push! (mut xs) 10) (al-push! (mut xs) 11) (al-push! (mut xs) 12)
    (al-for [v xs] (store! s (iadd (load s) v)))
    (load s)))
EOF
expect_rc 33 "al-for still iterates (now a thin alias over the Iterator protocol)" "$COIL" run "$T/it-alfor.coil"

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

echo "== export-c on the arm64 backend (unblocks the LLVM-free compiled metaprogram engine) =="
# Was: the arm64 backend hard-errored "export-c is not supported by the arm64 backend yet"
# for ANY (export-c …), so main_a64 could register no metaprogram object builder and the
# LLVM-free compiler was stuck on the interpreter (mac-12). It is AAPCS64-native, so
# scalar/pointer params and struct RETURNS are emitted directly under the C symbol with
# external linkage — no thunk. A by-value STRUCT param is the one case that still needs a
# marshaling thunk: a clear hard error, never a silently-wrong symbol.
cat > "$T/expc.coil" <<'EOF'
(module shapes)
(defstruct Point [(x i64) (y i64)])
(defn clamp0 [(n i64)] (-> i64) (if (icmp-lt n 0) 0 n))
(defn make-point [(x i64) (y i64)] (-> Point)
  (let [p (alloc-stack Point)] (store! (field p x) (clamp0 x)) (store! (field p y) (clamp0 y)) (load p)))
(defn add3 [(x i64) (y i64) (z i64)] (-> i64) (iadd (iadd x y) z))
(export-c [make-point :as "shapes_make_point"] [add3 :as "shapes_add3"])
EOF
if "$COIL" emit-obj "$T/expc.coil" -o "$T/expc.o" --backend arm64 >/dev/null 2>&1; then
  cat > "$T/expc_drv.c" <<'EOF'
#include <stdint.h>
typedef struct { int64_t x, y; } Point;
extern Point   shapes_make_point(int64_t, int64_t);
extern int64_t shapes_add3(int64_t, int64_t, int64_t);
int main(void){ Point p = shapes_make_point(3, -4);          /* (3,0) */
  return (int)(p.x + p.y + shapes_add3(10,20,30) - 63); }    /* 3 + 60 - 63 = 0 */
EOF
  if cc "$T/expc_drv.c" "$T/expc.o" -o "$T/expc_test" 2>/dev/null && "$T/expc_test"; then
    ok "export-c --backend arm64: scalar + struct-return callable from C (AAPCS64, no thunk)"
  else
    bad "export-c --backend arm64: C call" "the linked object did not return the expected value"
  fi
else
  bad "export-c --backend arm64: emit-obj rejected a thunk-free export" "seed hard-errors on all exports"
fi
# a by-value struct param is a clear located hard error (SIGABRT), naming the reason
printf '(module s)\n(defstruct P [(x i64)(y i64)])\n(defn d [(p P)] (-> i64) (load (field p x)))\n(export-c [d :as "s_d"])\n' > "$T/expc_bad.coil"
expect_out "by-value struct parameter isn't supported" \
  "export-c arm64: by-value struct param is a clear error, not a bad symbol" \
  "$COIL" emit-obj "$T/expc_bad.coil" -o "$T/expc_bad.o" --backend arm64

echo "== std-3: string HashMap keys are OWNED by default (copied on insert/freed on remove) =="
# Was: str-keyops stored the caller's (slice u8) fat pointer VERBATIM, so two keys built
# over one reused buffer aliased it. Overwrite the buffer between inserts and the map ends
# up holding two keys BOTH reading the buffer's latest bytes ("gamma"), the earlier key
# ("alpha") unreachable. Now str-keyops deep-copies each inserted key into the map's own
# allocator (and frees it on remove/clear/free), so a key never aliases caller storage.
# The program returns alpha*100 + gamma*10 + len : owning => 1*100+2*10+2 = 122 ;
# the old borrow bug => 0*100+2*10+2 = 22 (alpha lost, two entries both "gamma").
cat > "$T/std3.coil" <<'EOF'
(module app)
(import "lib/hashmap.coil" :use *) (import "lib/str.coil"    :use *)
(import "lib/slice.coil"   :use *) (import "lib/mem.coil"    :use *)
(import "lib/alloc.coil"   :use *) (import "lib/result.coil" :use *)
(import "lib/control.coil" :use *)
(defn probe [(ops (ptr KeyOps))] (-> i64)
  (let [a (malloc-allocator) (mut m) (hm-new [(slice u8) i64] a ops)
        buf (alloc-stack (array u8 8))]
    (mem-copy [u8] (cast (ptr u8) buf) (slice-data "alpha") 5)
    (hm-put! (mut m) (slice-new (cast (ptr u8) buf) 5) 1)       ; key "alpha" -> 1
    (mem-copy [u8] (cast (ptr u8) buf) (slice-data "gamma") 5)  ; OVERWRITE the buffer
    (hm-put! (mut m) (slice-new (cast (ptr u8) buf) 5) 2)       ; key "gamma" -> 2
    (iadd (imul (match (hm-get [(slice u8) i64] m "alpha") (Some [v] v) (None [] 0)) 100)
          (iadd (imul (match (hm-get [(slice u8) i64] m "gamma") (Some [v] v) (None [] 0)) 10)
                (hm-len m)))))
(defn main [] (-> i64) (probe (str-keyops)))
EOF
# FAILS on the seed (borrow: alpha aliased away -> 22), PASSES here (owning -> 122).
expect_rc 122 "str-keyops OWNS keys: alpha survives a reused-buffer overwrite" \
  "$COIL" run "$T/std3.coil"
# The opt-in escape hatch str-keyops-borrowed keeps the old (unsafe) borrow behavior,
# reproducing the aliasing on purpose (-> 22). FAILS on the seed (no such function).
sed 's/(str-keyops)/(str-keyops-borrowed)/' "$T/std3.coil" > "$T/std3b.coil"
expect_rc 22 "str-keyops-borrowed is the opt-in borrow path (alpha aliased away -> 22)" \
  "$COIL" run "$T/std3b.coil"

echo "== --debug-checks: library bounds checks (mem-6), zero-cost when off =="
# The build-mode predicate (debug-checks?) gates slice bounds checks inside macros, so
# the check is emitted ONLY under --debug-checks and the off-path IR is byte-identical.
cat > "$T/dbgget.coil" <<'EOF'
(module m)
(import "slice.coil" :use *)
(defn main [] (-> i64)
  (let [arr (alloc-stack (array i64 3))]
    (store! (index arr 0) 10)
    (let [s (slice-new (index arr 0) 3)] (slice-get s 7))))
EOF
# ON: a located OOB message before the crash. FAILS on the seed ('unknown flag --debug-checks').
expect_out "slice-get index out of bounds" "--debug-checks catches a slice-get OOB" \
  "$COIL" run "$T/dbgget.coil" --debug-checks
# OFF (default): NO check emitted — the read runs past the end WITHOUT the debug message.
out=$("$COIL" run "$T/dbgget.coil" 2>&1)
echo "$out" | grep -q "out of bounds" && bad "off: the bounds check must NOT fire (zero-cost)" "$out" \
                                      || ok "off: no bounds check emitted (zero-cost when off)"
# the mem-6 headline: subslice lo>hi used to yield a slice reporting length -2.
cat > "$T/dbgsub.coil" <<'EOF'
(module m)
(import "slice.coil" :use *)
(defn main [] (-> i64)
  (let [arr (alloc-stack (array i64 4))]
    (let [s (slice-new (index arr 0) 4)] (slice-len (subslice s 3 1)))))
EOF
# OFF: the invariant break is unchanged (length hi-lo = -2 -> exit 254).
"$COIL" run "$T/dbgsub.coil" >/dev/null 2>&1
[ $? = 254 ] && ok "off: subslice(3,1) still yields length -2 (behavior unchanged)" \
             || bad "off: subslice(3,1)" "want rc=254 (length -2)"
# ON: rejected with the located message. FAILS on the seed ('unknown flag --debug-checks').
expect_out "subslice out of range or lo>hi" "--debug-checks rejects a lo>hi subslice (mem-6)" \
  "$COIL" run "$T/dbgsub.coil" --debug-checks

echo "== --sanitize=address + a sanitizer --link-flag no longer aborts the compiler (mem-7) =="
# was: a program's --link-flag reached the metaprogram dylib, which is dlopen'd into the
# compiler — an ASan-instrumented dylib loaded that late ABORTED the compiler
# ("interceptors are not working"). Now sanitizer flags are filtered off the dylib link.
# FAILS on the seed (the compiler aborts), PASSES here (builds).
"$COIL" build "$T/seven.coil" -o "$T/san1" --link-flag -fsanitize=address >/dev/null 2>&1
[ $? = 0 ] && ok "--link-flag -fsanitize=address does not abort the compiler" \
           || bad "--link-flag -fsanitize=address" "the compiler aborted (mem-7)"
# --sanitize=address runs LLVM's AddressSanitizer pass: the object gains __asan symbols.
# --lib keeps it off the exe link (whose ASan runtime version is a toolchain concern).
# FAILS on the seed ('unknown flag --sanitize=address').
rm -f "$T/sanobj.o"
"$COIL" build "$T/seven.coil" --lib --sanitize=address -o "$T/sanobj.a" >/dev/null 2>&1
nm "$T/sanobj.o" 2>/dev/null | grep -q asan \
  && ok "--sanitize=address runs the ASan pass (object is instrumented)" \
  || bad "--sanitize=address" "no __asan symbols in the emitted object"
# and WITHOUT the flag the SAME object has no ASan — proving the flag is what adds it.
rm -f "$T/plainobj.o"
"$COIL" build "$T/seven.coil" --lib -o "$T/plainobj.a" >/dev/null 2>&1
nm "$T/plainobj.o" 2>/dev/null | grep -q asan \
  && bad "a plain build must not be instrumented" "found __asan without --sanitize" \
  || ok "no ASan symbols without --sanitize=address"
# ASan needs the LLVM backend — the native arm64 backend is a clear error, never a silent
# uninstrumented binary. FAILS on the seed ('unknown flag').
expect_out "requires the LLVM backend" "--sanitize=address --backend arm64 is rejected" \
  "$COIL" build "$T/seven.coil" -o "$T/san2" --sanitize=address --backend arm64

echo "== poison-on-free debug-allocator: detects double-free (mem-2) =="
# (debug-allocator a) from dbgalloc.coil — under --debug-checks it detects a double-free
# with a located abort (was: a bare signal or silent reuse). FAILS on the seed
# (dbgalloc.coil is not bundled there, so the build fails and prints no such message).
cat > "$T/df.coil" <<'EOF'
(module m)
(import "alloc.coil" :use *)
(import "dbgalloc.coil" :use *)
(defn main [] (-> i64)
  (let [a (debug-allocator (malloc-allocator))
        p (unwrap-ptr [i64] (create [i64] a))]
    (store! p 5)
    (destroy [i64] a p)
    (destroy [i64] a p)
    0))
EOF
expect_out "double free in debug-allocator" "--debug-checks detects a double free (mem-2)" \
  "$COIL" run "$T/df.coil" --debug-checks
# OFF (default): (debug-allocator a) is exactly `a` — no wrapper, the double free is not
# detected and the program exits 0 (zero-cost, behavior unchanged).
"$COIL" run "$T/df.coil" >/dev/null 2>&1
[ $? = 0 ] && ok "off: debug-allocator is a passthrough (no detection, zero cost)" \
           || bad "off: debug-allocator passthrough" "want rc=0"
# a use-after-free reads the 0xDE poison (222) rather than the freed value under the flag.
cat > "$T/uaf.coil" <<'EOF'
(module m)
(import "alloc.coil" :use *)
(import "dbgalloc.coil" :use *)
(defn main [] (-> i64)
  (let [a (debug-allocator (malloc-allocator))
        p (unwrap-ptr [i64] (create [i64] a))]
    (store! p 1234) (destroy [i64] a p) (iand (load p) 255)))
EOF
expect_rc 222 "--debug-checks poisons freed memory (use-after-free reads 0xDE)" \
  "$COIL" run "$T/uaf.coil" --debug-checks

echo "== stack-return lint under --debug-checks (mem-8) =="
# A bundled (checker …) the driver auto-applies under --debug-checks warns when a
# function returns a pointer to a stack local (clang warns on the identical C; Coil was
# silent). FAILS on the seed ('unknown flag --debug-checks', no such warning).
cat > "$T/dangle.coil" <<'EOF'
(module m)
(defn dangling [] (-> (ptr i64))
  (let [x (alloc-stack i64)] (store! x 42) x))
(defn main [] (-> i64) (load (dangling)))
EOF
expect_out "returns a pointer to a stack local" "--debug-checks warns on a stack-local pointer return (mem-8)" \
  "$COIL" build "$T/dangle.coil" -o "$T/dl1" --debug-checks
# it is a WARNING (like clang), not an error — the build still succeeds.
"$COIL" build "$T/dangle.coil" -o "$T/dl2" --debug-checks >/dev/null 2>&1
[ $? = 0 ] && ok "the stack-return lint is a warning (build still succeeds)" \
           || bad "stack-return lint severity" "the build failed"
# OFF (default): the checker is not even loaded — silent, zero cost.
out=$("$COIL" build "$T/dangle.coil" -o "$T/dl3" 2>&1)
echo "$out" | grep -q "stack local" && bad "off: the lint must not run" "$out" \
                                    || ok "off: the stack-return lint is not loaded (zero cost)"
# no false positive: a function returning a HEAP pointer is fine under the flag.
cat > "$T/heapret.coil" <<'EOF'
(module m)
(import "alloc.coil" :use *)
(defn mk [(a (ptr Allocator))] (-> (ptr i64))
  (let [p (unwrap-ptr [i64] (create [i64] a))] (store! p 9) p))
(defn main [] (-> i64) (load (mk (malloc-allocator))))
EOF
out=$("$COIL" build "$T/heapret.coil" -o "$T/hr" --debug-checks 2>&1)
echo "$out" | grep -q "stack local" && bad "no false positive: heap return" "flagged a heap ptr" \
                                    || ok "no false positive: a heap-pointer return is not flagged"

echo "== assert / assert-eq: located failures via the span machinery (tool-12) =="
# lib/assert.coil (a bundled library) bakes the offending expression AND its file:line
# into the emitted code via the new code-src / code-file / code-line comptime ops. ALL of
# this FAILS on the seed: assert.coil is not bundled there, and code-line/code-src are
# unknown ops.
cat > "$T/assf.coil" <<'EOF'
(module m)
(import "assert.coil" :use *)
(defn main [] (-> i64)
  (assert (> 2 1))
  (assert (< 9 3))
  0)
EOF
expect_rc 134 "assert failure aborts (SIGABRT = 128+6)"                     "$COIL" run "$T/assf.coil"
expect_out "assertion failed: \(< 9 3\)" "assert prints the offending expression (code-src)" "$COIL" run "$T/assf.coil"
expect_out "assf\.coil:5"  "assert prints file:line (code-file/code-line)"  "$COIL" run "$T/assf.coil"
cat > "$T/asseq.coil" <<'EOF'
(module m)
(import "assert.coil" :use *)
(defn main [] (-> i64) (assert-eq (+ 40 1) 99))
EOF
expect_out "assertion failed: \(\+ 40 1\) == 99" "assert-eq prints BOTH expressions" "$COIL" run "$T/asseq.coil"

echo "== deftest + the test transform: discovery, fork isolation, exit code (tool-12) =="
# A file of (deftest …) with NO main: importing assert.coil registers a (transform …) that
# DISCOVERS every coil-test$… and synthesizes a forking main. FAILS on the seed (assert.coil
# not bundled -> the import errors).
cat > "$T/suite.coil" <<'EOF'
(module m)
(import "assert.coil" :use *)
(deftest passes (assert-eq (* 6 7) 42))
(deftest fails  (assert-eq (+ 1 1) 3))
(deftest tail   (assert (> 5 0)))
EOF
expect_out "running 3 tests"       "the transform discovers every test"                    "$COIL" run "$T/suite.coil"
expect_out "test passes \.\.\. ok" "a passing test reports ok"                             "$COIL" run "$T/suite.coil"
expect_out "test tail \.\.\. ok"   "the suite continues past a failing test (fork isolation)" "$COIL" run "$T/suite.coil"
expect_out "1 failed"              "the summary counts the failure"                        "$COIL" run "$T/suite.coil"
expect_rc 1  "a suite with a failing test exits 1"                                         "$COIL" run "$T/suite.coil"

echo "== coil test: the project test runner (tool-12) =="
# `coil test FILE` auto-loads assert.coil (--use), so a test file needs NO import at all,
# then runs the synthesized suite. FAILS on the seed ('unknown command test').
cat > "$T/noimp.coil" <<'EOF'
(module m)
(deftest a (assert-eq (* 3 3) 9))
(deftest b (assert (> 7 0)))
EOF
expect_rc 0  "coil test: an all-passing suite (no import needed) exits 0" "$COIL" test "$T/noimp.coil"
expect_out "2 passed; 0 failed" "coil test: reports the pass count"       "$COIL" test "$T/noimp.coil"
cat > "$T/redf.coil" <<'EOF'
(module m)
(deftest willfail (assert-eq (+ 2 2) 5))
EOF
expect_rc 1  "coil test: a failing suite exits 1"                         "$COIL" test "$T/redf.coil"
expect_out "0 passed; 1 failed" "coil test: reports the failure count"    "$COIL" test "$T/redf.coil"
expect_out "discovers every \(deftest" "coil test --help documents itself" "$COIL" test --help

echo "== -g: dsymutil runs, the .o is kept, lldb maps source (tool-11) =="
# The arm64 backend now stamps __text-relative RELOCATIONS on every DWARF address (CU +
# subprogram low_pc, the line program's set_address). WITHOUT them dsymutil printed "No
# valid relocations found. Skipping." and produced an EMPTY dSYM — 0 line rows, "No source
# available" in lldb. The driver also runs dsymutil after a -g link. FAILS on the seed
# (it emits no relocations and never runs dsymutil -> no .dSYM, breakpoints don't resolve).
if command -v dsymutil >/dev/null 2>&1; then
  cat > "$T/dbg.coil" <<'EOF'
(module dbg)
(defn addup [(x i64) (y i64)] (-> i64) (iadd x y))
(defn main [] (-> i64) (addup 3 4))
EOF
  "$COIL" build "$T/dbg.coil" -g -o "$T/dbgx" >/dev/null 2>&1
  [ -d "$T/dbgx.dSYM" ] && ok "coil build -g gathers a .dSYM" \
                        || bad "coil build -g gathers a .dSYM" "no .dSYM (dsymutil skipped or not run)"
  [ -e "$T/dbgx.o" ]    && ok "coil build -g keeps the .o (dsymutil reads it)" \
                        || bad "coil build -g keeps the .o" "the .o was removed"
  if command -v dwarfdump >/dev/null 2>&1; then
    n=$(dwarfdump --debug-line "$T/dbgx.dSYM/Contents/Resources/DWARF/dbgx" 2>/dev/null | grep -cE '^0x[0-9a-f]+ +[0-9]')
    [ "${n:-0}" -gt 0 ] && ok "the .dSYM line table has rows (was 0 = 'No source available')" \
                        || bad ".dSYM line rows" "0 rows — dsymutil skipped the object (no relocations)"
  fi
  # REMOVE the .o so lldb MUST use the .dSYM (the portable artifact) — the scenario the
  # finding describes. On the seed there is no .dSYM and now no .o -> lldb has nothing.
  rm -f "$T/dbgx.o"
  if command -v lldb >/dev/null 2>&1; then
    bp=$(lldb "$T/dbgx" -o "breakpoint set --file dbg.coil --line 3" -o quit 2>&1)
    echo "$bp" | grep -qE 'Breakpoint 1: where = .*dbg\.coil:3' \
      && ok "lldb maps source from the .dSYM alone (no .o)" \
      || bad "lldb maps source from the .dSYM" "$(echo "$bp" | grep -iE 'breakpoint|pending' | head -1)"
  fi
else
  echo "  skip — dsymutil not on PATH (not a macOS toolchain host)"
fi

echo "== comptime/const route through the compiled engine (mac-8; interp deletion step 1) =="
# The tree-walk interpreter is a strictly WEAKER sublanguage than a macro body: no
# generic calls, no sizeof/alignof/offsetof, no strings. Those sites now fold through
# the COMPILED metaprogram engine (real codegen) and the program runs. On the pre-mac-8
# seed the interpreter errors and `run` never yields the folded exit code — a tripwire.
printf '(defn main [] (-> i64) (comptime (sizeof i64)))\n'                                  > "$T/ct_sizeof.coil"
expect_rc 8 "comptime (sizeof i64) folds to 8 (interp can't)"        "$COIL" run "$T/ct_sizeof.coil"
printf '(defn id [T] [(x T)] (-> T) x)\n(defn main [] (-> i64) (comptime (id [i64] 7)))\n' > "$T/ct_generic.coil"
expect_rc 7 "comptime of a generic call folds to 7 (interp can't)"   "$COIL" run "$T/ct_generic.coil"
printf '(const SZ (sizeof i64))\n(defn main [] (-> i64) SZ)\n'                              > "$T/ct_const.coil"
expect_rc 8 "(const NAME (sizeof …)) folds to 8 (interp can't)"      "$COIL" run "$T/ct_const.coil"
printf '(defn main [] (-> i64) (comptime (do c"hi" 5)))\n'                                  > "$T/ct_str.coil"
expect_rc 5 "comptime using a c-string folds to 5 (interp can't)"    "$COIL" run "$T/ct_str.coil"
# NO REGRESSION: an aggregate comptime still folds (on the interpreter — the compiled
# engine declines the readback and the interpreter's own capability is preserved).
printf '(defstruct P [(x i64) (y i64)])\n(defn mk [] (-> P) (let [(mut p) (zeroed P)] (do (store! (field p x) 3) (store! (field p y) 4) (load p))))\n(defn main [] (-> i64) (let [q (comptime (mk))] (iadd (load (field q x)) (load (field q y)))))\n' > "$T/ct_agg.coil"
expect_rc 7 "aggregate comptime still folds (interp path, no regression)" "$COIL" run "$T/ct_agg.coil"

echo "== C size types are target-width: the prelude's size_t/ssize_t are i32 on wasm32 =="
# `usize`/`isize` (a C machine word: size_t/ssize_t/long) are i32 on wasm32 (ILP32) and
# i64 on LP64. The prelude's write/read/malloc size args use `isize`, so on wasm32 they
# emit i32 — matching the real C ABI. On a pre-usize compiler (the seed) they are i64.
# (Checked via emit-ir, not a full wasm build, because the C0/C1 wasm finalizer that lets
# such a module link is not on this branch — the LLVM IR is what carries the width.)
printf '(defn main [] (-> i64) (println "hi") 0)\n' > "$T/sizet.coil"
w_native=$("$COIL" emit-ir "$T/sizet.coil" 2>/dev/null | grep -oE 'declare i(64|32) @write\([^)]*\)' | head -1)
w_wasm=$("$COIL" emit-ir "$T/sizet.coil" --target wasm32-unknown-unknown 2>/dev/null | grep -oE 'declare i(64|32) @write\([^)]*\)' | head -1)
echo "$w_native" | grep -q 'i64 @write(i32, ptr, i64)' \
  && ok "write is (int fd=i32, ptr, size_t=i64) -> ssize_t=i64 on native"    || bad "native write width" "got: $w_native"
echo "$w_wasm"   | grep -q 'i32 @write(i32, ptr, i32)' \
  && ok "write's fd stays i32 and size_t/ssize_t narrow to i32 on wasm32" || bad "wasm32 write width (fd/usize/isize)" "got: $w_wasm"

echo "== A1: a wasm32 module exports __stack_pointer so a host longjmp can restore SP =="
# When the object uses the shadow stack (an alloc-stack address escapes to a C extern),
# the C0 finalizer defines the mutable i32 stack-pointer global. It must ALSO export it —
# otherwise a host-implemented longjmp/panic landing pad (wasm32 has no setjmp) cannot
# restore SP on unwind, and every panic strands the frames (measured ~111 B leaked/panic).
# FAILS on the seed (it exports only main/memory/__heap_base); PASSES here. Needs wasm-tools.
if command -v wasm-tools >/dev/null 2>&1; then
  printf '(extern sink :cc c [(ptr i8)] (-> void))\n(defstruct Big [(a i64) (b i64) (c i64)])\n(defn use-stack [(n i64)] (-> i64) (let [p (alloc-stack Big)] (do (store! (field p a) n) (sink (cast (ptr i8) p)) (load (field p a)))))\n(defn main [] (-> i64) (use-stack 3))\n' > "$T/sp.coil"
  if "$COIL" build "$T/sp.coil" --target wasm32-unknown-unknown -o "$T/sp.wasm" >/dev/null 2>&1; then
    sp_line=$(wasm-tools print "$T/sp.wasm" 2>/dev/null | grep '(export "__stack_pointer"')
    [ -n "$sp_line" ] && ok "wasm32 build exports __stack_pointer ($sp_line)" \
      || bad "wasm32 __stack_pointer export" "absent — host longjmp cannot restore SP"
  else
    bad "wasm32 __stack_pointer export" "shadow-stack build failed"
  fi
else
  echo "  (skip: wasm-tools not on PATH)"
fi

echo
[ "$FAIL" = 0 ] && echo "gate-cli: PASS" || echo "gate-cli: FAIL"
exit $FAIL
