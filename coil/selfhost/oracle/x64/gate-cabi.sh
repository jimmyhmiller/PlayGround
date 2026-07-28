#!/usr/bin/env bash
# C-ABI gate for the x86-64 backend: the SysV eightbyte rules, checked against
# a real gcc-compiled translation unit.
#
# Why this gate exists and the corpus cannot replace it. For calls between Coil
# functions this backend passes aggregates BY POINTER, so the eightbyte
# classification — which register a struct's halves travel in, when the whole
# thing spills to the stack, how a mixed int/float struct splits — is only ever
# exercised at the C boundary. A Coil-only test cannot reach it. So here the
# CALLEE is compiled by gcc and the CALLER by coil (and vice versa): if the two
# disagree about the ABI by even one register, the numbers come back wrong.
#
# Every case uses DISTINCT PRIME WEIGHTS per field. That is deliberate: the
# first version of this test summed the fields with equal weight, so a backend
# that swapped two arguments still produced the right total and the bug walked
# straight through. With distinct weights any permutation changes the answer.
#
# Usage: selfhost/oracle/x64/gate-cabi.sh <coil-binary>
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN=${1:?usage: gate-cabi.sh <coil-binary>}
[ -x "$BIN" ] || { echo "GATE FAIL: binary not executable: $BIN"; exit 2; }
command -v cc >/dev/null 2>&1 || { echo "GATE FAIL: no cc"; exit 2; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

# ---------------- the C side ----------------
cat > "$WORK/cabi.c" <<'CEOF'
#include <stdint.h>
typedef struct { long a, b; }            P2;    /* INTEGER,INTEGER      */
typedef struct { double x, y; }          F2;    /* SSE,SSE              */
typedef struct { double x; long n; }     MIX;   /* SSE,INTEGER          */
typedef struct { float a, b; long n; }   PACK;  /* SSE(2xf32),INTEGER   */
typedef struct { long a,b,c,d; }         BIG;   /* MEMORY (>16B)        */
typedef struct { long a; }               P1;    /* one INTEGER eightbyte*/

/* 16B INTEGER pair in registers (rdi:rsi), trailing int in rdx */
long c_pair_regs(P2 p, long g)          { return p.a + p.b*7 + g*31; }
/* six ints exhaust the GPRs, so the pair SPILLS whole and g follows it */
long c_pair_spill(long a,long b,long c,long d,long e,long f,P2 p,long g)
                                        { return a+b*2+c*3+d*5+e*7+f*11 + p.a*13 + p.b*17 + g*19; }
/* two SSE eightbytes in xmm0:xmm1 */
double c_f2_regs(F2 q, double z)        { return q.x + q.y*7.0 + z*31.0; }
/* seven doubles leave one xmm free, so the SSE pair cannot fit and spills */
double c_f2_spill(double a,double b,double c,double d,double e,double f,double g,F2 q,long n)
                                        { return a+b*2+c*3+d*5+e*7+f*11+g*13 + q.x*17 + q.y*19 + (double)n*23; }
/* one eightbyte per register file: xmm0 and rdi */
long c_mix(MIX m, long k)               { return (long)m.x + m.n*7 + k*31; }
/* two floats packed into ONE SSE eightbyte, then an INTEGER eightbyte */
long c_pack(PACK p, long k)             { return (long)p.a + (long)p.b*7 + p.n*31 + k*97; }
/* >16B: memory in, memory out (hidden pointer) */
BIG  c_big_ret(long n)                  { BIG r = {n, n+1, n+2, n+3}; return r; }
long c_big_arg(BIG v, long k)           { return v.a + v.b*7 + v.c*31 + v.d*97 + k*193; }
/* small aggregate returned in one register */
P1   c_p1_ret(long n)                   { P1 r = {n*3}; return r; }
P2   c_p2_ret(long n)                   { P2 r = {n, n*2}; return r; }
F2   c_f2_ret(double v)                 { F2 r = {v, v*2}; return r; }
MIX  c_mix_ret(double v, long n)        { MIX r = {v, n}; return r; }
/* variadic: al must carry the SSE count or glibc reads the wrong registers */
long c_variadic(long n, ...)            { return n; }

/* NOTE: this gate checks the CALLER side only (coil calls gcc). Two reasons the
   callee side is not here: the metaprogram dylib inherits the program's
   --link-flag list, so a C object with an undefined symbol fails that dylib's
   link before the program runs; and a Coil `defn` receives a struct parameter
   as a REFERENCE (internal calls pass aggregates by pointer), so a Coil
   function cannot express a by-value struct callee at all. The caller side is
   where the eightbyte classification actually lives in this backend. */
CEOF
cc -c -O0 "$WORK/cabi.c" -o "$WORK/cabi.o" 2>"$WORK/cc.log" \
  || { echo "GATE FAIL: could not compile the C side"; cat "$WORK/cc.log"; exit 2; }

# ---------------- the Coil side ----------------
cat > "$WORK/cabi.coil" <<'KEOF'
(module cabitest)
(import "io.coil" :use *)
(import "control.coil" :use *)

(defstruct P2 [(a i64) (b i64)])
(defstruct F2 [(x f64) (y f64)])
(defstruct MIX [(x f64) (n i64)])
(defstruct PACK [(a f32) (b f32) (n i64)])
(defstruct BIG [(a i64) (b i64) (c i64) (d i64)])
(defstruct P1 [(a i64)])

(extern c_pair_regs  :cc c [P2 i64] (-> i64))
(extern c_pair_spill :cc c [i64 i64 i64 i64 i64 i64 P2 i64] (-> i64))
(extern c_f2_regs    :cc c [F2 f64] (-> f64))
(extern c_f2_spill   :cc c [f64 f64 f64 f64 f64 f64 f64 F2 i64] (-> f64))
(extern c_mix        :cc c [MIX i64] (-> i64))
(extern c_pack       :cc c [PACK i64] (-> i64))
(extern c_big_ret    :cc c [i64] (-> BIG))
(extern c_big_arg    :cc c [BIG i64] (-> i64))
(extern c_p1_ret     :cc c [i64] (-> P1))
(extern c_p2_ret     :cc c [i64] (-> P2))
(extern c_f2_ret     :cc c [f64] (-> F2))
(extern c_mix_ret    :cc c [f64 i64] (-> MIX))

(defn note! [(fb (ptr i64)) (n i64) (got i64) (want i64)] (-> i64)
  (unless (icmp-eq got want)
          (print-str (stderr) "cabi: check ")
          (print-int (stderr) n)
          (print-str (stderr) " got ")
          (print-int (stderr) got)
          (print-str (stderr) " want ")
          (print-int (stderr) want)
          (print-str (stderr) "\n")
          (when (icmp-eq (load fb) 0) (store! fb n) 0)
          0)
  0)

(defn main [] (-> i64)
  (let [w (stdout)
        fb (alloc-stack i64)
        p (alloc-stack P2)
        q (alloc-stack F2)
        m (alloc-stack MIX)
        k (alloc-stack PACK)]
    (store! fb 0)
    (store! (field p a) 100) (store! (field p b) 200)
    (store! (field q x) 3.0) (store! (field q y) 5.0)
    (store! (field m x) 2.0) (store! (field m n) 5)
    (store! (field k a) (cast f32 1.0)) (store! (field k b) (cast f32 2.0)) (store! (field k n) 4)

    ; 100 + 200*7 + 7*31 = 100 + 1400 + 217 = 1717
    (note! fb 1 (c_pair_regs (load p) 7) 1717)
    ; 1+4+9+20+35+66 = 135; 100*13 + 200*17 + 7*19 = 1300+3400+133 = 4833 -> 4968
    (note! fb 2 (c_pair_spill 1 2 3 4 5 6 (load p) 7) 4968)
    ; 3 + 5*7 + 2*31 = 3 + 35 + 62 = 100
    (note! fb 3 (cast i64 (c_f2_regs (load q) 2.0)) 100)
    ; 1+4+9+20+35+66+91 = 226; 3*17 + 5*19 + 2*23 = 51+95+46 = 192 -> 418
    (note! fb 4 (cast i64 (c_f2_spill 1.0 2.0 3.0 4.0 5.0 6.0 7.0 (load q) 2)) 418)
    ; 2 + 5*7 + 3*31 = 2 + 35 + 93 = 130
    (note! fb 5 (c_mix (load m) 3) 130)
    ; 1 + 2*7 + 4*31 + 2*97 = 1 + 14 + 124 + 194 = 333
    (note! fb 6 (c_pack (load k) 2) 333)
    ; c_big_ret 10 -> {10,11,12,13}; 10 + 11*7 + 12*31 + 13*97 + 2*193 = 10+77+372+1261+386
    (let [v (c_big_ret 10)]
      (note! fb 7 (c_big_arg v 2) 2106))
    (let [r1 (c_p1_ret 5)]
      (note! fb 8 (load (field r1 a)) 15))
    (let [r (c_p2_ret 9)]
      (note! fb 9 (iadd (load (field r a)) (imul (load (field r b)) 7)) 135))
    (let [r (c_f2_ret 1.5)]
      (note! fb 10 (cast i64 (fadd (load (field r x)) (fmul (load (field r y)) 7.0))) 22))
    (let [r (c_mix_ret 4.0 6)]
      (note! fb 11 (iadd (cast i64 (load (field r x))) (imul (load (field r n)) 7)) 46))
    (if (icmp-eq (load fb) 0)
        (do (print-str w "cabi: all checks passed\n") 0)
        (load fb))))
KEOF

fail=0
run_one() {  # run_one <label> <backend-args...>
  local label="$1"; shift
  if ! "$BIN" build "$WORK/cabi.coil" -o "$WORK/t" "$@" --link-flag "$WORK/cabi.o" \
        >"$WORK/build.log" 2>&1; then
    echo "  FAIL — $label: build failed"; head -5 "$WORK/build.log"; fail=$((fail+1)); return
  fi
  out=$("$WORK/t" 2>&1); rc=$?
  if [ "$rc" = 0 ]; then echo "  ok   — $label"
  else echo "  FAIL — $label (exit $rc)"; printf '%s\n' "$out" | head -6; fail=$((fail+1)); fi
}

# The x64 backend is what this gate protects.
run_one "x64 backend   <-> gcc" --backend x64

# The LLVM backend is run as a CONTROL — normally it should agree with gcc too,
# which is what proves the expected values are right and not merely
# self-consistent. It currently fails ONE case, and that failure is real but
# pre-existing and NOT this backend's:
#
#   check 4 (c_f2_spill): a {double,double} struct passed after seven doubles.
#   codegen.coil flattens the struct into two separate `double` parameters
#   (see `emit-ir`: "double %abi.slot, double %abi.slot2"). SysV says the
#   STRUCT spills to the stack as a unit once both its eightbytes cannot fit,
#   but two independent doubles put the first in the last free xmm and only
#   spill the second — so the callee reads qx=5, qy=0 instead of qx=3, qy=5.
#   Verified against gcc directly: gcc says 418, the LLVM path says 357, and
#   the x64 backend says 418.
#
# So the control is reported but does not fail the gate. Drop this allowance
# the moment codegen.coil stops flattening aggregates at the C boundary.
LLVM_KNOWN_BAD=4
run_llvm_control() {
  if "$BIN" build "$WORK/cabi.coil" -o "$WORK/tl" --link-flag "$WORK/cabi.o" >/dev/null 2>&1; then
    "$WORK/tl" >/dev/null 2>&1; rc=$?
    if [ "$rc" = 0 ]; then echo "  ok   — LLVM backend <-> gcc (control)"
    elif [ "$rc" = "$LLVM_KNOWN_BAD" ]; then
      echo "  known — LLVM backend <-> gcc fails check $rc (pre-existing aggregate-flattening bug, see above)"
    else
      echo "  FAIL — LLVM backend <-> gcc failed check $rc, which is NOT the known one"; fail=$((fail+1))
    fi
  else
    echo "  FAIL — LLVM backend <-> gcc: build failed"; fail=$((fail+1))
  fi
}
run_llvm_control

echo
[ "$fail" = 0 ] && { echo "x64 gate-cabi: PASS"; exit 0; } || { echo "x64 gate-cabi: $fail case(s) FAILED"; exit 1; }
