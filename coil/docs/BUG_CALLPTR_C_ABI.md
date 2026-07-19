# `call-ptr` ignores the C ABI for small by-value structs

**Status:** FIXED 2026-07-18. `run.sh` exits 0. It was **two** bugs — the missing
ABI lowering described below, plus an out-of-bounds ABI spill slot (see
"What it actually took" at the end); fixing only the first left the 4-byte case
corrupt.
**Found:** 2026-07-18, extracting a render-agnostic widget library that reaches its
renderer through a `(fnptr c …)` vtable
**Backend:** LLVM (default). The arm64 backend is not affected, but rejects the
case outright.
**Severity:** silent data corruption at a language boundary, no diagnostic

## Symptom

A struct passed by value through `call-ptr` to a C function arrives corrupted
when it is smaller than 16 bytes. The same struct passed by a *direct*
`extern … :cc c` call is fine.

    call-ptr  4 bytes (4x u8)    16,223,40,109   expected 1,2,3,4    CORRUPT
    call-ptr  8 bytes (2x i32)   11,0            expected 11,22      CORRUPT
    call-ptr 16 bytes (2x i64)   111,222         expected 111,222    ok
    call-ptr 32 bytes (4x f64)   1,2,3,4         expected 1,2,3,4    ok
    direct    4 bytes (4x u8)    1,2,3,4         expected 1,2,3,4    ok

Nothing warns. The 4-byte case yields whatever happened to be in the register,
so it looks like uninitialised memory rather than an ABI fault; the 8-byte case
is nastier still, because field 0 survives and only field 1 is lost, which reads
as a logic bug in the caller.

## Reproduce

    docs/repro/callptr-c-abi/run.sh

Exits 0 once the ABI is honoured, 1 while the bug is live. Sources:
`docs/repro/callptr-c-abi/repro.coil` and `host.c` — one struct per size class
through a `(fnptr c …)` table, plus the 4-byte struct through a direct `extern`
as the control.

## Cause

`emit-callptr` never applies C-ABI argument lowering. It builds the call from
the raw parameter types:

`selfhost/src/codegen.coil:1153-1160`

```
(defn emit-callptr [(cg (ptr Cg)) (fp (ptr Expr)) (args …) (scp (ptr Scope))] (-> Tv)
  (let […
        cc-params-ret (match-else (tv-ty fpv) (TFn [c p r] (mk-fnsig c p r)) …)
        fnty (cg-fn-type cg (fnsig-params cc-params-ret) (fnsig-ret cc-params-ret))
        vals (emit-vals cg args scp)
        cs (LLVMBuildCall2 b fnty (tv-val fpv) …)]
    …))
```

The convention `c` is destructured out of `(TFn c p r)`, stored into the
`FnSig`, and then never read — `fnsig-cc` has zero uses in the tree. `fnty`
comes from `cg-fn-type` on the declared parameter types, and `vals` from
`emit-vals`, so no aggregate is ever classified, spilled, or re-loaded as
register slots.

The AAPCS64 rule that *should* apply lives in `abi-classify-aapcs64`
(`codegen.coil:1779-1795`): a non-HFA aggregate of ≤8 bytes becomes one `i64`
slot, ≤16 becomes `[2 x i64]`, and >16 goes `ACIndirect`. That path is reachable
only through `cg-csig` → `emit-c-call`, and `cg-csig` (`codegen.coil:1812`) is
keyed by **callee name**. An indirect call has no name, so it cannot participate.

The two paths diverge at `codegen.coil:832`, in `emit-call-native`:

```
(let [sig (cg-csig cg func)]
  (if (icmp-ne (cast i64 sig) 0)
      (emit-c-call cg callee sig (emit-argtv cg args scp) ret-ty)   ; direct: ABI-lowered
    …plain LLVMBuildCall2…))                                        ; call-ptr does this
```

This explains the exact size pattern. `fnptr-of` (`codegen.coil:1146-1152`)
returns the function as declared, and `defn`s with struct params *are* declared
with the C-lowered signature (`codegen.coil:2122-2130`). So the callee expects
packed `i64` slots while the call site passes the raw LLVM struct:

| size | call site passes | callee expects | outcome |
| --- | --- | --- | --- |
| 4 (4×u8) | `{i8,i8,i8,i8}` → w0–w3 | packed x0 | garbage |
| 8 (2×i32) | `{i32,i32}` → w0,w1 | packed x0 | field 0 survives, field 1 lost |
| 16 (2×i64) | `{i64,i64}` → x0,x1 | `[2 x i64]` → x0,x1 | agrees by coincidence |
| 32 | already memory/pointer | `ACIndirect` pointer | agrees by coincidence |

The two "ok" rows are not correct behaviour, only cases where the unlowered
form happens to match the lowered one.

## Suggested fix

Route `emit-callptr` through the existing machinery rather than duplicating it.
`c-signature` (`codegen.coil:1827`) already takes `params` / `ret` directly
rather than a name, so it can be called straight from the `(TFn c p r)` payload
— no registry lookup needed.

One friction point: `emit-c-call` (`codegen.coil:1953`) takes `callee (ptr i8)`
and derives two things from it, only at the very end —

```
(LLVMBuildCall2 b (LLVMGlobalGetValueType callee) callee …)
(LLVMSetInstructionCallConv cs (LLVMGetFunctionCallConv callee))
```

A raw function-pointer value has neither a global value type nor a function
call conv. So parameterise `emit-c-call` with an explicit `fnty` and cc id (or
split the body into an `emit-c-call-ty` that both callers share). Everything
else in it — classification, sret slot, `byval` copies, the `ACDirect` slot
loads, and `cg-apply-csig-attrs`, which already operates on a call instruction
rather than a function — is reused unchanged.

Roughly 30-40 lines in one file, no structural change.

## Two things worth deciding while in there

- **`fnsig-cc` is dead.** The convention on `(fnptr cc …)` is parsed and then
  discarded, so a non-`c` fnptr would silently receive whatever treatment the
  fix applies. Gate the new path on the convention actually being `c`.
- **`call-ptr` skips the C-ABI type guard.** Every extern is checked by
  `cg-check-c-abi-types` (`codegen.coil:1940`); `emit-callptr` never calls it,
  so a by-value `(slice u8)` through a `(fnptr c …)` slips past a rejection that
  the direct path would have raised.

## What it actually took

The analysis above is correct, and the suggested fix was applied close to as
written: `emit-c-call` split into an `emit-c-call-ty` that takes the callee's
function type and callconv explicitly, with `emit-callptr` calling `c-signature`
straight from the `(TFn c p r)` payload. `CSig` already carries its lowered
`fn_ty`, so nothing had to be synthesised. Both follow-ups are done — the path is
gated on the convention being `c` (`fnsig-cc` did not exist; it was added), and
`cg-check-c-abi-types` now runs on the indirect path.

That fixed the 8-byte case and left the 4-byte one corrupt, now reading
`0,0,0,0` rather than garbage. **A second, independent bug:** both sides of the
ABI spill an aggregate into an alloca sized to the *struct*, then move whole
`i64` slots through it. For any struct that is not a multiple of 8 that reads or
writes past the end of the object, and LLVM folds the out-of-bounds access to
poison. The disassembly showed it plainly — the argument register zeroed and the
incoming one never read:

    1c: aa1f03e0    mov  x0, xzr     ; the S4 argument
    3c: d63f0100    blr  x8

Two call sites needed the slot padded up to a whole number of 8-byte slots:
`cg-struct-arg-ptr` (slot *loads*, call side) and the `export-c` thunk
(slot *stores*, entry side).

This refines the size table above. 8, 16 and 32 bytes are already slot multiples,
so they never overran — exactly the rows that stayed green once the ABI lowering
landed. It also means `direct 4 bytes` was never a clean control: its thunk has
the same out-of-bounds store and happened to survive it.

One incidental note for anyone re-treading this: routing `c` fnptr calls through
the C path renames their SSA value, and *every* such call has one, including
scalar-only ones. The call label is threaded through `emit-c-call-ty` so the
indirect path keeps its historic `callptr` name — otherwise the IR gate reports
all 60 corpus files as changed and the one real diff is lost in the noise.

Regression coverage: `selfhost/oracle/features/export_c.coil` now passes a
4-byte struct through a `(fnptr c …)` held in a struct field, which pins the
padded slots, the packed `i64` argument, and the call name. It is IR-gate only —
the arm64 backend rejects this shape, so it is deliberately not in the arm64
behavioural corpus.

## Also noticed nearby

Unrelated to the ABI, but hit while producing this report:

- **`--emit-header` is broken.** `coil build x.coil --lib -o x.a --emit-header
  x.h` writes the archive, then fails with
  `(error@6:8416:8421:0 "unexpected keyword :done in expression")` and writes no
  header. Reproduces on a two-line library whose only export is
  `(defn add2 [(a i64) (b i64)] (-> i64) (+ a b))`, so it is not input-dependent.
- **`--backend arm64` rejects this shape entirely**, with a clear message:
  `export-c with a by-value struct parameter isn't supported by the arm64
  backend yet (it needs a C-ABI marshaling thunk); scalar/pointer params and
  struct returns work`. Worth noting the LLVM backend accepts the same program
  and miscompiles it instead of rejecting it.
