GPU VecAdd Surface Syntax → Verbose MLIR: Spec & Transform Plan

Status: draft design spec for the sugar layer and the exact, reproducible lowering back to the original verbose MLIR s‑expr form. This doc assumes @ (symbol sigil) and % (SSA sigil) are printer/parser syntax only and cannot be expressed or emulated by macros.

⸻

0) Goals & Non‑Goals

Goals
	•	Provide a compact, lispy surface syntax for MLIR that round‑trips byte‑for‑byte (modulo whitespace) to a chosen verbose s‑expr representation of MLIR ops.
	•	Keep sugar opt‑in and local: op remains the escape hatch for any dialect op.
	•	Precisely preserve: op order, region structure, types, attributes, visibilities, symbol names, and all SSA use‑def relationships.

Non‑Goals
	•	No semantic optimizations (no CSE, DCE, constant folding) in the desugarer.
	•	Do not rename user symbols. The printer re‑introduces @/% required by MLIR textual form, but names remain identical to the original.

⸻

1) Ground Truth (Original)

(mlir
  (op (builtin.module
    (region
      (block
        (arguments [])
        (operation
          (name func.func)
          (attributes {:function_type (!function (inputs memref<5xf32> memref<5xf32> memref<5xf32>) (results)) :sym_name @vecadd})
          (regions
            (region
              (block
                [^bb0]
                (arguments [(: %arg1 memref<5xf32>) (: %arg2 memref<5xf32>) (: %arg3 memref<5xf32>)])
                (constant %19 (: 0 index))
                (constant %20 (: 1 index))
                (constant %21 (: 5 index))
                (op (gpu.launch
                  {:workgroup_attributions (: 0 i64)}
                  [%20 %20 %20 %21 %20 %20]
                  (region
                    (block
                      [^bb0]
                      (arguments [(: %arg4 index) (: %arg5 index) (: %arg6 index) (: %arg7 index) (: %arg8 index) (: %arg9 index) (: %arg10 index) (: %arg11 index) (: %arg12 index) (: %arg13 index) (: %arg14 index) (: %arg15 index)])
                      (op %22 (: f32) (memref.load [%arg1 %arg7]))
                      (op %23 (: f32) (memref.load [%arg2 %arg7]))
                      (op %24 (: f32) (arith.addf {:fastmath #arith.fastmath<none>} [%22 %23]))
                      (op (memref.store [%24 %arg3 %arg7]))
                      (op (gpu.terminator))))))
                (operation (name func.return)))))
        (operation
          (name func.func)
          (attributes {:function_type (!function (inputs) (results)) :sym_name @main})
          (regions
            (region
              (block
                (arguments [])
                (constant %0 (: 0 index))
                (constant %1 (: 1 index))
                (constant %2 (: 5 index))
                (constant %3 (: 1.230000e+00 f32))
                (op %4 (: memref<5xf32>) (memref.alloc))
                (op %5 (: memref<5xf32>) (memref.alloc))
                (op %6 (: memref<5xf32>) (memref.alloc))
                (op %7 (: memref<?xf32>) (memref.cast [%4]))
                (op %8 (: memref<?xf32>) (memref.cast [%5]))
                (op %9 (: memref<?xf32>) (memref.cast [%6]))
                (op (scf.for
                  [%0 %2 %1]
                  (region
                    (block
                      [^bb0]
                      (arguments [(: %arg0 index)])
                      (op (memref.store [%3 %7 %arg0]))
                      (op (memref.store [%3 %8 %arg0]))
                      (op (scf.yield))))))
                (op %10 (: memref<*xf32>) (memref.cast [%7]))
                (op %11 (: memref<*xf32>) (memref.cast [%8]))
                (op %12 (: memref<*xf32>) (memref.cast [%9]))
                (op (gpu.host_register [%10]))
                (op (gpu.host_register [%11]))
                (op (gpu.host_register [%12]))
                (call %13 @mgpuMemGetDeviceMemRef1dFloat %7 memref<?xf32>)
                (call %14 @mgpuMemGetDeviceMemRef1dFloat %8 memref<?xf32>)
                (call %15 @mgpuMemGetDeviceMemRef1dFloat %9 memref<?xf32>)
                (op %16 (: memref<5xf32>) (memref.cast [%13]))
                (op %17 (: memref<5xf32>) (memref.cast [%14]))
                (op %18 (: memref<5xf32>) (memref.cast [%15]))
                (call @vecadd %16 %17 %18 ())
                (call @printMemrefF32 %12 ())
                (operation (name func.return)))))
        (operation
          (name func.func)
          (attributes {:function_type (!function (inputs memref<?xf32>) (results memref<?xf32>)) :sym_name @mgpuMemGetDeviceMemRef1dFloat :sym_visibility "private"})
          (regions (region)))
        (operation
          (name func.func)
          (attributes {:function_type (!function (inputs memref<*xf32>) (results)) :sym_name @printMemrefF32 :sym_visibility "private"})
          (regions (region)))))))


⸻

2) Revised Surface Syntax (no %/@, multi‑bind let, explicit op)

(declare ^{:visibility private} mgpuMemGetDeviceMemRef1dFloat (-> (memref<?xf32>) (memref<?xf32>)))
(declare ^{:visibility private} printMemrefF32                (-> (memref<*xf32>) ()))

(defn vecadd [(arg1 memref<5xf32>) (arg2 memref<5xf32>) (arg3 memref<5xf32>)] (-> () ())
  (op (gpu.launch
        {:workgroup_attributions (: 0 i64)}
        [1 1 1 5 1 1]
        (region
          (block
            (arguments [(arg4 index) (arg5 index) (arg6 index) (arg7 index)
                        (arg8 index) (arg9 index) (arg10 index) (arg11 index)
                        (arg12 index) (arg13 index) (arg14 index) (arg15 index)])
            (op x (: f32) (memref.load [arg1 arg7]))
            (op y (: f32) (memref.load [arg2 arg7]))
            (op z (: f32) (arith.addf {:fastmath #arith.fastmath<none>} [x y]))
            (op (memref.store [z arg3 arg7]))
            (op (gpu.terminator)))))))

(defn main [] (-> () ())
  (let [A? (memref.cast (memref.alloc memref<5xf32>) memref<?xf32>)
        B? (memref.cast (memref.alloc memref<5xf32>) memref<?xf32>)
        C? (memref.cast (memref.alloc memref<5xf32>) memref<?xf32>)
        A* (memref.cast A? memref<*xf32>)
        B* (memref.cast B? memref<*xf32>)
        C* (memref.cast C? memref<*xf32>)]
    (for [i 0 5 1]
      (memref.store 1.23 A? [i])
      (memref.store 1.23 B? [i]))
    (gpu.host-register A*)
    (gpu.host-register B*)
    (gpu.host-register C*)
    (vecadd
      (memref.cast (mgpuMemGetDeviceMemRef1dFloat A? (memref<?xf32>)) memref<5xf32>)
      (memref.cast (mgpuMemGetDeviceMemRef1dFloat B? (memref<?xf32>)) memref<5xf32>)
      (memref.cast (mgpuMemGetDeviceMemRef1dFloat C? (memref<?xf32>)) memref<5xf32>))
    (printMemrefF32 C* ())))


⸻

3) Syntax/Printing Model for @ and %

Key constraint: @ (symbols) and % (SSA values) are not macros. They are lexical markers of the textual layer and must be handled by the parser/printer, not the macro expander.

3.1 Internal IDs
	•	Values carry opaque IDs v# and a user‑preferred name (optional). Printer renders as %name or %# if unnamed.
	•	Symbols carry global IDs s# and a user name; printer renders as @name.

3.2 Parser
	•	Accepts both sigiled and unsigiled names in the surface grammar. During parse, assign internal IDs and record the exact source name (without sigils).

3.3 Printer
	•	Re‑inserts % for SSA uses/defs and @ for symbol references/definitions.
	•	Preserves original user spelling where available. No alpha‑renaming.

⸻

4) Macro Layer (Pure S‑expression Rewrites)

The following are context‑free and can be implemented as ordinary lisp macros:
	•	Multi‑bind let
	•	(let [b1 e1 b2 e2 … bn en] body…) → nested single‑bind lets left‑to‑right.
	•	for sugar
	•	(for [iv lb ub step] body…) → scf.for with a region argument iv and trailing scf.yield if body lacks a terminator.
	•	defn → func.func
	•	Construct !function type from arguments and (-> (ins) (outs)) signature.
	•	Append func.return if last op is not a terminator.
	•	declare
	•	(declare ^{:visibility v} name (-> (Tin…) (Tout…))) → one func.func declaration with :sym_visibility v.
	•	Literal shorthands
	•	Numeric/float literals in operand position expand to (constant %tmp (: N index)) or (: 1.23 f32) only when an SSA value is required by the consumer op (see §6.2 policy). Otherwise they print inline.
	•	Call sugar
	•	(callee args…) → MLIR call op with explicit results if required by the context; printer re‑inserts @.

Out of scope for macros: SSA formation, symbol table management, dominance, launch ABI packing. Those require IR context.

⸻

5) Proper Transforms (IR‑Aware)

These require analysis/state beyond local s‑expr rewriting:

5.1 SSA/Dominance & Value Materialization
	•	Insert constant ops for numeric operands when an SSA value is required by the target op (e.g., gpu.launch operand pack, loop bounds).
	•	Duplication policy: to preserve the original structure (e.g., multiple uses of %20), allow identical constants to be emitted separately when needed. No CSE.
	•	Ensure all uses dominate their defs.

5.2 Symbol Table & Declarations
	•	Build module‑level symbol table from declare and defn.
	•	Emit exactly one func.func declaration per (declare …) with :sym_visibility from metadata.
	•	Validate that call sites/defs agree on function types.

5.3 GPU Launch ABI
	•	Map the surface launch vector [1 1 1 5 1 1] to the operand segment the dialect expects. In the original, %20 %20 %20 %21 %20 %20 corresponds to [1 1 1 5 1 1] materialized by constants.
	•	Materialize the 12 region args in the exact order used in the original: arg4 … arg15.
	•	Insert gpu.terminator in the launch body.

5.4 Cast Chain Legality
	•	Preserve the precise sequence: memref<5xf32> → memref<?xf32> → memref<*xf32> and back, matching consumer requirements (gpu.host_register expects *, vecadd expects 5xf32).

5.5 One‑Use Substitution
	•	Inline single‑use temporaries during macro expansion only within the same block and if dominance is preserved.
	•	Never inline across region boundaries or if it would change operand segment sizes/types.

5.6 Terminators
	•	Ensure: func.return at end of each function body; scf.yield present; gpu.terminator present.

⸻

6) Deterministic Lowering Algorithm

Input: Surface AST (no %/@), module with forms: declare/defn/op/…
Output: Verbose MLIR s-expr (with %/@) matching §1 structure

1. Parse → Internal IR
   - Assign internal SSA IDs and symbol IDs; record user names.

2. Expand Macros (pure): let/for/defn/declare/call literals
   - Produce only well-typed op skeletons; do not alter names.

3. Materialize Required Values
   - For each operand that requires SSA, if source is a literal, insert a `constant` (new SSA def) immediately before first use; reuse only when the surface explicitly shares a name. Do not CSE.

4. Insert Required Terminators
   - Add func.return / scf.yield / gpu.terminator where missing.

5. GPU Launch ABI
   - Pack the 6 launch integers into the operand list; if they originate as literals, ensure step 3 emitted distinct constants so they print as `%20 …` as in §1.
   - Populate 12 region args in the specified order.

6. Cast Chain Enforcement
   - Verify each consumer’s type and insert/retain casts to match exactly the original path.

7. Declarations & Symbol Visibility
   - Emit `func.func` decls for every `declare` exactly once; set `:sym_visibility` from metadata.

8. Name/ID Stabilization
   - Freeze SSA def order to match original; preserve user spellings. Do not alpha-rename.

9. Print
   - Re-insert `%` for SSA and `@` for symbols; print attributes and types verbatim.


⸻

7) What’s Macro vs Transform?

Macros (simple):
	•	let (multi-bind → nested)
	•	for → scf.for
	•	defn header → func.func + implicit func.return
	•	declare → func.func decl with metadata→attributes
	•	Call surface sugar → call

Transforms (proper):
	•	SSA materialization of literals (constants), duplication policy
	•	Dominance enforcement
	•	Symbol table + visibility plumbing (cannot be a macro)
	•	GPU launch operand packing & region arg arity/order
	•	Cast legality checks and insertions
	•	Terminator insertion in non-macro contexts
	•	Name/ID stabilization for byte‑accurate round‑trip

⸻

8) Worked Micro‑Diffs

A) declare

Surface:

(declare ^{:visibility private} mgpuMemGetDeviceMemRef1dFloat (-> (memref<?xf32>) (memref<?xf32>)))

Verbose:

(operation
  (name func.func)
  (attributes {:function_type (!function (inputs memref<?xf32>) (results memref<?xf32>))
               :sym_name @mgpuMemGetDeviceMemRef1dFloat
               :sym_visibility "private"})
  (regions (region)))

B) for

Surface:

(for [i 0 5 1]
  (memref.store 1.23 A? [i]))

Verbose:

(op (scf.for [ %0 %2 %1 ]
      (region (block [^bb0]
        (arguments [(: %arg0 index)])
        (op (memref.store [%3 %7 %arg0]))
        (op (scf.yield))))))

(Exact % numbers depend on global numbering; see §6.3.)

C) gpu.launch operands

Surface:

(op (gpu.launch {...} [1 1 1 5 1 1] ...))

Transform:
	•	Insert three %const_1 and one %const_5 as distinct constant ops before the launch to match %20/%21 reuse pattern.
	•	Feed them in order to the operand pack.

⸻

9) Type Rules (enforced during transform)
	•	memref.load [m idx] → result type = element type of m.
	•	memref.store [v m idx] → v type == element type of m.
	•	arith.addf → operands/results f32 here; no fastmath changes.
	•	Casts:
	•	memref.cast must be layout-compatible; the exact chain here is allowed and must be preserved.

⸻

10) Diagnostics
	•	Missing declaration for a called symbol → error with symbol name and site.
	•	Mismatched function type at call vs declaration/def → error with diff.
	•	Bad launch vector length (≠6) or bad region arg count (≠12) → error.
	•	Dominance violation after inlining one-use → error pointing to offending op.

⸻

11) Testing Strategy
	•	Golden round‑trip tests: surface → verbose → parse → print equals §1.
	•	Property tests: generate random let/call/cast chains; assert printer stability and SSA well-formedness.
	•	GPU ABI tests: verify operand/arg ordering using small kernels.

⸻

12) Open Questions
	•	Do we need a knob to force reusing the same constant SSA across multiple operands (to model %20 reuse) vs duplicating? Current plan prefers duplicating only when structure demands it.
	•	Should the surface allow an explicit symbol namespace form (sym vecadd) to pin user spelling independent of defn name?

⸻

13) Final Surface Snippet (authoritative)

(declare ^{:visibility private} mgpuMemGetDeviceMemRef1dFloat (-> (memref<?xf32>) (memref<?xf32>)))
(declare ^{:visibility private} printMemrefF32                (-> (memref<*xf32>) ()))

(defn vecadd [(arg1 memref<5xf32>) (arg2 memref<5xf32>) (arg3 memref<5xf32>)] (-> () ())
  (op (gpu.launch {:workgroup_attributions (: 0 i64)} [1 1 1 5 1 1]
        (region (block
          (arguments [(arg4 index) (arg5 index) (arg6 index) (arg7 index)
                      (arg8 index) (arg9 index) (arg10 index) (arg11 index)
                      (arg12 index) (arg13 index) (arg14 index) (arg15 index)])
          (op x (: f32) (memref.load [arg1 arg7]))
          (op y (: f32) (memref.load [arg2 arg7]))
          (op z (: f32) (arith.addf {:fastmath #arith.fastmath<none>} [x y]))
          (op (memref.store [z arg3 arg7]))
          (op (gpu.terminator)))))))

(defn main [] (-> () ())
  (let [A? (memref.cast (memref.alloc memref<5xf32>) memref<?xf32>)
        B? (memref.cast (memref.alloc memref<5xf32>) memref<?xf32>)
        C? (memref.cast (memref.alloc memref<5xf32>) memref<?xf32>)
        A* (memref.cast A? memref<*xf32>)
        B* (memref.cast B? memref<*xf32>)
        C* (memref.cast C? memref<*xf32>)]
    (for [i 0 5 1]
      (memref.store 1.23 A? [i])
      (memref.store 1.23 B? [i]))
    (gpu.host-register A*)
    (gpu.host-register B*)
    (gpu.host-register C*)
    (vecadd
      (memref.cast (mgpuMemGetDeviceMemRef1dFloat A? (memref<?xf32>)) memref<5xf32>)
      (memref.cast (mgpuMemGetDeviceMemRef1dFloat B? (memref<?xf32>)) memref<5xf32>)
      (memref.cast (mgpuMemGetDeviceMemRef1dFloat C? (memref<?xf32>)) memref<5xf32>))
    (printMemrefF32 C* ())))