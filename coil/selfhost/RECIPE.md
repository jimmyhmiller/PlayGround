# Self-hosting Coil — the pass recipe (anti-hack)

Goal: port the Coil compiler (`src/*.rs`) into Coil, pass by pass, so it compiles
itself. The owner has been burned by agents taking HACKY shortcuts on self-host
ports. This document encodes the method that makes shortcuts **mechanically
impossible to land**. Follow it exactly. The reader pass is the worked example:
`selfhost/reader.coil` + `selfhost/oracle/` — study it before doing a new pass.

## Invariant rules (never violate)
1. **The gate is the contract.** A pass is correct iff its self-host output is
   BYTE-IDENTICAL to the Rust reference across the whole corpus. Not "looks right" —
   identical. The Rust compiler is the spec.
2. **Hard-error stubs only.** Any unimplemented function must `(unimplemented "id")`
   (prints to stderr + `(abort)`, exit 134). NEVER return a fake/zero/sentinel value
   that the gate might accept. A stub that silently returns a plausible value is the
   exact hack we forbid.
3. **Faithful port.** Mirror the Rust function structure. No "simplified" rewrite —
   "I simplified it" = "I dropped a feature the corpus exercises." The Rust source
   is the line-by-line spec.
4. **No overfit, no tampering.** Never special-case corpus filenames. Never edit the
   reference snapshot, the gate script, or the Rust reference to make a diff pass.
   The reader must be genuinely correct for ARBITRARY input.
5. **Verify teeth.** A gate that only ever passes is worthless. Before trusting a
   gate, prove it FAILS on a deliberately-wrong implementation (drop a field, change
   a number). The reader gate fails 84/86 on a span-dropping reader — that's teeth.
6. **If blocked, STOP and report.** Don't hack around a missing Coil feature. Leave
   the hard-error stub, report precisely what's blocking, the current gate count,
   and what you tried.

## The 5-step recipe (per pass)
1. **Oracle.** Add a canonical, lossless, deterministic DUMP of the pass's output to
   the Rust reference (a new `coil dump-<pass>` subcommand: reader→`dump-read`,
   parser→`dump-ast`, …). Make formatting un-divergeable: numbers/bits explicit,
   strings via a fixed per-byte escape, errors dumped in the same canonical shape.
2. **Corpus + gate.** Snapshot the Rust dump over the whole corpus into
   `selfhost/oracle/<pass>/reference/`, plus curated error/edge fixtures. Write a
   `gate-<pass>.sh <coil-self-bin>` that diffs self-host vs reference across all of
   it; exit 0 iff byte-identical everywhere. PROVE it has teeth.
3. **Scaffold.** Write `selfhost/<pass>.coil`: the data types (mirror the Rust ones),
   the deterministic dumper + IO + `main` (glue you write AND verify yourself — the
   dumper must be byte-exact; smoke-test it on a hand-built value), and the actual
   ported logic as `(unimplemented "…")` hard-error stubs. It must COMPILE in stub
   state; the gate then reports 0 pass.
4. **Fill.** Implement the stubs faithfully from the Rust spec. Build, run the gate
   with VERBOSE diffs, fix the first divergence, repeat until ALL pass.
5. **Verify independently.** Clean rebuild + gate; regenerate the reference from Rust
   truth + re-gate (catch doctored snapshots); grep the gate for filename
   special-casing; diff Rust-vs-self on NOVEL inputs not in the corpus (catch
   overfit). Only then is the pass done. Commit it.

## Pass order + each pass's oracle
- **reader** — DONE. text → `Vec<Sexp>`. `coil dump-read`. 86/86 verified.
- **parser** — `&[Sexp] → Program` via `parse::parse_program` (standalone, no
  resolve/check). Oracle `coil dump-ast` = read → parse_program → canonical dump of
  the whole `Program` (Func/StructDef/SumDef/TraitDef/ImplDef/Extern/Const/Static +
  the 48-variant `ExprKind` and 15-variant `Type`). CORPUS = the `coil expand` output
  of each real .coil file (69/73 expand cleanly; exclude+LOG the rest), because the
  parser consumes POST-MACRO core forms. The self-host parses the SAME expanded text,
  so no macro expander is needed yet and both sides parse identical input. Types live
  in `src/ast.rs`; logic in `src/parse.rs`.
- **resolve** — `Program → Program` (name resolution). Oracle dumps the resolved
  program. `src/resolve.rs`.
- **check** — typed AST. The big one (`src/check.rs`, 3041 lines: traits, generics,
  inference). Oracle dumps the checked/typed program.
- **mono** — monomorphized program. `src/mono.rs`. Oracle dumps it.
- **codegen** — LLVM IR. Owner's choice: emit via **LLVM-C API through Coil FFI**
  (build the module with LLVMBuild*, then `LLVMPrintModuleToString` for the gate).
  Gate diffs that text vs `coil emit-ir` (normalize SSA temp names/block labels if
  needed). `src/codegen.rs` + `src/abi.rs`.
- **bootstrap fixpoint** — FINALE. `coil` builds the self-host compiler → stage1;
  stage1 compiles its own source → stage2; stage2 → stage3. stage2 == stage3
  byte-identical = self-hosting proven.

## Coil cheatsheet (learned porting the reader)
- Low-level Lisp. Study `lib/sexp.coil`, `lib/str.coil`, `lib/slice.coil`,
  `lib/arraylist.coil`, `lib/result.coil`, `lib/hashmap.coil`, `examples/json.coil`,
  `examples/calc.coil`, and `selfhost/reader.coil` (the worked example).
- `defn`/`let` bodies allow MULTIPLE forms (implicit `do`); `if`/`loop` bodies take
  ONE form (wrap multiples in `(do …)`). Both `if` branches must have the SAME type,
  even when the value is discarded inside a `do`.
- Int ops: `iadd isub imul idiv irem udiv urem iand ior ixor ishl ishr`. Compares:
  `icmp-eq/ne/lt/le/gt/ge` (+ unsigned via casts). LOGICAL on bools: `and or not`.
  Bitwise-and is `iand` NOT `and`. Shifts are `ishl`/`ishr` (`lshr`/`shr` are
  LLVM-IR-only, not Coil ops).
- Memory: `(alloc-stack T)`, `(field p name)`, `(index ptr i)`, `(load place)`,
  `(store! place val)`, `(cast T expr)`. Allocator threaded as `(ptr Allocator)`;
  `(malloc-allocator)` is a static singleton.
- Recursion in sums goes through `(ptr (ArrayList X))` (see json.coil/Json,
  reader.coil/Sexp). Heap-box: `(unwrap-ptr [T] (create [T] a))` then `store!`.
- ArrayList: `(al-new [T] a)`, `(al-push! (mut xs) v)` (infers T), `(al-get [T] l i)`,
  `(al-len [T] l)`, `(al-slice [T] l)`. `(mut x)` local = slot: `(load x)`/`(store! x v)`.
- Strings = `(slice u8)`: `(slice-len s)`, `(slice-get s i)`→u8, `(subslice s lo hi)`.
  Literal `"…"` is `(slice u8)`. HashMap with string keys: `(str-keyops)`.
- Sums/match: `(match v (Variant [binds] body) …)`. `Ok/Err/Some/None`. `(None [T])`
  when T can't be inferred.
- Error propagation: `lib/try.coil` — `(try …)`, `(try! e)` unwrap-or-Err, `(try? e)`.
- Gotchas hit on the reader: `raw-alloc` returns `(Option (ptr i8))` so use
  `(cast (ptr u8) (unwrap-ptr [i8] …))`; generic calls need explicit `[T]` when
  inference gives up; `if` both-branch type unification bites when discarding values.
- After edits: `paredit-like balance <file> --in-place`, then build.

## Build/run
- Reference compiler: `cargo build` → `./target/debug/coil`.
- Build a self-host pass binary: `./target/debug/coil build selfhost/<pass>.coil -o /tmp/coil-self-<pass>`.
- Run all .coil work from the REPO ROOT (imports are `"lib/…"` CWD-relative).
