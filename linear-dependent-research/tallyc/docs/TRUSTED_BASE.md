# The Trusted Base — every kernel-opaque primitive, and why its safe type is sound

*Phase 0 deliverable (docs/PHASE_SAFETY_PLAN.md §0.3). This is the honest TCB:
the finite list of things tallyc asks you to trust beyond the kernel itself.
Everything else — the elaborator, the pattern-matrix compiler, the convoy/absurd
classifiers, the totality checker — is UNTRUSTED: every term they synthesize is
re-checked by the kernel, so a bug there yields a rejected program, never an
unsound accepted one. Every future primitive must join this table with the same
two columns filled in.*

The kernel's own trusted core: the QTT rig (`src/mult.rs`, `0/1/ω`, `1+1=ω`,
`0⋢1`), the type checker (`src/dep.rs` — Π/Σ/Nat/Eq/inductives with positional
one-method-per-constructor eliminators, checked at `dep.rs` `Elim`/`Case`
method-count sites), strict positivity + predicative universes, and the LLVM
lowering (`src/dep_codegen.rs`). A primitive below is "sound at its safe type"
when: erased (`0`) arguments never influence emitted code, linear (`1`)
arguments' single-use discipline makes the underlying raw operation
unmisusable, and the lowering computes what the type says.

## 1. Memory (the `Own` layer)

| primitive | safe type (as exposed) | why the safe type is sound |
|---|---|---|
| `Own` | `linear Type -> Type` | abstract linear box; only mintable by `alloc`, only disposable by `free`/`unbox` — linearity makes exactly one disposal reachable on every path. |
| `alloc` | `{0 a} -> (1 x : a) -> Own a` | `malloc` + store of an already-constructed value: no uninitialized cell is ever observable on the common path. |
| `free` | `{0 a} -> (1 o : Own a) -> Unit` | libc `free` of the unique token; linearity = no double-free, no dangling survivor. **Rejected at a LINEAR payload type** (the dropping-destructor gate, Phase A3): dropping the cell would leak the resource inside — consume it first via `unbox`. |
| `ralloc` | `{0 a} -> Unit -> RawCell a` (`= ∃l. Ptr l ⊗ RawTo l a`) | Phase A3 raw allocation: `malloc` with NO store. Sound because the returned permission `RawTo l a` is UNREADABLE BY TYPE — no read/write/free op accepts it; only `winit` and `rfree` do. Zero-width, erased (IR-tested). |
| `winit` | `{0 a}{0 l} -> Ptr l -> (1 v : RawTo l a) -> a -> PtsTo l a` | the first write: consumes the raw permission, yields the ordinary initialized view. Same machine code as `vwrite`; the cell was sized by the same erased `a`. |
| `rfree` | `{0 a}{0 l} -> Ptr l -> (1 v : RawTo l a) -> Unit` | reclaim a never-initialized cell; nothing was stored, nothing can have been read. |
| `unbox` | `{0 a} -> (1 o : Own a) -> a` | load + `free`; the payload is moved out (the box token is consumed), never aliased. |

## 2. Views (address/permission split — L3/ATS)

| primitive | safe type | why sound |
|---|---|---|
| `Ptr l` / `Loc` | `Loc : Type`, `Ptr : Loc -> Type` | an address is ω data; holding one grants nothing. `l` is a static erased name. |
| `PtsTo` | `linear Loc -> Type -> Type` | THE permission. Zero-width (no runtime trace, IR-tested); at most one per location, so no stale access is typable. |
| `valloc` | `{0 a} -> (1 x : a) -> Cell a` | `malloc`+store, returns `∃l. Ptr l ⊗ PtsTo l a`; the fresh `l` is abstract so the permission can never be confused with another cell's. |
| `vwrite` | `… Ptr l -> (1 v : PtsTo l a) -> b -> PtsTo l b` | strong update: consuming the only `PtsTo l a` means no reader of the old type survives — type-changing store is sound. |
| `vread` | `… Ptr l -> (1 v : PtsTo l a) -> a` | destructive read + free of the cell (move out, not copy). |
| `vtake` | `… Ptr l -> (1 v : PtsTo l a) -> Taken a l` | moves the payload out, retyping the slot `PtsTo l (Hole a)`; the hole is unreadable by type until `vwrite` refills it. |
| `vfree` | `… Ptr l -> (1 v : PtsTo l a) -> Unit` | free requires the whole permission; a borrowed/split cell cannot be freed. **Rejected at a LINEAR payload type** (Phase A3 dropping-destructor gate) — use `vread` to extract the payload first. |
| `borrow` | `{0 a} -> (1 o : Own a) -> Borrowed a` (`= ∃l. Ptr l ⊗ PtsTo l a ⊗ Loan l a`) | identity on the address (zero-cost, IR-tested); the loan is the linear obligation to reunite. |
| `restore` | `… Ptr l -> (1 v : PtsTo l a) -> (1 ln : Loan l a) -> Own a` | only the matching view+loan pair (same `l`) reunifies — a swapped or stale pair is a type error. |

## 3. Regions / pools

| primitive | safe type | why sound |
|---|---|---|
| `rnew` | `Unit -> RegionPack` (`∃r. RegionCap r`) | fresh abstract region name + its unique linear capability. |
| `pnew` | `{0 r}{0 a} -> (1 cap : RegionCap r) -> Pool r a` | consumes the region's only capability: one pool per region, the pool inherits uniqueness. |
| `palloc`/`pget`/`pset`/`pfree` | thread `(1 P : Pool r a)` linearly; `RPtr r` is ω | all access is through the single pool token — a freed slot's `RPtr` can only be re-observed via pool ops that see the current pool state (freelist reuse is visible, never wild). |
| `prelease` | `{0 a}{0 r} -> (1 P : Pool r a) -> Unit` | frees the arena; the pool token dies with it, and every `RPtr r` is dead weight afterwards (no op accepts it without a pool). |
| `peq` | `{0 r} -> RPtr r -> RPtr r -> Nat` | pointer identity within one region: pure comparison, no deref. |

## 4. Contiguous arrays (erased bounds)

| primitive | safe type | why sound |
|---|---|---|
| `Arr` | `linear Type -> Nat -> Type` | one flat `malloc(n·w)`; the length exists only in the erased index. |
| `anew` | `{0 a} -> (n : Nat) -> a -> Arr a n` | allocates AND fills (memset-style loop): no uninitialized slot. |
| `aget` | `{0 a}{0 n} -> (i : Nat) -> (0 p : Lt i n) -> (1 arr) -> ARead a n` | the bare indexed load; in-bounds is a THEOREM (`Lt i n`, erased — no branch in IR, tested), not a check. |
| `aset` | like `aget`, storing | same; store is at the element's true width (packed layout). |
| `afree` | `{0 a}{0 n} -> (1 arr) -> Unit` | frees the unique token. |
| `dlt` | `(i : Nat) -> (n : Nat) -> DecLt i n` | ONE `icmp`; `DYes` carries the erased proof. Sound because the compare's truth IS the proposition; the proof payload is zero-width (IR-tested). |

## 5. Scalars, casts, floats, native arithmetic

| primitive | safe type | why sound |
|---|---|---|
| `U8…I64`, `F32`, `F64` | `postulate … : Type` | opaque storage types; values ride the i64 register (floats as bit patterns), narrowed only at typed stores / decoded at float ops. |
| `sub mul div mod ltb leb eqb band bor bxor shl shr` | `Nat -> Nat -> Nat` | single machine ops, TOTAL at the edges (`n/0=0`, `n%0=n`, wrap mod 2⁶⁴); kernel-opaque so type-level Nat stays the total fragment. |
| `u8…i64`, `nat_*` | `Nat -> T` / `T -> Nat` | mask-to-width / reinterpret; no UB in either direction. |
| `sadd…sshr`, `sneg`, `slt sle seq` | `{0 a} -> a -> a -> a` (cmp → `Nat`) | C semantics selected by the ERASED `a` (width+signedness recovered at compile time — the erased type directs codegen but leaves no runtime trace); total edges as above; non-scalar `a` is a guided compile error. |
| `cast` | `{0 a}{0 b} -> a -> b` | the one conversion; both endpoints erased-known at compile time. Float→int uses the SATURATING intrinsics (`llvm.fptosi.sat`/`llvm.fptoui.sat`: NaN → 0, out-of-range → min/max) — a bare `fptosi` is poison on those inputs, which was the one UB reachable from safe code found and closed by the Phase 0 audit. |
| `fadd…fneg`, `flt fle feq`, `f*_bits`, `f*_of_nat`, `nat_of_f*` | `{0 a} -> a -> a -> a` etc. | IEEE-754 ops on the decoded register; ordered comparisons; bit-pattern constructors are identities. |

## 6. I/O and FFI

| primitive | safe type | why sound |
|---|---|---|
| `print` / `prints` | `Nat -> Unit` / `Str -> Unit` | `printf`; observable effect, no memory interaction with the linear layer. |
| `putc` / `getc` | `Nat -> Unit` / `Unit -> Nat` | `putchar`/`getchar`; `getc` encodes EOF as `Zero` and a byte as `Succ b` — total, no sentinel confusion. |
| `%foreign "sym" : T` | any first-order scalar/record type | **the honest escape hatch.** The C side is unchecked; soundness is exactly the C function's conformance to the declared ABI (i64/double registers, records flattened by value). Every use is a visible per-program declaration. |
| `%builtin Nat T` | pragma | re-representation of a Nat-shaped enum as machine i64; the shape is validated (two ctors, zero + self-succ). |

## 7. The `unreachable` ledger (§0.1)

Every `build_unreachable()` the backend can emit, and the kernel-checked fact
that guards it. Adversarial corpus: `phase0_adversarial_coverage_corpus` in
`src/rust_surface/tests.rs` (all bad matches rejected) and the
`phase0_*`/`nested_patterns_*` runtime tests in `src/dep_codegen.rs`.

| site (`src/dep_codegen.rs`) | where it sits | why it is unreachable |
|---|---|---|
| `fold.default` (~1692) | switch default in the boxed-`Elim` accumulator-fold helper | a cell's tag is only ever stored by constructor compilation with `tag = ctor index < #ctors`; the kernel accepts an `Elim` only with exactly one method per constructor (`dep.rs` method-count check), so every live tag has an arm. |
| `elim.default` (~3906) | switch default in the boxed-`Elim` helper | same tag invariant + same kernel method-count check (also re-asserted at the helper: `methods.len() == decl.ctors.len()`). |
| `vcase.default` (~4227) | switch default of a VALUE-enum `Case` | the tag COMPONENT of a value enum is only constructed from ctor indices (construction-side invariant); kernel `Case` method count as above. |
| refuted value-enum arm (~4239) | an argful ctor whose method is a binderless sentinel | emitted only when the elaborator classified the arm ABSURD (index refuted); the classification is untrusted, but the synthesized `Case` including its motive is KERNEL-re-checked — a wrongly-refuted reachable arm fails re-check (the `try_absurd_match` linchpin), so the block is provably dead when it exists. |
| `case.default` (~4335) | switch default of a boxed `Case` | tag invariant + kernel method count; the elaborator additionally rejects a missing arm before lowering (`missing a case for …`). |

The null-pointer-niche match (~4148) emits NO default: a one-slot niche value
is a two-way conditional branch, both arms covered by construction.

**P0 invariant (tested):** any non-exhaustive match that COMPILES is a bug.
The corpus pins: flat missing arm, missing arm under `%partial`, nested
missing (2- and 3-deep), missing inside an arm's nested match, value-enum
missing, `%builtin Nat` missing successor, mixed absurd+reachable missing the
reachable arm, absurd discharge claimed at a non-empty index, wrong-arity
pattern, cross-family constructor, and shadowed-unreachable arms — ALL
rejected at elaboration; the kernel's method-count check backstops each.

## 8. Erasure invariant (§0.2)

`0 ⇒ no runtime trace`, asserted in IR per construct (in
`src/dep_codegen.rs` tests): `Vec` indices (`vec_ir_has_zero_overhead`),
region/cursor ghosts (`dll_ir_has_zero_overhead`), erased `Lt` bounds — no
branch (`arr_ir_is_bare_indexed_load_no_bounds_check`), views/loans and
`borrow`/`restore` (`borrows_mutate_in_place_with_zero_trace`,
`phase0_views_loans_holes_fully_erased`), the null niche
(`phase0_null_niche_is_bare_pointer_compare`), `dlt`'s proof payload
(`phase0_dlt_proof_leaves_no_trace`), and packed-scalar leaves' erased
types/bounds (`phase0_packed_scalar_erased_witnesses_leave_no_trace`).
