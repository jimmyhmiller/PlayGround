# 04 — Research roadmap and proof obligations

The agenda. Ordered so that each step rests on settled ground beneath it. Each
milestone names a *deliverable* (a definition, a theorem, or a decision) rather
than code, since we are building formalisms, not a compiler.

## Phase 0 — Pin the foundations (literature + decisions)

- [ ] **D0.1** Verify and annotate the core references (`notes/bibliography.md`):
      McBride 2016, Atkey 2018, Ahmed–Fluet–Morrisett L3, ATS (Xi), Boyland
      fractional permissions, Tofte–Talpin regions, Shen type system. Get exact
      titles/venues/years right before building on them.
- [ ] **D0.2** Choose the rig `R` for the first formal pass. Recommend: start with
      `{0,1,ω}` (provably enough for linearity + erasure), defer affine `≤1` and
      fractions to a second pass so the first metatheory is small.
- [ ] **D0.3** Decide the kernel's definitional equality (β? βη? with/without
      proof irrelevance). This gates decidability of (B).

## Phase 1 — λ-Tally, the pure core (no memory yet)

Goal: nail QTT on paper for *our* rig and notation, independent of memory.

- [ ] **M1.1** Full grammar + all typing rules (Π, Σ⊗, &, Id, `Nat`/`Fin`/`Vec`,
      universe) with multiplicities, in the 0/1 modal judgment.
- [ ] **M1.2** Operational semantics for the relevant (`σ=1`) fragment.
- [ ] **T1.1 (Type safety).** Progress + preservation for λ-Tally.
- [ ] **T1.2 (Erasure).** Define `|·|`; prove `Γ ⊢ M :^1 A` ⇒ `|M|` simulates `M`
      and contains no `0`-budget variable. *This is the "free dependent types"
      theorem.* (Mirror Atkey's semantic erasure; aim for a syntactic version too.)
- [ ] **T1.3 (Substitution/scaling lemmas).** The algebraic lemmas about `+`/`·`
      on contexts that all later proofs lean on.

## Phase 2 — The memory layer (views)

Goal: add `docs/02` and prove the safety payoff.

- [ ] **M2.1** Formal view algebra: `A @ ℓ`, `emp`, `∗`, `∃ℓ`, `⊕`; the typing of
      `alloc/read/take/write/free`; runtime heaps `H`.
- [ ] **M2.2** **Heap typing**: define when `(Γ, H)` is consistent — the live
      views in `Γ` partition the live cells of `H` (`∗` ⇒ disjoint).
- [ ] **T2.1 (Preservation w/ heap).** Each primitive preserves heap typing.
- [ ] **T2.2 (Memory safety).** No reachable step dereferences a freed/unallocated
      location; no double-free; strong update never observes a stale type.
- [ ] **T2.3 (No-leak).** Programs ending with no `1`-budget views free all they
      allocate (linear, non-region fragment).
- [ ] **M2.4** Regions: `newregion/allocIn/freeregion`; the `ω`-region = opt-in GC
      account; prove region `free` reclaims exactly its locations.

## Phase 3 — Ergonomics: borrows / fractions (the unsettled part)

Goal: resolve Tension T4. Expect iteration.

- [ ] **D3.1** Decide: fractions-only, borrows-only, or both. Prototype each on
      `swap`, shared-read, and a tree-with-parent-pointers example.
- [ ] **M3.1** Extend the rig and/or add the borrow judgment; extend the static
      domain (A) with the needed constraint fragment (rational `+`, lifetimes).
- [ ] **T3.1 (Data-race freedom for reads).** A `write` (needs `q=1`) cannot
      coexist with any outstanding read fraction.
- [ ] **T3.2** Re-prove T2.* under the enriched permission structure.

## Phase 4 — The programmable layer

Goal: `docs/03`, with the soundness fence.

- [ ] **M4.1** Judgments-as-data; the rulebase interpreter; fuel semantics.
- [ ] **M4.2** The elaboration discipline: user rule ⇒ kernel derivation.
- [ ] **T4.1 (Conservativity).** Any program accepted via user rules has a kernel
      derivation; hence the kernel's safety theorems transfer. *The fence holds.*
- [ ] **M4.3** Fix the static domain (A) decision procedure(s) and their trust
      story (certificates preferred).
- [ ] **W4.4** Worked extension end-to-end: binary session types as a user
      rulebase elaborating onto the linear view kernel.

## Phase 5 — Consolidation

- [ ] **W5.1** A canon of worked examples that *must* type-check: safe `malloc`/
      `free`, in-place list reversal, an arena allocator, a length-indexed API,
      a file-handle protocol, a session-typed channel, swap of linear values.
- [ ] **W5.2** A canon of programs that *must be rejected*, each mapped to the
      rule that rejects it (use-after-free, double-free, leak, OOB, race, stale
      strong-update alias).
- [ ] **D5.3** Decide mechanization target (Agda / Rocq / Lean) and port T1–T4.
      The erasure + linear-substitution lemmas are the most error-prone by hand
      and most benefit from a proof assistant.

## Cross-cutting open problems (tracked, not yet scheduled)

- Definitional equality vs. decidability (the eternal dependent-types knob).
- The separation/disjointness constraint fragment for the static solver — least
  off-the-shelf piece.
- Whether borrows are kernel or derivable in the programmable layer.
- Concurrency: do views extend to a concurrent separation logic (CSL) story, and
  does data-race freedom generalize beyond shared reads to locks/atomics?
- FFI / the trusted leaf: minimal set of kernel-blessed primitive views for
  MMIO/`volatile`/C interop, and the proof obligations at that boundary.

## Suggested attack order (the short version)

1. Phase 1 (λ-Tally + erasure) — small, high-confidence, unblocks everything.
2. Phase 2 (views + memory safety) — the headline result; the reason the project
   exists.
3. Phase 4 fence (conservativity) *before* going wild in Phase 3, so ergonomics
   experiments can live in the safe programmable layer.
4. Phase 3 ergonomics — the genuinely open design work, done last on purpose.
