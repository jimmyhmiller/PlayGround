# Annotated bibliography

Core references the formal design rests on. Citation details below were
web-verified on 2026-06-16; where a venue/year was corrected from first-draft
memory it is noted. Always re-check page numbers against the PDF before quoting.

## Quantitative / linear-dependent type theory (the kernel — `docs/01`)

- **Conor McBride. "I Got Plenty o' Nuttin'." 2016.**
  In *A List of Successes That Can Change the World: Essays Dedicated to Philip
  Wadler on the Occasion of His 60th Birthday* (WadlerFest), Springer LNCS 9600,
  pp. 207–233. — Origin of the rig-annotated approach. Key move: use the rig's
  **0** to mark context entries that are present "for contemplation rather than
  consumption" (types/erased), reconciling linearity with dependency. Motivating
  example: `(l : List X) ⊸ Vector X (length l)` (in-place list→vector).
  PDF: https://personal.cis.strath.ac.uk/conor.mcbride/pub/Rig.pdf

- **Robert Atkey. "Syntax and Semantics of Quantitative Type Theory." LICS 2018.**
  Proc. 33rd ACM/IEEE LICS, pp. 56–65. DOI 10.1145/3209108.3209189. — The full
  type theory ("QTT") with realizability semantics via Quantitative Categories
  with Families. This is **the** reference for our kernel rules and the erasure
  result. Shipped (in adapted form) in **Idris 2**.
  Page: https://bentnib.org/quantitative-type-theory.html ·
  PDF: https://strathprints.strath.ac.uk/64031/

- **(Background) Idris 2 / Edwin Brady.** "Idris 2: Quantitative Type Theory in
  Practice." (ECOOP 2021). — Existence proof that QTT works in a real
  implementation; useful for pragmatics of multiplicity inference.

- **(Compare) "A Two-Level Linear Dependent Type Theory"; "Foundations of
  Substructural Dependent Type Theory" (arXiv 2401.15258).** — More recent
  alternative formulations to cross-check our kernel against.

## Low-level memory safety: locations & capabilities (the view layer — `docs/02`)

- **Amal Ahmed, Matthew Fluet, Greg Morrisett. "L3: A Linear Language with
  Locations." TLCA 2005** (Springer LNCS 3461); journal version in *Fundamenta
  Informaticae* 77(4), 2007; tech report Harvard, Oct 2004. — *The* minimal core
  for our pointer/capability split. First-class, explicit **capabilities**;
  pointers freely duplicable; capability linear ⇒ **strong update** (retype a cell)
  is sound. Directly the model behind `alloc/read/take/write/free`.
  PDFs: https://www.cs.cornell.edu/people/fluet/research/lin-loc/POPL05/popl05.pdf
  · https://www.ccs.neu.edu/home/amal/papers/linloc-fi07.pdf

- **Hongwei Xi et al. ATS / Applied Type System.** "Applied Type System: An
  Approach to Practical Programming with Theorem-Proving" (arXiv 1703.08683);
  ATS first appeared ~2006. — Dependent types over a **decidable static index
  domain** + linear **views** (`dataview`/`viewtype`: `T @ L`) and **props**
  (`dataprop`). The closest existing whole-system relative; source of the
  stratification idea (`docs/03` stratum A) and the view vocabulary.
  Also: "A linear type system for multicore programming in ATS" (SCP, 2012).

- **John Boyland. "Checking Interference with Fractional Permissions." SAS 2003**
  (Springer LNCS 2694). — Reads at fraction `0<q<1`, writes need `q=1`; fractions
  split/recombine. The basis for our shared-read ergonomics knob (`docs/02` §5.1).
  Also: "Semantics of fractional permissions with nesting" (TOPLAS 2010).

- **Separation logic.** Reynolds, "Separation Logic: A Logic for Shared Mutable
  Data Structures" (LICS 2002); O'Hearn, "Resources, Concurrency, and Local
  Reasoning" (CSL/Concur, → concurrent SL). — Our view algebra (`A @ ℓ`, `∗`,
  frame) *is* separation logic embedded in the type system. Needed for the heap
  -typing/disjointness proofs (`docs/04` M2.2, T2.*). *(Citations from memory;
  verify.)*

- **Mads Tofte, Jean-Pierre Talpin. "Region-Based Memory Management."
  Information and Computation, 1997.** — Regions = batched, lifetime-scoped
  allocation. Basis for `docs/02` §4 (and "GC = an ω-region you never free").
  *(Verify exact venue/year.)*

## The programmable type system (the "Shen" goal — `docs/03`)

- **Mark Tarver. Shen / "The Book of Shen."** — Optional static typing via a
  **sequent calculus** that is **Turing-complete**, implemented over a Prolog/
  Horn-clause engine; "can model dependent types." The inspiration for
  rules-as-data and the extensible rule layer. We add the soundness *fence*
  (elaborate-to-kernel, `docs/03` §4) that Shen does not impose.
  Wiki: https://github.com/Shen-Language/wiki/wiki/Sequent-Calculus

- **(Architecture) LCF / Milner; "tactics vs. axioms."** — The discipline that
  makes the programmable layer safe: extensions must *produce a kernel-checkable
  derivation*, never assert one. Standard proof-assistant lore (Rocq/Lean/Agda
  kernels). Cite a concrete source when we write up §4 formally.

## Adjacent systems worth a comparison pass (not yet load-bearing)

- **Rust** — affine ownership + borrows + lifetimes; no dependent types; fixed
  checker. The ergonomics target for `docs/02` §5 / Phase 3.
- **Vale** — generational references / "Linear-aliasing model"; another point in
  the safe-low-level space.
- **Cogent, Granule, Linear Haskell (Bernardy et al., POPL 2018), Dafny, F\*,
  Low\*/KaRaMeL** — each combines some of {linear, dependent, low-level,
  verified}; mine for design moves and pitfalls.
- **RustBelt / Iris** — machine-checked semantic soundness for a
  Rust-like language via a concurrent separation logic in Coq/Rocq; the gold
  standard for the kind of safety proof in `docs/04` Phase 2, and a candidate
  mechanization substrate (D5.3).

## Verification status

Verified 2026-06-16 via web search: McBride 2016 (WadlerFest LNCS 9600), Atkey
LICS 2018 (pp. 56–65), L3 (TLCA 2005 / Fund. Inf. 2007), Boyland SAS 2003 (LNCS
2694), ATS (Xi, ~2006; arXiv 1703.08683), Shen (Tarver, sequent-calculus, over
Prolog). **Still to verify:** Reynolds 2002, O'Hearn, Tofte–Talpin 1997, Idris 2
ECOOP 2021, Linear Haskell POPL 2018, the arXiv comparison papers.
