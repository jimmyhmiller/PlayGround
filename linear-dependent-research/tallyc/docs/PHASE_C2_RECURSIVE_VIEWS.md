# Phase C2 — recursive views that ERASE (design sketch, not built)

**STATUS: PLAN.** Slice 1 (v2.1) delivered zero-width views and zero-trace
`&mut` read-back borrows. This doc scopes the remaining, research-grade half of
Phase C: **inductive heap predicates for unbounded linked structures under
views** — what an in-language, O(1)-remove DLL needs — with the same erasure
bar (a view leaves no runtime trace).

## The problem

A recursive view describes a chain of permissions:

    Seg l  =  (l = null)  ∨  ∃l'. (PtsTo l (NodeShape l')) ⊗ Seg l'

Two obstacles, in order of difficulty:

1. **The shadow-structure cost.** Declared as an ordinary `boxed enum`, a `Seg`
   value is a heap chain of tags mirroring the real structure — O(n) memory
   for a proposition that should be free. All its FIELDS are already
   zero-width (views, sub-segs); what remains is the TAG. So a `Seg` value is
   morally one bit of information per node… which the heap already contains
   (the node's next pointer is null or not).
2. **Dispatch without a tag.** If `Seg` erases fully (width 0), a `match` on
   it has nothing to switch on. The arm is determined by RUNTIME data — the
   null-ness of the corresponding `Ptr` — so the elimination must be a
   **dependent null-check**: comparing `p` against null must REFINE the
   erased `Seg` index in each branch. Connecting a runtime comparison to an
   erased type index is exactly the step F\*/Steel discharge with SMT and ATS
   with its constraint solver; tallyc wants it decidable and kernel-checked.

## Candidate design (the niche-seg)

Make the null-dispatch a PRIMITIVE with a convoy-shaped type, mirroring how
`getc` folds EOF into `Zero` and how the null-niche folds `Opt (Own T)`:

    pnull : {0 l : Loc} -> (p : Ptr l) -> (1 s : Seg l) -> SegView l
    enum SegView (l : Loc) {
      VNil  : … -> SegView l                      -- p was null; s refined SNil
      VCons : {0 l' : Loc} -> (p' : Ptr l')
              -> (1 hd : PtsTo l (NodeShape l'))
              -> (1 tl : Seg l')
              -> SegView l
    }

- `Seg` itself is a **width-0 linear postulate** (like `PtsTo`): no tag, no
  cell, pure accounting.
- `pnull` lowers to `icmp eq p, 0` — ONE instruction. Its RESULT carries the
  refined pieces, all zero-width except the next pointer `p'`, which is
  loaded from the head cell (a load the traversal needs anyway).
- Folding/unfolding `Seg` (cons-ing a head view onto a tail seg, splitting a
  seg at a cursor) are further postulates whose lowerings are identity/loads —
  each must ship with an IR-trace test and a red-team, per §4.4.

This stays decidable because the "entailment" is fixed-menu: the primitives
are the only introduction/elimination forms, checked against their declared
types by the ordinary kernel; nothing is solved.

## What it buys, and the honest O(1)-remove gap

With `Seg`, unbounded in-place traversal/mutation under views (in-place list
reversal, the separation-logic hello-world) becomes writable in-language at
zero overhead. **O(1) remove-by-cursor is still not free**: extracting the
permission for a MIDDLE node from an inductive `Seg` is O(n) by construction
(the doc's magic-wand/ghost-state problem). The known decidable answers:

- **Region/arena discipline (Phase D):** one linear token owns ALL nodes'
  permissions collectively; pointers are ω within the region; unlink is plain
  pointer surgery authorized by the token; reclamation is bulk `release`.
  Memory-safe (no UAF after release, type-stable if the pool is homogeneous),
  C-fast, and expressible with today's machinery plus an allocator API —
  this is the pragmatic in-language DLL.
- **Seg-split lemmas (LCF-style):** `split : Seg l -> (Seg l@c, PtsTo c _,
  Seg c')` given a cursor — but producing the split WITNESS is where the
  O(n) or the ghost state hides. Parked.

Recommendation: build the niche-seg for singly-linked in-place algorithms
(C2 proper), and route the O(1)-remove DLL through Phase D regions — the
canned `new`/`insert`/`remove` primitives already measure C-identical
meanwhile.

## Acceptance bar (unchanged)

Every new primitive: an IR-trace test proving zero runtime trace beyond the
loads/stores/compares the algorithm itself needs, and a red-team showing
leak / double-use / use-after-free / refill-overflow all reject.
