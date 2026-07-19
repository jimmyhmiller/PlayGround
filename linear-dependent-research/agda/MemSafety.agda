------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 7: MEMORY SAFETY for the heap machine (the Agda port of Rosette E1).
--
-- The Rosette models proved memory safety bounded-exhaustively + via an
-- inductive invariant, but with Z3 in the trusted base.  This module re-proves
-- the operational core in Agda: an abstract memory cell with the primitives
-- (alloc / read / write / resize / free / mkclaim / staleread), a store-typing
-- INVARIANT, and machine-checked proofs that
--   * the invariant is INDUCTIVE   (preserved by every operation), and
--   * the invariant IMPLIES SAFETY  (every access is in-bounds of live memory),
-- hence -- with the initial state satisfying it -- safety holds for programs of
-- ANY length (the `run`/`run-Inv` corollary), UNBOUNDED and with NO SMT solver
-- in the trusted base.
--
-- This is the "preservation + progress" for the heap machine -- strong update
-- (resize changes the stored length) and the stale-claim hazard (mkclaim /
-- staleread) included.  The invariant: holding a live linear view implies the
-- location is alive and the view's claimed length equals the actual length --
-- exactly what makes a read in-bounds.  (Locations are independent, so a single
-- cell captures the per-location discipline; a full heap is a product of cells.)
--
-- Remaining (docs): wiring this to the term-level linear type checker, so that
-- "well-typed term ⟹ Inv holds throughout its run" (the resource-soundness link
-- that Rosette E4 showed bounded).
--
-- Check:  LC_ALL=C.UTF-8 agda MemSafety.agda
------------------------------------------------------------------------

module MemSafety where

open import Rig using (_≡_; refl; cong; sym; trans)

------------------------------------------------------------------------
-- minimal prelude
------------------------------------------------------------------------

transp : ∀ {A : Set} (P : A → Set) {x y} → x ≡ y → P x → P y
transp P refl p = p

data Bool : Set where true false : Bool

data Nat : Set where
  zero : Nat
  suc  : Nat → Nat

data _≤_ : Nat → Nat → Set where
  z≤n : ∀ {n}           → zero  ≤ n
  s≤s : ∀ {m n} → m ≤ n → suc m ≤ suc n

_<_ : Nat → Nat → Set
m < n = suc m ≤ n

record _×_ (A B : Set) : Set where
  constructor _,_
  field fst : A
        snd : B
open _×_

data List (A : Set) : Set where
  []  : List A
  _∷_ : A → List A → List A

------------------------------------------------------------------------
-- machine state: one heap cell + the program's view/claim on it
--   alive : is the cell allocated?       len   : its actual array length
--   vview : a live LINEAR view held?      vlen  : the length that view claims
--   scap  : a stale secondary claim?      slen  : the length it asserts
------------------------------------------------------------------------

record St : Set where
  constructor mkSt
  field alive : Bool ; len : Nat ; vview : Bool ; vlen : Nat ; scap : Bool ; slen : Nat
open St

data Op : Set where
  alloc : Nat → Op      -- allocate an array of length n; yields the linear view
  rd    : Nat → Op      -- read element n   (needs the view)
  wr    : Nat → Op      -- write element n  (needs the view)
  srd   : Nat → Op      -- read element n via the view (the SOUND staleread)
  rsz   : Nat → Op      -- resize to length n  (STRONG UPDATE; needs the view)
  fr    : Op            -- free               (consumes the view)
  mkc   : Op            -- form a stale secondary claim of the current length

------------------------------------------------------------------------
-- one operation (the SOUND discipline)
------------------------------------------------------------------------

-- helpers take the guard booleans explicitly, so the operation reduces as soon
-- as the guards are known (this is what lets `preservation` analyse them).
stepAlloc : Bool → Nat → St → St
stepAlloc false n s = mkSt true n true n false zero       -- allocate; one linear view
stepAlloc true  n s = s

stepRsz : Bool → Bool → Nat → St → St
stepRsz true  true  n s = mkSt true n true n (scap s) (slen s)  -- strong update (still alive/owned)
stepRsz true  false n s = s
stepRsz false b     n s = s

stepFr : Bool → Bool → St → St
stepFr true  true  s = mkSt false zero false (vlen s) (scap s) (slen s)   -- reclaim
stepFr true  false s = s
stepFr false b     s = s

stepMkc : Bool → Bool → St → St
stepMkc true  true  s = mkSt (alive s) (len s) (vview s) (vlen s) true (vlen s)
stepMkc true  false s = s
stepMkc false b     s = s

step : Op → St → St
step (alloc n) s = stepAlloc (alive s) n s
step (rd _)  s = s
step (wr _)  s = s
step (srd _) s = s
step (rsz n) s = stepRsz (vview s) (alive s) n s
step fr      s = stepFr  (vview s) (alive s) s
step mkc     s = stepMkc (vview s) (alive s) s

------------------------------------------------------------------------
-- the store-typing invariant
------------------------------------------------------------------------

Inv : St → Set
Inv s = vview s ≡ true → (alive s ≡ true) × (vlen s ≡ len s)

------------------------------------------------------------------------
-- (1) PRESERVATION: every operation preserves the invariant
------------------------------------------------------------------------

-- per-operation lemmas: case on the GUARD BOOLEANS (helper parameters), so the
-- operation reduces and `s`'s fields are never abstracted away.
preservation : ∀ op s → Inv s → Inv (step op s)
preservation (alloc n) s inv = lemA (alive s) n s inv
  where lemA : ∀ b n s → Inv s → Inv (stepAlloc b n s)
        lemA false n s inv = λ _ → (refl , refl)
        lemA true  n s inv = inv
preservation (rd _)  s inv = inv
preservation (wr _)  s inv = inv
preservation (srd _) s inv = inv
preservation (rsz n) s inv = lemR (vview s) (alive s) n s inv
  where lemR : ∀ b₁ b₂ n s → Inv s → Inv (stepRsz b₁ b₂ n s)
        lemR true  true  n s inv = λ _ → (refl , refl)
        lemR true  false n s inv = inv
        lemR false b₂    n s inv = inv
preservation fr s inv = lemF (vview s) (alive s) s inv
  where lemF : ∀ b₁ b₂ s → Inv s → Inv (stepFr b₁ b₂ s)
        lemF true  true  s inv = λ ()
        lemF true  false s inv = inv
        lemF false b₂    s inv = inv
preservation mkc s inv = lemM (vview s) (alive s) s inv
  where lemM : ∀ b₁ b₂ s → Inv s → Inv (stepMkc b₁ b₂ s)
        lemM true  true  s inv = inv
        lemM true  false s inv = inv
        lemM false b₂    s inv = inv

------------------------------------------------------------------------
-- (2) SAFETY: in any Inv-state, a view-authorised access is in-bounds of
-- live memory.  (The access fires only when the view is held, the cell is
-- alive, and the index is below the view's claimed length.)
------------------------------------------------------------------------

access-ok : ∀ s n
          → Inv s
          → vview s ≡ true → alive s ≡ true → n < vlen s
          → (alive s ≡ true) × (n < len s)
access-ok s n inv vv al lt with inv vv
... | (al′ , veq) = (al , transp (λ z → n < z) veq lt)

------------------------------------------------------------------------
-- the initial state satisfies Inv, and Inv is preserved along any run, so
-- safety holds for programs of ANY length (unbounded, no SMT).
------------------------------------------------------------------------

init : St
init = mkSt false zero false zero false zero

init-Inv : Inv init
init-Inv ()

run : List Op → St → St
run []         s = s
run (op ∷ ops) s = run ops (step op s)

run-Inv : ∀ ops s → Inv s → Inv (run ops s)
run-Inv []         s inv = inv
run-Inv (op ∷ ops) s inv = run-Inv ops (step op s) (preservation op s inv)

-- MEMORY SAFETY: after running any program from the initial state, the
-- store-typing invariant still holds -- hence (by access-ok) every read is
-- in-bounds of live memory.  No bound on program length; no Z3.
safety : ∀ ops → Inv (run ops init)
safety ops = run-Inv ops init init-Inv
