------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 5b, stage 3: PRESERVATION  (the other half of type safety).
--
-- Reduction preserves typing (and the type): if  Φ ⊢[ Γ ] t ⦂ A  and  t ⟶ t′
-- then t′ is well typed at the same type (at some usage context Γ′ -- a reduct
-- may use FEWER resources, which is exactly the expected behaviour; we report
-- the witness existentially, so no subusaging machinery is needed).
--
-- The β case is discharged by the quantitative substitution lemma (stage 2);
-- the congruence cases recurse.  Together with Progress.agda this is TYPE SAFETY
-- for lambda-Tally's linear core -- the unbounded, machine-checked form of the
-- Rosette resource-soundness result.
--
-- Check:  LC_ALL=C.UTF-8 agda Preservation.agda
------------------------------------------------------------------------

module Preservation where

open import Rig
open import Context
open import Syntax
open import Semantics
open import Substitution

-- a usage context together with a derivation at that context
record Typed {n} (Φ : TyCtx n) (t : Term n) (A : Ty) : Set where
  constructor _,_
  field
    usage : Ctx n
    deriv : Φ ⊢[ usage ] t ⦂ A

------------------------------------------------------------------------
-- single-step preservation
------------------------------------------------------------------------

preservation : ∀ {n} {Φ : TyCtx n} {Γ : Ctx n} {t t′ : Term n} {A}
             → Φ ⊢[ Γ ] t ⦂ A → t ⟶ t′ → Typed Φ t′ A
preservation (⊢app fd ad) (ξ-app₁ fstep) with preservation fd fstep
... | (_ , fd′) = _ , ⊢app fd′ ad
preservation (⊢app fd ad) (ξ-app₂ _ astep) with preservation ad astep
... | (_ , ad′) = _ , ⊢app fd ad′
preservation {Φ = Φ} (⊢app {Δ = Δv} (⊢lam {A = A′} bodyD le) vD) (β-lam {v = v} _)
  = _ , subst-lemma {σ = sub-head v} {Δs = Δs₀} hyp₀ bodyD
  where
    Δs₀ : Fin _ → Ctx _
    Δs₀ zero    = Δv
    Δs₀ (suc x) = only x
    hyp₀ : ∀ i → Φ ⊢[ Δs₀ i ] sub-head v i ⦂ lookupV (A′ ∷ Φ) i
    hyp₀ zero    = vD
    hyp₀ (suc x) = ⊢var x

------------------------------------------------------------------------
-- multi-step preservation (capstone): well-typedness survives any number of
-- reduction steps.  With progress (Progress.agda), this is type safety.
------------------------------------------------------------------------

preservation* : ∀ {n} {Φ : TyCtx n} {Γ : Ctx n} {t t′ : Term n} {A}
              → Φ ⊢[ Γ ] t ⦂ A → t ⟶* t′ → Typed Φ t′ A
preservation* {Γ = Γ} d ∎          = Γ , d
preservation* d (st ◅ rest) with preservation d st
... | (_ , d′) = preservation* d′ rest
