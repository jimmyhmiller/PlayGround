------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 5a: PROGRESS  (one half of type safety).
--
-- A well-typed CLOSED term is either a value or it can take a step.  Proved by
-- recursion on the term (so the absurd `var` case is refuted directly via the
-- empty `Fin zero`), using the typing derivation for canonical forms (a closed
-- term at a function type that is a value must be a lambda).
--
-- The unbounded counterpart of the Rosette "no stuck states" check.  The other
-- half, preservation, is in Preservation.agda.
--
-- Check:  LC_ALL=C.UTF-8 agda Progress.agda
------------------------------------------------------------------------

module Progress where

open import Rig
open import Context
open import Syntax
open import Semantics

⊥-elim : ∀ {A : Set} → ⊥ → A
⊥-elim ()

-- a closed term of function type is never ⟨⟩
⟨⟩-not-fun : ∀ {Φ : TyCtx zero} {Γ A B π} → ¬ (Φ ⊢[ Γ ] ⟨⟩ ⦂ (A ⇒[ π ] B))
⟨⟩-not-fun ()

data Progress (t : Term zero) : Set where
  done : Value t          → Progress t
  step : ∀ {t′} → t ⟶ t′ → Progress t

-- contexts are left abstract: at n = zero they are vacuous, and keeping them
-- abstract lets the App/Var rules' index expressions unify directly (Vec has no
-- eta, so a fixed `[]` would get stuck against `Γ +ᶜ π ·ᶜ Δ`).
progress : ∀ {Φ : TyCtx zero} {Γ : Ctx zero} {A}
         (t : Term zero) → Φ ⊢[ Γ ] t ⦂ A → Progress t
progress (var ()) _
progress ⟨⟩       _            = done V-⟨⟩
progress (lam b)  _            = done V-lam
progress (app f a) (⊢app fd ad) with progress f fd
... | step fs                  = step (ξ-app₁ fs)
... | done V-⟨⟩                = ⊥-elim (⟨⟩-not-fun fd)   -- f:⟨⟩ at function type: impossible
... | done V-lam with progress a ad                       -- f is a lambda
...   | step as                = step (ξ-app₂ V-lam as)
...   | done va                = step (β-lam va)
