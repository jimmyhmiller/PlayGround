------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 3: syntax and QUANTITATIVE TYPING of the linear core.
--
-- A de Bruijn lambda calculus whose typing judgment
--       Φ ⊢[ Γ ] t ⦂ A
-- carries, alongside the type context Φ, a USAGE context Γ (a vector of
-- multiplicities) recording how many times each variable is used.  The rules
-- are the quantitative ones, and they are where Modules 1-2 pay off:
--   * a variable uses itself once          (only i : 1# at i, 0# elsewhere)
--   * application combines and SCALES       (Γ +ᶜ (π ·ᶜ Δ))
--   * a lambda checks the bound variable's usage σ against its budget π via ⊑
--     (σ ⊑ π:  1# admits only 1# = linear; 0# only 0# = erased; ω anything)
--
-- The worked derivations at the bottom show a LINEAR identity and a K combinator
-- whose second argument is ERASED (multiplicity 0#) -- linearity and erasure in
-- the same system, type-checked.
--
-- Check:  agda Syntax.agda
------------------------------------------------------------------------

module Syntax where

open import Rig
open import Context

------------------------------------------------------------------------
-- de Bruijn indices, vector lookup, and the "use exactly variable i" context
------------------------------------------------------------------------

data Fin : Nat → Set where
  zero : ∀ {n} → Fin (suc n)
  suc  : ∀ {n} → Fin n → Fin (suc n)

lookupV : ∀ {A : Set} {n} → Vec A n → Fin n → A
lookupV (x ∷ xs) zero    = x
lookupV (x ∷ xs) (suc i) = lookupV xs i

-- the usage context that uses variable i exactly once
only : ∀ {n} → Fin n → Ctx n
only zero    = 1# ∷ 𝟘
only (suc i) = 0# ∷ only i

------------------------------------------------------------------------
-- Types and terms
------------------------------------------------------------------------

infixr 7 _⇒[_]_

data Ty : Set where
  ⋆      : Ty                    -- unit
  _⇒[_]_ : Ty → Mult → Ty → Ty   -- function whose argument has multiplicity π

TyCtx : Nat → Set
TyCtx n = Vec Ty n

data Term : Nat → Set where
  var : ∀ {n} → Fin n → Term n
  ⟨⟩  : ∀ {n} → Term n
  lam : ∀ {n} → Term (suc n) → Term n
  app : ∀ {n} → Term n → Term n → Term n

------------------------------------------------------------------------
-- The quantitative typing judgment
------------------------------------------------------------------------

infix 3 _⊢[_]_⦂_

data _⊢[_]_⦂_ : ∀ {n} → TyCtx n → Ctx n → Term n → Ty → Set where

  ⊢var : ∀ {n} {Φ : TyCtx n} (i : Fin n)
       → Φ ⊢[ only i ] var i ⦂ lookupV Φ i

  ⊢⟨⟩  : ∀ {n} {Φ : TyCtx n}
       → Φ ⊢[ 𝟘 ] ⟨⟩ ⦂ ⋆

  -- the bound variable is used σ times; that must be within its budget π
  ⊢lam : ∀ {n} {Φ : TyCtx n} {Γ : Ctx n} {A B} {π σ} {t : Term (suc n)}
       → (A ∷ Φ) ⊢[ σ ∷ Γ ] t ⦂ B
       → σ ⊑ π
       → Φ ⊢[ Γ ] lam t ⦂ (A ⇒[ π ] B)

  -- the argument's usage Δ is scaled by the function's argument-multiplicity π
  ⊢app : ∀ {n} {Φ : TyCtx n} {Γ Δ : Ctx n} {A B} {π} {f a : Term n}
       → Φ ⊢[ Γ ] f ⦂ (A ⇒[ π ] B)
       → Φ ⊢[ Δ ] a ⦂ A
       → Φ ⊢[ Γ +ᶜ (π ·ᶜ Δ) ] app f a ⦂ B

------------------------------------------------------------------------
-- Worked derivations (closed terms: empty type/usage contexts)
------------------------------------------------------------------------

-- the LINEAR identity:  λx. x   :   ⋆ ⇒[1] ⋆     (x used exactly once)
id-linear : [] ⊢[ [] ] lam (var zero) ⦂ (⋆ ⇒[ 1# ] ⋆)
id-linear = ⊢lam (⊢var zero) ⊑-refl

-- unit
unit : [] ⊢[ [] ] ⟨⟩ ⦂ ⋆
unit = ⊢⟨⟩

-- K with an ERASED second argument:  λx. λy. x   :   ⋆ ⇒[1] (⋆ ⇒[0] ⋆)
-- y is bound at multiplicity 0# and used 0 times (0# ⊑ 0#); x is linear.
K-erased : [] ⊢[ [] ] lam (lam (var (suc zero))) ⦂ (⋆ ⇒[ 1# ] (⋆ ⇒[ 0# ] ⋆))
K-erased = ⊢lam (⊢lam (⊢var (suc zero)) ⊑-refl) ⊑-refl
