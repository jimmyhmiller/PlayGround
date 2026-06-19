------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 4: OPERATIONAL SEMANTICS of the core (call-by-value small-step).
--
-- Standard de Bruijn renaming + substitution, values, and a β/ξ reduction
-- relation _⟶_.  This is the dynamics the metatheory (M5) will prove the
-- quantitative typing of Module 3 sound for -- i.e. the unbounded counterpart
-- of the Rosette resource-soundness check.
--
-- Check:  agda Semantics.agda
------------------------------------------------------------------------

module Semantics where

open import Rig
open import Context
open import Syntax

------------------------------------------------------------------------
-- Renaming and substitution (de Bruijn)
------------------------------------------------------------------------

ext : ∀ {m n} → (Fin m → Fin n) → Fin (suc m) → Fin (suc n)
ext ρ zero    = zero
ext ρ (suc x) = suc (ρ x)

rename : ∀ {m n} → (Fin m → Fin n) → Term m → Term n
rename ρ (var x)   = var (ρ x)
rename ρ ⟨⟩        = ⟨⟩
rename ρ (lam t)   = lam (rename (ext ρ) t)
rename ρ (app f a) = app (rename ρ f) (rename ρ a)

exts : ∀ {m n} → (Fin m → Term n) → Fin (suc m) → Term (suc n)
exts σ zero    = var zero
exts σ (suc x) = rename suc (σ x)

subst : ∀ {m n} → (Fin m → Term n) → Term m → Term n
subst σ (var x)   = σ x
subst σ ⟨⟩        = ⟨⟩
subst σ (lam t)   = lam (subst (exts σ) t)
subst σ (app f a) = app (subst σ f) (subst σ a)

-- the substitution that replaces de Bruijn 0 by s (named so the metatheory can
-- refer to it)
sub-head : ∀ {n} → Term n → Fin (suc n) → Term n
sub-head s zero    = s
sub-head s (suc x) = var x

-- single-variable substitution: t [ s ]  replaces de Bruijn 0 by s
_[_] : ∀ {n} → Term (suc n) → Term n → Term n
t [ s ] = subst (sub-head s) t

------------------------------------------------------------------------
-- Values and call-by-value reduction
------------------------------------------------------------------------

data Value : ∀ {n} → Term n → Set where
  V-⟨⟩  : ∀ {n} → Value (⟨⟩ {n})
  V-lam : ∀ {n} {t : Term (suc n)} → Value (lam t)

infix 2 _⟶_

data _⟶_ : ∀ {n} → Term n → Term n → Set where
  ξ-app₁ : ∀ {n} {f f′ a : Term n}
         → f ⟶ f′ → app f a ⟶ app f′ a
  ξ-app₂ : ∀ {n} {f a a′ : Term n}
         → Value f → a ⟶ a′ → app f a ⟶ app f a′
  β-lam  : ∀ {n} {t : Term (suc n)} {v : Term n}
         → Value v → app (lam t) v ⟶ (t [ v ])

-- reflexive-transitive closure (a reduction sequence)
infix 2 _⟶*_

data _⟶*_ : ∀ {n} → Term n → Term n → Set where
  ∎   : ∀ {n} {t : Term n} → t ⟶* t
  _◅_ : ∀ {n} {t u w : Term n} → t ⟶ u → u ⟶* w → t ⟶* w

infixr 5 _◅_

------------------------------------------------------------------------
-- Worked reductions
------------------------------------------------------------------------

-- (λx. x) ⟨⟩  ⟶  ⟨⟩
id-app : ∀ {n} → app {n} (lam (var zero)) ⟨⟩ ⟶ ⟨⟩
id-app = β-lam V-⟨⟩

-- (λx. λy. x) ⟨⟩  ⟶  λy. ⟨⟩   in one step (the erased y is discarded)
K-app : ∀ {n} → app {n} (lam (lam (var (suc zero)))) ⟨⟩ ⟶ lam ⟨⟩
K-app = β-lam V-⟨⟩

-- a two-step sequence:  (λx. x) ((λx. x) ⟨⟩)  ⟶*  ⟨⟩
seq2 : ∀ {n} → app {n} (lam (var zero)) (app (lam (var zero)) ⟨⟩) ⟶* ⟨⟩
seq2 = ξ-app₂ V-lam (β-lam V-⟨⟩) ◅ β-lam V-⟨⟩ ◅ ∎
