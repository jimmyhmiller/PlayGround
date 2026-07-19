------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 5b, stage 1: RENAMING / WEAKENING.
--
-- The substitution lemma (stage 2) needs to push a typing derivation under a
-- binder, which requires weakening: inserting a fresh, UNUSED (multiplicity 0#)
-- variable anywhere in the context.  We prove the general "insert a variable at
-- position k" weakening lemma, which is what closes the induction under binders.
--
-- Check:  LC_ALL=C.UTF-8 agda Renaming.agda
------------------------------------------------------------------------

module Renaming where

open import Rig
open import Context
open import Syntax
open import Semantics

-- transport along propositional equality
transp : ∀ {A : Set} (P : A → Set) {x y} → x ≡ y → P x → P y
transp P refl px = px

------------------------------------------------------------------------
-- renamings respect pointwise equality
------------------------------------------------------------------------

ext-cong : ∀ {m n} {ρ ρ′ : Fin m → Fin n}
         → (∀ x → ρ x ≡ ρ′ x) → ∀ x → ext ρ x ≡ ext ρ′ x
ext-cong eq zero    = refl
ext-cong eq (suc x) = cong suc (eq x)

rename-cong : ∀ {m n} {ρ ρ′ : Fin m → Fin n}
            → (∀ x → ρ x ≡ ρ′ x) → ∀ t → rename ρ t ≡ rename ρ′ t
rename-cong eq (var x)   = cong var (eq x)
rename-cong eq ⟨⟩        = refl
rename-cong eq (lam t)   = cong lam (rename-cong (ext-cong eq) t)
rename-cong eq (app f a) = cong₂ app (rename-cong eq f) (rename-cong eq a)

------------------------------------------------------------------------
-- inserting a variable at position k, and the renaming that skips it
------------------------------------------------------------------------

insertV : ∀ {A : Set} {n} → Fin (suc n) → A → Vec A n → Vec A (suc n)
insertV zero    a xs       = a ∷ xs
insertV (suc k) a (x ∷ xs) = x ∷ insertV k a xs

punchIn : ∀ {n} → Fin (suc n) → Fin n → Fin (suc n)
punchIn zero    i       = suc i
punchIn (suc k) zero    = zero
punchIn (suc k) (suc i) = suc (punchIn k i)

-- punchIn at a successor is ext of punchIn (pointwise)
punchIn-ext : ∀ {n} (k : Fin (suc n)) (i : Fin (suc n))
            → ext (punchIn k) i ≡ punchIn (suc k) i
punchIn-ext k zero    = refl
punchIn-ext k (suc i) = refl

------------------------------------------------------------------------
-- how insertion interacts with lookup and the context operations
------------------------------------------------------------------------

lookup-insert : ∀ {A : Set} {m} (k : Fin (suc m)) (C : A) (Φ : Vec A m) (i : Fin m)
              → lookupV Φ i ≡ lookupV (insertV k C Φ) (punchIn k i)
lookup-insert zero    C Φ        i       = refl
lookup-insert (suc k) C (x ∷ Φ)  zero    = refl
lookup-insert (suc k) C (x ∷ Φ)  (suc i) = lookup-insert k C Φ i

insert-𝟘 : ∀ {n} (k : Fin (suc n)) → insertV k 0# (𝟘 {n}) ≡ 𝟘
insert-𝟘         zero    = refl
insert-𝟘 {suc n} (suc k) = cong (0# ∷_) (insert-𝟘 k)

insert-only : ∀ {n} (k : Fin (suc n)) (i : Fin n)
            → insertV k 0# (only i) ≡ only (punchIn k i)
insert-only zero    i       = refl
insert-only (suc k) zero    = cong (1# ∷_) (insert-𝟘 k)
insert-only (suc k) (suc i) = cong (0# ∷_) (insert-only k i)

insert-+ᶜ : ∀ {n} (k : Fin (suc n)) (Γ Δ : Ctx n)
          → insertV k 0# (Γ +ᶜ Δ) ≡ insertV k 0# Γ +ᶜ insertV k 0# Δ
insert-+ᶜ zero    Γ        Δ        = refl
insert-+ᶜ (suc k) (g ∷ Γ)  (d ∷ Δ)  = cong ((g + d) ∷_) (insert-+ᶜ k Γ Δ)

insert-·ᶜ : ∀ {n} (k : Fin (suc n)) (π : Mult) (Γ : Ctx n)
          → insertV k 0# (π ·ᶜ Γ) ≡ π ·ᶜ insertV k 0# Γ
insert-·ᶜ zero    π Γ        = cong (_∷ (π ·ᶜ Γ)) (sym (*-zeroʳ π))
insert-·ᶜ (suc k) π (g ∷ Γ)  = cong ((π * g) ∷_) (insert-·ᶜ k π Γ)

------------------------------------------------------------------------
-- the weakening lemma
------------------------------------------------------------------------

wk : ∀ {m} {Φ : TyCtx m} {Γ : Ctx m} {t : Term m} {A}
     (k : Fin (suc m)) (C : Ty)
   → Φ ⊢[ Γ ] t ⦂ A
   → insertV k C Φ ⊢[ insertV k 0# Γ ] rename (punchIn k) t ⦂ A
wk {Φ = Φ} k C (⊢var i)
  rewrite insert-only k i | lookup-insert k C Φ i = ⊢var (punchIn k i)
wk k C ⊢⟨⟩
  rewrite insert-𝟘 k = ⊢⟨⟩
wk k C (⊢app {Γ = Γ} {Δ = Δ} {π = π} fd ad)
  rewrite insert-+ᶜ k Γ (π ·ᶜ Δ) | insert-·ᶜ k π Δ = ⊢app (wk k C fd) (wk k C ad)
wk {Φ = Φ} {Γ = Γ} k C (⊢lam {A = A′} {σ = σ} {t = u} bodyD le) =
  ⊢lam (transp (λ T → (A′ ∷ insertV k C Φ) ⊢[ σ ∷ insertV k 0# Γ ] T ⦂ _)
               (sym (rename-cong (punchIn-ext k) u))
               (wk (suc k) C bodyD))
       le

-- weakening at the front: a fresh unused variable as de Bruijn 0
weaken0 : ∀ {m} {Φ : TyCtx m} {Γ : Ctx m} {t : Term m} {A} (C : Ty)
        → Φ ⊢[ Γ ] t ⦂ A
        → (C ∷ Φ) ⊢[ 0# ∷ Γ ] rename suc t ⦂ A
weaken0 {t = t} C d =
  transp (λ T → (C ∷ _) ⊢[ 0# ∷ _ ] T ⦂ _)
         (rename-cong (λ _ → refl) t)
         (wk zero C d)
