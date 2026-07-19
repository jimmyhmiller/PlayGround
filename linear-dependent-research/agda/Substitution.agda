------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 5b, stage 2: the QUANTITATIVE SUBSTITUTION LEMMA.
--
-- A typed simultaneous substitution σ : Fin m → Term n carries, per source
-- variable i, a usage context Δs i typing σ i.  Applying it to a term typed at
-- usage Γ produces a result whose usage is the Γ-weighted sum of the Δs i --
-- the "matrix-vector product"  mvp Γ Δs = Σ_i Γ[i] ·ᶜ Δs i.  We prove
-- substitution preserves typing at exactly that usage.
--
-- Check:  LC_ALL=C.UTF-8 agda Substitution.agda
------------------------------------------------------------------------

module Substitution where

open import Rig
open import Context
open import Syntax
open import Semantics
open import Renaming

------------------------------------------------------------------------
-- the usage matrix-vector product
------------------------------------------------------------------------

mvp : ∀ {m n} → Ctx m → (Fin m → Ctx n) → Ctx n
mvp []      Δs = 𝟘
mvp (g ∷ Γ) Δs = (g ·ᶜ Δs zero) +ᶜ mvp Γ (λ i → Δs (suc i))

-- a "middle-four" swap, used to reassociate the App case
swap-mid : ∀ {n} (a b c d : Ctx n) → (a +ᶜ b) +ᶜ (c +ᶜ d) ≡ (a +ᶜ c) +ᶜ (b +ᶜ d)
swap-mid a b c d =
  trans (+ᶜ-assoc a b (c +ᶜ d))
  (trans (cong (a +ᶜ_) (sym (+ᶜ-assoc b c d)))
  (trans (cong (λ z → a +ᶜ (z +ᶜ d)) (+ᶜ-comm b c))
  (trans (cong (a +ᶜ_) (+ᶜ-assoc c b d))
         (sym (+ᶜ-assoc a c (b +ᶜ d))))))

------------------------------------------------------------------------
-- algebraic laws of mvp
------------------------------------------------------------------------

mvp-𝟘 : ∀ {m n} (Δs : Fin m → Ctx n) → mvp (𝟘 {m}) Δs ≡ 𝟘
mvp-𝟘 {zero}  Δs = refl
mvp-𝟘 {suc m} Δs
  rewrite ·ᶜ-zeroˡ (Δs zero) | mvp-𝟘 {m} (λ i → Δs (suc i)) = +ᶜ-identityˡ 𝟘

mvp-only : ∀ {m n} (i : Fin m) (Δs : Fin m → Ctx n) → mvp (only i) Δs ≡ Δs i
mvp-only zero    Δs
  rewrite ·ᶜ-identityˡ (Δs zero) | mvp-𝟘 (λ i → Δs (suc i)) = +ᶜ-identityʳ (Δs zero)
mvp-only (suc i) Δs
  rewrite ·ᶜ-zeroˡ (Δs zero) | mvp-only i (λ j → Δs (suc j)) = +ᶜ-identityˡ (Δs (suc i))

mvp-+ᶜ : ∀ {m n} (Γ Δ : Ctx m) (Δs : Fin m → Ctx n)
       → mvp (Γ +ᶜ Δ) Δs ≡ mvp Γ Δs +ᶜ mvp Δ Δs
mvp-+ᶜ []      []      Δs = sym (+ᶜ-identityˡ 𝟘)
mvp-+ᶜ (g ∷ Γ) (d ∷ Δ) Δs
  rewrite ·ᶜ-distribʳ-+ g d (Δs zero) | mvp-+ᶜ Γ Δ (λ i → Δs (suc i))
  = swap-mid (g ·ᶜ Δs zero) (d ·ᶜ Δs zero)
             (mvp Γ (λ i → Δs (suc i))) (mvp Δ (λ i → Δs (suc i)))

mvp-·ᶜ : ∀ {m n} (π : Mult) (Γ : Ctx m) (Δs : Fin m → Ctx n)
       → mvp (π ·ᶜ Γ) Δs ≡ π ·ᶜ mvp Γ Δs
mvp-·ᶜ π []      Δs = sym (·ᶜ-zeroʳ π)
mvp-·ᶜ π (g ∷ Γ) Δs
  rewrite ·ᶜ-assoc π g (Δs zero) | mvp-·ᶜ π Γ (λ i → Δs (suc i))
  = sym (·ᶜ-distribˡ-+ᶜ π (g ·ᶜ Δs zero) (mvp Γ (λ i → Δs (suc i))))

mvp-cons0 : ∀ {m n} (Γ : Ctx m) (Δs : Fin m → Ctx n)
          → mvp Γ (λ x → 0# ∷ Δs x) ≡ 0# ∷ mvp Γ Δs
mvp-cons0 []      Δs = refl
mvp-cons0 (g ∷ Γ) Δs
  rewrite mvp-cons0 Γ (λ i → Δs (suc i))
  = cong (_∷ ((g ·ᶜ Δs zero) +ᶜ mvp Γ (λ i → Δs (suc i))))
         (trans (+-identityʳ (g * 0#)) (*-zeroʳ g))

------------------------------------------------------------------------
-- extending a typed substitution under a binder
------------------------------------------------------------------------

extΔ : ∀ {m n} → (Fin m → Ctx n) → (Fin (suc m) → Ctx (suc n))
extΔ Δs zero    = only zero
extΔ Δs (suc x) = 0# ∷ Δs x

mvp-extend : ∀ {m n} (σb : Mult) (Γ : Ctx m) (Δs : Fin m → Ctx n)
           → mvp (σb ∷ Γ) (extΔ Δs) ≡ σb ∷ mvp Γ Δs
mvp-extend σb Γ Δs
  rewrite mvp-cons0 Γ Δs
  = cong₂ _∷_ (trans (+-identityʳ (σb * 1#)) (*-identityʳ σb))
              (trans (cong (_+ᶜ mvp Γ Δs) (·ᶜ-zeroʳ σb)) (+ᶜ-identityˡ (mvp Γ Δs)))

------------------------------------------------------------------------
-- the substitution lemma
------------------------------------------------------------------------

subst-lemma : ∀ {m n} {Φm : TyCtx m} {Φn : TyCtx n} {Γ : Ctx m} {t : Term m} {A}
              {σ : Fin m → Term n} {Δs : Fin m → Ctx n}
            → (∀ i → Φn ⊢[ Δs i ] σ i ⦂ lookupV Φm i)
            → Φm ⊢[ Γ ] t ⦂ A
            → Φn ⊢[ mvp Γ Δs ] subst σ t ⦂ A
subst-lemma {Δs = Δs} hyp (⊢var i)
  rewrite mvp-only i Δs = hyp i
subst-lemma {Δs = Δs} hyp ⊢⟨⟩
  rewrite mvp-𝟘 Δs = ⊢⟨⟩
subst-lemma {Δs = Δs} hyp (⊢app {Γ = Γf} {Δ = Δa} {π = π} fd ad)
  rewrite mvp-+ᶜ Γf (π ·ᶜ Δa) Δs | mvp-·ᶜ π Δa Δs
  = ⊢app (subst-lemma hyp fd) (subst-lemma hyp ad)
subst-lemma {Φm = Φm} {Φn = Φn} {Γ = Γ} {σ = σ} {Δs = Δs} hyp
            (⊢lam {A = A′} {σ = σb} {t = u} bodyD le)
  = ⊢lam (transp (λ G → (A′ ∷ Φn) ⊢[ G ] subst (exts σ) u ⦂ _)
                 (mvp-extend σb Γ Δs)
                 (subst-lemma {Φm = A′ ∷ Φm} {Φn = A′ ∷ Φn}
                              {σ = exts σ} {Δs = extΔ Δs} hyp′ bodyD))
         le
  where
    hyp′ : ∀ j → (A′ ∷ Φn) ⊢[ extΔ Δs j ] exts σ j ⦂ lookupV (A′ ∷ Φm) j
    hyp′ zero    = ⊢var zero
    hyp′ (suc x) = weaken0 A′ (hyp x)
