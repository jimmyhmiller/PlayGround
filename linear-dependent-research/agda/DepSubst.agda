------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 6b: the SUBSTITUTION ALGEBRA for the dependent syntax.
--
-- The "σ-algebra": how renaming and substitution compose.  This is the reusable
-- foundation for BOTH (a) confluence of conversion (parallel reduction is closed
-- under substitution) and (b) the dependent substitution lemma that preservation
-- needs (substitution into types).  All by induction over Tm, using the
-- ext/exts pointwise lemmas for the binder cases.  The capstone is `sub-comm`,
-- the law that single substitution commutes with a parallel substitution.
--
-- Check:  LC_ALL=C.UTF-8 agda DepSubst.agda
------------------------------------------------------------------------

module DepSubst where

open import Rig
open import Context using (Nat; zero; suc)
open import Dependent

------------------------------------------------------------------------
-- renaming respects pointwise equality, and composes
------------------------------------------------------------------------

extr-cong : ∀ {m n} {ρ ρ′ : Fin m → Fin n}
          → (∀ x → ρ x ≡ ρ′ x) → ∀ x → extr ρ x ≡ extr ρ′ x
extr-cong eq zero    = refl
extr-cong eq (suc x) = cong suc (eq x)

ren-cong : ∀ {m n} {ρ ρ′ : Fin m → Fin n}
         → (∀ x → ρ x ≡ ρ′ x) → ∀ t → ren ρ t ≡ ren ρ′ t
ren-cong eq (var x)    = cong var (eq x)
ren-cong eq ⋆          = refl
ren-cong eq (Π π A B)  = cong₂ (Π π) (ren-cong eq A) (ren-cong (extr-cong eq) B)
ren-cong eq (ƛ b)      = cong ƛ (ren-cong (extr-cong eq) b)
ren-cong eq (app f a)  = cong₂ app (ren-cong eq f) (ren-cong eq a)
ren-cong eq (Σ′ π A B) = cong₂ (Σ′ π) (ren-cong eq A) (ren-cong (extr-cong eq) B)
ren-cong eq (pair a b) = cong₂ pair (ren-cong eq a) (ren-cong eq b)

extr-comp : ∀ {ℓ m n} (ρ : Fin m → Fin n) (ρ′ : Fin ℓ → Fin m)
          → ∀ x → extr ρ (extr ρ′ x) ≡ extr (λ y → ρ (ρ′ y)) x
extr-comp ρ ρ′ zero    = refl
extr-comp ρ ρ′ (suc x) = refl

ren-comp : ∀ {ℓ m n} (ρ : Fin m → Fin n) (ρ′ : Fin ℓ → Fin m)
         → ∀ t → ren ρ (ren ρ′ t) ≡ ren (λ y → ρ (ρ′ y)) t
ren-comp ρ ρ′ (var x)    = refl
ren-comp ρ ρ′ ⋆          = refl
ren-comp ρ ρ′ (Π π A B)  = cong₂ (Π π) (ren-comp ρ ρ′ A)
                             (trans (ren-comp (extr ρ) (extr ρ′) B) (ren-cong (extr-comp ρ ρ′) B))
ren-comp ρ ρ′ (ƛ b)      = cong ƛ (trans (ren-comp (extr ρ) (extr ρ′) b) (ren-cong (extr-comp ρ ρ′) b))
ren-comp ρ ρ′ (app f a)  = cong₂ app (ren-comp ρ ρ′ f) (ren-comp ρ ρ′ a)
ren-comp ρ ρ′ (Σ′ π A B) = cong₂ (Σ′ π) (ren-comp ρ ρ′ A)
                             (trans (ren-comp (extr ρ) (extr ρ′) B) (ren-cong (extr-comp ρ ρ′) B))
ren-comp ρ ρ′ (pair a b) = cong₂ pair (ren-comp ρ ρ′ a) (ren-comp ρ ρ′ b)

------------------------------------------------------------------------
-- substitution respects pointwise equality, and is the identity on `var`
------------------------------------------------------------------------

exts-cong : ∀ {m n} {σ σ′ : Fin m → Tm n}
          → (∀ x → σ x ≡ σ′ x) → ∀ x → exts σ x ≡ exts σ′ x
exts-cong eq zero    = refl
exts-cong eq (suc x) = cong (ren suc) (eq x)

sub-cong : ∀ {m n} {σ σ′ : Fin m → Tm n}
         → (∀ x → σ x ≡ σ′ x) → ∀ t → sub σ t ≡ sub σ′ t
sub-cong eq (var x)    = eq x
sub-cong eq ⋆          = refl
sub-cong eq (Π π A B)  = cong₂ (Π π) (sub-cong eq A) (sub-cong (exts-cong eq) B)
sub-cong eq (ƛ b)      = cong ƛ (sub-cong (exts-cong eq) b)
sub-cong eq (app f a)  = cong₂ app (sub-cong eq f) (sub-cong eq a)
sub-cong eq (Σ′ π A B) = cong₂ (Σ′ π) (sub-cong eq A) (sub-cong (exts-cong eq) B)
sub-cong eq (pair a b) = cong₂ pair (sub-cong eq a) (sub-cong eq b)

exts-id : ∀ {n} (x : Fin (suc n)) → exts var x ≡ var x
exts-id zero    = refl
exts-id (suc x) = refl

sub-id : ∀ {n} (t : Tm n) → sub var t ≡ t
sub-id (var x)    = refl
sub-id ⋆          = refl
sub-id (Π π A B)  = cong₂ (Π π) (sub-id A) (trans (sub-cong exts-id B) (sub-id B))
sub-id (ƛ b)      = cong ƛ (trans (sub-cong exts-id b) (sub-id b))
sub-id (app f a)  = cong₂ app (sub-id f) (sub-id a)
sub-id (Σ′ π A B) = cong₂ (Σ′ π) (sub-id A) (trans (sub-cong exts-id B) (sub-id B))
sub-id (pair a b) = cong₂ pair (sub-id a) (sub-id b)

------------------------------------------------------------------------
-- fusion laws: sub-after-ren, ren-after-sub, sub-after-sub
------------------------------------------------------------------------

exts-extr : ∀ {ℓ m n} (σ : Fin m → Tm n) (ρ : Fin ℓ → Fin m)
          → ∀ x → exts σ (extr ρ x) ≡ exts (λ y → σ (ρ y)) x
exts-extr σ ρ zero    = refl
exts-extr σ ρ (suc x) = refl

sub-ren : ∀ {ℓ m n} (σ : Fin m → Tm n) (ρ : Fin ℓ → Fin m)
        → ∀ t → sub σ (ren ρ t) ≡ sub (λ y → σ (ρ y)) t
sub-ren σ ρ (var x)    = refl
sub-ren σ ρ ⋆          = refl
sub-ren σ ρ (Π π A B)  = cong₂ (Π π) (sub-ren σ ρ A)
                           (trans (sub-ren (exts σ) (extr ρ) B) (sub-cong (exts-extr σ ρ) B))
sub-ren σ ρ (ƛ b)      = cong ƛ (trans (sub-ren (exts σ) (extr ρ) b) (sub-cong (exts-extr σ ρ) b))
sub-ren σ ρ (app f a)  = cong₂ app (sub-ren σ ρ f) (sub-ren σ ρ a)
sub-ren σ ρ (Σ′ π A B) = cong₂ (Σ′ π) (sub-ren σ ρ A)
                           (trans (sub-ren (exts σ) (extr ρ) B) (sub-cong (exts-extr σ ρ) B))
sub-ren σ ρ (pair a b) = cong₂ pair (sub-ren σ ρ a) (sub-ren σ ρ b)

extr-exts : ∀ {ℓ m n} (ρ : Fin m → Fin n) (σ : Fin ℓ → Tm m)
          → ∀ x → ren (extr ρ) (exts σ x) ≡ exts (λ y → ren ρ (σ y)) x
extr-exts ρ σ zero    = refl
extr-exts ρ σ (suc x) =
  trans (ren-comp (extr ρ) suc (σ x)) (sym (ren-comp suc ρ (σ x)))

ren-sub : ∀ {ℓ m n} (ρ : Fin m → Fin n) (σ : Fin ℓ → Tm m)
        → ∀ t → ren ρ (sub σ t) ≡ sub (λ y → ren ρ (σ y)) t
ren-sub ρ σ (var x)    = refl
ren-sub ρ σ ⋆          = refl
ren-sub ρ σ (Π π A B)  = cong₂ (Π π) (ren-sub ρ σ A)
                           (trans (ren-sub (extr ρ) (exts σ) B) (sub-cong (extr-exts ρ σ) B))
ren-sub ρ σ (ƛ b)      = cong ƛ (trans (ren-sub (extr ρ) (exts σ) b) (sub-cong (extr-exts ρ σ) b))
ren-sub ρ σ (app f a)  = cong₂ app (ren-sub ρ σ f) (ren-sub ρ σ a)
ren-sub ρ σ (Σ′ π A B) = cong₂ (Σ′ π) (ren-sub ρ σ A)
                           (trans (ren-sub (extr ρ) (exts σ) B) (sub-cong (extr-exts ρ σ) B))
ren-sub ρ σ (pair a b) = cong₂ pair (ren-sub ρ σ a) (ren-sub ρ σ b)

exts-comp : ∀ {ℓ m n} (σ : Fin m → Tm n) (τ : Fin ℓ → Tm m)
          → ∀ x → sub (exts σ) (exts τ x) ≡ exts (λ y → sub σ (τ y)) x
exts-comp σ τ zero    = refl
exts-comp σ τ (suc x) =
  trans (sub-ren (exts σ) suc (τ x)) (sym (ren-sub suc σ (τ x)))

sub-comp : ∀ {ℓ m n} (σ : Fin m → Tm n) (τ : Fin ℓ → Tm m)
         → ∀ t → sub σ (sub τ t) ≡ sub (λ y → sub σ (τ y)) t
sub-comp σ τ (var x)    = refl
sub-comp σ τ ⋆          = refl
sub-comp σ τ (Π π A B)  = cong₂ (Π π) (sub-comp σ τ A)
                            (trans (sub-comp (exts σ) (exts τ) B) (sub-cong (exts-comp σ τ) B))
sub-comp σ τ (ƛ b)      = cong ƛ (trans (sub-comp (exts σ) (exts τ) b) (sub-cong (exts-comp σ τ) b))
sub-comp σ τ (app f a)  = cong₂ app (sub-comp σ τ f) (sub-comp σ τ a)
sub-comp σ τ (Σ′ π A B) = cong₂ (Σ′ π) (sub-comp σ τ A)
                            (trans (sub-comp (exts σ) (exts τ) B) (sub-cong (exts-comp σ τ) B))
sub-comp σ τ (pair a b) = cong₂ pair (sub-comp σ τ a) (sub-comp σ τ b)

------------------------------------------------------------------------
-- the capstone: single substitution commutes with a parallel substitution
--   sub σ (t [ a ])  ≡  (sub (exts σ) t) [ sub σ a ]
------------------------------------------------------------------------

sub-comm : ∀ {m n} (σ : Fin m → Tm n) (t : Tm (suc m)) (a : Tm m)
         → sub σ (t [ a ]) ≡ (sub (exts σ) t) [ sub σ a ]
sub-comm σ t a =
  trans (sub-comp σ (sub-head a) t)
        (trans (sub-cong pointwise t)
               (sym (sub-comp (sub-head (sub σ a)) (exts σ) t)))
  where
    pointwise : ∀ y → sub σ (sub-head a y) ≡ sub (sub-head (sub σ a)) (exts σ y)
    pointwise zero    = refl
    pointwise (suc x) = sym (trans (sub-ren (sub-head (sub σ a)) suc (σ x)) (sub-id (σ x)))
