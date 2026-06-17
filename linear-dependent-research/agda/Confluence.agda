------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 6c: CONFLUENCE of conversion, via Takahashi's complete development.
--
-- This is the genuinely new ingredient the dependent metatheory needs (for
-- canonical forms / Π-injectivity up to conversion).  We define PARALLEL
-- REDUCTION _⇛_, show it is closed under renaming and substitution (using the
-- substitution algebra), define the COMPLETE DEVELOPMENT `cd`, and prove the
-- Takahashi TRIANGLE  (t ⇛ u → u ⇛ cd t).  The DIAMOND property is immediate:
-- if t ⇛ u and t ⇛ v then both reduce to cd t.
--
-- Check:  LC_ALL=C.UTF-8 agda Confluence.agda
------------------------------------------------------------------------

module Confluence where

open import Rig
open import Context using (Nat; zero; suc)
open import Dependent
open import DepSubst

transp : ∀ {A : Set} (P : A → Set) {x y} → x ≡ y → P x → P y
transp P refl px = px

------------------------------------------------------------------------
-- parallel reduction: reduce any set of redexes simultaneously
------------------------------------------------------------------------

data _⇛_ {n} : Tm n → Tm n → Set where
  ⇛var  : ∀ x → var x ⇛ var x
  ⇛⋆    : ⋆ ⇛ ⋆
  ⇛Π    : ∀ {π A A′ B B′} → A ⇛ A′ → B ⇛ B′ → Π π A B ⇛ Π π A′ B′
  ⇛ƛ    : ∀ {b b′} → b ⇛ b′ → ƛ b ⇛ ƛ b′
  ⇛app  : ∀ {f f′ a a′} → f ⇛ f′ → a ⇛ a′ → app f a ⇛ app f′ a′
  ⇛β    : ∀ {b b′ a a′} → b ⇛ b′ → a ⇛ a′ → app (ƛ b) a ⇛ (b′ [ a′ ])
  ⇛Σ    : ∀ {π A A′ B B′} → A ⇛ A′ → B ⇛ B′ → Σ′ π A B ⇛ Σ′ π A′ B′
  ⇛pair : ∀ {a a′ b b′} → a ⇛ a′ → b ⇛ b′ → pair a b ⇛ pair a′ b′

infix 4 _⇛_

par-refl : ∀ {n} (t : Tm n) → t ⇛ t
par-refl (var x)    = ⇛var x
par-refl ⋆          = ⇛⋆
par-refl (Π π A B)  = ⇛Π (par-refl A) (par-refl B)
par-refl (ƛ b)      = ⇛ƛ (par-refl b)
par-refl (app f a)  = ⇛app (par-refl f) (par-refl a)
par-refl (Σ′ π A B) = ⇛Σ (par-refl A) (par-refl B)
par-refl (pair a b) = ⇛pair (par-refl a) (par-refl b)

------------------------------------------------------------------------
-- ⇛ is closed under renaming and substitution
------------------------------------------------------------------------

-- single substitution commutes with renaming (the ren analogue of sub-comm)
ren-comm : ∀ {m n} (ρ : Fin m → Fin n) (t : Tm (suc m)) (a : Tm m)
         → ren ρ (t [ a ]) ≡ (ren (extr ρ) t) [ ren ρ a ]
ren-comm ρ t a =
  trans (ren-sub ρ (sub-head a) t)
        (trans (sub-cong pw t)
               (sym (sub-ren (sub-head (ren ρ a)) (extr ρ) t)))
  where
    pw : ∀ y → ren ρ (sub-head a y) ≡ sub-head (ren ρ a) (extr ρ y)
    pw zero    = refl
    pw (suc x) = refl

par-ren : ∀ {m n} (ρ : Fin m → Fin n) {t t′ : Tm m} → t ⇛ t′ → ren ρ t ⇛ ren ρ t′
par-ren ρ (⇛var x)     = ⇛var (ρ x)
par-ren ρ ⇛⋆           = ⇛⋆
par-ren ρ (⇛Π dA dB)   = ⇛Π (par-ren ρ dA) (par-ren (extr ρ) dB)
par-ren ρ (⇛ƛ db)      = ⇛ƛ (par-ren (extr ρ) db)
par-ren ρ (⇛app df da) = ⇛app (par-ren ρ df) (par-ren ρ da)
par-ren ρ (⇛β {b′ = b′} {a′ = a′} db da) =
  transp (λ z → app (ƛ _) _ ⇛ z) (sym (ren-comm ρ b′ a′))
         (⇛β (par-ren (extr ρ) db) (par-ren ρ da))
par-ren ρ (⇛Σ dA dB)   = ⇛Σ (par-ren ρ dA) (par-ren (extr ρ) dB)
par-ren ρ (⇛pair da db)= ⇛pair (par-ren ρ da) (par-ren ρ db)

par-exts : ∀ {m n} {σ σ′ : Fin m → Tm n}
         → (∀ x → σ x ⇛ σ′ x) → ∀ x → exts σ x ⇛ exts σ′ x
par-exts ps zero    = ⇛var zero
par-exts ps (suc x) = par-ren suc (ps x)

par-sub : ∀ {m n} {σ σ′ : Fin m → Tm n} {t t′ : Tm m}
        → (∀ x → σ x ⇛ σ′ x) → t ⇛ t′ → sub σ t ⇛ sub σ′ t′
par-sub ps (⇛var x)     = ps x
par-sub ps ⇛⋆           = ⇛⋆
par-sub ps (⇛Π dA dB)   = ⇛Π (par-sub ps dA) (par-sub (par-exts ps) dB)
par-sub ps (⇛ƛ db)      = ⇛ƛ (par-sub (par-exts ps) db)
par-sub ps (⇛app df da) = ⇛app (par-sub ps df) (par-sub ps da)
par-sub {σ′ = σ′} ps (⇛β {b′ = b′} {a′ = a′} db da) =
  transp (λ z → app (ƛ _) _ ⇛ z) (sym (sub-comm σ′ b′ a′))
         (⇛β (par-sub (par-exts ps) db) (par-sub ps da))
par-sub ps (⇛Σ dA dB)   = ⇛Σ (par-sub ps dA) (par-sub (par-exts ps) dB)
par-sub ps (⇛pair da db)= ⇛pair (par-sub ps da) (par-sub ps db)

par-sub-head : ∀ {n} {a a′ : Tm n} → a ⇛ a′ → ∀ x → sub-head a x ⇛ sub-head a′ x
par-sub-head da zero    = da
par-sub-head da (suc x) = ⇛var x

-- single-substitution congruence for ⇛ (the shape used by β)
par-[] : ∀ {n} {b b′ : Tm (suc n)} {a a′ : Tm n}
       → b ⇛ b′ → a ⇛ a′ → (b [ a ]) ⇛ (b′ [ a′ ])
par-[] db da = par-sub (par-sub-head da) db

------------------------------------------------------------------------
-- complete development: contract ALL current redexes
------------------------------------------------------------------------

cd : ∀ {n} → Tm n → Tm n
cd (var x)        = var x
cd ⋆              = ⋆
cd (Π π A B)      = Π π (cd A) (cd B)
cd (ƛ b)          = ƛ (cd b)
cd (app (ƛ b) a)  = (cd b) [ cd a ]
cd (app f a)      = app (cd f) (cd a)
cd (Σ′ π A B)     = Σ′ π (cd A) (cd B)
cd (pair a b)     = pair (cd a) (cd b)

------------------------------------------------------------------------
-- Takahashi triangle:  t ⇛ u  ⟹  u ⇛ cd t
------------------------------------------------------------------------

triangle : ∀ {n} {t u : Tm n} → t ⇛ u → u ⇛ cd t
triangle (⇛var x)            = ⇛var x
triangle ⇛⋆                  = ⇛⋆
triangle (⇛Π dA dB)          = ⇛Π (triangle dA) (triangle dB)
triangle (⇛ƛ db)             = ⇛ƛ (triangle db)
triangle (⇛β db da)          = par-[] (triangle db) (triangle da)
triangle (⇛Σ dA dB)          = ⇛Σ (triangle dA) (triangle dB)
triangle (⇛pair da db)       = ⇛pair (triangle da) (triangle db)
-- application: split on the function's reduction so `cd` reduces
triangle (⇛app (⇛var x) da)      = ⇛app (⇛var x) (triangle da)
triangle (⇛app ⇛⋆ da)            = ⇛app ⇛⋆ (triangle da)
triangle (⇛app (⇛Π dA dB) da)    = ⇛app (triangle (⇛Π dA dB)) (triangle da)
triangle (⇛app (⇛ƛ db) da)       = ⇛β (triangle db) (triangle da)
triangle (⇛app (⇛app df da′) da) = ⇛app (triangle (⇛app df da′)) (triangle da)
triangle (⇛app (⇛β db da′) da)   = ⇛app (triangle (⇛β db da′)) (triangle da)
triangle (⇛app (⇛Σ dA dB) da)    = ⇛app (triangle (⇛Σ dA dB)) (triangle da)
triangle (⇛app (⇛pair da′ db) da)= ⇛app (triangle (⇛pair da′ db)) (triangle da)

------------------------------------------------------------------------
-- the diamond property of parallel reduction
------------------------------------------------------------------------

record Diamond {n} (u v : Tm n) : Set where
  constructor diamond⟨_,_,_⟩
  field
    apex   : Tm n
    left   : u ⇛ apex
    right  : v ⇛ apex

diamond : ∀ {n} {t u v : Tm n} → t ⇛ u → t ⇛ v → Diamond u v
diamond {t = t} du dv = diamond⟨ cd t , triangle du , triangle dv ⟩
