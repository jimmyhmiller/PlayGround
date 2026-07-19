------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 2: usage contexts as a MODULE over the rig.
--
-- A typing context carries, for each variable, the multiplicity at which it is
-- used.  The quantitative typing rules manipulate these with two operations:
--   * Γ +ᶜ Δ : combine the usages of two subderivations (the App/pair rules);
--   * π ·ᶜ Γ : scale a context by a multiplicity (passing through a binder of
--              multiplicity π -- the App rule's `π · usage(arg)`).
-- The right abstraction (Wood & Atkey, "A linear algebra approach to linear
-- metatheory") is: contexts form a *left module* over the semiring R.  This
-- module proves exactly those module laws, by lifting the rig laws of Module 1
-- pointwise over vectors.  They are the algebraic facts every context-splitting
-- step in the metatheory will appeal to.
--
-- Self-contained except for `Rig` (same directory).  Check:  agda Context.agda
------------------------------------------------------------------------

module Context where

open import Rig

------------------------------------------------------------------------
-- Vectors of multiplicities, indexed by length
------------------------------------------------------------------------

data Nat : Set where
  zero : Nat
  suc  : Nat → Nat

data Vec (A : Set) : Nat → Set where
  []  : Vec A zero
  _∷_ : ∀ {n} → A → Vec A n → Vec A (suc n)

infixr 5 _∷_

Ctx : Nat → Set
Ctx n = Vec Mult n

-- the all-zero context: every variable used zero times
𝟘 : ∀ {n} → Ctx n
𝟘 {zero}  = []
𝟘 {suc n} = 0# ∷ 𝟘

infixl 6 _+ᶜ_
infixl 7 _·ᶜ_

-- pointwise addition of usages
_+ᶜ_ : ∀ {n} → Ctx n → Ctx n → Ctx n
[]       +ᶜ []       = []
(x ∷ xs) +ᶜ (y ∷ ys) = (x + y) ∷ (xs +ᶜ ys)

-- scaling a whole context by a multiplicity
_·ᶜ_ : ∀ {n} → Mult → Ctx n → Ctx n
π ·ᶜ []       = []
π ·ᶜ (x ∷ xs) = (π * x) ∷ (π ·ᶜ xs)

------------------------------------------------------------------------
-- (Ctx n, +ᶜ, 𝟘) is a commutative monoid  (lifting the + monoid)
------------------------------------------------------------------------

+ᶜ-identityˡ : ∀ {n} (Γ : Ctx n) → 𝟘 +ᶜ Γ ≡ Γ
+ᶜ-identityˡ []       = refl
+ᶜ-identityˡ (x ∷ xs) = cong₂ _∷_ (+-identityˡ x) (+ᶜ-identityˡ xs)

+ᶜ-identityʳ : ∀ {n} (Γ : Ctx n) → Γ +ᶜ 𝟘 ≡ Γ
+ᶜ-identityʳ []       = refl
+ᶜ-identityʳ (x ∷ xs) = cong₂ _∷_ (+-identityʳ x) (+ᶜ-identityʳ xs)

+ᶜ-comm : ∀ {n} (Γ Δ : Ctx n) → Γ +ᶜ Δ ≡ Δ +ᶜ Γ
+ᶜ-comm []       []       = refl
+ᶜ-comm (x ∷ xs) (y ∷ ys) = cong₂ _∷_ (+-comm x y) (+ᶜ-comm xs ys)

+ᶜ-assoc : ∀ {n} (Γ Δ Θ : Ctx n) → (Γ +ᶜ Δ) +ᶜ Θ ≡ Γ +ᶜ (Δ +ᶜ Θ)
+ᶜ-assoc []       []       []       = refl
+ᶜ-assoc (x ∷ xs) (y ∷ ys) (z ∷ zs) = cong₂ _∷_ (+-assoc x y z) (+ᶜ-assoc xs ys zs)

------------------------------------------------------------------------
-- Scaling makes Ctx a left module over the rig
------------------------------------------------------------------------

-- 1 · Γ = Γ
·ᶜ-identityˡ : ∀ {n} (Γ : Ctx n) → 1# ·ᶜ Γ ≡ Γ
·ᶜ-identityˡ []       = refl
·ᶜ-identityˡ (x ∷ xs) = cong₂ _∷_ (*-identityˡ x) (·ᶜ-identityˡ xs)

-- 0 · Γ = 𝟘   (erasure: scaling by 0 zeroes every usage)
·ᶜ-zeroˡ : ∀ {n} (Γ : Ctx n) → 0# ·ᶜ Γ ≡ 𝟘
·ᶜ-zeroˡ []       = refl
·ᶜ-zeroˡ (x ∷ xs) = cong₂ _∷_ (*-zeroˡ x) (·ᶜ-zeroˡ xs)

-- π · 𝟘 = 𝟘
·ᶜ-zeroʳ : ∀ {n} (π : Mult) → π ·ᶜ 𝟘 {n} ≡ 𝟘
·ᶜ-zeroʳ {zero}  π = refl
·ᶜ-zeroʳ {suc n} π = cong₂ _∷_ (*-zeroʳ π) (·ᶜ-zeroʳ {n} π)

-- π · (Γ +ᶜ Δ) = (π · Γ) +ᶜ (π · Δ)   (scaling distributes over context add)
·ᶜ-distribˡ-+ᶜ : ∀ {n} (π : Mult) (Γ Δ : Ctx n)
               → π ·ᶜ (Γ +ᶜ Δ) ≡ (π ·ᶜ Γ) +ᶜ (π ·ᶜ Δ)
·ᶜ-distribˡ-+ᶜ π []       []       = refl
·ᶜ-distribˡ-+ᶜ π (x ∷ xs) (y ∷ ys) =
  cong₂ _∷_ (*-distribˡ-+ π x y) (·ᶜ-distribˡ-+ᶜ π xs ys)

-- (π + ρ) · Γ = (π · Γ) +ᶜ (ρ · Γ)
·ᶜ-distribʳ-+ : ∀ {n} (π ρ : Mult) (Γ : Ctx n)
              → (π + ρ) ·ᶜ Γ ≡ (π ·ᶜ Γ) +ᶜ (ρ ·ᶜ Γ)
·ᶜ-distribʳ-+ π ρ []       = refl
·ᶜ-distribʳ-+ π ρ (x ∷ xs) =
  cong₂ _∷_ (*-distribʳ-+ x π ρ) (·ᶜ-distribʳ-+ π ρ xs)

-- (π * ρ) · Γ = π · (ρ · Γ)
·ᶜ-assoc : ∀ {n} (π ρ : Mult) (Γ : Ctx n)
         → (π * ρ) ·ᶜ Γ ≡ π ·ᶜ (ρ ·ᶜ Γ)
·ᶜ-assoc π ρ []       = refl
·ᶜ-assoc π ρ (x ∷ xs) = cong₂ _∷_ (*-assoc π ρ x) (·ᶜ-assoc π ρ xs)
