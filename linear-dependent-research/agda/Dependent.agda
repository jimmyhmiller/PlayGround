------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 6: the DEPENDENT quantitative calculus (syntax + typing).
--
-- Extends the linear core (Modules 1-5) to DEPENDENT types: a universe, a
-- dependent function type Π[ π ] A B whose codomain B may mention the argument,
-- λ, application, and definitional conversion.  Crucially it realises QTT's
-- "0-fragment": TYPE FORMATION happens at multiplicity 0 (types are built in the
-- erased fragment), while the binder multiplicity π governs RUNTIME use.  So a
-- single system carries:  erasure (0), linearity (1), unrestricted (ω), AND
-- dependency -- which is the whole thesis.
--
-- The worked example `dep-id` is the dependent identity
--     λA. λx. x  :  Π (A :^0 ⋆). Π (x :^1 A). A
-- machine-checked: the type argument A is used at multiplicity 0 (only in
-- types), while x is used linearly (exactly once) -- in a genuinely DEPENDENT
-- setting where x's type *is* the bound type variable A.
--
-- Scope: this module establishes the syntax, the quantitative dependent typing
-- rules, and inhabitation. The metatheory (progress + preservation) for the
-- dependent system needs confluence of conversion (and consistency needs
-- normalisation); see Modules 1-5 for the linear core's full type-safety proof
-- and README for the dependent metatheory plan.  We use ⋆ : ⋆ here (type safety
-- does not need consistency); a universe hierarchy is the orthogonal fix.
--
-- Check:  LC_ALL=C.UTF-8 agda Dependent.agda
------------------------------------------------------------------------

module Dependent where

open import Rig
open import Context using (Nat; zero; suc; Vec; []; _∷_; Ctx; 𝟘; _+ᶜ_; _·ᶜ_)

------------------------------------------------------------------------
-- de Bruijn indices and the "use variable i once" usage context
------------------------------------------------------------------------

data Fin : Nat → Set where
  zero : ∀ {n} → Fin (suc n)
  suc  : ∀ {n} → Fin n → Fin (suc n)

only : ∀ {n} → Fin n → Ctx n
only zero    = 1# ∷ 𝟘
only (suc i) = 0# ∷ only i

------------------------------------------------------------------------
-- syntax: terms and types are one sort (PTS-style)
------------------------------------------------------------------------

data Tm : Nat → Set where
  var  : ∀ {n} → Fin n → Tm n
  ⋆    : ∀ {n} → Tm n                            -- universe (⋆ : ⋆)
  Π    : ∀ {n} → Mult → Tm n → Tm (suc n) → Tm n  -- Π[ π ] A B
  ƛ    : ∀ {n} → Tm (suc n) → Tm n
  app  : ∀ {n} → Tm n → Tm n → Tm n
  Σ′   : ∀ {n} → Mult → Tm n → Tm (suc n) → Tm n  -- Σ[ π ] A B
  pair : ∀ {n} → Tm n → Tm n → Tm n

------------------------------------------------------------------------
-- renaming and substitution (needed for context lookup and the app rule)
------------------------------------------------------------------------

extr : ∀ {m n} → (Fin m → Fin n) → Fin (suc m) → Fin (suc n)
extr ρ zero    = zero
extr ρ (suc x) = suc (ρ x)

ren : ∀ {m n} → (Fin m → Fin n) → Tm m → Tm n
ren ρ (var x)    = var (ρ x)
ren ρ ⋆          = ⋆
ren ρ (Π π A B)  = Π π (ren ρ A) (ren (extr ρ) B)
ren ρ (ƛ b)      = ƛ (ren (extr ρ) b)
ren ρ (app f a)  = app (ren ρ f) (ren ρ a)
ren ρ (Σ′ π A B) = Σ′ π (ren ρ A) (ren (extr ρ) B)
ren ρ (pair a b) = pair (ren ρ a) (ren ρ b)

exts : ∀ {m n} → (Fin m → Tm n) → Fin (suc m) → Tm (suc n)
exts σ zero    = var zero
exts σ (suc x) = ren suc (σ x)

sub : ∀ {m n} → (Fin m → Tm n) → Tm m → Tm n
sub σ (var x)    = σ x
sub σ ⋆          = ⋆
sub σ (Π π A B)  = Π π (sub σ A) (sub (exts σ) B)
sub σ (ƛ b)      = ƛ (sub (exts σ) b)
sub σ (app f a)  = app (sub σ f) (sub σ a)
sub σ (Σ′ π A B) = Σ′ π (sub σ A) (sub (exts σ) B)
sub σ (pair a b) = pair (sub σ a) (sub σ b)

sub-head : ∀ {n} → Tm n → Fin (suc n) → Tm n
sub-head s zero    = s
sub-head s (suc x) = var x

_[_] : ∀ {n} → Tm (suc n) → Tm n → Tm n
t [ s ] = sub (sub-head s) t

------------------------------------------------------------------------
-- dependent contexts (telescopes) and variable lookup (with weakening)
------------------------------------------------------------------------

data Con : Nat → Set where
  ε   : Con zero
  _◂_ : ∀ {n} → Con n → Tm n → Con (suc n)

infixl 5 _◂_

lkup : ∀ {n} → Con n → Fin n → Tm n
lkup (Γ ◂ A) zero    = ren suc A
lkup (Γ ◂ A) (suc i) = ren suc (lkup Γ i)

------------------------------------------------------------------------
-- definitional conversion: the congruence-closure of β (judgmental)
------------------------------------------------------------------------

data _≅_ {n} : Tm n → Tm n → Set where
  ≅-refl  : ∀ {t}                → t ≅ t
  ≅-sym   : ∀ {t u}   → t ≅ u    → u ≅ t
  ≅-trans : ∀ {t u v} → t ≅ u    → u ≅ v → t ≅ v
  ≅-β     : ∀ {b a}              → app (ƛ b) a ≅ (b [ a ])
  ≅-Π     : ∀ {π A A′ B B′} → A ≅ A′ → B ≅ B′ → Π π A B ≅ Π π A′ B′
  ≅-ƛ     : ∀ {b b′}  → b ≅ b′   → ƛ b ≅ ƛ b′
  ≅-app   : ∀ {f f′ a a′} → f ≅ f′ → a ≅ a′ → app f a ≅ app f′ a′
  ≅-Σ     : ∀ {π A A′ B B′} → A ≅ A′ → B ≅ B′ → Σ′ π A B ≅ Σ′ π A′ B′
  ≅-pair  : ∀ {a a′ b b′} → a ≅ a′ → b ≅ b′ → pair a b ≅ pair a′ b′

infix 4 _≅_

------------------------------------------------------------------------
-- the dependent QUANTITATIVE typing judgment   Γ ⊢[ γ ] t ⦂ A
------------------------------------------------------------------------

infix 3 _⊢[_]_⦂_

data _⊢[_]_⦂_ : ∀ {n} → Con n → Ctx n → Tm n → Tm n → Set where

  ⊢var : ∀ {n} {Γ : Con n} (i : Fin n)
       → Γ ⊢[ only i ] var i ⦂ lkup Γ i

  -- universe and type formation live in the 0-fragment (usage 𝟘)
  ⊢⋆ : ∀ {n} {Γ : Con n}
     → Γ ⊢[ 𝟘 ] ⋆ ⦂ ⋆

  -- type formation lives in the 0-fragment: a type's well-formedness does not
  -- track usage (the premises may use any usage γA, γB -- e.g. a dependent
  -- domain that mentions a variable), and the type itself has runtime usage 𝟘.
  ⊢Π : ∀ {n} {Γ : Con n} {γA : Ctx n} {γB : Ctx (suc n)} {π} {A B}
     → Γ ⊢[ γA ] A ⦂ ⋆
     → (Γ ◂ A) ⊢[ γB ] B ⦂ ⋆
     → Γ ⊢[ 𝟘 ] Π π A B ⦂ ⋆

  ⊢Σ : ∀ {n} {Γ : Con n} {γA : Ctx n} {γB : Ctx (suc n)} {π} {A B}
     → Γ ⊢[ γA ] A ⦂ ⋆
     → (Γ ◂ A) ⊢[ γB ] B ⦂ ⋆
     → Γ ⊢[ 𝟘 ] Σ′ π A B ⦂ ⋆

  -- λ: the bound variable is used σ times at runtime, within its budget π
  ⊢ƛ : ∀ {n} {Γ : Con n} {γ : Ctx n} {π σ} {A B} {b}
     → (Γ ◂ A) ⊢[ σ ∷ γ ] b ⦂ B
     → σ ⊑ π
     → Γ ⊢[ γ ] ƛ b ⦂ Π π A B

  -- application: the codomain type B depends on the argument (B [ a ]); the
  -- argument's usage δ is scaled by the binder multiplicity π
  ⊢app : ∀ {n} {Γ : Con n} {γ δ : Ctx n} {π} {A B} {f a}
       → Γ ⊢[ γ ] f ⦂ Π π A B
       → Γ ⊢[ δ ] a ⦂ A
       → Γ ⊢[ γ +ᶜ (π ·ᶜ δ) ] app f a ⦂ (B [ a ])

  -- dependent pair: first component used π times (scaled); second at type B[a]
  ⊢pair : ∀ {n} {Γ : Con n} {γa δb : Ctx n} {π} {A B} {a b}
        → Γ ⊢[ γa ] a ⦂ A
        → Γ ⊢[ δb ] b ⦂ (B [ a ])
        → Γ ⊢[ (π ·ᶜ γa) +ᶜ δb ] pair a b ⦂ Σ′ π A B

  -- conversion: types are identified up to definitional equality
  ⊢conv : ∀ {n} {Γ : Con n} {γ} {t} {A B}
        → Γ ⊢[ γ ] t ⦂ A
        → A ≅ B
        → Γ ⊢[ γ ] t ⦂ B

------------------------------------------------------------------------
-- THE worked example: the dependent identity, with erasure + linearity
--   λA. λx. x  :  Π (A :^0 ⋆). Π (x :^1 A). A
------------------------------------------------------------------------

dep-id : ε ⊢[ [] ] ƛ (ƛ (var zero))
              ⦂ Π 0# ⋆ (Π 1# (var zero) (var (suc zero)))
dep-id = ⊢ƛ (⊢ƛ (⊢var zero) ⊑-refl) ⊑-refl

-- a DEPENDENT Σ is well-formed: Σ (A :^1 ⋆). A  (the second component's type IS
-- the first component) -- type formation in the 0-fragment lets A mention the
-- bound variable.
dep-sigma : ε ⊢[ 𝟘 ] Σ′ 1# ⋆ (var zero) ⦂ ⋆
dep-sigma = ⊢Σ ⊢⋆ (⊢var zero)
