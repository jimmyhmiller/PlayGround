------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 8: COMBINING the linear type system with the memory primitives.
--
-- One calculus that has BOTH a linear (quantitative) type system AND the memory
-- operations over a heap, plus the soundness theorem tying them together:
--
--     a closed, well-typed program runs to completion with NO error
--     (no double-free / use-after-free) and NO leak.
--
-- This is the resource-soundness link of Rosette E4 -- "the multiplicities
-- deliver the operational guarantee" -- as a machine-checked Agda theorem rather
-- than a bounded SMT check.  The language is the minimal one that exhibits the
-- interaction:  unit | new | use e | var | let | seq, with a linear resource
-- type `Cp` (a capability to one live heap cell).
--
-- This module: the calculus, the linear typing, the heap evaluator, and typed
-- examples.  Soundness is proved in Combined-Sound (built on this).
--
-- Check:  LC_ALL=C.UTF-8 agda Combined.agda
------------------------------------------------------------------------

module Combined where

open import Rig using (_≡_; refl; cong; sym; trans; Mult; 0#; 1#; ω; _⊑_; ⊑-refl)
open import Context using (Nat; zero; suc; Vec; []; _∷_; Ctx; 𝟘; _+ᶜ_)

------------------------------------------------------------------------
-- prelude: Bool, de Bruijn indices, the "use variable i once" context
------------------------------------------------------------------------

data Bool : Set where true false : Bool

data Fin : Nat → Set where
  zero : ∀ {n} → Fin (suc n)
  suc  : ∀ {n} → Fin n → Fin (suc n)

only : ∀ {n} → Fin n → Ctx n
only zero    = 1# ∷ 𝟘
only (suc i) = 0# ∷ only i

------------------------------------------------------------------------
-- types and terms
------------------------------------------------------------------------

data Ty : Set where
  Un : Ty           -- unit
  Cp : Ty           -- a linear capability to a live heap cell

TyCtx : Nat → Set
TyCtx n = Vec Ty n

lookupT : ∀ {n} → TyCtx n → Fin n → Ty
lookupT (A ∷ _)  zero    = A
lookupT (_ ∷ Φ) (suc i) = lookupT Φ i

data Tm : Nat → Set where
  var : ∀ {n} → Fin n → Tm n
  tt  : ∀ {n} → Tm n               -- unit value
  new : ∀ {n} → Tm n               -- allocate a cell, return its capability
  use : ∀ {n} → Tm n → Tm n        -- consume (free) the capability
  lt  : ∀ {n} → Tm n → Tm (suc n) → Tm n   -- let x = e₁ in e₂  (x linear)
  sq  : ∀ {n} → Tm n → Tm n → Tm n         -- e₁ ; e₂   (e₁ : Un)

------------------------------------------------------------------------
-- linear typing:  Φ ⊢[ γ ] t ⦂ A    (γ = per-variable use counts)
-- every bound variable must be used exactly once (σ ⊑ 1# ⟹ σ ≡ 1#).
------------------------------------------------------------------------

infix 3 _⊢[_]_⦂_

data _⊢[_]_⦂_ : ∀ {n} → TyCtx n → Ctx n → Tm n → Ty → Set where

  ⊢var : ∀ {n} {Φ : TyCtx n} (i : Fin n)
       → Φ ⊢[ only i ] var i ⦂ lookupT Φ i

  ⊢tt  : ∀ {n} {Φ : TyCtx n} → Φ ⊢[ 𝟘 ] tt ⦂ Un

  ⊢new : ∀ {n} {Φ : TyCtx n} → Φ ⊢[ 𝟘 ] new ⦂ Cp

  ⊢use : ∀ {n} {Φ : TyCtx n} {γ} {e}
       → Φ ⊢[ γ ] e ⦂ Cp
       → Φ ⊢[ γ ] use e ⦂ Un

  ⊢lt  : ∀ {n} {Φ : TyCtx n} {γ δ} {σ} {A B} {e₁ e₂}
       → Φ ⊢[ γ ] e₁ ⦂ A
       → (A ∷ Φ) ⊢[ σ ∷ δ ] e₂ ⦂ B            -- bound var uses σ; outer ctx uses δ
       → σ ⊑ 1#                                   -- the bound variable is linear
       → Φ ⊢[ γ +ᶜ δ ] lt e₁ e₂ ⦂ B

  ⊢sq  : ∀ {n} {Φ : TyCtx n} {γ δ} {A} {e₁ e₂}
       → Φ ⊢[ γ ] e₁ ⦂ Un
       → Φ ⊢[ δ ] e₂ ⦂ A
       → Φ ⊢[ γ +ᶜ δ ] sq e₁ e₂ ⦂ A

------------------------------------------------------------------------
-- values, heap, and the big-step evaluator (structural, hence total)
--   Heap: a fresh-counter `nxt` and a liveness map `liv`.
--   eval returns (value, heap, ok?), where ok? is false iff a use hit a
--   non-capability or a dead/freed cell (a double-free / use-after-free).
------------------------------------------------------------------------

data Val : Set where
  vunit : Val
  vcap  : Nat → Val          -- a capability to heap cell ℓ

Env : Nat → Set
Env n = Vec Val n

lookupV : ∀ {n} → Env n → Fin n → Val
lookupV (v ∷ _)  zero    = v
lookupV (_ ∷ ρ) (suc i) = lookupV ρ i

eqn : Nat → Nat → Bool
eqn zero    zero    = true
eqn zero    (suc _) = false
eqn (suc _) zero    = false
eqn (suc m) (suc n) = eqn m n

record Heap : Set where
  constructor mkH
  field nxt : Nat ; liv : Nat → Bool
open Heap

upd : (Nat → Bool) → Nat → Bool → (Nat → Bool)
upd f k v = λ j → ifb (eqn k j) v (f j)
  where ifb : Bool → Bool → Bool → Bool
        ifb true  x _ = x
        ifb false _ y = y

and : Bool → Bool → Bool
and true  b = b
and false _ = false

-- result of evaluation
record Res : Set where
  constructor mkR
  field rval : Val ; rheap : Heap ; rok : Bool
open Res

-- free a capability value: ok only if it is a cap to a currently-live cell
freeCap : Val → Heap → Bool → Res
freeCap (vcap ℓ) h ok = mkR vunit (mkH (nxt h) (upd (liv h) ℓ false)) (and ok (liv h ℓ))
freeCap vunit    h ok = mkR vunit h false        -- using a non-capability: error

eval : ∀ {n} → Env n → Tm n → Heap → Bool → Res
eval ρ (var i) h ok = mkR (lookupV ρ i) h ok
eval ρ tt      h ok = mkR vunit h ok
eval ρ new     h ok = mkR (vcap (nxt h)) (mkH (suc (nxt h)) (upd (liv h) (nxt h) true)) ok
eval ρ (use e) h ok with eval ρ e h ok
... | mkR v h₁ ok₁ = freeCap v h₁ ok₁
eval ρ (lt e₁ e₂) h ok with eval ρ e₁ h ok
... | mkR v₁ h₁ ok₁ = eval (v₁ ∷ ρ) e₂ h₁ ok₁
eval ρ (sq e₁ e₂) h ok with eval ρ e₁ h ok
... | mkR _ h₁ ok₁ = eval ρ e₂ h₁ ok₁

------------------------------------------------------------------------
-- typed examples
------------------------------------------------------------------------

-- let x = new in use x   :   Un     (allocate then free -- accepted)
ex-good : [] ⊢[ [] ] lt new (use (var zero)) ⦂ Un
ex-good = ⊢lt ⊢new (⊢use (⊢var zero)) ⊑-refl
