------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 9: SOUNDNESS of the combined system -- well-typed ==> memory-safe.
--
-- The headline theorem of "combine them":  a closed, well-typed program of the
-- combined linear+memory calculus runs to completion with NO error (no
-- double-free / use-after-free, the `ok` flag stays true) and leaves NO leak
-- (the final heap has no live cell).  This is the machine-checked, UNBOUNDED
-- counterpart of the Rosette E4 resource-soundness result -- the multiplicities
-- of the linear type system DELIVER the operational memory guarantee, proved by
-- induction with NO SMT solver in the trusted base.
--
-- The calculus here is the canonical memory idiom: var | tt | use e | nu e | sq,
-- where `nu e` allocates a fresh heap cell, binds a LINEAR capability to it (used
-- exactly once: 1#), and runs e; `use` consumes (frees) a capability; `sq` is
-- sequencing.  The proof is a separation/frame argument keyed on the rig: the
-- typing's usage context tracks exactly which live cells a subterm OWNS, and
-- linearity (sigma = 1#, and `1# + 1# = ω ⋢ 1#`) forces owned sets to be disjoint
-- across a `sq` split -- which is what makes the frame preserved and every `use`
-- hit a live cell.
--
-- Check:  LC_ALL=C.UTF-8 agda CombinedSound.agda
------------------------------------------------------------------------

module CombinedSound where

open import Rig using (_≡_; refl; cong; sym; trans; Mult; 0#; 1#; ω; _+_)
open import Context using (Nat; zero; suc; Vec; []; _∷_; Ctx; 𝟘; _+ᶜ_)

------------------------------------------------------------------------
-- tiny prelude
------------------------------------------------------------------------

data Bool : Set where true false : Bool

record ⊤ : Set where constructor unit
data ⊥ : Set where
⊥-elim : ∀ {A : Set} → ⊥ → A
⊥-elim ()

record Σ (A : Set) (B : A → Set) : Set where
  constructor _,_
  field fst : A
        snd : B fst
open Σ
infixr 2 _,_

_×_ : Set → Set → Set
A × B = Σ A (λ _ → B)

data _⊎_ (A B : Set) : Set where
  inl : A → A ⊎ B
  inr : B → A ⊎ B

subst : ∀ {A : Set} (P : A → Set) {x y} → x ≡ y → P x → P y
subst P refl p = p

cong₂ : ∀ {A B C : Set} (f : A → B → C) {x x′ y y′} → x ≡ x′ → y ≡ y′ → f x y ≡ f x′ y′
cong₂ f refl refl = refl

bcontra : false ≡ true → ⊥
bcontra ()

------------------------------------------------------------------------
-- Bool algebra
------------------------------------------------------------------------

or : Bool → Bool → Bool
or true  _ = true
or false b = b

and : Bool → Bool → Bool
and true  b = b
and false _ = false

cond : Bool → Bool → Bool → Bool
cond true  x _ = x
cond false _ y = y

or-true : ∀ a → or a true ≡ true
or-true true  = refl
or-true false = refl

or-false : ∀ a → or a false ≡ a
or-false true  = refl
or-false false = refl

or-true-split : ∀ a b → or a b ≡ true → (a ≡ true) ⊎ (b ≡ true)
or-true-split true  b e = inl refl
or-true-split false b e = inr e

or-false-split : ∀ a b → or a b ≡ false → (a ≡ false) × (b ≡ false)
or-false-split false b e = refl , e

or4 : ∀ a b c d → or (or a b) (or c d) ≡ or (or a c) (or b d)
or4 true  b           c d = refl
or4 false true        c d = sym (or-true c)
or4 false false true  d = refl
or4 false false false d = refl

------------------------------------------------------------------------
-- Nat equality test and order
------------------------------------------------------------------------

eqn : Nat → Nat → Bool
eqn zero    zero    = true
eqn zero    (suc _) = false
eqn (suc _) zero    = false
eqn (suc m) (suc n) = eqn m n

eqn-refl : ∀ k → eqn k k ≡ true
eqn-refl zero    = refl
eqn-refl (suc k) = eqn-refl k

eqn-≡ : ∀ k ℓ → eqn k ℓ ≡ true → k ≡ ℓ
eqn-≡ zero    zero    e = refl
eqn-≡ (suc k) (suc ℓ) e = cong suc (eqn-≡ k ℓ e)

eqn-≠ : ∀ k ℓ → (k ≡ ℓ → ⊥) → eqn k ℓ ≡ false
eqn-≠ zero    zero    ne = ⊥-elim (ne refl)
eqn-≠ zero    (suc ℓ) ne = refl
eqn-≠ (suc k) zero    ne = refl
eqn-≠ (suc k) (suc ℓ) ne = eqn-≠ k ℓ (λ e → ne (cong suc e))

data _≤_ : Nat → Nat → Set where
  z≤n : ∀ {n}            → zero  ≤ n
  s≤s : ∀ {m n} → m ≤ n → suc m ≤ suc n

_<_ : Nat → Nat → Set
m < n = suc m ≤ n

≤-refl : ∀ {n} → n ≤ n
≤-refl {zero}  = z≤n
≤-refl {suc n} = s≤s ≤-refl

≤-trans : ∀ {a b c} → a ≤ b → b ≤ c → a ≤ c
≤-trans z≤n       q         = z≤n
≤-trans (s≤s p)   (s≤s q)   = s≤s (≤-trans p q)

n≤sucn : ∀ n → n ≤ suc n
n≤sucn zero    = z≤n
n≤sucn (suc n) = s≤s (n≤sucn n)

<-irrefl : ∀ n → suc n ≤ n → ⊥
<-irrefl (suc n) (s≤s p) = <-irrefl n p

<→≢ : ∀ {m n} → m < n → m ≡ n → ⊥
<→≢ {m} p refl = <-irrefl m p

≤-split : ∀ {n m} → n ≤ m → (n ≡ m) ⊎ (suc n ≤ m)
≤-split {zero}  {zero}  z≤n     = inl refl
≤-split {zero}  {suc m} z≤n     = inr (s≤s z≤n)
≤-split {suc n} {suc m} (s≤s p) with ≤-split p
... | inl e = inl (cong suc e)
... | inr q = inr (s≤s q)

-- comparison: a < b  or  b ≤ a
cmp : ∀ a b → (suc a ≤ b) ⊎ (b ≤ a)
cmp a       zero    = inr z≤n
cmp zero    (suc b) = inl (s≤s z≤n)
cmp (suc a) (suc b) with cmp a b
... | inl p = inl (s≤s p)
... | inr q = inr (s≤s q)

------------------------------------------------------------------------
-- de Bruijn indices, types, terms
------------------------------------------------------------------------

data Fin : Nat → Set where
  zero : ∀ {n} → Fin (suc n)
  suc  : ∀ {n} → Fin n → Fin (suc n)

data Ty : Set where
  Un : Ty            -- unit
  Cp : Ty            -- a linear capability to a live heap cell

TyCtx : Nat → Set
TyCtx n = Vec Ty n

lookupT : ∀ {n} → TyCtx n → Fin n → Ty
lookupT (A ∷ _) zero    = A
lookupT (_ ∷ Φ) (suc i) = lookupT Φ i

lookupC : ∀ {n} → Ctx n → Fin n → Mult
lookupC (m ∷ _) zero    = m
lookupC (_ ∷ γ) (suc i) = lookupC γ i

only : ∀ {n} → Fin n → Ctx n
only zero    = 1# ∷ 𝟘
only (suc i) = 0# ∷ only i

data Tm : Nat → Set where
  var : ∀ {n} → Fin n → Tm n
  tt  : ∀ {n} → Tm n
  use : ∀ {n} → Tm n → Tm n
  nu  : ∀ {n} → Tm (suc n) → Tm n         -- allocate fresh, bind linear cap, run
  sq  : ∀ {n} → Tm n → Tm n → Tm n        -- e₁ ; e₂   (e₁ : Un)

------------------------------------------------------------------------
-- linear typing.  σ = 1# at the nu-binder (used exactly once), since the only
-- value of `m ⊑ 1#` is m ≡ 1#; and 1# + 1# = ω which is NOT ⊑ 1#.
------------------------------------------------------------------------

infix 3 _⊢[_]_⦂_

data _⊢[_]_⦂_ : ∀ {n} → TyCtx n → Ctx n → Tm n → Ty → Set where
  ⊢var : ∀ {n} {Φ : TyCtx n} (i : Fin n)
       → Φ ⊢[ only i ] var i ⦂ lookupT Φ i
  ⊢tt  : ∀ {n} {Φ : TyCtx n} → Φ ⊢[ 𝟘 ] tt ⦂ Un
  ⊢use : ∀ {n} {Φ : TyCtx n} {γ} {e}
       → Φ ⊢[ γ ] e ⦂ Cp → Φ ⊢[ γ ] use e ⦂ Un
  ⊢nu  : ∀ {n} {Φ : TyCtx n} {δ} {A} {e}
       → (Cp ∷ Φ) ⊢[ 1# ∷ δ ] e ⦂ A → Φ ⊢[ δ ] nu e ⦂ A
  ⊢sq  : ∀ {n} {Φ : TyCtx n} {γ δ} {A} {e₁ e₂}
       → Φ ⊢[ γ ] e₁ ⦂ Un → Φ ⊢[ δ ] e₂ ⦂ A → Φ ⊢[ γ +ᶜ δ ] sq e₁ e₂ ⦂ A

------------------------------------------------------------------------
-- values, heap, evaluator
------------------------------------------------------------------------

data Val : Set where
  vunit : Val
  vcap  : Nat → Val

vcap-inj : ∀ {a b} → vcap a ≡ vcap b → a ≡ b
vcap-inj refl = refl

Env : Nat → Set
Env n = Vec Val n

lookupV : ∀ {n} → Env n → Fin n → Val
lookupV (v ∷ _) zero    = v
lookupV (_ ∷ ρ) (suc i) = lookupV ρ i

record Heap : Set where
  constructor mkH
  field nxt : Nat ; liv : Nat → Bool
open Heap

upd : (Nat → Bool) → Nat → Bool → (Nat → Bool)
upd f k v = λ j → cond (eqn k j) v (f j)

upd-same : ∀ f k v → upd f k v k ≡ v
upd-same f k v rewrite eqn-refl k = refl

upd-other : ∀ f k v j → (k ≡ j → ⊥) → upd f k v j ≡ f j
upd-other f k v j ne rewrite eqn-≠ k j ne = refl

record Res : Set where
  constructor mkR
  field rval : Val ; rheap : Heap ; rok : Bool
open Res

freeCap : Val → Heap → Bool → Res
freeCap (vcap ℓ) h ok = mkR vunit (mkH (nxt h) (upd (liv h) ℓ false)) (and ok (liv h ℓ))
freeCap vunit    h ok = mkR vunit h false

eval : ∀ {n} → Env n → Tm n → Heap → Bool → Res
eval ρ (var i)    h ok = mkR (lookupV ρ i) h ok
eval ρ tt         h ok = mkR vunit h ok
eval ρ (use e)    h ok with eval ρ e h ok
... | mkR v h₁ ok₁ = freeCap v h₁ ok₁
eval ρ (nu e)     h ok = eval (vcap (nxt h) ∷ ρ) e (mkH (suc (nxt h)) (upd (liv h) (nxt h) true)) ok
eval ρ (sq e₁ e₂) h ok with eval ρ e₁ h ok
... | mkR _ h₁ ok₁ = eval ρ e₂ h₁ ok₁

-- the with-clauses make the sq/use reductions opaque; these helpers fire them.
eval-sq : ∀ {n} (ρ : Env n) (e₁ e₂ : Tm n) (h : Heap) (ok : Bool) {v₁ h₁ ok₁}
        → eval ρ e₁ h ok ≡ mkR v₁ h₁ ok₁
        → eval ρ (sq e₁ e₂) h ok ≡ eval ρ e₂ h₁ ok₁
eval-sq ρ e₁ e₂ h ok eq with eval ρ e₁ h ok | eq
... | mkR _ _ _ | refl = refl

eval-use : ∀ {n} (ρ : Env n) (e : Tm n) (h : Heap) (ok : Bool) {v₁ h₁ ok₁}
         → eval ρ e h ok ≡ mkR v₁ h₁ ok₁
         → eval ρ (use e) h ok ≡ freeCap v₁ h₁ ok₁
eval-use ρ e h ok eq with eval ρ e h ok | eq
... | mkR _ _ _ | refl = refl

init : Heap
init = mkH zero (λ _ → false)

------------------------------------------------------------------------
-- ownership: which live cell (if any) a (value, usage) slot owns
------------------------------------------------------------------------

slot : Val → Mult → Nat → Bool
slot (vcap k) 1# ℓ = eqn k ℓ
slot (vcap k) 0# ℓ = false
slot (vcap k) ω  ℓ = false
slot vunit    m  ℓ = false

slot-true : ∀ v m ℓ → slot v m ℓ ≡ true → (m ≡ 1#) × (v ≡ vcap ℓ)
slot-true (vcap k) 1# ℓ e = refl , cong vcap (eqn-≡ k ℓ e)
slot-true (vcap k) 0# ℓ ()
slot-true (vcap k) ω  ℓ ()
slot-true vunit    m  ℓ ()

ownedχ : ∀ {n} → Env n → Ctx n → Nat → Bool
ownedχ []      []      ℓ = false
ownedχ (v ∷ ρ) (m ∷ γ) ℓ = or (slot v m ℓ) (ownedχ ρ γ ℓ)

resOwn : Val → Nat → Bool
resOwn (vcap r) ℓ = eqn r ℓ
resOwn vunit    ℓ = false

------------------------------------------------------------------------
-- linearity of a usage context: every entry is 0# or 1# (never ω)
------------------------------------------------------------------------

data NoΩ : Mult → Set where
  is0 : NoΩ 0#
  is1 : NoΩ 1#

noΩ-ω : NoΩ ω → ⊥
noΩ-ω ()

data Lin : ∀ {n} → Ctx n → Set where
  []  : Lin []
  _∷_ : ∀ {n} {m} {γ : Ctx n} → NoΩ m → Lin γ → Lin (m ∷ γ)

noΩ-split : ∀ m m′ → NoΩ (m + m′) → NoΩ m × NoΩ m′
noΩ-split 0# 0# p = is0 , is0
noΩ-split 0# 1# p = is0 , is1
noΩ-split 1# 0# p = is1 , is0
noΩ-split 1# 1# ()
noΩ-split 0# ω  ()
noΩ-split 1# ω  ()
noΩ-split ω  m′ ()

lin-split : ∀ {n} {γ δ : Ctx n} → Lin (γ +ᶜ δ) → Lin γ × Lin δ
lin-split {γ = []}    {[]}    [] = [] , []
lin-split {γ = m ∷ γ} {m′ ∷ δ} (p ∷ lp) with noΩ-split m m′ p | lin-split lp
... | (nm , nm′) | (Lγ , Lδ) = (nm ∷ Lγ) , (nm′ ∷ Lδ)

lin-lookup : ∀ {n} {γ : Ctx n} → Lin γ → ∀ i → NoΩ (lookupC γ i)
lin-lookup (p ∷ lp) zero    = p
lin-lookup (p ∷ lp) (suc i) = lin-lookup lp i

lookup-+ᶜ : ∀ {n} (γ δ : Ctx n) (i : Fin n)
          → lookupC (γ +ᶜ δ) i ≡ lookupC γ i + lookupC δ i
lookup-+ᶜ (m ∷ γ) (m′ ∷ δ) zero    = refl
lookup-+ᶜ (m ∷ γ) (m′ ∷ δ) (suc i) = lookup-+ᶜ γ δ i

slot-+ : ∀ v m m′ ℓ → NoΩ (m + m′) → slot v (m + m′) ℓ ≡ or (slot v m ℓ) (slot v m′ ℓ)
slot-+ vunit    m  m′ ℓ p  = sym (or-false (slot vunit m ℓ))
slot-+ (vcap k) 0# 0# ℓ p  = refl
slot-+ (vcap k) 0# 1# ℓ p  = refl
slot-+ (vcap k) 1# 0# ℓ p  = sym (or-false (eqn k ℓ))
slot-+ (vcap k) 1# 1# ℓ ()
slot-+ (vcap k) 0# ω  ℓ ()
slot-+ (vcap k) 1# ω  ℓ ()
slot-+ (vcap k) ω  m′ ℓ ()

------------------------------------------------------------------------
-- ownedχ computation lemmas
------------------------------------------------------------------------

ownedχ-𝟘 : ∀ {n} (ρ : Env n) ℓ → ownedχ ρ 𝟘 ℓ ≡ false
ownedχ-𝟘 []           ℓ = refl
ownedχ-𝟘 (vunit  ∷ ρ) ℓ = ownedχ-𝟘 ρ ℓ
ownedχ-𝟘 (vcap k ∷ ρ) ℓ = ownedχ-𝟘 ρ ℓ

ownedχ-only : ∀ {n} (ρ : Env n) (i : Fin n) ℓ
            → ownedχ ρ (only i) ℓ ≡ slot (lookupV ρ i) 1# ℓ
ownedχ-only (v ∷ ρ) zero    ℓ = trans (cong (or (slot v 1# ℓ)) (ownedχ-𝟘 ρ ℓ))
                                       (or-false (slot v 1# ℓ))
ownedχ-only (vunit  ∷ ρ) (suc i) ℓ = ownedχ-only ρ i ℓ
ownedχ-only (vcap k ∷ ρ) (suc i) ℓ = ownedχ-only ρ i ℓ

ownedχ-+ : ∀ {n} {γ δ : Ctx n} (ρ : Env n) → Lin (γ +ᶜ δ)
         → ∀ ℓ → ownedχ ρ (γ +ᶜ δ) ℓ ≡ or (ownedχ ρ γ ℓ) (ownedχ ρ δ ℓ)
ownedχ-+ {γ = []}    {[]}     []      lin       ℓ = refl
ownedχ-+ {γ = m ∷ γ} {m′ ∷ δ} (v ∷ ρ) (p ∷ lp) ℓ =
  trans (cong₂ or (slot-+ v m m′ ℓ p) (ownedχ-+ ρ lp ℓ))
        (or4 (slot v m ℓ) (slot v m′ ℓ) (ownedχ ρ γ ℓ) (ownedχ ρ δ ℓ))

ownedχ→pos : ∀ {n} (ρ : Env n) (γ : Ctx n) ℓ → ownedχ ρ γ ℓ ≡ true
           → Σ (Fin n) λ i → (lookupC γ i ≡ 1#) × (lookupV ρ i ≡ vcap ℓ)
ownedχ→pos []      []      ℓ ()
ownedχ→pos (v ∷ ρ) (m ∷ γ) ℓ e with or-true-split (slot v m ℓ) (ownedχ ρ γ ℓ) e
... | inl se with slot-true v m ℓ se
...   | (m≡1 , v≡cap) = zero , (m≡1 , v≡cap)
ownedχ→pos (v ∷ ρ) (m ∷ γ) ℓ e | inr re with ownedχ→pos ρ γ ℓ re
...   | (i , (mc , vc)) = suc i , (mc , vc)

------------------------------------------------------------------------
-- invariant predicates
------------------------------------------------------------------------

VTy : Ty → Val → Set
VTy Un v = v ≡ vunit
VTy Cp v = Σ Nat λ r → v ≡ vcap r

VT : ∀ {n} → Env n → TyCtx n → Set
VT {n} ρ Φ = ∀ (i : Fin n) → VTy (lookupT Φ i) (lookupV ρ i)

CapBound : ∀ {n} → Env n → Nat → Set
CapBound {n} ρ N = ∀ (i : Fin n) ℓ → lookupV ρ i ≡ vcap ℓ → ℓ < N

FreshDead : Heap → Set
FreshDead h = ∀ ℓ → nxt h ≤ ℓ → liv h ℓ ≡ false

OwnLive : ∀ {n} → Env n → Ctx n → Heap → Set
OwnLive ρ γ h = ∀ ℓ → ownedχ ρ γ ℓ ≡ true → liv h ℓ ≡ true

RInj : ∀ {n} → Env n → Set
RInj {n} ρ = ∀ (i j : Fin n) ℓ → lookupV ρ i ≡ vcap ℓ → lookupV ρ j ≡ vcap ℓ → i ≡ j

ResBound : Val → Nat → Set
ResBound (vcap r) N = r < N
ResBound vunit    N = ⊤

ResOF : ∀ {n} → Env n → Ctx n → Val → Heap → Set
ResOF ρ γ (vcap r) h = (ownedχ ρ γ r ≡ true) ⊎ (nxt h ≤ r)
ResOF ρ γ vunit    h = ⊤

record Pre {n} (ρ : Env n) (γ : Ctx n) (Φ : TyCtx n) (h : Heap) : Set where
  field
    lin : Lin γ
    vt  : VT ρ Φ
    cb  : CapBound ρ (nxt h)
    fd  : FreshDead h
    ri  : RInj ρ
    ol  : OwnLive ρ γ h
open Pre

record Spec {n} (ρ : Env n) (γ : Ctx n) (A : Ty) (h : Heap) (v : Val) (hp : Heap) : Set where
  field
    grow    : nxt h ≤ nxt hp
    frame   : ∀ ℓ → ownedχ ρ γ ℓ ≡ false → ℓ < nxt h → liv hp ℓ ≡ liv h ℓ
    onOwned : ∀ ℓ → ownedχ ρ γ ℓ ≡ true  → liv hp ℓ ≡ resOwn v ℓ
    onFresh : ∀ ℓ → nxt h ≤ ℓ            → liv hp ℓ ≡ resOwn v ℓ
    vty     : VTy A v
    rbnd    : ResBound v (nxt hp)
    rof     : ResOF ρ γ v h
open Spec

------------------------------------------------------------------------
-- generic helper lemmas
------------------------------------------------------------------------

owned-<nxt : ∀ {n} (ρ : Env n) (γ : Ctx n) (h : Heap) → CapBound ρ (nxt h)
           → ∀ m → ownedχ ρ γ m ≡ true → m < nxt h
owned-<nxt ρ γ h cb m e with ownedχ→pos ρ γ m e
... | (i , (_ , vc)) = cb i m vc

owned-false-above : ∀ {n} (ρ : Env n) (γ : Ctx n) (h : Heap) → CapBound ρ (nxt h)
                  → ∀ m → nxt h ≤ m → ownedχ ρ γ m ≡ false
owned-false-above ρ γ h cb m le with ownedχ ρ γ m in eq
... | true  = ⊥-elim (<→≢ (≤-trans (owned-<nxt ρ γ h cb m eq) le) refl)
... | false = refl

Disj : ∀ {n} {γ δ : Ctx n} (ρ : Env n) → RInj ρ → Lin (γ +ᶜ δ)
     → ∀ m → ownedχ ρ γ m ≡ true → ownedχ ρ δ m ≡ true → ⊥
Disj {γ = γ} {δ} ρ ri lin m eγ eδ with ownedχ→pos ρ γ m eγ | ownedχ→pos ρ δ m eδ
... | (i , (gi , vi)) | (j , (dj , vj)) with ri i j m vi vj
...   | refl = noΩ-ω (subst NoΩ
                        (trans (lookup-+ᶜ γ δ i) (cong₂ _+_ gi dj))
                        (lin-lookup lin i))

disj-γ : ∀ {n} {γ δ : Ctx n} (ρ : Env n) → RInj ρ → Lin (γ +ᶜ δ)
       → ∀ m → ownedχ ρ δ m ≡ true → ownedχ ρ γ m ≡ false
disj-γ {γ = γ} ρ ri lin m eδ with ownedχ ρ γ m in eq
... | true  = ⊥-elim (Disj ρ ri lin m eq eδ)
... | false = refl

disj-δ : ∀ {n} {γ δ : Ctx n} (ρ : Env n) → RInj ρ → Lin (γ +ᶜ δ)
       → ∀ m → ownedχ ρ γ m ≡ true → ownedχ ρ δ m ≡ false
disj-δ {δ = δ} ρ ri lin m eγ with ownedχ ρ δ m in eq
... | true  = ⊥-elim (Disj ρ ri lin m eγ eq)
... | false = refl

mk-rbound : ∀ (v : Val) (N : Nat) → ((k : Nat) → v ≡ vcap k → k < N) → ResBound v N
mk-rbound (vcap k) N f = f k refl
mk-rbound vunit    N f = unit

mk-rof : ∀ {n} (ρ : Env n) (γ : Ctx n) (h : Heap) (v : Val)
       → ((k : Nat) → v ≡ vcap k → ownedχ ρ γ k ≡ true) → ResOF ρ γ v h
mk-rof ρ γ h (vcap k) f = inl (f k refl)
mk-rof ρ γ h vunit    f = unit

resOwn-fresh : ∀ (v : Val) (N m : Nat) → ((k : Nat) → v ≡ vcap k → k < N) → N ≤ m
             → resOwn v m ≡ false
resOwn-fresh (vcap k) N m f le = eqn-≠ k m (λ e → <→≢ (≤-trans (f k refl) le) e)
resOwn-fresh vunit    N m f le = refl

------------------------------------------------------------------------
-- the `nu` step: reconstruct the precondition for the body, and turn the
-- body's spec into the conclusion's spec.
------------------------------------------------------------------------

nu-pre : ∀ {n} {Φ : TyCtx n} {δ : Ctx n} (ρ : Env n) (h : Heap)
       → Pre ρ δ Φ h
       → Pre (vcap (nxt h) ∷ ρ) (1# ∷ δ) (Cp ∷ Φ)
             (mkH (suc (nxt h)) (upd (liv h) (nxt h) true))
nu-pre {δ = δ} ρ h pre = record
  { lin = is1 ∷ lin pre
  ; vt  = vt-e
  ; cb  = cb-e
  ; fd  = fd-e
  ; ri  = ri-e
  ; ol  = ol-e
  }
  where
    vt-e : VT (vcap (nxt h) ∷ ρ) (Cp ∷ _)
    vt-e zero    = nxt h , refl
    vt-e (suc i) = vt pre i

    cb-e : CapBound (vcap (nxt h) ∷ ρ) (suc (nxt h))
    cb-e zero    ℓ e = subst (λ z → z < suc (nxt h)) (vcap-inj e) ≤-refl
    cb-e (suc i) ℓ e = ≤-trans (cb pre i ℓ e) (n≤sucn (nxt h))

    fd-e : FreshDead (mkH (suc (nxt h)) (upd (liv h) (nxt h) true))
    fd-e ℓ sle = trans (upd-other (liv h) (nxt h) true ℓ (λ e → <→≢ sle e))
                       (fd pre ℓ (≤-trans (n≤sucn (nxt h)) sle))

    ri-e : RInj (vcap (nxt h) ∷ ρ)
    ri-e zero    zero    ℓ e1 e2 = refl
    ri-e zero    (suc j) ℓ e1 e2 =
      ⊥-elim (<→≢ (cb pre j (nxt h) (trans e2 (cong vcap (sym (vcap-inj e1))))) refl)
    ri-e (suc i) zero    ℓ e1 e2 =
      ⊥-elim (<→≢ (cb pre i (nxt h) (trans e1 (cong vcap (sym (vcap-inj e2))))) refl)
    ri-e (suc i) (suc j) ℓ e1 e2 = cong suc (ri pre i j ℓ e1 e2)

    ol-e : OwnLive (vcap (nxt h) ∷ ρ) (1# ∷ δ)
                   (mkH (suc (nxt h)) (upd (liv h) (nxt h) true))
    ol-e ℓ oe with or-true-split (slot (vcap (nxt h)) 1# ℓ) (ownedχ ρ δ ℓ) oe
    ... | inl ee = subst (λ z → upd (liv h) (nxt h) true z ≡ true)
                         (eqn-≡ (nxt h) ℓ ee) (upd-same (liv h) (nxt h) true)
    ... | inr od = trans (upd-other (liv h) (nxt h) true ℓ
                            (λ e → <→≢ (owned-<nxt ρ δ h (cb pre) ℓ od) (sym e)))
                         (ol pre ℓ od)

rof-nu : ∀ {n} {δ : Ctx n} (ρ : Env n) (h : Heap) (v : Val)
       → ResOF (vcap (nxt h) ∷ ρ) (1# ∷ δ) v (mkH (suc (nxt h)) (upd (liv h) (nxt h) true))
       → ResOF ρ δ v h
rof-nu {δ = δ} ρ h (vcap r) (inl o) with or-true-split (slot (vcap (nxt h)) 1# r) (ownedχ ρ δ r) o
... | inl er = inr (subst (λ z → nxt h ≤ z) (eqn-≡ (nxt h) r er) ≤-refl)
... | inr od = inl od
rof-nu ρ h (vcap r) (inr le) = inr (≤-trans (n≤sucn (nxt h)) le)
rof-nu ρ h vunit    _        = unit

nu-conc : ∀ {n} {Φ : TyCtx n} {δ : Ctx n} {A} {v h′} (ρ : Env n) (h : Heap)
        → Pre ρ δ Φ h
        → Spec (vcap (nxt h) ∷ ρ) (1# ∷ δ) A
               (mkH (suc (nxt h)) (upd (liv h) (nxt h) true)) v h′
        → Spec ρ δ A h v h′
nu-conc {δ = δ} {A} {v} {h′} ρ h pre sp = record
  { grow    = ≤-trans (n≤sucn (nxt h)) (grow sp)
  ; frame   = frame-c
  ; onOwned = onOwned-c
  ; onFresh = onFresh-c
  ; vty     = vty sp
  ; rbnd    = rbnd sp
  ; rof     = rof-nu ρ h v (rof sp)
  }
  where
    frame-c : ∀ ℓ → ownedχ ρ δ ℓ ≡ false → ℓ < nxt h → liv h′ ℓ ≡ liv h ℓ
    frame-c ℓ of lt =
      trans (frame sp ℓ
               (trans (cong (λ b → or b (ownedχ ρ δ ℓ))
                            (eqn-≠ (nxt h) ℓ (λ e → <→≢ lt (sym e)))) of)
               (≤-trans lt (n≤sucn (nxt h))))
            (upd-other (liv h) (nxt h) true ℓ (λ e → <→≢ lt (sym e)))

    onOwned-c : ∀ ℓ → ownedχ ρ δ ℓ ≡ true → liv h′ ℓ ≡ resOwn v ℓ
    onOwned-c ℓ od = onOwned sp ℓ
      (trans (cong (or (slot (vcap (nxt h)) 1# ℓ)) od) (or-true (slot (vcap (nxt h)) 1# ℓ)))

    onFresh-c : ∀ ℓ → nxt h ≤ ℓ → liv h′ ℓ ≡ resOwn v ℓ
    onFresh-c ℓ le with ≤-split le
    ... | inl e  = onOwned sp ℓ
                     (cong (λ b → or b (ownedχ ρ δ ℓ))
                           (trans (sym (cong (eqn (nxt h)) e)) (eqn-refl (nxt h))))
    ... | inr lt = onFresh sp ℓ lt

------------------------------------------------------------------------
-- the `sq` step.  This is where linearity does its work: owned(γ) and owned(δ)
-- are disjoint (Disj), so e₁'s effects and e₂'s frame don't collide.
------------------------------------------------------------------------

sq-pre-e1 : ∀ {n} {Φ : TyCtx n} (γ δ : Ctx n) (ρ : Env n) (h : Heap)
          → Pre ρ (γ +ᶜ δ) Φ h → Pre ρ γ Φ h
sq-pre-e1 γ δ ρ h pre = record
  { lin = fst (lin-split (lin pre))
  ; vt  = vt pre
  ; cb  = cb pre
  ; fd  = fd pre
  ; ri  = ri pre
  ; ol  = λ ℓ eγ → ol pre ℓ
            (trans (ownedχ-+ ρ (lin pre) ℓ) (cong (λ b → or b (ownedχ ρ δ ℓ)) eγ))
  }

sq-pre-e2 : ∀ {n} {Φ : TyCtx n} (γ δ : Ctx n) {h₁ : Heap} (ρ : Env n) (h : Heap)
          → Pre ρ (γ +ᶜ δ) Φ h → Spec ρ γ Un h vunit h₁ → Pre ρ δ Φ h₁
sq-pre-e2 γ δ {h₁} ρ h pre sp₁ = record
  { lin = snd (lin-split (lin pre))
  ; vt  = vt pre
  ; cb  = λ i ℓ e → ≤-trans (cb pre i ℓ e) (grow sp₁)
  ; fd  = λ ℓ le → onFresh sp₁ ℓ (≤-trans (grow sp₁) le)
  ; ri  = ri pre
  ; ol  = λ ℓ od → trans (frame sp₁ ℓ (disj-γ ρ (ri pre) (lin pre) ℓ od)
                                       (owned-<nxt ρ δ h (cb pre) ℓ od))
                         (ol pre ℓ (trans (ownedχ-+ ρ (lin pre) ℓ)
                            (trans (cong (or (ownedχ ρ γ ℓ)) od)
                                   (or-true (ownedχ ρ γ ℓ)))))
  }

rof-sq : ∀ {n} {γ δ : Ctx n} (ρ : Env n) (h h₁ : Heap)
       → Lin (γ +ᶜ δ) → nxt h ≤ nxt h₁ → (v : Val)
       → ResOF ρ δ v h₁ → ResOF ρ (γ +ᶜ δ) v h
rof-sq {γ = γ} {δ} ρ h h₁ lin gr (vcap r) (inl δr) =
  inl (trans (ownedχ-+ ρ lin r) (trans (cong (or (ownedχ ρ γ r)) δr) (or-true (ownedχ ρ γ r))))
rof-sq ρ h h₁ lin gr (vcap r) (inr le) = inr (≤-trans gr le)
rof-sq ρ h h₁ lin gr vunit    _        = unit

sq-conc : ∀ {n} {Φ : TyCtx n} (γ δ : Ctx n) {A} {v₂ h₁ h₂} (ρ : Env n) (h : Heap)
        → Pre ρ (γ +ᶜ δ) Φ h
        → Spec ρ γ Un h vunit h₁ → Spec ρ δ A h₁ v₂ h₂
        → Spec ρ (γ +ᶜ δ) A h v₂ h₂
sq-conc {Φ = Φ} γ δ {A} {v₂} {h₁} {h₂} ρ h pre sp₁ sp₂ = record
  { grow    = ≤-trans (grow sp₁) (grow sp₂)
  ; frame   = frame-c
  ; onOwned = onOwned-c
  ; onFresh = onFresh-c
  ; vty     = vty sp₂
  ; rbnd    = rbnd sp₂
  ; rof     = rof-sq ρ h h₁ (lin pre) (grow sp₁) v₂ (rof sp₂)
  }
  where
    -- resOwn w ℓ ≡ false at any ℓ that is γ-owned (and so below nxt h)
    resOwn-false-at : ∀ (w : Val) → ResOF ρ δ w h₁ → ∀ ℓ
                    → ownedχ ρ γ ℓ ≡ true → ℓ < nxt h → resOwn w ℓ ≡ false
    resOwn-false-at (vcap r) (inl δr) ℓ γℓ lt =
      eqn-≠ r ℓ (λ e → Disj ρ (ri pre) (lin pre) ℓ γℓ
                          (subst (λ z → ownedχ ρ δ z ≡ true) e δr))
    resOwn-false-at (vcap r) (inr le) ℓ γℓ lt =
      eqn-≠ r ℓ (λ e → <→≢ (≤-trans lt (≤-trans (grow sp₁) le)) (sym e))
    resOwn-false-at vunit _ ℓ γℓ lt = refl

    -- resOwn w ℓ ≡ false at any freshly-allocated ℓ in [nxt h, nxt h₁)
    resOwn-false-fresh : ∀ (w : Val) → ResOF ρ δ w h₁ → ∀ ℓ
                       → nxt h ≤ ℓ → ℓ < nxt h₁ → resOwn w ℓ ≡ false
    resOwn-false-fresh (vcap r) (inl δr) ℓ ge lt =
      eqn-≠ r ℓ (λ e → <→≢ (≤-trans (owned-<nxt ρ δ h (cb pre) r δr) ge) e)
    resOwn-false-fresh (vcap r) (inr le) ℓ ge lt =
      eqn-≠ r ℓ (λ e → <→≢ (≤-trans lt le) (sym e))
    resOwn-false-fresh vunit _ ℓ ge lt = refl

    frame-c : ∀ ℓ → ownedχ ρ (γ +ᶜ δ) ℓ ≡ false → ℓ < nxt h → liv h₂ ℓ ≡ liv h ℓ
    frame-c ℓ of lt with or-false-split (ownedχ ρ γ ℓ) (ownedχ ρ δ ℓ)
                            (trans (sym (ownedχ-+ ρ (lin pre) ℓ)) of)
    ... | (oγ , oδ) = trans (frame sp₂ ℓ oδ (≤-trans lt (grow sp₁))) (frame sp₁ ℓ oγ lt)

    onOwned-c : ∀ ℓ → ownedχ ρ (γ +ᶜ δ) ℓ ≡ true → liv h₂ ℓ ≡ resOwn v₂ ℓ
    onOwned-c ℓ oe with or-true-split (ownedχ ρ γ ℓ) (ownedχ ρ δ ℓ)
                           (trans (sym (ownedχ-+ ρ (lin pre) ℓ)) oe)
    ... | inl oγ = trans (trans (frame sp₂ ℓ (disj-δ ρ (ri pre) (lin pre) ℓ oγ)
                                            (≤-trans (owned-<nxt ρ γ h (cb pre) ℓ oγ) (grow sp₁)))
                                (onOwned sp₁ ℓ oγ))
                         (sym (resOwn-false-at v₂ (rof sp₂) ℓ oγ
                                 (owned-<nxt ρ γ h (cb pre) ℓ oγ)))
    ... | inr oδ = onOwned sp₂ ℓ oδ

    onFresh-c : ∀ ℓ → nxt h ≤ ℓ → liv h₂ ℓ ≡ resOwn v₂ ℓ
    onFresh-c ℓ ge with cmp ℓ (nxt h₁)
    ... | inl lt  = trans (trans (frame sp₂ ℓ (owned-false-above ρ δ h (cb pre) ℓ ge) lt)
                                 (onFresh sp₁ ℓ ge))
                          (sym (resOwn-false-fresh v₂ (rof sp₂) ℓ ge lt))
    ... | inr ge₁ = onFresh sp₂ ℓ ge₁

------------------------------------------------------------------------
-- the soundness lemma (the induction): a well-typed term, run from a heap
-- whose precondition holds, stays `ok = true` and satisfies the frame spec.
------------------------------------------------------------------------

sound : ∀ {n} {Φ : TyCtx n} {γ} {t} {A} (ρ : Env n) (h : Heap)
      → Φ ⊢[ γ ] t ⦂ A → Pre ρ γ Φ h
      → Σ Val λ v → Σ Heap λ hp → (eval ρ t h true ≡ mkR v hp true) × Spec ρ γ A h v hp

sound ρ h (⊢var i) pre =
  lookupV ρ i , h , refl , record
    { grow    = ≤-refl
    ; frame   = λ ℓ _ _ → refl
    ; onOwned = onOwned-v
    ; onFresh = λ ℓ le → trans (fd pre ℓ le)
                 (sym (resOwn-fresh (lookupV ρ i) (nxt h) ℓ (λ k e → cb pre i k e) le))
    ; vty     = vt pre i
    ; rbnd    = mk-rbound (lookupV ρ i) (nxt h) (λ k e → cb pre i k e)
    ; rof     = mk-rof ρ (only i) h (lookupV ρ i)
                  (λ k e → trans (ownedχ-only ρ i k)
                                 (trans (cong (λ w → slot w 1# k) e) (eqn-refl k)))
    }
  where
    onOwned-v : ∀ ℓ → ownedχ ρ (only i) ℓ ≡ true → liv h ℓ ≡ resOwn (lookupV ρ i) ℓ
    onOwned-v ℓ oe with slot-true (lookupV ρ i) 1# ℓ (trans (sym (ownedχ-only ρ i ℓ)) oe)
    ... | (_ , veq) = trans (ol pre ℓ oe)
                            (sym (trans (cong (λ w → resOwn w ℓ) veq) (eqn-refl ℓ)))

sound ρ h ⊢tt pre =
  vunit , h , refl , record
    { grow    = ≤-refl
    ; frame   = λ ℓ _ _ → refl
    ; onOwned = λ ℓ oe → ⊥-elim (bcontra (trans (sym (ownedχ-𝟘 ρ ℓ)) oe))
    ; onFresh = λ ℓ le → fd pre ℓ le
    ; vty     = refl
    ; rbnd    = unit
    ; rof     = unit
    }

sound {γ = γ} ρ h (⊢use {e = e} d) pre with sound ρ h d pre
... | (v₁ , h₁ , ev₁ , sp₁) with vty sp₁
...   | (r , refl) =
        vunit
      , mkH (nxt h₁) (upd (liv h₁) r false)
      , trans (eval-use ρ e h true ev₁)
              (cong (mkR vunit (mkH (nxt h₁) (upd (liv h₁) r false))) livr)
      , record
          { grow    = grow sp₁
          ; frame   = frame-u
          ; onOwned = onOwned-u
          ; onFresh = onFresh-u
          ; vty     = refl
          ; rbnd    = unit
          ; rof     = unit
          }
  where
    livr : liv h₁ r ≡ true
    livr with rof sp₁
    ... | inl oe = trans (onOwned sp₁ r oe) (eqn-refl r)
    ... | inr le = trans (onFresh sp₁ r le) (eqn-refl r)

    frame-u : ∀ ℓ → ownedχ ρ γ ℓ ≡ false → ℓ < nxt h
            → liv (mkH (nxt h₁) (upd (liv h₁) r false)) ℓ ≡ liv h ℓ
    frame-u ℓ of lt with eqn r ℓ in eq
    ... | false = frame sp₁ ℓ of lt
    ... | true with eqn-≡ r ℓ eq | rof sp₁
    ...   | refl | inl oer = ⊥-elim (bcontra (trans (sym of) oer))
    ...   | refl | inr le  = ⊥-elim (<→≢ (≤-trans lt le) refl)

    onOwned-u : ∀ ℓ → ownedχ ρ γ ℓ ≡ true
              → liv (mkH (nxt h₁) (upd (liv h₁) r false)) ℓ ≡ resOwn vunit ℓ
    onOwned-u ℓ oe with eqn r ℓ in eq
    ... | true  = refl
    ... | false = trans (onOwned sp₁ ℓ oe) eq

    onFresh-u : ∀ ℓ → nxt h ≤ ℓ
              → liv (mkH (nxt h₁) (upd (liv h₁) r false)) ℓ ≡ resOwn vunit ℓ
    onFresh-u ℓ le with eqn r ℓ in eq
    ... | true  = refl
    ... | false = trans (onFresh sp₁ ℓ le) eq

sound ρ h (⊢nu d) pre
  with sound (vcap (nxt h) ∷ ρ) (mkH (suc (nxt h)) (upd (liv h) (nxt h) true)) d (nu-pre ρ h pre)
... | (v , h′ , ev , sp) = v , h′ , ev , nu-conc ρ h pre sp

sound ρ h (⊢sq {γ = γ} {δ = δ} {e₁ = e₁} {e₂ = e₂} d₁ d₂) pre with sound ρ h d₁ (sq-pre-e1 γ δ ρ h pre)
... | (v₁ , h₁ , ev₁ , sp₁) with vty sp₁
...   | refl with sound ρ h₁ d₂ (sq-pre-e2 γ δ ρ h pre sp₁)
...     | (v₂ , h₂ , ev₂ , sp₂) =
          v₂ , h₂ , trans (eval-sq ρ e₁ e₂ h true ev₁) ev₂ , sq-conc γ δ ρ h pre sp₁ sp₂

------------------------------------------------------------------------
-- CLOSED-PROGRAM SOUNDNESS:  well-typed ==> memory-safe.
--
-- A closed, well-typed program of type Un, run from the empty heap, terminates
-- with NO error (rok ≡ true: no double-free / use-after-free) and leaves NO
-- live cell in the heap (no leak).  Unbounded; no SMT.
------------------------------------------------------------------------

init-Pre : Pre [] [] [] init
init-Pre = record
  { lin = []
  ; vt  = λ ()
  ; cb  = λ ()
  ; fd  = λ ℓ _ → refl
  ; ri  = λ ()
  ; ol  = λ ℓ ()
  }

soundness : ∀ {t} → ([] ⊢[ [] ] t ⦂ Un)
          → Σ Heap λ h′ → (eval [] t init true ≡ mkR vunit h′ true)
                        × (∀ ℓ → liv h′ ℓ ≡ false)
soundness d with sound [] init d init-Pre
... | (v , h′ , ev , sp) with vty sp
...   | refl = h′ , ev , λ ℓ → onFresh sp ℓ z≤n

------------------------------------------------------------------------
-- worked examples (closed, well-typed; soundness applies to each)
------------------------------------------------------------------------

-- allocate a cell, then free it:   nu x. use x
ex1 : [] ⊢[ [] ] nu (use (var zero)) ⦂ Un
ex1 = ⊢nu (⊢use (⊢var zero))

-- allocate two cells, free both:   nu x. nu y. (use x ; use y)
ex2 : [] ⊢[ [] ] nu (nu (sq (use (var (suc zero))) (use (var zero)))) ⦂ Un
ex2 = ⊢nu (⊢nu (⊢sq (⊢use (⊢var (suc zero))) (⊢use (⊢var zero))))

-- soundness instantiated: each runs without error and leaks nothing
ex1-safe : Σ Heap λ h′ → (eval [] (nu (use (var zero))) init true ≡ mkR vunit h′ true)
                       × (∀ ℓ → liv h′ ℓ ≡ false)
ex1-safe = soundness ex1

ex2-safe : Σ Heap λ h′
         → (eval [] (nu (nu (sq (use (var (suc zero))) (use (var zero))))) init true
              ≡ mkR vunit h′ true)
         × (∀ ℓ → liv h′ ℓ ≡ false)
ex2-safe = soundness ex2
