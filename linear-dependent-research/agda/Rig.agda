------------------------------------------------------------------------
-- lambda-Tally, Agda development.
--
-- Module 1: the multiplicity rig R = {0, 1, omega}.
--
-- This is the algebraic foundation of the whole system (roadmap D0.2 in
-- docs/04): multiplicities form an ordered semiring, and the type system's
-- context arithmetic is exactly +/* in this rig, with the usage check being
-- the order _⊑_ ("computed usage is within the declared budget").
--
-- Where the Rosette work gave bounded/automated EVIDENCE, the Agda work gives
-- unbounded, machine-checked PROOF. This module proves the rig laws once and
-- for all, by exhaustive case analysis (the type has three elements, so most
-- cases collapse definitionally to `refl`).
--
-- Self-contained: a tiny inline prelude, so it type-checks without configuring
-- the standard library.  Check:  agda Rig.agda
------------------------------------------------------------------------

module Rig where

------------------------------------------------------------------------
-- Minimal prelude (propositional equality and negation)
------------------------------------------------------------------------

data _≡_ {A : Set} (x : A) : A → Set where
  refl : x ≡ x

infix 4 _≡_

{-# BUILTIN EQUALITY _≡_ #-}   -- enables `rewrite` in later modules

sym : ∀ {A : Set} {x y : A} → x ≡ y → y ≡ x
sym refl = refl

trans : ∀ {A : Set} {x y z : A} → x ≡ y → y ≡ z → x ≡ z
trans refl q = q

cong : ∀ {A B : Set} (f : A → B) {x y : A} → x ≡ y → f x ≡ f y
cong f refl = refl

cong₂ : ∀ {A B C : Set} (f : A → B → C) {x x′ : A} {y y′ : B}
      → x ≡ x′ → y ≡ y′ → f x y ≡ f x′ y′
cong₂ f refl refl = refl

data ⊥ : Set where

¬_ : Set → Set
¬ A = A → ⊥

infix 3 ¬_

------------------------------------------------------------------------
-- The carrier and operations
------------------------------------------------------------------------

-- 0# = erased / used zero times ; 1# = linear / used once ; ω = unrestricted
data Mult : Set where
  0# 1# ω : Mult

infixl 6 _+_
infixl 7 _*_

-- addition: how usages combine when a variable is used in two places
_+_ : Mult → Mult → Mult
0# + y  = y
1# + 0# = 1#
1# + 1# = ω
1# + ω  = ω
ω  + _  = ω

-- multiplication: how a usage scales when passed through a binder of a given
-- multiplicity (the App / Pi rule's `σ · π`)
_*_ : Mult → Mult → Mult
0# * _  = 0#
1# * y  = y
ω  * 0# = 0#
ω  * 1# = ω
ω  * ω  = ω

------------------------------------------------------------------------
-- (R, +, 0#) is a commutative monoid
------------------------------------------------------------------------

+-identityˡ : ∀ x → 0# + x ≡ x
+-identityˡ x = refl

+-identityʳ : ∀ x → x + 0# ≡ x
+-identityʳ 0# = refl
+-identityʳ 1# = refl
+-identityʳ ω  = refl

+-comm : ∀ x y → x + y ≡ y + x
+-comm 0# y  = sym (+-identityʳ y)
+-comm 1# 0# = refl
+-comm 1# 1# = refl
+-comm 1# ω  = refl
+-comm ω  0# = refl
+-comm ω  1# = refl
+-comm ω  ω  = refl

+-assoc : ∀ x y z → (x + y) + z ≡ x + (y + z)
+-assoc 0# y  z  = refl
+-assoc ω  y  z  = refl
+-assoc 1# 0# z  = refl
+-assoc 1# 1# 0# = refl
+-assoc 1# 1# 1# = refl
+-assoc 1# 1# ω  = refl
+-assoc 1# ω  z  = refl

------------------------------------------------------------------------
-- (R, *, 1#) is a commutative monoid, 0# annihilates
------------------------------------------------------------------------

*-identityˡ : ∀ x → 1# * x ≡ x
*-identityˡ x = refl

*-identityʳ : ∀ x → x * 1# ≡ x
*-identityʳ 0# = refl
*-identityʳ 1# = refl
*-identityʳ ω  = refl

*-zeroˡ : ∀ x → 0# * x ≡ 0#
*-zeroˡ x = refl

*-zeroʳ : ∀ x → x * 0# ≡ 0#
*-zeroʳ 0# = refl
*-zeroʳ 1# = refl
*-zeroʳ ω  = refl

*-comm : ∀ x y → x * y ≡ y * x
*-comm 0# y  = sym (*-zeroʳ y)
*-comm 1# y  = sym (*-identityʳ y)
*-comm ω  0# = refl
*-comm ω  1# = refl
*-comm ω  ω  = refl

*-assoc : ∀ x y z → (x * y) * z ≡ x * (y * z)
*-assoc 0# y  z  = refl
*-assoc 1# y  z  = refl
*-assoc ω  0# z  = refl
*-assoc ω  1# z  = refl
*-assoc ω  ω  0# = refl
*-assoc ω  ω  1# = refl
*-assoc ω  ω  ω  = refl

------------------------------------------------------------------------
-- Distributivity: (R, +, *) is a semiring
------------------------------------------------------------------------

*-distribˡ-+ : ∀ x y z → x * (y + z) ≡ (x * y) + (x * z)
*-distribˡ-+ 0# y  z  = refl
*-distribˡ-+ 1# y  z  = refl
*-distribˡ-+ ω  0# z  = refl
*-distribˡ-+ ω  1# 0# = refl
*-distribˡ-+ ω  1# 1# = refl
*-distribˡ-+ ω  1# ω  = refl
*-distribˡ-+ ω  ω  z  = refl

*-distribʳ-+ : ∀ x y z → (y + z) * x ≡ (y * x) + (z * x)
*-distribʳ-+ x y z =
  trans (*-comm (y + z) x)
        (trans (*-distribˡ-+ x y z)
               (cong₂ _+_ (*-comm x y) (*-comm x z)))

------------------------------------------------------------------------
-- The usage order  _⊑_  : "computed usage is within the declared budget".
--   x ⊑ x   and   x ⊑ ω    (so 0# and 1# are INCOMPARABLE).
-- Consequences: declared 1# admits only usage 1# (exactly-once / linear);
-- declared 0# admits only 0# (erased); declared ω admits anything.
------------------------------------------------------------------------

data _⊑_ : Mult → Mult → Set where
  ⊑-refl : ∀ {x}         → x ⊑ x
  ⊑-ω    : ∀ {x}         → x ⊑ ω

infix 4 _⊑_

⊑-trans : ∀ {x y z} → x ⊑ y → y ⊑ z → x ⊑ z
⊑-trans ⊑-refl q       = q
⊑-trans ⊑-ω    ⊑-refl  = ⊑-ω
⊑-trans ⊑-ω    ⊑-ω     = ⊑-ω

ω-top : ∀ x → x ⊑ ω
ω-top _ = ⊑-ω

-- linearity really is exactly-once: a linear budget rejects "used zero times"
-- and "used many times".
0⋢1 : ¬ (0# ⊑ 1#)
0⋢1 ()

ω⋢1 : ¬ (ω ⊑ 1#)
ω⋢1 ()

1⋢0 : ¬ (1# ⊑ 0#)
1⋢0 ()

-- The headline characterisations: a usage WITHIN a linear (1#) budget is
-- *exactly* 1#, and within an erased (0#) budget is *exactly* 0#.  This is the
-- formal content of "1 = used exactly once" and "0 = erased", proven once.
⊑1→≡1 : ∀ {x} → x ⊑ 1# → x ≡ 1#
⊑1→≡1 ⊑-refl = refl

⊑0→≡0 : ∀ {x} → x ⊑ 0# → x ≡ 0#
⊑0→≡0 ⊑-refl = refl
