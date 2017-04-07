import Data.ZZ

import Pruviloj
import Pruviloj.Induction

%hide (<+>)




%language ElabReflection


auto : Elab ()
auto =
  do compute
     attack
     try intros
     hs <- map fst <$> getEnv
     for_ hs $
       \ih => try (rewriteWith (Var ih))
     hypothesis <|> search
     solve




partial
mush : Elab ()
mush =
    do attack
       n <- gensym "j"
       intro n
       try intros
       ignore $ induction (Var n) `andThen` auto
       solve



interface Group a where
  (<+>) : a -> a -> a
  identity : a
  inverse : a -> a
  

interface Group a => VerifiedGroup a where
  prove_identity : (x : a) -> (x <+> Main.identity) = x
  prove_identity_r : (x : a) -> (Main.identity <+> x) = x
  prove_inverse : (x : a) -> (x <+> Main.inverse x) = Main.identity
  prove_inverse_r : (x : a) -> (inverse x <+> x) = Main.identity
  prove_assoc : {x : a} -> {y : a} -> {z : a} -> x <+> (y <+> z) = x <+> y <+> z





identity_proof_r : (x : ZZ) -> plusZ (Pos 0) x = x
identity_proof_r (Pos k) = Refl
identity_proof_r (NegS k) = Refl


stuff : (k: Nat) -> {auto prf: LTE k k} ->  k - k = 0
stuff Z = Refl
stuff (S k) {prf = (LTESucc x)} = stuff k
  
otherStuff : (k: Nat) -> {auto prf: LTE k k} -> minusNatZ k k = Pos (k - k)
otherStuff Z = Refl
otherStuff (S k) {prf = (LTESucc x)} = otherStuff k


inverse_proof : (x : ZZ) -> x + (negate x) = Pos 0
inverse_proof (Pos Z) = Refl
inverse_proof (Pos (S k)) with (lteRefl {n=k})
  inverse_proof (Pos (S Z)) | LTEZero = Refl
  inverse_proof (Pos (S (S right))) | (LTESucc x) = rewrite otherStuff right in rewrite stuff right in Refl
inverse_proof (NegS Z) = Refl
inverse_proof (NegS (S k)) with (lteRefl {n=k})
  inverse_proof (NegS (S Z)) | LTEZero = Refl
  inverse_proof (NegS (S (S right))) | (LTESucc x) = rewrite otherStuff right in rewrite stuff right in Refl


inverse_proof_r : (x : ZZ) -> plusZ (negate x) x = Pos 0
inverse_proof_r x = rewrite plusCommutativeZ (negate x) x in inverse_proof x

test_rhs1 : (k : Nat) -> (j : Nat) -> (i : Nat) -> (\replaced => replaced = S (plus (plus k j) i)) (S (plus k (plus j i)))
test_rhs1 k j i = cong {f=S} (plusAssociative k j i)





test : (k : Nat) -> (j : Nat) -> (i : Nat) -> plus k (S (plus j i)) = S (plus (plus k j) i)
test k j i = rewrite (sym (plusSuccRightSucc k (plus j i))) in (test_rhs1 k j i)




identity_proof : (x : ZZ) -> plusZ x (Pos 0) = x
identity_proof (Pos Z) = Refl
identity_proof (Pos (S k)) = rewrite plusZeroRightNeutral k in Refl
identity_proof (NegS Z) = Refl
identity_proof (NegS (S k)) = Refl

identity_proof' : (x : ZZ) -> plusZ (Pos 0) x = x
identity_proof' (Pos k) = Refl
identity_proof' (NegS k) = Refl

total
assoc_proof_neg : (k, j, i : Nat) -> plusZ (Pos k) (minusNatZ j (S i)) = minusNatZ (plus k j) (S i)
assoc_proof_neg Z Z Z = Refl
assoc_proof_neg Z Z k = Refl
assoc_proof_neg Z (S k) i = rewrite identity_proof' (minusNatZ k i) in Refl
assoc_proof_neg k j i = ?test1


total
assoc_proof_rhs_1 : (k, j, i : Nat) -> plusZ (Pos k) (minusNatZ i (S j)) = plusZ (minusNatZ k (S j)) (Pos i)
assoc_proof_rhs_1 Z Z Z = Refl
assoc_proof_rhs_1 Z Z k = rewrite identity_proof' (minusNatZ k 1) in Refl
assoc_proof_rhs_1 Z k i = rewrite identity_proof' (minusNatZ i (S k)) in Refl
assoc_proof_rhs_1 k j i = ?test2

assoc_proof_rhs_4 : (k, j, i : Nat) -> minusNatZ k (S (S (plus j i))) = plusZ (minusNatZ k (S j)) (NegS i)


assoc_proof : (x, y, z : ZZ) -> plusZ x (plusZ y z) = plusZ (plusZ x y) z
assoc_proof (Pos k) (Pos j) (Pos i) = cong {f=Pos} (plusAssociative k j i)
assoc_proof (Pos k) (Pos j) (NegS i) = assoc_proof_neg k j i
assoc_proof (Pos k) (NegS j) (Pos i) = assoc_proof_rhs_1 k j i
assoc_proof (Pos k) (NegS j) (NegS i) = assoc_proof_rhs_4 k j i
assoc_proof x y z = ?assoc_proof_rhs_2



Group ZZ where
    (<+>) x y = x + y
    identity = Pos Z
    inverse x = negate x

VerifiedGroup ZZ where
    prove_identity = identity_proof
    prove_identity_r = identity_proof_r
    prove_inverse = inverse_proof
    prove_inverse_r = inverse_proof_r
    prove_assoc {x} {y} {z} = assoc_proof x y z



inverse_cancel : VerifiedGroup a => {x : a} -> {y : a} -> Main.inverse x <+> x <+> y = y
inverse_cancel {x} {y} = rewrite (prove_inverse_r x) in rewrite (prove_identity_r y) in Refl


unique_identity : VerifiedGroup g => (e : g) -> ((a : g) -> a <+> e = a) -> e = Main.identity
unique_identity e prf = trans (sym (prove_identity_r e)) (prf identity)


unique_inverse : VerifiedGroup g => (inv : g) -> (x : g) -> (x <+> inv) = Main.identity -> inv = inverse x
unique_inverse inv x prf = t4 
  where
    t1 : x <+> inv = x <+> inverse x
    t1 = trans prf (sym (prove_inverse x))
    
    t2 : inverse x <+> (x <+> inv) = inverse x <+> (x <+> inverse x)
    t2 = cong t1
    
    t3 : inverse x <+> x <+> inv = inverse x <+> x <+> inverse x
    t3 = trans (sym prove_assoc) (trans t2 prove_assoc)
    
    t4 : inv = inverse x
    t4 = trans (sym inverse_cancel) (trans t3 inverse_cancel)




cancel : VerifiedGroup g => (a : g) -> (b : g) -> (c : g) -> a <+> b = a <+> c -> b = c
cancel a b c prf = t3
  where
    t1 : inverse a <+> (a <+> b) = inverse a <+> (a <+> c)
    t1 = cong prf
    
    t2 : inverse a <+> a <+> b = inverse a <+> a <+> c
    t2 = trans (sym prove_assoc) (trans t1 prove_assoc)
    
    t3 : b = c
    t3 = trans (sym inverse_cancel) (trans t2 inverse_cancel)
    



    

-- Local Variables:
-- idris-load-packages: ("pruviloj")
-- End:
