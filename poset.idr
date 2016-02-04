
%hide Nat.LTE


--data Nat = Z | S Nat





data LTE : (n, m : Nat) -> Type where
  LteZero : LTE Z m
  LteSucc : LTE n m -> LTE (S n) (S m)
  
  
succEqual : (n, m : Nat) -> n = m -> (S n) = (S m)
succEqual n n Refl = Refl


lteTransitive : (n, m, k : Nat) -> LTE n m -> LTE m k -> LTE n k
lteTransitive Z m k (LteZero) _  = LteZero
lteTransitive (S n) (S m) (S k) (LteSucc l1) (LteSucc l2) = LteSucc (lteTransitive n m k l1 l2)

lteReflexive : (n : Nat) -> LTE n n
lteReflexive Z = LteZero
lteReflexive (S n) = LteSucc (lteReflexive n)

lteAntiSymetric : (n, m : Nat) -> LTE n m -> LTE m n -> n = m
lteAntiSymetric Z Z LteZero LteZero = Refl
lteAntiSymetric (S n) (S m) (LteSucc l1) (LteSucc l2) = succEqual n m (lteAntiSymetric n m l1 l2)


main : IO ()
main = putStr "hello world"



