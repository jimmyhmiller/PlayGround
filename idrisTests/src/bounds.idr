import Data.Vect

data Bounded : (n : Nat) -> Type where
     Bounds : (k : Nat) -> Bounded (plus (S k) n)


x : Bounded 4
x = Bounds 0

y : Bounded 4
y = Bounds 3


range : Bounded n -> (Nat, Nat)
range (Bounds k) {n = (S (plus k j))} = (k, (S (plus k j)))

allBounds : Bounded n -> List Nat
allBounds x = case range x of
                   (a, b) => ?allBounds_rhs_1


lookup : Bounded n -> Vect n a -> a
lookup (Bounds Z)     (x :: xs) = x
lookup (Bounds (S k)) (x :: xs) = lookup (Bounds k) xs
