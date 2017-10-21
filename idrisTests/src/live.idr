

data Vect : Nat -> Type -> Type where
  Nil : Vect Z a
  (::) : a -> Vect n a -> Vect (S n) a






-- append [1,2] [3, 4]
-- [1, 2, 3, 4] 

append : Vect n a -> Vect m a -> Vect (n + m) a
append [] y = y
append (x :: z) y = x :: append z y





























---------

factorial : Nat -> Nat
factorial Z = 1
factorial (S k) = (S k) * factorial k


factorial5 : factorial 5 = 120
factorial5 = Refl

factorial1 : factorial 1 = 1
factorial1 = Refl















---------

total
complex : Nat -> Nat
complex Z = Z
complex (S k) = case (S k) of
                  two => complex k + 2 + (complex k) + 1
                  five => (complex k + 3 * 3) * complex k
                  seven => complex k + 2
                  ten => 2 + complex k * (complex k + 2 * 7)
                  n => complex k * complex k + complex k

























two : Nat
two = cast 2

five : Nat
five = cast 5

seven : Nat
seven = cast 7

ten : Nat
ten = cast 10
