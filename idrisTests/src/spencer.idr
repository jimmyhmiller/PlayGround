identity : a -> a
identity x = x



twoPlusTwo : 2+2 = 4
twoPlusTwo = Refl



data Vect : Nat -> Type -> Type where
  Nil : Vect 0 a
  (::) : a -> Vect n a -> Vect (S n) a



append : Vect n a -> Vect m a -> Vect (n + m) a
append [] y = y
append (x :: z) y = x :: append z y

z : Vect 1 Int
z = [1]
