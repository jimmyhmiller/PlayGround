




data Vect : Nat -> Type -> Type where
  Nil : Vect Z a
  (::) : a -> Vect n a -> Vect (S n) a


append : Vect n a -> Vect m a -> Vect (n+m) a
append [] y = y
append (x :: z) y = x :: append z y

q : Vect 1 Int
q = [1]

q' : Vect 2 Int
q' = [1,2]


total
fact : Nat -> Nat
fact Z = 1
fact n@(S m) = n * Main.fact(m)




fact5 : Main.fact 5 = 120
fact5 = Refl
