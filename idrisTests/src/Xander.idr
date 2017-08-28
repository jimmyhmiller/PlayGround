

identity : a -> a
identity x = x







data Vect : Nat -> Type -> Type where
  Nil : Vect Z a
  (::) : a -> Vect n a -> Vect (S n) a
  
  
  
q : Vect 1 Int
q = [1]



head' : Vect (S n) a -> a
head' (x :: y) = x


append : Vect n a -> Vect m a -> Vect (n + m) a
append [] y = y
append (x :: z) [] = x :: append z []
append (x :: z) (y :: w) = x :: append z (y :: w)


total
factorial : Nat -> Nat
factorial Z = 1
factorial (S n) = (S n) * fact n





factorial5 : factorial 5 = 120
factorial5 = Refl
