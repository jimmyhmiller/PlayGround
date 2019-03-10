
data Vect : Nat -> Type -> Type where
  Nil : Vect Z a
  (::) : a -> Vect n a -> Vect (S n) a

%name Vect xs,ys,zs,ws

myVector : Vect 2 Int
myVector = [1, 2]

bar : (f : a -> b) -> (xs : Vect n a) -> Vect n b
bar f [] = []
bar f (x :: xs) = f x :: bar f xs

append : Vect n a -> Vect m a -> Vect (n + m) a
append [] ys = ys
append (x :: xs) ys = x :: append xs ys

zipwith : (f : a -> b -> c) -> Vect n a -> Vect n b -> Vect n c
zipwith f [] ys = []
zipwith f (x :: xs) (y :: ys) = f x y :: zipwith f xs ys

map : (f : (1 x : a) -> b) -> (1 xs: List a) -> List b
map f [] = []
map f (x :: xs) = f x :: map f xs
