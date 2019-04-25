data Vect : Nat -> Type -> Type where
  Nil : Vect Z a
  (::) : a -> Vect n a -> Vect (S n) a

%name Vect xs,ys,zs,ws


-- Ctrl + Cmd + a = Add Clause
-- Ctrl + Cmd + c = Case Split
-- Ctrl + Cmd + s = Proof Search
-- Ctrl + Cmd + t = Show Type
-- Ctrl + Cmd + R = Type Check
-- Ctrl + Cmd + f = Generate Definition

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
