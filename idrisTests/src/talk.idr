import Data.Vect


append : Vect n a -> Vect m a -> Vect (n + m) a 
append [] ys = ys
append (x :: xs) ys = x :: append xs ys
