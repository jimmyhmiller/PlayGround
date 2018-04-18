import Data.Vect

append : Vect n Int -> Vect m Int -> Vect (n + m) Int
append [] ys = ys
append (x :: xs) ys = x :: append xs ys
