import Data.List
import Data.Vect

total
factorial : Nat -> Nat
factorial Z = 1
factorial (S k) = (S k) * factorial(k)



fact5 : factorial 5 = 120
fact5 = Refl


