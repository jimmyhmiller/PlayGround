

total
factorial : Nat -> Nat
factorial Z = 1
factorial m@(S n) = m * factorial n


fact5 : factorial 5 = 120
fact5 = Refl
