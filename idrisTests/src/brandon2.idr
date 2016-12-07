

total
factorial : Nat -> Nat
factorial Z = 1
factorial n@(S m) = n * factorial m


fact5 : factorial 5 = 120
fact5 = Refl
