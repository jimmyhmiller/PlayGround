

unwind : Nat -> (r -> r) -> r -> r
unwind Z f x = x
unwind (S k) f x = unwind k f (f x)

dec : Nat -> Nat
dec Z = Z
dec (S k) = k

bound : Nat -> (Nat -> Nat) -> Nat -> Nat
bound k f j = let n = f j in if n < k then k else n

add : Nat -> Nat -> Nat
add n m = unwind n S m

sub : Nat -> Nat -> Nat
sub n m = unwind m dec n

multi : Nat -> Nat -> Nat
multi n m = unwind n (add m) Z


factorial : Nat -> Nat
factorial n = fst $ unwind n (\(r, n) => (mult r $ n , (S n))) (1, (S Z))

toInt : Nat -> Int
toInt n = unwind n (+1) 0
