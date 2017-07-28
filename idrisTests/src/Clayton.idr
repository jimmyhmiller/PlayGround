import Data.List



indexOf : (coll : List Int) -> (n : Int) -> {auto prf: Elem n coll} -> Int
indexOf (n :: xs) n {prf = Here} = 0
indexOf (y :: xs) n {prf = (There later)} = 1 + indexOf xs n

inc : Nat -> Nat
inc n = S n

dec : (n : Nat) -> {auto prf: LTE (S Z) n} -> Nat
dec (S x) = x

add : Nat -> Nat -> Nat
add Zero m = m
add n Zero = n
add n (S m) = add (inc n) (dec (S m))

multi : Nat -> Nat -> Nat
multi Zero m = Zero
multi n Zero = Zero
multi (S Zero) m = m
multi n (S Zero) = n
multi n (S m) = multi (add n n) (dec (S m))

sub : (n : Nat) -> (m: Nat) -> {auto prf: LTE m n} -> Nat
sub Z Z {prf = LTEZero} = 0
sub n Z {prf = LTEZero} = n
sub (S right) (S left) {prf = (LTESucc x)} = sub right left
