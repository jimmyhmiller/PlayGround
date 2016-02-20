

nthPrime : Nat -> Nat
nthPrime k = ?nthPrime_rhs

isPrime : (n, m : Nat) -> LTE 2 m -> (m == nthPrime n) = False  -> ((mod (nthPrime n) m) == 0) = False
isPrime n (S right) (LTESucc x) prf = ?isPrime_rhs_1
