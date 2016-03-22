import Data.Vect
%hide cmp

repeat : (n : Nat) -> a -> Vect n a
repeat Z x = []
repeat (S k) x = x :: repeat k x


take : (n : Nat) -> Vect m a -> {auto ok: LTE n m} -> Vect n a
take Z xs {ok = LTEZero} = []
take (S left) (x :: xs) {ok = (LTESucc y)} = x :: take left xs


drop : (n : Nat) -> Vect m a -> {auto ok: LTE n m} -> Vect (m - n) a
drop Z [] {ok = LTEZero} = []
drop Z (x :: xs) {ok = LTEZero} = (x :: xs)
drop (S left) (x :: xs) {ok = (LTESucc y)} = drop left xs



data Cmp : Nat -> Nat -> Type where
  CmpLT : (y : _) -> Cmp x (x + S y)
  CmpEQ : Cmp x x
  CmpGT : (x : _) -> Cmp (y + S x) y


cmp : (n : Nat) -> (m : Nat) -> Cmp n m
cmp Z Z = CmpEQ
cmp Z (S k) = CmpLT k
cmp (S k) Z = CmpGT k
cmp (S k) (S j) with (cmp k j)
  cmp (S k) (S (plus k (S y))) | (CmpLT y) = CmpLT y
  cmp (S k) (S k) | CmpEQ = CmpEQ
  cmp (S (plus j (S x))) (S j) | (CmpGT x) = CmpGT x



test : (f : a -> b) -> x = y -> f x = f y
test f Refl = Refl

plusk0 : (k : Nat) -> 0 + k = k
plusk0 k = Refl



plus_nSm : (n : Nat) -> (m : Nat) -> n + S m = S (n + m)
plus_nSm Z Z = Refl
plus_nSm Z (S k) = Refl
plus_nSm (S k) Z = test S ?plus_nSm_rhs_1
plus_nSm (S k) (S j) = ?plus_nSm_rhs_3
