module AVL


-------------------------------------------
-- AVL trees



data Balance : Nat -> Nat -> Type where
  LeftLeaning : Balance (S n) n
  RightLeaning : Balance n (S n)
  Balanced : Balance n n

%name Balance b, bal


total
height : Balance n m -> Nat
height (LeftLeaning {n}) = S (S n)
height (RightLeaning {n}) = S (S n)
height {n} (Balanced {n}) = S n


data Tree : Nat -> Type -> Type -> Type where
  Empty : Tree 0 k v
  Node : (l : Tree n k v) ->
         (key : k) -> (val : v) ->
         (r : Tree m k v) ->
         (b : Balance n m) -> Tree (height b) k v

%name Tree t, tree




-- Bad tree:
-- Node Empty 0 "0" (Node Empty 1 "1" (Node Empty 2 "2" Empty RightLeaning) RightLeaning) RightLeaning

lookup : (Ord k) => k -> Tree n k v -> Maybe v
lookup x Empty = Nothing
lookup x (Node l key val r b) =
  case compare x key of
    LT => lookup x l
    EQ => Just val
    GT => lookup x r






data InsertRes : Nat -> Type -> Type -> Type where
  Flat : Tree n k v -> InsertRes n k v
  Taller : Tree (S n) k v -> InsertRes n k v

%name InsertRes res, r


rotLeft : Tree n k v -> k -> v -> Tree (S (S n)) k v -> InsertRes (S (S n)) k v

-- Impossible because Empty has depth 0 and we know the depth is at least 2 from the type
rotLeft l key val Empty impossible

rotLeft {n=m} l key val (Node rl key' val' rr Balanced) = ?test
  --Taller $ Node (Node l key val rl RightLeaning) key' val' rr LeftLeaning

-- Impossible because a left-leaning tree with a leaning left branch is too far out of balance
rotLeft  l key val (Node (Node rll key'' val'' rlr LeftLeaning) key' val' rr LeftLeaning) = ?test2
--rotLeft {n=m} l key val (Node (Node rll key'' val'' rlr RightLeaning) key' val' rr LeftLeaning) impossible

rotLeft {n=m} l key val (Node (Node rll key'' val'' rlr  Balanced) key' val' rr LeftLeaning) =
  Flat $ Node (Node l key val rll Balanced) key'' val'' (Node rlr key' val' rlr Balanced) Balanced

rotLeft l key val (Node rl key' val' rr RightLeaning) =
  Flat $ Node  (Node  l key val rl Balanced) key' val' rr Balanced



rotRight : Tree (S (S n)) k v -> k -> v -> Tree n k v -> InsertRes (S (S n)) k v
rotRight Empty key val r impossible
rotRight (Node ll key' val' lr Balanced) key val r =
  Taller $ Node ll key' val' (Node lr key val r LeftLeaning) RightLeaning
rotRight (Node ll key' val' lr LeftLeaning) key val r =
  Flat $ Node ll key' val' (Node lr key val r Balanced) Balanced
--rotRight {n=m} (Node key' val' ll (Node key'' val'' lrl lrr LeftLeaning) RightLeaning) key val r impossible
--rotRight {n=m} (Node key' val' ll (Node key'' val'' lrl lrr RightLeaning) RightLeaning) key val r impossible
rotRight (Node ll key' val' (Node lrl key'' val'' lrr Balanced) RightLeaning) key val r =
  Flat $ Node (Node ll key' val' lrl Balanced) key'' val'' (Node lrr key val r Balanced) Balanced


insert : (Ord k) => k -> v -> (t : Tree n k v) -> InsertRes n k v
insert newKey newVal Empty = Taller (Node Empty newKey newVal Empty Balanced)
insert newKey newVal (Node l key val  r b) with (compare newKey key)
  insert newKey newVal (Node l key val r b) | EQ = Flat (Node l newKey newVal r b)
  insert newKey newVal (Node l key val r b) | LT with (insert newKey newVal l)
    insert newKey newVal (Node l key val r b)            | LT | (Flat l')   = Flat (Node l' key val r b)
    insert newKey newVal (Node l key val r LeftLeaning)  | LT | (Taller l') = rotRight l' key val r
    insert newKey newVal (Node l key val r Balanced)     | LT | (Taller l') = Taller (Node l' key val r LeftLeaning)
    insert newKey newVal (Node l key val r RightLeaning) | LT | (Taller l') = Flat (Node l' key val r Balanced)
  insert newKey newVal (Node l key val r b) | GT with (insert newKey newVal r)
    insert newKey newVal (Node l key val r b)            | GT | (Flat r')   = Flat (Node l key val r' b)
    insert newKey newVal (Node l key val r LeftLeaning)  | GT | (Taller r') = Flat (Node l key val r' Balanced)
    insert newKey newVal (Node l key val r Balanced)     | GT | (Taller r') = Taller (Node l key val r' RightLeaning)
    insert newKey newVal (Node l key val r RightLeaning) | GT | (Taller r') = rotLeft l key val r'


fromList : Ord k => List (k, v) -> (n : Nat ** Tree n k v)
fromList [] = (0 ** Empty)
fromList ((k, v) :: xs) with (insert k v (getProof (AVL.fromList xs)))
  fromList ((k, v) :: xs) | (Flat x) = (_ ** x)
  fromList ((k, v) :: xs) | (Taller x) = (_ ** x)


flatten : Tree n k v -> List (k, v)
flatten Empty = []
flatten (Node l key val r b) = flatten l ++ [(key, val)] ++ flatten r

