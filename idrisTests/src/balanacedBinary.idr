data Balanced : Nat -> Nat -> Type where
  Perfect : Balanced n n
  LeftBias : Balanced (S n) n
  RightBias : Balanced n (S n)


height : Balanced n m -> Nat
height (Perfect {n}) = S n
height (LeftBias {n}) = S $ S n
height (RightBias {n}) = S $ S n



data Tree : Nat -> Type -> Type -> Type where
  Empty : Tree 0 k v
  Node : (l : Tree n k v) -> 
         (key : k) ->
         (value : v) ->
         (r : Tree m k v) ->
         {auto b : Balanced n m} -> Tree (height b) k v
         
         


x : Tree 0 String String
x = Empty

y : Tree 1 String String
y = Node Empty "name" "jimmy" Empty


z : Tree 2 String String 
z = Node y "test" "test" x


