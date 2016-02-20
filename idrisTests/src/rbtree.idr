data Color = Red | Black


data RbTree : Type -> Type where
  Empty : RbTree a 
  Node : Color -> RbTree a -> a -> RbTree a -> RbTree a

member : Ord a => a -> RbTree a -> Bool
member x Empty = False
member x (Node c l y r) = case compare x y of
                               LT => member x l
                               EQ => True
                               GT => member x r


balance : RbTree a -> RbTree a
balance (Node Black (Node Red (Node Red a x b) y c) z d) = Node Red (Node Black a x b) y (Node Black c z d)
balance (Node Black (Node Red a x (Node Red b y c)) z d) = Node Red (Node Black a x b) y (Node Black c z d)
balance (Node Black a x (Node Red (Node Red b y c) z d)) =  Node Red (Node Black a x b) y (Node Black c z d)
balance (Node Black a x (Node Red b y (Node Red c z d))) =  Node Red (Node Black a x b) y (Node Black c z d)
balance n = n


paintBlack : RbTree a -> RbTree a
paintBlack (Node _ y z w) = Node Black y z w

insert : Ord a => a -> RbTree a -> RbTree a
insert x n = paintBlack (ins n) where
  ins : RbTree a -> RbTree a
  ins Empty = Node Red Empty x Empty
  ins (Node c l y r) = case compare x y of
                            LT => balance (Node c (ins l) y r)
                            EQ => n
                            GT => balance (Node c l y (ins r))


fromList : Ord a => List a -> RbTree a
fromList [] = Empty
fromList (x :: xs) = insert x (fromList xs)


x : RbTree Int
x = fromList [1..100]
