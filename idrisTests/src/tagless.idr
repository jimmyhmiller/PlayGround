

data BinaryTree : Type -> Type where
  Empty : BinaryTree a
  Node : a -> BinaryTree a -> BinaryTree a -> BinaryTree a
  


insert : Ord a => a -> BinaryTree a -> BinaryTree a
insert x Empty = Node x Empty Empty
insert x (Node v l r) = case compare x v of
                             LT => Node v (insert x l) r
                             EQ => Node v l r
                             GT => Node v l (insert x r)
                             
                             
combine : Ord a => BinaryTree a -> BinaryTree a -> BinaryTree a              
combine Empty y = y
combine x Empty = x
combine (Node x z w) y = combine y (combine z (insert x y))

contains : Ord a => a -> BinaryTree a -> Bool        
contains x Empty = False
contains x (Node v l r) = case compare x v of
                               LT => contains x l
                               EQ => True
                               GT => contains x r



rightMost : (t : BinaryTree a) -> a
rightMost (Node v l Empty) = v
rightMost (Node v l r@(Node _ _ _)) = rightMost r


delete : Ord a => a -> BinaryTree a -> BinaryTree a     
delete x Empty = Empty
delete x (Node v Empty Empty) = case x == v of
                                     False => Node v Empty Empty
                                     True => Empty
delete x (Node v Empty r) = case x == v of
                                 False => Node v Empty (delete x r)
                                 True => r
delete x (Node v l Empty) = case x == v of
                                 False => Node v (delete x l) Empty
                                 True => l
delete x (Node v l r) = case compare x v of
                             LT => Node v (delete x l) r                  
                             EQ => let nextLeast = rightMost l in
                                   let allButNextLeast = delete nextLeast l in
                                   (Node nextLeast allButNextLeast r)
                             GT => Node v l (delete x r)
  





fromList : Ord a => List a -> BinaryTree a
fromList [] = Empty
fromList (x :: xs) = insert x (fromList xs)




instance Functor BinaryTree where  
  map f Empty = Empty
  map f (Node x l r) = Node (f x) (map f l) (map f r)
  

instance Applicative BinaryTree where
    pure x = Node x Empty Empty
    (<*>) Empty y = Empty
    (<*>) x Empty = Empty
    (<*>) (Node f fl fr) (Node x l r) = Node (f x) (fl <*> l) (fr <*> r)
