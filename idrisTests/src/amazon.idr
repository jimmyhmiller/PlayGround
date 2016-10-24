data Tree a = Empty | Node (Tree a) a (Tree a)


insert : Ord a => a -> Tree a -> Tree a
insert x Empty = Node Empty x Empty
insert n node@(Node l v r) = case compare n v of
                             LT => Node (insert n l) v r
                             EQ => node
                             GT => Node l v (insert n r) 



