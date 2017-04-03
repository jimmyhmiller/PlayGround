module Main where
import Prelude hiding (lookup)

data Tree a = Empty | Node (Tree a) a (Tree a) deriving Show
type Map a b = Tree (a,b)

data Heap a = HEmpty | HNode Int a (Heap a) (Heap a) deriving Show



merge :: Ord a => Heap a -> Heap a -> Heap a
merge HEmpty h = h
merge h HEmpty = h
merge h1@(HNode _ x l1 r1) h2@(HNode _ y l2 r2) = case compare x y of
  LT -> swapChildren x l1 (merge r1 h2)
  EQ -> swapChildren x l1 (merge r1 h2)
  GT -> swapChildren y l2 (merge h1 r2)


rank :: Heap a -> Int
rank HEmpty = 0
rank (HNode r _ _ _) = r

swapChildren :: a -> Heap a -> Heap a -> Heap a
swapChildren x a b
  | rank a >= rank b = HNode (rank b + 1) x a b
  | otherwise = HNode (rank a + 1) x b a

insertH :: Ord a => a -> Heap a -> Heap a
insertH x = merge (HNode 1 x HEmpty HEmpty)

findMin :: Heap a -> Maybe a
findMin HEmpty = Nothing
findMin (HNode _ x _ _) = Just x

deleteMin :: Ord a => Heap a -> Heap a
deleteMin HEmpty = HEmpty
deleteMin (HNode _ _ a b) = merge a b


insert :: Ord a => a -> Tree a -> Tree a
insert x Empty = Node Empty x Empty
insert x (Node l v r) = case compare x v of
  LT -> Node (insert x l) v r
  EQ -> Node l x r
  GT -> Node l x (insert x r)


find :: Ord a => a -> Tree a -> Maybe a
find  _ Empty = Nothing
find x (Node l v r) = case compare x v of
  LT -> find x l
  EQ -> Just v
  GT -> find x r


lookup :: Ord a => a -> Map a b -> Maybe b
lookup _ Empty = Nothing
lookup x (Node l (k, v) r) = case compare x k of
  LT -> lookup x l
  EQ -> Just v
  GT -> lookup x r


q :: Maybe Integer
q = lookup "test" $ insert ("test", 1) Empty


heap1 :: Heap Int
heap1 = insertH 2 $ insertH 3 $ insertH 1  HEmpty


main :: IO ()
main = print heap1
