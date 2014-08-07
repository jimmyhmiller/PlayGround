import Control.Applicative
import Data.Monoid
import qualified Data.Foldable

data Deque a = Deque [a] [a] deriving (Show, Eq)

--Functionally, I'm fairly sure my monad instances hold to the three laws.
--You may end up with a different balance between the two lists, but it should still work the same.
--Now, the one problem I haven't address is keeping the time complexity. I think it is way worse now.

main = putStrLn $ show $ Deque [0] [1]

splitHalf l = splitAt ((length l + 1) `div` 2) l

instance Functor Deque where
    fmap f (Deque a b) = Deque (fmap f (a ++ (reverse b))) []

instance Applicative Deque where
    pure x = Deque [x] []
    (Deque fas fbs) <*> (Deque xs ys) = Deque ((fas ++ reverse fbs) <*> (xs ++ reverse ys)) []

instance Monoid (Deque a) where
    mempty = Deque [] []
    mappend (Deque xs ys) (Deque xs' ys') = Deque (xs ++ (reverse ys) ++ xs' ++ (reverse ys')) []

instance Data.Foldable.Foldable Deque where
    foldr f z (Deque [] []) = z
    foldr f z (Deque (x:xs) ys) = f x (foldr f z (xs ++ (reverse ys)))

instance Monad Deque where
    return x = Deque [x] []
    (Deque a b) >>= f = (foldr mappend (Deque [] [])) (fmap f (a ++ (reverse b)))

isEmpty :: Deque a -> Bool
isEmpty (Deque [] []) = True
isEmpty _ = False

fromList :: [a] -> Deque a
fromList [] = Deque [] []
fromList (x:xs) = cons' x (fromList xs)

cons' :: a -> Deque a -> Deque a
cons' a (Deque f r) = checkf' $ Deque (a:f) r

head' :: Deque a -> a
head' (Deque [] []) = error "Error Empty"
head' (Deque (x:_) _) = x
head' (Deque [] r) = head' $ checkf' $ Deque [] r

tail' :: Deque a -> Deque a
tail' (Deque [] []) = error "Error Empty"
tail' (Deque (_:xs) r) = checkf' $ Deque xs r
tail' (Deque [] r) = tail' $ checkf' $ Deque [] r

snoc' :: a -> Deque a -> Deque a
snoc' a (Deque f r) = Deque f (a:r)

last' :: Deque a -> a
last' (Deque [] []) = error "Error Empty"
last' (Deque _ (x:_)) = x
last' (Deque f []) = last' $ checkf' $ Deque f []

init' :: Deque a -> Deque a
init' (Deque [] []) = error "Error Empty"
init' (Deque f (_:xs)) = Deque f xs
init' (Deque f []) = init' $ checkf' $ Deque f []

checkf' :: Deque a -> Deque a
checkf' (Deque [x] []) = Deque [] [x]
checkf' (Deque [] [x]) = Deque [x] []
checkf' (Deque [] r) = Deque (reverse fh) lh
    where (lh, fh) = splitHalf r
checkf' (Deque f []) = Deque lh (reverse fh)
    where (lh, fh) = splitHalf f
checkf' d = d