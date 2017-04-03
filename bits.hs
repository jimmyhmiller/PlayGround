module Main where

import Prelude hiding (or, and, (||), (>>), not)

data Bit = Zero | One deriving (Show, Eq)

fromBool :: Bool -> Bit
fromBool True = One
fromBool _ = Zero


toString :: [Bit] -> String
toString [] = []
toString (Zero : xs) = '0' : toString xs
toString (One : xs) = '1' : toString xs

b :: String -> [Bit]
b [] = []
b ('0' : xs) = Zero : b xs
b ('1' : xs) = One : b xs
b _ = error "must be 0 or 1"

(||) :: [Bit] -> [Bit] -> [Bit]
[] || [] = []
(Zero : xs) || (One : ys) = One : xs || ys
(One : xs) || (Zero : ys) = One : xs || ys
(One : xs) || (One : ys) = One : xs || ys
(Zero : xs) || (Zero : ys) = Zero : xs || ys
_ || _ = error "must be same size"


(&) :: [Bit] -> [Bit] -> [Bit]
[] & [] = []
(One : xs) & (One : ys) = One : xs & ys
(_ : xs) & (_ : ys) = Zero : xs & ys
_ & _= error "must be same size"

not :: [Bit] -> [Bit]
not [] = []
not (One : xs) = Zero : not xs
not (Zero : xs) = One : not xs

(<<) :: [Bit] -> Int -> [Bit]
bs << n = drop n $ bs ++ replicate n Zero

(>>) :: [Bit] -> Int -> [Bit]
bs >> n = take (length bs) $ replicate n Zero ++ bs

(...) :: Bit -> Int -> [Bit]
bit ... n = replicate n bit

pad :: Int -> [Bit] -> [Bit]
pad n bits = replicate (n - length bits) Zero ++ bits

get :: Int -> [Bit] -> Bit
get i num = fromBool $ elem One $ num & (pad (length num) [One] << i)

set :: Int -> [Bit] -> [Bit]
set i num = num || pad (length num) [One] << i

clear :: Int -> [Bit] -> [Bit]
clear i num = num & not (pad (length num) [One] << i)

update :: Int -> [Bit] -> Bit -> [Bit]
update i num v = (num & mask) || (pad (length num) [v] << i) where
  mask = not (pad (length num) [One] << i )

insert :: [Bit] -> [Bit] -> Int -> Int -> [Bit]
insert m [] _ _ = m
insert m n j k = insert (update j m (get 0 n)) (take (length n - 1) n) (j + 1) k




main :: IO ()
-- main = print $ update 1 (b "1111") Zero
main = putStrLn $ toString $ insert (b "10000000000") (b "10011") 2 6
