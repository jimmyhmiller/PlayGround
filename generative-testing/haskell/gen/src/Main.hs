module Main where

import Test.QuickCheck

evenOrOdd :: Int -> Bool
evenOrOdd x = even x || odd x

answer :: [Int] -> [Int]
answer [] = []
answer (42 : xs) = [42]
answer (x : xs) = x : answer xs

allEven :: [Int] -> Bool
allEven xs = all even xs

allSame :: [Int] -> Bool
allSame xs = answer xs == xs

main :: IO ()
main = do
    quickCheck evenOrOdd
