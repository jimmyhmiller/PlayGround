{-# LANGUAGE DeriveFunctor  #-}
module Lib
    ( someFunc
    ) where

data Thing a = Stuff a | OtherStuff deriving Functor

someFunc :: IO ()
someFunc = putStrLn "someFunc!"

thing :: Int
thing = "asdfsad"
