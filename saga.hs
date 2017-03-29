{-# LANGUAGE DeriveFunctor  #-}
module Main where

import Control.Monad.Trans.Class
import Control.Monad.Trans.Free

data Type = Increment | Decrement deriving (Show)

data Action a b = Action a b deriving (Show)


data Saga a b r = Put a r deriving (Functor)

increment :: Action Type ()
increment = Action Increment ()

decrement :: Action Type ()
decrement = Action Decrement ()

put :: (Monad m) => a -> FreeT (Saga a b) m ()
put action = liftF $ Put action ()


call :: (Monad m) => m b -> FreeT (Saga a b) m b
call = lift


runTIO :: (Show a) => FreeT (Saga a b) IO r -> IO r
runTIO s = do
  x <- runFreeT s
  case x of
    Pure r -> return r
    Free (Put a r) -> do
      print a
      runTIO r

doIt :: FreeT (Saga (Action Type ()) String) IO String
doIt = do
  put increment
  put decrement
  a <- call getLine
  b <- call getLine
  return $ a ++ b

main :: IO ()
main = do
  str <- runTIO doIt
  putStrLn str
