{-# LANGUAGE DeriveFunctor  #-}
module Main where

import Control.Monad.Trans.Class
import Control.Monad.Free
import Control.Monad.Extra

data Angle = Angle Int deriving (Show)
data Tank = Tank Vec deriving (Show)
data Vec = Vec Int Int deriving (Show)
data Entity = Entity Int deriving (Show)

data Move next
    = Accelerate next
    | RotateLeft (Maybe Angle) next
    | RotateRight (Maybe Angle) next
    | Fire next
    | AngleTo Vec (Angle -> next)
    | IsAt Vec (Bool -> next) deriving (Functor)

interpreter :: Move a -> IO a
interpreter (Accelerate next) = do 
  putStr "(Accelerate) -> "
  return next
interpreter (RotateLeft angle next) = do 
  putStrLn $ "(RotateLeft " ++ show angle ++ ") -> "
  return next
interpreter (RotateRight angle next) = do 
  putStrLn $ "(RotateRight (" ++ show angle ++ ")) -> "
  return next
interpreter (Fire next) = do 
  putStrLn $ "(Fire) -> "
  return next
interpreter (AngleTo pos f) = do
  putStr "Angle: "
  angle <- getLine
  putStrLn $ "(AngleTo" ++ (show pos) ++ ") -> "
  return (f (Angle (read angle :: Int)))
interpreter (IsAt pos f) = do
  putStr "Is at: "
  input <- getLine
  putStrLn $ "(IsAt " ++ (show pos) ++ ") -> " 
  return (f (input == "True"))

type Ai a = Free Move a

accelerate :: Ai ()
accelerate = liftF $ Accelerate ()

fire :: Ai ()
fire = liftF $ Fire ()

angleTo :: Vec -> Ai Angle
angleTo pos = liftF $ AngleTo pos id

isAt :: Vec -> Ai Bool
isAt pos = liftF $ IsAt pos id

rotateTowards :: Angle -> Ai ()
rotateTowards (Angle x) = if even x then 
    liftF $ RotateRight (Just (Angle x)) ()
  else 
    liftF $ RotateLeft (Just (Angle x)) ()

moveTo :: Vec -> Ai ()
moveTo pos = do
  arrived <- isAt pos
  unless arrived $ do
    angle <- angleTo pos
    rotateTowards angle
    accelerate
    moveTo pos

main :: IO ()
main = do 
  -- putStr "Hello World"
  foldFree interpreter (moveTo (Vec 0 0))
  -- interpret (moveTo (Vec 1))
