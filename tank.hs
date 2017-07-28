{-# LANGUAGE DeriveFunctor  #-}
module Main where

import Control.Monad.Trans.Class
import Control.Monad.Free
import Control.Monad.Extra

data Type = Increment | Decrement deriving (Show)

data Action a b = Action a b deriving (Show)


data Angle = Angle Int deriving (Show)
data Tank = Tank Vec deriving (Show)
data Vec = Vec Int deriving (Show)
data Entity = Entity Int deriving (Show)

instance Show (a -> b) where
  show f = "f"

data Move a
    = Accelerate a
    | RotateLeft (Maybe Angle) a
    | RotateRight (Maybe Angle) a
    | Delay a
    | Fire a
    | FindNearestTank (Tank -> a)
    | AngleTo Vec (Angle -> a)
    | IsAt Vec (Bool -> a)
    | IsFacing Angle (Bool -> a)
    | Me (Entity -> a) deriving (Functor, Show)

type Ai a = Free Move a

accelerate :: Ai ()
accelerate = liftF $ Accelerate ()


rotateLeft = RotateLeft Nothing ()
rotateRight = RotateRight Nothing ()
rotateLeftUpTo a = RotateLeft (Just a) ()
rotateRightUpTo a = RotateRight (Just a) ()
delay = Delay ()

fire :: Ai ()
fire = liftF $ Fire ()

findNearestTank :: Ai Tank
findNearestTank = liftF $ FindNearestTank id

angleTo :: Vec -> Ai Angle
angleTo pos = liftF $ AngleTo pos id

isAt :: Vec -> Ai Bool
isAt pos = liftF $ IsAt pos id


isFacing a = IsFacing a id
me = Me id



loop ai = ai >> loop ai
when b = b whenM
-- unless b = b unlessM

aimAtTank :: Tank -> Ai ()
aimAtTank (Tank pos) = do
  angle <- angleTo pos
  rotateTowards angle
  return ()

rotateTowards angle = return ()


position :: Tank -> Vec
position (Tank v) = v

moveTo :: Vec -> Ai ()
moveTo pos = do
  arrived <- isAt pos
  unless arrived $ do
    angle <- angleTo pos
    rotateTowards angle
    accelerate
    moveTo pos



searchAndDestroy :: Ai ()
searchAndDestroy = do
  tank <- findNearestTank
  angle <- angleTo (position tank)
  fire



main :: IO ()
main = putStr "hello"
