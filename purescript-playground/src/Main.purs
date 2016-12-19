module Main where

import Prelude
import Control.Monad.Eff (Eff)
import Control.Monad.Eff.Console (CONSOLE, log)
import Data.Generic (class Generic, gEq, gShow)



double :: Int -> Int
double x = x * 2


data Color = Red | Green

derive instance genericColor :: Generic Color

instance showColor :: Show Color where
  show = gShow

instance eqColor :: Eq Color where
  eq = gEq




isFavorite :: Color -> Boolean
isFavorite Red = ?isFavorite
isFavorite Green = ?isFavorite


main :: forall e. Eff (console :: CONSOLE | e) Unit
main = do
  log "Hello sailor!"
