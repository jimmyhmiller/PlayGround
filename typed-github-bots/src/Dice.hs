{-# LANGUAGE FlexibleContexts #-}
module Dice
    ( parseDice
    , rollDice
    , roll
    , Dice(Dice)
    ) where

import Text.ParserCombinators.Parsec
import System.Random
import Control.Monad.Random

data Dice = Dice Int Int Int deriving (Show, Eq)

mkDice :: Int -> Int -> Int -> Dice
mkDice = Dice

toInt :: String -> Int
toInt s = read s :: Int

diceParser = do
  amount <- many1 digit
  char 'd'
  faces <- many1 digit
  modifier <- option "0" $ do
    char '+'
    many1 digit
  return $ Dice (toInt amount) (toInt faces) (toInt modifier)

roll :: (MonadRandom m) => String -> Either ParseError (m Int)
roll s = rollDice <$> parseDice s

rollDice :: (MonadRandom m) => Dice -> m Int
rollDice (Dice amount faces modifier) = do
  rolls <- getRandomRs (1,faces)
  return $ sum (take amount rolls) + modifier


parseDice :: String -> Either ParseError Dice
parseDice = parse diceParser ""
