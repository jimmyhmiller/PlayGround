{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators  #-}
module Lib
    ( someFunc
    ) where

import Control.Monad.Free (Free, liftF, foldFree, hoistFree)
import Data.Map.Strict as Map (Map, lookup, (!), fromList, insert, elems)
import Control.Monad.State.Lazy as State (MonadState, gets, modify, runState, forM, liftM, runStateT)
import Prelude hiding (id, log)
import Data.List (minimumBy)
import Data.Ord (comparing)
import Data.Function (on)
import Control.Monad.IO.Class (MonadIO, liftIO)

data Entity = Entity {
  id :: Int,
  hp :: Int,
  strength :: Int,
  location :: (Float, Float)
} deriving (Show, Eq)

data MoveF next
  = GetEnemies ([Entity] -> next)
  | Me (Entity -> next)
  | Attack Int Entity next deriving Functor

type Move = Free MoveF

getEnemies :: Move [Entity]
getEnemies = liftF $ GetEnemies (\x -> x)

me :: Move Entity
me = liftF $ Me (\x -> x)

attack :: Int -> Entity -> Move ()
attack strength e = liftF $ Attack strength e ()

type TestEntities = Map Int Entity

testInterpreter :: MonadState TestEntities m => Int -> MoveF next -> m next
testInterpreter me (GetEnemies f) = do
  Just m <- gets (Map.lookup me)
  e <- gets (filter (/= m) . Map.elems)
  return $ f e
testInterpreter me (Me f) = do
   Just e <- gets (Map.lookup me)
   return $ f e
testInterpreter _ (Attack strength enemy next) = do
  modify $ Map.insert (id enemy) (enemy { hp = hp enemy - strength})
  return next

data LogF a = Log String a deriving Functor

type Log = Free LogF

log :: String -> Log ()
log s = liftF $ Log s ()

data (f :+: g) a = L (f a) | R (g a) deriving Functor

decorateLog :: MonadIO m => (MoveF next -> m next) -> MoveF next -> m next
decorateLog interpret move@(GetEnemies f) = do
  liftIO $ putStrLn "GetEnemies"
  interpret move
decorateLog interpret move@(Me f) = do
   liftIO $ putStrLn "Me"
   interpret move
decorateLog interpret move@(Attack strength enemy next) = do
  liftIO $ putStrLn ("Attack " ++ show strength ++ " " ++ show enemy)
  interpret move

distance :: Entity -> Entity -> Float
distance Entity { location = (x1 , y1)} Entity {location= (x2 , y2)} = sqrt (x'*x' + y'*y')
    where
      x' = x1 - x2
      y' = y1 - y2

allOthers :: (Eq a) => (a -> a -> b) -> a -> [a] -> [(a, b)]
allOthers f x xs = map (\x' -> (x', f x x')) xs

attackAll :: Move [()]
attackAll = do
  m <- me
  enemies <- getEnemies
  mapM (attack (strength m)) enemies

attackClosest :: Move [()]
attackClosest = do
  m <- me
  enemies <- getEnemies
  let otherDistances = allOthers distance m enemies
  let (closest, _) = minimumBy (compare `on` snd) otherDistances
  pure <$> attack (strength m) closest

attackRadius :: Float -> Move [()]
attackRadius radius = do
  m <- me
  enemies <- getEnemies
  let otherDistances = allOthers distance m enemies
  let inRadius = filter ((<= radius) . snd) otherDistances
  mapM (attack (strength m) . fst) inRadius

scenario1 :: TestEntities
scenario1 = Map.fromList [(2, Entity { id = 2, hp = 2, strength = 50, location = (0, 0)}),
                          (3, Entity { id = 3, hp = 2, strength = 50, location = (0, 2)}),
                          (4, Entity { id = 4, hp = 2, strength = 50, location = (0, 10)}),
                          (5, Entity { id = 5, hp = 2, strength = 50, location = (0, 20)})]

interpreter :: (MonadState TestEntities m) => Move a -> m a
interpreter = foldFree $ testInterpreter 2

x = runState (interpreter attackAll) scenario1

someFunc :: IO ()
someFunc = putStrLn "someFunc!"
